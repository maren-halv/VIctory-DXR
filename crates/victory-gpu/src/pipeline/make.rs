//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// Inline storage for 0..=2 shaders (covers 99.9% of pipelines: CS, VS+PS,
/// MS+PS) with zero heap allocation, spilling to a boxed slice only for 3+.
pub(crate) enum ShaderList {
    Empty,
    One([Arc<Shader>; 1]),
    Two([Arc<Shader>; 2]),
    Many(Box<[Arc<Shader>]>),
}

impl ShaderList {
    fn from_vec(mut v: Vec<Arc<Shader>>) -> Self {
        match v.len() {
            0 => Self::Empty,
            1 => Self::One([v.pop().unwrap()]),
            2 => {
                let b = v.pop().unwrap();
                let a = v.pop().unwrap();
                Self::Two([a, b])
            }
            _ => Self::Many(v.into_boxed_slice()),
        }
    }

    pub(crate) fn as_slice(&self) -> &[Arc<Shader>] {
        match self {
            Self::Empty => &[],
            Self::One(s) => s,
            Self::Two(s) => s,
            Self::Many(s) => s,
        }
    }
}

impl std::ops::Deref for ShaderList {
    type Target = [Arc<Shader>];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::fmt::Debug for ShaderList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_slice(), f)
    }
}

/// Lifecycle state of [`Pending::active_ptr`].
const STATE_UNBUILT: u8 = 0;
const STATE_READY: u8 = 1;
const STATE_FAILED: u8 = 2;
const STATE_EVICTED: u8 = 3;

/// Wrapper for deferring `Box<HostPipeline>` drop onto `host::retire` +
/// `victory-bury` so in-flight command lists and recording threads never see a
/// freed COM PSO.
struct RetiredPipeline(*mut HostPipeline);

// SAFETY: `HostPipeline` is Send and ownership is exclusively transferred here
// after unlinking `active_ptr` and observing the CPU grace period.
unsafe impl Send for RetiredPipeline {}

impl Drop for RetiredPipeline {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: originally allocated via `Box::into_raw` and uniquely owned here.
            unsafe { drop(Box::from_raw(self.0)) };
        }
    }
}

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: ShaderList,
    pub(crate) graphics: Option<Arc<GraphicsState>>,
    /// Active materialized `HostPipeline` (`Box::into_raw`), or null when
    /// unbuilt, failed, or dormant-evicted.
    active_ptr: AtomicPtr<HostPipeline>,
    made_state: AtomicU8,
    /// Last frame and coarse millisecond timestamp on which this PSO was bound.
    last_used_frame: AtomicU64,
    last_used_ms: AtomicU64,
    /// Pinned PSOs (`builder`, `screen`, `scene_input`, `is_bake_shader`) are
    /// NEVER evicted by the dormant LRU sweeper.
    pinned: AtomicBool,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    /// Number of threads currently waiting in [`wait`] for `made`.
    waiters: AtomicU32,
    /// Whether this pipeline has already been retried single-threaded after a
    /// parallel driver compiler crash.
    retried: AtomicBool,
    /// What identifies it across runs (shader containers and state), stored
    /// inline as 16 bytes (zero heap allocation).
    key: [u8; 16],
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// Why this pipeline is intentionally skipped (e.g. RT resolve without BVH).
    left_out: Option<&'static str>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        let ptr = self.active_ptr.swap(std::ptr::null_mut(), Ordering::AcqRel);
        if !ptr.is_null() {
            crate::host::retire(RetiredPipeline(ptr));
        }
    }
}

fn left_out(shaders: &[Arc<Shader>]) -> Option<&'static str> {
    if !shaders.iter().any(|s| s.stage == "compute") {
        return None;
    }
    let rays_active = || {
        crate::dxr::enabled()
            || std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1")
    };
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    let skip_rt = *SKIP_RT_RESOLVE.get_or_init(|| {
        if rays_active() {
            return false;
        }
        // Without BVH or native DXR, ~1,000 CS_RaytracingResolve shaders shade no
        // hits while clogging the cold compiler queue; skip unless VICTORY_RT_RESOLVE=1.
        std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v == "0")
    });
    if skip_rt && shaders.iter().any(|s| s.has_flag(super::FLAG_RT_RESOLVE)) {
        return Some("shades ray-tracing hits without BVH, left out");
    }
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate = *SKIP_ACTIVATE_RAYS.get_or_init(|| {
        if rays_active() {
            return std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0");
        }
        std::env::var_os("VICTORY_ACTIVATE_RAYS").is_none_or(|v| v == "0")
    });
    if skip_activate && shaders.iter().any(|s| s.has_flag(super::FLAG_ACTIVATE_RAYS)) {
        return Some("ray activation shader without BVH, left out");
    }
    // Pure screen-space local lighting (0/32 contain any BVH/RT instructions);
    // keep active by default in both RT-off and RT-on modes, with an explicit
    // opt-out (`VICTORY_SKIP_LOCAL_LIGHTS=1`) for A/B timing.
    static SKIP_LOCAL_LIGHTS: OnceLock<bool> = OnceLock::new();
    let skip_local_lights = *SKIP_LOCAL_LIGHTS.get_or_init(|| {
        std::env::var_os("VICTORY_SKIP_LOCAL_LIGHTS").is_some_and(|v| v == "1")
            || std::env::var_os("VICTORY_RAYMARCHED_LOCAL_LIGHTS").is_some_and(|v| v == "0")
    });
    if skip_local_lights && shaders.iter().any(|s| s.has_flag(super::FLAG_LOCAL_LIGHTS)) {
        return Some("screen-space local lights compute pass skipped (VICTORY_SKIP_LOCAL_LIGHTS=1)");
    }
    None
}

/// Confirmed first-scene producers. Prepare these at creation even on a cold
/// launch so cutscene lighting, tiling, and bakes never miss their first frame.
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| {
        matches!(
            s.hash.as_str(),
            "970c4fbb3b017c0e"
                | "068c90badc86f1fd"
                | "e4a7e088c7780ce8"
                | "ebe4e53f031f5249"
                | "2cc4c197ee29e291"
                | "369cbe906ea72764"
        ) || crate::draw::is_bake_shader_hash(s.hash.as_str())
    })
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

struct Blake3Writer<'a>(&'a mut blake3::Hasher);

impl std::fmt::Write for Blake3Writer<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.update(s.as_bytes());
        Ok(())
    }
}

impl Pending {
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Arc<GraphicsState>>,
        builder: bool,
    ) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        if let Some(g) = graphics.as_deref() {
            let _ = g.describe_into(&mut Blake3Writer(&mut key));
        }
        let scene_input = scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders);
        let pinned = builder || screen || scene_input;
        let digest = key.finalize();
        let mut key_bytes = [0u8; 16];
        key_bytes.copy_from_slice(&digest.as_bytes()[..16]);
        Pending {
            screen,
            scene_input,
            left_out,
            shaders: ShaderList::from_vec(shaders),
            graphics,
            active_ptr: AtomicPtr::new(std::ptr::null_mut()),
            made_state: AtomicU8::new(STATE_UNBUILT),
            last_used_frame: AtomicU64::new(0),
            last_used_ms: AtomicU64::new(0),
            pinned: AtomicBool::new(pinned),
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            waiters: AtomicU32::new(0),
            retried: AtomicBool::new(false),
            key: key_bytes,
            used: AtomicBool::new(false),
        }
    }

    pub(super) fn shaders(&self) -> &[Arc<Shader>] {
        self.shaders.as_slice()
    }

    pub(super) fn key(&self) -> [u8; 16] {
        self.key
    }

    fn is_settled(&self) -> bool {
        matches!(
            self.made_state.load(Ordering::Acquire),
            STATE_READY | STATE_FAILED
        )
    }

    fn is_pinned(&self) -> bool {
        self.pinned.load(Ordering::Relaxed)
    }

    fn note_bound(&self, frame: u64, now_ms: u64) {
        if self.last_used_frame.load(Ordering::Relaxed) != frame {
            self.last_used_frame.store(frame, Ordering::Relaxed);
        }
        if self.last_used_ms.load(Ordering::Relaxed) != now_ms {
            self.last_used_ms.store(now_ms, Ordering::Relaxed);
        }
    }

    fn make(self: &Arc<Pending>) {
        // Not through `OnceLock::get_or_init`: a worker the driver's compiler
        // crashed under stays inside the call for good (see [`guard_start`]),
        // and anyone else asking would wait for it forever.
        let state = self.made_state.load(Ordering::Acquire);
        if state == STATE_UNBUILT || state == STATE_EVICTED {
            let made = if is_bad_key(&self.key) {
                SKIPPED_BAD.fetch_add(1, Ordering::Relaxed);
                let key_buf = super::key_hex(&self.key);
                tracing::warn!(
                    key = super::key_hex_str(&key_buf),
                    shaders = ?self.shaders,
                    "not made: the driver's compiler crashed on it before"
                );
                None
            } else {
                let began = Instant::now();
                let made = HostPipeline::new(self.shaders.as_slice(), self.graphics.as_deref());
                super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
                made
            };
            match made {
                Some(hp) => {
                    let is_bake = hp.is_bake_shader;
                    if is_bake {
                        self.pinned.store(true, Ordering::Relaxed);
                    }
                    let raw = Box::into_raw(Box::new(hp));
                    self.note_bound(crate::frame::current(), crate::host::coarse_ms());
                    if self
                        .active_ptr
                        .compare_exchange(
                            std::ptr::null_mut(),
                            raw,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        self.made_state.store(STATE_READY, Ordering::Release);
                        if !self.is_pinned() {
                            register_live_pso(self);
                        }
                    } else {
                        crate::host::retire(RetiredPipeline(raw));
                    }
                }
                None => {
                    self.made_state.store(STATE_FAILED, Ordering::Release);
                }
            }
        }
        notify_done(self);
    }
}

/// Non-pinned materialized PSOs tracked for 60 s dormant COM eviction.
static LIVE_PSOS: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());
/// Evicted PSO pointers held for a 2 s CPU grace period before handing to
/// `host::retire()` (which then waits for all GPU flights + `bury_lag()` and
/// drops COM objects asynchronously on `victory-bury`).
static EVICT_QUARANTINE: Mutex<VecDeque<(u64, RetiredPipeline)>> = Mutex::new(VecDeque::new());
static EVICTED_COUNT: AtomicU64 = AtomicU64::new(0);

fn register_live_pso(p: &Arc<Pending>) {
    LIVE_PSOS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(p));
}

/// Dormant idle threshold before non-pinned `ID3D12PipelineState` COM objects
/// are evicted (`VICTORY_PSO_IDLE_MS`, default `60_000` ms; `0` disables).
fn dormant_idle_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_PSO_IDLE_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(60_000)
    })
}

const DORMANT_MIN_FRAMES: u64 = 60;
const QUARANTINE_MS: u64 = 2_000;

fn pso_dormant(
    pinned: bool,
    now_ms: u64,
    last_ms: u64,
    frame: u64,
    last_frame: u64,
    idle_ms: u64,
) -> bool {
    !pinned
        && idle_ms > 0
        && last_ms > 0
        && now_ms.saturating_sub(last_ms) >= idle_ms
        && frame.saturating_sub(last_frame) >= DORMANT_MIN_FRAMES
}

/// Evicts dormant non-pinned PSO COM objects idle for >= 60 s and flushes
/// quarantined COM objects into `host::retire` (`victory-bury`). Called once
/// per second from `pipeline::trim_library()` (`host::sweep()`).
pub(super) fn evict_dormant() {
    let now = crate::host::coarse_ms();
    // 1. Flush quarantine entries past the 2 s CPU borrow grace window into
    //    `host::retire()` (which enforces GPU fence + `bury_lag()` + async drop).
    {
        let mut q = EVICT_QUARANTINE.lock().unwrap_or_else(|e| e.into_inner());
        while let Some((ts, _)) = q.front() {
            if now.saturating_sub(*ts) < QUARANTINE_MS {
                break;
            }
            if let Some((_, retired)) = q.pop_front() {
                crate::host::retire(retired);
            }
        }
    }

    let idle_ms = dormant_idle_ms();
    if idle_ms == 0 {
        return;
    }
    static LAST_SCAN_MS: AtomicU64 = AtomicU64::new(0);
    let last_scan = LAST_SCAN_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last_scan) < 5_000
        || LAST_SCAN_MS
            .compare_exchange(last_scan, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }

    let frame = crate::frame::current();
    let mut to_evict = Vec::new();
    {
        let mut live = LIVE_PSOS.lock().unwrap_or_else(|e| e.into_inner());
        live.retain(|weak| {
            let Some(p) = weak.upgrade() else {
                return false;
            };
            if p.is_pinned() || p.active_ptr.load(Ordering::Acquire).is_null() {
                return false;
            }
            if p.waiters.load(Ordering::Acquire) == 0
                && pso_dormant(
                    false,
                    now,
                    p.last_used_ms.load(Ordering::Relaxed),
                    frame,
                    p.last_used_frame.load(Ordering::Relaxed),
                    idle_ms,
                )
            {
                to_evict.push(p);
                return false;
            }
            true
        });
    }
    if to_evict.is_empty() {
        return;
    }
    let mut newly_evicted = Vec::new();
    {
        // Coordinate resetting `queue_state` to `NOT_QUEUED` under `QUEUE` so
        // any concurrent `start(p, REQUESTED)` cleanly re-enqueues `p`.
        let _queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        for p in to_evict {
            if p.is_pinned()
                || p.waiters.load(Ordering::Acquire) > 0
                || !pso_dormant(
                    false,
                    now,
                    p.last_used_ms.load(Ordering::Relaxed),
                    frame,
                    p.last_used_frame.load(Ordering::Relaxed),
                    idle_ms,
                )
            {
                continue;
            }
            let ptr = p.active_ptr.swap(std::ptr::null_mut(), Ordering::AcqRel);
            if !ptr.is_null() {
                p.made_state.store(STATE_EVICTED, Ordering::Release);
                p.queue_state.store(NOT_QUEUED, Ordering::Release);
                newly_evicted.push(RetiredPipeline(ptr));
            }
        }
    }
    if !newly_evicted.is_empty() {
        let count = newly_evicted.len() as u64;
        EVICTED_COUNT.fetch_add(count, Ordering::Relaxed);
        let mut q = EVICT_QUARANTINE.lock().unwrap_or_else(|e| e.into_inner());
        for retired in newly_evicted {
            q.push_back((now, retired));
        }
        crate::host::trail(|| format!("pipeline LRU: evicted {count} dormant host PSOs"));
    }
}

/// Pipelines not made for being on the list of those the driver's compiler
/// crashes on.
static SKIPPED_BAD: AtomicU64 = AtomicU64::new(0);

/// The keys of the pipelines the driver's compiler crashed on
/// (`warm-bad.txt` beside the pipeline library, one key to a line, then
/// anything; `#` begins a comment), loaded on first use. `VICTORY_WARM_BAD=0`
/// leaves the file unread.
static BAD_KEYS: OnceLock<RwLock<std::collections::HashSet<[u8; 16]>>> = OnceLock::new();

fn bad_keys_lock() -> &'static RwLock<std::collections::HashSet<[u8; 16]>> {
    BAD_KEYS.get_or_init(|| {
        if std::env::var_os("VICTORY_WARM_BAD").is_some_and(|v| v == "0") {
            return RwLock::new(Default::default());
        }
        let listed = super::library::folder()
            .and_then(|d| std::fs::read_to_string(d.join("warm-bad.txt")).ok())
            .unwrap_or_default();
        RwLock::new(
            listed
                .lines()
                .filter(|l| !l.trim_start().starts_with('#'))
                .filter_map(|l| l.split_whitespace().next())
                .filter_map(super::parse_key_hex)
                .collect(),
        )
    })
}

fn bad_keys<R>(f: impl FnOnce(&mut std::collections::HashSet<[u8; 16]>) -> R) -> R {
    let mut keys = bad_keys_lock().write().unwrap_or_else(|e| e.into_inner());
    f(&mut keys)
}

fn is_bad_key(key: &[u8; 16]) -> bool {
    bad_keys_lock()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(key)
}

/// Whether the driver's compiler crashes are caught
/// (`VICTORY_COMPILER_GUARD=0`: not).
fn guard_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMPILER_GUARD").is_none_or(|v| v != "0"))
}

/// Worker threads that can be told apart by the guard, over the run.
const MAX_SLOTS: usize = 128;
/// Per worker slot: its compile crashed (set in the handler), where.
static CRASHED: [AtomicBool; MAX_SLOTS] = [const { AtomicBool::new(false) }; MAX_SLOTS];
static FAULT_ADDR: [AtomicU64; MAX_SLOTS] = [const { AtomicU64::new(0) }; MAX_SLOTS];
static FAULT_RIP: [AtomicU64; MAX_SLOTS] = [const { AtomicU64::new(0) }; MAX_SLOTS];
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
/// The event the handler signals (a Win32 handle).
static CRASH_EVENT: AtomicUsize = AtomicUsize::new(0);
/// What each slot's worker is making, whether it takes passes over the screen
/// alone, and whether this compile is a single-threaded retry after a parallel
/// crash. Never locked by the handler.
static ACTIVE_JOBS: Mutex<Vec<Option<(Arc<Pending>, bool, bool)>>> = Mutex::new(Vec::new());

/// Crash-safe reader/writer gate coordinating parallel compiles (`enter_shared`)
/// against single-threaded crash retries (`enter_exclusive`). Unlike an OS
/// `RwLock`, a worker that crashes and sleeps forever inside `compiler_crashed`
/// has its count released explicitly by [`lost`] so retries never deadlock.
struct CompileGate {
    state: Mutex<(usize, bool)>,
    cv: Condvar,
}

impl CompileGate {
    const fn new() -> Self {
        Self {
            state: Mutex::new((0, false)),
            cv: Condvar::new(),
        }
    }

    fn enter_shared(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.1 {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.0 += 1;
    }

    fn exit_shared(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.0 = s.0.saturating_sub(1);
        if s.0 == 0 {
            self.cv.notify_all();
        }
    }

    fn enter_exclusive(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.1 || s.0 > 0 {
            s = self.cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.1 = true;
    }

    fn exit_exclusive(&self) {
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        s.1 = false;
        self.cv.notify_all();
    }
}

static COMPILE_GATE: CompileGate = CompileGate::new();

thread_local! {
    /// This worker's slot (`usize::MAX`: none).
    static WORKER_SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    /// Whether this thread is inside the compile of a pipeline.
    static COMPILING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A worker's compile of `p`: noted for the guard while it lasts.
struct Compile {
    slot: usize,
    is_retry: bool,
    gated: bool,
}

impl Compile {
    fn begin(slot: usize, p: &Arc<Pending>, screens_only: bool, is_retry: bool) -> Compile {
        if is_retry {
            COMPILE_GATE.enter_exclusive();
        } else {
            COMPILE_GATE.enter_shared();
        }
        if slot < MAX_SLOTS && guard_on() {
            let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
            if jobs.len() < MAX_SLOTS {
                jobs.resize(MAX_SLOTS, None);
            }
            jobs[slot] = Some((p.clone(), screens_only, is_retry));
            COMPILING.set(true);
        }
        Compile {
            slot,
            is_retry,
            gated: true,
        }
    }
}

impl Drop for Compile {
    fn drop(&mut self) {
        if COMPILING.replace(false) && self.slot < MAX_SLOTS {
            let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(job) = jobs.get_mut(self.slot) {
                *job = None;
            }
        }
        if self.gated {
            self.gated = false;
            if self.is_retry {
                COMPILE_GATE.exit_exclusive();
            } else {
                COMPILE_GATE.exit_shared();
            }
        }
    }
}

/// Installs the handler for the driver's compiler crashing (AMD's does on an
/// RX 9070 XT, and not on the same pipelines in every run: an access of the
/// first pages in `amdxc64.dll`) and the thread that deals with what it
/// left. The handler does the least there is: it notes where, signals, and
/// leaves the worker asleep for good. No allocation, no lock, no file: the
/// crash may have been under any of them. The watcher puts the pipeline on
/// the list of those left out (`warm-bad.txt`), says it is not made, gives
/// the worker's place in the counts back and starts another worker.
fn guard_start() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    use windows_sys::Win32::System::Threading::CreateEventW;
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !guard_on() {
        return;
    }
    ONCE.call_once(|| {
        // SAFETY: an unnamed auto-reset event, no security attributes.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if event.is_null() {
            tracing::warn!("no event for the driver's compiler crashes: they are not caught");
            return;
        }
        CRASH_EVENT.store(event as usize, Ordering::Release);
        let spawned = std::thread::Builder::new().name("victory-pipeline-watch".into()).spawn(watch);
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no thread for the driver's compiler crashes: they are not caught");
            return;
        }
        // SAFETY: a handler that lives as long as the process.
        unsafe {
            AddVectoredExceptionHandler(1, Some(compiler_crashed));
        }
        tracing::info!("the driver's compiler crashing on a worker is caught (VICTORY_COMPILER_GUARD=0: not)");
    });
}

/// The handler: a worker inside a compile read or wrote one of the first
/// pages. See [`guard_start`].
unsafe extern "system" fn compiler_crashed(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::System::Threading::{SetEvent, Sleep};
    const ACCESS_VIOLATION: i32 = 0xc000_0005_u32 as i32;
    const CONTINUE_SEARCH: i32 = 0;
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).ExceptionRecord };
    if record.ExceptionCode != ACCESS_VIOLATION || record.NumberParameters < 2 || record.ExceptionInformation[1] >= 0x10000 {
        return CONTINUE_SEARCH;
    }
    if !COMPILING.try_with(|c| c.get()).unwrap_or(false) {
        return CONTINUE_SEARCH;
    }
    let slot = WORKER_SLOT.try_with(|s| s.get()).unwrap_or(usize::MAX);
    if slot >= MAX_SLOTS {
        return CONTINUE_SEARCH;
    }
    let _ = COMPILING.try_with(|c| c.set(false));
    FAULT_ADDR[slot].store(record.ExceptionInformation[1] as u64, Ordering::Relaxed);
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the system passes a live context.
    FAULT_RIP[slot].store(unsafe { (*(*info).ContextRecord).Rip }, Ordering::Relaxed);
    CRASHED[slot].store(true, Ordering::Release);
    let event = CRASH_EVENT.load(Ordering::Acquire);
    if event != 0 {
        // SAFETY: the event lives as long as the process.
        unsafe { SetEvent(event as windows_sys::Win32::Foundation::HANDLE) };
    }
    loop {
        // SAFETY: a plain sleep.
        unsafe { Sleep(u32::MAX - 1) };
    }
}

/// The thread that deals with the workers the compiler crashed under.
fn watch() {
    use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};
    loop {
        let event = CRASH_EVENT.load(Ordering::Acquire);
        // SAFETY: the event lives as long as the process.
        unsafe { WaitForSingleObject(event as windows_sys::Win32::Foundation::HANDLE, INFINITE) };
        for slot in 0..MAX_SLOTS {
            if CRASHED[slot].swap(false, Ordering::AcqRel) {
                lost(slot);
            }
        }
    }
}

/// Spawns a one-shot worker thread to retry `p` single-threaded under
/// [`COMPILE_GATE`]'s exclusive lock after a parallel `amdxc64` crash.
fn spawn_retry(p: Arc<Pending>, screens_only: bool) -> bool {
    let slot = NEXT_SLOT.fetch_add(1, Ordering::Relaxed);
    if slot >= MAX_SLOTS {
        return false;
    }
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipeline-retry-{slot}"))
        .spawn(move || {
            WORKER_SLOT.set(slot);
            {
                let _compile = Compile::begin(slot, &p, screens_only, true);
                p.make();
            }
            if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                drained();
            } else {
                super::library::save(false);
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no worker to retry crashed pipeline single-threaded");
        false
    } else {
        true
    }
}

/// The worker at `slot` is gone with the pipeline it was making.
fn lost(slot: usize) {
    let job = {
        let mut jobs = ACTIVE_JOBS.lock().unwrap_or_else(|e| e.into_inner());
        jobs.get_mut(slot).and_then(Option::take)
    };
    let Some((p, screens_only, was_retry)) = job else {
        return;
    };
    // Release the crashed thread's gate hold so exclusive retries / parallel workers proceed.
    if was_retry {
        COMPILE_GATE.exit_exclusive();
    } else {
        COMPILE_GATE.exit_shared();
    }
    if p.was_slow.swap(false, Ordering::Relaxed) {
        let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
        queue.slow_now = queue.slow_now.saturating_sub(1);
        GENERAL_AVAILABLE.notify_one();
    }
    let rip = FAULT_RIP[slot].load(Ordering::Relaxed);
    let addr = FAULT_ADDR[slot].load(Ordering::Relaxed);
    let key_buf = super::key_hex(&p.key);
    let key_str = super::key_hex_str(&key_buf);
    // Transient parallel compiler races in amdxc64: retry once single-threaded
    // before permanently blacklisting into `warm-bad.txt`.
    if !was_retry && !p.retried.swap(true, Ordering::AcqRel) {
        tracing::warn!(
            key = key_str,
            shaders = ?p.shaders,
            rip = format_args!("{rip:#x}"),
            address = format_args!("{addr:#x}"),
            "the driver's compiler crashed during parallel compile: retrying once single-threaded"
        );
        if !spawn_worker(screens_only) {
            tracing::warn!("no worker to take the place of the one the compiler crashed under");
        }
        if spawn_retry(p.clone(), screens_only) {
            return;
        }
    }
    tracing::error!(
        key = key_str,
        shaders = ?p.shaders,
        rip = format_args!("{rip:#x}"),
        address = format_args!("{addr:#x}"),
        retried = was_retry,
        "the driver's compiler crashed making a pipeline: it is left out, the worker is replaced"
    );
    let known = bad_keys(|keys| !keys.insert(p.key));
    if !known && let Some(dir) = super::library::folder() {
        use std::io::Write;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("warm-bad.txt"));
        if let Ok(mut f) = file {
            let _ = writeln!(f, "{key_str} ended the compiler in a run: {:?}", p.shaders);
        }
    }
    p.made_state.store(STATE_FAILED, Ordering::Release);
    notify_done(&p);
    if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
        drained();
    }
    if !was_retry && !spawn_worker(screens_only) {
        tracing::warn!("no worker to take the place of the one the compiler crashed under");
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    if let Some(reason) = p.left_out {
        return Some(reason.into());
    }
    match p.made_state.load(Ordering::Acquire) {
        STATE_READY => None,
        STATE_UNBUILT | STATE_EVICTED => Some("being made".into()),
        _ => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Sharded completion condvars: avoids thundering-herd wakeups and allows
/// [`Pending::make`] to skip locking completely when no thread is waiting.
const DONE_SHARDS: usize = 16;
static DONE: [(Mutex<()>, Condvar); DONE_SHARDS] =
    [const { (Mutex::new(()), Condvar::new()) }; DONE_SHARDS];

fn done_shard(p: &Pending) -> &'static (Mutex<()>, Condvar) {
    &DONE[((p as *const Pending as usize) >> 6) & (DONE_SHARDS - 1)]
}

fn notify_done(p: &Pending) {
    // Pairs with `wait()` incrementing `p.waiters` with SeqCst before checking `p.made`.
    std::sync::atomic::fence(Ordering::SeqCst);
    if p.waiters.load(Ordering::SeqCst) > 0 {
        let (lock, done) = done_shard(p);
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Waiting allowed per title frame, over all threads (`VICTORY_PIPELINE_WAIT_MS`,
/// default `16` ms shared wall-clock deadline: finishes cached `LoadPipeline` hits
/// across all concurrent recording threads while capping cold compile stalls).
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(16);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use (or re-materializes if previously dormant-evicted).
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // Fast path: already materialized in RAM (zero locks, updates LRU stamp).
    let ptr = p.active_ptr.load(Ordering::Acquire);
    if !ptr.is_null() {
        p.note_bound(crate::frame::current(), crate::host::coarse_ms());
        // SAFETY: `ptr` remains valid for at least the 2 s eviction quarantine
        // + GPU fence + `bury_lag()` after `active_ptr` is cleared.
        return Some(unsafe { &*ptr });
    }
    if p.made_state.load(Ordering::Acquire) == STATE_FAILED {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    start(p, REQUESTED);
    wait(p, crate::frame::current());
    let ptr = p.active_ptr.load(Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    p.note_bound(crate::frame::current(), crate::host::coarse_ms());
    // SAFETY: as above.
    Some(unsafe { &*ptr })
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    !p.active_ptr.load(Ordering::Acquire).is_null()
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is a confirmed opening-scene / bake producer.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits up to `left` for `p.made` to be populated, tracking active waiters so
/// [`notify_done`] skips locking when no thread is waiting.
fn wait_for_made(p: &Pending, left: Duration) {
    let started = Instant::now();
    struct WaiterGuard<'a>(&'a AtomicU32);
    impl Drop for WaiterGuard<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::AcqRel);
        }
    }
    p.waiters.fetch_add(1, Ordering::SeqCst);
    let _waiter = WaiterGuard(&p.waiters);
    if !p.is_settled() {
        let (lock, done) = done_shard(p);
        let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !p.is_settled() {
            let rest = left.saturating_sub(started.elapsed());
            if rest.is_zero() {
                break;
            }
            held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

/// Per-frame shared wall-clock deadline: the first thread to wait in `frame`
/// anchors `deadline = now + budget`, and every concurrent thread in `frame`
/// shares the same remaining wall time instead of pre-debiting the whole budget
/// to thread 1 and starving threads 2–6 to 0 ms.
struct SharedDeadline {
    frame: u64,
    deadline: Option<Instant>,
}

impl SharedDeadline {
    const fn new() -> Self {
        Self {
            frame: u64::MAX,
            deadline: None,
        }
    }

    fn remaining(&mut self, frame: u64, now: Instant, budget: Duration) -> Duration {
        if self.frame != frame {
            self.frame = frame;
            self.deadline = Some(now + budget);
        }
        self.deadline.map_or(Duration::ZERO, |d| d.saturating_duration_since(now))
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    let shared = shared_budget();
    if shared {
        static DEADLINE: [Mutex<SharedDeadline>; 2] = [const { Mutex::new(SharedDeadline::new()) }; 2];
        let started = Instant::now();
        let left = DEADLINE[usize::from(screen)]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remaining(frame, started, budget);
        if left.is_zero() {
            return;
        }
        wait_for_made(p, left);
        let waited = started.elapsed().min(left);
        if screen && waited > Duration::from_millis(500) {
            tracing::info!(
                ms = waited.as_millis() as u64,
                made = p.is_settled(),
                shaders = ?p.shaders,
                "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
            );
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::PipelineWaitUs,
            started.elapsed().as_micros() as u64,
        );
        return;
    }
    // Legacy non-shared cumulative per-frame budget (`VICTORY_PIPELINE_WAIT_SHARED=0`).
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        budget.saturating_sub(spent.1)
    };
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    wait_for_made(p, left);
    let waited = started.elapsed().min(left);
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.is_settled(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 == frame {
            spent.1 += waited;
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if std::env::var_os("VICTORY_PIPELINE_SLOW").is_some_and(|v| v == "0") {
            usize::MAX
        } else {
            (worker_count(None, std::thread::available_parallelism().map_or(2, |n| n.get())) / 4).max(1)
        }
    })
}

fn writes_depth_only(state: &GraphicsState) -> bool {
    use windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_WRITE_MASK_ALL;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
    state.targets.NumRenderTargets == 0
        && state.depth_format != DXGI_FORMAT_UNKNOWN
        && state.depth.DepthEnable.as_bool()
        && state.depth.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    /// Prewarmed passes over the screen, prioritized ahead of depth/other
    /// prewarmed pipelines on general workers without blocking [`SCREEN_WORKERS`].
    early_screen: VecDeque<Arc<Pending>>,
    /// Prewarmed depth-only passes (shadow maps and depth prepasses), prioritized
    /// ahead of other prewarmed pipelines when no draw is waiting.
    early_depth: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early_screen: VecDeque::new(),
            early_depth: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    #[cfg(test)]
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let is_slow = priority == EARLY && slow_limit() != usize::MAX && super::library::slow(&p.key);
        self.push_as(p, priority, is_slow)
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else if p.screen {
            self.early_screen.push_back(p.clone());
        } else if p.graphics.as_ref().is_some_and(|state| writes_depth_only(state)) {
            self.early_depth.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    /// Pops the highest-priority pending pipeline and whether it was actively
    /// `REQUESTED` by a draw/dispatch (as opposed to background `EARLY` pre-warm).
    fn pop_with_kind(&mut self, screens_only: bool, slow_limit: usize) -> Option<(Arc<Pending>, bool)> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut is_requested = false;
        let mut p = self.screen.pop_front();
        if p.is_some() {
            is_requested = true;
        } else if !screens_only {
            if let Some(req) = self.requested.pop_front() {
                p = Some(req);
                is_requested = true;
            } else {
                p = unclaimed(&mut self.early_screen)
                    .or_else(|| unclaimed(&mut self.early_depth))
                    .or_else(|| unclaimed(&mut self.early));
                if p.is_none() && self.slow_now < slow_limit {
                    p = unclaimed(&mut self.slow);
                    if let Some(p) = &p {
                        p.was_slow.store(true, Ordering::Relaxed);
                        self.slow_now += 1;
                    }
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some((p, is_requested))
    }

    #[cfg(test)]
    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        self.pop_with_kind(screens_only, slow_limit).map(|(p, _)| p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static SCREEN_AVAILABLE: Condvar = Condvar::new();
static GENERAL_AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Acquire) >= priority
        && p.made_state.load(Ordering::Acquire) != STATE_EVICTED
    {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let is_slow = priority == EARLY && slow_limit() != usize::MAX && super::library::slow(&p.key);
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push_as(p, priority, is_slow) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        if priority == REQUESTED && p.screen && screens_first() {
            SCREEN_AVAILABLE.notify_one();
            GENERAL_AVAILABLE.notify_one();
        } else if !is_slow || queue.slow_now < slow_limit() {
            GENERAL_AVAILABLE.notify_one();
        }
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool, on_idle: &mut dyn FnMut()) -> (Arc<Pending>, bool) {
    let cv = if screens_only {
        &SCREEN_AVAILABLE
    } else {
        &GENERAL_AVAILABLE
    };
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(item) = queue.pop_with_kind(screens_only, slow_limit()) {
            return item;
        }
        on_idle();
        queue = cv.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count(asked: Option<usize>, processors: usize) -> usize {
    worker_count_for(asked, processors, false)
}

fn worker_count_for(asked: Option<usize>, processors: usize, limited_ram: bool) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if limited_ram => (processors * 3 / 4).clamp(2, 8),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count_for(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            crate::residency::limited_host_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        guard_start();
        for i in 0..workers + screens {
            if spawn_worker(i >= workers) {
                running = true;
            }
        }
        running
    })
}

/// Starts one worker: one for passes over the screen alone, or one of the
/// others.
fn spawn_worker(screens_only: bool) -> bool {
    let slot = NEXT_SLOT.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{slot}"))
        .spawn(move || {
            WORKER_SLOT.set(slot);
            // Below the title's own threads (loading, recording),
            // and the streaming one, which is below normal itself
            // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
            let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
            let base_priority = {
                use windows::Win32::System::Threading::*;
                if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL }
            };
            let set_priority = |priority| {
                // SAFETY: a plain call on our own thread.
                unsafe {
                    use windows::Win32::System::Threading::{GetCurrentThread, SetThreadPriority};
                    let _ = SetThreadPriority(GetCurrentThread(), priority);
                }
            };
            set_priority(base_priority);
            let mut boosted = false;
            loop {
                let (p, is_requested) = next(screens_only, &mut || {
                    if boosted {
                        set_priority(base_priority);
                        boosted = false;
                    }
                });
                if is_requested != boosted {
                    use windows::Win32::System::Threading::THREAD_PRIORITY_NORMAL;
                    set_priority(if is_requested { THREAD_PRIORITY_NORMAL } else { base_priority });
                    boosted = is_requested;
                }
                {
                    // (If the compiler crashes in here this thread never
                    // returns: `lost` does what follows for it.)
                    let _compile = Compile::begin(slot, &p, screens_only, false);
                    p.make();
                }
                if p.was_slow.load(Ordering::Relaxed) {
                    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                    queue.slow_now -= 1;
                    GENERAL_AVAILABLE.notify_one();
                }
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no pipeline worker");
        false
    } else {
        true
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved promptly whenever new compiled PSOs were added (bypassing
/// `SAVE_EVERY_MS` so late gameplay PSOs survive abrupt process exit).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(super::library::has_added());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count(None, 1), 2);
        assert_eq!(worker_count(None, 24), 16);
        assert_eq!(worker_count(None, 64), 16);
        assert_eq!(worker_count(Some(0), 24), 1);
        assert_eq!(worker_count(Some(20), 24), 20);
        assert_eq!(worker_count_for(None, 8, true), 6);
        assert_eq!(worker_count_for(None, 16, true), 8);
        assert_eq!(worker_count_for(None, 24, true), 8);
        assert_eq!(worker_count_for(Some(12), 8, true), 12);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None, false));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while !p.is_settled() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert_eq!(p.made_state.load(Ordering::Acquire), STATE_FAILED);
    }

    /// A pipeline the compiler crashed on is not made again: no compile.
    #[test]
    fn a_pipeline_on_the_bad_list_is_not_compiled() {
        let mut p = Pending::new(Vec::new(), None, false);
        p.key = [0xBA; 16];
        bad_keys(|keys| keys.insert(p.key));
        let p = Arc::new(p);
        let before = SKIPPED_BAD.load(Ordering::Relaxed);
        p.make();
        assert_eq!(p.made_state.load(Ordering::Acquire), STATE_FAILED);
        assert!(SKIPPED_BAD.load(Ordering::Relaxed) > before);
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None, false);
        p.key = [0x5A; 16];
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(!p.is_settled());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None, false));
        let promoted = Arc::new(Pending::new(Vec::new(), None, false));
        let requested = Arc::new(Pending::new(Vec::new(), None, false));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None, false));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None, false);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }

    #[test]
    fn early_depth_precedes_other_prewarming_and_requests_precede_both() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;
        let depth_pipeline = || {
            let mut depth = GraphicsState::default();
            let format = [16u32, DXGI_FORMAT_D32_FLOAT.0 as u32];
            // SAFETY: a complete depth-format subobject of the title pipeline stream.
            unsafe { depth.read(16, format.as_ptr().cast()) };
            Arc::new(Pending::new(Vec::new(), Some(Arc::new(depth)), false))
        };
        let normal = Arc::new(Pending::new(Vec::new(), None, false));
        let promoted = depth_pipeline();
        let shadow = depth_pipeline();
        let mut queue = Queue::new();
        for p in [&normal, &shadow, &promoted] {
            assert_eq!(queue.push(p, EARLY), Some(true));
        }
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        for expected in [&promoted, &shadow, &normal] {
            assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), expected));
        }
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pending_key_matches_reference_describe_hash() {
        use std::ffi::CString;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::PCSTR;

        let mut complex = GraphicsState {
            sample_mask: 0x0f0f,
            depth_bounds: true,
            ..Default::default()
        };
        complex.targets.NumRenderTargets = 2;
        complex.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        complex.targets.RTFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
        complex.depth_format = DXGI_FORMAT_D32_FLOAT;
        complex.names = vec![
            CString::new("POSITION").unwrap(),
            CString::new("TEX\"COORD\\0'").unwrap(),
        ];
        complex.inputs = complex
            .names
            .iter()
            .enumerate()
            .map(|(i, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: i as u32,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: i as u32,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            })
            .collect();

        let states: [Option<Arc<GraphicsState>>; 3] = [
            None,
            Some(Arc::new(GraphicsState::default())),
            Some(Arc::new(complex)),
        ];
        for state in states {
            let expected_desc = state.as_ref().map(|g| g.describe()).unwrap_or_default();
            let expected_key = blake3::hash(expected_desc.as_bytes()).to_hex()[..32].to_owned();
            let p = Pending::new(Vec::new(), state, false);
            let got = super::super::key_hex(&p.key());
            assert_eq!(super::super::key_hex_str(&got), expected_key);
        }
    }

    #[test]
    fn early_screen_priority_and_screen_workers_isolation() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;

        let make_screen = || {
            let mut p = Pending::new(Vec::new(), None, false);
            p.screen = true;
            Arc::new(p)
        };
        let make_depth = || {
            let mut depth = GraphicsState::default();
            let format = [16u32, DXGI_FORMAT_D32_FLOAT.0 as u32];
            // SAFETY: a complete depth-format subobject.
            unsafe { depth.read(16, format.as_ptr().cast()) };
            Arc::new(Pending::new(Vec::new(), Some(Arc::new(depth)), false))
        };
        let make_plain = || Arc::new(Pending::new(Vec::new(), None, false));

        let req_screen = make_screen();
        let req_normal = make_plain();
        let early_screen = make_screen();
        let early_depth = make_depth();
        let early_normal = make_plain();
        let early_slow = make_plain();

        let mut queue = Queue::new();
        assert_eq!(queue.push_as(&early_slow, EARLY, true), Some(true));
        assert_eq!(queue.push_as(&early_normal, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&early_depth, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&early_screen, EARLY, false), Some(true));
        assert_eq!(queue.push_as(&req_normal, REQUESTED, false), Some(true));
        assert_eq!(queue.push_as(&req_screen, REQUESTED, false), Some(true));

        // Dedicated screen workers only take REQUESTED screen passes, never early_screen or others.
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &req_screen));
        assert!(queue.pop(true, 1).is_none());

        // General workers drain in strict priority: REQUESTED > early_screen > early_depth > early > slow.
        for expected in [&req_normal, &early_screen, &early_depth, &early_normal, &early_slow] {
            assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), expected));
        }
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pending_wait_and_make_concurrent_stress() {
        use std::sync::Barrier;

        // First exercise unwaited make (fast path with waiters == 0).
        let unwaited = Arc::new(Pending::new(Vec::new(), None, false));
        assert_eq!(unwaited.waiters.load(Ordering::SeqCst), 0);
        unwaited.make();
        assert_eq!(unwaited.made_state.load(Ordering::Acquire), STATE_FAILED);
        assert_eq!(unwaited.waiters.load(Ordering::SeqCst), 0);

        // Now stress concurrent waiters racing with `Pending::make()` so no wakeup is ever lost.
        const ROUNDS: usize = 64;
        const WAITERS: usize = 4;
        for _ in 0..ROUNDS {
            let p = Arc::new(Pending::new(Vec::new(), None, false));
            let barrier = Arc::new(Barrier::new(WAITERS + 1));
            let handles: Vec<_> = (0..WAITERS)
                .map(|_| {
                    let p = p.clone();
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        wait_for_made(&p, Duration::from_secs(5));
                        assert!(p.is_settled(), "waiter returned without observing made");
                    })
                })
                .collect();
            barrier.wait();
            p.make();
            for h in handles {
                h.join().unwrap();
            }
            assert_eq!(p.waiters.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn shared_deadline_allows_concurrent_threads_and_resets_per_frame() {
        let mut deadline = SharedDeadline::new();
        let t0 = Instant::now();
        let budget = Duration::from_millis(16);
        // Thread 1 sets frame 10's deadline; concurrent threads 2..6 at t0 + 1ms still get 15ms (not 0ms).
        assert_eq!(deadline.remaining(10, t0, budget), budget);
        let t1 = t0 + Duration::from_millis(1);
        for _ in 0..5 {
            assert_eq!(deadline.remaining(10, t1, budget), Duration::from_millis(15));
        }
        // Past deadline on frame 10 -> 0 ms left.
        assert_eq!(deadline.remaining(10, t0 + Duration::from_millis(20), budget), Duration::ZERO);
        // Next frame resets to a fresh 16 ms deadline.
        let t2 = t0 + Duration::from_millis(25);
        assert_eq!(deadline.remaining(11, t2, budget), budget);
    }

    #[test]
    fn compile_gate_serializes_exclusive_retry_and_recovers_after_crash_exit() {
        let gate = CompileGate::new();
        gate.enter_shared();
        gate.enter_shared();
        // Simulate one worker crashing and `lost()` releasing its hold, and the other finishing.
        gate.exit_shared();
        gate.exit_shared();
        gate.enter_exclusive();
        assert_eq!(*gate.state.lock().unwrap(), (0, true));
        gate.exit_exclusive();
        assert_eq!(*gate.state.lock().unwrap(), (0, false));
    }

    #[test]
    fn local_lights_compute_shader_is_not_left_out_by_default() {
        let shader = super::super::shader(
            "compute",
            b"prefix_CS_GatherRaymarchedLocalLightsClassified_suffix",
        );
        assert!(shader.has_flag(super::super::FLAG_LOCAL_LIGHTS));
        assert_eq!(left_out(&[shader]), None);
    }

    #[test]
    fn dormant_lru_respects_pinning_and_60s_idle_window() {
        assert!(!pso_dormant(true, 200_000, 1_000, 10_000, 1, 60_000), "pinned PSOs never evict");
        assert!(!pso_dormant(false, 200_000, 1_000, 10_000, 1, 0), "idle_ms=0 disables eviction");
        assert!(!pso_dormant(false, 60_999, 1_000, 10_000, 1, 60_000), "below 60 s idle");
        assert!(!pso_dormant(false, 70_000, 1_000, 50, 1, 60_000), "below 60 frames idle");
        assert!(pso_dormant(false, 61_000, 1_000, 100, 40, 60_000), "non-pinned >= 60 s & >= 60 frames evicts");

        // Builder / screen / scene_input pipelines are pinned at creation.
        let builder = Pending::new(Vec::new(), None, true);
        assert!(builder.is_pinned());
        let plain = Pending::new(Vec::new(), None, false);
        assert!(!plain.is_pinned());
    }
}
