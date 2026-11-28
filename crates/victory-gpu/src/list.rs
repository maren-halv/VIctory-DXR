//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

#[cfg(windows)]
pub(crate) mod thin;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (the title asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: the title's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Upload>>,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    /// The parts of the recording before `host`, in order, each closed: the
    /// list is cut at its waits for memory values ([`cut`]).
    #[cfg(windows)]
    earlier: Vec<Part>,
    /// The waits recorded before the first work of `host`: the queue stands
    /// at them before it runs `host`.
    #[cfg(windows)]
    gate: Vec<ValueWait>,
    /// With `VICTORY_THIN=1`, of a list the title has: what it wrote down
    /// ([`thin`]; then there is no `host`).
    #[cfg(windows)]
    thin: thin::Front,
    /// True for a `VICTORY_THIN=1` front-end list (never holds a `host`);
    /// false for direct lists and thin's backing lists (which re-acquire
    /// `host` on [`reset`] after [`detach`]).
    #[cfg(windows)]
    is_thin: bool,
}

/// One part of a title list cut at its waits: the waits the queue stands
/// at, then the host list that runs once they are met.
#[cfg(windows)]
struct Part {
    waits: Vec<ValueWait>,
    host: crate::host::HostList,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(feature = "diagnostics")]
    crate::copystat::recorded(started);
}

/// The submission of `lists` has run: each gives its descriptor tables'
/// slots back ([`crate::draw::State::tables_done`]). A list cut at waits
/// keeps them: its state goes with its last part.
#[cfg(windows)]
pub(crate) fn tables_done(lists: &[This]) {
    for &this in lists {
        if this.is_null() {
            continue;
        }
        // SAFETY: the title passes our lists.
        let list = unsafe { &*this.cast::<List>() };
        if list.earlier.is_empty() {
            list.state.lock().unwrap_or_else(|e| e.into_inner()).tables_done();
        }
    }
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

/// The host side of a submitted list, taken from it: a queue that waits
/// for a fence runs it later, and the title may reset the list meanwhile.
/// With it go the upload memory and the dynamic descriptor slots its
/// commands name, kept until it has run.
#[cfg(windows)]
pub(crate) struct Detached {
    host: crate::host::HostList,
    _uploads: Vec<Upload>,
    tables: crate::draw::DynamicLease,
}

#[cfg(windows)]
impl Detached {
    /// A closed list of ours, to run among a title's.
    pub(crate) fn ours(host: crate::host::HostList) -> Self {
        Detached { host, _uploads: Vec::new(), tables: Default::default() }
    }

    pub(crate) fn host(&self) -> &crate::host::HostList {
        &self.host
    }

    /// Releases this list's dynamic descriptor slots once its GPU submission
    /// has completed, without waiting for the full `bury_lag` allocator delay.
    pub(crate) fn release_tables(&mut self) {
        self.tables.release();
    }
}

/// A host list that has run and waits to be used again, and the bytes of
/// upload memory it keeps.
#[cfg(windows)]
struct Idle(crate::host::HostList, u64);

// SAFETY: D3D12 objects are free-threaded, and an idle list is touched by
// nobody until it is taken out again.
#[cfg(windows)]
unsafe impl Send for Idle {}

/// Host lists that have run, reset and ready to record: the host side of
/// a list a queue held, or of a part of a list cut at a wait, is taken
/// from the title's list at its submission, and the title's list needs
/// another. Each was made new and thrown away after its run: a hundred a
/// frame in gameplay, each two command allocators and two lists, and for
/// four in ten a budget and an argument buffer (run 61: "104 host lists
/// made 21.2 ms" a frame, and the graphics kernel's time to free them).
#[cfg(windows)]
static IDLE: std::sync::Mutex<(Vec<Idle>, u64)> = std::sync::Mutex::new((Vec::new(), 0));
#[cfg(windows)]
static IDLE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Maximum upload scratch bytes idle host lists may keep between them when
/// scratch chunks are not pooled (`VICTORY_SCRATCH_POOL=0`).
#[cfg(windows)]
const IDLE_BYTES_MOST: u64 = 32 << 20;

/// Number of idle `HostList`s currently pooled for reuse.
#[cfg(windows)]
pub(crate) fn idle_count() -> u64 {
    IDLE_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Drops excess idle `HostList`s past the current pressure-adjusted cap
/// outside the pool lock so the driver frees their command-allocator commit.
#[cfg(windows)]
pub(crate) fn trim_idle() {
    let cap = crate::host::current_allocator_caps().0;
    let dropped: Vec<Idle> = {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        if idle.0.len() <= cap {
            return;
        }
        let excess: Vec<Idle> = idle.0.drain(cap..).collect();
        let freed_bytes: u64 = excess.iter().map(|Idle(_, b)| *b).sum();
        idle.1 = idle.1.saturating_sub(freed_bytes);
        IDLE_COUNT.store(idle.0.len() as u64, std::sync::atomic::Ordering::Relaxed);
        excess
    };
    if !dropped.is_empty() {
        crate::host::retire_com_async(dropped);
    }
}

/// Whether host lists that have run are used again (on by default;
/// `VICTORY_LIST_POOL=0`: each is made new and thrown away, as before).
#[cfg(windows)]
fn pools() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_POOL").is_none_or(|v| v != "0"))
}

/// A host list open for recording: one that has run, as a title's list is
/// after its reset, or a new one.
#[cfg(windows)]
pub(crate) fn fresh_list() -> Option<crate::host::HostList> {
    let take_idle = || {
        let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
        let taken = idle.0.pop();
        if let Some(Idle(_, bytes)) = &taken {
            idle.1 = idle.1.saturating_sub(*bytes);
            IDLE_COUNT.store(idle.0.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        taken
    };
    if let Some(Idle(mut list, _)) = take_idle() {
        list.open_for_recording_mut();
        return Some(list);
    }
    crate::host::reclaim_landed();
    if let Some(Idle(mut list, _)) = take_idle() {
        list.open_for_recording_mut();
        return Some(list);
    }
    crate::host::host().and_then(|h| h.new_list().ok())
}

#[cfg(windows)]
#[derive(Default)]
struct RecycleState {
    queue: std::collections::VecDeque<Idle>,
    done: Vec<Idle>,
    pending: usize,
}

#[cfg(windows)]
struct RecyclePool {
    turn: std::sync::Mutex<()>,
    state: std::sync::Mutex<RecycleState>,
    work_cv: std::sync::Condvar,
    done_cv: std::sync::Condvar,
}

#[cfg(windows)]
impl RecyclePool {
    fn get() -> &'static Self {
        static POOL: std::sync::OnceLock<&'static RecyclePool> = std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            let pool: &'static RecyclePool = Box::leak(Box::new(RecyclePool {
                turn: std::sync::Mutex::new(()),
                state: std::sync::Mutex::new(RecycleState::default()),
                work_cv: std::sync::Condvar::new(),
                done_cv: std::sync::Condvar::new(),
            }));
            let workers = 2usize;
            for i in 1..workers {
                let _ = std::thread::Builder::new()
                    .name(format!("victory-recycle-{i}"))
                    .spawn(move || {
                        loop {
                            let mut item = {
                                let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                                while s.queue.is_empty() {
                                    s = pool.work_cv.wait(s).unwrap_or_else(|e| e.into_inner());
                                }
                                s.queue.pop_front().unwrap()
                            };
                            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                item.0.clear_for_idle_mut();
                                item.1 = item.0.scratch_kept_mut();
                            }))
                            .is_ok();
                            let ready = ok.then_some(item);
                            let mut s = pool.state.lock().unwrap_or_else(|e| e.into_inner());
                            if let Some(item) = ready {
                                s.done.push(item);
                            }
                            s.pending -= 1;
                            if s.pending == 0 {
                                pool.done_cv.notify_one();
                            }
                        }
                    });
            }
            pool
        })
    }

    fn reset_all(&self, items: Vec<Idle>) -> Vec<Idle> {
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        let count = items.len();
        {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            s.queue.extend(items);
            s.done.reserve(count);
            s.pending = count;
        }
        self.work_cv.notify_all();
        loop {
            let item = {
                let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
                s.queue.pop_front()
            };
            let Some(mut item) = item else {
                break;
            };
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                item.0.clear_for_idle_mut();
                item.1 = item.0.scratch_kept_mut();
            }))
            .is_ok();
            let ready = ok.then_some(item);
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(item) = ready {
                s.done.push(item);
            }
            s.pending -= 1;
        }
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.pending > 0 {
            s = self.done_cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        std::mem::take(&mut s.done)
    }
}

#[cfg(windows)]
fn recycle_collected(mut to_reset: Vec<Idle>, cap: usize) {
    if !to_reset.is_empty() {
        // `clear_for_idle_mut` detaches allocator leases and clears CPU state
        // without calling D3D12 `Reset`, so small batches stay inline.
        const PARALLEL_MIN: usize = 32;
        let reset = if to_reset.len() >= PARALLEL_MIN {
            RecyclePool::get().reset_all(to_reset)
        } else {
            for Idle(host, bytes) in &mut to_reset {
                host.clear_for_idle_mut();
                *bytes = host.scratch_kept_mut();
            }
            to_reset
        };
        let mut excess = Vec::new();
        {
            let mut idle = IDLE.lock().unwrap_or_else(|e| e.into_inner());
            for item in reset {
                if idle.0.len() < cap && idle.1 + item.1 <= IDLE_BYTES_MOST {
                    idle.1 += item.1;
                    idle.0.push(item);
                } else {
                    excess.push(item);
                }
            }
            IDLE_COUNT.store(idle.0.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        if !excess.is_empty() {
            crate::host::retire_com_async(excess);
        }
    }
    if IDLE_COUNT.load(std::sync::atomic::Ordering::Relaxed) > cap as u64 {
        trim_idle();
    }
}

/// The host sides of lists whose submission has run (submissions are
/// waited for; `ran` names the ones it ran, by their place in `lists`):
/// what they held is let go, as before, and the host lists are reset and
/// kept for [`fresh_list`] up to the current pressure-adjusted cap.
#[cfg(windows)]
pub(crate) fn recycle(lists: Vec<Detached>, ran: &[usize]) {
    if !pools() {
        return;
    }
    let cap = crate::host::current_allocator_caps().0;
    let mut to_reset = Vec::new();
    let mut current_idle = IDLE_COUNT.load(std::sync::atomic::Ordering::Relaxed) as usize;
    for (i, Detached { host, _uploads, tables }) in lists.into_iter().enumerate() {
        // The descriptor slots and the upload memory its commands named.
        drop(tables);
        drop(_uploads);
        if !ran.contains(&i) || cap == 0 || current_idle >= cap {
            host.release_prelude();
            continue;
        }
        current_idle += 1;
        to_reset.push(Idle(host, 0));
    }
    recycle_collected(to_reset, cap);
}

/// Batch variant of [`recycle`] for `host::land`: combines multiple settled
/// flights into one parallel reset wave and a single `IDLE` lock push.
#[cfg(windows)]
pub(crate) fn recycle_many(batches: Vec<(Vec<Detached>, Vec<usize>)>) {
    if !pools() {
        return;
    }
    let cap = crate::host::current_allocator_caps().0;
    let mut to_reset = Vec::new();
    let mut current_idle = IDLE_COUNT.load(std::sync::atomic::Ordering::Relaxed) as usize;
    for (lists, ran) in batches {
        for (i, Detached { host, _uploads, tables }) in lists.into_iter().enumerate() {
            drop(tables);
            drop(_uploads);
            if !ran.contains(&i) || cap == 0 || current_idle >= cap {
                host.release_prelude();
                continue;
            }
            current_idle += 1;
            to_reset.push(Idle(host, 0));
        }
    }
    recycle_collected(to_reset, cap);
}

/// A wait for a memory value recorded into list `this`, with waits cut
/// into lists (`crate::queue::cuts`). On the console the queue runs the
/// list up to the wait, stands there until the value is in memory, and
/// runs on; the list's bottom-of-pipe writes before the wait are done by
/// then, and other queues may be waiting for exactly those. So the
/// recording is cut here: the host list so far is closed and kept as a
/// part of its own ([`Part`]), the title's calls record into a new host
/// list from here on, and the wait stands before that one. A submission
/// then gives the queue the parts in order ([`detach_parts`]).
///
/// A wait before any work (the first call after a reset, or a wait right
/// after another) cuts nothing: it joins the waits before the current
/// host list.
/// shortcut: a bottom-of-pipe write is done when its part has run, not at
/// its place in the part: a queue waiting for it starts later than on the
/// console, never earlier, and nothing can wait for itself that way (a
/// part has no wait after its first work).
#[cfg(windows)]
fn cut(this: This, wait: ValueWait) {
    if this.is_null() {
        return;
    }
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time and is not submitted while it records.
    let list = unsafe { &mut *this.cast_mut().cast::<List>() };
    let Some(current) = list.host.as_ref() else {
        return;
    };
    if current.untouched() {
        list.gate.push(wait);
        return;
    }
    let Some(fresh) = fresh_list() else {
        // shortcut: without a new host list the wait stands before the part
        // it was recorded in.
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no host list for the part after a wait: the wait moves to the part's start"));
        list.gate.push(wait);
        return;
    };
    current.close();
    static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        tracing::info!(parts = list.earlier.len() + 2, "a list is cut at a wait for a memory value");
    }
    if let Some(done) = list.host.replace(fresh) {
        let waits = std::mem::replace(&mut list.gate, vec![wait]);
        list.earlier.push(Part { waits, host: done });
    }
    // What the title set so far, on the new host list.
    if let Some(host) = list.host.as_ref() {
        list.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .replay(host);
    }
}

/// Whether list `this` recorded a wait for a memory value (with waits cut
/// into lists).
#[cfg(windows)]
pub(crate) fn cut_list(this: This) -> bool {
    if this.is_null() {
        return false;
    }
    // SAFETY: one of our lists, closed and submitted: nothing records
    // into it.
    let list = unsafe { &*this.cast::<List>() };
    !list.earlier.is_empty() || !list.gate.is_empty()
}

/// What a queue is given for submitted lists that were cut at their waits:
/// the waits it stands at, the host lists it runs once they are met, and
/// the order those run in.
#[cfg(windows)]
pub(crate) type Parts = Vec<(Vec<ValueWait>, Vec<Detached>, Vec<usize>)>;

/// Takes the host sides of the submitted `lists`, part by part ([`cut`]),
/// and leaves each list a new, closed host list for its next reset, as
/// [`detach`] does. What comes back is in the order the queue runs it: a
/// new entry starts at every part that has waits; host lists without
/// waits join the entry before them. The list's upload memory and draw
/// state (its dynamic descriptor slots) go with its last part, which runs
/// last.
/// shortcut: a list the submission names a second time runs again only if
/// it has one part and that part is in the entry being filled; otherwise
/// the second naming is dropped (logged once).
#[cfg(windows)]
pub(crate) fn detach_parts(lists: &[This]) -> Parts {
    let mut parts: Parts = vec![(Vec::new(), Vec::new(), Vec::new())];
    // The lists taken: where its one host list is (entry, place), or none
    // for a list of several parts.
    let mut seen: Vec<(This, Option<(usize, usize)>)> = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(&(_, at)) = seen.iter().find(|(t, _)| *t == this) {
            match at {
                Some((entry, place)) if entry + 1 == parts.len() => parts[entry].2.push(place),
                _ => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!("a list named twice by a submission that waits for memory values runs once")
                    });
                }
            }
            continue;
        }
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it (which lazily takes a fresh list).
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(host) = list.host.take() else {
            continue;
        };
        let earlier = std::mem::take(&mut list.earlier);
        let gate = std::mem::take(&mut list.gate);
        let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
        let tables = list.state.lock().unwrap_or_else(|e| e.into_inner()).take_dynamic();
        let single = earlier.is_empty() && gate.is_empty();
        let mut pieces: Vec<(Vec<ValueWait>, Detached)> = earlier
            .into_iter()
            .map(|part| {
                (
                    part.waits,
                    Detached {
                        host: part.host,
                        _uploads: Vec::new(),
                        tables: Default::default(),
                    },
                )
            })
            .collect();
        pieces.push((
            gate,
            Detached {
                host,
                _uploads: uploads,
                tables,
            },
        ));
        let mut last = None;
        for (waits, detached) in pieces {
            if !waits.is_empty() {
                parts.push((waits, Vec::new(), Vec::new()));
            }
            let entry = parts.len() - 1;
            let (_, taken, order) = &mut parts[entry];
            let place = taken.len();
            order.push(place);
            taken.push(detached);
            last = Some((entry, place));
        }
        seen.push((this, if single { last } else { None }));
    }
    // (The first entry is empty if the first list begins with a wait.)
    parts.retain(|(_, taken, _)| !taken.is_empty());
    parts
}

/// Takes the host sides of submitted lists (each once, however often the
/// submission names it) and leaves each list a new, closed host list for
/// its next reset: what was taken, and which of them run in what order.
#[cfg(windows)]
pub(crate) fn detach(lists: &[This]) -> (Vec<Detached>, Vec<usize>) {
    let mut taken: Vec<(This, Detached)> = Vec::new();
    let mut order = Vec::new();
    for &this in lists {
        if this.is_null() {
            continue;
        }
        if let Some(i) = taken.iter().position(|(t, _)| *t == this) {
            order.push(i);
            continue;
        }
        let list = this.cast_mut().cast::<List>();
        // SAFETY: one of our lists, closed and submitted: nothing records
        // into it until the title resets it (which lazily takes a fresh list).
        let (host, uploads, tables) = unsafe {
            let Some(host) = (*list).host.take() else {
                continue;
            };
            let uploads =
                std::mem::take(&mut *(*list).uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let tables = (*list)
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take_dynamic();
            (host, uploads, tables)
        };
        order.push(taken.len());
        taken.push((
            this,
            Detached {
                host,
                _uploads: uploads,
                tables,
            },
        ));
    }
    (taken.into_iter().map(|(_, d)| d).collect(), order)
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

/// `pipeline` is the list's initial pipeline (null: none), bound until the
/// title sets another, as after a reset.
fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    #[cfg(windows)]
    if thin::on() {
        let mut list = boxed(kind, pipeline, true);
        list.thin.begin(kind, pipeline);
        return object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    }
    object::hand_out(LIST, boxed(kind, pipeline, false), iid, &LIST_IIDS, out)
}

/// A backing list for `thin::back_list` with no open host list until `reset`.
#[cfg(windows)]
fn boxed_closed(kind: u32) -> Box<List> {
    Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        host: None,
        state: std::sync::Mutex::new(crate::draw::State::default()),
        earlier: Vec::new(),
        gate: Vec::new(),
        thin: Default::default(),
        is_thin: false,
    })
}

/// A list; with `thin` one whose calls are written down (it has no host list).
fn boxed(kind: u32, pipeline: This, thin: bool) -> Box<List> {
    #[cfg(not(windows))]
    let _ = thin;
    let list = Box::new(List {
        object: match (crate::device::newer(), thin) {
            (true, false) => Object::new(&NEWER_LIST_FUNCTIONS),
            (false, false) => Object::new(&LIST_FUNCTIONS),
            #[cfg(windows)]
            (true, true) => Object::new(&NEWER_THIN_FUNCTIONS),
            #[cfg(windows)]
            (false, true) => Object::new(&THIN_FUNCTIONS),
            #[cfg(not(windows))]
            _ => Object::new(&LIST_FUNCTIONS),
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: (!thin).then(fresh_list).flatten(),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
        #[cfg(windows)]
        earlier: Vec::new(),
        #[cfg(windows)]
        gate: Vec::new(),
        #[cfg(windows)]
        thin: Default::default(),
        #[cfg(windows)]
        is_thin: thin,
    });
    #[cfg(not(windows))]
    let _ = pipeline;
    list
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    let _called = crate::timeline::span("the title's Close");
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        // Return any unused tail of this list's dynamic descriptor bump slab
        // immediately so closed/in-flight lists hold zero wasted slots.
        // SAFETY: `this` is one of our lists (checked non-null by `host_list`).
        unsafe { &(*this.cast::<List>()).state }
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .trim_dynamic_slab();
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    let _called = crate::timeline::span("the title's list Reset");
    // A wait recorded and never submitted goes with the recording.
    if VALUE_WAITS_COUNT.load(std::sync::atomic::Ordering::Acquire) > 0 {
        let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
        waits.retain(|&(list, _)| list != this as usize);
        VALUE_WAITS_COUNT.store(waits.len(), std::sync::atomic::Ordering::Release);
    }
    // Every submission of the list has run (submissions are waited for).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).clear();
    // The parts of a recording that was cut and never submitted go with it
    // (a submission takes the parts it runs).
    #[cfg(windows)]
    {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: the title passes one of our lists, and neither records
        // into it nor submits it during its reset.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        list.earlier.clear();
        list.gate.clear();
        if let Some(host) = list.host.as_mut() {
            host.reset_mut();
        } else if !list.is_thin {
            list.host = fresh_list();
        }
        if list.host.is_some() {
            list.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset(pipeline);
            #[cfg(feature = "diagnostics")]
            crate::copystat::recorded(started);
        }
    }
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                // A depth texture's unordered access is its colour twin's
                // (the host gives the depth texture none): the barrier is
                // on the texture shaders write.
                match crate::resource::twin_resource(b.first) {
                    Some(twin) => twin.as_raw().cast_const().cast(),
                    None => host(b.first)?,
                }
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Whether guest buffer barriers emit resource-scoped UAV barriers targeting
/// each buffer's host `Shadow` resource when unambiguous, instead of a global
/// `pResource = null` UAV barrier (`VICTORY_SCOPED_UAV_BARRIERS=1`; off by default).
#[cfg(windows)]
fn scoped_uav_barriers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SCOPED_UAV_BARRIERS").is_some_and(|v| v == "1");
        if on {
            tracing::info!("resource-scoped buffer UAV barriers enabled (VICTORY_SCOPED_UAV_BARRIERS=1)");
        }
        on
    })
}

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        if count > 0 {
            with_state(this, |_, state| state.synced());
        }
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        if scoped_uav_barriers() {
            let has_global_uav = host.iter().any(|b| b.kind == 2 && b.first.is_null());
            if !has_global_uav {
                let mut scoped: Vec<std::sync::Arc<crate::shadow::Shadow>> = Vec::new();
                let mut any_buffer = false;
                let mut all_resolved = true;
                for b in barriers {
                    if b.kind == 1 || b.first.is_null() {
                        continue;
                    }
                    let Some((ptr, len)) = crate::resource::buffer_bytes(b.first) else {
                        continue;
                    };
                    any_buffer = true;
                    if let Some(shadow) = crate::shadow::scoped_barrier_shadow(ptr as u64, len) {
                        let raw = windows::core::Interface::as_raw(&shadow.buffer);
                        if !scoped
                            .iter()
                            .any(|s| windows::core::Interface::as_raw(&s.buffer) == raw)
                        {
                            scoped.push(shadow);
                        }
                    } else {
                        all_resolved = false;
                        break;
                    }
                }
                if any_buffer {
                    if all_resolved && !scoped.is_empty() {
                        for shadow in scoped {
                            let raw = windows::core::Interface::as_raw(&shadow.buffer)
                                .cast_const()
                                .cast();
                            list.keep_shadow(shadow);
                            host.push(Barrier {
                                kind: 2,
                                flags: 0,
                                first: raw,
                                second: 0,
                                last: 0,
                            });
                        }
                    } else {
                        host.push(UAV_BARRIER);
                        #[cfg(feature = "diagnostics")]
                        crate::copystat::add(crate::copystat::C::WaitsForBuffers, 1);
                    }
                }
            }
        } else if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            // Buffers have host copies, not resources the barrier can name:
            // a barrier on one waits for every shader write before it.
            host.push(UAV_BARRIER);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WaitsForBuffers, 1);
        }
        if crate::host::coalesce_uav_barriers() && list.is_uav_clean() {
            host.retain(|b| !(b.kind == 2 && b.first.is_null()));
        }
        let emits_global_uav = host.iter().any(|b| b.kind == 2 && b.first.is_null());
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
            if emits_global_uav {
                list.mark_uav_clean();
            }
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

/// Appends `(key, before, after, payload)` into `out`, folding consecutive
/// transitions for the same `key` (`A -> B` + `B -> C` becomes `A -> C`, and
/// `A -> B` + `B -> A` is dropped) so one `ResourceBarrier` call never names
/// the same subresource twice.
fn fold_transition<T: Copy>(
    out: &mut [(usize, u32, u32, T)],
    len: &mut usize,
    key: usize,
    before: u32,
    after: u32,
    payload: T,
) {
    if before == after {
        return;
    }
    if let Some(pos) = out[..*len].iter().position(|e| e.0 == key) {
        if out[pos].1 == after {
            out.copy_within(pos + 1..*len, pos);
            *len -= 1;
        } else {
            out[pos].2 = after;
        }
        return;
    }
    out[*len] = (key, before, after, payload);
    *len += 1;
}

/// Moves multiple title textures' host copies to their requested states in a
/// single batched `ResourceBarrier` call, folding duplicate-resource transitions.
#[cfg(windows)]
pub(crate) fn ensure_states(list: &crate::host::HostList, transitions: &[(This, u32)]) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_STATES as State, ID3D12Resource,
    };
    use windows::core::Interface;

    match transitions {
        [] => return,
        &[(resource, state)] => {
            ensure_state(list, resource, state);
            return;
        }
        _ => {}
    }

    const INLINE: usize = 16;
    if transitions.len() <= INLINE {
        let mut entries: [(usize, u32, u32, Option<&'static ID3D12Resource>); INLINE] =
            [(0, 0, 0, None); INLINE];
        let mut len = 0usize;
        for &(resource, state) in transitions {
            if let Some((texture, before)) = list.transition(resource, state) {
                let after = crate::host::legal_state(texture, state);
                let key = texture.as_raw() as usize;
                fold_transition(&mut entries, &mut len, key, before, after, Some(texture));
            }
        }
        if len == 0 {
            return;
        }
        let mut barriers: [std::mem::MaybeUninit<D3D12_RESOURCE_BARRIER>; INLINE] =
            [const { std::mem::MaybeUninit::uninit() }; INLINE];
        for i in 0..len {
            let (_, before, after, texture) = entries[i];
            if let Some(texture) = texture {
                barriers[i].write(crate::host::transition(
                    texture,
                    State(before as i32),
                    State(after as i32),
                ));
            }
        }
        // SAFETY: `barriers[..len]` are all initialized (every pushed entry had `Some(texture)`);
        // open list and live textures.
        unsafe {
            let slice = std::slice::from_raw_parts(barriers.as_ptr().cast(), len);
            list.list.ResourceBarrier(slice);
        }
        return;
    }

    let mut entries: Vec<(usize, u32, u32, &'static ID3D12Resource)> =
        Vec::with_capacity(transitions.len());
    for &(resource, state) in transitions {
        if let Some((texture, before)) = list.transition(resource, state) {
            let after = crate::host::legal_state(texture, state);
            let key = texture.as_raw() as usize;
            if before == after {
                continue;
            }
            if let Some(pos) = entries.iter().position(|e| e.0 == key) {
                if entries[pos].1 == after {
                    entries.remove(pos);
                } else {
                    entries[pos].2 = after;
                }
            } else {
                entries.push((key, before, after, texture));
            }
        }
    }
    if entries.is_empty() {
        return;
    }
    let barriers: Vec<D3D12_RESOURCE_BARRIER> = entries
        .into_iter()
        .map(|(_, before, after, texture)| {
            crate::host::transition(texture, State(before as i32), State(after as i32))
        })
        .collect();
    // SAFETY: an open list and live textures.
    unsafe { list.list.ResourceBarrier(&barriers) };
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// A copy of title memory that `list` records: on the card at its place in
/// the list ([`crate::host::HostList::place_copy`]), or at submission.
#[cfg(windows)]
fn buffer_copy(this: This, list: &crate::host::HostList, to: *mut u8, from: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    if list.place_copy(to as usize, from as usize, len) {
        list.mark_uav_dirty();
        with_state(this, |_, state| state.copied());
    } else {
        list.defer_copy(to, from, len);
    }
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        buffer_copy(this, list, to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

/// Whether one of two resources is a target in a stand-in format and the
/// other is not ([`crate::resource::target_format`]): their texels differ
/// in size, so the host cannot copy between them (logged once).
#[cfg(windows)]
fn unlike(dst: This, src: This) -> bool {
    let unlike = crate::resource::stands_in(dst) != crate::resource::stands_in(src);
    if unlike {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(?dst, ?src, "copy between a stand-in target and title-format texels skipped")
        });
    }
    unlike
}

/// Copies left out because both sides were one host texture
/// ([`copy_resource`], [`copy_texture_region`]), since the start.
#[cfg(windows)]
pub(crate) static SELF_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if unlike(dst, src) {
                return;
            }
            // Two title textures over the same memory have one host
            // texture between them: the copy would be of that texture onto
            // itself, which the host does not allow, and it would change
            // nothing. It was recorded all the same, with the texture left
            // in the source state as the copy's destination (run 205's
            // debug layer: "CopyResource with its destination in the
            // COPY_SOURCE state", 37 times, a 2560x1440 depth texture).
            // Only the two state changes and the copy are left out: what
            // is noted of the textures stays as it was. After
            // triippywhale 6e502b3b.
            let same = windows::core::Interface::as_raw(to) == windows::core::Interface::as_raw(from);
            // A source the title fills itself is uploaded first (upstream
            // PR #671): a copy read whatever the host texture held.
            if crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            // Depth textures shaders also write: what they wrote, first.
            list.twin_use(dst, false, true);
            list.twin_use(src, false, false);
            // A destination made level by level gets its levels' memory
            // ahead of this list (`crate::tiles`).
            crate::texture::card_writes(list, dst, to, None);
            crate::texture::host_written(dst);
            if same {
                SELF_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        ?dst,
                        ?src,
                        "a copy between two title textures that share one host texture is left out: the first one"
                    );
                });
                return;
            }
            ensure_states(list, &[(dst, COPY_DEST), (src, COPY_SOURCE)]);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            buffer_copy(this, list, to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            // Texture to buffer: a read back. The list copies the texels
            // into a host buffer of ours; once it has run they go to the
            // title's buffer (`HostList::defer_read_back`).
            // The title reads two tiny float targets back every frame (5x1
            // and 1x1 RGBA32F, 80 and 16 bytes: run 90), and read a buffer
            // nobody wrote. `VICTORY_READ_BACK=0` leaves them unmade, as
            // before.
            static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            let made = read_back(list, dst, x, y, z, src, source_box);
            if n <= 16 || n.is_multiple_of(4096) || made.is_err() {
                static REFUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if made.is_err() && REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 16 {
                    return;
                }
                let source = crate::resource::host_resource(src.resource).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{}x{} f{} mips {}", d.Width, d.Height, d.Format.0, d.MipLevels)
                });
                tracing::info!(
                    copies = n,
                    made = ?made,
                    source = ?source,
                    source_kind = src.kind,
                    destination_kind = dst.kind,
                    destination_offset = dst.footprint.offset,
                    destination_bytes = ?dst.footprint.bytes(),
                    "a texture to buffer copy (a read back)"
                );
            }
            return;
        };
        if unlike(dst.resource, src.resource) {
            return;
        }
        let mut from = src;
        // Both sides are the same subresource of one host texture (set below).
        let mut same = false;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let (buffer, upload_offset) = match list.place_source(data as usize, bytes as usize) {
                // What shaders wrote to it, at this place of the list.
                Some(buffer) => (buffer, 0u64),
                None => {
                    let Some((buffer, offset, staging)) = list.upload_at(host, bytes) else {
                        return;
                    };
                    // The list owns this mapped D3D12 upload buffer. Wine can
                    // report its address as MEM_FREE, so use the host-upload
                    // path.
                    list.defer_fill_buffer(&buffer, staging, data, bytes as usize);
                    (buffer, offset)
                }
            };
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = upload_offset;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() } + upload_offset, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            // The same subresource of one host texture on both sides (two
            // title textures over one memory): not a copy the host allows.
            // The states and the copy are left out below; the rest is
            // noted as before.
            same = texture.as_raw() == to.as_raw()
                && src.kind != FOOTPRINT
                && dst.kind != FOOTPRINT
                && src.footprint.offset as u32 == dst.footprint.offset as u32;
            list.twin_use(src.resource, false, false);
            if crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            from.resource = texture.as_raw().cast_const().cast();
        }
        // A depth texture shaders also write: what they wrote, first (the
        // source above too).
        list.twin_use(dst.resource, false, true);
        // The first copy into a texture of the title's: what it does not
        // write is zeros, as in a texture that was an allocation of its own
        // (`VICTORY_COPY_OVER_TITLE=title`: what title memory holds, which
        // may be another texture's: specks on the waitress's forehead were
        // left in run n04). (The host
        // copy is cut from a block and held whatever lay there, and no
        // upload is made once a copy wrote it: the atlas the title copies
        // its baked decorations into, a 2048x2048 BC3 texture of which a
        // bake writes 256x256 at most, was drawn over a character's face
        // with other textures' bytes wherever nothing was baked, runs n01 and
        // textures-1.)
        static OVER: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let over = *OVER.get_or_init(|| match std::env::var("VICTORY_COPY_OVER_TITLE").as_deref() {
            Ok("0") => 0,
            Ok("title") => 1,
            _ => 2,
        });
        if over != 0 && crate::texture::title_written(to) && !crate::texture::host_wrote(dst.resource as usize) {
            // One a list uploaded, or a draw sampled, holds the title's.
            if over == 2 && !list.used(dst.resource as usize) && !crate::texture::is_staged(dst.resource) {
                if let Some(host) = crate::host::host() {
                    ensure_state(list, dst.resource, COPY_DEST);
                    let _ = list.first_use(dst.resource as usize);
                    crate::texture::zeros(list, host, to);
                }
            } else {
                crate::texture::upload(list, dst.resource, to);
            }
            // (Done before the title's copy writes over it.)
            if list.used(dst.resource as usize) {
                ensure_state(list, dst.resource, COPY_SOURCE);
            }
        }
        // A destination made level by level gets the level's memory ahead
        // of this list (`crate::tiles`): the title's bake copies each level
        // of its result into a block-compressed texture of that kind, and
        // a copy into a level without memory is dropped.
        crate::texture::card_writes(list, dst.resource, to, Some(dst.footprint.offset as u32));
        crate::texture::host_written(dst.resource);
        if same {
            SELF_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        if src.kind == FOOTPRINT {
            ensure_state(list, dst.resource, COPY_DEST);
        } else {
            ensure_states(list, &[(src.resource, COPY_SOURCE), (dst.resource, COPY_DEST)]);
        }
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// Records a texture to buffer copy (a read back) of [`copy_texture_region`]
/// into `list`, or says why not.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)] // the method's own arguments
fn read_back(
    list: &crate::host::HostList,
    dst: CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: CopyLocation,
    source_box: *const [u32; 6],
) -> Result<(), &'static str> {
    use windows::Win32::Graphics::Direct3D12::D3D12_BOX;
    use windows::core::Interface;
    const FOOTPRINT: u32 = 1;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_READ_BACK").is_none_or(|v| v != "0")) {
        return Err("switched off");
    }
    if dst.kind != FOOTPRINT || src.kind == FOOTPRINT {
        return Err("not a texture subresource into a placed footprint");
    }
    if (x, y, z) != (0, 0, 0) {
        return Err("a destination corner other than the footprint's");
    }
    let host = crate::host::host().ok_or("no host")?;
    let (footprint, [size, row, rows, pitch]) = dst.footprint.host_rows().ok_or("an unknown format")?;
    // The title's buffer holds every row at its own pitch.
    let bytes = pitch * (rows - 1) + row;
    let data = buffer_range(dst.resource, dst.footprint.offset, bytes).ok_or("a destination outside its buffer")?;
    let texture = crate::resource::host_resource(src.resource).ok_or("no host texture")?;
    let buffer = host.read_back_buffer(size).ok_or("no read-back buffer")?;
    // As a texture source of any copy: what shaders wrote to a depth twin
    // and what the title wrote itself first, then the state.
    list.twin_use(src.resource, false, false);
    if crate::texture::title_written(texture) {
        crate::texture::upload(list, src.resource, texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let mut from = src;
    from.resource = texture.as_raw().cast_const().cast();
    let mut to = dst;
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint = footprint;
    let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "read back of {:?} sub {} into buffer {:?}+{:#x}: {rows} rows of {row} bytes",
            src.resource, src.footprint.offset as u32, dst.resource, dst.footprint.offset
        )
    });
    // SAFETY: an open list; both locations in the host's layout (asserted
    // in `copy_texture_region`), naming a live texture and our buffer, which
    // the list keeps until it is reset; the box is the title's.
    unsafe {
        list.list
            .CopyTextureRegion((&raw const to).cast(), 0, 0, 0, (&raw const from).cast(), host_box)
    };
    list.defer_read_back(buffer, data, [row as usize, rows as usize, pitch as usize]);
    Ok(())
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        if unlike(dst, src) {
            return;
        }
        // Depth textures shaders also write: what they wrote, first.
        list.twin_use(dst, false, true);
        list.twin_use(src, false, false);
        // As for a copy (`crate::tiles`).
        crate::texture::card_writes(list, dst, to, Some(dst_sub));
        ensure_states(list, &[(dst, RESOLVE_DEST), (src, RESOLVE_SOURCE)]);
        crate::texture::host_written(dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = if crate::resource::stands_in(dst) { a.Format } else { DXGI_FORMAT(format as i32) };
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(Some(texture), record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        // A depth texture shaders also write: what they wrote, first (the
        // clear may be of one slice or of some rectangles).
        list.twin_use(record.resource, false, true);
        ensure_state(list, record.resource, DEPTH_WRITE);
        if crate::resource::depth_and_unordered(record.resource) {
            crate::resource::DEPTH_UNORDERED[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        if list.place_copy(dst as usize, src as usize, len) {
            list.mark_uav_dirty();
            with_state(this, |_, state| state.copied());
            return;
        }
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, of which the
/// host follows one: the counts are threads, not groups (`draw.rs`
/// `THREAD_DIMENSIONS`; the ray-tracing structure builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, flags: u32) {
    count_dispatch();
    #[cfg(windows)]
    with_state(this, |list, s| s.dispatch(list, x, y, z, flags));
    #[cfg(not(windows))]
    let _ = (this, x, y, z, flags);
}

/// `BeginQuery(heap, type, index)`: records a native GPU query when host
/// queries are enabled (`crate::query::enabled()`).
extern "system" fn begin_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        crate::query::begin_query(list, heap, kind, index);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `EndQuery(heap, type, index)`: records a native GPU query end (or
/// timestamp sample) when host queries are enabled.
extern "system" fn end_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        crate::query::end_query(list, heap, kind, index);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `SetPredication(buffer, aligned_offset, operation)`: binds or clears GPU
/// draw/dispatch predication on the host list when native queries are enabled.
extern "system" fn set_predication(this: This, buffer: This, offset: u64, operation: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        crate::query::set_predication(list, buffer, offset, operation);
    }
    #[cfg(not(windows))]
    let _ = (this, buffer, offset, operation);
}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: resolves real
/// host GPU query results into the destination shadow buffer when native
/// queries are enabled (`crate::query::enabled()`), falling back to plausible
/// synthetic deferred writes when disabled (`VICTORY_QUERIES=0`) or headless.
extern "system" fn resolve_queries(
    this: This,
    heap: This,
    kind: u32,
    first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && crate::query::resolve_query_data(list, heap, kind, first, count, buffer, offset)
    {
        list.mark_uav_dirty();
        with_state(this, |_, state| state.copied());
        return;
    }
    #[cfg(not(windows))]
    let _ = (heap, first);
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Attaches a native DXR operation to guest command list `this` in exact
/// command-recording order (safe under both `VICTORY_THIN=0` and `VICTORY_THIN=1`
/// with zero global mutex contention on thin recording threads).
#[cfg(windows)]
pub(crate) fn queue_dxr_op(this: This, op: crate::dxr::DxrOp) {
    if this.is_null() {
        return;
    }
    if let Some(list) = host_list(this) {
        list.push_dxr_op(op);
        return;
    }
    // SAFETY: `this` is a valid guest `List`; thin front-end lists set `is_thin == true`.
    let is_thin = unsafe { (*this.cast::<List>()).is_thin };
    if is_thin {
        thin::queue_dxr_op(this, op);
    }
}

/// Whether `barrier_groups()` and `sync()` use narrowed Enhanced Barriers
/// (`VICTORY_ENHANCED_GROUPS`, on by default when the host device supports
/// Enhanced Barriers + `ID3D12GraphicsCommandList7`; `0` falls back to legacy
/// global `UAV(NULL)` barriers).
#[cfg(windows)]
fn enhanced_groups_enabled(list: &crate::host::HostList) -> bool {
    static WANTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let wanted = *WANTED.get_or_init(|| {
        enhanced_groups_wanted_for(std::env::var_os("VICTORY_ENHANCED_GROUPS").as_deref())
    });
    wanted && list.enhanced_barriers_supported()
}

fn enhanced_groups_wanted_for(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Console `D3D12_BARRIER_TYPE` discriminants (matching PC Enhanced Barriers).
pub(crate) const BARRIER_GROUP_GLOBAL: u32 = 0;
pub(crate) const BARRIER_GROUP_TEXTURE: u32 = 1;
pub(crate) const BARRIER_GROUP_BUFFER: u32 = 2;

/// `D3D12_BARRIER_GROUP` layout passed to slot 146 (`0x490`).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct BarrierGroup {
    pub(crate) kind: u32,
    pub(crate) count: u32,
    pub(crate) barriers: *const u8,
}

/// `D3D12_GLOBAL_BARRIER` layout (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlobalBarrierDesc {
    pub(crate) sync_before: u32,
    pub(crate) sync_after: u32,
    pub(crate) access_before: u32,
    pub(crate) access_after: u32,
}

/// `D3D12_BUFFER_BARRIER` layout (40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BufferBarrierDesc {
    pub(crate) sync_before: u32,
    pub(crate) sync_after: u32,
    pub(crate) access_before: u32,
    pub(crate) access_after: u32,
    pub(crate) resource: This,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// `D3D12_TEXTURE_BARRIER` layout (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextureBarrierDesc {
    pub(crate) sync_before: u32,
    pub(crate) sync_after: u32,
    pub(crate) access_before: u32,
    pub(crate) access_after: u32,
    pub(crate) layout_before: u32,
    pub(crate) layout_after: u32,
    pub(crate) resource: This,
    pub(crate) subresources: [u32; 6],
    pub(crate) flags: u32,
}

#[cfg(windows)]
const _: () = {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ,
        D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE, D3D12_BARRIER_GROUP,
        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
        D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
        D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO,
        D3D12_BARRIER_SYNC_RAYTRACING, D3D12_BARRIER_SYNC_SPLIT, D3D12_BUFFER_BARRIER,
        D3D12_GLOBAL_BARRIER, D3D12_TEXTURE_BARRIER,
    };
    assert!(size_of::<BarrierGroup>() == 16);
    assert!(size_of::<BarrierGroup>() == size_of::<D3D12_BARRIER_GROUP>());
    assert!(size_of::<GlobalBarrierDesc>() == 16);
    assert!(size_of::<GlobalBarrierDesc>() == size_of::<D3D12_GLOBAL_BARRIER>());
    assert!(size_of::<BufferBarrierDesc>() == 40);
    assert!(size_of::<BufferBarrierDesc>() == size_of::<D3D12_BUFFER_BARRIER>());
    assert!(size_of::<TextureBarrierDesc>() == 64);
    assert!(size_of::<TextureBarrierDesc>() == size_of::<D3D12_TEXTURE_BARRIER>());
    assert!(SYNC_RAYTRACING == D3D12_BARRIER_SYNC_RAYTRACING.0 as u32);
    assert!(
        SYNC_EMIT_RAYTRACING_POSTBUILD
            == D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO.0 as u32
    );
    assert!(
        SYNC_BUILD_RAYTRACING_AS
            == D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
    );
    assert!(
        SYNC_COPY_RAYTRACING_AS
            == D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
    );
    assert!(SYNC_SPLIT == D3D12_BARRIER_SYNC_SPLIT.0 as u32);
    assert!(
        ACCESS_RTAS_READ == D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ.0 as u32
    );
    assert!(
        ACCESS_RTAS_WRITE == D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE.0 as u32
    );
};

/// Element size in bytes for a known `BarrierGroup::kind`, or `None` if unknown.
pub(crate) const fn barrier_group_item_bytes(kind: u32) -> Option<usize> {
    match kind {
        BARRIER_GROUP_GLOBAL => Some(size_of::<GlobalBarrierDesc>()),
        BARRIER_GROUP_TEXTURE => Some(size_of::<TextureBarrierDesc>()),
        BARRIER_GROUP_BUFFER => Some(size_of::<BufferBarrierDesc>()),
        _ => None,
    }
}

// D3D12_BARRIER_SYNC bit values.
const SYNC_NONE: u32 = 0x0;
const SYNC_ALL: u32 = 0x1;
const SYNC_DRAW: u32 = 0x2;
const SYNC_INDEX_INPUT: u32 = 0x4;
const SYNC_VERTEX_SHADING: u32 = 0x8;
const SYNC_PIXEL_SHADING: u32 = 0x10;
const SYNC_DEPTH_STENCIL: u32 = 0x20;
const SYNC_RENDER_TARGET: u32 = 0x40;
const SYNC_COMPUTE_SHADING: u32 = 0x80;
const SYNC_RAYTRACING: u32 = 0x100;
const SYNC_COPY: u32 = 0x200;
const SYNC_RESOLVE: u32 = 0x400;
const SYNC_EXECUTE_INDIRECT: u32 = 0x800; // aliased with PREDICATION
const SYNC_ALL_SHADING: u32 = 0x1000;
const SYNC_NON_PIXEL_SHADING: u32 = 0x2000;
const SYNC_EMIT_RAYTRACING_POSTBUILD: u32 = 0x4000;
const SYNC_CLEAR_UAV: u32 = 0x8000;
const SYNC_BUILD_RAYTRACING_AS: u32 = 0x80_0000;
const SYNC_COPY_RAYTRACING_AS: u32 = 0x100_0000;
const SYNC_SPLIT: u32 = 0x8000_0000;

/// Sync bits recognized and legal on a PC Direct queue (unrecognized, video,
/// or Xbox-private bits force fallback).
const KNOWN_DIRECT_SYNC: u32 = SYNC_ALL
    | SYNC_DRAW
    | SYNC_INDEX_INPUT
    | SYNC_VERTEX_SHADING
    | SYNC_PIXEL_SHADING
    | SYNC_DEPTH_STENCIL
    | SYNC_RENDER_TARGET
    | SYNC_COMPUTE_SHADING
    | SYNC_RAYTRACING
    | SYNC_COPY
    | SYNC_RESOLVE
    | SYNC_EXECUTE_INDIRECT
    | SYNC_ALL_SHADING
    | SYNC_NON_PIXEL_SHADING
    | SYNC_EMIT_RAYTRACING_POSTBUILD
    | SYNC_CLEAR_UAV
    | SYNC_BUILD_RAYTRACING_AS
    | SYNC_COPY_RAYTRACING_AS
    | SYNC_SPLIT;

// D3D12_BARRIER_ACCESS bit values.
const ACCESS_COMMON: u32 = 0x0;
const ACCESS_VERTEX_BUFFER: u32 = 0x1;
const ACCESS_CONSTANT_BUFFER: u32 = 0x2;
const ACCESS_INDEX_BUFFER: u32 = 0x4;
const ACCESS_RENDER_TARGET: u32 = 0x8;
const ACCESS_UNORDERED_ACCESS: u32 = 0x10;
const ACCESS_DEPTH_STENCIL_WRITE: u32 = 0x20;
const ACCESS_DEPTH_STENCIL_READ: u32 = 0x40;
const ACCESS_SHADER_RESOURCE: u32 = 0x80;
const ACCESS_STREAM_OUTPUT: u32 = 0x100;
const ACCESS_INDIRECT_ARGUMENT: u32 = 0x200; // aliased with PREDICATION
const ACCESS_COPY_DEST: u32 = 0x400;
const ACCESS_COPY_SOURCE: u32 = 0x800;
const ACCESS_RESOLVE_DEST: u32 = 0x1000;
const ACCESS_RESOLVE_SOURCE: u32 = 0x2000;
const ACCESS_RTAS_READ: u32 = 0x4000;
const ACCESS_RTAS_WRITE: u32 = 0x8000;
const ACCESS_SHADING_RATE_SOURCE: u32 = 0x1_0000;
const ACCESS_NO_ACCESS: u32 = 0x8000_0000;

const KNOWN_BUFFER_ACCESS: u32 = ACCESS_VERTEX_BUFFER
    | ACCESS_CONSTANT_BUFFER
    | ACCESS_INDEX_BUFFER
    | ACCESS_UNORDERED_ACCESS
    | ACCESS_SHADER_RESOURCE
    | ACCESS_STREAM_OUTPUT
    | ACCESS_INDIRECT_ARGUMENT
    | ACCESS_COPY_DEST
    | ACCESS_COPY_SOURCE
    | ACCESS_RTAS_READ
    | ACCESS_RTAS_WRITE;

const KNOWN_DIRECT_ACCESS: u32 = KNOWN_BUFFER_ACCESS
    | ACCESS_RENDER_TARGET
    | ACCESS_DEPTH_STENCIL_WRITE
    | ACCESS_DEPTH_STENCIL_READ
    | ACCESS_RESOLVE_DEST
    | ACCESS_RESOLVE_SOURCE
    | ACCESS_SHADING_RATE_SOURCE
    | ACCESS_NO_ACCESS;

/// Conservative global fallback matching legacy `UAV(NULL)` shader ordering.
const CONSERVATIVE_GLOBAL: GlobalBarrierDesc = GlobalBarrierDesc {
    sync_before: SYNC_ALL_SHADING,
    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT | SYNC_INDEX_INPUT,
    access_before: ACCESS_UNORDERED_ACCESS,
    access_after: ACCESS_COMMON,
};

/// Conservative global fallback when a barrier also involves `COPY` writes.
const CONSERVATIVE_GLOBAL_COPY: GlobalBarrierDesc = GlobalBarrierDesc {
    sync_before: SYNC_ALL_SHADING | SYNC_COPY,
    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT | SYNC_INDEX_INPUT | SYNC_COPY,
    access_before: ACCESS_UNORDERED_ACCESS | ACCESS_COPY_DEST,
    access_after: ACCESS_COMMON,
};

/// Checks whether `(sync, access)` is legal per D3D12 Enhanced Barriers rules.
fn sync_access_compatible(sync: u32, access: u32) -> bool {
    if sync == SYNC_NONE {
        return access == ACCESS_NO_ACCESS;
    }
    if (sync & SYNC_SPLIT) != 0 {
        return sync == SYNC_SPLIT && access == ACCESS_NO_ACCESS;
    }
    if (sync & !KNOWN_DIRECT_SYNC) != 0 {
        return false;
    }
    if access == ACCESS_NO_ACCESS {
        return true;
    }
    if (access & ACCESS_NO_ACCESS) != 0 || (access & !KNOWN_DIRECT_ACCESS) != 0 {
        return false;
    }
    if access == ACCESS_COMMON || (sync & SYNC_ALL) != 0 {
        return true;
    }
    const SHADING: u32 = SYNC_VERTEX_SHADING
        | SYNC_PIXEL_SHADING
        | SYNC_COMPUTE_SHADING
        | SYNC_RAYTRACING
        | SYNC_DRAW
        | SYNC_ALL_SHADING
        | SYNC_NON_PIXEL_SHADING;
    const VERTEX: u32 = SYNC_VERTEX_SHADING | SYNC_DRAW | SYNC_ALL_SHADING | SYNC_NON_PIXEL_SHADING;
    if (access & ACCESS_VERTEX_BUFFER) != 0 && (sync & VERTEX) == 0 {
        return false;
    }
    if (access & ACCESS_CONSTANT_BUFFER) != 0 && (sync & SHADING) == 0 {
        return false;
    }
    if (access & ACCESS_INDEX_BUFFER) != 0 && (sync & (SYNC_INDEX_INPUT | SYNC_DRAW)) == 0 {
        return false;
    }
    if (access & ACCESS_RENDER_TARGET) != 0 && (sync & (SYNC_RENDER_TARGET | SYNC_DRAW)) == 0 {
        return false;
    }
    if (access & ACCESS_UNORDERED_ACCESS) != 0
        && (sync
            & (SHADING
                | SYNC_CLEAR_UAV
                | SYNC_BUILD_RAYTRACING_AS
                | SYNC_EMIT_RAYTRACING_POSTBUILD))
            == 0
    {
        return false;
    }
    if (access & (ACCESS_DEPTH_STENCIL_WRITE | ACCESS_DEPTH_STENCIL_READ)) != 0
        && (sync & (SYNC_DEPTH_STENCIL | SYNC_DRAW)) == 0
    {
        return false;
    }
    if (access & ACCESS_SHADER_RESOURCE) != 0
        && (sync & (SHADING | SYNC_BUILD_RAYTRACING_AS)) == 0
    {
        return false;
    }
    if (access & ACCESS_STREAM_OUTPUT) != 0 && (sync & VERTEX) == 0 {
        return false;
    }
    if (access & ACCESS_INDIRECT_ARGUMENT) != 0 && (sync & SYNC_EXECUTE_INDIRECT) == 0 {
        return false;
    }
    if (access & (ACCESS_COPY_DEST | ACCESS_COPY_SOURCE)) != 0 && (sync & SYNC_COPY) == 0 {
        return false;
    }
    if (access & (ACCESS_RESOLVE_DEST | ACCESS_RESOLVE_SOURCE)) != 0 && (sync & SYNC_RESOLVE) == 0 {
        return false;
    }
    if (access & ACCESS_RTAS_READ) != 0
        && (sync
            & (SHADING
                | SYNC_BUILD_RAYTRACING_AS
                | SYNC_COPY_RAYTRACING_AS
                | SYNC_EMIT_RAYTRACING_POSTBUILD))
            == 0
    {
        return false;
    }
    if (access & ACCESS_RTAS_WRITE) != 0
        && (sync & (SYNC_BUILD_RAYTRACING_AS | SYNC_COPY_RAYTRACING_AS)) == 0
    {
        return false;
    }
    if (access & ACCESS_SHADING_RATE_SOURCE) != 0
        && (sync & (SYNC_PIXEL_SHADING | SYNC_DRAW | SYNC_ALL_SHADING)) == 0
    {
        return false;
    }
    true
}

/// Widens an indirect-consumer barrier so Victory's compute-based indirect
/// argument clamper (`COMPUTE_SHADING` + `SHADER_RESOURCE`) is also ordered.
fn widen_for_indirect_clamp(mut b: GlobalBarrierDesc) -> GlobalBarrierDesc {
    if (b.sync_after & SYNC_EXECUTE_INDIRECT) != 0
        || (b.access_after != ACCESS_NO_ACCESS && (b.access_after & ACCESS_INDIRECT_ARGUMENT) != 0)
    {
        b.sync_after |= SYNC_COMPUTE_SHADING;
        if b.access_after != ACCESS_COMMON && b.access_after != ACCESS_NO_ACCESS {
            b.access_after |= ACCESS_SHADER_RESOURCE;
        }
    }
    b
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodedBarrier {
    /// No GPU barrier needed.
    Skip,
    /// Validated narrow Enhanced Barrier sync/access parameters.
    Exact(GlobalBarrierDesc),
    /// Unrecognized flag, incompatible sync/access, or conservative fallback.
    Fallback(GlobalBarrierDesc),
}

fn decode_global_barrier(b: GlobalBarrierDesc) -> DecodedBarrier {
    if b.sync_before == SYNC_NONE || b.sync_after == SYNC_NONE {
        return DecodedBarrier::Skip;
    }
    // D3D12 Enhanced Barriers forbid `SYNC_SPLIT` on global barriers.
    if ((b.sync_before | b.sync_after) & SYNC_SPLIT) != 0
        || !sync_access_compatible(b.sync_before, b.access_before)
        || !sync_access_compatible(b.sync_after, b.access_after)
    {
        let touches_copy = ((b.sync_before | b.sync_after) & SYNC_COPY) != 0
            || (b.access_before != ACCESS_NO_ACCESS && (b.access_before & ACCESS_COPY_DEST) != 0);
        return DecodedBarrier::Fallback(if touches_copy {
            CONSERVATIVE_GLOBAL_COPY
        } else {
            CONSERVATIVE_GLOBAL
        });
    }
    DecodedBarrier::Exact(widen_for_indirect_clamp(b))
}

fn decode_buffer_barrier(b: &BufferBarrierDesc) -> DecodedBarrier {
    if b.sync_before == SYNC_NONE || b.sync_after == SYNC_NONE {
        return DecodedBarrier::Skip;
    }
    // D3D12 Enhanced Barriers forbid `SYNC_SPLIT` on buffer barriers.
    if ((b.sync_before | b.sync_after) & SYNC_SPLIT) != 0 {
        return DecodedBarrier::Fallback(CONSERVATIVE_GLOBAL);
    }
    let access_before_ok =
        b.access_before == ACCESS_NO_ACCESS || (b.access_before & !KNOWN_BUFFER_ACCESS) == 0;
    let access_after_ok =
        b.access_after == ACCESS_NO_ACCESS || (b.access_after & !KNOWN_BUFFER_ACCESS) == 0;
    if !access_before_ok
        || !access_after_ok
        || !sync_access_compatible(b.sync_before, b.access_before)
        || !sync_access_compatible(b.sync_after, b.access_after)
    {
        return DecodedBarrier::Fallback(CONSERVATIVE_GLOBAL);
    }
    DecodedBarrier::Exact(widen_for_indirect_clamp(GlobalBarrierDesc {
        sync_before: b.sync_before,
        sync_after: b.sync_after,
        access_before: b.access_before,
        access_after: b.access_after,
    }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextureBarrierPlan {
    /// Layout/state transition already handled by `HostList::transition` / `ensure_state`.
    Skip,
    /// Same-resource UAV ordering; emit resource-scoped legacy `UAV(host_or_twin)`.
    ScopedUav,
    /// Copy-to-copy write hazard on a texture.
    CopyToCopy,
    /// Unrecognized bits or invalid descriptor; fall back to conservative global barrier.
    Fallback,
}

fn decode_texture_barrier(b: &TextureBarrierDesc) -> TextureBarrierPlan {
    if b.sync_before == SYNC_NONE || b.sync_after == SYNC_NONE {
        return TextureBarrierPlan::Skip;
    }
    if !sync_access_compatible(b.sync_before, b.access_before)
        || !sync_access_compatible(b.sync_after, b.access_after)
    {
        return TextureBarrierPlan::Fallback;
    }
    if b.access_before == ACCESS_COMMON || (b.access_before & ACCESS_UNORDERED_ACCESS) != 0 {
        return TextureBarrierPlan::ScopedUav;
    }
    if b.access_before != ACCESS_NO_ACCESS
        && b.access_after != ACCESS_NO_ACCESS
        && (b.access_before & ACCESS_COPY_DEST) != 0
        && (b.access_after & ACCESS_COPY_DEST) != 0
    {
        return TextureBarrierPlan::CopyToCopy;
    }
    TextureBarrierPlan::Skip
}

/// Decoded action for console pipeline `sync(flags, control, address, size)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyncPlan {
    /// Both `flags` and `control` are zero: no GPU work to wait for.
    Skip,
    /// Emit an Enhanced `D3D12_GLOBAL_BARRIER` (or `D3D12_BUFFER_BARRIER` when
    /// a narrow address range resolves unambiguously to one shadow buffer).
    Barrier {
        desc: GlobalBarrierDesc,
        /// Whether this barrier flushes all shader-stage UAV writes globally
        /// (safe to call `list.mark_uav_clean()`).
        covers_all_uavs: bool,
        /// Whether this barrier can be skipped when `list.is_uav_clean()`.
        coalesce_if_uav_clean: bool,
    },
}

/// Console `sync()` bit constants observed in title command streams:
/// - `0x100000` / `0x200000` / `0x400000`: CS / PS / VS partial flushes (`0x700000` = all shaders).
/// - `0xc280` / `0xc380` / `0xc3c0`: RDNA2 `GCR_CNTL` GL2_WB|GL2_INV|GL2_US|GLV_INV (+GL1_INV, +GLK_INV).
/// - `control = 0x20000` (GCR acquire) / `0x80020000` (+ PFP sync for indirect/index reads).
fn decode_sync_shape(flags: u32, control: u32) -> SyncPlan {
    const BOP_CS: u32 = 0x0010_0000;
    const BOP_PS: u32 = 0x0020_0000;
    const BOP_VS: u32 = 0x0040_0000;
    const BOP_ALL_SHADERS: u32 = BOP_CS | BOP_PS | BOP_VS; // 0x0070_0000

    const GCR_GLV: u32 = 0x0000_c280;
    const GCR_GL1_GLV: u32 = 0x0000_c380;
    const GCR_GL1_GLV_GLK: u32 = 0x0000_c3c0;
    const CTRL_GCR: u32 = 0x0002_0000;
    const CTRL_GCR_PFP: u32 = 0x8002_0000;

    if flags == 0 && control == 0 {
        return SyncPlan::Skip;
    }
    if control == 0 && (flags & !BOP_ALL_SHADERS) == 0 {
        if flags == BOP_ALL_SHADERS {
            return SyncPlan::Barrier {
                desc: GlobalBarrierDesc {
                    sync_before: SYNC_ALL_SHADING,
                    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT,
                    access_before: ACCESS_UNORDERED_ACCESS,
                    access_after: ACCESS_COMMON,
                },
                covers_all_uavs: true,
                coalesce_if_uav_clean: true,
            };
        }
        let mut sync_before = 0u32;
        if (flags & BOP_CS) != 0 {
            sync_before |= SYNC_COMPUTE_SHADING;
        }
        if (flags & BOP_PS) != 0 {
            sync_before |= SYNC_PIXEL_SHADING;
        }
        if (flags & BOP_VS) != 0 {
            sync_before |= SYNC_VERTEX_SHADING;
        }
        return SyncPlan::Barrier {
            desc: GlobalBarrierDesc {
                sync_before,
                sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT,
                access_before: ACCESS_UNORDERED_ACCESS,
                access_after: ACCESS_COMMON,
            },
            covers_all_uavs: false,
            coalesce_if_uav_clean: true,
        };
    }
    match (flags, control) {
        (GCR_GLV, CTRL_GCR) | (GCR_GL1_GLV, CTRL_GCR) => SyncPlan::Barrier {
            desc: GlobalBarrierDesc {
                sync_before: SYNC_ALL_SHADING,
                sync_after: SYNC_ALL_SHADING,
                access_before: ACCESS_UNORDERED_ACCESS,
                access_after: ACCESS_UNORDERED_ACCESS | ACCESS_SHADER_RESOURCE | ACCESS_VERTEX_BUFFER,
            },
            covers_all_uavs: true,
            coalesce_if_uav_clean: true,
        },
        (GCR_GL1_GLV_GLK, CTRL_GCR_PFP) => SyncPlan::Barrier {
            desc: GlobalBarrierDesc {
                sync_before: SYNC_ALL_SHADING | SYNC_COPY,
                sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT | SYNC_INDEX_INPUT,
                access_before: ACCESS_UNORDERED_ACCESS | ACCESS_COPY_DEST,
                access_after: ACCESS_COMMON,
            },
            covers_all_uavs: true,
            coalesce_if_uav_clean: false,
        },
        // Any unrecognized (flags, control): conservative full-shading UAV fallback.
        _ => SyncPlan::Barrier {
            desc: CONSERVATIVE_GLOBAL,
            covers_all_uavs: true,
            coalesce_if_uav_clean: true,
        },
    }
}

/// Emits an Enhanced `D3D12_GLOBAL_BARRIER` on `list7`.
#[cfg(windows)]
unsafe fn emit_enhanced_global(
    list7: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList7,
    desc: GlobalBarrierDesc,
) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BARRIER_ACCESS, D3D12_BARRIER_GROUP, D3D12_BARRIER_GROUP_0, D3D12_BARRIER_SYNC,
        D3D12_BARRIER_TYPE_GLOBAL, D3D12_GLOBAL_BARRIER,
    };
    let barrier = [D3D12_GLOBAL_BARRIER {
        SyncBefore: D3D12_BARRIER_SYNC(desc.sync_before as i32),
        SyncAfter: D3D12_BARRIER_SYNC(desc.sync_after as i32),
        AccessBefore: D3D12_BARRIER_ACCESS(desc.access_before as i32),
        AccessAfter: D3D12_BARRIER_ACCESS(desc.access_after as i32),
    }];
    // SAFETY: caller guarantees an open list; `barrier` stays live for the call.
    unsafe {
        list7.Barrier(&[D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_GLOBAL,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pGlobalBarriers: barrier.as_ptr(),
            },
        }]);
    }
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. The host orders
/// nothing between dispatches by itself: what shaders wrote is waited for
/// here as at the title's barriers (the ray-tracing structure builder has
/// no other wait between its stages).
/// (Arguments may be what earlier shaders wrote from here on:
/// `State::synced`.)
extern "system" fn sync(this: This, flags: u32, control: u32, address: u64, size: u64) {
    #[cfg(windows)]
    {
        let plan = decode_sync_shape(flags, control);
        if matches!(plan, SyncPlan::Skip) {
            return;
        }
        with_state(this, |_, state| state.synced());
        if let Some(list) = host_list(this) {
            #[cfg(feature = "diagnostics")]
            {
                // The shapes of the title's calls, each once.
                static SEEN: std::sync::Mutex<Vec<(u32, u32, bool)>> = std::sync::Mutex::new(Vec::new());
                let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                let shape = (flags, control, address == 0);
                if seen.len() < 64 && !seen.contains(&shape) {
                    seen.push(shape);
                    tracing::info!(
                        flags = format_args!("{flags:#x}"),
                        control = format_args!("{control:#x}"),
                        address = format_args!("{address:#x}"),
                        size = format_args!("{size:#x}"),
                        "a pipeline sync"
                    );
                }
            }
            if enhanced_groups_enabled(list)
                && let Ok(list7) = windows::core::Interface::cast::<
                    windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList7,
                >(&list.list)
            {
                if let SyncPlan::Barrier {
                    desc,
                    covers_all_uavs,
                    coalesce_if_uav_clean,
                } = plan
                {
                    if coalesce_if_uav_clean
                        && crate::host::coalesce_uav_barriers()
                        && list.is_uav_clean()
                    {
                        return;
                    }
                    // When a narrow non-zero range resolves unambiguously to one shadow
                    // buffer, emit a resource-scoped D3D12_BUFFER_BARRIER instead of global.
                    if address != 0
                        && size != 0
                        && size < 0x0001_0000_0000_0000
                        && let Some(shadow) = crate::shadow::scoped_barrier_shadow(address, size)
                    {
                        use windows::Win32::Graphics::Direct3D12::{
                            D3D12_BARRIER_ACCESS, D3D12_BARRIER_GROUP, D3D12_BARRIER_GROUP_0,
                            D3D12_BARRIER_SYNC, D3D12_BARRIER_TYPE_BUFFER, D3D12_BUFFER_BARRIER,
                        };
                        let buf_barrier = [D3D12_BUFFER_BARRIER {
                            SyncBefore: D3D12_BARRIER_SYNC(desc.sync_before as i32),
                            SyncAfter: D3D12_BARRIER_SYNC(desc.sync_after as i32),
                            AccessBefore: D3D12_BARRIER_ACCESS(desc.access_before as i32),
                            AccessAfter: D3D12_BARRIER_ACCESS(desc.access_after as i32),
                            // SAFETY: borrowed COM pointer; `keep_shadow` keeps it alive.
                            pResource: unsafe { std::mem::transmute_copy(&shadow.buffer) },
                            Offset: 0,
                            Size: u64::MAX,
                        }];
                        list.keep_shadow(shadow);
                        // SAFETY: open list and live shadow buffer.
                        unsafe {
                            list7.Barrier(&[D3D12_BARRIER_GROUP {
                                Type: D3D12_BARRIER_TYPE_BUFFER,
                                NumBarriers: 1,
                                Anonymous: D3D12_BARRIER_GROUP_0 {
                                    pBufferBarriers: buf_barrier.as_ptr(),
                                },
                            }]);
                        }
                        return;
                    }
                    // SAFETY: open list.
                    unsafe { emit_enhanced_global(&list7, desc) };
                    if covers_all_uavs {
                        list.mark_uav_clean();
                    }
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
                }
                return;
            }
            if list.emit_global_uav_barrier() {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
            }
        }
    }
    let _ = (this, flags, control, address, size);
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

#[cfg(windows)]
use crate::upload::Upload;

/// The memory of [`allocate_upload`] where no host GPU reads it by address.
#[cfg(not(windows))]
struct Upload(Vec<u8>);

#[cfg(not(windows))]
impl Upload {
    fn new(len: usize) -> Option<Upload> {
        Some(Upload(vec![0; len]))
    }
    fn start(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. The title's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(mut block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth order and shading rate hints): any arguments,
/// nothing returned.
extern "system" fn ignored_state(_: This) {}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        // (Through the draw state, which keeps them for the host list
        // that continues this one after a cut: `State::replay`.)
        with_state(this, |list, s| s.set_viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.set_scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        with_state(this, |list, s| {
            // SAFETY: the title passes four floats or null.
            s.set_blend_factor(list, unsafe { factor.as_ref() })
        });
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.set_stencil_ref(list, reference));
    }
    pub(super) extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
        with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let mut buf = [0u64; 8];
        let n = if handles.is_null() { 0 } else { (count as usize).min(8) };
        if one_range != 0 && n > 0 {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            for (i, h) in buf[..n].iter_mut().enumerate() {
                *h = first + 128 * i as u64;
            }
        } else if n > 0 {
            buf[..n].copy_from_slice(items(handles, n as u32));
        }
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &buf[..n], depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z, 0));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// When `indirect != 0`, routes through native `ExecuteIndirect` with a
    /// 1D clamped mesh-dispatch record (`indirect::KIND_DISPATCH_MESH_1D`).
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            with_state(this, |list, s| s.dispatch_mesh_indirect(list, indirect));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as depth_bounds;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn execute_indirect_with(
    list: &crate::host::HostList,
    state: &mut crate::draw::State,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let ran = if let Some(address) = address
        && !args.is_empty()
        && (count.is_null() || counter.is_some())
    {
        state.execute_indirect_signature(list, stride, args, max, address, counter)
    } else {
        #[cfg(feature = "diagnostics")]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
        false
    };
    if ran {
        return;
    }
    record_dropped_indirect(signature, stride, args, max, address, !count.is_null());
}

fn record_dropped_indirect(
    signature: This,
    stride: u32,
    args: &[[u32; 4]],
    max: u32,
    address: Option<u64>,
    counted: bool,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    INDIRECT.fetch_add(1, Ordering::Relaxed);
    let sig = signature as usize;
    static FAST_SEEN: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
    if sig != 0 && FAST_SEEN.iter().any(|slot| slot.load(Ordering::Relaxed) == sig) {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&sig) {
        if sig != 0 && seen.len() < FAST_SEEN.len() {
            FAST_SEEN[seen.len()].store(sig, Ordering::Relaxed);
        }
        seen.push(sig);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted,
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Supports both single-argument signatures (draw, indexed draw,
/// dispatch, single-pass dispatch type 105, mesh dispatch) and multi-argument
/// signatures with root constants / root CBV/SRV/UAV views preceding the draw
/// or dispatch (`State::execute_indirect_signature`).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    #[cfg(windows)]
    {
        if host_list(this).is_some() {
            with_state(this, |list, s| {
                execute_indirect_with(list, s, signature, max, arguments, offset, count, count_offset);
            });
            return;
        }
    }
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    record_dropped_indirect(signature, stride, args, max, address, !count.is_null());
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// A GPU wait for a memory value: the list's commands after it run when
/// the 32 or 64 bits at `address` compare as asked with `reference`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValueWait {
    address: u64,
    comparison: u32,
    reference: u64,
    wide: bool,
}

impl ValueWait {
    /// Whether the memory holds what is waited for now.
    /// shortcut: the comparison codes are taken to be the comparison
    /// functions' (1 never .. 8 always, value against reference); one not
    /// known counts as met, and so does memory that is not mapped.
    ///
    /// The value is read through the kernel, which reports memory that is
    /// not there instead of faulting (as the copies of title memory are
    /// read, `crate::host::copy_mapped`). Asking Windows first whether the
    /// memory is committed (`VirtualQuery`) walks the page tables to the
    /// end of the region: the title keeps these values in a 1.1 GB
    /// allocation (0xd00000000), where one question took about 8 ms, and a
    /// queue asks at every submission and every signal, for each queue that
    /// stands at a wait (the title has five queues). That was
    /// 1.0-2.2 s of every cutscene frame on the thread that submits (runs
    /// 36-49: "between its submissions ... kernel"; 47 of 110 looks at
    /// that thread in run 49 found it in this question).
    /// `VICTORY_WAIT_ASK=1` asks first, as before.
    pub(crate) fn met(&self) -> bool {
        #[cfg(all(windows, feature = "diagnostics"))]
        let started = std::time::Instant::now();
        let met = self.holds();
        #[cfg(all(windows, feature = "diagnostics"))]
        {
            use crate::copystat::{C, add};
            add(C::WaitChecks, 1);
            add(C::WaitCheckNs, started.elapsed().as_nanos() as u64);
        }
        met
    }

    fn holds(&self) -> bool {
        let bytes = if self.wide { 8 } else { 4 };
        if self.address == 0 {
            return true;
        }
        static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ASK.get_or_init(|| std::env::var_os("VICTORY_WAIT_ASK").is_some_and(|v| v == "1"))
            && !crate::texture::committed(self.address as *const u8, bytes)
        {
            return true;
        }
        // The low `bytes` bytes; the rest stays zero.
        let mut raw = [0u8; 8];
        if !crate::host::read_into(self.address as *const u8, &mut raw[..bytes]) {
            return true;
        }
        self.holds_for(u64::from_le_bytes(raw))
    }

    /// Where the value waited for is: address and bytes.
    #[cfg(windows)]
    pub(crate) fn place(&self) -> (usize, usize) {
        (self.address as usize, if self.wide { 8 } else { 4 })
    }

    /// Whether the wait is met with `value` at its address (its low half
    /// for a 32-bit wait).
    pub(crate) fn holds_for(&self, value: u64) -> bool {
        let (value, reference) = if self.wide {
            (value, self.reference)
        } else {
            (value & 0xFFFF_FFFF, self.reference & 0xFFFF_FFFF)
        };
        match self.comparison {
            1 => false,
            2 => value < reference,
            3 => value == reference,
            4 => value <= reference,
            5 => value > reference,
            6 => value != reference,
            7 => value >= reference,
            _ => true,
        }
    }
}

/// The waits lists recorded since their last reset, by list.
static VALUE_WAITS: std::sync::Mutex<Vec<(usize, ValueWait)>> = std::sync::Mutex::new(Vec::new());
static VALUE_WAITS_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The waits of the submitted `lists` that are not met yet (taken from
/// them): what the console's GPU would stop at. The title orders work
/// between its queues this way (the title: the compute queue's
/// lighting passes wait for a value the graphics queue's depth pass writes
/// at the bottom of the pipe; run at once, they read a cleared depth
/// buffer and nothing was lit).
/// shortcut: a wait holds its whole list (and submission) back, not only
/// the commands after it.
pub(crate) fn unmet_waits(lists: &[This]) -> Vec<ValueWait> {
    if VALUE_WAITS_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut unmet = Vec::new();
    waits.retain(|&(list, wait)| {
        if !lists.iter().any(|&l| l as usize == list) {
            return true;
        }
        if !wait.met() {
            unmet.push(wait);
        }
        false
    });
    VALUE_WAITS_COUNT.store(waits.len(), std::sync::atomic::Ordering::Release);
    unmet
}

/// The waits each of the submitted `lists` recorded (taken from them),
/// met or not, at the list's first place in the submission.
#[cfg(windows)]
pub(crate) fn waits_by_list(lists: &[This]) -> Vec<Vec<ValueWait>> {
    if VALUE_WAITS_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return vec![Vec::new(); lists.len()];
    }
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut by = vec![Vec::new(); lists.len()];
    waits.retain(|&(list, wait)| {
        match lists.iter().position(|&l| l as usize == list) {
            Some(i) => {
                by[i].push(wait);
                false
            }
            None => true,
        }
    });
    VALUE_WAITS_COUNT.store(waits.len(), std::sync::atomic::Ordering::Release);
    by
}

fn wait_for(this: This, address: u64, comparison: u32, reference: u64, wide: bool) {
    let wait = ValueWait {
        address,
        comparison,
        reference,
        wide,
    };
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
        tracing::info!(
            address = format_args!("{address:#x}"),
            comparison,
            reference,
            wide,
            met = wait.met(),
            "a list waits for a memory value"
        );
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(host) = host_list(this) {
        crate::capture::op(std::ptr::from_ref(host) as usize, || {
            format!(
                "WAIT VALUE at {address:#x} comparison {comparison} reference {reference:#x} wide {wide} met {}",
                wait.met()
            )
        });
    }
    // The list is cut here, and the wait stands before what follows.
    #[cfg(windows)]
    if crate::queue::cuts() {
        cut(this, wait);
        return;
    }
    let mut waits = VALUE_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    waits.push((this as usize, wait));
    VALUE_WAITS_COUNT.store(waits.len(), std::sync::atomic::Ordering::Release);
}
extern "system" fn wait_value(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, false);
}
extern "system" fn wait_value64(this: This, address: u64, comparison: u32, reference: u64) {
    wait_for(this, address, comparison, reference, true);
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. When Enhanced Barriers are
/// supported and `VICTORY_ENHANCED_GROUPS != 0`, parses the console barrier
/// groups into targeted `D3D12_BUFFER_BARRIER` / narrowed `D3D12_GLOBAL_BARRIER`
/// / resource-scoped texture UAV barriers, falling back conservatively on any
/// unknown flag or unresolved graphics/copy dependency.
extern "system" fn barrier_groups(this: This, count: u32, groups: *const u8) {
    // Diagnostics: left out under `VICTORY_BARRIER_TEST=groups`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if count > 0 && crate::copystat::barrier_test() & 1 != 0 {
        crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
        return;
    }
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        with_state(this, |_, state| state.synced());
        if enhanced_groups_enabled(list)
            && let Ok(list7) = windows::core::Interface::cast::<
                windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList7,
            >(&list.list)
        {
            emit_parsed_barrier_groups(list, &list7, count, groups);
            return;
        }
        if list.emit_global_uav_barrier() {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, count, groups);
}

#[derive(Debug, PartialEq, Eq)]
enum ResolvedTexUav<T> {
    Uav(T),
    Skip,
    Fallback,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedBarrierPlan<B, T> {
    buf_barriers: Vec<(B, GlobalBarrierDesc)>,
    global_barriers: Vec<GlobalBarrierDesc>,
    tex_uavs: Vec<T>,
    fallback: Option<(GlobalBarrierDesc, bool)>,
}

/// Pure planner for console `BarrierGroup` arrays. Preserves every narrow sibling
/// buffer/global/texture barrier even when one item in a mixed batch requires a
/// conservative fallback, instead of discarding siblings.
fn plan_parsed_barrier_groups<B, T: PartialEq>(
    groups: &[BarrierGroup],
    mut resolve_buf: impl FnMut(&BufferBarrierDesc) -> Option<B>,
    mut resolve_tex: impl FnMut(&TextureBarrierDesc) -> ResolvedTexUav<T>,
) -> ParsedBarrierPlan<B, T> {
    let mut plan = ParsedBarrierPlan {
        buf_barriers: Vec::new(),
        global_barriers: Vec::new(),
        tex_uavs: Vec::new(),
        fallback: None,
    };

    let mut set_fallback = |desc: GlobalBarrierDesc, copy: bool| {
        let now = match plan.fallback {
            Some((_, true)) => (CONSERVATIVE_GLOBAL_COPY, true),
            _ if copy => (CONSERVATIVE_GLOBAL_COPY, true),
            Some((prev, false)) => (prev, false),
            None => (desc, false),
        };
        plan.fallback = Some(now);
    };

    for g in groups {
        if g.count == 0 {
            continue;
        }
        if g.barriers.is_null() {
            set_fallback(CONSERVATIVE_GLOBAL, false);
            continue;
        }
        match g.kind {
            BARRIER_GROUP_GLOBAL => {
                // SAFETY: caller guarantees `g.count` valid `GlobalBarrierDesc` entries.
                let items = unsafe {
                    std::slice::from_raw_parts(
                        g.barriers.cast::<GlobalBarrierDesc>(),
                        g.count as usize,
                    )
                };
                for &b in items {
                    match decode_global_barrier(b) {
                        DecodedBarrier::Skip => {}
                        DecodedBarrier::Exact(desc) => plan.global_barriers.push(desc),
                        DecodedBarrier::Fallback(desc) => {
                            set_fallback(desc, (desc.sync_before & SYNC_COPY) != 0);
                        }
                    }
                }
            }
            BARRIER_GROUP_BUFFER => {
                // SAFETY: caller guarantees `g.count` valid `BufferBarrierDesc` entries.
                let items = unsafe {
                    std::slice::from_raw_parts(
                        g.barriers.cast::<BufferBarrierDesc>(),
                        g.count as usize,
                    )
                };
                for b in items {
                    let desc = match decode_buffer_barrier(b) {
                        DecodedBarrier::Skip => continue,
                        DecodedBarrier::Exact(d) => d,
                        DecodedBarrier::Fallback(d) => {
                            set_fallback(d, false);
                            continue;
                        }
                    };
                    if let Some(handle) = resolve_buf(b) {
                        plan.buf_barriers.push((handle, desc));
                    } else {
                        // Aliased or unmapped buffer: preserve exact narrowed sync/access
                        // as a global barrier when valid, else record conservative fallback.
                        match decode_global_barrier(desc) {
                            DecodedBarrier::Skip => {}
                            DecodedBarrier::Exact(gdesc) => plan.global_barriers.push(gdesc),
                            DecodedBarrier::Fallback(gdesc) => {
                                set_fallback(gdesc, (gdesc.sync_before & SYNC_COPY) != 0);
                            }
                        }
                    }
                }
            }
            BARRIER_GROUP_TEXTURE => {
                // SAFETY: caller guarantees `g.count` valid `TextureBarrierDesc` entries.
                let items = unsafe {
                    std::slice::from_raw_parts(
                        g.barriers.cast::<TextureBarrierDesc>(),
                        g.count as usize,
                    )
                };
                for b in items {
                    match decode_texture_barrier(b) {
                        TextureBarrierPlan::Skip => {}
                        TextureBarrierPlan::CopyToCopy => {
                            plan.global_barriers.push(GlobalBarrierDesc {
                                sync_before: SYNC_COPY,
                                sync_after: SYNC_COPY,
                                access_before: ACCESS_COPY_DEST,
                                access_after: ACCESS_COPY_DEST,
                            });
                        }
                        TextureBarrierPlan::ScopedUav => match resolve_tex(b) {
                            ResolvedTexUav::Uav(handle) => {
                                if !plan.tex_uavs.contains(&handle) {
                                    plan.tex_uavs.push(handle);
                                }
                            }
                            ResolvedTexUav::Skip => {}
                            ResolvedTexUav::Fallback => {
                                set_fallback(CONSERVATIVE_GLOBAL, false);
                            }
                        },
                        TextureBarrierPlan::Fallback => {
                            set_fallback(CONSERVATIVE_GLOBAL, false);
                        }
                    }
                }
            }
            _ => {
                set_fallback(CONSERVATIVE_GLOBAL, false);
            }
        }
    }

    plan
}

/// Parses console `BarrierGroup` descriptors and emits the narrowest safe
/// Enhanced Barrier set on `list7` (plus resource-scoped legacy texture UAV
/// barriers where required). An undecoded item in a mixed batch appends a
/// conservative global barrier without discarding narrow sibling barriers.
#[cfg(windows)]
fn emit_parsed_barrier_groups(
    list: &crate::host::HostList,
    list7: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList7,
    count: u32,
    groups_ptr: *const u8,
) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BARRIER_ACCESS, D3D12_BARRIER_GROUP, D3D12_BARRIER_GROUP_0, D3D12_BARRIER_SYNC,
        D3D12_BARRIER_TYPE_BUFFER, D3D12_BARRIER_TYPE_GLOBAL, D3D12_BUFFER_BARRIER,
        D3D12_GLOBAL_BARRIER, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
        D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_UAV,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_UAV_BARRIER,
    };
    use windows::core::Interface;

    let emit_fallback = |desc: GlobalBarrierDesc, covers_copy: bool| {
        if !covers_copy && crate::host::coalesce_uav_barriers() && list.is_uav_clean() {
            return;
        }
        // SAFETY: open list.
        unsafe { emit_enhanced_global(list7, desc) };
        list.mark_uav_clean();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
    };

    if groups_ptr.is_null() {
        emit_fallback(CONSERVATIVE_GLOBAL, false);
        return;
    }

    // SAFETY: caller passes `count` valid `BarrierGroup` entries.
    let groups =
        unsafe { std::slice::from_raw_parts(groups_ptr.cast::<BarrierGroup>(), count as usize) };

    let plan = plan_parsed_barrier_groups(
        groups,
        |b| {
            if b.resource.is_null() {
                return None;
            }
            crate::resource::buffer_bytes(b.resource).and_then(|(ptr, buf_len)| {
                let (addr, len) = if b.offset == 0 && (b.size == u64::MAX || b.size >= buf_len) {
                    (ptr as u64, buf_len)
                } else {
                    let avail = buf_len.saturating_sub(b.offset);
                    let span = if b.size == u64::MAX { avail } else { b.size.min(avail) };
                    if span > 0 {
                        ((ptr as u64).wrapping_add(b.offset), span)
                    } else {
                        (ptr as u64, buf_len)
                    }
                };
                crate::shadow::scoped_barrier_shadow(addr, len)
            })
        },
        |b| {
            if b.resource.is_null() {
                return ResolvedTexUav::Fallback;
            }
            if let Some(twin) = crate::resource::twin_resource(b.resource) {
                list.keep(twin);
                ResolvedTexUav::Uav(twin.as_raw())
            } else if let Some(copy) = crate::resource::host_copy(b.resource) {
                if (copy.flags() & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32) != 0 {
                    list.keep(&copy.texture);
                    ResolvedTexUav::Uav(copy.texture.as_raw())
                } else if (b.access_before & ACCESS_UNORDERED_ACCESS) != 0 {
                    ResolvedTexUav::Fallback
                } else {
                    ResolvedTexUav::Skip
                }
            } else {
                ResolvedTexUav::Fallback
            }
        },
    );

    if let Some((desc, covers_copy)) = plan.fallback
        && plan.buf_barriers.is_empty()
        && plan.global_barriers.is_empty()
        && plan.tex_uavs.is_empty()
    {
        emit_fallback(desc, covers_copy);
        return;
    }

    let mut buf_barriers: Vec<D3D12_BUFFER_BARRIER> = Vec::with_capacity(plan.buf_barriers.len());
    for (shadow, desc) in plan.buf_barriers {
        buf_barriers.push(D3D12_BUFFER_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC(desc.sync_before as i32),
            SyncAfter: D3D12_BARRIER_SYNC(desc.sync_after as i32),
            AccessBefore: D3D12_BARRIER_ACCESS(desc.access_before as i32),
            AccessAfter: D3D12_BARRIER_ACCESS(desc.access_after as i32),
            // SAFETY: borrowed COM pointer; `keep_shadow` keeps it alive.
            pResource: unsafe { std::mem::transmute_copy(&shadow.buffer) },
            Offset: 0,
            Size: u64::MAX,
        });
        list.keep_shadow(shadow);
    }

    let mut global_barriers: Vec<D3D12_GLOBAL_BARRIER> =
        Vec::with_capacity(plan.global_barriers.len() + usize::from(plan.fallback.is_some()));
    for desc in plan.global_barriers {
        global_barriers.push(D3D12_GLOBAL_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC(desc.sync_before as i32),
            SyncAfter: D3D12_BARRIER_SYNC(desc.sync_after as i32),
            AccessBefore: D3D12_BARRIER_ACCESS(desc.access_before as i32),
            AccessAfter: D3D12_BARRIER_ACCESS(desc.access_after as i32),
        });
    }
    let mut mark_clean = false;
    if let Some((desc, covers_copy)) = plan.fallback
        && (covers_copy || !crate::host::coalesce_uav_barriers() || !list.is_uav_clean())
    {
        global_barriers.push(D3D12_GLOBAL_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC(desc.sync_before as i32),
            SyncAfter: D3D12_BARRIER_SYNC(desc.sync_after as i32),
            AccessBefore: D3D12_BARRIER_ACCESS(desc.access_before as i32),
            AccessAfter: D3D12_BARRIER_ACCESS(desc.access_after as i32),
        });
        mark_clean = true;
    }

    let mut out_groups: [std::mem::MaybeUninit<D3D12_BARRIER_GROUP>; 2] =
        [const { std::mem::MaybeUninit::uninit() }; 2];
    let mut out_len = 0usize;
    if !buf_barriers.is_empty() {
        out_groups[out_len].write(D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: buf_barriers.len() as u32,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: buf_barriers.as_ptr(),
            },
        });
        out_len += 1;
    }
    if !global_barriers.is_empty() {
        out_groups[out_len].write(D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_GLOBAL,
            NumBarriers: global_barriers.len() as u32,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pGlobalBarriers: global_barriers.as_ptr(),
            },
        });
        out_len += 1;
    }
    if out_len > 0 {
        // SAFETY: `out_groups[..out_len]` are initialized and all barrier slices stay live.
        unsafe {
            let slice = std::slice::from_raw_parts(out_groups.as_ptr().cast(), out_len);
            list7.Barrier(slice);
        }
        if mark_clean {
            list.mark_uav_clean();
        }
        #[cfg(feature = "diagnostics")]
        if !global_barriers.is_empty() {
            crate::copystat::add(crate::copystat::C::WaitsForGroups, 1);
        }
    }
    if !plan.tex_uavs.is_empty() {
        let uavs: Vec<D3D12_RESOURCE_BARRIER> = plan
            .tex_uavs
            .into_iter()
            .map(|raw| D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        // SAFETY: borrowed COM pointer to a live host texture/twin.
                        pResource: unsafe { std::mem::transmute_copy(&raw) },
                    }),
                },
            })
            .collect();
        // SAFETY: open list and live host textures/twins.
        unsafe { list.list.ResourceBarrier(&uavs) };
    }
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
    value_written(this, address, u64::from(value));
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
    if let Written::Value(value) = what {
        value_written(this, address, value);
    }
}

/// The title wrote a value at the bottom of the pipe (not a timestamp, not
/// a query result): listed (diagnostics), and with `VICTORY_CUT_WRITES=1`
/// the host list is cut right behind it, so that the value is stored when
/// the card has run the list up to there, as on the console, and not when
/// the whole list has run. Lists of other queues wait for these values
/// (the title: the compute list that builds the depth pyramid
/// waits for one, and the meshlet culling of the same frame reads that
/// pyramid). Off by default: every such write then ends a host list, and
/// each part is a submission of its own.
fn value_written(this: This, address: u64, value: u64) {
    #[cfg(windows)]
    {
        if address == 0 || this.is_null() {
            return;
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = value;
        #[cfg(feature = "diagnostics")]
        if let Some(host) = host_list(this) {
            crate::capture::op(std::ptr::from_ref(host) as usize, || {
                format!("WRITE VALUE at {address:#x} value {value:#x}")
            });
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WRITES").is_some_and(|v| v == "1"))
            || !crate::queue::cuts()
        {
            return;
        }
        // SAFETY: the title passes one of our lists; a list is recorded by
        // one thread at a time and is not submitted while it records.
        let list = unsafe { &mut *this.cast_mut().cast::<List>() };
        let Some(current) = list.host.as_ref() else {
            return;
        };
        let Some(fresh) = fresh_list() else {
            return;
        };
        current.close();
        if let Some(done) = list.host.replace(fresh) {
            // The part so far keeps its waits; nothing stands before the
            // next one but this part itself.
            let waits = std::mem::take(&mut list.gate);
            list.earlier.push(Part { waits, host: done });
        }
        if let Some(host) = list.host.as_ref() {
            list.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .replay(host);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, address, value);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, state::depth_bounds as *const ()),       // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait_value as *const ()),                // 0x238, 32 bits
    (72, wait_value64 as *const ()),              // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 10] = [
    (28, begin_query as *const ()),            // 0xe0 BeginQuery
    (29, end_query as *const ()),              // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0 ResolveQueryData
    (31, set_predication as *const ()),        // 0xf8 SetPredication
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// The title's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from the title's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));
/// Of lists whose calls are written down.
#[cfg(windows)]
static THIN_FUNCTIONS: Table<160> = Table(thin::table(false));
#[cfg(windows)]
static NEWER_THIN_FUNCTIONS: Table<160> = Table(thin::table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as the title asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_rewritten_back_texture_after_reuse_is_read_back() {
        use crate::object::guid;
        use crate::resource::{create_committed, readback_test_descriptions};
        let Some(_) = crate::host::host() else {
            return;
        };
        let _test_kinds = ResetTestKinds(crate::texback::TEST_KINDS.replace(Some(&[(256, 256, 41)])));
        struct ResetTestKinds(Option<&'static [(u64, u32, u32)]>);
        impl Drop for ResetTestKinds {
            fn drop(&mut self) {
                crate::texback::TEST_KINDS.set(self.0);
            }
        }
        let null = std::ptr::null();
        let (heap, desc) = readback_test_descriptions();
        let resource = guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut tex = null;
        assert_eq!(
            create_committed(null, &heap, 0, &desc, 0, null.cast(), &resource, &mut tex),
            S_OK
        );
        let (_, at, _) = crate::resource::title_texture(tex).expect("synthetic title texture");
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let mut allocator = null;
        assert_eq!(create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator), S_OK);
        let mut list = null;
        assert_eq!(create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list), S_OK);
        let host = host_list(list).expect("a host list");
        host.texture_written(tex);
        assert!(host.back_wanted_fresh(tex), "the write registered a read-back");
        crate::texback::replaced(at);
        assert!(!host.back_wanted_fresh(tex), "the reuse invalidated the read-back");
        host.texture_written(tex);
        assert!(host.back_wanted_fresh(tex), "the second write re-registered the read-back");
        host.close();
        assert_eq!(host.texture_back_count(), 1, "the rewritten texture was read back");
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
        assert_eq!(release(tex), 0);
    }

    #[test]
    fn fold_transition_coalesces_duplicates_and_drops_noops() {
        let mut buf = [(0usize, 0u32, 0u32, 0u8); 8];
        let mut len = 0;
        // Distinct resources are preserved in order.
        fold_transition(&mut buf, &mut len, 1, 0x4, 0x400, 1);
        fold_transition(&mut buf, &mut len, 2, 0x80, 0x800, 2);
        // Same-state transition is ignored.
        fold_transition(&mut buf, &mut len, 3, 0x8, 0x8, 3);
        assert_eq!(len, 2);
        assert_eq!(&buf[..len], &[(1, 0x4, 0x400, 1), (2, 0x80, 0x800, 2)]);
        // A -> B followed by B -> C folds into A -> C.
        fold_transition(&mut buf, &mut len, 1, 0x400, 0x80, 1);
        assert_eq!(len, 2);
        assert_eq!(&buf[..len], &[(1, 0x4, 0x80, 1), (2, 0x80, 0x800, 2)]);
        // A -> C followed by C -> A cancels out completely.
        fold_transition(&mut buf, &mut len, 1, 0x80, 0x4, 1);
        assert_eq!(len, 1);
        assert_eq!(&buf[..len], &[(2, 0x80, 0x800, 2)]);
    }

    #[test]
    fn enhanced_groups_env_default_on_and_zero_disables() {
        assert!(enhanced_groups_wanted_for(None));
        assert!(enhanced_groups_wanted_for(Some(std::ffi::OsStr::new("1"))));
        assert!(!enhanced_groups_wanted_for(Some(std::ffi::OsStr::new("0"))));
    }

    #[test]
    fn decode_sync_known_five_shapes_and_unknown_fallback() {
        assert_eq!(decode_sync_shape(0, 0), SyncPlan::Skip);

        // 1. Compute-only partial flush (0x100000 / 0x0) avoids draining graphics/PS/VS.
        assert_eq!(
            decode_sync_shape(0x100000, 0),
            SyncPlan::Barrier {
                desc: GlobalBarrierDesc {
                    sync_before: SYNC_COMPUTE_SHADING,
                    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT,
                    access_before: ACCESS_UNORDERED_ACCESS,
                    access_after: ACCESS_COMMON,
                },
                covers_all_uavs: false,
                coalesce_if_uav_clean: true,
            }
        );

        // 2. All-shader partial flush (0x700000 / 0x0).
        assert_eq!(
            decode_sync_shape(0x700000, 0),
            SyncPlan::Barrier {
                desc: GlobalBarrierDesc {
                    sync_before: SYNC_ALL_SHADING,
                    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT,
                    access_before: ACCESS_UNORDERED_ACCESS,
                    access_after: ACCESS_COMMON,
                },
                covers_all_uavs: true,
                coalesce_if_uav_clean: true,
            }
        );

        // 3 & 4. RDNA2 GCR GLV and GL1+GLV shader-cache flushes (no PFP sync):
        // flush all shader UAVs globally (`covers_all_uavs: true`) to unlock
        // back-to-back coalescing.
        let expected_gcr = SyncPlan::Barrier {
            desc: GlobalBarrierDesc {
                sync_before: SYNC_ALL_SHADING,
                sync_after: SYNC_ALL_SHADING,
                access_before: ACCESS_UNORDERED_ACCESS,
                access_after: ACCESS_UNORDERED_ACCESS | ACCESS_SHADER_RESOURCE | ACCESS_VERTEX_BUFFER,
            },
            covers_all_uavs: true,
            coalesce_if_uav_clean: true,
        };
        assert_eq!(decode_sync_shape(0xc280, 0x20000), expected_gcr);
        assert_eq!(decode_sync_shape(0xc380, 0x20000), expected_gcr);

        // 5. RDNA2 GCR full shader-cache + GLK + PFP sync (0xc3c0 / 0x80020000).
        assert_eq!(
            decode_sync_shape(0xc3c0, 0x80020000),
            SyncPlan::Barrier {
                desc: GlobalBarrierDesc {
                    sync_before: SYNC_ALL_SHADING | SYNC_COPY,
                    sync_after: SYNC_ALL_SHADING | SYNC_EXECUTE_INDIRECT | SYNC_INDEX_INPUT,
                    access_before: ACCESS_UNORDERED_ACCESS | ACCESS_COPY_DEST,
                    access_after: ACCESS_COMMON,
                },
                covers_all_uavs: true,
                coalesce_if_uav_clean: false,
            }
        );

        // Unknown flags/controls must fall back conservatively.
        let conservative = SyncPlan::Barrier {
            desc: CONSERVATIVE_GLOBAL,
            covers_all_uavs: true,
            coalesce_if_uav_clean: true,
        };
        assert_eq!(decode_sync_shape(0xdead_beef, 0), conservative);
        assert_eq!(decode_sync_shape(0x100000, 0x1), conservative);
        assert_eq!(decode_sync_shape(0x800000, 0), conservative);
        assert_eq!(decode_sync_shape(0xc3c0, 0x20000), conservative);
    }

    #[test]
    fn decode_buffer_and_global_barriers_narrow_known_and_fallback_unknown() {
        // Compute -> compute UAV buffer barrier stays narrowed.
        let cs_buf = BufferBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_UNORDERED_ACCESS | ACCESS_SHADER_RESOURCE,
            resource: std::ptr::null(),
            offset: 0,
            size: u64::MAX,
        };
        assert_eq!(
            decode_buffer_barrier(&cs_buf),
            DecodedBarrier::Exact(GlobalBarrierDesc {
                sync_before: SYNC_COMPUTE_SHADING,
                sync_after: SYNC_COMPUTE_SHADING,
                access_before: ACCESS_UNORDERED_ACCESS,
                access_after: ACCESS_UNORDERED_ACCESS | ACCESS_SHADER_RESOURCE,
            })
        );

        // Compute -> ExecuteIndirect widens for Victory's compute indirect clamper.
        let ind_buf = BufferBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_EXECUTE_INDIRECT,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_INDIRECT_ARGUMENT,
            resource: std::ptr::null(),
            offset: 0,
            size: u64::MAX,
        };
        assert_eq!(
            decode_buffer_barrier(&ind_buf),
            DecodedBarrier::Exact(GlobalBarrierDesc {
                sync_before: SYNC_COMPUTE_SHADING,
                sync_after: SYNC_EXECUTE_INDIRECT | SYNC_COMPUTE_SHADING,
                access_before: ACCESS_UNORDERED_ACCESS,
                access_after: ACCESS_INDIRECT_ARGUMENT | ACCESS_SHADER_RESOURCE,
            })
        );

        // SYNC_NONE is a no-op.
        let noop_buf = BufferBarrierDesc {
            sync_before: SYNC_NONE,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_NO_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
            resource: std::ptr::null(),
            offset: 0,
            size: u64::MAX,
        };
        assert_eq!(decode_buffer_barrier(&noop_buf), DecodedBarrier::Skip);

        // Unknown sync bit, split barrier, or mismatched sync/access falls back conservatively.
        let unknown_sync = BufferBarrierDesc {
            sync_before: 0x4000_0000,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
            resource: std::ptr::null(),
            offset: 0,
            size: u64::MAX,
        };
        assert_eq!(
            decode_buffer_barrier(&unknown_sync),
            DecodedBarrier::Fallback(CONSERVATIVE_GLOBAL)
        );
        let mismatched = BufferBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_COPY_DEST,
            access_after: ACCESS_SHADER_RESOURCE,
            resource: std::ptr::null(),
            offset: 0,
            size: u64::MAX,
        };
        assert_eq!(
            decode_buffer_barrier(&mismatched),
            DecodedBarrier::Fallback(CONSERVATIVE_GLOBAL)
        );

        // Global barriers preserve exact SYNC_RENDER_TARGET, SYNC_DEPTH_STENCIL, SYNC_COPY,
        // and RTAS/Raytracing scopes; unknown bits or SYNC_SPLIT fall back conservatively.
        let cs_to_ps = GlobalBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
        };
        assert_eq!(decode_global_barrier(cs_to_ps), DecodedBarrier::Exact(cs_to_ps));

        let rt_global = GlobalBarrierDesc {
            sync_before: SYNC_RENDER_TARGET,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_RENDER_TARGET,
            access_after: ACCESS_SHADER_RESOURCE,
        };
        assert_eq!(decode_global_barrier(rt_global), DecodedBarrier::Exact(rt_global));

        let ds_global = GlobalBarrierDesc {
            sync_before: SYNC_DEPTH_STENCIL,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_DEPTH_STENCIL_WRITE,
            access_after: ACCESS_SHADER_RESOURCE,
        };
        assert_eq!(decode_global_barrier(ds_global), DecodedBarrier::Exact(ds_global));

        let copy_global = GlobalBarrierDesc {
            sync_before: SYNC_COPY,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_COPY_DEST,
            access_after: ACCESS_SHADER_RESOURCE,
        };
        assert_eq!(decode_global_barrier(copy_global), DecodedBarrier::Exact(copy_global));

        let rtas_global = GlobalBarrierDesc {
            sync_before: SYNC_BUILD_RAYTRACING_AS,
            sync_after: SYNC_RAYTRACING,
            access_before: ACCESS_RTAS_WRITE,
            access_after: ACCESS_RTAS_READ,
        };
        assert_eq!(decode_global_barrier(rtas_global), DecodedBarrier::Exact(rtas_global));

        let split_global = GlobalBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_SPLIT,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_NO_ACCESS,
        };
        assert_eq!(
            decode_global_barrier(split_global),
            DecodedBarrier::Fallback(CONSERVATIVE_GLOBAL)
        );
    }

    #[test]
    fn mixed_barrier_groups_preserve_narrow_siblings_on_fallback_and_accept_split_textures() {
        // 1. Valid narrowed compute buffer barrier + 1 unknown-bit buffer barrier in one group.
        let valid_buf = BufferBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
            resource: 0x1000 as This,
            offset: 0,
            size: u64::MAX,
        };
        let bad_buf = BufferBarrierDesc {
            sync_before: 0x4000_0000,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
            resource: 0x2000 as This,
            offset: 0,
            size: u64::MAX,
        };
        let buf_items = [valid_buf, bad_buf];

        // 2. Texture group: 1 valid split RT->SRV begin (Skip), 1 valid scoped UAV texture,
        //    and 1 unrecognized texture bit (Fallback).
        let split_tex = TextureBarrierDesc {
            sync_before: SYNC_RENDER_TARGET,
            sync_after: SYNC_SPLIT,
            access_before: ACCESS_RENDER_TARGET,
            access_after: ACCESS_NO_ACCESS,
            layout_before: 4,
            layout_after: 5,
            resource: 0x3000 as This,
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&split_tex), TextureBarrierPlan::Skip);
        let split_end_tex = TextureBarrierDesc {
            sync_before: SYNC_SPLIT,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_NO_ACCESS,
            access_after: ACCESS_SHADER_RESOURCE,
            layout_before: 4,
            layout_after: 5,
            resource: 0x3000 as This,
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&split_end_tex), TextureBarrierPlan::Skip);

        let uav_tex = TextureBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_UNORDERED_ACCESS,
            layout_before: 3,
            layout_after: 3,
            resource: 0x4000 as This,
            subresources: [0; 6],
            flags: 0,
        };
        let tex_items = [split_tex, split_end_tex, uav_tex];

        // 3. Global group: exact RT->PS global barrier preserved alongside the buffer fallback.
        let rt_global = GlobalBarrierDesc {
            sync_before: SYNC_RENDER_TARGET,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_RENDER_TARGET,
            access_after: ACCESS_SHADER_RESOURCE,
        };
        let global_items = [rt_global];

        let groups = [
            BarrierGroup {
                kind: BARRIER_GROUP_BUFFER,
                count: buf_items.len() as u32,
                barriers: buf_items.as_ptr().cast(),
            },
            BarrierGroup {
                kind: BARRIER_GROUP_TEXTURE,
                count: tex_items.len() as u32,
                barriers: tex_items.as_ptr().cast(),
            },
            BarrierGroup {
                kind: BARRIER_GROUP_GLOBAL,
                count: global_items.len() as u32,
                barriers: global_items.as_ptr().cast(),
            },
        ];

        let plan = plan_parsed_barrier_groups(
            &groups,
            |b| (b.resource == 0x1000 as This).then_some(1u32),
            |t| {
                if t.resource == 0x4000 as This {
                    ResolvedTexUav::Uav(42u32)
                } else {
                    ResolvedTexUav::Fallback
                }
            },
        );

        // One undecoded buffer item must NOT discard the narrow sibling buffer, texture UAV,
        // or exact RT global barrier!
        assert_eq!(
            plan.buf_barriers,
            vec![(
                1u32,
                GlobalBarrierDesc {
                    sync_before: SYNC_COMPUTE_SHADING,
                    sync_after: SYNC_COMPUTE_SHADING,
                    access_before: ACCESS_UNORDERED_ACCESS,
                    access_after: ACCESS_SHADER_RESOURCE,
                }
            )]
        );
        assert_eq!(plan.tex_uavs, vec![42u32]);
        assert_eq!(plan.global_barriers, vec![rt_global]);
        assert_eq!(plan.fallback, Some((CONSERVATIVE_GLOBAL, false)));
    }

    #[test]
    fn decode_texture_barriers_skips_layout_only_and_scopes_uav() {
        let rt_to_srv = TextureBarrierDesc {
            sync_before: SYNC_RENDER_TARGET,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_RENDER_TARGET,
            access_after: ACCESS_SHADER_RESOURCE,
            layout_before: 4,
            layout_after: 5,
            resource: std::ptr::null(),
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&rt_to_srv), TextureBarrierPlan::Skip);

        let depth_to_srv = TextureBarrierDesc {
            sync_before: SYNC_DEPTH_STENCIL,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_DEPTH_STENCIL_WRITE,
            access_after: ACCESS_SHADER_RESOURCE,
            layout_before: 6,
            layout_after: 5,
            resource: std::ptr::null(),
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&depth_to_srv), TextureBarrierPlan::Skip);

        let uav_to_uav = TextureBarrierDesc {
            sync_before: SYNC_COMPUTE_SHADING,
            sync_after: SYNC_COMPUTE_SHADING,
            access_before: ACCESS_UNORDERED_ACCESS,
            access_after: ACCESS_UNORDERED_ACCESS,
            layout_before: 3,
            layout_after: 3,
            resource: std::ptr::null(),
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&uav_to_uav), TextureBarrierPlan::ScopedUav);

        let copy_to_copy = TextureBarrierDesc {
            sync_before: SYNC_COPY,
            sync_after: SYNC_COPY,
            access_before: ACCESS_COPY_DEST,
            access_after: ACCESS_COPY_DEST,
            layout_before: 7,
            layout_after: 7,
            resource: std::ptr::null(),
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&copy_to_copy), TextureBarrierPlan::CopyToCopy);

        let unknown_tex = TextureBarrierDesc {
            sync_before: 0x0800_0000,
            sync_after: SYNC_PIXEL_SHADING,
            access_before: ACCESS_COMMON,
            access_after: ACCESS_SHADER_RESOURCE,
            layout_before: 0,
            layout_after: 5,
            resource: std::ptr::null(),
            subresources: [0; 6],
            flags: 0,
        };
        assert_eq!(decode_texture_barrier(&unknown_tex), TextureBarrierPlan::Fallback);
    }
}
