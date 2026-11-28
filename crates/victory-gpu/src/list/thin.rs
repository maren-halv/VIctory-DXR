//! List calls that only write down their arguments (`VICTORY_THIN=1`); what
//! they mean is worked out by threads of ours, beside the title's.
//!
//! Why: the title makes 45,000 to 60,000 list calls a frame. Each was
//! translated inside the call, 3.5 us of it, 160 to 190 ms of processor time a
//! frame on the title's five recording threads; its own code between the calls
//! is a tenth of that. Nothing else of the title's frame could go on
//! meanwhile: recording took 55 of 65 ms, and behind it the card stood idle
//! for half the frame (timeline waits-3). On the console such a call writes a
//! few words into the list's memory and the card reads them when the list
//! runs. So it is here now: a call appends its arguments (and copies of what
//! they point at) to a stream of words; threads of ours run the stream, call
//! by call and in order, through the list methods there were before, on a list
//! of ours ([`Recording::back`]) that the title never sees. They begin while
//! the title still records (every [`CHUNK`] words are handed over), several
//! lists at a time. A submission goes into the line of submissions
//! (`crate::queue::behind`) as it was made, by the recordings' names; the
//! thread that serves the line goes on with it when they are translated.
//!
//! What may be looked at later than before, and why that is all right: all
//! that the card of the console looks at when the list runs (descriptor heaps,
//! constants, upload memory, indirect arguments): it must stay as it is until
//! the list has run, and no fence behind the list is set before. What D3D12
//! reads at the call is copied at the call: the arrays and structures passed,
//! and the render target and depth views, whose slots may be written again
//! right after. Objects named (pipelines, resources) must live until the list
//! has run; a recording the title resets or releases without submitting it is
//! not run any further.

use super::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::*};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::descriptor::RtvRecord;
use crate::draw::{IndexView, VertexView};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

/// Words after which a stream is handed to the translating threads.
const CHUNK: usize = 1024;

/// Whether list calls are only written down (`VICTORY_THIN=1`; needs the line
/// of submissions, which it turns on).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wanted() && crate::host::host().is_some() && crate::queue::line_on() && {
            start();
            true
        }
    })
}

/// Whether it is asked for.
pub(crate) fn wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THIN").is_some_and(|v| v == "1"))
}

/// What a title list keeps: the recording being written and the words of it
/// not handed over yet.
#[derive(Default)]
pub(super) struct Front {
    recording: Option<Arc<Recording>>,
    words: Vec<u64>,
    dxr_ops: Vec<crate::dxr::DxrOp>,
    closed: bool,
    submitted: bool,
}

impl Drop for Front {
    fn drop(&mut self) {
        self.leave();
    }
}

impl Front {
    /// Lets go of the recording there is: one never submitted is not run any
    /// further.
    fn leave(&mut self) {
        if let Some(recording) = self.recording.take()
            && !self.submitted
        {
            recording.abandoned.store(true, Release);
        }
        self.words.clear();
        self.dxr_ops.clear();
    }

    /// Appends a native DXR operation to this list's thin stream in exact
    /// command-recording order with zero global mutex contention.
    fn put_dxr_op(&mut self, op: crate::dxr::DxrOp) {
        if self.closed || self.recording.is_none() {
            return;
        }
        self.dxr_ops.push(op);
        self.put(DXR_OP, &[]);
    }

    /// Begins a recording, with `pipeline` bound.
    pub(super) fn begin(&mut self, kind: u32, pipeline: This) {
        self.leave();
        self.recording = Some(Arc::new(Recording::new(kind)));
        self.closed = false;
        self.submitted = false;
        self.put(RESET, &[pipeline as u64]);
    }

    #[inline]
    fn put(&mut self, op: u64, args: &[u64]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        self.words.push(op | (args.len() as u64) << 16);
        self.words.extend_from_slice(args);
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    /// [`Self::put`] of some words and then the bytes of some things.
    fn put_with(&mut self, op: u64, args: &[u64], blobs: &[&[u8]]) {
        if self.closed || self.recording.is_none() {
            return;
        }
        let at = self.words.len();
        self.words.push(0);
        self.words.extend_from_slice(args);
        for blob in blobs {
            let from = self.words.len();
            self.words.resize(from + blob.len().div_ceil(8), 0);
            // SAFETY: the words just made hold the bytes; plain data.
            unsafe {
                std::ptr::copy_nonoverlapping(blob.as_ptr(), self.words.as_mut_ptr().add(from).cast(), blob.len())
            };
        }
        let len = (self.words.len() - at - 1) as u64;
        self.words[at] = op | len << 16;
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    /// Serializes `count` console `BarrierGroup` descriptors (and their inline
    /// barrier structs) into the thin stream so translation threads can decode
    /// them without reading guest memory after the call returns.
    fn put_barrier_groups(&mut self, count: u32, groups: *const u8) {
        if self.closed || self.recording.is_none() || count == 0 {
            return;
        }
        const MAX_BARRIERS_PER_GROUP: u32 = 4096;
        if groups.is_null() {
            self.put(BARRIER_GROUPS, &[u64::from(count), 1]);
            return;
        }
        // SAFETY: the title passes `count` `BarrierGroup` headers.
        let slice = unsafe {
            std::slice::from_raw_parts(groups.cast::<super::BarrierGroup>(), count as usize)
        };
        let valid = slice.iter().all(|g| {
            super::barrier_group_item_bytes(g.kind).is_some()
                && g.count <= MAX_BARRIERS_PER_GROUP
                && (g.count == 0 || !g.barriers.is_null())
        });
        if !valid {
            self.put(BARRIER_GROUPS, &[u64::from(count), 1]);
            return;
        }
        let at = self.words.len();
        self.words.push(0);
        self.words.push(u64::from(count));
        self.words.push(0); // opaque = 0
        for g in slice {
            self.words.push(u64::from(g.kind) | (u64::from(g.count) << 32));
            if g.count > 0
                && let Some(item_bytes) = super::barrier_group_item_bytes(g.kind)
            {
                let total_bytes = g.count as usize * item_bytes;
                let from = self.words.len();
                self.words.resize(from + total_bytes.div_ceil(8), 0);
                // SAFETY: `g.barriers` is non-null with `g.count` plain barrier structs;
                // `self.words[from..]` holds `total_bytes`.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        g.barriers,
                        self.words.as_mut_ptr().add(from).cast::<u8>(),
                        total_bytes,
                    );
                }
            }
        }
        let len = (self.words.len() - at - 1) as u64;
        self.words[at] = BARRIER_GROUPS | (len << 16);
        if self.words.len() >= CHUNK {
            self.hand_over();
        }
    }

    fn hand_over(&mut self) {
        let Some(recording) = &self.recording else { return };
        if self.words.is_empty() {
            return;
        }
        let fresh = held(&WORDS).pop().unwrap_or_else(|| Vec::with_capacity(CHUNK + 256));
        let words = std::mem::replace(&mut self.words, fresh);
        let dxr_ops = std::mem::take(&mut self.dxr_ops);
        CHUNKS.fetch_add(1, Relaxed);
        let schedule = {
            let mut inner = held(&recording.inner);
            inner.chunks.push_back((words, dxr_ops));
            !std::mem::replace(&mut inner.scheduled, true)
        };
        if schedule {
            held(&READY).push_back(recording.clone());
            WORK.notify_one();
        }
    }

    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.put(CLOSE, &[]);
        self.closed = true;
        self.hand_over();
    }
}

/// Streams' memory to use again.
static WORDS: Mutex<Vec<Vec<u64>>> = Mutex::new(Vec::new());
/// Recordings with words to translate.
static READY: Mutex<VecDeque<Arc<Recording>>> = Mutex::new(VecDeque::new());
static WORK: Condvar = Condvar::new();
/// Lists of ours nobody uses, closed.
static BACKS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
/// Since the last line of the log: streams handed over; calls translated and
/// the microseconds that took (all threads), recordings translated, and
/// recordings let go of in the middle.
static CHUNKS: AtomicU64 = AtomicU64::new(0);
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
#[cfg(feature = "diagnostics")]
static OP_NS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
#[cfg(feature = "diagnostics")]
static OP_CNT: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

fn held<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One recording of a title list: what was written of it and not translated
/// yet, and the list of ours it goes to.
pub(crate) struct Recording {
    kind: u32,
    inner: Mutex<Inner>,
    /// Translated to its end.
    done: AtomicBool,
    moved: Condvar,
    abandoned: AtomicBool,
}

#[derive(Default)]
struct Inner {
    chunks: VecDeque<(Vec<u64>, Vec<crate::dxr::DxrOp>)>,
    /// Streams translated (commands may name places in them until the end).
    kept: Vec<Vec<u64>>,
    /// It is among [`READY`] or being translated.
    scheduled: bool,
    /// The title needs the open back list for a synchronous host API call.
    waiting_for_back: bool,
    /// The list of ours (0: none yet).
    back: usize,
    /// That list was closed by the stream.
    closed: bool,
}

impl Recording {
    fn new(kind: u32) -> Self {
        Recording {
            kind,
            inner: Mutex::default(),
            done: AtomicBool::new(false),
            moved: Condvar::new(),
            abandoned: AtomicBool::new(false),
        }
    }

    pub(crate) fn done(&self) -> bool {
        self.done.load(Acquire)
    }

    /// Waits until it is translated.
    pub(crate) fn wait(&self) {
        let mut inner = held(&self.inner);
        while !self.done() {
            inner = self.moved.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The list of ours it went to (null: it was never begun).
    pub(crate) fn back(&self) -> This {
        held(&self.inner).back as This
    }

    /// Translates what there is.
    fn run(&self) {
        loop {
            let (chunk, chunk_dxr_ops, back) = {
                let mut inner = held(&self.inner);
                let Some((chunk, chunk_dxr_ops)) = inner.chunks.pop_front() else {
                    inner.scheduled = false;
                    if inner.waiting_for_back {
                        self.moved.notify_all();
                    }
                    return;
                };
                if inner.back == 0
                    && inner.chunks.is_empty()
                    && chunk_dxr_ops.is_empty()
                    && chunk.len() == 3
                    && chunk[0] == (RESET | (1 << 16))
                    && chunk[2] == CLOSE
                {
                    let pipeline = chunk[1] as This;
                    inner.closed = true;
                    inner.scheduled = false;
                    self.done.store(true, Release);
                    self.moved.notify_all();
                    drop(inner);
                    crate::pipeline::bound(pipeline);
                    give_back(vec![chunk]);
                    STATS[2].fetch_add(1, Relaxed);
                    crate::queue::line_moved();
                    return;
                }
                if inner.back == 0 {
                    inner.back = back_list(self.kind) as usize;
                }
                (chunk, chunk_dxr_ops, inner.back as This)
            };
            let began = std::time::Instant::now();
            let (calls, closed) = translate(back, &chunk, chunk_dxr_ops, || self.abandoned.load(Acquire));
            STATS[0].fetch_add(calls, Relaxed);
            STATS[1].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
            let mut inner = held(&self.inner);
            inner.kept.push(chunk);
            if closed {
                inner.closed = true;
                let kept = std::mem::take(&mut inner.kept);
                self.done.store(true, Release);
                self.moved.notify_all();
                drop(inner);
                give_back(kept);
                STATS[2].fetch_add(1, Relaxed);
                crate::queue::line_moved();
            }
        }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        give_back(
            inner
                .chunks
                .drain(..)
                .map(|(words, _)| words)
                .chain(std::mem::take(&mut inner.kept))
                .collect(),
        );
        if inner.back != 0 {
            if !inner.closed {
                // Left in the middle.
                STATS[3].fetch_add(1, Relaxed);
                super::close(inner.back as This);
            }
            let mut backs = held(&BACKS);
            if backs.len() < 384 {
                backs.push(inner.back);
            } else {
                drop(backs);
                // SAFETY: `back_list` allocates this `List` with `Box::into_raw`
                // and nothing else references it once its `Recording` drops.
                unsafe { drop(Box::from_raw(inner.back as *mut List)) };
            }
        }
    }
}

fn give_back(streams: Vec<Vec<u64>>) {
    let mut words = held(&WORDS);
    for mut s in streams {
        if words.len() < 1024 && s.capacity() <= 16 * CHUNK {
            s.clear();
            words.push(s);
        }
    }
}

/// A backing list of ours, without an open host list until `RESET` runs.
fn back_list(kind: u32) -> This {
    if let Some(back) = held(&BACKS).pop() {
        // SAFETY: a list of ours nobody uses.
        unsafe { (*(back as *mut List)).kind = kind };
        return back as This;
    }
    Box::into_raw(super::boxed_closed(kind)).cast_const().cast()
}

fn start() {
    let threads = std::env::var("VICTORY_THIN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8usize).clamp(1, 32);
    for i in 0..threads {
        let worker = move || {
            crate::cores::keep_many_on_fast_cores("victory-list");
            crate::queue::no_drain_here();
            loop {
                let recording = {
                    let mut ready = held(&READY);
                    loop {
                        match ready.pop_front() {
                            Some(r) => break r,
                            None => ready = WORK.wait(ready).unwrap_or_else(|e| e.into_inner()),
                        }
                    }
                };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| recording.run())).is_err() {
                    crate::host::fail(format_args!("a thread that translates lists panicked"));
                }
            }
        };
        if let Err(e) = std::thread::Builder::new().name(format!("victory-list-{i}")).spawn(worker) {
            tracing::warn!(error = %e, "a thread less to translate lists");
        }
    }
    tracing::info!(threads, "list calls are written down and translated by threads of ours (VICTORY_THIN=1)");
}

/// The recordings of submitted lists, in order (one that was not begun is left
/// out).
pub(crate) fn submitted(lists: &[This]) -> Arc<[Arc<Recording>]> {
    lists
        .iter()
        .filter(|l| !l.is_null())
        .filter_map(|&l| {
            // SAFETY: the title passes its lists, which nobody records into now.
            let front = unsafe { &mut (*l.cast_mut().cast::<List>()).thin };
            front.close();
            front.submitted = true;
            front.recording.clone()
        })
        .collect()
}

/// For the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !wanted() || !on() {
        return;
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64 / frames).collect();
    tracing::info!(
        "list calls translated by threads of ours per frame (VICTORY_THIN): {:.0} in {:.1} ms (all threads), {:.1} lists, {:.1} pieces; left in the middle {:.2}",
        s[0],
        s[1] / 1000.0,
        s[2],
        CHUNKS.swap(0, Relaxed) as f64 / frames,
        s[3]
    );
    #[cfg(feature = "diagnostics")]
    {
        let mut ops: Vec<(u64, u64, usize)> = (0..64)
            .filter_map(|op| {
                let ns = OP_NS[op].swap(0, Relaxed);
                let cnt = OP_CNT[op].swap(0, Relaxed);
                (ns > 0).then_some((ns, cnt, op))
            })
            .collect();
        ops.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        let top: Vec<String> = ops
            .into_iter()
            .take(8)
            .map(|(ns, cnt, op)| {
                format!(
                    "op{op}={:.1}ms({:.0}/f)",
                    ns as f64 / (frames * 1e6),
                    cnt as f64 / frames
                )
            })
            .collect();
        if !top.is_empty() {
            tracing::info!("list::thin top ops/f: {}", top.join(", "));
        }
    }
}

#[inline]
fn front<'a>(this: This) -> &'a mut Front {
    // SAFETY: the title passes one of our lists; a list is recorded by one
    // thread at a time.
    unsafe { &mut (*this.cast_mut().cast::<List>()).thin }
}

/// Queues a native DXR operation onto a `VICTORY_THIN=1` front-end list's
/// local command stream without touching any global mutex.
pub(super) fn queue_dxr_op(this: This, op: crate::dxr::DxrOp) {
    front(this).put_dxr_op(op);
}

/// Lets a host API record at the current position of the title's open list.
/// The caller is the sole writer of this list until its API call returns.
/// Only preceding chunks of this recording have to finish translation.
pub(crate) fn for_external(this: This) -> Option<This> {
    if this.is_null() {
        return None;
    }
    if super::host_list(this).is_some() {
        return Some(this);
    }
    let front = front(this);
    if front.closed {
        return None;
    }
    front.hand_over();
    let recording = front.recording.as_ref()?;
    let mut inner = held(&recording.inner);
    if inner.scheduled {
        inner.waiting_for_back = true;
        inner = recording
            .moved
            .wait_while(inner, |i| i.scheduled)
            .unwrap_or_else(|e| e.into_inner());
        inner.waiting_for_back = false;
    }
    (inner.back != 0).then_some(inner.back as This)
}

/// Runs `f` with the host list and draw state of a title list for a
/// synchronous host API call.
pub(crate) fn with_external<R>(
    this: This,
    f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State) -> R,
) -> Option<R> {
    let list = for_external(this)?;
    let mut out = None;
    super::with_state(list, |host, state| {
        out = Some(f(host, state));
    });
    out
}

/// An argument as a word.
trait Word: Copy {
    fn word(self) -> u64;
    fn unword(w: u64) -> Self;
}
macro_rules! words {
    ($($t:ty),*) => { $(impl Word for $t {
        #[inline]
        fn word(self) -> u64 { self as u64 }
        #[inline]
        fn unword(w: u64) -> Self { w as $t }
    })* };
}
words!(u8, u32, i32, u64, This);
impl Word for f32 {
    #[inline]
    fn word(self) -> u64 {
        u64::from(self.to_bits())
    }
    #[inline]
    fn unword(w: u64) -> Self {
        f32::from_bits(w as u32)
    }
}

/// The methods whose arguments are all plain values: the one the title calls,
/// and the call made of it later.
macro_rules! plain {
    ($($op:ident $name:ident = $was:path, ($($a:ident: $t:ty),*);)*) => {
        #[allow(clippy::upper_case_acronyms, non_camel_case_types)]
        #[repr(u64)]
        enum Plain { $($op),* , END }
        $(extern "system" fn $name(this: This $(, $a: $t)*) {
            front(this).put(Plain::$op as u64, &[$(Word::word($a)),*]);
        })*
        /// Whether `op` was one of them; done on `list`.
        #[inline]
        fn run_plain(op: u64, list: This, w: &[u64]) -> bool {
            match op {
                $(n if n == Plain::$op as u64 => {
                    let mut words = w.iter();
                    $(let $a: $t = Word::unword(words.next().copied().unwrap_or(0));)*
                    let _ = words;
                    $was(list $(, $a)*);
                })*
                _ => return false,
            }
            true
        }
    };
}

plain! {
    TOPOLOGY topology = state::topology, (v: u32);
    STENCIL_REF stencil_ref = state::stencil_ref, (v: u32);
    DEPTH_BOUNDS depth_bounds = state::depth_bounds, (min: f32, max: f32);
    COMPUTE_TABLE compute_table = state::compute_table, (slot: u32, handle: u64);
    GRAPHICS_TABLE graphics_table = state::graphics_table, (slot: u32, handle: u64);
    COMPUTE_CONSTANT compute_constant = state::compute_constant, (slot: u32, value: u32, offset: u32);
    GRAPHICS_CONSTANT graphics_constant = state::graphics_constant, (slot: u32, value: u32, offset: u32);
    COMPUTE_VIEW compute_view = state::compute_view, (slot: u32, address: u64);
    GRAPHICS_VIEW graphics_view = state::graphics_view, (slot: u32, address: u64);
    SET_PIPELINE set_pipeline = super::set_pipeline, (pipeline: This);
    DRAW draw = state::draw, (vertices: u32, instances: u32, first: u32, first_instance: u32);
    DRAW_INDEXED draw_indexed = state::draw_indexed, (indices: u32, instances: u32, first: u32, base: i32, first_instance: u32);
    DISPATCH dispatch = state::dispatch, (x: u32, y: u32, z: u32);
    DISPATCH_X dispatch_x = super::dispatch_x, (x: u32, y: u32, z: u32, flags: u32);
    DISPATCH_MESH dispatch_mesh = state::dispatch_mesh, (x: u32, y: u32, z: u32);
    DISPATCH_MESH_X dispatch_mesh_x = state::dispatch_mesh_x, (x: u32, flags: u64, indirect: u64, payload: u64);
    COPY_BUFFER copy_buffer_region = super::copy_buffer_region, (dst: This, dst_offset: u64, src: This, src_offset: u64, len: u64);
    COPY_RESOURCE copy_resource = super::copy_resource, (dst: This, src: This);
    COPY_RESOURCE_X copy_resource_x = super::copy_resource_x, (dst: This, src: This, flags: u32);
    COPY_MEMORY copy_memory = super::copy_memory, (dst: u64, src: u64, len: u64, flags: u32);
    RESOLVE resolve = super::resolve, (dst: This, dst_sub: u32, src: This, src_sub: u32, format: u32);
    BEGIN_QUERY begin_query = super::begin_query, (heap: This, kind: u32, index: u32);
    END_QUERY end_query = super::end_query, (heap: This, kind: u32, index: u32);
    RESOLVE_QUERIES resolve_queries = super::resolve_queries, (heap: This, kind: u32, first: u32, count: u32, buffer: This, offset: u64);
    SET_PREDICATION set_predication = super::set_predication, (buffer: This, offset: u64, operation: u32);
    EXECUTE_INDIRECT execute_indirect = super::execute_indirect, (signature: This, max: u32, arguments: This, offset: u64, count: This, count_offset: u64);
    WRITE32 write32 = super::write32, (address: u64, value: u32, flush: u32, flags: u32);
    WRITE64 write64 = super::write64, (address: u64, value: u64, flush: u32, flags: u32);
    WAIT_VALUE wait_value = super::wait_value, (address: u64, comparison: u32, reference: u64);
    WAIT_VALUE64 wait_value64 = super::wait_value64, (address: u64, comparison: u32, reference: u64);
    SYNC sync = super::sync, (flags: u32, control: u32, address: u64, size: u64);
}

const fn after(n: u64) -> u64 {
    Plain::END as u64 + n
}
const RESET: u64 = after(0);
const CLOSE: u64 = after(1);
const UPLOAD: u64 = after(2);
const BARRIER: u64 = after(3);
const BARRIER_GROUPS: u64 = after(4);
const VIEWPORTS: u64 = after(5);
const SCISSORS: u64 = after(6);
const BLEND_FACTOR: u64 = after(7);
const HEAPS: u64 = after(8);
const COMPUTE_CONSTANTS: u64 = after(9);
const GRAPHICS_CONSTANTS: u64 = after(10);
const INDEX_BUFFER: u64 = after(11);
const VERTEX_BUFFERS: u64 = after(12);
const TARGETS: u64 = after(13);
const CLEAR_RTV: u64 = after(14);
const CLEAR_DSV: u64 = after(15);
const COPY_TEXTURE: u64 = after(16);
const DXR_OP: u64 = after(17);

/// The bytes of `count` things at `items` (none if null).
fn bytes_of<'a, T>(items: *const T, count: usize) -> &'a [u8] {
    if items.is_null() || count == 0 {
        return &[];
    }
    // SAFETY: the title passes `count` items.
    unsafe { std::slice::from_raw_parts(items.cast(), count * size_of::<T>()) }
}

/// Where the things are in a stream (null for none).
fn at<T>(w: &[u64], from: usize, count: usize) -> *const T {
    if count == 0 || from >= w.len() { std::ptr::null() } else { w[from..].as_ptr().cast() }
}

/// [`at`] where an empty array is not null.
fn at_or<T>(w: &[u64], from: usize) -> *const T {
    // SAFETY: at most one past the end.
    unsafe { w.as_ptr().add(from.min(w.len())).cast() }
}

/// A view's record as it is now (zeros for no view), so that its slot may be
/// written again: the copy stands for the handle.
fn record(handle: u64) -> [u64; RECORD] {
    let mut words = [0u64; RECORD];
    if handle != 0 {
        // SAFETY: a descriptor slot the title passes (they are larger).
        unsafe {
            std::ptr::copy_nonoverlapping(handle as *const u8, words.as_mut_ptr().cast(), size_of::<RtvRecord>())
        };
    }
    words
}
const RECORD: usize = size_of::<RtvRecord>().div_ceil(8);

extern "system" fn list_close(this: This) -> HRESULT {
    front(this).close();
    S_OK
}
extern "system" fn list_reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our lists.
    let kind = unsafe { (*this.cast::<List>()).kind };
    front(this).begin(kind, pipeline);
    S_OK
}
extern "system" fn list_reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    list_reset(this, allocator, pipeline)
}
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let front = front(this);
    if front.closed || front.recording.is_none() {
        return std::ptr::null_mut();
    }
    let Some(mut block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let place = unsafe { start.add(start.align_offset(align)) };
    front.put(UPLOAD, &[Box::into_raw(Box::new(block)) as u64]);
    place
}
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    if !barriers.is_null() {
        front(this).put_with(BARRIER, &[u64::from(count)], &[bytes_of(barriers, count as usize)]);
    }
    0
}
extern "system" fn barrier_groups(this: This, count: u32, groups: *const u8) {
    front(this).put_barrier_groups(count, groups);
}
extern "system" fn viewports(this: This, count: u32, items: *const D3D12_VIEWPORT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(VIEWPORTS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn scissors(this: This, count: u32, items: *const RECT) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(SCISSORS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
    let given = !factor.is_null();
    front(this).put_with(BLEND_FACTOR, &[u64::from(given)], &[bytes_of(factor, usize::from(given))]);
}
extern "system" fn heaps(this: This, count: u32, items: *const This) {
    let count = if items.is_null() { 0 } else { count };
    front(this).put_with(HEAPS, &[u64::from(count)], &[bytes_of(items, count as usize)]);
}
fn constants(this: This, op: u64, slot: u32, count: u32, values: *const u32, offset: u32) {
    let count = if values.is_null() { 0 } else { count };
    front(this).put_with(
        op,
        &[u64::from(slot) | u64::from(offset) << 32, u64::from(count)],
        &[bytes_of(values, count as usize)],
    );
}
extern "system" fn compute_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, COMPUTE_CONSTANTS, slot, count, values, offset);
}
extern "system" fn graphics_constants(this: This, slot: u32, count: u32, values: *const u32, offset: u32) {
    constants(this, GRAPHICS_CONSTANTS, slot, count, values, offset);
}
extern "system" fn index_buffer(this: This, view: *const IndexView) {
    let given = !view.is_null();
    front(this).put_with(INDEX_BUFFER, &[u64::from(given)], &[bytes_of(view, usize::from(given))]);
}
extern "system" fn vertex_buffers(this: This, start: u32, count: u32, views: *const VertexView) {
    let given = !views.is_null();
    front(this).put_with(
        VERTEX_BUFFERS,
        &[u64::from(start) | u64::from(count) << 32, u64::from(given)],
        &[bytes_of(views, count as usize)],
    );
}
extern "system" fn targets(this: This, count: u32, handles: *const u64, one_range: i32, depth: *const u64) {
    let count = if handles.is_null() { 0 } else { (count as usize).min(8) };
    // SAFETY: the title passes a handle or null.
    let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
    let mut records = [0u64; RECORD * 9];
    let mut present = 0u64;
    for i in 0..count {
        // SAFETY: the title passes `count` handles, or the first of a range
        // of views of 128 bytes.
        let handle = unsafe { if one_range != 0 { *handles + 128 * i as u64 } else { *handles.add(i) } };
        present |= u64::from(handle != 0) << i;
        records[i * RECORD..(i + 1) * RECORD].copy_from_slice(&record(handle));
    }
    records[count * RECORD..(count + 1) * RECORD].copy_from_slice(&record(depth));
    let records = &records[..(count + 1) * RECORD];
    front(this).put_with(
        TARGETS,
        &[count as u64, present, u64::from(depth != 0)],
        &[bytes_of(records.as_ptr(), records.len())],
    );
}
extern "system" fn clear_rtv(this: This, handle: u64, color: *const [f32; 4], count: u32, rects: *const Rect) {
    if handle == 0 || color.is_null() {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_RTV,
        &[u64::from(count)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(color, 1), bytes_of(rects, count as usize)],
    );
}
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    if handle == 0 {
        return;
    }
    let count = if rects.is_null() { 0 } else { count };
    let view = record(handle);
    front(this).put_with(
        CLEAR_DSV,
        &[u64::from(count), u64::from(flags) | u64::from(stencil) << 32, Word::word(depth)],
        &[bytes_of(view.as_ptr(), RECORD), bytes_of(rects, count as usize)],
    );
}
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    if dst.is_null() || src.is_null() {
        return;
    }
    let boxed = !source_box.is_null();
    front(this).put_with(
        COPY_TEXTURE,
        &[u64::from(x) | u64::from(y) << 32, u64::from(z) | u64::from(boxed) << 32],
        &[bytes_of(dst, 1), bytes_of(src, 1), bytes_of(source_box, usize::from(boxed))],
    );
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

#[inline]
fn is_state_op(op: u64) -> bool {
    op == Plain::GRAPHICS_TABLE as u64
        || op == Plain::COMPUTE_TABLE as u64
        || op == Plain::GRAPHICS_CONSTANT as u64
        || op == Plain::COMPUTE_CONSTANT as u64
        || op == Plain::GRAPHICS_VIEW as u64
        || op == Plain::COMPUTE_VIEW as u64
        || op == Plain::SET_PIPELINE as u64
        || op == Plain::TOPOLOGY as u64
        || op == Plain::STENCIL_REF as u64
        || op == Plain::DEPTH_BOUNDS as u64
        || op == Plain::DRAW as u64
        || op == Plain::DRAW_INDEXED as u64
        || op == Plain::DISPATCH as u64
        || op == Plain::DISPATCH_X as u64
        || op == Plain::DISPATCH_MESH as u64
        || op == Plain::DISPATCH_MESH_X as u64
        || op == Plain::EXECUTE_INDIRECT as u64
        || matches!(
            op,
            GRAPHICS_CONSTANTS
                | COMPUTE_CONSTANTS
                | VIEWPORTS
                | SCISSORS
                | BLEND_FACTOR
                | HEAPS
                | INDEX_BUFFER
                | VERTEX_BUFFERS
                | TARGETS
        )
}

/// Attempts to execute a hot state/draw/dispatch opcode directly against an
/// already-held `(&HostList, &mut draw::State)` without re-locking `list.state`
/// or taking per-call diagnostic timestamps. Returns `true` if handled.
#[inline]
fn run_state_op(op: u64, host: &crate::host::HostList, state: &mut crate::draw::State, w: &[u64]) -> bool {
    let word = |idx: usize| w.get(idx).copied().unwrap_or(0);
    let slice = |from: usize, count: usize| -> *const u8 {
        if count == 0 || from >= w.len() {
            std::ptr::null()
        } else {
            w[from..].as_ptr().cast()
        }
    };
    match op {
        n if n == Plain::GRAPHICS_TABLE as u64 => state.set_table(false, word(0) as u32, word(1)),
        n if n == Plain::COMPUTE_TABLE as u64 => state.set_table(true, word(0) as u32, word(1)),
        n if n == Plain::GRAPHICS_CONSTANT as u64 => {
            state.set_constants(false, word(0) as u32, &[word(1) as u32], word(2) as u32)
        }
        n if n == Plain::COMPUTE_CONSTANT as u64 => {
            state.set_constants(true, word(0) as u32, &[word(1) as u32], word(2) as u32)
        }
        n if n == Plain::GRAPHICS_VIEW as u64 => state.set_view(false, word(0) as u32, word(1)),
        n if n == Plain::COMPUTE_VIEW as u64 => state.set_view(true, word(0) as u32, word(1)),
        n if n == Plain::SET_PIPELINE as u64 => {
            let p = word(0) as This;
            crate::pipeline::bound(p);
            state.set_pipeline(p);
        }
        n if n == Plain::TOPOLOGY as u64 => state.set_topology(host, word(0) as u32),
        n if n == Plain::STENCIL_REF as u64 => state.set_stencil_ref(host, word(0) as u32),
        n if n == Plain::DEPTH_BOUNDS as u64 => {
            state.set_depth_bounds(host, Word::unword(word(0)), Word::unword(word(1)))
        }
        n if n == Plain::DRAW as u64 => {
            super::count_draw();
            state.draw(host, word(0) as u32, word(1) as u32, word(2) as u32, word(3) as u32);
        }
        n if n == Plain::DRAW_INDEXED as u64 => {
            super::count_draw();
            state.draw_indexed(
                host,
                word(0) as u32,
                word(1) as u32,
                word(2) as u32,
                Word::unword(word(3)),
                word(4) as u32,
            );
        }
        n if n == Plain::DISPATCH as u64 => {
            super::count_dispatch();
            state.dispatch(host, word(0) as u32, word(1) as u32, word(2) as u32, 0);
        }
        n if n == Plain::DISPATCH_X as u64 => {
            super::count_dispatch();
            state.dispatch(host, word(0) as u32, word(1) as u32, word(2) as u32, word(3) as u32);
        }
        n if n == Plain::DISPATCH_MESH as u64 => {
            super::count_draw();
            let (x, y, z) = (word(0) as u32, word(1) as u32, word(2) as u32);
            if y * z != 1 {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
            }
            state.dispatch_mesh(host, x);
        }
        n if n == Plain::DISPATCH_MESH_X as u64 => {
            super::count_draw();
            let (x, indirect) = (word(0) as u32, word(2));
            if indirect != 0 {
                state.dispatch_mesh_indirect(host, indirect);
            } else {
                state.dispatch_mesh(host, x);
            }
        }
        n if n == Plain::EXECUTE_INDIRECT as u64 => {
            super::execute_indirect_with(
                host,
                state,
                word(0) as This,
                word(1) as u32,
                word(2) as This,
                word(3),
                word(4) as This,
                word(5),
            );
        }
        GRAPHICS_CONSTANTS | COMPUTE_CONSTANTS => {
            let count = word(1) as usize;
            let ptr = slice(2, count).cast::<u32>();
            // SAFETY: stream holds `count` u32 values written by `put_with`.
            let vals = if ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(ptr, count) } };
            state.set_constants(op == COMPUTE_CONSTANTS, word(0) as u32, vals, (word(0) >> 32) as u32);
        }
        VIEWPORTS => {
            let n = word(0) as usize;
            let ptr = slice(1, n).cast::<D3D12_VIEWPORT>();
            // SAFETY: stream holds `n` viewports written by `put_with`.
            let items = if ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(ptr, n) } };
            state.set_viewports(host, items);
        }
        SCISSORS => {
            let n = word(0) as usize;
            let ptr = slice(1, n).cast::<RECT>();
            // SAFETY: stream holds `n` rects written by `put_with`.
            let items = if ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(ptr, n) } };
            state.set_scissors(host, items);
        }
        BLEND_FACTOR => {
            let n = word(0) as usize;
            let ptr = slice(1, n).cast::<[f32; 4]>();
            // SAFETY: stream holds `[f32; 4]` when `n != 0`.
            state.set_blend_factor(host, unsafe { ptr.as_ref() });
        }
        HEAPS => {
            let n = word(0) as usize;
            let ptr = slice(1, n).cast::<This>();
            // SAFETY: stream holds `n` heap handles written by `put_with`.
            let items = if ptr.is_null() { &[] } else { unsafe { std::slice::from_raw_parts(ptr, n) } };
            state.set_heaps(items);
        }
        INDEX_BUFFER => {
            let n = word(0) as usize;
            let ptr = slice(1, n).cast::<IndexView>();
            // SAFETY: stream holds an `IndexView` when `n != 0`.
            state.set_index_buffer(unsafe { ptr.as_ref() });
        }
        VERTEX_BUFFERS => {
            let start = word(0) as u32;
            let count = (word(0) >> 32) as u32;
            if word(1) != 0 {
                let ptr = at_or::<VertexView>(w, 2);
                // SAFETY: stream holds `count` `VertexView` entries.
                let items = if count == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(ptr, count as usize) }
                };
                state.set_vertex_buffers(start, items);
            } else {
                state.clear_vertex_buffers(start, count);
            }
        }
        TARGETS => {
            let n = (word(0) as usize).min(8);
            let mut handles = [0u64; 8];
            let place = |k: usize| w[3 + k * RECORD..].as_ptr() as u64;
            for (k, h) in handles.iter_mut().enumerate().take(n) {
                if word(1) >> k & 1 != 0 {
                    *h = place(k);
                }
            }
            let depth = if word(2) != 0 { place(n) } else { 0 };
            state.set_targets(host, &handles[..n], depth);
        }
        _ => return false,
    }
    true
}

/// Reconstructs `count` [`super::BarrierGroup`] headers pointing into the
/// serialized stream slice `words`. Returns `None` if malformed.
fn unpack_barrier_groups(count: usize, words: &[u64]) -> Option<Vec<super::BarrierGroup>> {
    let mut out = Vec::with_capacity(count);
    let mut pos = 0usize;
    for _ in 0..count {
        let hdr = *words.get(pos)?;
        pos += 1;
        let kind = hdr as u32;
        let num = (hdr >> 32) as u32;
        let item_bytes = super::barrier_group_item_bytes(kind)?;
        let item_words = item_bytes.div_ceil(8);
        let needed = (num as usize).checked_mul(item_words)?;
        if pos.checked_add(needed)? > words.len() {
            return None;
        }
        let barriers = if num == 0 {
            std::ptr::null()
        } else {
            words[pos..].as_ptr().cast::<u8>()
        };
        pos += needed;
        out.push(super::BarrierGroup {
            kind,
            count: num,
            barriers,
        });
    }
    Some(out)
}

/// Makes the calls of a stream on `list`: how many, and whether the last one
/// closed it. Once `left` says so only what must not be lost is done.
fn translate(
    list: This,
    stream: &[u64],
    dxr_ops: Vec<crate::dxr::DxrOp>,
    left: impl Fn() -> bool,
) -> (u64, bool) {
    let _retire_batch = crate::host::RetireBatchGuard::enter();
    let (mut i, mut calls, mut closed) = (0, 0, false);
    let mut skip = left();
    let mut dxr_iter = dxr_ops.into_iter();
    // Hold `list.state` across consecutive pure-state/draw/dispatch calls so
    // the worker does not re-acquire the mutex and query QPC on every call.
    let mut held_state: Option<(&crate::host::HostList, std::sync::MutexGuard<'_, crate::draw::State>)> = None;
    #[cfg(feature = "diagnostics")]
    let mut batch: Option<(u64, std::time::Instant)> = None;
    let flush_held = |held: &mut Option<(&crate::host::HostList, std::sync::MutexGuard<'_, crate::draw::State>)>,
                      #[cfg(feature = "diagnostics")] batch: &mut Option<(u64, std::time::Instant)>| {
        if held.take().is_some() {
            #[cfg(feature = "diagnostics")]
            if let Some((n, started)) = batch.take()
                && n > 0
            {
                crate::copystat::recorded_batch(n, started.elapsed());
            }
        }
    };
    while i < stream.len() {
        let (op, len) = (stream[i] & 0xffff, (stream[i] >> 16) as usize);
        let w = &stream[i + 1..i + 1 + len];
        i += 1 + len;
        calls += 1;
        if calls % 64 == 0 {
            skip = left();
        }
        #[cfg(feature = "diagnostics")]
        let op_t0 = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let op_idx = (op as usize).min(63);
        #[cfg(feature = "diagnostics")]
        struct OpTimer(usize, std::time::Instant);
        #[cfg(feature = "diagnostics")]
        impl Drop for OpTimer {
            fn drop(&mut self) {
                OP_NS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Relaxed);
                OP_CNT[self.0].fetch_add(1, Relaxed);
            }
        }
        #[cfg(feature = "diagnostics")]
        let _op_timer = OpTimer(op_idx, op_t0);
        if op == UPLOAD {
            // SAFETY: made by `allocate_upload`, named once.
            let block = *unsafe { Box::from_raw(w[0] as *mut Upload) };
            // SAFETY: a list of ours.
            let uploads = unsafe { &(*list.cast::<List>()).uploads };
            held(uploads).push(block);
            continue;
        }
        if skip && op != CLOSE && op != RESET {
            continue;
        }
        if is_state_op(op) {
            if held_state.is_none()
                && let Some(host) = super::host_list(list)
            {
                // SAFETY: `list` is one of our backing `List`s, exclusively owned by this worker.
                let state = unsafe { &(*list.cast::<List>()).state }
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                #[cfg(feature = "diagnostics")]
                {
                    batch = Some((0, std::time::Instant::now()));
                }
                held_state = Some((host, state));
            }
            if let Some((host, state)) = held_state.as_mut()
                && run_state_op(op, host, state, w)
            {
                #[cfg(feature = "diagnostics")]
                if let Some((n, _)) = batch.as_mut() {
                    *n += 1;
                }
                continue;
            }
        }
        flush_held(
            &mut held_state,
            #[cfg(feature = "diagnostics")]
            &mut batch,
        );
        if run_plain(op, list, w) {
            continue;
        }
        let n = w.first().map_or(0, |&n| n as usize);
        match op {
            RESET => {
                // (Only whether there is an allocator is looked at.)
                let allocator: This = std::ptr::NonNull::<u64>::dangling().as_ptr().cast_const().cast();
                super::reset(list, allocator, w[0] as This);
            }
            CLOSE => {
                super::close(list);
                closed = true;
            }
            BARRIER => {
                super::barrier(list, n as u32, at_or(w, 1));
            }
            BARRIER_GROUPS => {
                let opaque = w.get(1).copied().unwrap_or(1);
                if n == 0 {
                    continue;
                }
                if opaque != 0 {
                    super::barrier_groups(list, n as u32, std::ptr::null());
                    continue;
                }
                if let Some(rebuilt) = unpack_barrier_groups(n, &w[2..]) {
                    super::barrier_groups(list, n as u32, rebuilt.as_ptr().cast());
                } else {
                    super::barrier_groups(list, n as u32, std::ptr::null());
                }
            }
            VIEWPORTS => state::viewports(list, n as u32, at(w, 1, n)),
            SCISSORS => state::scissors(list, n as u32, at(w, 1, n)),
            BLEND_FACTOR => state::blend_factor(list, at(w, 1, n)),
            HEAPS => state::heaps(list, n as u32, at(w, 1, n)),
            COMPUTE_CONSTANTS => {
                state::compute_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            GRAPHICS_CONSTANTS => {
                state::graphics_constants(list, w[0] as u32, w[1] as u32, at(w, 2, w[1] as usize), (w[0] >> 32) as u32)
            }
            INDEX_BUFFER => state::index_buffer(list, at(w, 1, n)),
            VERTEX_BUFFERS => {
                let views = if w[1] != 0 { at_or(w, 2) } else { std::ptr::null() };
                state::vertex_buffers(list, w[0] as u32, (w[0] >> 32) as u32, views);
            }
            TARGETS => {
                let mut handles = [0u64; 8];
                let place = |k: usize| w[3 + k * RECORD..].as_ptr() as u64;
                for (k, h) in handles.iter_mut().enumerate().take(n) {
                    if w[1] >> k & 1 != 0 {
                        *h = place(k);
                    }
                }
                let depth = if w[2] != 0 { place(n) } else { 0 };
                state::targets(list, n as u32, handles.as_ptr(), 0, &depth);
            }
            CLEAR_RTV => {
                super::clear_rtv(
                    list,
                    w[1..].as_ptr() as u64,
                    at_or(w, 1 + RECORD),
                    n as u32,
                    at(w, 1 + RECORD + 2, n),
                );
            }
            CLEAR_DSV => {
                super::clear_dsv(
                    list,
                    w[3..].as_ptr() as u64,
                    w[1] as u32,
                    Word::unword(w[2]),
                    (w[1] >> 32) as u8,
                    n as u32,
                    at(w, 3 + RECORD, n),
                );
            }
            COPY_TEXTURE => {
                let step = size_of::<CopyLocation>().div_ceil(8);
                let boxed = (w[1] >> 32) as usize;
                super::copy_texture_region(
                    list,
                    at_or(w, 2),
                    w[0] as u32,
                    (w[0] >> 32) as u32,
                    w[1] as u32,
                    at_or(w, 2 + step),
                    at(w, 2 + 2 * step, boxed),
                );
            }
            DXR_OP => {
                if let Some(dxr_op) = dxr_iter.next()
                    && let Some(host) = super::host_list(list)
                {
                    host.push_dxr_op(dxr_op);
                }
            }
            _ => {}
        }
    }
    flush_held(
        &mut held_state,
        #[cfg(feature = "diagnostics")]
        &mut batch,
    );
    (calls, closed)
}

/// The table of a list whose calls are written down: that of the others with
/// the methods that record replaced (slots as in [`super::METHODS`]).
pub(super) const fn table(newer: bool) -> [*const (); 160] {
    let mut f = super::list_table(newer);
    let mut i = 0;
    while i < MINE.len() {
        let (slot, method) = MINE[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    if newer {
        f[146] = barrier_groups as *const ();
        f[148] = allocate_upload as *const ();
    }
    f
}

const MINE: [(usize, *const ()); 53] = [
    (9, list_close as *const ()),
    (10, list_reset as *const ()),
    (12, draw as *const ()),
    (13, draw_indexed as *const ()),
    (14, dispatch as *const ()),
    (15, copy_buffer_region as *const ()),
    (16, copy_texture_region as *const ()),
    (17, copy_resource as *const ()),
    (19, resolve as *const ()),
    (20, topology as *const ()),
    (21, viewports as *const ()),
    (22, scissors as *const ()),
    (23, blend_factor as *const ()),
    (24, stencil_ref as *const ()),
    (25, set_pipeline as *const ()),
    (26, barrier as *const ()),
    (28, begin_query as *const ()),
    (29, end_query as *const ()),
    (30, resolve_queries as *const ()),
    (31, set_predication as *const ()),
    (32, heaps as *const ()),
    (35, compute_table as *const ()),
    (36, graphics_table as *const ()),
    (37, compute_constant as *const ()),
    (38, graphics_constant as *const ()),
    (39, compute_constants as *const ()),
    (40, graphics_constants as *const ()),
    (41, compute_view as *const ()),
    (42, graphics_view as *const ()),
    (43, compute_view as *const ()),
    (44, graphics_view as *const ()),
    (45, compute_view as *const ()),
    (46, graphics_view as *const ()),
    (47, index_buffer as *const ()),
    (48, vertex_buffers as *const ()),
    (50, targets as *const ()),
    (51, clear_dsv as *const ()),
    (52, clear_rtv as *const ()),
    (59, execute_indirect as *const ()),
    (60, copy_memory as *const ()),
    (68, depth_bounds as *const ()),
    (69, write32 as *const ()),
    (70, write64 as *const ()),
    (71, wait_value as *const ()),
    (72, wait_value64 as *const ()),
    (78, dispatch_x as *const ()),
    (83, copy_texture_region_x as *const ()),
    (84, copy_resource_x as *const ()),
    (98, dispatch_mesh_x as *const ()),
    (127, dispatch_mesh as *const ()),
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, list_reset_x as *const ()),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method of the tables that records is written down too: what is
    /// left in both tables alike records nothing.
    #[test]
    fn every_recording_method_is_written_down() {
        const NOTHING: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 33, 34, 73, 99, 124];
        let mine: Vec<usize> = MINE.iter().map(|m| m.0).collect();
        for (slot, _) in METHODS.iter().chain(MORE_METHODS.iter()) {
            assert!(mine.contains(slot) != NOTHING.contains(slot), "slot {slot}");
        }
        for &slot in &mine {
            assert!(
                METHODS.iter().chain(MORE_METHODS.iter()).any(|m| m.0 == slot),
                "slot {slot} is written down and not there"
            );
        }
    }

    /// Arguments come out of a stream as they went in.
    #[test]
    fn a_stream_gives_back_what_was_written() {
        let mut f = Front::default();
        f.recording = Some(Arc::new(Recording::new(0)));
        f.put(Plain::DRAW_INDEXED as u64, &[3, 1, 0, Word::word(-7i32), 5]);
        let viewport = [1.0f32, 2.0, 3.0, 4.0, 0.0, 1.0];
        f.put_with(VIEWPORTS, &[1], &[bytes_of(viewport.as_ptr(), 6)]);
        f.put_with(COMPUTE_CONSTANTS, &[4 | 2 << 32, 3], &[bytes_of([7u32, 8, 9].as_ptr(), 3)]);
        let s = std::mem::take(&mut f.words);
        assert_eq!((s[0] & 0xffff, s[0] >> 16), (Plain::DRAW_INDEXED as u64, 5));
        assert_eq!(<i32 as Word>::unword(s[4]), -7);
        let w = &s[6..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16, w[1]), (VIEWPORTS, 4, 1));
        // SAFETY: what was written just now.
        assert_eq!(unsafe { *at::<[f32; 6]>(&w[1..], 1, 1) }, viewport);
        let w = &w[5..];
        assert_eq!((w[0] & 0xffff, w[0] >> 16), (COMPUTE_CONSTANTS, 4));
        // SAFETY: as above.
        assert_eq!(unsafe { *at::<[u32; 3]>(&w[1..], 2, 3) }, [7, 8, 9]);
        assert!(at::<u32>(&w[1..], 2, 0).is_null());
    }

    /// A view's record is the one at the call, also if its slot is written
    /// again right after.
    #[test]
    fn a_target_is_the_one_at_the_call() {
        let mut slot = [0x5256_5476u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let kept = record(slot.as_ptr() as u64);
        slot[1] = 99;
        assert_eq!(kept[1], 2);
        assert_eq!(record(0), [0; RECORD]);
    }

    /// Serialized barrier groups preserve child Global/Buffer/Texture structs
    /// even after the caller overwrites its stack arrays, and mark unknown
    /// group kinds as opaque for conservative fallback.
    #[test]
    fn barrier_groups_round_trip_in_thin_stream() {
        let mut f = Front::default();
        f.recording = Some(Arc::new(Recording::new(0)));

        let mut g_bar = super::GlobalBarrierDesc {
            sync_before: 0x80,
            sync_after: 0x80,
            access_before: 0x10,
            access_after: 0x80,
        };
        let mut b_bar = super::BufferBarrierDesc {
            sync_before: 0x80,
            sync_after: 0x800,
            access_before: 0x10,
            access_after: 0x200,
            resource: 0x1234usize as This,
            offset: 0,
            size: u64::MAX,
        };
        let groups = [
            super::BarrierGroup {
                kind: super::BARRIER_GROUP_GLOBAL,
                count: 1,
                barriers: (&raw const g_bar).cast(),
            },
            super::BarrierGroup {
                kind: super::BARRIER_GROUP_BUFFER,
                count: 1,
                barriers: (&raw const b_bar).cast(),
            },
        ];
        f.put_barrier_groups(groups.len() as u32, groups.as_ptr().cast());
        // Mutate caller stack structs after recording to prove deep copy.
        g_bar.sync_before = 0;
        b_bar.offset = 999;
        assert_eq!((g_bar.sync_before, b_bar.offset), (0, 999));

        let s = std::mem::take(&mut f.words);
        assert_eq!(s[0] & 0xffff, BARRIER_GROUPS);
        let len = (s[0] >> 16) as usize;
        let w = &s[1..1 + len];
        assert_eq!((w[0], w[1]), (2, 0));
        let unpacked = unpack_barrier_groups(w[0] as usize, &w[2..]).expect("valid groups");
        assert_eq!(unpacked.len(), 2);
        assert_eq!((unpacked[0].kind, unpacked[0].count), (super::BARRIER_GROUP_GLOBAL, 1));
        assert_eq!((unpacked[1].kind, unpacked[1].count), (super::BARRIER_GROUP_BUFFER, 1));
        // SAFETY: unpacked pointers point into `w` with valid aligned structs.
        let got_g = unsafe { *unpacked[0].barriers.cast::<super::GlobalBarrierDesc>() };
        let got_b = unsafe { *unpacked[1].barriers.cast::<super::BufferBarrierDesc>() };
        assert_eq!(got_g.sync_before, 0x80);
        assert_eq!(got_b.resource as usize, 0x1234);
        assert_eq!(got_b.offset, 0);

        // Unknown group kind marks opaque = 1 so translation falls back conservatively.
        let bad = [super::BarrierGroup {
            kind: 99,
            count: 1,
            barriers: (&raw const g_bar).cast(),
        }];
        f.put_barrier_groups(1, bad.as_ptr().cast());
        let s2 = std::mem::take(&mut f.words);
        assert_eq!(s2[0] & 0xffff, BARRIER_GROUPS);
        assert_eq!((s2[1], s2[2]), (1, 1));
    }

    /// DXR operations are recorded directly into each list's local thin stream
    /// (preserving FIFO order across `hand_over` and clearing cleanly on `leave`).
    #[test]
    fn dxr_ops_round_trip_per_list_without_global_mutex() {
        let mut f = Front::default();
        let rec = Arc::new(Recording::new(0));
        f.recording = Some(rec.clone());
        f.put_dxr_op(crate::dxr::DxrOp::CopyAs {
            dest_va: 0x1000,
            source_va: 0x2000,
            mode: 0,
        });
        f.put_dxr_op(crate::dxr::DxrOp::ExtractStaticBlas {
            dest_va: 0x3000,
            size_bytes: 4096,
        });
        assert_eq!(f.words, vec![DXR_OP, DXR_OP]);
        assert_eq!(f.dxr_ops.len(), 2);

        // `leave()` clears pending ops cleanly without leaking or panicking.
        f.leave();
        assert!(f.words.is_empty());
        assert!(f.dxr_ops.is_empty());
        assert!(rec.abandoned.load(Acquire));

        // Closed front drops incoming DXR ops cleanly.
        f.recording = Some(Arc::new(Recording::new(0)));
        f.closed = true;
        f.put_dxr_op(crate::dxr::DxrOp::CopyAs {
            dest_va: 0x4000,
            source_va: 0x5000,
            mode: 1,
        });
        assert!(f.words.is_empty());
        assert!(f.dxr_ops.is_empty());
    }
}
