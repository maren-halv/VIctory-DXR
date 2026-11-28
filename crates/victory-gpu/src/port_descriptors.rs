//! Renderer port #2: the title's writes of descriptor records into its
//! shader-visible heap say which slots they wrote, and the per-frame look at
//! every changed heap page goes (`VICTORY_PORT_DESCRIPTORS=1`, off by
//! default; read in [`on`]).
//!
//! What the title does (`victory-loader`'s
//! `descriptor_sites` has the code): its placed shader-visible view heap is
//! the memory of its frame allocator, which hands out the
//! draws' and dispatches' descriptor tables and much else (constants, list
//! nodes). The tables are filled one 32-byte record at a time by six stores
//! in two functions (+0x4d9efd0, +0x4eec8e0); views made by our calls
//! (`Create*View`, `CopyDescriptors*`) are mirrored where they are made
//! (`descriptor::mirror`). What the layer did before: the write watch's
//! changed 64 KiB pages of the heap, every slot on them read and compared
//! with the one last seen (~290 pages, ~600,000 slots, 15-16 ms a frame of
//! the heap-sync thread, 5-7 of it asking the watch, run port1-3).
//!
//! What is done here: the six stores are patched (`descriptor_sites`) to set
//! a byte per slot and per 128 slots in maps of ours ([`MARKS`]), and our own
//! mirroring marks too; `descriptor::sync_heaps` takes the marked slots
//! ([`take`]) in place of the changed pages, and compares and makes only
//! those, as before.
//!
//! Correctness first, the fallback: once a frame a few of the pages the
//! write watch says were written ([`check_budget`], every
//! [`check_every`] frames asked) are looked at whole, as before; a slot found
//! with a new view record that no mark told of is made (so it is right from
//! then on) and counted, with the place in the title's code that last
//! faulted on its 64 KiB block when the guards saw one ([`faulted`]): a writer
//! the patches miss shows in the log. Records that are not views (the
//! allocator's other data) are made as before and counted apart.
//!
//! Assumption: a slot whose view record the title overwrote with other data
//! (not a view) and no mark told of keeps showing its old view on the host
//! until the fallback looks at its page; nothing reads such a slot as a
//! descriptor (the title wrote no descriptor there).

use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering::{AcqRel, Acquire, Relaxed, Release}};

/// Whether renderer port #2 is on.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Read once, here: the switch of renderer port #2.
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_DESCRIPTORS").is_ok_and(|v| v.trim() == "1"))
}

/// What the patched stores read (`victory_loader::descriptor_sites`,
/// `MARKS_LEN` bytes): the heap's first byte, its length (0: none), the map
/// of slots, the map of chunks of [`CHUNK`] slots.
#[repr(C)]
pub struct Marks {
    start: AtomicU64,
    len: AtomicU64,
    slots: AtomicPtr<AtomicU8>,
    chunks: AtomicPtr<AtomicU8>,
}

const _: () = assert!(size_of::<Marks>() == 32);

pub static MARKS: Marks = Marks {
    start: AtomicU64::new(0),
    len: AtomicU64::new(0),
    slots: AtomicPtr::new(std::ptr::null_mut()),
    chunks: AtomicPtr::new(std::ptr::null_mut()),
};

/// Slots a byte of the map of chunks stands for (4 KiB of records).
const CHUNK: usize = 128;

/// Where the stores' cells are, for `descriptor_sites::apply`.
pub fn marks_address() -> usize {
    std::ptr::from_ref(&MARKS) as usize
}

/// The title's image (base, size) and how many of its stores were patched.
static IMAGE: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// Told by the launcher: `patched` of the title's record stores in the image
/// at `base` (`size` bytes) mark their slots.
pub fn hooked(base: usize, size: usize, patched: usize) {
    IMAGE[0].store(base, Relaxed);
    IMAGE[1].store(size, Relaxed);
    IMAGE[2].store(patched, Relaxed);
}

/// Whether [`take`] stands for the heap's changed pages: the switch is on and
/// the stores were patched.
fn working() -> bool {
    on() && IMAGE[2].load(Relaxed) > 0
}

/// The heap whose slots are marked: its slots, and the places in the title's
/// code that last faulted on each of its 64 KiB blocks.
///
/// Both `slots` and `chunks` are backed by 8-byte-aligned `[AtomicU64]` arrays
/// so `take()` consumes 8 slots / 8 chunks per `AtomicU64::swap(0, AcqRel)`
/// while patched title assembly continues issuing unconditional 1-byte
/// `mov byte [base + idx], 1` stores (TSO store-store ordered with `chunks[]`).
struct Adopted {
    start: u64,
    count: usize,
    chunk_count: usize,
    slots: &'static [AtomicU64],
    chunks: &'static [AtomicU64],
    writers: &'static [AtomicUsize],
}

static ADOPTED: std::sync::RwLock<Option<Adopted>> = std::sync::RwLock::new(None);

/// A placed shader-visible view heap of `count` records at `start`: the
/// first such is the one whose slots are marked. Whether it is.
pub(crate) fn adopt(start: u64, count: u32) -> bool {
    working() && adopt_now(start, count)
}

fn adopt_now(start: u64, count: u32) -> bool {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.is_some() {
        tracing::info!(start = format_args!("{start:#x}"), count, "renderer port 2: another placed view heap, looked at by its pages as before");
        return false;
    }
    let count = count as usize;
    let slot_words = count.div_ceil(8);
    let chunk_count = count.div_ceil(CHUNK);
    let chunk_words = chunk_count.div_ceil(8);
    let leak_u64 = |n: usize| -> &'static [AtomicU64] { Box::leak((0..n).map(|_| AtomicU64::new(0)).collect()) };
    // Keep leaked mark arrays monotonically non-shrinking across re-adoptions
    // so concurrent readers observing a larger `MARKS.len` never write OOB.
    static LEAKED: std::sync::Mutex<Option<(&'static [AtomicU64], &'static [AtomicU64])>> = std::sync::Mutex::new(None);
    let (slots, chunks) = {
        let mut kept = LEAKED.lock().unwrap_or_else(|e| e.into_inner());
        match *kept {
            Some((s, c)) if s.len() >= slot_words && c.len() >= chunk_words => {
                for a in &s[..slot_words] {
                    a.store(0, Relaxed);
                }
                for a in &c[..chunk_words] {
                    a.store(0, Relaxed);
                }
                (&s[..slot_words], &c[..chunk_words])
            }
            _ => {
                let s = leak_u64(slot_words);
                let c = leak_u64(chunk_words);
                *kept = Some((s, c));
                (s, c)
            }
        }
    };
    let writers: &'static [AtomicUsize] = Box::leak((0..(count * 32).div_ceil(1 << 16)).map(|_| AtomicUsize::new(0)).collect());
    MARKS.slots.store(slots.as_ptr().cast::<AtomicU8>().cast_mut(), Release);
    MARKS.chunks.store(chunks.as_ptr().cast::<AtomicU8>().cast_mut(), Release);
    MARKS.start.store(start, Release);
    // Last: a store that sees a length sees the rest.
    MARKS.len.store(count as u64 * 32, Release);
    *adopted = Some(Adopted { start, count, chunk_count, slots, chunks, writers });
    tracing::info!(
        start = format_args!("{start:#x}"),
        count,
        stores = IMAGE[2].load(Relaxed),
        "renderer port 2: the title's view heap's slots are told by its record stores and our calls, not looked for on its pages (VICTORY_PORT_DESCRIPTORS)"
    );
    true
}

/// The heap at `start` goes: nothing more is marked for it.
pub(crate) fn let_go(start: u64) {
    let mut adopted = ADOPTED.write().unwrap_or_else(|e| e.into_inner());
    if adopted.as_ref().is_some_and(|a| a.start == start) {
        MARKS.len.store(0, Release);
        *adopted = None;
        tracing::info!(start = format_args!("{start:#x}"), "renderer port 2: the marked view heap went");
    }
}

/// Whether the heap at `start` is the one whose slots are marked.
pub(crate) fn adopted(start: u64) -> bool {
    ADOPTED.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|a| a.start == start)
}

/// Marks every touched slot in `[dst, dst + len)` of the marked descriptor heap
/// in bulk (word-granular `AtomicU64` writes on `MARKS.slots` followed by
/// `MARKS.chunks`, preserving x86-64 TSO / `Release` ordering with [`take`]).
/// Returns the number of slots marked (`0` if outside the marked heap or empty).
#[inline]
pub fn mark_range(dst: u64, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let heap_len = MARKS.len.load(Acquire);
    if heap_len == 0 {
        return 0;
    }
    let start = MARKS.start.load(Relaxed);
    let end = dst.saturating_add(len as u64);
    // Fast-path unsigned range check avoids work on non-heap copies.
    if dst.wrapping_sub(start) >= heap_len && !(dst < start && end > start) {
        return 0;
    }
    let lo = dst.max(start);
    let hi = end.min(start.saturating_add(heap_len));
    if lo >= hi {
        return 0;
    }
    let first_slot = ((lo - start) / 32) as usize;
    let last_slot = ((hi - 1 - start) / 32) as usize;
    let first_chunk = first_slot / CHUNK;
    let last_chunk = last_slot / CHUNK;
    // SAFETY: `MARKS.slots` and `MARKS.chunks` point to leaked 8-byte-aligned
    // `[AtomicU64]` arrays covering at least `MARKS.len / 32` slots and their
    // chunks; `slots` are always marked before `chunks` so concurrent `take()`
    // never misses a newly marked slot.
    unsafe {
        let slots_base = MARKS.slots.load(Relaxed).cast::<AtomicU64>();
        let chunks_base = MARKS.chunks.load(Relaxed).cast::<AtomicU64>();
        mark_u64_range(slots_base, first_slot, last_slot);
        mark_u64_range(chunks_base, first_chunk, last_chunk);
    }
    last_slot - first_slot + 1
}

/// Bulk-marks byte lanes `first..=last` (inclusive) to `1` in an 8-byte-aligned
/// `[AtomicU64]` mark array without clobbering sibling bytes during concurrent
/// `AtomicU64::swap(0, AcqRel)` in [`take`].
///
/// # Safety
/// `base` must point to a valid `[AtomicU64]` slice covering word `last >> 3`.
#[inline]
unsafe fn mark_u64_range(base: *const AtomicU64, first: usize, last: usize) {
    const ALL_ONES: u64 = 0x0101_0101_0101_0101;
    let w_first = first >> 3;
    let w_last = last >> 3;
    let lane_first = first & 7;
    let lane_last = last & 7;
    if w_first == w_last {
        let mut mask = 0u64;
        for b in lane_first..=lane_last {
            mask |= 1u64 << (b * 8);
        }
        // SAFETY: caller guarantees `w_first` is in bounds.
        unsafe { (*base.add(w_first)).fetch_or(mask, Release) };
        return;
    }
    let mut w = w_first;
    if lane_first != 0 {
        let mut mask = 0u64;
        for b in lane_first..8 {
            mask |= 1u64 << (b * 8);
        }
        // SAFETY: `w` is in bounds.
        unsafe { (*base.add(w)).fetch_or(mask, Release) };
        w += 1;
    }
    let full_end = if lane_last == 7 { w_last + 1 } else { w_last };
    while w < full_end {
        // Setting all 8 byte lanes to `1` can never clear a concurrent mark in
        // any sibling lane and matches `memset(1)` throughput at 8 slots/store.
        // SAFETY: `w` is in bounds.
        unsafe { (*base.add(w)).store(ALL_ONES, Release) };
        w += 1;
    }
    if lane_last != 7 {
        let mut mask = 0u64;
        for b in 0..=lane_last {
            mask |= 1u64 << (b * 8);
        }
        // SAFETY: `w_last` is in bounds.
        unsafe { (*base.add(w_last)).fetch_or(mask, Release) };
    }
}

/// Marks the slot at `handle` (a record our own calls wrote), if it is in the
/// marked heap.
#[inline]
pub(crate) fn mark(handle: u64) {
    let len = MARKS.len.load(Acquire);
    if len == 0 {
        return;
    }
    let at = handle.wrapping_sub(MARKS.start.load(Relaxed));
    if at >= len {
        return;
    }
    let slot = (at / 32) as usize;
    mark_slot(slot);
    MARKED_BY_US.fetch_add(1, Relaxed);
}

/// Whether slot `index`'s mark is set (told, not yet taken).
fn marked(a: &Adopted, index: u32) -> bool {
    let idx = index as usize;
    idx < a.count
        && a.slots
            .get(idx >> 3)
            .is_some_and(|w| (w.load(Acquire) >> ((idx & 7) * 8)) & 0xff != 0)
}

/// The marked slots of the heap at `start`, each mark cleared, as runs of
/// slots `(first, count)`; `None` if it is not the marked heap.
pub(crate) fn take(start: u64) -> Option<Vec<(u32, u32)>> {
    let began = std::time::Instant::now();
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let a = adopted.as_ref().filter(|a| a.start == start)?;
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut n = 0u64;
    const WORDS_PER_CHUNK: usize = CHUNK / 8;
    for (cw, chunk_word) in a.chunks.iter().enumerate() {
        if chunk_word.load(Relaxed) == 0 {
            continue;
        }
        // Cleared before its slots are read: a mark set meanwhile sets its
        // chunk byte unconditionally after its slot byte (x86-64 TSO).
        let mut cmask = chunk_word.swap(0, AcqRel);
        while cmask != 0 {
            let lane = (cmask.trailing_zeros() / 8) as usize;
            cmask &= !(0xffu64 << (lane * 8));
            let c = cw * 8 + lane;
            if c >= a.chunk_count {
                break;
            }
            let first_word = c * WORDS_PER_CHUNK;
            let end_word = (first_word + WORDS_PER_CHUNK).min(a.slots.len());
            for (wi, slot_word) in a.slots[first_word..end_word].iter().enumerate() {
                if slot_word.load(Relaxed) == 0 {
                    continue;
                }
                let mut smask = slot_word.swap(0, AcqRel);
                let base = (first_word + wi) * 8;
                while smask != 0 {
                    let b = (smask.trailing_zeros() / 8) as usize;
                    smask &= !(0xffu64 << (b * 8));
                    let slot_idx = base + b;
                    if slot_idx >= a.count {
                        break;
                    }
                    let index = slot_idx as u32;
                    n += 1;
                    match runs.last_mut() {
                        Some((at, len)) if *at + *len == index => *len += 1,
                        _ => runs.push((index, 1)),
                    }
                }
            }
        }
    }
    TAKEN[0].fetch_add(n, Relaxed);
    TAKEN[1].fetch_add(1, Relaxed);
    TAKEN[2].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    Some(runs)
}

/// A write fault of the title's at `at` (from its instruction at `rip`),
/// if in the marked heap: who wrote its block last, for [`missed`].
#[allow(dead_code)]
pub(crate) fn faulted(at: usize, rip: usize) {
    if MARKS.len.load(Relaxed) == 0 {
        return;
    }
    let Ok(adopted) = ADOPTED.try_read() else { return };
    if let Some(a) = adopted.as_ref() {
        let off = (at as u64).wrapping_sub(a.start);
        if let Some(w) = a.writers.get((off >> 16) as usize) {
            w.store(rip, Relaxed);
        }
    }
}

/// Maximum adaptive fallback cadence (frames between write-watch checks) when
/// zero missed descriptor writes are observed.
const MAX_ADAPTIVE_EVERY: u64 = 32;

/// Explicit fixed fallback cadence override (`VICTORY_PORT_DESCRIPTORS_EVERY=<frames>`),
/// or `None` when adaptive throttling (`1 -> 2 -> 4 -> 8 -> 16 -> 32`) is active.
fn check_every_override() -> Option<u64> {
    static N: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_PORT_DESCRIPTORS_EVERY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|n| n.max(1))
    })
}

/// Current adaptive fallback cadence (1..=32 frames), immediately reset to `1`
/// whenever [`missed`] detects any unmarked view record.
static ADAPTIVE_EVERY: AtomicU64 = AtomicU64::new(1);

/// Effective frames between write-watch fallback checks (`check_every`).
pub(crate) fn current_check_every() -> u64 {
    check_every_override().unwrap_or_else(|| ADAPTIVE_EVERY.load(Acquire).clamp(1, MAX_ADAPTIVE_EVERY))
}

/// Pages the fallback looks at a frame (`VICTORY_PORT_DESCRIPTORS_CHECK=<pages>`,
/// 16 if not given; 0: none).
fn check_budget() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_PORT_DESCRIPTORS_CHECK").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(16usize))
}

/// The fallback's pages: those the watch said were written, not looked at yet;
/// the frame it last asked in, the frame it last looked in, and the cumulative
/// `MISSED_ALL` count snapshot at the previous ask (for adaptive throttling).
struct Fallback {
    pages: std::collections::BTreeSet<usize>,
    asked: u64,
    looked: u64,
    /// The page it goes on from (round the heap).
    next: usize,
    last_missed_all: u64,
}

static FALLBACK: std::sync::Mutex<Fallback> = std::sync::Mutex::new(Fallback {
    pages: std::collections::BTreeSet::new(),
    asked: 0,
    looked: 0,
    next: 0,
    last_missed_all: 0,
});

/// Whether to ask the write watch of the marked heap this frame.
/// Adaptively throttles from `1` up to [`MAX_ADAPTIVE_EVERY`] (`32`) frames
/// while zero missed view writes are observed, immediately resetting to `1`
/// if any missed write is detected.
pub(crate) fn ask_due(frame: u64) -> bool {
    let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
    let every = current_check_every();
    if check_budget() == 0 || (f.asked != 0 && frame < f.asked + every) {
        return false;
    }
    if check_every_override().is_none() {
        let cur_missed = MISSED_ALL.load(Acquire);
        if cur_missed > f.last_missed_all {
            ADAPTIVE_EVERY.store(1, Release);
            f.last_missed_all = cur_missed;
        } else if f.asked != 0 && (f.looked >= f.asked || f.pages.is_empty()) {
            let next = (every.saturating_mul(2)).min(MAX_ADAPTIVE_EVERY);
            ADAPTIVE_EVERY.store(next, Release);
        } else if f.asked == 0 {
            f.last_missed_all = cur_missed;
        }
    }
    f.asked = frame.max(1);
    true
}

/// What the watch said was written (byte ranges), for later.
pub(crate) fn asked(ranges: &[(usize, usize)]) {
    let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
    for &(from, to) in ranges {
        let mut page = from & !(crate::watch::PAGE - 1);
        while page < to {
            f.pages.insert(page);
            page += crate::watch::PAGE;
        }
    }
}

/// The pages to look at whole this frame (once a frame; none after).
pub(crate) fn to_check(frame: u64) -> Vec<usize> {
    let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
    if f.looked == frame {
        return Vec::new();
    }
    f.looked = frame;
    let mut out = Vec::new();
    while out.len() < check_budget() {
        let from = f.next;
        let Some(&page) = f.pages.range(from..).next().or_else(|| f.pages.iter().next()) else {
            break;
        };
        f.pages.remove(&page);
        f.next = page + crate::watch::PAGE;
        out.push(page);
    }
    CHECKED[0].fetch_add(out.len() as u64, Relaxed);
    out
}

/// Returns unvisited fallback pages back into the pending set when
/// maintenance yields early to a waiting submission (`CPU-10`).
pub(crate) fn put_back_pages(pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
    for &page in pages {
        f.pages.insert(page);
    }
    if let Some(&first) = pages.first() {
        f.next = first;
    }
}

/// The fallback found slot `index` (at `handle`) with a new record: a view
/// record that no mark told of is counted and said.
pub(crate) fn missed(index: u32, handle: u64, view: bool, tabled: bool) {
    if !view {
        CHECKED[2].fetch_add(1, Relaxed);
        return;
    }
    if tabled {
        // A draw read it through a table lately: the draw makes it anyway.
        CHECKED[5].fetch_add(1, Relaxed);
    }
    let adopted = ADOPTED.read().unwrap_or_else(|e| e.into_inner());
    let Some(a) = adopted.as_ref() else { return };
    if marked(a, index) {
        // Told, not taken yet.
        CHECKED[3].fetch_add(1, Relaxed);
        return;
    }
    CHECKED[1].fetch_add(1, Relaxed);
    let n = MISSED_ALL.fetch_add(1, AcqRel);
    // Any missed view record immediately resets adaptive fallback cadence to 1 frame.
    ADAPTIVE_EVERY.store(1, Release);
    #[cfg(feature = "diagnostics")]
    {
        let rip = a.writers.get(((handle - a.start) >> 16) as usize).map_or(0, |w| w.load(Relaxed));
        let (base, size) = (IMAGE[0].load(Relaxed), IMAGE[1].load(Relaxed));
        let writer = if rip >= base && rip < base + size {
            format!("title+{:#x}", rip - base)
        } else if rip != 0 {
            format!("{rip:#x} (not the title's)")
        } else {
            "not known (no fault on its block)".into()
        };
        {
            let mut by = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
            *by.entry(writer.clone()).or_default() += 1;
        }
        if n < 20 || n.is_power_of_two() {
            tracing::warn!(
                slot = index,
                handle = format_args!("{handle:#x}"),
                %writer,
                "renderer port 2: a view record in the title's heap that no store told of (found by the fallback; made now)"
            );
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (handle, n);
}

/// The runtime's own `memcpy`, `memmove`, and `memset` ([`import`]).
static REAL: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

/// Since the last line: the title's copies/fills into the marked heap, slots they marked.
static COPIES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether `memcpy`/`memmove`/`memset` imports are hooked with a fast range filter
/// (`dst.wrapping_sub(start) < len`) to bulk-mark touched slots via [`mark_range`]
/// (`VICTORY_PORT_DESCRIPTORS_MEMCPY=0` disables; default ON).
fn hook_memcpy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_PORT_DESCRIPTORS_MEMCPY").is_none_or(|v| v != "0")
    })
}

/// What the title's import of `name` (bound to `address`) is bound to: with
/// `VICTORY_PORT_DESCRIPTORS=1` (and unless `VICTORY_PORT_DESCRIPTORS_MEMCPY=0`),
/// its `memcpy`, `memmove`, and `memset` bulk-mark touched descriptor-heap slots.
pub fn import(name: &str, address: usize) -> usize {
    let (which, ours) = match name {
        "memcpy" => (0, copy as *const () as usize),
        "memmove" => (1, shift as *const () as usize),
        "memset" => (2, fill as *const () as usize),
        _ => return address,
    };
    if !on() || !hook_memcpy() {
        return address;
    }
    REAL[which].store(address, Relaxed);
    ours
}

type Copy = unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8;
type Fill = unsafe extern "system-unwind" fn(*mut u8, i32, usize) -> *mut u8;

unsafe extern "system-unwind" fn copy(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: the title's copy, by the function it asked for.
    unsafe { copied(to, from, n, REAL[0].load(Relaxed)) }
}

unsafe extern "system-unwind" fn shift(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
    // SAFETY: as above.
    unsafe { copied(to, from, n, REAL[1].load(Relaxed)) }
}

unsafe extern "system-unwind" fn fill(to: *mut u8, val: i32, n: usize) -> *mut u8 {
    // SAFETY: the title's memset, by the function it asked for.
    unsafe { filled(to, val, n, REAL[2].load(Relaxed)) }
}

/// # Safety
/// `real` copies as `memcpy` does; the rest are the title's.
#[inline]
unsafe fn copied(to: *mut u8, from: *const u8, n: usize, real: usize) -> *mut u8 {
    // SAFETY: the host's function, noted by `import` before the title ran.
    let real: Copy = unsafe { std::mem::transmute(real) };
    // SAFETY: by the caller.
    let out = unsafe { real(to, from, n) };
    let marked = mark_range(to as u64, n);
    if marked > 0 {
        COPIES[0].fetch_add(1, Relaxed);
        COPIES[1].fetch_add(marked as u64, Relaxed);
    }
    out
}

/// # Safety
/// `real` fills as `memset` does; the rest are the title's.
#[inline]
unsafe fn filled(to: *mut u8, val: i32, n: usize, real: usize) -> *mut u8 {
    // SAFETY: the host's function, noted by `import` before the title ran.
    let real: Fill = unsafe { std::mem::transmute(real) };
    // SAFETY: by the caller.
    let out = unsafe { real(to, val, n) };
    let marked = mark_range(to as u64, n);
    if marked > 0 {
        COPIES[0].fetch_add(1, Relaxed);
        COPIES[1].fetch_add(marked as u64, Relaxed);
    }
    out
}

/// Marks slot `slot` of the marked heap (in it: checked by the caller) using
/// `AtomicU64::fetch_or` on the 8-byte-aligned backing arrays so concurrent
/// `AtomicU64::swap(0, AcqRel)` consumers in [`take`] never drop sibling bytes.
#[inline]
fn mark_slot(slot: usize) {
    let slot_word = slot >> 3;
    let slot_bit = 1u64 << ((slot & 7) * 8);
    let chunk = slot / CHUNK;
    let chunk_word = chunk >> 3;
    let chunk_bit = 1u64 << ((chunk & 7) * 8);
    // SAFETY: the maps are leaked 8-byte-aligned `[AtomicU64]` arrays covering
    // at least `MARKS.len / 32` slots and their chunks.
    unsafe {
        (*MARKS.slots.load(Relaxed).cast::<AtomicU64>().add(slot_word)).fetch_or(slot_bit, Release);
        (*MARKS.chunks.load(Relaxed).cast::<AtomicU64>().add(chunk_word)).fetch_or(chunk_bit, Release);
    }
}

/// Our own calls' marks since the last line.
static MARKED_BY_US: AtomicU64 = AtomicU64::new(0);
/// Since the last line: slots taken, takings, nanoseconds taking.
static TAKEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Since the last line, of the fallback: pages looked at whole, view records
/// no mark told of, other records changed (not views), view records told and
/// not taken yet; nanoseconds; and of the view records no mark told of, those
/// of draws' tables.
pub(crate) static CHECKED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Since the start: view records the fallback found that no mark told of.
static MISSED_ALL: AtomicU64 = AtomicU64::new(0);
/// Since the start: the missed records by the place that last faulted on their block.
#[cfg(feature = "diagnostics")]
static WRITERS: std::sync::Mutex<std::collections::BTreeMap<String, u64>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
/// Since the start: slots taken, of them marked by our calls; pages checked.
static ALL: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// With the frame line: what port #2 did per frame over `frames`.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [taken, takings, ns] = [0, 1, 2].map(|i| TAKEN[i].swap(0, Relaxed));
    let ours = MARKED_BY_US.swap(0, Relaxed);
    let [pages, missed, other, pending, check_ns, tabled] = [0, 1, 2, 3, 4, 5].map(|i| CHECKED[i].swap(0, Relaxed));
    let copies = [0, 1].map(|i| COPIES[i].swap(0, Relaxed));
    ALL[0].fetch_add(taken, Relaxed);
    ALL[1].fetch_add(ours, Relaxed);
    ALL[2].fetch_add(pages, Relaxed);
    if !working() {
        tracing::info!("renderer port 2 (VICTORY_PORT_DESCRIPTORS=1): not working, no store of the title's was patched");
        return;
    }
    #[cfg(feature = "diagnostics")]
    let writers = {
        let by = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<(&String, &u64)> = by.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.iter().take(6).map(|(w, n)| format!("{w} {n}")).collect::<Vec<_>>().join(", ")
    };
    #[cfg(not(feature = "diagnostics"))]
    let writers = String::new();
    tracing::info!(
        "renderer port 2 (VICTORY_PORT_DESCRIPTORS=1) per frame: {:.0} slots told (of them {:.0} marked by our calls, the rest by the title's {} patched stores) in {:.1} takings, {:.3} ms taking; \
         of the title's memcpy/memmove/memset calls into the heap {:.0}, {:.0} slots they marked; fallback (every {}f): {:.1} pages looked at whole, {:.2} view records no store told of ({:.2} of them of draws' tables), {:.0} other records changed, {:.1} told and not taken yet, {:.2} ms; \
         since the start: {} slots told ({} by our calls), {} pages looked at, {} view records missed; their writers: {}",
        taken as f64 / frames,
        ours as f64 / frames,
        IMAGE[2].load(Relaxed),
        takings as f64 / frames,
        ns as f64 / 1e6 / frames,
        copies[0] as f64 / frames,
        copies[1] as f64 / frames,
        current_check_every(),
        pages as f64 / frames,
        missed as f64 / frames,
        tabled as f64 / frames,
        other as f64 / frames,
        pending as f64 / frames,
        check_ns as f64 / 1e6 / frames,
        ALL[0].load(Relaxed),
        ALL[1].load(Relaxed),
        ALL[2].load(Relaxed),
        MISSED_ALL.load(Relaxed),
        if writers.is_empty() { "none".into() } else { writers }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marks_layout_is_what_the_stores_read() {
        assert_eq!(size_of::<Marks>(), victory_loader_marks_len());
        assert_eq!(std::mem::offset_of!(Marks, start), 0);
        assert_eq!(std::mem::offset_of!(Marks, len), 8);
        assert_eq!(std::mem::offset_of!(Marks, slots), 16);
        assert_eq!(std::mem::offset_of!(Marks, chunks), 24);
    }

    /// `victory_loader::descriptor_sites::MARKS_LEN` (no dependency on the loader).
    fn victory_loader_marks_len() -> usize {
        32
    }

    /// Slots marked as the stores mark them (1-byte stores), and by our calls
    /// (`AtomicU64::fetch_or`), come out once each via `AtomicU64::swap(0, AcqRel)`
    /// across byte lanes, chunk boundaries, and non-8-multiple heap counts.
    #[test]
    fn marked_slots_are_taken_once_as_runs() {
        // Serialize tests touching the global ADOPTED/MARKS state.
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let start = 0x1000_0000u64;
        // Non-8-multiple count (1003) verifies tail clamping on the last AtomicU64 word.
        assert!(adopt_now(start, 1003));
        assert!(adopted(start) && !adopted(start + 32));
        // Simulate patched assembly's unconditional 1-byte `mov byte [base+idx], 1` stores.
        let asm_store = |slot: usize| {
            // SAFETY: MARKS pointers point to 8-byte-aligned `[AtomicU64]` backing buffers.
            unsafe {
                (*MARKS.slots.load(Acquire).add(slot)).store(1, Release);
                (*MARKS.chunks.load(Acquire).add(slot / CHUNK)).store(1, Release);
            }
        };
        assert_eq!(MARKS.slots.load(Acquire) as usize % 8, 0, "slots must be 8-byte aligned");
        assert_eq!(MARKS.chunks.load(Acquire) as usize % 8, 0, "chunks must be 8-byte aligned");
        asm_store(5);
        mark(start + 6 * 32 + 7);
        mark(start + 7 * 32);
        asm_store(8); // crosses AtomicU64 word boundary (7 -> 8)
        mark(start + 127 * 32);
        asm_store(128); // crosses CHUNK=128 boundary (127 -> 128)
        mark(start + 300 * 32);
        asm_store(1002); // last valid slot in tail AtomicU64 word
        mark(start - 32);
        mark(start + 1003 * 32);
        assert_eq!(
            take(start).unwrap(),
            [(5, 4), (127, 2), (300, 1), (1002, 1)]
        );
        assert_eq!(take(start).unwrap(), []);
        assert!(take(start + 32).is_none());
        let_go(start);
        mark(start + 5 * 32);
        assert!(take(start).is_none());
        assert_eq!(MARKS.len.load(Relaxed), 0);
    }

    #[test]
    fn put_back_pages_preserves_unvisited_fallback_pages() {
        let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
        f.pages.clear();
        f.looked = 0;
        f.next = 0;
        drop(f);
        asked(&[(0x10_0000, 0x13_0000)]);
        let pages = to_check(99);
        assert_eq!(pages, [0x10_0000, 0x11_0000, 0x12_0000]);
        put_back_pages(&pages[1..]);
        let again = to_check(100);
        assert_eq!(again, [0x11_0000, 0x12_0000]);
    }

    #[test]
    fn mark_range_marks_exact_slots_and_chunks_and_hooks_memcpy_memset() {
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut heap = vec![0u8; 400 * 32];
        let start = heap.as_mut_ptr() as u64;
        assert!(adopt_now(start, 400));

        // Out-of-range and empty ranges mark 0 slots.
        assert_eq!(mark_range(start, 0), 0);
        assert_eq!(mark_range(start - 64, 64), 0);
        assert_eq!(mark_range(start + 400 * 32, 64), 0);

        // Partial unaligned range inside slot 3..=4 (bytes 3*32+12 .. 4*32+4).
        assert_eq!(mark_range(start + 3 * 32 + 12, 24), 2);
        // Cross-u64-word AND cross-CHUNK=128 range (slots 120..=140 = 21 slots).
        assert_eq!(mark_range(start + 120 * 32, 21 * 32), 21);
        // Clamped range starting before heap and entering slots 0..=1.
        assert_eq!(mark_range(start - 16, 16 + 33), 2);
        // Clamped range extending past end of heap (slots 398..=399).
        assert_eq!(mark_range(start + 398 * 32, 128), 2);

        assert_eq!(
            take(start).unwrap(),
            [(0, 2), (3, 2), (120, 21), (398, 2)]
        );
        assert_eq!(take(start).unwrap(), []);

        // Hooked `copied` (memcpy) and `filled` (memset) mark touched slots even
        // when payload is NOT `VIEW_TAG` (e.g. non-view overwrites or zeroing).
        unsafe extern "system-unwind" fn fake_memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8 {
            // SAFETY: valid test slices.
            unsafe { std::ptr::copy_nonoverlapping(src, dst, n) };
            dst
        }
        unsafe extern "system-unwind" fn fake_memset(dst: *mut u8, val: i32, n: usize) -> *mut u8 {
            // SAFETY: valid test slice.
            unsafe { std::ptr::write_bytes(dst, val as u8, n) };
            dst
        }
        let payload = [0xAAu8; 96]; // 3 slots, non-VIEW_TAG data
        // SAFETY: valid pointers into `heap` and `payload`.
        unsafe {
            copied(
                (start + 10 * 32) as *mut u8,
                payload.as_ptr(),
                payload.len(),
                fake_memcpy as *const () as usize,
            );
            filled(
                (start + 255 * 32 + 4) as *mut u8,
                0,
                64, // spans slots 255, 256, 257 (crosses chunk 1 -> chunk 2)
                fake_memset as *const () as usize,
            );
        }
        assert_eq!(take(start).unwrap(), [(10, 3), (255, 3)]);
        let_go(start);
    }

    #[test]
    fn adaptive_check_every_throttles_to_32_on_zero_misses_and_resets_to_1_on_miss() {
        static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if check_every_override().is_some() {
            return; // explicit env override disables adaptive mode
        }
        let start = 0x2000_0000u64;
        assert!(adopt_now(start, 256));
        {
            let mut f = FALLBACK.lock().unwrap_or_else(|e| e.into_inner());
            f.pages.clear();
            f.asked = 0;
            f.looked = 0;
            f.next = 0;
            f.last_missed_all = MISSED_ALL.load(Acquire);
        }
        ADAPTIVE_EVERY.store(1, Release);
        assert_eq!(current_check_every(), 1);

        // Initial ask on frame 1 -> still 1; after clean check, ramps 1 -> 2 -> 4 -> 8 -> 16 -> 32.
        assert!(ask_due(1));
        assert_eq!(current_check_every(), 1);
        let _ = to_check(1);

        let mut frame = 2u64;
        for expected_next in [2u64, 4, 8, 16, 32, 32] {
            assert!(ask_due(frame), "expected ask_due({frame}) to trigger");
            assert_eq!(current_check_every(), expected_next);
            let _ = to_check(frame);
            // Frames strictly inside `frame + expected_next` must be throttled.
            assert!(!ask_due(frame + 1));
            assert!(!ask_due(frame + expected_next - 1));
            frame += expected_next;
        }
        assert_eq!(current_check_every(), 32);

        // Any missed view record immediately resets cadence back to 1 frame.
        missed(7, start + 7 * 32, true, false);
        assert_eq!(current_check_every(), 1);
        // Very next frame after last ask (`frame - 32 + 1`) is immediately due!
        assert!(ask_due(frame - 32 + 1));
        let_go(start);
    }
}
