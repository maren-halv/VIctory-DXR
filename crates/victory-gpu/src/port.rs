//! Renderer port #1: the title's 32-bit graphics heap as the card's own
//! memory (`VICTORY_PORT_CONSTANTS`, off by default).
//!
//! What the title does (the initialization routine reached at startup):
//! one `XMemVirtualAlloc` of the five
//! sizes it is given together (kind 0x42000: a reservation of 64 KiB
//! pages, flags 7: graphics memory below 4 GiB, 185 MB at 0x34310000 in
//! every run so far), split into five placeholders (`VirtualFree` with
//! `MEM_PRESERVE_PLACEHOLDER`), each turned into graphics memory of its own
//! protection (`XMemVirtualAlloc`, kind 0x20406000, flags 0x27); then ONE
//! `XMemAllocatePhysicalPages` for all their pages and five
//! `XMemMapPhysicalPages`, in order, each from the next page of that one
//! array. `xmem` makes one section of all the pages, so the whole
//! reservation is one stretch of one section: views that follow one
//! another in memory and in the section alike. The title's 174 MB buffer at
//! 0x34e10000 (where its ring of constants lies, and what the skinning
//! passes write through one view of all of it) is in it; it is never
//! tile-mapped (`CopyPageMappings` maps the GPU heap's pages, not these).
//!
//! What the layer did with it: a host copy in the card's memory, the title's
//! writes found and sent up (17 MB a frame in ~600 pieces), what shaders
//! wrote fetched back and merged, and every constant buffer read out of it
//! (~6,000 fills a frame) either waited for flights or done on the card
//! from the copy.
//!
//! What is done here instead: the stretch is mapped once more, at an
//! address of ours (a view of the same section: the same physical pages),
//! and opened as a host heap there (`OpenExistingHeapFromAddress1`); title
//! buffers and windows of title memory in it are buffers placed in that
//! heap ([`crate::shadow::Shadow`]'s in-place form): the card reads and
//! writes the title's bytes where the title wrote them. Nothing is tracked,
//! sent up, fetched back or compared for it, and no compare or hash pass
//! of ours ever runs over it (they are for copies in the card's memory:
//! one such pass over memory shared with the PC removed the device on
//! another tree). No tile mapping is ever made into this heap (that removes
//! the device).
//!
//! Why our own view: a heap opened from an address holds the pages it was
//! opened on: the title could not unmap, decommit or protect them while it
//! stands. Opened on our view, the title's own views stay free to change.
//! When the title maps other pages at such an address, what is in place
//! there is let go (`crate::inplace::title_memory_freeing`) and the region
//! forgotten; the view of ours stays (shortcut: it is never unmapped, a
//! few hundred MB of address space at most).
//!
//! What it assumes: the CPU steps of ours that read this memory at
//! submission (fills of constant buffers, the title's buffer copies done on
//! the CPU) read what the title's CPU wrote, not what a submission still in
//! the air writes there: in place nothing tells the flights' writes, so
//! nothing waits for them. The title's constants are its CPU's; how many
//! such steps there are is on the frame-rate line ([`log`]).

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{Cover, OwnedView, Placed, cover, stretch};

const K64: u64 = 64 << 10;

/// The ported memory lies below this: the title's graphics memory for
/// 32-bit addresses, which only that heap is.
#[allow(dead_code)]
const LOW: u64 = 1 << 32;

/// The most bytes of one region: its pages are locked into the PC's memory
/// while the heap stands.
#[allow(dead_code)]
const MOST: u64 = 512 << 20;

/// Upper bound on automatic stretch expansion for 64-bit `SEC_RESERVE` section
/// views so a single small buffer in a multi-GB pool never locks 512 MiB.
const EXPAND_STRETCH_CAP: u64 = 64 << 20;

/// Cached heap chunk size for 64-bit committed `MEM_PRIVATE` regions.
const PRIVATE_CHUNK: usize = 16 << 20;

/// Parses `VICTORY_PORT_EXPAND`: `0` off (default), `1`/`safe` non-VRAM 64-bit
/// buffers and shared-fallback windows, `2`/`sections` adds 64-bit section-stretch
/// buffers up to `VICTORY_PORT_EXPAND_MAX_MB` (default 32 MiB) and windows,
/// `3`/`all` (or `VICTORY_INPLACE_ALL=1`) full 64-bit coverage.
fn parse_expand_mode(raw: Option<&str>, all_on: bool) -> u8 {
    match raw.map(str::trim) {
        Some("1") | Some("safe") => 1,
        Some("2") | Some("sections") => 2,
        Some("3") | Some("all") => 3,
        Some("0") => 0,
        _ if all_on => 3,
        _ => 0,
    }
}

/// Active `VICTORY_PORT_EXPAND` level (`0..=3`).
pub(crate) fn expand_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let raw = std::env::var("VICTORY_PORT_EXPAND").ok();
        parse_expand_mode(raw.as_deref(), crate::inplace::all_on())
    })
}

/// Maximum size in bytes of a card-eligible 64-bit section-stretch buffer placed
/// in-place under `VICTORY_PORT_EXPAND=2` (`VICTORY_PORT_EXPAND_MAX_MB`, default 32 MiB
/// so 11–24 MiB CPU-written buffers go in-place while 36–90 MiB GPU-hot buffers stay in VRAM).
fn expand_max_bytes() -> u64 {
    static MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_PORT_EXPAND_MAX_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(32)
            << 20
    })
}

/// Whether `[at, at + len)` lies entirely in the title's 32-bit graphics space.
pub(crate) fn is_low(at: u64, len: u64) -> bool {
    at != 0 && len != 0 && at.checked_add(len).is_some_and(|end| end <= LOW)
}

/// `VICTORY_PORT_CONSTANTS`: `1` turns renderer port #1 on (title buffers
/// and windows of title memory in the title's 32-bit graphics memory, view
/// memory that is one stretch of a section, used in place); `VICTORY_PORT_EXPAND`
/// or `VICTORY_INPLACE_ALL=1` extends placement to 64-bit pools.
#[allow(dead_code)]
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Read once, here: the switch of renderer port #1 (or expanded/full in-place mode).
    *ON.get_or_init(|| {
        std::env::var("VICTORY_PORT_CONSTANTS").is_ok_and(|v| v.trim() == "1")
            || expand_mode() > 0
            || crate::inplace::all_on()
    })
}

/// A stretch of title memory opened as a heap (through a view of ours for
/// section memory, or directly on a committed private sub-range).
struct Region {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    /// Declared after `heap` so `UnmapViewOfFile` runs only after the heap drops.
    view: Option<Arc<OwnedView>>,
    private: bool,
}

// SAFETY: D3D12 heaps and `Arc<OwnedView>` are free-threaded.
unsafe impl Send for Region {}

static REGIONS: Mutex<Vec<Region>> = Mutex::new(Vec::new());

/// Bytes pinned in working set by address-opened `OpenExistingHeapFromAddress1`
/// regions currently cached in `REGIONS`.
static PINNED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Current pinned bytes held by `port::REGIONS`.
pub(crate) fn pinned_bytes() -> u64 {
    PINNED_BYTES.load(Ordering::Relaxed)
}

/// The lowest start and highest end of every region (both 0: none), for a
/// quick look on the paths that run thousands of times a frame.
static BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Active `port 1` in-place intervals `(start, end)` (including zero-pinned
/// committed-section placements that bypass `REGIONS`), coalesced and sorted,
/// so overlapping shadowed copies and textures invalidate properly (`ARCH-06 #4`).
static PLACED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
static PLACED_BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Monotonic epoch bumped whenever `PLACED` or `REGIONS` changes so hot callers
/// (e.g. `CLEAN_SPANS`) can validate non-overlap in O(1) without locking `REGIONS`.
static PORT_EPOCH: AtomicU64 = AtomicU64::new(1);

#[inline]
pub(crate) fn epoch() -> u64 {
    PORT_EPOCH.load(Ordering::Acquire)
}

fn placed_bounds(intervals: &[(u64, u64)]) {
    let lo = intervals.first().map_or(0, |r| r.0);
    let hi = intervals.last().map_or(0, |r| r.1);
    PLACED_BOUNDS[0].store(lo, Ordering::Relaxed);
    PLACED_BOUNDS[1].store(hi, Ordering::Relaxed);
}

fn note_placed_interval(start: u64, len: u64) {
    let Some(end) = start.checked_add(len).filter(|&e| e > start) else {
        return;
    };
    let mut intervals = PLACED.lock().unwrap_or_else(|e| e.into_inner());
    let mut pos = intervals.partition_point(|r| r.1 < start);
    let mut lo = start;
    let mut hi = end;
    let mut merge_end = pos;
    while merge_end < intervals.len() && intervals[merge_end].0 <= hi {
        lo = lo.min(intervals[merge_end].0);
        hi = hi.max(intervals[merge_end].1);
        merge_end += 1;
    }
    if merge_end == pos {
        intervals.insert(pos, (lo, hi));
    } else {
        intervals[pos] = (lo, hi);
        if merge_end > pos + 1 {
            intervals.drain(pos + 1..merge_end);
        }
    }
    let _ = &mut pos;
    placed_bounds(&intervals);
    PORT_EPOCH.fetch_add(1, Ordering::Release);
}

fn forget_placed_interval(at: u64, end: u64) {
    if PLACED_BOUNDS[1].load(Ordering::Relaxed) <= at || PLACED_BOUNDS[0].load(Ordering::Relaxed) >= end {
        return;
    }
    let mut intervals = PLACED.lock().unwrap_or_else(|e| e.into_inner());
    let mut next = Vec::with_capacity(intervals.len());
    for &(lo, hi) in intervals.iter() {
        if hi <= at || lo >= end {
            next.push((lo, hi));
        } else {
            if lo < at {
                next.push((lo, at));
            }
            if hi > end {
                next.push((end, hi));
            }
        }
    }
    *intervals = next;
    placed_bounds(&intervals);
    PORT_EPOCH.fetch_add(1, Ordering::Release);
}

/// Whether `[start, start + len)` overlaps any active cached `MEM_PRIVATE`
/// (`OpenExistingHeapFromAddress1`) region in `port::REGIONS` (Spec-Trap #8).
pub(crate) fn has_private_region_overlap(start: u64, len: u64) -> bool {
    let Some(end) = start.checked_add(len).filter(|&e| e > start) else {
        return false;
    };
    let r_hi = BOUNDS[1].load(Ordering::Relaxed);
    if r_hi > start && BOUNDS[0].load(Ordering::Relaxed) < end {
        return REGIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|r| r.private && r.end > start && r.start < end);
    }
    false
}

/// Whether `[start, start + len)` overlaps any active `port 1` in-place
/// placement or cached region (including zero-pinned committed sections).
pub(crate) fn overlaps(start: u64, len: u64) -> bool {
    let Some(end) = start.checked_add(len).filter(|&e| e > start) else {
        return false;
    };
    let p_hi = PLACED_BOUNDS[1].load(Ordering::Relaxed);
    if p_hi > start && PLACED_BOUNDS[0].load(Ordering::Relaxed) < end {
        let intervals = PLACED.lock().unwrap_or_else(|e| e.into_inner());
        let i = intervals.partition_point(|r| r.1 <= start);
        if intervals.get(i).is_some_and(|r| r.0 < end) {
            return true;
        }
    }
    let r_hi = BOUNDS[1].load(Ordering::Relaxed);
    if r_hi > start && BOUNDS[0].load(Ordering::Relaxed) < end {
        if REGIONS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|r| r.end > start && r.start < end)
        {
            return true;
        }
    }
    false
}

/// Since the start: `[0]` view regions opened, `[1]` their bytes; `[2]` title
/// buffers and windows placed; refused `[3]` not one stretch/private, `[4]` not
/// committed, `[5]` refused by host/budget, `[6]` no buffer; `[7]` regions
/// forgotten; `[8]` unused; `[9..13]` per-frame CPU fills/copies touching port;
/// `[13]` zero-pinned section-heap placements, `[14]` their bytes;
/// `[15]` cached private regions opened, `[16]` their bytes.
static STATS: [AtomicU64; 17] = [const { AtomicU64::new(0) }; 17];

/// Pure placement policy decision for `[start, start + len)`.
/// Returns `Some((allow_section, allow_private))` when `port` should attempt
/// in-place placement, or `None` to leave the caller to its normal path.
#[allow(clippy::too_many_arguments)]
fn placement_policy(
    start: u64,
    len: u64,
    window: bool,
    card_eligible: bool,
    after_card_fallback: bool,
    port_on: bool,
    expand: u8,
    max_bytes: u64,
) -> Option<(bool, bool)> {
    if start == 0 || len == 0 || start.checked_add(len).is_none() {
        return None;
    }
    if is_low(start, len) {
        if !port_on {
            return None;
        }
        return Some((true, expand >= 3));
    }
    if expand == 0 {
        return None;
    }
    if window {
        if after_card_fallback {
            // Shared-memory fallback window (BAR/VRAM declined): section-backed
            // below expand=3 so L0 fallbacks do not hammer VirtualQuery under
            // REGIONS.lock() or pin working set on low-hit MEM_PRIVATE chunks.
            Some((true, expand >= 3))
        } else if expand >= 2 {
            Some((true, expand >= 3))
        } else {
            None
        }
    } else if !card_eligible {
        // Non-card buffer (`< card_limits().0` or VRAM-refused): section-backed only
        // below expand=3 to avoid pinning/bypassing MEM_PRIVATE indirect pools.
        Some((true, expand >= 3))
    } else if expand >= 3 {
        Some((true, true))
    } else if expand == 2 && len <= max_bytes {
        // 64-bit section-stretch buffers up to `max_bytes` (e.g. 11-24 MiB CPU-written
        // upload rings in Pool A), leaving larger GPU-hot UAV heaps in local VRAM.
        Some((true, false))
    } else {
        None
    }
}

/// Whether `[at, at + len)` may be ported: the title's 32-bit graphics memory
/// under `VICTORY_PORT_CONSTANTS=1`, or any valid range under `VICTORY_PORT_EXPAND`
/// / `VICTORY_INPLACE_ALL=1`.
#[allow(dead_code)]
pub(crate) fn wanted(at: u64, len: u64) -> bool {
    if !on() || at == 0 || len == 0 {
        return false;
    }
    match at.checked_add(len) {
        Some(end) => expand_mode() > 0 || end <= LOW,
        None => false,
    }
}

/// Title memory `[start, start + len)` in place, if [`wanted`] and the
/// memory is one stretch of a section: a buffer placed in the region's
/// heap. `None` leaves the caller to do what it did before.
#[allow(dead_code)]
pub(crate) fn place_for(start: u64, len: u64, window: bool) -> Option<Placed> {
    if !wanted(start, len) {
        return None;
    }
    place_with(start, len, window, true, expand_mode() >= 3)
}

/// Policy-gated in-place placement for a title buffer (`card_eligible`: whether
/// the buffer qualifies for local-VRAM `card_for`).
pub(crate) fn place_for_buffer(start: u64, len: u64, card_eligible: bool) -> Option<Placed> {
    let (allow_section, allow_private) = placement_policy(
        start,
        len,
        false,
        card_eligible,
        false,
        on(),
        expand_mode(),
        expand_max_bytes(),
    )?;
    place_with(start, len, false, allow_section, allow_private)
}

/// Policy-gated in-place placement for a loose window (`after_card_fallback`:
/// `true` right before falling back to a L0 `shared_buffer` copy).
pub(crate) fn place_for_window(start: u64, len: u64, after_card_fallback: bool) -> Option<Placed> {
    let (allow_section, allow_private) = placement_policy(
        start,
        len,
        true,
        false,
        after_card_fallback,
        on(),
        expand_mode(),
        expand_max_bytes(),
    )?;
    // Spec-Trap #8: never open `OpenExistingHeapFromAddress1` on pages that
    // overlap a GPU-only/Mode-2 tripwire-eligible card buffer or active tripwire.
    let allow_private = allow_private && !crate::shadow::overlaps_gpu_only_or_tripwired(start, len);
    place_with(start, len, true, allow_section, allow_private)
}

fn place_with(start: u64, len: u64, window: bool, allow_section: bool, allow_private: bool) -> Option<Placed> {
    // Hold a non-blocking mapping read lock so section handles and virtual
    // stretches stay valid across `MapViewOfFile`, `OpenExistingHeapFromAddress1`,
    // and `inplace::counted` without deadlocking against `title_memory_remapping`.
    // Briefly yield on transient contention so boot-time remaps don't cause us
    // to permanently fall back to a shadowed copy for the 32-bit graphics heap.
    let mut reading = crate::watch::title_memory_reading_now();
    if reading.is_none() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5);
        while reading.is_none() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
            reading = crate::watch::title_memory_reading_now();
        }
    }
    let _reading = reading?;
    let host = crate::host::host()?;
    match place_in_with(&host.device, start, len, allow_section, allow_private) {
        Ok(placed) => {
            crate::inplace::counted(window, len);
            let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 32 || n.is_power_of_two() {
                tracing::info!(
                    start = format_args!("{start:#x}"),
                    len,
                    window,
                    "renderer port 1: title memory in place, read and written by the card where the title has it"
                );
            }
            Some(placed)
        }
        Err((at, why)) => {
            let n = STATS[at].fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(start = format_args!("{start:#x}"), len, window, why, "renderer port 1: title memory not in place, a copy as before");
            }
            None
        }
    }
}

/// [`place_for`]'s work: the region holding the pages of `[start, start +
/// len)`, opened if none does, and a buffer placed in it. `Err` with the
/// statistic to count and why.
#[allow(dead_code)]
pub(crate) fn place_in(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, (usize, String)> {
    place_in_with(device, start, len, true, false)
}

/// Extended placement accepting both section stretches and cached committed
/// `MEM_PRIVATE` sub-regions.
#[allow(dead_code)]
pub(crate) fn place_in_with(
    device: &ID3D12Device,
    start: u64,
    len: u64,
    allow_section: bool,
    allow_private: bool,
) -> Result<Placed, (usize, String)> {
    if start == 0 || len == 0 {
        return Err((3, "empty or null range".to_owned()));
    }
    let at = start & !(K64 - 1);
    let Some(end) = start.checked_add(len).and_then(|e| e.checked_next_multiple_of(K64)) else {
        return Err((3, "address overflow".to_owned()));
    };
    // Prefer zero-pinned `OpenExistingHeapFromFileMapping` when the stretch
    // belongs to a `SEC_COMMIT` section: 0 MB locked working set, no extra view.
    if allow_section {
        match crate::inplace_sections::place_section(device, start, len) {
            Ok(placed) => {
                STATS[13].fetch_add(1, Ordering::Relaxed);
                STATS[14].fetch_add(len, Ordering::Relaxed);
                note_placed_interval(start, len);
                return Ok(placed);
            }
            Err(crate::inplace_sections::SectionPlaceError::NotCommittedRun) => {}
            Err(crate::inplace_sections::SectionPlaceError::Remapped) => {
                return Err((3, "remapped concurrently during committed-section placement".to_owned()));
            }
            Err(crate::inplace_sections::SectionPlaceError::Host(e)) => {
                return Err((5, format!("committed section refused by host: {e}")));
            }
        }
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match regions.iter().position(|r| r.start <= at && end <= r.end) {
        Some(i) => i,
        None => {
            let cov = cover(at as usize, (end - at) as usize);
            let region = match cov {
                Cover::Run(..) if allow_section => open_section_region(device, at, end)?,
                Cover::None if allow_private => open_private_region(device, at, end, &regions)?,
                _ => return Err((3, format!("not one stretch of a section: {cov:?}"))),
            };
            let private = region.private;
            let region_bytes = region.end - region.start;
            regions.push(region);
            let n = regions.len() - 1;
            if private {
                STATS[15].fetch_add(1, Ordering::Relaxed);
                STATS[16].fetch_add(region_bytes, Ordering::Relaxed);
            } else {
                STATS[0].fetch_add(1, Ordering::Relaxed);
                STATS[1].fetch_add(region_bytes, Ordering::Relaxed);
            }
            bounds(&regions);
            PORT_EPOCH.fetch_add(1, Ordering::Release);
            n
        }
    };
    let region = &regions[i];
    let buffer = crate::inplace::place(device, &region.heap, at - region.start, end - at).map_err(|e| (6, format!("no buffer placed in the region: {e}")))?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    let placed = Placed {
        buffer,
        heap: region.heap.clone(),
        view: region.view.clone(),
        gpu,
        bias: start - at,
        pinned: 0,
        private_guard: None,
    };
    drop(regions);
    note_placed_interval(start, len);
    Ok(placed)
}

/// Maps and opens `[first, first + bytes)` of `section` at `offset` as a
/// `Region`, releasing its pinned budget and unmapping the view on any error.
fn open_section_slice(
    device: &ID3D12Device,
    first: u64,
    bytes: u64,
    section: usize,
    offset: u64,
    generation: u64,
) -> Result<Region, (usize, String)> {
    let limit = crate::inplace::most().saturating_sub(crate::inplace::pinned_bytes());
    if !crate::inplace_sections::try_reserve_budget(&PINNED_BYTES, bytes, limit) {
        return Err((5, format!("{bytes:#x} bytes would exceed VICTORY_INPLACE_MB budget")));
    }
    // SAFETY: the section of a view the title has; the runtime keeps its
    // handle open while a view of it is there, and the range is inside it.
    let view_addr = unsafe { mem::MapViewOfFile(section as _, mem::FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, bytes as usize) }.Value as usize;
    if view_addr == 0 {
        crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
        return Err((5, format!("no view of the stretch: {}", std::io::Error::last_os_error())));
    }
    let view = Arc::new(OwnedView::new(view_addr));
    if !crate::inplace::is_committed_section(section) && !committed(view_addr, bytes as usize) {
        drop(view);
        crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
        return Err((4, "pages of the stretch are not committed".to_owned()));
    }
    if crate::inplace_sections::mapping_generation() != generation {
        drop(view);
        crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
        return Err((3, "remapped concurrently before opening view heap".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(view_addr, bytes as usize)) {
        Ok(heap) => {
            if crate::inplace_sections::mapping_generation() != generation {
                drop(heap);
                drop(view);
                crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
                return Err((3, "remapped concurrently while opening view heap".to_owned()));
            }
            tracing::debug!(
                start = format_args!("{first:#x}"),
                bytes,
                section = format_args!("{section:#x}"),
                offset = format_args!("{offset:#x}"),
                view = format_args!("{view_addr:#x}"),
                "renderer port 1: a stretch of the title's graphics memory opened as a host heap through a view of ours"
            );
            Ok(Region { start: first, end: first + bytes, heap, view: Some(view), private: false })
        }
        Err(e) => {
            drop(view);
            crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
            Err((5, format!("refused by the host: {e}")))
        }
    }
}

/// Opens the stretch around `[at, end)` (64 KiB aligned) as a heap through
/// a view of ours of the same pages, narrowing to `[at, end)` when the whole
/// stretch exceeds the region cap or has uncommitted pages outside `[at, end)`.
fn open_section_region(device: &ID3D12Device, at: u64, end: u64) -> Result<Region, (usize, String)> {
    let generation = crate::inplace_sections::mapping_generation();
    let Some((raw_first, raw_bytes, section, raw_offset)) = stretch(at as usize, (end - at) as usize) else {
        return Err((3, format!("not one stretch of a section: {:?}", cover(at as usize, (end - at) as usize))));
    };
    let (mut first, mut bytes, mut offset) = (raw_first as u64, raw_bytes as u64, raw_offset);
    if expand_mode() == 0 && first < LOW && first + bytes > LOW {
        bytes = LOW.saturating_sub(first);
    }
    let stretch_cap = if first < LOW { MOST } else { EXPAND_STRETCH_CAP };
    if bytes > stretch_cap {
        offset += at - first;
        (first, bytes) = (at, end - at);
    }
    if bytes > MOST {
        return Err((3, format!("{bytes:#x} bytes, over the most of a region")));
    }
    if first > at || first + bytes < end || offset % K64 != 0 {
        return Err((3, format!("the stretch {first:#x}+{bytes:#x} does not hold it")));
    }
    match open_section_slice(device, first, bytes, section, offset, generation) {
        Ok(region) => Ok(region),
        // Partial-commit fallback: if an expanded stretch had uncommitted pages
        // outside `[at, end)`, retry with the exact requested `[at, end)` sub-range.
        Err((4, _)) if first < at || first + bytes > end => {
            let sub_offset = raw_offset + (at - raw_first as u64);
            let sub_bytes = end - at;
            if sub_bytes > MOST || sub_offset % K64 != 0 {
                return Err((4, "pages of the stretch are not committed".to_owned()));
            }
            open_section_slice(device, at, sub_bytes, section, sub_offset, generation)
        }
        Err(e) => Err(e),
    }
}

/// Opens a cached `MEM_PRIVATE` committed sub-range covering `[at, end)` via
/// `OpenExistingHeapFromAddress1`, expanding to a `PRIVATE_CHUNK` boundary when
/// the full chunk is committed in the same `AllocationBase` and disjoint from
/// existing cached regions and GPU-only/tripwire-eligible buffers (Spec-Trap #8).
fn open_private_region(device: &ID3D12Device, at: u64, end: u64, existing: &[Region]) -> Result<Region, (usize, String)> {
    let generation = crate::inplace_sections::mapping_generation();
    let Some((chunk_start, chunk_len)) = crate::inplace_sections::private_chunk_bounds(at as usize, end as usize, PRIVATE_CHUNK) else {
        return Err((4, "private memory is not committed read-write within one allocation".to_owned()));
    };
    let (mut first, mut bytes) = (chunk_start as u64, chunk_len as u64);
    if (first < at || first + bytes > end)
        && (existing.iter().any(|r| r.end > first && r.start < first + bytes)
            || crate::shadow::overlaps_gpu_only_or_tripwired(first, bytes))
    {
        (first, bytes) = (at, end - at);
    }
    if crate::shadow::overlaps_gpu_only_or_tripwired(first, bytes) {
        return Err((5, "private range overlaps GPU-only/tripwire-eligible buffer".to_owned()));
    }
    if bytes > MOST {
        return Err((3, format!("{bytes:#x} bytes, over the most of a region")));
    }
    let limit = crate::inplace::most().saturating_sub(crate::inplace::pinned_bytes());
    if !crate::inplace_sections::try_reserve_budget(&PINNED_BYTES, bytes, limit) {
        return Err((5, format!("{bytes:#x} bytes would exceed VICTORY_INPLACE_MB budget")));
    }
    if crate::inplace_sections::mapping_generation() != generation {
        crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
        return Err((3, "remapped concurrently before opening private heap".to_owned()));
    }
    match crate::inplace::open(device, crate::inplace::From::Part(first as usize, bytes as usize)) {
        Ok(heap) => {
            if crate::inplace_sections::mapping_generation() != generation {
                drop(heap);
                crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
                return Err((3, "remapped concurrently while opening private heap".to_owned()));
            }
            tracing::debug!(
                start = format_args!("{first:#x}"),
                bytes,
                "renderer port 1: a committed private region opened as a cached host heap"
            );
            Ok(Region { start: first, end: first + bytes, heap, view: None, private: true })
        }
        Err(e) => {
            crate::inplace_sections::release_budget(&PINNED_BYTES, bytes);
            Err((5, format!("private region refused by the host: {e}")))
        }
    }
}

/// Whether every page of `[at, at + len)` is committed.
#[allow(dead_code)]
fn committed(at: usize, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if crate::safecopy::on() {
        // Zero-syscall in-process read probe across every 4 KiB page of our
        // freshly mapped section view; avoids VirtualQuery's page-table walk.
        // SAFETY: `safecopy::on()` installed the fault handler; non-mutating probe.
        return unsafe { crate::safecopy::probe_readable(at as *const u8, len, 4096) };
    }
    let mut here = at;
    while here < at + len {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { mem::VirtualQuery(here as *const _, &mut info, size_of_val(&info)) } == 0 || info.State != mem::MEM_COMMIT {
            return false;
        }
        here = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    }
    true
}

fn bounds(regions: &[Region]) {
    let lo = regions.iter().map(|r| r.start).min().unwrap_or(0);
    let hi = regions.iter().map(|r| r.end).max().unwrap_or(0);
    BOUNDS[0].store(lo, Ordering::Relaxed);
    BOUNDS[1].store(hi, Ordering::Relaxed);
}

/// The title maps other pages at `[at, at + len)`: the regions over it are
/// forgotten (what is placed in them is let go by the caller).
pub(crate) fn forget(at: usize, len: usize) {
    let (at, end) = (at as u64, (at as u64).saturating_add(len as u64));
    forget_placed_interval(at, end);
    if BOUNDS[1].load(Ordering::Relaxed) <= at || BOUNDS[0].load(Ordering::Relaxed) >= end {
        return;
    }
    let mut dropped: Vec<(ID3D12Heap, Option<Arc<OwnedView>>)> = Vec::new();
    let mut released_bytes = 0u64;
    {
        let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < regions.len() {
            if regions[i].end > at && regions[i].start < end {
                let r = regions.swap_remove(i);
                released_bytes += r.end - r.start;
                dropped.push((r.heap, r.view));
            } else {
                i += 1;
            }
        }
        STATS[7].fetch_add(dropped.len() as u64, Ordering::Relaxed);
        bounds(&regions);
        if !dropped.is_empty() {
            PORT_EPOCH.fetch_add(1, Ordering::Release);
        }
    }
    if released_bytes > 0 {
        crate::inplace_sections::release_budget(&PINNED_BYTES, released_bytes);
    }
    for (heap, view) in dropped {
        // Tuple field order drops `heap` before `view` (`UnmapViewOfFile`) once retired.
        crate::host::retire((heap, view));
    }
}

/// Whether `[at, at + len)` touches ported memory (a quick look).
#[allow(dead_code)]
fn touches(at: usize, len: usize) -> bool {
    overlaps(at as u64, len as u64)
}

/// A fill of upload memory read `len` bytes at `src` on the CPU.
#[allow(dead_code)]
pub(crate) fn filled(src: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    if touches(src, len) {
        STATS[9].fetch_add(1, Ordering::Relaxed);
        STATS[10].fetch_add(len as u64, Ordering::Relaxed);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (src, len);
}

/// A title buffer copy was done on the CPU.
#[allow(dead_code)]
pub(crate) fn copied(dst: usize, src: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    if touches(dst, len) || touches(src, len) {
        STATS[11].fetch_add(1, Ordering::Relaxed);
        STATS[12].fetch_add(len as u64, Ordering::Relaxed);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (dst, src, len);
}

/// For the frame-rate line: what is in place, and what of ours read or
/// wrote it on the CPU per frame.
#[allow(dead_code)]
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    let per: Vec<f64> = STATS[9..13].iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "renderer port 1 (VICTORY_PORT_CONSTANTS=1, expand={}): {} view regions ({} MB) + {} private regions ({} MB) + {} committed-section placements ({} MB, 0 MB pinned), \
         {} title buffers and windows placed in them (no host copy, nothing sent up or fetched back for them); per frame on the CPU: {:.0} fills read out of them ({:.1} KB), \
         {:.1} buffer copies touching them ({:.1} KB); since the start refused: {} not one stretch of a section, {} not committed, {} by the host, {} no buffer; {} regions \
         forgotten (other pages mapped there)",
        expand_mode(),
        s[0],
        s[1] >> 20,
        s[15],
        s[16] >> 20,
        s[13],
        s[14] >> 20,
        s[2],
        per[0],
        per[1] / 1024.0,
        per[2],
        per[3] / 1024.0,
        s[3],
        s[4],
        s[5],
        s[6],
        s[7],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the title's 32-bit memory is asked for.
    #[test]
    fn only_memory_below_4_gib_is_wanted_with_the_switch() {
        // The switch is read once from the environment; tests do not set
        // it, so nothing is wanted.
        if !on() {
            assert!(!wanted(0x3431_0000, 0x1000));
        }
        let low = |at: u64, len: u64| at != 0 && at.checked_add(len).is_some_and(|end| end <= LOW && len > 0);
        assert!(low(0x34e1_0000, 0xae0_0000));
        assert!(!low(0, 4 << 20), "null-address loose windows are never section stretches");
        assert!(!low(0xffff_0000, 0x2_0000));
        assert!(!low(0xe_1c80_0000, 0x1000));
        assert!(!low(u64::MAX, 2));
    }

    /// A stretch of a reserved section, mapped as `xmem` maps physical
    /// pages (views of consecutive pages in turn, committed), is opened
    /// through a view of ours: the card reads the bytes the title's views
    /// show, at the buffer's offset, and its writes show in them; a second
    /// buffer in the stretch is placed in the same region.
    #[test]
    fn warp_a_stretch_of_a_reserved_section_is_used_in_place_through_a_view_of_ours() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY (the whole test): the section, placeholder and views made
        // here.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                8 * K as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(process, std::ptr::null(), 8 * K, mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER, mem::PAGE_NOACCESS, std::ptr::null_mut(), 0)
                .cast::<u8>();
            assert!(!hole.is_null());
            // Two views, pages 0..3 and 3..8 of the section, in turn.
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);
            let mut views = Vec::new();
            for (page, n) in [(0usize, 3usize), (3, 5)] {
                let view = mem::MapViewOfFile3(section, process, hole.add(page * K).cast(), (page * K) as u64, n * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
                assert!(!view.Value.is_null(), "{}", std::io::Error::last_os_error());
                assert!(!mem::VirtualAlloc(view.Value, n * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
                crate::inplace::title_memory_view(view.Value as usize, n * K, section as usize, (page * K) as u64);
                views.push(view);
            }
            let base = hole as u64;
            let (start, len) = (base + K as u64 + 0x40, 5 * K as u64);
            let placed = place_in(&device, start, len).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(placed.bias, 0x40);
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            for (i, b) in std::slice::from_raw_parts_mut(start as *mut u8, len as usize).iter_mut().enumerate() {
                *b = (i * 7 + 3) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert!(got == std::slice::from_raw_parts(start as *const u8, len as usize), "the card reads the title's bytes, across both views");
            rig.store(&placed.buffer, placed.bias + 3 * K as u64, &[0x5a; 256]).unwrap();
            assert!(std::slice::from_raw_parts((start + 3 * K as u64) as *const u8, 256).iter().all(|&b| b == 0x5a), "the card's write shows in the title's view");
            // Another buffer in the same stretch: the same region.
            let regions = REGIONS.lock().unwrap().len();
            let second = place_in(&device, base + 6 * K as u64, 100).unwrap_or_else(|e| panic!("not placed: {e:?}"));
            assert_eq!(REGIONS.lock().unwrap().len(), regions, "one region for the stretch");
            assert_eq!(second.bias, 0);
            // The title's views may change under the heap.
            drop((placed, second));
            forget(base as usize, 8 * K);
            for view in views {
                assert!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER) != 0, "the title's own view can be unmapped");
            }
            crate::inplace::views_gone(base as usize, 8 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn expand_mode_parsing_and_placement_policy_cover_all_tiers() {
        assert_eq!(parse_expand_mode(None, false), 0);
        assert_eq!(parse_expand_mode(Some("0"), true), 0);
        assert_eq!(parse_expand_mode(None, true), 3);
        assert_eq!(parse_expand_mode(Some("1"), false), 1);
        assert_eq!(parse_expand_mode(Some("safe"), false), 1);
        assert_eq!(parse_expand_mode(Some("2"), false), 2);
        assert_eq!(parse_expand_mode(Some("sections"), false), 2);
        assert_eq!(parse_expand_mode(Some("3"), false), 3);
        assert_eq!(parse_expand_mode(Some("all"), false), 3);

        let low_addr = 0x1eed_0000u64;
        let high_addr = 0xf_dd3e_0000u64;
        let cap = 32 << 20;

        // Mode 0 (`VICTORY_PORT_CONSTANTS=1` only): 32-bit section only; 64-bit rejected.
        assert_eq!(placement_policy(low_addr, 16 << 20, false, true, false, true, 0, cap), Some((true, false)));
        assert_eq!(placement_policy(high_addr, 128 << 10, false, false, false, true, 0, cap), None);
        assert_eq!(placement_policy(0, 4 << 20, true, false, false, true, 3, cap), None);

        // Mode 1 (safe): non-card 64-bit section buffers and shared-fallback section windows
        // allowed (no MEM_PRIVATE); card-eligible 64-bit buffers and pre-BAR windows stay on VRAM/BAR.
        assert_eq!(placement_policy(high_addr, 128 << 10, false, false, false, true, 1, cap), Some((true, false)));
        assert_eq!(placement_policy(high_addr, 12 << 20, false, true, false, true, 1, cap), None);
        assert_eq!(placement_policy(high_addr, 1 << 20, true, false, false, true, 1, cap), None);
        assert_eq!(placement_policy(high_addr, 1 << 20, true, false, true, true, 1, cap), Some((true, false)));

        // Mode 2 (sections): adds 64-bit section-stretch buffers <= cap (e.g. 11-24 MiB Pool A)
        // and section windows (no MEM_PRIVATE below expand=3), keeping > cap GPU-hot heaps in VRAM.
        assert_eq!(placement_policy(high_addr, 24 << 20, false, true, false, true, 2, cap), Some((true, false)));
        assert_eq!(placement_policy(high_addr, 50 << 20, false, true, false, true, 2, cap), None);
        assert_eq!(placement_policy(high_addr, 1 << 20, true, false, false, true, 2, cap), Some((true, false)));
        assert_eq!(placement_policy(high_addr, 1 << 20, true, false, true, true, 2, cap), Some((true, false)));

        // Mode 3 (all): places all 64-bit buffers (section + private) regardless of size cap.
        assert_eq!(placement_policy(high_addr, 90 << 20, false, true, false, true, 3, cap), Some((true, true)));
        assert_eq!(placement_policy(high_addr, 1 << 20, true, false, false, true, 3, cap), Some((true, true)));
    }

    /// A `SEC_RESERVE` stretch where only an interior sub-range is committed:
    /// `place_in` falls back from the whole stretch to the exact committed sub-range,
    /// rejects uncommitted ranges cleanly, and unmaps our view when forgotten.
    #[test]
    fn warp_partial_commit_stretch_falls_back_to_committed_subrange() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY: section, placeholder, and views created and cleaned up by this test.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_RESERVE,
                0,
                (8 * K) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let hole = mem::VirtualAlloc2(
                process,
                std::ptr::null(),
                8 * K,
                mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                mem::PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
            .cast::<u8>();
            assert!(!hole.is_null());
            let view = mem::MapViewOfFile3(
                section,
                process,
                hole.cast(),
                0,
                8 * K,
                mem::MEM_REPLACE_PLACEHOLDER,
                mem::PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert!(!view.Value.is_null());
            // Commit ONLY pages 2..5 (leaving pages 0..2 and 5..8 uncommitted).
            assert!(!mem::VirtualAlloc(hole.add(2 * K).cast(), 3 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            crate::inplace::title_memory_view(view.Value as usize, 8 * K, section as usize, 0);

            let base = hole as u64;
            let pinned_before = pinned_bytes();

            // Uncommitted range [0..2K) is cleanly rejected with stat 4 ("not committed").
            let err = place_in(&device, base, 2 * K as u64).err().expect("uncommitted must fail");
            assert_eq!(err.0, 4, "expected uncommitted refusal: {err:?}");
            assert_eq!(pinned_bytes(), pinned_before, "pinned budget must not leak on failure");

            // Committed sub-range [2K + 0x80 .. 4K + 0x80) succeeds via sub-range fallback!
            let start = base + (2 * K) as u64 + 0x80;
            let len = (2 * K) as u64;
            let placed = place_in(&device, start, len).unwrap_or_else(|e| panic!("sub-range fallback failed: {e:?}"));
            assert_eq!(placed.bias, 0x80);
            assert_eq!(pinned_bytes(), pinned_before + (3 * K) as u64, "only committed 3 pages pinned");

            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            let cpu = std::slice::from_raw_parts_mut(start as *mut u8, len as usize);
            for (i, b) in cpu.iter_mut().enumerate() {
                *b = (i.wrapping_mul(11).wrapping_add(7)) as u8;
            }
            let got = rig.fetch(&placed.buffer, placed.bias, len as usize).unwrap();
            assert_eq!(got.as_slice(), cpu);
            rig.store(&placed.buffer, placed.bias + K as u64, &[0x3c; 128]).unwrap();
            assert!(std::slice::from_raw_parts((start + K as u64) as *const u8, 128).iter().all(|&b| b == 0x3c));

            drop(placed);
            forget(base as usize, 8 * K);
            assert_eq!(pinned_bytes(), pinned_before, "forget releases pinned budget");
            assert_ne!(mem::UnmapViewOfFile2(process, view, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            crate::inplace::views_gone(base as usize, 8 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Two adjacent stretches from different sections in one reservation: each stretch
    /// places cleanly in its own region, a range crossing the boundary is rejected as
    /// scattered, and forgetting stretch A leaves stretch B cached and coherent.
    #[test]
    fn warp_multi_stretch_boundaries_place_each_stretch_and_reject_crossing() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY: sections, placeholder, and views created and cleaned up by this test.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let make_sec = || {
                let s = mem::CreateFileMappingW(
                    windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    mem::PAGE_READWRITE | mem::SEC_RESERVE,
                    0,
                    (3 * K) as u32,
                    std::ptr::null(),
                );
                assert!(!s.is_null());
                s
            };
            let (sec_a, sec_b) = (make_sec(), make_sec());
            let hole = mem::VirtualAlloc2(
                process,
                std::ptr::null(),
                6 * K,
                mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                mem::PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
            .cast::<u8>();
            assert!(!hole.is_null());
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);

            let va = mem::MapViewOfFile3(sec_a, process, hole.cast(), 0, 3 * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            let vb = mem::MapViewOfFile3(sec_b, process, hole.add(3 * K).cast(), 0, 3 * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert!(!va.Value.is_null() && !vb.Value.is_null());
            assert!(!mem::VirtualAlloc(va.Value, 3 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            assert!(!mem::VirtualAlloc(vb.Value, 3 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            crate::inplace::title_memory_view(va.Value as usize, 3 * K, sec_a as usize, 0);
            crate::inplace::title_memory_view(vb.Value as usize, 3 * K, sec_b as usize, 0);

            let base = hole as u64;
            let pinned_before = pinned_bytes();
            let regions_before = REGIONS.lock().unwrap().len();

            // Place inside stretch A and inside stretch B -> two independent regions.
            let pa = place_in(&device, base + K as u64, K as u64).expect("stretch A placed");
            let pb = place_in(&device, base + (4 * K) as u64, K as u64).expect("stretch B placed");
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before + 2);
            assert_eq!(pinned_bytes(), pinned_before + (6 * K) as u64);

            // Crossing the boundary between stretch A and stretch B is rejected cleanly.
            let cross = place_in(&device, base + (2 * K) as u64, (2 * K) as u64).err().expect("boundary cross must fail");
            assert_eq!(cross.0, 3, "expected scattered refusal: {cross:?}");

            // Forgetting stretch A evicts only stretch A; stretch B stays cached & coherent.
            drop(pa);
            forget(base as usize, 3 * K);
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before + 1);
            assert_eq!(pinned_bytes(), pinned_before + (3 * K) as u64);

            let pb2 = place_in(&device, base + (3 * K) as u64, K as u64).expect("stretch B reused");
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before + 1, "stretch B region reused");
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            let ptr_b = (base + (3 * K) as u64) as *mut u8;
            std::slice::from_raw_parts_mut(ptr_b, 256).fill(0x7e);
            assert_eq!(rig.fetch(&pb2.buffer, pb2.bias, 256).unwrap(), vec![0x7e; 256]);

            drop((pb, pb2));
            forget(base as usize + 3 * K, 3 * K);
            assert_eq!(pinned_bytes(), pinned_before);
            assert_ne!(mem::UnmapViewOfFile2(process, va, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            assert_ne!(mem::UnmapViewOfFile2(process, vb, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            crate::inplace::views_gone(base as usize, 6 * K);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(sec_a);
            windows_sys::Win32::Foundation::CloseHandle(sec_b);
        }
    }

    /// Cached private-region placement: opens only committed sub-ranges of a partial-commit
    /// `VirtualAlloc`, reuses the cached region for a second buffer, rejects uncommitted pages,
    /// and releases budget + allows decommit/release after `forget()`.
    #[test]
    fn warp_cached_private_region_handles_partial_commit_and_forget() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<ID3D12Device13>(&device).is_err() {
            return;
        }
        const K: usize = 64 << 10;
        // SAFETY: test-owned VirtualAlloc allocation, released at end.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 16 * K, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            // Commit ONLY pages 4..8.
            assert!(!mem::VirtualAlloc(base.add(4 * K).cast(), 4 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());

            let pinned_before = pinned_bytes();
            let regions_before = REGIONS.lock().unwrap().len();
            let start = base.add(4 * K) as u64 + 0x40;
            let len = (2 * K) as u64;

            // Uncommitted page 3 rejected cleanly.
            assert_eq!(place_in_with(&device, base.add(3 * K) as u64, (2 * K) as u64, true, true).err().map(|e| e.0), Some(4));

            // Committed sub-range [4K + 0x40 .. 6K + 0x40) opens exact 3 pages (cannot expand into uncommitted).
            let p1 = place_in_with(&device, start, len, true, true).expect("private partial commit placed");
            assert_eq!(p1.bias, 0x40);
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before + 1);
            assert_eq!(pinned_bytes(), pinned_before + (3 * K) as u64);

            // Second buffer inside the same cached 3-page private region reuses it!
            let p2 = place_in_with(&device, base.add(5 * K) as u64, K as u64, true, true).expect("cached private region reused");
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before + 1);

            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            std::slice::from_raw_parts_mut(start as *mut u8, 512).fill(0xab);
            assert_eq!(rig.fetch(&p1.buffer, p1.bias, 512).unwrap(), vec![0xab; 512]);
            rig.store(&p2.buffer, p2.bias, &[0x55; 128]).unwrap();
            assert!(std::slice::from_raw_parts(base.add(5 * K), 128).iter().all(|&b| b == 0x55));

            drop((p1, p2));
            forget(base as usize, 16 * K);
            assert_eq!(pinned_bytes(), pinned_before);
            // Drain retired heaps in this test by checking VirtualFree succeeds once retired.
            // (In unit tests without Host retire loop, drop of retired items happens on sweep;
            // verify region is gone from `REGIONS` and budget restored.)
            assert_eq!(REGIONS.lock().unwrap().len(), regions_before);
            let _ = mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn placed_intervals_track_overlaps_and_split_on_forget() {
        let base = 0x6_E000_0000_0000u64;
        assert!(!overlaps(base, 0x30_0000));
        note_placed_interval(base, 0x10_0000);
        note_placed_interval(base + 0x10_0000, 0x10_0000);
        assert!(overlaps(base + 0x08_0000, 0x4000));
        assert!(overlaps(base + 0x18_0000, 0x4000));
        assert!(!overlaps(base + 0x20_0000, 0x1000));
        // Punch a hole in the middle: both ends stay tracked while the middle clears.
        forget_placed_interval(base + 0x08_0000, base + 0x18_0000);
        assert!(overlaps(base, 0x1000));
        assert!(!overlaps(base + 0x08_0000, 0x10_0000));
        assert!(overlaps(base + 0x18_0000, 0x1000));
        forget_placed_interval(base, base + 0x20_0000);
        assert!(!overlaps(base, 0x20_0000));
    }
}
