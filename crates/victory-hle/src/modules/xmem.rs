//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{
    Arc, Condvar, LazyLock, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT,
    MEM_EXTENDED_PARAMETER, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
    MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile3, MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE,
    ResetWriteWatch, SEC_COMMIT, SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2,
    VirtualFree, VirtualProtect, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;
/// `VirtualFree` flag to join adjacent placeholders back into one.
const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
/// Host 4 KiB page granularity for private virtual allocations.
const HOST_PAGE: usize = 4096;

/// Monotonic generation bumped on every view map/unmap, private virtual
/// allocation/decommit/release, and section teardown.
static MAPPING_GEN: AtomicU64 = AtomicU64::new(1);

fn bump_generation() -> u64 {
    MAPPING_GEN.fetch_add(1, Ordering::AcqRel) + 1
}

/// Current mapping generation counter (lock-free snapshot).
pub fn mapping_generation() -> u64 {
    MAPPING_GEN.load(Ordering::Acquire)
}

/// A maximal contiguous stretch of mapped physical-page views backed by one
/// section at contiguous section offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedStretch {
    pub base: usize,
    pub len: usize,
    pub section: usize,
    pub section_offset: u64,
    pub committed_section: bool,
    pub generation: u64,
}

/// A tracked non-placeholder private virtual allocation from `XMemVirtualAlloc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivateRegion {
    pub base: usize,
    pub len: usize,
    pub committed: bool,
    pub write_watched: bool,
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrivateRegionEntry {
    len: usize,
    alloc_base: usize,
    committed: bool,
    write_watched: bool,
}

/// Tracked committed private virtual bytes owned by the title.
static TRACKED_VIRTUAL_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Tracked live 64 KiB physical pages allocated by the title.
static TRACKED_PHYSICAL_PAGES: AtomicUsize = AtomicUsize::new(0);
/// Tracked live non-32-bit `XMemAllocDefault` heap bytes.
static TRACKED_HEAP_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Tracked non-placeholder `XMemVirtualAlloc` regions, keyed by range base.
static PRIVATE_REGIONS: LazyLock<RwLock<BTreeMap<usize, PrivateRegionEntry>>> =
    LazyLock::new(Default::default);

fn sync_tracked_virtual_bytes(regions: &BTreeMap<usize, PrivateRegionEntry>) {
    let total: usize = regions
        .values()
        .filter(|e| e.committed)
        .map(|e| e.len)
        .sum();
    TRACKED_VIRTUAL_BYTES.store(total, Ordering::Relaxed);
}

fn clear_private_overlaps(
    regions: &mut BTreeMap<usize, PrivateRegionEntry>,
    start: usize,
    end: usize,
) {
    if start >= end {
        return;
    }
    let first = regions
        .range(..=start)
        .next_back()
        .filter(|(base, e)| **base + e.len > start)
        .map(|(&base, _)| base);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(regions.range(start + 1..end).map(|(&base, _)| base))
        .collect();
    for base in hit {
        let Some(entry) = regions.remove(&base) else {
            continue;
        };
        let eend = base + entry.len;
        if base < start {
            regions.insert(base, PrivateRegionEntry { len: start - base, ..entry });
        }
        if eend > end {
            regions.insert(end, PrivateRegionEntry { len: eend - end, ..entry });
        }
    }
}

fn coalesce_private_reservation(
    regions: &mut BTreeMap<usize, PrivateRegionEntry>,
    alloc_base: usize,
) {
    let entries: Vec<(usize, PrivateRegionEntry)> = regions
        .range(alloc_base..)
        .take_while(|(_, e)| e.alloc_base == alloc_base)
        .map(|(&k, &v)| (k, v))
        .collect();
    if entries.len() < 2 {
        return;
    }
    let mut prev_base = entries[0].0;
    let mut prev = entries[0].1;
    for (base, entry) in entries.into_iter().skip(1) {
        if prev_base + prev.len == base
            && prev.committed == entry.committed
            && prev.write_watched == entry.write_watched
        {
            prev.len += entry.len;
            regions.remove(&base);
            if let Some(slot) = regions.get_mut(&prev_base) {
                slot.len = prev.len;
            }
        } else {
            prev_base = base;
            prev = entry;
        }
    }
}

fn set_private_committed(
    regions: &mut BTreeMap<usize, PrivateRegionEntry>,
    start: usize,
    end: usize,
    committed: bool,
) -> bool {
    if start >= end {
        return false;
    }
    let first = regions
        .range(..=start)
        .next_back()
        .filter(|(base, e)| **base + e.len > start)
        .map(|(&base, _)| base);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(regions.range(start + 1..end).map(|(&base, _)| base))
        .collect();
    if hit.is_empty() {
        return false;
    }
    let mut alloc_bases: Vec<usize> = Vec::new();
    for base in hit {
        let Some(entry) = regions.remove(&base) else {
            continue;
        };
        if !alloc_bases.contains(&entry.alloc_base) {
            alloc_bases.push(entry.alloc_base);
        }
        let eend = base + entry.len;
        if base < start {
            regions.insert(base, PrivateRegionEntry { len: start - base, ..entry });
        }
        let mid_start = base.max(start);
        let mid_end = eend.min(end);
        regions.insert(
            mid_start,
            PrivateRegionEntry {
                len: mid_end - mid_start,
                committed,
                ..entry
            },
        );
        if eend > end {
            regions.insert(end, PrivateRegionEntry { len: eend - end, ..entry });
        }
    }
    for alloc_base in alloc_bases {
        coalesce_private_reservation(regions, alloc_base);
    }
    true
}

fn private_region_alloc(requested_base: usize, at: usize, size: usize, kind: u32) {
    let actual_end = if requested_base == 0 {
        at.saturating_add(size.checked_next_multiple_of(HOST_PAGE).unwrap_or(size))
    } else {
        requested_base
            .saturating_add(size)
            .checked_next_multiple_of(HOST_PAGE)
            .unwrap_or(usize::MAX)
    };
    let actual_len = actual_end.saturating_sub(at);
    if actual_len == 0 {
        return;
    }
    let committed = kind & MEM_COMMIT != 0;
    let write_watched = kind & MEM_WRITE_WATCH != 0;
    let mut regions = PRIVATE_REGIONS.write().unwrap_or_else(|e| e.into_inner());
    if kind & MEM_RESERVE != 0 {
        clear_private_overlaps(&mut regions, at, actual_end);
        regions.insert(
            at,
            PrivateRegionEntry {
                len: actual_len,
                alloc_base: at,
                committed,
                write_watched,
            },
        );
    } else if committed && !set_private_committed(&mut regions, at, actual_end, true) {
        regions.insert(
            at,
            PrivateRegionEntry {
                len: actual_len,
                alloc_base: at,
                committed: true,
                write_watched: false,
            },
        );
    }
    sync_tracked_virtual_bytes(&regions);
    bump_generation();
}

fn private_region_free(address: usize, size: usize, kind: u32) {
    let mut regions = PRIVATE_REGIONS.write().unwrap_or_else(|e| e.into_inner());
    if kind & MEM_RELEASE != 0 {
        let keys: Vec<usize> = regions
            .range(address..)
            .take_while(|(_, e)| e.alloc_base == address)
            .map(|(&k, _)| k)
            .collect();
        for k in keys {
            regions.remove(&k);
        }
    } else if kind & MEM_DECOMMIT != 0 {
        if size == 0 {
            let keys: Vec<usize> = regions
                .range(address..)
                .take_while(|(_, e)| e.alloc_base == address)
                .map(|(&k, _)| k)
                .collect();
            for k in keys {
                if let Some(entry) = regions.get_mut(&k) {
                    entry.committed = false;
                }
            }
            coalesce_private_reservation(&mut regions, address);
        } else {
            let start = address & !(HOST_PAGE - 1);
            let end = address
                .saturating_add(size)
                .checked_next_multiple_of(HOST_PAGE)
                .unwrap_or(usize::MAX);
            set_private_committed(&mut regions, start, end, false);
        }
    }
    sync_tracked_virtual_bytes(&regions);
    bump_generation();
}

fn private_region_in(
    regions: &BTreeMap<usize, PrivateRegionEntry>,
    at: usize,
    req_end: usize,
) -> Option<(usize, usize, bool, bool)> {
    let (&first_base, &first) = regions.range(..=at).next_back()?;
    let first_end = first_base.checked_add(first.len)?;
    if at >= first_end {
        return None;
    }
    if req_end <= first_end {
        return Some((first_base, first.len, first.committed, first.write_watched));
    }
    let mut span_base = first_base;
    for (&ebase, &entry) in regions.range(..first_base).rev() {
        if entry.alloc_base != first.alloc_base || ebase + entry.len != span_base {
            break;
        }
        span_base = ebase;
    }
    let mut span_end = first_end;
    let mut all_committed = first.committed;
    let mut all_watched = first.write_watched;
    for (&ebase, &entry) in regions.range(first_base + 1..) {
        if entry.alloc_base != first.alloc_base || ebase != span_end {
            break;
        }
        span_end = ebase.checked_add(entry.len)?;
        all_committed &= entry.committed;
        all_watched &= entry.write_watched;
    }
    (req_end <= span_end).then_some((span_base, span_end - span_base, all_committed, all_watched))
}

/// Finds the tracked private allocation containing `[at, at + len)`.
pub fn query_private_region(at: usize, len: usize) -> Option<PrivateRegion> {
    if len == 0 {
        return None;
    }
    let req_end = at.checked_add(len)?;
    for _ in 0..2 {
        let gen_before = mapping_generation();
        let found = {
            let regions = PRIVATE_REGIONS.read().unwrap_or_else(|e| e.into_inner());
            private_region_in(&regions, at, req_end)
        };
        let gen_after = mapping_generation();
        if gen_before == gen_after {
            return found.map(|(base, len, committed, write_watched)| PrivateRegion {
                base,
                len,
                committed,
                write_watched,
                generation: gen_after,
            });
        }
    }
    None
}

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

/// Maximum single allocation eligible for the `XMemVirtualAlloc` freelist.
const RECYCLE_MAX_ALLOC: usize = 16 * MIB;
/// Default total bytes retained in the `XMemVirtualAlloc` freelist (`VICTORY_XMEM_RECYCLE_MB`).
const DEFAULT_RECYCLE_POOL_MB: usize = 64;
/// Maximum cached blocks per `(alloc_size, align, lowest, highest)` class.
const RECYCLE_BUCKET_CAP: usize = 8;

fn recycle_max_bytes() -> usize {
    static MAX: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("VICTORY_XMEM_RECYCLE_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECYCLE_POOL_MB)
            .saturating_mul(MIB)
    });
    *MAX
}

/// Rounds `size` up to a tight 64 KiB-aligned size class (`<= 25%` slack).
fn recycle_class_size(size: usize) -> usize {
    let min = size.next_multiple_of(PAGE);
    if min <= 256 * 1024 {
        min.next_power_of_two().max(PAGE)
    } else {
        let step = (min.next_power_of_two() >> 3).max(PAGE);
        min.next_multiple_of(step)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RecycleKey {
    alloc_size: usize,
    align: usize,
    lowest: usize,
    highest: usize,
}

#[derive(Default)]
struct VirtualFreelist {
    /// Live allocations eligible to return to `buckets` on `MEM_RELEASE`.
    live: HashMap<usize, RecycleKey>,
    /// Cached committed `MEM_WRITE_WATCH` blocks ready for zero-filled reuse.
    buckets: HashMap<RecycleKey, Vec<usize>>,
    pooled_bytes: usize,
}

static VIRTUAL_FREELIST: LazyLock<Mutex<VirtualFreelist>> = LazyLock::new(Default::default);

fn take_recycled_virtual(key: RecycleKey) -> Option<usize> {
    let mut fl = VIRTUAL_FREELIST.lock().unwrap_or_else(|e| e.into_inner());
    let bucket = fl.buckets.get_mut(&key)?;
    let at = bucket.pop()?;
    if bucket.is_empty() {
        fl.buckets.remove(&key);
    }
    fl.pooled_bytes = fl.pooled_bytes.saturating_sub(key.alloc_size);
    fl.live.insert(at, key);
    Some(at)
}

fn note_recyclable_virtual(at: usize, key: RecycleKey) {
    let mut fl = VIRTUAL_FREELIST.lock().unwrap_or_else(|e| e.into_inner());
    fl.live.insert(at, key);
}

fn untrack_recyclable_range(at: usize, size: usize) {
    let mut fl = VIRTUAL_FREELIST.lock().unwrap_or_else(|e| e.into_inner());
    if fl.live.is_empty() {
        return;
    }
    if fl.live.remove(&at).is_some() {
        return;
    }
    let end = at.saturating_add(size.max(1));
    fl.live.retain(|&base, key| end <= base || at >= base.saturating_add(key.alloc_size));
}

/// Attempts to return a `MEM_RELEASE` allocation to the freelist. Returns `true`
/// if retained (caller skips `VirtualFree`), `false` to release to the OS.
fn try_recycle_virtual_release(at: usize) -> Option<usize> {
    let max_bytes = recycle_max_bytes();
    let key = {
        let mut fl = VIRTUAL_FREELIST.lock().unwrap_or_else(|e| e.into_inner());
        fl.live.remove(&at)?
    };
    if max_bytes == 0 || key.alloc_size > max_bytes {
        return None;
    }
    // Restore PAGE_READWRITE before pooling; fails (e.g. error 87) if pinned.
    let mut old = 0;
    // SAFETY: live VirtualAlloc2 reservation owned by this process.
    if unsafe { VirtualProtect(at as *const c_void, key.alloc_size, PAGE_READWRITE, &mut old) } == 0
    {
        return None;
    }
    let mut fl = VIRTUAL_FREELIST.lock().unwrap_or_else(|e| e.into_inner());
    let bucket_len = fl.buckets.get(&key).map_or(0, Vec::len);
    if bucket_len >= RECYCLE_BUCKET_CAP || fl.pooled_bytes.saturating_add(key.alloc_size) > max_bytes
    {
        return None;
    }
    fl.pooled_bytes += key.alloc_size;
    fl.buckets.entry(key).or_default().push(at);
    Some(key.alloc_size)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let align = if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
        2 * MIB
    } else {
        0
    };
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: align,
    };
    let asked = protect;
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    let recyclable = constrained
        && !placeholder
        && kind & MEM_RESERVE_PLACEHOLDER == 0
        && kind & MEM_COMMIT != 0
        && protect == PAGE_READWRITE
        && size > 0
        && size <= RECYCLE_MAX_ALLOC
        && recycle_max_bytes() > 0;
    let recycle_key = recyclable.then(|| RecycleKey {
        alloc_size: recycle_class_size(size),
        align,
        lowest,
        highest,
    });
    if let Some(key) = recycle_key
        && let Some(at) = take_recycled_virtual(key)
    {
        // Zero-fill and reset write-watch so the reused block matches a fresh
        // `VirtualAlloc2(MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH)` region.
        // SAFETY: committed PAGE_READWRITE region of `key.alloc_size` bytes.
        unsafe {
            (at as *mut u8).write_bytes(0, key.alloc_size);
            ResetWriteWatch(at as *const c_void, key.alloc_size);
        }
        private_region_alloc(0, at, size, kind);
        victory_gpu::title_memory_allocated(at, size);
        #[cfg(feature = "diagnostics")]
        note_protection(asked, kind, flags, at, size);
        #[cfg(not(feature = "diagnostics"))]
        let _ = asked;
        return at as *mut c_void;
    }
    let alloc_size = recycle_key.map_or(size, |k| k.alloc_size);
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            alloc_size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(alloc_size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    if !at.is_null() && !placeholder {
        if let Some(key) = recycle_key {
            note_recyclable_virtual(at as usize, key);
        }
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            private_region_alloc(base as usize, at as usize, size, kind);
        }
        victory_gpu::title_memory_allocated(at as usize, size);
    }
    // (The title asks for its 32-bit heap with kind 0x42000,
    // without the 64 KiB page bits: not `placeholder` above, though the host
    // makes a placeholder of it all the same.)
    if !at.is_null() && constrained && kind & MEM_COMMIT == 0 {
        if flags & ADDRESS_32BIT != 0 {
            LOW_RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()).push((at as usize, size));
        } else if kind & MEM_RESERVE_PLACEHOLDER != 0 {
            HIGH_RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()).push((at as usize, size));
        }
    }
    #[cfg(feature = "diagnostics")]
    note_protection(asked, kind, flags, at as usize, size);
    #[cfg(not(feature = "diagnostics"))]
    let _ = asked;
    at
}

/// What the title asks its memory to be (the console's protection, with the
/// bits that say how each processor reaches it: 0x200 uncached, 0x400
/// write-combined, 0x800 and on the GPU's): said for the 1st, 2nd, 4th.. time of
/// each, with how much so far. (For placing it: memory the CPU reaches
/// uncached or write-combined is memory it does not read.)
#[cfg(feature = "diagnostics")]
fn note_protection(protect: u32, kind: u32, flags: u64, at: usize, size: usize) {
    type Seen = std::collections::HashMap<(u32, u32, u64), (u64, usize)>;
    static SEEN: std::sync::Mutex<Option<Seen>> = std::sync::Mutex::new(None);
    let (n, bytes) = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let e = seen.get_or_insert_with(Default::default).entry((protect, kind, flags)).or_default();
        *e = (e.0 + 1, e.1 + size);
        *e
    };
    if n.is_power_of_two() {
        tracing::info!(
            "title memory asked for as protection {protect:#x}, kind {kind:#x}, flags {flags:#x}: {n} times, {} MB, last {at:#x}+{size:#x}",
            bytes >> 20
        );
    }
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        TRACKED_HEAP_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            title_virtual_free(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        let _ = TRACKED_HEAP_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_sub(size))
        });
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// Sliding host-VA window chunk size (`16 MiB` = 256 × 64 KiB pages).
const WINDOW_CHUNK_PAGES: usize = 256;
/// Default total bytes of cached host-VA sliding windows (`VICTORY_SECTION_WINDOW_MB`).
const DEFAULT_WINDOW_CACHE_MB: usize = 128;

fn window_cache_max_bytes() -> usize {
    static MAX: LazyLock<usize> = LazyLock::new(|| {
        if std::env::var_os("VICTORY_SECTION_WINDOW").is_some_and(|v| v == "0") {
            return 0;
        }
        std::env::var("VICTORY_SECTION_WINDOW_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_WINDOW_CACHE_MB)
            .saturating_mul(MIB)
    });
    *MAX
}

static NEXT_SECTION_ID: AtomicU64 = AtomicU64::new(1);

/// A mapped sliding host-VA view of one `<= 16 MiB` chunk of a section.
/// Unmaps only when both the LRU cache and every active [`WindowGuard`] drop it.
struct MappedChunk {
    base: usize,
    len: usize,
}

impl Drop for MappedChunk {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        if self.base != 0 {
            // SAFETY: our own MapViewOfFile view; Arc guarantees no live
            // WindowGuard or cache entry still points into it.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.base as *mut c_void,
                })
            };
        }
    }
}

/// Safe guard keeping a sliding section window chunk mapped while the caller
/// reads or writes `[at, at + len)`.
pub(crate) struct WindowGuard {
    at: usize,
    len: usize,
    _chunk: Arc<MappedChunk>,
}

impl WindowGuard {
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: committed bytes inside `_chunk`, kept mapped while `self` lives.
        unsafe { std::slice::from_raw_parts_mut(self.at as *mut u8, self.len) }
    }

    #[cfg(test)]
    fn addr(&self) -> usize {
        self.at
    }
}

struct WindowCacheEntry {
    chunk: Arc<MappedChunk>,
    last_used: u64,
}

#[derive(Default)]
struct WindowCache {
    entries: HashMap<(u64, usize), WindowCacheEntry>,
    total_bytes: usize,
    tick: u64,
}

static WINDOW_CACHE: LazyLock<Mutex<WindowCache>> = LazyLock::new(Default::default);

fn window_cache_get(section_id: u64, chunk_idx: usize) -> Option<Arc<MappedChunk>> {
    let mut cache = WINDOW_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.tick = cache.tick.wrapping_add(1);
    let tick = cache.tick;
    let entry = cache.entries.get_mut(&(section_id, chunk_idx))?;
    entry.last_used = tick;
    Some(entry.chunk.clone())
}

fn window_cache_insert(
    section_id: u64,
    chunk_idx: usize,
    chunk: Arc<MappedChunk>,
    max_bytes: usize,
) -> Arc<MappedChunk> {
    let mut to_drop: Vec<Arc<MappedChunk>> = Vec::new();
    let kept = {
        let mut cache = WINDOW_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        let key = (section_id, chunk_idx);
        if let Some(existing) = cache.entries.get_mut(&key) {
            existing.last_used = tick;
            let winner = existing.chunk.clone();
            to_drop.push(chunk);
            winner
        } else {
            let chunk_len = chunk.len;
            cache.total_bytes += chunk_len;
            cache.entries.insert(
                key,
                WindowCacheEntry {
                    chunk: chunk.clone(),
                    last_used: tick,
                },
            );
            while cache.total_bytes > max_bytes && cache.entries.len() > 1 {
                let victim = cache
                    .entries
                    .iter()
                    .filter(|&(&k, _)| k != key)
                    .min_by_key(|(_, e)| e.last_used)
                    .map(|(&k, _)| k);
                let Some(victim_key) = victim else {
                    break;
                };
                if let Some(removed) = cache.entries.remove(&victim_key) {
                    cache.total_bytes = cache.total_bytes.saturating_sub(removed.chunk.len);
                    to_drop.push(removed.chunk);
                }
            }
            chunk
        }
    };
    // Drop evicted / duplicate views outside `WINDOW_CACHE` lock so
    // `UnmapViewOfFile` never blocks other workers.
    drop(to_drop);
    kept
}

fn window_cache_evict_section(section_id: u64) {
    let mut to_drop: Vec<Arc<MappedChunk>> = Vec::new();
    {
        let mut cache = WINDOW_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, usize)> = cache
            .entries
            .keys()
            .copied()
            .filter(|&(sid, _)| sid == section_id)
            .collect();
        for key in keys {
            if let Some(removed) = cache.entries.remove(&key) {
                cache.total_bytes = cache.total_bytes.saturating_sub(removed.chunk.len);
                to_drop.push(removed.chunk);
            }
        }
    }
    drop(to_drop);
}

struct Section {
    id: u64,
    handle: HANDLE,
    pages: usize,
    /// Whether the section was created with `SEC_COMMIT` (all pages committed).
    sec_committed: bool,
    /// Bytes reserved from [`COMMITTED_SECTION_BYTES`] for this section (released on drop).
    budget_bytes: usize,
    /// A bit per page: committed, as far as [`Section::window`] asked (also
    /// marks pages touched for [`arrange`]).
    committed: Box<[std::sync::atomic::AtomicU64]>,
    /// Monotonic word cursor skipping leading `u64::MAX` words in [`find_fresh_run`].
    fresh_word: std::sync::atomic::AtomicUsize,
    /// Reverse map `slice_index -> page_number` (`0` when freed), maintained
    /// under [`PAGES`] write lock so `arrange`/`arrange_fresh_run` swap in O(1).
    owners: Box<[std::sync::atomic::AtomicUsize]>,
}

impl Section {
    fn new(
        handle: HANDLE,
        pages: usize,
        first_page: usize,
        sec_committed: bool,
        budget_bytes: usize,
    ) -> Section {
        Section {
            id: NEXT_SECTION_ID.fetch_add(1, Ordering::Relaxed),
            handle,
            pages,
            sec_committed,
            budget_bytes,
            committed: (0..pages.div_ceil(64))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            fresh_word: std::sync::atomic::AtomicUsize::new(0),
            owners: (0..pages)
                .map(|k| {
                    std::sync::atomic::AtomicUsize::new(if first_page == 0 {
                        0
                    } else {
                        first_page + k * FRAMES
                    })
                })
                .collect(),
        }
    }

    /// Bounded, evictable `<= 16 MiB` sliding view covering `n` pages from
    /// `index` on, committed and kept mapped by the returned [`WindowGuard`].
    ///
    /// Returns `None` when disabled (`VICTORY_SECTION_WINDOW=0` or
    /// `VICTORY_SECTION_WINDOW_MB=0`), when a run straddles a 16 MiB chunk
    /// boundary, or on host failure — callers fall back to a one-shot [`View`].
    fn window(&self, index: usize, n: usize) -> Option<WindowGuard> {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let max_bytes = window_cache_max_bytes();
        if max_bytes == 0 || n == 0 || index.checked_add(n).is_none_or(|end| end > self.pages) {
            return None;
        }
        let chunk_idx = index / WINDOW_CHUNK_PAGES;
        let end_chunk_idx = (index + n - 1) / WINDOW_CHUNK_PAGES;
        if chunk_idx != end_chunk_idx {
            return None;
        }
        let chunk_start = chunk_idx * WINDOW_CHUNK_PAGES;
        let chunk_pages = (self.pages - chunk_start).min(WINDOW_CHUNK_PAGES);
        let chunk_bytes = chunk_pages * PAGE;
        let chunk = match window_cache_get(self.id, chunk_idx) {
            Some(chunk) => chunk,
            None => {
                let byte_offset = (chunk_start * PAGE) as u64;
                // SAFETY: bounded chunk inside a section handle we own.
                let base = unsafe {
                    MapViewOfFile(
                        self.handle,
                        FILE_MAP_WRITE,
                        (byte_offset >> 32) as u32,
                        byte_offset as u32,
                        chunk_bytes,
                    )
                }
                .Value as usize;
                if base == 0 {
                    return None;
                }
                let mapped = Arc::new(MappedChunk {
                    base,
                    len: chunk_bytes,
                });
                window_cache_insert(self.id, chunk_idx, mapped, max_bytes)
            }
        };
        let within_chunk = (index - chunk_start) * PAGE;
        if self.sec_committed {
            // Pages are already committed by the section; still mark them
            // touched so `arrange` never swaps written slices.
            for p in index..index + n {
                let bit = 1u64 << (p % 64);
                if self.committed[p / 64].load(Ordering::Relaxed) & bit == 0 {
                    self.committed[p / 64].fetch_or(bit, Ordering::AcqRel);
                }
            }
            return Some(WindowGuard {
                at: chunk.base + within_chunk,
                len: n * PAGE,
                _chunk: chunk,
            });
        }
        let missing =
            |page: usize| self.committed[page / 64].load(Ordering::Acquire) >> (page % 64) & 1 == 0;
        let mut page = index;
        while page < index + n {
            if !missing(page) {
                page += 1;
                continue;
            }
            let run = (page..index + n).take_while(|&p| missing(p)).count();
            if run == 0 {
                continue; // another worker committed it meanwhile
            }
            if !commit(chunk.base + (page - chunk_start) * PAGE, run * PAGE) {
                return None;
            }
            for p in page..page + run {
                self.committed[p / 64].fetch_or(1 << (p % 64), Ordering::AcqRel);
            }
            page += run;
        }
        Some(WindowGuard {
            at: chunk.base + within_chunk,
            len: n * PAGE,
            _chunk: chunk,
        })
    }
}

impl Section {
    /// Commits the pages from `index` on that a view at `base` shows, unless
    /// all of them are ([`lean`]): committed for one view is for all.
    fn commit_shown(&self, index: usize, base: usize, len: usize) -> bool {
        use std::sync::atomic::Ordering;
        let pages = index..(index + len / PAGE).min(self.pages);
        let bit = |p: usize| (&self.committed[p / 64], 1u64 << (p % 64));
        if self.sec_committed {
            // Pages are already committed by the section; still mark them
            // touched so `arrange` never swaps mapped slices.
            for p in pages {
                let (word, bit) = bit(p);
                if word.load(Ordering::Relaxed) & bit == 0 {
                    word.fetch_or(bit, Ordering::AcqRel);
                }
            }
            return true;
        }
        let done = || {
            pages.clone().all(|p| {
                let (word, bit) = bit(p);
                word.load(Ordering::Acquire) & bit != 0
            })
        };
        if lean() && done() {
            return true;
        }
        if !counted(Call::Commit, || commit(base, len)) {
            return false;
        }
        for p in pages {
            let (word, bit) = bit(p);
            word.fetch_or(bit, Ordering::AcqRel);
        }
        true
    }
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        window_cache_evict_section(self.id);
        if self.budget_bytes > 0 {
            release_commit_section_bytes(self.budget_bytes);
        }
        bump_generation();
        victory_gpu::title_memory_section_gone(self.handle as usize);
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Placeholders that are some view's own, left by its unmapping: address
    /// and length. The title may have joined one with others since.
    holes: BTreeMap<usize, usize>,
}

impl Physical {
    /// Removes any tracked hole overlapping `[start, end)`.
    fn clear_holes(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let first = self
            .holes
            .range(..=start)
            .next_back()
            .filter(|&(&base, &len)| base + len > start)
            .map(|(&base, _)| base);
        let hit: Vec<usize> = first
            .into_iter()
            .chain(self.holes.range(start + 1..end).map(|(&base, _)| base))
            .collect();
        for base in hit {
            self.holes.remove(&base);
        }
    }
    /// What the page at `at` shows.
    fn slice_at(&self, at: usize) -> Option<Slice> {
        self.views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
    }

    /// Whether any mapped view overlaps `[start, end)`.
    fn overlaps_view(&self, start: usize, end: usize) -> bool {
        if start >= end {
            return false;
        }
        let before = self
            .views
            .range(..=start)
            .next_back()
            .is_some_and(|(&base, &(size, _))| base + size > start);
        before || self.views.range(start + 1..end).next().is_some()
    }
}

/// The title's pages by number.
struct Pages {
    next_page: usize,
    pages: HashMap<usize, Slice>,
}

/// Where the title sees its pages: [`Physical::views`] the other way.
/// Locked apart from them, and only for a moment ([`PAGES`]).
#[derive(Default)]
struct Shown {
    /// The addresses that show a page, by the page's section and index.
    /// Views of more than [`SHORT`] pages are in `long`.
    pages: HashMap<(usize, usize), Vec<usize>>,
    /// Views longer than [`SHORT`] pages, keyed by `Arc::as_ptr(&section)`:
    /// `(base, len, first_index)`.
    long: HashMap<usize, Vec<(usize, usize, usize)>>,
    /// Lock-minimal view index for [`query_mapped_stretch`]:
    /// `base -> (len, section_handle, section_offset, sec_committed)`.
    views: BTreeMap<usize, (usize, usize, u64, bool)>,
}

/// Pages of a view noted page by page in [`Shown::pages`].
const SHORT: usize = 64;

/// Taken after [`PHYSICAL`] where both are.
static SHOWN: LazyLock<RwLock<Shown>> = LazyLock::new(Default::default);

fn shown_mut() -> RwLockWriteGuard<'static, Shown> {
    SHOWN.write().unwrap_or_else(|e| e.into_inner())
}

impl Shown {
    fn note(&mut self, base: usize, len: usize, slice: &Slice) {
        self.views.insert(
            base,
            (
                len,
                slice.section.handle as usize,
                (slice.index * PAGE) as u64,
                slice.section.sec_committed,
            ),
        );
        let section = Arc::as_ptr(&slice.section) as usize;
        if len / PAGE > SHORT {
            self.long.entry(section).or_default().push((base, len, slice.index));
            return;
        }
        for k in 0..len / PAGE {
            self.pages.entry((section, slice.index + k)).or_default().push(base + k * PAGE);
        }
    }

    fn forget(&mut self, base: usize, len: usize, slice: &Slice) {
        self.views.remove(&base);
        let section = Arc::as_ptr(&slice.section) as usize;
        if len / PAGE > SHORT {
            if let Some(views) = self.long.get_mut(&section) {
                views.retain(|view| view.0 != base);
                if views.is_empty() {
                    self.long.remove(&section);
                }
            }
            return;
        }
        for k in 0..len / PAGE {
            let key = (section, slice.index + k);
            if let Some(at) = self.pages.get_mut(&key) {
                at.retain(|&at| at != base + k * PAGE);
                if at.is_empty() {
                    self.pages.remove(&key);
                }
            }
        }
    }

    /// Where the title sees the page at `slice`: addresses and bytes.
    fn at(&self, slice: &Slice) -> Vec<(usize, usize)> {
        let section = Arc::as_ptr(&slice.section) as usize;
        let mut out: Vec<(usize, usize)> =
            self.pages.get(&(section, slice.index)).into_iter().flatten().map(|&at| (at, PAGE)).collect();
        if let Some(views) = self.long.get(&section) {
            for &(base, len, index) in views {
                if (index..index + len / PAGE).contains(&slice.index) {
                    out.push((base + (slice.index - index) * PAGE, PAGE));
                }
            }
        }
        out
    }
}

fn mapped_stretch_in(
    views: &BTreeMap<usize, (usize, usize, u64, bool)>,
    at: usize,
    req_end: usize,
) -> Option<(usize, usize, usize, u64, bool)> {
    let (&first_base, &(first_len, section, first_off, committed)) =
        views.range(..=at).next_back()?;
    let first_end = first_base.checked_add(first_len)?;
    if at >= first_end {
        return None;
    }
    let mut stretch_base = first_base;
    let mut stretch_off = first_off;
    for (&vbase, &(vlen, vsec, voff, vcom)) in views.range(..first_base).rev() {
        if vsec != section
            || vcom != committed
            || vbase + vlen != stretch_base
            || voff.checked_add(vlen as u64) != Some(stretch_off)
        {
            break;
        }
        stretch_base = vbase;
        stretch_off = voff;
    }
    let mut stretch_end = first_end;
    let mut expected_off = first_off.checked_add(first_len as u64)?;
    for (&vbase, &(vlen, vsec, voff, vcom)) in views.range(first_base + 1..) {
        if vbase != stretch_end
            || vsec != section
            || vcom != committed
            || voff != expected_off
        {
            break;
        }
        stretch_end = stretch_end.checked_add(vlen)?;
        expected_off = expected_off.checked_add(vlen as u64)?;
    }
    (stretch_end >= req_end).then_some((
        stretch_base,
        stretch_end - stretch_base,
        section,
        stretch_off,
        committed,
    ))
}

/// Finds the maximal contiguous stretch of mapped views covering `[at, at + len)`
/// from one section at contiguous section offsets, without taking [`PHYSICAL`].
pub fn query_mapped_stretch(at: usize, len: usize) -> Option<MappedStretch> {
    if len == 0 {
        return None;
    }
    let req_end = at.checked_add(len)?;
    for _ in 0..2 {
        let gen_before = mapping_generation();
        let found = {
            let shown = SHOWN.read().unwrap_or_else(|e| e.into_inner());
            mapped_stretch_in(&shown.views, at, req_end)
        };
        let gen_after = mapping_generation();
        if gen_before == gen_after {
            return found.map(
                |(base, len, section, section_offset, committed_section)| MappedStretch {
                    base,
                    len,
                    section,
                    section_offset,
                    committed_section,
                    generation: gen_after,
                },
            );
        }
    }
    None
}

/// See [`pages_written`].
pub(crate) fn stream_notice() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_STREAM_NOTICE").is_none_or(|v| v != "0"));
    *ON
}

/// The views, changed by the title's mapping calls; held for as long as
/// those take (the host maps, commits and unmaps under it).
static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(Default::default);

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title's pages. Apart from the views: DirectStorage looks pages up at
/// every request, from 24 workers, and used to wait there for the title's
/// mapping calls (12,000 waits of 1 to 170 ms in a load of 90 s). Written
/// only where the title allocates or frees pages. Taken after [`PHYSICAL`]
/// where both are.
static PAGES: LazyLock<RwLock<Pages>> = LazyLock::new(|| {
    RwLock::new(Pages {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        pages: HashMap::new(),
    })
});

fn page_table() -> RwLockReadGuard<'static, Pages> {
    PAGES.read().unwrap_or_else(|e| e.into_inner())
}

fn page_table_mut() -> RwLockWriteGuard<'static, Pages> {
    PAGES.write().unwrap_or_else(|e| e.into_inner())
}

/// Whether new physical-page sections should attempt `SEC_COMMIT` backing so
/// the GPU side can open them via `OpenExistingHeapFromFileMapping`.
fn commit_sections() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        if on_wine() {
            return true;
        }
        let explicit = std::env::var("VICTORY_INPLACE_SECTIONS")
            .ok()
            .map(|v| v.trim().to_owned());
        if explicit.as_deref() == Some("0") {
            return false;
        }
        explicit.as_deref() == Some("1")
            || port_expand_on()
            || std::env::var("VICTORY_INPLACE_ALL").is_ok_and(|v| v.trim() == "1")
            || std::env::var("VICTORY_INPLACE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .is_some_and(|n| n >= 3)
    });
    *ON
}

/// Default total/per-allocation `SEC_COMMIT` section cap (`1024 MiB`): keeps
/// the 185 MB 32-bit heap and 710 MB Pool A eligible for zero-pin `port 1`
/// while preventing gigabyte streaming pools from pre-committing RAM.
const DEFAULT_COMMIT_SECTION_MAX_MB: usize = 1024;
/// `SEC_COMMIT` allocations up to this size remain a single section (preserving
/// contiguous `arrange()` on the 185 MB 32-bit heap).
const COMMIT_SUBSECTION_THRESHOLD: usize = 256 * MIB;
/// Sub-section size for larger `SEC_COMMIT` pools (e.g. Pool A) so D3D12 only
/// opens sub-sections that actually contain placed buffers.
const COMMIT_SUBSECTION_BYTES: usize = 128 * MIB;
/// Default chunk size for large `SEC_RESERVE` physical pools (`32 MiB`, clamped
/// `16..=64 MiB` via `VICTORY_RESERVE_CHUNK_MB`) so freed slices reclaim commit
/// as soon as their sub-section empties.
const DEFAULT_RESERVE_CHUNK_MB: usize = 32;

/// Live bytes backed by `SEC_COMMIT` physical sections (excluding sparse Wine).
static COMMITTED_SECTION_BYTES: AtomicUsize = AtomicUsize::new(0);

fn try_reserve_commit_section_bytes(bytes: usize, limit: usize) -> bool {
    let mut cur = COMMITTED_SECTION_BYTES.load(Ordering::Relaxed);
    loop {
        let Some(next) = cur.checked_add(bytes) else {
            return false;
        };
        if next > limit {
            return false;
        }
        match COMMITTED_SECTION_BYTES.compare_exchange_weak(
            cur,
            next,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
}

fn release_commit_section_bytes(bytes: usize) {
    let _ = COMMITTED_SECTION_BYTES.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |cur| {
        Some(cur.saturating_sub(bytes))
    });
}

/// Largest section / total budget (in bytes) eligible for `SEC_COMMIT` backing
/// (`VICTORY_INPLACE_SECTION_MAX_MB`, default `1024` MiB).
fn commit_section_max_bytes() -> usize {
    static MAX: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("VICTORY_INPLACE_SECTION_MAX_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_COMMIT_SECTION_MAX_MB)
            .saturating_mul(MIB)
    });
    *MAX
}

fn reserve_chunk_bytes() -> usize {
    static CHUNK: LazyLock<usize> = LazyLock::new(|| {
        std::env::var("VICTORY_RESERVE_CHUNK_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_RESERVE_CHUNK_MB)
            .clamp(16, 64)
            .saturating_mul(MIB)
    });
    *CHUNK
}

/// Event-driven epoch bumped whenever physical pages are allocated or mapped,
/// replacing `Sleep(1)` polling in DirectStorage page-list waits.
static PAGE_EPOCH: AtomicU64 = AtomicU64::new(0);
static PAGE_WAITERS: AtomicUsize = AtomicUsize::new(0);
static PAGE_WAIT: LazyLock<(Mutex<u64>, Condvar)> = LazyLock::new(|| (Mutex::new(0), Condvar::new()));

pub(crate) fn page_epoch() -> u64 {
    PAGE_EPOCH.load(Ordering::Acquire)
}

fn notify_page_waiters() {
    let next = PAGE_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
    if PAGE_WAITERS.load(Ordering::Acquire) > 0 {
        let mut guard = PAGE_WAIT.0.lock().unwrap_or_else(|e| e.into_inner());
        *guard = next;
        drop(guard);
        PAGE_WAIT.1.notify_all();
    }
}

pub(crate) fn wait_page_epoch(seen: u64, timeout: std::time::Duration) -> u64 {
    let cur = PAGE_EPOCH.load(Ordering::Acquire);
    if cur != seen || timeout.is_zero() {
        return cur;
    }
    struct WaiterGuard;
    impl Drop for WaiterGuard {
        fn drop(&mut self) {
            PAGE_WAITERS.fetch_sub(1, Ordering::AcqRel);
        }
    }
    PAGE_WAITERS.fetch_add(1, Ordering::AcqRel);
    let _guard = WaiterGuard;
    let lock = PAGE_WAIT.0.lock().unwrap_or_else(|e| e.into_inner());
    if PAGE_EPOCH.load(Ordering::Acquire) != seen {
        return PAGE_EPOCH.load(Ordering::Acquire);
    }
    let _ = PAGE_WAIT
        .1
        .wait_timeout_while(lock, timeout, |_| PAGE_EPOCH.load(Ordering::Acquire) == seen)
        .unwrap_or_else(|e| e.into_inner());
    PAGE_EPOCH.load(Ordering::Acquire)
}

/// Creates a pagefile-backed section of `size` bytes, optionally trying
/// `SEC_COMMIT` first and falling back to `SEC_RESERVE` on Windows if commit
/// fails. Returns `(handle, sec_committed)`.
fn create_section_handle(size: usize, try_commit: bool) -> Option<(HANDLE, bool)> {
    let hi = (size as u64 >> 32) as u32;
    let lo = size as u32;
    if try_commit {
        // SAFETY: plain section creation.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                hi,
                lo,
                std::ptr::null(),
            )
        };
        if !handle.is_null() {
            return Some((handle, true));
        }
        if on_wine() {
            return None;
        }
        static WARNED: AtomicU64 = AtomicU64::new(0);
        if WARNED.fetch_add(1, Ordering::Relaxed) < 8 {
            let error = std::io::Error::last_os_error();
            tracing::warn!(
                size,
                %error,
                "SEC_COMMIT section creation failed; falling back to SEC_RESERVE"
            );
        }
    }
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | SEC_RESERVE,
            hi,
            lo,
            std::ptr::null(),
        )
    };
    (!handle.is_null()).then_some((handle, false))
}

#[cfg(test)]
fn allocate_section(n: usize, try_commit: bool) -> Option<Arc<Section>> {
    let size = n.checked_mul(PAGE).filter(|&s| s > 0)?;
    let (handle, sec_committed) = create_section_handle(size, try_commit)?;
    Some(Arc::new(Section::new(handle, n, 0, sec_committed, 0)))
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    let wine = on_wine();
    let max_commit = commit_section_max_bytes();
    let budget_reserved = !wine
        && commit_sections()
        && size <= max_commit
        && try_reserve_commit_section_bytes(size, max_commit);
    let want_commit = wine || budget_reserved;

    let chunk_pages = if want_commit {
        if wine || size <= COMMIT_SUBSECTION_THRESHOLD {
            n
        } else {
            (COMMIT_SUBSECTION_BYTES / PAGE).max(1)
        }
    } else {
        let reserve_chunk = (reserve_chunk_bytes() / PAGE).max(1);
        if n <= reserve_chunk { n } else { reserve_chunk }
    };

    // `(handle, chunk_pages, sec_committed, budget_bytes)` per created sub-section.
    let mut created: Vec<(HANDLE, usize, bool, usize)> = Vec::with_capacity(n.div_ceil(chunk_pages));
    let mut remaining_budget = if budget_reserved { size } else { 0 };
    let mut offset_pages = 0usize;
    while offset_pages < n {
        let this_pages = (n - offset_pages).min(chunk_pages);
        let this_bytes = this_pages * PAGE;
        let Some((handle, sec_committed)) = create_section_handle(this_bytes, want_commit) else {
            let error = std::io::Error::last_os_error();
            tracing::warn!(n, offset_pages, %error, "XMemAllocatePhysicalPages failed");
            for (h, _, _, b) in created {
                if b > 0 {
                    release_commit_section_bytes(b);
                }
                // SAFETY: handle created in this call and not published yet.
                unsafe { CloseHandle(h) };
            }
            if remaining_budget > 0 {
                release_commit_section_bytes(remaining_budget);
            }
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        };
        let section_budget = if budget_reserved {
            remaining_budget = remaining_budget.saturating_sub(this_bytes);
            if sec_committed {
                this_bytes
            } else {
                release_commit_section_bytes(this_bytes);
                0
            }
        } else {
            0
        };
        created.push((handle, this_pages, sec_committed, section_budget));
        offset_pages += this_pages;
    }

    let sub_sections = created.len();
    let any_committed = created.iter().any(|c| c.2);
    tracing::debug!(
        n,
        sub_sections,
        sec_committed = any_committed,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only committed sections can be
    // opened as a host heap.
    for &(handle, this_pages, sec_committed, _) in &created {
        victory_gpu::title_memory_section(handle as usize, (this_pages * PAGE) as u64, sec_committed);
    }
    let mut table = page_table_mut();
    let first = table.next_page;
    table.next_page += n * FRAMES;
    let mut page_offset = 0usize;
    for (handle, this_pages, sec_committed, section_budget) in created {
        let section = Arc::new(Section::new(
            handle,
            this_pages,
            first + page_offset * FRAMES,
            sec_committed,
            section_budget,
        ));
        for index in 0..this_pages {
            let slice = Slice {
                section: section.clone(),
                index,
            };
            table.pages.insert(first + (page_offset + index) * FRAMES, slice);
        }
        page_offset += this_pages;
    }
    TRACKED_PHYSICAL_PAGES.fetch_add(n, Ordering::Relaxed);
    // Title memory is touched without our locks: a fault there may wait for
    // a mapping call, which takes them.
    drop(table);
    for index in 0..n {
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(first + index * FRAMES) };
    }
    notify_page_waiters();
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` page numbers.
    let numbers = unsafe { std::slice::from_raw_parts(pages, n) }.to_vec();
    let mut table = page_table_mut();
    // Dropped without the lock: the last page of a sub-section evicts its
    // sliding windows and closes its handle immediately.
    let gone: Vec<Slice> = numbers
        .iter()
        .filter_map(|number| {
            let slice = table.pages.remove(number)?;
            slice.section.owners[slice.index].store(0, Ordering::Relaxed);
            Some(slice)
        })
        .collect();
    let freed = gone.len();
    drop(table);
    if freed > 0 {
        let _ = TRACKED_PHYSICAL_PAGES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_sub(freed))
        });
    }
    drop(gone);
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            tracing::warn!(error = %std::io::Error::last_os_error(), n, "physical pages not mapped for a write");
            return None;
        }
        let view = View { at, len };
        if !slice.section.sec_committed && !commit(at.Value as usize, len) {
            tracing::warn!(error = %std::io::Error::last_os_error(), n, "physical pages not committed for a write");
            return None;
        }
        // Mark touched so `arrange` / `arrange_fresh_run` never swaps written slices.
        for p in slice.index..(slice.index + n).min(slice.section.pages) {
            let bit = 1u64 << (p % 64);
            if slice.section.committed[p / 64].load(Ordering::Relaxed) & bit == 0 {
                slice.section.committed[p / 64].fetch_or(bit, Ordering::AcqRel);
            }
        }
        Some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let table = page_table();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let Some(slice) = table.pages.get(page) else {
            tracing::warn!(page, offset, len, "a physical page the title does not own (any more)");
            return None;
        };
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether the title owns every page under bytes `[offset, offset + len)`.
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    let table = page_table();
    pages
        .get(offset / PAGE..(offset + len).div_ceil(PAGE))
        .is_some_and(|pages| pages.iter().all(|page| table.pages.contains_key(page)))
}

/// Where the title sees the pages under bytes `[offset, offset + len)` now,
/// in order (for the streaming trace: whether that changed while a request
/// wrote them).
pub(crate) fn pages_shown(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let numbers = pages.get(offset / PAGE..(offset + len).div_ceil(PAGE)).unwrap_or_default();
    let slices: Vec<Slice> = {
        let table = page_table();
        numbers.iter().filter_map(|n| table.pages.get(n)).cloned().collect()
    };
    let shown = SHOWN.read().unwrap_or_else(|e| e.into_inner());
    let mut at: Vec<(usize, usize)> = slices.iter().flat_map(|slice| shown.at(slice)).collect();
    at.sort_unstable();
    at
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let k = (n * PAGE - within).min(data.len() - done);
        if let Some(mut guard) = slice.section.window(slice.index, *n) {
            guard.bytes_mut()[within..within + k].copy_from_slice(&data[done..done + k]);
        } else {
            let Some(mut view) = View::map(slice, *n) else {
                return false;
            };
            view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        }
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: Window,
    within: usize,
    len: usize,
}

enum Window {
    /// Sliding window guard kept mapped until dropped; `Arc<Section>` drops
    /// second so the section handle outlives the view.
    Kept(WindowGuard, #[expect(dead_code, reason = "held for the section")] Arc<Section>),
    Own(View),
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        match &mut self.view {
            Window::Kept(guard, _) => &mut guard.bytes_mut()[within..within + len],
            Window::Own(view) => &mut view.bytes()[within..within + len],
        }
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    let view = match slice.section.window(slice.index, *n) {
        Some(guard) => Window::Kept(guard, slice.section.clone()),
        None => Window::Own(View::map(slice, *n)?),
    };
    Some(PageWindow {
        view,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it. Empty if no view shows them. Byte for byte, not by
/// whole pages: the title packs several small textures into one page and
/// fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let slices: Vec<Option<Slice>> = {
        let table = page_table();
        touched.iter().map(|page| table.pages.get(page).cloned()).collect()
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let shown = SHOWN.read().unwrap_or_else(|e| e.into_inner());
        for (i, slice) in slices.iter().enumerate() {
            let Some(slice) = slice else { continue };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(shown.at(slice).into_iter().map(|(at, _)| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// DirectStorage has put bytes `[offset, offset + len)` of `pages` into
/// memory, through a view of our own: the GPU side is told where the title
/// sees them, so that what it copied there before they arrived (a texture
/// uploaded when its view was written holds whatever the pages held then)
/// is copied again. Call when the bytes are there.
/// `VICTORY_STREAM_NOTICE=0`: not told; what is then done is
/// `VICTORY_PAGE_WRITES`' to say (`victory_gpu::title_memory_written`).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    if len == 0 {
        return;
    }
    for (at, bytes) in shown_at(pages, offset, len) {
        if stream_notice() {
            victory_gpu::kernel_will_write(at, bytes);
        } else {
            victory_gpu::title_memory_written(at, bytes, true);
        }
    }
}

/// What the title's mapping calls cost since the last asking, for the
/// streaming log, counted only while that log is on: calls, pages,
/// milliseconds in all, the longest one's and waiting for the lock; the same
/// of the GPU's page-mapping copies ([`map_alias`]; calls and milliseconds);
/// and by system call (unmap, split, map, commit) how many and their
/// milliseconds.
static MAPPING: [std::sync::atomic::AtomicU64; 15] = [const { std::sync::atomic::AtomicU64::new(0) }; 15];

pub(crate) fn map_costs() -> Option<String> {
    let n: [u64; 15] = std::array::from_fn(|i| MAPPING[i].swap(0, Ordering::Relaxed));
    let time = |i: usize| matches!(i, 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14);
    (n[0] + n[5] > 0).then(|| {
        let all: Vec<String> = n
            .iter()
            .enumerate()
            .map(|(i, &n)| if time(i) { n / 1_000_000 } else { n })
            .map(|n| n.to_string())
            .collect();
        all.join("\t")
    })
}

/// The system calls that change a mapping, as [`MAPPING`] counts them.
#[derive(Clone, Copy)]
enum Call {
    Unmap = 7,
    Split = 9,
    Map = 11,
    Commit = 13,
}

/// Runs `f`, a system call, and counts it while the streaming log is on.
fn counted<T>(call: Call, f: impl FnOnce() -> T) -> T {
    if !super::dstorage::stream_log::on() {
        return f();
    }
    let started = std::time::Instant::now();
    let done = f();
    MAPPING[call as usize].fetch_add(1, Ordering::Relaxed);
    MAPPING[call as usize + 1].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    done
}

/// Counts a mapping call when dropped.
struct Costed {
    started: std::time::Instant,
    /// `None`: a copy of mappings.
    pages: Option<usize>,
    waited: u64,
}

impl Costed {
    fn begin(pages: Option<usize>) -> Option<Costed> {
        super::dstorage::stream_log::on().then(|| Costed { started: std::time::Instant::now(), pages, waited: 0 })
    }
}

impl Drop for Costed {
    fn drop(&mut self) {
        let ns = self.started.elapsed().as_nanos() as u64;
        let Some(pages) = self.pages else {
            MAPPING[5].fetch_add(1, Ordering::Relaxed);
            MAPPING[6].fetch_add(ns, Ordering::Relaxed);
            return;
        };
        for (i, n) in [1, pages as u64, ns].into_iter().enumerate() {
            MAPPING[i].fetch_add(n, Ordering::Relaxed);
        }
        MAPPING[3].fetch_max(ns, Ordering::Relaxed);
        MAPPING[4].fetch_add(self.waited, Ordering::Relaxed);
    }
}

/// Whether a mapping change asks the host only for what it needs: no split
/// of a placeholder that is already the view's own, no commit of section
/// pages that are committed (`VICTORY_MAP_LEAN=0`: both every time, as
/// before).
///
/// All of the title's streaming goes through one thread, which maps the
/// pages of a batch and then submits it. Each call waits for the process's
/// one address-space lock, behind the guards' and the driver's: 200 to 400
/// us a page in the world against 10 in a menu, 2 to 4 s of every 5 s of
/// that thread (runs stream-1 and -2; all six stacks taken of it in the
/// opening were in these calls), so that models came as fast as their pages
/// could be mapped. A page took up to four calls (unmap, split, map,
/// commit), twice: the title maps a page into its pool and then has the GPU
/// map it into its tiled heap.
fn lean() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("VICTORY_MAP_LEAN").is_none_or(|v| v != "0"));
    *ON
}

/// The title's streaming thread runs below normal priority, as do background
/// workers when pipelines are made; on the console nothing else wants its
/// core. Called where that thread maps and submits: lifts a thread that is
/// below normal to normal priority (`VICTORY_STREAM_PRIORITY=0`: left as the
/// title set it) without preempting title render threads.
pub(crate) fn lift_thread() {
    use std::cell::Cell;
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetThreadPriority, SetThreadPriority, THREAD_PRIORITY_NORMAL,
    };
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("VICTORY_STREAM_PRIORITY").is_none_or(|v| v != "0"));
    if !*ON {
        return;
    }
    thread_local! {
        static CHECKED: Cell<bool> = const { Cell::new(false) };
    }
    if CHECKED.with(Cell::get) {
        return;
    }
    CHECKED.with(|c| c.set(true));
    // SAFETY: plain calls on this thread.
    unsafe {
        let thread = GetCurrentThread();
        if matches!(GetThreadPriority(thread), -2..0) {
            SetThreadPriority(thread, THREAD_PRIORITY_NORMAL);
        }
    }
}

/// Expanded in-place placement (`VICTORY_PORT_EXPAND` non-zero, or
/// `VICTORY_INPLACE_ALL=1`): enables `SEC_COMMIT` backing and 64-bit fresh-slice
/// arrangement so 64-bit placeholder pools become contiguous section stretches.
fn port_expand_on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("VICTORY_PORT_EXPAND")
            .ok()
            .is_some_and(|v| matches!(v.trim(), "1" | "2" | "3" | "safe" | "sections" | "all"))
            || std::env::var("VICTORY_INPLACE_ALL").is_ok_and(|v| v.trim() == "1")
    });
    *ON
}

/// Renderer port 1 (`VICTORY_PORT_CONSTANTS=1`, or expanded in-place mode):
/// read here for xmem's part, [`arrange`].
fn port_on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("VICTORY_PORT_CONSTANTS").is_ok_and(|v| v.trim() == "1") || port_expand_on()
    });
    *ON
}

/// The title's reservations of 64 KiB pages for 32-bit graphics memory:
/// base and bytes (one, of 185 MB, in the title).
static LOW_RESERVATIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// The title's 64-bit placeholder reservations (`MEM_RESERVE_PLACEHOLDER`), used
/// when [`port_expand_on`] is active.
static HIGH_RESERVATIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

/// Pages given another place in their section so far by [`arrange`], and
/// the times it found a page that was not fresh.
static ARRANGED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Finds the first contiguous run of `need` fresh (never mapped, never written)
/// slice indices in `section` using word-level skipping.
fn find_fresh_run(section: &Section, need: usize) -> Option<usize> {
    if need == 0 || need > section.pages {
        return None;
    }
    let words = section.committed.len();
    let start_word = section.fresh_word.load(Ordering::Relaxed).min(words);
    let mut run_start = 0usize;
    let mut run_len = 0usize;
    let mut all_full_so_far = true;
    for w in start_word..words {
        let word_start = w * 64;
        let valid_bits = (section.pages - word_start).min(64);
        let raw = section.committed[w].load(Ordering::Acquire);
        let occupied = if valid_bits < 64 {
            raw | (!0u64 << valid_bits)
        } else {
            raw
        };
        if occupied == u64::MAX {
            run_len = 0;
            if all_full_so_far {
                section.fresh_word.fetch_max(w + 1, Ordering::Relaxed);
            }
            continue;
        }
        all_full_so_far = false;
        if occupied == 0 {
            if run_len == 0 {
                run_start = word_start;
            }
            run_len += 64;
            if run_len >= need {
                return Some(run_start);
            }
            continue;
        }
        let mut bits = occupied;
        let mut bit = 0usize;
        while bit < valid_bits {
            if bits & 1 == 0 {
                let zeros = (bits.trailing_zeros() as usize).min(valid_bits - bit);
                if run_len == 0 {
                    run_start = word_start + bit;
                }
                run_len += zeros;
                if run_len >= need {
                    return Some(run_start);
                }
                bit += zeros;
                bits = if zeros == 64 { 0 } else { bits >> zeros };
            } else {
                let ones = (bits.trailing_ones() as usize).min(valid_bits - bit);
                run_len = 0;
                bit += ones;
                bits = if ones == 64 { 0 } else { bits >> ones };
            }
        }
    }
    None
}

/// Reorders `pages` (when `pages.len() > 1`, distinct, all fresh, all from one
/// section, and not already ascending consecutive slices) into a single contiguous
/// run of fresh slices in that section so `XMemMapPhysicalPages` maps them as
/// one contiguous view (`Cover::Run`).
fn arrange_fresh_run(pages: &[usize]) -> u64 {
    if pages.len() < 2 {
        return 0;
    }
    // Fast path under read lock: verify single section, all fresh, and not already consecutive.
    {
        let table = page_table();
        let Some(first) = table.pages.get(&pages[0]) else {
            return 0;
        };
        let section = &first.section;
        if section.pages < pages.len() {
            return 0;
        }
        let mut already_consecutive = true;
        for (i, p) in pages.iter().enumerate() {
            let Some(s) = table.pages.get(p) else {
                return 0;
            };
            if !Arc::ptr_eq(&s.section, section) {
                return 0;
            }
            if section.committed[s.index / 64].load(Ordering::Acquire) >> (s.index % 64) & 1 != 0 {
                return 0;
            }
            if s.index != first.index + i {
                already_consecutive = false;
            }
        }
        if already_consecutive || find_fresh_run(section, pages.len()).is_none() {
            return 0;
        }
    }
    // Reject duplicate page numbers within the same mapping call before mutating.
    let mut seen = std::collections::HashSet::with_capacity(pages.len());
    if !pages.iter().all(|&p| seen.insert(p)) {
        return 0;
    }
    let mut table = page_table_mut();
    let Some(section) = pages.first().and_then(|p| table.pages.get(p)).map(|s| s.section.clone()) else {
        return 0;
    };
    let fresh = |index: usize| section.committed[index / 64].load(Ordering::Acquire) >> (index % 64) & 1 == 0;
    if !pages.iter().all(|p| {
        table
            .pages
            .get(p)
            .is_some_and(|s| Arc::ptr_eq(&s.section, &section) && fresh(s.index))
    }) {
        return 0;
    }
    let Some(run_start) = find_fresh_run(&section, pages.len()) else {
        return 0;
    };
    let mut moved = 0u64;
    for (i, &number) in pages.iter().enumerate() {
        let want = run_start + i;
        let Some(have) = table.pages.get(&number).map(|s| s.index) else {
            continue;
        };
        if have == want {
            continue;
        }
        let other = section.owners[want].load(Ordering::Relaxed);
        if other != 0 {
            if let Some(slice) = table.pages.get_mut(&other) {
                slice.index = have;
            }
            section.owners[have].store(other, Ordering::Relaxed);
        } else {
            section.owners[have].store(0, Ordering::Relaxed);
        }
        if let Some(slice) = table.pages.get_mut(&number) {
            slice.index = want;
        }
        section.owners[want].store(number, Ordering::Relaxed);
        moved += 1;
    }
    if moved > 0 {
        ARRANGED[0].fetch_add(moved, Ordering::Relaxed);
    }
    moved
}

/// Before `pages` are mapped at `base`: if `base` lies in a tracked reservation
/// ([`LOW_RESERVATIONS`], or [`HIGH_RESERVATIONS`] when expanded) whose pages
/// are all one section of at least the reservation size, the page numbers are
/// given the slices of that section in the reservation's address order so the
/// reservation becomes one stretch of the section. Under [`port_expand_on`],
/// also coalesces multi-page mappings of fresh single-section pages into a
/// contiguous fresh run even when a reservation is backed by multiple sections.
///
/// The title takes all the pages of its 185 MB heap in one call
/// and maps its five pieces in another order than their addresses (run
/// port1-1: "Scattered"). Only fresh slices are exchanged: never mapped,
/// never written (no commit bit), so every page still reads as zeros and
/// which slice a number names is not observable by the title.
fn arrange(base: usize, pages: &[usize]) {
    let reservation = LOW_RESERVATIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|&(at, len)| at <= base && base + pages.len() * PAGE <= at + len)
        .or_else(|| {
            port_expand_on().then(|| {
                HIGH_RESERVATIONS
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .copied()
                    .find(|&(at, len)| at <= base && base + pages.len() * PAGE <= at + len)
            })?
        });
    let Some((first, bytes)) = reservation else {
        if port_expand_on() {
            arrange_fresh_run(pages);
        }
        return;
    };
    let start_index = (base - first) / PAGE;
    // Fast path under read lock: skip the write lock when pages are already in
    // address order or ineligible.
    {
        let table = page_table();
        let Some(section) = pages.first().and_then(|p| table.pages.get(p)).map(|s| &s.section)
        else {
            return;
        };
        if section.pages * PAGE < bytes
            || !pages
                .iter()
                .all(|p| table.pages.get(p).is_some_and(|s| Arc::ptr_eq(&s.section, section)))
        {
            drop(table);
            if port_expand_on() {
                arrange_fresh_run(pages);
                return;
            }
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 4 {
                tracing::warn!(
                    at = format_args!("{base:#x}"),
                    pages = pages.len(),
                    reservation = bytes / PAGE,
                    "renderer port 1: pages mapped into the 32-bit heap are not of one section as big as it: left where they are"
                );
            }
            return;
        }
        if pages
            .iter()
            .enumerate()
            .all(|(i, p)| table.pages.get(p).is_some_and(|s| s.index == start_index + i))
        {
            return;
        }
    }
    let mut table = page_table_mut();
    let Some(section) = pages.first().and_then(|p| table.pages.get(p)).map(|s| s.section.clone()) else {
        return;
    };
    if section.pages * PAGE < bytes
        || !pages.iter().all(|p| table.pages.get(p).is_some_and(|s| Arc::ptr_eq(&s.section, &section)))
    {
        return;
    }
    let fresh = |index: usize| section.committed[index / 64].load(Ordering::Acquire) >> (index % 64) & 1 == 0;
    let mut moved = 0u64;
    for (i, &number) in pages.iter().enumerate() {
        let want = start_index + i;
        let Some(have) = table.pages.get(&number).map(|s| s.index) else {
            continue;
        };
        if have == want {
            continue;
        }
        if !fresh(have) || !fresh(want) {
            ARRANGED[1].fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let other = section.owners[want].load(Ordering::Relaxed);
        if other != 0 {
            if let Some(slice) = table.pages.get_mut(&other) {
                slice.index = have;
            }
            section.owners[have].store(other, Ordering::Relaxed);
        } else {
            section.owners[have].store(0, Ordering::Relaxed);
        }
        if let Some(slice) = table.pages.get_mut(&number) {
            slice.index = want;
        }
        section.owners[want].store(number, Ordering::Relaxed);
        moved += 1;
    }
    if moved > 0 {
        let all = ARRANGED[0].fetch_add(moved, Ordering::Relaxed) + moved;
        tracing::info!(
            at = format_args!("{base:#x}"),
            pages = pages.len(),
            moved,
            all,
            not_fresh = ARRANGED[1].load(Ordering::Relaxed),
            "renderer port 1: the title's 32-bit graphics pages given their section's slices in address order"
        );
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    // Read before any lock is taken: a fault in title memory may wait for
    // the mapping call that holds them.
    // SAFETY: the title passes `n` page numbers, or null.
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) }.to_vec());
    let mut costed = Costed::begin(Some(n));
    let _called = victory_gpu::timeline::span("the title's XMemMapPhysicalPages");
    let _busy = Busy::begin();
    let mut physical = physical();
    if let Some(costed) = &mut costed {
        costed.waited = costed.started.elapsed().as_nanos() as u64;
    }
    if port_on()
        && let Some(pages) = &pages
    {
        arrange(base, pages);
    }
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, base, n * PAGE);
    let Some(pages) = pages else {
        return address;
    };
    let slices: Vec<Option<Slice>> = {
        let table = page_table();
        pages.iter().map(|page| table.pages.get(page).cloned()).collect()
    };
    let mut i = 0;
    while i < n {
        let Some(first) = slices[i].clone() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + slices[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(slice, k)| {
                slice.as_ref().is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    drop(_changing);
    drop(physical);
    notify_page_waiters();
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    lift_thread();
    if n > 1 {
        // Read before any lock is taken, as in `XMemMapPhysicalPages`.
        let wanted: Vec<(usize, usize)> = (0..n)
            // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
            .map(|i| unsafe { (*addresses.add(i) as usize, if pages.is_null() { 0 } else { *pages.add(i) }) })
            .collect();
        if wanted.iter().all(|w| w.0 % PAGE == 0) {
            let ok = scatter(&wanted);
            notify_page_waiters();
            return ok.into();
        }
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Coalesces consecutive slices of one section into multi-page views;
/// false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let _costed = Costed::begin(None);
    let _busy = Busy::begin();
    let mut physical = physical();
    let runs: Vec<(usize, usize, Slice)> = if from != 0 {
        let mut runs: Vec<(usize, usize, Slice)> = Vec::new();
        for k in (0..len).step_by(PAGE) {
            let Some(slice) = physical.slice_at(from + k) else {
                return false;
            };
            match runs.last_mut() {
                Some((run_offset, run_pages, first))
                    if *run_offset + *run_pages * PAGE == k
                        && Arc::ptr_eq(&first.section, &slice.section)
                        && slice.index == first.index + *run_pages =>
                {
                    *run_pages += 1;
                }
                _ => runs.push((k, 1, slice)),
            }
        }
        runs
    } else {
        Vec::new()
    };
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, to, len);
    for (run_offset, run_pages, first_slice) in runs {
        if !map(&mut physical, to + run_offset, run_pages * PAGE, first_slice) {
            return false;
        }
    }
    true
}

/// Whether `(to, slice)` extends `last` into one contiguous multi-page op.
fn can_coalesce(last: &Op, to: usize, slice: &Option<Slice>) -> bool {
    if last.at + last.len != to {
        return false;
    }
    match (&last.slice, slice) {
        (None, None) => true,
        (Some(prev), Some(next)) => {
            Arc::ptr_eq(&prev.section, &next.section) && next.index == prev.index + last.len / PAGE
        }
        _ => false,
    }
}

/// [`map_alias`] of many ranges, in order.
pub fn map_aliases(ranges: &[(u64, u64, u64)]) -> bool {
    let _costed = Costed::begin(None);
    lift_thread();
    if ranges.len() <= 1 {
        return ranges.iter().all(|&(to, from, len)| map_alias(to, from, len));
    }
    // By the page: the address, and the one whose page it is to show.
    let pages: Vec<(usize, Option<usize>)> = ranges
        .iter()
        .flat_map(|&(to, from, len)| {
            (0..len as usize).step_by(PAGE).map(move |k| (to as usize + k, (from != 0).then_some(from as usize + k)))
        })
        .collect();
    if pages.iter().any(|p| p.0 % PAGE != 0 || p.1.is_some_and(|from| from % PAGE != 0)) {
        return ranges.iter().all(|&(to, from, len)| map_alias(to, from, len));
    }
    let mut ok = true;
    let mut at = 0;
    while at < pages.len() {
        let _busy = Busy::begin();
        let mut physical = physical();
        let mut ops: Vec<Op> = Vec::new();
        let mut group_pages = 0usize;
        let mut touched = std::collections::HashSet::new();
        for &(to, from) in &pages[at..] {
            // A page the group changes already, or one it would be mapped
            // from: after it.
            if group_pages >= MAX_GROUP_PAGES
                || touched.contains(&to)
                || from.is_some_and(|from| touched.contains(&from))
            {
                break;
            }
            let slice = from.map(|from| physical.slice_at(from));
            let missing = matches!(slice, Some(None));
            let resolved = slice.flatten();
            if let Some(last) = ops.last_mut().filter(|last| can_coalesce(last, to, &resolved)) {
                last.len += PAGE;
            } else {
                if ops.len() == GROUP {
                    break;
                }
                ops.push(Op::new(to, resolved));
            }
            ok &= !missing;
            touched.insert(to);
            group_pages += 1;
            at += 1;
        }
        let _changing = victory_gpu::title_memory_remapping();
        ok &= change(&mut physical, ops);
    }
    ok
}

/// Maps (page number not 0) or unmaps each address; false if a page is not
/// the title's or the host refuses one.
fn scatter(wanted: &[(usize, usize)]) -> bool {
    let mut ok = true;
    let mut at = 0;
    while at < wanted.len() {
        let _busy = Busy::begin();
        let mut costed = Costed::begin(None);
        let mut ops: Vec<Op> = Vec::new();
        let mut group_pages = 0usize;
        let mut physical = physical();
        if let Some(costed) = &mut costed {
            costed.waited = costed.started.elapsed().as_nanos() as u64;
        }
        {
            let table = page_table();
            let mut touched = std::collections::HashSet::new();
            for &(to, page) in &wanted[at..] {
                if group_pages >= MAX_GROUP_PAGES || touched.contains(&to) {
                    break;
                }
                let slice = if page == 0 {
                    None
                } else {
                    table.pages.get(&page).cloned()
                };
                let unknown = page != 0 && slice.is_none();
                if let Some(last) = ops.last_mut().filter(|last| can_coalesce(last, to, &slice)) {
                    last.len += PAGE;
                } else {
                    if ops.len() == GROUP {
                        break;
                    }
                    ops.push(Op::new(to, slice));
                }
                if unknown {
                    tracing::warn!(page, "XMemMapPhysicalPagesScatter: unknown page");
                    ok = false;
                }
                touched.insert(to);
                group_pages += 1;
                at += 1;
            }
        }
        if let Some(costed) = &mut costed {
            costed.pages = Some(group_pages);
        }
        let _changing = victory_gpu::title_memory_remapping();
        ok &= change(&mut physical, ops);
    }
    ok
}

/// Distinct view ops changed under the locks at once.
const GROUP: usize = 64;
/// Upper bound on total pages coalesced into one group so [`PHYSICAL`] is not
/// held indefinitely on huge ranges.
const MAX_GROUP_PAGES: usize = GROUP * 16;

/// How many threads map a group of pages side by side (`VICTORY_MAP_THREADS`;
/// 8 unless set; 1: the caller alone, a page after the other; `auto`: 1 while
/// mapping takes little of the time, [`WIDE`] while the title streams as fast
/// as its pages are mapped, [`Busy`]).
///
/// shortcut: `auto` is not the default. The intro's diner was there in time
/// with it (one run, textures-d3), but in the dorm, where each call takes 0.3
/// ms of something that only one at a time gets (until a step of the mission,
/// after which it takes 0.01), 16 threads map no more pages than one, and the
/// groups hold the mapping lock: 260 ms a frame against 125 (textures-d4, -d2).
///
/// Mapping and unmapping a view wait for a lock that every protection change
/// of a view takes too, and our guards make 100,000 of those a second, from
/// many threads, in the world: 0.2 to 4 ms a page. One thread stands in that
/// queue once at a time. In a bench (60,000 views, five threads changing
/// protections) one thread mapped 1,500 pages a second, eight 30,000.
fn side_by_side() -> usize {
    // Diagnostics: 1 and 8 by turns.
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some(on) = victory_gpu::switched("VICTORY_MAP_THREADS") {
        return if on { 8 } else { 1 };
    }
    match asked_threads() {
        Some(n) => n,
        None if BUSY.2.load(Ordering::Relaxed) => WIDE,
        None => 1,
    }
}

/// Threads while the title streams fast: they stand in the host's queue side
/// by side, each for a millisecond or two a page in the world's first seconds.
const WIDE: usize = 16;

/// `VICTORY_MAP_THREADS` as a number (default `1` to avoid multi-thread
/// `AddressCreationLock` / VAD contention); `None` (`auto`): adaptive.
fn asked_threads() -> Option<usize> {
    static ASKED: LazyLock<Option<usize>> = LazyLock::new(|| {
        match std::env::var("VICTORY_MAP_THREADS") {
            Ok(v) if v.trim() == "auto" => None,
            v => Some(v.ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1).clamp(1, 32)),
        }
    });
    *ASKED
}

/// Of the window under way: when it began, the nanoseconds spent mapping in
/// it; and whether the last one was busy.
static BUSY: (Mutex<Option<std::time::Instant>>, AtomicU64, std::sync::atomic::AtomicBool) =
    (Mutex::new(None), AtomicU64::new(0), std::sync::atomic::AtomicBool::new(false));

/// Counts the time a mapping call of the title's takes, as long as it lives.
/// More than a quarter of a window, and the calls are what loading waits for
/// (one thread of the title's makes them); less than a twentieth, they are
/// not any more.
struct Busy(Option<std::time::Instant>);

impl Busy {
    fn begin() -> Busy {
        if asked_threads().is_some() {
            return Busy(None);
        }
        Busy(Some(std::time::Instant::now()))
    }
}

/// Whether mapping is busy after a window of `window` with `spent` in it.
fn busy_after(was: bool, spent: std::time::Duration, window: std::time::Duration) -> bool {
    spent * if was { 20 } else { 4 } > window
}

impl Drop for Busy {
    fn drop(&mut self) {
        let Some(started) = self.0 else {
            return;
        };
        const WINDOW: std::time::Duration = std::time::Duration::from_millis(250);
        let now = std::time::Instant::now();
        let elapsed = (now - started).as_nanos() as u64;
        let all = BUSY.1.fetch_add(elapsed, Ordering::Relaxed) + elapsed;
        let mut began = BUSY.0.lock().unwrap_or_else(|e| e.into_inner());
        let window = now - *began.get_or_insert(started);
        if window < WINDOW {
            return;
        }
        *began = Some(now);
        BUSY.1.store(0, Ordering::Relaxed);
        let was = BUSY.2.load(Ordering::Relaxed);
        let is = busy_after(was, std::time::Duration::from_nanos(all), window);
        if was != is {
            BUSY.2.store(is, Ordering::Relaxed);
            static SAID: AtomicU64 = AtomicU64::new(0);
            if asked_threads().is_none() && SAID.fetch_add(1, Ordering::Relaxed) < 200 {
                tracing::info!(
                    "title pages are mapped by {} (VICTORY_MAP_THREADS): {:.0} ms of the last {:.0} ms went into mapping calls",
                    if is { "several threads" } else { "the caller alone" },
                    all as f64 / 1e6,
                    window.as_secs_f64() * 1e3
                );
            }
        }
    }
}

/// A contiguous range of address space to unmap and, with a slice, map anew.
struct Op {
    at: usize,
    len: usize,
    slice: Option<Slice>,
    /// A view of this exact range is there, to be unmapped.
    had: bool,
    /// The placeholder is this range's own (or will be, after the unmapping).
    own: bool,
}

impl Op {
    fn new(at: usize, slice: Option<Slice>) -> Op {
        Op::new_run(at, PAGE, slice)
    }

    fn new_run(at: usize, len: usize, slice: Option<Slice>) -> Op {
        Op { at, len, slice, had: false, own: false }
    }

    /// The system calls; whether the host did as asked.
    fn run(&self) -> bool {
        if self.had {
            // SAFETY: a view we mapped; it becomes a placeholder again.
            let gone = counted(Call::Unmap, || unsafe {
                UnmapViewOfFile2(
                    GetCurrentProcess(),
                    MEMORY_MAPPED_VIEW_ADDRESS { Value: self.at as *mut c_void },
                    MEM_PRESERVE_PLACEHOLDER,
                )
            });
            if gone == 0 {
                tracing::warn!(
                    address = format_args!("{:#x}", self.at),
                    len = self.len,
                    error = %std::io::Error::last_os_error(),
                    "unmapping physical pages failed"
                );
                return false;
            }
        }
        let Some(slice) = &self.slice else {
            return true;
        };
        show(self.at, self.len, slice, self.own)
    }
}

/// Does `ops`, each of a disjoint range, side by side: first what is noted of
/// the views goes, then the system calls, then what is there now is noted.
/// With [`PHYSICAL`] and the change of mapping (`title_memory_remapping`)
/// held.
fn change(physical: &mut Physical, mut ops: Vec<Op>) -> bool {
    // Merge covered intervals of ops that do not match an exact existing view
    // so unmapping or overwriting pieces of a larger view unmaps each covered
    // stretch once instead of unmapping and remapping a shrinking remainder
    // per op. Exact existing views stay untouched for parallel unmapping.
    let mut non_exact: Vec<(usize, usize)> = ops
        .iter()
        .filter(|op| physical.views.get(&op.at).is_none_or(|(size, _)| *size != op.len))
        .map(|op| (op.at, op.at + op.len))
        .collect();
    if !non_exact.is_empty() {
        non_exact.sort_unstable_by_key(|&(start, _)| start);
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(non_exact.len());
        for (start, end) in non_exact {
            match merged.last_mut() {
                Some((_, last_end)) if *last_end == start => *last_end = end,
                _ => merged.push((start, end)),
            }
        }
        for (start, end) in merged {
            if physical.overlaps_view(start, end) {
                unmap(physical, start, end - start);
            }
        }
    }

    let mut to_forget: Vec<(usize, usize, Slice)> = Vec::new();
    for op in &mut ops {
        if physical.views.get(&op.at).is_some_and(|(size, _)| *size == op.len) {
            if let Some((size, slice)) = physical.views.remove(&op.at) {
                to_forget.push((op.at, size, slice));
            }
            victory_gpu::title_memory_unmapping(op.at, op.len);
            physical.clear_holes(op.at, op.at + op.len);
            (op.had, op.own) = (true, lean());
            continue;
        }
        if op.slice.is_some() {
            let exact_hole = physical.holes.get(&op.at) == Some(&op.len);
            physical.clear_holes(op.at, op.at + op.len);
            op.own = exact_hole && lean();
            if !op.own {
                // Cut out of (or coalesce within) placeholders here, not side by
                // side: a thread that maps a piece while others cut the same one
                // can fail.
                prepare_placeholder(op.at, op.len);
                op.own = true;
            }
        }
    }
    if !to_forget.is_empty() {
        let mut shown = shown_mut();
        for (at, len, slice) in &to_forget {
            shown.forget(*at, *len, slice);
        }
        bump_generation();
    }
    if ops.iter().all(|op| !op.had && op.slice.is_none()) {
        return true;
    }
    let done = pool::all(ops);
    let mut ok = true;
    let mut to_note: Vec<(usize, usize, Slice)> = Vec::new();
    for (op, done) in done {
        match op.slice {
            Some(slice) if done => {
                to_note.push((op.at, op.len, slice));
            }
            None if op.had && done => {
                physical.clear_holes(op.at, op.at + op.len);
                physical.holes.insert(op.at, op.len);
            }
            _ => ok &= done,
        }
    }
    if !to_note.is_empty() {
        {
            let mut shown = shown_mut();
            for (at, len, slice) in &to_note {
                shown.note(*at, *len, slice);
            }
            bump_generation();
        }
        for (at, len, slice) in to_note {
            // As in `map`.
            victory_gpu::title_memory_remapped(at, len);
            victory_gpu::title_memory_view(
                at,
                len,
                slice.section.handle as usize,
                (slice.index * PAGE) as u64,
            );
            physical.views.insert(at, (len, slice));
        }
    }
    ok
}

/// The threads that do the system calls of [`change`].
mod pool {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, OnceLock};

    use super::{Op, side_by_side};

    struct Batch {
        ops: Vec<(Op, AtomicBool)>,
        next: AtomicUsize,
        /// Ops not done yet.
        left: AtomicUsize,
        over: (Mutex<bool>, Condvar),
    }

    impl Batch {
        /// Does ops until none is left to begin.
        fn work(&self) {
            loop {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                let Some((op, done)) = self.ops.get(i) else {
                    return;
                };
                done.store(op.run(), Ordering::Release);
                if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
                    *self.over.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
                    self.over.1.notify_all();
                }
            }
        }
    }

    struct Pool {
        batches: Mutex<VecDeque<Arc<Batch>>>,
        more: Condvar,
    }

    fn pool() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        static STARTED: std::sync::Once = std::sync::Once::new();
        let pool = POOL.get_or_init(|| Pool { batches: Mutex::default(), more: Condvar::new() });
        STARTED.call_once(|| {
            let started = (1..super::asked_threads().unwrap_or(super::WIDE).max(side_by_side()))
                .filter(|i| std::thread::Builder::new().name(format!("mapping {i}")).spawn(|| serve(pool)).is_ok())
                .count();
            tracing::info!(started, "threads that map pages side by side (VICTORY_MAP_THREADS=1: none)");
        });
        pool
    }

    fn serve(pool: &Pool) {
        let mut batches = pool.batches.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let Some(batch) = batches.front().cloned() else {
                batches = pool.more.wait(batches).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            if batch.next.load(Ordering::Relaxed) >= batch.ops.len() {
                // All begun: nobody needs to see it any more.
                if batches.front().is_some_and(|b| Arc::ptr_eq(b, &batch)) {
                    batches.pop_front();
                }
                continue;
            }
            drop(batches);
            batch.work();
            batches = pool.batches.lock().unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Runs `ops`; each with whether the host did it.
    pub(super) fn all(ops: Vec<Op>) -> Vec<(Op, bool)> {
        // Run inline when single-threaded (`VICTORY_MAP_THREADS=1`) or too small
        // to justify waking worker threads.
        if ops.len() < 3 || side_by_side() <= 1 {
            return ops
                .into_iter()
                .map(|op| {
                    let done = op.run();
                    (op, done)
                })
                .collect();
        }
        let n = ops.len();
        let batch = Arc::new(Batch {
            ops: ops.into_iter().map(|op| (op, AtomicBool::new(false))).collect(),
            next: AtomicUsize::new(0),
            left: AtomicUsize::new(n),
            over: Default::default(),
        });
        let pool = pool();
        pool.batches.lock().unwrap_or_else(|e| e.into_inner()).push_back(batch.clone());
        if n > side_by_side() {
            pool.more.notify_all();
        } else {
            (1..n).for_each(|_| pool.more.notify_one());
        }
        batch.work();
        let mut over = batch.over.0.lock().unwrap_or_else(|e| e.into_inner());
        while !*over {
            over = batch.over.1.wait(over).unwrap_or_else(|e| e.into_inner());
        }
        drop(over);
        // No worker changes it any more (one may still hold it for a moment).
        batch
            .ops
            .iter()
            .map(|(op, done)| {
                let op = Op {
                    at: op.at,
                    len: op.len,
                    slice: op.slice.clone(),
                    had: op.had,
                    own: op.own,
                };
                (op, done.load(Ordering::Acquire))
            })
            .collect()
    }
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder (or adjacent placeholders of one reservation).
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    let exact_hole = physical.holes.get(&base) == Some(&len);
    physical.clear_holes(base, base + len);
    let own = exact_hole && lean();
    if !show(base, len, &slice, own) {
        return false;
    }
    // Noted before the GPU side is told: bytes DirectStorage put in earlier,
    // without notice at these addresses ([`pages_written`]), are there by
    // the time the copies of them go.
    {
        let mut shown = shown_mut();
        shown.note(base, len, &slice);
        bump_generation();
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(base, len, slice.section.handle as usize, (slice.index * PAGE) as u64);
    physical.views.insert(base, (len, slice));
    true
}

/// Cuts `[base, base + len)` out of the placeholder it is in (fails harmlessly
/// when the range is already one whole placeholder or spans several).
fn split(base: usize, len: usize) -> bool {
    // SAFETY: acts only on a placeholder.
    counted(Call::Split, || unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    }) != 0
}

/// Ensures `[base, base + len)` is a single placeholder ready for
/// `MapViewOfFile3`, splitting a larger placeholder or coalescing adjacent
/// placeholders within the same reservation.
fn prepare_placeholder(base: usize, len: usize) {
    if split(base, len) || len <= PAGE {
        return;
    }
    let coalesce = || {
        // SAFETY: acts only on adjacent placeholders from one reservation.
        counted(Call::Split, || unsafe {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
            )
        }) != 0
    };
    if coalesce() {
        return;
    }
    // `VirtualQueryEx` merges adjacent identical placeholders in its output,
    // hiding internal placeholder boundaries. Splitting the first and last
    // pages guarantees no placeholder overhangs either endpoint so coalescing
    // `[base, base + len)` into one placeholder succeeds.
    split(base, PAGE);
    split(base + len - PAGE, PAGE);
    coalesce();
}

/// The system calls of [`map`]; `own`: the placeholder at `base` is `len`
/// long already.
fn show(base: usize, len: usize, slice: &Slice, own: bool) -> bool {
    let map_view = || {
        // SAFETY: maps the section over a placeholder; acts only on one.
        counted(Call::Map, || unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                base as *const c_void,
                (slice.index * PAGE) as u64,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        })
    };
    let mut view = if own { map_view() } else { MEMORY_MAPPED_VIEW_ADDRESS { Value: std::ptr::null_mut() } };
    if view.Value as usize != base {
        prepare_placeholder(base, len);
        view = map_view();
    }
    if view.Value as usize != base {
        let error = std::io::Error::last_os_error();
        let mut mbi = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: querying our own address space into an initialized struct.
        let queried = unsafe {
            VirtualQueryEx(
                GetCurrentProcess(),
                base as *const c_void,
                &mut mbi,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        } != 0;
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            own,
            %error,
            mbi_state = format_args!("{:#x}", if queried { mbi.State } else { 0 }),
            mbi_type = format_args!("{:#x}", if queried { mbi.Type } else { 0 }),
            mbi_region = if queried { mbi.RegionSize } else { 0 },
            mbi_protect = format_args!("{:#x}", if queried { mbi.Protect } else { 0 }),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !slice.section.commit_shown(slice.index, base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            queried.then_some(&info),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of {what}: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        {
            let mut shown = shown_mut();
            shown.forget(at, size, &slice);
            bump_generation();
        }
        if at < base || at + size > end {
            // Parts of it stay, and are mapped again below.
            victory_gpu::title_memory_splitting(at, size);
        }
        victory_gpu::title_memory_unmapping(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        let gone = counted(Call::Unmap, || unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        });
        physical.clear_holes(at, at + size);
        if gone != 0 && at >= base && at + size <= end {
            physical.holes.insert(at, size);
        }
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and tracked title memory usage (virtual allocations,
/// physical pages, and default heap — excluding the host layer's VRAM/SHR/cache
/// commit). No tools memory: a retail console.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let physical_pages = TRACKED_PHYSICAL_PAGES.load(Ordering::Relaxed);
    let tracked_used = TRACKED_VIRTUAL_BYTES
        .load(Ordering::Relaxed)
        .saturating_add(physical_pages.saturating_mul(PAGE))
        .saturating_add(TRACKED_HEAP_BYTES.load(Ordering::Relaxed));
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: if tracked_used > 0 {
            tracked_used
        } else {
            counters.PagefileUsage
        },
        physical_pages: physical_pages.min(u32::MAX as usize) as u32,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            process_mb = counters.PagefileUsage >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

/// The title's own `VirtualFree`. What the GPU side noted of the memory goes
/// first: the address may be other memory's the moment it is free.
extern "system" fn title_virtual_free(address: *mut c_void, size: usize, kind: u32) -> BOOL {
    // Splitting and joining placeholders changes nothing they show.
    if kind & (MEM_PRESERVE_PLACEHOLDER | MEM_COALESCE_PLACEHOLDERS) == 0 {
        let at = address as usize;
        if kind & MEM_RELEASE != 0 {
            LOW_RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()).retain(|r| r.0 != at);
            HIGH_RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner()).retain(|r| r.0 != at);
        }
        private_region_free(at, size, kind);
        victory_gpu::title_memory_freed(at, size);
        if kind == MEM_RELEASE && size == 0 {
            if let Some(recycled_len) = try_recycle_virtual_release(at) {
                if recycled_len > size {
                    victory_gpu::title_memory_freed(at, recycled_len);
                }
                return 1;
            }
        } else {
            untrack_recyclable_range(at, size);
        }
    }
    // SAFETY: the title's own arguments.
    unsafe { VirtualFree(address, size, kind) }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll", "kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-*"],
        exports: vec![
            crate::HleExport { name: "VirtualFree", address: title_virtual_free as *const () as usize },
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn the_32_bit_heap_gets_its_slices_in_address_order() {
        // As the title's heap: a 32-bit reservation of 64 KiB pages, all of
        // its pages from one call, mapped piece by piece in another order.
        // The title's own kind: reserve and 0x40000, no 64 KiB page bits.
        let base = XMemVirtualAlloc(std::ptr::null_mut(), 4 * PAGE, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, GRAPHICS | ADDRESS_32BIT, PAGE_NOACCESS);
        assert!(!base.is_null() && (base as usize) < 1 << 32);
        assert!(LOW_RESERVATIONS.lock().unwrap().iter().any(|r| r.0 == base as usize), "the reservation is known");
        let pages = allocate(4);
        let index = |number: usize| page_table().pages.get(&number).map(|s| s.index);
        let at = |page: usize| base.wrapping_byte_add(page * PAGE);
        // The last two pages' place first, from the array's start.
        arrange(at(2) as usize, &pages[..2]);
        assert_eq!(XMemMapPhysicalPages(at(2), 2, pages.as_ptr()), at(2));
        arrange(base as usize, &pages[2..]);
        assert_eq!(XMemMapPhysicalPages(base, 2, pages[2..].as_ptr()), base);
        assert_eq!([index(pages[0]), index(pages[1]), index(pages[2]), index(pages[3])], [Some(2), Some(3), Some(0), Some(1)]);
        // Mapped already: nothing moves any more.
        arrange(base as usize, &pages[..2]);
        assert_eq!(index(pages[0]), Some(2));
        // SAFETY: mapped read-write.
        unsafe { at(3).cast::<u32>().write(7) };
        assert_eq!(XMemMapPhysicalPages(base, 4, std::ptr::null()), base);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    #[test]
    fn arrange_fresh_run_coalesces_scattered_fresh_pages_and_preserves_committed_pages() {
        let pages = allocate(8);
        let base = XMemVirtualAlloc(
            std::ptr::null_mut(),
            8 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!base.is_null());
        let at = |k: usize| base.wrapping_byte_add(k * PAGE);

        // Map and write pages 0 and 1 so slices 0..2 are committed (not fresh).
        assert_eq!(XMemMapPhysicalPages(at(0), 2, pages[..2].as_ptr()), at(0));
        // SAFETY: pages 0..2 are mapped read-write.
        unsafe {
            at(0).cast::<u64>().write(0x1111_2222);
            at(1).cast::<u64>().write(0x3333_4444);
        }

        // Duplicate page numbers must be rejected without mutating slice indices.
        assert_eq!(arrange_fresh_run(&[pages[7], pages[7], pages[3]]), 0);
        // Any committed page in the batch must prevent swapping.
        assert_eq!(arrange_fresh_run(&[pages[0], pages[7], pages[3]]), 0);

        // Four out-of-order fresh pages from the same section coalesce into one contiguous run.
        let scrambled = [pages[7], pages[3], pages[5], pages[2]];
        assert!(arrange_fresh_run(&scrambled) > 0);
        let indices: Vec<usize> = {
            let table = page_table();
            scrambled.iter().map(|p| table.pages[p].index).collect()
        };
        assert!(indices[0] >= 2, "must not overwrite committed slices 0..2: {indices:?}");
        assert_eq!(
            indices,
            vec![indices[0], indices[0] + 1, indices[0] + 2, indices[0] + 3],
            "fresh pages must be arranged into a contiguous ascending run"
        );
        // All 8 pages still map to 8 distinct slice indices (exact bijection).
        let all_indices: std::collections::HashSet<usize> = {
            let table = page_table();
            pages.iter().map(|p| table.pages[p].index).collect()
        };
        assert_eq!(all_indices.len(), 8);

        // Mapping the arranged pages creates ONE single 4-page view instead of 4 scattered views.
        assert_eq!(XMemMapPhysicalPages(at(2), 4, scrambled.as_ptr()), at(2));
        let view_len = physical().views.get(&(at(2) as usize)).map(|v| v.0);
        assert_eq!(view_len, Some(4 * PAGE));

        // Committed pages 0 and 1 retain their exact bytes.
        // SAFETY: pages 0..2 are still mapped read-write.
        unsafe {
            assert_eq!(at(0).cast::<u64>().read(), 0x1111_2222);
            assert_eq!(at(1).cast::<u64>().read(), 0x3333_4444);
        }

        assert_eq!(XMemMapPhysicalPages(base, 8, std::ptr::null()), base);
        // SAFETY: unmapped reservation owned by this test.
        unsafe { VirtualFree(base, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(8, pages.as_ptr()), 1);
    }

    #[test]
    fn find_fresh_run_word_skips_across_word_boundaries_and_partial_tail() {
        let section = allocate_section(135, false).expect("section");
        // Mark word 0 completely full, word 1 full except bits 61..64, word 2 bit 0..3 fresh then bit 3 full.
        section.committed[0].store(u64::MAX, Ordering::Release);
        section.committed[1].store(!(0b111u64 << 61), Ordering::Release);
        section.committed[2].store(1u64 << 3, Ordering::Release);

        // Run of 6 spans word 1 (61..64) into word 2 (128..131).
        assert_eq!(find_fresh_run(&section, 6), Some(125));
        // Cursor advanced past full word 0, stopped before partially-free word 1.
        assert_eq!(section.fresh_word.load(Ordering::Relaxed), 1);
        // Run of 7 does not fit in 125..131 (len 6) or tail 132..135 (len 3).
        assert_eq!(find_fresh_run(&section, 7), None);
        // Run of 3 fits in 125..128 first, and if 125..131 are filled, fits at 132..135 (partial tail).
        section.committed[1].store(u64::MAX, Ordering::Release);
        section.committed[2].store(0b1111u64, Ordering::Release);
        assert_eq!(find_fresh_run(&section, 3), Some(132));
        assert_eq!(section.fresh_word.load(Ordering::Relaxed), 2);
        // Partial tail only has 3 valid bits (132, 133, 134); requesting 4 must not read past section.pages.
        assert_eq!(find_fresh_run(&section, 4), None);
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                // (Once at least: the mapping thread may be done before this
                // one's first turn.)
                while reads == 0 || !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A title thread that reads and writes a page of a view while other
    /// pages of the view are unmapped and mapped again, as the console lets
    /// it: the page was never given up.
    #[test]
    fn a_title_thread_goes_on_using_the_rest_of_a_view() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let pages = allocate(4);
        for _ in 0..200 {
            // A new reservation each time: the pieces of a view cut up stay
            // cut up.
            let a = XMemVirtualAlloc(
                std::ptr::null_mut(),
                4 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!a.is_null());
            assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
            let (first, stop) = (a as usize, AtomicBool::new(false));
            // SAFETY: mapped read-write.
            unsafe { (first as *mut u64).write_volatile(0) };
            std::thread::scope(|s| {
                let title = s.spawn(|| {
                    let mut n = 0u64;
                    // SAFETY (the block): the first page is the title's all
                    // the time; a fault would end the process.
                    unsafe {
                        while !stop.load(Ordering::Relaxed) {
                            assert_eq!((first as *const u64).read_volatile(), n);
                            n += 1;
                            (first as *mut u64).write_volatile(n);
                        }
                    }
                    n
                });
                // SAFETY: as above.
                while !title.is_finished() && unsafe { (first as *const u64).read_volatile() } < 100 {}
                for page in [3, 2, 1] {
                    let at = a.wrapping_byte_add(page * PAGE);
                    assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
                    assert_eq!(XMemMapPhysicalPages(at, 1, pages[page..].as_ptr()), at);
                }
                stop.store(true, Ordering::Relaxed);
                assert!(title.join().expect("the title's thread never faulted") > 0);
            });
            assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        }
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// Many pages in one call: mapped, moved over each other's addresses,
    /// given twice in one list, copied by the GPU's mappings and unmapped.
    #[test]
    fn a_scatter_of_many_pages_is_what_one_after_the_other_is() {
        const N: usize = 200;
        let pages = allocate(N);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 2 * N * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE) as usize;
        let at = |k: usize| (a + k * PAGE) as *mut c_void;
        let first = |k: usize| at(k).cast::<u64>();
        for round in 0..4_usize {
            // Every address shows another page each round.
            let to: Vec<*mut c_void> = (0..N).map(|i| at((i + 7 * round) % N)).collect();
            assert_eq!(XMemMapPhysicalPagesScatter(to.as_ptr(), N, pages.as_ptr()), 1);
            for i in 0..N {
                // SAFETY: mapped read-write.
                unsafe {
                    if round > 0 {
                        assert_eq!(first((i + 7 * round) % N).read(), (1000 * (round - 1) + i) as u64);
                    }
                    first((i + 7 * round) % N).write((1000 * round + i) as u64);
                }
            }
        }
        // The GPU's copies of the mappings, in ranges and one by one; a page
        // mapped from one the same call has mapped shows the same.
        let half = (N / 2 * PAGE) as u64;
        let (from, to) = (a as u64, at(N) as u64);
        assert!(map_aliases(&[(to, from, half), (to + half, to, half)]));
        for k in 0..N / 2 {
            // SAFETY: mapped read-write.
            unsafe {
                assert_eq!(first(N + k).read(), first(k).read());
                assert_eq!(first(N + N / 2 + k).read(), first(k).read());
                first(N + k).write(5);
                assert_eq!(first(k).read(), 5);
            }
        }
        // An address twice: the last one counts; page 0 unmaps.
        let to = [at(0), at(1), at(0), at(2), at(3), at(0)];
        let what = [pages[5], pages[6], 0, 0, pages[7], pages[8]];
        assert_eq!(XMemMapPhysicalPagesScatter(to.as_ptr(), 6, what.as_ptr()), 1);
        assert_eq!(query(at(2)).host.State, MEM_RESERVE);
        // SAFETY: mapped read-write; the same page at both.
        unsafe {
            first(0).write(77);
            assert_eq!(XMemMapPhysicalPagesScatter([at(2), at(4), at(6)].as_ptr(), 3, [pages[8]; 3].as_ptr()), 1);
            assert_eq!((first(2).read(), first(4).read(), first(6).read()), (77, 77, 77));
        }
        // A page the title does not own.
        assert_eq!(XMemMapPhysicalPagesScatter([at(0), at(1)].as_ptr(), 2, [pages[0], 1].as_ptr()), 0);
        // All gone, also parts of a larger view.
        assert_eq!(XMemMapPhysicalPages(at(0), 4, std::ptr::null()), at(0));
        // SAFETY: placeholders of ours, nothing mapped in them.
        unsafe { VirtualFree(at(0), 4 * PAGE, MEM_RELEASE | 1 /* MEM_COALESCE_PLACEHOLDERS */) };
        assert_eq!(XMemMapPhysicalPages(at(0), 4, pages.as_ptr()), at(0));
        assert!(map_aliases(&[(a as u64, 0, (2 * N * PAGE) as u64)]));
        assert!((0..2 * N).all(|k| query(at(k)).host.State == MEM_RESERVE));
        assert!(physical().views.range(a..a + 2 * N * PAGE).next().is_none());
        assert_eq!(XMemFreePhysicalPages(N, pages.as_ptr()), 1);
    }

    /// Pages keep what they hold however often they are mapped, moved and
    /// unmapped without a commit or a split of their own.
    #[test]
    fn pages_mapped_again_hold_what_they_held() {
        let pages = allocate(3);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 8 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        assert!(!a.is_null());
        let at = |k: usize| a.wrapping_byte_add(k * PAGE);
        for round in 0..6_usize {
            for (k, page) in pages.iter().enumerate() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, page), to);
                let first = to.cast::<u64>();
                // SAFETY: mapped read-write; the far end of the page too.
                unsafe {
                    if round > 0 {
                        assert_eq!(first.read(), (round * 16 + k - 16) as u64);
                        assert_eq!(first.add(PAGE / 8 - 1).read(), k as u64);
                    }
                    first.write((round * 16 + k) as u64);
                    first.add(PAGE / 8 - 1).write(k as u64);
                }
            }
            for k in 0..pages.len() {
                let to = at((round + 2 * k) % 8);
                assert_eq!(XMemMapPhysicalPages(to, 1, std::ptr::null()), to);
                assert_eq!(query(to).host.State, MEM_RESERVE);
            }
            // The title joins what it unmapped: the next view splits it anew.
            if round == 3 {
                // SAFETY: placeholders of ours, nothing mapped in them.
                unsafe {
                    VirtualFree(a, 8 * PAGE, MEM_RELEASE | 1 /* MEM_COALESCE_PLACEHOLDERS */)
                };
            }
        }
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// What a title that maps and unmaps pages all the time costs the
    /// workers that write others: 24 threads look pages up, fill one and
    /// tell of it, as a request does, while one maps.
    /// `cargo test --release -p victory-hle --lib map_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "a benchmark"]
    fn map_bench() {
        use std::sync::atomic::AtomicU64;
        use std::time::{Duration, Instant};
        let pages = allocate(1024);
        let a = XMemVirtualAlloc(std::ptr::null_mut(), 1024 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE) as usize;
        for mapping in [false, true] {
            let (done, waited, worst, maps) =
                (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
            let until = Instant::now() + Duration::from_secs(3);
            std::thread::scope(|s| {
                if mapping {
                    s.spawn(|| {
                        let at: Vec<*mut c_void> = (0..256).map(|i| (a + i * PAGE) as *mut c_void).collect();
                        while Instant::now() < until {
                            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 256, pages.as_ptr()), 1);
                            assert_eq!(XMemMapPhysicalPagesScatter(at.as_ptr(), 256, std::ptr::null()), 1);
                            maps.fetch_add(512, Ordering::Relaxed);
                        }
                    });
                }
                for t in 0..24 {
                    let (pages, done, waited, worst) = (&pages, &done, &waited, &worst);
                    s.spawn(move || {
                        let mut i = 256 + t * 16;
                        while Instant::now() < until {
                            let page = &pages[i..=i];
                            i = if i == 256 + t * 16 + 15 { 256 + t * 16 } else { i + 1 };
                            let began = Instant::now();
                            assert!(pages_known(page, 0, PAGE));
                            let mut window = page_window(page, 0, PAGE).unwrap();
                            let looked = began.elapsed();
                            window.bytes().fill(t as u8);
                            let began = Instant::now();
                            pages_written(page, 0, PAGE);
                            let took = (looked + began.elapsed()).as_nanos() as u64;
                            waited.fetch_add(took, Ordering::Relaxed);
                            worst.fetch_max(took, Ordering::Relaxed);
                            done.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
            let n = done.load(Ordering::Relaxed).max(1);
            println!(
                "mapping {mapping}: {:.0} requests/s, {:.1} us of each on lookups and notices (worst {:.1} ms); {:.0} maps/s",
                n as f64 / 3.0,
                waited.load(Ordering::Relaxed) as f64 / n as f64 / 1e3,
                worst.load(Ordering::Relaxed) as f64 / 1e6,
                maps.load(Ordering::Relaxed) as f64 / 3.0
            );
        }
        assert_eq!(XMemFreePhysicalPages(1024, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = page_table().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_is_busy_from_a_quarter_of_the_time_to_a_twentieth() {
        let ms = std::time::Duration::from_millis;
        assert!(!busy_after(false, ms(60), ms(250)) && busy_after(false, ms(70), ms(250)));
        assert!(busy_after(true, ms(15), ms(250)) && !busy_after(true, ms(12), ms(250)));
        // A window without a call, then one short one.
        assert!(!busy_after(true, ms(5), ms(4000)));
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn map_alias_and_aliases_coalesce_contiguous_runs() {
        let (a, b) = (allocate(8), allocate(2));
        let reserve = |n: usize| {
            let p = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!p.is_null());
            p
        };
        let (src, dst) = (reserve(10), reserve(10));
        let mut src_pages = Vec::with_capacity(10);
        src_pages.extend_from_slice(&a[..4]);
        src_pages.extend_from_slice(&b[..2]);
        src_pages.extend_from_slice(&a[4..8]);
        assert_eq!(XMemMapPhysicalPages(src, 10, src_pages.as_ptr()), src);
        for i in 0..10usize {
            // SAFETY: all 10 pages of `src` are mapped read-write.
            unsafe { src.wrapping_byte_add(i * PAGE).cast::<u64>().write(0x100 + i as u64) };
        }

        let view_sizes = |base: usize, count: usize| -> Vec<usize> {
            physical()
                .views
                .range(base..base + count * PAGE)
                .map(|(_, (len, _))| *len)
                .collect()
        };

        // `map_alias` coalesces into 3 runs matching the 3 contiguous section runs.
        assert!(map_alias(dst as u64, src as u64, (10 * PAGE) as u64));
        assert_eq!(view_sizes(dst as usize, 10), vec![4 * PAGE, 2 * PAGE, 4 * PAGE]);
        for i in 0..10usize {
            // SAFETY: aliased read-write pages.
            assert_eq!(
                unsafe { dst.wrapping_byte_add(i * PAGE).cast::<u64>().read() },
                0x100 + i as u64
            );
        }

        // Unmap through `map_aliases`, then re-alias in pieces that still coalesce.
        assert!(map_aliases(&[(dst as u64, 0, (10 * PAGE) as u64)]));
        assert!(view_sizes(dst as usize, 10).is_empty());
        assert!(map_aliases(&[
            (dst as u64, src as u64, (2 * PAGE) as u64),
            (
                dst.wrapping_byte_add(2 * PAGE) as u64,
                src.wrapping_byte_add(2 * PAGE) as u64,
                (8 * PAGE) as u64,
            ),
        ]));
        assert_eq!(view_sizes(dst as usize, 10), vec![4 * PAGE, 2 * PAGE, 4 * PAGE]);
        for i in 0..10usize {
            // SAFETY: aliased read-write pages.
            assert_eq!(
                unsafe { dst.wrapping_byte_add(i * PAGE).cast::<u64>().read() },
                0x100 + i as u64
            );
        }

        assert!(map_aliases(&[(dst as u64, 0, (10 * PAGE) as u64)]));
        assert_eq!(XMemMapPhysicalPages(src, 10, std::ptr::null()), src);
        // SAFETY: unmapped reservations we own.
        unsafe {
            VirtualFree(src, 0, MEM_RELEASE);
            VirtualFree(dst, 0, MEM_RELEASE);
        }
        assert_eq!(XMemFreePhysicalPages(8, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(2, b.as_ptr()), 1);
    }

    #[test]
    fn query_mapped_stretch_and_generation_track_remaps() {
        let (a, b) = (allocate(4), allocate(2));
        let base_ptr = XMemVirtualAlloc(
            std::ptr::null_mut(),
            6 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!base_ptr.is_null());
        let base = base_ptr as usize;
        let g0 = mapping_generation();
        assert!(query_mapped_stretch(base, PAGE).is_none());

        // Map `a` as two adjacent 2-page views; `query_mapped_stretch` merges them.
        assert_eq!(XMemMapPhysicalPages(base_ptr, 2, a.as_ptr()), base_ptr);
        let mid = base_ptr.wrapping_byte_add(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(mid, 2, a[2..].as_ptr()), mid);
        let tail = base_ptr.wrapping_byte_add(4 * PAGE);
        assert_eq!(XMemMapPhysicalPages(tail, 2, b.as_ptr()), tail);

        let g1 = mapping_generation();
        assert!(g1 > g0);

        let stretch_a = query_mapped_stretch(base + PAGE, 2 * PAGE).expect("merged stretch");
        assert_eq!(stretch_a.base, base);
        assert_eq!(stretch_a.len, 4 * PAGE);
        assert_eq!(stretch_a.section_offset, 0);
        assert!(stretch_a.generation >= g1);

        // Crossing into another section returns None.
        assert!(query_mapped_stretch(base + 3 * PAGE, 2 * PAGE).is_none());
        let stretch_b = query_mapped_stretch(base + 4 * PAGE, 2 * PAGE).expect("second section");
        assert_eq!(stretch_b.base, base + 4 * PAGE);
        assert_eq!(stretch_b.len, 2 * PAGE);
        assert_ne!(stretch_b.section, stretch_a.section);

        // Unmapping a middle page splits the stretch and bumps the generation.
        let hole = base_ptr.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(hole, 1, std::ptr::null()), hole);
        let g2 = mapping_generation();
        assert!(g2 > g1);
        assert!(query_mapped_stretch(base + PAGE, PAGE).is_none());
        assert!(query_mapped_stretch(base, 2 * PAGE).is_none());

        let left = query_mapped_stretch(base, PAGE).expect("left piece");
        assert_eq!((left.base, left.len, left.section_offset), (base, PAGE, 0));
        let right = query_mapped_stretch(base + 2 * PAGE, 2 * PAGE).expect("right piece");
        assert_eq!(
            (right.base, right.len, right.section_offset),
            (base + 2 * PAGE, 2 * PAGE, (2 * PAGE) as u64)
        );
        assert!(right.generation >= g2);

        assert_eq!(XMemMapPhysicalPages(base_ptr, 6, std::ptr::null()), base_ptr);
        // SAFETY: unmapped reservation we own.
        unsafe { VirtualFree(base_ptr, 0, MEM_RELEASE) };
        assert_eq!(XMemFreePhysicalPages(4, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(2, b.as_ptr()), 1);
    }

    #[test]
    fn query_private_region_tracks_alloc_commit_and_free() {
        let ptr = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!ptr.is_null());
        let base = ptr as usize;

        let r0 = query_private_region(base, 4 * PAGE).expect("reserved region");
        assert_eq!(r0.base, base);
        assert_eq!(r0.len, 4 * PAGE);
        assert!(!r0.committed);
        assert!(r0.write_watched);

        // Commit the middle 2 pages.
        let mid_ptr = ptr.wrapping_byte_add(PAGE);
        assert_eq!(
            XMemVirtualAlloc(mid_ptr, 2 * PAGE, MEM_COMMIT, GRAPHICS, PAGE_READWRITE),
            mid_ptr
        );
        let r_mid = query_private_region(base + PAGE, 2 * PAGE).expect("committed subrange");
        assert_eq!((r_mid.base, r_mid.len), (base + PAGE, 2 * PAGE));
        assert!(r_mid.committed);
        assert!(r_mid.write_watched);
        assert!(r_mid.generation > r0.generation);

        let r_lo = query_private_region(base, PAGE).expect("uncommitted prefix");
        assert!(!r_lo.committed);
        let r_hi = query_private_region(base + 3 * PAGE, PAGE).expect("uncommitted suffix");
        assert!(!r_hi.committed);

        // Committing both ends coalesces the reservation back into one committed region.
        assert_eq!(XMemVirtualAlloc(ptr, PAGE, MEM_COMMIT, GRAPHICS, PAGE_READWRITE), ptr);
        let end_ptr = ptr.wrapping_byte_add(3 * PAGE);
        assert_eq!(
            XMemVirtualAlloc(end_ptr, PAGE, MEM_COMMIT, GRAPHICS, PAGE_READWRITE),
            end_ptr
        );
        let r_all = query_private_region(base, 4 * PAGE).expect("coalesced committed region");
        assert_eq!((r_all.base, r_all.len, r_all.committed), (base, 4 * PAGE, true));

        // Decommit a sub-range through `title_virtual_free`.
        assert_ne!(title_virtual_free(mid_ptr, PAGE, MEM_DECOMMIT), 0);
        let r_dec = query_private_region(base + PAGE, PAGE).expect("decommitted subrange");
        assert!(!r_dec.committed);
        assert!(r_dec.generation > r_all.generation);
        let r_rem = query_private_region(base + 2 * PAGE, 2 * PAGE).expect("still committed");
        assert!(r_rem.committed);

        // Release the whole reservation.
        assert_ne!(title_virtual_free(ptr, 0, MEM_RELEASE), 0);
        assert!(query_private_region(base, PAGE).is_none());
        assert!(query_private_region(base + 2 * PAGE, PAGE).is_none());
        assert!(mapping_generation() > r_dec.generation);
    }

    #[test]
    fn shown_long_views_indexed_by_section() {
        let sec_a = allocate_section(SHORT + 4, false).expect("section A");
        let sec_b = allocate_section(SHORT + 4, false).expect("section B");
        let mut shown = Shown::default();
        let long_len = (SHORT + 2) * PAGE;

        let slice_a = Slice { section: sec_a.clone(), index: 1 };
        let slice_b = Slice { section: sec_b.clone(), index: 0 };
        shown.note(0x1000_0000, long_len, &slice_a);
        shown.note(0x2000_0000, long_len, &slice_b);
        assert_eq!(shown.long.len(), 2);

        assert_eq!(
            shown.at(&Slice { section: sec_a.clone(), index: 1 }),
            vec![(0x1000_0000, PAGE)]
        );
        assert_eq!(
            shown.at(&Slice { section: sec_a.clone(), index: SHORT + 2 }),
            vec![(0x1000_0000 + (SHORT + 1) * PAGE, PAGE)]
        );
        assert!(shown.at(&Slice { section: sec_a.clone(), index: 0 }).is_empty());
        assert_eq!(
            shown.at(&Slice { section: sec_b.clone(), index: 0 }),
            vec![(0x2000_0000, PAGE)]
        );

        shown.forget(0x1000_0000, long_len, &slice_a);
        assert_eq!(shown.long.len(), 1);
        assert!(shown.at(&Slice { section: sec_a.clone(), index: 1 }).is_empty());
        assert_eq!(
            shown.at(&Slice { section: sec_b.clone(), index: 0 }),
            vec![(0x2000_0000, PAGE)]
        );

        shown.forget(0x2000_0000, long_len, &slice_b);
        assert!(shown.long.is_empty());
    }

    #[test]
    fn committed_section_helper_creates_sec_commit_section() {
        let committed = allocate_section(2, true).expect("SEC_COMMIT section");
        assert!(committed.sec_committed);
        let guard = committed.window(0, 2).expect("window on committed section");
        let at = guard.addr();
        // SAFETY: 2 committed 64 KiB pages mapped writable through `committed`'s window.
        unsafe {
            (at as *mut u64).write(0xdead_beef_cafe_babe);
            assert_eq!((at as *const u64).read(), 0xdead_beef_cafe_babe);
            ((at + PAGE) as *mut u64).write(0x1234_5678);
            assert_eq!(((at + PAGE) as *const u64).read(), 0x1234_5678);
        }
        assert!(committed.commit_shown(0, at, 2 * PAGE));
        drop(guard);

        let reserved = allocate_section(2, false).expect("SEC_RESERVE section");
        assert!(!reserved.sec_committed);
        let rguard = reserved.window(0, 2).expect("window on reserved section");
        let rat = rguard.addr();
        // SAFETY: `window` commits reserved pages before returning.
        unsafe {
            (rat as *mut u64).write(42);
            assert_eq!((rat as *const u64).read(), 42);
        }
        drop(rguard);
    }

    #[test]
    fn large_reserve_pool_chunks_into_independent_subsections_and_evicts_windows() {
        // Allocate 40 MiB (640 pages > 32 MiB reserve chunk): must split into >= 2 sub-sections.
        let n = 40 * MIB / PAGE;
        let pages = allocate(n);
        let (first_sec, last_sec) = {
            let table = page_table();
            (
                table.pages[&pages[0]].section.clone(),
                table.pages[&pages[n - 1]].section.clone(),
            )
        };
        assert!(
            !Arc::ptr_eq(&first_sec, &last_sec),
            "40 MiB SEC_RESERVE pool must split into multiple sub-sections"
        );
        assert!(!first_sec.sec_committed && !last_sec.sec_committed);
        let first_id = first_sec.id;

        // Write through bounded sliding window across a chunk boundary (> 16 MiB) and verify.
        let boundary = WINDOW_CHUNK_PAGES * PAGE - 16;
        let payload = [0x3cu8; 64];
        assert!(write_physical(&pages, boundary, &payload));
        let mut w = page_window(&pages, 0, PAGE).expect("first page window");
        w.bytes()[..8].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());

        // Evicting the section from cache while `w` is live must NOT unmap `w`'s chunk.
        window_cache_evict_section(first_id);
        assert_eq!(w.bytes()[..8], 0x1122_3344_5566_7788u64.to_le_bytes());
        drop(w);

        // Freeing the first sub-section's pages drops it cleanly while the tail stays alive.
        let first_pages = first_sec.pages;
        drop(first_sec);
        assert_eq!(XMemFreePhysicalPages(first_pages, pages[..first_pages].as_ptr()), 1);
        assert!(page_table().pages.get(&pages[0]).is_none());
        assert!(page_table().pages.get(&pages[n - 1]).is_some());
        drop(last_sec);
        assert_eq!(
            XMemFreePhysicalPages(n - first_pages, pages[first_pages..].as_ptr()),
            1
        );
    }

    #[test]
    fn virtual_alloc_freelist_recycles_with_zero_fill_and_write_watch() {
        let p1 = XMemVirtualAlloc(
            std::ptr::null_mut(),
            128 * 1024,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert!(!p1.is_null());
        // SAFETY: 128 KiB committed RW block.
        unsafe {
            p1.cast::<u8>().write_bytes(0xEE, 128 * 1024);
        }
        assert_ne!(title_virtual_free(p1, 0, MEM_RELEASE), 0);
        assert!(query_private_region(p1 as usize, 128 * 1024).is_none());

        let p2 = XMemVirtualAlloc(
            std::ptr::null_mut(),
            128 * 1024,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert_eq!(p2, p1, "matching size class should recycle block from freelist");
        // SAFETY: reused committed RW block must be zero-filled.
        let bytes = unsafe { std::slice::from_raw_parts(p2.cast::<u8>(), 128 * 1024) };
        assert!(bytes.iter().all(|&b| b == 0), "recycled block must be zeroed");

        // Must preserve MEM_WRITE_WATCH and start clean (0 dirty pages after ResetWriteWatch).
        let (mut found, mut count, mut gran) = ([0usize; 4], 4, 0);
        // SAFETY: valid write-watched allocation.
        let rc = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p2,
                128 * 1024,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut gran,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(count, 0, "ResetWriteWatch clears zero-fill dirty bits");
        assert_ne!(title_virtual_free(p2, 0, MEM_RELEASE), 0);
    }

    #[test]
    fn working_set_stats_track_title_allocations_and_free() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);

        let pages = allocate(8);
        let mut with_pages = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut with_pages), 1);
        assert!(with_pages.physical_pages >= 8);
        assert!(with_pages.title_used >= 8 * PAGE);

        assert_eq!(XMemFreePhysicalPages(8, pages.as_ptr()), 1);
    }

    #[test]
    fn clear_holes_removes_overlapping_and_interior_holes() {
        let mut phys = Physical::default();
        phys.holes.insert(0x1000 * PAGE, PAGE);
        phys.holes.insert(0x1001 * PAGE, PAGE);
        phys.holes.insert(0x1002 * PAGE, PAGE);
        phys.holes.insert(0x1004 * PAGE, 3 * PAGE);
        phys.holes.insert(0x1008 * PAGE, PAGE);

        // Coalesced range covering 0x1000..0x1003 and overhanging into 0x1004..0x1007.
        phys.clear_holes(0x1000 * PAGE, 0x1005 * PAGE);
        assert!(!phys.holes.contains_key(&(0x1000 * PAGE)));
        assert!(!phys.holes.contains_key(&(0x1001 * PAGE)));
        assert!(!phys.holes.contains_key(&(0x1002 * PAGE)));
        assert!(!phys.holes.contains_key(&(0x1004 * PAGE)));
        assert_eq!(phys.holes.get(&(0x1008 * PAGE)), Some(&PAGE));
    }
}
