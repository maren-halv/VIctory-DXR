//! Section-backed and partial-commit native in-place D3D12 heaps.
//!
//! Companion to [`crate::inplace`] and [`crate::port`]:
//! - Caches `ID3D12Device3::OpenExistingHeapFromFileMapping` heaps opened on
//!   committed `xmem` sections (`SEC_COMMIT`). Opening such a heap takes ~0.2 ms
//!   and locks 0 MB of working set, while leaving all of the title's own views
//!   free to unmap, remap or guard without error 87.
//! - Tracks a monotonic mapping generation counter bumped on every view remap,
//!   section teardown, or memory free, so in-place buffer placement never races
//!   against concurrent remapping.
//! - Validates contiguous committed private sub-ranges within a single
//!   `VirtualAlloc` reservation for `ID3D12Device13::OpenExistingHeapFromAddress1`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Memory as mem;

use crate::inplace::{From, Placed, committed_run, open, place};

const K64: u64 = 64 << 10;
const MIB: u64 = 1 << 20;
const DEFAULT_SECTION_HEAP_MAX_MB: u64 = 1024;

/// Maximum total bytes of live `OpenExistingHeapFromFileMapping` section heaps
/// cached at once (`VICTORY_INPLACE_SECTION_MAX_MB`, default `1024` MiB).
fn section_heap_max_bytes() -> u64 {
    static MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("VICTORY_INPLACE_SECTION_MAX_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_SECTION_HEAP_MAX_MB)
            .saturating_mul(MIB)
    })
}

/// Live bytes currently covered by cached `OpenExistingHeapFromFileMapping` heaps.
static LIVE_SECTION_BYTES: AtomicU64 = AtomicU64::new(0);

/// Monotonic generation counter bumped whenever title memory views, sections,
/// or allocations change.
static MAPPING_GEN: AtomicU64 = AtomicU64::new(1);

/// Current title-memory mapping generation.
pub(crate) fn mapping_generation() -> u64 {
    MAPPING_GEN.load(Ordering::Acquire)
}

/// Advances the mapping generation and returns the new value.
pub(crate) fn bump_generation() -> u64 {
    MAPPING_GEN.fetch_add(1, Ordering::AcqRel) + 1
}

struct SectionHeap {
    heap: ID3D12Heap,
    bytes: u64,
    #[allow(dead_code)]
    generation: u64,
}

// SAFETY: D3D12 heaps are free-threaded.
unsafe impl Send for SectionHeap {}

static SECTION_HEAPS: Mutex<Option<HashMap<usize, SectionHeap>>> = Mutex::new(None);

/// Section heap stats since startup:
/// `[0]` heaps opened, `[1]` total section bytes opened,
/// `[2]` buffers/windows placed, `[3]` total bytes placed,
/// `[4]` placements aborted by concurrent remap (generation mismatch),
/// `[5]` section heaps evicted on section teardown / handle reuse.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Snapshot of section-heap stats for periodic logging.
pub(crate) fn stats() -> [u64; 6] {
    std::array::from_fn(|i| STATS[i].load(Ordering::Relaxed))
}

/// Evicts any cached `ID3D12Heap` for `section` (called when a section is
/// closed or its handle number is reused) and retires it safely.
pub(crate) fn section_evict(section: usize) {
    bump_generation();
    let evicted = {
        let mut guard = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_mut().and_then(|map| map.remove(&section))
    };
    if let Some(entry) = evicted {
        release_budget(&LIVE_SECTION_BYTES, entry.bytes);
        STATS[5].fetch_add(1, Ordering::Relaxed);
        crate::host::retire(entry.heap);
    }
}

/// Why [`place_section`] declined to place a buffer in a committed section.
#[derive(Debug)]
pub(crate) enum SectionPlaceError {
    /// Range is not a single contiguous run of a `SEC_COMMIT` section.
    NotCommittedRun,
    /// A concurrent remap/unmap invalidated the mapping during placement.
    Remapped,
    /// Host D3D12 refused `OpenExistingHeapFromFileMapping` or `CreatePlacedResource`.
    Host(windows::core::Error),
}

fn open_or_get(device: &ID3D12Device, section: usize, section_bytes: u64, generation: u64) -> Result<ID3D12Heap, SectionPlaceError> {
    {
        let guard = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = guard.as_ref().and_then(|map| map.get(&section))
            && entry.bytes == section_bytes
        {
            return Ok(entry.heap.clone());
        }
    }
    // Check generation before asking the driver to open the handle.
    if mapping_generation() != generation {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return Err(SectionPlaceError::Remapped);
    }
    if !try_reserve_budget(&LIVE_SECTION_BYTES, section_bytes, section_heap_max_bytes()) {
        return Err(SectionPlaceError::NotCommittedRun);
    }
    let heap = match open(device, From::Section(section)) {
        Ok(heap) => heap,
        Err(e) => {
            release_budget(&LIVE_SECTION_BYTES, section_bytes);
            return Err(SectionPlaceError::Host(e));
        }
    };
    if mapping_generation() != generation {
        release_budget(&LIVE_SECTION_BYTES, section_bytes);
        STATS[4].fetch_add(1, Ordering::Relaxed);
        crate::host::retire(heap);
        return Err(SectionPlaceError::Remapped);
    }
    let mut guard = SECTION_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    // Re-check generation under `SECTION_HEAPS` lock to prevent inserting a stale
    // heap if `section_evict` ran between opening the handle and acquiring the lock.
    if mapping_generation() != generation {
        drop(guard);
        release_budget(&LIVE_SECTION_BYTES, section_bytes);
        STATS[4].fetch_add(1, Ordering::Relaxed);
        crate::host::retire(heap);
        return Err(SectionPlaceError::Remapped);
    }
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(existing) = map.get(&section)
        && existing.bytes == section_bytes
    {
        let existing_heap = existing.heap.clone();
        drop(guard);
        release_budget(&LIVE_SECTION_BYTES, section_bytes);
        crate::host::retire(heap);
        return Ok(existing_heap);
    }
    let replaced = map.insert(section, SectionHeap { heap: heap.clone(), bytes: section_bytes, generation });
    drop(guard);
    if let Some(old) = replaced {
        release_budget(&LIVE_SECTION_BYTES, old.bytes);
        STATS[5].fetch_add(1, Ordering::Relaxed);
        crate::host::retire(old.heap);
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(section_bytes, Ordering::Relaxed);
    tracing::info!(
        section = format_args!("{section:#x}"),
        bytes_mb = section_bytes >> 20,
        live_mb = LIVE_SECTION_BYTES.load(Ordering::Relaxed) >> 20,
        "in-place section heap opened via OpenExistingHeapFromFileMapping (0 MB locked working set)"
    );
    Ok(heap)
}

/// Places a D3D12 buffer over `[start, start + len)` directly in a cached
/// `OpenExistingHeapFromFileMapping` heap if the range is covered by one
/// contiguous run of a committed (`SEC_COMMIT`) section.
///
/// Validates the mapping generation counter before and after heap/resource
/// creation so concurrent unmaps or remaps abort safely.
pub(crate) fn place_section(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, SectionPlaceError> {
    if len == 0 {
        return Err(SectionPlaceError::NotCommittedRun);
    }
    let at = start & !(K64 - 1);
    let Some(end) = start.checked_add(len).and_then(|e| e.checked_next_multiple_of(K64)) else {
        return Err(SectionPlaceError::NotCommittedRun);
    };
    let bytes = end - at;
    let generation = mapping_generation();
    let Some((section, sec_offset, sec_bytes)) = committed_run(at as usize, bytes as usize) else {
        return Err(SectionPlaceError::NotCommittedRun);
    };
    if !sec_offset.is_multiple_of(K64) || sec_offset.checked_add(bytes).is_none_or(|e| e > sec_bytes) {
        return Err(SectionPlaceError::NotCommittedRun);
    }
    let heap = open_or_get(device, section, sec_bytes, generation)?;
    if mapping_generation() != generation {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return Err(SectionPlaceError::Remapped);
    }
    let buffer = place(device, &heap, sec_offset, bytes).map_err(SectionPlaceError::Host)?;
    if mapping_generation() != generation {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return Err(SectionPlaceError::Remapped);
    }
    // SAFETY: a live buffer just placed in `heap`.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    STATS[2].fetch_add(1, Ordering::Relaxed);
    STATS[3].fetch_add(len, Ordering::Relaxed);
    Ok(Placed { buffer, heap, view: None, gpu, bias: start - at, pinned: 0 })
}

/// High-level entry point for [`crate::inplace::place_for`] and [`crate::port::place_for`]:
/// holds the non-blocking title-memory reading lock across section heap lookup
/// and placement so section handles cannot be closed concurrently.
pub(crate) fn place_section_for(device: &ID3D12Device, start: u64, len: u64) -> Result<Placed, SectionPlaceError> {
    let Some(_reading) = crate::watch::title_memory_reading_now() else {
        STATS[4].fetch_add(1, Ordering::Relaxed);
        return Err(SectionPlaceError::Remapped);
    };
    place_section(device, start, len)
}

/// Whether `[at, at + len)` is a contiguous committed, read-write, `MEM_PRIVATE`
/// sub-range belonging to ONE `VirtualAlloc` reservation (`AllocationBase`).
///
/// Unlike a single `VirtualQuery` check, this walks adjacent sub-regions within
/// the same `AllocationBase` (e.g. a reservation committed in multiple chunks)
/// while rejecting ranges that cross into reserved/uncommitted pages, guarded
/// pages, mapped views, or a second `VirtualAlloc` allocation.
pub(crate) fn private_committed_subrange(at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let Some(end) = at.checked_add(len) else {
        return false;
    };
    let mut cursor = at;
    let mut alloc_base: *mut c_void = std::ptr::null_mut();
    while cursor < end {
        // SAFETY: plain data filled in by `VirtualQuery`.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer of exact size.
        let n = unsafe { mem::VirtualQuery(cursor as *const c_void, &mut info, size_of_val(&info)) };
        if n == 0
            || info.State != mem::MEM_COMMIT
            || info.Type != mem::MEM_PRIVATE
            || info.Protect != mem::PAGE_READWRITE
        {
            return false;
        }
        if alloc_base.is_null() {
            alloc_base = info.AllocationBase;
        } else if info.AllocationBase != alloc_base {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= cursor {
            return false;
        }
        cursor = next;
    }
    true
}

/// Chooses the cached private region `(base, len)` covering 64 KiB-aligned
/// `[at, end)`. Expands to a `chunk_bytes`-aligned chunk when the entire chunk
/// is one committed `MEM_PRIVATE` `PAGE_READWRITE` sub-range of the same
/// `AllocationBase`; otherwise falls back to exact `[at, end)`.
pub(crate) fn private_chunk_bounds(at: usize, end: usize, chunk_bytes: usize) -> Option<(usize, usize)> {
    let len = end.checked_sub(at)?;
    if len == 0 || !private_committed_subrange(at, len) {
        return None;
    }
    if chunk_bytes.is_power_of_two() && chunk_bytes >= K64 as usize && len < chunk_bytes {
        let chunk_start = at & !(chunk_bytes - 1);
        if let Some(chunk_end) = end.checked_next_multiple_of(chunk_bytes)
            && chunk_end - chunk_start == chunk_bytes
            && private_committed_subrange(chunk_start, chunk_bytes)
        {
            return Some((chunk_start, chunk_bytes));
        }
    }
    Some((at, len))
}

/// Atomically reserves `bytes` from a working-set budget `limit` tracked in
/// `used`. Returns `true` if reserved (caller must call [`release_budget`] on
/// failure or teardown).
pub(crate) fn try_reserve_budget(used: &AtomicU64, bytes: u64, limit: u64) -> bool {
    let mut cur = used.load(Ordering::Relaxed);
    loop {
        let Some(next) = cur.checked_add(bytes) else {
            return false;
        };
        if next > limit {
            return false;
        }
        match used.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
}

/// Releases `bytes` previously reserved with [`try_reserve_budget`].
pub(crate) fn release_budget(used: &AtomicU64, bytes: u64) {
    let _ = used.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |cur| Some(cur.saturating_sub(bytes)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::core::Interface;

    const K: usize = 64 << 10;

    fn hardware_device() -> Option<(String, ID3D12Device)> {
        // SAFETY: plain DXGI enumeration and D3D12 device creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            for idx in 0u32..16 {
                let Ok(adapter) = factory.EnumAdapters1(idx) else { break };
                let Ok(desc) = adapter.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_ok()
                    && let Some(dev) = device
                    && dev.cast::<ID3D12Device13>().is_ok()
                {
                    let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                    let name = String::from_utf16_lossy(&desc.Description[..len]);
                    return Some((name, dev));
                }
            }
            None
        }
    }

    #[test]
    fn budget_reservation_is_atomic_and_never_exceeds_limit() {
        let used = AtomicU64::new(0);
        assert!(try_reserve_budget(&used, 600, 1000));
        assert!(try_reserve_budget(&used, 400, 1000));
        assert!(!try_reserve_budget(&used, 1, 1000));
        assert_eq!(used.load(Ordering::Acquire), 1000);
        release_budget(&used, 400);
        assert!(try_reserve_budget(&used, 300, 1000));
        assert_eq!(used.load(Ordering::Acquire), 900);
        release_budget(&used, 2000);
        assert_eq!(used.load(Ordering::Acquire), 0);
    }

    #[test]
    fn private_committed_subrange_handles_partial_and_multi_commit_regions() {
        // SAFETY: our own test allocation, released at end.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 16 * K, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert!(!private_committed_subrange(base as usize, K), "uncommitted reservation");

            // Commit two adjacent sub-ranges [4..7) and [7..10) separately.
            assert!(!mem::VirtualAlloc(base.add(4 * K).cast(), 3 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());
            assert!(!mem::VirtualAlloc(base.add(7 * K).cast(), 3 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());

            assert!(private_committed_subrange(base.add(4 * K) as usize, 6 * K), "adjacent commits in one AllocationBase");
            assert!(private_committed_subrange(base.add(5 * K) as usize, 3 * K), "interior sub-range");
            assert!(!private_committed_subrange(base.add(3 * K) as usize, 4 * K), "starts in uncommitted");
            assert!(!private_committed_subrange(base.add(8 * K) as usize, 4 * K), "ends in uncommitted");

            // Decommit page 6: splits committed range into [4..6) and [7..10).
            assert_ne!(mem::VirtualFree(base.add(6 * K).cast(), K, mem::MEM_DECOMMIT), 0);
            assert!(!private_committed_subrange(base.add(4 * K) as usize, 6 * K), "hole at page 6");
            assert!(private_committed_subrange(base.add(4 * K) as usize, 2 * K), "left committed part");
            assert!(private_committed_subrange(base.add(7 * K) as usize, 3 * K), "right committed part");

            // Guard page 7 as read-only: rejected.
            let mut old = 0;
            assert_ne!(mem::VirtualProtect(base.add(7 * K).cast(), K, mem::PAGE_READONLY, &mut old), 0);
            assert!(!private_committed_subrange(base.add(7 * K) as usize, 3 * K), "read-only rejected");
            assert!(private_committed_subrange(base.add(8 * K) as usize, 2 * K), "remaining rw accepted");

            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn private_chunk_bounds_respects_partial_commit_and_allocation_boundaries() {
        // SAFETY: test-owned VirtualAlloc allocation, released at end.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 16 * K, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(private_chunk_bounds(base as usize, base as usize + K, 4 * K), None);

            // Commit [4..12) (8 pages). With a 4-page chunk, an aligned 4-page chunk inside
            // [4..12) may expand to 4 pages, while a partial-commit edge falls back to exact.
            assert!(!mem::VirtualAlloc(base.add(4 * K).cast(), 8 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());

            let aligned_at = ((base as usize + 4 * K).next_multiple_of(4 * K)).min(base as usize + 8 * K);
            if aligned_at + 4 * K <= base as usize + 12 * K {
                assert_eq!(
                    private_chunk_bounds(aligned_at + K, aligned_at + 2 * K, 4 * K),
                    Some((aligned_at, 4 * K)),
                    "fully committed aligned chunk expands cleanly"
                );
            }
            // Decommit page 10: a request on [8..10) can no longer expand across page 10.
            assert_ne!(mem::VirtualFree(base.add(10 * K).cast(), K, mem::MEM_DECOMMIT), 0);
            let sub = private_chunk_bounds(base.add(8 * K) as usize, base.add(10 * K) as usize, 4 * K)
                .expect("committed subrange falls back to exact bounds");
            assert!(sub.0 >= base.add(4 * K) as usize && sub.0 + sub.1 <= base.add(10 * K) as usize);
            assert!(sub.0 <= base.add(8 * K) as usize && sub.0 + sub.1 >= base.add(10 * K) as usize);

            // Request overlapping the decommitted page 10 is rejected.
            assert_eq!(private_chunk_bounds(base.add(9 * K) as usize, base.add(11 * K) as usize, 4 * K), None);

            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    fn run_committed_section_coherency(device: &ID3D12Device) {
        // SAFETY: section, placeholder and views created and cleaned up by this test.
        unsafe {
            let process = windows_sys::Win32::System::Threading::GetCurrentProcess();
            let section = mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_COMMIT,
                0,
                (10 * K) as u32,
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
            mem::VirtualFree(hole.cast(), 3 * K, mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER);

            crate::inplace::title_memory_section(section as usize, (10 * K) as u64, true);

            // Map section pages 2..5 at hole+0 and pages 5..10 at hole+3*K (contiguous run!).
            let v1 = mem::MapViewOfFile3(section, process, hole.cast(), (2 * K) as u64, 3 * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            let v2 = mem::MapViewOfFile3(section, process, hole.add(3 * K).cast(), (5 * K) as u64, 5 * K, mem::MEM_REPLACE_PLACEHOLDER, mem::PAGE_READWRITE, std::ptr::null_mut(), 0);
            assert!(!v1.Value.is_null() && !v2.Value.is_null());
            crate::inplace::title_memory_view(v1.Value as usize, 3 * K, section as usize, (2 * K) as u64);
            crate::inplace::title_memory_view(v2.Value as usize, 5 * K, section as usize, (5 * K) as u64);

            let base = hole as u64;
            let start = base + K as u64 + 0x20;
            let len = 4 * K as u64;

            let before_opens = STATS[0].load(Ordering::Relaxed);
            let placed1 = place_section(device, start, len).expect("first committed section placement");
            assert_eq!(placed1.bias, 0x20);
            let after_first = STATS[0].load(Ordering::Relaxed);
            assert_eq!(after_first, before_opens + 1, "opened section heap once");

            // Second buffer in same section reuses the cached section heap!
            let placed2 = place_section(device, base + 5 * K as u64, K as u64).expect("second placement reuses heap");
            assert_eq!(STATS[0].load(Ordering::Relaxed), after_first, "cached section heap reused");

            // Verify CPU write -> GPU read AND GPU UAV shader write -> CPU read!
            let mut rig = crate::inplace::Rig::new(device).unwrap();
            let cpu_slice = std::slice::from_raw_parts_mut(start as *mut u8, len as usize);
            for (i, b) in cpu_slice.iter_mut().enumerate() {
                *b = (i.wrapping_mul(19).wrapping_add(11)) as u8;
            }
            let got = rig.fetch(&placed1.buffer, placed1.bias, len as usize).unwrap();
            assert_eq!(got.as_slice(), cpu_slice, "GPU reads CPU bytes across both views of SEC_COMMIT section");

            // GPU UAV shader writes word[2] = word[0] + word[1] directly into title RAM.
            let words = (base + 5 * K as u64) as *mut [u32; 3];
            words.write([1234, 5678, 0]);
            rig.shade(placed2.gpu).unwrap();
            assert_eq!(words.read()[2], 1234 + 5678, "CPU reads GPU UAV write directly in title view");

            // Title views remain 100% protectable and unmappable while the section heap is alive!
            let mut old = 0;
            assert_ne!(mem::VirtualProtect(v1.Value, K, mem::PAGE_READONLY, &mut old), 0, "view stays protectable");
            assert_ne!(mem::VirtualProtect(v1.Value, K, mem::PAGE_READWRITE, &mut old), 0);

            let gen_before = mapping_generation();
            crate::inplace::views_gone(base as usize, 8 * K);
            assert!(mapping_generation() > gen_before, "views_gone bumps generation");
            assert!(matches!(place_section(device, start, len), Err(SectionPlaceError::NotCommittedRun)), "unmapped range falls back cleanly");

            drop((placed1, placed2));
            assert_ne!(mem::UnmapViewOfFile2(process, v1, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            assert_ne!(mem::UnmapViewOfFile2(process, v2, mem::MEM_PRESERVE_PLACEHOLDER), 0);
            crate::inplace::title_memory_section_gone(section as usize);
            windows_sys::Win32::Foundation::CloseHandle(section);
            mem::VirtualFree(hole.cast(), 0, mem::MEM_RELEASE);
            mem::VirtualFree(hole.add(3 * K).cast(), 0, mem::MEM_RELEASE);
        }
    }

    fn run_partial_commit_private_coherency(device: &ID3D12Device) {
        // SAFETY: test-owned VirtualAlloc allocation, released at end.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 16 * K, mem::MEM_RESERVE, mem::PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert!(!mem::VirtualAlloc(base.add(4 * K).cast(), 4 * K, mem::MEM_COMMIT, mem::PAGE_READWRITE).is_null());

            let start = base.add(4 * K) as u64 + 0x100;
            let len = 2 * K as u64;
            let placed = crate::inplace::place_part(device, start, len).expect("partial commit subrange opens");
            let mut rig = crate::inplace::Rig::new(device).unwrap();

            // CPU -> GPU read
            let payload: Vec<u8> = (0..len as usize).map(|i| (i * 13 + 5) as u8).collect();
            std::ptr::copy_nonoverlapping(payload.as_ptr(), start as *mut u8, payload.len());
            let fetched = rig.fetch(&placed.buffer, placed.bias, payload.len()).unwrap();
            assert_eq!(fetched, payload, "GPU reads CPU writes from partial-commit private subrange");

            // GPU UAV shader write -> CPU read
            let words = base.add(5 * K) as *mut [u32; 3];
            words.write([0x1000, 0x2345, 0]);
            rig.shade(placed.gpu - 0x100 + K as u64).unwrap();
            assert_eq!(words.read()[2], 0x3345, "CPU reads GPU UAV write in partial-commit private subrange");

            drop(placed);
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }

    #[test]
    fn warp_committed_section_heap_and_partial_commit_private_coherency() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        run_committed_section_coherency(&device);
        run_partial_commit_private_coherency(&device);
    }

    #[test]
    fn hw_gpu_inplace_private_and_section_coherency() {
        let Some((name, device)) = hardware_device() else {
            eprintln!("no hardware ID3D12Device13 adapter available; skipping live GPU coherency test");
            return;
        };
        eprintln!("running live hardware GPU in-place coherency test on: {name}");
        run_committed_section_coherency(&device);
        run_partial_commit_private_coherency(&device);
        eprintln!("live hardware GPU in-place coherency PASSED on: {name}");
    }
}
