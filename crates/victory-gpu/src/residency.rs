//! Host-memory admission for speculative texture work, reusable upload storage,
//! and asynchronous D3D12 residency (`ID3D12Device3::EnqueueMakeResident`).
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FENCE_FLAG_NONE, D3D12_RESIDENCY_FLAG_NONE, ID3D12CommandQueue, ID3D12Device,
    ID3D12Device3, ID3D12Fence, ID3D12Pageable,
};
use windows::core::Interface;
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

thread_local! {
    static STAGING_DEFERRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn note_staging_deferred() {
    STAGING_DEFERRED.set(true);
}

pub(crate) fn take_staging_deferred() -> bool {
    STAGING_DEFERRED.replace(false)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// Whether the staging allocator may commit a brand-new OS upload buffer
/// (reusing existing free ranges in already-committed pool blocks stays open
/// whenever `admit_texture_pass` is true).
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_growth_reserve(sample_commit(now), bake)
}

pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB))
}

const AMPLE_COMMIT: u64 = 6 * GIB;

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    let commit_tight = if was { commit < 6 * GIB } else { commit <= 4 * GIB };
    if commit_tight {
        return true;
    }
    // On 32 GiB hosts with ample pagefile commit headroom (>= 6 GiB), Windows
    // Standby cache routinely holds `ullAvailPhys` around 2-4 GiB; only engage
    // physical pressure when free physical RAM drops dangerously low (<= 1.5 GiB)
    // and allow recovering once it eases back to >= 2.5 GiB.
    if commit >= AMPLE_COMMIT {
        if was {
            physical < 2560 * MIB
        } else {
            physical <= 1536 * MIB
        }
    } else if was {
        physical < 4 * GIB
    } else {
        physical <= 3 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(16 << 20)
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

/// Pure lock-free state machine tracking monotonically assigned
/// `EnqueueMakeResident` fence values and per-queue GPU `Wait` high-water marks.
pub(crate) struct ResidencyFenceState {
    next_value: AtomicU64,
    pending_value: AtomicU64,
    enqueued: AtomicU64,
    queue_waits: AtomicU64,
}

impl ResidencyFenceState {
    pub(crate) const fn new() -> Self {
        Self {
            next_value: AtomicU64::new(0),
            pending_value: AtomicU64::new(0),
            enqueued: AtomicU64::new(0),
            queue_waits: AtomicU64::new(0),
        }
    }

    /// Reserves the next monotonic fence value (callers on a shared D3D12 fence
    /// must serialize submission order around this call).
    pub(crate) fn reserve_next(&self) -> u64 {
        self.next_value.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Publishes `value` as pending after `EnqueueMakeResident` succeeds.
    pub(crate) fn record_enqueued(&self, value: u64) {
        self.pending_value.fetch_max(value, Ordering::Release);
        self.enqueued.fetch_add(1, Ordering::Relaxed);
    }

    /// Highest residency fence value enqueued so far.
    #[cfg(test)]
    pub(crate) fn pending(&self) -> u64 {
        self.pending_value.load(Ordering::Acquire)
    }

    /// Returns `Some(target)` when `queue_waited` lags behind `pending_value` and
    /// the queue must ensure GPU residency visibility before `ExecuteCommandLists`.
    pub(crate) fn wait_target(&self, queue_waited: &AtomicU64) -> Option<u64> {
        let target = self.pending_value.load(Ordering::Acquire);
        if target == 0 || target <= queue_waited.load(Ordering::Acquire) {
            return None;
        }
        Some(target)
    }

    /// Records that `queue_waited` has satisfied residency up to `target`.
    pub(crate) fn mark_waited(&self, queue_waited: &AtomicU64, target: u64, issued_wait: bool) {
        queue_waited.fetch_max(target, Ordering::Release);
        if issued_wait {
            self.queue_waits.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `(enqueued_count, gpu_queue_waits, pending_fence_value)`.
    pub(crate) fn stats(&self) -> (u64, u64, u64) {
        (
            self.enqueued.load(Ordering::Relaxed),
            self.queue_waits.load(Ordering::Relaxed),
            self.pending_value.load(Ordering::Acquire),
        )
    }
}

struct AsyncResidency {
    device3: ID3D12Device3,
    fence: ID3D12Fence,
    /// Serializes `reserve_next()` + `EnqueueMakeResident()` so WDDM receives
    /// strictly monotonic fence signal values without ever blocking on the GPU.
    enqueue_lock: Mutex<()>,
    state: ResidencyFenceState,
}

// SAFETY: `ID3D12Device3` and `ID3D12Fence` are free-threaded D3D12 objects.
unsafe impl Send for AsyncResidency {}
// SAFETY: as above.
unsafe impl Sync for AsyncResidency {}

impl AsyncResidency {
    fn try_new(device: &ID3D12Device) -> Option<Self> {
        let device3: ID3D12Device3 = device.cast().ok()?;
        // SAFETY: plain D3D12 fence creation on a live device.
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
        crate::host::name(&fence, || "victory residency fence".into());
        Some(Self {
            device3,
            fence,
            enqueue_lock: Mutex::new(()),
            state: ResidencyFenceState::new(),
        })
    }
}

fn parse_async_resident(val: Option<&str>) -> bool {
    val.is_none_or(|v| v.trim() != "0")
}

fn async_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_async_resident(std::env::var("VICTORY_ASYNC_RESIDENT").ok().as_deref()))
}

static PRIMARY_DEVICE: AtomicUsize = AtomicUsize::new(0);
static PRIMARY_ASYNC: OnceLock<Arc<AsyncResidency>> = OnceLock::new();
static EXTRA_DEVICES: Mutex<Vec<(usize, Option<Arc<AsyncResidency>>)>> = Mutex::new(Vec::new());

fn async_for_device(device: &ID3D12Device) -> Option<Arc<AsyncResidency>> {
    if !async_enabled() {
        return None;
    }
    let key = device.as_raw() as usize;
    if PRIMARY_DEVICE.load(Ordering::Acquire) == key {
        return PRIMARY_ASYNC.get().cloned();
    }
    let mut map = EXTRA_DEVICES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, existing)) = map.iter().find(|(k, _)| *k == key) {
        return existing.clone();
    }
    let made = AsyncResidency::try_new(device).map(Arc::new);
    if let Some(ref rt) = made
        && PRIMARY_ASYNC.set(Arc::clone(rt)).is_ok()
    {
        PRIMARY_DEVICE.store(key, Ordering::Release);
        return Some(Arc::clone(rt));
    }
    map.push((key, made.clone()));
    made
}

/// Whether `device` uses non-blocking `ID3D12Device3::EnqueueMakeResident` with
/// GPU-side queue fence waits instead of synchronous CPU `MakeResident`/`Aside`.
pub(crate) fn async_active(device: &ID3D12Device) -> bool {
    async_for_device(device).is_some()
}

/// Enqueues `object` for asynchronous residency via `ID3D12Device3::EnqueueMakeResident`
/// without blocking the calling CPU thread.
///
/// Returns `Ok(true)` when enqueued asynchronously (callers must attach
/// [`wait_on_queue`] before executing GPU work), or `Ok(false)` when
/// `ID3D12Device3` async residency is unavailable so the caller falls back to
/// synchronous `ID3D12Device::MakeResident`.
pub(crate) fn enqueue_resident(
    device: &ID3D12Device,
    object: &impl Interface,
) -> windows::core::Result<bool> {
    let Some(rt) = async_for_device(device) else {
        return Ok(false);
    };
    let pageable: ID3D12Pageable = object.cast()?;
    let _guard = rt.enqueue_lock.lock().unwrap_or_else(|e| e.into_inner());
    let val = rt.state.reserve_next();
    // SAFETY: `pageable` belongs to `device`; `enqueue_lock` guarantees strictly
    // monotonic fence values on `rt.fence` without waiting on paging completion.
    let res = unsafe {
        rt.device3
            .EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, &[Some(pageable)], &rt.fence, val)
    };
    match res {
        Ok(()) => {
            rt.state.record_enqueued(val);
            Ok(true)
        }
        Err(e) => {
            allocation_error(&e);
            Err(e)
        }
    }
}

/// Inserts a GPU-side `queue.Wait(&residency_fence, target)` if any
/// `EnqueueMakeResident` calls have not yet been observed by `queue`.
///
/// Never blocks the CPU thread; skips `queue.Wait` when the paging engine has
/// already completed past `target`.
pub(crate) fn wait_on_queue(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    queue_waited: &AtomicU64,
) -> bool {
    let Some(rt) = async_for_device(device) else {
        return false;
    };
    let Some(target) = rt.state.wait_target(queue_waited) else {
        return false;
    };
    // SAFETY: live fence owned by `rt`.
    let completed = unsafe { rt.fence.GetCompletedValue() };
    if completed != u64::MAX && completed >= target {
        rt.state.mark_waited(queue_waited, target, false);
        return false;
    }
    // SAFETY: live command queue and residency fence on `device`.
    match unsafe { queue.Wait(&rt.fence, target) } {
        Ok(()) => {
            rt.state.mark_waited(queue_waited, target, true);
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, target, "residency: queue.Wait(&residency_fence) failed");
            false
        }
    }
}

/// Returns `(enqueued_count, gpu_queue_waits, pending_fence_value)` for `device`.
pub(crate) fn async_stats(device: &ID3D12Device) -> (u64, u64, u64) {
    async_for_device(device).map_or((0, 0, 0), |rt| rt.state.stats())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, false), growth);
        }
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        // Low commit always enters pressure and requires >= 6 GiB to clear.
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(true, 12 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 4 * GIB, 6 * GIB));

        // Moderate commit (4 GiB < commit < 6 GiB): enters at <= 3 GiB physical, clears at >= 4 GiB.
        assert!(next_pressure(false, 3 * GIB, 5 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
    }

    #[test]
    fn ample_commit_avoids_latching_and_recovers_on_physical_ease() {
        // With >= 6 GiB commit headroom, Standby cache holding 2-3.5 GiB free physical RAM does NOT latch.
        assert!(!next_pressure(false, 2700 * MIB, 6 * GIB));
        assert!(!next_pressure(false, 3 * GIB, 8 * GIB));
        assert!(!next_pressure(false, 2 * GIB, 16 * GIB));
        // Critically low physical RAM (<= 1.5 GiB) still engages pressure with 1 GiB hysteresis.
        assert!(next_pressure(false, 1536 * MIB, 6 * GIB));
        assert!(next_pressure(true, 2 * GIB, 8 * GIB));
        // Recovers cleanly once physical pressure eases to >= 2.5 GiB.
        assert!(!next_pressure(true, 2560 * MIB, 6 * GIB));
        assert!(!next_pressure(true, 3 * GIB, 8 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit(0, true, true, false, 0), 0);
        assert_eq!(prefetch_limit(8 << 20, true, true, false, 0), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, true, false, 0), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false, false, false, 0), 192 << 20);
    }

    #[test]
    fn async_residency_fence_state_tracks_per_queue_waits_without_blocking() {
        assert!(parse_async_resident(None));
        assert!(parse_async_resident(Some("1")));
        assert!(!parse_async_resident(Some("0")));

        let state = ResidencyFenceState::new();
        let direct_waited = AtomicU64::new(0);
        let compute_waited = AtomicU64::new(0);

        assert_eq!(state.pending(), 0);
        assert_eq!(state.wait_target(&direct_waited), None);

        let v1 = state.reserve_next();
        let v2 = state.reserve_next();
        assert_eq!((v1, v2), (1, 2));
        state.record_enqueued(v1);
        state.record_enqueued(v2);
        assert_eq!(state.pending(), 2);

        // Direct queue sees pending target 2, marks waited, then sees None.
        assert_eq!(state.wait_target(&direct_waited), Some(2));
        state.mark_waited(&direct_waited, 2, true);
        assert_eq!(state.wait_target(&direct_waited), None);

        // Compute queue independently still needs to wait up to 2.
        assert_eq!(state.wait_target(&compute_waited), Some(2));
        state.mark_waited(&compute_waited, 2, false);
        assert_eq!(state.wait_target(&compute_waited), None);

        // A new enqueue advances pending to 3 for both queues.
        let v3 = state.reserve_next();
        state.record_enqueued(v3);
        assert_eq!(state.wait_target(&direct_waited), Some(3));
        assert_eq!(state.wait_target(&compute_waited), Some(3));
        assert_eq!(state.stats(), (3, 1, 3));
    }

    #[test]
    fn async_enqueue_make_resident_attaches_queue_wait_on_live_device() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

        let Some(host) = crate::host::host() else {
            return;
        };
        if !async_active(&host.device) {
            return;
        }
        let before = async_stats(&host.device);

        // Create a NOT_RESIDENT default-heap buffer and a NOT_RESIDENT L0 shared buffer,
        // enqueue both via `EnqueueMakeResident` without CPU blocking, map + write the L0
        // buffer, copy L0 -> default -> readback after attaching GPU-side queue wait.
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 4096,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let l0_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let default_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut src: Option<ID3D12Resource> = None;
        let mut dst: Option<ID3D12Resource> = None;
        // SAFETY: plain committed buffer creation with `CREATE_NOT_RESIDENT`.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &l0_heap,
                    D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut src,
                )
                .unwrap();
            host.device
                .CreateCommittedResource(
                    &default_heap,
                    D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut dst,
                )
                .unwrap();
        }
        let src = src.unwrap();
        let dst = dst.unwrap();

        assert!(enqueue_resident(&host.device, &src).unwrap());
        assert!(enqueue_resident(&host.device, &dst).unwrap());
        let after = async_stats(&host.device);
        assert!(after.0 >= before.0 + 2);
        assert!(after.2 >= before.2 + 2);

        // `Map()` on L0 buffer + write pattern.
        let mut ptr = std::ptr::null_mut();
        // SAFETY: CPU-visible L0 buffer.
        unsafe {
            src.Map(0, None, Some(&mut ptr)).unwrap();
            let slice = std::slice::from_raw_parts_mut(ptr.cast::<u8>(), 4096);
            for (i, b) in slice.iter_mut().enumerate() {
                *b = (i.wrapping_mul(37) ^ 0xA5) as u8;
            }
            src.Unmap(0, None);
        }

        let rb = host.read_back_buffer(4096).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: open direct list copying between live buffers.
        unsafe {
            list.list.CopyBufferRegion(&dst, 0, &src, 0, 4096);
            list.list.ResourceBarrier(&[crate::host::transition(
                &dst,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&rb, 0, &dst, 0, 4096);
        }
        list.close();
        // `execute_sync_internal` attaches `wait_pending_residency_on_direct()` before ExecuteCommandLists.
        host.execute_sync_internal(&list);

        let mut rb_ptr = std::ptr::null_mut();
        // SAFETY: readback completed.
        let got = unsafe {
            rb.Map(0, None, Some(&mut rb_ptr)).unwrap();
            let v = std::slice::from_raw_parts(rb_ptr.cast::<u8>(), 4096).to_vec();
            rb.Unmap(0, None);
            v
        };
        for (i, &b) in got.iter().enumerate() {
            assert_eq!(b, (i.wrapping_mul(37) ^ 0xA5) as u8, "byte {i} mismatch");
        }
        // When D3D12 Debug Layer is active, assert zero ERROR/CORRUPTION messages.
        if let Ok(iq) = host.device.cast::<ID3D12InfoQueue>() {
            // SAFETY: live info queue query.
            let n = unsafe { iq.GetNumStoredMessages() };
            for i in 0..n {
                let mut len = 0usize;
                // SAFETY: query message size then read into buffer.
                if unsafe { iq.GetMessage(i, None, &mut len) }.is_ok() && len > 0 {
                    let mut buf = vec![0u8; len];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if unsafe { iq.GetMessage(i, Some(msg), &mut len) }.is_ok() {
                        // SAFETY: filled by `GetMessage`.
                        let sev = unsafe { (*msg).Severity };
                        assert!(
                            sev != D3D12_MESSAGE_SEVERITY_CORRUPTION
                                && sev != D3D12_MESSAGE_SEVERITY_ERROR,
                            "unexpected D3D12 debug error/corruption during async residency"
                        );
                    }
                }
            }
        }
    }
}
