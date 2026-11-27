//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
    /// 1-descriptor shader-visible fallback heaps bound when a command list
    /// omits a view or sampler heap while using directly-indexed root signatures.
    fallback_view: OnceLock<Option<ID3D12DescriptorHeap>>,
    fallback_sampler: OnceLock<Option<ID3D12DescriptorHeap>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<std::collections::VecDeque<(u64, Retired)>> = Mutex::new(std::collections::VecDeque::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
#[allow(dead_code)]
enum Retired {
    Any(Box<dyn std::any::Any>),
    /// What a descriptor slot showed: 20,000 a frame, without a box each.
    Resource(ID3D12Resource),
    /// A thread-local batch of retired descriptor-slot resources flushed together
    /// to avoid per-slot contention on the global `RETIRED` mutex.
    Resources(Vec<ID3D12Resource>),
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// [`now_ms`] to the system's tick (16 ms): a few nanoseconds, where the
/// performance counter's are 25. For what is noted tens of thousands of
/// times a frame and read in tenths of a second.
pub(crate) fn coarse_ms() -> u64 {
    use windows_sys::Win32::System::SystemInformation::GetTickCount64;
    // The same beginning as `now_ms`.
    static BEGAN: OnceLock<u64> = OnceLock::new();
    // SAFETY: a plain query.
    let now = unsafe { GetTickCount64() };
    now.saturating_sub(*BEGAN.get_or_init(|| now.saturating_sub(now_ms())))
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((coarse_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// How long host copies of textures the title fills must have been idle to
/// be let go, in milliseconds, and how many bytes of them go at most (the
/// longest idle first): none but those of ten minutes while what is in use
/// of the card (`used`: without what our own blocks have free, with what is
/// kept free for others) is under 91 % of its budget, else enough to come
/// down to 88 %. `VICTORY_TEXTURE_KEEP=0`: [`vram_idle_ms`]' older rule,
/// all that are idle; `=1`: that rule with its two longer times.
pub(crate) fn vram_keep(budget: u64, used: u64) -> (u64, u64) {
    const KEPT_MS: u64 = 600_000;
    const IDLE_MS: u64 = 5_000;
    const SHORT_MS: u64 = 1_500;
    if budget == 0 {
        return (IDLE_MS, u64::MAX);
    }
    let share = |percent: u64| budget / 100 * percent;
    if used < share(91) {
        (KEPT_MS, u64::MAX)
    } else if used < share(95) {
        (IDLE_MS, used - share(88))
    } else {
        (SHORT_MS, used - share(88))
    }
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
///
/// With room on the card nothing has to go: in session play-1125-0127 88,000
/// of them (31 GB) went after 5 s with 7 GB of the card free, and 24,700
/// (12 GB) had to be made and uploaded again.
/// `VICTORY_TEXTURE_KEEP=0`: 5 s there too, as before.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    static KEEP: OnceLock<bool> = OnceLock::new();
    let keep = *KEEP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_KEEP").is_none_or(|v| v != "0"));
    vram_idle_for(video_memory, keep)
}

fn vram_idle_for(video_memory: Option<[u64; 4]>, keep: bool) -> u64 {
    const KEPT_MS: u64 = 600_000;
    const LONG_MS: u64 = 30_000;
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else if !keep {
        NORMAL_MS
    } else if used_pct >= 68 {
        LONG_MS
    } else {
        KEPT_MS
    }
}

/// Whether new memory is made resident, and waited for, before anything can
/// name it (on by default; `VICTORY_WAIT_RESIDENT=0`: as before). A heap or
/// resource of the default kind is resident once it is made, but its
/// mapping into the card's address space is work of the system's that runs
/// later: only lists sent after it wait for it. Lists sent before run on,
/// and read our descriptor heaps as they are: a slot that shows a texture
/// in a new block is a page fault for them until the block is mapped. The
/// device losses of the night of 2026-11-25 were that (run n08: the fault
/// lay in a texture made in the same tenth of a second, in a new block).
fn wait_resident() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_RESIDENT").is_none_or(|v| v != "0"))
}

/// The flag memory is made with that [`resident`] waits for.
pub(crate) fn not_resident() -> D3D12_HEAP_FLAGS {
    if wait_resident() { D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT } else { D3D12_HEAP_FLAG_NONE }
}

/// A queue of its own that new memory is waited for with ([`mapped`]).
struct Aside {
    queue: ID3D12CommandQueue,
    list: ID3D12CommandList,
    fence: ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    value: u64,
    _keep: (ID3D12CommandAllocator, [ID3D12Resource; 2]),
}
// SAFETY: D3D12 objects are free-threaded, the event is waited on by one
// thread at a time (the lock).
unsafe impl Send for Aside {}

#[derive(Default)]
struct MappedCoalescerState {
    requested: u64,
    completed: u64,
    running: bool,
}

struct MappedCoalescer {
    state: Mutex<MappedCoalescerState>,
    done: std::sync::Condvar,
}

impl MappedCoalescer {
    const fn new() -> Self {
        Self {
            state: Mutex::new(MappedCoalescerState {
                requested: 0,
                completed: 0,
                running: false,
            }),
            done: std::sync::Condvar::new(),
        }
    }

    /// Waits until a flush started strictly after this caller arrived has
    /// finished. Callers arriving while a flush is already running coalesce
    /// into one follow-up flush. Returns `Ok(true)` if this caller ran `flush`.
    fn wait<E>(&self, idle: impl Fn() -> bool, flush: impl FnOnce() -> Result<(), E>) -> Result<bool, E> {
        if idle() {
            return Ok(false);
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.requested += 1;
        let target = state.requested;
        loop {
            if state.completed >= target {
                return Ok(false);
            }
            if state.running {
                state = self.done.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            if idle() {
                state.completed = state.requested;
                drop(state);
                self.done.notify_all();
                return Ok(false);
            }
            state.running = true;
            let batch_end = state.requested;
            drop(state);

            struct ResetOnDrop<'a>(&'a MappedCoalescer, u64, bool);
            impl Drop for ResetOnDrop<'_> {
                fn drop(&mut self) {
                    let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
                    if self.2 {
                        s.completed = s.completed.max(self.1);
                    }
                    s.running = false;
                    drop(s);
                    self.0.done.notify_all();
                }
            }
            let mut guard = ResetOnDrop(self, batch_end, false);
            flush()?;
            guard.2 = true;
            drop(guard);
            return Ok(true);
        }
    }
}

/// New memory of the card is mapped into its address space when this
/// returns: a copy of a few bytes is sent to a queue of ours after the
/// memory was made, which the system starts only when all mappings asked
/// for so far are done, and waited for. (`MakeResident` came back before
/// that: run n11 was lost as n08 was with it alone.) Best done off a hot
/// path: a tenth of a millisecond to some.
pub(crate) fn mapped(device: &ID3D12Device) -> windows::core::Result<()> {
    static COALESCER: MappedCoalescer = MappedCoalescer::new();
    static ASIDE: Mutex<Option<Aside>> = Mutex::new(None);
    if !wait_resident() {
        return Ok(());
    }
    let no_flights = || STARTED.load(Ordering::Acquire) == FINISHED.load(Ordering::Acquire);
    if no_flights() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let ran = COALESCER.wait(no_flights, || -> windows::core::Result<()> {
        let mut aside = ASIDE.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: plain object creation; the objects the list names are kept.
        unsafe {
            if aside.is_none() {
                let buffer = |kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> windows::core::Result<ID3D12Resource> {
                    let mut made: Option<ID3D12Resource> = None;
                    device.CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: 256,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut made,
                    )?;
                    made.ok_or_else(windows::core::Error::empty)
                };
                let from = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
                let to = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?;
                let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                list.CopyBufferRegion(&to, 0, &from, 0, 16);
                list.Close()?;
                *aside = Some(Aside {
                    queue,
                    list: list.cast()?,
                    fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                    event: CreateEventW(None, false, false, None)?,
                    value: 0,
                    _keep: (allocator, [from, to]),
                });
            }
            let Some(a) = aside.as_mut() else { return Ok(()) };
            a.queue.ExecuteCommandLists(&[Some(a.list.clone())]);
            a.value += 1;
            a.queue.Signal(&a.fence, a.value)?;
            if a.fence.GetCompletedValue() < a.value {
                a.fence.SetEventOnCompletion(a.value, a.event)?;
                WaitForSingleObject(a.event, INFINITE);
            }
        }
        Ok(())
    })?;
    if !ran {
        return Ok(());
    }
    let us = started.elapsed().as_micros() as u64;
    MAPPED[0].fetch_add(1, Ordering::Relaxed);
    MAPPED[1].fetch_add(us, Ordering::Relaxed);
    MAPPED[2].fetch_max(us, Ordering::Relaxed);
    let n = MAPPED[0].load(Ordering::Relaxed);
    if n == 1 || n.is_multiple_of(200) {
        tracing::info!(
            "new card memory waited for until it is mapped (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            MAPPED[1].load(Ordering::Relaxed) as f64 / 1000.0,
            MAPPED[2].load(Ordering::Relaxed) as f64 / 1000.0,
        );
    }
    Ok(())
}

/// Times, microseconds and the longest of [`mapped`].
static MAPPED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
#[cfg(feature = "diagnostics")]
static PRESENT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// New memory (made with [`not_resident`]) is resident when this returns.
pub(crate) fn resident(device: &ID3D12Device, object: &impl Interface) -> windows::core::Result<()> {
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let object: ID3D12Pageable = object.cast()?;
    // SAFETY: a live heap or resource of the device's.
    let made = unsafe { device.MakeResident(&[Some(object)]) };
    let us = started.elapsed().as_micros() as u64;
    RESIDENT[0].fetch_add(1, Ordering::Relaxed);
    RESIDENT[1].fetch_add(us, Ordering::Relaxed);
    RESIDENT[2].fetch_max(us, Ordering::Relaxed);
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = RESIDENT[0].load(Ordering::Relaxed);
    if n >= 1000 * (SAID.load(Ordering::Relaxed) + 1) {
        SAID.store(n / 1000, Ordering::Relaxed);
        tracing::info!(
            "new memory waited for (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            RESIDENT[1].load(Ordering::Relaxed) as f64 / 1000.0,
            RESIDENT[2].load(Ordering::Relaxed) as f64 / 1000.0
        );
    }
    made
}

/// Times, microseconds and the longest of [`resident`].
static RESIDENT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether memory the card was given stays with it while there is room (on
/// by default; `VICTORY_KEEP_FREED=0`: it is given back as before). A shader
/// that reads past a view (a raw load through a typed, structured or
/// constant buffer view has no end on AMD: `indirect::tests::raw_loads_past_views`)
/// or through a slot that shows wrong data reads whatever lies there, and
/// nothing happens; where memory was given back it is a page fault, and the
/// device is lost. Both losses of 2026-11-25 were reads in memory given back
/// seconds before (a block of textures; a list's scratch and a window of
/// title memory). Kept are: empty blocks of textures ([`crate::pool`]) below
/// 85 % of the card's budget, windows of title memory and staging buffers
/// for five minutes with 12 GiB of the PC's memory free, and 1 GiB of
/// scratch chunks.
pub(crate) fn keep_freed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_FREED").is_none_or(|v| v != "0"))
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
/// With plenty of it five minutes ([`keep_freed`]).
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    let effective = match (avail_phys, crate::residency::headroom()) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    ram_idle_for(effective, keep_freed() && !crate::residency::limited_host_ram())
}

fn ram_idle_for(avail_phys: Option<u64>, keep: bool) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const KEPT_MS: u64 = 300_000;
    if keep && avail_phys.is_some_and(|avail| avail >= 12 * GIB) {
        return KEPT_MS;
    }
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// System-RAM pressure band for optional caches. Resident working set and
/// private bytes describe this process; free physical and commit headroom
/// guard the PC and other processes. Windows API queries never change the
/// pagefile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RamBand {
    Calm,
    Watch,
    High,
}

/// Band for one sample. Entering Watch or High needs a higher watermark than
/// leaving it, so a value near a threshold does not toggle the policy every
/// sample.
pub(crate) fn ram_band(
    previous: RamBand,
    working: u64,
    private: u64,
    free_physical: u64,
    commit_headroom: u64,
    total_physical: u64,
    commit_limit: u64,
) -> RamBand {
    const GIB: u64 = 1024 * 1024 * 1024;
    // Fractions of the machine, so a 64 GB host is not treated like the
    // 32 GB one. The free and headroom floors stay absolute: pressure is
    // pressure however big the machine is.
    let high = private * 5 >= commit_limit * 4
        || working * 2 >= total_physical
        || free_physical < 2 * GIB
        || commit_headroom < GIB;
    let watch = private * 10 >= commit_limit * 7
        || working * 4 >= total_physical
        || free_physical < 4 * GIB
        || commit_headroom < 2 * GIB;
    let wanted = if high {
        RamBand::High
    } else if watch {
        RamBand::Watch
    } else {
        RamBand::Calm
    };
    let safe_high = private * 20 <= commit_limit * 13
        && working * 4 <= total_physical
        && free_physical >= 4 * GIB
        && commit_headroom >= 3 * GIB;
    let safe_watch = private * 5 <= commit_limit * 3
        && working * 5 <= total_physical
        && free_physical >= 6 * GIB
        && commit_headroom >= 3 * GIB;
    match (previous, wanted) {
        (RamBand::High, RamBand::Calm) if !safe_high => RamBand::Watch,
        (RamBand::Watch, RamBand::Calm) if !safe_watch => RamBand::Watch,
        _ => wanted,
    }
}

/// The six numbers [`ram_band`] needs, from Windows. `None` if a query fails,
/// so a caller can keep its old band instead of guessing.
#[cfg(windows)]
pub(crate) fn ram_sample() -> Option<(u64, u64, u64, u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: plain data the call fills in.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return None;
    }
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return None;
    }
    Some((
        c.WorkingSetSize as u64,
        c.PagefileUsage as u64,
        m.ullAvailPhys,
        m.ullAvailPageFile,
        m.ullTotalPhys,
        m.ullTotalPageFile,
    ))
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::book::check();
    crate::residency::refresh();
    if crate::residency::critical_commit() {
        trim_spare_chunks();
    }
    crate::list::trim_idle();
    trim_idle_preludes();
    crate::pool::room(host().and_then(|h| h.video_memory()));
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// The number of the newest submission begun, and of the newest of which all
/// before it have finished too.
pub(crate) fn submissions() -> (u64, u64) {
    (STARTED.load(Ordering::Acquire), FINISHED.load(Ordering::Acquire))
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    // (Those the title made and the host has not begun get the next numbers.)
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((started, Retired::Any(Box::new(object))));
}

const RETIRE_BATCH_CAP: usize = 128;

struct RetireBatch {
    depth: usize,
    items: Vec<ID3D12Resource>,
}

thread_local! {
    static RETIRE_BATCH: std::cell::RefCell<RetireBatch> =
        const { std::cell::RefCell::new(RetireBatch { depth: 0, items: Vec::new() }) };
}

fn flush_retire_vec(items: Vec<ID3D12Resource>) {
    if items.is_empty() {
        return;
    }
    let count = items.len() as u64;
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(count, Ordering::Relaxed);
        drop(items);
        return;
    }
    RETIREMENTS[1].fetch_add(count, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((started, Retired::Resources(items)));
}

/// RAII scope that batches `retire_resource` calls on this thread and flushes
/// them to `RETIRED` in chunks of [`RETIRE_BATCH_CAP`] and on drop.
pub(crate) struct RetireBatchGuard;

impl RetireBatchGuard {
    pub(crate) fn enter() -> Self {
        RETIRE_BATCH.with(|b| b.borrow_mut().depth += 1);
        RetireBatchGuard
    }
}

impl Drop for RetireBatchGuard {
    fn drop(&mut self) {
        let flushed = RETIRE_BATCH.with(|b| {
            let mut b = b.borrow_mut();
            b.depth = b.depth.saturating_sub(1);
            if b.depth == 0 && !b.items.is_empty() {
                Some(std::mem::take(&mut b.items))
            } else {
                None
            }
        });
        if let Some(items) = flushed {
            flush_retire_vec(items);
        }
    }
}

/// [`retire`] for a resource.
pub(crate) fn retire_resource(object: ID3D12Resource) {
    let buffered = RETIRE_BATCH.with(|b| {
        let mut b = b.borrow_mut();
        if b.depth == 0 {
            return Err(object);
        }
        b.items.push(object);
        if b.items.len() >= RETIRE_BATCH_CAP {
            Ok(Some(std::mem::replace(&mut b.items, Vec::with_capacity(RETIRE_BATCH_CAP))))
        } else {
            Ok(None)
        }
    });
    match buffered {
        Ok(Some(items)) => flush_retire_vec(items),
        Ok(None) => {}
        Err(object) => {
            let queued = crate::queue::queued();
            let started = STARTED.load(Ordering::Acquire) + queued;
            if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
                RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
                drop(object);
                return;
            }
            RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
            RETIRED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push_back((started, Retired::Resource(object)));
        }
    }
}

/// How many submissions later than it could (the last one that may use it
/// has finished) an object let go is dropped, and a host list that has run
/// is reset, about two frames: the device losses of the night of 2026-11-25
/// were page faults in memory nothing of ours named any more, at no command
/// in particular. Objects let go while nothing runs wait as well: lists
/// recorded and not yet sent may name them. `VICTORY_BURY_LAG=0`: not.
fn bury_lag() -> u64 {
    static LAG: OnceLock<u64> = OnceLock::new();
    *LAG.get_or_init(|| {
        let default = if crate::residency::limited_host_ram() { 16 } else { 64 };
        std::env::var("VICTORY_BURY_LAG").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    })
}

/// How many fence-completed flights remain buffered in `line.landed` before
/// their `HostList`s are reset and recycled. Shrinks under RAM pressure so
/// finished command allocators are reused or freed instead of forcing new ones.
fn land_lag(base_lag: usize, private_bytes: u64, pressured: bool, critical: bool) -> usize {
    const HIGH_PRIVATE: u64 = 18 << 30;
    const WATCH_PRIVATE: u64 = 14 << 30;
    if critical || private_bytes >= HIGH_PRIVATE {
        0
    } else if pressured || private_bytes >= WATCH_PRIVATE {
        (base_lag / 4).min(4)
    } else {
        base_lag
    }
}

/// Whether descriptor slots written again with the object they already
/// held let the old reference go at once instead of retiring it, and
/// [`bury`] partitions what it drops outside the lock (`VICTORY_RETIRE_LEAN=1`;
/// `alt`: twenty seconds with, twenty without, in turn, each change said in
/// the log). Off by default.
pub(crate) fn retire_lean() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_RETIRE_LEAN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "descriptor slots written again with the same object let the old reference go at once from now (VICTORY_RETIRE_LEAN=alt): {}",
                    if on { "ON" } else { "off, retired as usual" }
                );
            }
            on
        }
    }
}

/// Descriptor slots written since the start (`HostHeap::keep`): with an
/// object where they held one, and of those the ones that held the same
/// object; and the old references let go at once for that
/// ([`retire_lean`]). Said every 2^20 writes of the first kind.
#[allow(dead_code)]
pub(crate) static SLOT_REWRITES: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Counts a slot write that replaced `old` by `new` and says whether the
/// old reference may go at once (the same object, [`retire_lean`] on).
#[allow(dead_code)]
pub(crate) fn slot_rewritten(old: &ID3D12Resource, new: Option<&ID3D12Resource>) -> bool {
    let same = new.is_some_and(|new| new.as_raw() == old.as_raw());
    let before = SLOT_REWRITES[0].fetch_add(1, Ordering::Relaxed);
    if same {
        SLOT_REWRITES[1].fetch_add(1, Ordering::Relaxed);
    }
    let lean = same && retire_lean();
    if lean {
        SLOT_REWRITES[2].fetch_add(1, Ordering::Relaxed);
    }
    if (before + 1).is_multiple_of(1 << 20) {
        tracing::info!(
            "descriptor slots written over an object they held: {} so far, {} of them with the same object; {} old references let go at once (VICTORY_RETIRE_LEAN), the rest retired",
            before + 1,
            SLOT_REWRITES[1].load(Ordering::Relaxed),
            SLOT_REWRITES[2].load(Ordering::Relaxed)
        );
    }
    lean
}

fn drop_retired_async(done: Vec<(u64, Retired)>) {
    static TX: OnceLock<std::sync::mpsc::SyncSender<Vec<(u64, Retired)>>> = OnceLock::new();
    let tx = TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<(u64, Retired)>>(64);
        let _ = std::thread::Builder::new().name("victory-bury".into()).spawn(move || {
            for batch in rx {
                drop(batch);
            }
        });
        tx
    });
    if tx.send(done).is_err() {
        // Worker exited during shutdown; nothing left to do.
    }
}

/// Drops what waited for submissions up to `finished` ([`bury_lag`]). The
/// numbers come in order, but for threads that overtake each other: one of
/// those is dropped a little later than it could be.
fn bury(finished: u64) {
    let finished = finished.saturating_sub(bury_lag());
    if retire_lean() {
        let all = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
        if all.is_empty() {
            return;
        }
        let (done, keep): (std::collections::VecDeque<_>, std::collections::VecDeque<_>) =
            all.into_iter().partition(|(at, _)| *at <= finished);
        if !keep.is_empty() {
            let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
            if retired.is_empty() {
                *retired = keep;
            } else {
                // Keep older waiting entries ahead of newly retired ones to preserve FIFO order.
                let newer = std::mem::take(&mut *retired);
                *retired = keep;
                retired.extend(newer);
            }
        }
        if !done.is_empty() {
            drop_retired_async(Vec::from(done));
        }
        return;
    }
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        let n = retired.iter().take_while(|(at, _)| *at <= finished).count();
        if n == 0 {
            return;
        }
        retired.drain(..n).collect()
    };
    drop_retired_async(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// Defaults on in `diagnostics` builds and off in release builds
/// (`VICTORY_TRAIL=1` enables it in release; `VICTORY_TRAIL=0` disables it).
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on_for(value: Option<&std::ffi::OsStr>, diagnostics: bool) -> bool {
    match value {
        Some(v) => v != "0",
        None => diagnostics,
    }
}

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        trail_on_for(
            std::env::var_os("VICTORY_TRAIL").as_deref(),
            cfg!(any(test, feature = "diagnostics")),
        )
    })
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !(trail_on() || dred_any()) || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
    if dred_any() {
        let data = host().map(|h| h.removal_data(&[])).unwrap_or_default();
        tracing::error!("device-removed data{}", if data.is_empty() { "; none" } else { &data });
    }
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Live preludes owned by `HostList`s and `IDLE_PRELUDES`, and those currently
/// sitting idle in `IDLE_PRELUDES`.
static PRELUDES_LIVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PRELUDES_POOLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A prelude whose lifetime is tracked in [`PRELUDES_LIVE`].
/// `open` is true when `allocator.Reset()` + `list.Reset(allocator, None)`
/// already ran off `victory-submit` (e.g. in `recycle_prelude` on `victory-land`).
struct TrackedPrelude {
    prelude: Prelude,
    open: bool,
}

// SAFETY: D3D12 command allocators and command lists are free-threaded; a
// pooled prelude is never touched while executing.
unsafe impl Send for TrackedPrelude {}

impl TrackedPrelude {
    fn new(prelude: Prelude, open: bool) -> Self {
        PRELUDES_LIVE.fetch_add(1, Ordering::Relaxed);
        Self { prelude, open }
    }
}

impl Drop for TrackedPrelude {
    fn drop(&mut self) {
        PRELUDES_LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Fence-completed preludes reset and kept for reuse across all `HostList`s.
static IDLE_PRELUDES: Mutex<Vec<TrackedPrelude>> = Mutex::new(Vec::new());

/// Computes pressure-adjusted `(idle_lists, idle_preludes)` caps from configured
/// base caps, current process private commit, and host RAM pressure flags.
pub(crate) fn allocator_pool_caps(
    base_lists: usize,
    base_preludes: usize,
    private_bytes: u64,
    pressured: bool,
    critical: bool,
) -> (usize, usize) {
    const WATCH_PRIVATE: u64 = 16 << 30;
    const HIGH_PRIVATE: u64 = 26 << 30;
    // Keep enough idle HostLists (~256-320) and tiny barrier preludes (~32-64,
    // ~4 MB total) so 2 frames of async-submit + thin list workers never drop
    // and recreate command allocators, budget shared_buffers, and indirect UAVs
    // (each drop+create triggers 3x MakeResident + mapped() GPU fence waits).
    if critical || private_bytes >= HIGH_PRIVATE {
        (base_lists.min(128), (base_preludes / 2).min(32))
    } else if pressured || private_bytes >= WATCH_PRIVATE {
        (base_lists.min(320), base_preludes.min(64))
    } else {
        (base_lists, base_preludes)
    }
}

fn resolve_base_allocator_caps(
    alloc_env: Option<&str>,
    prelude_env: Option<&str>,
    limited_ram: bool,
) -> (usize, usize) {
    let default_lists = if limited_ram { 256 } else { 384 };
    let lists = alloc_env
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default_lists);
    let default_preludes = if alloc_env.is_some() {
        (lists / 2).max(16).min(lists)
    } else if limited_ram {
        64
    } else {
        128
    };
    let preludes = prelude_env
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default_preludes);
    (lists, preludes)
}

fn base_allocator_caps() -> (usize, usize) {
    static CAPS: OnceLock<(usize, usize)> = OnceLock::new();
    *CAPS.get_or_init(|| {
        let alloc_env = std::env::var("VICTORY_ALLOCATOR_CAP").ok();
        let prelude_env = std::env::var("VICTORY_PRELUDE_CAP").ok();
        resolve_base_allocator_caps(
            alloc_env.as_deref(),
            prelude_env.as_deref(),
            crate::residency::limited_host_ram(),
        )
    })
}

/// Current idle pool caps `(host_lists, preludes)` after RAM pressure scaling.
pub(crate) fn current_allocator_caps() -> (usize, usize) {
    let (base_lists, base_preludes) = base_allocator_caps();
    allocator_pool_caps(
        base_lists,
        base_preludes,
        process_private_bytes(),
        crate::residency::pressured(),
        crate::residency::critical_commit(),
    )
}

/// Live and pooled allocator counts: `(lists_live, lists_pooled, preludes_live, preludes_pooled)`.
pub(crate) fn allocator_counts() -> (u64, u64, u64, u64) {
    (
        ALIVE[0].load(Ordering::Relaxed),
        crate::list::idle_count(),
        PRELUDES_LIVE.load(Ordering::Relaxed),
        PRELUDES_POOLED.load(Ordering::Relaxed),
    )
}

/// Returns a fence-completed prelude to the global pool up to its cap, or drops
/// it outside the lock so the driver frees its command-allocator commit.
fn recycle_prelude(mut prelude: TrackedPrelude) {
    let cap = current_allocator_caps().1;
    if cap == 0 || PRELUDES_POOLED.load(Ordering::Relaxed) >= cap as u64 {
        drop(prelude);
        return;
    }
    // Pre-reset both allocator and command list off `victory-submit` so the
    // next submission only records barriers/copies and calls `list.Close()`.
    // SAFETY: called only after the submission that ran this prelude has completed.
    let reset_ok = unsafe {
        if prelude.open {
            true
        } else {
            prelude
                .prelude
                .0
                .Reset()
                .and_then(|()| prelude.prelude.1.Reset(&prelude.prelude.0, None))
                .is_ok()
        }
    };
    if !reset_ok {
        drop(prelude);
        return;
    }
    prelude.open = true;
    let excess = {
        let mut pool = IDLE_PRELUDES.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() < cap {
            pool.push(prelude);
            PRELUDES_POOLED.store(pool.len() as u64, Ordering::Relaxed);
            None
        } else {
            Some(prelude)
        }
    };
    drop(excess);
}

/// Drops excess idle preludes past the current pressure-adjusted cap.
fn trim_idle_preludes() {
    let cap = current_allocator_caps().1;
    let dropped: Vec<TrackedPrelude> = {
        let mut pool = IDLE_PRELUDES.lock().unwrap_or_else(|e| e.into_inner());
        if pool.len() <= cap {
            return;
        }
        let excess = pool.drain(cap..).collect();
        PRELUDES_POOLED.store(pool.len() as u64, Ordering::Relaxed);
        excess
    };
    drop(dropped);
}

/// Process memory counters: `(working_set_bytes, private_commit_bytes, page_fault_count)`.
pub(crate) fn process_memory_counters() -> Option<(u64, u64, u32)> {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return None;
    }
    Some((c.WorkingSetSize as u64, c.PagefileUsage as u64, c.PageFaultCount))
}

/// 32-bit wraparound-safe delta of `PROCESS_MEMORY_COUNTERS::PageFaultCount`.
pub(crate) fn page_fault_delta(prev: u32, now: u32) -> u64 {
    u64::from(now.wrapping_sub(prev))
}

/// Refresh interval for cached [`process_private_bytes`] queries so
/// per-reset allocator cap checks don't storm `K32GetProcessMemoryInfo`.
const PRIVATE_BYTES_REFRESH_MS: u64 = 500;
static CACHED_PRIVATE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `coarse_ms() + 1` of the last `K32GetProcessMemoryInfo` refresh; `0` = uninitialized.
static CACHED_PRIVATE_BYTES_AT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The process's private committed bytes (`PagefileUsage`), cached for
/// [`PRIVATE_BYTES_REFRESH_MS`] to avoid repeated kernel VM counter syscalls.
pub(crate) fn process_private_bytes() -> u64 {
    let now = coarse_ms().saturating_add(1);
    let at = CACHED_PRIVATE_BYTES_AT.load(Ordering::Relaxed);
    if at != 0 && now.saturating_sub(at) < PRIVATE_BYTES_REFRESH_MS {
        return CACHED_PRIVATE_BYTES.load(Ordering::Relaxed);
    }
    if CACHED_PRIVATE_BYTES_AT
        .compare_exchange(at, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
        && at != 0
    {
        return CACHED_PRIVATE_BYTES.load(Ordering::Relaxed);
    }
    let private = process_memory_counters().map_or(0, |(_, private, _)| private);
    CACHED_PRIVATE_BYTES.store(private, Ordering::Relaxed);
    private
}

/// Periodic RAM, DXGI local/shared VRAM, allocator pool, and OS page-fault
/// summary emitted in both release and diagnostics builds.
#[cfg(feature = "diagnostics")]
static PREP_PHASE_NS: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
#[cfg(feature = "diagnostics")]
static PREP_PHASE_FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn memory_summary_line(frames: f64) {
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let (working, private, faults_now) = process_memory_counters().unwrap_or((0, 0, 0));
    static PREV_FAULTS: AtomicU32 = AtomicU32::new(0);
    static HAS_PREV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let faults_delta = if HAS_PREV.swap(true, Ordering::Relaxed) {
        page_fault_delta(PREV_FAULTS.swap(faults_now, Ordering::Relaxed), faults_now)
    } else {
        PREV_FAULTS.store(faults_now, Ordering::Relaxed);
        0
    };
    let [local_budget, local_used, shared_budget, shared_used] =
        host().and_then(|h| h.video_memory()).unwrap_or_default();
    let (lists_live, lists_pooled, preludes_live, preludes_pooled) = allocator_counts();
    let frames = frames.max(1.0);
    tracing::info!(
        "memory summary: RAM private {:.0} MB, working set {:.0} MB; \
         VRAM local {:.0}/{:.0} MB, SHR {:.0}/{:.0} MB; \
         buffers: shadows {:.0} MB (bar {:.0} MB, card {:.0} MB), tex_stage {:.0} MB, scratch {:.0} MB; \
         allocators: lists {lists_live} live ({lists_pooled} pooled), \
         preludes {preludes_live} live ({preludes_pooled} pooled); \
         OS page faults {:.1}/frame",
        mb(private),
        mb(working),
        mb(local_used),
        mb(local_budget),
        mb(shared_used),
        mb(shared_budget),
        mb(crate::shadow::live_span_bytes()),
        mb(crate::shadow::window_mapped_bytes()),
        mb(crate::shadow::card_live_bytes()),
        mb(crate::texture::live_staging_bytes()),
        mb(ALIVE[2].load(Ordering::Relaxed)),
        faults_delta as f64 / frames,
    );
    #[cfg(feature = "diagnostics")]
    {
        let p_frames = PREP_PHASE_FRAMES.swap(0, Ordering::Relaxed).max(1) as f64;
        let p: Vec<f64> = PREP_PHASE_NS
            .iter()
            .map(|a| a.swap(0, Ordering::Relaxed) as f64 / 1e6 / p_frames)
            .collect();
        let present_ms = PRESENT_NS.swap(0, Ordering::Relaxed) as f64 / 1e6 / p_frames;
        if p.iter().sum::<f64>() > 0.1 {
            tracing::info!(
                "prep phases/f: pre {:.1}, watch {:.1}, stale {:.1}, front {:.1}, deferred {:.1}, \
                 prel_wait {:.1}, flush {:.1}, snap_card {:.1}, wait_heaps {:.1}, dxr {:.1}, exec {:.1}, post {:.1}, present {:.1} (sum {:.1} ms)",
                p[9], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[11], p[8], p[10], present_ms, p.iter().sum::<f64>()
            );
        }
    }
}

/// The process's exact private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    process_memory_counters().map_or(0, |(_, private, _)| private)
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU if it is made already.
#[cfg(feature = "diagnostics")]
pub(crate) fn made_host() -> Option<&'static Host> {
    HOST.get()?.as_ref()
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => {
            if crate::dxr::requested() {
                let _ = host.raytracing_tier_1_1();
            }
            Some(host)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_any() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                if dred_on() || dred_light() {
                    dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                tracing::info!(
                    breadcrumbs = dred_on() || dred_light(),
                    markers = dred_on(),
                    "crash tracing on (device-removed data)"
                );
            } else {
                tracing::warn!("crash tracing asked for and not there: no device-removed data settings");
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // Before anything is made with it (`VICTORY_RECORD`).
        #[cfg(feature = "diagnostics")]
        crate::replay::record::from_env(&device);
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
            // (RDNA has waves of 32 and 64, the vendor's older cards of 64 alone.)
            let amd = crate::indirect::amd(&device);
            victory_shader::emit::set_host_rdna(amd && o1.WaveLaneCountMin == 32 && o1.WaveLaneCountMax == 64);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
            fallback_view: OnceLock::new(),
            fallback_sampler: OnceLock::new(),
        })
    }
}

impl Host {
    /// The driver's version, as the system counts it (0: not known).
    pub(crate) fn driver(&self) -> i64 {
        let adapter = self.adapter.as_ref();
        // SAFETY: a plain query.
        adapter.map_or(0, |a| unsafe { a.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0))
    }

    /// Host direct queue GPU timestamp frequency in ticks per second, if non-zero.
    pub(crate) fn timestamp_frequency(&self) -> Option<u64> {
        // SAFETY: plain D3D12 command queue query.
        unsafe { self.queue.GetTimestampFrequency() }.ok().filter(|&f| f > 0)
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    /// A list of ours is found by its name, with the debug layer's note or not.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_is_found_by_the_number_in_its_name() {
        assert_eq!(super::list_number("victory host list 7"), Some(7));
        assert_eq!(super::list_number("victory host list 158 (debug layer indirect)"), Some(158));
        assert_eq!(super::list_number("victory host list 7x"), None);
        assert_eq!(super::list_number("victory host list 7 x"), None);
        assert_eq!(super::list_number("another list 7"), None);
        assert_eq!(super::list_number(""), None);
    }

    /// What is there of a range some of whose pages are not: those bytes,
    /// counted, and zeros for the rest, whatever the buffer held.
    #[test]
    fn what_is_there_is_read_and_the_rest_is_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x8_0000;
        // SAFETY: a reservation of ours; only committed pages are written.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            // 0-0x1_1000 and 0x3_0000-0x4_0000 are there.
            for (from, len) in [(0, 0x1_1000), (0x3_0000, 0x1_0000)] {
                assert!(!VirtualAlloc(at.add(from).cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.add(from).write_bytes(0x5a, len);
            }
            // From an address inside a page, as textures are packed.
            let mut out = vec![0x77u8; SIZE - 0x300];
            assert_eq!(super::read_there(at.add(0x300), &mut out), 0x1_1000 - 0x300 + 0x1_0000);
            for (i, &b) in out.iter().enumerate() {
                let from = i + 0x300;
                let there = from < 0x1_1000 || (0x3_0000..0x4_0000).contains(&from);
                assert_eq!(b, if there { 0x5a } else { 0 }, "byte {from:#x}");
            }
            // All there: one read.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x1000), &mut out), 0x1_0000);
            assert!(out.iter().all(|&b| b == 0x5a));
            // Nothing there.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x5_0000), &mut out), 0);
            assert!(out.iter().all(|&b| b == 0));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Zeros for a title destination that went meanwhile are no fault.
    #[test]
    fn zeros_where_nothing_is_are_lost() {
        use windows_sys::Win32::System::Memory::*;
        // SAFETY: a reservation of ours, its first page committed.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE).is_null());
            at.write_bytes(7, 4096);
            super::zero(at, 4096);
            assert!(std::slice::from_raw_parts(at, 4096).iter().all(|&b| b == 0));
            super::zero(at.add(4096), 4096);
            VirtualFree(at.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn deferred_clear_retains_normal_capacity_and_resets_flags() {
        let mut d = Deferred::default();
        for i in 0..64usize {
            d.copies.push(Step::Copy(i, i, 16));
            d.refreshed.insert((i, i, 16));
            d.textures.insert(i);
            d.bound.insert(i);
        }
        d.has_placed = true;
        d.budget_zeroed = true;
        d.indirect_readable = 0x1234;
        let cap_copies = d.copies.capacity();
        let cap_refreshed = d.refreshed.capacity();
        let cap_textures = d.textures.capacity();
        d.clear();
        assert!(d.copies.is_empty());
        assert!(d.refreshed.is_empty());
        assert!(d.textures.is_empty());
        assert!(d.bound.is_empty());
        assert!(!d.has_placed);
        assert!(!d.budget_zeroed);
        assert_eq!(d.indirect_readable, 0);
        assert!(d.copies.capacity() >= cap_copies);
        assert!(d.refreshed.capacity() >= cap_refreshed);
        assert!(d.textures.capacity() >= cap_textures);
    }

    #[test]
    fn retire_batch_guard_handles_nesting_cleanly() {
        let _outer = RetireBatchGuard::enter();
        {
            let _inner = RetireBatchGuard::enter();
            RETIRE_BATCH.with(|b| assert_eq!(b.borrow().depth, 2));
        }
        RETIRE_BATCH.with(|b| assert_eq!(b.borrow().depth, 1));
        drop(_outer);
        RETIRE_BATCH.with(|b| assert_eq!(b.borrow().depth, 0));
    }

    #[test]
    fn mapped_coalescer_skips_when_idle_and_coalesces_waiters() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize};

        let coalescer = Arc::new(MappedCoalescer::new());
        let flushes = Arc::new(AtomicUsize::new(0));
        // Idle fast-skip: flush never runs.
        assert!(!coalescer.wait(|| true, || -> Result<(), ()> {
            flushes.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }).unwrap());
        assert_eq!(flushes.load(Ordering::Relaxed), 0);

        // Concurrent waiters arriving during flush #1 coalesce into one flush #2.
        let entered_first = Arc::new(std::sync::Barrier::new(2));
        let release_first = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
        let arrived = Arc::new(AtomicUsize::new(0));
        let all_waiting = Arc::new(AtomicBool::new(false));

        let c1 = coalescer.clone();
        let f1 = flushes.clone();
        let e1 = entered_first.clone();
        let r1 = release_first.clone();
        let t1 = std::thread::spawn(move || {
            c1.wait(|| false, || -> Result<(), ()> {
                f1.fetch_add(1, Ordering::SeqCst);
                e1.wait();
                let mut go = r1.0.lock().unwrap();
                while !*go {
                    go = r1.1.wait(go).unwrap();
                }
                Ok(())
            })
            .unwrap()
        });

        // Wait until t1 is inside flush #1.
        entered_first.wait();

        // Spawn 4 waiters while flush #1 is in flight.
        let mut waiters = Vec::new();
        for _ in 0..4 {
            let cw = coalescer.clone();
            let fw = flushes.clone();
            let aw = arrived.clone();
            let flag = all_waiting.clone();
            waiters.push(std::thread::spawn(move || {
                aw.fetch_add(1, Ordering::SeqCst);
                cw.wait(|| false, || -> Result<(), ()> {
                    assert!(flag.load(Ordering::SeqCst));
                    fw.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .unwrap()
            }));
        }

        // Wait until all 4 waiters have registered in `requested`.
        loop {
            if coalescer.state.lock().unwrap().requested >= 5 {
                break;
            }
            std::thread::yield_now();
        }
        all_waiting.store(true, Ordering::SeqCst);
        *release_first.0.lock().unwrap() = true;
        release_first.1.notify_all();

        assert!(t1.join().unwrap());
        let leaders: usize = waiters.into_iter().map(|h| usize::from(h.join().unwrap())).sum();
        assert_eq!(leaders, 1, "4 concurrent waiters should coalesce into 1 follow-up flush");
        assert_eq!(flushes.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn trail_defaults_to_diagnostics_and_honors_override() {
        use std::ffi::OsStr;
        assert!(!trail_on_for(None, false));
        assert!(trail_on_for(None, true));
        assert!(trail_on_for(Some(OsStr::new("1")), false));
        assert!(!trail_on_for(Some(OsStr::new("0")), true));
        assert!(!trail_on_for(Some(OsStr::new("0")), false));
    }

    #[test]
    fn process_private_bytes_returns_cached_nonzero_value() {
        let first = process_private_bytes();
        let second = process_private_bytes();
        assert!(first > 0);
        assert_eq!(first, second);
    }
}

/// Pooled indirect-batch UAV buffer in `D3D12_RESOURCE_STATE_COMMON`, shared
/// across all `HostList`s so idle lists never retain duplicate VRAM buffers.
struct BatchBuf(ID3D12Resource);
// SAFETY: D3D12 resources are free-threaded; buffers enter the pool only after
// their owning list's last GPU submission has completed and transitioned them
// back to COMMON (or kept them in COMMON under Enhanced Barriers).
unsafe impl Send for BatchBuf {}

/// Pooled mapped 256-byte indirect budget buffer in `D3D12_RESOURCE_STATE_COMMON`.
struct BudgetBuf(ID3D12Resource, usize);
// SAFETY: D3D12 resources are free-threaded and the persistent mapping pointer
// stays valid while the resource is alive.
unsafe impl Send for BudgetBuf {}

static SPARE_BATCH: Mutex<Vec<BatchBuf>> = Mutex::new(Vec::new());
const SPARE_BATCH_MAX: usize = 256;

static SPARE_BUDGET: Mutex<Vec<BudgetBuf>> = Mutex::new(Vec::new());
const SPARE_BUDGET_MAX: usize = 256;

static SPARE_CARD_BUDGET: Mutex<Vec<BatchBuf>> = Mutex::new(Vec::new());
const SPARE_CARD_BUDGET_MAX: usize = 256;

struct IndirectPool {
    kept: Vec<(BatchBuf, u64)>,
    bytes: u64,
}

static SPARE_INDIRECT: Mutex<IndirectPool> = Mutex::new(IndirectPool {
    kept: Vec::new(),
    bytes: 0,
});
const SPARE_INDIRECT_MAX_BYTES: u64 = 64 << 20;

fn take_batch_buffer(size: u64) -> Option<ID3D12Resource> {
    if let Some(BatchBuf(buf)) = SPARE_BATCH.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(buf);
    }
    let made = host()?.internal_uav_buffer(size)?;
    name(&made, || format!("indirect batches {size:#x}"));
    Some(made)
}

fn give_batch_buffers(buffers: Vec<ID3D12Resource>) {
    if buffers.is_empty() {
        return;
    }
    // BatchBuf lives in local VRAM (~128 KiB each, < 8 MiB at cap), not host RAM;
    // keep the full pool so recording threads don't churn internal allocations.
    let limit = SPARE_BATCH_MAX;
    let mut overflow = Vec::new();
    {
        let mut spare = SPARE_BATCH.lock().unwrap_or_else(|e| e.into_inner());
        for buf in buffers {
            if spare.len() < limit {
                spare.push(BatchBuf(buf));
            } else {
                overflow.push(buf);
            }
        }
    }
    drop(overflow);
}

fn take_budget_buffer() -> Option<(ID3D12Resource, usize)> {
    if let Some(BudgetBuf(buf, ptr)) = SPARE_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some((buf, ptr));
    }
    let (b, d) = host()?.internal_shared_buffer(256)?;
    Some((b, d as usize))
}

fn give_budget_buffer(buf: (ID3D12Resource, usize)) {
    let overflow = {
        let mut spare = SPARE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < SPARE_BUDGET_MAX {
            spare.push(BudgetBuf(buf.0, buf.1));
            None
        } else {
            Some(buf.0)
        }
    };
    drop(overflow);
}

fn take_card_budget_buffer() -> Option<ID3D12Resource> {
    if let Some(BatchBuf(buf)) = SPARE_CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).pop() {
        return Some(buf);
    }
    let buffer = host()?.internal_uav_buffer(256)?;
    name(&buffer, || "indirect budget".to_owned());
    Some(buffer)
}

fn give_card_budget_buffer(buf: ID3D12Resource) {
    let overflow = {
        let mut spare = SPARE_CARD_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < SPARE_CARD_BUDGET_MAX {
            spare.push(BatchBuf(buf));
            None
        } else {
            Some(buf)
        }
    };
    drop(overflow);
}

fn take_indirect_buffer(needed: u64) -> Option<(ID3D12Resource, u64)> {
    let max_fit = needed.saturating_mul(4);
    {
        let mut pool = SPARE_INDIRECT.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((idx, _)) = pool
            .kept
            .iter()
            .enumerate()
            .filter(|(_, (_, cap))| *cap >= needed && *cap <= max_fit)
            .min_by_key(|(_, (_, cap))| *cap)
        {
            let (BatchBuf(buf), cap) = pool.kept.swap_remove(idx);
            pool.bytes -= cap;
            return Some((buf, cap));
        }
    }
    let buffer = host()?.internal_uav_buffer(needed)?;
    name(&buffer, || format!("indirect arguments {needed:#x}"));
    Some((buffer, needed))
}

fn give_indirect_buffer(buf: ID3D12Resource, cap: u64) {
    let overflow = {
        let mut pool = SPARE_INDIRECT.lock().unwrap_or_else(|e| e.into_inner());
        if pool.bytes.saturating_add(cap) <= SPARE_INDIRECT_MAX_BYTES {
            pool.bytes += cap;
            pool.kept.push((BatchBuf(buf), cap));
            None
        } else {
            Some(buf)
        }
    };
    drop(overflow);
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Cached `ID3D12GraphicsCommandList1` interface (avoids COM `QueryInterface` per depth-bounds call).
    list1: OnceLock<Option<ID3D12GraphicsCommandList1>>,
    /// Cached `ID3D12GraphicsCommandList6` interface (avoids COM `QueryInterface` per `DispatchMesh`).
    list6: OnceLock<Option<ID3D12GraphicsCommandList6>>,
    /// Cached `ID3D12GraphicsCommandList7` interface (for Enhanced Barriers).
    list7: OnceLock<Option<ID3D12GraphicsCommandList7>>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<ByCopy<TwinUse>>,
    /// True only while at least one entry in `twins` has `writing == true`,
    /// avoiding a `twins` mutex lock and scan on every `State::bind`.
    twins_writing: std::sync::atomic::AtomicBool,
    /// Format-aliased texture views read by this list (`crate::texture::alias`),
    /// keyed by the placed `HostTexture` pointer so cross-list copies run only
    /// when `source.write_gen` advanced since the last sync.
    aliases: Mutex<ByCopy<AliasUse>>,
    /// The list that runs those transitions just before this one, borrowed from
    /// [`IDLE_PRELUDES`] when needed and returned on [`HostList::reset`].
    prelude: Mutex<Option<TrackedPrelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// What its indirect executions ran, for the report of a removed device.
    #[cfg(feature = "diagnostics")]
    indirects: indirects::Shared,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps,
    /// pooled globally in [`SPARE_BUDGET`] across list resets/drops.
    budget: Mutex<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes, and
    /// pooled globally in [`SPARE_CARD_BUDGET`].
    card_budget: Mutex<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list and
    /// pooled globally in [`SPARE_INDIRECT`] across list resets/drops.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Scratch buffers of the batched clamps (`crate::indirect::batch`), and
    /// how many batches of this recording have a place in them.
    batch_args: Mutex<(Vec<ID3D12Resource>, u32)>,
    /// Refreshes of host copies that a thread of ours is still finding, and
    /// where among the list's steps they belong ([`aside`]).
    coming: Mutex<Option<(usize, std::sync::Arc<aside::Coming>)>>,
    /// Another number each time something changes that what a draw reads
    /// depends on ([`HostList::changed_something`]); no two lists have the
    /// same.
    epoch: std::sync::atomic::AtomicU64,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: std::sync::Arc<Mutex<Vec<Note>>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

impl Drop for HostList {
    fn drop(&mut self) {
        let indirect_clean = self
            .deferred
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .indirect_readable
            == 0;
        if let Some(budget) = self.budget.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            give_budget_buffer(budget);
        }
        if let Some(card_budget) = self.card_budget.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            give_card_budget_buffer(card_budget);
        }
        if let Some((buf, cap)) = self.indirect_args.get_mut().unwrap_or_else(|e| e.into_inner()).take()
            && indirect_clean
        {
            give_indirect_buffer(buf, cap);
        }
        let args = self.batch_args.get_mut().unwrap_or_else(|e| e.into_inner());
        let left = std::mem::take(&mut args.1);
        let buffers = std::mem::take(&mut args.0);
        if left == 0 {
            give_batch_buffers(buffers);
        }
    }
}

/// A list's prelude: an allocator and a closed list.
pub(crate) type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Planned state transitions, depth-twin syncs, and texture uploads for one
/// `HostList` prelude so the D3D12 command recording can run in parallel.
struct PreludePlan {
    syncs: Vec<(std::sync::Arc<crate::resource::HostTexture>, std::sync::Arc<crate::resource::Twin>, bool)>,
    alias_syncs: Vec<(
        std::sync::Arc<crate::resource::HostTexture>,
        std::sync::Arc<crate::resource::HostTexture>,
        std::sync::Arc<crate::texture::Through>,
    )>,
    aliasing: Vec<D3D12_RESOURCE_BARRIER>,
    to_copy: Vec<D3D12_RESOURCE_BARRIER>,
    barriers: Vec<D3D12_RESOURCE_BARRIER>,
    uploads: Vec<(crate::texture::PreludeUpload, crate::texture::UploadPlan)>,
}

// SAFETY: resources referenced by `D3D12_RESOURCE_BARRIER` stay alive in
// `syncs`, `uploads`, and `HostList::states` until `record_prelude` finishes.
unsafe impl Send for PreludePlan {}

struct PreludeJob {
    list: usize,
    slot: usize,
    plan: PreludePlan,
}

#[derive(Default)]
struct PreludeWorkerState {
    queue: std::collections::VecDeque<PreludeJob>,
    done: Vec<(usize, Option<ID3D12CommandList>)>,
    pending: usize,
}

// SAFETY: `ID3D12CommandList` is free-threaded once closed.
unsafe impl Send for PreludeWorkerState {}

struct PreludeWorker {
    state: Mutex<PreludeWorkerState>,
    work_cv: std::sync::Condvar,
    done_cv: std::sync::Condvar,
}

fn prelude_threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("VICTORY_PRELUDE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(2usize)
            .clamp(1, 16)
    })
}

fn prelude_worker() -> &'static PreludeWorker {
    static WORKER: OnceLock<PreludeWorker> = OnceLock::new();
    WORKER.get_or_init(|| {
        let worker = PreludeWorker {
            state: Mutex::default(),
            work_cv: std::sync::Condvar::new(),
            done_cv: std::sync::Condvar::new(),
        };
        for i in 0..prelude_threads() {
            let _ = std::thread::Builder::new()
                .name(format!("victory-prelude-{i}"))
                .spawn(|| {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
                    // SAFETY: matches `victory-submit` so prelude recording finishes
                    // during `flush + snap_card + wait_heaps` without preemption.
                    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
                    let w = prelude_worker();
                    loop {
                        let job = {
                            let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
                            while s.queue.is_empty() {
                                s = w.work_cv.wait(s).unwrap_or_else(|e| e.into_inner());
                            }
                            s.queue.pop_front().unwrap()
                        };
                        // SAFETY: `PreludeBatch` blocks before `&[&HostList]` drops and ensures
                        // distinct `HostList`s run concurrently.
                        let res = unsafe { &*(job.list as *const HostList) }.record_prelude(job.plan);
                        let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
                        s.done.push((job.slot, res));
                        s.pending -= 1;
                        if s.pending == 0 {
                            w.done_cv.notify_one();
                        }
                    }
                });
        }
        worker
    })
}

/// Overlaps `HostList::record_prelude` D3D12 command recording with the rest of
/// `Host::submit` (fills, snapshots, card sync, and descriptor heap sync).
#[derive(Default)]
struct PreludeBatch {
    seen: Vec<*const HostList>,
    active: bool,
}

impl PreludeBatch {
    fn push(
        &mut self,
        list: &HostList,
        slot: usize,
        plan: PreludePlan,
        lists: &mut [Option<ID3D12CommandList>],
    ) {
        if !wide_refresh() {
            lists[slot] = list.record_prelude(plan);
            return;
        }
        let raw = std::ptr::from_ref(list);
        if self.seen.contains(&raw) {
            self.finish(lists);
        }
        self.seen.push(raw);
        let w = prelude_worker();
        {
            let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
            s.queue.push_back(PreludeJob {
                list: raw as usize,
                slot,
                plan,
            });
            s.pending += 1;
        }
        self.active = true;
        w.work_cv.notify_one();
    }

    fn finish(&mut self, lists: &mut [Option<ID3D12CommandList>]) {
        if !self.active {
            return;
        }
        let w = prelude_worker();
        loop {
            let job = {
                let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
                s.queue.pop_front()
            };
            let Some(job) = job else {
                break;
            };
            // SAFETY: same lifetime guarantee as worker threads; `finish` owns `lists`.
            let res = unsafe { &*(job.list as *const HostList) }.record_prelude(job.plan);
            lists[job.slot] = res;
            let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
            s.pending -= 1;
        }
        let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.pending > 0 {
            s = w.done_cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        for (slot, res) in s.done.drain(..) {
            lists[slot] = res;
        }
        self.active = false;
        self.seen.clear();
    }
}

impl Drop for PreludeBatch {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let w = prelude_worker();
        let mut s = w.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.pending > 0 {
            s = w.done_cv.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.done.clear();
        self.active = false;
    }
}

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

/// `(first_state, last_state, local_writes, copy)` per `HostTexture` used by a list.
type States = ByCopy<(u32, u32, u32, std::sync::Arc<crate::resource::HostTexture>)>;

/// Whether a D3D12 resource state mask grants GPU write access to a texture.
const fn is_texture_write_state(state: u32) -> bool {
    const WRITE_STATES: u32 = 0x4 | 0x8 | 0x10 | 0x400 | 0x1000;
    state & WRITE_STATES != 0
}

/// What a list does with a format-aliased placed texture view
/// (`crate::texture::alias`). Mirrors [`TwinUse`]: if the list reads `placed`
/// before any local write to `source`, `plan_prelude` copies once in submission
/// order only when `source.write_gen != through.synced_gen`.
struct AliasUse {
    source: std::sync::Arc<crate::resource::HostTexture>,
    placed: std::sync::Arc<crate::resource::HostTexture>,
    through: std::sync::Arc<crate::texture::Through>,
    needs_prelude: bool,
    synced_local_write: u32,
}

/// Pure state step for [`HostList::alias_use`]: returns `true` if the list must
/// record an inline alias copy right now because `source` was written earlier
/// in this same list since the last inline sync.
fn alias_use_step(src_writes: u32, needs_prelude: &mut bool, synced_local_write: &mut u32) -> bool {
    if src_writes == 0 {
        *needs_prelude = true;
        false
    } else if src_writes > *synced_local_write {
        *synced_local_write = src_writes;
        true
    } else {
        false
    }
}

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED") && !dred_faults() && !dred_light())
}

/// Breadcrumbs and the faulting address, but no markers and no texts
/// (`VICTORY_D3D12_DRED=light`): which list a removed device stopped in, at
/// which of its commands, and the pipelines the list binds; formatting a text
/// for every command is what makes the whole of it slow.
pub(crate) fn dred_light() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "light")
    })
}

/// Any of the three.
pub(crate) fn dred_any() -> bool {
    dred_on() || dred_light() || dred_faults()
}

/// Whether only the faulting address of a removed device is kept
/// (`VICTORY_D3D12_DRED=faults`): no breadcrumbs, no markers and no texts, so
/// it costs next to nothing and can stay on while the title is played.
pub(crate) fn dred_faults() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "faults")
    })
}

/// The lists alive while device-removed data is on, by the number their name
/// ends in, with their marker texts and the pipelines they bind: a report made
/// on another thread than the one that submits has no lists of its own.
#[cfg(feature = "diagnostics")]
#[allow(clippy::type_complexity)]
static LIST_NOTES: Mutex<
    std::collections::BTreeMap<
        u32,
        (
            std::sync::Weak<Mutex<Vec<Note>>>,
            std::sync::Weak<Mutex<std::collections::BTreeSet<String>>>,
            std::sync::Weak<Mutex<indirects::Indirects>>,
        ),
    >,
> = Mutex::new(std::collections::BTreeMap::new());

/// The number in the name of a list of ours ("victory host list 7", which the
/// debug layer ends with a note of its own in brackets).
#[cfg(feature = "diagnostics")]
fn list_number(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("victory host list ")?;
    let digits = rest.split(' ').next()?;
    (rest.len() == digits.len() || rest[digits.len()..].starts_with(" (")).then(|| digits.parse().ok()).flatten()
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Indirect records that ran empty for want of their list's budget, and the
/// largest cost of one (told from 2^22 on), likewise.
static BUDGET_NOTES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
pub(crate) fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on), or for
/// the frame recorder (`VICTORY_RECORD`).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::book::on()
        && let Ok(resource) = object.cast::<ID3D12Resource>()
    {
        let text = text();
        crate::book::made(&resource, || text.clone());
        return name_it(object, || text);
    }
    name_it(object, text);
}

fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        if dred_any()
            && let Ok(object) = object.cast::<ID3D12Object>()
        {
            // SAFETY: a plain call on a live object.
            let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text)) };
        }
        return;
    }
    if dred_any()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping.
struct Chunk {
    buffer: ID3D12Resource,
    data: usize,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;
/// The same with [`keep_freed`].
const SPARE_KEPT: u64 = 1 << 30;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }

    /// Like [`give`], but caps large chunks (`> small_max`) to `most - small_reserve`
    /// so `4 MiB` chunks cannot fill the pool and evict every `256 KiB` chunk.
    fn give_reserved(&mut self, thing: T, size: u64, most: u64, small_max: u64, small_reserve: u64) -> Option<T> {
        if size > small_max {
            let large_bytes: u64 = self.kept.iter().filter(|(_, s)| *s > small_max).map(|(_, s)| *s).sum();
            if large_bytes + size > most.saturating_sub(small_reserve) {
                return Some(thing);
            }
        }
        self.give(thing, size, most)
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// Pooled `HostList::upload` buffers (power-of-two `4 KiB..=4 MiB`, 64 MiB cap).
static LIST_UPLOADS: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());
const LIST_UPLOAD_POOL_MAX: u64 = 64 << 20;
const LIST_UPLOAD_ITEM_MAX: u64 = 4 << 20;

fn take_list_upload(host: &Host, size: u64) -> Option<(Chunk, u64)> {
    let class = size.max(4096).next_power_of_two();
    if class <= LIST_UPLOAD_ITEM_MAX
        && let Some(chunk) = LIST_UPLOADS.lock().unwrap_or_else(|e| e.into_inner()).take(class)
    {
        return Some((chunk, class));
    }
    let alloc = if class <= LIST_UPLOAD_ITEM_MAX { class } else { size.max(1) };
    let (buffer, data) = if card_steps::stage() > 1 {
        host.internal_shared_buffer(alloc)?
    } else {
        host.upload_buffer(alloc)?
    };
    name(&buffer, || format!("list upload {alloc:#x}"));
    Some((
        Chunk {
            buffer,
            data: data as usize,
        },
        alloc,
    ))
}

fn give_list_uploads(uploads: Vec<(Chunk, u64)>) {
    if uploads.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut pool = LIST_UPLOADS.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in uploads {
            if size <= LIST_UPLOAD_ITEM_MAX {
                if let Some(excess) = pool.give_reserved(chunk, size, LIST_UPLOAD_POOL_MAX, 64 << 10, 16 << 20) {
                    freed.push(excess);
                }
            } else {
                freed.push(chunk);
            }
        }
    }
    drop(freed);
}

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        buffer,
        data: data as usize,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                let limit = if keep_freed() && !crate::residency::limited_host_ram() {
                    SPARE_KEPT
                } else {
                    SPARE_MOST
                };
                spare.give_reserved(chunk, size, limit, SCRATCH_FIRST, 32 << 20)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// Drops spare scratch chunks kept idle when host memory is pressured.
fn trim_spare_chunks() {
    let freed: Vec<(Chunk, u64)> = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.kept.is_empty() {
            return;
        }
        spare.bytes = 0;
        std::mem::take(&mut spare.kept)
    };
    let count = freed.len() as u64;
    let bytes: u64 = freed.iter().map(|(_, s)| *s).sum();
    ALIVE[1].fetch_sub(count, Ordering::Relaxed);
    ALIVE[2].fetch_sub(bytes, Ordering::Relaxed);
    ALIVE[3].fetch_sub(count, Ordering::Relaxed);
    ALIVE[4].fetch_sub(bytes, Ordering::Relaxed);
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<(Prelude, bool)>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<(Prelude, bool)>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// With `VICTORY_CARD_HASH_WIDE=alt` ([`card_hash_wide_now`]): the compare
    /// of [`CARD_COMPARE_WIDE`], `pso` being [`CARD_COMPARE`]'s. Both shaders
    /// declare the same root signature.
    pso_wide: Option<ID3D12PipelineState>,
    /// One dword per piece compared, odd where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Batched piece indices for disjoint re-hashes (`compare == 3`): PC
    /// memory, mapped (buffer, mapping, bytes).
    piece_indices: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// The dwords of `piece_indices` taken by plans still in flight.
    taken_indices: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one group per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, odd where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
///
/// A piece's 64 threads walk it side by side, each every 64th 16 bytes, and
/// one of them hashes their 64 hashes: 64 steps after one another. One thread
/// walked the piece before, 4,096 steps; the passes took 24.8 ms of a frame's
/// 82 on the card (capture replay1, 405 of them), most of it waiting for
/// those steps.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1), SRV(t1)"
cbuffer Span : register(b0) {
    uint first;    // contiguous: first piece; indexed (compare >= 3): byte offset in `piece_table`
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: rehash; 1: compare; 2: compare & store; 3: indexed rehash; 4: indexed compare; 5: indexed compare & store
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
ByteAddressBuffer piece_table : register(t1);
groupshared uint4 parts[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_GroupID, uint thread : SV_GroupIndex) {
    uint piece = (compare >= 3u) ? piece_table.Load(first + id.x * 4u) : (first + id.x);
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) + thread;
    uint at = lo + thread * 16u;
    // Eight loads at once: one after the other, each waits for memory.
    [loop]
    for (; at + 7168u < hi; at += 8192u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 1024u);
        uint4 v2 = now.Load4(at + 2048u);
        uint4 v3 = now.Load4(at + 3072u);
        uint4 v4 = now.Load4(at + 4096u);
        uint4 v5 = now.Load4(at + 5120u);
        uint4 v6 = now.Load4(at + 6144u);
        uint4 v7 = now.Load4(at + 7168u);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    parts[thread] = h;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 32u; s > 0u; s >>= 1u) {
        if (thread < s) {
            uint4 x = (parts[thread] ^ parts[thread + s]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            parts[thread] = x.yzwx ^ (x >> 15);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread != 0u) return;
    h = parts[0];
    if (compare == 1u || compare == 2u || compare == 4u || compare == 5u) {
        // The flag, and above it 31 bits of the hash (`torn`).
        uint folded = h.x ^ (h.y * 3u) ^ (h.z * 5u) ^ (h.w * 7u);
        differs.Store(out_at + id.x * 4u, (folded << 1u) | (any(h != hashes.Load4(piece * 16u)) ? 1u : 0u));
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u || compare == 5u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// Alternate 64-thread-per-piece card compare with sequential lane fold in
/// thread 0 (`VICTORY_CARD_HASH_WIDE=1|alt`; off by default: [`CARD_COMPARE`]'s
/// 8x-unrolled 6-step parallel tree reduction runs instead). Supports the same
/// root signature, indexed modes (`compare >= 3`), and `torn` flag bit layout.
pub(crate) const CARD_COMPARE_WIDE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1), SRV(t1)"
cbuffer Span : register(b0) {
    uint first;    // contiguous: first piece; indexed (compare >= 3): byte offset in `piece_table`
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: rehash; 1: compare; 2: compare & store; 3: indexed rehash; 4: indexed compare; 5: indexed compare & store
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
ByteAddressBuffer piece_table : register(t1);
groupshared uint4 lanes[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    uint i = group.x;
    if (i >= pieces) return;
    uint piece = (compare >= 3u) ? piece_table.Load(first + i * 4u) : (first + i);
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) ^ t;
    [loop]
    for (uint at = lo + t * 16u; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    lanes[t] = h;
    GroupMemoryBarrierWithGroupSync();
    if (t != 0u) return;
    uint4 all = uint4(0x27D4EB2Fu, 0x165667B1u, 0xD3A2646Cu, 0xFD7046C5u);
    [loop]
    for (uint k = 0u; k < 64u; k++) {
        all = (all ^ lanes[k]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        all = all.yzwx ^ (all >> 15);
    }
    if (compare == 1u || compare == 2u || compare == 4u || compare == 5u) {
        uint folded = all.x ^ (all.y * 3u) ^ (all.z * 5u) ^ (all.w * 7u);
        differs.Store(out_at + i * 4u, (folded << 1u) | (any(all != hashes.Load4(piece * 16u)) ? 1u : 0u));
        if (compare == 2u || compare == 5u) hashes.Store4(piece * 16u, all);
    } else {
        hashes.Store4(piece * 16u, all);
    }
}
"#;

/// `VICTORY_CARD_HASH_WIDE`: 0 default ([`CARD_COMPARE`] parallel tree), 1
/// [`CARD_COMPARE_WIDE`] (64-thread lane fold), 2 alternating every 20 s (`alt`).
fn card_hash_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_CARD_HASH_WIDE").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        if mode == 1 {
            tracing::info!("the card's hashes of copies' pieces use CARD_COMPARE_WIDE (VICTORY_CARD_HASH_WIDE=1)");
        }
        mode
    })
}

/// Whether [`CARD_COMPARE_WIDE`] is selected for the whole process (`VICTORY_CARD_HASH_WIDE=1`).
pub(crate) fn card_hash_wide() -> bool {
    card_hash_mode() == 1
}

/// Whether this submission uses [`CARD_COMPARE_WIDE`] (`1`: always; `alt`: 20 s on / 20 s off).
fn card_hash_wide_now() -> bool {
    match card_hash_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the card's hashes of copies' pieces from now (VICTORY_CARD_HASH_WIDE=alt): {}",
                    if on { "ON (CARD_COMPARE_WIDE)" } else { "off (CARD_COMPARE parallel tree)" }
                );
            }
            on
        }
    }
}

/// The compare's default HLSL source for this process ([`card_hash_wide`]).
pub(crate) fn card_compare_source() -> &'static str {
    if card_hash_wide() { CARD_COMPARE_WIDE } else { CARD_COMPARE }
}

impl CardWork {
    /// The compare pipeline for a submission ([`card_hash_wide_now`]).
    fn compare_for(&self, wide_now: bool) -> &ID3D12PipelineState {
        match (&self.pso_wide, wide_now) {
            (Some(wide), true) => wide,
            _ => &self.pso,
        }
    }
}

/// The hash [`CARD_COMPARE`] gives of a piece with its flag.
#[cfg(test)]
pub(crate) fn piece_hash(bytes: &[u8]) -> u32 {
    torn::hash(bytes)
}

/// One pass of [`CARD_COMPARE`] over a span's pieces, into `differs` from
/// dword `at` on (`mode` as the shader's `compare`): a group per piece, in
/// parts a dispatch can hold.
///
/// # Safety
/// The list is open with the compare's root signature, pipeline and buffers
/// set, and the pieces lie in the buffer.
unsafe fn card_pass(list: &ID3D12GraphicsCommandList, span: &crate::shadow::CardSpan, size: u32, at: u32, mode: u32) {
    const MOST: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    let mut done = 0;
    while done < span.pieces {
        let n = (span.pieces - done).min(MOST);
        let constants = [span.first + done, span.skew, size, n, (at + done) * 4, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
        }
        done += n;
    }
}

/// One indexed pass (`mode == 3`: rehash; `4`: compare; `5`: compare & store)
/// of [`CARD_COMPARE`] reading `count` piece indices from `t1` (`piece_table`)
/// starting at dword `idx_at`, and writing `differs` (`u1`) starting at dword
/// `out_at` (ignored for `mode == 3`).
///
/// # Safety
/// Same as [`card_pass`], with `t1` bound to the piece-index buffer.
unsafe fn card_pass_indexed(
    list: &ID3D12GraphicsCommandList,
    skew: u32,
    size: u32,
    count: u32,
    idx_at: u32,
    out_at: u32,
    mode: u32,
) -> u64 {
    const MOST: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    let mut done = 0;
    let mut passes = 0u64;
    while done < count {
        let n = (count - done).min(MOST);
        let constants = [(idx_at + done) * 4, skew, size, n, (out_at + done) * 4, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
        }
        done += n;
        passes += 1;
    }
    passes
}

/// Groups consecutive compared ranges `(shadow_key, mode, first_piece, pieces, at)`
/// into non-overlapping, contiguous-`at` batches that can run in a single indexed
/// `CARD_COMPARE` dispatch without two groups touching the same piece index.
fn batch_compare_groups(items: &[(usize, u32, u32, u32, u32)]) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    while start < items.len() {
        let (key, mode, first, pieces, at) = items[start];
        let mut end = start + 1;
        let mut prev_piece_end = first.saturating_add(pieces);
        let mut next_at = at.saturating_add(pieces);
        while end < items.len() {
            let (k2, m2, f2, p2, a2) = items[end];
            if k2 == key && m2 == mode && f2 >= prev_piece_end && a2 == next_at {
                prev_piece_end = f2.saturating_add(p2);
                next_at = a2.saturating_add(p2);
                end += 1;
            } else {
                break;
            }
        }
        groups.push(start..end);
        start = end;
    }
    groups
}

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists: a pass takes the time of one piece however few its pieces are.
/// Run 102 sent up 514 ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each. (Not with [`relay`], whose
/// landings do not come here: `crate::land` has threads of its own.)
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
    /// Looked at by `crate::picked`, and not declared: what is found changed
    /// there was written through a view nobody declared. Not landed.
    picked: bool,
}

impl CardPlan {
    /// `crate::picked` is told where what it looked at changed, by the
    /// compare's results.
    fn picked_found(&self, differs: &[u32]) {
        for r in self.ranges.iter().filter(|r| r.picked) {
            let changed = (0..r.span.pieces).find(|p| differs.get((r.at + p) as usize).is_some_and(|&d| torn::flagged(d)));
            if let Some(p) = changed {
                crate::picked::found(&r.shadow, r.shadow.card_piece(&r.span, p).0);
            }
        }
    }
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// Dirty uploads and lazy re-hashes, prepended ahead of the submission's
    /// lists in the same `ExecuteCommandLists`.
    sync: Option<ID3D12CommandList>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`]: the number of the send to copies in the card's memory
    /// that the submission ran behind ([`QUEUE_ORDER`]).
    sent_as: u64,
    /// With [`upcheck`]: the look at what it sent up.
    looks: Option<upcheck::Looks>,
    /// With [`send`]: the memory its sends come from.
    sends: Option<send::Memory>,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory, and how
    /// the card could bring the bytes instead.
    Fill(usize, usize, usize, Option<card_steps::FillTo>),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only, and the title memory it reads
    /// (from, to; empty for none).
    Host(Box<dyn Fn() + Send>, (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date and uploads it on the
    /// list's prelude when changed: once per submission, whichever lists
    /// sample it, until a `Copy` or `Run` intervenes.
    Texture(crate::texture::PreludeUpload),
    /// Reads a CPU-written linear movie plane into a per-list upload buffer.
    Plane((usize, usize), Box<dyn Fn() + Send>),
    /// A title buffer copy that the list does on the card ([`place`]): what
    /// is done on the CPU for it.
    Placed(place::Placed),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// Indices into `copies` of steps that are not `Step::Refresh`, in order,
    /// so `run_deferred` skips covered refreshes in O(non-refresh) time while
    /// `skip_front_refreshes && batch.front_intact` holds.
    non_refreshes: Vec<usize>,
    /// Total `Step::Refresh` entries and bytes in `copies` (diagnostics stats
    /// when `run_deferred` stays on the `non_refreshes` fast path).
    #[cfg(feature = "diagnostics")]
    refresh_count: u64,
    #[cfg(feature = "diagnostics")]
    refresh_bytes: u64,
    uploads: Vec<ID3D12Resource>,
    /// Standalone upload footprints kept through this list's fence (movie planes).
    plane_uploads: Vec<(Chunk, u64)>,
    /// Pooled per-list upload buffers (`HostList::upload`) returned to `LIST_UPLOADS` on drop.
    pooled_uploads: Vec<(Chunk, u64)>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Host buffer copies kept alive through execution (for resource-scoped
    /// buffer barriers, including any backing placed heap or card twin).
    kept_shadows: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The pipelines [`HostList::bind_pipeline`] kept, by address.
    bound: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Ranges declared written by shaders since the list last wrote title
    /// memory (host copy, address, bytes): declaring them again would
    /// change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Scratch memory [`HostList::fill_scratch`] fills since the list last
    /// wrote title memory (source, bytes): its host address.
    filled: std::collections::HashMap<(usize, usize), u64, std::hash::BuildHasherDefault<AddressHash>>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// True once at least one `Step::Placed` has been recorded in `copies`,
    /// allowing `forward_fill` to skip scanning `copies` in O(1) otherwise.
    has_placed: bool,
    /// Whether a clamp dispatch in this recording has already written the
    /// list's indirect-work budget (subsequent clamps order against it with a
    /// resource-scoped UAV barrier).
    budget_used: bool,
    /// Whether a global UAV barrier has already flushed all prior shader/copy
    /// UAV writes on this recording, with no intervening UAV-capable shader
    /// or buffer copy recorded since. Starts `false` on every fresh/reset list
    /// because an earlier list in the same `ExecuteCommandLists` may have
    /// unflushed UAV writes.
    uav_clean: bool,
    /// Whether a graphics draw that can store to buffers ran since the last
    /// global UAV sync on this recording (determines whether Enhanced pre-clamp
    /// barriers need `ALL_SHADING` instead of `COMPUTE_SHADING`).
    graphics_uav_dirty: bool,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
    /// Native DXR acceleration structure operations queued during recording
    /// and replayed on the submission thread (`crate::dxr`).
    dxr_ops: Vec<crate::dxr::DxrOp>,
}

impl Deferred {
    /// Clears all recorded state while retaining allocated collection capacities
    /// across `HostList::reset` calls (shrinking extreme outliers).
    fn clear(&mut self) {
        if !self.plane_uploads.is_empty() {
            give_chunks(std::mem::take(&mut self.plane_uploads));
        }
        if !self.pooled_uploads.is_empty() {
            give_list_uploads(std::mem::take(&mut self.pooled_uploads));
        }
        if !self.uploads.is_empty() || self.upload_bytes != 0 {
            ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
            ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
            self.uploads.clear();
            self.upload_bytes = 0;
        }
        self.copies.clear();
        if self.copies.capacity() > 16_384 {
            self.copies.shrink_to(4_096);
        }
        self.non_refreshes.clear();
        if self.non_refreshes.capacity() > 4_096 {
            self.non_refreshes.shrink_to(1_024);
        }
        #[cfg(feature = "diagnostics")]
        {
            self.refresh_count = 0;
            self.refresh_bytes = 0;
        }
        self.kept.clear();
        if self.kept.capacity() > 4_096 {
            self.kept.shrink_to(1_024);
        }
        self.kept_shadows.clear();
        if self.kept_shadows.capacity() > 2_048 {
            self.kept_shadows.shrink_to(512);
        }
        self.bound.clear();
        self.refreshed.clear();
        if self.refreshed.capacity() > 16_384 {
            self.refreshed.shrink_to(4_096);
        }
        self.gpu_written.clear();
        self.filled.clear();
        self.textures.clear();
        self.writes.clear();
        self.backs.clear();
        self.back_wanted.clear();
        self.texture_backs.clear();
        self.dxr_ops.clear();
        self.budget_zeroed = false;
        self.has_placed = false;
        self.budget_used = false;
        self.uav_clean = false;
        self.graphics_uav_dirty = false;
        self.indirect_readable = 0;
    }

    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }

    /// Appends a non-`Step::Refresh` step and records its index for `run_deferred`.
    fn push_step(&mut self, step: Step) {
        debug_assert!(!matches!(step, Step::Refresh(..)));
        let idx = self.copies.len();
        self.copies.push(step);
        self.non_refreshes.push(idx);
    }

    /// Appends a `Step::Refresh` step without adding it to `non_refreshes`.
    fn push_refresh(&mut self, shadow: std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        self.copies.push(Step::Refresh(shadow, address, len));
        #[cfg(feature = "diagnostics")]
        {
            self.refresh_count += 1;
            self.refresh_bytes += len as u64;
        }
    }

    /// The steps recorded last write title memory: what was brought up to
    /// date before them is not any more.
    fn title_written(&mut self) {
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
        // `place.rs` pushes `Step::Placed` directly into `copies` right before
        // calling `title_written()`; record its index in `non_refreshes` here.
        if let Some(Step::Placed(_)) = self.copies.last() {
            let idx = self.copies.len() - 1;
            if self.non_refreshes.last().copied() != Some(idx) {
                self.non_refreshes.push(idx);
            }
        }
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.plane_uploads));
        give_list_uploads(std::mem::take(&mut self.pooled_uploads));
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Whether the pieces the relay fetches from the card's memory come over the
/// card's copy queue (`VICTORY_RELAY_COPY=1`) and not over the queue that
/// draws.
///
/// On the drawing queue a fetch runs behind whatever was sent meanwhile,
/// and with the relay that is most of a frame: the thread that lands flights
/// stood 48 ms of a 120 ms frame waiting for seven fetches of 94 MB in all,
/// a few milliseconds of copying, and everything that waits for a landing
/// (the title's fences, reads of memory flights may write) waited with it.
///
/// What changes: the fetch runs when its flights are done and beside later
/// submissions, so of the sends of title bytes it is only known to have run
/// behind those its own flights ran behind (`CardPlan::sent_as`). Bytes sent
/// later are left out of the merge (`Shadow::card_merge_relay`): they are
/// the title's, newer than what the flights wrote. A later flight's bytes
/// may be in what is fetched, as before.
fn relay_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_RELAY_COPY") {
        Some(v) => v != "0",
        None => relay_exact(),
    })
}

/// True once `victory-relay` has created its dedicated Copy Queue, after which
/// `relay_card_back` never locks `QUEUE_ORDER` (ordering via `CardPlan::sent_as`).
static COPY_QUEUE_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range, in order: all
/// of them flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> Vec<(usize, u32)> {
    let mut pieces: Vec<(usize, u32)> = flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect();
    pieces.sort_unstable();
    pieces.dedup();
    pieces
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_pieces, landable};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    });
    if settle_room() { most * 2 } else { most }
}

/// Whether the flights get twice the room before "lists in the air"
/// settles them ([`late_most`]) and eight times the changed runs before
/// the merge under shaders' writes gives up and settles them
/// (`VICTORY_MERGE_RUNS`, [`stale_under_flights`]) (`VICTORY_SETTLE_ROOM=1`;
/// off by default: the caps as set; `alt`: twenty seconds with, twenty
/// without, in turn, each change said in the log).
fn settle_room() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_SETTLE_ROOM").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "flights get more room (lists in the air x2, changed runs merged x8) before they are settled from now (VICTORY_SETTLE_ROOM=alt): {}",
                    if on { "ON" } else { "off, the caps as set" }
                );
            }
            on
        }
    }
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// What was taken off it to be done on the card.
    own: Option<card_steps::Taken>,
    /// Title-memory ranges written when its shader textures are copied back.
    texture_back_ranges: Vec<(usize, usize)>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Native DXR pooled lists/scratch kept alive until the submission finishes.
    dxr: Option<crate::dxr::DxrSubmission>,
    /// Ahead-of-submission texture upload lists (`crate::heap_worker`) kept
    /// alive with the flight until recycled.
    ahead: Vec<HostList>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// The list of ours that ran ahead of them ([`card_steps`]).
    own: Option<card_steps::Taken>,
    texture_back_ranges: Vec<(usize, usize)>,
    dxr: Option<crate::dxr::DxrSubmission>,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders and texture readbacks may write
    /// (address, end), in order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`relay`]: the fence signals of the flights being landed
    /// (taken out of `line`, not signalled yet), for [`late_signalled`].
    in_hand_signals: Vec<(usize, u64)>,
    /// With [`card_steps`]: what the flights being landed write.
    landing: Vec<card_steps::Writer>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    in_hand_signals: Vec::new(),
    landing: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
#[inline]
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to || ranges.is_empty() {
        return false;
    }
    if to <= ranges[0].0 || from >= ranges[ranges.len() - 1].1 {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let (written, copies) = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        (line.written.clone(), card_steps::copies(&line))
    };
    blocked_by(&written, runs, true).or_else(|| card_steps::copied_read(&copies, runs).then_some(WHY_READ))
}

/// Whether a small fill of upload memory, or copy of a title buffer, goes on without waiting for
/// flights when the title itself wrote the memory since they were given it
/// (`VICTORY_FILL_TITLE=1`; off by default).
///
/// The title's 174 MB heap is declared written as a whole (a root view has
/// no size), and the title keeps a ring of constants in it: one fill of 480
/// bytes a frame, at a new place each time, waited for every flight's
/// copy-back, some 30 ms of a 130 ms frame (run blame1: 38 of the 40 waits
/// named). The title writes those bytes just before the draw that reads
/// them, which takes the block's guard off ([`crate::watch::written_since_taken`]).
///
/// Or no shader's write went into that page in the last 600 frames
/// ([`crate::watch::landed_lately`]): the first alone left two thirds of
/// the wait, for constants at the end of a 64 KiB page that the title wrote
/// before the page was taken.
///
/// What can go wrong: a constant a flight's shader really writes, in a
/// block that the title (or an earlier flight's copy-back) also wrote, or
/// for the first time, is read as it was before that flight.
fn fill_is_titles(src: usize, len: usize) -> bool {
    const SMALL: usize = 64 << 10;
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_TITLE").is_some_and(|v| v == "1"));
    on && len <= SMALL && (crate::watch::written_since_taken(src, len) || !crate::watch::landed_lately(src, len, 600))
}

/// Diagnostics: the waits for flights, by what waited, why it was not done
/// on the card ([`card_steps`]), its bytes, and the declared write it lies
/// in (start, bytes).
#[cfg(feature = "diagnostics")]
type Blamed = (&'static str, &'static str, usize, usize, usize);
/// How often and how long, and the address the last one read.
#[cfg(feature = "diagnostics")]
static BLAME: Mutex<Option<std::collections::HashMap<Blamed, (u64, u64, usize)>>> = Mutex::new(None);
/// What the wait that comes next is for, and why not on the card.
#[cfg(feature = "diagnostics")]
static BLAMED: Mutex<(Option<(Blamed, usize)>, &'static str)> = Mutex::new((None, ""));

/// Diagnostics: shader writes declared over a megabyte or more: the range
/// (from, to) and what declares it ([`wide_write`]): how often, and its
/// shaders.
#[cfg(feature = "diagnostics")]
static WIDE_WRITES: Mutex<Option<std::collections::HashMap<(usize, usize, usize), (u64, String)>>> = Mutex::new(None);

/// Diagnostics: a draw or dispatch declares that it writes `len` bytes at
/// `address`, through a root view of `pipeline`, a view in a table (0) or a
/// raw address (1).
#[cfg(feature = "diagnostics")]
pub(crate) fn wide_write(address: u64, len: u64, pipeline: usize) {
    if len < 1 << 20 {
        return;
    }
    let mut all = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let e = all
        .get_or_insert_with(Default::default)
        .entry((address as usize, (address + len) as usize, pipeline))
        .or_insert_with(|| {
            let shaders = match pipeline {
                0 => "a view in a table".into(),
                1 => "a raw address".into(),
                p => crate::pipeline::host_pipeline(p as crate::object::This).map_or_else(String::new, |p| {
                    p.stages.iter().map(|s| s.shader.hash.to_string()).collect::<Vec<_>>().join("+")
                }),
            };
            (0, shaders)
        });
    e.0 += 1;
}

#[cfg(feature = "diagnostics")]
fn blame(kind: &'static str, written: &[(usize, usize)], at: usize, len: usize) {
    let i = written.partition_point(|r| r.1 <= at);
    let (from, to) = written.get(i).copied().unwrap_or((0, 0));
    let mut blamed = BLAMED.lock().unwrap_or_else(|e| e.into_inner());
    let why = blamed.1;
    blamed.0 = Some(((kind, why, len, from, to - from), at));
}

/// Why [`card_steps`] left the submission to the wait that comes next.
#[cfg(feature = "diagnostics")]
fn refused(why: &'static str) {
    *BLAMED.lock().unwrap_or_else(|e| e.into_inner()) = (None, why);
}

/// The wait that comes next is for these ranges ([`stale_under_flights`]).
#[cfg(feature = "diagnostics")]
fn blame_stale(stale: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]) {
    if let Some((shadow, from, _)) = stale.first() {
        let (start, len) = shadow.span();
        let key = ("title memory changed under shader writes", crate::shadow::stale_why(), 0, start as usize, len as usize);
        BLAMED.lock().unwrap_or_else(|e| e.into_inner()).0 = Some((key, *from));
    }
}

/// The wait for what [`blame`] last named, or a wait of another reason than
/// title memory read, took `us`.
#[cfg(feature = "diagnostics")]
fn blamed(why: usize, us: u64) {
    let (named, refused) = std::mem::replace(&mut *BLAMED.lock().unwrap_or_else(|e| e.into_inner()), (None, ""));
    let (key, at) = match (why, named) {
        (WHY_READ | WHY_STALE, Some(named)) => named,
        (WHY_READ, None) => (("a host copy of what a copy done on the card wrote", refused, 0, 0, 0), 0),
        _ => {
            let kind = match why {
                WHY_SYNC => "a synchronous submission, a present or the sweep",
                WHY_VALUE => "a memory value",
                WHY_STALE => "title memory changed under shader writes",
                WHY_RUN => "a step of ours",
                WHY_MOST => "lists in the air",
                WHY_FENCE => "a blocking fence wait",
                _ => "another reason",
            };
            ((kind, "", 0, 0, 0), 0)
        }
    };
    let mut all = BLAME.lock().unwrap_or_else(|e| e.into_inner());
    let e = all.get_or_insert_with(Default::default).entry(key).or_default();
    *e = (e.0 + 1, e.1 + us, at);
}

/// For the frame-rate line: the five that waited longest.
#[cfg(feature = "diagnostics")]
fn log_blame(frames: f64) {
    let all = BLAME.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let wide = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let mut all: Vec<_> = all.into_iter().collect();
    all.sort_unstable_by_key(|(_, (_, us, _))| std::cmp::Reverse(*us));
    for ((kind, why, len, from, size), (n, us, at)) in all.into_iter().take(5) {
        // The passes that declare it written, the ones that do so most.
        let mut passes: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for ((a, b, _), (n, shaders)) in &wide {
            if *a < at + len && at < *b {
                *passes.entry(shaders).or_default() += n;
            }
        }
        let mut passes: Vec<(&str, u64)> = passes.into_iter().collect();
        passes.sort_unstable_by_key(|p| std::cmp::Reverse(p.1));
        let passes: Vec<String> =
            passes.into_iter().take(5).map(|(shaders, n)| format!("{shaders} {:.1}x", n as f64 / frames)).collect();
        tracing::info!(
            "waited for flights, per frame: {:.1} ms in {:.2} waits for {kind} +{len:#x} (last at {at:#x}), in shader writes declared \
             over {from:#x}+{size:#x} (declared per frame over a megabyte or more there, by shader: {}){}{why}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            passes.join(", "),
            match (why.is_empty(), kind) {
                (true, _) => "",
                (_, "title memory changed under shader writes") => "; not put in under them: ",
                _ => "; not done on the card: ",
            },
        );
    }
}

/// Whether flights that write `written` stop a read of `len` bytes at `src`.
fn read_blocked(written: &[(usize, usize)], src: usize, len: usize) -> bool {
    touches(written, src, src + len) && !fill_is_titles(src, len)
}

/// Whether they stop a title buffer copy, and if so whether by its source.
fn copy_blocked(written: &[(usize, usize)], dst: usize, src: usize, len: usize) -> Option<bool> {
    let read = read_blocked(written, src, len);
    (read || touches(written, dst, dst + len)).then_some(read)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    // Destinations of copies at their place, if a step reads them.
    let mut placed: Vec<(usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if let Some(_read) = copy_blocked(written, dst, src, len) {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            if _read {
                                blame("a buffer copy's source", written, src, len);
                            } else {
                                blame("a buffer copy's destination", written, dst, len);
                            }
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len, _) => {
                    if read_blocked(written, src, len) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a fill of upload memory", written, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(upload) => {
                    let (at, len) = upload.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a texture", written, at, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Plane((at, len), _) => {
                    if touches(written, at, at.saturating_add(len)) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a movie plane", written, at, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Run(_) => return Some(WHY_RUN),
                Step::Placed(p) => match p.waits(written) {
                    place::Waits::No => {}
                    place::Waits::ForReaders => placed.push((p.dst, p.dst + p.len)),
                    place::Waits::Yes => {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a buffer copy at its place", written, p.src, p.len);
                        }
                        return Some(WHY_READ);
                    }
                },
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => {}
            }
        }
    }
    (!placed.is_empty() && place::read_by(runs, placed)).then_some(WHY_READ)
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    let mut places: ByCopy<usize> = ByCopy::default();
    let mut last: Option<(usize, usize)> = None;
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step
                && shadow.writes_pending()
            {
                let key = Arc::as_ptr(shadow) as usize;
                let end = *address + *len;
                let i = match last {
                    Some((lk, li)) if lk == key => li,
                    _ => {
                        let i = *places.entry(key).or_insert_with(|| {
                            spans.push((shadow.clone(), *address, end));
                            spans.len() - 1
                        });
                        last = Some((key, i));
                        i
                    }
                };
                let s = &mut spans[i];
                (s.1, s.2) = (s.1.min(*address), s.2.max(end));
            }
        }
    }
    stale_under_flights_spans(spans)
}

fn stale_under_flights_spans(
    mut spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    if spans.is_empty() {
        return spans;
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there. With [`card_steps`] the pages of the
        // title's ring of constants come here too, which were waited for
        // before: hundreds of short runs each, at about 1.5 us a run.
        static RUNS: OnceLock<usize> = OnceLock::new();
        let mut runs_left = *RUNS.get_or_init(|| {
            std::env::var("VICTORY_MERGE_RUNS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .map_or_else(|| if card_steps::stage() > 0 { 4096 } else { 512 }, |n| n.clamp(1, 1 << 16))
        });
        if settle_room() {
            runs_left = (runs_left * 8).min(1 << 16);
        }
        spans.retain(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
        return spans;
    }
    spans.retain(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from));
    spans
}

/// How many parts of submissions, and lists, the line's thread gives to the
/// host as one (`VICTORY_JOIN_PARTS=<n>`; off by default, 1; `VICTORY_JOIN_LISTS`,
/// 48).
///
/// The idea is fxceless/victory-fork's, commit 39731795 (GPL-3.0-or-later, as
/// this): the title's lists wait for memory values, which cuts a frame into
/// some 29 parts, each prepared on its own: the write watch asked, the heaps'
/// records, the title's bytes under the flights' writes, a list that sends
/// what changed and one that compares. (Run submit-1, the dorm, 12 of 28 parts
/// joined, eight at most: the thread works 51.4 ms a frame instead of 54.8, the
/// frame is 0.6 to 3 ms shorter.) What differs here: the parts are taken
/// off one queue's entries of the line (`crate::queue`), still go to the queue
/// in calls of their own, and everything they refresh is brought up to date in
/// front of the first ([`refresh_front`], without which a later part's title
/// bytes under an earlier part's declared writes would go missing).
pub(crate) fn joins() -> (usize, usize) {
    if ahead_of_writes() { joins_asked() } else { (1, 1) }
}

fn joins_asked() -> (usize, usize) {
    static MOST: OnceLock<(usize, usize)> = OnceLock::new();
    let of = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let (parts, lists) = *MOST.get_or_init(|| (of("VICTORY_JOIN_PARTS").unwrap_or(1).max(1), of("VICTORY_JOIN_LISTS").unwrap_or(48).max(1)));
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_PARTS") {
        return (if on { 8 } else { 1 }, lists);
    }
    (parts, lists)
}

/// Whether the part `next` can be prepared together with `ahead`, which goes
/// before it and has not been sent: nothing done on the CPU for it reads what
/// those give their shaders, write back or write when they are done, and it
/// changes no title memory that they read ([`Step::Copy`], [`Step::Run`]).
pub(crate) fn joinable<'a>(ahead: impl Iterator<Item = &'a HostList>, next: &[&HostList]) -> bool {
    if next
        .iter()
        .any(|l| l.has_coming() || l.deferred().copies.iter().any(|s| matches!(s, Step::Copy(..) | Step::Run(_))))
    {
        return false;
    }
    let ahead: Vec<&HostList> = ahead.collect();
    let mut written: Vec<(usize, usize)> = declared_writes(&ahead).into_iter().flat_map(|(_, ranges)| ranges).collect();
    for list in &ahead {
        let deferred = list.deferred();
        written.extend(deferred.writes.iter().map(|&(at, _, bytes)| (at, at + bytes)));
        written.extend(deferred.backs.iter().map(|&(_, at, [row, rows, pitch])| (at, at + rows.saturating_sub(1) * pitch + row)));
    }
    let mut joined = Vec::new();
    join_ranges(&mut joined, written);
    blocked_by(&joined, next, false).is_none()
}

/// Whether a bottom-of-pipe write of `list` meets `wait`.
pub(crate) fn will_meet(list: &HostList, wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    // The last write there says.
    let deferred = list.deferred();
    let last = deferred.writes.iter().rev().find(|(at, _, n)| *at < address + bytes && address < *at + *n);
    matches!(last, Some((at, crate::queue::Written::Value(v), n)) if *at == address && *n >= bytes && wait.holds_for(*v))
}

/// Lists and the order they run in.
type Run<T> = (Vec<T>, Vec<usize>);

/// `order`'s first `cut` places and the rest, each with the lists they name;
/// as it was if a list is named on both sides.
fn split<T>(lists: Vec<T>, order: Vec<usize>, cut: usize) -> Result<[Run<T>; 2], Run<T>> {
    let (first, second) = order.split_at(cut);
    if first.is_empty() || second.is_empty() || first.iter().any(|i| second.contains(i)) {
        return Err((lists, order));
    }
    let behind: Vec<bool> = (0..lists.len()).map(|i| second.contains(&i)).collect();
    let place = |i: usize| behind[..i].iter().filter(|&&b| b == behind[i]).count();
    let orders = [first, second].map(|half| half.iter().map(|&i| place(i)).collect::<Vec<_>>());
    let mut halves: [Vec<T>; 2] = [Vec::new(), Vec::new()];
    for (list, behind) in lists.into_iter().zip(&behind) {
        halves[usize::from(*behind)].push(list);
    }
    let [a, b] = halves;
    let [x, y] = orders;
    Ok([(a, x), (b, y)])
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// Whether a `Late::Signal(fence, v)` with `v >= value` is already queued
/// behind flights submitted to the single host queue (or currently being
/// landed before `fence::set`). Safe for GPU-to-GPU queue waits (`Job::Wait`)
/// because the single host D3D12 queue preserves strict FIFO order and
/// `line.written` still guards CPU reads of shader-written memory.
pub(crate) fn late_signalled(fence: usize, value: u64) -> bool {
    let line = line();
    line.in_hand_signals.iter().any(|&(f, v)| f == fence && v >= value)
        || line.line.iter().any(|item| matches!(item, Late::Signal(f, v) if *f == fence && *v >= value))
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    if let Some(met) = late_met_known(wait) {
        return met;
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without a wait for the card; `None`: it has to be waited for.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> Option<bool> {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return Some(false);
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return Some(true);
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return Some(false);
        }
    }
    None
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let lag = land_lag(
        bury_lag() as usize,
        process_private_bytes(),
        crate::residency::pressured(),
        crate::residency::critical_commit(),
    );
    let landed: Vec<_> = {
        let mut line = line();
        let n = line.landed.len().saturating_sub(lag);
        line.landed.drain(..n).collect()
    };
    if !landed.is_empty() {
        crate::list::recycle_many(landed);
    }
}

/// Reclaims any fence-completed flights past `land_lag` so `list::fresh_list`
/// can refill the idle pool before allocating a new command allocator.
pub(crate) fn reclaim_landed() {
    land();
}

fn fence_reached(completed: u64, target: u64) -> bool {
    completed != u64::MAX && completed >= target
}

/// Waits for `fence` to reach `target` and rejects device-removed (`u64::MAX`).
/// The event is private to the waiting thread and the fence remains alive.
unsafe fn wait_host_fence(fence: &ID3D12Fence, target: u64, event: HANDLE, context: &str) {
    // SAFETY: caller owns the event and fence for the entire wait.
    if let Err(e) = unsafe { fence.SetEventOnCompletion(target, event) } {
        fail(format_args!("{context}: SetEventOnCompletion({target}) failed: {e}"));
    }
    let status = unsafe { WaitForSingleObject(event, INFINITE) };
    if status != windows::Win32::Foundation::WAIT_OBJECT_0 {
        fail(format_args!("{context}: fence wait failed: {status:?}"));
    }
    let completed = unsafe { fence.GetCompletedValue() };
    if !fence_reached(completed, target) {
        fail(format_args!("{context}: fence {completed} did not reach {target}"));
    }
}

#[cfg(test)]
#[test]
fn host_fence_completion_rejects_removed_device() {
    assert!(!fence_reached(u64::MAX, 1));
    assert!(!fence_reached(6, 7));
    assert!(fence_reached(7, 7));
    assert!(fence_reached(8, 7));
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    crate::cores::keep_on_fast_cores("victory-late-wait");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            let mut helping = crate::timeline::span("helper: landed lists reset and given back");
            land();
            helping.next("helper: looking at what queues hold");
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            #[cfg(feature = "diagnostics")]
            let _card = crate::timeline::card_wait(Some(target));
            wait_host_fence(&host.fence, target, event, "late helper");
        }
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    let _waiting = crate::timeline::span(match why {
        WHY_SYNC => "waiting for landings: everything (a present, the sweep)",
        WHY_VALUE => "waiting for landings: a memory value",
        WHY_STALE => "waiting for landings: title memory changed under shader writes",
        WHY_READ => "waiting for landings: title memory read",
        WHY_MOST => "waiting for landings: lists in the air",
        WHY_FENCE => "waiting for landings: a blocking fence wait",
        _ => "waiting for landings: another reason",
    });
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    blamed(why, us);
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    /// With [`relay_copy`]: the card's copy queue, which `list` is for.
    queue: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// Pieces caught while a later flight was writing them.
    suspects: torn::Suspects,
    /// What earlier fetches brought of flights not landed yet.
    had: crate::land::Had,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    crate::cores::keep_on_fast_cores("victory-relay");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                if completed == u64::MAX {
                    drop(line);
                    fail(format_args!("relay fence reported device removal"));
                }
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                line.in_hand_signals = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Signal(fence, value) => Some((*fence, *value)),
                        _ => None,
                    })
                    .collect();
                if card_steps::stage() > 0 {
                    line.landing = card_steps::writers(&items);
                }
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(Some(target));
                // SAFETY: the host's fence and this thread's event.
                unsafe { wait_host_fence(&host.fence, target, event, "relay") };
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards). In two steps:
/// what is to be looked through and what of it differs ([`RelayBack::look`],
/// which needs nothing that the fetch from the card's memory brings), and
/// the merge.
struct RelayBack {
    work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    found: Option<Vec<Vec<(usize, usize)>>>,
}

fn relay_hand_back(back: &[Declared]) -> RelayBack {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() || shadow.in_place() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    RelayBack { work, found: None }
}

impl RelayBack {
    /// Looks the ranges through, once; its time counts as `stat`.
    fn look(&mut self, stat: usize) {
        if self.work.is_empty() || self.found.is_some() {
            return;
        }
        let scanning = std::time::Instant::now();
        // Looked through without a lock: a piece that changes meanwhile (a
        // later flight writing, the title's bytes going in) is found now or
        // by that flight's landing, and the merge looks again under the
        // lock.
        self.found = Some(crate::shadow::find_all(&self.work));
        crate::land::spent(stat, scanning);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    }

    /// What differs goes into title memory.
    fn merge(mut self) {
        use crate::land;
        self.look(land::stat::SCAN);
        let Some(found) = self.found else {
            return;
        };
        let merging = std::time::Instant::now();
        let mut written: Vec<u64> = Vec::new();
        if land::on() {
            // Pieces in the same block of title memory in one job, in order.
            let pieces: Vec<(usize, usize, usize)> =
                found.iter().enumerate().flat_map(|(i, found)| found.iter().map(move |&(at, n)| (i, at, n))).collect();
            let start = |i: usize| self.work[i].0.span().0 as usize;
            let mut order: Vec<usize> = (0..pieces.len()).collect();
            order.sort_by_key(|&p| (start(pieces[p].0) + pieces[p].1) / crate::land::BLOCK);
            let cut: Vec<(usize, usize)> = order.iter().map(|&p| (start(pieces[p].0) + pieces[p].1, pieces[p].2)).collect();
            let jobs = land::jobs(&cut);
            let put: Vec<std::sync::atomic::AtomicU64> = self.work.iter().map(|_| Default::default()).collect();
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                let mut notes = land::Notes::default();
                for &p in &order[jobs[j].clone()] {
                    let (i, at, n) = pieces[p];
                    put[i].fetch_add(self.work[i].0.land_back(at, n, &mut notes), Ordering::Relaxed);
                }
                notes.flush();
                land::spent(land::stat::BACK_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written.extend(put.iter().map(|p| p.load(Ordering::Relaxed)));
        } else {
            let (mut now, mut title) = (Vec::new(), Vec::new());
            for ((shadow, ..), found) in self.work.iter().zip(found) {
                let merge = |(at, n)| shadow.relay_merge(at, n, &mut now, &mut title) as u64;
                written.push(found.into_iter().map(merge).sum());
            }
        }
        land::spent(land::stat::BACK, merging);
        for ((_, from, to), written) in self.work.iter().zip(written) {
            COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written as usize);
            let _ = (from, to);
        }
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame (VICTORY_RELAY=1): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it)",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if send::own() {
            let t: Vec<f64> = send::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "bytes of submissions' own sent up per frame (VICTORY_SEND_OWN): {:.1} MB in {:.0} runs, {:.0} words in scatters; {:.2} ms \
                 taking them out of the CPU sides",
                t[0] / (1 << 20) as f64,
                t[1],
                t[2],
                t[3] / 1000.0,
            );
        }
        let t: Vec<f64> = torn::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        if t[0] > 0.0 {
            tracing::info!(
                "fetched pieces checked per frame (VICTORY_FETCH_CHECK): {:.1}, of which {:.2} were not what their flight ended                  with (caught while a later flight was at them, or title bytes went up meanwhile); {:.2} were found whole when                  fetched again",
                t[0],
                t[1],
                t[2],
            );
        }
    }
    crate::land::log(frames);
    if place::on() {
        let t: Vec<f64> = place::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "title buffer copies at their place in the lists, per frame (VICTORY_COPIES_IN_PLACE): {:.1} of {:.1} KB, {:.1} of them out              of what shaders wrote; {:.2} copies into textures out of what shaders wrote",
            t[0],
            t[1] / 1024.0,
            t[2],
            t[3],
        );
    }
    card_steps::log(frames);
    crate::port::log(frames);
    crate::watch::port_guard_line(frames);
    crate::port_journal::line(frames);
    crate::port_rt::line(frames);
    crate::port_cbv::log(frames);
    crate::picked::log(frames);
    #[cfg(feature = "diagnostics")]
    log_blame(frames);
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
    }
}

pub(crate) mod aside;
pub(crate) mod awake;
mod card_steps;
pub(crate) mod indirects;
mod panel;
mod place;
pub(crate) mod send;
pub(crate) use send::{goes_as_words, words_under};
#[cfg(feature = "diagnostics")]
pub(crate) mod spans;
mod torn;
#[cfg(feature = "diagnostics")]
mod traffic;
pub(crate) mod upcheck;
#[cfg(feature = "diagnostics")]
mod stamps;
#[cfg(feature = "diagnostics")]
pub(crate) use stamps::{timeline_ready, timeline_restart, timeline_stamps};

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// See [`HostList::defer_gpu_write`].
fn declare_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DECLARE_ONCE").map_or(true, |v| v != "0"))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    // Where each copy is in it, and the ranges seen: dispatches declare the
    // same ranges thousands of times a frame.
    let mut places: ByCopy<usize> = ByCopy::default();
    let mut seen = std::collections::HashSet::<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>::default();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                if !seen.insert((key, *address, *len)) {
                    continue;
                }
                let i = *places.entry(key).or_insert_with(|| {
                    back.push((shadow.clone(), Vec::new()));
                    back.len() - 1
                });
                back[i].1.push((*address, *address + *len));
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    // Textures shaders wrote that the title reads on the CPU
    // ([`crate::texback`]).
    for back in &list.deferred().texture_backs {
        crate::texback::land(back);
    }
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() || shadow.in_place() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() || shadow.in_place() {
            // Done before (`Host::card_back`), or nothing to do (in place).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// Whether the host GPU supports DXR Tier 1.1 (inline raytracing / `TraceRayInline`)
    /// and exposes `ID3D12Device5`; logged once.
    pub(crate) fn raytracing_tier_1_1(&self) -> bool {
        static TIER11: OnceLock<bool> = OnceLock::new();
        *TIER11.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS5,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let has_device5 = self.device.cast::<ID3D12Device5>().is_ok();
            let supported = ok
                && options.RaytracingTier.0 >= D3D12_RAYTRACING_TIER_1_1.0
                && has_device5;
            tracing::info!(
                supported,
                tier = options.RaytracingTier.0,
                has_device5,
                "host DXR Tier 1.1 support (VICTORY_NATIVE_RT)"
            );
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    pub(crate) fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    /// An allocator and an open (ready-to-record) list for a prelude.
    fn make_prelude_open(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    /// Takes a fence-completed prelude from [`IDLE_PRELUDES`] or creates a new one.
    fn take_prelude(&self) -> Option<TrackedPrelude> {
        let pooled = {
            let mut pool = IDLE_PRELUDES.lock().unwrap_or_else(|e| e.into_inner());
            let taken = pool.pop();
            if taken.is_some() {
                PRELUDES_POOLED.store(pool.len() as u64, Ordering::Relaxed);
            }
            taken
        };
        pooled.or_else(|| self.make_prelude_open().map(|p| TrackedPrelude::new(p, true)))
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = Mutex::new(if !lazy_prelude() { self.take_prelude() } else { None });
            #[cfg(feature = "diagnostics")]
            let notes = std::sync::Arc::<Mutex<Vec<Note>>>::default();
            let pipelines = std::sync::Arc::<Mutex<std::collections::BTreeSet<String>>>::default();
            #[cfg(feature = "diagnostics")]
            let indirects = indirects::Shared::default();
            #[cfg(feature = "diagnostics")]
            if dred_on() || dred_light() {
                static NUMBER: AtomicU32 = AtomicU32::new(0);
                let number = NUMBER.fetch_add(1, Ordering::Relaxed);
                name(&list, || format!("victory host list {number}"));
                let mut all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                if number.is_multiple_of(256) {
                    all.retain(|_, l| l.0.strong_count() > 0);
                }
                all.insert(
                    number,
                    (
                        std::sync::Arc::downgrade(&notes),
                        std::sync::Arc::downgrade(&pipelines),
                        std::sync::Arc::downgrade(&indirects),
                    ),
                );
            }
            Ok(HostList {
                allocator,
                list,
                list1: OnceLock::new(),
                list6: OnceLock::new(),
                list7: OnceLock::new(),
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                twins_writing: std::sync::atomic::AtomicBool::new(false),
                aliases: Mutex::default(),
                prelude,
                pipelines,
                #[cfg(feature = "diagnostics")]
                indirects,
                budget: Mutex::default(),
                card_budget: Mutex::default(),
                indirect_args: Mutex::default(),
                batch_args: Mutex::default(),
                coming: Mutex::default(),
                epoch: std::sync::atomic::AtomicU64::new(next_epoch()),
                _alive: Counted::new(),
                #[cfg(feature = "diagnostics")]
                notes,
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            let source = card_compare_source();
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // Four megabytes of results: 1,048,576 pieces across in-flight
        // submissions, avoiding mid-frame `WHY_MOST` settles.
        const RESULTS: u64 = 4 << 20;
        const PIECE_INDICES: u64 = 1 << 20;
        let (differs, data) = self.internal_shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        let (piece_indices, idx_data) = self.internal_shared_buffer(PIECE_INDICES).ok_or_else(windows::core::Error::empty)?;
        name(&piece_indices, || "card compare rehash piece indices".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            #[cfg(feature = "diagnostics")]
            name(&pso, || crate::replay::play::COMPARE.into());
            let pso_wide = (card_hash_mode() == 2).then(|| self.make_card_compare_wide(&root)).flatten();
            Ok(CardWork {
                sync: vec![(list()?, false)],
                check: vec![(list()?, false)],
                fetch: list()?,
                root,
                pso,
                pso_wide,
                differs: (differs, data as usize, RESULTS),
                piece_indices: (piece_indices, idx_data as usize, PIECE_INDICES),
                fetched: None,
                taken: 0,
                taken_indices: 0,
                out: 0,
            })
        }
    }

    /// [`CARD_COMPARE_WIDE`]'s pipeline with the compare's root signature `root`,
    /// for `VICTORY_CARD_HASH_WIDE=alt`; `None` if compilation or creation fails.
    fn make_card_compare_wide(&self, root: &ID3D12RootSignature) -> Option<ID3D12PipelineState> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<ID3D12PipelineState> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_COMPARE_WIDE.as_ptr().cast(),
                    CARD_COMPARE_WIDE.len(),
                    windows::core::s!("card compare wide"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            // SAFETY: the blob's own bytes; the root signature matches the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
            }
        })();
        match made {
            Ok(pso) => Some(pso),
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_CARD_HASH_WIDE=alt: CARD_COMPARE_WIDE could not be made; keeping CARD_COMPARE");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    /// `picked`: ranges that may be written too, and ranges to look at
    /// (`crate::picked`).
    fn card_before(
        &self,
        declared: &[Declared],
        picked: &crate::picked::Ranges,
        flown: bool,
    ) -> Option<(CardPlan, Option<std::sync::MutexGuard<'static, u64>>)> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let declared: Vec<Declared> = declared.iter().filter(|(shadow, _)| shadow.carded()).cloned().collect();
        let declared = if crate::picked::lands() { crate::picked::with(declared, &picked.written) } else { declared };
        // (Of a copy with a declared range here, that says it.)
        let looked: Vec<&crate::picked::Range> = if picked.looked.is_empty() {
            Vec::new()
        } else {
            let declared_keys: ByCopy<()> =
                declared.iter().map(|(s, _)| (Arc::as_ptr(s) as usize, ())).collect();
            picked
                .looked
                .iter()
                .filter(|l| !declared_keys.contains_key(&(Arc::as_ptr(&l.0) as usize)))
                .collect()
        };
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered. Once
        // `victory-relay` is on its dedicated Copy Queue it orders via
        // `CardPlan::sent_as` rather than `QUEUE_ORDER`, so drop the lock
        // immediately after claiming `sent_as`.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        if COPY_QUEUE_ACTIVE.load(Ordering::Acquire) {
            order = None;
        }
        let gathering = crate::timeline::span("copies in the card's memory: what changed");
        let dirty = crate::shadow::card_dirty(sent_as);
        gathering.end();
        if declared.is_empty() && dirty.is_empty() && looked.is_empty() {
            return None;
        }
        // Take the dirty bytes out of CPU staging BEFORE locking `CARD_WORK` so
        // `victory-land` (`relay_card_back` / `card_done`) is not blocked during copies.
        let taking = crate::timeline::span("copies in the card's memory: taken");
        let own = if send::own() && !dirty.is_empty() { send::take(self, &dirty) } else { None };
        taking.end();
        // A copy whose shader writes are declared for the first time gets its
        // `before` hash buffer with all pieces marked stale; lock-free fast path
        // once initialized.
        for shadow in declared.iter().map(|d| &d.0).chain(looked.iter().map(|l| &l.0)) {
            if !shadow.card_has_before() {
                shadow.card_make_before(|size| {
                    let before = self.internal_uav_buffer(crate::shadow::card_hash_bytes(size))?;
                    name(&before, || "title buffer on the card, hashes of its pieces".into());
                    Some(before)
                });
            }
        }
        // Mark CPU-uploaded pieces stale instead of eagerly hashing pieces no
        // shader is writing in this submission.
        for (shadow, ranges) in &dirty {
            shadow.card_mark_hash_stale(ranges);
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // One compare pipeline for all passes of this submission (`card_hash_wide_now`).
        let compare_pso = work.compare_for(card_hash_wide_now()).clone();
        // Build the compare plan first so we know `declared ∪ looked` pieces
        // and only re-hash `hash_stale ∩ (declared ∪ looked)`.
        let mut plan_ranges = Vec::with_capacity(declared.len() + looked.len());
        let mut at = work.taken;
        let upcheck_on = upcheck::on();
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::with_capacity(sorted.len());
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_has_before();
            let mut card_spans: Vec<crate::shadow::CardSpan> =
                spans.into_iter().filter_map(|(from, to)| shadow.card_span(from, to)).collect();
            // Coalesce piece-overlapping or piece-adjacent declared spans on the same
            // card buffer so `CARD_COMPARE` never dispatches the same 64 KiB piece twice
            // or splits adjacent pieces across separate dispatches.
            if card_spans.len() > 1 && !upcheck_on {
                card_spans = crate::shadow::joined_piece_runs(card_spans);
            }
            let base = shadow.span().0 as usize;
            for span in card_spans {
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan_ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (base + span.from as usize, base + span.to as usize),
                    span,
                    at,
                    compared,
                    picked: false,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        for (shadow, from, len) in looked {
            let twin = shadow.card_has_before();
            if let Some(span) = shadow.card_span(*from, from + len)
                && twin
                && u64::from(at + span.pieces) * 4 <= work.differs.2
            {
                plan_ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (*from, from + len),
                    span,
                    at,
                    compared: true,
                    picked: true,
                });
                at += span.pieces;
            }
        }
        work.taken = at;
        // Group compared spans by buffer using O(1) map lookup and take only stale pieces.
        let mut compared_by_shadow: Vec<(Arc<crate::shadow::Shadow>, Vec<crate::shadow::CardSpan>)> = Vec::new();
        let mut compared_places: ByCopy<usize> = ByCopy::default();
        for r in plan_ranges.iter().filter(|r| r.compared) {
            let key = Arc::as_ptr(&r.shadow) as usize;
            match compared_places.get(&key) {
                Some(&idx) => compared_by_shadow[idx].1.push(r.span),
                None => {
                    compared_places.insert(key, compared_by_shadow.len());
                    compared_by_shadow.push((r.shadow.clone(), vec![r.span]));
                }
            }
        }
        let mut rehashes: Vec<(Arc<crate::shadow::Shadow>, Vec<crate::shadow::CardSpan>)> = Vec::new();
        let mut rehash_keys: ByCopy<()> = ByCopy::default();
        for (shadow, spans) in compared_by_shadow {
            let runs = shadow.card_take_stale_runs(&spans);
            if !runs.is_empty() {
                rehash_keys.insert(Arc::as_ptr(&shadow) as usize, ());
                rehashes.push((shadow, runs));
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        let mut looks = None;
        let mut sync_list: Option<ID3D12CommandList> = None;
        if !dirty.is_empty() || !rehashes.is_empty() {
            let Some((sync, open)) = work.sync.pop().or_else(|| self.make_prelude_open().map(|p| (p, true))) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let _recording = crate::timeline::span("copies in the card's memory: the list");
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    if !open {
                        allocator.Reset()?;
                        list.Reset(allocator, None)?;
                    }
                    if !dirty.is_empty() {
                        if upcheck_on {
                            looks = upcheck::record(self, list, &dirty);
                        }
                        // Every copy first, then one barrier call, then every
                        // pass: a wait between each copy and its pass had the
                        // card do them one after the other.
                        let mut sent = Vec::new();
                        let mut transitioned: ByCopy<()> = ByCopy::default();
                        if let Some(own) = &own {
                            own.record(
                                self,
                                list,
                                &dirty,
                                |shadow| looks.as_ref().is_some_and(|l| l.hold(shadow)),
                                |shadow| rehash_keys.contains_key(&(std::ptr::from_ref(shadow) as usize)),
                            );
                        }
                        for (shadow, ranges) in &dirty {
                            if own.is_none()
                                && let Some(staging) = shadow.card_staging_gpu()
                            {
                                for &(a, b) in ranges {
                                    list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                                }
                            }
                            let bytes: u64 = ranges.iter().map(|&(a, b)| (b - a) as u64).sum();
                            crate::shadow::CARD_STATS[0].fetch_add(bytes, Ordering::Relaxed);
                            crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            traffic::moved(traffic::UP, shadow.span(), ranges.iter().copied());
                            if own.is_none()
                                && (!ranges.is_empty() || looks.as_ref().is_some_and(|l| l.hold(shadow)))
                            {
                                let raw = shadow.buffer.as_raw() as usize;
                                if transitioned.insert(raw, ()).is_none() {
                                    let target = if rehash_keys.contains_key(&(Arc::as_ptr(shadow) as usize)) {
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                    } else {
                                        D3D12_RESOURCE_STATE_COMMON
                                    };
                                    sent.push(transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST, target));
                                }
                            }
                        }
                        if !sent.is_empty() {
                            list.ResourceBarrier(&sent);
                        }
                    }
                    if !rehashes.is_empty() {
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&compare_pso);
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootShaderResourceView(4, work.piece_indices.0.GetGPUVirtualAddress());
                        let mut post_barriers = Vec::with_capacity(rehashes.len() * 2);
                        let mut befores: Vec<&ID3D12Resource> = Vec::with_capacity(rehashes.len());
                        for (shadow, runs) in &rehashes {
                            let (Some((before, before_gpu)), size) =
                                (shadow.card_before_ref(), shadow.card_hash_size())
                            else {
                                continue;
                            };
                            list.SetComputeRootShaderResourceView(1, shadow.buffer_gpu());
                            list.SetComputeRootUnorderedAccessView(2, before_gpu);
                            let total_pieces: u32 = runs.iter().map(|r| r.pieces).sum();
                            CARD_PASSES[1].fetch_add(u64::from(total_pieces), Ordering::Relaxed);
                            if runs.len() == 1 {
                                CARD_PASSES[0].fetch_add(1, Ordering::Relaxed);
                                card_pass(list, &runs[0], size as u32, 0, 0);
                            } else if u64::from(work.taken_indices + total_pieces) * 4 <= work.piece_indices.2 {
                                let base = work.taken_indices;
                                let dst = (work.piece_indices.1 as *mut u32).add(base as usize);
                                let mut k = 0usize;
                                for run in runs {
                                    for p in run.first..run.first + run.pieces {
                                        dst.add(k).write(p);
                                        k += 1;
                                    }
                                }
                                work.taken_indices += total_pieces;
                                let passes = card_pass_indexed(list, runs[0].skew, size as u32, total_pieces, base, 0, 3);
                                CARD_PASSES[0].fetch_add(passes, Ordering::Relaxed);
                            } else {
                                CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                                for run in runs {
                                    card_pass(list, run, size as u32, 0, 0);
                                }
                            }
                            post_barriers.push(transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                D3D12_RESOURCE_STATE_COMMON,
                            ));
                            befores.push(before);
                        }
                        // Return re-hashed buffers and their `before` hash UAVs
                        // to COMMON so subsequent lists and `check` in the same
                        // ExecuteCommandLists see completed writes.
                        for before in befores {
                            post_barriers.push(transition(
                                before,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_COMMON,
                            ));
                        }
                        if !post_barriers.is_empty() {
                            list.ResourceBarrier(&post_barriers);
                        }
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => sync_list = Some(list),
                Err(e) => fail(format_args!("host copies in the card's memory could not be brought up to date ({e})")),
            }
            used.push((false, sync));
        }
        // Keep `QUEUE_ORDER` held until `sync` is submitted on `self.queue` so
        // `relay_card_back` fetches stay strictly ordered relative to `sent_as`.
        let held_order = if sync_list.is_some() { order } else { None };
        let mut plan = CardPlan {
            ranges: plan_ranges,
            sync: sync_list,
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            sent_as,
            looks,
            sends: own.map(send::Own::memory),
        };
        if plan.ranges.iter().any(|r| r.compared) {
            let Some((check, open)) = work.check.pop().or_else(|| self.make_prelude_open().map(|p| (p, true))) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    if !open {
                        allocator.Reset()?;
                        list.Reset(allocator, None)?;
                    }
                    // Resource-scoped UAV barriers for only the buffers compared.
                    let mut seen: ByCopy<()> = ByCopy::default();
                    let mut barriers = Vec::with_capacity(plan.ranges.len());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let raw = r.shadow.buffer.as_raw() as usize;
                        if seen.insert(raw, ()).is_none() {
                            barriers.push(uav_barrier(&r.shadow.buffer));
                        }
                    }
                    if !barriers.is_empty() {
                        list.ResourceBarrier(&barriers);
                    }
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&compare_pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    list.SetComputeRootShaderResourceView(4, work.piece_indices.0.GetGPUVirtualAddress());
                    let compared: Vec<&CardRange> = plan.ranges.iter().filter(|r| r.compared).collect();
                    let descriptors: Vec<(usize, u32, u32, u32, u32)> = compared
                        .iter()
                        .map(|r| {
                            let mode = if flown && !r.picked { 2 } else { 1 };
                            (Arc::as_ptr(&r.shadow) as usize, mode, r.span.first, r.span.pieces, r.at)
                        })
                        .collect();
                    let mut bound_shadow: usize = 0;
                    for group in batch_compare_groups(&descriptors) {
                        let first = compared[group.start];
                        let (Some((_, before_gpu)), size) =
                            (first.shadow.card_before_ref(), first.shadow.card_hash_size())
                        else {
                            continue;
                        };
                        let mode = descriptors[group.start].1;
                        let key = Arc::as_ptr(&first.shadow) as usize;
                        if bound_shadow != key {
                            bound_shadow = key;
                            list.SetComputeRootShaderResourceView(1, first.shadow.buffer_gpu());
                            list.SetComputeRootUnorderedAccessView(2, before_gpu);
                        }
                        let total_pieces: u32 = compared[group.clone()].iter().map(|r| r.span.pieces).sum();
                        if group.len() > 1
                            && u64::from(work.taken_indices + total_pieces) * 4 <= work.piece_indices.2
                        {
                            let base = work.taken_indices;
                            let dst = (work.piece_indices.1 as *mut u32).add(base as usize);
                            let mut k = 0usize;
                            for r in &compared[group.clone()] {
                                for p in r.span.first..r.span.first + r.span.pieces {
                                    dst.add(k).write(p);
                                    k += 1;
                                }
                            }
                            work.taken_indices += total_pieces;
                            let passes = card_pass_indexed(
                                list,
                                first.span.skew,
                                size as u32,
                                total_pieces,
                                base,
                                first.at,
                                mode + 3,
                            );
                            CARD_PASSES[2].fetch_add(passes, Ordering::Relaxed);
                        } else {
                            for r in &compared[group] {
                                // (What is only looked at keeps its hashes: what
                                // changed there is the next declared range's to land.)
                                card_pass(list, &r.span, size as u32, r.at, mode);
                                CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some((plan, held_order))
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        if let Some(looks) = plan.looks {
            looks.read();
        }
        if let Some(memory) = plan.sends {
            send::give_back(memory);
        }
        // Pre-reset `sync` and `check` command allocators/lists off `victory-submit`
        // (and outside `CARD_WORK.lock()`) so `card_before` skips both Reset calls.
        let mut reset_lists: Vec<(bool, Prelude, bool)> = Vec::with_capacity(plan.lists.len());
        for (check, list) in plan.lists {
            // SAFETY: this plan's GPU submission has completed.
            let open = unsafe { list.0.Reset().and_then(|()| list.1.Reset(&list.0, None)).is_ok() };
            reset_lists.push((check, list, open));
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list, open) in reset_lists {
            if check {
                work.check.push((list, open));
            } else {
                work.sync.push((list, open));
            }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
            work.taken_indices = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        // SAFETY: plain object creation.
        let copying = relay_copy().then(|| unsafe {
            (|| -> windows::core::Result<(Prelude, ID3D12CommandQueue)> {
                let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList =
                    self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                list.Close()?;
                let queue: ID3D12CommandQueue = self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    ..Default::default()
                })?;
                Ok(((allocator, list), queue))
            })()
        });
        let (list, queue) = match copying {
            Some(Ok((list, queue))) => {
                COPY_QUEUE_ACTIVE.store(true, Ordering::Release);
                (list, Some(queue))
            }
            Some(Err(e)) => {
                tracing::warn!(error = %e, "no copy queue: what shaders wrote to the card's memory is fetched on the drawing queue");
                (self.make_prelude()?, None)
            }
            None => (self.make_prelude()?, None),
        };
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                queue,
                fence,
                event,
                sent: 0,
                buffer: None,
                suspects: torn::Suspects::default(),
                had: crate::land::Had::default(),
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up. `plans` come with the value signalled behind
    /// their flight. `meanwhile` is called while the fetch is on its way, if
    /// there is one.
    fn relay_card_back(&self, plans: &[(&CardPlan, u64)], fetch: &mut Option<RelayFetch>, meanwhile: &mut dyn FnMut()) {
        use crate::land;
        let gathering = std::time::Instant::now();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        if let (Some(fetch), Some(first)) = (fetch.as_mut(), plans.iter().map(|p| p.1).min()) {
            fetch.had.landing(first);
        }
        // SAFETY: the mapping of the results. The compares of these plans
        // have run, and their dwords are given to no other plan before these
        // are done with (`card_done`).
        let differs = differs.map(|(differs, dwords)| unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) });
        if let Some(differs) = differs {
            // Pieces caught while a later flight was at them (`torn`).
            let suspects = fetch.as_ref().map(|f| &f.suspects);
            let mut seen = std::collections::HashSet::new();
            for &(plan, behind) in plans {
                plan.picked_found(differs);
                let flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .filter(|r| !r.picked)
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| torn::flagged(d));
                            let wanted = suspects.map_or(flagged, |s| s.wanted((r.shadow.id(), r.span.first + p), flagged));
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, wanted)
                        })
                    })
                    .collect();
                let fetched = flagged_pieces(&flags);
                for r in plan.ranges.iter().filter(|r| !r.picked) {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    if upcheck::on() {
                        upcheck::declared(r.shadow.id(), r.span.from as usize, r.span.to as usize);
                    }
                    for p in 0..r.span.pieces {
                        if fetched.binary_search(&(copy, r.span.first + p)).is_err() {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo < hi && !seen.contains(&(copy, lo, hi)) {
                            let key = (r.shadow.id(), r.shadow.span().0 as usize + lo, hi - lo);
                            if fetch.as_ref().is_some_and(|f| f.had.holds(behind, key)) {
                                land::count(land::stat::HAD, 1);
                                land::count(land::stat::HAD_BYTES, (hi - lo) as u64);
                                if land::skip_on() {
                                    continue;
                                }
                            }
                            if upcheck::on() {
                                upcheck::fetched(r.shadow.id(), lo);
                            }
                            seen.insert((copy, lo, hi));
                            pieces.push((r.shadow.clone(), lo, hi, total));
                            total += (hi - lo) as u64;
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
        }
        if land::on() {
            // In the order they are landed, each byte of a copy once.
            let keys: Vec<land::Piece> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)).collect();
            total = 0;
            pieces = land::ordered(&keys)
                .into_iter()
                .map(|(i, title, len)| {
                    let shadow = &pieces[i].0;
                    let lo = title - shadow.span().0 as usize;
                    let at = total;
                    total += len as u64;
                    (shadow.clone(), lo, lo + len, at)
                })
                .collect();
        }
        land::spent(land::stat::GATHER, gathering);
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        // On the copy queue a later flight may be at a piece while it is
        // fetched: the pieces come down whole, to be hashed.
        let check = (torn::on() && fetch.as_ref().is_some_and(|f| f.queue.is_some()))
            .then(|| torn::Check::new(plans.iter().map(|p| p.0), differs.unwrap_or_default(), &mut pieces));
        if let Some(check) = &check {
            total = check.total;
        }
        #[cfg(feature = "diagnostics")]
        for (shadow, lo, hi, _) in &pieces {
            traffic::moved(traffic::FETCHED, shadow.span(), std::iter::once((*lo, *hi)));
        }
        if crate::shadow::gpu_only_mode() > 0 && !pieces.is_empty() {
            for run in pieces.chunk_by(|a, b| std::sync::Arc::ptr_eq(&a.0, &b.0)) {
                let ranges: Vec<(usize, usize)> = run.iter().map(|(_, lo, hi, _)| (*lo, *hi)).collect();
                run[0].0.gpu_only_note_fetched(&ranges);
            }
        }
        // The sends these flights ran behind, all a fetch on the copy queue
        // is known to have run behind.
        let behind = plans.iter().map(|p| p.0.sent_as).max().unwrap_or(0);
        if pieces.is_empty() {
            // Nothing to fetch: every send so far is older than the next
            // fetch, if that goes to the queue the sends went to.
            let after = match fetch.as_ref().is_some_and(|f| f.queue.is_some()) {
                true => behind,
                false => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            };
            crate::shadow::relay_fetched(after);
            return;
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.internal_shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let copy_queue = f.queue.clone();
        let waiting = std::time::Instant::now();
        // The flights the fetch runs behind: those in the line by now have
        // been given to the queue. On the copy queue, those landed here.
        let ahead = match copy_queue {
            Some(_) => 0,
            None => line().target,
        };
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in check.as_ref().map_or(&pieces, |c| &c.copies) {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                let _fetching = crate::timeline::span("landing: fetching pieces");
                if let Some(queue) = &copy_queue {
                    queue.ExecuteCommandLists(&lists);
                    queue.Signal(&fence, sent)?;
                    meanwhile();
                    fence.SetEventOnCompletion(sent, event)?;
                    #[cfg(feature = "diagnostics")]
                    let _card = crate::timeline::card_wait(None);
                    WaitForSingleObject(event, INFINITE);
                    return Ok(behind);
                }
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                meanwhile();
                fence.SetEventOnCompletion(sent, event)?;
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(None);
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
        };
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        let merging = std::time::Instant::now();
        let _merging = crate::timeline::span("landing: merging fetched pieces");
        let mut written = 0;
        if land::on() {
            let titles: Vec<(usize, usize)> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.span().0 as usize + lo, hi - lo)).collect();
            let jobs = land::jobs(&titles);
            let put = std::sync::atomic::AtomicU64::new(0);
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                if let Some(check) = &check {
                    for i in jobs[j].clone() {
                        // SAFETY: as below.
                        unsafe { check.look(i, data) };
                    }
                }
                let mut notes = land::Notes::default();
                for run in pieces[jobs[j].clone()].chunk_by(|a, b| std::sync::Arc::ptr_eq(&a.0, &b.0)) {
                    let addressed: Vec<(usize, usize, usize)> =
                        run.iter().map(|(_, lo, hi, at)| (*lo, *hi, data + *at as usize)).collect();
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run, and the next waits
                    // for this call.
                    put.fetch_add(unsafe { run[0].0.land_card(&addressed, after, &mut notes) }, Ordering::Relaxed);
                }
                notes.flush();
                land::spent(land::stat::CARD_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written = put.into_inner();
        } else {
            let mut title = Vec::new();
            for (i, (shadow, lo, hi, at)) in pieces.iter().enumerate() {
                if let Some(check) = &check {
                    // SAFETY: as below.
                    unsafe { check.look(i, data) };
                }
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged as u64;
            }
        }
        land::spent(land::stat::CARD, merging);
        COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
        if let Some(f) = fetch.as_mut() {
            if let Some(check) = &check {
                check.done(&mut f.suspects);
            }
            f.had.fetched(
                ahead,
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)),
            );
        }
        // Sends up to `after` are of no use to any later merge.
        crate::shadow::relay_fetched(after);
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        let mut stage = crate::timeline::span("landing: what is to be looked through");
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<(&CardPlan, u64)> = flights.iter().filter_map(|f| Some((f.card.as_ref()?, f.fence))).collect();
                // (Each byte once: the ranges are joined.)
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in flight.back.iter().filter(|(shadow, _)| !shadow.carded()) {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => all.extend_from_slice(ranges),
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                let mut hand_back = relay_hand_back(&back);
                stage.next("landing: copies in the card's memory");
                self.relay_card_back(&plans, fetch, &mut || {
                    if crate::land::on() {
                        let _looking = crate::timeline::span("landing: looking through copies, beside the fetch");
                        hand_back.look(crate::land::stat::SCAN_BESIDE);
                    }
                });
                stage.next("landing: copies in the PC's memory");
                hand_back.merge();
                stage.next("landing: ranges done with");
                let finishing = std::time::Instant::now();
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                crate::land::spent(crate::land::stat::DONE, finishing);
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        stage.next("landing: read-backs, telling the title");
        let mut last = 0;
        let mut landed = Vec::new();
        let mut telling = std::time::Instant::now();
        for item in items {
            telling = match item {
                Late::Run(mut flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    if let Some(dxr) = flight.dxr {
                        crate::dxr::finish_submission(dxr);
                    }
                    if let Some(own) = flight.own {
                        own.landed();
                    }
                    for detached in &mut flight.lists {
                        detached.release_tables();
                    }
                    landed.push((flight.lists, flight.order));
                    crate::land::spent(crate::land::stat::READ_BACKS, telling)
                }
                Late::Write(at, what, bytes) => {
                    what.write(at as *mut u8, bytes);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
                Late::Signal(fence, value) => {
                    crate::fence::set(fence as crate::object::This, value);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
            };
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        stage.next("landing: the rest");
        self.log_messages();
        crate::land::spent(crate::land::stat::REST, telling);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                in_hand_signals,
                landing,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            in_hand_signals.clear();
            landing.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    ranges.extend(flight.texture_back_ranges.iter().copied());
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        plan.picked_found(differs);
        for (i, r) in plan.ranges.iter().enumerate().filter(|(_, r)| !r.picked) {
            for p in 0..r.span.pieces {
                if r.compared && !torn::flagged(differs[(r.at + p) as usize]) {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.internal_shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            let mut written_ranges = Vec::new();
            if crate::shadow::gpu_only_mode() > 0 {
                for &(i, lo, hi, _) in &pieces {
                    written_ranges.push((plan.ranges[i].shadow.clone(), lo, hi));
                }
                for run in written_ranges.chunk_by(|a, b| std::sync::Arc::ptr_eq(&a.0, &b.0)) {
                    let ranges: Vec<(usize, usize)> = run.iter().map(|(_, lo, hi)| (*lo, *hi)).collect();
                    run[0].0.gpu_only_note_fetched(&ranges);
                }
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the title's queues were given before it.
        crate::queue::before_sync();
        let _ = self.submit(runs, &[], false, false, true);
    }

    /// Runs a closed internal upload-only host list synchronously without
    /// draining title queues (`queue::before_sync`), so it is deadlock-safe
    /// when invoked from `VICTORY_THIN=1` worker threads. Settles in-flight
    /// GPU work and waits for the upload list's fence before returning.
    pub(crate) fn execute_sync_internal(&self, list: &HostList) {
        let _ = self.submit(&[list], &[], false, false, true);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        self.execute_parts(lists, order, &[]);
    }

    /// [`Host::execute_late`] of parts of submissions in one: another begins
    /// at each place of `order` in `parts` ([`joins`]).
    pub(crate) fn execute_parts(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>, parts: &[usize]) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        if let Some(cut) = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            card_steps::cut_for(&runs)
        } {
            match split(lists, order, cut) {
                Ok([(lists, order), (more, behind)]) => {
                    let first: Vec<usize> = parts.iter().copied().filter(|&p| p < cut).collect();
                    let second: Vec<usize> = parts.iter().filter(|&&p| p > cut).map(|p| p - cut).collect();
                    self.execute_parts(lists, order, &first);
                    self.execute_parts(more, behind, &second);
                }
                Err((lists, order)) => self.send_late(lists, order, parts),
            }
            return;
        }
        self.send_late(lists, order, parts);
    }

    fn send_late(&self, mut lists: Vec<crate::list::Detached>, mut order: Vec<usize>, parts: &[usize]) {
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, parts, true, true, true)
        };
        let Some((mut submitted, mut sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        if !sent.ahead.is_empty() {
            let first = lists.len();
            order.splice(0..0, first..first + sent.ahead.len());
            lists.extend(sent.ahead.drain(..).map(crate::list::Detached::ours));
        }
        let told = !sent.writes.is_empty();
        let sizes = crate::memory::asked_sizes(order.iter().map(|&i| lists[i].host()));
        let (most, wake_land) = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, landed, poke, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
                dxr: sent.dxr,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            let wake_land = landed.len() > bury_lag() as usize;
            if wake_land {
                *poke = true;
            }
            (*flying >= late_most(), wake_land)
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if sizes {
            self.settle(&mut submitted, WHY_READ);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        if wake_land {
            LINE_MOVED.notify_one();
        }
    }

    /// Sends `list`, a closed list of ours, as a flight ([`late_free_present`]):
    /// not waited for, settled with the others. The caller keeps the list and
    /// does not reset it before the submission whose number comes back has
    /// finished ([`FINISHED`]); 0 if it ran and was waited for after all.
    #[cfg_attr(not(test), allow(dead_code))]
    fn execute_own(&self, list: &HostList) -> u64 {
        self.execute_own_inner(list, true)
    }

    /// Copy-only variant of [`Host::execute_own`] for swapchain present copies
    /// that bind no title descriptor heaps, touch no buffer shadows, and record
    /// no DXR ops.
    fn execute_own_copy(&self, list: &HostList) -> u64 {
        self.execute_own_inner(list, false)
    }

    fn execute_own_inner(&self, list: &HostList, needs_heaps: bool) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], &[], true, false, needs_heaps) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
                dxr: sent.dxr,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let _settling = crate::timeline::span("settling flights");
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            #[cfg(feature = "diagnostics")]
                            let _card = crate::timeline::card_wait(Some(*submitted));
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                hand_back(&back);
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(mut flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        if let Some(dxr) = flight.dxr {
                            crate::dxr::finish_submission(dxr);
                        }
                        for detached in &mut flight.lists {
                            detached.release_tables();
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// For a submission of `runs` that refreshes the ranges `stale` of host
    /// copies, which flights may be writing and the title wrote too
    /// ([`stale_under_flights`]): the flights land first. `ranges` are
    /// those of the submission's one asking, which is made again.
    ///
    /// With [`card_steps`] only the flights that declared them: a range of
    /// a copy nobody declared any more is refreshed like any other. What
    /// they landed is title memory written, so everything is asked and
    /// looked at again; what is left after three times waits as before.
    fn settle_stale(
        &self,
        submitted: &mut u64,
        runs: &[&HostList],
        mut stale: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        ranges: &[(usize, usize)],
    ) {
        if relay_exact() && card_steps::stage() > 0 {
            for _ in 0..3 {
                #[cfg(feature = "diagnostics")]
                blame_stale(&stale);
                relay_wait(WHY_STALE, |_| stale.iter().all(|(shadow, from, to)| !shadow.writes_pending_in(*from, *to)));
                crate::watch::ask_once(ranges);
                stale = stale_under_flights(runs);
                if stale.is_empty() {
                    return;
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        blame_stale(&stale);
        self.settle(submitted, WHY_STALE);
        crate::watch::ask_once(ranges);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    ///
    /// `parts`: the lists of `runs` at which another joined part begins
    /// ([`joins`]); used for per-part front-refresh planning while the closed
    /// lists go to the queue in one `ExecuteCommandLists` call.
    fn submit<'a>(
        &'a self,
        runs: &[&HostList],
        parts: &[usize],
        late: bool,
        take_ahead: bool,
        needs_heaps: bool,
    ) -> Option<(std::sync::MutexGuard<'a, u64>, Sent)> {
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        // Splice any asynchronous heap-pass refreshes before taking the submission
        // lock so waiting on `victory-aside` never blocks other `submitted` callers.
        for list in runs {
            list.take_coming();
        }
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut stage = crate::timeline::span("submit: the submission lock");
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        // Textures uploaded ahead (`crate::heap_worker`): taken under the
        // submission lock so no other submission can observe published slots
        // before these upload lists execute, and prepended ahead of the
        // title's lists (and ahead of `crate::dxr::record_submission`).
        let (ahead_lists, ahead_publish) = if take_ahead {
            crate::heap_worker::take_ready()
        } else {
            (Vec::new(), crate::heap_worker::PendingPublish::empty())
        };
        let runs_with_ahead: Vec<&HostList>;
        let parts_with_ahead: Vec<usize>;
        let (runs, parts) = if !ahead_lists.is_empty() {
            let n = ahead_lists.len();
            runs_with_ahead = ahead_lists.iter().chain(runs.iter().copied()).collect();
            parts_with_ahead = parts.iter().map(|&p| p + n).collect();
            (&runs_with_ahead[..], &parts_with_ahead[..])
        } else {
            (runs, parts)
        };
        submitting(1, runs.len());
        #[cfg(feature = "diagnostics")]
        let ahead = Split::begin();
        #[cfg(feature = "diagnostics")]
        SPLIT[11].fetch_add(called.elapsed().as_nanos() as u64, Ordering::Relaxed);
        stage.next("submit: steps given to the card");
        // Flights in the air ([`late`]): a submission that is waited for
        #[cfg(feature = "diagnostics")]
        let p_pre = std::time::Instant::now();
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        // Or the reads are done on the card, behind the flights.
        let own = if late { self.card_steps(runs) } else { None };
        #[cfg(feature = "diagnostics")]
        ahead.part(12);
        stage.next("submit: waiting for flights it reads from");
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if own.is_none()
            && let Some(why) = late_blocked(runs)
        {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| {
                    blocked_by(&line.written, runs, false).is_none() && !card_steps::copied_read(&card_steps::copies(line), runs)
                });
            } else {
                self.settle(&mut submitted, why);
            }
        }
        #[cfg(feature = "diagnostics")]
        ahead.part(13);
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        // Renderer port #7: the round asked for when the title submitted it,
        // if that is known (`crate::port_journal`).
        let heaps_on = needs_heaps && crate::port_journal::on();
        let heaps_synced = needs_heaps.then(|| match crate::port_journal::needed() {
            0 => crate::descriptor::begin_sync_heaps(),
            mark => crate::descriptor::begin_sync_heaps_after(mark),
        });
        // That list runs first, and is one of the submission's from here on.
        let with: Vec<&HostList>;
        let runs = match &own {
            Some(own) => {
                with = std::iter::once(own.list()).chain(runs.iter().copied()).collect();
                &with[..]
            }
            None => runs,
        };
        let parts: Vec<usize> = parts.iter().map(|&n| n + usize::from(own.is_some())).collect();
        stage.next("submit: the sweep");
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            if upcheck::on() {
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 5000 {
            // `sweep()` uses refcounts and `retire()`/`bury_lag()` rather than
            // requiring GPU idle; only drain flights when `upcheck::at_rest` is on.
            if upcheck::on() {
                self.settle(&mut submitted, WHY_SYNC);
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        stage.next("submit: gathering what to ask about");
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        ahead.part(14);
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        #[cfg(feature = "diagnostics")]
        let split = Split::begin();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        #[cfg(feature = "diagnostics")]
        PREP_PHASE_NS[9].fetch_add(p_pre.elapsed().as_nanos() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let mut p_t = std::time::Instant::now();
        let mut prepared_front: Option<PreparedFront> = None;
        if crate::watch::once() {
            let (ranges, stale_spans) = if refresh_front() {
                let (front, ranges, stale_spans) = plan_and_join_fronts(runs, &parts, wide_textures(), behind);
                prepared_front = Some(front);
                (ranges, stale_spans)
            } else {
                let mut spans: ByCopy<(usize, usize)> = ByCopy::default();
                let mut last: Option<(usize, usize, usize)> = None;
                let mut stale_spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
                let mut stale_places: ByCopy<usize> = ByCopy::default();
                let mut last_stale: Option<(usize, usize)> = None;
                for list in runs {
                    for step in &list.deferred().copies {
                        if let Step::Refresh(shadow, address, len) = step {
                            let key = std::sync::Arc::as_ptr(shadow) as usize;
                            let end = *address + *len;
                            if shadow.asked_ahead() {
                                match &mut last {
                                    Some((k, lo, hi)) if *k == key => {
                                        *lo = (*lo).min(*address);
                                        *hi = (*hi).max(end);
                                    }
                                    prev => {
                                        if let Some((pk, plo, phi)) = prev.replace((key, *address, end)) {
                                            let span = spans.entry(pk).or_insert((plo, phi));
                                            *span = (span.0.min(plo), span.1.max(phi));
                                        }
                                    }
                                }
                            }
                            if behind && shadow.writes_pending() {
                                let i = match last_stale {
                                    Some((lk, li)) if lk == key => li,
                                    _ => {
                                        let i = *stale_places.entry(key).or_insert_with(|| {
                                            stale_spans.push((shadow.clone(), *address, end));
                                            stale_spans.len() - 1
                                        });
                                        last_stale = Some((key, i));
                                        i
                                    }
                                };
                                let s = &mut stale_spans[i];
                                (s.1, s.2) = (s.1.min(*address), s.2.max(end));
                            }
                        }
                    }
                }
                if let Some((pk, plo, phi)) = last {
                    let span = spans.entry(pk).or_insert((plo, phi));
                    *span = (span.0.min(plo), span.1.max(phi));
                }
                (spans.into_values().collect::<Vec<(usize, usize)>>(), stale_spans)
            };
            #[cfg(feature = "diagnostics")]
            SPLIT[8].fetch_add(split.so_far(), Ordering::Relaxed);
            stage.next("submit: asking the write watch");
            crate::watch::ask_once(&ranges);
            #[cfg(feature = "diagnostics")]
            {
                PREP_PHASE_NS[0].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                p_t = std::time::Instant::now();
            }
            #[cfg(feature = "diagnostics")]
            split.part(0);
            stage.next("submit: title bytes under flights' writes");
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if upcheck::on() {
                upcheck::order(runs);
                if behind {
                    upcheck::reads(runs);
                }
            }
            let stale = if behind { stale_under_flights_spans(stale_spans) } else { Vec::new() };
            if !stale.is_empty() {
                let settling = std::time::Instant::now();
                self.settle_stale(&mut submitted, runs, stale, &ranges);
                settled += settling.elapsed();
            }
        } else if behind && !stale_under_flights(runs).is_empty() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[1].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        split.part(1);
        // What flights in the air may write.
        let air = line().written.clone();
        stage.next("submit: refreshes ahead of writes");
        let front_covered = if let Some(front) = prepared_front {
            batch.refresh_front_prepared(front)
        } else if refresh_front() {
            batch.refresh_front(runs, &parts)
        } else {
            if ahead_of_writes() {
                batch.refresh_ahead(runs, |list, i| own.as_ref().is_some_and(|own| own.takes(list, i)));
            }
            vec![false; runs.len()]
        };
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[2].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        stage.next("submit: the lists' steps");
        let mut prelude_batch = PreludeBatch::default();
        for (run_idx, list) in runs.iter().enumerate() {
            let uploads = list.run_deferred(
                &mut batch,
                &air,
                |i| own.as_ref().is_some_and(|own| own.takes(list, i)),
                front_covered[run_idx],
            );
            firsts.push(lists.len());
            if let Some(plan) = list.plan_prelude(uploads) {
                let slot = lists.len();
                lists.push(None);
                prelude_batch.push(list, slot, plan, &mut lists);
            }
            lists.push(list.list.cast().ok());
        }
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[3].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        batch.fill();
        stage.next("submit: the last refreshes");
        batch.flush();
        crate::watch::ask_once_end();
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[5].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        split.part(2);
        stage.next("submit: snapshots");
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[6].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        split.part(3);
        // Wait for placed-heap sync BEFORE `card_before` so `QUEUE_ORDER` is
        // never held across `wait_heaps` even when Direct Queue fallback is used.
        if let Some(heaps_synced) = heaps_synced {
            stage.next("submit: waiting for the heaps' records");
            let heaps_began = std::time::Instant::now();
            drop(heaps_synced);
            crate::port_journal::heaps_waited(heaps_began.elapsed().as_nanos() as u64, heaps_on);
        }
        // Publish slots for the heap worker's texture uploads after
        // placed-heap sync completes and before `make_waiting` /
        // `dxr::record_submission` / `ExecuteCommandLists`.
        ahead_publish.publish();
        if needs_heaps {
            // The slots that waited for a texture's host copy, before the
            // lists that may read them run.
            crate::descriptor::make_waiting();
        }
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[7].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        stage.next("submit: copies in the card's memory");
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if needs_heaps && flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        // What shaders may write through views they pick: last, so that no
        // refresh of this submission leaves it alone.
        let picked = if needs_heaps {
            let picked = crate::picked::declare(runs);
            for (shadow, at, len) in &picked.written {
                shadow.writing(*at, *len);
            }
            picked
        } else {
            crate::picked::Ranges::default()
        };
        let declared = std::mem::take(&mut batch.declared);
        let (card, mut card_order) = if needs_heaps {
            match self.card_before(&declared, &picked, flown) {
                Some((plan, order)) => (Some(plan), order),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        // The sends are taken: from here on the references hold what the
        // copies in the lists put.
        for (shadow, at, bytes) in batch.placed.drain(..) {
            shadow.placed_copy(at, &bytes);
        }
        #[cfg(feature = "diagnostics")]
        {
            PREP_PHASE_NS[6].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            p_t = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        split.part(4);
        // Collect any async state preludes before DXR splicing / ExecuteCommandLists.
        prelude_batch.finish(&mut lists);
        #[cfg(feature = "diagnostics")]
        PREP_PHASE_NS[4].fetch_add(p_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // Replay queued native DXR AS builds/copies/extractions (and initialize
        // any newly allocated empty TLAS slots from `heaps_synced`/`make_waiting`)
        // right after `card_before` uploads dirty buffers and before `ExecuteCommandLists`.
        #[cfg(feature = "diagnostics")]
        let p_dxr = std::time::Instant::now();
        let dxr = if needs_heaps {
            crate::dxr::record_submission(self, runs, &mut lists, &mut firsts)
        } else {
            None
        };
        #[cfg(feature = "diagnostics")]
        PREP_PHASE_NS[11].fetch_add(p_dxr.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // Prepend `card_before`'s `sync` AFTER `dxr::record_submission` so `runs`
        // and `firsts` stay aligned during DXR splicing, while `sync` executes
        // FIRST ahead of `ahead_lists`, DXR ops, and title lists in one
        // `ExecuteCommandLists`.
        if let Some(sync) = card.as_ref().and_then(|c| c.sync.clone()) {
            lists.insert(0, Some(sync));
            firsts.iter_mut().for_each(|first| *first += 1);
        }
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        #[cfg(feature = "diagnostics")]
        let p_exec = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        split.part(15);
        let prepared = std::time::Instant::now();
        stage.next("submit: sending");
        // When it runs on the card, for the timeline.
        #[cfg(feature = "diagnostics")]
        let stamp = stamps::take();
        #[cfg(feature = "diagnostics")]
        if let Some((ahead, behind)) = stamp.and_then(stamps::lists) {
            lists.insert(0, Some(ahead));
            lists.push(Some(behind));
            firsts.iter_mut().for_each(|first| *first += 1);
        }
        #[cfg(feature = "diagnostics")]
        let sending = crate::timeline::ticks();
        if late {
            *submitted += 1;
            #[cfg(feature = "diagnostics")]
            if let Some([front, behind]) = self.span_lists() {
                lists.insert(0, Some(front));
                lists.push(Some(behind));
                firsts.iter_mut().for_each(|first| *first += 1);
            }
            // SAFETY: closed lists; the fence is ours. All joined parts share
            // one signal with no intervening fence wait/signal, and preludes
            // handle all cross-list texture transitions, so one batched
            // `ExecuteCommandLists` is semantically identical.
            let signalled = unsafe {
                if !lists.is_empty() {
                    self.queue.ExecuteCommandLists(&lists);
                }
                // Release `QUEUE_ORDER` as soon as the batch (holding `sync`)
                // is on the queue.
                drop(card_order.take());
                self.queue.Signal(&self.fence, *submitted)
            };
            #[cfg(feature = "diagnostics")]
            PREP_PHASE_NS[8].fetch_add(p_exec.elapsed().as_nanos() as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            let p_post = std::time::Instant::now();
            drop(card_order.take());
            awake::sent(&self.device);
            #[cfg(feature = "diagnostics")]
            crate::timeline::sent(sending, *submitted, stamp);
            self.card_sent();
            stage.next("submit: noting what is in the air");
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            #[cfg(feature = "diagnostics")]
            split.part(5);
            let back = crate::picked::with(declared, &picked.written);
            let mut texture_back_ranges = Vec::new();
            for list in runs {
                texture_back_ranges.extend(list.deferred().texture_backs.iter().map(crate::texback::Back::range));
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            #[cfg(feature = "diagnostics")]
            PREP_PHASE_NS[10].fetch_add(p_post.elapsed().as_nanos() as u64, Ordering::Relaxed);
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                own,
                texture_back_ranges,
                writes,
                dxr,
                ahead: ahead_lists,
            };
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() && !firsts.is_empty() {
            firsts
                .iter()
                .enumerate()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|((i, &a), &b)| &lists[if i == 0 { 0 } else { a }..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                drop(card_order.take());
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                #[cfg(feature = "diagnostics")]
                crate::timeline::sent(sending, *submitted, stamp);
                match waited {
                    Ok(()) => {
                        working(false);
                        #[cfg(feature = "diagnostics")]
                        let _card = crate::timeline::card_wait(Some(*submitted));
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        stage.next("submit: copying back");
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = crate::picked::with(declared, &picked.written);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        if let Some(dxr) = dxr {
            crate::dxr::finish_submission(dxr);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
                #[cfg(feature = "diagnostics")]
                text.push_str(&crate::book::at_loss(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            if dred_faults() {
                return text;
            }
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    let named = [
                        (!p.pCommandListDebugNameW.is_null()).then(|| p.pCommandListDebugNameW.to_string().unwrap_or_default()),
                        (!p.pCommandListDebugNameA.is_null()).then(|| {
                            std::ffi::CStr::from_ptr(p.pCommandListDebugNameA.cast()).to_string_lossy().into_owned()
                        }),
                    ]
                    .into_iter()
                    .flatten()
                    .next()
                    .unwrap_or_default();
                    text.push_str(&format!(" of \"{named}\""));
                    // Not one of the caller's: by the name.
                    #[cfg(feature = "diagnostics")]
                    let by_name = list_number(&named).and_then(|n| {
                        let all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                        all.get(&n).map(|(notes, pipelines, _)| (notes.upgrade(), pipelines.upgrade()))
                    });
                    #[cfg(feature = "diagnostics")]
                    let indirects = run.map(|r| r.indirects.clone()).or_else(|| {
                        let n = list_number(&named)?;
                        LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner()).get(&n)?.2.upgrade()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    #[allow(clippy::type_complexity)]
                    let by_name: Option<(Option<()>, Option<std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>>)> = None;
                    #[cfg(feature = "diagnostics")]
                    let noted = run.map(|r| r.notes.clone()).or_else(|| by_name.as_ref().and_then(|b| b.0.clone()));
                    let bound = run.map(|r| r.pipelines.clone()).or_else(|| by_name.as_ref().and_then(|b| b.1.clone()));
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = noted.map_or_else(Vec::new, |notes| {
                        let notes = notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    }
                    #[cfg(feature = "diagnostics")]
                    if let Some(indirects) = indirects {
                        text.push_str(&indirects_around(history, done as usize, &indirects));
                    }
                    // (Without markers a list that stopped midway says what
                    // it may have been at only by what it binds.)
                    if (done == 0 || markers.is_empty())
                        && let Some(bound) = bound
                    {
                        let pipelines = bound.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// 1-descriptor shader-visible CBV/SRV/UAV fallback heap for directly-indexed root signatures.
    pub(crate) fn fallback_view_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_view
            .get_or_init(|| {
                // SAFETY: plain 1-descriptor shader-visible heap + null SRV initialization.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV::default(),
                        },
                    };
                    self.device.CreateShaderResourceView(
                        None,
                        Some(&desc),
                        heap.GetCPUDescriptorHandleForHeapStart(),
                    );
                    Some(heap)
                }
            })
            .as_ref()
    }

    /// 1-descriptor shader-visible Sampler fallback heap for directly-indexed root signatures.
    pub(crate) fn fallback_sampler_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_sampler
            .get_or_init(|| {
                // SAFETY: plain 1-descriptor shader-visible sampler heap initialization.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SAMPLER_DESC {
                        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    };
                    self.device
                        .CreateSampler(&desc, heap.GetCPUDescriptorHandleForHeapStart());
                    Some(heap)
                }
            })
            .as_ref()
    }
}

impl HostList {
    #[inline]
    pub(crate) fn list1(&self) -> Option<&ID3D12GraphicsCommandList1> {
        self.list1.get_or_init(|| self.list.cast().ok()).as_ref()
    }

    #[inline]
    pub(crate) fn list6(&self) -> Option<&ID3D12GraphicsCommandList6> {
        self.list6.get_or_init(|| self.list.cast().ok()).as_ref()
    }

    #[inline]
    pub(crate) fn list7(&self) -> Option<&ID3D12GraphicsCommandList7> {
        self.list7.get_or_init(|| self.list.cast().ok()).as_ref()
    }

    /// Returns `ID3D12GraphicsCommandList7` when the host device supports
    /// Enhanced Barriers (`D3D12_FEATURE_D3D12_OPTIONS12::EnhancedBarriersSupported`).
    pub(crate) fn enhanced_barriers(&self) -> Option<&ID3D12GraphicsCommandList7> {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        let supported = *SUPPORTED.get_or_init(|| {
            let Some(host) = host() else {
                return false;
            };
            let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
            // SAFETY: a feature structure of its size.
            let known = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS12,
                    (&raw mut o).cast(),
                    size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
                )
            };
            known.is_ok() && o.EnhancedBarriersSupported.as_bool()
        });
        supported.then(|| self.list7()).flatten()
    }

    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        let _closing = crate::timeline::span("closing a host list");
        self.twins_to_depth();
        self.textures_back();
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            self.indirect_common(&scratch);
        }
        self.batches_common();
        self.card_budget_out();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// `work` finds, on a thread of ours, refreshes that belong at this
    /// place among the list's steps; they are there when the list is closed.
    pub(crate) fn refreshes_coming(&self, work: impl FnOnce() -> Vec<aside::Refresh> + Send + 'static) {
        self.take_coming();
        let at = self.deferred().copies.len();
        let coming = aside::begin(work);
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = Some((at, coming));
    }

    /// Whether asynchronous refreshes are still pending for this list.
    fn has_coming(&self) -> bool {
        self.coming.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// The refreshes that were still to come are put among the steps.
    fn take_coming(&self) {
        let Some((at, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let found = coming.wait();
        let inserted = found.len();
        if inserted == 0 {
            return;
        }
        let mut deferred = self.deferred();
        let at = at.min(deferred.copies.len());
        #[cfg(feature = "diagnostics")]
        {
            deferred.refresh_count += inserted as u64;
            deferred.refresh_bytes += found.iter().map(|(_, from, to)| *to - *from).sum::<u64>();
        }
        deferred
            .copies
            .splice(at..at, found.into_iter().map(|(shadow, from, to)| Step::Refresh(shadow, from as usize, (to - from) as usize)));
        for idx in &mut deferred.non_refreshes {
            if *idx >= at {
                *idx += inserted;
            }
        }
    }

    /// Returns any prelude held by this finished list to the global pool (or
    /// drops it if over cap). Only call after the list's last submission finished.
    pub(crate) fn release_prelude(&self) {
        if let Some(prelude) = self.prelude.lock().unwrap_or_else(|e| e.into_inner()).take() {
            recycle_prelude(prelude);
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.changed_something();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _resetting = crate::timeline::span("resetting a host list");
        self._alive.used();
        self.release_prelude();
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let indirect_clean = {
            let mut deferred = self.deferred();
            let clean = deferred.indirect_readable == 0;
            deferred.clear();
            clean
        };
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins_writing.store(false, Ordering::Relaxed);
        self.aliases.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        if dred_any() {
            self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        #[cfg(feature = "diagnostics")]
        self.indirects.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        if let Some(budget) = self.budget.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_budget_buffer(budget);
        }
        if let Some(card_budget) = self.card_budget.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_card_budget_buffer(card_budget);
        }
        if let Some((buf, cap)) = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).take()
            && indirect_clean
        {
            give_indirect_buffer(buf, cap);
        }
        {
            let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
            let left = std::mem::take(&mut args.1);
            let buffers = std::mem::take(&mut args.0);
            drop(args);
            if left == 0 {
                give_batch_buffers(buffers);
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Exclusive-access [`HostList::reset`]: skips per-field mutex locking via
    /// `.get_mut()` when the caller owns `&mut HostList` (during list recycle or
    /// title list reset).
    pub(crate) fn reset_mut(&mut self) {
        *self.epoch.get_mut() = next_epoch();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _resetting = crate::timeline::span("resetting a host list");
        self._alive.used();
        if let Some(prelude) = self.prelude.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            recycle_prelude(prelude);
        }
        *self.coming.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
        let indirect_clean = {
            let deferred = self.deferred.get_mut().unwrap_or_else(|e| e.into_inner());
            let clean = deferred.indirect_readable == 0;
            deferred.clear();
            clean
        };
        self.states.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        *self.twins_writing.get_mut() = false;
        self.aliases.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        if dred_any() {
            if let Some(pipelines) = std::sync::Arc::get_mut(&mut self.pipelines) {
                pipelines.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
            } else {
                self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some(indirects) = std::sync::Arc::get_mut(&mut self.indirects) {
            indirects.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        } else {
            self.indirects.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        #[cfg(feature = "diagnostics")]
        if let Some(notes) = std::sync::Arc::get_mut(&mut self.notes) {
            notes.get_mut().unwrap_or_else(|e| e.into_inner()).clear();
        } else {
            self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        {
            let scratch = self.scratch.get_mut().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        if let Some(budget) = self.budget.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            give_budget_buffer(budget);
        }
        if let Some(card_budget) = self.card_budget.get_mut().unwrap_or_else(|e| e.into_inner()).take() {
            give_card_budget_buffer(card_budget);
        }
        if let Some((buf, cap)) = self.indirect_args.get_mut().unwrap_or_else(|e| e.into_inner()).take()
            && indirect_clean
        {
            give_indirect_buffer(buf, cap);
        }
        {
            let args = self.batch_args.get_mut().unwrap_or_else(|e| e.into_inner());
            let left = std::mem::take(&mut args.1);
            let buffers = std::mem::take(&mut args.0);
            if left == 0 {
                give_batch_buffers(buffers);
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Something changed in this list that bringing a draw's textures and
    /// buffers up to date depends on: a texture's state, a twin, what was
    /// refreshed.
    pub(crate) fn changed_something(&self) {
        self.epoch.store(next_epoch(), Ordering::Relaxed);
    }

    /// See [`HostList::changed_something`].
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Queues a native DXR operation onto this list and schedules shadow
    /// refreshes for its guest input/postbuild buffers.
    pub(crate) fn push_dxr_op(&self, op: crate::dxr::DxrOp) {
        crate::dxr::prepare_list_op(self, &op);
        self.deferred().dxr_ops.push(op);
    }

    /// Whether any native DXR operations are queued on this list.
    pub(crate) fn has_dxr_ops(&self) -> bool {
        !self.deferred().dxr_ops.is_empty()
    }

    /// Clones the queued native DXR operations in recorded order.
    pub(crate) fn dxr_ops(&self) -> Vec<crate::dxr::DxrOp> {
        self.deferred().dxr_ops.clone()
    }

    /// Whether any deferred steps or texture state transitions have been
    /// recorded into this list since creation or its last [`HostList::reset`].
    pub(crate) fn has_recorded_work(&self) -> bool {
        let deferred = self.deferred();
        !deferred.copies.is_empty()
            || !deferred.dxr_ops.is_empty()
            || !self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let copy = crate::resource::host_copy(resource)?;
        // SAFETY: as in `crate::resource::host_resource`; the list's states
        // hold the copy from here on.
        let texture: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&copy.texture) };
        let wanted = legal_for(copy.flags(), wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let write_inc = u32::from(is_texture_write_state(wanted));
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, write_inc, copy));
                self.changed_something();
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let entry = o.get_mut();
                let before = std::mem::replace(&mut entry.1, wanted);
                entry.2 = entry.2.saturating_add(write_inc);
                if before != wanted || write_inc != 0 {
                    self.changed_something();
                }
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_for(copy.flags(), wanted);
        self.changed_something();
        let write_inc = u32::from(is_texture_write_state(wanted));
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, write_inc, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let entry = o.get_mut();
                let before = std::mem::replace(&mut entry.1, wanted);
                entry.2 = entry.2.saturating_add(write_inc);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// Records a read of format-aliased `placed` backed by `source` through
    /// `through`. Returns `true` if `source` was written earlier in this list
    /// since the last inline sync (caller must record an inline copy); `false`
    /// if the sync is handled by `plan_prelude` or already up to date.
    pub(crate) fn alias_use(
        &self,
        source: &std::sync::Arc<crate::resource::HostTexture>,
        placed: &std::sync::Arc<crate::resource::HostTexture>,
        through: &std::sync::Arc<crate::texture::Through>,
    ) -> bool {
        let src_key = std::sync::Arc::as_ptr(source) as usize;
        let src_writes = self
            .states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&src_key)
            .map_or(0, |s| s.2);
        let mut aliases = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
        let entry = aliases
            .entry(std::sync::Arc::as_ptr(placed) as usize)
            .or_insert_with(|| AliasUse {
                source: source.clone(),
                placed: placed.clone(),
                through: through.clone(),
                needs_prelude: false,
                synced_local_write: 0,
            });
        let was = (entry.needs_prelude, entry.synced_local_write);
        let inline = alias_use_step(src_writes, &mut entry.needs_prelude, &mut entry.synced_local_write);
        let changed = was != (entry.needs_prelude, entry.synced_local_write);
        drop(aliases);
        if changed {
            self.changed_something();
            self.keep(&through.buffer);
        }
        inline
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let was = (u.needs, u.behind, u.writing);
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => {
                u.writing = true;
                self.twins_writing.store(true, Ordering::Release);
            }
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        if was != (u.needs, u.behind, u.writing) || copy || colour {
            self.changed_something();
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        // Re-register if the title reused or remapped the texture's memory.
        let existing = deferred
            .back_wanted
            .iter_mut()
            .find(|target| target.owner() == resource as usize);
        match existing {
            Some(entry) if !entry.fresh() => {
                if let Some(target) = crate::texback::Target::new(resource) {
                    *entry = target;
                }
            }
            None => deferred.back_wanted.extend(crate::texback::Target::new(resource)),
            Some(_) => {}
        }
    }

    #[cfg(test)]
    pub(crate) fn back_wanted_fresh(&self, resource: crate::object::This) -> bool {
        self.deferred()
            .back_wanted
            .iter()
            .any(|t| t.owner() == resource as usize && t.fresh())
    }

    #[cfg(test)]
    pub(crate) fn texture_back_count(&self) -> usize {
        self.deferred().texture_backs.len()
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        if !self.twins_writing.load(Ordering::Acquire) {
            return;
        }
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        self.twins_writing.store(false, Ordering::Relaxed);
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
            self.changed_something();
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// Plans a list's state prelude sequentially (updating shared texture state
    /// trackers in submission order); the actual D3D12 command recording happens
    /// in [`HostList::record_prelude`], which can run in parallel across lists.
    fn plan_prelude(&self, uploads: Vec<(crate::texture::PreludeUpload, crate::texture::UploadPlan)>) -> Option<PreludePlan> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        let aliases = self.aliases.lock().unwrap_or_else(|e| e.into_inner());
        let mut alias_syncs = Vec::new();
        for u in aliases.values() {
            if u.needs_prelude {
                let wgen = u.source.write_gen.load(Ordering::Relaxed);
                if u.through.synced_gen.load(Ordering::Relaxed) != wgen {
                    alias_syncs.push((u.source.clone(), u.placed.clone(), u.through.clone()));
                    u.through.synced_gen.store(wgen, Ordering::Relaxed);
                }
            }
        }
        // Advance `write_gen` for textures written by this list, and record
        // any alias whose last inline copy already covers the final write.
        for (key, &(_, _, local_writes, ref copy)) in states.iter() {
            if local_writes > 0 {
                let new_gen = copy.write_gen.fetch_add(1, Ordering::Relaxed) + 1;
                for u in aliases.values() {
                    if std::sync::Arc::as_ptr(&u.source) as usize == *key
                        && u.synced_local_write == local_writes
                    {
                        u.through.synced_gen.store(new_gen, Ordering::Relaxed);
                    }
                }
            }
        }
        drop(aliases);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut aliasing = Vec::new();
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying: ByCopy<u32> = ByCopy::default();
        let mut restore: ByCopy<(u32, std::sync::Arc<crate::resource::HostTexture>)> = ByCopy::default();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let orig = texture.state.load(Ordering::Relaxed);
                let cur = copying.get(&key).copied().unwrap_or(orig);
                copying.insert(key, state);
                if cur != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(cur),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        restore.entry(key).or_insert((orig, texture.clone()));
                    }
                }
            }
        }
        for (source, placed, _) in &alias_syncs {
            for (texture, state) in [(source, COPY_SOURCE), (placed, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let orig = texture.state.load(Ordering::Relaxed);
                let cur = copying.get(&key).copied().unwrap_or(orig);
                copying.insert(key, state);
                if cur != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(cur),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        restore.entry(key).or_insert((orig, texture.clone()));
                    }
                }
            }
        }
        for (upload, plan) in &uploads {
            let copy = &upload.copy;
            let key = std::sync::Arc::as_ptr(copy) as usize;
            // A texture cut from a block may lie where another did: what the card
            // holds of that one is done with before the first upload into this one.
            if crate::pool::on() && plan.first_upload {
                aliasing.push(D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                            pResourceBefore: std::mem::ManuallyDrop::new(None),
                            // SAFETY: a borrowed pointer the barrier never releases.
                            pResourceAfter: unsafe { std::mem::transmute_copy(&copy.texture) },
                        }),
                    },
                });
            }
            let orig = copy.state.load(Ordering::Relaxed);
            let cur = copying.get(&key).copied().unwrap_or(orig);
            copying.insert(key, COPY_DEST);
            if cur != COPY_DEST {
                to_copy.push(transition(
                    &copy.texture,
                    host_state(cur),
                    host_state(COPY_DEST),
                ));
                if !states.contains_key(&key) {
                    restore.entry(key).or_insert((orig, copy.clone()));
                }
            }
        }
        for (key, (orig, texture)) in restore {
            if let Some(&cur) = copying.get(&key)
                && cur != orig
            {
                barriers.push(transition(&texture.texture, host_state(cur), host_state(orig)));
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, _, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if barriers.is_empty() && syncs.is_empty() && alias_syncs.is_empty() && uploads.is_empty() {
            return None;
        }
        Some(PreludePlan {
            syncs,
            alias_syncs,
            aliasing,
            to_copy,
            barriers,
            uploads,
        })
    }

    fn record_prelude(&self, plan: PreludePlan) -> Option<ID3D12CommandList> {
        let mut guard = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            *guard = host().and_then(|h| h.take_prelude());
        }
        let Some(tp) = guard.as_mut() else {
            fail(format_args!("no host list for a state prelude"));
        };
        let (allocator, list) = &tp.prelude;
        let was_open = tp.open;
        tp.open = false;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live inside `plan` and `self.states`.
        let recorded = unsafe {
            let ready = if was_open {
                Ok(())
            } else {
                allocator.Reset().and_then(|()| list.Reset(allocator, None))
            };
            ready.and_then(|()| {
                if !plan.aliasing.is_empty() {
                    list.ResourceBarrier(&plan.aliasing);
                }
                if !plan.to_copy.is_empty() {
                    list.ResourceBarrier(&plan.to_copy);
                }
                for (depth, twin, colour) in &plan.syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                for (source, placed, through) in &plan.alias_syncs {
                    crate::texture::record_alias_copy(list, &source.texture, &placed.texture, through);
                }
                for (upload, uplan) in &plan.uploads {
                    upload.record(list, uplan.sub_mask);
                }
                if !plan.barriers.is_empty() {
                    list.ResourceBarrier(&plan.barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.kept_shadows.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty()
                && deferred.texture_backs.is_empty()
                && deferred.dxr_ops.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.aliases.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes an indirect execution, while device-removed data is kept
    /// ([`indirects`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn ran_indirect(&self, pipeline: usize, name: &str, kind: u32, count: u32, user: &[u32]) {
        if dred_on() || dred_light() {
            self.indirects.lock().unwrap_or_else(|e| e.into_inner()).note(pipeline, name, kind, count, user);
        }
    }

    /// Notes a pipeline the list binds (only when DRED diagnostics are active).
    pub(crate) fn used_pipeline(&self, name: &str) {
        #[cfg(not(feature = "diagnostics"))]
        {
            let _ = name;
            return;
        }
        #[cfg(feature = "diagnostics")]
        {
            if !dred_any() {
                return;
            }
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(name) {
                pipelines.insert(name.to_owned());
            }
        }
    }

        /// The list binds a pipeline: it and its root signature are kept until
    /// the list is reset, and its name noted, the first time only (a kept
    /// pipeline's address names no other).
    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred.push_step(Step::Copy(dst as usize, src as usize, len));
        deferred.title_written();
        self.changed_something();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        let src = Self::forward_fill_in(&deferred, src as usize, len);
        deferred.push_step(Step::Fill(dst as usize, src, len, None));
    }

    /// Renderer port RT: forwards a fill behind a same-list placed copy out of
    /// that copy's upload source when safe (`crate::port_rt::forwarded`).
    fn forward_fill_in(deferred: &Deferred, src: usize, len: usize) -> usize {
        if !deferred.has_placed || len == 0 || !crate::port_rt::on() {
            return src;
        }
        let seen = deferred.copies.iter().rev().map(|step| match step {
            Step::Placed(p) => crate::port_rt::Seen::Placed { dst: p.dst, src: p.src, len: p.len, from_upload: p.from_upload() },
            &Step::Copy(d, _, n) => crate::port_rt::Seen::Writes(d, d + n),
            &Step::GpuWrite(_, a, n) => crate::port_rt::Seen::Writes(a, a + n),
            Step::Run(_) => crate::port_rt::Seen::Anything,
            _ => crate::port_rt::Seen::Nothing,
        });
        match crate::port_rt::forwarded(seen, src, len) {
            Some(from) => {
                crate::port_rt::count(crate::port_rt::FORWARDED);
                from
            }
            None => src,
        }
    }

    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        let mut deferred = self.deferred();
        let forwarded = Self::forward_fill_in(&deferred, src as usize, len as usize);
        deferred.push_step(Step::Fill(to as usize, forwarded, len as usize, None));
        deferred.filled.insert(key, gpu);
        Some(gpu)
    }

    /// As [`HostList::defer_fill`], for upload memory that only `slot` (a
    /// title heap's address, and the slot) shows.
    pub(crate) fn defer_constants(&self, dst: *mut u8, src: *const u8, len: usize, slot: (usize, u32)) {
        let mut deferred = self.deferred();
        let src = Self::forward_fill_in(&deferred, src as usize, len);
        deferred.push_step(Step::Fill(dst as usize, src, len, Some(card_steps::FillTo::Slot(slot.0, slot.1))));
    }

    /// As [`HostList::defer_fill`], for the beginning of a buffer
    /// [`HostList::upload`] made.
    pub(crate) fn defer_fill_buffer(&self, buffer: &ID3D12Resource, dst: *mut u8, src: *const u8, len: usize) {
        let to = (card_steps::stage() > 1).then(|| card_steps::FillTo::Buffer(buffer.clone()));
        let mut deferred = self.deferred();
        let src = Self::forward_fill_in(&deferred, src as usize, len);
        deferred.push_step(Step::Fill(dst as usize, src, len, to));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred.push_refresh(shadow.clone(), address, len);
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    /// Declared once until the list next writes title memory (but with
    /// `VICTORY_DECLARE_ONCE=0`): the dispatches of a frame declared the
    /// same ranges thousands of times, each a step of its own.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) || !declare_once() {
            deferred.push_step(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset (pooled in `LIST_UPLOADS` for sizes up to `LIST_UPLOAD_ITEM_MAX`).
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (chunk, alloc) = take_list_upload(host, size)?;
        let buffer = chunk.buffer.clone();
        let data = chunk.data as *mut u8;
        self.deferred().pooled_uploads.push((chunk, alloc));
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        Some((unsafe { buffer.GetGPUVirtualAddress() } + at, data))
    }

    /// [`HostList::scratch`] and the buffer it lies in, for a descriptor
    /// slot to hold: the slot outlives the list's use of it.
    pub(crate) fn scratch_held(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() } + at;
        Some((buffer, gpu, data))
    }

    /// [`HostList::scratch`]: the buffer, the offset in it and the CPU
    /// pointer.
    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            self.deferred().hold_upload(buffer.clone(), len);
            return Some((buffer, 0, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    #[allow(dead_code)]
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// Exclusive-access [`HostList::scratch_kept`], skipping the scratch mutex.
    pub(crate) fn scratch_kept_mut(&mut self) -> u64 {
        let scratch = self.scratch.get_mut().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    #[allow(dead_code)]
    pub(crate) fn budget(&self) -> Option<u64> {
        self.budget_resource().map(|(_, gpu, _)| gpu)
    }

    /// The host resource, GPU virtual address, and prior-use flag of the list's
    /// indirect-work budget, zeroed at each submission of the list from the
    /// first use after a reset on. Returns `budget_dirty = true` from the
    /// second clamp on the same recording onward so callers can emit a
    /// resource-scoped UAV barrier on the budget buffer.
    pub(crate) fn budget_resource(&self) -> Option<(ID3D12Resource, u64, bool)> {
        let (buffer, data) = {
            let mut slot = self.budget.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = take_budget_buffer();
            }
            slot.clone()?
        };
        let card = if crate::indirect::budget_enabled() && card_budget() {
            let mut slot = self.card_budget.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = self.make_card_budget();
            }
            slot.clone()
        } else {
            None
        };
        let mut deferred = self.deferred();
        let was_used = deferred.budget_used;
        deferred.budget_used = true;
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; the records refused, the
            // largest cost; with the card's copy, as
            // the list's end copied them here): taken, then zeroed.
            let notes = Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                let (refused, biggest) = (words.add(3).read_volatile(), words.add(4).read_volatile());
                BUDGET_NOTES[0].fetch_add(u64::from(refused), Ordering::Relaxed);
                BUDGET_NOTES[1].fetch_max(u64::from(biggest), Ordering::Relaxed);
                for k in 0..5 {
                    words.add(k).write_volatile(0);
                }
            });
            deferred.push_step(Step::Host(notes, (0, 0)));
            drop(deferred);
            if let Some(card) = &card {
                self.zero_card_budget(card);
            }
        }
        let res = card.unwrap_or(buffer);
        // SAFETY: a live buffer.
        let gpu = unsafe { res.GetGPUVirtualAddress() };
        Some((res, gpu, was_used))
    }

    /// Whether a global UAV barrier has already flushed all prior shader/copy
    /// UAV writes on this recording with no intervening UAV write since.
    pub(crate) fn is_uav_clean(&self) -> bool {
        self.deferred().uav_clean
    }

    /// Marks that a global UAV barrier has flushed all prior UAV writes on this
    /// recording.
    pub(crate) fn mark_uav_clean(&self) {
        let mut deferred = self.deferred();
        deferred.uav_clean = true;
        deferred.graphics_uav_dirty = false;
    }

    /// Marks that a shader dispatch, UAV-capable draw, or GPU buffer copy may
    /// have written a UAV since the last global UAV barrier on this recording.
    pub(crate) fn mark_uav_dirty(&self) {
        self.deferred().uav_clean = false;
    }

    /// Whether a graphics draw that can store to buffers ran since the last
    /// global UAV sync on this recording.
    pub(crate) fn is_graphics_uav_dirty(&self) -> bool {
        self.deferred().graphics_uav_dirty
    }

    /// Marks that a graphics draw capable of storing to buffers ran on this
    /// recording.
    pub(crate) fn mark_graphics_uav_dirty(&self) {
        let mut deferred = self.deferred();
        deferred.uav_clean = false;
        deferred.graphics_uav_dirty = true;
    }

    /// Records a global `NULL` UAV barrier on this list unless redundant
    /// consecutive global UAV barrier coalescing (`VICTORY_COALESCE_UAV_BARRIERS`,
    /// on by default) sees that no UAV-capable work ran since the last global
    /// UAV barrier on this recording. Returns `true` if a barrier was recorded.
    pub(crate) fn emit_global_uav_barrier(&self) -> bool {
        if coalesce_uav_barriers() {
            let mut deferred = self.deferred();
            if deferred.uav_clean {
                return false;
            }
            deferred.uav_clean = true;
            deferred.graphics_uav_dirty = false;
        } else {
            let mut deferred = self.deferred();
            deferred.uav_clean = true;
            deferred.graphics_uav_dirty = false;
        }
        // SAFETY: an open list; a null-resource UAV barrier is always valid.
        unsafe { self.list.ResourceBarrier(&[crate::indirect::uav_barrier()]) };
        true
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !crate::indirect::budget_enabled() || !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        take_card_budget_buffer()
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        if !crate::indirect::budget_enabled() {
            return;
        }
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8, 12, 16].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        if let Some(list7) = crate::indirect::enhanced_clamp_list(self) {
            let post = [crate::indirect::enhanced_buffer_barrier(
                card,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            )];
            // SAFETY: an open list; `card` stays in COMMON under Enhanced Barriers.
            unsafe {
                list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
                list7.Barrier(&[D3D12_BARRIER_GROUP {
                    Type: D3D12_BARRIER_TYPE_BUFFER,
                    NumBarriers: 1,
                    Anonymous: D3D12_BARRIER_GROUP_0 {
                        pBufferBarriers: post.as_ptr(),
                    },
                }]);
            }
            return;
        }
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !crate::indirect::budget_enabled() || !self.deferred().budget_zeroed {
            return;
        }
        let card = self.card_budget.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mapped = self.budget.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|(b, _)| b.clone());
        let (Some(card), Some(mapped)) = (card, mapped) else {
            return;
        };
        if let Some(list7) = crate::indirect::enhanced_clamp_list(self) {
            let pre = [crate::indirect::enhanced_buffer_barrier(
                &card,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
            )];
            // SAFETY: an open list; both buffers stay in COMMON under Enhanced Barriers.
            unsafe {
                list7.Barrier(&[D3D12_BARRIER_GROUP {
                    Type: D3D12_BARRIER_TYPE_BUFFER,
                    NumBarriers: 1,
                    Anonymous: D3D12_BARRIER_GROUP_0 {
                        pBufferBarriers: pre.as_ptr(),
                    },
                }]);
                self.list.CopyBufferRegion(&mapped, 4, &card, 4, 16);
            }
            return;
        }
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                &card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(&mapped, 4, &card, 4, 16);
            self.list.ResourceBarrier(&[transition(
                &card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and pooled
    /// globally across resets/drops. Smaller buffers replaced mid-list stay
    /// alive until [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let (buffer, actual_cap) = take_indirect_buffer(alloc)?;
        let old = slot.replace((buffer.clone(), actual_cap));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Where the next batch of clamped records goes: a dedicated device-local
    /// buffer per batch (starting in `COMMON` at offset 0) so consecutive
    /// batches never drain prior `ExecuteIndirect` reads via
    /// `INDIRECT_ARGUMENT -> UNORDERED_ACCESS` transitions.
    pub(crate) fn batch_place(&self) -> Option<(ID3D12Resource, u32, D3D12_RESOURCE_STATES)> {
        use crate::indirect::batch::{JOBS, SLOT};
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let index = args.1 as usize;
        if index == args.0.len() {
            let size = u64::from(JOBS * SLOT).next_multiple_of(64 << 10);
            let made = take_batch_buffer(size)?;
            args.0.push(made);
        }
        args.1 += 1;
        Some((args.0[index].clone(), 0, D3D12_RESOURCE_STATE_COMMON))
    }

    /// Before the list closes: returns batch scratch buffers to `COMMON` when
    /// legacy barriers were used. Under Enhanced Barriers the buffers never
    /// leave `COMMON`, so no closing barriers are needed.
    fn batches_common(&self) {
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let used = std::mem::take(&mut args.1) as usize;
        if used == 0 || crate::indirect::enhanced_clamp_list(self).is_some() {
            return;
        }
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = args.0[..used]
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON))
            .collect();
        // SAFETY: an open list and live buffers this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&barriers) };
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted. Under Enhanced Barriers `scratch`
    /// stays in `COMMON`, so no closing legacy transition is needed.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        if crate::indirect::enhanced_clamp_list(self).is_some() {
            return;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // SAFETY: an open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Keeps a host buffer copy (including any backing placed heap or card twin)
    /// alive until the list is reset.
    pub(crate) fn keep_shadow(&self, shadow: std::sync::Arc<crate::shadow::Shadow>) {
        let mut deferred = self.deferred();
        if !deferred
            .kept_shadows
            .iter()
            .rev()
            .take(16)
            .any(|s| std::sync::Arc::ptr_eq(s, &shadow))
        {
            deferred.kept_shadows.push(shadow);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Whether `key` was asked about ([`HostList::first_use`]) since the
    /// reset.
    pub(crate) fn used(&self, key: usize) -> bool {
        self.deferred().textures.contains(&key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.push_step(Step::Run(Box::new(work)));
        deferred.title_written();
        drop(deferred);
        self.changed_something();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().push_step(Step::Host(Box::new(work), (0, 0)));
    }

    /// As [`HostList::defer_host`], for work that reads `len` bytes of title
    /// memory at `from`.
    pub(crate) fn defer_host_read(&self, from: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.deferred().push_step(Step::Host(Box::new(work), (from, from + len)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted and records its upload on the prelude if changed, keeping
    /// both staging and host texture alive until the list is reset.
    pub(crate) fn defer_texture(&self, upload: crate::texture::PreludeUpload) {
        self.deferred().push_step(Step::Texture(upload));
    }

    /// Detiles the textures the list uploads now, ahead of its submission
    /// ([`crate::texture::Staging::refresh_ahead`]); how many were left.
    pub(crate) fn refresh_textures_ahead(&self) -> u64 {
        let textures: Vec<_> = self
            .deferred()
            .copies
            .iter()
            .filter_map(|step| match step {
                Step::Texture(upload) => Some(upload.staging.clone()),
                _ => None,
            })
            .collect();
        textures.iter().filter(|staging| !staging.refresh_ahead()).count() as u64
    }

    /// A copy source owned exclusively by this list until reset after its
    /// last execution. A different list can prepare its movie plane concurrently.
    pub(crate) fn plane_upload(
        &self,
        size: u64,
        read: (usize, usize),
        fill: impl Fn(usize) + Send + 'static,
    ) -> Option<ID3D12Resource> {
        let chunk = take_chunk(host()?, size)?;
        let source = chunk.buffer.clone();
        let destination = chunk.data;
        let mut deferred = self.deferred();
        deferred.plane_uploads.push((chunk, size));
        deferred.push_step(Step::Plane(read, Box::new(move || fill(destination))));
        Some(source)
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Copies and fills that
    /// `taken` names (by their place) are left out: they are done on the
    /// card ([`card_steps`]). Returns the textures whose host copies must be
    /// uploaded on this list's prelude, paired with whether it is their first
    /// upload.
    fn run_deferred(
        &self,
        batch: &mut Batch,
        air: &[(usize, usize)],
        taken: impl Fn(usize) -> bool,
        skip_front_refreshes: bool,
    ) -> Vec<(crate::texture::PreludeUpload, crate::texture::UploadPlan)> {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        // The time and the bytes of the copies and fills, some 13,000 a
        // frame between 20,000 refreshes: summed here and added once.
        let (mut us, mut bytes) = (std::time::Duration::ZERO, 0u64);
        #[cfg(feature = "diagnostics")]
        let (mut copies, mut refreshes, mut refreshed) = (0u64, 0u64, 0u64);
        let mut textures: Vec<crate::texture::PreludeUpload> = Vec::new();
        let (gpu_lo, gpu_hi) = crate::shadow::gpu_only_bounds();
        let wide_fills_on = wide_fills();
        let parallel_tex = parallel_refresh();
        let send_own = send::own();
        let deferred = self.deferred();
        let copies_slice = &deferred.copies;
        let non_refreshes = &deferred.non_refreshes;
        let mut nr_pos = 0usize;
        let mut linear_pos = 0usize;
        let mut using_fast = skip_front_refreshes && batch.front_intact;
        #[cfg(feature = "diagnostics")]
        let started_fast = using_fast;
        loop {
            let i = if using_fast {
                if !batch.front_intact {
                    using_fast = false;
                    #[cfg(feature = "diagnostics")]
                    for s in &copies_slice[..linear_pos] {
                        if let Step::Refresh(_, _, len) = s {
                            refreshes += 1;
                            refreshed += *len as u64;
                        }
                    }
                    continue;
                }
                let Some(&idx) = non_refreshes.get(nr_pos) else {
                    #[cfg(feature = "diagnostics")]
                    if started_fast {
                        refreshes += deferred.refresh_count;
                        refreshed += deferred.refresh_bytes;
                    }
                    break;
                };
                nr_pos += 1;
                linear_pos = idx + 1;
                idx
            } else {
                if linear_pos >= copies_slice.len() {
                    break;
                }
                let idx = linear_pos;
                linear_pos += 1;
                idx
            };
            let step = &copies_slice[i];
            if matches!(step, Step::Copy(..) | Step::Fill(..)) && taken(i) {
                continue;
            }
            let (started, copied) = match step {
                &Step::Copy(dst, src, len) => {
                    let started = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    let dst_end = dst.saturating_add(len);
                    batch.fill_overlapping(dst, dst_end);
                    batch.title_written_range(dst, dst_end);
                    crate::port::copied(dst, src, len);
                    if gpu_hi != 0
                        && len > 0
                        && (src as u64) < gpu_hi
                        && (src as u64).saturating_add(len as u64) > gpu_lo
                    {
                        batch.cpu_reads.push((src, len));
                    }
                    // Renderer port #7: into a heap's records, the rounds
                    // asked earlier did not see it.
                    if crate::descriptor::in_shader_heap(dst, len) {
                        crate::port_journal::prep_wrote();
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    // What the card holds there may be another value although
                    // ours did not change.
                    if send_own {
                        crate::shadow::send_copied(dst, len);
                    }
                    (started, len)
                }
                &Step::Fill(dst, src, len, _) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    crate::port::filled(src, len);
                    if gpu_hi != 0
                        && len > 0
                        && (src as u64) < gpu_hi
                        && (src as u64).saturating_add(len as u64) > gpu_lo
                    {
                        batch.cpu_reads.push((src, len));
                    }
                    if wide_fills_on {
                        batch.push_fill(dst, src, len);
                        bytes += len as u64;
                        continue;
                    }
                    let started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    (started, len)
                }
                Step::Placed(p) => {
                    let started = std::time::Instant::now();
                    let free = !touches(air, p.src, p.src + p.len) && !touches(air, p.dst, p.dst + p.len);
                    if free {
                        if crate::descriptor::in_shader_heap(p.dst, p.len) {
                            crate::port_journal::prep_wrote();
                        }
                        let dst_end = p.dst.saturating_add(p.len);
                        batch.fill_overlapping(p.dst, dst_end);
                        batch.title_written_range(p.dst, dst_end);
                    } else if crate::descriptor::in_shader_heap(p.dst, p.len) {
                        crate::port_journal::prep_wrote();
                    }
                    if let Some(bytes) = p.prepare(free) {
                        batch.placed.push((p.shadow.clone(), p.dst, bytes));
                    }
                    (started, p.len)
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        refreshes += 1;
                        refreshed += *len as u64;
                    }
                    if !(skip_front_refreshes && batch.front_intact) {
                        batch.add(shadow, *address, *len);
                    }
                    continue;
                }
                Step::Run(work) | Step::Host(work, _) => {
                    batch.fill();
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                        // (It may write anywhere, a heap's records too.)
                        crate::port_journal::prep_wrote();
                    } else if !batch.due_textures.is_empty() || !batch.due.is_empty() {
                        // Host callbacks can use a refresh result, including the
                        // mip mask that maps a reserved texture for this upload.
                        batch.check_textures();
                        batch.refresh_textures();
                    }
                    let _step = crate::timeline::span("a step of ours");
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    batch.note_gpu_write(shadow, *address, *len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(upload) => {
                    let (at, len) = upload.staging.memory();
                    if gpu_hi != 0
                        && len > 0
                        && (at as u64) < gpu_hi
                        && (at as u64).saturating_add(len as u64) > gpu_lo
                    {
                        batch.cpu_reads.push((at, len));
                    }
                    if batch.texture_due(&upload.staging) {
                        if parallel_tex {
                            // Checked with the others of the batch, several
                            // at a time, before the next step that writes
                            // title memory ([`Batch::check_textures`]).
                            batch.due_textures.push(upload.staging.clone());
                        } else {
                            batch.due.push(upload.staging.clone());
                        }
                    }
                    textures.push(upload.clone());
                    continue;
                }
                Step::Plane((at, len), fill) => {
                    if gpu_hi != 0
                        && *len > 0
                        && (*at as u64) < gpu_hi
                        && (*at as u64).saturating_add(*len as u64) > gpu_lo
                    {
                        batch.cpu_reads.push((*at, *len));
                    }
                    let started = std::time::Instant::now();
                    fill();
                    PREPARING[2].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    PREPARING[3].fetch_add(*len as u64, Ordering::Relaxed);
                    continue;
                }
            };
            us += started.elapsed();
            bytes += copied as u64;
        }
        drop(deferred);
        if !batch.cpu_reads.is_empty() {
            crate::shadow::note_cpu_reads(&batch.cpu_reads);
            batch.cpu_reads.clear();
        }
        PREPARING[0].fetch_add(us.as_micros() as u64, Ordering::Relaxed);
        PREPARING[1].fetch_add(bytes, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            add(C::Copies, copies);
            add(C::CopyBytes, bytes);
            add(C::Refreshes, refreshes);
            add(C::RefreshBytes, refreshed);
            SPLIT[7].fetch_add(us.as_nanos() as u64, Ordering::Relaxed);
        }
        // Finish refreshing any textures not already refreshed in `refresh_front`
        // (or invalidated by a title write), then claim the ones whose staging
        // buffer changed so `self.prelude` uploads them on this exact submission.
        batch.check_textures();
        batch.refresh_textures();
        let mut uploads = Vec::new();
        for upload in textures {
            if let Some(plan) = upload.finish_refresh() {
                uploads.push((upload, plan));
            }
        }
        uploads
    }
}

/// What `VICTORY_LIST_REUSE` says: 0 off, 1 on (default, matching `ab86ef7` + `c96525d`), 2 in turn (`alt`).
fn list_reuse_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LIST_REUSE").as_deref() {
        Ok("0") => 0,
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 1,
    })
}

/// Whether a list keeps a pipeline it binds once (`HostList::bind_pipeline`)
/// and copies a repeated unaligned root view's title memory into upload memory
/// once per bind (`HostList::fill_scratch`): on by default (`VICTORY_LIST_REUSE=0`
/// disables reuse; `alt`: twenty seconds on, twenty seconds off).
pub(crate) fn list_reuse() -> bool {
    match list_reuse_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a list keeps a pipeline once and copies a repeated root view once from now (VICTORY_LIST_REUSE=alt): {}",
                    if on { "ON" } else { "off, at every bind as usual" }
                );
            }
            on
        }
    }
}

/// The numbers of [`HostList::epoch`].
fn next_epoch() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A table by a copy's address.
pub(crate) type ByCopy<T> = std::collections::HashMap<usize, T, std::hash::BuildHasherDefault<AddressHash>>;

/// A hash for addresses of ours: the bits that tell them apart, spread. The
/// tables it serves are asked 40,000 times a frame, and hold no one's keys
/// but our own.
#[derive(Default)]
pub(crate) struct AddressHash(u64);

impl std::hash::Hasher for AddressHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_usize(usize::from(b));
        }
    }
    fn write_usize(&mut self, n: usize) {
        let h = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
    fn write_u64(&mut self, n: u64) {
        self.write_usize(n as usize);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_usize(n as usize);
    }
}

/// Since the last frame-rate line, nanoseconds of the submitting thread's
/// preparing: the write-watch asking for a submission; the title's bytes put
/// in under flights' writes; the lists' steps, the refreshes among them; the
/// snapshots; what goes to copies in the card's memory; the lists sent; and
/// of the steps: the refreshes, and the copies and fills (those with the
/// refreshes a copy of title memory flushes). Of the asking, which ranges it
/// is of. Of the refreshes, those before ranges are given to shaders, one
/// copy at a time: nanoseconds, and how many. Ahead of all that: the wait for
/// the submission lock, the steps done on the card planned ([`card_steps`]),
/// the flights waited for, the heaps' records and what is swept; and, at the
/// end, the wait for the heaps' records.
#[cfg(feature = "diagnostics")]
static SPLIT: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// Since the last frame-rate line: flushes of refreshes, their refreshes,
/// those known to copy nothing, and those done on several threads.
#[cfg(feature = "diagnostics")]
static FLUSHES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Counts into [`SPLIT`].
#[cfg(feature = "diagnostics")]
struct Split(std::cell::Cell<std::time::Instant>);

#[cfg(feature = "diagnostics")]
impl Split {
    fn begin() -> Split {
        Split(std::cell::Cell::new(std::time::Instant::now()))
    }

    /// Nanoseconds since the last part.
    fn so_far(&self) -> u64 {
        self.0.get().elapsed().as_nanos() as u64
    }

    /// What came since the last part was part `i`.
    fn part(&self, i: usize) {
        let now = std::time::Instant::now();
        SPLIT[i].fetch_add((now - self.0.replace(now)).as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The split of the preparing, per frame, for the frame-rate line.
#[cfg(feature = "diagnostics")]
fn split_line(frames: f64) {
    let ms: Vec<f64> = SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let f: Vec<f64> = FLUSHES.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let e: Vec<f64> = crate::watch::early_asks().iter().map(|&n| n as f64 / frames).collect();
    let a: Vec<f64> = crate::watch::asks().iter().map(|&n| n as f64 / frames).collect();
    let h: Vec<f64> = crate::draw::HEAP_SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let s: Vec<f64> = crate::shadow::send_stats().iter().map(|&n| n as f64 / frames).collect();
    tracing::info!(
        "preparing per frame, the submitting thread: {:.1} ms asking the write watch ({:.1} of them finding what to ask), {:.1} putting the title's bytes under flights' \
         writes, {:.1} in the lists' steps (refreshes {:.1} in {:.0} flushes of {:.0}, {:.0} of them known to copy nothing, {:.0} on \
         several threads, {:.1} one at a time in {:.0} before shader writes; copies and fills {:.1}, the refreshes before copies among them), {:.1} snapshots, {:.1} for copies in the \
         card's memory, {:.1} sending the lists (VICTORY_LIGHT_REFRESH is {}); ahead of these {:.1} waiting for the submission lock, {:.1} \
         planning steps done on the card, {:.1} waiting for flights, {:.1} the heaps' records and what is swept; at the end {:.1} waiting \
         for the heaps' records; asked early by a thread of its own {:.0} pieces of 2 MiB, {:.1} ms (VICTORY_ASK_EARLY)",
        ms[0],
        ms[8],
        ms[1],
        ms[2],
        ms[6],
        f[0],
        f[1],
        f[2],
        f[3],
        ms[9],
        ms[10] * 1e6,
        ms[7],
        ms[3],
        ms[4],
        ms[5],
        if crate::watch::light() { "on" } else { "off" },
        ms[11],
        ms[12],
        ms[13],
        ms[14],
        ms[15],
        e[0],
        e[1] / 1e3,
    );
    tracing::info!(
        "asking about title memory per frame, all threads: {:.0} kernel askings over {:.0} MB ({:.1} ms, {:.0} pages of 4 KiB found \
         written, {:.0} askings refused); {:.0} questions about holes; view ranges: {:.0} answered by their \
         bits, {:.0} by the guards whole, {:.0} block by block; {:.0} blocks made read-only, {:.0} opened by ourselves; {:.1} ms \
         waiting for the guards, {:.1} ms changing protections; pages taken for the card: {:.1} MB compared, {:.1} MB of them \
         as they were, {:.1} MB to go up in {:.0} runs (VICTORY_SEND_CHANGED)",
        a[0],
        a[1] / 1024.0,
        a[11] / 1e6,
        a[12],
        a[10],
        a[9],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7] / 1e6,
        a[8] / 1e6,
        s[0] / 1048576.0,
        s[1] / 1048576.0,
        s[2] / 1048576.0,
        s[3],
    );
    tracing::info!(
        "the pass over the heap's views per frame: {:.1} ms looking at slots, {:.1} their copies' spans and refreshes, {:.1} textures; the \
         first two beside the recording (VICTORY_HEAP_PASS_ASIDE), {:.1} ms waited for at the lists' close",
        h[0],
        h[1],
        h[2],
        aside::WAITED.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
    );
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=0` disables it; `=1` forces it).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_refresh_enabled(
            std::env::var("VICTORY_WIDE_REFRESH").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_refresh_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: ByCopy<usize>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// mapped to their title-memory `[at, at + len)` span so range-scoped
    /// title writes only evict overlapping textures.
    textures: ByCopy<(usize, usize)>,
    /// Those due ([`Batch::refresh_textures`]).
    due: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// With [`parallel_refresh`]: textures due a check, not checked yet
    /// ([`Batch::check_textures`], before any write of ours and at every
    /// flush).
    due_textures: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// What copies at their place in the submission's lists put into title
    /// memory ([`place`]): the host copy, the title address, the bytes.
    placed: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, Vec<u8>)>,
    /// Fills not done yet (destination, source, bytes): [`wide_fills`].
    fills: Vec<(usize, usize, usize)>,
    /// Bounding box `[min_src, max_src_end]` of pending `fills` (`fills_hi == 0`
    /// when empty) so `fill_overlapping` skips scanning in O(1) when disjoint.
    fills_lo: usize,
    fills_hi: usize,
    /// Batched `(src, len)` CPU reads within `GPU_ONLY_BOUNDS` flushed once per
    /// list instead of acquiring `GPU_ONLY_WATCHED.read()` per step.
    cpu_reads: Vec<(usize, usize)>,
    /// Copies (by address) brought up to date ahead of the steps, and the
    /// range (from, to) ([`Batch::refresh_ahead`]).
    ahead: ByCopy<(usize, usize)>,
    /// `ahead` holds everything the submission refreshes
    /// ([`Batch::refresh_front`]).
    front: bool,
    /// `true` while front spans remain intact (cleared on overlapping title writes).
    front_intact: bool,
    /// Bounding box `[min_from, max_to)` of front spans when `all_covered`
    /// omitted populating `ahead`, so `title_written_range` still clears
    /// `front_intact` on any overlapping title write.
    ahead_omitted: Option<(usize, usize)>,
    /// Unique `(shadow_ptr, address, len)` GPU writes declared in this submission,
    /// avoiding repeated `pending`/`flights` locking and eliminating a second
    /// pass over all lists for `declared_writes`.
    declared_seen: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    declared_places: ByCopy<usize>,
    declared: Vec<Declared>,
}

/// Whether the copies that a submission's shaders write are brought up to
/// date together, on several threads, ahead of its steps
/// (`VICTORY_REFRESH_AHEAD=0`: each before the step that gives it to them).
///
/// 580 copies a frame, one after the other on the submitting thread, 3.4 to
/// 6.5 ms (session play-1125-0203).
fn ahead_of_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_AHEAD").is_none_or(|v| v != "0") && wide_refresh() && crate::watch::light())
}

/// Whether everything a submission refreshes is brought up to date in one go,
/// on several threads, ahead of its steps (`VICTORY_REFRESH_FRONT=1`, and with
/// `VICTORY_JOIN_PARTS`, which needs it; else in groups where the steps ask
/// for it). Measured on its own at nothing (run submit-1: refreshes 17.1 ms a
/// frame without, 18.3 with).
///
/// A submission's refreshes were done in up to five groups one after the other
/// (the copies its shaders write, those between its steps, what is left),
/// each ending with the threads waiting for the last copy; and the submitting
/// thread went over all the steps once for each question it has (what to ask
/// about, what flights may write too, what shaders write). A refresh may come
/// earlier than its step: the title's bytes are the same until a step writes
/// title memory, and behind such a step everything is asked for again, as it
/// was.
fn refresh_front() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ahead_of_writes() && (joins_asked().0 > 1 || *ON.get_or_init(|| crate::switch::on("VICTORY_REFRESH_FRONT")))
}

/// Per copy what is refreshed in front of a submission's steps, as
/// [`written_spans`]'s but of all copies, up to the copy's first write by
/// shaders in each part of the submission (`Touch::Part`: another begins).
/// What a part refreshes of a copy it has given to its shaders stays where it
/// stands: around what they write. What a later part refreshes is the
/// title's bytes of before either.
fn front_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    let mut at: ByCopy<usize> = ByCopy::default();
    // (Copy, from, to), in the order they come, and the part that gave it to
    // shaders last.
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut given: Vec<u32> = Vec::new();
    let (mut part, mut last) = (1, usize::MAX);
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                // (Most come several times on end.)
                let i = if spans.get(last).is_some_and(|s| s.0 == copy) {
                    last
                } else {
                    *at.entry(copy).or_insert_with(|| {
                        spans.push((copy, usize::MAX, 0));
                        given.push(0);
                        spans.len() - 1
                    })
                };
                last = i;
                if given[i] != part {
                    let span = &mut spans[i];
                    (span.1, span.2) = (span.1.min(from), span.2.max(to));
                }
            }
            Touch::Write(copy) => {
                let i = *at.entry(copy).or_insert_with(|| {
                    spans.push((copy, usize::MAX, 0));
                    given.push(0);
                    spans.len() - 1
                });
                given[i] = part;
            }
            Touch::Part => part += 1,
            Touch::Title | Touch::Nothing => {}
        }
    }
    spans.retain(|s| s.1 < s.2);
    spans
}

/// Steps of a submission's lists from which [`plan_lists_parallel`] scans each
/// list on the worker pool.
const WIDE_PLAN_STEPS: usize = 256;

const COPY_WRITTEN: u8 = 1 << 0;
const COPY_ASK: u8 = 1 << 1;
const COPY_STALE: u8 = 1 << 2;

/// One copy's folded touches within a single command list.
struct CopyFront {
    key: usize,
    pre_from: usize,
    pre_to: usize,
    all_from: usize,
    all_to: usize,
    flags: u8,
    shadow: Option<std::sync::Arc<crate::shadow::Shadow>>,
}

/// One list's folded touches for front planning, `ask_once`, and
/// `stale_under_flights`, gathered in a single pass over `list.deferred().copies`.
#[derive(Default)]
struct ListFront {
    copies: Vec<CopyFront>,
    /// True if any `Step::Refresh` in this list followed a `Step::GpuWrite` of
    /// the same copy, so `run_deferred` cannot skip `Step::Refresh` via `front_intact`.
    has_post_write_refresh: bool,
    /// The textures to bring up to date, in order.
    textures: Vec<std::sync::Arc<crate::texture::Staging>>,
}

/// Result of planning and joining all lists in a submission ahead of `ask_once`.
struct PreparedFront {
    spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    covered: Vec<bool>,
    textures: Vec<std::sync::Arc<crate::texture::Staging>>,
}

/// Whether parallel list-front scanning is allowed (enabled whenever
/// `wide_refresh()` is on, unless explicitly disabled via `VICTORY_PORT_GUARD_PLAN=0`).
fn plan_parallel_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    wide_refresh() && *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_GUARD_PLAN").is_none_or(|v| v != "0"))
}

/// A list's steps folded per copy ([`ListFront`]). Uses a reusable thread-local
/// `ByCopy` scratch table to avoid per-list HashMap allocations and rehashes,
/// and checks `asked_ahead()` / `writes_pending()` once per unique copy.
fn list_front(steps: &[Step], textures: bool, collect_ask: bool, collect_stale: bool) -> ListFront {
    thread_local! {
        static FRONT_AT: std::cell::RefCell<ByCopy<usize>> = std::cell::RefCell::new(ByCopy::default());
    }
    FRONT_AT.with_borrow_mut(|at| {
        at.clear();
        let mut copies: Vec<CopyFront> = Vec::with_capacity((steps.len() / 2).min(512));
        let mut staged: Vec<std::sync::Arc<crate::texture::Staging>> = Vec::new();
        let mut last = usize::MAX;
        let mut has_post_write_refresh = false;
        for step in steps {
            match step {
                Step::Refresh(shadow, from, len) => {
                    let key = std::sync::Arc::as_ptr(shadow) as usize;
                    let to = *from + *len;
                    let i = if copies.get(last).is_some_and(|c| c.key == key) {
                        last
                    } else {
                        *at.entry(key).or_insert_with(|| {
                            let mut flags = 0u8;
                            if collect_ask && shadow.asked_ahead() {
                                flags |= COPY_ASK;
                            }
                            if collect_stale && shadow.writes_pending() {
                                flags |= COPY_STALE;
                            }
                            copies.push(CopyFront {
                                key,
                                pre_from: usize::MAX,
                                pre_to: 0,
                                all_from: usize::MAX,
                                all_to: 0,
                                flags,
                                shadow: None,
                            });
                            copies.len() - 1
                        })
                    };
                    last = i;
                    let c = &mut copies[i];
                    if c.flags & COPY_WRITTEN == 0 {
                        if c.shadow.is_none() {
                            c.shadow = Some(shadow.clone());
                        }
                        (c.pre_from, c.pre_to) = (c.pre_from.min(*from), c.pre_to.max(to));
                    } else {
                        has_post_write_refresh = true;
                        if c.flags & COPY_STALE != 0 && c.shadow.is_none() {
                            c.shadow = Some(shadow.clone());
                        }
                    }
                    if c.flags & (COPY_ASK | COPY_STALE) != 0 {
                        (c.all_from, c.all_to) = (c.all_from.min(*from), c.all_to.max(to));
                    }
                }
                Step::GpuWrite(shadow, ..) => {
                    let key = std::sync::Arc::as_ptr(shadow) as usize;
                    let i = if copies.get(last).is_some_and(|c| c.key == key) {
                        last
                    } else {
                        *at.entry(key).or_insert_with(|| {
                            let mut flags = 0u8;
                            if collect_ask && shadow.asked_ahead() {
                                flags |= COPY_ASK;
                            }
                            if collect_stale && shadow.writes_pending() {
                                flags |= COPY_STALE;
                            }
                            copies.push(CopyFront {
                                key,
                                pre_from: usize::MAX,
                                pre_to: 0,
                                all_from: usize::MAX,
                                all_to: 0,
                                flags,
                                shadow: None,
                            });
                            copies.len() - 1
                        })
                    };
                    last = i;
                    copies[i].flags |= COPY_WRITTEN;
                }
                Step::Texture(upload) if textures => {
                    staged.push(upload.staging.clone());
                }
                _ => {}
            }
        }
        ListFront {
            copies,
            has_post_write_refresh,
            textures: staged,
        }
    })
}

/// Plans [`ListFront`] for every list in `runs` in parallel when the submission
/// is large enough to amortize waking the worker pool.
fn plan_lists_parallel(runs: &[&HostList], textures: bool, collect_ask: bool, collect_stale: bool) -> Vec<ListFront> {
    let steps: usize = runs.iter().map(|list| list.deferred().copies.len()).sum();
    if plan_parallel_on() && runs.len() >= 2 && steps >= WIDE_PLAN_STEPS {
        #[derive(Clone, Copy)]
        struct Lists(*const *const HostList, usize);
        // SAFETY: `runs` outlives `wide_each`, and each list's deferred steps are behind its lock.
        unsafe impl Send for Lists {}
        unsafe impl Sync for Lists {}
        let lists: Vec<*const HostList> = runs.iter().map(|&list| std::ptr::from_ref(list)).collect();
        let at = Lists(lists.as_ptr(), lists.len());
        let fronts: Vec<std::sync::Mutex<Option<ListFront>>> =
            (0..runs.len()).map(|_| std::sync::Mutex::new(None)).collect();
        crate::shadow::wide_each(runs.len(), |n| {
            let at = at;
            // SAFETY: `n < at.1`, `lists` outlives `wide_each`.
            let list = unsafe { &**at.0.add(n) };
            debug_assert!(n < at.1);
            let front = list_front(&list.deferred().copies, textures, collect_ask, collect_stale);
            *fronts[n].lock().unwrap_or_else(|e| e.into_inner()) = Some(front);
        });
        drop(lists);
        fronts
            .into_iter()
            .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()).unwrap_or_default())
            .collect()
    } else {
        runs.iter()
            .map(|list| list_front(&list.deferred().copies, textures, collect_ask, collect_stale))
            .collect()
    }
}

/// Plans all lists in `runs` and fuses `join_fronts`, `ask_once` ranges, and
/// `stale_under_flights` spans into a single pass using a reusable thread-local
/// scratch map (eliminating redundant HashMaps per submission).
fn plan_and_join_fronts(
    runs: &[&HostList],
    parts: &[usize],
    textures: bool,
    collect_stale: bool,
) -> (PreparedFront, Vec<(usize, usize)>, Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>) {
    let mut fronts = plan_lists_parallel(runs, textures, true, collect_stale);
    if fronts.len() == 1 {
        let front = fronts.pop().unwrap();
        let covered = vec![!front.has_post_write_refresh];
        let mut spans = Vec::with_capacity(front.copies.len());
        let mut ask_ranges = Vec::with_capacity(front.copies.len());
        let mut stale_spans = Vec::new();
        for c in front.copies {
            let has_pre = c.pre_from < c.pre_to;
            let has_ask = c.flags & COPY_ASK != 0 && c.all_from < c.all_to;
            let has_stale = collect_stale && c.flags & COPY_STALE != 0 && c.all_from < c.all_to;
            if has_ask {
                ask_ranges.push((c.all_from, c.all_to));
            }
            if has_pre && has_stale {
                if let Some(shadow) = c.shadow {
                    stale_spans.push((shadow.clone(), c.all_from, c.all_to));
                    spans.push((shadow, c.pre_from, c.pre_to));
                }
            } else if has_pre {
                if let Some(shadow) = c.shadow {
                    spans.push((shadow, c.pre_from, c.pre_to));
                }
            } else if has_stale {
                if let Some(shadow) = c.shadow {
                    stale_spans.push((shadow, c.all_from, c.all_to));
                }
            }
        }
        return (
            PreparedFront {
                spans,
                covered,
                textures: front.textures,
            },
            ask_ranges,
            stale_spans,
        );
    }
    struct MergedCopy {
        front_from: usize,
        front_to: usize,
        given: u32,
        ask_from: usize,
        ask_to: usize,
        stale_from: usize,
        stale_to: usize,
        shadow: Option<std::sync::Arc<crate::shadow::Shadow>>,
    }
    thread_local! {
        static JOIN_AT: std::cell::RefCell<ByCopy<usize>> = std::cell::RefCell::new(ByCopy::default());
    }
    JOIN_AT.with_borrow_mut(|at| {
        at.clear();
        let mut merged: Vec<MergedCopy> = Vec::with_capacity(fronts.iter().map(|f| f.copies.len()).sum::<usize>().min(4096));
        let mut covered = Vec::with_capacity(fronts.len());
        let mut all_textures = Vec::new();
        let mut part = 1u32;
        for (n, front) in fronts.into_iter().enumerate() {
            if parts.contains(&n) {
                part += 1;
            }
            all_textures.extend(front.textures);
            let mut all_covered = !front.has_post_write_refresh;
            for c in front.copies {
                let i = *at.entry(c.key).or_insert_with(|| {
                    merged.push(MergedCopy {
                        front_from: usize::MAX,
                        front_to: 0,
                        given: 0,
                        ask_from: usize::MAX,
                        ask_to: 0,
                        stale_from: usize::MAX,
                        stale_to: 0,
                        shadow: None,
                    });
                    merged.len() - 1
                });
                let m = &mut merged[i];
                if m.shadow.is_none() && c.shadow.is_some() {
                    m.shadow = c.shadow;
                }
                if c.pre_from < c.pre_to {
                    if m.given != part {
                        (m.front_from, m.front_to) = (m.front_from.min(c.pre_from), m.front_to.max(c.pre_to));
                    } else {
                        all_covered = false;
                    }
                }
                if c.flags & COPY_WRITTEN != 0 {
                    m.given = part;
                }
                if c.flags & COPY_ASK != 0 && c.all_from < c.all_to {
                    (m.ask_from, m.ask_to) = (m.ask_from.min(c.all_from), m.ask_to.max(c.all_to));
                }
                if collect_stale && c.flags & COPY_STALE != 0 && c.all_from < c.all_to {
                    (m.stale_from, m.stale_to) = (m.stale_from.min(c.all_from), m.stale_to.max(c.all_to));
                }
            }
            covered.push(all_covered);
        }
        let mut spans = Vec::with_capacity(merged.len());
        let mut ask_ranges = Vec::with_capacity(merged.len());
        let mut stale_spans = Vec::new();
        for m in merged {
            if m.ask_from < m.ask_to {
                ask_ranges.push((m.ask_from, m.ask_to));
            }
            let has_front = m.front_from < m.front_to;
            let has_stale = collect_stale && m.stale_from < m.stale_to;
            if has_front && has_stale {
                if let Some(shadow) = m.shadow {
                    stale_spans.push((shadow.clone(), m.stale_from, m.stale_to));
                    spans.push((shadow, m.front_from, m.front_to));
                }
            } else if has_front {
                if let Some(shadow) = m.shadow {
                    spans.push((shadow, m.front_from, m.front_to));
                }
            } else if has_stale {
                if let Some(shadow) = m.shadow {
                    stale_spans.push((shadow, m.stale_from, m.stale_to));
                }
            }
        }
        (
            PreparedFront {
                spans,
                covered,
                textures: all_textures,
            },
            ask_ranges,
            stale_spans,
        )
    })
}

/// Touches of one list, per copy in the order of their first touch: the span
/// of its refreshes before its first write, and whether it is written.
#[cfg(test)]
fn fold_touches(touches: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize, bool)> {
    fold_touches_checked(touches).0
}

#[cfg(test)]
fn fold_touches_checked(touches: impl Iterator<Item = Touch>) -> (Vec<(usize, usize, usize, bool)>, bool) {
    let mut at: ByCopy<usize> = ByCopy::default();
    let mut copies: Vec<(usize, usize, usize, bool)> = Vec::new();
    let mut last = usize::MAX;
    let mut has_post_write_refresh = false;
    for touch in touches {
        let (key, range) = match touch {
            Touch::Refresh(key, from, to) => (key, Some((from, to))),
            Touch::Write(key) => (key, None),
            _ => continue,
        };
        // (Most come several times on end.)
        let i = if copies.get(last).is_some_and(|c| c.0 == key) {
            last
        } else {
            *at.entry(key).or_insert_with(|| {
                copies.push((key, usize::MAX, 0, false));
                copies.len() - 1
            })
        };
        last = i;
        let c = &mut copies[i];
        match range {
            Some((from, to)) if !c.3 => (c.1, c.2) = (c.1.min(from), c.2.max(to)),
            Some(_) => has_post_write_refresh = true,
            None => c.3 = true,
        }
    }
    (copies, has_post_write_refresh)
}

/// [`front_spans`] of the lists whose parts are `lists` ([`fold_touches`]), a
/// new part of the submission beginning at each of `parts`: the same spans,
/// in the same order (`front_tests`).
#[cfg(test)]
fn join_keys<'a>(lists: impl Iterator<Item = &'a [(usize, usize, usize, bool)]>, parts: &[usize]) -> Vec<(usize, usize, usize)> {
    join_keys_covered(lists, parts).0
}

#[cfg(test)]
fn join_keys_covered<'a>(
    lists: impl Iterator<Item = &'a [(usize, usize, usize, bool)]>,
    parts: &[usize],
) -> (Vec<(usize, usize, usize)>, Vec<bool>) {
    let mut at: ByCopy<usize> = ByCopy::default();
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut given: Vec<u32> = Vec::new();
    let mut covered: Vec<bool> = Vec::new();
    let mut part = 1;
    for (n, copies) in lists.enumerate() {
        if parts.contains(&n) {
            part += 1;
        }
        let mut all_covered = true;
        for &(key, from, to, written) in copies {
            let i = *at.entry(key).or_insert_with(|| {
                spans.push((key, usize::MAX, 0));
                given.push(0);
                spans.len() - 1
            });
            // What the list refreshes before it writes the copy counts unless
            // an earlier list of this part gave the copy to shaders.
            if from < to {
                if given[i] != part {
                    (spans[i].1, spans[i].2) = (spans[i].1.min(from), spans[i].2.max(to));
                } else {
                    all_covered = false;
                }
            }
            if written {
                given[i] = part;
            }
        }
        covered.push(all_covered);
    }
    spans.retain(|s| s.1 < s.2);
    (spans, covered)
}

/// [`join_keys`] of the lists' parts, with their copies and per-list coverage flags.
fn join_fronts(
    mut fronts: Vec<ListFront>,
    parts: &[usize],
) -> (Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>, Vec<bool>) {
    if fronts.len() == 1 {
        let front = fronts.pop().unwrap();
        let covered = vec![!front.has_post_write_refresh];
        let mut spans = Vec::with_capacity(front.copies.len());
        for c in front.copies {
            if c.pre_from < c.pre_to {
                if let Some(shadow) = c.shadow {
                    spans.push((shadow, c.pre_from, c.pre_to));
                }
            }
        }
        return (spans, covered);
    }
    thread_local! {
        static JOIN_AT: std::cell::RefCell<ByCopy<usize>> = std::cell::RefCell::new(ByCopy::default());
    }
    JOIN_AT.with_borrow_mut(|at| {
        at.clear();
        let mut merged: Vec<(usize, usize, u32, Option<std::sync::Arc<crate::shadow::Shadow>>)> = Vec::new();
        let mut covered = Vec::with_capacity(fronts.len());
        let mut part = 1u32;
        for (n, front) in fronts.into_iter().enumerate() {
            if parts.contains(&n) {
                part += 1;
            }
            let mut all_covered = !front.has_post_write_refresh;
            for c in front.copies {
                let i = *at.entry(c.key).or_insert_with(|| {
                    merged.push((usize::MAX, 0, 0, None));
                    merged.len() - 1
                });
                let m = &mut merged[i];
                if m.3.is_none() && c.shadow.is_some() {
                    m.3 = c.shadow;
                }
                if c.pre_from < c.pre_to {
                    if m.2 != part {
                        (m.0, m.1) = (m.0.min(c.pre_from), m.1.max(c.pre_to));
                    } else {
                        all_covered = false;
                    }
                }
                if c.flags & COPY_WRITTEN != 0 {
                    m.2 = part;
                }
            }
            covered.push(all_covered);
        }
        let spans = merged
            .into_iter()
            .filter_map(|(from, to, _, shadow)| (from < to).then_some((shadow?, from, to)))
            .collect();
        (spans, covered)
    })
}

#[cfg(test)]
mod front_tests {
    use super::{Touch, fold_touches, front_spans, join_keys};

    /// Random submissions of lists (refreshes and shaders' writes of a few
    /// copies, new parts now and then): each list folded on its own and the
    /// folds joined in order give what the whole submission's touches give,
    /// span for span, in the same order (renderer port #6's planning).
    #[test]
    fn lists_folded_apart_and_joined_give_the_submissions_spans() {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for _ in 0..3000 {
            let lists = 1 + next(8);
            let copies = 1 + next(6);
            let mut parts = Vec::new();
            let mut whole: Vec<Touch> = Vec::new();
            let mut folds = Vec::new();
            for n in 0..lists {
                if n > 0 && next(4) == 0 {
                    parts.push(n);
                    whole.push(Touch::Part);
                }
                let mut mine: Vec<Touch> = Vec::new();
                for _ in 0..next(12) {
                    let key = 0x1000 * (1 + next(copies));
                    if next(5) == 0 {
                        mine.push(Touch::Write(key));
                        whole.push(Touch::Write(key));
                    } else {
                        let from = next(100);
                        let to = from + 1 + next(50);
                        mine.push(Touch::Refresh(key, from, to));
                        whole.push(Touch::Refresh(key, from, to));
                    }
                }
                folds.push(fold_touches(mine.into_iter()));
            }
            let want = front_spans(whole.into_iter());
            let got = join_keys(folds.iter().map(|f| f.as_slice()), &parts);
            assert_eq!(got, want, "parts {parts:?}");
        }
    }

    #[test]
    fn card_compare_hlsl_compiles_cleanly() {
        let code = super::compile(
            super::CARD_COMPARE,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("CARD_COMPARE HLSL must compile cleanly for cs_5_1");
        assert!(!code.is_empty());
    }

    #[test]
    fn card_compare_wide_hlsl_compiles_cleanly() {
        let code = super::compile(
            super::CARD_COMPARE_WIDE,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("CARD_COMPARE_WIDE HLSL must compile cleanly for cs_5_1");
        assert!(!code.is_empty());
    }

    #[test]
    fn coalesce_uav_barriers_default_on_and_opt_out() {
        use std::ffi::OsStr;
        assert!(super::coalesce_uav_barriers_for(None));
        assert!(super::coalesce_uav_barriers_for(Some(OsStr::new("1"))));
        assert!(!super::coalesce_uav_barriers_for(Some(OsStr::new("0"))));
    }

    #[test]
    fn texture_write_state_and_alias_use_step_track_dirty_writes() {
        // Read-only states do not count as writes; write states do.
        assert!(!super::is_texture_write_state(0x40 | 0x80)); // SHADER_RESOURCE
        assert!(!super::is_texture_write_state(0x800)); // COPY_SOURCE
        assert!(!super::is_texture_write_state(0x20)); // DEPTH_READ
        assert!(super::is_texture_write_state(0x4)); // RENDER_TARGET
        assert!(super::is_texture_write_state(0x8)); // UNORDERED_ACCESS
        assert!(super::is_texture_write_state(0x10)); // DEPTH_WRITE
        assert!(super::is_texture_write_state(0x400)); // COPY_DEST
        assert!(super::is_texture_write_state(0x1000)); // RESOLVE_DEST

        // Read before any local write -> defers to prelude, no inline copy.
        let (mut needs_prelude, mut synced) = (false, 0u32);
        assert!(!super::alias_use_step(0, &mut needs_prelude, &mut synced));
        assert!(needs_prelude);
        assert_eq!(synced, 0);
        // Repeated read before any write -> still no inline copy.
        assert!(!super::alias_use_step(0, &mut needs_prelude, &mut synced));
        // Local write occurs (src_writes == 1) -> triggers inline copy once.
        assert!(super::alias_use_step(1, &mut needs_prelude, &mut synced));
        assert_eq!(synced, 1);
        // Repeated read after SRV transition (src_writes still 1) -> no redundant copy.
        assert!(!super::alias_use_step(1, &mut needs_prelude, &mut synced));
        // Second local write (src_writes == 2) -> triggers inline copy again.
        assert!(super::alias_use_step(2, &mut needs_prelude, &mut synced));
        assert_eq!(synced, 2);
    }

    #[test]
    fn batch_compare_groups_merges_disjoint_and_splits_overlaps() {
        // Items: (shadow_key, mode, first_piece, pieces, at)
        let items = [
            (1usize, 2u32, 0u32, 2u32, 0u32),   // shadow 1, pieces 0..2, at 0
            (1, 2, 5, 1, 2),                    // shadow 1, disjoint piece 5, contiguous at 2 -> merges
            (1, 2, 10, 3, 3),                   // shadow 1, disjoint pieces 10..13, at 3 -> merges
            (1, 2, 12, 2, 6),                   // shadow 1, overlaps piece 12 (< 13) -> splits!
            (1, 1, 20, 1, 8),                   // shadow 1, different mode (1 vs 2) -> splits!
            (2, 2, 0, 4, 9),                    // shadow 2 -> splits!
            (2, 2, 8, 2, 14),                   // shadow 2, non-contiguous at (14 != 13) -> splits!
        ];
        assert_eq!(
            super::batch_compare_groups(&items),
            vec![0..3, 3..4, 4..5, 5..6, 6..7]
        );
    }
}

/// What a step is to the copies' refreshes, for [`written_spans`].
enum Touch {
    /// A refresh of a copy (by address), from, to.
    Refresh(usize, usize, usize),
    /// The copy is given to shaders.
    Write(usize),
    /// Title memory is written.
    Title,
    Nothing,
    /// Another part of the submission begins.
    Part,
}

/// Of `steps`, up to the first that writes title memory: the copies given to
/// shaders, each with the range its refreshes before that cover.
fn written_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    // (From, to, given to shaders.)
    let mut spans: ByCopy<(usize, usize, bool)> = ByCopy::default();
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                let span = spans.entry(copy).or_insert((from, to, false));
                if !span.2 {
                    (span.0, span.1) = (span.0.min(from), span.1.max(to));
                }
            }
            // (Not refreshed before: nothing of it is due.)
            Touch::Write(copy) => {
                if let Some(span) = spans.get_mut(&copy) {
                    span.2 = true;
                }
            }
            Touch::Title => break,
            Touch::Nothing | Touch::Part => {}
        }
    }
    spans.into_iter().filter(|(_, s)| s.2).map(|(copy, (from, to, _))| (copy, from, to)).collect()
}

#[cfg(test)]
mod written_spans_tests {
    use super::{Touch, front_spans, split, written_spans};

    #[test]
    fn everything_is_refreshed_in_front_but_what_a_part_gave_its_shaders() {
        use Touch::{Nothing, Part, Refresh, Title, Write};
        let found = front_spans(
            [
                Refresh(1, 10, 20),
                Refresh(1, 12, 25),
                Refresh(2, 0, 5),
                Nothing,
                Write(1),
                // (Around what the part's own shaders write.)
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Title,
                Refresh(4, 7, 9),
                Part,
                // The next part's bytes under the first one's writes.
                Refresh(1, 30, 40),
                Write(4),
                Refresh(4, 0, 9),
            ]
            .into_iter(),
        );
        assert_eq!(found, [(1, 10, 40), (2, 0, 5), (4, 7, 9)]);
    }

    #[test]
    fn a_submission_is_cut_between_its_lists() {
        // Lists c and e are in neither half's order: they go with the first.
        let Ok([first, second]) = split(vec!['a', 'b', 'c', 'd', 'e'], vec![1, 3, 0], 1) else {
            panic!("no list is named twice");
        };
        let run = |(lists, order): &(Vec<char>, Vec<usize>)| order.iter().map(|&i| lists[i]).collect::<String>();
        assert_eq!((run(&first), run(&second)), ("b".into(), "da".into()));
        assert_eq!(first.0.len() + second.0.len(), 5);
        assert!(split(vec!['a', 'b'], vec![0, 1, 0], 2).is_err());
        assert!(split(vec!['a', 'b'], vec![0, 1], 0).is_err());
    }

    #[test]
    fn what_is_written_is_refreshed_ahead_up_to_its_write_and_to_title_writes() {
        use Touch::{Nothing, Refresh, Title, Write};
        let mut found = written_spans(
            [
                Refresh(1, 10, 20),
                Refresh(2, 0, 5),
                Nothing,
                Refresh(1, 30, 40),
                Write(1),
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Refresh(4, 7, 9),
                Write(4),
                Title,
                Write(2),
            ]
            .into_iter(),
        );
        found.sort_unstable();
        assert_eq!(found, [(1, 10, 40), (4, 7, 9)]);
    }

    #[test]
    fn title_written_range_only_evicts_overlapping_entries() {
        let mut batch = super::Batch::default();
        batch.front = true;
        batch.front_intact = true;
        batch.ahead.insert(1, (0x1000, 0x2000));
        batch.ahead.insert(2, (0x4000, 0x5000));
        batch.textures.insert(10, (0x1000, 0x1800));
        batch.textures.insert(20, (0x6000, 0x7000));

        // Disjoint write leaves `ahead`, `textures`, and `front_intact` untouched.
        batch.title_written_range(0x2000, 0x4000);
        assert!(batch.front_intact);
        assert_eq!(batch.ahead.len(), 2);
        assert_eq!(batch.textures.len(), 2);

        // Write overlapping only entry 1 and texture 10 evicts only those.
        batch.title_written_range(0x1500, 0x1600);
        assert!(!batch.front_intact);
        assert!(!batch.ahead.contains_key(&1));
        assert!(batch.ahead.contains_key(&2));
        assert!(!batch.textures.contains_key(&10));
        assert!(batch.textures.contains_key(&20));

        // When `all_covered` omitted `ahead`, `title_written_range` must still
        // clear `front_intact` on overlapping title writes.
        let mut omitted = super::Batch::default();
        omitted.front = true;
        omitted.front_intact = true;
        omitted.ahead_omitted = Some((0x1000, 0x5000));
        omitted.title_written_range(0x5000, 0x6000);
        assert!(omitted.front_intact, "disjoint write keeps omitted front intact");
        omitted.title_written_range(0x2000, 0x2100);
        assert!(!omitted.front_intact, "overlapping write must clear omitted front_intact");
        assert!(omitted.ahead_omitted.is_none());

        // `fill_overlapping` keeps disjoint fills queued and flushes overlapping ones.
        let src = [0xabu8; 32];
        let mut dst = [0u8; 32];
        let s_addr = src.as_ptr() as usize;
        let d_addr = dst.as_mut_ptr() as usize;
        batch.push_fill(d_addr, s_addr, src.len());
        batch.fill_overlapping(s_addr + 64, s_addr + 128);
        assert_eq!(batch.fills.len(), 1, "disjoint write should not flush fills");
        batch.fill_overlapping(s_addr + 16, s_addr + 24);
        assert!(batch.fills.is_empty(), "overlapping write must flush fills");
        assert_eq!(dst, src);
    }

    #[test]
    fn dirty_span_indices_orders_largest_first_in_serial_and_parallel_modes() {
        let mut backing = vec![0u8; 4096];
        let ptr = backing.as_mut_ptr();
        let make = |addr: u64, len: u64| crate::shadow::Shadow::unbacked(addr, ptr, len, false);
        // Spread spans across distinct 64 MiB pieces (unsorted order) with varying lengths.
        let spans: Vec<_> = (0..40usize)
            .map(|i| {
                let piece = (i * 7) % 19;
                let base = (piece + 1) * (64 << 20) + i * 4096;
                let len = ((i % 10) + 1) * 4096;
                (make(base as u64, len as u64), base, base + len)
            })
            .collect();
        for parallel in [false, true] {
            let dirty = super::Batch::dirty_span_indices(&spans, parallel);
            assert_eq!(dirty.len(), spans.len());
            // Verify descending span-length ordering (LPT scheduling).
            for w in dirty.windows(2) {
                let len_a = spans[w[0]].2 - spans[w[0]].1;
                let len_b = spans[w[1]].2 - spans[w[1]].1;
                assert!(len_a >= len_b, "expected largest-first order: {len_a} < {len_b}");
            }
            let mut sorted = dirty;
            sorted.sort_unstable();
            assert_eq!(sorted, (0..spans.len()).collect::<Vec<_>>());
        }
    }
}

/// Whether the fills of a submission's upload memory are done together on
/// several threads (`VICTORY_WIDE_FILLS=0`: one after the other, where the
/// lists name them).
///
/// They are 7,000 to 12,000 a frame, some 500 bytes each, 5 ms of the
/// submitting thread. Each reads title memory and writes memory of its own
/// that the card alone reads, when the lists run; they are done before the
/// next step that writes title memory or runs code of ours, and before the
/// submission goes on from its steps.
fn wide_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_FILLS").is_none_or(|v| v != "0") && wide_refresh())
}

/// Whether a submission's textures are brought up to date on several
/// threads (`VICTORY_WIDE_TEXTURES=0`: one after the other, where the lists
/// name them, as before).
///
/// They are read, hashed and, if they changed, detiled into their staging
/// buffers: 5 ms a frame of the submitting thread, for some 16 MB. Each has
/// its own buffer, notes and lock, and is refreshed by the title's own
/// threads as it is. They are done before the next step that writes title
/// memory, as the copies' refreshes are, and before the lists go to the
/// card. What is new: the note of a list that its upload takes what the
/// buffer then holds (`texture.rs`, `dirty`) may come ahead of the refresh
/// an earlier list of the submission asked for, not behind it. The texture
/// is then uploaded by a later list too, once more than needed.
fn wide_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_TEXTURES").is_none_or(|v| v != "0") && wide_refresh())
}

trait TextureSpan {
    fn title_span(&self) -> (usize, usize);
}

impl TextureSpan for crate::texture::Staging {
    fn title_span(&self) -> (usize, usize) {
        let (at, len) = self.memory();
        (at, at.saturating_add(len))
    }
}

#[cfg(test)]
impl TextureSpan for u32 {
    fn title_span(&self) -> (usize, usize) {
        (0, 0)
    }
}

#[cfg(test)]
impl TextureSpan for Vec<u8> {
    fn title_span(&self) -> (usize, usize) {
        (0, 0)
    }
}

#[cfg(test)]
impl<const N: usize> TextureSpan for [u8; N] {
    fn title_span(&self) -> (usize, usize) {
        (0, 0)
    }
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if self.ahead.get(&key).is_some_and(|&(from, to)| from <= address && end <= to) {
            return;
        }
        match self.index.get(&key) {
            Some(&at) => {
                let r = &mut self.refreshes[at];
                (r.1, r.2) = (r.1.min(address), r.2.max(end));
            }
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Declares a shader write once per unique `(shadow, address, len)` in this
    /// submission, calling `shadow.writing` and recording it for `card_before`.
    fn note_gpu_write(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if !self.declared_seen.insert((key, address, len)) {
            return;
        }
        shadow.writing(address, len);
        let i = *self.declared_places.entry(key).or_insert_with(|| {
            self.declared.push((shadow.clone(), Vec::new()));
            self.declared.len() - 1
        });
        self.declared[i].1.push((address, address + len));
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        // From here on it is refreshed around what the shaders write. (What
        // is in front of all the steps was, up to here.)
        if !self.front && !self.ahead.is_empty() {
            self.ahead.remove(&key);
        }
        if self.index.is_empty() {
            return;
        }
        let Some(at) = self.index.remove(&key) else {
            return;
        };
        let (shadow, from, to) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        if !(crate::watch::light() && shadow.nothing_new(from, to - from)) {
            Self::update(&shadow, from, to);
        }
        #[cfg(feature = "diagnostics")]
        {
            let ns = started.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
            SPLIT[10].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let _one = crate::timeline::span("a copy's refresh");
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }

    /// Parallel-flush variant of [`Batch::update`]: wall time is accounted once
    /// by the caller, avoiding per-span `Instant::now` and `PREPARING[0]` churn.
    fn update_parallel(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _one = crate::timeline::span("a copy's refresh");
        let copied = shadow.update(from, to - from);
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
        if copied != 0 {
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed along with its title-memory span).
    fn texture_due<T: TextureSpan>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        let key = std::sync::Arc::as_ptr(staging) as usize;
        if self.textures.contains_key(&key) {
            return false;
        }
        self.textures.insert(key, staging.title_span());
        true
    }

    /// Queues a deferred upload fill and updates the pending source bounding box.
    #[inline]
    fn push_fill(&mut self, dst: usize, src: usize, len: usize) {
        if self.fills.is_empty() {
            self.fills_lo = src;
            self.fills_hi = src.saturating_add(len);
        } else {
            self.fills_lo = self.fills_lo.min(src);
            self.fills_hi = self.fills_hi.max(src.saturating_add(len));
        }
        self.fills.push((dst, src, len));
    }

    /// Flushes queued `fills` only if any fill reads from `[dst, end)`.
    fn fill_overlapping(&mut self, dst: usize, end: usize) {
        if self.fills.is_empty() || dst >= end || self.fills_hi <= dst || self.fills_lo >= end {
            return;
        }
        if self.fills.iter().any(|&(_, src, len)| src < end && src.saturating_add(len) > dst) {
            self.fill();
        }
    }

    /// Before a step that writes a known title range `[dst, end)`: flushes and
    /// invalidates only the pending refreshes, front spans, and textures whose
    /// title memory actually overlaps `[dst, end)`.
    fn title_written_range(&mut self, dst: usize, end: usize) {
        if dst >= end {
            return;
        }
        if !self.due_textures.is_empty()
            && self.due_textures.iter().any(|s| {
                let (at, len) = s.memory();
                at < end && at.saturating_add(len) > dst
            })
        {
            self.check_textures();
        }
        if !self.due.is_empty()
            && self.due.iter().any(|s| {
                let (at, len) = s.memory();
                at < end && at.saturating_add(len) > dst
            })
        {
            self.refresh_textures();
        }
        if !self.textures.is_empty() {
            self.textures.retain(|_, &mut (at, to)| to <= dst || at >= end);
        }
        if !self.refreshes.is_empty() {
            let overlapping: Vec<std::sync::Arc<crate::shadow::Shadow>> = self
                .refreshes
                .iter()
                .filter(|(_, from, to)| *from < end && *to > dst)
                .map(|(shadow, _, _)| shadow.clone())
                .collect();
            for shadow in overlapping {
                self.flush_one(&shadow);
            }
        }
        if let Some((lo, hi)) = self.ahead_omitted {
            if lo < end && hi > dst {
                self.front_intact = false;
                self.ahead_omitted = None;
            }
        } else if !self.ahead.is_empty() {
            let before = self.ahead.len();
            self.ahead.retain(|_, &mut (from, to)| to <= dst || from >= end);
            if self.ahead.len() < before {
                self.front_intact = false;
            }
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.check_textures();
        self.flush();
        self.ahead.clear();
        self.ahead_omitted = None;
        self.front_intact = false;
        self.textures.clear();
    }

    /// Finishes [`Batch::refresh_front`] using the pre-joined [`PreparedFront`]
    /// from the fused `watch::once()` planning pass.
    fn refresh_front_prepared(&mut self, front: PreparedFront) -> Vec<bool> {
        let planning = crate::timeline::span("refreshes in front: the spans");
        let mut due = Vec::new();
        for staging in front.textures {
            if self.texture_due(&staging) {
                due.push(staging);
            }
        }
        let all_covered = front.covered.iter().all(|&c| c);
        self.refresh_spans(front.spans, due, all_covered, planning);
        front.covered
    }

    /// Before the steps of `runs` ([`refresh_front`]), a new part of the
    /// submission beginning at each of `parts`. Returns per-list flags
    /// indicating whether every `Step::Refresh` in that list was covered.
    fn refresh_front(&mut self, runs: &[&HostList], parts: &[usize]) -> Vec<bool> {
        let mut planning = crate::timeline::span("refreshes in front: the steps gathered");
        let mut due = Vec::new();
        let steps: usize = runs.iter().map(|list| list.deferred().copies.len()).sum();
        if plan_parallel_on() && runs.len() >= 2 && steps >= WIDE_PLAN_STEPS {
            let fronts = plan_lists_parallel(runs, wide_textures(), false, false);
            planning.next("refreshes in front: the spans");
            for front in &fronts {
                for staging in &front.textures {
                    if self.texture_due(staging) {
                        due.push(staging.clone());
                    }
                }
            }
            let (spans, covered) = join_fronts(fronts, parts);
            let all_covered = covered.iter().all(|&c| c);
            self.refresh_spans(spans, due, all_covered, planning);
            return covered;
        }
        let mut copies: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
        let mut touches = Vec::new();
        let mut written_in_part: ByCopy<()> = ByCopy::default();
        let mut covered = Vec::with_capacity(runs.len());
        for (n, list) in runs.iter().enumerate() {
            if parts.contains(&n) {
                touches.push(Touch::Part);
                written_in_part.clear();
            }
            let mut last = 0;
            let mut all_covered = true;
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        if written_in_part.contains_key(&key) {
                            all_covered = false;
                        }
                        if key != last {
                            copies.entry(key).or_insert_with(|| shadow.clone());
                            last = key;
                        }
                        touches.push(Touch::Refresh(key, *at, *at + *len));
                    }
                    Step::GpuWrite(shadow, ..) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        written_in_part.insert(key, ());
                        touches.push(Touch::Write(key));
                    }
                    Step::Texture(upload) if wide_textures() => {
                        if self.texture_due(&upload.staging) {
                            due.push(upload.staging.clone());
                        }
                    }
                    _ => {}
                }
            }
            covered.push(all_covered);
        }
        planning.next("refreshes in front: the spans");
        let spans: Vec<_> = front_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.remove(&key)?, from, to)))
            .collect();
        let all_covered = covered.iter().all(|&c| c);
        self.refresh_spans(spans, due, all_covered, planning);
        covered
    }

    /// Filters `spans` down to the indices that may have new writes in the
    /// journal, grouping spans by 64 MiB journal piece (`from >> 26`) so
    /// `JournalScan`'s 1-shard read lock stays hot across consecutive spans,
    /// then orders dirty indices largest-span-first so heavy copies start first.
    fn dirty_span_indices(
        spans: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)],
        parallel: bool,
    ) -> Vec<usize> {
        if spans.is_empty() {
            return Vec::new();
        }
        const SORT_MIN: usize = 16;
        let scan_slice = |offset: usize, slice: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]| -> Vec<usize> {
            let mut out = Vec::new();
            let j = crate::watch::journal_read();
            let mut scan = j.scan();
            if slice.len() < SORT_MIN {
                for (i, (shadow, from, to)) in slice.iter().enumerate() {
                    if !shadow.nothing_new_with_scan(&mut scan, *from, *to - *from) {
                        out.push(offset + i);
                    }
                }
            } else {
                thread_local! {
                    static ORDER: std::cell::RefCell<Vec<(usize, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
                }
                ORDER.with_borrow_mut(|order| {
                    order.clear();
                    order.extend(slice.iter().enumerate().map(|(i, &(_, from, _))| (from >> 26, i as u32)));
                    order.sort_unstable_by_key(|&(piece, _)| piece);
                    for &(_, i) in order.iter() {
                        let i = i as usize;
                        let (shadow, from, to) = &slice[i];
                        if !shadow.nothing_new_with_scan(&mut scan, *from, *to - *from) {
                            out.push(offset + i);
                        }
                    }
                });
            }
            out
        };
        let mut dirty = if parallel {
            let workers = crate::shadow::wide_worker_count();
            let chunks = workers.min(spans.len().div_ceil(128)).max(1);
            let chunk_len = spans.len().div_ceil(chunks);
            let buckets: Vec<std::sync::Mutex<Vec<usize>>> =
                (0..chunks).map(|_| std::sync::Mutex::new(Vec::new())).collect();
            crate::shadow::wide_each(chunks, |c| {
                let start = c * chunk_len;
                let end = (start + chunk_len).min(spans.len());
                if start < end {
                    let local = scan_slice(start, &spans[start..end]);
                    if !local.is_empty() {
                        *buckets[c].lock().unwrap_or_else(|e| e.into_inner()) = local;
                    }
                }
            });
            let mut parts: Vec<Vec<usize>> = buckets
                .into_iter()
                .map(|b| b.into_inner().unwrap_or_else(|e| e.into_inner()))
                .collect();
            if parts.len() == 1 {
                parts.pop().unwrap()
            } else {
                let total: usize = parts.iter().map(Vec::len).sum();
                let mut all = Vec::with_capacity(total);
                for p in parts {
                    all.extend(p);
                }
                all
            }
        } else {
            scan_slice(0, spans)
        };
        if dirty.len() > 1 {
            dirty.sort_unstable_by(|&a, &b| {
                spans[b]
                    .2
                    .saturating_sub(spans[b].1)
                    .cmp(&spans[a].2.saturating_sub(spans[a].1))
            });
        }
        dirty
    }

    /// The rest of [`Batch::refresh_front`]: `spans` brought up to date in
    /// front, and the textures `due`.
    fn refresh_spans(
        &mut self,
        spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        due: Vec<std::sync::Arc<crate::texture::Staging>>,
        all_covered: bool,
        planning: crate::timeline::Span,
    ) {
        planning.end();
        self.front = true;
        self.front_intact = true;
        // When every list in `runs` is `front_covered`, `run_deferred` skips
        // `Step::Refresh` while `front_intact` holds; track the spans' bounding
        // box so `title_written_range` still clears `front_intact` on overlap.
        if all_covered {
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            for &(_, from, to) in &spans {
                lo = lo.min(from);
                hi = hi.max(to);
            }
            self.ahead_omitted = (lo < hi).then_some((lo, hi));
        } else {
            self.ahead_omitted = None;
            self.ahead.reserve(spans.len());
            for (shadow, from, to) in &spans {
                self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(spans.len() as u64, Ordering::Relaxed);
        }
        let _flush = crate::timeline::span("refreshes of copies");
        let started = std::time::Instant::now();
        let tex_before = PREPARING[2].load(Ordering::Relaxed);
        // Filter known-clean spans in piece/shard order first so dirty spans run
        // at 1-span granularity (largest first) across workers and clean flushes
        // never wake the worker pool.
        const FEW: usize = 512;
        let parallel_scan = spans.len() > FEW;
        let dirty = if !spans.is_empty() {
            let _known = crate::timeline::span("refreshes in front: those known to copy nothing");
            let dirty = Self::dirty_span_indices(&spans, parallel_scan);
            #[cfg(feature = "diagnostics")]
            {
                FLUSHES[2].fetch_add((spans.len() - dirty.len()) as u64, Ordering::Relaxed);
                if parallel_scan {
                    FLUSHES[3].fetch_add(spans.len() as u64, Ordering::Relaxed);
                } else if !due.is_empty() || !dirty.is_empty() {
                    FLUSHES[3].fetch_add(dirty.len() as u64, Ordering::Relaxed);
                }
            }
            dirty
        } else {
            Vec::new()
        };
        let ahead = dirty.len() > 1 && !crate::watch::once();
        if ahead {
            let ranges: Vec<(usize, usize)> =
                dirty.iter().map(|&idx| &spans[idx]).filter(|r| r.0.asked_ahead()).map(|r| (r.1, r.2)).collect();
            crate::watch::ask_ahead(&ranges);
        }
        if !due.is_empty() || !dirty.is_empty() {
            crate::shadow::wide_each(due.len() + dirty.len(), |i| {
                let Some(di) = i.checked_sub(due.len()) else {
                    due[i].refresh();
                    return;
                };
                let (shadow, from, to) = &spans[dirty[di]];
                Self::update_parallel(shadow, *from, *to);
            });
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
        // The time counted is the time it took, not the threads' sum.
        let took = started.elapsed().as_micros() as u64;
        let tex_summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(tex_before);
        PREPARING[2].fetch_sub(tex_summed, Ordering::Relaxed);
        PREPARING[0].fetch_add(took, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[6].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Before the steps of `runs` ([`ahead_of_writes`]); copies and fills
    /// that `taken` names are not done on the CPU.
    fn refresh_ahead(&mut self, runs: &[&HostList], taken: impl Fn(&HostList, usize) -> bool) {
        let mut copies: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
        let mut touches = Vec::new();
        'steps: for list in runs {
            for (i, step) in list.deferred().copies.iter().enumerate() {
                touches.push(match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        copies.entry(key).or_insert_with(|| shadow.clone());
                        Touch::Refresh(key, *at, *at + *len)
                    }
                    Step::GpuWrite(shadow, ..) => Touch::Write(std::sync::Arc::as_ptr(shadow) as usize),
                    Step::Copy(..) if taken(list, i) => Touch::Nothing,
                    Step::Copy(..) | Step::Run(_) | Step::Placed(_) => break 'steps,
                    _ => Touch::Nothing,
                });
            }
        }
        let due: Vec<_> = written_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.get(&key)?.clone(), from, to)))
            .collect();
        self.ahead.reserve(due.len());
        for (shadow, from, to) in &due {
            self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        #[cfg(feature = "diagnostics")]
        let began = std::time::Instant::now();
        let started = std::time::Instant::now();
        if !due.is_empty() {
            let dirty = Self::dirty_span_indices(&due, due.len() > 512);
            crate::shadow::wide_each(dirty.len(), |di| {
                let (shadow, from, to) = &due[dirty[di]];
                Self::update_parallel(shadow, *from, *to);
            });
        }
        PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            let ns = began.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// The fills noted are done.
    fn fill(&mut self) {
        // Below this many the threads cost more than they save.
        const FEW: usize = 96;
        const PART: usize = 64;
        if self.fills.is_empty() {
            return;
        }
        self.fills_lo = usize::MAX;
        self.fills_hi = 0;
        let started = std::time::Instant::now();
        if self.fills.len() < FEW {
            for (dst, src, len) in self.fills.drain(..) {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        } else {
            let fills = std::mem::take(&mut self.fills);
            crate::shadow::wide_each(fills.len().div_ceil(PART), move |i| {
                for &(dst, src, len) in fills.iter().skip(i * PART).take(PART) {
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                }
            });
        }
        let took = started.elapsed();
        PREPARING[0].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[7].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }

    /// The textures due are brought up to date.
    fn refresh_textures(&mut self) {
        if self.due.len() < 2 || !wide_textures() {
            self.due.drain(..).for_each(|staging| staging.refresh());
            return;
        }
        let _textures = crate::timeline::span("textures brought up to date");
        // The time counted is the time it took, not the threads' sum.
        let started = std::time::Instant::now();
        let before = PREPARING[2].load(Ordering::Relaxed);
        let due = std::mem::take(&mut self.due);
        crate::shadow::wide_each(due.len(), move |i| due[i].refresh());
        let summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(before);
        PREPARING[2].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
    }

    /// With [`parallel_refresh`]: checks the textures due ([`Batch::due_textures`])
    /// on the helper threads (`crate::shadow::wide_run`, the submitting
    /// thread among them), all done before it returns. Each check takes
    /// its texture's own lock, the journal's for its asking, the runtime's
    /// mapping lock for reading (shared), and a buffer of its thread's own;
    /// the hashing and detiling run side by side. "textures N ms" counts
    /// the time it took, not the threads' sum.
    fn check_textures(&mut self) {
        if self.due_textures.is_empty() {
            return;
        }
        let due = std::mem::take(&mut self.due_textures);
        PARALLEL_CHECKS[0].fetch_add(1, Ordering::Relaxed);
        PARALLEL_CHECKS[1].fetch_add(due.len() as u64, Ordering::Relaxed);
        if due.len() < 2 {
            for staging in due {
                staging.refresh();
            }
            return;
        }
        let started = std::time::Instant::now();
        let before = PREPARING[2].load(Ordering::Relaxed);
        let jobs: Vec<Box<dyn FnOnce() + Send>> = due
            .into_iter()
            .map(|staging| Box::new(move || staging.refresh()) as Box<dyn FnOnce() + Send>)
            .collect();
        crate::shadow::wide_run(jobs);
        let summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(before);
        let wall = started.elapsed().as_micros() as u64;
        PREPARING[2].fetch_sub(summed.saturating_sub(wall), Ordering::Relaxed);
        PARALLEL_CHECKS[2].fetch_add(summed, Ordering::Relaxed);
        PARALLEL_CHECKS[3].fetch_add(wall, Ordering::Relaxed);
        let batches = PARALLEL_CHECKS[0].load(Ordering::Relaxed);
        if batches.is_multiple_of(1 << 14) {
            let [b, n, sum, w] = [0, 1, 2, 3].map(|i| PARALLEL_CHECKS[i].load(Ordering::Relaxed));
            tracing::info!(
                "texture checks side by side (VICTORY_PARALLEL_REFRESH) so far: {b} batches, {n} textures; the threads' sum {} ms, the time taken {} ms",
                sum / 1000,
                w / 1000
            );
        }
    }

    fn flush(&mut self) {
        self.check_textures();
        self.refresh_textures();
        self.index.clear();
        if self.refreshes.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let (began, all) = (std::time::Instant::now(), self.refreshes.len());
        let _flush = crate::timeline::span("refreshes of copies");
        self.flush_all();
        #[cfg(feature = "diagnostics")]
        {
            SPLIT[6].fetch_add(began.elapsed().as_nanos() as u64, Ordering::Relaxed);
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(all as u64, Ordering::Relaxed);
        }
    }

    fn flush_all(&mut self) {
        // Most of a flush's copies are as they were: they are told so here,
        // from what the submission's asking found, before anything is asked
        // and any thread woken. (Thousands of them, the heap's buffer views
        // once a frame: on the threads, below.)
        const FEW: usize = 512;
        let light = crate::watch::light();
        let refreshes = std::mem::take(&mut self.refreshes);
        let dirty: Option<Vec<usize>> = if light && (refreshes.len() <= FEW || wide_refresh()) {
            let started = std::time::Instant::now();
            let parallel = refreshes.len() > FEW && wide_refresh();
            let dirty = Self::dirty_span_indices(&refreshes, parallel);
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            FLUSHES[2].fetch_add((refreshes.len() - dirty.len()) as u64, Ordering::Relaxed);
            if dirty.is_empty() {
                return;
            }
            Some(dirty)
        } else {
            None
        };
        let count = dirty.as_ref().map_or(refreshes.len(), Vec::len);
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        let ahead = count > 1 && !crate::watch::once();
        if ahead {
            let started = std::time::Instant::now();
            let ranges: Vec<(usize, usize)> = match &dirty {
                Some(dirty) => dirty
                    .iter()
                    .map(|&idx| &refreshes[idx])
                    .filter(|r| r.0.asked_ahead())
                    .map(|r| (r.1, r.2))
                    .collect(),
                None => refreshes
                    .iter()
                    .filter(|r| r.0.asked_ahead())
                    .map(|r| (r.1, r.2))
                    .collect(),
            };
            crate::watch::ask_ahead(&ranges);
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if wide_refresh() && count >= 4 {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            FLUSHES[3].fetch_add(
                if light && refreshes.len() > FEW { refreshes.len() } else { count } as u64,
                Ordering::Relaxed,
            );
            match &dirty {
                Some(dirty) => {
                    crate::shadow::wide_each(dirty.len(), |di| {
                        let (shadow, from, to) = &refreshes[dirty[di]];
                        Self::update_parallel(shadow, *from, *to);
                    });
                }
                None => {
                    crate::shadow::wide_each(refreshes.len(), |i| {
                        let (shadow, from, to) = &refreshes[i];
                        Self::update_parallel(shadow, *from, *to);
                    });
                }
            }
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        } else {
            match dirty {
                Some(dirty) => {
                    for idx in dirty {
                        let (shadow, from, to) = &refreshes[idx];
                        Self::update(shadow, *from, *to);
                    }
                }
                None => {
                    for (shadow, from, to) in refreshes {
                        Self::update(&shadow, from, to);
                    }
                }
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Whether a submission's mid-submission texture checks
/// (`crate::texture::Staging::refresh`: the write watch or guards asked,
/// then the memory hashed and detiled if it may have changed) are gathered
/// between writes of ours and run on the helper threads side by side via
/// `shadow::wide_run` (`VICTORY_PARALLEL_REFRESH=1`; off by default; `alt`:
/// twenty seconds with, twenty without, in turn, each change said in the
/// log).
fn parallel_refresh() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_PARALLEL_REFRESH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a submission's texture checks run side by side from now (VICTORY_PARALLEL_REFRESH=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Since the start, of [`Batch::check_textures`]: batches, textures, the
/// threads' summed microseconds, the microseconds taken.
static PARALLEL_CHECKS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    let _ = copy_mapped_ok(dst, src, len, title_dst);
}

/// [`copy_mapped`], returning `true` when the entire `[dst, dst + len)` range
/// was copied in one shot (used by `place::Placed::prepare` to skip redundant
/// readback-and-memcmp verification).
pub(crate) fn copy_mapped_ok(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) -> bool {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return true;
        }
        2 => return false,
        _ => {}
    }
    if len == 0 {
        return true;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return false;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // (A title destination that is away only while the runtime changes a
    // mapping beside it is written when that is done.)
    let whole = if title_dst {
        let since = crate::watch::changes();
        read(src, dst, len) || crate::watch::again_after_change(since, || read(src, dst, len))
    } else {
        read(src, dst, len)
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return true;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        crate::watch::lost("a copy of the title's", dst as usize, len);
        return false;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
    false
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    zero(to, n);
                }
            });
        }
    });
}

/// Zeros at `to`, in one 4 KiB page or two. It may be title memory (the
/// destination of a title's copy), which can go between the question whether
/// it is there and this.
fn zero(to: *mut u8, n: usize) {
    static ZEROS: [u8; 4096] = [0; 4096];
    if n <= ZEROS.len() && crate::safecopy::on() {
        // SAFETY: a destination that is not there fails the copy.
        if !unsafe { crate::safecopy::copy(to, ZEROS.as_ptr(), n) } {
            crate::watch::lost("zeros", to as usize, n);
        }
        return;
    }
    // SAFETY: inside the destination, which is mapped.
    unsafe { to.write_bytes(0, n) };
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there. Safe while the
/// runtime remaps title memory (the kernel, or a copy that takes the fault,
/// reads). What the runtime's record says is mapped is read in one go each,
/// and by pieces where that fails; memory it has no record of by pieces.
pub(crate) fn read_there(src: *const u8, dst: &mut [u8]) -> usize {
    dst.fill(0);
    let (start, len) = (src as usize, dst.len());
    let mut got = 0;
    let mut runs = crate::inplace::shown_runs(start, len);
    if runs.is_empty() {
        runs.push((start, len));
    }
    for (at, n) in runs {
        let piece = &mut dst[at - start..at - start + n];
        if read_into(at as *const u8, piece) {
            got += n;
            continue;
        }
        // (What failed may have written its first pages.)
        piece.fill(0);
        for_pieces(at, n, 64 << 10, |from, n| {
            let piece = &mut dst[from - start..from - start + n];
            if read_into(from as *const u8, piece) {
                got += n;
                return;
            }
            piece.fill(0);
            for_pieces(from, n, 4 << 10, |from, n| {
                let piece = &mut dst[from - start..from - start + n];
                if read_into(from as *const u8, piece) {
                    got += n;
                } else {
                    piece.fill(0);
                }
            });
        });
    }
    got
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if crate::safecopy::on() {
        // SAFETY: the destination is mapped writable memory of `len` bytes;
        // a source page that is not there fails the copy.
        return unsafe { crate::safecopy::copy(dst, src, len) };
    }
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let since = crate::watch::changes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        if crate::safecopy::on() {
            // SAFETY: the source is ours; a destination page that is not
            // there, or not writable, fails the copy.
            return unsafe { crate::safecopy::copy(at as *mut u8, from, n) };
        }
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                if write(at, n) {
                    return;
                }
                // The kernel's write (`VICTORY_SAFE_COPY=0`) fails where a
                // refresh guarded the block again since the caller opened
                // it; and any where the runtime was changing a mapping.
                crate::watch::writing(at, n);
                if !write(at, n) && !crate::watch::again_after_change(since, || write(at, n)) {
                    crate::watch::lost("a host copy's bytes", at, n);
                }
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer (never
        // published into a live bindless heap; D3D12 orders mappings before
        // subsequent ExecuteCommandLists).
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        #[cfg(feature = "diagnostics")]
        if made.is_ok()
            && let Some(b) = &buffer
        {
            crate::book::made(b, || "an upload buffer".into());
        }
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                crate::residency::allocation_error(&e);
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    present_state: crate::present::PresentState,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
    composer: Option<crate::compose::Composer>,
    panel: panel::Panel,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    last_frame: std::time::Instant,
    frame_ms: Vec<f32>,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            last_frame: now,
            frame_ms: Vec::with_capacity(640),
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Whether the next frame counted makes the frame-rate line.
    fn line_due(&self) -> bool {
        self.since.elapsed() >= std::time::Duration::from_secs(1)
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        let now = std::time::Instant::now();
        let dt_ms = now.duration_since(self.last_frame).as_secs_f32() * 1000.0;
        self.last_frame = now;
        if self.frame_ms.len() < 4096 {
            self.frame_ms.push(dt_ms);
        }
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        PREP_PHASE_FRAMES.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        let refused = BUDGET_NOTES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        let biggest = BUDGET_NOTES[1].swap(0, Ordering::Relaxed);
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame; {refused:.2} indirect records a frame ran empty for want of their list's budget (2^27), the largest cost of one {biggest} (0: under 2^22)"
        );
        crate::shadow::exact_line(frames);
        crate::descriptor::sync_line(frames);
        late_line(frames);
        crate::queue::behind_line(frames);
        card_clock_line(frames, ms);
        crate::dxr::log_stats(frames);
        #[cfg(feature = "diagnostics")]
        spans::line(frames);
        crate::shadow::back_line(frames);
        crate::shadow::gpu_only_log(frames);
        #[cfg(feature = "diagnostics")]
        split_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        {
            let [own, asking, still, all] =
                [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [through, left, filled, short] =
                [0, 1, 2, 3].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left + filled > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there), \
                     {filled:.1} first fillings from the pages there (VICTORY_TEXTURE_FILL=0: left empty), {short:.1} of them \
                     with less than was mapped (looked at again)"
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        {
            let [busy, detiling, lists, shown, again, left] = std::array::from_fn::<f64, 6, _>(|i| {
                crate::heap_worker::STATS[i].swap(0, Ordering::Relaxed) as f64 / frames
            });
            if busy > 0.0 {
                tracing::info!(
                    "textures picked by index, their own thread (VICTORY_HEAP_WORKER), per frame: {:.1} ms at work, of them {:.1} detiling; \
                     {lists:.1} lists, {shown:.0} slots shown, {again:.1} written again before, {left:.1} textures left to the submission",
                    busy / 1e6,
                    detiling / 1e6
                );
            }
        }
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            crate::tiles::status();
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            if crate::resource::held_only() {
                let [copies, bytes, grown, whole] = [0, 1, 2, 3].map(|i| crate::resource::HELD[i].load(Ordering::Relaxed));
                let [flipped, missed] = [0, 1].map(|i| crate::descriptor::host::FLIPPED[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures from the first level viewed (VICTORY_HELD_LEVELS=0: whole), since the start: {copies} host copies of the \
                     levels viewed alone ({} MB on the card, {} MB of title memory whole), {grown} replaced by one with more levels, \
                     {flipped} slots made from their new view after its upload had run ({missed} written again first); {} uploads left \
                     out as the host copy held those bytes (VICTORY_TEXTURE_SAME=0: made); {} MB of texture blocks free; \
                     {} looks for textures to let go, {:.2} ms each, the longest {:.1}",
                    bytes >> 20,
                    whole >> 20,
                    crate::draw::HEAP_TEXTURES_SAME.load(Ordering::Relaxed),
                    crate::pool::texture_room() >> 20,
                    crate::resource::SWEPT_US[0].load(Ordering::Relaxed),
                    crate::resource::SWEPT_US[1].load(Ordering::Relaxed) as f64
                        / 1e3
                        / crate::resource::SWEPT_US[0].load(Ordering::Relaxed).max(1) as f64,
                    crate::resource::SWEPT_US[2].load(Ordering::Relaxed) as f64 / 1e3
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain, plain_told] =
                    [0, 1, 2, 3, 4].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, dropped, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the \
                     title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to \
                     plain addresses finished ({plain_told} {}); {kept} ranges under texture views kept for the texture pass ({dropped} dropped); \
                     {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({}) \
                     (VICTORY_PAGE_WRITES: unset=1|2|8, guards=1, 0=count, 1, slots, all=15)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 {
                        "told as well"
                    } else if mode & 8 != 0 {
                        "passed to texture pass"
                    } else {
                        "counted only"
                    },
                    if mode & (2 | 8) != 0 { "made due again" } else { "left as they were" }
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms;                  texture slots written so far: {placed} placed, {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{} ({} copies made of views off their buffer's elements); depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::shadow::OWN_MADE.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            memory_summary_line(frames);
            if !self.frame_ms.is_empty() {
                let n = self.frame_ms.len();
                self.frame_ms.sort_unstable_by(f32::total_cmp);
                let sum: f64 = self.frame_ms.iter().map(|&v| f64::from(v)).sum();
                let avg_ms = sum / n as f64;
                let avg_fps = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };
                let p99_idx = ((n * 99) / 100).min(n - 1);
                let p95_idx = ((n * 95) / 100).min(n - 1);
                let p99_ms = f64::from(self.frame_ms[p99_idx]);
                let p95_ms = f64::from(self.frame_ms[p95_idx]);
                let max_ms = f64::from(self.frame_ms[n - 1]);
                // 1% low FPS: harmonic mean of the slowest 1% of frames (at least 1 frame).
                let slow_1 = (n / 100).max(1);
                let slow_1_avg = self.frame_ms[n - slow_1..].iter().map(|&v| f64::from(v)).sum::<f64>() / slow_1 as f64;
                let p1_fps = if slow_1_avg > 0.0 { 1000.0 / slow_1_avg } else { 0.0 };
                let p5_fps = if p95_ms > 0.0 { 1000.0 / p95_ms } else { 0.0 };
                tracing::info!(
                    "frame pacing: avg {avg_ms:.1} ms ({avg_fps:.1} fps), p1 {p1_fps:.1} fps (p99 {p99_ms:.1} ms), p5 {p5_fps:.1} fps, max {max_ms:.1} ms over {n} frames"
                );
                self.frame_ms.clear();
            }
            #[cfg(feature = "diagnostics")]
            {
                let put = [0, 1, 2, 3].map(|i| crate::shadow::PUT[i].swap(0, Ordering::Relaxed) as f64 / frames);
                tracing::info!(
                    "what shaders wrote put into title memory per frame (VICTORY_EXACT_BACK): {:.0} pieces in {:.0} writes, {:.1} MB; {:.0} pieces joined over gaps",
                    put[0],
                    put[1],
                    put[2] / 1e6,
                    put[3]
                );
                traffic::line(self.copy_stats_frames as f64);
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    #[allow(dead_code)]
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.present_planes(texture, state, None, frame)
    }

    /// Shows `texture`, optionally alpha-compositing a second `overlay` plane
    /// over it (Series S two-plane presentation), and returns them to their states.
    pub(crate) fn present_planes(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
    ) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let planes = overlay.and_then(|(over, over_state)| {
            // SAFETY: a live overlay resource.
            let over_desc = unsafe { over.GetDesc() };
            Some((
                over,
                D3D12_RESOURCE_STATES(over_state as i32),
                crate::compose::view(&desc)?,
                crate::compose::view(&over_desc)?,
            ))
        });
        let Some(format) = crate::present::select_swap_format(desc.Format, planes.is_some()) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let state_u32 = state;
        let p = presenter.as_mut()?;
        if planes.is_some() && p.composer.is_none() {
            match crate::compose::Composer::new(self, format) {
                Ok(c) => p.composer = Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "two-plane composer not made; showing base plane only");
                    drop(presenter);
                    return self.present_planes(texture, state_u32, None, frame);
                }
            }
        }
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain.ResizeBuffers(
                    p.present_state.buffer_count(),
                    width,
                    height,
                    format,
                    p.present_state.swap_chain_flags(),
                )
            };
            match resized {
                Ok(()) => {
                    if p.format != format {
                        p.composer = None;
                    }
                    (p.width, p.height, p.format) = (width, height, format);
                    p.present_state.on_resized(&p.swap_chain, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        if planes.is_some() && p.composer.is_none() {
            match crate::compose::Composer::new(self, format) {
                Ok(c) => p.composer = Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "two-plane composer not made after resize; showing base plane only");
                    drop(presenter);
                    return self.present_planes(texture, state_u32, None, frame);
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state_u32 as i32);
        // SAFETY: our swap chain, list and live source texture(s).
        unsafe {
            p.present_state.wait_for_latency();
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            let take_shot = crate::present::screenshot_pending();
            let want_shot = crate::capture::take_want();
            #[cfg(feature = "diagnostics")]
            let read_now =
                take_shot || want_shot || crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = take_shot || want_shot;
            // In release builds without a pending screenshot, nothing copies
            // from/onto `back` after two-plane compositing, so transition
            // `RENDER_TARGET -> PRESENT` directly. In diagnostics builds,
            // `Panel::draw` can be toggled at runtime and expects `COPY_DEST`.
            #[cfg(feature = "diagnostics")]
            let needs_back_copy = true;
            #[cfg(not(feature = "diagnostics"))]
            let needs_back_copy = take_shot || want_shot || p.panel.visible();
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            copy.keep(texture);
            let list = &copy.list;
            let composed = if let Some((over, over_state, base_fmt, over_fmt)) = planes {
                copy.keep(over);
                if let Some(composer) = p.composer.as_mut() {
                    let mut pre = Vec::with_capacity(3);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        pre.push(transition(texture, state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                    }
                    if over_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        pre.push(transition(over, over_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                    }
                    pre.push(transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
                    list.ResourceBarrier(&pre);
                    let compose_slot = slot.map_or(0, |i| i + 1);
                    if let Err(err) = composer.record(
                        self,
                        compose_slot,
                        list,
                        [(texture, base_fmt), (over, over_fmt)],
                        &back,
                        (width, height),
                    ) {
                        copy.close();
                        drop(back);
                        drop(presenter);
                        tracing::warn!(%err, "could not record display composite; presenting the base plane");
                        return self.present_planes(texture, state_u32, None, frame);
                    }
                    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !ONCE.swap(true, Ordering::Relaxed) {
                        tracing::info!(
                            frame,
                            width,
                            height,
                            base_format = base_fmt.0,
                            overlay_format = over_fmt.0,
                            "display output: compositing two planes (overlay active)"
                        );
                    }
                    let mut post = Vec::with_capacity(3);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        post.push(transition(texture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, state));
                    }
                    if over_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        post.push(transition(over, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, over_state));
                    }
                    post.push(transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        if needs_back_copy {
                            D3D12_RESOURCE_STATE_COPY_DEST
                        } else {
                            D3D12_RESOURCE_STATE_PRESENT
                        },
                    ));
                    list.ResourceBarrier(&post);
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !composed {
                let mut pre = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    pre.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                pre.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&pre);
                list.CopyResource(&back, texture);
            }
            let screenshot = take_shot
                .then(|| crate::present::record_screenshot_readback(&self.device, list, &back))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let do_capture = want_shot || crate::capture::wanted(frame);
            #[cfg(not(feature = "diagnostics"))]
            let do_capture = want_shot;
            let capture = do_capture
                .then(|| {
                    if composed {
                        list.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        let res = crate::capture::read_back(&self.device, list, &back);
                        list.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )]);
                        res
                    } else {
                        crate::capture::read_back(&self.device, list, texture)
                    }
                })
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() || composed {
                None
            } else {
                crate::capture::probe(&self.device, list, texture)
            };
            #[cfg(feature = "diagnostics")]
            let watched = if composed {
                None
            } else {
                crate::popwatch::take(&self.device, list, texture)
            };
            p.panel.draw(self, copy, &back, (width, height), format);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            if composed {
                if needs_back_copy {
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                }
            } else {
                let mut post = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    post.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                post.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&post);
            }
            copy.close();
            let flown = match slot {
                Some(i) => {
                    let flown = self.execute_own_copy(copy);
                    p.ahead[i].1 = flown;
                    flown
                }
                None => {
                    self.execute(&[copy]);
                    0
                }
            };
            if let Some((buffer, footprint)) = screenshot {
                crate::present::deliver_screenshot(&buffer, &footprint, p.present_state.hdr_active());
            }
            #[cfg(feature = "diagnostics")]
            if let Some(taken) = watched {
                crate::popwatch::sent(taken, flown, frame);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = flown;
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            #[cfg(feature = "diagnostics")]
            crate::replay::record::present(texture, state);
            let mut showing = crate::timeline::span("the host's Present");
            #[cfg(feature = "diagnostics")]
            let present_t = std::time::Instant::now();
            p.present_state.present(&p.swap_chain);
            #[cfg(feature = "diagnostics")]
            PRESENT_NS.fetch_add(present_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            showing.next("the frame meter");
            p.panel.frame(p.meter.line_due());
            p.meter.frame(p.hwnd);
            showing.end();
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        let (swap_chain, format, present_state) =
            crate::present::create_swap_chain(&self.queue, hwnd, width, height, format)?;
        let copy = self.new_list()?;
        copy.close();
        self.log_video_memory("right after the swap chain is made");
        Ok(Presenter {
            swap_chain,
            present_state,
            width,
            height,
            format,
            copy,
            ahead: Vec::new(),
            hwnd,
            meter: Meter::new(),
            composer: None,
            panel: panel::Panel::new(),
        })
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    ///
    /// Waits for residency + WDDM page-table mapping so buffers published into
    /// a live shader-visible descriptor heap cannot fault in-flight lists.
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.shared_buffer_inner(size, true)
    }

    /// Internal non-bindless [`Self::shared_buffer`]: skips `not_resident()`,
    /// `MakeResident()`, and `mapped()` because the resource is never published
    /// into a live shader-visible descriptor heap while flights run.
    pub(crate) fn internal_shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.shared_buffer_inner(size, false)
    }

    fn shared_buffer_inner(&self, size: u64, bindless: bool) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap_flags = if bindless { not_resident() } else { D3D12_HEAP_FLAG_NONE };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    heap_flags,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    if bindless {
                        resident(&self.device, b)?;
                        mapped(&self.device)?;
                    }
                    b.Map(0, None, Some(&mut data))
                })
        };
        #[cfg(feature = "diagnostics")]
        if made.is_ok()
            && let Some(b) = &buffer
        {
            crate::book::made(b, || "a shared buffer".into());
        }
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// Whether buffers can lie in the card's memory and be mapped by the CPU
    /// (`D3D12_HEAP_TYPE_GPU_UPLOAD`: a resizable BAR, and a system and driver
    /// that offer it); logged once.
    pub(crate) fn mapped_card_memory(&self) -> bool {
        static THERE: OnceLock<bool> = OnceLock::new();
        *THERE.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS16,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.GPUUploadHeapSupported.as_bool();
            tracing::info!(supported, "card memory the CPU maps");
            supported
        })
    }

    /// A buffer in the card's own memory that the CPU maps, shaders read and
    /// (with `written`) may write. The CPU should only write it: a read of the
    /// mapping crosses the bus a few bytes at a time.
    pub(crate) fn mapped_card_buffer(&self, size: u64, written: bool) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.mapped_card_memory() {
            return None;
        }
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if written {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    resident(&self.device, b)?;
                    mapped(&self.device)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no buffer in card memory the CPU maps");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    ///
    /// Waits for residency + WDDM page-table mapping so buffers published into
    /// a live shader-visible descriptor heap cannot fault in-flight lists.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.uav_buffer_inner(size, true)
    }

    /// Internal non-bindless [`Self::uav_buffer`]: skips `not_resident()`,
    /// `MakeResident()`, and `mapped()` because the resource is never published
    /// into a live shader-visible descriptor heap while flights run.
    pub(crate) fn internal_uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.uav_buffer_inner(size, false)
    }

    fn uav_buffer_inner(&self, size: u64, bindless: bool) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let heap_flags = if bindless { not_resident() } else { D3D12_HEAP_FLAG_NONE };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                heap_flags,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        #[cfg(feature = "diagnostics")]
        if let Some(b) = &buffer {
            crate::book::made(b, || "an unordered access buffer".into());
        }
        let made = if bindless {
            made.and_then(|()| buffer.as_ref().map_or(Ok(()), |b| resident(&self.device, b)))
                .and_then(|()| mapped(&self.device))
        } else {
            made
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }

    /// Shared depth-twin stencil-copy scratch buffer of at least `size` bytes,
    /// one per power-of-two class (>= 64 KiB). Safe to share across twins
    /// because every copy in [`record_twin_copy`] begins and ends in `COMMON`
    /// on the single direct queue.
    pub(crate) fn shared_twin_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        struct TwinScratch(Vec<(u64, ID3D12Resource)>);
        // SAFETY: D3D12 resources are free-threaded.
        unsafe impl Send for TwinScratch {}
        static SCRATCH: Mutex<TwinScratch> = Mutex::new(TwinScratch(Vec::new()));

        let class = twin_scratch_class(size);
        let mut all = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, buffer)) = all.0.iter().find(|(c, _)| *c == class) {
            return Some(buffer.clone());
        }
        let buffer = self.internal_uav_buffer(class)?;
        name(&buffer, || format!("depth twin copies {class:#x}"));
        all.0.push((class, buffer.clone()));
        Some(buffer)
    }
}

/// Power-of-two size class (>= 64 KiB) for shared depth-twin copy scratch buffers.
fn twin_scratch_class(size: u64) -> u64 {
    size.max(1 << 16).checked_next_power_of_two().unwrap_or(size.max(1 << 16))
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
/// For a list that stopped at command `done`: what the indirect executions
/// there ran, from the one before it to the next barrier (four at most), and
/// what the heap slots their user data names hold now.
#[cfg(feature = "diagnostics")]
fn indirects_around(history: &[D3D12_AUTO_BREADCRUMB_OP], done: usize, all: &indirects::Shared) -> String {
    let indirect = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT;
    let all = all.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    let mut shown = 0;
    let mesh = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH;
    for (at, op) in history.iter().enumerate().skip(done.saturating_sub(1)) {
        if shown == 4 || (at > done && *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER) {
            break;
        }
        if !indirect(op) && !mesh(op) {
            continue;
        }
        shown += 1;
        let direct = mesh(op);
        let nth = if direct { indirects::before(history, at, mesh) } else { indirects::before(history, at, indirect) };
        let Some((what, words)) = all.nth(nth, direct) else {
            continue;
        };
        text.push_str(&format!("; command {at} ran {what}"));
        for (word, slot, what) in crate::descriptor::host::describe_slots(words) {
            text.push_str(&format!(", word {word} slot {slot}: {what}"));
        }
    }
    text
}

fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Whether consecutive redundant global `NULL` UAV barriers on the same
/// recording (with zero intervening UAV-capable draws, dispatches, or GPU
/// buffer copies) are coalesced (`VICTORY_COALESCE_UAV_BARRIERS=0` disables).
pub(crate) fn coalesce_uav_barriers() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        coalesce_uav_barriers_for(std::env::var_os("VICTORY_COALESCE_UAV_BARRIERS").as_deref())
    })
}

fn coalesce_uav_barriers_for(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod refresh_tests {
    use super::Batch;
    use crate::shadow::Shadow;
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RESERVE, MapViewOfFile, PAGE_READWRITE,
        VirtualAlloc,
    };

    const PAGE: usize = 64 << 10;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Title memory for the tests (never given back: copies of it stay in
    /// the guards): write-watched, or a view of a section.
    fn title(len: usize, view: bool) -> usize {
        // SAFETY: plain allocations.
        let at = unsafe {
            if view {
                let section =
                    CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
                // As the runtime tells of it.
                crate::title_memory_section(section as usize, len as u64, true);
                crate::title_memory_remapped(at, len);
                crate::title_memory_view(at, len, section as usize, 0);
                at
            } else {
                VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize
            }
        };
        assert_ne!(at, 0);
        at
    }

    /// Title memory of `blocks` pages in a placeholder, one view to a page,
    /// and every fourth page not there: as a title that streams has it.
    fn with_holes(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS,
            VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a placeholder and a section of our own.
        unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(at, 0);
            let len = blocks * PAGE;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, std::ptr::null());
            for block in (0..blocks).filter(|b| b % 4 != 3) {
                let to = at + block * PAGE;
                VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    to as *const _,
                    (block * PAGE) as u64,
                    PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, to);
                crate::title_memory_remapped(to, PAGE);
            }
            at
        }
    }

    fn memory(len: usize) -> *mut u8 {
        // SAFETY: a plain allocation.
        unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast() }
    }

    struct Random(u64);

    impl Random {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as usize % n
        }
    }

    /// Prepares as a submission does: one asking for all the ranges, the
    /// refreshes in several flushes, the asking's end.
    fn submission(ranges: &[(Arc<Shadow>, usize, usize)], flushes: usize) {
        // One at a time, as submissions are.
        static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let asked: Vec<(usize, usize)> = ranges.iter().filter(|r| !r.0.unwatched()).map(|r| (r.1, r.1 + r.2)).collect();
        crate::watch::ask_once(&asked);
        let mut batch = Batch::default();
        for part in ranges.chunks(ranges.len().div_ceil(flushes).max(1)) {
            for (shadow, at, len) in part {
                batch.add(shadow, *at, *len);
            }
            batch.flush();
        }
        crate::watch::ask_once_end();
    }

    /// Whatever is written where, by the title or by us, and whichever
    /// copies (of both kinds of memory, over one another, none on a page
    /// line) are refreshed in which ranges: a refreshed range holds what
    /// title memory holds.
    #[test]
    fn refreshed_ranges_hold_title_memory() {
        // SAFETY (all blocks): title memory and copies of the test's own,
        // written by this thread and, in `Batch::flush`, by the threads that
        // refresh, which are done when it returns.
        let mut random = Random(0x1234_5678_9ABC_DEF1);
        const LEN: usize = 24 * PAGE;
        let regions = [title(LEN, false), title(LEN, true)];
        let mut copies = Vec::new();
        for &region in &regions {
            for (from, to) in [(0, LEN), (100, LEN - 5000), (3 * PAGE + 777, 9 * PAGE + 10), (8 * PAGE, 8 * PAGE + 300), (PAGE, 20 * PAGE)] {
                let data = memory(to - from);
                copies.push((Shadow::unbacked((region + from) as u64, data, (to - from) as u64, false), data));
            }
        }
        let byte = |at: usize| unsafe { (at as *const u8).read_volatile() };
        for round in 0..600 {
            for _ in 0..random.below(4) {
                let at = regions[random.below(2)] + random.below(LEN - 64);
                let n = 1 + random.below(60);
                if random.below(3) == 0 {
                    // What shaders wrote.
                    let was: Vec<u8> = (0..n).map(|i| byte(at + i)).collect();
                    let now: Vec<u8> = was.iter().map(|b| if random.below(2) == 0 { *b } else { random.below(256) as u8 }).collect();
                    Shadow::put(at, &was, &now);
                    assert!(now.iter().enumerate().all(|(i, &b)| byte(at + i) == b), "not put");
                    continue;
                }
                for i in 0..n {
                    unsafe { ((at + i) as *mut u8).write_volatile(random.below(256) as u8) };
                }
            }
            let mut ranges = Vec::new();
            for (shadow, _) in &copies {
                if random.below(3) == 0 {
                    continue;
                }
                let (start, len) = shadow.span();
                let (start, len) = (start as usize, len as usize);
                // Often the range of the time before, as a list's is.
                let from = if random.below(2) == 0 { 0 } else { random.below(len) };
                let n = if random.below(2) == 0 { len - from } else { 1 + random.below(len - from) };
                ranges.push((shadow.clone(), start + from, n));
            }
            if ranges.is_empty() {
                continue;
            }
            let flushes = 1 + random.below(3);
            submission(&ranges, flushes);
            for (shadow, at, n) in &ranges {
                let data = copies.iter().find(|c| Arc::ptr_eq(&c.0, shadow)).unwrap().1;
                let start = shadow.span().0 as usize;
                let same = unsafe {
                    std::slice::from_raw_parts(*at as *const u8, *n) == std::slice::from_raw_parts(data.add(at - start), *n)
                };
                assert!(same, "round {round}: {at:#x}+{n:#x} of a copy at {start:#x}; first byte {}", byte(*at));
            }
        }
    }

    /// A copy in the card's memory, with the card's side and the sends to it
    /// played by the test.
    struct Carded {
        shadow: Arc<Shadow>,
        data: usize,
        start: usize,
        len: usize,
        /// The last send, the last send every fetch to come is behind, and
        /// the card's bytes: the queue's order is the order it is locked in.
        card: std::sync::Mutex<(u64, u64, Vec<u8>)>,
        /// Whether a flight that writes it has not landed.
        flying: std::sync::atomic::AtomicBool,
        /// The dword of each eight its shaders write.
        class: usize,
    }

    impl Carded {
        fn new(start: usize, len: usize, class: usize) -> Carded {
            let data = memory(len);
            Carded {
                shadow: Shadow::unbacked(start as u64, data, len as u64, true),
                data: data as usize,
                start,
                len,
                card: std::sync::Mutex::new((0, 0, vec![0; len])),
                flying: std::sync::atomic::AtomicBool::new(false),
                class,
            }
        }

        /// What changed on the CPU's side goes up.
        fn send(&self) {
            let mut card = self.card.lock().unwrap();
            card.0 += 1;
            let (number, merged) = (card.0, card.1);
            for (a, b) in self.shadow.send_up(number, merged) {
                // SAFETY: the copy's bytes; nothing writes those that are to
                // go up.
                unsafe { std::ptr::copy_nonoverlapping((self.data + a) as *const u8, card.2[a..b].as_mut_ptr(), b - a) };
            }
        }
    }

    /// Everything at once: the title writes its memory on two threads, the
    /// submitting thread puts its bytes in under shader writes
    /// (`Shadow::merge_under_writes`), refreshes copies that overlap (two in
    /// the card's memory, one not), sends up what changed and declares what
    /// the next flight's shaders write, and what flights wrote is landed
    /// meanwhile (`crate::land`, beside the guards). When all of it stands
    /// still and the copies are refreshed once more, each of them, and the
    /// card's side of those that have one, hold what title memory holds, and
    /// title memory what was written last.
    ///
    /// Into the first copy shaders write besides through a view nobody
    /// declared, words the title does not write: nothing that goes up ever
    /// puts an older value over those.
    ///
    /// (It fails in the first rounds if `landed_beside` notes nothing, or
    /// only takes the block out of the guards' clean ones and leaves its bit,
    /// or if a fault leaves the bit; in some thousand rounds,
    /// `VICTORY_TEST_ROUNDS`, if what is landed is noted before it is in.)
    #[test]
    fn landings_title_writes_and_light_refreshes_at_once() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        assert!(crate::safecopy::on());
        const LEN: usize = 24 * PAGE;
        let rounds: usize = std::env::var("VICTORY_TEST_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let region = title(LEN, true);
        let dword = |at: usize| unsafe { &*((region + at) as *const AtomicU32) };
        let carded = [Carded::new(region, LEN, 3), Carded::new(region + 4 * PAGE + 1000, LEN - 6 * PAGE - 4, 5)];
        let plain = {
            let (start, len) = (region + 700, LEN - 9000);
            let data = memory(len);
            (Shadow::unbacked(start as u64, data, len as u64, false), start, len, data as usize)
        };
        // What was last written so, by offset.
        let mut undeclared = std::collections::HashMap::new();
        let exact = std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0");
        let stop = AtomicBool::new(false);
        // Also when an assertion fails on one of the threads.
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        // Held shared by who writes title memory, alone while it is looked at.
        let still = std::sync::RwLock::new(());
        let (fly, flights) = std::sync::mpsc::channel::<(usize, Vec<(usize, usize)>, u32)>();
        std::thread::scope(|s| {
            for t in 0..2usize {
                let (stop, still) = (&stop, &still);
                s.spawn(move || {
                    let mut random = Random(0x7777_1234_0000_0001 + t as u64 * 0x1111);
                    let mut value = 0u32;
                    while !stop.load(Ordering::Relaxed) {
                        {
                            let _moving = still.read().unwrap();
                            for _ in 0..3 {
                                value += 1;
                                // Of each eight dwords, one a thread.
                                let at = random.below(LEN / 32) * 32 + 4 * (1 + t);
                                dword(at).store(value, Ordering::Relaxed);
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_micros(20));
                    }
                });
            }
            // Lands flights.
            {
                let (carded, stop) = (&carded, &stop);
                s.spawn(move || {
                    let _stop = Stop(stop);
                    while let Ok((which, ranges, value)) = flights.recv() {
                        let c = &carded[which];
                        // Fetched: the card's bytes, behind every send so far.
                        let (after, now) = {
                            let card = c.card.lock().unwrap();
                            (card.0, card.2.clone())
                        };
                        // By piece: 64 KiB of title memory.
                        let mut pieces = Vec::new();
                        for &(from, to) in &ranges {
                            let mut at = from;
                            while at < to {
                                let next = ((c.start + at) / PAGE * PAGE + PAGE - c.start).min(to);
                                pieces.push((at, next, now.as_ptr() as usize + at));
                                at = next;
                            }
                        }
                        let parts: Vec<_> = pieces.chunks(3).collect();
                        crate::land::each(parts.len(), &|i| {
                            let mut notes = crate::land::Notes::default();
                            // SAFETY: the bytes of `now`; each piece once.
                            unsafe { c.shadow.land_card(parts[i], after, &mut notes) };
                            notes.flush();
                        });
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                // (The next fetch may have brought more.)
                                assert!(dword(c.start - region + at).load(Ordering::Relaxed) >= value, "not landed");
                            }
                        }
                        let done: Vec<_> = ranges.iter().map(|&(a, b)| (c.start + a, c.start + b)).collect();
                        c.shadow.copy_back_done(&done);
                        c.card.lock().unwrap().1 = after;
                        c.flying.store(false, Ordering::Release);
                    }
                });
            }
            let _stop = Stop(&stop);
            let mut random = Random(0xABCD_EF01_2345_6789);
            let check = |round: usize, random: &mut Random, quiet: bool| {
                for c in &carded {
                    if c.shadow.writes_pending() {
                        let mut runs = usize::MAX;
                        assert!(!c.shadow.merge_under_writes(c.start, c.len, &mut runs), "round {round}");
                    }
                }
                let mut ranges = Vec::new();
                for (shadow, start, len) in
                    carded.iter().map(|c| (&c.shadow, c.start, c.len)).chain([(&plain.0, plain.1, plain.2)])
                {
                    // All of it, or a part; often the same part.
                    let (from, n) = match random.below(3) {
                        _ if quiet => (0, len),
                        0 => (0, len),
                        1 => (len / 3 / 4 * 4, len / 2 / 4 * 4),
                        _ => {
                            let from = random.below(len / 4) * 4;
                            (from, (4 + random.below((len - from) / 4) * 4).min(len - from))
                        }
                    };
                    ranges.push((shadow.clone(), start + from, n));
                }
                submission(&ranges, 1 + random.below(3));
                carded.iter().for_each(Carded::send);
            };
            let mut value = 1u32;
            for round in 0..rounds {
                check(round, &mut random, false);
                if exact {
                    let c = &carded[0];
                    let mut card = c.card.lock().unwrap();
                    for _ in 0..40 {
                        value += 1;
                        let at = random.below(LEN / 32) * 32 + 4 * 6;
                        card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                        undeclared.insert(at, value);
                    }
                }
                // The flight of this submission: its shaders' writes.
                for (which, c) in carded.iter().enumerate() {
                    if c.flying.swap(true, Ordering::Acquire) {
                        continue;
                    }
                    value += 1;
                    let mut ranges: Vec<(usize, usize)> = Vec::new();
                    let first = random.below(6) * PAGE / 2;
                    for k in 0..2 {
                        let from = first + k * 9 * PAGE + 4 * random.below(9000);
                        let to = (from + 4 * (1 + random.below(7 * PAGE / 4))).min(c.len);
                        if from < to {
                            ranges.push((from, to));
                        }
                    }
                    for &(from, to) in &ranges {
                        c.shadow.writing(c.start + from, to - from);
                    }
                    {
                        let mut card = c.card.lock().unwrap();
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    }
                    fly.send((which, ranges, value)).unwrap();
                }
                if round % 20 != 19 {
                    continue;
                }
                // Still: no flight in the air, no title write.
                let _still = still.write().unwrap();
                while carded.iter().any(|c| c.flying.load(Ordering::Acquire)) {
                    assert!(!stop.load(Ordering::Relaxed), "the landing thread failed");
                    std::thread::yield_now();
                }
                check(round, &mut random, true);
                // SAFETY: the test's own memory, written by nobody now.
                let bytes = |at: usize, len: usize| unsafe { std::slice::from_raw_parts(at as *const u8, len) };
                let first = |a: &[u8], b: &[u8]| a.iter().zip(b).position(|(x, y)| x != y);
                for c in &carded {
                    let (held, there) = (bytes(c.data, c.len), bytes(c.start, c.len));
                    assert_eq!(first(held, there), None, "round {round}: a card copy's CPU side (class {})", c.class);
                    let card = c.card.lock().unwrap();
                    let mut side = card.2.clone();
                    if c.class == carded[0].class {
                        // (They may have been landed meanwhile, by a flight
                        // that declared their range, and may not.)
                        for (&at, &value) in &undeclared {
                            let on_card = u32::from_le_bytes(side[at..at + 4].try_into().unwrap());
                            assert_eq!(on_card, value, "round {round}: an older value at {at:#x}");
                            side[at..at + 4].copy_from_slice(&there[at..at + 4]);
                        }
                    }
                    assert_eq!(first(&side, there), None, "round {round}: a card copy's card side (class {})", c.class);
                }
                assert_eq!(first(bytes(plain.3, plain.2), bytes(plain.1, plain.2)), None, "round {round}: the other copy");
            }
            drop(fly);
        });
    }

    /// The title's bytes put in under shader writes, for a copy of 174 MB
    /// whose shader writes were declared as 40 ranges that overlap, with 12
    /// pages written since. Run with `--ignored --nocapture`, and with
    /// `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn bytes_under_overlapping_ranges() {
        const BIG: usize = 174 << 20;
        let big = title(BIG, true);
        // SAFETY (both): the test's own memory.
        unsafe { std::ptr::write_bytes(big as *mut u8, 1, BIG) };
        let copy = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        copy.update(big, BIG);
        copy.take_dirty();
        for i in 0..40 {
            copy.writing(big + i * 5920, BIG - i * 5920);
        }
        let mut best = f64::MAX;
        for round in 0..30 {
            for i in 0..12 {
                unsafe { ((big + (i * 211 + round) % 2700 * PAGE + 64) as *mut u8).write_volatile(round as u8) };
            }
            let started = std::time::Instant::now();
            let mut runs = usize::MAX;
            assert!(!copy.merge_under_writes(big, BIG, &mut runs));
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
            copy.take_dirty();
        }
        println!("VICTORY_LIGHT_REFRESH {}: best {best:.3} ms", crate::watch::light());
    }

    /// A range nothing wrote is known to be as it was, at its ends too, and
    /// is so no more once something wrote it.
    #[test]
    fn unchanged_ranges_are_known() {
        for view in [false, true] {
            let region = title(8 * PAGE, view);
            // (A page read for the first time counts as written.)
            // SAFETY: the test's own memory.
            unsafe { std::ptr::write_bytes(region as *mut u8, 1, 8 * PAGE) };
            let data = memory(8 * PAGE);
            let copy = Shadow::unbacked(region as u64, data, (8 * PAGE) as u64, false);
            // Ends in the middle of pages.
            let (at, len) = (region + 500, 5 * PAGE);
            let look = |expected: bool, what: &str| {
                // The submission's one asking, which takes two ranges. (Again
                // if a submission of another test ended it.)
                let mut known = false;
                for _ in 0..100 {
                    crate::watch::ask_once(&[(at, at + len), (region + 7 * PAGE, region + 8 * PAGE)]);
                    known = copy.nothing_new(at, len);
                    if known || !expected {
                        break;
                    }
                }
                assert_eq!(known, expected, "view {view}: {what}");
                let copied = copy.update(at, len);
                crate::watch::ask_once_end();
                copied
            };
            assert_eq!(look(false, "never taken"), len);
            assert_eq!(look(true, "taken"), 0);
            assert_eq!(look(true, "taken"), 0);
            // SAFETY: the test's own memory.
            unsafe { ((region + 600) as *mut u8).write_volatile(1) };
            // (The kernel names the 4 KiB, the guards only the block.)
            assert_eq!(look(false, "written at its end"), if view { PAGE } else { 4096 } - 500);
            assert_eq!(look(true, "taken again"), 0);
            // Outside the range, on its last page.
            // SAFETY: as above.
            unsafe { ((region + 5 * PAGE + 600) as *mut u8).write_volatile(1) };
            assert_eq!(look(false, "written on its last page"), 500);
            // A longer range was never taken whole.
            assert!(!copy.nothing_new(region, 6 * PAGE));
        }
    }

    /// Of a copy in the card's memory only the dwords title memory changed go
    /// up, and none beside them, not even between two in 64 bytes; a page
    /// never taken goes up whole.
    #[test]
    fn only_what_differs_goes_up() {
        let region = title(4 * PAGE, true);
        let data = memory(4 * PAGE);
        let copy = Shadow::unbacked(region as u64, data, (4 * PAGE) as u64, true);
        copy.update(region, 4 * PAGE);
        assert_eq!(copy.take_dirty(), [(0, 4 * PAGE)], "all of it, though zeros are no change");
        // What the relay does: shader writes into title memory, and into the
        // CPU's side.
        crate::watch::landing(region + PAGE + 640, 64);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes((region + PAGE + 640) as *mut u8, 7, 64);
            std::ptr::write_bytes(data.add(PAGE + 640), 7, 64);
        }
        copy.update(region, 4 * PAGE);
        let again = copy.take_dirty();
        // SAFETY: as above.
        unsafe {
            ((region + 2 * PAGE + 129) as *mut u8).write_volatile(9);
            ((region + 2 * PAGE + 136) as *mut u32).write_volatile(9);
            ((region + 2 * PAGE + 140) as *mut u32).write_volatile(9);
            // The value that was there.
            ((region + 2 * PAGE + 132) as *mut u32).write_volatile(0);
        }
        copy.update(region, 4 * PAGE);
        let written = copy.take_dirty();
        if std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0") {
            assert_eq!(again, []);
            assert_eq!(written, [(2 * PAGE + 128, 2 * PAGE + 132), (2 * PAGE + 136, 2 * PAGE + 144)]);
        } else {
            assert_eq!(again, [(PAGE, 2 * PAGE)]);
            assert_eq!(written, [(2 * PAGE, 3 * PAGE)]);
        }
        // SAFETY: as above.
        assert_eq!(unsafe { data.add(2 * PAGE + 129).read() }, 9);
    }

    /// The refreshes of a frame as run record-1 had them: 30 submissions, the
    /// first with a copy for each of 4,400 ranges, the others with 60 and a
    /// big copy that 55 landed pieces and 9 title writes opened, while a
    /// thread lands what shaders wrote. Run with `--ignored --nocapture`,
    /// and with `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn frame_of_refreshes() {
        let mut random = Random(0x9876_5432_1234_5671);
        const EACH: usize = 224 << 10;
        const BIG: usize = 174 << 20;
        let watched = title(3000 * EACH, false);
        // `VICTORY_BENCH_HOLES=0`: views without holes.
        let holes = std::env::var_os("VICTORY_BENCH_HOLES").is_none_or(|v| v != "0");
        let views = if holes { with_holes(1400 * EACH / PAGE) } else { title(1400 * EACH, true) };
        let there = |at: usize| !holes || (at - views) / PAGE % 4 != 3;
        let big = title(BIG, true);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes(watched as *mut u8, 1, 3000 * EACH);
            for at in (views..views + 1400 * EACH).step_by(PAGE).filter(|&at| there(at)) {
                std::ptr::write_bytes(at as *mut u8, 1, PAGE);
            }
            std::ptr::write_bytes(big as *mut u8, 1, BIG);
        }
        let mut small = Vec::new();
        for i in 0..4400 {
            let (start, at) = if i < 3000 { (watched, i * EACH) } else { (views, (i - 3000) * EACH) };
            let data = memory(EACH);
            // The range views read, not all of the copy.
            small.push((Shadow::unbacked((start + at) as u64, data, EACH as u64, false), start + at + 4000, EACH - 9000));
        }
        let heap = (Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true), big, BIG);
        // The same memory, as another copy: what is landed is landed through
        // it, and the refreshes are of the first.
        let lands = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        let (mut best, mut asking) = ([f64::MAX; 2], [0f64; 2]);
        let (mut sum, mut landing) = (0.0, 0.0);
        const FRAMES: usize = 12;
        for frame in 0..FRAMES {
            let mut took = [0f64; 2];
            for i in 0..30 {
                // The title's writes, and the relay's beside the submission.
                for _ in 0..9 {
                    let at = views + random.below(1400 * EACH / 8) * 8;
                    if there(at) {
                        // SAFETY: the test's own memory.
                        unsafe { (at as *mut u64).write_volatile(frame as u64) };
                    }
                }
                for _ in 0..60 {
                    let at = watched + random.below(3000 * EACH / 8) * 8;
                    // SAFETY: the test's own memory.
                    unsafe { (at as *mut u64).write_volatile(frame as u64) };
                }
                let first = random.below(BIG / PAGE - 60);
                let lands = lands.clone();
                let land = std::thread::spawn(move || {
                    let now = vec![frame as u8 ^ i as u8 ^ 0x55; PAGE];
                    let started = std::time::Instant::now();
                    for pieces in (first..first + 55).collect::<Vec<_>>().chunks(4) {
                        let pieces: Vec<_> =
                            pieces.iter().map(|p| (p * PAGE + 8, (p + 1) * PAGE - 8, now.as_ptr() as usize)).collect();
                        let mut notes = crate::land::Notes::default();
                        // SAFETY: the bytes of `now`; nothing else lands here.
                        unsafe { lands.land_card(&pieces, 0, &mut notes) };
                        notes.flush();
                    }
                    started.elapsed().as_secs_f64() * 1e3
                });
                let ranges: Vec<(Arc<Shadow>, usize, usize)> = if i == 0 {
                    small.iter().cloned().chain([heap.clone()]).collect()
                } else {
                    (0..60).map(|_| small[random.below(4400)].clone()).chain([heap.clone()]).collect()
                };
                let started = std::time::Instant::now();
                submission(&ranges, 3);
                took[usize::from(i != 0)] += started.elapsed().as_secs_f64() * 1e3;
                let put = land.join().unwrap();
                if frame >= 2 {
                    landing += put;
                }
                heap.0.take_dirty();
            }
            if frame >= 2 {
                sum += took[0] + took[1];
                for k in 0..2 {
                    best[k] = best[k].min(took[k]);
                    asking[k] += took[k];
                }
            }
        }
        println!(
            "VICTORY_LIGHT_REFRESH {}: a frame's refreshes {:.2} ms in the mean; the submission of all copies {:.2} (best {:.2}), the 29 \
             others {:.2} (best {:.2}); the landing thread's puts {:.2}",
            crate::watch::light(),
            sum / (FRAMES - 2) as f64,
            asking[0] / (FRAMES - 2) as f64,
            best[0],
            asking[1] / (FRAMES - 2) as f64,
            best[1],
            landing / (FRAMES - 2) as f64,
        );
        #[cfg(feature = "diagnostics")]
        println!("per frame, as `crate::watch::asks`: {:?}", crate::watch::asks().map(|n| n / FRAMES as u64));
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, RamBand, ram_band};
    use std::sync::Arc;

    #[test]
    fn ram_band_tracks_resident_and_commit_pressure() {
        const M: u64 = 1 << 20;
        // The measured local Series S baseline stayed calm (32 GB machine,
        // roughly 11 GB commit limit).
        assert_eq!(
            ram_band(RamBand::Calm, 3857 * M, 7500 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Calm
        );
        // Private commit alone crosses the watch watermark.
        assert_eq!(
            ram_band(RamBand::Calm, 3857 * M, 8200 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Watch
        );
        // The measured Series X commit failure was High.
        assert_eq!(
            ram_band(RamBand::Calm, 4452 * M, 10191 * M, 16047 * M, 154 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::High
        );
    }

    #[test]
    fn ram_band_exits_only_below_lower_watermarks() {
        const M: u64 = 1 << 20;
        assert_eq!(
            ram_band(RamBand::High, 4452 * M, 8900 * M, 16047 * M, 1200 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Watch,
            "just under the entry mark does not drop straight to calm"
        );
        assert_eq!(
            ram_band(RamBand::Watch, 3857 * M, 6600 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Calm,
            "all lower watermarks met"
        );
    }

    #[test]
    fn ram_band_does_not_toggle_below_the_entry_watermark() {
        const M: u64 = 1 << 20;
        let mut band = RamBand::Calm;
        for private in [7500 * M, 8200 * M, 8200 * M, 8200 * M] {
            band = ram_band(band, 3857 * M, private, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M);
        }
        assert_eq!(band, RamBand::Watch);
        band = ram_band(band, 3857 * M, 7800 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M);
        assert_eq!(band, RamBand::Watch);
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        let vram_idle_ms = |memory| super::vram_idle_for(memory, false);
        let keeps = |memory| super::vram_idle_for(memory, true);
        let gib = 1024 * 1024 * 1024;
        assert_eq!(keeps(Some([100, 67, 0, 0])), 600_000);
        assert_eq!(keeps(Some([100, 68, 0, 0])), 30_000);
        assert_eq!(keeps(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(keeps(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(keeps(None), 5_000);
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn textures_are_kept_while_the_card_has_room() {
        use super::vram_keep;
        let gib = 1 << 30;
        assert_eq!(vram_keep(100 * gib, 90 * gib).0, 600_000);
        // Past 91%: the longest idle ones, enough to come down to 88 %.
        let (ms, bytes) = vram_keep(100 * gib, 93 * gib);
        assert_eq!((ms, bytes), (5_000, 5 * gib));
        assert_eq!(vram_keep(100 * gib, 95 * gib), (1_500, 7 * gib));
        assert_eq!(vram_keep(100 * gib, 120 * gib), (1_500, 32 * gib));
        assert_eq!(vram_keep(0, 7).0, 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        let ram_idle_ms = |avail| super::ram_idle_for(avail, false);
        assert_eq!(super::ram_idle_for(Some(12 << 30), true), 300_000);
        assert_eq!(super::ram_idle_for(Some((12 << 30) - 1), true), 10_000);
        assert_eq!(super::ram_idle_for(None, true), 10_000);
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }

    #[test]
    fn allocator_pool_caps_trim_under_ram_pressure() {
        const GIB: u64 = 1 << 30;
        // Defaults scale down on <=32 GB hosts and honor explicit env overrides.
        assert_eq!(super::resolve_base_allocator_caps(None, None, true), (256, 64));
        assert_eq!(super::resolve_base_allocator_caps(None, None, false), (384, 128));
        assert_eq!(super::resolve_base_allocator_caps(Some("96"), None, true), (96, 48));
        assert_eq!(super::resolve_base_allocator_caps(Some("96"), Some("24"), true), (96, 24));
        assert_eq!(super::resolve_base_allocator_caps(Some("0"), None, true), (0, 0));

        // Calm keeps configured caps.
        assert_eq!(super::allocator_pool_caps(256, 64, 10 * GIB, false, false), (256, 64));
        assert_eq!(super::allocator_pool_caps(384, 128, 15 * GIB, false, false), (384, 128));
        // Watch / >=16 GiB private keeps 2-frame list reuse (<=320) and tiny barrier preludes (<=64).
        assert_eq!(super::allocator_pool_caps(256, 64, 16 * GIB, false, false), (256, 64));
        assert_eq!(super::allocator_pool_caps(384, 128, 8 * GIB, true, false), (320, 64));
        // Critical / >=26 GiB private trims lists to <=128 and preludes to <=32.
        assert_eq!(super::allocator_pool_caps(256, 64, 22 * GIB, false, false), (256, 64));
        assert_eq!(super::allocator_pool_caps(256, 64, 26 * GIB, false, false), (128, 32));
        assert_eq!(super::allocator_pool_caps(384, 128, 8 * GIB, true, true), (128, 32));
        // Respects smaller explicit caps.
        assert_eq!(super::allocator_pool_caps(40, 20, 26 * GIB, false, false), (40, 10));

        // Landed-flight lag shrinks to 0 under high commit so finished allocators recycle immediately.
        assert_eq!(super::land_lag(16, 10 * GIB, false, false), 16);
        assert_eq!(super::land_lag(16, 14 * GIB, false, false), 4);
        assert_eq!(super::land_lag(64, 10 * GIB, true, false), 4);
        assert_eq!(super::land_lag(16, 18 * GIB, false, false), 0);
        assert_eq!(super::land_lag(16, 10 * GIB, false, true), 0);
    }

    #[test]
    fn twin_scratch_class_rounds_up_to_power_of_two_with_64k_floor() {
        assert_eq!(super::twin_scratch_class(0), 1 << 16);
        assert_eq!(super::twin_scratch_class(1), 1 << 16);
        assert_eq!(super::twin_scratch_class(1 << 16), 1 << 16);
        assert_eq!(super::twin_scratch_class((1 << 16) + 1), 1 << 17);
        assert_eq!(super::twin_scratch_class(3 << 20), 4 << 20);
        assert_eq!(super::twin_scratch_class(u64::MAX), u64::MAX);
    }

    #[test]
    fn page_fault_delta_handles_normal_and_u32_wraparound() {
        assert_eq!(super::page_fault_delta(100, 100), 0);
        assert_eq!(super::page_fault_delta(100, 250), 150);
        assert_eq!(super::page_fault_delta(u32::MAX - 9, 5), 15);
    }
}

