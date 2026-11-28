//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// Title memory's pages of 4 KiB (from the one `start` lies in on), a bit
    /// each, that what shaders wrote was put into since a refresh last looked
    /// at them ([`Shadow::take_landed`]); empty until then.
    landings: Mutex<Vec<u64>>,
    /// Whether any is.
    landing: std::sync::atomic::AtomicBool,
    /// The bytes of `buffer`, once asked.
    width: std::sync::OnceLock<u64>,
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never), paired with the two ranges
    /// taken whole (the last one and the longest; [`Shadow::copy_stale`]).
    copied: Mutex<(Vec<u64>, [(usize, usize, u64); 2])>,
    /// Lock-free mirror of the two ranges copied whole (`copied.1`: last and
    /// longest), guarded by a seqlock (`ranged_seq`: even = stable, odd = updating).
    ranged_seq: AtomicU64,
    ranged_from: [AtomicUsize; 2],
    ranged_to: [AtomicUsize; 2],
    ranged_at: [AtomicU64; 2],
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Exact length of `pending`, maintained under `pending.lock()` so empty
    /// fast-paths avoid acquiring the mutex.
    pending_count: AtomicUsize,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// Held shared while pieces of the copy are landed (`crate::land`), each
    /// its own bytes of it on several threads, and alone while title bytes
    /// go in under shader writes ([`Shadow::merge_under_writes`]), which
    /// takes `before`'s lock too. Taken ahead of that one.
    merging: RwLock<()>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// Last frame number this loose window was touched (for BAR hot-tier promotion).
    touch_frame: AtomicU64,
    /// Number of distinct frames this loose window has been touched (`ARCH-05`).
    touch_frames: AtomicU64,
    /// Cumulative bytes copied into this loose window (`ARCH-05`).
    copied_bytes: AtomicU64,
    /// L0 `shared_buffer` bytes accounted in [`L0_SHADOW_BYTES`] (`0` for card/BAR/placed).
    l0_bytes: u64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    #[cfg(feature = "diagnostics")]
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    #[cfg(feature = "diagnostics")]
    unsized_asks: AtomicU64,
    /// Whether shaders were ever declared to write it.
    written: std::sync::atomic::AtomicBool,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// Set once this buffer/window is unregistered or replaced in-place, so
    /// cached descriptor-sweep entries can skip global epoch checks for live
    /// buffer shadows.
    dead: std::sync::atomic::AtomicBool,
    /// Of a copy with two sides: only the dwords the title changed go up,
    /// as with [`send_changed`] ([`by_address`]).
    changes_only: std::sync::atomic::AtomicBool,
    /// A window in card memory the CPU maps ([`window_mapped_for`]).
    mapped: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way.
    place: Option<InPlace>,
    /// Adaptive GPU-only copyback state (`VICTORY_GPU_ONLY_COPYBACK`):
    /// `0` observing, `1` promoted (skip `gpu_writes`), `2` demoted (sticky).
    gpu_only_state: AtomicU8,
    /// `true` strictly while Mode-1 or Mode-2 `PAGE_NOACCESS` hardware tripwire(s)
    /// are armed on this buffer (proving zero title CPU reads/writes without VEH fault).
    tripwire_armed: std::sync::atomic::AtomicBool,
    /// Number of 64 KiB pieces currently promoted to Mode-2 GPU-only on this shadow.
    gpu_only_promoted_pieces: AtomicUsize,
    /// Total 64 KiB pieces covering `[start, start + len)`.
    gpu_only_total_pieces: usize,
    /// For an in-place (`port 1`) shadow: whether any non-placed shadowed copy
    /// overlaps this range so in-place GPU writes invalidate `watch` for it.
    in_place_overlap: std::sync::atomic::AtomicBool,
}

/// Ownership state for an in-place [`Shadow`]. Field order ensures `heap`
/// drops before `_view` (`UnmapViewOfFile`), and `Shadow::buffer` drops first.
struct InPlace {
    #[allow(dead_code)]
    heap: windows::Win32::Graphics::Direct3D12::ID3D12Heap,
    bias: u64,
    _view: Option<Arc<crate::inplace::OwnedView>>,
    pinned: u64,
    _private_guard: Option<crate::inplace::PrivatePlacedGuard>,
}

/// Total bytes currently committed across all demand-committed [`CpuStaging`]
/// allocations (for memory accounting / diagnostics).
static CPU_STAGING_COMMITTED_BYTES: AtomicU64 = AtomicU64::new(0);
/// Total L0 `shared_buffer` bytes live across all non-placed/non-card/non-BAR shadows.
static L0_SHADOW_BYTES: AtomicU64 = AtomicU64::new(0);
/// Subset of [`L0_SHADOW_BYTES`] belonging to `window == true` loose windows.
static L0_LOOSE_BYTES: AtomicU64 = AtomicU64::new(0);

/// Demand-committed, 64 KiB-aligned private CPU memory backing `Card::staging`
/// when `VICTORY_SEND_OWN=1` (`ARCH-05`): reserves address space up front via
/// `MEM_RESERVE` and commits 64 KiB (`PIECE`) pages strictly on demand when
/// written or merged, freeing ~450–600 MB of pure CPU RAM on untouched regions.
struct CpuStaging {
    ptr: *mut u8,
    len: usize,
    /// Per-64 KiB piece committed bitmap (`1` = `MEM_COMMIT` completed).
    committed: Box<[AtomicU64]>,
    /// Serializes `VirtualAlloc(MEM_COMMIT)` calls for this buffer.
    commit_lock: Mutex<()>,
    /// Committed bytes in this buffer (decremented from global counter on drop).
    committed_bytes: AtomicU64,
}

// SAFETY: plain virtual memory owned solely by `Card`; pieces transition
// monotonically from reserved -> committed and are freed only on drop.
unsafe impl Send for CpuStaging {}
unsafe impl Sync for CpuStaging {}

impl CpuStaging {
    fn alloc(size: u64) -> Option<(Self, *mut u8)> {
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        let len = usize::try_from(size).ok()?.checked_next_multiple_of(PIECE)?;
        if len == 0 {
            return None;
        }
        // SAFETY: OS reserves 64 KiB-aligned private virtual address space with zero commit.
        let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_READWRITE) } as *mut u8;
        if ptr.is_null() {
            return None;
        }
        let pieces = len / PIECE;
        let committed = (0..pieces.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Some((
            Self {
                ptr,
                len,
                committed,
                commit_lock: Mutex::new(()),
                committed_bytes: AtomicU64::new(0),
            },
            ptr,
        ))
    }

    #[inline]
    fn is_piece_committed(&self, piece: usize) -> bool {
        self.committed
            .get(piece / 64)
            .is_some_and(|w| (w.load(Ordering::Acquire) >> (piece % 64)) & 1 != 0)
    }

    /// Commits every 64 KiB piece intersecting `[from, to)` on demand so reads
    /// and writes never fault on reserved-only pages. Lock-free fast path once
    /// the pieces are committed.
    fn ensure_committed(&self, from: usize, to: usize) -> bool {
        let to = to.min(self.len);
        if from >= to {
            return true;
        }
        let first = from / PIECE;
        let last = (to - 1) / PIECE;
        let w0 = first / 64;
        let w1 = last / 64;
        // Fast path: check committed bitmap with Acquire ordering.
        if w0 == w1 {
            let count = last - first + 1;
            let mask = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << (first % 64) };
            if self.committed[w0].load(Ordering::Acquire) & mask == mask {
                return true;
            }
        } else {
            let mut all = true;
            for w in w0..=w1 {
                let lo = if w == w0 { first % 64 } else { 0 };
                let hi = if w == w1 { last % 64 } else { 63 };
                let count = hi - lo + 1;
                let mask = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << lo };
                if self.committed[w].load(Ordering::Acquire) & mask != mask {
                    all = false;
                    break;
                }
            }
            if all {
                return true;
            }
        }
        self.commit_slow(first, last)
    }

    #[cold]
    fn commit_slow(&self, first: usize, last: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
        let _guard = self.commit_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut p = first;
        while p <= last {
            if self.is_piece_committed(p) {
                p += 1;
                continue;
            }
            let run_start = p;
            p += 1;
            while p <= last && !self.is_piece_committed(p) {
                p += 1;
            }
            let run_pieces = p - run_start;
            let byte_off = run_start * PIECE;
            let byte_len = run_pieces * PIECE;
            // SAFETY: `[byte_off, byte_off + byte_len)` lies within our `MEM_RESERVE` region.
            let ok = unsafe {
                VirtualAlloc(self.ptr.add(byte_off).cast(), byte_len, MEM_COMMIT, PAGE_READWRITE)
            };
            if ok.is_null() {
                return false;
            }
            for piece in run_start..p {
                self.committed[piece / 64].fetch_or(1u64 << (piece % 64), Ordering::Release);
            }
            self.committed_bytes.fetch_add(byte_len as u64, Ordering::Relaxed);
            CPU_STAGING_COMMITTED_BYTES.fetch_add(byte_len as u64, Ordering::Relaxed);
        }
        true
    }

    /// Decommits page-aligned interior 64 KiB piece ranges within `[offset, offset + len)`
    /// via `VirtualFree(MEM_DECOMMIT)`, preserving the `MEM_RESERVE` virtual reservation
    /// so base pointers never move and [`Self::ensure_committed`] can re-commit on demand.
    /// Returns total bytes decommitted.
    #[allow(dead_code)]
    fn decommit_range(&self, offset: usize, len: usize) -> usize {
        let end = offset.saturating_add(len).min(self.len);
        let first_piece = offset.div_ceil(PIECE);
        let last_piece = end / PIECE;
        if first_piece >= last_piece {
            return 0;
        }
        self.decommit_pieces(first_piece, last_piece)
    }

    /// Decommits committed 64 KiB pieces in `[piece_lo, piece_hi)` via
    /// `VirtualFree(MEM_DECOMMIT)`, keeping the `MEM_RESERVE` virtual address range intact.
    fn decommit_pieces(&self, piece_lo: usize, piece_hi: usize) -> usize {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        let total_pieces = self.len / PIECE;
        let lo = piece_lo.min(total_pieces);
        let hi = piece_hi.min(total_pieces);
        if lo >= hi {
            return 0;
        }
        let _guard = self.commit_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut freed = 0usize;
        let mut p = lo;
        while p < hi {
            if !self.is_piece_committed(p) {
                p += 1;
                continue;
            }
            let run_start = p;
            p += 1;
            while p < hi && self.is_piece_committed(p) {
                p += 1;
            }
            // Clear committed bits BEFORE `VirtualFree(MEM_DECOMMIT)` so concurrent
            // callers see uncommitted pieces and acquire `commit_lock` before reading.
            for piece in run_start..p {
                self.committed[piece / 64].fetch_and(!(1u64 << (piece % 64)), Ordering::AcqRel);
            }
            let byte_off = run_start * PIECE;
            let byte_len = (p - run_start) * PIECE;
            // SAFETY: `[byte_off, byte_off + byte_len)` is a committed interior sub-range
            // of our `MEM_RESERVE` region; `MEM_DECOMMIT` leaves the reservation intact.
            let ok = unsafe { VirtualFree(self.ptr.add(byte_off).cast(), byte_len, MEM_DECOMMIT) != 0 };
            if ok {
                self.committed_bytes.fetch_sub(byte_len as u64, Ordering::Relaxed);
                CPU_STAGING_COMMITTED_BYTES.fetch_sub(byte_len as u64, Ordering::Relaxed);
                freed += byte_len;
            } else {
                for piece in run_start..p {
                    self.committed[piece / 64].fetch_or(1u64 << (piece % 64), Ordering::Release);
                }
            }
        }
        freed
    }

    /// Reads `[offset, offset + dst.len())` into `dst` without faulting or committing
    /// decommitted/reserved pieces (uncommitted pieces read as zeros). Serializes
    /// against [`Self::decommit_pieces`] via `commit_lock`.
    fn read_committed_into(&self, offset: usize, dst: &mut [u8]) {
        if dst.is_empty() || offset >= self.len {
            dst.fill(0);
            return;
        }
        let _guard = self.commit_lock.lock().unwrap_or_else(|e| e.into_inner());
        let limit = (offset + dst.len()).min(self.len);
        let mut pos = offset;
        while pos < limit {
            let piece = pos / PIECE;
            let end = ((piece + 1) * PIECE).min(limit);
            let out = &mut dst[pos - offset..end - offset];
            if self.is_piece_committed(piece) {
                // SAFETY: `piece` is committed and `commit_lock` prevents concurrent decommit.
                let src = unsafe { std::slice::from_raw_parts(self.ptr.add(pos).cast_const(), end - pos) };
                out.copy_from_slice(src);
            } else {
                out.fill(0);
            }
            pos = end;
        }
        if limit - offset < dst.len() {
            dst[limit - offset..].fill(0);
        }
    }
}

impl Drop for CpuStaging {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        let committed = *self.committed_bytes.get_mut();
        if committed > 0 {
            CPU_STAGING_COMMITTED_BYTES.fetch_sub(committed, Ordering::Relaxed);
        }
        // SAFETY: `self.ptr` came from `VirtualAlloc(..., MEM_RESERVE, ...)`.
        unsafe {
            VirtualFree(self.ptr.cast(), 0, MEM_RELEASE);
        }
    }
}

enum CardStaging {
    /// Held for its `Drop` (`VirtualFree`) while `Shadow::data` points into it.
    Cpu(#[allow(dead_code)] CpuStaging),
    Gpu(ID3D12Resource),
}

impl CardStaging {
    fn as_gpu(&self) -> Option<&ID3D12Resource> {
        match self {
            CardStaging::Cpu(_) => None,
            CardStaging::Gpu(buffer) => Some(buffer),
        }
    }
}

fn card_staging_for(
    host: &crate::host::Host,
    size: u64,
    label: impl FnOnce() -> String,
) -> Option<(CardStaging, *mut u8)> {
    if crate::host::send::cpu_staging_safe(host) {
        let (cpu, data) = CpuStaging::alloc(size)?;
        return Some((CardStaging::Cpu(cpu), data));
    }
    let (buffer, data) = host.shared_buffer(size)?;
    crate::host::name(&buffer, label);
    Some((CardStaging::Gpu(buffer), data))
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: CardStaging,
    /// The card side's piece-hash buffer (`ID3D12Resource`) and its cached GPU
    /// virtual address: write-once when shader writes are first declared, so
    /// `card_before` reads it lock-free without COM AddRef/Release or `GetGPUVirtualAddress`.
    before: std::sync::OnceLock<(ID3D12Resource, u64)>,
    /// Serializes cold first-time creation of `before` and `hash_stale`.
    before_init: Mutex<()>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<SentRuns>,
    /// Whether something was put into `dirty` since it was last taken.
    touched: std::sync::atomic::AtomicBool,
    /// Per 64 KiB piece: 1 if the CPU side modified the piece since its hash in
    /// `before` was last updated (or `before` was just created). Empty until
    /// `before` exists.
    hash_stale: Mutex<Vec<u64>>,
    /// Adaptive GPU-only copyback tracker (`VICTORY_GPU_ONLY_COPYBACK`).
    gpu_only: Mutex<GpuOnlyTracker>,
    /// Per-64 KiB-piece Mode-2 GPU-only promoted bitmap (1 bit per piece, lock-free reads).
    gpu_only_bitmap: Vec<AtomicU64>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

impl Card {
    fn new(staging: CardStaging, size: u64) -> Self {
        // +1 piece accommodates unaligned `start % PIECE != 0` straddling an extra 64 KiB boundary.
        let piece_words = ((size as usize).div_ceil(PIECE) + 1).div_ceil(64);
        let gpu_only_bitmap = (0..piece_words).map(|_| AtomicU64::new(0)).collect();
        Self {
            staging,
            before: std::sync::OnceLock::new(),
            before_init: Mutex::new(()),
            dirty: Mutex::default(),
            sent: Mutex::default(),
            touched: Default::default(),
            hash_stale: Mutex::default(),
            gpu_only: Mutex::default(),
            gpu_only_bitmap,
            size,
        }
    }
}

const GPU_ONLY_OBSERVING: u8 = 0;
const GPU_ONLY_PROMOTED: u8 = 1;
const GPU_ONLY_DEMOTED: u8 = 2;

/// Minimum warmup span (in frames) and fetched pieces before promoting a card
/// buffer to GPU-only copyback elision.
const GPU_ONLY_WARMUP_FRAMES: u64 = 60;
const GPU_ONLY_MIN_FETCHED_PIECES: u64 = 16;
const GPU_ONLY_MAX_RESETS: u32 = 4;
/// Minimum contiguous 64 KiB pieces (`4 * 64 KiB = 256 KiB`) for a Mode-2
/// piece-level GPU-only tripwire run.
const GPU_ONLY_MODE2_MIN_RUN_PIECES: usize = 4;
/// Consecutive GPU writebacks with zero CPU reads/writes on a 64 KiB piece
/// before a contiguous `>= 4`-piece run promotes in Mode-2.
const GPU_ONLY_MODE2_CONSECUTIVE: u16 = 8;
/// Boot-time population grace window (frames `< 600`): initial CPU uploads
/// during world boot reset observation without consuming `GPU_ONLY_MAX_RESETS`
/// so Mode-1 pure-GPU buffers can promote cleanly once steady-state begins (`ARCH-05`).
const GPU_ONLY_BOOT_GRACE_FRAME: u64 = 600;

/// `VICTORY_GPU_ONLY_COPYBACK`: `2` (default ON: piece-level Mode-2 with per-run
/// `PAGE_NOACCESS` hardware tripwires on `>= 256 KiB` runs of `>= min_mb` card
/// buffers); `1` legacy whole-buffer Mode-1; `0` disables.
pub(crate) fn gpu_only_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        if crate::host::upcheck::on() {
            return 0;
        }
        match std::env::var("VICTORY_GPU_ONLY_COPYBACK").as_deref().map(str::trim) {
            Ok("0") => 0,
            Ok("1") => 1,
            _ => 2,
        }
    })
}

/// `VICTORY_GPU_ONLY_TRIPWIRE`: default ON (`true`) when `gpu_only_mode() > 0`
/// (unless set to `0`), protecting both Mode-1 whole buffers and Mode-2 piece runs.
pub(crate) fn gpu_only_tripwire_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        gpu_only_mode() > 0 && std::env::var_os("VICTORY_GPU_ONLY_TRIPWIRE").is_none_or(|v| v != "0")
    })
}

/// Minimum card buffer size in bytes eligible for `VICTORY_GPU_ONLY_COPYBACK`
/// (`VICTORY_GPU_ONLY_MIN_MB`, default `1` MiB).
fn gpu_only_min_bytes() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_GPU_ONLY_MIN_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(1)
            << 20
    })
}

/// One active hardware `PAGE_NOACCESS` tripwire range registered by Mode-1
/// (whole buffer) or Mode-2 (contiguous piece run `piece_lo..piece_hi`).
struct TripwireRun {
    run_id: u64,
    start: usize,
    end: usize,
    piece_lo: usize,
    piece_hi: usize,
    shadow_id: u64,
    shadow: std::sync::Weak<Shadow>,
    mode2: bool,
}

static NEXT_TRIPWIRE_RUN_ID: AtomicU64 = AtomicU64::new(1u64 << 62);
static TRIPWIRE_RUNS: RwLock<Vec<TripwireRun>> = RwLock::new(Vec::new());
static TRIPWIRE_RUN_COUNT: AtomicUsize = AtomicUsize::new(0);
static TRIPWIRE_RUN_LO: AtomicUsize = AtomicUsize::new(usize::MAX);
static TRIPWIRE_RUN_HI: AtomicUsize = AtomicUsize::new(0);

fn tripwire_runs_refresh_bounds(runs: &[TripwireRun]) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for r in runs {
        lo = lo.min(r.start);
        hi = hi.max(r.end);
    }
    TRIPWIRE_RUN_LO.store(lo, Ordering::Relaxed);
    TRIPWIRE_RUN_HI.store(hi, Ordering::Relaxed);
    TRIPWIRE_RUN_COUNT.store(runs.len(), Ordering::Release);
}

fn tripwire_run_register(entry: TripwireRun) {
    let mut runs = TRIPWIRE_RUNS.write().unwrap_or_else(|e| e.into_inner());
    runs.retain(|r| r.run_id != entry.run_id && r.shadow.strong_count() > 0);
    runs.push(entry);
    tripwire_runs_refresh_bounds(&runs);
}

fn tripwire_run_take(run_id: u64) -> Option<TripwireRun> {
    if TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let mut runs = TRIPWIRE_RUNS.write().unwrap_or_else(|e| e.into_inner());
    let pos = runs.iter().position(|r| r.run_id == run_id)?;
    let removed = runs.swap_remove(pos);
    tripwire_runs_refresh_bounds(&runs);
    Some(removed)
}

fn tripwire_runs_remove_shadow(shadow_id: u64) -> Vec<u64> {
    if TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let mut runs = TRIPWIRE_RUNS.write().unwrap_or_else(|e| e.into_inner());
    let mut removed_ids = Vec::new();
    let mut i = 0;
    while i < runs.len() {
        if runs[i].shadow_id == shadow_id {
            removed_ids.push(runs.swap_remove(i).run_id);
        } else {
            i += 1;
        }
    }
    if !removed_ids.is_empty() {
        tripwire_runs_refresh_bounds(&runs);
    }
    removed_ids
}

/// Lock-free bounds-checked probe for whether `[va, va + len)` intersects any
/// active Mode-1 or Mode-2 `PAGE_NOACCESS` tripwire managed by `shadow` (Spec-Trap #4).
#[inline]
pub(crate) fn is_tripwired(va: usize, len: usize) -> bool {
    if len == 0 || TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let end = va.saturating_add(len);
    if end <= TRIPWIRE_RUN_LO.load(Ordering::Relaxed)
        || va >= TRIPWIRE_RUN_HI.load(Ordering::Relaxed)
    {
        return false;
    }
    let runs = TRIPWIRE_RUNS.read().unwrap_or_else(|e| e.into_inner());
    runs.iter().any(|r| r.start < end && r.end > va)
}

/// Returns sorted, disjoint `[start, end)` sub-ranges of `[va, va + len)` that
/// are currently covered by active `shadow` tripwires so `safecopy` can split
/// partially-overlapping reads without touching `PAGE_NOACCESS` pages.
pub(crate) fn tripwired_overlap_runs(va: usize, len: usize) -> Vec<(usize, usize)> {
    if len == 0 || TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let end = va.saturating_add(len);
    if end <= TRIPWIRE_RUN_LO.load(Ordering::Relaxed)
        || va >= TRIPWIRE_RUN_HI.load(Ordering::Relaxed)
    {
        return Vec::new();
    }
    let mut raw: Vec<(usize, usize)> = {
        let runs = TRIPWIRE_RUNS.read().unwrap_or_else(|e| e.into_inner());
        runs.iter()
            .filter(|r| r.start < end && r.end > va)
            .map(|r| (r.start.max(va), r.end.min(end)))
            .collect()
    };
    if raw.len() <= 1 {
        return raw;
    }
    raw.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(raw.len());
    for (s, e) in raw {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// Satisfies an internal Victory read of guest VA `[va, va + dst.len())` directly
/// from host-side `shadow.data` when the entire range is covered by active
/// Mode-1/Mode-2 tripwire(s), WITHOUT disarming `watch::tripwire_touch` and
/// WITHOUT faulting `PAGE_NOACCESS` (Spec-Trap #4).
pub(crate) fn read_tripwired_into(va: usize, dst: &mut [u8]) -> bool {
    let len = dst.len();
    if len == 0 {
        return true;
    }
    if TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let end = va.saturating_add(len);
    if end <= TRIPWIRE_RUN_LO.load(Ordering::Relaxed)
        || va >= TRIPWIRE_RUN_HI.load(Ordering::Relaxed)
    {
        return false;
    }
    let mut matches: Vec<(usize, usize, Arc<Shadow>)> = {
        let runs = TRIPWIRE_RUNS.read().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for r in runs.iter() {
            if r.start < end && r.end > va
                && let Some(shadow) = r.shadow.upgrade()
            {
                out.push((r.start.max(va), r.end.min(end), shadow));
            }
        }
        out
    };
    if matches.is_empty() {
        return false;
    }
    matches.sort_unstable_by_key(|m| m.0);
    // Verify the matched active tripwire runs completely cover `[va, end)`.
    let mut covered = va;
    for &(s, e, _) in &matches {
        if s > covered {
            return false;
        }
        covered = covered.max(e);
        if covered >= end {
            break;
        }
    }
    if covered < end {
        return false;
    }
    let mut cursor = va;
    for (s, e, shadow) in matches {
        let lo = s.max(cursor);
        let hi = e.min(end);
        if lo >= hi {
            continue;
        }
        let off = lo.saturating_sub(shadow.start as usize);
        let dst_slice = &mut dst[lo - va..hi - va];
        shadow.read_staging_into(off, dst_slice);
        cursor = hi;
        if cursor >= end {
            break;
        }
    }
    true
}

/// Fast-path check for `land::ordered` to skip queuing GPU readbacks for pieces
/// of `shadow_id` currently covered by an active Mode-1/Mode-2 tripwire run.
#[inline]
pub(crate) fn is_gpu_only_piece_va(shadow_id: u64, va: usize, len: usize) -> bool {
    if len == 0 || TRIPWIRE_RUN_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let end = va.saturating_add(len);
    if end <= TRIPWIRE_RUN_LO.load(Ordering::Relaxed)
        || va >= TRIPWIRE_RUN_HI.load(Ordering::Relaxed)
    {
        return false;
    }
    let runs = TRIPWIRE_RUNS.read().unwrap_or_else(|e| e.into_inner());
    runs.iter()
        .any(|r| r.shadow_id == shadow_id && r.start <= va && end <= r.end)
}

/// Spec-Trap #8 mutual exclusion query: returns `true` if `[start, start + len)`
/// overlaps any active `PAGE_NOACCESS` tripwire OR any active GPU-only/Mode-2
/// tripwire-eligible card buffer so `OpenExistingHeapFromAddress1` is NEVER
/// opened on pages that need or hold tripwires.
pub(crate) fn overlaps_gpu_only_or_tripwired(start: u64, len: u64) -> bool {
    if len == 0 {
        return false;
    }
    if is_tripwired(start as usize, len as usize) {
        return true;
    }
    if gpu_only_mode() == 0 {
        return false;
    }
    let hi = GPU_ONLY_BOUNDS[1].load(Ordering::Relaxed);
    let end = start.saturating_add(len);
    if hi == 0 || start >= hi || end <= GPU_ONLY_BOUNDS[0].load(Ordering::Relaxed) {
        return false;
    }
    let watched = GPU_ONLY_WATCHED.read().unwrap_or_else(|e| e.into_inner());
    watched.iter().any(|&(s_start, s_end, _, ref weak)| {
        s_start < end && s_end > start && weak.strong_count() > 0
    })
}

/// Min start and max end of eligible card buffers for fast-path CPU read
/// demotion (`[0, 0]` when none).
static GPU_ONLY_BOUNDS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Active eligible (`>= min_bytes`, non-demoted) card buffers `(start, end, id, weak)`
/// scanned without `Weak::upgrade` or heap allocation in `note_cpu_read`.
static GPU_ONLY_WATCHED: std::sync::RwLock<Vec<(u64, u64, u64, std::sync::Weak<Shadow>)>> =
    std::sync::RwLock::new(Vec::new());

/// `[0]` promoted buffers currently active, `[1]` promoted bytes,
/// `[2]` total demotions, `[3]` skipped `gpu_writes` calls since last log,
/// `[4]` skipped `gpu_writes` bytes since last log.
static GPU_ONLY_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn gpu_only_register(shadow: &Arc<Shadow>) {
    if gpu_only_mode() == 0 || shadow.len < gpu_only_min_bytes() {
        return;
    }
    let start = shadow.start;
    let end = start.saturating_add(shadow.len);
    let _ = GPU_ONLY_BOUNDS[0].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(if cur == 0 { start } else { cur.min(start) })
    });
    GPU_ONLY_BOUNDS[1].fetch_max(end, Ordering::Relaxed);
    let mut watched = GPU_ONLY_WATCHED.write().unwrap_or_else(|e| e.into_inner());
    watched.retain(|(_, _, _, w)| w.strong_count() > 0);
    watched.push((start, end, shadow.id, Arc::downgrade(shadow)));
}

#[derive(Default)]
struct GpuOnlyTracker {
    window_start_frame: u64,
    fetched_pieces: u64,
    up_bits: Vec<u64>,
    fetched_bits: Vec<u64>,
    /// Dynamic per-64-KiB-piece bitset updated from both warmup GPU fetches AND
    /// post-promotion declared GPU write ranges (`ARCH-06`: never frozen at frame +60).
    gpu_dirty_bits: Vec<u64>,
    resets: u32,
    /// Mode-2: per-piece consecutive GPU writeback count since last CPU upload/demotion.
    piece_fetches: Vec<u16>,
    /// Mode-2: per-piece currently promoted bitmap (mirrors `Card::gpu_only_bitmap`).
    piece_promoted: Vec<u64>,
    /// Mode-2: per-piece sticky demoted bitmap (prevents promote/demote thrashing).
    piece_demoted: Vec<u64>,
    /// Mode-2: currently armed tripwire runs `(run_id, piece_lo, piece_hi)`.
    active_runs: Vec<(u64, usize, usize)>,
}

impl GpuOnlyTracker {
    fn set_bit(bits: &mut Vec<u64>, piece: usize) {
        let w = piece / 64;
        if w >= bits.len() {
            bits.resize(w + 1, 0);
        }
        bits[w] |= 1u64 << (piece % 64);
    }

    fn clear_bit(bits: &mut [u64], piece: usize) {
        if let Some(w) = bits.get_mut(piece / 64) {
            *w &= !(1u64 << (piece % 64));
        }
    }

    fn has_bit(bits: &[u64], piece: usize) -> bool {
        bits.get(piece / 64).is_some_and(|&w| (w >> (piece % 64)) & 1 != 0)
    }

    fn clear_window(&mut self) {
        // Start the next warmup window on the first GPU fetch after the reset.
        self.window_start_frame = 0;
        self.fetched_pieces = 0;
        self.up_bits.iter_mut().for_each(|w| *w = 0);
        self.fetched_bits.iter_mut().for_each(|w| *w = 0);
        self.gpu_dirty_bits.iter_mut().for_each(|w| *w = 0);
    }

    fn reset_window(&mut self) -> bool {
        self.resets += 1;
        self.clear_window();
        self.resets > GPU_ONLY_MAX_RESETS
    }

    /// Dynamically records declared GPU write piece indices `[first_piece, last_piece)`
    /// even after promotion so newly written GPU pieces are never silently overwritten
    /// by a later CPU upload (`ARCH-06`).
    fn note_declared_write(&mut self, first_piece: usize, last_piece: usize) {
        for p in first_piece..last_piece {
            Self::set_bit(&mut self.gpu_dirty_bits, p);
        }
    }

    /// Mode-2: records a CPU upload over `[first_piece, last_piece)`. Resets
    /// consecutive GPU writeback counters on those pieces and returns any active
    /// promoted runs overlapping them that must be locally demoted.
    fn note_upload_mode2(&mut self, first_piece: usize, last_piece: usize) -> Vec<(u64, usize, usize)> {
        if first_piece >= last_piece {
            return Vec::new();
        }
        for p in first_piece..last_piece {
            if let Some(cnt) = self.piece_fetches.get_mut(p) {
                *cnt = 0;
            }
            Self::set_bit(&mut self.up_bits, p);
        }
        self.demote_overlapping_runs(first_piece, last_piece)
    }

    /// Mode-2: removes and returns all active promoted runs overlapping
    /// `[first_piece, last_piece)`, marking their pieces demoted.
    fn demote_overlapping_runs(&mut self, first_piece: usize, last_piece: usize) -> Vec<(u64, usize, usize)> {
        if first_piece >= last_piece {
            return Vec::new();
        }
        let mut hit = Vec::new();
        let mut i = 0;
        while i < self.active_runs.len() {
            let (_, lo, hi) = self.active_runs[i];
            if lo < last_piece && hi > first_piece {
                let run = self.active_runs.swap_remove(i);
                for p in run.1..run.2 {
                    Self::clear_bit(&mut self.piece_promoted, p);
                    Self::set_bit(&mut self.piece_demoted, p);
                    if let Some(cnt) = self.piece_fetches.get_mut(p) {
                        *cnt = 0;
                    }
                }
                hit.push(run);
            } else {
                i += 1;
            }
        }
        hit
    }

    /// Mode-2: marks one specific run `(run_id, piece_lo, piece_hi)` demoted.
    /// Returns `true` if the run was active.
    fn demote_run_id(&mut self, run_id: u64, piece_lo: usize, piece_hi: usize) -> bool {
        let was_active = if let Some(pos) = self.active_runs.iter().position(|r| r.0 == run_id) {
            self.active_runs.swap_remove(pos);
            true
        } else {
            false
        };
        for p in piece_lo..piece_hi {
            Self::clear_bit(&mut self.piece_promoted, p);
            Self::set_bit(&mut self.piece_demoted, p);
            if let Some(cnt) = self.piece_fetches.get_mut(p) {
                *cnt = 0;
            }
        }
        was_active
    }

    /// Mode-2: increments consecutive GPU writeback counters for `pieces` and
    /// returns maximal contiguous runs of `>= min_run_pieces` pieces ready to
    /// promote (`piece_fetches[p] >= min_consecutive` and neither promoted nor demoted).
    fn note_fetched_mode2(
        &mut self,
        pieces: &[usize],
        total_pieces: usize,
        min_consecutive: u16,
        min_run_pieces: usize,
    ) -> Vec<(usize, usize)> {
        if pieces.is_empty() || total_pieces < min_run_pieces {
            return Vec::new();
        }
        if self.piece_fetches.len() < total_pieces {
            self.piece_fetches.resize(total_pieces, 0);
        }
        let mut any_ready = false;
        for &p in pieces {
            if p >= total_pieces {
                continue;
            }
            Self::set_bit(&mut self.gpu_dirty_bits, p);
            if Self::has_bit(&self.piece_promoted, p) || Self::has_bit(&self.piece_demoted, p) {
                continue;
            }
            let c = self.piece_fetches[p].saturating_add(1);
            self.piece_fetches[p] = c;
            if c >= min_consecutive {
                any_ready = true;
            }
        }
        if !any_ready {
            return Vec::new();
        }
        let mut runs = Vec::new();
        let mut p = 0usize;
        while p < total_pieces {
            if self.piece_fetches[p] >= min_consecutive
                && !Self::has_bit(&self.piece_promoted, p)
                && !Self::has_bit(&self.piece_demoted, p)
            {
                let lo = p;
                p += 1;
                while p < total_pieces
                    && self.piece_fetches[p] >= min_consecutive
                    && !Self::has_bit(&self.piece_promoted, p)
                    && !Self::has_bit(&self.piece_demoted, p)
                {
                    p += 1;
                }
                if p - lo >= min_run_pieces {
                    runs.push((lo, p));
                }
            } else {
                p += 1;
            }
        }
        runs
    }

    fn note_run_promoted(&mut self, run_id: u64, piece_lo: usize, piece_hi: usize) {
        self.active_runs.push((run_id, piece_lo, piece_hi));
        for p in piece_lo..piece_hi {
            Self::set_bit(&mut self.piece_promoted, p);
        }
    }

    fn mark_pieces_demoted(&mut self, piece_lo: usize, piece_hi: usize) {
        for p in piece_lo..piece_hi {
            Self::clear_bit(&mut self.piece_promoted, p);
            Self::set_bit(&mut self.piece_demoted, p);
            if let Some(cnt) = self.piece_fetches.get_mut(p) {
                *cnt = 0;
            }
        }
    }

    /// Records CPU-uploaded piece indices `[first_piece, last_piece)` (post-boot steady state).
    #[cfg(test)]
    fn note_upload(&mut self, mode: u8, promoted: bool, first_piece: usize, last_piece: usize) -> bool {
        self.note_upload_at(mode, promoted, GPU_ONLY_BOOT_GRACE_FRAME, first_piece, last_piece)
    }

    /// Records CPU-uploaded piece indices `[first_piece, last_piece)` at `frame`.
    /// Returns `true` if the buffer should be sticky-demoted. During boot frames
    /// (`frame < GPU_ONLY_BOOT_GRACE_FRAME`), unpromoted observation resets do not
    /// burn `GPU_ONLY_MAX_RESETS` so steady-state Mode-1 promotion still succeeds (`ARCH-05`).
    fn note_upload_at(
        &mut self,
        mode: u8,
        promoted: bool,
        frame: u64,
        first_piece: usize,
        last_piece: usize,
    ) -> bool {
        if first_piece >= last_piece {
            return false;
        }
        if mode == 1 {
            if promoted {
                return true;
            }
            // In Mode 1, unpromoted uploads (including world-load/streaming setup well past
            // frame 600) simply restart the 60-frame quiet observation window without sticky-
            // banning the buffer. Once promoted, any CPU upload or tripwire read sticky-demotes.
            let _ = frame;
            self.clear_window();
            return false;
        }
        // Mode 2 (disjoint): demote/reset if any uploaded piece overlaps a GPU-fetched
        // OR post-promotion declared GPU-dirty piece.
        let mut overlaps = false;
        for p in first_piece..last_piece {
            if Self::has_bit(&self.fetched_bits, p) || Self::has_bit(&self.gpu_dirty_bits, p) {
                overlaps = true;
                break;
            }
            if !promoted {
                Self::set_bit(&mut self.up_bits, p);
            }
        }
        if overlaps {
            if promoted {
                return true;
            }
            if frame < GPU_ONLY_BOOT_GRACE_FRAME {
                self.clear_window();
                return false;
            }
            return self.reset_window();
        }
        false
    }

    /// Records GPU-fetched piece indices at `frame`. Returns `(promote, demote)`.
    fn note_fetched(&mut self, mode: u8, frame: u64, pieces: &[usize]) -> (bool, bool) {
        if pieces.is_empty() {
            return (false, false);
        }
        if self.window_start_frame == 0 {
            self.window_start_frame = frame;
        }
        for &p in pieces {
            if mode == 2 && Self::has_bit(&self.up_bits, p) {
                return (false, self.reset_window());
            }
            Self::set_bit(&mut self.fetched_bits, p);
            Self::set_bit(&mut self.gpu_dirty_bits, p);
        }
        self.fetched_pieces += pieces.len() as u64;
        let ready = self.fetched_pieces >= GPU_ONLY_MIN_FETCHED_PIECES
            && frame.saturating_sub(self.window_start_frame) >= GPU_ONLY_WARMUP_FRAMES;
        (ready, false)
    }
}

/// Checks whether any other live `Shadow` (buffer, loose window, or exact view)
/// overlaps `[start, start + len)`.
fn has_overlapping_shadow(start: u64, len: u64, self_id: u64) -> bool {
    let Some(end) = start.checked_add(len) else {
        return true;
    };
    {
        let map = buffers_read();
        for (_, entry) in map.range(..=(start, usize::MAX)).rev().take(64) {
            if entry.end > start
                && let Some(s) = &entry.shadow
                && s.id != self_id
            {
                return true;
            }
        }
        for (_, entry) in map.range((start.saturating_add(1), 0)..(end, 0)) {
            if let Some(s) = &entry.shadow
                && s.id != self_id
            {
                return true;
            }
        }
    }
    {
        let exact = EXACT.lock().unwrap_or_else(|e| e.into_inner());
        if !exact.1.is_empty() {
            const EXACT_MOST: u64 = 64 << 20;
            for (&(ex_start, ex_len), (s, _)) in exact.1.range((start.saturating_sub(EXACT_MOST), 0)..(end, 0)) {
                if ex_start.saturating_add(ex_len) > start && s.id != self_id {
                    return true;
                }
            }
        }
    }
    {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if !loose.is_empty() {
            for (&(w_start, w_len), entry) in loose.range((start.saturating_sub(WINDOW), 0)..(end, 0)) {
                if w_start.saturating_add(w_len) > start
                    && entry.as_ready().is_some_and(|s| s.id != self_id)
                {
                    return true;
                }
            }
        }
    }
    crate::port::overlaps(start, len)
}

/// Fast-path bounds `[min_start, max_end]` for `GPU_ONLY_WATCHED` (`max_end == 0`
/// when `VICTORY_GPU_ONLY_COPYBACK=0` or no eligible card buffers are active).
#[inline]
pub(crate) fn gpu_only_bounds() -> (u64, u64) {
    if gpu_only_mode() == 0 {
        return (0, 0);
    }
    let hi = GPU_ONLY_BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 {
        return (0, 0);
    }
    (GPU_ONLY_BOUNDS[0].load(Ordering::Relaxed), hi)
}

/// Demotes any card buffer overlapping `[at, at + len)` (other than `skip_id`)
/// when a CPU step or another host copy reads/aliases its title memory.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn note_cpu_read(at: usize, len: usize) {
    if gpu_only_mode() == 0 || len == 0 {
        return;
    }
    demote_overlapping_gpu_only(at as u64, len as u64, 0, "cpu read");
}

/// Batched [`note_cpu_read`] for a list's CPU reads, acquiring
/// `GPU_ONLY_WATCHED.read()` at most once across all ranges.
pub(crate) fn note_cpu_reads(ranges: &[(usize, usize)]) {
    if ranges.is_empty() || gpu_only_mode() == 0 {
        return;
    }
    let hi = GPU_ONLY_BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 {
        return;
    }
    let lo = GPU_ONLY_BOUNDS[0].load(Ordering::Relaxed);
    let mut hits: Vec<(Arc<Shadow>, u64, u64)> = Vec::new();
    {
        let watched = GPU_ONLY_WATCHED.read().unwrap_or_else(|e| e.into_inner());
        if watched.is_empty() {
            return;
        }
        for &(at, len) in ranges {
            if len == 0 {
                continue;
            }
            let start = at as u64;
            let end = start.saturating_add(len as u64);
            if start >= hi || end <= lo {
                continue;
            }
            for &(s_start, s_end, _, ref weak) in watched.iter() {
                if s_start < end && s_end > start
                    && let Some(shadow) = weak.upgrade()
                {
                    let ov_lo = s_start.max(start);
                    let ov_hi = s_end.min(end);
                    hits.push((shadow, ov_lo, ov_hi));
                }
            }
        }
    }
    if hits.is_empty() {
        return;
    }
    let mode = gpu_only_mode();
    for (shadow, ov_lo, ov_hi) in &hits {
        if mode == 2 {
            shadow.demote_mode2_overlap(*ov_lo, *ov_hi - *ov_lo, "cpu read");
        } else {
            shadow.demote_gpu_only("cpu read");
        }
    }
    if mode != 2 {
        let mut watched = GPU_ONLY_WATCHED.write().unwrap_or_else(|e| e.into_inner());
        watched.retain(|(_, _, _, w)| {
            w.upgrade()
                .is_some_and(|s| s.gpu_only_state.load(Ordering::Relaxed) != GPU_ONLY_DEMOTED)
        });
    }
}

fn demote_overlapping_gpu_only(start: u64, len: u64, skip_id: u64, reason: &'static str) {
    if gpu_only_mode() == 0 || len == 0 {
        return;
    }
    let hi = GPU_ONLY_BOUNDS[1].load(Ordering::Relaxed);
    let end = start.saturating_add(len);
    if hi == 0 || start >= hi || end <= GPU_ONLY_BOUNDS[0].load(Ordering::Relaxed) {
        return;
    }
    let mut hits: Vec<Arc<Shadow>> = Vec::new();
    {
        let watched = GPU_ONLY_WATCHED.read().unwrap_or_else(|e| e.into_inner());
        for &(s_start, s_end, s_id, ref weak) in watched.iter() {
            if s_id != skip_id && s_start < end && s_end > start {
                if let Some(shadow) = weak.upgrade() {
                    hits.push(shadow);
                }
            }
        }
    }
    if hits.is_empty() {
        return;
    }
    let mode = gpu_only_mode();
    for shadow in &hits {
        if mode == 2 {
            shadow.demote_mode2_overlap(start, len, reason);
        } else {
            shadow.demote_gpu_only(reason);
        }
    }
    if mode != 2 {
        let mut watched = GPU_ONLY_WATCHED.write().unwrap_or_else(|e| e.into_inner());
        watched.retain(|(_, _, _, w)| {
            w.upgrade()
                .is_some_and(|s| s.gpu_only_state.load(Ordering::Relaxed) != GPU_ONLY_DEMOTED)
        });
    }
}

/// Called by `watch.rs` after disarming a Mode-1 or Mode-2 `PAGE_NOACCESS`
/// tripwire on `id`: in Mode-2 demotes ONLY the faulted run (`piece_lo..piece_hi`),
/// leaving all other promoted runs on the same buffer intact; in Mode-1
/// permanently demotes the whole buffer.
pub(crate) fn tripwire_demote(id: u64, fault_addr: usize, access: usize, rip: usize, reason: &'static str) {
    if let Some(run) = tripwire_run_take(id) {
        if let Some(shadow) = run.shadow.upgrade() {
            if run.mode2 {
                shadow.demote_mode2_run(
                    run.run_id,
                    run.piece_lo,
                    run.piece_hi,
                    reason,
                    Some((fault_addr, access, rip)),
                );
            } else {
                shadow.demote_gpu_only_inner(reason, Some((fault_addr, access, rip)));
                let mut watched = GPU_ONLY_WATCHED.write().unwrap_or_else(|e| e.into_inner());
                watched.retain(|(_, _, _, w)| {
                    w.upgrade()
                        .is_some_and(|s| s.gpu_only_state.load(Ordering::Relaxed) != GPU_ONLY_DEMOTED)
                });
            }
        }
        return;
    }
    let target = {
        let watched = GPU_ONLY_WATCHED.read().unwrap_or_else(|e| e.into_inner());
        watched
            .iter()
            .find(|&&(_, _, s_id, _)| s_id == id)
            .and_then(|(_, _, _, w)| w.upgrade())
    };
    if let Some(shadow) = target {
        shadow.demote_gpu_only_inner(reason, Some((fault_addr, access, rip)));
        let mut watched = GPU_ONLY_WATCHED.write().unwrap_or_else(|e| e.into_inner());
        watched.retain(|(_, _, _, w)| {
            w.upgrade()
                .is_some_and(|s| s.gpu_only_state.load(Ordering::Relaxed) != GPU_ONLY_DEMOTED)
        });
    }
}

/// Periodic diagnostic summary for `VICTORY_GPU_ONLY_COPYBACK`.
pub(crate) fn gpu_only_log(frames: f64) {
    let mode = gpu_only_mode();
    if mode == 0 {
        return;
    }
    let promoted = GPU_ONLY_STATS[0].load(Ordering::Relaxed);
    let promoted_mb = GPU_ONLY_STATS[1].load(Ordering::Relaxed) >> 20;
    let demoted = GPU_ONLY_STATS[2].load(Ordering::Relaxed);
    let skipped = GPU_ONLY_STATS[3].swap(0, Ordering::Relaxed) as f64 / frames.max(1.0);
    let skipped_mb = GPU_ONLY_STATS[4].swap(0, Ordering::Relaxed) as f64 / 1e6 / frames.max(1.0);
    let (armed, tripped) = crate::watch::tripwire_stats();
    tracing::info!(
        "gpu-only card copyback (VICTORY_GPU_ONLY_COPYBACK={mode}, min={} MB, tripwire={}): {promoted} buffers promoted ({promoted_mb} MB), \
         {demoted} demoted (tripwire armed={armed}, tripped={tripped}); skipped {skipped:.1} gpu_writes/f ({skipped_mb:.1} MB/f)",
        gpu_only_min_bytes() >> 20,
        if gpu_only_tripwire_on() { "on" } else { "off" },
    );
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// Default `(min_mb, max_mb)` for [`card_limits`] chosen from the adapter's
/// local VRAM budget (`DXGI_QUERY_VIDEO_MEMORY_INFO.Budget`), in bytes.
///
/// - Unknown (`None`, e.g. WARP) or `<= 12 GiB` (8–12 GB cards, whose WDDM
///   budget is below 12 GiB): `(16, 512)` MiB, matching the 12 GB baseline.
/// - `> 12 GiB` and `< 20 GiB` (16 GB cards, whose WDDM budget is typically
///   ~13.5–15.2 GiB): `(4, 2048)` MiB.
/// - `>= 20 GiB` (24–32+ GB cards): `(1, (budget_mb / 4).min(8192))` MiB,
///   capping at 25% of local budget (at most 8 GiB) so "before" twins (up to 2x)
///   still leave headroom for textures and render targets.
fn default_card_limits_mb(local_vram_budget: Option<u64>) -> (u64, u64) {
    let Some(budget_bytes) = local_vram_budget else {
        return (16, 512);
    };
    let budget_mb = budget_bytes >> 20;
    const TWELVE_GIB_MB: u64 = 12 * 1024;
    const TWENTY_GIB_MB: u64 = 20 * 1024;
    if budget_mb <= TWELVE_GIB_MB {
        (16, 512)
    } else if budget_mb < TWENTY_GIB_MB {
        (4, 2048)
    } else {
        (1, (budget_mb / 4).min(8192))
    }
}

/// Resolve `(min_bytes, max_bytes)` from the adapter's local VRAM budget and
/// optional explicit MiB overrides from `VICTORY_CARD_BUFFERS_MIN_MB` /
/// `VICTORY_CARD_BUFFERS_MB`.
fn resolve_card_limits(
    local_vram_budget: Option<u64>,
    min_mb_override: Option<u64>,
    max_mb_override: Option<u64>,
) -> (u64, u64) {
    const MIB: u64 = 1 << 20;
    let (default_min_mb, default_max_mb) = default_card_limits_mb(local_vram_budget);
    let min_bytes = min_mb_override.unwrap_or(default_min_mb).saturating_mul(MIB).max(MIB);
    let max_bytes = max_mb_override.unwrap_or(default_max_mb).saturating_mul(MIB);
    (min_bytes, max_bytes)
}

/// Optional explicit VRAM headroom override (`VICTORY_CARD_BUFFERS_VRAM_HEADROOM_MB`),
/// in MiB (`Some(0)` disables the headroom cap and uses only the percentage cap).
fn card_vram_headroom_override_mb() -> Option<u64> {
    static OVERRIDE: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("VICTORY_CARD_BUFFERS_VRAM_HEADROOM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    })
}

/// Minimum local VRAM headroom (in bytes) kept below `DXGI_QUERY_VIDEO_MEMORY_INFO.Budget`
/// before new card buffers stay in shared memory.
///
/// Opt-in via `VICTORY_CARD_BUFFERS_VRAM_HEADROOM_MB` (`0` / unset uses only
/// the percentage cap, avoiding premature shared-memory snapshot fallback on
/// 16 GB adapters).
fn resolve_card_vram_headroom(
    _budget: u64,
    _size: u64,
    _default_min_bytes: u64,
    explicit_headroom_mb: Option<u64>,
) -> u64 {
    const MIB: u64 = 1 << 20;
    explicit_headroom_mb.unwrap_or(0).saturating_mul(MIB)
}

/// Maximum local VRAM usage (in bytes) at which a new card buffer may be allocated:
/// the tighter of `budget * hundredths / 100` and `budget - headroom_bytes`.
fn resolve_card_vram_ceiling(budget: u64, hundredths: u64, headroom_bytes: u64) -> u64 {
    let pct_cap = (budget / 100).saturating_mul(hundredths);
    let headroom_cap = budget.saturating_sub(headroom_bytes);
    pct_cap.min(headroom_cap)
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` / `VICTORY_CARD_BUFFERS_MIN_KB` and
/// `VICTORY_CARD_BUFFERS_MAX_MB` (or `VICTORY_CARD_BUFFERS_MB`).
/// Defaults scale with the adapter's local VRAM budget (16 MiB / 512 MiB on
/// `<= 12 GiB` budgets, 4 MiB / 2048 MiB on 16 GB cards, 1 MiB / up to 8192 MiB
/// on `>= 20 GiB` budgets), and explicit env vars always override.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let env_mb = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok());
        let vram_budget = crate::host::host().and_then(|h| h.video_memory()).map(|[budget, ..]| budget);
        let max_override = env_mb("VICTORY_CARD_BUFFERS_MAX_MB").or_else(|| env_mb("VICTORY_CARD_BUFFERS_MB"));
        let (mut min_bytes, max_bytes) = resolve_card_limits(
            vram_budget,
            env_mb("VICTORY_CARD_BUFFERS_MIN_MB"),
            max_override,
        );
        // `VICTORY_CARD_BUFFERS_MIN_KB` (after fxceless c36971b2) says the
        // least size in KiB, so that it can go under 1 MiB: 64 is every title
        // buffer. Copies under 1 MiB are nearly all there are and few bytes;
        // in the PC's memory every load, store and atomic a shader does on
        // them crosses the bus.
        if let Some(kib) = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB").ok().and_then(|v| v.trim().parse::<u64>().ok())
            && kib > 0
        {
            min_bytes = kib << 10;
        }
        if card_all() {
            min_bytes = 0;
        }
        let headroom_mb = vram_budget.map(|b| {
            let default_min = default_card_limits_mb(Some(b)).0 << 20;
            resolve_card_vram_headroom(b, min_bytes, default_min, card_vram_headroom_override_mb()) >> 20
        });
        tracing::info!(
            vram_budget_mb = vram_budget.map(|b| b >> 20),
            card_buffers_min_kb = min_bytes >> 10,
            card_buffers_max_mb = max_bytes >> 20,
            card_buffers_vram_headroom_mb = headroom_mb,
            "card buffer limits"
        );
        (min_bytes, max_bytes)
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Timestamp (`crate::host::coarse_ms`) when the most recent card buffer was
/// allocated, so [`card_room_kept`] decays cleanly to 0 after world-load settle.
static CARD_LAST_ADDED_MS: AtomicU64 = AtomicU64::new(0);

const CARD_SETTLE_DECAY_START_MS: u64 = 5_000;
const CARD_SETTLE_DECAY_END_MS: u64 = 10_000;

/// Pure calculation for [`card_room_kept`]: caps the reservation at `SEEN`
/// (800 MiB, covering the ~738–762 MiB world-load buffer burst with headroom
/// without locking ~286 MiB of phantom VRAM once buffers settle).
fn card_room_kept_for(limit: u64, live: u64, refused: u64) -> u64 {
    const SEEN: u64 = 800 << 20;
    limit.min(SEEN).saturating_sub(live.saturating_add(refused))
}

/// Applies post-world-settle decay to [`card_room_kept_for`]: holds full reserve
/// before any card buffer exists or within `5 s` of the last allocation, decays
/// linearly over `5..10 s`, and releases to `0` at `>= 10 s` so phantom headroom
/// is never permanently locked (`ARCH-01` / `ARCH-06`).
fn card_room_kept_decayed(limit: u64, live: u64, refused: u64, elapsed_ms: u64) -> u64 {
    let base = card_room_kept_for(limit, live, refused);
    if base == 0 || live == 0 || elapsed_ms <= CARD_SETTLE_DECAY_START_MS {
        return base;
    }
    if elapsed_ms >= CARD_SETTLE_DECAY_END_MS {
        return 0;
    }
    let remaining = CARD_SETTLE_DECAY_END_MS - elapsed_ms;
    let span = CARD_SETTLE_DECAY_END_MS - CARD_SETTLE_DECAY_START_MS;
    base.saturating_mul(remaining) / span
}

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    let live = CARD_BYTES[0].load(Ordering::Relaxed);
    let last_added = CARD_LAST_ADDED_MS.load(Ordering::Relaxed);
    let elapsed_ms = if live > 0 && last_added != 0 {
        crate::host::coarse_ms().saturating_sub(last_added)
    } else {
        0
    };
    card_room_kept_decayed(
        card_limits().1,
        live,
        CARD_REFUSED[1].load(Ordering::Relaxed),
        elapsed_ms,
    )
}

/// The copies in the card's memory: `(start, end, weak)` so overlap checks
/// (`send_copied`) skip non-overlapping buffers without upgrading `Weak`.
static CARDED: Mutex<Vec<(u64, u64, std::sync::Weak<Shadow>)>> = Mutex::new(Vec::new());

/// Fast-path bounding box `[min_start, max_end]` of `CARDED` (`max_end == 0` when empty).
static CARDED_BOUNDS: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];

fn push_carded(shadow: &Arc<Shadow>) {
    let start = shadow.start;
    let end = start.saturating_add(shadow.len);
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).push((start, end, Arc::downgrade(shadow)));
    CARDED_BOUNDS[0].fetch_min(start, Ordering::Relaxed);
    CARDED_BOUNDS[1].fetch_max(end, Ordering::Relaxed);
}

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

pub(crate) fn live_span_bytes() -> u64 {
    CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed)
}

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|(_, _, w)| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The number of 64 KiB piece hash slots allocated in `before` for a card
/// copy of `size` bytes.
fn card_hash_pieces(size: u64) -> usize {
    (size / PIECE as u64 + 2) as usize
}

/// Marks bits `[first, last)` in `bits`.
fn mark_stale_bits(bits: &mut [u64], first: usize, last: usize) {
    let cap = bits.len() * 64;
    let first = first.min(cap);
    let last = last.min(cap);
    if first >= last {
        return;
    }
    let (w0, b0) = (first / 64, first % 64);
    let (w1, b1) = ((last - 1) / 64, (last - 1) % 64);
    if w0 == w1 {
        let count = b1 - b0 + 1;
        let mask = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << b0 };
        bits[w0] |= mask;
        return;
    }
    bits[w0] |= u64::MAX << b0;
    for w in &mut bits[w0 + 1..w1] {
        *w = u64::MAX;
    }
    let mask = if b1 == 63 { u64::MAX } else { (1u64 << (b1 + 1)) - 1 };
    bits[w1] |= mask;
}

/// Clears and returns coalesced `CardSpan` runs of all set bits in `bits`
/// intersecting `[span.first, span.first + span.pieces)`.
fn take_stale_span(bits: &mut [u64], span: CardSpan, out: &mut Vec<CardSpan>) {
    let cap = bits.len() * 64;
    let first = (span.first as usize).min(cap);
    let last = (span.first as usize + span.pieces as usize).min(cap);
    if first >= last {
        return;
    }
    let mut run_start: Option<u32> = None;
    let mut run_end: u32 = 0;
    let mut w = first / 64;
    let w_end = (last - 1) / 64;
    while w <= w_end {
        let lo = if w == first / 64 { first % 64 } else { 0 };
        let hi = if w == w_end { (last - 1) % 64 } else { 63 };
        let count = hi - lo + 1;
        let window = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << lo };
        let mut hit = bits[w] & window;
        if hit != 0 {
            bits[w] &= !hit;
            while hit != 0 {
                let bit = hit.trailing_zeros() as usize;
                let piece = (w * 64 + bit) as u32;
                hit &= hit - 1;
                match run_start {
                    Some(_) if piece == run_end => run_end += 1,
                    Some(start) => {
                        out.push(CardSpan {
                            from: span.from,
                            to: span.to,
                            first: start,
                            pieces: run_end - start,
                            skew: span.skew,
                        });
                        run_start = Some(piece);
                        run_end = piece + 1;
                    }
                    None => {
                        run_start = Some(piece);
                        run_end = piece + 1;
                    }
                }
            }
        }
        w += 1;
    }
    if let Some(start) = run_start {
        out.push(CardSpan {
            from: span.from,
            to: span.to,
            first: start,
            pieces: run_end - start,
            skew: span.skew,
        });
    }
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    card_hash_pieces(size) as u64 * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    let there = card_room_for(size);
    if there.is_some() || !card_all() || size >= 1 << 30 {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !cards() || !host.card_ready() {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
    CARD_LAST_ADDED_MS.store(crate::host::coarse_ms().max(1), Ordering::Relaxed);
    Some(buffer)
}

/// Whether every title buffer's copy has two sides, however small
/// (`VICTORY_CARD_ALL=1`; off by default).
///
/// A copy with one side, in the PC's memory, is what the card reads while
/// our refreshes write it: a submission that is prepared changes it under
/// the ones in the air ahead of it. The title's CPU does not write what they
/// still read, but its own buffer copies are done by us at preparing
/// (`crate::host::send` has the order of those right for copies with two
/// sides), and so are the bytes put in beside shaders' writes.
fn card_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ALL").is_some_and(|v| v == "1"))
}

/// [`card_for`] in the card's own memory.
fn card_room_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory() {
        let default_min_bytes = default_card_limits_mb(Some(budget)).0 << 20;
        let headroom = resolve_card_vram_headroom(
            budget,
            size,
            default_min_bytes,
            card_vram_headroom_override_mb(),
        );
        let ceiling = resolve_card_vram_ceiling(budget, hundredths, headroom);
        if used.saturating_add(size) > ceiling {
            // Said, with the numbers it was decided by: run 193 made 31 copies
            // of 95 and no line named the others. The first 64, then every
            // 32nd.
            let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
            let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
            if n <= 64 || n.is_multiple_of(32) {
                tracing::info!(
                    "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                     {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                    size >> 10,
                    used >> 20,
                    budget >> 20,
                    ceiling >> 20,
                    CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                    most >> 20,
                    n,
                    bytes >> 20,
                    crate::frame::current()
                );
            }
            return None;
        }
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed).saturating_add(size) > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: card-buffer cap reached ({} MB of at most {} MB); \
                 {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_some() {
        CARD_LAST_ADDED_MS.store(crate::host::coarse_ms().max(1), Ordering::Relaxed);
    } else {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let there = window_room_for(size);
    if there.is_some() || !windows_own() || !cards() {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed);
    Some(buffer)
}

/// Default demand-tiered BAR hot-window budget in MiB (`640 MiB`, replacing
/// greedy `4096 MiB` BAR flooding while keeping the full ~3 ms/f GPU read win
/// on hot streaming windows; `ARCH-05`). Override with `VICTORY_WINDOWS_BAR_MB`
/// (`0` disables BAR promotion).
const DEFAULT_WINDOW_BAR_MB: u64 = 640;
/// Minimum distinct frames touched before an L0 loose window promotes to BAR.
const BAR_PROMOTE_FRAMES: u64 = 6;
/// Minimum bytes copied before an L0 loose window promotes to BAR (clamped to window size).
const BAR_PROMOTE_MIN_BYTES: u64 = 256 << 10;
/// Cold BAR window idle demotion/eviction threshold (4 seconds).
const BAR_IDLE_MS: u64 = 4_000;
/// Local VRAM ceilings for BAR tier hysteresis: promote below `80%`, demote above `82%`.
const BAR_PROMOTE_VRAM_PCT: u64 = 80;
const BAR_DEMOTE_VRAM_PCT: u64 = 82;

fn resolve_window_bar_mb(env: Option<&str>) -> u64 {
    env.and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_WINDOW_BAR_MB)
}

#[inline]
fn bar_should_promote(touch_frames: u64, copied_bytes: u64, window_len: u64) -> bool {
    touch_frames >= BAR_PROMOTE_FRAMES && copied_bytes >= window_len.min(BAR_PROMOTE_MIN_BYTES)
}

/// The most bytes of hot loose windows promoted into BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`)
/// memory (`VICTORY_WINDOWS_BAR_MB`; default `640` MiB, `0` disables BAR).
/// Fresh loose windows always allocate in L0 first and promote here only once hot
/// (`>= 6` frames touched and `>= 256 KiB` copied), with direct CPU `Map()` intact
/// (never `CreateReservedResource`, Spec-Trap #1).
fn window_mapped_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        let env = std::env::var("VICTORY_WINDOWS_BAR_MB").ok();
        resolve_window_bar_mb(env.as_deref()).saturating_mul(1 << 20)
    })
}

/// Bytes alive of windows in card memory the CPU maps.
static WINDOW_MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn window_mapped_bytes() -> u64 {
    WINDOW_MAPPED_BYTES.load(Ordering::Relaxed)
}

pub(crate) fn card_live_bytes() -> u64 {
    CARD_BYTES[0].load(Ordering::Relaxed)
}

/// Total live L0 (`shared_buffer`) shadow bytes (buffers + loose windows + exact views).
#[allow(dead_code)]
pub(crate) fn l0_shadow_bytes() -> u64 {
    L0_SHADOW_BYTES.load(Ordering::Relaxed)
}

/// Live L0 (`shared_buffer`) bytes held by `window == true` loose windows.
#[allow(dead_code)]
pub(crate) fn l0_loose_bytes() -> u64 {
    L0_LOOSE_BYTES.load(Ordering::Relaxed)
}

/// Committed CPU RAM bytes across all demand-committed [`CpuStaging`] buffers.
#[allow(dead_code)]
pub(crate) fn cpu_staging_committed_bytes() -> u64 {
    CPU_STAGING_COMMITTED_BYTES.load(Ordering::Relaxed)
}

/// `(l0_total_bytes, l0_loose_bytes, bar_window_bytes, cpu_staging_committed_bytes)`
/// accessor for `host.rs` `memory summary:`.
#[allow(dead_code)]
pub(crate) fn memory_tier_info() -> (u64, u64, u64, u64) {
    (
        L0_SHADOW_BYTES.load(Ordering::Relaxed),
        L0_LOOSE_BYTES.load(Ordering::Relaxed),
        WINDOW_MAPPED_BYTES.load(Ordering::Relaxed),
        CPU_STAGING_COMMITTED_BYTES.load(Ordering::Relaxed),
    )
}

/// A buffer for a window of `size` bytes in card memory the CPU maps, if it is
/// to be there ([`window_mapped_most`]).
fn window_mapped_for(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let most = window_mapped_most();
    if most == 0 {
        return None;
    }
    let host = host()?;
    if let Some([budget, used, ..]) = host.video_memory()
        && used.saturating_add(size) > (budget / 100).saturating_mul(BAR_PROMOTE_VRAM_PCT)
    {
        return None;
    }
    if WINDOW_MAPPED_BYTES.fetch_add(size, Ordering::Relaxed).saturating_add(size) > most {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let made = host.mapped_card_buffer(size, true);
    if made.is_none() {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    made
}

/// Whether windows outside buffers have two sides too, so that submissions
/// in the air read the bytes they were prepared with (`VICTORY_WINDOWS_OWN=1`;
/// off by default: it takes the windows' bytes again, 3.3 GB in capture
/// replay2, and what changes in them under submissions in the air is what the
/// title's CPU writes, which it does not while they read it).
fn windows_own() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOWS_OWN").is_some_and(|v| v == "1"))
}

/// [`window_card_for`] in the card's own memory.
fn window_room_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * 88
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// What went up, by send: its number, and the offsets, in order and joined
/// (a refresh sends thousands of runs of a few words).
#[derive(Default)]
struct SentRuns(Vec<(u64, Vec<(usize, usize)>)>);

impl SentRuns {
    #[cfg(test)]
    fn note(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        self.note_joined(number, joined(ranges));
    }

    /// `ranges` are in order and joined.
    fn note_joined(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        if ranges.is_empty() {
            return;
        }
        match self.0.iter_mut().find(|s| s.0 == number) {
            Some((_, all)) => *all = joined([std::mem::take(all), ranges].concat()),
            None => self.0.push((number, ranges)),
        }
    }

    /// Sends up to `number` are of no use any more.
    fn forget(&mut self, number: u64) {
        self.0.retain(|s| s.0 > number);
    }

    /// Adds to `out` what sends after `number` took of `[from, to)`.
    fn near(&self, number: u64, from: usize, to: usize, out: &mut Vec<(usize, usize)>) {
        for (_, ranges) in self.0.iter().filter(|s| s.0 > number) {
            let first = ranges.partition_point(|r| r.1 <= from);
            out.extend(ranges[first..].iter().take_while(|r| r.0 < to).copied());
        }
    }

    #[cfg(test)]
    fn of(all: &[(usize, usize, u64)]) -> SentRuns {
        let mut sent = SentRuns::default();
        for &(a, b, number) in all {
            sent.note(number, vec![(a, b)]);
        }
        sent
    }
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|(_, _, weak)| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .filter(|c| c.touched.swap(false, Ordering::AcqRel))
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = joined(std::mem::take(&mut *dirty));
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.forget(fetched);
                    sent.note_joined(sent_as, ranges.clone());
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            if gpu_only_mode() > 0 {
                shadow.gpu_only_note_upload(&ranges);
            }
            out.push((shadow, ranges));
        }
        true
    });
    out
}

/// Diagnostics (`crate::host::upcheck`, nothing in the air): copies whose
/// pages the write watch and the guards hold for unchanged since they were
/// taken, compared with title memory, `most` bytes a call in turn. `say` gets
/// the copy, the title address of a page that is not the same, the words that
/// differ and the first of them (its address, the copy's value, title
/// memory's). Returns the bytes compared.
pub(crate) fn stale_pages(most: usize, say: &mut dyn FnMut(&Shadow, usize, usize, (usize, u32, u32))) -> usize {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut all: Vec<Arc<Shadow>> = buffers_read().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(|e| e.as_ready().cloned()),
    );
    all.retain(|s| s.place.is_none() && s.watched.load(Ordering::Relaxed) == 1);
    all.sort_by_key(|s| s.id);
    let next = NEXT.load(Ordering::Relaxed);
    let turn = all.partition_point(|s| s.id < next);
    let mut done = 0;
    let mut title = vec![0u8; PAGE];
    for shadow in all[turn..].iter().chain(&all[..turn]) {
        if done >= most {
            NEXT.store(shadow.id, Ordering::Relaxed);
            break;
        }
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        // (What it finds written stays noted for the next refresh.)
        let Some((_, written)) = crate::watch::writes(start, end, true) else {
            continue;
        };
        let written: std::collections::HashMap<usize, u64> = written.into_iter().collect();
        let copied = shadow.copied.lock().unwrap_or_else(|e| e.into_inner()).0.clone();
        for (i, &at) in copied.iter().enumerate() {
            let page = start / PAGE + i;
            if at == 0 || written.get(&page).is_some_and(|&w| w > at) {
                continue;
            }
            let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            let title = &mut title[..hi - lo];
            if !crate::host::read_into(lo as *const u8, title) {
                continue;
            }
            done += hi - lo;
            if !shadow.ensure_staging_committed(lo - start, hi - start) {
                continue;
            }
            // SAFETY: inside the mapping (committed above); nothing is in the air and the
            // submission lock is held.
            let held = unsafe { std::slice::from_raw_parts(shadow.data.add(lo - start).cast_const(), hi - lo) };
            if held == title {
                continue;
            }
            let word = |b: &[u8], k: usize| u32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            let differ: Vec<usize> = (0..(hi - lo) & !3).step_by(4).filter(|&k| word(held, k) != word(title, k)).collect();
            // Written meanwhile: not stale, and the next refresh sees it.
            let again = crate::watch::writes(lo, hi, true).is_none_or(|(_, w)| w.iter().any(|&(p, w)| p == page && w > at));
            if let Some(&k) = differ.first()
                && !again
            {
                say(shadow, lo, differ.len(), (lo + k, word(held, k), word(title, k)));
            }
        }
    }
    done
}

/// Copies of memory (from, to, bytes: apart), on several threads if they are
/// many.
pub(crate) fn copy_all(copies: Vec<(usize, usize, usize)>) {
    copy_ranges_parallel(copies, 128 << 10);
}

/// A title buffer copy was done on the CPU into `[dst, dst + len)`: all of it
/// goes up to the copies in the card's memory that hold it, whether their
/// CPU side held the same before or not. Left out is what shaders in the air
/// may be writing (such a copy waits for them, or is done on the card).
pub(crate) fn send_copied(dst: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (d_start, d_end) = (dst as u64, (dst as u64).saturating_add(len as u64));
    let hi = CARDED_BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 || d_start >= hi || d_end <= CARDED_BOUNDS[0].load(Ordering::Relaxed) {
        return;
    }
    let mut hits: Vec<Arc<Shadow>> = Vec::new();
    {
        let carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
        for &(s_start, s_end, ref weak) in carded.iter() {
            if s_start < d_end && s_end > d_start
                && let Some(shadow) = weak.upgrade()
            {
                hits.push(shadow);
            }
        }
    }
    for shadow in hits {
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        let (from, to) = (dst.max(start), (dst + len).min(end));
        if from >= to {
            continue;
        }
        let busy: Vec<(usize, usize)> = if shadow.pending_count.load(Ordering::Acquire) == 0 {
            Vec::new()
        } else {
            shadow.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().filter(|&(a, b)| a < to && b > from).collect()
        };
        for (a, b) in outside(from, to, busy) {
            shadow.copy(a, b);
        }
    }
}

/// The copies in the card's memory (diagnostics: `crate::host::upcheck`).
pub(crate) fn carded_copies() -> Vec<Arc<Shadow>> {
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter_map(|(_, _, w)| w.upgrade()).collect()
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    // (The sort that finds what is in order already: they come as a few
    // stretches of tens of thousands in order, put behind one another.)
    ranges.sort();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| (r.first, r.from));
    let mut out: Vec<CardSpan> = Vec::new();
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.from = last.from.min(run.from);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// The offsets [`Shadow::card_hashed`] gives, for a card's buffer of `size`
/// bytes for title memory from `start` on.
fn card_hashed_of(start: usize, size: usize, piece: usize) -> (usize, usize) {
    let (lo, hi) = piece_at(start, size, piece);
    (lo & !15, hi.next_multiple_of(16))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_hashed_of, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
        // Same-piece sub-ranges in reverse order keep minimum `from` and maximum `to`.
        let rev = vec![card_span_of(0, 0x2000, 0x3000), card_span_of(0, 0x0400, 0x1000)];
        assert_eq!(
            joined_piece_runs(rev),
            [CardSpan { from: 0x0400, to: 0x3000, first: 0, pieces: 1, skew: 0 }]
        );
    }

    #[test]
    fn gpu_only_tracker_mode1_promotes_after_warmup_and_demotes_on_upload() {
        use super::{GPU_ONLY_MAX_RESETS, GPU_ONLY_MIN_FETCHED_PIECES, GPU_ONLY_WARMUP_FRAMES, GpuOnlyTracker};
        let mut t = GpuOnlyTracker::default();
        // Initial boot upload resets warmup so the 60-frame window begins on the first post-boot fetch.
        assert!(!t.note_upload(1, false, 0, 16));
        assert_eq!(t.note_fetched(1, 2, &(0..8).collect::<Vec<_>>()), (false, false));
        assert_eq!(
            t.note_fetched(1, 2 + GPU_ONLY_WARMUP_FRAMES - 1, &(8..GPU_ONLY_MIN_FETCHED_PIECES as usize).collect::<Vec<_>>()),
            (false, false)
        );
        // Once warmup frames and piece threshold are met with 0 uploads: promote.
        assert_eq!(t.note_fetched(1, 2 + GPU_ONLY_WARMUP_FRAMES, &[0]), (true, false));
        // Any post-promotion CPU upload triggers immediate demotion.
        assert!(t.note_upload(1, true, 5, 6));
        // Repeated unpromoted uploads in Mode 1 restart observation without sticky-banning,
        // allowing promotion once steady-state warmup completes with zero uploads.
        let mut noisy = GpuOnlyTracker::default();
        for _ in 0..=GPU_ONLY_MAX_RESETS {
            assert!(!noisy.note_upload(1, false, 0, 1));
        }
        assert_eq!(noisy.note_fetched(1, 100, &(0..GPU_ONLY_MIN_FETCHED_PIECES as usize).collect::<Vec<_>>()), (false, false));
        assert_eq!(noisy.note_fetched(1, 100 + GPU_ONLY_WARMUP_FRAMES, &[0]), (true, false));
    }

    #[test]
    fn gpu_only_tracker_mode2_allows_disjoint_uploads_and_demotes_on_overlap() {
        use super::{GPU_ONLY_MIN_FETCHED_PIECES, GPU_ONLY_WARMUP_FRAMES, GpuOnlyTracker};
        let mut t = GpuOnlyTracker::default();
        // CPU writes pieces 0..4; GPU writes disjoint pieces 10..(10+N).
        assert!(!t.note_upload(2, false, 0, 4));
        let gpu_pieces: Vec<usize> = (10..10 + GPU_ONLY_MIN_FETCHED_PIECES as usize).collect();
        assert_eq!(t.note_fetched(2, 1, &gpu_pieces), (false, false));
        assert!(!t.note_upload(2, false, 0, 4));
        assert_eq!(t.note_fetched(2, 1 + GPU_ONLY_WARMUP_FRAMES, &[10]), (true, false));
        // Post-promotion upload on disjoint CPU pieces does NOT demote.
        assert!(!t.note_upload(2, true, 0, 4));
        // Post-promotion upload overlapping a GPU-fetched piece demotes immediately.
        assert!(t.note_upload(2, true, 10, 11));
        // Dynamic `gpu_dirty_bits`: post-promotion declared GPU write on a brand-new
        // piece (40..44) catches a later CPU upload overlapping piece 42!
        t.note_declared_write(40, 44);
        assert!(!t.note_upload(2, true, 0, 4));
        assert!(t.note_upload(2, true, 42, 43));
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    /// What is fetched of a piece to hash it is what the shader hashes.
    #[test]
    fn a_piece_is_hashed_here_as_on_the_card() {
        for start in [0usize, 0x10_0000, 0xd_5edb_c780, 0x12_3456_7008] {
            for size in [1usize << 16, 5 << 16, 0x20_0000] {
                let span = card_span_of(start, 0, size);
                for i in 0..span.pieces {
                    // As `CARD_COMPARE` has it.
                    let (piece, skew, size) = (span.first + i, span.skew, size as u32);
                    let lo = (piece * 65536).saturating_sub(skew);
                    let hi = ((piece + 1) * 65536 - skew).min(size);
                    let shader = (lo & !15, (hi + 15) & !15);
                    let here = card_hashed_of(start, size as usize, piece as usize);
                    assert_eq!((here.0 as u32, here.1 as u32), shader, "start {start:#x} size {size:#x} piece {piece}");
                }
            }
        }
    }

    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }

    #[test]
    fn hash_stale_bitset_marks_and_clears_across_word_boundaries() {
        use super::{mark_stale_bits, take_stale_span};
        let mut bits = vec![0u64; 3]; // 192 pieces
        mark_stale_bits(&mut bits, 60, 70);
        mark_stale_bits(&mut bits, 127, 130);
        mark_stale_bits(&mut bits, 180, 181);
        let span = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 0,
        };
        let mut out = Vec::new();
        take_stale_span(&mut bits, span(62, 70), &mut out);
        assert_eq!(
            out.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(),
            [(62, 8), (127, 3)]
        );
        // Taking the same span again finds nothing (bits were cleared).
        out.clear();
        take_stale_span(&mut bits, span(62, 70), &mut out);
        assert!(out.is_empty());
        // Pieces outside the queried span remain stale.
        take_stale_span(&mut bits, span(0, 192), &mut out);
        assert_eq!(
            out.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(),
            [(60, 2), (180, 1)]
        );
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(std::collections::HashMap<(usize, usize), (u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        let f = self.0.entry(range).or_insert((0, number));
        if f.0 == 0 || f.1 != number {
            *f = (f.0 + 1, number);
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds, in order.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.get_mut(range) {
                Some(f) if f.0 > 1 => f.0 -= 1,
                Some(_) => {
                    self.0.remove(range);
                    free.push(*range);
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free.sort_unstable();
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// How many snapshot buffers are kept (`VICTORY_SNAPSHOT_POOL`; `8`: as
/// before).
///
/// A frame gives some 330 ranges of copies in the PC's memory to shaders, and
/// most are of copies of their own; with two or three frames in the air eight
/// buffers were the eight largest, and every other snapshot was memory fresh
/// from the kernel and given back to it: 11 % of the submitting thread in the
/// opening (profile cpu-1: `NtAllocateVirtualMemory` under
/// [`Shadow::writing`]), and the page faults of filling it.
fn snapshots_kept() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_SNAPSHOT_POOL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024))
}

/// Returns a snapshot buffer to the pool: at most [`snapshots_kept`] kept,
/// the largest, and of those the largest one and the next ones up to `KEPT`
/// bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 64 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let (mut kept, mut total) = (0, 0usize);
    for v in pool.iter().take(snapshots_kept()) {
        total = total.saturating_add(v.capacity());
        if kept > 0 && total > KEPT {
            break;
        }
        kept += 1;
    }
    // (Given back to the kernel without the lock: milliseconds.)
    let gone = pool.split_off(kept);
    drop(pool);
    drop(gone);
}

const NOT_WATCHED: u8 = 2;
/// Watched, not by the kernel but by the guards, when last asked about.
const GUARDED: u8 = 3;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Calls `f` with `[from, to)` cut at multiples of [`PIECE`].
fn for_pieces_of(from: usize, to: usize, mut f: impl FnMut(usize, usize)) {
    let mut at = from;
    while at < to {
        let next = ((at / PIECE + 1) * PIECE).min(to);
        f(at, next);
        at = next;
    }
}

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Stack-allocated work descriptor shared with [`WidePool`] workers for the
/// duration of one [`WidePool::each`] call (zero per-call heap allocations and
/// safe for nested/concurrent calls).
struct WorkSlot {
    work: *const (dyn Fn(usize) + Sync),
    count: usize,
    next: AtomicUsize,
    running: AtomicUsize,
    panicked: std::sync::atomic::AtomicBool,
}

// SAFETY: `work` is `Sync`, and `WidePool::each` removes the slot from
// `state.active` and waits until `running == 0` before returning.
unsafe impl Send for WorkSlot {}
unsafe impl Sync for WorkSlot {}

struct WideState {
    active: Vec<*const WorkSlot>,
}

// SAFETY: raw pointers in `active` are only dereferenced while `running` keeps
// their owning stack frame alive.
unsafe impl Send for WideState {}

/// Threads kept for [`wide`] and [`wide_each`]: a stack of active work slots
/// so nested calls (`refresh_spans` -> `take_all`) and fast clean flushes never
/// allocate, deadlock, or block waiting for late-waking workers.
struct WidePool {
    state: Mutex<WideState>,
    more: std::sync::Condvar,
    idle: std::sync::Condvar,
    /// Number of active slots with unclaimed work, for [`spin`].
    there: AtomicUsize,
    threads: usize,
}

/// Waits awake until `there` says so, `VICTORY_WIDE_SPIN` microseconds at
/// most (0, the default: not at all): asleep, a thread is woken in 50 us and
/// more, 95 times a frame.
fn spin(there: impl Fn() -> bool) {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| std::env::var("VICTORY_WIDE_SPIN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    if most == 0 {
        return;
    }
    let began = std::time::Instant::now();
    while !there() && (began.elapsed().as_micros() as u64) < most {
        for _ in 0..16 {
            std::hint::spin_loop();
        }
    }
}

/// How many threads are kept for [`wide`] (`VICTORY_WIDE_THREADS`).
fn wide_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("VICTORY_WIDE_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get() / 2).clamp(2, 6))
            .clamp(1, 14)
    })
}

fn wide_pool() -> &'static WidePool {
    static POOL: std::sync::OnceLock<&'static WidePool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = wide_threads();
        let pool: &'static WidePool = Box::leak(Box::new(WidePool {
            state: Mutex::new(WideState { active: Vec::with_capacity(4) }),
            more: std::sync::Condvar::new(),
            idle: std::sync::Condvar::new(),
            there: AtomicUsize::new(0),
            threads,
        }));
        for i in 0..threads {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
                // SAFETY: raises priority of this helper thread so `victory-submit`
                // (also ABOVE_NORMAL) is not priority-inverted behind `victory-list`.
                unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
                crate::cores::keep_helper_on_fast_cores("victory-wide");
                pool.help();
            });
        }
        pool
    })
}

impl WidePool {
    fn help(&self) {
        loop {
            spin(|| self.there.load(Ordering::Acquire) > 0);
            let slot_ptr = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    // Pick the newest (LIFO) slot with unclaimed work so nested calls finish first.
                    let found = state.active.iter().rev().copied().find(|&ptr| {
                        // SAFETY: slot stays alive while listed in `state.active`.
                        let slot = unsafe { &*ptr };
                        slot.next.load(Ordering::Relaxed) < slot.count
                    });
                    if let Some(ptr) = found {
                        // SAFETY: slot stays alive while listed in `state.active`;
                        // incrementing `running` under `state` pins it after removal.
                        unsafe { (*ptr).running.fetch_add(1, Ordering::AcqRel) };
                        break ptr;
                    }
                    state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            };
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: pinned by `running` increment above.
                let slot = unsafe { &*slot_ptr };
                loop {
                    let i = slot.next.fetch_add(1, Ordering::Relaxed);
                    if i >= slot.count {
                        break;
                    }
                    let _working = crate::timeline::span("a job for another thread");
                    // SAFETY: `work` is valid until `running` drops to 0.
                    unsafe { (*slot.work)(i) };
                }
            }));
            if ran.is_err() {
                // SAFETY: still pinned by `running`.
                unsafe { (*slot_ptr).panicked.store(true, Ordering::Release) };
            }
            // SAFETY: last use of `slot_ptr`; do not dereference after `fetch_sub`.
            let was = unsafe { (*slot_ptr).running.fetch_sub(1, Ordering::AcqRel) };
            if was == 1 {
                let _state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                self.idle.notify_all();
            }
        }
    }

    fn each(&self, n: usize, work: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        if n == 1 {
            work(0);
            return;
        }
        let _wide = crate::timeline::span("work spread over threads");
        // SAFETY: lifetime erased only while `slot` is on this stack frame;
        // `each` removes `slot` from `active` and waits for `running == 0` before returning.
        let work_ptr = unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(work)
        };
        let slot = WorkSlot {
            work: work_ptr,
            count: n,
            next: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            panicked: std::sync::atomic::AtomicBool::new(false),
        };
        let slot_ptr: *const WorkSlot = &slot;
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.active.push(slot_ptr);
            self.there.fetch_add(1, Ordering::AcqRel);
        }
        if n > self.threads {
            self.more.notify_all();
        } else {
            (1..n).for_each(|_| self.more.notify_one());
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let i = slot.next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                work(i);
            }
        }));
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pos) = state.active.iter().rposition(|&p| p == slot_ptr) {
                state.active.remove(pos);
            }
            self.there.fetch_sub(1, Ordering::AcqRel);
            while slot.running.load(Ordering::Acquire) > 0 {
                state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        }
        if let Err(panic) = ran {
            std::panic::resume_unwind(panic);
        }
        assert!(!slot.panicked.load(Ordering::Acquire), "a wide worker job panicked");
    }
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let slots: Vec<Mutex<Option<Job>>> = jobs.into_iter().map(|j| Mutex::new(Some(j))).collect();
    wide_pool().each(slots.len(), &|i| {
        if let Some(job) = slots[i].lock().unwrap_or_else(|e| e.into_inner()).take() {
            job();
        }
    });
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// Total threads participating in [`wide_each`] (helpers plus the caller).
pub(crate) fn wide_worker_count() -> usize {
    wide_threads() + 1
}

/// Calls `work` with every number under `n`, on the kept threads and on
/// this one, the next number to whichever is free: zero per-call heap
/// allocations and immediate return when no helper entered the batch.
pub(crate) fn wide_each(n: usize, work: impl Fn(usize) + Sync) {
    if n == 0 {
        return;
    }
    if n == 1 {
        work(0);
        return;
    }
    wide_pool().each(n, &work);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    copy_ranges_parallel(snaps, 64 << 10);
}

/// Partitions `(src, dst, len)` copies into flat sub-ranges grouped into chunks
/// of about `chunk_bytes` bytes each (pure CPU helper for `copy_ranges_parallel`).
fn split_copy_chunks(
    copies: &[(usize, usize, usize)],
    chunk_bytes: usize,
) -> (Vec<(usize, usize, usize)>, Vec<std::ops::Range<usize>>) {
    let chunk_bytes = chunk_bytes.max(1);
    let mut flat: Vec<(usize, usize, usize)> = Vec::with_capacity(copies.len() + 8);
    let mut chunks: Vec<std::ops::Range<usize>> = Vec::new();
    let mut start = 0usize;
    let mut bytes = 0usize;
    for &(src, dst, len) in copies {
        let mut at = 0usize;
        while at < len {
            let n = (chunk_bytes - bytes).min(len - at);
            let (s, d) = (src + at, dst + at);
            if flat.len() > start
                && let Some(prev) = flat.last_mut()
                && prev.0 + prev.2 == s
                && prev.1 + prev.2 == d
            {
                prev.2 += n;
            } else {
                flat.push((s, d, n));
            }
            at += n;
            bytes += n;
            if bytes >= chunk_bytes {
                chunks.push(start..flat.len());
                start = flat.len();
                bytes = 0;
            }
        }
    }
    if start < flat.len() {
        chunks.push(start..flat.len());
    }
    (flat, chunks)
}

/// Copies disjoint `(src, dst, len)` ranges inline when below `min_parallel_bytes`,
/// otherwise splits adaptively across `wide_threads() + 1` workers via `wide_each`
/// (avoiding per-job `Box`/`Vec` allocations and single-threaded sub-1 MiB batches).
fn copy_ranges_parallel(copies: Vec<(usize, usize, usize)>, min_parallel_bytes: usize) {
    let total: usize = copies.iter().map(|c| c.2).sum();
    if total == 0 {
        return;
    }
    let threads = wide_threads() + 1;
    if total < min_parallel_bytes || threads < 2 {
        for (src, dst, len) in copies {
            if len > 0 {
                // SAFETY: caller guarantees disjoint valid ranges.
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
            }
        }
        return;
    }
    let chunk_bytes = total.div_ceil(threads).clamp(32 << 10, WIDE_JOB);
    let (flat, chunks) = split_copy_chunks(&copies, chunk_bytes);
    if chunks.len() < 2 {
        for (src, dst, len) in flat {
            // SAFETY: caller guarantees disjoint valid ranges.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
        }
        return;
    }
    wide_each(chunks.len(), move |i| {
        for &(src, dst, len) in &flat[chunks[i].clone()] {
            // SAFETY: caller guarantees disjoint valid ranges.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
        }
    });
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
#[cfg_attr(not(test), allow(dead_code))]
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, copy_ranges_parallel, find_pieces, split_copy_chunks, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn adaptive_snapshot_copy_splits_and_copies_exact_bytes() {
        let src: Vec<u8> = (0..(300 << 10)).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        let ranges = vec![
            (src.as_ptr() as usize, dst.as_mut_ptr() as usize, 100 << 10),
            (
                unsafe { src.as_ptr().add(100 << 10) } as usize,
                unsafe { dst.as_mut_ptr().add(100 << 10) } as usize,
                200 << 10,
            ),
        ];
        let (flat, chunks) = split_copy_chunks(&ranges, 48 << 10);
        assert!(chunks.len() >= 6, "expected multi-chunk split for 300 KiB");
        assert_eq!(flat.iter().map(|c| c.2).sum::<usize>(), src.len());

        copy_ranges_parallel(ranges, 64 << 10);
        assert_eq!(dst, src);

        // Small (< min_parallel_bytes) inline path also copies exact bytes.
        let mut small_dst = vec![0u8; 4096];
        copy_ranges_parallel(vec![(src.as_ptr() as usize, small_dst.as_mut_ptr() as usize, 4096)], 64 << 10);
        assert_eq!(small_dst, src[..4096]);
    }

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Why [`Shadow::merge_under_writes`] last left the caller to wait.
static STALE_WHY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn note_stale(why: usize) {
    STALE_WHY.store(why, Ordering::Relaxed);
}

/// [`STALE_WHY`] in words (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn stale_why() -> &'static str {
    const WHY: [&str; 4] = [
        "the memory is not watched",
        "the journal does not say what was written",
        "more changed runs than a submission sends up",
        "no snapshot",
    ];
    WHY[STALE_WHY.load(Ordering::Relaxed).min(3)]
}

/// Since the last frame-rate line, where [`Shadow::merge_under_writes`]'s
/// time went, in ns: 0 asking which pages were written, 1 finding the
/// stale ones, 2 waiting for the locks, 3 the pages compared (the wall
/// time), 4 noting the runs; summed over threads: 5 reading title memory,
/// 6 comparing and putting in. Then 7 (unused), 8 stale pages, 9 calls.
pub(crate) static UNDER_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

fn under_ns(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    UNDER_NS[i].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// [`UNDER_NS`], taken.
#[allow(dead_code)]
pub(crate) fn under_stats() -> [u64; 10] {
    std::array::from_fn(|i| UNDER_NS[i].swap(0, Ordering::Relaxed))
}

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Adds the time since it was made to [`UNDER_NS`]`[6]` when dropped.
struct Compared(std::time::Instant);

impl Drop for Compared {
    fn drop(&mut self) {
        under_ns(6, self.0);
    }
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, many at a time while they last (a page looked at
        // under a flight's writes hardly ever differs at all).
        while i + 512 <= n && now[i..i + 512] == was[i..i + 512] {
            i += 512;
        }
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// The kernel's page.
const KERNEL_PAGE: usize = 4096;

/// Whether a refresh looks if what counts as written is only what a landing
/// put there, and then sends nothing of it up (`VICTORY_LAND_ECHO=0`: it copies
/// and sends it, as before).
///
/// What shaders wrote comes down, goes into title memory and into the CPU's
/// side of the copy alike. In memory the kernel watches that is a write like
/// any other (it cannot be told as the guards' blocks can, `crate::land`), and
/// the next refresh of it copied the pages and sent them up again: in the dorm
/// 14.4 MB a frame of one copy of 90 MB whose 263 pieces were all fetched and
/// all sent, 28 of 72 MB over the bus in all (run submit-2). Only pages a
/// landing was noted in are compared, 4 KiB at a time: a page that is not the
/// same goes up whole, as before, and so does every other.
fn land_echo() -> bool {
    crate::toggled!("VICTORY_LAND_ECHO")
}

/// Since the last frame-rate line: bytes found to be a landing's alone, and
/// bytes a landing was noted in that were the title's too.
pub(crate) static ECHO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a refresh of a copy in the card's memory takes, and sends up,
/// the dwords that title memory changed and no others
/// (`VICTORY_SEND_CHANGED=1`; off: the 64 KiB pages written since it last
/// took them, whole). Off since run tile1-0: the room lost its indirect light
/// and geometry exploded; run bis1-0, the same but for this, was right.
///
/// A page that counts as written is mostly not the title's doing, or not all
/// of it: a word the title wrote, and around it what shaders wrote and the
/// title never touched; or only what the relay put into title memory, which
/// the card and the CPU's side hold already ([`Shadow::card_merge_relay`]).
/// Sent up whole, such a page goes over whatever the card holds there that
/// did not come back yet, or never will (a write through a view nobody
/// declared): what stood on the card, newer, was gone.
///
/// What it rests on: the CPU's side is what the card's side was last known
/// to hold, for every range no shader may be writing (no refresh touches
/// the others). So a dword goes up only if title memory holds another value
/// of it than the CPU's side, which then holds it too; title memory is read
/// once, what is compared is what is stored and sent. Dwords, not bytes:
/// no shader writes less, and the title's bytes beside a changed one in its
/// dword are those the CPU's side holds.
///
/// A page never taken goes up whole as long as no shader's write was ever
/// declared for the copy. After that it is compared as any other: the CPU's
/// side and the card's began as zeros, both.
///
/// What it cannot know: that the title wrote a value over the same value.
/// Where a shader wrote in between, through a view nobody declared, the
/// shader's stays.
fn send_changed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() != 0 {
            return false;
        }
        std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0")
    })
}

/// Since the last frame-rate line, of pages taken into copies in the
/// card's memory ([`send_changed`]): bytes compared, those of pages that
/// held the same, bytes to go up, and the runs they are.
static SEND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`SEND`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn send_stats() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| SEND[i].swap(0, Ordering::Relaxed))
}

/// Whether what shaders wrote goes into title memory run by run
/// ([`put_changed`]; on by default, `VICTORY_EXACT_BACK=0` writes the whole
/// stretch from the first changed byte to the last, as before).
pub(crate) fn exact_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXACT_BACK").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: pieces put back by [`put_changed`], the
/// writes they took, the bytes written, and the pieces whose runs were
/// joined over bytes that had not changed.
pub(crate) static PUT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The dwords (of title memory from `at` on) where `merged` differs from
/// `title`, as (offset, bytes) in order, joined where at most `gap` bytes
/// lie between.
fn dword_runs(at: usize, title: &[u8], merged: &[u8], gap: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (from, n) in changed_runs(merged, title) {
        // Shaders write dwords: the whole of each.
        let lo = ((at + from) & !3).max(at) - at;
        let hi = ((at + from + n + 3) & !3).min(at + title.len()) - at;
        match runs.last_mut() {
            Some((a, len)) if lo <= *a + *len + gap => *len = (*len).max(hi - *a),
            _ => runs.push((lo, hi - lo)),
        }
    }
    runs
}

/// [`dword_runs`] with no gap, the fast way where it can be: a frame compares
/// 12 MB this way and finds 80,000 runs.
fn word_runs(at: usize, title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if at % 4 == 0 && title.len() % 4 == 0 && title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_avx2(title, held) };
    }
    dword_runs(at, title, held, 0)
}

/// # Safety
/// AVX2, and two slices of one length, whole words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_avx2(title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if *a + *n == from => *n += len,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let mut i = 0;
    while i + 256 <= n {
        let mut differ = 0u64;
        for k in 0..8 {
            // SAFETY: inside both slices (`i + 256 <= n`).
            let same = unsafe {
                _mm256_cmpeq_epi32(
                    _mm256_loadu_si256(title.as_ptr().add(i + 32 * k).cast()),
                    _mm256_loadu_si256(held.as_ptr().add(i + 32 * k).cast()),
                )
            };
            differ |= u64::from(!_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u8) << (8 * k);
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 4 <= n {
        if title[i..i + 4] != held[i..i + 4] {
            push(i, 4);
        }
        i += 4;
    }
    runs
}

/// Whether [`Shadow::merge_under_writes`] puts the title's bytes into the
/// CPU's side of a copy in the card's memory as it compares, a block of 32
/// bytes where one of its words differs (`VICTORY_UNDER_FUSED=0`: the runs
/// are found first and copied one by one afterwards, as before). The same
/// bytes end up there: the words of a block that do not differ are equal.
fn under_fused() -> bool {
    crate::toggled!("VICTORY_UNDER_FUSED")
}

/// Maximum unchanged byte gap bridged between adjacent changed 4-byte words
/// on the same dirty page in [`word_runs_put`] (`before` already matches guest
/// memory in the gap, so bridging is 100% bit-identical while collapsing micro-runs).
const WORD_RUN_COALESCE_GAP: usize = 32;

/// [`word_runs`], coalescing gaps `<= 32` bytes between changed 4-byte words on
/// the same page, and updating every 32-byte block of `held` where a word differs:
/// `held` is `title` afterwards.
fn word_runs_put(at: usize, title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if at % 4 == 0 && title.len() % 4 == 0 && title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_put_avx2(title, held) };
    }
    let runs = dword_runs(at, title, held, WORD_RUN_COALESCE_GAP);
    for &(r, m) in &runs {
        held[r..r + m].copy_from_slice(&title[r..r + m]);
    }
    runs
}

/// # Safety
/// AVX2, and two slices of one length, whole words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_put_avx2(title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if from <= *a + *n + WORD_RUN_COALESCE_GAP => *n = (from + len) - *a,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let mut i = 0;
    while i + 256 <= n {
        let mut differ = 0u64;
        for k in 0..8 {
            let at = i + 32 * k;
            // SAFETY: inside both slices (`i + 256 <= n`).
            unsafe {
                let now = _mm256_loadu_si256(title.as_ptr().add(at).cast());
                let old = _mm256_loadu_si256(held.as_ptr().add(at).cast());
                let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
                if block != 0 {
                    _mm256_storeu_si256(held.as_mut_ptr().add(at).cast(), now);
                }
                differ |= u64::from(block) << (8 * k);
            }
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 4 <= n {
        if title[i..i + 4] != held[i..i + 4] {
            held[i..i + 4].copy_from_slice(&title[i..i + 4]);
            push(i, 4);
        }
        i += 4;
    }
    runs
}

/// Whether the pages of [`Shadow::merge_under_writes`] are compared on several
/// threads (`VICTORY_UNDER_WIDE=0`: on the submitting one).
fn under_wide() -> bool {
    crate::toggled!("VICTORY_UNDER_WIDE")
}

/// What shaders wrote goes into title memory from `at` on: `title` is what
/// was read there, `merged` the same with their bytes put in. Only the
/// dwords that differ are written. The whole stretch from the first changed
/// byte to the last was: between the read and the write the title's own
/// threads, or data streamed in, may have written the bytes between (a heap
/// holds what shaders write beside what the CPU does), and those writes were
/// undone. Beside running title threads (the relay) that is a piece of
/// memory going back in time now and then: runs pool1, blame1 and seriess2
/// ended in the title decoding garbage.
///
/// shortcut: a piece with very many runs has them joined over small gaps,
/// where the old race is left.
fn put_changed(at: usize, title: &[u8], merged: &[u8]) -> usize {
    crate::watch::landing(at, title.len());
    if !exact_back() {
        crate::host::write_mapped(at as *mut u8, merged.as_ptr(), merged.len());
        return merged.len();
    }
    if crate::safecopy::on() {
        PUT[0].fetch_add(1, Ordering::Relaxed);
        PUT[2].fetch_add(merged.len() as u64, Ordering::Relaxed);
        let since = crate::watch::changes();
        // SAFETY: title memory; a page of it that is not there fails the
        // put, and the rest is put page by page.
        if unsafe { crate::safecopy::put_changed(at as *mut u8, title, merged) } {
            return merged.len();
        }
        let mut from = 0;
        while from < merged.len() {
            let n = (4096 - (at + from) % 4096).min(merged.len() - from);
            let (was, now) = (&title[from..from + n], &merged[from..from + n]);
            // SAFETY: as above.
            let put = || unsafe { crate::safecopy::put_changed((at + from) as *mut u8, was, now) };
            // A page that is away only while the runtime changes a mapping
            // beside it gets its bytes when that is done.
            if !put() && !crate::watch::again_after_change(since, put) {
                crate::watch::lost("what shaders wrote", at + from, n);
            }
            from += n;
        }
        return merged.len();
    }
    let mut runs = dword_runs(at, title, merged, 0);
    if runs.len() > 64 {
        runs = dword_runs(at, title, merged, 32);
        PUT[3].fetch_add(1, Ordering::Relaxed);
    }
    let mut written = 0;
    for &(from, n) in &runs {
        crate::host::write_mapped((at + from) as *mut u8, merged[from..].as_ptr(), n);
        written += n;
    }
    PUT[0].fetch_add(1, Ordering::Relaxed);
    PUT[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
    PUT[2].fetch_add(written as u64, Ordering::Relaxed);
    written
}

/// For `crate::land`'s measurement: a piece as [`Shadow::card_merge_relay`]
/// lands it without `VICTORY_LAND`: `n` bytes at `now`, `held` what the CPU's
/// side held, both the caller's alone.
#[cfg(test)]
pub(crate) fn land_as_before(start: usize, now: usize, held: usize, n: usize, title: &mut Vec<u8>) {
    // SAFETY: the caller's.
    let (gpu, held) = unsafe { (std::slice::from_raw_parts(now as *const u8, n), std::slice::from_raw_parts_mut(held as *mut u8, n)) };
    let Some(first) = first_difference(gpu, held) else {
        return;
    };
    let last = last_difference(gpu, held).unwrap_or(first);
    let n = last + 1 - first;
    title.clear();
    title.resize(n, 0);
    assert!(crate::host::read_into((start + first) as *const u8, title));
    let was = title.clone();
    if merge_piece(&gpu[first..=last], &mut held[first..=last], title) {
        put_changed(start + first, &was, title);
    }
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs, dword_runs, put_changed, word_runs};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }


    /// What shaders wrote reaches a page that is away only while the runtime
    /// changes a mapping; a page that stays away loses it, and nothing else.
    #[test]
    fn a_put_waits_for_a_change_of_mapping() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a reservation of ours, nothing of it committed.
        let at = unsafe { VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(at, 0);
        put_changed(at, &[0; 64], &[3; 64]);
        let (began, wait) = std::sync::mpsc::channel();
        let change = std::thread::spawn(move || {
            let _changing = crate::watch::title_memory_remapping();
            began.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            // SAFETY: the reservation's first page.
            assert!(!unsafe { VirtualAlloc(at as _, 4096, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        });
        wait.recv().unwrap();
        put_changed(at, &[0; 64], &[1; 64]);
        change.join().unwrap();
        // SAFETY: committed by now.
        assert_eq!(unsafe { std::slice::from_raw_parts(at as *const u8, 64) }, [1; 64]);
        // SAFETY: our reservation.
        unsafe { VirtualFree(at as _, 0, MEM_RELEASE) };
    }

    /// Whole dwords around every changed byte and no other, whatever the
    /// alignment; small gaps joined only when asked.
    #[test]
    fn dword_runs_hold_the_dwords_that_changed() {
        let title: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
        for at in [0x1000usize, 0x1001, 0x1002, 0x1003] {
            for changes in [vec![0], vec![199], vec![5, 6], vec![8, 12, 40, 41, 43, 120]] {
                let mut merged = title.clone();
                for &c in &changes {
                    merged[c] ^= 0x55;
                }
                let runs = dword_runs(at, &title, &merged, 0);
                let held: Vec<usize> = runs.iter().flat_map(|&(a, n)| a..a + n).collect();
                let mut expect: Vec<usize> = (0..200).filter(|i| changes.iter().any(|c| (at + c) / 4 == (at + i) / 4)).collect();
                expect.dedup();
                assert_eq!(held, expect, "{at:#x} {changes:?}");
                let joined = dword_runs(at, &title, &merged, 32);
                assert!(joined.len() <= runs.len());
                assert!(runs.iter().all(|r| joined.iter().any(|j| j.0 <= r.0 && r.0 + r.1 <= j.0 + j.1)));
            }
        }
        assert!(dword_runs(0, &title, &title, 0).is_empty());
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    /// What looking at a page that did not change costs, against the look
    /// eight bytes at a time that it was (a frame has some 250 of them, on
    /// the submitting thread). Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn runs_of_equal_pages() {
        let (a, b) = (vec![7u8; 64 << 10], vec![7u8; 64 << 10]);
        let eights = || {
            let (now, was) = (std::hint::black_box(&a), std::hint::black_box(&b));
            let mut i = 0;
            while i + 8 <= now.len() && now[i..i + 8] == was[i..i + 8] {
                i += 8;
            }
            std::hint::black_box(i);
        };
        let best = |f: &dyn Fn()| {
            (0..500)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::MAX, f64::min)
        };
        let runs = || drop(std::hint::black_box(changed_runs(std::hint::black_box(&a), std::hint::black_box(&b))));
        println!("64 KiB that are the same: {:.1} us by eights, {:.1} us now", best(&eights), best(&runs));
    }

    #[test]
    fn word_runs_put_finds_the_word_runs_and_puts_the_title_in() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..300 {
            let n = [super::PAGE, 4096, 256, 260, 4, 0, 1000][round % 7];
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            // Sparse words, dense stretches, single bytes of words.
            let changes = (next() % 600) as usize;
            for _ in 0..changes.min(n) {
                let at = (next() as usize) % n.max(1);
                let len = [1, 3, 4, 40, 300][(next() % 5) as usize].min(n - at);
                for b in &mut title[at..at + len] {
                    *b = b.wrapping_add(1 + (next() % 255) as u8);
                }
            }
            let want = dword_runs(0, &title, &held, super::WORD_RUN_COALESCE_GAP);
            let mut merged = held.clone();
            let runs = super::word_runs_put(0, &title, &mut merged);
            assert_eq!(runs, want, "round {round}");
            assert_eq!(merged, title, "round {round}");
            // Verify every run's bytes match `title` and every byte outside runs was already equal.
            let mut reconstructed = held.clone();
            for &(r, m) in &runs {
                reconstructed[r..r + m].copy_from_slice(&merged[r..r + m]);
            }
            assert_eq!(reconstructed, title, "round {round} bit-identical upload");
            // At most this many runs of more than 256 bytes (the page's
            // copies of their own, `merge_page_under_writes`).
            assert!(runs.iter().filter(|r| r.1 > 256).count() <= n / 260 + 1, "round {round}");
        }
    }

    #[test]
    fn word_runs_put_coalesces_gaps_up_to_32_bytes_bit_identically() {
        let base: Vec<u8> = (0..1024usize).map(|i| (i.wrapping_mul(29).wrapping_add(3)) as u8).collect();
        let mut title = base.clone();
        // Mutate words at 0, 12 (gap=8 <= 32), 48 (gap=32 <= 32) -> all coalesce into (0, 52).
        // Mutate word at 88 (gap=36 > 32) -> separate run (88, 4).
        // Mutate word at 252 and 260 (crosses 256-byte AVX2 chunk boundary, gap=4 <= 32) -> coalesces into (252, 12).
        for off in [0usize, 12, 48, 88, 252, 260] {
            title[off..off + 4].copy_from_slice(&[0xDE, 0xAD, 0xBE, (off & 0xff) as u8]);
        }
        let mut held = base.clone();
        let runs = super::word_runs_put(0, &title, &mut held);
        assert_eq!(runs, vec![(0, 52), (88, 4), (252, 12)]);
        assert_eq!(held, title);
        let mut uploaded = base;
        for &(r, m) in &runs {
            uploaded[r..r + m].copy_from_slice(&held[r..r + m]);
        }
        assert_eq!(uploaded, title, "coalesced runs produce bit-identical bytes");
    }

    #[test]
    fn word_runs_are_the_dword_runs() {
        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for round in 0..200 {
            let n = 4 * (random() as usize % 400);
            let was: Vec<u8> = (0..n).map(|_| random() as u8).collect();
            let mut now = was.clone();
            for _ in 0..random() % 200 {
                if n == 0 {
                    break;
                }
                let at = random() as usize % n;
                let len = (random() as usize % 24).min(n - at);
                for b in &mut now[at..at + len] {
                    *b = random() as u8;
                }
            }
            assert_eq!(word_runs(0x1000, &now, &was), dword_runs(0x1000, &now, &was, 0), "round {round}");
        }
    }

    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(place) = &self.place {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len, place.pinned);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if self.l0_bytes > 0 {
            L0_SHADOW_BYTES.fetch_sub(self.l0_bytes, Ordering::Relaxed);
            if window {
                L0_LOOSE_BYTES.fetch_sub(self.l0_bytes, Ordering::Relaxed);
            }
        }
        if self.mapped {
            WINDOW_MAPPED_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        if self.gpu_only_state.load(Ordering::Relaxed) == GPU_ONLY_PROMOTED
            || self.gpu_only_promoted_pieces.load(Ordering::Relaxed) > 0
            || self.tripwire_armed.load(Ordering::Relaxed)
        {
            self.disarm_all_tripwires();
        }
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get().is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// Minimum buffer size (bytes) eligible for `VICTORY_KEEP_SNAPSHOTS=1`
/// (`VICTORY_KEEP_SNAPSHOTS_MIN_KB`, default `0` so `< 1 MiB` CPU-compared UAV
/// buffers like the 160 KiB buffers are kept within `VICTORY_KEEP_SNAPSHOTS_MB`).
fn kept_min_bytes() -> u64 {
    static MIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MIN_KB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
            << 10
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in_with_taken(
    base: usize,
    copied: &[u64],
    a: usize,
    e: usize,
    written: &[(usize, u64)],
) -> Vec<(usize, u64)> {
    let first = (a / PAGE).max(base);
    let last = e.div_ceil(PAGE).min(base + copied.len());
    if first >= last {
        return Vec::new();
    }
    let mut wi = written.partition_point(|&(p, _)| p < first);
    let mut out = Vec::new();
    for page in first..last {
        let w = if wi < written.len() && written[wi].0 == page {
            let w = written[wi].1;
            wi += 1;
            w
        } else {
            0
        };
        let at = copied[page - base];
        if at == 0 || w > at {
            out.push((page, at));
        }
    }
    out
}

fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    stale_pages_in_with_taken(base, copied, a, e, written)
        .into_iter()
        .map(|(page, _)| page)
        .collect()
}

/// `runs` (from, to, whether the copy held it before), cut in those of pages
/// held before (`pages`: page number and asking, in order) to the parts of
/// each page that were written since (`parts`: a bit for each 4 KiB).
fn by_parts(runs: Vec<(usize, usize, bool)>, pages: &[(usize, u64)], parts: &[u16]) -> Vec<(usize, usize, bool)> {
    const PART: usize = PAGE / 16;
    if parts.iter().all(|&p| p == u16::MAX) {
        return runs;
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + pages.len());
    let mut pages = pages.iter().zip(parts).peekable();
    for (from, to, known) in runs {
        if !known {
            out.push((from, to, known));
            continue;
        }
        while let Some((&(page, _), &parts)) = pages.next_if(|((page, _), _)| page * PAGE < to) {
            let mut mask = parts;
            while mask != 0 {
                let first = mask.trailing_zeros() as usize;
                let past = first + (!(mask >> first)).trailing_zeros() as usize;
                mask &= if past >= 16 { 0 } else { u16::MAX << past };
                let (a, b) = ((page * PAGE + first * PART).max(from), (page * PAGE + past.min(16) * PART).min(to));
                if a >= b {
                    continue;
                }
                match out.last_mut() {
                    Some((_, end, true)) if *end == a => *end = b,
                    _ => out.push((a, b, true)),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod landing_tests {
    use super::{KERNEL_PAGE, Shadow};

    /// A page that only a landing wrote is not sent up again; one the title
    /// wrote too is, and so is every page no landing was noted in.
    #[test]
    fn what_a_landing_wrote_is_not_sent_up_again() {
        const PAGES: usize = 6;
        let room = vec![7u8; (PAGES + 1) * KERNEL_PAGE];
        let title = room.as_ptr() as usize;
        let title = title.next_multiple_of(KERNEL_PAGE);
        let mut side = vec![7u8; PAGES * KERNEL_PAGE];
        let copy = Shadow::unbacked(title as u64, side.as_mut_ptr(), side.len() as u64, true);
        let write = |page: usize| {
            // SAFETY: in `room`.
            unsafe { ((title + page * KERNEL_PAGE + 9) as *mut u8).write_volatile(8) }
        };
        let page = |from: usize, to: usize| (title + from * KERNEL_PAGE, title + to * KERNEL_PAGE);
        copy.note_landing(KERNEL_PAGE, 4 * KERNEL_PAGE);
        // Page 2: the title's too.
        write(2);
        let (from, to) = page(0, 6);
        copy.take_landed(from, to);
        let mut dirty = copy.take_dirty();
        dirty.sort_unstable();
        let part = |a: usize, b: usize| (a * KERNEL_PAGE, b * KERNEL_PAGE);
        assert_eq!(super::joined(dirty), [part(0, 1), part(2, 3), part(4, 6)]);
        assert_eq!(side[2 * KERNEL_PAGE + 9], 8);
        // Looked at: from here on as any other.
        let (from, to) = page(1, 2);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), [part(1, 2)]);
        // In part: looked at again when the rest is.
        copy.note_landing(3 * KERNEL_PAGE, 3 * KERNEL_PAGE + 1);
        let (from, to) = page(3, 4);
        copy.take_landed(from, from + 100);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), []);
        drop(room);
    }
}

#[cfg(test)]
mod parts_tests {
    use super::{PAGE, by_parts};

    #[test]
    fn runs_are_cut_to_the_parts_written() {
        const P: usize = PAGE / 16;
        let at = |page: usize, part: usize| page * PAGE + part * P;
        // Pages 10 and 11 again (from the middle of 10's part 1 on), 12
        // never taken, 14 again.
        let runs = vec![
            (at(10, 1) + 100, at(12, 0), true),
            (at(12, 0), at(13, 0), false),
            (at(14, 0), at(14, 9) + 7, true),
        ];
        let pages = [(10, 5), (11, 5), (14, 6)];
        assert_eq!(by_parts(runs.clone(), &pages, &[u16::MAX; 3]), runs);
        assert_eq!(
            by_parts(runs, &pages, &[0b1000_0000_0000_0011, 0b11, 0b10_0000_0000]),
            [
                (at(10, 1) + 100, at(10, 2), true),
                (at(10, 15), at(11, 2), true),
                (at(12, 0), at(13, 0), false),
                (at(14, 9), at(14, 9) + 7, true),
            ]
        );
    }
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .filter_map(LooseEntry::as_ready)
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.values().filter_map(LooseEntry::as_ready) {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.0.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Resolves `VICTORY_WINDOW_BUDGET_MB`: explicit `0` disables the budget cap,
/// a positive integer overrides it, and unset defaults to 512 MiB (`ARCH-05`).
fn resolve_window_budget_mb(env: Option<&str>, limited_ram: bool) -> Option<u64> {
    if let Some(raw) = env {
        if let Ok(v) = raw.trim().parse::<u64>() {
            return (v > 0).then_some(v);
        }
    }
    let _ = limited_ram;
    Some(512)
}

/// Optional byte cap for loose title-window copies, in bytes. `VICTORY_WINDOW_BUDGET_MB=0`
/// disables the budget cap; positive values (default 1280 MiB) cap loose copies
/// in the Watch and High RAM bands while leaving Calm unchanged.
fn window_budget() -> Option<u64> {
    static MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        let env = std::env::var("VICTORY_WINDOW_BUDGET_MB").ok();
        resolve_window_budget_mb(env.as_deref(), crate::residency::limited_host_ram())
            .map(|v| v.saturating_mul(1 << 20))
    })
}

/// Oldest eligible copies to remove until their unique bytes fit `keep`.
/// Entries are `(last-used moment, bytes, unique copy identity)`, already
/// sorted oldest first; one copy standing under several keys counts once.
fn window_victims(entries: &[(u64, u64, usize)], keep: u64) -> Vec<usize> {
    let mut left = 0u64;
    let mut seen = std::collections::HashSet::new();
    for &(_, bytes, id) in entries {
        if seen.insert(id) {
            left = left.saturating_add(bytes);
        }
    }
    let mut out = Vec::new();
    for (i, &(_, bytes, id)) in entries.iter().enumerate() {
        if left <= keep {
            break;
        }
        if seen.remove(&id) {
            left = left.saturating_sub(bytes);
            out.push(i);
        }
    }
    out
}

/// Kept between sweeps so one sample near a watermark does not toggle the
/// policy on the next submission.
static WINDOW_BAND: std::sync::Mutex<crate::host::RamBand> =
    std::sync::Mutex::new(crate::host::RamBand::Calm);

/// Lets go of idle loose windows, unpins idle `HostHeap::kept` title slots
/// (Null-Before-Retire), demotes cold/VRAM-pressured BAR windows back to L0,
/// and promotes hot L0 windows (`>= 6` frames touched and `>= 256 KiB` copied)
/// into the bounded BAR tier (`ARCH-05`).
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    const MAX_PROMOTE_PER_SWEEP: usize = 4;
    const MAX_DEMOTE_PER_SWEEP: usize = 8;

    let budget = window_budget();
    let mut idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    let mut keep = u64::MAX;
    if let Some(bytes) = budget {
        if let Some((working, private, free, commit, total, limit)) = crate::host::ram_sample() {
            let mut band = WINDOW_BAND.lock().unwrap_or_else(|e| e.into_inner());
            *band = crate::host::ram_band(*band, working, private, free, commit, total, limit);
            match *band {
                crate::host::RamBand::Calm => {
                    idle_ms = idle_ms.min(4_000);
                    keep = bytes;
                }
                crate::host::RamBand::Watch => {
                    idle_ms = idle_ms.min(2_000);
                    keep = (bytes * 3) / 4;
                }
                crate::host::RamBand::High => {
                    idle_ms = idle_ms.min(1_000);
                    keep = bytes / 2;
                }
            }
        } else {
            idle_ms = idle_ms.min(4_000);
            keep = bytes;
        }
    }
    let l0_over_budget = keep != u64::MAX && L0_LOOSE_BYTES.load(Ordering::Relaxed) > keep;
    let l0_idle_ms = if l0_over_budget { idle_ms.min(500) } else { idle_ms };
    let bar_idle_ms = idle_ms.min(BAR_IDLE_MS);
    let vram_info = host().and_then(|h| h.video_memory());
    let vram_pct = vram_info.map_or(0, |[vbudget, vused, ..]| {
        if vbudget == 0 {
            0
        } else {
            vused.saturating_mul(100) / vbudget
        }
    });
    let vram_over_82 = vram_pct > BAR_DEMOTE_VRAM_PCT;

    // When `l0_loose` exceeds its budget, safely revalidate title-heap slots pinning
    // idle (`>= 300` frames) loose windows outside `LOOSE.write()` (Spec-Trap #16 & #6).
    if l0_over_budget {
        let now_frame = crate::host::moment() >> 32;
        let stale_candidates: Vec<(usize, u64, u64, u64)> = {
            let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
            map.values()
                .filter_map(LooseEntry::as_ready)
                .filter_map(|s| {
                    if s.mapped || s.place.is_some() || Arc::strong_count(s) != 1 {
                        return None;
                    }
                    let last_used_frame = s.used.load(Ordering::Relaxed) >> 32;
                    if now_frame.saturating_sub(last_used_frame) >= 300 && crate::host::references(&s.buffer) > 1 {
                        Some((windows::core::Interface::as_raw(&s.buffer) as usize, s.start, s.len, last_used_frame))
                    } else {
                        None
                    }
                })
                .collect()
        };
        if !stale_candidates.is_empty() {
            let _ = crate::descriptor::host::unpin_stale_window_slots(&stale_candidates, now_frame);
        }
    }

    let mut gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let mut eligible: Vec<(u64, (u64, u64), usize, u64, bool)> = Vec::new();
        let mut active_seen = std::collections::HashSet::new();
        let mut active_bytes = 0u64;
        let mut bar_candidates: Vec<(u64, (u64, u64), Arc<Shadow>)> = Vec::new();
        for (&key, entry) in map.iter() {
            let Some(s) = entry.as_ready() else {
                continue;
            };
            let id = Arc::as_ptr(s) as usize;
            let used = s.used.load(Ordering::Relaxed);
            let eff_idle = if s.mapped { bar_idle_ms } else { l0_idle_ms };
            let strong = Arc::strong_count(s);
            let aged = crate::host::long_ago(used, eff_idle, IDLE_FRAMES);
            // Only evict loose windows held by NO list (`strong == 1`) AND NO live
            // descriptor slot (`refs == 1`). Never speculatively null live mapped
            // bindless title-heap slots (`loose_unmapped` handles freed VAs).
            if strong == 1 && aged && crate::host::references(&s.buffer) == 1 {
                eligible.push((used, key, id, s.len, s.mapped));
                continue;
            }
            if s.mapped && vram_over_82 && strong == 1 && crate::host::references(&s.buffer) == 1 {
                bar_candidates.push((used, key, s.clone()));
            }
            if active_seen.insert(id) {
                active_bytes = active_bytes.saturating_add(s.len);
            }
        }
        eligible.sort_unstable_by_key(|&(used, _, _, _, _)| used);
        let mut gone: Vec<Arc<Shadow>> = Vec::new();
        if keep == u64::MAX {
            for (_, key, _, _, _) in &eligible {
                if let Some(s) = map.remove(key).and_then(LooseEntry::into_ready) {
                    gone.push(s);
                }
            }
        } else {
            // Cold unreferenced BAR windows idle > 4s are evicted immediately.
            let mut l0_eligible = Vec::new();
            for entry in eligible {
                if entry.4 {
                    if let Some(s) = map.remove(&entry.1).and_then(LooseEntry::into_ready) {
                        gone.push(s);
                    }
                } else {
                    l0_eligible.push(entry);
                }
            }
            let entries: Vec<(u64, u64, usize)> = l0_eligible
                .iter()
                .map(|&(used, _, id, len, _)| (used, len, id))
                .collect();
            let victims = window_victims(&entries, keep.saturating_sub(active_bytes));
            for i in victims {
                if let Some(s) = map.remove(&l0_eligible[i].1).and_then(LooseEntry::into_ready) {
                    gone.push(s);
                }
            }
        }
        // Under VRAM > 82%, evict coldest unreferenced BAR windows cleanly.
        if vram_over_82 && !bar_candidates.is_empty() {
            bar_candidates.sort_unstable_by_key(|&(used, _, _)| used);
            for (_, key, s) in bar_candidates.into_iter().take(MAX_DEMOTE_PER_SWEEP) {
                if Arc::strong_count(&s) == 2 && crate::host::references(&s.buffer) == 1 {
                    drop(s);
                    if let Some(removed) = map.remove(&key).and_then(LooseEntry::into_ready) {
                        gone.push(removed);
                    }
                }
            }
        }
        if !gone.is_empty() {
            for shadow in &gone {
                shadow.dead.store(true, Ordering::Release);
            }
            bump_shadow_epoch();
        }
        gone
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone.drain(..) {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        if shadow.mapped {
            let vram = shadow.len;
            crate::host::retire_vram(shadow, vram);
        } else {
            crate::host::retire(shadow);
        }
    }
    // Promote hot unreferenced L0 windows (`refs == 1`, zero live bindless slots)
    // into BAR while VRAM < 80% and BAR budget has headroom; never null live slots.
    let bar_cap = window_mapped_most();
    if bar_cap > 0
        && !vram_over_82
        && vram_pct < BAR_PROMOTE_VRAM_PCT
        && WINDOW_MAPPED_BYTES.load(Ordering::Relaxed) < bar_cap
    {
        let mut promote: Vec<(u64, (u64, u64), Arc<Shadow>)> = {
            let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
            map.iter()
                .filter_map(|(&key, entry)| {
                    let s = entry.as_ready()?;
                    if !s.window || s.mapped || s.card.is_some() || s.place.is_some() {
                        return None;
                    }
                    if crate::host::references(&s.buffer) != 1 {
                        return None;
                    }
                    let touches = s.touch_frames.load(Ordering::Relaxed);
                    let copied = s.copied_bytes.load(Ordering::Relaxed);
                    if !bar_should_promote(touches, copied, s.len) {
                        return None;
                    }
                    if crate::host::long_ago(s.used.load(Ordering::Relaxed), 2_000, IDLE_FRAMES) {
                        return None;
                    }
                    Some((copied, key, s.clone()))
                })
                .collect()
        };
        if !promote.is_empty() {
            promote.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (_, key, old) in promote.into_iter().take(MAX_PROMOTE_PER_SWEEP) {
                if crate::host::references(&old.buffer) != 1 {
                    continue;
                }
                let (start, len) = key;
                let Some((buffer, data)) = window_mapped_for(len) else {
                    break;
                };
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} mapped on the card"));
                // SAFETY: live D3D12 BAR buffer just created.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let mut promoted = Shadow::new(buffer, data, gpu, start, len);
                promoted.window = true;
                promoted.mapped = true;
                promoted.used.store(old.used.load(Ordering::Relaxed), Ordering::Relaxed);
                promoted.touch_frame.store(old.touch_frame.load(Ordering::Relaxed), Ordering::Relaxed);
                promoted.touch_frames.store(old.touch_frames.load(Ordering::Relaxed), Ordering::Relaxed);
                promoted.copied_bytes.store(old.copied_bytes.load(Ordering::Relaxed), Ordering::Relaxed);
                let promoted = Arc::new(promoted);
                let swapped = {
                    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
                    if map
                        .get(&key)
                        .and_then(LooseEntry::as_ready)
                        .is_some_and(|cur| Arc::ptr_eq(cur, &old))
                        && crate::host::references(&old.buffer) == 1
                    {
                        map.insert(key, LooseEntry::Ready(promoted));
                        old.dead.store(true, Ordering::Release);
                        bump_shadow_epoch();
                        true
                    } else {
                        false
                    }
                };
                if swapped {
                    crate::host::retire(old);
                }
            }
        }
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter_map(LooseEntry::as_ready)
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Cached GPU virtual address of the underlying `self.buffer` resource.
    #[inline]
    pub(crate) fn buffer_gpu(&self) -> u64 {
        self.gpu - self.place.as_ref().map_or(0, |p| p.bias)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether the memory behind the copy is confirmed `MEM_PRIVATE` with
    /// kernel write-watching (never a section view, never guarded).
    #[inline]
    pub(crate) fn is_private_watched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == 1
    }

    /// Whether a submission asks the kernel about the copy's ranges ahead:
    /// not of memory that is not watched, nor of view memory, which the
    /// kernel would refuse (and with it the ranges asked about together),
    /// nor of buffers fully covered by active GPU-only `PAGE_NOACCESS` tripwires.
    pub(crate) fn asked_ahead(&self) -> bool {
        if self.gpu_only_tripwired() {
            return false;
        }
        let watched = self.watched.load(Ordering::Relaxed);
        watched != NOT_WATCHED && (watched != GUARDED || !crate::watch::light())
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Whether this shadow is a loose window of title memory outside buffer objects.
    #[inline]
    pub(crate) fn is_window(&self) -> bool {
        self.window
    }

    /// Ensures the demand-committed [`CpuStaging`] pages covering offsets
    /// `[from, to)` are committed before reading or writing `self.data` (`ARCH-05`).
    #[inline]
    pub(crate) fn ensure_staging_committed(&self, from: usize, to: usize) -> bool {
        if let Some(card) = &self.card
            && let CardStaging::Cpu(cpu) = &card.staging
        {
            return cpu.ensure_committed(from, to);
        }
        true
    }

    /// Decommits demand-committed [`CpuStaging`] pieces in `[piece_lo, piece_hi)`
    /// (keeping virtual address space reserved for instant re-commit on demotion).
    fn decommit_staging_pieces(&self, piece_lo: usize, piece_hi: usize) -> usize {
        if let Some(card) = &self.card
            && let CardStaging::Cpu(cpu) = &card.staging
        {
            return cpu.decommit_pieces(piece_lo, piece_hi);
        }
        0
    }

    /// Reads `[offset, offset + dst.len())` from host-side `self.data` safely
    /// without faulting on decommitted/reserved [`CpuStaging`] pieces.
    fn read_staging_into(&self, offset: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        let total = self.len as usize;
        if offset >= total {
            dst.fill(0);
            return;
        }
        let n = dst.len().min(total - offset);
        if let Some(card) = &self.card
            && let CardStaging::Cpu(cpu) = &card.staging
        {
            cpu.read_committed_into(offset, &mut dst[..n]);
        } else if !self.data.is_null() {
            // SAFETY: non-CpuStaging `self.data` is persistently mapped for `self.len` bytes.
            let src = unsafe { std::slice::from_raw_parts(self.data.add(offset).cast_const(), n) };
            dst[..n].copy_from_slice(src);
        } else {
            dst[..n].fill(0);
        }
        if n < dst.len() {
            dst[n..].fill(0);
        }
    }

    /// Whether it is no copy but title memory in place (`crate::inplace`).
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.bias)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some(InPlace {
            heap: placed.heap,
            bias: placed.bias,
            _view: placed.view,
            pinned: placed.pinned,
            _private_guard: placed.private_guard,
        });
        let end = start.saturating_add(len);
        IN_PLACE_BOUNDS[0].fetch_min(start, Ordering::Relaxed);
        IN_PLACE_BOUNDS[1].fetch_max(end, Ordering::Relaxed);
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        if self.place.is_some() {
            list.keep_shadow(self.clone());
            return;
        }
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// The bytes of the host buffer.
    pub(crate) fn width(&self) -> u64 {
        // SAFETY: a live buffer.
        *self.width.get_or_init(|| unsafe { self.buffer.GetDesc() }.Width)
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            landings: Mutex::default(),
            landing: Default::default(),
            width: Default::default(),
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new((vec![0; pages as usize], [(0, 0, 0); 2])),
            ranged_seq: AtomicU64::new(0),
            ranged_from: [AtomicUsize::new(0), AtomicUsize::new(0)],
            ranged_to: [AtomicUsize::new(0), AtomicUsize::new(0)],
            ranged_at: [AtomicU64::new(0), AtomicU64::new(0)],
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            pending_count: AtomicUsize::new(0),
            flights: Mutex::default(),
            before: Mutex::default(),
            merging: RwLock::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            touch_frame: AtomicU64::new(0),
            touch_frames: AtomicU64::new(0),
            copied_bytes: AtomicU64::new(0),
            l0_bytes: 0,
            #[cfg(feature = "diagnostics")]
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            #[cfg(feature = "diagnostics")]
            unsized_asks: AtomicU64::new(0),
            written: Default::default(),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            dead: std::sync::atomic::AtomicBool::new(false),
            changes_only: Default::default(),
            mapped: false,
            place: None,
            gpu_only_state: AtomicU8::new(GPU_ONLY_OBSERVING),
            tripwire_armed: std::sync::atomic::AtomicBool::new(false),
            gpu_only_promoted_pieces: AtomicUsize::new(0),
            gpu_only_total_pieces: if len == 0 {
                0
            } else {
                pieces_of(start as usize, 0, len as usize).end
            },
            in_place_overlap: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether this buffer/window was unregistered or replaced in-place.
    #[inline]
    pub(crate) fn is_dead(&self) -> bool {
        self.dead.load(Ordering::Acquire)
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account. Notify
        // overlapping GPU-only buffers, textures, and any overlapping shadowed
        // copies so they never hold stale bytes (`ARCH-06 #4`).
        if self.place.is_some() {
            if len > 0 {
                demote_overlapping_gpu_only(address, len, self.id, "overlapping in-place gpu write");
                if self.in_place_overlap.load(Ordering::Relaxed) {
                    crate::descriptor::host::remapped(address as usize, len as usize);
                    crate::watch::wrote(address as usize, len as usize);
                }
            }
            return;
        }
        self.written.store(true, Ordering::Relaxed);
        if self.gpu_only_skip()
            || (len > 0 && {
                let (from, to) = self.inside(address as usize, len as usize);
                from < to && self.range_all_gpu_only(from, to)
            })
        {
            GPU_ONLY_STATS[3].fetch_add(1, Ordering::Relaxed);
            GPU_ONLY_STATS[4].fetch_add(len, Ordering::Relaxed);
        }
        // Always defer `Step::GpuWrite` even for promoted buffers so `batch.flush_one()`,
        // lightweight `self.pending`/`self.flights`, and dynamic `gpu_dirty_bits` remain
        // coherent; `card_span()` skips `CARD_COMPARE` while promoted (`ARCH-06`).
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        // Dynamically track declared GPU write pieces even after GPU-only promotion
        // so later CPU uploads overlapping newly written GPU pieces cleanly demote (`ARCH-06`).
        if let Some(card) = &self.card
            && gpu_only_mode() > 0
            && (self.gpu_only_state.load(Ordering::Acquire) == GPU_ONLY_PROMOTED || self.any_piece_gpu_only())
        {
            let (from, to) = self.inside(address, len);
            if from < to {
                let pieces = self.pieces(from, to);
                card.gpu_only
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .note_declared_write(pieces.start, pieces.end);
            }
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            self.pending_count.store(pending.len(), Ordering::Release);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        self.pending_count.store(pending.len(), Ordering::Release);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                // Title memory as read, beside it with the card's bytes put
                // in: only the dwords that differ are written.
                let was: Vec<u8> = title.iter().zip(&mask).map(|(now, changed)| now ^ changed).collect();
                if !runs.is_empty() {
                    written += put_changed(start + at + first, &was[..n], &title[..n]);
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        let was = title.clone();
        if !merge_piece(now, held, title) {
            return 0;
        }
        put_changed(start + at, &was, title)
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| free.binary_search(r).is_err());
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        self.pending_count.store(pending.len(), Ordering::Release);
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission if it meets the size threshold and
            // fits in `VICTORY_KEEP_SNAPSHOTS_MB`.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= kept_min_bytes() {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        self.changed_runs(std::iter::once((from, to)));
    }

    /// [`Shadow::changed`] for each `(from, to)` offset range under a single lock.
    fn changed_runs(&self, runs: impl Iterator<Item = (usize, usize)>) {
        if !keeps() {
            return;
        }
        let mut runs = runs.filter(|r| r.0 < r.1).peekable();
        if runs.peek().is_none() {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for (from, to) in runs {
            for piece in self.pieces(from, to) {
                kept.0[piece] = false;
            }
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        self.pending_count.load(Ordering::Acquire) != 0
    }

    /// Whether one of those submissions may be writing in title memory
    /// `[from, to)`.
    pub(crate) fn writes_pending_in(&self, from: usize, to: usize) -> bool {
        self.pending_count.load(Ordering::Acquire) != 0
            && self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        if self.gpu_only_tripwired() {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to || self.pending_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        if self.range_all_gpu_only(from - start, to - start) {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let has_mode2 = self.any_piece_gpu_only();
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => {
                let stale = stale_pages_in(base, &copied.0, a, e, &written);
                if has_mode2 {
                    stale.into_iter().any(|p| !self.piece_is_gpu_only(p - base))
                } else {
                    !stale.is_empty()
                }
            }
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        if self.gpu_only_tripwired() {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to || self.pending_count.load(Ordering::Acquire) == 0 {
            return false;
        }
        if self.range_all_gpu_only(from - start, to - start) {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            note_stale(0);
            return true;
        }
        // The declared ranges overlap (views of the title's whole heap from
        // dozens of offsets): as one range each page is asked about and
        // compared once. Every byte of it is in a declared range, and of a
        // copy in the PC's memory in the snapshot.
        let busy = if crate::watch::light() { joined(busy) } else { busy };
        let base = start / PAGE;
        UNDER_NS[9].fetch_add(1, Ordering::Relaxed);
        for (a, e) in busy {
            let t = std::time::Instant::now();
            // Fast path: scan the journal directly without allocating `written`
            // when `ahead` or `PUBLISHED` already covers `[a, e)`. Acquire
            // `JOURNAL.read()` before `self.copied.lock()` to match `nothing_new`'s
            // lock order.
            let fast = {
                let j = crate::watch::journal_read();
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                crate::watch::stale_pages_in_journal(&j, a, e, base, &copied.0)
            };
            let (now, mut stale, t) = if let Some((now, stale)) = fast {
                let t = under_ns(0, t);
                UNDER_NS[8].fetch_add(stale.len() as u64, Ordering::Relaxed);
                let t = under_ns(1, t);
                (now, stale, t)
            } else {
                // As the refresh itself asks (`update_free`): the journal if
                // the submission's asking covered the range, else the kernel,
                // else the guards. The journal alone said nothing for copies
                // of view memory, which is guarded and not write-watched, and
                // those were all of run 99's cases: the flights were settled
                // 4-5 times a frame for "0 pages looked at".
                let Some((now, written)) = crate::watch::writes(a, e, true) else {
                    note_stale(1);
                    return true;
                };
                let t = under_ns(0, t);
                let stale: Vec<(usize, u64)> = {
                    let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    stale_pages_in_with_taken(base, &copied.0, a, e, &written)
                };
                UNDER_NS[8].fetch_add(stale.len() as u64, Ordering::Relaxed);
                let t = under_ns(1, t);
                (now, stale, t)
            };
            if self.any_piece_gpu_only() {
                stale.retain(|&(page, _)| !self.piece_is_gpu_only(page - base));
            }
            if stale.is_empty() {
                continue;
            }
            // Narrow write-watched stale pages to their dirty 4 KiB parts (up to
            // 16x less `read_into` + AVX2 compare traffic on partial-page writes).
            let page_parts: Option<Vec<u16>> = if crate::watch::dirty4k_narrow()
                && stale.iter().any(|&(_, taken)| taken != 0)
            {
                let mut parts = crate::watch::parts_written(&stale);
                for (p, &(_, taken)) in parts.iter_mut().zip(&stale) {
                    if taken == 0 || *p == 0 {
                        *p = u16::MAX;
                    }
                }
                Some(parts)
            } else {
                None
            };
            // The lock of the copy's reference (the snapshot, or the
            // CPU's side of a copy in the card's memory), from before
            // title memory is read until the title's bytes are in: with
            // the copy-back on its own thread ([`relay`]) that thread
            // puts what the card wrote into title memory and into the
            // reference under the same lock, and a page read before
            // such a merge and compared after it would take the card's
            // bytes for the title's.
            let merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
            let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
            let held = if self.card.is_some() {
                self.data as usize
            } else {
                before.as_mut().map_or(0, |snapshot| snapshot.as_mut_ptr() as usize)
            };
            if held == 0 {
                note_stale(3);
                return true;
            }
            let t = under_ns(2, t);
            // A page's part in the range, compared and put in: the runs it
            // changed; none for a page that is not there (left as it is, and
            // looked at again next time) or that had too many.
            type Done = Option<Vec<(usize, usize)>>;
            let left = std::sync::atomic::AtomicI64::new(*runs_left as i64);
            let over = std::sync::atomic::AtomicBool::new(false);
            let run_page = |i: usize| -> Result<Done, ()> {
                let page = stale[i].0;
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    return Ok(Some(Vec::new()));
                }
                let parts = page_parts.as_ref().map_or(u16::MAX, |p| p[i]);
                // SAFETY: the reference holds the copy's bytes at the copy's
                // offsets, and these are in a declared range; the locks held
                // above keep out what else writes them, and no two pages are
                // the same.
                unsafe { self.merge_page_parts_under_writes(page, lo, hi, parts, held as *mut u8, &left) }
            };
            let mut taken = Vec::new();
            // The runs of all the pages, noted at once (`card.dirty` is
            // sorted and joined when it is taken).
            let mut changed: Vec<(usize, usize)> = Vec::new();
            let (mut runs_in, mut bytes_in) = (0u64, 0u64);
            let mut record = |i: usize, runs: Done| {
                let Some(runs) = runs else {
                    return;
                };
                let page = stale[i].0;
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                runs_in += runs.len() as u64;
                bytes_in += runs.iter().map(|r| r.1 as u64).sum::<u64>();
                if self.card.is_some() {
                    changed.extend(runs.into_iter().map(|(r, m)| (lo - start + r, lo - start + r + m)));
                }
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if lo < hi && (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    taken.push(page);
                }
            };
            // (Not on the threads of `wide`: the thread that waits there
            // takes whatever work comes, and a landing's takes the lock.)
            if stale.len() > 2 && under_wide() {
                struct DisjointSlots<T>(Vec<std::cell::UnsafeCell<T>>);
                // SAFETY: `land::each_under(len, f)` executes each unique index
                // `i in 0..len` at most once and joins all workers before return.
                unsafe impl<T: Send> Sync for DisjointSlots<T> {}
                impl<T> DisjointSlots<T> {
                    /// # Safety
                    /// Caller guarantees each index `i` is written by at most one thread.
                    unsafe fn set(&self, i: usize, val: T) {
                        unsafe { *self.0[i].get() = val };
                    }
                }
                let done = DisjointSlots((0..stale.len()).map(|_| std::cell::UnsafeCell::new(None)).collect());
                let one = |i: usize| {
                    if !over.load(Ordering::Relaxed) {
                        match run_page(i) {
                            // SAFETY: index `i` is owned exclusively by this worker invocation.
                            Ok(runs) => unsafe { done.set(i, runs) },
                            Err(()) => over.store(true, Ordering::Relaxed),
                        }
                    }
                };
                crate::land::each_under(stale.len(), &one);
                for (i, cell) in done.0.into_iter().enumerate() {
                    record(i, cell.into_inner());
                }
            } else {
                for i in 0..stale.len() {
                    if over.load(Ordering::Relaxed) {
                        break;
                    }
                    match run_page(i) {
                        Ok(runs) => record(i, runs),
                        Err(()) => over.store(true, Ordering::Relaxed),
                    }
                }
            }
            let t = under_ns(3, t);
            *runs_left = left.into_inner().max(0) as usize;
            MERGED[1].fetch_add(runs_in, Ordering::Relaxed);
            MERGED[2].fetch_add(bytes_in, Ordering::Relaxed);
            self.card_changed_runs(changed.into_iter());
            drop(before);
            drop(merging);
            {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in taken {
                    if let Some(at) = copied.0.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
            under_ns(4, t);
            if over.into_inner() {
                note_stale(2);
                return true;
            }
        }
        false
    }

    /// Runs [`Self::merge_page_under_writes`] over either the whole `[lo, hi)`
    /// of `page` (`parts == u16::MAX`) or only the contiguous 4 KiB sub-pages
    /// selected by `parts`.
    unsafe fn merge_page_parts_under_writes(
        &self,
        page: usize,
        lo: usize,
        hi: usize,
        parts: u16,
        held: *mut u8,
        left: &std::sync::atomic::AtomicI64,
    ) -> Result<Option<Vec<(usize, usize)>>, ()> {
        if parts == u16::MAX {
            // SAFETY: caller's contract.
            return unsafe { self.merge_page_under_writes(lo, hi - lo, held, left) };
        }
        let mut all_runs = Vec::new();
        let mut mask = parts;
        while mask != 0 {
            let first = mask.trailing_zeros() as usize;
            let past = first + (!(mask >> first)).trailing_zeros() as usize;
            mask &= if past >= 16 { 0 } else { u16::MAX << past };
            let sub_lo = (page * PAGE + first * KERNEL_PAGE).max(lo);
            let sub_hi = (page * PAGE + past.min(16) * KERNEL_PAGE).min(hi);
            if sub_lo >= sub_hi {
                continue;
            }
            // SAFETY: caller's contract; `[sub_lo, sub_hi) ⊆ [lo, hi)`.
            match unsafe { self.merge_page_under_writes(sub_lo, sub_hi - sub_lo, held, left)? } {
                Some(runs) => {
                    let delta = sub_lo - lo;
                    all_runs.extend(runs.into_iter().map(|(r, m)| (r + delta, m)));
                }
                None => return Ok(None),
            }
        }
        Ok(Some(all_runs))
    }

    /// Of [`Shadow::merge_under_writes`]: title memory `[lo, lo + n)`, in one
    /// page, compared with the reference, which begins at `held`, and put in.
    /// The runs that changed, none if it cannot be read, and an error if
    /// there were more to copy one by one than `left`.
    ///
    /// # Safety
    /// The reference holds the copy's length, and nothing else writes these
    /// bytes of it during the call.
    unsafe fn merge_page_under_writes(
        &self,
        lo: usize,
        n: usize,
        held: *mut u8,
        left: &std::sync::atomic::AtomicI64,
    ) -> Result<Option<Vec<(usize, usize)>>, ()> {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let title = &mut title[..n];
            let t = std::time::Instant::now();
            if !crate::host::read_into(lo as *const u8, title) {
                return Ok(None);
            }
            let t = under_ns(5, t);
            let _compared = Compared(t);
            MERGED[0].fetch_add(1, Ordering::Relaxed);
            let at = lo - self.start as usize;
            if self.card.is_some() && !self.ensure_staging_committed(at, at + n) {
                return Ok(None);
            }
            // SAFETY: the caller's (committed above if backed by `CpuStaging`).
            let held = unsafe { std::slice::from_raw_parts_mut(held.add(at), n) };
            if self.card.is_some() {
                // Whole words go up as pairs, all of them in one
                // dispatch: only what is a copy of its own counts.
                // Compared and put in, in one pass ([`under_fused`]), where
                // the most copies of their own the page can have are left
                // to take: they are taken first, and what is not used is
                // given back. (Whole words at word offsets of the copy: a
                // copy of its own is a run of more than 256 bytes.)
                let most = (n / 260 + 1) as i64;
                let fused = crate::host::words_under()
                    && under_fused()
                    && !crate::host::upcheck::on()
                    && at % 4 == 0
                    && lo % 4 == 0
                    && n % 4 == 0
                    && {
                        if left.fetch_sub(most, Ordering::Relaxed) >= most {
                            true
                        } else {
                            left.fetch_add(most, Ordering::Relaxed);
                            false
                        }
                    };
                let runs = if fused {
                    word_runs_put(lo, title, held)
                } else if crate::host::words_under() {
                    word_runs(lo, title, held)
                } else {
                    changed_runs(title, held)
                };
                let own = |&&(r, m): &&(usize, usize)| !crate::host::words_under() || !crate::host::goes_as_words(at + r, m);
                let copies = runs.iter().filter(own).count() as i64;
                if fused {
                    debug_assert!(copies <= most);
                    left.fetch_add(most - copies, Ordering::Relaxed);
                    return Ok(Some(runs));
                }
                if copies > 0 && left.fetch_sub(copies, Ordering::Relaxed) < copies {
                    return Err(());
                }
                let was = (!runs.is_empty() && crate::host::upcheck::on()).then(|| held.to_vec());
                for &(r, m) in &runs {
                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                }
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at, &was, held);
                }
                return Ok(Some(runs));
            }
            let runs = changed_runs(title, held);
            for &(r, m) in &runs {
                // SAFETY: inside the copy's mapping. The card may be
                // writing the copy beside these bytes; they are
                // written here and nothing around them.
                unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                held[r..r + m].copy_from_slice(&title[r..r + m]);
            }
            Ok(Some(runs))
        })
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if self.pending_count.load(Ordering::Acquire) != 0 {
            let mut busy: Vec<(usize, usize)> = self
                .pending
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .copied()
                .filter(|&(a, e)| a < to && e > from)
                .collect();
            if !busy.is_empty() {
                // Around ranges shaders are writing: the gaps, each as below
                // (in the world most submissions have such ranges; copying the
                // gaps whole cost ~1 s per frame).
                busy.sort_unstable();
                let (mut at, mut bytes) = (from, 0);
                for (a, e) in busy {
                    if a > at {
                        bytes += self.update_free(at, a);
                    }
                    at = at.max(e);
                }
                if at < to {
                    bytes += self.update_free(at, to);
                }
                return bytes;
            }
        }
        self.update_free(from, to)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would copy
    /// nothing, and knows it without asking anything
    /// (`crate::watch::nothing_new`). `false` if it would have to ask.
    pub(crate) fn nothing_new(&self, address: usize, len: usize) -> bool {
        let j = crate::watch::journal_read();
        let mut scan = j.scan();
        self.nothing_new_with_scan(&mut scan, address, len)
    }

    /// Same as [`Shadow::nothing_new`], reusing a held [`crate::watch::JournalRead`]
    /// so a chunk of spans acquires `JOURNAL.read()` only once.
    #[allow(dead_code)]
    pub(crate) fn nothing_new_with_journal(&self, j: &crate::watch::JournalRead, address: usize, len: usize) -> bool {
        let mut scan = j.scan();
        self.nothing_new_with_scan(&mut scan, address, len)
    }

    /// Same as [`Shadow::nothing_new`], reusing a [`crate::watch::JournalScan`]
    /// so consecutive spans in the same journal shard avoid SRWLock RMW churn.
    pub(crate) fn nothing_new_with_scan(&self, scan: &mut crate::watch::JournalScan<'_>, address: usize, len: usize) -> bool {
        if self.place.is_some() || self.gpu_only_tripwired() {
            return true;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return true;
        }
        if self.range_all_gpu_only(from - start, to - start) {
            return true;
        }
        let watched = self.watched.load(Ordering::Relaxed);
        if !matches!(watched, 1 | GUARDED) {
            return false;
        }
        // Around ranges shaders are writing the refresh asks gap by gap.
        if self.pending_count.load(Ordering::Acquire) != 0
            && self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from)
        {
            return false;
        }
        let prefer_published = watched == GUARDED;
        // Lock-free seqlock fast path against both mirrored whole ranges (`copied.1`).
        let s1 = self.ranged_seq.load(Ordering::Acquire);
        if s1 != 0 && s1 & 1 == 0 {
            let r0 = (
                self.ranged_from[0].load(Ordering::Relaxed),
                self.ranged_to[0].load(Ordering::Relaxed),
                self.ranged_at[0].load(Ordering::Relaxed),
            );
            let r1 = (
                self.ranged_from[1].load(Ordering::Relaxed),
                self.ranged_to[1].load(Ordering::Relaxed),
                self.ranged_at[1].load(Ordering::Relaxed),
            );
            std::sync::atomic::fence(Ordering::Acquire);
            if self.ranged_seq.load(Ordering::Relaxed) == s1 {
                let mut ra = 0u64;
                if r0.0 <= from && to <= r0.1 {
                    ra = ra.max(r0.2);
                }
                if r1.0 <= from && to <= r1.1 {
                    ra = ra.max(r1.2);
                }
                if ra != 0 {
                    let same = crate::watch::nothing_new_ranged_scan(scan, from, to, ra, prefer_published);
                    if same {
                        #[cfg(feature = "diagnostics")]
                        {
                            use crate::copystat::{C, add};
                            add(C::Checked, (to - from) as u64);
                            add(C::Unchanged, (to - from) as u64);
                        }
                        return true;
                    }
                }
            }
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let ranged = Self::ranged_in(&copied.1, from, to);
        let same = crate::watch::nothing_new_scan(scan, from, to, start / PAGE, &copied.0, ranged, prefer_published);
        #[cfg(feature = "diagnostics")]
        if same {
            use crate::copystat::{C, add};
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from) as u64);
        }
        same
    }

    /// The latest asking after which a range that holds `[from, to)` was
    /// taken whole (0: none is known).
    fn ranged_in(ranged: &[(usize, usize, u64); 2], from: usize, to: usize) -> u64 {
        ranged.iter().filter(|r| r.0 <= from && to <= r.1).map(|r| r.2).max().unwrap_or(0)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.gpu_only_tripwired() {
            return 0;
        }
        let start = self.start as usize;
        if from < to && self.range_all_gpu_only(from - start, to - start) {
            return 0;
        }
        let watched = self.watched.load(Ordering::Relaxed);
        if watched != NOT_WATCHED {
            if matches!(watched, 1 | GUARDED)
                && let Some(copied) = self.update_free_journal(from, to, watched)
            {
                return copied;
            }
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                if watched != GUARDED {
                    let next = if crate::watch::guarded(from, to) { GUARDED } else { 1 };
                    if watched != next {
                        self.watched.store(next, Ordering::Relaxed);
                    }
                }
                let copied = self.copy_stale(from, to, now, &written);
                if self.window && copied > 0 {
                    self.copied_bytes.fetch_add(copied as u64, Ordering::Relaxed);
                }
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let copied = self.copy(from, to);
        if self.window && copied > 0 {
            self.copied_bytes.fetch_add(copied as u64, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        copied
    }

    /// Fast path for [`Shadow::update_free`] when the journal already covers
    /// `[from, to)` (`ahead` or `PUBLISHED`): scans stale pages directly
    /// without allocating or sorting a full `written` Vec.
    fn update_free_journal(&self, from: usize, to: usize, watched: u8) -> Option<usize> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let first = from / PAGE;
        let last = (to - 1) / PAGE;
        let whole_bounds = (first * PAGE).max(start) >= from && ((last + 1) * PAGE).min(end) <= to;
        // Acquire `JOURNAL.read()` before `self.copied.lock()` to match
        // `nothing_new`'s lock order, and pass `j` into `copy_stale_pages` so
        // `parts_written_with_journal` never re-acquires `JOURNAL.read()` under
        // `self.copied` (which would ABBA-deadlock with a waiting `JOURNAL.write()`).
        let j = crate::watch::journal_read();
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let ranged = if crate::watch::light() { Self::ranged_in(&copied.1, from, to) } else { 0 };
        if ranged != 0 && !whole_bounds {
            return None;
        }
        let (now, stale) = crate::watch::stale_pages_in_journal(&j, from, to, base, &copied.0)?;
        let bytes = self.copy_stale_pages(&mut copied, Some(j), from, to, now, ranged, &stale);
        drop(copied);
        if self.window && bytes > 0 {
            self.copied_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        }
        if watched != GUARDED && crate::watch::guarded(from, to) {
            self.watched.store(GUARDED, Ordering::Relaxed);
        }
        self.refresh_check(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(bytes) as u64);
        }
        Some(bytes)
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n])
                && self.ensure_staging_committed(at - start, at - start + n)
            {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // committed above if backed by `CpuStaging`.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        if self.gpu_only_tripwired() {
            return 0;
        }
        let base = self.start as usize / PAGE;
        // Acquire `JOURNAL.read()` before `self.copied.lock()` whenever dirty-4K
        // narrowing/accounting is active so lock order matches `nothing_new`.
        let j = (crate::watch::dirty4k_narrow() || crate::watch::dirty4k_kept())
            .then(crate::watch::journal_read);
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        // Pages the range holds in part are never noted in `copied`, and
        // were copied again at every refresh of the range, written or not:
        // of the heap's buffer views alone two ends for each of 4,400
        // copies, every frame. A range that was taken whole, and not written
        // since, is as it was.
        let ranged = if crate::watch::light() { Self::ranged_in(&copied.1, from, to) } else { 0 };
        let (mut runs, mut run) = (Vec::new(), None::<(usize, bool)>);
        // Pages taken before and written since: in the order of their runs.
        let mut again_at: Vec<(usize, u64)> = Vec::new();
        let last = (to - 1) / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_mode2 = self.any_piece_gpu_only();
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            if has_mode2 && self.piece_is_gpu_only(page - base) {
                if let Some((first, known)) = run.take() {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                }
                continue;
            }
            let at = &mut copied.0[page - base];
            let taken = (*at).max(ranged);
            if taken == 0 && self.gpu_only_tripwired() && self.gpu_only_page_is_gpu_dirty(page - base) {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                if let Some((first, known)) = run.take() {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                }
                continue;
            }
            if taken == 0 || write > taken {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                // Cut where pages taken before meet pages never taken.
                let again = taken != 0;
                if again {
                    again_at.push((page, taken));
                }
                if let Some((first, known)) = run
                    && known != again
                {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                    run = None;
                }
                run.get_or_insert((page, again));
            } else if let Some((first, known)) = run.take() {
                runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((first, known)) = run {
            runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to), known));
        }
        self.finish_copy_stale(&mut copied, j, from, to, now, runs, &again_at)
    }

    /// Builds stale runs directly from `stale` `(page, copied_at)` pairs found
    /// by [`crate::watch::stale_pages_in_journal`] and executes the copy.
    fn copy_stale_pages(
        &self,
        copied: &mut (Vec<u64>, [(usize, usize, u64); 2]),
        j: Option<crate::watch::JournalRead>,
        from: usize,
        to: usize,
        now: u64,
        ranged: u64,
        stale: &[(usize, u64)],
    ) -> usize {
        if self.gpu_only_tripwired() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        let mut again_at: Vec<(usize, u64)> = Vec::new();
        let mut run: Option<(usize, usize, bool)> = None;
        let has_mode2 = self.any_piece_gpu_only();
        for &(page, at) in stale {
            if has_mode2 && self.piece_is_gpu_only(page - base) {
                if let Some((first, next, known)) = run.take() {
                    runs.push(((first * PAGE).max(from), (next * PAGE).min(to), known));
                }
                continue;
            }
            let taken = at.max(ranged);
            if taken == 0 && self.gpu_only_tripwired() && self.gpu_only_page_is_gpu_dirty(page - base) {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    copied.0[page - base] = now;
                }
                if let Some((first, next, known)) = run.take() {
                    runs.push(((first * PAGE).max(from), (next * PAGE).min(to), known));
                }
                continue;
            }
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                copied.0[page - base] = now;
            }
            let again = taken != 0;
            if again {
                again_at.push((page, taken));
            }
            match run {
                Some((first, next, known)) if next == page && known == again => {
                    run = Some((first, page + 1, known));
                }
                Some((first, next, known)) => {
                    runs.push(((first * PAGE).max(from), (next * PAGE).min(to), known));
                    run = Some((page, page + 1, again));
                }
                None => {
                    run = Some((page, page + 1, again));
                }
            }
        }
        if let Some((first, next, known)) = run {
            runs.push(((first * PAGE).max(from), (next * PAGE).min(to), known));
        }
        self.finish_copy_stale(copied, j, from, to, now, runs, &again_at)
    }

    /// Narrows `runs` by 4 KiB dirty parts, copies them into the shadow, and
    /// updates the whole-range mirror (`copied.1` + seqlock).
    fn finish_copy_stale(
        &self,
        copied: &mut (Vec<u64>, [(usize, usize, u64); 2]),
        j: Option<crate::watch::JournalRead>,
        from: usize,
        to: usize,
        now: u64,
        runs: Vec<(usize, usize, bool)>,
        again_at: &[(usize, u64)],
    ) -> usize {
        let narrow = crate::watch::dirty4k_narrow();
        let kept = crate::watch::dirty4k_kept();
        let runs = if !again_at.is_empty() && (narrow || kept) {
            let parts = match j.as_ref() {
                Some(j) => crate::watch::parts_written_with_journal(j, again_at),
                None => crate::watch::parts_written(again_at),
            };
            drop(j);
            if kept {
                for &bits in &parts {
                    let found = (bits != u16::MAX).then_some(bits);
                    crate::watch::dirty4k_count(found, narrow && found.is_some());
                }
            }
            if narrow { by_parts(runs, again_at, &parts) } else { runs }
        } else {
            drop(j);
            runs
        };
        let bytes = self.take_all(runs);
        if crate::watch::light() {
            // All of the range is in the copy as of this asking. Keep both
            // slots distinct when refreshing the longest range or alternating
            // equal-sized ranges.
            let ranged = &mut copied.1;
            if ranged[1].0 == from && ranged[1].1 == to {
                ranged[1].2 = now;
            } else if to - from > ranged[1].1 - ranged[1].0 {
                ranged[0] = (from, to, now);
                ranged[1] = (from, to, now);
            } else {
                ranged[0] = (from, to, now);
            }
            let seq = self.ranged_seq.load(Ordering::Relaxed);
            self.ranged_seq.store(seq + 1, Ordering::Release);
            for slot in 0..2 {
                self.ranged_from[slot].store(ranged[slot].0, Ordering::Relaxed);
                self.ranged_to[slot].store(ranged[slot].1, Ordering::Relaxed);
                self.ranged_at[slot].store(ranged[slot].2, Ordering::Relaxed);
            }
            self.ranged_seq.store(seq + 2, Ordering::Release);
        }
        bytes
    }

    /// Packs `(from, to, known)` runs into byte-balanced chunks of about
    /// `chunk_bytes` bytes each (splitting long runs at `PAGE` boundaries and
    /// grouping many small 4 KiB runs into one chunk).
    fn pack_take_chunks(
        runs: &[(usize, usize, bool)],
        chunk_bytes: usize,
    ) -> (Vec<(usize, usize, bool)>, Vec<std::ops::Range<usize>>) {
        let chunk_bytes = chunk_bytes.max(PAGE);
        let mut flat: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + 8);
        let mut chunks: Vec<std::ops::Range<usize>> = Vec::new();
        let mut start = 0usize;
        let mut bytes = 0usize;
        for &(from, to, known) in runs {
            let mut at = from;
            while at < to {
                let budget = chunk_bytes.saturating_sub(bytes);
                let next = if to - at <= budget {
                    to
                } else {
                    ((at / PAGE) * PAGE + budget.max(PAGE)).min(to)
                };
                let next = next.max((at + 1).min(to));
                flat.push((at, next, known));
                bytes += next - at;
                at = next;
                if bytes >= chunk_bytes {
                    chunks.push(start..flat.len());
                    start = flat.len();
                    bytes = 0;
                }
            }
        }
        if start < flat.len() {
            chunks.push(start..flat.len());
        }
        (flat, chunks)
    }

    /// [`Shadow::take`] of `runs` (from, to, known), on several threads if
    /// they are long: the big heap's copy has some 4 MB of them in each of
    /// 14 refreshes a frame, on one thread while the others had nothing.
    fn take_all(&self, runs: Vec<(usize, usize, bool)>) -> usize {
        const LONG: usize = 256 << 10;
        const PART_MIN: usize = 64 << 10;
        const PART_MAX: usize = 256 << 10;
        let bytes: usize = runs.iter().map(|r| r.1 - r.0).sum();
        if bytes == 0 {
            return 0;
        }
        // Fast path: plain copy without per-dword diffing or landing echo.
        // Workers only copy bytes (zero mutex contention on `card.dirty` or
        // `self.kept`), and dirty/kept ranges are recorded once after copies.
        let changes_only = self.card.is_some() && (self.changes_only.load(Ordering::Relaxed) || send_changed());
        let check_landed = self.card.is_some()
            && self.landing.load(Ordering::Acquire)
            && land_echo()
            && !crate::host::upcheck::on();
        let plain = !changes_only && (!check_landed || !runs.iter().any(|r| r.2));
        let parallel = bytes >= LONG && crate::watch::light();
        if plain {
            if !parallel {
                for &(from, to, _) in &runs {
                    self.copy_bytes_only(from, to);
                }
            } else {
                let threads = wide_worker_count();
                let chunk_bytes = bytes.div_ceil(threads).clamp(PART_MIN, PART_MAX);
                let (flat, chunks) = Self::pack_take_chunks(&runs, chunk_bytes);
                if chunks.len() < 2 {
                    for &(from, to, _) in &flat {
                        self.copy_bytes_only(from, to);
                    }
                } else {
                    wide_each(chunks.len(), |i| {
                        for &(from, to, _) in &flat[chunks[i].clone()] {
                            self.copy_bytes_only(from, to);
                        }
                    });
                }
            }
            let (start, end) = (self.start as usize, (self.start + self.len) as usize);
            self.changed_runs(runs.iter().map(|&(a, b, _)| (a.max(start) - start, b.min(end).saturating_sub(start))));
            self.card_changed_runs(runs.iter().map(|&(a, b, _)| (a.max(start) - start, b.min(end).saturating_sub(start))));
            return bytes;
        }
        if !parallel {
            for (from, to, known) in runs {
                self.take(from, to, known);
            }
            return bytes;
        }
        let threads = wide_worker_count();
        let chunk_bytes = bytes.div_ceil(threads).clamp(PART_MIN, PART_MAX);
        let (flat, chunks) = Self::pack_take_chunks(&runs, chunk_bytes);
        if chunks.len() < 2 {
            for (from, to, known) in flat {
                self.take(from, to, known);
            }
        } else {
            wide_each(chunks.len(), |i| {
                for &(from, to, known) in &flat[chunks[i].clone()] {
                    self.take(from, to, known);
                }
            });
        }
        bytes
    }

    /// What shaders wrote in `[from, to)` (offsets) was put into title memory.
    fn note_landing(&self, from: usize, to: usize) {
        let first = self.start as usize / KERNEL_PAGE;
        let pages = (self.start as usize + from) / KERNEL_PAGE - first..(self.start as usize + to).div_ceil(KERNEL_PAGE) - first;
        let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
        if bits.is_empty() {
            let all = (self.start + self.len).div_ceil(KERNEL_PAGE as u64) as usize - first;
            bits.resize(all.div_ceil(64), 0);
        }
        for page in pages {
            bits[page / 64] |= 1 << (page % 64);
        }
        self.landing.store(true, Ordering::Release);
    }

    /// [`Shadow::take`] of `[from, to)`, which the copy held before and which
    /// is written since: where it may be only the landing that wrote it, the
    /// 4 KiB that title memory and the CPU's side have alike are left as they
    /// are, and nothing of them is sent ([`land_echo`]).
    fn take_landed(&self, from: usize, to: usize) -> usize {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let first = start / KERNEL_PAGE;
        // Runs of pages noted, and not noted (any more).
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        {
            let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = from;
            while at < to {
                let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                let page = at / KERNEL_PAGE - first;
                let noted = bits.get(page / 64).is_some_and(|word| word >> (page % 64) & 1 != 0);
                // (A page looked at in part is looked at again.)
                if noted && next - at == KERNEL_PAGE {
                    bits[page / 64] &= !(1 << (page % 64));
                }
                match runs.last_mut() {
                    Some(last) if last.2 == noted && next - last.0 <= PAGE => last.1 = next,
                    _ => runs.push((at, next, noted)),
                }
                at = next;
            }
        }
        let mut dirty_runs: Vec<(usize, usize)> = Vec::new();
        let mut push_dirty = |a: usize, b: usize| {
            if a >= b {
                return;
            }
            match dirty_runs.last_mut() {
                Some(last) if a <= last.1 && b >= last.0 => *last = (last.0.min(a), last.1.max(b)),
                _ => dirty_runs.push((a, b)),
            }
        };
        TITLE.with_borrow_mut(|title| {
            for (from, to, noted) in runs {
                if !noted {
                    self.copy_bytes_only(from, to);
                    push_dirty(from - start, to - start);
                    continue;
                }
                title.resize(to - from, 0);
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, to - from, false);
                if !self.ensure_staging_committed(from - start, to - start) {
                    continue;
                }
                let mut at = from;
                while at < to {
                    let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                    let title = &title[at - from..next - from];
                    // SAFETY: inside the mapping of the copy's CPU side (committed above),
                    // which nothing else reads or writes there: no shader's range
                    // holds it, and the parts of a refresh lie apart.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), next - at) };
                    if title == held {
                        ECHO[0].fetch_add((next - at) as u64, Ordering::Relaxed);
                    } else {
                        ECHO[1].fetch_add((next - at) as u64, Ordering::Relaxed);
                        held.copy_from_slice(title);
                        push_dirty(at - start, next - start);
                    }
                    at = next;
                }
            }
        });
        if !dirty_runs.is_empty() {
            self.changed_runs(dirty_runs.iter().copied());
            self.card_changed_runs(dirty_runs.into_iter());
        }
        to - from
    }

    /// Copies title memory `[from, to)` (inside the buffer) into the copy;
    /// `known`: the copy held it before ([`send_changed`]).
    fn take(&self, from: usize, to: usize, known: bool) -> usize {
        thread_local! {
            /// A page of title memory, as read.
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let Some(card) = self.card.as_ref().filter(|_| self.changes_only.load(Ordering::Relaxed) || send_changed()) else {
            if known && self.card.is_some() && self.landing.load(Ordering::Acquire) && land_echo() && !crate::host::upcheck::on() {
                return self.take_landed(from, to);
            }
            return self.copy(from, to);
        };
        let whole = !known && card.before.get().is_none();
        let start = self.start as usize;
        if !self.ensure_staging_committed(from - start, to - start) {
            return 0;
        }
        let mut all_dirty: Vec<(usize, usize)> = Vec::new();
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let mut at = from;
            while at < to {
                let n = (PAGE - at % PAGE).min(to - at);
                let title = &mut title[..n];
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), at as *const u8, n, false);
                // SAFETY: inside the mapping of the copy's CPU side, which
                // nothing else reads or writes there (the card copies from
                // it): no shader's range holds it, and the parts of a
                // refresh lie apart.
                let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), n) };
                let was = crate::host::upcheck::on().then(|| held.to_vec());
                let runs = if whole {
                    held.copy_from_slice(title);
                    vec![(0, n)]
                } else {
                    word_runs_put(at, title, held)
                };
                all_dirty.extend(runs.iter().map(|&(a, len)| (at - start + a, at - start + a + len)));
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at - start, &was, held);
                }
                SEND[0].fetch_add(n as u64, Ordering::Relaxed);
                if runs.is_empty() {
                    SEND[1].fetch_add(n as u64, Ordering::Relaxed);
                }
                SEND[2].fetch_add(runs.iter().map(|r| r.1 as u64).sum(), Ordering::Relaxed);
                SEND[3].fetch_add(runs.len() as u64, Ordering::Relaxed);
                at += n;
            }
        });
        if !all_dirty.is_empty() {
            self.changed_runs(all_dirty.iter().copied());
            self.card_changed_runs(all_dirty.into_iter());
        }
        to - from
    }

    /// Copies title memory `[from, to)` (cut to the buffer) into `self.data`
    /// without acquiring `self.kept` or `card.dirty`.
    fn copy_bytes_only(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        if !self.ensure_staging_committed(from - start, to - start) {
            return 0;
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer,
        // committed above if backed by `CpuStaging`).
        let at = unsafe { self.data.add(from - start) };
        if self.card.is_some() && crate::host::upcheck::on() {
            for_pieces_of(from - start, to - start, |a, b| {
                let was = self.card_held(a, b - a);
                // SAFETY: inside the mapping, as above.
                crate::host::copy_mapped(unsafe { self.data.add(a) }, (start + a) as *const u8, b - a, false);
                crate::host::upcheck::note(self.id, a, &was, &self.card_held(a, b - a));
            });
        } else {
            crate::host::copy_mapped(at, from as *const u8, to - from, false);
        }
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        let copied = self.copy_bytes_only(from, to);
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        copied
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Shaders store in it what is not declared: from here on only the dwords
    /// the title changes go up ([`by_address`]).
    pub(crate) fn keep_stores(&self) {
        self.changes_only.store(true, Ordering::Relaxed);
    }

    /// Offsets `[from, to)` go up again whole, as the CPU's side has them.
    pub(crate) fn resend(&self, from: usize, to: usize) {
        self.card_changed(from, to);
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        self.card_changed_runs(std::iter::once((from, to)));
    }

    /// [`Shadow::card_changed`] of each of `runs`.
    fn card_changed_runs(&self, runs: impl Iterator<Item = (usize, usize)>) {
        let Some(card) = &self.card else {
            return;
        };
        let most = runs.size_hint().0;
        let mut runs = runs.filter(|r| r.0 < r.1).peekable();
        if runs.peek().is_none() {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        dirty.reserve(most);
        for (from, to) in runs {
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
                _ => dirty.push((from, to)),
            }
        }
        // (Under the lock: what takes the list sees it or finds it set.)
        card.touched.store(true, Ordering::Release);
    }

    /// Whether shaders were ever declared to write it.
    pub(crate) fn written(&self) -> bool {
        self.written.load(Ordering::Relaxed)
    }

    /// `bytes` were put into title memory at `address`, and the submission
    /// being prepared copies the same to the copy at that copy's place in its
    /// list: the reference (the CPU's side of a copy in the card's memory, the
    /// snapshot) takes them and nothing is sent. What the card makes of them
    /// is then what differs from them when the submission has run. The range
    /// is declared written by it, and its sends are taken.
    pub(crate) fn placed_copy(&self, address: usize, bytes: &[u8]) {
        let (from, to) = self.inside(address, bytes.len());
        if from >= to || self.place.is_some() {
            return;
        }
        let bytes = &bytes[self.start as usize + from - address..][..to - from];
        let _merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
        if self.card.is_some() {
            if self.ensure_staging_committed(from, to) {
                // SAFETY: inside the mapping of the CPU's side (committed above),
                // in a range no refresh writes; the lock keeps the landing out.
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(from), bytes.len()) };
            }
        } else if let Some(snapshot) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // SAFETY: the snapshot's allocation holds the copy's length.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), snapshot.as_mut_ptr().add(from), bytes.len()) };
        }
        self.changed(from, to);
    }

    /// Of a copy in the card's memory: where its CPU side is.
    pub(crate) fn card_data(&self) -> *const u8 {
        self.data
    }

    /// Diagnostics: of a copy in the card's memory, what its CPU side holds
    /// at offsets `[at, at + len)` (cut to the copy). Uncommitted demand-paged
    /// [`CpuStaging`] pieces read as zeros without forcing OS commit.
    pub(crate) fn card_held(&self, at: usize, len: usize) -> Vec<u8> {
        let to = (at + len).min(self.len as usize);
        let Some(card) = &self.card else {
            return Vec::new();
        };
        if at >= to {
            return Vec::new();
        }
        if let CardStaging::Cpu(cpu) = &card.staging {
            let mut out = vec![0u8; to - at];
            let mut pos = at;
            while pos < to {
                let piece = pos / PIECE;
                let end = ((piece + 1) * PIECE).min(to);
                if cpu.is_piece_committed(piece) {
                    // SAFETY: piece is committed within `self.data`.
                    let src = unsafe { std::slice::from_raw_parts(self.data.add(pos).cast_const(), end - pos) };
                    out[pos - at..end - at].copy_from_slice(src);
                }
                pos = end;
            }
            return out;
        }
        // SAFETY: inside the mapping; the caller holds the submission lock,
        // under which it is written but for what is being landed.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast_const(), to - at) }.to_vec()
    }

    /// Diagnostics: the offsets whose bytes have not gone to the card's side
    /// yet.
    pub(crate) fn card_dirty_now(&self) -> Vec<(usize, usize)> {
        self.card.as_ref().map_or_else(Vec::new, |c| c.dirty.lock().unwrap_or_else(|e| e.into_inner()).clone())
    }

    /// Of a copy in the card's memory: the CPU's side as a GPU resource when
    /// backed by a `shared_buffer` (`None` when backed by plain CPU memory for
    /// `VICTORY_SEND_OWN=1`), the card side's bytes from before if made yet,
    /// and their size.
    pub(crate) fn card_sides(&self) -> Option<(Option<&ID3D12Resource>, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.get().map(|(b, _)| b.clone());
        Some((card.staging.as_gpu(), before, card.size))
    }

    /// Lock-free check for whether this card buffer has its `before` hash buffer.
    #[inline]
    pub(crate) fn card_has_before(&self) -> bool {
        self.card.as_ref().is_some_and(|c| c.before.get().is_some())
    }

    /// Lock-free borrowed access to this card buffer's `before` hash resource
    /// and its cached GPU virtual address (zero mutex locks, zero COM AddRef).
    #[inline]
    pub(crate) fn card_before_ref(&self) -> Option<(&ID3D12Resource, u64)> {
        let (res, gpu) = self.card.as_ref()?.before.get()?;
        Some((res, *gpu))
    }

    /// Lock-free borrowed access to this card buffer's GPU staging resource (if any).
    #[inline]
    pub(crate) fn card_staging_gpu(&self) -> Option<&ID3D12Resource> {
        self.card.as_ref()?.staging.as_gpu()
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        if card.before.get().is_some() {
            return false;
        }
        let _init = card.before_init.lock().unwrap_or_else(|e| e.into_inner());
        if card.before.get().is_some() {
            return false;
        }
        let Some(res) = make(card.size) else {
            return false;
        };
        // SAFETY: live D3D12 buffer just created by `make`.
        let gpu = unsafe { res.GetGPUVirtualAddress() };
        CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        let pieces = card_hash_pieces(card.size);
        {
            let mut stale = card.hash_stale.lock().unwrap_or_else(|e| e.into_inner());
            stale.resize(pieces.div_ceil(64), u64::MAX);
            let rem = pieces % 64;
            if rem != 0
                && let Some(last) = stale.last_mut()
            {
                *last = (1u64 << rem) - 1;
            }
        }
        let _ = card.before.set((res, gpu));
        true
    }

    /// Marks all pieces in `Card::hash_stale` stale (e.g. after demoting a
    /// `VICTORY_GPU_ONLY_COPYBACK` buffer so the next compare re-hashes it).
    pub(crate) fn card_mark_all_hash_stale(&self) {
        let Some(card) = &self.card else {
            return;
        };
        let mut stale = card.hash_stale.lock().unwrap_or_else(|e| e.into_inner());
        if stale.is_empty() {
            return;
        }
        let pieces = card_hash_pieces(card.size);
        mark_stale_bits(&mut stale, 0, pieces);
    }

    /// Marks `[piece_lo, piece_hi)` in `Card::hash_stale` stale (used by localized
    /// Mode-2 run demotion so only demoted pieces are re-hashed).
    fn card_mark_piece_range_hash_stale(&self, piece_lo: usize, piece_hi: usize) {
        let Some(card) = &self.card else {
            return;
        };
        let mut stale = card.hash_stale.lock().unwrap_or_else(|e| e.into_inner());
        if stale.is_empty() {
            return;
        }
        let limit = card_hash_pieces(card.size);
        mark_stale_bits(&mut stale, piece_lo.min(limit), piece_hi.min(limit));
    }

    /// Whether this card buffer is whole-buffer promoted to GPU-only copyback elision.
    #[inline]
    pub(crate) fn gpu_only_skip(&self) -> bool {
        gpu_only_mode() > 0 && self.gpu_only_state.load(Ordering::Acquire) == GPU_ONLY_PROMOTED
    }

    /// Lock-free check for whether 64 KiB piece `piece` is currently promoted
    /// to Mode-2 GPU-only (or the whole buffer is Mode-1 promoted).
    #[inline]
    pub(crate) fn piece_is_gpu_only(&self, piece: usize) -> bool {
        if self.gpu_only_skip() {
            return true;
        }
        if self.gpu_only_promoted_pieces.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.card.as_ref().is_some_and(|card| {
            card.gpu_only_bitmap
                .get(piece / 64)
                .is_some_and(|w| (w.load(Ordering::Acquire) >> (piece % 64)) & 1 != 0)
        })
    }

    /// Lock-free check for whether any 64 KiB piece on this buffer is Mode-2 promoted.
    #[inline]
    pub(crate) fn any_piece_gpu_only(&self) -> bool {
        self.gpu_only_promoted_pieces.load(Ordering::Acquire) != 0
    }

    /// Lock-free check for whether every 64 KiB piece on this buffer is Mode-2 promoted.
    #[inline]
    fn all_pieces_gpu_only(&self) -> bool {
        self.gpu_only_total_pieces > 0
            && self.gpu_only_promoted_pieces.load(Ordering::Acquire) >= self.gpu_only_total_pieces
    }

    /// Lock-free check for whether every 64 KiB piece intersecting copy offsets
    /// `[from, to)` (`from < to`) is promoted to GPU-only (Mode-1 or Mode-2).
    #[inline]
    fn range_all_gpu_only(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        if self.gpu_only_skip() {
            return true;
        }
        if self.gpu_only_promoted_pieces.load(Ordering::Acquire) == 0 {
            return false;
        }
        let Some(card) = &self.card else {
            return false;
        };
        for piece in self.pieces(from, to) {
            let bit = card
                .gpu_only_bitmap
                .get(piece / 64)
                .is_some_and(|w| (w.load(Ordering::Acquire) >> (piece % 64)) & 1 != 0);
            if !bit {
                return false;
            }
        }
        true
    }

    /// Lock-free check for whether guest VA `title_va` falls in a GPU-only promoted piece.
    #[inline]
    pub(crate) fn is_va_gpu_only(&self, title_va: usize) -> bool {
        if self.gpu_only_skip() {
            return true;
        }
        if self.gpu_only_promoted_pieces.load(Ordering::Acquire) == 0 {
            return false;
        }
        let Some(off) = title_va.checked_sub(self.start as usize) else {
            return false;
        };
        if off >= self.len as usize {
            return false;
        }
        let piece = self.pieces(off, off + 1).start;
        self.piece_is_gpu_only(piece)
    }

    /// `true` strictly while the entire buffer is protected by active `PAGE_NOACCESS`
    /// hardware tripwire(s) (Mode-1 whole-buffer OR Mode-2 with every piece promoted),
    /// guaranteeing zero silent title CPU reads or writes.
    #[inline]
    fn gpu_only_tripwired(&self) -> bool {
        self.tripwire_armed.load(Ordering::Acquire) && (self.gpu_only_skip() || self.all_pieces_gpu_only())
    }

    fn gpu_only_page_is_gpu_dirty(&self, piece: usize) -> bool {
        self.card.as_ref().is_some_and(|card| {
            let tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            GpuOnlyTracker::has_bit(&tracker.gpu_dirty_bits, piece)
        })
    }

    /// Populates any untouched (`taken == 0`) non-GPU-dirty pages in
    /// `[piece_lo, piece_hi)` into `self.data` and marks them stamped in `self.copied`
    /// right before arming a `PAGE_NOACCESS` hardware tripwire.
    fn seal_unfetched_pages_range(&self, card: &Card, piece_lo: usize, piece_hi: usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let lo = piece_lo.min(copied.0.len());
        let hi = piece_hi.min(copied.0.len());
        if lo >= hi {
            return;
        }
        let stamp = copied
            .0
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .max(copied.1[0].2)
            .max(copied.1[1].2)
            .max(1);
        let range_va_lo = ((base + lo) * PAGE).max(start);
        let range_va_hi = ((base + hi) * PAGE).min(end);
        let ranged_all = Self::ranged_in(&copied.1, range_va_lo, range_va_hi);
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        let mut run_start: Option<usize> = None;
        {
            let tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            for idx in lo..hi {
                let slot = &mut copied.0[idx];
                let taken = (*slot).max(ranged_all);
                if taken == 0 {
                    *slot = stamp;
                    if !GpuOnlyTracker::has_bit(&tracker.gpu_dirty_bits, idx) {
                        run_start.get_or_insert(base + idx);
                    } else if let Some(first) = run_start.take() {
                        runs.push(((first * PAGE).max(start), ((base + idx) * PAGE).min(end), false));
                    }
                } else if let Some(first) = run_start.take() {
                    runs.push(((first * PAGE).max(start), ((base + idx) * PAGE).min(end), false));
                }
            }
        }
        if let Some(first) = run_start {
            let last_end = ((base + hi) * PAGE).min(end);
            runs.push(((first * PAGE).max(start), last_end, false));
        }
        if !runs.is_empty() {
            let _ = self.take_all(runs);
        }
    }

    /// Populates any untouched (`taken == 0`) non-GPU-dirty pages into `self.data`
    /// and marks all pages stamped in `self.copied` right before arming a Mode-1
    /// `PAGE_NOACCESS` hardware tripwire.
    fn seal_unfetched_pages_for_tripwire(&self, card: &Card) {
        let total = self.gpu_only_total_pieces.max(1);
        self.seal_unfetched_pages_range(card, 0, total);
    }

    /// Disarms every active Mode-1 or Mode-2 tripwire owned by this shadow and
    /// cleans up `TRIPWIRE_RUNS` + global stats (called on `unregister` / `Drop`).
    fn disarm_all_tripwires(&self) {
        self.tripwire_armed.store(false, Ordering::Release);
        let removed_ids = tripwire_runs_remove_shadow(self.id);
        for rid in removed_ids {
            crate::watch::tripwire_disarm_id(rid);
        }
        crate::watch::tripwire_disarm_id(self.id);
        let prev_state = self.gpu_only_state.swap(GPU_ONLY_DEMOTED, Ordering::AcqRel);
        if prev_state == GPU_ONLY_PROMOTED {
            GPU_ONLY_STATS[0].fetch_sub(1, Ordering::Relaxed);
            GPU_ONLY_STATS[1].fetch_sub(self.len, Ordering::Relaxed);
        }
        if let Some(card) = &self.card {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            let runs = std::mem::take(&mut tracker.active_runs);
            for (rid, lo, hi) in runs {
                crate::watch::tripwire_disarm_id(rid);
                for p in lo..hi {
                    GpuOnlyTracker::clear_bit(&mut tracker.piece_promoted, p);
                }
            }
            for w in &card.gpu_only_bitmap {
                w.store(0, Ordering::Release);
            }
            let prev_pieces = self.gpu_only_promoted_pieces.swap(0, Ordering::AcqRel);
            if prev_pieces > 0 && prev_state != GPU_ONLY_PROMOTED {
                GPU_ONLY_STATS[0].fetch_sub(1, Ordering::Relaxed);
                GPU_ONLY_STATS[1].fetch_sub((prev_pieces * PIECE) as u64, Ordering::Relaxed);
            }
        }
    }

    /// Mode-2 localized demotion of a single promoted run `(run_id, piece_lo..piece_hi)`:
    /// clears ONLY those bits in `gpu_only_bitmap`, re-commits their `CpuStaging`
    /// slices if decommitted, and marks only those pieces hash-stale — keeping all
    /// other promoted runs on this buffer promoted!
    fn demote_mode2_run(
        &self,
        run_id: u64,
        piece_lo: usize,
        piece_hi: usize,
        reason: &'static str,
        tripwire_fault: Option<(usize, usize, usize)>,
    ) {
        let Some(card) = &self.card else {
            return;
        };
        let mut cleared = 0usize;
        for p in piece_lo..piece_hi {
            if let Some(word) = card.gpu_only_bitmap.get(p / 64) {
                let bit = 1u64 << (p % 64);
                if word.fetch_and(!bit, Ordering::AcqRel) & bit != 0 {
                    cleared += 1;
                }
            }
        }
        let remaining = if cleared > 0 {
            let prev = self.gpu_only_promoted_pieces.fetch_sub(cleared, Ordering::AcqRel);
            prev.saturating_sub(cleared)
        } else {
            self.gpu_only_promoted_pieces.load(Ordering::Acquire)
        };
        let was_active = {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            tracker.demote_run_id(run_id, piece_lo, piece_hi)
        };
        if remaining == 0 {
            self.tripwire_armed.store(false, Ordering::Release);
        }
        if cleared > 0 || was_active {
            // Re-commit CpuStaging slices for the demoted run so subsequent
            // uploads/merges find committed memory immediately.
            if piece_lo < piece_hi {
                let (off_lo, _) = self.piece(piece_lo);
                let (_, off_hi) = self.piece(piece_hi - 1);
                let _ = self.ensure_staging_committed(off_lo, off_hi);
            }
            self.card_mark_piece_range_hash_stale(piece_lo, piece_hi);
            let demoted_bytes = (cleared * PIECE) as u64;
            GPU_ONLY_STATS[1].fetch_sub(demoted_bytes, Ordering::Relaxed);
            GPU_ONLY_STATS[2].fetch_add(1, Ordering::Relaxed);
            if remaining == 0 && cleared > 0 {
                GPU_ONLY_STATS[0].fetch_sub(1, Ordering::Relaxed);
            }
            if let Some((fault_addr, access, rip)) = tripwire_fault {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    pieces = format_args!("{piece_lo}..{piece_hi}"),
                    remaining_promoted = remaining,
                    fault = format_args!("{fault_addr:#x}"),
                    access = if access == 0 { "read" } else { "write" },
                    rip = format_args!("{rip:#x}"),
                    reason,
                    "gpu-only Mode-2 tripwire triggered: locally demoted faulted run while keeping disjoint runs promoted"
                );
            } else {
                tracing::debug!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    pieces = format_args!("{piece_lo}..{piece_hi}"),
                    remaining_promoted = remaining,
                    reason,
                    "gpu-only Mode-2 piece run locally demoted"
                );
            }
        }
    }

    /// Mode-2 localized demotion for any active promoted runs overlapping guest VA
    /// `[va_start, va_start + va_len)`.
    fn demote_mode2_overlap(&self, va_start: u64, va_len: u64, reason: &'static str) {
        let Some(card) = &self.card else {
            return;
        };
        let (from, to) = self.inside(va_start as usize, va_len as usize);
        if from >= to {
            return;
        }
        let pieces = self.pieces(from, to);
        let hits = {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            let hits = tracker.demote_overlapping_runs(pieces.start, pieces.end);
            // Also reset counters on non-promoted overlapped pieces so they don't promote.
            for p in pieces.start..pieces.end {
                if let Some(cnt) = tracker.piece_fetches.get_mut(p) {
                    *cnt = 0;
                }
            }
            hits
        };
        for (run_id, lo, hi) in hits {
            let _ = tripwire_run_take(run_id);
            crate::watch::tripwire_disarm_id(run_id);
            self.demote_mode2_run(run_id, lo, hi, reason, None);
        }
    }

    /// Permanently demotes this buffer out of GPU-only copyback elision and
    /// marks its `Card::before` hashes stale so the next compare is exact.
    pub(crate) fn demote_gpu_only(&self, reason: &'static str) {
        self.demote_gpu_only_inner(reason, None);
    }

    fn demote_gpu_only_inner(&self, reason: &'static str, tripwire_fault: Option<(usize, usize, usize)>) {
        self.tripwire_armed.store(false, Ordering::Release);
        let removed_ids = tripwire_runs_remove_shadow(self.id);
        for rid in removed_ids {
            crate::watch::tripwire_disarm_id(rid);
        }
        let prev = self.gpu_only_state.swap(GPU_ONLY_DEMOTED, Ordering::AcqRel);
        if prev == GPU_ONLY_PROMOTED {
            crate::watch::tripwire_disarm_id(self.id);
            let _ = self.ensure_staging_committed(0, self.len as usize);
            GPU_ONLY_STATS[0].fetch_sub(1, Ordering::Relaxed);
            GPU_ONLY_STATS[1].fetch_sub(self.len, Ordering::Relaxed);
            GPU_ONLY_STATS[2].fetch_add(1, Ordering::Relaxed);
            self.card_mark_all_hash_stale();
            if let Some((fault_addr, access, rip)) = tripwire_fault {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    fault = format_args!("{fault_addr:#x}"),
                    access = if access == 0 { "read" } else { "write" },
                    rip = format_args!("{rip:#x}"),
                    reason,
                    "gpu-only Mode-1 tripwire triggered: restored PAGE_READWRITE and demoted buffer back to normal copyback"
                );
            } else {
                tracing::info!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    reason,
                    "gpu-only card buffer demoted back to normal copyback (VICTORY_GPU_ONLY_COPYBACK)"
                );
            }
        }
        // Also demote any Mode-2 promoted runs if present.
        if let Some(card) = &self.card
            && self.any_piece_gpu_only()
        {
            let runs = {
                let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
                std::mem::take(&mut tracker.active_runs)
            };
            for (rid, lo, hi) in runs {
                crate::watch::tripwire_disarm_id(rid);
                self.demote_mode2_run(rid, lo, hi, reason, tripwire_fault);
            }
        }
    }

    /// Records CPU upload ranges (copy offsets) for `VICTORY_GPU_ONLY_COPYBACK`.
    fn gpu_only_note_upload(&self, ranges: &[(usize, usize)]) {
        let mode = gpu_only_mode();
        if mode == 0 || self.window || self.len < gpu_only_min_bytes() {
            return;
        }
        let Some(card) = &self.card else {
            return;
        };
        if mode == 2 {
            let mut hits = Vec::new();
            {
                let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
                for &(a, b) in ranges {
                    if a >= b {
                        continue;
                    }
                    let pieces = self.pieces(a, b);
                    hits.extend(tracker.note_upload_mode2(pieces.start, pieces.end));
                }
            }
            for (run_id, lo, hi) in hits {
                let _ = tripwire_run_take(run_id);
                crate::watch::tripwire_disarm_id(run_id);
                self.demote_mode2_run(run_id, lo, hi, "overlapping cpu upload", None);
            }
            return;
        }
        let state = self.gpu_only_state.load(Ordering::Acquire);
        if state == GPU_ONLY_DEMOTED {
            return;
        }
        let promoted = state == GPU_ONLY_PROMOTED;
        let frame = crate::frame::current();
        let demote = {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            let mut hit = false;
            for &(a, b) in ranges {
                if a >= b {
                    continue;
                }
                let pieces = self.pieces(a, b);
                if tracker.note_upload_at(mode, promoted, frame, pieces.start, pieces.end) {
                    hit = true;
                    break;
                }
            }
            hit
        };
        if demote {
            self.demote_gpu_only("cpu upload");
        }
    }

    /// Records GPU-fetched ranges `(lo, hi)` (copy offsets) for
    /// `VICTORY_GPU_ONLY_COPYBACK` and promotes eligible buffers/piece-runs.
    pub(crate) fn gpu_only_note_fetched(self: &Arc<Self>, ranges: &[(usize, usize)]) {
        let mode = gpu_only_mode();
        if mode == 0 || self.window || self.len < gpu_only_min_bytes() || ranges.is_empty() {
            return;
        }
        if mode == 2 {
            self.gpu_only_note_fetched_mode2(ranges, GPU_ONLY_MODE2_CONSECUTIVE, gpu_only_tripwire_on());
            return;
        }
        if self.gpu_only_state.load(Ordering::Acquire) != GPU_ONLY_OBSERVING {
            return;
        }
        let Some(card) = &self.card else {
            return;
        };
        let frame = crate::frame::current();
        let mut pieces: Vec<usize> = Vec::with_capacity(ranges.len());
        for &(lo, hi) in ranges {
            if lo < hi {
                pieces.extend(self.pieces(lo, hi));
            }
        }
        let (promote, demote) = {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            tracker.note_fetched(mode, frame, &pieces)
        };
        if demote {
            self.demote_gpu_only("overlapping cpu/gpu warmup");
            return;
        }
        if promote {
            if has_overlapping_shadow(self.start, self.len, self.id)
                || crate::inplace::overlaps_private_placed(self.start, self.len)
            {
                self.demote_gpu_only("overlapping host copy or inplace private heap");
                return;
            }
            // Hold `self.merging.write()` -> `self.before.lock()` (matching `land_back`
            // lock order) while transitioning to `PROMOTED` + arming `PAGE_NOACCESS`
            // so no concurrent `land_card` or `card_merge_relay` can be mid-`safecopy`.
            let _merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
            let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
            if self
                .gpu_only_state
                .compare_exchange(GPU_ONLY_OBSERVING, GPU_ONLY_PROMOTED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Preserve GPU-written bytes if the CPU ever writes another dword in the same page.
                self.keep_stores();
                // Seed dynamic `gpu_dirty_bits` from any currently in-flight declared ranges.
                if self.pending_count.load(Ordering::Acquire) != 0 {
                    let in_flight: Vec<(usize, usize)> =
                        self.pending.lock().unwrap_or_else(|e| e.into_inner()).clone();
                    let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
                    for (a, e) in in_flight {
                        let (from, to) = self.inside(a, e.saturating_sub(a));
                        if from < to {
                            let pieces = self.pieces(from, to);
                            tracker.note_declared_write(pieces.start, pieces.end);
                        }
                    }
                }
                // Snapshot any untouched (`copied == 0`), non-GPU-dirty pages ONCE
                // while `PAGE_READWRITE` is still active so post-promotion refreshes
                // never touch guest memory and self-trigger the `PAGE_NOACCESS` tripwire.
                if gpu_only_tripwire_on() {
                    self.seal_unfetched_pages_for_tripwire(card);
                }
                let tripwire_armed = if gpu_only_tripwire_on() {
                    crate::watch::tripwire_arm(self.id, self.start as usize, self.len as usize)
                } else {
                    false
                };
                if tripwire_armed {
                    let arm_start = (self.start as usize).next_multiple_of(KERNEL_PAGE);
                    let arm_end = ((self.start + self.len) as usize) & !(KERNEL_PAGE - 1);
                    if arm_start < arm_end {
                        tripwire_run_register(TripwireRun {
                            run_id: self.id,
                            start: arm_start,
                            end: arm_end,
                            piece_lo: 0,
                            piece_hi: self.gpu_only_total_pieces,
                            shadow_id: self.id,
                            shadow: Arc::downgrade(self),
                            mode2: false,
                        });
                    }
                    self.tripwire_armed.store(true, Ordering::Release);
                }
                // Reclaim pure CPU staging RAM for promoted GPU-only buffer.
                if card.dirty.lock().unwrap_or_else(|e| e.into_inner()).is_empty() {
                    let _ = self.decommit_staging_pieces(0, self.gpu_only_total_pieces);
                }
                drop(_side);
                drop(_merging);
                GPU_ONLY_STATS[0].fetch_add(1, Ordering::Relaxed);
                GPU_ONLY_STATS[1].fetch_add(self.len, Ordering::Relaxed);
                tracing::info!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    mode,
                    tripwire = tripwire_armed,
                    "gpu-only card buffer promoted: skipping compare/fetch/merge copyback (VICTORY_GPU_ONLY_COPYBACK)"
                );
            }
        }
    }

    /// Piece-level Mode-2 promotion path: promotes contiguous runs of `>= 4`
    /// 64 KiB pieces (`>= 256 KiB`) with `>= min_consecutive` consecutive GPU
    /// writebacks and zero CPU reads/writes by arming a per-run `PAGE_NOACCESS`
    /// tripwire and decommitting their `CpuStaging` slices.
    fn gpu_only_note_fetched_mode2(
        self: &Arc<Self>,
        ranges: &[(usize, usize)],
        min_consecutive: u16,
        arm_tripwire: bool,
    ) {
        let Some(card) = &self.card else {
            return;
        };
        let total_pieces = self.gpu_only_total_pieces;
        if total_pieces < GPU_ONLY_MODE2_MIN_RUN_PIECES {
            return;
        }
        let mut pieces: Vec<usize> = Vec::with_capacity(ranges.len());
        for &(lo, hi) in ranges {
            if lo < hi {
                pieces.extend(self.pieces(lo, hi));
            }
        }
        pieces.sort_unstable();
        pieces.dedup();
        let candidates = {
            let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
            tracker.note_fetched_mode2(&pieces, total_pieces, min_consecutive, GPU_ONLY_MODE2_MIN_RUN_PIECES)
        };
        if candidates.is_empty() {
            return;
        }
        for (piece_lo, piece_hi) in candidates {
            let (off_lo, _) = self.piece(piece_lo);
            let (_, off_hi) = self.piece(piece_hi - 1);
            if off_lo >= off_hi {
                continue;
            }
            let run_va_start = self.start + off_lo as u64;
            let run_va_len = (off_hi - off_lo) as u64;
            // Spec-Trap #8 & aliasing guard: NEVER arm a Mode-2 tripwire on pages
            // overlapping another shadow or an active `OpenExistingHeapFromAddress1` heap!
            if has_overlapping_shadow(run_va_start, run_va_len, self.id)
                || crate::inplace::overlaps_private_placed(run_va_start, run_va_len)
            {
                card.gpu_only
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .mark_pieces_demoted(piece_lo, piece_hi);
                continue;
            }
            // Hold `self.merging.write()` -> `self.before.lock()` across sealing + arming
            // so no concurrent `land_card` / `merge_under_writes` touches this run mid-arm.
            let _merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
            let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
            // Re-check under lock that none of the pieces were concurrently uploaded or demoted.
            let still_eligible = {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let dirty_overlap = dirty.iter().any(|&(a, b)| a < off_hi && b > off_lo);
                if dirty_overlap {
                    false
                } else {
                    let tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
                    (piece_lo..piece_hi).all(|p| {
                        !GpuOnlyTracker::has_bit(&tracker.piece_promoted, p)
                            && !GpuOnlyTracker::has_bit(&tracker.piece_demoted, p)
                    })
                }
            };
            if !still_eligible {
                continue;
            }
            // Seal untouched non-GPU-dirty pages in `[piece_lo, piece_hi)` BEFORE arming `PAGE_NOACCESS`.
            if arm_tripwire {
                self.seal_unfetched_pages_range(card, piece_lo, piece_hi);
            }
            let run_id = NEXT_TRIPWIRE_RUN_ID.fetch_add(1, Ordering::Relaxed);
            let armed = if arm_tripwire {
                crate::watch::tripwire_arm(run_id, run_va_start as usize, run_va_len as usize)
            } else {
                true
            };
            if !armed {
                continue;
            }
            if arm_tripwire {
                let arm_start = (run_va_start as usize).next_multiple_of(KERNEL_PAGE);
                let arm_end = ((run_va_start + run_va_len) as usize) & !(KERNEL_PAGE - 1);
                if arm_start < arm_end {
                    tripwire_run_register(TripwireRun {
                        run_id,
                        start: arm_start,
                        end: arm_end,
                        piece_lo,
                        piece_hi,
                        shadow_id: self.id,
                        shadow: Arc::downgrade(self),
                        mode2: true,
                    });
                }
                self.tripwire_armed.store(true, Ordering::Release);
            }
            self.keep_stores();
            for p in piece_lo..piece_hi {
                if let Some(word) = card.gpu_only_bitmap.get(p / 64) {
                    word.fetch_or(1u64 << (p % 64), Ordering::Release);
                }
            }
            let count = piece_hi - piece_lo;
            let prev_promoted = self.gpu_only_promoted_pieces.fetch_add(count, Ordering::AcqRel);
            {
                let mut tracker = card.gpu_only.lock().unwrap_or_else(|e| e.into_inner());
                tracker.note_run_promoted(run_id, piece_lo, piece_hi);
            }
            // Reclaim pure CPU staging RAM for the promoted piece run.
            let _ = self.decommit_staging_pieces(piece_lo, piece_hi);
            drop(_side);
            drop(_merging);
            if prev_promoted == 0 {
                GPU_ONLY_STATS[0].fetch_add(1, Ordering::Relaxed);
            }
            GPU_ONLY_STATS[1].fetch_add((count * PIECE) as u64, Ordering::Relaxed);
            tracing::info!(
                start = format_args!("{:#x}", self.start),
                len = self.len,
                pieces = format_args!("{piece_lo}..{piece_hi}"),
                run_kb = run_va_len >> 10,
                tripwire = arm_tripwire,
                "gpu-only Mode-2 piece run promoted: skipping compare/fetch/merge copyback"
            );
        }
    }

    /// Marks the 64 KiB pieces touched by `ranges` (copy offsets) as having
    /// stale hashes in `Card::before` so they are lazily re-hashed before a
    /// submission next compares them.
    pub(crate) fn card_mark_hash_stale(&self, ranges: &[(usize, usize)]) {
        let Some(card) = &self.card else {
            return;
        };
        if card.before.get().is_none() {
            return;
        }
        let mut stale = card.hash_stale.lock().unwrap_or_else(|e| e.into_inner());
        if stale.is_empty() {
            return;
        }
        for &(a, b) in ranges {
            if a >= b {
                continue;
            }
            let span = self.card_run(a, b);
            mark_stale_bits(&mut stale, span.first as usize, (span.first + span.pieces) as usize);
        }
    }

    /// Takes and clears the stale piece bits intersecting `spans`, returning
    /// coalesced `CardSpan` runs to re-hash before GPU writes run.
    pub(crate) fn card_take_stale_runs(&self, spans: &[CardSpan]) -> Vec<CardSpan> {
        let Some(card) = &self.card else {
            return Vec::new();
        };
        let mut stale = card.hash_stale.lock().unwrap_or_else(|e| e.into_inner());
        if stale.is_empty() {
            return Vec::new();
        }
        let mut runs = Vec::new();
        for &span in spans {
            take_stale_span(&mut stale, span, &mut runs);
        }
        if runs.len() > 1 {
            joined_piece_runs(runs)
        } else {
            runs
        }
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy or copyback is elided
    /// for a promoted GPU-only buffer/piece-run (`ARCH-06`).
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        if self.gpu_only_skip() {
            return None;
        }
        let (mut from, mut to) = self.inside(address, end.saturating_sub(address));
        if from >= to {
            return None;
        }
        if self.any_piece_gpu_only() {
            // Trim leading and trailing Mode-2 promoted pieces; if every piece is
            // promoted, skip the compare dispatch completely.
            let range = self.pieces(from, to);
            let mut p_lo = range.start;
            let mut p_hi = range.end;
            while p_lo < p_hi && self.piece_is_gpu_only(p_lo) {
                p_lo += 1;
            }
            while p_hi > p_lo && self.piece_is_gpu_only(p_hi - 1) {
                p_hi -= 1;
            }
            if p_lo >= p_hi {
                return None;
            }
            if p_lo > range.start {
                from = from.max(self.piece(p_lo).0);
            }
            if p_hi < range.end {
                to = to.min(self.piece(p_hi - 1).1);
            }
            if from >= to {
                return None;
            }
        }
        Some(card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The piece that holds offset `at`.
    pub(crate) fn card_piece_of(&self, at: usize) -> u32 {
        self.pieces(at, at + 1).start as u32
    }

    /// The offsets the compare on the card hashes of a piece: what lies of
    /// it in the card's buffer (which may be longer than the copy), widened
    /// to 16 bytes.
    pub(crate) fn card_hashed(&self, piece: u32) -> (usize, usize) {
        let size = self.card_hash_size() as usize;
        card_hashed_of(self.start as usize, size, piece as usize)
    }

    /// The bytes of the copy the card's hashes cover: the title buffer's own,
    /// rounded up to the hash's 16-byte step. The card's buffer is the title
    /// buffer rounded up to 64 KiB, of which a title buffer of 288 bytes can
    /// give nothing back but those (after fxceless c36971b2).
    /// `VICTORY_CARD_HASH_LEN=0`: the whole buffer.
    pub(crate) fn card_hash_size(&self) -> u64 {
        static OWN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let size = self.card.as_ref().map_or(self.len, |c| c.size);
        if *OWN.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "0")) {
            self.len.next_multiple_of(16).min(size)
        } else {
            size
        }
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        if self.gpu_only_skip() || self.range_all_gpu_only(at, at + gpu.len()) {
            return 0;
        }
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        if !self.ensure_staging_committed(at, at + gpu.len()) {
            return 0;
        }
        // SAFETY: inside the mapping (checked & committed above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        let mut card_differs = 0u8;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            let was = title.clone();
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                card_differs |= merged ^ *g;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                written = put_changed(start + at, &was, title);
                if written > 0 {
                    // Prevent `take_landed` from echoing our own landing back up (`CPU-01`).
                    self.note_landing(at, at + n);
                }
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        if card_differs != 0 {
            // Title CPU modified bytes beside GPU writes: upload merged bytes to card.
            self.card_changed(at, at + n);
        } else {
            // Card already holds `gpu`; only re-hash `Card::before` without 2x self-echo (`CPU-01`).
            self.card_mark_hash_stale(&[(at, at + n)]);
        }
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        if self.gpu_only_skip() || self.range_all_gpu_only(at, at + gpu.len()) {
            return 0;
        }
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        if self.gpu_only_skip() || self.range_all_gpu_only(at, at + gpu.len()) {
            return 0;
        }
        if !self.ensure_staging_committed(at, at + gpu.len()) {
            return 0;
        }
        // SAFETY: inside the mapping (checked & committed above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < at + n && b > at).collect();
            sent.near(fetched_after, at, at + n, &mut skip);
            skip
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        let was = title.clone();
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        put_changed(start + at, &was, title)
    }
}

/// For tests: copies with no host buffer behind them.
#[cfg(test)]
impl Shadow {
    /// A copy at `data` of title memory `[start, start + len)`, with a card
    /// side if `card`. It is never let go: there is no buffer to release.
    pub(crate) fn unbacked(start: u64, data: *mut u8, len: u64, card: bool) -> Arc<Shadow> {
        // SAFETY: an interface pointer nothing calls.
        let none = || unsafe { std::mem::transmute::<usize, ID3D12Resource>(8) };
        let mut shadow = Shadow::new(none(), data, 0, start, len);
        if card {
            shadow.card = Some(Card::new(CardStaging::Gpu(none()), len.next_multiple_of(PIECE as u64)));
        }
        let shadow = Arc::new(shadow);
        std::mem::forget(shadow.clone());
        shadow
    }

    /// What shaders wrote goes into title memory ([`put_changed`]).
    pub(crate) fn put(at: usize, title: &[u8], merged: &[u8]) {
        put_changed(at, title, merged);
    }

    /// What is to go up to the card, noted as sent with `number`, as
    /// [`card_dirty`] does; notes of sends up to `merged` go.
    pub(crate) fn send_up(&self, number: u64, merged: u64) -> Vec<(usize, usize)> {
        let card = self.card.as_ref().unwrap();
        let mut dirty = card.dirty.lock().unwrap();
        let ranges = std::mem::take(&mut *dirty);
        let mut sent = card.sent.lock().unwrap();
        sent.forget(merged);
        sent.note(number, ranges.clone());
        ranges
    }

    /// What is to go up to the card, taken.
    pub(crate) fn take_dirty(&self) -> Vec<(usize, usize)> {
        std::mem::take(&mut *self.card.as_ref().unwrap().dirty.lock().unwrap())
    }
}

impl Shadow {
    /// [`Shadow::card_merge_relay`] of several pieces, each within one
    /// 64 KiB block of title memory: from and to (offsets in the copy), and
    /// where the fetched bytes are. On any thread: several may land pieces
    /// of one copy, no two the same bytes. Returns the bytes put into title
    /// memory.
    ///
    /// What is left out is as there, asked once for all of them: the notes
    /// of bytes that went up (or are to go up) behind the fetch only move
    /// from `dirty` to `sent`, with a later number, and grow only by
    /// [`Shadow::merge_under_writes`], which the lock held here keeps out.
    ///
    /// # Safety
    /// The third of each piece is the address of its bytes, which stay as
    /// they are during the call.
    pub(crate) unsafe fn land_card(&self, pieces: &[(usize, usize, usize)], fetched_after: u64, notes: &mut crate::land::Notes) -> u64 {
        if self.gpu_only_skip() {
            return 0;
        }
        let Some(card) = &self.card else {
            return 0;
        };
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        if self.gpu_only_skip() {
            return 0;
        }
        let (from, to) = pieces.iter().fold((usize::MAX, 0), |(a, b), p| (a.min(p.0), b.max(p.1)));
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < to && b > from).collect();
            sent.near(fetched_after, from, to, &mut skip);
            skip
        };
        let mut put = 0;
        let has_mode2 = self.any_piece_gpu_only();
        for &(lo, hi, now) in pieces {
            if lo >= hi || hi > self.len as usize {
                continue;
            }
            if has_mode2 && self.range_all_gpu_only(lo, hi) {
                continue;
            }
            if !self.ensure_staging_committed(lo, hi) {
                continue;
            }
            let near: Vec<(usize, usize)> = skip.iter().copied().filter(|&(a, b)| a < hi && b > lo).collect();
            let parts: Vec<(usize, usize)> = outside(lo, hi, near).into_iter().map(|(a, b)| (a - lo, b - lo)).collect();
            // SAFETY: inside the mapping (checked above), in a range
            // declared and not done with, which no refresh writes; the
            // title's bytes are kept out by the lock; the caller's.
            // Where title memory is not mapped the CPU's side keeps what
            // the card wrote, as before.
            let merged =
                unsafe { crate::land::piece(self.start as usize + lo, now as *const u8, self.data.add(lo), &parts, true, notes, self) };
            self.walls_merged(hi - lo, merged as usize);
            if merged > 0 {
                self.note_landing(lo, hi);
            }
            put += merged;
        }
        put
    }

    /// [`Shadow::relay_merge`] of a piece `[at, at + n)` (within one 64 KiB
    /// block of title memory) of a copy in the PC's memory; on any thread,
    /// as [`Shadow::land_card`]. The copy is read once and never written.
    /// Returns the bytes put into title memory.
    pub(crate) fn land_back(&self, at: usize, n: usize, notes: &mut crate::land::Notes) -> u64 {
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        // The snapshot's memory stays where it is until the range is done
        // with (`copy_back_done`), which comes after this.
        let Some(held) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(Vec::as_mut_ptr) else {
            return 0;
        };
        // SAFETY: inside the copy's mapping (checked above), which only the
        // card writes there, and inside the snapshot's allocation, which
        // holds the copy's length (`writing`): these offsets are in a range
        // declared and not done with, whose bytes it holds, and the lock
        // keeps out what else writes them. Title memory that is not there
        // is looked at again when the next flight that declared the range
        // lands.
        unsafe { crate::land::piece(self.start as usize + at, self.data.add(at), held.add(at), &[(0, n)], false, notes, self) }
    }

    /// What shaders wrote in 64 KiB block `block` (of title memory) of this
    /// copy was put into title memory beside the guard, which was noted as a
    /// write with number `now`, `before` the number of the block's last write
    /// before. If the copy held title memory's bytes of the block ahead of
    /// that it holds them still, and is noted to ([`crate::land`] says why);
    /// whether it was.
    ///
    /// That it did: it took the whole block after that write, and no other
    /// came that is not noted yet. Looked at after the note, whatever comes
    /// later has a number above it: a write of the title's takes the block's
    /// bit, until an asking has noted it.
    pub(crate) fn landed(&self, block: usize, before: u64, now: u64) -> bool {
        let Some(page) = block.checked_sub(self.start as usize / PAGE) else {
            return false;
        };
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let Some(at) = copied.0.get_mut(page) else {
            return false;
        };
        let held = *at != 0 && *at >= before && crate::watch::block_published(block);
        if held {
            *at = (*at).max(now);
        }
        held
    }
}

#[cfg(test)]
mod land_tests {
    use super::{Card, CardStaging, PIECE, Shadow};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// A copy in the card's memory of title memory at `title` (its card side
    /// is never looked at here), all bytes 0 on the CPU's side.
    fn copy_of(title: usize, len: usize) -> Option<Shadow> {
        let (buffer, data) = crate::host::host()?.shared_buffer(len as u64)?;
        // SAFETY: the mapping, `len` bytes.
        unsafe { std::ptr::write_bytes(data, 0, len) };
        let mut shadow = Shadow::new(buffer.clone(), data, 0, title as u64, len as u64);
        shadow.card = Some(Card::new(CardStaging::Gpu(buffer), len as u64));
        Some(shadow)
    }

    /// Memory that stays (`crate::watch` knows its addresses): private, or a
    /// view noted as the runtime notes the title's.
    fn memory(len: usize, view: bool) -> usize {
        use windows_sys::Win32::System::Memory as mem;
        // SAFETY: plain allocations of our own.
        unsafe {
            if !view {
                return mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize;
            }
            let section = mem::CreateFileMappingW(-1isize as _, std::ptr::null(), mem::PAGE_READWRITE, 0, len as u32, std::ptr::null());
            let at = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, len).Value as usize;
            crate::inplace::title_memory_section(section as usize, len as u64, true);
            crate::title_memory_remapped(at, len);
            crate::inplace::title_memory_view(at, len, section as usize, 0);
            at
        }
    }

    fn random(seed: &mut u64) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 8) as usize
    }

    /// [`Shadow::land_card`] leaves title memory and the CPU's side as
    /// [`Shadow::card_merge_relay`] does, with title bytes that went up
    /// behind the fetch, or have not gone up, left out.
    #[test]
    fn pieces_land_as_the_merge_one_by_one_lands_them() {
        assert!(crate::safecopy::on());
        const LEN: usize = 3 * PIECE;
        let mut seed = 0x1234_5678_9abcu64;
        for view in [false, true] {
            let (a, b) = (memory(LEN, view), memory(LEN, view));
            let (Some(one), Some(many)) = (copy_of(a, LEN), copy_of(b, LEN)) else {
                return; // no host GPU
            };
            // SAFETY: the two memories and the two mappings.
            let bytes = |at: usize| unsafe { std::slice::from_raw_parts_mut(at as *mut u8, LEN) };
            for round in 0..40u32 {
                let base: Vec<u8> = (0..LEN).map(|_| random(&mut seed) as u8).collect();
                let (mut held, mut title, mut now) = (base.clone(), base.clone(), base);
                let (mut dirty, mut sent) = (Vec::new(), Vec::new());
                for k in 0..120 {
                    let at = random(&mut seed) % LEN / 4 * 4;
                    let n = (random(&mut seed) % 900 / 4 * 4).min(LEN - at);
                    match k % 4 {
                        0 | 1 => now[at..at + n].fill(random(&mut seed) as u8),
                        2 => title[at..at + n].fill(random(&mut seed) as u8),
                        // Title bytes put into the CPU's side meanwhile.
                        _ => {
                            let byte = random(&mut seed) as u8;
                            held[at..at + n].fill(byte);
                            title[at..at + n].fill(byte);
                            if k % 8 == 3 {
                                dirty.push((at, at + n));
                            } else {
                                sent.push((at, at + n, u64::from(round % 3) + (k as u64 / 16)));
                            }
                        }
                    }
                }
                let after = 4u64;
                for (shadow, at) in [(&one, a), (&many, b)] {
                    crate::watch::writing(at, LEN);
                    bytes(at).copy_from_slice(&title);
                    bytes(shadow.data as usize).copy_from_slice(&held);
                    let card = shadow.card.as_ref().expect("made with one");
                    *card.dirty.lock().unwrap() = dirty.clone();
                    *card.sent.lock().unwrap() = super::SentRuns::of(&sent);
                }
                let pieces: Vec<(usize, usize)> = (0..3).map(|p| (p * PIECE + 16, (p + 1) * PIECE - 4)).collect();
                let mut scratch = Vec::new();
                for &(lo, hi) in &pieces {
                    one.card_merge_relay(lo, &now[lo..hi], after, &mut scratch);
                }
                let addressed: Vec<(usize, usize, usize)> =
                    pieces.iter().map(|&(lo, hi)| (lo, hi, now.as_ptr() as usize + lo)).collect();
                let mut notes = crate::land::Notes::default();
                // SAFETY: the fetched bytes are `now`'s.
                unsafe { many.land_card(&addressed, after, &mut notes) };
                notes.flush();
                assert!(bytes(a) == bytes(b), "title memory, round {round}");
                assert!(bytes(one.data as usize) == bytes(many.data as usize), "the CPU's side, round {round}");
            }
        }
    }

    /// Landings of one copy on several threads, the title writing dwords of
    /// the same blocks, and the title's bytes put into the CPU's side under
    /// the shaders' writes ([`Shadow::merge_under_writes`]), all at once. The
    /// card has never received the title's bytes: what is fetched holds older
    /// ones there, which must not be taken for shaders' writes.
    #[test]
    fn landings_and_title_bytes_going_in_keep_each_others_bytes() {
        assert!(crate::safecopy::on());
        const BLOCKS: usize = 16;
        const LEN: usize = BLOCKS * PIECE;
        let title = memory(LEN, true);
        let Some(shadow) = copy_of(title, LEN) else {
            return; // no host GPU
        };
        shadow.pending.lock().unwrap().push((title, title + LEN));
        shadow.pending_count.store(1, Ordering::Release);
        // SAFETY: committed, aligned; touched as atomics or by the landing's
        // stores of whole dwords.
        let dword = |i: usize| unsafe { &*((title + i * 4) as *const AtomicU32) };
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                let mut value = 1u32;
                while !stop.load(Ordering::Relaxed) {
                    for i in (1..LEN / 4).step_by(4 * 331) {
                        assert_eq!(dword(i).swap(value, Ordering::Relaxed), value - 1, "dword {i}: undone");
                    }
                    value += 1;
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                value - 1
            });
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let mut runs = usize::MAX;
                    assert!(!shadow.merge_under_writes(title, LEN, &mut runs));
                    // Sent up, as the preparing of a submission would: noted
                    // with a number behind the fetches'.
                    let card = shadow.card.as_ref().expect("made with one");
                    let mut dirty = card.dirty.lock().unwrap();
                    let mut sent = card.sent.lock().unwrap();
                    sent.note(9, std::mem::take(&mut *dirty));
                }
            });
            let mut now = vec![0u8; LEN];
            for round in 1..200u32 {
                for block in (0..BLOCKS).filter(|b| (*b as u32 * 5 + round) % 3 != 0) {
                    for i in (block * PIECE / 4 + 3..(block + 1) * PIECE / 4).step_by(4) {
                        now[i * 4..i * 4 + 4].copy_from_slice(&(round << 20 | i as u32).to_le_bytes());
                    }
                }
                let at = now.as_ptr() as usize;
                crate::land::each(BLOCKS, &|b| {
                    let mut notes = crate::land::Notes::default();
                    let (lo, hi) = (b * PIECE, (b + 1) * PIECE);
                    // SAFETY: the fetched bytes are `now`'s; each block once.
                    unsafe { shadow.land_card(&[(lo, hi, at + lo)], 8, &mut notes) };
                    notes.flush();
                });
                for i in (3..LEN / 4).step_by(4 * 53) {
                    let expect = u32::from_le_bytes(now[i * 4..i * 4 + 4].try_into().expect("four"));
                    assert_eq!(dword(i).load(Ordering::Relaxed), expect, "round {round} dword {i}");
                }
            }
            stop.store(true, Ordering::Relaxed);
            let last = writer.join().expect("nothing undone");
            assert!((1..LEN / 4).step_by(4 * 331).all(|i| dword(i).load(Ordering::Relaxed) == last));
        });
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
///
/// Read by every view a draw names, on all recording threads at once, and
/// written when a buffer comes, goes or gets its copy: readers share it.
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

fn buffers_mut() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// Monotonic generation bumped (with `Release` while holding the mutating table's
/// write lock) on every insert, remove, eviction, or shadow replacement in
/// [`BUFFERS`] or [`LOOSE`], invalidating thread-local lookup caches.
static SHADOW_EPOCH: AtomicU64 = AtomicU64::new(1);

#[inline]
fn bump_shadow_epoch() {
    SHADOW_EPOCH.fetch_add(1, Ordering::Release);
}

/// Number of MRU spans kept per thread in [`SPANS`].
const SPAN_CACHE_CAP: usize = 32;

enum SpanRef {
    /// Persistent buffer shadow from [`BUFFERS`] (cleared when [`SHADOW_EPOCH`] advances).
    Buffer(Arc<Shadow>),
    /// Loose window shadow from [`LOOSE`] (`Weak` so [`let_go_idle`]'s `strong_count == 1` stays exact).
    Loose(std::sync::Weak<Shadow>),
}

struct SpanEntry {
    start: u64,
    end: u64,
    shadow: SpanRef,
}

struct SpanCache {
    epoch: u64,
    entries: Vec<SpanEntry>,
}

impl SpanCache {
    const fn new() -> Self {
        Self {
            epoch: 0,
            entries: Vec::new(),
        }
    }

    #[inline]
    fn sync(&mut self, epoch: u64) {
        if self.epoch != epoch {
            self.entries.clear();
            self.epoch = epoch;
        }
    }

    fn insert(&mut self, epoch: u64, start: u64, end: u64, shadow: SpanRef) {
        self.sync(epoch);
        if let Some(pos) = self.entries.iter().position(|e| e.start == start && e.end == end) {
            self.entries[pos].shadow = shadow;
            if pos > 0 {
                self.entries[..=pos].rotate_right(1);
            }
            return;
        }
        if self.entries.len() >= SPAN_CACHE_CAP {
            self.entries.pop();
        }
        self.entries.insert(0, SpanEntry { start, end, shadow });
    }
}

/// Answers a thread remembers (covers ~6k live buffers without conflict thrashing).
const FOUND_MOST: usize = 8192;

/// An ask (address, bytes), the epoch it was answered at, and the answer.
type Found = (u64, u64, u64, std::sync::Weak<Shadow>);

thread_local! {
    /// Per-thread MRU span cache over non-overlapping sub-spans of [`BUFFERS`]
    /// and buffer-free single-cell [`LOOSE`] windows: ring-buffer CBV, VB/IB,
    /// and indirect-arg offsets inside the same persistent buffer all hit here
    /// with zero global `RwLock` contention.
    static SPANS: std::cell::RefCell<SpanCache> = const { std::cell::RefCell::new(SpanCache::new()) };

    /// The last answers of [`lookup`] out of [`BUFFERS`], by a hash of the
    /// exact ask `(address, len)` for working sets wider than [`SPAN_CACHE_CAP`].
    static FOUND: std::cell::RefCell<Vec<Found>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether [`SPANS`] and [`FOUND`] are kept (`VICTORY_QUICK_FIND=0`: not).
fn quick_find() -> bool {
    crate::toggled!("VICTORY_QUICK_FIND")
}

#[inline]
fn note_loose_hit(shadow: &Shadow, address: u64, end: u64, unsized_ask: bool) {
    // (Read first: the cache line is shared by all recording threads.)
    let now = crate::host::moment();
    if shadow.used.load(Ordering::Relaxed) != now {
        shadow.used.store(now, Ordering::Relaxed);
    }
    let frame = crate::frame::current();
    if shadow.touch_frame.load(Ordering::Relaxed) != frame
        && shadow.touch_frame.swap(frame, Ordering::Relaxed) != frame
    {
        shadow.touch_frames.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(feature = "diagnostics")]
    {
        if shadow.asked[0].load(Ordering::Relaxed) > address {
            shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        }
        if shadow.asked[1].load(Ordering::Relaxed) < end {
            shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        }
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (address, end, unsized_ask);
}

fn span_find(address: u64, end: u64, unsized_ask: bool) -> Option<Arc<Shadow>> {
    let epoch = SHADOW_EPOCH.load(Ordering::Acquire);
    SPANS.with(|cell| {
        if let Ok(mut cache) = cell.try_borrow_mut() {
            cache.sync(epoch);
            if let Some(idx) = cache.entries.iter().position(|e| address >= e.start && end <= e.end) {
                if idx > 0 {
                    cache.entries[..=idx].rotate_right(1);
                }
                match &cache.entries[0].shadow {
                    SpanRef::Buffer(shadow) => {
                        if !shadow.is_dead() {
                            return Some(shadow.clone());
                        }
                        cache.entries.remove(0);
                    }
                    SpanRef::Loose(weak) => {
                        if let Some(shadow) = weak.upgrade()
                            && !shadow.is_dead()
                        {
                            note_loose_hit(&shadow, address, end, unsized_ask);
                            return Some(shadow);
                        }
                        cache.entries.remove(0);
                    }
                }
            }
            return None;
        }
        // Re-entrant fallback: read-only scan while an outer `with_in_place` borrows `SPANS`.
        let cache = cell.try_borrow().ok()?;
        if cache.epoch != epoch {
            return None;
        }
        let entry = cache.entries.iter().find(|e| address >= e.start && end <= e.end)?;
        match &entry.shadow {
            SpanRef::Buffer(shadow) if !shadow.is_dead() => Some(shadow.clone()),
            SpanRef::Loose(weak) => {
                let shadow = weak.upgrade()?;
                if shadow.is_dead() {
                    return None;
                }
                note_loose_hit(&shadow, address, end, unsized_ask);
                Some(shadow)
            }
            _ => None,
        }
    })
}

fn span_remember(epoch: u64, start: u64, end: u64, shadow: SpanRef) {
    if SHADOW_EPOCH.load(Ordering::Acquire) != epoch {
        return;
    }
    SPANS.with(|cell| {
        if let Ok(mut cache) = cell.try_borrow_mut() {
            cache.insert(epoch, start, end, shadow);
        }
    });
}

fn found_at(address: u64, len: u64) -> usize {
    let h = (address ^ len.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 51) as usize & (FOUND_MOST - 1)
}

/// What this thread remembers of an exact ask, if [`SHADOW_EPOCH`] did not advance.
fn found(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let epoch = SHADOW_EPOCH.load(Ordering::Acquire);
    let shadow = FOUND.with(|f| {
        let f = f.borrow();
        let (a, n, at, shadow) = f.get(found_at(address, len))?;
        ((*a, *n, *at) == (address, len, epoch)).then(|| shadow.upgrade()).flatten()
    })?;
    if shadow.is_dead() {
        return None;
    }
    (SHADOW_EPOCH.load(Ordering::Acquire) == epoch).then_some(shadow)
}

/// Remembers an answer given while `epoch` was active.
fn remember(address: u64, len: u64, epoch: u64, shadow: &Arc<Shadow>) {
    if SHADOW_EPOCH.load(Ordering::Acquire) != epoch {
        return;
    }
    FOUND.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_empty() {
            f.resize_with(FOUND_MOST, || (0, 0, 0, std::sync::Weak::new()));
        }
        f[found_at(address, len)] = (address, len, epoch, Arc::downgrade(shadow));
    });
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    let end = start + len;
    let mut map = buffers_mut();
    map.insert(
        (start, resource),
        Entry {
            end,
            shadow: None,
        },
    );
    bump_shadow_epoch();
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = {
        let mut map = buffers_mut();
        if let Some(entry) = map.remove(&(start, resource)) {
            if let Some(shadow) = &entry.shadow {
                shadow.dead.store(true, Ordering::Release);
            }
            bump_shadow_epoch();
            entry.shadow
        } else {
            None
        }
    };
    if let Some(shadow) = removed {
        shadow.disarm_all_tripwires();
        if crate::dxr::enabled() {
            crate::dxr::evict_range(shadow.start, shadow.len);
        }
    }
}

/// Invalidates `copied` / `ranged_seq` on non-placed shadows overlapping
/// `[start, start + len)` when a `port 1` in-place shadow is created, marks
/// `in_place_overlap` so future in-place GPU writes notify `watch::wrote`, and
/// notifies descriptor textures (`ARCH-06 #4`).
fn on_placed_shadow_created(placed: &Shadow) {
    let (start, len, self_id) = (placed.start, placed.len, placed.id);
    let Some(end) = start.checked_add(len).filter(|&e| e > start) else {
        return;
    };
    demote_overlapping_gpu_only(start, len, self_id, "overlapping in-place shadow");
    crate::descriptor::host::remapped(start as usize, len as usize);
    let mut overlapping: Vec<Arc<Shadow>> = Vec::new();
    {
        let map = buffers_read();
        for (_, entry) in map.range(..=(start, usize::MAX)).rev().take(64) {
            if entry.end > start
                && let Some(s) = &entry.shadow
                && s.id != self_id
                && s.place.is_none()
            {
                overlapping.push(s.clone());
            }
        }
        for (_, entry) in map.range((start.saturating_add(1), 0)..(end, 0)) {
            if let Some(s) = &entry.shadow
                && s.id != self_id
                && s.place.is_none()
            {
                overlapping.push(s.clone());
            }
        }
    }
    {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let span = LOOSE_MAX_LEN.load(Ordering::Relaxed);
        for (&(w_start, w_len), entry) in loose.range((start.saturating_sub(span), 0)..(end, 0)) {
            if w_start.saturating_add(w_len) > start
                && let Some(s) = entry.as_ready()
                && s.id != self_id
                && s.place.is_none()
            {
                overlapping.push(s.clone());
            }
        }
    }
    {
        let exact = EXACT.lock().unwrap_or_else(|e| e.into_inner());
        if !exact.1.is_empty() {
            const EXACT_MOST: u64 = 64 << 20;
            for (&(ex_start, ex_len), (s, _)) in exact.1.range((start.saturating_sub(EXACT_MOST), 0)..(end, 0)) {
                if ex_start.saturating_add(ex_len) > start && s.id != self_id && s.place.is_none() {
                    overlapping.push(s.clone());
                }
            }
        }
    }
    if !overlapping.is_empty() {
        placed.in_place_overlap.store(true, Ordering::Relaxed);
        for s in overlapping {
            let mut copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.0.fill(0);
            copied.1 = [(0, 0, 0); 2];
            s.ranged_seq.store(0, Ordering::Release);
        }
    }
}

/// If a newly created non-placed shadow overlaps an existing `port 1` in-place
/// region, mark overlapping placed shadows so their future GPU writes notify `watch::wrote`.
fn note_shadow_overlapping_in_place(start: u64, len: u64, self_id: u64) {
    if !crate::port::overlaps(start, len) {
        return;
    }
    let Some(end) = start.checked_add(len) else {
        return;
    };
    {
        let map = buffers_read();
        for (_, entry) in map.range(..(end, 0)).rev().take(256) {
            if entry.end > start
                && let Some(s) = &entry.shadow
                && s.id != self_id
                && s.place.is_some()
            {
                s.in_place_overlap.store(true, Ordering::Relaxed);
            }
        }
    }
    {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let span = LOOSE_MAX_LEN.load(Ordering::Relaxed);
        for (&(w_start, w_len), entry) in loose.range((start.saturating_sub(span), 0)..(end, 0)) {
            if w_start.saturating_add(w_len) > start
                && let Some(s) = entry.as_ready()
                && s.id != self_id
                && s.place.is_some()
            {
                s.in_place_overlap.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers_read()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Bounding box `[min_start, max_end]` of all in-place shadows ever created
/// (`max_end == 0` when none) so unrelated streaming unmaps skip `LOOSE`/`BUFFERS` locks.
static IN_PLACE_BOUNDS: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(0)];

/// Largest loose window length inserted into `LOOSE`, so overlap scans never miss
/// windows larger than `WINDOW` (4 MiB).
static LOOSE_MAX_LEN: AtomicU64 = AtomicU64::new(WINDOW);

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let hi = IN_PLACE_BOUNDS[1].load(Ordering::Relaxed);
    if hi == 0 || hi <= at || IN_PLACE_BOUNDS[0].load(Ordering::Relaxed) >= end {
        return 0;
    }
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    let loose_span = LOOSE_MAX_LEN.load(Ordering::Relaxed);
    let has_loose = {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        map.range((at.saturating_sub(loose_span), 0)..(end, 0)).any(|(key, entry)| {
            key.0 + key.1 > at && entry.as_ready().is_some_and(|s| s.place.is_some())
        })
    };
    if has_loose {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(loose_span), 0)..(end, 0))
            .filter(|(key, entry)| {
                key.0 + key.1 > at && entry.as_ready().is_some_and(|s| s.place.is_some())
            })
            .map(|(key, _)| *key)
            .collect();
        let before = gone.len();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key).and_then(LooseEntry::into_ready)));
        if gone.len() > before {
            for shadow in &gone[before..] {
                shadow.dead.store(true, Ordering::Release);
            }
            bump_shadow_epoch();
        }
    }
    let has_buf = {
        let map = buffers_read();
        map.range(..(end, 0))
            .rev()
            .take(256)
            .any(|(_, entry)| entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()))
    };
    if has_buf {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers_mut();
        let before = gone.len();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
        if gone.len() > before {
            for shadow in &gone[before..] {
                shadow.dead.store(true, Ordering::Release);
            }
            bump_shadow_epoch();
        }
    }
    if !gone.is_empty() {
        use windows::core::Interface;
        let mut raws: Vec<usize> = gone
            .iter()
            .filter(|s| s.window)
            .map(|s| s.buffer.as_raw() as usize)
            .collect();
        if !raws.is_empty() {
            raws.sort_unstable();
            raws.dedup();
            let _ = crate::descriptor::host::unpin_idle_window_slots(&raws);
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// Evicts and unpins any non-placed loose shadow windows whose guest memory
/// lies completely inside `[at, at + len)` when the title unmaps/decommits/remaps
/// that VA range (`inplace::title_memory_freeing`). Uses Null-Before-Retire
/// (`unpin_idle_window_slots`, Spec-Trap #16) strictly on freed guest VAs so live
/// bindless descriptors are never nulled.
pub(crate) fn loose_unmapped(at: u64, len: u64) -> u64 {
    use windows::core::Interface;
    if len == 0 {
        return 0;
    }
    let end = at.saturating_add(len);
    let has_any = {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        map.range((at, 0)..(end, 0)).any(|(key, entry)| {
            key.0 >= at
                && key.0.saturating_add(key.1) <= end
                && entry.as_ready().is_some_and(|s| s.place.is_none())
        })
    };
    if !has_any {
        return 0;
    }
    let gone: Vec<Arc<Shadow>> = {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at, 0)..(end, 0))
            .filter(|(key, entry)| {
                key.0 >= at
                    && key.0.saturating_add(key.1) <= end
                    && entry.as_ready().is_some_and(|s| s.place.is_none())
            })
            .map(|(key, _)| *key)
            .collect();
        let removed: Vec<Arc<Shadow>> = keys
            .into_iter()
            .filter_map(|k| map.remove(&k).and_then(LooseEntry::into_ready))
            .collect();
        if !removed.is_empty() {
            for s in &removed {
                s.dead.store(true, Ordering::Release);
            }
            bump_shadow_epoch();
        }
        removed
    };
    if gone.is_empty() {
        return 0;
    }
    let mut raws: Vec<usize> = gone.iter().map(|s| s.buffer.as_raw() as usize).collect();
    raws.sort_unstable();
    raws.dedup();
    let _ = crate::descriptor::host::unpin_idle_window_slots(&raws);
    let n = gone.len() as u64;
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        if shadow.mapped {
            let vram = shadow.len;
            crate::host::retire_vram(shadow, vram);
        } else {
            crate::host::retire(shadow);
        }
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.iter().filter(|(_, e)| e.as_ready().is_some()).map(|(&(start, len), _)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// Calls `f` with the [`Shadow`] holding `[address, address + len)` if it is
/// in-place, borrowing directly from the thread-local span cache on a hit to
/// avoid both global `RwLock` acquisition and `Arc::clone`/`drop` traffic.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn with_in_place<R>(address: u64, len: u64, f: impl FnOnce(&Arc<Shadow>) -> R) -> Option<R> {
    let end = address.checked_add(len.max(1))?;
    if quick_find() {
        let epoch = SHADOW_EPOCH.load(Ordering::Acquire);
        enum Probe {
            BufferAtZero,
            Loose(Arc<Shadow>),
            Miss,
        }
        let probe = SPANS.with(|cell| {
            let Ok(mut cache) = cell.try_borrow_mut() else {
                return Probe::Miss;
            };
            cache.sync(epoch);
            let Some(idx) = cache.entries.iter().position(|e| address >= e.start && end <= e.end) else {
                return Probe::Miss;
            };
            if idx > 0 {
                cache.entries[..=idx].rotate_right(1);
            }
            match &cache.entries[0].shadow {
                SpanRef::Buffer(shadow) => {
                    if !shadow.is_dead() {
                        Probe::BufferAtZero
                    } else {
                        cache.entries.remove(0);
                        Probe::Miss
                    }
                }
                SpanRef::Loose(weak) => {
                    if let Some(shadow) = weak.upgrade()
                        && !shadow.is_dead()
                    {
                        note_loose_hit(&shadow, address, end, len <= 1);
                        Probe::Loose(shadow)
                    } else {
                        cache.entries.remove(0);
                        Probe::Miss
                    }
                }
            }
        });
        match probe {
            Probe::BufferAtZero => {
                return SPANS.with(|cell| {
                    let cache = cell.borrow();
                    if let Some(SpanEntry {
                        shadow: SpanRef::Buffer(shadow),
                        ..
                    }) = cache.entries.first()
                    {
                        shadow.in_place().then(|| f(shadow))
                    } else {
                        None
                    }
                });
            }
            Probe::Loose(shadow) => {
                return shadow.in_place().then(|| f(&shadow));
            }
            Probe::Miss => {}
        }
    }
    let shadow = lookup(address, len, false)?;
    shadow.in_place().then(|| f(&shadow))
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// The copy [`find`] gives for a range inside a title buffer, if it was made.
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    buffers_read().range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end).and_then(|(_, e)| e.shadow.clone())
}

/// Resolves a guest buffer `[address, address + len)` to its single unambiguous
/// live host [`Shadow`] for a resource-scoped UAV barrier.
///
/// Returns `None` if no live `Shadow` covers the full range, or if any aliased
/// buffer, exact-view copy, or loose window overlaps `[address, address + len)`.
pub(crate) fn scoped_barrier_shadow(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if address == 0 || len == 0 {
        return None;
    }
    let end = address.checked_add(len)?;
    let chosen = {
        let map = buffers_read();
        let mut found: Option<Arc<Shadow>> = None;
        for (_, entry) in map.range(..=(address, usize::MAX)).rev().take(64) {
            if entry.end <= address {
                continue;
            }
            let Some(shadow) = &entry.shadow else {
                continue;
            };
            if found.is_none() && entry.end >= end && shadow.left(address) >= len {
                found = Some(shadow.clone());
            } else if found.as_ref().is_none_or(|f| !Arc::ptr_eq(f, shadow)) {
                return None;
            }
        }
        let chosen = found?;
        for (_, entry) in map.range((address.saturating_add(1), 0)..(end, 0)) {
            if let Some(shadow) = &entry.shadow
                && !Arc::ptr_eq(&chosen, shadow)
            {
                return None;
            }
        }
        chosen
    };
    {
        let exact = EXACT.lock().unwrap_or_else(|e| e.into_inner());
        if !exact.1.is_empty() {
            const EXACT_MOST: u64 = 64 << 20;
            for (&(ex_start, ex_len), _) in exact.1.range((address.saturating_sub(EXACT_MOST), 0)..(end, 0)) {
                if ex_start.saturating_add(ex_len) > address {
                    return None;
                }
            }
        }
    }
    {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        if !loose.is_empty() {
            for (&(w_start, w_len), _) in loose.range((address.saturating_sub(WINDOW), 0)..(end, 0)) {
                if w_start.saturating_add(w_len) > address {
                    return None;
                }
            }
        }
    }
    Some(chosen)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let quick = quick_find();
    if quick {
        if let Some(shadow) = span_find(address, end, len <= 1) {
            return Some(shadow);
        }
        if let Some(shadow) = found(address, len) {
            return Some(shadow);
        }
    }
    {
        let map = buffers_read();
        let epoch = SHADOW_EPOCH.load(Ordering::Acquire);
        let mut skipped_max_end = 0u64;
        let mut matched: Option<((u64, usize), &Entry)> = None;
        for (k, e) in map.range(..=(address, usize::MAX)).rev().take(64) {
            if e.end >= end {
                matched = Some((*k, e));
                break;
            }
            skipped_max_end = skipped_max_end.max(e.end);
        }
        match matched {
            // The copy is made already (nearly always).
            Some((key, Entry { end: entry_end, shadow: Some(shadow) })) => {
                let out = shadow.clone();
                if quick {
                    let safe_start = key.0.max(skipped_max_end);
                    let next_after = map
                        .range((std::ops::Bound::Excluded((address, usize::MAX)), std::ops::Bound::Unbounded))
                        .next()
                        .map(|(k, _)| k.0);
                    let safe_end = (*entry_end).min(next_after.unwrap_or(u64::MAX));
                    drop(map);
                    if safe_start <= address && end <= safe_end {
                        span_remember(epoch, safe_start, safe_end, SpanRef::Buffer(out.clone()));
                    }
                    remember(address, len, epoch, &out);
                }
                return Some(out);
            }
            Some(_) => {}
            None => {
                let loose_span = if quick && !fine() {
                    let (cell_start, cell_len) = window_of(address, end, WINDOW);
                    let cell_end = cell_start.saturating_add(WINDOW);
                    let free = cell_len == WINDOW
                        && map.range(..=(cell_start, usize::MAX)).rev().take(64).all(|(_, e)| e.end <= cell_start)
                        && map.range((cell_start, 0)..(cell_end, 0)).next().is_none();
                    free.then_some((cell_start, cell_end))
                } else {
                    None
                };
                drop(map);
                let shadow = loose(address, end, len <= 1, any)?;
                if let Some((cell_start, cell_end)) = loose_span
                    && shadow.span() == (cell_start, WINDOW)
                {
                    span_remember(epoch, cell_start, cell_end, SpanRef::Loose(Arc::downgrade(&shadow)));
                }
                return Some(shadow);
            }
        }
    }
    let mut map = buffers_mut();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && (size >= card_limits().0 || card_all());
        if let Some(placed) = crate::port::place_for_buffer(key.0, len, big) {
            let shadow = Arc::new(Shadow::placed(placed, key.0, len, false));
            entry.shadow = Some(shadow.clone());
            bump_shadow_epoch();
            drop(map);
            on_placed_shadow_created(&shadow);
            return Some(shadow);
        }
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            let shadow = Arc::new(Shadow::placed(placed, key.0, len, false));
            entry.shadow = Some(shadow.clone());
            bump_shadow_epoch();
            drop(map);
            on_placed_shadow_created(&shadow);
            return Some(shadow);
        }
        let host = host()?;
        // Allocate the card copy first so `VICTORY_SEND_OWN=1` can back the
        // CPU side with plain demand-paged memory instead of a D3D12 `SHR` buffer.
        let (buffer, data, card, l0_bytes) = match card_for(size) {
            Some(on_card) => {
                let Some((staging, data)) =
                    card_staging_for(host, size, || format!("title buffer {:#x}+{len:#x}", key.0))
                else {
                    CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
                    return None;
                };
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card::new(staging, size);
                (on_card, data, Some(card), 0)
            }
            None => {
                // VRAM copy refused or ineligible: prefer in-place over a L0 `shared_buffer` copy.
                if big && let Some(placed) = crate::port::place_for_buffer(key.0, len, false) {
                    let shadow = Arc::new(Shadow::placed(placed, key.0, len, false));
                    entry.shadow = Some(shadow.clone());
                    bump_shadow_epoch();
                    drop(map);
                    on_placed_shadow_created(&shadow);
                    return Some(shadow);
                }
                let (buffer, data) = host.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
                (buffer, data, None, size)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        if l0_bytes > 0 {
            shadow.l0_bytes = l0_bytes;
            L0_SHADOW_BYTES.fetch_add(l0_bytes, Ordering::Relaxed);
        }
        let shadow = Arc::new(shadow);
        let new_id = shadow.id;
        if shadow.card.is_some() {
            push_carded(&shadow);
            gpu_only_register(&shadow);
        }
        entry.shadow = Some(shadow.clone());
        bump_shadow_epoch();
        drop(map);
        demote_overlapping_gpu_only(key.0, len, new_id, "overlapping buffer shadow");
        note_shadow_overlapping_in_place(key.0, len, new_id);
        return Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;
/// Default loose-window grid in KiB (`512 KiB`, fitting cleanly inside section
/// stretches via `fine()` + `tails()` while slashing `l0_loose` slack).
const DEFAULT_WINDOW_KB: u64 = 512;

/// The grid the windows of asks with a length are cut on, in bytes: `512 KiB`
/// by default unless `VICTORY_WINDOW_KB` names a power of two from 64 to 4096.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, `512 KiB` (`DEFAULT_WINDOW_KB`) for unset/invalid values.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => DEFAULT_WINDOW_KB << 10,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Minimum clamped section-run window size (`512 KiB`) so section clamping
/// never fragments loose windows into thousands of tiny 64–256 KiB `port` view heaps.
const MIN_CLAMPED_SECTION_RUN: u64 = 512 << 10;

/// Clamps a candidate loose window `(win_start, win_len)` to a contiguous
/// single-section stretch `(run_start, run_end)` containing `[address, end)` so
/// the window never crosses section boundaries and opens via zero-copy
/// `OpenExistingHeapFromFileMapping` instead of failing `"not one stretch of a section"`.
fn clamp_window_to_section_run(
    address: u64,
    end: u64,
    win_start: u64,
    win_len: u64,
    run: Option<(u64, u64)>,
) -> (u64, u64) {
    let Some((run_start, run_end)) = run else {
        return (win_start, win_len);
    };
    if run_start > address || run_end < end {
        return (win_start, win_len);
    }
    let win_end = win_start.saturating_add(win_len);
    if run_start <= win_start && win_end <= run_end {
        return (win_start, win_len);
    }
    let clamped_start = win_start.max(run_start);
    let clamped_end = win_end.min(run_end);
    let clamped_len = clamped_end.saturating_sub(clamped_start);
    if clamped_start <= address && clamped_end >= end && clamped_len >= MIN_CLAMPED_SECTION_RUN {
        (clamped_start, clamped_len)
    } else {
        (win_start, win_len)
    }
}

/// Whether the window of an ask without a length begins at the ask's
/// `TAIL_STEP` (`256 KiB`) instead of at the 4 MiB cell boundary. Default ON
/// (`VICTORY_WINDOW_TAILS=0` disables; `ARCH-05`).
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_none_or(|v| v != "0"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range, including windows starting in a prior grid cell (`CPU-33`).
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start.saturating_add(bytes) >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Looks up the best covering loose window in `map` for `[address, end)`,
/// scanning back by `max_len` so requests straddling 4 MiB or grid boundaries
/// find cross-cell windows instead of allocating overlapping duplicates (`CPU-33`).
fn holder_in<V>(map: &BTreeMap<(u64, u64), V>, address: u64, end: u64, max_len: u64) -> Option<(u64, u64)> {
    let search_lo = address.saturating_sub(max_len);
    let there = map.range((search_lo, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
    holder(there, address, end)
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.iter().filter(|(_, e)| e.as_ready().is_some()).map(|(k, _)| *k))
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
#[cfg_attr(not(test), allow(dead_code))]
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    longest_tail_bounded(windows, address, cell, cell + len)
}

/// [`longest_tail`] generalized to a section-clamped tail cell `[min_start, target_end)`.
fn longest_tail_bounded(
    windows: impl Iterator<Item = (u64, u64)>,
    address: u64,
    min_start: u64,
    target_end: u64,
) -> Option<(u64, u64)> {
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= min_start && start <= step && start.saturating_add(bytes) == target_end)
        .min()
}

/// Whether `[address, address + len)` lies in one title buffer.
pub(crate) fn in_buffer(address: u64, len: u64) -> Option<(u64, u64)> {
    let end = address.checked_add(len)?;
    let map = buffers_read();
    let found = map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end);
    found.map(|(k, e)| (k.0, e.end - k.0))
}

/// Whether a title buffer of at most `most` bytes lies in `[address, address
/// + len)`, all of it or in part.
pub(crate) fn small_buffer_in(address: u64, len: u64, most: u64) -> bool {
    let map = buffers_read();
    map.range((address.saturating_sub(most), 0)..(address + len, 0))
        .any(|(key, entry)| entry.end > address && entry.end - key.0 <= most)
}

/// A copy of the memory `[start, start + len)` (an arena of
/// [`crate::memory`]'s, its directory, title memory around a small buffer)
/// for the programs that find memory by address, on the card: the structures
/// rays are traced through, read with every step of every ray.
///
/// What shaders write there by address is not brought back, so the copy is
/// the only place that has it. The CPU's side holds the bytes as they were
/// last taken, and only dwords that changed since go up.
pub(crate) fn by_address(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let host = host()?;
    if !cards() || !host.card_ready() {
        return None;
    }
    // At most so much on the card's own memory (4 GiB unless
    // `VICTORY_MEMORY_CARD_MB` says otherwise), and while it has room; past
    // that the side shaders see is in the PC's memory.
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static USED: AtomicU64 = AtomicU64::new(0);
    let most = *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_MEMORY_CARD_MB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        mb.unwrap_or(4096) << 20
    });
    let room = host.video_memory().is_none_or(|[budget, used, ..]| used + len < budget / 100 * 97);
    let (staging, data) = card_staging_for(host, len, || format!("title memory {start:#x}+{len:#x} by address, the CPU's side"))?;
    let on_card = if room && USED.fetch_add(len, Ordering::Relaxed) + len <= most {
        host.uav_buffer(len)
    } else {
        None
    };
    let buffer = match on_card {
        Some(buffer) => buffer,
        None => host.shared_buffer(len)?.0,
    };
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} by address"));
    // (Counted down with the windows' on the card when it goes.)
    WINDOW_CARD_BYTES.fetch_add(len, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.changes_only.store(true, Ordering::Relaxed);
    shadow.card = Some(Card::new(staging, len));
    let shadow = Arc::new(shadow);
    push_carded(&shadow);
    demote_overlapping_gpu_only(start, len, shadow.id, "overlapping address shadow");
    Some(shadow)
}

/// Two-phase reservation cell for a loose window currently being allocated
/// outside `LOOSE.write()`: threads requesting the exact same `(start, len)`
/// wait on this cell, while threads requesting any other key proceed lock-free.
struct PendingLoose {
    result: std::sync::OnceLock<Option<Arc<Shadow>>>,
    done: Mutex<bool>,
    ready: std::sync::Condvar,
}

impl PendingLoose {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            result: std::sync::OnceLock::new(),
            done: Mutex::new(false),
            ready: std::sync::Condvar::new(),
        })
    }

    fn wait(&self) -> Option<Arc<Shadow>> {
        if let Some(res) = self.result.get() {
            return res.clone();
        }
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = self.ready.wait(done).unwrap_or_else(|e| e.into_inner());
        }
        self.result.get().and_then(Option::clone)
    }

    fn complete(&self, val: Option<Arc<Shadow>>) {
        let _ = self.result.set(val);
        let mut done = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *done = true;
        self.ready.notify_all();
    }
}

enum LooseEntry {
    Ready(Arc<Shadow>),
    Pending(Arc<PendingLoose>),
}

impl LooseEntry {
    #[inline]
    fn as_ready(&self) -> Option<&Arc<Shadow>> {
        match self {
            LooseEntry::Ready(s) => Some(s),
            LooseEntry::Pending(_) => None,
        }
    }

    #[inline]
    fn into_ready(self) -> Option<Arc<Shadow>> {
        match self {
            LooseEntry::Ready(s) => Some(s),
            LooseEntry::Pending(_) => None,
        }
    }
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<BTreeMap<(u64, u64), LooseEntry>> = std::sync::RwLock::new(BTreeMap::new());

/// RAII guard guaranteeing that a `PendingLoose` slot is always removed and
/// waiters are always unblocked on early return, allocation failure, or panic.
struct PendingLooseGuard<'a> {
    map_lock: &'a std::sync::RwLock<BTreeMap<(u64, u64), LooseEntry>>,
    key: (u64, u64),
    pending: Arc<PendingLoose>,
    finished: bool,
}

impl Drop for PendingLooseGuard<'_> {
    fn drop(&mut self) {
        if !self.finished {
            {
                let mut map = self.map_lock.write().unwrap_or_else(|e| e.into_inner());
                if matches!(map.get(&self.key), Some(LooseEntry::Pending(p)) if Arc::ptr_eq(p, &self.pending)) {
                    map.remove(&self.key);
                }
            }
            self.pending.complete(None);
        }
    }
}

/// Two-phase lock-free slot reservation in `map_lock`: checks under `read()`
/// first, then reserves a `PendingLoose` slot under a sub-microsecond `write()`
/// and invokes `create` completely OUTSIDE `map_lock`. Only threads requesting
/// the exact same key wait on `PendingLoose`.
fn loose_get_or_create_in(
    map_lock: &std::sync::RwLock<BTreeMap<(u64, u64), LooseEntry>>,
    key: (u64, u64),
    create: impl FnOnce() -> Option<Arc<Shadow>>,
    on_hit: impl Fn(&Shadow),
) -> Option<(Arc<Shadow>, bool)> {
    // Phase 1: check under read lock (no write lock contention).
    {
        let map = map_lock.read().unwrap_or_else(|e| e.into_inner());
        match map.get(&key) {
            Some(LooseEntry::Ready(s)) => {
                on_hit(s);
                return Some((s.clone(), false));
            }
            Some(LooseEntry::Pending(p)) => {
                let p = p.clone();
                drop(map);
                let s = p.wait()?;
                on_hit(&s);
                return Some((s, false));
            }
            None => {}
        }
    }
    // Phase 2: sub-microsecond write lock to reserve a `Pending` slot.
    let pending = {
        let mut map = map_lock.write().unwrap_or_else(|e| e.into_inner());
        match map.get(&key) {
            Some(LooseEntry::Ready(s)) => {
                on_hit(s);
                return Some((s.clone(), false));
            }
            Some(LooseEntry::Pending(p)) => {
                let p = p.clone();
                drop(map);
                let s = p.wait()?;
                on_hit(&s);
                return Some((s, false));
            }
            None => {
                let p = PendingLoose::new();
                map.insert(key, LooseEntry::Pending(p.clone()));
                p
            }
        }
    };
    let mut guard = PendingLooseGuard {
        map_lock,
        key,
        pending: pending.clone(),
        finished: false,
    };
    // Phase 3: perform all D3D12 resource creation, `inplace` placement, and
    // residency OUTSIDE `map_lock`.
    let created = create();
    // Phase 4: sub-microsecond write lock to publish or remove the slot.
    let published = {
        let mut map = map_lock.write().unwrap_or_else(|e| e.into_inner());
        match created {
            Some(shadow) => {
                on_hit(&shadow);
                LOOSE_MAX_LEN.fetch_max(key.1, Ordering::Relaxed);
                map.insert(key, LooseEntry::Ready(shadow.clone()));
                bump_shadow_epoch();
                Some(shadow)
            }
            None => {
                if matches!(map.get(&key), Some(LooseEntry::Pending(p)) if Arc::ptr_eq(p, &pending)) {
                    map.remove(&key);
                }
                None
            }
        }
    };
    guard.finished = true;
    pending.complete(published.clone());
    published.map(|s| (s, true))
}

fn create_loose_shadow(start: u64, len: u64) -> Option<Arc<Shadow>> {
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::port::place_for_window(start, len, false).or_else(|| crate::inplace::place_for(start, len, true)) {
        return Some(Arc::new(Shadow::placed(placed, start, len, true)));
    }
    // Allocate up to `VICTORY_WINDOWS_BAR_MB` (640 MiB, VRAM < 80%) directly in
    // BAR at creation so hot windows get BAR bandwidth without ever swapping/nulling
    // live bindless title-heap descriptors; overflow tries safe `port` placement
    // (section or `MEM_PRIVATE` within `VICTORY_INPLACE_MB`) before L0 `shared_buffer`.
    let host = host()?;
    let (buffer, data, card, mapped, l0_bytes) = match window_card_for(len) {
        Some(on_card) => {
            let Some((staging, data)) =
                card_staging_for(host, len, || format!("title memory {start:#x}+{len:#x}"))
            else {
                WINDOW_CARD_BYTES.fetch_sub(len, Ordering::Relaxed);
                return None;
            };
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card::new(staging, len);
            (on_card, data, Some(card), false, 0)
        }
        None => {
            if let Some((buffer, data)) = window_mapped_for(len) {
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} mapped on the card"));
                (buffer, data, None, true, 0)
            } else if let Some(placed) = crate::port::place_for_window(start, len, true) {
                return Some(Arc::new(Shadow::placed(placed, start, len, true)));
            } else {
                let (buffer, data) = host.shared_buffer(len)?;
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
                (buffer, data, None, false, len)
            }
        }
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.mapped = mapped;
    shadow.card = card;
    if l0_bytes > 0 {
        shadow.l0_bytes = l0_bytes;
        L0_SHADOW_BYTES.fetch_add(l0_bytes, Ordering::Relaxed);
        L0_LOOSE_BYTES.fetch_add(l0_bytes, Ordering::Relaxed);
    }
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        push_carded(&shadow);
    }
    Some(shadow)
}

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let (cell, cell_len) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let use_tails = unsized_ask && tails();
    let use_holder = any && fine() && !unsized_ask;
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| note_loose_hit(shadow, address, end, unsized_ask);
    // Fast path: check `LOOSE.read()` first with zero `VIEWS` / `LOOSE.write()` overhead.
    {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let hit_key = if use_tails {
            let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
            longest_tail_bounded(there, address, cell, cell + cell_len).or_else(|| {
                // Also hit a clamped in-place `SEC_COMMIT` tail without locking `VIEWS`.
                map.range((cell, 0)..=(address, u64::MAX))
                    .find(|((s, n), e)| {
                        *s <= address
                            && s.saturating_add(*n) >= end
                            && e.as_ready().is_some_and(|sh| sh.place.is_some())
                    })
                    .map(|(&k, _)| k)
            })
        } else if use_holder {
            holder_in(&map, address, end, LOOSE_MAX_LEN.load(Ordering::Relaxed))
        } else if map.contains_key(&named) {
            Some(named)
        } else {
            // Also hit a clamped in-place `SEC_COMMIT` window without locking `VIEWS`.
            let lo = address.saturating_sub(WINDOW);
            map.range((lo, 0)..=(address, u64::MAX))
                .find(|((s, n), e)| {
                    *s <= address
                        && s.saturating_add(*n) >= end
                        && e.as_ready().is_some_and(|sh| sh.place.is_some())
                })
                .map(|(&k, _)| k)
        };
        if let Some(key) = hit_key {
            match map.get(&key) {
                Some(LooseEntry::Ready(shadow)) => {
                    note(shadow);
                    return Some(shadow.clone());
                }
                Some(LooseEntry::Pending(p)) => {
                    let p = p.clone();
                    drop(map);
                    let shadow = p.wait()?;
                    note(&shadow);
                    return Some(shadow);
                }
                None => {}
            }
        }
    }
    // Miss path: clamp candidate window ONLY along `SEC_COMMIT` contiguous section
    // stretches so windows open via zero-pinned `OpenExistingHeapFromFileMapping`
    // without fragmenting `SEC_RESERVE` stretches into overlapping L0 copies.
    let section_run = crate::inplace::committed_section_run_bounds(address, end);
    let key = if use_tails {
        let raw_tail = tail_of(address, end);
        let clamped_tail = clamp_window_to_section_run(address, end, raw_tail.0, raw_tail.1, section_run);
        let min_start = cell.max(section_run.map_or(0, |r| r.0));
        let target_end = clamped_tail.0.saturating_add(clamped_tail.1);
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let there = map.range((min_start, 0)..=(address, u64::MAX)).map(|(k, _)| *k);
        longest_tail_bounded(there, address, min_start, target_end).unwrap_or(clamped_tail)
    } else if use_holder {
        let clamped_named = clamp_window_to_section_run(address, end, named.0, named.1, section_run);
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        holder_in(&map, address, end, LOOSE_MAX_LEN.load(Ordering::Relaxed)).unwrap_or(clamped_named)
    } else {
        clamp_window_to_section_run(address, end, named.0, named.1, section_run)
    };
    let (start, len) = key;
    let (shadow, created) = loose_get_or_create_in(&LOOSE, key, || create_loose_shadow(start, len), note)?;
    if created {
        if shadow.place.is_some() {
            on_placed_shadow_created(&shadow);
        } else {
            demote_overlapping_gpu_only(start, len, shadow.id, "overlapping loose window");
            note_shadow_overlapping_in_place(start, len, shadow.id);
        }
    }
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `view_home`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    /// The largest, and no view the title made on purpose: off its dwords.
    const MOST: u64 = 64 << 20;
    if len == 0 || len > MOST || !address.is_multiple_of(4) {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let exact_bytes = len.next_multiple_of(256);
    let (buffer, data) = host()?.shared_buffer(exact_bytes)?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, address, len);
    shadow.l0_bytes = exact_bytes;
    L0_SHADOW_BYTES.fetch_add(exact_bytes, Ordering::Relaxed);
    let shadow = Arc::new(shadow);
    let new_id = shadow.id;
    map.insert((address, len), (shadow.clone(), now));
    drop(guard);
    demote_overlapping_gpu_only(address, len, new_id, "overlapping exact view");
    note_shadow_overlapping_in_place(address, len, new_id);
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    OWN_MADE.fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// Copies of single views' bytes made since the start.
pub(crate) static OWN_MADE: AtomicU64 = AtomicU64::new(0);

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_BYTES=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{
        WINDOW, cover, default_card_limits_mb, grid, holder, longest_tail, resolve_card_limits,
        resolve_card_vram_ceiling, resolve_card_vram_headroom, resolve_window_budget_mb, tail_of,
        window_of, window_victims,
    };

    #[test]
    fn window_budget_evicts_oldest_only_until_it_fits() {
        const M: u64 = 1 << 20;
        // Oldest first: 10, 20 and 30 MiB. Keeping 50 MiB must free only the
        // oldest copy (60 MiB left before, 50 MiB after).
        let entries = [(1u64, 10 * M, 1usize), (2, 20 * M, 2), (3, 30 * M, 3)];
        assert_eq!(window_victims(&entries, 50 * M), vec![0]);
        // A tighter cap takes the newest as well.
        assert_eq!(window_victims(&entries, 20 * M), vec![0, 1, 2]);
        // Zero remaining budget (e.g. active windows already fill cap) evicts all idle entries.
        assert_eq!(window_victims(&entries, 0), vec![0, 1, 2]);
    }

    #[test]
    fn window_budget_counts_one_copy_under_several_keys_once() {
        const M: u64 = 1 << 20;
        // The first two entries are one copy; 50 MiB is live, not 80.
        // Without dedupe both would be picked to reach a 40 MiB cap.
        let entries = [(1u64, 30 * M, 7usize), (2, 30 * M, 7), (3, 20 * M, 8)];
        assert_eq!(window_victims(&entries, 40 * M), vec![0]);
    }

    #[test]
    fn window_budget_defaults_and_env_override() {
        assert_eq!(resolve_window_budget_mb(None, true), Some(512));
        assert_eq!(resolve_window_budget_mb(None, false), Some(512));
        assert_eq!(resolve_window_budget_mb(Some("0"), true), None);
        assert_eq!(resolve_window_budget_mb(Some(" 1024 "), true), Some(1024));
        assert_eq!(resolve_window_budget_mb(Some("invalid"), true), Some(512));
    }

    #[test]
    fn card_vram_headroom_caps_small_and_16gb_buffers_below_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        const KIB: u64 = 1 << 10;

        // Default (unset): opt-in headroom is 0, using the existing percentage cap.
        let budget_16g = 14_643 * MIB;
        let default_min_16g = default_card_limits_mb(Some(budget_16g)).0 * MIB;
        assert_eq!(resolve_card_vram_headroom(budget_16g, 64 * KIB, default_min_16g, None), 0);
        assert_eq!(resolve_card_vram_headroom(budget_16g, 16 * MIB, default_min_16g, None), 0);
        assert_eq!(
            resolve_card_vram_ceiling(budget_16g, 97, 0),
            (budget_16g / 100) * 97
        );

        let budget_12g = 10 * GIB;
        let default_min_12g = default_card_limits_mb(Some(budget_12g)).0 * MIB;
        assert_eq!(resolve_card_vram_headroom(budget_12g, 64 * KIB, default_min_12g, None), 0);
        assert_eq!(resolve_card_vram_headroom(budget_12g, 16 * MIB, default_min_12g, None), 0);

        // Explicit headroom override applies the tighter of percentage cap and budget - headroom.
        assert_eq!(
            resolve_card_vram_headroom(budget_16g, 64 * KIB, default_min_16g, Some(1024)),
            1024 * MIB
        );
        assert_eq!(
            resolve_card_vram_ceiling(budget_16g, 97, 1024 * MIB),
            budget_16g - 1024 * MIB
        );
        assert_eq!(
            resolve_card_vram_headroom(budget_16g, 64 * KIB, default_min_16g, Some(0)),
            0
        );
        assert_eq!(
            resolve_card_vram_headroom(budget_12g, 16 * MIB, default_min_12g, Some(2048)),
            2048 * MIB
        );
        assert_eq!(
            resolve_card_vram_ceiling(budget_12g, 97, u64::MAX),
            0
        );
    }

    #[test]
    fn card_limits_scale_with_local_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;

        // Unknown (WARP / query failure) and <= 12 GiB keep the 12 GB baseline.
        assert_eq!(default_card_limits_mb(None), (16, 512));
        assert_eq!(default_card_limits_mb(Some(8 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(11 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(12 * GIB)), (16, 512));

        // 16 GB tier (> 12 GiB, < 20 GiB), including WDDM budgets around 13.5–15 GiB.
        assert_eq!(default_card_limits_mb(Some(13 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(14 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(15 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(16 * GIB)), (4, 2048));

        // 24–32+ GB tier (>= 20 GiB): 1 MiB min, 25% of budget capped at 8192 MiB.
        assert_eq!(default_card_limits_mb(Some(20 * GIB)), (1, 5120));
        assert_eq!(default_card_limits_mb(Some(24 * GIB)), (1, 6144));
        assert_eq!(default_card_limits_mb(Some(32 * GIB)), (1, 8192));
        assert_eq!(default_card_limits_mb(Some(48 * GIB)), (1, 8192));

        // Explicit env overrides always win, clamp min to >= 1 MiB, and saturate on overflow.
        assert_eq!(resolve_card_limits(Some(32 * GIB), Some(8), Some(4096)), (8 * MIB, 4096 * MIB));
        assert_eq!(resolve_card_limits(Some(12 * GIB), Some(0), Some(1024)), (MIB, 1024 * MIB));
        assert_eq!(resolve_card_limits(None, Some(u64::MAX), Some(u64::MAX)), (u64::MAX, u64::MAX));
        assert_eq!(resolve_card_limits(None, None, None), (16 * MIB, 512 * MIB));
    }

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; unset/invalid
    /// defaults to 512 KiB.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        let default = super::DEFAULT_WINDOW_KB << 10;
        assert_eq!(default, 512 << 10);
        assert_eq!(grid(None), default);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(512)), 512 << 10);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), default, "{bad}");
        }
    }

    #[test]
    fn clamp_window_to_section_run_respects_section_boundaries() {
        let run = Some((0x1008_0000u64, 0x1020_0000u64)); // 1.5 MiB section stretch
        // 4 MiB tail starting at 0x1008_0000 clamps end to 0x1020_0000.
        assert_eq!(
            super::clamp_window_to_section_run(0x1008_1000, 0x1008_2000, 0x1008_0000, 4 << 20, run),
            (0x1008_0000, 0x18_0000)
        );
        // 512 KiB window [0x1000_0000, 0x1008_0000 + 0x8_0000) straddling run start clamps start to 0x1008_0000.
        assert_eq!(
            super::clamp_window_to_section_run(0x1009_0000, 0x100A_0000, 0x1000_0000, 1 << 20, run),
            (0x1008_0000, 0x8_0000)
        );
        // Non-section memory (None) stays unclamped.
        assert_eq!(
            super::clamp_window_to_section_run(0x1009_0000, 0x100A_0000, 0x1008_0000, 512 << 10, None),
            (0x1008_0000, 512 << 10)
        );
    }

    #[test]
    fn cross_cell_holder_finds_straddling_windows_without_duplicates() {
        use std::collections::BTreeMap;
        let cell = 0xd_1200_0000u64;
        // A 512 KiB window straddling the 4 MiB cell boundary `[cell - 256 KiB, cell + 256 KiB)`.
        let straddle = (cell - (256 << 10), 512 << 10);
        let mut map = BTreeMap::new();
        map.insert(straddle, ());
        map.insert((cell + (512 << 10), 256 << 10), ());
        // An ask starting inside `cell` (`[cell + 4 KiB, cell + 64 KiB)`) MUST find `straddle`
        // even though `straddle.0 < cell` (`CPU-33`).
        let got = super::holder_in(&map, cell + 4096, cell + (64 << 10), WINDOW);
        assert_eq!(got, Some(straddle));
        // An ask past `straddle`'s end does not falsely match it.
        assert_eq!(super::holder_in(&map, cell + (256 << 10), cell + (300 << 10), WINDOW), None);
    }

    #[test]
    fn cpu_staging_demand_commits_only_touched_64k_pieces() {
        use std::sync::atomic::Ordering;
        let before_global = super::cpu_staging_committed_bytes();
        let total_size = 16u64 << 20; // 16 MiB = 256 pieces
        let (staging, ptr) = super::CpuStaging::alloc(total_size).expect("MEM_RESERVE");
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 0);
        assert!(!staging.is_piece_committed(0));
        assert!(!staging.is_piece_committed(100));

        // Commit only piece 3 and pieces 100..=101 (3 * 64 KiB = 192 KiB out of 16 MiB).
        assert!(staging.ensure_committed(3 * super::PIECE + 128, 3 * super::PIECE + 512));
        assert!(staging.ensure_committed(100 * super::PIECE + 4096, 102 * super::PIECE - 64));
        assert!(staging.is_piece_committed(3));
        assert!(staging.is_piece_committed(100));
        assert!(staging.is_piece_committed(101));
        assert!(!staging.is_piece_committed(0));
        assert!(!staging.is_piece_committed(4));
        assert!(!staging.is_piece_committed(102));
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 3 * super::PIECE as u64);

        // Verify committed pages are readable/writable with zero AV.
        // SAFETY: pieces 3 and 100..=101 were committed above.
        unsafe {
            ptr.add(3 * super::PIECE + 128).write(0xA5);
            ptr.add(101 * super::PIECE + 4096).write(0x5A);
            assert_eq!(ptr.add(3 * super::PIECE + 128).read(), 0xA5);
            assert_eq!(ptr.add(101 * super::PIECE + 4096).read(), 0x5A);
        }

        // Re-committing an already-committed range takes the fast path without double-counting.
        assert!(staging.ensure_committed(3 * super::PIECE, 4 * super::PIECE));
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 3 * super::PIECE as u64);
        assert!(super::cpu_staging_committed_bytes() >= before_global + 3 * super::PIECE as u64);
        drop(staging);
        assert_eq!(super::cpu_staging_committed_bytes(), before_global);
    }

    #[test]
    fn gpu_only_mode1_boot_grace_allows_post_boot_promotion() {
        use super::{
            GPU_ONLY_BOOT_GRACE_FRAME, GPU_ONLY_MAX_RESETS, GPU_ONLY_MIN_FETCHED_PIECES,
            GPU_ONLY_WARMUP_FRAMES, GpuOnlyTracker,
        };
        let mut t = GpuOnlyTracker::default();
        // 20 boot-time uploads (< frame 600) must NOT sticky-demote Mode-1 buffers (`ARCH-05`).
        for f in 1..=20u64 {
            assert!(!t.note_upload_at(1, false, f * 20, 0, 8));
        }
        assert_eq!(t.resets, 0);
        // Once steady state begins (>= frame 600) with 0 CPU uploads, Mode 1 promotes after warmup.
        let start_f = GPU_ONLY_BOOT_GRACE_FRAME + 10;
        let pieces: Vec<usize> = (0..GPU_ONLY_MIN_FETCHED_PIECES as usize).collect();
        assert_eq!(t.note_fetched(1, start_f, &pieces), (false, false));
        assert_eq!(t.note_fetched(1, start_f + GPU_ONLY_WARMUP_FRAMES, &[0]), (true, false));
        // Even during boot frames, an already-promoted buffer demotes immediately on CPU upload.
        assert!(t.note_upload_at(1, true, 100, 0, 1));
        // Mode-1 unpromoted uploads past boot/world-load restart observation without sticky ban,
        // while Mode-2 overlapping uploads still sticky-demote past `GPU_ONLY_MAX_RESETS`.
        let mut late_world = GpuOnlyTracker::default();
        for f in 0..20u64 {
            assert!(!late_world.note_upload_at(1, false, 6000 + f, 0, 1));
        }
        assert_eq!(late_world.note_fetched(1, 6100, &pieces), (false, false));
        assert_eq!(late_world.note_fetched(1, 6100 + GPU_ONLY_WARMUP_FRAMES, &[0]), (true, false));
        let mut noisy_m2 = GpuOnlyTracker::default();
        for _ in 0..GPU_ONLY_MAX_RESETS {
            assert_eq!(noisy_m2.note_fetched(2, GPU_ONLY_BOOT_GRACE_FRAME + 5, &[0]), (false, false));
            assert!(!noisy_m2.note_upload_at(2, false, GPU_ONLY_BOOT_GRACE_FRAME + 5, 0, 1));
        }
        assert_eq!(noisy_m2.note_fetched(2, GPU_ONLY_BOOT_GRACE_FRAME + 5, &[0]), (false, false));
        assert!(noisy_m2.note_upload_at(2, false, GPU_ONLY_BOOT_GRACE_FRAME + 5, 0, 1));
    }

    #[test]
    fn bar_tier_budget_and_promotion_policy() {
        assert_eq!(super::resolve_window_bar_mb(None), 640);
        assert_eq!(super::resolve_window_bar_mb(Some("0")), 0);
        assert_eq!(super::resolve_window_bar_mb(Some("768")), 768);
        assert_eq!(super::resolve_window_bar_mb(Some("bad")), 640);
        // Cold windows (<6 frames or <256 KiB copied) stay in L0; hot ones promote.
        assert!(!super::bar_should_promote(5, 512 << 10, 512 << 10));
        assert!(!super::bar_should_promote(6, 128 << 10, 512 << 10));
        assert!(super::bar_should_promote(6, 256 << 10, 512 << 10));
        assert!(super::bar_should_promote(6, 256 << 10, 256 << 10));
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }

    #[test]
    fn wide_each_supports_nested_and_concurrent_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let total = AtomicUsize::new(0);
        super::wide_each(16, |outer| {
            super::wide_each(8, |inner| {
                total.fetch_add(outer * 8 + inner + 1, Ordering::Relaxed);
            });
        });
        let expected: usize = (1..=128).sum();
        assert_eq!(total.load(Ordering::Relaxed), expected);

        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    let sum = AtomicUsize::new(0);
                    super::wide_each(32, |i| {
                        sum.fetch_add(i + 1, Ordering::Relaxed);
                    });
                    assert_eq!(sum.load(Ordering::Relaxed), (1..=32).sum::<usize>());
                });
            }
        });
    }

    #[test]
    fn stale_pages_in_with_taken_matches_exact_stale_rules() {
        let base = 100;
        let copied = [0u64, 5, 10, 10, 20, 0];
        let written = [(99, 50), (100, 1), (101, 5), (102, 11), (104, 19), (105, 3), (110, 99)];
        let got = super::stale_pages_in_with_taken(
            base,
            &copied,
            100 * super::PAGE,
            106 * super::PAGE,
            &written,
        );
        // 100: copied=0 -> stale; 101: 5 <= 5 -> clean; 102: 11 > 10 -> stale;
        // 103: not written -> clean; 104: 19 <= 20 -> clean; 105: copied=0 -> stale.
        assert_eq!(got, vec![(100, 0), (102, 10), (105, 0)]);
    }

    #[test]
    fn merge_page_parts_narrows_to_dirty_4k_subpages() {
        let mut title = vec![0x11u8; 2 * super::PAGE];
        let align = (title.as_ptr() as usize).next_multiple_of(super::PAGE) - title.as_ptr() as usize;
        let title_base = title.as_ptr() as usize + align;
        let mut held = vec![0x11u8; super::PAGE];
        // Modify 4 KiB parts 2 and 11 in title memory (and also part 5 which we
        // leave masked out to prove only the masked 4 KiB parts are read/merged).
        title[align + 2 * super::KERNEL_PAGE + 16..align + 2 * super::KERNEL_PAGE + 20].copy_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]);
        title[align + 5 * super::KERNEL_PAGE + 8..align + 5 * super::KERNEL_PAGE + 12].copy_from_slice(&[0x99; 4]);
        title[align + 11 * super::KERNEL_PAGE + 64..align + 11 * super::KERNEL_PAGE + 68].copy_from_slice(&[0x12, 0x34, 0x56, 0x78]);
        let shadow = super::Shadow::unbacked(title_base as u64, held.as_mut_ptr(), super::PAGE as u64, true);
        let left = std::sync::atomic::AtomicI64::new(64);
        let page = title_base / super::PAGE;
        let parts = (1u16 << 2) | (1u16 << 11);
        // SAFETY: `held` matches `shadow.len` and is exclusively owned by this test.
        let runs = unsafe {
            shadow.merge_page_parts_under_writes(page, title_base, title_base + super::PAGE, parts, held.as_mut_ptr(), &left)
        }
        .expect("within run budget")
        .expect("readable memory");
        assert_eq!(
            runs,
            vec![(2 * super::KERNEL_PAGE + 16, 4), (11 * super::KERNEL_PAGE + 64, 4)]
        );
        assert_eq!(&held[2 * super::KERNEL_PAGE + 16..2 * super::KERNEL_PAGE + 20], &[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(&held[11 * super::KERNEL_PAGE + 64..11 * super::KERNEL_PAGE + 68], &[0x12, 0x34, 0x56, 0x78]);
        assert_eq!(&held[5 * super::KERNEL_PAGE + 8..5 * super::KERNEL_PAGE + 12], &[0x11; 4]);
    }

    #[test]
    fn pack_take_chunks_groups_small_runs_and_splits_large_runs() {
        // 32 small 4 KiB runs (128 KiB total) should pack into ~2 chunks of 64 KiB, not 32 jobs.
        let small: Vec<(usize, usize, bool)> = (0..32)
            .map(|i| {
                let a = 0x1000_0000 + i * 2 * super::KERNEL_PAGE;
                (a, a + super::KERNEL_PAGE, true)
            })
            .collect();
        let (flat, chunks) = super::Shadow::pack_take_chunks(&small, 64 << 10);
        assert_eq!(flat, small);
        assert_eq!(chunks.len(), 2);

        // One large 512 KiB run splits cleanly into 4 page-aligned chunks of 128 KiB.
        let base = 0x2000_0000;
        let (flat, chunks) = super::Shadow::pack_take_chunks(&[(base, base + (512 << 10), false)], 128 << 10);
        assert_eq!(chunks.len(), 4);
        let covered: usize = flat.iter().map(|r| r.1 - r.0).sum();
        assert_eq!(covered, 512 << 10);
        for w in flat.windows(2) {
            assert_eq!(w[0].1, w[1].0);
            assert_eq!(w[0].1 % super::PAGE, 0);
        }
    }

    #[test]
    fn split_copy_chunks_coalesces_adjacent_ranges() {
        let copies = vec![
            (0x1000, 0x5000, 4096),
            (0x2000, 0x6000, 4096),
            (0x4000, 0x8000, 4096),
        ];
        let (flat, chunks) = super::split_copy_chunks(&copies, 64 << 10);
        assert_eq!(chunks, vec![0..2]);
        assert_eq!(flat, vec![(0x1000, 0x5000, 8192), (0x4000, 0x8000, 4096)]);
    }

    #[test]
    fn take_all_plain_parallel_copies_all_runs_and_records_dirty() {
        assert!(crate::safecopy::on());
        let len: usize = 512 << 10;
        let title: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) ^ (i >> 8)) as u8).collect();
        let mut held = vec![0u8; len];
        let base = title.as_ptr() as usize;
        let shadow = super::Shadow::unbacked(base as u64, held.as_mut_ptr(), len as u64, true);
        // Copy every even 4 KiB page (256 KiB total >= LONG) to exercise parallel chunked `take_all`.
        let runs: Vec<(usize, usize, bool)> = (0..(len / super::KERNEL_PAGE))
            .step_by(2)
            .map(|p| (base + p * super::KERNEL_PAGE, base + (p + 1) * super::KERNEL_PAGE, false))
            .collect();
        let copied = shadow.take_all(runs);
        assert_eq!(copied, len / 2);
        for p in 0..(len / super::KERNEL_PAGE) {
            let lo = p * super::KERNEL_PAGE;
            let hi = lo + super::KERNEL_PAGE;
            if p % 2 == 0 {
                assert_eq!(&held[lo..hi], &title[lo..hi]);
            } else {
                assert!(held[lo..hi].iter().all(|&b| b == 0));
            }
        }
        let dirty = shadow.take_dirty();
        assert_eq!(dirty.len(), len / (2 * super::KERNEL_PAGE));
        assert_eq!(dirty.iter().map(|r| r.1 - r.0).sum::<usize>(), len / 2);
    }

    #[test]
    fn copy_stale_pages_retains_two_distinct_equal_ranged_slots() {
        assert!(crate::safecopy::on());
        let len = 4 * super::PAGE;
        let raw: Vec<u8> = (0..len + super::PAGE).map(|i| (i & 0xff) as u8).collect();
        let align = (raw.as_ptr() as usize).next_multiple_of(super::PAGE) - raw.as_ptr() as usize;
        let base = raw.as_ptr() as usize + align;
        let mut held = vec![0u8; len];
        let shadow = super::Shadow::unbacked(base as u64, held.as_mut_ptr(), len as u64, false);
        let a = (base, base + 2 * super::PAGE);
        let b = (base + 2 * super::PAGE, base + 4 * super::PAGE);
        {
            let mut copied = shadow.copied.lock().unwrap();
            let stale_a = vec![(a.0 / super::PAGE, 0), (a.0 / super::PAGE + 1, 0)];
            shadow.copy_stale_pages(&mut copied, None, a.0, a.1, 10, 0, &stale_a);
            let stale_b = vec![(b.0 / super::PAGE, 0), (b.0 / super::PAGE + 1, 0)];
            shadow.copy_stale_pages(&mut copied, None, b.0, b.1, 20, 0, &stale_b);
            // Both equal-sized ranges A and B must remain cached in `copied.1`.
            assert_eq!(copied.1[0], (b.0, b.1, 20));
            assert_eq!(copied.1[1], (a.0, a.1, 10));
            // Refreshing A again updates slot 1 without evicting B in slot 0.
            shadow.copy_stale_pages(&mut copied, None, a.0, a.1, 30, 10, &[]);
            assert_eq!(copied.1[0], (b.0, b.1, 20));
            assert_eq!(copied.1[1], (a.0, a.1, 30));
        }
        assert_eq!(&held[..], &raw[align..align + len]);
    }

    #[test]
    fn card_room_kept_trims_phantom_headroom() {
        const MIB: u64 = 1 << 20;
        // Respects smaller card-buffer caps (e.g. 512 MiB on <=12 GB GPUs).
        assert_eq!(super::card_room_kept_for(512 * MIB, 100 * MIB, 0), 412 * MIB);
        // Covers the ~738-762 MiB burst on 16+ GB GPUs without locking 286 MiB phantom.
        assert_eq!(super::card_room_kept_for(2048 * MIB, 0, 0), 800 * MIB);
        assert_eq!(super::card_room_kept_for(2048 * MIB, 738 * MIB, 0), 62 * MIB);
        assert_eq!(super::card_room_kept_for(2048 * MIB, 738 * MIB, 100 * MIB), 0);
        // Post-settle decay holds full reserve before buffers exist or <=5s, decays over 5..10s, hits 0 at 10s.
        assert_eq!(super::card_room_kept_decayed(2048 * MIB, 0, 0, 20_000), 800 * MIB);
        assert_eq!(super::card_room_kept_decayed(2048 * MIB, 600 * MIB, 0, 5_000), 200 * MIB);
        assert_eq!(super::card_room_kept_decayed(2048 * MIB, 600 * MIB, 0, 7_500), 100 * MIB);
        assert_eq!(super::card_room_kept_decayed(2048 * MIB, 600 * MIB, 0, 10_000), 0);
    }

    #[test]
    fn card_merge_notes_landing_and_avoids_self_echo_dirty_upload() {
        assert!(crate::safecopy::on());
        let len = 4096usize;
        let mut title = vec![0x11u8; len];
        let mut held = vec![0x11u8; len];
        let mut gpu = vec![0x11u8; len];
        gpu[64..128].fill(0x22);
        let shadow = super::Shadow::unbacked(title.as_mut_ptr() as u64, held.as_mut_ptr(), len as u64, true);
        let mut scratch = Vec::new();
        let written = shadow.card_merge(0, &gpu, &mut scratch);
        assert!(written > 0);
        assert_eq!(&title[64..128], &[0x22u8; 64]);
        assert_eq!(&held[64..128], &[0x22u8; 64]);
        // Pure GPU write (no concurrent title diff): must NOT mark `card.dirty` (no 2x self-echo!),
        // and MUST mark `self.landing` so `take_landed` skips echoing landed bytes back up.
        assert!(shadow.take_dirty().is_empty());
        assert!(shadow.landing.load(std::sync::atomic::Ordering::Acquire));

        // Now simulate a concurrent title write beside a GPU write: `card_differs != 0` MUST mark dirty.
        title[16..20].fill(0x33);
        gpu[64..128].fill(0x44);
        let written2 = shadow.card_merge(0, &gpu, &mut scratch);
        assert!(written2 > 0);
        assert_eq!(&title[16..20], &[0x33u8; 4]);
        assert_eq!(&title[64..128], &[0x44u8; 64]);
        // Only bytes 64..128 differed between `gpu` and `old`, so `at..at+n` is 64..128;
        // if title concurrently wrote INSIDE 64..128 on an unchanged byte, card_dirty triggers.
        title[70] = 0x77;
        gpu[64] = 0x55;
        gpu[127] = 0x55;
        let _ = shadow.card_merge(0, &gpu, &mut scratch);
        assert!(!shadow.take_dirty().is_empty());
    }

    /// Helper that creates an unbacked `Shadow` (with optional fake `InPlace`)
    /// leaked once via `mem::forget` so fake COM pointers are never released.
    fn dummy_shadow(start: u64, len: u64, in_place: bool) -> std::sync::Arc<super::Shadow> {
        use windows::Win32::Graphics::Direct3D12::{ID3D12Heap, ID3D12Resource};
        // SAFETY: dummy interface pointers never dereferenced by unit tests;
        // `mem::forget` prevents `Drop` from ever invoking `Release`.
        let buffer: ID3D12Resource = unsafe { std::mem::transmute(8usize) };
        let mut shadow = super::Shadow::new(buffer, std::ptr::null_mut(), start, start, len);
        if in_place {
            let heap: ID3D12Heap = unsafe { std::mem::transmute(16usize) };
            shadow.place = Some(super::InPlace {
                heap,
                bias: 0,
                _view: None,
                pinned: 0,
                _private_guard: None,
            });
            super::IN_PLACE_BOUNDS[0].fetch_min(start, std::sync::atomic::Ordering::Relaxed);
            super::IN_PLACE_BOUNDS[1].fetch_max(start + len, std::sync::atomic::Ordering::Relaxed);
        }
        let shadow = std::sync::Arc::new(shadow);
        std::mem::forget(shadow.clone());
        shadow
    }

    fn insert_dummy_buffer(res: usize, start: u64, len: u64, shadow: std::sync::Arc<super::Shadow>) {
        let mut map = super::buffers_mut();
        map.insert(
            (start, res),
            super::Entry {
                end: start + len,
                shadow: Some(shadow),
            },
        );
        super::bump_shadow_epoch();
    }

    static SPAN_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn tls_span_cache_hits_ring_offsets_and_respects_boundaries() {
        let _guard = SPAN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = 0x7_F100_0000_0000u64;
        let len = 0x10_0000u64;
        let res = 0xDEAD_0001usize;
        let shadow = dummy_shadow(base, len, true);
        insert_dummy_buffer(res, base, len, shadow.clone());

        // First lookup populates the TLS span cache.
        let got = super::find(base + 0x100, 256).expect("hit");
        assert!(std::sync::Arc::ptr_eq(&got, &shadow));
        drop(got);

        // Distinct offsets/lengths inside the same span hit TLS directly (verified via `span_find`).
        for off in [0u64, 0x200, 0x4_1234, len - 64] {
            let hit = super::span_find(base + off, base + off + 64, false).expect("TLS span hit");
            assert!(std::sync::Arc::ptr_eq(&hit, &shadow));
        }

        // `with_in_place` borrows directly from TLS without incrementing `Arc::strong_count`.
        let count_before = std::sync::Arc::strong_count(&shadow);
        let id = super::with_in_place(base + 0x8000, 256, |s| {
            assert_eq!(std::sync::Arc::strong_count(s), count_before);
            s.id()
        });
        assert_eq!(id, Some(shadow.id()));

        // Boundary misses: before start, past end, and straddling end must miss the TLS span cache.
        assert!(super::span_find(base - 16, base + 16, false).is_none());
        assert!(super::span_find(base + len - 16, base + len + 16, false).is_none());
        assert!(super::span_find(base + len, base + len + 1, true).is_none());

        // Clean up and verify `unregister` bumps `SHADOW_EPOCH` and invalidates TLS.
        let epoch_before = super::SHADOW_EPOCH.load(std::sync::atomic::Ordering::Acquire);
        super::unregister(res, base);
        assert!(super::SHADOW_EPOCH.load(std::sync::atomic::Ordering::Acquire) > epoch_before);
        assert!(shadow.is_dead());
        assert!(super::span_find(base + 0x100, base + 0x200, false).is_none());
    }

    #[test]
    fn tls_span_cache_handles_nested_buffers_and_epoch_invalidation() {
        let _guard = SPAN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = 0x7_F200_0000_0000u64;
        let outer_len = 0x10_0000u64;
        let inner_off = 0x4_0000u64;
        let inner_len = 0x1_0000u64;
        let outer_res = 0xDEAD_0010usize;
        let inner_res = 0xDEAD_0011usize;

        let outer = dummy_shadow(base, outer_len, false);
        insert_dummy_buffer(outer_res, base, outer_len, outer.clone());

        // Non-in-place shadow returns None from `with_in_place` while populating TLS.
        assert!(super::with_in_place(base + 0x1000, 256, |_| ()).is_none());
        assert!(super::span_find(base + inner_off + 0x100, base + inner_off + 0x200, false).is_some());

        // Registering an inner buffer bumps `SHADOW_EPOCH`, invalidating the full outer span.
        let inner = dummy_shadow(base + inner_off, inner_len, true);
        insert_dummy_buffer(inner_res, base + inner_off, inner_len, inner.clone());
        assert!(super::span_find(base + 0x1000, base + 0x1100, false).is_none());

        // Lookup below inner caches only `[base, base + inner_off)`, never shadowing inner.
        let below = super::find(base + 0x1000, 256).expect("outer below");
        assert!(std::sync::Arc::ptr_eq(&below, &outer));
        assert!(super::span_find(base + inner_off + 0x100, base + inner_off + 0x200, false).is_none());

        // Lookup inside inner resolves to `inner` and caches `[inner_start, inner_end)`.
        let inside = super::find(base + inner_off + 0x100, 256).expect("inner");
        assert!(std::sync::Arc::ptr_eq(&inside, &inner));

        // Lookup above inner caches `[inner_end, outer_end)` and resolves to `outer`.
        let above = super::find(base + inner_off + inner_len + 0x100, 256).expect("outer above");
        assert!(std::sync::Arc::ptr_eq(&above, &outer));
        assert!(super::span_find(base + outer_len - 256, base + outer_len, false).is_some());

        // `in_place_let_go` evicts `inner`, marks it dead, and invalidates TLS.
        let evicted = super::in_place_let_go(base + inner_off, inner_len);
        assert_eq!(evicted, 1);
        assert!(inner.is_dead());
        assert!(super::span_find(base + inner_off + 0x100, base + inner_off + 0x200, false).is_none());

        // Clean up.
        super::unregister(inner_res, base + inner_off);
        super::unregister(outer_res, base);
    }

    #[test]
    fn cpu_staging_decommit_range_and_recommit_round_trip() {
        use std::sync::atomic::Ordering;
        let before_global = super::cpu_staging_committed_bytes();
        let total_size = 16u64 << 20;
        let (staging, ptr) = super::CpuStaging::alloc(total_size).expect("MEM_RESERVE");

        // Commit pieces 0..8 (512 KiB) and write distinct sentinel values.
        assert!(staging.ensure_committed(0, 8 * super::PIECE));
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 8 * super::PIECE as u64);
        for p in 0..8usize {
            // SAFETY: pieces 0..8 were committed above.
            unsafe { ptr.add(p * super::PIECE + 16).write((0xA0 + p) as u8) };
        }

        // Decommit interior pieces of `[2*PIECE + 123, 6*PIECE - 200)` => only pieces 3..5.
        let freed = staging.decommit_range(2 * super::PIECE + 123, 4 * super::PIECE - 323);
        assert_eq!(freed, 2 * super::PIECE);
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 6 * super::PIECE as u64);
        assert!(staging.is_piece_committed(2));
        assert!(!staging.is_piece_committed(3));
        assert!(!staging.is_piece_committed(4));
        assert!(staging.is_piece_committed(5));

        // `read_committed_into` must read committed pieces accurately and return zeros for
        // decommitted pieces without faulting or forcing OS re-commit.
        let mut probe = vec![0xFFu8; 4 * super::PIECE];
        staging.read_committed_into(2 * super::PIECE, &mut probe);
        assert_eq!(probe[16], 0xA2);
        assert!(probe[super::PIECE..3 * super::PIECE].iter().all(|&b| b == 0));
        assert_eq!(probe[3 * super::PIECE + 16], 0xA5);
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 6 * super::PIECE as u64);

        // Re-commit pieces 3..5 on demand at the exact same base pointer and verify round-trip.
        assert!(staging.ensure_committed(3 * super::PIECE, 5 * super::PIECE));
        assert_eq!(staging.committed_bytes.load(Ordering::Relaxed), 8 * super::PIECE as u64);
        assert!(staging.is_piece_committed(3));
        assert!(staging.is_piece_committed(4));
        // SAFETY: pieces 3..5 re-committed above.
        unsafe {
            ptr.add(3 * super::PIECE + 16).write(0x7E);
            ptr.add(4 * super::PIECE + 16).write(0x8D);
            assert_eq!(ptr.add(3 * super::PIECE + 16).read(), 0x7E);
            assert_eq!(ptr.add(4 * super::PIECE + 16).read(), 0x8D);
            assert_eq!(ptr.add(2 * super::PIECE + 16).read(), 0xA2);
        }

        drop(staging);
        assert_eq!(super::cpu_staging_committed_bytes(), before_global);
    }

    #[test]
    fn loose_two_phase_concurrent_creation_never_blocks_different_keys() {
        use std::collections::BTreeMap;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier, RwLock};

        let table: Arc<RwLock<BTreeMap<(u64, u64), super::LooseEntry>>> = Arc::new(RwLock::new(BTreeMap::new()));
        let entered_k1 = Arc::new(Barrier::new(2));
        let release_k1 = Arc::new(Barrier::new(2));
        let k1_creates = Arc::new(AtomicUsize::new(0));
        let k1 = (0x7_F300_0000_0000u64, 512 << 10);
        let k2 = (0x7_F300_0010_0000u64, 512 << 10);
        let k3 = (0x7_F300_0020_0000u64, 512 << 10);

        std::thread::scope(|s| {
            // Thread A begins slow creation of K1 and blocks inside `create()`.
            let t_a = {
                let table = table.clone();
                let entered = entered_k1.clone();
                let release = release_k1.clone();
                let creates = k1_creates.clone();
                s.spawn(move || {
                    super::loose_get_or_create_in(
                        &table,
                        k1,
                        || {
                            creates.fetch_add(1, Ordering::Relaxed);
                            entered.wait();
                            release.wait();
                            Some(dummy_shadow(k1.0, k1.1, false))
                        },
                        |_| {},
                    )
                    .expect("k1 created")
                })
            };

            // Wait until Thread A is inside `create(K1)` (proving `LOOSE.write()` was dropped).
            entered_k1.wait();

            // Thread B requests DIFFERENT key K2 while K1 is still blocked: MUST finish immediately!
            let (s_k2, created_k2) = super::loose_get_or_create_in(
                &table,
                k2,
                || Some(dummy_shadow(k2.0, k2.1, false)),
                |_| {},
            )
            .expect("k2 created concurrently without blocking on k1");
            assert!(created_k2);
            assert_eq!(s_k2.span(), k2);

            // Thread C requests SAME key K1: waits on `PendingLoose`, shares Thread A's result.
            let t_c = {
                let table = table.clone();
                let creates = k1_creates.clone();
                s.spawn(move || {
                    super::loose_get_or_create_in(
                        &table,
                        k1,
                        || {
                            creates.fetch_add(1, Ordering::Relaxed);
                            Some(dummy_shadow(k1.0, k1.1, false))
                        },
                        |_| {},
                    )
                    .expect("k1 shared")
                })
            };

            // Unblock Thread A so both A and C receive the same `Arc<Shadow>`.
            release_k1.wait();
            let (s_k1_a, created_a) = t_a.join().unwrap();
            let (s_k1_c, created_c) = t_c.join().unwrap();
            assert!(created_a);
            assert!(!created_c);
            assert!(Arc::ptr_eq(&s_k1_a, &s_k1_c));
            assert_eq!(k1_creates.load(Ordering::Relaxed), 1, "K1 created exactly once");
        });

        // Failed creation must remove the pending cell cleanly so a retry succeeds and never wedges.
        let failed = super::loose_get_or_create_in(&table, k3, || None, |_| {});
        assert!(failed.is_none());
        let (s_k3, created_k3) = super::loose_get_or_create_in(
            &table,
            k3,
            || Some(dummy_shadow(k3.0, k3.1, false)),
            |_| {},
        )
        .expect("retry after failed creation succeeds");
        assert!(created_k3);
        assert_eq!(s_k3.span(), k3);
    }

    #[test]
    fn mode2_piece_promotion_localized_demotion_and_safecopy_never_self_trips() {
        use std::sync::atomic::Ordering;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        assert!(crate::safecopy::on());

        const PIECES: usize = 16;
        const LEN: usize = PIECES * super::PIECE;
        // SAFETY: dedicated 1 MiB guest allocation owned by this test.
        let guest = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        assert!(!guest.is_null());
        let mut staging = vec![0u8; LEN];
        for p in 0..PIECES {
            let pat = (0x30 + p) as u8;
            // SAFETY: inside `guest`.
            unsafe { std::ptr::write_bytes(guest.add(p * super::PIECE), pat, super::PIECE) };
            staging[p * super::PIECE..(p + 1) * super::PIECE].fill(pat ^ 0x55);
        }

        let shadow = super::Shadow::unbacked(guest as u64, staging.as_mut_ptr(), LEN as u64, true);
        let run_a_ranges: Vec<(usize, usize)> = (0..4).map(|p| (p * super::PIECE, (p + 1) * super::PIECE)).collect();
        let run_b_ranges: Vec<(usize, usize)> = (8..12).map(|p| (p * super::PIECE, (p + 1) * super::PIECE)).collect();

        // 1 fetch is below threshold (2): nothing promoted yet.
        shadow.gpu_only_note_fetched_mode2(&run_a_ranges, 2, true);
        shadow.gpu_only_note_fetched_mode2(&run_b_ranges, 2, true);
        assert!(!shadow.any_piece_gpu_only());

        // 2nd consecutive GPU writeback promotes BOTH disjoint 4-piece runs (0..4 and 8..12).
        shadow.gpu_only_note_fetched_mode2(&run_a_ranges, 2, true);
        shadow.gpu_only_note_fetched_mode2(&run_b_ranges, 2, true);
        for p in 0..PIECES {
            let expected = (0..4).contains(&p) || (8..12).contains(&p);
            assert_eq!(shadow.piece_is_gpu_only(p), expected, "piece {p}");
        }
        assert!(super::is_tripwired(guest as usize, 4 * super::PIECE));
        assert!(!super::is_tripwired(guest as usize + 4 * super::PIECE, 4 * super::PIECE));
        assert!(super::is_tripwired(guest as usize + 8 * super::PIECE, 4 * super::PIECE));

        // `card_span` skips fully promoted runs and trims promoted leading/trailing pieces.
        assert!(shadow.card_span(guest as usize, guest as usize + 4 * super::PIECE).is_none());
        let trimmed = shadow
            .card_span(guest as usize, guest as usize + 8 * super::PIECE)
            .expect("non-promoted tail remains");
        assert_eq!(trimmed.first, 4);
        assert_eq!(trimmed.pieces, 4);

        // `land::ordered` filters out promoted Mode-2 pieces before GPU readback.
        let keys = vec![
            (shadow.id(), guest as usize + super::PIECE, super::PIECE),     // promoted (skip)
            (shadow.id(), guest as usize + 5 * super::PIECE, super::PIECE), // non-promoted (keep)
            (shadow.id(), guest as usize + 9 * super::PIECE, super::PIECE), // promoted (skip)
        ];
        let ordered = crate::land::ordered(&keys);
        assert_eq!(ordered.len(), 1);
        assert_eq!(ordered[0].1, guest as usize + 5 * super::PIECE);

        // Spec-Trap #4: internal `safecopy::copy` / `probe_readable` / `host::read_into`
        // must read from host-side `shadow.data` WITHOUT disarming or demoting `0..4`!
        let mut dst = vec![0u8; 2 * super::PIECE];
        assert!(crate::host::read_into(unsafe { guest.add(super::PIECE) }, &mut dst));
        assert_eq!(&dst[..super::PIECE], &staging[super::PIECE..2 * super::PIECE]);
        assert_eq!(&dst[super::PIECE..], &staging[2 * super::PIECE..3 * super::PIECE]);
        // Cross-boundary read spanning promoted piece 3 and non-promoted piece 4:
        let mut cross = vec![0u8; 2 * super::PIECE];
        assert!(crate::host::read_into(unsafe { guest.add(3 * super::PIECE) }, &mut cross));
        assert_eq!(&cross[..super::PIECE], &staging[3 * super::PIECE..4 * super::PIECE]);
        assert!(cross[super::PIECE..].iter().all(|&b| b == (0x30 + 4)));
        // `probe_readable` must also succeed without tripping!
        assert!(unsafe { crate::safecopy::probe_readable(guest, 4 * super::PIECE, 4096) });
        // Run 0..4 MUST STILL BE PROMOTED AND TRIPWIRED (zero self-trip)!
        assert!(shadow.piece_is_gpu_only(1));
        assert!(super::is_tripwired(guest as usize, 4 * super::PIECE));

        // Now trigger a REAL guest CPU read on piece 1 (inside run 0..4):
        // VEH handler disarms and locally demotes ONLY run 0..4, keeping run 8..12 promoted!
        // SAFETY: `watch` VEH handler restores PAGE_READWRITE and retries on fault.
        let guest_byte = unsafe { guest.add(super::PIECE + 64).read_volatile() };
        assert_eq!(guest_byte, 0x31);
        for p in 0..4 {
            assert!(!shadow.piece_is_gpu_only(p), "faulted run piece {p} demoted");
        }
        for p in 8..12 {
            assert!(shadow.piece_is_gpu_only(p), "disjoint run piece {p} stays promoted");
        }
        assert!(!super::is_tripwired(guest as usize, 4 * super::PIECE));
        assert!(super::is_tripwired(guest as usize + 8 * super::PIECE, 4 * super::PIECE));

        // Also verify `asked_ahead()` returns `false` when an entire buffer is tripwired.
        let all_len = 4 * super::PIECE;
        let guest2 = unsafe { VirtualAlloc(std::ptr::null(), all_len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        assert!(!guest2.is_null());
        let mut staging2 = vec![0u8; all_len];
        let shadow2 = super::Shadow::unbacked(guest2 as u64, staging2.as_mut_ptr(), all_len as u64, true);
        shadow2.watched.store(1, Ordering::Relaxed);
        assert!(shadow2.asked_ahead());
        let all_ranges: Vec<(usize, usize)> = (0..4).map(|p| (p * super::PIECE, (p + 1) * super::PIECE)).collect();
        shadow2.gpu_only_note_fetched_mode2(&all_ranges, 1, true);
        assert!(shadow2.gpu_only_tripwired());
        assert!(!shadow2.asked_ahead(), "fully tripwired buffer skips GetWriteWatch");

        shadow.disarm_all_tripwires();
        shadow2.disarm_all_tripwires();
        // SAFETY: test-owned VirtualAlloc allocations.
        unsafe {
            VirtualFree(guest.cast(), 0, MEM_RELEASE);
            VirtualFree(guest2.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn spec_trap_8_mutual_exclusion_between_inplace_private_and_mode2_tripwires() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if windows::core::Interface::cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device13>(&device).is_err() {
            return;
        }
        assert!(crate::safecopy::on());

        const LEN: usize = 8 * super::PIECE;
        // SAFETY: dedicated committed MEM_PRIVATE allocation owned by this test.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        assert!(!mem.is_null());

        // Case A: Active Mode-2 tripwire on `[mem, mem + 4*PIECE)` blocks `inplace::place_part`
        // AND `port::place_in_with` from opening `OpenExistingHeapFromAddress1` on those pages!
        let mut staging = vec![0u8; LEN];
        let shadow = super::Shadow::unbacked(mem as u64, staging.as_mut_ptr(), LEN as u64, true);
        let run_a: Vec<(usize, usize)> = (0..4).map(|p| (p * super::PIECE, (p + 1) * super::PIECE)).collect();
        shadow.gpu_only_note_fetched_mode2(&run_a, 1, true);
        assert!(super::overlaps_gpu_only_or_tripwired(mem as u64, 4 * super::PIECE as u64));
        assert!(
            crate::inplace::place_part(&device, mem as u64, super::PIECE as u64).is_err(),
            "inplace::place_part must refuse tripwired pages (Spec-Trap #8)"
        );
        assert!(
            crate::port::place_in_with(&device, mem as u64, super::PIECE as u64, true, true).is_err(),
            "port::place_in_with must refuse tripwired pages (Spec-Trap #8)"
        );

        // Case B: Active `inplace::place_part` on `[mem + 4*PIECE, mem + 8*PIECE)` blocks
        // Mode-2 from arming a `PAGE_NOACCESS` tripwire on pieces 4..8!
        let placed = crate::inplace::place_part(&device, mem as u64 + 4 * super::PIECE as u64, 4 * super::PIECE as u64)
            .expect("disjoint private pages open cleanly");
        assert!(crate::inplace::overlaps_private_placed(
            mem as u64 + 4 * super::PIECE as u64,
            4 * super::PIECE as u64
        ));
        let run_b: Vec<(usize, usize)> = (4..8).map(|p| (p * super::PIECE, (p + 1) * super::PIECE)).collect();
        shadow.gpu_only_note_fetched_mode2(&run_b, 1, true);
        assert!(!shadow.piece_is_gpu_only(4), "Mode-2 must refuse arming over active inplace private heap");
        assert!(!super::is_tripwired(mem as usize + 4 * super::PIECE, 4 * super::PIECE));

        drop(placed);
        shadow.disarm_all_tripwires();
        assert!(!crate::inplace::overlaps_private_placed(
            mem as u64 + 4 * super::PIECE as u64,
            4 * super::PIECE as u64
        ));
        // SAFETY: all heaps and tripwires released above.
        unsafe { VirtualFree(mem.cast(), 0, MEM_RELEASE) };
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter_map(LooseEntry::as_ready)
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
