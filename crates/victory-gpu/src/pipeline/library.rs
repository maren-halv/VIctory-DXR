//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Under one lock of
//! ours for every call, the eight workers made pipelines one at a time:
//! 87 to 96 of 110 looks at each worker found it waiting for that lock
//! (run 49), and an earlier run's 2,300 pipelines, all in the library, were
//! still being loaded 46 s into the opening cutscene.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes it was made from for as long as it lives,
//! and read into a buffer they were 1,352 MB of the process's own memory
//! for the whole run (run 60), all of it in physical memory. Mapped, the
//! pages are the file's: only those the host reads are in memory, and the
//! system can drop them without writing them anywhere. A save writes the
//! new library through a mapping of the new file in the same way (a buffer
//! of the library's size was made, filled and written out at each save).
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.
//!
//! The mapped file's handle stays open for as long as the mapping lives
//! ([`holds`]). A save renames the new library over the file this run has
//! mapped, and with the handle closed Windows then gives up the old file's
//! contents: every page of the mapping that is not in memory at that
//! moment, or leaves memory later (the trim below, or Windows taking pages
//! back), reads as zeros from then on. The host's library reads such pages
//! as stored pipelines; the driver, handed a pipeline of zeros, removes the
//! device with DXGI_ERROR_DRIVER_INTERNAL_ERROR (runs 72, 73 and 82, each
//! within 5 ms of the first library call after a save and a trim; and the
//! library run 77 saved from such pages was refused by run 78). Measured
//! with a 64 MB file written past the system's cache, mapped and replaced:
//! 16,352 of 16,384 pages read zeros with the handle closed, none with it
//! open (the test below). `VICTORY_LIBRARY_HANDLE=0`: closed, as before.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The folder the library is kept in.
pub(super) fn folder() -> Option<PathBuf> {
    DIR.get().cloned()
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// For the memory lines: the bytes of the library's file, kept in memory
/// while the library lives, and the bytes the last save wrote (a buffer of
/// that size is made, filled and written every time pipelines were added,
/// at most every 20 s).
pub(crate) static BLOB: AtomicU64 = AtomicU64::new(0);
pub(crate) static SAVED: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's handle is kept open while it is
/// mapped (see the module's notes: without it a save in the same run turns
/// the mapping's pages into zeros as they leave memory). On by default;
/// `VICTORY_LIBRARY_HANDLE=0` closes it after mapping, as before.
fn holds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_HANDLE").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`] ([`holds`]): open
    /// until after the view is unmapped, so that the file a rename took
    /// the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages
        // after the mapping object and the file are closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(mapping, if write { FILE_MAP_WRITE } else { FILE_MAP_READ }, 0, 0, 0);
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped { at, len, _file: None })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        Mapped::open_holding(path, holds())
    }

    /// As [`Mapped::open`]; `hold` says whether the file stays open with
    /// the mapping.
    fn open_holding(path: &std::path::Path, hold: bool) -> Option<Mapped> {
        // Opened as the standard library opens a file to read: others may
        // read, write, rename and delete it meanwhile, so a save still
        // renames the new library over this one.
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        if hold {
            mapped._file = Some(file);
        }
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    fn of(dir: &std::path::Path) -> Blob {
        let path = dir.join(LIBRARY_FILE);
        let next = dir.join(NEXT_FILE);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!("the host pipeline library saved last could not take the old one's place");
        }
        if mapped()
            && let Some(file) = Mapped::open(&path)
        {
            return Blob::Mapped(file);
        }
        match std::fs::read(&path) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

/// The mapped library file's view (address, bytes; 0: not mapped), when a
/// pipeline was last loaded from the library (`crate::host::now_ms`), and
/// whether the view's pages were let go since ([`trim`]).
static VIEW: [std::sync::atomic::AtomicUsize; 2] = [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the view let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the view's pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new library over the
/// mapped file: for [`removal_note`].
static TRIMMED_AT: AtomicU64 = AtomicU64::new(0);
static REPLACED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Fixed slots for tracking in-flight library calls without global lock contention
/// or per-call string formatting (only formatted inside [`removal_note`]).
const CALL_SLOTS: usize = 128;

enum CallName {
    Inline([u8; 64], u8),
    Heap(String),
}

impl CallName {
    fn new(name: &str) -> Self {
        let bytes = name.as_bytes();
        if bytes.len() <= 64 {
            let mut buf = [0u8; 64];
            buf[..bytes.len()].copy_from_slice(bytes);
            Self::Inline(buf, bytes.len() as u8)
        } else {
            Self::Heap(name.to_owned())
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Inline(buf, len) => std::str::from_utf8(&buf[..usize::from(*len)]).unwrap_or(""),
            Self::Heap(s) => s,
        }
    }
}

struct ActiveCall {
    thread: u32,
    what: &'static str,
    name: CallName,
    label: Option<Arc<str>>,
    since: u64,
}

struct CallSlot {
    in_use: AtomicBool,
    entry: Mutex<Option<ActiveCall>>,
}

static CALL_SLOTS_TABLE: [CallSlot; CALL_SLOTS] = [const {
    CallSlot {
        in_use: AtomicBool::new(false),
        entry: Mutex::new(None),
    }
}; CALL_SLOTS];
static CALL_OVERFLOW: Mutex<Vec<ActiveCall>> = Mutex::new(Vec::new());

thread_local! {
    /// The pipeline this thread is making (its shaders), for [`CALL_SLOTS_TABLE`].
    static LABEL: std::cell::RefCell<Option<Arc<str>>> = const { std::cell::RefCell::new(None) };
}

/// Names the pipeline this thread makes next.
pub(crate) fn label(text: String) {
    let arc: Arc<str> = Arc::from(text);
    LABEL.with(|l| *l.borrow_mut() = Some(arc));
}

/// A library call of this thread, listed in [`CALL_SLOTS_TABLE`] until dropped.
struct Call {
    slot: Option<usize>,
    thread: u32,
    since: u64,
}

fn call(what: &'static str, name: &str) -> Call {
    let (thread, since) = (crate::host::thread_id(), crate::host::now_us());
    let label = LABEL.with(|l| l.borrow().clone());
    let active = ActiveCall {
        thread,
        what,
        name: CallName::new(name),
        label,
        since,
    };
    let start = (thread as usize).wrapping_mul(31) & (CALL_SLOTS - 1);
    for offset in 0..CALL_SLOTS {
        let idx = (start + offset) & (CALL_SLOTS - 1);
        let slot = &CALL_SLOTS_TABLE[idx];
        if slot
            .in_use
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            *slot.entry.lock().unwrap_or_else(|e| e.into_inner()) = Some(active);
            return Call {
                slot: Some(idx),
                thread,
                since,
            };
        }
    }
    CALL_OVERFLOW
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(active);
    Call {
        slot: None,
        thread,
        since,
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        if let Some(idx) = self.slot {
            let slot = &CALL_SLOTS_TABLE[idx];
            *slot.entry.lock().unwrap_or_else(|e| e.into_inner()) = None;
            slot.in_use.store(false, Ordering::Release);
        } else {
            CALL_OVERFLOW
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|c| (c.thread, c.since) != (self.thread, self.since));
        }
    }
}

/// For the report when the host device is removed: the library calls
/// running at that moment, whether this run replaced the mapped file and
/// let its pages go, and how many pages of the mapping read as zeros now.
/// A library holds no page of zeros (the 259 MB one of run 83 has no run of
/// 2,048 zero bytes): each such page is one whose contents were lost (see
/// the module's notes), and a pipeline stored there is what the driver was
/// handed.
pub(crate) fn removal_note() -> String {
    use std::fmt::Write;
    let now = crate::host::now_us();
    let mut text = String::from("pipeline library: ");
    {
        let mut count = 0usize;
        let mut write_active = |c: &ActiveCall| {
            let label = c.label.as_deref().unwrap_or("");
            let _ = write!(
                text,
                "{}{} of {} {} running for {:.1} ms on thread {}",
                if count == 0 { "" } else { ", " },
                c.what,
                c.name.as_str(),
                label,
                now.saturating_sub(c.since) as f64 / 1000.0,
                c.thread
            );
            count += 1;
        };
        for slot in &CALL_SLOTS_TABLE {
            if !slot.in_use.load(Ordering::Acquire) {
                continue;
            }
            let entry = slot.entry.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(c) = entry.as_ref() {
                write_active(c);
            }
        }
        let overflow = CALL_OVERFLOW.lock().unwrap_or_else(|e| e.into_inner());
        for c in overflow.iter() {
            write_active(c);
        }
        if count == 0 {
            text.push_str("no call running");
        }
    }
    let guard = LIBRARY.try_read().ok();
    let Some(lib) = guard.as_ref().and_then(|g| g.as_ref()) else {
        text.push_str("; its file is not mapped (library closed or idle-torn-down)");
        return text;
    };
    let Some(Blob::Mapped(mapped_file)) = lib._blob.as_ref() else {
        text.push_str("; its file is not mapped");
        return text;
    };
    let len = mapped_file.len;
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; its file is mapped ({len} bytes), the handle {}",
        if holds() { "kept open" } else { "closed (VICTORY_LIBRARY_HANDLE=0)" }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced the file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced the file under the mapping {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(text, "; its pages were last let go {:.1} s ago", ago(TRIMMED_AT.load(Ordering::Relaxed)));
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    let bytes = mapped_file.bytes();
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    for (n, page) in bytes.chunks(4096).enumerate() {
        if n % 4096 == 0 && started.elapsed() > std::time::Duration::from_secs(2) {
            break;
        }
        looked += 1;
        if page.iter().all(|&b| b == 0) {
            zero += 1;
            if first.is_none() {
                first = Some(n);
            }
        }
    }
    let _ = write!(text, "; {zero} of the {looked} pages looked at read as zeros now");
    if let Some(first) = first {
        let _ = write!(text, " (the first is page {first}): the library's contents were lost there");
    }
    text
}

/// Minimum mapped library file size (64 MiB) before [`trim`] releases working-set
/// pages via `VirtualUnlock` (skipping small files avoids pointless disk fault thrash).
const TRIM_MIN_BYTES: usize = 64 << 20;

fn trim_eligible(len: usize) -> bool {
    len >= TRIM_MIN_BYTES
}

/// Idle duration (ms) after the last load/store before [`trim`] flushes delta
/// additions and tears down `ID3D12PipelineLibrary1` + its mapped file
/// (`VICTORY_LIBRARY_TEARDOWN_MS`, default 30 s; `0` disables teardown).
fn teardown_idle_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_LIBRARY_TEARDOWN_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(30_000)
    })
}

static LAST_ACTIVITY_MS: AtomicU64 = AtomicU64::new(0);
static TEARING_DOWN: AtomicBool = AtomicBool::new(false);

fn note_activity() {
    LAST_ACTIVITY_MS.store(crate::host::now_ms(), Ordering::Relaxed);
}

fn teardown_eligible(now_ms: u64, last_activity_ms: u64, idle_ms: u64) -> bool {
    idle_ms > 0 && last_activity_ms > 0 && now_ms.saturating_sub(last_activity_ms) >= idle_ms
}

/// Lets the mapped library file's pages go from memory when idle (`>= 64 MiB`),
/// and tears down the entire `ID3D12PipelineLibrary1` + mapping cleanly once
/// pipeline activity has been idle for `VICTORY_LIBRARY_TEARDOWN_MS` (30 s).
/// Reopens automatically on demand if a later draw loads/stores a pipeline.
pub(crate) fn trim() {
    static TRIM_ON: OnceLock<bool> = OnceLock::new();
    let now = crate::host::now_ms();
    let idle_ms = teardown_idle_ms();
    let last = LAST_ACTIVITY_MS.load(Ordering::Relaxed);
    // 1. Idle teardown: flush any unsaved delta, then drop COM library BEFORE
    //    unmapping `Mapped` / closing `_file` (Spec-Trap #9, runs 72/73/82).
    if teardown_eligible(now, last, idle_ms)
        && LIBRARY.try_read().is_ok_and(|g| g.is_some())
        && !TEARING_DOWN.swap(true, Ordering::AcqRel)
    {
        let spawned = std::thread::Builder::new()
            .name("victory-library-teardown".into())
            .spawn(move || {
                struct Reset;
                impl Drop for Reset {
                    fn drop(&mut self) {
                        TEARING_DOWN.store(false, Ordering::Release);
                    }
                }
                let _reset = Reset;
                if has_added() || USED_CHANGED.load(Ordering::Relaxed) {
                    save(true);
                }
                let old = {
                    let mut guard = LIBRARY.write().unwrap_or_else(|e| e.into_inner());
                    let now = crate::host::now_ms();
                    let last = LAST_ACTIVITY_MS.load(Ordering::Relaxed);
                    let still_idle = teardown_eligible(now, last, idle_ms)
                        && guard.as_ref().is_some_and(|l| l.added.load(Ordering::Relaxed) == 0);
                    if !still_idle {
                        return;
                    }
                    VIEW[0].store(0, Ordering::Release);
                    VIEW[1].store(0, Ordering::Release);
                    BLOB.store(0, Ordering::Release);
                    TRIMMED.store(false, Ordering::Release);
                    guard.take()
                };
                if let Some(old) = old {
                    // Explicit `Library::drop` guarantees `ID3D12PipelineLibrary1`
                    // is released strictly before `_blob` (`Mapped` + `_file`).
                    drop(old);
                    crate::host::trail(|| "pipeline library: torn down after idle period".to_owned());
                }
            });
        if spawned.is_err() {
            TEARING_DOWN.store(false, Ordering::Release);
        }
        return;
    }

    // 2. Working-set trim via `VirtualUnlock` while library is still active (>= 64 MiB only).
    let (at, len) = (VIEW[0].load(Ordering::Relaxed), VIEW[1].load(Ordering::Relaxed));
    if at == 0
        || !trim_eligible(len)
        || TRIMMED.load(Ordering::Relaxed)
        || !*TRIM_ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TRIM").is_none_or(|v| v != "0"))
    {
        return;
    }
    let quiet_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    if now < LOADED_AT.load(Ordering::Relaxed) + quiet_ms
        || TRIMMED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let work = move || {
        // Hold `LIBRARY.read()` so idle teardown cannot unmap the view while
        // `VirtualUnlock` is inspecting it.
        let guard = LIBRARY.read().unwrap_or_else(|e| e.into_inner());
        let Some(Blob::Mapped(file)) = guard.as_ref().and_then(|l| l._blob.as_ref()) else {
            return;
        };
        let (view_at, view_len) = (file.at.Value, file.len);
        if !trim_eligible(view_len) {
            return;
        }
        // SAFETY: `guard` keeps the `Mapped` view alive for the duration of `VirtualUnlock`.
        unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(view_at, view_len) };
        TRIMMED_BYTES.fetch_add(view_len as u64, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| format!("pipeline library: the mapped file's pages let go from memory ({view_len} bytes)"));
    };
    if std::thread::Builder::new().name("victory-library-trim".into()).spawn(work).is_err() {
        TRIMMED.store(false, Ordering::Relaxed);
    }
}

/// Structurally enforces Spec-Trap #9 (runs 72/73/82): `library` is declared
/// FIRST and explicitly dropped BEFORE `_blob` (`Mapped` + `_file`) in `Drop`.
struct Library {
    library: Option<ID3D12PipelineLibrary1>,
    /// The blob the library was made from: it must outlive `library`.
    _blob: Option<Blob>,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Stores share it and a save takes it alone: the size the library
    /// asks for must still be its size when it is written out.
    saving: RwLock<()>,
}

impl Library {
    fn com(&self) -> &ID3D12PipelineLibrary1 {
        self.library.as_ref().expect("library COM object alive until Drop")
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // CRITICAL SAFETY INVARIANT (Spec-Trap #9, runs 72/73/82):
        // ALWAYS release `ID3D12PipelineLibrary1` BEFORE unmapping `Mapped`
        // (`UnmapViewOfFile`) or closing `Mapped._file`.
        drop(self.library.take());
        drop(self._blob.take());
    }
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}
// SAFETY: as above; the other fields are an atomic and a lock.
unsafe impl Sync for Library {}

/// Held by a load of the pipeline named `name`: loads of the same name
/// wait for each other (the host's one exception to the library's thread
/// safety: such a load may change that pipeline's entry). Names share 128
/// locks, so two different names seldom wait.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 128] = [const { Mutex::new(()) }; 128];
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    LOADING[(h as usize) & 127].lock().unwrap_or_else(|e| e.into_inner())
}

/// With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before.
fn one_call() -> Option<MutexGuard<'static, ()>> {
    static ON: OnceLock<bool> = OnceLock::new();
    static CALLING: Mutex<()> = Mutex::new(());
    ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_LOCK").is_some_and(|v| v == "1"))
        .then(|| CALLING.lock().unwrap_or_else(|e| e.into_inner()))
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 100_000;

static LIBRARY: RwLock<Option<Library>> = RwLock::new(None);
static LIBRARY_UNAVAILABLE: AtomicBool = AtomicBool::new(false);

/// Acquires a read guard holding an open `Library`, reopening lazily if it was
/// not yet opened or was torn down while idle.
fn library_read() -> Option<std::sync::RwLockReadGuard<'static, Option<Library>>> {
    if LIBRARY_UNAVAILABLE.load(Ordering::Relaxed) {
        return None;
    }
    {
        let guard = LIBRARY.read().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return Some(guard);
        }
    }
    {
        let mut guard = LIBRARY.write().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
                LIBRARY_UNAVAILABLE.store(true, Ordering::Relaxed);
                return None;
            };
            let blob = DIR.get().map_or(Blob::None, |d| Blob::of(d));
            let opened = open(&device, blob);
            if opened.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
                LIBRARY_UNAVAILABLE.store(true, Ordering::Relaxed);
                return None;
            }
            *guard = opened;
            note_activity();
        }
    }
    let guard = LIBRARY.read().unwrap_or_else(|e| e.into_inner());
    guard.is_some().then_some(guard)
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.bytes().is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                BLOB.store(bytes as u64, Ordering::Relaxed);
                tracing::info!(bytes, mapped = matches!(blob, Blob::Mapped(_)), "host pipeline library loaded");
                if let Blob::Mapped(file) = &blob {
                    VIEW[1].store(file.len, Ordering::Relaxed);
                    VIEW[0].store(file.at.Value as usize, Ordering::Relaxed);
                    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                }
                return Some(Library {
                    library: Some(library),
                    _blob: Some(blob),
                    added: AtomicU64::new(0),
                    saving: RwLock::new(()),
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library: Some(library),
        _blob: Some(Blob::None),
        added: AtomicU64::new(0),
        saving: RwLock::new(()),
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// Stack-allocated UTF-16 NUL-terminated pipeline name for ASCII names < 64 bytes
/// (all 32-char hex pipeline keys), falling back to `HSTRING` otherwise.
enum WideName {
    Stack([u16; 64]),
    Heap(HSTRING),
}

impl WideName {
    fn new(name: &str) -> Self {
        let bytes = name.as_bytes();
        if bytes.len() < 64 && bytes.is_ascii() {
            let mut buf = [0u16; 64];
            for (dst, &b) in buf.iter_mut().zip(bytes) {
                *dst = u16::from(b);
            }
            Self::Stack(buf)
        } else {
            Self::Heap(HSTRING::from(name))
        }
    }

    fn as_pcwstr(&self) -> PCWSTR {
        match self {
            Self::Stack(buf) => PCWSTR(buf.as_ptr()),
            Self::Heap(s) => PCWSTR(s.as_ptr()),
        }
    }
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = WideName::new(name);
    let began = std::time::Instant::now();
    let loaded = {
        let Some(guard) = library_read() else {
            MADE.fetch_add(1, Ordering::Relaxed);
            return create();
        };
        let lib = guard.as_ref().expect("library_read guarantees Some");
        note_activity();
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        load(lib.com(), wide.as_pcwstr())
    };
    // The load read the library's file (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                LABEL.with(|l| {
                    let label = l.borrow();
                    let label = label.as_deref().unwrap_or("");
                    format!("pipeline library: loaded {label} in {:.1} ms", began.elapsed().as_secs_f64() * 1e3)
                })
            });
            return Ok(pso);
        }
        Err(e) => e,
    };
    // Compiled outside any library lock: compiles take up to seconds.
    let pso = {
        let _call = call("make", name);
        create()?
    };
    MADE.fetch_add(1, Ordering::Relaxed);
    crate::host::trail(|| {
        LABEL.with(|l| {
            let label = l.borrow();
            let label = label.as_deref().unwrap_or("");
            format!(
                "pipeline library: made {label} in {:.1} ms (the library did not load it: {refused})",
                began.elapsed().as_secs_f64() * 1e3
            )
        })
    });
    if let Some(guard) = library_read()
        && let Some(lib) = guard.as_ref()
    {
        note_activity();
        let _all = one_call();
        let _stores = lib.saving.read().unwrap_or_else(|e| e.into_inner());
        let _call = call("store", name);
        // SAFETY: a live pipeline; a name already stored (a description that
        // differs) is refused, which leaves the library as it was.
        if unsafe { lib.com().StorePipeline(wide.as_pcwstr(), &pso) }.is_ok() {
            lib.added.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(pso)
}

/// Whether new compiled pipelines have been stored in the library since the
/// last save (never reopens a torn-down library).
pub(crate) fn has_added() -> bool {
    LIBRARY
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|l| l.added.load(Ordering::Relaxed) > 0)
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY_MS` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST_SAVE_MS: AtomicU64 = AtomicU64::new(u64::MAX);
    static SAVING_NOW: AtomicBool = AtomicBool::new(false);
    static SAVE_LOCK: Mutex<()> = Mutex::new(());
    const SAVE_EVERY_MS: u64 = 20_000;
    if !now && SAVING_NOW.load(Ordering::Relaxed) {
        return;
    }
    let Some(dir) = DIR.get() else { return };
    if !now && !has_added() && !USED_CHANGED.load(Ordering::Relaxed) {
        return;
    }
    let _save_guard = if now {
        let guard = SAVE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SAVING_NOW.store(true, Ordering::Release);
        guard
    } else {
        let now_ms = crate::host::coarse_ms();
        let last = LAST_SAVE_MS.load(Ordering::Relaxed);
        if last != u64::MAX && now_ms.saturating_sub(last) < SAVE_EVERY_MS {
            return;
        }
        if !has_added() && !USED_CHANGED.load(Ordering::Relaxed) {
            return;
        }
        let Ok(guard) = SAVE_LOCK.try_lock() else {
            return;
        };
        let last = LAST_SAVE_MS.load(Ordering::Relaxed);
        if last != u64::MAX && now_ms.saturating_sub(last) < SAVE_EVERY_MS {
            return;
        }
        SAVING_NOW.store(true, Ordering::Release);
        guard
    };
    struct ClearSaving;
    impl Drop for ClearSaving {
        fn drop(&mut self) {
            SAVING_NOW.store(false, Ordering::Release);
        }
    }
    let _clear = ClearSaving;
    LAST_SAVE_MS.store(crate::host::coarse_ms(), Ordering::Relaxed);
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let tmp = dir.join(LIBRARY_FILE).with_extension("tmp");
    // The bytes of the library written to `tmp`; 0 when no pipeline was
    // added (or the library is currently torn down idle).
    let bytes = {
        let lib_guard = LIBRARY.read().unwrap_or_else(|e| e.into_inner());
        if let Some(library) = lib_guard.as_ref() {
            let _all = one_call();
            // No store between asking the size and writing; loads go on.
            let _alone = library.saving.write().unwrap_or_else(|e| e.into_inner());
            let added = library.added.load(Ordering::Relaxed);
            if !USED_CHANGED.swap(false, Ordering::Relaxed) && added == 0 {
                return;
            }
            if added == 0 {
                0
            } else {
                // SAFETY: a plain query.
                let size = unsafe { library.com().GetSerializedSize() };
                let _call = call("serialise", "the whole library");
                if let Err(e) = serialise(library.com(), &tmp, size) {
                    tracing::warn!(error = %e, "host pipeline library not serialised");
                    let _ = std::fs::remove_file(&tmp);
                    return;
                }
                library.added.store(0, Ordering::Relaxed);
                SAVED.store(size as u64, Ordering::Relaxed);
                size
            }
        } else if !USED_CHANGED.swap(false, Ordering::Relaxed) {
            return;
        } else {
            0
        }
    };
    // Through `used_set`: a process that asked for none yet (`victory shaders`)
    // must not write the earlier runs' list over with nothing.
    let used: Vec<String> = used_set()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|k| {
            let hex = super::key_hex(k);
            super::key_hex_str(&hex).to_owned()
        })
        .collect();
    let library_written = if bytes > 0 { replace_library(dir, &tmp) } else { Ok(()) };
    if bytes > 0 && library_written.is_ok() {
        // In place unless the system refused and the new one waits beside it.
        let in_place = !dir.join(NEXT_FILE).exists();
        if in_place && VIEW[0].load(Ordering::Relaxed) != 0 {
            REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
            REPLACED[0].fetch_add(1, Ordering::Relaxed);
        }
        crate::host::trail(|| {
            format!(
                "pipeline library: saved {bytes} bytes, {}",
                if in_place { "renamed over the library's file" } else { "left beside the library's file for the next start" }
            )
        });
    }
    let slow: Vec<String> = slow_map()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(key, ms)| {
            let hex = super::key_hex(key);
            format!("{} {ms}", super::key_hex_str(&hex))
        })
        .collect();
    let written = library_written
        .and_then(|()| write_atomically(&dir.join(SLOW_FILE), slow.join("\n").as_bytes()))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
    match written {
        Ok(()) => tracing::info!(
            bytes,
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
    }
}

/// Writes the library's `size` serialised bytes to the file `tmp`: through
/// a mapping of that file, so that the bytes are the file's pages and not
/// 1.3 GB more of the process's own memory for the seconds a save takes;
/// through a buffer where the file cannot be mapped (and with
/// `VICTORY_LIBRARY_MAP=0`).
fn serialise(library: &ID3D12PipelineLibrary1, tmp: &std::path::Path, size: usize) -> std::io::Result<()> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped()
        && let Some(mut file) = Mapped::create(tmp, size)
    {
        // SAFETY: a buffer of the size the library asks for.
        return unsafe { library.Serialize(file.bytes_mut()) }.map_err(failed);
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: a buffer of the size the library asks for.
    unsafe { library.Serialize(&mut bytes) }.map_err(failed)?;
    std::fs::write(tmp, &bytes)
}

/// Renames the written library `tmp` over the library's file in `dir`. This
/// run may have that file mapped: Windows 11 replaces a mapped file, and
/// the mapping goes on reading the old one for as long as a handle on it
/// is open ([`holds`]; with none, only the pages in memory at that moment
/// stay: the rest read as zeros). Where a system refuses, the new library
/// is left beside it as [`NEXT_FILE`] for the next start.
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(tmp, dir.join(LIBRARY_FILE)).or_else(|_| std::fs::rename(tmp, dir.join(NEXT_FILE)))
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::AtomicU64;
    static WRITING: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    let tmp = path.with_file_name(name);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// How long each title pipeline that took long to make took, in ms (beside
/// the early list; a key may be there more than once).
const SLOW_FILE: &str = "slow-pipelines.txt";
/// A pipeline that takes this long is one ([`slow`]).
pub(crate) const SLOW_MS: u64 = 1000;
static SLOW: OnceLock<RwLock<std::collections::HashMap<[u8; 16], u64>>> = OnceLock::new();

fn slow_map() -> &'static RwLock<std::collections::HashMap<[u8; 16], u64>> {
    SLOW.get_or_init(|| {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(SLOW_FILE)).ok())
            .unwrap_or_default();
        RwLock::new(
            listed
                .lines()
                .filter_map(|l| l.split_once(' '))
                .filter_map(|(key, ms)| Some((super::parse_key_hex(key)?, ms.parse().ok()?)))
                .collect(),
        )
    })
}

/// Whether making the title pipeline with `key` took [`SLOW_MS`] in this or
/// an earlier run.
pub(crate) fn slow(key: &[u8; 16]) -> bool {
    slow_map()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(key)
}

/// The title pipeline with `key` took `ms` to make.
pub(crate) fn note_time(key: &[u8; 16], ms: u64) {
    if ms < SLOW_MS {
        return;
    }
    if slow_map()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(key)
        .is_some_and(|&known| ms <= known)
    {
        return;
    }
    let mut slow = slow_map().write().unwrap_or_else(|e| e.into_inner());
    let known = slow.entry(*key).or_insert(0);
    if ms > *known {
        *known = ms;
        // With the next save of the early list.
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

/// Title pipeline keys requested by draws or dispatches (this run and the last).
static USED: OnceLock<RwLock<HashSet<[u8; 16]>>> = OnceLock::new();
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used_set() -> &'static RwLock<HashSet<[u8; 16]>> {
    USED.get_or_init(|| {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<[u8; 16]> = listed
            .lines()
            .filter_map(|l| super::parse_key_hex(l.trim()))
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        RwLock::new(set)
    })
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
///
/// `VICTORY_EARLY_ALL=1`: every title pipeline is started at creation,
/// requested before or not, so that the driver's own cache holds it for
/// later runs.
pub(crate) fn wanted_early(key: &[u8; 16]) -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    if *ALL.get_or_init(|| std::env::var_os("VICTORY_EARLY_ALL").is_some_and(|v| v == "1")) {
        return true;
    }
    used_set()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .contains(key)
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &[u8; 16]) {
    {
        let used = used_set().read().unwrap_or_else(|e| e.into_inner());
        if used.len() >= EARLY_MAX || used.contains(key) {
            return;
        }
    }
    let mut used = used_set().write().unwrap_or_else(|e| e.into_inner());
    if used.len() < EARLY_MAX && used.insert(*key) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16))).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.com().GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Some(Blob::None)));
    }

    /// A mapped file reads as the file's bytes, a file written through a
    /// mapping holds what was written, and the library's file can be
    /// replaced while it is mapped (or the new one waits as the next).
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let tmp = path.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        {
            let mut file = Mapped::create(&tmp, new.len()).expect("a new file, mapped");
            file.bytes_mut().copy_from_slice(&new);
        }
        assert_eq!(std::fs::read(&tmp).unwrap(), new);
        replace_library(dir.path(), &tmp).unwrap();
        assert!(!tmp.exists());
        // The mapping still reads the old library.
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(4).enumerate() {
            let v = ((at / 4 + i) as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
            word.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The device loss of runs 72, 73 and 82: the library's file is
    /// replaced while it is mapped, and pages of the mapping that were not
    /// in memory then are read afterwards (in the runs: pages the trim let
    /// go and Windows gave to something else). With the file's handle kept
    /// open they still read as the old file. (With it closed, as before,
    /// all but the first few read as zeros on the PC this was found on.)
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open_holding(&path, true).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        let tmp = path.with_extension("tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("a new file, mapped");
            file.bytes_mut().fill(0xEE);
        }
        replace_library(dir.path(), &tmp).unwrap();
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(bytes == &expected[..], "page {page} of the replaced library no longer reads as the old file");
        }
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }

    #[test]
    fn wide_name_matches_hstring_for_ascii_long_and_non_ascii() {
        let cases: [(&str, bool); 6] = [
            ("", true),
            ("0123456789abcdef0123456789abcdef", true),
            (&"a".repeat(63), true),
            (&"b".repeat(64), false),
            (&"c".repeat(100), false),
            ("pîpeline-λ-🔥", false),
        ];
        for (input, expect_stack) in cases {
            let wide = WideName::new(input);
            assert_eq!(
                matches!(wide, WideName::Stack(_)),
                expect_stack,
                "unexpected storage variant for {input:?}"
            );
            let hstr = HSTRING::from(input);
            let wide_ptr = wide.as_pcwstr();
            let expected_ptr = PCWSTR(hstr.as_ptr());
            // SAFETY: both pointers are NUL-terminated UTF-16 strings.
            let got = unsafe { wide_ptr.as_wide() };
            let expected = unsafe { expected_ptr.as_wide() };
            assert_eq!(got, expected, "UTF-16 mismatch for {input:?}");
        }
    }

    #[test]
    fn active_call_tracker_reports_and_clears() {
        label("[vertex aabbccdd]".to_owned());
        let guard = call("load", "deadbeef01234567deadbeef01234567");
        let note = removal_note();
        assert!(
            note.contains("load of deadbeef01234567deadbeef01234567 [vertex aabbccdd] running for "),
            "missing active call in note: {note}"
        );
        drop(guard);
        let note_after = removal_note();
        assert!(
            !note_after.contains("deadbeef01234567deadbeef01234567"),
            "dropped call still present in note: {note_after}"
        );
    }

    #[test]
    fn trim_skips_small_pipeline_libraries_below_64_mib() {
        assert!(!trim_eligible(0));
        assert!(!trim_eligible(983 << 10));
        assert!(!trim_eligible((64 << 20) - 1));
        assert!(trim_eligible(64 << 20));
        assert!(trim_eligible(256 << 20));
    }

    #[test]
    fn teardown_eligibility_respects_idle_window_and_zero_disable() {
        assert!(!teardown_eligible(100_000, 10_000, 0));
        assert!(!teardown_eligible(100_000, 0, 30_000));
        assert!(!teardown_eligible(39_999, 10_000, 30_000));
        assert!(teardown_eligible(40_000, 10_000, 30_000));
        assert!(teardown_eligible(100_000, 10_000, 30_000));
    }

    /// Spec-Trap #9 (runs 72/73/82): verify `Library` drops its COM object
    /// BEFORE unmapping `Mapped` and closing `_file`, and that a mapped
    /// `Library` can be dropped and reopened cleanly.
    #[test]
    fn library_drop_order_releases_com_before_unmapping_and_closing_file() {
        let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        struct Probe {
            tag: &'static str,
            log: Arc<Mutex<Vec<&'static str>>>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                self.log.lock().unwrap().push(self.tag);
            }
        }
        struct FakeMapped {
            unmap_log: Arc<Mutex<Vec<&'static str>>>,
            _file: Probe,
        }
        impl Drop for FakeMapped {
            fn drop(&mut self) {
                self.unmap_log.lock().unwrap().push("unmap_view");
            }
        }
        struct FakeLibrary {
            com: Option<Probe>,
            blob: Option<FakeMapped>,
        }
        impl Drop for FakeLibrary {
            fn drop(&mut self) {
                drop(self.com.take());
                drop(self.blob.take());
            }
        }
        let lib = FakeLibrary {
            com: Some(Probe { tag: "drop_com", log: log.clone() }),
            blob: Some(FakeMapped {
                unmap_log: log.clone(),
                _file: Probe { tag: "close_file", log: log.clone() },
            }),
        };
        drop(lib);
        assert_eq!(*log.lock().unwrap(), vec!["drop_com", "unmap_view", "close_file"]);

        // Also exercise real `Library` drop with a mapped file if a host GPU is present.
        if let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) {
            let dir = tempfile::tempdir().unwrap();
            let empty = open(&device, Blob::None).expect("empty library");
            // SAFETY: plain query.
            let size = unsafe { empty.com().GetSerializedSize() };
            let path = dir.path().join(LIBRARY_FILE);
            serialise(empty.com(), &path, size).expect("serialise");
            drop(empty);

            let mapped_lib = open(&device, Blob::of(dir.path())).expect("mapped library");
            assert!(matches!(mapped_lib._blob, Some(Blob::Mapped(_))));
            drop(mapped_lib);
            // File handle closed cleanly after unmap -> file can be removed immediately.
            std::fs::remove_file(&path).expect("library file unlocked after clean Library drop");
        }
    }
}
