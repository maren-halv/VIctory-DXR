//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! (the BLAKE3 of its sources) and its settings, so a changed translator
//! never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use victory_pipeline::{Cache, CacheKey, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static CACHE_DIR: OnceLock<PathBuf> = OnceLock::new();
static WORKER_EXE: OnceLock<PathBuf> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    let _ = CACHE_DIR.set(dir.clone());
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// The configured shader cache directory, if any.
pub(crate) fn dir() -> Option<&'static Path> {
    CACHE_DIR.get().map(PathBuf::as_path)
}

/// Enables out-of-process `--compile-worker` spawning using the current
/// `victory.exe` binary for cold shader translation misses.
pub(crate) fn enable_compile_worker() {
    if let Ok(exe) = std::env::current_exe() {
        let _ = WORKER_EXE.set(exe);
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies a translation beside its container: the translator
/// (the BLAKE3 of its sources, `build.rs`), the settings it runs with, and
/// what it was told of the host. Not our executable: a rebuild of anything
/// else used to translate every shader again.
fn build() -> [u8; 32] {
    // The view heap's null slot differs by host GPU: a translation cached
    // without the clamp must not be served (`host_settings` below).
    crate::descriptor::host::probe_view_limit();
    translation_key()
}

fn translation_key() -> [u8; 32] {
    static FIXED: OnceLock<Vec<u8>> = OnceLock::new();
    let fixed = FIXED.get_or_init(|| {
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        let mut bytes = env!("VICTORY_TRANSLATOR").as_bytes().to_vec();
        bytes.extend_from_slice(&limit.to_le_bytes());
        // Which group barriers wait is part of the translation too.
        bytes.push(victory_shader::emit::group_sync());
        // So are the switches that pick the form of a wave64 program:
        // without them here a run with `VICTORY_WAVE_WIDE=0` was
        // served the translations an earlier run of the same
        // executable had cached with the switch unset.
        for name in [
            "VICTORY_WAVE_WIDE",
            "VICTORY_WAVE_DEEP_SKIPS",
            "VICTORY_WAVE_OWN_BITS",
            "VICTORY_WAVE_LOCAL",
            "VICTORY_F16_NEAREST",
            "VICTORY_HELPER_OWN",
            "VICTORY_FIRSTLANE_OWN",
            "VICTORY_FIRST_HALF",
            "VICTORY_STRICT_JOINS",
            "VICTORY_MESH_OUTPUT_STRICT",
            "VICTORY_BVH_FENCES",
            "VICTORY_BVH_BOUNDS",
            "VICTORY_NATIVE_RT",
            "VICTORY_NATIVE_RT_ANYHIT",
            "VICTORY_NATIVE_RT_FILTER",
            "VICTORY_NATIVE_RT_FBT_TWOPASS",
        ] {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(b'=');
            if let Some(value) = std::env::var_os(name) {
                bytes.extend_from_slice(value.as_encoded_bytes());
            }
            bytes.push(0);
        }
        // Closest-hit pre-shrink is distinct from the two-pass state-machine
        // and front/back traversal options. Use the lowerer's effective bit.
        if victory_shader::emit::native_rt() {
            bytes.extend_from_slice(b"VICTORY_NATIVE_RT_TWO_PASS=");
            bytes.push(u8::from(victory_shader::emit::native_rt_two_pass()));
            bytes.push(0);
        }
        // And which clean-up passes ran over the translated code.
        bytes.extend_from_slice(&victory_shader::emit::shader_opt().to_le_bytes());
        // Renderer port 5 (`VICTORY_PORT_CBV=1`): constant buffers read as raw
        // bytes. Only when on, so the translations cached without it keep
        // their key.
        if victory_shader::emit::constants_raw() {
            bytes.extend_from_slice(b"VICTORY_PORT_CBV=1");
        }
        let rdna_w32 = victory_shader::emit::rdna_wave32();
        if rdna_w32 != 0 {
            bytes.extend_from_slice(b"VICTORY_RDNA_WAVE32=");
            bytes.push(b'0' + rdna_w32);
        }
        if !victory_shader::emit::loop_limit_enabled() {
            bytes.extend_from_slice(b"VICTORY_LOOP_LIMIT=0");
        }
        if !victory_shader::emit::lds_init_hoist_on() {
            bytes.extend_from_slice(b"VICTORY_LDS_INIT_HOIST=0");
        }
        bytes
    });
    let mut hash = blake3::Hasher::new();
    hash.update(fixed);
    for value in victory_shader::emit::host_settings() {
        hash.update(&value.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

/// What tells one set of translations from another ([`build`]).
pub(crate) fn translation() -> [u8; 32] {
    build()
}

/// Puts the translation of `container` into the cache before any title asks
/// for it (`victory shaders`): whether it was there already, or why it does
/// not translate. The host is opened first, as a run does: what the card
/// can do decides the form of some programs.
pub fn precompile(container: &[u8]) -> Result<bool, String> {
    let _ = crate::host::host();
    let misses = MISSES.load(Ordering::Relaxed);
    host_shader(container)?;
    // (Counts other threads' misses too: the caller only adds them up.)
    Ok(MISSES.load(Ordering::Relaxed) == misses)
}

/// The translation of a console shader container, from the cache when
/// there is one. Warm cache hits stay 100% in-process; cold translation misses
/// route through the out-of-process `--compile-worker` pool when enabled so
/// LLVM / structurizer NT-heap fragmentation stays out of the main process.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    host_shader_with_path(container).map(|(shader, _)| shader)
}

/// As [`host_shader`], also returning the content-addressed cache file path
/// when backed by disk so transient DXIL bytes can be reloaded on demand
/// without retaining `.dxil` in RAM.
pub(crate) fn host_shader_with_path(container: &[u8]) -> Result<(HostShader, Option<PathBuf>), String> {
    let translate = |c: &[u8]| translate_cold(c);
    match CACHE.get() {
        Some(cache) => cached_with_path(cache, &build(), container, translate),
        None => translate(container).map(|s| (s, None)),
    }
}

/// Loads only the `dxil` byte slice from a previously written `VSH3` cache
/// entry on disk without decoding the metadata lists.
pub(crate) fn load_dxil_only(path: &Path) -> Option<Vec<u8>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    file.read_exact(&mut head).ok()?;
    if &head[0..4] != MAGIC {
        return None;
    }
    let dxil_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
    let file_len = file.metadata().ok()?.len() as usize;
    if 12usize.checked_add(dxil_len)? > file_len {
        return None;
    }
    let mut dxil = vec![0u8; dxil_len];
    file.read_exact(&mut dxil).ok()?;
    Some(dxil)
}

fn translate_in_process(container: &[u8]) -> Result<HostShader, String> {
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    victory_shader::emit::with_shader_id(&hash, || victory_shader::emit::translate(container))
        .map_err(|e| e.to_string())
}

fn translate_cold(container: &[u8]) -> Result<HostShader, String> {
    if let Some(pool) = worker_pool() {
        let hash = blake3::hash(container).to_hex()[..16].to_owned();
        let hs = victory_shader::emit::host_settings();
        let settings = [hs[0], hs[1], hs[2], hs[3]];
        match pool.translate(settings, &hash, container) {
            Ok(res) => return res,
            Err(ipc_err) => {
                static LOGGED: AtomicBool = AtomicBool::new(false);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(error = %ipc_err, "compile-worker IPC failed; falling back to in-process shader translation");
                }
            }
        }
    }
    translate_in_process(container)
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // Bumped when HostShader serialization or shader codegen changes.
        2
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

#[cfg(test)]
fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    cached_with_path(cache, build, container, translate).map(|(s, _)| s)
}

fn cached_with_path(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<(HostShader, Option<PathBuf>), String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let key = CacheKey::new(&converter, &input);
        let existing = cache.path_for(&key);
        if existing.is_file()
            && let Some(shader) = std::fs::read(&existing).ok().and_then(|b| decode(&b))
        {
            HITS.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %existing.display(), "shader from the cache");
            return Ok((shader, Some(existing)));
        }
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok((shader, Some(path)));
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container).map(|s| (s, None))
}

// ---------------------------------------------------------------------------
// Out-of-process `--compile-worker` sidecar pool (cold misses only)
// ---------------------------------------------------------------------------

const IPC_MAGIC: [u8; 4] = *b"VCW1";
const IPC_MAX_FRAME: usize = 64 << 20;
const OP_SHUTDOWN: u8 = 0;
const OP_TRANSLATE: u8 = 1;
const STATUS_OK: u8 = 0;
const STATUS_ERR: u8 = 1;
const IN_WORKER_ENV: &str = "VICTORY_IN_COMPILE_WORKER";

fn write_frame(w: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "IPC frame exceeds u32"))?;
    if payload.len() > IPC_MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "IPC frame too large"));
    }
    let mut header = [0u8; 8];
    header[..4].copy_from_slice(&IPC_MAGIC);
    header[4..8].copy_from_slice(&len.to_le_bytes());
    w.write_all(&header)?;
    w.write_all(payload)?;
    w.flush()
}

fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 8];
    match r.read_exact(&mut header) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    if header[..4] != IPC_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad compile-worker IPC magic"));
    }
    let len = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    if len > IPC_MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "compile-worker IPC frame too large"));
    }
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    Ok(Some(payload))
}

fn encode_translate_request(settings: [u32; 4], hash: &str, container: &[u8]) -> Vec<u8> {
    let hash_bytes = hash.as_bytes();
    let hash_len = hash_bytes.len().min(u16::MAX as usize) as u16;
    let mut buf = Vec::with_capacity(1 + 16 + 2 + usize::from(hash_len) + 4 + container.len());
    buf.push(OP_TRANSLATE);
    for w in settings {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    buf.extend_from_slice(&hash_len.to_le_bytes());
    buf.extend_from_slice(&hash_bytes[..usize::from(hash_len)]);
    buf.extend_from_slice(&(container.len() as u32).to_le_bytes());
    buf.extend_from_slice(container);
    buf
}

fn decode_translate_request(payload: &[u8]) -> Option<([u32; 4], &str, &[u8])> {
    let (&op, mut rest) = payload.split_first()?;
    if op != OP_TRANSLATE {
        return None;
    }
    let mut settings = [0u32; 4];
    for slot in &mut settings {
        let (head, after) = rest.split_first_chunk::<4>()?;
        *slot = u32::from_le_bytes(*head);
        rest = after;
    }
    let (hlen_bytes, after) = rest.split_first_chunk::<2>()?;
    let hlen = u16::from_le_bytes(*hlen_bytes) as usize;
    let (hash_bytes, after) = after.split_at_checked(hlen)?;
    let hash = std::str::from_utf8(hash_bytes).ok()?;
    let (clen_bytes, after) = after.split_first_chunk::<4>()?;
    let clen = u32::from_le_bytes(*clen_bytes) as usize;
    let (container, tail) = after.split_at_checked(clen)?;
    tail.is_empty().then_some((settings, hash, container))
}

fn encode_translate_response(result: &Result<HostShader, String>) -> Vec<u8> {
    match result {
        Ok(shader) => {
            let encoded = encode(shader);
            let mut buf = Vec::with_capacity(1 + encoded.len());
            buf.push(STATUS_OK);
            buf.extend_from_slice(&encoded);
            buf
        }
        Err(msg) => {
            let mut buf = Vec::with_capacity(1 + msg.len());
            buf.push(STATUS_ERR);
            buf.extend_from_slice(msg.as_bytes());
            buf
        }
    }
}

fn decode_translate_response(payload: &[u8]) -> io::Result<Result<HostShader, String>> {
    let (&status, rest) = payload
        .split_first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty IPC response"))?;
    match status {
        STATUS_OK => decode(rest)
            .map(Ok)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corrupt VSH3 in IPC response")),
        STATUS_ERR => Ok(Err(String::from_utf8_lossy(rest).into_owned())),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown IPC status")),
    }
}

/// Serves compile requests on `reader`/`writer` until EOF or `OP_SHUTDOWN`.
fn serve_compile_worker_with(
    reader: &mut impl Read,
    writer: &mut impl Write,
    mut translate_fn: impl FnMut([u32; 4], &str, &[u8]) -> Result<HostShader, String>,
) -> io::Result<()> {
    while let Some(frame) = read_frame(reader)? {
        if frame.first() == Some(&OP_SHUTDOWN) {
            break;
        }
        let Some((settings, hash, container)) = decode_translate_request(&frame) else {
            let resp = encode_translate_response(&Err("malformed compile-worker request".into()));
            write_frame(writer, &resp)?;
            continue;
        };
        let result = translate_fn(settings, hash, container);
        let resp = encode_translate_response(&result);
        write_frame(writer, &resp)?;
    }
    Ok(())
}

/// Entry point for `victory.exe --compile-worker`. Reads length-prefixed
/// `VCW1` translation requests on stdin and writes `VCW1` responses to stdout.
pub fn run_compile_worker() -> io::Result<()> {
    let mut stdin = io::BufReader::new(io::stdin().lock());
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    serve_compile_worker_with(&mut stdin, &mut stdout, |settings, hash, container| {
        if settings[0] > 0 {
            victory_shader::emit::set_host_max_wave(settings[0]);
        }
        if settings[1] > 0 {
            victory_shader::emit::set_view_heap_limit(settings[1]);
        }
        if settings[2] > 0 {
            victory_shader::emit::set_raw_views(settings[2]);
        }
        victory_shader::emit::set_host_rdna(settings[3] != 0);
        victory_shader::emit::with_shader_id(hash, || victory_shader::emit::translate(container))
            .map_err(|e| e.to_string())
    })
}

/// Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` so
/// every `--compile-worker` child is terminated immediately if the parent exits
/// or crashes, leaving zero orphan worker processes.
mod job {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    type Handle = *mut c_void;

    const JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS: u32 = 9;
    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x0000_2000;

    #[repr(C)]
    #[derive(Default)]
    struct JobObjectBasicLimitInformation {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        _pad0: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        _pad1: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct JobObjectExtendedLimitInformation {
        basic_limit_information: JobObjectBasicLimitInformation,
        io_info: IoCounters,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    #[cfg(target_pointer_width = "64")]
    const _: () = {
        assert!(size_of::<JobObjectBasicLimitInformation>() == 64);
        assert!(size_of::<JobObjectExtendedLimitInformation>() == 144);
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateJobObjectW(lp_job_attributes: *const c_void, lp_name: *const u16) -> Handle;
        fn SetInformationJobObject(
            h_job: Handle,
            info_class: u32,
            info: *const c_void,
            info_len: u32,
        ) -> i32;
        fn AssignProcessToJobObject(h_job: Handle, h_process: Handle) -> i32;
        fn CloseHandle(h_object: Handle) -> i32;
    }

    pub(super) struct KillOnCloseJob(Handle);

    // SAFETY: Win32 kernel job handles are process-global and thread-safe.
    unsafe impl Send for KillOnCloseJob {}
    unsafe impl Sync for KillOnCloseJob {}

    impl KillOnCloseJob {
        pub(super) fn new() -> Option<Self> {
            // SAFETY: unnamed job object with default security attributes.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return None;
            }
            let mut info = JobObjectExtendedLimitInformation::default();
            info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `info` matches `JobObjectExtendedLimitInformation` layout and size.
            let ok = unsafe {
                SetInformationJobObject(
                    handle,
                    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                    (&raw const info).cast(),
                    size_of::<JobObjectExtendedLimitInformation>() as u32,
                )
            };
            if ok == 0 {
                // SAFETY: closing the handle we just created.
                unsafe { CloseHandle(handle) };
                return None;
            }
            Some(Self(handle))
        }

        pub(super) fn assign(&self, child: &Child) -> bool {
            // SAFETY: valid job handle and live child process handle.
            unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle().cast()) != 0 }
        }
    }

    impl Drop for KillOnCloseJob {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: owned job handle.
                unsafe { CloseHandle(self.0) };
            }
        }
    }
}

struct ChildWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: io::BufReader<ChildStdout>,
    compiles: u32,
}

impl ChildWorker {
    fn spawn(exe: &Path, job: Option<&job::KillOnCloseJob>) -> io::Result<Self> {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let mut child = Command::new(exe)
            .arg("--compile-worker")
            .env(IN_WORKER_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .creation_flags(BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW)
            .spawn()?;
        if let Some(job) = job {
            let _ = job.assign(&child);
        }
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("compile-worker missing stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("compile-worker missing stdout"))?;
        Ok(Self {
            child,
            stdin,
            stdout: io::BufReader::new(stdout),
            compiles: 0,
        })
    }

    fn shutdown(mut self) {
        let _ = write_frame(&mut self.stdin, &[OP_SHUTDOWN]);
        drop(self.stdin);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct WorkerPool {
    exe: PathBuf,
    slots: Vec<Mutex<Option<ChildWorker>>>,
    next: AtomicUsize,
    recycle_after: u32,
    job: Option<job::KillOnCloseJob>,
}

impl WorkerPool {
    fn new(exe: PathBuf) -> Self {
        let count = std::env::var("VICTORY_COMPILE_WORKERS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| {
                (std::thread::available_parallelism().map_or(4, |n| n.get()) / 4).clamp(2, 4)
            })
            .clamp(1, 8);
        let recycle_after = std::env::var("VICTORY_COMPILE_WORKER_RECYCLE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(64)
            .max(1);
        let job = job::KillOnCloseJob::new();
        let slots = (0..count).map(|_| Mutex::new(None)).collect();
        tracing::info!(
            workers = count,
            recycle_after,
            job_object = job.is_some(),
            "out-of-process shader compile-worker pool ready (cold misses only)"
        );
        Self {
            exe,
            slots,
            next: AtomicUsize::new(0),
            recycle_after,
            job,
        }
    }

    fn translate(
        &self,
        settings: [u32; 4],
        hash: &str,
        container: &[u8],
    ) -> io::Result<Result<HostShader, String>> {
        let req = encode_translate_request(settings, hash, container);
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.slots.len();
        let mut guard = self.slots[idx].lock().unwrap_or_else(|e| e.into_inner());
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(ChildWorker::spawn(&self.exe, self.job.as_ref())?);
            }
            let worker = guard.as_mut().expect("spawned above");
            let io_res = write_frame(&mut worker.stdin, &req)
                .and_then(|()| read_frame(&mut worker.stdout))
                .and_then(|opt| {
                    opt.ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "compile-worker exited"))
                })
                .and_then(|resp| decode_translate_response(&resp));
            match io_res {
                Ok(res) => {
                    worker.compiles += 1;
                    if worker.compiles >= self.recycle_after
                        && let Some(old) = guard.take()
                    {
                        old.shutdown();
                    }
                    return Ok(res);
                }
                Err(e) => {
                    if let Some(dead) = guard.take() {
                        dead.shutdown();
                    }
                    if attempt == 1 {
                        return Err(e);
                    }
                }
            }
        }
        Err(io::Error::other("compile-worker retry exhausted"))
    }
}

fn compile_worker_enabled() -> Option<PathBuf> {
    if std::env::var_os(IN_WORKER_ENV).is_some() {
        return None;
    }
    if std::env::var_os("VICTORY_COMPILE_WORKER").is_some_and(|v| v == "0") {
        return None;
    }
    if let Some(exe) = WORKER_EXE.get() {
        return Some(exe.clone());
    }
    let exe = std::env::current_exe().ok()?;
    let stem = exe.file_stem()?.to_str()?;
    stem.eq_ignore_ascii_case("victory").then_some(exe)
}

fn worker_pool() -> Option<&'static WorkerPool> {
    static POOL: OnceLock<Option<WorkerPool>> = OnceLock::new();
    POOL.get_or_init(|| compile_worker_enabled().map(WorkerPool::new))
        .as_ref()
}

const MAGIC: &[u8; 4] = b"VSH3";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    word(&mut out, s.half_views.len() as u32);
    for &(slot, dword) in &s.half_views {
        word(&mut out, u32::from(slot));
        word(&mut out, dword);
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    let n = r.word()? as usize;
    let mut half_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        half_views.push((u8::try_from(r.word()?).ok()?, r.word()?));
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
        half_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
            half_views: vec![(0, 8), (4, 0)],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH3 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    #[test]
    fn native_rt_two_pass_modes_get_separate_disk_entries() {
        const CACHE_DIR: &str = "VICTORY_TEST_NATIVE_RT_CACHE_DIR";
        const EXPECT_MISS: &str = "VICTORY_TEST_NATIVE_RT_CACHE_MISS";
        const NATIVE: &str = "VICTORY_NATIVE_RT";
        const TWO_PASS: &str = "VICTORY_NATIVE_RT_TWO_PASS";
        if let Some(dir) = std::env::var_os(CACHE_DIR) {
            let cache = Cache::open(PathBuf::from(dir)).unwrap();
            let mut expected = shader();
            let mode = if victory_shader::emit::native_rt() {
                1 + u8::from(victory_shader::emit::native_rt_two_pass())
            } else {
                0
            };
            expected.dxil.push(mode);
            let ran = AtomicBool::new(false);
            let actual = cached(&cache, &translation_key(), b"same container in all modes", |_| {
                ran.store(true, Ordering::Relaxed);
                Ok(expected.clone())
            })
            .unwrap();
            assert_eq!(actual, expected, "a warm hit returned another mode's bytes");
            assert_eq!(ran.load(Ordering::Relaxed), std::env::var_os(EXPECT_MISS).is_some());
            return;
        }

        // The effective options and fixed key are initialized once per process.
        let executable = std::env::current_exe().unwrap();
        let dir = tempfile::tempdir().unwrap();
        for (native, two_pass, miss) in [
            ("1", None, true),
            ("1", Some("0"), true),
            ("1", Some("1"), false),
            ("1", Some("0"), false),
            ("0", None, true),
            ("0", Some("0"), false),
        ] {
            let mut cmd = Command::new(&executable);
            cmd.args(["--exact", "shader_cache::tests::native_rt_two_pass_modes_get_separate_disk_entries"])
                .env(CACHE_DIR, dir.path())
                .env(NATIVE, native)
                .env_remove(TWO_PASS)
                .env_remove(EXPECT_MISS);
            if let Some(value) = two_pass {
                cmd.env(TWO_PASS, value);
            }
            if miss {
                cmd.env(EXPECT_MISS, "1");
            }
            let out = cmd.output().unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "native={native} two-pass={two_pass:?}: {stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        assert_eq!(decode(b"VSH1"), None);
        assert_eq!(decode(b"VSH2"), None);
    }

    #[test]
    fn load_dxil_only_reads_just_dxil_from_vsh3() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("entry.vsh3");
        let expected = shader();
        std::fs::write(&path, encode(&expected)).unwrap();
        assert_eq!(load_dxil_only(&path).as_deref(), Some(expected.dxil.as_slice()));
        std::fs::write(&path, b"BAD0short").unwrap();
        assert_eq!(load_dxil_only(&path), None);
    }

    #[test]
    fn compile_worker_ipc_round_trip_errors_and_shutdown() {
        let mut req_stream = Vec::new();
        let req_ok = encode_translate_request([32, 1_000_000, 500_000, 1], "0123456789abcdef", b"cdxil-ok");
        let req_err = encode_translate_request([64, 0, 0, 0], "fedcba9876543210", b"cdxil-bad");
        write_frame(&mut req_stream, &req_ok).unwrap();
        write_frame(&mut req_stream, &req_err).unwrap();
        write_frame(&mut req_stream, &[OP_SHUTDOWN]).unwrap();
        // Trailing request after shutdown must not be processed.
        write_frame(&mut req_stream, &req_ok).unwrap();

        let mut reader = io::Cursor::new(req_stream);
        let mut resp_stream = Vec::new();
        let mut seen = Vec::new();
        serve_compile_worker_with(&mut reader, &mut resp_stream, |settings, hash, container| {
            seen.push((settings, hash.to_owned(), container.to_vec()));
            if container == b"cdxil-ok" {
                Ok(shader())
            } else {
                Err("unsupported op".into())
            }
        })
        .unwrap();

        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0], ([32, 1_000_000, 500_000, 1], "0123456789abcdef".into(), b"cdxil-ok".to_vec()));
        assert_eq!(seen[1], ([64, 0, 0, 0], "fedcba9876543210".into(), b"cdxil-bad".to_vec()));

        let mut resp_reader = io::Cursor::new(resp_stream);
        let f1 = read_frame(&mut resp_reader).unwrap().unwrap();
        assert_eq!(decode_translate_response(&f1).unwrap(), Ok(shader()));
        let f2 = read_frame(&mut resp_reader).unwrap().unwrap();
        assert_eq!(decode_translate_response(&f2).unwrap(), Err("unsupported op".into()));
        assert!(read_frame(&mut resp_reader).unwrap().is_none());

        // Bad magic is rejected.
        let mut bad = io::Cursor::new(b"NOPE\x00\x00\x00\x00".to_vec());
        assert!(read_frame(&mut bad).is_err());
    }

    #[test]
    fn kill_on_close_job_object_creates_cleanly() {
        let job = job::KillOnCloseJob::new();
        assert!(job.is_some(), "CreateJobObjectW + SetInformationJobObject(KILL_ON_JOB_CLOSE) must succeed");
    }
}
