//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! The title creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::offset_of;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

/// 16-character lowercase hex shader hash stored inline (zero heap allocation).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Hex16([u8; 16]);

impl Hex16 {
    pub(crate) fn from_hash(full: &blake3::Hash) -> Self {
        let hex = full.to_hex();
        let mut out = [0u8; 16];
        out.copy_from_slice(&hex.as_bytes()[..16]);
        Self(out)
    }

    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: `self.0` contains only ASCII hex digits.
        unsafe { std::str::from_utf8_unchecked(&self.0) }
    }
}

impl std::ops::Deref for Hex16 {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for Hex16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Debug for Hex16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Formats a 16-byte pipeline key as 32 lowercase ASCII hex bytes on the stack.
#[cfg(windows)]
pub(crate) fn key_hex(key: &[u8; 16]) -> [u8; 32] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 32];
    for (i, &b) in key.iter().enumerate() {
        out[2 * i] = HEX[usize::from(b >> 4)];
        out[2 * i + 1] = HEX[usize::from(b & 0xf)];
    }
    out
}

#[cfg(windows)]
pub(crate) fn key_hex_str(buf: &[u8; 32]) -> &str {
    // SAFETY: `key_hex` produces only ASCII hex digits.
    unsafe { std::str::from_utf8_unchecked(buf) }
}

/// Parses a 32-character hex pipeline key into 16 inline bytes.
#[cfg(windows)]
pub(crate) fn parse_key_hex(s: &str) -> Option<[u8; 16]> {
    let bytes = s.as_bytes();
    if bytes.len() != 32 {
        return None;
    }
    let hex_val = |b: u8| match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    };
    let mut out = [0u8; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_val(bytes[2 * i])?;
        let lo = hex_val(bytes[2 * i + 1])?;
        *slot = (hi << 4) | lo;
    }
    Some(out)
}

/// Precomputed container flags evaluated once while the raw `.cdxil` bytes are
/// in hand so `CreatePipelineState` never needs to read back from disk.
pub(crate) const FLAG_BUILDER: u8 = 1 << 0;
pub(crate) const FLAG_RT_RESOLVE: u8 = 1 << 1;
pub(crate) const FLAG_ACTIVATE_RAYS: u8 = 1 << 2;
pub(crate) const FLAG_LOCAL_LIGHTS: u8 = 1 << 3;

const PACK_MAGIC: [u8; 4] = *b"VCP1";
const PACK_HEADER_LEN: usize = 4 + 32 + 4;
const PACK_FILE: &str = "shader-containers.pack";

/// Append-only, deduplicated on-disk pack of raw `.cdxil` containers so
/// `SHADERS` does not retain ~220 MB of raw containers in RAM forever.
struct ContainerPack {
    read: std::fs::File,
    writer: Mutex<PackWriter>,
}

struct PackWriter {
    file: std::fs::File,
    end: u64,
    index: HashMap<[u8; 32], (u64, u32)>,
}

impl ContainerPack {
    fn open(dir: &std::path::Path) -> Option<Self> {
        if std::env::var_os("VICTORY_CONTAINER_PACK").is_some_and(|v| v == "0") {
            return None;
        }
        std::fs::create_dir_all(dir).ok()?;
        let path = dir.join(PACK_FILE);
        // Exclusive writer share mode (`FILE_SHARE_READ = 1`) ensures only one
        // process appends to the pack at a time; concurrent helper processes
        // (e.g. `victory warm` children) cleanly fall back to inline buffers.
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true).create(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_SHARE_READ: u32 = 1;
            opts.share_mode(FILE_SHARE_READ);
        }
        let mut write_file = opts.open(&path).ok()?;
        let file_len = write_file.metadata().ok()?.len();
        let mut index = HashMap::new();
        let mut at = 0u64;
        {
            let mut reader = std::io::BufReader::new(&mut write_file);
            let mut head = [0u8; PACK_HEADER_LEN];
            while reader.read_exact(&mut head).is_ok() {
                if head[0..4] != PACK_MAGIC {
                    break;
                }
                let len = u32::from_le_bytes([head[36], head[37], head[38], head[39]]);
                let next = at.saturating_add(PACK_HEADER_LEN as u64).saturating_add(u64::from(len));
                if next > file_len || reader.seek_relative(i64::from(len)).is_err() {
                    break;
                }
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&head[4..36]);
                index.insert(hash, (at + PACK_HEADER_LEN as u64, len));
                at = next;
            }
        }
        if file_len > at {
            let _ = write_file.set_len(at);
        }
        write_file.seek(SeekFrom::Start(at)).ok()?;
        let read = {
            let mut ropts = std::fs::OpenOptions::new();
            ropts.read(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                // Must share read+write with our own writer handle.
                ropts.share_mode(1 | 2);
            }
            ropts.open(&path).ok()?
        };
        tracing::info!(
            entries = index.len(),
            bytes = at,
            path = %path.display(),
            "shader container pack ready"
        );
        Some(Self {
            read,
            writer: Mutex::new(PackWriter {
                file: write_file,
                end: at,
                index,
            }),
        })
    }

    fn spill(&self, hash: [u8; 32], bytes: &[u8]) -> Option<(u64, u32)> {
        let len = u32::try_from(bytes.len()).ok()?;
        let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(&(off, existing_len)) = w.index.get(&hash)
            && existing_len == len
        {
            return Some((off, len));
        }
        let mut head = [0u8; PACK_HEADER_LEN];
        head[0..4].copy_from_slice(&PACK_MAGIC);
        head[4..36].copy_from_slice(&hash);
        head[36..40].copy_from_slice(&len.to_le_bytes());
        let start = w.end;
        if w.file.seek(SeekFrom::Start(start)).is_err()
            || w.file.write_all(&head).is_err()
            || w.file.write_all(bytes).is_err()
        {
            let _ = w.file.set_len(start);
            return None;
        }
        let payload_offset = start + PACK_HEADER_LEN as u64;
        w.end = payload_offset + u64::from(len);
        w.index.insert(hash, (payload_offset, len));
        Some((payload_offset, len))
    }

    fn read_at(&self, offset: u64, len: u32) -> Option<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut filled = 0usize;
            while filled < buf.len() {
                let n = self
                    .read
                    .seek_read(&mut buf[filled..], offset + filled as u64)
                    .ok()?;
                if n == 0 {
                    return None;
                }
                filled += n;
            }
            Some(buf)
        }
        #[cfg(not(windows))]
        {
            let mut file = self.read.try_clone().ok()?;
            file.seek(SeekFrom::Start(offset)).ok()?;
            file.read_exact(&mut buf).ok()?;
            Some(buf)
        }
    }
}

static CONTAINER_PACK: Mutex<Option<Option<Arc<ContainerPack>>>> = Mutex::new(None);

fn container_pack() -> Option<Arc<ContainerPack>> {
    let mut slot = CONTAINER_PACK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = slot.as_ref() {
        return existing.clone();
    }
    #[cfg(windows)]
    let dir = crate::shader_cache::dir()
        .map(PathBuf::from)
        .or_else(library::folder);
    #[cfg(not(windows))]
    let dir: Option<PathBuf> = None;
    let Some(dir) = dir else {
        // Cache dir not configured yet (e.g. unit tests); do not latch None
        // permanently in case `use_shader_cache` is called later.
        return None;
    };
    let opened = ContainerPack::open(&dir).map(Arc::new);
    *slot = Some(opened.clone());
    opened
}

enum ContainerBacking {
    Packed {
        pack: Arc<ContainerPack>,
        offset: u64,
        len: u32,
    },
    Inline(Vec<u8>),
}

impl ContainerBacking {
    fn read(&self) -> Option<Vec<u8>> {
        match self {
            Self::Packed { pack, offset, len } => pack.read_at(*offset, *len),
            Self::Inline(bytes) => Some(bytes.clone()),
        }
    }
}

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes. Raw `.cdxil` containers spill to `shader-containers.pack` when
/// a writable cache dir is configured, keeping zero raw container bytes in RAM.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged (inline, zero heap alloc).
    pub(crate) hash: Hex16,
    pub(crate) full_hash: [u8; 32],
    /// Raw container byte length (kept inline for logging without reading disk).
    pub(crate) size: u32,
    flags: u8,
    container: ContainerBacking,
    /// Its part in the stand-in for the indirect light.
    #[cfg(windows)]
    pub(crate) ambient: Option<crate::ambient::Role>,
    #[cfg(windows)]
    translated: OnceLock<Option<Box<Translated>>>,
}

impl Shader {
    pub(crate) fn has_flag(&self, bit: u8) -> bool {
        (self.flags & bit) != 0
    }

    pub(crate) fn container_bytes(&self) -> Option<Vec<u8>> {
        self.container.read()
    }
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// For the memory lines: containers kept and their resident RAM bytes,
/// translations kept and their resident metadata bytes.
static KEPT: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the shader table and the pipeline library keep in memory:
/// containers and their bytes, translations and their bytes, the bytes of
/// the library's file (kept whole while the library lives) and what its
/// last save wrote.
#[cfg(windows)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn kept() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        KEPT[0].load(Relaxed),
        KEPT[1].load(Relaxed),
        KEPT[2].load(Relaxed),
        KEPT[3].load(Relaxed),
        library::BLOB.load(Relaxed),
        library::SAVED.load(Relaxed),
        library::TRIMMED_BYTES.load(Relaxed),
    ]
}

fn compute_container_flags(stage: &'static str, bytes: &[u8]) -> u8 {
    if stage != "compute" {
        return 0;
    }
    let mut flags = 0u8;
    if bytes.windows(b"CS_RaytracingResolve".len()).any(|w| w == b"CS_RaytracingResolve") {
        flags |= FLAG_RT_RESOLVE;
    }
    if bytes
        .windows(b"CS_ScreenProbes_ActivateRays".len())
        .any(|w| w == b"CS_ScreenProbes_ActivateRays")
    {
        flags |= FLAG_ACTIVATE_RAYS;
    }
    if bytes
        .windows(b"CS_GatherRaymarchedLocalLightsClassified".len())
        .any(|w| w == b"CS_GatherRaymarchedLocalLightsClassified")
    {
        flags |= FLAG_LOCAL_LIGHTS;
    }
    #[cfg(windows)]
    {
        use victory_shader::{Container, FourCC};
        if let Ok(c) = Container::parse(bytes)
            && c.part(FourCC::DXIL)
                .and_then(|part| victory_shader::native_programs(part.data).ok())
                .is_some_and(|programs| programs.iter().any(victory_shader::spill::is_spilled))
        {
            flags |= FLAG_BUILDER;
        }
    }
    flags
}

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let full_bytes = *full.as_bytes();
    {
        let shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(existing) = shaders.as_ref().and_then(|m| m.get(&full_bytes)) {
            return existing.clone();
        }
    }
    let flags = compute_container_flags(stage, bytes);
    #[cfg(windows)]
    let ambient = (stage == "compute")
        .then(|| crate::ambient::role(bytes))
        .flatten();
    let (container, ram_bytes) = if let Some(pack) = container_pack()
        && let Some((offset, len)) = pack.spill(full_bytes, bytes)
    {
        (ContainerBacking::Packed { pack, offset, len }, 0u64)
    } else {
        (ContainerBacking::Inline(bytes.to_vec()), bytes.len() as u64)
    };
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(full_bytes)
        .or_insert_with(|| {
            KEPT[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            KEPT[1].fetch_add(ram_bytes, std::sync::atomic::Ordering::Relaxed);
            Arc::new(Shader {
                stage,
                hash: Hex16::from_hash(&full),
                full_hash: full_bytes,
                size: bytes.len().min(u32::MAX as usize) as u32,
                flags,
                container,
                #[cfg(windows)]
                ambient,
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// Compact resident metadata of a translated host shader: zero `.dxil` bytes
/// retained in RAM, zero disk I/O on draw threads.
#[cfg(windows)]
pub(crate) struct HostMeta {
    pub(crate) user_data: Vec<victory_shader::emit::UserData>,
    pub(crate) static_samplers: Vec<victory_shader::emit::StaticSampler>,
    pub(crate) root_cbvs: Vec<u8>,
    pub(crate) root_srvs: Vec<u8>,
    pub(crate) root_uavs: Vec<u8>,
    pub(crate) half_views: Vec<(u8, u32)>,
}

#[cfg(windows)]
impl HostMeta {
    pub(crate) fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }

    fn resident_bytes(&self) -> usize {
        self.user_data.len() * size_of::<victory_shader::emit::UserData>()
            + self.static_samplers.len() * size_of::<victory_shader::emit::StaticSampler>()
            + self.root_cbvs.len()
            + self.root_srvs.len()
            + self.root_uavs.len()
            + self.half_views.len() * size_of::<(u8, u32)>()
    }
}

/// Resident metadata of a translated shader and the console resources its code
/// reads. Compiled `.dxil` bytecode is dropped immediately after `HostPipeline`
/// creation and transiently reloaded from `shader_cache` only if another PSO
/// uses the same shader or a dormant PSO is re-materialized.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: HostMeta,
    dxil_cache_path: Option<PathBuf>,
    /// A compute program's threads in a group.
    pub(crate) threads: Option<[u32; 3]>,
    pub(crate) used: Vec<victory_shader::UsedResource>,
    pub(crate) tables: crate::draw::TablePlan,
    /// Whether the shader's code can store to buffer memory ([`stores`]);
    /// true when its code is not known.
    pub(crate) stores: bool,
    /// Whether it can store through raw addresses ([`stores_by_address`]).
    pub(crate) stores_by_address: bool,
    /// Whether it stores through a view it picks from the heap by an index,
    /// once asked (`crate::picked`).
    pub(crate) picks: std::sync::atomic::AtomicU8,
}

#[cfg(windows)]
impl Shader {
    /// The resident translation metadata, made on first use; zero disk I/O
    /// once initialized. A shader that does not translate is logged once.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        if let Some(existing) = self.translated.get() {
            return existing.as_deref();
        }
        self.init_translated().0
    }

    /// Returns resident translation metadata plus transient `.dxil` bytecode
    /// for `HostPipeline::new`. On first translation, returns the freshly
    /// produced `dxil` directly without an extra disk read; on subsequent PSOs
    /// or dormant re-materialization, reloads only `dxil` from `shader_cache`.
    pub(crate) fn translate_with_dxil(&self) -> Option<(&Translated, Vec<u8>)> {
        let (meta, initial_dxil) = self.init_translated();
        let meta = meta?;
        if let Some(dxil) = initial_dxil {
            return Some((meta, dxil));
        }
        if let Some(path) = meta.dxil_cache_path.as_deref()
            && let Some(dxil) = crate::shader_cache::load_dxil_only(path)
        {
            return Some((meta, dxil));
        }
        let container = self.container_bytes()?;
        let host = crate::shader_cache::host_shader(&container).ok()?;
        Some((meta, host.dxil))
    }

    fn init_translated(&self) -> (Option<&Translated>, Option<Vec<u8>>) {
        if let Some(existing) = self.translated.get() {
            return (existing.as_deref(), None);
        }
        let mut transient_dxil: Option<Vec<u8>> = None;
        let slot = self.translated.get_or_init(|| {
            let Some(container) = self.container_bytes() else {
                tracing::warn!(
                    stage = self.stage,
                    hash = %self.hash,
                    "shader container could not be read back from pack"
                );
                return None;
            };
            let done = translate(&container);
            // `VICTORY_SHADER_SAVE=<dir>`: each shader as the title gave it (`.cdxil`) and as
            // translated (`.dxil`), by the hash crash reports name it with
            if let Some(dir) = std::env::var_os("VICTORY_SHADER_SAVE") {
                let dir = PathBuf::from(dir);
                let _ = std::fs::create_dir_all(&dir);
                let stem = format!("{}.{}", self.hash, self.stage);
                let _ = std::fs::write(dir.join(format!("{stem}.cdxil")), &container);
                if let Ok((_, dxil)) = &done {
                    let _ = std::fs::write(dir.join(format!("{stem}.dxil")), dxil);
                }
            }
            match done {
                Ok((t, dxil)) => {
                    KEPT[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    KEPT[3].fetch_add(
                        t.host.resident_bytes() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        static_samplers = t.host.static_samplers.len(),
                        "shader translated"
                    );
                    transient_dxil = Some(dxil);
                    Some(Box::new(t))
                }
                Err(e) => {
                    tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        "shader does not translate; its pipelines' draws are dropped"
                    );
                    None
                }
            }
        });
        (slot.as_deref(), transient_dxil)
    }
}

#[cfg(windows)]
impl Shader {
    /// The console programs of the container; `None` if they are not known.
    pub(crate) fn programs(&self) -> Option<Vec<victory_shader::NativeProgram>> {
        use victory_shader::{Container, FourCC};
        let container = self.container_bytes()?;
        let c = Container::parse(&container).ok()?;
        let part = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))?;
        victory_shader::native_programs(part.data).ok()
    }
}

#[cfg(windows)]
fn translate(container: &[u8]) -> Result<(Translated, Vec<u8>), String> {
    use victory_shader::{Container, FourCC, HwStage};
    // The translation depends on it (a view index past a capped heap).
    crate::descriptor::host::probe_view_limit();
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let programs = match c.part(FourCC::DXIL).or(c.part(FourCC::ACCC)) {
        Some(part) => victory_shader::native_programs(part.data).map_err(|e| e.to_string())?,
        None => Vec::new(),
    };
    let (host, dxil_cache_path) = crate::shader_cache::host_shader_with_path(container)?;
    let stores = programs.is_empty() || programs.iter().any(stores);
    let threads = programs.iter().find_map(|p| match &p.io {
        victory_shader::native::StageIo::Cs(cs) if cs.threads.iter().all(|&n| n > 0) => Some(cs.threads),
        _ => None,
    });
    let stores_by_address = programs.is_empty() || programs.iter().any(stores_by_address);
    let used = programs
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
            )
        })
        .map(|p| p.used)
        .unwrap_or_default();
    let tables = crate::draw::TablePlan::new(&used, &host.root_indexed_views);
    let dxil = host.dxil;
    let meta = HostMeta {
        user_data: host.user_data,
        static_samplers: host.static_samplers,
        root_cbvs: host.root_cbvs,
        root_srvs: host.root_srvs,
        root_uavs: host.root_uavs,
        half_views: host.half_views,
    };
    Ok((
        Translated {
            host: meta,
            dxil_cache_path,
            threads,
            used,
            tables,
            stores,
            stores_by_address,
            picks: Default::default(),
        },
        dxil,
    ))
}

/// The compute pipelines of the console's ray-tracing structure builder
/// (`XboxBvhBuilder.dll`; its programs are the ones with a spilled
/// user-data table) run **all or none**. Its stages feed each other
/// through memory: a stage that runs on what a dropped stage never wrote
/// computes its store offsets from garbage. The structures lie in plain
/// title memory, between the title's own objects, and a root view has no
/// size, so those stores land on the neighbours (run 98, with 14 of 35
/// stages translated: the title died within a second of the first builder
/// submission, on pointers that had become bounding-box bit patterns).
///
/// So a builder dispatch runs only while every builder pipeline the title
/// has created has a host pipeline; the first builder dispatch makes them
/// all on the spot instead of over the next frames (a stage "still being
/// made" would be a dropped stage). While one cannot be made the builder
/// is off; it is looked at again when the title creates another. A stage
/// that could not be bound stops the stages after it for the rest of that
/// frame and the next ([`builder_rests`]); it turned the builder off for the
/// rest of the run at first.
///
/// shortcut: and only with `VICTORY_BVH_BUILDER=1`, until a run has shown
/// the whole builder and the host side of title memory by address
/// (`memory.rs`) working on a card; nothing reads the structures before
/// the programs that walk them translate.
#[cfg(windows)]
struct Builder {
    pipelines: Vec<Arc<make::Pending>>,
    /// Every one of `pipelines` has a host pipeline.
    whole: bool,
    /// One could not be made, of so many (0: none).
    off: usize,
    /// No stage runs before this frame.
    rests: u64,
}

#[cfg(windows)]
static BUILDER: Mutex<Builder> = Mutex::new(Builder {
    pipelines: Vec::new(),
    whole: false,
    off: 0,
    rests: 0,
});

/// Whether `shaders` are a pipeline of the structure builder.
#[cfg(windows)]
fn of_builder(shaders: &[Arc<Shader>]) -> bool {
    let [shader] = shaders else { return false };
    shader.has_flag(FLAG_BUILDER)
}

/// `VICTORY_BVH_QUARANTINE=1`: what the structure builder's stages store
/// stays in the host's copies and is not copied back into title memory. A
/// check for a first run of the whole builder: a stage that stores where
/// it should not can then not reach the title's own objects (run 98), and
/// the programs that walk the structures read the host's copies anyway.
/// shortcut: not a way to run: the title also reads results back (sizes,
/// counters) and gets none, and a copy's pages are overwritten from title
/// memory when the title writes them.
#[cfg(windows)]
pub(crate) fn builder_quarantined(this: This) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    is_builder(this) && *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_QUARANTINE").is_some_and(|v| v == "1"))
}

/// Whether the title's pipeline `this` is one of the structure builder's.
#[cfg(windows)]
pub(crate) fn is_builder(this: This) -> bool {
    !this.is_null() && pipeline(this).builder
}

/// A stage of the builder was dropped after all: none runs for the rest of
/// this frame and the next (what it fed is not there).
#[cfg(windows)]
pub(crate) fn builder_rests(why: &'static str) {
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    b.rests = crate::frame::current() + 2;
    static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
        tracing::warn!(why, "the ray-tracing structure builder rests for this frame and the next: its stages run all or none");
    }
}

#[cfg(windows)]
fn builder_allowed_for(bvh_builder_env: Option<&std::ffi::OsStr>, rehydrate_via_builder: bool) -> bool {
    bvh_builder_env.is_some_and(|v| v == "1") || rehydrate_via_builder
}

/// At a dispatch of a builder pipeline: whether the builder runs (see
/// [`Builder`]).
#[cfg(windows)]
pub(crate) fn builder_runs() -> bool {
    static ALLOWED: OnceLock<bool> = OnceLock::new();
    if !*ALLOWED.get_or_init(|| {
        builder_allowed_for(
            std::env::var_os("VICTORY_BVH_BUILDER").as_deref(),
            crate::dxr::rehydrate_via_builder(),
        )
    }) {
        return false;
    }
    let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
    if b.off == b.pipelines.len() || crate::frame::current() < b.rests {
        return false;
    }
    if b.whole {
        return true;
    }
    let missing: Vec<String> = b.pipelines.iter().filter(|p| !make::now(p)).map(|p| make::hashes(p)).collect();
    if !missing.is_empty() {
        b.off = b.pipelines.len();
        tracing::warn!(
            missing = missing.len(),
            of = b.pipelines.len(),
            which = ?missing,
            "the ray-tracing structure builder stays off: not all of its programs have a host pipeline, and its stages run all or none"
        );
        return false;
    }
    b.off = 0;
    tracing::info!(
        pipelines = b.pipelines.len(),
        "the ray-tracing structure builder runs: all of its programs have host pipelines"
    );
    b.whole = true;
    true
}

/// Whether a console program can store to buffer memory: it has a buffer,
/// typed-buffer, scalar-memory, flat or global store or atomic, or code
/// that is not all known (an instruction of those kinds outside our table,
/// an indirect jump, code that does not decode). Stores to textures
/// (image instructions) and to the program's own scratch memory do not
/// count: neither reaches a buffer's host copy. After upstream #413,
/// which leaves scalar-memory stores out and counts image stores.
#[cfg(windows)]
fn stores(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        let name = i.name();
        match i.encoding {
            Encoding::Mubuf
            | Encoding::Mtbuf
            | Encoding::Smem
            | Encoding::Flat
            | Encoding::Global => name.is_none_or(|n| {
                (n.contains("store") || n.contains("atomic")) && !n.contains("scratch")
            }),
            Encoding::Sop1 => name.is_none_or(|n| matches!(n, "s_setpc_b64" | "s_swappc_b64")),
            _ => false,
        }
    })
}

/// Whether a console program can store to memory through raw addresses
/// (`crate::memory`): a flat or global store or atomic, or code that does
/// not decode. The programs that only walk ray-tracing structures do not,
/// whatever they store through views.
#[cfg(windows)]
fn stores_by_address(program: &victory_shader::NativeProgram) -> bool {
    use victory_shader::gfx10::{Encoding, decode};
    let Ok(code) = decode(&program.code) else {
        return true;
    };
    code.iter().any(|i| {
        matches!(i.encoding, Encoding::Flat | Encoding::Global)
            && i.name()
                .is_none_or(|n| (n.contains("store") || n.contains("atomic")) && !n.contains("scratch"))
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container (on Windows, shared with `host` to avoid
    /// duplicating `Vec<Arc<Shader>>` across 84,000 pipelines).
    #[cfg(not(windows))]
    shaders: Vec<Arc<Shader>>,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// One of the ray-tracing structure builder's ([`Builder`]).
    #[cfg(windows)]
    builder: bool,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

impl Pipeline {
    fn shaders(&self) -> &[Arc<Shader>] {
        #[cfg(windows)]
        {
            self.host.shaders()
        }
        #[cfg(not(windows))]
        {
            &self.shaders
        }
    }
}

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in p.shaders() {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.size, hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Draws and dispatches dropped since the last [`take_drops`], by reason
/// (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<String, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(
                    hash,
                    "VICTORY_SKIP_PIPELINES expects 16-digit shader hashes"
                );
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    let hashes = HASHES.get_or_init(|| {
        let value = std::env::var("VICTORY_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    });
    if hashes.is_empty() {
        return None;
    }
    p.shaders()
        .iter()
        .find(|shader| hashes.iter().any(|h| h == shader.hash.as_str()))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this` intentionally skipped or dropped
/// for want of a host pipeline (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(this: This) {
    let why = if this.is_null() {
        Some("no pipeline set".to_owned())
    } else {
        let p = pipeline(this);
        skipped_shader(p)
            .map(|hash| format!("skipped shader {hash}"))
            .or_else(|| make::missing(&p.host))
    };
    if let Some(why) = why {
        note_why(why);
    }
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    crate::capture::op(0, || format!("DROPPED {why}"));
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops.get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
}

/// The drops counted since the last call, most first (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut drops: Vec<(String, u64)> = drops.into_iter().flatten().collect();
    // Frame reports only show the first 40 reasons. Keep ties ordered so
    // a HashMap iteration order change cannot change which reasons appear.
    drops.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    drops
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<Arc<host::GraphicsState>>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let builder = of_builder(&shaders);
    #[cfg(windows)]
    let host = Arc::new(make::Pending::new(shaders, graphics, builder));
    #[cfg(windows)]
    known::note(host.key(), host.shaders(), host.graphics.as_deref());
    #[cfg(windows)]
    make::early(&host);
    #[cfg(windows)]
    if builder {
        // Another stage: the group is looked at again at its next dispatch.
        let mut b = BUILDER.lock().unwrap_or_else(|e| e.into_inner());
        b.pipelines.push(host.clone());
        b.whole = false;
    }
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(not(windows))]
        shaders,
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at))
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some(graphics.intern()),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
pub mod known;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;

/// Lets the pipeline library's mapped file go (or tears it down when idle)
/// and evicts 60 s dormant non-pinned PSO COM objects; called by the host's sweep.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
    make::evict_dormant();
}

/// The pipeline library's part of the report when the host device is
/// removed (`library::removal_note`).
#[cfg(windows)]
pub(crate) fn library_removal_note() -> String {
    library::removal_note()
}

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders()
            .iter()
            .map(|s| (s.stage, s.container_bytes().unwrap()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders()[1],
            &pipeline(q).shaders()[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout the title builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn builder_allowed_decouples_rehydration_from_bvh_builder_env() {
        use std::ffi::OsStr;
        // RT-Off: BVH_BUILDER unset or "0", rehydrate_via_builder false -> builder stays off.
        assert!(!builder_allowed_for(None, false));
        assert!(!builder_allowed_for(Some(OsStr::new("0")), false));
        // Dual-run: BVH_BUILDER=1 -> builder runs regardless of rehydrate_via_builder.
        assert!(builder_allowed_for(Some(OsStr::new("1")), false));
        assert!(builder_allowed_for(Some(OsStr::new("1")), true));
        // Suppress-builder with BVH_BUILDER=0: rehydrate_via_builder allows rehydration dispatches.
        assert!(builder_allowed_for(None, true));
        assert!(builder_allowed_for(Some(OsStr::new("0")), true));
    }

    #[test]
    fn container_pack_spills_dedups_reads_concurrently_and_truncates_torn_tail() {
        let dir = tempfile::tempdir().unwrap();
        let pack = Arc::new(ContainerPack::open(dir.path()).expect("open pack"));
        let a = b"fake-cdxil-container-alpha-0123456789".as_slice();
        let b = b"fake-cdxil-container-beta-9876543210abcdef".as_slice();
        let ha = *blake3::hash(a).as_bytes();
        let hb = *blake3::hash(b).as_bytes();

        let (off_a, len_a) = pack.spill(ha, a).expect("spill a");
        let (off_b, len_b) = pack.spill(hb, b).expect("spill b");
        // Duplicate spill returns exact same offset without growing the file.
        assert_eq!(pack.spill(ha, a), Some((off_a, len_a)));
        assert_eq!(pack.read_at(off_a, len_a).as_deref(), Some(a));
        assert_eq!(pack.read_at(off_b, len_b).as_deref(), Some(b));

        // Concurrent positional reads return exact bytes.
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let p = pack.clone();
                std::thread::spawn(move || {
                    for _ in 0..32 {
                        assert_eq!(p.read_at(off_a, len_a).as_deref(), Some(a));
                        assert_eq!(p.read_at(off_b, len_b).as_deref(), Some(b));
                    }
                })
            })
            .collect();
        for r in readers {
            r.join().unwrap();
        }
        drop(pack);

        // Append a torn record tail and reopen: existing records survive and torn tail is cut.
        let pack_path = dir.path().join(PACK_FILE);
        let good_len = std::fs::metadata(&pack_path).unwrap().len();
        {
            let mut f = std::fs::OpenOptions::new().append(true).open(&pack_path).unwrap();
            f.write_all(&PACK_MAGIC).unwrap();
            f.write_all(&[0x55; 12]).unwrap();
        }
        let reopened = ContainerPack::open(dir.path()).expect("reopen pack");
        assert_eq!(std::fs::metadata(&pack_path).unwrap().len(), good_len);
        assert_eq!(reopened.spill(ha, a), Some((off_a, len_a)));
        assert_eq!(reopened.read_at(off_a, len_a).as_deref(), Some(a));
        assert_eq!(reopened.read_at(off_b, len_b).as_deref(), Some(b));
    }
}
