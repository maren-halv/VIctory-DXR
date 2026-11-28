//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// Shader interface metadata shared by every pipeline variant and draw.
pub(crate) struct TablePlan {
    groups: Vec<TableGroup>,
    // Direct entries can interleave slots. Preserve their declaration order
    // when recording resource reads, writes, and transitions.
    direct: Vec<(usize, usize)>,
}

struct TableGroup {
    slot: u8,
    has_cbv: bool,
    has_uav: bool,
    max_dword: u32,
    entries: Vec<UsedResource>,
    indexed: Vec<RootIndexedView>,
    // Descriptor number -> first declaration, matching the bind-time search.
    by_descriptor: Vec<(u32, usize)>,
}

impl TablePlan {
    pub(crate) fn new(used: &[UsedResource], indexed: &[RootIndexedView]) -> Self {
        let mut slots: Vec<_> = used
            .iter()
            .filter(|u| u.root == RootType::Table)
            .map(|u| u.slot)
            .collect();
        slots.sort_unstable();
        slots.dedup();
        let mut groups: Vec<_> = slots
            .iter()
            .map(|&slot| TableGroup {
                slot,
                has_cbv: false,
                has_uav: false,
                max_dword: 0,
                entries: Vec::new(),
                by_descriptor: Vec::new(),
                indexed: indexed.iter().filter(|v| v.table_slot == slot).copied().collect(),
            })
            .collect();
        let mut direct = Vec::new();
        for &u in used.iter().filter(|u| u.root == RootType::Table) {
            let i = slots
                .binary_search(&u.slot)
                .expect("table slot collected from these declarations");
            let group = &mut groups[i];
            group.has_cbv |= u.kind == ResourceType::Cbv;
            group.has_uav |= u.kind == ResourceType::Uav;
            group.max_dword = group.max_dword.max(u.table_offset_dwords);
            direct.push((i, group.entries.len()));
            group
                .by_descriptor
                .push((u.table_offset_dwords / 8, group.entries.len()));
            group.entries.push(u);
        }
        for group in &mut groups {
            group.by_descriptor.sort_unstable();
            group.by_descriptor.dedup_by_key(|item| item.0);
        }
        Self { groups, direct }
    }
}

impl TableGroup {
    fn entry(&self, descriptor: u32) -> Option<&UsedResource> {
        let i = self
            .by_descriptor
            .binary_search_by_key(&descriptor, |item| item.0)
            .ok()?;
        Some(&self.entries[self.by_descriptor[i].1])
    }

    fn needs_snapshot(&self, heap: &crate::descriptor::Heap, host_count: u32, table: u64) -> bool {
        let index_at = |dword: u32| {
            table
                .checked_add(u64::from(dword) * 4)
                .and_then(|end| heap.index(end))
        };
        match index_at(self.max_dword) {
            Some(i) => i >= host_count,
            None => self
                .entries
                .iter()
                .any(|u| index_at(u.table_offset_dwords).is_some_and(|i| i >= host_count)),
        }
    }
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Few<u32, 16>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64, bool)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// Whether a draw of the title's bake finds the textures it picks by index
/// as title memory has them at that moment (`VICTORY_BAKE_EXACT=0`: as the
/// pass leaves them, as before). The pass looks at a slot whose memory was
/// written or mapped a frame or two later, by when the rest of it has
/// landed, and at one uploaded whole a few frames later again, for a title
/// that fills memory unnoticed: a bake in between took the bytes from before
/// (another texture's), and what it bakes stays: coloured specks all over
/// the waitress's face for the whole diner scene (run n01), none in other
/// runs. So the pass ahead of a bake takes those slots too, and its uploads
/// wait for title memory that is being remapped (`crate::texture` `BAKING`).
fn bake_exact() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BAKE_EXACT").is_none_or(|v| v != "0"))
}

/// Whether a texture found changed with nothing to tell of it is looked at
/// again in the next frame (`VICTORY_RECHECK_SOON=0`: after four, as
/// before).
fn recheck_soon() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RECHECK_SOON").is_none_or(|v| v != "0"))
}

/// Frames ahead the pass before a bake takes slots from that are due only
/// then ([`bake_exact`]): those of notices (1 and 2), and the first looks
/// again ([`RECHECK_FRAMES`]), these at the frame's first bake alone (a
/// look reads 48 KiB, and a frame bakes up to sixteen times).
const BAKE_AHEAD: [u64; 2] = [12, 2];

/// What [`crate::descriptor::host::HostHeap::texture_news`] said when the
/// pass before a bake last served all there was, plus one (0: it did not):
/// until something is new, the next bakes of the frame need no pass.
static BAKE_SERVED: AtomicU64 = AtomicU64::new(0);

/// Pending descriptor publications for an async decoration bake upload pass:
/// uploads record onto the bake draw's own command list (running in its prelude
/// ahead-list) and slots are published via `list.defer_host` right before
/// `ExecuteCommandLists([prelude, list])` (`ARCH-02`).
pub(crate) struct BakeSync {
    publish: Vec<(u32, u64, u64)>,
    keep: Vec<std::sync::Arc<crate::resource::HostTexture>>,
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
pub(crate) const BAKE_PIXEL_SHADERS: [&str; 7] = [
    "c0a7fe77b1862e10",
    "0d1575a396dfe513",
    "b7cfd8dda91ebb6b",
    "280967ad158cee0b",
    "fa455c8a3c32577b",
    "d126bcae5d5d8d26",
    "fcee612943c008e0",
];

/// Geometry stages shared by every ped texture compositor / decoration bake
/// permutation (`vertex 089d6271be6650ef` fullscreen-triangle ped baker and
/// `mesh d4111a6e6ea49d91` decoration projection mesh), catching pixel-shader
/// permutations beyond `BAKE_PIXEL_SHADERS`.
const BAKE_GEOMETRY_SHADERS: [&str; 2] = ["089d6271be6650ef", "d4111a6e6ea49d91"];

pub(crate) fn is_bake_shader_hash(hash: &str) -> bool {
    BAKE_PIXEL_SHADERS.contains(&hash) || BAKE_GEOMETRY_SHADERS.contains(&hash)
}

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours and no barrier (the scratch
/// buffer stays readable as arguments until the next clamp), until one of
/// these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// Whether indirect executions whose arguments or count lie in title memory
/// the card reads in place (renderer port 1, `VICTORY_PORT_CONSTANTS=1`) are
/// clamped alone, right before they run (`VICTORY_PORT_CONSTANTS_ORDER=1`;
/// default off: in batches and windows as any other).
fn in_place_order() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_CONSTANTS_ORDER").is_some_and(|v| v == "1"))
}

/// With [`in_place_order`]: such executions never in a batch
/// (`VICTORY_PORT_CONSTANTS_ORDER_ALONE=1`; default off: share a batch opened
/// after the last shader that may write buffers).
fn in_place_alone() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_CONSTANTS_ORDER_ALONE").is_some_and(|v| v == "1"))
}

/// Cache-line-aligned wrapper preventing false sharing across concurrent
/// recording threads (`CPU-26`).
#[repr(align(64))]
struct CachePadded<T>(T);

static IN_PLACE_INDIRECT: [CachePadded<AtomicU64>; 2] =
    [const { CachePadded(AtomicU64::new(0)) }; 2];
#[cfg(feature = "diagnostics")]
static IN_PLACE_RANGES: std::sync::Mutex<std::collections::BTreeMap<u64, [u64; 3]>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

fn in_place_indirect(arguments: u64, kind: u32) {
    IN_PLACE_INDIRECT[0].0.fetch_add(1, Ordering::Relaxed);
    IN_PLACE_INDIRECT[1].0.fetch_add(u64::from(in_place_order()), Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        let slot = match kind {
            2 => 1,
            10 => 2,
            _ => 0,
        };
        let mut ranges = IN_PLACE_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        ranges.entry(arguments >> 20).or_default()[slot] += 1;
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (arguments, kind);
}

pub(crate) fn in_place_indirect_line(frames: f64) {
    let [all, alone] = [0, 1].map(|i| IN_PLACE_INDIRECT[i].0.swap(0, Ordering::Relaxed));
    if all == 0 {
        return;
    }
    #[cfg(feature = "diagnostics")]
    {
        let ranges = std::mem::take(&mut *IN_PLACE_RANGES.lock().unwrap_or_else(|e| e.into_inner()));
        let mut by: Vec<(u64, [u64; 3])> = ranges.into_iter().collect();
        by.sort_by_key(|(_, n)| std::cmp::Reverse(n.iter().sum::<u64>()));
        let top: Vec<String> = by
            .iter()
            .take(6)
            .map(|(mb, n)| {
                format!(
                    "{:#x}+1MiB: {:.0} draws {:.0} dispatches {:.0} mesh",
                    mb << 20,
                    n[0] as f64 / frames,
                    n[1] as f64 / frames,
                    n[2] as f64 / frames
                )
            })
            .collect();
        tracing::info!(
            "renderer port 1: indirect executions with arguments or count in place per frame: {:.0}, of them {:.0} clamped after every shader write before them (VICTORY_PORT_CONSTANTS_ORDER); from {} ranges of 1 MiB, the busiest: {}",
            all as f64 / frames,
            alone as f64 / frames,
            by.len(),
            top.join("; ")
        );
    }
    #[cfg(not(feature = "diagnostics"))]
    tracing::info!(
        "renderer port 1: indirect executions with arguments or count in place per frame: {:.0}, of them {:.0} clamped after every shader write before them (VICTORY_PORT_CONSTANTS_ORDER)",
        all as f64 / frames,
        alone as f64 / frames
    );
}

/// Whether indirect executions of one record are clamped in batches
/// (`crate::indirect::batch`; `VICTORY_INDIRECT_BATCH=0`: each on its own,
/// or in windows ([`windows`])).
///
/// A frame of gameplay has 4,809 indirect executions (a capture: 3,523 mesh
/// draws, 948 dispatches, 338 draws), and with the windows still 2,355
/// clamps: each a dispatch between two full waits, with the scratch buffer
/// moved twice. The replayed frame's 84.6 ms are 75.3 ms with the clamps in
/// 133 passes, and 77.7 ms in the 980 of this.
///
/// A batch is opened by the first execution after a point where arguments
/// may have changed, and all executions up to the next such point read
/// theirs from it, each exactly the record it asks for (windows guess which
/// are next, and clamp what they do not know). Arguments in buffers change
/// by dispatches and by draws whose shaders store to buffers; those end the
/// batch, so the record an execution reads is clamped after all that was
/// recorded before it, and the pass waits for all of it, exactly as the
/// clamp of its own did. Between two executions of a batch is no wait of
/// ours: there is nothing between them to wait for but draws that store to
/// no buffer.
///
/// `VICTORY_INDIRECT_BATCH=2`: a dispatch does not end the batch before the
/// title waits for it (a barrier, or a pipeline sync) or draws: what a
/// dispatch writes is not there for a later command without a wait, and
/// dispatches the title runs side by side (its tile classes, up to 92 at a
/// time) are not held up by waits of ours. Off: whether the title has ways
/// to wait that do not pass through its lists' methods is not known.
fn batches() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| batches_for(std::env::var_os("VICTORY_INDIRECT_BATCH").as_deref()))
}

fn batches_for(value: Option<&std::ffi::OsStr>) -> u8 {
    match value.and_then(|v| v.to_str()) {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    }
}

/// The records clamped by the pass opened last ([`batches`]); the backing
/// [`crate::indirect::batch::Table`] lives on [`State::batch_table`] so it is
/// reused across batches without 4.6 KiB stack zeroing (`CPU-29`).
struct Batch {
    scratch: ID3D12Resource,
    /// Dispatches since, which may have written arguments
    /// (`VICTORY_INDIRECT_BATCH=2`).
    wrote: bool,
}

/// Whether any stage of graphics pipeline `p` can write to a buffer (excluding
/// texture-only UAV stores so they do not invalidate an open clamp batch).
fn pipeline_stores_buffers(p: &HostPipeline) -> bool {
    let picked = crate::picked::on();
    p.stages.iter().any(|stage| {
        let t = stage.translated();
        if !t.stores {
            return false;
        }
        t.stores_by_address
            || !stage.root_uavs.is_empty()
            || t.used.iter().any(|u| u.kind == ResourceType::Uav && u.dwords != 8)
            || (picked && crate::picked::picks(&stage.shader, &t.picks))
    })
}

const ROOT_CACHE_PARAMS: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum RootGpuKind {
    #[default]
    None,
    Cbv,
    Srv,
    Uav,
    Table,
}

/// Per-domain (graphics or compute) root-argument bind cache on [`State`].
/// Cleared whenever the domain's root signature changes or an internal/external
/// pass clobbers the command list's root state.
#[derive(Clone, Copy)]
struct RootArgCache {
    /// Up to 2 root-constant parameters per domain (e.g. VS/MS + PS, or CS):
    /// `(param, dword_len, values)`. `param == u32::MAX` means empty.
    consts: [(u32, u8, [u32; 64]); 2],
    /// Root descriptors / descriptor tables for params `0..ROOT_CACHE_PARAMS`.
    gpu: [(RootGpuKind, u64); ROOT_CACHE_PARAMS],
}

impl Default for RootArgCache {
    fn default() -> Self {
        Self {
            consts: [(u32::MAX, 0, [0; 64]); 2],
            gpu: [(RootGpuKind::None, 0); ROOT_CACHE_PARAMS],
        }
    }
}

impl RootArgCache {
    #[inline]
    fn clear(&mut self) {
        self.consts[0].0 = u32::MAX;
        self.consts[1].0 = u32::MAX;
        self.gpu = [(RootGpuKind::None, 0); ROOT_CACHE_PARAMS];
    }

    #[inline]
    fn check_or_set_consts(&mut self, param: u32, values: &[u32]) -> bool {
        let len = values.len();
        if len > 64 {
            self.invalidate_consts(param);
            return false;
        }
        for slot in &mut self.consts {
            if slot.0 == param {
                if usize::from(slot.1) == len && &slot.2[..len] == values {
                    return true;
                }
                slot.1 = len as u8;
                slot.2[..len].copy_from_slice(values);
                return false;
            }
        }
        let dst = if self.consts[0].0 == u32::MAX {
            &mut self.consts[0]
        } else if self.consts[1].0 == u32::MAX {
            &mut self.consts[1]
        } else {
            &mut self.consts[0]
        };
        dst.0 = param;
        dst.1 = len as u8;
        dst.2[..len].copy_from_slice(values);
        false
    }

    #[inline]
    fn invalidate_consts(&mut self, param: u32) {
        for slot in &mut self.consts {
            if slot.0 == param {
                slot.0 = u32::MAX;
            }
        }
    }

    #[inline]
    fn check_or_set_gpu(&mut self, param: u32, kind: RootGpuKind, val: u64) -> bool {
        let idx = param as usize;
        if idx >= ROOT_CACHE_PARAMS {
            return false;
        }
        let slot = &mut self.gpu[idx];
        if *slot == (kind, val) {
            return true;
        }
        *slot = (kind, val);
        false
    }
}

#[inline]
fn keep_shadow_dedup(
    kept: &mut [usize; 8],
    next: &mut u8,
    list: &HostList,
    shadow: &std::sync::Arc<crate::shadow::Shadow>,
) {
    let key = std::sync::Arc::as_ptr(shadow) as usize;
    if kept.contains(&key) {
        return;
    }
    kept[*next as usize] = key;
    *next = (*next + 1) & 7;
    list.keep_shadow(shadow.clone());
}

/// Compact cached buffer-UAV write declaration captured during an uncached
/// `State::tables()` pass and replayed identically on `last_stage_tables` hits.
#[derive(Clone)]
struct CachedUavBuffer {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    address: u64,
    len: u64,
    /// Whether `STORES.get() || !write_skip()` was true for the stage.
    writes: bool,
}

/// Scratch accumulator populated during an uncached `State::tables()` pass
/// when `can_cache_tables` is true.
#[derive(Default)]
pub(crate) struct UavCapture {
    buffers: Vec<CachedUavBuffer>,
    textures: Few<usize, 4>,
    srv_textures: Few<usize, 8>,
    srv_tex_hazard: bool,
    skipped_srv_check: bool,
}

/// Cached per-stage descriptor-table bind result for consecutive draws/dispatches
/// on the same `HostList` when pipeline, heaps, table roots, descriptor view
/// epochs, and list epoch are unchanged — including UAV-writing dispatches whose
/// write marks and barriers are replayed directly from `uav_buffers` / `uav_textures`.
struct CachedStageTables {
    list_ptr: usize,
    list_epoch: u64,
    pipeline: usize,
    heaps: [usize; 2],
    table_gen: u64,
    const_gen: u64,
    view_epochs: (u64, u64),
    writes_mask: u8,
    has_indexed: bool,
    tested_depth: usize,
    tested_depth_read: bool,
    stages_made: [Few<(u8, u32), 16>; 4],
    uav_buffers: Vec<CachedUavBuffer>,
    uav_textures: Few<usize, 4>,
    /// True if any SRV in the cached tables could observe a texture-UAV write
    /// (format alias, depth twin, or resource overlap), preventing `list_epoch`
    /// advancement after a texture-UAV hit.
    srv_tex_hazard: bool,
}

impl CachedStageTables {
    #[inline]
    fn matches(
        &self,
        list_ptr: usize,
        list_epoch: u64,
        pipeline: usize,
        heaps: [usize; 2],
        table_gen: u64,
        const_gen: u64,
        tested_depth: usize,
        view_epochs: (u64, u64),
        writes_mask: u8,
    ) -> bool {
        self.list_ptr == list_ptr
            && self.list_epoch == list_epoch
            && self.pipeline == pipeline
            && self.heaps == heaps
            && self.table_gen == table_gen
            && self.tested_depth == tested_depth
            && (!self.has_indexed || self.const_gen == const_gen)
            && self.view_epochs == view_epochs
            && self.writes_mask == writes_mask
            && !self.uav_buffers.iter().any(|u| u.shadow.is_dead())
    }

    /// Replays every UAV refresh, `mark_uav_dirty`, `gpu_writes`, `texture_written`,
    /// depth-twin sync, and `UNORDERED_ACCESS` state barrier that the uncached
    /// `readable()` pass would emit.
    fn replay_uavs(&self, list: &HostList) {
        const UNORDERED_ACCESS: u32 = 0x8;
        for u in &self.uav_buffers {
            u.shadow.refresh(list, u.address, u.len);
            list.mark_uav_dirty();
            if u.writes {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(u.address, u.len, 0);
                u.shadow.gpu_writes(list, u.address, u.len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, u.len);
                }
            }
        }
        for &resource in &*self.uav_textures {
            list.mark_uav_dirty();
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(resource as This);
            list.texture_written(resource as This);
            if !list.twin_use(resource as This, true, true) {
                crate::list::ensure_state(list, resource as This, UNORDERED_ACCESS);
            }
        }
    }
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    /// The render target views and the depth view the host list has, once
    /// it has any ([`State::set_targets`]): per view the host texture (0:
    /// a null view) and what its descriptor says.
    host_targets: Option<([TargetKey; 8], usize, Option<TargetKey>)>,
    /// Fixed-function state last set on the host list (cleared on reset/replay).
    host_topology: Option<u32>,
    host_viewports_set: bool,
    host_scissors_set: bool,
    host_blend: Option<Option<[u32; 4]>>,
    host_stencil: Option<u32>,
    host_depth_bounds: Option<[u32; 2]>,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// `HostList` pointer that owns `self.window` (0: none), avoiding per-draw
    /// `list.first_use(WINDOW_USE)` `deferred.lock()` (`CPU-29`).
    window_list: usize,
    /// The records clamped in a batch, if later executions may still read
    /// them ([`batches`]).
    batch: Option<Batch>,
    /// Reusable [`crate::indirect::batch::Table`] backing `self.batch` (`CPU-29`).
    batch_table: Option<Box<crate::indirect::batch::Table>>,
    /// `HostList` pointer that owns `self.batch` (0: none), avoiding per-draw
    /// `list.first_use(BATCH_USE)` `deferred.lock()` (`CPU-29`).
    batch_list: usize,
    /// Per-list MRU of host copies indirect arguments were read from last,
    /// avoiding global mutex contention and cross-list pollution.
    argument_copies: Vec<std::sync::Weak<crate::shadow::Shadow>>,
    /// Reusable scratch buffer for [`State::argument_copies`] (`CPU-29`).
    recent_args: Vec<(u64, ID3D12Resource, bool)>,
    /// Per-list cached `heap_buffers` result `(list_ptr, view_heap, strides_gpu)`
    /// (`CPU-29`), eliminating per-draw `list.deferred.lock()` and heap walks.
    cached_heap_buffers: (usize, usize, u64),
    /// Per-list cached `heap_strides` result `(list_ptr, view_heap, strides_gpu)`.
    cached_heap_strides: (usize, usize, u64),
    /// Last ensured render-target/depth transition tuple
    /// `(list_ptr, list_epoch, targets_len, targets, depth, depth_wanted)`
    /// (`CPU-21`), skipping redundant `ensure_state` + epoch-bump on consecutive
    /// draws when no target has a format alias.
    ensured_targets: (usize, u64, usize, [usize; 8], usize, u32),
    /// Reusable scratch records buffer for [`State::tables`] (`CPU-18`),
    /// avoiding `Few<[u32; 3], 64>` stack zeroing on every dynamic table.
    table_records: Vec<[u32; 3]>,
    /// Reusable per-list scratch buffers for multi-arg indirect VA rebasing
    /// (`State::execute_indirect_signature`), avoiding per-call allocations.
    indirect_rebase: Vec<[u32; 8]>,
    indirect_rebase_ranges: Vec<(u64, u64, u64, bool)>,
    indirect_rebase_seen: Vec<u64>,
    /// Per-list batched draw/dispatch/spread counters flushed on list close/reset
    /// (`CPU-26`), eliminating per-draw cross-core atomic cache-line ping-pong.
    local_drawn: u64,
    local_dispatched: u64,
    local_spread: [u64; 2],
    /// A direct dispatch spread across z, for [`State::host_word`] at its
    /// bind: `[row, asked x]` (`[0, 0]`: not spread; an indirect one gets
    /// its words from the clamp through its command signature).
    dispatch_spread: [u32; 2],
    /// A `DispatchX` that counts threads, for [`State::host_word`] at its
    /// bind: the count per axis where it is not a whole number of groups,
    /// else 0 (`USER_DISPATCH_THREADS`; fork pull request #646).
    dispatch_threads: [u32; 3],
    /// The user data bound last, of the last stage that has some (for the
    /// report of a removed device, `crate::host::indirects`).
    #[cfg(feature = "diagnostics")]
    user: Vec<u32>,
    /// The tables the last bind made afresh for a program the stand-in for
    /// the indirect light follows (`crate::ambient`): root parameter, bytes
    /// into the view heap.
    ambient_tables: Vec<(u8, u32)>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    /// Per-domain root-argument shadow cache (`[graphics, compute]`).
    host_root_args: [RootArgCache; 2],
    /// Bitmask of vertex buffer slots currently bound on the host list matching `self.vertex`.
    host_vb_bound: u32,
    /// Whether `self.index` is currently bound on the host list.
    host_ib_bound: bool,
    host_heaps: [usize; 2],
    host_heaps_set: bool,
    /// Tables made afresh in this list, by a hash of the table and what it
    /// holds ([`table_hash`]).
    fresh_tables: std::collections::HashMap<u64, FreshTable, std::hash::BuildHasherDefault<Plain>>,
    /// The record words of [`FreshTable`] tables, one after the other.
    fresh_records: Vec<[u32; 3]>,
    /// The dynamic slots the list holds: title heap, that heap's identity
    /// (`HostHeap::id`), first, count; given back when the list has run, is
    /// reset or released.
    dynamic: Vec<(usize, u64, u32, u32)>,
    /// Unused tail of this list's per-list dynamic descriptor bump slab:
    /// `(title heap, host heap id, next slot, remaining count)`. Returned on
    /// list close/reset/detach so idle/in-flight lists hold zero wasted slots.
    slab: Option<(usize, u64, u32, u32)>,
    /// Next bump-slab chunk size (64 -> 128 -> 256) as a list allocates more tables.
    slab_step: u32,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<
        (usize, u64),
        RootSlot,
        std::hash::BuildHasherDefault<crate::host::AddressHash>,
    >,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<
        (usize, u64),
        RootSlot,
        std::hash::BuildHasherDefault<crate::host::AddressHash>,
    >,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    /// Per-list readiness `(epoch, tested_depth, depth_read)` and table-stamp
    /// state `(title_slot, read)` for listed views, split out of the persistent
    /// thread-local CPU descriptor cache (`ListViews`).
    listed_state: std::collections::HashMap<
        [u32; 3],
        ((u64, usize, bool), Option<(u32, bool)>),
        std::hash::BuildHasherDefault<crate::host::AddressHash>,
    >,
    /// Cached `(raw COM ptr, base GPU VA)` for the current list's scratch
    /// upload buffer, avoiding per-CBV `GetGPUVirtualAddress()` COM calls.
    scratch_base: (usize, u64),
    /// Per-list dedup ring of `Arc<Shadow>` pointers already retained on the
    /// current `HostList`, avoiding `list.deferred.lock()` on repeated in-place uses.
    kept_shadows: [usize; 8],
    kept_shadows_next: u8,
    /// Monotonic generation bumped when root descriptor tables or constants change (`[graphics, compute]`).
    table_gen: [u64; 2],
    const_gen: [u64; 2],
    /// Last stage-table bind result per domain (`[graphics, compute]`), skipped when
    /// consecutive draws share identical pipeline/heaps/tables/epoch with no UAV stores.
    last_stage_tables: [Option<CachedStageTables>; 2],
    /// Failure category of the most recent dropped draw/dispatch on this list.
    last_drop_reason: Option<DropReason>,
}

/// Owned dynamic descriptor slot ranges detached from a submitted list's
/// [`State`] so the `State` can retain its collection capacities across reuse.
#[derive(Default)]
pub(crate) struct DynamicLease(Vec<(usize, u64, u32, u32)>);

impl DynamicLease {
    pub(crate) fn release(&mut self) {
        if !self.0.is_empty() {
            crate::descriptor::release_dynamic_batch(&self.0);
            self.0.clear();
        }
    }
}

impl Drop for DynamicLease {
    fn drop(&mut self) {
        self.release();
    }
}

impl Drop for State {
    fn drop(&mut self) {
        self.close_batch();
        self.flush_counters();
        self.trim_dynamic_slab();
        if !self.dynamic.is_empty() {
            crate::descriptor::release_dynamic_batch(&self.dynamic);
            self.dynamic.clear();
        }
    }
}

impl State {
    /// Flushes and closes any open indirect clamp batch, writing WC `table[0]`
    /// once while retaining `batch_table` for reuse (`CPU-29`).
    #[inline]
    fn close_batch(&mut self) {
        if self.batch.take().is_some()
            && let Some(table) = self.batch_table.as_deref_mut()
        {
            table.finish();
        }
        self.batch_list = 0;
    }

    /// Flushes per-list batched draw/dispatch counters to the padded globals
    /// (`CPU-26`).
    #[inline]
    fn flush_counters(&mut self) {
        if self.local_drawn != 0 {
            DRAWN.0.fetch_add(std::mem::take(&mut self.local_drawn), Ordering::Relaxed);
        }
        if self.local_dispatched != 0 {
            DISPATCHED
                .0
                .fetch_add(std::mem::take(&mut self.local_dispatched), Ordering::Relaxed);
        }
        if self.local_spread[0] != 0 {
            SPREAD[0].fetch_add(std::mem::take(&mut self.local_spread[0]), Ordering::Relaxed);
        }
        if self.local_spread[1] != 0 {
            SPREAD[1].fetch_add(std::mem::take(&mut self.local_spread[1]), Ordering::Relaxed);
        }
    }

    /// Clears per-list transient caches bound to the current `HostList`.
    #[inline]
    fn clear_list_caches(&mut self) {
        self.close_batch();
        self.window = None;
        self.window_list = 0;
        self.cached_heap_buffers = (0, 0, 0);
        self.cached_heap_strides = (0, 0, 0);
        self.ensured_targets.0 = 0;
        self.last_stage_tables = [None, None];
    }

    /// Returns any unused tail of this list's dynamic bump slab to its heap,
    /// flushes any open indirect batch table count, and flushes batched draw
    /// counters. Called on `Close()` so closed/in-flight lists hold zero wasted
    /// slots and every indirect batch has its final job count written.
    pub(crate) fn trim_dynamic_slab(&mut self) {
        self.clear_list_caches();
        self.flush_counters();
        if let Some((heap, id, first, n)) = self.slab.take()
            && n > 0
        {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
    }

    /// Allocates `n` contiguous dynamic slots from `host` using this list's
    /// lock-free per-list bump slab (falling back to exact allocation under
    /// slot pressure) and records the used span in `self.dynamic` (coalescing
    /// contiguous runs).
    fn alloc_dynamic(
        &mut self,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        n: u32,
    ) -> Option<u32> {
        const SLAB_MAX: u32 = 256;
        let heap_ptr = std::ptr::from_ref(heap) as usize;
        let host_id = host.id();
        if let Some((sh, sid, cur, rem)) = self.slab
            && sh == heap_ptr
            && sid == host_id
            && rem >= n
        {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
            let first = cur;
            let left = rem - n;
            self.slab = (left > 0).then_some((sh, sid, cur + n, left));
            self.record_dynamic(heap_ptr, host_id, first, n);
            return Some(first);
        }
        let recycle = match self.slab.take() {
            Some((sh, sid, cur, rem)) if sh == heap_ptr && sid == host_id && rem > 0 => Some((cur, rem)),
            Some((sh, sid, cur, rem)) => {
                if rem > 0 {
                    crate::descriptor::release_dynamic(sh, sid, cur, rem);
                }
                None
            }
            None => None,
        };
        let want = if n <= SLAB_MAX {
            let step = self.slab_step.clamp(64, SLAB_MAX);
            self.slab_step = (step * 2).min(SLAB_MAX);
            step.max(n)
        } else {
            n
        };
        let (first, granted) = host.dynamic_chunk(n, want, recycle).or_else(|| {
            crate::host::settle_for_fence();
            host.dynamic_chunk(n, n, None)
        })?;
        self.record_dynamic(heap_ptr, host_id, first, n);
        if granted > n {
            self.slab = Some((heap_ptr, host_id, first + n, granted - n));
        }
        Some(first)
    }

    #[inline]
    fn record_dynamic(&mut self, heap_ptr: usize, host_id: u64, first: u32, n: u32) {
        if let Some(last) = self.dynamic.last_mut()
            && last.0 == heap_ptr
            && last.1 == host_id
            && last.2.wrapping_add(last.3) == first
        {
            last.3 += n;
        } else {
            self.dynamic.push((heap_ptr, host_id, first, n));
        }
    }

    /// Detaches this list's dynamic descriptor slots for a submitted [`Detached`]
    /// host list while clearing `State` in place to keep its heap capacities.
    pub(crate) fn take_dynamic(&mut self) -> DynamicLease {
        self.trim_dynamic_slab();
        let taken = std::mem::take(&mut self.dynamic);
        self.fresh_tables.clear();
        self.fresh_records.clear();
        self.listed_state.clear();
        self.scratch_base = (0, 0);
        self.kept_shadows = [0; 8];
        self.kept_shadows_next = 0;
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.slab_step = 0;
        self.tables_gone = true;
        DynamicLease(taken)
    }

    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        self.trim_dynamic_slab();
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        if !self.dynamic.is_empty() {
            crate::descriptor::release_dynamic_batch(&self.dynamic);
            self.dynamic.clear();
        }
        self.fresh_tables.clear();
        self.fresh_records.clear();
        self.listed_state.clear();
        self.scratch_base = (0, 0);
        self.kept_shadows = [0; 8];
        self.kept_shadows_next = 0;
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.slab_step = 0;
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.0.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Why a recorded draw, mesh dispatch or compute dispatch was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropReason {
    /// No translated/built host PSO or pipeline kind mismatch.
    NoPso,
    /// Intentionally skipped (RT-off builder, diagnostics skip, unsupported topology/mesh test).
    LeftOut,
    /// Missing host shadow buffer for index/vertex/indirect arguments or direct root UAV.
    NoBuffer,
    /// Exhausted dynamic descriptor slots or missing bound view heap.
    NoSlots,
}

static DROPPED_NO_PSO: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));
static DROPPED_LEFT_OUT: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));
static DROPPED_NO_BUFFER: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));
static DROPPED_NO_SLOTS: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));

#[inline]
fn record_drop(reason: DropReason) {
    DROPPED.0.fetch_add(1, Ordering::Relaxed);
    match reason {
        DropReason::NoPso => DROPPED_NO_PSO.0.fetch_add(1, Ordering::Relaxed),
        DropReason::LeftOut => DROPPED_LEFT_OUT.0.fetch_add(1, Ordering::Relaxed),
        DropReason::NoBuffer => DROPPED_NO_BUFFER.0.fetch_add(1, Ordering::Relaxed),
        DropReason::NoSlots => DROPPED_NO_SLOTS.0.fetch_add(1, Ordering::Relaxed),
    };
}

/// Returns `(no_pso, left_out, no_buffer, no_slots)` drops since the last call.
pub(crate) fn take_dropped_split() -> (u64, u64, u64, u64) {
    (
        DROPPED_NO_PSO.0.swap(0, Ordering::Relaxed),
        DROPPED_LEFT_OUT.0.swap(0, Ordering::Relaxed),
        DROPPED_NO_BUFFER.0.swap(0, Ordering::Relaxed),
        DROPPED_NO_SLOTS.0.swap(0, Ordering::Relaxed),
    )
}

/// The draws dropped since [`take_dropped`].
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn dropped_now() -> u64 {
    DROPPED.0.load(Ordering::Relaxed)
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// since the last call.
pub(crate) fn take_dropped() -> u64 {
    let total = DROPPED.0.swap(0, Ordering::Relaxed);
    let (no_pso, left_out, no_buffer, no_slots) = take_dropped_split();
    if total > 0 {
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        let n = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() || n.is_multiple_of(64) {
            tracing::warn!(
                frame = crate::frame::current(),
                total,
                no_pso,
                left_out,
                no_buffer,
                no_slots,
                "draws/dispatches dropped"
            );
        }
    }
    total
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Of the passes, nanoseconds since the last frame line: looking at slots,
/// the spans and refreshes of their copies, the texture side.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_SPLIT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Passes after which every buffer slot of the heap was looked at again.
const HEAP_ROUND: u64 = 16;

/// Whether the slots looked at again for no write are a part of them at
/// every pass, each slot at every sixteenth (`VICTORY_HEAP_SPREAD=0`: all of
/// them at every sixteenth pass, as before).
///
/// All at once was some 115 ms on a thread that records the frame, every
/// sixteenth frame (72,000 slots; run record-1: 11 ms a pass in the lines
/// without such a pass, 22 in those of ten frames with one). A slot is
/// looked at as often as before.
fn heap_spread() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SPREAD").is_none_or(|v| v != "0"))
}

/// The host copy and range the buffer view of each of a heap's slots reads
/// ([`State::heap_buffers`]), by slot (kept for unit tests).
#[cfg(test)]
#[derive(Default)]
struct HeapRanges<T>(Vec<Option<(T, u64, u64)>>);

/// Cached slot entry: `(shadow, record_key, is_uav)`.
type CachedSlot = (std::sync::Arc<crate::shadow::Shadow>, u64, bool);

/// Compact active-slot table for [`BufferSweep`]: maps sparse heap slot
/// indices (~1M) to a dense vector of occupied buffer slots (~200k) so the
/// per-frame span reduction scans only occupied entries (~5x less memory).
#[derive(Default)]
struct DenseRanges {
    slot_pos: Vec<u32>,
    dense: Vec<(u32, CachedSlot, u64, u64)>,
}

impl DenseRanges {
    #[inline]
    fn get(&self, slot: u32) -> Option<&(u32, CachedSlot, u64, u64)> {
        let pos = *self.slot_pos.get(slot as usize)?;
        (pos != u32::MAX).then(|| &self.dense[pos as usize])
    }

    fn clear(&mut self) {
        for &(slot, ..) in &self.dense {
            self.slot_pos[slot as usize] = u32::MAX;
        }
        self.dense.clear();
    }

    fn set(&mut self, slot: u32, entry: Option<(CachedSlot, u64, u64)>) {
        let idx = slot as usize;
        match entry {
            Some((cached, from, to)) => {
                if idx >= self.slot_pos.len() {
                    self.slot_pos.resize(idx + 1, u32::MAX);
                }
                let pos = self.slot_pos[idx];
                if pos != u32::MAX {
                    self.dense[pos as usize] = (slot, cached, from, to);
                } else {
                    self.slot_pos[idx] = self.dense.len() as u32;
                    self.dense.push((slot, cached, from, to));
                }
            }
            None => {
                if idx < self.slot_pos.len() {
                    let pos = self.slot_pos[idx];
                    if pos != u32::MAX {
                        self.slot_pos[idx] = u32::MAX;
                        let last = self.dense.len() - 1;
                        self.dense.swap_remove(pos as usize);
                        if (pos as usize) < last {
                            let moved_slot = self.dense[pos as usize].0;
                            self.slot_pos[moved_slot as usize] = pos;
                        }
                    }
                }
            }
        }
    }
}

/// Cached in-place buffer slot: `(weak shadow, record_key, uav span if picked_on && written)`.
type InPlaceSlot = (std::sync::Weak<crate::shadow::Shadow>, u64, Option<(u64, u64)>);

/// Descriptor-derived references belong to the heap that produced them.
pub(crate) struct BufferSweep {
    passes: u64,
    ranges: DenseRanges,
    /// In-place shadow slots kept out of `ranges.dense` so the per-frame
    /// refresh-span merge skips them while `buffer_slots_of` still validates
    /// them in O(1).
    in_place: std::collections::HashMap<u32, InPlaceSlot, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
    // Last refresh frame and covered interval for throttled, unwatched copies.
    unwatched: std::collections::HashMap<u64, (u64, u64, u64)>,
    /// Merged unique `(shadow, from, to)` spans from the last full frame sweep,
    /// enabling O(unique_shadows) intra-frame dirty checks between command lists.
    shadow_spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
}

impl Default for BufferSweep {
    fn default() -> Self {
        Self {
            passes: 0,
            ranges: DenseRanges::default(),
            in_place: Default::default(),
            unwatched: Default::default(),
            shadow_spans: Vec::new(),
        }
    }
}

#[cfg(test)]
impl<T> HeapRanges<T> {
    fn set(&mut self, slot: u32, range: Option<(T, u64, u64)>) {
        let slot = slot as usize;
        if range.is_none() && slot >= self.0.len() {
            return;
        }
        if slot >= self.0.len() {
            self.0.resize_with(slot + 1, || None);
        }
        self.0[slot] = range;
    }

    /// Forgets the slots of one of `parts` parts
    /// (`HostHeap::buffer_slots_of`).
    fn forget_part(&mut self, part: u32, parts: u32) {
        self.0.iter_mut().skip(part as usize).step_by(parts as usize).for_each(|r| *r = None);
    }

    fn ranges(&self) -> impl Iterator<Item = &(T, u64, u64)> {
        self.0.iter().flatten()
    }
}

#[cfg(test)]
mod heap_range_tests {
    use super::{HEAP_ROUND, HeapRanges};

    /// Round by round every slot is looked at again once, whatever was
    /// written between, and what a slot holds is what was found last.
    #[test]
    fn every_slot_is_looked_at_again_in_a_round() {
        const SLOTS: u32 = 1000;
        let mut ranges = HeapRanges(Vec::new());
        ranges.set(7, None);
        assert_eq!(ranges.ranges().count(), 0);
        for slot in (0..SLOTS).step_by(3) {
            ranges.set(slot, Some(((), 0, 1)));
        }
        let parts = HEAP_ROUND as u32;
        for pass in 1..=HEAP_ROUND {
            let part = (pass % HEAP_ROUND) as u32;
            ranges.forget_part(part, parts);
            for slot in (0..SLOTS).step_by(3).filter(|slot| slot % parts == part) {
                assert!(ranges.0[slot as usize].is_none(), "slot {slot} not forgotten");
                ranges.set(slot, Some(((), pass, pass)));
            }
            assert_eq!(ranges.ranges().count(), SLOTS.div_ceil(3) as usize);
        }
        assert!(ranges.ranges().all(|r| r.1 != 0), "a slot not looked at in a round");
        ranges.set(3, None);
        assert_eq!(ranges.ranges().count(), SLOTS.div_ceil(3) as usize - 1);
    }

    /// The spans of 4,400 copies from the ranges of 72,000 slots, as a pass
    /// makes them, by slot and from the table they were in. Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn spans_of_a_pass() {
        use std::collections::HashMap;
        use std::sync::Arc;
        let copies: Vec<Arc<u64>> = (0..4400).map(Arc::new).collect();
        let (mut table, mut slots) = (HashMap::new(), HeapRanges(Vec::new()));
        for slot in 0..72_000u32 {
            let copy = &copies[slot as usize * 7919 % 4400];
            table.insert(slot, (copy.clone(), u64::from(slot), u64::from(slot) + 64));
            slots.set(slot, Some((copy.clone(), u64::from(slot), u64::from(slot) + 64)));
        }
        fn join<S: std::hash::BuildHasher + Default>(ranges: &mut dyn Iterator<Item = &(Arc<u64>, u64, u64)>) -> usize {
            let mut spans: HashMap<usize, (Arc<u64>, u64, u64), S> = HashMap::default();
            for (copy, from, to) in ranges {
                let span = spans.entry(Arc::as_ptr(copy) as usize).or_insert_with(|| (copy.clone(), *from, *to));
                span.1 = span.1.min(*from);
                span.2 = span.2.max(*to);
            }
            spans.len()
        }
        let best = |f: &mut dyn FnMut() -> usize| {
            (0..30)
                .map(|_| {
                    let t = std::time::Instant::now();
                    assert_eq!(f(), 4400);
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };
        println!(
            "{:.2} ms from the table, {:.2} ms by slot",
            best(&mut || join::<std::hash::RandomState>(&mut table.values())),
            best(&mut || join::<std::hash::BuildHasherDefault<crate::host::AddressHash>>(&mut slots.ranges())),
        );
    }
}

/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) enum Into<'a> {
    List(&'a HostList),
    Ahead(&'a mut crate::heap_worker::Batch),
}

/// Whether a newly created host texture may take its first upload on
/// `texture::ahead`'s dedicated COPY queue instead of the DIRECT list/batch.
///
/// With `VICTORY_HEAP_WORKER=1` (`ahead_worker`), COPY-queue offloading is
/// race-free only for placed textures (`is_placed`) when `VICTORY_SHOW_AFTER=1`
/// (`show_after`) is also active:
/// - non-placed `View::Shader` objects register in `descriptor::WAITING` and
///   are mirrored immediately by `descriptor::make_waiting()` during
///   `Host::submit` without waiting for the COPY queue fence;
/// - without `show_after`, a concurrent table draw calls `HostHeap::mirror`
///   in `mirror_after()` before `readable()` waits in `ahead::listed()`,
///   exposing the texture to in-flight DIRECT submissions while the COPY
///   queue is still writing it.
#[allow(clippy::too_many_arguments)]
fn use_ahead_copy_queue(
    ahead_worker: bool,
    late: bool,
    is_placed: bool,
    shows_other: bool,
    baking: bool,
    sync_bake: bool,
    show_after: bool,
    slot_made_from: bool,
    tabled: bool,
    fresh: bool,
) -> bool {
    if tabled || !fresh || baking || sync_bake || shows_other {
        return false;
    }
    if ahead_worker {
        is_placed && show_after && !slot_made_from
    } else {
        late
    }
}

pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Frames between looks at an uploaded texture picked by index once
/// [`RECHECK_FRAMES`] is through (`VICTORY_HEAP_RECHECK_EVERY=<frames>`; unset
/// or 0: it is not looked at again, as before). The title streams a
/// texture's memory in and rewrites it long after its view was written:
/// runs 338-339 logged changes found at the schedule's last look, and what
/// changes later stays as it was on the card for the rest of the run (skin
/// that turns black or red). After RyanCase94's 920423a9.
fn recheck_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let every = std::env::var("VICTORY_HEAP_RECHECK_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if every > 0 {
            tracing::info!("uploaded textures picked by index are looked at again every {every} frames (VICTORY_HEAP_RECHECK_EVERY)");
        }
        every
    })
}

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_SAME: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.0.load(Ordering::Relaxed)
}

/// What a render target or depth view of the host is made from: the host
/// texture, and [`crate::descriptor::RtvRecord::key`].
type TargetKey = (usize, (usize, [u32; 7]));

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = if srv {
        crate::shadow::find(address, 1)?
    } else {
        crate::shadow::find_any(address, 1 << 16)?
    };
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        if srv && shadow.in_place() && crate::port_cbv::on() && crate::port_cbv::root_cbv_ok(gpu) {
            crate::port_cbv::root_in_place();
        }
        shadow.refresh(list, address, len);
        #[cfg(feature = "diagnostics")]
        crate::book::root(gpu, len);
        return Some(gpu);
    }
    if crate::host::list_reuse() {
        return list.fill_scratch(address as *const u8, len);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

/// The pass over the buffer views of `heap` that shaders pick by index
/// ([`State::heap_buffers`]): their host slots are made, and the host copies
/// they show are to be refreshed.
fn heap_buffer_pass(heap: &crate::descriptor::Heap, host: &crate::descriptor::host::HostHeap, frame: u64) -> Vec<crate::host::aside::Refresh> {
    let started = std::time::Instant::now();
    // Per host heap: the host copy and range each slot's buffer view
    // reads, kept from pass to pass. Only the slots written since the
    // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
    // at best); all of them every 16 passes, because a view of a
    // released resource goes stale without a write, and a slot written
    // before its memory had a host copy is only made then.
    let mut kept = host.buffer_sweep.lock().unwrap_or_else(|e| e.into_inner());
    let BufferSweep { passes, ranges, in_place, unwatched, shadow_spans } = &mut *kept;
    let changed = host.buffer_changes();
    let all = *passes == 0 || (*passes % HEAP_ROUND == 0 && !heap_spread());
    let mut slots = if all {
        ranges.clear();
        in_place.clear();
        host.buffer_slots()
    } else if heap_spread() {
        // The written ones, and this pass's part of the others (validated in
        // place against cached keys/liveness instead of blindly clearing `part`).
        let (part, parts) = ((*passes % HEAP_ROUND) as u32, HEAP_ROUND as u32);
        let mut slots = host.buffer_slots_of(part, parts);
        slots.extend(changed);
        slots
    } else {
        changed
    };
    slots.sort_unstable();
    slots.dedup();
    *passes += 1;
    // Whether the device was there before the pass: a view that removes
    // it is named below.
    #[cfg(feature = "diagnostics")]
    static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    #[cfg(feature = "diagnostics")]
    let alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
    let mut unordered = Vec::new();
    /// The title's heap, for the threads that look.
    struct Shared<'a>(&'a crate::descriptor::Heap);
    // SAFETY: a heap is free-threaded (its methods are called by any thread of
    // the title's), and it lives until all of them have looked.
    unsafe impl Sync for Shared<'_> {}
    impl Shared<'_> {
        fn handle(&self, index: u32) -> Option<u64> {
            self.0.handle(index)
        }
    }
    let heap = Shared(heap);
    let picked_on = crate::picked::on();
    enum SlotUpdate {
        KeepUav,
        KeepInPlaceUav,
        Clear,
        Set(CachedSlot, u64, u64),
        SetInPlace(InPlaceSlot),
    }
    // Immutable snapshots of cached entries for lock-free validation across worker threads.
    let cached = &*ranges;
    let cached_in_place = &*in_place;
    let has_cached = |index: u32| cached.get(index).is_some() || cached_in_place.contains_key(&index);
    let finish_mirror = |index: u32,
                         view: Option<crate::descriptor::View>,
                         direct_copy: Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>,
                         made_now: bool,
                         key: u64|
     -> Option<SlotUpdate> {
        let Some(view) = view else {
            host.untrack_buffer(index);
            return has_cached(index).then_some(SlotUpdate::Clear);
        };
        let used_direct = direct_copy.is_some();
        // The copy the slot shows (`buffer_copy`), reusing the one `mirror_heap_pass_batch`
        // already resolved when freshly built.
        let Some((shadow, address, len)) =
            direct_copy.or_else(|| crate::descriptor::host::buffer_copy(&view))
        else {
            host.untrack_buffer(index);
            return has_cached(index).then_some(SlotUpdate::Clear);
        };
        #[cfg(feature = "diagnostics")]
        if !(made_now && used_direct) && host.shows(index, &shadow.buffer) == Some(false) {
            HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
        }
        let _ = (made_now, used_direct);
        let written = picked_on && view_kind(&view) == Some(ResourceType::Uav);
        if shadow.in_place() {
            let uav_span = written.then_some((address, address + len));
            return Some(SlotUpdate::SetInPlace((std::sync::Arc::downgrade(&shadow), key, uav_span)));
        }
        Some(SlotUpdate::Set((shadow, key, written), address, address + len))
    };
    #[cfg(feature = "diagnostics")]
    let (diag_tabled, diag_hits, diag_mirrors) = (
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    );
    // Some slots at a time, on several threads.
    const AT_A_TIME: usize = 256;
    let parts: Vec<&[u32]> = slots.chunks(AT_A_TIME).collect();
    let found: Vec<std::sync::OnceLock<Vec<_>>> = parts.iter().map(|_| Default::default()).collect();
    crate::land::each_view(false, parts.len(), &|n| {
        let mut looked: Vec<(u32, SlotUpdate)> = Vec::new();
        let mut quick_hits: Vec<(u32, crate::descriptor::View, u64)> = Vec::new();
        let mut misses: Vec<(u32, u64, Option<[u32; 3]>)> = Vec::new();
        #[cfg(feature = "diagnostics")]
        let (mut loc_tabled, mut loc_hits) = (0u64, 0u64);
        {
            // Acquire `VIEWS.read()` once per 256-slot chunk for cached liveness checks.
            let views = crate::descriptor::lock_views_read();
            for &index in parts[n] {
                // A draw's table's: the draw brings what it reads up to date.
                if host.copied_tabled(index, frame) {
                    #[cfg(feature = "diagnostics")]
                    {
                        loc_tabled += 1;
                    }
                    host.untrack_buffer(index);
                    if has_cached(index) {
                        looked.push((index, SlotUpdate::Clear));
                    }
                    continue;
                }
                let Some(handle) = heap.handle(index) else {
                    host.untrack_buffer(index);
                    if has_cached(index) {
                        looked.push((index, SlotUpdate::Clear));
                    }
                    continue;
                };
                let record = crate::descriptor::record_at(handle);
                if let Some([_, i, generation]) = record {
                    let key = u64::from(generation) << 32 | u64::from(i);
                    if host.quick_matches(index, key) {
                        if let Some((_, (shadow, prev_key, prev_uav), _, _)) = cached.get(index)
                            && *prev_key == key
                            && !shadow.is_dead()
                            && views.view_alive(i, generation)
                        {
                            #[cfg(feature = "diagnostics")]
                            {
                                loc_hits += 1;
                            }
                            if picked_on && *prev_uav {
                                looked.push((index, SlotUpdate::KeepUav));
                            }
                            continue;
                        }
                        if let Some((weak, prev_key, prev_uav)) = cached_in_place.get(&index)
                            && *prev_key == key
                            && weak.upgrade().is_some_and(|s| !s.is_dead())
                            && views.view_alive(i, generation)
                        {
                            #[cfg(feature = "diagnostics")]
                            {
                                loc_hits += 1;
                            }
                            if picked_on && prev_uav.is_some() {
                                looked.push((index, SlotUpdate::KeepInPlaceUav));
                            }
                            continue;
                        }
                        // Slot already mirrors `key` on the host heap (e.g. built by a draw):
                        // resolve `shadow` once without entering `mirror_heap_pass_batch`.
                        if let Some(view) = views.view_at(i, generation) {
                            #[cfg(feature = "diagnostics")]
                            {
                                loc_hits += 1;
                            }
                            quick_hits.push((index, view, key));
                            continue;
                        }
                    }
                }
                misses.push((index, handle, record));
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            diag_tabled.fetch_add(loc_tabled, Ordering::Relaxed);
            diag_hits.fetch_add(loc_hits, Ordering::Relaxed);
            diag_mirrors.fetch_add(misses.len() as u64, Ordering::Relaxed);
        }
        for (index, view, key) in quick_hits {
            if let Some(update) = finish_mirror(index, Some(view), None, false, key) {
                looked.push((index, update));
            }
        }
        let mirrored = host.mirror_heap_pass_batch(&misses);
        for ((_, _, record), (index, view, direct_copy, made_now)) in misses.into_iter().zip(mirrored) {
            let key = record.map_or(0, |[_, i, generation]| u64::from(generation) << 32 | u64::from(i));
            if let Some(update) = finish_mirror(index, view, direct_copy, made_now, key) {
                looked.push((index, update));
            }
        }
        // (After each piece while the device lives: a cheap query.)
        #[cfg(feature = "diagnostics")]
        if alive
            && let Some(why) = crate::descriptor::host::device_removed()
            && !GONE.swap(true, Ordering::Relaxed)
        {
            let last: Vec<String> = parts[n]
                .iter()
                .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                .map(|s| s.chars().take(200).collect())
                .collect();
            tracing::error!(why, "the host device went while buffer views of the heap were made; perhaps one of: {last:#?}");
            crate::host::log_removal_context();
        }
        let _ = found[n].set(looked);
    });
    #[cfg(feature = "diagnostics")]
    {
        static DIAG_PASS: AtomicU64 = AtomicU64::new(0);
        let p = DIAG_PASS.fetch_add(1, Ordering::Relaxed);
        if p % 120 == 119 {
            tracing::info!(
                slots = slots.len(),
                tabled = diag_tabled.load(Ordering::Relaxed),
                hits = diag_hits.load(Ordering::Relaxed),
                mirrors = diag_mirrors.load(Ordering::Relaxed),
                dense = ranges.dense.len(),
                look_ms = format_args!("{:.2}", started.elapsed().as_secs_f64() * 1000.0),
                "heap_buffer_pass breakdown"
            );
        }
    }
    for (index, update) in found.into_iter().filter_map(std::sync::OnceLock::into_inner).flatten() {
        match update {
            SlotUpdate::KeepUav => {
                if let Some((_, (shadow, _, true), from, to)) = ranges.get(index) {
                    unordered.push((shadow.clone(), *from, *to));
                }
            }
            SlotUpdate::KeepInPlaceUav => {
                if let Some((weak, _, Some((from, to)))) = in_place.get(&index)
                    && let Some(shadow) = weak.upgrade()
                {
                    unordered.push((shadow, *from, *to));
                }
            }
            SlotUpdate::Clear => {
                ranges.set(index, None);
                in_place.remove(&index);
            }
            SlotUpdate::Set(slot_data, from, to) => {
                in_place.remove(&index);
                if picked_on && slot_data.2 {
                    unordered.push((slot_data.0.clone(), from, to));
                }
                ranges.set(index, Some((slot_data, from, to)));
            }
            SlotUpdate::SetInPlace(entry) => {
                ranges.set(index, None);
                if picked_on
                    && let Some((from, to)) = entry.2
                    && let Some(shadow) = entry.0.upgrade()
                {
                    unordered.push((shadow, from, to));
                }
                in_place.insert(index, entry);
            }
        }
    }
    if picked_on {
        crate::picked::shown(unordered, all);
    }
    #[cfg(feature = "diagnostics")]
    let looked = std::time::Instant::now();
    // Per host copy, the span its picked views cover: scan only the dense
    // active slots (~200k vs ~1M sparse) across 4 coarse worker chunks so the
    // final merge combines only 4 small maps instead of ~60.
    let views = ranges.dense.len() as u32;
    let mut spans: crate::host::ByCopy<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> = Default::default();
    if !ranges.dense.is_empty() {
        let chunk_len = ranges.dense.len().div_ceil(4).max(4096);
        let span_chunks: Vec<&[(u32, CachedSlot, u64, u64)]> = ranges.dense.chunks(chunk_len).collect();
        let partial: Vec<std::sync::OnceLock<crate::host::ByCopy<(usize, u64, u64)>>> =
            span_chunks.iter().map(|_| Default::default()).collect();
        crate::land::each_view(false, span_chunks.len(), &|c| {
            let mut local: crate::host::ByCopy<(usize, u64, u64)> = Default::default();
            let base_idx = c * chunk_len;
            for (offset, (_, (shadow, _, _), from, to)) in span_chunks[c].iter().enumerate() {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                let span = local.entry(key).or_insert((base_idx + offset, *from, *to));
                span.1 = span.1.min(*from);
                span.2 = span.2.max(*to);
            }
            let _ = partial[c].set(local);
        });
        for local in partial.into_iter().filter_map(std::sync::OnceLock::into_inner) {
            for (key, (dense_idx, from, to)) in local {
                let span = spans.entry(key).or_insert_with(|| (ranges.dense[dense_idx].1.0.clone(), from, to));
                span.1 = span.1.min(from);
                span.2 = span.2.max(to);
            }
        }
    }
    let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
    // Memory that is not write-watched is copied whole by every refresh.
    // It was once refreshed only every 120 frames (unmapped pages read
    // one by one cost 14 GB a frame in run 11, before they were
    // remembered): streamed geometry then drew from stale buffers, and
    // the garbage ended runs (the card stopped answering 14-90 frames
    // into the opening cutscene; with every frame it ran on). The pass
    // costs some 30 ms a frame this way.
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let every = *EVERY.get_or_init(|| {
        std::env::var("VICTORY_UNWATCHED_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
    });
    let mut found = Vec::with_capacity(spans.len());
    let mut next_unwatched = std::collections::HashMap::new();
    for (shadow, from, to) in spans.values() {
        if every > 1 && shadow.unwatched() {
            let id = shadow.id();
            if let Some(&(last, low, high)) = unwatched.get(&id)
                && frame >= last && frame - last < every && low <= *from && *to <= high
            {
                next_unwatched.insert(id, (last, low, high));
                continue;
            }
            next_unwatched.insert(id, (frame, *from, *to));
        }
        found.push((shadow.clone(), *from, (*to).min(*from + shadow.left(*from))));
    }
    *unwatched = next_unwatched;
    shadow_spans.clear();
    shadow_spans.extend(spans.values().map(|(s, from, to)| (s.clone(), *from, *to)));
    drop(kept);
    #[cfg(feature = "diagnostics")]
    let refreshed = std::time::Instant::now();
    crate::watch::look_at_holes();
    #[cfg(feature = "diagnostics")]
    for (sum, time) in HEAP_SPLIT.iter().zip([looked - started, refreshed - looked]) {
        sum.fetch_add(time.as_nanos() as u64, Ordering::Relaxed);
    }
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    // The first pass and every growth of the set by half.
    let seen = LOGGED.load(Ordering::Relaxed);
    if u64::from(views) > seen + seen / 2 {
        LOGGED.store(u64::from(views), Ordering::Relaxed);
        tracing::info!(
            slots = slots.len(),
            views,
            copies = spans.len(),
            mb = bytes >> 20,
            ms = started.elapsed().as_millis(),
            "buffer views of the heap made ready for shaders that pick them by index"
        );
    }
    found
}

/// Intra-frame incremental bindless buffer refresh (`ARCH-08`): mirrors only
/// newly changed buffer descriptor slots since the frame's first sweep so
/// mid-frame bindless buffer slot updates are visible without re-scanning all
/// watched shadows on every list.
fn heap_buffer_incremental(
    heap: &crate::descriptor::Heap,
    host: &crate::descriptor::host::HostHeap,
    frame: u64,
) -> Vec<crate::host::aside::Refresh> {
    let changed = host.buffer_changes();
    if changed.is_empty() {
        return Vec::new();
    }
    let mut kept = host.buffer_sweep.lock().unwrap_or_else(|e| e.into_inner());
    let BufferSweep { ranges, in_place, shadow_spans, .. } = &mut *kept;
    let picked_on = crate::picked::on();
    let mut out = Vec::new();
    let mut misses = Vec::new();
    for index in changed {
        if host.copied_tabled(index, frame) {
            host.untrack_buffer(index);
            ranges.set(index, None);
            in_place.remove(&index);
            continue;
        }
        let Some(handle) = heap.handle(index) else {
            host.untrack_buffer(index);
            ranges.set(index, None);
            in_place.remove(&index);
            continue;
        };
        misses.push((index, handle, crate::descriptor::record_at(handle)));
    }
    if !misses.is_empty() {
        let mirrored = host.mirror_heap_pass_batch(&misses);
        let mut unordered = Vec::new();
        for ((_, _, record), (index, view, direct_copy, _)) in misses.into_iter().zip(mirrored) {
            let key = record.map_or(0, |[_, i, generation]| u64::from(generation) << 32 | u64::from(i));
            let Some(view) = view else {
                host.untrack_buffer(index);
                ranges.set(index, None);
                in_place.remove(&index);
                continue;
            };
            let Some((shadow, address, len)) =
                direct_copy.or_else(|| crate::descriptor::host::buffer_copy(&view))
            else {
                host.untrack_buffer(index);
                ranges.set(index, None);
                in_place.remove(&index);
                continue;
            };
            let to = address.saturating_add(len);
            let written = picked_on && view_kind(&view) == Some(ResourceType::Uav);
            if shadow.in_place() {
                ranges.set(index, None);
                let uav_span = written.then_some((address, to));
                if written {
                    unordered.push((shadow.clone(), address, to));
                }
                in_place.insert(index, (std::sync::Arc::downgrade(&shadow), key, uav_span));
            } else {
                in_place.remove(&index);
                if written {
                    unordered.push((shadow.clone(), address, to));
                }
                ranges.set(index, Some(((shadow.clone(), key, written), address, to)));
                let ptr = std::sync::Arc::as_ptr(&shadow) as usize;
                if let Some(existing) = shadow_spans.iter_mut().find(|(s, _, _)| std::sync::Arc::as_ptr(s) as usize == ptr) {
                    existing.1 = existing.1.min(address);
                    existing.2 = existing.2.max(to);
                } else {
                    shadow_spans.push((shadow.clone(), address, to));
                }
                let clamped_to = to.min(address.saturating_add(shadow.left(address)));
                if clamped_to > address {
                    out.push((shadow, address, clamped_to));
                }
            }
        }
        if picked_on && !unordered.is_empty() {
            crate::picked::shown(unordered, false);
        }
    }
    out
}

#[inline]
fn bind_cache() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("VICTORY_BIND_CACHE").map_or(true, |v| v != "0"))
}

#[inline]
fn viewports_eq(a: &[D3D12_VIEWPORT], b: &[D3D12_VIEWPORT]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.TopLeftX.to_bits() == y.TopLeftX.to_bits()
                && x.TopLeftY.to_bits() == y.TopLeftY.to_bits()
                && x.Width.to_bits() == y.Width.to_bits()
                && x.Height.to_bits() == y.Height.to_bits()
                && x.MinDepth.to_bits() == y.MinDepth.to_bits()
                && x.MaxDepth.to_bits() == y.MaxDepth.to_bits()
        })
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        self.trim_dynamic_slab();
        if !self.dynamic.is_empty() {
            crate::descriptor::release_dynamic_batch(&self.dynamic);
            self.dynamic.clear();
        }
        self.pipeline = pipeline as usize;
        self.heaps = [0; 2];
        self.graphics.clear();
        self.compute.clear();
        self.vertex = [VertexView::default(); 32];
        self.index = IndexView::default();
        self.targets.clear();
        self.depth = 0;
        self.depth_state = 0;
        self.topology = 0;
        self.topology_set = false;
        self.target_handles.clear();
        self.depth_handle = 0;
        self.targets_set = false;
        self.host_targets = None;
        self.host_topology = None;
        self.host_viewports_set = false;
        self.host_scissors_set = false;
        self.host_blend = None;
        self.host_stencil = None;
        self.host_depth_bounds = None;
        self.viewports.clear();
        self.scissors.clear();
        self.blend = None;
        self.stencil = None;
        self.depth_bounds = None;
        self.clear_list_caches();
        self.recent_args.clear();
        self.table_records.clear();
        self.indirect_rebase.clear();
        if self.indirect_rebase.capacity() > 4096 {
            self.indirect_rebase.shrink_to(1024);
        }
        self.indirect_rebase_ranges.clear();
        if self.indirect_rebase_ranges.capacity() > 4096 {
            self.indirect_rebase_ranges.shrink_to(1024);
        }
        self.indirect_rebase_seen.clear();
        if self.indirect_rebase_seen.capacity() > 4096 {
            self.indirect_rebase_seen.shrink_to(1024);
        }
        self.dispatch_spread = [0; 2];
        self.dispatch_threads = [0; 3];
        #[cfg(feature = "diagnostics")]
        self.user.clear();
        self.ambient_tables.clear();
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_root_args[0].clear();
        self.host_root_args[1].clear();
        self.host_vb_bound = 0;
        self.host_ib_bound = false;
        self.host_heaps = [0; 2];
        self.host_heaps_set = false;
        self.fresh_tables.clear();
        if self.fresh_tables.capacity() > 4096 {
            self.fresh_tables.shrink_to(1024);
        }
        self.fresh_records.clear();
        if self.fresh_records.capacity() > 32_768 {
            self.fresh_records.shrink_to(8_192);
        }
        self.slab_step = 0;
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = false;
        self.listed_state.clear();
        if self.listed_state.capacity() > 4096 {
            self.listed_state.shrink_to(1024);
        }
        self.scratch_base = (0, 0);
        self.kept_shadows = [0; 8];
        self.kept_shadows_next = 0;
        self.last_drop_reason = None;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        if bind_cache() && self.host_topology == Some(host) {
            return;
        }
        self.host_topology = Some(host);
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// A host compute library bound its own pipeline, root signature and heaps.
    pub(crate) fn external_compute(&mut self) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_root_args[0].clear();
        self.host_root_args[1].clear();
        self.host_heaps = [usize::MAX; 2];
        self.host_heaps_set = false;
    }

    #[inline]
    fn invalidate_compute_host_state(&mut self) {
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        self.host_root_args[1].clear();
    }

    #[inline]
    fn set_root_gpu(
        &mut self,
        list: &HostList,
        compute: bool,
        param: u32,
        kind: RootGpuKind,
        val: u64,
    ) {
        let domain = usize::from(compute);
        if bind_cache() && self.host_root_args[domain].check_or_set_gpu(param, kind, val) {
            return;
        }
        let l = &list.list;
        // SAFETY: an open list; `val` is a live GPU VA or shader-visible descriptor handle.
        unsafe {
            match (compute, kind) {
                (false, RootGpuKind::Cbv) => l.SetGraphicsRootConstantBufferView(param, val),
                (false, RootGpuKind::Srv) => l.SetGraphicsRootShaderResourceView(param, val),
                (false, RootGpuKind::Uav) => l.SetGraphicsRootUnorderedAccessView(param, val),
                (false, RootGpuKind::Table) => {
                    l.SetGraphicsRootDescriptorTable(param, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: val })
                }
                (true, RootGpuKind::Cbv) => l.SetComputeRootConstantBufferView(param, val),
                (true, RootGpuKind::Srv) => l.SetComputeRootShaderResourceView(param, val),
                (true, RootGpuKind::Uav) => l.SetComputeRootUnorderedAccessView(param, val),
                (true, RootGpuKind::Table) => {
                    l.SetComputeRootDescriptorTable(param, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: val })
                }
                (_, RootGpuKind::None) => {}
            }
        }
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
        self.cached_heap_buffers = (0, 0, 0);
        self.cached_heap_strides = (0, 0, 0);
        self.table_gen[0] = self.table_gen[0].wrapping_add(1);
        self.table_gen[1] = self.table_gen[1].wrapping_add(1);
        self.last_stage_tables = [None, None];
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        let domain = usize::from(compute);
        let unchanged = matches!(
            self.roots_ref(compute).get(slot as usize),
            Some(Root::Table(prev)) if *prev == handle
        );
        if !unchanged {
            set(self.roots(compute), slot, Root::Table(handle));
            self.table_gen[domain] = self.table_gen[domain].wrapping_add(1);
        }
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let domain = usize::from(compute);
        self.const_gen[domain] = self.const_gen[domain].wrapping_add(1);
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Few::default()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            let slot_idx = start as usize + i;
            if let Some(slot) = self.vertex.get_mut(slot_idx)
                && *slot != *v
            {
                *slot = *v;
                self.host_vb_bound &= !(1u32 << slot_idx);
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        let empty = VertexView::default();
        for (i, slot) in self.vertex.iter_mut().enumerate().skip(start as usize).take(count as usize) {
            if *slot != empty {
                *slot = empty;
                self.host_vb_bound &= !(1u32 << i);
            }
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        let next = view.copied().unwrap_or_default();
        if self.index != next {
            self.index = next;
            self.host_ib_bound = false;
        }
    }

    /// Render targets and depth, set on the host list at once, unless it has
    /// the same ones already: the title sets the same targets again and
    /// again, each time a host view per target and a call of the driver
    /// (25,662 times in a frame of another fork). `VICTORY_BIND_CACHE=0`
    /// sets them every time.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut views: [(Option<&ID3D12Resource>, Option<D3D12_RENDER_TARGET_VIEW_DESC>); 8] =
            [(None, None); 8];
        let mut keys: [TargetKey; 8] = [(0, (0, [0; 7])); 8];
        let mut count = 0usize;
        for &h in handles.iter().take(8) {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    views[count] = (None, Some(desc));
                    keys[count] = (0, record.key());
                    count += 1;
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            views[count] = (Some(texture), record.desc());
            keys[count] = (texture.as_raw() as usize, record.key());
            count += 1;
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            #[cfg(feature = "diagnostics")]
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some((texture, record))
        });
        let dsv_key = dsv.as_ref().map(|(texture, record)| (texture.as_raw() as usize, record.key()));
        if bind_cache()
            && let Some((prev_keys, prev_count, prev_dsv)) = &self.host_targets
            && *prev_count == count
            && *prev_dsv == dsv_key
            && prev_keys[..count] == keys[..count]
        {
            return;
        }
        self.host_targets = Some((keys, count, dsv_key));
        self.ensured_targets.0 = 0;
        let mut rtvs = [D3D12_CPU_DESCRIPTOR_HANDLE::default(); 8];
        for i in 0..count {
            let (texture, desc) = &views[i];
            rtvs[i] = host.rtv(*texture, desc.as_ref());
        }
        let dsv = dsv.map(|(texture, record)| host.dsv(texture, record.depth_desc().as_ref()));
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                count as u32,
                (count > 0).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        #[cfg(feature = "diagnostics")]
        if let Some(v) = viewports.first() {
            crate::host::spans::viewport(v.Width as u32, v.Height as u32);
        }
        if bind_cache() && self.host_viewports_set && viewports_eq(&self.viewports, viewports) {
            return;
        }
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        self.host_viewports_set = true;
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        if bind_cache() && self.host_scissors_set && self.scissors.as_slice() == rects {
            return;
        }
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        self.host_scissors_set = true;
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        let bits = factor.map(|f| f.map(f32::to_bits));
        self.blend = factor.copied();
        if bind_cache() && self.host_blend == Some(bits) {
            return;
        }
        self.host_blend = Some(bits);
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        if bind_cache() && self.host_stencil == Some(reference) {
            return;
        }
        self.host_stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        let bits = [min.to_bits(), max.to_bits()];
        self.depth_bounds = Some([min, max]);
        if bind_cache() && self.host_depth_bounds == Some(bits) {
            return;
        }
        self.host_depth_bounds = Some(bits);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.clear_list_caches();
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_root_args[0].clear();
        self.host_root_args[1].clear();
        self.host_vb_bound = 0;
        self.host_ib_bound = false;
        self.kept_shadows = [0; 8];
        self.kept_shadows_next = 0;
        self.scratch_base = (0, 0);
        self.host_heaps = [0; 2];
        self.host_heaps_set = false;
        self.host_topology = None;
        self.host_viewports_set = false;
        self.host_scissors_set = false;
        self.host_blend = None;
        self.host_stencil = None;
        self.host_depth_bounds = None;
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        self.host_targets = None;
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            self.host_viewports_set = true;
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            self.host_blend = Some(Some(factor.map(f32::to_bits)));
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            self.host_stencil = Some(reference);
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            self.host_depth_bounds = Some([min.to_bits(), max.to_bits()]);
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{
            USER_DISPATCH_SPREAD, USER_DISPATCH_THREADS, USER_MEMORY_BASE, USER_ROOT_ADDRESS,
            USER_VIEW_ADDRESS, USER_VIEW_BIAS,
        };
        if kind == USER_DISPATCH_SPREAD {
            return self.dispatch_spread[usize::from(dword != 0)];
        }
        if kind == USER_DISPATCH_THREADS {
            return self.dispatch_threads.get(dword as usize).copied().unwrap_or(0);
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| if dword == 0 { m.base } else { m.count });
        }
        // All of every group, unless the dispatch says otherwise.
        if kind == victory_shader::emit::USER_DISPATCH_THREADS_X {
            return 0;
        }
        if kind == USER_ROOT_ADDRESS || kind == USER_ROOT_ADDRESS + 1 {
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) if kind == USER_ROOT_ADDRESS => address as u32,
                Some(&Root::View(address)) => (address >> 32) as u32,
                _ => 0,
            };
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`). Loose windows are >= 64 KiB aligned,
            // so outside title buffers the 16-byte bias equals `address & 15`.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::peek(address, 1)
                    .map_or((address & 15) as u32, |s| (s.offset(address) & 15) as u32),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        let address = self.view_address(compute, slot, dword);
        if address == 0 {
            // The first few say what lies there instead.
            static SAID: AtomicU64 = AtomicU64::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 8 {
                let view = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4),
                    _ => None,
                };
                tracing::warn!(slot, dword, ?view, "a shader takes the address of a buffer view that is not there: it reads 0");
            }
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// The title address the view at dword `dword` of the table in root
    /// slot `slot` starts at, 0 without one.
    fn view_address(&self, compute: bool, slot: u8, dword: u32) -> u64 {
        match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::view_address(&view))
                .unwrap_or(0),
            _ => 0,
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages made ready of the root views the list has set and of
    /// the views whose addresses it takes (`t`: the ray-tracing structures
    /// it walks).
    fn memory_ready(
        &self,
        list: &HostList,
        t: &crate::pipeline::Translated,
        compute: bool,
        use_: crate::memory::Use,
    ) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        // (What it may write: its root views of unordered access.)
        let written: Vec<u64> = t
            .used
            .iter()
            .filter(|u| u.kind == ResourceType::Uav && u.root == RootType::Descriptor)
            .filter_map(|u| match self.roots_ref(compute).get(usize::from(u.slot)) {
                Some(&Root::View(address)) => Some(address),
                _ => None,
            })
            .collect();
        for root in self.roots_ref(compute) {
            match root {
                // (Of a program that only walks structures, the structures'
                // views below: its root views are constants and tables.)
                Root::View(address) if use_ != crate::memory::Use::Reads => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        for u in &t.host.user_data {
            if u.root == RootType::Other(victory_shader::emit::USER_VIEW_ADDRESS) {
                addresses.push(self.view_address(compute, u.slot, u.dword));
            }
        }
        crate::memory::ready(list, heap, &addresses, &written, &constants, use_, t.threads)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; `None` if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> Option<&'static HostPipeline> {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> Option<&'static HostPipeline> {
        self.last_drop_reason = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            self.last_drop_reason = Some(DropReason::LeftOut);
            return None;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            self.last_drop_reason = Some(DropReason::NoPso);
            return None;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            self.last_drop_reason = Some(DropReason::NoPso);
            return None;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return None;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads (bind only dirty/unbound runs).
        let need_vb = if bind_cache() { p.slots & !self.host_vb_bound } else { p.slots };
        if need_vb != 0 {
            let mut rem = need_vb;
            while rem != 0 {
                let start = rem.trailing_zeros() as usize;
                let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
                let mut run = 0usize;
                while start + run < 32 && (rem & (1u32 << (start + run))) != 0 {
                    let slot = start + run;
                    rem &= !(1u32 << slot);
                    let v = self.vertex[slot];
                    if v.address == 0 || v.size == 0 {
                        self.host_vb_bound |= 1u32 << slot;
                    } else if let Some(view) = crate::shadow::with_in_place(v.address, u64::from(v.size), |s| {
                        keep_shadow_dedup(&mut self.kept_shadows, &mut self.kept_shadows_next, list, s);
                        D3D12_VERTEX_BUFFER_VIEW {
                            BufferLocation: s.gpu(v.address),
                            SizeInBytes: v.size.min(u32::try_from(s.left(v.address)).unwrap_or(u32::MAX)),
                            StrideInBytes: v.stride,
                        }
                    }) {
                        views[run] = view;
                        self.host_vb_bound |= 1u32 << slot;
                    } else if let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) {
                        shadow.refresh(list, v.address, u64::from(v.size));
                        views[run] = D3D12_VERTEX_BUFFER_VIEW {
                            BufferLocation: shadow.gpu(v.address),
                            SizeInBytes: v.size.min(u32::try_from(shadow.left(v.address)).unwrap_or(u32::MAX)),
                            StrideInBytes: v.stride,
                        };
                        self.host_vb_bound |= 1u32 << slot;
                    }
                    run += 1;
                }
                // SAFETY: an open list; the views name live host buffers (or null).
                unsafe { l.IASetVertexBuffers(start as u32, Some(&views[..run])) };
            }
        }
        if indexed && (!bind_cache() || !self.host_ib_bound) {
            let v = self.index;
            let view = if let Some(view) = crate::shadow::with_in_place(v.address, u64::from(v.size), |s| {
                keep_shadow_dedup(&mut self.kept_shadows, &mut self.kept_shadows_next, list, s);
                D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: s.gpu(v.address),
                    SizeInBytes: v.size.min(u32::try_from(s.left(v.address)).unwrap_or(u32::MAX)),
                    Format: DXGI_FORMAT(v.format as i32),
                }
            }) {
                view
            } else {
                let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                    self.last_drop_reason = Some(DropReason::NoBuffer);
                    return None;
                };
                shadow.refresh(list, v.address, u64::from(v.size));
                D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: shadow.gpu(v.address),
                    SizeInBytes: v.size.min(u32::try_from(shadow.left(v.address)).unwrap_or(u32::MAX)),
                    Format: DXGI_FORMAT(v.format as i32),
                }
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
            self.host_ib_bound = true;
        }
        const RENDER_TARGET: u32 = 0x4;
        let depth_wanted = if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            // A read-only DSV cannot modify either plane, so it must not dirty
            // the colour twin and trigger a spurious depth->twin copy.
            let writes_depth = self.depth_state != DEPTH_READ;
            list.twin_use(self.depth as This, false, writes_depth);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            }
        } else {
            0
        };
        // Fast path (`CPU-21`): if nothing on `list` changed texture states
        // since the previous draw ensured these exact targets, and none of
        // them has a format-aliased view (`has_alias`), skip redundant
        // `ensure_state` calls so `list.epoch()` stays constant across draws
        // in the same render pass (keeping `FreshTable::ready` and
        // `listed_state` hot!).
        let list_ptr = std::ptr::from_ref(list) as usize;
        let epoch = list.epoch();
        let target_count = self.targets.len();
        let targets_cached = target_count <= 8
            && self.ensured_targets.0 == list_ptr
            && self.ensured_targets.1 == epoch
            && self.ensured_targets.2 == target_count
            && self.ensured_targets.3[..target_count] == self.targets[..]
            && self.ensured_targets.4 == self.depth
            && self.ensured_targets.5 == depth_wanted
            && !self.targets.iter().any(|&t| crate::resource::has_alias(t as This))
            && (self.depth == 0 || !crate::resource::has_alias(self.depth as This));
        if !targets_cached {
            for &t in &self.targets {
                crate::list::ensure_state(list, t as This, RENDER_TARGET);
            }
            if self.depth != 0 {
                crate::list::ensure_state(list, self.depth as This, depth_wanted);
            }
            if target_count <= 8 {
                let mut arr = [0usize; 8];
                arr[..target_count].copy_from_slice(&self.targets);
                self.ensured_targets = (list_ptr, list.epoch(), target_count, arr, self.depth, depth_wanted);
            } else {
                self.ensured_targets.0 = 0;
            }
        }
        Some(p)
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // A list without a view heap: one of the title's for the views the
        // host itself puts into heap slots.
        if self.heaps[0] == 0
            && p.stages.iter().any(|s| !s.root_uavs.is_empty() || !s.root_srvs.is_empty())
            && let Some(heap) = crate::descriptor::lent_heap()
        {
            once!("a dispatch without a view heap that needs one gets the title's");
            self.heaps[0] = heap;
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them, with both a view heap and a sampler heap bound.
        if !self.host_heaps_set || self.host_heaps != self.heaps {
            let mut view_heap: Option<ID3D12DescriptorHeap> = None;
            let mut sampler_heap: Option<ID3D12DescriptorHeap> = None;
            for &h in &self.heaps {
                if h == 0 {
                    continue;
                }
                let title_heap = crate::descriptor::heap(h as This);
                if let Some(host_heap) = title_heap.host() {
                    if title_heap.samplers() {
                        sampler_heap = Some(host_heap.heap.clone());
                    } else {
                        view_heap = Some(host_heap.heap.clone());
                    }
                }
            }
            if view_heap.is_none() {
                view_heap = crate::host::host().and_then(|h| h.fallback_view_heap().cloned());
            }
            if sampler_heap.is_none() {
                sampler_heap = crate::host::host().and_then(|h| h.fallback_sampler_heap().cloned());
            }
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = Vec::with_capacity(2);
            if let Some(h) = view_heap {
                list.keep(&h);
                heaps.push(Some(h));
            }
            if let Some(h) = sampler_heap {
                list.keep(&h);
                heaps.push(Some(h));
            }
            if !heaps.is_empty() {
                // SAFETY: an open list and shader-visible heaps.
                unsafe { l.SetDescriptorHeaps(&heaps) };
            }
            self.host_heaps = self.heaps;
            self.host_heaps_set = true;
            self.host_roots = [0; 2];
            self.host_root_args[0].clear();
            self.host_root_args[1].clear();
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
            self.host_root_args[usize::from(compute)].clear();
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            // The title may release the pipeline before it submits the list.
            if crate::host::list_reuse() {
                list.bind_pipeline(&p.name, &p.pso, &p.root);
            } else {
                list.used_pipeline(&p.name);
                list.keep(&p.pso);
                list.keep(&p.root);
            }
        }
        // With `table_stages`: what the stages of this draw read of each
        // table, and the tables its earlier stages made afresh (address,
        // slots).
        let spans: &[(u8, u32, u32)] = if table_stages() { &p.stage_spans } else { &[] };
        let mut staged: Few<(u64, u32), 8> = Few::default();
        let domain = usize::from(compute);
        let list_ptr = std::ptr::from_ref(list) as usize;
        let pso_raw = p.pso.as_raw() as usize;
        let epoch_before = list.epoch();
        let view_epochs = crate::descriptor::host::view_cache_epochs();
        let quarantined = crate::pipeline::builder_quarantined(self.pipeline as This);
        let mut writes_mask = 0u8;
        for (stage_idx, stage) in p.stages.iter().take(4).enumerate() {
            let t = stage.translated();
            if (t.stores || !write_skip()) && !quarantined {
                writes_mask |= 1u8 << stage_idx;
            }
        }
        let tables_hit = self.last_stage_tables[domain].as_ref().is_some_and(|c| {
            c.matches(
                list_ptr,
                epoch_before,
                pso_raw,
                self.heaps,
                self.table_gen[domain],
                self.const_gen[domain],
                tested,
                view_epochs,
                writes_mask,
            )
        });
        if tables_hit && let Some(c) = &self.last_stage_tables[domain] {
            if c.tested_depth_read {
                TESTED_DEPTH.set((tested, true));
            }
            c.replay_uavs(list);
        }
        let mut can_cache_tables = !tables_hit && p.stages.len() <= 4;
        let mut cache_has_indexed = false;
        let mut cached_made: [Few<(u8, u32), 16>; 4] = Default::default();
        let mut uav_capture = UavCapture::default();
        if can_cache_tables {
            let _ = crate::residency::take_staging_deferred();
        }
        #[cfg(feature = "diagnostics")]
        static BIND_DIAG: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
        #[cfg(feature = "diagnostics")]
        {
            BIND_DIAG[usize::from(!tables_hit)].fetch_add(1, Ordering::Relaxed);
        }
        for (stage_idx, stage) in p.stages.iter().enumerate() {
            let t = stage.translated();
            let writes = (t.stores || !write_skip()) && !quarantined;
            let made: Few<(u8, u32), 16> = if tables_hit {
                self.last_stage_tables[domain]
                    .as_ref()
                    .map(|c| c.stages_made[stage_idx].clone())
                    .unwrap_or_default()
            } else {
                #[cfg(feature = "diagnostics")]
                let t_tbl = std::time::Instant::now();
                STORES.set(t.stores);
                if t.stores && crate::picked::on() && crate::picked::picks(&stage.shader, &t.picks) {
                    crate::picked::runs_in(list);
                    can_cache_tables = false;
                }
                if stage.shader.ambient.is_some() {
                    can_cache_tables = false;
                }
                if t.tables.groups.iter().any(|g| !g.indexed.is_empty()) {
                    cache_has_indexed = true;
                }
                let made = self.tables(
                    list,
                    &t.tables,
                    &t.host.half_views,
                    compute,
                    spans,
                    &mut staged,
                    can_cache_tables.then_some(&mut uav_capture),
                );
                STORES.set(true);
                #[cfg(feature = "diagnostics")]
                BIND_DIAG[2].fetch_add(t_tbl.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if made.iter().any(|m| m.1 == u32::MAX) {
                    once!("a draw whose descriptor table found no free slots is dropped");
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why(format!("no free dynamic slots for a table of {}", p.name));
                    self.last_drop_reason = Some(DropReason::NoSlots);
                    return false;
                }
                if stage.shader.ambient.is_some() {
                    self.ambient_tables.clear();
                    self.ambient_tables.extend(made.iter().copied());
                }
                if can_cache_tables {
                    cached_made[stage_idx] = made.clone();
                }
                made
            };
            #[cfg(feature = "diagnostics")]
            let t_rest = std::time::Instant::now();
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                // Only through a store by address: what it stores through
                // views is declared with them.
                let use_ = if crate::pipeline::is_builder(self.pipeline as This) || writes && t.stores_by_address {
                    crate::memory::Use::Builds { back: writes }
                } else {
                    crate::memory::Use::Reads
                };
                let Some(ready) = self.memory_ready(list, &t, compute, use_) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    self.last_drop_reason = Some(DropReason::NoSlots);
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                // (A root signature holds 64 dwords.)
                let mut room = [0u32; 64];
                let mut more = Vec::new();
                let values: &mut [u32] = match room.get_mut(..dwords as usize) {
                    Some(values) => values,
                    None => {
                        more.resize(dwords as usize, 0);
                        &mut more
                    }
                };
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                #[cfg(feature = "diagnostics")]
                {
                    self.user.clear();
                    self.user.extend_from_slice(values);
                }
                if !bind_cache() || !self.host_root_args[usize::from(compute)].check_or_set_consts(param, values) {
                    // SAFETY: an open list; `values` covers the parameter.
                    unsafe {
                        if compute {
                            l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                        } else {
                            l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                        }
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for (srv_idx, &(slot, param)) in views.iter().enumerate() {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            self.set_root_gpu(list, compute, param, RootGpuKind::Srv, m.directory);
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            Some(gpu) => self.set_root_gpu(list, compute, param, RootGpuKind::Srv, gpu),
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_TABLE_STRIDES {
                        // Ours: the shader inspects typed view kinds for its
                        // descriptor tables only (already mirrored by `tables`),
                        // so bind the strides buffer without sweeping the heap.
                        match self.heap_strides(list) {
                            Some(gpu) => self.set_root_gpu(list, compute, param, RootGpuKind::Srv, gpu),
                            None => once!("a shader inspects table view kinds without a view heap set"),
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_DXR_NORMALS {
                        // Ours: GPU octahedral geometric-normal lookup table for lifted
                        // `RayQuery` shaders using `CommittedTriangleVertexPositions`.
                        let gpu = crate::dxr::normals_gpu_va();
                        if gpu != 0 {
                            self.set_root_gpu(list, compute, param, RootGpuKind::Srv, gpu);
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv && stage.root_srv_is_biased(srv_idx);
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    let kind = if srv { RootGpuKind::Srv } else { RootGpuKind::Cbv };
                    self.set_root_gpu(list, compute, param, kind, gpu);
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        self.set_root_gpu(list, compute, param, RootGpuKind::Uav, m.faults);
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
            #[cfg(feature = "diagnostics")]
            BIND_DIAG[3].fetch_add(t_rest.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        if tables_hit {
            if let Some(c) = self.last_stage_tables[domain].as_mut()
                && !c.srv_tex_hazard
            {
                c.list_epoch = list.epoch();
            }
        } else if can_cache_tables && !crate::residency::take_staging_deferred() {
            if !uav_capture.textures.is_empty()
                && (uav_capture.skipped_srv_check
                    || uav_capture
                        .textures
                        .iter()
                        .any(|t| uav_capture.srv_textures.contains(t)))
            {
                uav_capture.srv_tex_hazard = true;
            }
            self.last_stage_tables[domain] = Some(CachedStageTables {
                list_ptr,
                list_epoch: list.epoch(),
                pipeline: pso_raw,
                heaps: self.heaps,
                table_gen: self.table_gen[domain],
                const_gen: self.const_gen[domain],
                view_epochs,
                writes_mask,
                has_indexed: cache_has_indexed,
                tested_depth: tested,
                tested_depth_read: TESTED_DEPTH.get().1,
                stages_made: cached_made,
                uav_buffers: uav_capture.buffers,
                uav_textures: uav_capture.textures,
                srv_tex_hazard: uav_capture.srv_tex_hazard,
            });
        }
        #[cfg(feature = "diagnostics")]
        {
            let n = BIND_DIAG[4].fetch_add(1, Ordering::Relaxed) + 1;
            if n % 200_000 == 0 {
                let hits = BIND_DIAG[0].swap(0, Ordering::Relaxed);
                let misses = BIND_DIAG[1].swap(0, Ordering::Relaxed);
                let tbl_ms = BIND_DIAG[2].swap(0, Ordering::Relaxed) as f64 / 1e6;
                let rest_ms = BIND_DIAG[3].swap(0, Ordering::Relaxed) as f64 / 1e6;
                tracing::info!(
                    hits,
                    misses,
                    hit_pct = format_args!("{:.1}%", hits as f64 * 100.0 / (hits + misses).max(1) as f64),
                    tables_ms = format_args!("{tbl_ms:.1}"),
                    rest_ms = format_args!("{rest_ms:.1}"),
                    "draw::bind stage-table cache (200k binds)"
                );
            }
        }
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline;
        let kept_shadows = &mut self.kept_shadows;
        let kept_shadows_next = &mut self.kept_shadows_next;
        let mut ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            if s.in_place() {
                keep_shadow_dedup(kept_shadows, kept_shadows_next, list, s);
            } else {
                s.refresh(list, a, len);
            }
            list.mark_uav_dirty();
            if writes {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(a, len, pipeline);
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                self.last_drop_reason = Some(DropReason::NoBuffer);
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            let gpu = s.gpu(a) - (s.offset(a) & 15);
            self.set_root_gpu(list, compute, param, RootGpuKind::Uav, gpu);
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            self.last_drop_reason = Some(DropReason::NoSlots);
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = self.alloc_dynamic(heap, host, 1) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    self.last_drop_reason = Some(DropReason::NoSlots);
                    return false;
                };
                if let Some((a, s)) = &shadow {
                    let offset = s.offset(*a);
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            let len = s.left(*a).min(limit);
            if s.in_place() {
                keep_shadow_dedup(&mut self.kept_shadows, &mut self.kept_shadows_next, list, s);
            } else {
                s.refresh(list, *a, len);
            }
            list.mark_uav_dirty();
            if writes {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(*a, len, pipeline);
                s.gpu_writes(list, *a, len);
            }
        }
        let handle = host.gpu_handle(index);
        self.set_root_gpu(list, compute, param, RootGpuKind::Table, handle.ptr);
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            self.last_drop_reason = Some(DropReason::NoSlots);
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = self.alloc_dynamic(heap, host, 1) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    self.last_drop_reason = Some(DropReason::NoSlots);
                    return false;
                };
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        let offset = s.offset(*a);
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if crate::port_cbv::root_table_cbv_ok(offset) {
                            if s.in_place() && crate::port_cbv::on() {
                                crate::port_cbv::root_in_place();
                            }
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            if s.in_place() {
                keep_shadow_dedup(&mut self.kept_shadows, &mut self.kept_shadows_next, list, s);
            } else {
                s.refresh(list, *a, s.left(*a));
            }
        }
        let handle = host.gpu_handle(index);
        self.set_root_gpu(list, compute, param, RootGpuKind::Table, handle.ptr);
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&mut self, list: &HostList) -> Option<u64> {
        let list_ptr = std::ptr::from_ref(list) as usize;
        let view_heap = self.heaps[0];
        if view_heap != 0
            && self.cached_heap_buffers.0 == list_ptr
            && self.cached_heap_buffers.1 == view_heap
        {
            return Some(self.cached_heap_buffers.2);
        }
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // Stride buffers have aligned GPU addresses. Tag the list key so it
        // is distinct from aligned guest resource pointers.
        if !list.first_use(gpu as usize | 1) {
            if view_heap != 0 {
                self.cached_heap_buffers = (list_ptr, view_heap, gpu);
            }
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        if view_heap != 0 {
            self.cached_heap_buffers = (list_ptr, view_heap, gpu);
        }
        // Full pass once per frame; subsequent lists in the same frame run the
        // lightweight `heap_buffer_incremental` pass over changed buffer slots
        // and dirty watched shadows (`ARCH-08`) instead of leaving mid-frame
        // CPU writes stale until the next frame.
        let frame = crate::frame::current();
        if host.buffer_frame.load(Ordering::Relaxed) == frame
            || host.buffer_frame.swap(frame, Ordering::Relaxed) == frame
        {
            if !host.has_buffer_changes() {
                return Some(gpu);
            }
            if crate::host::aside::on() {
                let at = std::ptr::from_ref(heap) as usize;
                list.refreshes_coming(move || {
                    crate::descriptor::with_heap(at, |heap| {
                        heap.host()
                            .filter(|h| h.strides_gpu() == Some(gpu))
                            .map(|h| heap_buffer_incremental(heap, h, frame))
                    })
                    .flatten()
                    .unwrap_or_default()
                });
            } else {
                for (shadow, from, to) in heap_buffer_incremental(heap, host, frame) {
                    shadow.refresh(list, from, to - from);
                }
            }
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        let mut stage = crate::timeline::span("the heap pass: begun, or done here");
        if crate::host::aside::on() {
            let at = std::ptr::from_ref(heap) as usize;
            list.refreshes_coming(move || {
                crate::descriptor::with_heap(at, |heap| {
                    // The list retains the original stride buffer. A newly
                    // allocated heap at the same CPU address has another one.
                    heap.host().filter(|h| h.strides_gpu() == Some(gpu)).map(|h| heap_buffer_pass(heap, h, frame))
                })
                    .flatten()
                    .unwrap_or_default()
            });
        } else {
            for (shadow, from, to) in heap_buffer_pass(heap, host, frame) {
                shadow.refresh(list, from, to - from);
            }
        }
        #[cfg(feature = "diagnostics")]
        let refreshed = std::time::Instant::now();
        stage.next("the heap pass: textures");
        if !crate::heap_worker::kick(heap) {
            let _ = Self::heap_textures(Into::List(list), heap, host, None);
        }
        stage.end();
        #[cfg(feature = "diagnostics")]
        HEAP_SPLIT[2].fetch_add(refreshed.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        Some(gpu)
    }

    /// The GPU address of the bound view heap's stride/kind table without
    /// running the bindless heap buffer/texture sweep (for shaders that only
    /// inspect descriptors already mirrored via their descriptor tables).
    fn heap_strides(&mut self, list: &HostList) -> Option<u64> {
        let list_ptr = std::ptr::from_ref(list) as usize;
        let view_heap = self.heaps[0];
        if view_heap != 0
            && self.cached_heap_strides.0 == list_ptr
            && self.cached_heap_strides.1 == view_heap
        {
            return Some(self.cached_heap_strides.2);
        }
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // Tag bit 1 (distinct from `heap_buffers`'s `| 1` and aligned guest pointers).
        if list.first_use(gpu as usize | 2)
            && let Some(buffer) = host.strides_buffer()
        {
            list.keep(buffer);
        }
        if view_heap != 0 {
            self.cached_heap_strides = (list_ptr, view_heap, gpu);
        }
        Some(gpu)
    }

    /// Collects strictly the view-heap descriptor slots referenced by this
    /// decoration bake draw's bound root tables, root constants, and root
    /// views/CBVs (`ARCH-02`), avoiding scanning 20k unrelated world textures.
    /// Returns a sorted, deduplicated list of candidate slot indices.
    fn collect_bake_slots(
        &self,
        p: &HostPipeline,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) -> Vec<u32> {
        const BUF_LEN: usize = 4096;
        let count = host.count();
        let has_pending = host.has_pending_list_flips();
        let mut slots = Vec::new();
        let mut buf = [0u8; BUF_LEN];
        let scan_words = |words: &[u32], out: &mut Vec<u32>| {
            for &idx in words {
                if idx < count && (host.is_texture_queued(idx) || has_pending) {
                    out.push(idx);
                }
            }
        };
        let mut scan_bytes = |addr: u64, size: u32, out: &mut Vec<u32>| {
            let n = (size as usize).min(BUF_LEN) & !3;
            if addr == 0 || n < 4 {
                return;
            }
            if crate::host::read_into(addr as *const u8, &mut buf[..n]) {
                for chunk in buf[..n].chunks_exact(4) {
                    let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if idx < count && (host.is_texture_queued(idx) || has_pending) {
                        out.push(idx);
                    }
                }
            }
        };

        // Determine per-root-slot table length from pipeline stage plans.
        let mut table_len = [0u32; 64];
        for stage in &p.stages {
            for group in &stage.translated().tables.groups {
                let s = usize::from(group.slot);
                if s < table_len.len() {
                    let n = (group.max_dword / 8 + 1).min(256);
                    table_len[s] = table_len[s].max(n);
                }
            }
        }
        for (slot_idx, root) in self.graphics.iter().enumerate() {
            match root {
                Root::None => {}
                Root::Constants(vals) => scan_words(vals, &mut slots),
                Root::View(addr) => scan_bytes(*addr, BUF_LEN as u32, &mut slots),
                Root::Table(table) => {
                    let Some(theap) = self.heap_of(*table) else {
                        continue;
                    };
                    if theap.samplers() || (*table - theap.start) % 32 != 0 {
                        continue;
                    }
                    let same_heap = std::ptr::eq(theap, heap);
                    let max_n = table_len.get(slot_idx).copied().unwrap_or(0);
                    let n = if max_n > 0 { max_n } else { 64 };
                    let Some(base) = theap.index(*table) else {
                        continue;
                    };
                    for k in 0..n {
                        let slot = base + k;
                        let Some(handle) = theap.handle(slot) else {
                            break;
                        };
                        let words = record(*table, k);
                        if max_n == 0
                            && words[0] != crate::descriptor::VIEW_TAG
                            && crate::descriptor::inline_constant(words).is_none()
                        {
                            break;
                        }
                        if let Some((addr, size)) = crate::descriptor::inline_constant(words) {
                            scan_bytes(addr, size, &mut slots);
                            continue;
                        }
                        if same_heap && slot < count && (host.is_texture_queued(slot) || has_pending) {
                            slots.push(slot);
                        }
                        if let Some(View::Constant(addr, size)) = crate::descriptor::view(handle) {
                            scan_bytes(addr, size, &mut slots);
                        }
                    }
                }
            }
        }
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// Ensures textures referenced by a decoration bake draw that were claimed
    /// by background copy-queue uploads or deferred behind `neutral()` are
    /// uploaded onto `list` (running in `list`'s prelude ahead-list) with zero
    /// GPU settle and zero `WaitForSingleObject` (`ARCH-02`).
    fn flush_for_bake(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        scoped: &[u32],
        mut sync: Option<&mut BakeSync>,
    ) {
        static RESCUED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        // Only drain copy-queue uploads if one of this bake's scoped textures
        // is actually in flight on the ahead queue (never stall on unrelated
        // world textures).
        let needs_ahead = crate::texture::ahead::on()
            && scoped.iter().any(|&index| {
                heap.handle(index)
                    .and_then(crate::descriptor::view)
                    .and_then(|v| {
                        let placed = crate::descriptor::host::placed_texture(&v);
                        match placed.as_ref().or(Some(&v)) {
                            Some(&View::Shader(r, _)) => Some(crate::texture::ahead::there(r as This)),
                            _ => None,
                        }
                    })
                    .unwrap_or(false)
            });
        let drained = if needs_ahead { crate::texture::ahead::drain() } else { 0 };
        let flipped = host.texture_views_remade() + host.texture_slots_flip();
        let pending = if !scoped.is_empty() {
            host.take_pending_list_flips_for(scoped)
        } else {
            // Bounded fallback when bake indices are computed purely on GPU.
            let mut all = host.take_pending_list_flips();
            if all.len() > 64 {
                for extra in all.drain(64..) {
                    host.restore_pending_list_flip(extra);
                }
            }
            all
        };
        let mut promoted = 0usize;
        if !pending.is_empty() {
            let this_list = (std::ptr::from_ref(list) as usize, list.epoch());
            let sampled = UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            };
            for item in pending {
                let (origin, index, handle, key) = (item.origin, item.index(), item.handle(), item.key());
                if !host.slot_still_at(handle, key) {
                    continue;
                }
                let Some(view) = crate::descriptor::view(handle) else {
                    continue;
                };
                if let View::Placed(_, resource, desc) = &view {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                    if crate::texture::mapped_bytes(address as *const u8, len as usize) == 0 {
                        host.restore_pending_list_flip(item);
                        continue;
                    }
                }
                if sync.is_some() || origin != this_list {
                    let placed = crate::descriptor::host::placed_texture(&view);
                    if let Some(&View::Shader(resource, _)) = placed.as_ref().or(Some(&view)) {
                        let resource = resource as This;
                        let texture = crate::resource::host_resource(resource);
                        let title_fills = texture.is_some_and(crate::texture::title_written);
                        let first = match (&placed, &view) {
                            (Some(_), View::Placed(_, _, desc)) => crate::resource::first_mip(desc),
                            _ => 0,
                        };
                        if let (Some(sync), Some(copy)) = (sync.as_deref_mut(), crate::resource::host_copy(resource)) {
                            sync.keep.push(copy);
                        }
                        if title_fills
                            && !crate::texture::ahead::there(resource)
                            && (sync.is_some() || !crate::texture::unchanged(resource, first))
                        {
                            let was_staged = crate::texture::is_staged(resource);
                            let _ = crate::residency::take_staging_deferred();
                            readable(list, sampled.kind, &view);
                            let failed = crate::residency::take_staging_deferred() || crate::residency::upload_refused();
                            if !was_staged {
                                crate::texture::unstage(resource);
                            }
                            if failed {
                                host.restore_pending_list_flip(item);
                                continue;
                            }
                        }
                    }
                }
                if let Some(sync) = sync.as_deref_mut() {
                    sync.publish.push((index, handle, key));
                } else {
                    let _ = host.mirror(index, handle, true);
                }
                promoted += 1;
            }
        }
        if drained > 0 || flipped > 0 || promoted > 0 {
            let n = RESCUED[0].fetch_add(1, Ordering::Relaxed) + 1;
            let all_drained = RESCUED[1].fetch_add(drained as u64, Ordering::Relaxed) + drained as u64;
            let all_flipped = RESCUED[2].fetch_add(flipped as u64, Ordering::Relaxed) + flipped as u64;
            let all_promoted = RESCUED[3].fetch_add(promoted as u64, Ordering::Relaxed) + promoted as u64;
            if n.is_power_of_two() {
                tracing::info!(
                    frame = crate::frame::current(),
                    flushes = n,
                    drained_ahead = all_drained,
                    flipped = all_flipped,
                    promoted_from_lists = all_promoted,
                    "flushed deferred texture stand-ins asynchronously before decoration bake"
                );
            }
        }
    }

    /// Whether a graphics draw is a one-shot texture compositor / decoration
    /// bake whose bindless texture inputs must be flushed and up to date
    /// before recording the draw. Matches known bake pixel/geometry shaders
    /// and structurally detects depth-less 4+ MRT square atlas bakes
    /// (`>= 512x512`) so unseen NPC/head/clothing shader permutations never
    /// bake `neutral()` or `null` stand-ins into permanent ped textures.
    fn is_bake_draw(&self, p: &HostPipeline) -> bool {
        if p.is_bake_shader {
            return true;
        }
        self.depth == 0
            && self.targets.len() >= 4
            && self
                .targets
                .first()
                .and_then(|&t| crate::resource::texture_kind(t as This))
                .is_some_and(|(w, h, _)| w >= 512 && w == u64::from(h))
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and before decoration bake
    /// draws scoped strictly to the bake draw's referenced texture slots.
    /// Bake uploads record onto `list` (which `Host::submit` prepends as a
    /// prelude ahead-list strictly before `list`) and publish their descriptor
    /// slots via `list.defer_host` immediately before `ExecuteCommandLists`
    /// — zero GPU settle, zero `WaitForSingleObject` (`ARCH-02`).
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let is_bake = bake_ms > 0 && self.is_bake_draw(p);
        static FRAME: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(u64::MAX));
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: CachePadded<AtomicU64> = CachePadded(AtomicU64::new(0));
        let frame = crate::frame::current();
        let first = FRAME.0.load(Ordering::Relaxed) != frame
            && FRAME.0.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.0.store(0, Ordering::Relaxed);
        }
        if !is_bake && !(first && early_ms > 0) {
            return;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        if is_bake {
            let began = std::time::Instant::now();
            let exact = bake_exact();
            let scoped = self.collect_bake_slots(p, heap, host);
            let mut sync = BakeSync { publish: Vec::new(), keep: Vec::new() };
            crate::texture::BAKING.set(exact);
            self.flush_for_bake(list, heap, host, &scoped, Some(&mut sync));
            let left = bake_ms.saturating_sub(BAKED_MS.0.load(Ordering::Relaxed));
            let run_pass = left > 0
                && (!scoped.is_empty()
                    || first
                    || !exact
                    || {
                        let (writes, noted) = host.texture_news();
                        noted || BAKE_SERVED.load(Ordering::Relaxed) != writes + 1
                    });
            let mut served = false;
            if run_pass {
                let n = EARLY_TEXTURE_PASSES[1].fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
                    tracing::info!(
                        frame,
                        bakes = n,
                        scoped_slots = scoped.len(),
                        queued_at_the_last_pass = slots,
                        due_at_the_last_pass = due,
                        uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                        "decoration bake: scoped textures uploaded asynchronously ahead of draw (ARCH-02)"
                    );
                }
                // Scoped bake uploads run strictly on this bake's referenced
                // slots; if no slots were statically discoverable, run a tiny
                // bounded fallback (<= 8 ms, <= 64 slots) instead of draining
                // 20k unrelated world textures.
                let pass_ms = if scoped.is_empty() { left.min(8) } else { left };
                let scoped_arg = (!scoped.is_empty()).then_some(scoped.as_slice());
                PASS_BUDGET.set(Some((pass_ms, 2 << 30, true)));
                served = Self::heap_textures_scoped(Into::List(list), heap, host, Some(&mut sync), scoped_arg);
                PASS_BUDGET.set(None);
            }
            crate::texture::BAKING.set(false);
            for copy in sync.keep {
                list.keep(&copy.texture);
            }
            if !sync.publish.is_empty() {
                let heap_ptr = std::ptr::from_ref(heap) as usize;
                let host_id = host.id();
                let publish = sync.publish;
                list.defer_host(move || {
                    crate::descriptor::with_heap(heap_ptr, |heap| {
                        if let Some(host) = heap.host().filter(|h| h.id() == host_id) {
                            for &(index, handle, key) in &publish {
                                let _ = host.publish_uploaded_slot(index, handle, key);
                            }
                            host.texture_views_remade();
                            host.texture_slots_flip();
                        }
                    });
                });
            }
            if run_pass && served && exact && scoped.is_empty() {
                BAKE_SERVED.store(host.texture_news().0 + 1, Ordering::Relaxed);
            }
            BAKED_MS.0.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
            return;
        }
        let budget = if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        // What can wait for a later list is the worker's.
        if crate::heap_worker::kick(heap) {
            return;
        }
        let _pass = crate::heap_worker::on().then(crate::heap_worker::urgent);
        EARLY_TEXTURE_PASSES[0].fetch_add(1, Ordering::Relaxed);
        PASS_BUDGET.set(Some((budget.0, budget.1, false)));
        let _ = Self::heap_textures(Into::List(list), heap, host, None);
        PASS_BUDGET.set(None);
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    /// On a thread of its own the pass records into lists of its own, and
    /// the slots show what they upload from then on
    /// (`crate::heap_worker`).
    pub(crate) fn heap_textures(
        into: Into<'_>,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        sync: Option<&mut BakeSync>,
    ) -> bool {
        Self::heap_textures_scoped(into, heap, host, sync, None)
    }

    fn heap_textures_scoped(
        mut into: Into<'_>,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        mut sync: Option<&mut BakeSync>,
        scoped: Option<&[u32]>,
    ) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return true;
        }
        let memory = crate::host::host().and_then(|h| h.video_memory());
        // `VICTORY_TEXTURE_STAGE_MB` sets it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        let asked = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok())
        });
        let per_frame = match asked {
            Some(mb) => mb << 20,
            None => {
                let base = staging_budget_per_frame(memory.map(|[budget, ..]| budget));
                if crate::residency::limited_host_ram() {
                    base.min(128 << 20)
                } else {
                    base
                }
            }
        };
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut room = crate::resource::room(memory);
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        let baking = crate::texture::BAKING.get();
        if baking {
            host.texture_slots_bake_written(frame);
        }
        host.texture_slots_short(frame);
        // The slots of textures whose first level with memory (levels on
        // demand), or whose host copy for one with more levels, changed since
        // the last pass are made again from it; those whose new view's upload
        // has run are made from that.
        host.texture_views_remade();
        host.texture_slots_flip();
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room {
            host.texture_slots_waiting(frame);
        }
        let required = PASS_BUDGET.get().is_some_and(|(_, _, bake)| bake);
        let (mut due, census) = if let Some(wanted) = scoped {
            (host.texture_slots_for_bake(wanted), [0; 5])
        } else {
            let cap = if baking || required { 64 } else { 4096 };
            host.texture_slots_due(frame, cap)
        };
        // (Those that are due later after all that are due.)
        let on_time = due.len();
        if scoped.is_none() && baking && due.len() < 64 {
            let known: std::collections::HashSet<u32> = due.iter().map(|&(index, _)| index).collect();
            static AHEAD: AtomicU64 = AtomicU64::new(u64::MAX);
            let again = AHEAD.swap(frame, Ordering::Relaxed) == frame;
            let (ahead, _) = host.texture_slots_due(frame + BAKE_AHEAD[usize::from(again)], 64 - due.len());
            due.extend(ahead.into_iter().filter(|(index, _)| !known.contains(index)));
        }
        if scoped.is_none() {
            HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
            for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
                count.store(now, Ordering::Relaxed);
            }
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        // The worker may run two passes in a frame: no shorter than that.
        let took = if matches!(into, Into::Ahead(_)) { took.max(100) } else { took };
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && on_time >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes, _)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        // Apply the shared frame ceiling after early-pass and catch-up budgets.
        // A bake samples its inputs immediately and cannot retry that draw.
        let bytes_at_most = if required || asked.is_some() {
            bytes_at_most
        } else {
            crate::residency::prefetch_limit(
                bytes_at_most,
                crate::residency::pressured(),
                crate::residency::critical_commit(),
                crate::residency::limited_host_ram(),
                crate::texture::live_staging_bytes(),
            )
        };

        let began = std::time::Instant::now();
        if sync.is_none() {
            BAKE_SERVED.store(0, Ordering::Relaxed);
        }
        let mut served = true;
        for (index, write) in due {
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                served = false;
                break;
            }
            let ahead_worker = matches!(into, Into::Ahead(_));
            if ahead_worker && crate::heap_worker::make_way() {
                served = false;
                break;
            }
            // The record the slot has now, and the texture it names held
            // while a thread of ours works on it. (Released meanwhile, or
            // not a texture: nothing to upload.)
            let record = heap.handle(index).and_then(crate::descriptor::view_and_key);
            let _held = match &record {
                Some((View::Shader(object, _), _)) if ahead_worker => {
                    let held = crate::resource::hold(*object as This);
                    if held.is_none() {
                        host.texture_slot_done(index, write, None);
                        continue;
                    }
                    held
                }
                _ => None,
            };
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            // One found changed by such a look is looked at in the next frame
            // again: the textures left that change unnoticed are ones the
            // title paints all the time through another view of their pages
            // (all their blocks are shown at other addresses too), and each
            // was shown four frames late, over and over.
            let mut again_in = RECHECK_FRAMES[0];
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint(at, len) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                        Some(wait) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, looks + 1, uploaded)
                        }
                        // The schedule is over: looked at again every so many
                        // frames for as long as the slot shows it
                        // (`recheck_every`), or never again.
                        None => match recheck_every() {
                            0 => host.texture_slot_done(index, write, None),
                            every => host.texture_slot_recheck(index, write, frame + every, (at, len), print, looks, uploaded),
                        },
                    }
                    continue;
                }
                if recheck_soon() {
                    again_in = 1;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                if n < 200 || n.is_multiple_of(1000) {
                    // Whether its pages are the title's at other addresses
                    // too: what is written there is told of there alone.
                    let elsewhere = crate::inplace::shown_elsewhere(at as usize & !0xffff, (at + len) as usize);
                    let first = heap.handle(index).and_then(crate::descriptor::view).map_or(0, |view| match view {
                        View::Placed(false, _, desc) => crate::resource::first_mip(&desc),
                        _ => 0,
                    });
                    tracing::warn!(
                        first_level = first,
                        blocks_shown_elsewhere = elsewhere.len(),
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            let mut there = None;
            if let Some((View::Placed(false, resource, desc), _)) = &record {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let first = crate::resource::first_mip(desc);
                let len = crate::resource::placed_bytes_from(resource, first).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go; index missing
                    // VA span so remapped/page_written wakes this slot in 1 frame.
                    host.texture_slot_gone(index);
                    host.texture_slot_missing_mip(index, address, address.saturating_add(len));
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                host.texture_slot_here(index);
                there = Some(mapped);
                if mapped < len {
                    host.texture_slot_missing_mip(
                        index,
                        address.saturating_add(mapped),
                        address.saturating_add(len),
                    );
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                } else {
                    host.texture_slot_clear_missing_mip(index);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some((View::Shader(object, _), _)) = &record
                && crate::resource::buffer_bytes(*object as This).is_none()
                && !crate::resource::title_fills(*object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                if !has_its_texture {
                    refused += 1;
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            break;
                        }
                    } else if refused > 256 {
                        break;
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(500) {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            // Gate new host texture creation / staging on commit reserve after
            // `mapped == 0` reclamation (`texture_slot_gone`) has run, and
            // before mirroring a slot to an unuploaded texture.
            if !crate::residency::admit_texture_pass(required) {
                served = false;
                break;
            }
            // A slot that shows the texture by an earlier view of it (one
            // from another level) goes on doing so until the upload of what
            // the new one shows has run.
            let sync_bake = sync.is_some();
            let check_slot_views = !ahead_worker || (crate::descriptor::host::show_after() && crate::texture::ahead::on());
            let shows_other_slot = check_slot_views
                && !baking
                && !sync_bake
                && crate::resource::held_only()
                && heap.handle(index).is_some_and(|handle| host.texture_slot_shows_other(index, handle));
            let shows_other = !ahead_worker && shows_other_slot;
            let slot_made_from = check_slot_views
                && !shows_other_slot
                && !baking
                && !sync_bake
                && crate::descriptor::host::show_after()
                && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
            // A slot that shows nothing of it yet, or something else, shows it
            // when the upload has run too.
            let late = !ahead_worker
                && !shows_other
                && !baking
                && !sync_bake
                && crate::descriptor::host::show_after()
                && !slot_made_from;
            let (view, bake_key) = if ahead_worker {
                // On the worker the slot shows the view when the upload is in
                // the queue (`crate::heap_worker::Batch::publish`).
                (record.as_ref().map(|(view, _)| view.clone()), None)
            } else if sync_bake {
                match heap.handle(index).and_then(|handle| host.prepare_bake_slot(index, handle)) {
                    Some((view, key)) => (Some(view), Some(key)),
                    None => (None, None),
                }
            } else {
                let view = heap.handle(index).and_then(|handle| {
                    if shows_other || late { crate::descriptor::view(handle) } else { host.mirror(index, handle, true) }
                });
                (view, None)
            };
            let at_once = || {
                if (late || bake_key.is_some()) && let Some(handle) = heap.handle(index) {
                    let _ = host.mirror(index, handle, true);
                }
            };
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            if crate::residency::upload_refused() {
                served = false;
                break;
            }
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    at_once();
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let texture = crate::resource::host_resource(resource);
            if crate::residency::upload_refused() {
                served = false;
                break;
            }
            let title_fills = texture.is_some_and(crate::texture::title_written);
            if !title_fills {
                at_once();
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            // The host copy holds these very bytes: nothing to upload.
            let first = match (&placed, &view) {
                (Some(_), Some(View::Placed(_, _, desc))) => crate::resource::first_mip(desc),
                _ => 0,
            };
            let same = !tabled && {
                let frame = crate::frame::current();
                let changes = crate::watch::changes();
                let copy = crate::resource::made_copy(resource);
                let published = copy.as_deref().is_some_and(|c| {
                    c.title_bytes > 0
                        && all_blocks_published(c.title_start, c.title_bytes as usize)
                });
                if !crate::texture::BAKING.get()
                    && published
                    && copy
                        .as_deref()
                        .is_some_and(|c| c.hashed_frame_clean(frame, first, changes))
                {
                    true
                } else {
                    if !published
                        && let Some(c) = copy.as_deref()
                    {
                        c.invalidate_hashed_frame();
                    }
                    let unchanged = crate::texture::unchanged(resource, first);
                    if unchanged && published
                        && let Some(c) = copy.as_deref()
                    {
                        c.mark_hashed_frame(frame, first, changes);
                    }
                    unchanged
                }
            };
            if same {
                HEAP_TEXTURES_SAME.fetch_add(1, Ordering::Relaxed);
                if let Into::Ahead(batch) = &mut into {
                    if let (Some(handle), Some((_, key))) = (heap.handle(index), &record) {
                        batch.publish(index, handle, *key);
                    }
                } else if let (Some(sync), Some(key), Some(handle)) = (sync.as_deref_mut(), bake_key, heap.handle(index)) {
                    if let Some(copy) = crate::resource::host_copy(resource) {
                        sync.keep.push(copy);
                    }
                    sync.publish.push((index, handle, key));
                } else {
                    if shows_other && let Some(handle) = heap.handle(index) {
                        let _ = host.mirror(index, handle, true);
                    }
                    // Its bytes are in, or on the way there in a list before.
                    if late
                        && texture.is_some_and(crate::pool::fresh)
                        && !crate::texture::ahead::there(resource)
                        && let Into::List(list) = &into
                    {
                        if let Some(handle) = heap.handle(index) {
                            host.texture_slot_show_after(list, index, handle);
                        }
                    } else {
                        at_once();
                    }
                }
                let again = partly.map(|mapped| (frame + 1, mapped));
                if again.is_none() {
                    host.texture_slot_clear_missing_mip(index);
                }
                let memory = if again.is_none() && heap_recheck() { crate::texture::title_span(resource, first) } else { None };
                match memory {
                    Some((at, len)) => {
                        let print = crate::texture::fingerprint(at, len);
                        host.texture_slot_recheck(index, write, frame + again_in, (at, len), print, 0, frame);
                    }
                    None => host.texture_slot_done(index, write, again),
                }
                continue;
            }
            // The first upload of a texture no list has used: on a queue of
            // its own, and the slot shows it as soon as that has run.
            let ahead = if check_slot_views
                && use_ahead_copy_queue(
                    ahead_worker,
                    late,
                    placed.is_some(),
                    shows_other_slot,
                    baking,
                    sync_bake,
                    crate::descriptor::host::show_after(),
                    slot_made_from,
                    tabled,
                    texture.is_some_and(crate::pool::fresh),
                )
            {
                crate::texture::ahead::take(resource, first, there.filter(|_| partly.is_none()).unwrap_or(u64::MAX))
            } else {
                None
            };
            if let Some(ahead) = &ahead {
                use crate::texture::ahead::Taken;
                let (Taken::New(job) | Taken::Known(job)) = ahead;
                if let Some(handle) = heap.handle(index) {
                    host.texture_slot_show_ahead(job, index, handle);
                }
                if let Taken::New(job) = ahead {
                    crate::texture::ahead::send(job);
                }
            }
            if ahead.is_none() && crate::residency::upload_refused() {
                served = false;
                break;
            }
            let late = late && ahead.is_none();
            let _ = crate::residency::take_staging_deferred();
            if let Some(view) = &view
                && ahead.is_none()
            {
                // (What was found mapped of it a moment ago is not asked
                // again, unless the host copy may need more.)
                crate::texture::PICKING.set(Some(there.filter(|_| partly.is_none()).unwrap_or(u64::MAX)));
                match &mut into {
                    Into::List(list) => readable(list, sampled.kind, view),
                    Into::Ahead(batch) => {
                        if !batch.record(|list| readable(list, sampled.kind, view)) {
                            crate::texture::PICKING.set(None);
                            return false;
                        }
                    }
                }
                crate::texture::PICKING.set(None);
            }
            // A failed or deferred staging allocation leaves this slot due.
            // Publishing it or marking it served would lose the only retry for
            // some materials.
            if ahead.is_none()
                && (crate::residency::take_staging_deferred() || crate::residency::upload_refused())
            {
                served = false;
                break;
            }
            if ahead.is_none()
                && let Into::Ahead(batch) = &mut into
                && let (Some(handle), Some((_, key))) = (heap.handle(index), &record)
            {
                batch.publish(index, handle, *key);
            }
            if let (Some(sync), Some(key), Some(handle)) = (sync.as_deref_mut(), bake_key, heap.handle(index)) {
                if let Some(copy) = crate::resource::host_copy(resource) {
                    sync.keep.push(copy);
                }
                sync.publish.push((index, handle, key));
            }
            if let Into::List(list) = &into {
                if shows_other && let Some(handle) = heap.handle(index) {
                    host.texture_slot_flip(list, index, handle);
                }
                if late && let Some(handle) = heap.handle(index) {
                    host.texture_slot_show_after(list, index, handle);
                }
            }
            #[cfg(feature = "diagnostics")]
            if crate::book::on()
                && let Some(texture) = texture.filter(|t| crate::pool::fresh(t))
            {
                use windows::core::Interface;
                let raw = texture.as_raw() as usize;
                let asked = crate::host::submissions();
                let list_opt = match &mut into {
                    Into::List(list) => Some(*list),
                    Into::Ahead(batch) => batch.list(),
                };
                if let Some(list) = list_opt {
                    list.defer_host(move || {
                        let (started, finished) = crate::host::submissions();
                        crate::book::note(raw, || {
                            format!(
                                "; upload for slot {index} recorded at submissions {asked:?}, sent as submission {started} (finished {finished}) at {:.3} s",
                                crate::host::now_ms() as f64 / 1e3
                            )
                        });
                    });
                }
            }
            // A texture view the host refuses removes the device: name it.
            #[cfg(feature = "diagnostics")]
            {
                static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !GONE.load(Ordering::Relaxed)
                    && let Some(why) = crate::descriptor::host::device_removed()
                {
                    GONE.store(true, Ordering::Relaxed);
                    let text: String = format!("{view:?}").chars().take(300).collect();
                    tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                    crate::host::log_removal_context();
                    return false;
                }
            }
            let (bytes, whole) = crate::texture::staged(resource, there.is_none()).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            if ahead.is_none() && let Into::Ahead(batch) = &mut into {
                batch.uploaded(bytes);
            }
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = crate::resource::room(crate::host::host().and_then(|h| h.video_memory()));
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            if again.is_none() {
                host.texture_slot_clear_missing_mip(index);
            }
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let print = crate::texture::fingerprint(at, len);
                    host.texture_slot_recheck(index, write, frame + again_in, (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
        if sync.is_none() && baking && served {
            BAKE_SERVED.store(host.texture_news().0 + 1, Ordering::Relaxed);
        }
        served
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Stamps `slot_tabled` for the descriptor tables bound on a dropped draw,
    /// dispatch, or indirect execution (`CPU-02`), preventing `~8.6k` RT-off
    /// indirect / dropped-dispatch table slots from flooding `heap_sync` as
    /// untracked `no_draw_came` copies.
    fn stamp_dropped_tables(&self, compute: bool, p: Option<&HostPipeline>) {
        if let Some(p) = p {
            let spans: &[(u8, u32, u32)] = if table_stages() { &p.stage_spans } else { &[] };
            for stage in &p.stages {
                for group in &stage.translated().tables.groups {
                    let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(group.slot)) else {
                        continue;
                    };
                    let Some(heap) = self.heap_of(table) else {
                        continue;
                    };
                    if heap.samplers() || (table - heap.start) % 32 != 0 {
                        continue;
                    }
                    let Some(host) = heap.host() else {
                        continue;
                    };
                    let mut n = group.max_dword / 8 + 1;
                    if let Some(&(_, with_constants, all)) = spans.iter().find(|s| s.0 == group.slot) {
                        let past = heap
                            .index(table)
                            .is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host.count()));
                        n = n.max(if past { all } else { with_constants });
                    }
                    if let Some(title) = heap.index(table).filter(|first| first + n <= host.count()) {
                        for k in (0..n).filter(|&k| n <= 64 || group.entry(k).is_some()) {
                            host.slot_tabled(title + k);
                        }
                    }
                }
            }
            return;
        }
        // No host pipeline (e.g. RT-off builder or skipped PSO): stamp only
        // short contiguous per-draw/dispatch frame-allocator tables (`1..=64`
        // slots), never touching >64-slot bindless arrays.
        for root in self.roots_ref(compute) {
            let Root::Table(table) = *root else {
                continue;
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let Some(title) = heap.index(table) else {
                continue;
            };
            let mut n = 0u32;
            while n <= 64 && title + n < host.count() {
                let words = record(table, n);
                if words[0] != crate::descriptor::VIEW_TAG && crate::descriptor::inline_constant(words).is_none() {
                    break;
                }
                n += 1;
            }
            if n > 0 && n <= 64 {
                for k in 0..n {
                    host.slot_tabled(title + k);
                }
            }
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        self.last_drop_reason = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            self.stamp_dropped_tables(true, None);
            record_drop(DropReason::LeftOut);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            self.stamp_dropped_tables(true, None);
            record_drop(DropReason::LeftOut);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            self.stamp_dropped_tables(true, None);
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            record_drop(DropReason::NoPso);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        // Counted in threads (`USE_THREAD_DIMENSIONS`, flags bit 5; fork pull
        // request #646): whole groups of them, and the per-axis counts where
        // the last group is cut (`USER_DISPATCH_THREADS`).
        let size = (flags & THREAD_DIMENSIONS != 0 && thread_dimensions())
            .then(|| p.stages.first().and_then(|s| s.translated().threads))
            .flatten();
        let threads = [x, y, z];
        let ([gx_req, gy_req, gz_req], partial) = match size {
            Some(group) => match dispatch_dimensions(threads, group, flags) {
                Some(dims) => dims,
                None => return,
            },
            None if flags & THREAD_DIMENSIONS != 0 && thread_dimensions() && threads.contains(&0) => return,
            None => (threads, [0; 3]),
        };
        if partial != [0; 3]
            && let Some(stage) = p.stages.first()
            && !stage.translated().host.user_data.iter().any(|u| {
                u.root == RootType::Other(victory_shader::emit::USER_DISPATCH_THREADS)
                    || u.root == RootType::Other(victory_shader::emit::USER_DISPATCH_THREADS_X)
            })
        {
            once!(
                pipeline = %p.name,
                threads = ?partial,
                group = ?size.unwrap_or([0; 3]),
                "a DispatchX counting threads cuts its last group, and its program does not take the count: \
                 the threads past it run"
            );
        }
        // More than 65,535 groups in x: spread across z for a program that
        // takes the spread words (`USER_DISPATCH_SPREAD`).
        let (row, asked, gx, gy, gz) =
            victory_shader::emit::spread_groups(gx_req, gy_req, gz_req, spreads(p).is_some());
        if row != 0 {
            self.local_spread[0] += 1;
        } else if gx > 65_535 || gy > 65_535 || gz > 65_535 {
            once!(gx, gy, gz, "a dispatch asks for more than 65,535 groups in a dimension and is not spread");
        }
        self.dispatch_spread = [row, asked];
        self.dispatch_threads = partial;
        let bound = self.bind(list, p, true);
        self.dispatch_spread = [0, 0];
        self.dispatch_threads = [0; 3];
        if !bound {
            self.stamp_dropped_tables(true, Some(p));
            record_drop(self.last_drop_reason.unwrap_or(DropReason::NoSlots));
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_rests("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_builder(self.pipeline as This) {
            crate::capture::op(list as *const _ as usize, || format!("builder {}", p.name));
            self.list_dropped_roots([x, y, z]);
        }
        if size.is_some() {
            if !crate::pipeline::is_builder(self.pipeline as This) {
                once!(pipeline = p.name, "a dispatch counted in threads that is not the structure builder's");
            }
            if let Some((param, word)) = compute_word(p, victory_shader::emit::USER_DISPATCH_THREADS_X) {
                // SAFETY: an open list; the root signature has the word.
                unsafe { list.list.SetComputeRoot32BitConstant(param, threads[0], word) };
                self.host_root_args[1].invalidate_consts(param);
            }
        }
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(gx, gy, gz) };
        self.buffers_written(list);
        self.local_dispatched += 1;
        if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
            && crate::ambient::on()
            && self.ambient(list, p, role)
        {
            self.invalidate_compute_host_state();
        }
    }

    /// The stand-in for the indirect light after a dispatch of `p`
    /// (`crate::ambient`). Whether it recorded anything.
    fn ambient(&self, list: &HostList, p: &HostPipeline, role: crate::ambient::Role) -> bool {
        use crate::ambient::{AIR, DEPTH, LIGHT, RADIANCE, REFLECTIONS, REFLECTIONS_ALPHA, Role, Views};
        if role == Role::ReflectAlpha && !crate::ambient::reflect_alpha_on() {
            return false;
        }
        let Some(t) = p.stages.first().map(|s| s.translated()) else {
            return false;
        };
        let Some(heap) = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This).host()).flatten()
        else {
            return false;
        };
        let entry = |kind: ResourceType, (slot, offset): (u8, u32)| {
            t.tables
                .groups
                .iter()
                .find(|g| g.slot == slot)
                .and_then(|g| g.entry(offset / 8))
                .filter(|u| u.kind == kind && u.table_offset_dwords == offset)?;
            let Some(Root::Table(table)) = self.roots_ref(true).get(usize::from(slot)) else {
                return None;
            };
            let resource = match crate::descriptor::view(table + u64::from(offset) * 4)? {
                View::Shader(r, _) | View::Unordered(r, ..) => r,
                _ => return None,
            };
            let start = match self.ambient_tables.iter().find(|m| m.0 == slot) {
                Some(m) => m.1,
                None => self.user_value(true, slot, RootType::Table, 0),
            };
            let host = crate::resource::host_resource(resource as This)?;
            // SAFETY: a live resource.
            let desc = unsafe { host.GetDesc() };
            Some((resource, desc, heap.gpu_handle((start + offset * 4) / 32)))
        };
        let at = match role {
            Role::Resolve => LIGHT,
            Role::Radiance => RADIANCE,
            Role::Air => AIR,
            Role::Reflect => REFLECTIONS,
            Role::ReflectAlpha => REFLECTIONS_ALPHA,
        };
        let Some((image_res, of, image)) = entry(ResourceType::Uav, at) else {
            once!(?role, pipeline = p.name, "no stand-in for the indirect light: other tables");
            return false;
        };
        // (R32_UINT holding R9G9B9E5; R11G11B10_FLOAT.)
        let format = if matches!(role, Role::Resolve | Role::Reflect | Role::ReflectAlpha) { 42 } else { 26 };
        let volume = of.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        if of.Format.0 != format || volume != (role == Role::Air) {
            once!(?role, pipeline = p.name, format = of.Format.0, "no stand-in for the indirect light: another image");
            return false;
        }
        let depth = match role {
            Role::Resolve => {
                let Some((depth_res, seen, depth)) = entry(ResourceType::Srv, DEPTH) else {
                    once!(?role, pipeline = p.name, "no stand-in for the indirect light: other tables");
                    return false;
                };
                if depth_res != image_res && (seen.Width, seen.Height) != (of.Width, of.Height) {
                    once!(?role, pipeline = p.name, format = of.Format.0, "no stand-in for the indirect light: another image");
                    return false;
                }
                depth
            }
            Role::Radiance | Role::Air | Role::Reflect | Role::ReflectAlpha => image,
        };
        crate::ambient::after(list, role, &Views {
            image,
            depth,
            size: [of.Width as u32, of.Height],
            slices: u32::from(of.DepthOrArraySize),
        })
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let skipped = mesh_test() == 1 || !mesh_kept(self.pipeline);
        let ran = !skipped
            && match (self.prepare_for(list, false, true), list.list6()) {
                (Some(p), Some(l6)) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    #[cfg(feature = "diagnostics")]
                    list.ran_indirect(p.pso.as_raw() as usize, &p.name, crate::host::indirects::DIRECT, x, &self.user);
                    // (The host takes 65,535 groups in a dimension: more is
                    // undefined there.)
                    if x > 65_535 {
                        once!(x, "a mesh dispatch of more groups than the host takes: cut");
                    }
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x.min(65_535), 1, 1) };
                    self.drew(list, Some(p));
                    self.mesh_scissor(list, false);
                    self.local_drawn += 1;
                    true
                }
                _ => false,
            };
        if !ran {
            let reason = if skipped {
                DropReason::LeftOut
            } else {
                self.last_drop_reason.unwrap_or(DropReason::NoPso)
            };
            record_drop(reason);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Runs an indirect mesh dispatch whose 1D group count lies in GPU memory at
    /// `indirect_va` (`DispatchMeshX` with a non-zero indirect argument address).
    pub(crate) fn dispatch_mesh_indirect(&mut self, list: &HostList, indirect_va: u64) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh_indirect", 1, 0);
        if !self.execute_indirect(
            list,
            (crate::indirect::KIND_DISPATCH_MESH_1D, 4, 1),
            indirect_va,
            None,
        ) {
            record_drop(self.last_drop_reason.unwrap_or(DropReason::NoPso));
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    /// Whether title slot `index` in `host` already mirrors the exact view
    /// record `words` on the lock-free quick table, allowing callers to skip
    /// `host.mirror_after` (`VIEWS.read()` + `View::clone()`).
    #[inline]
    fn quick_slot_matches(host: &crate::descriptor::host::HostHeap, index: u32, words: [u32; 3]) -> bool {
        if words[0] != crate::descriptor::VIEW_TAG {
            return false;
        }
        let key = u64::from(words[2]) << 32 | u64::from(words[1]);
        host.quick_matches(index, key)
    }

    /// Checks whether a CBV-bearing `TableGroup` satisfies all four V1 guards
    /// (`CPU-18`) to bind directly in its title host-heap slots instead of
    /// allocating a dynamic table snapshot. When eligible, retains each
    /// in-place CBV shadow buffer on `list` via `keep_shadow_dedup`.
    fn can_bind_cbv_table_in_place(
        &mut self,
        list: &HostList,
        group: &TableGroup,
        has_half: bool,
        compute: bool,
        spans: &[(u8, u32, u32)],
    ) -> bool {
        let direct_on = crate::port_cbv::table_direct_on();
        let has_indexed = !group.indexed.is_empty();
        if !crate::port_cbv::table_direct_eligible(direct_on, has_half, has_indexed, true, true) {
            return false;
        }
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(group.slot)) else {
            return false;
        };
        let Some(heap) = self.heap_of(table) else {
            return false;
        };
        if heap.samplers() || (table - heap.start) % 32 != 0 {
            return false;
        }
        let Some(host) = heap.host() else {
            return false;
        };
        let host_count = host.count();
        let n = group.max_dword / 8 + 1;
        let span_all = spans
            .iter()
            .find(|s| s.0 == group.slot)
            .map_or(n, |&(_, _, all)| all.max(n));
        let fits_host = !group.needs_snapshot(heap, host_count, table)
            && heap
                .index(table)
                .is_some_and(|at| u64::from(at) + u64::from(span_all) <= u64::from(host_count));
        if !crate::port_cbv::table_direct_eligible(direct_on, has_half, has_indexed, fits_host, true)
        {
            return false;
        }
        let mut all_cbvs_in_place = true;
        for u in group.entries.iter().filter(|u| u.kind == ResourceType::Cbv) {
            if u.table_offset_dwords % 8 != 0 {
                all_cbvs_in_place = false;
                break;
            }
            let words = record(table, u.table_offset_dwords / 8);
            let cbv = crate::descriptor::inline_constant(words).or_else(|| {
                match crate::descriptor::view_from_words(words) {
                    Some(View::Constant(address, size)) => Some((address, size)),
                    _ => None,
                }
            });
            let Some((address, size)) = cbv else {
                all_cbvs_in_place = false;
                break;
            };
            let mut ok = false;
            let _ = crate::shadow::with_in_place(address, u64::from(size), |shadow| {
                let gpu = shadow.gpu(address);
                let base = shadow.gpu(shadow.span().0);
                let width = shadow.width();
                if crate::port_cbv::raw_cbv_layout(base, width, gpu, size).is_some() {
                    keep_shadow_dedup(&mut self.kept_shadows, &mut self.kept_shadows_next, list, shadow);
                    ok = true;
                }
            });
            if !ok {
                all_cbvs_in_place = false;
                break;
            }
        }
        crate::port_cbv::table_direct_eligible(
            direct_on,
            has_half,
            has_indexed,
            fits_host,
            all_cbvs_in_place,
        )
    }

    fn table_entry(
        &mut self,
        list: &HostList,
        u: &UsedResource,
        compute: bool,
        mut uav_capture: Option<&mut UavCapture>,
    ) {
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        // Fast path for in-place CBV entries (`CPU-18`): raw twin already lives
        // in the title host heap; skip `mirror()` when `quick` matches.
        if u.kind == ResourceType::Cbv {
            // SAFETY: `handle` points to a descriptor slot inside `heap`.
            let words = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let size = crate::descriptor::inline_constant(words)
                .map(|(_, s)| s)
                .unwrap_or(0);
            if !Self::quick_slot_matches(host, index, words) {
                let _ = host.mirror(index, handle, true);
            }
            host.slot_tabled(index);
            crate::port_cbv::in_place(size);
            return;
        }
        // Fast path for unchanged SRV entries in the same list/epoch (`CPU-18`/`CPU-21`).
        if u.kind == ResourceType::Srv {
            // SAFETY: `handle` points to a descriptor slot inside `heap`.
            let words = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            if Self::quick_slot_matches(host, index, words) {
                let epoch = list.epoch();
                let tested = TESTED_DEPTH.get();
                if let Some(&(ready, tabled)) = self.listed_state.get(&words)
                    && ready.0 == epoch
                    && ready.1 == tested.0
                    && tabled == Some((index, true))
                {
                    if ready.2 {
                        TESTED_DEPTH.set((tested.0, true));
                    }
                    if let Some(cap) = uav_capture {
                        cap.skipped_srv_check = true;
                    }
                    return;
                }
                if let Some(view) = host.mirror(index, handle, true) {
                    let _ = crate::residency::take_staging_deferred();
                    TESTED_DEPTH.set((tested.0, false));
                    readable_inner(list, u.kind, &view, uav_capture.as_deref_mut());
                    let depth = TESTED_DEPTH.get().1;
                    TESTED_DEPTH.set((tested.0, tested.1 || depth));
                    if crate::descriptor::host::is_texture_view(&view) {
                        host.texture_slot_tabled(index, true);
                    } else {
                        host.slot_tabled(index);
                    }
                    if !crate::residency::take_staging_deferred() {
                        self.listed_state
                            .insert(words, ((list.epoch(), tested.0, depth), Some((index, true))));
                    }
                }
                return;
            }
        }
        if let Some(view) = host.mirror(index, handle, true) {
            readable_inner(list, u.kind, &view, uav_capture);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            } else {
                host.slot_tabled(index);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    ///
    /// `half`: the table descriptors the shader moves sixteen bits a
    /// channel as unsigned integers (`HostShader::half_views`). A table
    /// with a 16-bit float or signed buffer view at one of them is made
    /// afresh as well, with that view in the unsigned format of the same
    /// layout (`descriptor::host::half_view`): the title's own slot keeps
    /// the title's format for the shaders that read it as floats.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        plan: &TablePlan,
        half: &[(u8, u32)],
        compute: bool,
        spans: &[(u8, u32, u32)],
        staged: &mut Few<(u64, u32), 8>,
        mut uav_capture: Option<&mut UavCapture>,
    ) -> Few<(u8, u32), 16> {
        let mut fresh = [0u64; 4];
        let selected_slot = |mask: &[u64; 4], slot: u8| mask[usize::from(slot / 64)] & (1u64 << (slot % 64)) != 0;
        for &(slot, at) in half {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(view) = crate::descriptor::view(table + u64::from(at) * 4) else {
                continue;
            };
            if crate::descriptor::host::half_view(&view).is_some() {
                fresh[usize::from(slot / 64)] |= 1u64 << (slot % 64);
            } else if let Some(format) = crate::descriptor::host::half_inexact(&view) {
                once!(
                    format,
                    "a shader moves a buffer's 16-bit channels as integers through a view whose values the console converts: they are read and written unconverted"
                );
            }
        }
        for group in &plan.groups {
            let slot = group.slot;
            let has_half = selected_slot(&fresh, slot);
            let needs_snapshot = if group.has_cbv {
                !self.can_bind_cbv_table_in_place(list, group, has_half, compute, spans)
            } else {
                match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => self.heap_of(table).is_some_and(|heap| {
                        heap.host()
                            .is_some_and(|host| group.needs_snapshot(heap, host.count(), table))
                    }),
                    _ => false,
                }
            };
            if needs_snapshot {
                fresh[usize::from(slot / 64)] |= 1u64 << (slot % 64);
            }
        }
        for &(g, e) in &plan.direct {
            let group = &plan.groups[g];
            if !selected_slot(&fresh, group.slot) {
                self.table_entry(list, &group.entries[e], compute, uav_capture.as_deref_mut());
            }
        }
        let mut made = Few::default();
        let mut records = std::mem::take(&mut self.table_records);
        for group in &plan.groups {
            let slot = group.slot;
            if !selected_slot(&fresh, slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries = &group.entries;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Few<u32, 8> = Few::default();
            for index in &group.indexed {
                let constant = self.user_value(
                    compute,
                    index.constant_slot,
                    RootType::Constants,
                    index.constant_dword,
                );
                // The shader adds i32 byte offsets and shifts by 5.
                selected.push(index.byte_offset.wrapping_add(constant << index.shift) / 32);
            }
            // The descriptors the shader moves halves through are made too,
            // and the table reaches them: they may lie past the range starts
            // the entries give (a range of several views names only its
            // first; review on #927).
            let mut half_slots: Few<u32, 8> = Few::default();
            for &(s, at) in half {
                if s == slot {
                    half_slots.push(at / 8);
                }
            }
            half_slots.sort_dedup();
            for &h in &*half_slots {
                selected.push(h);
            }
            selected.sort_dedup();
            let n = (group.max_dword / 8).max(selected.last().copied().unwrap_or(0)) + 1;
            // With `table_stages`: as many slots as the stage of this draw
            // that reads furthest into the table needs among those that
            // make it afresh too: the ones reading a constant view in it,
            // or every stage where the table reaches past the host heap's
            // slots (`spans` is empty with the switch off).
            let n = match spans.iter().find(|s| s.0 == slot) {
                Some(&(_, with_constants, all)) => {
                    let past = heap
                        .index(table)
                        .is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host.count()));
                    n.max(if past { all } else { with_constants })
                }
                None => n,
            };
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // Reuse `self.table_records` (`CPU-18`/`CPU-45`) so hashing,
            // collision check, `fresh_records` append, and slot building read
            // title memory once without per-table stack zeroing or heap churn.
            records.clear();
            records.reserve(n as usize);
            for k in 0..n {
                records.push(record(table, k));
            }
            // The table and what its entries hold (record words), plus which
            // slots are shown as unsigned 16-bit twins (`half_slots`).
            let key = table_hash_records(table, &records, &half_slots);
            // Who reads of it, and as what, as far as [`readable`] goes by it.
            let reader = (std::ptr::from_ref(plan) as usize, plan.direct.len(), STORES.get(), TESTED_DEPTH.get().0);
            let same = {
                let fresh_records = &self.fresh_records;
                self.fresh_tables.get_mut(&key).filter(|f| {
                    f.table == table
                        && f.n == n
                        && f.half.as_slice() == &*half_slots
                        && fresh_records[f.at..f.at + n as usize] == *records
                })
            };
            if let Some(known) = same {
                let first = known.first;
                // Found in the table an earlier stage of this draw made
                // ([`table_stages`]): the title's own slots of the texture
                // views this stage reads are stamped and made as the
                // making of a table of its own did it (below).
                let staged_title = staged
                    .contains(&(table, n))
                    .then(|| heap.index(table).filter(|at| at + n <= host.count()))
                    .flatten();
                // This reader had it brought up to date, and nothing in the
                // list changed since: nothing to do again.
                let once = ready_once()
                    && selected.is_empty()
                    && (!group.has_uav || !STORES.get())
                    && uav_capture.is_none();
                if once
                    && staged_title.is_none()
                    && known.epoch == list.epoch()
                    && let Some((_, depth)) = known.ready.iter().find(|(r, _)| *r == reader)
                {
                    if *depth {
                        TESTED_DEPTH.set((reader.3, true));
                    }
                    made.push((slot, first * 32));
                    continue;
                }
                let depth_before = TESTED_DEPTH.get();
                TESTED_DEPTH.set((depth_before.0, false));
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let k = (u.table_offset_dwords / 8) as usize;
                    if let Some(view) = crate::descriptor::view_from_words(records[k]) {
                        readable_inner(list, u.kind, &view, uav_capture.as_deref_mut());
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let handle = table + u64::from(u.table_offset_dwords) * 4;
                            host.texture_slot_tabled(title + k as u32, true);
                            if !Self::quick_slot_matches(host, title + k as u32, records[k]) {
                                host.mirror_after(list, title + k as u32, handle);
                            }
                        }
                    }
                }
                for &k in &*selected {
                    if group.entry(k).is_some() {
                        continue;
                    }
                    if let Some(view) = crate::descriptor::view_from_words(records[k as usize])
                        && let Some(kind) = view_kind(&view)
                    {
                        readable_inner(list, kind, &view, uav_capture.as_deref_mut());
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let handle = table + u64::from(k) * 32;
                            host.texture_slot_tabled(title + k, true);
                            if !Self::quick_slot_matches(host, title + k, records[k as usize]) {
                                host.mirror_after(list, title + k, handle);
                            }
                        }
                    }
                }
                let depth = TESTED_DEPTH.get().1;
                TESTED_DEPTH.set((depth_before.0, depth_before.1 || depth));
                if once {
                    // (The epoch as it is now: what this changed itself it
                    // need not do again.)
                    let epoch = list.epoch();
                    if known.epoch != epoch {
                        known.epoch = epoch;
                        known.ready.clear();
                    }
                    if known.ready.len() < 16 {
                        known.ready.push((reader, depth));
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = self.alloc_dynamic(heap, host, n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            let at = self.fresh_records.len();
            self.fresh_records.extend_from_slice(&records);
            let staged_ok = !spans.is_empty() && half_slots.is_empty();
            self.fresh_tables.insert(
                key,
                FreshTable {
                    table,
                    n,
                    at,
                    first,
                    half: half_slots.to_vec(),
                    epoch: 0,
                    ready: Few::default(),
                },
            );
            if staged_ok {
                staged.push((table, n));
            }
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // They are such a table's, whatever they hold. (Of a big one,
            // which may lie over slots shaders pick by index, only the ones
            // the shader names.)
            if let Some(title) = title_first {
                for k in (0..n).filter(|&k| n <= 64 || group.entry(k).is_some()) {
                    host.slot_tabled(title + k);
                }
            }
            let list_views = crate::descriptor::host::list_views();
            let listed_state = &mut self.listed_state;
            let scratch_base = &mut self.scratch_base;
            let kept_shadows = &mut self.kept_shadows;
            let kept_shadows_next = &mut self.kept_shadows_next;
            let scratch_va = |scratch_base: &mut (usize, u64), buffer: &ID3D12Resource| -> u64 {
                let raw = buffer.as_raw() as usize;
                if scratch_base.0 == raw {
                    scratch_base.1
                } else {
                    // SAFETY: live scratch buffer retained by `list` until reset.
                    let va = unsafe { buffer.GetGPUVirtualAddress() };
                    *scratch_base = (raw, va);
                    va
                }
            };
            crate::descriptor::host::with_table_batch(|tb| {
                // The whole range: a shader may index past the entries its
                // used-resource list names (descriptor arrays).
                for k in 0..n {
                    let index = first + k;
                    let words = records[k as usize];
                    let is_half = !half_slots.is_empty() && half_slots.binary_search(&k).is_ok();
                    // Fast path for inline constant-buffer records: avoid entering
                    // `LIST_VIEWS` and decoding twice.
                    if !is_half
                        && let Some((address, size)) = crate::descriptor::inline_constant(words)
                    {
                        if crate::port_cbv::in_place_on() {
                            let mut done = false;
                            let found = crate::shadow::with_in_place(address, u64::from(size), |shadow| {
                                let gpu = shadow.gpu(address);
                                let base = shadow.gpu(shadow.span().0);
                                let width = shadow.width();
                                if host.constant_raw_at_known(index, &shadow.buffer, base, width, gpu, size, tb) {
                                    keep_shadow_dedup(kept_shadows, kept_shadows_next, list, shadow);
                                    crate::port_cbv::in_place(size);
                                    done = true;
                                } else if size == 0 || crate::port_cbv::raw_ok(gpu) {
                                    crate::port_cbv::copied_oob_or_empty();
                                } else {
                                    crate::port_cbv::copied(false);
                                }
                            });
                            if done {
                                continue;
                            }
                            if found.is_none() {
                                crate::port_cbv::copied(true);
                            }
                        }
                        let Some((buffer, gpu, to)) = list.scratch_held(u64::from(size)) else {
                            host.make_null_batched(index, &mut tb.copy);
                            continue;
                        };
                        let base = scratch_va(scratch_base, &buffer);
                        let slot_id = (std::ptr::from_ref(heap) as usize, index);
                        list.defer_constants(to, address as *const u8, size as usize, slot_id);
                        host.constant_at_batched(index, &buffer, base, gpu, size, tb);
                        continue;
                    }
                    // Persistent per-thread CPU descriptor cache + per-list readiness:
                    // single TLS lookup and zero `View::clone()` on readiness hit.
                    if list_views && !is_half && words[0] == crate::descriptor::VIEW_TAG {
                        let entry_kind = group.entry(k).map(|u| u.kind);
                        let is_selected = entry_kind.is_none() && selected.binary_search(&k).is_ok();
                        let (mut ready, mut tabled) = listed_state.get(&words).copied().unwrap_or(((0, 0, false), None));
                        let tested = TESTED_DEPTH.get();
                        let epoch = list.epoch();
                        if let Some((cached_kind, texture, view_opt)) = host.make_listed_cached(
                            words,
                            index,
                            &mut tb.copy,
                            |lk| {
                                let kind = entry_kind.or_else(|| is_selected.then_some(lk).flatten());
                                match kind {
                                    Some(ResourceType::Srv) => !(ready.0 == epoch && ready.1 == tested.0),
                                    Some(_) => true,
                                    None => false,
                                }
                            },
                        ) {
                            let kind = entry_kind.or_else(|| is_selected.then_some(cached_kind).flatten());
                            let was = (ready, tabled);
                            if let Some(kind) = kind {
                                if kind == ResourceType::Srv && ready.0 == epoch && ready.1 == tested.0 {
                                    if ready.2 {
                                        TESTED_DEPTH.set((tested.0, true));
                                    }
                                    if let Some(cap) = uav_capture.as_deref_mut() {
                                        cap.skipped_srv_check = true;
                                    }
                                } else if let Some(view) = view_opt {
                                    let _ = crate::residency::take_staging_deferred();
                                    TESTED_DEPTH.set((tested.0, false));
                                    readable_inner(list, kind, &view, uav_capture.as_deref_mut());
                                    let depth = TESTED_DEPTH.get().1;
                                    TESTED_DEPTH.set((tested.0, tested.1 || depth));
                                    if kind == ResourceType::Srv && !crate::residency::take_staging_deferred() {
                                        ready = (list.epoch(), tested.0, depth);
                                    }
                                }
                            }
                            if let Some(title) = title_first
                                && texture
                                && tabled != Some((title + k, kind.is_some()))
                            {
                                tabled = Some((title + k, kind.is_some()));
                                host.texture_slot_tabled(title + k, kind.is_some());
                                if kind.is_some()
                                    && !Self::quick_slot_matches(host, title + k, words)
                                {
                                    host.mirror_after(list, title + k, table + u64::from(k) * 32);
                                }
                            }
                            if was != (ready, tabled) {
                                listed_state.insert(words, (ready, tabled));
                            }
                            continue;
                        }
                    }
                    let used = group.entry(k);
                    match crate::descriptor::view_from_words(words) {
                        Some(View::Constant(address, size)) => {
                            // Renderer port 5: title memory in place is the
                            // card's to read where it is; nothing is copied.
                            if crate::port_cbv::in_place_on() {
                                let mut done = false;
                                let found = crate::shadow::with_in_place(address, u64::from(size), |shadow| {
                                    let gpu = shadow.gpu(address);
                                    let base = shadow.gpu(shadow.span().0);
                                    let width = shadow.width();
                                    if host.constant_raw_at_known(index, &shadow.buffer, base, width, gpu, size, tb) {
                                        keep_shadow_dedup(kept_shadows, kept_shadows_next, list, shadow);
                                        crate::port_cbv::in_place(size);
                                        done = true;
                                    } else if size == 0 || crate::port_cbv::raw_ok(gpu) {
                                        crate::port_cbv::copied_oob_or_empty();
                                    } else {
                                        crate::port_cbv::copied(false);
                                    }
                                });
                                if done {
                                    continue;
                                }
                                if found.is_none() {
                                    crate::port_cbv::copied(true);
                                }
                            }
                            let Some((buffer, gpu, to)) = list.scratch_held(u64::from(size)) else {
                                host.make_null_batched(index, &mut tb.copy);
                                continue;
                            };
                            let base = scratch_va(scratch_base, &buffer);
                            let slot_id = (std::ptr::from_ref(heap) as usize, index);
                            list.defer_constants(to, address as *const u8, size as usize, slot_id);
                            host.constant_at_batched(index, &buffer, base, gpu, size, tb);
                        }
                        Some(view) => {
                            let shown = is_half.then(|| crate::descriptor::host::half_view(&view)).flatten();
                            match &shown {
                                Some(shown) => host.make_at(index, shown),
                                None => host.make_at(index, &view),
                            }
                            let kind = used.map(|u| u.kind).or_else(|| {
                                selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                            });
                            if let Some(kind) = kind {
                                readable_inner(list, kind, &view, uav_capture.as_deref_mut());
                            }
                            // The title's slot is a table's: out of the queue
                            // of textures picked by index if the draw read it
                            // (its texture is up to date now), and made in
                            // place as the texture pass would have made it,
                            // for a shader that picks it by index all the same.
                            if let Some(title) = title_first
                                && crate::descriptor::host::is_texture_view(&view)
                            {
                                host.texture_slot_tabled(title + k, kind.is_some());
                                if kind.is_some()
                                    && !Self::quick_slot_matches(host, title + k, words)
                                {
                                    host.mirror_after(list, title + k, table + u64::from(k) * 32);
                                }
                            }
                        }
                        None => {
                            host.make_null_batched(index, &mut tb.copy);
                            if used.is_some() {
                                once!(slot, "a table entry without a view");
                            }
                        }
                    }
                }
            });
            made.push((slot, first * 32));
        }
        self.table_records = records;
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && ready.is_none_or(|p| !p.rects);
        let ran = match ready {
            Some(p) if !rects || self.rectangles(list, p, vertices, first) => {
                // Rectangles were expanded into their own vertex buffers.
                let (count, from) = if rects {
                    (vertices / 3 * 6, 0)
                } else {
                    (vertices, first)
                };
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawInstanced(count, instances, from, first_instance)
                };
                self.drew(list, Some(p));
                self.local_drawn += 1;
                true
            }
            _ => false,
        };
        if !ran {
            let reason = if ready.is_some() && rects {
                DropReason::NoBuffer
            } else {
                self.last_drop_reason.unwrap_or(DropReason::NoPso)
            };
            record_drop(reason);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&mut self, list: &HostList, p: &HostPipeline, count: u32, first: u32) -> bool {
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host_read(from, n * 3 * stride, move || {
                // The vertices are read through the kernel, at submission; in
                // place only if that fails, with no change of mapping under
                // the read (see `crate::watch::title_memory_reading`): while
                // the world streams a change is under way most of the time,
                // and this is the thread that submits.
                let mut read = vec![0u8; n * 3 * stride];
                let held = (!crate::host::read_into(from as *const u8, &mut read)).then(crate::watch::title_memory_reading);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        if held.is_some() { std::slice::from_raw_parts(from as *const u8, n * 3 * stride) } else { &read[..] },
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
            self.host_vb_bound &= !(1u32 << slot);
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = if let Some(p) = (!lost).then(|| self.prepare(list, true)).flatten() {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            self.drew(list, Some(p));
            self.local_drawn += 1;
            true
        } else {
            false
        };
        if !ran {
            let reason = if lost {
                DropReason::LeftOut
            } else {
                self.last_drop_reason.unwrap_or(DropReason::NoPso)
            };
            record_drop(reason);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Shaders that can write buffers were recorded: arguments may be others
    /// from here on.
    fn buffers_written(&mut self, list: &HostList) {
        list.mark_uav_dirty();
        match &mut self.batch {
            Some(batch) if batches() == 2 => batch.wrote = true,
            _ => self.close_batch(),
        }
    }

    /// A draw with the pipeline `p` was recorded (as [`State::buffers_written`]
    /// if its shaders can store to buffers).
    fn drew(&mut self, list: &HostList, p: Option<&HostPipeline>) {
        if p.is_none_or(pipeline_stores_buffers) {
            list.mark_graphics_uav_dirty();
            self.close_batch();
        }
    }

    /// Updates this list's MRU of recent indirect argument copies (excluding
    /// `first`, which is moved to the front) into reusable `self.recent_args`
    /// with zero heap allocations (`CPU-29`).
    fn fill_argument_copies(&mut self, first: &std::sync::Arc<crate::shadow::Shadow>) {
        let mut live: [Option<std::sync::Arc<crate::shadow::Shadow>>; crate::indirect::batch::SOURCES] =
            std::array::from_fn(|_| None);
        let mut n = 0usize;
        let first_id = first.id();
        for weak in &self.argument_copies {
            if n >= crate::indirect::batch::SOURCES - 1 {
                break;
            }
            if let Some(s) = weak.upgrade()
                && s.id() != first_id
            {
                live[n] = Some(s);
                n += 1;
            }
        }
        self.argument_copies.clear();
        self.argument_copies.push(std::sync::Arc::downgrade(first));
        self.recent_args.clear();
        for s in live.into_iter().take(n).flatten() {
            self.argument_copies.push(std::sync::Arc::downgrade(&s));
            self.recent_args.push((s.id(), s.buffer.clone(), s.in_place()));
        }
    }

    /// A buffer was copied to: what was clamped of it ahead is clamped anew.
    pub(crate) fn copied(&mut self) {
        self.window = None;
        self.window_list = 0;
        self.close_batch();
    }

    /// The title waits for earlier work (a barrier or a pipeline sync).
    pub(crate) fn synced(&mut self) {
        if self.batch.as_ref().is_some_and(|b| b.wrote) {
            self.close_batch();
        }
    }

    /// Where the clamped record of `kind` at `arguments` of the copy `args`
    /// is, from the open batch or a new one ([`batches`]). `None` if it
    /// cannot be clamped in one.
    fn batched(
        &mut self,
        list: &HostList,
        args: &std::sync::Arc<crate::shadow::Shadow>,
        arguments: u64,
        kind: u32,
    ) -> Option<(ID3D12Resource, u64)> {
        const DISPATCH: u32 = 2;
        let list_ptr = std::ptr::from_ref(list) as usize;
        // Another host list, or this one after its reset/cut (`CPU-29`:
        // checked per-`State` without `list.first_use` `deferred.lock()`).
        if self.batch_list != list_ptr {
            self.close_batch();
        }
        // A draw after dispatches waits for them.
        if !matches!(kind, DISPATCH | crate::indirect::KIND_DISPATCH_SPREAD) {
            self.synced();
        }
        let (id, offset) = (args.id(), args.offset(arguments));
        if let (Some(b), Some(table)) = (&self.batch, self.batch_table.as_deref_mut())
            && let Some(at) = table.place_deferred(id, offset, kind)
        {
            return Some((b.scratch.clone(), at));
        }
        self.close_batch();
        self.fill_argument_copies(args);
        let table = self
            .batch_table
            .get_or_insert_with(|| Box::new(crate::indirect::batch::Table::default()));
        let scratch = crate::indirect::open_batch_into(
            list,
            (id, &args.buffer, args.in_place()),
            &self.recent_args,
            table,
        )?;
        let at = table.place_deferred(id, offset, kind)?;
        self.invalidate_compute_host_state();
        self.batch = Some(Batch { scratch: scratch.clone(), wrote: false });
        self.batch_list = list_ptr;
        Some((scratch, at))
    }

    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        self.last_drop_reason = None;
        const DRAW: u32 = crate::indirect::KIND_DRAW;
        const INDEXED: u32 = crate::indirect::KIND_DRAW_INDEXED;
        const DISPATCH: u32 = crate::indirect::KIND_DISPATCH;
        const MESH: u32 = crate::indirect::KIND_DISPATCH_MESH;
        const MESH_1D: u32 = crate::indirect::KIND_DISPATCH_MESH_1D;
        let is_compute = kind == DISPATCH;
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            self.stamp_dropped_tables(is_compute, None);
            self.last_drop_reason = Some(DropReason::LeftOut);
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            self.stamp_dropped_tables(is_compute, None);
            self.last_drop_reason = Some(DropReason::LeftOut);
            return false;
        }
        let is_mesh = matches!(kind, MESH | MESH_1D);
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            self.stamp_dropped_tables(is_compute, None);
            self.last_drop_reason = Some(DropReason::NoPso);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH | MESH_1D => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            self.stamp_dropped_tables(p.compute, Some(p));
            self.last_drop_reason = Some(DropReason::NoPso);
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            self.stamp_dropped_tables(p.compute, Some(p));
            self.last_drop_reason = Some(DropReason::NoBuffer);
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    self.stamp_dropped_tables(p.compute, Some(p));
                    self.last_drop_reason = Some(DropReason::NoBuffer);
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        // A dispatch of a program that takes the spread words: the clamp
        // writes [row, asked x, x, y, z] records, run through a signature
        // that sets the two words, then dispatches.
        let spread = (kind == DISPATCH && p.compute)
            .then(|| spreads(p))
            .flatten()
            .and_then(|(param, index)| host_spread_signature(&p.root, param, index))
            .map(SigRef::Owned);
        let is_spread = spread.is_some();
        let Some(signature) = spread.or_else(|| {
            if kind == MESH_1D {
                host_signature(MESH, 12)
            } else {
                matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
                    .then(|| host_signature(kind, stride))
                    .flatten()
            }
        }) else {
            self.stamp_dropped_tables(p.compute, Some(p));
            self.last_drop_reason = Some(DropReason::LeftOut);
            why("no host signature");
            return false;
        };
        if is_mesh && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            self.stamp_dropped_tables(p.compute, Some(p));
            self.last_drop_reason = Some(DropReason::LeftOut);
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // Renderer port 1: arguments (or their count) in the title's memory
        // read in place ([`in_place_order`]): clamped alone, right before
        // their execution, after a wait for every earlier shader write.
        let in_place = args.in_place() || counter.as_ref().is_some_and(|(s, _)| s.in_place());
        if in_place {
            in_place_indirect(arguments, kind);
        }
        let ordered = in_place && in_place_order();
        let alone = ordered && in_place_alone();
        if ordered {
            if alone || self.batch.as_ref().is_some_and(|b| b.wrote) {
                self.close_batch();
            }
            self.window = None;
            self.window_list = 0;
        }
        let offset = args.offset(arguments);
        let count_off = counter.as_ref().map(|(s, at)| s.offset(*at));
        let (bounded, at) = if !crate::indirect::clamp_enabled()
            && !is_spread
            && crate::indirect::can_bypass_clamp(kind, stride, offset, count_off)
        {
            // shortcut: VICTORY_INDIRECT_CLAMP=0 executes directly from the
            // title buffer's host copy without clamping; unfilled garbage slots
            // can TDR the GPU.
            args.refresh(list, arguments, len);
            if let Some((s, at)) = &counter {
                s.refresh(list, *at, 4);
            }
            self.window = None;
            self.window_list = 0;
            // Ensure earlier compute/graphics UAV writes to argument or count
            // buffers are visible to ExecuteIndirect (scoped Enhanced Barrier
            // with PFP sync when supported, else scoped UAV barrier).
            crate::indirect::sync_bypass_indirect(
                list,
                &args.buffer,
                counter.as_ref().map(|(s, _)| &s.buffer),
            );
            (args.buffer.clone(), offset)
        } else {
            // One record without a count: clamped in a batch (`batches`),
            // including 20-byte spread dispatches (fits inside 32-byte slot).
            let batch_kind = if is_spread {
                crate::indirect::KIND_DISPATCH_SPREAD
            } else {
                kind
            };
            let batch = if !alone && batches() > 0 && max == 1 && count.is_none() {
                args.refresh(list, arguments, len);
                self.batched(list, &args, arguments, batch_kind)
            } else {
                None
            };
            if batch.is_some() {
                self.window = None;
                self.window_list = 0;
            }
            // A window of records clamped in one pass (`windows`): for single
            // records without a count buffer, drawn by shaders that store to no
            // buffer. Anything else ends the window: it uses the scratch buffer
            // or may write the arguments.
            let list_ptr = std::ptr::from_ref(list) as usize;
            let windowed = windows()
                && !ordered
                && kind != DISPATCH
                && kind != MESH_1D
                && max == 1
                && count.is_none()
                && !pipeline_stores_buffers(p);
            // Another host list (a new one after a cut, or this one after its
            // reset) has another scratch buffer, or this one's with old bytes.
            if !windowed || self.window_list != list_ptr {
                self.window = None;
                self.window_list = 0;
            }
            let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
            let hit = self
                .window
                .as_ref()
                .filter(|w| same(w) && offset >= w.base)
                .filter(|w| {
                    (offset - w.base) % u64::from(w.step) == 0
                        && (offset - w.base) / u64::from(w.step) < u64::from(w.records)
                })
                .map(|w| (w.scratch.clone(), offset - w.base));
            match batch.or(hit) {
                Some(hit) => hit,
                None => {
                    // A record a short step after the last window's last one
                    // (the same step as that window's, if it had one): the
                    // title walks the buffer, so the ones that step apart
                    // after it are clamped with it.
                    let (step, records) = match &self.window {
                        Some(w) if windowed && same(w) => {
                            let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                            let step = offset.saturating_sub(last);
                            let walks = step >= u64::from(stride)
                                && step <= 256
                                && step % 4 == 0
                                && (w.records == 1 || step == u64::from(w.step));
                            if walks {
                                let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                                (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                            } else {
                                (stride, max)
                            }
                        }
                        _ => (stride, max),
                    };
                    args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                    if let Some((s, at)) = &counter {
                        s.refresh(list, *at, 4);
                    }
                    // Bounded first: the clamp binds a compute pipeline of its
                    // own and copies clamped active records into a scratch
                    // buffer at offset 0.
                    let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                    // More than the record asked for: a window clamped ahead,
                    // whose unfilled slots must not spend the list's budget.
                    let bounded = if windowed && records > 1 {
                        crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records), in_place)
                    } else {
                        // (A spread dispatch is never windowed: `windowed` is
                        // false for dispatches.)
                        crate::indirect::bound(
                            list,
                            (&args.buffer, offset),
                            count_arg,
                            (kind, step, records),
                            is_spread,
                            in_place,
                        )
                    };
                    let Some(bounded) = bounded else {
                        self.window = None;
                        self.window_list = 0;
                        why("not bounded (no clamp, or arguments not 4-byte aligned)");
                        return false;
                    };
                    self.invalidate_compute_host_state();
                    self.window = windowed.then(|| Window {
                        copy: args.id(),
                        kind,
                        stride,
                        step,
                        base: offset,
                        records,
                        scratch: bounded.clone(),
                    });
                    self.window_list = if windowed { list_ptr } else { 0 };
                    (bounded, 0)
                }
            }
        };
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED).is_some(),
            MESH | MESH_1D => self.prepare_for(list, false, true).is_some(),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            self.stamp_dropped_tables(p.compute, Some(p));
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        #[cfg(feature = "diagnostics")]
        list.ran_indirect(p.pso.as_raw() as usize, &p.name, kind, max, &self.user);
        if is_mesh {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &*signature,
                max,
                &bounded,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if is_mesh {
            self.mesh_scissor(list, false);
        }
        if kind == DISPATCH {
            self.buffers_written(list);
        } else {
            self.drew(list, Some(p));
        }
        // The scratch buffer stays readable as arguments: the next draw of
        // the window reads it with no barrier, and the next clamp moves it
        // back (`crate::indirect`).
        if kind == DISPATCH {
            self.local_dispatched += 1;
            if is_spread {
                // `host_spread_signature` mutates compute root constants on GPU.
                self.host_root_args[1].clear();
                self.local_spread[1] += 1;
            }
            if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
                && crate::ambient::on()
                && self.ambient(list, p, role)
            {
                self.invalidate_compute_host_state();
            }
        } else {
            self.local_drawn += 1;
        }
        true
    }

    /// Indirect execution of a general console command signature (`stride`,
    /// `args`): single-argument signatures delegate straight to
    /// [`State::execute_indirect`]; multi-argument signatures with any prefix
    /// of root constants (`CONSTANT = 5`) and root descriptors (`CBV = 6`,
    /// `SRV = 7`, `UAV = 8`) followed by `DRAW` / `DRAW_INDEXED` / `DISPATCH` /
    /// `DISPATCH_SINGLE` / `DISPATCH_MESH` are repacked, VA-rebased, and
    /// clamped on the GPU before executing through a cached host command
    /// signature.
    pub(crate) fn execute_indirect_signature(
        &mut self,
        list: &HostList,
        stride: u32,
        args: &[[u32; 4]],
        max: u32,
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        use crate::indirect::{
            KIND_DISPATCH, KIND_DISPATCH_MESH, KIND_DISPATCH_SINGLE, KIND_DRAW, KIND_DRAW_INDEXED,
        };
        match args {
            [] => return false,
            [[kind, ..]] => {
                let kind = if *kind == KIND_DISPATCH_SINGLE {
                    KIND_DISPATCH
                } else {
                    *kind
                };
                return self.execute_indirect(list, (kind, stride, max), arguments, count);
            }
            _ => {}
        }
        let is_compute_hint = args.last().is_some_and(|a| matches!(a[0], KIND_DISPATCH | KIND_DISPATCH_SINGLE));
        self.last_drop_reason = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            self.stamp_dropped_tables(is_compute_hint, None);
            self.last_drop_reason = Some(DropReason::LeftOut);
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            self.stamp_dropped_tables(is_compute_hint, None);
            self.last_drop_reason = Some(DropReason::LeftOut);
            return false;
        }
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("multi-arg indirect {args:?}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            self.stamp_dropped_tables(is_compute_hint, None);
            self.last_drop_reason = Some(DropReason::NoPso);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        let Some(cached) = cached_multi_indirect(
            p,
            stride,
            args,
            crate::pipeline::host::bounded_root_srvs(),
        ) else {
            why("unsupported multi-argument signature layout or no host signature");
            return false;
        };
        let plan = &cached.plan;
        let signature = &cached.signature;
        let mutates_root_state = cached.mutates_root_state;
        if plan.skipped_tabled_views {
            once!(
                "indirect root SRV/UAV targets a bounded descriptor-table parameter; set VICTORY_ROOT_SRV_DIRECT=1 to enable indirect root SRV/UAV updates"
            );
        }
        let kind = plan.exec_kind;
        // Fast path: when a multi-argument console signature performs no root-state
        // mutations for this pipeline (`!mutates_root_state && plan.ops.is_empty()`),
        // delegate straight to `execute_indirect` so batched/windowed clamping applies
        // and we skip both `repack` and `invalidate_compute_host_state()`.
        if !mutates_root_state && plan.ops.is_empty() {
            if plan.src_exec_off == 0 {
                return self.execute_indirect(list, (kind, stride, max), arguments, count);
            }
            if max == 1 && count.is_none() {
                let exec_stride = match kind {
                    KIND_DRAW => 16,
                    KIND_DRAW_INDEXED => 20,
                    KIND_DISPATCH | KIND_DISPATCH_MESH => 12,
                    _ => stride,
                };
                if let Some(exec_addr) = arguments.checked_add(u64::from(plan.src_exec_off)) {
                    return self.execute_indirect(list, (kind, exec_stride, 1), exec_addr, None);
                }
            }
        }
        let made_rects = !p.rects
            && kind == KIND_DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            KIND_DRAW | KIND_DRAW_INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            KIND_DISPATCH_MESH => !p.compute && p.mesh,
            KIND_DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        if kind == KIND_DISPATCH_MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        let Some(len) = u64::from(stride).checked_mul(u64::from(max.max(1))) else {
            why("argument byte length overflow");
            return false;
        };
        let Some(arg_shadow) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        arg_shadow.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        if plan.needs_rebase {
            let roots = if p.compute { &self.compute } else { &self.graphics };
            collect_rebase_entries_into(
                list,
                roots,
                &mut self.indirect_rebase_ranges,
                &mut self.indirect_rebase_seen,
                &mut self.indirect_rebase,
            );
        } else {
            self.indirect_rebase.clear();
        }
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        let any_in_place = arg_shadow.in_place() || counter.as_ref().is_some_and(|(s, _)| s.in_place());
        let Some(bounded) = crate::indirect::repack(
            list,
            (&arg_shadow.buffer, arg_shadow.offset(arguments)),
            count_arg,
            plan,
            &self.indirect_rebase,
            max,
            any_in_place,
        ) else {
            self.window = None;
            why("multi-arg repack failed");
            return false;
        };
        // `repack` only binds its compute root signature/PSO when `max > 0`.
        if max > 0 {
            self.invalidate_compute_host_state();
        }
        self.window = None;
        let ran = match kind {
            KIND_DRAW | KIND_DRAW_INDEXED => self.prepare(list, kind == KIND_DRAW_INDEXED).is_some(),
            KIND_DISPATCH_MESH => self.prepare_for(list, false, true).is_some(),
            KIND_DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            self.stamp_dropped_tables(p.compute, Some(p));
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect multi kind {kind} max {max} at {arguments:#x}"),
                kind == KIND_DISPATCH,
            )
        });
        #[cfg(feature = "diagnostics")]
        list.ran_indirect(p.pso.as_raw() as usize, &p.name, kind, max, &self.user);
        if kind == KIND_DISPATCH_MESH {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; `bounded`, `signature`, and
        // any rebased shadow buffers are kept alive for the list.
        unsafe {
            list.list.ExecuteIndirect(
                signature,
                max,
                &bounded,
                0,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if mutates_root_state {
            // Multi-arg indirect signature modifies root constants/views on GPU.
            self.host_root_args[usize::from(p.compute)].clear();
        }
        if kind == KIND_DISPATCH_MESH {
            self.mesh_scissor(list, false);
        }
        if kind == KIND_DISPATCH {
            self.buffers_written(list);
            self.local_dispatched += 1;
            if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
                && crate::ambient::on()
                && self.ambient(list, p, role)
            {
                self.invalidate_compute_host_state();
            }
        } else {
            self.drew(list, Some(p));
            self.local_drawn += 1;
        }
        true
    }
}

/// `DispatchX`: the counts are threads (the console's dispatch initiator bit
/// "use thread dimensions"; DexNom/VIctory bc3cd4b0). Taken for groups, a
/// rehydration stage of the structure builder ran up to 64 times its threads
/// and stored over the structures after its own (runs light-1 to light-9).
const THREAD_DIMENSIONS: u32 = 0x20;

/// `VICTORY_THREAD_DIMENSIONS=0`: such counts are taken for groups, as before.
fn thread_dimensions() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_THREAD_DIMENSIONS").is_none_or(|v| v != "0"))
}

/// The groups of `size` threads that hold `threads`.
#[cfg(test)]
fn dispatch_groups(threads: [u32; 3], size: [u32; 3]) -> [u32; 3] {
    std::array::from_fn(|i| threads[i].div_ceil(size[i]))
}

/// A console dispatch `(x, y, z)` with `DispatchX` flags `flags`, as host
/// groups and the thread counts the program is to keep to: per axis the
/// count where it is not a whole number of `group`-thread groups, else 0
/// (`USER_DISPATCH_THREADS`). Without [`THREAD_DIMENSIONS`] the counts are
/// groups and go as they are. `None`: no thread along some axis, so nothing
/// to run. (Fork pull request #646, mynzjerko.)
fn dispatch_dimensions(dims: [u32; 3], group: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & THREAD_DIMENSIONS == 0 {
        return Some((dims, [0; 3]));
    }
    if dims.contains(&0) {
        return None;
    }
    let groups = std::array::from_fn(|c| dims[c].div_ceil(group[c].max(1)));
    let partial = std::array::from_fn(|c| {
        let g = group[c].max(1);
        if g > 1 && !dims[c].is_multiple_of(g) { dims[c] } else { 0 }
    });
    Some((groups, partial))
}

/// Where a compute pipeline's user data has a word the host computes: the
/// root parameter and the dword in it.
fn compute_word(p: &HostPipeline, kind: u32) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let word = stage.translated().host.user_data.iter().find(|u| u.root == RootType::Other(kind))?;
    Some((param, word.index))
}

#[cfg(test)]
mod thread_tests {
    use super::*;

    #[test]
    fn threads_are_counted_in_the_programs_groups() {
        assert_eq!(dispatch_groups([8847, 1, 1], [64, 1, 1]), [139, 1, 1]);
        assert_eq!(dispatch_groups([74, 1, 1], [64, 1, 1]), [2, 1, 1]);
        assert_eq!(dispatch_groups([32, 1, 1], [32, 1, 1]), [1, 1, 1]);
        assert_eq!(dispatch_groups([0, 12, 9], [64, 4, 4]), [0, 3, 3]);
    }

    /// Fork pull request #646: a `DispatchX` with bit 5 counts threads.
    #[test]
    fn a_dispatch_counting_threads_runs_the_groups_that_hold_them() {
        // Without bit 5 (plain Dispatch, or only the 32-lane bit 0x8000): groups.
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0), Some(([48, 1, 1], [0; 3])));
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0x8000), Some(([48, 1, 1], [0; 3])));
        // Threads: the groups that hold them, and the count where the last is cut.
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0x20), Some(([1, 1, 1], [48, 0, 0])));
        assert_eq!(dispatch_dimensions([1, 1, 1], [64, 1, 1], 0x8020), Some(([1, 1, 1], [1, 0, 0])));
        assert_eq!(dispatch_dimensions([65, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], [65, 0, 0])));
        // The builder's box stage: 5,348 nodes of one thread each in groups of 64.
        assert_eq!(dispatch_dimensions([5348, 1, 1], [64, 1, 1], 0x20), Some(([84, 1, 1], [5348, 0, 0])));
        // A whole number of groups needs no count.
        assert_eq!(dispatch_dimensions([128, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], [0; 3])));
        assert_eq!(dispatch_dimensions([10, 16, 5], [8, 8, 4], 0x20), Some(([2, 2, 2], [10, 0, 5])));
        // No thread along an axis: nothing runs.
        assert_eq!(dispatch_dimensions([0, 1, 1], [64, 1, 1], 0x20), None);
    }
}

/// Dispatches spread across z since the frame line last took them: direct,
/// indirect ([`victory_shader::emit::USER_DISPATCH_SPREAD`]).
pub(crate) static SPREAD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The user-data root parameter of compute pipeline `p` whose program takes
/// the spread words, and the index of the first in it (the second follows:
/// one root-constant argument sets both); `None` for any other pipeline.
#[inline]
fn spreads(p: &HostPipeline) -> Option<(u32, u32)> {
    spread_on().then_some(p.spread).flatten()
}

/// Whether dispatches past 65,535 groups in x are spread across z
/// (`VICTORY_DISPATCH_SPREAD`: on by default; `0` off, clamped to 65,535;
/// `alt`: twenty seconds with, twenty without, in turn, each change logged).
/// Off, a translated program's words stay 0: it runs as dispatched.
fn spread_on() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_DISPATCH_SPREAD").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        _ => 1,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "dispatches past 65,535 groups are spread across z from now (VICTORY_DISPATCH_SPREAD=alt): {}",
                    if on { "ON" } else { "off, clamped to 65,535 as before" }
                );
            }
            on
        }
    }
}

/// Free-threaded wrapper around an immutable `ID3D12CommandSignature` for
/// lock-free `OnceLock` caching (`CPU-29`).
struct SendSig(ID3D12CommandSignature);
// SAFETY: D3D12 command signatures are immutable free-threaded COM objects.
unsafe impl Send for SendSig {}
// SAFETY: as above.
unsafe impl Sync for SendSig {}

/// Borrowed static or owned command signature handle avoiding COM AddRef/Release
/// on the 99%+ standard indirect signature fast path (`CPU-29`).
enum SigRef {
    Static(&'static ID3D12CommandSignature),
    Owned(ID3D12CommandSignature),
}

impl std::ops::Deref for SigRef {
    type Target = ID3D12CommandSignature;
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            SigRef::Static(s) => s,
            SigRef::Owned(s) => s,
        }
    }
}

/// Maps the four standard single-argument `(kind, stride)` signatures to
/// lock-free `OnceLock` slots (`CPU-29`).
#[inline]
fn sig_fast_slot(kind: u32, stride: u32) -> Option<usize> {
    match (kind, stride) {
        (crate::indirect::KIND_DRAW, 16) => Some(0),
        (crate::indirect::KIND_DRAW_INDEXED, 20) => Some(1),
        (crate::indirect::KIND_DISPATCH, 12) => Some(2),
        (crate::indirect::KIND_DISPATCH_MESH, 12) => Some(3),
        _ => None,
    }
}

/// The command signature of a spread dispatch: two root constants (row,
/// asked x) at `index` of root parameter `param`, then the dispatch; 20
/// bytes a record (the clamp's spread layout). Made once per root
/// signature, parameter and index, cached in TLS + shared `RwLock` (`CPU-29`).
fn host_spread_signature(
    root: &ID3D12RootSignature,
    param: u32,
    index: u32,
) -> Option<ID3D12CommandSignature> {
    type Key = (usize, u32, u32);
    static MADE: std::sync::RwLock<Vec<(Key, ID3D12CommandSignature)>> =
        std::sync::RwLock::new(Vec::new());
    thread_local! {
        static TLS_SPREAD: std::cell::RefCell<Vec<(Key, ID3D12CommandSignature)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let key = (root.as_raw() as usize, param, index);
    if let Some(sig) = TLS_SPREAD.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, s)| s.clone())
    }) {
        return Some(sig);
    }
    if let Some(sig) = MADE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, s)| s.clone())
    {
        TLS_SPREAD.with(|c| c.borrow_mut().push((key, sig.clone())));
        return Some(sig);
    }
    let mut made = MADE.write().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == key) {
        let sig = s.clone();
        TLS_SPREAD.with(|c| c.borrow_mut().push((key, sig.clone())));
        return Some(sig);
    }
    let args = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: param,
                    DestOffsetIn32BitValues: index,
                    Num32BitValuesToSet: 2,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 20,
        NumArgumentDescs: args.len() as u32,
        pArgumentDescs: args.as_ptr(),
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: two valid argument descriptions and a live root signature.
    let result = unsafe {
        crate::host::host()?
            .device
            .CreateCommandSignature(&desc, Some(root), &mut signature)
    };
    if let Err(e) = result {
        once!(error = %e, param, index, "spread command signature refused");
        return None;
    }
    let signature = signature?;
    made.push((key, signature.clone()));
    TLS_SPREAD.with(|c| c.borrow_mut().push((key, signature.clone())));
    Some(signature)
}

fn create_single_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    signature
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once. Lock-free `OnceLock` fast path for standard strides (`CPU-29`),
/// eliminating global mutex contention and COM `AddRef`/`Release` churn.
fn host_signature(kind: u32, stride: u32) -> Option<SigRef> {
    if let Some(slot) = sig_fast_slot(kind, stride) {
        static FAST: [std::sync::OnceLock<Option<SendSig>>; 4] =
            [const { std::sync::OnceLock::new() }; 4];
        return FAST[slot]
            .get_or_init(|| create_single_signature(kind, stride).map(SendSig))
            .as_ref()
            .map(|s| SigRef::Static(&s.0));
    }
    static MADE: std::sync::RwLock<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::RwLock::new(Vec::new());
    thread_local! {
        static TLS_OTHER: std::cell::RefCell<Vec<((u32, u32), ID3D12CommandSignature)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    if let Some(sig) = TLS_OTHER.with(|c| {
        c.borrow()
            .iter()
            .find(|(k, _)| *k == (kind, stride))
            .map(|(_, s)| s.clone())
    }) {
        return Some(SigRef::Owned(sig));
    }
    if let Some(sig) = MADE
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .find(|(k, _)| *k == (kind, stride))
        .map(|(_, s)| s.clone())
    {
        TLS_OTHER.with(|c| c.borrow_mut().push(((kind, stride), sig.clone())));
        return Some(SigRef::Owned(sig));
    }
    let mut made = MADE.write().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        let sig = s.clone();
        TLS_OTHER.with(|c| c.borrow_mut().push(((kind, stride), sig.clone())));
        return Some(SigRef::Owned(sig));
    }
    let signature = create_single_signature(kind, stride)?;
    made.push(((kind, stride), signature.clone()));
    TLS_OTHER.with(|c| c.borrow_mut().push(((kind, stride), signature.clone())));
    Some(SigRef::Owned(signature))
}

/// Collects active live shadow ranges into a sorted, non-overlapping GPU
/// VA-rebase table for indirect root CBV/SRV/UAV translation, keeping every
/// referenced shadow buffer alive on `list` and reusing caller scratch `Vec`s.
fn collect_rebase_entries_into(
    list: &HostList,
    roots: &[Root],
    ranges: &mut Vec<(u64, u64, u64, bool)>,
    seen_ids: &mut Vec<u64>,
    out: &mut Vec<[u32; 8]>,
) {
    const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    ranges.clear();
    seen_ids.clear();
    let mut push_shadow = |s: std::sync::Arc<crate::shadow::Shadow>| {
        let id = s.id();
        if seen_ids.contains(&id) {
            return;
        }
        seen_ids.push(id);
        let (start, len) = s.span();
        if len == 0 {
            return;
        }
        let host_gpu = s.gpu(start);
        // SAFETY: live D3D12 buffer resource.
        let writable = unsafe { s.buffer.GetDesc() }.Flags.0 & UNORDERED != 0;
        list.keep_shadow(s);
        ranges.push((start, len, host_gpu, writable));
    };
    for (start, len, _) in crate::shadow::spans() {
        if len > 0
            && let Some(s) = crate::shadow::peek(start, 1)
        {
            push_shadow(s);
        }
    }
    for root in roots {
        if let &Root::View(va) = root
            && va != 0
            && let Some(s) = crate::shadow::find(va, 1)
        {
            push_shadow(s);
        }
    }
    crate::indirect::build_rebase_table_into(ranges, out);
}

/// Memoized multi-argument indirect execution entry for a `(stride, args,
/// host root signature, stage shaders, bounded_root_srvs)` tuple.
///
/// Retains `_root` so `root_raw` cannot be recycled by COM while cached;
/// `shader_ptrs` point to permanently interned `Arc<Shader>`s so they can
/// never be recycled; `stride` and `args` are verified by value on every hit
/// so guest signature pointer recycling can never alias.
struct CachedMultiIndirect {
    stride: u32,
    args: Vec<[u32; 4]>,
    root_raw: usize,
    shader_ptrs: [usize; 5],
    bounded_root_srvs: bool,
    _root: Option<ID3D12RootSignature>,
    plan: crate::indirect::MultiArgPlan,
    signature: ID3D12CommandSignature,
    mutates_root_state: bool,
}

// SAFETY: `ID3D12RootSignature` and `ID3D12CommandSignature` are free-threaded
// immutable D3D12 device objects safe to share across threads behind `Arc`.
unsafe impl Send for CachedMultiIndirect {}
// SAFETY: as above.
unsafe impl Sync for CachedMultiIndirect {}

impl CachedMultiIndirect {
    #[inline]
    fn matches(
        &self,
        stride: u32,
        args: &[[u32; 4]],
        root_raw: usize,
        shader_ptrs: &[usize; 5],
        bounded_root_srvs: bool,
    ) -> bool {
        self.stride == stride
            && self.root_raw == root_raw
            && self.bounded_root_srvs == bounded_root_srvs
            && self.shader_ptrs == *shader_ptrs
            && self.args.as_slice() == args
    }
}

#[inline]
fn multi_indirect_key_hash(
    stride: u32,
    args: &[[u32; 4]],
    root_raw: usize,
    shader_ptrs: &[usize; 5],
    bounded_root_srvs: bool,
) -> u64 {
    let mix = |h: u64, x: u64| (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    let mut h = mix(u64::from(stride), root_raw as u64);
    h = mix(h, u64::from(bounded_root_srvs) | ((args.len() as u64) << 1));
    for &p in shader_ptrs {
        if p != 0 {
            h = mix(h, p as u64);
        }
    }
    for &[a, b, c, d] in args {
        h = mix(h, u64::from(a) | (u64::from(b) << 32));
        h = mix(h, u64::from(c) | (u64::from(d) << 32));
    }
    h ^ (h >> 32)
}

/// Lock-free thread-local MRU + shared `RwLock` memoization cache for
/// `(signature layout, pipeline)` → `Arc<CachedMultiIndirect>`.
fn cached_multi_indirect(
    p: &HostPipeline,
    stride: u32,
    args: &[[u32; 4]],
    bounded_root_srvs: bool,
) -> Option<std::sync::Arc<CachedMultiIndirect>> {
    use crate::indirect::StageBinding;
    use std::sync::{Arc, LazyLock, RwLock};

    let root_raw = p.root.as_raw() as usize;
    let mut shader_ptrs = [0usize; 5];
    for (dst, stage) in shader_ptrs.iter_mut().zip(&p.stages) {
        *dst = Arc::as_ptr(&stage.shader) as usize;
    }
    let key_hash = multi_indirect_key_hash(stride, args, root_raw, &shader_ptrs, bounded_root_srvs);

    // 1. Lock-free thread-local MRU cache (zero mutex, zero allocation).
    thread_local! {
        static TLS_CACHE: std::cell::RefCell<Vec<(u64, Arc<CachedMultiIndirect>)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    const TLS_CAP: usize = 32;
    if let Some(hit) = TLS_CACHE.with(|cell| {
        let cache = cell.borrow();
        cache
            .iter()
            .find(|(h, e)| *h == key_hash && e.matches(stride, args, root_raw, &shader_ptrs, bounded_root_srvs))
            .map(|(_, e)| e.clone())
    }) {
        return Some(hit);
    }

    static SHARED: LazyLock<
        RwLock<std::collections::HashMap<u64, Vec<Arc<CachedMultiIndirect>>, std::hash::BuildHasherDefault<Plain>>>,
    > = LazyLock::new(|| RwLock::new(std::collections::HashMap::default()));

    let tls_insert = |entry: &Arc<CachedMultiIndirect>| {
        TLS_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            if cache.len() >= TLS_CAP {
                cache.remove(0);
            }
            cache.push((key_hash, entry.clone()));
        });
    };

    // 2. Shared concurrent read-lock lookup across recording threads.
    if let Some(hit) = SHARED
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .get(&key_hash)
        .and_then(|bucket| {
            bucket
                .iter()
                .find(|e| e.matches(stride, args, root_raw, &shader_ptrs, bounded_root_srvs))
                .cloned()
        })
    {
        tls_insert(&hit);
        return Some(hit);
    }

    // 3. Cold miss: translate signature once per unique `(signature layout, pipeline)`.
    let mut bindings: Vec<StageBinding<'_>> = Vec::with_capacity(p.stages.len());
    for stage in &p.stages {
        bindings.push(StageBinding {
            user_data_param: stage.user_data.map(|(param, _)| param),
            user_data: &stage.translated().host.user_data,
            root_cbvs: &stage.root_cbvs,
            root_srvs: &stage.root_srvs,
            root_uavs: &stage.root_uavs,
        });
    }
    let plan = crate::indirect::translate_signature(stride, args, &bindings, bounded_root_srvs)?;
    let mutates_root_state = plan.has_root_args;
    let signature = host_multi_signature(
        plan.has_root_args.then_some(&p.root),
        plan.dst_stride,
        &plan.host_args,
    )?;
    let entry = Arc::new(CachedMultiIndirect {
        stride,
        args: args.to_vec(),
        root_raw,
        shader_ptrs,
        bounded_root_srvs,
        _root: Some(p.root.clone()),
        plan,
        signature,
        mutates_root_state,
    });
    // Only cache pipelines with <= 5 stages (all D3D12 graphics/compute pipelines).
    if p.stages.len() <= 5 {
        let mut map = SHARED.write().unwrap_or_else(|e| e.into_inner());
        let bucket = map.entry(key_hash).or_default();
        if let Some(existing) = bucket
            .iter()
            .find(|e| e.matches(stride, args, root_raw, &shader_ptrs, bounded_root_srvs))
        {
            let existing = existing.clone();
            drop(map);
            tls_insert(&existing);
            return Some(existing);
        }
        bucket.push(entry.clone());
        drop(map);
        tls_insert(&entry);
    }
    Some(entry)
}

/// Cached host multi-argument command signature keyed by
/// `(root_sig_ptr, host_stride, host_args_layout)` using `RwLock` (`CPU-29`).
fn host_multi_signature(
    root: Option<&ID3D12RootSignature>,
    stride: u32,
    args: &[crate::indirect::HostIndirectArg],
) -> Option<ID3D12CommandSignature> {
    use crate::indirect::HostIndirectArg;
    type Key = (usize, u32, Vec<HostIndirectArg>);
    type Entry = (Option<ID3D12RootSignature>, ID3D12CommandSignature);
    static MADE: std::sync::LazyLock<
        std::sync::RwLock<std::collections::HashMap<Key, Entry>>,
    > = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));
    let root_ptr = root.map_or(0, |r| r.as_raw() as usize);
    let key = (root_ptr, stride, args.to_vec());
    if let Some((_, s)) = MADE.read().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Some(s.clone());
    }
    let mut made = MADE.write().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.get(&key) {
        return Some(s.clone());
    }
    let descs: Vec<D3D12_INDIRECT_ARGUMENT_DESC> = args
        .iter()
        .map(|&arg| match arg {
            HostIndirectArg::Constant {
                param,
                dest_offset,
                count,
            } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                        RootParameterIndex: param,
                        DestOffsetIn32BitValues: dest_offset,
                        Num32BitValuesToSet: count,
                    },
                },
            },
            HostIndirectArg::Cbv { param } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    ConstantBufferView: D3D12_INDIRECT_ARGUMENT_DESC_0_2 {
                        RootParameterIndex: param,
                    },
                },
            },
            HostIndirectArg::Srv { param } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    ShaderResourceView: D3D12_INDIRECT_ARGUMENT_DESC_0_3 {
                        RootParameterIndex: param,
                    },
                },
            },
            HostIndirectArg::Uav { param } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    UnorderedAccessView: D3D12_INDIRECT_ARGUMENT_DESC_0_4 {
                        RootParameterIndex: param,
                    },
                },
            },
            HostIndirectArg::Exec { kind } => D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            },
        })
        .collect();
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: descs.len() as u32,
        pArgumentDescs: descs.as_ptr(),
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: valid argument descriptions and matching optional root signature.
    let result = unsafe {
        crate::host::host()?
            .device
            .CreateCommandSignature(&desc, root, &mut signature)
    };
    if let Err(e) = result {
        once!(error = %e, stride, ?args, "multi-argument command signature refused");
        return None;
    }
    let signature = signature?;
    made.insert(key, (root.cloned(), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
pub(crate) fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

/// A few things, up to `N` of them without memory of the heap: what a draw's
/// tables are gone through with, some six lists a stage.
#[derive(Debug, Clone)]
pub(crate) struct Few<T, const N: usize> {
    few: [T; N],
    len: usize,
    /// All of them, if they are more.
    many: Vec<T>,
}

impl<T: Copy + Default, const N: usize> Default for Few<T, N> {
    fn default() -> Self {
        Few { few: [T::default(); N], len: 0, many: Vec::new() }
    }
}

impl<T: Copy, const N: usize> Few<T, N> {
    fn push(&mut self, what: T) {
        if self.many.is_empty() && self.len < N {
            self.few[self.len] = what;
            self.len += 1;
        } else {
            if self.many.is_empty() {
                self.many.extend_from_slice(&self.few);
            }
            self.many.push(what);
        }
    }

    fn resize(&mut self, new_len: usize, value: T) {
        if self.many.is_empty() && new_len <= N {
            if new_len > self.len {
                self.few[self.len..new_len].fill(value);
            }
            self.len = new_len;
        } else {
            if self.many.is_empty() {
                self.many.extend_from_slice(&self.few[..self.len]);
            }
            self.many.resize(new_len, value);
        }
    }

    fn clear(&mut self) {
        self.len = 0;
        self.many.clear();
    }
}

impl<T: Copy + Ord, const N: usize> Few<T, N> {
    fn sort_dedup(&mut self) {
        if self.many.is_empty() {
            let s = &mut self.few[..self.len];
            s.sort_unstable();
            if s.len() <= 1 {
                return;
            }
            let mut w = 1usize;
            for r in 1..s.len() {
                if s[r] != s[w - 1] {
                    s[w] = s[r];
                    w += 1;
                }
            }
            self.len = w;
        } else {
            self.many.sort_unstable();
            self.many.dedup();
        }
    }
}

impl<T, const N: usize> std::ops::Deref for Few<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        if self.many.is_empty() { &self.few[..self.len] } else { &self.many }
    }
}

impl<T, const N: usize> std::ops::DerefMut for Few<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.many.is_empty() { &mut self.few[..self.len] } else { &mut self.many }
    }
}

/// A table made afresh in dynamic slots, in a list ([`State::tables`]):
/// what it was made from (the `n` records at `at` in [`State::fresh_records`]),
/// its first dynamic slot, and the readers brought up to date.
struct FreshTable {
    table: u64,
    n: u32,
    at: usize,
    first: u32,
    /// Descriptor indices within the table shown as unsigned 16-bit twins
    /// (`HostShader::half_views`), so shaders with different half-view sets do
    /// not share dynamic slots.
    half: Vec<u32>,
    /// The list's epoch ([`HostList::epoch`]) at which all its entries that
    /// these readers (a stage's list of what it reads, whether it stores, the
    /// tested depth target) read were brought up to date, and whether that read the
    /// tested depth target.
    epoch: u64,
    ready: Few<((usize, usize, bool, usize), bool), 4>,
}

/// A hash that is mixed already, used as it is.
#[derive(Default)]
struct Plain(u64);

impl std::hash::Hasher for Plain {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0 << 8 | u64::from(b);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
}

/// A quick hash of `table` and its already-read records.
fn table_hash_records(table: u64, records: &[[u32; 3]], half_slots: &[u32]) -> u64 {
    let mix = |h: u64, x: u64| (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    let h = records.iter().fold(table, |h, &[a, b, c]| {
        mix(mix(h, u64::from(a) | u64::from(b) << 32), u64::from(c))
    });
    let h = half_slots
        .iter()
        .fold(mix(h, half_slots.len() as u64), |h, &k| mix(h, u64::from(k)));
    h ^ h >> 32
}

/// A quick hash of `table` and its `n` records. It only finds the table
/// made before: the records are compared too, so a collision costs a copy,
/// not wrong descriptors.
#[cfg_attr(not(test), allow(dead_code))]
fn table_hash(table: u64, n: u32, half_slots: &[u32]) -> u64 {
    let mix = |h: u64, x: u64| (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
    let h = (0..n).fold(table, |h, k| {
        let [a, b, c] = record(table, k);
        mix(mix(h, u64::from(a) | u64::from(b) << 32), u64::from(c))
    });
    let h = half_slots
        .iter()
        .fold(mix(h, half_slots.len() as u64), |h, &k| mix(h, u64::from(k)));
    h ^ h >> 32
}

/// The words of entry `k` of `table`.
fn record(table: u64, k: u32) -> [u32; 3] {
    // SAFETY: a descriptor slot in title memory.
    unsafe { ((table + u64::from(k) * 32) as *const [u32; 3]).read_unaligned() }
}

/// Whether a table a list reads again, with nothing changed in the list
/// since, is left as it is (`VICTORY_READY_ONCE=0`: every entry is gone
/// through again at every draw).
fn ready_once() -> bool {
    crate::toggled!("VICTORY_READY_ONCE")
}

/// Whether the stages of one draw share a table they both need made afresh
/// (`VICTORY_TABLE_STAGES=0|1|alt|alt40`; on by default).
fn table_stages() -> bool {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_TABLE_STAGES").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        Ok("alt40") => 3,
        _ => 1,
    }) {
        0 => false,
        1 => true,
        mode => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let period = if mode == 3 { 40 } else { 20 };
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / period) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the stages of a draw share the tables they need made afresh from now (VICTORY_TABLE_STAGES={}): {}",
                    if mode == 3 { "alt40" } else { "alt" },
                    if on { "ON" } else { "off, each stage makes its own as usual" }
                );
            }
            on
        }
    }
}

/// Precomputes per-root-table spans across a pipeline's stages:
/// `(root slot, max slots among stages with a CBV in that table, max slots among all stages)`.
pub(crate) fn stage_spans(stages: &[crate::pipeline::host::Stage]) -> Vec<(u8, u32, u32)> {
    let mut spans: Vec<(u8, u32, u32)> = Vec::new();
    for stage in stages {
        for group in &stage.translated().tables.groups {
            let n = group.max_dword / 8 + 1;
            let with = if group.has_cbv { n } else { 0 };
            match spans.iter_mut().find(|s| s.0 == group.slot) {
                Some(s) => {
                    s.1 = s.1.max(with);
                    s.2 = s.2.max(n);
                }
                None => spans.push((group.slot, with, n)),
            }
        }
    }
    spans
}

/// Returns `true` iff every 64 KiB guard block covering `[start, start + len)`
/// is currently read-only published in `watch` (`CPU-24`), guaranteeing zero
/// unrecorded CPU writes since the last watch/hash check.
#[inline]
fn all_blocks_published(start: usize, len: usize) -> bool {
    const BLOCK: usize = 64 << 10;
    if start == 0 || len == 0 {
        return false;
    }
    let Some(end) = start.checked_add(len) else {
        return false;
    };
    let first = start / BLOCK;
    let last = end.div_ceil(BLOCK);
    (first..last).all(crate::watch::block_published)
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    readable_inner(list, kind, view, None);
}

fn readable_inner(
    list: &HostList,
    kind: ResourceType,
    view: &View,
    mut uav_capture: Option<&mut UavCapture>,
) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            list.mark_uav_dirty();
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            let do_write = STORES.get() || !write_skip();
            if do_write {
                #[cfg(feature = "diagnostics")]
                crate::host::wide_write(address, len, 0);
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
            if let Some(cap) = uav_capture.as_deref_mut() {
                cap.buffers.push(CachedUavBuffer {
                    shadow,
                    address,
                    len,
                    writes: do_write,
                });
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_copy(*resource as This)
                && t.flags() & 0x7 == 0
            {
                // (A texture the title writes: no target, depth or unordered
                // access, `crate::texture::title_written`.)
                // Of a texture the title placed, the levels the view shows.
                let first = match (&placed, view) {
                    (Some(_), View::Shader(_, Some(desc))) => crate::resource::first_mip(desc),
                    _ => 0,
                };
                // Same-frame hash dedup (`CPU-24`): if this guarded texture was
                // already checked/uploaded in this frame at `<= first` with no
                // remaps (`changes`) and all its 64 KiB guard blocks remain
                // read-only published, skip redundant per-list `upload_from`
                // scheduling and blind re-hashing.
                let baking = crate::texture::BAKING.get();
                let frame = crate::frame::current();
                let changes = crate::watch::changes();
                let can_skip = !baking
                    && t.tiled.is_none()
                    && !t.grown.load(Ordering::Relaxed)
                    && t.title_bytes > 0
                    && t.hashed_frame_clean(frame, first, changes);
                if can_skip
                    && all_blocks_published(t.title_start, t.title_bytes as usize)
                    && crate::texture::is_staged(*resource as This)
                {
                    // Already verified & staged this frame with read-only guard intact.
                } else {
                    let published = t.title_bytes > 0
                        && all_blocks_published(t.title_start, t.title_bytes as usize);
                    if !published {
                        t.invalidate_hashed_frame();
                    }
                    crate::texture::upload_from(list, *resource as This, first);
                    if !baking
                        && published
                        && t.tiled.is_none()
                        && !t.grown.load(Ordering::Relaxed)
                        && crate::texture::is_staged(*resource as This)
                        && t.holds.try_lock().is_ok_and(|g| g.is_some())
                    {
                        t.mark_hashed_frame(frame, first, changes);
                    }
                }
            }
            // A depth texture shaders also write: what they wrote, first.
            let twinned = list.twin_use(*resource as This, false, false);
            if let Some(cap) = uav_capture {
                if aliased || twinned {
                    cap.srv_tex_hazard = true;
                }
                cap.srv_textures.push(*resource);
            }
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if tracing::enabled!(tracing::Level::DEBUG)
                && let Some(t) = crate::resource::host_resource(*resource as This)
            {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            list.mark_uav_dirty();
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Some compute-written textures are also read by the title CPU.
            list.texture_written(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
            if let Some(cap) = uav_capture {
                cap.textures.push(*resource);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Some(l1) = list.list1()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

/// Per-frame texture staging byte budget scaled from the adapter's local VRAM
/// budget (`None` keeps the 192 MiB 12 GiB baseline).
fn staging_budget_per_frame(vram_budget: Option<u64>) -> u64 {
    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    match vram_budget {
        Some(budget) if budget <= 6 * GIB => 96 * MIB,
        Some(budget) if budget >= 20 * GIB => 384 * MIB,
        Some(budget) if budget > 12 * GIB => 256 * MIB,
        _ => 192 * MIB,
    }
}

#[cfg(test)]
mod tests {
    use super::staging_budget_per_frame;

    #[test]
    fn staging_budget_scales_with_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        assert_eq!(staging_budget_per_frame(None), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(4 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(6 * GIB)), 96 * MIB);
        assert_eq!(staging_budget_per_frame(Some(8 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(12 * GIB)), 192 * MIB);
        assert_eq!(staging_budget_per_frame(Some(13 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(14 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(16 * GIB)), 256 * MIB);
        assert_eq!(staging_budget_per_frame(Some(20 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(24 * GIB)), 384 * MIB);
        assert_eq!(staging_budget_per_frame(Some(32 * GIB)), 384 * MIB);
    }

    #[test]
    fn indirect_batch_defaults_to_level_2_and_honors_overrides() {
        use std::ffi::OsStr;
        assert_eq!(super::batches_for(None), 2);
        assert_eq!(super::batches_for(Some(OsStr::new("2"))), 2);
        assert_eq!(super::batches_for(Some(OsStr::new("1"))), 1);
        assert_eq!(super::batches_for(Some(OsStr::new("0"))), 0);
    }
}

#[cfg(test)]
mod record_hash_tests {
    use super::{Few, FreshTable, State, record, table_hash};

    #[test]
    fn a_few_things_are_kept_in_order_however_many() {
        for n in 0..10u32 {
            let mut few: Few<u32, 4> = Few::default();
            (0..n).for_each(|i| few.push(n - i));
            assert_eq!(few.to_vec(), (0..n).map(|i| n - i).collect::<Vec<_>>());
            few.sort_unstable();
            assert_eq!(few.to_vec(), (1..=n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn fresh_table_lookup_verifies_record_words_on_hash_collision() {
        let mut slots = [[0u32; 8]; 2];
        slots[0][..3].copy_from_slice(&[1, 2, 3]);
        slots[1][..3].copy_from_slice(&[4, 5, 6]);
        let ptr = slots.as_mut_ptr();
        let table = ptr as u64;
        let key = table_hash(table, 2, &[]);
        assert_ne!(table_hash(table, 2, &[1]), key);
        let mut state = State::default();
        let at = state.fresh_records.len();
        state.fresh_records.extend((0..2).map(|k| record(table, k)));
        state.fresh_tables.insert(
            key,
            FreshTable { table, n: 2, at, first: 42, half: Vec::new(), epoch: 0, ready: Few::default() },
        );
        let matches = |state: &State, key: u64, table: u64, n: u32| {
            state.fresh_tables.get(&key).is_some_and(|f| {
                f.table == table
                    && f.n == n
                    && state.fresh_records[f.at..f.at + n as usize]
                        .iter()
                        .enumerate()
                        .all(|(k, r)| *r == record(table, k as u32))
            })
        };
        assert!(matches(&state, key, table, 2));
        // Mutating a record word while looking up under the same hash key must miss.
        // SAFETY: `ptr` points to the live two-slot array on the stack.
        unsafe { (*ptr.add(1))[2] = 99 };
        assert!(!matches(&state, key, table, 2));
        assert_ne!(table_hash(table, 2, &[]), key);
    }

    #[test]
    fn few_sort_dedup_works_inline_and_spilled() {
        let mut inline: Few<u32, 4> = Few::default();
        for v in [5, 2, 5, 1] {
            inline.push(v);
        }
        inline.sort_dedup();
        assert_eq!(&*inline, &[1, 2, 5]);

        let mut spilled: Few<u32, 4> = Few::default();
        for v in [9, 3, 7, 3, 1, 9, 5, 1] {
            spilled.push(v);
        }
        spilled.sort_dedup();
        assert_eq!(&*spilled, &[1, 3, 5, 7, 9]);
        spilled.clear();
        assert!(spilled.is_empty());
    }

    #[test]
    fn table_hash_records_matches_table_hash() {
        let mut slots = [[0u32; 8]; 4];
        for (i, s) in slots.iter_mut().enumerate() {
            s[0] = 0x5649_4557;
            s[1] = (i as u32) * 17 + 3;
            s[2] = (i as u32) + 1;
        }
        let table = slots.as_ptr() as u64;
        let recs: Vec<[u32; 3]> = (0..4).map(|k| record(table, k)).collect();
        assert_eq!(super::table_hash_records(table, &recs, &[]), table_hash(table, 4, &[]));
        assert_eq!(super::table_hash_records(table, &recs, &[1, 3]), table_hash(table, 4, &[1, 3]));
    }

    #[test]
    fn state_coalesces_dynamic_spans_and_retains_capacity_on_reset() {
        let mut state = State::default();
        state.record_dynamic(0x1000, 1, 10, 4);
        state.record_dynamic(0x1000, 1, 14, 6);
        state.record_dynamic(0x1000, 1, 25, 2);
        state.record_dynamic(0x2000, 2, 27, 3);
        assert_eq!(
            state.dynamic,
            vec![(0x1000, 1, 10, 10), (0x1000, 1, 25, 2), (0x2000, 2, 27, 3)]
        );
        // Clear dynamic manually before reset so Drop/reset doesn't probe SHADER_HEAPS.
        state.dynamic.clear();
        for k in 0..64u64 {
            state.fresh_records.push([k as u32, 1, 2]);
            state.fresh_tables.insert(
                k,
                FreshTable { table: k, n: 1, at: k as usize, first: k as u32, half: Vec::new(), epoch: 0, ready: Few::default() },
            );
        }
        let cap_tables = state.fresh_tables.capacity();
        let cap_records = state.fresh_records.capacity();
        state.reset(std::ptr::null());
        assert!(state.fresh_tables.is_empty());
        assert!(state.fresh_records.is_empty());
        assert!(state.fresh_tables.capacity() >= cap_tables);
        assert!(state.fresh_records.capacity() >= cap_records);
    }

    #[test]
    fn viewports_eq_uses_bitwise_float_comparison() {
        use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;
        let a = D3D12_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: 1920.0,
            Height: 1080.0,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        let mut b = a;
        assert!(super::viewports_eq(&[a], &[b]));
        b.TopLeftX = -0.0;
        assert!(!super::viewports_eq(&[a], &[b]));
        b.TopLeftX = f32::NAN;
        let mut c = a;
        c.TopLeftX = f32::NAN;
        assert!(super::viewports_eq(&[b], &[c]));
    }

    #[test]
    fn use_ahead_copy_queue_enforces_worker_safety_invariants() {
        // Non-worker path: preserves `late && !tabled && fresh`.
        assert!(super::use_ahead_copy_queue(false, true, false, false, false, false, true, false, false, true));
        assert!(!super::use_ahead_copy_queue(false, false, true, false, false, false, true, false, false, true));
        assert!(!super::use_ahead_copy_queue(false, true, true, false, false, false, true, false, true, true));
        assert!(!super::use_ahead_copy_queue(false, true, true, false, false, false, true, false, false, false));

        // Heap-worker path: requires placed texture + show_after + unmade slot + no other view.
        assert!(super::use_ahead_copy_queue(true, false, true, false, false, false, true, false, false, true));
        // Non-placed View::Shader must stay off COPY queue (make_waiting hazard).
        assert!(!super::use_ahead_copy_queue(true, false, false, false, false, false, true, false, false, true));
        // Without VICTORY_SHOW_AFTER=1, table draws can mirror before ahead::listed() waits.
        assert!(!super::use_ahead_copy_queue(true, false, true, false, false, false, false, false, false, true));
        // Already-made, shows_other, baking, sync_bake, tabled, or non-fresh are excluded.
        assert!(!super::use_ahead_copy_queue(true, false, true, false, false, false, true, true, false, true));
        assert!(!super::use_ahead_copy_queue(true, false, true, true, false, false, true, false, false, true));
        assert!(!super::use_ahead_copy_queue(true, false, true, false, true, false, true, false, false, true));
        assert!(!super::use_ahead_copy_queue(true, false, true, false, false, true, true, false, false, true));
        assert!(!super::use_ahead_copy_queue(true, false, true, false, false, false, true, false, true, true));
        assert!(!super::use_ahead_copy_queue(true, false, true, false, false, false, true, false, false, false));
    }

    #[test]
    fn dropped_split_counters_track_every_drop_reason() {
        use super::{DropReason, record_drop, take_dropped_split};
        let _ = take_dropped_split();
        record_drop(DropReason::NoPso);
        record_drop(DropReason::LeftOut);
        record_drop(DropReason::LeftOut);
        record_drop(DropReason::NoBuffer);
        record_drop(DropReason::NoSlots);
        record_drop(DropReason::NoSlots);
        record_drop(DropReason::NoSlots);
        let (no_pso, left_out, no_buffer, no_slots) = take_dropped_split();
        assert!(no_pso >= 1);
        assert!(left_out >= 2);
        assert!(no_buffer >= 1);
        assert!(no_slots >= 3);
        // Clean up `DROPPED` total too.
        let _ = super::take_dropped();
    }

    #[test]
    fn cache_padded_is_64_byte_aligned_and_state_flushes_batched_counters() {
        assert_eq!(std::mem::align_of::<super::CachePadded<std::sync::atomic::AtomicU64>>(), 64);
        let before_drawn = super::drawn();
        let before_disp = super::dispatched();
        let mut state = State::default();
        state.local_drawn = 7;
        state.local_dispatched = 3;
        state.cached_heap_buffers = (0x1234, 0x5678, 0x9000);
        state.cached_heap_strides = (0x1234, 0x5678, 0xA000);
        state.batch_list = 0x1234;
        state.window_list = 0x1234;
        state.ensured_targets.0 = 0x1234;
        // `trim_dynamic_slab` (called on Close) flushes counters and clears list caches.
        state.trim_dynamic_slab();
        assert_eq!(state.local_drawn, 0);
        assert_eq!(state.local_dispatched, 0);
        assert_eq!(state.batch_list, 0);
        assert_eq!(state.window_list, 0);
        assert_eq!(state.cached_heap_buffers, (0, 0, 0));
        assert_eq!(state.cached_heap_strides, (0, 0, 0));
        assert_eq!(state.ensured_targets.0, 0);
        assert!(super::drawn() >= before_drawn + 7);
        assert!(super::dispatched() >= before_disp + 3);
    }

    #[test]
    fn sig_fast_slot_maps_standard_indirect_signatures_lock_free() {
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DRAW, 16), Some(0));
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DRAW_INDEXED, 20), Some(1));
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DISPATCH, 12), Some(2));
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DISPATCH_MESH, 12), Some(3));
        // Non-standard strides fall through to the RwLock/TLS cache.
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DRAW, 32), None);
        assert_eq!(super::sig_fast_slot(crate::indirect::KIND_DISPATCH, 20), None);
        assert_eq!(super::sig_fast_slot(99, 12), None);
    }

    #[test]
    fn state_take_dynamic_and_tables_done_clear_indirect_and_target_caches() {
        let mut state = State::default();
        state.batch_list = 0xDEAD;
        state.window_list = 0xBEEF;
        state.cached_heap_buffers = (0x1, 0x2, 0x3);
        state.cached_heap_strides = (0x1, 0x2, 0x4);
        state.ensured_targets.0 = 0xCAFE;
        let _lease = state.take_dynamic();
        assert_eq!(state.batch_list, 0);
        assert_eq!(state.window_list, 0);
        assert_eq!(state.cached_heap_buffers, (0, 0, 0));
        assert_eq!(state.cached_heap_strides, (0, 0, 0));
        assert_eq!(state.ensured_targets.0, 0);

        state.batch_list = 0x1111;
        state.window_list = 0x2222;
        state.ensured_targets.0 = 0x3333;
        state.tables_done();
        assert_eq!(state.batch_list, 0);
        assert_eq!(state.window_list, 0);
        assert_eq!(state.ensured_targets.0, 0);
    }

    #[test]
    fn uav_cached_stage_tables_replays_barriers_and_gpu_writes_and_invalidates_on_gen_change() {
        use super::{CachedStageTables, CachedUavBuffer, Few};
        let Some(h) = crate::host::host() else {
            return; // no host GPU
        };
        let Ok(list) = h.new_list() else {
            return;
        };
        let mut backing = vec![0u8; 4096];
        let start = backing.as_ptr() as u64;
        let shadow = crate::shadow::Shadow::unbacked(start, backing.as_mut_ptr(), 4096, false);
        assert!(!shadow.written());

        let list_ptr = std::ptr::from_ref(&list) as usize;
        let epochs = crate::descriptor::host::view_cache_epochs();
        let entry = CachedStageTables {
            list_ptr,
            list_epoch: list.epoch(),
            pipeline: 0xDEAD_0001,
            heaps: [0x1000, 0x2000],
            table_gen: 7,
            const_gen: 3,
            view_epochs: epochs,
            writes_mask: 1,
            has_indexed: true,
            tested_depth: 0,
            tested_depth_read: false,
            stages_made: Default::default(),
            uav_buffers: vec![CachedUavBuffer {
                shadow: shadow.clone(),
                address: start,
                len: 256,
                writes: true,
            }],
            uav_textures: Few::default(),
            srv_tex_hazard: false,
        };

        // Matches with identical state.
        assert!(entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 7, 3, 0, epochs, 1));
        // Invalidates on root table gen, indexed const gen, pipeline, heap, list epoch, or view epoch change.
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 8, 3, 0, epochs, 1));
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 7, 4, 0, epochs, 1));
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0002, [0x1000, 0x2000], 7, 3, 0, epochs, 1));
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1001, 0x2000], 7, 3, 0, epochs, 1));
        assert!(!entry.matches(list_ptr, list.epoch() + 1, 0xDEAD_0001, [0x1000, 0x2000], 7, 3, 0, epochs, 1));
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 7, 3, 0, (epochs.0 + 1, epochs.1), 1));
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 7, 3, 0, (epochs.0, epochs.1 + 1), 1));

        // Mark UAV clean first, then replay cached UAVs: must dirty UAV barrier state AND mark shadow written!
        list.mark_uav_clean();
        assert!(list.is_uav_clean());
        entry.replay_uavs(&list);
        assert!(!list.is_uav_clean(), "UAV-cached bind replay must emit UAV dirty barrier state");
        assert!(shadow.written(), "UAV-cached bind replay must mark buffer shadow as GPU-written");

        // Descriptor view epoch bump invalidates cached entry.
        crate::descriptor::host::bump_buffer_view_cache_epoch();
        let new_epochs = crate::descriptor::host::view_cache_epochs();
        assert_ne!(new_epochs, epochs);
        assert!(!entry.matches(list_ptr, list.epoch(), 0xDEAD_0001, [0x1000, 0x2000], 7, 3, 0, new_epochs, 1));
    }

    #[test]
    fn multi_indirect_cache_matches_uncached_plan_and_mutates_root_state_flag() {
        use super::{CachedMultiIndirect, multi_indirect_key_hash};
        use crate::indirect::{
            KIND_CONSTANT, KIND_DISPATCH, StageBinding, translate_signature,
        };
        use victory_shader::RootType;
        use victory_shader::emit::UserData;

        let user_data = [UserData {
            root: RootType::Constants,
            slot: 0,
            dword: 0,
            index: 2,
        }];
        let bindings = [StageBinding {
            user_data_param: Some(0),
            user_data: &user_data,
            root_cbvs: &[],
            root_srvs: &[],
            root_uavs: &[],
        }];

        // Signature 1: mutates root constant slot 0 dword 0 -> `mutates_root_state == true`.
        let args_mut = [[KIND_CONSTANT, 0, 0, 1], [KIND_DISPATCH, 0, 0, 0]];
        let plan_mut = translate_signature(16, &args_mut, &bindings, true).expect("plan_mut");
        assert!(plan_mut.has_root_args);

        // Signature 2: writes unused console slot 3 -> `mutates_root_state == false`!
        let args_nomut = [[KIND_CONSTANT, 3, 0, 1], [KIND_DISPATCH, 0, 0, 0]];
        let plan_nomut = translate_signature(16, &args_nomut, &bindings, true).expect("plan_nomut");
        assert!(!plan_nomut.has_root_args);

        let shaders = [0x1111usize, 0, 0, 0, 0];
        let h1 = multi_indirect_key_hash(16, &args_mut, 0xAA00, &shaders, true);
        let h2 = multi_indirect_key_hash(16, &args_nomut, 0xAA00, &shaders, true);
        assert_ne!(h1, h2);

        let Some(sig_nomut) = super::host_multi_signature(None, plan_nomut.dst_stride, &plan_nomut.host_args) else {
            return; // no host GPU
        };
        let cached = CachedMultiIndirect {
            stride: 16,
            args: args_nomut.to_vec(),
            root_raw: 0xAA00,
            shader_ptrs: shaders,
            bounded_root_srvs: true,
            _root: None,
            plan: plan_nomut.clone(),
            signature: sig_nomut,
            mutates_root_state: plan_nomut.has_root_args,
        };
        assert_eq!(cached.plan, plan_nomut);
        assert!(!cached.mutates_root_state);
        assert!(cached.matches(16, &args_nomut, 0xAA00, &shaders, true));
        // Must miss if args, stride, root_raw, shader_ptrs, or bounded flag differ.
        assert!(!cached.matches(16, &args_mut, 0xAA00, &shaders, true));
        assert!(!cached.matches(20, &args_nomut, 0xAA00, &shaders, true));
        assert!(!cached.matches(16, &args_nomut, 0xBB00, &shaders, true));
        assert!(!cached.matches(16, &args_nomut, 0xAA00, &[0x2222, 0, 0, 0, 0], true));
        assert!(!cached.matches(16, &args_nomut, 0xAA00, &shaders, false));

        // Verify `State` preserves `indirect_rebase` scratch capacity across `reset()`.
        let mut state = State::default();
        state.indirect_rebase = Vec::with_capacity(64);
        state.indirect_rebase.push([1; 8]);
        let cap = state.indirect_rebase.capacity();
        state.reset(std::ptr::null());
        assert!(state.indirect_rebase.is_empty());
        assert!(state.indirect_rebase.capacity() >= cap);
    }
}
