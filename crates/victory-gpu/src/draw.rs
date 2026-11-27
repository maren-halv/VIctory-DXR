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

/// Free-threaded wrapper around a reusable upload-only [`HostList`] protected
/// by [`BAKE_UPLOAD`]. D3D12 command allocators, lists, and retained COM
/// resources are free-threaded when externally synchronized.
struct BakeList(HostList);
// SAFETY: accessed exclusively under `BAKE_UPLOAD`'s mutex.
unsafe impl Send for BakeList {}

/// Serializes synchronous one-shot ped/decoration bake uploads across
/// concurrent `VICTORY_THIN=1` recording threads and owns the reusable
/// upload-only host command list.
static BAKE_UPLOAD: std::sync::Mutex<Option<BakeList>> = std::sync::Mutex::new(None);

/// Pending descriptor publications for a synchronous bake upload pass: slots
/// stay on `neutral()` (or their previous compatible view) until the dedicated
/// upload list has finished executing on the host GPU.
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

static IN_PLACE_INDIRECT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
#[cfg(feature = "diagnostics")]
static IN_PLACE_RANGES: std::sync::Mutex<std::collections::BTreeMap<u64, [u64; 3]>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

fn in_place_indirect(arguments: u64, kind: u32) {
    IN_PLACE_INDIRECT[0].fetch_add(1, Ordering::Relaxed);
    IN_PLACE_INDIRECT[1].fetch_add(u64::from(in_place_order()), Ordering::Relaxed);
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
    let [all, alone] = [0, 1].map(|i| IN_PLACE_INDIRECT[i].swap(0, Ordering::Relaxed));
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

/// The records clamped by the pass opened last ([`batches`]).
struct Batch {
    table: crate::indirect::batch::Table,
    scratch: ID3D12Resource,
    /// Dispatches since, which may have written arguments
    /// (`VICTORY_INDIRECT_BATCH=2`).
    wrote: bool,
}

/// Whether any stage of graphics pipeline `p` can write to a buffer (excluding
/// texture-only UAV stores so they do not invalidate an open clamp batch).
fn pipeline_stores_buffers(p: &HostPipeline) -> bool {
    p.stages.iter().any(|stage| {
        let t = stage.translated();
        if !t.stores {
            return false;
        }
        t.stores_by_address
            || !stage.root_uavs.is_empty()
            || t.used.iter().any(|u| u.kind == ResourceType::Uav && u.dwords != 8)
            || crate::picked::picks(&stage.shader, &t.picks)
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
    /// The records clamped in a batch, if later executions may still read
    /// them ([`batches`]).
    batch: Option<Batch>,
    /// Per-list MRU of host copies indirect arguments were read from last,
    /// avoiding global mutex contention and cross-list pollution.
    argument_copies: Vec<std::sync::Weak<crate::shadow::Shadow>>,
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
        self.trim_dynamic_slab();
        if !self.dynamic.is_empty() {
            crate::descriptor::release_dynamic_batch(&self.dynamic);
            self.dynamic.clear();
        }
    }
}

impl State {
    /// Returns any unused tail of this list's dynamic bump slab to its heap.
    /// Called on `Close()` so closed/in-flight lists hold zero wasted slots.
    pub(crate) fn trim_dynamic_slab(&mut self) {
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
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
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

/// The draws dropped since [`take_dropped`].
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn dropped_now() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
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
}

impl Default for BufferSweep {
    fn default() -> Self {
        Self {
            passes: 0,
            ranges: DenseRanges::default(),
            in_place: Default::default(),
            unwatched: Default::default(),
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
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
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
    let shadow = crate::shadow::find(address, 1)?;
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
    let BufferSweep { passes, ranges, in_place, unwatched } = &mut *kept;
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
        self.window = None;
        self.batch = None;
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
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
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
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| (s.offset(address) & 15) as u32),
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
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return None;
        }
        let p = crate::pipeline::host_pipeline(self.pipeline as This)?;
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
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
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
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
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
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
        let mut staged: Vec<(u64, u32)> = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            if t.stores && crate::picked::on() && crate::picked::picks(&stage.shader, &t.picks) {
                crate::picked::runs_in(list);
            }
            let made = self.tables(list, &t.tables, &t.host.half_views, compute, spans, &mut staged);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_why(format!("no free dynamic slots for a table of {}", p.name));
                return false;
            }
            if stage.shader.ambient.is_some() {
                self.ambient_tables.clear();
                self.ambient_tables.extend(made.iter().copied());
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
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
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
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
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
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
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        let frame = crate::frame::current();
        if host.buffer_frame.swap(frame, Ordering::Relaxed) == frame {
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
    fn heap_strides(&self, list: &HostList) -> Option<u64> {
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
        Some(gpu)
    }

    /// Ensures textures claimed by background copy-queue uploads
    /// (`VICTORY_UPLOAD_AHEAD=1`) or deferred behind the `neutral()` gray
    /// stand-in (`VICTORY_SHOW_AFTER=1` / `shows_other`) have finished
    /// uploading and flipped to their real descriptors before a one-shot
    /// decoration bake draw samples them.
    fn flush_for_bake(
        &self,
        list: &HostList,
        host: &crate::descriptor::host::HostHeap,
        mut sync: Option<&mut BakeSync>,
    ) {
        static RESCUED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
        let drained = crate::texture::ahead::drain();
        let mut settled = false;
        let mut flipped = 0usize;
        for _ in 0..4 {
            let need_settle = host.needs_gpu_settle_for_bake();
            if need_settle {
                settled = true;
                crate::host::settle_for_fence();
            }
            flipped += host.texture_views_remade();
            flipped += host.texture_slots_flip();
            if !need_settle || !host.needs_gpu_settle_for_bake() {
                break;
            }
        }
        let pending = host.take_pending_list_flips();
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
        if drained > 0 || settled || flipped > 0 || promoted > 0 {
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
                    settled,
                    "flushed deferred texture stand-ins before decoration bake"
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
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. For decoration bakes, uploads run synchronously
    /// on a dedicated internal command list and descriptors are published only
    /// after the GPU copy finishes so concurrent `VICTORY_THIN=1` lists can
    /// never submit a bake draw ahead of its texture uploads.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let is_bake = bake_ms > 0 && self.is_bake_draw(p);
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
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
            let _pass = crate::heap_worker::on().then(crate::heap_worker::urgent);
            let began = std::time::Instant::now();
            let mut bake_slot = BAKE_UPLOAD.lock().unwrap_or_else(|e| e.into_inner());
            let gpu = crate::host::host();
            if bake_slot.is_none()
                && let Some(gpu) = gpu
            {
                match gpu.new_list() {
                    Ok(upload_list) => *bake_slot = Some(BakeList(upload_list)),
                    Err(e) => tracing::warn!(error = %e, "no host list for synchronous bake uploads"),
                }
            }
            let exact = bake_exact();
            if let (Some(gpu), Some(BakeList(bake_list))) = (gpu, bake_slot.as_ref()) {
                let mut sync = BakeSync { publish: Vec::new(), keep: Vec::new() };
                crate::texture::BAKING.set(exact);
                self.flush_for_bake(bake_list, host, Some(&mut sync));
                let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
                let run_pass = left > 0
                    && (first
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
                            queued_at_the_last_pass = slots,
                            due_at_the_last_pass = due,
                            uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                            "a draw of the title's decoration bake: textures picked by index are uploaded synchronously first (VICTORY_BAKE_TEXTURES_MS=0: not)"
                        );
                    }
                    BAKE_SERVED.store(0, Ordering::Relaxed);
                    PASS_BUDGET.set(Some((left, 2 << 30, true)));
                    served = Self::heap_textures(Into::List(bake_list), heap, host, Some(&mut sync));
                    PASS_BUDGET.set(None);
                }
                crate::texture::BAKING.set(false);
                if bake_list.has_recorded_work() {
                    bake_list.close();
                    gpu.execute_sync_internal(bake_list);
                    bake_list.reset();
                } else if !sync.publish.is_empty() {
                    crate::host::settle_for_fence();
                }
                for (index, handle, key) in sync.publish {
                    let _ = host.publish_uploaded_slot(index, handle, key);
                }
                host.texture_views_remade();
                host.texture_slots_flip();
                drop(sync.keep);
                if run_pass && served && exact {
                    BAKE_SERVED.store(host.texture_news().0 + 1, Ordering::Relaxed);
                }
                BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
                return;
            }
            // Fallback when no host GPU list can be created (e.g. headless tests).
            crate::texture::BAKING.set(exact);
            self.flush_for_bake(list, host, None);
            crate::texture::BAKING.set(false);
            let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
            if left == 0 {
                return;
            }
            if !first && exact {
                let (writes, noted) = host.texture_news();
                if !noted && BAKE_SERVED.load(Ordering::Relaxed) == writes + 1 {
                    return;
                }
            }
            EARLY_TEXTURE_PASSES[1].fetch_add(1, Ordering::Relaxed);
            PASS_BUDGET.set(Some((left, 2 << 30, true)));
            crate::texture::BAKING.set(exact);
            let _ = Self::heap_textures(Into::List(list), heap, host, None);
            crate::texture::BAKING.set(false);
            PASS_BUDGET.set(None);
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
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
        mut into: Into<'_>,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        mut sync: Option<&mut BakeSync>,
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
        let (mut due, census) = host.texture_slots_due(frame, if baking { 1 << 16 } else { 4096 });
        // (Those that are due later after all that are due.)
        let on_time = due.len();
        if baking {
            let known: std::collections::HashSet<u32> = due.iter().map(|&(index, _)| index).collect();
            static AHEAD: AtomicU64 = AtomicU64::new(u64::MAX);
            let again = AHEAD.swap(frame, Ordering::Relaxed) == frame;
            let (ahead, _) = host.texture_slots_due(frame + BAKE_AHEAD[usize::from(again)], 1 << 16);
            due.extend(ahead.into_iter().filter(|(index, _)| !known.contains(index)));
        }
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
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
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                host.texture_slot_here(index);
                there = Some(mapped);
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
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
            let same = !tabled && crate::texture::unchanged(resource, first);
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

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
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
            SPREAD[0].fetch_add(1, Ordering::Relaxed);
        } else if gx > 65_535 || gy > 65_535 || gz > 65_535 {
            once!(gx, gy, gz, "a dispatch asks for more than 65,535 groups in a dimension and is not spread");
        }
        self.dispatch_spread = [row, asked];
        self.dispatch_threads = partial;
        let bound = self.bind(list, p, true);
        self.dispatch_spread = [0, 0];
        self.dispatch_threads = [0; 3];
        if !bound {
            DROPPED.fetch_add(1, Ordering::Relaxed);
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
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
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
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
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
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                _ => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
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
            DROPPED.fetch_add(1, Ordering::Relaxed);
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

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
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
        staged: &mut Vec<(u64, u32)>,
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
            let needs_snapshot = group.has_cbv
                || match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::Table(table)) => self.heap_of(table).is_some_and(|heap| {
                        heap.host()
                            .is_some_and(|host| group.needs_snapshot(heap, host.count(), table))
                    }),
                    _ => false,
                };
            if needs_snapshot {
                fresh[usize::from(slot / 64)] |= 1u64 << (slot % 64);
            }
        }
        for &(g, e) in &plan.direct {
            let group = &plan.groups[g];
            if !selected_slot(&fresh, group.slot) {
                self.table_entry(list, &group.entries[e], compute);
            }
        }
        let mut made = Few::default();
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
            // Snapshot the table's `n` records once on the stack (when `n <= 64`)
            // so hashing, collision check, `fresh_records` append, and slot building
            // read title memory once instead of three times.
            let mut records: Few<[u32; 3], 64> = Few::default();
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
                let once = ready_once() && selected.is_empty() && (!group.has_uav || !STORES.get());
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
                        readable(list, u.kind, &view);
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let handle = table + u64::from(u.table_offset_dwords) * 4;
                            host.texture_slot_tabled(title + k as u32, true);
                            host.mirror_after(list, title + k as u32, handle);
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
                        readable(list, kind, &view);
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let handle = table + u64::from(k) * 32;
                            host.texture_slot_tabled(title + k, true);
                            host.mirror_after(list, title + k, handle);
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
                                } else if let Some(view) = view_opt {
                                    let _ = crate::residency::take_staging_deferred();
                                    TESTED_DEPTH.set((tested.0, false));
                                    readable(list, kind, &view);
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
                                if kind.is_some() {
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
                                readable(list, kind, &view);
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
                                if kind.is_some() {
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
                DRAWN.fetch_add(1, Ordering::Relaxed);
                true
            }
            _ => false,
        };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
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
            DRAWN.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
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
            batch => *batch = None,
        }
    }

    /// A draw with the pipeline `p` was recorded (as [`State::buffers_written`]
    /// if its shaders can store to buffers).
    fn drew(&mut self, list: &HostList, p: Option<&HostPipeline>) {
        if p.is_none_or(pipeline_stores_buffers) {
            list.mark_graphics_uav_dirty();
            self.batch = None;
        }
    }

    /// Updates and returns this list's MRU of recent indirect argument copies
    /// (excluding `first`, which is moved to the front).
    fn argument_copies(&mut self, first: &std::sync::Arc<crate::shadow::Shadow>) -> Vec<(u64, ID3D12Resource, bool)> {
        let live: Vec<_> = self
            .argument_copies
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .filter(|s| s.id() != first.id())
            .collect();
        self.argument_copies.clear();
        self.argument_copies.push(std::sync::Arc::downgrade(first));
        self.argument_copies.extend(
            live.iter()
                .take(crate::indirect::batch::SOURCES - 1)
                .map(std::sync::Arc::downgrade),
        );
        live.into_iter().map(|s| (s.id(), s.buffer.clone(), s.in_place())).collect()
    }

    /// A buffer was copied to: what was clamped of it ahead is clamped anew.
    pub(crate) fn copied(&mut self) {
        self.window = None;
        self.batch = None;
    }

    /// The title waits for earlier work (a barrier or a pipeline sync).
    pub(crate) fn synced(&mut self) {
        if self.batch.as_ref().is_some_and(|b| b.wrote) {
            self.batch = None;
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
        const BATCH_USE: usize = usize::MAX - 0x78;
        const DISPATCH: u32 = 2;
        // Another host list, or this one after its reset ([`windows`]).
        if list.first_use(BATCH_USE) {
            self.batch = None;
        }
        // A draw after dispatches waits for them.
        if !matches!(kind, DISPATCH | crate::indirect::KIND_DISPATCH_SPREAD) {
            self.synced();
        }
        let (id, offset) = (args.id(), args.offset(arguments));
        if let Some(b) = &mut self.batch
            && let Some(at) = b.table.place(id, offset, kind)
        {
            return Some((b.scratch.clone(), at));
        }
        self.batch = None;
        let recent = self.argument_copies(args);
        let (mut table, scratch) = crate::indirect::open_batch(
            list,
            (id, &args.buffer, args.in_place()),
            &recent,
        )?;
        self.invalidate_compute_host_state();
        let at = table.place(id, offset, kind)?;
        self.batch = Some(Batch { table, scratch: scratch.clone(), wrote: false });
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
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = crate::indirect::KIND_DRAW;
        const INDEXED: u32 = crate::indirect::KIND_DRAW_INDEXED;
        const DISPATCH: u32 = crate::indirect::KIND_DISPATCH;
        const MESH: u32 = crate::indirect::KIND_DISPATCH_MESH;
        const MESH_1D: u32 = crate::indirect::KIND_DISPATCH_MESH_1D;
        let is_mesh = matches!(kind, MESH | MESH_1D);
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
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
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
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
        // A dispatch of a program that takes the spread words: the clamp
        // writes [row, asked x, x, y, z] records, run through a signature
        // that sets the two words, then dispatches.
        let spread = (kind == DISPATCH && p.compute)
            .then(|| spreads(p))
            .flatten()
            .and_then(|(param, index)| host_spread_signature(&p.root, param, index));
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
            why("no host signature");
            return false;
        };
        if is_mesh && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
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
                self.batch = None;
            }
            self.window = None;
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
            }
            // A window of records clamped in one pass (`windows`): for single
            // records without a count buffer, drawn by shaders that store to no
            // buffer. Anything else ends the window: it uses the scratch buffer
            // or may write the arguments.
            const WINDOW_USE: usize = usize::MAX - 0x77;
            let windowed = windows()
                && !ordered
                && kind != DISPATCH
                && kind != MESH_1D
                && max == 1
                && count.is_none()
                && !pipeline_stores_buffers(p);
            // Another host list (a new one after a cut, or this one after its
            // reset) has another scratch buffer, or this one's with old bytes.
            if !windowed || list.first_use(WINDOW_USE) {
                self.window = None;
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
                &signature,
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
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if is_spread {
                // `host_spread_signature` mutates compute root constants on GPU.
                self.host_root_args[1].clear();
                SPREAD[1].fetch_add(1, Ordering::Relaxed);
            }
            if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
                && crate::ambient::on()
                && self.ambient(list, p, role)
            {
                self.invalidate_compute_host_state();
            }
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
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
            StageBinding,
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
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("multi-arg indirect {args:?}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        let bindings: Vec<StageBinding<'_>> = p
            .stages
            .iter()
            .map(|stage| StageBinding {
                user_data_param: stage.user_data.map(|(param, _)| param),
                user_data: &stage.translated().host.user_data,
                root_cbvs: &stage.root_cbvs,
                root_srvs: &stage.root_srvs,
                root_uavs: &stage.root_uavs,
            })
            .collect();
        let Some(plan) = crate::indirect::translate_signature(
            stride,
            args,
            &bindings,
            crate::pipeline::host::bounded_root_srvs(),
        ) else {
            why("unsupported multi-argument signature layout");
            return false;
        };
        if plan.skipped_tabled_views {
            once!(
                "indirect root SRV/UAV targets a bounded descriptor-table parameter; set VICTORY_ROOT_SRV_DIRECT=1 to enable indirect root SRV/UAV updates"
            );
        }
        let kind = plan.exec_kind;
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
        let Some(signature) = host_multi_signature(
            plan.has_root_args.then_some(&p.root),
            plan.dst_stride,
            &plan.host_args,
        ) else {
            why("no host multi-argument signature");
            return false;
        };
        arg_shadow.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        let rebase = if plan.needs_rebase {
            collect_rebase_entries(list, self.roots_ref(p.compute))
        } else {
            Vec::new()
        };
        let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
        let any_in_place = arg_shadow.in_place() || counter.as_ref().is_some_and(|(s, _)| s.in_place());
        let Some(bounded) = crate::indirect::repack(
            list,
            (&arg_shadow.buffer, arg_shadow.offset(arguments)),
            count_arg,
            &plan,
            &rebase,
            max,
            any_in_place,
        ) else {
            self.window = None;
            why("multi-arg repack failed");
            return false;
        };
        self.invalidate_compute_host_state();
        self.window = None;
        let ran = match kind {
            KIND_DRAW | KIND_DRAW_INDEXED => self.prepare(list, kind == KIND_DRAW_INDEXED).is_some(),
            KIND_DISPATCH_MESH => self.prepare_for(list, false, true).is_some(),
            KIND_DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
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
                &signature,
                max,
                &bounded,
                0,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if plan.has_root_args {
            // Multi-arg indirect signature modifies root constants/views on GPU.
            self.host_root_args[usize::from(p.compute)].clear();
        }
        if kind == KIND_DISPATCH_MESH {
            self.mesh_scissor(list, false);
        }
        if kind == KIND_DISPATCH {
            self.buffers_written(list);
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if let Some(role) = p.stages.first().and_then(|s| s.shader.ambient)
                && crate::ambient::on()
                && self.ambient(list, p, role)
            {
                self.invalidate_compute_host_state();
            }
        } else {
            self.drew(list, Some(p));
            DRAWN.fetch_add(1, Ordering::Relaxed);
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

/// The command signature of a spread dispatch: two root constants (row,
/// asked x) at `index` of root parameter `param`, then the dispatch; 20
/// bytes a record (the clamp's spread layout). Made once per root
/// signature, parameter and index. (After upstream VIctory pull request
/// #2312.)
fn host_spread_signature(
    root: &ID3D12RootSignature,
    param: u32,
    index: u32,
) -> Option<ID3D12CommandSignature> {
    type Key = (usize, u32, u32);
    static MADE: std::sync::Mutex<Vec<(Key, ID3D12CommandSignature)>> = std::sync::Mutex::new(Vec::new());
    let key = (root.as_raw() as usize, param, index);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == key) {
        return Some(s.clone());
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
    Some(signature)
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
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
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Collects active live shadow ranges into a sorted, non-overlapping GPU
/// VA-rebase table for indirect root CBV/SRV/UAV translation, keeping every
/// referenced shadow buffer alive on `list`.
fn collect_rebase_entries(list: &HostList, roots: &[Root]) -> Vec<[u32; 8]> {
    const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
    let mut ranges: Vec<(u64, u64, u64, bool)> = Vec::new();
    let mut seen_ids: Vec<u64> = Vec::new();
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
    crate::indirect::build_rebase_table(&ranges)
}

/// Cached host multi-argument command signature keyed by
/// `(root_sig_ptr, host_stride, host_args_layout)`.
fn host_multi_signature(
    root: Option<&ID3D12RootSignature>,
    stride: u32,
    args: &[crate::indirect::HostIndirectArg],
) -> Option<ID3D12CommandSignature> {
    use crate::indirect::HostIndirectArg;
    type Key = (usize, u32, Vec<HostIndirectArg>);
    type Entry = (Option<ID3D12RootSignature>, ID3D12CommandSignature);
    static MADE: std::sync::LazyLock<
        std::sync::Mutex<std::collections::HashMap<Key, Entry>>,
    > = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let root_ptr = root.map_or(0, |r| r.as_raw() as usize);
    let key = (root_ptr, stride, args.to_vec());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
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

fn readable(list: &HostList, kind: ResourceType, view: &View) {
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
            if STORES.get() || !write_skip() {
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
                crate::texture::upload_from(list, *resource as This, first);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
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
}
