//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! The scratch buffer is left readable as indirect arguments after a clamp
//! and stays so until the next clamp into it, which moves it straight back
//! to unordered access; it goes back to the common state once, when the
//! list closes. The executions that read it record no barrier of their
//! own, so draws that read records clamped together (`crate::draw`'s
//! windows) follow each other with nothing between them. Before, each
//! execution was followed by a move of the scratch buffer back to the
//! common state, which drivers may treat as a full wait.
//!
//! The executions of one record without a count buffer (all the title's)
//! are clamped in batches ([`batch`], `VICTORY_INDIRECT_BATCH`): the rest of
//! this is for the others.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (below 2^23 vertices an instance, 2^26 vertex instances, 65,535
//! groups per dimension, in x 2^20 ([`widest`]), and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

pub(crate) mod batch;

pub(crate) const KIND_DRAW: u32 = 0;
pub(crate) const KIND_DRAW_INDEXED: u32 = 1;
pub(crate) const KIND_DISPATCH: u32 = 2;
pub(crate) const KIND_CONSTANT: u32 = 5;
pub(crate) const KIND_CBV: u32 = 6;
pub(crate) const KIND_SRV: u32 = 7;
pub(crate) const KIND_UAV: u32 = 8;
pub(crate) const KIND_DISPATCH_MESH: u32 = 10;
/// Internal clamp kind for 1D `DispatchMeshX` GPU-indirect arguments: loads a
/// single `u32` `ThreadGroupCountX` and emits a 12-byte D3D12
/// `D3D12_DISPATCH_MESH_ARGUMENTS` `[x, 1, 1]` (or `[0, 0, 0]` when `x == 0`
/// or `x > 65535`).
pub(crate) const KIND_DISPATCH_MESH_1D: u32 = 11;
/// Internal clamp kind for spread indirect compute dispatches in [`batch`]:
/// loads 3 words `[x, y, z]` and emits a 20-byte spread record
/// `[row, asked_x, gx, gy, gz]`.
pub(crate) const KIND_DISPATCH_SPREAD: u32 = 12;
pub(crate) const KIND_DISPATCH_SINGLE: u32 = 105;

/// Whether indirect argument clamping is active (on by default;
/// `VICTORY_INDIRECT_CLAMP=0` executes plain single-argument indirect records
/// directly from the title buffer's host copy without running the clamp pass).
/// shortcut: disabling clamping risks a GPU TDR if the title leaves unfilled
/// garbage slots or out-of-range counts in its indirect argument buffers.
pub(crate) fn clamp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| clamp_enabled_for(std::env::var_os("VICTORY_INDIRECT_CLAMP").as_deref()))
}

fn clamp_enabled_for(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Whether indirect clamp passes use D3D12 Enhanced `D3D12_BUFFER_BARRIER`
/// (`VICTORY_ENHANCED_CLAMP`, on by default when `ID3D12GraphicsCommandList7`
/// and `EnhancedBarriersSupported` are available; `0` falls back to legacy
/// `ResourceBarrier`).
pub(crate) fn enhanced_clamp() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enhanced_clamp_for(std::env::var_os("VICTORY_ENHANCED_CLAMP").as_deref()))
}

fn enhanced_clamp_for(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Returns `ID3D12GraphicsCommandList7` for Enhanced clamp barriers when both
/// `VICTORY_ENHANCED_CLAMP` and device Enhanced Barriers support are active.
pub(crate) fn enhanced_clamp_list(list: &HostList) -> Option<&ID3D12GraphicsCommandList7> {
    enhanced_clamp().then(|| list.enhanced_barriers()).flatten()
}

/// Whether indirect clamp passes track the per-list work budget and diagnostic
/// counters (`VICTORY_INDIRECT_BUDGET`; defaults to on in `test` and
/// `diagnostics` builds, off in release builds so `batch::SOURCE` runs as a
/// lean branchless clamp with zero `groupshared` syncs, zero global atomics,
/// and zero `zero_card_budget`/`card_budget_out` passes).
pub(crate) fn budget_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| budget_enabled_for(std::env::var_os("VICTORY_INDIRECT_BUDGET").as_deref()))
}

fn budget_enabled_for(value: Option<&std::ffi::OsStr>) -> bool {
    match value.and_then(|v| v.to_str()) {
        Some("0") => false,
        Some("1") => true,
        _ => cfg!(any(test, feature = "diagnostics")),
    }
}

/// Whether `resource` was created with `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS`
/// (required before naming `D3D12_BARRIER_ACCESS_UNORDERED_ACCESS` on it).
#[inline]
pub(crate) fn uav_capable(resource: &ID3D12Resource) -> bool {
    // SAFETY: a live D3D12 resource.
    unsafe { resource.GetDesc() }
        .Flags
        .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
}

/// Builds a whole-buffer Enhanced `D3D12_BUFFER_BARRIER` borrowing `resource`.
#[inline]
pub(crate) fn enhanced_buffer_barrier(
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
) -> D3D12_BUFFER_BARRIER {
    D3D12_BUFFER_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
        // SAFETY: borrowed pointer; the barrier never releases it.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Offset: 0,
        Size: u64::MAX,
    }
}

/// Synchronizes `args` (and optional `count`) before unclamped `ExecuteIndirect`
/// under `VICTORY_INDIRECT_CLAMP=0`: uses Enhanced `D3D12_BUFFER_BARRIER`
/// (`UNORDERED_ACCESS -> INDIRECT_ARGUMENT`, issuing the required CP/PFP sync)
/// when available, otherwise resource-scoped legacy UAV barriers.
pub(crate) fn sync_bypass_indirect(
    list: &HostList,
    args: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
) {
    let distinct_count = count.filter(|c| {
        windows::core::Interface::as_raw(*c) != windows::core::Interface::as_raw(args)
    });
    if let Some(list7) = enhanced_clamp_list(list) {
        let sync_before = if list.is_graphics_uav_dirty() {
            D3D12_BARRIER_SYNC_ALL_SHADING
        } else {
            D3D12_BARRIER_SYNC_COMPUTE_SHADING
        };
        let mut barriers: [D3D12_BUFFER_BARRIER; 2] =
            std::array::from_fn(|_| D3D12_BUFFER_BARRIER::default());
        let mut len = 0usize;
        if uav_capable(args) {
            barriers[len] = enhanced_buffer_barrier(
                args,
                sync_before,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            );
            len += 1;
        }
        if let Some(c) = distinct_count
            && uav_capable(c)
        {
            barriers[len] = enhanced_buffer_barrier(
                c,
                sync_before,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
            );
            len += 1;
        }
        if len > 0 {
            let group = [D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_BUFFER,
                NumBarriers: len as u32,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pBufferBarriers: barriers.as_ptr(),
                },
            }];
            // SAFETY: an open list and live UAV-capable buffers in COMMON.
            unsafe { list7.Barrier(&group) };
        }
        return;
    }
    let mut barriers: [D3D12_RESOURCE_BARRIER; 2] =
        std::array::from_fn(|_| D3D12_RESOURCE_BARRIER::default());
    barriers[0] = uav_barrier_for(args);
    let mut len = 1usize;
    if let Some(c) = distinct_count {
        barriers[len] = uav_barrier_for(c);
        len += 1;
    }
    // SAFETY: an open list and live buffers.
    unsafe { list.list.ResourceBarrier(&barriers[..len]) };
}

/// How indirect-clamp passes emit UAV barriers (`VICTORY_SCOPED_CLAMP_BARRIERS`):
/// - `0`: legacy global `NULL` UAV barrier before and after single clamps
/// - `1` (default): resource-scoped UAV barriers on non-in-place inputs + budget,
///   skipping input UAV barriers when `HostList` already has a clean UAV barrier
///   state, and omitting the redundant post-clamp global `NULL` UAV barrier
/// - `2` / `all`: always resource-scoped even for placed in-place buffers.
pub(crate) fn scoped_clamp_barriers() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        scoped_clamp_barriers_for(std::env::var_os("VICTORY_SCOPED_CLAMP_BARRIERS").as_deref())
    })
}

fn scoped_clamp_barriers_for(value: Option<&std::ffi::OsStr>) -> u8 {
    match value.and_then(|v| v.to_str()) {
        Some("0") => 0,
        Some("2" | "all") => 2,
        _ => 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreInputUav {
    None,
    Scoped,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ClampBarrierPlan {
    pub(crate) pre_inputs: PreInputUav,
    pub(crate) pre_budget: bool,
    pub(crate) post_global_uav: bool,
    pub(crate) marks_uav_clean: bool,
}

pub(crate) fn plan_clamp_barriers(
    scoped_mode: u8,
    waits: bool,
    uav_clean: bool,
    any_in_place: bool,
    budget_dirty: bool,
    legacy_post_uav: bool,
) -> ClampBarrierPlan {
    if !waits {
        return ClampBarrierPlan {
            pre_inputs: PreInputUav::None,
            pre_budget: false,
            post_global_uav: false,
            marks_uav_clean: false,
        };
    }
    if scoped_mode == 0 {
        return ClampBarrierPlan {
            pre_inputs: PreInputUav::Global,
            pre_budget: false,
            post_global_uav: legacy_post_uav,
            marks_uav_clean: true,
        };
    }
    let pre_inputs = if uav_clean {
        PreInputUav::None
    } else if any_in_place && scoped_mode == 1 {
        PreInputUav::Global
    } else {
        PreInputUav::Scoped
    };
    ClampBarrierPlan {
        pre_inputs,
        pre_budget: budget_dirty && pre_inputs != PreInputUav::Global,
        post_global_uav: false,
        marks_uav_clean: pre_inputs == PreInputUav::Global,
    }
}

/// Whether a plain single-argument indirect execution can safely bypass the
/// clamp pass under `VICTORY_INDIRECT_CLAMP=0` (only native PC record layouts
/// whose stride and offsets are valid).
pub(crate) fn can_bypass_clamp(
    kind: u32,
    stride: u32,
    src_offset: u64,
    count_offset: Option<u64>,
) -> bool {
    matches!(kind, KIND_DRAW | KIND_DRAW_INDEXED | KIND_DISPATCH | KIND_DISPATCH_MESH)
        && validate_layout((kind, stride, 1), src_offset, count_offset).is_some()
}

const SOURCE: &str = r#"
#if BUDGET_TRACK
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
#else
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0)"
#endif
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 3: dispatch records spread
    // across z past 65,535 groups in x (`crate::draw`'s spread command
    // signature): record i goes to byte 20 * i as [row, asked x, x, y, z],
    // row 0 when not spread.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
#if BUDGET_TRACK
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
#endif
// The vertices or indices an instance from which a draw record runs empty.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
#if BUDGET_TRACK
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64 threads racing for one word, cost a
// round for each record of a window, and in memory the CPU maps (where the
// budget was, and is with VICTORY_INDIRECT_CARD_BUDGET=0) every atomic
// crosses the bus on a card of its own.
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
#if !BUDGET_TRACK
    if (id.x >= eff_count) {
        return;
    }
    bool active = true;
#else
    uint first = gid.x * 64u;
    if (first >= eff_count) {
        return;
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
#endif
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * (kind == 11u ? max(stride, 12u) : stride);
    bool spread = (has_count & 8u) != 0;
    uint row = 0;
    uint asked_x = 0;
    if (spread) {
        dst_at = id.x * 20u;
    }
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
#if BUDGET_TRACK
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
#endif
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        // A record of MOST_DRAW vertices or indices an instance, or more, is
        // no draw of the title's (a slot it has not filled): it runs empty
        // and charges nothing (victory-fork f85101b6, its window checker
        // found 90 of 2^24 vertices in one list).
        if (want.x >= MOST_DRAW) {
            want = uint2(0, 0);
        }
        uint count = want.x;
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 raw = src_args.Load3(src_at);
        // Diagnostics: what the record asked for before the clamp. A record
        // ahead of its dispatch the title has not filled is not counted.
        asked = max(raw.x, max(raw.y, raw.z));
        if (ahead && asked > (1u << 24)) {
            asked = 0;
        }
        uint3 g;
        asked_x = raw.x;
        // Past 65,535 groups in x (one row): spread across z, the program
        // takes (x + z * row, y, 0) and leaves past `asked_x` (after
        // upstream VIctory pull request #2312).
        if (spread && raw.x > 65535u && raw.y >= 1u && raw.y <= 65535u && raw.z == 1u
            && !(ahead && raw.x > (1u << 24))) {
            uint rows = (raw.x + 65534u) / 65535u;
            row = (raw.x + rows - 1u) / rows;
            g = uint3(row, raw.y, rows);
        } else if (spread) {
            g = min(raw, uint3(65535u, 65535u, 65535u));
        } else {
            // Up to WIDEST groups in x run as asked (`widest`).
            g = min(raw, uint3(raw.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
        }
        // x * y fits 32 bits where it is looked at; the whole count may not.
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active && kind == 11) {
        // 1D indirect mesh dispatch (`DispatchMeshX`): 1 word in, 3 words (x, 1, 1) out.
        uint x = src_args.Load(src_at);
        if (x == 0u || x > 65535u) {
            groups = uint3(0, 0, 0);
        } else {
            asked = x;
            groups = uint3(x, 1u, 1u);
        }
        cost = groups.x * 128u;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // A record over the most a dispatch runs is a slot the title has
        // not filled (the title leaves all ones there): empty,
        // and not counted below either. (65,535 groups of meshlets read
        // through the garbage beside it read past every buffer.)
        if (g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
#if BUDGET_TRACK
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind >= 2) {
        // Diagnostics, in the budget buffer's next words (below): the
        // largest group count a mesh or compute record asked for, and the
        // groups asked in all.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    uint n = min(64u, eff_count - first);
    if (gi == 0) {
        uint group_cost = 0;
        for (uint ci = 0; ci < n; ci++) {
            group_cost |= costs[ci];
        }
        if (group_cost == 0) {
            for (uint k = 0; k < n; k++) {
                fits[k] = 1u;
            }
        } else {
            // What the records charge, from `before` on, in lane order: a
            // record that does not fit whole runs empty and takes the rest
            // (`rest`).
            // Only the group's own records are walked (one, clamped alone).
            uint before;
            budget.InterlockedOr(0, 0, before);
            [allow_uav_condition]
            while (before < BUDGET) {
                uint next = before;
                for (uint j = 0; j < n; j++) {
                    uint c = costs[j];
                    if (c > 0 && next < BUDGET) {
                        next = (BUDGET - next < c) ? (REST ? BUDGET : next) : (next + c);
                    }
                }
                if (next == before) break;
                uint prev;
                budget.InterlockedCompareExchange(0, before, next, prev);
                if (prev == before) break;
                before = prev;
            }
            uint at = before;
            uint refused = 0;
            uint biggest = 0;
            for (uint k = 0; k < n; k++) {
                uint c = costs[k];
                uint fit = 1;
                if (c > 0) {
                    fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                    at = (at >= BUDGET || BUDGET - at < c) ? (REST ? BUDGET : at) : (at + c);
                    refused += 1 - fit;
                    biggest = max(biggest, c);
                }
                fits[k] = fit;
            }
            // Diagnostics: the records that run empty for want of budget, and
            // the largest cost there was (small ones are not told).
            uint told;
            if (refused > 0) {
                budget.InterlockedAdd(12, refused, told);
            }
            if (biggest >= (1u << 22)) {
                budget.InterlockedMax(16, biggest, told);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind >= 2 && all_asked > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
#else
    bool over = false;
#endif
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (spread) {
            dst_args.Store2(dst_at, uint2(row, asked_x));
            dst_args.Store3(dst_at + 8u, groups);
        } else {
            dst_args.Store3(dst_at, groups);
            for (uint off = 12; off < stride; off += 4) {
                dst_args.Store(dst_at + off, src_args.Load(src_at + off));
            }
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_from(device, SOURCE)
}

/// The most groups in x that an indirect dispatch runs as the title asks:
/// 65,535 is the most D3D12 names for a dispatch, 2^20 ([`SOURCE`]'s GROUPS)
/// what is bounded here anyway (`VICTORY_INDIRECT_WIDE=0`: 65,535; a record
/// that asks for more still runs 65,535).
///
/// The title's tile lists are dispatches of one group per tile, and two of
/// them are classes of the picture's 8x4 tiles, 115,200 in all at
/// 2560x1440: in a captured frame 67,440 and 47,760 (the program
/// 2dc364df5d3278b8 and 940c2635dc1312f4). Cut to 65,535 the first list lost
/// its last 1,905 tiles, the lowest of the picture. The console runs what it
/// is asked (its dispatch counts are 32 bits), and so does an AMD card, whose
/// dispatch packets and group numbers are the same; D3D12 does not look at
/// indirect counts. Other cards stay at 65,535.
fn widest(amd: bool) -> u32 {
    let on = std::env::var_os("VICTORY_INDIRECT_WIDE").is_none_or(|v| v != "0");
    if amd && on { 1 << 20 } else { 65_535 }
}

/// Whether `device` is an AMD card's.
pub(crate) fn amd(device: &ID3D12Device) -> bool {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4};
    // SAFETY: plain queries.
    unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid()))
            .and_then(|a| a.GetDesc1())
            .is_ok_and(|d| d.VendorId == 0x1002)
    }
}

/// Whether a record that does not fit in what is left of its list's budget
/// takes the rest, so that every record after it runs empty too
/// (`VICTORY_INDIRECT_REST=1`; off by default so one oversized record takes
/// nothing and later valid draws still run).
fn rest() -> u32 {
    u32::from(std::env::var("VICTORY_INDIRECT_REST").is_ok_and(|v| v == "1"))
}

/// The clamp compiled from `source` (ours, or in the benchmark the one it
/// replaced).
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    make_from_with(device, source, budget_enabled())
}

fn make_from_with(
    device: &ID3D12Device,
    source: &str,
    budget_track: bool,
) -> windows::core::Result<Clamp> {
    let source = &format!(
        "#define WIDEST {}u\n#define REST {}\n#define BUDGET_TRACK {}\n{source}",
        widest(amd(device)),
        rest(),
        u32::from(budget_track),
    );
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
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
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        #[cfg(feature = "diagnostics")]
        crate::host::name(&pso, || crate::replay::play::CLAMP.into());
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`
/// (plus internal `KIND_DISPATCH_MESH_1D = 11` for 1-word `DispatchMeshX`
/// and `KIND_DISPATCH_SPREAD = 12` for batched spread dispatches).
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        KIND_DRAW => Some(16),
        KIND_DRAW_INDEXED => Some(20),
        KIND_DISPATCH | KIND_DISPATCH_MESH | KIND_DISPATCH_SPREAD => Some(12),
        KIND_DISPATCH_MESH_1D => Some(4),
        _ => None,
    }
}

/// Validates record alignment and returns the source byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Scratch output bytes required for `(kind, stride, records)` with optional `spread`.
fn scratch_bytes((kind, stride, records): (u32, u32, u32), spread: bool) -> Option<u64> {
    let dst_stride = if spread && kind == KIND_DISPATCH {
        20u32
    } else if kind == KIND_DISPATCH_MESH_1D {
        stride.max(12)
    } else {
        stride
    };
    u64::from(dst_stride)
        .checked_mul(u64::from(records))
        .filter(|&b| b <= u64::from(u32::MAX))
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, at offset `0`. The scratch buffer is left in `INDIRECT_ARGUMENT`
/// until the next clamp into it: executions read it with no barrier of
/// their own. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp or alignment is not
/// available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    spread: bool,
    any_in_place: bool,
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let dst_bytes = bytes.max(scratch_bytes(spec, spread)?);
    let c = clamp()?;
    let budget = if budget_enabled() {
        Some(list.budget_resource()?)
    } else {
        None
    };
    let dst = list.indirect_scratch(dst_bytes)?;
    clamp_into(
        list,
        c,
        &dst,
        budget.as_ref().map(|(res, gpu, dirty)| (res, *gpu, *dirty)),
        src,
        count,
        spec,
        false,
        spread,
        any_in_place,
    )
    .then_some(dst)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
    any_in_place: bool,
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let dst_bytes = bytes.max(scratch_bytes(spec, false)?);
    let c = clamp()?;
    let budget = if budget_enabled() {
        Some(list.budget_resource()?)
    } else {
        None
    };
    let dst = list.indirect_scratch(dst_bytes)?;
    clamp_into(
        list,
        c,
        &dst,
        budget.as_ref().map(|(res, gpu, dirty)| (res, *gpu, *dirty)),
        src,
        None,
        spec,
        true,
        false,
        any_in_place,
    )
    .then_some(dst)
}

/// [`record_clamp_as`] into `list`'s scratch buffer `dst`, from the state
/// the list left it in; afterwards the list knows it is readable as
/// indirect arguments.
#[allow(clippy::too_many_arguments)]
fn clamp_into(
    list: &HostList,
    c: &Clamp,
    dst: &ID3D12Resource,
    budget: Option<(&ID3D12Resource, u64, bool)>,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
    spread: bool,
    any_in_place: bool,
) -> bool {
    let before = list.indirect_state(dst);
    let enhanced = enhanced_clamp_list(list);
    let graphics_uav_dirty = list.is_graphics_uav_dirty();
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let budget_dirty = budget.as_ref().is_some_and(|(_, _, dirty)| *dirty);
    let plan = plan_clamp_barriers(
        scoped_clamp_barriers(),
        waits,
        list.is_uav_clean(),
        any_in_place,
        budget_dirty,
        true,
    );
    let recorded = record_clamp_with(
        c,
        &list.list,
        enhanced,
        graphics_uav_dirty,
        (dst, before),
        budget.map(|(res, gpu, _)| (Some(res), gpu)),
        src,
        count,
        spec,
        ahead,
        spread,
        plan,
    );
    if recorded {
        list.indirect_readable(dst);
        // Enhanced pre-clamp barriers only sync `-> COMPUTE_SHADING / SRV`, so
        // they never mark the whole list's UAV state clean for other stages.
        if enhanced.is_none() && plan.marks_uav_clean {
            list.mark_uav_clean();
        }
    }
    recorded
}

/// Records the clamp into `dst`, which is in state `before` (common, or
/// indirect arguments after an earlier clamp into it), and leaves `dst` in
/// `INDIRECT_ARGUMENT`. `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]). False, with nothing recorded, if the
/// layout or the addresses do not fit the clamp.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    let plan = plan_clamp_barriers(0, true, false, false, false, true);
    record_clamp_with(
        c,
        l,
        None,
        false,
        dst,
        Some((None, budget)),
        src,
        count,
        spec,
        ahead,
        false,
        plan,
    )
}

/// [`record_clamp_as`]; `spread`: dispatch records past 65,535 groups in x
/// are spread across z, every record written as [row, asked x, x, y, z]
/// (20 bytes) for `crate::draw`'s spread command signature.
#[allow(clippy::too_many_arguments)]
fn record_clamp_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    enhanced: Option<&ID3D12GraphicsCommandList7>,
    graphics_uav_dirty: bool,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: Option<(Option<&ID3D12Resource>, u64)>,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
    spread: bool,
    barrier_plan: ClampBarrierPlan,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    if scratch_bytes((kind, stride, records), spread).is_none() {
        return false;
    }
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4)
        || !dst_gpu.is_multiple_of(4)
        || budget.is_some_and(|(_, b)| !b.is_multiple_of(4))
    {
        return false;
    }
    if records == 0 {
        if enhanced.is_none() && before != D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
            // SAFETY: an open list and a live scratch buffer.
            unsafe {
                l.ResourceBarrier(&[transition(
                    dst,
                    before,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
            }
        }
        return true;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [
        kind,
        stride,
        records,
        args_offset,
        has_count | (u32::from(ahead) << 1) | (u32::from(spread && kind == 2) << 3),
        count_offset,
    ];
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);

    if let Some(list7) = enhanced {
        let sync_before = if graphics_uav_dirty {
            D3D12_BARRIER_SYNC_ALL_SHADING
        } else {
            D3D12_BARRIER_SYNC_COMPUTE_SHADING
        };
        let mut buf_barriers: [D3D12_BUFFER_BARRIER; 4] =
            std::array::from_fn(|_| D3D12_BUFFER_BARRIER::default());
        let mut buf_len = 0usize;
        let mut global_barrier = [D3D12_GLOBAL_BARRIER::default()];
        let mut has_global = false;
        match barrier_plan.pre_inputs {
            PreInputUav::None => {}
            PreInputUav::Global => {
                global_barrier[0] = D3D12_GLOBAL_BARRIER {
                    SyncBefore: sync_before,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                };
                has_global = true;
            }
            PreInputUav::Scoped => {
                if uav_capable(src) {
                    buf_barriers[buf_len] = enhanced_buffer_barrier(
                        src,
                        sync_before,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    );
                    buf_len += 1;
                }
                if let Some((count_res, _)) = count
                    && windows::core::Interface::as_raw(count_res)
                        != windows::core::Interface::as_raw(src)
                    && uav_capable(count_res)
                {
                    buf_barriers[buf_len] = enhanced_buffer_barrier(
                        count_res,
                        sync_before,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    );
                    buf_len += 1;
                }
            }
        }
        if barrier_plan.pre_budget
            && let Some((budget_res, _)) = budget
        {
            if let Some(res) = budget_res {
                buf_barriers[buf_len] = enhanced_buffer_barrier(
                    res,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                );
                buf_len += 1;
            } else if !has_global {
                global_barrier[0] = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                };
                has_global = true;
            }
        }
        if before == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
            buf_barriers[buf_len] = enhanced_buffer_barrier(
                dst,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            );
            buf_len += 1;
        }
        let mut pre_groups: [D3D12_BARRIER_GROUP; 2] =
            std::array::from_fn(|_| D3D12_BARRIER_GROUP::default());
        let mut pre_len = 0usize;
        if has_global {
            pre_groups[pre_len] = D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: global_barrier.as_ptr(),
                },
            };
            pre_len += 1;
        }
        if buf_len > 0 {
            pre_groups[pre_len] = D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_BUFFER,
                NumBarriers: buf_len as u32,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pBufferBarriers: buf_barriers.as_ptr(),
                },
            };
            pre_len += 1;
        }
        let post_buf = [enhanced_buffer_barrier(
            dst,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
        )];
        let post_group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: post_buf.as_ptr(),
            },
        }];
        // SAFETY: an open list; our root signature and pipeline; live buffers
        // kept in COMMON for Enhanced Barriers.
        unsafe {
            if pre_len > 0 {
                list7.Barrier(&pre_groups[..pre_len]);
            }
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            if let Some((_, budget_gpu)) = budget {
                l.SetComputeRootUnorderedAccessView(4, budget_gpu);
            }
            l.Dispatch(records.div_ceil(64), 1, 1);
            list7.Barrier(&post_group);
        }
        return true;
    }

    let mut first: [D3D12_RESOURCE_BARRIER; 4] =
        std::array::from_fn(|_| D3D12_RESOURCE_BARRIER::default());
    let mut first_len = 0usize;
    match barrier_plan.pre_inputs {
        PreInputUav::None => {}
        PreInputUav::Global => {
            first[first_len] = uav_barrier();
            first_len += 1;
        }
        PreInputUav::Scoped => {
            first[first_len] = uav_barrier_for(src);
            first_len += 1;
            if let Some((count_res, _)) = count
                && windows::core::Interface::as_raw(count_res)
                    != windows::core::Interface::as_raw(src)
            {
                first[first_len] = uav_barrier_for(count_res);
                first_len += 1;
            }
        }
    }
    if barrier_plan.pre_budget
        && let Some((budget_res, _)) = budget
    {
        first[first_len] = match budget_res {
            Some(res) => uav_barrier_for(res),
            None => uav_barrier(),
        };
        first_len += 1;
    }
    first[first_len] = transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    first_len += 1;
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let second_slice = if barrier_plan.post_global_uav {
        &second[..]
    } else {
        &second[1..]
    };
    // SAFETY: an open list; our root signature and pipeline; live buffers.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[..first_len]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        if let Some((_, budget_gpu)) = budget {
            l.SetComputeRootUnorderedAccessView(4, budget_gpu);
        }
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(second_slice);
    }
    true
}

/// The batched clamp's pipeline ([`batch`]).
fn batch_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let made = host().ok_or_else(windows::core::Error::empty).and_then(|h| make_from(&h.device, batch::SOURCE));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no batched argument clamp: records are clamped one at a time");
                    None
                }
            }
        })
        .as_ref()
}

/// Opens a batch in `list` ([`batch`]): a pass that clamps the records of
/// the executions recorded after it, of `source` (a host copy's number,
/// buffer, and whether it is placed in place) and of the `recent` ones after
/// it. Reuses `table` in place (avoiding 4.6 KiB stack zeroing on every batch
/// open) and returns the scratch buffer its places are in.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp is not available.
pub(crate) fn open_batch_into(
    list: &HostList,
    source: (u64, &ID3D12Resource, bool),
    recent: &[(u64, ID3D12Resource, bool)],
    table: &mut batch::Table,
) -> Option<ID3D12Resource> {
    let c = batch_clamp()?;
    let budget = if budget_enabled() {
        Some(list.budget_resource()?)
    } else {
        None
    };
    let (jobs, at) = list.scratch(batch::TABLE)?;
    let (dst, base, before) = list.batch_place()?;
    let gpu = |b: &ID3D12Resource| {
        // SAFETY: a live buffer.
        Some(unsafe { b.GetGPUVirtualAddress() }).filter(|at| at.is_multiple_of(4))
    };
    // A source there is none of is the first again, which no job names.
    let mut ids = [0u64; batch::SOURCES];
    let mut sources = [gpu(source.1)?; batch::SOURCES];
    let mut unique_sources: [&ID3D12Resource; batch::SOURCES] = [source.1; batch::SOURCES];
    let mut unique_len = 1usize;
    // Do not poison a non-in-place batch by pulling in-place MRU buffers into
    // unused SRV slots: only include in-place MRU slots when slot 0 itself is
    // already in-place.
    let any_in_place = source.2;
    list.keep(source.1);
    ids[0] = source.0;
    let others = recent
        .iter()
        .filter(|r| r.0 != source.0 && (any_in_place || !r.2))
        .take(batch::SOURCES - 1);
    for (i, (id, buffer, _)) in others.enumerate() {
        if let Some(at) = gpu(buffer) {
            // Bound, and read only if a job names it: kept alive like the
            // copies draws read (`Shadow::refresh`).
            list.keep(buffer);
            (ids[i + 1], sources[i + 1]) = (*id, at);
            let raw = windows::core::Interface::as_raw(buffer);
            if !unique_sources[..unique_len]
                .iter()
                .any(|s| windows::core::Interface::as_raw(*s) == raw)
            {
                unique_sources[unique_len] = buffer;
                unique_len += 1;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let enhanced = enhanced_clamp_list(list);
    let graphics_uav_dirty = list.is_graphics_uav_dirty();
    let budget_dirty = budget.as_ref().is_some_and(|(_, _, dirty)| *dirty);
    let plan = plan_clamp_barriers(
        scoped_clamp_barriers(),
        waits,
        list.is_uav_clean(),
        any_in_place,
        budget_dirty,
        false,
    );
    batch::record(
        c,
        &list.list,
        enhanced,
        graphics_uav_dirty,
        (&dst, before),
        budget.as_ref().map(|(res, gpu, _)| (Some(res), *gpu)),
        sources,
        &unique_sources[..unique_len],
        jobs,
        plan,
    );
    if enhanced.is_none() && plan.marks_uav_clean {
        list.mark_uav_clean();
    }
    // SAFETY: the list's own upload memory of that many bytes, 256-byte
    // aligned, until it is reset.
    unsafe { table.reset(at, base, ids) };
    Some(dst)
}

/// [`open_batch_into`] returning a freshly constructed [`batch::Table`].
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn open_batch(
    list: &HostList,
    source: (u64, &ID3D12Resource, bool),
    recent: &[(u64, ID3D12Resource, bool)],
) -> Option<(batch::Table, ID3D12Resource)> {
    let mut table = batch::Table::default();
    let dst = open_batch_into(list, source, recent, &mut table)?;
    Some((table, dst))
}

/// The batched clamp's root signature and pipeline on `device`, for the
/// replayer.
#[cfg(feature = "diagnostics")]
pub(crate) fn batch_for(device: &ID3D12Device) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
    make_from_with(device, batch::SOURCE, true).map(|c| (c.root, c.pso))
}

pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

pub(crate) fn uav_barrier_for(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

/// Translated stage bindings needed to map console indirect root arguments to
/// host root parameters.
pub(crate) struct StageBinding<'a> {
    pub(crate) user_data_param: Option<u32>,
    pub(crate) user_data: &'a [victory_shader::emit::UserData],
    pub(crate) root_cbvs: &'a [(u8, u32)],
    pub(crate) root_srvs: &'a [(u8, u32)],
    pub(crate) root_uavs: &'a [(u8, u32)],
}

/// A single argument in a translated host multi-argument command signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum HostIndirectArg {
    Constant {
        param: u32,
        dest_offset: u32,
        count: u32,
    },
    Cbv {
        param: u32,
    },
    Srv {
        param: u32,
    },
    Uav {
        param: u32,
    },
    Exec {
        kind: u32,
    },
}

/// Translated multi-argument indirect execution plan: host command signature
/// layout plus the per-record copy/VA-rebase ops for [`REPACK_SOURCE`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MultiArgPlan {
    pub(crate) exec_kind: u32,
    pub(crate) src_stride: u32,
    pub(crate) dst_stride: u32,
    pub(crate) src_exec_off: u32,
    pub(crate) dst_exec_off: u32,
    pub(crate) host_args: Vec<HostIndirectArg>,
    /// Packed `[op_kind, src_off, dst_off, count]` ops for [`REPACK_SOURCE`]:
    /// 0 = copy 1 dword, 1 = copy `count` dwords, 2 = rebase CBV VA,
    /// 3 = rebase SRV VA, 4 = rebase UAV VA.
    pub(crate) ops: Vec<[u32; 4]>,
    pub(crate) needs_rebase: bool,
    pub(crate) has_root_args: bool,
    pub(crate) skipped_tabled_views: bool,
}

const OP_COPY_DWORD: u32 = 0;
const OP_COPY_DWORDS: u32 = 1;
const OP_REBASE_CBV: u32 = 2;
const OP_REBASE_SRV: u32 = 3;
const OP_REBASE_UAV: u32 = 4;

fn console_arg_bytes(arg: [u32; 4]) -> Option<u32> {
    match arg[0] {
        KIND_DRAW => Some(16),
        KIND_DRAW_INDEXED => Some(20),
        KIND_DISPATCH | KIND_DISPATCH_SINGLE | KIND_DISPATCH_MESH => Some(12),
        KIND_CONSTANT => arg[3].checked_mul(4),
        KIND_CBV | KIND_SRV | KIND_UAV => Some(8),
        _ => None,
    }
}

/// Translates a console multi-argument command signature (`stride`, `args`)
/// against the bound pipeline's `stages` into a packed host command signature
/// and repack op list.
///
/// Supports any prefix of `CONSTANT` (5), `CBV` (6), `SRV` (7), `UAV` (8)
/// followed by a single final `DRAW` (0), `DRAW_INDEXED` (1), `DISPATCH` (2 or
/// 105), or `DISPATCH_MESH` (10). Returns `None` for unsupported argument types
/// or invalid strides/layouts.
pub(crate) fn translate_signature(
    stride: u32,
    args: &[[u32; 4]],
    stages: &[StageBinding<'_>],
    bounded_root_srvs: bool,
) -> Option<MultiArgPlan> {
    use victory_shader::RootType;
    use victory_shader::emit::USER_ROOT_ADDRESS;

    if args.is_empty() || !stride.is_multiple_of(4) {
        return None;
    }
    let (&last, prefix) = args.split_last()?;
    let exec_kind = match last[0] {
        KIND_DRAW => KIND_DRAW,
        KIND_DRAW_INDEXED => KIND_DRAW_INDEXED,
        KIND_DISPATCH | KIND_DISPATCH_SINGLE => KIND_DISPATCH,
        KIND_DISPATCH_MESH => KIND_DISPATCH_MESH,
        _ => return None,
    };
    let exec_bytes = console_arg_bytes(last)?;

    // Walk prefix arguments to compute exact console record byte offsets and
    // collect every host constant / root descriptor target.
    let mut src_off = 0u32;
    // (host_param, dst_dword, src_byte_off)
    let mut const_writes: Vec<(u32, u32, u32)> = Vec::new();
    // (op_kind [CBV=2, SRV=3, UAV=4], host_param, src_byte_off)
    let mut desc_writes: Vec<(u32, u32, u32)> = Vec::new();
    let mut skipped_tabled_views = false;

    for &arg in prefix {
        let arg_bytes = console_arg_bytes(arg)?;
        match arg[0] {
            KIND_CONSTANT => {
                let console_slot = arg[1];
                let dest_offset = arg[2];
                let num_dwords = arg[3];
                let end_dword = dest_offset.checked_add(num_dwords)?;
                for stage in stages {
                    let Some(host_param) = stage.user_data_param else {
                        continue;
                    };
                    for u in stage.user_data {
                        if u.root == RootType::Constants
                            && u32::from(u.slot) == console_slot
                            && u.dword >= dest_offset
                            && u.dword < end_dword
                        {
                            let word_src = src_off.checked_add((u.dword - dest_offset).checked_mul(4)?)?;
                            const_writes.push((host_param, u.index, word_src));
                        }
                    }
                }
            }
            KIND_CBV | KIND_SRV | KIND_UAV => {
                let console_slot = arg[1];
                // Shaders that read raw guest root addresses via user_data get
                // the low/high 32 bits straight from the indirect record.
                for stage in stages {
                    if let Some(host_param) = stage.user_data_param {
                        for u in stage.user_data {
                            if u32::from(u.slot) == console_slot {
                                if u.root == RootType::Other(USER_ROOT_ADDRESS) {
                                    const_writes.push((host_param, u.index, src_off));
                                } else if u.root == RootType::Other(USER_ROOT_ADDRESS + 1) {
                                    const_writes.push((host_param, u.index, src_off.checked_add(4)?));
                                }
                            }
                        }
                    }
                    match arg[0] {
                        KIND_CBV => {
                            for &(slot, host_param) in stage.root_cbvs {
                                if u32::from(slot) == console_slot {
                                    desc_writes.push((OP_REBASE_CBV, host_param, src_off));
                                }
                            }
                        }
                        KIND_SRV => {
                            for &(slot, host_param) in stage.root_srvs {
                                if u32::from(slot) == console_slot {
                                    if bounded_root_srvs {
                                        skipped_tabled_views = true;
                                    } else {
                                        desc_writes.push((OP_REBASE_SRV, host_param, src_off));
                                    }
                                }
                            }
                        }
                        KIND_UAV => {
                            for &(slot, host_param) in stage.root_uavs {
                                if u32::from(slot) == console_slot {
                                    if bounded_root_srvs {
                                        skipped_tabled_views = true;
                                    } else {
                                        desc_writes.push((OP_REBASE_UAV, host_param, src_off));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => return None,
        }
        src_off = src_off.checked_add(arg_bytes)?;
    }

    let src_exec_off = src_off;
    let total_src = src_exec_off.checked_add(exec_bytes)?;
    if stride < total_src {
        return None;
    }

    // Deduplicate constant writes by `(host_param, dst_dword)` keeping the last
    // console argument that wrote that slot, then sort by `(host_param, dst_dword)`.
    let mut dedup_const: Vec<(u32, u32, u32)> = Vec::with_capacity(const_writes.len());
    for (param, dst, src) in const_writes {
        if let Some(existing) = dedup_const.iter_mut().find(|(p, d, _)| *p == param && *d == dst) {
            existing.2 = src;
        } else {
            dedup_const.push((param, dst, src));
        }
    }
    dedup_const.sort_unstable_by_key(|&(p, d, _)| (p, d));

    // Deduplicate descriptor writes by `(op_kind, host_param)` keeping the last.
    let mut dedup_desc: Vec<(u32, u32, u32)> = Vec::with_capacity(desc_writes.len());
    for (kind, param, src) in desc_writes {
        if let Some(existing) = dedup_desc.iter_mut().find(|(k, p, _)| *k == kind && *p == param) {
            existing.2 = src;
        } else {
            dedup_desc.push((kind, param, src));
        }
    }
    dedup_desc.sort_unstable_by_key(|&(k, p, _)| (k, p));

    let mut host_args = Vec::new();
    let mut ops = Vec::new();
    let mut dst_off = 0u32;

    // Coalesce contiguous `(host_param, dst_dword)` runs into single
    // `D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT` entries and contiguous source
    // runs into `OP_COPY_DWORDS`.
    let mut i = 0;
    while i < dedup_const.len() {
        let param = dedup_const[i].0;
        let run_start = dedup_const[i].1;
        let mut j = i + 1;
        while j < dedup_const.len()
            && dedup_const[j].0 == param
            && dedup_const[j].1 == run_start + (j - i) as u32
        {
            j += 1;
        }
        let count = (j - i) as u32;
        host_args.push(HostIndirectArg::Constant {
            param,
            dest_offset: run_start,
            count,
        });
        let mut k = i;
        while k < j {
            let src0 = dedup_const[k].2;
            let dst0 = dst_off;
            let mut run = 1u32;
            while k + (run as usize) < j && dedup_const[k + run as usize].2 == src0 + run * 4 {
                run += 1;
            }
            if run == 1 {
                ops.push([OP_COPY_DWORD, src0, dst0, 1]);
            } else {
                ops.push([OP_COPY_DWORDS, src0, dst0, run]);
            }
            dst_off = dst_off.checked_add(run.checked_mul(4)?)?;
            k += run as usize;
        }
        i = j;
    }

    let needs_rebase = !dedup_desc.is_empty();
    for (op_kind, param, desc_src) in dedup_desc {
        let arg = match op_kind {
            OP_REBASE_CBV => HostIndirectArg::Cbv { param },
            OP_REBASE_SRV => HostIndirectArg::Srv { param },
            OP_REBASE_UAV => HostIndirectArg::Uav { param },
            _ => unreachable!(),
        };
        host_args.push(arg);
        ops.push([op_kind, desc_src, dst_off, 0]);
        dst_off = dst_off.checked_add(8)?;
    }

    let has_root_args = !host_args.is_empty();
    let dst_exec_off = dst_off;
    host_args.push(HostIndirectArg::Exec { kind: exec_kind });
    let dst_stride = dst_exec_off.checked_add(exec_bytes)?;

    Some(MultiArgPlan {
        exec_kind,
        src_stride: stride,
        dst_stride,
        src_exec_off,
        dst_exec_off,
        host_args,
        ops,
        needs_rebase,
        has_root_args,
        skipped_tabled_views,
    })
}

/// Builds a sorted, non-overlapping GPU VA-rebase table for [`REPACK_SOURCE`]
/// from `(guest_start, len, host_gpu, writable)` shadow ranges.
///
/// When two ranges overlap (e.g. a sub-buffer inside a larger buffer), the
/// smaller (innermost) range takes precedence and the outer range is split
/// around it so GPU binary search is exact in O(log N).
pub(crate) fn build_rebase_table(ranges: &[(u64, u64, u64, bool)]) -> Vec<[u32; 8]> {
    #[derive(Clone, Copy)]
    struct Span {
        start: u64,
        end: u64,
        host: u64,
        writable: bool,
    }
    let mut valid: Vec<Span> = ranges
        .iter()
        .filter_map(|&(start, len, host, writable)| {
            if start == 0 || len == 0 || host == 0 {
                return None;
            }
            let end = start.checked_add(len)?;
            host.checked_add(len)?;
            Some(Span { start, end, host, writable })
        })
        .collect();
    if valid.is_empty() {
        return Vec::new();
    }
    // Sort largest first, then split/overlay smaller (innermost) spans on top
    // so narrower buffers win on overlapping regions.
    valid.sort_unstable_by(|a, b| {
        (b.end - b.start)
            .cmp(&(a.end - a.start))
            .then_with(|| a.start.cmp(&b.start))
    });
    // Fast path when sorted by start has no overlaps at all.
    let mut by_start = valid.clone();
    by_start.sort_unstable_by_key(|s| s.start);
    let has_overlap = by_start.windows(2).any(|w| w[0].end > w[1].start);
    let resolved: Vec<Span> = if !has_overlap {
        by_start
    } else {
        let mut pieces: Vec<Span> = Vec::with_capacity(valid.len());
        for span in valid {
            let mut next = Vec::with_capacity(pieces.len() + 2);
            for p in pieces {
                if p.end <= span.start || p.start >= span.end {
                    next.push(p);
                } else {
                    if p.start < span.start {
                        next.push(Span {
                            start: p.start,
                            end: span.start,
                            host: p.host,
                            writable: p.writable,
                        });
                    }
                    if p.end > span.end {
                        next.push(Span {
                            start: span.end,
                            end: p.end,
                            host: p.host + (span.end - p.start),
                            writable: p.writable,
                        });
                    }
                }
            }
            next.push(span);
            pieces = next;
        }
        pieces.sort_unstable_by_key(|s| s.start);
        pieces
    };
    resolved
        .into_iter()
        .map(|s| {
            [
                s.start as u32,
                (s.start >> 32) as u32,
                s.end as u32,
                (s.end >> 32) as u32,
                s.host as u32,
                (s.host >> 32) as u32,
                u32::from(s.writable),
                0,
            ]
        })
        .collect()
}

const REPACK_SOURCE: &str = r#"
#if BUDGET_TRACK
#define RS "RootConstants(num32BitConstants=13, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), UAV(u0), UAV(u1)"
#else
#define RS "RootConstants(num32BitConstants=13, b0), SRV(t0), SRV(t1), SRV(t2), SRV(t3), UAV(u0)"
#endif
cbuffer Repack : register(b0) {
    uint kind;
    uint src_stride;
    uint dst_stride;
    uint records;
    uint src_exec_off;
    uint dst_exec_off;
    uint args_offset;
    uint flags;
    uint count_offset;
    uint op_count;
    uint rebase_count;
    uint fallback_lo;
    uint fallback_hi;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
ByteAddressBuffer ops : register(t2);
ByteAddressBuffer rebase : register(t3);
RWByteAddressBuffer dst_args : register(u0);
#if BUDGET_TRACK
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
#endif
static const uint MOST_DRAW = 1u << 23;
static const uint GROUPS = 1u << 20;
#if BUDGET_TRACK
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
#endif

bool u64_lt(uint2 a, uint2 b) {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
}
bool u64_ge(uint2 a, uint2 b) {
    return (a.y > b.y) || (a.y == b.y && a.x >= b.x);
}
uint2 u64_sub(uint2 a, uint2 b) {
    uint lo = a.x - b.x;
    uint borrow = (a.x < b.x) ? 1u : 0u;
    uint hi = a.y - b.y - borrow;
    return uint2(lo, hi);
}
uint2 u64_add(uint2 a, uint2 b) {
    uint lo = a.x + b.x;
    uint carry = (lo < a.x) ? 1u : 0u;
    uint hi = a.y + b.y + carry;
    return uint2(lo, hi);
}

bool rebase_va(uint2 guest, uint op_kind, out uint2 out_va) {
    out_va = uint2(fallback_lo, fallback_hi);
    if (rebase_count == 0u || (guest.x == 0u && guest.y == 0u)) {
        return false;
    }
    uint lo = 0u;
    uint hi = rebase_count;
    [loop]
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        uint4 r0 = rebase.Load4(mid * 32u);
        uint2 g_start = r0.xy;
        uint2 g_end = r0.zw;
        if (u64_lt(guest, g_start)) {
            hi = mid;
        } else if (u64_ge(guest, g_end)) {
            lo = mid + 1u;
        } else {
            uint4 r1 = rebase.Load4(mid * 32u + 16u);
            if (op_kind == 4u && (r1.z & 1u) == 0u) {
                return false;
            }
            uint2 host_va = u64_add(r1.xy, u64_sub(guest, g_start));
            uint align_mask = (op_kind == 2u) ? 255u : 3u;
            if ((host_va.x & align_mask) != 0u) {
                return false;
            }
            out_va = host_va;
            return true;
        }
    }
    return false;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    if ((flags & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
#if !BUDGET_TRACK
    if (id.x >= eff_count) {
        return;
    }
    bool active = true;
#else
    uint first = gid.x * 64u;
    if (first >= eff_count) {
        return;
    }
    bool active = id.x < eff_count;
#endif
    uint src_at = args_offset + id.x * src_stride;
    uint dst_at = id.x * dst_stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    bool record_valid = true;
#if BUDGET_TRACK
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
#endif
    if (active) {
        [loop]
        for (uint oi = 0; oi < op_count; oi++) {
            uint4 op = ops.Load4(oi * 16u);
            if (op.x == 0u) {
                dst_args.Store(dst_at + op.z, src_args.Load(src_at + op.y));
            } else if (op.x == 1u) {
                [loop]
                for (uint w = 0; w < op.w; w++) {
                    dst_args.Store(dst_at + op.z + w * 4u, src_args.Load(src_at + op.y + w * 4u));
                }
            } else {
                uint2 guest_va = src_args.Load2(src_at + op.y);
                uint2 host_va;
                if (!rebase_va(guest_va, op.x, host_va)) {
                    record_valid = false;
                }
                dst_args.Store2(dst_at + op.z, host_va);
            }
        }
        uint exec_src = src_at + src_exec_off;
        if (!record_valid) {
            // An unmapped/unaligned root descriptor forces a no-op execution
            // so the GPU never faults on an invalid VA.
            cost = 0;
        } else if (kind <= 1u) {
            uint2 want = src_args.Load2(exec_src);
            if (want.x >= MOST_DRAW) {
                want = uint2(0, 0);
            }
            uint count = want.x;
            uint instances = min(want.y, (1u << 26) / max(count, 1u));
            cost = count * instances;
            draw_counts = uint2(count, instances);
        } else if (kind == 2u) {
            uint3 raw = src_args.Load3(exec_src);
            asked = max(raw.x, max(raw.y, raw.z));
            uint3 g = min(raw, uint3(raw.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
            if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            groups = g;
        } else {
            uint3 g = src_args.Load3(exec_src);
            if (g.x > 65535u) {
                g = uint3(0, 0, 0);
            }
            asked = g.x;
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128u;
            groups = g;
        }
    }
#if BUDGET_TRACK
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind >= 2u) {
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    uint n = min(64u, eff_count - first);
    if (gi == 0) {
        uint group_cost = 0;
        for (uint ci = 0; ci < n; ci++) {
            group_cost |= costs[ci];
        }
        if (group_cost == 0) {
            for (uint k = 0; k < n; k++) {
                fits[k] = 1u;
            }
        } else {
            uint before;
            budget.InterlockedOr(0, 0, before);
            [allow_uav_condition]
            while (before < BUDGET) {
                uint next = before;
                for (uint j = 0; j < n; j++) {
                    uint c = costs[j];
                    if (c > 0 && next < BUDGET) {
                        next = (BUDGET - next < c) ? (REST ? BUDGET : next) : (next + c);
                    }
                }
                if (next == before) break;
                uint prev;
                budget.InterlockedCompareExchange(0, before, next, prev);
                if (prev == before) break;
                before = prev;
            }
            uint at = before;
            uint refused = 0;
            uint biggest = 0;
            for (uint k = 0; k < n; k++) {
                uint c = costs[k];
                uint fit = 1;
                if (c > 0) {
                    fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                    at = (at >= BUDGET || BUDGET - at < c) ? (REST ? BUDGET : at) : (at + c);
                    refused += 1 - fit;
                    biggest = max(biggest, c);
                }
                fits[k] = fit;
            }
            uint told;
            if (refused > 0) {
                budget.InterlockedAdd(12, refused, told);
            }
            if (biggest >= (1u << 22)) {
                budget.InterlockedMax(16, biggest, told);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind >= 2u && all_asked > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    bool over = !record_valid || fits[gi] == 0;
#else
    bool over = !record_valid;
#endif
    uint exec_src = src_at + src_exec_off;
    uint exec_dst = dst_at + dst_exec_off;
    if (kind <= 1u) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(exec_dst, draw_counts);
        uint tail = (kind == 0u) ? 16u : 20u;
        for (uint off = 8u; off < tail; off += 4u) {
            dst_args.Store(exec_dst + off, src_args.Load(exec_src + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(exec_dst, groups);
    }
}
"#;

fn repack_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let made = host()
                .ok_or_else(windows::core::Error::empty)
                .and_then(|h| make_from(&h.device, REPACK_SOURCE));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no multi-argument repack clamp: multi-arg indirect dropped");
                    None
                }
            }
        })
        .as_ref()
}

/// Repacks and clamps up to `records` multi-argument console indirect records
/// into `list`'s indirect scratch buffer according to `plan` and `rebase`.
/// Leaves the scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
pub(crate) fn repack(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    plan: &MultiArgPlan,
    rebase: &[[u32; 8]],
    records: u32,
    any_in_place: bool,
) -> Option<ID3D12Resource> {
    if !src.1.is_multiple_of(4) || count.is_some_and(|(_, off)| !off.is_multiple_of(4)) {
        return None;
    }
    let src_bytes = u64::from(plan.src_stride)
        .checked_mul(u64::from(records))
        .filter(|&b| b <= u64::from(u32::MAX))?;
    let dst_bytes = u64::from(plan.dst_stride)
        .checked_mul(u64::from(records))
        .filter(|&b| b <= u64::from(u32::MAX))?;
    let c = repack_clamp()?;
    let budget = if budget_enabled() {
        Some(list.budget_resource()?)
    } else {
        None
    };
    let dst = list.indirect_scratch(dst_bytes)?;
    let ops_bytes = ((plan.ops.len().max(1)) * 16) as u64;
    let (ops_gpu, ops_ptr) = list.scratch(ops_bytes)?;
    // SAFETY: `ops_ptr` is freshly allocated mapped upload memory of `ops_bytes`.
    unsafe {
        if plan.ops.is_empty() {
            std::ptr::write_bytes(ops_ptr, 0, 16);
        } else {
            std::ptr::copy_nonoverlapping(
                plan.ops.as_ptr().cast::<u8>(),
                ops_ptr,
                plan.ops.len() * 16,
            );
        }
    }
    let rebase_bytes = ((rebase.len().max(1)) * 32) as u64;
    let (rebase_gpu, rebase_ptr) = list.scratch(rebase_bytes)?;
    // SAFETY: `rebase_ptr` is freshly allocated mapped upload memory of `rebase_bytes`.
    unsafe {
        if rebase.is_empty() {
            std::ptr::write_bytes(rebase_ptr, 0, 32);
        } else {
            std::ptr::copy_nonoverlapping(
                rebase.as_ptr().cast::<u8>(),
                rebase_ptr,
                rebase.len() * 32,
            );
        }
    }
    let before = list.indirect_state(&dst);
    let enhanced = enhanced_clamp_list(list);
    let graphics_uav_dirty = list.is_graphics_uav_dirty();
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let budget_dirty = budget.as_ref().is_some_and(|(_, _, dirty)| *dirty);
    let barrier_plan = plan_clamp_barriers(
        scoped_clamp_barriers(),
        waits,
        list.is_uav_clean(),
        any_in_place,
        budget_dirty,
        true,
    );
    let recorded = record_repack_with(
        c,
        &list.list,
        enhanced,
        graphics_uav_dirty,
        (&dst, before),
        budget.as_ref().map(|(res, gpu, _)| (Some(res), *gpu)),
        src,
        src_bytes,
        count,
        plan,
        (ops_gpu, plan.ops.len() as u32),
        (rebase_gpu, rebase.len() as u32),
        records,
        barrier_plan,
    );
    if recorded {
        list.indirect_readable(&dst);
        if enhanced.is_none() && barrier_plan.marks_uav_clean {
            list.mark_uav_clean();
        }
        Some(dst)
    } else {
        None
    }
}

#[allow(clippy::too_many_arguments)]
fn record_repack_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    enhanced: Option<&ID3D12GraphicsCommandList7>,
    graphics_uav_dirty: bool,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: Option<(Option<&ID3D12Resource>, u64)>,
    (src, src_offset): (&ID3D12Resource, u64),
    src_bytes: u64,
    count: Option<(&ID3D12Resource, u64)>,
    plan: &MultiArgPlan,
    (ops_gpu, op_count): (u64, u32),
    (rebase_gpu, rebase_count): (u64, u32),
    records: u32,
    barrier_plan: ClampBarrierPlan,
) -> bool {
    if !src_offset.is_multiple_of(4) || count.is_some_and(|(_, off)| !off.is_multiple_of(4)) {
        return false;
    }
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4)
        || !dst_gpu.is_multiple_of(4)
        || budget.is_some_and(|(_, b)| !b.is_multiple_of(4))
        || !ops_gpu.is_multiple_of(4)
        || !rebase_gpu.is_multiple_of(4)
    {
        return false;
    }
    if records == 0 {
        if enhanced.is_none() && before != D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
            // SAFETY: an open list and live scratch buffer.
            unsafe {
                l.ResourceBarrier(&[transition(
                    dst,
                    before,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
            }
        }
        return true;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(src_bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, flags, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: live buffer.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    // `dst_gpu` is a committed D3D12 buffer (64 KiB aligned), so it is always a
    // valid 256-byte-aligned fallback VA for zeroed invalid records.
    let constants = [
        plan.exec_kind,
        plan.src_stride,
        plan.dst_stride,
        records,
        plan.src_exec_off,
        plan.dst_exec_off,
        args_offset,
        flags,
        count_offset,
        op_count,
        rebase_count,
        dst_gpu as u32,
        (dst_gpu >> 32) as u32,
    ];
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);

    if let Some(list7) = enhanced {
        let sync_before = if graphics_uav_dirty {
            D3D12_BARRIER_SYNC_ALL_SHADING
        } else {
            D3D12_BARRIER_SYNC_COMPUTE_SHADING
        };
        let mut buf_barriers: [D3D12_BUFFER_BARRIER; 4] =
            std::array::from_fn(|_| D3D12_BUFFER_BARRIER::default());
        let mut buf_len = 0usize;
        let mut global_barrier = [D3D12_GLOBAL_BARRIER::default()];
        let mut has_global = false;
        match barrier_plan.pre_inputs {
            PreInputUav::None => {}
            PreInputUav::Global => {
                global_barrier[0] = D3D12_GLOBAL_BARRIER {
                    SyncBefore: sync_before,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                };
                has_global = true;
            }
            PreInputUav::Scoped => {
                if uav_capable(src) {
                    buf_barriers[buf_len] = enhanced_buffer_barrier(
                        src,
                        sync_before,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    );
                    buf_len += 1;
                }
                if let Some((count_res, _)) = count
                    && windows::core::Interface::as_raw(count_res)
                        != windows::core::Interface::as_raw(src)
                    && uav_capable(count_res)
                {
                    buf_barriers[buf_len] = enhanced_buffer_barrier(
                        count_res,
                        sync_before,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    );
                    buf_len += 1;
                }
            }
        }
        if barrier_plan.pre_budget
            && let Some((budget_res, _)) = budget
        {
            if let Some(res) = budget_res {
                buf_barriers[buf_len] = enhanced_buffer_barrier(
                    res,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                );
                buf_len += 1;
            } else if !has_global {
                global_barrier[0] = D3D12_GLOBAL_BARRIER {
                    SyncBefore: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    SyncAfter: D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    AccessAfter: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                };
                has_global = true;
            }
        }
        if before == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
            buf_barriers[buf_len] = enhanced_buffer_barrier(
                dst,
                D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            );
            buf_len += 1;
        }
        let mut pre_groups: [D3D12_BARRIER_GROUP; 2] =
            std::array::from_fn(|_| D3D12_BARRIER_GROUP::default());
        let mut pre_len = 0usize;
        if has_global {
            pre_groups[pre_len] = D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: global_barrier.as_ptr(),
                },
            };
            pre_len += 1;
        }
        if buf_len > 0 {
            pre_groups[pre_len] = D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_BUFFER,
                NumBarriers: buf_len as u32,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pBufferBarriers: buf_barriers.as_ptr(),
                },
            };
            pre_len += 1;
        }
        let post_buf = [enhanced_buffer_barrier(
            dst,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_SYNC_EXECUTE_INDIRECT,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT,
        )];
        let post_group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: post_buf.as_ptr(),
            },
        }];
        // SAFETY: an open list, live resources, matching root signature & pipeline.
        unsafe {
            if pre_len > 0 {
                list7.Barrier(&pre_groups[..pre_len]);
            }
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootShaderResourceView(3, ops_gpu);
            l.SetComputeRootShaderResourceView(4, rebase_gpu);
            l.SetComputeRootUnorderedAccessView(5, dst_gpu);
            if let Some((_, budget_gpu)) = budget {
                l.SetComputeRootUnorderedAccessView(6, budget_gpu);
            }
            l.Dispatch(records.div_ceil(64), 1, 1);
            list7.Barrier(&post_group);
        }
        return true;
    }

    let mut first: [D3D12_RESOURCE_BARRIER; 4] =
        std::array::from_fn(|_| D3D12_RESOURCE_BARRIER::default());
    let mut first_len = 0usize;
    match barrier_plan.pre_inputs {
        PreInputUav::None => {}
        PreInputUav::Global => {
            first[first_len] = uav_barrier();
            first_len += 1;
        }
        PreInputUav::Scoped => {
            first[first_len] = uav_barrier_for(src);
            first_len += 1;
            if let Some((count_res, _)) = count
                && windows::core::Interface::as_raw(count_res)
                    != windows::core::Interface::as_raw(src)
            {
                first[first_len] = uav_barrier_for(count_res);
                first_len += 1;
            }
        }
    }
    if barrier_plan.pre_budget
        && let Some((budget_res, _)) = budget
    {
        first[first_len] = match budget_res {
            Some(res) => uav_barrier_for(res),
            None => uav_barrier(),
        };
        first_len += 1;
    }
    first[first_len] = transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    first_len += 1;
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let second_slice = if barrier_plan.post_global_uav {
        &second[..]
    } else {
        &second[1..]
    };
    // SAFETY: an open list, live resources, matching root signature & pipeline.
    unsafe {
        l.ResourceBarrier(&first[..first_len]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootShaderResourceView(3, ops_gpu);
        l.SetComputeRootShaderResourceView(4, rebase_gpu);
        l.SetComputeRootUnorderedAccessView(5, dst_gpu);
        if let Some((_, budget_gpu)) = budget {
            l.SetComputeRootUnorderedAccessView(6, budget_gpu);
        }
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(second_slice);
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
        /// Last, so given up last.
        _turn: std::sync::MutexGuard<'static, ()>,
    }

    /// The software adapter is one test's at a time. Two tests that compile
    /// and run shaders on it at once (it is one device for the process) make
    /// `d3d10warp.dll` (10.0.26100) crash now and then: access violations in
    /// its own shader compiler, on its own worker threads, in 12% of runs of
    /// all tests and 3% of runs of these alone on two threads; none in 100
    /// runs one after the other.
    pub(crate) fn warp_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            Self::on(true)
        }

        /// On the software adapter, or the first one (for measurements).
        fn on(software: bool) -> Option<Self> {
            let turn = warp_turn();
            // SAFETY: plain DXGI query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = if software {
                    factory.EnumWarpAdapter().ok()?
                } else {
                    factory.EnumAdapters(0).ok()?
                };
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if software != (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0) {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                    _turn: turn,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
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
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
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
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.SetComputeRootShaderResourceView(4, differs_gpu);
                warp.list.Dispatch(PIECES, 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            let found = [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() });
            if mode == 0 {
                return found;
            }
            // Above the flag, the hash: as `host::torn` makes it of the bytes.
            for (piece, d) in found.iter().enumerate() {
                // SAFETY: inside the mapping, which the card has read.
                let bytes = unsafe { std::slice::from_raw_parts(now_ptr.add(piece * PIECE).cast_const(), PIECE) };
                assert_eq!(d >> 1, crate::host::piece_hash(bytes), "piece {piece}");
            }
            found.map(|d| d & 1)
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// What the passes over copies in the card's memory cost on the card
    /// (ignored: a measurement): rounds of a few passes of some pieces and a
    /// wait, as a frame has them. `VICTORY_BENCH_CS=<file>` times another
    /// source of the shader.
    #[test]
    #[ignore]
    fn card_compare_costs() {
        let Some(mut gpu) = WarpHarness::on(false) else {
            return;
        };
        let other = std::env::var("VICTORY_BENCH_CS").ok().map(|f| std::fs::read_to_string(f).unwrap());
        let source: &'static str = other.map_or(crate::host::CARD_COMPARE, |s| Box::leak(s.into_boxed_str()));
        let walked_alone = source.contains("SV_DispatchThreadID");
        let compare = compile_cs(&gpu.device, source).expect("the compare must compile");
        let now = gpu.default_uav(256 << 20);
        let hashes = gpu.default_uav(64 << 10);
        let differs = gpu.default_uav(64 << 10);
        for (rounds, passes, pieces) in [(80u32, 5u32, 1u32), (80, 5, 5), (80, 5, 64), (80, 5, 500), (1, 1, 4000)] {
            let mut best = f64::MAX;
            for _ in 0..6 {
                // SAFETY: our open list, live buffers, the compare's own
                // root signature; the pieces lie in the buffers.
                unsafe {
                    gpu.list.SetComputeRootSignature(&compare.root);
                    gpu.list.SetPipelineState(&compare.pso);
                    gpu.list.SetComputeRootShaderResourceView(1, now.GetGPUVirtualAddress());
                    gpu.list.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
                    gpu.list.SetComputeRootUnorderedAccessView(3, differs.GetGPUVirtualAddress());
                    gpu.list.SetComputeRootShaderResourceView(4, differs.GetGPUVirtualAddress());
                    for _ in 0..rounds {
                        for pass in 0..passes {
                            let first = if pieces * passes < 4096 { pass * pieces } else { 0 };
                            let constants = [first, 0u32, 256 << 20, pieces, 0, 1];
                            gpu.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            gpu.list.Dispatch(if walked_alone { pieces.div_ceil(64) } else { pieces }, 1, 1);
                        }
                        gpu.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                            ..Default::default()
                        }]);
                    }
                }
                let t = std::time::Instant::now();
                gpu.submit_and_wait();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let mb = f64::from(rounds * passes * pieces) / 16.0;
            println!("{rounds} rounds of {passes} passes of {pieces} pieces: {best:.2} ms ({mb:.0} MB, {:.0} GB/s)", mb / best / 1.024);
        }
    }

    /// What raw loads through views that are not raw give past the view's
    /// end (ignored: a measurement of the card). A buffer view of 4 KiB in a
    /// buffer of 1 MiB whose every dword says where it lies: zeros past the
    /// view, or the bytes there (whatever the card has in that place).
    #[test]
    #[ignore]
    fn raw_loads_past_views() {
        let Some(mut gpu) = WarpHarness::on(std::env::var_os("VICTORY_BENCH_WARP").is_some()) else {
            return;
        };
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4,b0), DescriptorTable(SRV(t0,numDescriptors=8)), UAV(u0), DescriptorTable(CBV(b1,numDescriptors=8))"
cbuffer C : register(b0) { uint which; uint first; uint step; uint at; }
ByteAddressBuffer v[8] : register(t0);
struct Block { uint4 d[4096]; };
ConstantBuffer<Block> c[8] : register(b1);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint g : SV_GroupIndex) {
    if (step == 0) { if (id.x == 0) o.Store(at, v[which].Load(first)); }
    else if (step == 1) { if (id.x == 0) o.Store(at, c[which].d[first / 16][(first / 4) & 3]); }
    else if (step == 16) { o.Store(at + id.x * 4, c[which].d[first / 16 + id.x][(first / 4) & 3]); }
    else { o.Store(at + id.x * 4, v[which].Load(first + id.x * step)); }
}
"#;
        let pass = compile_cs(&gpu.device, SOURCE).expect("the pass must compile");
        let (buffer, data, _) = gpu.shared_buffer(1 << 20);
        for i in 0..(1usize << 18) {
            // SAFETY: inside the mapping.
            unsafe { data.cast::<u32>().add(i).write(0xA000_0000 | i as u32) };
        }
        let (_out, out_ptr, out_gpu) = gpu.shared_buffer(1 << 16);
        // SAFETY: plain heap creation, views of a live buffer.
        let heap: ID3D12DescriptorHeap = unsafe {
            gpu.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 16,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
        }
        .unwrap();
        // SAFETY: as above.
        let (cpu, table, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                heap.GetGPUDescriptorHandleForHeapStart(),
                gpu.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize,
            )
        };
        let names = ["raw", "structured 24 at 24", "structured 16", "typed u32", "typed u16", "typed 4 x u32", "constant buffer", "raw, moved to 8"];
        let view = |i: usize, format: i32, first: u64, count: u32, stride: u32, raw: bool, null: bool| {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(format),
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first,
                        NumElements: count,
                        StructureByteStride: stride,
                        Flags: if raw { D3D12_BUFFER_SRV_FLAG_RAW } else { D3D12_BUFFER_SRV_FLAG_NONE },
                    },
                },
            };
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + i * step };
            // SAFETY: a slot of our heap, a live buffer.
            unsafe { gpu.device.CreateShaderResourceView(if null { None } else { Some(&buffer) }, Some(&desc), at) };
        };
        for base in [0, 8] {
            view(base, 39, 0, 1024, 0, true, false);
            view(base + 1, 0, 1, 170, 24, false, false);
            view(base + 2, 0, 0, 256, 16, false, false);
            view(base + 3, 42, 0, 1024, 0, false, false);
            view(base + 4, 57, 0, 2048, 0, false, false);
            view(base + 5, 3, 0, 256, 0, false, false);
            // 1,024 dwords from 0 made in a heap the card does not see, moved
            // on by 8 bytes (`descriptor::host::move_raw`), copied in.
            // SAFETY: plain heap creation, slots of live heaps.
            unsafe {
                let beside: ID3D12DescriptorHeap = gpu
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let at = beside.GetCPUDescriptorHandleForHeapStart();
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT(39),
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: 1024,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                gpu.device.CreateShaderResourceView(&buffer, Some(&desc), at);
                let moved = crate::descriptor::host::move_raw(at, buffer.GetGPUVirtualAddress(), 4096, 8);
                println!("the raw view was moved: {moved}");
                gpu.device.CopyDescriptorsSimple(
                    1,
                    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + (base + 7) * step },
                    at,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                );
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                // SAFETY: a live buffer.
                BufferLocation: unsafe { buffer.GetGPUVirtualAddress() },
                SizeInBytes: 4096,
            };
            // SAFETY: a slot of our heap.
            unsafe { gpu.device.CreateConstantBufferView(Some(&desc), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu + (base + 6) * step }) };
        }
        let table2 = D3D12_GPU_DESCRIPTOR_HANDLE { ptr: table.ptr + 8 * step as u64 };
        let offsets = [0u32, 4092, 4096, 4104, 8192, 65536, 524_288, (1 << 20) - 4];
        for mode in ["one load (uniform)", "a load a thread", "constant buffer load", "Constant buffer load a thread"] {
            println!("{mode}:");
            for (which, name) in names.iter().enumerate() {
                // SAFETY: our open list, live buffers.
                unsafe {
                    std::ptr::write_bytes(out_ptr, 0x77, 1 << 16);
                    gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                    gpu.list.SetComputeRootSignature(&pass.root);
                    gpu.list.SetPipelineState(&pass.pso);
                    gpu.list.SetComputeRootDescriptorTable(1, table);
                    gpu.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    gpu.list.SetComputeRootDescriptorTable(3, table2);
                    for (k, &offset) in offsets.iter().enumerate() {
                        let step = match mode.as_bytes()[0] {
                            b'o' => 0u32,
                            b'c' => 1,
                            b'C' => 16,
                            _ => 4,
                        };
                        // (A thread's loads end at the offset.)
                        let first = if step == 4 { offset.saturating_sub(63 * 4) } else { offset };
                        let constants = [which as u32, first, step, k as u32 * 256];
                        gpu.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                        gpu.list.Dispatch(1, 1, 1);
                    }
                }
                gpu.submit_and_wait();
                let found: Vec<String> = offsets
                    .iter()
                    .enumerate()
                    .map(|(k, &offset)| {
                        let thread = if mode.as_bytes()[0] == b'a' { (offset.min(63 * 4) / 4) as usize } else { 0 };
                        // SAFETY: inside the mapping.
                        let word = unsafe { out_ptr.cast::<u32>().add(k * 64 + thread).read_volatile() };
                        format!("{offset}: {word:#x}")
                    })
                    .collect();
                println!("  {name:20} {}", found.join("  "));
            }
        }
        // SAFETY: a plain query.
        println!("device: {:?}", unsafe { gpu.device.GetDeviceRemovedReason() });
    }

    /// Whether the card can read a texture's descriptor half written
    /// (ignored: a measurement of the card, `VICTORY_BENCH_S` seconds a way,
    /// 5). Two slots change between a texture of 64x64 and one of 2048x2048
    /// as fast as a thread can write them, while groups that each load the
    /// descriptor anew ask its size and read its last texel: size and texel
    /// of two textures are one descriptor half written. (Both lie in one heap
    /// that is there as a whole: a read past the small one finds memory.)
    #[test]
    #[ignore]
    fn descriptors_written_under_readers() {
        let Some(mut gpu) = WarpHarness::on(std::env::var_os("VICTORY_BENCH_WARP").is_some()) else {
            return;
        };
        let seconds = std::env::var("VICTORY_BENCH_S").ok().and_then(|v| v.parse().ok()).unwrap_or(5u64);
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1,b0), DescriptorTable(SRV(t0,numDescriptors=2,flags=DESCRIPTORS_VOLATILE)), UAV(u0)"
cbuffer C : register(b0) { uint odd; }
Texture2D<float4> t[2] : register(t0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint i : SV_GroupIndex) {
    uint w, h, n;
    uint which = (g.x + odd) & 1;
    t[which].GetDimensions(0, w, h, n);
    uint r = (uint)(t[which].Load(int3(w - 1, h - 1, 0)).x * 255.0 + 0.5);
    // 0: small, 1: large, 2: no size of the two, 3: small with another texel, 4: large with another
    uint kind = (w == 64 && h == 64) ? (r == 17 ? 0 : 3) : (w == 2048 && h == 2048) ? (r == 34 ? 1 : 4) : 2;
    if (i == 0) {
        uint was;
        o.InterlockedAdd(kind * 4, 1, was);
        if (kind > 1) { o.Store(32, w); o.Store(36, h); o.Store(40, r); }
    }
}
"#;
        let pass = compile_cs(&gpu.device, SOURCE).expect("the pass must compile");
        let texture = |size: u64| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: size,
            Height: size as u32,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT(28),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: plain creation; each texture in its own half of the heap.
        let (heap, textures) = unsafe {
            let mut heap: Option<ID3D12Heap> = None;
            gpu.device
                .CreateHeap(
                    &D3D12_HEAP_DESC {
                        SizeInBytes: 64 << 20,
                        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        Alignment: 0,
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                    },
                    &mut heap,
                )
                .unwrap();
            let heap = heap.unwrap();
            let textures = [(0u64, 64u64), (32 << 20, 2048)].map(|(at, size)| {
                let mut made: Option<ID3D12Resource> = None;
                gpu.device
                    .CreatePlacedResource(&heap, at, &texture(size), D3D12_RESOURCE_STATE_COPY_DEST, None, &mut made)
                    .unwrap();
                made.unwrap()
            });
            (heap, textures)
        };
        // Fill them: 17 and 34 in every byte.
        for (texture, (size, byte)) in textures.iter().zip([(64u32, 17u8), (2048, 34)]) {
            let pitch = size * 4;
            let (from, data, _) = gpu.shared_buffer(u64::from(pitch) * u64::from(size));
            // SAFETY: the mapping holds the bytes; an open list, live resources.
            unsafe {
                std::ptr::write_bytes(data, byte, (pitch * size) as usize);
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&from),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT(28),
                                Width: size,
                                Height: size,
                                Depth: 1,
                                RowPitch: pitch,
                            },
                        },
                    },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                gpu.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            gpu.submit_and_wait();
        }
        let view_heap = |visible: bool| -> ID3D12DescriptorHeap {
            // SAFETY: plain heap creation.
            unsafe {
                gpu.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4,
                    Flags: if visible { D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE } else { D3D12_DESCRIPTOR_HEAP_FLAG_NONE },
                    NodeMask: 0,
                })
            }
            .unwrap()
        };
        let (shown, kept) = (view_heap(true), view_heap(false));
        // SAFETY: live heaps.
        let (to, from, step) = unsafe {
            (
                shown.GetCPUDescriptorHandleForHeapStart().ptr,
                kept.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize,
            )
        };
        let at = |base: usize, i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base + i * step };
        for (i, texture) in textures.iter().enumerate() {
            // SAFETY: slots of our heaps, live textures.
            unsafe {
                gpu.device.CreateShaderResourceView(texture, None, at(from, i));
                gpu.device.CreateShaderResourceView(texture, None, at(to, i));
            }
        }
        let bytes = |base: usize, i: usize| {
            // SAFETY: a heap's handle; what it points at is read if it is memory.
            let mut out = vec![0u8; step];
            crate::host::read_into((base + i * step) as *const u8, &mut out).then_some(out)
        };
        println!("step {step}, the heaps' handles point at {:#x} and {:#x} (shown, kept)", to, from);
        for i in 0..2 {
            println!("  shown {i}: {:02x?}", bytes(to, i));
            println!("  kept  {i}: {:02x?}", bytes(from, i));
        }
        let same = bytes(to, 0).is_some() && bytes(to, 0) == bytes(from, 0) && bytes(to, 1) == bytes(from, 1);
        println!("the shown heap's memory is the descriptors' own bytes: {same}");
        let images = [bytes(from, 0), bytes(from, 1)];
        // SAFETY: a live heap.
        let table = unsafe { shown.GetGPUDescriptorHandleForHeapStart() };
        for way in ["view creation", "copies from a CPU heap", "one 32 byte store"] {
            if way.starts_with("one") && !(same && step == 32) {
                continue;
            }
            let (_out, out_ptr, out_gpu) = gpu.shared_buffer(1 << 12);
            // SAFETY: the mapping.
            unsafe { std::ptr::write_bytes(out_ptr, 0, 1 << 12) };
            let stop = std::sync::atomic::AtomicBool::new(false);
            let writes = std::sync::atomic::AtomicU64::new(0);
            std::thread::scope(|scope| {
                let device = gpu.device.clone();
                let (stop, writes, textures, images) = (&stop, &writes, &textures, &images);
                scope.spawn(move || {
                    let mut k = 0usize;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        for slot in 0..2 {
                            let which = (k + slot) & 1;
                            // SAFETY: slots of our heaps, live textures.
                            unsafe {
                                match way.as_bytes()[0] {
                                    b'v' => device.CreateShaderResourceView(&textures[which], None, at(to, slot)),
                                    b'c' => device.CopyDescriptorsSimple(
                                        1,
                                        at(to, slot),
                                        at(from, which),
                                        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                    ),
                                    _ => store32(to + slot * step, images[which].as_deref().unwrap()),
                                }
                            }
                        }
                        k += 1;
                        writes.fetch_add(2, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                let started = std::time::Instant::now();
                let mut odd = 0u32;
                while started.elapsed().as_secs() < seconds {
                    // SAFETY: our open list, live objects.
                    unsafe {
                        gpu.list.SetDescriptorHeaps(&[Some(shown.clone())]);
                        gpu.list.SetComputeRootSignature(&pass.root);
                        gpu.list.SetPipelineState(&pass.pso);
                        gpu.list.SetComputeRootDescriptorTable(1, table);
                        gpu.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                        for _ in 0..8 {
                            gpu.list.SetComputeRoot32BitConstants(0, 1, (&raw const odd).cast(), 0);
                            gpu.list.Dispatch(65_535, 1, 1);
                            odd += 1;
                        }
                    }
                    gpu.submit_and_wait();
                }
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            // SAFETY: the mapping, the card is done.
            let found: Vec<u32> = (0..11).map(|i| unsafe { out_ptr.cast::<u32>().add(i).read_volatile() }).collect();
            println!(
                "{way}: {} writes; groups that saw the small texture {}, the large {}; no size of the two {}, the small size with another texel {}, the large size with another texel {}; the last such: {}x{} texel {}",
                writes.load(std::sync::atomic::Ordering::Relaxed),
                found[0],
                found[1],
                found[2],
                found[3],
                found[4],
                found[8],
                found[9],
                found[10]
            );
            // SAFETY: a plain query.
            println!("  device: {:?}", unsafe { gpu.device.GetDeviceRemovedReason() });
        }
        drop(heap);
    }

    /// 32 bytes to `to` (32 byte aligned) in one store, then a fence.
    ///
    /// # Safety
    /// `to` is 32 bytes of memory that may be written.
    unsafe fn store32(to: usize, bytes: &[u8]) {
        #[target_feature(enable = "avx")]
        unsafe fn store(to: usize, bytes: *const u8) {
            use std::arch::x86_64::*;
            // SAFETY: the caller's.
            unsafe {
                _mm256_stream_si256(to as *mut __m256i, _mm256_loadu_si256(bytes.cast()));
                _mm_sfence();
            }
        }
        assert!(std::is_x86_feature_detected!("avx") && bytes.len() == 32 && to.is_multiple_of(32));
        // SAFETY: asked for above; the caller's memory.
        unsafe { store(to, bytes.as_ptr()) }
    }

    /// What a texture read in another dimension than its view's gives
    /// (ignored: a measurement of the card): a view of a small 2D texture
    /// read as an array, a volume and a cube, and a small volume read as 2D,
    /// far out. The texture lies at the start of a heap whose other megabytes
    /// hold other bytes: a texel of those is a read past the texture.
    #[test]
    #[ignore]
    fn textures_read_in_another_dimension() {
        let Some(mut gpu) = WarpHarness::on(std::env::var_os("VICTORY_BENCH_WARP").is_some()) else {
            return;
        };
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=2,b0), DescriptorTable(SRV(t0,numDescriptors=4,flags=DESCRIPTORS_VOLATILE)), UAV(u0), StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_POINT,addressU=TEXTURE_ADDRESS_WRAP,addressV=TEXTURE_ADDRESS_WRAP,addressW=TEXTURE_ADDRESS_WRAP)"
cbuffer C : register(b0) { uint how; uint far; }
Texture2D<float4> flat_ : register(t0);
Texture2DArray<float4> layers : register(t1);
Texture3D<float4> volume : register(t2);
TextureCubeArray<float4> cube : register(t3);
SamplerState s : register(s0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint2 at = uint2(t.x & 7, t.x >> 3);
    float4 c = 0;
    if (how == 0) c = flat_.Load(int3(at, 0));
    if (how == 1) c = layers.Load(int4(at, far, 0));
    if (how == 2) c = volume.Load(int4(at, far, 0));
    if (how == 3) c = layers.SampleLevel(s, float3((at + 0.5) / 64, far), 0);
    if (how == 4) c = volume.SampleLevel(s, float3((at + 0.5) / 64, (far + 0.5) / 4096.0), 0);
    if (how == 5) c = cube.SampleLevel(s, float4(1, (at + 0.5) / 64, far), 0);
    if (how == 6) c = flat_.Load(int3(at, far));
    uint r = (uint)(c.x * 255.0 + 0.5), was;
    // the texture's own bytes, zeros, anything else
    o.InterlockedAdd(r == 17 ? 0 : r == 0 ? 4 : 8, 1, was);
    if (r != 17 && r != 0) o.Store(12, r);
}
"#;
        let pass = compile_cs(&gpu.device, SOURCE).expect("the pass must compile");
        // SAFETY: plain creation, each thing in its own part of the heap.
        let (_heap, small) = unsafe {
            let mut heap: Option<ID3D12Heap> = None;
            gpu.device
                .CreateHeap(
                    &D3D12_HEAP_DESC {
                        SizeInBytes: 256 << 20,
                        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        Alignment: 0,
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                    },
                    &mut heap,
                )
                .unwrap();
            let heap = heap.unwrap();
            let place = |at: u64, size: u64| {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: size,
                    Height: size as u32,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT(28),
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let mut made: Option<ID3D12Resource> = None;
                gpu.device
                    .CreatePlacedResource(&heap, at, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut made)
                    .unwrap();
                made.unwrap()
            };
            let small = place(0, 64);
            // The rest: 34 in every byte.
            let mut fill: Vec<(ID3D12Resource, u32, u8)> = vec![(small.clone(), 64, 17)];
            for k in 0..15u64 {
                fill.push((place((1 + k) << 24, 2048), 2048, 34));
            }
            for k in 1..16u64 {
                fill.push((place(k << 20, 512), 512, 34));
            }
            for (texture, size, byte) in fill {
                let pitch = size * 4;
                let (from, data, _) = gpu.shared_buffer(u64::from(pitch) * u64::from(size));
                std::ptr::write_bytes(data, byte, (pitch * size) as usize);
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&from),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT(28),
                                Width: size,
                                Height: size,
                                Depth: 1,
                                RowPitch: pitch,
                            },
                        },
                    },
                };
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                gpu.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                gpu.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                gpu.submit_and_wait();
            }
            (heap, small)
        };
        // SAFETY: plain heap creation; slots of it, a live texture.
        let heap: ID3D12DescriptorHeap = unsafe {
            let heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = gpu.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            for i in 0..4 {
                let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap.GetCPUDescriptorHandleForHeapStart().ptr + i * step };
                // A 2D view of the small texture in every slot.
                gpu.device.CreateShaderResourceView(&small, None, at);
            }
            heap
        };
        for (how, name) in
            ["2D", "array, loaded", "volume, loaded", "array, sampled", "volume, sampled", "cube array", "2D, a level it has not"]
                .iter()
                .enumerate()
        {
            for far in [0u32, 1, 7, 63, 1000, 4000] {
                let (_out, out_ptr, out_gpu) = gpu.shared_buffer(1 << 12);
                // SAFETY: the mapping; our open list, live objects.
                unsafe {
                    std::ptr::write_bytes(out_ptr, 0, 1 << 12);
                    gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                    gpu.list.SetComputeRootSignature(&pass.root);
                    gpu.list.SetPipelineState(&pass.pso);
                    gpu.list.SetComputeRoot32BitConstants(0, 2, [how as u32, far].as_ptr().cast(), 0);
                    gpu.list.SetComputeRootDescriptorTable(1, heap.GetGPUDescriptorHandleForHeapStart());
                    gpu.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    gpu.list.Dispatch(1, 1, 1);
                }
                gpu.submit_and_wait();
                // SAFETY: the mapping, the card is done.
                let found: Vec<u32> = (0..4).map(|i| unsafe { out_ptr.cast::<u32>().add(i).read_volatile() }).collect();
                println!("{name} at {far}: its own texels {}, zeros {}, others {} (the last {})", found[0], found[1], found[2], found[3]);
            }
        }
        // SAFETY: a plain query.
        println!("device: {:?}", unsafe { gpu.device.GetDeviceRemovedReason() });
    }

    /// Loads far outside their views (ignored: it may lose the device, and
    /// says which way did): raw, typed, constant and texture loads at the
    /// ends of 32 bits, uniform and per lane, and samples at coordinates,
    /// levels and gradients that are no numbers. Each way is a submission of
    /// its own.
    #[test]
    #[ignore]
    fn loads_far_outside_views() {
        let Some(mut gpu) = WarpHarness::on(std::env::var_os("VICTORY_BENCH_WARP").is_some()) else {
            return;
        };
        const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4,b0), DescriptorTable(SRV(t0,numDescriptors=4),CBV(b1)), UAV(u0), StaticSampler(s0,filter=FILTER_MIN_MAG_MIP_LINEAR,addressU=TEXTURE_ADDRESS_WRAP,addressV=TEXTURE_ADDRESS_WRAP)"
cbuffer C : register(b0) { uint how; uint far; uint lane; uint out_at; }
ByteAddressBuffer raw : register(t0);
Buffer<float4> typed : register(t1);
Texture2D<float4> tex : register(t2);
struct Block { uint4 d[4096]; };
ConstantBuffer<Block> block : register(b1);
SamplerState s : register(s0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 t : SV_DispatchThreadID) {
    uint at = far + t.x * lane * 4;
    float4 c = 0;
    if (how == 0) c = asfloat(raw.Load4(at));
    if (how == 1) c = asfloat(raw.Load(at + 4092));
    if (how == 2) c = typed.Load(at);
    if (how == 3) c = asfloat(block.d[at]);
    if (how == 4) c = tex.Load(int3(at, 0, 0));
    if (how == 5) c = tex.Load(int3(0, 0, at));
    if (how == 6) c = tex.SampleLevel(s, asfloat(uint2(at, at)), 0);
    if (how == 7) c = tex.SampleLevel(s, float2(0.5, 0.5), asfloat(at));
    if (how == 8) c = tex.SampleGrad(s, float2(0.5, 0.5), asfloat(at).xx, asfloat(at).xx);
    uint was;
    o.InterlockedAdd(any(asuint(c) != 0) ? 4 : 0, 1, was);
}
"#;
        let pass = compile_cs(&gpu.device, SOURCE).expect("the pass must compile");
        let (buffer, data, gpu_at) = gpu.shared_buffer(1 << 16);
        // SAFETY: the mapping.
        unsafe { std::ptr::write_bytes(data, 0x3f, 1 << 16) };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 7,
            Format: DXGI_FORMAT(28),
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: plain creation; slots of our heap, live objects.
        let (_texture, heap) = unsafe {
            let mut made: Option<ID3D12Resource> = None;
            gpu.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    None,
                    &mut made,
                )
                .unwrap();
            let texture = made.unwrap();
            let heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 5,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = gpu.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let at = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: heap.GetCPUDescriptorHandleForHeapStart().ptr + i * step };
            for (i, format, count, flags) in
                [(0, 39, 1024, D3D12_BUFFER_SRV_FLAG_RAW), (1, 2, 256, D3D12_BUFFER_SRV_FLAG_NONE)]
            {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT(format),
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: count, StructureByteStride: 0, Flags: flags },
                    },
                };
                gpu.device.CreateShaderResourceView(&buffer, Some(&view), at(i));
            }
            gpu.device.CreateShaderResourceView(&texture, None, at(2));
            gpu.device.CreateShaderResourceView(&texture, None, at(3));
            gpu.device.CreateConstantBufferView(
                Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC { BufferLocation: gpu_at, SizeInBytes: 4096 }),
                at(4),
            );
            (texture, heap)
        };
        let names = [
            "raw, 4 dwords",
            "raw, offset added",
            "typed",
            "constant buffer",
            "texel",
            "level of a texel",
            "sample",
            "level of a sample",
            "gradients of a sample",
        ];
        for (how, name) in names.iter().enumerate() {
            for far in [
                0u32,
                0x1000,
                0x7fff_fff0,
                0x8000_0000,
                0xffff_f000,
                0xffff_fff0,
                0xffff_fffc,
                0x7fc0_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7f7f_ffff,
            ] {
                for lane in [0u32, 1] {
                    let (_out, out_ptr, out_gpu) = gpu.shared_buffer(1 << 12);
                    // SAFETY: the mapping; our open list, live objects.
                    unsafe {
                        std::ptr::write_bytes(out_ptr, 0, 1 << 12);
                        gpu.list.SetDescriptorHeaps(&[Some(heap.clone())]);
                        gpu.list.SetComputeRootSignature(&pass.root);
                        gpu.list.SetPipelineState(&pass.pso);
                        gpu.list.SetComputeRoot32BitConstants(0, 4, [how as u32, far, lane, 0].as_ptr().cast(), 0);
                        gpu.list.SetComputeRootDescriptorTable(1, heap.GetGPUDescriptorHandleForHeapStart());
                        gpu.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                        gpu.list.Dispatch(1, 1, 1);
                        // (Not `submit_and_wait`, which expects to go on.)
                        gpu.list.Close().unwrap();
                        gpu.submitted += 1;
                        let cmd: ID3D12CommandList = gpu.list.cast().unwrap();
                        gpu.queue.ExecuteCommandLists(&[Some(cmd)]);
                        let _ = gpu.queue.Signal(&gpu.fence, gpu.submitted);
                        let _ = gpu.fence.SetEventOnCompletion(gpu.submitted, gpu.event);
                        WaitForSingleObject(gpu.event, 8_000);
                    }
                    // SAFETY: the mapping; plain queries.
                    let (zeros, others, device) = unsafe {
                        (
                            out_ptr.cast::<u32>().read_volatile(),
                            out_ptr.cast::<u32>().add(1).read_volatile(),
                            gpu.device.GetDeviceRemovedReason(),
                        )
                    };
                    println!(
                        "{name} at {far:#x}, lanes {lane}: zeros {zeros}, others {others}{}",
                        if device.is_ok() { "" } else { " DEVICE LOST" }
                    );
                    if device.is_err() {
                        return;
                    }
                    // SAFETY: the card is done with them.
                    unsafe {
                        gpu.allocator.Reset().unwrap();
                        gpu.list.Reset(&gpu.allocator, None).unwrap();
                    }
                }
            }
        }
    }

    /// Whether lists on the card may read a texture of a block made after
    /// they were sent, through a slot they read by index (ignored: a
    /// measurement that loses the device, so each way runs in a process of
    /// its own; `VICTORY_BENCH_S` seconds a way, 30; `VICTORY_BENCH_N` runs of
    /// each, 3). Threads (`VICTORY_BENCH_THREADS`, 4) make blocks of 64 MiB, a texture in each and its
    /// view, and copies the view into a slot; long dispatches that read all
    /// the slots are always on the card.
    #[test]
    #[ignore]
    fn new_blocks_under_readers() {
        if let Ok(way) = std::env::var("VICTORY_BLOCK_WAY") {
            return new_blocks(&way);
        }
        let times: u32 = std::env::var("VICTORY_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
        let seconds = std::env::var("VICTORY_BENCH_S").unwrap_or_else(|_| "30".into());
        // "never": never resident, what must lose the device. "fresh-": no
        // block goes (280 of them, 18 GB), so each has addresses no block had.
        for way in ["fresh-made", "fresh-waited", "fresh-enqueued", "made", "waited", "enqueued", "never"] {
            let fresh = way.starts_with("fresh-");
            for n in 0..if way == "never" { 1 } else if fresh { times * 5 } else { times } {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["indirect::tests::new_blocks_under_readers", "--exact", "--ignored", "--nocapture"])
                    .env("VICTORY_BLOCK_WAY", way)
                    .env("VICTORY_BENCH_S", if fresh { "3" } else { &seconds })
                    .output()
                    .unwrap();
                let text = String::from_utf8_lossy(&out.stdout).into_owned();
                let said = |what: &str| text.lines().find(|l| l.contains(what)).unwrap_or("(nothing said)").trim().to_owned();
                println!("{way}, run {n}: {}; {} (exit {:?})", said("dispatches a"), said("blocks,"), out.status.code());
            }
        }
    }

    fn new_blocks(way: &str) {
        let Some(gpu) = WarpHarness::on(false) else {
            println!("no card: 0 blocks,");
            return;
        };
        let (way, fresh) = match way.strip_prefix("fresh-") {
            Some(way) => (way, true),
            None => (way, false),
        };
        const SLOTS: usize = 64;
        const BLOCK: u64 = 64 << 20;
        let seconds = std::env::var("VICTORY_BENCH_S").ok().and_then(|v| v.parse().ok()).unwrap_or(30u64);
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0,numDescriptors=64,flags=DESCRIPTORS_VOLATILE|DATA_VOLATILE)), UAV(u0)"
Texture2D<float4> t[64] : register(t0);
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint i : SV_GroupIndex) {
    uint which = g.x & 63;
    float4 c = 0;
    for (uint k = 0; k < 8; k++)
        c += t[(which + k * 8) & 63].Load(int3((i + k * 13) & 127, (g.y + k * 29) & 255, 0));
    if (c.x < -1) o.Store(0, 1);
}
"#;
        let pass = compile_cs(&gpu.device, SOURCE).expect("the pass must compile");
        // SAFETY: plain heap creation.
        let heap = |visible: bool, n: u32| -> ID3D12DescriptorHeap {
            unsafe {
                gpu.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: n,
                    Flags: if visible { D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE } else { D3D12_DESCRIPTOR_HEAP_FLAG_NONE },
                    NodeMask: 0,
                })
            }
            .unwrap()
        };
        let (shown, beside) = (heap(true, SLOTS as u32), heap(false, 65));
        // SAFETY: live heaps.
        let (to, from, step) = unsafe {
            (
                shown.GetCPUDescriptorHandleForHeapStart().ptr,
                beside.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize,
            )
        };
        let null = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT(28),
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        };
        let slot = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: to + i * step };
        let own = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: from + i * step };
        // SAFETY: slots of our heaps.
        unsafe {
            gpu.device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&null), own(0));
            for i in 0..SLOTS {
                gpu.device.CopyDescriptorsSimple(1, slot(i), own(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
        }
        let (_out, _, out_gpu) = gpu.shared_buffer(1 << 12);
        // SAFETY: a live heap.
        let table = unsafe { shown.GetGPUDescriptorHandleForHeapStart() };
        let list = |dispatches: u32| -> (ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
            // SAFETY: plain creation and recording.
            unsafe {
                let allocator: ID3D12CommandAllocator =
                    gpu.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList =
                    gpu.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                list.SetDescriptorHeaps(&[Some(shown.clone())]);
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                list.SetComputeRootDescriptorTable(0, table);
                list.SetComputeRootUnorderedAccessView(1, out_gpu);
                for _ in 0..dispatches {
                    list.Dispatch(1024, 64, 1);
                }
                list.Close().unwrap();
                (allocator, list)
            }
        };
        // Lists of some 5 ms, always five sent: some run while a slot changes.
        let mut sent = 0u64;
        let mut dispatches = 8;
        let ms = loop {
            let one = list(dispatches);
            let began = std::time::Instant::now();
            // SAFETY: a closed list of ours; waited for.
            unsafe {
                gpu.queue.ExecuteCommandLists(&[Some(one.1.cast().unwrap())]);
                sent += 1;
                gpu.queue.Signal(&gpu.fence, sent).unwrap();
                let _ = gpu.fence.SetEventOnCompletion(sent, gpu.event);
                WaitForSingleObject(gpu.event, 10_000);
            }
            let ms = began.elapsed().as_secs_f64() * 1e3;
            if ms >= 4.0 || dispatches >= 1 << 14 {
                break ms;
            }
            dispatches *= 2;
        };
        println!("{dispatches} dispatches a list, {ms:.1} ms each");
        let lists: Vec<_> = (0..6).map(|_| list(dispatches)).collect();
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
        let sent = AtomicU64::new(sent);
        let stop = AtomicBool::new(false);
        // Blocks; microseconds making a block, the longest; waiting, the longest.
        let counts: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
        let threads: usize = std::env::var("VICTORY_BENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
        let started = std::time::Instant::now();
        let lost = std::thread::scope(|scope| {
            for thread in 0..threads {
                let (device, fence) = (gpu.device.clone(), gpu.fence.clone());
                let (sent, stop, counts) = (&sent, &stop, &counts);
                scope.spawn(move || {
                    // Slots of its own, and a staged view of its own.
                    let mine = SLOTS / threads;
                    let slot = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: to + (thread * mine + i) * step };
                    let own = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: from + i * step };
                    let wait = way != "made";
                    let mut alive: Vec<(ID3D12Heap, ID3D12Resource)> = Vec::new();
                    let mut n = 0u64;
                    while !stop.load(SeqCst) {
                        // SAFETY: plain creation; slots of our heaps, this thread's.
                        unsafe {
                            let mut block: Option<ID3D12Heap> = None;
                            let flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES
                                | if wait { D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT } else { D3D12_HEAP_FLAG_NONE };
                            let heap = D3D12_HEAP_DESC {
                                SizeInBytes: BLOCK,
                                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                                Alignment: 0,
                                Flags: flags,
                            };
                            let began = std::time::Instant::now();
                            if device.CreateHeap(&heap, &mut block).is_err() {
                                break;
                            }
                            let us = began.elapsed().as_micros() as u64;
                            counts[1].fetch_add(us, SeqCst);
                            counts[2].fetch_max(us, SeqCst);
                            let block = block.unwrap();
                            let pageable: ID3D12Pageable = block.cast().unwrap();
                            let began = std::time::Instant::now();
                            if way == "waited" && device.MakeResident(&[Some(pageable.clone())]).is_err() {
                                break;
                            }
                            if way == "enqueued" {
                                let device: ID3D12Device3 = device.cast().unwrap();
                                let paged: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                                let asked =
                                    device.EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, &[Some(pageable)], &paged, 1);
                                if asked.is_err() {
                                    break;
                                }
                                while paged.GetCompletedValue() < 1 {
                                    std::hint::spin_loop();
                                }
                            }
                            let us = began.elapsed().as_micros() as u64;
                            counts[3].fetch_add(us, SeqCst);
                            counts[4].fetch_max(us, SeqCst);
                            let desc = D3D12_RESOURCE_DESC {
                                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                                Width: 128,
                                Height: 256,
                                DepthOrArraySize: 1,
                                MipLevels: 1,
                                Format: DXGI_FORMAT(28),
                                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                                ..Default::default()
                            };
                            let mut texture: Option<ID3D12Resource> = None;
                            // Somewhere in the block, as the pool's pieces are.
                            let at = (n * 7 % 900) << 16;
                            if device
                                .CreatePlacedResource(&block, at, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
                                .is_err()
                            {
                                break;
                            }
                            let texture = texture.unwrap();
                            let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
                            device.CreateShaderResourceView(&texture, None, own(1 + thread));
                            device.CopyDescriptorsSimple(1, slot(n as usize % mine), own(1 + thread), kind);
                            alive.push((block, texture));
                            n += 1;
                            counts[0].fetch_add(1, SeqCst);
                            if fresh {
                                if alive.len() >= 280 / threads {
                                    break;
                                }
                            } else if alive.len() >= 24 {
                                // Nothing shows or reads them any more when they go.
                                for i in 0..mine {
                                    device.CopyDescriptorsSimple(1, slot(i), own(0), kind);
                                }
                                let until = sent.load(SeqCst) + 1;
                                while fence.GetCompletedValue() < until && !stop.load(SeqCst) {
                                    std::thread::sleep(std::time::Duration::from_millis(1));
                                }
                                if fence.GetCompletedValue() < until {
                                    break;
                                }
                                alive.clear();
                            }
                        }
                    }
                    // The lists sent last may read them: the process goes next.
                    std::mem::forget(alive);
                });
            }
            let lost = loop {
                // SAFETY: closed lists of ours, each run again only when it has run.
                unsafe {
                    let done = gpu.fence.GetCompletedValue();
                    if done == u64::MAX || gpu.device.GetDeviceRemovedReason().is_err() {
                        break true;
                    }
                    while sent.load(SeqCst) < done + 5 {
                        let n = sent.load(SeqCst);
                        let list: ID3D12CommandList = lists[(n % 6) as usize].1.cast().unwrap();
                        gpu.queue.ExecuteCommandLists(&[Some(list)]);
                        let _ = gpu.queue.Signal(&gpu.fence, n + 1);
                        sent.store(n + 1, SeqCst);
                    }
                }
                if started.elapsed().as_secs() >= seconds {
                    break false;
                }
                std::thread::sleep(std::time::Duration::from_micros(300));
            };
            stop.store(true, SeqCst);
            lost
        });
        let sent = sent.into_inner();
        let counts = counts.map(|c| c.into_inner());
        let blocks = counts[0];
        // SAFETY: a plain query.
        let reason = unsafe { gpu.device.GetDeviceRemovedReason() };
        println!(
            "{blocks} blocks, {sent} lists in {:.1} s; making {:.0} us a block, the longest {}; waiting {:.0} us, the longest {}; {}",
            started.elapsed().as_secs_f64(),
            counts[1] as f64 / blocks.max(1) as f64,
            counts[2],
            counts[3] as f64 / blocks.max(1) as f64,
            counts[4],
            if lost { format!("LOST: {reason:?}") } else { "the device lives".to_owned() }
        );
        if !lost {
            // SAFETY: waits for what was sent.
            unsafe {
                let _ = gpu.fence.SetEventOnCompletion(sent, gpu.event);
                WaitForSingleObject(gpu.event, 10_000);
            }
        }
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// Dispatch records spread across z (bit 3, `crate::draw`'s spread
    /// signature): every record is written as [row, asked x, x, y, z], 20
    /// bytes apart; one over 65,535 groups in x with z 1 is spread (115,200:
    /// rows of 57,600, two of them), one within the limit is not (row 0).
    /// (After upstream VIctory pull request #2312's test.)
    #[test]
    fn warp_spread_records_split_large_x_and_keep_small_unspread() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let records: [[u32; 3]; 3] = [[115_200, 1, 1], [100, 2, 1], [65_537, 1, 1]];
        let (src_buf, src_ptr, _) = warp.shared_buffer(256);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(256);
        // SAFETY: buffers mapped with the capacities above.
        unsafe {
            for (i, r) in records.iter().enumerate() {
                for (w, v) in r.iter().enumerate() {
                    (src_ptr as *mut u32).add(i * 3 + w).write_volatile(*v);
                }
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(readback_ptr, 0, 256);
        }
        assert!(record_clamp_with(
            &warp.clamp,
            &warp.list,
            None,
            false,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            Some((None, budget_gpu)),
            (&src_buf, 0),
            None,
            (KIND_DISPATCH, STRIDE, records.len() as u32),
            false,
            true,
            plan_clamp_barriers(0, true, false, false, false, true),
        ));
        // SAFETY: open command list; the clamp left `dst_buf` in `INDIRECT_ARGUMENT`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 60);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; a mapped buffer.
        let out: Vec<u32> = unsafe {
            std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), 15).to_vec()
        };
        assert_eq!(&out[0..5], &[57_600, 115_200, 57_600, 1, 2], "115,200 x 1 x 1: spread");
        assert_eq!(&out[5..10], &[0, 100, 100, 2, 1], "100 x 2 x 1: not spread");
        assert_eq!(&out[10..15], &[32_769, 65_537, 32_769, 1, 2], "65,537 x 1 x 1: spread");
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds what any card runs in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [(1 << 20) + 1, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }

        // Snapshot `dst_buf` after Call #1 (copy 5 * STRIDE bytes, then leave
        // `dst_buf` readable as arguments again, as the clamp left it).
        // SAFETY: open command list; the clamp left `dst_buf` in `INDIRECT_ARGUMENT`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        // The scratch buffer goes straight from arguments back to unordered
        // access.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            // SAFETY: an open list; the scratch buffer, which the clamp left
            // readable as arguments, is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            // As one window ahead of its draws and one by one alike.
            for i in filled.len()..RECORDS as usize {
                assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
            }
            assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
            assert_eq!(largest, 519, "and are not counted as asked for");
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                // The last is no draw: empty, and free.
                if i + 1 < RECORDS as usize { 1 << 22 } else { 1 << 23 }, // IndexCount
                u32::MAX,                    // InstanceCount -> clamps to 16 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            false,
        ));

        // SAFETY: open command list; copy clamped scratch buffer (left
        // readable as arguments) for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 22) && rec[1] == 16 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, the scratch buffer's
    /// tracked state, `Host::execute`, and `HostList::reset` end-to-end on
    /// the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`, which stays readable
        // as arguments between them.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64), false, false).unwrap();
        assert_eq!(
            list.indirect_state(&b1),
            args,
            "a clamp leaves the scratch readable as arguments"
        );
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128), false, false).unwrap();
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2), args);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let grown = list.indirect_scratch(5000 * 20).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            D3D12_RESOURCE_STATE_COMMON,
            "a new scratch buffer is common"
        );
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000), false, false).unwrap();
        assert_eq!(b3.as_raw(), grown.as_raw());
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3), args);
        assert_eq!(
            list.indirect_state(&b1),
            D3D12_RESOURCE_STATE_COMMON,
            "the replaced buffer went back to common"
        );

        list.close();
        // With the debug layer on (`VICTORY_D3D12_DEBUG=1`, diagnostics):
        // every barrier recorded named the state the buffer was in. Read
        // before the submission, which moves the messages to the log.
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations; the buffer decayed to common when the
        // submission ran.
        assert_eq!(
            list.indirect_state(&b3),
            D3D12_RESOURCE_STATE_COMMON,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096), false, false).unwrap();
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4), args);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty. (Racing for the
    /// word record by record, which of them fitted was up to the GPU.)
    #[test]
    fn warp_a_group_charges_its_records_in_lane_order() {
        let Some((left, out)) = four_records_of_a_group(true) else {
            return;
        };
        assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits");
        for i in 1..4 {
            assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
        }
        assert_eq!(left, 0, "the record that did not fit took the rest");
    }

    /// As above where a record that does not fit takes nothing
    /// (`VICTORY_INDIRECT_REST=0`): those after it that fit run.
    #[test]
    fn warp_a_record_that_does_not_fit_may_take_nothing() {
        let Some((left, out)) = four_records_of_a_group(false) else {
            return;
        };
        for (i, count) in [100, 0, 50, 100].into_iter().enumerate() {
            assert_eq!(out[i * 4..i * 4 + 4], [count, count.min(1), 7, 9], "record {i}");
        }
        assert_eq!(left, 50);
    }

    /// Draws of 100, 250, 50 and 100 with 300 left of the budget: what is
    /// left then, and the records as clamped.
    fn four_records_of_a_group(rest: bool) -> Option<(u32, Vec<u32>)> {
        let mut warp = WarpHarness::new()?;
        let source = SOURCE.replace("REST ?", if rest { "true ?" } else { "false ?" });
        let clamp = make_from(&warp.device, &source).expect("the clamp must compile on WARP");
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;
        const BUDGET_CAP: u32 = 1 << 27;
        let counts = [100u32, 250, 50, 100];
        let total = counts.len() * STRIDE as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            for (i, &count) in counts.iter().enumerate() {
                let words = [count, 1, 7, 9];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STRIDE as usize),
                    16,
                );
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (budget_ptr as *mut u32).write_volatile(BUDGET_CAP - 300);
        }
        assert!(record_clamp_as(
            &clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DRAW, STRIDE, counts.len() as u32),
            false,
        ));
        // SAFETY: an open list; the clamped records are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                    .to_vec(),
            )
        };
        Some((BUDGET_CAP - spent, out))
    }

    /// The list's budget starts empty at each submission, wherever it
    /// lives (the card's copy is zeroed in the list): a recording whose
    /// clamps spend all of it is followed, after the reset, by one whose
    /// small record still runs.
    #[test]
    fn host_list_budget_starts_empty_at_each_submission() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        // Three records of 65,535 x 16 groups: about half the budget each.
        let Some((big, big_ptr)) = host.shared_buffer(3 * u64::from(STRIDE)) else {
            return;
        };
        let Some((small, small_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            for i in 0..3 {
                let words = [65535u32, 16, 1];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    big_ptr.add(i * STRIDE as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), small_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for round in 0..2 {
            // Spends the budget (clamped, not executed), then one group.
            bound(&list, (&big, 0), None, (KIND_DISPATCH, STRIDE, 3), false, false).unwrap();
            let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1), false, false).unwrap();
            // SAFETY: an open list; the worker's root signature and
            // pipeline; buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            // The small record comes after the big ones spent the budget:
            // it runs empty, in every round alike (none carries over).
            assert_eq!(hits, 0, "round {round}: the spent budget empties the record after it");
        }
        // A recording with only the small record: the budget is empty
        // again, so it runs.
        let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1), false, false).unwrap();
        // SAFETY: as above.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, 1, "the budget starts empty at the next submission");
    }

    /// Records the dispatch records `groups` (three words each, `step`
    /// bytes apart, the rest of each record ones) into a fresh mapped
    /// buffer of `warp`'s device.
    fn dispatch_records(warp: &WarpHarness, groups: &[u32], step: u32) -> ID3D12Resource {
        let total = groups.len() * step as usize;
        let (buffer, ptr, _) = warp.shared_buffer(total as u64);
        let mut bytes = vec![0xffu8; total];
        for (i, &g) in groups.iter().enumerate() {
            for (w, word) in [g, 1, 1].into_iter().enumerate() {
                let at = i * step as usize + w * 4;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        // SAFETY: the mapping holds `total` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
        buffer
    }

    /// Two windows in one list, as `crate::draw` records them: each clamps
    /// its records ahead, then one execution per record reads the scratch
    /// buffer with no barrier between them; the second clamp moves the
    /// scratch buffer straight from arguments back to unordered access.
    /// Every record runs once, with the group count it asked for.
    #[test]
    fn warp_window_executions_read_their_records_with_no_barrier_between() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        // The title names 12 bytes and keeps its records 32 apart.
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let sources: Vec<ID3D12Resource> =
            windows.iter().map(|g| dispatch_records(&warp, g, STEP)).collect();
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(u64::from(STEP) * 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let mut state = D3D12_RESOURCE_STATE_COMMON;
        for (groups, src) in windows.iter().zip(&sources) {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, state),
                budget_gpu,
                (src, 0),
                None,
                (KIND_DISPATCH, STEP, groups.len() as u32),
                true,
            ));
            state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: an open list; the worker's root signature and
            // pipeline; the scratch buffer is readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                for i in 0..groups.len() as u64 {
                    warp.list.ExecuteIndirect(&sig, 1, &dst_buf, i * u64::from(STEP), None, 0);
                }
            }
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (hits, spent) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
            )
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        assert_eq!(hits, expected, "every record of both windows runs once");
        assert_eq!(spent, expected * 64, "and is charged what it asked for");
    }

    /// The windows of `warp_window_executions_read_their_records_with_no_barrier_between`
    /// through a host list (`bound_ahead`, the list's tracked scratch state,
    /// `Host::execute`), twice: on the first recording the scratch buffer
    /// starts common, on the second (after the reset) it starts common
    /// again. With the debug layer on (`VICTORY_D3D12_DEBUG=1` in a
    /// diagnostics build) no barrier names a state the buffer is not in.
    #[test]
    fn host_list_window_executions_need_no_barrier_of_their_own() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let mut sources = Vec::new();
        for groups in windows {
            let total = groups.len() * STEP as usize;
            let Some((buffer, ptr)) = host.shared_buffer(total as u64) else {
                return;
            };
            let mut bytes = vec![0xffu8; total];
            for (i, &g) in groups.iter().enumerate() {
                for (w, word) in [g, 1, 1].into_iter().enumerate() {
                    let at = i * STEP as usize + w * 4;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            // SAFETY: the mapping holds `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            for (groups, src) in windows.iter().zip(&sources) {
                let spec = (KIND_DISPATCH, STEP, groups.len() as u32);
                let scratch = bound_ahead(&list, (src, 0), spec, false).unwrap();
                assert_eq!(list.indirect_state(&scratch), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    for i in 0..groups.len() as u64 {
                        let at = i * u64::from(STEP);
                        list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0);
                    }
                }
            }
            list.close();
            // With the debug layer on: nothing it objects to was recorded.
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every record of both windows runs once");
        }
    }

    /// A command signature of one argument of `kind` on `device`.
    fn one_argument(device: &ID3D12Device, kind: u32, stride: u32) -> ID3D12CommandSignature {
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: a valid single-argument command signature description.
        unsafe {
            device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig).unwrap();
        }
        sig.unwrap()
    }

    /// An AMD card runs every group of an indirect dispatch of more than
    /// 65,535, each under its own number: the title's tile lists
    /// ([`widest`]), clamped one by one and in a batch. Others are not asked.
    #[test]
    fn card_runs_wide_indirect_dispatches() {
        const CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer seen : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_DispatchThreadID) {
    uint was;
    seen.InterlockedAdd(0, 1, was);
    seen.InterlockedMax(4, group.x, was);
    seen.InterlockedMax(8, thread.x, was);
    seen.InterlockedAdd(16 + group.x * 4, 1, was);
}
"#;
        let Some(mut gpu) = WarpHarness::on(false) else {
            return;
        };
        if !amd(&gpu.device) || widest(true) == 65_535 {
            return;
        }
        let worker = compile_cs(&gpu.device, CS).expect("the worker must compile");
        let pass = make_from(&gpu.device, batch::SOURCE).expect("the batched clamp must compile");
        let sig = gpu.signature(2, 12);
        for asked in [67_440u32, 115_200, 1 << 20, (1 << 20) + 1] {
            let runs = if asked > 1 << 20 { 65_535 } else { asked };
            for batched in [false, true] {
                let (src, src_ptr, src_gpu) = gpu.shared_buffer(1024);
                let (_budget, budget_ptr, budget_gpu) = gpu.shared_buffer(256);
                let (_table, table_ptr, table_gpu) = gpu.shared_buffer(batch::TABLE);
                let size = 16 + (4 << 20) + 16;
                let seen = gpu.default_uav(size);
                let (back, back_ptr, _) = gpu.shared_buffer(size);
                let dst = gpu.default_uav(64 << 10);
                // SAFETY: mapped buffers, which nothing reads yet; an open
                // list and live buffers.
                unsafe {
                    std::ptr::write_bytes(budget_ptr, 0, 256);
                    for (w, word) in [asked, 1, 1].into_iter().enumerate() {
                        src_ptr.cast::<u32>().add(w).write_volatile(word);
                    }
                    if batched {
                        batch::record(
                            &pass,
                            &gpu.list,
                            None,
                            false,
                            (&dst, D3D12_RESOURCE_STATE_COMMON),
                            Some((None, budget_gpu)),
                            [src_gpu; batch::SOURCES],
                            &[],
                            table_gpu,
                            plan_clamp_barriers(0, true, false, false, false, false),
                        );
                        let mut ids = [0u64; batch::SOURCES];
                        ids[0] = 1;
                        assert_eq!(batch::Table::new(table_ptr, 0, ids).place(1, 0, 2), Some(0));
                    } else {
                        let to = (&dst, D3D12_RESOURCE_STATE_COMMON);
                        assert!(record_clamp_as(&gpu.clamp, &gpu.list, to, budget_gpu, (&src, 0), None, (2, 12, 1), false));
                    }
                    gpu.list.SetComputeRootSignature(&worker.root);
                    gpu.list.SetPipelineState(&worker.pso);
                    gpu.list.SetComputeRootUnorderedAccessView(0, seen.GetGPUVirtualAddress());
                    gpu.list.ExecuteIndirect(&sig, 1, &dst, 0, None, 0);
                    gpu.list.ResourceBarrier(&[uav_barrier()]);
                    gpu.list.ResourceBarrier(&[transition(&seen, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    gpu.list.CopyBufferRegion(&back, 0, &seen, 0, size);
                }
                gpu.submit_and_wait();
                // SAFETY: the list has run; mapped words.
                let word = |i: usize| unsafe { back_ptr.cast::<u32>().add(i).read_volatile() };
                assert_eq!([word(0), word(1), word(2)], [runs, runs - 1, runs - 1], "asked for {asked}, batched {batched}");
                let wrong = (0..runs as usize).find(|&g| word(4 + g) != 1);
                assert_eq!((wrong, word(4 + runs as usize)), (None, 0), "every group once, asked for {asked}");
            }
        }
    }

    /// What a batch's table says of records added to it: one job per
    /// record, charged for every execution that reads it, and none for what
    /// the pass cannot read.
    #[test]
    fn batch_tables() {
        use batch::{JOBS, SLOT, TABLE, TIMES, Table, job};
        let mut memory = vec![0xffff_ffffu32; TABLE as usize / 4 + 1];
        // SAFETY: the vector holds the table and outlives its use.
        let mut ids = [0u64; batch::SOURCES];
        ids[0] = 7;
        ids[1] = 9;
        let mut t = unsafe { Table::new(memory.as_mut_ptr().cast(), 4096, ids) };
        assert_eq!(t.place(7, 64, 10), Some(4096));
        assert_eq!(t.place(9, 64, 10), Some(4096 + u64::from(SLOT)));
        assert_eq!(t.place(7, 64, 2), Some(4096 + 2 * u64::from(SLOT)), "another kind is another record");
        // The same record again: the same place, charged twice.
        assert_eq!(t.place(9, 64, 10), Some(4096 + u64::from(SLOT)));
        assert_eq!(&memory[..4], [3, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]);
        assert_eq!(memory[4..8], [job(0, 10, 1), 64, 4096, 0]);
        assert_eq!(memory[8..12], [job(1, 10, 2), 64, 4096 + SLOT, 0]);
        assert_eq!(job(1, 10, 2), 1 | 10 << 4 | 3 << 8 | 2 << 16);
        // Not a copy of the pass, not on 4 bytes, past 32 bits, not a kind.
        for (id, offset, kind) in [(8, 64, 10), (7, 66, 10), (7, 1 << 32, 10), (7, u64::from(u32::MAX) & !3, 10), (7, 64, 3)] {
            assert_eq!(t.place(id, offset, kind), None);
        }
        // Read more often than one job may be charged for: another job.
        for _ in 1..TIMES {
            assert_eq!(t.place(7, 64, 10), Some(4096));
        }
        assert_eq!(t.place(7, 64, 10), Some(4096 + 3 * u64::from(SLOT)));
        assert_eq!(t.place(7, 64, 10), Some(4096 + 3 * u64::from(SLOT)));
        for i in 4..u64::from(JOBS) {
            assert_eq!(t.place(7, 1024 + 16 * i, 0), Some(4096 + i * u64::from(SLOT)));
        }
        assert_eq!(t.place(7, 1024, 1), None, "full");
        assert_eq!(t.place(7, 1024 + 16 * 5, 0), Some(4096 + 5 * u64::from(SLOT)));
        assert_eq!((memory[0], memory[TABLE as usize / 4]), (JOBS, 0xffff_ffff));
    }

    /// A batch clamps records exactly as the clamp of one execution does, of
    /// any kind, sensible or not; they are added to its table after the pass
    /// was recorded; it charges each as often as it is read.
    #[test]
    fn warp_batch_clamps_as_the_clamp_of_one_does() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let pass = make_from(&warp.device, batch::SOURCE).expect("the batched clamp must compile on WARP");
        let kinds = [0u32, 1, 2, 10];
        let values = [0u32, 1, 3, 100, 65_535, 65_536, 1 << 20, (1 << 23) - 1, 1 << 23, 1 << 24, (1 << 26) + 1, u32::MAX];
        // Every kind with many records, in two sources.
        let mut records: Vec<(u32, [u32; 5])> = Vec::new();
        for kind in kinds {
            for (i, a) in values.into_iter().enumerate() {
                for (k, b) in values.into_iter().enumerate() {
                    records.push((kind, [a, b, values[(i * 3 + k) % 4], 77, 0xABCD]));
                }
            }
        }
        // In passes of so few that the budget holds whatever one asks for,
        // and one pass of them all, which it does not.
        let (src, src_ptr, src_gpu) = warp.shared_buffer(64 << 10);
        let (_other, other_ptr, other_gpu) = warp.shared_buffer(64 << 10);
        for (i, (_, words)) in records.iter().enumerate() {
            for (w, word) in words.iter().enumerate() {
                // SAFETY: 400 records of 32 bytes in 64 KiB.
                unsafe {
                    src_ptr.cast::<u32>().add(i * 8 + w).write_volatile(*word);
                    other_ptr.cast::<u32>().add(i * 8 + w).write_volatile(*word);
                }
            }
        }
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(64 << 10);
        let dst = warp.default_uav(64 << 10);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        for (i, (kind, _)) in records.iter().enumerate().step_by(3) {
            let offset = i as u64 * 32;
            let words = batch::words(*kind).unwrap();
            let stride = words * 4;
            let read = |p: *mut u8| -> Vec<u32> {
                // SAFETY: the mapping holds more.
                (0..words as usize).map(|w| unsafe { p.cast::<u32>().add(w).read_volatile() }).collect()
            };
            let mut seen = Vec::new();
            for batched in [false, true] {
                // SAFETY: mapped buffers; nothing runs.
                unsafe { std::ptr::write_bytes(budget_ptr, 0, 256) };
                if batched {
                    let mut srcs = [src_gpu; batch::SOURCES];
                    srcs[1] = other_gpu;
                    batch::record(
                        &pass,
                        &warp.list,
                        None,
                        false,
                        (&dst, D3D12_RESOURCE_STATE_COMMON),
                        Some((None, budget_gpu)),
                        srcs,
                        &[],
                        table_gpu,
                        plan_clamp_barriers(0, true, false, false, false, false),
                    );
                    let mut ids = [0u64; batch::SOURCES];
                    ids[0] = 1;
                    ids[1] = 2;
                    // SAFETY: the mapped table; the list has not run.
                    let mut table = unsafe { batch::Table::new(table_ptr, 0, ids) };
                    assert_eq!(table.place(1 + i as u64 % 2, offset, *kind), Some(0));
                } else {
                    assert!(record_clamp_as(
                        &warp.clamp,
                        &warp.list,
                        (&dst, D3D12_RESOURCE_STATE_COMMON),
                        budget_gpu,
                        (&src, offset),
                        None,
                        (*kind, stride, 1),
                        false,
                    ));
                }
                // SAFETY: an open list; the records are readable as arguments.
                unsafe {
                    warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    warp.list.CopyBufferRegion(&back, 0, &dst, 0, u64::from(stride));
                    warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                }
                warp.submit_and_wait();
                // SAFETY: the list has run; mapped words.
                let budget: Vec<u32> = (0..3).map(|k| unsafe { budget_ptr.cast::<u32>().add(k).read_volatile() }).collect();
                seen.push((read(back_ptr), budget));
            }
            // (A mesh record that asks for nothing notes nothing.)
            assert_eq!(seen[0], seen[1], "record {i} of kind {kind}: {:?}", records[i].1);
        }
    }

    /// Records past the list's budget come out empty, and one read by many
    /// executions is charged for each.
    #[test]
    fn warp_batch_charges_every_execution() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let pass = make_from(&warp.device, batch::SOURCE).expect("the batched clamp must compile on WARP");
        const BUDGET: u32 = 1 << 27;
        // 2^26 vertices each, as much as a draw is let run: two fill it.
        for (times, records, filled) in [(1u32, 3u64, 2usize), (2, 1, 1), (1, 1, 1), (3, 1, 0)] {
            let (_src, src_ptr, src_gpu) = warp.shared_buffer(64 << 10);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (back, back_ptr, _) = warp.shared_buffer(1024);
            let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
            let dst = warp.default_uav(64 << 10);
            batch::record(
                &pass,
                &warp.list,
                None,
                false,
                (&dst, D3D12_RESOURCE_STATE_COMMON),
                Some((None, budget_gpu)),
                [src_gpu; batch::SOURCES],
                &[],
                table_gpu,
                plan_clamp_barriers(0, true, false, false, false, false),
            );
            // SAFETY: mapped buffers, which the list has not read yet.
            let mut ids = [0u64; batch::SOURCES];
            ids[0] = 1;
            let mut table = unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
                for r in 0..records {
                    for (w, word) in [1u32 << 22, 16, 5, 6].into_iter().enumerate() {
                        src_ptr.cast::<u32>().add(r as usize * 4 + w).write_volatile(word);
                    }
                }
                batch::Table::new(table_ptr, 0, ids)
            };
            for _ in 0..times {
                for r in 0..records {
                    assert_eq!(table.place(1, r * 16, 0), Some(r * u64::from(batch::SLOT)));
                }
            }
            // SAFETY: an open list; the records are readable as arguments.
            unsafe {
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&back, 0, &dst, 0, 256);
            }
            warp.submit_and_wait();
            // SAFETY: the list has run; mapped words.
            let (spent, out): (u32, Vec<u32>) = unsafe {
                (
                    budget_ptr.cast::<u32>().read_volatile(),
                    (0..records as usize).map(|r| back_ptr.cast::<u32>().add(r * batch::SLOT as usize / 4).read_volatile()).collect(),
                )
            };
            assert_eq!(out.iter().filter(|&&v| v == 1 << 22).count(), filled, "{times} times {records}: {out:?}");
            assert!(out.iter().all(|&v| v == 0 || v == 1 << 22));
            assert_eq!(spent, BUDGET.min((1 << 26) * times * records as u32));
        }
    }

    /// Two batches through a host list, twice (the second recording finds
    /// the scratch buffer as the first one's end left it): executions with
    /// nothing between them read what they asked for, of two source buffers,
    /// the second batch while the first one's places are still read. With
    /// the debug layer on no barrier names a state the buffer is not in.
    #[test]
    fn host_list_batches() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        let signature = one_argument(&host.device, 2, 12);
        let mut sources = Vec::new();
        for first in [1u32, 100] {
            let Some((buffer, ptr)) = host.shared_buffer(1024) else {
                return;
            };
            for i in 0..16u32 {
                for (w, word) in [first + i, 1, 1].into_iter().enumerate() {
                    // SAFETY: inside the mapping.
                    unsafe { ptr.cast::<u32>().add(i as usize * 8 + w).write_volatile(word) };
                }
            }
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            let mut expected = 0;
            // More batches than one scratch buffer holds.
            for k in 0..12u32 {
                let (mut table, scratch) = open_batch(
                    &list,
                    (1, &sources[0], false),
                    &[(1, sources[0].clone(), false), (2, sources[1].clone(), false)],
                )
                .unwrap();
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                }
                for (id, record) in [(1u64, 3 + k % 4), (2, 5), (1, 0), (2, 5)] {
                    let at = table.place(id, u64::from(record) * 32, 2).unwrap();
                    // SAFETY: as above.
                    unsafe { list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0) };
                    expected += record + if id == 1 { 1 } else { 100 };
                }
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every execution runs what it asked for");
        }
    }

    /// What clamps of a frame cost on the card one by one and in batches
    /// (ignored: a measurement): 2,355 executions of an empty dispatch, in
    /// 49 lists' worth of runs, as in the captured frame.
    #[test]
    #[ignore]
    fn batch_costs() {
        let Some(mut gpu) = WarpHarness::on(false) else {
            return;
        };
        const EXECUTIONS: u32 = 2355;
        let pass = make_from(&gpu.device, batch::SOURCE).expect("the batched clamp must compile");
        let sig = gpu.signature(2, 12);
        let groups: Vec<u32> = vec![1; 2048];
        let src = dispatch_records(&gpu, &groups, 32);
        // SAFETY: a live buffer.
        let src_gpu = unsafe { src.GetGPUVirtualAddress() };
        let budget = gpu.default_uav(256);
        let hits = gpu.default_uav(256);
        // SAFETY: live buffers.
        let (budget_gpu, hits_gpu) = unsafe { (budget.GetGPUVirtualAddress(), hits.GetGPUVirtualAddress()) };
        let (_tables, tables_ptr, tables_gpu) = gpu.shared_buffer(batch::TABLE * 2400);
        let dst = gpu.default_uav(4 << 20);
        for run in [0u32, 2355 / 49, 5, 1] {
            let mut best = f64::MAX;
            for _ in 0..6 {
                let mut state = D3D12_RESOURCE_STATE_COMMON;
                let mut table = batch::Table::default();
                for i in 0..EXECUTIONS {
                    if run == 0 {
                        assert!(record_clamp_as(&gpu.clamp, &gpu.list, (&dst, state), budget_gpu, (&src, u64::from(i % 2048) * 32), None, (2, 12, 1), false));
                    } else if i % run == 0 {
                        let k = u64::from(i / run) * batch::TABLE;
                        batch::record(
                            &pass,
                            &gpu.list,
                            None,
                            false,
                            (&dst, state),
                            Some((None, budget_gpu)),
                            [src_gpu; batch::SOURCES],
                            &[],
                            tables_gpu + k,
                            plan_clamp_barriers(0, true, false, false, false, false),
                        );
                        let mut ids = [0u64; batch::SOURCES];
                        ids[0] = 1;
                        // SAFETY: the mapped table; the list has not run.
                        table = unsafe { batch::Table::new(tables_ptr.add(k as usize), 0, ids) };
                    }
                    state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                    let at = if run == 0 { 0 } else { table.place(1, u64::from(i % 2048) * 32, 2).unwrap() };
                    // SAFETY: an open list; live buffers.
                    unsafe {
                        if run == 0 || i % run == 0 {
                            gpu.list.SetComputeRootSignature(&gpu.worker.root);
                            gpu.list.SetPipelineState(&gpu.worker.pso);
                            gpu.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                        }
                        gpu.list.ExecuteIndirect(&sig, 1, &dst, at, None, 0);
                    }
                }
                let t = std::time::Instant::now();
                gpu.submit_and_wait();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            let what = if run == 0 { "each clamped on its own".to_owned() } else { format!("a batch every {run}") };
            println!("{EXECUTIONS} indirect dispatches, {what}: {best:.2} ms");
        }
    }

    #[test]
    fn clamp_escape_hatch_and_bypass_validation() {
        use std::ffi::OsStr;
        assert!(clamp_enabled_for(None));
        assert!(clamp_enabled_for(Some(OsStr::new("1"))));
        assert!(clamp_enabled_for(Some(OsStr::new(""))));
        assert!(!clamp_enabled_for(Some(OsStr::new("0"))));

        // Native PC single-arg kinds with valid stride & alignment can bypass.
        assert!(can_bypass_clamp(KIND_DRAW, 16, 0, None));
        assert!(can_bypass_clamp(KIND_DRAW_INDEXED, 20, 64, Some(128)));
        assert!(can_bypass_clamp(KIND_DISPATCH, 12, 4, None));
        assert!(can_bypass_clamp(KIND_DISPATCH_MESH, 12, 0, None));
        // 1D mesh indirect MUST NOT bypass (requires 1->3 word expansion).
        assert!(!can_bypass_clamp(KIND_DISPATCH_MESH_1D, 4, 0, None));
        // Undersized / unaligned layouts cannot bypass.
        assert!(!can_bypass_clamp(KIND_DRAW, 12, 0, None));
        assert!(!can_bypass_clamp(KIND_DISPATCH, 12, 2, None));
        assert!(!can_bypass_clamp(KIND_DISPATCH, 12, 0, Some(6)));
    }

    #[test]
    fn translate_signature_maps_constants_descriptors_and_rejects_invalid() {
        use victory_shader::RootType;
        use victory_shader::emit::{USER_ROOT_ADDRESS, UserData};

        let vs_user = [
            // Console slot 1, dwords 2..4 -> host param 0, indices 5..7 (contiguous)
            UserData { index: 5, slot: 1, root: RootType::Constants, dword: 2 },
            UserData { index: 6, slot: 1, root: RootType::Constants, dword: 3 },
            // Console slot 1, dword 0 -> host param 0, index 7 (contiguous dst, non-contiguous src)
            UserData { index: 7, slot: 1, root: RootType::Constants, dword: 0 },
            // Console slot 2 root SRV raw guest VA -> host param 0, indices 10..12
            UserData { index: 10, slot: 2, root: RootType::Other(USER_ROOT_ADDRESS), dword: 0 },
            UserData { index: 11, slot: 2, root: RootType::Other(USER_ROOT_ADDRESS + 1), dword: 0 },
        ];
        let ps_user = [
            // Pixel stage also reads console slot 1, dword 2 -> host param 3, index 0
            UserData { index: 0, slot: 1, root: RootType::Constants, dword: 2 },
        ];
        let vs_cbvs = [(0u8, 1u32)];
        let vs_srvs = [(2u8, 2u32)];
        let ps_uavs = [(3u8, 4u32)];
        let stages = [
            StageBinding {
                user_data_param: Some(0),
                user_data: &vs_user,
                root_cbvs: &vs_cbvs,
                root_srvs: &vs_srvs,
                root_uavs: &[],
            },
            StageBinding {
                user_data_param: Some(3),
                user_data: &ps_user,
                root_cbvs: &[],
                root_srvs: &[],
                root_uavs: &ps_uavs,
            },
        ];

        // Console record layout:
        // [0..8)   CBV slot 0
        // [8..24)  CONSTANT slot 1, dest_offset=0, num=4 (dwords 0,1,2,3; dword 1 is unused!)
        // [24..32) SRV slot 2
        // [32..40) UAV slot 3
        // [40..48) CONSTANT slot 9 (completely unused by shader -> skipped cleanly)
        // [48..64) DRAW (16 bytes)
        let args = [
            [KIND_CBV, 0, 0, 0],
            [KIND_CONSTANT, 1, 0, 4],
            [KIND_SRV, 2, 0, 0],
            [KIND_UAV, 3, 0, 0],
            [KIND_CONSTANT, 9, 0, 2],
            [KIND_DRAW, 0, 0, 0],
        ];

        // With direct root SRVs/UAVs (`bounded_root_srvs = false`): CBV, SRV, UAV all translated.
        let plan = translate_signature(64, &args, &stages, false).expect("valid multi-arg signature");
        assert_eq!(plan.exec_kind, KIND_DRAW);
        assert_eq!(plan.src_exec_off, 48);
        assert!(plan.needs_rebase);
        assert!(plan.has_root_args);
        assert!(!plan.skipped_tabled_views);
        assert_eq!(
            plan.host_args,
            vec![
                HostIndirectArg::Constant { param: 0, dest_offset: 5, count: 3 },
                HostIndirectArg::Constant { param: 0, dest_offset: 10, count: 2 },
                HostIndirectArg::Constant { param: 3, dest_offset: 0, count: 1 },
                HostIndirectArg::Cbv { param: 1 },
                HostIndirectArg::Srv { param: 2 },
                HostIndirectArg::Uav { param: 4 },
                HostIndirectArg::Exec { kind: KIND_DRAW },
            ]
        );
        // 6 constant dwords (24 bytes) + 3 descriptors (24 bytes) = 48 bytes before DRAW.
        assert_eq!(plan.dst_exec_off, 48);
        assert_eq!(plan.dst_stride, 64);
        assert_eq!(
            plan.ops,
            vec![
                [OP_COPY_DWORDS, 16, 0, 2],  // slot 1 dwords 2..4 -> dst 0..8
                [OP_COPY_DWORD, 8, 8, 1],    // slot 1 dword 0 -> dst 8..12
                [OP_COPY_DWORDS, 24, 12, 2], // slot 2 guest VA -> USER_ROOT_ADDRESS dst 12..20
                [OP_COPY_DWORD, 16, 20, 1],  // PS slot 1 dword 2 -> dst 20..24
                [OP_REBASE_CBV, 0, 24, 0],   // CBV slot 0 -> dst 24..32
                [OP_REBASE_SRV, 24, 32, 0],  // SRV slot 2 -> dst 32..40
                [OP_REBASE_UAV, 32, 40, 0],  // UAV slot 3 -> dst 40..48
            ]
        );

        // With `bounded_root_srvs = true`: SRV/UAV are flagged as skipped tabled views,
        // while CBV, constants, and USER_ROOT_ADDRESS still translate cleanly.
        let bounded = translate_signature(64, &args, &stages, true).expect("valid bounded plan");
        assert!(bounded.skipped_tabled_views);
        assert!(bounded.needs_rebase); // CBV still needs rebase
        assert_eq!(
            bounded.host_args,
            vec![
                HostIndirectArg::Constant { param: 0, dest_offset: 5, count: 3 },
                HostIndirectArg::Constant { param: 0, dest_offset: 10, count: 2 },
                HostIndirectArg::Constant { param: 3, dest_offset: 0, count: 1 },
                HostIndirectArg::Cbv { param: 1 },
                HostIndirectArg::Exec { kind: KIND_DRAW },
            ]
        );

        // All-unused constants collapse to a root-arg-free signature (pRootSignature = None)
        // and normalize DISPATCH_SINGLE (105) -> DISPATCH (2).
        let dead_only = [[KIND_CONSTANT, 9, 0, 2], [KIND_DISPATCH_SINGLE, 0, 0, 0]];
        let dead_plan = translate_signature(24, &dead_only, &stages, true).expect("dead constants ok");
        assert_eq!(dead_plan.exec_kind, KIND_DISPATCH);
        assert!(!dead_plan.has_root_args);
        assert!(!dead_plan.needs_rebase);
        assert_eq!(dead_plan.src_exec_off, 8);
        assert_eq!(dead_plan.dst_exec_off, 0);
        assert_eq!(dead_plan.dst_stride, 12);
        assert!(dead_plan.ops.is_empty());

        // Reject undersized stride, unaligned stride, or unsupported arg kind (VB=3).
        assert!(translate_signature(60, &args, &stages, false).is_none());
        assert!(translate_signature(66, &args, &stages, false).is_none());
        assert!(translate_signature(32, &[[3, 0, 0, 0], [KIND_DRAW, 0, 0, 0]], &stages, false).is_none());
        assert!(translate_signature(16, &[], &stages, false).is_none());
    }

    #[test]
    fn build_rebase_table_resolves_nested_overlaps() {
        // Outer buffer [0x1000, 0x5000) -> host 0xA0000 (read-only),
        // nested sub-buffer [0x2000, 0x3000) -> host 0xB0000 (writable).
        let table = build_rebase_table(&[
            (0x1000, 0x4000, 0xA_0000, false),
            (0x2000, 0x1000, 0xB_0000, true),
            (0, 0x1000, 0xC_0000, true), // invalid guest=0 filtered
        ]);
        assert_eq!(table.len(), 3);
        let decode = |e: [u32; 8]| {
            (
                u64::from(e[0]) | (u64::from(e[1]) << 32),
                u64::from(e[2]) | (u64::from(e[3]) << 32),
                u64::from(e[4]) | (u64::from(e[5]) << 32),
                e[6] != 0,
            )
        };
        assert_eq!(decode(table[0]), (0x1000, 0x2000, 0xA_0000, false));
        assert_eq!(decode(table[1]), (0x2000, 0x3000, 0xB_0000, true));
        assert_eq!(decode(table[2]), (0x3000, 0x5000, 0xA_2000, false));
    }

    /// Verifies 1D indirect mesh dispatch clamping (`KIND_DISPATCH_MESH_1D = 11`,
    /// used by `DispatchMeshX`) in both `SOURCE` and `batch::SOURCE`:
    /// - 1-word records followed by zeros (`y=0, z=0`) or garbage expand to `[x, 1, 1]`
    /// - `x == 0`, `x == 65536`, and `u32::MAX` zero out to `[0, 0, 0]` at 0 budget cost
    /// - budget cap is enforced across large mesh dispatches
    #[test]
    fn warp_mesh_1d_clamp_valid_garbage_and_budget() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let batch_pass = make_from(&warp.device, batch::SOURCE).expect("batch clamp compiles on WARP");
        // 5 records spaced 16 bytes apart; words 1..3 hold zeros or garbage that
        // `KIND_DISPATCH_MESH_1D` must ignore.
        let inputs: [[u32; 4]; 5] = [
            [7, 0, 0, 0],                        // valid 1D with trailing zeros -> [7, 1, 1]
            [65_535, 0xDEAD_BEEF, 0, 0x1234],    // max valid 1D with garbage y/z -> [65535, 1, 1]
            [0, 1, 1, 0],                        // zero x -> [0, 0, 0]
            [65_536, 1, 1, 0],                   // over 65535 -> [0, 0, 0]
            [u32::MAX, u32::MAX, u32::MAX, 0],   // unfilled garbage -> [0, 0, 0]
        ];
        let expected: [[u32; 3]; 5] = [
            [7, 1, 1],
            [65_535, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ];
        let (src, src_ptr, src_gpu) = warp.shared_buffer(1024);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(1024);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let dst = warp.default_uav(4096);
        // SAFETY: mapped buffers with ample capacity.
        unsafe {
            for (i, rec) in inputs.iter().enumerate() {
                for (w, &v) in rec.iter().enumerate() {
                    src_ptr.cast::<u32>().add(i * 4 + w).write_volatile(v);
                }
            }
        }
        // Test 1: single-record clamp (`SOURCE`) with stride=4 per record.
        for (i, want) in expected.iter().enumerate() {
            // SAFETY: mapped buffers.
            unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(back_ptr, 0xAA, 64);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src, (i * 16) as u64),
                None,
                (KIND_DISPATCH_MESH_1D, 4, 1),
                false,
            ));
            // SAFETY: open list, live buffers.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&back, 0, &dst, 0, 12);
                warp.list.ResourceBarrier(&[transition(
                    &dst,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: completed work, mapped buffers.
            let (out, spent) = unsafe {
                let r = back_ptr.cast::<u32>();
                ([r.read_volatile(), r.add(1).read_volatile(), r.add(2).read_volatile()], budget_ptr.cast::<u32>().read_volatile())
            };
            assert_eq!(out, *want, "single KIND_DISPATCH_MESH_1D record {i}");
            assert_eq!(spent, want[0] * 128, "single KIND_DISPATCH_MESH_1D budget {i}");
        }

        // Test 2: batched clamp (`batch::SOURCE`) with all 5 records + budget exhaustion check.
        // SAFETY: mapped buffers.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(back_ptr, 0xBB, 512);
        }
        batch::record(
            &batch_pass,
            &warp.list,
            None,
            false,
            (&dst, D3D12_RESOURCE_STATE_COMMON),
            Some((None, budget_gpu)),
            [src_gpu; batch::SOURCES],
            &[],
            table_gpu,
            plan_clamp_barriers(0, true, false, false, false, false),
        );
        let mut ids = [0u64; batch::SOURCES];
        ids[0] = 1;
        // SAFETY: mapped table before submission.
        let mut table = unsafe { batch::Table::new(table_ptr, 0, ids) };
        let places: Vec<u64> = (0..inputs.len())
            .map(|i| table.place(1, (i * 16) as u64, KIND_DISPATCH_MESH_1D).unwrap())
            .collect();
        // SAFETY: open list, live buffers.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&back, 0, &dst, 0, (inputs.len() as u64) * u64::from(batch::SLOT));
        }
        warp.submit_and_wait();
        for (i, (&at, want)) in places.iter().zip(&expected).enumerate() {
            // SAFETY: completed work, mapped buffer.
            let out = unsafe {
                let r = back_ptr.add(at as usize).cast::<u32>();
                [r.read_volatile(), r.add(1).read_volatile(), r.add(2).read_volatile()]
            };
            assert_eq!(out, *want, "batched KIND_DISPATCH_MESH_1D record {i}");
        }
        // SAFETY: completed work.
        let (spent, largest) = unsafe {
            (budget_ptr.cast::<u32>().read_volatile(), budget_ptr.cast::<u32>().add(1).read_volatile())
        };
        assert_eq!(spent, (7 + 65_535) * 128);
        assert_eq!(largest, 65_535);
    }

    /// End-to-end WARP GPU execution test of multi-argument `ExecuteIndirect`
    /// with `D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT` + `DISPATCH`:
    /// repacks console records containing unused constants + remapped constants
    /// + dispatch counts (plus a garbage record), executes them on WARP, and
    /// verifies the compute shader receives the exact indirect root constants
    /// while preserving statically bound root constants!
    #[test]
    fn warp_repack_root_constants_and_execute_indirect() {
        use victory_shader::RootType;
        use victory_shader::emit::UserData;

        const CONST_WORKER_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer Params : register(b0) {
    uint base_slot;
    uint tag_a;
    uint tag_b;
    uint static_magic;
};
RWByteAddressBuffer out_buf : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
    uint prev;
    out_buf.InterlockedAdd(0, 1, prev);
    uint slot = base_slot + gid.x;
    out_buf.Store(16 + slot * 16, tag_a);
    out_buf.Store(16 + slot * 16 + 4, tag_b);
    out_buf.Store(16 + slot * 16 + 8, static_magic);
}
"#;

        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let repack = make_from(&warp.device, REPACK_SOURCE).expect("REPACK_SOURCE compiles on WARP");
        let worker = compile_cs(&warp.device, CONST_WORKER_CS).expect("const worker compiles on WARP");

        // Console signature:
        // [0..16) CONSTANT console_slot=1, dest=0, num=4 (dwords 0,1,2,3; dword 3 unused!)
        // [16..28) DISPATCH (x, y, z)
        // Stride = 32 (4 bytes trailing padding)
        let console_args = [[KIND_CONSTANT, 1, 0, 4], [KIND_DISPATCH, 0, 0, 0]];
        let user_data = [
            // dword 0 -> host b0[0] (`base_slot`)
            UserData { index: 0, slot: 1, root: RootType::Constants, dword: 0 },
            // dword 1 -> host b0[1] (`tag_a`)
            UserData { index: 1, slot: 1, root: RootType::Constants, dword: 1 },
            // dword 2 -> host b0[2] (`tag_b`)
            UserData { index: 2, slot: 1, root: RootType::Constants, dword: 2 },
        ];
        let stages = [StageBinding {
            user_data_param: Some(0),
            user_data: &user_data,
            root_cbvs: &[],
            root_srvs: &[],
            root_uavs: &[],
        }];
        let plan = translate_signature(32, &console_args, &stages, true).expect("valid plan");
        assert_eq!(
            plan.host_args,
            vec![
                HostIndirectArg::Constant { param: 0, dest_offset: 0, count: 3 },
                HostIndirectArg::Exec { kind: KIND_DISPATCH },
            ]
        );
        assert_eq!(plan.dst_stride, 24);

        let sig_args = [
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                    Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                        RootParameterIndex: 0,
                        DestOffsetIn32BitValues: 0,
                        Num32BitValuesToSet: 3,
                    },
                },
            },
            D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                ..Default::default()
            },
        ];
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: plan.dst_stride,
            NumArgumentDescs: sig_args.len() as u32,
            pArgumentDescs: sig_args.as_ptr(),
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: valid command signature against worker's root signature.
        unsafe {
            warp.device
                .CreateCommandSignature(&sig_desc, Some(&worker.root), &mut sig)
                .unwrap();
        }
        let sig = sig.unwrap();

        // 3 records (32 bytes each):
        // rec 0: base_slot=0, tag_a=0x1111, tag_b=0x2222, unused=0xDEAD, dispatch (2, 1, 1)
        // rec 1: base_slot=2, tag_a=0x3333, tag_b=0x4444, unused=0xBEEF, dispatch (3, 1, 1)
        // rec 2: garbage > GROUPS -> clamped to (0, 0, 0) so never runs
        let records: [[u32; 8]; 3] = [
            [0, 0x1111, 0x2222, 0xDEAD, 2, 1, 1, 0],
            [2, 0x3333, 0x4444, 0xBEEF, 3, 1, 1, 0],
            [5, 0x9999, 0x8888, 0xFFFF, 65_535, 65_535, 1, 0],
        ];
        let (src, src_ptr, _) = warp.shared_buffer(256);
        let (_ops, ops_ptr, ops_gpu) = warp.shared_buffer(256);
        let (_rebase, rebase_ptr, rebase_gpu) = warp.shared_buffer(256);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_out, out_ptr, out_gpu) = warp.shared_buffer(1024);
        let dst = warp.default_uav(1024);
        // SAFETY: mapped buffers with ample capacity.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(out_ptr, 0, 1024);
            std::ptr::write_bytes(rebase_ptr, 0, 256);
            for (i, r) in records.iter().enumerate() {
                std::ptr::copy_nonoverlapping(r.as_ptr().cast::<u8>(), src_ptr.add(i * 32), 32);
            }
            std::ptr::copy_nonoverlapping(
                plan.ops.as_ptr().cast::<u8>(),
                ops_ptr,
                plan.ops.len() * 16,
            );
        }

        assert!(record_repack_with(
            &repack,
            &warp.list,
            None,
            false,
            (&dst, D3D12_RESOURCE_STATE_COMMON),
            Some((None, budget_gpu)),
            (&src, 0),
            96,
            None,
            &plan,
            (ops_gpu, plan.ops.len() as u32),
            (rebase_gpu, 0),
            3,
            plan_clamp_barriers(0, true, false, false, false, true),
        ));
        // Bind static constant index 3 (`static_magic = 0xCAFE_BABE`) before ExecuteIndirect;
        // ExecuteIndirect must overwrite only 0..3 and preserve index 3!
        let init_consts = [0u32, 0, 0, 0xCAFE_BABE];
        // SAFETY: open list, live worker pipeline/root/buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&worker.root);
            warp.list.SetPipelineState(&worker.pso);
            warp.list
                .SetComputeRoot32BitConstants(0, 4, init_consts.as_ptr().cast(), 0);
            warp.list.SetComputeRootUnorderedAccessView(1, out_gpu);
            warp.list.ExecuteIndirect(&sig, 3, &dst, 0, None, 0);
        }
        warp.submit_and_wait();

        // SAFETY: completed GPU work, mapped output buffer.
        let (total_groups, slots) = unsafe {
            let r = out_ptr.cast::<u32>();
            let total = r.read_volatile();
            let slots: Vec<[u32; 3]> = (0..6)
                .map(|s| {
                    let base = 4 + s * 4;
                    [
                        r.add(base).read_volatile(),
                        r.add(base + 1).read_volatile(),
                        r.add(base + 2).read_volatile(),
                    ]
                })
                .collect();
            (total, slots)
        };
        assert_eq!(total_groups, 5, "only 2+3 valid groups should run (garbage clamped to 0)");
        assert_eq!(slots[0], [0x1111, 0x2222, 0xCAFE_BABE]);
        assert_eq!(slots[1], [0x1111, 0x2222, 0xCAFE_BABE]);
        assert_eq!(slots[2], [0x3333, 0x4444, 0xCAFE_BABE]);
        assert_eq!(slots[3], [0x3333, 0x4444, 0xCAFE_BABE]);
        assert_eq!(slots[4], [0x3333, 0x4444, 0xCAFE_BABE]);
        assert_eq!(slots[5], [0, 0, 0], "garbage record 2 must not execute");
    }

    /// WARP GPU test for 64-bit guest VA -> host GPU VA rebasing in `REPACK_SOURCE`:
    /// verifies valid CBV/SRV/UAV VAs rebase across 32-bit boundaries (including
    /// nested shadow ranges), while unaligned CBVs, read-only UAVs, and unmapped
    /// VAs fall back safely and zero out the trailing draw counts.
    #[test]
    fn warp_repack_root_descriptors_rebase_and_guard() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let repack = make_from(&warp.device, REPACK_SOURCE).expect("REPACK_SOURCE compiles on WARP");

        let stages = [StageBinding {
            user_data_param: None,
            user_data: &[],
            root_cbvs: &[(0, 0)],
            root_srvs: &[(1, 1)],
            root_uavs: &[(2, 2)],
        }];
        // [0..8) CBV, [8..16) SRV, [16..24) UAV, [24..40) DRAW
        let console_args = [
            [KIND_CBV, 0, 0, 0],
            [KIND_SRV, 1, 0, 0],
            [KIND_UAV, 2, 0, 0],
            [KIND_DRAW, 0, 0, 0],
        ];
        let plan = translate_signature(40, &console_args, &stages, false).expect("valid plan");
        assert!(plan.needs_rebase);
        assert_eq!(plan.dst_exec_off, 24);
        assert_eq!(plan.dst_stride, 40);

        // Two overlapping ranges above 4 GiB to exercise full 64-bit arithmetic!
        let rebase = build_rebase_table(&[
            (0x1_0000_0000, 0x10_0000, 0x2_0000_0000, false), // outer read-only
            (0x1_0004_0000, 0x01_0000, 0x3_0000_0000, true),  // inner writable
        ]);

        // 4 records:
        // 0: valid CBV (256B aligned in outer), valid SRV (4B aligned in outer), valid UAV (in inner writable) -> kept
        // 1: CBV not 256B aligned (+16) -> invalid, draw zeroed
        // 2: UAV pointing to read-only outer -> invalid, draw zeroed
        // 3: unmapped guest VA -> invalid, draw zeroed
        let mut src_bytes = vec![0u8; 4 * 40];
        let mut put_rec = |idx: usize, cbv: u64, srv: u64, uav: u64, draw: [u32; 4]| {
            let b = idx * 40;
            src_bytes[b..b + 8].copy_from_slice(&cbv.to_le_bytes());
            src_bytes[b + 8..b + 16].copy_from_slice(&srv.to_le_bytes());
            src_bytes[b + 16..b + 24].copy_from_slice(&uav.to_le_bytes());
            for (w, v) in draw.into_iter().enumerate() {
                src_bytes[b + 24 + w * 4..b + 28 + w * 4].copy_from_slice(&v.to_le_bytes());
            }
        };
        put_rec(0, 0x1_0000_0200, 0x1_0000_0104, 0x1_0004_0080, [36, 2, 10, 5]);
        put_rec(1, 0x1_0000_0010, 0x1_0000_0104, 0x1_0004_0080, [36, 2, 10, 5]);
        put_rec(2, 0x1_0000_0200, 0x1_0000_0104, 0x1_0000_0080, [36, 2, 10, 5]);
        put_rec(3, 0x9_9999_0000, 0x1_0000_0104, 0x1_0004_0080, [36, 2, 10, 5]);

        let (src, src_ptr, _) = warp.shared_buffer(512);
        let (_ops, ops_ptr, ops_gpu) = warp.shared_buffer(256);
        let (_rebase_buf, rebase_ptr, rebase_gpu) = warp.shared_buffer(256);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(512);
        let dst = warp.default_uav(1024);
        // SAFETY: live resource.
        let dst_gpu = unsafe { dst.GetGPUVirtualAddress() };
        // SAFETY: mapped buffers with ample capacity.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), src_ptr, src_bytes.len());
            std::ptr::copy_nonoverlapping(plan.ops.as_ptr().cast::<u8>(), ops_ptr, plan.ops.len() * 16);
            std::ptr::copy_nonoverlapping(rebase.as_ptr().cast::<u8>(), rebase_ptr, rebase.len() * 32);
        }

        assert!(record_repack_with(
            &repack,
            &warp.list,
            None,
            false,
            (&dst, D3D12_RESOURCE_STATE_COMMON),
            Some((None, budget_gpu)),
            (&src, 0),
            src_bytes.len() as u64,
            None,
            &plan,
            (ops_gpu, plan.ops.len() as u32),
            (rebase_gpu, rebase.len() as u32),
            4,
            plan_clamp_barriers(0, true, false, false, false, true),
        ));
        // SAFETY: open list, live buffers.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&back, 0, &dst, 0, src_bytes.len() as u64);
        }
        warp.submit_and_wait();

        let read_rec = |idx: usize| -> (u64, u64, u64, [u32; 4]) {
            // SAFETY: completed work, mapped buffer.
            let slice = unsafe { std::slice::from_raw_parts(back_ptr.add(idx * 40).cast_const(), 40) };
            let u64_at = |o: usize| u64::from_le_bytes(slice[o..o + 8].try_into().unwrap());
            let u32_at = |o: usize| u32::from_le_bytes(slice[o..o + 4].try_into().unwrap());
            (
                u64_at(0),
                u64_at(8),
                u64_at(16),
                [u32_at(24), u32_at(28), u32_at(32), u32_at(36)],
            )
        };

        assert_eq!(
            read_rec(0),
            (0x2_0000_0200, 0x2_0000_0104, 0x3_0000_0080, [36, 2, 10, 5]),
            "valid 64-bit guest VAs must rebase cleanly"
        );
        // Invalid records must zero draw counts and substitute valid fallback GPU VA.
        let r1 = read_rec(1);
        assert_eq!(r1.0, dst_gpu, "unaligned CBV must use fallback VA");
        assert_eq!(r1.3, [0, 0, 10, 5], "unaligned CBV must zero draw");

        let r2 = read_rec(2);
        assert_eq!(r2.2, dst_gpu, "read-only UAV must use fallback VA");
        assert_eq!(r2.3, [0, 0, 10, 5], "read-only UAV must zero draw");

        let r3 = read_rec(3);
        assert_eq!(r3.0, dst_gpu, "unmapped CBV must use fallback VA");
        assert_eq!(r3.3, [0, 0, 10, 5], "unmapped CBV must zero draw");

        // Only record 0 charged the budget (36 * 2 = 72).
        // SAFETY: completed work.
        let spent = unsafe { budget_ptr.cast::<u32>().read_volatile() };
        assert_eq!(spent, 72);
    }

    /// Verifies the zero-cost group fast path (skipping global budget atomics)
    /// and early group exit past `count` in both `SOURCE` and `batch::SOURCE`.
    #[test]
    fn warp_zero_cost_and_empty_tail_groups_leave_budget_untouched() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let batch_pass = make_from(&warp.device, batch::SOURCE).expect("batch clamp compiles on WARP");
        let (_src, src_ptr, src_gpu) = warp.shared_buffer(4096);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back, back_ptr, _) = warp.shared_buffer(4096);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let dst = warp.default_uav(8192);
        // SAFETY: mapped buffers.
        unsafe {
            // 4 zero-vertex draws -> cost 0 each (group_cost == 0); tail groups 1..3 exit early.
            std::ptr::write_bytes(src_ptr, 0, 4096);
            for i in 0..4usize {
                src_ptr.cast::<u32>().add(i * 4 + 2).write_volatile(100 + i as u32);
            }
            // Pre-seed budget word 0 to a known value; zero-cost group must not alter it.
            std::ptr::write_bytes(budget_ptr, 0, 256);
            budget_ptr.cast::<u32>().write_volatile(1234);
        }
        batch::record(
            &batch_pass,
            &warp.list,
            None,
            false,
            (&dst, D3D12_RESOURCE_STATE_COMMON),
            Some((None, budget_gpu)),
            [src_gpu; batch::SOURCES],
            &[],
            table_gpu,
            plan_clamp_barriers(0, true, false, false, false, false),
        );
        let mut ids = [0u64; batch::SOURCES];
        ids[0] = 1;
        // SAFETY: mapped table.
        let mut table = unsafe { batch::Table::new(table_ptr, 0, ids) };
        for i in 0..4u64 {
            assert!(table.place(1, i * 16, KIND_DRAW).is_some());
        }
        // SAFETY: open list.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&back, 0, &dst, 0, 4 * u64::from(batch::SLOT));
        }
        warp.submit_and_wait();
        // SAFETY: completed work.
        let spent = unsafe { budget_ptr.cast::<u32>().read_volatile() };
        assert_eq!(spent, 1234, "zero-cost batch group must not modify budget");
        for i in 0..4usize {
            // SAFETY: completed work.
            let rec = unsafe {
                let r = back_ptr.add(i * batch::SLOT as usize).cast::<u32>();
                [r.read_volatile(), r.add(1).read_volatile(), r.add(2).read_volatile()]
            };
            assert_eq!(rec, [0, 0, 100 + i as u32]);
        }
    }

    /// Verifies `KIND_DISPATCH_SPREAD = 12` inside `batch::SOURCE`, 8-source
    /// indexing (`src0..src7`), and the release `BUDGET_TRACK = 0` shader
    /// variant (zero `groupshared` / atomics) matching `BUDGET_TRACK = 1`.
    #[test]
    fn warp_batch_spread_and_budget_free_clamp() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let batch_full = make_from_with(&warp.device, batch::SOURCE, true)
            .expect("batch clamp (BUDGET_TRACK=1) compiles on WARP");
        let batch_lean = make_from_with(&warp.device, batch::SOURCE, false)
            .expect("batch clamp (BUDGET_TRACK=0) compiles on WARP");
        let single_lean = make_from_with(&warp.device, SOURCE, false)
            .expect("single clamp (BUDGET_TRACK=0) compiles on WARP");

        // Test spread records (kind 12) + draw + dispatch + mesh + mesh_1d,
        // including source slot 7 (testing 3-bit source mask `job.x & 7`).
        let records: &[(u32, [u32; 5])] = &[
            (KIND_DISPATCH_SPREAD, [100, 1, 1, 0, 0]),          // small -> unspread [0, 100, 100, 1, 1]
            (KIND_DISPATCH_SPREAD, [65_535, 2, 1, 0, 0]),       // exact max -> unspread
            (KIND_DISPATCH_SPREAD, [115_200, 1, 1, 0, 0]),      // >65535 -> spread across z=2, row=57600
            (KIND_DISPATCH_SPREAD, [1 << 21, 1, 1, 0, 0]),      // >GROUPS -> zeroed groups
            (KIND_DRAW, [120, 4, 10, 20, 0]),                   // valid draw
            (KIND_DRAW, [1 << 23, 2, 10, 20, 0]),               // >= MOST_DRAW -> zeroed
            (KIND_DISPATCH, [300, 2, 1, 0, 0]),                 // valid compute dispatch
            (KIND_DISPATCH_MESH, [40_000, 5, 9, 0, 0]),         // mesh clamps y/z to 1
            (KIND_DISPATCH_MESH, [65_536, 1, 1, 0, 0]),         // mesh >65535 -> zeroed
            (KIND_DISPATCH_MESH_1D, [77, 0xDEAD, 0xBEEF, 0, 0]),// 1D mesh -> [77, 1, 1]
            (KIND_DISPATCH_MESH_1D, [u32::MAX, 1, 1, 0, 0]),    // 1D mesh garbage -> [0, 0, 0]
        ];

        let (src0, src0_ptr, src0_gpu) = warp.shared_buffer(4096);
        let (_src7, src7_ptr, src7_gpu) = warp.shared_buffer(4096);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (back_full, back_full_ptr, _) = warp.shared_buffer(4096);
        let (back_lean, back_lean_ptr, _) = warp.shared_buffer(4096);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
        let dst = warp.default_uav(8192);
        // SAFETY: mapped buffers with ample space.
        unsafe {
            std::ptr::write_bytes(src0_ptr, 0, 4096);
            std::ptr::write_bytes(src7_ptr, 0, 4096);
            for (i, &(_, words)) in records.iter().enumerate() {
                for (w, &val) in words.iter().enumerate() {
                    src0_ptr.cast::<u32>().add(i * 8 + w).write_volatile(val);
                    src7_ptr.cast::<u32>().add(i * 8 + w).write_volatile(val);
                }
            }
        }

        let mut srcs = [src0_gpu; batch::SOURCES];
        srcs[7] = src7_gpu;
        let mut ids = [0u64; batch::SOURCES];
        ids[0] = 10;
        ids[7] = 77;

        for (lean, back) in [(false, &back_full), (true, &back_lean)] {
            // SAFETY: mapped buffers; nothing runs.
            unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
            }
            let pass = if lean { &batch_lean } else { &batch_full };
            let budget_arg = (!lean).then_some((None, budget_gpu));
            batch::record(
                pass,
                &warp.list,
                None,
                false,
                (&dst, D3D12_RESOURCE_STATE_COMMON),
                budget_arg,
                srcs,
                &[],
                table_gpu,
                plan_clamp_barriers(0, true, false, false, false, false),
            );
            // SAFETY: mapped table before submit.
            let mut table = unsafe { batch::Table::new(table_ptr, 0, ids) };
            for (i, &(kind, _)) in records.iter().enumerate() {
                let id = if i % 2 == 1 { 77 } else { 10 };
                assert_eq!(table.place(id, (i * 32) as u64, kind), Some((i as u64) * u64::from(batch::SLOT)));
            }
            // SAFETY: open list, live buffers.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(
                    back,
                    0,
                    &dst,
                    0,
                    (records.len() as u64) * u64::from(batch::SLOT),
                );
                warp.list.ResourceBarrier(&[transition(
                    &dst,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
        }

        for (i, &(kind, _)) in records.iter().enumerate() {
            let out_words = match kind {
                KIND_DISPATCH_SPREAD => 5,
                KIND_DISPATCH_MESH_1D => 3,
                _ => batch::words(kind).unwrap() as usize,
            };
            // SAFETY: completed work, mapped readbacks.
            let (full_out, lean_out): (Vec<u32>, Vec<u32>) = unsafe {
                let f = back_full_ptr.add(i * batch::SLOT as usize).cast::<u32>();
                let l = back_lean_ptr.add(i * batch::SLOT as usize).cast::<u32>();
                (
                    (0..out_words).map(|w| f.add(w).read_volatile()).collect(),
                    (0..out_words).map(|w| l.add(w).read_volatile()).collect(),
                )
            };
            assert_eq!(full_out, lean_out, "BUDGET_TRACK=0 vs 1 mismatch on record {i} (kind {kind})");
            if i == 0 {
                assert_eq!(full_out, [0, 100, 100, 1, 1]);
            } else if i == 2 {
                assert_eq!(full_out, [57_600, 115_200, 57_600, 1, 2]);
            } else if i == 3 {
                assert_eq!(full_out, [63_551, 1 << 21, 0, 0, 0]);
            }
        }

        // Also verify single-record `SOURCE` with `BUDGET_TRACK=0` runs cleanly.
        assert!(record_clamp_with(
            &single_lean,
            &warp.list,
            None,
            false,
            (&dst, D3D12_RESOURCE_STATE_COMMON),
            None,
            (&src0, 64), // record 2: [115_200, 1, 1]
            None,
            (KIND_DISPATCH, 12, 1),
            false,
            true,
            plan_clamp_barriers(0, true, false, false, false, false),
        ));
        // SAFETY: open list, live buffers.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&back_lean, 0, &dst, 0, 20);
        }
        warp.submit_and_wait();
        // SAFETY: completed work.
        let single_out: Vec<u32> = (0..5)
            .map(|w| unsafe { back_lean_ptr.cast::<u32>().add(w).read_volatile() })
            .collect();
        assert_eq!(single_out, [57_600, 115_200, 57_600, 1, 2]);
    }

    /// Exercises Enhanced Barriers end-to-end on a real `HostList` (including
    /// UAV-written argument source, `open_batch` with in-place MRU segregation,
    /// `emit_global_uav_barrier` interop, `bound`, `sync_bypass_indirect`, and
    /// `list.close()` / `reset()` across two rounds) with zero D3D12 Debug Layer errors.
    #[test]
    fn host_list_enhanced_clamp_and_bypass_debug_layer() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        let signature = one_argument(&host.device, 2, 12);
        let Some((src_normal, src_normal_ptr)) = host.shared_buffer(1024) else {
            return;
        };
        let Some((src_inplace, src_inplace_ptr)) = host.shared_buffer(1024) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers with ample capacity.
        unsafe {
            for (ptr, val) in [(src_normal_ptr, 3u32), (src_inplace_ptr, 5u32)] {
                std::ptr::write_bytes(ptr, 0, 1024);
                for (w, word) in [val, 1, 1].into_iter().enumerate() {
                    ptr.cast::<u32>().add(w).write_volatile(word);
                }
            }
        }
        // SAFETY: live buffers.
        let (hits_gpu, src_normal_gpu) =
            unsafe { (hits_buf.GetGPUVirtualAddress(), src_normal.GetGPUVirtualAddress()) };
        for _ in 0..2 {
            // SAFETY: nothing runs yet.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };

            // 1. Write to `src_normal` tail (offset 512) via CS UAV, mark dirty,
            // then open a non-in-place batch with an in-place MRU entry present:
            // the in-place MRU entry must NOT be bound into this non-in-place batch.
            // SAFETY: open list, live resources.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, src_normal_gpu + 512);
                list.list.Dispatch(1, 1, 1);
            }
            list.mark_uav_dirty();

            let (mut table1, scratch1) = open_batch(
                &list,
                (1, &src_normal, false),
                &[(2, src_inplace.clone(), true)],
            )
            .unwrap();
            // Slot 2 (in-place) must have been segregated out of non-in-place batch.
            assert_eq!(table1.place(2, 0, KIND_DISPATCH), None);
            let at1 = table1.place(1, 0, KIND_DISPATCH).unwrap();

            // 2. Execute indirect from batch 1, then open batch 2 (in-place + graphics_uav_dirty).
            list.mark_graphics_uav_dirty();
            let (mut table2, scratch2) = open_batch(
                &list,
                (2, &src_inplace, true),
                &[(1, src_normal.clone(), false)],
            )
            .unwrap();
            let at2 = table2.place(2, 0, KIND_DISPATCH).unwrap();
            let at2_other = table2.place(1, 0, KIND_DISPATCH).unwrap();

            // 3. Single `bound` clamp + `sync_bypass_indirect` + global UAV interop.
            let single_scratch = bound(&list, (&src_normal, 0), None, (KIND_DISPATCH, 12, 1), false, false).unwrap();
            list.emit_global_uav_barrier();
            sync_bypass_indirect(&list, &src_normal, None);

            // SAFETY: open list, live worker and argument buffers.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list.ExecuteIndirect(&signature, 1, &scratch1, at1, None, 0);
                list.list.ExecuteIndirect(&signature, 1, &scratch2, at2, None, 0);
                list.list.ExecuteIndirect(&signature, 1, &scratch2, at2_other, None, 0);
                list.list.ExecuteIndirect(&signature, 1, &single_scratch, 0, None, 0);
                list.list.ExecuteIndirect(&signature, 1, &src_normal, 0, None, 0);
            }

            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: completed work.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, 3 + 5 + 3 + 3 + 3);
        }
    }
}

/// The benchmark of the clamp and its barriers (ignored by default).
#[cfg(test)]
mod bench;

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}

#[cfg(test)]
mod clamp_barrier_tests {
    use std::ffi::OsStr;
    use super::{
        ClampBarrierPlan, PreInputUav, budget_enabled_for, enhanced_clamp_for,
        plan_clamp_barriers, scoped_clamp_barriers_for,
    };

    #[test]
    fn enhanced_clamp_and_budget_toggles() {
        assert!(enhanced_clamp_for(None));
        assert!(enhanced_clamp_for(Some(OsStr::new("1"))));
        assert!(!enhanced_clamp_for(Some(OsStr::new("0"))));

        assert_eq!(budget_enabled_for(None), cfg!(any(test, feature = "diagnostics")));
        assert!(!budget_enabled_for(Some(OsStr::new("0"))));
        assert!(budget_enabled_for(Some(OsStr::new("1"))));
    }

    #[test]
    fn scoped_clamp_mode_parses_defaults_and_overrides() {
        assert_eq!(scoped_clamp_barriers_for(None), 1);
        assert_eq!(scoped_clamp_barriers_for(Some(OsStr::new("1"))), 1);
        assert_eq!(scoped_clamp_barriers_for(Some(OsStr::new("0"))), 0);
        assert_eq!(scoped_clamp_barriers_for(Some(OsStr::new("2"))), 2);
        assert_eq!(scoped_clamp_barriers_for(Some(OsStr::new("all"))), 2);
    }

    #[test]
    fn plan_clamp_barriers_scopes_and_coalesces_safely() {
        // Barrier experiment off: zero UAV barriers.
        assert_eq!(
            plan_clamp_barriers(1, false, false, false, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::None,
                pre_budget: false,
                post_global_uav: false,
                marks_uav_clean: false,
            }
        );
        // Legacy mode (VICTORY_SCOPED_CLAMP_BARRIERS=0): global before and after single clamps.
        assert_eq!(
            plan_clamp_barriers(0, true, true, false, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::Global,
                pre_budget: false,
                post_global_uav: true,
                marks_uav_clean: true,
            }
        );
        // Default mode (1), non-in-place, dirty UAVs: resource-scoped inputs, no post global UAV.
        assert_eq!(
            plan_clamp_barriers(1, true, false, false, false, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::Scoped,
                pre_budget: false,
                post_global_uav: false,
                marks_uav_clean: false,
            }
        );
        // Second clamp on same list: adds scoped budget UAV barrier when pre is not global.
        assert_eq!(
            plan_clamp_barriers(1, true, false, false, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::Scoped,
                pre_budget: true,
                post_global_uav: false,
                marks_uav_clean: false,
            }
        );
        // Clean UAV state (e.g. immediately after title barrier_groups): skips input UAVs even for in-place.
        assert_eq!(
            plan_clamp_barriers(1, true, true, true, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::None,
                pre_budget: true,
                post_global_uav: false,
                marks_uav_clean: false,
            }
        );
        // Dirty UAV state with placed in-place buffer in default mode (1): conservative global pre, no post.
        assert_eq!(
            plan_clamp_barriers(1, true, false, true, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::Global,
                pre_budget: false,
                post_global_uav: false,
                marks_uav_clean: true,
            }
        );
        // Mode 2 ("all"): scopes even placed in-place buffers.
        assert_eq!(
            plan_clamp_barriers(2, true, false, true, true, true),
            ClampBarrierPlan {
                pre_inputs: PreInputUav::Scoped,
                pre_budget: true,
                post_global_uav: false,
                marks_uav_clean: false,
            }
        );
    }

    #[test]
    fn batch_table_defers_wc_count_write_until_flush_reset_or_drop() {
        use super::{KIND_DISPATCH, KIND_DRAW, batch};
        let mut buf1 = vec![0xFFu32; (batch::TABLE / 4) as usize];
        let mut buf2 = vec![0xFFu32; (batch::TABLE / 4) as usize];
        let mut ids1 = [0u64; batch::SOURCES];
        ids1[0] = 10;
        let mut ids2 = [0u64; batch::SOURCES];
        ids2[0] = 20;

        {
            // SAFETY: `buf1` and `buf2` hold `batch::TABLE` 4-byte-aligned bytes.
            let mut table = unsafe { batch::Table::new(buf1.as_mut_ptr().cast(), 0, ids1) };
            assert_eq!(buf1[0], 0, "Table::new initializes count word to 0");
            assert_eq!(table.place_deferred(10, 0, KIND_DRAW), Some(0));
            assert_eq!(table.place_deferred(10, 16, KIND_DRAW), Some(u64::from(batch::SLOT)));
            // `place_deferred` must NOT bounce WC memory offset 0 on every job.
            assert_eq!(buf1[0], 0, "table[0] stays unflushed during place_deferred");
            // Deduplicating an existing job reuses the same slot.
            assert_eq!(table.place_deferred(10, 0, KIND_DRAW), Some(0));

            // Reusing `table` via `reset()` flushes `buf1[0]` once and binds `buf2`.
            // SAFETY: `buf2` lives through `table`'s drop.
            unsafe { table.reset(buf2.as_mut_ptr().cast(), 64, ids2) };
            assert_eq!(buf1[0], 2, "reset() must flush prior batch count once");
            assert_eq!(buf2[0], 0, "new batch starts at 0");
            assert_eq!(table.place_deferred(20, 32, KIND_DISPATCH), Some(64));
            assert_eq!(buf2[0], 0, "table[0] stays unflushed until drop/finish");
        }
        assert_eq!(buf2[0], 1, "Drop for Table flushes final batch count once");
    }
}
