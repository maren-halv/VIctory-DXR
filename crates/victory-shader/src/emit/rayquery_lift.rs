//! Lifts Microsoft XDXR inline `RayQuery` GFX10 BVH traversal loops into
//! hardware SM 6.6 DXIL `RayQuery` intrinsics (`dx.op.allocateRayQuery`,
//! `dx.op.rayQuery_TraceRayInline`, `dx.op.rayQuery_Proceed`, `dx.op.rayQuery_State*`).
//!
//! Recognises both XDXR traversal templates emitted on Xbox Series X:
//! - Template A (`image_bvh64_intersect_ray`): per-lane 64-bit pointer traversal.
//! - Template B (`image_bvh_intersect_ray`): shared-TLAS outer loop enclosing a
//!   per-lane BLAS inner loop.
//!
//! Fail-safe invariant: any loop with an unrecognised side effect, ambiguous
//! `InstanceMask`/`RayFlags`, post-loop BVH pointer dereference, or unclassified
//! live-out SSA value is rejected so `lower.rs` falls back cleanly to software
//! intersection (`emit/intersect.rs`).
//!
//! ## Hookup in `crates/victory-shader/src/emit/lower.rs` (~10 lines)
//!
//! 1. In `Lower::run` (right after `self.find_coherent()`):
//!    ```ignore
//!    let ray_loops = super::rayquery_lift::find_ray_loops(p, self.x.st, self.x.res, &self.scalar);
//!    let ray_by_entry: HashMap<usize, usize> = ray_loops
//!        .iter()
//!        .enumerate()
//!        .map(|(idx, rl)| (rl.entry_block, idx))
//!        .collect();
//!    let ray_skipped: HashSet<usize> = ray_loops
//!        .iter()
//!        .flat_map(|rl| rl.skipped_blocks.iter().copied())
//!        .collect();
//!    let tlas_root_loads: HashSet<Site> = ray_loops
//!        .iter()
//!        .filter_map(|rl| rl.tlas_root_load)
//!        .collect();
//!    ```
//! 2. When allocating `self.block[b]` in `Lower::run`: skip `b` if `ray_skipped.contains(&b)`
//!    (leaving `self.block[b] = None` also keeps `self.counters` off the lifted header).
//! 3. In `Lower::inst` (before `s_load_*` / `global_load_*` lowering):
//!    if `tlas_root_loads.contains(&(b, i))`, write `Val::I32(self.s.m.i32(1))` to the load's
//!    written value and return `Ok(())` so the pre-loop `TLAS + 48 > 0` check passes without
//!    faulting on `SLOT_MEMORY`.
//! 4. In `Lower::run`'s `for &b in &order` loop:
//!    - If `ray_skipped.contains(&b)`: `continue;`
//!    - If `let Some(&idx) = ray_by_entry.get(&b)`:
//!      resolve `tlas = self.handle(rl.tlas_load, &Source::Loaded { load: rl.tlas_load, first: 0 }, false, (16, 0))?`,
//!      resolve `active_ray` (if `rl.active_ray_mask` has a non-`TRUE` `Bool` in `self.x.st.masks`)
//!      via `self.boolean(...)`, resolve `origin`/`t_min`/`dir`/`t_max`/dynamic mask/flags/fallbacks
//!      via `self.i32_of(...)`, call
//!      `let (tail, outs) = super::rayquery_lift::emit_ray_loop(&mut self.s, self.cur, self.block[rl.exit_block].unwrap(), rl, &inputs);`,
//!      insert `(v, Val::I32(x))` for `(v, x)` in `outs` into `self.vals`,
//!      set `self.tail[rl.exit_pred] = Some(tail);`, and `continue;`!

use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value, idoms, operand_reads, reverse_postorder};
use crate::resource::{Address, BvhNodes, CODE, MEMORY, Resources, Site, Source, inline};
use crate::structure::{Atom, Bool, LaneExit, Structured};

use super::bitcode::{Attr, Bin, Cast, Tag, Term, V, pred};
use super::shader::{
    DXR_NORMALS_DATA_BASE, DXR_NORMALS_GEOM_BASE, DXR_NORMALS_TLAS_STRIDE, DXR_NORMALS_WORDS,
    SLOT_DXR_NORMALS, Shader, rq_op,
};

/// DXR `D3D12_RAY_FLAGS` constants.
#[allow(dead_code)]
pub(crate) mod ray_flag {
    pub const NONE: u32 = 0x00;
    pub const FORCE_OPAQUE: u32 = 0x01;
    pub const FORCE_NON_OPAQUE: u32 = 0x02;
    pub const ACCEPT_FIRST_HIT_AND_END_SEARCH: u32 = 0x04;
    pub const SKIP_CLOSEST_HIT_SHADER: u32 = 0x08;
    pub const CULL_BACK_FACING_TRIANGLES: u32 = 0x10;
    pub const CULL_FRONT_FACING_TRIANGLES: u32 = 0x20;
    pub const CULL_OPAQUE: u32 = 0x40;
    pub const CULL_NON_OPAQUE: u32 = 0x80;
    pub const SKIP_TRIANGLES: u32 = 0x100;
    pub const SKIP_PROCEDURAL_PRIMITIVES: u32 = 0x200;
}

/// XDXR BVH traversal template on GFX10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RayTemplate {
    /// Template A: per-lane 64-bit pointer traversal (`image_bvh64_intersect_ray`).
    Bvh64,
    /// Template B: shared-TLAS outer loop + per-lane BLAS inner loop (`image_bvh_intersect_ray`).
    Bvh32SharedTlas,
}

/// Constant or dynamic 32-bit input for `TraceRayInline`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RayInput {
    /// Compile-time constant.
    Const(u32),
    /// Dynamic 32-bit IR value ready to pass directly.
    Val(Value),
    /// Dynamic 32-bit IR value holding an 8-bit `InstanceMask` in bits `31:24`
    /// (`emit_ray_loop` shifts right by 24).
    ShiftedMask(Value),
}

/// A BVH node metadata `global_load_*` replaced by an SM 6.6 `RayQuery` state query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RayMetaKind {
    /// `inst_addr + 48` (bits `23:0` = `InstanceID`).
    InstanceId,
    /// `inst_addr + 52` (bits `23:0` = `InstanceContributionToHitGroupIndex`).
    InstanceContribution,
    /// `inst_addr + 60` (`blas_base >> 8`, only feeds replaced `tri_addr` loads -> `0`).
    BlasBaseZero,
    /// `inst_addr + 112` (`0x70` = `InstanceIndex`).
    InstanceIndex,
    /// `inst_addr + 0..44` (`ObjectToWorld3x4`, starting byte offset for 1..4 dword load).
    ObjectToWorld(u32),
    /// `tri_addr + 48` (`(GeometryIndex << 8) | flags` -> `GeometryIndex << 8`).
    GeometryIndexShl8,
    /// `tri_addr + 52/56` (`PrimitiveIndex`).
    PrimitiveIndex,
    /// `CommittedTriangleVertexPositions()` vertex `v0`: unit tangent `T ⊥ N`
    /// derived from `SLOT_DXR_NORMALS` so `cross(v0 - v2, v1 - v2) == +N`.
    TriFetchBasisT,
    /// `CommittedTriangleVertexPositions()` vertex `v1`: unit bitangent `B = cross(N, T)`.
    TriFetchBasisB,
    /// `CommittedTriangleVertexPositions()` vertex `v2`: `(0, 0, 0)`.
    TriFetchZero3,
}

/// Replacement metadata load bound to a specific lifted [`RayLoop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RayMetaLoad {
    pub loop_idx: usize,
    pub candidate: bool,
    pub kind: RayMetaKind,
}

/// Candidate value injected at the entry of a preserved in-loop any-hit sub-CFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CandidateInput {
    RayT,
    Bary(u8),
    FrontFace,
    /// `1` (valid non-zero node sentinel for `!= 0` / `< 4` status guards).
    One,
    /// `0` (dead BVH address/base feeding only replaced [`RayMetaLoad`]s).
    Zero,
    /// Pre-loop value copied through deleted BVH scaffolding.
    CopyOf(Value),
    /// Index into [`AnyHitInfo::state_vars`] (reads the `Proceed()` loop PHI).
    StateVar(usize),
}

/// Loop-carried user state variable read/updated inside an in-loop any-hit sub-CFG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AnyHitStateVar {
    pub init: RayInput,
    pub next: Value,
    pub reg: Reg,
}

/// Preserved in-loop any-hit sub-CFG lowered inside `while (q.Proceed())`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AnyHitInfo {
    /// Deleted `opacity_block` whose fallthrough entered `entry_block`.
    pub opacity_block: usize,
    /// First IR block of the preserved any-hit sub-CFG.
    pub entry_block: usize,
    /// Unique exit IR block of the preserved any-hit sub-CFG.
    pub tail_block: usize,
    /// All IR blocks in the preserved any-hit sub-CFG (NOT skipped during lowering).
    pub body_blocks: Vec<usize>,
    /// Trailing commit-select `v_cndmask_b32` instruction indices in `tail_block` to skip.
    pub skipped_tail_insts: Vec<usize>,
    /// Lane-mask `Value` in `tail_block` selecting whether the candidate is accepted.
    pub accept_mask: Value,
    /// When `true`, accept when `accept_mask` is `false`.
    pub accept_invert: bool,
    /// When `true`, also call `rayQuery_Abort(rq)` after `CommitNonOpaqueTriangleHit`.
    pub accept_first_hit: bool,
    /// Deleted-scaffolding values read inside `body_blocks`, mapped to injected DXIL values.
    pub inputs: Vec<(Value, CandidateInput)>,
    /// Deleted-scaffolding `Atom::Lane(v)` predicates provably constant on `reach[entry_block]`.
    pub const_lane_atoms: Vec<(Value, bool)>,
    /// Loop-carried user state variables carried across `Proceed()` iterations.
    pub state_vars: Vec<AnyHitStateVar>,
    /// Candidate BVH metadata `global_load_*` sites inside `body_blocks`.
    pub cand_meta_loads: Vec<(Site, RayMetaKind)>,
}

/// DXR `RayQuery` state field represented by a live-out IR `Value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) enum RayOutput {
    /// Committed triangle or instance node pointer: `0` on miss, `1` on triangle hit.
    CommittedStatusTriNode,
    /// `CommittedRayT()` on hit, or pre-loop `t_max` on miss.
    CommittedRayT,
    /// `CommittedTriangleBarycentrics()` component (`0` = U, `1` = V).
    CommittedBary(u8),
    /// `CommittedTriangleFrontFace()` using XDXR's `0xfe` (front) / `0xff` (back) encoding.
    CommittedFrontFace,
    /// `CommittedInstanceIndex()`.
    CommittedInstanceIndex,
    /// `CommittedInstanceID()`.
    CommittedInstanceId,
    /// `CommittedInstanceContributionToHitGroupIndex()`.
    CommittedInstanceContribution,
    /// `CommittedGeometryIndex()`.
    CommittedGeometryIndex,
    /// `CommittedPrimitiveIndex()`.
    CommittedPrimitiveIndex,
    /// `CommittedObjectToWorld3x4()[row][col]`.
    CommittedObjectToWorld(u32, u8),
    /// `CommittedWorldToObject3x4()[row][col]`.
    CommittedWorldToObject(u32, u8),
    /// `FrontBackTrack` minimum front-face hit distance (`f32` bitcast to `i32`).
    FrontBackMinFrontT,
    /// Loop-carried user state variable `idx` from [`RayLoopKind::AnyHit`].
    AnyHitState(usize),
    /// Loop iteration counter or timeout flag (always `0` under hardware DXR).
    Zero,
    /// Unmodified pre-loop value carried through the loop.
    CopyOf(Value),
}

/// Live-out IR `Value` defined inside a lifted loop and its `RayQuery` source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RayOutputBinding {
    pub value: Value,
    pub output: RayOutput,
    /// Pre-loop value preserved on miss or in lanes where `active_ray` is false.
    pub fallback: Option<Value>,
}

/// Classification of a lifted traversal loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RayLoopKind {
    /// Pure opaque traversal with no in-loop any-hit handler.
    OpaqueOnly,
    /// `FORCE_NON_OPAQUE` loop tracking `minFrontFaceT` on front faces and committing back faces.
    FrontBackTrack {
        init_min_front_t: RayInput,
    },
    /// General in-loop any-hit sub-CFG lowered normally inside `while (q.Proceed())`.
    AnyHit(AnyHitInfo),
}

/// Matched direct-normal consumer for a `CommittedTriangleVertexPositions` triple
/// (`TriFetchBasisB`, `TriFetchBasisT`, `TriFetchZero3`): all uses of the three vertex
/// loads feed exclusively `normalize(cross(normalize?(v_a - v_b), normalize?(v_c - v_d)))`
/// (or unnormalized `cross(v_a - v_b, v_c - v_d)`), so we fetch `N` once from `SLOT_DXR_NORMALS`
/// and substitute `±N` directly into the three output SSA values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DirectNormalSub {
    /// Earliest of the three `TriFetch*` load sites (dominates all `out_sites`), where
    /// the single `t251` lookup + octahedral decode runs.
    pub fetch_site: Site,
    /// All three `TriFetch*` load sites (`B`, `T`, `Zero3`), suppressed to zero because
    /// every consumer of the synthetic vertices is replaced by `out_vals`.
    pub dead_fetch_sites: [Site; 3],
    /// The three output SSA `Value`s `[nx, ny, nz]` replaced directly with `±N`.
    pub out_vals: [Value; 3],
    /// Defining instruction `Site`s of `out_vals` (skipped during lowering).
    pub out_sites: [Site; 3],
    /// `true` when the shader's cross-product winding produces `-N` relative to `cross(T, B) = +N`.
    pub negate: bool,
}

/// A verified XDXR BVH traversal loop ready to be lowered as SM 6.6 `RayQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RayLoop {
    /// Shared `RayQuery` handle group index (loops that feed the same post-loop
    /// `committed_meta_loads` block share one `AllocateRayQuery` handle).
    pub rq_group: usize,
    /// Loop header IR block where `emit_ray_loop` runs.
    pub entry_block: usize,
    /// Canonical post-loop IR block reached after traversal finishes.
    pub exit_block: usize,
    /// Predecessor in `p.blocks[exit_block].preds` whose `self.tail[exit_pred]`
    /// receives the final DXIL block emitted by `emit_ray_loop`.
    pub exit_pred: usize,
    /// Blocks omitted during lowering: deleted BVH scaffolding plus folded pure-jump
    /// exit trampolines (excludes `entry_block` and any preserved [`AnyHitInfo::body_blocks`]).
    pub skipped_blocks: Vec<usize>,
    pub template: RayTemplate,
    pub kind: RayLoopKind,
    /// `s_load_dwordx4` site loading the TLAS descriptor from a table or resource heap.
    pub tlas_load: Site,
    /// Pre-loop `TLAS + 48` root-node-pointer load to fold to constant `1` so the
    /// TLAS-non-empty branch passes without touching `SLOT_MEMORY`.
    pub tlas_root_load: Option<Site>,
    /// Optional per-lane ray-enable mask SGPR (Template B checkerboard/sparse rays).
    pub active_ray_mask: Option<Value>,
    /// World-space ray origin `(x, y, z)` (already shifted by `dir * TMin` by XDXR).
    pub origin: [RayInput; 3],
    /// Optional `TMin` (`None` -> `0.0`, matching XDXR's pre-shifted origin).
    pub t_min: Option<RayInput>,
    /// World-space ray direction `(x, y, z)`.
    pub dir: [RayInput; 3],
    /// Ray `TMax` (`TMax - TMin`).
    pub t_max: RayInput,
    /// 8-bit `InstanceInclusionMask`.
    pub instance_mask: RayInput,
    /// `RAY_FLAG_*` bitmask.
    pub ray_flags: RayInput,
    /// Every live-out IR `Value` defined in the loop, mapped to its `RayQuery` output.
    pub outputs: Vec<RayOutputBinding>,
    /// Post-loop BVH metadata `global_load_*` sites replaced by `COMMITTED_*` queries.
    pub committed_meta_loads: Vec<(Site, RayMetaKind)>,
    /// Matched direct-normal substitutions for post-loop `TriFetch` triples.
    pub direct_normals: Vec<DirectNormalSub>,
}

/// A single committed hit scalar dword extracted from Pass 1 (`q1`) right after Pass 1
/// finishes (before Pass 2 starts) so every two-pass loop reuses a single `RayQuery` handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EarlyQ1Field {
    Bary(u8),
    FrontFace,
    InstanceIndex,
    InstanceId,
    InstanceContribution,
    GeometryIndex,
    PrimitiveIndex,
    ObjectToWorld(u32, u8),
    WorldToObject(u32, u8),
}

/// Post-loop two-pass hit predicates and early-extracted Pass 1 fields.
#[derive(Debug, Clone)]
pub(crate) struct TwoPassPost {
    /// `CommittedStatus(rq) == 1` (Pass 2 hit), dominating `exit_dxil`.
    pub q2_hit: V,
    /// `CommittedStatus(p1) == 1` (Pass 1 hit), dominating `exit_dxil`.
    pub q1_hit: V,
    /// Pass 1 metadata extracted before Pass 2 started (holding `0` on Pass 1 miss),
    /// so no second `RayQuery` handle is ever kept live across Pass 2.
    pub q1_fields: HashMap<EarlyQ1Field, V>,
}

/// Collects the deduplicated set of scalar fields needed from Pass 1 (`q1`) for both
/// `rl.outputs` and `rl.committed_meta_loads` (excluding `CommittedStatus` and
/// `CommittedRayT`, which are always extracted once for `t_clamp`).
pub(crate) fn collect_early_q1_fields(
    rl: &RayLoop,
    meta_dwords: &[(RayMetaKind, usize)],
) -> Vec<EarlyQ1Field> {
    let mut seen: HashSet<EarlyQ1Field> = HashSet::new();
    let mut out = Vec::new();
    let mut push = |f: EarlyQ1Field| {
        if seen.insert(f) {
            out.push(f);
        }
    };
    for b in &rl.outputs {
        match b.output {
            RayOutput::CommittedBary(c) => push(EarlyQ1Field::Bary(c)),
            RayOutput::CommittedFrontFace => push(EarlyQ1Field::FrontFace),
            RayOutput::CommittedInstanceIndex => push(EarlyQ1Field::InstanceIndex),
            RayOutput::CommittedInstanceId => push(EarlyQ1Field::InstanceId),
            RayOutput::CommittedInstanceContribution => push(EarlyQ1Field::InstanceContribution),
            RayOutput::CommittedGeometryIndex => push(EarlyQ1Field::GeometryIndex),
            RayOutput::CommittedPrimitiveIndex => push(EarlyQ1Field::PrimitiveIndex),
            RayOutput::CommittedObjectToWorld(r, c) => push(EarlyQ1Field::ObjectToWorld(r, c)),
            RayOutput::CommittedWorldToObject(r, c) => push(EarlyQ1Field::WorldToObject(r, c)),
            _ => {}
        }
    }
    for &(kind, n) in meta_dwords {
        match kind {
            RayMetaKind::BlasBaseZero | RayMetaKind::TriFetchZero3 => {}
            RayMetaKind::InstanceId => push(EarlyQ1Field::InstanceId),
            RayMetaKind::InstanceContribution => push(EarlyQ1Field::InstanceContribution),
            RayMetaKind::InstanceIndex => push(EarlyQ1Field::InstanceIndex),
            RayMetaKind::GeometryIndexShl8 => push(EarlyQ1Field::GeometryIndex),
            RayMetaKind::PrimitiveIndex => push(EarlyQ1Field::PrimitiveIndex),
            RayMetaKind::ObjectToWorld(base_off) => {
                for k in 0..n {
                    let off = base_off + (k as u32) * 4;
                    push(EarlyQ1Field::ObjectToWorld(off / 16, ((off % 16) / 4) as u8));
                }
            }
            RayMetaKind::TriFetchBasisT | RayMetaKind::TriFetchBasisB => {
                push(EarlyQ1Field::InstanceIndex);
                push(EarlyQ1Field::GeometryIndex);
                push(EarlyQ1Field::PrimitiveIndex);
            }
        }
    }
    out
}

/// Resolves a post-loop `RayMetaKind` from early-extracted Pass 1 `q1_fields` (0 on miss).
/// Returns `(updated_block, dwords)` because `TriFetchBasis*` uses a branched 2-level lookup.
pub(crate) fn resolve_early_q1_meta(
    s: &mut Shader,
    block: usize,
    fields: &HashMap<EarlyQ1Field, V>,
    kind: RayMetaKind,
    num_dwords: usize,
    tlas_meta_base: Option<V>,
) -> Option<(usize, Vec<V>)> {
    match kind {
        RayMetaKind::BlasBaseZero | RayMetaKind::TriFetchZero3 => {
            Some((block, vec![s.m.i32(0); num_dwords]))
        }
        RayMetaKind::InstanceId => fields.get(&EarlyQ1Field::InstanceId).map(|&v| (block, vec![v])),
        RayMetaKind::InstanceContribution => fields
            .get(&EarlyQ1Field::InstanceContribution)
            .map(|&v| (block, vec![v])),
        RayMetaKind::InstanceIndex => fields
            .get(&EarlyQ1Field::InstanceIndex)
            .map(|&v| (block, vec![v])),
        RayMetaKind::GeometryIndexShl8 => fields.get(&EarlyQ1Field::GeometryIndex).map(|&g| {
            let c8 = s.m.i32(8);
            (block, vec![s.m.bin(block, Bin::Shl, g, c8)])
        }),
        RayMetaKind::PrimitiveIndex => fields
            .get(&EarlyQ1Field::PrimitiveIndex)
            .map(|&v| (block, vec![v])),
        RayMetaKind::ObjectToWorld(base_off) => {
            let mut vals = Vec::with_capacity(num_dwords);
            for k in 0..num_dwords {
                let off = base_off + (k as u32) * 4;
                let f = EarlyQ1Field::ObjectToWorld(off / 16, ((off % 16) / 4) as u8);
                vals.push(*fields.get(&f)?);
            }
            Some((block, vals))
        }
        RayMetaKind::TriFetchBasisT | RayMetaKind::TriFetchBasisB => {
            let inst_idx = *fields.get(&EarlyQ1Field::InstanceIndex)?;
            let geom_idx = *fields.get(&EarlyQ1Field::GeometryIndex)?;
            let prim_idx = *fields.get(&EarlyQ1Field::PrimitiveIndex)?;
            let base = tlas_meta_base.unwrap_or_else(|| s.m.i32(0));
            Some(emit_tri_fetch_basis(
                s,
                block,
                base,
                inst_idx,
                geom_idx,
                prim_idx,
                matches!(kind, RayMetaKind::TriFetchBasisB),
                num_dwords,
            ))
        }
    }
}

/// Resolved DXIL `V` operands passed from `Lower` into [`emit_ray_loop`].
pub(crate) struct ResolvedRayInputs {
    /// Pre-allocated `RayQuery` handle (allocated in function entry block for dominance).
    pub rq: V,
    /// When `Some`, two-pass `tMax` pre-shrink (or 3-pass `fbt_mask_split`) extracts these
    /// Pass 1 fields immediately after Pass 1 so the single `rq` handle is reused for Pass 2.
    pub two_pass_early_fields: Option<Vec<EarlyQ1Field>>,
    /// Bound `RaytracingAccelerationStructure` SRV handle (or HostHeap twin-slot Mirror TLAS
    /// when `fbt_mirror_tlas` is `true`).
    pub tlas: V,
    /// Per-lane ray-active `i1` predicate when `active_ray_mask` is non-trivial.
    pub active_ray: Option<V>,
    /// Ray origin dwords (`i32` bit patterns of `f32`).
    pub origin: [V; 3],
    /// Optional `t_min` dword (`i32` bit pattern of `f32`); `None` uses `0.0f32`.
    pub t_min: Option<V>,
    /// Ray direction dwords (`i32` bit patterns of `f32`).
    pub dir: [V; 3],
    /// Ray `t_max` dword (`i32` bit pattern of `f32`).
    pub t_max: V,
    /// Dynamic `instance_mask` `i32` (`None` when `rl.instance_mask` is `Const`).
    pub instance_mask: Option<V>,
    /// Dynamic `ray_flags` `i32` (`None` when `rl.ray_flags` is `Const`).
    pub ray_flags: Option<V>,
    /// Constant `constRayFlags` baked into `AllocateRayQuery` for `rq`.
    pub group_const_flags: u32,
    /// Non-zero `InstanceInclusionMask` bitmask for fallback 3-pass mask-split `FrontBackTrack`
    /// (zero when `fbt_mirror_tlas` is active or `VICTORY_NATIVE_RT_FBT_SPLIT_MASK=0`).
    pub fbt_split_mask: u8,
    /// `true` when `FrontBackTrack` traces 2 pure-opaque HW passes (`CULL_BACK` + `CULL_FRONT`)
    /// against the HostHeap twin-slot cull-enabled Mirror TLAS (`ARCH-13`).
    pub fbt_mirror_tlas: bool,
    /// Initial `minFrontFaceT` `i32` (`f32` bits) for [`RayLoopKind::FrontBackTrack`].
    pub init_min_front_t: Option<V>,
    /// Initial loop-carried user state values for [`RayLoopKind::AnyHit`].
    pub anyhit_inits: Vec<V>,
    /// DXIL block of [`AnyHitInfo::entry_block`] for [`RayLoopKind::AnyHit`] (Pass 2 when un-fused).
    pub anyhit_entry_dxil: Option<usize>,
    /// Pre-loop fallback `i32` values keyed by IR `Value`.
    pub fallbacks: HashMap<Value, V>,
}

/// Un-fused Pass 1 / Pass 2 plumbing for a speculative opaque Pass 1 (`FORCE_OPAQUE`) ->
/// out-of-loop one-shot `Committed*` `AnyHit` check -> guarded `nextafter_down(t1)` Pass 2
/// `while (Proceed())` loop (`ARCH-12`, zero `spec_phi` / `need_trace_phi` inside the loop).
#[derive(Debug, Clone)]
pub(crate) struct SpeculativeAnyHit {
    /// Entry bridge block for Pass 1's out-of-loop `Committed*` `AnyHit` check.
    pub p1_bridge_blk: usize,
    /// Pass 1 `CandidateInput` values materialized from `Committed*` / `init_vals` in `p1_bridge_blk`.
    pub p1_cand_outs: Vec<(Value, V)>,
    /// Pass 1 `COMMITTED_RAY_T(rq)` (`f32`) read once in `p1_bridge_blk` (when needed).
    pub p1_t_f32: Option<V>,
    /// Pass 1 `COMMITTED_RAY_T(rq)` (`i32` bitcast) read once in `p1_bridge_blk` (when needed).
    pub p1_t_i32: Option<V>,
    /// Pass 1 hit/miss check block (branches to `p1_bridge_blk` on hit or `after_blk` on miss).
    pub p1_check_blk: usize,
    /// Present for dynamic `ray_flags`: `start_blk` bypasses Pass 1 directly into `p2_trace_blk`.
    pub dyn_bypass_blk: Option<usize>,
    /// `t_min` (`f32`) passed to `TraceRayInline`, needed to wire `p2_t1_phi` on dynamic bypass.
    pub t_min_f32: V,
    /// `p2_trace_blk` `i1` PHI tracking whether Pass 2 came from Pass 1 reject (`true`) vs dynamic bypass (`false`).
    pub p2_from_p1_phi: Option<V>,
    /// `p2_trace_blk` `f32` PHI merging `(p1_t_f32, p1_tail_blk)` and `(t_min_f32, dyn_bypass_blk)`.
    pub p2_t1_phi: Option<V>,
    /// Guard 4 (AMD RDNA 4 `2 * tMin` bug): `(p2_t_phi, pt_latch, commit_ray_t)` completed at Pass 2's tail.
    pub p2_t_track: Option<(V, V, V)>,
    /// Target when Pass 1's hit is rejected (`TraceRayInline(p2)` runs here, strictly before `proceed_blk`).
    pub p2_trace_blk: usize,
    /// Final Pass 2 exit block feeding `after_blk`.
    pub p2_exit_blk: usize,
    /// Post-loop join block reached from `p1_check_blk` (miss), `p1_tail_blk` (accept), and `p2_exit_blk`.
    pub after_blk: usize,
    /// `after_blk` `CommittedRayT` PHI `(phi, inputs.t_max, p2_ray_t)` awaiting `(p1_t_i32, p1_tail_blk)`.
    pub final_t_phi: Option<(V, V, V)>,
    /// `after_blk` PHIs merging `(init, p1_check_blk)`, `(p1_next, p1_tail_blk)`, and
    /// `(p2_state, p2_exit_blk)` for each `AnyHitStateVar`.
    pub final_state_phis: Vec<V>,
}

/// Pending `Proceed()` loop blocks/PHIs completed when `Lower` reaches [`AnyHitInfo::tail_block`].
pub(crate) struct PendingAnyHit {
    pub start_blk: usize,
    /// `None` when `SKIP_PROCEDURAL_PRIMITIVES` is constant and `cand_check_blk` was elided.
    pub cand_check_blk: Option<usize>,
    pub anyhit_bridge_blk: usize,
    pub do_commit_blk: usize,
    pub latch_blk: usize,
    pub state_phis: Vec<V>,
    pub latch_phis: Vec<V>,
    pub init_vals: Vec<V>,
    /// Populated when this `FORCE_NON_OPAQUE` `AnyHit` loop uses the speculative opaque fast-path.
    pub spec: Option<SpeculativeAnyHit>,
}

/// Returns `true` when `rl` is eligible for two-pass `tMax` pre-shrink:
/// a closest-hit `AnyHit` loop (optionally with `state_vars` when
/// `VICTORY_NATIVE_RT_TWOPASS_STATE` is enabled, since opaque triangles bypass `AnyHit`
/// and never touch `state_vars`) and constant `ray_flags` that do not force or cull
/// opaque/non-opaque geometry.
pub(crate) fn is_two_pass_eligible(rl: &RayLoop) -> bool {
    let RayLoopKind::AnyHit(ah) = &rl.kind else {
        return false;
    };
    if ah.accept_first_hit {
        return false;
    }
    if !ah.state_vars.is_empty() && !super::native_rt_twopass_state() {
        return false;
    }
    matches!(
        rl.ray_flags,
        RayInput::Const(f)
            if (f
                & (ray_flag::FORCE_OPAQUE
                    | ray_flag::FORCE_NON_OPAQUE
                    | ray_flag::CULL_OPAQUE
                    | ray_flag::CULL_NON_OPAQUE
                    | ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH))
                == 0
    )
}

/// Returns `true` when a `FrontBackTrack` loop's static flags are structurally
/// compatible with two pure-opaque zero-exit traces (Pass A: `CULL_BACK`, Pass B: `CULL_FRONT`).
pub(crate) fn is_front_back_two_pass_shape(rl: &RayLoop) -> bool {
    matches!(rl.kind, RayLoopKind::FrontBackTrack { .. })
        && matches!(
            rl.ray_flags,
            RayInput::Const(f)
                if (f & ray_flag::FORCE_NON_OPAQUE) != 0
                    && (f
                        & (ray_flag::FORCE_OPAQUE
                            | ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
                            | ray_flag::CULL_BACK_FACING_TRIANGLES
                            | ray_flag::CULL_FRONT_FACING_TRIANGLES
                            | ray_flag::CULL_OPAQUE
                            | ray_flag::CULL_NON_OPAQUE
                            | ray_flag::SKIP_TRIANGLES))
                        == 0
        )
}

#[cfg(test)]
thread_local! {
    /// Thread-local override for `super::shader::raw_views() != 0` in unit tests so tests
    /// never mutate process-global `RAW_VIEWS` concurrently with other shader tests.
    pub(super) static TEST_RAW_VIEWS_NONZERO: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn raw_views_nonzero() -> bool {
    #[cfg(test)]
    if let Some(v) = TEST_RAW_VIEWS_NONZERO.get() {
        return v;
    }
    super::shader::raw_views() != 0
}

/// Returns `true` when a `FrontBackTrack` loop should be lowered as 2-pass pure HW
/// `FORCE_OPAQUE` (`CULL_BACK` + `CULL_FRONT`) against the `HostHeap`
/// twin-slot cull-enabled Mirror TLAS (`ARCH-13`: active when `raw_views() != 0` and
/// `VICTORY_NATIVE_RT_MIRROR_TLAS` is not `"0"`).
pub(crate) fn is_front_back_mirror_tlas_eligible(rl: &RayLoop) -> bool {
    raw_views_nonzero()
        && super::lower::native_rt_mirror_tlas_on()
        && is_front_back_two_pass_shape(rl)
}

/// Returns `true` when a `FrontBackTrack` loop should be lowered as fallback 3-pass
/// mask-split (`VICTORY_NATIVE_RT_FBT_SPLIT_MASK != 0` and Mirror TLAS inactive):
/// Pass A (`CULL_BACK`) + Pass B (`CULL_FRONT`) pure-opaque on `mask & !cull_bit`,
/// followed by `tMax`-clamped `FORCE_NON_OPAQUE` Pass C on `mask & cull_bit`.
pub(crate) fn is_front_back_mask_split_eligible(rl: &RayLoop) -> bool {
    !is_front_back_mirror_tlas_eligible(rl)
        && super::native_rt_fbt_split_mask() != 0
        && is_front_back_two_pass_shape(rl)
}

/// Returns `true` when a `FrontBackTrack` loop should be lowered as legacy opt-in two
/// pure-opaque zero-exit traces (`VICTORY_NATIVE_RT_FBT_TWOPASS=1`, superseded by
/// Mirror TLAS and `VICTORY_NATIVE_RT_FBT_SPLIT_MASK != 0`).
pub(crate) fn is_front_back_two_pass_eligible(rl: &RayLoop) -> bool {
    !is_front_back_mirror_tlas_eligible(rl)
        && super::native_rt_fbt_split_mask() == 0
        && super::native_rt_fbt_two_pass()
        && is_front_back_two_pass_shape(rl)
}

/// Returns `true` when `rl` is a `FORCE_NON_OPAQUE` (or dynamic-flag) `AnyHit` loop
/// structurally compatible with speculative opaque Pass 1 (`FORCE_OPAQUE`) -> one-shot
/// `AnyHit` check -> exact `FORCE_NON_OPAQUE` fallback (purity is already enforced by
/// `has_no_forbidden_side_effects`). When `state_vars` is non-empty, `ACCEPT_FIRST_HIT`
/// is excluded so Pass 1 always returns the globally closest triangle `T_0`.
pub(crate) fn is_fno_speculative_shape(rl: &RayLoop) -> bool {
    let RayLoopKind::AnyHit(ah) = &rl.kind else {
        return false;
    };
    let has_state = !ah.state_vars.is_empty();
    if has_state && ah.accept_first_hit {
        return false;
    }
    let bad_mask = ray_flag::FORCE_OPAQUE
        | ray_flag::CULL_OPAQUE
        | ray_flag::CULL_NON_OPAQUE
        | ray_flag::SKIP_TRIANGLES
        | if has_state {
            ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
        } else {
            0
        };
    match rl.ray_flags {
        RayInput::Const(f) => (f & ray_flag::FORCE_NON_OPAQUE) != 0 && (f & bad_mask) == 0,
        RayInput::Val(_) | RayInput::ShiftedMask(_) => true,
    }
}

/// Returns `true` when `rl` should use the speculative opaque fast-path
/// (`VICTORY_NATIVE_RT_FNO_MODE=2`, default).
pub(crate) fn is_fno_speculative_eligible(rl: &RayLoop) -> bool {
    super::native_rt_fno_mode() >= 2 && is_fno_speculative_shape(rl)
}

/// Constant `constRayFlags` mask for `AllocateRayQuery`.
/// For Mirror-TLAS `FrontBackTrack`, mask-split `FrontBackTrack`, and speculative
/// `FORCE_NON_OPAQUE` `AnyHit`, strips `FORCE_NON_OPAQUE` (keeping `512`, NOT `513`)
/// so all disjoint loops coalesce onto a single `AllocateRayQuery(512)` handle (`ARCH-13`/`ARCH-14`).
pub(crate) fn effective_const_ray_flags(rl: &RayLoop) -> Option<u32> {
    match rl.ray_flags {
        RayInput::Const(f) => Some(
            if is_front_back_mirror_tlas_eligible(rl)
                || is_front_back_mask_split_eligible(rl)
                || is_fno_speculative_eligible(rl)
            {
                f & !ray_flag::FORCE_NON_OPAQUE
            } else if is_front_back_two_pass_eligible(rl) {
                (f & !ray_flag::FORCE_NON_OPAQUE) | ray_flag::FORCE_OPAQUE
            } else {
                f
            },
        ),
        _ => None,
    }
}

/// Output of [`emit_ray_loop`].
pub(crate) struct RayLoopEmission {
    pub tail: usize,
    pub outs: Vec<(Value, V)>,
    pub pending_anyhit: Option<PendingAnyHit>,
    pub two_pass_post: Option<TwoPassPost>,
}

// ---------------------------------------------------------------------------
// Analysis pass
// ---------------------------------------------------------------------------

/// Definition site of an SSA `Value`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Def {
    Input(Reg),
    Phi(usize, usize),
    Inst(usize, usize),
}

struct Analyzer<'a> {
    p: &'a Program,
    st: &'a Structured,
    res: &'a Resources,
    scalar: &'a HashMap<Site, Address>,
    order: Vec<usize>,
    rpo: Vec<usize>,
    idom: Vec<usize>,
    defs: Vec<Option<Def>>,
    /// Direct instruction/phi users of each `Value`.
    uses: Vec<Vec<(usize, Option<usize>)>>,
    /// `Resources::bvh` indexed by `Site`.
    bvh_nodes: HashMap<Site, BvhNodes>,
    /// `Resources::descriptors` indexed by `Site`.
    descriptors: HashMap<Site, &'a Source>,
}

/// Finds all verified XDXR BVH traversal loops in `p` that can be safely lifted
/// to SM 6.6 `RayQuery`.
pub(crate) fn find_ray_loops(
    p: &Program,
    structured: &Structured,
    res: &Resources,
    scalar: &HashMap<Site, Address>,
) -> Vec<RayLoop> {
    if res.bvh.is_empty() || p.blocks.is_empty() {
        return Vec::new();
    }
    let a = Analyzer::new(p, structured, res, scalar);
    a.run()
}

impl<'a> Analyzer<'a> {
    fn new(
        p: &'a Program,
        st: &'a Structured,
        res: &'a Resources,
        scalar: &'a HashMap<Site, Address>,
    ) -> Self {
        let n = p.blocks.len();
        let order = reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; n];
        for (i, &b) in order.iter().enumerate() {
            rpo[b] = i;
        }
        let idom = idoms(&p.blocks, &order, &rpo);

        let mut max_val = 0usize;
        for &(_, v) in &p.inputs {
            max_val = max_val.max(v.0 as usize + 1);
        }
        for blk in &p.blocks {
            for phi in &blk.phis {
                max_val = max_val.max(phi.value.0 as usize + 1);
            }
            for inst in &blk.insts {
                for &(_, v) in &inst.writes {
                    max_val = max_val.max(v.0 as usize + 1);
                }
            }
        }
        let mut defs = vec![None; max_val];
        let mut uses: Vec<Vec<(usize, Option<usize>)>> = vec![Vec::new(); max_val];
        for &(r, v) in &p.inputs {
            defs[v.0 as usize] = Some(Def::Input(r));
        }
        for (b, blk) in p.blocks.iter().enumerate() {
            for (pi, phi) in blk.phis.iter().enumerate() {
                defs[phi.value.0 as usize] = Some(Def::Phi(b, pi));
                for &arg in &phi.args {
                    if let Some(u) = uses.get_mut(arg.0 as usize) {
                        u.push((b, None));
                    }
                }
            }
            for (i, inst) in blk.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some(Def::Inst(b, i));
                }
                for &(_, v) in &inst.reads {
                    if let Some(u) = uses.get_mut(v.0 as usize) {
                        u.push((b, Some(i)));
                    }
                }
            }
        }

        let bvh_nodes = res.bvh.iter().copied().collect();
        let descriptors = res
            .descriptors
            .iter()
            .map(|d| (d.site, &d.source))
            .collect();

        Analyzer {
            p,
            st,
            res,
            scalar,
            order,
            rpo,
            idom,
            defs,
            uses,
            bvh_nodes,
            descriptors,
        }
    }

    /// Fast, budget-safe conjunction satisfiability check: catches syntactic `(b, true) && (b, false)`
    /// contradictions and pairwise disjointness before calling full `st.bools.meet` (which returns
    /// `true` if its 64K split budget expires across many large BDDs).
    fn bools_meet(&self, all: &[(Bool, bool)]) -> bool {
        for (i, &(a, ha)) in all.iter().enumerate() {
            if a == Bool::FALSE && ha || a == Bool::TRUE && !ha {
                return false;
            }
            for &(b, hb) in &all[i + 1..] {
                if a == b && ha != hb {
                    return false;
                }
            }
        }
        if all.len() > 2 {
            for (i, &p1) in all.iter().enumerate() {
                for &p2 in &all[i + 1..] {
                    if !self.st.bools.meet(&[p1, p2]) {
                        return false;
                    }
                }
            }
        }
        self.st.bools.meet(all)
    }

    /// Returns `base` extended with the per-lane conditions holding on CFG edge `pred -> succ`.
    fn edge_facts(&self, pred: usize, succ: usize, base: &[(Bool, bool)]) -> Vec<(Bool, bool)> {
        let mut nf = base.to_vec();
        let pr = self.st.reach[pred];
        if pr != Bool::TRUE && !nf.contains(&(pr, true)) {
            nf.push((pr, true));
        }
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = self.st.exits[pred]
        {
            if taken == succ && not_taken != succ {
                if when != Bool::TRUE && !nf.contains(&(when, true)) {
                    nf.push((when, true));
                }
            } else if not_taken == succ && taken != succ {
                if when != Bool::FALSE && !nf.contains(&(when, false)) {
                    nf.push((when, false));
                }
            }
        }
        nf
    }

    fn dominates(&self, a: usize, mut b: usize) -> bool {
        if self.idom[a] == usize::MAX || self.idom[b] == usize::MAX {
            return false;
        }
        let entry = self.order[0];
        while b != a {
            if b == entry {
                return false;
            }
            let d = self.idom[b];
            if d == usize::MAX || d == b {
                return false;
            }
            b = d;
        }
        true
    }

    fn def_block(&self, v: Value) -> Option<usize> {
        match self.defs.get(v.0 as usize).copied().flatten()? {
            Def::Input(_) => Some(self.order[0]),
            Def::Phi(b, _) | Def::Inst(b, _) => Some(b),
        }
    }

    fn def_inst(&self, v: Value) -> Option<(Site, &'a Inst)> {
        match self.defs.get(v.0 as usize).copied().flatten()? {
            Def::Inst(b, i) => Some(((b, i), &self.p.blocks[b].insts[i])),
            _ => None,
        }
    }

    fn run(&self) -> Vec<RayLoop> {
        // 1. Collect natural loops keyed by header from dominator back-edges.
        let mut by_header: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &u in &self.order {
            for h in self.p.blocks[u].exit.successors() {
                if self.dominates(h, u) {
                    let entry = by_header.entry(h).or_default();
                    entry.insert(h);
                    if entry.insert(u) {
                        let mut work = vec![u];
                        while let Some(x) = work.pop() {
                            if x == h {
                                continue;
                            }
                            for &pred in &self.p.blocks[x].preds {
                                if self.rpo[pred] != usize::MAX && entry.insert(pred) {
                                    work.push(pred);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Find BVH sites inside each natural loop.
        let bvh_sites: Vec<Site> = self.res.bvh.iter().map(|&(s, _)| s).collect();
        let mut candidates: Vec<(usize, HashSet<usize>, Vec<Site>)> = Vec::new();
        for (h, nodes) in by_header {
            let bvhs: Vec<Site> = bvh_sites
                .iter()
                .copied()
                .filter(|&(b, _)| nodes.contains(&b))
                .collect();
            if bvhs.len() == 1 || bvhs.len() == 2 {
                candidates.push((h, nodes, bvhs));
            }
        }
        candidates.sort_by_key(|(h, _, _)| self.rpo[*h]);

        // Keep only outermost BVH loops (collapsing Template B's inner BLAS loop into its outer TLAS loop).
        let outer: Vec<(usize, HashSet<usize>, Vec<Site>)> = candidates
            .iter()
            .filter(|(h, nodes, _)| {
                !candidates
                    .iter()
                    .any(|(h2, n2, _)| h2 != h && nodes.is_subset(n2))
            })
            .cloned()
            .collect();

        let mut lifted: Vec<RayLoop> = Vec::new();
        let mut claimed: HashSet<usize> = HashSet::new();
        // Post-loop committed metadata load sites -> `(kind, rq_group)`.
        // Loops that feed the same post-loop VBuffer extraction block (e.g. Near + Far branches
        // of the same HLSL `RayQuery`) share one `rq_group` and one `AllocateRayQuery` handle.
        let mut claimed_committed_meta: HashMap<Site, (RayMetaKind, usize)> = HashMap::new();
        let mut claimed_cand_meta: HashSet<Site> = HashSet::new();
        let mut next_rq_group: usize = 0;
        // Fixed-point pass in reverse RPO order: lifting a later loop removes its internal
        // scratch VGPR uses from earlier loops' live-out sets (e.g. Near + Far shadow traces).
        let mut remaining: Vec<(usize, HashSet<usize>, Vec<Site>)> = outer.into_iter().rev().collect();
        loop {
            let mut progressed = false;
            let mut next_remaining = Vec::new();
            for (header, nodes, bvhs) in remaining {
                if nodes.iter().any(|b| claimed.contains(b)) {
                    continue;
                }
                if let Some(mut rl) = self.analyze_loop(
                    header,
                    &nodes,
                    &bvhs,
                    &lifted,
                    &claimed,
                    &claimed_committed_meta,
                    &claimed_cand_meta,
                ) {
                    if rl.skipped_blocks.iter().any(|b| claimed.contains(b))
                        || claimed.contains(&rl.entry_block)
                    {
                        continue;
                    }
                    // Reject if any candidate load collides, or if any committed load collides
                    // with a DIFFERENT `RayMetaKind` or with a candidate load.
                    let cand_clash = matches!(&rl.kind, RayLoopKind::AnyHit(ah) if ah
                        .cand_meta_loads
                        .iter()
                        .any(|&(s, _)| claimed_cand_meta.contains(&s) || claimed_committed_meta.contains_key(&s)));
                    if cand_clash {
                        continue;
                    }
                    let mut shared_groups: Vec<usize> = Vec::new();
                    let mut kind_clash = false;
                    for &(s, kind) in &rl.committed_meta_loads {
                        if claimed_cand_meta.contains(&s) {
                            kind_clash = true;
                            break;
                        }
                        if let Some(&(prev_kind, prev_grp)) = claimed_committed_meta.get(&s) {
                            if prev_kind != kind {
                                kind_clash = true;
                                break;
                            }
                            if !shared_groups.contains(&prev_grp) {
                                shared_groups.push(prev_grp);
                            }
                        }
                    }
                    if kind_clash {
                        continue;
                    }
                    let grp = if let Some(&first_grp) = shared_groups.first() {
                        // Merge all colliding groups into `first_grp`.
                        for &other_grp in &shared_groups[1..] {
                            for prev_rl in &mut lifted {
                                if prev_rl.rq_group == other_grp {
                                    prev_rl.rq_group = first_grp;
                                }
                            }
                            for (_, g) in claimed_committed_meta.values_mut() {
                                if *g == other_grp {
                                    *g = first_grp;
                                }
                            }
                        }
                        first_grp
                    } else {
                        let g = next_rq_group;
                        next_rq_group += 1;
                        g
                    };
                    rl.rq_group = grp;
                    claimed.insert(rl.entry_block);
                    claimed.extend(rl.skipped_blocks.iter().copied());
                    if let RayLoopKind::AnyHit(ah) = &rl.kind {
                        claimed.extend(ah.body_blocks.iter().copied());
                        claimed_cand_meta.extend(ah.cand_meta_loads.iter().map(|&(s, _)| s));
                    }
                    for &(s, kind) in &rl.committed_meta_loads {
                        claimed_committed_meta.insert(s, (kind, grp));
                    }
                    lifted.push(rl);
                    progressed = true;
                } else {
                    next_remaining.push((header, nodes, bvhs));
                }
            }
            remaining = next_remaining;
            if !progressed || remaining.is_empty() {
                break;
            }
        }
        lifted.sort_by_key(|rl| self.rpo[rl.entry_block]);
        // Coalesce non-interfering `rq_group`s with identical constant ray flags onto a single
        // `AllocateRayQuery` handle so multi-loop shaders (`020`/`018` with 6 disjoint loops)
        // don't spill 6 concurrent traversal structs into VGPRs/scratch.
        self.coalesce_ray_query_groups(&mut lifted);
        // Renumber `rq_group` densely `0..num_groups`.
        let mut group_remap: HashMap<usize, usize> = HashMap::new();
        for rl in &mut lifted {
            let next_id = group_remap.len();
            rl.rq_group = *group_remap.entry(rl.rq_group).or_insert(next_id);
        }
        lifted
    }

    fn coalesce_ray_query_groups(&self, lifted: &mut [RayLoop]) {
        if lifted.len() <= 1 {
            return;
        }
        // Group loop indices by current `rq_group`, preserving RPO order.
        let mut group_ids: Vec<usize> = Vec::new();
        let mut group_loops: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, rl) in lifted.iter().enumerate() {
            if !group_loops.contains_key(&rl.rq_group) {
                group_ids.push(rl.rq_group);
            }
            group_loops.entry(rl.rq_group).or_default().push(i);
        }
        if group_ids.len() <= 1 {
            return;
        }

        // Exact per-group constant flag mask and CFG live/body block sets.
        // Because every two-pass / mask-split loop uses single-handle early extraction
        // (`TwoPassQ1::Live` removed), any disjoint groups with matching `const_flags`
        // (e.g. `512` across `fno_spec`, Mirror-TLAS `FrontBackTrack`, and `TwoPass` `AnyHit`)
        // can coalesce onto one `AllocateRayQuery` handle (`ARCH-14`).
        struct GroupInfo {
            id: usize,
            const_flags: Option<u32>,
            defs: HashSet<usize>,
            live: HashSet<usize>,
        }
        let mut infos: Vec<GroupInfo> = Vec::with_capacity(group_ids.len());
        for &gid in &group_ids {
            let mut const_flags: Option<u32> = None;
            let mut mixed_flags = false;
            let mut defs: HashSet<usize> = HashSet::new();
            let mut live: HashSet<usize> = HashSet::new();
            let mut work: Vec<usize> = Vec::new();
            for &idx in &group_loops[&gid] {
                let rl = &lifted[idx];
                match effective_const_ray_flags(rl) {
                    Some(f) => match const_flags {
                        Some(prev) if prev != f => mixed_flags = true,
                        None => const_flags = Some(f),
                        _ => {}
                    },
                    None => mixed_flags = true,
                }
                defs.insert(rl.entry_block);
                live.insert(rl.entry_block);
                live.extend(rl.skipped_blocks.iter().copied());
                if let RayLoopKind::AnyHit(ah) = &rl.kind {
                    live.extend(ah.body_blocks.iter().copied());
                }
                for &(site, _) in &rl.committed_meta_loads {
                    if live.insert(site.0) {
                        work.push(site.0);
                    }
                }
            }
            // Forward-reachable blocks from `defs`: keeps backward liveness restricted to
            // paths that can execute after this group's trace, preventing spurious
            // interference via pre-loop bypass edges.
            let mut fwd_reach: HashSet<usize> = defs.clone();
            let mut fwd_work: Vec<usize> = defs.iter().copied().collect();
            while let Some(b) = fwd_work.pop() {
                for succ in self.p.blocks[b].exit.successors() {
                    if fwd_reach.insert(succ) {
                        fwd_work.push(succ);
                    }
                }
            }
            // Backward dataflow from post-loop committed metadata loads back to `defs`.
            while let Some(b) = work.pop() {
                for &pred in &self.p.blocks[b].preds {
                    if defs.contains(&pred) || !fwd_reach.contains(&pred) {
                        continue;
                    }
                    if live.insert(pred) {
                        work.push(pred);
                    }
                }
            }
            infos.push(GroupInfo {
                id: gid,
                const_flags: if mixed_flags { None } else { const_flags },
                defs,
                live,
            });
        }

        // Greedy interference-free merging of disjoint groups sharing the exact same `const_flags`.
        let mut merged: Vec<GroupInfo> = Vec::new();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for g in infos {
            let mut target_idx: Option<usize> = None;
            if let Some(cf) = g.const_flags {
                for (mi, m) in merged.iter().enumerate() {
                    if m.const_flags != Some(cf) {
                        continue;
                    }
                    let clash = g.defs.iter().any(|b| m.live.contains(b))
                        || m.defs.iter().any(|b| g.live.contains(b));
                    if !clash {
                        target_idx = Some(mi);
                        break;
                    }
                }
            }
            if let Some(mi) = target_idx {
                let target_id = merged[mi].id;
                merged[mi].defs.extend(g.defs);
                merged[mi].live.extend(g.live);
                remap.insert(g.id, target_id);
            } else {
                remap.insert(g.id, g.id);
                merged.push(g);
            }
        }
        for rl in lifted.iter_mut() {
            if let Some(&new_g) = remap.get(&rl.rq_group) {
                rl.rq_group = new_g;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_loop(
        &self,
        entry_block: usize,
        nodes: &HashSet<usize>,
        bvhs: &[Site],
        lifted: &[RayLoop],
        lifted_blocks: &HashSet<usize>,
        claimed_committed_meta: &HashMap<Site, (RayMetaKind, usize)>,
        claimed_cand_meta: &HashSet<Site>,
    ) -> Option<RayLoop> {
        // 1. Determine template and primary (world-space) vs triangle-testing BVH site.
        let (template, primary_bvh, tri_bvh) = self.classify_template(nodes, bvhs)?;

        // 2. Fold pure-jump exit trampolines to verify a single canonical exit block.
        let (exit_block, exit_pred, folded_tramps) = self.canonical_exit(nodes)?;
        let mut loop_all = nodes.clone();
        loop_all.extend(folded_tramps.iter().copied());

        // 3. Reject forbidden side effects anywhere in `nodes`.
        if !self.has_no_forbidden_side_effects(nodes) {
            return None;
        }

        // 4. Trace TLAS descriptor load (`s_load_dwordx4` from Table/ResourceHeap) and pre-loop TLAS+48 root load.
        let tlas_load = self.find_tlas_load(template, primary_bvh, nodes, entry_block)?;
        let tlas_root_load = self.find_tlas_root_load(tlas_load, nodes, entry_block);

        // 5. Extract world-space ray inputs (origin, dir, t_max) and optional per-lane active mask.
        let (origin, dir, t_max, extent_val) =
            self.extract_ray_vectors(template, primary_bvh, nodes, entry_block)?;
        let active_ray_mask = self.extract_active_ray_mask(template, nodes, entry_block, tlas_root_load);

        // 6. Locate triangle-hit commit site (pure opaque, FrontBackTrack, or AnyHit sub-CFG).
        let tri_hit = self.find_triangle_hit(template, tri_bvh, extent_val, t_max, nodes, entry_block)?;

        // Verify no buffer/image fetches occur outside preserved any-hit blocks.
        let anyhit_set: HashSet<usize> = match &tri_hit.mode {
            HitMode::AnyHit(ah) => ah.body_blocks.iter().copied().collect(),
            _ => HashSet::new(),
        };
        if !self.has_no_scaffolding_fetches(nodes, &anyhit_set) {
            return None;
        }

        // 7. Extract `RayFlags` + `InstanceMask`.
        let ray_flags = self.extract_ray_flags(nodes, &tri_hit)?;
        let instance_mask = self.extract_instance_mask(template, nodes, entry_block)?;

        // Closest-hit `AnyHit` loops with constant `FORCE_NON_OPAQUE` or dynamic `ray_flags`
        // are controlled by `VICTORY_NATIVE_RT_FNO_MODE` (default 2: speculative opaque fast-path
        // + exact fallback; 1: direct `Proceed()`; 0: disabled unless `VICTORY_NATIVE_RT_ANYHIT=1`).
        let is_const_fno = matches!(
            ray_flags,
            RayInput::Const(f) if (f & ray_flag::FORCE_NON_OPAQUE) != 0
        );
        let is_dynamic_flags = !matches!(ray_flags, RayInput::Const(_));
        if matches!(&tri_hit.mode, HitMode::AnyHit(ah) if !ah.accept_first_hit)
            && (is_const_fno || is_dynamic_flags)
            && super::native_rt_fno_mode() == 0
            && !super::native_rt_anyhit()
        {
            return None;
        }

        // For FrontBackTrack, `RAY_FLAG_FORCE_NON_OPAQUE` must be set.
        if matches!(tri_hit.mode, HitMode::FrontBackTrack { .. })
            && !matches!(ray_flags, RayInput::Const(f) if (f & ray_flag::FORCE_NON_OPAQUE) != 0)
        {
            return None;
        }

        // 8. Compute true live-outs outside `loop_all`, classify every single one, and collect
        //    any verified post-loop committed BVH metadata loads.
        let (outputs, committed_meta_loads) = self.classify_live_outs(
            &loop_all,
            nodes,
            entry_block,
            exit_block,
            exit_pred,
            t_max,
            &tri_hit,
            lifted,
            lifted_blocks,
            claimed_committed_meta,
            claimed_cand_meta,
        )?;

        let mut skipped_blocks: Vec<usize> = loop_all
            .iter()
            .copied()
            .filter(|&b| b != entry_block && !anyhit_set.contains(&b))
            .collect();
        skipped_blocks.sort_by_key(|&b| self.rpo[b]);

        let kind = match tri_hit.mode {
            HitMode::Opaque => RayLoopKind::OpaqueOnly,
            HitMode::FrontBackTrack { init_min_front_t, .. } => {
                RayLoopKind::FrontBackTrack { init_min_front_t }
            }
            HitMode::AnyHit(ah) => RayLoopKind::AnyHit(ah),
        };

        let direct_normals = self.find_direct_normals(&committed_meta_loads);

        Some(RayLoop {
            rq_group: 0,
            entry_block,
            exit_block,
            exit_pred,
            skipped_blocks,
            template,
            kind,
            tlas_load,
            tlas_root_load,
            active_ray_mask,
            origin,
            t_min: tri_hit.t_min,
            dir,
            t_max,
            instance_mask,
            ray_flags,
            outputs,
            committed_meta_loads,
            direct_normals,
        })
    }

    fn classify_template(
        &self,
        nodes: &HashSet<usize>,
        bvhs: &[Site],
    ) -> Option<(RayTemplate, Site, Site)> {
        match *bvhs {
            [site] => {
                let name = self.p.blocks[site.0].insts[site.1].name();
                if name == "image_bvh64_intersect_ray" {
                    Some((RayTemplate::Bvh64, site, site))
                } else {
                    None
                }
            }
            [a, b] => {
                let na = self.p.blocks[a.0].insts[a.1].name();
                let nb = self.p.blocks[b.0].insts[b.1].name();
                if na != "image_bvh_intersect_ray" || nb != "image_bvh_intersect_ray" {
                    return None;
                }
                // In Template B, the outer TLAS BVH sits outside the inner BLAS loop
                // (higher in RPO, or dominates the inner loop). Verify by checking which
                // one has a descriptor derived from a table/root TLAS load vs BLAS instance.
                let (tlas_site, blas_site) = if self.rpo[a.0] <= self.rpo[b.0] {
                    (a, b)
                } else {
                    (b, a)
                };
                if !nodes.contains(&tlas_site.0) || !nodes.contains(&blas_site.0) {
                    return None;
                }
                Some((RayTemplate::Bvh32SharedTlas, tlas_site, blas_site))
            }
            _ => None,
        }
    }

    fn canonical_exit(&self, nodes: &HashSet<usize>) -> Option<(usize, usize, Vec<usize>)> {
        let mut raw_exits: Vec<usize> = Vec::new();
        for &u in nodes {
            for s in self.p.blocks[u].exit.successors() {
                if !nodes.contains(&s) && !raw_exits.contains(&s) {
                    raw_exits.push(s);
                }
            }
        }
        if raw_exits.is_empty() {
            return None;
        }
        raw_exits.sort_by_key(|&b| self.rpo[b]);

        let mut folded: Vec<usize> = Vec::new();
        let mut folded_set: HashSet<usize> = HashSet::new();
        let mut targets: HashSet<usize> = HashSet::new();

        for &ex in &raw_exits {
            let mut cur = ex;
            let mut chain: Vec<usize> = Vec::new();
            let mut hops = 0u32;
            while hops < 8 && self.is_pure_jump_trampoline(cur, nodes, &folded_set) {
                let Exit::Jump(next) = self.p.blocks[cur].exit else {
                    break;
                };
                if nodes.contains(&next) || chain.contains(&next) {
                    return None;
                }
                chain.push(cur);
                cur = next;
                hops += 1;
            }
            for t in chain {
                if folded_set.insert(t) {
                    folded.push(t);
                }
            }
            targets.insert(cur);
        }

        if targets.len() != 1 {
            return None;
        }
        let exit_block = *targets.iter().next()?;
        let mut loop_all = nodes.clone();
        loop_all.extend(folded.iter().copied());

        // Pick one canonical predecessor of `exit_block` that comes from `loop_all`.
        let exit_pred = self.p.blocks[exit_block]
            .preds
            .iter()
            .copied()
            .find(|p| loop_all.contains(p))?;

        Some((exit_block, exit_pred, folded))
    }

    fn is_pure_jump_trampoline(
        &self,
        b: usize,
        nodes: &HashSet<usize>,
        folded: &HashSet<usize>,
    ) -> bool {
        let blk = &self.p.blocks[b];
        if !matches!(blk.exit, Exit::Jump(_)) || !blk.phis.is_empty() {
            return false;
        }
        // Only fold if every predecessor comes from the loop (never steal a pre-loop edge).
        if blk.preds.is_empty()
            || !blk
                .preds
                .iter()
                .all(|p| nodes.contains(p) || folded.contains(p))
        {
            return false;
        }
        blk.insts
            .iter()
            .all(|i| matches!(i.name(), "s_branch" | "s_nop" | "s_waitcnt" | "s_waitcnt_vscnt"))
    }

    fn has_no_forbidden_side_effects(&self, nodes: &HashSet<usize>) -> bool {
        for &b in nodes {
            if matches!(self.p.blocks[b].exit, Exit::End | Exit::Indirect) {
                return false;
            }
            for inst in &self.p.blocks[b].insts {
                let n = inst.name();
                if n.starts_with("buffer_store")
                    || n.starts_with("global_store")
                    || n.starts_with("flat_store")
                    || n.starts_with("scratch_store")
                    || n.starts_with("image_store")
                    || n.starts_with("s_store")
                    || n.contains("atomic")
                    || n.starts_with("exp")
                    || n == "s_endpgm"
                    || n == "s_barrier"
                {
                    return false;
                }
            }
        }
        true
    }

    fn has_no_scaffolding_fetches(
        &self,
        nodes: &HashSet<usize>,
        anyhit_set: &HashSet<usize>,
    ) -> bool {
        for &b in nodes {
            if anyhit_set.contains(&b) {
                continue;
            }
            for inst in &self.p.blocks[b].insts {
                let n = inst.name();
                if n.starts_with("image_sample")
                    || n.starts_with("image_load")
                    || n.starts_with("image_gather")
                    || n.starts_with("buffer_load")
                    || n.starts_with("s_buffer_load")
                    || n.starts_with("tbuffer_")
                {
                    return false;
                }
            }
        }
        true
    }

    /// Traces the TLAS descriptor `s_load_dwordx4` site from the traversal loop's TLAS BVH.
    fn find_tlas_load(
        &self,
        template: RayTemplate,
        primary_bvh: Site,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<Site> {
        let inst = &self.p.blocks[primary_bvh.0].insts[primary_bvh.1];
        // Direct descriptor resource (when `resource::track` saw a table descriptor directly).
        if matches!(self.bvh_nodes.get(&primary_bvh), Some(BvhNodes::Resource))
            && let Some(Source::Loaded { load, first: 0 }) = self.descriptors.get(&primary_bvh)
            && self.is_table_scalar_load(*load)
            && self.dominates(load.0, entry_block)
        {
            return Some(*load);
        }

        let seed = match template {
            RayTemplate::Bvh32SharedTlas => {
                let Fields::Image {
                    rsrc: Operand::Scalar { code, .. },
                    ..
                } = &inst.fields
                else {
                    return None;
                };
                inst.read(Reg::S(*code))?
            }
            RayTemplate::Bvh64 => {
                // `vaddr[0]` is `v_add_co_u32(v_lshlrev_b64(5, bvh_base_shr8), node_ptr)`.
                let addrs = image_addr_values(inst)?;
                *addrs.first()?
            }
        };

        let mut visited = HashSet::new();
        let mut work = vec![(seed, 0u32)];
        let mut found: Option<Site> = None;
        while let Some((v, depth)) = work.pop() {
            if depth > 16 || !visited.insert(v) {
                continue;
            }
            let Some(def) = self.defs.get(v.0 as usize).copied().flatten() else {
                continue;
            };
            match def {
                Def::Input(_) => continue,
                Def::Phi(b, pi) => {
                    let phi = &self.p.blocks[b].phis[pi];
                    for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                        // Prefer pre-loop / non-back-edge incoming edges so we trace the TLAS, not BLAS!
                        if arg == v || self.dominates(b, pred) {
                            continue;
                        }
                        if b == entry_block && nodes.contains(&pred) {
                            continue;
                        }
                        work.push((arg, depth + 1));
                    }
                }
                Def::Inst(b, i) => {
                    let site = (b, i);
                    let di = &self.p.blocks[b].insts[i];
                    if di.name().starts_with("s_load_dword") && self.is_table_scalar_load(site) {
                        if !self.dominates(b, entry_block) {
                            continue;
                        }
                        if let Some(prev) = found {
                            if prev != site {
                                return None; // Ambiguous TLAS load.
                            }
                        } else {
                            found = Some(site);
                        }
                        continue;
                    }
                    // Follow select/cndmask data arms only (never the predicate/mask operand).
                    if di.name() == "v_cndmask_b32"
                        && let Fields::Valu(f) = &di.fields
                        && f.src.len() >= 2
                    {
                        for op in &f.src[..2] {
                            if let Some(src_v) = operand_val(di, op) {
                                work.push((src_v, depth + 1));
                            }
                        }
                        continue;
                    }
                    if matches!(di.name(), "s_cselect_b32" | "s_cselect_b64")
                        && let Fields::Sop { src, .. } = &di.fields
                        && src.len() >= 2
                    {
                        for op in &src[..2] {
                            if let Some(src_v) = operand_val(di, op) {
                                work.push((src_v, depth + 1));
                            }
                        }
                        continue;
                    }
                    // Follow address/copy/shift/mask/select ops backward (never across unrelated loads).
                    if matches!(
                        di.name(),
                        "s_mov_b32"
                            | "s_mov_b64"
                            | "v_mov_b32"
                            | "s_and_b32"
                            | "s_and_b64"
                            | "s_or_b32"
                            | "s_or_b64"
                            | "s_lshr_b32"
                            | "s_lshr_b64"
                            | "s_lshl_b32"
                            | "s_lshl_b64"
                            | "v_lshrrev_b32"
                            | "v_lshrrev_b64"
                            | "v_lshlrev_b32"
                            | "v_lshlrev_b64"
                            | "v_add_co_u32"
                            | "v_add_nc_u32"
                            | "v_or_b32"
                            | "v_and_b32"
                            | "v_readfirstlane_b32"
                            | "v_readlane_b32"
                            | "s_bfe_u32"
                            | "v_bfe_u32"
                            | "v_bfi_b32"
                    ) {
                        for &r in &operand_reads(self.p, di) {
                            if let Some(src_v) = di.read(r) {
                                work.push((src_v, depth + 1));
                            }
                        }
                    }
                }
            }
        }
        found
    }

    fn is_table_scalar_load(&self, site: Site) -> bool {
        self.scalar
            .get(&site)
            .is_some_and(|addr| addr.param != CODE && addr.param != MEMORY)
    }

    /// Finds the pre-loop `s_load_dword` or `global_load_dword` reading `TLAS + 48`
    /// whose result only feeds a `> 0` check and/or the lifted loop.
    fn find_tlas_root_load(
        &self,
        tlas_load: Site,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<Site> {
        let tlas_inst = &self.p.blocks[tlas_load.0].insts[tlas_load.1];
        let lo_val = tlas_inst.writes.first()?.1;

        // Forward BFS from `lo_val` to values derived as `(lo & 0xffffff00)` or `lo | 48` / `lo + 48`.
        let mut derived: HashMap<Value, bool> = HashMap::new(); // value -> has_offset_48
        derived.insert(lo_val, false);
        let mut work = vec![(lo_val, false, 0u32)];
        let mut candidates: Vec<Site> = Vec::new();

        while let Some((v, has48, depth)) = work.pop() {
            if depth > 10 {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if nodes.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else { continue };
                let inst = &self.p.blocks[b].insts[i];
                let n = inst.name();
                match n {
                    "s_load_dword" => {
                        let Fields::Smem { offset, soffset, .. } = &inst.fields else {
                            continue;
                        };
                        let off = *offset + const_operand(soffset).unwrap_or(0) as i32;
                        if (has48 && off == 0) || (!has48 && off == 48) {
                            candidates.push((b, i));
                        }
                    }
                    "global_load_dword" => {
                        let Fields::Flat { offset, .. } = &inst.fields else {
                            continue;
                        };
                        if (has48 && *offset == 0) || (!has48 && *offset == 48) {
                            candidates.push((b, i));
                        }
                    }
                    "s_mov_b32"
                    | "v_mov_b32"
                    | "s_and_b32"
                    | "v_and_b32"
                    | "s_lshr_b64"
                    | "s_lshl_b64"
                    | "s_lshr_b32"
                    | "s_lshl_b32"
                    | "v_readlane_b32"
                    | "v_readfirstlane_b32" => {
                        if let Some(&(_, w)) = inst.writes.first() {
                            if derived.get(&w).copied() != Some(has48) {
                                derived.insert(w, has48);
                                work.push((w, has48, depth + 1));
                            }
                        }
                    }
                    "s_or_b32" | "s_add_u32" | "s_add_i32" | "v_or_b32" | "v_add_nc_u32"
                    | "v_add_co_u32" => {
                        let adds_48 = other_const_arg(inst, v) == Some(48);
                        if adds_48 {
                            if let Some(&(_, w)) = inst.writes.first() {
                                if derived.get(&w).copied() != Some(true) {
                                    derived.insert(w, true);
                                    work.push((w, true, depth + 1));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Keep a candidate that dominates `entry_block` and whose loaded root value is only
        // used by pre-loop `> 0` compares or by the lifted loop itself.
        candidates.sort_by_key(|&(b, i)| (self.rpo[b], i));
        candidates.dedup();
        candidates.into_iter().find(|&(b, i)| {
            if !self.dominates(b, entry_block) {
                return false;
            }
            let Some(&(_, root_val)) = self.p.blocks[b].insts[i].writes.first() else {
                return false;
            };
            self.root_load_only_feeds_loop(root_val, nodes)
        })
    }

    fn root_load_only_feeds_loop(&self, root_val: Value, nodes: &HashSet<usize>) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![root_val];
        let mut fed_loop_or_cmp = false;
        while let Some(v) = work.pop() {
            if !seen.insert(v) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if nodes.contains(&b) {
                    fed_loop_or_cmp = true;
                    continue;
                }
                let Some(i) = idx_opt else {
                    return false;
                };
                let inst = &self.p.blocks[b].insts[i];
                match inst.name() {
                    "s_mov_b32" | "v_mov_b32" => {
                        if let Some(&(_, w)) = inst.writes.first() {
                            work.push(w);
                        }
                    }
                    "s_cmp_gt_i32" | "s_cmp_lt_i32" | "s_cmp_eq_i32" | "s_cmp_lg_i32"
                    | "s_cmp_gt_u32" | "s_cmp_lt_u32" | "s_cmp_eq_u32" | "s_cmp_lg_u32"
                    | "v_cmp_gt_i32" | "v_cmp_lt_i32" => {
                        fed_loop_or_cmp = true;
                    }
                    _ => return false,
                }
            }
        }
        fed_loop_or_cmp
    }

    /// Extracts world-space `(origin, dir, t_max)` and the in-loop `extent` `Value`
    /// passed to the primary BVH intersection instruction.
    fn extract_ray_vectors(
        &self,
        template: RayTemplate,
        primary_bvh: Site,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<([RayInput; 3], [RayInput; 3], RayInput, Value)> {
        let inst = &self.p.blocks[primary_bvh.0].insts[primary_bvh.1];
        let addrs = image_addr_values(inst)?;
        let (ext_idx, orig_idx, dir_idx) = match template {
            RayTemplate::Bvh32SharedTlas if addrs.len() >= 8 => (1, 2, 5),
            RayTemplate::Bvh64 if addrs.len() >= 9 => (2, 3, 6),
            _ => return None,
        };
        let extent_val = addrs[ext_idx];
        let t_max = self.pre_loop_input(extent_val, nodes, entry_block, 0)?;
        let origin = [
            self.pre_loop_input(addrs[orig_idx], nodes, entry_block, 0)?,
            self.pre_loop_input(addrs[orig_idx + 1], nodes, entry_block, 0)?,
            self.pre_loop_input(addrs[orig_idx + 2], nodes, entry_block, 0)?,
        ];
        let dir = [
            self.pre_loop_input(addrs[dir_idx], nodes, entry_block, 0)?,
            self.pre_loop_input(addrs[dir_idx + 1], nodes, entry_block, 0)?,
            self.pre_loop_input(addrs[dir_idx + 2], nodes, entry_block, 0)?,
        ];
        // Every dynamic ray input must dominate the loop header.
        for inp in [t_max, origin[0], origin[1], origin[2], dir[0], dir[1], dir[2]] {
            if let RayInput::Val(v) = inp {
                let db = self.def_block(v)?;
                if nodes.contains(&db) || !self.dominates(db, entry_block) {
                    return None;
                }
            }
        }
        Some((origin, dir, t_max, extent_val))
    }

    /// Traces a value inside `nodes` back to a unique pre-loop `Value` or compile-time constant.
    fn pre_loop_input(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
        depth: u32,
    ) -> Option<RayInput> {
        if depth > 12 {
            return None;
        }
        let def = self.defs.get(v.0 as usize).copied().flatten()?;
        match def {
            Def::Input(_) => Some(RayInput::Val(v)),
            Def::Phi(b, pi) => {
                if !nodes.contains(&b) {
                    return Some(RayInput::Val(v));
                }
                let phi = &self.p.blocks[b].phis[pi];
                let mut pre: Option<RayInput> = None;
                for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                    if arg == v || self.dominates(b, pred) {
                        continue;
                    }
                    if b == entry_block && nodes.contains(&pred) {
                        continue;
                    }
                    let p_inp = self.pre_loop_input(arg, nodes, entry_block, depth + 1)?;
                    if let Some(prev) = pre {
                        if prev != p_inp {
                            return None;
                        }
                    } else {
                        pre = Some(p_inp);
                    }
                }
                pre
            }
            Def::Inst(b, i) => {
                if !nodes.contains(&b) {
                    return Some(RayInput::Val(v));
                }
                let inst = &self.p.blocks[b].insts[i];
                if matches!(inst.name(), "v_mov_b32" | "s_mov_b32") {
                    let src_op = first_src_operand(inst)?;
                    if let Some(k) = const_operand(src_op) {
                        return Some(RayInput::Const(k));
                    }
                    let src = operand_val(inst, src_op)?;
                    return self.pre_loop_input(src, nodes, entry_block, depth + 1);
                }
                if inst.name() == "s_movk_i32"
                    && let Fields::Sop { imm: Some(imm), .. } = &inst.fields
                {
                    return Some(RayInput::Const(*imm as i16 as i32 as u32));
                }
                // Handle `v_cndmask_b32 dst, a, b, mask` when one arm is the pre-loop world ray
                // (or an inline constant world ray component like `0.0` / `1.0` in GatherAlphaVBuffer!).
                if inst.name() == "v_cndmask_b32"
                    && let Fields::Valu(f) = &inst.fields
                    && f.src.len() >= 2
                {
                    let a_val = operand_val(inst, &f.src[0]);
                    let b_val = operand_val(inst, &f.src[1]);
                    if let Some(va) = a_val
                        && self.def_block(va).is_some_and(|db| !nodes.contains(&db))
                    {
                        return Some(RayInput::Val(va));
                    }
                    if let Some(vb) = b_val
                        && self.def_block(vb).is_some_and(|db| !nodes.contains(&db))
                    {
                        return Some(RayInput::Val(vb));
                    }
                    if let Some(ka) = const_operand(&f.src[0])
                        && b_val.is_some()
                    {
                        return Some(RayInput::Const(ka));
                    }
                    if let Some(kb) = const_operand(&f.src[1])
                        && a_val.is_some()
                    {
                        return Some(RayInput::Const(kb));
                    }
                }
                None
            }
        }
    }

    /// Traces a value inside `nodes` back to its unique pre-loop incoming value at `entry_block`.
    fn pre_loop_value(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
        depth: u32,
    ) -> Option<Value> {
        if depth > 12 {
            return None;
        }
        let def = self.defs.get(v.0 as usize).copied().flatten()?;
        match def {
            Def::Input(_) => Some(v),
            Def::Phi(b, pi) => {
                if !nodes.contains(&b) {
                    return Some(v);
                }
                let phi = &self.p.blocks[b].phis[pi];
                let mut pre: Option<Value> = None;
                for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                    if arg == v || self.dominates(b, pred) {
                        continue;
                    }
                    if b == entry_block && nodes.contains(&pred) {
                        continue;
                    }
                    let p_val = self.pre_loop_value(arg, nodes, entry_block, depth + 1)?;
                    if let Some(prev) = pre {
                        if prev != p_val {
                            return None;
                        }
                    } else {
                        pre = Some(p_val);
                    }
                }
                pre
            }
            Def::Inst(b, i) => {
                if !nodes.contains(&b) {
                    return Some(v);
                }
                let inst = &self.p.blocks[b].insts[i];
                if matches!(inst.name(), "v_mov_b32" | "s_mov_b32") {
                    let src = single_reg_src(inst)?;
                    return self.pre_loop_value(src, nodes, entry_block, depth + 1);
                }
                // Handle `v_cndmask_b32 dst, a, b, mask` when one arm is the pre-loop value
                // (e.g. Template A's world-ray restore on instance exit!).
                if inst.name() == "v_cndmask_b32"
                    && let Fields::Valu(f) = &inst.fields
                    && f.src.len() >= 2
                {
                    let a = operand_val(inst, &f.src[0]);
                    let b = operand_val(inst, &f.src[1]);
                    if let Some(va) = a
                        && self.def_block(va).is_some_and(|db| !nodes.contains(&db))
                    {
                        return Some(va);
                    }
                    if let Some(vb) = b
                        && self.def_block(vb).is_some_and(|db| !nodes.contains(&db))
                    {
                        return Some(vb);
                    }
                }
                None
            }
        }
    }

    /// In Template B, checks whether the outer shared-TLAS loop tracks an explicit per-lane
    /// active-ray mask (e.g. Checkerboard's `s84 & exec_lo`) stricter than `reach[entry_block]`.
    fn extract_active_ray_mask(
        &self,
        template: RayTemplate,
        nodes: &HashSet<usize>,
        entry_block: usize,
        tlas_root_load: Option<Site>,
    ) -> Option<Value> {
        if template != RayTemplate::Bvh32SharedTlas {
            return None;
        }
        let root_val = tlas_root_load
            .and_then(|(b, i)| self.p.blocks[b].insts[i].writes.first().map(|w| w.1));

        // Look for an SGPR lane-mask `Value` defined outside `nodes` that is used inside
        // `nodes` as a mask operand (e.g. `s_and_b32 _, s_active, exec_lo` or `v_and_b32 _, s_active, lane_bit`).
        let mut best: Option<Value> = None;
        for &b in nodes {
            for inst in &self.p.blocks[b].insts {
                if !matches!(inst.name(), "s_and_b32" | "s_and_b64") {
                    continue;
                }
                let Fields::Sop { src, .. } = &inst.fields else {
                    continue;
                };
                if src.len() != 2 {
                    continue;
                }
                let is_exec = |op: &Operand| {
                    matches!(op, Operand::Scalar { code, .. } if *code == EXEC || *code == EXEC + 1)
                };
                let other = if is_exec(&src[0]) {
                    &src[1]
                } else if is_exec(&src[1]) {
                    &src[0]
                } else {
                    continue;
                };
                let Some(mask_v) = operand_val(inst, other) else {
                    continue;
                };
                let Some(pre) = self.pre_loop_value(mask_v, nodes, entry_block, 0) else {
                    continue;
                };
                let peeled = self.peel_root_select(pre, root_val);
                let Some(&bool_id) = self.st.masks.get(&peeled) else {
                    continue;
                };
                if bool_id == Bool::TRUE || bool_id == self.st.reach[entry_block] {
                    continue;
                }
                // Check that `peeled` is defined outside the loop and dominates `entry_block`.
                if let Some(db) = self.def_block(peeled)
                    && !nodes.contains(&db)
                    && self.dominates(db, entry_block)
                {
                    best = Some(peeled);
                    break;
                }
            }
            if best.is_some() {
                break;
            }
        }
        best
    }

    /// Peels `s_cselect_b32/b64(exec, 0, tlas_root > 0)` or `s_and(mask, cselect)` when
    /// `tlas_root` is forced to `1`.
    fn peel_root_select(&self, v: Value, root_val: Option<Value>) -> Value {
        let Some(rv) = root_val else {
            return v;
        };
        let Some((_, inst)) = self.def_inst(v) else {
            return v;
        };
        if self.is_root_cselect(v, rv)
            && let Fields::Sop { src, .. } = &inst.fields
            && let Some(src0) = src.first()
            && let Some(ev) = operand_val(inst, src0)
        {
            return ev;
        }
        if matches!(inst.name(), "s_and_b32" | "s_and_b64")
            && let Fields::Sop { src, .. } = &inst.fields
            && src.len() == 2
        {
            if let (Some(a), Some(b)) = (operand_val(inst, &src[0]), operand_val(inst, &src[1])) {
                if self.is_root_cselect(a, rv) {
                    return b;
                }
                if self.is_root_cselect(b, rv) {
                    return a;
                }
            }
        }
        v
    }

    fn is_root_cselect(&self, v: Value, root_val: Value) -> bool {
        let Some((_, inst)) = self.def_inst(v) else {
            return false;
        };
        if !matches!(inst.name(), "s_cselect_b32" | "s_cselect_b64") {
            return false;
        }
        let Some(scc) = inst.read(Reg::Scc) else {
            return false;
        };
        let Some((_, cmp)) = self.def_inst(scc) else {
            return false;
        };
        cmp.reads.iter().any(|&(_, x)| x == root_val)
    }

    /// Collects `root` and all SSA values in `nodes` (skipping any block in `exclude`) reachable
    /// from it via PHIs or `v_mov_b32` (needed when a mutually-exclusive branch like the box-node
    /// branch clobbers the same VGPRs).
    fn forward_phi_closure(
        &self,
        root: Value,
        nodes: &HashSet<usize>,
        exclude: Option<&HashSet<usize>>,
    ) -> HashSet<Value> {
        let mut seen = HashSet::new();
        let mut work = vec![root];
        seen.insert(root);
        while let Some(v) = work.pop() {
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !nodes.contains(&b) || exclude.is_some_and(|ex| ex.contains(&b)) {
                    continue;
                }
                match idx_opt {
                    None => {
                        for phi in &self.p.blocks[b].phis {
                            if phi.args.contains(&v) && seen.insert(phi.value) {
                                work.push(phi.value);
                            }
                        }
                    }
                    Some(i) => {
                        let inst = &self.p.blocks[b].insts[i];
                        if inst.name() == "v_mov_b32"
                            && single_reg_src(inst) == Some(v)
                            && let Some(&(_, w)) = inst.writes.first()
                            && seen.insert(w)
                        {
                            work.push(w);
                        }
                    }
                }
            }
        }
        seen
    }

    /// Locates the triangle-hit commit site inside `nodes` and classifies the loop's
    /// hit handling (`Opaque`, `FrontBackTrack`, or `AnyHit` sub-CFG).
    fn find_triangle_hit(
        &self,
        template: RayTemplate,
        tri_bvh: Site,
        extent_val: Value,
        t_max: RayInput,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<TriangleHitInfo> {
        let bvh_inst = &self.p.blocks[tri_bvh.0].insts[tri_bvh.1];
        let bvh_res = image_data_values(bvh_inst)?;
        if bvh_res.len() < 4 {
            return None;
        }
        let (t_num, det, bary_u_num, bary_v_num) = (bvh_res[0], bvh_res[1], bvh_res[2], bvh_res[3]);
        let det_vals = self.forward_phi_closure(det, nodes, None);
        let t_num_vals = self.forward_phi_closure(t_num, nodes, None);
        let bary_u_vals = self.forward_phi_closure(bary_u_num, nodes, None);
        let bary_v_vals = self.forward_phi_closure(bary_v_num, nodes, None);

        // Find `v_rcp_f32 rcp_det, det` inside `nodes`.
        let mut rcp_det_site: Option<(Site, Value)> = None;
        'find_rcp: for &dv in &det_vals {
            for &(b, idx_opt) in &self.uses[dv.0 as usize] {
                if !nodes.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else { continue };
                let inst = &self.p.blocks[b].insts[i];
                if inst.name() == "v_rcp_f32" {
                    if let Some(&(_, w)) = inst.writes.first() {
                        rcp_det_site = Some(((b, i), w));
                        break 'find_rcp;
                    }
                }
            }
        }
        let ((hit_block, _), rcp_det) = rcp_det_site?;

        // Find `t_hit = v_mul*_f32(rcp_det, t_num)` and optional barycentrics in `hit_block`.
        let mut t_hit_val: Option<Value> = None;
        let mut bary_u_val: Option<Value> = None;
        let mut bary_v_val: Option<Value> = None;
        for inst in &self.p.blocks[hit_block].insts {
            if !matches!(inst.name(), "v_mul_f32" | "v_mul_legacy_f32") {
                continue;
            }
            if !inst.reads.iter().any(|&(_, v)| v == rcp_det) {
                continue;
            }
            let Some(&(_, w)) = inst.writes.first() else {
                continue;
            };
            if inst.reads.iter().any(|&(_, v)| t_num_vals.contains(&v)) {
                t_hit_val = Some(w);
            } else if inst.reads.iter().any(|&(_, v)| bary_u_vals.contains(&v)) {
                bary_u_val = Some(w);
            } else if inst.reads.iter().any(|&(_, v)| bary_v_vals.contains(&v)) {
                bary_v_val = Some(w);
            }
        }
        let t_hit = t_hit_val?;

        // --- Layout A: inline `v_cndmask_b32` commit in `hit_block` (pure opaque) ---
        let mut hit_mask: Option<Value> = None;
        let mut miss_is_true = false;
        let mut committed_t_write: Option<Value> = None;
        let mut committed_t_reg: Option<Reg> = None;
        let mut committed_t_old: Option<Value> = None;

        for inst in &self.p.blocks[hit_block].insts {
            if inst.name() != "v_cndmask_b32" {
                continue;
            }
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            if f.src.len() < 3 {
                continue;
            }
            let s0 = operand_val(inst, &f.src[0]);
            let s1 = operand_val(inst, &f.src[1]);
            let Some(mask) = operand_val(inst, &f.src[2]) else {
                continue;
            };
            if s0 == Some(t_hit) && s1.is_some() {
                hit_mask = Some(mask);
                miss_is_true = true;
                committed_t_old = s1;
                if let Some(&(r, w)) = inst.writes.first() {
                    committed_t_reg = Some(r);
                    committed_t_write = Some(w);
                }
                break;
            } else if s1 == Some(t_hit) && s0.is_some() {
                hit_mask = Some(mask);
                miss_is_true = false;
                committed_t_old = s0;
                if let Some(&(r, w)) = inst.writes.first() {
                    committed_t_reg = Some(r);
                    committed_t_write = Some(w);
                }
                break;
            }
        }

        if let (Some(hit_mask), Some(committed_t_write), Some(committed_t_reg), Some(committed_t_old)) =
            (hit_mask, committed_t_write, committed_t_reg, committed_t_old)
        {
            let pre_ext = self.pre_loop_input(extent_val, nodes, entry_block, 0)?;
            let pre_old = self.pre_loop_input(committed_t_old, nodes, entry_block, 0)?;
            if pre_ext != pre_old || pre_ext != t_max {
                return None;
            }
            let t_min = self.extract_t_min_in_hit_block(
                hit_block,
                t_hit,
                committed_t_old,
                miss_is_true,
                nodes,
                entry_block,
            )?;

            let mut accept_first_hit = false;
            let mut committed_tri_reg: Option<Reg> = None;
            let mut committed_tri_write: Option<Value> = None;
            let mut committed_bary_u_reg: Option<Reg> = None;
            let mut committed_bary_v_reg: Option<Reg> = None;

            for inst in &self.p.blocks[hit_block].insts {
                if inst.name() != "v_cndmask_b32" {
                    continue;
                }
                let Fields::Valu(f) = &inst.fields else {
                    continue;
                };
                if f.src.len() < 3 || operand_val(inst, &f.src[2]) != Some(hit_mask) {
                    continue;
                }
                let Some(&(dst_reg, dst_val)) = inst.writes.first() else {
                    continue;
                };
                if dst_val == committed_t_write {
                    continue;
                }
                let c0 = const_operand(&f.src[0]);
                let c1 = const_operand(&f.src[1]);
                if (miss_is_true && c0 == Some(0) && c1 == Some(3))
                    || (!miss_is_true && c0 == Some(3) && c1 == Some(0))
                {
                    accept_first_hit = true;
                    continue;
                }
                let (new_op, old_op) = if miss_is_true {
                    (&f.src[0], &f.src[1])
                } else {
                    (&f.src[1], &f.src[0])
                };
                let new_v = operand_val(inst, new_op);
                let old_v = operand_val(inst, old_op);

                if new_v.is_some() && new_v == bary_u_val {
                    committed_bary_u_reg = Some(dst_reg);
                } else if new_v.is_some() && new_v == bary_v_val {
                    committed_bary_v_reg = Some(dst_reg);
                } else if let Some(ov) = old_v
                    && self.pre_loop_const(ov, nodes, entry_block) == Some(0)
                    && committed_tri_write.is_none()
                {
                    committed_tri_reg = Some(dst_reg);
                    committed_tri_write = Some(dst_val);
                }
            }

            return Some(TriangleHitInfo {
                hit_block,
                opacity_block: None,
                accept_first_hit,
                t_min,
                committed_t_reg,
                committed_tri_reg,
                committed_inst_reg: None,
                committed_bary_u_reg,
                committed_bary_v_reg,
                committed_front_face_reg: None,
                mode: HitMode::Opaque,
            });
        }

        // --- Layouts B, C, D: `s_and_saveexec_b32/b64` gated hit block ---
        let saveexec_idx = self.p.blocks[hit_block]
            .insts
            .iter()
            .position(|i| matches!(i.name(), "s_and_saveexec_b32" | "s_and_saveexec_b64"))?;

        // Extract `committed_t_old` and `t_min` from `t_hit` range compares before `saveexec_idx`.
        let (committed_t_old, t_min) = self.extract_range_bounds_in_hit_block(
            hit_block,
            saveexec_idx,
            t_hit,
            t_max,
            nodes,
            entry_block,
        )?;
        let committed_t_reg = self.value_reg(committed_t_old)?;

        // Check whether `hit_block` has straight-line commit after `saveexec_idx` (Layout B)
        // or branches via `s_cbranch_execz` to an `opacity_block` (Layouts C & D).
        let branches_after_saveexec = self.p.blocks[hit_block].insts[saveexec_idx + 1..]
            .iter()
            .any(|i| i.name() == "s_cbranch_execz");

        if !branches_after_saveexec {
            // --- Layout B: Straight-line `s_and_saveexec` opaque commit (`004..007`) ---
            let mut saw_t_commit = false;
            let mut committed_bary_u_reg = None;
            let mut committed_bary_v_reg = None;
            let mut committed_front_face_reg = None;
            let mut node_commits: Vec<(Reg, Value)> = Vec::new();
            let mut ff_cndmask_vals: HashSet<Value> = HashSet::new();

            for inst in &self.p.blocks[hit_block].insts[saveexec_idx + 1..] {
                match inst.name() {
                    "v_mov_b32" => {
                        let Some(&(dst_reg, _)) = inst.writes.first() else {
                            continue;
                        };
                        let Some(src_v) = single_reg_src(inst) else {
                            continue;
                        };
                        if src_v == t_hit && dst_reg == committed_t_reg {
                            saw_t_commit = true;
                        } else if self.reg_pre_loop_const(dst_reg, nodes, entry_block) == Some(0) {
                            node_commits.push((dst_reg, src_v));
                        }
                    }
                    "v_mul_f32" | "v_mul_legacy_f32" => {
                        if inst.reads.iter().any(|&(_, v)| v == rcp_det)
                            && let Some(&(dst_reg, _)) = inst.writes.first()
                        {
                            if inst.reads.iter().any(|&(_, v)| bary_u_vals.contains(&v)) {
                                committed_bary_u_reg = Some(dst_reg);
                            } else if inst.reads.iter().any(|&(_, v)| bary_v_vals.contains(&v)) {
                                committed_bary_v_reg = Some(dst_reg);
                            }
                        }
                    }
                    "v_cndmask_b32" => {
                        if self.is_front_face_cndmask(inst)
                            && let Some(&(_, w)) = inst.writes.first()
                        {
                            ff_cndmask_vals.insert(w);
                        }
                    }
                    "v_xor_b32" => {
                        if inst.reads.iter().any(|&(_, v)| ff_cndmask_vals.contains(&v))
                            && let Some(&(dst_reg, _)) = inst.writes.first()
                        {
                            committed_front_face_reg = Some(dst_reg);
                        }
                    }
                    _ => {}
                }
            }
            if !saw_t_commit {
                return None;
            }
            let (committed_tri_reg, committed_inst_reg) =
                self.disambiguate_tri_and_inst_regs(template, tri_bvh, nodes, &node_commits);

            return Some(TriangleHitInfo {
                hit_block,
                opacity_block: None,
                accept_first_hit: false,
                t_min,
                committed_t_reg,
                committed_tri_reg,
                committed_inst_reg,
                committed_bary_u_reg,
                committed_bary_v_reg,
                committed_front_face_reg,
                mode: HitMode::Opaque,
            });
        }

        // --- Layouts C & D: `hit_block` falls through to `opacity_block` ---
        let Exit::Branch { not_taken: opacity_block, .. } = self.p.blocks[hit_block].exit else {
            return None;
        };
        if !nodes.contains(&opacity_block) {
            return None;
        }

        // Locate `v_bfe_u32 _, _, 4, 2` -> `v_and_or_b32 op, _, 1, bfe` -> `v_cmpx_ne_u32 exec, 1, op`.
        let cmpx_idx = self.find_opacity_cmpx_in_block(opacity_block)?;

        let opacity_branches = self.p.blocks[opacity_block].insts[cmpx_idx + 1..]
            .iter()
            .any(|i| i.name() == "s_cbranch_execz");

        if !opacity_branches {
            // --- Layout C: `FrontBackTrack` (straight-line masked front/back tracking) ---
            return self.match_front_back_track(
                template,
                tri_bvh,
                hit_block,
                opacity_block,
                cmpx_idx,
                t_hit,
                t_min,
                committed_t_old,
                committed_t_reg,
                nodes,
                entry_block,
            );
        }

        // --- Layout D: General `AnyHit` sub-CFG (`opacity_block` -> `anyhit_entry` .. `anyhit_tail` -> `commit_block`) ---
        let Exit::Branch {
            taken: commit_block,
            not_taken: anyhit_entry,
        } = self.p.blocks[opacity_block].exit
        else {
            return None;
        };
        if !nodes.contains(&anyhit_entry) || !nodes.contains(&commit_block) || anyhit_entry == commit_block {
            return None;
        }

        self.analyze_anyhit_region(
            template,
            tri_bvh,
            hit_block,
            opacity_block,
            anyhit_entry,
            commit_block,
            rcp_det,
            t_hit,
            t_min,
            t_max,
            committed_t_old,
            committed_t_reg,
            &bary_u_vals,
            &bary_v_vals,
            bary_u_val,
            bary_v_val,
            nodes,
            entry_block,
        )
    }

    fn extract_t_min_in_hit_block(
        &self,
        hit_block: usize,
        t_hit: Value,
        committed_t_old: Value,
        miss_is_true: bool,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<Option<RayInput>> {
        let mut t_min: Option<RayInput> = None;
        for inst in &self.p.blocks[hit_block].insts {
            let n = inst.name();
            let Some((lhs, rhs)) = two_src_operands(inst) else {
                continue;
            };
            let lhs_is_t = operand_val(inst, lhs) == Some(t_hit);
            let rhs_is_t = operand_val(inst, rhs) == Some(t_hit);
            if !lhs_is_t && !rhs_is_t {
                continue;
            }
            let is_lower_bound = if !miss_is_true {
                (lhs_is_t && matches!(n, "v_cmp_ge_f32" | "v_cmp_gt_f32"))
                    || (rhs_is_t && matches!(n, "v_cmp_le_f32" | "v_cmp_lt_f32"))
            } else {
                (lhs_is_t && matches!(n, "v_cmp_lt_f32" | "v_cmp_le_f32"))
                    || (rhs_is_t && matches!(n, "v_cmp_gt_f32" | "v_cmp_ge_f32"))
            };
            if !is_lower_bound {
                continue;
            }
            let bound_op = if lhs_is_t { rhs } else { lhs };
            if let Some(k) = const_operand(bound_op) {
                if k == 0 || k == 0x8000_0000 {
                    continue;
                }
                t_min = Some(RayInput::Const(k));
                continue;
            }
            let bound_v = operand_val(inst, bound_op)?;
            if bound_v == committed_t_old {
                continue;
            }
            let pre_min = self.pre_loop_input(bound_v, nodes, entry_block, 0)?;
            match pre_min {
                RayInput::Const(0) | RayInput::Const(0x8000_0000) => continue,
                RayInput::Const(k) => t_min = Some(RayInput::Const(k)),
                RayInput::Val(pv) => {
                    let db = self.def_block(pv)?;
                    if nodes.contains(&db) || !self.dominates(db, entry_block) {
                        return None;
                    }
                    if self.eval_const(pv, 0).is_some_and(|k| k == 0 || k == 0x8000_0000) {
                        continue;
                    }
                    t_min = Some(RayInput::Val(pv));
                }
                RayInput::ShiftedMask(_) => return None,
            }
        }
        Some(t_min)
    }

    fn extract_range_bounds_in_hit_block(
        &self,
        hit_block: usize,
        saveexec_idx: usize,
        t_hit: Value,
        t_max: RayInput,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<(Value, Option<RayInput>)> {
        let mut committed_t_old: Option<Value> = None;
        let mut t_min: Option<RayInput> = None;
        for inst in &self.p.blocks[hit_block].insts[..saveexec_idx] {
            let n = inst.name();
            if !matches!(
                n,
                "v_cmp_ge_f32" | "v_cmp_gt_f32" | "v_cmp_lt_f32" | "v_cmp_le_f32"
            ) {
                continue;
            }
            let Some((lhs, rhs)) = two_src_operands(inst) else {
                continue;
            };
            let lhs_is_t = operand_val(inst, lhs) == Some(t_hit);
            let rhs_is_t = operand_val(inst, rhs) == Some(t_hit);
            if !lhs_is_t && !rhs_is_t {
                continue;
            }
            let other_op = if lhs_is_t { rhs } else { lhs };
            if let Some(k) = const_operand(other_op) {
                if k == 0 || k == 0x8000_0000 {
                    continue;
                }
                t_min = Some(RayInput::Const(k));
                continue;
            }
            let other_v = operand_val(inst, other_op)?;
            if self.pre_loop_input(other_v, nodes, entry_block, 0) == Some(t_max) {
                committed_t_old = Some(other_v);
                continue;
            }
            // Otherwise must be dynamic `t_min` dominating `entry_block`.
            let pre_min = self.pre_loop_input(other_v, nodes, entry_block, 0)?;
            match pre_min {
                RayInput::Const(0) | RayInput::Const(0x8000_0000) => continue,
                RayInput::Const(k) => t_min = Some(RayInput::Const(k)),
                RayInput::Val(pv) => {
                    let db = self.def_block(pv)?;
                    if nodes.contains(&db) || !self.dominates(db, entry_block) {
                        return None;
                    }
                    if self.eval_const(pv, 0).is_some_and(|k| k == 0 || k == 0x8000_0000) {
                        continue;
                    }
                    t_min = Some(RayInput::Val(pv));
                }
                RayInput::ShiftedMask(_) => return None,
            }
        }
        Some((committed_t_old?, t_min))
    }

    fn find_opacity_cmpx_in_block(&self, b: usize) -> Option<usize> {
        let mut bfe_vals: HashSet<Value> = HashSet::new();
        let mut op_vals: HashSet<Value> = HashSet::new();
        for (idx, inst) in self.p.blocks[b].insts.iter().enumerate() {
            match inst.name() {
                "v_bfe_u32" => {
                    if let Fields::Valu(f) = &inst.fields
                        && f.src.len() >= 3
                        && const_operand(&f.src[1]) == Some(4)
                        && const_operand(&f.src[2]) == Some(2)
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        bfe_vals.insert(w);
                    }
                }
                "v_and_or_b32" => {
                    if let Fields::Valu(f) = &inst.fields
                        && f.src.len() >= 3
                        && const_operand(&f.src[1]) == Some(1)
                        && operand_val(inst, &f.src[2]).is_some_and(|v| bfe_vals.contains(&v))
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        op_vals.insert(w);
                    }
                }
                "v_cmpx_ne_u32" => {
                    if has_const_operand(inst, 1)
                        && inst.reads.iter().any(|&(_, v)| op_vals.contains(&v))
                    {
                        return Some(idx);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn is_front_face_cndmask(&self, inst: &Inst) -> bool {
        if inst.name() != "v_cndmask_b32" {
            return false;
        }
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        if f.src.len() < 2 {
            return false;
        }
        let c0 = const_operand(&f.src[0])
            .or_else(|| operand_val(inst, &f.src[0]).and_then(|v| self.eval_const(v, 0)));
        let c1 = const_operand(&f.src[1])
            .or_else(|| operand_val(inst, &f.src[1]).and_then(|v| self.eval_const(v, 0)));
        matches!((c0, c1), (Some(0xFF), Some(0xFE)) | (Some(0xFE), Some(0xFF)))
    }

    fn reg_pre_loop_const(&self, reg: Reg, nodes: &HashSet<usize>, entry_block: usize) -> Option<u32> {
        let phi = self.p.blocks[entry_block].phis.iter().find(|p| p.reg == reg)?;
        self.pre_loop_const(phi.value, nodes, entry_block)
    }

    /// Distinguishes `committed_tri_reg` vs `committed_inst_reg` among 0-initialized node commits.
    fn disambiguate_tri_and_inst_regs(
        &self,
        template: RayTemplate,
        tri_bvh: Site,
        nodes: &HashSet<usize>,
        commits: &[(Reg, Value)],
    ) -> (Option<Reg>, Option<Reg>) {
        if commits.is_empty() {
            return (None, None);
        }
        // Identify the register carrying `cur_tri` from `tri_bvh` or triangle-branch `v_cmpx_gt_u32 4, cur_node`.
        let cur_tri_regs = self.find_cur_tri_regs(template, tri_bvh, nodes);
        let cur_inst_regs = self.find_cur_inst_regs(template, nodes);

        let mut tri_reg = None;
        let mut inst_reg = None;
        let mut unknown = Vec::new();
        // Check `cur_inst_regs` FIRST (unambiguous: feeds instance-node `+48` load!),
        // then `cur_tri_regs` (excluding any reg also in `cur_inst_regs`).
        for &(dst_reg, src_val) in commits {
            let src_reg = self.value_reg(src_val);
            if src_reg.is_some_and(|r| cur_inst_regs.contains(&r)) && inst_reg.is_none() {
                inst_reg = Some(dst_reg);
            } else if src_reg.is_some_and(|r| cur_tri_regs.contains(&r)) && tri_reg.is_none() {
                tri_reg = Some(dst_reg);
            } else {
                unknown.push(dst_reg);
            }
        }
        for r in unknown {
            if tri_reg.is_none() {
                tri_reg = Some(r);
            } else if inst_reg.is_none() {
                inst_reg = Some(r);
            }
        }
        (tri_reg, inst_reg)
    }

    fn find_cur_tri_regs(
        &self,
        template: RayTemplate,
        tri_bvh: Site,
        nodes: &HashSet<usize>,
    ) -> HashSet<Reg> {
        let mut regs = HashSet::new();
        let bvh_inst = &self.p.blocks[tri_bvh.0].insts[tri_bvh.1];
        if let Some(addrs) = image_addr_values(bvh_inst) {
            if template == RayTemplate::Bvh32SharedTlas
                && let Some(&node_v) = addrs.first()
                && let Some(r) = self.value_reg(node_v)
            {
                regs.insert(r);
            }
        }
        // Only collect registers directly copied from `cur_node` in a block containing
        // `v_cmpx_gt_u32 exec, 4, cur_node` (triangle-node test), NOT general box/inst copies.
        for &b in nodes {
            for inst in &self.p.blocks[b].insts {
                if inst.name() == "v_cmpx_gt_u32"
                    && has_const_operand(inst, 4)
                    && let Some(node_v) = other_val_from_const_op(inst, 4)
                {
                    if let Some(r) = self.value_reg(node_v) {
                        regs.insert(r);
                    }
                    for &(ub, uidx) in &self.uses[node_v.0 as usize] {
                        if ub == b
                            && let Some(ui) = uidx
                            && self.p.blocks[ub].insts[ui].name() == "v_mov_b32"
                            && let Some(&(wr, _)) = self.p.blocks[ub].insts[ui].writes.first()
                        {
                            regs.insert(wr);
                        }
                    }
                }
            }
        }
        regs
    }

    fn find_cur_inst_regs(&self, template: RayTemplate, nodes: &HashSet<usize>) -> HashSet<Reg> {
        let mut regs = HashSet::new();
        for &b in nodes {
            for inst in &self.p.blocks[b].insts {
                // In Template B, the outer TLAS loop checks `(cur_inst & 7) == 6` (`s_and_b32 _, s_inst, 7`).
                if template == RayTemplate::Bvh32SharedTlas
                    && inst.name() == "s_and_b32"
                    && has_const_operand(inst, 7)
                    && let Some(cand_v) = other_val_from_const_op(inst, 7)
                    && let Some(&(_, dst_v)) = inst.writes.first()
                {
                    let is_inst_test = self.uses[dst_v.0 as usize].iter().any(|&(ub, uidx)| {
                        nodes.contains(&ub)
                            && uidx.is_some_and(|ui| {
                                let uinst = &self.p.blocks[ub].insts[ui];
                                matches!(uinst.name(), "s_cmp_eq_u32" | "s_cmp_lg_u32")
                                    && has_const_operand(uinst, 6)
                            })
                    });
                    if is_inst_test && let Some(r) = self.value_reg(cand_v) {
                        regs.insert(r);
                    }
                }
                let is_inst_load = match template {
                    RayTemplate::Bvh64 => inst.name() == "global_load_dwordx4",
                    RayTemplate::Bvh32SharedTlas => inst.name() == "s_load_dwordx4",
                };
                if !is_inst_load {
                    continue;
                }
                let addr_lo = match &inst.fields {
                    Fields::Flat {
                        addr: Some(Operand::Vector { reg, .. }),
                        offset,
                        ..
                    } if *offset == 48
                        || inst
                            .read(Reg::V(*reg))
                            .is_some_and(|v| self.addr_has_offset_48(v, 0)) =>
                    {
                        inst.read(Reg::V(*reg))
                    }
                    Fields::Smem {
                        base: Operand::Scalar { code, .. },
                        offset,
                        soffset,
                        ..
                    } if (*offset + const_operand(soffset).unwrap_or(0) as i32) == 48
                        || inst
                            .read(Reg::S(*code))
                            .is_some_and(|v| self.addr_has_offset_48(v, 0)) =>
                    {
                        inst.read(Reg::S(*code))
                    }
                    _ => None,
                };
                let Some(seed) = addr_lo else { continue };
                let mut seen = HashSet::new();
                let mut work = vec![(seed, 0u32)];
                while let Some((v, depth)) = work.pop() {
                    if depth > 10 || !seen.insert(v) {
                        continue;
                    }
                    let Some(def) = self.defs.get(v.0 as usize).copied().flatten() else {
                        continue;
                    };
                    match def {
                        Def::Input(_) => continue,
                        Def::Phi(pb, pi) if nodes.contains(&pb) => {
                            for &arg in &self.p.blocks[pb].phis[pi].args {
                                if arg != v {
                                    work.push((arg, depth + 1));
                                }
                            }
                            continue;
                        }
                        Def::Phi(_, _) => continue,
                        Def::Inst(_, _) => {}
                    }
                    let Some((_, di)) = self.def_inst(v) else {
                        continue;
                    };
                    if matches!(di.name(), "v_lshrrev_b32" | "s_lshr_b32")
                        && has_const_operand(di, 3)
                        && let Some(inst_node_v) = other_val_from_const_op(di, 3)
                    {
                        if let Some(r) = self.value_reg(inst_node_v) {
                            regs.insert(r);
                        }
                        for &(ub, uidx) in &self.uses[inst_node_v.0 as usize] {
                            if nodes.contains(&ub)
                                && let Some(ui) = uidx
                                && matches!(self.p.blocks[ub].insts[ui].name(), "v_mov_b32" | "s_mov_b32")
                                && let Some(&(wr, _)) = self.p.blocks[ub].insts[ui].writes.first()
                            {
                                regs.insert(wr);
                            }
                        }
                        continue;
                    }
                    if matches!(
                        di.name(),
                        "v_or_b32"
                            | "s_or_b32"
                            | "v_add_co_u32"
                            | "v_add_nc_u32"
                            | "s_add_u32"
                            | "v_lshlrev_b64"
                            | "s_lshl_b64"
                            | "v_mov_b32"
                            | "s_mov_b32"
                    ) {
                        for &r in &operand_reads(self.p, di) {
                            if let Some(sv) = di.read(r) {
                                work.push((sv, depth + 1));
                            }
                        }
                    }
                }
            }
        }
        regs
    }

    #[allow(clippy::too_many_arguments)]
    fn match_front_back_track(
        &self,
        template: RayTemplate,
        tri_bvh: Site,
        hit_block: usize,
        opacity_block: usize,
        cmpx_idx: usize,
        t_hit: Value,
        t_min: Option<RayInput>,
        committed_t_old: Value,
        committed_t_reg: Reg,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<TriangleHitInfo> {
        let insts = &self.p.blocks[opacity_block].insts[cmpx_idx + 1..];
        let mut ff_cndmasks: HashSet<Value> = HashSet::new();
        let mut ff_vals: HashSet<Value> = HashSet::new();
        let mut ff_mask: Option<Value> = None;
        let mut ff_mask_is_front = true;
        let mut min_v: Option<Value> = None;
        let mut min_old: Option<Value> = None;
        let mut min_front_t_reg: Option<Reg> = None;
        let mut t_sel_val: Option<Value> = None;
        let mut node_sels: Vec<(Value, Value, Reg)> = Vec::new(); // (sel_val, cur_node_val, old_reg)
        let mut saw_t_commit = false;
        let mut committed_nodes: Vec<(Reg, Value)> = Vec::new();

        for inst in insts {
            match inst.name() {
                "v_cndmask_b32" if self.is_front_face_cndmask(inst) => {
                    if let Some(&(_, w)) = inst.writes.first() {
                        ff_cndmasks.insert(w);
                    }
                }
                "v_xor_b32" => {
                    if inst.reads.iter().any(|&(_, v)| ff_cndmasks.contains(&v))
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        ff_vals.insert(w);
                    }
                }
                "v_cmp_eq_u32" | "v_cmp_ne_u32" => {
                    if inst.reads.iter().any(|&(_, v)| ff_vals.contains(&v))
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        if (inst.name() == "v_cmp_eq_u32" && has_const_operand(inst, 0xFE))
                            || (inst.name() == "v_cmp_ne_u32" && has_const_operand(inst, 0xFF))
                        {
                            ff_mask = Some(w);
                            ff_mask_is_front = true;
                        } else if (inst.name() == "v_cmp_eq_u32" && has_const_operand(inst, 0xFF))
                            || (inst.name() == "v_cmp_ne_u32" && has_const_operand(inst, 0xFE))
                        {
                            ff_mask = Some(w);
                            ff_mask_is_front = false;
                        }
                    }
                }
                "v_min_f32" => {
                    if let Some((a, b)) = two_src_operands(inst) {
                        let va = operand_val(inst, a);
                        let vb = operand_val(inst, b);
                        if va == Some(t_hit) && let Some(o) = vb {
                            min_old = Some(o);
                            min_v = inst.writes.first().map(|&(_, w)| w);
                        } else if vb == Some(t_hit) && let Some(o) = va {
                            min_old = Some(o);
                            min_v = inst.writes.first().map(|&(_, w)| w);
                        }
                    }
                }
                "v_cndmask_b32" => {
                    let Fields::Valu(f) = &inst.fields else {
                        continue;
                    };
                    if f.src.len() < 3 || operand_val(inst, &f.src[2]) != ff_mask {
                        continue;
                    }
                    let s0 = operand_val(inst, &f.src[0]);
                    let s1 = operand_val(inst, &f.src[1]);
                    let Some(&(dst_reg, dst_val)) = inst.writes.first() else {
                        continue;
                    };
                    // `minFrontFaceT = select(is_front, min_v, min_old)`
                    if ff_mask_is_front && s0 == min_old && s1 == min_v && min_v.is_some() {
                        min_front_t_reg = Some(dst_reg);
                    } else if !ff_mask_is_front && s0 == min_v && s1 == min_old && min_v.is_some() {
                        min_front_t_reg = Some(dst_reg);
                    } else if ff_mask_is_front && s0 == Some(t_hit) && s1 == Some(committed_t_old) {
                        // Back-face commits `t_hit`!
                        t_sel_val = Some(dst_val);
                    } else if !ff_mask_is_front && s0 == Some(committed_t_old) && s1 == Some(t_hit) {
                        t_sel_val = Some(dst_val);
                    } else {
                        let (new_v, old_v) = if ff_mask_is_front { (s0, s1) } else { (s1, s0) };
                        if let (Some(nv), Some(ov)) = (new_v, old_v)
                            && self.pre_loop_const(ov, nodes, entry_block) == Some(0)
                            && let Some(old_reg) = self.value_reg(ov)
                        {
                            node_sels.push((dst_val, nv, old_reg));
                        }
                    }
                }
                "v_mov_b32" => {
                    let Some(&(dst_reg, _)) = inst.writes.first() else {
                        continue;
                    };
                    let Some(src_v) = single_reg_src(inst) else {
                        continue;
                    };
                    if Some(src_v) == t_sel_val && dst_reg == committed_t_reg {
                        saw_t_commit = true;
                    } else if let Some(&(_, cur_v, _)) = node_sels.iter().find(|&&(sv, _, _)| sv == src_v) {
                        committed_nodes.push((dst_reg, cur_v));
                    }
                }
                _ => {}
            }
        }

        let min_front_t_reg = min_front_t_reg?;
        let min_old = min_old?;
        if !saw_t_commit {
            return None;
        }
        let init_min_front_t = self.pre_loop_input(min_old, nodes, entry_block, 0)?;
        if let RayInput::Val(iv) = init_min_front_t {
            let db = self.def_block(iv)?;
            if nodes.contains(&db) || !self.dominates(db, entry_block) {
                return None;
            }
        }
        let (committed_tri_reg, committed_inst_reg) =
            self.disambiguate_tri_and_inst_regs(template, tri_bvh, nodes, &committed_nodes);

        Some(TriangleHitInfo {
            hit_block,
            opacity_block: Some(opacity_block),
            accept_first_hit: false,
            t_min,
            committed_t_reg,
            committed_tri_reg,
            committed_inst_reg,
            committed_bary_u_reg: None,
            committed_bary_v_reg: None,
            committed_front_face_reg: None,
            mode: HitMode::FrontBackTrack {
                init_min_front_t,
                min_front_t_reg,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_anyhit_region(
        &self,
        template: RayTemplate,
        tri_bvh: Site,
        hit_block: usize,
        opacity_block: usize,
        anyhit_entry: usize,
        commit_block: usize,
        rcp_det: Value,
        t_hit: Value,
        t_min: Option<RayInput>,
        t_max: RayInput,
        committed_t_old: Value,
        committed_t_reg: Reg,
        bary_u_vals: &HashSet<Value>,
        bary_v_vals: &HashSet<Value>,
        hit_bary_u: Option<Value>,
        hit_bary_v: Option<Value>,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<TriangleHitInfo> {
        macro_rules! fail_ah {
            ($($arg:tt)*) => {{
                return None;
            }};
        }
        // 1. Collect all blocks reachable from `anyhit_entry` without passing through `commit_block`.
        let mut anyhit_set: HashSet<usize> = HashSet::new();
        let mut work = vec![anyhit_entry];
        anyhit_set.insert(anyhit_entry);
        while let Some(b) = work.pop() {
            for s in self.p.blocks[b].exit.successors() {
                if s == commit_block {
                    continue;
                }
                if !nodes.contains(&s) || s == entry_block || s == hit_block || s == opacity_block {
                    fail_ah!("step1 bad succ B{s} from B{b}");
                }
                if anyhit_set.insert(s) {
                    work.push(s);
                }
            }
        }
        if anyhit_set.contains(&commit_block) || anyhit_set.is_empty() {
            fail_ah!("step1 empty or contains commit");
        }

        // 2. Verify single-entry (no pred outside `anyhit_set` except `opacity_block -> anyhit_entry`,
        //    and no back-edge to `anyhit_entry`!) and single-exit (`tail_block` jumps to `commit_block`).
        let mut tail_candidates: Vec<usize> = Vec::new();
        for &b in &anyhit_set {
            for &pred in &self.p.blocks[b].preds {
                if b == anyhit_entry {
                    if pred != opacity_block {
                        fail_ah!("step2 entry B{b} has pred B{pred} != opacity B{opacity_block}");
                    }
                } else if !anyhit_set.contains(&pred) {
                    fail_ah!("step2 body B{b} has outside pred B{pred}");
                }
            }
            for s in self.p.blocks[b].exit.successors() {
                if s == commit_block {
                    tail_candidates.push(b);
                } else if !anyhit_set.contains(&s) {
                    fail_ah!("step2 body B{b} exits to non-commit B{s}");
                }
            }
            // Reject BVH/LDS/side-effect instructions inside any-hit body.
            for inst in &self.p.blocks[b].insts {
                let n = inst.name();
                if n.contains("bvh") || n.starts_with("ds_") {
                    fail_ah!("step2 forbidden inst {n} in B{b}");
                }
            }
        }
        if tail_candidates.len() != 1 {
            fail_ah!("step2 tail_candidates={tail_candidates:?}");
        }
        let tail_block = tail_candidates[0];
        if !matches!(self.p.blocks[tail_block].exit, Exit::Jump(t) if t == commit_block) {
            fail_ah!("step2 tail exit={:?} != Jump(B{commit_block})", self.p.blocks[tail_block].exit);
        }

        // Verify `commit_block` only copies committed state and restores EXEC.
        for inst in &self.p.blocks[commit_block].insts {
            if !matches!(
                inst.name(),
                "s_mov_b32" | "s_mov_b64" | "v_mov_b32" | "s_nop" | "s_branch"
            ) {
                fail_ah!("step2 commit_block B{commit_block} has inst {}", inst.name());
            }
        }

        // 3. Collect candidate values computed in `hit_block` and `opacity_block`.
        let mut cand_t_vals: HashSet<Value> = HashSet::new();
        cand_t_vals.insert(t_hit);
        let mut cand_bary_u_vals: HashSet<Value> = HashSet::new();
        if let Some(u) = hit_bary_u {
            cand_bary_u_vals.insert(u);
        }
        let mut cand_bary_v_vals: HashSet<Value> = HashSet::new();
        if let Some(v) = hit_bary_v {
            cand_bary_v_vals.insert(v);
        }
        let mut cand_ff_vals: HashSet<Value> = HashSet::new();
        let mut ff_cndmasks: HashSet<Value> = HashSet::new();
        // Track `v_mov_b32 dst, src` in `opacity_block` (opaque path initialisers for `commit_block`).
        let mut opacity_copies: HashMap<Value, Value> = HashMap::new();

        for inst in &self.p.blocks[opacity_block].insts {
            match inst.name() {
                "v_mul_f32" | "v_mul_legacy_f32" => {
                    if inst.reads.iter().any(|&(_, v)| v == rcp_det)
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        if inst.reads.iter().any(|&(_, v)| bary_u_vals.contains(&v)) {
                            cand_bary_u_vals.insert(w);
                        } else if inst.reads.iter().any(|&(_, v)| bary_v_vals.contains(&v)) {
                            cand_bary_v_vals.insert(w);
                        }
                    }
                }
                "v_cndmask_b32" if self.is_front_face_cndmask(inst) => {
                    if let Some(&(_, w)) = inst.writes.first() {
                        ff_cndmasks.insert(w);
                    }
                }
                "v_xor_b32" => {
                    if inst.reads.iter().any(|&(_, v)| ff_cndmasks.contains(&v))
                        && let Some(&(_, w)) = inst.writes.first()
                    {
                        cand_ff_vals.insert(w);
                    }
                }
                "v_mov_b32" => {
                    if let (Some(&(_, w)), Some(src_v)) = (inst.writes.first(), single_reg_src(inst)) {
                        opacity_copies.insert(w, src_v);
                        if cand_t_vals.contains(&src_v) {
                            cand_t_vals.insert(w);
                        }
                    }
                }
                _ => {}
            }
        }

        // 4. Find the trailing commit-select `v_cndmask_b32` cluster in `tail_block`.
        let mut accept_mask: Option<Value> = None;
        let mut accept_invert = false;
        let mut t_sel_val: Option<Value> = None;

        for inst in &self.p.blocks[tail_block].insts {
            if inst.name() != "v_cndmask_b32" {
                continue;
            }
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            if f.src.len() < 3 {
                continue;
            }
            let s0 = operand_val(inst, &f.src[0]);
            let s1 = operand_val(inst, &f.src[1]);
            let Some(mask) = operand_val(inst, &f.src[2]) else {
                continue;
            };
            let is_old_t = |ov: Option<Value>| {
                ov.is_some_and(|v| {
                    v == committed_t_old
                        || self.pre_loop_input(v, nodes, entry_block, 0) == Some(t_max)
                })
            };
            if s1.is_some_and(|v| cand_t_vals.contains(&v)) && is_old_t(s0) {
                accept_mask = Some(mask);
                accept_invert = false;
                t_sel_val = inst.writes.first().map(|&(_, w)| w);
            } else if s0.is_some_and(|v| cand_t_vals.contains(&v)) && is_old_t(s1) {
                accept_mask = Some(mask);
                accept_invert = true;
                t_sel_val = inst.writes.first().map(|&(_, w)| w);
            }
        }
        let Some(accept_mask) = accept_mask else {
            fail_ah!("step4 no accept_mask in tail B{tail_block}@{:#x} cand_t={cand_t_vals:?}", self.p.blocks[tail_block].offset);
        };
        let Some(t_sel_val) = t_sel_val else {
            fail_ah!("step4 no t_sel_val");
        };

        // Collect ALL `v_cndmask_b32` instructions in `tail_block` using `accept_mask`.
        let mut skipped_tail_insts: Vec<usize> = Vec::new();
        let mut accept_first_hit = false;
        // Map `sel_val` -> `(new_cand_val, old_committed_val)` for commit_block classification.
        let mut tail_sels: HashMap<Value, (Option<Value>, Option<Value>)> = HashMap::new();
        let mut cand_node_vals: Vec<(Value, Value)> = Vec::new(); // (sel_val, cur_node_val)

        for (idx, inst) in self.p.blocks[tail_block].insts.iter().enumerate() {
            if inst.name() != "v_cndmask_b32" {
                continue;
            }
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            if f.src.len() < 3 || operand_val(inst, &f.src[2]) != Some(accept_mask) {
                continue;
            }
            let Some(&(_, dst_val)) = inst.writes.first() else {
                continue;
            };
            skipped_tail_insts.push(idx);
            let c0 = const_operand(&f.src[0]);
            let c1 = const_operand(&f.src[1]);
            if (!accept_invert && c0 == Some(3) && c1 == Some(0))
                || (accept_invert && c0 == Some(0) && c1 == Some(3))
            {
                accept_first_hit = true;
                continue;
            }
            let (new_op, old_op) = if accept_invert {
                (&f.src[0], &f.src[1])
            } else {
                (&f.src[1], &f.src[0])
            };
            let new_v = operand_val(inst, new_op);
            let old_v = operand_val(inst, old_op);
            tail_sels.insert(dst_val, (new_v, old_v));
            if let (Some(nv), Some(ov)) = (new_v, old_v)
                && dst_val != t_sel_val
                && !cand_bary_u_vals.contains(&nv)
                && !cand_bary_v_vals.contains(&nv)
                && !cand_ff_vals.contains(&nv)
                && self.pre_loop_const(ov, nodes, entry_block) == Some(0)
            {
                cand_node_vals.push((dst_val, nv));
            }
        }
        let skipped_tail_set: HashSet<usize> = skipped_tail_insts.iter().copied().collect();

        // 5. Inspect `commit_block` to map every committed register to its field.
        //    Each `v_mov_b32 committed_reg, phi_val` reads a PHI joining `(opacity_val, tail_sel_val)`.
        let mut committed_bary_u_reg = None;
        let mut committed_bary_v_reg = None;
        let mut committed_front_face_reg = None;
        let mut node_commits: Vec<(Reg, Value)> = Vec::new();
        let mut cur_node_candidates: Vec<(Reg, Value)> = Vec::new();

        for inst in &self.p.blocks[commit_block].insts {
            if inst.name() != "v_mov_b32" {
                continue;
            }
            let Some(&(dst_reg, _)) = inst.writes.first() else {
                continue;
            };
            let Some(src_v) = single_reg_src(inst) else {
                continue;
            };
            // Resolve through `commit_block`'s PHI to find the `tail_block` incoming value.
            let tail_v = if let Some(Def::Phi(pb, pi)) = self.defs.get(src_v.0 as usize).copied().flatten()
                && pb == commit_block
            {
                let phi = &self.p.blocks[pb].phis[pi];
                self.p.blocks[pb]
                    .preds
                    .iter()
                    .zip(&phi.args)
                    .find(|&(&q, _)| q == tail_block)
                    .map(|(_, &a)| a)
                    .unwrap_or(src_v)
            } else {
                src_v
            };
            if tail_v == t_sel_val {
                continue;
            }
            if let Some(&(Some(new_v), _)) = tail_sels.get(&tail_v) {
                if cand_bary_u_vals.contains(&new_v) {
                    committed_bary_u_reg = Some(dst_reg);
                } else if cand_bary_v_vals.contains(&new_v) {
                    committed_bary_v_reg = Some(dst_reg);
                } else if cand_ff_vals.contains(&new_v) {
                    committed_front_face_reg = Some(dst_reg);
                } else if cand_node_vals.iter().any(|&(sv, _)| sv == tail_v) {
                    node_commits.push((dst_reg, new_v));
                    cur_node_candidates.push((dst_reg, new_v));
                }
            }
        }
        let (mut committed_tri_reg, mut committed_inst_reg) =
            self.disambiguate_tri_and_inst_regs(template, tri_bvh, nodes, &node_commits);

        // Helper to build `(cur_tri_vals, cur_inst_vals, boff_roots, hi_roots)` for a given
        // `(committed_tri_reg, committed_inst_reg)` assignment and run candidate metadata analysis.
        let scaffolding_reads = self.collect_anyhit_scaffolding_reads(
            &anyhit_set,
            tail_block,
            &skipped_tail_set,
            nodes,
        );
        let try_meta = |tri_r: Option<Reg>, inst_r: Option<Reg>| {
            let mut cur_tri_vals: HashSet<Value> = HashSet::new();
            let mut cur_inst_vals: HashSet<Value> = HashSet::new();
            for &(dst_reg, new_v) in &cur_node_candidates {
                if Some(dst_reg) == tri_r {
                    cur_tri_vals.insert(new_v);
                    if let Some(&orig) = opacity_copies.get(&new_v) {
                        cur_tri_vals.insert(orig);
                    }
                } else if Some(dst_reg) == inst_r {
                    cur_inst_vals.insert(new_v);
                    if let Some(&orig) = opacity_copies.get(&new_v) {
                        cur_inst_vals.insert(orig);
                    }
                }
            }
            let mut cur_tri_reg_set: HashSet<Reg> = cur_tri_vals
                .iter()
                .filter_map(|&v| self.value_reg(v))
                .collect();
            let mut cur_inst_reg_set: HashSet<Reg> = cur_inst_vals
                .iter()
                .filter_map(|&v| self.value_reg(v))
                .collect();
            if cur_tri_reg_set.is_empty() {
                cur_tri_reg_set.extend(self.find_cur_tri_regs(template, tri_bvh, nodes));
            }
            if cur_inst_reg_set.is_empty() {
                cur_inst_reg_set.extend(self.find_cur_inst_regs(template, nodes));
            }
            // Fallback: if no `scaffolding_reads` matched `cur_inst_reg_set` (e.g. `024` L3 where
            // `cur_inst` was copied to `v12` across the BLAS loop), detect a scaffolding value
            // inside `anyhit_set` shifted `>> 3` into `(v >> 3) << 6 + base` (not `cur_tri`).
            let has_scaff_inst = scaffolding_reads
                .iter()
                .any(|v| self.value_reg(*v).is_some_and(|r| cur_inst_reg_set.contains(&r)));
            if !has_scaff_inst {
                for &v in &scaffolding_reads {
                    if cur_tri_vals.contains(&v)
                        || self.value_reg(v).is_some_and(|r| cur_tri_reg_set.contains(&r))
                    {
                        continue;
                    }
                    let used_as_shr3 = self.uses[v.0 as usize].iter().any(|&(ub, uidx)| {
                        anyhit_set.contains(&ub)
                            && uidx.is_some_and(|ui| {
                                let uinst = &self.p.blocks[ub].insts[ui];
                                matches!(uinst.name(), "v_lshrrev_b32" | "s_lshr_b32")
                                    && other_const_arg(uinst, v) == Some(3)
                            })
                    });
                    if used_as_shr3 && let Some(r) = self.value_reg(v) {
                        cur_inst_reg_set.insert(r);
                    }
                }
            }
            for &v in &scaffolding_reads {
                if let Some(r) = self.value_reg(v) {
                    if cur_inst_reg_set.contains(&r) {
                        cur_inst_vals.insert(v);
                    } else if cur_tri_reg_set.contains(&r) {
                        cur_tri_vals.insert(v);
                    }
                }
            }
            let mut inst_byte_off_lo_roots: Vec<Value> = Vec::new();
            let mut inst_addr_hi_roots: Vec<Value> = Vec::new();
            for &v in &scaffolding_reads {
                if cur_tri_vals.contains(&v)
                    || cur_inst_vals.contains(&v)
                    || cand_t_vals.contains(&v)
                    || cand_bary_u_vals.contains(&v)
                    || cand_bary_v_vals.contains(&v)
                    || cand_ff_vals.contains(&v)
                {
                    continue;
                }
                if let Some(tag) =
                    self.classify_scaffolding_inst_addr_val(v, &cur_inst_reg_set, nodes)
                {
                    match tag {
                        BvhTag::InstByteOff(_) => inst_byte_off_lo_roots.push(v),
                        BvhTag::AddrHi => inst_addr_hi_roots.push(v),
                        _ => {}
                    }
                }
            }
            let tri_roots_vec: Vec<Value> = cur_tri_vals.iter().copied().collect();
            let inst_roots_vec: Vec<Value> = cur_inst_vals.iter().copied().collect();
            let meta = self.analyze_meta_uses(
                &anyhit_set,
                &inst_roots_vec,
                &tri_roots_vec,
                &inst_byte_off_lo_roots,
                &inst_addr_hi_roots,
                Some((tail_block, &skipped_tail_set)),
                &[],
                &HashSet::new(),
            )?;
            Some((
                cur_tri_vals,
                cur_inst_vals,
                inst_byte_off_lo_roots,
                inst_addr_hi_roots,
                meta,
            ))
        };

        let meta_attempt = try_meta(committed_tri_reg, committed_inst_reg).or_else(|| {
            if committed_tri_reg.is_some()
                && committed_inst_reg.is_some()
                && committed_tri_reg != committed_inst_reg
            {
                let res = try_meta(committed_inst_reg, committed_tri_reg)?;
                std::mem::swap(&mut committed_tri_reg, &mut committed_inst_reg);
                Some(res)
            } else {
                None
            }
        });
        let Some((
            cur_tri_vals,
            cur_inst_vals,
            inst_byte_off_lo_roots,
            inst_addr_hi_roots,
            meta_res,
        )) = meta_attempt
        else {
            fail_ah!(
                "step6 cand analyze_meta_uses failed tri_reg={committed_tri_reg:?} inst_reg={committed_inst_reg:?}"
            );
        };
        let cand_meta_loads = meta_res.loads;

        let mut dead_addr_inputs: HashSet<Value> = HashSet::new();
        dead_addr_inputs.extend(inst_byte_off_lo_roots.iter().copied());
        dead_addr_inputs.extend(inst_addr_hi_roots.iter().copied());

        // 7. Discover loop-carried user state variables (read/written inside `anyhit_set`,
        //    untouched by BVH scaffolding outside `anyhit_set`).
        let mut committed_regs: HashSet<Reg> = HashSet::new();
        committed_regs.insert(committed_t_reg);
        if let Some(r) = committed_tri_reg {
            committed_regs.insert(r);
        }
        if let Some(r) = committed_inst_reg {
            committed_regs.insert(r);
        }
        if let Some(r) = committed_bary_u_reg {
            committed_regs.insert(r);
        }
        if let Some(r) = committed_bary_v_reg {
            committed_regs.insert(r);
        }
        if let Some(r) = committed_front_face_reg {
            committed_regs.insert(r);
        }

        let mut state_vars: Vec<AnyHitStateVar> = Vec::new();
        let mut state_val_to_idx: HashMap<Value, usize> = HashMap::new();

        for phi in &self.p.blocks[entry_block].phis {
            if !matches!(phi.reg, Reg::V(_)) || committed_regs.contains(&phi.reg) {
                continue;
            }
            if let Some((init, next, scaff_vals)) =
                self.match_anyhit_state_var(phi, nodes, entry_block, &anyhit_set, tail_block)
            {
                // Only treat `phi` as a real loop-carried state variable if `anyhit_set` actually
                // reads it (not just a dead implicit guarded overwrite) OR if `phi.reg` is read
                // as an operand after the loop. Dead local any-hit scratch temps (like `v63`/`v64`)
                // must NOT become fake `state_vars` that pull in earlier loops' dead VGPRs!
                let read_in_anyhit = scaff_vals.iter().any(|sv| {
                    scaffolding_reads.contains(sv)
                        && !self.is_dead_implicit_guarded_read(
                            *sv,
                            anyhit_entry,
                            &anyhit_set,
                            tail_block,
                            &skipped_tail_set,
                        )
                });
                let read_post_loop = scaff_vals.iter().any(|sv| {
                    self.uses[sv.0 as usize].iter().any(|&(ub, uidx)| {
                        !nodes.contains(&ub)
                            && uidx.is_some_and(|ui| {
                                let uinst = &self.p.blocks[ub].insts[ui];
                                let ops = operand_reads(self.p, uinst);
                                uinst.reads.iter().any(|&(r, rv)| rv == *sv && ops.contains(&r))
                            })
                    })
                });
                if !read_in_anyhit && !read_post_loop {
                    continue;
                }
                let idx = state_vars.len();
                state_vars.push(AnyHitStateVar {
                    init,
                    next,
                    reg: phi.reg,
                });
                for sv in scaff_vals {
                    state_val_to_idx.insert(sv, idx);
                }
            }
        }

        // 8. Classify every scaffolding SSA value read inside `anyhit_set`.
        let mut inputs: Vec<(Value, CandidateInput)> = Vec::new();
        let mut sorted_reads: Vec<Value> = scaffolding_reads.into_iter().collect();
        sorted_reads.sort_unstable();
        for v in sorted_reads {
            if cand_t_vals.contains(&v) {
                inputs.push((v, CandidateInput::RayT));
            } else if cand_bary_u_vals.contains(&v) {
                inputs.push((v, CandidateInput::Bary(0)));
            } else if cand_bary_v_vals.contains(&v) {
                inputs.push((v, CandidateInput::Bary(1)));
            } else if cand_ff_vals.contains(&v) {
                inputs.push((v, CandidateInput::FrontFace));
            } else if cur_tri_vals.contains(&v) || cur_inst_vals.contains(&v) {
                inputs.push((v, CandidateInput::One));
            } else if dead_addr_inputs.contains(&v) {
                inputs.push((v, CandidateInput::Zero));
            } else if let Some(&sidx) = state_val_to_idx.get(&v) {
                inputs.push((v, CandidateInput::StateVar(sidx)));
            } else if self.is_dead_implicit_guarded_read(
                v,
                anyhit_entry,
                &anyhit_set,
                tail_block,
                &skipped_tail_set,
            ) {
                // Scratch VGPR only touched as the implicit old-value of a guarded overwrite
                // whose result is never operand-read in lanes where the guard was false!
                // Check this BEFORE `is_unchanged_copy` so dead guarded reads don't create
                // spurious cross-loop dependencies on earlier loops' dead scratch VGPRs!
                inputs.push((v, CandidateInput::Zero));
            } else if let Some(pre) = self.is_unchanged_copy(v, nodes, entry_block, 0)
                && self
                    .def_block(pre)
                    .is_some_and(|db| !nodes.contains(&db) && self.dominates(db, entry_block))
            {
                inputs.push((v, CandidateInput::CopyOf(pre)));
            } else {
                // Unknown scaffolding value read inside any-hit: fail-safe reject!
                fail_ah!(
                    "step8 unknown scaffolding read {v:?} reg={:?} def={:?} state_regs={:?}",
                    self.value_reg(v),
                    self.defs.get(v.0 as usize),
                    state_vars.iter().map(|s| s.reg).collect::<Vec<_>>()
                );
            }
        }

        // 9. Verify every `Atom` referenced inside `anyhit_set` (guards, exits, phi_masks, masks)
        //    defined in `nodes \ anyhit_set` is provably constant on `reach[anyhit_entry]`.
        let Some(const_lane_atoms) = self.check_anyhit_atoms(
            &anyhit_set,
            anyhit_entry,
            tail_block,
            &skipped_tail_set,
            accept_mask,
            nodes,
        ) else {
            fail_ah!("step9 check_anyhit_atoms failed");
        };

        let mut body_blocks: Vec<usize> = anyhit_set.into_iter().collect();
        body_blocks.sort_by_key(|&b| self.rpo[b]);

        Some(TriangleHitInfo {
            hit_block,
            opacity_block: Some(opacity_block),
            accept_first_hit,
            t_min,
            committed_t_reg,
            committed_tri_reg,
            committed_inst_reg,
            committed_bary_u_reg,
            committed_bary_v_reg,
            committed_front_face_reg,
            mode: HitMode::AnyHit(AnyHitInfo {
                opacity_block,
                entry_block: anyhit_entry,
                tail_block,
                body_blocks,
                skipped_tail_insts,
                accept_mask,
                accept_invert,
                accept_first_hit,
                inputs,
                const_lane_atoms,
                state_vars,
                cand_meta_loads,
            }),
        })
    }

    /// Returns `true` if `root` is NEVER directly operand-read in `scope`, and for every
    /// guarded VGPR overwrite where `root` is an implicit old-value read, the surviving
    /// `!guard` lanes never reach any operand read inside `scope`.
    fn is_dead_implicit_guarded_read(
        &self,
        root: Value,
        anyhit_entry: usize,
        scope: &HashSet<usize>,
        tail_block: usize,
        skipped_tail_set: &HashSet<usize>,
    ) -> bool {
        let entry_reach = self.st.reach[anyhit_entry];
        let init_facts = if entry_reach == Bool::TRUE {
            Vec::new()
        } else {
            vec![(entry_reach, true)]
        };
        let mut seen: HashSet<(Value, Vec<(Bool, bool)>)> = HashSet::new();
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> = vec![(root, init_facts)];
        let mut steps = 0usize;
        while let Some((v, facts)) = work.pop() {
            steps += 1;
            if steps > 128 {
                return false;
            }
            if !seen.insert((v, facts.clone())) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if !phi.args.contains(&v) {
                            continue;
                        }
                        if let Some(margs) = self.st.phi_masks.get(&phi.value) {
                            for (&arg, &mb) in phi.args.iter().zip(margs) {
                                if arg != v {
                                    continue;
                                }
                                let mut next_facts = facts.clone();
                                if mb != Bool::TRUE && !next_facts.contains(&(mb, true)) {
                                    next_facts.push((mb, true));
                                }
                                if self.bools_meet(&next_facts) {
                                    work.push((phi.value, next_facts));
                                }
                            }
                        } else {
                            for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                                if arg != v {
                                    continue;
                                }
                                let next_facts = self.edge_facts(pred, b, &facts);
                                if self.bools_meet(&next_facts) {
                                    work.push((phi.value, next_facts));
                                }
                            }
                        }
                    }
                    continue;
                };
                if b == tail_block && skipped_tail_set.contains(&i) {
                    continue;
                }
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                let is_op_read = inst
                    .reads
                    .iter()
                    .any(|&(r, rv)| rv == v && ops.contains(&r));
                if is_op_read {
                    let mut check = facts.clone();
                    check.push((self.st.reach[b], true));
                    if let Some(g) = self.st.guards[b][i] {
                        check.push((g, true));
                    } else if inst.name().starts_with("v_cmp_")
                        && let Some(eg) = self.same_exec_scope_guard(b, i)
                    {
                        check.push((eg, true));
                    }
                    if self.bools_meet(&check) {
                        return false;
                    }
                    continue;
                }
                if let Some(g) = self.st.guards[b][i] {
                    let mut next_facts = facts.clone();
                    if !next_facts.contains(&(g, false)) {
                        next_facts.push((g, false));
                    }
                    if self.bools_meet(&next_facts) {
                        for &(r, rv) in &inst.reads {
                            if rv == v
                                && let Some(w) = inst.write(r)
                            {
                                work.push((w, next_facts.clone()));
                            }
                        }
                    }
                }
            }
        }
        true
    }

    fn collect_anyhit_scaffolding_reads(
        &self,
        anyhit_set: &HashSet<usize>,
        tail_block: usize,
        skipped_tail_set: &HashSet<usize>,
        nodes: &HashSet<usize>,
    ) -> HashSet<Value> {
        let mut out = HashSet::new();
        for &b in anyhit_set {
            for phi in &self.p.blocks[b].phis {
                for &arg in &phi.args {
                    if self
                        .def_block(arg)
                        .is_some_and(|db| nodes.contains(&db) && !anyhit_set.contains(&db))
                    {
                        out.insert(arg);
                    }
                }
            }
            for (i, inst) in self.p.blocks[b].insts.iter().enumerate() {
                if b == tail_block && skipped_tail_set.contains(&i) {
                    continue;
                }
                let ops = operand_reads(self.p, inst);
                let guarded = self.st.guards[b][i].is_some();
                for &(r, v) in &inst.reads {
                    let needed = match r {
                        Reg::S(c) if c == EXEC || c == EXEC + 1 => false,
                        Reg::V(_) => ops.contains(&r) || guarded,
                        _ => true,
                    };
                    if needed
                        && self
                            .def_block(v)
                            .is_some_and(|db| nodes.contains(&db) && !anyhit_set.contains(&db))
                    {
                        out.insert(v);
                    }
                }
            }
        }
        out
    }

    /// Recognises Template B's pre-computed `(cur_inst >> 3) << 6` lo/hi address words
    /// passed from the outer TLAS loop into the any-hit body (e.g. `s40:s41` in `035`).
    fn classify_scaffolding_inst_addr_val(
        &self,
        v: Value,
        cur_inst_regs: &HashSet<Reg>,
        nodes: &HashSet<usize>,
    ) -> Option<BvhTag> {
        let mut cur = v;
        for _ in 0..6 {
            let Some(def) = self.defs.get(cur.0 as usize).copied().flatten() else {
                return None;
            };
            match def {
                Def::Phi(b, pi) if nodes.contains(&b) => {
                    let phi = &self.p.blocks[b].phis[pi];
                    let next = phi.args.iter().copied().find(|&a| a != cur)?;
                    cur = next;
                }
                Def::Inst(b, i) if nodes.contains(&b) => {
                    let inst = &self.p.blocks[b].insts[i];
                    match inst.name() {
                        "s_mov_b32" | "v_mov_b32" => cur = single_reg_src(inst)?,
                        "s_lshl_b64" | "v_lshlrev_b64" => {
                            let is_shl6 = has_const_operand(inst, 6);
                            if !is_shl6 {
                                return None;
                            }
                            let is_hi = inst.writes.get(1).is_some_and(|&(_, w)| w == cur);
                            // Verify low input comes from `inst_node >> 3`.
                            let lo_in = inst.reads.first().map(|&(_, rv)| rv)?;
                            if let Some((_, ldi)) = self.def_inst(lo_in)
                                && matches!(ldi.name(), "s_lshr_b32" | "v_lshrrev_b32")
                                && has_const_operand(ldi, 3)
                                && other_val_from_const_op(ldi, 3).is_some_and(|nv| {
                                    self.value_reg(nv)
                                        .is_some_and(|r| cur_inst_regs.contains(&r))
                                })
                            {
                                return Some(if is_hi {
                                    BvhTag::AddrHi
                                } else {
                                    BvhTag::InstByteOff(0)
                                });
                            }
                            return None;
                        }
                        _ => return None,
                    }
                }
                _ => return None,
            }
        }
        None
    }

    /// Checks whether `phi` at `entry_block` is a loop-carried user state variable updated
    /// ONLY inside `anyhit_set` (untouched by BVH scaffolding).
    fn match_anyhit_state_var(
        &self,
        phi: &crate::ir::Phi,
        nodes: &HashSet<usize>,
        entry_block: usize,
        anyhit_set: &HashSet<usize>,
        tail_block: usize,
    ) -> Option<(RayInput, Value, HashSet<Value>)> {
        // Must have a unique pre-loop init dominating `entry_block`.
        let init = self.pre_loop_input(phi.value, nodes, entry_block, 0)?;
        if let RayInput::Val(iv) = init {
            let db = self.def_block(iv)?;
            if nodes.contains(&db) || !self.dominates(db, entry_block) {
                return None;
            }
        }

        // Trace backward from back-edges of `phi` through `nodes \ anyhit_set` (PHIs and `v_mov_b32`
        // copies only). Every path must end at EITHER `phi.value` OR a SINGLE `next` inside `anyhit_set`.
        let mut scaff_vals: HashSet<Value> = HashSet::new();
        scaff_vals.insert(phi.value);
        let mut visited: HashSet<Value> = HashSet::new();
        visited.insert(phi.value);
        let mut work: Vec<Value> = Vec::new();
        for (&pred, &arg) in self.p.blocks[entry_block].preds.iter().zip(&phi.args) {
            if nodes.contains(&pred) && arg != phi.value {
                work.push(arg);
            }
        }

        let mut next_val: Option<Value> = None;
        while let Some(v) = work.pop() {
            if !visited.insert(v) {
                continue;
            }
            let def = self.defs.get(v.0 as usize).copied().flatten()?;
            match def {
                Def::Input(_) => return None,
                Def::Phi(b, pi) => {
                    if anyhit_set.contains(&b) {
                        if let Some(prev) = next_val {
                            if prev != v {
                                return None;
                            }
                        } else {
                            next_val = Some(v);
                        }
                        continue;
                    }
                    if !nodes.contains(&b) {
                        return None;
                    }
                    scaff_vals.insert(v);
                    for &arg in &self.p.blocks[b].phis[pi].args {
                        if !visited.contains(&arg) {
                            work.push(arg);
                        }
                    }
                }
                Def::Inst(b, i) => {
                    if anyhit_set.contains(&b) {
                        if let Some(prev) = next_val {
                            if prev != v {
                                return None;
                            }
                        } else {
                            next_val = Some(v);
                        }
                        continue;
                    }
                    if !nodes.contains(&b) {
                        return None;
                    }
                    let inst = &self.p.blocks[b].insts[i];
                    if inst.name() == "v_mov_b32"
                        && let Some(src) = single_reg_src(inst)
                    {
                        scaff_vals.insert(v);
                        if !visited.contains(&src) {
                            work.push(src);
                        }
                        continue;
                    }
                    return None;
                }
            }
        }

        let next = next_val?;
        // `next` must dominate `tail_block` so we can read it at the end of `tail_block`.
        let next_blk = self.def_block(next)?;
        if !anyhit_set.contains(&next_blk) || !self.dominates(next_blk, tail_block) {
            return None;
        }
        // Also collect any forward PHI/copy of `phi.value` in `nodes \ anyhit_set` before `anyhit_set`.
        let fwd = self.forward_phi_closure(phi.value, nodes, Some(anyhit_set));
        for v in fwd {
            if self
                .def_block(v)
                .is_some_and(|db| nodes.contains(&db) && !anyhit_set.contains(&db))
            {
                scaff_vals.insert(v);
            }
        }
        Some((init, next, scaff_vals))
    }

    fn check_anyhit_atoms(
        &self,
        anyhit_set: &HashSet<usize>,
        anyhit_entry: usize,
        tail_block: usize,
        skipped_tail_set: &HashSet<usize>,
        accept_mask: Value,
        nodes: &HashSet<usize>,
    ) -> Option<Vec<(Value, bool)>> {
        let mut bools_to_check: Vec<Bool> = Vec::new();
        if let Some(&bm) = self.st.masks.get(&accept_mask) {
            bools_to_check.push(bm);
        }
        for &b in anyhit_set {
            if b != tail_block
                && let LaneExit::Branch { when, .. } = self.st.exits[b]
            {
                bools_to_check.push(when);
            }
            for phi in &self.p.blocks[b].phis {
                if let Some(margs) = self.st.phi_masks.get(&phi.value) {
                    bools_to_check.extend(margs.iter().copied());
                }
            }
            for (i, inst) in self.p.blocks[b].insts.iter().enumerate() {
                if b == tail_block && skipped_tail_set.contains(&i) {
                    continue;
                }
                if let Some(g) = self.st.guards[b][i] {
                    bools_to_check.push(g);
                }
                for &(_, v) in &inst.reads {
                    if let Some(&bm) = self.st.masks.get(&v) {
                        bools_to_check.push(bm);
                    }
                }
            }
        }

        let entry_reach = self.st.reach[anyhit_entry];
        // Find every atom (scaffolding or pre-loop) forced constant by `entry_reach` alone.
        let mut fixed_atoms: HashMap<Atom, bool> = HashMap::new();
        for atom in self.st.bools.support(entry_reach) {
            let can_f = bdd_can_reach_true_with(&self.st.bools, entry_reach, atom, false);
            let can_t = bdd_can_reach_true_with(&self.st.bools, entry_reach, atom, true);
            match (can_t, can_f) {
                (true, false) => {
                    fixed_atoms.insert(atom, true);
                }
                (false, true) => {
                    fixed_atoms.insert(atom, false);
                }
                _ => {}
            }
        }
        // Also collect every scaffolding `Atom::Lane(av)` forced constant via `st.masks` under `entry_reach`.
        let mut resolved: HashMap<Value, bool> = HashMap::new();
        let mut pending_scaff_lane: Vec<(Bool, Value)> = Vec::new();

        for &bm in &bools_to_check {
            for atom in self.st.bools.support(bm) {
                let Some(av) = atom_value(atom) else {
                    continue;
                };
                if !self
                    .def_block(av)
                    .is_some_and(|db| nodes.contains(&db) && !anyhit_set.contains(&db))
                {
                    continue;
                }
                let Atom::Lane(_) = atom else {
                    return None;
                };
                if let Some(&val) = fixed_atoms.get(&atom) {
                    resolved.insert(av, val);
                    continue;
                }
                if resolved.contains_key(&av) {
                    continue;
                }
                let Some(&atom_bool) = self.st.masks.get(&av) else {
                    return None;
                };
                let can_be_false = self.bools_meet(&[(entry_reach, true), (atom_bool, false)]);
                let can_be_true = self.bools_meet(&[(entry_reach, true), (atom_bool, true)]);
                match (can_be_true, can_be_false) {
                    (true, false) => {
                        fixed_atoms.insert(atom, true);
                        resolved.insert(av, true);
                    }
                    (false, true) => {
                        fixed_atoms.insert(atom, false);
                        resolved.insert(av, false);
                    }
                    _ => {
                        pending_scaff_lane.push((bm, av));
                    }
                }
            }
        }
        // Any remaining scaffolding `Atom::Lane(av)` is allowed ONLY if it is unreachable in `bm`
        // once `fixed_atoms` are substituted (e.g. old `vcc_lo` preserved in `!reach[tail]` lanes).
        for (bm, av) in pending_scaff_lane {
            if resolved.contains_key(&av) {
                continue;
            }
            let live_support = bdd_support_under_fixed(&self.st.bools, bm, &fixed_atoms);
            if !live_support.contains(&Atom::Lane(av)) {
                resolved.insert(av, false);
                continue;
            }
            return None;
        }
        let mut out: Vec<(Value, bool)> = resolved.into_iter().collect();
        out.sort_unstable_by_key(|&(v, _)| v);
        Some(out)
    }

    fn pre_loop_const(&self, v: Value, nodes: &HashSet<usize>, entry_block: usize) -> Option<u32> {
        let pre = self.pre_loop_value(v, nodes, entry_block, 0)?;
        self.eval_const(pre, 0)
    }

    /// Evaluates compile-time constant integer values (including `s_brev_b32`, `s_movk_i32`, etc.).
    fn eval_const(&self, v: Value, depth: u32) -> Option<u32> {
        if depth > 8 {
            return None;
        }
        let (_, inst) = self.def_inst(v)?;
        match inst.name() {
            "s_mov_b32" | "v_mov_b32" => {
                let src = first_src_operand(inst)?;
                if let Some(k) = const_operand(src) {
                    return Some(k);
                }
                operand_val(inst, src).and_then(|x| self.eval_const(x, depth + 1))
            }
            "s_movk_i32" => {
                let Fields::Sop { imm: Some(imm), .. } = &inst.fields else {
                    return None;
                };
                Some(*imm as i16 as i32 as u32)
            }
            "s_brev_b32" => {
                let src = first_src_operand(inst)?;
                let k = const_operand(src)
                    .or_else(|| operand_val(inst, src).and_then(|x| self.eval_const(x, depth + 1)))?;
                Some(k.reverse_bits())
            }
            "s_not_b32" | "v_not_b32" => {
                let src = first_src_operand(inst)?;
                let k = const_operand(src)
                    .or_else(|| operand_val(inst, src).and_then(|x| self.eval_const(x, depth + 1)))?;
                Some(!k)
            }
            _ => None,
        }
    }

    /// Extracts `RAY_FLAG_*` from the loop's opacity, face-culling, accept-first-hit, and procedural checks.
    fn extract_ray_flags(
        &self,
        nodes: &HashSet<usize>,
        tri_hit: &TriangleHitInfo,
    ) -> Option<RayInput> {
        let anyhit_set: HashSet<usize> = match &tri_hit.mode {
            HitMode::AnyHit(ah) => ah.body_blocks.iter().copied().collect(),
            _ => HashSet::new(),
        };

        let mut flags = ray_flag::FORCE_OPAQUE;
        if tri_hit.accept_first_hit {
            flags |= ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH;
        }

        // Opacity mode:
        // - No `v_bfe_u32 _, flags, 4, 2` in scaffolding -> `RAY_FLAG_FORCE_OPAQUE`.
        // - In a block strictly dominating `hit_block` gating `hit_block` -> `RAY_FLAG_CULL_NON_OPAQUE`.
        // - In `tri_hit.opacity_block` (FrontBackTrack / AnyHit) -> decode bits 5:4 from instance-entry
        //   `v_bfi_b32 flags, 60, (packed << 2), _`:
        //     - `packed` ORs constant `8` -> `RAY_FLAG_FORCE_NON_OPAQUE`
        //     - `packed` ORs constant `4` -> `RAY_FLAG_FORCE_OPAQUE`
        //     - `packed` ORs `(inst_word1 & 12)` -> `RAY_FLAG_NONE`
        let mut opacity_flag_override: Option<u32> = None;
        for &b in nodes {
            if anyhit_set.contains(&b) {
                continue;
            }
            for inst in &self.p.blocks[b].insts {
                if inst.name() != "v_bfe_u32" {
                    continue;
                }
                let Fields::Valu(f) = &inst.fields else {
                    continue;
                };
                if f.src.len() < 3
                    || const_operand(&f.src[1]) != Some(4)
                    || const_operand(&f.src[2]) != Some(2)
                {
                    continue;
                }
                let Some(&(_, bfe_val)) = inst.writes.first() else {
                    return None;
                };
                if Some(b) == tri_hit.opacity_block {
                    let op_flag = self.decode_instance_opacity_flag(nodes, &anyhit_set)?;
                    if let Some(prev) = opacity_flag_override
                        && prev != op_flag
                    {
                        return None;
                    }
                    opacity_flag_override = Some(op_flag);
                    continue;
                }
                // Must dominate `hit_block` (strictly) and gate `hit_block` on `((tri_w0 & 1) | inst_op) == 1`.
                if b == tri_hit.hit_block || !self.dominates(b, tri_hit.hit_block) {
                    return None;
                }
                let mut op_val = None;
                for &(ub, ui) in &self.uses[bfe_val.0 as usize] {
                    if ub != b {
                        continue;
                    }
                    let Some(idx) = ui else { continue };
                    let uinst = &self.p.blocks[ub].insts[idx];
                    if uinst.name() == "v_and_or_b32"
                        && let Fields::Valu(uf) = &uinst.fields
                        && uf.src.len() >= 3
                        && const_operand(&uf.src[1]) == Some(1)
                        && operand_val(uinst, &uf.src[2]) == Some(bfe_val)
                    {
                        op_val = uinst.writes.first().map(|&(_, w)| w);
                    }
                }
                let op_val = op_val?;
                let mut gated_opaque = false;
                for &(ub, ui) in &self.uses[op_val.0 as usize] {
                    if ub != b {
                        continue;
                    }
                    let Some(idx) = ui else { continue };
                    let uinst = &self.p.blocks[ub].insts[idx];
                    if uinst.name() == "v_cmp_eq_u32"
                        && has_const_operand(uinst, 1)
                        && let Some(&(_, mask_val)) = uinst.writes.first()
                    {
                        for &(mb, mi) in &self.uses[mask_val.0 as usize] {
                            if mb != b {
                                continue;
                            }
                            let Some(midx) = mi else { continue };
                            let minst = &self.p.blocks[mb].insts[midx];
                            if matches!(minst.name(), "s_and_saveexec_b32" | "s_and_saveexec_b64") {
                                gated_opaque = true;
                            }
                        }
                    }
                }
                if !gated_opaque {
                    return None;
                }
                opacity_flag_override = Some(ray_flag::CULL_NON_OPAQUE);
            }
        }
        if let Some(op_flag) = opacity_flag_override {
            flags = (flags & !ray_flag::FORCE_OPAQUE) | op_flag;
        }

        // Face culling: scan only BVH scaffolding blocks (never inside preserved any-hit body).
        let mut cull_flag: Option<u32> = None;
        let mut skip_procedural = false;

        for &b in nodes {
            if anyhit_set.contains(&b) {
                continue;
            }
            for inst in &self.p.blocks[b].insts {
                if matches!(inst.name(), "v_cmp_ne_u32" | "v_cmpx_ne_u32")
                    && let Fields::Valu(f) = &inst.fields
                    && f.src.len() >= 2
                    && (const_operand(&f.src[0]) == Some(7) || const_operand(&f.src[1]) == Some(7))
                {
                    skip_procedural = true;
                }
                if inst.name() != "v_mul_i32_i24" || b == tri_hit.hit_block {
                    continue;
                }
                let Fields::Valu(f) = &inst.fields else {
                    continue;
                };
                if f.sdwa.is_some() || f.src.len() < 2 {
                    continue;
                }
                let k = const_operand(&f.src[0]).or_else(|| const_operand(&f.src[1]));
                let Some(k) = k else {
                    let is_sign_norm = f.src[..2].iter().any(|op| {
                        operand_val(inst, op)
                            .and_then(|v| self.def_inst(v))
                            .is_some_and(|(_, di)| {
                                matches!(di.name(), "v_and_b32" | "s_and_b32")
                                    && has_const_operand(di, 1)
                            })
                    });
                    if !is_sign_norm {
                        return None;
                    }
                    continue;
                };
                let mapped = match k {
                    0 => 0,
                    1 => ray_flag::CULL_BACK_FACING_TRIANGLES,
                    0xFFFF_FFFF => ray_flag::CULL_FRONT_FACING_TRIANGLES,
                    _ => return None,
                };
                if let Some(prev) = cull_flag
                    && prev != mapped
                {
                    return None;
                }
                cull_flag = Some(mapped);
            }
        }

        flags |= cull_flag.unwrap_or(0);
        if skip_procedural {
            flags |= ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        }
        Some(RayInput::Const(flags))
    }

    /// Decodes the opacity flag written into bits `5:4` via `v_bfi_b32 flags, 60, (packed << 2), _`.
    fn decode_instance_opacity_flag(
        &self,
        nodes: &HashSet<usize>,
        anyhit_set: &HashSet<usize>,
    ) -> Option<u32> {
        let mut found: Option<u32> = None;
        for &b in nodes {
            if anyhit_set.contains(&b) {
                continue;
            }
            for inst in &self.p.blocks[b].insts {
                if inst.name() != "v_bfi_b32" {
                    continue;
                }
                let Fields::Valu(f) = &inst.fields else {
                    continue;
                };
                if f.src.len() < 3 || const_operand(&f.src[0]) != Some(60) {
                    continue;
                }
                let shifted_v = operand_val(inst, &f.src[1])?;
                let (_, sh_inst) = self.def_inst(shifted_v)?;
                if sh_inst.name() != "v_lshlrev_b32" || !has_const_operand(sh_inst, 2) {
                    return None;
                }
                let packed_v = other_val_from_const_op(sh_inst, 2)?;
                let (_, pk_inst) = self.def_inst(packed_v)?;
                let op_flag = match pk_inst.name() {
                    "v_and_or_b32" => {
                        let Fields::Valu(pf) = &pk_inst.fields else {
                            return None;
                        };
                        if pf.src.len() < 3 || const_operand(&pf.src[1]) != Some(3) {
                            return None;
                        }
                        if let Some(k) = const_operand(&pf.src[2]) {
                            match k {
                                8 => ray_flag::FORCE_NON_OPAQUE,
                                4 => ray_flag::FORCE_OPAQUE,
                                _ => return None,
                            }
                        } else if let Some(ov) = operand_val(pk_inst, &pf.src[2])
                            && self.is_inst_opacity_mask_12(ov)
                        {
                            ray_flag::NONE
                        } else {
                            return None;
                        }
                    }
                    "v_or_b32" => {
                        if let Some(k) = other_const_arg_any(pk_inst) {
                            match k {
                                8 => ray_flag::FORCE_NON_OPAQUE,
                                4 => ray_flag::FORCE_OPAQUE,
                                _ => return None,
                            }
                        } else if pk_inst
                            .reads
                            .iter()
                            .any(|&(_, rv)| self.is_inst_opacity_mask_12(rv))
                        {
                            ray_flag::NONE
                        } else {
                            return None;
                        }
                    }
                    _ => return None,
                };
                if let Some(prev) = found
                    && prev != op_flag
                {
                    return None;
                }
                found = Some(op_flag);
            }
        }
        found
    }

    fn is_inst_opacity_mask_12(&self, v: Value) -> bool {
        let Some((_, di)) = self.def_inst(v) else {
            return false;
        };
        if !matches!(di.name(), "v_and_b32" | "s_and_b32") {
            return false;
        }
        if has_const_operand(di, 12) {
            return true;
        }
        let Some((a, b)) = two_src_operands(di) else {
            return false;
        };
        operand_val(di, a).and_then(|x| self.eval_const(x, 0)) == Some(12)
            || operand_val(di, b).and_then(|x| self.eval_const(x, 0)) == Some(12)
    }

    /// Extracts the 8-bit `InstanceInclusionMask` tested against `inst_word0` (bits `31:24`).
    fn extract_instance_mask(
        &self,
        template: RayTemplate,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<RayInput> {
        let mut word0_candidates: Vec<Value> = Vec::new();
        for &b in nodes {
            for inst in &self.p.blocks[b].insts {
                let n = inst.name();
                match template {
                    RayTemplate::Bvh64 if n == "global_load_dwordx4" => {
                        let Fields::Flat {
                            addr: Some(Operand::Vector { reg, .. }),
                            offset,
                            ..
                        } = &inst.fields
                        else {
                            continue;
                        };
                        let is_off48 = *offset == 48
                            || inst
                                .read(Reg::V(*reg))
                                .is_some_and(|addr_lo| self.addr_has_offset_48(addr_lo, 0));
                        if is_off48 && let Some(&(_, w0)) = inst.writes.first() {
                            word0_candidates.push(w0);
                        }
                    }
                    RayTemplate::Bvh32SharedTlas if n == "s_load_dwordx4" => {
                        let Fields::Smem {
                            base: Operand::Scalar { code, .. },
                            offset,
                            soffset,
                            ..
                        } = &inst.fields
                        else {
                            continue;
                        };
                        let off = *offset + const_operand(soffset).unwrap_or(0) as i32;
                        let is_off48 = off == 48
                            || (off == 0
                                && inst
                                    .read(Reg::S(*code))
                                    .is_some_and(|base_lo| self.addr_has_offset_48(base_lo, 0)));
                        if is_off48 && let Some(&(_, w0)) = inst.writes.first() {
                            word0_candidates.push(w0);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut found: Option<RayInput> = None;
        for w0 in word0_candidates {
            if let Some(mask) = self.decode_instance_mask_from_word0(w0, nodes, entry_block) {
                if let Some(prev) = found {
                    if prev != mask {
                        return None;
                    }
                } else {
                    found = Some(mask);
                }
            }
        }
        found
    }

    fn addr_has_offset_48(&self, v: Value, depth: u32) -> bool {
        if depth > 8 {
            return false;
        }
        let Some((_, inst)) = self.def_inst(v) else {
            return false;
        };
        if matches!(
            inst.name(),
            "s_mov_b32" | "v_mov_b32" | "v_readfirstlane_b32" | "v_readlane_b32"
        ) {
            return single_reg_src(inst).is_some_and(|s| self.addr_has_offset_48(s, depth + 1));
        }
        if inst.name() == "v_cndmask_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.src.len() >= 2
        {
            return f.src[..2].iter().any(|op| {
                operand_val(inst, op).is_some_and(|s| self.addr_has_offset_48(s, depth + 1))
            });
        }
        if matches!(inst.name(), "s_cselect_b32" | "s_cselect_b64")
            && let Fields::Sop { src, .. } = &inst.fields
            && src.len() >= 2
        {
            return src[..2].iter().any(|op| {
                operand_val(inst, op).is_some_and(|s| self.addr_has_offset_48(s, depth + 1))
            });
        }
        matches!(
            inst.name(),
            "v_or_b32" | "v_add_nc_u32" | "v_add_co_u32" | "s_or_b32" | "s_add_u32"
        ) && has_const_operand(inst, 48)
    }

    fn decode_instance_mask_from_word0(
        &self,
        w0: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<RayInput> {
        // BFS through simple copies and join PHIs of `w0` within `nodes`.
        let mut seen = HashSet::new();
        let mut work = vec![w0];
        while let Some(cur) = work.pop() {
            if !seen.insert(cur) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[cur.0 as usize] {
                if !nodes.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if phi.args.contains(&cur) {
                            work.push(phi.value);
                        }
                    }
                    continue;
                };
                let inst = &self.p.blocks[b].insts[i];
                let n = inst.name();
                if matches!(n, "s_mov_b32" | "v_mov_b32") {
                    if let Some(&(_, dst)) = inst.writes.first() {
                        work.push(dst);
                    }
                    continue;
                }
                if let Some(mask) = self.match_mask_op(inst, cur, nodes, entry_block) {
                    return Some(mask);
                }
            }
        }
        None
    }

    fn match_mask_op(
        &self,
        inst: &Inst,
        w0: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<RayInput> {
        let n = inst.name();
        match n {
            // 1. `and(w0, mask_shifted)`
            "v_and_b32" | "s_and_b32" => {
                if let Some(k) = other_const_arg(inst, w0) {
                    let m = k >> 24;
                    return (m != 0).then_some(RayInput::Const(m));
                }
                let other_v = other_val_arg(inst, w0)?;
                if let Some(k) = self.eval_const(other_v, 0) {
                    let m = k >> 24;
                    return (m != 0).then_some(RayInput::Const(m));
                }
                // Check if `other_v` is `shl(raw_mask, 24)` defined outside the loop.
                if let Some(raw) = self.unshifted_mask_24(other_v, nodes, entry_block) {
                    return Some(RayInput::Val(raw));
                }
                let pre = self.pre_loop_value(other_v, nodes, entry_block, 0)?;
                let db = self.def_block(pre)?;
                (!nodes.contains(&db) && self.dominates(db, entry_block))
                    .then_some(RayInput::ShiftedMask(pre))
            }
            // 2. `lshr(w0, shift)` with `shift in 24..=31`
            "v_lshrrev_b32" | "s_lshr_b32" => {
                let shift = other_const_arg(inst, w0)?;
                if !(24..=31).contains(&shift) {
                    return None;
                }
                let dst = inst.writes.first()?.1;
                // Check whether `dst` is ANDed with 1 / (kind!=7 ? 1 : 0) or a constant mask!
                for &(ub, uidx) in &self.uses[dst.0 as usize] {
                    if !nodes.contains(&ub) {
                        continue;
                    }
                    let Some(ui) = uidx else { continue };
                    let uinst = &self.p.blocks[ub].insts[ui];
                    if matches!(uinst.name(), "v_and_b32" | "s_and_b32") {
                        if let Some(k) = other_const_arg(uinst, dst) {
                            let m = (k << (shift - 24)) & 0xFF;
                            return (m != 0).then_some(RayInput::Const(m));
                        }
                        if let Some(ov) = other_val_arg(uinst, dst)
                            && self.is_bool_0_or_1(ov)
                        {
                            // `(w0 >> shift) & bool01 != 0` tests ONLY bit `shift`!
                            return Some(RayInput::Const(1u32 << (shift - 24)));
                        }
                    }
                }
                // Direct `(w0 >> shift) != 0` tests bits `31..=shift`.
                let bits = 0xFFu32 & !((1u32 << (shift - 24)) - 1);
                (bits != 0).then_some(RayInput::Const(bits))
            }
            // 3. `bfe_u32(w0, shift, width)`
            "v_bfe_u32" => {
                let Fields::Valu(f) = &inst.fields else {
                    return None;
                };
                if f.src.len() < 3 || operand_val(inst, &f.src[0]) != Some(w0) {
                    return None;
                }
                let shift = const_operand(&f.src[1])?;
                let width = const_operand(&f.src[2])?;
                if !(24..=31).contains(&shift) || width == 0 {
                    return None;
                }
                let m = (((1u32 << width.min(8)) - 1) << (shift - 24)) & 0xFF;
                (m != 0).then_some(RayInput::Const(m))
            }
            "s_bfe_u32" => {
                let Fields::Sop { src, .. } = &inst.fields else {
                    return None;
                };
                if src.len() < 2 || operand_val(inst, &src[0]) != Some(w0) {
                    return None;
                }
                let packed = const_operand(&src[1])
                    .or_else(|| operand_val(inst, &src[1]).and_then(|x| self.eval_const(x, 0)))?;
                let shift = packed & 0x1F;
                let width = (packed >> 16) & 0x7F;
                if !(24..=31).contains(&shift) || width == 0 {
                    return None;
                }
                let m = (((1u32 << width.min(8)) - 1) << (shift - 24)) & 0xFF;
                (m != 0).then_some(RayInput::Const(m))
            }
            // 4. Unsigned threshold compares: `w0 > ((1 << k) - 1)` or `w0 >= (1 << k)`
            "v_cmp_lt_u32" | "v_cmpx_lt_u32" | "s_cmp_lt_u32"
            | "v_cmp_gt_u32" | "v_cmpx_gt_u32" | "s_cmp_gt_u32" => {
                let (lhs, rhs) = two_src_operands(inst)?;
                let is_gt = n.contains("_gt_");
                // We want `w0 > K`, which is `gt(w0, K)` or `lt(K, w0)`.
                let k_op = if is_gt && operand_val(inst, lhs) == Some(w0) {
                    rhs
                } else if !is_gt && operand_val(inst, rhs) == Some(w0) {
                    lhs
                } else {
                    return None;
                };
                let k = const_operand(k_op)
                    .or_else(|| operand_val(inst, k_op).and_then(|x| self.eval_const(x, 0)))?;
                decode_gt_threshold_mask(k).map(RayInput::Const)
            }
            "v_cmp_ge_u32" | "v_cmpx_ge_u32" | "s_cmp_ge_u32"
            | "v_cmp_le_u32" | "v_cmpx_le_u32" | "s_cmp_le_u32" => {
                let (lhs, rhs) = two_src_operands(inst)?;
                let is_ge = n.contains("_ge_");
                // We want `w0 >= K`, which is `ge(w0, K)` or `le(K, w0)`.
                let k_op = if is_ge && operand_val(inst, lhs) == Some(w0) {
                    rhs
                } else if !is_ge && operand_val(inst, rhs) == Some(w0) {
                    lhs
                } else {
                    return None;
                };
                let k = const_operand(k_op)
                    .or_else(|| operand_val(inst, k_op).and_then(|x| self.eval_const(x, 0)))?;
                if k == 0 {
                    return None;
                }
                decode_gt_threshold_mask(k - 1).map(RayInput::Const)
            }
            // 5. Signed sign-bit compare: `w0 < 0` -> bit 31 (`0x80`)
            "v_cmp_lt_i32" | "v_cmpx_lt_i32" | "s_cmp_lt_i32"
            | "v_cmp_gt_i32" | "v_cmpx_gt_i32" | "s_cmp_gt_i32" => {
                let (lhs, rhs) = two_src_operands(inst)?;
                let is_lt = n.contains("_lt_");
                let zero_op = if is_lt && operand_val(inst, lhs) == Some(w0) {
                    rhs
                } else if !is_lt && operand_val(inst, rhs) == Some(w0) {
                    lhs
                } else {
                    return None;
                };
                (const_operand(zero_op) == Some(0)).then_some(RayInput::Const(0x80))
            }
            // 6. Scalar bit test: `s_bitcmp1_b32 w0, k`
            "s_bitcmp1_b32" => {
                let (lhs, rhs) = two_src_operands(inst)?;
                if operand_val(inst, lhs) != Some(w0) {
                    return None;
                }
                let k = const_operand(rhs)?;
                (24..=31).contains(&k).then(|| RayInput::Const(1u32 << (k - 24)))
            }
            _ => None,
        }
    }

    fn unshifted_mask_24(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
    ) -> Option<Value> {
        let mut cur = v;
        for _ in 0..4 {
            if let Some((_, inst)) = self.def_inst(cur) {
                if matches!(inst.name(), "v_mov_b32" | "s_mov_b32")
                    && let Some(src) = single_reg_src(inst)
                {
                    cur = src;
                    continue;
                }
                if matches!(inst.name(), "v_lshlrev_b32" | "s_lshl_b32") {
                    let (lhs, rhs) = two_src_operands(inst)?;
                    let raw = if inst.name() == "v_lshlrev_b32" && const_operand(lhs) == Some(24) {
                        operand_val(inst, rhs)?
                    } else if inst.name() == "s_lshl_b32" && const_operand(rhs) == Some(24) {
                        operand_val(inst, lhs)?
                    } else {
                        return None;
                    };
                    let pre_raw = self.pre_loop_value(raw, nodes, entry_block, 0)?;
                    let db = self.def_block(pre_raw)?;
                    if !nodes.contains(&db) && self.dominates(db, entry_block) {
                        return Some(pre_raw);
                    }
                    return None;
                }
            }
            break;
        }
        let pre = self.pre_loop_value(v, nodes, entry_block, 0)?;
        let (_, inst) = self.def_inst(pre)?;
        if matches!(inst.name(), "v_lshlrev_b32" | "s_lshl_b32") {
            let (lhs, rhs) = two_src_operands(inst)?;
            let raw = if inst.name() == "v_lshlrev_b32" && const_operand(lhs) == Some(24) {
                operand_val(inst, rhs)?
            } else if inst.name() == "s_lshl_b32" && const_operand(rhs) == Some(24) {
                operand_val(inst, lhs)?
            } else {
                return None;
            };
            let db = self.def_block(raw)?;
            if !nodes.contains(&db) && self.dominates(db, entry_block) {
                return Some(raw);
            }
        }
        None
    }

    fn is_bool_0_or_1(&self, v: Value) -> bool {
        let Some((_, inst)) = self.def_inst(v) else {
            return false;
        };
        if inst.name() == "v_cndmask_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.src.len() >= 2
        {
            let c0 = const_operand(&f.src[0]);
            let c1 = const_operand(&f.src[1]);
            return matches!((c0, c1), (Some(0), Some(1)) | (Some(1), Some(0)));
        }
        false
    }

    /// Computes every SSA `Value` defined inside `loop_all` that is truly live after the
    /// loop (ignoring EXEC masks and unguarded VGPR overwrite old-values), maps each
    /// unambiguously to a [`RayOutput`], and collects verified post-loop BVH metadata loads.
    /// Returns `None` if ANY live-out is unclassified or any BVH pointer escapes.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn classify_live_outs(
        &self,
        loop_all: &HashSet<usize>,
        nodes: &HashSet<usize>,
        entry_block: usize,
        exit_block: usize,
        exit_pred: usize,
        t_max: RayInput,
        tri_hit: &TriangleHitInfo,
        lifted: &[RayLoop],
        lifted_blocks: &HashSet<usize>,
        claimed_committed_meta: &HashMap<Site, (RayMetaKind, usize)>,
        claimed_cand_meta: &HashSet<Site>,
    ) -> Option<(Vec<RayOutputBinding>, Vec<(Site, RayMetaKind)>)> {
        // 1. Seed true live values from all instructions & structured conditions outside `loop_all`
        //    and outside other already-lifted ray loops (whose real inputs are seeded explicitly).
        let mut live: HashSet<Value> = HashSet::new();
        let mut work: Vec<Value> = Vec::new();
        let mut mark = |v: Value| {
            if live.insert(v) {
                work.push(v);
            }
        };
        let mark_input = |inp: RayInput, mark: &mut dyn FnMut(Value)| {
            if let RayInput::Val(v) | RayInput::ShiftedMask(v) = inp {
                mark(v);
            }
        };
        let mark_bool = |b: Bool, mark: &mut dyn FnMut(Value)| {
            for atom in self.st.bools.support(b) {
                if let Some(v) = atom_value(atom) {
                    mark(v);
                }
            }
        };

        for other in lifted {
            for inp in other.origin {
                mark_input(inp, &mut mark);
            }
            if let Some(inp) = other.t_min {
                mark_input(inp, &mut mark);
            }
            for inp in other.dir {
                mark_input(inp, &mut mark);
            }
            mark_input(other.t_max, &mut mark);
            mark_input(other.instance_mask, &mut mark);
            mark_input(other.ray_flags, &mut mark);
            if let RayLoopKind::FrontBackTrack { init_min_front_t } = other.kind {
                mark_input(init_min_front_t, &mut mark);
            }
            if let RayLoopKind::AnyHit(ah) = &other.kind {
                for sv in &ah.state_vars {
                    mark_input(sv.init, &mut mark);
                }
                for &(_, ci) in &ah.inputs {
                    if let CandidateInput::CopyOf(pre) = ci {
                        mark(pre);
                    }
                }
            }
            if let Some(mv) = other.active_ray_mask {
                mark(mv);
                if let Some(&bm) = self.st.masks.get(&mv) {
                    mark_bool(bm, &mut mark);
                }
            }
            for out in &other.outputs {
                if let Some(fb) = out.fallback {
                    mark(fb);
                }
                if let RayOutput::CopyOf(pre) = out.output {
                    mark(pre);
                }
            }
        }

        // Also include preserved `AnyHit::body_blocks` of other lifted loops when scanning live reads!
        let mut other_skipped_blocks: HashSet<usize> = HashSet::new();
        for other in lifted {
            other_skipped_blocks.insert(other.entry_block);
            other_skipped_blocks.extend(other.skipped_blocks.iter().copied());
        }

        for &b in &self.order {
            if loop_all.contains(&b) || other_skipped_blocks.contains(&b) {
                continue;
            }
            if let LaneExit::Branch { when, .. } = self.st.exits[b] {
                mark_bool(when, &mut mark);
            }
            for (i, inst) in self.p.blocks[b].insts.iter().enumerate() {
                if let Some(g) = self.st.guards[b][i] {
                    mark_bool(g, &mut mark);
                }
                let ops = operand_reads(self.p, inst);
                let guarded = self.st.guards[b][i].is_some();
                for &(r, v) in &inst.reads {
                    let needed = match r {
                        Reg::S(c) if c == EXEC || c == EXEC + 1 => false,
                        Reg::V(_) => ops.contains(&r) || guarded,
                        _ => true,
                    };
                    if needed {
                        mark(v);
                    }
                }
            }
        }

        // Propagate liveness backward through PHIs outside `loop_all` and `other_skipped_blocks`.
        while let Some(v) = work.pop() {
            if let Some(Def::Phi(b, pi)) = self.defs.get(v.0 as usize).copied().flatten()
                && !loop_all.contains(&b)
                && !other_skipped_blocks.contains(&b)
            {
                let phi = &self.p.blocks[b].phis[pi];
                for &arg in &phi.args {
                    if live.insert(arg) {
                        work.push(arg);
                    }
                }
                if let Some(mask_args) = self.st.phi_masks.get(&phi.value) {
                    for &mb in mask_args {
                        for atom in self.st.bools.support(mb) {
                            if let Some(av) = atom_value(atom)
                                && live.insert(av)
                            {
                                work.push(av);
                            }
                        }
                    }
                }
            }
        }

        // 2. Check all live PHIs at `exit_block` with multiple loop-exit edges: every
        //    loop-exit incoming edge must classify identically!
        let exit_pred_pos = self.p.blocks[exit_block]
            .preds
            .iter()
            .position(|&q| q == exit_pred)?;
        let mut required: HashSet<Value> = HashSet::new();

        for phi in &self.p.blocks[exit_block].phis {
            if !live.contains(&phi.value) {
                continue;
            }
            if let Some(mask_args) = self.st.phi_masks.get(&phi.value) {
                for (&q, &mb) in self.p.blocks[exit_block].preds.iter().zip(mask_args) {
                    if !loop_all.contains(&q) {
                        continue;
                    }
                    for atom in self.st.bools.support(mb) {
                        if let Some(av) = atom_value(atom)
                            && self.def_block(av).is_some_and(|db| loop_all.contains(&db))
                        {
                            return None;
                        }
                    }
                }
            }
            let canonical_arg = phi.args[exit_pred_pos];
            let canonical_cls = if self
                .def_block(canonical_arg)
                .is_some_and(|db| loop_all.contains(&db))
            {
                required.insert(canonical_arg);
                Some(self.classify_value(
                    canonical_arg,
                    nodes,
                    loop_all,
                    entry_block,
                    t_max,
                    tri_hit,
                    lifted,
                    &other_skipped_blocks,
                )?)
            } else {
                None
            };
            for (&q, &arg) in self.p.blocks[exit_block].preds.iter().zip(&phi.args) {
                if !loop_all.contains(&q) || q == exit_pred {
                    continue;
                }
                if self.def_block(arg).is_some_and(|db| loop_all.contains(&db)) {
                    let other_cls = self.classify_value(
                        arg,
                        nodes,
                        loop_all,
                        entry_block,
                        t_max,
                        tri_hit,
                        lifted,
                        &other_skipped_blocks,
                    )?;
                    let canon = canonical_cls?;
                    if other_cls.output != canon.output {
                        return None;
                    }
                } else if canonical_cls.is_some() && canonical_arg != arg {
                    return None;
                }
            }
        }

        // 3. Also include every direct non-PHI live value defined inside `loop_all`.
        for &v in &live {
            if !self.def_block(v).is_some_and(|db| loop_all.contains(&db)) {
                continue;
            }
            let used_by_other_lifted = lifted.iter().any(|other| ray_loop_reads_value(other, v));
            let used_directly_or_on_exit = used_by_other_lifted
                || self.uses[v.0 as usize].iter().any(|&(ub, uidx)| {
                    if loop_all.contains(&ub) || other_skipped_blocks.contains(&ub) {
                        return false;
                    }
                    if uidx.is_some() {
                        return true;
                    }
                    if ub == exit_block {
                        return self.p.blocks[ub].phis.iter().any(|phi| {
                            live.contains(&phi.value) && phi.args[exit_pred_pos] == v
                        });
                    }
                    true
                });
            if used_directly_or_on_exit {
                required.insert(v);
            }
        }

        // 4. Classify every required live-out value and collect committed tri/inst pointer roots
        //    for post-loop BVH metadata analysis.
        let mut sorted: Vec<Value> = required.into_iter().collect();
        sorted.sort_unstable();
        let mut bindings = Vec::with_capacity(sorted.len());
        let mut post_tri_roots: Vec<Value> = Vec::new();
        let mut post_inst_roots: Vec<Value> = Vec::new();

        for v in sorted {
            if self.st.masks.contains_key(&v) || self.st.highs.contains_key(&v) {
                let pre = self.pre_loop_value(v, nodes, entry_block, 0)?;
                bindings.push(RayOutputBinding {
                    value: v,
                    output: RayOutput::CopyOf(pre),
                    fallback: Some(pre),
                });
                continue;
            }
            let binding = self.classify_value(
                v,
                nodes,
                loop_all,
                entry_block,
                t_max,
                tri_hit,
                lifted,
                &other_skipped_blocks,
            )?;
            if binding.output == RayOutput::CommittedStatusTriNode {
                let reg = self.value_reg(v);
                if reg == tri_hit.committed_inst_reg && tri_hit.committed_inst_reg != tri_hit.committed_tri_reg {
                    post_inst_roots.push(v);
                } else {
                    post_tri_roots.push(v);
                }
            }
            bindings.push(binding);
        }

        // 5. Verify ALL post-loop uses of `committed_tri_node` and `committed_inst_node` are either
        //    status tests or verified BVH metadata loads (`committed_meta_loads`).
        let mut post_scope: HashSet<usize> = HashSet::new();
        for &b in &self.order {
            if !loop_all.contains(&b) && !other_skipped_blocks.contains(&b) {
                post_scope.insert(b);
            }
        }
        let committed_meta_loads = if !post_tri_roots.is_empty() || !post_inst_roots.is_empty() {
            let meta = self.analyze_meta_uses(
                &post_scope,
                &post_inst_roots,
                &post_tri_roots,
                &[],
                &[],
                None,
                lifted,
                lifted_blocks,
            )?;
            if meta.loads.iter().any(|&(s, kind)| {
                claimed_cand_meta.contains(&s)
                    || claimed_committed_meta
                        .get(&s)
                        .is_some_and(|&(prev_kind, _)| prev_kind != kind)
            }) {
                return None;
            }
            meta.loads
        } else {
            Vec::new()
        };

        Some((bindings, committed_meta_loads))
    }

    #[allow(clippy::too_many_arguments)]
    fn classify_value(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        loop_all: &HashSet<usize>,
        entry_block: usize,
        t_max: RayInput,
        tri_hit: &TriangleHitInfo,
        lifted: &[RayLoop],
        lifted_blocks: &HashSet<usize>,
    ) -> Option<RayOutputBinding> {
        // 1. Copy of an unmodified pre-loop value?
        if let Some(pre) = self.is_unchanged_copy(v, nodes, entry_block, 0) {
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CopyOf(pre),
                fallback: Some(pre),
            });
        }

        // 2. Which register carries `v`?
        let reg = self.value_reg(v)?;

        // 3. CommittedRayT (`extent` / `committed_t` register).
        if reg == tri_hit.committed_t_reg {
            let fallback = match t_max {
                RayInput::Val(tv) => Some(tv),
                _ => None,
            };
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CommittedRayT,
                fallback,
            });
        }

        // 4. CommittedStatusTriNode (`committed_tri_node` or `committed_inst_node` register).
        //    Post-loop pointer dereferences are validated as a group by `analyze_meta_uses`!
        if Some(reg) == tri_hit.committed_tri_reg || Some(reg) == tri_hit.committed_inst_reg {
            let fallback = self.pre_loop_value(v, nodes, entry_block, 0);
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CommittedStatusTriNode,
                fallback,
            });
        }

        // 5. Committed barycentrics & front-face.
        if Some(reg) == tri_hit.committed_bary_u_reg {
            let fallback = self.pre_loop_value(v, nodes, entry_block, 0);
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CommittedBary(0),
                fallback,
            });
        }
        if Some(reg) == tri_hit.committed_bary_v_reg {
            let fallback = self.pre_loop_value(v, nodes, entry_block, 0);
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CommittedBary(1),
                fallback,
            });
        }
        if Some(reg) == tri_hit.committed_front_face_reg {
            let fallback = self.pre_loop_value(v, nodes, entry_block, 0);
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::CommittedFrontFace,
                fallback,
            });
        }

        // 6. FrontBackTrack `minFrontFaceT` output or AnyHit loop-carried user state output.
        if let HitMode::FrontBackTrack {
            init_min_front_t,
            min_front_t_reg,
        } = tri_hit.mode
            && reg == min_front_t_reg
        {
            let fallback = match init_min_front_t {
                RayInput::Val(iv) => Some(iv),
                _ => None,
            };
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::FrontBackMinFrontT,
                fallback,
            });
        }
        if let HitMode::AnyHit(ref ah) = tri_hit.mode
            && let Some(idx) = ah.state_vars.iter().position(|sv| sv.reg == reg)
        {
            let fallback = match ah.state_vars[idx].init {
                RayInput::Val(iv) => Some(iv),
                _ => None,
            };
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::AnyHitState(idx),
                fallback,
            });
        }

        // 7. Loop iteration counter or timeout flag -> `RayOutput::Zero`.
        if self.is_iter_counter_or_timeout(v, nodes, entry_block, 0) {
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::Zero,
                fallback: None,
            });
        }

        // 8. Value whose post-loop uses are provably disjoint from `reach[entry_block]`, OR
        //    a dead BVH-traversal scratch VGPR that only flows into post-loop merge PHIs.
        //    Since `entry_block` lanes never observe this value, emit `Zero` (no `CopyOf(pre)`)
        //    so we never create spurious dependencies on earlier loops' scratch VGPRs!
        if matches!(reg, Reg::V(_))
            && (self.uses_disjoint_from_entry(v, loop_all, entry_block, lifted, lifted_blocks)
                || self.is_dead_bvh_scratch_liveout(v, reg, nodes, loop_all, tri_hit, lifted, lifted_blocks))
        {
            return Some(RayOutputBinding {
                value: v,
                output: RayOutput::Zero,
                fallback: None,
            });
        }

        // Unclassified live-out: fail-safe reject!
        None
    }

    /// Returns `true` if every post-loop operand use of `root` (outside `loop_all` and
    /// `lifted_blocks`) occurs only in lanes provably disjoint from `self.st.reach[entry_block]`
    /// (tracking negated guards across guarded VGPR overwrites), or is an unguarded overwrite.
    fn uses_disjoint_from_entry(
        &self,
        root: Value,
        loop_all: &HashSet<usize>,
        entry_block: usize,
        lifted: &[RayLoop],
        lifted_blocks: &HashSet<usize>,
    ) -> bool {
        let entry_reach = self.st.reach[entry_block];
        let mut seen: HashSet<(Value, Vec<(Bool, bool)>)> = HashSet::new();
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> = vec![(root, vec![(entry_reach, true)])];
        let mut steps = 0usize;
        while let Some((v, facts)) = work.pop() {
            steps += 1;
            if steps > 256 {
                return false;
            }
            if !seen.insert((v, facts.clone())) {
                continue;
            }
            if lifted.iter().any(|other| {
                ray_loop_reads_value(other, v)
                    && self
                        .bools_meet(&[(entry_reach, true), (self.st.reach[other.entry_block], true)])
            }) {
                return false;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if loop_all.contains(&b) || lifted_blocks.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if !phi.args.contains(&v) {
                            continue;
                        }
                        if let Some(margs) = self.st.phi_masks.get(&phi.value) {
                            for (&arg, &mb) in phi.args.iter().zip(margs) {
                                if arg != v {
                                    continue;
                                }
                                let mut next_facts = facts.clone();
                                if mb != Bool::TRUE && !next_facts.contains(&(mb, true)) {
                                    next_facts.push((mb, true));
                                }
                                if self.bools_meet(&next_facts) {
                                    work.push((phi.value, next_facts));
                                }
                            }
                        } else {
                            for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                                if arg != v {
                                    continue;
                                }
                                let next_facts = self.edge_facts(pred, b, &facts);
                                if self.bools_meet(&next_facts) {
                                    work.push((phi.value, next_facts));
                                }
                            }
                        }
                    }
                    continue;
                };
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                let is_op_read = inst
                    .reads
                    .iter()
                    .any(|&(r, rv)| rv == v && ops.contains(&r));
                if !is_op_read {
                    // Partial-register / single-lane writes preserve bits/lanes of `old`
                    // even when active; conservatively reject.
                    let is_partial = inst.name() == "v_writelane_b32"
                        || matches!(&inst.fields, Fields::Valu(f) if f.sdwa.is_some());
                    if is_partial {
                        return false;
                    }
                    if let Some(g) = self.st.guards[b][i] {
                        // A guarded VGPR write lowers to `select(g, new, old)`, so `old` (`v`)
                        // survives into `w` only in lanes where `g == false`.
                        // Inside a 1-block GFX10 waterfall loop (`v_readfirstlane` + `s_andn2_wrexec`),
                        // every lane active on entry (`reach[b]`) executes the body once, so a full
                        // write to a register never operand-read in `b` kills `old` on all `reach[b]` lanes!
                        let kill_cond = if self.is_waterfall_kill_block(b, v) {
                            self.st.reach[b]
                        } else {
                            g
                        };
                        let mut next_facts = facts.clone();
                        if !next_facts.contains(&(kill_cond, false)) {
                            next_facts.push((kill_cond, false));
                        }
                        if self.bools_meet(&next_facts) {
                            for &(r, rv) in &inst.reads {
                                if rv == v
                                    && let Some(w) = inst.write(r)
                                {
                                    work.push((w, next_facts.clone()));
                                }
                            }
                        }
                    }
                    continue;
                }
                let mut check = facts.clone();
                check.push((self.st.reach[b], true));
                if let Some(g) = self.st.guards[b][i] {
                    check.push((g, true));
                }
                if self.bools_meet(&check) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns `true` if `b` is a 1-block GFX10 waterfall loop (`v_readfirstlane_b32` +
    /// `s_andn2_wrexec_b32/b64` looping to `b`) where `v` is never operand-read inside `b`.
    fn is_waterfall_kill_block(&self, b: usize, v: Value) -> bool {
        let blk = &self.p.blocks[b];
        let self_loop = blk.exit.successors().contains(&b);
        if !self_loop {
            return false;
        }
        let has_wrexec = blk
            .insts
            .iter()
            .any(|i| matches!(i.name(), "s_andn2_wrexec_b32" | "s_andn2_wrexec_b64"));
        let has_rfl = blk.insts.iter().any(|i| i.name() == "v_readfirstlane_b32");
        if !has_wrexec || !has_rfl {
            return false;
        }
        for inst in &blk.insts {
            let ops = operand_reads(self.p, inst);
            if inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if `v` (in VGPR `reg`) is purely an internal BVH traversal scratch
    /// register (never written in hit-commit / any-hit blocks) that has zero direct
    /// post-loop operand reads and only enters post-loop control-flow merge PHIs.
    #[allow(clippy::too_many_arguments)]
    fn is_dead_bvh_scratch_liveout(
        &self,
        v: Value,
        reg: Reg,
        nodes: &HashSet<usize>,
        loop_all: &HashSet<usize>,
        tri_hit: &TriangleHitInfo,
        lifted: &[RayLoop],
        lifted_blocks: &HashSet<usize>,
    ) -> bool {
        if lifted.iter().any(|other| ray_loop_reads_real_input(other, v)) {
            return false;
        }
        // Reject if `reg` is written in commit_block, opacity_block (after `cmpx`), or hit_block
        // (after `s_and_saveexec`). Dead local scratch temps written inside `ah.body_blocks` (that
        // are NOT `state_vars`, NOT committed, and have zero direct post-loop operand reads) are safe.
        let mut forbidden_blocks: HashSet<usize> = HashSet::new();
        if let HitMode::AnyHit(ref ah) = tri_hit.mode {
            if let Exit::Jump(commit_b) = self.p.blocks[ah.tail_block].exit {
                forbidden_blocks.insert(commit_b);
            }
        }
        for &b in &forbidden_blocks {
            if self.p.blocks[b]
                .insts
                .iter()
                .any(|inst| inst.writes.iter().any(|&(r, _)| r == reg))
            {
                return false;
            }
        }
        if let Some(op_b) = tri_hit.opacity_block {
            let start = self
                .find_opacity_cmpx_in_block(op_b)
                .map_or(0, |idx| idx + 1);
            if self.p.blocks[op_b].insts[start..]
                .iter()
                .any(|inst| inst.writes.iter().any(|&(r, _)| r == reg))
            {
                return false;
            }
        }
        let hb = tri_hit.hit_block;
        if let Some(se_idx) = self.p.blocks[hb]
            .insts
            .iter()
            .position(|i| matches!(i.name(), "s_and_saveexec_b32" | "s_and_saveexec_b64"))
        {
            if self.p.blocks[hb].insts[se_idx + 1..]
                .iter()
                .any(|inst| inst.writes.iter().any(|&(r, _)| r == reg))
            {
                return false;
            }
        }
        // Must actually be written by BVH scaffolding inside `nodes`.
        let written_in_scaff = nodes.iter().any(|&b| {
            !forbidden_blocks.contains(&b)
                && self.p.blocks[b]
                    .insts
                    .iter()
                    .any(|inst| inst.writes.iter().any(|&(r, _)| r == reg))
        });
        if !written_in_scaff {
            return false;
        }
        // Every post-loop use of `v` must be either a non-operand overwrite (`op=false`)
        // or a merge PHI (with at least one incoming edge != `v`), or a fallback/copy in another lifted loop.
        let used_as_other_fallback = lifted.iter().any(|other| ray_loop_reads_value(other, v));
        let mut saw_use = used_as_other_fallback;
        for &(b, idx_opt) in &self.uses[v.0 as usize] {
            if loop_all.contains(&b) || lifted_blocks.contains(&b) {
                continue;
            }
            saw_use = true;
            match idx_opt {
                Some(i) => {
                    let inst = &self.p.blocks[b].insts[i];
                    let ops = operand_reads(self.p, inst);
                    if inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                        return false;
                    }
                }
                None => {
                    for phi in &self.p.blocks[b].phis {
                        if phi.args.contains(&v) && phi.args.iter().all(|&a| a == v) {
                            return false;
                        }
                    }
                }
            }
        }
        saw_use
    }

    fn value_reg(&self, v: Value) -> Option<Reg> {
        match self.defs.get(v.0 as usize).copied().flatten()? {
            Def::Input(r) => Some(r),
            Def::Phi(b, pi) => Some(self.p.blocks[b].phis[pi].reg),
            Def::Inst(b, i) => self.p.blocks[b].insts[i]
                .writes
                .iter()
                .find(|&&(_, w)| w == v)
                .map(|&(r, _)| r),
        }
    }

    fn is_unchanged_copy(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
        depth: u32,
    ) -> Option<Value> {
        if depth > 8 {
            return None;
        }
        match self.defs.get(v.0 as usize).copied().flatten()? {
            Def::Input(_) => Some(v),
            Def::Inst(b, i) => {
                if !nodes.contains(&b) {
                    return Some(v);
                }
                let inst = &self.p.blocks[b].insts[i];
                if matches!(inst.name(), "v_mov_b32" | "s_mov_b32") {
                    let src = single_reg_src(inst)?;
                    return self.is_unchanged_copy(src, nodes, entry_block, depth + 1);
                }
                None
            }
            Def::Phi(b, pi) => {
                if !nodes.contains(&b) {
                    return Some(v);
                }
                let phi = &self.p.blocks[b].phis[pi];
                let mut pre: Option<Value> = None;
                for &arg in &phi.args {
                    if arg == v {
                        continue;
                    }
                    let c = self.is_unchanged_copy(arg, nodes, entry_block, depth + 1)?;
                    if let Some(prev) = pre {
                        if prev != c {
                            return None;
                        }
                    } else {
                        pre = Some(c);
                    }
                }
                pre
            }
        }
    }

    /// Verifies that every use of `inst_roots` / `tri_roots` inside `scope` is either a
    /// hit/miss status test (`== 0`, `!= 0`, `& 4`, `< 4`) or an XDXR BVH address chain feeding
    /// ONLY replaced [`RayMetaKind`] `global_load_*` instructions. Rejects immediately if any
    /// pointer escapes (e.g. raw triangle vertex loads at `tri + 60` / `tri + 12*idx`).
    #[allow(clippy::too_many_arguments)]
    fn analyze_meta_uses(
        &self,
        scope: &HashSet<usize>,
        inst_roots: &[Value],
        tri_roots: &[Value],
        inst_byte_off_lo_roots: &[Value],
        inst_addr_hi_roots: &[Value],
        skipped_tail: Option<(usize, &HashSet<usize>)>,
        lifted: &[RayLoop],
        _lifted_blocks: &HashSet<usize>,
    ) -> Option<MetaAnalysisResult> {
        macro_rules! meta_fail {
            ($($arg:tt)*) => {{
                return None;
            }};
        }

        // Collect `Bool` predicates provably `true` whenever any committed node root is non-zero
        // (i.e. `if (q.CommittedStatus() != COMMITTED_NOTHING)`). On lanes where `tri_node == 0`
        // (miss), hardware `RayQuery` produces the exact same `0` as software BVH, so a raw BVH
        // pointer can only differ from `1` on lanes where these hit-status predicates hold!
        let mut hit_facts: Vec<(Bool, bool)> = Vec::new();
        if skipped_tail.is_none() {
            let mut root_set: HashSet<Value> = HashSet::new();
            root_set.extend(tri_roots.iter().copied());
            root_set.extend(inst_roots.iter().copied());
            self.collect_hit_status_facts(&root_set, scope, &mut hit_facts);
        }

        let mut tags: HashMap<Value, BvhTag> = HashMap::new();
        let mut tag_facts: HashMap<Value, Vec<(Bool, bool)>> = HashMap::new();
        let mut work: Vec<Value> = Vec::new();
        fn set_tag(
            tags: &mut HashMap<Value, BvhTag>,
            tag_facts: &mut HashMap<Value, Vec<(Bool, bool)>>,
            work: &mut Vec<Value>,
            v: Value,
            tag: BvhTag,
            facts: Vec<(Bool, bool)>,
        ) -> bool {
            if let Some(&prev) = tags.get(&v) {
                if prev != tag {
                    return false;
                }
                // Merge lane facts conservatively: keep only facts common to both incoming paths.
                if let Some(prev_f) = tag_facts.get_mut(&v) {
                    let old_len = prev_f.len();
                    prev_f.retain(|item| facts.contains(item));
                    if prev_f.len() != old_len {
                        work.push(v);
                    }
                }
                return true;
            }
            tags.insert(v, tag);
            tag_facts.insert(v, facts);
            work.push(v);
            true
        }

        for &v in inst_roots {
            if !set_tag(&mut tags, &mut tag_facts, &mut work, v, BvhTag::InstNode, hit_facts.clone()) {
                return None;
            }
        }
        for &v in tri_roots {
            if !set_tag(&mut tags, &mut tag_facts, &mut work, v, BvhTag::TriNode, hit_facts.clone()) {
                return None;
            }
        }
        for &v in inst_byte_off_lo_roots {
            if !set_tag(&mut tags, &mut tag_facts, &mut work, v, BvhTag::InstByteOff(0), Vec::new()) {
                return None;
            }
        }
        for &v in inst_addr_hi_roots {
            if !set_tag(&mut tags, &mut tag_facts, &mut work, v, BvhTag::AddrHi, Vec::new()) {
                return None;
            }
        }

        let mut loads: HashMap<Site, RayMetaKind> = HashMap::new();
        // Deferred two-operand adds whose partner wasn't tagged yet (makes worklist order-independent).
        let mut deferred: Vec<(Value, usize, usize)> = Vec::new();

        for round in 0..6 {
            while let Some(v) = work.pop() {
            let tag = tags[&v];
            let v_facts = tag_facts.get(&v).cloned().unwrap_or_default();

            // If `v` feeds another already-lifted loop, only allow it as a `CommittedStatusTriNode`
            // or `CopyOf` fallback (never as ray coordinates, mask, or flags).
            for other in lifted {
                if ray_input_contains_val(other.origin, v)
                    || ray_input_contains_val(other.dir, v)
                    || matches!(other.t_max, RayInput::Val(x) if x == v)
                    || matches!(other.t_min, Some(RayInput::Val(x)) if x == v)
                    || other.active_ray_mask == Some(v)
                    || matches!(other.instance_mask, RayInput::Val(x) | RayInput::ShiftedMask(x) if x == v)
                    || matches!(other.ray_flags, RayInput::Val(x) | RayInput::ShiftedMask(x) if x == v)
                {
                    return None;
                }
                for out in &other.outputs {
                    if out.fallback == Some(v)
                        && !matches!(out.output, RayOutput::CommittedStatusTriNode | RayOutput::CopyOf(_))
                    {
                        return None;
                    }
                    if out.output == RayOutput::CopyOf(v) {
                        if !set_tag(&mut tags, &mut tag_facts, &mut work, out.value, tag, v_facts.clone()) {
                            return None;
                        }
                    }
                }
            }

            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if phi.args.contains(&v) {
                            // Dead merge PHIs or incoming values whose lanes are disjoint from all
                            // operand uses of `phi` do not propagate!
                            if !self.phi_incoming_reaches_use_in_scope(
                                v,
                                phi.value,
                                &v_facts,
                                scope,
                                skipped_tail,
                            ) {
                                continue;
                            }
                            // For intermediate BVH address/offset scratch tags (not root pointers),
                            // any merge PHI that never reaches any `global_load_*`/`s_load_*` is
                            // dead scratch register reuse after the metadata loads completed.
                            let is_root_tag = matches!(tag, BvhTag::InstNode | BvhTag::TriNode);
                            if !is_root_tag && !self.reaches_global_or_smem_load(phi.value, scope) {
                                continue;
                            }
                            let mut phi_f_opt: Option<Vec<(Bool, bool)>> = None;
                            if let Some(margs) = self.st.phi_masks.get(&phi.value) {
                                for (&arg, &mb) in phi.args.iter().zip(margs) {
                                    if arg != v {
                                        continue;
                                    }
                                    let mut f = v_facts.clone();
                                    if mb != Bool::TRUE && !f.contains(&(mb, true)) {
                                        f.push((mb, true));
                                    }
                                    if !self.bools_meet(&f) {
                                        continue;
                                    }
                                    match &mut phi_f_opt {
                                        None => phi_f_opt = Some(f),
                                        Some(prev) => prev.retain(|item| f.contains(item)),
                                    }
                                }
                            } else {
                                for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                                    if arg != v {
                                        continue;
                                    }
                                    let f = self.edge_facts(pred, b, &v_facts);
                                    if !self.bools_meet(&f) {
                                        continue;
                                    }
                                    match &mut phi_f_opt {
                                        None => phi_f_opt = Some(f),
                                        Some(prev) => prev.retain(|item| f.contains(item)),
                                    }
                                }
                            }
                            let Some(phi_f) = phi_f_opt else {
                                continue;
                            };
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, phi.value, tag, phi_f) {
                                meta_fail!(
                                    "PHI clash at B{b}@{:#x} phi={:?}({:?}) incoming={v:?}({tag:?}) prev={:?}",
                                    self.p.blocks[b].offset,
                                    phi.value,
                                    phi.reg,
                                    tags.get(&phi.value)
                                );
                            }
                        }
                    }
                    continue;
                };
                if let Some((tb, sk)) = skipped_tail
                    && b == tb
                    && sk.contains(&i)
                {
                    continue;
                }
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                let is_op_read = inst
                    .reads
                    .iter()
                    .any(|&(r, rv)| rv == v && ops.contains(&r));
                if !is_op_read {
                    if let Some(g) = self.st.guards[b][i] {
                        let keep_taint = matches!(tag, BvhTag::InstNode | BvhTag::TriNode);
                        let kill_cond = if self.is_waterfall_kill_block(b, v) {
                            self.st.reach[b]
                        } else {
                            g
                        };
                        let mut nf = v_facts.clone();
                        if !nf.contains(&(kill_cond, false)) {
                            nf.push((kill_cond, false));
                        }
                        if self.bools_meet(&nf) {
                            for &(r, rv) in &inst.reads {
                                if rv == v
                                    && let Some(w) = inst.write(r)
                                    && self.value_reaches_op_use_under(w, nf.clone(), scope, skipped_tail)
                                    && (keep_taint || self.reaches_global_or_smem_load(w, scope))
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, w, tag, nf.clone())
                                {
                                    meta_fail!(
                                        "guarded overwrite clash at B{b}:{i}@{:#x} w={w:?} tag={tag:?} prev={:?}",
                                        inst.machine.offset,
                                        tags.get(&w)
                                    );
                                }
                            }
                        }
                    }
                    continue;
                }

                // Skip any operand read whose active lanes are provably disjoint from `v_facts`
                // (e.g. a `v_cmp_ge_f32` reading the newly-overwritten float value inside the same
                // guarded EXEC scope where `v`'s old `TriNode` only survived on `!g`!).
                {
                    let mut use_check = v_facts.clone();
                    use_check.push((self.st.reach[b], true));
                    if let Some(g) = self.st.guards[b][i] {
                        use_check.push((g, true));
                    } else if inst.name().starts_with("v_cmp_")
                        && let Some(eg) = self.same_exec_scope_guard(b, i)
                    {
                        use_check.push((eg, true));
                    }
                    if !self.bools_meet(&use_check) {
                        continue;
                    }
                }

                let n = inst.name();
                // Simple copies preserve every tag.
                if matches!(n, "v_mov_b32" | "s_mov_b32") {
                    if let Some(&(_, w)) = inst.writes.first() {
                        if !set_tag(&mut tags, &mut tag_facts, &mut work, w, tag, v_facts.clone()) {
                            meta_fail!(
                                "mov clash at B{b}:{i}@{:#x} w={w:?} tag={tag:?} prev={:?}",
                                inst.machine.offset,
                                tags.get(&w)
                            );
                        }
                    }
                    continue;
                }

                // Pure status tests allowed on `InstNode`, `TriNode`, `StatusBits`.
                if matches!(tag, BvhTag::InstNode | BvhTag::TriNode | BvhTag::StatusBits) {
                    match n {
                        "v_cmp_eq_u32" | "v_cmp_ne_u32" | "v_cmpx_eq_u32" | "v_cmpx_ne_u32"
                        | "v_cmp_eq_i32" | "v_cmp_ne_i32" | "v_cmpx_eq_i32" | "v_cmpx_ne_i32"
                        | "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_i32" | "s_cmp_lg_i32"
                            if other_const_arg(inst, v) == Some(0) =>
                        {
                            continue;
                        }
                        "v_cmp_lt_u32" | "v_cmp_gt_u32" | "v_cmpx_lt_u32" | "v_cmpx_gt_u32"
                        | "v_cmp_ge_u32" | "v_cmp_le_u32"
                            if matches!(other_const_arg(inst, v), Some(0) | Some(4)) =>
                        {
                            continue;
                        }
                        "v_and_b32" | "s_and_b32" if other_const_arg(inst, v) == Some(4) => {
                            if let Some(&(_, w)) = inst.writes.first()
                                && !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::StatusBits, v_facts.clone())
                            {
                                return None;
                            }
                            continue;
                        }
                        _ => {}
                    }
                }

                match tag {
                    BvhTag::StatusBits => meta_fail!("StatusBits used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                    BvhTag::InstNode => {
                        if matches!(n, "v_lshrrev_b32" | "s_lshr_b32")
                            && other_const_arg(inst, v) == Some(3)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::InstNodeShr3, v_facts.clone()) {
                                meta_fail!("InstNodeShr3 clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("InstNode used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::InstNodeShr3 => {
                        if matches!(n, "v_lshlrev_b64" | "s_lshl_b64") && has_const_operand(inst, 6)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::InstByteOff(0), v_facts.clone()) {
                                meta_fail!("InstByteOff clash at B{b}:{i}");
                            }
                            if let Some(&(_, hi)) = inst.writes.get(1)
                                && !set_tag(&mut tags, &mut tag_facts, &mut work, hi, BvhTag::AddrHi, v_facts.clone())
                            {
                                meta_fail!("AddrHi clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v) == Some(6)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::InstByteOff(0), v_facts.clone()) {
                                meta_fail!("InstByteOff32 clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("InstNodeShr3 used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::InstByteOff(off) => {
                        if matches!(n, "v_or_b32" | "s_or_b32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 128
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::InstByteOff(off | k), v_facts.clone()) {
                                meta_fail!("InstByteOff or clash at B{b}:{i}");
                            }
                        } else if matches!(
                            n,
                            "v_add_co_u32" | "v_add_nc_u32" | "s_add_u32" | "s_add_i32"
                        ) {
                            if let Some(k) = other_const_arg(inst, v)
                                && k < 128
                            {
                                let w = inst.writes.first()?.1;
                                if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::InstByteOff(off + k), v_facts.clone()) {
                                    meta_fail!("InstByteOff addk clash at B{b}:{i}");
                                }
                            } else if let Some(base_v) = other_val_arg(inst, v) {
                                let base_off = self.extract_base_low_offset(base_v, 0)?;
                                let lo = inst.writes.first()?.1;
                                if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::InstAddrLo(off + base_off), v_facts.clone()) {
                                    meta_fail!("InstAddrLo clash at B{b}:{i}");
                                }
                                // Allow carry-out into `addc` / `add_co_ci`.
                                for &(wr, wv) in inst.writes.iter().skip(1) {
                                    if matches!(wr, Reg::S(_) | Reg::Scc)
                                        && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                    {
                                        meta_fail!("AddrCarry clash at B{b}:{i}");
                                    }
                                }
                            } else {
                                meta_fail!("InstByteOff({off}) add no other_val at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                            }
                        } else {
                            meta_fail!("InstByteOff({off}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::InstAddrLo(off) => {
                        if matches!(n, "v_or_b32" | "s_or_b32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 128
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::InstAddrLo(off | k), v_facts.clone()) {
                                meta_fail!("InstAddrLo or clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32" | "s_add_u32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 128
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::InstAddrLo(off + k), v_facts.clone()) {
                                meta_fail!("InstAddrLo addk clash at B{b}:{i}");
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("InstAddrLo carry clash at B{b}:{i}");
                                }
                            }
                        } else if let Some((ndw, imm_off)) = flat_or_smem_load_info(inst, v) {
                            let total = off.checked_add(imm_off)?;
                            let kind = match (ndw, total) {
                                (1, 48) => {
                                    let dst = inst.writes.first()?.1;
                                    if !self.uses_only_low_24_bits(dst, scope) {
                                        meta_fail!("InstId low24 failed at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                                    }
                                    RayMetaKind::InstanceId
                                }
                                (1, 52) => {
                                    let dst = inst.writes.first()?.1;
                                    if !self.uses_only_low_24_bits(dst, scope) {
                                        meta_fail!("InstContrib low24 failed at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                                    }
                                    RayMetaKind::InstanceContribution
                                }
                                (1, 60) => {
                                    let dst = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, dst, BvhTag::BlasBaseShr8, v_facts.clone()) {
                                        meta_fail!("BlasBaseShr8 clash at B{b}:{i}");
                                    }
                                    RayMetaKind::BlasBaseZero
                                }
                                (1, 112) => RayMetaKind::InstanceIndex,
                                (n_dw, t) if t < 48 && t % 4 == 0 && t + 4 * n_dw <= 48 => {
                                    RayMetaKind::ObjectToWorld(t)
                                }
                                _ => meta_fail!("InstAddrLo({total}, ndw={ndw}) unsupported load at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                            };
                            if let Some(&prev) = loads.get(&(b, i))
                                && prev != kind
                            {
                                meta_fail!("InstAddrLo load kind clash at B{b}:{i}");
                            }
                            loads.insert((b, i), kind);
                        } else {
                            meta_fail!("InstAddrLo({off}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::AddrHi => {
                        if matches!(n, "v_add_co_ci_u32" | "s_addc_u32") {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::AddrHi, v_facts.clone()) {
                                meta_fail!("AddrHi addc clash at B{b}:{i}");
                            }
                        } else if flat_or_smem_load_hi_only(inst, v) {
                            // High address word of a replaced `global_load_*` / `s_load_*`: checked via lo word!
                        } else {
                            meta_fail!("AddrHi used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::AddrCarry => {
                        if matches!(n, "v_add_co_ci_u32" | "s_addc_u32") {
                            let w = inst.writes.first()?.1;
                            if !tags.contains_key(&w)
                                && !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::AddrHi, v_facts.clone())
                            {
                                meta_fail!("AddrCarry hi clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("AddrCarry used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::BlasBaseShr8 => {
                        if matches!(n, "v_lshlrev_b64" | "s_lshl_b64") && has_const_operand(inst, 8)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::BlasBaseLo, v_facts.clone()) {
                                meta_fail!("BlasBaseLo clash at B{b}:{i}");
                            }
                            if let Some(&(_, hi)) = inst.writes.get(1)
                                && !set_tag(&mut tags, &mut tag_facts, &mut work, hi, BvhTag::AddrHi, v_facts.clone())
                            {
                                meta_fail!("AddrHi clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v) == Some(8)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::BlasBaseLo, v_facts.clone()) {
                                meta_fail!("BlasBaseLo32 clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("BlasBaseShr8 used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::BlasBaseLo => {
                        if matches!(n, "v_or_b32" | "s_or_b32" | "v_add_co_u32" | "v_add_nc_u32")
                            && other_const_arg(inst, v) == Some(60)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::BlasBaseHdrLo(60), v_facts.clone()) {
                                meta_fail!("BlasBaseHdrLo clash at B{b}:{i}");
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("BlasBaseHdrLo carry clash at B{b}:{i}");
                                }
                            }
                        } else if matches!(
                            n,
                            "v_add_co_u32" | "v_add_nc_u32" | "s_add_u32" | "v_or_b32"
                        ) {
                            let other_v = other_val_arg(inst, v)?;
                            let other_tag = tags.get(&other_v).copied();
                            let out_tag = match other_tag {
                                Some(BvhTag::TriByteOff(off)) => BvhTag::TriAddrLo(off),
                                Some(BvhTag::TriByteOffPrim) => BvhTag::TriAddrPrimLo,
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _ => meta_fail!("BlasBaseLo add bad other_tag={other_tag:?} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                            };
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, out_tag, v_facts.clone()) {
                                meta_fail!("BlasBaseLo out_tag={out_tag:?} clash (prev={:?}) at B{b}:{i}@{:#x}", tags.get(&lo), self.p.blocks[b].insts[i].machine.offset);
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("BlasBaseLo carry clash at B{b}:{i}");
                                }
                            }
                        } else {
                            meta_fail!("BlasBaseLo used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::BlasBaseHdrLo(off) => {
                        if let Some((1, imm_off)) = flat_or_smem_load_info(inst, v)
                            && off.checked_add(imm_off) == Some(60)
                        {
                            let kind = RayMetaKind::BlasBaseZero;
                            if let Some(&prev) = loads.get(&(b, i))
                                && prev != kind
                            {
                                meta_fail!("BlasBaseHdrLo load clash at B{b}:{i}");
                            }
                            loads.insert((b, i), kind);
                        } else {
                            meta_fail!("BlasBaseHdrLo({off}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriNode => {
                        if matches!(n, "v_lshrrev_b32" | "s_lshr_b32")
                            && other_const_arg(inst, v) == Some(3)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriNodeShr3, v_facts.clone()) {
                                meta_fail!("TriNodeShr3 clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_and_b32" | "s_and_b32")
                            && other_const_arg(inst, v) == Some(7)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriSlot, v_facts.clone()) {
                                meta_fail!("TriSlot clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!(
                                "TriNode used by {n} at B{b}:{i}@{:#x} v={v:?} def={:?} v_facts={v_facts:?} g={:?} eg={:?}",
                                self.p.blocks[b].insts[i].machine.offset,
                                self.defs.get(v.0 as usize),
                                self.st.guards[b][i],
                                self.same_exec_scope_guard(b, i)
                            );
                        }
                    }
                    BvhTag::TriNodeShr3 => {
                        if matches!(n, "v_lshlrev_b64" | "s_lshl_b64") && has_const_operand(inst, 6)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::TriByteOff(0), v_facts.clone()) {
                                meta_fail!("TriByteOff clash at B{b}:{i}");
                            }
                            if let Some(&(_, hi)) = inst.writes.get(1)
                                && !set_tag(&mut tags, &mut tag_facts, &mut work, hi, BvhTag::AddrHi, v_facts.clone())
                            {
                                meta_fail!("TriNodeShr3 hi clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v) == Some(6)
                        {
                            let lo = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, lo, BvhTag::TriByteOff(0), v_facts.clone()) {
                                meta_fail!("TriByteOff32 clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("TriNodeShr3 used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriSlot => {
                        if n == "v_lshl_add_u32"
                            && let Fields::Valu(f) = &inst.fields
                            && f.src.len() >= 3
                            && const_operand(&f.src[1]) == Some(2)
                            && const_operand(&f.src[2]) == Some(52)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriPrimOff, v_facts.clone()) {
                                meta_fail!("TriPrimOff clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v) == Some(2)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriSlotShl2, v_facts.clone()) {
                                meta_fail!("TriSlotShl2 clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v) == Some(3)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchScratch, v_facts.clone()) {
                                meta_fail!("TriFetchScratch clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_min_u32" | "v_min_i32") {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriSlot, v_facts.clone()) {
                                meta_fail!("TriSlot min clash at B{b}:{i}");
                            }
                        } else if matches!(
                            n,
                            "v_cmp_lt_u32" | "v_cmp_gt_u32" | "v_cmp_eq_u32" | "v_cmp_ne_u32"
                        ) {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriSlotCmp, v_facts.clone()) {
                                meta_fail!("TriSlotCmp clash at B{b}:{i}");
                            }
                        } else if (n == "v_cmpx_gt_u32"
                            && two_src_operands(inst)
                                .is_some_and(|(a, b)| const_operand(a) == Some(4) && operand_val(inst, b) == Some(v)))
                            || (n == "v_cmpx_lt_u32"
                                && two_src_operands(inst)
                                    .is_some_and(|(a, b)| operand_val(inst, a) == Some(v) && const_operand(b) == Some(4)))
                        {
                            // `TriSlot < 4` (triangle node vs procedural/box): always true on
                            // `CommittedStatusTriNode == 1` (`1 & 7 == 1 < 4`) and writes only EXEC.
                        } else {
                            meta_fail!("TriSlot used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriSlotShl2 => {
                        if matches!(n, "v_add_nc_u32" | "v_add_co_u32" | "v_or_b32")
                            && other_const_arg(inst, v) == Some(52)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriPrimOff, v_facts.clone()) {
                                meta_fail!("TriPrimOff2 clash at B{b}:{i}");
                            }
                        } else if (matches!(n, "v_add_nc_u32" | "v_add_co_u32" | "v_or_b32")
                            && other_const_arg(inst, v) == Some(16))
                            || (n == "v_lshrrev_b32" && other_const_arg(inst, v) == Some(1058))
                            || (n == "v_bfe_u32"
                                && matches!(
                                    &inst.fields,
                                    Fields::Valu(f)
                                        if f.src.len() >= 3
                                            && const_operand(&f.src[0]) == Some(1_127_293_456)
                                            && operand_val(inst, &f.src[1]) == Some(v)
                                            && const_operand(&f.src[2]) == Some(3)
                                ))
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchScratch, v_facts.clone()) {
                                meta_fail!("TriSlotShl2 fetch scratch clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("TriSlotShl2 used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriSlotCmp => {
                        if n == "v_cndmask_b32" {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriPrimOff, v_facts.clone()) {
                                meta_fail!("TriSlotCmp cnd clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("TriSlotCmp used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriPrimOff => {
                        if n == "v_cndmask_b32" {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriPrimOff, v_facts.clone()) {
                                meta_fail!("TriPrimOff cnd clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32" | "v_or_b32") {
                            let other_v = other_val_arg(inst, v)?;
                            let out_tag = match tags.get(&other_v).copied() {
                                Some(BvhTag::TriByteOff(_)) => BvhTag::TriByteOffPrim,
                                Some(BvhTag::TriAddrLo(0)) => BvhTag::TriAddrPrimLo,
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriPrimOff add bad other={_other:?} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                            };
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, out_tag, v_facts.clone()) {
                                meta_fail!("TriPrimOff add clash at B{b}:{i}@{:#x} out={out_tag:?} prev={:?}", self.p.blocks[b].insts[i].machine.offset, tags.get(&w));
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("TriPrimOff carry clash at B{b}:{i}");
                                }
                            }
                        } else {
                            meta_fail!("TriPrimOff used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriByteOff(off) => {
                        if matches!(n, "v_or_b32" | "s_or_b32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 64
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriByteOff(off | k), v_facts.clone()) {
                                meta_fail!("TriByteOff or clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32" | "v_or_b32") {
                            let other_v = other_val_arg(inst, v)?;
                            let out_tag = match tags.get(&other_v).copied() {
                                Some(BvhTag::BlasBaseLo) => BvhTag::TriAddrLo(off),
                                Some(BvhTag::TriPrimOff) => BvhTag::TriByteOffPrim,
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriByteOff({off}) add bad other={_other:?} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                            };
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, out_tag, v_facts.clone()) {
                                meta_fail!("TriByteOff({off}) add clash at B{b}:{i}@{:#x} out={out_tag:?} prev={:?}", self.p.blocks[b].insts[i].machine.offset, tags.get(&w));
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("TriByteOff carry clash at B{b}:{i}");
                                }
                            }
                        } else {
                            meta_fail!("TriByteOff({off}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriByteOffPrim => {
                        if matches!(n, "v_add_co_u32" | "v_add_nc_u32") {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriAddrPrimLo, v_facts.clone()) {
                                meta_fail!("TriByteOffPrim clash at B{b}:{i}@{:#x} prev={:?}", self.p.blocks[b].insts[i].machine.offset, tags.get(&w));
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("TriByteOffPrim carry clash at B{b}:{i}");
                                }
                            }
                        } else {
                            meta_fail!("TriByteOffPrim used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriAddrLo(off) => {
                        if matches!(n, "v_or_b32" | "s_or_b32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 64
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriAddrLo(off | k), v_facts.clone()) {
                                meta_fail!("TriAddrLo or clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32")
                            && let Some(k) = other_const_arg(inst, v)
                            && k < 64
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriAddrLo(off + k), v_facts.clone()) {
                                meta_fail!("TriAddrLo addk clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32")
                            && let Some(ov) = other_val_arg(inst, v)
                        {
                            let out_tag = match (off, tags.get(&ov).copied()) {
                                (_, Some(BvhTag::TriPrimOff)) => BvhTag::TriAddrPrimLo,
                                (0, Some(BvhTag::TriFetchVertOff(slot))) => {
                                    BvhTag::TriFetchVertAddrLo(slot)
                                }
                                (_, None) if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                (_, _other) => meta_fail!(
                                    "TriAddrLo({off}) add bad other={_other:?} at B{b}:{i}@{:#x}",
                                    self.p.blocks[b].insts[i].machine.offset
                                ),
                            };
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, out_tag, v_facts.clone()) {
                                meta_fail!("TriAddrLo add clash at B{b}:{i}");
                            }
                            for &(wr, wv) in inst.writes.iter().skip(1) {
                                if matches!(wr, Reg::S(_) | Reg::Scc)
                                    && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                {
                                    meta_fail!("TriAddrLo carry clash at B{b}:{i}");
                                }
                            }
                        } else if let Some((ndw, imm_off)) = flat_or_smem_load_info(inst, v) {
                            let total = off.checked_add(imm_off)?;
                            let kind = match (ndw, total) {
                                (1, 48) => {
                                    let dst = inst.writes.first()?.1;
                                    if !self.uses_only_shr_8(dst, scope) {
                                        meta_fail!("GeomIdx shr8 failed at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                                    }
                                    RayMetaKind::GeometryIndexShl8
                                }
                                (1, 52) | (1, 56) => RayMetaKind::PrimitiveIndex,
                                (1, 60) => {
                                    let dst = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, dst, BvhTag::TriFetchTriId, v_facts.clone()) {
                                        meta_fail!("TriFetchTriId clash at B{b}:{i}");
                                    }
                                    RayMetaKind::BlasBaseZero
                                }
                                _ => meta_fail!("TriAddrLo({total}, ndw={ndw}) unsupported load at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset),
                            };
                            if let Some(&prev) = loads.get(&(b, i))
                                && prev != kind
                            {
                                meta_fail!("TriAddrLo load clash at B{b}:{i}");
                            }
                            loads.insert((b, i), kind);
                        } else {
                            meta_fail!("TriAddrLo({off}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriAddrPrimLo => {
                        if let Some((1, 0)) = flat_or_smem_load_info(inst, v) {
                            let kind = RayMetaKind::PrimitiveIndex;
                            if let Some(&prev) = loads.get(&(b, i))
                                && prev != kind
                            {
                                meta_fail!("TriAddrPrimLo load clash at B{b}:{i}");
                            }
                            loads.insert((b, i), kind);
                        } else {
                            meta_fail!("TriAddrPrimLo used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriFetchScratch => {
                        if matches!(n, "v_and_b32" | "s_and_b32" | "v_lshlrev_b32" | "s_lshl_b32")
                            && other_const_arg(inst, v).is_some()
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchScratch, v_facts.clone()) {
                                meta_fail!("TriFetchScratch mask/shl clash at B{b}:{i}");
                            }
                        } else if n == "v_lshrrev_b32"
                            && let Some((s0, s1)) = two_src_operands(inst)
                            && operand_val(inst, s0) == Some(v)
                        {
                            let s1_const = const_operand(s1)
                                .or_else(|| operand_val(inst, s1).and_then(|x| self.eval_const(x, 0)));
                            if s1_const == Some(1_127_293_456) {
                                let w = inst.writes.first()?.1;
                                if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchScratch, v_facts.clone()) {
                                    meta_fail!("TriFetchScratch lut clash at B{b}:{i}");
                                }
                            } else if let Some(ov) = operand_val(inst, s1) {
                                match tags.get(&ov).copied() {
                                    Some(BvhTag::TriFetchTriId) => {
                                        let w = inst.writes.first()?.1;
                                        if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchTriId, v_facts.clone()) {
                                            meta_fail!("TriFetchScratch->TriId clash at B{b}:{i}");
                                        }
                                    }
                                    None if round < 5 => {
                                        deferred.push((v, b, i));
                                        continue;
                                    }
                                    _other => meta_fail!("TriFetchScratch lshr bad other={_other:?} at B{b}:{i}"),
                                }
                            } else {
                                meta_fail!("TriFetchScratch lshr bad s1 at B{b}:{i}");
                            }
                        } else if matches!(n, "v_or3_b32" | "v_or_b32")
                            && let Fields::Valu(f) = &inst.fields
                        {
                            let mut any_none = false;
                            let mut all_ok = true;
                            for op in &f.src {
                                if let Some(ov) = operand_val(inst, op) {
                                    match tags.get(&ov).copied() {
                                        Some(BvhTag::TriFetchScratch) => {}
                                        None => any_none = true,
                                        _ => all_ok = false,
                                    }
                                } else if const_operand(op).is_none() {
                                    all_ok = false;
                                }
                            }
                            if !all_ok {
                                meta_fail!("TriFetchScratch {n} bad arg at B{b}:{i}");
                            }
                            if any_none {
                                if round < 5 {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                meta_fail!("TriFetchScratch {n} unresolved arg at B{b}:{i}");
                            }
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchScratch, v_facts.clone()) {
                                meta_fail!("TriFetchScratch {n} clash at B{b}:{i}");
                            }
                        } else if n == "v_bfe_u32"
                            && let Fields::Valu(f) = &inst.fields
                            && f.src.len() >= 3
                            && operand_val(inst, &f.src[0]) == Some(v)
                            && const_operand(&f.src[2]) == Some(3)
                        {
                            let Some(sh_v) = operand_val(inst, &f.src[1]) else {
                                meta_fail!("TriFetchScratch bfe missing sh_v at B{b}:{i}");
                            };
                            match tags.get(&sh_v).copied() {
                                Some(BvhTag::TriFetchVertShift(slot)) => {
                                    let w = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertOff(slot), v_facts.clone()) {
                                        meta_fail!("TriFetchVertOff clash at B{b}:{i}");
                                    }
                                }
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriFetchScratch bfe bad shift={_other:?} at B{b}:{i}"),
                            }
                        } else {
                            meta_fail!("TriFetchScratch used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriFetchTriId => {
                        if n == "v_lshrrev_b32"
                            && let Some((s0, s1)) = two_src_operands(inst)
                            && operand_val(inst, s1) == Some(v)
                            && let Some(sh_v) = operand_val(inst, s0)
                        {
                            match tags.get(&sh_v).copied() {
                                Some(BvhTag::TriFetchScratch) => {
                                    let w = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchTriId, v_facts.clone()) {
                                        meta_fail!("TriFetchTriId shift clash at B{b}:{i}");
                                    }
                                }
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriFetchTriId shift bad other={_other:?} at B{b}:{i}"),
                            }
                        } else if matches!(n, "v_and_b32" | "s_and_b32")
                            && other_const_arg(inst, v) == Some(12)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertShift(0), v_facts.clone()) {
                                meta_fail!("TriFetchVertShift(0) clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                            && let Some(sh) = other_const_arg(inst, v)
                            && matches!(sh, 1 | 2)
                        {
                            let slot = if sh == 2 { 1 } else { 2 };
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertShift(slot), v_facts.clone()) {
                                meta_fail!("TriFetchVertShift({slot}) clash at B{b}:{i}");
                            }
                        } else {
                            meta_fail!("TriFetchTriId used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriFetchVertShift(slot) => {
                        if (matches!(n, "v_and_b32" | "s_and_b32")
                            && matches!(other_const_arg(inst, v), Some(12 | 30)))
                            || (matches!(n, "v_lshlrev_b32" | "s_lshl_b32")
                                && other_const_arg(inst, v) == Some(2))
                            || (n == "v_lshrrev_b32"
                                && two_src_operands(inst).is_some_and(|(s0, s1)| {
                                    operand_val(inst, s0) == Some(v)
                                        && const_operand(s1) == Some(66_072)
                                }))
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertShift(slot), v_facts.clone()) {
                                meta_fail!("TriFetchVertShift({slot}) step clash at B{b}:{i}");
                            }
                        } else if n == "v_bfe_u32"
                            && let Fields::Valu(f) = &inst.fields
                            && f.src.len() >= 3
                            && operand_val(inst, &f.src[1]) == Some(v)
                            && const_operand(&f.src[2]) == Some(3)
                        {
                            let Some(src0) = operand_val(inst, &f.src[0]) else {
                                meta_fail!("TriFetchVertShift bfe missing src0 at B{b}:{i}");
                            };
                            match tags.get(&src0).copied() {
                                Some(BvhTag::TriFetchScratch) => {
                                    let w = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertOff(slot), v_facts.clone()) {
                                        meta_fail!("TriFetchVertOff({slot}) clash at B{b}:{i}");
                                    }
                                }
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriFetchVertShift bfe bad src0={_other:?} at B{b}:{i}"),
                            }
                        } else {
                            meta_fail!("TriFetchVertShift({slot}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriFetchVertOff(slot) => {
                        if matches!(n, "v_mul_u32_u24" | "v_mul_lo_u32")
                            && other_const_arg(inst, v) == Some(12)
                        {
                            let w = inst.writes.first()?.1;
                            if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertOff(slot), v_facts.clone()) {
                                meta_fail!("TriFetchVertOff({slot}) mul12 clash at B{b}:{i}");
                            }
                        } else if matches!(n, "v_add_co_u32" | "v_add_nc_u32")
                            && let Some(ov) = other_val_arg(inst, v)
                        {
                            match tags.get(&ov).copied() {
                                Some(BvhTag::TriAddrLo(0)) => {
                                    let w = inst.writes.first()?.1;
                                    if !set_tag(&mut tags, &mut tag_facts, &mut work, w, BvhTag::TriFetchVertAddrLo(slot), v_facts.clone()) {
                                        meta_fail!("TriFetchVertAddrLo({slot}) clash at B{b}:{i}");
                                    }
                                    for &(wr, wv) in inst.writes.iter().skip(1) {
                                        if matches!(wr, Reg::S(_) | Reg::Scc)
                                            && !set_tag(&mut tags, &mut tag_facts, &mut work, wv, BvhTag::AddrCarry, v_facts.clone())
                                        {
                                            meta_fail!("TriFetchVertAddrLo carry clash at B{b}:{i}");
                                        }
                                    }
                                }
                                None if round < 5 => {
                                    deferred.push((v, b, i));
                                    continue;
                                }
                                _other => meta_fail!("TriFetchVertOff({slot}) add bad other={_other:?} at B{b}:{i}"),
                            }
                        } else {
                            meta_fail!("TriFetchVertOff({slot}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                    BvhTag::TriFetchVertAddrLo(slot) => {
                        if let Some((3, 0)) = flat_or_smem_load_info(inst, v) {
                            let kind = match slot {
                                0 => RayMetaKind::TriFetchBasisB,
                                1 => RayMetaKind::TriFetchBasisT,
                                2 => RayMetaKind::TriFetchZero3,
                                _ => meta_fail!("TriFetchVertAddrLo bad slot={slot}"),
                            };
                            if let Some(&prev) = loads.get(&(b, i))
                                && prev != kind
                            {
                                meta_fail!("TriFetchVertAddrLo load clash at B{b}:{i}");
                            }
                            loads.insert((b, i), kind);
                        } else {
                            meta_fail!("TriFetchVertAddrLo({slot}) used by {n} at B{b}:{i}@{:#x}", self.p.blocks[b].insts[i].machine.offset);
                        }
                    }
                }
            }
            }
            if deferred.is_empty() {
                break;
            }
            let retry = std::mem::take(&mut deferred);
            let mut pushed = HashSet::new();
            for (dv, _, _) in retry {
                if pushed.insert(dv) {
                    work.push(dv);
                }
            }
        }

        // Verify every live `AddrHi` load site ALSO had its lo word verified in `loads`!
        for (&v, &tag) in &tags {
            if matches!(tag, BvhTag::AddrHi) {
                let v_facts = tag_facts.get(&v).cloned().unwrap_or_default();
                for &(b, idx_opt) in &self.uses[v.0 as usize] {
                    if !scope.contains(&b) {
                        continue;
                    }
                    if let Some(i) = idx_opt {
                        let inst = &self.p.blocks[b].insts[i];
                        if flat_or_smem_load_hi_only(inst, v) && !loads.contains_key(&(b, i)) {
                            let mut use_check = v_facts.clone();
                            use_check.push((self.st.reach[b], true));
                            if let Some(g) = self.st.guards[b][i] {
                                use_check.push((g, true));
                            }
                            if !self.bools_meet(&use_check) {
                                continue;
                            }
                            meta_fail!("hi-only load without lo at B{b}:{i}@{:#x} tag={tag:?}", inst.machine.offset);
                        }
                    }
                }
            }
        }

        let mut out: Vec<(Site, RayMetaKind)> = loads.into_iter().collect();
        out.sort_unstable_by_key(|&(s, _)| s);
        Some(MetaAnalysisResult { loads: out })
    }

    /// Matches `CommittedTriangleVertexPositions` triples (`TriFetchBasisB`, `TriFetchBasisT`,
    /// `TriFetchZero3`) whose vertex values are consumed exclusively to compute the triangle's
    /// geometric normal (`normalize(cross(normalize?(v_a - v_b), normalize?(v_c - v_d)))` or
    /// unnormalized `cross(v_a - v_b, v_c - v_d)`), returning [`DirectNormalSub`] descriptors
    /// so lowering fetches `N` once from `SLOT_DXR_NORMALS` and substitutes `±N` directly
    /// without computing synthetic `(T, B)` bases or extra `Rsqrt`s (`ARCH-14`).
    /// Any triple whose vertices escape to other consumers falls back to paired `(T, B)`.
    fn find_direct_normals(&self, committed_meta_loads: &[(Site, RayMetaKind)]) -> Vec<DirectNormalSub> {
        let mut by_block: HashMap<usize, (Vec<Site>, Vec<Site>, Vec<Site>)> = HashMap::new();
        for &(site, kind) in committed_meta_loads {
            let entry = by_block.entry(site.0).or_default();
            match kind {
                RayMetaKind::TriFetchBasisB => entry.0.push(site),
                RayMetaKind::TriFetchBasisT => entry.1.push(site),
                RayMetaKind::TriFetchZero3 => entry.2.push(site),
                _ => {}
            }
        }
        let mut subs = Vec::new();
        for (blk, (mut bs, mut ts, mut zs)) in by_block {
            bs.sort_unstable();
            ts.sort_unstable();
            zs.sort_unstable();
            let count = bs.len().min(ts.len()).min(zs.len());
            for k in 0..count {
                if let Some(sub) = self.match_direct_normal_triple(blk, bs[k], ts[k], zs[k]) {
                    subs.push(sub);
                }
            }
        }
        subs.sort_unstable_by_key(|s| s.fetch_site);
        subs
    }

    fn match_direct_normal_triple(
        &self,
        blk: usize,
        site_b: Site,
        site_t: Site,
        site_z: Site,
    ) -> Option<DirectNormalSub> {
        let wb = &self.p.blocks[site_b.0].insts[site_b.1].writes;
        let wt = &self.p.blocks[site_t.0].insts[site_t.1].writes;
        let wz = &self.p.blocks[site_z.0].insts[site_z.1].writes;
        if wb.len() < 3 || wt.len() < 3 || wz.len() < 3 {
            return None;
        }
        let v_b = [wb[0].1, wb[1].1, wb[2].1];
        let v_t = [wt[0].1, wt[1].1, wt[2].1];
        let v_z = [wz[0].1, wz[1].1, wz[2].1];

        // Track `(t_coeff, b_coeff)` in `(T, B)` coordinates for each component value:
        // `v_b = (0, 1)`, `v_t = (1, 0)`, `v_z = (0, 0)`.
        // Also follow pure `v_mov_b32` copies (recording all intermediate values as internal).
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        struct CompExpr {
            comp: usize,
            tc: i8,
            bc: i8,
            /// Unique edge group ID (`0..3` for raw vertices `B/T/Z`, `>= 3` for computed edges).
            vec_id: usize,
            normalized: bool,
        }

        let mut comp_map: HashMap<Value, CompExpr> = HashMap::new();
        let mut internal_vals: HashSet<Value> = HashSet::new();
        for c in 0..3 {
            comp_map.insert(
                v_b[c],
                CompExpr { comp: c, tc: 0, bc: 1, vec_id: 0, normalized: false },
            );
            comp_map.insert(
                v_t[c],
                CompExpr { comp: c, tc: 1, bc: 0, vec_id: 1, normalized: false },
            );
            comp_map.insert(
                v_z[c],
                CompExpr { comp: c, tc: 0, bc: 0, vec_id: 2, normalized: false },
            );
            internal_vals.insert(v_b[c]);
            internal_vals.insert(v_t[c]);
            internal_vals.insert(v_z[c]);
        }

        fn plain_valu(inst: &Inst) -> Option<&crate::gfx10::Valu> {
            let Fields::Valu(f) = &inst.fields else {
                return None;
            };
            if f.abs == 0 && !f.clamp && f.omod == 0 && f.sdwa.is_none() && f.dpp.is_none() {
                Some(f)
            } else {
                None
            }
        }

        let signed_src = |inst: &Inst, f: &crate::gfx10::Valu, idx: usize| -> Option<(Value, i8)> {
            let v = operand_val(inst, f.src.get(idx)?)?;
            let sign = if (f.neg >> idx) & 1 != 0 { -1 } else { 1 };
            Some((v, sign))
        };

        // Propagate through `v_mov_b32` copies and collect edge subtractions `v_p - v_q`.
        let mut next_vec_id = 3usize;
        let mut edge_ids: HashMap<(i8, i8), usize> = HashMap::new();
        let mut edge_comps: HashMap<usize, [Option<Value>; 3]> = HashMap::new();
        let mut edge_dir: HashMap<usize, (i8, i8, bool)> = HashMap::new();

        // Walk instructions in `blk` (and any dominated blocks using the vertices) in RPO order.
        let mut scan_blocks: Vec<usize> = self
            .order
            .iter()
            .copied()
            .filter(|&b| self.dominates(blk, b))
            .collect();
        if scan_blocks.is_empty() {
            scan_blocks.push(blk);
        }

        for &b in &scan_blocks {
            for inst in &self.p.blocks[b].insts {
                let Some(f) = plain_valu(inst) else { continue };
                let Some(&(_, dst)) = inst.writes.first() else { continue };
                match inst.name() {
                    "v_mov_b32" if f.src.len() == 1 => {
                        if let Some((sv, sign)) = signed_src(inst, f, 0)
                            && let Some(&ce) = comp_map.get(&sv)
                        {
                            comp_map.insert(
                                dst,
                                CompExpr {
                                    comp: ce.comp,
                                    tc: ce.tc * sign,
                                    bc: ce.bc * sign,
                                    vec_id: ce.vec_id,
                                    normalized: ce.normalized,
                                },
                            );
                            internal_vals.insert(dst);
                        }
                    }
                    "v_sub_f32" | "v_subrev_f32" | "v_add_f32" if f.src.len() >= 2 => {
                        let Some((v0, s0)) = signed_src(inst, f, 0) else { continue };
                        let Some((v1, s1)) = signed_src(inst, f, 1) else { continue };
                        let (Some(&e0), Some(&e1)) = (comp_map.get(&v0), comp_map.get(&v1)) else {
                            continue;
                        };
                        if e0.comp != e1.comp || e0.vec_id >= 3 || e1.vec_id >= 3 {
                            continue;
                        }
                        let (m0, m1) = match inst.name() {
                            "v_sub_f32" => (s0, -s1),
                            "v_subrev_f32" => (-s0, s1),
                            _ => (s0, s1),
                        };
                        let tc = e0.tc * m0 + e1.tc * m1;
                        let bc = e0.bc * m0 + e1.bc * m1;
                        if tc == 0 && bc == 0 {
                            continue;
                        }
                        let vid = *edge_ids.entry((tc, bc)).or_insert_with(|| {
                            let id = next_vec_id;
                            next_vec_id += 1;
                            edge_dir.insert(id, (tc, bc, false));
                            id
                        });
                        edge_comps.entry(vid).or_default()[e0.comp] = Some(dst);
                        comp_map.insert(
                            dst,
                            CompExpr { comp: e0.comp, tc, bc, vec_id: vid, normalized: false },
                        );
                        internal_vals.insert(dst);
                    }
                    _ => {}
                }
            }
        }

        // Helper: match `normalize(vec3)` -> returns `(out_vals, out_sites, internal_set)`.
        let match_normalize = |vec_vals: [Value; 3]| -> Option<([Value; 3], [Site; 3], Vec<Value>)> {
            let mut sq_vals: [Option<Value>; 3] = [None; 3];
            let mut extra = Vec::new();
            for &b in &scan_blocks {
                for inst in &self.p.blocks[b].insts {
                    if !matches!(inst.name(), "v_mul_f32" | "v_mul_legacy_f32") {
                        continue;
                    }
                    let Some(f) = plain_valu(inst) else { continue };
                    let Some(&(_, dst)) = inst.writes.first() else { continue };
                    if let (Some((a, _)), Some((b_v, _))) =
                        (signed_src(inst, f, 0), signed_src(inst, f, 1))
                        && a == b_v
                        && let Some(c) = vec_vals.iter().position(|&x| x == a)
                    {
                        sq_vals[c] = Some(dst);
                        extra.push(dst);
                    }
                }
            }
            // Find `dot = v0^2 + v1^2 + v2^2` via `mul + fma/mad/mac/add`.
            let mut partial_mask: HashMap<Value, u8> = HashMap::new();
            for c in 0..3 {
                if let Some(sq) = sq_vals[c] {
                    partial_mask.insert(sq, 1 << c);
                }
            }
            let mut dot_vals: Vec<Value> = Vec::new();
            for &b in &scan_blocks {
                for inst in &self.p.blocks[b].insts {
                    let Some(f) = plain_valu(inst) else { continue };
                    let Some(&(_, dst)) = inst.writes.first() else { continue };
                    let n = inst.name();
                    if matches!(n, "v_fma_f32" | "v_fma_legacy_f32" | "v_mad_f32" | "v_mad_legacy_f32" | "v_fmac_f32" | "v_mac_f32")
                        && f.src.len() >= 3
                    {
                        let (Some((s0, _)), Some((s1, _)), Some((s2, _))) = (
                            signed_src(inst, f, 0),
                            signed_src(inst, f, 1),
                            signed_src(inst, f, 2),
                        ) else {
                            continue;
                        };
                        if s0 == s1
                            && let Some(c) = vec_vals.iter().position(|&x| x == s0)
                            && let Some(&m) = partial_mask.get(&s2)
                            && (m & (1 << c)) == 0
                        {
                            let nm = m | (1 << c);
                            partial_mask.insert(dst, nm);
                            extra.push(dst);
                            if nm == 0b111 {
                                dot_vals.push(dst);
                            }
                        }
                    } else if n == "v_add_f32" && f.src.len() >= 2 {
                        let (Some((s0, _)), Some((s1, _))) =
                            (signed_src(inst, f, 0), signed_src(inst, f, 1))
                        else {
                            continue;
                        };
                        if let (Some(&m0), Some(&m1)) = (partial_mask.get(&s0), partial_mask.get(&s1))
                            && (m0 & m1) == 0
                        {
                            let nm = m0 | m1;
                            partial_mask.insert(dst, nm);
                            extra.push(dst);
                            if nm == 0b111 {
                                dot_vals.push(dst);
                            }
                        }
                    } else if matches!(n, "v_max_f32" | "v_mov_b32") {
                        // Optional tiny-epsilon clamp `max(dot, eps)` before `rsq`.
                        if let Some((s0, _)) = signed_src(inst, f, 0)
                            && dot_vals.contains(&s0)
                        {
                            dot_vals.push(dst);
                            extra.push(dst);
                        } else if f.src.len() >= 2
                            && let Some((s1, _)) = signed_src(inst, f, 1)
                            && dot_vals.contains(&s1)
                        {
                            dot_vals.push(dst);
                            extra.push(dst);
                        }
                    }
                }
            }
            if dot_vals.is_empty() {
                return None;
            }
            // Find `inv = rsq(dot)` (or `rcp(sqrt(dot))`).
            let mut sqrt_vals: Vec<Value> = Vec::new();
            let mut inv_vals: Vec<Value> = Vec::new();
            for &b in &scan_blocks {
                for inst in &self.p.blocks[b].insts {
                    let Some(f) = plain_valu(inst) else { continue };
                    let Some(&(_, dst)) = inst.writes.first() else { continue };
                    let Some((s0, sgn)) = signed_src(inst, f, 0) else { continue };
                    if sgn < 0 {
                        continue;
                    }
                    match inst.name() {
                        "v_rsq_f32" if dot_vals.contains(&s0) => {
                            inv_vals.push(dst);
                            extra.push(dst);
                        }
                        "v_sqrt_f32" if dot_vals.contains(&s0) => {
                            sqrt_vals.push(dst);
                            extra.push(dst);
                        }
                        "v_rcp_f32" if sqrt_vals.contains(&s0) => {
                            inv_vals.push(dst);
                            extra.push(dst);
                        }
                        "v_mov_b32" if inv_vals.contains(&s0) => {
                            inv_vals.push(dst);
                            extra.push(dst);
                        }
                        _ => {}
                    }
                }
            }
            if inv_vals.is_empty() {
                return None;
            }
            // Match `out[c] = vec_vals[c] * inv`.
            let mut out_vals = [None; 3];
            let mut out_sites = [None; 3];
            for &b in &scan_blocks {
                for (i, inst) in self.p.blocks[b].insts.iter().enumerate() {
                    if !matches!(inst.name(), "v_mul_f32" | "v_mul_legacy_f32") {
                        continue;
                    }
                    let Some(f) = plain_valu(inst) else { continue };
                    let Some(&(_, dst)) = inst.writes.first() else { continue };
                    let (Some((s0, sg0)), Some((s1, sg1))) =
                        (signed_src(inst, f, 0), signed_src(inst, f, 1))
                    else {
                        continue;
                    };
                    if sg0 * sg1 < 0 {
                        continue;
                    }
                    let comp = if inv_vals.contains(&s1) {
                        vec_vals.iter().position(|&x| x == s0)
                    } else if inv_vals.contains(&s0) {
                        vec_vals.iter().position(|&x| x == s1)
                    } else {
                        None
                    };
                    if let Some(c) = comp
                        && out_vals[c].is_none()
                    {
                        out_vals[c] = Some(dst);
                        out_sites[c] = Some((b, i));
                    }
                }
            }
            Some((
                [out_vals[0]?, out_vals[1]?, out_vals[2]?],
                [out_sites[0]?, out_sites[1]?, out_sites[2]?],
                extra,
            ))
        };

        // Detect optional `normalize(edge)` for each complete 3-component edge vector.
        let mut candidate_vecs: Vec<(usize, [Value; 3], i8, i8, bool)> = Vec::new();
        // Raw vertices `v_b` and `v_t` can also act as edges directly when `v_z == 0` was folded!
        candidate_vecs.push((0, v_b, 0, 1, false));
        candidate_vecs.push((1, v_t, 1, 0, false));
        let edge_ids_snapshot: Vec<usize> = edge_comps.keys().copied().collect();
        for vid in edge_ids_snapshot {
            let comps = edge_comps[&vid];
            if let [Some(x), Some(y), Some(z)] = comps {
                let (tc, bc, _) = edge_dir[&vid];
                candidate_vecs.push((vid, [x, y, z], tc, bc, false));
                if let Some((n_vals, _, extra)) = match_normalize([x, y, z]) {
                    let nvid = next_vec_id;
                    next_vec_id += 1;
                    for c in 0..3 {
                        comp_map.insert(
                            n_vals[c],
                            CompExpr { comp: c, tc, bc, vec_id: nvid, normalized: true },
                        );
                        internal_vals.insert(n_vals[c]);
                    }
                    internal_vals.extend(extra);
                    candidate_vecs.push((nvid, n_vals, tc, bc, true));
                }
            }
        }

        // Match `cross(e_a, e_b)` for every linearly independent pair `(e_a, e_b)`.
        for ia in 0..candidate_vecs.len() {
            for ib in 0..candidate_vecs.len() {
                if ia == ib {
                    continue;
                }
                let (_, va, at, ab, norm_a) = candidate_vecs[ia];
                let (_, vb, bt, bb, norm_b) = candidate_vecs[ib];
                let det = i32::from(at) * i32::from(bb) - i32::from(ab) * i32::from(bt);
                if det == 0 {
                    continue;
                }
                // Record products `va[i] * vb[j]` with sign.
                let mut prod_map: HashMap<Value, (usize, usize, i8)> = HashMap::new();
                let mut cross_vals: [Option<Value>; 3] = [None; 3];
                let mut cross_sites: [Option<Site>; 3] = [None; 3];
                let mut cross_extra: Vec<Value> = Vec::new();
                for &b in &scan_blocks {
                    for (idx, inst) in self.p.blocks[b].insts.iter().enumerate() {
                        let Some(f) = plain_valu(inst) else { continue };
                        let Some(&(_, dst)) = inst.writes.first() else { continue };
                        let n = inst.name();
                        if matches!(n, "v_mul_f32" | "v_mul_legacy_f32") && f.src.len() >= 2 {
                            let (Some((s0, sg0)), Some((s1, sg1))) =
                                (signed_src(inst, f, 0), signed_src(inst, f, 1))
                            else {
                                continue;
                            };
                            if let (Some(ca), Some(cb)) = (
                                va.iter().position(|&x| x == s0),
                                vb.iter().position(|&x| x == s1),
                            ) && ca != cb
                            {
                                prod_map.insert(dst, (ca, cb, sg0 * sg1));
                                cross_extra.push(dst);
                            } else if let (Some(cb), Some(ca)) = (
                                vb.iter().position(|&x| x == s0),
                                va.iter().position(|&x| x == s1),
                            ) && ca != cb
                            {
                                prod_map.insert(dst, (ca, cb, sg0 * sg1));
                                cross_extra.push(dst);
                            }
                        } else if matches!(
                            n,
                            "v_fma_f32" | "v_fma_legacy_f32" | "v_mad_f32" | "v_mad_legacy_f32"
                        ) && f.src.len() >= 3
                        {
                            let (Some((s0, sg0)), Some((s1, sg1)), Some((s2, sg2))) = (
                                signed_src(inst, f, 0),
                                signed_src(inst, f, 1),
                                signed_src(inst, f, 2),
                            ) else {
                                continue;
                            };
                            let Some(&(pa, pb, psg)) = prod_map.get(&s2) else { continue };
                            let fma_ab = if let (Some(ca), Some(cb)) = (
                                va.iter().position(|&x| x == s0),
                                vb.iter().position(|&x| x == s1),
                            ) {
                                Some((ca, cb, sg0 * sg1))
                            } else if let (Some(cb), Some(ca)) = (
                                vb.iter().position(|&x| x == s0),
                                va.iter().position(|&x| x == s1),
                            ) {
                                Some((ca, cb, sg0 * sg1))
                            } else {
                                None
                            };
                            let Some((fa, fb, fsg)) = fma_ab else { continue };
                            let add_sg = psg * sg2;
                            for k in 0..3 {
                                let (ci, cj) = ((k + 1) % 3, (k + 2) % 3);
                                // `va[ci]*vb[cj] - va[cj]*vb[ci]`
                                let ok = (fa == ci && fb == cj && fsg == 1 && pa == cj && pb == ci && add_sg == -1)
                                    || (fa == cj && fb == ci && fsg == -1 && pa == ci && pb == cj && add_sg == 1);
                                if ok && cross_vals[k].is_none() {
                                    cross_vals[k] = Some(dst);
                                    cross_sites[k] = Some((b, idx));
                                }
                            }
                        } else if matches!(n, "v_sub_f32" | "v_subrev_f32" | "v_add_f32")
                            && f.src.len() >= 2
                        {
                            let (Some((s0, sg0)), Some((s1, sg1))) =
                                (signed_src(inst, f, 0), signed_src(inst, f, 1))
                            else {
                                continue;
                            };
                            let (Some(&(a0, b0, p0)), Some(&(a1, b1, p1))) =
                                (prod_map.get(&s0), prod_map.get(&s1))
                            else {
                                continue;
                            };
                            let (m0, m1) = match n {
                                "v_sub_f32" => (p0 * sg0, -p1 * sg1),
                                "v_subrev_f32" => (-p0 * sg0, p1 * sg1),
                                _ => (p0 * sg0, p1 * sg1),
                            };
                            for k in 0..3 {
                                let (ci, cj) = ((k + 1) % 3, (k + 2) % 3);
                                let ok = (a0 == ci && b0 == cj && m0 == 1 && a1 == cj && b1 == ci && m1 == -1)
                                    || (a0 == cj && b0 == ci && m0 == -1 && a1 == ci && b1 == cj && m1 == 1);
                                if ok && cross_vals[k].is_none() {
                                    cross_vals[k] = Some(dst);
                                    cross_sites[k] = Some((b, idx));
                                }
                            }
                        }
                    }
                }
                let [Some(cx), Some(cy), Some(cz)] = cross_vals else { continue };
                let [Some(csx), Some(csy), Some(csz)] = cross_sites else { continue };

                // Check for trailing `normalize(cross)` (always exact `sign(det) * N`),
                // or accept unnormalized `cross` when neither edge was normalized (`|det| == 1`).
                let (out_vals, out_sites, mut consumed) =
                    if let Some((nv, ns, norm_extra)) = match_normalize([cx, cy, cz]) {
                        let mut c = internal_vals.clone();
                        c.extend(cross_extra);
                        c.extend([cx, cy, cz]);
                        c.extend(norm_extra);
                        (nv, ns, c)
                    } else if !norm_a && !norm_b && det.abs() == 1 {
                        let mut c = internal_vals.clone();
                        c.extend(cross_extra);
                        ([cx, cy, cz], [csx, csy, csz], c)
                    } else {
                        continue;
                    };
                consumed.extend(out_vals);

                // Exact backward slice from `out_vals` through `consumed` back to `raw_vertices`:
                // excludes unrelated arithmetic (e.g. `v_t + v_t`) that never feeds `out_vals`.
                let raw_vertices = [
                    v_b[0], v_b[1], v_b[2], v_t[0], v_t[1], v_t[2], v_z[0], v_z[1], v_z[2],
                ];
                let raw_set: HashSet<Value> = raw_vertices.into_iter().collect();
                let mut slice: HashSet<Value> = out_vals.into_iter().collect();
                let mut work: Vec<Value> = out_vals.to_vec();
                while let Some(v) = work.pop() {
                    if raw_set.contains(&v) {
                        continue;
                    }
                    if let Some(Def::Inst(db, di)) = self.defs.get(v.0 as usize).copied().flatten() {
                        for &(_, rv) in &self.p.blocks[db].insts[di].reads {
                            if (consumed.contains(&rv) || raw_set.contains(&rv)) && slice.insert(rv) {
                                work.push(rv);
                            }
                        }
                    }
                }
                // Safety check: EVERY user of every raw vertex AND every intermediate value in
                // `slice \ out_vals` must write a value inside `slice` (zero escapes!).
                let out_set: HashSet<Value> = out_vals.into_iter().collect();
                let escapes = raw_vertices.iter().copied().chain(slice.iter().copied()).any(|v| {
                    if out_set.contains(&v) {
                        return false;
                    }
                    self.uses[v.0 as usize].iter().any(|&(ub, ui_opt)| {
                        let Some(ui) = ui_opt else { return true };
                        let uinst = &self.p.blocks[ub].insts[ui];
                        uinst.writes.is_empty()
                            || uinst.writes.iter().any(|&(_, wv)| !slice.contains(&wv))
                    })
                });
                if escapes {
                    continue;
                }

                // Verify `fetch_site` dominates every `out_site`.
                let fetch_site = site_b.min(site_t).min(site_z);
                if !out_sites.iter().all(|&os| {
                    self.dominates(fetch_site.0, os.0)
                        && (fetch_site.0 != os.0 || fetch_site.1 < os.1)
                }) {
                    continue;
                }
                return Some(DirectNormalSub {
                    fetch_site,
                    dead_fetch_sites: [site_b, site_t, site_z],
                    out_vals,
                    out_sites,
                    negate: det < 0,
                });
            }
        }
        None
    }

    fn extract_base_low_offset(&self, v: Value, depth: u32) -> Option<u32> {
        if depth > 6 {
            return None;
        }
        let Some((_, inst)) = self.def_inst(v) else {
            return Some(0);
        };
        match inst.name() {
            "s_mov_b32" | "v_mov_b32" => single_reg_src(inst)
                .map_or(Some(0), |s| self.extract_base_low_offset(s, depth + 1)),
            "s_and_b32" | "v_and_b32" => {
                // `tlas_lo & 0xffffff00` -> 256-byte aligned, offset 0!
                if has_const_operand(inst, 0xFFFF_FF00) {
                    Some(0)
                } else {
                    Some(0)
                }
            }
            "s_or_b32" | "v_or_b32" => {
                if let Some(k) = other_const_arg_any(inst)
                    && k < 128
                {
                    return Some(k);
                }
                Some(0)
            }
            "s_add_u32" | "s_add_i32" | "v_add_nc_u32" | "v_add_co_u32" => {
                if let Some(k) = other_const_arg_any(inst)
                    && k < 128
                {
                    return Some(k);
                }
                Some(0)
            }
            _ => Some(0),
        }
    }

    /// Collects `(Bool, true)` predicates provably implied by `root != 0` for post-loop
    /// committed node pointers (matching XDXR's `status = (node != 0) ? (1 or 2) : 0; if (status != 0 || timeout)`).
    fn collect_hit_status_facts(
        &self,
        roots: &HashSet<Value>,
        scope: &HashSet<usize>,
        out: &mut Vec<(Bool, bool)>,
    ) {
        let mut nonzero_vals: HashSet<Value> = roots.clone();
        let mut true_masks: HashSet<Value> = HashSet::new();
        let mut work_nz: Vec<Value> = roots.iter().copied().collect();
        let mut work_mask: Vec<Value> = Vec::new();

        for _ in 0..6 {
            while let Some(v) = work_nz.pop() {
                for &(b, idx_opt) in &self.uses[v.0 as usize] {
                    if !scope.contains(&b) {
                        continue;
                    }
                    let Some(i) = idx_opt else {
                        // Step through a merge PHI (e.g. `exit_block` joining loop hit with bypass)
                        // so `v_cmp_ne_u32 _, 0, phi_tri_node` is recognised!
                        for phi in &self.p.blocks[b].phis {
                            if phi.args.contains(&v) && nonzero_vals.insert(phi.value) {
                                work_nz.push(phi.value);
                            }
                        }
                        continue;
                    };
                    let inst = &self.p.blocks[b].insts[i];
                    if matches!(inst.name(), "v_mov_b32" | "s_mov_b32")
                        && single_reg_src(inst) == Some(v)
                        && let Some(&(_, w)) = inst.writes.first()
                        && nonzero_vals.insert(w)
                    {
                        work_nz.push(w);
                        continue;
                    }
                    if matches!(inst.name(), "v_cmp_ne_u32" | "v_cmp_ne_i32" | "v_cmpx_ne_u32")
                        && other_const_arg(inst, v) == Some(0)
                        && let Some(&(_, mw)) = inst.writes.first()
                        && let Some(&mb) = self.st.masks.get(&mw)
                        && true_masks.insert(mw)
                    {
                        if mb != Bool::TRUE && !out.contains(&(mb, true)) {
                            out.push((mb, true));
                        }
                        work_mask.push(mw);
                    }
                }
            }
            while let Some(mv) = work_mask.pop() {
                for &(b, idx_opt) in &self.uses[mv.0 as usize] {
                    if !scope.contains(&b) {
                        continue;
                    }
                    let Some(i) = idx_opt else { continue };
                    let inst = &self.p.blocks[b].insts[i];
                    match inst.name() {
                        "s_or_b32" | "s_or_b64" => {
                            if let Some(&(_, w)) = inst.writes.first()
                                && let Some(&mb) = self.st.masks.get(&w)
                                && true_masks.insert(w)
                            {
                                if mb != Bool::TRUE && !out.contains(&(mb, true)) {
                                    out.push((mb, true));
                                }
                                work_mask.push(w);
                            }
                        }
                        "s_and_b32" | "s_and_b64" => {
                            // `s_and(tri_node != 0, inst_node != 0)` is true whenever a hit occurred!
                            if let Some((lhs, rhs)) = two_src_operands(inst)
                                && operand_val(inst, lhs).is_some_and(|a| true_masks.contains(&a))
                                && operand_val(inst, rhs).is_some_and(|b| true_masks.contains(&b))
                                && let Some(&(_, w)) = inst.writes.first()
                                && let Some(&mb) = self.st.masks.get(&w)
                                && true_masks.insert(w)
                            {
                                if mb != Bool::TRUE && !out.contains(&(mb, true)) {
                                    out.push((mb, true));
                                }
                                work_mask.push(w);
                            }
                        }
                        "v_cndmask_b32" => {
                            if let Fields::Valu(f) = &inst.fields
                                && f.src.len() >= 3
                                && operand_val(inst, &f.src[2]) == Some(mv)
                                && const_operand(&f.src[0]) == Some(0)
                                && self.is_definitely_nonzero_operand(inst, &f.src[1])
                                && let Some(&(_, w)) = inst.writes.first()
                                && nonzero_vals.insert(w)
                            {
                                work_nz.push(w);
                            }
                        }
                        _ => {}
                    }
                }
            }
            if work_nz.is_empty() && work_mask.is_empty() {
                break;
            }
        }
    }

    fn is_definitely_nonzero_operand(&self, inst: &Inst, op: &Operand) -> bool {
        if let Some(k) = const_operand(op) {
            return k != 0;
        }
        let Some(v) = operand_val(inst, op) else {
            return false;
        };
        let Some((_, di)) = self.def_inst(v) else {
            return false;
        };
        if di.name() == "v_cndmask_b32"
            && let Fields::Valu(f) = &di.fields
            && f.src.len() >= 2
        {
            return const_operand(&f.src[0]).is_some_and(|k| k != 0)
                && const_operand(&f.src[1]).is_some_and(|k| k != 0);
        }
        false
    }

    /// Returns `true` if `incoming_v` (defined in `def_b`) can actually reach a real operand
    /// read of `phi_val` inside `scope` (checking lane disjointness via `st.bools.meet`,
    /// tracking `phi_masks`, negated write guards, and waterfall-loop kills).
    fn phi_incoming_reaches_use_in_scope(
        &self,
        incoming_v: Value,
        phi_val: Value,
        extra_facts: &[(Bool, bool)],
        scope: &HashSet<usize>,
        skipped_tail: Option<(usize, &HashSet<usize>)>,
    ) -> bool {
        let def_reach = self
            .def_block(incoming_v)
            .map(|db| self.st.reach[db])
            .unwrap_or(Bool::TRUE);
        let mut init_facts: Vec<(Bool, bool)> = extra_facts.to_vec();
        if def_reach != Bool::TRUE && !init_facts.contains(&(def_reach, true)) {
            init_facts.push((def_reach, true));
        }
        if let Some(Def::Phi(pb, pi)) = self.defs.get(phi_val.0 as usize).copied().flatten() {
            let phi = &self.p.blocks[pb].phis[pi];
            let mut any_reaches = false;
            if let Some(margs) = self.st.phi_masks.get(&phi_val) {
                for (&arg, &mb) in phi.args.iter().zip(margs) {
                    if arg != incoming_v {
                        continue;
                    }
                    let mut f = init_facts.clone();
                    if mb != Bool::TRUE && !f.contains(&(mb, true)) {
                        f.push((mb, true));
                    }
                    if self.bools_meet(&f)
                        && self.value_reaches_op_use_under(phi_val, f, scope, skipped_tail)
                    {
                        any_reaches = true;
                        break;
                    }
                }
            } else {
                for (&pred, &arg) in self.p.blocks[pb].preds.iter().zip(&phi.args) {
                    if arg != incoming_v {
                        continue;
                    }
                    let f = self.edge_facts(pred, pb, &init_facts);
                    if self.bools_meet(&f)
                        && self.value_reaches_op_use_under(phi_val, f, scope, skipped_tail)
                    {
                        any_reaches = true;
                        break;
                    }
                }
            }
            return any_reaches;
        }
        self.value_reaches_op_use_under(phi_val, init_facts, scope, skipped_tail)
    }

    fn value_reaches_op_use_under(
        &self,
        root: Value,
        init_facts: Vec<(Bool, bool)>,
        scope: &HashSet<usize>,
        skipped_tail: Option<(usize, &HashSet<usize>)>,
    ) -> bool {
        if !self.bools_meet(&init_facts) {
            return false;
        }
        let mut seen: HashSet<(Value, Vec<(Bool, bool)>)> = HashSet::new();
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> = vec![(root, init_facts)];
        let mut steps = 0usize;
        while let Some((v, facts)) = work.pop() {
            steps += 1;
            if steps > 256 {
                return true; // Conservative fail-safe on huge graphs.
            }
            if !seen.insert((v, facts.clone())) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if !phi.args.contains(&v) {
                            continue;
                        }
                        if let Some(margs) = self.st.phi_masks.get(&phi.value) {
                            for (&arg, &mb) in phi.args.iter().zip(margs) {
                                if arg != v {
                                    continue;
                                }
                                let mut nf = facts.clone();
                                if mb != Bool::TRUE && !nf.contains(&(mb, true)) {
                                    nf.push((mb, true));
                                }
                                if self.bools_meet(&nf) {
                                    work.push((phi.value, nf));
                                }
                            }
                        } else {
                            for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                                if arg != v {
                                    continue;
                                }
                                let nf = self.edge_facts(pred, b, &facts);
                                if self.bools_meet(&nf) {
                                    work.push((phi.value, nf));
                                }
                            }
                        }
                    }
                    continue;
                };
                if let Some((tb, sk)) = skipped_tail
                    && b == tb
                    && sk.contains(&i)
                {
                    continue;
                }
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                if inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                    let mut check = facts.clone();
                    check.push((self.st.reach[b], true));
                    if let Some(g) = self.st.guards[b][i] {
                        check.push((g, true));
                    } else if inst.name().starts_with("v_cmp_") {
                        // `v_cmp_*` has no `st.guards[b][i]` in `structure.rs`, even when it sits
                        // inside an `s_and_saveexec` / `v_cmpx` sub-scope of block `b`. Find any
                        // guarded VALU instruction in the exact same EXEC sub-scope of `b` to
                        // recover `exec_at_inst`!
                        if let Some(eg) = self.same_exec_scope_guard(b, i) {
                            check.push((eg, true));
                        }
                    }
                    if self.bools_meet(&check) {
                        return true;
                    }
                    continue;
                }
                if let Some(g) = self.st.guards[b][i] {
                    let kill_cond = if self.is_waterfall_kill_block(b, v) {
                        self.st.reach[b]
                    } else {
                        g
                    };
                    let mut nf = facts.clone();
                    if !nf.contains(&(kill_cond, false)) {
                        nf.push((kill_cond, false));
                    }
                    if self.bools_meet(&nf) {
                        for &(r, rv) in &inst.reads {
                            if rv == v
                                && let Some(w) = inst.write(r)
                            {
                                work.push((w, nf.clone()));
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Returns the guard `Bool` of the EXEC sub-scope containing `(b, i)`, found by scanning
    /// neighbor instructions in `b` without crossing any instruction that writes `EXEC`.
    fn same_exec_scope_guard(&self, b: usize, i: usize) -> Option<Bool> {
        let insts = &self.p.blocks[b].insts;
        let writes_exec = |inst: &Inst| {
            inst.writes
                .iter()
                .any(|&(r, _)| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1))
        };
        for j in (0..i).rev() {
            if writes_exec(&insts[j]) {
                break;
            }
            if let Some(g) = self.st.guards[b][j] {
                return Some(g);
            }
        }
        for j in (i + 1)..insts.len() {
            if writes_exec(&insts[j]) {
                break;
            }
            if let Some(g) = self.st.guards[b][j] {
                return Some(g);
            }
        }
        None
    }

    fn reaches_global_or_smem_load(&self, root: Value, scope: &HashSet<usize>) -> bool {
        let mut seen = HashSet::new();
        let mut work = vec![(root, 0u32)];
        while let Some((v, depth)) = work.pop() {
            if depth > 16 || !seen.insert(v) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if phi.args.contains(&v) {
                            work.push((phi.value, depth + 1));
                        }
                    }
                    continue;
                };
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                if !inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                    if self.st.guards[b][i].is_some() {
                        for &(r, rv) in &inst.reads {
                            if rv == v
                                && let Some(w) = inst.write(r)
                            {
                                work.push((w, depth + 1));
                            }
                        }
                    }
                    continue;
                }
                let n = inst.name();
                if n.starts_with("global_load_") || n.starts_with("s_load_") {
                    return true;
                }
                if matches!(
                    n,
                    "v_mov_b32"
                        | "s_mov_b32"
                        | "v_or_b32"
                        | "s_or_b32"
                        | "v_add_co_u32"
                        | "v_add_co_ci_u32"
                        | "v_add_nc_u32"
                        | "s_add_u32"
                        | "s_addc_u32"
                ) {
                    if let Some(&(_, w)) = inst.writes.first() {
                        work.push((w, depth + 1));
                    }
                }
            }
        }
        false
    }

    /// Verifies that every user of an `inst_addr + 48` (`InstanceID`) or `+ 52` (`InstanceContribution`)
    /// load reads ONLY bits `23:0` (never the upper byte `InstanceMask` / `InstanceFlags`).
    fn uses_only_low_24_bits(&self, root: Value, scope: &HashSet<usize>) -> bool {
        let def_reach = self
            .def_block(root)
            .map(|db| self.st.reach[db])
            .unwrap_or(Bool::TRUE);
        let init_facts = if def_reach == Bool::TRUE {
            Vec::new()
        } else {
            vec![(def_reach, true)]
        };
        let mut seen: HashSet<(Value, Vec<(Bool, bool)>, u32)> = HashSet::new();
        let mut work: Vec<(Value, Vec<(Bool, bool)>, u32)> = vec![(root, init_facts, 24)];
        let mut steps = 0usize;
        while let Some((v, facts, max_bits)) = work.pop() {
            steps += 1;
            if steps > 256 {
                return false;
            }
            if !seen.insert((v, facts.clone(), max_bits)) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if !phi.args.contains(&v) {
                            continue;
                        }
                        for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                            if arg != v {
                                continue;
                            }
                            let nf = self.edge_facts(pred, b, &facts);
                            if self.bools_meet(&nf) {
                                work.push((phi.value, nf, max_bits));
                            }
                        }
                    }
                    continue;
                };
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                if !inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                    if let Some(g) = self.st.guards[b][i] {
                        let kill_cond = if self.is_waterfall_kill_block(b, v) {
                            self.st.reach[b]
                        } else {
                            g
                        };
                        let mut nf = facts.clone();
                        if !nf.contains(&(kill_cond, false)) {
                            nf.push((kill_cond, false));
                        }
                        if self.bools_meet(&nf) {
                            for &(r, rv) in &inst.reads {
                                if rv == v
                                    && let Some(w) = inst.write(r)
                                {
                                    work.push((w, nf.clone(), max_bits));
                                }
                            }
                        }
                    }
                    continue;
                }
                let mut check = facts.clone();
                check.push((self.st.reach[b], true));
                if let Some(g) = self.st.guards[b][i] {
                    check.push((g, true));
                } else if inst.name().starts_with("v_cmp_")
                    && let Some(eg) = self.same_exec_scope_guard(b, i)
                {
                    check.push((eg, true));
                }
                if !self.bools_meet(&check) {
                    continue;
                }
                match inst.name() {
                    "v_mov_b32"
                    | "s_mov_b32"
                    | "v_readfirstlane_b32"
                    | "v_add_nc_u32"
                    | "v_add_co_u32"
                    | "s_add_u32"
                    | "s_add_i32"
                    | "v_sub_nc_u32"
                    | "v_or_b32"
                    | "s_or_b32"
                    | "v_xor_b32"
                    | "s_xor_b32" => {
                        // Modular add/sub/bitwise ops depend only on bits <= k for output bits <= k,
                        // so check that all downstream uses mask to <= `max_bits` bits!
                        if let Some(&(_, w)) = inst.writes.first() {
                            work.push((w, facts.clone(), max_bits));
                        }
                    }
                    "v_lshrrev_b32" | "s_lshr_b32" => {
                        let Some((s0, s1)) = two_src_operands(inst) else {
                            return false;
                        };
                        let (shift_op, val_op) = if inst.name() == "v_lshrrev_b32" {
                            (s0, s1)
                        } else {
                            (s1, s0)
                        };
                        let Some(sh) = const_operand(shift_op) else {
                            return false;
                        };
                        if operand_val(inst, val_op) != Some(v) || sh >= max_bits {
                            return false;
                        }
                        if let Some(&(_, w)) = inst.writes.first() {
                            work.push((w, facts.clone(), max_bits - sh));
                        }
                    }
                    "v_and_b32" | "s_and_b32" => {
                        let k = other_const_arg(inst, v).or_else(|| {
                            other_val_arg(inst, v).and_then(|ov| self.eval_const(ov, 0))
                        });
                        let Some(k) = k else {
                            return false;
                        };
                        let allowed_mask = (1u32 << max_bits) - 1;
                        if k > allowed_mask {
                            return false;
                        }
                    }
                    "v_bfe_u32" | "v_bfe_i32" => {
                        let Fields::Valu(f) = &inst.fields else {
                            return false;
                        };
                        if f.src.len() < 3 || operand_val(inst, &f.src[0]) != Some(v) {
                            return false;
                        }
                        let (Some(sh), Some(w)) = (const_operand(&f.src[1]), const_operand(&f.src[2]))
                        else {
                            return false;
                        };
                        if sh + w > max_bits {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// Verifies that every user of a `tri_addr + 48` (`(GeometryIndex << 8) | flags`) load
    /// only reads bits `31:8` (via `>> 8` or `v_bfe_u32(_, v, 8, ...)`).
    fn uses_only_shr_8(&self, root: Value, scope: &HashSet<usize>) -> bool {
        let def_reach = self
            .def_block(root)
            .map(|db| self.st.reach[db])
            .unwrap_or(Bool::TRUE);
        let init_facts = if def_reach == Bool::TRUE {
            Vec::new()
        } else {
            vec![(def_reach, true)]
        };
        let mut seen: HashSet<(Value, Vec<(Bool, bool)>)> = HashSet::new();
        let mut work: Vec<(Value, Vec<(Bool, bool)>)> = vec![(root, init_facts)];
        let mut steps = 0usize;
        while let Some((v, facts)) = work.pop() {
            steps += 1;
            if steps > 256 {
                return false;
            }
            if !seen.insert((v, facts.clone())) {
                continue;
            }
            for &(b, idx_opt) in &self.uses[v.0 as usize] {
                if !scope.contains(&b) {
                    continue;
                }
                let Some(i) = idx_opt else {
                    for phi in &self.p.blocks[b].phis {
                        if !phi.args.contains(&v) {
                            continue;
                        }
                        for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                            if arg != v {
                                continue;
                            }
                            let nf = self.edge_facts(pred, b, &facts);
                            if self.bools_meet(&nf) {
                                work.push((phi.value, nf));
                            }
                        }
                    }
                    continue;
                };
                let inst = &self.p.blocks[b].insts[i];
                let ops = operand_reads(self.p, inst);
                if !inst.reads.iter().any(|&(r, rv)| rv == v && ops.contains(&r)) {
                    if let Some(g) = self.st.guards[b][i] {
                        let kill_cond = if self.is_waterfall_kill_block(b, v) {
                            self.st.reach[b]
                        } else {
                            g
                        };
                        let mut nf = facts.clone();
                        if !nf.contains(&(kill_cond, false)) {
                            nf.push((kill_cond, false));
                        }
                        if self.bools_meet(&nf) {
                            for &(r, rv) in &inst.reads {
                                if rv == v
                                    && let Some(w) = inst.write(r)
                                {
                                    work.push((w, nf.clone()));
                                }
                            }
                        }
                    }
                    continue;
                }
                let mut check = facts.clone();
                check.push((self.st.reach[b], true));
                if let Some(g) = self.st.guards[b][i] {
                    check.push((g, true));
                } else if inst.name().starts_with("v_cmp_")
                    && let Some(eg) = self.same_exec_scope_guard(b, i)
                {
                    check.push((eg, true));
                }
                if !self.bools_meet(&check) {
                    continue;
                }
                match inst.name() {
                    "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" => {
                        if let Some(&(_, w)) = inst.writes.first() {
                            work.push((w, facts.clone()));
                        }
                    }
                    "v_lshrrev_b32" | "s_lshr_b32" => {
                        let Some(sh) = other_const_arg(inst, v) else {
                            return false;
                        };
                        if sh < 8 {
                            return false;
                        }
                    }
                    "v_bfe_u32" | "v_bfe_i32" => {
                        let Fields::Valu(f) = &inst.fields else {
                            return false;
                        };
                        if f.src.len() < 3 || operand_val(inst, &f.src[0]) != Some(v) {
                            return false;
                        }
                        let Some(sh) = const_operand(&f.src[1]) else {
                            return false;
                        };
                        if sh < 8 {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// Recognises XDXR's per-iteration counter (`iter = 0; iter += 1; cmp iter, 0xffff`)
    /// and timeout flag (`iter > 0xfffe ? -1 : 0`).
    fn is_iter_counter_or_timeout(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
        depth: u32,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        let Some(def) = self.defs.get(v.0 as usize).copied().flatten() else {
            return false;
        };
        match def {
            Def::Input(_) => false,
            Def::Phi(b, pi) => {
                if !nodes.contains(&b) {
                    return false;
                }
                let phi = &self.p.blocks[b].phis[pi];
                // Pre-loop initialiser must be 0, and every in-loop back-edge must be
                // `phi + 1` (or a copy/inner-phi of `phi + 1`).
                let mut saw_zero = false;
                let mut saw_inc = false;
                for (&pred, &arg) in self.p.blocks[b].preds.iter().zip(&phi.args) {
                    if arg == v {
                        continue;
                    }
                    if b == entry_block && !nodes.contains(&pred) {
                        if self.eval_const(arg, 0) == Some(0) {
                            saw_zero = true;
                        } else {
                            return false;
                        }
                    } else if self.is_increment_of(arg, v, nodes, &mut HashSet::new(), 0) {
                        saw_inc = true;
                    } else if self.is_iter_counter_or_timeout(arg, nodes, entry_block, depth + 1) {
                        saw_inc = true;
                    } else {
                        return false;
                    }
                }
                if b == entry_block {
                    saw_zero && saw_inc
                } else {
                    saw_inc
                }
            }
            Def::Inst(b, i) => {
                if !nodes.contains(&b) {
                    return false;
                }
                let inst = &self.p.blocks[b].insts[i];
                match inst.name() {
                    "v_mov_b32" | "s_mov_b32" => single_reg_src(inst)
                        .is_some_and(|s| self.is_iter_counter_or_timeout(s, nodes, entry_block, depth + 1)),
                    "v_add_nc_u32" | "v_add_co_u32" | "s_add_i32" | "s_add_u32" => {
                        other_const_arg_any(inst) == Some(1)
                            && other_val_from_const_op(inst, 1).is_some_and(|base| {
                                self.is_iter_counter_or_timeout(base, nodes, entry_block, depth + 1)
                            })
                    }
                    // Timeout flag: `v_cndmask_b32 dst, 0, -1, cond` where `cond` compares an iter counter.
                    "v_cndmask_b32" => {
                        let Fields::Valu(f) = &inst.fields else {
                            return false;
                        };
                        if f.src.len() < 3 {
                            return false;
                        }
                        let c0 = const_operand(&f.src[0]);
                        let c1 = const_operand(&f.src[1]);
                        matches!((c0, c1), (Some(0), Some(0xFFFF_FFFF)) | (Some(0xFFFF_FFFF), Some(0)))
                            && operand_val(inst, &f.src[2]).is_some_and(|cond| {
                                self.is_timeout_cond(cond, nodes, entry_block, 0)
                            })
                    }
                    _ => false,
                }
            }
        }
    }

    fn is_timeout_cond(
        &self,
        v: Value,
        nodes: &HashSet<usize>,
        entry_block: usize,
        depth: u32,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        let Some(((b, _), inst)) = self.def_inst(v) else {
            return false;
        };
        if !nodes.contains(&b) {
            return false;
        }
        match inst.name() {
            "s_cmpk_gt_u32" | "s_cmpk_lt_u32" | "s_cmpk_ge_u32" | "s_cmpk_le_u32"
            | "s_cmp_gt_u32" | "s_cmp_lt_u32" | "s_cmp_ge_u32" | "s_cmp_le_u32"
            | "v_cmp_gt_u32" | "v_cmp_lt_u32" | "v_cmp_ge_u32" | "v_cmp_le_u32" => {
                inst.reads.iter().any(|&(_, rv)| {
                    self.is_iter_counter_or_timeout(rv, nodes, entry_block, 0)
                })
            }
            "s_cselect_b32" | "s_cselect_b64" => {
                inst.read(Reg::Scc)
                    .is_some_and(|scc| self.is_timeout_cond(scc, nodes, entry_block, depth + 1))
            }
            "v_and_b32" | "s_and_b32" => {
                other_val_from_const_op(inst, 1)
                    .is_some_and(|s| self.is_timeout_cond(s, nodes, entry_block, depth + 1))
            }
            "v_cmp_eq_u32" | "v_cmp_ne_u32" | "s_cmp_eq_u32" | "s_cmp_lg_u32"
            | "s_cmpk_eq_u32" | "s_cmpk_lg_u32" => {
                inst.reads
                    .iter()
                    .any(|&(_, rv)| self.is_iter_counter_or_timeout(rv, nodes, entry_block, 0))
                    || (other_val_from_const_op(inst, 1).or_else(|| other_val_from_const_op(inst, 0)))
                        .is_some_and(|s| self.is_timeout_cond(s, nodes, entry_block, depth + 1))
            }
            _ => false,
        }
    }

    fn is_increment_of(
        &self,
        v: Value,
        target: Value,
        nodes: &HashSet<usize>,
        visited: &mut HashSet<Value>,
        depth: u32,
    ) -> bool {
        if depth > 16 {
            return false;
        }
        if v == target || !visited.insert(v) {
            return true;
        }
        match self.defs.get(v.0 as usize).copied().flatten() {
            Some(Def::Inst(b, i)) if nodes.contains(&b) => {
                let inst = &self.p.blocks[b].insts[i];
                match inst.name() {
                    "v_mov_b32" | "s_mov_b32" => single_reg_src(inst)
                        .is_some_and(|s| self.is_increment_of(s, target, nodes, visited, depth + 1)),
                    "v_add_nc_u32" | "v_add_co_u32" | "s_add_i32" | "s_add_u32" => {
                        other_const_arg_any(inst) == Some(1)
                            && other_val_from_const_op(inst, 1)
                                .is_some_and(|s| self.is_increment_of(s, target, nodes, visited, depth + 1))
                    }
                    "v_cndmask_b32" => {
                        let Fields::Valu(f) = &inst.fields else {
                            return false;
                        };
                        f.src.len() >= 2
                            && operand_val(inst, &f.src[0])
                                .is_some_and(|s| self.is_increment_of(s, target, nodes, visited, depth + 1))
                            && operand_val(inst, &f.src[1])
                                .is_some_and(|s| self.is_increment_of(s, target, nodes, visited, depth + 1))
                    }
                    _ => false,
                }
            }
            Some(Def::Phi(b, pi)) if nodes.contains(&b) => self.p.blocks[b].phis[pi]
                .args
                .iter()
                .all(|&a| a == v || self.is_increment_of(a, target, nodes, visited, depth + 1)),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BvhTag {
    InstNode,
    InstNodeShr3,
    InstByteOff(u32),
    InstAddrLo(u32),
    AddrHi,
    AddrCarry,
    BlasBaseShr8,
    BlasBaseLo,
    BlasBaseHdrLo(u32),
    TriNode,
    TriNodeShr3,
    TriByteOff(u32),
    TriSlot,
    TriSlotShl2,
    TriSlotCmp,
    TriPrimOff,
    TriByteOffPrim,
    TriAddrLo(u32),
    TriAddrPrimLo,
    TriFetchScratch,
    TriFetchTriId,
    TriFetchVertShift(u8),
    TriFetchVertOff(u8),
    TriFetchVertAddrLo(u8),
    StatusBits,
}

struct MetaAnalysisResult {
    loads: Vec<(Site, RayMetaKind)>,
}

enum HitMode {
    Opaque,
    FrontBackTrack {
        init_min_front_t: RayInput,
        min_front_t_reg: Reg,
    },
    AnyHit(AnyHitInfo),
}

struct TriangleHitInfo {
    hit_block: usize,
    opacity_block: Option<usize>,
    accept_first_hit: bool,
    t_min: Option<RayInput>,
    committed_t_reg: Reg,
    committed_tri_reg: Option<Reg>,
    committed_inst_reg: Option<Reg>,
    committed_bary_u_reg: Option<Reg>,
    committed_bary_v_reg: Option<Reg>,
    committed_front_face_reg: Option<Reg>,
    mode: HitMode,
}

// ---------------------------------------------------------------------------
// DXIL RayQuery emission pass
// ---------------------------------------------------------------------------

/// Reads a bounds-clamped `u32` word from the `SLOT_DXR_NORMALS` root SRV.
fn raw_load_normals_u32(s: &mut Shader, block: usize, srv: V, word_idx: V, max_word: u32) -> V {
    let i32t = s.t.i32;
    let last = s.m.i32(max_word);
    let clamped = s.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[word_idx, last]);
    s.m.tag(clamped, Tag::Clamp);
    let two = s.m.i32(2);
    let byte = s.m.bin(block, Bin::Shl, clamped, two);
    let iu = s.m.undef(i32t);
    let mask = s.m.i8(1);
    let align = s.m.i32(4);
    let r = s.op(
        block,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        139,
        &[srv, byte, iu, mask, align],
    );
    let f = s.m.extract(block, r, 0);
    s.m.cast(block, Cast::Bitcast, f, i32t)
}

/// Fetches the octahedral-packed object-space geometric normal `N = [nx, ny, nz]` (`f32`)
/// from `SLOT_DXR_NORMALS` for `(tlas_meta_base, inst_idx, geom_idx, prim_idx)` using a
/// **branched** 2-level fast path for single-geometry BLASes (`level1_val >= DXR_NORMALS_DATA_BASE`
/// skips the Region 1 `geom_table` load completely; `ARCH-14`).
/// Returns `(join_block, [nx, ny, nz])`.
pub(crate) fn emit_tri_fetch_normal(
    s: &mut Shader,
    block: usize,
    tlas_meta_base: V,
    inst_idx: V,
    geom_idx: V,
    prim_idx: V,
) -> (usize, [V; 3]) {
    let i32t = s.t.i32;
    let f32t = s.t.f32;
    let srv = s.root_srv_handle(SLOT_DXR_NORMALS);
    let zero_i = s.m.i32(0);

    // Level 1: `tlas_meta_base + min(inst_idx, STRIDE - 1)` -> either:
    // - `0` (invalid / fallback),
    // - `[GEOM_BASE, DATA_BASE)` (multi-geometry BLAS -> Region 1 `geom_table_off`), or
    // - `>= DATA_BASE` (single-geometry BLAS direct shortcut -> Region 2 `prim_data_off`).
    let inst_max = s.m.i32(DXR_NORMALS_TLAS_STRIDE - 1);
    let inst_clamped = s.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[inst_idx, inst_max]);
    let inst_word = s.m.bin(block, Bin::Add, tlas_meta_base, inst_clamped);
    let has_tlas = s.m.cmp(block, pred::NE, tlas_meta_base, zero_i);
    let level1_raw = raw_load_normals_u32(s, block, srv, inst_word, DXR_NORMALS_GEOM_BASE - 1);
    let level1_val = s.m.select(block, has_tlas, level1_raw, zero_i);

    // Branched Level 2: only execute Region 1 `geom_table` load when `GEOM_BASE <= val < DATA_BASE`.
    // Single-geometry BLASes (`val >= DATA_BASE`) and fallback (`val == 0`) branch around it!
    let geom_floor = s.m.i32(DXR_NORMALS_GEOM_BASE);
    let data_floor = s.m.i32(DXR_NORMALS_DATA_BASE);
    let ge_geom = s.m.cmp(block, pred::IUGE, level1_val, geom_floor);
    let lt_data = s.m.cmp(block, pred::IULT, level1_val, data_floor);
    let need_l2 = s.m.bin(block, Bin::And, ge_geom, lt_data);

    let l2_blk = s.m.new_block();
    let join_blk = s.m.new_block();
    s.m.terminate(block, Term::CondBr(need_l2, l2_blk, join_blk));

    let geom_lim = s.m.i32(DXR_NORMALS_DATA_BASE - 1);
    let geom_clamped = s.op(l2_blk, "binary.i32", Attr::ReadNone, i32t, 40, &[geom_idx, geom_lim]);
    let geom_word = s.m.bin(l2_blk, Bin::Add, level1_val, geom_clamped);
    let prim_base_l2 = raw_load_normals_u32(s, l2_blk, srv, geom_word, DXR_NORMALS_DATA_BASE - 1);
    s.m.terminate(l2_blk, Term::Br(join_blk));

    let prim_base = s.m.phi(join_blk, i32t);
    s.m.set_phi(prim_base, vec![(prim_base_l2, l2_blk), (level1_val, block)]);

    // Level 3: `prim_base + prim_idx` -> packed octahedral u32 (or word 0 fallback `(0,0,1)`).
    let has_prim = s.m.cmp(join_blk, pred::IUGE, prim_base, data_floor);
    let words_lim = s.m.i32(DXR_NORMALS_WORDS - 1);
    let prim_clamped = s.op(join_blk, "binary.i32", Attr::ReadNone, i32t, 40, &[prim_idx, words_lim]);
    let norm_word_raw = s.m.bin(join_blk, Bin::Add, prim_base, prim_clamped);
    let norm_word = s.m.select(join_blk, has_prim, norm_word_raw, zero_i);
    let packed = raw_load_normals_u32(s, join_blk, srv, norm_word, DXR_NORMALS_WORDS - 1);

    // Unpack octahedral `u32` (`qx | (qy << 16)`, 0x8000 = 0.0) -> unit normal `(nx, ny, nz)`.
    let mask16 = s.m.i32(0xFFFF);
    let c16 = s.m.i32(16);
    let qx_i = s.m.bin(join_blk, Bin::And, packed, mask16);
    let qy_i = s.m.bin(join_blk, Bin::LShr, packed, c16);
    let qx_f = s.m.cast(join_blk, Cast::UiToFp, qx_i, f32t);
    let qy_f = s.m.cast(join_blk, Cast::UiToFp, qy_i, f32t);

    let f32768 = s.m.f32(32768.0f32.to_bits());
    let f_inv32767 = s.m.f32((1.0f32 / 32767.0f32).to_bits());
    let f_neg1 = s.m.f32((-1.0f32).to_bits());
    let f_zero = s.m.f32(0.0f32.to_bits());
    let f_one = s.m.f32(1.0f32.to_bits());

    let ox_sub = s.m.bin(join_blk, Bin::Sub, qx_f, f32768);
    let ox_raw = s.m.bin(join_blk, Bin::Mul, ox_sub, f_inv32767);
    let oy_sub = s.m.bin(join_blk, Bin::Sub, qy_f, f32768);
    let oy_raw = s.m.bin(join_blk, Bin::Mul, oy_sub, f_inv32767);
    let ox_hi = s.op(join_blk, "binary.f32", Attr::ReadNone, f32t, 36, &[ox_raw, f_one]);
    let ox = s.op(join_blk, "binary.f32", Attr::ReadNone, f32t, 35, &[ox_hi, f_neg1]);
    let oy_hi = s.op(join_blk, "binary.f32", Attr::ReadNone, f32t, 36, &[oy_raw, f_one]);
    let oy = s.op(join_blk, "binary.f32", Attr::ReadNone, f32t, 35, &[oy_hi, f_neg1]);

    let abs_ox = s.op(join_blk, "unary.f32", Attr::ReadNone, f32t, 6, &[ox]);
    let abs_oy = s.op(join_blk, "unary.f32", Attr::ReadNone, f32t, 6, &[oy]);
    let nz_sub1 = s.m.bin(join_blk, Bin::Sub, f_one, abs_ox);
    let nz_u = s.m.bin(join_blk, Bin::Sub, nz_sub1, abs_oy);
    let neg_nz = s.m.bin(join_blk, Bin::Sub, f_zero, nz_u);
    let t_fold = s.op(join_blk, "binary.f32", Attr::ReadNone, f32t, 35, &[neg_nz, f_zero]);
    let neg_t = s.m.bin(join_blk, Bin::Sub, f_zero, t_fold);
    let ox_ge0 = s.m.cmp(join_blk, pred::OGE, ox, f_zero);
    let oy_ge0 = s.m.cmp(join_blk, pred::OGE, oy, f_zero);
    let dx = s.m.select(join_blk, ox_ge0, neg_t, t_fold);
    let dy = s.m.select(join_blk, oy_ge0, neg_t, t_fold);
    let nx_u = s.m.bin(join_blk, Bin::Add, ox, dx);
    let ny_u = s.m.bin(join_blk, Bin::Add, oy, dy);

    let nx2 = s.m.bin(join_blk, Bin::Mul, nx_u, nx_u);
    let ny2 = s.m.bin(join_blk, Bin::Mul, ny_u, ny_u);
    let nz2 = s.m.bin(join_blk, Bin::Mul, nz_u, nz_u);
    let nxy2 = s.m.bin(join_blk, Bin::Add, nx2, ny2);
    let n_len2 = s.m.bin(join_blk, Bin::Add, nxy2, nz2);
    let inv_n = s.op(join_blk, "unary.f32", Attr::ReadNone, f32t, 25, &[n_len2]);
    let nx = s.m.bin(join_blk, Bin::Mul, nx_u, inv_n);
    let ny = s.m.bin(join_blk, Bin::Mul, ny_u, inv_n);
    let nz = s.m.bin(join_blk, Bin::Mul, nz_u, inv_n);

    (join_blk, [nx, ny, nz])
}

/// Derives unit tangent `T` (`want_b == false`) or bitangent `B = cross(N, T)` (`want_b == true`)
/// from an already-decoded unit normal `N = [nx, ny, nz]` (`f32`) as `num_dwords` `i32` bit patterns.
pub(crate) fn emit_basis_from_normal(
    s: &mut Shader,
    block: usize,
    n: [V; 3],
    want_b: bool,
    num_dwords: usize,
) -> Vec<V> {
    let [nx, ny, nz] = n;
    let i32t = s.t.i32;
    let f32t = s.t.f32;
    let zero_i = s.m.i32(0);
    let f_zero = s.m.f32(0.0f32.to_bits());

    // Orthonormal tangent `T = normalize(cross(up, N))` with branchless `up` pick.
    let abs_nz = s.op(block, "unary.f32", Attr::ReadNone, f32t, 6, &[nz]);
    let thresh = s.m.f32(0.999f32.to_bits());
    let use_z = s.m.cmp(block, pred::OLT, abs_nz, thresh);
    let neg_ny = s.m.bin(block, Bin::Sub, f_zero, ny);
    let neg_nz_n = s.m.bin(block, Bin::Sub, f_zero, nz);
    let tx_u = s.m.select(block, use_z, neg_ny, f_zero);
    let ty_u = s.m.select(block, use_z, nx, neg_nz_n);
    let tz_u = s.m.select(block, use_z, f_zero, ny);
    let tx2 = s.m.bin(block, Bin::Mul, tx_u, tx_u);
    let ty2 = s.m.bin(block, Bin::Mul, ty_u, ty_u);
    let tz2 = s.m.bin(block, Bin::Mul, tz_u, tz_u);
    let txy2 = s.m.bin(block, Bin::Add, tx2, ty2);
    let t_len2 = s.m.bin(block, Bin::Add, txy2, tz2);
    let inv_t = s.op(block, "unary.f32", Attr::ReadNone, f32t, 25, &[t_len2]);
    let tx = s.m.bin(block, Bin::Mul, tx_u, inv_t);
    let ty = s.m.bin(block, Bin::Mul, ty_u, inv_t);
    let tz = s.m.bin(block, Bin::Mul, tz_u, inv_t);

    let vec3 = if want_b {
        // `B = cross(N, T)` -> `cross(T, B) == +N`.
        let bx0 = s.m.bin(block, Bin::Mul, ny, tz);
        let bx1 = s.m.bin(block, Bin::Mul, nz, ty);
        let bx = s.m.bin(block, Bin::Sub, bx0, bx1);
        let by0 = s.m.bin(block, Bin::Mul, nz, tx);
        let by1 = s.m.bin(block, Bin::Mul, nx, tz);
        let by = s.m.bin(block, Bin::Sub, by0, by1);
        let bz0 = s.m.bin(block, Bin::Mul, nx, ty);
        let bz1 = s.m.bin(block, Bin::Mul, ny, tx);
        let bz = s.m.bin(block, Bin::Sub, bz0, bz1);
        [bx, by, bz]
    } else {
        [tx, ty, tz]
    };
    (0..num_dwords)
        .map(|k| {
            if k < 3 {
                s.m.cast(block, Cast::Bitcast, vec3[k], i32t)
            } else {
                zero_i
            }
        })
        .collect()
}

/// Fetches the octahedral-packed object-space geometric normal `N` from
/// `SLOT_DXR_NORMALS` for `(tlas_meta_base, inst_idx, geom_idx, prim_idx)` and
/// returns `(updated_block, dwords)` for unit tangent `T` (`want_b == false`) or unit
/// bitangent `B = cross(N, T)` (`want_b == true`), so `cross(T - 0, B - 0) == +N`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_tri_fetch_basis(
    s: &mut Shader,
    block: usize,
    tlas_meta_base: V,
    inst_idx: V,
    geom_idx: V,
    prim_idx: V,
    want_b: bool,
    num_dwords: usize,
) -> (usize, Vec<V>) {
    let (next_blk, n) = emit_tri_fetch_normal(s, block, tlas_meta_base, inst_idx, geom_idx, prim_idx);
    let dwords = emit_basis_from_normal(s, next_blk, n, want_b, num_dwords);
    (next_blk, dwords)
}

/// Emits a single replaced BVH metadata load (`global_load_*` / `s_load_*`) as SM 6.6
/// `dx.op.rayQuery_State*` dwords in `block`. Returns `(updated_block, dwords)`.
pub(crate) fn emit_ray_meta_load(
    s: &mut Shader,
    block: usize,
    rq: V,
    candidate: bool,
    kind: RayMetaKind,
    num_dwords: usize,
    tlas_meta_base: Option<V>,
) -> (usize, Vec<V>) {
    let i32t = s.t.i32;
    match kind {
        RayMetaKind::BlasBaseZero | RayMetaKind::TriFetchZero3 => (block, vec![s.m.i32(0); num_dwords]),
        RayMetaKind::InstanceId => {
            let op = if candidate {
                rq_op::CANDIDATE_INSTANCE_ID
            } else {
                rq_op::COMMITTED_INSTANCE_ID
            };
            (block, vec![s.ray_query_state_i32(block, op, rq)])
        }
        RayMetaKind::InstanceContribution => {
            let op = if candidate {
                rq_op::CANDIDATE_INSTANCE_CONTRIBUTION
            } else {
                rq_op::COMMITTED_INSTANCE_CONTRIBUTION
            };
            (block, vec![s.ray_query_state_i32(block, op, rq)])
        }
        RayMetaKind::InstanceIndex => {
            let op = if candidate {
                rq_op::CANDIDATE_INSTANCE_INDEX
            } else {
                rq_op::COMMITTED_INSTANCE_INDEX
            };
            (block, vec![s.ray_query_state_i32(block, op, rq)])
        }
        RayMetaKind::GeometryIndexShl8 => {
            let op = if candidate {
                rq_op::CANDIDATE_GEOMETRY_INDEX
            } else {
                rq_op::COMMITTED_GEOMETRY_INDEX
            };
            let g = s.ray_query_state_i32(block, op, rq);
            let c8 = s.m.i32(8);
            (block, vec![s.m.bin(block, Bin::Shl, g, c8)])
        }
        RayMetaKind::PrimitiveIndex => {
            let op = if candidate {
                rq_op::CANDIDATE_PRIMITIVE_INDEX
            } else {
                rq_op::COMMITTED_PRIMITIVE_INDEX
            };
            (block, vec![s.ray_query_state_i32(block, op, rq)])
        }
        RayMetaKind::ObjectToWorld(base_off) => {
            let op = if candidate {
                rq_op::CANDIDATE_OBJECT_TO_WORLD_3X4
            } else {
                rq_op::COMMITTED_OBJECT_TO_WORLD_3X4
            };
            let vals = (0..num_dwords)
                .map(|k| {
                    let off = base_off + (k as u32) * 4;
                    let row = off / 16;
                    let col = ((off % 16) / 4) as u8;
                    let m = s.ray_query_state_mat_f32(block, op, rq, row, col);
                    s.m.cast(block, Cast::Bitcast, m, i32t)
                })
                .collect();
            (block, vals)
        }
        RayMetaKind::TriFetchBasisT | RayMetaKind::TriFetchBasisB => {
            let inst_op = if candidate {
                rq_op::CANDIDATE_INSTANCE_INDEX
            } else {
                rq_op::COMMITTED_INSTANCE_INDEX
            };
            let geom_op = if candidate {
                rq_op::CANDIDATE_GEOMETRY_INDEX
            } else {
                rq_op::COMMITTED_GEOMETRY_INDEX
            };
            let prim_op = if candidate {
                rq_op::CANDIDATE_PRIMITIVE_INDEX
            } else {
                rq_op::COMMITTED_PRIMITIVE_INDEX
            };
            let inst_idx = s.ray_query_state_i32(block, inst_op, rq);
            let geom_idx = s.ray_query_state_i32(block, geom_op, rq);
            let prim_idx = s.ray_query_state_i32(block, prim_op, rq);
            let base = tlas_meta_base.unwrap_or_else(|| s.m.i32(0));
            emit_tri_fetch_basis(
                s,
                block,
                base,
                inst_idx,
                geom_idx,
                prim_idx,
                matches!(kind, RayMetaKind::TriFetchBasisB),
                num_dwords,
            )
        }
    }
}

/// Emits SM 6.6 `RayQuery` DXIL instructions starting at `entry_dxil` and branching
/// to `exit_dxil`. For [`RayLoopKind::AnyHit`], returns [`PendingAnyHit`] so `Lower`
/// can complete the `Proceed()` back-edge when lowering reaches [`AnyHitInfo::tail_block`].
pub(crate) fn emit_ray_loop(
    s: &mut Shader,
    entry_dxil: usize,
    exit_dxil: usize,
    rl: &RayLoop,
    inputs: &ResolvedRayInputs,
) -> RayLoopEmission {
    let i32t = s.t.i32;
    let f32t = s.t.f32;

    if let Some(active) = inputs.active_ray {
        let trace_blk = s.m.new_block();
        let merge_blk = s.m.new_block();
        s.m.terminate(entry_dxil, Term::CondBr(active, trace_blk, merge_blk));

        let (after_blk, traced, cand_outs, pending_anyhit, two_pass_post) =
            emit_trace_body(s, trace_blk, rl, inputs, i32t, f32t);
        s.m.terminate(after_blk, Term::Br(merge_blk));

        let mut merged = Vec::with_capacity(rl.outputs.len() + cand_outs.len());
        let mut active_phi_cache: HashMap<(V, V), V> = HashMap::new();
        for (binding, &(val_id, traced_v)) in rl.outputs.iter().zip(&traced) {
            let out_v = match binding.output {
                RayOutput::Zero => s.m.i32(0),
                RayOutput::CopyOf(pre) => inputs
                    .fallbacks
                    .get(&pre)
                    .copied()
                    .unwrap_or_else(|| s.m.i32(0)),
                RayOutput::FrontBackMinFrontT => {
                    let inactive_v = inputs
                        .init_min_front_t
                        .unwrap_or(inputs.t_max);
                    *active_phi_cache.entry((traced_v, inactive_v)).or_insert_with(|| {
                        let phi = s.m.phi(merge_blk, i32t);
                        s.m.set_phi(phi, vec![(traced_v, after_blk), (inactive_v, entry_dxil)]);
                        phi
                    })
                }
                RayOutput::AnyHitState(k) => {
                    let inactive_v = inputs
                        .anyhit_inits
                        .get(k)
                        .copied()
                        .unwrap_or_else(|| s.m.i32(0));
                    *active_phi_cache.entry((traced_v, inactive_v)).or_insert_with(|| {
                        let phi = s.m.phi(merge_blk, i32t);
                        s.m.set_phi(phi, vec![(traced_v, after_blk), (inactive_v, entry_dxil)]);
                        phi
                    })
                }
                _ => {
                    let inactive_v = binding
                        .fallback
                        .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                        .unwrap_or_else(|| s.m.i32(0));
                    *active_phi_cache.entry((traced_v, inactive_v)).or_insert_with(|| {
                        let phi = s.m.phi(merge_blk, i32t);
                        s.m.set_phi(phi, vec![(traced_v, after_blk), (inactive_v, entry_dxil)]);
                        phi
                    })
                }
            };
            merged.push((val_id, out_v));
        }
        merged.extend(cand_outs);

        let two_pass_post = two_pass_post.map(|tp| {
            let i1t = s.t.i1;
            let false_v = s.m.bool(false);
            let q2_hit = s.m.phi(merge_blk, i1t);
            s.m.set_phi(q2_hit, vec![(tp.q2_hit, after_blk), (false_v, entry_dxil)]);
            let q1_hit = s.m.phi(merge_blk, i1t);
            s.m.set_phi(q1_hit, vec![(tp.q1_hit, after_blk), (false_v, entry_dxil)]);
            let zero = s.m.i32(0);
            let mut q1_fields = HashMap::with_capacity(tp.q1_fields.len());
            for (field, val) in tp.q1_fields {
                let phi = s.m.phi(merge_blk, i32t);
                s.m.set_phi(phi, vec![(val, after_blk), (zero, entry_dxil)]);
                q1_fields.insert(field, phi);
            }
            TwoPassPost {
                q2_hit,
                q1_hit,
                q1_fields,
            }
        });

        s.m.terminate(merge_blk, Term::Br(exit_dxil));
        RayLoopEmission {
            tail: merge_blk,
            outs: merged,
            pending_anyhit,
            two_pass_post,
        }
    } else {
        let (after_blk, mut outs, cand_outs, pending_anyhit, two_pass_post) =
            emit_trace_body(s, entry_dxil, rl, inputs, i32t, f32t);
        outs.extend(cand_outs);
        s.m.terminate(after_blk, Term::Br(exit_dxil));
        RayLoopEmission {
            tail: after_blk,
            outs,
            pending_anyhit,
            two_pass_post,
        }
    }
}

/// Materializes a lazy candidate triangle query (`RayT`, `Bary`, `FrontFace`) in `block`.
pub(crate) fn emit_cand_input(s: &mut Shader, block: usize, rq: V, ci: CandidateInput) -> V {
    let i32t = s.t.i32;
    match ci {
        CandidateInput::RayT => {
            let tf = s.ray_query_state_f32(block, rq_op::CANDIDATE_TRIANGLE_RAY_T, rq);
            s.m.cast(block, Cast::Bitcast, tf, i32t)
        }
        CandidateInput::Bary(comp) => {
            let bf = s.ray_query_state_vec_f32(
                block,
                rq_op::CANDIDATE_TRIANGLE_BARYCENTRICS,
                rq,
                comp,
            );
            s.m.cast(block, Cast::Bitcast, bf, i32t)
        }
        CandidateInput::FrontFace => {
            let ff = s.ray_query_state_i1(block, rq_op::CANDIDATE_TRIANGLE_FRONT_FACE, rq);
            let front = s.m.i32(0xFE);
            let back = s.m.i32(0xFF);
            s.m.select(block, ff, front, back)
        }
        _ => unreachable!("only lazy candidate queries use emit_cand_input"),
    }
}

fn emit_committed_hit_output(
    s: &mut Shader,
    block: usize,
    rq: V,
    output: RayOutput,
    i32t: super::bitcode::Ty,
) -> V {
    match output {
        RayOutput::CommittedStatusTriNode => s.m.i32(1),
        RayOutput::CommittedRayT => {
            let t_f = s.ray_query_state_f32(block, rq_op::COMMITTED_RAY_T, rq);
            s.m.cast(block, Cast::Bitcast, t_f, i32t)
        }
        RayOutput::CommittedBary(comp) => {
            let b_f = s.ray_query_state_vec_f32(
                block,
                rq_op::COMMITTED_TRIANGLE_BARYCENTRICS,
                rq,
                comp,
            );
            s.m.cast(block, Cast::Bitcast, b_f, i32t)
        }
        RayOutput::CommittedFrontFace => {
            let ff = s.ray_query_state_i1(block, rq_op::COMMITTED_TRIANGLE_FRONT_FACE, rq);
            let front = s.m.i32(0xFE);
            let back = s.m.i32(0xFF);
            s.m.select(block, ff, front, back)
        }
        RayOutput::CommittedInstanceIndex => {
            s.ray_query_state_i32(block, rq_op::COMMITTED_INSTANCE_INDEX, rq)
        }
        RayOutput::CommittedInstanceId => {
            s.ray_query_state_i32(block, rq_op::COMMITTED_INSTANCE_ID, rq)
        }
        RayOutput::CommittedInstanceContribution => {
            s.ray_query_state_i32(block, rq_op::COMMITTED_INSTANCE_CONTRIBUTION, rq)
        }
        RayOutput::CommittedGeometryIndex => {
            s.ray_query_state_i32(block, rq_op::COMMITTED_GEOMETRY_INDEX, rq)
        }
        RayOutput::CommittedPrimitiveIndex => {
            s.ray_query_state_i32(block, rq_op::COMMITTED_PRIMITIVE_INDEX, rq)
        }
        RayOutput::CommittedObjectToWorld(row, col) => {
            let m = s.ray_query_state_mat_f32(
                block,
                rq_op::COMMITTED_OBJECT_TO_WORLD_3X4,
                rq,
                row,
                col,
            );
            s.m.cast(block, Cast::Bitcast, m, i32t)
        }
        RayOutput::CommittedWorldToObject(row, col) => {
            let m = s.ray_query_state_mat_f32(
                block,
                rq_op::COMMITTED_WORLD_TO_OBJECT_3X4,
                rq,
                row,
                col,
            );
            s.m.cast(block, Cast::Bitcast, m, i32t)
        }
        _ => unreachable!("non-committed output in emit_committed_hit_output"),
    }
}

fn emit_early_q1_field(
    s: &mut Shader,
    block: usize,
    rq: V,
    field: EarlyQ1Field,
    i32t: super::bitcode::Ty,
) -> V {
    let out = match field {
        EarlyQ1Field::Bary(c) => RayOutput::CommittedBary(c),
        EarlyQ1Field::FrontFace => RayOutput::CommittedFrontFace,
        EarlyQ1Field::InstanceIndex => RayOutput::CommittedInstanceIndex,
        EarlyQ1Field::InstanceId => RayOutput::CommittedInstanceId,
        EarlyQ1Field::InstanceContribution => RayOutput::CommittedInstanceContribution,
        EarlyQ1Field::GeometryIndex => RayOutput::CommittedGeometryIndex,
        EarlyQ1Field::PrimitiveIndex => RayOutput::CommittedPrimitiveIndex,
        EarlyQ1Field::ObjectToWorld(r, c) => RayOutput::CommittedObjectToWorld(r, c),
        EarlyQ1Field::WorldToObject(r, c) => RayOutput::CommittedWorldToObject(r, c),
    };
    emit_committed_hit_output(s, block, rq, out, i32t)
}

fn early_field_for_output(output: RayOutput) -> Option<EarlyQ1Field> {
    match output {
        RayOutput::CommittedBary(c) => Some(EarlyQ1Field::Bary(c)),
        RayOutput::CommittedFrontFace => Some(EarlyQ1Field::FrontFace),
        RayOutput::CommittedInstanceIndex => Some(EarlyQ1Field::InstanceIndex),
        RayOutput::CommittedInstanceId => Some(EarlyQ1Field::InstanceId),
        RayOutput::CommittedInstanceContribution => Some(EarlyQ1Field::InstanceContribution),
        RayOutput::CommittedGeometryIndex => Some(EarlyQ1Field::GeometryIndex),
        RayOutput::CommittedPrimitiveIndex => Some(EarlyQ1Field::PrimitiveIndex),
        RayOutput::CommittedObjectToWorld(r, c) => Some(EarlyQ1Field::ObjectToWorld(r, c)),
        RayOutput::CommittedWorldToObject(r, c) => Some(EarlyQ1Field::WorldToObject(r, c)),
        _ => None,
    }
}

/// Early-extracted Pass 1 state (`ARCH-14`: `TwoPassQ1::Live` completely removed so every
/// two-pass / mask-split loop reuses a single `RayQuery` handle).
struct EarlyQ1 {
    q1_hit: V,
    t_clamp: V,
    q1_fields: HashMap<EarlyQ1Field, V>,
}

/// Completes Pass 1's out-of-loop `Committed*` `AnyHit` check for an un-fused `fno_spec` loop:
/// terminates `p1_tail_blk` with `CondBr(p1_accept, spec.after_blk, spec.p2_trace_blk)` and
/// populates `p2_from_p1_phi`, `p2_t1_phi`, and `final_t_phi`.
pub(crate) fn finish_fno_spec_pass1(
    s: &mut Shader,
    spec: &SpeculativeAnyHit,
    p1_tail_blk: usize,
    p1_accept: V,
) {
    s.m.terminate(
        p1_tail_blk,
        Term::CondBr(p1_accept, spec.after_blk, spec.p2_trace_blk),
    );
    if let (Some(fp1), Some(bypass)) = (spec.p2_from_p1_phi, spec.dyn_bypass_blk) {
        let true_v = s.m.bool(true);
        let false_v = s.m.bool(false);
        s.m.set_phi(fp1, vec![(true_v, p1_tail_blk), (false_v, bypass)]);
    }
    if let (Some(t1_phi), Some(p1_t), Some(bypass)) =
        (spec.p2_t1_phi, spec.p1_t_f32, spec.dyn_bypass_blk)
    {
        s.m.set_phi(t1_phi, vec![(p1_t, p1_tail_blk), (spec.t_min_f32, bypass)]);
    }
    if let Some((final_t, t_max_i, p2_t)) = spec.final_t_phi {
        let p1_t_i = spec.p1_t_i32.expect("p1_t_i32 for final_t_phi");
        s.m.set_phi(
            final_t,
            vec![
                (t_max_i, spec.p1_check_blk),
                (p1_t_i, p1_tail_blk),
                (p2_t, spec.p2_exit_blk),
            ],
        );
    }
}

fn emit_trace_body(
    s: &mut Shader,
    start_blk: usize,
    rl: &RayLoop,
    inputs: &ResolvedRayInputs,
    i32t: super::bitcode::Ty,
    f32t: super::bitcode::Ty,
) -> (
    usize,
    Vec<(Value, V)>,
    Vec<(Value, V)>,
    Option<PendingAnyHit>,
    Option<TwoPassPost>,
) {
    let mask = match rl.instance_mask {
        RayInput::Const(m) => s.m.i32(m & 0xFF),
        RayInput::Val(_) => {
            let v = inputs.instance_mask.unwrap_or_else(|| s.m.i32(0xFF));
            let ff = s.m.i32(0xFF);
            s.m.bin(start_blk, Bin::And, v, ff)
        }
        RayInput::ShiftedMask(_) => {
            let v = inputs.instance_mask.unwrap_or_else(|| s.m.i32(0xFF00_0000));
            let c24 = s.m.i32(24);
            s.m.bin(start_blk, Bin::LShr, v, c24)
        }
    };

    let origin = [
        s.m.cast(start_blk, Cast::Bitcast, inputs.origin[0], f32t),
        s.m.cast(start_blk, Cast::Bitcast, inputs.origin[1], f32t),
        s.m.cast(start_blk, Cast::Bitcast, inputs.origin[2], f32t),
    ];
    let dir = [
        s.m.cast(start_blk, Cast::Bitcast, inputs.dir[0], f32t),
        s.m.cast(start_blk, Cast::Bitcast, inputs.dir[1], f32t),
        s.m.cast(start_blk, Cast::Bitcast, inputs.dir[2], f32t),
    ];
    let t_min = match inputs.t_min {
        Some(v) => s.m.cast(start_blk, Cast::Bitcast, v, f32t),
        None => s.m.f32(0.0f32.to_bits()),
    };
    let t_max = s.m.cast(start_blk, Cast::Bitcast, inputs.t_max, f32t);

    let rq = inputs.rq;
    let skip_proc = (inputs.group_const_flags & ray_flag::SKIP_PROCEDURAL_PRIMITIVES) != 0
        || matches!(rl.ray_flags, RayInput::Const(f) if (f & ray_flag::SKIP_PROCEDURAL_PRIMITIVES) != 0);

    // ARCH-13: 2-pass pure HW `FrontBackTrack` (`CULL_BACK` + `CULL_FRONT`) against the
    // HostHeap twin-slot Mirror TLAS supersedes 3-pass `fbt_mask_split` and keeps
    // `constRayFlags = 512` (not `513`) so it coalesces with `fno_spec` loops (`ARCH-14`).
    let fbt_two_pass = is_front_back_two_pass_shape(rl)
        && (inputs.fbt_mirror_tlas
            || (inputs.fbt_split_mask == 0
                && (inputs.group_const_flags & ray_flag::FORCE_NON_OPAQUE) == 0
                && (inputs.group_const_flags & ray_flag::FORCE_OPAQUE) != 0));

    let fbt_mask_split = !fbt_two_pass
        && inputs.fbt_split_mask != 0
        && is_front_back_two_pass_shape(rl)
        && (inputs.group_const_flags & ray_flag::FORCE_OPAQUE) == 0;

    let fno_spec = is_fno_speculative_eligible(rl)
        && (inputs.group_const_flags & ray_flag::FORCE_NON_OPAQUE) == 0
        && (inputs.group_const_flags & ray_flag::FORCE_OPAQUE) == 0;

    // Fallback 3-pass mask-split FrontBackTrack (only when Mirror TLAS is disabled or
    // `raw_views == 0`, always using single-handle early extraction):
    if fbt_mask_split {
        let orig_f = match rl.ray_flags {
            RayInput::Const(f) => f,
            _ => unreachable!("mask-split FrontBackTrack requires constant ray flags"),
        };
        let cull_bit = u32::from(inputs.fbt_split_mask);
        let ss_const = s.m.i32((!cull_bit) & 0xFF);
        let ds_const = s.m.i32(cull_bit & 0xFF);
        let mask_ss = s.m.bin(start_blk, Bin::And, mask, ss_const);
        let mask_ds = s.m.bin(start_blk, Bin::And, mask, ds_const);

        let base_opaque = (orig_f & !ray_flag::FORCE_NON_OPAQUE) | ray_flag::FORCE_OPAQUE;
        let needs_min_front = rl
            .outputs
            .iter()
            .any(|b| matches!(b.output, RayOutput::FrontBackMinFrontT));
        let init_i = inputs.init_min_front_t.unwrap_or(inputs.t_max);
        let init_f = s.m.cast(start_blk, Cast::Bitcast, init_i, f32t);
        let one = s.m.i32(1);
        let zero = s.m.i32(0);

        // Pass A: single-sided front-face trace (`FORCE_OPAQUE | CULL_BACK`).
        let (b_start_blk, min_front_a) = if needs_min_front {
            let dyn_flags_front = s.m.i32(
                (base_opaque | ray_flag::CULL_BACK_FACING_TRIANGLES) & !inputs.group_const_flags,
            );
            s.ray_query_trace_inline(
                start_blk,
                rq,
                inputs.tlas,
                dyn_flags_front,
                mask_ss,
                origin,
                t_min,
                dir,
                t_max,
            );
            let f_proceed_blk = s.m.new_block();
            let next_blk = s.m.new_block();
            s.m.terminate(start_blk, Term::Br(f_proceed_blk));
            let f_more = s.ray_query_proceed(f_proceed_blk, rq);
            s.m.terminate(f_proceed_blk, Term::CondBr(f_more, f_proceed_blk, next_blk));

            let f_status = s.ray_query_state_i32(next_blk, rq_op::COMMITTED_STATUS, rq);
            let f_hit = s.m.cmp(next_blk, pred::EQ, f_status, one);
            let f_t = s.ray_query_state_f32(next_blk, rq_op::COMMITTED_RAY_T, rq);
            let f_lt = s.m.cmp(next_blk, pred::OLT, f_t, init_f);
            let use_f = s.m.bin(next_blk, Bin::And, f_hit, f_lt);
            let min_front = s.m.select(next_blk, use_f, f_t, init_f);
            (next_blk, min_front)
        } else {
            (start_blk, init_f)
        };

        // Pass B: single-sided back-face trace (`FORCE_OPAQUE | CULL_FRONT`).
        let dyn_flags_back = s.m.i32(
            (base_opaque | ray_flag::CULL_FRONT_FACING_TRIANGLES) & !inputs.group_const_flags,
        );
        s.ray_query_trace_inline(
            b_start_blk,
            rq,
            inputs.tlas,
            dyn_flags_back,
            mask_ss,
            origin,
            t_min,
            dir,
            t_max,
        );
        let b_proceed_blk = s.m.new_block();
        let b_done_blk = s.m.new_block();
        s.m.terminate(b_start_blk, Term::Br(b_proceed_blk));
        let b_more = s.ray_query_proceed(b_proceed_blk, rq);
        s.m.terminate(b_proceed_blk, Term::CondBr(b_more, b_proceed_blk, b_done_blk));

        let b_status = s.ray_query_state_i32(b_done_blk, rq_op::COMMITTED_STATUS, rq);
        let b_hit = s.m.cmp(b_done_blk, pred::EQ, b_status, one);

        // Always extract Pass B fields early so `rq` is reused cleanly for Pass C.
        let auto_early: Option<Vec<EarlyQ1Field>> = if inputs.two_pass_early_fields.is_none() {
            Some(collect_early_q1_fields(rl, &[]))
        } else {
            None
        };
        let fields = inputs
            .two_pass_early_fields
            .as_deref()
            .or(auto_early.as_deref())
            .unwrap_or(&[]);
        let b_extract_blk = s.m.new_block();
        let c_prep_blk = s.m.new_block();
        s.m.terminate(b_done_blk, Term::CondBr(b_hit, b_extract_blk, c_prep_blk));
        let b_t_hit = s.ray_query_state_f32(b_extract_blk, rq_op::COMMITTED_RAY_T, rq);
        let mut extracted: Vec<(EarlyQ1Field, V)> = Vec::with_capacity(fields.len());
        for &field in fields {
            let v = emit_early_q1_field(s, b_extract_blk, rq, field, i32t);
            extracted.push((field, v));
        }
        s.m.terminate(b_extract_blk, Term::Br(c_prep_blk));

        let b_t = s.m.phi(c_prep_blk, f32t);
        s.m.set_phi(b_t, vec![(b_t_hit, b_extract_blk), (t_max, b_done_blk)]);
        let mut q1_fields = HashMap::with_capacity(fields.len());
        for (field, v_hit) in extracted {
            let phi = s.m.phi(c_prep_blk, i32t);
            s.m.set_phi(phi, vec![(v_hit, b_extract_blk), (zero, b_done_blk)]);
            q1_fields.insert(field, phi);
        }
        let two_pass_q1 = EarlyQ1 {
            q1_hit: b_hit,
            t_clamp: b_t,
            q1_fields,
        };

        // Pass C: double-sided `FORCE_NON_OPAQUE` `Proceed()` over `mask_ds`, clamped to
        // `b_hit ? min(t_max, max(min_front_a, b_t)) : t_max`.
        let t_max_c = if needs_min_front {
            let mf_gt_bt = s.m.cmp(c_prep_blk, pred::OGT, min_front_a, b_t);
            let mf_lt_tmax = s.m.cmp(c_prep_blk, pred::OLT, min_front_a, t_max);
            let mf_clamped = s.m.select(c_prep_blk, mf_lt_tmax, min_front_a, t_max);
            let tight = s.m.select(c_prep_blk, mf_gt_bt, mf_clamped, b_t);
            s.m.select(c_prep_blk, b_hit, tight, t_max)
        } else {
            s.m.select(c_prep_blk, b_hit, b_t, t_max)
        };
        let has_ds = s.m.cmp(c_prep_blk, pred::NE, mask_ds, zero);
        let c_trace_blk = s.m.new_block();
        let c_proceed_blk = s.m.new_block();
        let c_tri_blk = s.m.new_block();
        let c_front_blk = s.m.new_block();
        let c_back_blk = s.m.new_block();
        let c_commit_blk = s.m.new_block();
        let c_latch_blk = s.m.new_block();
        let c_done_blk = s.m.new_block();
        let after_blk = s.m.new_block();
        s.m.terminate(c_prep_blk, Term::CondBr(has_ds, c_trace_blk, after_blk));

        let dyn_flags_c = s.m.i32(orig_f & !inputs.group_const_flags);
        s.ray_query_trace_inline(
            c_trace_blk,
            rq,
            inputs.tlas,
            dyn_flags_c,
            mask_ds,
            origin,
            t_min,
            dir,
            t_max_c,
        );
        s.m.terminate(c_trace_blk, Term::Br(c_proceed_blk));

        let min_phi = s.m.phi(c_proceed_blk, f32t);
        let back_t_phi = s.m.phi(c_proceed_blk, f32t);
        let c_more = s.ray_query_proceed(c_proceed_blk, rq);
        let c_cand_blk = if skip_proc {
            s.m.terminate(c_proceed_blk, Term::CondBr(c_more, c_tri_blk, c_done_blk));
            None
        } else {
            let cb = s.m.new_block();
            s.m.terminate(c_proceed_blk, Term::CondBr(c_more, cb, c_done_blk));
            let ctype = s.ray_query_state_i32(cb, rq_op::CANDIDATE_TYPE, rq);
            let is_tri = s.m.cmp(cb, pred::EQ, ctype, zero);
            s.m.terminate(cb, Term::CondBr(is_tri, c_tri_blk, c_latch_blk));
            Some(cb)
        };

        let is_ff = s.ray_query_state_i1(c_tri_blk, rq_op::CANDIDATE_TRIANGLE_FRONT_FACE, rq);
        s.m.terminate(c_tri_blk, Term::CondBr(is_ff, c_front_blk, c_back_blk));

        let new_min = if needs_min_front {
            let cand_t = s.ray_query_state_f32(c_front_blk, rq_op::CANDIDATE_TRIANGLE_RAY_T, rq);
            let lt = s.m.cmp(c_front_blk, pred::OLT, cand_t, min_phi);
            s.m.select(c_front_blk, lt, cand_t, min_phi)
        } else {
            min_phi
        };
        s.m.terminate(c_front_blk, Term::Br(c_latch_blk));

        // Only commit a double-sided back-face when Pass B missed OR it is strictly closer than `b_t`.
        let cand_t_b = s.ray_query_state_f32(c_back_blk, rq_op::CANDIDATE_TRIANGLE_RAY_T, rq);
        let lt_b = s.m.cmp(c_back_blk, pred::OLT, cand_t_b, b_t);
        let true_v = s.m.bool(true);
        let not_b_hit = s.m.bin(c_back_blk, Bin::Xor, b_hit, true_v);
        let allow_commit = s.m.bin(c_back_blk, Bin::Or, not_b_hit, lt_b);
        s.m.terminate(c_back_blk, Term::CondBr(allow_commit, c_commit_blk, c_latch_blk));

        s.ray_query_commit_non_opaque_triangle_hit(c_commit_blk, rq);
        s.m.terminate(c_commit_blk, Term::Br(c_latch_blk));

        let min_latch = s.m.phi(c_latch_blk, f32t);
        let back_t_latch = s.m.phi(c_latch_blk, f32t);
        let mut latch_in = vec![
            (new_min, c_front_blk),
            (min_phi, c_back_blk),
            (min_phi, c_commit_blk),
        ];
        let mut back_t_latch_in = vec![
            (back_t_phi, c_front_blk),
            (back_t_phi, c_back_blk),
            (cand_t_b, c_commit_blk),
        ];
        if let Some(cb) = c_cand_blk {
            latch_in.push((min_phi, cb));
            back_t_latch_in.push((back_t_phi, cb));
        }
        s.m.set_phi(min_latch, latch_in);
        s.m.set_phi(back_t_latch, back_t_latch_in);
        s.m.terminate(c_latch_blk, Term::Br(c_proceed_blk));
        s.m.set_phi(min_phi, vec![(min_front_a, c_trace_blk), (min_latch, c_latch_blk)]);
        s.m.set_phi(back_t_phi, vec![(t_max_c, c_trace_blk), (back_t_latch, c_latch_blk)]);

        let c_status = s.ray_query_state_i32(c_done_blk, rq_op::COMMITTED_STATUS, rq);
        let c_hit_raw = s.m.cmp(c_done_blk, pred::EQ, c_status, one);
        s.m.terminate(c_done_blk, Term::Br(after_blk));

        let false_v = s.m.bool(false);
        let c_hit = s.m.phi(after_blk, s.t.i1);
        s.m.set_phi(c_hit, vec![(c_hit_raw, c_done_blk), (false_v, c_prep_blk)]);
        // Track Pass C's committed back-face `t` directly from `cand_t_b` (RDNA 4 `2*tMin` guard).
        let c_back_t = s.m.phi(after_blk, f32t);
        s.m.set_phi(c_back_t, vec![(back_t_phi, c_done_blk), (b_t, c_prep_blk)]);
        let final_min_f = if needs_min_front {
            let mf = s.m.phi(after_blk, f32t);
            s.m.set_phi(mf, vec![(min_phi, c_done_blk), (min_front_a, c_prep_blk)]);
            Some(mf)
        } else {
            None
        };

        return emit_two_pass_merge(
            s,
            after_blk,
            rq,
            rl,
            inputs,
            two_pass_q1,
            c_hit,
            Some(c_back_t),
            final_min_f,
            &[],
            Vec::new(),
            None,
            i32t,
        );
    }

    // ARCH-12: Un-fused speculative opaque Pass 1 (`FORCE_OPAQUE`) -> out-of-loop one-shot
    // `Committed*` `AnyHit` check -> guarded `nextafter_down(t1)` Pass 2 `while (Proceed())` loop.
    // Eliminates `spec_phi`, `need_trace_phi`, and all `br i1 %spec_phi` diamonds inside the loop,
    // placing `TraceRayInline(p2)` strictly before Pass 2's clean `while (Proceed())` loop.
    if fno_spec && let RayLoopKind::AnyHit(ah) = &rl.kind {
        let p1_proceed_blk = s.m.new_block();
        let p1_check_blk = s.m.new_block();
        let p1_bridge_blk = s.m.new_block();
        let p2_trace_blk = s.m.new_block();
        let proceed_blk = s.m.new_block();
        let anyhit_bridge_blk = s.m.new_block();
        let do_commit_blk = s.m.new_block();
        let latch_blk = s.m.new_block();
        let p2_done_blk = s.m.new_block();
        let after_blk = s.m.new_block();

        let (dyn_flags_p2, dyn_bypass_blk, dyn_flags_raw) = match rl.ray_flags {
            RayInput::Const(orig_f) => {
                let dyn_flags_p1 = s.m.i32(
                    ((orig_f & !ray_flag::FORCE_NON_OPAQUE) | ray_flag::FORCE_OPAQUE)
                        & !inputs.group_const_flags,
                );
                s.ray_query_trace_inline(
                    start_blk,
                    rq,
                    inputs.tlas,
                    dyn_flags_p1,
                    mask,
                    origin,
                    t_min,
                    dir,
                    t_max,
                );
                s.m.terminate(start_blk, Term::Br(p1_proceed_blk));
                (s.m.i32(orig_f & !inputs.group_const_flags), None, None)
            }
            RayInput::Val(_) | RayInput::ShiftedMask(_) => {
                let dyn_f = inputs.ray_flags.unwrap_or_else(|| s.m.i32(0));
                let bad_bits_const = ray_flag::FORCE_OPAQUE
                    | ray_flag::CULL_OPAQUE
                    | ray_flag::CULL_NON_OPAQUE
                    | ray_flag::SKIP_TRIANGLES
                    | if !ah.state_vars.is_empty() {
                        ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
                    } else {
                        0
                    };
                let zero = s.m.i32(0);
                let bad_m = s.m.i32(bad_bits_const);
                let fno_m = s.m.i32(ray_flag::FORCE_NON_OPAQUE);
                let bad = s.m.bin(start_blk, Bin::And, dyn_f, bad_m);
                let no_bad = s.m.cmp(start_blk, pred::EQ, bad, zero);
                let fno = s.m.bin(start_blk, Bin::And, dyn_f, fno_m);
                let has_fno = s.m.cmp(start_blk, pred::NE, fno, zero);
                let can_spec = s.m.bin(start_blk, Bin::And, no_bad, has_fno);
                let p1_trace_blk = s.m.new_block();
                s.m.terminate(start_blk, Term::CondBr(can_spec, p1_trace_blk, p2_trace_blk));

                let clear_m = s.m.i32(!ray_flag::FORCE_NON_OPAQUE & !inputs.group_const_flags);
                let fo_m = s.m.i32(ray_flag::FORCE_OPAQUE & !inputs.group_const_flags);
                let p1_cleared = s.m.bin(p1_trace_blk, Bin::And, dyn_f, clear_m);
                let dyn_flags_p1 = s.m.bin(p1_trace_blk, Bin::Or, p1_cleared, fo_m);
                s.ray_query_trace_inline(
                    p1_trace_blk,
                    rq,
                    inputs.tlas,
                    dyn_flags_p1,
                    mask,
                    origin,
                    t_min,
                    dir,
                    t_max,
                );
                s.m.terminate(p1_trace_blk, Term::Br(p1_proceed_blk));

                let p2_flags = if inputs.group_const_flags != 0 {
                    let keep_m = s.m.i32(!inputs.group_const_flags);
                    s.m.bin(p2_trace_blk, Bin::And, dyn_f, keep_m)
                } else {
                    dyn_f
                };
                (p2_flags, Some(start_blk), Some(dyn_f))
            }
        };

        let p1_more = s.ray_query_proceed(p1_proceed_blk, rq);
        s.m.terminate(p1_proceed_blk, Term::CondBr(p1_more, p1_proceed_blk, p1_check_blk));

        let p1_status = s.ray_query_state_i32(p1_check_blk, rq_op::COMMITTED_STATUS, rq);
        let one = s.m.i32(1);
        let p1_hit = s.m.cmp(p1_check_blk, pred::EQ, p1_status, one);
        s.m.terminate(p1_check_blk, Term::CondBr(p1_hit, p1_bridge_blk, after_blk));

        // Guard 2 (`ACCEPT_FIRST_HIT` Ban): `nextafter_down(t1)` is ONLY legal when
        // `!ah.accept_first_hit` AND `(flags & ACCEPT_FIRST_HIT_AND_END_SEARCH) == 0`.
        let allow_prefix_skip = !ah.accept_first_hit
            && match rl.ray_flags {
                RayInput::Const(orig_f) => (orig_f & ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH) == 0,
                RayInput::Val(_) | RayInput::ShiftedMask(_) => true,
            };

        let needs_committed_ray_t = rl
            .outputs
            .iter()
            .any(|b| matches!(b.output, RayOutput::CommittedRayT));
        let ah_reads_ray_t = ah
            .inputs
            .iter()
            .any(|&(_, ci)| ci == CandidateInput::RayT);

        // Read `p1_t = COMMITTED_RAY_T(rq)` ONCE in `p1_bridge_blk` (before Pass 2 starts).
        let (p1_t_f32, p1_t_i32) = if allow_prefix_skip || needs_committed_ray_t || ah_reads_ray_t {
            let tf = s.ray_query_state_f32(p1_bridge_blk, rq_op::COMMITTED_RAY_T, rq);
            let ti = s.m.cast(p1_bridge_blk, Cast::Bitcast, tf, i32t);
            (Some(tf), Some(ti))
        } else {
            (None, None)
        };

        // Materialize Pass 1's `AnyHit` inputs directly from `Committed*` + `init_vals` in
        // `p1_bridge_blk` (strictly outside Pass 2's loop; zero `spec_phi`).
        let mut p1_cache: HashMap<CandidateInput, V> = HashMap::new();
        if let Some(ti) = p1_t_i32 {
            p1_cache.insert(CandidateInput::RayT, ti);
        }
        let mut p1_cand_outs: Vec<(Value, V)> = Vec::with_capacity(ah.inputs.len());
        for &(val_id, ci) in &ah.inputs {
            let v = match ci {
                CandidateInput::RayT | CandidateInput::Bary(_) | CandidateInput::FrontFace => {
                    *p1_cache.entry(ci).or_insert_with(|| {
                        let comm_out = match ci {
                            CandidateInput::RayT => RayOutput::CommittedRayT,
                            CandidateInput::Bary(c) => RayOutput::CommittedBary(c),
                            CandidateInput::FrontFace => RayOutput::CommittedFrontFace,
                            _ => unreachable!(),
                        };
                        emit_committed_hit_output(s, p1_bridge_blk, rq, comm_out, i32t)
                    })
                }
                CandidateInput::One => s.m.i32(1),
                CandidateInput::Zero => s.m.i32(0),
                CandidateInput::CopyOf(pre) => inputs
                    .fallbacks
                    .get(&pre)
                    .copied()
                    .unwrap_or_else(|| s.m.i32(0)),
                CandidateInput::StateVar(k) => inputs.anyhit_inits[k],
            };
            p1_cand_outs.push((val_id, v));
        }

        // Pass 2 preheader (`p2_trace_blk`): strictly OUTSIDE Pass 2's `while (Proceed())` loop.
        let i1t = s.t.i1;
        let p2_from_p1_phi =
            (dyn_bypass_blk.is_some() && (needs_committed_ray_t || allow_prefix_skip))
                .then(|| s.m.phi(p2_trace_blk, i1t));
        let p2_t1_phi = (dyn_bypass_blk.is_some() && allow_prefix_skip)
            .then(|| s.m.phi(p2_trace_blk, f32t));

        // Exact `nextafter_down(t1)` prefix skip with Guard 1 (NaN Underflow: `t1 > tMin && t1 > 0.0`)
        // and Guard 2 (`ACCEPT_FIRST_HIT` ban, including dynamic flag check when `ray_flags` is dynamic).
        let t_min_p2 = if allow_prefix_skip {
            let t1 = p2_t1_phi.unwrap_or_else(|| p1_t_f32.expect("p1_t_f32"));
            let f_zero = s.m.f32(0.0f32.to_bits());
            let gt_tmin = s.m.cmp(p2_trace_blk, pred::OGT, t1, t_min);
            let gt_zero = s.m.cmp(p2_trace_blk, pred::OGT, t1, f_zero);
            let mut ok_skip = s.m.bin(p2_trace_blk, Bin::And, gt_tmin, gt_zero);
            if let Some(fp1) = p2_from_p1_phi {
                ok_skip = s.m.bin(p2_trace_blk, Bin::And, ok_skip, fp1);
            }
            if let Some(dyn_f) = dyn_flags_raw {
                let afh_m = s.m.i32(ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH);
                let zero = s.m.i32(0);
                let afh_bits = s.m.bin(p2_trace_blk, Bin::And, dyn_f, afh_m);
                let no_afh = s.m.cmp(p2_trace_blk, pred::EQ, afh_bits, zero);
                ok_skip = s.m.bin(p2_trace_blk, Bin::And, ok_skip, no_afh);
            }
            let t1_bits = s.m.cast(p2_trace_blk, Cast::Bitcast, t1, i32t);
            let prev_bits = s.m.bin(p2_trace_blk, Bin::Sub, t1_bits, one);
            let prev_f = s.m.cast(p2_trace_blk, Cast::Bitcast, prev_bits, f32t);
            let clamped = s.op(p2_trace_blk, "binary.f32", Attr::ReadNone, f32t, 35, &[t_min, prev_f]);
            s.m.select(p2_trace_blk, ok_skip, clamped, t_min)
        } else {
            t_min
        };

        s.ray_query_trace_inline(
            p2_trace_blk,
            rq,
            inputs.tlas,
            dyn_flags_p2,
            mask,
            origin,
            t_min_p2,
            dir,
            t_max,
        );
        s.m.terminate(p2_trace_blk, Term::Br(proceed_blk));

        // Clean Pass 2 `while (Proceed())` loop: zero `spec_phi`, zero `need_trace_phi`.
        // Guard 3 (State-Var Reset Rule): `state_phis[k]` enters from `p2_trace_blk` with `init_vals[k]`.
        let mut state_phis = Vec::with_capacity(ah.state_vars.len());
        for _ in &ah.state_vars {
            state_phis.push(s.m.phi(proceed_blk, i32t));
        }
        // Guard 4 (AMD RDNA 4 `2 * tMin` Bug Guard): track Pass 2's committed `t` via `p2_t_phi`
        // from `CANDIDATE_TRIANGLE_RAY_T`, NEVER reading `COMMITTED_RAY_T(rq)` after Pass 2 commit!
        let p2_t_phi = needs_committed_ray_t.then(|| s.m.phi(proceed_blk, i32t));

        let more = s.ray_query_proceed(proceed_blk, rq);
        let cand_check_blk = if skip_proc {
            s.m.terminate(proceed_blk, Term::CondBr(more, anyhit_bridge_blk, p2_done_blk));
            None
        } else {
            let cb = s.m.new_block();
            s.m.terminate(proceed_blk, Term::CondBr(more, cb, p2_done_blk));
            let ctype = s.ray_query_state_i32(cb, rq_op::CANDIDATE_TYPE, rq);
            let zero = s.m.i32(0);
            let is_tri = s.m.cmp(cb, pred::EQ, ctype, zero);
            s.m.terminate(cb, Term::CondBr(is_tri, anyhit_bridge_blk, latch_blk));
            Some(cb)
        };

        // Pass 2 `AnyHit` inputs: zero-cost bindings here, lazy `RayT`/`Bary`/`FrontFace` via `Lower`.
        let mut cand_outs: Vec<(Value, V)> = Vec::with_capacity(ah.inputs.len());
        for &(val_id, ci) in &ah.inputs {
            let v = match ci {
                CandidateInput::RayT | CandidateInput::Bary(_) | CandidateInput::FrontFace => {
                    continue;
                }
                CandidateInput::One => s.m.i32(1),
                CandidateInput::Zero => s.m.i32(0),
                CandidateInput::CopyOf(pre) => inputs
                    .fallbacks
                    .get(&pre)
                    .copied()
                    .unwrap_or_else(|| s.m.i32(0)),
                CandidateInput::StateVar(k) => state_phis[k],
            };
            cand_outs.push((val_id, v));
        }
        let anyhit_entry_dxil = inputs.anyhit_entry_dxil.expect("anyhit entry dxil block");
        s.m.terminate(anyhit_bridge_blk, Term::Br(anyhit_entry_dxil));

        let commit_ray_t = needs_committed_ray_t
            .then(|| emit_cand_input(s, do_commit_blk, rq, CandidateInput::RayT));
        s.ray_query_commit_non_opaque_triangle_hit(do_commit_blk, rq);
        if ah.accept_first_hit {
            s.ray_query_abort(do_commit_blk, rq);
        }
        s.m.terminate(do_commit_blk, Term::Br(latch_blk));

        let mut latch_phis = Vec::new();
        if cand_check_blk.is_some() {
            latch_phis.reserve(ah.state_vars.len());
            for _ in &ah.state_vars {
                latch_phis.push(s.m.phi(latch_blk, i32t));
            }
        }
        let p2_t_track = if let (Some(pt_phi), Some(ct)) = (p2_t_phi, commit_ray_t) {
            let pt_latch = s.m.phi(latch_blk, i32t);
            s.m.set_phi(pt_phi, vec![(inputs.t_max, p2_trace_blk), (pt_latch, latch_blk)]);
            Some((pt_phi, pt_latch, ct))
        } else {
            None
        };
        s.m.terminate(latch_blk, Term::Br(proceed_blk));

        let mut final_state_phis = Vec::with_capacity(ah.state_vars.len());
        for _ in &ah.state_vars {
            final_state_phis.push(s.m.phi(after_blk, i32t));
        }

        let (p2_exit_blk, p2_ray_t) = if let Some(pt_phi) = p2_t_phi {
            if let Some(fp1) = p2_from_p1_phi {
                // Dynamic bypass (`!fp1`): Pass 1 never ran on `rq` and `t_min_p2 == t_min`,
                // so `COMMITTED_RAY_T` is safe and covers opaque auto-commits too.
                let p2_dyn_read_blk = s.m.new_block();
                let p2_done_join = s.m.new_block();
                s.m.terminate(
                    p2_done_blk,
                    Term::CondBr(fp1, p2_done_join, p2_dyn_read_blk),
                );
                let dyn_t = emit_committed_hit_output(
                    s,
                    p2_dyn_read_blk,
                    rq,
                    RayOutput::CommittedRayT,
                    i32t,
                );
                s.m.terminate(p2_dyn_read_blk, Term::Br(p2_done_join));
                let joined_t = s.m.phi(p2_done_join, i32t);
                s.m.set_phi(
                    joined_t,
                    vec![(pt_phi, p2_done_blk), (dyn_t, p2_dyn_read_blk)],
                );
                s.m.terminate(p2_done_join, Term::Br(after_blk));
                (p2_done_join, Some(joined_t))
            } else {
                s.m.terminate(p2_done_blk, Term::Br(after_blk));
                (p2_done_blk, Some(pt_phi))
            }
        } else {
            s.m.terminate(p2_done_blk, Term::Br(after_blk));
            (p2_done_blk, None)
        };

        let needs_status = rl.outputs.iter().any(|b| {
            matches!(
                b.output,
                RayOutput::CommittedStatusTriNode
                    | RayOutput::CommittedRayT
                    | RayOutput::CommittedBary(_)
                    | RayOutput::CommittedFrontFace
            )
        });
        let (hit_tri, hit_any) = if needs_status {
            let status = s.ray_query_state_i32(after_blk, rq_op::COMMITTED_STATUS, rq);
            let zero = s.m.i32(0);
            (
                Some(s.m.cmp(after_blk, pred::EQ, status, one)),
                Some(s.m.cmp(after_blk, pred::NE, status, zero)),
            )
        } else {
            (None, None)
        };
        let mut final_t_phi: Option<(V, V, V)> = None;
        let mut raw_cache: HashMap<RayOutput, V> = HashMap::new();
        let mut outs = Vec::with_capacity(rl.outputs.len());
        for binding in &rl.outputs {
            let v = match binding.output {
                RayOutput::CommittedStatusTriNode => {
                    let zero = s.m.i32(0);
                    s.m.select(after_blk, hit_tri.expect("status"), one, zero)
                }
                RayOutput::CommittedRayT => {
                    let t_i = *raw_cache.entry(RayOutput::CommittedRayT).or_insert_with(|| {
                        let p2t = p2_ray_t.expect("p2_ray_t");
                        let phi = s.m.phi(after_blk, i32t);
                        final_t_phi = Some((phi, inputs.t_max, p2t));
                        phi
                    });
                    let miss_t = binding
                        .fallback
                        .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                        .unwrap_or(inputs.t_max);
                    s.m.select(after_blk, hit_any.expect("status"), t_i, miss_t)
                }
                RayOutput::CommittedBary(comp) => {
                    let b_i = *raw_cache
                        .entry(RayOutput::CommittedBary(comp))
                        .or_insert_with(|| {
                            emit_committed_hit_output(
                                s,
                                after_blk,
                                rq,
                                RayOutput::CommittedBary(comp),
                                i32t,
                            )
                        });
                    let miss_b = binding
                        .fallback
                        .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                        .unwrap_or_else(|| s.m.i32(0));
                    s.m.select(after_blk, hit_any.expect("status"), b_i, miss_b)
                }
                RayOutput::CommittedFrontFace => {
                    let ff_v = *raw_cache
                        .entry(RayOutput::CommittedFrontFace)
                        .or_insert_with(|| {
                            emit_committed_hit_output(
                                s,
                                after_blk,
                                rq,
                                RayOutput::CommittedFrontFace,
                                i32t,
                            )
                        });
                    let miss_ff = binding
                        .fallback
                        .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                        .unwrap_or_else(|| s.m.i32(0));
                    s.m.select(after_blk, hit_any.expect("status"), ff_v, miss_ff)
                }
                committed @ (RayOutput::CommittedInstanceIndex
                | RayOutput::CommittedInstanceId
                | RayOutput::CommittedInstanceContribution
                | RayOutput::CommittedGeometryIndex
                | RayOutput::CommittedPrimitiveIndex
                | RayOutput::CommittedObjectToWorld(_, _)
                | RayOutput::CommittedWorldToObject(_, _)) => *raw_cache
                    .entry(committed)
                    .or_insert_with(|| emit_committed_hit_output(s, after_blk, rq, committed, i32t)),
                RayOutput::AnyHitState(k) => final_state_phis[k],
                RayOutput::Zero => s.m.i32(0),
                RayOutput::CopyOf(pre) => inputs
                    .fallbacks
                    .get(&pre)
                    .copied()
                    .unwrap_or_else(|| s.m.i32(0)),
                RayOutput::FrontBackMinFrontT => {
                    unreachable!("speculative FNO has no FrontBackMinFrontT")
                }
            };
            outs.push((binding.value, v));
        }

        let pending_anyhit = Some(PendingAnyHit {
            start_blk: p2_trace_blk,
            cand_check_blk,
            anyhit_bridge_blk,
            do_commit_blk,
            latch_blk,
            state_phis,
            latch_phis,
            init_vals: inputs.anyhit_inits.clone(),
            spec: Some(SpeculativeAnyHit {
                p1_bridge_blk,
                p1_cand_outs,
                p1_t_f32,
                p1_t_i32,
                p1_check_blk,
                dyn_bypass_blk,
                t_min_f32: t_min,
                p2_from_p1_phi,
                p2_t1_phi,
                p2_t_track,
                p2_trace_blk,
                p2_exit_blk,
                after_blk,
                final_t_phi,
                final_state_phis,
            }),
        });
        return (after_blk, outs, cand_outs, pending_anyhit, None);
    }

    // Two-pass tMax pre-shrink (AnyHit, always single-handle early-extract) or 2-pass pure-opaque
    // FrontBackTrack (ARCH-13 Mirror TLAS or legacy opt-in FBT two-pass).
    let (trace_start_blk, two_pass_q1, fbt_two_pass_min_f) = if let RayLoopKind::AnyHit(_) = &rl.kind
        && let Some(fields) = &inputs.two_pass_early_fields
    {
        let orig_f = match rl.ray_flags {
            RayInput::Const(f) => f,
            _ => unreachable!("two-pass requires constant ray flags"),
        };
        let dyn_flags_p1 = s.m.i32((orig_f | ray_flag::CULL_NON_OPAQUE) & !inputs.group_const_flags);
        s.ray_query_trace_inline(
            start_blk,
            rq,
            inputs.tlas,
            dyn_flags_p1,
            mask,
            origin,
            t_min,
            dir,
            t_max,
        );
        let p1_proceed_blk = s.m.new_block();
        let p1_done_blk = s.m.new_block();
        let p1_extract_blk = s.m.new_block();
        let p2_start_blk = s.m.new_block();
        s.m.terminate(start_blk, Term::Br(p1_proceed_blk));

        let p1_more = s.ray_query_proceed(p1_proceed_blk, rq);
        s.m.terminate(p1_proceed_blk, Term::CondBr(p1_more, p1_proceed_blk, p1_done_blk));

        let p1_status = s.ray_query_state_i32(p1_done_blk, rq_op::COMMITTED_STATUS, rq);
        let one = s.m.i32(1);
        let q1_hit = s.m.cmp(p1_done_blk, pred::EQ, p1_status, one);
        s.m.terminate(p1_done_blk, Term::CondBr(q1_hit, p1_extract_blk, p2_start_blk));

        // Read `CommittedRayT(rq)` and all unique Pass 1 fields once before Pass 2 starts,
        // releasing Pass 1's `RayQuery` state and reusing the single `rq` handle directly.
        let p1_t_hit = s.ray_query_state_f32(p1_extract_blk, rq_op::COMMITTED_RAY_T, rq);
        let mut extracted_hit: Vec<(EarlyQ1Field, V)> = Vec::with_capacity(fields.len());
        for &field in fields {
            let v = emit_early_q1_field(s, p1_extract_blk, rq, field, i32t);
            extracted_hit.push((field, v));
        }
        s.m.terminate(p1_extract_blk, Term::Br(p2_start_blk));

        let t_clamp = s.m.phi(p2_start_blk, f32t);
        s.m.set_phi(t_clamp, vec![(p1_t_hit, p1_extract_blk), (t_max, p1_done_blk)]);
        let zero = s.m.i32(0);
        let mut q1_fields: HashMap<EarlyQ1Field, V> = HashMap::with_capacity(fields.len());
        for (field, v_hit) in extracted_hit {
            let phi = s.m.phi(p2_start_blk, i32t);
            s.m.set_phi(phi, vec![(v_hit, p1_extract_blk), (zero, p1_done_blk)]);
            q1_fields.insert(field, phi);
        }

        let dyn_flags_p2 =
            s.m.i32((orig_f | ray_flag::CULL_OPAQUE) & !inputs.group_const_flags);
        s.ray_query_trace_inline(
            p2_start_blk,
            rq,
            inputs.tlas,
            dyn_flags_p2,
            mask,
            origin,
            t_min,
            dir,
            t_clamp,
        );
        (
            p2_start_blk,
            Some(EarlyQ1 {
                q1_hit,
                t_clamp,
                q1_fields,
            }),
            None,
        )
    } else if fbt_two_pass {
        let orig_f = match rl.ray_flags {
            RayInput::Const(f) => f,
            _ => unreachable!("two-pass FrontBackTrack requires constant ray flags"),
        };
        let base_opaque = (orig_f & !ray_flag::FORCE_NON_OPAQUE) | ray_flag::FORCE_OPAQUE;
        let needs_min_front = rl
            .outputs
            .iter()
            .any(|b| matches!(b.output, RayOutput::FrontBackMinFrontT));

        // Pass A: pure-opaque front-face trace (`FORCE_OPAQUE | CULL_BACK_FACING_TRIANGLES`).
        // Read `COMMITTED_STATUS` and `COMMITTED_RAY_T` once before Pass B starts so no
        // `COMMITTED_RAY_T` is ever re-read across a subsequent trace (RDNA 4 quirk).
        let (b_start_blk, min_f) = if needs_min_front {
            let dyn_flags_front = s.m.i32(
                (base_opaque | ray_flag::CULL_BACK_FACING_TRIANGLES) & !inputs.group_const_flags,
            );
            s.ray_query_trace_inline(
                start_blk,
                rq,
                inputs.tlas,
                dyn_flags_front,
                mask,
                origin,
                t_min,
                dir,
                t_max,
            );
            let f_proceed_blk = s.m.new_block();
            let next_blk = s.m.new_block();
            s.m.terminate(start_blk, Term::Br(f_proceed_blk));

            let f_more = s.ray_query_proceed(f_proceed_blk, rq);
            s.m.terminate(f_proceed_blk, Term::CondBr(f_more, f_proceed_blk, next_blk));

            let f_status = s.ray_query_state_i32(next_blk, rq_op::COMMITTED_STATUS, rq);
            let one = s.m.i32(1);
            let f_hit = s.m.cmp(next_blk, pred::EQ, f_status, one);
            let f_t = s.ray_query_state_f32(next_blk, rq_op::COMMITTED_RAY_T, rq);
            let init_i = inputs.init_min_front_t.unwrap_or(inputs.t_max);
            let init_f = s.m.cast(next_blk, Cast::Bitcast, init_i, f32t);
            let f_lt = s.m.cmp(next_blk, pred::OLT, f_t, init_f);
            let use_f = s.m.bin(next_blk, Bin::And, f_hit, f_lt);
            let min_front = s.m.select(next_blk, use_f, f_t, init_f);
            (next_blk, Some(min_front))
        } else {
            (start_blk, None)
        };

        // Pass B: pure-opaque back-face trace (`FORCE_OPAQUE | CULL_FRONT_FACING_TRIANGLES`)
        // with full `t_max` so back-faces behind front-faces still commit in `rq` (also
        // serving all post-loop `committed_meta_loads` directly from `rq`).
        let dyn_flags_back = s.m.i32(
            (base_opaque | ray_flag::CULL_FRONT_FACING_TRIANGLES) & !inputs.group_const_flags,
        );
        s.ray_query_trace_inline(
            b_start_blk,
            rq,
            inputs.tlas,
            dyn_flags_back,
            mask,
            origin,
            t_min,
            dir,
            t_max,
        );
        (b_start_blk, None, min_f)
    } else {
        let dyn_flags = match rl.ray_flags {
            RayInput::Const(f) => s.m.i32(f & !inputs.group_const_flags),
            RayInput::Val(_) => inputs.ray_flags.unwrap_or_else(|| s.m.i32(0)),
            RayInput::ShiftedMask(_) => s.m.i32(0),
        };
        s.ray_query_trace_inline(
            start_blk,
            rq,
            inputs.tlas,
            dyn_flags,
            mask,
            origin,
            t_min,
            dir,
            t_max,
        );
        (start_blk, None, None)
    };

    let proceed_blk = s.m.new_block();
    let after_blk = s.m.new_block();
    s.m.terminate(trace_start_blk, Term::Br(proceed_blk));

    let mut front_back_min_f: Option<V> = None;
    let mut state_phis: Vec<V> = Vec::new();
    let mut cand_outs: Vec<(Value, V)> = Vec::new();
    let mut pending_anyhit: Option<PendingAnyHit> = None;

    match &rl.kind {
        RayLoopKind::OpaqueOnly => {
            let more = s.ray_query_proceed(proceed_blk, rq);
            s.m.terminate(proceed_blk, Term::CondBr(more, proceed_blk, after_blk));
        }
        RayLoopKind::FrontBackTrack { .. } if fbt_two_pass => {
            let more = s.ray_query_proceed(proceed_blk, rq);
            s.m.terminate(proceed_blk, Term::CondBr(more, proceed_blk, after_blk));
            front_back_min_f = fbt_two_pass_min_f;
        }
        RayLoopKind::FrontBackTrack { .. } => {
            let init_i = inputs.init_min_front_t.unwrap_or(inputs.t_max);
            let init_f = s.m.cast(trace_start_blk, Cast::Bitcast, init_i, f32t);
            let min_phi = s.m.phi(proceed_blk, f32t);
            let tri_blk = s.m.new_block();
            let front_blk = s.m.new_block();
            let back_blk = s.m.new_block();
            let latch_blk = s.m.new_block();

            let more = s.ray_query_proceed(proceed_blk, rq);
            let cand_blk = if skip_proc {
                s.m.terminate(proceed_blk, Term::CondBr(more, tri_blk, after_blk));
                None
            } else {
                let cb = s.m.new_block();
                s.m.terminate(proceed_blk, Term::CondBr(more, cb, after_blk));
                let ctype = s.ray_query_state_i32(cb, rq_op::CANDIDATE_TYPE, rq);
                let zero = s.m.i32(0);
                let is_tri = s.m.cmp(cb, pred::EQ, ctype, zero);
                s.m.terminate(cb, Term::CondBr(is_tri, tri_blk, latch_blk));
                Some(cb)
            };

            let is_ff = s.ray_query_state_i1(tri_blk, rq_op::CANDIDATE_TRIANGLE_FRONT_FACE, rq);
            s.m.terminate(tri_blk, Term::CondBr(is_ff, front_blk, back_blk));

            let cand_t = s.ray_query_state_f32(front_blk, rq_op::CANDIDATE_TRIANGLE_RAY_T, rq);
            let lt = s.m.cmp(front_blk, pred::OLT, cand_t, min_phi);
            let new_min = s.m.select(front_blk, lt, cand_t, min_phi);
            s.m.terminate(front_blk, Term::Br(latch_blk));

            s.ray_query_commit_non_opaque_triangle_hit(back_blk, rq);
            s.m.terminate(back_blk, Term::Br(latch_blk));

            let min_latch = s.m.phi(latch_blk, f32t);
            let mut latch_in = vec![(new_min, front_blk), (min_phi, back_blk)];
            if let Some(cb) = cand_blk {
                latch_in.push((min_phi, cb));
            }
            s.m.set_phi(min_latch, latch_in);
            s.m.terminate(latch_blk, Term::Br(proceed_blk));
            s.m.set_phi(min_phi, vec![(init_f, trace_start_blk), (min_latch, latch_blk)]);
            front_back_min_f = Some(min_phi);
        }
        RayLoopKind::AnyHit(ah) => {
            let anyhit_bridge_blk = s.m.new_block();
            let do_commit_blk = s.m.new_block();
            let latch_blk = s.m.new_block();

            for _ in &ah.state_vars {
                state_phis.push(s.m.phi(proceed_blk, i32t));
            }
            let more = s.ray_query_proceed(proceed_blk, rq);
            let cand_check_blk = if skip_proc {
                s.m.terminate(proceed_blk, Term::CondBr(more, anyhit_bridge_blk, after_blk));
                None
            } else {
                let cb = s.m.new_block();
                s.m.terminate(proceed_blk, Term::CondBr(more, cb, after_blk));
                let ctype = s.ray_query_state_i32(cb, rq_op::CANDIDATE_TYPE, rq);
                let zero = s.m.i32(0);
                let is_tri = s.m.cmp(cb, pred::EQ, ctype, zero);
                s.m.terminate(cb, Term::CondBr(is_tri, anyhit_bridge_blk, latch_blk));
                Some(cb)
            };

            // Only bind zero-cost candidate inputs here; `RayT`, `Bary`, and `FrontFace` are
            // materialized lazily on demand inside `Lower` so skipped/fast-rejected candidates
            // avoid unnecessary `rayQuery_State*` intrinsics.
            for &(val_id, ci) in &ah.inputs {
                let v = match ci {
                    CandidateInput::RayT | CandidateInput::Bary(_) | CandidateInput::FrontFace => {
                        continue;
                    }
                    CandidateInput::One => s.m.i32(1),
                    CandidateInput::Zero => s.m.i32(0),
                    CandidateInput::CopyOf(pre) => inputs
                        .fallbacks
                        .get(&pre)
                        .copied()
                        .unwrap_or_else(|| s.m.i32(0)),
                    CandidateInput::StateVar(k) => state_phis[k],
                };
                cand_outs.push((val_id, v));
            }
            let anyhit_entry_dxil = inputs.anyhit_entry_dxil.expect("anyhit entry dxil block");
            s.m.terminate(anyhit_bridge_blk, Term::Br(anyhit_entry_dxil));

            s.ray_query_commit_non_opaque_triangle_hit(do_commit_blk, rq);
            if ah.accept_first_hit {
                s.ray_query_abort(do_commit_blk, rq);
            }
            s.m.terminate(do_commit_blk, Term::Br(latch_blk));

            // When `cand_check_blk` is elided, `tail_blk` dominates `latch_blk` and `do_commit_blk`
            // does not alter `state_vars`, so `next_vals[k]` feeds `state_phis[k]` directly.
            let mut latch_phis = Vec::new();
            if cand_check_blk.is_some() {
                latch_phis.reserve(ah.state_vars.len());
                for _ in &ah.state_vars {
                    latch_phis.push(s.m.phi(latch_blk, i32t));
                }
            }
            s.m.terminate(latch_blk, Term::Br(proceed_blk));

            pending_anyhit = Some(PendingAnyHit {
                start_blk: trace_start_blk,
                cand_check_blk,
                anyhit_bridge_blk,
                do_commit_blk,
                latch_blk,
                state_phis: state_phis.clone(),
                latch_phis,
                init_vals: inputs.anyhit_inits.clone(),
                spec: None,
            });
        }
    }

    if let Some(tp) = two_pass_q1 {
        let q2_status = s.ray_query_state_i32(after_blk, rq_op::COMMITTED_STATUS, rq);
        let one = s.m.i32(1);
        let q2_hit = s.m.cmp(after_blk, pred::EQ, q2_status, one);
        return emit_two_pass_merge(
            s,
            after_blk,
            rq,
            rl,
            inputs,
            tp,
            q2_hit,
            None,
            front_back_min_f,
            &state_phis,
            cand_outs,
            pending_anyhit,
            i32t,
        );
    }

    // Read `CommittedStatus` once if any output needs hit/miss selection.
    let needs_status = rl.outputs.iter().any(|b| {
        matches!(
            b.output,
            RayOutput::CommittedStatusTriNode
                | RayOutput::CommittedRayT
                | RayOutput::CommittedBary(_)
                | RayOutput::CommittedFrontFace
        )
    });
    let (hit_tri, hit_any) = if needs_status {
        let status = s.ray_query_state_i32(after_blk, rq_op::COMMITTED_STATUS, rq);
        let zero = s.m.i32(0);
        let one = s.m.i32(1);
        (
            Some(s.m.cmp(after_blk, pred::EQ, status, one)),
            Some(s.m.cmp(after_blk, pred::NE, status, zero)),
        )
    } else {
        (None, None)
    };

    let mut raw_cache: HashMap<RayOutput, V> = HashMap::new();
    let mut outs = Vec::with_capacity(rl.outputs.len());
    for binding in &rl.outputs {
        let v = match binding.output {
            RayOutput::CommittedStatusTriNode => {
                let one = s.m.i32(1);
                let zero = s.m.i32(0);
                s.m.select(after_blk, hit_tri.expect("status"), one, zero)
            }
            RayOutput::CommittedRayT => {
                let t_i = *raw_cache.entry(RayOutput::CommittedRayT).or_insert_with(|| {
                    emit_committed_hit_output(s, after_blk, rq, RayOutput::CommittedRayT, i32t)
                });
                let miss_t = binding
                    .fallback
                    .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                    .unwrap_or(inputs.t_max);
                s.m.select(after_blk, hit_any.expect("status"), t_i, miss_t)
            }
            RayOutput::CommittedBary(comp) => {
                let b_i = *raw_cache
                    .entry(RayOutput::CommittedBary(comp))
                    .or_insert_with(|| {
                        emit_committed_hit_output(
                            s,
                            after_blk,
                            rq,
                            RayOutput::CommittedBary(comp),
                            i32t,
                        )
                    });
                let miss_b = binding
                    .fallback
                    .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                    .unwrap_or_else(|| s.m.i32(0));
                s.m.select(after_blk, hit_any.expect("status"), b_i, miss_b)
            }
            RayOutput::CommittedFrontFace => {
                let ff_v = *raw_cache
                    .entry(RayOutput::CommittedFrontFace)
                    .or_insert_with(|| {
                        emit_committed_hit_output(
                            s,
                            after_blk,
                            rq,
                            RayOutput::CommittedFrontFace,
                            i32t,
                        )
                    });
                let miss_ff = binding
                    .fallback
                    .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                    .unwrap_or_else(|| s.m.i32(0));
                s.m.select(after_blk, hit_any.expect("status"), ff_v, miss_ff)
            }
            committed @ (RayOutput::CommittedInstanceIndex
            | RayOutput::CommittedInstanceId
            | RayOutput::CommittedInstanceContribution
            | RayOutput::CommittedGeometryIndex
            | RayOutput::CommittedPrimitiveIndex
            | RayOutput::CommittedObjectToWorld(_, _)
            | RayOutput::CommittedWorldToObject(_, _)) => *raw_cache
                .entry(committed)
                .or_insert_with(|| emit_committed_hit_output(s, after_blk, rq, committed, i32t)),
            RayOutput::FrontBackMinFrontT => {
                let mf = front_back_min_f.expect("front_back min phi");
                s.m.cast(after_blk, Cast::Bitcast, mf, i32t)
            }
            RayOutput::AnyHitState(k) => state_phis[k],
            RayOutput::Zero => s.m.i32(0),
            RayOutput::CopyOf(pre) => inputs
                .fallbacks
                .get(&pre)
                .copied()
                .unwrap_or_else(|| s.m.i32(0)),
        };
        outs.push((binding.value, v));
    }

    (after_blk, outs, cand_outs, pending_anyhit, None)
}

#[allow(clippy::too_many_arguments)]
fn emit_two_pass_merge(
    s: &mut Shader,
    after_blk: usize,
    rq: V,
    rl: &RayLoop,
    inputs: &ResolvedRayInputs,
    tp: EarlyQ1,
    q2_hit: V,
    q2_ray_t_override: Option<V>,
    front_back_min_f: Option<V>,
    state_phis: &[V],
    cand_outs: Vec<(Value, V)>,
    pending_anyhit: Option<PendingAnyHit>,
    i32t: super::bitcode::Ty,
) -> (
    usize,
    Vec<(Value, V)>,
    Vec<(Value, V)>,
    Option<PendingAnyHit>,
    Option<TwoPassPost>,
) {
    let one = s.m.i32(1);
    let needs_hit_fields = rl.outputs.iter().any(|b| {
        !matches!(
            b.output,
            RayOutput::CommittedStatusTriNode
                | RayOutput::Zero
                | RayOutput::CopyOf(_)
                | RayOutput::AnyHitState(_)
                | RayOutput::FrontBackMinFrontT
        )
    });
    let EarlyQ1 {
        q1_hit,
        t_clamp,
        q1_fields,
    } = tp;
    if !needs_hit_fields {
        let any_hit = s.m.bin(after_blk, Bin::Or, q2_hit, q1_hit);
        let zero = s.m.i32(0);
        let mut outs = Vec::with_capacity(rl.outputs.len());
        for binding in &rl.outputs {
            let v = match binding.output {
                RayOutput::CommittedStatusTriNode => s.m.select(after_blk, any_hit, one, zero),
                RayOutput::AnyHitState(k) => state_phis[k],
                RayOutput::FrontBackMinFrontT => {
                    let mf = front_back_min_f.expect("front_back min phi");
                    s.m.cast(after_blk, Cast::Bitcast, mf, i32t)
                }
                RayOutput::Zero => zero,
                RayOutput::CopyOf(pre) => inputs
                    .fallbacks
                    .get(&pre)
                    .copied()
                    .unwrap_or(zero),
                _ => unreachable!(),
            };
            outs.push((binding.value, v));
        }
        return (
            after_blk,
            outs,
            cand_outs,
            pending_anyhit,
            Some(TwoPassPost {
                q2_hit,
                q1_hit,
                q1_fields,
            }),
        );
    }

    let miss_val_for = |s: &mut Shader, binding: &RayOutputBinding| -> V {
        match binding.output {
            RayOutput::CommittedRayT => binding
                .fallback
                .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                .unwrap_or(inputs.t_max),
            RayOutput::CommittedBary(_) | RayOutput::CommittedFrontFace => binding
                .fallback
                .and_then(|fb| inputs.fallbacks.get(&fb).copied())
                .unwrap_or_else(|| s.m.i32(0)),
            _ => s.m.i32(0),
        }
    };

    let mut q2_cache: HashMap<RayOutput, V> = HashMap::new();
    let mut phi_cache: HashMap<(RayOutput, Option<Value>), V> = HashMap::new();
    let from_q2_blk = s.m.new_block();
    let not_q2_blk = s.m.new_block();
    let done_blk = s.m.new_block();
    s.m.terminate(after_blk, Term::CondBr(q2_hit, from_q2_blk, not_q2_blk));

    let mut t_clamp_i: Option<V> = None;
    let mut outs = Vec::with_capacity(rl.outputs.len());
    for binding in &rl.outputs {
        let v = match binding.output {
            RayOutput::Zero => s.m.i32(0),
            RayOutput::CopyOf(pre) => inputs
                .fallbacks
                .get(&pre)
                .copied()
                .unwrap_or_else(|| s.m.i32(0)),
            RayOutput::AnyHitState(k) => state_phis[k],
            RayOutput::FrontBackMinFrontT => {
                let mf = front_back_min_f.expect("front_back min phi");
                s.m.cast(done_blk, Cast::Bitcast, mf, i32t)
            }
            committed => {
                if let Some(&cached) = phi_cache.get(&(committed, binding.fallback)) {
                    cached
                } else {
                    let v2 = *q2_cache.entry(committed).or_insert_with(|| {
                        if matches!(committed, RayOutput::CommittedRayT)
                            && let Some(t_f) = q2_ray_t_override
                        {
                            s.m.cast(from_q2_blk, Cast::Bitcast, t_f, i32t)
                        } else {
                            emit_committed_hit_output(s, from_q2_blk, rq, committed, i32t)
                        }
                    });
                    let v_miss = miss_val_for(s, binding);
                    let v_not_q2 = match committed {
                        RayOutput::CommittedStatusTriNode => {
                            let zero = s.m.i32(0);
                            s.m.select(not_q2_blk, q1_hit, one, zero)
                        }
                        RayOutput::CommittedRayT => {
                            let ti = *t_clamp_i.get_or_insert_with(|| {
                                s.m.cast(not_q2_blk, Cast::Bitcast, t_clamp, i32t)
                            });
                            s.m.select(not_q2_blk, q1_hit, ti, v_miss)
                        }
                        _ => {
                            let field = early_field_for_output(committed)
                                .expect("early q1 field for committed output");
                            let v1 = *q1_fields
                                .get(&field)
                                .expect("early q1 field pre-extracted");
                            if matches!(
                                committed,
                                RayOutput::CommittedBary(_) | RayOutput::CommittedFrontFace
                            ) {
                                s.m.select(not_q2_blk, q1_hit, v1, v_miss)
                            } else {
                                v1
                            }
                        }
                    };
                    let phi = s.m.phi(done_blk, i32t);
                    s.m.set_phi(phi, vec![(v2, from_q2_blk), (v_not_q2, not_q2_blk)]);
                    phi_cache.insert((committed, binding.fallback), phi);
                    phi
                }
            }
        };
        outs.push((binding.value, v));
    }
    s.m.terminate(from_q2_blk, Term::Br(done_blk));
    s.m.terminate(not_q2_blk, Term::Br(done_blk));
    (
        done_blk,
        outs,
        cand_outs,
        pending_anyhit,
        Some(TwoPassPost {
            q2_hit,
            q1_hit,
            q1_fields,
        }),
    )
}

// ---------------------------------------------------------------------------
// Small operand / constant helpers
// ---------------------------------------------------------------------------

fn atom_value(atom: Atom) -> Option<Value> {
    match atom {
        Atom::Lane(v) | Atom::Uniform(v) | Atom::Phi(v) | Atom::Quad(v) | Atom::Opaque(v) => {
            Some(v)
        }
        Atom::Constant(_) => None,
    }
}

/// Returns `true` if `root` can evaluate to `Bool::TRUE` when `target` is fixed to `val`.
fn bdd_can_reach_true_with(
    bools: &crate::structure::Bools,
    root: Bool,
    target: Atom,
    val: bool,
) -> bool {
    let mut seen: HashSet<Bool> = HashSet::new();
    let mut work: Vec<Bool> = vec![root];
    while let Some(b) = work.pop() {
        if b == Bool::TRUE {
            return true;
        }
        if b == Bool::FALSE || !seen.insert(b) {
            continue;
        }
        let Some((atom, lo, hi)) = bools.node(b) else {
            continue;
        };
        if atom == target {
            work.push(if val { hi } else { lo });
        } else {
            work.push(lo);
            work.push(hi);
        }
    }
    false
}

/// Returns the set of atoms still reachable in `root` after fixing every atom in `fixed`.
fn bdd_support_under_fixed(
    bools: &crate::structure::Bools,
    root: Bool,
    fixed: &HashMap<Atom, bool>,
) -> HashSet<Atom> {
    let mut out: HashSet<Atom> = HashSet::new();
    let mut seen: HashSet<Bool> = HashSet::new();
    let mut work: Vec<Bool> = vec![root];
    while let Some(b) = work.pop() {
        if b == Bool::TRUE || b == Bool::FALSE || !seen.insert(b) {
            continue;
        }
        let Some((atom, lo, hi)) = bools.node(b) else {
            continue;
        };
        match fixed.get(&atom) {
            Some(&true) => work.push(hi),
            Some(&false) => work.push(lo),
            None => {
                out.insert(atom);
                work.push(lo);
                work.push(hi);
            }
        }
    }
    out
}

fn ray_input_contains_val(arr: [RayInput; 3], v: Value) -> bool {
    arr.iter()
        .any(|inp| matches!(inp, RayInput::Val(x) | RayInput::ShiftedMask(x) if *x == v))
}

fn ray_loop_reads_real_input(rl: &RayLoop, v: Value) -> bool {
    ray_input_contains_val(rl.origin, v)
        || ray_input_contains_val(rl.dir, v)
        || matches!(rl.t_max, RayInput::Val(x) if x == v)
        || matches!(rl.t_min, Some(RayInput::Val(x)) if x == v)
        || rl.active_ray_mask == Some(v)
        || matches!(rl.instance_mask, RayInput::Val(x) | RayInput::ShiftedMask(x) if x == v)
        || matches!(rl.ray_flags, RayInput::Val(x) | RayInput::ShiftedMask(x) if x == v)
        || matches!(rl.kind, RayLoopKind::FrontBackTrack { init_min_front_t: RayInput::Val(x) } if x == v)
        || matches!(&rl.kind, RayLoopKind::AnyHit(ah) if ah.state_vars.iter().any(|sv| matches!(sv.init, RayInput::Val(x) if x == v))
            || ah.inputs.iter().any(|&(_, ci)| ci == CandidateInput::CopyOf(v)))
}

fn ray_loop_reads_value(rl: &RayLoop, v: Value) -> bool {
    ray_loop_reads_real_input(rl, v)
        || rl.outputs.iter().any(|o| {
            o.fallback == Some(v) || o.output == RayOutput::CopyOf(v)
        })
}

/// If `inst` is a `global_load_dword*` or `s_load_dword*` using `v` as its 64-bit lo address,
/// returns `(num_dwords, immediate_byte_offset)`.
fn flat_or_smem_load_info(inst: &Inst, v: Value) -> Option<(u32, u32)> {
    let n = inst.name();
    let ndw = match n {
        "global_load_dword" | "s_load_dword" => 1,
        "global_load_dwordx2" | "s_load_dwordx2" => 2,
        "global_load_dwordx3" => 3,
        "global_load_dwordx4" | "s_load_dwordx4" => 4,
        _ => return None,
    };
    match &inst.fields {
        Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            offset,
            ..
        } if inst.read(Reg::V(*reg)) == Some(v) => u32::try_from(*offset).ok().map(|o| (ndw, o)),
        Fields::Smem {
            base: Operand::Scalar { code, .. },
            offset,
            soffset,
            ..
        } if inst.read(Reg::S(*code)) == Some(v) => {
            let off = *offset + const_operand(soffset).unwrap_or(0) as i32;
            u32::try_from(off).ok().map(|o| (ndw, o))
        }
        _ => None,
    }
}

/// Returns `true` if `inst` is a `global_load_dword*` or `s_load_dword*` using `v` as its 64-bit hi address.
fn flat_or_smem_load_hi_only(inst: &Inst, v: Value) -> bool {
    let n = inst.name();
    if !matches!(
        n,
        "global_load_dword"
            | "global_load_dwordx2"
            | "global_load_dwordx3"
            | "global_load_dwordx4"
            | "s_load_dword"
            | "s_load_dwordx2"
            | "s_load_dwordx4"
    ) {
        return false;
    }
    match &inst.fields {
        Fields::Flat {
            addr: Some(Operand::Vector { reg, dwords: 2 }),
            ..
        } => inst.read(Reg::V(*reg + 1)) == Some(v),
        Fields::Smem {
            base: Operand::Scalar { code, dwords: 2 },
            ..
        } => inst.read(Reg::S(*code + 1)) == Some(v),
        _ => false,
    }
}

fn image_addr_values(inst: &Inst) -> Option<Vec<Value>> {
    let Fields::Image { addr, .. } = &inst.fields else {
        return None;
    };
    let mut out = Vec::new();
    for op in addr {
        let Operand::Vector { reg, dwords } = *op else {
            return None;
        };
        for k in 0..dwords {
            out.push(inst.read(Reg::V(reg + k))?);
        }
    }
    Some(out)
}

fn image_data_values(inst: &Inst) -> Option<Vec<Value>> {
    let Fields::Image { data, .. } = &inst.fields else {
        return None;
    };
    let Operand::Vector { reg, dwords } = *data else {
        return None;
    };
    let mut out = Vec::with_capacity(usize::from(dwords));
    for k in 0..dwords {
        out.push(inst.write(Reg::V(reg + k))?);
    }
    Some(out)
}

fn const_operand(op: &Operand) -> Option<u32> {
    match *op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        Operand::Scalar { code, .. } if code == NULL => Some(0),
        _ => None,
    }
}

fn operand_val(inst: &Inst, op: &Operand) -> Option<Value> {
    match *op {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
        _ => None,
    }
}

fn first_src_operand(inst: &Inst) -> Option<&Operand> {
    match &inst.fields {
        Fields::Sop { src, .. } => src.first(),
        Fields::Valu(v) => v.src.first(),
        _ => None,
    }
}

fn two_src_operands(inst: &Inst) -> Option<(&Operand, &Operand)> {
    match &inst.fields {
        Fields::Sop { src, .. } if src.len() >= 2 => Some((&src[0], &src[1])),
        Fields::Valu(v) if v.src.len() >= 2 => Some((&v.src[0], &v.src[1])),
        _ => None,
    }
}

fn single_reg_src(inst: &Inst) -> Option<Value> {
    first_src_operand(inst).and_then(|op| operand_val(inst, op))
}

fn other_const_arg(inst: &Inst, known: Value) -> Option<u32> {
    let (a, b) = two_src_operands(inst)?;
    if operand_val(inst, a) == Some(known) {
        const_operand(b)
    } else if operand_val(inst, b) == Some(known) {
        const_operand(a)
    } else {
        None
    }
}

fn other_val_arg(inst: &Inst, known: Value) -> Option<Value> {
    let (a, b) = two_src_operands(inst)?;
    if operand_val(inst, a) == Some(known) {
        operand_val(inst, b)
    } else if operand_val(inst, b) == Some(known) {
        operand_val(inst, a)
    } else {
        None
    }
}

fn other_const_arg_any(inst: &Inst) -> Option<u32> {
    let (a, b) = two_src_operands(inst)?;
    const_operand(a).or_else(|| const_operand(b))
}

fn other_val_from_const_op(inst: &Inst, k: u32) -> Option<Value> {
    let (a, b) = two_src_operands(inst)?;
    if const_operand(a) == Some(k) {
        operand_val(inst, b)
    } else if const_operand(b) == Some(k) {
        operand_val(inst, a)
    } else {
        None
    }
}

fn has_const_operand(inst: &Inst, k: u32) -> bool {
    let Some((a, b)) = two_src_operands(inst) else {
        return false;
    };
    const_operand(a) == Some(k) || const_operand(b) == Some(k)
}

/// Decodes an unsigned `inst_word0 > K` threshold into an 8-bit `InstanceMask`:
/// `K == (1 << bit) - 1` for `bit in 24..=31` tests `(inst_word0 >> bit) != 0`.
fn decode_gt_threshold_mask(k: u32) -> Option<u32> {
    for bit in 24..=31u32 {
        if k == (1u32 << bit) - 1 {
            let mask = 0xFFu32 & !((1u32 << (bit - 24)) - 1);
            return Some(mask);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::shader::Shader;

    #[test]
    fn gt_threshold_masks_match_xdxr_constants() {
        assert_eq!(decode_gt_threshold_mask(0x00FF_FFFF), Some(0xFF));
        assert_eq!(decode_gt_threshold_mask(0x3FFF_FFFF), Some(0xC0));
        assert_eq!(decode_gt_threshold_mask((-4i32 as u32).reverse_bits()), Some(0xC0));
        assert_eq!(decode_gt_threshold_mask(0x7FFF_FFFF), Some(0x80));
        assert_eq!(decode_gt_threshold_mask(0x0FFF_FFFF), Some(0xF0));
        assert_eq!(decode_gt_threshold_mask(0x0000_FFFF), None);
    }

    #[test]
    fn emit_opaque_ray_loop_produces_valid_rayquery_dxil() {
        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 2,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh32SharedTlas,
            kind: RayLoopKind::OpaqueOnly,
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [
                RayInput::Val(Value(1)),
                RayInput::Val(Value(2)),
                RayInput::Val(Value(3)),
            ],
            t_min: None,
            dir: [
                RayInput::Val(Value(4)),
                RayInput::Val(Value(5)),
                RayInput::Val(Value(6)),
            ],
            t_max: RayInput::Val(Value(7)),
            instance_mask: RayInput::Const(0xC0),
            ray_flags: RayInput::Const(
                ray_flag::FORCE_OPAQUE
                    | ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
                    | ray_flag::CULL_BACK_FACING_TRIANGLES
                    | ray_flag::SKIP_PROCEDURAL_PRIMITIVES,
            ),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(7)),
                },
                RayOutputBinding {
                    value: Value(12),
                    output: RayOutput::Zero,
                    fallback: None,
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let rq = s.allocate_ray_query(
            entry,
            ray_flag::FORCE_OPAQUE
                | ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
                | ray_flag::CULL_BACK_FACING_TRIANGLES
                | ray_flag::SKIP_PROCEDURAL_PRIMITIVES,
        );
        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);
        fallbacks.insert(Value(7), ten_f);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: None,
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: ray_flag::FORCE_OPAQUE
                | ray_flag::ACCEPT_FIRST_HIT_AND_END_SEARCH
                | ray_flag::CULL_BACK_FACING_TRIANGLES
                | ray_flag::SKIP_PROCEDURAL_PRIMITIVES,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![],
            anyhit_entry_dxil: None,
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
        assert_ne!(emitted.tail, entry);
        assert_eq!(emitted.outs.len(), 3);
        s.m.terminate(exit, Term::Ret);

        let dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
    }

    #[test]
    fn emit_front_back_track_ray_loop_produces_valid_rayquery_dxil() {
        let flags = ray_flag::FORCE_NON_OPAQUE | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 2,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::FrontBackTrack {
                init_min_front_t: RayInput::Val(Value(7)),
            },
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [
                RayInput::Val(Value(1)),
                RayInput::Val(Value(2)),
                RayInput::Val(Value(3)),
            ],
            t_min: None,
            dir: [
                RayInput::Val(Value(4)),
                RayInput::Val(Value(5)),
                RayInput::Val(Value(6)),
            ],
            t_max: RayInput::Val(Value(7)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(flags),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(7)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::FrontBackMinFrontT,
                    fallback: Some(Value(7)),
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        assert!(is_front_back_two_pass_shape(&rl));
        assert!(
            !is_front_back_two_pass_eligible(&rl),
            "FrontBackTrack two-pass must be opt-in (off by default for TRIANGLE_CULL_DISABLE parity)"
        );
        // Default mask-split (`0x02`) clears FORCE_NON_OPAQUE on the shared handle so Pass A/B can use FORCE_OPAQUE;
        // disabling mask-split (`Some(0)`) preserves FORCE_NON_OPAQUE for the single-pass Proceed() fallback.
        assert_eq!(
            effective_const_ray_flags(&rl),
            Some(flags & !ray_flag::FORCE_NON_OPAQUE),
            "default mask-split FrontBackTrack clears FORCE_NON_OPAQUE"
        );
        let prev_split = super::super::lower::TEST_NATIVE_RT_FBT_SPLIT_MASK.replace(Some(0));
        let prev_mirror = super::super::lower::TEST_NATIVE_RT_MIRROR_TLAS.replace(Some(false));
        let nosplit_flags = effective_const_ray_flags(&rl);
        super::super::lower::TEST_NATIVE_RT_MIRROR_TLAS.set(prev_mirror);
        super::super::lower::TEST_NATIVE_RT_FBT_SPLIT_MASK.set(prev_split);
        assert_eq!(
            nosplit_flags,
            Some(flags),
            "non-split FrontBackTrack preserves FORCE_NON_OPAQUE"
        );
        let two_pass_flags = (flags & !ray_flag::FORCE_NON_OPAQUE) | ray_flag::FORCE_OPAQUE;
        let mirror_flags = flags & !ray_flag::FORCE_NON_OPAQUE;

        // Validate exact Proceed(), opt-in two-pass, and ARCH-13 2-pass Mirror-TLAS (constRayFlags=512).
        for (label, alloc_flags, mirror_tlas) in [
            ("default exact FrontBackTrack", flags, false),
            ("opt-in two-pass pure-opaque FrontBackTrack", two_pass_flags, false),
            ("ARCH-13 mirror-TLAS 2-pass FrontBackTrack (constFlags=512)", mirror_flags, true),
        ] {
            let mut s = Shader::new(false);
            s.threads = Some([8, 8, 1]);
            let entry = s.entry;
            let exit = s.m.new_block();
            let tlas = s.rtas_root_srv_handle(entry, 0);

            let rq = s.allocate_ray_query(entry, alloc_flags);
            let zero = s.m.i32(0);
            let one_f = s.m.i32(1.0f32.to_bits());
            let ten_f = s.m.i32(10.0f32.to_bits());
            let mut fallbacks = HashMap::new();
            fallbacks.insert(Value(7), ten_f);

            let inputs = ResolvedRayInputs {
                rq,
                two_pass_early_fields: None,
                tlas,
                active_ray: Some(s.m.bool(true)),
                origin: [zero, zero, zero],
                t_min: None,
                dir: [one_f, zero, zero],
                t_max: ten_f,
                instance_mask: None,
                ray_flags: None,
                group_const_flags: alloc_flags,
                fbt_split_mask: 0,
                fbt_mirror_tlas: mirror_tlas,
                init_min_front_t: Some(ten_f),
                anyhit_inits: vec![],
                anyhit_entry_dxil: None,
                fallbacks,
            };

            let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
            assert!(emitted.pending_anyhit.is_none());
            assert_eq!(emitted.outs.len(), 2);
            s.m.terminate(exit, Term::Ret);

            let mut dxbc = s.finish();
            assert!(dxbc.starts_with(b"DXBC"));
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut dxbc)
                    .unwrap_or_else(|e| panic!("{label} DXIL invalid: {e}"));
            }
        }
    }

    #[test]
    fn emit_anyhit_ray_loop_and_meta_loads_produce_valid_rayquery_dxil() {
        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let anyhit_body = s.m.new_block();
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let flags = ray_flag::CULL_BACK_FACING_TRIANGLES | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 3,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::AnyHit(AnyHitInfo {
                opacity_block: 1,
                entry_block: 2,
                tail_block: 2,
                body_blocks: vec![2],
                skipped_tail_insts: vec![],
                accept_mask: Value(20),
                accept_invert: false,
                accept_first_hit: true,
                inputs: vec![
                    (Value(15), CandidateInput::RayT),
                    (Value(16), CandidateInput::Bary(0)),
                    (Value(17), CandidateInput::FrontFace),
                    (Value(18), CandidateInput::StateVar(0)),
                ],
                const_lane_atoms: vec![],
                state_vars: vec![AnyHitStateVar {
                    init: RayInput::Const(0),
                    next: Value(19),
                    reg: Reg::V(0),
                }],
                cand_meta_loads: vec![],
            }),
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [
                RayInput::Val(Value(1)),
                RayInput::Val(Value(2)),
                RayInput::Val(Value(3)),
            ],
            t_min: None,
            dir: [
                RayInput::Val(Value(4)),
                RayInput::Val(Value(5)),
                RayInput::Val(Value(6)),
            ],
            t_max: RayInput::Val(Value(7)),
            instance_mask: RayInput::Const(0xDA),
            ray_flags: RayInput::Const(flags),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::AnyHitState(0),
                    fallback: Some(Value(0)),
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let rq = s.allocate_ray_query(entry, flags);
        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: None,
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: flags,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![zero],
            anyhit_entry_dxil: Some(anyhit_body),
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
        let pending = emitted.pending_anyhit.expect("pending anyhit");
        // SKIP_PROCEDURAL_PRIMITIVES elides `cand_check_blk` and redundant `latch_phis`.
        assert!(pending.cand_check_blk.is_none());
        assert!(pending.latch_phis.is_empty());
        // Lazy `CandidateInput::{RayT, Bary, FrontFace}` are NOT eagerly emitted in `outs`.
        assert!(!emitted.outs.iter().any(|&(v, _)| matches!(v, Value(15) | Value(16) | Value(17))));
        assert!(emitted.outs.iter().any(|&(v, _)| v == Value(18)));
        // Exercise lazy candidate query + candidate metadata emission inside the any-hit block.
        let _cand_t = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::RayT);
        let _cand_bary = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::Bary(0));
        let _cand_ff = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::FrontFace);
        let (ah_b, cand_inst) =
            emit_ray_meta_load(&mut s, anyhit_body, rq, true, RayMetaKind::InstanceId, 1, None);
        let (ah_b, cand_geom) = emit_ray_meta_load(
            &mut s,
            ah_b,
            rq,
            true,
            RayMetaKind::GeometryIndexShl8,
            1,
            None,
        );
        let accept = s.m.cmp(ah_b, pred::NE, cand_inst[0], zero);
        s.m.terminate(
            ah_b,
            Term::CondBr(accept, pending.do_commit_blk, pending.latch_blk),
        );
        s.m.set_phi(
            pending.state_phis[0],
            vec![
                (pending.init_vals[0], pending.start_blk),
                (cand_geom[0], pending.latch_blk),
            ],
        );
        // Exercise committed metadata emission post-loop.
        let (ex_b, _xform) = emit_ray_meta_load(
            &mut s,
            exit,
            rq,
            false,
            RayMetaKind::ObjectToWorld(0),
            4,
            None,
        );
        let (ex_b, _prim) =
            emit_ray_meta_load(&mut s, ex_b, rq, false, RayMetaKind::PrimitiveIndex, 1, None);
        s.m.terminate(ex_b, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc).expect("AnyHit + meta loads DXIL valid");
        }
    }

    #[test]
    fn anyhit_without_skip_procedural_keeps_cand_check_blk_and_validates() {
        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let anyhit_body = s.m.new_block();
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let flags = ray_flag::CULL_BACK_FACING_TRIANGLES;
        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 3,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::AnyHit(AnyHitInfo {
                opacity_block: 1,
                entry_block: 2,
                tail_block: 2,
                body_blocks: vec![2],
                skipped_tail_insts: vec![],
                accept_mask: Value(20),
                accept_invert: false,
                accept_first_hit: false,
                inputs: vec![(Value(18), CandidateInput::StateVar(0))],
                const_lane_atoms: vec![],
                state_vars: vec![AnyHitStateVar {
                    init: RayInput::Const(0),
                    next: Value(19),
                    reg: Reg::V(0),
                }],
                cand_meta_loads: vec![],
            }),
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Val(Value(1)); 3],
            t_min: None,
            dir: [RayInput::Val(Value(2)); 3],
            t_max: RayInput::Val(Value(3)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(flags),
            outputs: vec![RayOutputBinding {
                value: Value(11),
                output: RayOutput::AnyHitState(0),
                fallback: Some(Value(0)),
            }],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let rq = s.allocate_ray_query(entry, flags);
        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: None,
            tlas,
            active_ray: None,
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: flags,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![zero],
            anyhit_entry_dxil: Some(anyhit_body),
            fallbacks: HashMap::new(),
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
        let pending = emitted.pending_anyhit.expect("pending anyhit");
        let cb = pending
            .cand_check_blk
            .expect("cand_check_blk kept when SKIP_PROCEDURAL_PRIMITIVES is unset");
        assert_eq!(pending.latch_phis.len(), 1);
        let cand_t = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::RayT);
        let accept = s.m.cmp(anyhit_body, pred::NE, cand_t, zero);
        s.m.terminate(
            anyhit_body,
            Term::CondBr(accept, pending.do_commit_blk, pending.latch_blk),
        );
        s.m.set_phi(
            pending.latch_phis[0],
            vec![
                (pending.state_phis[0], cb),
                (cand_t, anyhit_body),
                (cand_t, pending.do_commit_blk),
            ],
        );
        s.m.set_phi(
            pending.state_phis[0],
            vec![
                (pending.init_vals[0], pending.start_blk),
                (pending.latch_phis[0], pending.latch_blk),
            ],
        );
        s.m.terminate(exit, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc)
                .expect("AnyHit without SKIP_PROCEDURAL_PRIMITIVES DXIL valid");
        }
    }

    #[test]
    fn two_pass_eligibility_and_dxil_valid() {
        let make_anyhit = |flags: RayInput, accept_first_hit: bool, with_state: bool| RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 3,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::AnyHit(AnyHitInfo {
                opacity_block: 1,
                entry_block: 2,
                tail_block: 2,
                body_blocks: vec![2],
                skipped_tail_insts: vec![],
                accept_mask: Value(20),
                accept_invert: false,
                accept_first_hit,
                inputs: vec![(Value(15), CandidateInput::RayT)],
                const_lane_atoms: vec![],
                state_vars: if with_state {
                    vec![AnyHitStateVar {
                        init: RayInput::Const(0),
                        next: Value(19),
                        reg: Reg::V(0),
                    }]
                } else {
                    vec![]
                },
                cand_meta_loads: vec![],
            }),
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Val(Value(1)); 3],
            t_min: None,
            dir: [RayInput::Val(Value(2)); 3],
            t_max: RayInput::Val(Value(3)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: flags,
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(3)),
                },
                RayOutputBinding {
                    value: Value(12),
                    output: RayOutput::CommittedBary(0),
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(13),
                    output: RayOutput::CommittedFrontFace,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(14),
                    output: RayOutput::CommittedInstanceIndex,
                    fallback: None,
                },
                RayOutputBinding {
                    value: Value(16),
                    output: RayOutput::CommittedObjectToWorld(0, 0),
                    fallback: None,
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let eligible = make_anyhit(
            RayInput::Const(
                ray_flag::CULL_BACK_FACING_TRIANGLES | ray_flag::SKIP_PROCEDURAL_PRIMITIVES,
            ),
            false,
            false,
        );
        assert!(is_two_pass_eligible(&eligible));
        assert!(is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::NONE),
            false,
            false
        )));
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::FORCE_NON_OPAQUE),
            false,
            false
        )));
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::FORCE_OPAQUE),
            false,
            false
        )));
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::CULL_OPAQUE),
            false,
            false
        )));
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::CULL_NON_OPAQUE),
            false,
            false
        )));
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::NONE),
            true,
            false
        )));
        // With `VICTORY_NATIVE_RT_TWOPASS_STATE` default ON, closest-hit `AnyHit` loops with
        // `state_vars` are eligible for two-pass `tMax` pre-shrink; setting it OFF disables it.
        let prev_tp_state = super::super::lower::TEST_NATIVE_RT_TWOPASS_STATE.replace(Some(true));
        let hs_on = super::super::host_settings();
        let fp_on = super::super::translation_fingerprint();
        assert!(is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::NONE),
            false,
            true
        )));
        super::super::lower::TEST_NATIVE_RT_TWOPASS_STATE.set(Some(false));
        let hs_off = super::super::host_settings();
        let fp_off = super::super::translation_fingerprint();
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Const(ray_flag::NONE),
            false,
            true
        )));
        assert_ne!(hs_on, hs_off, "VICTORY_NATIVE_RT_TWOPASS_STATE must update host_settings()");
        assert_ne!(fp_on, fp_off, "VICTORY_NATIVE_RT_TWOPASS_STATE must update translation_fingerprint()");
        super::super::lower::TEST_NATIVE_RT_TWOPASS_STATE.set(prev_tp_state);
        assert!(!is_two_pass_eligible(&make_anyhit(
            RayInput::Val(Value(99)),
            false,
            false
        )));

        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let anyhit_body = s.m.new_block();
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let base_flags =
            ray_flag::CULL_BACK_FACING_TRIANGLES | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rq = s.allocate_ray_query(entry, base_flags);
        let early_fields = collect_early_q1_fields(&eligible, &[]);

        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);
        fallbacks.insert(Value(3), ten_f);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: Some(early_fields),
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: base_flags,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![],
            anyhit_entry_dxil: Some(anyhit_body),
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &eligible, &inputs);
        let pending = emitted.pending_anyhit.expect("pending anyhit");
        assert!(pending.cand_check_blk.is_none());
        let cand_t = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::RayT);
        let accept = s.m.cmp(anyhit_body, pred::NE, cand_t, zero);
        s.m.terminate(
            anyhit_body,
            Term::CondBr(accept, pending.do_commit_blk, pending.latch_blk),
        );
        s.m.terminate(exit, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc)
                .expect("two-pass CULL_NON_OPAQUE + CULL_OPAQUE DXIL valid");
        }
    }

    #[test]
    fn two_pass_anyhit_with_state_vars_and_early_extract_validates() {
        let base_flags =
            ray_flag::CULL_BACK_FACING_TRIANGLES | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 3,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::AnyHit(AnyHitInfo {
                opacity_block: 1,
                entry_block: 2,
                tail_block: 2,
                body_blocks: vec![2],
                skipped_tail_insts: vec![],
                accept_mask: Value(20),
                accept_invert: false,
                accept_first_hit: false,
                inputs: vec![
                    (Value(15), CandidateInput::RayT),
                    (Value(18), CandidateInput::StateVar(0)),
                ],
                const_lane_atoms: vec![],
                state_vars: vec![AnyHitStateVar {
                    init: RayInput::Const(0),
                    next: Value(19),
                    reg: Reg::V(0),
                }],
                cand_meta_loads: vec![],
            }),
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Val(Value(1)); 3],
            t_min: None,
            dir: [RayInput::Val(Value(2)); 3],
            t_max: RayInput::Val(Value(3)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(base_flags),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(3)),
                },
                RayOutputBinding {
                    value: Value(12),
                    output: RayOutput::AnyHitState(0),
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(13),
                    output: RayOutput::CommittedGeometryIndex,
                    fallback: None,
                },
                // Duplicate output binding should be deduplicated inside `emit_trace_body`.
                RayOutputBinding {
                    value: Value(14),
                    output: RayOutput::CommittedGeometryIndex,
                    fallback: None,
                },
            ],
            committed_meta_loads: vec![
                ((3, 0), RayMetaKind::InstanceIndex),
                ((3, 1), RayMetaKind::GeometryIndexShl8),
                ((3, 2), RayMetaKind::PrimitiveIndex),
            ],
            direct_normals: vec![],
        };

        let fields = collect_early_q1_fields(
            &rl,
            &[
                (RayMetaKind::InstanceIndex, 1),
                (RayMetaKind::GeometryIndexShl8, 1),
                (RayMetaKind::PrimitiveIndex, 1),
            ],
        );
        // GeometryIndex is shared between `rl.outputs` and `GeometryIndexShl8`.
        assert_eq!(fields.len(), 3, "expected 3 deduplicated early q1 fields: {fields:?}");

        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let anyhit_body = s.m.new_block();
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        // Single AllocateRayQuery handle reused across Pass 1 and Pass 2!
        let rq = s.allocate_ray_query(entry, base_flags);
        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);
        fallbacks.insert(Value(3), ten_f);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: Some(fields),
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: base_flags,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![zero],
            anyhit_entry_dxil: Some(anyhit_body),
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
        let tp = emitted.two_pass_post.expect("early-extract two_pass_post");
        let q1_fields = tp.q1_fields;
        assert!(q1_fields.contains_key(&EarlyQ1Field::GeometryIndex));
        assert!(q1_fields.contains_key(&EarlyQ1Field::InstanceIndex));
        assert!(q1_fields.contains_key(&EarlyQ1Field::PrimitiveIndex));
        // Duplicate CommittedGeometryIndex output bindings produce the exact same SSA value.
        let g13 = emitted.outs.iter().find(|&&(v, _)| v == Value(13)).unwrap().1;
        let g14 = emitted.outs.iter().find(|&&(v, _)| v == Value(14)).unwrap().1;
        assert_eq!(g13, g14, "duplicate CommittedGeometryIndex outputs must share one query");

        let pending = emitted.pending_anyhit.expect("pending anyhit");
        let cand_t = emit_cand_input(&mut s, anyhit_body, rq, CandidateInput::RayT);
        let accept = s.m.cmp(anyhit_body, pred::NE, cand_t, zero);
        s.m.terminate(
            anyhit_body,
            Term::CondBr(accept, pending.do_commit_blk, pending.latch_blk),
        );
        s.m.set_phi(
            pending.state_phis[0],
            vec![
                (pending.init_vals[0], pending.start_blk),
                (cand_t, pending.latch_blk),
            ],
        );
        let (ex_b, _geom_shl8) = resolve_early_q1_meta(
            &mut s,
            exit,
            &q1_fields,
            RayMetaKind::GeometryIndexShl8,
            1,
            None,
        )
        .expect("resolve early GeometryIndexShl8");
        s.m.terminate(ex_b, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc)
                .expect("single-handle early-extract two-pass AnyHit + state_vars DXIL valid");
        }
    }

    #[test]
    fn fno_speculative_fast_path_eligibility_and_dxil_valid() {
        let flags = ray_flag::FORCE_NON_OPAQUE | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let make_fno = |accept_first_hit: bool, has_state_vars: bool| RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 3,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::AnyHit(AnyHitInfo {
                opacity_block: 1,
                entry_block: 2,
                tail_block: 2,
                body_blocks: vec![2],
                skipped_tail_insts: vec![],
                accept_mask: Value(20),
                accept_invert: false,
                accept_first_hit,
                inputs: vec![
                    (Value(15), CandidateInput::RayT),
                    (Value(17), CandidateInput::FrontFace),
                ],
                const_lane_atoms: vec![],
                state_vars: if has_state_vars {
                    vec![AnyHitStateVar {
                        init: RayInput::Const(0),
                        next: Value(19),
                        reg: Reg::V(0),
                    }]
                } else {
                    vec![]
                },
                cand_meta_loads: vec![],
            }),
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Val(Value(1)); 3],
            t_min: None,
            dir: [RayInput::Val(Value(2)); 3],
            t_max: RayInput::Val(Value(3)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(flags),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(3)),
                },
                RayOutputBinding {
                    value: Value(12),
                    output: RayOutput::CommittedGeometryIndex,
                    fallback: None,
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let rl_closest = make_fno(false, false);
        let rl_afh = make_fno(true, false);
        let rl_state = make_fno(false, true);
        let rl_afh_state = make_fno(true, true);
        let mut rl_dyn_state = make_fno(false, true);
        rl_dyn_state.ray_flags = RayInput::Val(Value(99));

        let prev_fno = super::super::lower::TEST_NATIVE_RT_FNO_MODE.replace(Some(2));
        let hs_m2 = super::super::host_settings();
        let fp_m2 = super::super::translation_fingerprint();
        assert!(is_fno_speculative_eligible(&rl_closest));
        assert!(is_fno_speculative_eligible(&rl_afh));
        assert!(is_fno_speculative_eligible(&rl_state));
        assert!(!is_fno_speculative_eligible(&rl_afh_state));
        assert!(is_fno_speculative_eligible(&rl_dyn_state));
        assert_eq!(
            effective_const_ray_flags(&rl_closest),
            Some(ray_flag::SKIP_PROCEDURAL_PRIMITIVES),
            "mode 2 strips FORCE_NON_OPAQUE from constRayFlags for Pass 1 FORCE_OPAQUE reuse"
        );
        assert_eq!(effective_const_ray_flags(&rl_dyn_state), None);

        super::super::lower::TEST_NATIVE_RT_FNO_MODE.set(Some(1));
        let hs_m1 = super::super::host_settings();
        let fp_m1 = super::super::translation_fingerprint();
        assert!(!is_fno_speculative_eligible(&rl_closest));
        assert_eq!(effective_const_ray_flags(&rl_closest), Some(flags));

        super::super::lower::TEST_NATIVE_RT_FNO_MODE.set(Some(0));
        let hs_m0 = super::super::host_settings();
        assert_ne!(hs_m2, hs_m1, "VICTORY_NATIVE_RT_FNO_MODE=2 vs 1 must update host_settings()");
        assert_ne!(hs_m1, hs_m0, "VICTORY_NATIVE_RT_FNO_MODE=1 vs 0 must update host_settings()");
        assert_ne!(fp_m2, fp_m1, "VICTORY_NATIVE_RT_FNO_MODE must update translation_fingerprint()");
        super::super::lower::TEST_NATIVE_RT_FNO_MODE.set(prev_fno);

        // Emit un-fused speculative fast-path DXIL (`ARCH-12`: Pass 1 outside loop, Pass 2 clean
        // `while (Proceed())` loop with guarded `nextafter_down(t1)`) and validate with `dxil.dll`.
        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let p1_anyhit_body = s.m.new_block();
        let p2_anyhit_body = s.m.new_block();
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let alloc_flags = ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rq = s.allocate_ray_query(entry, alloc_flags);
        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);
        fallbacks.insert(Value(3), ten_f);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: None,
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: alloc_flags,
            fbt_split_mask: 0,
            fbt_mirror_tlas: false,
            init_min_front_t: None,
            anyhit_inits: vec![],
            anyhit_entry_dxil: Some(p2_anyhit_body),
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl_closest, &inputs);
        let pending = emitted.pending_anyhit.expect("pending anyhit");
        let spec = pending.spec.expect("speculative FNO metadata");

        // Pass 1 out-of-loop `Committed*` AnyHit check (zero `spec_phi`!).
        s.m.terminate(spec.p1_bridge_blk, Term::Br(p1_anyhit_body));
        let p1_t = spec
            .p1_cand_outs
            .iter()
            .find(|&&(v, _)| v == Value(15))
            .unwrap()
            .1;
        let (p1_b, g_comm) = emit_ray_meta_load(
            &mut s,
            p1_anyhit_body,
            rq,
            false,
            RayMetaKind::GeometryIndexShl8,
            1,
            None,
        );
        let p1_comb = s.m.bin(p1_b, Bin::Or, g_comm[0], p1_t);
        let p1_accept = s.m.cmp(p1_b, pred::NE, p1_comb, zero);
        finish_fno_spec_pass1(&mut s, &spec, p1_b, p1_accept);

        // Pass 2 clean `Candidate*` AnyHit check inside `while (Proceed())` (zero `spec_phi`!).
        let p2_t = emit_cand_input(&mut s, p2_anyhit_body, rq, CandidateInput::RayT);
        let (p2_b, g_cand) = emit_ray_meta_load(
            &mut s,
            p2_anyhit_body,
            rq,
            true,
            RayMetaKind::GeometryIndexShl8,
            1,
            None,
        );
        let p2_comb = s.m.bin(p2_b, Bin::Or, g_cand[0], p2_t);
        let p2_accept = s.m.cmp(p2_b, pred::NE, p2_comb, zero);
        s.m.terminate(
            p2_b,
            Term::CondBr(p2_accept, pending.do_commit_blk, pending.latch_blk),
        );
        if let Some((pt_phi, pt_latch, ct)) = spec.p2_t_track {
            s.m.set_phi(pt_latch, vec![(pt_phi, p2_b), (ct, pending.do_commit_blk)]);
        }
        s.m.terminate(exit, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc)
                .expect("un-fused speculative FORCE_NON_OPAQUE AnyHit DXIL must pass dxil.dll validation");
        }
    }

    #[test]
    fn front_back_mask_split_eligibility_and_dxil_valid() {
        let flags = ray_flag::FORCE_NON_OPAQUE | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rl = RayLoop {
            rq_group: 0,
            entry_block: 1,
            exit_block: 2,
            exit_pred: 1,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind: RayLoopKind::FrontBackTrack {
                init_min_front_t: RayInput::Val(Value(7)),
            },
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Val(Value(1)); 3],
            t_min: None,
            dir: [RayInput::Val(Value(2)); 3],
            t_max: RayInput::Val(Value(7)),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(flags),
            outputs: vec![
                RayOutputBinding {
                    value: Value(10),
                    output: RayOutput::CommittedStatusTriNode,
                    fallback: Some(Value(0)),
                },
                RayOutputBinding {
                    value: Value(11),
                    output: RayOutput::CommittedRayT,
                    fallback: Some(Value(7)),
                },
                RayOutputBinding {
                    value: Value(12),
                    output: RayOutput::FrontBackMinFrontT,
                    fallback: Some(Value(7)),
                },
                RayOutputBinding {
                    value: Value(13),
                    output: RayOutput::CommittedGeometryIndex,
                    fallback: None,
                },
            ],
            committed_meta_loads: vec![],
            direct_normals: vec![],
        };

        let prev = super::super::lower::TEST_NATIVE_RT_FBT_SPLIT_MASK.replace(Some(0));
        let prev_mirror = super::super::lower::TEST_NATIVE_RT_MIRROR_TLAS.replace(Some(false));
        let hs_off = super::super::host_settings();
        let fp_off = super::super::translation_fingerprint();
        assert!(!is_front_back_mask_split_eligible(&rl));
        assert_eq!(effective_const_ray_flags(&rl), Some(flags));

        super::super::lower::TEST_NATIVE_RT_FBT_SPLIT_MASK.set(Some(0x80));
        let hs_on = super::super::host_settings();
        let fp_on = super::super::translation_fingerprint();
        assert!(is_front_back_mask_split_eligible(&rl));
        assert!(!is_front_back_two_pass_eligible(&rl));
        assert_eq!(
            effective_const_ray_flags(&rl),
            Some(ray_flag::SKIP_PROCEDURAL_PRIMITIVES),
            "mask-split FrontBackTrack strips FORCE_NON_OPAQUE so single rq runs FORCE_OPAQUE + FORCE_NON_OPAQUE"
        );
        assert_ne!(hs_off, hs_on, "VICTORY_NATIVE_RT_FBT_SPLIT_MASK must update host_settings()");
        assert_ne!(
            fp_off, fp_on,
            "VICTORY_NATIVE_RT_FBT_SPLIT_MASK must update translation_fingerprint()"
        );
        super::super::lower::TEST_NATIVE_RT_MIRROR_TLAS.set(prev_mirror);
        super::super::lower::TEST_NATIVE_RT_FBT_SPLIT_MASK.set(prev);

        // Validate single-handle early-extract fallback 3-pass mask-split DXIL with dxil.dll.
        let mut s = Shader::new(false);
        s.threads = Some([8, 8, 1]);
        let entry = s.entry;
        let exit = s.m.new_block();
        let tlas = s.rtas_root_srv_handle(entry, 0);

        let base_fl = ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let rq = s.allocate_ray_query(entry, base_fl);
        let early = Some(collect_early_q1_fields(&rl, &[]));

        let zero = s.m.i32(0);
        let one_f = s.m.i32(1.0f32.to_bits());
        let ten_f = s.m.i32(10.0f32.to_bits());
        let mut fallbacks = HashMap::new();
        fallbacks.insert(Value(0), zero);
        fallbacks.insert(Value(7), ten_f);

        let inputs = ResolvedRayInputs {
            rq,
            two_pass_early_fields: early,
            tlas,
            active_ray: Some(s.m.bool(true)),
            origin: [zero, zero, zero],
            t_min: None,
            dir: [one_f, zero, zero],
            t_max: ten_f,
            instance_mask: None,
            ray_flags: None,
            group_const_flags: base_fl,
            fbt_split_mask: 0x80,
            fbt_mirror_tlas: false,
            init_min_front_t: Some(ten_f),
            anyhit_inits: vec![],
            anyhit_entry_dxil: None,
            fallbacks,
        };

        let emitted = emit_ray_loop(&mut s, entry, exit, &rl, &inputs);
        assert!(emitted.pending_anyhit.is_none());
        assert!(emitted.two_pass_post.is_some());
        assert_eq!(emitted.outs.len(), 4);
        s.m.terminate(exit, Term::Ret);

        let mut dxbc = s.finish();
        assert!(dxbc.starts_with(b"DXBC"));
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxbc)
                .unwrap_or_else(|e| panic!("mask-split FrontBackTrack DXIL invalid: {e}"));
        }
    }

    #[test]
    fn coalesce_ray_query_groups_ignores_pre_loop_bypass_and_separates_two_pass() {
        use crate::ir::Block;
        // CFG:
        //   0 (entry) -> 1 (L1) -> 2 (branch)
        //   2 -> 3 (L2) -> 4 (join, reads L2 committed meta) -> 5 (end)
        //   2 -> 4 (pre-loop bypass around L2!)
        let blocks = vec![
            Block {
                offset: 0,
                preds: vec![],
                phis: vec![],
                insts: vec![],
                exit: Exit::Jump(1),
            },
            Block {
                offset: 4,
                preds: vec![0],
                phis: vec![],
                insts: vec![],
                exit: Exit::Jump(2),
            },
            Block {
                offset: 8,
                preds: vec![1],
                phis: vec![],
                insts: vec![],
                exit: Exit::Branch {
                    taken: 3,
                    not_taken: 4,
                },
            },
            Block {
                offset: 12,
                preds: vec![2],
                phis: vec![],
                insts: vec![],
                exit: Exit::Jump(4),
            },
            Block {
                offset: 16,
                preds: vec![2, 3],
                phis: vec![],
                insts: vec![],
                exit: Exit::Jump(5),
            },
            Block {
                offset: 20,
                preds: vec![4],
                phis: vec![],
                insts: vec![],
                exit: Exit::End,
            },
        ];
        let prog = Program {
            wave64: false,
            vgprs: 32,
            blocks,
            inputs: vec![],
        };
        let st = crate::structure::structure(&prog).unwrap();
        let res = crate::resource::track(&prog, &[], crate::native::HwStage::Cs, Some(&st)).unwrap();
        let scalar = HashMap::new();
        let analyzer = Analyzer::new(&prog, &st, &res, &scalar);

        let make_loop = |rq_group: usize, entry_block: usize, exit_block: usize, kind: RayLoopKind, meta: Vec<(Site, RayMetaKind)>| RayLoop {
            rq_group,
            entry_block,
            exit_block,
            exit_pred: entry_block,
            skipped_blocks: vec![],
            template: RayTemplate::Bvh64,
            kind,
            tlas_load: (0, 0),
            tlas_root_load: None,
            active_ray_mask: None,
            origin: [RayInput::Const(0); 3],
            t_min: None,
            dir: [RayInput::Const(0); 3],
            t_max: RayInput::Const(0),
            instance_mask: RayInput::Const(0xFF),
            ray_flags: RayInput::Const(ray_flag::SKIP_PROCEDURAL_PRIMITIVES),
            outputs: vec![],
            committed_meta_loads: meta,
            direct_normals: vec![],
        };

        // Two OpaqueOnly loops separated by a pre-loop bypass around L2:
        // L2's committed meta load at block 4 must NOT leak backward past the bypass into L1!
        let mut loops_bypass = vec![
            make_loop(0, 1, 2, RayLoopKind::OpaqueOnly, vec![]),
            make_loop(1, 3, 4, RayLoopKind::OpaqueOnly, vec![((4, 0), RayMetaKind::PrimitiveIndex)]),
        ];
        analyzer.coalesce_ray_query_groups(&mut loops_bypass);
        assert_eq!(
            loops_bypass[0].rq_group, loops_bypass[1].rq_group,
            "pre-loop bypass must not create spurious interference between L1 and L2"
        );

        // ARCH-14: With single-handle early-extract universal (`TwoPassQ1::Live` removed),
        // disjoint-lifetime groups sharing matching `const_flags` (e.g. `512` across `OpaqueOnly`,
        // `TwoPass` `AnyHit`, `fno_spec`, and Mirror-TLAS `FrontBackTrack`) coalesce onto 1 handle!
        let ah_two_pass = RayLoopKind::AnyHit(AnyHitInfo {
            opacity_block: 3,
            entry_block: 3,
            tail_block: 3,
            body_blocks: vec![],
            skipped_tail_insts: vec![],
            accept_mask: Value(1),
            accept_invert: false,
            accept_first_hit: false,
            inputs: vec![],
            const_lane_atoms: vec![],
            state_vars: vec![],
            cand_meta_loads: vec![],
        });
        let mut loops_mixed = vec![
            make_loop(0, 1, 2, RayLoopKind::OpaqueOnly, vec![]),
            make_loop(1, 3, 4, ah_two_pass, vec![((4, 0), RayMetaKind::PrimitiveIndex)]),
        ];
        analyzer.coalesce_ray_query_groups(&mut loops_mixed);
        assert_eq!(
            loops_mixed[0].rq_group, loops_mixed[1].rq_group,
            "disjoint matching-const_flags two-pass and non-two-pass groups must coalesce onto 1 handle"
        );

        // Groups with DIFFERENT `const_flags` still stay separate.
        let mut loops_diff_flags = vec![
            make_loop(0, 1, 2, RayLoopKind::OpaqueOnly, vec![]),
            make_loop(1, 3, 4, RayLoopKind::OpaqueOnly, vec![]),
        ];
        loops_diff_flags[1].ray_flags = RayInput::Const(ray_flag::FORCE_OPAQUE | ray_flag::SKIP_PROCEDURAL_PRIMITIVES);
        analyzer.coalesce_ray_query_groups(&mut loops_diff_flags);
        assert_ne!(
            loops_diff_flags[0].rq_group, loops_diff_flags[1].rq_group,
            "groups with differing const_flags must not coalesce"
        );
    }

    #[test]
    fn forward_phi_closure_excludes_anyhit_set_from_state_vars() {
        use crate::ir::{Block, Phi};

        // Decode real GFX10 instructions for `v_mov_b32` with literal/reg operands.
        let decode_one = |words: &[u32]| {
            let mut bytes = Vec::with_capacity((words.len() + 1) * 4);
            for &w in words {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            bytes.extend_from_slice(&0xBF81_0000u32.to_le_bytes()); // s_endpgm
            let list = crate::gfx10::decode(&bytes).expect("valid GFX10 inst");
            assert_eq!(list.len(), 2);
            let m = list.into_iter().next().unwrap();
            let fields = m.fields(false);
            (m, fields)
        };
        let make_inst = |words: &[u32], reads: Vec<(Reg, Value)>, writes: Vec<(Reg, Value)>| {
            let (machine, fields) = decode_one(words);
            Inst {
                machine,
                fields,
                reads,
                writes,
                wait: 0,
            }
        };

        // CFG:
        //   0 (pre-loop): v10 = 10000.0f (min_t init), v11 = 0 (dead local temp init)
        //   1 (loop entry): phi_min_t (v10) = phi(v10_init, v10_commit),
        //                   phi_tmp   (v11) = phi(v11_init, v11_commit)
        //   2 (anyhit_set): v_mov_b32 v10, v11 (inside anyhit_set, copies dead temp into min_t),
        //                   v_mov_b32 v11, 0   (updates dead local temp)
        //   3 (commit_block, in nodes \ anyhit_set): merges (1, 2) -> back-edge to 1, exit to 4
        //   4 (exit)
        // Without excluding `anyhit_set` in `forward_phi_closure`, tracing forward from `phi_tmp`
        // enters block 2, follows `v_mov_b32 v10, v11` to `v10_next`, exits to block 3's `v10_commit`
        // PHI and block 1's `phi_min_t`, polluting `phi_tmp`'s `scaff_vals` with `phi_min_t`!
        let min_t_bits = 10_000.0f32.to_bits();
        let v10_init = Value(0);
        let v11_init = Value(1);
        let phi_min_t = Value(2);
        let phi_tmp = Value(3);
        let v10_next = Value(4);
        let v11_next = Value(5);
        let v10_commit = Value(6);
        let v11_commit = Value(7);

        // VOP1 `v_mov_b32 vdst, src0`: 0x7E00_0200 | (vdst << 17) | (1 << 9) | src0
        // src0 = 0xFF (literal) for v10 = 10000.0f, src0 = 0x80 (inline 0) for v11 = 0,
        // src0 = 0x10B (v11) for `v_mov_b32 v10, v11`.
        let inst_init_min_t = make_inst(
            &[0x7E00_02FF | (10 << 17), min_t_bits],
            vec![],
            vec![(Reg::V(10), v10_init)],
        );
        let inst_init_tmp = make_inst(
            &[0x7E00_0280 | (11 << 17)],
            vec![],
            vec![(Reg::V(11), v11_init)],
        );
        let inst_ah_copy_to_min_t = make_inst(
            &[0x7E00_030B | (10 << 17)],
            vec![(Reg::V(11), phi_tmp)],
            vec![(Reg::V(10), v10_next)],
        );
        let inst_ah_write_tmp = make_inst(
            &[0x7E00_0280 | (11 << 17)],
            vec![],
            vec![(Reg::V(11), v11_next)],
        );

        let blocks = vec![
            Block {
                offset: 0,
                preds: vec![],
                phis: vec![],
                insts: vec![inst_init_min_t, inst_init_tmp],
                exit: Exit::Jump(1),
            },
            Block {
                offset: 12,
                preds: vec![0, 3],
                phis: vec![
                    Phi {
                        reg: Reg::V(10),
                        value: phi_min_t,
                        args: vec![v10_init, v10_commit],
                    },
                    Phi {
                        reg: Reg::V(11),
                        value: phi_tmp,
                        args: vec![v11_init, v11_commit],
                    },
                ],
                insts: vec![],
                exit: Exit::Branch {
                    taken: 2,
                    not_taken: 3,
                },
            },
            Block {
                offset: 16,
                preds: vec![1],
                phis: vec![],
                insts: vec![inst_ah_copy_to_min_t, inst_ah_write_tmp],
                exit: Exit::Jump(3),
            },
            Block {
                offset: 24,
                preds: vec![1, 2],
                phis: vec![
                    Phi {
                        reg: Reg::V(10),
                        value: v10_commit,
                        args: vec![phi_min_t, v10_next],
                    },
                    Phi {
                        reg: Reg::V(11),
                        value: v11_commit,
                        args: vec![phi_tmp, v11_next],
                    },
                ],
                insts: vec![],
                exit: Exit::Branch {
                    taken: 1,
                    not_taken: 4,
                },
            },
            Block {
                offset: 28,
                preds: vec![3],
                phis: vec![],
                insts: vec![],
                exit: Exit::End,
            },
        ];
        let prog = Program {
            wave64: false,
            vgprs: 32,
            blocks,
            inputs: vec![],
        };
        let st = crate::structure::structure(&prog).unwrap();
        let res = crate::resource::track(&prog, &[], crate::native::HwStage::Cs, Some(&st)).unwrap();
        let scalar = HashMap::new();
        let analyzer = Analyzer::new(&prog, &st, &res, &scalar);

        let nodes: HashSet<usize> = [1, 2, 3].into_iter().collect();
        let anyhit_set: HashSet<usize> = [2].into_iter().collect();

        // Confirm `forward_phi_closure` with `exclude = None` would have crossed block 2 into `phi_min_t`,
        // while `exclude = Some(&anyhit_set)` stays strictly within `nodes \ anyhit_set`.
        let leaked = analyzer.forward_phi_closure(phi_tmp, &nodes, None);
        assert!(leaked.contains(&phi_min_t));
        let clean = analyzer.forward_phi_closure(phi_tmp, &nodes, Some(&anyhit_set));
        assert!(!clean.contains(&phi_min_t));
        assert!(!clean.contains(&v10_commit));

        // Verify `match_anyhit_state_var` for the dead temp does NOT capture `phi_min_t` or `v10_commit`,
        // and `phi_min_t` keeps its real `init = 10000.0f` cleanly separated.
        let (tmp_init, tmp_next, tmp_scaff) = analyzer
            .match_anyhit_state_var(&prog.blocks[1].phis[1], &nodes, 1, &anyhit_set, 2)
            .expect("tmp matches state var shape");
        assert_eq!(tmp_init, RayInput::Val(v11_init));
        assert_eq!(analyzer.eval_const(v11_init, 0), Some(0));
        assert_eq!(tmp_next, v11_next);
        assert!(tmp_scaff.contains(&phi_tmp));
        assert!(tmp_scaff.contains(&v11_commit));
        assert!(
            !tmp_scaff.contains(&phi_min_t) && !tmp_scaff.contains(&v10_commit),
            "dead local temp inside anyhit_set must not pollute min_t scaffolding values: {tmp_scaff:?}"
        );

        let (min_init, min_next, min_scaff) = analyzer
            .match_anyhit_state_var(&prog.blocks[1].phis[0], &nodes, 1, &anyhit_set, 2)
            .expect("min_t matches state var shape");
        assert_eq!(min_init, RayInput::Val(v10_init));
        assert_eq!(analyzer.eval_const(v10_init, 0), Some(min_t_bits));
        assert_eq!(min_next, v10_next);
        assert!(min_scaff.contains(&phi_min_t));
        assert!(min_scaff.contains(&v10_commit));
        assert!(!min_scaff.contains(&phi_tmp));
    }

    #[test]
    fn direct_normal_substitution_matches_cross_and_keeps_paired_fallback_on_escape() {
        use crate::ir::Value;

        // Encode real RDNA2 `global_load_dwordx3`, `v_sub_f32`, `v_mul_f32`, `v_add_f32`, `v_rsq_f32`.
        let gload3 = |vdst: u32| -> [u32; 2] {
            [0xDC00_8000 | (0x00f << 18), (vdst << 24) | (0x7C << 16)]
        };
        let vop2 = |op: u32, vdst: u32, src0: u32, vsrc1: u32| -> u32 {
            (op << 25) | (vdst << 17) | (vsrc1 << 9) | (256 + src0)
        };
        let v_add = |d, a, b| vop2(0x003, d, a, b);
        let v_sub = |d, a, b| vop2(0x004, d, a, b);
        let v_mul = |d, a, b| vop2(0x008, d, a, b);
        let v_rsq = |d: u32, a: u32| -> u32 { 0x7E00_0000 | (d << 17) | (0x02e << 9) | (256 + a) };

        // Build `normalize(cross(v_t - v_z, v_b - v_z))` -> must match with `negate == false`.
        let build_prog = |escape_t: bool| -> Program {
            let mut w: Vec<u32> = Vec::new();
            w.extend_from_slice(&gload3(0)); // inst 0: v[0:2] = TriFetchZero3
            w.extend_from_slice(&gload3(4)); // inst 1: v[4:6] = TriFetchBasisT
            w.extend_from_slice(&gload3(8)); // inst 2: v[8:10] = TriFetchBasisB
            // e1 (v12..14) = v_t - v_z; e2 (v15..17) = v_b - v_z
            w.push(v_sub(12, 4, 0));
            w.push(v_sub(13, 5, 1));
            w.push(v_sub(14, 6, 2));
            w.push(v_sub(15, 8, 0));
            w.push(v_sub(16, 9, 1));
            w.push(v_sub(17, 10, 2));
            // cross(e1, e2) -> v24..26
            w.push(v_mul(18, 13, 17));
            w.push(v_mul(19, 14, 15));
            w.push(v_mul(20, 12, 16));
            w.push(v_mul(21, 14, 16));
            w.push(v_mul(22, 12, 17));
            w.push(v_mul(23, 13, 15));
            w.push(v_sub(24, 18, 21));
            w.push(v_sub(25, 19, 22));
            w.push(v_sub(26, 20, 23));
            // normalize(cr): len2 = cr.x^2 + cr.y^2 + cr.z^2, rsq = rsqrt(len2), out = cr * rsq (v33..35)
            w.push(v_mul(27, 24, 24));
            w.push(v_mul(28, 25, 25));
            w.push(v_mul(29, 26, 26));
            w.push(v_add(30, 27, 28));
            w.push(v_add(31, 30, 29));
            w.push(v_rsq(32, 31));
            w.push(v_mul(33, 24, 32));
            w.push(v_mul(34, 25, 32));
            w.push(v_mul(35, 26, 32));
            if escape_t {
                // Extra consumer of `v_t.x` outside normal computation -> must keep (T, B) fallback!
                w.push(v_add(36, 4, 4));
            }
            w.push(0xBF81_0000); // s_endpgm
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            crate::ir::build(&bytes, false, 40).expect("build SSA IR")
        };

        let meta_loads = vec![
            ((0, 0), RayMetaKind::TriFetchZero3),
            ((0, 1), RayMetaKind::TriFetchBasisT),
            ((0, 2), RayMetaKind::TriFetchBasisB),
        ];
        let scalar = HashMap::new();

        // Case 1: exclusive normal consumer -> DirectNormalSub matches with `negate == false`!
        let prog_ok = build_prog(false);
        let st_ok = crate::structure::structure(&prog_ok).unwrap();
        let res_ok = crate::resource::track(&prog_ok, &[], crate::native::HwStage::Cs, Some(&st_ok)).unwrap();
        let a_ok = Analyzer::new(&prog_ok, &st_ok, &res_ok, &scalar);
        let subs = a_ok.find_direct_normals(&meta_loads);
        assert_eq!(subs.len(), 1, "exclusive cross(T-0, B-0) normalizer must match DirectNormalSub");
        assert!(!subs[0].negate, "cross(T, B) must yield +N (negate=false)");
        let want_out: [Value; 3] = [
            prog_ok.blocks[0].insts[24].writes[0].1,
            prog_ok.blocks[0].insts[25].writes[0].1,
            prog_ok.blocks[0].insts[26].writes[0].1,
        ];
        assert_eq!(subs[0].out_vals, want_out);

        // Case 2: T escapes to an extra add -> paired (T, B) fallback preserved!
        let prog_esc = build_prog(true);
        let st_esc = crate::structure::structure(&prog_esc).unwrap();
        let res_esc = crate::resource::track(&prog_esc, &[], crate::native::HwStage::Cs, Some(&st_esc)).unwrap();
        let a_esc = Analyzer::new(&prog_esc, &st_esc, &res_esc, &scalar);
        assert!(
            a_esc.find_direct_normals(&meta_loads).is_empty(),
            "escaping vertex use must fall back to paired (T, B)"
        );
    }

    #[test]
    fn mirror_tlas_fbt_coalesces_with_fno_spec_into_single_rayquery_handle() {
        use crate::ir::Block;
        let blocks = vec![
            Block { offset: 0, preds: vec![], phis: vec![], insts: vec![], exit: Exit::Jump(1) },
            Block { offset: 4, preds: vec![0], phis: vec![], insts: vec![], exit: Exit::Jump(2) },
            Block { offset: 8, preds: vec![1], phis: vec![], insts: vec![], exit: Exit::Jump(3) },
            Block { offset: 12, preds: vec![2], phis: vec![], insts: vec![], exit: Exit::Jump(4) },
            Block { offset: 16, preds: vec![3], phis: vec![], insts: vec![], exit: Exit::End },
        ];
        let prog = Program { wave64: false, vgprs: 8, blocks, inputs: vec![] };
        let st = crate::structure::structure(&prog).unwrap();
        let res = crate::resource::track(&prog, &[], crate::native::HwStage::Cs, Some(&st)).unwrap();
        let scalar = HashMap::new();
        let analyzer = Analyzer::new(&prog, &st, &res, &scalar);

        let fno_flags = ray_flag::FORCE_NON_OPAQUE | ray_flag::SKIP_PROCEDURAL_PRIMITIVES;
        let mut lifted = vec![
            RayLoop {
                rq_group: 0,
                entry_block: 1,
                exit_block: 2,
                exit_pred: 1,
                skipped_blocks: vec![],
                template: RayTemplate::Bvh64,
                kind: RayLoopKind::FrontBackTrack { init_min_front_t: RayInput::Const(0) },
                tlas_load: (0, 0),
                tlas_root_load: None,
                active_ray_mask: None,
                origin: [RayInput::Const(0); 3],
                t_min: None,
                dir: [RayInput::Const(0); 3],
                t_max: RayInput::Const(0),
                instance_mask: RayInput::Const(0xFF),
                ray_flags: RayInput::Const(fno_flags),
                outputs: vec![],
                committed_meta_loads: vec![((2, 0), RayMetaKind::InstanceId)],
                direct_normals: vec![],
            },
            RayLoop {
                rq_group: 1,
                entry_block: 3,
                exit_block: 4,
                exit_pred: 3,
                skipped_blocks: vec![],
                template: RayTemplate::Bvh64,
                kind: RayLoopKind::AnyHit(AnyHitInfo {
                    opacity_block: 3,
                    entry_block: 3,
                    tail_block: 3,
                    body_blocks: vec![3],
                    skipped_tail_insts: vec![],
                    accept_mask: Value(1),
                    accept_invert: false,
                    accept_first_hit: false,
                    inputs: vec![],
                    const_lane_atoms: vec![],
                    state_vars: vec![],
                    cand_meta_loads: vec![],
                }),
                tlas_load: (0, 0),
                tlas_root_load: None,
                active_ray_mask: None,
                origin: [RayInput::Const(0); 3],
                t_min: None,
                dir: [RayInput::Const(0); 3],
                t_max: RayInput::Const(0),
                instance_mask: RayInput::Const(0xFF),
                ray_flags: RayInput::Const(fno_flags),
                outputs: vec![],
                committed_meta_loads: vec![((4, 0), RayMetaKind::InstanceId)],
                direct_normals: vec![],
            },
        ];

        let prev_mt = crate::emit::lower::TEST_NATIVE_RT_MIRROR_TLAS.replace(Some(true));
        let prev_rv = TEST_RAW_VIEWS_NONZERO.replace(Some(true));
        analyzer.coalesce_ray_query_groups(&mut lifted);
        TEST_RAW_VIEWS_NONZERO.set(prev_rv);
        crate::emit::lower::TEST_NATIVE_RT_MIRROR_TLAS.set(prev_mt);

        assert_eq!(
            lifted[0].rq_group, lifted[1].rq_group,
            "Mirror-TLAS FrontBackTrack (512) and fno_spec AnyHit (512) must coalesce onto 1 RayQuery handle"
        );
    }
}
