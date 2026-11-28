//! Native DXR Tier 1.1 acceleration structure manager.
//!
//! Opt-in via `VICTORY_NATIVE_RT=1`. Builds native host BLASes (dynamic from
//! guest geometry descriptors, static extracted on the GPU from rehydrated
//! console BVH4 buffers in `crate::memory`) and host TLASes (patching guest
//! instance descriptors' BLAS VAs through a GPU hash table).
//!
//! All `ID3D12GraphicsCommandList4` raytracing work is queued onto the guest
//! command list as owned [`DxrOp`] records and replayed on the submission
//! thread inside `Host::submit` right after `card_before` uploads dirty guest
//! buffers and arenas — never on `VICTORY_THIN` recording/worker threads.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::mem::{ManuallyDrop, offset_of, size_of};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, s};

use crate::host::{Host, HostList, Prelude, compile, host, mapped, name, not_resident, resident, transition};
use crate::object::This;

/// Size of each 256B-aligned tiered AS backing slab (32 MiB, reclaimable when empty, `ARCH-10`).
const AS_SLAB_SIZE: u64 = 32 << 20;
/// Required DXR alignment for acceleration structures and scratch buffers.
const AS_ALIGN: u64 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT as u64;
/// Default initial reservation for a pre-bound TLAS slot so `HostHeap` SRVs
/// keep a stable GPU VA once the first `BuildTlas` runs.
const DEFAULT_TLAS_RESERVE: u64 = 8 << 20;
/// Default initial reservation for a cull-enabled mirror TLAS slot (`ARCH-13`).
const DEFAULT_MIRROR_TLAS_RESERVE: u64 = 4 << 20;
/// Minimum scratch buffer chunk allocated per pooled DXR submission (4 MiB).
const MIN_SCRATCH_SIZE: u64 = 4 << 20;
/// Upper bound on static BVH4 leaf nodes extracted per BLAS.
const MAX_STATIC_NODES: u64 = 262_144;
/// Maximum valid unpacked BVH4 size (matches `memory.rs::PACKED`).
const MAX_PACKED_BVH_BYTES: u64 = 64 << 20;
/// Maximum readback bytes copied per submission for async static BLAS upgrades (16 MiB).
const MAX_READBACK_BYTES_PER_SUB: u64 = 16 << 20;
/// Maximum geometries accepted per upgraded static BLAS.
const MAX_UPGRADE_GEOMS: usize = 2048;
/// Maximum triangle slots (including sparse NaN holes) per upgraded static BLAS (524,288 tris = 18 MiB).
const MAX_UPGRADE_TRIS: usize = 524_288;
/// Maximum upgraded static BLASes drained per submission.
const MAX_UPGRADES_PER_SUB: usize = 16;
/// Maximum vertex bytes uploaded in one multi-upgrade batch (16 MiB; a single valid upgrade up to 18 MiB still runs).
const MAX_UPGRADE_BATCH_BYTES: u64 = 16 << 20;
/// Maximum active static upgrades in the primary GPU-build queue before spilling to overflow ring.
const MAX_PENDING_UPGRADES: usize = 256;
/// Maximum decoded upgrades buffered in the decode worker's bounded overflow re-queue ring (`ARCH-10`).
const MAX_OVERFLOW_UPGRADES: usize = 4096;
/// Maximum Frame-0 static BLASes awaiting deferred node readback re-queue when a single burst exceeds `MAX_READBACK_BYTES_PER_SUB`.
const MAX_PENDING_READBACK_REQUEUE: usize = 2048;
/// Maximum static BLAS compactions drained per submission.
const MAX_COMPACTIONS_PER_SUB: usize = 16;
/// Maximum queued static BLAS compactions awaiting GPU copy.
const MAX_PENDING_COMPACTIONS: usize = 8192;
/// Maximum 8-byte compacted-size postbuild queries recorded per submission (8192 × 8B = 64 KiB).
const COMPACT_INFO_SLOTS: usize = 8192;
const COMPACT_INFO_BYTES: u64 = (COMPACT_INFO_SLOTS * 8) as u64;
/// Number of GPU timestamp query slots pooled per `DxrSubmission` when `VICTORY_GPU_TIMING=1`.
#[cfg(feature = "diagnostics")]
const TIMING_SLOTS: u32 = 1024;
/// Pooled per-submission auxiliary buffers larger than this are dropped on fence completion.
const MAX_POOLED_AUX_BYTES: u64 = 4 << 20;
/// Pooled per-submission scratch buffers larger than this are dropped on fence completion (`ARCH-10`: 8 MiB).
const MAX_POOLED_SCRATCH_BYTES: u64 = 8 << 20;
/// Pooled per-submission static-extraction vertex buffers larger than this are dropped on fence completion.
const MAX_POOLED_EXTRACT_VERTS_BYTES: u64 = 4 << 20;
/// Pooled per-submission patched host-instance buffers larger than this are dropped on fence completion.
const MAX_POOLED_HOST_INSTANCES_BYTES: u64 = 2 << 20;
/// Number of idle DXR submissions after which unused static staging buffers on a `DxrSubmission` are dropped asynchronously (`ARCH-10`).
const IDLE_STAGING_TRIM_SUBS: u64 = 60;
/// Evict BLASes unreachable from any scanned TLAS for more than this many TLAS epochs (`ARCH-10`).
const BLAS_GC_UNREACHABLE_EPOCHS: u64 = 300;
/// Minimum TLAS epochs between periodic `AS_MAP` reachability GC sweeps.
const BLAS_GC_SWEEP_INTERVAL_EPOCHS: u64 = 60;
/// Dynamic `dxr_room_kept()` steady-state Local VRAM headroom reserve (128 MiB).
const DXR_STEADY_RESERVE_BYTES: u64 = 128 << 20;
/// Dynamic `dxr_room_kept()` streaming Local VRAM headroom reserve (256 MiB).
const DXR_STREAMING_RESERVE_BYTES: u64 = 256 << 20;
/// Submissions after the last static BLAS build/upgrade/compaction that count as streaming.
const DXR_STREAMING_WINDOW_SUBS: u64 = 120;
/// Maximum pooled DXR command lists kept in `DxrState::free_lists`.
const MAX_FREE_LISTS: usize = 16;
/// Maximum pooled DXR submission containers kept in `DxrState::free_submissions`.
const MAX_FREE_SUBS: usize = 4;
/// Maximum cached dynamic BLAS `GetRaytracingAccelerationStructurePrebuildInfo` entries.
const MAX_PREBUILD_CACHE: usize = 4096;
/// Maximum cached TLAS `GetRaytracingAccelerationStructurePrebuildInfo` entries.
const MAX_TLAS_PREBUILD_CACHE: usize = 2048;
/// Maximum queued async BVH4 readback decode batches in the bounded decode queue.
const MAX_DECODE_QUEUE_JOBS: usize = 128;
/// Maximum individual dirty hash-table slots tracked before falling back to `[min..=max]`.
const MAX_DIRTY_HASH_SLOTS: usize = 64;
/// Coalesce dirty hash-table runs separated by at most this many 16B slots (4 KiB).
const HASH_DIRTY_MERGE_GAP: usize = 256;
/// Open-addressed GPU hash table capacity (65536 × 32B = 2 MiB).
const HASH_CAPACITY: usize = 65_536;
const HASH_MASK: u32 = (HASH_CAPACITY - 1) as u32;
const HASH_ENTRY_BYTES: usize = 32;
const HASH_TABLE_BYTES: u64 = (HASH_CAPACITY * HASH_ENTRY_BYTES) as u64;

/// Whether `VICTORY_NATIVE_RT=1` was requested in the environment.
pub fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NATIVE_RT").is_some_and(|v| v == "1"))
}

/// Whether native DXR is active (`VICTORY_NATIVE_RT=1` and host supports
/// DXR Tier 1.1 + `ID3D12Device5`).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| requested() && host().is_some_and(Host::raytracing_tier_1_1))
}

/// Whether dynamic `BuildRaytracingAccelerationStructureBatched2X` and
/// `CopyRaytracingAccelerationStructure` calls should skip forwarding to
/// `XboxBvhBuilder.dll` and fulfill prebuild/postbuild directly from host DXR
/// (`VICTORY_NATIVE_RT_SUPPRESS_BUILDER=0` re-enables dual-run; default ON
/// when Native DXR is active).
pub fn suppress_builder() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        enabled()
            && std::env::var_os("VICTORY_NATIVE_RT_SUPPRESS_BUILDER")
                .is_none_or(|v| v != "0")
    })
}

fn should_rehydrate_via_builder(
    native_rt: bool,
    suppress_builder: bool,
    rehydrate_env: Option<&std::ffi::OsStr>,
) -> bool {
    native_rt && suppress_builder && rehydrate_env.is_none_or(|v| v != "0")
}

/// Whether `XboxBvhBuilder.dll`'s static BVH4 rehydration compute dispatches
/// should be allowed to run through `pipeline::builder_runs()` even when
/// `VICTORY_BVH_BUILDER=0`, because `VICTORY_NATIVE_RT=1` +
/// `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1` suppresses dynamic builds/copies
/// while still relying on rehydration for static world BLAS extraction.
/// Set `VICTORY_NATIVE_RT_REHYDRATE=0` to disable.
pub fn rehydrate_via_builder() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        should_rehydrate_via_builder(
            enabled(),
            suppress_builder(),
            std::env::var_os("VICTORY_NATIVE_RT_REHYDRATE").as_deref(),
        )
    })
}

fn should_skip_builder_rt_off(
    skip_when_off: bool,
    native_rt: bool,
    bvh_builder_env: bool,
    builder_runs: bool,
) -> bool {
    skip_when_off && !native_rt && !bvh_builder_env && !builder_runs
}

/// Whether `XboxBvhBuilder.dll` command recording (`build_batched_2x`, `copy_as`,
/// `rehydrate_batched_2x`) should be skipped because ray tracing is completely off
/// (neither Native DXR nor `VICTORY_BVH_BUILDER=1` is active).
/// Set `VICTORY_BVH_SKIP_OFF=0` to keep forwarding in RT-off mode.
pub fn skip_builder_rt_off() -> bool {
    static SKIP_WHEN_OFF: OnceLock<bool> = OnceLock::new();
    static BVH_BUILDER_ENV: OnceLock<bool> = OnceLock::new();
    let skip_when_off = *SKIP_WHEN_OFF
        .get_or_init(|| std::env::var_os("VICTORY_BVH_SKIP_OFF").is_none_or(|v| v != "0"));
    let bvh_builder_env = *BVH_BUILDER_ENV
        .get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1"));
    should_skip_builder_rt_off(
        skip_when_off,
        enabled(),
        bvh_builder_env,
        crate::pipeline::builder_runs(),
    )
}

/// Default number of consecutive dynamic BLAS refits before forcing a clean full
/// rebuild to prevent BVH quality degradation on deforming skinned meshes.
const DEFAULT_REFIT_PERIOD: u32 = 64;

/// Whether dynamic BLAS builds automatically enable `ALLOW_UPDATE` and promote
/// topology-matching rebuilds to in-place `PERFORM_UPDATE` refits
/// (`VICTORY_NATIVE_RT_REFIT=0` disables; default ON).
pub fn refit_dynamic_blas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_REFIT").is_none_or(|v| v != "0")
    })
}

fn parse_refit_period(raw: Option<&str>) -> u32 {
    raw.and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_REFIT_PERIOD)
}

/// Number of consecutive dynamic BLAS refits before forcing a clean full rebuild
/// (`VICTORY_NATIVE_RT_REFIT_PERIOD`, default `64`; `0` = never force periodic rebuild).
fn refit_period() -> u32 {
    static PERIOD: OnceLock<u32> = OnceLock::new();
    *PERIOD.get_or_init(|| parse_refit_period(std::env::var("VICTORY_NATIVE_RT_REFIT_PERIOD").ok().as_deref()))
}

/// Bit in `cull_mask_cfg` selecting tag/set mode (`VICTORY_NATIVE_RT_CULL_MASK=tag:0x80`)
/// instead of clear mode (default: clear mask bit(s) on `TRIANGLE_CULL_DISABLE` instances).
const CULL_MASK_TAG_MODE: u32 = 0x100;

fn parse_u8_mask(s: &str) -> Option<u8> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u8::from_str_radix(hex, 16).ok()
    } else if s.eq_ignore_ascii_case("1") || s.eq_ignore_ascii_case("on") || s.eq_ignore_ascii_case("true") {
        Some(0x80)
    } else if s.eq_ignore_ascii_case("0") || s.eq_ignore_ascii_case("off") || s.eq_ignore_ascii_case("false") {
        Some(0)
    } else {
        s.parse::<u8>().ok()
    }
}

/// Whether cull-enabled mirror TLASes are built alongside primary TLASes
/// (`VICTORY_NATIVE_RT_MIRROR_TLAS=0` disables; default ON, `ARCH-13`).
pub(crate) fn mirror_tlas_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_MIRROR_TLAS").is_none_or(|v| v != "0")
    })
}

/// Clears `D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE` (bit 0 of the
/// 8-bit `Flags` field packed in bits 24..31 of `r3.y`) while strictly preserving
/// `TRIANGLE_FRONT_COUNTERCLOCKWISE` (bit 1 = `0x0200_0000`), all other instance
/// flags, and the 24-bit `InstanceContributionToHitGroupIndex`.
#[cfg(test)]
#[inline]
fn clear_cull_disable_flag(r3_y: u32) -> u32 {
    r3_y & !0x0100_0000u32
}

/// Default `cull_mask_cfg` when Native DXR is active, mirror TLAS is disabled,
/// and `VICTORY_NATIVE_RT_CULL_MASK` is unset: tags `TRIANGLE_CULL_DISABLE`
/// instances with unused mask bit `0x02` (`tag:0x02`) to pair with
/// `VICTORY_NATIVE_RT_FBT_SPLIT_MASK=0x02`.
const DEFAULT_CULL_MASK_CFG: u32 = CULL_MASK_TAG_MODE | 0x02;

/// Parses `VICTORY_NATIVE_RT_CULL_MASK` into the shader root constant `cull_mask_cfg`:
/// - unset / empty -> `0` when `mirror_tlas` is enabled (preserving full 8-bit guest
///   `InstanceInclusionMask`), else `DEFAULT_CULL_MASK_CFG` (`tag:0x02`).
/// - `"0"` / `"off"` / `"false"` -> `0` (disabled; instances keep exact guest `InstanceMask`).
/// - `"1"` / `"0x80"` / `"128"` / custom `u8` -> `mask` (clear `mask` bits from
///   `InstanceMask` on valid patched instances with `TRIANGLE_CULL_DISABLE`, keeping
///   the original non-zero mask if clearing would zero it).
/// - `"tag:<mask>"` -> `0x100 | mask` (OR `mask` into `InstanceMask` on valid patched
///   instances with `TRIANGLE_CULL_DISABLE`).
fn parse_cull_mask_cfg_for(raw: Option<&str>, mirror_tlas: bool) -> u32 {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return if mirror_tlas { 0 } else { DEFAULT_CULL_MASK_CFG };
    };
    let (tag_mode, rest) = if let Some(r) = s.strip_prefix("tag:").or_else(|| s.strip_prefix("TAG:")) {
        (true, r)
    } else {
        (false, s)
    };
    let mask = parse_u8_mask(rest).unwrap_or(0);
    if mask == 0 {
        0
    } else if tag_mode {
        CULL_MASK_TAG_MODE | u32::from(mask)
    } else {
        u32::from(mask)
    }
}

#[cfg(test)]
fn parse_cull_mask_cfg(raw: Option<&str>) -> u32 {
    parse_cull_mask_cfg_for(raw, false)
}

fn cull_mask_cfg() -> u32 {
    static CFG: OnceLock<u32> = OnceLock::new();
    *CFG.get_or_init(|| {
        parse_cull_mask_cfg_for(
            std::env::var("VICTORY_NATIVE_RT_CULL_MASK").ok().as_deref(),
            mirror_tlas_enabled(),
        )
    })
}

/// Configured `InstanceMask` bitmask for separating `TRIANGLE_CULL_DISABLE` instances
/// (`VICTORY_NATIVE_RT_CULL_MASK`; `0` when disabled).
#[allow(dead_code)]
pub(crate) fn cull_disable_mask() -> u8 {
    (cull_mask_cfg() & 0xFF) as u8
}

/// Cumulative GPU instance-patching telemetry `[cull_disable_instances, cull_masked_instances]`.
static CULL_TELEMETRY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Returns cumulative `(cull_disable_instances, cull_masked_instances)` observed by
/// `PATCH_INSTANCES_HLSL` across logged frames.
#[allow(dead_code)]
pub(crate) fn cull_disable_telemetry() -> (u64, u64) {
    (
        CULL_TELEMETRY[0].load(Ordering::Relaxed),
        CULL_TELEMETRY[1].load(Ordering::Relaxed),
    )
}

/// Byte size written by D3D12 for each `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE`:
/// - `0` (`COMPACTED_SIZE`): 8 bytes
/// - `1` (`TOOLS_VISUALIZATION`): 8 bytes
/// - `2` (`SERIALIZATION`): 16 bytes
/// - `3` (`CURRENT_SIZE`): 8 bytes
fn postbuild_byte_size(info_type: u32) -> Option<u64> {
    match info_type {
        0 | 1 | 3 => Some(8),
        2 => Some(16),
        _ => None,
    }
}

/// Whether opaque BLAS geometries should also set
/// `D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION`
/// (`VICTORY_NATIVE_RT_NO_DUP_OPAQUE=1`, default OFF) for A/B testing
/// `FORCE_NON_OPAQUE` ray-query loops.
fn no_dup_opaque() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_NO_DUP_OPAQUE").is_some_and(|v| v == "1")
    })
}

/// Whether static BLAS builds set `ALLOW_COMPACTION`, query postbuild compacted sizes,
/// and asynchronously compact into smaller slab slices (`VICTORY_NATIVE_RT_COMPACT=0` disables).
fn compact_static_blas() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_NATIVE_RT_COMPACT").is_none_or(|v| v != "0")
    })
}

fn static_blas_build_flags_for(allow_compaction: bool) -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
    if allow_compaction {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION
    } else {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
    }
}

fn static_blas_build_flags() -> D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS {
    static_blas_build_flags_for(compact_static_blas())
}

/// Whether DXR command lists use scoped Enhanced `D3D12_GLOBAL_BARRIER`s instead of
/// legacy `uav_barrier(None)` when `ID3D12GraphicsCommandList7` + Enhanced Barriers
/// are supported (`VICTORY_DXR_ENHANCED_BARRIERS=0` disables; default ON).
fn dxr_enhanced_barriers_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_DXR_ENHANCED_BARRIERS").is_none_or(|v| v != "0")
    })
}

fn enhanced_barriers_supported(host: &Host) -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: valid D3D12 feature structure of matching size.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        ok.is_ok() && o.EnhancedBarriersSupported.as_bool()
    })
}

fn dxr_list7(host: &Host, list: &ID3D12GraphicsCommandList) -> Option<ID3D12GraphicsCommandList7> {
    if !dxr_enhanced_barriers_wanted() || !enhanced_barriers_supported(host) {
        return None;
    }
    list.cast::<ID3D12GraphicsCommandList7>().ok()
}

/// Returns `true` when compacting a static BLAS from `uncompacted_size` to
/// `compacted_size` (rounded up to 256B `AS_ALIGN`) saves meaningful slab VRAM.
fn should_compact(uncompacted_size: u64, compacted_size: u64) -> bool {
    if compacted_size == 0 {
        return false;
    }
    let aligned = compacted_size.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
    let Some(saved) = uncompacted_size.checked_sub(aligned) else {
        return false;
    };
    saved > 4096 && (saved.saturating_mul(10) >= uncompacted_size || saved >= 64 * 1024)
}

fn static_geom_flags_with(opaque: bool, add_no_dup_opaque: bool) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    if opaque {
        if add_no_dup_opaque {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                | D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION
        } else {
            D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
        }
    } else {
        D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION
    }
}

fn static_geom_flags(opaque: bool) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    static_geom_flags_with(opaque, no_dup_opaque())
}

fn dynamic_geom_flags_with(raw_flags: u32, add_no_dup_opaque: bool) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    let opaque = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE.0 as u32;
    let no_dup = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION.0 as u32;
    let flags = if (raw_flags & opaque) == 0 || add_no_dup_opaque {
        raw_flags | no_dup
    } else {
        raw_flags
    };
    D3D12_RAYTRACING_GEOMETRY_FLAGS(flags as i32)
}

fn dynamic_geom_flags(raw_flags: u32) -> D3D12_RAYTRACING_GEOMETRY_FLAGS {
    dynamic_geom_flags_with(raw_flags, no_dup_opaque())
}

/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBuildAsInputs {
    pub kind: u32,
    pub flags: u32,
    pub num_descs: u32,
    pub descs_layout: u32,
    pub ptr_or_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsInputs>() == 0x18);

/// Guest `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBuildAsDesc {
    pub dest_va: u64,
    pub inputs: GuestBuildAsInputs,
    pub source_va: u64,
    pub scratch_va: u64,
}
const _: () = assert!(size_of::<GuestBuildAsDesc>() == 0x30);
const _: () = assert!(offset_of!(GuestBuildAsDesc, inputs) == 0x08);
const _: () = assert!(offset_of!(GuestBuildAsDesc, source_va) == 0x20);
const _: () = assert!(offset_of!(GuestBuildAsDesc, scratch_va) == 0x28);

/// Guest `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestPostbuildInfoDesc {
    pub dest_buffer: u64,
    pub info_type: u32,
    pub _pad: u32,
}
const _: () = assert!(size_of::<GuestPostbuildInfoDesc>() == 0x10);

/// Guest `D3D12XBOX_XDXR_STREAM_DESC` (`0x10` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestStreamDesc {
    pub size_in_bytes: u64,
    pub data: *const c_void,
}
const _: () = assert!(size_of::<GuestStreamDesc>() == 0x10);

/// Guest `D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2` (`0x50` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestBatchedDesc2 {
    pub desc: GuestBuildAsDesc,
    pub num_postbuild_info_descs: u32,
    pub _pad: u32,
    pub postbuild_info_descs: *const GuestPostbuildInfoDesc,
    pub stream_desc: GuestStreamDesc,
}
const _: () = assert!(size_of::<GuestBatchedDesc2>() == 0x50);
const _: () = assert!(offset_of!(GuestBatchedDesc2, num_postbuild_info_descs) == 0x30);
const _: () = assert!(offset_of!(GuestBatchedDesc2, postbuild_info_descs) == 0x38);
const _: () = assert!(offset_of!(GuestBatchedDesc2, stream_desc) == 0x40);

/// Guest `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GuestPrebuildInfo {
    pub result_data_max_size_in_bytes: u64,
    pub scratch_data_size_in_bytes: u64,
    pub update_scratch_data_size_in_bytes: u64,
}
const _: () = assert!(size_of::<GuestPrebuildInfo>() == 0x18);

/// Guest `D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC` (`0x30` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryTrianglesDesc {
    pub transform3x4: u64,
    pub index_format: u32,
    pub vertex_format: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    pub index_buffer: u64,
    pub vertex_buffer_va: u64,
    pub vertex_buffer_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryTrianglesDesc>() == 0x30);

/// Guest `D3D12_RAYTRACING_GEOMETRY_AABBS_DESC` (`0x18` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryAabbsDesc {
    pub aabb_count: u64,
    pub aabbs_va: u64,
    pub aabbs_stride: u64,
}
const _: () = assert!(size_of::<GuestGeometryAabbsDesc>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy)]
pub union GuestGeometryUnion {
    pub triangles: GuestGeometryTrianglesDesc,
    pub aabbs: GuestGeometryAabbsDesc,
}

/// Guest `D3D12_RAYTRACING_GEOMETRY_DESC` (`0x38` bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GuestGeometryDesc {
    pub kind: u32,
    pub flags: u32,
    pub data: GuestGeometryUnion,
}
const _: () = assert!(size_of::<GuestGeometryDesc>() == 0x38);

/// Owned geometry specification copied out of guest memory during command recording.
#[derive(Clone)]
pub(crate) enum GeometryKind {
    Triangles {
        transform3x4: u64,
        index_format: u32,
        vertex_format: u32,
        index_count: u32,
        vertex_count: u32,
        index_buffer: u64,
        vertex_buffer_va: u64,
        vertex_buffer_stride: u64,
    },
    Aabbs {
        aabb_count: u64,
        aabbs_va: u64,
        aabbs_stride: u64,
    },
}

#[derive(Clone)]
pub(crate) struct GeometrySpec {
    pub(crate) flags: u32,
    pub(crate) kind: GeometryKind,
}

#[derive(Clone, Copy)]
pub(crate) struct PostbuildSpec {
    pub(crate) dest_buffer: u64,
    pub(crate) info_type: u32,
}

/// Queued DXR operation attached to a guest command list and replayed on `victory-submit`.
#[derive(Clone)]
pub(crate) enum DxrOp {
    BuildBlas {
        dest_va: u64,
        source_va: u64,
        flags: u32,
        geometries: Vec<GeometrySpec>,
        postbuild: Vec<PostbuildSpec>,
    },
    BuildTlas {
        dest_va: u64,
        flags: u32,
        num_instances: u32,
        instance_descs_va: u64,
        postbuild: Vec<PostbuildSpec>,
    },
    CopyAs {
        dest_va: u64,
        source_va: u64,
        mode: u32,
    },
    ExtractStaticBlas {
        dest_va: u64,
        size_bytes: u64,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AsKind {
    Blas,
    Tlas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlabTier {
    /// Long-lived compacted static BLASes (`record_static_compactions`, `CopyAs(COMPACT)`).
    StaticCompacted,
    /// Dynamic BLASes, TLASes, mirror TLASes, clones, and uncompacted Frame-0/pre-compact static BLASes.
    TransientDynamicTlas,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlabSlice {
    tier: SlabTier,
    slab_idx: usize,
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RetiredSlice {
    release_after_tlas_epoch: u64,
    release_after_sub_seq: u64,
    release_after_dxr_fence: u64,
    slice: SlabSlice,
}

struct MirrorTlas {
    #[allow(dead_code)]
    buffer: ID3D12Resource,
    slice: SlabSlice,
    gpu_va: u64,
}

struct HostAs {
    buffer: ID3D12Resource,
    slice: SlabSlice,
    gpu_va: u64,
    /// Optional cull-enabled mirror TLAS (`ARCH-13`) stored alongside primary TLAS entries.
    mirror_tlas: Option<MirrorTlas>,
    kind: AsKind,
    flags: u32,
    num_instances: u32,
    built: bool,
    generation: u64,
    topo_hash: u64,
    update_scratch_bytes: u64,
    consecutive_refits: u32,
    /// Compacted byte size read back from `POSTBUILD_INFO_COMPACTED_SIZE` (`0` if unknown).
    compacted_size: u64,
    /// Last `tlas_build_epoch` in which this AS was created, updated, or observed in a TLAS.
    last_seen_tlas_epoch: AtomicU64,
    /// Fingerprint of resolved dynamic BLAS VB/IB/transform GPU VAs & strides for safe `BUILD_NORMALS` skip.
    geom_va_hash: u64,
    /// Word slice in `dxr_normals` Region 1 (per-geometry offsets; `0` = none).
    normals_geom_off: u32,
    normals_geom_len: u32,
    /// Word slice in `dxr_normals` Region 2 (packed octahedral triangle normals; `0` = none).
    normals_data_off: u32,
    normals_data_len: u32,
}

/// Simple best-fit bump + deferred-reclaim allocator over a contiguous `u32` word range
/// inside `dxr_normals` (Region 1 geometry tables and Region 2 packed triangle normals).
struct WordPool {
    base: u32,
    end: u32,
    used: u32,
    free: Vec<(u32, u32)>,
    retired: VecDeque<(u64, u64, u64, u32, u32)>,
}

impl WordPool {
    fn new(base: u32, end: u32) -> Self {
        Self {
            base,
            end,
            used: base,
            free: Vec::new(),
            retired: VecDeque::new(),
        }
    }

    fn alloc(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return None;
        }
        let mut best: Option<(usize, u32)> = None;
        for (i, &(_, sz)) in self.free.iter().enumerate() {
            if sz >= count && best.is_none_or(|(_, bsz)| sz < bsz) {
                best = Some((i, sz));
                if sz == count {
                    break;
                }
            }
        }
        if let Some((idx, sz)) = best {
            let (off, _) = self.free[idx];
            if sz == count {
                self.free.remove(idx);
            } else {
                self.free[idx] = (off + count, sz - count);
            }
            return Some(off);
        }
        let next = self.used.checked_add(count)?;
        if next > self.end {
            return None;
        }
        let off = self.used;
        self.used = next;
        Some(off)
    }

    fn retire(
        &mut self,
        off: u32,
        len: u32,
        release_tlas_epoch: u64,
        release_sub_seq: u64,
        release_dxr_fence: u64,
    ) {
        if off >= self.base && len > 0 {
            self.retired.push_back((
                release_tlas_epoch,
                release_sub_seq,
                release_dxr_fence,
                off,
                len,
            ));
        }
    }

    fn free_now(&mut self, off: u32, len: u32) {
        if off < self.base || len == 0 {
            return;
        }
        let pos = self.free.partition_point(|&(o, _)| o < off);
        self.free.insert(pos, (off, len));
        // Coalesce with right neighbor.
        if pos + 1 < self.free.len() && self.free[pos].0 + self.free[pos].1 == self.free[pos + 1].0
        {
            self.free[pos].1 += self.free[pos + 1].1;
            self.free.remove(pos + 1);
        }
        // Coalesce with left neighbor.
        if pos > 0 && self.free[pos - 1].0 + self.free[pos - 1].1 == self.free[pos].0 {
            self.free[pos - 1].1 += self.free[pos].1;
            self.free.remove(pos);
        }
        // Trim trailing free block against bump pointer.
        while let Some(&(last_off, last_len)) = self.free.last() {
            if last_off + last_len == self.used {
                self.used = last_off;
                self.free.pop();
            } else {
                break;
            }
        }
    }

    fn drain_retired(
        &mut self,
        completed_tlas_epoch: u64,
        completed_sub_seq: u64,
        completed_dxr_fence: u64,
    ) {
        let mut i = 0;
        while i < self.retired.len() {
            let (t_ep, s_seq, d_fence, off, len) = self.retired[i];
            if completed_tlas_epoch >= t_ep
                && completed_sub_seq >= s_seq
                && completed_dxr_fence >= d_fence
            {
                self.retired.swap_remove_back(i);
                self.free_now(off, len);
            } else {
                i += 1;
            }
        }
    }
}

/// Lightweight `Copy` handle returned by [`allocate_or_reuse_as`] and [`allocate_fresh_as`]
/// so hot AS-recording paths never clone `HostAs::buffer` (avoiding COM `AddRef`/`Release`
/// refcount traffic under the `AS_MAP` write lock).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AllocatedAs {
    gpu_va: u64,
    mirror_gpu_va: Option<u64>,
    generation: u64,
    slice_size: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded; entries are synchronized by `RwLock`.
unsafe impl Send for HostAs {}
unsafe impl Sync for HostAs {}

struct Slab<B = ID3D12Resource> {
    buffer: B,
    gpu_base: u64,
    size: u64,
    used: u64,
    /// Sorted, non-overlapping, non-adjacent `(offset, size)` free ranges below `used`.
    free: Vec<(u64, u64)>,
}

// SAFETY: `ID3D12Resource` is free-threaded; accessed under `DXR_STATE` mutex.
unsafe impl Send for Slab<ID3D12Resource> {}

struct SlabAllocator<B = ID3D12Resource> {
    tier: SlabTier,
    slabs: Vec<Option<Slab<B>>>,
    retired: VecDeque<RetiredSlice>,
}

impl<B> Default for SlabAllocator<B> {
    fn default() -> Self {
        Self::new(SlabTier::TransientDynamicTlas)
    }
}

impl<B> SlabAllocator<B> {
    fn new(tier: SlabTier) -> Self {
        Self {
            tier,
            slabs: Vec::new(),
            retired: VecDeque::new(),
        }
    }
}

impl<B: Clone> SlabAllocator<B> {
    fn alloc_with<F>(&mut self, size: u64, make_slab: F) -> Option<(B, SlabSlice, u64)>
    where
        F: FnOnce(u64, usize) -> Option<(B, u64)>,
    {
        let need = size.max(AS_ALIGN).next_multiple_of(AS_ALIGN);

        // Best-fit search across reclaimed free ranges before bump-allocating.
        let mut best: Option<(usize, usize, u64)> = None;
        'search: for (slab_idx, slot) in self.slabs.iter().enumerate() {
            let Some(slab) = slot.as_ref() else { continue };
            for (free_idx, &(_, range_size)) in slab.free.iter().enumerate() {
                if range_size >= need {
                    if best.is_none_or(|(_, _, best_sz)| range_size < best_sz) {
                        best = Some((slab_idx, free_idx, range_size));
                        if range_size == need {
                            break 'search;
                        }
                    }
                }
            }
        }
        if let Some((slab_idx, free_idx, _)) = best {
            let slab = self.slabs[slab_idx].as_mut()?;
            let (offset, range_size) = slab.free[free_idx];
            if range_size == need {
                slab.free.remove(free_idx);
            } else {
                slab.free[free_idx] = (offset + need, range_size - need);
            }
            let slice = SlabSlice {
                tier: self.tier,
                slab_idx,
                offset,
                size: need,
            };
            return Some((slab.buffer.clone(), slice, slab.gpu_base + offset));
        }

        // Fall back to lowest-index-first bump allocation so upper slabs drain and empty.
        if let Some((slab_idx, slab)) = self
            .slabs
            .iter_mut()
            .enumerate()
            .find_map(|(i, slot)| slot.as_mut().filter(|s| s.used + need <= s.size).map(|s| (i, s)))
        {
            let offset = slab.used;
            slab.used += need;
            let slice = SlabSlice {
                tier: self.tier,
                slab_idx,
                offset,
                size: need,
            };
            return Some((slab.buffer.clone(), slice, slab.gpu_base + offset));
        }

        // Allocate a fresh 32 MiB (or oversized) backing slab, reusing any reclaimed slot index.
        let slab_size = need.max(AS_SLAB_SIZE).next_multiple_of(AS_ALIGN);
        let reuse_idx = self.slabs.iter().position(Option::is_none);
        let slab_idx = reuse_idx.unwrap_or(self.slabs.len());
        let (buffer, gpu_base) = make_slab(slab_size, slab_idx)?;
        let new_slab = Slab {
            buffer: buffer.clone(),
            gpu_base,
            size: slab_size,
            used: need,
            free: Vec::new(),
        };
        if let Some(slot) = reuse_idx.and_then(|i| self.slabs.get_mut(i)) {
            *slot = Some(new_slab);
        } else {
            self.slabs.push(Some(new_slab));
        }
        let slice = SlabSlice {
            tier: self.tier,
            slab_idx,
            offset: 0,
            size: need,
        };
        Some((buffer, slice, gpu_base))
    }

    fn retire(
        &mut self,
        slice: SlabSlice,
        release_after_tlas_epoch: u64,
        release_after_sub_seq: u64,
        release_after_dxr_fence: u64,
    ) {
        if slice.size == 0 {
            return;
        }
        self.retired.push_back(RetiredSlice {
            release_after_tlas_epoch,
            release_after_sub_seq,
            release_after_dxr_fence,
            slice,
        });
    }

    fn drain_retired(
        &mut self,
        completed_tlas_epoch: u64,
        completed_sub_seq: u64,
        completed_dxr_fence: u64,
    ) -> usize {
        let mut freed = 0usize;
        let slabs = &mut self.slabs;
        self.retired.retain(|r| {
            if completed_tlas_epoch >= r.release_after_tlas_epoch
                && completed_sub_seq >= r.release_after_sub_seq
                && completed_dxr_fence >= r.release_after_dxr_fence
            {
                Self::free_in_slabs(slabs, r.slice);
                freed += 1;
                false
            } else {
                true
            }
        });
        freed
    }

    /// Reclaims any completely empty slab (`used == 0 && free.is_empty()`) that has no
    /// pending retired slices referencing it, returning `(buffer, size)` pairs for async
    /// COM + VRAM retirement.
    fn reclaim_empty_slabs(&mut self) -> Vec<(B, u64)> {
        let mut out = Vec::new();
        for (idx, slot) in self.slabs.iter_mut().enumerate() {
            let empty = slot
                .as_ref()
                .is_some_and(|s| s.used == 0 && s.free.is_empty());
            if empty && !self.retired.iter().any(|r| r.slice.slab_idx == idx) {
                if let Some(slab) = slot.take() {
                    out.push((slab.buffer, slab.size));
                }
            }
        }
        out
    }

    fn free_now(&mut self, slice: SlabSlice) {
        Self::free_in_slabs(&mut self.slabs, slice);
    }

    #[cfg(test)]
    fn free(&mut self, slice: SlabSlice) {
        Self::free_in_slabs(&mut self.slabs, slice);
    }

    fn free_in_slabs(slabs: &mut [Option<Slab<B>>], slice: SlabSlice) {
        if slice.size == 0 {
            return;
        }
        let Some(slab) = slabs.get_mut(slice.slab_idx).and_then(Option::as_mut) else {
            return;
        };
        let mut start = slice.offset;
        let mut end = match slice.offset.checked_add(slice.size) {
            Some(e) if e <= slab.used => e,
            _ => {
                debug_assert!(false, "dxr: slab free out of bounds");
                return;
            }
        };
        let idx = slab.free.partition_point(|&(off, _)| off < start);
        if idx > 0 {
            let (p_off, p_sz) = slab.free[idx - 1];
            if p_off + p_sz > start {
                debug_assert!(false, "dxr: overlapping slab free with previous range");
                return;
            }
        }
        if idx < slab.free.len() && end > slab.free[idx].0 {
            debug_assert!(false, "dxr: overlapping slab free with next range");
            return;
        }

        let merge_prev = idx > 0 && slab.free[idx - 1].0 + slab.free[idx - 1].1 == start;
        let merge_next = idx < slab.free.len() && end == slab.free[idx].0;
        let insert_idx = if merge_prev {
            start = slab.free[idx - 1].0;
            idx - 1
        } else {
            idx
        };
        if merge_next {
            end = slab.free[idx].0 + slab.free[idx].1;
        }

        match (merge_prev, merge_next) {
            (true, true) => {
                slab.free.remove(idx);
                if end == slab.used {
                    slab.free.remove(insert_idx);
                    slab.used = start;
                } else {
                    slab.free[insert_idx] = (start, end - start);
                }
            }
            (true, false) => {
                if end == slab.used {
                    slab.free.remove(insert_idx);
                    slab.used = start;
                } else {
                    slab.free[insert_idx] = (start, end - start);
                }
            }
            (false, true) => {
                if end == slab.used {
                    slab.free.remove(idx);
                    slab.used = start;
                } else {
                    slab.free[idx] = (start, end - start);
                }
            }
            (false, false) => {
                if end == slab.used {
                    slab.used = start;
                } else {
                    slab.free.insert(idx, (start, end - start));
                }
            }
        }
    }

    /// Returns `(total_allocated_bytes, available_bytes)` where `available_bytes`
    /// is free-list ranges plus unused bump tails across all live slabs.
    fn stats(&self) -> (u64, u64) {
        let mut total = 0u64;
        let mut free = 0u64;
        for s in self.slabs.iter().flatten() {
            total = total.saturating_add(s.size);
            let tail = s.size.saturating_sub(s.used);
            let recycled: u64 = s.free.iter().map(|&(_, sz)| sz).sum();
            free = free.saturating_add(tail).saturating_add(recycled);
        }
        (total, free)
    }
}

/// Two-tier 32 MiB AS slab allocator separating long-lived compacted static BLASes
/// from high-churn dynamic BLASes, TLASes, and uncompacted pre-compact BLASes (`ARCH-10`).
struct TieredSlabAllocator<B = ID3D12Resource> {
    static_compacted: SlabAllocator<B>,
    transient_dynamic_tlas: SlabAllocator<B>,
}

impl<B> Default for TieredSlabAllocator<B> {
    fn default() -> Self {
        Self {
            static_compacted: SlabAllocator::new(SlabTier::StaticCompacted),
            transient_dynamic_tlas: SlabAllocator::new(SlabTier::TransientDynamicTlas),
        }
    }
}

impl<B: Clone> TieredSlabAllocator<B> {
    fn tier_mut(&mut self, tier: SlabTier) -> &mut SlabAllocator<B> {
        match tier {
            SlabTier::StaticCompacted => &mut self.static_compacted,
            SlabTier::TransientDynamicTlas => &mut self.transient_dynamic_tlas,
        }
    }

    fn retire(
        &mut self,
        slice: SlabSlice,
        release_after_tlas_epoch: u64,
        release_after_sub_seq: u64,
        release_after_dxr_fence: u64,
    ) {
        self.tier_mut(slice.tier).retire(
            slice,
            release_after_tlas_epoch,
            release_after_sub_seq,
            release_after_dxr_fence,
        );
    }

    fn free_now(&mut self, slice: SlabSlice) {
        self.tier_mut(slice.tier).free_now(slice);
    }

    fn drain_retired(
        &mut self,
        completed_tlas_epoch: u64,
        completed_sub_seq: u64,
        completed_dxr_fence: u64,
    ) -> usize {
        self.static_compacted
            .drain_retired(completed_tlas_epoch, completed_sub_seq, completed_dxr_fence)
            + self.transient_dynamic_tlas.drain_retired(
                completed_tlas_epoch,
                completed_sub_seq,
                completed_dxr_fence,
            )
    }

    fn stats(&self) -> (u64, u64) {
        let (t0, f0) = self.static_compacted.stats();
        let (t1, f1) = self.transient_dynamic_tlas.stats();
        (t0.saturating_add(t1), f0.saturating_add(f1))
    }
}

static DXR_SLAB_BYTES: AtomicU64 = AtomicU64::new(0);
/// Local VRAM held by non-slab DXR buffers (`normals_gpu`, `hash_gpu`, pooled scratch/staging UAVs).
static DXR_AUX_VRAM_BYTES: AtomicU64 = AtomicU64::new(0);
/// Monotonic DXR submission sequence number (mirrors `DxrState::sub_seq` for lock-free reserve sizing).
static CURRENT_SUB_SEQ: AtomicU64 = AtomicU64::new(0);
/// Submission sequence of the most recent static BLAS extract/upgrade/compaction activity.
static LAST_STREAM_SUB_SEQ: AtomicU64 = AtomicU64::new(0);
/// Highest `Host::dxr_fence` value submitted on `Host::compute_queue` (for Spec-Trap #16 retirement).
static SUBMITTED_DXR_FENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_submitted_dxr_fence(val: u64) {
    if val > 0 {
        SUBMITTED_DXR_FENCE.fetch_max(val, Ordering::AcqRel);
    }
}

fn retire_dxr_fence_target() -> u64 {
    if host().is_some_and(Host::dxr_async_compute) {
        SUBMITTED_DXR_FENCE.load(Ordering::Acquire).saturating_add(2)
    } else {
        0
    }
}

/// RAII wrapper tracking a DEFAULT-heap auxiliary DXR buffer in `DXR_AUX_VRAM_BYTES`.
struct AuxVramBuf {
    buf: ID3D12Resource,
    size: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded.
unsafe impl Send for AuxVramBuf {}

impl AuxVramBuf {
    fn new(buf: ID3D12Resource, size: u64) -> Self {
        DXR_AUX_VRAM_BYTES.fetch_add(size, Ordering::Relaxed);
        Self { buf, size }
    }
}

impl Drop for AuxVramBuf {
    fn drop(&mut self) {
        let _ = DXR_AUX_VRAM_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_sub(self.size))
        });
    }
}

fn note_static_streaming(sub_seq: u64) {
    LAST_STREAM_SUB_SEQ.fetch_max(sub_seq.max(1), Ordering::Relaxed);
}

fn dxr_room_kept_for(
    enabled: bool,
    override_mb: Option<u64>,
    slab_bytes: u64,
    aux_bytes: u64,
    streaming: bool,
) -> u64 {
    if !enabled {
        return 0;
    }
    if let Some(mb) = override_mb {
        return (mb << 20).saturating_sub(slab_bytes.saturating_add(aux_bytes));
    }
    let base = if streaming {
        DXR_STREAMING_RESERVE_BYTES
    } else {
        DXR_STEADY_RESERVE_BYTES
    };
    base.saturating_sub(aux_bytes).max(AS_SLAB_SIZE)
}

/// Dynamic Local VRAM headroom reserve for Native DXR (`128 MiB` steady / `256 MiB`
/// streaming, accounting for `DXR_AUX_VRAM_BYTES` and floored at 1 slab = `32 MiB`,
/// or `VICTORY_CARD_ROOM_FOR_DXR_MB` when explicitly overridden; `ARCH-10`).
pub(crate) fn dxr_room_kept() -> u64 {
    if !enabled() {
        return 0;
    }
    static OVERRIDE_MB: OnceLock<Option<u64>> = OnceLock::new();
    let override_mb = *OVERRIDE_MB.get_or_init(|| {
        std::env::var("VICTORY_CARD_ROOM_FOR_DXR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
    });
    let slab_bytes = DXR_SLAB_BYTES.load(Ordering::Relaxed);
    let aux_bytes = DXR_AUX_VRAM_BYTES.load(Ordering::Relaxed);
    let cur_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
    let last_stream = LAST_STREAM_SUB_SEQ.load(Ordering::Relaxed);
    let streaming = slab_bytes == 0
        || (last_stream > 0 && cur_sub.saturating_sub(last_stream) <= DXR_STREAMING_WINDOW_SUBS);
    dxr_room_kept_for(true, override_mb, slab_bytes, aux_bytes, streaming)
}

impl TieredSlabAllocator<ID3D12Resource> {
    fn alloc(&mut self, host: &Host, tier: SlabTier, size: u64) -> Option<(ID3D12Resource, SlabSlice, u64)> {
        let label = match tier {
            SlabTier::StaticCompacted => "static",
            SlabTier::TransientDynamicTlas => "transient",
        };
        self.tier_mut(tier).alloc_with(size, |slab_size, slab_idx| {
            let buffer = create_as_slab(host, slab_size, slab_idx, label)?;
            DXR_SLAB_BYTES.fetch_add(slab_size, Ordering::Relaxed);
            // SAFETY: live committed buffer resource.
            let gpu_base = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, gpu_base))
        })
    }

    fn reclaim_empty_slabs(&mut self) -> u64 {
        let mut freed = 0u64;
        for (buf, sz) in self
            .static_compacted
            .reclaim_empty_slabs()
            .into_iter()
            .chain(self.transient_dynamic_tlas.reclaim_empty_slabs())
        {
            let _ = DXR_SLAB_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(cur.saturating_sub(sz))
            });
            crate::host::retire_vram(buf, sz);
            freed = freed.saturating_add(sz);
        }
        freed
    }
}

fn create_as_slab(host: &Host, size: u64, index: usize, tier: &str) -> Option<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
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
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain committed AS slab creation in the required initial state.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            not_resident(),
            &desc,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            None,
            &mut buffer,
        )
    };
    let made = made
        .and_then(|()| buffer.as_ref().map_or(Ok(()), |b| resident(&host.device, b)))
        .and_then(|()| mapped(&host.device));
    match made {
        Ok(()) => {
            let buf = buffer?;
            name(&buf, || format!("victory dxr as slab {tier} {index} ({} MiB)", size >> 20));
            Some(buf)
        }
        Err(e) => {
            tracing::warn!(error = %e, size, "dxr: failed to allocate AS slab");
            None
        }
    }
}

struct StaticReadbackRecord {
    guest_key: u64,
    generation: u64,
    offset: u64,
    len: u64,
}

struct CompactQueryRecord {
    guest_key: u64,
    generation: u64,
    uncompacted_size: u64,
    slot: u32,
    /// When `true`, `finish_submission` queues async compaction into `pending_compactions`;
    /// when `false` (guest dynamic `BuildBlas` with `ALLOW_COMPACTION`), records `compacted_size`
    /// in `AS_MAP` so a subsequent guest `CopyAs(COMPACT)` sizes its destination accurately.
    auto_compact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UpgradedGeom {
    vert_offset: u32,
    vert_count: u32,
    opaque: bool,
}

#[derive(Debug)]
struct PendingStaticUpgrade {
    guest_key: u64,
    generation: u64,
    geometries: Vec<UpgradedGeom>,
    vertices: Vec<[f32; 3]>,
    /// Per-geometry relative primitive offset table precomputed on `victory-dxr-decode` (`ARCH-14`).
    geom_table: Vec<u32>,
    /// Packed octahedral geometric triangle normals precomputed on `victory-dxr-decode` (`ARCH-14`).
    normals_data: Vec<u32>,
}

/// Async-compute DXR command-list batch scheduled alongside a `Host::submit` call.
pub(crate) struct DxrAsyncBatch {
    /// Index into `runs` after which these compute lists execute (via `skin_fence`),
    /// or `None` for pre-run work (`empty_tlas_init`, `requeued_readbacks`, `static_upgrades`,
    /// `static_compactions`) executed ahead of `runs[0]`.
    pub(crate) after_run: Option<usize>,
    /// Closed `D3D12_COMMAND_LIST_TYPE_COMPUTE` lists to run on `Host::compute_queue`.
    pub(crate) lists: Vec<Option<ID3D12CommandList>>,
    /// Whether `DirectQ` must wait on `dxr_fence` before subsequent Direct shaders.
    pub(crate) blocks_direct_shaders: bool,
    /// Whether any op in this batch wrote a guest postbuild buffer read back by `card.check`.
    pub(crate) wrote_guest_postbuild: bool,
}

// SAFETY: D3D12 command lists are free-threaded; each batch is consumed once on `victory-submit`.
unsafe impl Send for DxrAsyncBatch {}

/// Host resources and pooled command lists owned by one in-flight submission.
pub(crate) struct DxrSubmission {
    lists: Vec<Prelude>,
    /// Pending async-compute DXR batches taken by `Host::submit` right before execution.
    async_batches: Vec<DxrAsyncBatch>,
    /// `Host::dxr_fence` value signalled on `compute_queue` for this submission (0 if Direct).
    dxr_fence_val: u64,
    /// Whether `lists` were recorded as `COMPUTE` lists on `Host::compute_queue`.
    ran_on_compute: bool,
    scratch: Option<AuxVramBuf>,
    host_instances: Option<AuxVramBuf>,
    extract_verts: Option<(AuxVramBuf, u64)>,
    hash_upload: Option<(ID3D12Resource, usize)>,
    /// Temporary `DEFAULT` UAV buffer for unpacked 64-byte TriangleNode dumps + last-used sub_seq.
    node_staging: Option<(AuxVramBuf, u64)>,
    /// `READBACK` buffer receiving packed per-BLAS node dumps for this submission + last-used sub_seq.
    node_readback: Option<(ID3D12Resource, u64, u64)>,
    readback_used: u64,
    readback_cap: u64,
    readbacks: Vec<StaticReadbackRecord>,
    /// Static BVH4 `(dest_va, size_bytes)` ranges whose extraction + readback completes with this
    /// submission so `finish_submission` can release their ephemeral `memory.rs` arena pages.
    extracted_bvh4s: Vec<(u64, u64)>,
    /// Pooled upload + default vertex buffers for draining `pending_static_upgrades` + last-used sub_seq.
    upgrade_upload: Option<(ID3D12Resource, usize, u64, u64)>,
    upgrade_verts: Option<(AuxVramBuf, u64)>,
    /// Pooled upload buffer for static BLAS octahedral normal uploads + word-0 fallback init + last-used sub_seq.
    normals_upload: Option<(ID3D12Resource, usize, u64, u64)>,
    /// Pooled 64 KiB `DEFAULT` UAV and `READBACK` buffers for 8-byte compacted-size queries + last-used sub_seq.
    compact_info_uav: Option<(AuxVramBuf, u64)>,
    compact_info_rb: Option<(ID3D12Resource, u64, u64)>,
    compact_queries: Vec<CompactQueryRecord>,
    kept: Vec<ID3D12Resource>,
    /// Keeps full `Shadow` objects (including any backing `ID3D12Heap` for placed
    /// in-place buffers) alive until this submission's GPU fence completes.
    kept_shadows: Vec<Arc<crate::shadow::Shadow>>,
    /// Monotonic DXR submission sequence number assigned when submitted (0 if unused).
    sub_seq: u64,
    /// Monotonic `tlas_build_epoch` achieved by this submission (0 if no `BuildTlas` ran).
    tlas_epoch: u64,
    /// Optional GPU timestamp heap + readback buffer for `VICTORY_GPU_TIMING=1`.
    #[cfg(feature = "diagnostics")]
    timing_queries: Option<(ID3D12QueryHeap, ID3D12Resource)>,
    #[cfg(feature = "diagnostics")]
    timing_spans: Vec<(&'static str, u32, u32)>,
    #[cfg(feature = "diagnostics")]
    timing_next: u32,
}

// SAFETY: D3D12 objects are free-threaded; a `DxrSubmission` has one owner at a time.
unsafe impl Send for DxrSubmission {}

impl DxrSubmission {
    fn new() -> Self {
        Self {
            lists: Vec::new(),
            async_batches: Vec::new(),
            dxr_fence_val: 0,
            ran_on_compute: false,
            scratch: None,
            host_instances: None,
            extract_verts: None,
            hash_upload: None,
            node_staging: None,
            node_readback: None,
            readback_used: 0,
            readback_cap: 0,
            readbacks: Vec::new(),
            extracted_bvh4s: Vec::new(),
            upgrade_upload: None,
            upgrade_verts: None,
            normals_upload: None,
            compact_info_uav: None,
            compact_info_rb: None,
            compact_queries: Vec::new(),
            kept: Vec::new(),
            kept_shadows: Vec::new(),
            sub_seq: 0,
            tlas_epoch: 0,
            #[cfg(feature = "diagnostics")]
            timing_queries: None,
            #[cfg(feature = "diagnostics")]
            timing_spans: Vec::new(),
            #[cfg(feature = "diagnostics")]
            timing_next: 0,
        }
    }

    pub(crate) fn take_async_batches(&mut self) -> Vec<DxrAsyncBatch> {
        std::mem::take(&mut self.async_batches)
    }

    pub(crate) fn set_dxr_fence_val(&mut self, val: u64) {
        self.dxr_fence_val = self.dxr_fence_val.max(val);
    }

    #[cfg(feature = "diagnostics")]
    fn ensure_timing(&mut self, host: &Host) -> Option<(ID3D12QueryHeap, ID3D12Resource)> {
        if !crate::host::timing() {
            return None;
        }
        if self.timing_queries.is_none() {
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let readback = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(TIMING_SLOTS) * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            // SAFETY: plain query heap + readback buffer creation.
            unsafe {
                host.device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: TIMING_SLOTS,
                            NodeMask: 0,
                        },
                        &mut heap,
                    )
                    .ok()?;
                host.device
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
            self.timing_queries = Some((heap?, back?));
        }
        self.timing_queries.clone()
    }

    #[cfg(feature = "diagnostics")]
    unsafe fn stamp(&mut self, list: &ID3D12GraphicsCommandList, heap: &ID3D12QueryHeap) -> Option<u32> {
        if self.timing_next >= TIMING_SLOTS {
            return None;
        }
        let slot = self.timing_next;
        self.timing_next += 1;
        // SAFETY: open command list and valid timestamp query heap.
        unsafe { list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, slot) };
        Some(slot)
    }

    #[cfg(feature = "diagnostics")]
    unsafe fn resolve_timing(
        &self,
        list: &ID3D12GraphicsCommandList,
        heap: &ID3D12QueryHeap,
        rb: &ID3D12Resource,
        base: u32,
    ) {
        let count = self.timing_next.saturating_sub(base);
        if count > 0 {
            // SAFETY: open command list, valid query heap, and readback buffer with `TIMING_SLOTS * 8` bytes.
            unsafe {
                list.ResolveQueryData(
                    heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    base,
                    count,
                    rb,
                    u64::from(base) * 8,
                );
            }
        }
    }

    #[cfg(feature = "diagnostics")]
    fn report_timing(&mut self) {
        let spans = std::mem::take(&mut self.timing_spans);
        let count = self.timing_next as usize;
        self.timing_next = 0;
        if spans.is_empty() || count == 0 {
            return;
        }
        let Some((_, rb)) = self.timing_queries.as_ref() else {
            return;
        };
        let Some(h) = host() else {
            return;
        };
        let frequency = if self.ran_on_compute {
            h.compute_timestamp_frequency()
        } else {
            h.timestamp_frequency()
        }
        .unwrap_or(1)
        .max(1);
        let mut ptr = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: count * 8 };
        // SAFETY: submission GPU fence has completed; readback buffer is idle.
        let mapped = unsafe { rb.Map(0, Some(&range), Some(&mut ptr)) };
        if mapped.is_ok() && !ptr.is_null() {
            // SAFETY: `ptr` is valid for `count` u64 timestamps until `Unmap`.
            let stamps = unsafe { std::slice::from_raw_parts(ptr.cast::<u64>(), count) };
            for (label, start, end) in spans {
                let (s, e) = (start as usize, end as usize);
                if e < stamps.len() && s < e {
                    let ticks = stamps[e].saturating_sub(stamps[s]);
                    let ms = ticks as f64 * 1e3 / frequency as f64;
                    crate::copystat::command_took(label, ms);
                }
            }
            let written_none = D3D12_RANGE { Begin: 0, End: 0 };
            // SAFETY: matches successful `Map(0, ...)`.
            unsafe { rb.Unmap(0, Some(&written_none)) };
        }
    }

    fn ensure_scratch(&mut self, host: &Host, need: u64) -> Option<(&ID3D12Resource, u64, u64)> {
        let want = need.max(MIN_SCRATCH_SIZE).next_multiple_of(AS_ALIGN);
        if self.scratch.as_ref().is_none_or(|b| b.size < want) {
            if let Some(old) = self.scratch.take() {
                self.kept.push(old.buf.clone());
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr scratch ({} MiB)", want >> 20));
            self.scratch = Some(AuxVramBuf::new(buf, want));
        }
        let aux = self.scratch.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { aux.buf.GetGPUVirtualAddress() };
        Some((&aux.buf, gpu, aux.size))
    }

    fn ensure_host_instances(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.host_instances.as_ref().is_none_or(|b| b.size < want) {
            if let Some(old) = self.host_instances.take() {
                self.kept.push(old.buf.clone());
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr host instances ({} KiB)", want >> 10));
            self.host_instances = Some(AuxVramBuf::new(buf, want));
        }
        let aux = self.host_instances.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { aux.buf.GetGPUVirtualAddress() };
        Some((aux.buf.clone(), gpu))
    }

    fn ensure_extract_verts(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.max(256 * 1024).next_multiple_of(AS_ALIGN);
        if self.extract_verts.as_ref().is_none_or(|(b, _)| b.size < want) {
            if let Some((old, _)) = self.extract_verts.take() {
                self.kept.push(old.buf.clone());
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr extracted verts ({} KiB)", want >> 10));
            self.extract_verts = Some((AuxVramBuf::new(buf, want), now_sub));
        } else if let Some((_, last)) = self.extract_verts.as_mut() {
            *last = now_sub;
        }
        let (aux, _) = self.extract_verts.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { aux.buf.GetGPUVirtualAddress() };
        Some((aux.buf.clone(), gpu))
    }

    fn ensure_hash_upload(&mut self, host: &Host) -> Option<(ID3D12Resource, *mut u8)> {
        if self.hash_upload.is_none() {
            let (buf, ptr) = host.upload_buffer(HASH_TABLE_BYTES)?;
            name(&buf, || "victory dxr blas hash upload".into());
            self.hash_upload = Some((buf, ptr as usize));
        }
        let (buf, ptr) = self.hash_upload.as_ref()?;
        Some((buf.clone(), *ptr as *mut u8))
    }

    fn ensure_node_staging(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.node_staging.as_ref().is_none_or(|(b, _)| b.size < want) {
            if let Some((old, _)) = self.node_staging.take() {
                self.kept.push(old.buf.clone());
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr node staging ({} KiB)", want >> 10));
            self.node_staging = Some((AuxVramBuf::new(buf, want), now_sub));
        } else if let Some((_, last)) = self.node_staging.as_mut() {
            *last = now_sub;
        }
        let (aux, _) = self.node_staging.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { aux.buf.GetGPUVirtualAddress() };
        Some((aux.buf.clone(), gpu))
    }

    fn prepare_node_readback(&mut self, host: &Host, need: u64) {
        self.readbacks.clear();
        self.extracted_bvh4s.clear();
        self.readback_used = 0;
        self.readback_cap = 0;
        if need == 0 {
            return;
        }
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.min(MAX_READBACK_BYTES_PER_SUB).max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.node_readback.as_ref().is_none_or(|(_, sz, _)| *sz < want) {
            if let Some((old, _, _)) = self.node_readback.take() {
                self.kept.push(old);
            }
            if let Some(buf) = host.read_back_buffer(want) {
                name(&buf, || format!("victory dxr node readback ({} KiB)", want >> 10));
                self.node_readback = Some((buf, want, now_sub));
            }
        } else if let Some((_, _, last)) = self.node_readback.as_mut() {
            *last = now_sub;
        }
        if let Some((_, sz, _)) = self.node_readback.as_ref() {
            self.readback_cap = (*sz).min(MAX_READBACK_BYTES_PER_SUB);
        }
    }

    fn ensure_upgrade_upload(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.upgrade_upload.as_ref().is_none_or(|(_, _, sz, _)| *sz < want) {
            if let Some((old, _, _, _)) = self.upgrade_upload.take() {
                self.kept.push(old);
            }
            let (buf, ptr) = host.upload_buffer(want)?;
            name(&buf, || format!("victory dxr upgrade upload ({} KiB)", want >> 10));
            self.upgrade_upload = Some((buf, ptr as usize, want, now_sub));
        } else if let Some((_, _, _, last)) = self.upgrade_upload.as_mut() {
            *last = now_sub;
        }
        let (buf, ptr, _, _) = self.upgrade_upload.as_ref()?;
        Some((buf.clone(), *ptr as *mut u8))
    }

    fn ensure_normals_upload(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.normals_upload.as_ref().is_none_or(|(_, _, sz, _)| *sz < want) {
            if let Some((old, _, _, _)) = self.normals_upload.take() {
                self.kept.push(old);
            }
            let (buf, ptr) = host.upload_buffer(want)?;
            name(&buf, || format!("victory dxr normals upload ({} KiB)", want >> 10));
            self.normals_upload = Some((buf, ptr as usize, want, now_sub));
        } else if let Some((_, _, _, last)) = self.normals_upload.as_mut() {
            *last = now_sub;
        }
        let (buf, ptr, _, _) = self.normals_upload.as_ref()?;
        Some((buf.clone(), *ptr as *mut u8))
    }

    fn ensure_upgrade_verts(&mut self, host: &Host, need: u64) -> Option<(ID3D12Resource, u64)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        let want = need.max(64 * 1024).next_multiple_of(AS_ALIGN);
        if self.upgrade_verts.as_ref().is_none_or(|(b, _)| b.size < want) {
            if let Some((old, _)) = self.upgrade_verts.take() {
                self.kept.push(old.buf.clone());
            }
            let buf = host.uav_buffer(want)?;
            name(&buf, || format!("victory dxr upgrade verts ({} KiB)", want >> 10));
            self.upgrade_verts = Some((AuxVramBuf::new(buf, want), now_sub));
        } else if let Some((_, last)) = self.upgrade_verts.as_mut() {
            *last = now_sub;
        }
        let (aux, _) = self.upgrade_verts.as_ref()?;
        // SAFETY: live buffer.
        let gpu = unsafe { aux.buf.GetGPUVirtualAddress() };
        Some((aux.buf.clone(), gpu))
    }

    fn ensure_compact_info(&mut self, host: &Host) -> Option<(ID3D12Resource, u64, ID3D12Resource)> {
        let now_sub = CURRENT_SUB_SEQ.load(Ordering::Relaxed);
        if self.compact_info_uav.is_none() {
            let buf = host.uav_buffer(COMPACT_INFO_BYTES)?;
            name(&buf, || "victory dxr compact info uav".into());
            self.compact_info_uav = Some((AuxVramBuf::new(buf, COMPACT_INFO_BYTES), now_sub));
        } else if let Some((_, last)) = self.compact_info_uav.as_mut() {
            *last = now_sub;
        }
        if self.compact_info_rb.is_none() {
            let buf = host.read_back_buffer(COMPACT_INFO_BYTES)?;
            name(&buf, || "victory dxr compact info readback".into());
            self.compact_info_rb = Some((buf, COMPACT_INFO_BYTES, now_sub));
        } else if let Some((_, _, last)) = self.compact_info_rb.as_mut() {
            *last = now_sub;
        }
        let (uav, _) = self.compact_info_uav.as_ref()?;
        let (rb, _, _) = self.compact_info_rb.as_ref()?;
        // SAFETY: live UAV buffer.
        let uav_gpu = unsafe { uav.buf.GetGPUVirtualAddress() };
        Some((uav.buf.clone(), uav_gpu, rb.clone()))
    }

    /// Drops oversized buffers and static staging buffers idle for `>= IDLE_STAGING_TRIM_SUBS`
    /// submissions asynchronously via `crate::host::retire_com_async` (`ARCH-10`).
    fn trim_pooled(&mut self, completed_sub_seq: u64) {
        let mut drop_aux: Vec<AuxVramBuf> = Vec::new();
        let mut drop_res: Vec<ID3D12Resource> = std::mem::take(&mut self.kept);
        let is_idle = |last_used: u64| {
            completed_sub_seq.saturating_sub(last_used) >= IDLE_STAGING_TRIM_SUBS
        };

        if self.scratch.as_ref().is_some_and(|b| b.size > MAX_POOLED_SCRATCH_BYTES)
            && let Some(old) = self.scratch.take()
        {
            drop_aux.push(old);
        }
        if self
            .host_instances
            .as_ref()
            .is_some_and(|b| b.size > MAX_POOLED_HOST_INSTANCES_BYTES)
            && let Some(old) = self.host_instances.take()
        {
            drop_aux.push(old);
        }
        if self
            .extract_verts
            .as_ref()
            .is_some_and(|(b, last)| b.size > MAX_POOLED_EXTRACT_VERTS_BYTES || is_idle(*last))
            && let Some((old, _)) = self.extract_verts.take()
        {
            drop_aux.push(old);
        }
        if self
            .node_staging
            .as_ref()
            .is_some_and(|(b, last)| b.size > MAX_POOLED_AUX_BYTES || is_idle(*last))
            && let Some((old, _)) = self.node_staging.take()
        {
            drop_aux.push(old);
        }
        if self
            .node_readback
            .as_ref()
            .is_some_and(|(_, sz, last)| *sz > MAX_POOLED_AUX_BYTES || is_idle(*last))
            && let Some((old, _, _)) = self.node_readback.take()
        {
            drop_res.push(old);
        }
        if self
            .upgrade_upload
            .as_ref()
            .is_some_and(|(_, _, sz, last)| *sz > MAX_POOLED_AUX_BYTES || is_idle(*last))
            && let Some((old, _, _, _)) = self.upgrade_upload.take()
        {
            drop_res.push(old);
        }
        if self
            .upgrade_verts
            .as_ref()
            .is_some_and(|(b, last)| b.size > MAX_POOLED_AUX_BYTES || is_idle(*last))
            && let Some((old, _)) = self.upgrade_verts.take()
        {
            drop_aux.push(old);
        }
        if self
            .normals_upload
            .as_ref()
            .is_some_and(|(_, _, sz, last)| *sz > MAX_POOLED_AUX_BYTES || is_idle(*last))
            && let Some((old, _, _, _)) = self.normals_upload.take()
        {
            drop_res.push(old);
        }
        if self
            .compact_info_uav
            .as_ref()
            .is_some_and(|(_, last)| is_idle(*last))
            && let Some((old, _)) = self.compact_info_uav.take()
        {
            drop_aux.push(old);
        }
        if self
            .compact_info_rb
            .as_ref()
            .is_some_and(|(_, _, last)| is_idle(*last))
            && let Some((old, _, _)) = self.compact_info_rb.take()
        {
            drop_res.push(old);
        }

        if !drop_aux.is_empty() {
            crate::host::retire_com_async(drop_aux);
        }
        if !drop_res.is_empty() {
            crate::host::retire_com_async(drop_res);
        }
    }
}

/// Tracks modified slots in `DxrState::blas_table` so `record_build_tlas` copies only
/// dirty slot runs (or `[min_slot..=max_slot]` on bursts) instead of the full 1 MiB table.
struct HashDirtyTracker {
    full: bool,
    has_range: bool,
    overflowed: bool,
    min_slot: usize,
    max_slot: usize,
    slots: Vec<usize>,
}

impl HashDirtyTracker {
    fn full() -> Self {
        Self {
            full: true,
            has_range: false,
            overflowed: false,
            min_slot: 0,
            max_slot: 0,
            slots: Vec::new(),
        }
    }

    fn is_dirty(&self) -> bool {
        self.full || self.has_range
    }

    fn mark_full(&mut self) {
        self.full = true;
        self.has_range = false;
        self.overflowed = false;
        self.slots.clear();
    }

    fn mark_slot(&mut self, slot: usize) {
        if self.full {
            return;
        }
        if !self.has_range {
            self.has_range = true;
            self.min_slot = slot;
            self.max_slot = slot;
        } else {
            self.min_slot = self.min_slot.min(slot);
            self.max_slot = self.max_slot.max(slot);
        }
        if !self.overflowed {
            if self.slots.len() < MAX_DIRTY_HASH_SLOTS {
                self.slots.push(slot);
            } else {
                self.slots.clear();
                self.overflowed = true;
            }
        }
    }

    /// Returns inclusive `(start_slot, end_slot)` dirty ranges and clears the tracker.
    fn take_ranges(&mut self) -> Vec<(usize, usize)> {
        if self.full {
            self.full = false;
            self.has_range = false;
            self.overflowed = false;
            self.slots.clear();
            return vec![(0, HASH_CAPACITY - 1)];
        }
        if !self.has_range {
            return Vec::new();
        }
        self.has_range = false;
        if self.overflowed {
            self.overflowed = false;
            self.slots.clear();
            return vec![(self.min_slot, self.max_slot)];
        }
        self.slots.sort_unstable();
        self.slots.dedup();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for &slot in &self.slots {
            if let Some(last) = ranges.last_mut()
                && slot.saturating_sub(last.1) <= HASH_DIRTY_MERGE_GAP
            {
                last.1 = slot;
            } else {
                ranges.push((slot, slot));
            }
        }
        self.slots.clear();
        ranges
    }
}

struct DxrState {
    slabs: TieredSlabAllocator,
    /// Open-addressed CPU mirror of `(guest_blas_va >> 8 -> (host_blas_va, normals_geom_off))`.
    /// Each 32-byte slot is `[key_lo, key_hi, val_lo, val_hi, geom_off, 0, 0, 0]`.
    blas_table: Vec<[u32; 8]>,
    hash_dirty: HashDirtyTracker,
    hash_gpu: Option<AuxVramBuf>,
    /// Shared 64 MiB default-heap `SLOT_DXR_NORMALS` lookup table (`t251` in `ROOT_CBV_SPACE`).
    normals_gpu: Option<(AuxVramBuf, u64)>,
    normals_init_needed: bool,
    geom_pool: WordPool,
    data_pool: WordPool,
    /// Mapped shared UAV buffer `[patched_ok, missing_blas, null_blas, total]` for diagnostics.
    patch_stats: Option<(ID3D12Resource, usize, u64)>,
    /// Monotonic generation stamped on `HostAs` entries to reject stale async upgrades.
    next_gen: u64,
    /// Monotonic submission sequence numbers (submitted vs GPU-completed).
    sub_seq: u64,
    completed_sub_seq: u64,
    /// Monotonic TLAS rebuild epochs (submitted vs GPU-completed) gating BLAS slice reclamation.
    tlas_build_epoch: u64,
    completed_tlas_epoch: u64,
    /// Last `tlas_build_epoch` on which periodic unreachable-BLAS GC ran.
    last_gc_tlas_epoch: u64,
    /// Last `tlas_build_epoch` on which a CPU-readable TLAS instance array was scanned.
    last_readable_tlas_scan_epoch: u64,
    /// Last `tlas_build_epoch` on which a GPU-written (non-CPU-readable) TLAS was observed.
    last_unreadable_tlas_epoch: u64,
    /// Reusable scratch buffer for SEH-safe TLAS instance BLAS-reachability scanning.
    inst_scan_buf: Vec<u8>,
    /// Cached dynamic BLAS `GetRaytracingAccelerationStructurePrebuildInfo` keyed by `(flags, topo_hash)`
    /// -> `(ResultDataMaxSizeInBytes, ScratchDataSizeInBytes, UpdateScratchDataSizeInBytes)`.
    prebuild_cache: HashMap<(u32, u64), (u64, u64, u64)>,
    /// Cached TLAS `GetRaytracingAccelerationStructurePrebuildInfo` keyed by `(flags, num_instances)`
    /// -> `(ResultDataMaxSizeInBytes, ScratchDataSizeInBytes)`.
    tlas_prebuild_cache: HashMap<(u32, u32), (u64, u64)>,
    /// Newly pre-allocated TLAS VAs that have not been built yet; initialized with an empty
    /// 0-instance TLAS before any shader can trace them.
    pending_empty_tlas: Vec<u64>,
    /// Frame-0 static BLASes `(guest_key, size_bytes, generation)` whose node readback overflowed
    /// a burst submission and will be re-extracted cleanly on a subsequent submission (`ARCH-10`).
    pending_readback_requeue: VecDeque<(u64, u64, u64)>,
    /// Decoded multi-geometry static BLAS upgrades waiting for GPU build on the next submission.
    pending_static_upgrades: VecDeque<PendingStaticUpgrade>,
    /// Static BLASes `(guest_key, generation, compacted_size)` awaiting async compaction copy.
    pending_compactions: VecDeque<(u64, u64, u64)>,
    /// Pooled `DIRECT` and `COMPUTE` command lists and reusable submission containers returned after GPU completion.
    free_lists: Vec<Prelude>,
    free_compute_lists: Vec<Prelude>,
    free_submissions: Vec<DxrSubmission>,
}

// SAFETY: D3D12 objects are free-threaded; accessed strictly under `DXR_STATE` mutex.
unsafe impl Send for DxrState {}

impl DxrState {
    fn new() -> Self {
        Self {
            slabs: TieredSlabAllocator::default(),
            blas_table: vec![[0u32; 8]; HASH_CAPACITY],
            hash_dirty: HashDirtyTracker::full(),
            hash_gpu: None,
            normals_gpu: None,
            normals_init_needed: true,
            geom_pool: WordPool::new(
                victory_shader::emit::DXR_NORMALS_GEOM_BASE,
                victory_shader::emit::DXR_NORMALS_DATA_BASE,
            ),
            data_pool: WordPool::new(
                victory_shader::emit::DXR_NORMALS_DATA_BASE,
                victory_shader::emit::DXR_NORMALS_WORDS,
            ),
            patch_stats: None,
            next_gen: 0,
            sub_seq: 0,
            completed_sub_seq: 0,
            tlas_build_epoch: 0,
            completed_tlas_epoch: 0,
            last_gc_tlas_epoch: 0,
            last_readable_tlas_scan_epoch: 0,
            last_unreadable_tlas_epoch: 0,
            inst_scan_buf: Vec::new(),
            prebuild_cache: HashMap::new(),
            tlas_prebuild_cache: HashMap::new(),
            pending_empty_tlas: Vec::new(),
            pending_readback_requeue: VecDeque::new(),
            pending_static_upgrades: VecDeque::new(),
            pending_compactions: VecDeque::new(),
            free_lists: Vec::new(),
            free_compute_lists: Vec::new(),
            free_submissions: Vec::new(),
        }
    }

    fn bump_gen(&mut self) -> u64 {
        self.next_gen = self.next_gen.wrapping_add(1).max(1);
        self.next_gen
    }

    /// Defers reclamation of a replaced AS slab slice (and optional mirror TLAS slice)
    /// until it is guaranteed unreachable on the GPU (`tlas_build_epoch + 2 && sub_seq + 2 && dxr_fence`).
    fn retire_as(&mut self, old: &HostAs) {
        let release_after_sub_seq = self.sub_seq.saturating_add(2);
        let release_after_tlas_epoch = self.tlas_build_epoch.saturating_add(2);
        let release_after_dxr_fence = retire_dxr_fence_target();
        self.slabs.retire(
            old.slice,
            release_after_tlas_epoch,
            release_after_sub_seq,
            release_after_dxr_fence,
        );
        if let Some(mirror) = &old.mirror_tlas {
            self.slabs.retire(
                mirror.slice,
                release_after_tlas_epoch,
                release_after_sub_seq,
                release_after_dxr_fence,
            );
        }
        if old.normals_geom_len > 0 {
            self.geom_pool.retire(
                old.normals_geom_off,
                old.normals_geom_len,
                release_after_tlas_epoch,
                release_after_sub_seq,
                release_after_dxr_fence,
            );
        }
        if old.normals_data_len > 0 {
            self.data_pool.retire(
                old.normals_data_off,
                old.normals_data_len,
                release_after_tlas_epoch,
                release_after_sub_seq,
                release_after_dxr_fence,
            );
        }
    }

    fn insert_blas_hash(&mut self, guest_va: u64, host_va: u64, normals_geom_off: u32) {
        insert_blas_hash_in(
            &mut self.blas_table,
            &mut self.hash_dirty,
            guest_va,
            host_va,
            normals_geom_off,
        );
    }

    /// Shift-deletes `guest_va` from the open-addressed 64-probe `blas_table` chain,
    /// preserving all subsequent probe chains without leaving empty-slot breaks (`ARCH-10`).
    fn remove_blas_hash(&mut self, guest_va: u64) -> bool {
        shift_delete_blas_hash(&mut self.blas_table, &mut self.hash_dirty, guest_va)
    }

    fn ensure_normals_gpu(&mut self, host: &Host) -> Option<(ID3D12Resource, u64)> {
        if self.normals_gpu.is_none() {
            let bytes = u64::from(victory_shader::emit::DXR_NORMALS_WORDS) * 4;
            let buf = host.uav_buffer(bytes)?;
            // SAFETY: live default-heap UAV buffer.
            let gpu = unsafe { buf.GetGPUVirtualAddress() };
            name(&buf, || "victory dxr normals table (64 MiB)".into());
            NORMALS_GPU_VA.store(gpu, Ordering::Release);
            self.normals_gpu = Some((AuxVramBuf::new(buf, bytes), gpu));
            self.normals_init_needed = true;
        }
        let (aux, gpu) = self.normals_gpu.as_ref()?;
        Some((aux.buf.clone(), *gpu))
    }

    fn ensure_hash_gpu(&mut self, host: &Host) -> Option<ID3D12Resource> {
        if self.hash_gpu.is_none() {
            let buf = host.uav_buffer(HASH_TABLE_BYTES)?;
            name(&buf, || "victory dxr blas hash table".into());
            self.hash_gpu = Some(AuxVramBuf::new(buf, HASH_TABLE_BYTES));
            self.hash_dirty.mark_full();
        }
        self.hash_gpu.as_ref().map(|a| a.buf.clone())
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn ensure_patch_stats(&mut self, host: &Host) -> Option<(ID3D12Resource, u64)> {
        if self.patch_stats.is_none() {
            let (buf, ptr) = host.shared_buffer(64)?;
            // SAFETY: fresh mapped 64-byte buffer.
            unsafe { std::ptr::write_bytes(ptr, 0, 64) };
            // SAFETY: live buffer.
            let gpu = unsafe { buf.GetGPUVirtualAddress() };
            name(&buf, || "victory dxr patch stats".into());
            self.patch_stats = Some((buf, ptr as usize, gpu));
        }
        let (buf, _, gpu) = self.patch_stats.as_ref()?;
        Some((buf.clone(), *gpu))
    }

    fn take_list(&mut self, host: &Host, compute: bool) -> Option<Prelude> {
        if compute {
            self.free_compute_lists
                .pop()
                .or_else(|| host.make_compute_prelude())
        } else {
            self.free_lists.pop().or_else(|| host.make_prelude())
        }
    }

    fn return_list(&mut self, prelude: Prelude, compute: bool) {
        let pool = if compute {
            &mut self.free_compute_lists
        } else {
            &mut self.free_lists
        };
        if pool.len() < MAX_FREE_LISTS {
            pool.push(prelude);
        } else {
            crate::host::retire_com_async(prelude);
        }
    }

    fn take_submission(&mut self) -> DxrSubmission {
        self.free_submissions.pop().unwrap_or_else(DxrSubmission::new)
    }

    fn tlas_prebuild_info(
        &mut self,
        device5: &ID3D12Device5,
        flags: u32,
        num_instances: u32,
    ) -> (u64, u64) {
        cached_tlas_prebuild_with(&mut self.tlas_prebuild_cache, flags, num_instances, || {
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(flags as i32),
                NumDescs: num_instances,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: 0,
                },
            };
            let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
            // SAFETY: valid device5 and TLAS prebuild inputs.
            unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
            (pre.ResultDataMaxSizeInBytes, pre.ScratchDataSizeInBytes)
        })
    }
}

fn cached_tlas_prebuild_with<F>(
    cache: &mut HashMap<(u32, u32), (u64, u64)>,
    flags: u32,
    num_instances: u32,
    query: F,
) -> (u64, u64)
where
    F: FnOnce() -> (u64, u64),
{
    if let Some(&cached) = cache.get(&(flags, num_instances)) {
        return cached;
    }
    let entry = query();
    if entry.0 > 0 {
        if cache.len() >= MAX_TLAS_PREBUILD_CACHE {
            cache.clear();
        }
        cache.insert((flags, num_instances), entry);
    }
    entry
}

fn hash_key(lo: u32, hi: u32) -> u32 {
    let mut x = lo ^ hi.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2_AE35);
    x ^= x >> 16;
    x
}

fn insert_blas_hash_in(
    blas_table: &mut [[u32; 8]],
    hash_dirty: &mut HashDirtyTracker,
    guest_va: u64,
    host_va: u64,
    normals_geom_off: u32,
) {
    let key = guest_va >> 8;
    if key == 0 {
        return;
    }
    let key_lo = key as u32;
    let key_hi = (key >> 32) as u32;
    let val_lo = host_va as u32;
    let val_hi = (host_va >> 32) as u32;
    let new_entry = [key_lo, key_hi, val_lo, val_hi, normals_geom_off, 0, 0, 0];
    let start = hash_key(key_lo, key_hi) & HASH_MASK;
    for step in 0..64u32 {
        let idx = ((start + step) & HASH_MASK) as usize;
        let entry = &mut blas_table[idx];
        if (entry[0] == key_lo && entry[1] == key_hi) || (entry[0] | entry[1]) == 0 {
            if *entry != new_entry {
                *entry = new_entry;
                hash_dirty.mark_slot(idx);
            }
            return;
        }
    }
    // shortcut: hash table saturated past 64-probe chain; overwrite initial bucket,
    // upgrade to dynamic resize if active BLASes ever approach 48K.
    let idx = start as usize;
    blas_table[idx] = new_entry;
    hash_dirty.mark_slot(idx);
}

/// Backward shift-deletion within the open-addressed 64-probe `blas_table` chain (`ARCH-10`).
/// Never blind-zeros a slot mid-chain: shifts every subsequent entry whose natural home bucket
/// precedes or equals the hole back toward its home so linear-probe lookups stay 100% intact.
fn shift_delete_blas_hash(
    blas_table: &mut [[u32; 8]],
    hash_dirty: &mut HashDirtyTracker,
    guest_va: u64,
) -> bool {
    let key = guest_va >> 8;
    if key == 0 {
        return false;
    }
    let key_lo = key as u32;
    let key_hi = (key >> 32) as u32;
    let mask = HASH_MASK as usize;
    let start = (hash_key(key_lo, key_hi) as usize) & mask;
    let mut hole = None;
    for step in 0..64usize {
        let idx = (start + step) & mask;
        let entry = blas_table[idx];
        if entry[0] == key_lo && entry[1] == key_hi {
            hole = Some(idx);
            break;
        }
        if (entry[0] | entry[1]) == 0 {
            return false;
        }
    }
    let Some(mut hole) = hole else {
        return false;
    };
    let mut cur = (hole + 1) & mask;
    for _ in 0..HASH_CAPACITY {
        let entry = blas_table[cur];
        if (entry[0] | entry[1]) == 0 {
            break;
        }
        let home = (hash_key(entry[0], entry[1]) as usize) & mask;
        let dist_cur = cur.wrapping_sub(home) & mask;
        let dist_hole = hole.wrapping_sub(home) & mask;
        if dist_hole < dist_cur {
            blas_table[hole] = entry;
            hash_dirty.mark_slot(hole);
            hole = cur;
        }
        cur = (cur + 1) & mask;
    }
    blas_table[hole] = [0u32; 8];
    hash_dirty.mark_slot(hole);
    true
}

#[cfg(test)]
fn lookup_blas_hash_in(blas_table: &[[u32; 8]], guest_va: u64) -> Option<(u64, u32)> {
    let key = guest_va >> 8;
    if key == 0 {
        return None;
    }
    let key_lo = key as u32;
    let key_hi = (key >> 32) as u32;
    let start = hash_key(key_lo, key_hi) & HASH_MASK;
    for step in 0..64u32 {
        let idx = ((start + step) & HASH_MASK) as usize;
        let e = blas_table[idx];
        if e[0] == key_lo && e[1] == key_hi {
            let host_va = u64::from(e[2]) | (u64::from(e[3]) << 32);
            return Some((host_va, e[4]));
        }
        if (e[0] | e[1]) == 0 {
            return None;
        }
    }
    None
}

/// Evicts any resident host AS entries whose guest VA overlaps `[start, start + len)`
/// (hook for `shadow::unregister` / `inplace` unmaps; `ARCH-10`).
/// Shift-deletes evicted BLASes from `blas_table` and retires AS slices + `dxr_normals`
/// words strictly behind `tlas_build_epoch + 2 && sub_seq + 2`.
#[allow(dead_code)]
pub(crate) fn evict_range(start: u64, len: u64) {
    if len == 0 {
        return;
    }
    let Some(map_lock) = AS_MAP.get() else {
        return;
    };
    let lo = start & !0xFF;
    let hi = start.saturating_add(len);
    // Fast read-only pre-check to avoid taking `DXR_STATE` or write locks when no AS overlaps.
    let has_overlap = {
        let map = map_lock.read().unwrap_or_else(|e| e.into_inner());
        map.keys().any(|&k| k != EMPTY_TLAS_KEY && k >= lo && k < hi)
    };
    if !has_overlap {
        return;
    }
    let mut state_guard = lock_state();
    let Some(state) = state_guard.as_mut() else {
        return;
    };
    let mut map = map_lock.write().unwrap_or_else(|e| e.into_inner());
    let dead_keys: Vec<u64> = map
        .keys()
        .copied()
        .filter(|&k| k != EMPTY_TLAS_KEY && k >= lo && k < hi)
        .collect();
    for key in dead_keys {
        if let Some(old) = map.remove(&key) {
            if old.kind == AsKind::Blas {
                state.remove_blas_hash(key);
            }
            state.retire_as(&old);
        }
    }
}

/// Periodic TLAS-reachability liveness sweep: evicts BLASes unreachable from any scanned
/// TLAS for more than [`BLAS_GC_UNREACHABLE_EPOCHS`] (`300`) TLAS epochs, shift-deleting
/// them from `blas_table` and retiring AS + `dxr_normals` slices behind `tlas_build_epoch + 2 && sub_seq + 2`.
fn sweep_unreachable_blases(state: &mut DxrState) -> usize {
    let epoch = state.tlas_build_epoch;
    if epoch <= BLAS_GC_UNREACHABLE_EPOCHS
        || epoch.saturating_sub(state.last_gc_tlas_epoch) < BLAS_GC_SWEEP_INTERVAL_EPOCHS
    {
        return 0;
    }
    // Require a recent CPU-readable TLAS scan and no recent unreadable GPU-written TLAS so
    // we never falsely evict a live BLAS referenced only by a GPU-written TLAS.
    if state.last_readable_tlas_scan_epoch == 0
        || epoch.saturating_sub(state.last_readable_tlas_scan_epoch) >= BLAS_GC_UNREACHABLE_EPOCHS
        || (state.last_unreadable_tlas_epoch > 0
            && epoch.saturating_sub(state.last_unreadable_tlas_epoch) <= BLAS_GC_UNREACHABLE_EPOCHS)
    {
        return 0;
    }
    state.last_gc_tlas_epoch = epoch;
    let cutoff = epoch.saturating_sub(BLAS_GC_UNREACHABLE_EPOCHS);
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    let dead_keys: Vec<u64> = map
        .iter()
        .filter_map(|(&k, e)| {
            (e.kind == AsKind::Blas
                && e.built
                && e.last_seen_tlas_epoch.load(Ordering::Relaxed) < cutoff)
                .then_some(k)
        })
        .collect();
    let evicted = dead_keys.len();
    for key in dead_keys {
        if let Some(old) = map.remove(&key) {
            state.remove_blas_hash(key);
            state.retire_as(&old);
        }
    }
    evicted
}

static AS_MAP: OnceLock<RwLock<HashMap<u64, HostAs>>> = OnceLock::new();
static DXR_STATE: Mutex<Option<DxrState>> = Mutex::new(None);

fn as_map() -> &'static RwLock<HashMap<u64, HostAs>> {
    AS_MAP.get_or_init(|| RwLock::new(HashMap::new()))
}

fn lock_state() -> std::sync::MutexGuard<'static, Option<DxrState>> {
    let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(DxrState::new());
    }
    guard
}

/// Diagnostic counters: `[dynamic_blas, static_blas, upgraded_static, tlas, copies, skipped, tlas_srvs, upg_geoms, upg_non_opaque, compacted, blas_refits]`.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const STAT_DYNAMIC_BLAS: usize = 0;
const STAT_STATIC_BLAS: usize = 1;
const STAT_UPGRADED_STATIC: usize = 2;
const STAT_TLAS: usize = 3;
const STAT_COPIES: usize = 4;
const STAT_SKIPPED: usize = 5;
const STAT_TLAS_SRVS: usize = 6;
const STAT_UPG_GEOMS: usize = 7;
const STAT_UPG_NON_OPAQUE: usize = 8;
const STAT_COMPACTED: usize = 9;
const STAT_BLAS_REFITS: usize = 10;

/// Logs native DXR statistics periodically alongside the host frame-rate line.
pub(crate) fn log_stats(frames: f64) {
    if !enabled() || frames <= 0.0 {
        return;
    }
    let [
        dyn_blas,
        stat_blas,
        upgraded,
        tlas,
        copies,
        skipped,
        srvs,
        upg_geoms,
        upg_non_opaque,
        compacted,
        refits,
    ] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(|i| STATS[i].swap(0, Ordering::Relaxed));
    let total_as = as_map().read().unwrap_or_else(|e| e.into_inner()).len();
    let (patched, missing, nulls, cull_disable, cull_masked, cd_mask, ss_mask, slab_total, slab_free) = {
        let guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let (slab_total, slab_free) = guard.as_ref().map_or((0, 0), |s| s.slabs.stats());
        let (p, m, n, cd, cm, cdm, ssm) =
            if let Some((_, ptr, _)) = guard.as_ref().and_then(|s| s.patch_stats.as_ref()) {
                let base = *ptr as *mut u32;
                // SAFETY: mapped 64-byte shared buffer written by GPU atomics (8 × u32 = 32 bytes).
                unsafe {
                    let p = base.add(0).read_volatile();
                    let m = base.add(1).read_volatile();
                    let n = base.add(2).read_volatile();
                    let cd = base.add(4).read_volatile();
                    let cm = base.add(5).read_volatile();
                    let cdm = base.add(6).read_volatile() & 0xFF;
                    let ssm = base.add(7).read_volatile() & 0xFF;
                    for k in 0..8 {
                        base.add(k).write_volatile(0);
                    }
                    (u64::from(p), u64::from(m), u64::from(n), u64::from(cd), u64::from(cm), cdm, ssm)
                }
            } else {
                (0, 0, 0, 0, 0, 0, 0)
            };
        (p, m, n, cd, cm, cdm, ssm, slab_total, slab_free)
    };
    if cull_disable > 0 {
        CULL_TELEMETRY[0].fetch_add(cull_disable, Ordering::Relaxed);
    }
    if cull_masked > 0 {
        CULL_TELEMETRY[1].fetch_add(cull_masked, Ordering::Relaxed);
    }
    tracing::info!(
        "native DXR per frame: dyn_blas={:.1} refits={:.1} static_blas={:.1} upgraded_static={:.1} compacted={:.1} tlas={:.1} copies={:.1} skipped={:.1} tlas_srvs={:.1} \
         (instances patched={:.0} missing_blas={:.0} null_blas={:.0} cull_disable={:.0} cull_masked={:.0} cd_mask=0x{:02x} ss_mask=0x{:02x}; resident_as={total_as}; slab_mb={:.1} slab_free_mb={:.1}; upg_geoms={upg_geoms} non_opaque={upg_non_opaque})",
        dyn_blas as f64 / frames,
        refits as f64 / frames,
        stat_blas as f64 / frames,
        upgraded as f64 / frames,
        compacted as f64 / frames,
        tlas as f64 / frames,
        copies as f64 / frames,
        skipped as f64 / frames,
        srvs as f64 / frames,
        patched as f64 / frames,
        missing as f64 / frames,
        nulls as f64 / frames,
        cull_disable as f64 / frames,
        cull_masked as f64 / frames,
        cd_mask,
        ss_mask,
        slab_total as f64 / (1024.0 * 1024.0),
        slab_free as f64 / (1024.0 * 1024.0),
    );
}

/// Sentinel guest key for the shared valid 0-instance empty TLAS used whenever
/// a shader binds a null (`Location = 0`) RTAS SRV.
const EMPTY_TLAS_KEY: u64 = u64::MAX & !0xFF;

/// Ensures `guest_va` has a stable host TLAS slot of at least `min_bytes`.
pub(crate) fn ensure_tlas_slot(guest_va: u64, min_bytes: u64) -> Option<(u64, ID3D12Resource)> {
    let key = guest_va & !0xFF;
    if key == 0 {
        return None;
    }
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&key)
            && entry.kind == AsKind::Tlas
            && entry.slice.size >= min_bytes
        {
            return Some((entry.gpu_va, entry.buffer.clone()));
        }
    }
    let host = host()?;
    let mut state_guard = lock_state();
    let state = state_guard.as_mut()?;
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = map.get(&key)
        && entry.kind == AsKind::Tlas
        && entry.slice.size >= min_bytes
    {
        return Some((entry.gpu_va, entry.buffer.clone()));
    }
    let reserve = if key == EMPTY_TLAS_KEY {
        min_bytes.max(4096)
    } else {
        min_bytes.max(DEFAULT_TLAS_RESERVE)
    };
    let (buffer, slice, gpu_va) = state.slabs.alloc(host, SlabTier::TransientDynamicTlas, reserve)?;
    let mirror_tlas = if mirror_tlas_enabled() && key != EMPTY_TLAS_KEY {
        let m_reserve = min_bytes.max(DEFAULT_MIRROR_TLAS_RESERVE);
        state
            .slabs
            .alloc(host, SlabTier::TransientDynamicTlas, m_reserve)
            .map(|(mbuf, mslice, mgpu)| MirrorTlas {
                buffer: mbuf,
                slice: mslice,
                gpu_va: mgpu,
            })
    } else {
        None
    };
    let generation = state.bump_gen();
    let cur_epoch = state.tlas_build_epoch;
    if let Some(old) = map.insert(
        key,
        HostAs {
            buffer: buffer.clone(),
            slice,
            gpu_va,
            mirror_tlas,
            kind: AsKind::Tlas,
            flags: 0,
            num_instances: 0,
            built: false,
            generation,
            topo_hash: 0,
            update_scratch_bytes: 0,
            consecutive_refits: 0,
            compacted_size: 0,
            last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
            geom_va_hash: 0,
            normals_geom_off: 0,
            normals_geom_len: 0,
            normals_data_off: 0,
            normals_data_len: 0,
        },
    ) {
        state.retire_as(&old);
    }
    if !state.pending_empty_tlas.contains(&key) {
        state.pending_empty_tlas.push(key);
    }
    Some((gpu_va, buffer))
}

/// Resolves the primary host TLAS GPU virtual address for `guest_tlas_va`
/// (allocating a pre-bound empty TLAS slot on demand if not yet present).
pub(crate) fn resolve_tlas_va(guest_tlas_va: u64) -> Option<u64> {
    if !enabled() {
        return None;
    }
    let key = if guest_tlas_va == 0 {
        EMPTY_TLAS_KEY
    } else {
        guest_tlas_va & !0xFF
    };
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&key)
            && entry.kind == AsKind::Tlas
        {
            return Some(entry.gpu_va);
        }
    }
    let reserve = if key == EMPTY_TLAS_KEY { 4096 } else { DEFAULT_TLAS_RESERVE };
    ensure_tlas_slot(key, reserve).map(|(va, _)| va)
}

/// Resolves the cull-enabled mirror TLAS host GPU virtual address for `guest_tlas_va`
/// (`ARCH-13`), falling back to [`resolve_tlas_va`] when mirror TLASes are disabled or
/// unavailable (e.g. `EMPTY_TLAS_KEY`), so `HostHeap` twin slots always bind a valid TLAS SRV.
#[allow(dead_code)]
pub(crate) fn resolve_mirror_tlas_va(guest_tlas_va: u64) -> Option<u64> {
    if !enabled() {
        return None;
    }
    if !mirror_tlas_enabled() || guest_tlas_va == 0 {
        return resolve_tlas_va(guest_tlas_va);
    }
    let key = guest_tlas_va & !0xFF;
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&key)
            && entry.kind == AsKind::Tlas
        {
            return Some(entry.mirror_tlas.as_ref().map_or(entry.gpu_va, |m| m.gpu_va));
        }
    }
    let (primary_va, _) = ensure_tlas_slot(key, DEFAULT_TLAS_RESERVE)?;
    let map = as_map().read().unwrap_or_else(|e| e.into_inner());
    Some(
        map.get(&key)
            .and_then(|e| e.mirror_tlas.as_ref().map(|m| m.gpu_va))
            .unwrap_or(primary_va),
    )
}

/// Lock-free cached GPU VA of the 64 MiB `SLOT_DXR_NORMALS` lookup buffer.
static NORMALS_GPU_VA: AtomicU64 = AtomicU64::new(0);

/// Returns the GPU virtual address of the shared `SLOT_DXR_NORMALS` buffer,
/// lazily allocating it on first call when native DXR is enabled.
pub(crate) fn normals_gpu_va() -> u64 {
    let va = NORMALS_GPU_VA.load(Ordering::Acquire);
    if va != 0 || !enabled() {
        return va;
    }
    let Some(host) = host() else {
        return 0;
    };
    let mut guard = lock_state();
    let Some(state) = guard.as_mut() else {
        return 0;
    };
    state.ensure_normals_gpu(host).map_or(0, |(_, gpu)| gpu)
}

/// Deterministic slot map assigning each active guest TLAS key (`va & !0xFF`) a
/// disjoint 16,384-word instance map base inside `SLOT_DXR_NORMALS` Region 0.
static TLAS_META_SLOTS: Mutex<([u64; victory_shader::emit::DXR_NORMALS_MAX_TLAS as usize], usize)> =
    Mutex::new(([0u64; victory_shader::emit::DXR_NORMALS_MAX_TLAS as usize], 0));

/// Returns the `SLOT_DXR_NORMALS` Region 0 base word (`1 + slot * 16384`) for
/// `guest_tlas_va`, or `0` for a null/empty TLAS.
pub(crate) fn tlas_meta_base(guest_tlas_va: u64) -> u32 {
    let key = guest_tlas_va & !0xFF;
    if key == 0 || key == EMPTY_TLAS_KEY {
        return 0;
    }
    let mut guard = TLAS_META_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let (slots, next) = &mut *guard;
    for (idx, &k) in slots.iter().enumerate() {
        if k == key {
            return 1 + (idx as u32) * victory_shader::emit::DXR_NORMALS_TLAS_STRIDE;
        }
    }
    let slot = slots.iter().position(|&k| k == 0).unwrap_or_else(|| {
        let s = *next % slots.len();
        *next = next.wrapping_add(1);
        s
    });
    slots[slot] = key;
    1 + (slot as u32) * victory_shader::emit::DXR_NORMALS_TLAS_STRIDE
}

/// Returns a valid 0-instance host empty TLAS `(gpu_va, backing_slab)` so null
/// RTAS SRVs never execute `TraceRayInline` against `Location = 0`.
pub(crate) fn empty_tlas_srv() -> Option<(u64, ID3D12Resource)> {
    if !enabled() {
        return None;
    }
    ensure_tlas_slot(EMPTY_TLAS_KEY, 4096)
}

/// Returns `(host_tlas_gpu_va, backing_slab_buffer)` for a guest TLAS SRV
/// (`D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`).
pub(crate) fn tlas_srv(guest_tlas_va: u64) -> Option<(u64, ID3D12Resource)> {
    if !enabled() {
        return None;
    }
    let res = if guest_tlas_va == 0 {
        ensure_tlas_slot(EMPTY_TLAS_KEY, 4096)?
    } else {
        ensure_tlas_slot(guest_tlas_va, DEFAULT_TLAS_RESERVE)?
    };
    STATS[STAT_TLAS_SRVS].fetch_add(1, Ordering::Relaxed);
    Some(res)
}

fn has_pending_work() -> bool {
    DXR_STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|s| {
            !s.pending_empty_tlas.is_empty()
                || !s.pending_readback_requeue.is_empty()
                || !s.pending_static_upgrades.is_empty()
                || !s.pending_compactions.is_empty()
        })
}

// ============================================================================
// GPU Compute Shaders (compiled once via D3DCompile / cs_5_1)
// ============================================================================

/// Whether `PATCH_INSTANCES_HLSL` accumulates and flushes GPU telemetry counters
/// to the mapped PCIe `patch_stats` buffer (enabled only in `diagnostics` builds).
#[cfg(feature = "diagnostics")]
const PATCH_ENABLE_STATS: u32 = 1;
#[cfg(not(feature = "diagnostics"))]
const PATCH_ENABLE_STATS: u32 = 0;

/// Patches guest `D3D12_RAYTRACING_INSTANCE_DESC` (64B stride) BLAS VAs through
/// the open-addressed GPU hash table `(guest_blas_va >> 8 -> host_blas_va)`.
/// Missing BLASes get `AccelerationStructure = 0` and `InstanceMask = 0`.
///
/// When `enable_stats != 0` (`diagnostics` builds), stats counters
/// (`[patched_ok, missing_blas, null_blas, total, cull_disable, cull_masked, cd_mask, ss_mask]`)
/// are accumulated per threadgroup in `groupshared` memory and flushed once by `gi == 0`
/// to avoid per-thread PCIe atomics against the mapped shared buffer. When `enable_stats == 0`
/// (release builds), all groupshared barriers and PCIe atomics are skipped.
const PATCH_INSTANCES_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Params : register(b0) {
    uint num_instances;
    uint src_offset;
    uint hash_mask;
    uint cull_mask_cfg;
    uint enable_stats;
    uint tlas_inst_base;
    uint mirror_offset;
};
ByteAddressBuffer guest_inst : register(t0);
ByteAddressBuffer blas_table : register(t1);
RWByteAddressBuffer host_inst : register(u0);
RWByteAddressBuffer stats : register(u1);
RWByteAddressBuffer dxr_normals : register(u2);

groupshared uint g_stats[8];

uint hash_key(uint lo, uint hi) {
    uint x = lo ^ (hi * 0x9E3779B9u);
    x ^= x >> 16u;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13u;
    x *= 0xC2B2AE35u;
    x ^= x >> 16u;
    return x;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (enable_stats != 0u) {
        if (gi < 8u) {
            g_stats[gi] = 0u;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    uint i = id.x;
    if (i < num_instances) {
        uint src = src_offset + i * 64u;
        uint dst = i * 64u;
        uint4 r0 = guest_inst.Load4(src + 0u);
        uint4 r1 = guest_inst.Load4(src + 16u);
        uint4 r2 = guest_inst.Load4(src + 32u);
        uint4 r3 = guest_inst.Load4(src + 48u);

        uint blas_lo = r3.z;
        uint blas_hi = r3.w;
        uint host_lo = 0u;
        uint host_hi = 0u;
        uint geom_off = 0u;
        bool found = false;

        if ((blas_lo | blas_hi) != 0u) {
            uint key_lo = (blas_lo >> 8u) | (blas_hi << 24u);
            uint key_hi = (blas_hi >> 8u);
            uint slot = hash_key(key_lo, key_hi) & hash_mask;
            [loop]
            for (uint step = 0u; step < 64u; ++step) {
                uint entry_off = ((slot + step) & hash_mask) * 32u;
                uint4 e = blas_table.Load4(entry_off);
                if (e.x == key_lo && e.y == key_hi) {
                    host_lo = e.z;
                    host_hi = e.w;
                    geom_off = blas_table.Load(entry_off + 16u);
                    found = true;
                    break;
                }
                if ((e.x | e.y) == 0u) {
                    break;
                }
            }
            if (enable_stats != 0u) {
                if (found) {
                    InterlockedAdd(g_stats[0], 1u);
                } else {
                    InterlockedAdd(g_stats[1], 1u);
                }
            }
        } else if (enable_stats != 0u) {
            InterlockedAdd(g_stats[2], 1u);
        }
        if (enable_stats != 0u) {
            InterlockedAdd(g_stats[3], 1u);
        }

        if (found) {
            r3.z = host_lo;
            r3.w = host_hi;
            uint cur_mask = (r3.x >> 24u) & 0xFFu;
            if ((r3.y & 0x01000000u) != 0u) {
                if (enable_stats != 0u) {
                    InterlockedAdd(g_stats[4], 1u);
                    InterlockedOr(g_stats[6], cur_mask);
                }
                uint cfg_mask = cull_mask_cfg & 0xFFu;
                if (cfg_mask != 0u && cur_mask != 0u) {
                    uint new_mask = ((cull_mask_cfg & 0x100u) != 0u)
                        ? (cur_mask | cfg_mask)
                        : (((cur_mask & ~cfg_mask) != 0u) ? (cur_mask & ~cfg_mask) : cur_mask);
                    if (new_mask != cur_mask) {
                        r3.x = (r3.x & 0x00FFFFFFu) | (new_mask << 24u);
                        if (enable_stats != 0u) {
                            InterlockedAdd(g_stats[5], 1u);
                        }
                    }
                }
            } else if (enable_stats != 0u) {
                InterlockedOr(g_stats[7], cur_mask);
            }
        } else {
            r3.x &= 0x00FFFFFFu;
            r3.z = 0u;
            r3.w = 0u;
        }
        host_inst.Store4(dst + 0u, r0);
        host_inst.Store4(dst + 16u, r1);
        host_inst.Store4(dst + 32u, r2);
        host_inst.Store4(dst + 48u, r3);
        if (mirror_offset != 0u) {
            uint mdst = mirror_offset + dst;
            uint4 r3_m = r3;
            // Clear bit 0 (TRIANGLE_CULL_DISABLE = 0x1) while preserving bit 1
            // (TRIANGLE_FRONT_COUNTERCLOCKWISE = 0x2) and all other flags/masks.
            r3_m.y &= ~0x01000000u;
            host_inst.Store4(mdst + 0u, r0);
            host_inst.Store4(mdst + 16u, r1);
            host_inst.Store4(mdst + 32u, r2);
            host_inst.Store4(mdst + 48u, r3_m);
        }
        if (tlas_inst_base != 0u && i < 16384u) {
            dxr_normals.Store((tlas_inst_base + i) * 4u, found ? geom_off : 0u);
        }
    }

    if (enable_stats != 0u) {
        GroupMemoryBarrierWithGroupSync();
        if (gi == 0u) {
            [unroll]
            for (uint k = 0u; k < 6u; ++k) {
                uint v = g_stats[k];
                if (v != 0u) {
                    stats.InterlockedAdd(k * 4u, v);
                }
            }
            uint cd_m = g_stats[6];
            if (cd_m != 0u) {
                stats.InterlockedOr(24u, cd_m);
            }
            uint ss_m = g_stats[7];
            if (ss_m != 0u) {
                stats.InterlockedOr(28u, ss_m);
            }
        }
    }
}
"#;

/// Computes octahedral-packed object-space geometric triangle normals for one
/// dynamic BLAS geometry directly into `dxr_normals` (Region 2), and writes the
/// geometry's Region 1 pointer entry (`geom_entry_word -> prim_data_off`) on
/// thread `0` so zero CPU upload or `COPY_DEST` barrier is needed in `record_ops`.
const BUILD_NORMALS_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=10, b0), SRV(t0), SRV(t1), SRV(t2), UAV(u0)"
cbuffer Params : register(b0) {
    uint tri_count;
    uint vb_stride;
    uint vb_bias;
    uint vb_fmt;
    uint ib_mode;
    uint ib_bias;
    uint has_xf;
    uint geom_entry_word;
    uint prim_data_off;
    uint vertex_count;
};
ByteAddressBuffer vb : register(t0);
ByteAddressBuffer ib : register(t1);
ByteAddressBuffer xf : register(t2);
RWByteAddressBuffer dxr_normals : register(u0);

uint load_u16_vb(uint byte_off) {
    uint w = vb.Load(byte_off & ~3u);
    return (w >> ((byte_off & 2u) * 8u)) & 0xFFFFu;
}

uint load_u16_ib(uint byte_off) {
    uint w = ib.Load(byte_off & ~3u);
    return (w >> ((byte_off & 2u) * 8u)) & 0xFFFFu;
}

float snorm16(uint u) {
    int s = (int)(u << 16u) >> 16;
    return max(-1.0f, (float)s / 32767.0f);
}

float3 load_vert(uint idx) {
    uint clamped = min(idx, vertex_count - 1u);
    uint off = vb_bias + clamped * vb_stride;
    float3 p = float3(0.0f, 0.0f, 0.0f);
    if (vb_fmt == 0u) {
        uint3 w = vb.Load3(off);
        p = asfloat(w);
    } else if (vb_fmt == 1u) {
        uint2 w = vb.Load2(off);
        p = float3(asfloat(w), 0.0f);
    } else if (vb_fmt == 2u) {
        p = float3(
            f16tof32(load_u16_vb(off + 0u)),
            f16tof32(load_u16_vb(off + 2u)),
            f16tof32(load_u16_vb(off + 4u))
        );
    } else if (vb_fmt == 3u) {
        p = float3(
            snorm16(load_u16_vb(off + 0u)),
            snorm16(load_u16_vb(off + 2u)),
            snorm16(load_u16_vb(off + 4u))
        );
    } else if (vb_fmt == 4u) {
        p = float3(
            f16tof32(load_u16_vb(off + 0u)),
            f16tof32(load_u16_vb(off + 2u)),
            0.0f
        );
    } else if (vb_fmt == 5u) {
        p = float3(
            snorm16(load_u16_vb(off + 0u)),
            snorm16(load_u16_vb(off + 2u)),
            0.0f
        );
    }
    if (has_xf != 0u) {
        float4 r0 = asfloat(xf.Load4(0u));
        float4 r1 = asfloat(xf.Load4(16u));
        float4 r2 = asfloat(xf.Load4(32u));
        p = float3(dot(r0.xyz, p), dot(r1.xyz, p), dot(r2.xyz, p));
    }
    return p;
}

uint load_idx(uint pos) {
    if (ib_mode == 1u) {
        return load_u16_ib(ib_bias + pos * 2u);
    } else if (ib_mode == 2u) {
        return ib.Load(ib_bias + pos * 4u);
    }
    return pos;
}

uint pack_oct_u32(float3 n) {
    float len2 = dot(n, n);
    if (!(len2 > 1e-20f)) {
        return 0x80008000u;
    }
    n *= rsqrt(len2);
    float inv_l1 = rcp(abs(n.x) + abs(n.y) + abs(n.z));
    float2 o = n.xy * inv_l1;
    if (n.z < 0.0f) {
        float ox = (1.0f - abs(o.y)) * (o.x >= 0.0f ? 1.0f : -1.0f);
        float oy = (1.0f - abs(o.x)) * (o.y >= 0.0f ? 1.0f : -1.0f);
        o = float2(ox, oy);
    }
    uint qx = (uint)clamp(round(clamp(o.x, -1.0f, 1.0f) * 32767.0f) + 32768.0f, 0.0f, 65535.0f);
    uint qy = (uint)clamp(round(clamp(o.y, -1.0f, 1.0f) * 32767.0f) + 32768.0f, 0.0f, 65535.0f);
    return (qx & 0xFFFFu) | ((qy & 0xFFFFu) << 16u);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint t = id.x;
    if (t == 0u && geom_entry_word != 0u) {
        dxr_normals.Store(geom_entry_word * 4u, tri_count > 0u ? prim_data_off : 0u);
    }
    if (t >= tri_count || vertex_count == 0u || prim_data_off == 0u) {
        return;
    }
    uint i0 = load_idx(t * 3u + 0u);
    uint i1 = load_idx(t * 3u + 1u);
    uint i2 = load_idx(t * 3u + 2u);
    float3 v0 = load_vert(i0);
    float3 v1 = load_vert(i1);
    float3 v2 = load_vert(i2);
    float3 n = cross(v1 - v0, v2 - v0);
    dxr_normals.Store((prim_data_off + t) * 4u, pack_oct_u32(n));
}
"#;

/// Extracts flat `float3` triangle vertices from unpacked 64-byte TriangleNodes
/// in a rehydrated console BVH4 living in `crate::memory`'s paged arenas, and
/// optionally dumps the raw 64-byte TriangleNodes plus 16-byte header to `u1`
/// for zero-stall async CPU decoding into multi-geometry BLASes.
///
/// Header fields: `dw[3]` = `numElements`, `dw[13]` = `offsetToElements` (bytes).
/// Each 64-byte TriangleNode stores:
/// - bytes `0..48`: 4 vertices (`v0..v3`, 3×f32 each)
/// - `dw[12]`: `GeometryIndexAndFlags` (bit 0 = OPAQUE, bit 7 = tri1 present, bits 31:8 = GeometryIndex)
/// - `dw[13]`: `tri0` `PrimitiveIndex`
/// - `dw[14]`: `tri1` `PrimitiveIndex`
/// - `dw[15]`: per-triangle permutation bytes matching `bvh.rs::TRIANGLE_VERTICES`
///   (`tri0: [0,1,2]`, `tri1: [1,3,2]`, winding `[first, i, j]` where `first = (3-i-j)&3`).
/// Unused or out-of-range slots emit degenerate NaN triangles (`0x7FC00000`) which
/// DXR treats as inactive primitives.
const EXTRACT_BVH4_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), SRV(t0), UAV(u0), UAV(u1), DescriptorTable(UAV(u0, numDescriptors=4096, space=1))"
cbuffer Params : register(b0) {
    uint base_lo;
    uint base_hi;
    uint max_nodes;
    uint size_bytes;
    uint dump_nodes;
};
ByteAddressBuffer directory : register(t0);
RWByteAddressBuffer out_verts : register(u0);
RWByteAddressBuffer out_nodes : register(u1);
RWByteAddressBuffer mem[4096] : register(u0, space1);

uint load_u32(uint byte_off) {
    uint lo = base_lo + byte_off;
    uint hi = base_hi + (lo < base_lo ? 1u : 0u);
    uint page = ((lo >> 16u) | (hi << 16u)) & 0x1FFFFFu;
    uint2 entry = directory.Load2(page * 8u);
    uint copy = entry.x & 0x7FFFFFFFu;
    if (copy == 0u || copy >= 4096u) return 0u;
    return mem[copy].Load(entry.y + (lo & 0xFFFFu));
}

void write_nan_tri(uint dst) {
    uint3 nan3 = uint3(0x7FC00000u, 0x7FC00000u, 0x7FC00000u);
    out_verts.Store3(dst + 0u, nan3);
    out_verts.Store3(dst + 12u, nan3);
    out_verts.Store3(dst + 24u, nan3);
}

void write_tri(uint dst, uint3 verts[4], uint map_nibbles, uint id_byte) {
    uint i = id_byte & 3u;
    uint j = (id_byte >> 2u) & 3u;
    if (i >= 3u || j >= 3u || i == j) {
        write_nan_tri(dst);
        return;
    }
    uint f = (3u - i - j) & 3u;
    uint v0 = (map_nibbles >> (4u * f)) & 3u;
    uint v1 = (map_nibbles >> (4u * i)) & 3u;
    uint v2 = (map_nibbles >> (4u * j)) & 3u;
    out_verts.Store3(dst + 0u, verts[v0]);
    out_verts.Store3(dst + 12u, verts[v1]);
    out_verts.Store3(dst + 24u, verts[v2]);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint node_idx = id.x;
    if (node_idx >= max_nodes) return;
    uint dst0 = node_idx * 72u;
    uint dst1 = dst0 + 36u;

    uint num_elements = load_u32(12u);
    uint offset_to_elements = load_u32(52u);
    bool hdr_ok = (offset_to_elements >= 64u && size_bytes >= 64u && offset_to_elements <= size_bytes - 64u);
    uint max_by_size = hdr_ok ? ((size_bytes - offset_to_elements) >> 6u) : 0u;
    uint eff_elements = min(num_elements, max_by_size);
    if (node_idx == 0u && dump_nodes > 0u) {
        out_nodes.Store4(0u, uint4(eff_elements, offset_to_elements, hdr_ok ? dump_nodes : 0u, size_bytes));
    }
    if (!hdr_ok || node_idx >= eff_elements) {
        write_nan_tri(dst0);
        write_nan_tri(dst1);
        return;
    }

    uint node_off = offset_to_elements + node_idx * 64u;
    uint lo = base_lo + node_off;
    uint hi = base_hi + (lo < base_lo ? 1u : 0u);
    uint page = ((lo >> 16u) | (hi << 16u)) & 0x1FFFFFu;
    uint2 entry = directory.Load2(page * 8u);
    uint copy = entry.x & 0x7FFFFFFFu;
    if (copy == 0u || copy >= 4096u) {
        write_nan_tri(dst0);
        write_nan_tri(dst1);
        if (node_idx < dump_nodes) {
            uint n_dst = 16u + node_idx * 64u;
            out_nodes.Store4(n_dst + 0u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 16u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 32u, uint4(0u, 0u, 0u, 0u));
            out_nodes.Store4(n_dst + 48u, uint4(0u, 0u, 0u, 0xFFFFFFFFu));
        }
        return;
    }
    uint off = entry.y + (lo & 0xFFFFu);
    uint4 w0 = mem[copy].Load4(off + 0u);
    uint4 w1 = mem[copy].Load4(off + 16u);
    uint4 w2 = mem[copy].Load4(off + 32u);
    uint4 w3 = mem[copy].Load4(off + 48u);

    if (node_idx < dump_nodes) {
        uint n_dst = 16u + node_idx * 64u;
        out_nodes.Store4(n_dst + 0u, w0);
        out_nodes.Store4(n_dst + 16u, w1);
        out_nodes.Store4(n_dst + 32u, w2);
        out_nodes.Store4(n_dst + 48u, w3);
    }

    uint3 verts[4];
    verts[0] = uint3(w0.x, w0.y, w0.z);
    verts[1] = uint3(w0.w, w1.x, w1.y);
    verts[2] = uint3(w1.z, w1.w, w2.x);
    verts[3] = uint3(w2.y, w2.z, w2.w);

    uint dw12 = w3.x;
    uint dw15 = w3.w;

    // Triangle 0: TRIANGLE_VERTICES[0] = [0, 1, 2] -> nibbles 0x210
    write_tri(dst0, verts, 0x210u, dw15 & 0xFFu);

    // Triangle 1 (when bit 7 of GeometryIndexAndFlags is set): [1, 3, 2] -> nibbles 0x231
    if ((dw12 & 0x80u) != 0u) {
        write_tri(dst1, verts, 0x231u, (dw15 >> 8u) & 0xFFu);
    } else {
        write_nan_tri(dst1);
    }
}
"#;

struct ComputePass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are free-threaded.
unsafe impl Send for ComputePass {}
unsafe impl Sync for ComputePass {}

fn make_compute_pass(host: &Host, hlsl: &str, label: &str) -> Option<ComputePass> {
    let code = match compile(hlsl, s!("main"), s!("cs_5_1")) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!(error = %e, label, "dxr: compute shader compile failed");
            return None;
        }
    };
    // SAFETY: bytecode lives through the calls; root signature is embedded in it.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).ok()?;
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .ok()?;
        name(&pso, || label.to_owned());
        Some(ComputePass { root, pso })
    }
}

fn patch_instances_pass(host: &Host) -> Option<&'static ComputePass> {
    static PASS: OnceLock<Option<ComputePass>> = OnceLock::new();
    PASS.get_or_init(|| make_compute_pass(host, PATCH_INSTANCES_HLSL, "victory dxr: patch instances"))
        .as_ref()
}

fn extract_bvh4_pass(host: &Host) -> Option<&'static ComputePass> {
    static PASS: OnceLock<Option<ComputePass>> = OnceLock::new();
    PASS.get_or_init(|| make_compute_pass(host, EXTRACT_BVH4_HLSL, "victory dxr: extract bvh4"))
        .as_ref()
}

fn build_normals_pass(host: &Host) -> Option<&'static ComputePass> {
    static PASS: OnceLock<Option<ComputePass>> = OnceLock::new();
    PASS.get_or_init(|| make_compute_pass(host, BUILD_NORMALS_HLSL, "victory dxr: build normals"))
        .as_ref()
}

// ============================================================================
// Public entry points called from `victory-hle::modules::bvh_builder`
// ============================================================================

fn prebuild_flags(kind: u32, raw_flags: u32, refit_enabled: bool) -> u32 {
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
    let mut flags = raw_flags & !PERFORM_UPDATE;
    let compaction_only = (raw_flags & ALLOW_COMPACTION) != 0 && (raw_flags & ALLOW_UPDATE) == 0;
    if kind == 1 && refit_enabled && !compaction_only {
        flags |= ALLOW_UPDATE;
    }
    flags
}

fn sanitize_prebuild_info(info: GuestPrebuildInfo, allow_update: bool) -> GuestPrebuildInfo {
    let result = info.result_data_max_size_in_bytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
    let scratch = info.scratch_data_size_in_bytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
    let update = if info.update_scratch_data_size_in_bytes > 0 {
        info.update_scratch_data_size_in_bytes.next_multiple_of(AS_ALIGN)
    } else if allow_update {
        AS_ALIGN
    } else {
        0
    };
    GuestPrebuildInfo {
        result_data_max_size_in_bytes: result,
        scratch_data_size_in_bytes: scratch,
        update_scratch_data_size_in_bytes: update,
    }
}

/// Safe 256B-aligned prebuild estimate used only when neither host DXR nor
/// `XboxBvhBuilder.dll` is available (e.g. unit tests or pure native fallback
/// for a non-empty `GuestStreamDesc`).
pub fn fallback_prebuild_info(
    inputs: &GuestBuildAsInputs,
    stream_desc: Option<&GuestStreamDesc>,
) -> GuestPrebuildInfo {
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    let flags = prebuild_flags(inputs.kind, inputs.flags, refit_dynamic_blas());
    let stream_bytes = stream_desc.map_or(0, |sd| sd.size_in_bytes);
    sanitize_prebuild_info(
        GuestPrebuildInfo {
            result_data_max_size_in_bytes: stream_bytes.max(AS_ALIGN),
            scratch_data_size_in_bytes: AS_ALIGN,
            update_scratch_data_size_in_bytes: if (flags & ALLOW_UPDATE) != 0 { AS_ALIGN } else { 0 },
        },
        (flags & ALLOW_UPDATE) != 0,
    )
}

/// Queries host DXR prebuild info from guest inputs (used when
/// `VICTORY_NATIVE_RT_SUPPRESS_BUILDER=1`).
///
/// Returns `None` when `stream_desc` carries console packed/streamed BVH4 data
/// (`size_in_bytes > 0`) so `bvh_builder::get_prebuild_info` falls back to
/// `XboxBvhBuilder.dll` for static BVH4 rehydration sizing (or to
/// [`fallback_prebuild_info`] if the DLL is absent).
pub fn prebuild_info_with_stream(
    inputs: &GuestBuildAsInputs,
    stream_desc: Option<&GuestStreamDesc>,
) -> Option<GuestPrebuildInfo> {
    if stream_desc.is_some_and(|sd| sd.size_in_bytes > 0) {
        return None;
    }
    if !enabled() {
        return None;
    }
    let host = host()?;
    let device5 = host.device.cast::<ID3D12Device5>().ok()?;
    let (host_inputs, _geoms) = translate_prebuild_inputs(inputs, refit_dynamic_blas())?;
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5, translated inputs, and output pointer.
    unsafe {
        device5.GetRaytracingAccelerationStructurePrebuildInfo(&host_inputs, &mut info);
    }
    const ALLOW_UPDATE: i32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0;
    Some(sanitize_prebuild_info(
        GuestPrebuildInfo {
            result_data_max_size_in_bytes: info.ResultDataMaxSizeInBytes,
            scratch_data_size_in_bytes: info.ScratchDataSizeInBytes,
            update_scratch_data_size_in_bytes: info.UpdateScratchDataSizeInBytes,
        },
        (host_inputs.Flags.0 & ALLOW_UPDATE) != 0,
    ))
}

/// Queries host DXR prebuild info from guest inputs without a stream descriptor.
pub fn prebuild_info(inputs: &GuestBuildAsInputs) -> Option<GuestPrebuildInfo> {
    prebuild_info_with_stream(inputs, None)
}

/// Queues a batch of dynamic BLAS/TLAS builds from `XboxBvhBuilder::BuildRaytracingAccelerationStructureBatched2X`.
pub fn enqueue_build_batched(cmd_list: *const c_void, num_descs: u32, descs: *const GuestBatchedDesc2) {
    if !enabled() || cmd_list.is_null() || descs.is_null() || num_descs == 0 {
        return;
    }
    // SAFETY: caller passes `num_descs` batched descriptors.
    let slice = unsafe { std::slice::from_raw_parts(descs, num_descs as usize) };
    for entry in slice {
        let desc = &entry.desc;
        if desc.dest_va == 0 {
            continue;
        }
        let postbuild = if entry.num_postbuild_info_descs > 0 && !entry.postbuild_info_descs.is_null() {
            // SAFETY: caller passes `num_postbuild_info_descs` descriptors.
            let pb = unsafe {
                std::slice::from_raw_parts(entry.postbuild_info_descs, entry.num_postbuild_info_descs as usize)
            };
            pb.iter()
                .map(|p| PostbuildSpec {
                    dest_buffer: p.dest_buffer,
                    info_type: p.info_type,
                })
                .collect()
        } else {
            Vec::new()
        };
        let op = match desc.inputs.kind {
            0 => DxrOp::BuildTlas {
                dest_va: desc.dest_va,
                flags: desc.inputs.flags,
                num_instances: desc.inputs.num_descs,
                instance_descs_va: desc.inputs.ptr_or_va,
                postbuild,
            },
            1 => {
                let geometries = copy_guest_geometries(&desc.inputs);
                DxrOp::BuildBlas {
                    dest_va: desc.dest_va,
                    source_va: desc.source_va,
                    flags: desc.inputs.flags,
                    geometries,
                    postbuild,
                }
            }
            _ => continue,
        };
        crate::list::queue_dxr_op(cmd_list as This, op);
    }
}

/// Queues an AS copy from `XboxBvhBuilder::CopyRaytracingAccelerationStructure`.
pub fn enqueue_copy(cmd_list: *const c_void, dest_va: u64, source_va: u64, mode: u32) {
    if !enabled() || cmd_list.is_null() || dest_va == 0 || source_va == 0 {
        return;
    }
    crate::list::queue_dxr_op(
        cmd_list as This,
        DxrOp::CopyAs {
            dest_va,
            source_va,
            mode,
        },
    );
}

/// Queues static BLAS extraction + DXR build for a batch of rehydrated console BVH4s from
/// `XboxBvhBuilder::RehydrateRaytracingAccelerationStructureBatched2X`.
pub fn enqueue_rehydrate_batched(
    cmd_list: *const c_void,
    num_descs: u32,
    descs: *const GuestBuildAsDesc,
    sizes: *const u64,
) {
    if !enabled() || cmd_list.is_null() || descs.is_null() || num_descs == 0 {
        return;
    }
    // SAFETY: caller passes `num_descs` descriptors (and optional `sizes` array).
    let slice = unsafe { std::slice::from_raw_parts(descs, num_descs as usize) };
    let size_slice = if sizes.is_null() {
        None
    } else {
        // SAFETY: parallel array of `num_descs` byte lengths.
        Some(unsafe { std::slice::from_raw_parts(sizes, num_descs as usize) })
    };
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    for (i, desc) in slice.iter().enumerate() {
        if desc.dest_va == 0 {
            continue;
        }
        let raw_arg = size_slice.map_or(0, |s| s[i]);
        let probed_dest = probe_packed_bvh_size(desc.dest_va);
        let probed_src = (desc.source_va != 0)
            .then(|| probe_packed_bvh_size(desc.source_va))
            .flatten();
        // `RehydrateRaytracingAccelerationStructureBatched2X`'s 5th parameter may be
        // either byte sizes or GPU VAs/flags; never trust values outside `(128..=64 MiB)`.
        let arg_size = (128..=MAX_PACKED_BVH_BYTES).contains(&raw_arg).then_some(raw_arg);
        let size_bytes = probed_dest.or(probed_src).or(arg_size).unwrap_or(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                dest = format_args!("{:#x}", desc.dest_va),
                src = format_args!("{:#x}", desc.source_va),
                raw_arg = format_args!("{:#x}", raw_arg),
                probed_dest = ?probed_dest,
                probed_src = ?probed_src,
                size_bytes,
                in_kind = desc.inputs.kind,
                in_flags = format_args!("{:#x}", desc.inputs.flags),
                in_num_descs = desc.inputs.num_descs,
                in_layout = desc.inputs.descs_layout,
                in_ptr = format_args!("{:#x}", desc.inputs.ptr_or_va),
                "dxr: rehydrate static BLAS sample"
            );
        }
        crate::list::queue_dxr_op(
            cmd_list as This,
            DxrOp::ExtractStaticBlas {
                dest_va: desc.dest_va,
                size_bytes,
            },
        );
    }
}

/// Infers unpacked byte size from a console packed BVH4 header (`dw[0] & 0xFFFF == 6`,
/// marks `0x8100`, `dw[1]` = node count minus one; matches `memory.rs::packed_len`).
fn probe_packed_bvh_size(va: u64) -> Option<u64> {
    if va == 0 || (va & 0xFF) != 0 {
        return None;
    }
    let mut header = [0u8; 8];
    crate::host::copy_mapped(header.as_mut_ptr(), va as *const u8, 8, false);
    let words = u64::from_le_bytes(header);
    let version = words as u32;
    let marks = version >> 16;
    let nodes = words >> 32;
    let len = nodes.saturating_add(1).saturating_mul(64);
    ((version & 0xFFFF) == 6 && (marks & 0x8100) == 0x8100 && nodes > 0 && len <= MAX_PACKED_BVH_BYTES).then_some(len)
}

/// Probes either a packed BVH4 header or an already-unpacked BVH4 header (`dw[3]` =
/// `num_elements`, `dw[13]` = `offset_to_elements`) at `va` to recover `(size_bytes, max_nodes)`.
fn probe_bvh_bounds(va: u64, hint_bytes: u64) -> Option<(u64, u64)> {
    let mut bytes = if (128..=MAX_PACKED_BVH_BYTES).contains(&hint_bytes) {
        hint_bytes
    } else {
        probe_packed_bvh_size(va).unwrap_or(0)
    };
    // Also read the 64-byte header: if the BVH is already unpacked in guest memory,
    // `dw[3]` is exact `numElements` and `dw[13]` is `offsetToElements` in bytes.
    let mut hdr = [0u8; 64];
    crate::host::copy_mapped(hdr.as_mut_ptr(), va as *const u8, 64, false);
    let dw = |idx: usize| u32::from_le_bytes(hdr[idx * 4..idx * 4 + 4].try_into().unwrap());
    let num_elements = u64::from(dw(3));
    let off_elements = u64::from(dw(13));
    if bytes == 0 && off_elements >= 64 && num_elements > 0 && num_elements <= MAX_STATIC_NODES {
        let unpacked = off_elements.saturating_add(num_elements.saturating_mul(64));
        if unpacked <= MAX_PACKED_BVH_BYTES {
            bytes = unpacked;
        }
    }
    if bytes < 128 || bytes > MAX_PACKED_BVH_BYTES {
        return None;
    }
    // Use exact `num_elements` when header looks unpacked; otherwise conservative upper bound
    // (at most `size_bytes / 64`, capped at `MAX_STATIC_NODES`).
    let max_nodes = if off_elements >= 64
        && off_elements < bytes
        && num_elements > 0
        && off_elements.saturating_add(num_elements.saturating_mul(64)) <= bytes
    {
        num_elements.min(MAX_STATIC_NODES)
    } else {
        (bytes / 64).clamp(1, MAX_STATIC_NODES)
    };
    Some((bytes, max_nodes))
}

fn copy_guest_geometries(inputs: &GuestBuildAsInputs) -> Vec<GeometrySpec> {
    let count = inputs.num_descs as usize;
    if count == 0 || inputs.ptr_or_va == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let g = if inputs.descs_layout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS.0 as u32 {
            // SAFETY: guest passes an array of `count` geometry descriptor pointers.
            let ptr = unsafe { *(inputs.ptr_or_va as *const *const GuestGeometryDesc).add(i) };
            if ptr.is_null() {
                continue;
            }
            // SAFETY: non-null guest geometry descriptor.
            unsafe { ptr.read_unaligned() }
        } else {
            // SAFETY: guest passes a contiguous array of `count` geometry descriptors.
            unsafe { (inputs.ptr_or_va as *const GuestGeometryDesc).add(i).read_unaligned() }
        };
        let kind = match g.kind {
            0 => {
                // SAFETY: `kind == 0` selects `triangles`.
                let t = unsafe { g.data.triangles };
                GeometryKind::Triangles {
                    transform3x4: t.transform3x4,
                    index_format: t.index_format,
                    vertex_format: t.vertex_format,
                    index_count: t.index_count,
                    vertex_count: t.vertex_count,
                    index_buffer: t.index_buffer,
                    vertex_buffer_va: t.vertex_buffer_va,
                    vertex_buffer_stride: t.vertex_buffer_stride,
                }
            }
            1 => {
                // SAFETY: `kind == 1` selects `aabbs`.
                let a = unsafe { g.data.aabbs };
                GeometryKind::Aabbs {
                    aabb_count: a.aabb_count,
                    aabbs_va: a.aabbs_va,
                    aabbs_stride: a.aabbs_stride,
                }
            }
            _ => continue,
        };
        out.push(GeometrySpec { flags: g.flags, kind });
    }
    out
}

fn translate_prebuild_inputs(
    inputs: &GuestBuildAsInputs,
    refit_enabled: bool,
) -> Option<(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS, Vec<D3D12_RAYTRACING_GEOMETRY_DESC>)> {
    let flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(
        prebuild_flags(inputs.kind, inputs.flags, refit_enabled) as i32,
    );
    match inputs.kind {
        0 => Some((
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: flags,
                NumDescs: inputs.num_descs,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: 0,
                },
            },
            Vec::new(),
        )),
        1 => {
            let specs = copy_guest_geometries(inputs);
            let mut geoms = Vec::with_capacity(specs.len());
            for spec in &specs {
                geoms.push(host_geom_desc_for_prebuild(spec));
            }
            let ptr = if geoms.is_empty() {
                std::ptr::null()
            } else {
                geoms.as_ptr()
            };
            Some((
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: flags,
                    NumDescs: geoms.len() as u32,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: ptr,
                    },
                },
                geoms,
            ))
        }
        _ => None,
    }
}

fn host_geom_desc_for_prebuild(spec: &GeometrySpec) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    match spec.kind {
        GeometryKind::Triangles {
            transform3x4,
            index_format,
            vertex_format,
            index_count,
            vertex_count,
            index_buffer,
            vertex_buffer_va,
            vertex_buffer_stride,
        } => D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: dynamic_geom_flags(spec.flags),
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: u64::from(transform3x4 != 0) * 16,
                    IndexFormat: DXGI_FORMAT(crate::resource::pc_format(index_format) as i32),
                    VertexFormat: DXGI_FORMAT(crate::resource::pc_format(vertex_format) as i32),
                    IndexCount: index_count,
                    VertexCount: vertex_count,
                    IndexBuffer: u64::from(index_buffer != 0) * 16,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: u64::from(vertex_buffer_va != 0) * 16,
                        StrideInBytes: vertex_buffer_stride,
                    },
                },
            },
        },
        GeometryKind::Aabbs {
            aabb_count,
            aabbs_va,
            aabbs_stride,
        } => D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
            Flags: dynamic_geom_flags(spec.flags),
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                    AABBCount: aabb_count,
                    AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: u64::from(aabbs_va != 0) * 16,
                        StrideInBytes: aabbs_stride,
                    },
                },
            },
        },
    }
}

// ============================================================================
// List-recording preparation (schedules shadow refreshes before `card_before`)
// ============================================================================

/// Schedules shadow buffer refreshes/write-backs on `list` for guest buffers
/// read or written by `op`, so `Host::submit` uploads them via `card_before`
/// before DXR replay runs.
pub(crate) fn prepare_list_op(list: &HostList, op: &DxrOp) {
    match op {
        DxrOp::BuildBlas { geometries, postbuild, .. } => {
            for g in geometries {
                match g.kind {
                    GeometryKind::Triangles {
                        transform3x4,
                        index_format,
                        index_count,
                        vertex_count,
                        index_buffer,
                        vertex_buffer_va,
                        vertex_buffer_stride,
                        ..
                    } => {
                        if vertex_buffer_va != 0 && vertex_count > 0 {
                            let stride = vertex_buffer_stride.max(12);
                            let len = u64::from(vertex_count).saturating_mul(stride);
                            if let Some(s) = crate::shadow::find(vertex_buffer_va, len) {
                                // Skip CPU refresh on GPU-written dynamic BLAS vertex buffers (skinning/cloth output, `CPU-19`).
                                if s.written() || s.gpu_only_skip() {
                                    list.keep_shadow(s);
                                } else {
                                    s.refresh(list, vertex_buffer_va, len);
                                }
                            }
                        }
                        if index_buffer != 0 && index_count > 0 {
                            let elem = if crate::resource::pc_format(index_format) == DXGI_FORMAT_R16_UINT.0 as u32 {
                                2u64
                            } else {
                                4u64
                            };
                            let len = u64::from(index_count).saturating_mul(elem);
                            if let Some(s) = crate::shadow::find(index_buffer, len) {
                                if s.written() || s.gpu_only_skip() {
                                    list.keep_shadow(s);
                                } else {
                                    s.refresh(list, index_buffer, len);
                                }
                            }
                        }
                        if transform3x4 != 0
                            && let Some(s) = crate::shadow::find(transform3x4, 48)
                        {
                            s.refresh(list, transform3x4, 48);
                        }
                    }
                    GeometryKind::Aabbs {
                        aabb_count,
                        aabbs_va,
                        aabbs_stride,
                    } => {
                        if aabbs_va != 0 && aabb_count > 0 {
                            let stride = aabbs_stride.max(24);
                            let len = aabb_count.saturating_mul(stride);
                            if let Some(s) = crate::shadow::find(aabbs_va, len) {
                                s.refresh(list, aabbs_va, len);
                            }
                        }
                    }
                }
            }
            if suppress_builder() {
                for pb in postbuild {
                    if let Some(bytes) = postbuild_byte_size(pb.info_type)
                        && pb.dest_buffer != 0
                        && pb.dest_buffer % 8 == 0
                        && let Some(s) = crate::shadow::find(pb.dest_buffer, bytes)
                    {
                        s.refresh(list, pb.dest_buffer, bytes);
                        s.gpu_writes(list, pb.dest_buffer, bytes);
                    }
                }
            }
        }
        DxrOp::BuildTlas {
            num_instances,
            instance_descs_va,
            postbuild,
            ..
        } => {
            if *num_instances > 0 && *instance_descs_va != 0 {
                let len = u64::from(*num_instances).saturating_mul(64);
                if let Some(s) = crate::shadow::find(*instance_descs_va, len) {
                    s.refresh(list, *instance_descs_va, len);
                }
            }
            if suppress_builder() {
                for pb in postbuild {
                    if let Some(bytes) = postbuild_byte_size(pb.info_type)
                        && pb.dest_buffer != 0
                        && pb.dest_buffer % 8 == 0
                        && let Some(s) = crate::shadow::find(pb.dest_buffer, bytes)
                    {
                        s.refresh(list, pb.dest_buffer, bytes);
                        s.gpu_writes(list, pb.dest_buffer, bytes);
                    }
                }
            }
        }
        DxrOp::CopyAs { .. } | DxrOp::ExtractStaticBlas { .. } => {}
    }
}

// ============================================================================
// Submission-time replay on `victory-submit`
// ============================================================================

/// Replays queued [`DxrOp`]s for `runs` onto pooled host `ID3D12GraphicsCommandList4`
/// lists and splices them into `lists` (updating `firsts`) in exact console submission
/// order. Called from `Host::submit` after `card_before` and before `ExecuteCommandLists`.
pub(crate) fn record_submission(
    host: &Host,
    runs: &[&HostList],
    lists: &mut Vec<Option<ID3D12CommandList>>,
    firsts: &mut Vec<usize>,
) -> Option<DxrSubmission> {
    if !enabled() {
        return None;
    }
    let has_ops = runs.iter().any(|l| l.has_dxr_ops());
    if !has_ops && !has_pending_work() {
        return None;
    }
    let Ok(device5) = host.device.cast::<ID3D12Device5>() else {
        return None;
    };

    let mut state_guard = lock_state();
    let state = state_guard.as_mut()?;
    let use_compute = host.dxr_async_compute();
    let mut sub = state.take_submission();
    sub.async_batches.clear();
    sub.dxr_fence_val = 0;
    sub.ran_on_compute = use_compute;
    let mut new_lists: Vec<Option<ID3D12CommandList>> = if use_compute {
        Vec::new()
    } else {
        Vec::with_capacity(lists.len() + runs.len() + 2)
    };
    let mut new_firsts: Vec<usize> = if use_compute {
        Vec::new()
    } else {
        Vec::with_capacity(firsts.len())
    };

    // Clone queued ops once per run so we don't duplicate allocations between
    // readback pre-sizing and command recording.
    let run_ops: Vec<Vec<DxrOp>> = if has_ops {
        runs.iter()
            .map(|r| if r.has_dxr_ops() { r.dxr_ops() } else { Vec::new() })
            .collect()
    } else {
        Vec::new()
    };

    // Pre-size this submission's node readback buffer upfront (capped at MAX_READBACK_BYTES_PER_SUB)
    // including any overflowed Frame-0 static BLASes queued in `pending_readback_requeue`.
    let mut need_readback = 0u64;
    for &(key, size_bytes, _) in &state.pending_readback_requeue {
        if let Some((_, max_nodes)) = probe_bvh_bounds(key, size_bytes) {
            let dump = (16u64 + max_nodes.saturating_mul(64)).next_multiple_of(64);
            if need_readback.saturating_add(dump) <= MAX_READBACK_BYTES_PER_SUB {
                need_readback = need_readback.saturating_add(dump);
            } else {
                break;
            }
        }
    }
    for ops in &run_ops {
        for op in ops {
            if let DxrOp::ExtractStaticBlas { dest_va, size_bytes } = *op {
                let key = dest_va & !0xFF;
                if key != 0
                    && let Some((_, max_nodes)) = probe_bvh_bounds(key, size_bytes)
                {
                    let dump = (16u64 + max_nodes.saturating_mul(64)).next_multiple_of(64);
                    if need_readback.saturating_add(dump) <= MAX_READBACK_BYTES_PER_SUB {
                        need_readback = need_readback.saturating_add(dump);
                    }
                }
            }
        }
    }
    sub.prepare_node_readback(host, need_readback);

    // Always initialize every newly pre-allocated TLAS slot to a valid 0-instance
    // empty TLAS at `i == 0` before any list in this submission executes (even if
    // a later list in the same submission also rebuilds it).
    let pending_init = std::mem::take(&mut state.pending_empty_tlas);
    let mut built_tlas = false;
    let mut had_static_activity = false;

    for (i, run) in runs.iter().enumerate() {
        let old_start = firsts[i];
        let old_end = firsts.get(i + 1).copied().unwrap_or(lists.len());
        if !use_compute {
            new_firsts.push(new_lists.len());
        }

        if i == 0 {
            if !pending_init.is_empty()
                && let Some(init_list) =
                    record_empty_tlas_init(host, &device5, state, &mut sub, &pending_init, use_compute)
            {
                if use_compute {
                    sub.async_batches.push(DxrAsyncBatch {
                        after_run: None,
                        lists: vec![Some(init_list)],
                        blocks_direct_shaders: true,
                        wrote_guest_postbuild: false,
                    });
                } else {
                    new_lists.push(Some(init_list));
                }
            }

            let mut bg_lists: Vec<Option<ID3D12CommandList>> = Vec::new();
            let mut bg_inited_normals = false;

            if !state.pending_readback_requeue.is_empty()
                && let Some(rb_list) = record_requeued_readbacks(host, state, &mut sub, run, use_compute)
            {
                had_static_activity = true;
                if use_compute {
                    bg_lists.push(Some(rb_list));
                } else {
                    new_lists.push(Some(rb_list));
                }
            }

            let normals_was_uninit = state.normals_init_needed;
            if !state.pending_static_upgrades.is_empty()
                && let Some(upg_list) =
                    record_static_upgrades(host, &device5, state, &mut sub, use_compute)
            {
                had_static_activity = true;
                if normals_was_uninit && !state.normals_init_needed {
                    bg_inited_normals = true;
                }
                if use_compute {
                    bg_lists.push(Some(upg_list));
                } else {
                    new_lists.push(Some(upg_list));
                }
            }

            if !state.pending_compactions.is_empty()
                && let Some(cmp_list) = record_static_compactions(host, state, &mut sub, use_compute)
            {
                had_static_activity = true;
                if use_compute {
                    bg_lists.push(Some(cmp_list));
                } else {
                    new_lists.push(Some(cmp_list));
                }
            }

            if !bg_lists.is_empty() {
                sub.async_batches.push(DxrAsyncBatch {
                    after_run: None,
                    lists: bg_lists,
                    blocks_direct_shaders: bg_inited_normals,
                    wrote_guest_postbuild: false,
                });
            }
        }

        let ops = run_ops.get(i).map(Vec::as_slice).unwrap_or(&[]);
        if ops.is_empty() || old_start >= old_end {
            if !use_compute {
                new_lists.extend(lists[old_start..old_end].iter().cloned());
            }
            continue;
        }
        if ops.iter().any(|op| matches!(op, DxrOp::ExtractStaticBlas { .. })) {
            had_static_activity = true;
        }

        // Run all DXR ops after `run.list` so skinning, buffer copies, and
        // `XboxBvhBuilder.dll` rehydration dispatches in `run.list` have
        // completed before host DXR reads vertex/index/instance/BVH4 buffers.
        if !use_compute {
            new_lists.extend(lists[old_start..old_end].iter().cloned());
        }
        let normals_was_uninit = state.normals_init_needed;
        if let Some(cmd) =
            record_ops(host, &device5, state, &mut sub, run, ops, &mut built_tlas, use_compute)
        {
            if use_compute {
                let inited_normals = normals_was_uninit && !state.normals_init_needed;
                let blocks_direct_shaders = inited_normals
                    || ops.iter().any(|op| {
                        matches!(
                            op,
                            DxrOp::BuildBlas { .. } | DxrOp::BuildTlas { .. } | DxrOp::CopyAs { .. }
                        )
                    });
                let wrote_guest_postbuild = ops.iter().any(|op| match op {
                    DxrOp::BuildBlas { postbuild, .. } | DxrOp::BuildTlas { postbuild, .. } => {
                        !postbuild.is_empty()
                    }
                    DxrOp::CopyAs { .. } | DxrOp::ExtractStaticBlas { .. } => false,
                });
                sub.async_batches.push(DxrAsyncBatch {
                    after_run: Some(i),
                    lists: vec![Some(cmd)],
                    blocks_direct_shaders,
                    wrote_guest_postbuild,
                });
            } else {
                new_lists.push(Some(cmd));
            }
        }
    }

    if sub.lists.is_empty() {
        sub.async_batches.clear();
        sub.readbacks.clear();
        sub.extracted_bvh4s.clear();
        sub.readback_used = 0;
        sub.readback_cap = 0;
        sub.compact_queries.clear();
        sub.kept.clear();
        sub.kept_shadows.clear();
        sub.sub_seq = 0;
        sub.tlas_epoch = 0;
        sub.dxr_fence_val = 0;
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_spans.clear();
            sub.timing_next = 0;
        }
        if state.free_submissions.len() < MAX_FREE_SUBS {
            state.free_submissions.push(sub);
        }
        return None;
    }

    state.sub_seq = state.sub_seq.saturating_add(1);
    CURRENT_SUB_SEQ.store(state.sub_seq, Ordering::Relaxed);
    if had_static_activity
        || !state.pending_static_upgrades.is_empty()
        || !state.pending_compactions.is_empty()
        || !state.pending_readback_requeue.is_empty()
    {
        note_static_streaming(state.sub_seq);
    }
    sub.sub_seq = state.sub_seq;
    sub.tlas_epoch = if built_tlas { state.tlas_build_epoch } else { 0 };

    if !use_compute {
        *lists = new_lists;
        *firsts = new_firsts;
    }
    Some(sub)
}

/// Drains Frame-0 static BLASes whose node readback overflowed an earlier burst
/// submission (`pending_readback_requeue`) so 100% of static BLASes upgrade & compact (`ARCH-10`).
fn record_requeued_readbacks(
    host: &Host,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    owner: &HostList,
    use_compute: bool,
) -> Option<ID3D12CommandList> {
    let pass = extract_bvh4_pass(host)?;
    let mem_binding = crate::descriptor::with_view_heap(|heap| {
        let ready = crate::memory::ready(owner, heap, &[], &[], &[], crate::memory::Use::Reads, None)?;
        Some((heap.heap.clone(), heap.gpu_handle(ready.base), ready.directory))
    })
    .flatten()?;
    let (view_heap, mem_table_gpu, directory_gpu) = mem_binding;

    struct RequeueJob {
        key: u64,
        size_bytes: u64,
        max_nodes: u64,
        generation: u64,
        dump_bytes: u64,
    }
    let mut jobs: Vec<RequeueJob> = Vec::new();
    let mut planned_rb = sub.readback_used;
    let mut max_vert_bytes = 0u64;
    let mut max_dump_bytes = 0u64;
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        while let Some(&(key, size_bytes, generation)) = state.pending_readback_requeue.front() {
            let still_live = map
                .get(&key)
                .is_some_and(|e| e.kind == AsKind::Blas && e.generation == generation);
            if !still_live {
                state.pending_readback_requeue.pop_front();
                continue;
            }
            let Some((bounds_bytes, max_nodes)) = probe_bvh_bounds(key, size_bytes) else {
                state.pending_readback_requeue.pop_front();
                continue;
            };
            let dump_bytes = 16u64 + max_nodes.saturating_mul(64);
            let next_rb = planned_rb.saturating_add(dump_bytes).next_multiple_of(64);
            if next_rb > sub.readback_cap {
                if dump_bytes > MAX_READBACK_BYTES_PER_SUB {
                    // Single dump larger than cap can never fit; release its memory pages and drop.
                    state.pending_readback_requeue.pop_front();
                    sub.extracted_bvh4s.push((key, bounds_bytes));
                    continue;
                }
                break;
            }
            state.pending_readback_requeue.pop_front();
            planned_rb = next_rb;
            max_vert_bytes = max_vert_bytes.max(max_nodes.saturating_mul(72));
            max_dump_bytes = max_dump_bytes.max(dump_bytes);
            jobs.push(RequeueJob {
                key,
                size_bytes: bounds_bytes,
                max_nodes,
                generation,
                dump_bytes,
            });
        }
    }
    if jobs.is_empty() {
        return None;
    }
    let (rb_buf, _) = sub.node_readback.as_ref().map(|(b, sz, _)| (b.clone(), *sz))?;
    let (_, vert_gpu) = sub.ensure_extract_verts(host, max_vert_bytes)?;
    let (staging_buf, staging_gpu) = sub.ensure_node_staging(host, max_dump_bytes)?;
    let (allocator, list) = state.take_list(host, use_compute)?;
    name(&list, || "victory dxr requeued readback".into());
    let list7 = dxr_list7(host, &list);

    let rb_start = sub.readbacks.len();
    let rb_used_start = sub.readback_used;
    let ext_start = sub.extracted_bvh4s.len();

    // SAFETY: idle allocator/list; view_heap, directory, vert_gpu, staging_buf, and rb_buf live.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            list.SetDescriptorHeaps(&[Some(view_heap)]);
            list.SetComputeRootSignature(&pass.root);
            list.SetPipelineState(&pass.pso);
            list.SetComputeRootShaderResourceView(1, directory_gpu);
            list.SetComputeRootUnorderedAccessView(2, vert_gpu);
            list.SetComputeRootUnorderedAccessView(3, staging_gpu);
            list.SetComputeRootDescriptorTable(4, mem_table_gpu);
            let mut staging_in_copy_src = false;
            for job in &jobs {
                if staging_in_copy_src {
                    emit_dxr_buffer_barrier(
                        &list,
                        list7.as_ref(),
                        &staging_buf,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        use_compute,
                    );
                } else {
                    emit_dxr_buffer_legacy_only(
                        &list,
                        list7.as_ref(),
                        &staging_buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    );
                }
                let constants = [
                    job.key as u32,
                    (job.key >> 32) as u32,
                    job.max_nodes as u32,
                    job.size_bytes.min(u64::from(u32::MAX)) as u32,
                    job.max_nodes as u32,
                ];
                list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                list.Dispatch((job.max_nodes as u32).div_ceil(64), 1, 1);
                emit_dxr_buffer_barrier(
                    &list,
                    list7.as_ref(),
                    &staging_buf,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    use_compute,
                );
                let rb_off = sub.readback_used;
                sub.readback_used = rb_off.saturating_add(job.dump_bytes).next_multiple_of(64);
                list.CopyBufferRegion(&rb_buf, rb_off, &staging_buf, 0, job.dump_bytes);
                staging_in_copy_src = true;
                sub.readbacks.push(StaticReadbackRecord {
                    guest_key: job.key,
                    generation: job.generation,
                    offset: rb_off,
                    len: job.dump_bytes,
                });
                sub.extracted_bvh4s.push((job.key, job.size_bytes));
            }
            if staging_in_copy_src {
                emit_dxr_buffer_legacy_only(
                    &list,
                    list7.as_ref(),
                    &staging_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: requeued static readback list failed");
        sub.readbacks.truncate(rb_start);
        sub.readback_used = rb_used_start;
        sub.extracted_bvh4s.truncate(ext_start);
        state.return_list((allocator, list), use_compute);
        return None;
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

struct DecodeJob {
    readbacks: Vec<StaticReadbackRecord>,
    bytes: Vec<u8>,
}

/// Non-blocking decode queue + bounded overflow re-queue ring so Frame-0 static BLAS
/// bursts never drop upgrades on `try_send` or `MAX_PENDING_UPGRADES` (`ARCH-10`).
struct DecodePipeline {
    jobs: Mutex<VecDeque<DecodeJob>>,
    wake: Condvar,
}

fn decode_pipeline() -> &'static DecodePipeline {
    static PIPE: OnceLock<DecodePipeline> = OnceLock::new();
    PIPE.get_or_init(|| {
        let _ = std::thread::Builder::new()
            .name("victory-dxr-decode".into())
            .spawn(|| {
                let pipe = decode_pipeline();
                let mut overflow: VecDeque<PendingStaticUpgrade> = VecDeque::new();
                loop {
                    // First flush any previously overflowed decoded upgrades whenever
                    // `pending_static_upgrades` has room.
                    if !overflow.is_empty() {
                        let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(state) = guard.as_mut() {
                            while state.pending_static_upgrades.len() < MAX_PENDING_UPGRADES {
                                let Some(upg) = overflow.pop_front() else {
                                    break;
                                };
                                state.pending_static_upgrades.push_back(upg);
                            }
                        }
                    }

                    let job = {
                        let mut q = pipe.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            // If overflow ring is near full, wait briefly for submit thread to drain
                            // `pending_static_upgrades` before decoding more jobs.
                            if overflow.len() >= MAX_OVERFLOW_UPGRADES {
                                let (g, _) = pipe
                                    .wake
                                    .wait_timeout(q, std::time::Duration::from_millis(8))
                                    .unwrap_or_else(|e| e.into_inner());
                                q = g;
                                break None;
                            }
                            if let Some(job) = q.pop_front() {
                                break Some(job);
                            }
                            if !overflow.is_empty() {
                                let (g, _) = pipe
                                    .wake
                                    .wait_timeout(q, std::time::Duration::from_millis(8))
                                    .unwrap_or_else(|e| e.into_inner());
                                q = g;
                                break None;
                            }
                            q = pipe.wake.wait(q).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    let Some(job) = job else {
                        continue;
                    };

                    let live: Vec<StaticReadbackRecord> = {
                        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
                        job.readbacks
                            .into_iter()
                            .filter(|rec| {
                                map.get(&rec.guest_key)
                                    .is_some_and(|e| e.generation == rec.generation)
                            })
                            .collect()
                    };
                    let mut decoded = Vec::new();
                    for rec in live {
                        let start = rec.offset as usize;
                        let end = start.saturating_add(rec.len as usize);
                        if end > job.bytes.len() {
                            continue;
                        }
                        if let Some(upg) =
                            decode_static_bvh4_nodes(rec.guest_key, rec.generation, &job.bytes[start..end])
                        {
                            decoded.push(upg);
                        }
                    }
                    if !decoded.is_empty() {
                        let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(state) = guard.as_mut() {
                            for upg in decoded {
                                if state.pending_static_upgrades.len() < MAX_PENDING_UPGRADES {
                                    state.pending_static_upgrades.push_back(upg);
                                } else if overflow.len() < MAX_OVERFLOW_UPGRADES {
                                    overflow.push_back(upg);
                                }
                            }
                        }
                    }
                }
            });
        DecodePipeline {
            jobs: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
        }
    })
}

fn enqueue_decode_job(job: DecodeJob) {
    let pipe = decode_pipeline();
    let mut q = pipe.jobs.lock().unwrap_or_else(|e| e.into_inner());
    if q.len() < MAX_DECODE_QUEUE_JOBS {
        q.push_back(job);
        pipe.wake.notify_one();
    }
}

fn wake_decode_worker() {
    let pipe = decode_pipeline();
    pipe.wake.notify_one();
}

/// Returns pooled DXR command allocators and scratch resources once the
/// submission's GPU fence has completed, reclaims eligible retired AS slab slices,
/// releases extracted ephemeral `memory.rs` arena pages, trims idle staging buffers,
/// and offloads any completed static BVH4 node readbacks to the background decoder.
pub(crate) fn finish_submission(mut sub: DxrSubmission) {
    // Ensure async-compute DXR work (if any) has completed on `compute_queue` before
    // reading back GPU timestamps, compaction sizes, node dumps, or recycling lists/slices.
    let dxr_fence_val = sub.dxr_fence_val;
    let ran_on_compute = sub.ran_on_compute;
    sub.dxr_fence_val = 0;
    sub.async_batches.clear();
    if dxr_fence_val > 0
        && let Some(h) = host()
    {
        h.wait_dxr_fence_cpu(dxr_fence_val);
    }
    #[cfg(feature = "diagnostics")]
    sub.report_timing();
    sub.kept_shadows.clear();
    let readbacks = std::mem::take(&mut sub.readbacks);
    let extracted_bvh4s = std::mem::take(&mut sub.extracted_bvh4s);
    let readback_used = sub.readback_used;
    sub.readback_used = 0;
    sub.readback_cap = 0;
    let compact_queries = std::mem::take(&mut sub.compact_queries);
    let completed_sub_seq = sub.sub_seq;
    let completed_tlas_epoch = sub.tlas_epoch;
    sub.sub_seq = 0;
    sub.tlas_epoch = 0;

    let mut new_compactions: Vec<(u64, u64, u64)> = Vec::new();
    if !compact_queries.is_empty()
        && let Some((rb_buf, rb_size, _)) = sub.compact_info_rb.as_ref()
    {
        let max_slots = compact_queries
            .iter()
            .map(|q| u64::from(q.slot) + 1)
            .max()
            .unwrap_or(0);
        let map_end = (max_slots.saturating_mul(8).min(*rb_size)) as usize;
        if map_end >= 8 {
            let mut ptr = std::ptr::null_mut();
            let read_range = D3D12_RANGE { Begin: 0, End: map_end };
            // SAFETY: submission GPU fence has completed; readback buffer is idle.
            let mapped = unsafe { rb_buf.Map(0, Some(&read_range), Some(&mut ptr)) };
            if mapped.is_ok() && !ptr.is_null() {
                // SAFETY: `ptr` is valid for `map_end` bytes until `Unmap`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), map_end) };
                let sizes: Vec<u64> = compact_queries
                    .iter()
                    .map(|q| {
                        let off = (q.slot as usize).saturating_mul(8);
                        if off + 8 <= bytes.len() {
                            u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
                        } else {
                            0
                        }
                    })
                    .collect();
                let written_none = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: matches successful `Map(0, ...)`.
                unsafe { rb_buf.Unmap(0, Some(&written_none)) };

                const ALLOW_COMPACTION: u32 =
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
                let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
                for (q, compacted_size) in compact_queries.into_iter().zip(sizes) {
                    if compacted_size == 0 {
                        continue;
                    }
                    if let Some(e) = map.get_mut(&q.guest_key)
                        && e.kind == AsKind::Blas
                        && e.built
                        && e.generation == q.generation
                        && (e.flags & ALLOW_COMPACTION) != 0
                    {
                        e.compacted_size = compacted_size;
                        if q.auto_compact
                            && should_compact(q.uncompacted_size, compacted_size)
                            && should_compact(e.slice.size, compacted_size)
                        {
                            new_compactions.push((q.guest_key, q.generation, compacted_size));
                        }
                    }
                }
            }
        }
    }

    if !readbacks.is_empty()
        && let Some((rb_buf, rb_size, _)) = sub.node_readback.as_ref()
    {
        let map_end = (readback_used.min(*rb_size)) as usize;
        if map_end >= 16 {
            let mut ptr = std::ptr::null_mut();
            let read_range = D3D12_RANGE { Begin: 0, End: map_end };
            // SAFETY: submission GPU fence has completed; readback buffer is idle.
            let mapped = unsafe { rb_buf.Map(0, Some(&read_range), Some(&mut ptr)) };
            if mapped.is_ok() && !ptr.is_null() {
                // SAFETY: `ptr` is valid for `map_end` bytes until `Unmap`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), map_end) }.to_vec();
                let written_none = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: matches successful `Map(0, ...)`.
                unsafe { rb_buf.Unmap(0, Some(&written_none)) };
                enqueue_decode_job(DecodeJob { readbacks, bytes });
            }
        }
    }

    // Release ephemeral `memory.rs` BVH4 rehydration arena pages once extraction + node
    // readback has completed on the GPU (strictly gated on `VICTORY_NATIVE_RT=1 && VICTORY_BVH_BUILDER=0`).
    for (dest_va, size_bytes) in extracted_bvh4s {
        crate::memory::release_extracted(dest_va, size_bytes);
    }

    // Drop oversized temporary buffers and static staging buffers idle for >= 60 submissions
    // asynchronously on `victory-bury` so `victory-relay` never stalls on COM drops.
    sub.trim_pooled(completed_sub_seq);

    let completed_dxr_fence = host().map_or(u64::MAX, Host::dxr_fence_completed);
    let mut guard = DXR_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_mut() else {
        return;
    };
    state.completed_sub_seq = state.completed_sub_seq.max(completed_sub_seq);
    state.completed_tlas_epoch = state.completed_tlas_epoch.max(completed_tlas_epoch);
    state.slabs.drain_retired(
        state.completed_tlas_epoch,
        state.completed_sub_seq,
        completed_dxr_fence,
    );
    state.slabs.reclaim_empty_slabs();
    state.geom_pool.drain_retired(
        state.completed_tlas_epoch,
        state.completed_sub_seq,
        completed_dxr_fence,
    );
    state.data_pool.drain_retired(
        state.completed_tlas_epoch,
        state.completed_sub_seq,
        completed_dxr_fence,
    );
    sweep_unreachable_blases(state);
    for item in new_compactions {
        if state.pending_compactions.len() < MAX_PENDING_COMPACTIONS {
            state.pending_compactions.push_back(item);
        }
    }
    for prelude in sub.lists.drain(..) {
        state.return_list(prelude, ran_on_compute);
    }
    if state.free_submissions.len() < MAX_FREE_SUBS {
        state.free_submissions.push(sub);
    } else {
        crate::host::retire_com_async(sub);
    }
}

/// Drains up to `MAX_UPGRADES_PER_SUB` decoded multi-geometry static BLAS upgrades
/// and builds them into host AS slabs, updating `AS_MAP` and the GPU BLAS hash table.
fn record_static_upgrades(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    use_compute: bool,
) -> Option<ID3D12CommandList> {
    let mut batch: Vec<(PendingStaticUpgrade, u64)> = Vec::new();
    let mut batch_bytes = 0u64;
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        while batch.len() < MAX_UPGRADES_PER_SUB {
            let Some(front) = state.pending_static_upgrades.front() else {
                break;
            };
            let vert_bytes = (front.vertices.len() as u64).saturating_mul(12);
            let aligned_off = batch_bytes.next_multiple_of(16);
            let next_bytes = aligned_off.saturating_add(vert_bytes);
            if !batch.is_empty() && next_bytes > MAX_UPGRADE_BATCH_BYTES {
                break;
            }
            let Some(upg) = state.pending_static_upgrades.pop_front() else {
                break;
            };
            let still_current = map
                .get(&upg.guest_key)
                .is_some_and(|e| e.generation == upg.generation);
            if !still_current || upg.geometries.is_empty() || upg.vertices.is_empty() {
                continue;
            }
            batch.push((upg, aligned_off));
            batch_bytes = next_bytes;
        }
    }
    if batch.is_empty() || batch_bytes == 0 {
        wake_decode_worker();
        return None;
    }
    wake_decode_worker();

    let (upload_buf, upload_ptr) = sub.ensure_upgrade_upload(host, batch_bytes)?;
    let (verts_buf, verts_gpu) = sub.ensure_upgrade_verts(host, batch_bytes)?;

    let build_flags = static_blas_build_flags();
    struct PreparedUpgrade {
        guest_key: u64,
        host_geoms: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
        result_bytes: u64,
        scratch_bytes: u64,
        geom_table: Vec<u32>,
        normals_data: Vec<u32>,
    }
    let mut prepared = Vec::with_capacity(batch.len());
    let mut max_scratch = AS_ALIGN;
    let mut sum_scratch = 0u64;
    for (upg, batch_off) in batch {
        let byte_len = upg.vertices.len() * 12;
        // SAFETY: `upload_ptr` is mapped for at least `batch_bytes >= batch_off + byte_len`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                upg.vertices.as_ptr().cast::<u8>(),
                upload_ptr.add(batch_off as usize),
                byte_len,
            );
        }
        let base_gpu = verts_gpu + batch_off;
        let mut host_geoms = Vec::with_capacity(upg.geometries.len());
        for g in &upg.geometries {
            let flags = static_geom_flags(g.opaque);
            let (start_addr, stride) = if g.vert_count > 0 {
                (base_gpu + u64::from(g.vert_offset) * 12, 12u64)
            } else {
                (0, 0)
            };
            host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: flags,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_UNKNOWN,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 0,
                        VertexCount: g.vert_count,
                        IndexBuffer: 0,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: start_addr,
                            StrideInBytes: stride,
                        },
                    },
                },
            });
        }
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: build_flags,
            NumDescs: host_geoms.len() as u32,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: host_geoms.as_ptr(),
            },
        };
        let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: valid device5 and `host_geoms` alive through call.
        unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
        if pre.ResultDataMaxSizeInBytes == 0 {
            continue;
        }
        let scratch_bytes = pre.ScratchDataSizeInBytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
        max_scratch = max_scratch.max(scratch_bytes);
        sum_scratch = sum_scratch.saturating_add(scratch_bytes);
        // Reuse pre-packed octahedral normals computed off-thread on `victory-dxr-decode` (`ARCH-14`).
        prepared.push(PreparedUpgrade {
            guest_key: upg.guest_key,
            host_geoms,
            result_bytes: pre.ResultDataMaxSizeInBytes,
            scratch_bytes,
            geom_table: upg.geom_table,
            normals_data: upg.normals_data,
        });
    }
    if prepared.is_empty() {
        return None;
    }

    let normals_gpu_buf = state.ensure_normals_gpu(host).map(|(b, _)| b);

    // Allocate fresh slab slices + normals table slices and swap into `AS_MAP` in one batched write lock,
    // deferring reclamation of replaced Frame-0 slices until 2 TLAS epochs complete.
    let mut built_items: Vec<(usize, u64, u64, u64, u64, u32, u32)> =
        Vec::with_capacity(prepared.len());
    let mut normal_copies: Vec<(u64, u64, u64)> = Vec::new();
    let mut normals_upload_words: Vec<u32> = Vec::new();
    if normals_gpu_buf.is_some() && state.normals_init_needed {
        normal_copies.push((0, 0, 4));
        normals_upload_words.push(victory_shader::emit::DXR_NORMALS_FALLBACK);
    }
    {
        let cur_epoch = state.tlas_build_epoch;
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for (idx, item) in prepared.iter().enumerate() {
            // Allocate a fresh transient slab slice so any already-built TLAS tracing the
            // previous frame-0 1-geometry BLAS never sees an in-place overwrite
            // before the next `BuildTlas` switches instances to the upgraded VA.
            let Some((buffer, slice, gpu_va)) =
                state.slabs.alloc(host, SlabTier::TransientDynamicTlas, item.result_bytes)
            else {
                continue;
            };
            let (ng_off, ng_len, nd_off, nd_len) = if normals_gpu_buf.is_some()
                && !item.geom_table.is_empty()
                && !item.normals_data.is_empty()
            {
                let gl = item.geom_table.len() as u32;
                let dl = item.normals_data.len() as u32;
                if let (Some(go), Some(doff)) = (state.geom_pool.alloc(gl), state.data_pool.alloc(dl))
                {
                    let src_g = (normals_upload_words.len() as u64) * 4;
                    for &rel in &item.geom_table {
                        normals_upload_words.push(if rel == u32::MAX { 0 } else { doff + rel });
                    }
                    normal_copies.push((u64::from(go) * 4, src_g, u64::from(gl) * 4));
                    let src_d = (normals_upload_words.len() as u64) * 4;
                    normals_upload_words.extend_from_slice(&item.normals_data);
                    normal_copies.push((u64::from(doff) * 4, src_d, u64::from(dl) * 4));
                    (go, gl, doff, dl)
                } else {
                    (0, 0, 0, 0)
                }
            } else {
                (0, 0, 0, 0)
            };
            let generation = state.bump_gen();
            let slice_size = slice.size;
            let entry = HostAs {
                buffer,
                slice,
                gpu_va,
                mirror_tlas: None,
                kind: AsKind::Blas,
                flags: build_flags.0 as u32,
                num_instances: 0,
                built: true,
                generation,
                topo_hash: 0,
                update_scratch_bytes: 0,
                consecutive_refits: 0,
                compacted_size: 0,
                last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
                geom_va_hash: 0,
                normals_geom_off: ng_off,
                normals_geom_len: ng_len,
                normals_data_off: nd_off,
                normals_data_len: nd_len,
            };
            if let Some(old) = map.insert(item.guest_key, entry) {
                state.retire_as(&old);
            }
            built_items.push((idx, item.guest_key, gpu_va, generation, slice_size, ng_off, nd_off));
        }
    }
    if built_items.is_empty() {
        return None;
    }

    let normals_upload = if !normals_upload_words.is_empty() && normals_gpu_buf.is_some() {
        let bytes = (normals_upload_words.len() as u64) * 4;
        if let Some((nbuf, nptr)) = sub.ensure_normals_upload(host, bytes) {
            // SAFETY: `nptr` is mapped for at least `bytes`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    normals_upload_words.as_ptr().cast::<u8>(),
                    nptr,
                    bytes as usize,
                );
            }
            Some(nbuf)
        } else {
            None
        }
    } else {
        None
    };

    let want_scratch = sum_scratch.min(MAX_POOLED_SCRATCH_BYTES).max(max_scratch);
    let (scratch_buf, scratch_gpu, scratch_cap) = sub.ensure_scratch(host, want_scratch)?;
    let scratch_id = scratch_buf.as_raw() as usize;
    let compact_info = if compact_static_blas() && sub.compact_queries.len() < COMPACT_INFO_SLOTS {
        sub.ensure_compact_info(host)
    } else {
        None
    };
    let (allocator, list) = state.take_list(host, use_compute)?;
    name(&list, || "victory dxr static upgrade".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;
    let list7 = dxr_list7(host, &list);

    let mut new_queries: Vec<CompactQueryRecord> = Vec::new();
    let first_slot = sub.compact_queries.len();
    let mut scratch_batch = ScratchBatcher::default();
    #[cfg(feature = "diagnostics")]
    let (timing, timing_base, spans_base) = (sub.ensure_timing(host), sub.timing_next, sub.timing_spans.len());

    // SAFETY: idle allocator/list; upload, vertex, scratch, compact info, and slab buffers live through submission.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            #[cfg(feature = "diagnostics")]
            let t_start = timing.as_ref().and_then(|(h, _)| sub.stamp(&list, h));
            list.CopyBufferRegion(&verts_buf, 0, &upload_buf, 0, batch_bytes);
            if let (Some(n_dst), Some(n_src)) = (&normals_gpu_buf, &normals_upload) {
                for &(dst_off, src_off, len) in &normal_copies {
                    list.CopyBufferRegion(n_dst, dst_off, n_src, src_off, len);
                }
                emit_dxr_buffer_barrier(
                    &list,
                    list7.as_ref(),
                    n_dst,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_ALL_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_DEST,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                    use_compute,
                );
            }
            emit_dxr_buffer_barrier(
                &list,
                list7.as_ref(),
                &verts_buf,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_ACCESS_COPY_DEST,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                use_compute,
            );
            if let Some((uav_buf, _, _)) = &compact_info {
                emit_dxr_buffer_legacy_only(
                    &list,
                    list7.as_ref(),
                    uav_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
            for &(idx, guest_key, gpu_va, generation, slice_size, _, _) in &built_items {
                let item = &prepared[idx];
                let (need_barrier, scratch_off) =
                    scratch_batch.alloc(scratch_id, scratch_cap, item.scratch_bytes, gpu_va, 0);
                if need_barrier {
                    emit_as_build_barrier(&list, list7.as_ref(), use_compute);
                }
                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                    Flags: build_flags,
                    NumDescs: item.host_geoms.len() as u32,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: item.host_geoms.as_ptr(),
                    },
                };
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: gpu_va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gpu + scratch_off,
                };
                let postbuild = if let Some((_, uav_gpu, _)) = &compact_info
                    && first_slot + new_queries.len() < COMPACT_INFO_SLOTS
                {
                    let slot = (first_slot + new_queries.len()) as u32;
                    new_queries.push(CompactQueryRecord {
                        guest_key,
                        generation,
                        uncompacted_size: slice_size,
                        slot,
                        auto_compact: true,
                    });
                    Some([D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                        DestBuffer: uav_gpu + u64::from(slot) * 8,
                        InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
                    }])
                } else {
                    None
                };
                list4.BuildRaytracingAccelerationStructure(
                    &desc,
                    postbuild.as_ref().map(|p| p.as_slice()),
                );
            }
            scratch_batch.flush(&list, list7.as_ref(), use_compute);
            if let Some((uav_buf, _, rb_buf)) = &compact_info {
                if !new_queries.is_empty() {
                    emit_dxr_buffer_barrier(
                        &list,
                        list7.as_ref(),
                        uav_buf,
                        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        use_compute,
                    );
                    let off = (first_slot as u64) * 8;
                    let len = (new_queries.len() as u64) * 8;
                    list.CopyBufferRegion(rb_buf, off, uav_buf, off, len);
                    emit_dxr_buffer_legacy_only(
                        &list,
                        list7.as_ref(),
                        uav_buf,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                } else {
                    emit_dxr_buffer_legacy_only(
                        &list,
                        list7.as_ref(),
                        uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                }
            }
            emit_dxr_buffer_legacy_only(
                &list,
                list7.as_ref(),
                &verts_buf,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            #[cfg(feature = "diagnostics")]
            if let Some((h, rb)) = timing.as_ref() {
                let t_end = sub.stamp(&list, h);
                if let (Some(s), Some(e)) = (t_start, t_end) {
                    sub.timing_spans.push(("dxr [static_upgrade]", s, e));
                }
                sub.resolve_timing(&list, h, rb, timing_base);
            }
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: static BLAS upgrade list failed");
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_next = timing_base;
            sub.timing_spans.truncate(spans_base);
        }
        state.return_list((allocator, list), use_compute);
        return None;
    }
    if normals_upload.is_some() {
        state.normals_init_needed = false;
    }
    sub.compact_queries.extend(new_queries);
    for &(_, guest_key, host_va, _, _, ng_off, _) in &built_items {
        state.insert_blas_hash(guest_key, host_va, ng_off);
    }
    let upg_geoms: u64 = built_items
        .iter()
        .map(|&(idx, _, _, _, _, _, _)| prepared[idx].host_geoms.len() as u64)
        .sum();
    let upg_non_opaque: u64 = built_items
        .iter()
        .flat_map(|&(idx, _, _, _, _, _, _)| prepared[idx].host_geoms.iter())
        .filter(|g| (g.Flags.0 & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE.0) == 0)
        .count() as u64;
    STATS[STAT_UPG_GEOMS].fetch_add(upg_geoms, Ordering::Relaxed);
    STATS[STAT_UPG_NON_OPAQUE].fetch_add(upg_non_opaque, Ordering::Relaxed);
    STATS[STAT_UPGRADED_STATIC].fetch_add(built_items.len() as u64, Ordering::Relaxed);
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

/// Drains up to `MAX_COMPACTIONS_PER_SUB` queued static BLAS compactions, copies them
/// into smaller slab slices via `COPY_MODE_COMPACT`, updates `AS_MAP` and the GPU hash
/// table, and retires the old uncompacted slices under the `tlas_build_epoch + 2` invariant.
fn record_static_compactions(
    host: &Host,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    use_compute: bool,
) -> Option<ID3D12CommandList> {
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    struct PreparedCompaction {
        guest_key: u64,
        generation: u64,
        src_va: u64,
        dst_buf: ID3D12Resource,
        dst_slice: SlabSlice,
        dst_va: u64,
        flags: u32,
    }
    let mut batch: Vec<PreparedCompaction> = Vec::new();
    {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        while batch.len() < MAX_COMPACTIONS_PER_SUB {
            let Some((guest_key, generation, compacted_size)) = state.pending_compactions.pop_front() else {
                break;
            };
            if batch.iter().any(|b| b.guest_key == guest_key) {
                continue;
            }
            let Some(entry) = map.get(&guest_key) else {
                continue;
            };
            if entry.kind != AsKind::Blas
                || !entry.built
                || entry.generation != generation
                || (entry.flags & ALLOW_COMPACTION) == 0
                || !should_compact(entry.slice.size, compacted_size)
            {
                continue;
            }
            let Some((dst_buf, dst_slice, dst_va)) =
                state.slabs.alloc(host, SlabTier::StaticCompacted, compacted_size)
            else {
                break;
            };
            if dst_slice.size >= entry.slice.size {
                state.slabs.free_now(dst_slice);
                continue;
            }
            batch.push(PreparedCompaction {
                guest_key,
                generation,
                src_va: entry.gpu_va,
                dst_buf,
                dst_slice,
                dst_va,
                flags: entry.flags & !ALLOW_COMPACTION,
            });
        }
    }
    if batch.is_empty() {
        return None;
    }

    let Some((allocator, list)) = state.take_list(host, use_compute) else {
        for item in batch {
            state.slabs.free_now(item.dst_slice);
        }
        return None;
    };
    name(&list, || "victory dxr static compact".into());
    let Ok(list4) = list.cast::<ID3D12GraphicsCommandList4>() else {
        for item in batch {
            state.slabs.free_now(item.dst_slice);
        }
        state.return_list((allocator, list), use_compute);
        return None;
    };
    let list7 = dxr_list7(host, &list);

    #[cfg(feature = "diagnostics")]
    let (timing, timing_base, spans_base) = (sub.ensure_timing(host), sub.timing_next, sub.timing_spans.len());

    // SAFETY: idle allocator/list; all source and destination AS slabs stay resident and
    // source slices remain unretired until `tlas_build_epoch + 2` and `sub_seq + 2`.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            #[cfg(feature = "diagnostics")]
            let t_start = timing.as_ref().and_then(|(h, _)| sub.stamp(&list, h));
            for item in &batch {
                list4.CopyRaytracingAccelerationStructure(
                    item.dst_va,
                    item.src_va,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
                );
            }
            emit_as_copy_barrier(&list, list7.as_ref(), use_compute);
            #[cfg(feature = "diagnostics")]
            if let Some((h, rb)) = timing.as_ref() {
                let t_end = sub.stamp(&list, h);
                if let (Some(s), Some(e)) = (t_start, t_end) {
                    sub.timing_spans.push(("dxr [static_compact]", s, e));
                }
                sub.resolve_timing(&list, h, rb, timing_base);
            }
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: static BLAS compaction list failed");
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_next = timing_base;
            sub.timing_spans.truncate(spans_base);
        }
        for item in batch {
            state.slabs.free_now(item.dst_slice);
        }
        state.return_list((allocator, list), use_compute);
        return None;
    }

    let mut swapped = 0u64;
    let cur_epoch = state.tlas_build_epoch;
    {
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for item in batch {
            let Some((old_gen, old_topo_hash, old_update_scratch, ng_off, ng_len, nd_off, nd_len)) =
                map.get(&item.guest_key)
                    .filter(|cur| cur.generation == item.generation)
                    .map(|cur| {
                        (
                            cur.generation,
                            cur.topo_hash,
                            cur.update_scratch_bytes,
                            cur.normals_geom_off,
                            cur.normals_geom_len,
                            cur.normals_data_off,
                            cur.normals_data_len,
                        )
                    })
            else {
                state.slabs.free_now(item.dst_slice);
                continue;
            };
            // Preserve `generation` and normals table ownership across compaction.
            if let Some(mut old) = map.insert(
                item.guest_key,
                HostAs {
                    buffer: item.dst_buf,
                    slice: item.dst_slice,
                    gpu_va: item.dst_va,
                    mirror_tlas: None,
                    kind: AsKind::Blas,
                    flags: item.flags,
                    num_instances: 0,
                    built: true,
                    generation: old_gen,
                    topo_hash: old_topo_hash,
                    update_scratch_bytes: old_update_scratch,
                    consecutive_refits: 0,
                    compacted_size: 0,
                    last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
                    geom_va_hash: 0,
                    normals_geom_off: ng_off,
                    normals_geom_len: ng_len,
                    normals_data_off: nd_off,
                    normals_data_len: nd_len,
                },
            ) {
                old.normals_geom_len = 0;
                old.normals_data_len = 0;
                state.retire_as(&old);
            }
            state.insert_blas_hash(item.guest_key, item.dst_va, ng_off);
            swapped += 1;
        }
    }
    if swapped > 0 {
        STATS[STAT_COMPACTED].fetch_add(swapped, Ordering::Relaxed);
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

fn uav_barrier(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: non-owning pointer for the lifetime of the barrier call.
                pResource: ManuallyDrop::new(unsafe {
                    match resource {
                        Some(r) => Some(std::mem::transmute_copy(r)),
                        None => None,
                    }
                }),
            }),
        },
    }
}

/// Strips graphics-only `D3D12_BARRIER_SYNC_*` bits when emitting onto a
/// `D3D12_COMMAND_LIST_TYPE_COMPUTE` command list so the D3D12 Debug Layer never
/// observes illegal graphics sync bits (`PIXEL_SHADING`, `VERTEX_SHADING`,
/// `RENDER_TARGET`, `DEPTH_STENCIL`, `ALL_SHADING`, `NON_PIXEL_SHADING`, `DRAW`,
/// `INDEX_INPUT`, `RESOLVE`) on `Host::compute_queue`.
fn sanitize_sync_for_queue(sync: D3D12_BARRIER_SYNC, is_compute_queue: bool) -> D3D12_BARRIER_SYNC {
    if !is_compute_queue || sync.0 == 0 {
        return sync;
    }
    let raw = sync.0 as u32;
    let compute_legal: u32 = (D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 as u32)
        | (D3D12_BARRIER_SYNC_COPY.0 as u32)
        | (D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32)
        | (D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32)
        | (D3D12_BARRIER_SYNC_EMIT_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO.0 as u32)
        | (D3D12_BARRIER_SYNC_EXECUTE_INDIRECT.0 as u32)
        | (D3D12_BARRIER_SYNC_CLEAR_UNORDERED_ACCESS_VIEW.0 as u32);
    let shading_any: u32 = (D3D12_BARRIER_SYNC_ALL_SHADING.0 as u32)
        | (D3D12_BARRIER_SYNC_NON_PIXEL_SHADING.0 as u32)
        | (D3D12_BARRIER_SYNC_PIXEL_SHADING.0 as u32)
        | (D3D12_BARRIER_SYNC_VERTEX_SHADING.0 as u32)
        | (D3D12_BARRIER_SYNC_DRAW.0 as u32)
        | (D3D12_BARRIER_SYNC_RAYTRACING.0 as u32);
    let mut out = raw & compute_legal;
    if (raw & shading_any) != 0 {
        out |= D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 as u32;
    }
    if out == 0 {
        out = D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 as u32;
    }
    D3D12_BARRIER_SYNC(out as i32)
}

/// Computes the narrowed Enhanced prelude barrier `(sync_before, sync_after, access_before, access_after)`
/// for a batch of `ops` on `DirectQ` (`None` on `ComputeQ`, where `skin_fence` already provides
/// full Direct -> Compute synchronization and L2 flush).
fn prelude_barrier_for_ops(
    has_build_blas: bool,
    has_build_tlas: bool,
    has_extract: bool,
    has_copy: bool,
    is_compute_queue: bool,
) -> Option<(
    D3D12_BARRIER_SYNC,
    D3D12_BARRIER_SYNC,
    D3D12_BARRIER_ACCESS,
    D3D12_BARRIER_ACCESS,
)> {
    if is_compute_queue || !(has_build_blas || has_build_tlas || has_extract || has_copy) {
        return None;
    }
    // Preceding `run.list` work on DirectQ is skinning/rehydration compute (`COMPUTE_SHADING`)
    // and buffer copies (`COPY`); never drain graphics/raster stages or already-flushed AS builds.
    let mut sync_before = D3D12_BARRIER_SYNC_COMPUTE_SHADING | D3D12_BARRIER_SYNC_COPY;
    let mut access_before =
        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_COPY_DEST;
    if has_copy {
        sync_before |= D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE;
        access_before |= D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
    }
    let mut sync_after = D3D12_BARRIER_SYNC_NONE;
    let mut access_after = D3D12_BARRIER_ACCESS(0);
    if has_build_blas || has_build_tlas || has_extract {
        sync_after |= D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
            | D3D12_BARRIER_SYNC_COMPUTE_SHADING;
        access_after |= D3D12_BARRIER_ACCESS_SHADER_RESOURCE
            | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
            | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
            | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
    }
    if has_copy {
        sync_after |= D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE;
        access_after |= D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
            | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
    }
    Some((sync_before, sync_after, access_before, access_after))
}

/// Emits a scoped Enhanced `D3D12_BUFFER_BARRIER` on `list7` (sanitizing sync bits for
/// `ComputeQ`), or falls back to a legacy `D3D12_RESOURCE_BARRIER_TYPE_TRANSITION` when
/// Enhanced Barriers are unavailable.
#[allow(clippy::too_many_arguments)]
unsafe fn emit_dxr_buffer_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
    legacy_before: D3D12_RESOURCE_STATES,
    legacy_after: D3D12_RESOURCE_STATES,
    is_compute_queue: bool,
) {
    if let Some(l7) = list7 {
        let barrier = [D3D12_BUFFER_BARRIER {
            SyncBefore: sanitize_sync_for_queue(sync_before, is_compute_queue),
            SyncAfter: sanitize_sync_for_queue(sync_after, is_compute_queue),
            AccessBefore: access_before,
            AccessAfter: access_after,
            // SAFETY: borrowed COM pointer valid for the `Barrier` call; caller keeps resource alive.
            pResource: unsafe { std::mem::transmute_copy(resource) },
            Offset: 0,
            Size: u64::MAX,
        }];
        let group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_BUFFER,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pBufferBarriers: barrier.as_ptr(),
            },
        }];
        // SAFETY: caller guarantees an open command list and spec-valid sync/access flags.
        unsafe { l7.Barrier(&group) };
    } else if legacy_before != legacy_after {
        // SAFETY: caller guarantees an open command list and live buffer.
        unsafe { list.ResourceBarrier(&[transition(resource, legacy_before, legacy_after)]) };
    }
}

/// Emits a legacy buffer state transition ONLY when Enhanced Barriers (`list7`) are unavailable
/// (under Enhanced Barriers, buffers remain in `COMMON` and need no `COMMON <-> state` noise).
unsafe fn emit_dxr_buffer_legacy_only(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    resource: &ID3D12Resource,
    legacy_before: D3D12_RESOURCE_STATES,
    legacy_after: D3D12_RESOURCE_STATES,
) {
    if list7.is_none() && legacy_before != legacy_after {
        // SAFETY: caller guarantees an open command list and live buffer.
        unsafe { list.ResourceBarrier(&[transition(resource, legacy_before, legacy_after)]) };
    }
}

/// Emits a scoped Enhanced `D3D12_GLOBAL_BARRIER` on `list7`, or falls back to
/// a legacy `uav_barrier(None)` when Enhanced Barriers / `ID3D12GraphicsCommandList7`
/// are unavailable or disabled via `VICTORY_DXR_ENHANCED_BARRIERS=0`.
unsafe fn emit_as_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
    is_compute_queue: bool,
) {
    if let Some(l7) = list7 {
        let barrier = [D3D12_GLOBAL_BARRIER {
            SyncBefore: sanitize_sync_for_queue(sync_before, is_compute_queue),
            SyncAfter: sanitize_sync_for_queue(sync_after, is_compute_queue),
            AccessBefore: access_before,
            AccessAfter: access_after,
        }];
        let group = [D3D12_BARRIER_GROUP {
            Type: D3D12_BARRIER_TYPE_GLOBAL,
            NumBarriers: 1,
            Anonymous: D3D12_BARRIER_GROUP_0 {
                pGlobalBarriers: barrier.as_ptr(),
            },
        }];
        // SAFETY: caller guarantees an open command list and spec-valid sync/access flags.
        unsafe { l7.Barrier(&group) };
    } else {
        // SAFETY: caller guarantees an open command list.
        unsafe { list.ResourceBarrier(&[uav_barrier(None)]) };
    }
}

/// Orders prior compute/copy work on `DirectQ` before DXR ops in `record_ops`
/// (skipped on `ComputeQ` where `skin_fence` already ordered Direct -> Compute).
unsafe fn emit_as_prelude_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    ops: &[DxrOp],
    is_compute_queue: bool,
) {
    let has_build_blas = ops.iter().any(|op| matches!(op, DxrOp::BuildBlas { .. }));
    let has_build_tlas = ops.iter().any(|op| matches!(op, DxrOp::BuildTlas { .. }));
    let has_extract = ops.iter().any(|op| matches!(op, DxrOp::ExtractStaticBlas { .. }));
    let has_copy = ops.iter().any(|op| matches!(op, DxrOp::CopyAs { .. }));
    let Some((sb, sa, ab, aa)) =
        prelude_barrier_for_ops(has_build_blas, has_build_tlas, has_extract, has_copy, is_compute_queue)
    else {
        return;
    };
    // SAFETY: valid D3D12 Enhanced Barrier sync/access pairing.
    unsafe { emit_as_barrier(list, list7, sb, sa, ab, aa, is_compute_queue) };
}

/// Orders mid-batch scratch reuse or overlapping dest/src AS VAs between AS builds.
unsafe fn emit_as_build_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    is_compute_queue: bool,
) {
    // SAFETY: valid D3D12 Enhanced Barrier sync/access pairing.
    unsafe {
        emit_as_barrier(
            list,
            list7,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
                | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            is_compute_queue,
        );
    }
}

/// Orders completed AS builds (and any postbuild UAV writes) before subsequent
/// AS builds, AS copies, and inline/DXR raytracing shaders.
unsafe fn emit_as_flush_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    is_compute_queue: bool,
) {
    // SAFETY: valid D3D12 Enhanced Barrier sync/access pairing.
    unsafe {
        emit_as_barrier(
            list,
            list7,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_ALL_SHADING
                | D3D12_BARRIER_SYNC_RAYTRACING,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
                | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
                | D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            is_compute_queue,
        );
    }
}

/// Orders `CopyRaytracingAccelerationStructure` writes before subsequent AS builds,
/// copies, and raytracing shaders.
unsafe fn emit_as_copy_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    is_compute_queue: bool,
) {
    // SAFETY: valid D3D12 Enhanced Barrier sync/access pairing.
    unsafe {
        emit_as_barrier(
            list,
            list7,
            D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_ALL_SHADING
                | D3D12_BARRIER_SYNC_RAYTRACING,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ
                | D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,
            is_compute_queue,
        );
    }
}

/// Orders contiguous dynamic BLAS builds/refits before the batched `BUILD_NORMALS_HLSL`
/// compute pass (`ARCH-14`).
unsafe fn emit_blas_to_compute_barrier(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    is_compute_queue: bool,
) {
    // SAFETY: valid D3D12 Enhanced Barrier sync/access pairing.
    unsafe {
        emit_as_barrier(
            list,
            list7,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE
                | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            is_compute_queue,
        );
    }
}

/// Splits a contiguous run of `BuildBlas` `(dest_key, src_key)` pairs into maximal
/// sub-batches with no destination/source VA overlap within a sub-batch so each sub-batch
/// takes `AS_MAP` once, records all AS builds contiguously, emits 1 barrier, and binds
/// `BUILD_NORMALS_HLSL` once (`ARCH-14`).
fn plan_blas_sub_batches(pairs: &[(u64, u64)]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut seen_dests: Vec<u64> = Vec::new();
    for (idx, &(dst, src)) in pairs.iter().enumerate() {
        let hazard = (dst != 0 && seen_dests.contains(&dst))
            || (src != 0 && src != dst && seen_dests.contains(&src));
        if hazard {
            ranges.push(start..idx);
            start = idx;
            seen_dests.clear();
        }
        if dst != 0 {
            seen_dests.push(dst);
        }
    }
    if start < pairs.len() {
        ranges.push(start..pairs.len());
    }
    ranges
}

/// Splits a contiguous run of `BuildTlas` destination keys into maximal contiguous
/// sub-batches with no duplicate non-zero destination key, so each sub-batch can
/// execute its patch dispatches and AS builds back-to-back with a single barrier.
fn plan_tlas_sub_batches(dest_keys: &[u64]) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut seen: Vec<u64> = Vec::new();
    for (idx, &key) in dest_keys.iter().enumerate() {
        if key != 0 && seen.contains(&key) {
            ranges.push(start..idx);
            start = idx;
            seen.clear();
        }
        if key != 0 {
            seen.push(key);
        }
    }
    if start < dest_keys.len() {
        ranges.push(start..dest_keys.len());
    }
    ranges
}

/// Bump-allocates 256B-aligned scratch slices across consecutive independent AS
/// builds within one command list so they execute without per-BLAS `uav_barrier`
/// stalls, flushing only on destination/source VA hazards, scratch exhaustion,
/// or before a dependent op (`CopyAs`, `BuildTlas`, or end of command list).
#[derive(Default)]
struct ScratchBatcher {
    buf_id: usize,
    cursor: u64,
    written_dests: Vec<u64>,
}

impl ScratchBatcher {
    fn has_pending(&self) -> bool {
        !self.written_dests.is_empty()
    }

    fn reset(&mut self) {
        self.cursor = 0;
        self.written_dests.clear();
    }

    /// Reserves a 256B-aligned scratch slice of `need` bytes within a scratch
    /// buffer identified by `buf_id` with total `capacity` bytes (`capacity >= aligned(need)`).
    /// Returns `(emit_uav_barrier_first, byte_offset)`.
    fn alloc(&mut self, buf_id: usize, capacity: u64, need: u64, dest_va: u64, src_va: u64) -> (bool, u64) {
        let aligned = need.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
        if buf_id != self.buf_id {
            self.buf_id = buf_id;
            self.cursor = 0;
        }
        let dest_conflict = self.written_dests.contains(&dest_va);
        let src_conflict = src_va != 0 && self.written_dests.contains(&src_va);
        let fits = self.cursor.checked_add(aligned).is_some_and(|end| end <= capacity);
        let need_barrier = self.has_pending() && (dest_conflict || src_conflict || !fits);
        if need_barrier {
            self.reset();
        }
        let offset = self.cursor;
        self.cursor = offset.saturating_add(aligned);
        self.written_dests.push(dest_va);
        (need_barrier, offset)
    }

    /// Emits a single scoped AS barrier (or legacy `uav_barrier(None)`) and clears
    /// pending state if any builds are currently unbarriered.
    unsafe fn flush(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        list7: Option<&ID3D12GraphicsCommandList7>,
        is_compute_queue: bool,
    ) {
        if self.has_pending() {
            // SAFETY: open command list.
            unsafe { emit_as_flush_barrier(list, list7, is_compute_queue) };
            self.reset();
        }
    }
}

fn record_empty_tlas_init(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    tlas_vas: &[u64],
    use_compute: bool,
) -> Option<ID3D12CommandList> {
    const EMPTY_TLAS_FLAGS: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
        NumDescs: 0,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            InstanceDescs: 0,
        },
    };
    let (result_bytes, scratch_bytes) = state.tlas_prebuild_info(device5, EMPTY_TLAS_FLAGS, 0);

    let (allocator, list) = state.take_list(host, use_compute)?;
    name(&list, || "victory dxr ops".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;
    let list7 = dxr_list7(host, &list);
    let one_scratch = scratch_bytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
    let want_scratch = one_scratch
        .saturating_mul(tlas_vas.len().max(1) as u64)
        .min(MAX_POOLED_SCRATCH_BYTES)
        .max(one_scratch);
    let (scratch_buf, scratch_gpu, scratch_cap) = sub.ensure_scratch(host, want_scratch)?;
    let scratch_id = scratch_buf.as_raw() as usize;
    let mut scratch_batch = ScratchBatcher::default();
    #[cfg(feature = "diagnostics")]
    let (timing, timing_base, spans_base) = (sub.ensure_timing(host), sub.timing_next, sub.timing_spans.len());

    // SAFETY: allocator/list are idle (returned after GPU fence); live resources.
    let recorded: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            #[cfg(feature = "diagnostics")]
            let t_start = timing.as_ref().and_then(|(h, _)| sub.stamp(&list, h));
            let map = as_map().read().unwrap_or_else(|e| e.into_inner());
            for &guest_va in tlas_vas {
                let Some(entry) = map.get(&guest_va) else {
                    continue;
                };
                if entry.built || entry.slice.size < result_bytes {
                    continue;
                }
                let (need_barrier, scratch_off) =
                    scratch_batch.alloc(scratch_id, scratch_cap, one_scratch, entry.gpu_va, 0);
                if need_barrier {
                    emit_as_build_barrier(&list, list7.as_ref(), use_compute);
                }
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: entry.gpu_va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gpu + scratch_off,
                };
                list4.BuildRaytracingAccelerationStructure(&desc, None);
                if let Some(mirror) = &entry.mirror_tlas
                    && mirror.slice.size >= result_bytes
                {
                    let (m_barrier, m_scratch_off) =
                        scratch_batch.alloc(scratch_id, scratch_cap, one_scratch, mirror.gpu_va, 0);
                    if m_barrier {
                        emit_as_build_barrier(&list, list7.as_ref(), use_compute);
                    }
                    let m_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: mirror.gpu_va,
                        Inputs: inputs,
                        SourceAccelerationStructureData: 0,
                        ScratchAccelerationStructureData: scratch_gpu + m_scratch_off,
                    };
                    list4.BuildRaytracingAccelerationStructure(&m_desc, None);
                }
            }
            scratch_batch.flush(&list, list7.as_ref(), use_compute);
            #[cfg(feature = "diagnostics")]
            if let Some((h, rb)) = timing.as_ref() {
                let t_end = sub.stamp(&list, h);
                if let (Some(s), Some(e)) = (t_start, t_end) {
                    sub.timing_spans.push(("dxr [empty_tlas_init]", s, e));
                }
                sub.resolve_timing(&list, h, rb, timing_base);
            }
            list.Close()
        })()
    };
    if let Err(e) = recorded {
        tracing::warn!(error = %e, "dxr: empty TLAS init list failed");
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_next = timing_base;
            sub.timing_spans.truncate(spans_base);
        }
        state.return_list((allocator, list), use_compute);
        return None;
    }
    {
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        for &guest_va in tlas_vas {
            if let Some(entry) = map.get_mut(&guest_va)
                && entry.slice.size >= result_bytes
            {
                entry.built = true;
            }
        }
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn record_ops(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    owner: &HostList,
    ops: &[DxrOp],
    built_tlas: &mut bool,
    use_compute: bool,
) -> Option<ID3D12CommandList> {
    let (allocator, list) = state.take_list(host, use_compute)?;
    name(&list, || "victory dxr ops".into());
    let list4: ID3D12GraphicsCommandList4 = list.cast().ok()?;
    let list7 = dxr_list7(host, &list);

    // Track per-command-list resource state of reusable buffers that start in COMMON.
    let mut hash_in_srv = false;
    let mut host_inst_in_srv = false;
    let mut extract_in_srv = false;
    let mut staging_in_copy_src: Option<ID3D12Resource> = None;
    let mut normals_state = D3D12_RESOURCE_STATE_COMMON;
    let normals_gpu_entry = state.ensure_normals_gpu(host);
    let normals_init_src = if normals_gpu_entry.is_some() && state.normals_init_needed {
        if let Some((nbuf, nptr)) = sub.ensure_normals_upload(host, 256) {
            // SAFETY: `nptr` is mapped for at least 256 bytes.
            unsafe {
                let w = victory_shader::emit::DXR_NORMALS_FALLBACK.to_le_bytes();
                std::ptr::copy_nonoverlapping(w.as_ptr(), nptr, 4);
            }
            Some(nbuf)
        } else {
            None
        }
    } else {
        None
    };
    let compact_start_slot = sub.compact_queries.len();
    let mut compact_uav_active: Option<(ID3D12Resource, ID3D12Resource)> = None;
    let mut scratch_batch = ScratchBatcher::default();
    let mut recorded_any = false;
    #[cfg(feature = "diagnostics")]
    let (timing, timing_base, spans_base) = (sub.ensure_timing(host), sub.timing_next, sub.timing_spans.len());

    // SAFETY: allocator/list are idle; all buffers referenced are kept in `sub` or slabs.
    let res: windows::core::Result<()> = unsafe {
        (|| {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            #[cfg(feature = "diagnostics")]
            let mut prev_stamp: Option<(&'static str, u32)> = None;
            // Ensure prior GPU compute/copy work on DirectQ is visible (skipped on ComputeQ
            // where `skin_fence` already ordered Direct -> Compute).
            emit_as_prelude_barrier(&list, list7.as_ref(), ops, use_compute);
            if let (Some((n_dst, _)), Some(n_src)) = (&normals_gpu_entry, &normals_init_src) {
                list.CopyBufferRegion(n_dst, 0, n_src, 0, 4);
                normals_state = D3D12_RESOURCE_STATE_COPY_DEST;
                recorded_any = true;
            }

            let mut op_idx = 0usize;
            while op_idx < ops.len() {
                let op = &ops[op_idx];
                #[cfg(feature = "diagnostics")]
                let op_label: &'static str = match op {
                    DxrOp::BuildBlas { .. } => "dxr [build_blas]",
                    DxrOp::CopyAs { .. } => "dxr [copy_as]",
                    DxrOp::ExtractStaticBlas { .. } => "dxr [extract_static_blas]",
                    DxrOp::BuildTlas { .. } => "dxr [build_tlas]",
                };
                #[cfg(feature = "diagnostics")]
                let op_start = timing.as_ref().and_then(|(h, _)| sub.stamp(&list, h));
                #[cfg(feature = "diagnostics")]
                if let (Some((prev_label, s)), Some(e)) = (prev_stamp.take(), op_start) {
                    sub.timing_spans.push((prev_label, s, e));
                }
                let mut did_op = false;
                if matches!(op, DxrOp::BuildBlas { .. }) {
                    let start_idx = op_idx;
                    while op_idx < ops.len() && matches!(ops[op_idx], DxrOp::BuildBlas { .. }) {
                        op_idx += 1;
                    }
                    if record_build_blas_batch(
                        host,
                        device5,
                        state,
                        sub,
                        &list,
                        &list4,
                        list7.as_ref(),
                        &mut scratch_batch,
                        &ops[start_idx..op_idx],
                        normals_gpu_entry.as_ref(),
                        &mut normals_state,
                        &mut compact_uav_active,
                        use_compute,
                    ) {
                        recorded_any = true;
                        did_op = true;
                    }
                } else if matches!(op, DxrOp::BuildTlas { .. }) {
                    let start_idx = op_idx;
                    while op_idx < ops.len() && matches!(ops[op_idx], DxrOp::BuildTlas { .. }) {
                        op_idx += 1;
                    }
                    scratch_batch.flush(&list, list7.as_ref(), use_compute);
                    if record_build_tlas_batch(
                        host,
                        device5,
                        state,
                        sub,
                        &list,
                        &list4,
                        list7.as_ref(),
                        &mut scratch_batch,
                        &ops[start_idx..op_idx],
                        &mut hash_in_srv,
                        &mut host_inst_in_srv,
                        normals_gpu_entry.as_ref(),
                        &mut normals_state,
                        built_tlas,
                        use_compute,
                    ) {
                        recorded_any = true;
                        did_op = true;
                    }
                } else {
                    op_idx += 1;
                    match op {
                        DxrOp::CopyAs {
                            dest_va,
                            source_va,
                            mode,
                        } => {
                            scratch_batch.flush(&list, list7.as_ref(), use_compute);
                            if record_copy_as(
                                host,
                                state,
                                &list,
                                &list4,
                                list7.as_ref(),
                                *dest_va,
                                *source_va,
                                *mode,
                                use_compute,
                            ) {
                                recorded_any = true;
                                did_op = true;
                            }
                        }
                        DxrOp::ExtractStaticBlas {
                            dest_va,
                            size_bytes,
                        } => {
                            if record_extract_static_blas(
                                host,
                                device5,
                                state,
                                sub,
                                owner,
                                &list,
                                &list4,
                                list7.as_ref(),
                                &mut scratch_batch,
                                *dest_va,
                                *size_bytes,
                                &mut extract_in_srv,
                                &mut staging_in_copy_src,
                                &mut compact_uav_active,
                                use_compute,
                            ) {
                                recorded_any = true;
                                did_op = true;
                            }
                        }
                        DxrOp::BuildBlas { .. } | DxrOp::BuildTlas { .. } => unreachable!(),
                    }
                }
                #[cfg(feature = "diagnostics")]
                if did_op && let Some(s) = op_start {
                    prev_stamp = Some((op_label, s));
                }
                #[cfg(not(feature = "diagnostics"))]
                let _ = did_op;
            }
            scratch_batch.flush(&list, list7.as_ref(), use_compute);
            if let Some((n_buf, _)) = &normals_gpu_entry
                && normals_state != D3D12_RESOURCE_STATE_COMMON
            {
                let (sync_b, acc_b) = if normals_state == D3D12_RESOURCE_STATE_COPY_DEST {
                    (D3D12_BARRIER_SYNC_COPY, D3D12_BARRIER_ACCESS_COPY_DEST)
                } else {
                    (
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    )
                };
                emit_dxr_buffer_barrier(
                    &list,
                    list7.as_ref(),
                    n_buf,
                    sync_b,
                    D3D12_BARRIER_SYNC_ALL_SHADING,
                    acc_b,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE | D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    normals_state,
                    D3D12_RESOURCE_STATE_COMMON,
                    use_compute,
                );
                normals_state = D3D12_RESOURCE_STATE_COMMON;
            }
            if hash_in_srv
                && let Some(hash_aux) = state.hash_gpu.as_ref()
            {
                emit_dxr_buffer_legacy_only(
                    &list,
                    list7.as_ref(),
                    &hash_aux.buf,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }
            if let Some(staging_buf) = staging_in_copy_src.take() {
                emit_dxr_buffer_legacy_only(
                    &list,
                    list7.as_ref(),
                    &staging_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
            }
            if let Some((uav_buf, rb_buf)) = compact_uav_active.take() {
                let count = sub.compact_queries.len().saturating_sub(compact_start_slot);
                if count > 0 {
                    emit_dxr_buffer_barrier(
                        &list,
                        list7.as_ref(),
                        &uav_buf,
                        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        use_compute,
                    );
                    let off = (compact_start_slot as u64) * 8;
                    let len = (count as u64) * 8;
                    list.CopyBufferRegion(&rb_buf, off, &uav_buf, off, len);
                    emit_dxr_buffer_legacy_only(
                        &list,
                        list7.as_ref(),
                        &uav_buf,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                } else {
                    emit_dxr_buffer_legacy_only(
                        &list,
                        list7.as_ref(),
                        &uav_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                }
            }
            #[cfg(feature = "diagnostics")]
            if let Some((h, rb)) = timing.as_ref() {
                if let Some((prev_label, s)) = prev_stamp.take()
                    && let Some(e) = sub.stamp(&list, h)
                {
                    sub.timing_spans.push((prev_label, s, e));
                }
                sub.resolve_timing(&list, h, rb, timing_base);
            }
            list.Close()
        })()
    };

    if let Err(e) = res {
        tracing::warn!(error = %e, "dxr: command list recording failed");
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_next = timing_base;
            sub.timing_spans.truncate(spans_base);
        }
        sub.compact_queries.truncate(compact_start_slot);
        state.return_list((allocator, list), use_compute);
        return None;
    }
    if !recorded_any {
        #[cfg(feature = "diagnostics")]
        {
            sub.timing_next = timing_base;
            sub.timing_spans.truncate(spans_base);
        }
        sub.compact_queries.truncate(compact_start_slot);
        state.return_list((allocator, list), use_compute);
        return None;
    }
    if normals_init_src.is_some() {
        state.normals_init_needed = false;
    }
    let out: ID3D12CommandList = list.cast().ok()?;
    sub.lists.push((allocator, list));
    Some(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExistingBlasRefitInfo {
    gpu_va: u64,
    flags: u32,
    built: bool,
    topo_hash: u64,
    update_scratch_bytes: u64,
    consecutive_refits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlasRefitDecision {
    flags: u32,
    host_src_va: u64,
    /// When `Some((gpu_va, update_scratch_bytes))`, refit in-place without a
    /// prebuild query or slab reallocation.
    in_place_update: Option<(u64, u64)>,
    next_consecutive_refits: u32,
    is_refit: bool,
}

/// Pure dynamic BLAS refit policy shared by explicit title `PERFORM_UPDATE`
/// calls and opt-in `VICTORY_NATIVE_RT_REFIT=1` automatic promotion.
#[allow(clippy::too_many_arguments)]
fn decide_blas_refit(
    raw_flags: u32,
    dest_key: u64,
    source_va: u64,
    topo_hash: u64,
    existing_dst: Option<ExistingBlasRefitInfo>,
    existing_src: Option<ExistingBlasRefitInfo>,
    refit_enabled: bool,
    refit_period: u32,
) -> BlasRefitDecision {
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
    const PREFER_FAST_BUILD: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;
    const PREFER_FAST_TRACE: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0 as u32;
    const PREF_MASK: u32 = PREFER_FAST_BUILD | PREFER_FAST_TRACE;
    const OTHER_MASK: u32 = !(ALLOW_UPDATE | PERFORM_UPDATE | PREF_MASK);

    // Never OR `ALLOW_UPDATE` onto `ALLOW_COMPACTION`-only BLASes: doing so bloats
    // compacted builds by 30-50% and defeats compaction (`ARCH-10`).
    let compaction_only = (raw_flags & ALLOW_COMPACTION) != 0 && (raw_flags & ALLOW_UPDATE) == 0;
    let effective_refit = refit_enabled && !compaction_only;

    let is_compatible = |entry: &ExistingBlasRefitInfo| -> bool {
        let pref_ok = (entry.flags & PREF_MASK) == (raw_flags & PREF_MASK)
            || (effective_refit
                && (entry.flags & PREF_MASK) == PREFER_FAST_TRACE
                && (raw_flags & PREFER_FAST_TRACE) == 0);
        entry.built
            && (entry.flags & ALLOW_UPDATE) != 0
            && (entry.flags & OTHER_MASK) == (raw_flags & OTHER_MASK)
            && pref_ok
            && entry.topo_hash == topo_hash
            && topo_hash != 0
    };
    let full_flags = (raw_flags & !PERFORM_UPDATE) | if effective_refit { ALLOW_UPDATE } else { 0 };
    // Periodic full rebuilds upgrade to PREFER_FAST_TRACE | ALLOW_UPDATE so the
    // subsequent 63 refits trace a higher-quality BVH topology.
    let periodic_full_flags = (raw_flags & !(PERFORM_UPDATE | PREFER_FAST_BUILD)) | PREFER_FAST_TRACE | ALLOW_UPDATE;

    if raw_flags & PERFORM_UPDATE != 0 && source_va != 0 {
        let src_key = source_va & !0xFF;
        let candidate = if src_key == dest_key { existing_dst } else { existing_src };
        if let Some(src) = candidate
            && is_compatible(&src)
        {
            if effective_refit && refit_period > 0 && src.consecutive_refits >= refit_period {
                return BlasRefitDecision {
                    flags: periodic_full_flags,
                    host_src_va: 0,
                    in_place_update: None,
                    next_consecutive_refits: 0,
                    is_refit: false,
                };
            }
            let update_flags = if effective_refit {
                (src.flags & !PERFORM_UPDATE) | ALLOW_UPDATE | PERFORM_UPDATE
            } else {
                raw_flags
            };
            let in_place = if src_key == dest_key && src.update_scratch_bytes > 0 {
                Some((src.gpu_va, src.update_scratch_bytes))
            } else {
                None
            };
            return BlasRefitDecision {
                flags: update_flags,
                host_src_va: src.gpu_va,
                in_place_update: in_place,
                next_consecutive_refits: src.consecutive_refits.saturating_add(1),
                is_refit: true,
            };
        }
        return BlasRefitDecision {
            flags: full_flags,
            host_src_va: 0,
            in_place_update: None,
            next_consecutive_refits: 0,
            is_refit: false,
        };
    }

    if effective_refit
        && let Some(dst) = existing_dst
        && is_compatible(&dst)
        && dst.update_scratch_bytes > 0
    {
        if refit_period > 0 && dst.consecutive_refits >= refit_period {
            return BlasRefitDecision {
                flags: periodic_full_flags,
                host_src_va: 0,
                in_place_update: None,
                next_consecutive_refits: 0,
                is_refit: false,
            };
        }
        return BlasRefitDecision {
            flags: (dst.flags & !PERFORM_UPDATE) | ALLOW_UPDATE | PERFORM_UPDATE,
            host_src_va: dst.gpu_va,
            in_place_update: Some((dst.gpu_va, dst.update_scratch_bytes)),
            next_consecutive_refits: dst.consecutive_refits.saturating_add(1),
            is_refit: true,
        };
    }

    BlasRefitDecision {
        flags: full_flags,
        host_src_va: 0,
        in_place_update: None,
        next_consecutive_refits: 0,
        is_refit: false,
    }
}

/// Records a contiguous batch of dynamic `DxrOp::BuildBlas` operations (`ARCH-14`):
/// acquires `AS_MAP` once per sub-batch, records all `BuildRaytracingAccelerationStructure`
/// calls contiguously with zero interleaved compute PSO switches, emits one
/// `BUILD_AS -> COMPUTE` barrier, and binds `BUILD_NORMALS_HLSL` once for all dirty dynamic
/// BLASes (skipping unskinned/unchanged refits where safe).
#[allow(clippy::too_many_arguments)]
unsafe fn record_build_blas_batch(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    list7: Option<&ID3D12GraphicsCommandList7>,
    scratch_batch: &mut ScratchBatcher,
    blas_ops: &[DxrOp],
    normals_gpu: Option<&(ID3D12Resource, u64)>,
    normals_state: &mut D3D12_RESOURCE_STATES,
    compact_uav_active: &mut Option<(ID3D12Resource, ID3D12Resource)>,
    use_compute: bool,
) -> bool {
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;

    let pairs: Vec<(u64, u64)> = blas_ops
        .iter()
        .map(|op| match *op {
            DxrOp::BuildBlas {
                dest_va, source_va, ..
            } => (dest_va & !0xFF, source_va & !0xFF),
            _ => (0, 0),
        })
        .collect();

    let refit_on = refit_dynamic_blas();
    let period = refit_period();
    let n_pass = normals_gpu.and_then(|_| build_normals_pass(host));
    let mut any_recorded = false;

    struct ResolvedBlas<'a> {
        key: u64,
        source_va: u64,
        raw_flags: u32,
        postbuild: &'a [PostbuildSpec],
        host_geoms: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
        geom_tris: Vec<u32>,
        total_tris: u32,
        topo_hash: u64,
        geom_va_hash: u64,
        any_gpu_written: bool,
        any_transform: bool,
    }

    struct PlannedBlas {
        key: u64,
        host_geoms: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
        geom_tris: Vec<u32>,
        flags: u32,
        host_dst_va: u64,
        host_src_va: u64,
        scratch_need: u64,
        dst_gen: u64,
        dst_slice_size: u64,
        is_refit: bool,
        host_postbuild: Vec<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>,
        ng_off: u32,
        nd_off: u32,
        normals_newly_allocated: bool,
        need_normals_dispatch: bool,
    }

    let to_refit_info = |e: &HostAs| -> Option<ExistingBlasRefitInfo> {
        (e.kind == AsKind::Blas).then_some(ExistingBlasRefitInfo {
            gpu_va: e.gpu_va,
            flags: e.flags,
            built: e.built,
            topo_hash: e.topo_hash,
            update_scratch_bytes: e.update_scratch_bytes,
            consecutive_refits: e.consecutive_refits,
        })
    };

    for range in plan_blas_sub_batches(&pairs) {
        let slice = &blas_ops[range];
        // Phase 1: Resolve shadow geometry buffers & fingerprints outside `AS_MAP` lock.
        let mut resolved: Vec<ResolvedBlas<'_>> = Vec::with_capacity(slice.len());
        for op in slice {
            let DxrOp::BuildBlas {
                dest_va,
                source_va,
                flags: raw_flags,
                geometries,
                postbuild,
            } = op
            else {
                continue;
            };
            let key = *dest_va & !0xFF;
            if key == 0 || geometries.is_empty() {
                STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }
            let mut host_geoms = Vec::with_capacity(geometries.len());
            let mut geom_tris = Vec::with_capacity(geometries.len());
            let mut total_tris = 0u32;
            let mut any_gpu_written = false;
            let mut any_transform = false;
            for g in geometries {
                match g.kind {
                    GeometryKind::Triangles {
                        transform3x4,
                        index_format,
                        vertex_format,
                        index_count,
                        vertex_count,
                        index_buffer,
                        vertex_buffer_va,
                        vertex_buffer_stride,
                    } => {
                        let pc_vtx_fmt = crate::resource::pc_format(vertex_format);
                        let (vtx_align, vtx_min_bytes, vtx_fmt_ok) = match DXGI_FORMAT(pc_vtx_fmt as i32) {
                            DXGI_FORMAT_R32G32B32_FLOAT => (4u64, 12u64, true),
                            DXGI_FORMAT_R32G32_FLOAT => (4, 8, true),
                            DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_SNORM => (2, 8, true),
                            DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM => (2, 4, true),
                            _ => (4, 12, false),
                        };
                        let (vb_gpu, vc, vb_stride) = if vtx_fmt_ok
                            && vertex_buffer_va != 0
                            && vertex_count > 0
                            && vertex_buffer_stride % vtx_align == 0
                        {
                            let stride = vertex_buffer_stride.max(vtx_min_bytes);
                            let len = u64::from(vertex_count)
                                .saturating_sub(1)
                                .saturating_mul(stride)
                                .saturating_add(vtx_min_bytes);
                            match crate::shadow::find(vertex_buffer_va, len) {
                                Some(s) if s.left(vertex_buffer_va) >= len => {
                                    let gpu = s.gpu(vertex_buffer_va);
                                    if gpu % vtx_align == 0 {
                                        if s.written() || s.gpu_only_skip() {
                                            any_gpu_written = true;
                                        }
                                        sub.kept_shadows.push(s);
                                        (gpu, vertex_count, vertex_buffer_stride)
                                    } else {
                                        (0, 0, 0)
                                    }
                                }
                                _ => (0, 0, 0),
                            }
                        } else {
                            (0, 0, 0)
                        };
                        let pc_idx_fmt = crate::resource::pc_format(index_format);
                        let (ib_gpu, ic, idx_fmt) = if index_buffer != 0 && index_count >= 3 && vc > 0 {
                            let elem = match DXGI_FORMAT(pc_idx_fmt as i32) {
                                DXGI_FORMAT_R16_UINT => Some(2u64),
                                DXGI_FORMAT_R32_UINT => Some(4u64),
                                _ => None,
                            };
                            let aligned_ic = index_count - (index_count % 3);
                            if let Some(elem) = elem {
                                let len = u64::from(aligned_ic).saturating_mul(elem);
                                match crate::shadow::find(index_buffer, len) {
                                    Some(s) if s.left(index_buffer) >= len => {
                                        let gpu = s.gpu(index_buffer);
                                        if gpu % elem == 0 {
                                            if s.written() || s.gpu_only_skip() {
                                                any_gpu_written = true;
                                            }
                                            sub.kept_shadows.push(s);
                                            (gpu, aligned_ic, DXGI_FORMAT(pc_idx_fmt as i32))
                                        } else {
                                            (0, 0, DXGI_FORMAT_UNKNOWN)
                                        }
                                    }
                                    _ => (0, 0, DXGI_FORMAT_UNKNOWN),
                                }
                            } else {
                                (0, 0, DXGI_FORMAT_UNKNOWN)
                            }
                        } else {
                            (0, 0, DXGI_FORMAT_UNKNOWN)
                        };
                        let (vb_gpu, vc, vb_stride) = if index_buffer != 0 && index_count > 0 && ib_gpu == 0 {
                            (0, 0, 0)
                        } else if ib_gpu == 0 {
                            let aligned_vc = vc - (vc % 3);
                            if aligned_vc == 0 {
                                (0, 0, 0)
                            } else {
                                (vb_gpu, aligned_vc, vb_stride)
                            }
                        } else {
                            (vb_gpu, vc, vb_stride)
                        };
                        let xf_gpu = if transform3x4 != 0 && vc > 0 {
                            match crate::shadow::find(transform3x4, 48) {
                                Some(s) if s.left(transform3x4) >= 48 => {
                                    let gpu = s.gpu(transform3x4);
                                    if gpu % 16 == 0 {
                                        any_transform = true;
                                        sub.kept_shadows.push(s);
                                        gpu
                                    } else {
                                        0
                                    }
                                }
                                _ => 0,
                            }
                        } else {
                            0
                        };
                        let host_vtx_fmt = if vtx_fmt_ok {
                            DXGI_FORMAT(pc_vtx_fmt as i32)
                        } else {
                            DXGI_FORMAT_R32G32B32_FLOAT
                        };
                        let tc = if vb_gpu == 0 || vc == 0 {
                            0
                        } else if ib_gpu != 0 && ic >= 3 {
                            ic / 3
                        } else {
                            vc / 3
                        };
                        geom_tris.push(tc);
                        total_tris = total_tris.saturating_add(tc);
                        host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                            Flags: dynamic_geom_flags(g.flags),
                            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                                    Transform3x4: xf_gpu,
                                    IndexFormat: idx_fmt,
                                    VertexFormat: host_vtx_fmt,
                                    IndexCount: ic,
                                    VertexCount: vc,
                                    IndexBuffer: ib_gpu,
                                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                        StartAddress: vb_gpu,
                                        StrideInBytes: vb_stride,
                                    },
                                },
                            },
                        });
                    }
                    GeometryKind::Aabbs {
                        aabb_count,
                        aabbs_va,
                        aabbs_stride,
                    } => {
                        let (ab_gpu, cnt) = if aabbs_va != 0 && aabb_count > 0 && aabbs_stride % 8 == 0 {
                            let stride = aabbs_stride.max(24);
                            let len = aabb_count
                                .saturating_sub(1)
                                .saturating_mul(stride)
                                .saturating_add(24);
                            match crate::shadow::find(aabbs_va, len) {
                                Some(s) if s.left(aabbs_va) >= len => {
                                    let gpu = s.gpu(aabbs_va);
                                    if gpu % 8 == 0 {
                                        sub.kept_shadows.push(s);
                                        (gpu, aabb_count)
                                    } else {
                                        (0, 0)
                                    }
                                }
                                _ => (0, 0),
                            }
                        } else {
                            (0, 0)
                        };
                        geom_tris.push(0);
                        host_geoms.push(D3D12_RAYTRACING_GEOMETRY_DESC {
                            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
                            Flags: dynamic_geom_flags(g.flags),
                            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                                AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                                    AABBCount: cnt,
                                    AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                        StartAddress: ab_gpu,
                                        StrideInBytes: if cnt > 0 { aabbs_stride } else { 0 },
                                    },
                                },
                            },
                        });
                    }
                }
            }
            let topo_hash = blas_topology_hash(&host_geoms);
            let geom_va_hash = blas_geom_va_hash(&host_geoms);
            resolved.push(ResolvedBlas {
                key,
                source_va: *source_va,
                raw_flags: *raw_flags,
                postbuild,
                host_geoms,
                geom_tris,
                total_tris,
                topo_hash,
                geom_va_hash,
                any_gpu_written,
                any_transform,
            });
        }

        if resolved.is_empty() {
            continue;
        }

        // Phase 2: Acquire `AS_MAP` write lock ONCE for the entire sub-batch (instead of 3×/BLAS!).
        let mut planned: Vec<PlannedBlas> = Vec::with_capacity(resolved.len());
        let mut max_scratch = AS_ALIGN;
        let mut sum_scratch = 0u64;
        let mut any_normals_work = false;
        let cur_epoch = state.tlas_build_epoch;
        {
            let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
            for item in resolved {
                let src_key = item.source_va & !0xFF;
                let existing_dst = map.get(&item.key).and_then(to_refit_info);
                let existing_src = if src_key != 0 && src_key != item.key {
                    map.get(&src_key).and_then(to_refit_info)
                } else {
                    None
                };
                let decision = decide_blas_refit(
                    item.raw_flags,
                    item.key,
                    item.source_va,
                    item.topo_hash,
                    existing_dst,
                    existing_src,
                    refit_on,
                    period,
                );
                let flags = decision.flags;
                let host_src_va = decision.host_src_va;

                let (host_dst_va, scratch_need, dst_gen, dst_slice_size) =
                    if let Some((gpu_va, update_scratch)) = decision.in_place_update {
                        (gpu_va, update_scratch.max(AS_ALIGN), 0u64, 0u64)
                    } else {
                        let (result_bytes, scratch_bytes, update_scratch_bytes) =
                            if let Some(&cached) = state.prebuild_cache.get(&(flags, item.topo_hash)) {
                                cached
                            } else {
                                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(flags as i32),
                                    NumDescs: item.host_geoms.len() as u32,
                                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                                        pGeometryDescs: item.host_geoms.as_ptr(),
                                    },
                                };
                                let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
                                // SAFETY: valid device5 and geometry array alive through call.
                                unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
                                let entry = (
                                    pre.ResultDataMaxSizeInBytes,
                                    pre.ScratchDataSizeInBytes,
                                    pre.UpdateScratchDataSizeInBytes,
                                );
                                if entry.0 > 0 {
                                    if state.prebuild_cache.len() >= MAX_PREBUILD_CACHE {
                                        state.prebuild_cache.clear();
                                    }
                                    state.prebuild_cache.insert((flags, item.topo_hash), entry);
                                }
                                entry
                            };
                        if result_bytes == 0 {
                            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                            continue;
                        }
                        let generation = state.bump_gen();
                        let allocated = if let Some(existing) = map.get_mut(&item.key)
                            && existing.kind == AsKind::Blas
                            && existing.slice.tier == SlabTier::TransientDynamicTlas
                            && existing.slice.size >= result_bytes
                        {
                            existing.flags = flags;
                            existing.num_instances = 0;
                            existing.built = true;
                            existing.generation = generation;
                            existing.topo_hash = item.topo_hash;
                            existing.update_scratch_bytes = update_scratch_bytes;
                            existing.consecutive_refits = decision.next_consecutive_refits;
                            existing.compacted_size = 0;
                            existing.last_seen_tlas_epoch.store(cur_epoch, Ordering::Relaxed);
                            Some((existing.gpu_va, generation, existing.slice.size))
                        } else if let Some((buffer, slice, gpu_va)) =
                            state.slabs.alloc(host, SlabTier::TransientDynamicTlas, result_bytes)
                        {
                            let slice_size = slice.size;
                            let entry = HostAs {
                                buffer,
                                slice,
                                gpu_va,
                                mirror_tlas: None,
                                kind: AsKind::Blas,
                                flags,
                                num_instances: 0,
                                built: true,
                                generation,
                                topo_hash: item.topo_hash,
                                update_scratch_bytes,
                                consecutive_refits: decision.next_consecutive_refits,
                                compacted_size: 0,
                                last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
                                geom_va_hash: 0,
                                normals_geom_off: 0,
                                normals_geom_len: 0,
                                normals_data_off: 0,
                                normals_data_len: 0,
                            };
                            if let Some(old) = map.insert(item.key, entry) {
                                state.retire_as(&old);
                            }
                            Some((gpu_va, generation, slice_size))
                        } else {
                            None
                        };
                        let Some((gpu_va, alloc_gen, slice_size)) = allocated else {
                            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        let scratch_need = if flags & PERFORM_UPDATE != 0 {
                            update_scratch_bytes.max(AS_ALIGN)
                        } else {
                            scratch_bytes.max(AS_ALIGN)
                        };
                        (gpu_va, scratch_need, alloc_gen, slice_size)
                    };

                // Update in-place refit metadata and allocate/reuse `dxr_normals` under this same lock.
                let need_geoms = item.host_geoms.len() as u32;
                let (ng_off, nd_off, normals_newly_allocated, same_geom_va_hash) =
                    if let Some(entry) = map.get_mut(&item.key) {
                        if decision.in_place_update.is_some() {
                            entry.flags = flags;
                            entry.consecutive_refits = decision.next_consecutive_refits;
                            entry.last_seen_tlas_epoch.store(cur_epoch, Ordering::Relaxed);
                        }
                        let prev_va_hash = entry.geom_va_hash;
                        entry.geom_va_hash = item.geom_va_hash;
                        let same_va = prev_va_hash != 0 && prev_va_hash == item.geom_va_hash;
                        if n_pass.is_some() {
                            if entry.normals_geom_off != 0
                                && entry.normals_geom_len == need_geoms
                                && entry.normals_data_len == item.total_tris
                            {
                                (entry.normals_geom_off, entry.normals_data_off, false, same_va)
                            } else {
                                let rel_sub = state.sub_seq.saturating_add(2);
                                let rel_ep = state.tlas_build_epoch.saturating_add(2);
                                let rel_dxr = retire_dxr_fence_target();
                                if entry.normals_geom_len > 0 {
                                    state.geom_pool.retire(
                                        entry.normals_geom_off,
                                        entry.normals_geom_len,
                                        rel_ep,
                                        rel_sub,
                                        rel_dxr,
                                    );
                                    entry.normals_geom_off = 0;
                                    entry.normals_geom_len = 0;
                                }
                                if entry.normals_data_len > 0 {
                                    state.data_pool.retire(
                                        entry.normals_data_off,
                                        entry.normals_data_len,
                                        rel_ep,
                                        rel_sub,
                                        rel_dxr,
                                    );
                                    entry.normals_data_off = 0;
                                    entry.normals_data_len = 0;
                                }
                                if need_geoms > 0
                                    && item.total_tris > 0
                                    && let (Some(go), Some(doff)) = (
                                        state.geom_pool.alloc(need_geoms),
                                        state.data_pool.alloc(item.total_tris),
                                    )
                                {
                                    entry.normals_geom_off = go;
                                    entry.normals_geom_len = need_geoms;
                                    entry.normals_data_off = doff;
                                    entry.normals_data_len = item.total_tris;
                                    (go, doff, true, false)
                                } else {
                                    (0, 0, false, same_va)
                                }
                            }
                        } else {
                            (0, 0, false, same_va)
                        }
                    } else {
                        (0, 0, false, false)
                    };

                // Skip redundant `BUILD_NORMALS` only when refitting an unskinned, transform-free
                // dynamic BLAS whose normals table and VB/IB VAs are completely unchanged.
                let can_skip_normals = decision.is_refit
                    && !normals_newly_allocated
                    && !item.any_gpu_written
                    && !item.any_transform
                    && same_geom_va_hash;
                let need_normals_dispatch = ng_off != 0 && nd_off != 0 && !can_skip_normals;
                if need_normals_dispatch {
                    any_normals_work = true;
                }
                state.insert_blas_hash(item.key, host_dst_va, ng_off);

                let host_postbuild = translate_postbuild(sub, item.postbuild);
                let aligned_scratch = scratch_need.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
                max_scratch = max_scratch.max(aligned_scratch);
                sum_scratch = sum_scratch.saturating_add(aligned_scratch);
                planned.push(PlannedBlas {
                    key: item.key,
                    host_geoms: item.host_geoms,
                    geom_tris: item.geom_tris,
                    flags,
                    host_dst_va,
                    host_src_va,
                    scratch_need: aligned_scratch,
                    dst_gen,
                    dst_slice_size,
                    is_refit: decision.is_refit,
                    host_postbuild,
                    ng_off,
                    nd_off,
                    normals_newly_allocated,
                    need_normals_dispatch,
                });
            }
        }

        if planned.is_empty() {
            continue;
        }

        let want_scratch = sum_scratch.min(MAX_POOLED_SCRATCH_BYTES).max(max_scratch);
        let Some((scratch_buf, scratch_gpu, scratch_cap)) = sub.ensure_scratch(host, want_scratch) else {
            STATS[STAT_SKIPPED].fetch_add(planned.len() as u64, Ordering::Relaxed);
            continue;
        };
        let scratch_id = scratch_buf.as_raw() as usize;

        // Phase 3: Record ALL dynamic `BuildRaytracingAccelerationStructure` calls contiguously.
        let mut refit_count = 0u64;
        for item in &mut planned {
            if (item.flags & ALLOW_COMPACTION) != 0
                && (item.flags & PERFORM_UPDATE) == 0
                && item.dst_gen != 0
                && sub.compact_queries.len() < COMPACT_INFO_SLOTS
                && let Some((uav_buf, uav_gpu, rb_buf)) = sub.ensure_compact_info(host)
            {
                if compact_uav_active.is_none() {
                    // SAFETY: open list; transition compact UAV once per list (legacy fallback only).
                    unsafe {
                        emit_dxr_buffer_legacy_only(
                            list,
                            list7,
                            &uav_buf,
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        );
                    }
                    *compact_uav_active = Some((uav_buf, rb_buf));
                }
                let slot = sub.compact_queries.len() as u32;
                sub.compact_queries.push(CompactQueryRecord {
                    guest_key: item.key,
                    generation: item.dst_gen,
                    uncompacted_size: item.dst_slice_size,
                    slot,
                    auto_compact: false,
                });
                item.host_postbuild.push(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                    DestBuffer: uav_gpu + u64::from(slot) * 8,
                    InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
                });
            }

            let (need_barrier, scratch_off) = scratch_batch.alloc(
                scratch_id,
                scratch_cap,
                item.scratch_need,
                item.host_dst_va,
                item.host_src_va,
            );
            let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(item.flags as i32),
                NumDescs: item.host_geoms.len() as u32,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: item.host_geoms.as_ptr(),
                },
            };
            let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                DestAccelerationStructureData: item.host_dst_va,
                Inputs: inputs,
                SourceAccelerationStructureData: item.host_src_va,
                ScratchAccelerationStructureData: scratch_gpu + scratch_off,
            };
            // SAFETY: open list4; dest, src, scratch, geometry buffers and postbuild buffers live.
            unsafe {
                if need_barrier {
                    emit_as_build_barrier(list, list7, use_compute);
                }
                list4.BuildRaytracingAccelerationStructure(
                    &desc,
                    (!item.host_postbuild.is_empty()).then_some(item.host_postbuild.as_slice()),
                );
            }
            if item.is_refit {
                refit_count += 1;
            }
        }

        // Phase 4: Emit 1 `BUILD_AS -> COMPUTE` barrier, then bind `BUILD_NORMALS_HLSL` ONCE
        // for all dirty dynamic BLASes in this batch (`ARCH-14`).
        if any_normals_work
            && let (Some((n_buf, n_gpu)), Some(n_pass)) = (normals_gpu, n_pass)
        {
            // SAFETY: open list; all shadow VB/IB/XF buffers are kept alive in `sub.kept_shadows`.
            unsafe {
                emit_blas_to_compute_barrier(list, list7, use_compute);
                if *normals_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
                    if *normals_state == D3D12_RESOURCE_STATE_COPY_DEST {
                        emit_dxr_buffer_barrier(
                            list,
                            list7,
                            n_buf,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            *normals_state,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            use_compute,
                        );
                    } else if *normals_state != D3D12_RESOURCE_STATE_COMMON {
                        emit_dxr_buffer_legacy_only(
                            list,
                            list7,
                            n_buf,
                            *normals_state,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        );
                    }
                    *normals_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                list.SetComputeRootSignature(&n_pass.root);
                list.SetPipelineState(&n_pass.pso);
                list.SetComputeRootUnorderedAccessView(4, *n_gpu);

                let mut last_vb = 0u64;
                let mut last_ib = 0u64;
                let mut last_xf = 0u64;
                for item in &planned {
                    if !item.need_normals_dispatch {
                        continue;
                    }
                    let fallback_srv = item
                        .host_geoms
                        .iter()
                        .find_map(|g| {
                            (g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES).then(|| {
                                // SAFETY: `Type == TRIANGLES`.
                                let va = g.Anonymous.Triangles.VertexBuffer.StartAddress & !3;
                                (va != 0).then_some(va)
                            })?
                        })
                        .unwrap_or(0);
                    let mut tri_cursor = 0u32;
                    for (g_idx, (g, &tc)) in item.host_geoms.iter().zip(&item.geom_tris).enumerate() {
                        let geom_entry_word = item.ng_off + g_idx as u32;
                        let prim_data_off = if tc > 0 { item.nd_off + tri_cursor } else { 0 };
                        tri_cursor = tri_cursor.saturating_add(tc);
                        if g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES && tc > 0 {
                            let t = g.Anonymous.Triangles;
                            let vb_fmt = match t.VertexFormat {
                                DXGI_FORMAT_R32G32B32_FLOAT => 0u32,
                                DXGI_FORMAT_R32G32_FLOAT => 1,
                                DXGI_FORMAT_R16G16B16A16_FLOAT => 2,
                                DXGI_FORMAT_R16G16B16A16_SNORM => 3,
                                DXGI_FORMAT_R16G16_FLOAT => 4,
                                DXGI_FORMAT_R16G16_SNORM => 5,
                                _ => 0,
                            };
                            let ib_mode = match t.IndexFormat {
                                DXGI_FORMAT_R16_UINT if t.IndexBuffer != 0 => 1u32,
                                DXGI_FORMAT_R32_UINT if t.IndexBuffer != 0 => 2,
                                _ => 0,
                            };
                            let vb_base = t.VertexBuffer.StartAddress & !3;
                            let vb_bias = (t.VertexBuffer.StartAddress & 3) as u32;
                            let ib_base = if ib_mode != 0 { t.IndexBuffer & !3 } else { vb_base };
                            let ib_bias = if ib_mode != 0 { (t.IndexBuffer & 3) as u32 } else { 0 };
                            let xf_base = if t.Transform3x4 != 0 { t.Transform3x4 } else { vb_base };
                            let has_xf = u32::from(t.Transform3x4 != 0);
                            let constants = [
                                tc,
                                t.VertexBuffer.StrideInBytes as u32,
                                vb_bias,
                                vb_fmt,
                                ib_mode,
                                ib_bias,
                                has_xf,
                                geom_entry_word,
                                prim_data_off,
                                t.VertexCount,
                            ];
                            list.SetComputeRoot32BitConstants(0, 10, constants.as_ptr().cast(), 0);
                            if vb_base != last_vb {
                                list.SetComputeRootShaderResourceView(1, vb_base);
                                last_vb = vb_base;
                            }
                            if ib_base != last_ib {
                                list.SetComputeRootShaderResourceView(2, ib_base);
                                last_ib = ib_base;
                            }
                            if xf_base != last_xf {
                                list.SetComputeRootShaderResourceView(3, xf_base);
                                last_xf = xf_base;
                            }
                            list.Dispatch(tc.div_ceil(64), 1, 1);
                        } else if item.normals_newly_allocated && fallback_srv != 0 {
                            // Write 0 into Region 1 slot once for newly allocated empty/AABB geometry.
                            let constants = [0u32, 0, 0, 0, 0, 0, 0, geom_entry_word, 0, 0];
                            list.SetComputeRoot32BitConstants(0, 10, constants.as_ptr().cast(), 0);
                            if fallback_srv != last_vb {
                                list.SetComputeRootShaderResourceView(1, fallback_srv);
                                last_vb = fallback_srv;
                            }
                            if fallback_srv != last_ib {
                                list.SetComputeRootShaderResourceView(2, fallback_srv);
                                last_ib = fallback_srv;
                            }
                            if fallback_srv != last_xf {
                                list.SetComputeRootShaderResourceView(3, fallback_srv);
                                last_xf = fallback_srv;
                            }
                            list.Dispatch(1, 1, 1);
                        }
                    }
                }
            }
        }

        STATS[STAT_DYNAMIC_BLAS].fetch_add(planned.len() as u64, Ordering::Relaxed);
        if refit_count > 0 {
            STATS[STAT_BLAS_REFITS].fetch_add(refit_count, Ordering::Relaxed);
        }
        any_recorded = true;
    }

    any_recorded
}

/// FNV-1a fingerprint of resolved dynamic BLAS vertex/index/transform GPU VAs & strides
/// so unskinned refits with unchanged buffers can safely skip redundant `BUILD_NORMALS`.
fn blas_geom_va_hash(geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    mix(geoms.len() as u64);
    for g in geoms {
        if g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            // SAFETY: `Type == TRIANGLES`.
            let t = unsafe { g.Anonymous.Triangles };
            mix(t.VertexBuffer.StartAddress);
            mix(t.VertexBuffer.StrideInBytes);
            mix(t.IndexBuffer);
            mix(t.Transform3x4);
        } else {
            // SAFETY: `Type == PROCEDURAL_PRIMITIVE_AABBS`.
            let a = unsafe { g.Anonymous.AABBs };
            mix(a.AABBs.StartAddress);
            mix(a.AABBs.StrideInBytes);
        }
    }
    if h == 0 { 1 } else { h }
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_copy_as(
    host: &Host,
    state: &mut DxrState,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    list7: Option<&ID3D12GraphicsCommandList7>,
    dest_va: u64,
    source_va: u64,
    mode: u32,
    use_compute: bool,
) -> bool {
    let dst_key = dest_va & !0xFF;
    let src_key = source_va & !0xFF;
    if dst_key == 0 || src_key == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    #[derive(Clone, Copy)]
    struct CopyAsSrcMeta {
        gpu_va: u64,
        slice_size: u64,
        compacted_size: u64,
        kind: AsKind,
        flags: u32,
        num_instances: u32,
        built: bool,
        topo_hash: u64,
        update_scratch_bytes: u64,
        consecutive_refits: u32,
        normals_geom_off: u32,
        normals_geom_len: u32,
        normals_data_off: u32,
        normals_data_len: u32,
    }
    let src = {
        let map = as_map().read().unwrap_or_else(|e| e.into_inner());
        map.get(&src_key).map(|e| CopyAsSrcMeta {
            gpu_va: e.gpu_va,
            slice_size: e.slice.size,
            compacted_size: e.compacted_size,
            kind: e.kind,
            flags: e.flags,
            num_instances: e.num_instances,
            built: e.built,
            topo_hash: e.topo_hash,
            update_scratch_bytes: e.update_scratch_bytes,
            consecutive_refits: e.consecutive_refits,
            normals_geom_off: e.normals_geom_off,
            normals_geom_len: e.normals_geom_len,
            normals_data_off: e.normals_data_off,
            normals_data_len: e.normals_data_len,
        })
    };
    let Some(src) = src else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if !src.built {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    const ALLOW_COMPACTION: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    let host_mode = match mode {
        0 => D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        1 => {
            if src.flags & ALLOW_COMPACTION != 0 {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT
            } else {
                // Fall back to CLONE if our host build did not set ALLOW_COMPACTION.
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE
            }
        }
        _ => {
            STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
            return false;
        }
    };
    let is_clone = host_mode == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE;
    // D3D12 forbids overlapping Dest == Src on CopyRaytracingAccelerationStructure.
    // Self-CLONE is an identity no-op; self-COMPACT allocates a fresh non-overlapping slice.
    if dst_key == src_key && is_clone {
        STATS[STAT_COPIES].fetch_add(1, Ordering::Relaxed);
        return true;
    }
    let (dst_flags, dst_topo_hash, dst_update_scratch, dst_refits, dst_size, dst_tier) = if is_clone {
        (
            src.flags,
            src.topo_hash,
            src.update_scratch_bytes,
            src.consecutive_refits,
            src.slice_size,
            SlabTier::TransientDynamicTlas,
        )
    } else {
        // Compacted copies cannot be compacted again or refitted with PERFORM_UPDATE.
        // Size destination to the read-back `compacted_size` when known and place it in
        // the `StaticCompacted` 32 MiB slab tier (`ARCH-10`).
        let sz = if src.compacted_size > 0 && src.compacted_size < src.slice_size {
            src.compacted_size.max(AS_ALIGN).next_multiple_of(AS_ALIGN)
        } else {
            src.slice_size
        };
        (
            src.flags & !(ALLOW_COMPACTION | ALLOW_UPDATE),
            0,
            0,
            0,
            sz,
            SlabTier::StaticCompacted,
        )
    };
    let Some(dst) = (if dst_key == src_key {
        allocate_fresh_as(
            host,
            state,
            dst_key,
            src.kind,
            dst_tier,
            dst_size,
            dst_flags,
            src.num_instances,
            dst_topo_hash,
            dst_update_scratch,
            dst_refits,
        )
    } else {
        allocate_or_reuse_as(
            host,
            state,
            dst_key,
            src.kind,
            dst_tier,
            dst_size,
            dst_flags,
            src.num_instances,
            dst_topo_hash,
            dst_update_scratch,
            dst_refits,
        )
    }) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    // SAFETY: open list4; both AS buffers are in RAYTRACING_ACCELERATION_STRUCTURE state.
    unsafe {
        list4.CopyRaytracingAccelerationStructure(dst.gpu_va, src.gpu_va, host_mode);
        emit_as_copy_barrier(list, list7, use_compute);
    }
    if !is_clone {
        // Transfer normals table ownership to the compacted destination and retire transient
        // uncompacted `src_key` strictly behind `tlas_build_epoch + 2 && sub_seq + 2` (`ARCH-10`).
        let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
        if let Some(dst_entry) = map.get_mut(&dst_key)
            && dst_entry.normals_geom_len == 0
            && src.normals_geom_len > 0
        {
            dst_entry.normals_geom_off = src.normals_geom_off;
            dst_entry.normals_geom_len = src.normals_geom_len;
            dst_entry.normals_data_off = src.normals_data_off;
            dst_entry.normals_data_len = src.normals_data_len;
        }
        if dst_key != src_key
            && let Some(mut old_src) = map.remove(&src_key)
        {
            if old_src.normals_geom_off == src.normals_geom_off {
                old_src.normals_geom_len = 0;
                old_src.normals_data_len = 0;
            }
            if old_src.kind == AsKind::Blas {
                state.remove_blas_hash(src_key);
            }
            state.retire_as(&old_src);
        }
    }
    if src.kind == AsKind::Blas {
        state.insert_blas_hash(dst_key, dst.gpu_va, src.normals_geom_off);
    }
    STATS[STAT_COPIES].fetch_add(1, Ordering::Relaxed);
    true
}

#[allow(clippy::too_many_arguments)]
unsafe fn record_extract_static_blas(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    owner: &HostList,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    list7: Option<&ID3D12GraphicsCommandList7>,
    scratch_batch: &mut ScratchBatcher,
    dest_va: u64,
    size_bytes: u64,
    extract_in_srv: &mut bool,
    staging_in_copy_src: &mut Option<ID3D12Resource>,
    compact_uav_active: &mut Option<(ID3D12Resource, ID3D12Resource)>,
    use_compute: bool,
) -> bool {
    let key = dest_va & !0xFF;
    if key == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }
    let Some((size_bytes, max_nodes)) = probe_bvh_bounds(key, size_bytes) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let Some(pass) = extract_bvh4_pass(host) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let mem_binding = crate::descriptor::with_view_heap(|heap| {
        let ready = crate::memory::ready(owner, heap, &[], &[], &[], crate::memory::Use::Reads, None)?;
        Some((heap.heap.clone(), heap.gpu_handle(ready.base), ready.directory))
    })
    .flatten();
    let Some((view_heap, mem_table_gpu, directory_gpu)) = mem_binding else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };

    let vert_bytes = max_nodes.saturating_mul(72); // 2 tris × 3 verts × 12B float3
    let prev_buf_raw = sub.extract_verts.as_ref().map(|(b, _)| b.buf.as_raw());
    let Some((vert_buf, vert_gpu)) = sub.ensure_extract_verts(host, vert_bytes) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    if prev_buf_raw != Some(vert_buf.as_raw()) {
        *extract_in_srv = false;
    }

    // Check up front whether an async multi-geometry upgrade readback will be queued.
    // When `dump_setup.is_some()`, this Frame-0 single-geometry BLAS will be replaced
    // within 1–2 frames, so skip `ALLOW_COMPACTION` and postbuild queries on it.
    let dump_bytes = 16u64 + max_nodes.saturating_mul(64);
    let dump_setup = if sub.readback_used.saturating_add(dump_bytes) <= sub.readback_cap
        && let Some((rb_buf, _)) = sub.node_readback.as_ref().map(|(b, sz, _)| (b.clone(), *sz))
        && let Some((staging_buf, staging_gpu)) = sub.ensure_node_staging(host, dump_bytes)
    {
        Some((staging_buf, staging_gpu, rb_buf))
    } else {
        None
    };
    let can_requeue_readback = dump_setup.is_none()
        && dump_bytes <= MAX_READBACK_BYTES_PER_SUB
        && state.pending_readback_requeue.len() < MAX_PENDING_READBACK_REQUEUE;
    let want_compact = compact_static_blas() && dump_setup.is_none() && !can_requeue_readback;

    let geom = D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: static_geom_flags(true),
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_UNKNOWN,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: 0,
                VertexCount: (max_nodes * 6) as u32,
                IndexBuffer: 0,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vert_gpu,
                    StrideInBytes: 12,
                },
            },
        },
    };
    let build_flags = static_blas_build_flags_for(want_compact);
    let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: build_flags,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: &geom,
        },
    };
    let mut pre = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: valid device5 and local geometry desc.
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut pre) };
    if pre.ResultDataMaxSizeInBytes == 0 {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    }

    let Some(host_dst) = allocate_or_reuse_as(
        host,
        state,
        key,
        AsKind::Blas,
        SlabTier::TransientDynamicTlas,
        pre.ResultDataMaxSizeInBytes,
        build_flags.0 as u32,
        0,
        0,
        0,
        0,
    ) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let scratch_need = pre.ScratchDataSizeInBytes.max(AS_ALIGN);
    let Some((scratch_buf, scratch_gpu, scratch_cap)) = sub.ensure_scratch(host, scratch_need) else {
        STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let scratch_id = scratch_buf.as_raw() as usize;

    // Reserve a slice of this submission's readback buffer for async multi-geometry upgrade,
    // or re-queue into `pending_readback_requeue` if this burst exceeded `MAX_READBACK_BYTES_PER_SUB`.
    let dump_copy = if let Some((staging_buf, staging_gpu, rb_buf)) = dump_setup {
        let rb_off = sub.readback_used;
        sub.readback_used = rb_off.saturating_add(dump_bytes).next_multiple_of(64);
        sub.readbacks.push(StaticReadbackRecord {
            guest_key: key,
            generation: host_dst.generation,
            offset: rb_off,
            len: dump_bytes,
        });
        crate::memory::pin_extracted(key, size_bytes);
        sub.extracted_bvh4s.push((key, size_bytes));
        Some((staging_buf, staging_gpu, rb_buf, rb_off, dump_bytes))
    } else {
        if can_requeue_readback {
            // Keep `memory.rs` pages pinned until the requeued readback finishes.
            crate::memory::pin_extracted(key, size_bytes);
            state
                .pending_readback_requeue
                .push_back((key, size_bytes, host_dst.generation));
        } else {
            // Un-dumpable oversized BVH4: release its `memory.rs` pages once this extract finishes.
            crate::memory::pin_extracted(key, size_bytes);
            sub.extracted_bvh4s.push((key, size_bytes));
        }
        None
    };

    // SAFETY: open list4; view_heap, directory, vert_buf, staging/readback, AS slab, and scratch are live.
    unsafe {
        if *extract_in_srv {
            emit_dxr_buffer_barrier(
                list,
                list7,
                &vert_buf,
                D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                use_compute,
            );
            *extract_in_srv = false;
        }
        if let Some((staging_buf, _, _, _, _)) = &dump_copy {
            if staging_in_copy_src
                .as_ref()
                .is_some_and(|prev| prev.as_raw() == staging_buf.as_raw())
            {
                emit_dxr_buffer_barrier(
                    list,
                    list7,
                    staging_buf,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    use_compute,
                );
            } else {
                if let Some(prev) = staging_in_copy_src.take() {
                    emit_dxr_buffer_legacy_only(
                        list,
                        list7,
                        &prev,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    );
                }
                emit_dxr_buffer_legacy_only(
                    list,
                    list7,
                    staging_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
        }
        list.SetDescriptorHeaps(&[Some(view_heap)]);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        let (dump_nodes, u1_gpu) = match &dump_copy {
            Some((_, staging_gpu, _, _, _)) => (max_nodes as u32, *staging_gpu),
            None => (0u32, vert_gpu),
        };
        let constants = [
            key as u32,
            (key >> 32) as u32,
            max_nodes as u32,
            size_bytes.min(u64::from(u32::MAX)) as u32,
            dump_nodes,
        ];
        list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
        list.SetComputeRootShaderResourceView(1, directory_gpu);
        list.SetComputeRootUnorderedAccessView(2, vert_gpu);
        list.SetComputeRootUnorderedAccessView(3, u1_gpu);
        list.SetComputeRootDescriptorTable(4, mem_table_gpu);
        let groups = (max_nodes as u32).div_ceil(64);
        list.Dispatch(groups, 1, 1);

        if let Some((staging_buf, _, rb_buf, rb_off, dump_len)) = dump_copy {
            emit_dxr_buffer_barrier(
                list,
                list7,
                &staging_buf,
                D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                use_compute,
            );
            list.CopyBufferRegion(&rb_buf, rb_off, &staging_buf, 0, dump_len);
            *staging_in_copy_src = Some(staging_buf);
        }

        emit_dxr_buffer_barrier(
            list,
            list7,
            &vert_buf,
            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            use_compute,
        );
        *extract_in_srv = true;

        let compact_postbuild = if want_compact
            && sub.compact_queries.len() < COMPACT_INFO_SLOTS
            && let Some((uav_buf, uav_gpu, rb_buf)) = sub.ensure_compact_info(host)
        {
            if compact_uav_active.is_none() {
                emit_dxr_buffer_legacy_only(
                    list,
                    list7,
                    &uav_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                *compact_uav_active = Some((uav_buf, rb_buf));
            }
            let slot = sub.compact_queries.len() as u32;
            sub.compact_queries.push(CompactQueryRecord {
                guest_key: key,
                generation: host_dst.generation,
                uncompacted_size: host_dst.slice_size,
                slot,
                auto_compact: true,
            });
            Some([D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: uav_gpu + u64::from(slot) * 8,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE,
            }])
        } else {
            None
        };

        let (need_barrier, scratch_off) =
            scratch_batch.alloc(scratch_id, scratch_cap, scratch_need, host_dst.gpu_va, 0);
        if need_barrier {
            emit_as_build_barrier(list, list7, use_compute);
        }
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: host_dst.gpu_va,
            Inputs: inputs,
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: scratch_gpu + scratch_off,
        };
        list4.BuildRaytracingAccelerationStructure(
            &desc,
            compact_postbuild.as_ref().map(|p| p.as_slice()),
        );
    }
    state.insert_blas_hash(key, host_dst.gpu_va, 0);
    STATS[STAT_STATIC_BLAS].fetch_add(1, Ordering::Relaxed);
    true
}

/// Records a contiguous run of `DxrOp::BuildTlas` operations in batched form:
/// splits only when two ops target the same destination TLAS VA, uploads the
/// dirty BLAS hash table once, dispatches all `PATCH_INSTANCES_HLSL` passes into
/// disjoint 256B-aligned slices of one `host_instances` buffer, transitions it
/// once, executes all TLAS builds back-to-back with disjoint scratch slices, and
/// emits a single trailing barrier per sub-batch.
#[allow(clippy::too_many_arguments)]
unsafe fn record_build_tlas_batch(
    host: &Host,
    device5: &ID3D12Device5,
    state: &mut DxrState,
    sub: &mut DxrSubmission,
    list: &ID3D12GraphicsCommandList,
    list4: &ID3D12GraphicsCommandList4,
    list7: Option<&ID3D12GraphicsCommandList7>,
    scratch_batch: &mut ScratchBatcher,
    tlas_ops: &[DxrOp],
    hash_in_srv: &mut bool,
    host_inst_in_srv: &mut bool,
    normals_gpu: Option<&(ID3D12Resource, u64)>,
    normals_state: &mut D3D12_RESOURCE_STATES,
    built_tlas: &mut bool,
    use_compute: bool,
) -> bool {
    const ALLOW_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
    const PERFORM_UPDATE: u32 = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
    const PREFER_FAST_BUILD: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;
    const PREFER_FAST_TRACE: u32 =
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0 as u32;

    struct PatchSpec {
        aligned_base: u64,
        src_offset: u32,
        inst_off: u64,
        mirror_inst_off: u64,
        mirror_rel_off: u32,
        num_instances: u32,
        tlas_inst_base: u32,
    }
    struct PreparedTlas {
        dest_gpu_va: u64,
        mirror_dest_gpu_va: Option<u64>,
        flags: u32,
        effective_instances: u32,
        patch: Option<PatchSpec>,
        scratch_need: u64,
        postbuild: Vec<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC>,
    }

    let want_mirror = mirror_tlas_enabled();
    let dest_keys: Vec<u64> = tlas_ops
        .iter()
        .map(|op| match *op {
            DxrOp::BuildTlas { dest_va, .. } => dest_va & !0xFF,
            _ => 0,
        })
        .collect();
    let Some(hash_gpu_buf) = state.ensure_hash_gpu(host) else {
        STATS[STAT_SKIPPED].fetch_add(tlas_ops.len() as u64, Ordering::Relaxed);
        return false;
    };
    let pass = patch_instances_pass(host);
    #[cfg(feature = "diagnostics")]
    let stats_gpu = state.ensure_patch_stats(host).map(|(_, g)| g);
    #[cfg(not(feature = "diagnostics"))]
    let stats_gpu: Option<u64> = Some(0);

    let mut any_recorded = false;
    for range in plan_tlas_sub_batches(&dest_keys) {
        let slice = &tlas_ops[range];
        let mut prepared: Vec<PreparedTlas> = Vec::with_capacity(slice.len());
        let mut total_inst_bytes = 0u64;
        let mut max_scratch = AS_ALIGN;
        let mut sum_scratch = 0u64;

        for op in slice {
            let DxrOp::BuildTlas {
                dest_va,
                flags: raw_flags,
                num_instances,
                instance_descs_va,
                postbuild,
                ..
            } = op
            else {
                continue;
            };
            let key = *dest_va & !0xFF;
            if key == 0 {
                STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Always rebuild TLAS from scratch with PREFER_FAST_TRACE: D3D12 forbids TLAS
            // `PERFORM_UPDATE` whenever any instance transitions `AccelerationStructure` or
            // `InstanceMask` between 0 and non-0, and a 3.5K-instance fast-trace TLAS build
            // takes <0.1 ms while reducing overlapping AABB candidate callbacks in AnyHit loops.
            let mut effective_instances = *num_instances;
            let mut patch = None;
            let flags = (*raw_flags & !(ALLOW_UPDATE | PERFORM_UPDATE | PREFER_FAST_BUILD)) | PREFER_FAST_TRACE;
            let (result_bytes, scratch_bytes) = if effective_instances > 0 && *instance_descs_va != 0 {
                let inst_bytes = u64::from(effective_instances).saturating_mul(64);
                let guest_shadow = crate::shadow::find(*instance_descs_va, inst_bytes);
                if let (Some(guest_shadow), Some(_), Some(_)) = (guest_shadow, pass, stats_gpu)
                    && guest_shadow.left(*instance_descs_va) >= inst_bytes
                {
                    let (rb, sb) = state.tlas_prebuild_info(device5, flags, effective_instances);
                    if rb > 0 {
                        let primary_aligned = inst_bytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
                        let inst_off = total_inst_bytes;
                        let build_mirror = want_mirror && key != EMPTY_TLAS_KEY;
                        let (mirror_inst_off, mirror_rel_off) = if build_mirror {
                            let m_off = inst_off.saturating_add(primary_aligned);
                            total_inst_bytes = m_off.saturating_add(primary_aligned);
                            (m_off, primary_aligned as u32)
                        } else {
                            total_inst_bytes = inst_off.saturating_add(primary_aligned);
                            (0, 0)
                        };
                        let guest_gpu = guest_shadow.gpu(*instance_descs_va);
                        // Periodic TLAS-reachability liveness scan for deterministic `AS_MAP` GC (`ARCH-10`).
                        let scan_epoch = state.tlas_build_epoch.saturating_add(1);
                        if scan_epoch <= 16 || (scan_epoch & 7) < 2 {
                            if guest_shadow.written() {
                                state.last_unreadable_tlas_epoch = scan_epoch;
                            } else if inst_bytes <= (4 << 20) {
                                let len = inst_bytes as usize;
                                state.inst_scan_buf.resize(len, 0);
                                crate::host::copy_mapped(
                                    state.inst_scan_buf.as_mut_ptr(),
                                    *instance_descs_va as *const u8,
                                    len,
                                    false,
                                );
                                let map = as_map().read().unwrap_or_else(|e| e.into_inner());
                                for chunk in state.inst_scan_buf.chunks_exact(64) {
                                    let va = u64::from_le_bytes(chunk[56..64].try_into().unwrap());
                                    if va != 0
                                        && let Some(entry) = map.get(&(va & !0xFF))
                                    {
                                        entry.last_seen_tlas_epoch.store(scan_epoch, Ordering::Relaxed);
                                    }
                                }
                                state.last_readable_tlas_scan_epoch = scan_epoch;
                            }
                        }
                        sub.kept_shadows.push(guest_shadow);
                        patch = Some(PatchSpec {
                            aligned_base: guest_gpu & !255,
                            src_offset: (guest_gpu & 255) as u32,
                            inst_off,
                            mirror_inst_off,
                            mirror_rel_off,
                            num_instances: effective_instances,
                            tlas_inst_base: if normals_gpu.is_some() { tlas_meta_base(key) } else { 0 },
                        });
                    }
                    (rb, sb)
                } else {
                    effective_instances = 0;
                    state.tlas_prebuild_info(device5, flags, 0)
                }
            } else {
                effective_instances = 0;
                state.tlas_prebuild_info(device5, flags, 0)
            };
            if result_bytes == 0 {
                STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            }

            let Some(host_dst) = allocate_or_reuse_as(
                host,
                state,
                key,
                AsKind::Tlas,
                SlabTier::TransientDynamicTlas,
                result_bytes,
                flags,
                effective_instances,
                0,
                0,
                0,
            ) else {
                STATS[STAT_SKIPPED].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let mirror_dest_gpu_va = host_dst.mirror_gpu_va;
            if mirror_dest_gpu_va.is_none()
                && let Some(p) = patch.as_mut()
            {
                p.mirror_rel_off = 0;
            }
            let scratch_need = scratch_bytes.max(AS_ALIGN).next_multiple_of(AS_ALIGN);
            let tlas_count = if mirror_dest_gpu_va.is_some() { 2u64 } else { 1u64 };
            max_scratch = max_scratch.max(scratch_need);
            sum_scratch = sum_scratch.saturating_add(scratch_need.saturating_mul(tlas_count));
            let host_postbuild = translate_postbuild(sub, postbuild);
            prepared.push(PreparedTlas {
                dest_gpu_va: host_dst.gpu_va,
                mirror_dest_gpu_va,
                flags,
                effective_instances,
                patch,
                scratch_need,
                postbuild: host_postbuild,
            });
        }

        if prepared.is_empty() {
            continue;
        }

        let inst_alloc = if total_inst_bytes > 0 {
            let prev_raw = sub.host_instances.as_ref().map(|b| b.buf.as_raw());
            let Some((inst_buf, inst_gpu)) = sub.ensure_host_instances(host, total_inst_bytes) else {
                STATS[STAT_SKIPPED].fetch_add(prepared.len() as u64, Ordering::Relaxed);
                continue;
            };
            if prev_raw != Some(inst_buf.as_raw()) {
                *host_inst_in_srv = false;
            }
            Some((inst_buf, inst_gpu))
        } else {
            None
        };

        // Upload only modified ranges of the GPU BLAS hash table ONCE before patching.
        if inst_alloc.is_some() && state.hash_dirty.is_dirty() {
            let Some((upload_buf, upload_ptr)) = sub.ensure_hash_upload(host) else {
                STATS[STAT_SKIPPED].fetch_add(prepared.len() as u64, Ordering::Relaxed);
                continue;
            };
            let ranges = state.hash_dirty.take_ranges();
            if !ranges.is_empty() {
                // SAFETY: `upload_ptr` maps `HASH_TABLE_BYTES` matching `state.blas_table`.
                unsafe {
                    if *hash_in_srv {
                        emit_dxr_buffer_barrier(
                            list,
                            list7,
                            &hash_gpu_buf,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            use_compute,
                        );
                    }
                    for (start_slot, end_slot) in ranges {
                        let byte_off = (start_slot * HASH_ENTRY_BYTES) as u64;
                        let byte_len = ((end_slot - start_slot + 1) * HASH_ENTRY_BYTES) as u64;
                        std::ptr::copy_nonoverlapping(
                            state.blas_table.as_ptr().add(start_slot).cast::<u8>(),
                            upload_ptr.add(byte_off as usize),
                            byte_len as usize,
                        );
                        list.CopyBufferRegion(&hash_gpu_buf, byte_off, &upload_buf, byte_off, byte_len);
                    }
                    emit_dxr_buffer_barrier(
                        list,
                        list7,
                        &hash_gpu_buf,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        use_compute,
                    );
                }
                *hash_in_srv = true;
            }
        }

        let want_scratch = sum_scratch.min(MAX_POOLED_SCRATCH_BYTES).max(max_scratch);
        let Some((scratch_buf, scratch_gpu, scratch_cap)) = sub.ensure_scratch(host, want_scratch) else {
            STATS[STAT_SKIPPED].fetch_add(prepared.len() as u64, Ordering::Relaxed);
            continue;
        };
        let scratch_id = scratch_buf.as_raw() as usize;

        // Patch all TLAS instance buffers back-to-back into disjoint 256B-aligned slices.
        if let (Some((inst_buf, inst_gpu)), Some(pass)) = (inst_alloc.as_ref(), pass) {
            let u1_gpu = if PATCH_ENABLE_STATS != 0 {
                stats_gpu.unwrap_or(*inst_gpu)
            } else {
                *inst_gpu
            };
            let u2_gpu = normals_gpu.map_or(*inst_gpu, |(_, g)| *g);
            // SAFETY: open list; all root views and buffers live through submission.
            unsafe {
                if let Some((n_buf, _)) = normals_gpu
                    && *normals_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                {
                    if *normals_state == D3D12_RESOURCE_STATE_COPY_DEST {
                        emit_dxr_buffer_barrier(
                            list,
                            list7,
                            n_buf,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            *normals_state,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            use_compute,
                        );
                    } else if *normals_state != D3D12_RESOURCE_STATE_COMMON {
                        emit_dxr_buffer_legacy_only(
                            list,
                            list7,
                            n_buf,
                            *normals_state,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        );
                    }
                    *normals_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                }
                if *host_inst_in_srv {
                    emit_dxr_buffer_barrier(
                        list,
                        list7,
                        inst_buf,
                        D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        use_compute,
                    );
                    *host_inst_in_srv = false;
                }
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                list.SetComputeRootShaderResourceView(2, hash_gpu_buf.GetGPUVirtualAddress());
                list.SetComputeRootUnorderedAccessView(4, u1_gpu);
                list.SetComputeRootUnorderedAccessView(5, u2_gpu);
                for item in &prepared {
                    if let Some(p) = &item.patch {
                        let constants = [
                            p.num_instances,
                            p.src_offset,
                            HASH_MASK,
                            cull_mask_cfg(),
                            PATCH_ENABLE_STATS,
                            p.tlas_inst_base,
                            p.mirror_rel_off,
                        ];
                        list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, p.aligned_base);
                        list.SetComputeRootUnorderedAccessView(3, *inst_gpu + p.inst_off);
                        list.Dispatch(p.num_instances.div_ceil(64), 1, 1);
                    }
                }
                emit_dxr_buffer_barrier(
                    list,
                    list7,
                    inst_buf,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    use_compute,
                );
            }
            *hash_in_srv = true;
            *host_inst_in_srv = true;
        }

        // Build all TLASes (and cull-enabled mirror TLASes) in this sub-batch back-to-back
        // with disjoint scratch slices, emitting a single trailing barrier once done.
        let base_inst_gpu = inst_alloc.as_ref().map_or(0, |(_, g)| *g);
        // SAFETY: open list4; dest, scratch, host_instances, and postbuild buffers live.
        unsafe {
            for item in &prepared {
                let (need_barrier, scratch_off) =
                    scratch_batch.alloc(scratch_id, scratch_cap, item.scratch_need, item.dest_gpu_va, 0);
                if need_barrier {
                    emit_as_build_barrier(list, list7, use_compute);
                }
                let host_inst_gpu = item.patch.as_ref().map_or(0, |p| base_inst_gpu + p.inst_off);
                let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                    Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                    Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(item.flags as i32),
                    NumDescs: item.effective_instances,
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        InstanceDescs: host_inst_gpu,
                    },
                };
                let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: item.dest_gpu_va,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: scratch_gpu + scratch_off,
                };
                list4.BuildRaytracingAccelerationStructure(
                    &desc,
                    (!item.postbuild.is_empty()).then_some(item.postbuild.as_slice()),
                );
                if let Some(mirror_va) = item.mirror_dest_gpu_va {
                    let (m_barrier, m_scratch_off) =
                        scratch_batch.alloc(scratch_id, scratch_cap, item.scratch_need, mirror_va, 0);
                    if m_barrier {
                        emit_as_build_barrier(list, list7, use_compute);
                    }
                    let mirror_inst_gpu = item
                        .patch
                        .as_ref()
                        .filter(|p| p.mirror_rel_off != 0)
                        .map_or(0, |p| base_inst_gpu + p.mirror_inst_off);
                    let m_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(item.flags as i32),
                        NumDescs: item.effective_instances,
                        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                            InstanceDescs: mirror_inst_gpu,
                        },
                    };
                    let m_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: mirror_va,
                        Inputs: m_inputs,
                        SourceAccelerationStructureData: 0,
                        ScratchAccelerationStructureData: scratch_gpu + m_scratch_off,
                    };
                    list4.BuildRaytracingAccelerationStructure(&m_desc, None);
                }
            }
            scratch_batch.flush(list, list7, use_compute);
        }

        if !*built_tlas {
            state.tlas_build_epoch = state.tlas_build_epoch.saturating_add(1);
            *built_tlas = true;
        }
        STATS[STAT_TLAS].fetch_add(prepared.len() as u64, Ordering::Relaxed);
        any_recorded = true;
    }

    any_recorded
}

fn translate_postbuild(
    sub: &mut DxrSubmission,
    postbuild: &[PostbuildSpec],
) -> Vec<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC> {
    if !suppress_builder() || postbuild.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(postbuild.len());
    for pb in postbuild {
        let Some(bytes) = postbuild_byte_size(pb.info_type) else {
            continue;
        };
        if pb.dest_buffer == 0 {
            continue;
        }
        if let Some(s) = crate::shadow::find(pb.dest_buffer, bytes)
            && s.left(pb.dest_buffer) >= bytes
        {
            let gpu = s.gpu(pb.dest_buffer);
            if gpu % 8 != 0 {
                continue;
            }
            sub.kept_shadows.push(s);
            out.push(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: gpu,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_TYPE(pb.info_type as i32),
            });
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn allocate_fresh_as(
    host: &Host,
    state: &mut DxrState,
    guest_key: u64,
    kind: AsKind,
    tier: SlabTier,
    min_bytes: u64,
    flags: u32,
    num_instances: u32,
    topo_hash: u64,
    update_scratch_bytes: u64,
    consecutive_refits: u32,
) -> Option<AllocatedAs> {
    let generation = state.bump_gen();
    let cur_epoch = state.tlas_build_epoch;
    let alloc_bytes = if kind == AsKind::Tlas {
        min_bytes.max(DEFAULT_TLAS_RESERVE)
    } else {
        min_bytes
    };
    let (buffer, slice, gpu_va) = state.slabs.alloc(host, tier, alloc_bytes)?;
    let mirror_tlas = if kind == AsKind::Tlas && mirror_tlas_enabled() && guest_key != EMPTY_TLAS_KEY {
        let m_bytes = min_bytes.max(DEFAULT_MIRROR_TLAS_RESERVE);
        state
            .slabs
            .alloc(host, SlabTier::TransientDynamicTlas, m_bytes)
            .map(|(mbuf, mslice, mgpu)| MirrorTlas {
                buffer: mbuf,
                slice: mslice,
                gpu_va: mgpu,
            })
    } else {
        None
    };
    let mirror_gpu_va = mirror_tlas.as_ref().map(|m| m.gpu_va);
    let allocated = AllocatedAs {
        gpu_va,
        mirror_gpu_va,
        generation,
        slice_size: slice.size,
    };
    let entry = HostAs {
        buffer,
        slice,
        gpu_va,
        mirror_tlas,
        kind,
        flags,
        num_instances,
        built: true,
        generation,
        topo_hash,
        update_scratch_bytes,
        consecutive_refits,
        compacted_size: 0,
        last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
        geom_va_hash: 0,
        normals_geom_off: 0,
        normals_geom_len: 0,
        normals_data_off: 0,
        normals_data_len: 0,
    };
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    if let Some(old) = map.insert(guest_key, entry) {
        state.retire_as(&old);
    }
    drop(map);
    if kind == AsKind::Tlas {
        crate::descriptor::host::bump_buffer_view_cache_epoch();
    }
    Some(allocated)
}

#[allow(clippy::too_many_arguments)]
fn allocate_or_reuse_as(
    host: &Host,
    state: &mut DxrState,
    guest_key: u64,
    kind: AsKind,
    tier: SlabTier,
    min_bytes: u64,
    flags: u32,
    num_instances: u32,
    topo_hash: u64,
    update_scratch_bytes: u64,
    consecutive_refits: u32,
) -> Option<AllocatedAs> {
    let generation = state.bump_gen();
    let cur_epoch = state.tlas_build_epoch;
    let want_mirror = kind == AsKind::Tlas && mirror_tlas_enabled() && guest_key != EMPTY_TLAS_KEY;
    let mut map = as_map().write().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = map.get_mut(&guest_key)
        && existing.kind == kind
        && existing.slice.tier == tier
        && existing.slice.size >= min_bytes
    {
        let mut mirror_replaced = false;
        if want_mirror
            && existing
                .mirror_tlas
                .as_ref()
                .is_none_or(|m| m.slice.size < min_bytes)
        {
            let m_bytes = min_bytes.max(DEFAULT_MIRROR_TLAS_RESERVE);
            if let Some((mbuf, mslice, mgpu)) =
                state.slabs.alloc(host, SlabTier::TransientDynamicTlas, m_bytes)
            {
                if let Some(old_m) = existing.mirror_tlas.replace(MirrorTlas {
                    buffer: mbuf,
                    slice: mslice,
                    gpu_va: mgpu,
                }) {
                    state.slabs.retire(
                        old_m.slice,
                        state.tlas_build_epoch.saturating_add(2),
                        state.sub_seq.saturating_add(2),
                        retire_dxr_fence_target(),
                    );
                }
                mirror_replaced = true;
            }
        }
        existing.flags = flags;
        existing.num_instances = num_instances;
        existing.built = true;
        existing.generation = generation;
        existing.topo_hash = topo_hash;
        existing.update_scratch_bytes = update_scratch_bytes;
        existing.consecutive_refits = consecutive_refits;
        existing.compacted_size = 0;
        existing.last_seen_tlas_epoch.store(cur_epoch, Ordering::Relaxed);
        let res = AllocatedAs {
            gpu_va: existing.gpu_va,
            mirror_gpu_va: existing.mirror_tlas.as_ref().map(|m| m.gpu_va),
            generation,
            slice_size: existing.slice.size,
        };
        drop(map);
        if mirror_replaced {
            crate::descriptor::host::bump_buffer_view_cache_epoch();
        }
        return Some(res);
    }
    let alloc_bytes = if kind == AsKind::Tlas {
        min_bytes.max(DEFAULT_TLAS_RESERVE)
    } else {
        min_bytes
    };
    let (buffer, slice, gpu_va) = state.slabs.alloc(host, tier, alloc_bytes)?;
    let mirror_tlas = if want_mirror {
        let m_bytes = min_bytes.max(DEFAULT_MIRROR_TLAS_RESERVE);
        state
            .slabs
            .alloc(host, SlabTier::TransientDynamicTlas, m_bytes)
            .map(|(mbuf, mslice, mgpu)| MirrorTlas {
                buffer: mbuf,
                slice: mslice,
                gpu_va: mgpu,
            })
    } else {
        None
    };
    let mirror_gpu_va = mirror_tlas.as_ref().map(|m| m.gpu_va);
    let allocated = AllocatedAs {
        gpu_va,
        mirror_gpu_va,
        generation,
        slice_size: slice.size,
    };
    let entry = HostAs {
        buffer,
        slice,
        gpu_va,
        mirror_tlas,
        kind,
        flags,
        num_instances,
        built: true,
        generation,
        topo_hash,
        update_scratch_bytes,
        consecutive_refits,
        compacted_size: 0,
        last_seen_tlas_epoch: AtomicU64::new(cur_epoch),
        geom_va_hash: 0,
        normals_geom_off: 0,
        normals_geom_len: 0,
        normals_data_off: 0,
        normals_data_len: 0,
    };
    if let Some(old) = map.insert(guest_key, entry) {
        state.retire_as(&old);
    }
    drop(map);
    if kind == AsKind::Tlas {
        crate::descriptor::host::bump_buffer_view_cache_epoch();
    }
    Some(allocated)
}

/// FNV-1a topology fingerprint of a BLAS's geometry descriptors so `PERFORM_UPDATE`
/// only runs when geometry count, types, flags, formats, counts, and transform
/// nullness match the source BLAS identically (D3D12 DXR spec requirement).
fn blas_topology_hash(geoms: &[D3D12_RAYTRACING_GEOMETRY_DESC]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    mix(geoms.len() as u64);
    for g in geoms {
        mix(g.Type.0 as u64);
        mix(g.Flags.0 as u64);
        if g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            // SAFETY: `Type == TRIANGLES`.
            let t = unsafe { g.Anonymous.Triangles };
            mix(t.IndexFormat.0 as u64);
            mix(t.VertexFormat.0 as u64);
            mix(u64::from(t.IndexCount));
            mix(u64::from(t.VertexCount));
            mix(u64::from(t.Transform3x4 != 0));
            mix(u64::from(t.IndexBuffer != 0));
            mix(u64::from(t.VertexBuffer.StartAddress != 0));
        } else {
            // SAFETY: `Type == PROCEDURAL_PRIMITIVE_AABBS`.
            let a = unsafe { g.Anonymous.AABBs };
            mix(a.AABBCount);
            mix(u64::from(a.AABBs.StartAddress != 0));
        }
    }
    if h == 0 { 1 } else { h }
}

// ============================================================================
// Pure-Rust unpacked BVH4 TriangleNode decoder (CPU, zero D3D12 dependency)
// ============================================================================

/// Extracts one triangle's 3 vertices from a 64-byte TriangleNode's 4 vertices
/// using `TRIANGLE_VERTICES` (`tri0: [0,1,2]`, `tri1: [1,3,2]`) and the winding
/// permutation byte from `dw[15]`. Returns `None` for invalid/inactive triangles.
fn decode_node_triangle(verts: &[[f32; 3]; 4], map: [usize; 3], id_byte: u32) -> Option<[[f32; 3]; 3]> {
    let i = (id_byte & 3) as usize;
    let j = ((id_byte >> 2) & 3) as usize;
    if i >= 3 || j >= 3 || i == j {
        return None;
    }
    let first = (3 - i - j) & 3;
    let tri = [verts[map[first]], verts[map[i]], verts[map[j]]];
    if tri.iter().any(|v| v.iter().any(|c| c.is_nan())) {
        return None;
    }
    Some(tri)
}

/// Decodes a readback buffer containing `[num_elements, offset_to_elements, dump_nodes, size_bytes]`
/// followed by `dump_nodes` unpacked 64-byte console BVH4 TriangleNodes into a multi-geometry
/// [`PendingStaticUpgrade`] preserving exact `GeometryIndex`, `PrimitiveIndex`, and `OPAQUE` flags.
///
/// Returns `None` if the dump is invalid, empty, exceeds hard OOM bounds, or is already
/// bit-exact with Frame 0's single-geometry opaque sequential extraction.
fn decode_static_bvh4_nodes(guest_key: u64, generation: u64, data: &[u8]) -> Option<PendingStaticUpgrade> {
    if data.len() < 16 {
        return None;
    }
    let read_u32 = |off: usize| u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    let num_elements = read_u32(0) as usize;
    let offset_to_elements = read_u32(4);
    let dump_nodes = read_u32(8) as usize;
    let size_bytes = read_u32(12);
    if offset_to_elements < 64 || num_elements == 0 || dump_nodes == 0 || size_bytes < offset_to_elements {
        return None;
    }
    let avail_nodes = (data.len() - 16) / 64;
    let max_by_size = ((size_bytes - offset_to_elements) / 64) as usize;
    let node_count = num_elements.min(dump_nodes).min(avail_nodes).min(max_by_size);
    if node_count == 0 {
        return None;
    }

    #[derive(Clone, Copy)]
    struct GeomMeta {
        has_tris: bool,
        opaque: bool,
        max_prim: u32,
    }
    let mut geoms: Vec<GeomMeta> = Vec::new();
    let mut needs_upgrade = false;
    let mut any_tri = false;

    let parse_node = |node_idx: usize| -> ([[f32; 3]; 4], u32, u32, u32, u32) {
        let base = 16 + node_idx * 64;
        let dw = |k: usize| read_u32(base + k * 4);
        let verts = [
            [f32::from_bits(dw(0)), f32::from_bits(dw(1)), f32::from_bits(dw(2))],
            [f32::from_bits(dw(3)), f32::from_bits(dw(4)), f32::from_bits(dw(5))],
            [f32::from_bits(dw(6)), f32::from_bits(dw(7)), f32::from_bits(dw(8))],
            [f32::from_bits(dw(9)), f32::from_bits(dw(10)), f32::from_bits(dw(11))],
        ];
        (verts, dw(12), dw(13), dw(14), dw(15))
    };

    // Pass 1: discover per-geometry primitive bounds, opacity, and upgrade necessity.
    for node_idx in 0..node_count {
        let (verts, dw12, prim0, prim1, dw15) = parse_node(node_idx);
        let geom_idx = (dw12 >> 8) as usize;
        let is_opaque = (dw12 & 1) != 0;
        let has_tri1 = (dw12 & 0x80) != 0;

        let t0 = decode_node_triangle(&verts, [0, 1, 2], dw15 & 0xFF);
        let t1 = if has_tri1 {
            decode_node_triangle(&verts, [1, 3, 2], (dw15 >> 8) & 0xFF)
        } else {
            None
        };
        if t0.is_none() && t1.is_none() {
            continue;
        }
        if geom_idx >= MAX_UPGRADE_GEOMS {
            return None;
        }
        if geoms.len() <= geom_idx {
            geoms.resize(
                geom_idx + 1,
                GeomMeta {
                    has_tris: false,
                    opaque: true,
                    max_prim: 0,
                },
            );
        }
        let meta = &mut geoms[geom_idx];
        for (sub_idx, tri_opt, prim_idx) in [(0usize, t0, prim0), (1usize, t1, prim1)] {
            if tri_opt.is_none() {
                continue;
            }
            if (prim_idx as usize) >= MAX_UPGRADE_TRIS {
                return None;
            }
            any_tri = true;
            if !meta.has_tris || prim_idx > meta.max_prim {
                meta.max_prim = prim_idx;
            }
            meta.has_tris = true;
            meta.opaque &= is_opaque;
            let frame0_prim = (node_idx * 2 + sub_idx) as u32;
            if geom_idx != 0 || !is_opaque || prim_idx != frame0_prim {
                needs_upgrade = true;
            }
        }
    }

    if !any_tri || !needs_upgrade {
        return None;
    }

    let mut total_tris = 0usize;
    let mut geom_tri_offsets = Vec::with_capacity(geoms.len());
    let mut upgraded_geoms = Vec::with_capacity(geoms.len());
    for g in &geoms {
        let prim_count = if g.has_tris { (g.max_prim as usize) + 1 } else { 0 };
        total_tris = total_tris.checked_add(prim_count)?;
        if total_tris > MAX_UPGRADE_TRIS {
            return None;
        }
        let offset_tris = total_tris - prim_count;
        geom_tri_offsets.push(offset_tris);
        upgraded_geoms.push(UpgradedGeom {
            vert_offset: (offset_tris * 3) as u32,
            vert_count: (prim_count * 3) as u32,
            opaque: g.opaque,
        });
    }

    // Pass 2: scatter valid triangles into exact `PrimitiveIndex` slots; holes stay NaN.
    let nan = f32::from_bits(0x7FC0_0000);
    let mut vertices = vec![[nan; 3]; total_tris * 3];
    for node_idx in 0..node_count {
        let (verts, dw12, prim0, prim1, dw15) = parse_node(node_idx);
        let geom_idx = (dw12 >> 8) as usize;
        let has_tri1 = (dw12 & 0x80) != 0;

        if let Some(tri0) = decode_node_triangle(&verts, [0, 1, 2], dw15 & 0xFF) {
            let slot = (geom_tri_offsets[geom_idx] + prim0 as usize) * 3;
            vertices[slot] = tri0[0];
            vertices[slot + 1] = tri0[1];
            vertices[slot + 2] = tri0[2];
        }
        if has_tri1 && let Some(tri1) = decode_node_triangle(&verts, [1, 3, 2], (dw15 >> 8) & 0xFF) {
            let slot = (geom_tri_offsets[geom_idx] + prim1 as usize) * 3;
            vertices[slot] = tri1[0];
            vertices[slot + 1] = tri1[1];
            vertices[slot + 2] = tri1[2];
        }
    }

    // Pass 3 (offloaded to `victory-dxr-decode`, off `victory-submit`): precompute
    // per-geometry relative offsets and packed octahedral geometric triangle normals (`ARCH-14`).
    let mut geom_table = Vec::with_capacity(upgraded_geoms.len());
    let mut normals_data = Vec::with_capacity(total_tris);
    let mut tri_cursor = 0u32;
    for g in &upgraded_geoms {
        let g_tris = g.vert_count / 3;
        geom_table.push(if g_tris > 0 { tri_cursor } else { u32::MAX });
        let v_off = g.vert_offset as usize;
        for t in 0..(g_tris as usize) {
            let b = v_off + t * 3;
            let packed = if b + 2 < vertices.len() {
                let v0 = vertices[b];
                let v1 = vertices[b + 1];
                let v2 = vertices[b + 2];
                let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let nx = e1[1] * e2[2] - e1[2] * e2[1];
                let ny = e1[2] * e2[0] - e1[0] * e2[2];
                let nz = e1[0] * e2[1] - e1[1] * e2[0];
                victory_shader::emit::pack_oct_normal_u32(nx, ny, nz)
            } else {
                victory_shader::emit::DXR_NORMALS_FALLBACK
            };
            normals_data.push(packed);
        }
        tri_cursor = tri_cursor.saturating_add(g_tris);
    }

    Some(PendingStaticUpgrade {
        guest_key,
        generation,
        geometries: upgraded_geoms,
        vertices,
        geom_table,
        normals_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_dump(num_elements: u32, offset_to_elements: u32, dump_nodes: u32, nodes: &[[u32; 16]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + nodes.len() * 64);
        let size_bytes = offset_to_elements.saturating_add((nodes.len() as u32) * 64);
        for w in [num_elements, offset_to_elements, dump_nodes, size_bytes] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for n in nodes {
            for w in n {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
        out
    }

    fn make_node(verts: [[f32; 3]; 4], dw12: u32, prim0: u32, prim1: u32, dw15: u32) -> [u32; 16] {
        let mut dw = [0u32; 16];
        for (vi, v) in verts.iter().enumerate() {
            dw[vi * 3] = v[0].to_bits();
            dw[vi * 3 + 1] = v[1].to_bits();
            dw[vi * 3 + 2] = v[2].to_bits();
        }
        dw[12] = dw12;
        dw[13] = prim0;
        dw[14] = prim1;
        dw[15] = dw15;
        dw
    }

    #[test]
    fn decode_multi_geom_opacity_winding_and_sparse_prims() {
        let v = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        // id_byte: canonical [0,1,2] is i=1, j=2 -> (2<<2)|1 = 0x09; rotated [1,2,0] is i=2, j=0 -> (0<<2)|2 = 0x02.
        let n0 = make_node(v, (0 << 8) | 0x80 | 1, 0, 1, 0x09 | (0x09 << 8));
        // Geometry 2 (non-opaque, leaving geometry 1 empty), sparse prim0=5 (rotated), prim1=2 (canonical).
        let n1 = make_node(v, (2 << 8) | 0x80, 5, 2, 0x02 | (0x09 << 8));
        let dump = pack_dump(2, 128, 2, &[n0, n1]);

        let upg = decode_static_bvh4_nodes(0x1000, 7, &dump).expect("upgrade expected");
        assert_eq!(upg.guest_key, 0x1000);
        assert_eq!(upg.generation, 7);
        assert_eq!(
            upg.geometries,
            vec![
                UpgradedGeom { vert_offset: 0, vert_count: 6, opaque: true },
                UpgradedGeom { vert_offset: 6, vert_count: 0, opaque: true },
                UpgradedGeom { vert_offset: 6, vert_count: 18, opaque: false },
            ]
        );
        assert_eq!(upg.vertices.len(), 24);
        // Geom 0 prim 0: [v0, v1, v2]
        assert_eq!(&upg.vertices[0..3], &[v[0], v[1], v[2]]);
        // Geom 0 prim 1 (tri1 map [1,3,2]): [v1, v3, v2]
        assert_eq!(&upg.vertices[3..6], &[v[1], v[3], v[2]]);
        // Geom 2 prim 0, 1, 3, 4 are inactive NaN holes.
        for hole_prim in [0usize, 1, 3, 4] {
            let s = 6 + hole_prim * 3;
            assert!(upg.vertices[s][0].is_nan(), "expected NaN at hole prim {hole_prim}");
        }
        // Geom 2 prim 2 (tri1 canonical): [v1, v3, v2]
        assert_eq!(&upg.vertices[6 + 2 * 3..6 + 3 * 3], &[v[1], v[3], v[2]]);
        // Geom 2 prim 5 (tri0 rotated i=2,j=0 => first=1 => [v1, v2, v0])
        assert_eq!(&upg.vertices[6 + 5 * 3..6 + 6 * 3], &[v[1], v[2], v[0]]);
    }

    #[test]
    fn decode_mixed_opacity_in_same_geom_is_non_opaque() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let n0 = make_node(v, (0 << 8) | 1, 0, 0, 0x09);
        let n1 = make_node(v, 0 << 8, 1, 0, 0x09);
        let dump = pack_dump(2, 64, 2, &[n0, n1]);

        let upg = decode_static_bvh4_nodes(0x2000, 1, &dump).expect("non-opaque must upgrade");
        assert_eq!(upg.geometries.len(), 1);
        assert!(!upg.geometries[0].opaque);
        assert_eq!(upg.geometries[0].vert_count, 6);
    }

    #[test]
    fn decode_skips_when_frame0_already_exact_and_upgrades_on_reorder() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        let exact = make_node(v, (0 << 8) | 0x80 | 1, 0, 1, 0x09 | (0x09 << 8));
        assert!(decode_static_bvh4_nodes(0x3000, 1, &pack_dump(1, 64, 1, &[exact])).is_none());

        // Same single opaque geometry, but PrimitiveIndex swapped -> must upgrade.
        let swapped = make_node(v, (0 << 8) | 0x80 | 1, 1, 0, 0x09 | (0x09 << 8));
        let upg = decode_static_bvh4_nodes(0x3000, 1, &pack_dump(1, 64, 1, &[swapped]))
            .expect("reordered prims must upgrade");
        assert_eq!(&upg.vertices[0..3], &[v[1], v[3], v[2]]);
        assert_eq!(&upg.vertices[3..6], &[v[0], v[1], v[2]]);
    }

    #[test]
    fn decode_rejects_corrupt_bounds_and_ignores_unmapped_nodes() {
        let v = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
        // Geom index out of bounds
        let bad_geom = make_node(v, ((MAX_UPGRADE_GEOMS as u32) << 8) | 1, 0, 0, 0x09);
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 64, 1, &[bad_geom])).is_none());
        // Prim index out of bounds
        let bad_prim = make_node(v, 1, MAX_UPGRADE_TRIS as u32, 0, 0x09);
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 64, 1, &[bad_prim])).is_none());
        // Corrupt header offset < 64
        assert!(decode_static_bvh4_nodes(0x4000, 1, &pack_dump(1, 32, 1, &[make_node(v, 0, 0, 0, 0x09)])).is_none());
        // Unmapped sentinel (dw15 = 0xFFFF_FFFF) alongside valid non-opaque node is skipped cleanly.
        let unmapped = make_node([[0.0; 3]; 4], 0xFFFF_FF00, 0xDEAD_BEEF, 0xDEAD_BEEF, 0xFFFF_FFFF);
        let valid = make_node(v, 0, 3, 0, 0x09);
        let upg = decode_static_bvh4_nodes(0x4000, 1, &pack_dump(2, 64, 2, &[unmapped, valid]))
            .expect("valid node should survive unmapped neighbor");
        assert_eq!(upg.geometries.len(), 1);
        assert_eq!(upg.geometries[0].vert_count, 12);
    }

    #[test]
    fn extract_bvh4_hlsl_compiles() {
        let code = compile(EXTRACT_BVH4_HLSL, s!("main"), s!("cs_5_1")).expect("EXTRACT_BVH4_HLSL must compile");
        assert!(!code.is_empty());
    }

    #[test]
    fn slab_allocator_deferred_retire_coalesce_and_best_fit_reuse() {
        let mut alloc = SlabAllocator::<usize>::new(SlabTier::TransientDynamicTlas);
        let make = |slab_size: u64, slab_idx: usize| -> Option<(usize, u64)> {
            let _ = slab_size;
            Some((slab_idx, (slab_idx as u64 + 1) << 32))
        };

        // Allocate four adjacent 256B-aligned slices inside slab 0.
        let (_, a, va_a) = alloc.alloc_with(200, make).unwrap();
        let (_, b, _) = alloc.alloc_with(512, make).unwrap();
        let (_, c, _) = alloc.alloc_with(256, make).unwrap();
        let (_, d, _) = alloc.alloc_with(1024, make).unwrap();
        assert_eq!(a, SlabSlice { tier: SlabTier::TransientDynamicTlas, slab_idx: 0, offset: 0, size: 256 });
        assert_eq!(b, SlabSlice { tier: SlabTier::TransientDynamicTlas, slab_idx: 0, offset: 256, size: 512 });
        assert_eq!(c, SlabSlice { tier: SlabTier::TransientDynamicTlas, slab_idx: 0, offset: 768, size: 256 });
        assert_eq!(d, SlabSlice { tier: SlabTier::TransientDynamicTlas, slab_idx: 0, offset: 1024, size: 1024 });
        assert_eq!(va_a, 1u64 << 32);

        // Defer-retire `b` (BLAS: requires tlas_epoch >= 3 && sub_seq >= 3 && dxr_fence >= 4)
        // and `d` (non-TLAS: requires sub_seq >= 2 && dxr_fence >= 0).
        alloc.retire(b, 3, 3, 4);
        alloc.retire(d, 0, 2, 0);

        // Neither is released at (tlas=1, sub=1, dxr=10).
        assert_eq!(alloc.drain_retired(1, 1, 10), 0);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().free, Vec::<(u64, u64)>::new());
        assert_eq!(alloc.slabs[0].as_ref().unwrap().used, 2048);

        // At (tlas=1, sub=2, dxr=0), `d` (tail) is released without being blocked by `b` and folds back into `used`.
        assert_eq!(alloc.drain_retired(1, 2, 0), 1);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().used, 1024);
        assert!(alloc.slabs[0].as_ref().unwrap().free.is_empty());

        // At (tlas=2, sub=3, dxr=4), `b` is still retained because tlas_epoch < 3.
        assert_eq!(alloc.drain_retired(2, 3, 4), 0);
        // At (tlas=3, sub=3, dxr=3), `b` is still retained because dxr_fence < 4 (Spec-Trap #16).
        assert_eq!(alloc.drain_retired(3, 3, 3), 0);
        // At (tlas=3, sub=3, dxr=4), `b` is freed into the free list.
        assert_eq!(alloc.drain_retired(3, 3, 4), 1);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().free, vec![(256, 512)]);

        // Free `a` (adjacent left) and `c` (adjacent right + touches tail): all three coalesce and reset `used` to 0!
        alloc.free(a);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().free, vec![(0, 768)]);
        alloc.free(c);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().used, 0);
        assert!(alloc.slabs[0].as_ref().unwrap().free.is_empty());

        // Reclaiming an empty 32 MiB slab releases it cleanly and reuses its slot index on next alloc.
        let reclaimed = alloc.reclaim_empty_slabs();
        assert_eq!(reclaimed, vec![(0usize, AS_SLAB_SIZE)]);
        assert!(alloc.slabs[0].is_none());
        assert_eq!(alloc.stats(), (0, 0));

        // Best-fit reuse: create two holes (512B and 256B), allocate 256B -> picks exact 256B hole.
        let (_, x0, _) = alloc.alloc_with(512, make).unwrap();
        let (_, _keep0, _) = alloc.alloc_with(256, make).unwrap();
        let (_, x1, _) = alloc.alloc_with(256, make).unwrap();
        let (_, _keep1, _) = alloc.alloc_with(256, make).unwrap();
        alloc.free(x0);
        alloc.free(x1);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().free, vec![(0, 512), (768, 256)]);

        let (_, reused, va_reused) = alloc.alloc_with(256, make).unwrap();
        assert_eq!(reused, x1);
        assert_eq!(va_reused, (1u64 << 32) + 768);
        assert_eq!(alloc.slabs[0].as_ref().unwrap().free, vec![(0, 512)]);

        let (total, free) = alloc.stats();
        assert_eq!(total, AS_SLAB_SIZE);
        assert_eq!(free, (AS_SLAB_SIZE - 1280) + 512);
    }

    #[test]
    fn hash_dirty_tracker_coalesces_slots_and_overflows_to_bounding_range() {
        let mut t = HashDirtyTracker::full();
        assert!(t.is_dirty());
        assert_eq!(t.take_ranges(), vec![(0, HASH_CAPACITY - 1)]);
        assert!(!t.is_dirty());
        assert!(t.take_ranges().is_empty());

        // Nearby slots merge; distant slots stay separate.
        t.mark_slot(100);
        t.mark_slot(100 + HASH_DIRTY_MERGE_GAP);
        t.mark_slot(50_000);
        t.mark_slot(100); // duplicate
        assert_eq!(
            t.take_ranges(),
            vec![(100, 100 + HASH_DIRTY_MERGE_GAP), (50_000, 50_000)]
        );
        assert!(!t.is_dirty());

        // Exceeding MAX_DIRTY_HASH_SLOTS collapses into a single [min..=max] range.
        for i in 0..=MAX_DIRTY_HASH_SLOTS {
            t.mark_slot(1000 + i * 500);
        }
        assert_eq!(t.take_ranges(), vec![(1000, 1000 + MAX_DIRTY_HASH_SLOTS * 500)]);
        assert!(!t.is_dirty());
    }

    #[test]
    fn geometry_flags_set_no_duplicate_anyhit_on_non_opaque() {
        let opaque = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        let no_dup = D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;

        assert_eq!(dynamic_geom_flags_with(0, false), no_dup);
        assert_eq!(dynamic_geom_flags_with(opaque.0 as u32, false), opaque);
        assert_eq!(dynamic_geom_flags_with((opaque | no_dup).0 as u32, false), opaque | no_dup);
        assert_eq!(dynamic_geom_flags_with(0, true), no_dup);
        assert_eq!(dynamic_geom_flags_with(opaque.0 as u32, true), opaque | no_dup);
        assert_eq!(dynamic_geom_flags_with((opaque | no_dup).0 as u32, true), opaque | no_dup);

        assert_eq!(static_geom_flags_with(false, false), no_dup);
        assert_eq!(static_geom_flags_with(true, false), opaque);
        assert_eq!(static_geom_flags_with(false, true), no_dup);
        assert_eq!(static_geom_flags_with(true, true), opaque | no_dup);
    }

    #[test]
    fn should_compact_filters_trivial_and_invalid_savings() {
        // Zero, equal, or larger compacted sizes are rejected.
        assert!(!should_compact(65_536, 0));
        assert!(!should_compact(65_536, 65_536));
        assert!(!should_compact(65_536, 70_000));
        // Saving <= 4 KiB (even after 256B alignment) is rejected.
        assert!(!should_compact(16_384, 12_288));
        assert!(!should_compact(16_384, 12_033)); // aligns to 12_288 -> saves 4096
        // Small BLAS saving > 4 KiB and >= 10% compacts.
        assert!(should_compact(16_384, 8_192));
        assert!(should_compact(65_536, 32_768));
        // Large BLAS saving only 8 KiB (< 10% and < 64 KiB) is skipped.
        assert!(!should_compact(1 << 20, (1 << 20) - 8192));
        // Large BLAS saving >= 64 KiB compacts even if under 10%.
        assert!(should_compact(4 << 20, (4 << 20) - 128 * 1024));
    }

    #[test]
    fn compaction_retires_old_blas_slice_until_tlas_epoch_plus_two() {
        let mut alloc = SlabAllocator::<usize>::default();
        let make = |_slab_size: u64, slab_idx: usize| -> Option<(usize, u64)> {
            Some((slab_idx, (slab_idx as u64 + 1) << 32))
        };

        // Simulate uncompacted static BLAS (256 KiB) followed by async compaction to 128 KiB.
        let (_, uncompacted, _) = alloc.alloc_with(256 * 1024, make).unwrap();
        assert!(should_compact(uncompacted.size, 128 * 1024));
        let (_, compacted, _) = alloc.alloc_with(128 * 1024, make).unwrap();
        assert_eq!(compacted.offset, 256 * 1024);
        assert_eq!(compacted.size, 128 * 1024);

        // Old uncompacted slice retired at tlas_epoch=5, sub_seq=10, dxr_fence=8 -> releases at (7, 12, 10).
        alloc.retire(uncompacted, 7, 12, 10);
        assert_eq!(alloc.drain_retired(6, 12, 10), 0);
        assert_eq!(alloc.drain_retired(7, 11, 10), 0);
        assert_eq!(alloc.drain_retired(7, 12, 9), 0);
        assert_eq!(alloc.drain_retired(7, 12, 10), 1);

        // Reclaimed 256 KiB hole is immediately reused by the next static BLAS allocation.
        let (_, reused, _) = alloc.alloc_with(256 * 1024, make).unwrap();
        assert_eq!(reused, uncompacted);
    }

    #[test]
    fn patch_instances_hlsl_compiles() {
        let code = compile(PATCH_INSTANCES_HLSL, s!("main"), s!("cs_5_1"))
            .expect("PATCH_INSTANCES_HLSL must compile for cs_5_1");
        assert!(!code.is_empty());
    }

    #[test]
    fn build_normals_hlsl_compiles() {
        let code = compile(BUILD_NORMALS_HLSL, s!("main"), s!("cs_5_1"))
            .expect("BUILD_NORMALS_HLSL must compile for cs_5_1");
        assert!(!code.is_empty());
    }

    #[test]
    fn word_pool_alloc_retire_coalesce_and_tlas_meta_base() {
        let mut pool = WordPool::new(100, 130);
        let a = pool.alloc(10).unwrap();
        let b = pool.alloc(10).unwrap();
        let c = pool.alloc(10).unwrap();
        assert_eq!((a, b, c), (100, 110, 120));
        assert_eq!(pool.alloc(1), None);

        // Retire `a` and `b` at different epochs; draining coalesces them into one 20-word block
        // only once tlas_epoch, sub_seq, AND dxr_fence all pass.
        pool.retire(a, 10, 5, 10, 7);
        pool.retire(b, 10, 5, 10, 7);
        pool.drain_retired(4, 10, 7);
        assert_eq!(pool.alloc(10), None);
        pool.drain_retired(5, 10, 6);
        assert_eq!(pool.alloc(10), None);
        pool.drain_retired(5, 10, 7);
        assert_eq!(pool.alloc(20), Some(100));

        // Tail free rewinds bump pointer cleanly.
        pool.retire(c, 10, 6, 11, 8);
        pool.drain_retired(6, 11, 7);
        assert_eq!(pool.used, 130);
        pool.drain_retired(6, 11, 8);
        assert_eq!(pool.used, 120);

        // `tlas_meta_base` is deterministic, non-zero for valid TLAS VAs, and 0 for null/empty.
        assert_eq!(tlas_meta_base(0), 0);
        assert_eq!(tlas_meta_base(EMPTY_TLAS_KEY), 0);
        let b1 = tlas_meta_base(0x1_0000_1000);
        let b2 = tlas_meta_base(0x1_0000_2000);
        assert!(b1 >= 1 && b1 < victory_shader::emit::DXR_NORMALS_GEOM_BASE);
        assert!(b2 >= 1 && b2 < victory_shader::emit::DXR_NORMALS_GEOM_BASE);
        assert_ne!(b1, b2);
        assert_eq!(tlas_meta_base(0x1_0000_1000), b1);
    }

    #[test]
    fn cull_mask_cfg_parsing_covers_default_hex_dec_and_tag_modes() {
        assert_eq!(parse_cull_mask_cfg(None), DEFAULT_CULL_MASK_CFG);
        assert_eq!(parse_cull_mask_cfg(Some("")), DEFAULT_CULL_MASK_CFG);
        assert_eq!(parse_cull_mask_cfg(Some("0")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("off")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("false")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("1")), 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("on")), 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("true")), 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("0x80")), 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("128")), 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("0x01")), 0x01);
        assert_eq!(parse_cull_mask_cfg(Some("64")), 0x40);
        assert_eq!(parse_cull_mask_cfg(Some("tag:0x80")), CULL_MASK_TAG_MODE | 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("tag:1")), CULL_MASK_TAG_MODE | 0x80);
        assert_eq!(parse_cull_mask_cfg(Some("TAG:0x40")), CULL_MASK_TAG_MODE | 0x40);
        assert_eq!(parse_cull_mask_cfg(Some("tag:0")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("256")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("0x100")), 0);
        assert_eq!(parse_cull_mask_cfg(Some("bogus")), 0);
    }

    #[test]
    fn postbuild_byte_size_matches_d3d12_spec() {
        assert_eq!(postbuild_byte_size(0), Some(8)); // COMPACTED_SIZE
        assert_eq!(postbuild_byte_size(1), Some(8)); // TOOLS_VISUALIZATION
        assert_eq!(postbuild_byte_size(2), Some(16)); // SERIALIZATION
        assert_eq!(postbuild_byte_size(3), Some(8)); // CURRENT_SIZE
        assert_eq!(postbuild_byte_size(4), None);
        assert_eq!(postbuild_byte_size(u32::MAX), None);
    }

    #[test]
    fn prebuild_flags_and_sanitization_handle_refit_and_empty_inputs() {
        let allow_upd = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
        let perf_upd = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
        let fast_build = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;

        assert_eq!(prebuild_flags(1, fast_build, false), fast_build);
        assert_eq!(prebuild_flags(1, fast_build | perf_upd, true), fast_build | allow_upd);
        assert_eq!(prebuild_flags(0, perf_upd, true), 0);

        let zero = sanitize_prebuild_info(GuestPrebuildInfo::default(), false);
        assert_eq!(zero.result_data_max_size_in_bytes, AS_ALIGN);
        assert_eq!(zero.scratch_data_size_in_bytes, AS_ALIGN);
        assert_eq!(zero.update_scratch_data_size_in_bytes, 0);

        let unaligned = sanitize_prebuild_info(
            GuestPrebuildInfo {
                result_data_max_size_in_bytes: 300,
                scratch_data_size_in_bytes: 1,
                update_scratch_data_size_in_bytes: 17,
            },
            true,
        );
        assert_eq!(unaligned.result_data_max_size_in_bytes, 512);
        assert_eq!(unaligned.scratch_data_size_in_bytes, 256);
        assert_eq!(unaligned.update_scratch_data_size_in_bytes, 256);

        let allow_upd_zero = sanitize_prebuild_info(GuestPrebuildInfo::default(), true);
        assert_eq!(allow_upd_zero.update_scratch_data_size_in_bytes, AS_ALIGN);

        let empty_inputs = GuestBuildAsInputs {
            kind: 1,
            flags: 0,
            num_descs: 0,
            descs_layout: 0,
            ptr_or_va: 0,
        };
        let stream = GuestStreamDesc {
            size_in_bytes: 1000,
            data: std::ptr::null(),
        };
        let fb = fallback_prebuild_info(&empty_inputs, Some(&stream));
        assert_eq!(fb.result_data_max_size_in_bytes, 1024);
        assert_eq!(fb.scratch_data_size_in_bytes, AS_ALIGN);
    }

    #[test]
    fn blas_refit_decision_auto_promotion_period_and_fallbacks() {
        let allow_upd = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
        let perf_upd = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0 as u32;
        let fast_build = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD.0 as u32;
        let fast_trace = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0 as u32;

        assert_eq!(parse_refit_period(None), DEFAULT_REFIT_PERIOD);
        assert_eq!(parse_refit_period(Some("0")), 0);
        assert_eq!(parse_refit_period(Some("128")), 128);
        assert_eq!(parse_refit_period(Some("nope")), DEFAULT_REFIT_PERIOD);

        let existing = ExistingBlasRefitInfo {
            gpu_va: 0xDEAD_0000,
            flags: fast_build | allow_upd,
            built: true,
            topo_hash: 0x1234_5678,
            update_scratch_bytes: 4096,
            consecutive_refits: 3,
        };

        // Refit OFF: ordinary build does not add ALLOW_UPDATE or auto-promote.
        let d_off = decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(existing), None, false, 64);
        assert_eq!(d_off.flags, fast_build);
        assert_eq!(d_off.host_src_va, 0);
        assert_eq!(d_off.in_place_update, None);
        assert!(!d_off.is_refit);

        // Refit ON, first build (no existing): adds ALLOW_UPDATE and does full build.
        let d_first = decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, None, None, true, 64);
        assert_eq!(d_first.flags, fast_build | allow_upd);
        assert_eq!(d_first.host_src_va, 0);
        assert_eq!(d_first.in_place_update, None);
        assert_eq!(d_first.next_consecutive_refits, 0);
        assert!(!d_first.is_refit);

        // Refit ON, matching existing below period: auto-promotes to in-place PERFORM_UPDATE.
        let d_auto = decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(existing), None, true, 64);
        assert_eq!(d_auto.flags, fast_build | allow_upd | perf_upd);
        assert_eq!(d_auto.host_src_va, 0xDEAD_0000);
        assert_eq!(d_auto.in_place_update, Some((0xDEAD_0000, 4096)));
        assert_eq!(d_auto.next_consecutive_refits, 4);
        assert!(d_auto.is_refit);

        // Refit ON, reached refit_period: forces clean full rebuild upgraded to
        // PREFER_FAST_TRACE | ALLOW_UPDATE and resets counter.
        let at_period = ExistingBlasRefitInfo {
            consecutive_refits: 64,
            ..existing
        };
        let d_period = decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(at_period), None, true, 64);
        assert_eq!(d_period.flags, fast_trace | allow_upd);
        assert_eq!(d_period.host_src_va, 0);
        assert_eq!(d_period.in_place_update, None);
        assert_eq!(d_period.next_consecutive_refits, 0);
        assert!(!d_period.is_refit);

        // Subsequent frame after periodic rebuild: existing is PREFER_FAST_TRACE | ALLOW_UPDATE,
        // guest still passes PREFER_FAST_BUILD -> stays compatible and refits with PREFER_FAST_TRACE!
        let promoted = ExistingBlasRefitInfo {
            flags: fast_trace | allow_upd,
            consecutive_refits: 0,
            ..existing
        };
        let d_after_period =
            decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(promoted), None, true, 64);
        assert!(d_after_period.is_refit);
        assert_eq!(d_after_period.flags, fast_trace | allow_upd | perf_upd);
        assert_eq!(d_after_period.in_place_update, Some((0xDEAD_0000, 4096)));
        assert_eq!(d_after_period.next_consecutive_refits, 1);

        // Period == 0 disables forced periodic rebuild even after many refits.
        let d_never = decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(at_period), None, true, 0);
        assert!(d_never.is_refit);
        assert_eq!(d_never.next_consecutive_refits, 65);

        // Topology mismatch falls back to full rebuild.
        let d_topo = decide_blas_refit(fast_build, 0x1000, 0, 0x9999_9999, Some(existing), None, true, 64);
        assert!(!d_topo.is_refit);
        assert_eq!(d_topo.flags, fast_build | allow_upd);

        // Build flag mismatch (PREFER_FAST_TRACE vs existing PREFER_FAST_BUILD) falls back.
        let d_flags = decide_blas_refit(fast_trace, 0x1000, 0, 0x1234_5678, Some(existing), None, true, 64);
        assert!(!d_flags.is_refit);
        assert_eq!(d_flags.flags, fast_trace | allow_upd);

        // Existing without ALLOW_UPDATE or with zero update scratch falls back.
        let no_upd = ExistingBlasRefitInfo { flags: fast_build, ..existing };
        assert!(!decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(no_upd), None, true, 64).is_refit);
        let zero_scratch = ExistingBlasRefitInfo { update_scratch_bytes: 0, ..existing };
        assert!(!decide_blas_refit(fast_build, 0x1000, 0, 0x1234_5678, Some(zero_scratch), None, true, 64).is_refit);

        // Out-of-place explicit PERFORM_UPDATE (src != dst) keeps is_refit without in-place shortcut.
        let d_oop = decide_blas_refit(
            fast_build | allow_upd | perf_upd,
            0x2000,
            0x1000,
            0x1234_5678,
            None,
            Some(existing),
            false,
            64,
        );
        assert!(d_oop.is_refit);
        assert_eq!(d_oop.host_src_va, 0xDEAD_0000);
        assert_eq!(d_oop.in_place_update, None);
        assert_eq!(d_oop.next_consecutive_refits, 4);
    }

    #[test]
    fn scratch_batcher_suballocates_and_flushes_on_hazard_or_exhaustion() {
        let mut b = ScratchBatcher::default();
        let cap = 16 * 1024;
        assert!(!b.has_pending());

        // Three independent builds sub-allocate sequentially with zero barriers.
        assert_eq!(b.alloc(1, cap, 3900, 0x1000, 0), (false, 0)); // aligns to 4096
        assert_eq!(b.alloc(1, cap, 4096, 0x2000, 0), (false, 4096));
        assert_eq!(b.alloc(1, cap, 4096, 0x3000, 0), (false, 8192));
        assert!(b.has_pending());

        // Fourth build overflows 16 KiB capacity -> flushes and restarts at offset 0.
        assert_eq!(b.alloc(1, cap, 8192, 0x4000, 0), (true, 0));

        // Destination VA hazard against 0x4000 -> flushes and restarts at offset 0.
        assert_eq!(b.alloc(1, cap, 256, 0x4000, 0), (true, 0));

        // Source VA hazard reading 0x4000 written in current batch -> flushes.
        assert_eq!(b.alloc(1, cap, 256, 0x5000, 0x4000), (true, 0));

        // Scratch buffer growth (new buf_id) resets cursor to 0 without spurious barrier
        // when VAs are disjoint, but still barriers on VA hazard.
        assert_eq!(b.alloc(2, 32 * 1024, 512, 0x6000, 0), (false, 0));
        assert_eq!(b.alloc(2, 32 * 1024, 512, 0x5000, 0), (true, 0));
    }

    #[test]
    fn tlas_sub_batch_planner_groups_disjoint_dests_and_splits_on_duplicate() {
        assert_eq!(plan_tlas_sub_batches(&[]), Vec::<std::ops::Range<usize>>::new());
        // 5 distinct TLASes collapse into a single batch.
        assert_eq!(
            plan_tlas_sub_batches(&[0x1000, 0x2000, 0x3000, 0x4000, 0x5000]),
            vec![0..5]
        );
        // Repeating a destination VA flushes before the repeat.
        assert_eq!(
            plan_tlas_sub_batches(&[0x1000, 0x2000, 0x1000, 0x3000, 0x3000]),
            vec![0..2, 2..4, 4..5]
        );
        // Zero/invalid keys do not spuriously split batches.
        assert_eq!(
            plan_tlas_sub_batches(&[0, 0x1000, 0, 0x2000]),
            vec![0..4]
        );
    }

    #[test]
    fn tlas_prebuild_cache_hits_ignores_zero_and_bounds_capacity() {
        let mut cache: HashMap<(u32, u32), (u64, u64)> = HashMap::new();
        let mut queries = 0usize;

        // First query populates cache; second query hits without invoking driver.
        let r1 = cached_tlas_prebuild_with(&mut cache, 0x4, 1024, || {
            queries += 1;
            (65_536, 32_768)
        });
        let r2 = cached_tlas_prebuild_with(&mut cache, 0x4, 1024, || {
            queries += 1;
            (999, 999)
        });
        assert_eq!(r1, (65_536, 32_768));
        assert_eq!(r2, (65_536, 32_768));
        assert_eq!(queries, 1);

        // Zero result size is never cached.
        let r_zero = cached_tlas_prebuild_with(&mut cache, 0x4, 0, || {
            queries += 1;
            (0, 0)
        });
        assert_eq!(r_zero, (0, 0));
        assert_eq!(cache.len(), 1);

        // Distinct flags or instance counts get separate entries, and exceeding
        // MAX_TLAS_PREBUILD_CACHE clears before inserting.
        for n in 1..=(MAX_TLAS_PREBUILD_CACHE as u32) {
            cached_tlas_prebuild_with(&mut cache, 0x8, n, || (u64::from(n) * 256, 256));
        }
        assert!(cache.len() <= MAX_TLAS_PREBUILD_CACHE);

        // Verify AllocatedAs is Copy and preserves fields cleanly.
        let alloc = AllocatedAs {
            gpu_va: 0xDEAD_BEEF_0000,
            mirror_gpu_va: Some(0xBEEF_DEAD_0000),
            generation: 42,
            slice_size: 4096,
        };
        let copy = alloc;
        assert_eq!(alloc, copy);
        assert_eq!(
            (copy.gpu_va, copy.mirror_gpu_va, copy.generation, copy.slice_size),
            (0xDEAD_BEEF_0000, Some(0xBEEF_DEAD_0000), 42, 4096)
        );
    }

    #[test]
    fn mirror_tlas_clears_cull_disable_and_preserves_front_ccw_and_other_flags() {
        // r3.y layout: bits 0..23 = InstanceContributionToHitGroupIndex,
        // bits 24..31 = Flags (bit 0 = TRIANGLE_CULL_DISABLE = 0x1,
        // bit 1 = TRIANGLE_FRONT_COUNTERCLOCKWISE = 0x2, bit 2 = FORCE_OPAQUE = 0x4,
        // bit 3 = FORCE_NON_OPAQUE = 0x8).
        let contrib = 0x00AB_CDEFu32;
        let all_flags = 0x0Fu32 << 24;
        let cleared = clear_cull_disable_flag(all_flags | contrib);
        // Bit 0 (TRIANGLE_CULL_DISABLE) cleared; bits 1..3 + 24-bit contribution preserved identically.
        assert_eq!(cleared, (0x0Eu32 << 24) | contrib);
        assert_eq!(cleared & 0x0100_0000, 0);
        assert_eq!(cleared & 0x0200_0000, 0x0200_0000);
        assert_eq!(cleared & 0x00FF_FFFF, contrib);

        // Instance with only TRIANGLE_FRONT_COUNTERCLOCKWISE stays bit-identical.
        let ccw_only = (0x02u32 << 24) | 0x123456;
        assert_eq!(clear_cull_disable_flag(ccw_only), ccw_only);

        // When mirror TLAS is enabled, default cull_mask_cfg is 0 (full 8-bit guest mask preserved),
        // while explicit VICTORY_NATIVE_RT_CULL_MASK overrides still take effect.
        assert_eq!(parse_cull_mask_cfg_for(None, true), 0);
        assert_eq!(parse_cull_mask_cfg_for(Some(""), true), 0);
        assert_eq!(parse_cull_mask_cfg_for(Some("tag:0x02"), true), CULL_MASK_TAG_MODE | 0x02);
        assert_eq!(parse_cull_mask_cfg_for(None, false), DEFAULT_CULL_MASK_CFG);
    }

    #[test]
    fn blas_table_shift_delete_preserves_64_probe_chains_and_wraparound() {
        let mut table = vec![[0u32; 8]; HASH_CAPACITY];
        let mut dirty = HashDirtyTracker::full();
        let _ = dirty.take_ranges();

        // Find 4 distinct guest VAs hashing to the exact same bucket near HASH_CAPACITY - 1
        // so the probe chain wraps across index 65535 -> 0 -> 1 -> 2!
        let target_bucket = HASH_MASK;
        let mut colliding: Vec<u64> = Vec::new();
        let mut next_bucket: Vec<u64> = Vec::new();
        for k in 1u64..1_000_000 {
            let va = k << 8;
            let b = hash_key(k as u32, (k >> 32) as u32) & HASH_MASK;
            if b == target_bucket && colliding.len() < 4 {
                colliding.push(va);
            } else if b == 0 && next_bucket.len() < 2 {
                next_bucket.push(va);
            }
            if colliding.len() == 4 && next_bucket.len() == 2 {
                break;
            }
        }
        assert_eq!(colliding.len(), 4);
        assert_eq!(next_bucket.len(), 2);

        // Insert colliding[0] at 65535, colliding[1] at 0, next_bucket[0] at 1 (home=0),
        // colliding[2] at 2 (home=65535), next_bucket[1] at 3 (home=0), colliding[3] at 4 (home=65535).
        let items = [
            (colliding[0], 0xA000u64, 10u32),
            (colliding[1], 0xA100, 11),
            (next_bucket[0], 0xB000, 20),
            (colliding[2], 0xA200, 12),
            (next_bucket[1], 0xB100, 21),
            (colliding[3], 0xA300, 13),
        ];
        for &(va, host_va, goff) in &items {
            insert_blas_hash_in(&mut table, &mut dirty, va, host_va, goff);
        }
        let _ = dirty.take_ranges();

        // Delete head of the colliding chain (`colliding[0]` at slot 65535):
        // `colliding[1]` must shift back to 65535, `next_bucket[0]` stays at/moves to home 0,
        // and EVERY remaining key stays reachable within 64 probes!
        assert!(shift_delete_blas_hash(&mut table, &mut dirty, colliding[0]));
        assert!(dirty.is_dirty());
        assert_eq!(lookup_blas_hash_in(&table, colliding[0]), None);
        for &(va, host_va, goff) in &items[1..] {
            assert_eq!(
                lookup_blas_hash_in(&table, va),
                Some((host_va, goff)),
                "entry {va:#x} lost after shift-delete of head"
            );
        }

        // Delete middle key (`next_bucket[0]`, home=0) and then `colliding[2]`:
        assert!(shift_delete_blas_hash(&mut table, &mut dirty, next_bucket[0]));
        assert_eq!(lookup_blas_hash_in(&table, next_bucket[0]), None);
        assert!(shift_delete_blas_hash(&mut table, &mut dirty, colliding[2]));
        assert_eq!(lookup_blas_hash_in(&table, colliding[2]), None);
        // Deleting an already-deleted key returns false cleanly.
        assert!(!shift_delete_blas_hash(&mut table, &mut dirty, colliding[2]));

        for &(va, host_va, goff) in &[items[1], items[4], items[5]] {
            assert_eq!(
                lookup_blas_hash_in(&table, va),
                Some((host_va, goff)),
                "entry {va:#x} lost after mid-chain shift-delete"
            );
        }
    }

    #[test]
    fn decide_blas_refit_preserves_allow_compaction_only_blases() {
        let allow_upd = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE.0 as u32;
        let allow_cmp = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION.0 as u32;
        let fast_trace = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE.0 as u32;

        // ALLOW_COMPACTION without ALLOW_UPDATE must NOT get ALLOW_UPDATE ORed in even when refit is ON.
        let cmp_only = fast_trace | allow_cmp;
        assert_eq!(prebuild_flags(1, cmp_only, true), cmp_only);
        let d_cmp = decide_blas_refit(cmp_only, 0x1000, 0, 0xAAAA, None, None, true, 64);
        assert_eq!(d_cmp.flags, cmp_only);
        assert_eq!(d_cmp.flags & allow_upd, 0);
        assert!(!d_cmp.is_refit);

        // If the guest explicitly passes BOTH ALLOW_COMPACTION | ALLOW_UPDATE, keep ALLOW_UPDATE.
        let both = fast_trace | allow_cmp | allow_upd;
        assert_eq!(prebuild_flags(1, both, true), both);
        let d_both = decide_blas_refit(both, 0x1000, 0, 0xAAAA, None, None, true, 64);
        assert_eq!(d_both.flags, both);
    }

    #[test]
    fn dxr_room_kept_dynamic_steady_and_streaming_reserves() {
        // Disabled DXR reserves 0.
        assert_eq!(dxr_room_kept_for(false, None, 0, 0, true), 0);

        // Default steady = 128 MiB, streaming = 256 MiB when aux == 0.
        assert_eq!(dxr_room_kept_for(true, None, 0, 0, false), 128 << 20);
        assert_eq!(dxr_room_kept_for(true, None, 0, 0, true), 256 << 20);

        // Accounts for DXR_AUX_VRAM_BYTES (e.g. 64 MiB normals + 2 MiB hash + 16 MiB scratch = 82 MiB),
        // floored at 1 slab (32 MiB) so growth always has headroom.
        assert_eq!(dxr_room_kept_for(true, None, 400 << 20, 82 << 20, false), 46 << 20);
        assert_eq!(dxr_room_kept_for(true, None, 400 << 20, 82 << 20, true), 174 << 20);
        assert_eq!(dxr_room_kept_for(true, None, 400 << 20, 200 << 20, false), AS_SLAB_SIZE);

        // Explicit VICTORY_CARD_ROOM_FOR_DXR_MB override subtracts both slabs and aux bytes.
        assert_eq!(
            dxr_room_kept_for(true, Some(512), 300 << 20, 80 << 20, false),
            132 << 20
        );
        assert_eq!(
            dxr_room_kept_for(true, Some(256), 300 << 20, 80 << 20, true),
            0
        );
    }

    #[test]
    fn blas_sub_batch_planner_and_decode_worker_prepacked_normals() {
        assert_eq!(plan_blas_sub_batches(&[]), Vec::<std::ops::Range<usize>>::new());
        // Disjoint dynamic BLASes collapse into a single batch (1 AS_MAP lock + 1 BUILD_NORMALS bind).
        assert_eq!(
            plan_blas_sub_batches(&[(0x1000, 0x1000), (0x2000, 0), (0x3000, 0x3000)]),
            vec![0..3]
        );
        // Duplicate destination VA or cross-BLAS source dependency splits cleanly.
        assert_eq!(
            plan_blas_sub_batches(&[(0x1000, 0), (0x2000, 0x1000), (0x2000, 0x2000)]),
            vec![0..1, 1..2, 2..3]
        );

        // Verify `decode_static_bvh4_nodes` precomputes octahedral normals & geom_table off-thread.
        let v = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        // Non-opaque triangle in geom 0 prim 0: cross(v1-v0, v2-v0) = (0, 0, +1).
        let n0 = make_node(v, 0, 0, 0, 0x09);
        let upg = decode_static_bvh4_nodes(0x5000, 1, &pack_dump(1, 64, 1, &[n0]))
            .expect("non-opaque static BLAS must decode");
        assert_eq!(upg.geom_table, vec![0]);
        assert_eq!(upg.normals_data.len(), 1);
        assert_eq!(
            upg.normals_data[0],
            victory_shader::emit::pack_oct_normal_u32(0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn skip_builder_rt_off_and_rehydrate_via_builder_matrix() {
        use std::ffi::OsStr;

        // 1. RT-Off (VICTORY_NATIVE_RT=0, VICTORY_BVH_BUILDER=0):
        //    rehydrate_via_builder is false, builder stays off, skip_builder_rt_off is true.
        assert!(!should_rehydrate_via_builder(false, false, None));
        assert!(!should_rehydrate_via_builder(false, true, None));
        assert!(should_skip_builder_rt_off(true, false, false, false));
        // VICTORY_BVH_SKIP_OFF=0 escape hatch: keep forwarding.
        assert!(!should_skip_builder_rt_off(false, false, false, false));

        // 2. RT-On dual-run (VICTORY_NATIVE_RT=1, VICTORY_BVH_BUILDER=1, SUPPRESS_BUILDER=0):
        //    rehydrate_via_builder is false (builder already runs via VICTORY_BVH_BUILDER=1),
        //    skip_builder_rt_off is false.
        assert!(!should_rehydrate_via_builder(true, false, None));
        assert!(!should_skip_builder_rt_off(true, true, true, true));

        // 3. RT-On suppress-builder (VICTORY_NATIVE_RT=1, SUPPRESS_BUILDER=1, VICTORY_BVH_BUILDER=0):
        //    rehydrate_via_builder is true by default (even when VICTORY_BVH_BUILDER=0),
        //    allowing rehydration compute dispatches while dynamic build/copy stay suppressed.
        assert!(should_rehydrate_via_builder(true, true, None));
        assert!(should_rehydrate_via_builder(true, true, Some(OsStr::new("1"))));
        assert!(!should_rehydrate_via_builder(true, true, Some(OsStr::new("0"))));
        assert!(!should_skip_builder_rt_off(true, true, false, true));

        // Software BVH builder requested or running: never skip.
        assert!(!should_skip_builder_rt_off(true, false, true, false));
        assert!(!should_skip_builder_rt_off(true, false, false, true));
    }

    #[test]
    fn compute_queue_sync_sanitization_strips_graphics_bits_and_narrows_prelude() {
        let graphics_heavy = D3D12_BARRIER_SYNC_ALL_SHADING
            | D3D12_BARRIER_SYNC_PIXEL_SHADING
            | D3D12_BARRIER_SYNC_VERTEX_SHADING
            | D3D12_BARRIER_SYNC_RENDER_TARGET
            | D3D12_BARRIER_SYNC_DEPTH_STENCIL
            | D3D12_BARRIER_SYNC_RAYTRACING
            | D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
            | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE
            | D3D12_BARRIER_SYNC_COPY;

        // DirectQ preserves every bit verbatim.
        assert_eq!(sanitize_sync_for_queue(graphics_heavy, false), graphics_heavy);

        // ComputeQ strips ALL graphics-only bits and maps shading bits to COMPUTE_SHADING.
        let sanitized = sanitize_sync_for_queue(graphics_heavy, true);
        let expected_compute = D3D12_BARRIER_SYNC_COMPUTE_SHADING
            | D3D12_BARRIER_SYNC_COPY
            | D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
            | D3D12_BARRIER_SYNC_COPY_RAYTRACING_ACCELERATION_STRUCTURE;
        assert_eq!(sanitized, expected_compute);
        let illegal_on_compute = (D3D12_BARRIER_SYNC_ALL_SHADING.0
            | D3D12_BARRIER_SYNC_NON_PIXEL_SHADING.0
            | D3D12_BARRIER_SYNC_PIXEL_SHADING.0
            | D3D12_BARRIER_SYNC_VERTEX_SHADING.0
            | D3D12_BARRIER_SYNC_RENDER_TARGET.0
            | D3D12_BARRIER_SYNC_DEPTH_STENCIL.0
            | D3D12_BARRIER_SYNC_DRAW.0
            | D3D12_BARRIER_SYNC_INDEX_INPUT.0
            | D3D12_BARRIER_SYNC_RESOLVE.0) as u32;
        assert_eq!((sanitized.0 as u32) & illegal_on_compute, 0);

        // Pure graphics sync bit never degenerates into SYNC_NONE on ComputeQ.
        assert_eq!(
            sanitize_sync_for_queue(D3D12_BARRIER_SYNC_RENDER_TARGET, true),
            D3D12_BARRIER_SYNC_COMPUTE_SHADING
        );
        assert_eq!(
            sanitize_sync_for_queue(D3D12_BARRIER_SYNC_NONE, true),
            D3D12_BARRIER_SYNC_NONE
        );

        // ComputeQ skips prelude drain (skin_fence orders Direct -> Compute).
        assert_eq!(prelude_barrier_for_ops(true, true, false, false, true), None);
        // DirectQ fallback narrows prelude drain to COMPUTE_SHADING | COPY (no ALL_SHADING / COPY_AS sledgehammer).
        let (sb, sa, _, _) = prelude_barrier_for_ops(true, true, false, false, false).unwrap();
        assert_eq!(sb, D3D12_BARRIER_SYNC_COMPUTE_SHADING | D3D12_BARRIER_SYNC_COPY);
        assert_eq!(
            sa,
            D3D12_BARRIER_SYNC_BUILD_RAYTRACING_ACCELERATION_STRUCTURE
                | D3D12_BARRIER_SYNC_COMPUTE_SHADING
        );
    }
}
