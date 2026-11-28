//! Native D3D12 compute shader texture detiler (`VICTORY_GPU_DETILE=1`).
//!
//! Detiles RDNA2/GFX10.3 tiled and linear textures directly on the host GPU
//! using root constants and raw buffer root descriptors (`ByteAddressBuffer` /
//! `RWByteAddressBuffer`), requiring no descriptor heap so it can be recorded
//! onto any direct or compute command list.
//!
//! # Ring-Buffered Batched Compute Queue Architecture (`ARCH-15`)
//!
//! Instead of serializing every texture through offset 0 of a single scratch
//! buffer with 3 legacy transition barriers per texture, `gpu_detile` uses a
//! multi-MiB (default 24 MiB, `VICTORY_GPU_DETILE_RING_MB`) sub-allocated ring
//! scratch buffer with fence-tracked tail reclamation:
//! - On the dedicated `D3D12_COMMAND_LIST_TYPE_COMPUTE` queue (opt-out with
//!   `VICTORY_GPU_DETILE_COMPUTE=0`), multiple pending tiled textures batch into
//!   **1 compute dispatch pass -> 1 scoped buffer barrier -> all `CopyTextureRegion`s**
//!   followed by a single `tex_fence` signal. Every destination texture enters
//!   and exits the compute queue strictly in `D3D12_RESOURCE_STATE_COMMON`.
//! - On direct prelude command lists, each texture sub-allocates a 512-byte-aligned
//!   slice from the fence-tracked ring buffer and uses Enhanced `D3D12_BUFFER_BARRIER`
//!   when supported (falling back cleanly to legacy barriers otherwise).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_formats::texture::{Dimension, Layout, Swizzle};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, s};

use crate::host::{Host, host};

/// Default scratch ring capacity (48 MiB, sized for large streaming texture batches).
const DEFAULT_RING_MB: u64 = 48;
const MIN_RING_MB: u64 = 16;
const MAX_RING_MB: u64 = 128;
/// `D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT` = 512 bytes so every sub-allocated
/// slice preserves `GetCopyableFootprints` offset alignment.
const RING_ALIGN: u64 = 512;
/// Number of command allocator/list slots on the dedicated compute queue.
const COMPUTE_SLOTS: usize = 4;

/// Telemetry counters (`V2 #5` visibility):
/// `[0]` batches submitted, `[1]` textures detiled, `[2]` compute dispatches,
/// `[3]` linear bytes detiled, `[4]` dedicated compute-queue batches,
/// `[5]` ring wrap-arounds, `[6]` ring wait stalls, `[7]` oversized fallbacks.
static TELEMETRY: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Whether GPU compute texture detiling is enabled (`VICTORY_GPU_DETILE=1`).
/// Opt-in, default off.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_DETILE").is_some_and(|v| v == "1"))
}

fn parse_compute_enabled(val: Option<&str>) -> bool {
    val.is_none_or(|v| v.trim() != "0")
}

/// Whether batched detiles may execute on the dedicated `COMPUTE` queue
/// (`VICTORY_GPU_DETILE_COMPUTE=0` opts out and falls back to the caller's list).
pub(crate) fn compute_queue_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_compute_enabled(std::env::var("VICTORY_GPU_DETILE_COMPUTE").ok().as_deref()))
}

fn parse_ring_bytes(val: Option<&str>) -> u64 {
    let mb = val
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_RING_MB, |mb| mb.clamp(MIN_RING_MB, MAX_RING_MB));
    mb << 20
}

fn ring_bytes() -> u64 {
    static BYTES: OnceLock<u64> = OnceLock::new();
    *BYTES.get_or_init(|| parse_ring_bytes(std::env::var("VICTORY_GPU_DETILE_RING_MB").ok().as_deref()))
}

fn enhanced_barriers_opt_in() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_GPU_DETILE_ENHANCED")
            .ok()
            .is_none_or(|v| v.trim() != "0")
    })
}

fn device_enhanced_barriers(device: &ID3D12Device) -> bool {
    if !enhanced_barriers_opt_in() {
        return false;
    }
    let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
    // SAFETY: plain feature query into a struct of its own size.
    let known = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS12,
            (&raw mut o).cast(),
            size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
        )
    };
    known.is_ok() && o.EnhancedBarriersSupported.as_bool()
}

fn host_enhanced_barriers(host: &Host) -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(|| device_enhanced_barriers(&host.device))
}

fn log_startup_once(host: &Host) {
    static LOGGED: OnceLock<()> = OnceLock::new();
    LOGGED.get_or_init(|| {
        tracing::info!(
            enabled = on(),
            queue = if compute_queue_enabled() { "compute" } else { "direct-prelude" },
            ring_mib = ring_bytes() >> 20,
            barriers = if host_enhanced_barriers(host) { "enhanced" } else { "legacy" },
            "gpu_detile initialized"
        );
    });
}

fn record_telemetry(
    textures: u64,
    dispatches: u64,
    bytes: u64,
    compute_queue: bool,
) {
    let batches = TELEMETRY[0].fetch_add(1, Ordering::Relaxed) + 1;
    let total_tex = TELEMETRY[1].fetch_add(textures, Ordering::Relaxed) + textures;
    let total_disp = TELEMETRY[2].fetch_add(dispatches, Ordering::Relaxed) + dispatches;
    let total_bytes = TELEMETRY[3].fetch_add(bytes, Ordering::Relaxed) + bytes;
    let compute_batches = if compute_queue {
        TELEMETRY[4].fetch_add(1, Ordering::Relaxed) + 1
    } else {
        TELEMETRY[4].load(Ordering::Relaxed)
    };
    let prev_tex = total_tex - textures;
    if prev_tex == 0 || (prev_tex / 2000) < (total_tex / 2000) {
        tracing::info!(
            batches,
            compute_batches,
            textures = total_tex,
            dispatches = total_disp,
            mib = total_bytes >> 20,
            avg_tex_per_batch = format_args!("{:.2}", total_tex as f64 / batches.max(1) as f64),
            ring_wraps = TELEMETRY[5].load(Ordering::Relaxed),
            ring_stalls = TELEMETRY[6].load(Ordering::Relaxed),
            oversized = TELEMETRY[7].load(Ordering::Relaxed),
            "gpu_detile telemetry"
        );
    }
}

/// Pure fence-tracked ring-buffer sub-allocator.
///
/// Maintains the invariant `head = (tail + used) % capacity`, guaranteeing every
/// returned `[offset, offset + aligned_size)` slice is disjoint from all
/// in-flight leases until [`Self::reclaim`] advances `tail` past their fence.
#[derive(Debug)]
struct RingAllocator {
    capacity: u64,
    head: u64,
    tail: u64,
    used: u64,
    current_batch_bytes: u64,
    in_flight: VecDeque<RingLease>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RingLease {
    fence: u64,
    bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RingAlloc {
    /// Sub-allocated at `offset` inside the ring buffer (`wrapped` indicates
    /// wrap-around to offset 0).
    Slice { offset: u64, aligned_size: u64, wrapped: bool },
    /// Ring currently lacks contiguous free space; caller must flush/wait on
    /// [`RingAllocator::oldest_fence`] and retry.
    Full,
    /// Single request exceeds `capacity`; caller must use a transient fallback buffer.
    Oversized,
}

impl RingAllocator {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            head: 0,
            tail: 0,
            used: 0,
            current_batch_bytes: 0,
            in_flight: VecDeque::new(),
        }
    }

    /// Reclaims all leases whose `fence <= completed_fence`.
    fn reclaim(&mut self, completed_fence: u64) -> u64 {
        let mut freed = 0u64;
        while let Some(&front) = self.in_flight.front() {
            if front.fence > completed_fence {
                break;
            }
            self.in_flight.pop_front();
            self.used -= front.bytes;
            self.tail = (self.tail + front.bytes) % self.capacity;
            freed += front.bytes;
        }
        if self.used == 0 {
            self.head = 0;
            self.tail = 0;
        }
        freed
    }

    /// Attempts to reserve an `align`-aligned slice of `size` bytes.
    fn alloc(&mut self, size: u64, align: u64) -> RingAlloc {
        let align = align.max(4);
        let aligned_size = size.max(4).next_multiple_of(align);
        if aligned_size > self.capacity {
            return RingAlloc::Oversized;
        }
        if self.head + aligned_size <= self.capacity {
            if self.used + aligned_size <= self.capacity {
                let offset = self.head;
                self.head = (self.head + aligned_size) % self.capacity;
                self.used += aligned_size;
                self.current_batch_bytes += aligned_size;
                return RingAlloc::Slice {
                    offset,
                    aligned_size,
                    wrapped: false,
                };
            }
            return RingAlloc::Full;
        }
        // Wrap to offset 0, accounting for the unused tail gap `[head, capacity)`.
        let pad = self.capacity - self.head;
        let total = pad + aligned_size;
        if self.used + total <= self.capacity {
            self.head = aligned_size % self.capacity;
            self.used += total;
            self.current_batch_bytes += total;
            return RingAlloc::Slice {
                offset: 0,
                aligned_size,
                wrapped: true,
            };
        }
        RingAlloc::Full
    }

    /// Commits all bytes allocated since the previous commit under `fence`.
    fn commit_batch(&mut self, fence: u64) {
        if self.current_batch_bytes == 0 {
            return;
        }
        let bytes = std::mem::take(&mut self.current_batch_bytes);
        if let Some(back) = self.in_flight.back_mut()
            && back.fence == fence
        {
            back.bytes += bytes;
        } else {
            self.in_flight.push_back(RingLease { fence, bytes });
        }
    }

    fn oldest_fence(&self) -> Option<u64> {
        self.in_flight.front().map(|l| l.fence)
    }

    #[cfg(test)]
    fn used_bytes(&self) -> u64 {
        self.used
    }
}

/// Whether `layout` can be detiled by the compute shader.
pub(crate) fn supported(layout: &Layout) -> bool {
    layout.addressable()
}

/// Whether a tiled texture upload of `read` tiled bytes and `linear_total`
/// linear footprint bytes can use GPU compute detiling on a direct prelude list
/// or dedicated compute queue.
pub(crate) fn can_use(layout: &Layout, read: usize, linear_total: u64) -> bool {
    on()
        && supported(layout)
        && layout.desc.swizzle != Swizzle::LINEAR
        && read > 0
        && u32::try_from(read).is_ok()
        && u32::try_from(linear_total.next_multiple_of(4)).is_ok()
        && pass().is_some()
}

/// One subresource detile (or zero-fill) work item for [`record_detile_subresources`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct SubresourceJob {
    pub mip: u32,
    pub slice: u32,
    pub dst_offset: u64,
    pub row_pitch: u32,
    pub depth_pitch: u32,
    pub zero: bool,
}

/// One texture upload item in a batched `gpu_detile` submission.
///
/// # Safety / State Invariant
/// For [`execute_compute_batch`], `dst_texture` MUST enter in
/// `D3D12_RESOURCE_STATE_COMMON` (`D3D12_BARRIER_LAYOUT_COMMON`) and will be
/// returned to `D3D12_RESOURCE_STATE_COMMON` before `tex_fence` signals.
pub(crate) struct BatchItem<'a> {
    pub layout: &'a Layout,
    pub src_gpu_va: u64,
    pub src_len: u64,
    pub dst_texture: &'a ID3D12Resource,
    pub sub_offset: u32,
    pub footprints: &'a [D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    pub jobs: &'a [SubresourceJob],
    pub indices: &'a [usize],
    pub linear_bytes: u64,
    pub need_aliasing: bool,
}

/// Packs the per-axis address-bit scatter positions into three `u32` shift words
/// (4 bits per coordinate bit, up to 8 coordinate bits per axis) plus bit counts,
/// using `victory_formats::texture::Layout::swizzle_shifts` as the single source of truth.
fn swizzle_encoding(layout: &Layout) -> ([u32; 3], [u32; 3]) {
    layout.swizzle_shifts().unwrap_or(([0; 3], [0; 3]))
}

const FLAG_THICK: u32 = 1 << 0;
const FLAG_ZERO: u32 = 1 << 1;

const DETILE_PARAM_WORDS: u32 = 22;

#[repr(C)]
#[derive(Clone, Copy)]
struct DetileParams {
    width: u32,
    height: u32,
    depth: u32,
    element_bytes: u32,
    row_pitch: u32,
    depth_pitch: u32,
    dst_offset: u32,
    slice_size: u32,
    mip_offset: u32,
    first_slice: u32,
    mip_pitch: u32,
    block_w: u32,
    block_h: u32,
    block_d: u32,
    block_log2: u32,
    tail_x: u32,
    tail_y: u32,
    flags: u32,
    x_shifts: u32,
    y_shifts: u32,
    z_shifts: u32,
    nbits_packed: u32,
}

const _: () = assert!(std::mem::size_of::<DetileParams>() == DETILE_PARAM_WORDS as usize * 4);

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=22, b0), SRV(t0), UAV(u0)"

cbuffer DetileParams : register(b0) {
    uint width;
    uint height;
    uint depth;
    uint element_bytes;
    uint row_pitch;
    uint depth_pitch;
    uint dst_offset;
    uint slice_size;
    uint mip_offset;
    uint first_slice;
    uint mip_pitch;
    uint block_w;
    uint block_h;
    uint block_d;
    uint block_log2;
    uint tail_x;
    uint tail_y;
    uint flags;
    uint x_shifts;
    uint y_shifts;
    uint z_shifts;
    uint nbits_packed;
};

ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);

uint scatter_bits(uint coord, uint shifts, uint nbits) {
    uint acc = 0u;
    [unroll]
    for (uint i = 0u; i < 8u; ++i) {
        if (i < nbits) {
            uint pos = (shifts >> (i * 4u)) & 0xFu;
            acc |= ((coord >> i) & 1u) << pos;
        }
    }
    return acc;
}

uint tiled_col_addr(uint x, uint log2_bw, uint bw_mask, uint x_nbits) {
    uint tx = x + tail_x;
    uint bx = tx >> log2_bw;
    uint lx = tx & bw_mask;
    return (bx << block_log2) + scatter_bits(lx, x_shifts, x_nbits);
}

[RootSignature(RS)]
[numthreads(8, 8, 1)]
void detile(uint3 tid : SV_DispatchThreadID) {
    uint y = tid.y;
    uint z = tid.z;
    if (y >= height || z >= depth) {
        return;
    }

    uint elems_per_thread = (element_bytes == 1u) ? 4u : ((element_bytes == 2u) ? 2u : 1u);
    uint x0 = tid.x * elems_per_thread;
    if (x0 >= width) {
        return;
    }

    uint dst_addr = dst_offset + z * depth_pitch + y * row_pitch + x0 * element_bytes;
    if ((flags & 2u) != 0u) {
        if (element_bytes <= 4u) {
            dst.Store(dst_addr, 0u);
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, uint2(0u, 0u));
        } else {
            dst.Store4(dst_addr, uint4(0u, 0u, 0u, 0u));
        }
        return;
    }

    uint s = first_slice + z;
    uint inner_z = 0u;
    if ((flags & 1u) != 0u) {
        uint log2_bd = firstbithigh(block_d);
        inner_z = s & (block_d - 1u);
        s = s >> log2_bd;
    }
    uint base = s * slice_size + mip_offset;

    if (block_log2 == 0u) {
        uint row_base = base + y * mip_pitch * element_bytes;
        uint src_addr = row_base + x0 * element_bytes;
        if (element_bytes == 1u) {
            uint w = src.Load(src_addr);
            uint rem = width - x0;
            if (rem < 4u) {
                w &= (1u << (rem * 8u)) - 1u;
            }
            dst.Store(dst_addr, w);
        } else if (element_bytes == 2u) {
            uint w = src.Load(src_addr);
            if (x0 + 1u >= width) {
                w &= 0xFFFFu;
            }
            dst.Store(dst_addr, w);
        } else if (element_bytes == 4u) {
            dst.Store(dst_addr, src.Load(src_addr));
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, src.Load2(src_addr));
        } else {
            dst.Store4(dst_addr, src.Load4(src_addr));
        }
        return;
    }

    uint x_nbits = nbits_packed & 0xFFu;
    uint y_nbits = (nbits_packed >> 8u) & 0xFFu;
    uint z_nbits = (nbits_packed >> 16u) & 0xFFu;

    uint log2_bw = firstbithigh(block_w);
    uint log2_bh = firstbithigh(block_h);
    uint bw_mask = block_w - 1u;

    uint ty = y + tail_y;
    uint by = ty >> log2_bh;
    uint ly = ty & (block_h - 1u);
    uint blocks_x = mip_pitch >> log2_bw;
    uint row_base = base
        + ((by * blocks_x) << block_log2)
        + (scatter_bits(ly, y_shifts, y_nbits) | scatter_bits(inner_z, z_shifts, z_nbits));

    if (element_bytes == 1u) {
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 4u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint sa = row_base + tiled_col_addr(x, log2_bw, bw_mask, x_nbits);
                uint w = src.Load(sa & ~3u);
                uint b = (w >> ((sa & 3u) * 8u)) & 0xFFu;
                packed |= b << (k * 8u);
            }
        }
        dst.Store(dst_addr, packed);
    } else if (element_bytes == 2u) {
        uint packed = 0u;
        [unroll]
        for (uint k = 0u; k < 2u; ++k) {
            uint x = x0 + k;
            if (x < width) {
                uint sa = row_base + tiled_col_addr(x, log2_bw, bw_mask, x_nbits);
                uint w = src.Load(sa & ~3u);
                uint h = (w >> ((sa & 2u) * 8u)) & 0xFFFFu;
                packed |= h << (k * 16u);
            }
        }
        dst.Store(dst_addr, packed);
    } else {
        uint sa = row_base + tiled_col_addr(x0, log2_bw, bw_mask, x_nbits);
        if (element_bytes == 4u) {
            dst.Store(dst_addr, src.Load(sa));
        } else if (element_bytes == 8u) {
            dst.Store2(dst_addr, src.Load2(sa));
        } else {
            dst.Store4(dst_addr, src.Load4(sa));
        }
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Dummy read-only buffer bound at `t0` when all jobs in a batch are `zero = true`
    /// and the caller passes `src_gpu_va == 0`.
    dummy_src: ID3D12Resource,
}

// SAFETY: D3D12 root signatures, pipelines and resources are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make_pass(host: &Host) -> windows::core::Result<Pass> {
    let code = crate::host::compile(SOURCE, s!("detile"), s!("cs_5_1"))?;
    let (dummy_src, _) = host
        .upload_buffer(256)
        .ok_or_else(windows::core::Error::empty)?;
    // SAFETY: root signature is embedded in the compiled shader blob; plain creation.
    unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        };
        let pso = host.device.CreateComputePipelineState(&desc)?;
        crate::host::name(&root, || "gpu detile root".into());
        crate::host::name(&pso, || "gpu detile pso".into());
        crate::host::name(&dummy_src, || "gpu detile dummy src".into());
        Ok(Pass {
            root,
            pso,
            dummy_src,
        })
    }
}

fn pass() -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        log_startup_once(host);
        make_pass(host)
            .inspect_err(|e| tracing::warn!(error = %e, "gpu detile compute pipeline unavailable"))
            .ok()
    })
    .as_ref()
}

/// Records compute detile dispatches for `jobs` onto `list`.
///
/// The caller owns resource state barriers on `dst_uav` (`COMMON -> UNORDERED_ACCESS`
/// before and `UNORDERED_ACCESS -> COPY_SOURCE` after). Returns `false` without
/// recording any commands if validation fails or any byte offset exceeds `u32::MAX`.
pub(crate) fn record_detile_subresources(
    list: &ID3D12GraphicsCommandList,
    layout: &Layout,
    src_gpu_va: u64,
    src_len: u64,
    dst_uav: &ID3D12Resource,
    dst_len: u64,
    jobs: &[SubresourceJob],
) -> bool {
    record_detile_subresources_inner(
        list, layout, src_gpu_va, src_len, dst_uav, 0, dst_len, jobs, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_detile_subresources_inner(
    list: &ID3D12GraphicsCommandList,
    layout: &Layout,
    src_gpu_va: u64,
    src_len: u64,
    dst_uav: &ID3D12Resource,
    dst_base_offset: u64,
    dst_len: u64,
    jobs: &[SubresourceJob],
    bind_pipeline_and_uav: bool,
) -> bool {
    if !supported(layout) {
        return false;
    }
    if jobs.is_empty() {
        return true;
    }
    let Some(block_log2) = layout.desc.swizzle.block_log2() else {
        return false;
    };
    if block_log2 != 0 && !layout.block.iter().all(|&b| b.is_power_of_two()) {
        return false;
    }
    let Some(pass) = pass() else {
        return false;
    };
    if !dst_base_offset.is_multiple_of(4) {
        return false;
    }
    // SAFETY: live D3D12 buffer resource.
    let (dst_uav_va, dst_uav_width) = unsafe { (dst_uav.GetGPUVirtualAddress(), dst_uav.GetDesc().Width) };
    if dst_uav_va == 0 || !dst_uav_va.is_multiple_of(4) {
        return false;
    }
    let needs_src = jobs.iter().any(|j| !j.zero);
    if needs_src && (src_gpu_va == 0 || !src_gpu_va.is_multiple_of(4)) {
        return false;
    }

    let (shifts, nbits) = swizzle_encoding(layout);
    let nbits_packed = nbits[0] | (nbits[1] << 8) | (nbits[2] << 16);
    let bpe = layout.desc.element_bytes;
    let elems_per_thread = if bpe < 4 { 4 / bpe } else { 1 };
    let base_flags = if layout.thick { FLAG_THICK } else { 0 };

    let mut dispatches = Vec::with_capacity(jobs.len());
    for job in jobs {
        let Some(m) = layout.mips.get(job.mip as usize) else {
            return false;
        };
        let first_slice = if layout.desc.dimension == Dimension::Tex3d {
            if job.slice != 0 {
                return false;
            }
            0
        } else {
            job.slice
        };
        if first_slice >= layout.slices {
            return false;
        }
        let row_bytes = u64::from(m.width) * u64::from(bpe);
        let row_bytes_aligned = row_bytes.next_multiple_of(4);
        if !job.dst_offset.is_multiple_of(4) || u64::from(job.row_pitch) < row_bytes {
            return false;
        }
        if m.height > 1
            && (!job.row_pitch.is_multiple_of(4) || u64::from(job.row_pitch) < row_bytes_aligned)
        {
            return false;
        }
        let Some(slice_exact) = (u64::from(m.height) - 1)
            .checked_mul(u64::from(job.row_pitch))
            .and_then(|o| o.checked_add(row_bytes))
        else {
            return false;
        };
        let Some(slice_aligned) = (u64::from(m.height) - 1)
            .checked_mul(u64::from(job.row_pitch))
            .and_then(|o| o.checked_add(row_bytes_aligned))
        else {
            return false;
        };
        if m.depth > 1
            && (!job.depth_pitch.is_multiple_of(4) || u64::from(job.depth_pitch) < slice_aligned)
        {
            return false;
        }
        let Some(depth_span) = (u64::from(m.depth) - 1).checked_mul(u64::from(job.depth_pitch))
        else {
            return false;
        };
        let Some(abs_offset) = dst_base_offset.checked_add(job.dst_offset) else {
            return false;
        };
        let Some(aligned_end) = abs_offset
            .checked_add(depth_span)
            .and_then(|o| o.checked_add(slice_aligned))
        else {
            return false;
        };
        let Some(rel_end) = job
            .dst_offset
            .checked_add(depth_span)
            .and_then(|o| o.checked_add(slice_exact))
        else {
            return false;
        };
        if rel_end > dst_len
            || aligned_end > dst_uav_width
            || u32::try_from(aligned_end).is_err()
        {
            return false;
        }
        let Ok(dst_offset) = u32::try_from(abs_offset) else {
            return false;
        };

        let (slice_size, mip_offset) = if job.zero {
            (0, 0)
        } else {
            let tiled_need = layout.bytes_from(job.mip);
            if src_len < tiled_need || u32::try_from(tiled_need).is_err() {
                return false;
            }
            let max_z = first_slice + m.depth - 1;
            let max_s = if layout.thick {
                max_z / layout.block[2]
            } else {
                max_z
            };
            let slice_size = if max_s > 0 {
                let Ok(v) = u32::try_from(layout.slice_size) else {
                    return false;
                };
                v
            } else {
                0
            };
            let Ok(mip_offset) = u32::try_from(m.offset) else {
                return false;
            };
            (slice_size, mip_offset)
        };

        let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
        let params = DetileParams {
            width: m.width,
            height: m.height,
            depth: m.depth,
            element_bytes: bpe,
            row_pitch: job.row_pitch,
            depth_pitch: job.depth_pitch,
            dst_offset,
            slice_size,
            mip_offset,
            first_slice,
            mip_pitch: m.pitch,
            block_w: layout.block[0],
            block_h: layout.block[1],
            block_d: layout.block[2],
            block_log2,
            tail_x,
            tail_y,
            flags: base_flags | if job.zero { FLAG_ZERO } else { 0 },
            x_shifts: shifts[0],
            y_shifts: shifts[1],
            z_shifts: shifts[2],
            nbits_packed,
        };
        let gx = m.width.div_ceil(elems_per_thread).div_ceil(8);
        let gy = m.height.div_ceil(8);
        let gz = m.depth;
        dispatches.push((params, gx, gy, gz));
    }

    let bound_src_va = if src_gpu_va != 0 {
        src_gpu_va
    } else {
        // SAFETY: live static upload buffer in `Pass`.
        unsafe { pass.dummy_src.GetGPUVirtualAddress() }
    };

    // SAFETY: open direct/compute command list; root signature, PSO and buffer VAs are valid and
    // aligned; all bounds and 32-bit offsets have been validated above.
    unsafe {
        if bind_pipeline_and_uav {
            list.SetComputeRootSignature(&pass.root);
            list.SetPipelineState(&pass.pso);
            list.SetComputeRootUnorderedAccessView(2, dst_uav_va);
        }
        list.SetComputeRootShaderResourceView(1, bound_src_va);
        for &(params, gx, gy, gz) in &dispatches {
            list.SetComputeRoot32BitConstants(
                0,
                DETILE_PARAM_WORDS,
                (&raw const params).cast(),
                0,
            );
            list.Dispatch(gx, gy, gz);
        }
    }
    true
}

/// Legacy reusable device-local UAV scratch buffer retained for unit tests and
/// direct-list callers that manage their own scratch lifecycle.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn shared_scratch(host: &Host, size: u64) -> Option<ID3D12Resource> {
    struct Scratch(Option<(u64, ID3D12Resource)>);
    // SAFETY: D3D12 resources are free-threaded.
    unsafe impl Send for Scratch {}
    static SCRATCH: Mutex<Scratch> = Mutex::new(Scratch(None));
    const MAX_RETAINED: u64 = 32 << 20;

    let min_size = size.max(1 << 16);
    let class = min_size.checked_next_power_of_two().unwrap_or(min_size);
    let mut slot = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cap, buffer)) = slot.0.as_ref()
        && *cap >= class
    {
        return Some(buffer.clone());
    }
    let buffer = host.internal_uav_buffer(class)?;
    crate::host::name(&buffer, || format!("gpu detile scratch {class:#x}"));
    if class <= MAX_RETAINED {
        if let Some((_, old)) = slot.0.replace((class, buffer.clone())) {
            crate::host::retire(old);
        }
    }
    Some(buffer)
}

/// Direct-prelude ring scratch buffer tracked against `crate::host::submissions()`.
struct PreludeRing {
    buffer: ID3D12Resource,
    alloc: RingAllocator,
    /// Pointer of the last command list that transitioned `buffer` to `COPY_SOURCE`
    /// under Enhanced Barriers so consecutive textures on the same list emit the
    /// minimal `COPY -> COMPUTE` transition while the first texture emits none.
    last_list_raw: usize,
    /// Fence-tracked reusable overflow scratch buffers `(fence, capacity, buf)` so
    /// burst uploads past the ring never thrash `CreateCommittedResource`.
    overflow_pool: Vec<(u64, u64, ID3D12Resource)>,
    overflow_bytes: u64,
}

// SAFETY: `ID3D12Resource` is free-threaded; state is guarded by `PRELUDE_RING`.
unsafe impl Send for PreludeRing {}

static PRELUDE_RING: Mutex<Option<PreludeRing>> = Mutex::new(None);

const OVERFLOW_POOL_MAX_BYTES: u64 = 64 << 20;

fn alloc_prelude_slice(
    host: &Host,
    list_raw: usize,
    size: u64,
) -> Option<(ID3D12Resource, u64, bool, bool)> {
    let mut guard = PRELUDE_RING.lock().unwrap_or_else(|e| e.into_inner());
    let ring = match guard.as_mut() {
        Some(r) => r,
        None => {
            let cap = ring_bytes();
            let buffer = host.internal_uav_buffer(cap)?;
            crate::host::name(&buffer, || format!("gpu detile prelude ring {} MiB", cap >> 20));
            log_startup_once(host);
            guard.insert(PreludeRing {
                buffer,
                alloc: RingAllocator::new(cap),
                last_list_raw: 0,
                overflow_pool: Vec::new(),
                overflow_bytes: 0,
            })
        }
    };
    let (started, finished) = crate::host::submissions();
    ring.alloc.reclaim(finished);
    let slice_fence = started.saturating_add(1);
    match ring.alloc.alloc(size, RING_ALIGN) {
        RingAlloc::Slice { offset, wrapped, .. } => {
            if wrapped {
                TELEMETRY[5].fetch_add(1, Ordering::Relaxed);
            }
            ring.alloc.commit_batch(slice_fence);
            let same_list = ring.last_list_raw == list_raw;
            ring.last_list_raw = list_raw;
            Some((ring.buffer.clone(), offset, false, same_list))
        }
        RingAlloc::Full | RingAlloc::Oversized => {
            TELEMETRY[7].fetch_add(1, Ordering::Relaxed);
            let alloc_size = size.max(4).next_multiple_of(RING_ALIGN);
            // Reuse the smallest completed overflow buffer that fits.
            let best_idx = ring
                .overflow_pool
                .iter()
                .enumerate()
                .filter(|(_, (fence, cap, _))| *fence <= finished && *cap >= alloc_size)
                .min_by_key(|(_, (_, cap, _))| *cap)
                .map(|(i, _)| i);
            if let Some(idx) = best_idx {
                ring.overflow_pool[idx].0 = slice_fence;
                return Some((ring.overflow_pool[idx].2.clone(), 0, false, false));
            }
            let class = alloc_size.max(64 << 10).checked_next_power_of_two().unwrap_or(alloc_size);
            let buf = host.internal_uav_buffer(class)?;
            crate::host::name(&buf, || format!("gpu detile overflow {class:#x}"));
            if class <= OVERFLOW_POOL_MAX_BYTES {
                // Evict completed buffers if needed to stay within the pool cap.
                while ring.overflow_bytes + class > OVERFLOW_POOL_MAX_BYTES {
                    let Some(evict) = ring
                        .overflow_pool
                        .iter()
                        .position(|(fence, _, _)| *fence <= finished)
                    else {
                        break;
                    };
                    let (_, cap, old) = ring.overflow_pool.swap_remove(evict);
                    ring.overflow_bytes = ring.overflow_bytes.saturating_sub(cap);
                    crate::host::retire(old);
                }
                if ring.overflow_bytes + class <= OVERFLOW_POOL_MAX_BYTES {
                    ring.overflow_bytes += class;
                    ring.overflow_pool.push((slice_fence, class, buf.clone()));
                    return Some((buf, 0, false, false));
                }
            }
            Some((buf, 0, true, false))
        }
    }
}

#[inline]
fn enhanced_buf_barrier(
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
        // SAFETY: borrowed COM pointer; never released by the barrier struct.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Offset: 0,
        Size: u64::MAX,
    }
}

#[inline]
fn emit_enhanced_buf_barriers(list7: &ID3D12GraphicsCommandList7, barriers: &[D3D12_BUFFER_BARRIER]) {
    if barriers.is_empty() {
        return;
    }
    let group = [D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_BUFFER,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pBufferBarriers: barriers.as_ptr(),
        },
    }];
    // SAFETY: open command list and live buffers.
    unsafe { list7.Barrier(&group) };
}

#[inline]
fn aliasing_barrier(texture: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                // SAFETY: borrowed pointer; never released by the barrier struct.
                pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
            }),
        },
    }
}

fn buffer_barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed COM pointer; never released by the barrier struct.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    // SAFETY: open command list and live resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Records a single ring-buffered GPU-detiled texture upload onto a direct
/// prelude command list (`texture` is already in `D3D12_RESOURCE_STATE_COPY_DEST`).
///
/// Uses Enhanced `D3D12_BUFFER_BARRIER` when supported (eliminating legacy
/// `COMMON -> UAV` and `COPY_SOURCE -> COMMON` round-trips) while preserving
/// the pooled-texture `ALIASING` barrier on first upload.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_prelude_detile(
    host: &Host,
    cmd_list: &ID3D12GraphicsCommandList,
    layout: &Layout,
    src_gpu_va: u64,
    src_len: u64,
    texture: &ID3D12Resource,
    sub_offset: u32,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    jobs: &[SubresourceJob],
    indices: &[usize],
    linear_bytes: u64,
    need_aliasing: bool,
) -> bool {
    if jobs.is_empty() {
        return true;
    }
    let scratch_len = linear_bytes.next_multiple_of(4).max(4);
    let list_raw = cmd_list.as_raw() as usize;
    let Some((scratch, ring_offset, transient, same_list)) =
        alloc_prelude_slice(host, list_raw, scratch_len)
    else {
        return false;
    };

    let list7 = host_enhanced_barriers(host)
        .then(|| cmd_list.cast::<ID3D12GraphicsCommandList7>().ok())
        .flatten();

    if let Some(l7) = &list7 {
        if same_list && !transient {
            emit_enhanced_buf_barriers(
                l7,
                &[enhanced_buf_barrier(
                    &scratch,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                )],
            );
        }
    } else {
        buffer_barrier(
            cmd_list,
            &scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
    }

    let ok = record_detile_subresources_inner(
        cmd_list,
        layout,
        src_gpu_va,
        src_len,
        &scratch,
        ring_offset,
        linear_bytes,
        jobs,
        true,
    );
    if !ok {
        if list7.is_none() {
            buffer_barrier(
                cmd_list,
                &scratch,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
        if transient {
            crate::host::retire(scratch);
        }
        return false;
    }

    // SAFETY: open prelude direct list; transition ring UAV -> COPY_SOURCE, emit
    // first-upload aliasing barrier when pooled, copy subresources, then restore
    // COMMON under legacy barriers.
    unsafe {
        if let Some(l7) = &list7 {
            if need_aliasing {
                cmd_list.ResourceBarrier(&[aliasing_barrier(texture)]);
            }
            emit_enhanced_buf_barriers(
                l7,
                &[enhanced_buf_barrier(
                    &scratch,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                )],
            );
        } else if need_aliasing {
            cmd_list.ResourceBarrier(&[
                aliasing_barrier(texture),
                crate::host::transition(
                    &scratch,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ),
            ]);
        } else {
            buffer_barrier(
                cmd_list,
                &scratch,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
        }

        for &i in indices {
            let footprint = &footprints[i];
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: sub_offset + i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&scratch),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: ring_offset + footprint.Offset,
                        ..*footprint
                    },
                },
            };
            cmd_list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }

        if list7.is_none() {
            buffer_barrier(
                cmd_list,
                &scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }

    if transient {
        crate::host::retire(scratch);
    }
    record_telemetry(1, jobs.len() as u64, linear_bytes, false);
    true
}

struct ComputeSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    list7: Option<ID3D12GraphicsCommandList7>,
    fence_val: u64,
    /// Transient oversized scratch buffers held alive until `fence_val` lands.
    transient_buffers: Vec<ID3D12Resource>,
}

struct ComputeDetiler {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: HANDLE,
    next_fence: u64,
    slots: Vec<ComputeSlot>,
    next_slot: usize,
    ring_buffer: ID3D12Resource,
    ring: RingAllocator,
    enhanced: bool,
}

// SAFETY: D3D12 objects are free-threaded; recording & ring state are serialized
// behind `COMPUTE_DETILER`'s mutex.
unsafe impl Send for ComputeDetiler {}

impl Drop for ComputeDetiler {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: owned Win32 event created by `CreateEventW`.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

impl ComputeDetiler {
    fn new(host: &Host) -> Option<Self> {
        let enhanced = host_enhanced_barriers(host);
        let cap = ring_bytes();
        let ring_buffer = host.internal_uav_buffer(cap)?;
        crate::host::name(&ring_buffer, || format!("gpu detile compute ring {} MiB", cap >> 20));
        // SAFETY: plain D3D12 compute queue, allocators, lists, fence and event creation.
        unsafe {
            let queue: ID3D12CommandQueue = host
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    ..Default::default()
                })
                .ok()?;
            crate::host::name(&queue, || "gpu detile compute queue".into());
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            crate::host::name(&fence, || "gpu detile tex_fence".into());
            let event = CreateEventW(None, false, false, None).ok()?;
            let mut slots = Vec::with_capacity(COMPUTE_SLOTS);
            for idx in 0..COMPUTE_SLOTS {
                let allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = host
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)
                    .ok()?;
                list.Close().ok()?;
                let list7 = enhanced.then(|| list.cast::<ID3D12GraphicsCommandList7>().ok()).flatten();
                crate::host::name(&list, || format!("gpu detile compute list {idx}"));
                slots.push(ComputeSlot {
                    allocator,
                    list,
                    list7,
                    fence_val: 0,
                    transient_buffers: Vec::new(),
                });
            }
            log_startup_once(host);
            Some(Self {
                queue,
                fence,
                event,
                next_fence: 0,
                slots,
                next_slot: 0,
                ring_buffer,
                ring: RingAllocator::new(cap),
                enhanced,
            })
        }
    }

    fn wait_fence(&self, target: u64) -> bool {
        if target == 0 {
            return true;
        }
        // SAFETY: live fence and event owned by `self`.
        unsafe {
            if self.fence.GetCompletedValue() >= target {
                return true;
            }
            if self.fence.SetEventOnCompletion(target, self.event).is_err() {
                return false;
            }
            WaitForSingleObject(self.event, INFINITE);
            true
        }
    }

    fn reclaim_completed(&mut self) {
        // SAFETY: live fence.
        let completed = unsafe { self.fence.GetCompletedValue() };
        self.ring.reclaim(completed);
        for slot in &mut self.slots {
            if slot.fence_val <= completed && !slot.transient_buffers.is_empty() {
                slot.transient_buffers.clear();
            }
        }
    }

    /// Executes one chunk of items on the dedicated `COMPUTE` queue:
    /// **1 compute dispatch batch -> 1 scoped buffer barrier -> all `CopyTextureRegion`s**
    /// with every destination texture entering and exiting in `COMMON`.
    fn submit_chunk(
        &mut self,
        pass: &Pass,
        chunk: &[(&BatchItem<'_>, Option<u64>, Option<ID3D12Resource>)],
    ) -> Option<u64> {
        if chunk.is_empty() {
            return Some(self.next_fence);
        }
        let slot_idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let prev_fence = self.slots[slot_idx].fence_val;
        if prev_fence > 0 {
            self.wait_fence(prev_fence);
            self.reclaim_completed();
        }

        let slot = &mut self.slots[slot_idx];
        slot.transient_buffers.clear();
        let mut uses_ring = false;
        let mut transients: Vec<ID3D12Resource> = Vec::new();
        for &(_, ring_off, ref transient) in chunk {
            if ring_off.is_some() {
                uses_ring = true;
            }
            if let Some(buf) = transient {
                transients.push(buf.clone());
            }
        }

        // SAFETY: slot allocator/list are idle after `wait_fence(prev_fence)`.
        unsafe {
            slot.allocator.Reset().ok()?;
            slot.list.Reset(&slot.allocator, None).ok()?;

            let l = &slot.list;
            let use_enhanced = self.enhanced && slot.list7.is_some();

            // Pre-dispatch transition (legacy only; Enhanced mode needs none on fresh list).
            if !use_enhanced {
                let mut pre = Vec::with_capacity(1 + transients.len());
                if uses_ring {
                    pre.push(crate::host::transition(
                        &self.ring_buffer,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ));
                }
                for buf in &transients {
                    pre.push(crate::host::transition(
                        buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ));
                }
                if !pre.is_empty() {
                    l.ResourceBarrier(&pre);
                }
            }

            // Phase 1: ALL compute dispatches across every texture in the chunk.
            l.SetComputeRootSignature(&pass.root);
            l.SetPipelineState(&pass.pso);
            let ring_va = self.ring_buffer.GetGPUVirtualAddress();
            let mut bound_uav_va = 0u64;
            if uses_ring {
                l.SetComputeRootUnorderedAccessView(2, ring_va);
                bound_uav_va = ring_va;
            }

            let mut total_dispatches = 0u64;
            let mut total_bytes = 0u64;
            let mut ok_flags = vec![false; chunk.len()];
            for (idx, &(item, ring_off, ref transient)) in chunk.iter().enumerate() {
                let (dst_buf, base_off) = if let Some(off) = ring_off {
                    (&self.ring_buffer, off)
                } else if let Some(buf) = transient {
                    (buf, 0)
                } else {
                    continue;
                };
                let dst_va = dst_buf.GetGPUVirtualAddress();
                if dst_va != bound_uav_va {
                    l.SetComputeRootUnorderedAccessView(2, dst_va);
                    bound_uav_va = dst_va;
                }
                if record_detile_subresources_inner(
                    l,
                    item.layout,
                    item.src_gpu_va,
                    item.src_len,
                    dst_buf,
                    base_off,
                    item.linear_bytes,
                    item.jobs,
                    false,
                ) {
                    ok_flags[idx] = true;
                    total_dispatches += item.jobs.len() as u64;
                    total_bytes += item.linear_bytes;
                }
            }

            // Phase 2: 1 scoped buffer barrier (UAV -> COPY_SOURCE) + batched
            // destination texture transitions (COMMON -> COPY_DEST + ALIASING).
            if use_enhanced {
                let mut buf_barriers = Vec::with_capacity(1 + transients.len());
                if uses_ring {
                    buf_barriers.push(enhanced_buf_barrier(
                        &self.ring_buffer,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    ));
                }
                for buf in &transients {
                    buf_barriers.push(enhanced_buf_barrier(
                        buf,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    ));
                }
                if let Some(l7) = &slot.list7 {
                    emit_enhanced_buf_barriers(l7, &buf_barriers);
                }
            }

            let mut tex_pre = Vec::with_capacity(chunk.len() * 2 + 1 + transients.len());
            if !use_enhanced {
                if uses_ring {
                    tex_pre.push(crate::host::transition(
                        &self.ring_buffer,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ));
                }
                for buf in &transients {
                    tex_pre.push(crate::host::transition(
                        buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ));
                }
            }
            for (idx, &(item, _, _)) in chunk.iter().enumerate() {
                if !ok_flags[idx] {
                    continue;
                }
                if item.need_aliasing {
                    tex_pre.push(aliasing_barrier(item.dst_texture));
                }
                tex_pre.push(crate::host::transition(
                    item.dst_texture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
            if !tex_pre.is_empty() {
                l.ResourceBarrier(&tex_pre);
            }

            // Phase 3: ALL CopyTextureRegions across every texture in the chunk.
            for (idx, &(item, ring_off, ref transient)) in chunk.iter().enumerate() {
                if !ok_flags[idx] {
                    continue;
                }
                let (src_buf, base_off) = if let Some(off) = ring_off {
                    (&self.ring_buffer, off)
                } else if let Some(buf) = transient {
                    (buf, 0)
                } else {
                    continue;
                };
                for &i in item.indices {
                    let footprint = &item.footprints[i];
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(item.dst_texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: item.sub_offset + i as u32,
                        },
                    };
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(src_buf),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                                Offset: base_off + footprint.Offset,
                                ..*footprint
                            },
                        },
                    };
                    l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
            }

            // Phase 4: Restore EVERY destination texture to COMMON (mandatory
            // cross-queue D3D12 invariant) and scratch buffers to COMMON in legacy mode.
            let mut post = Vec::with_capacity(chunk.len() + 1 + transients.len());
            for (idx, &(item, _, _)) in chunk.iter().enumerate() {
                if !ok_flags[idx] {
                    continue;
                }
                post.push(crate::host::transition(
                    item.dst_texture,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
            if !use_enhanced {
                if uses_ring {
                    post.push(crate::host::transition(
                        &self.ring_buffer,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
                for buf in &transients {
                    post.push(crate::host::transition(
                        buf,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                }
            }
            if !post.is_empty() {
                l.ResourceBarrier(&post);
            }

            l.Close().ok()?;
            let cmd: ID3D12CommandList = l.cast().ok()?;
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            self.next_fence += 1;
            let fence_val = self.next_fence;
            self.queue.Signal(&self.fence, fence_val).ok()?;
            slot.fence_val = fence_val;
            slot.transient_buffers = transients;
            self.ring.commit_batch(fence_val);

            let ok_textures = ok_flags.iter().filter(|&&b| b).count() as u64;
            if ok_textures > 0 {
                record_telemetry(ok_textures, total_dispatches, total_bytes, true);
            }
            Some(fence_val)
        }
    }
}

struct ComputeRuntime {
    /// Clone of `tex_fence` so `wait_on_queue` / `wait_cpu` can sync without
    /// locking the `ComputeDetiler` recording mutex.
    fence: ID3D12Fence,
    detiler: Mutex<ComputeDetiler>,
}

// SAFETY: `ID3D12Fence` is free-threaded; `ComputeDetiler` is `Send` behind `Mutex`.
unsafe impl Send for ComputeRuntime {}
// SAFETY: as above.
unsafe impl Sync for ComputeRuntime {}

fn compute_runtime(host: &Host) -> Option<&'static ComputeRuntime> {
    static RT: OnceLock<Option<ComputeRuntime>> = OnceLock::new();
    RT.get_or_init(|| {
        let detiler = ComputeDetiler::new(host)?;
        let fence = detiler.fence.clone();
        Some(ComputeRuntime {
            fence,
            detiler: Mutex::new(detiler),
        })
    })
    .as_ref()
}

/// Executes a batch of tiled texture detiles on the dedicated `COMPUTE` queue
/// using the fence-tracked ring scratch buffer.
///
/// Every item's `dst_texture` MUST be in `D3D12_RESOURCE_STATE_COMMON` upon entry
/// and is returned to `D3D12_RESOURCE_STATE_COMMON` before the returned `tex_fence`
/// value is signaled. When `wait_cpu` is `false`, the caller MUST synchronize
/// via [`wait_on_queue`] or [`wait_compute_fence`] before any shader samples the
/// destination textures (and keep source upload buffers alive until then).
pub(crate) fn execute_compute_batch(
    host: &Host,
    items: &[BatchItem<'_>],
    wait_cpu: bool,
) -> Option<u64> {
    if items.is_empty() {
        return Some(0);
    }
    let pass = pass()?;
    let rt = compute_runtime(host)?;
    let mut detiler = rt.detiler.lock().unwrap_or_else(|e| e.into_inner());
    detiler.reclaim_completed();

    let mut chunk: Vec<(&BatchItem<'_>, Option<u64>, Option<ID3D12Resource>)> =
        Vec::with_capacity(items.len());
    let mut last_fence = 0u64;

    for item in items {
        if item.jobs.is_empty() {
            continue;
        }
        let need = item.linear_bytes.next_multiple_of(4).max(4);
        loop {
            match detiler.ring.alloc(need, RING_ALIGN) {
                RingAlloc::Slice { offset, wrapped, .. } => {
                    if wrapped {
                        TELEMETRY[5].fetch_add(1, Ordering::Relaxed);
                    }
                    chunk.push((item, Some(offset), None));
                    break;
                }
                RingAlloc::Oversized => {
                    TELEMETRY[7].fetch_add(1, Ordering::Relaxed);
                    let alloc_size = need.next_multiple_of(RING_ALIGN);
                    let buf = host.internal_uav_buffer(alloc_size)?;
                    crate::host::name(&buf, || format!("gpu detile compute oversized {alloc_size:#x}"));
                    chunk.push((item, None, Some(buf)));
                    break;
                }
                RingAlloc::Full => {
                    // Flush any accumulated items first so their fence can advance,
                    // then wait on the oldest in-flight ring lease.
                    if !chunk.is_empty() {
                        last_fence = detiler.submit_chunk(pass, &chunk)?;
                        chunk.clear();
                    }
                    detiler.reclaim_completed();
                    if let Some(oldest) = detiler.ring.oldest_fence() {
                        TELEMETRY[6].fetch_add(1, Ordering::Relaxed);
                        if !detiler.wait_fence(oldest) {
                            return None;
                        }
                        detiler.reclaim_completed();
                    } else {
                        // Should never happen because Oversized is handled above.
                        return None;
                    }
                }
            }
        }
    }

    if !chunk.is_empty() {
        last_fence = detiler.submit_chunk(pass, &chunk)?;
    }
    if wait_cpu && last_fence > 0 && !detiler.wait_fence(last_fence) {
        return None;
    }
    Some(last_fence)
}

/// Inserts a GPU-side `queue.Wait(&tex_fence, fence_val)` so `queue` (e.g.,
/// `Host::queue`) cannot sample textures uploaded by [`execute_compute_batch`]
/// until the compute queue has completed and flushed L2.
#[allow(dead_code)]
pub(crate) fn wait_on_queue(host: &Host, queue: &ID3D12CommandQueue, fence_val: u64) -> bool {
    if fence_val == 0 {
        return true;
    }
    let Some(rt) = compute_runtime(host) else {
        return false;
    };
    // SAFETY: live queue and `tex_fence`.
    unsafe { queue.Wait(&rt.fence, fence_val).is_ok() }
}

/// Blocks the calling CPU thread until `tex_fence` reaches `fence_val`.
pub(crate) fn wait_compute_fence(host: &Host, fence_val: u64) -> bool {
    if fence_val == 0 {
        return true;
    }
    let Some(rt) = compute_runtime(host) else {
        return false;
    };
    // SAFETY: live fence; passing null event makes `SetEventOnCompletion` block synchronously.
    unsafe {
        if rt.fence.GetCompletedValue() >= fence_val {
            return true;
        }
        rt.fence
            .SetEventOnCompletion(fence_val, HANDLE::default())
            .is_ok()
    }
}

/// Records a batch of tiled texture detiles onto a caller-supplied command list
/// (`1 compute dispatch batch -> 1 scoped buffer barrier -> all CopyTextureRegions`).
/// Used when `VICTORY_GPU_DETILE_COMPUTE=0` or in direct-list batching tests.
/// Every destination texture in `items` must already be in `D3D12_RESOURCE_STATE_COPY_DEST`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn record_batch_on_list(
    host: &Host,
    cmd_list: &ID3D12GraphicsCommandList,
    items: &[BatchItem<'_>],
) -> bool {
    if items.is_empty() {
        return true;
    }
    let Some(pass) = pass() else {
        return false;
    };
    let mut total_bytes = 0u64;
    let mut offsets = Vec::with_capacity(items.len());
    for item in items {
        let aligned_off = total_bytes.next_multiple_of(RING_ALIGN);
        offsets.push(aligned_off);
        total_bytes = aligned_off + item.linear_bytes.next_multiple_of(4).max(4);
    }
    let list_raw = cmd_list.as_raw() as usize;
    let Some((scratch, ring_offset, transient, same_list)) =
        alloc_prelude_slice(host, list_raw, total_bytes)
    else {
        return false;
    };
    let list7 = host_enhanced_barriers(host)
        .then(|| cmd_list.cast::<ID3D12GraphicsCommandList7>().ok())
        .flatten();

    if let Some(l7) = &list7 {
        if same_list && !transient {
            emit_enhanced_buf_barriers(
                l7,
                &[enhanced_buf_barrier(
                    &scratch,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                )],
            );
        }
    } else {
        buffer_barrier(
            cmd_list,
            &scratch,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
    }

    // SAFETY: open command list; root signature, PSO and ring buffer live through execution.
    unsafe {
        cmd_list.SetComputeRootSignature(&pass.root);
        cmd_list.SetPipelineState(&pass.pso);
        cmd_list.SetComputeRootUnorderedAccessView(2, scratch.GetGPUVirtualAddress());
    }

    let mut ok_flags = vec![false; items.len()];
    let mut total_dispatches = 0u64;
    for (idx, item) in items.iter().enumerate() {
        if item.jobs.is_empty() {
            continue;
        }
        let base_off = ring_offset + offsets[idx];
        if record_detile_subresources_inner(
            cmd_list,
            item.layout,
            item.src_gpu_va,
            item.src_len,
            &scratch,
            base_off,
            item.linear_bytes,
            item.jobs,
            false,
        ) {
            ok_flags[idx] = true;
            total_dispatches += item.jobs.len() as u64;
        }
    }

    // 1 scoped buffer barrier for ALL textures in the batch + any aliasing barriers.
    // SAFETY: open list and live resources.
    unsafe {
        if let Some(l7) = &list7 {
            let mut aliases = Vec::new();
            for (idx, item) in items.iter().enumerate() {
                if ok_flags[idx] && item.need_aliasing {
                    aliases.push(aliasing_barrier(item.dst_texture));
                }
            }
            if !aliases.is_empty() {
                cmd_list.ResourceBarrier(&aliases);
            }
            emit_enhanced_buf_barriers(
                l7,
                &[enhanced_buf_barrier(
                    &scratch,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                )],
            );
        } else {
            let mut barriers = Vec::with_capacity(1 + items.len());
            barriers.push(crate::host::transition(
                &scratch,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ));
            for (idx, item) in items.iter().enumerate() {
                if ok_flags[idx] && item.need_aliasing {
                    barriers.push(aliasing_barrier(item.dst_texture));
                }
            }
            cmd_list.ResourceBarrier(&barriers);
        }

        for (idx, item) in items.iter().enumerate() {
            if !ok_flags[idx] {
                continue;
            }
            let base_off = ring_offset + offsets[idx];
            for &i in item.indices {
                let footprint = &item.footprints[i];
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(item.dst_texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: item.sub_offset + i as u32,
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&scratch),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: base_off + footprint.Offset,
                            ..*footprint
                        },
                    },
                };
                cmd_list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
        }

        if list7.is_none() {
            buffer_barrier(
                cmd_list,
                &scratch,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );
        }
    }

    if transient {
        crate::host::retire(scratch);
    }
    let ok_count = ok_flags.iter().filter(|&&b| b).count() as u64;
    if ok_count > 0 {
        record_telemetry(ok_count, total_dispatches, total_bytes, false);
    }
    ok_count > 0
}

/// Standalone synchronous GPU detile helper for testing and benchmarking.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn detile_to_vec(
    layout: &Layout,
    tiled: &[u8],
    mip: u32,
    slice: u32,
    row_pitch: usize,
    depth_pitch: usize,
    dst_len: usize,
) -> Option<Vec<u8>> {
    let host = host()?;
    let row_pitch = u32::try_from(row_pitch).ok()?;
    let depth_pitch = u32::try_from(depth_pitch).ok()?;

    let src_alloc = (tiled.len() as u64).next_multiple_of(4).max(4);
    let (src_buf, src_ptr) = host.upload_buffer(src_alloc)?;
    // SAFETY: `src_ptr` is mapped for `src_alloc >= tiled.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len());
        if (src_alloc as usize) > tiled.len() {
            std::ptr::write_bytes(src_ptr.add(tiled.len()), 0, src_alloc as usize - tiled.len());
        }
    }

    let uav_size = (dst_len as u64).next_multiple_of(4).max(4);
    let dst_uav = host.internal_uav_buffer(uav_size)?;
    let readback = host.read_back_buffer(uav_size)?;
    let list = host.new_list().ok()?;

    // SAFETY: live upload buffer.
    let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
    buffer_barrier(
        &list.list,
        &dst_uav,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let job = SubresourceJob {
        mip,
        slice,
        dst_offset: 0,
        row_pitch,
        depth_pitch,
        zero: false,
    };
    if !record_detile_subresources(
        &list.list,
        layout,
        src_va,
        tiled.len() as u64,
        &dst_uav,
        uav_size,
        &[job],
    ) {
        return None;
    }
    buffer_barrier(
        &list.list,
        &dst_uav,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: open list; `dst_uav` is in `COPY_SOURCE` and `readback` is in `COPY_DEST`.
    unsafe {
        list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, uav_size);
    }
    list.close();
    host.execute_sync_internal(&list);

    let mut data = std::ptr::null_mut();
    let read_range = D3D12_RANGE {
        Begin: 0,
        End: dst_len,
    };
    let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: GPU execution has completed; `readback` is >= `dst_len` bytes.
    let out = unsafe {
        readback.Map(0, Some(&read_range), Some(&mut data)).ok()?;
        let bytes = std::slice::from_raw_parts(data.cast::<u8>(), dst_len).to_vec();
        readback.Unmap(0, Some(&empty_write));
        bytes
    };
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::Desc;
    use windows::core::Interface;

    fn fill_pattern(len: usize, seed: u64) -> Vec<u8> {
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0u8; len];
        for chunk in out.chunks_mut(8) {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        out
    }

    fn max_mips(w: u32, h: u32, d: u32) -> u32 {
        32 - w.max(h).max(d).leading_zeros()
    }

    /// Runs a batched GPU detile of all specified `(mip, slice)` subresources into a single
    /// UAV scratch buffer and compares every subresource bit-for-bit against `Layout::detile`.
    fn verify_layout_subresources(layout: &Layout, seed: u64, subresources: &[(u32, u32)]) {
        let Some(host) = host() else {
            return;
        };
        let tiled = fill_pattern(layout.size as usize, seed);
        let bpe = layout.desc.element_bytes as usize;

        let mut jobs = Vec::with_capacity(subresources.len());
        let mut ranges = Vec::with_capacity(subresources.len());
        let mut cursor = 0u64;
        for &(mip, slice) in subresources {
            let m = &layout.mips[mip as usize];
            let row_pitch = (m.width as usize * bpe).next_multiple_of(256);
            let depth_pitch = row_pitch * m.height as usize;
            let bytes = if m.depth > 1 {
                depth_pitch * m.depth as usize
            } else {
                row_pitch * m.height as usize
            };
            cursor = cursor.next_multiple_of(512);
            jobs.push(SubresourceJob {
                mip,
                slice,
                dst_offset: cursor,
                row_pitch: row_pitch as u32,
                depth_pitch: depth_pitch as u32,
                zero: false,
            });
            ranges.push((cursor as usize, bytes, row_pitch, depth_pitch));
            cursor += bytes as u64;
        }
        let total = cursor.next_multiple_of(4).max(4);
        let (src_buf, src_ptr) = host.upload_buffer(layout.size.next_multiple_of(4).max(4)).unwrap();
        // SAFETY: mapped upload buffer covers `tiled.len()`.
        unsafe {
            std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len());
        }
        let dst_uav = shared_scratch(host, total).unwrap();
        let readback = host.read_back_buffer(total).unwrap();
        let list = host.new_list().unwrap();

        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        assert!(
            record_detile_subresources(
                &list.list,
                layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                total,
                &jobs,
            ),
            "record_detile_subresources failed for {:?}",
            layout.desc
        );
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: open list; `dst_uav` in `COPY_SOURCE`, `readback` in `COPY_DEST`.
        unsafe {
            list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, total);
        }
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        host.execute_sync_internal(&list);

        let mut data = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: total as usize,
        };
        let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: GPU execution finished.
        let gpu_all = unsafe {
            readback.Map(0, Some(&read_range), Some(&mut data)).unwrap();
            let v = std::slice::from_raw_parts(data.cast::<u8>(), total as usize).to_vec();
            readback.Unmap(0, Some(&empty_write));
            v
        };

        for (idx, &(mip, slice)) in subresources.iter().enumerate() {
            let (off, len, row_pitch, depth_pitch) = ranges[idx];
            let m = &layout.mips[mip as usize];
            let mut cpu = vec![0u8; len];
            layout
                .detile(&tiled, mip, slice, &mut cpu, row_pitch, depth_pitch)
                .unwrap();
            let gpu = &gpu_all[off..off + len];
            let row_bytes = m.width as usize * bpe;
            for z in 0..m.depth as usize {
                for y in 0..m.height as usize {
                    let row_start = z * depth_pitch + y * row_pitch;
                    assert_eq!(
                        &gpu[row_start..row_start + row_bytes],
                        &cpu[row_start..row_start + row_bytes],
                        "mismatch for desc={:?} mip={mip} slice={slice} z={z} y={y}",
                        layout.desc
                    );
                    // Verify trailing bytes within the final 4-byte dword are zeroed for bpe < 4.
                    let pad_end = row_bytes.next_multiple_of(4);
                    if pad_end > row_bytes {
                        assert!(
                            gpu[row_start + row_bytes..row_start + pad_end]
                                .iter()
                                .all(|&b| b == 0),
                            "non-zero trailing dword padding for desc={:?} mip={mip} slice={slice} z={z} y={y}",
                            layout.desc
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn swizzle_encoding_matches_block_dimensions() {
        let swizzles = [
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let (shifts, nbits) = swizzle_encoding(&layout);
                assert_eq!(1u32 << nbits[0], layout.block[0]);
                assert_eq!(1u32 << nbits[1], layout.block[1]);
                assert_eq!(1u32 << nbits[2], layout.block[2]);
                // Every scatter bit position must lie below block_log2 and be unique.
                let block_log2 = swizzle.block_log2().unwrap();
                let mut used = 0u32;
                for a in 0..3 {
                    for i in 0..nbits[a] {
                        let pos = (shifts[a] >> (i * 4)) & 0xF;
                        assert!(pos < block_log2);
                        assert_eq!(used & (1 << pos), 0);
                        used |= 1 << pos;
                    }
                }
            }
        }
        // Thick 3D modes
        for swizzle in [Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 64,
                    height: 64,
                    depth: 64,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(layout.thick);
                let (_, nbits) = swizzle_encoding(&layout);
                assert_eq!(1u32 << nbits[0], layout.block[0]);
                assert_eq!(1u32 << nbits[1], layout.block[1]);
                assert_eq!(1u32 << nbits[2], layout.block[2]);
            }
        }
    }

    #[test]
    fn gpu_detile_all_swizzles_and_element_sizes() {
        let Some(_) = host() else {
            return;
        };
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_256B_D,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 1u64;
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 137,
                    height: 83,
                    depth: 1,
                    mips: 1,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(supported(&layout));
                let tiled = fill_pattern(layout.size as usize, seed);
                seed += 1;
                let row_pitch = (layout.mips[0].width as usize * bpe as usize).next_multiple_of(4);
                let dst_len = row_pitch * layout.mips[0].height as usize;
                let mut cpu = vec![0u8; dst_len];
                layout
                    .detile(&tiled, 0, 0, &mut cpu, row_pitch, dst_len)
                    .unwrap();
                let gpu = detile_to_vec(&layout, &tiled, 0, 0, row_pitch, dst_len, dst_len)
                    .expect("detile_to_vec failed");
                assert_eq!(
                    gpu, cpu,
                    "bit-for-bit mismatch for swizzle={swizzle:?} bpe={bpe}"
                );
            }
            // Block-compressed formats (BC1/BC4 = 8B, BC3/BC5/BC6H/BC7 = 16B).
            for bpe in [8u32, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 260,
                    height: 140,
                    depth: 1,
                    mips: 4,
                    element_bytes: bpe,
                    block: 4,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_mip_chains_tails_and_npot() {
        let Some(_) = host() else {
            return;
        };
        let dims = [(73u32, 39u32), (109, 71), (255, 129), (5, 3), (1, 1)];
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_4KB_D,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 1000u64;
        for (w, h) in dims {
            let mips = max_mips(w, h, 1);
            for swizzle in swizzles {
                for bpe in [1u32, 2, 4, 8, 16] {
                    let layout = Layout::new(&Desc {
                        dimension: Dimension::Tex2d,
                        width: w,
                        height: h,
                        depth: 1,
                        mips,
                        element_bytes: bpe,
                        block: 1,
                        samples: 1,
                        swizzle,
                    })
                    .unwrap();
                    if matches!(
                        swizzle,
                        Swizzle::SW_4KB_S
                            | Swizzle::SW_4KB_D
                            | Swizzle::SW_64KB_S
                            | Swizzle::SW_64KB_D
                    ) && mips > 1
                    {
                        assert!(
                            layout.first_tail_mip < layout.desc.mips,
                            "expected mip tail for {w}x{h} {swizzle:?} bpe={bpe}"
                        );
                    }
                    let subs: Vec<(u32, u32)> = (0..mips).map(|m| (m, 0)).collect();
                    verify_layout_subresources(&layout, seed, &subs);
                    seed += 1;
                }
            }
        }

        // Also test 1D textures with odd widths and small bpe (partial final dwords).
        for swizzle in [Swizzle::LINEAR, Swizzle::SW_256B_S, Swizzle::SW_4KB_S] {
            for bpe in [1u32, 2, 4, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex1d,
                    width: 131,
                    height: 1,
                    depth: 3,
                    mips: 6,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let mut subs = Vec::new();
                for s in 0..3 {
                    for m in 0..6 {
                        subs.push((m, s));
                    }
                }
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_2d_arrays() {
        let Some(_) = host() else {
            return;
        };
        let swizzles = [
            Swizzle::LINEAR,
            Swizzle::SW_256B_S,
            Swizzle::SW_4KB_S,
            Swizzle::SW_64KB_S,
            Swizzle::SW_64KB_D,
        ];
        let mut seed = 5000u64;
        for swizzle in swizzles {
            for bpe in [1u32, 2, 4, 8, 16] {
                let slices = 5u32;
                let mips = 7u32;
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex2d,
                    width: 97,
                    height: 67,
                    depth: slices,
                    mips,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                let mut subs = Vec::new();
                for s in 0..slices {
                    for m in 0..mips {
                        subs.push((m, s));
                    }
                }
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_3d_thin_and_thick_volumes() {
        let Some(_) = host() else {
            return;
        };
        let mut seed = 8000u64;

        // Thin linear 3D.
        for bpe in [1u32, 2, 4, 8, 16] {
            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex3d,
                width: 37,
                height: 23,
                depth: 11,
                mips: 6,
                element_bytes: bpe,
                block: 1,
                samples: 1,
                swizzle: Swizzle::LINEAR,
            })
            .unwrap();
            assert!(!layout.thick);
            let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
            verify_layout_subresources(&layout, seed, &subs);
            seed += 1;
        }

        // Thick 4KB_S and 64KB_S 3D volumes, including deep volumes with multiple depth slabs
        // in both standard mips and tail mips.
        for swizzle in [Swizzle::SW_4KB_S, Swizzle::SW_64KB_S] {
            for bpe in [1u32, 2, 4, 8, 16] {
                let layout = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 41,
                    height: 29,
                    depth: 35,
                    mips: 6,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(layout.thick);
                assert!(layout.slices > 1, "expected multiple depth slabs");
                assert!(layout.first_tail_mip < layout.desc.mips, "expected 3D mip tail");
                let subs: Vec<(u32, u32)> = (0..layout.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&layout, seed, &subs);
                seed += 1;

                // Deep narrow volume where mip 0 itself starts in the tail and spans multiple slabs.
                let tail_deep = Layout::new(&Desc {
                    dimension: Dimension::Tex3d,
                    width: 3,
                    height: 3,
                    depth: 37,
                    mips: 4,
                    element_bytes: bpe,
                    block: 1,
                    samples: 1,
                    swizzle,
                })
                .unwrap();
                assert!(tail_deep.thick);
                assert_eq!(tail_deep.first_tail_mip, 0);
                assert!(tail_deep.slices > 1);
                let subs: Vec<(u32, u32)> = (0..tail_deep.desc.mips).map(|m| (m, 0)).collect();
                verify_layout_subresources(&tail_deep, seed, &subs);
                seed += 1;
            }
        }
    }

    #[test]
    fn gpu_detile_batched_prelude_layout_and_zero_subresources() {
        let Some(host) = host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 193,
            height: 115,
            depth: 2,
            mips: 8,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        assert!(layout.first_tail_mip < layout.desc.mips);

        // Verify shared_scratch reuses buffers within the same power-of-two class
        // and grows to a new buffer when exceeding the cached capacity.
        let s1 = shared_scratch(host, 4096).unwrap();
        // SAFETY: live D3D12 buffer.
        let cap1 = unsafe { s1.GetDesc().Width };
        let s2 = shared_scratch(host, cap1).unwrap();
        let s3 = shared_scratch(host, cap1 + 1).unwrap();
        assert_eq!(s1.as_raw(), s2.as_raw());
        assert_ne!(s1.as_raw(), s3.as_raw());

        let tiled = fill_pattern(layout.size as usize, 0xCAFE_BABE);
        let bpe = layout.desc.element_bytes as usize;
        let mut jobs = Vec::new();
        let mut meta = Vec::new();
        let mut cursor = 0u64;
        for slice in 0..2u32 {
            for mip in 0..layout.desc.mips {
                let m = &layout.mips[mip as usize];
                let row_pitch = (m.width as usize * bpe).next_multiple_of(256);
                let bytes = row_pitch * m.height as usize;
                cursor = cursor.next_multiple_of(512);
                // Mark mip 0 of slice 0 and mip 6 (tail) of slice 1 as zero-filled (unheld on_demand).
                let zero = (slice == 0 && mip == 0) || (slice == 1 && mip == 6);
                jobs.push(SubresourceJob {
                    mip,
                    slice,
                    dst_offset: cursor,
                    row_pitch: row_pitch as u32,
                    depth_pitch: bytes as u32,
                    zero,
                });
                meta.push((cursor as usize, bytes, row_pitch, mip, slice, zero));
                cursor += bytes as u64;
            }
        }
        let total = cursor;

        // Pre-fill dst_uav with 0xA5 so zero-fill subresources are proven to actively write 0x00.
        let (poison_buf, poison_ptr) = host.upload_buffer(total).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::write_bytes(poison_ptr, 0xA5, total as usize) };
        let (src_buf, src_ptr) = host.upload_buffer(layout.size).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };

        let dst_uav = shared_scratch(host, total).unwrap();
        let readback = host.read_back_buffer(total).unwrap();
        let list = host.new_list().unwrap();

        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: open list; poison_buf is upload (`GENERIC_READ`), dst_uav is `COPY_DEST`.
        unsafe { list.list.CopyBufferRegion(&dst_uav, 0, &poison_buf, 0, total) };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };
        assert!(record_detile_subresources(
            &list.list,
            &layout,
            src_va,
            tiled.len() as u64,
            &dst_uav,
            total,
            &jobs,
        ));
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: open list.
        unsafe { list.list.CopyBufferRegion(&readback, 0, &dst_uav, 0, total) };
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        host.execute_sync_internal(&list);

        let mut data = std::ptr::null_mut();
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: total as usize,
        };
        let empty_write = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: GPU execution finished.
        let gpu_all = unsafe {
            readback.Map(0, Some(&read_range), Some(&mut data)).unwrap();
            let v = std::slice::from_raw_parts(data.cast::<u8>(), total as usize).to_vec();
            readback.Unmap(0, Some(&empty_write));
            v
        };

        for (off, len, row_pitch, mip, slice, zero) in meta {
            let m = &layout.mips[mip as usize];
            let row_bytes = m.width as usize * bpe;
            let gpu = &gpu_all[off..off + len];
            if zero {
                for y in 0..m.height as usize {
                    let row = &gpu[y * row_pitch..y * row_pitch + row_bytes];
                    assert!(
                        row.iter().all(|&b| b == 0),
                        "expected zeroed row for mip={mip} slice={slice} y={y}"
                    );
                }
            } else {
                let mut cpu = vec![0u8; len];
                layout
                    .detile(&tiled, mip, slice, &mut cpu, row_pitch, len)
                    .unwrap();
                for y in 0..m.height as usize {
                    let s = y * row_pitch;
                    assert_eq!(
                        &gpu[s..s + row_bytes],
                        &cpu[s..s + row_bytes],
                        "mismatch for mip={mip} slice={slice} y={y}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_detile_validation_rejects_invalid_and_oversized() {
        let xor_layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S_X,
        })
        .unwrap();
        assert!(!supported(&xor_layout));

        let Some(host) = host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let dst_uav = shared_scratch(host, 65536).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: live resource.
        let va = unsafe { dst_uav.GetGPUVirtualAddress() };

        // >4 GiB dst_offset rejected cleanly.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size,
            &dst_uav,
            u64::MAX,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: (u32::MAX as u64) + 1,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // Unaligned dst_offset rejected.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 2,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // Too-small src_len rejected.
        assert!(!record_detile_subresources(
            &list.list,
            &layout,
            va,
            layout.size - 1,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 0,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: false,
            }],
        ));
        // All-zero batch with src_gpu_va == 0 and src_len == 0 succeeds cleanly.
        assert!(record_detile_subresources(
            &list.list,
            &layout,
            0,
            0,
            &dst_uav,
            65536,
            &[SubresourceJob {
                mip: 0,
                slice: 0,
                dst_offset: 0,
                row_pitch: 256,
                depth_pitch: 16384,
                zero: true,
            }],
        ));
    }

    #[test]
    #[ignore = "benchmark"]
    fn gpu_detile_bench() {
        let Some(host) = host() else {
            eprintln!("no host D3D12 device available");
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 2048,
            height: 2048,
            depth: 1,
            mips: 12,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let tiled = fill_pattern(layout.size as usize, 0x1234_5678);

        let mut jobs = Vec::new();
        let mut cursor = 0u64;
        for mip in 0..layout.desc.mips {
            let m = &layout.mips[mip as usize];
            let row_pitch = (m.width as usize * 16).next_multiple_of(256);
            let bytes = row_pitch * m.height as usize;
            cursor = cursor.next_multiple_of(512);
            jobs.push(SubresourceJob {
                mip,
                slice: 0,
                dst_offset: cursor,
                row_pitch: row_pitch as u32,
                depth_pitch: bytes as u32,
                zero: false,
            });
            cursor += bytes as u64;
        }
        let total_linear = cursor as usize;

        // Warmup CPU
        let mut cpu_buf = vec![0u8; total_linear];
        for job in &jobs {
            let off = job.dst_offset as usize;
            let len = job.depth_pitch as usize;
            layout
                .detile(
                    &tiled,
                    job.mip,
                    0,
                    &mut cpu_buf[off..off + len],
                    job.row_pitch as usize,
                    len,
                )
                .unwrap();
        }

        const ITERS: u32 = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            for job in &jobs {
                let off = job.dst_offset as usize;
                let len = job.depth_pitch as usize;
                layout
                    .detile(
                        &tiled,
                        job.mip,
                        0,
                        &mut cpu_buf[off..off + len],
                        job.row_pitch as usize,
                        len,
                    )
                    .unwrap();
            }
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        // Warmup GPU (pre-uploaded source + shared UAV scratch, matching prelude path).
        let (src_buf, src_ptr) = host.upload_buffer(layout.size).unwrap();
        // SAFETY: mapped upload buffer.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };
        let dst_uav = shared_scratch(host, cursor).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: live upload buffer.
        let src_va = unsafe { src_buf.GetGPUVirtualAddress() };

        let run_gpu_once = || {
            buffer_barrier(
                &list.list,
                &dst_uav,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            assert!(record_detile_subresources(
                &list.list,
                &layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                cursor,
                &jobs,
            ));
            buffer_barrier(
                &list.list,
                &dst_uav,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            );
            list.close();
            host.execute_sync_internal(&list);
            list.reset();
        };
        for _ in 0..5 {
            run_gpu_once();
        }

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_gpu_once();
        }
        let gpu_sync_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        // Batched GPU (all ITERS recorded into one command list, matching prelude batching).
        let t2 = std::time::Instant::now();
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        for _ in 0..ITERS {
            assert!(record_detile_subresources(
                &list.list,
                &layout,
                src_va,
                tiled.len() as u64,
                &dst_uav,
                cursor,
                &jobs,
            ));
        }
        buffer_barrier(
            &list.list,
            &dst_uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        list.close();
        let record_us = t2.elapsed().as_secs_f64() * 1e6 / f64::from(ITERS);
        host.execute_sync_internal(&list);
        list.reset();
        let gpu_batch_ms = t2.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        let mb = total_linear as f64 / (1024.0 * 1024.0);
        println!(
            "2048x2048 BC7 full 12-mip chain ({mb:.2} MiB linear, {ITERS} iters):\n  \
             CPU detile           = {cpu_ms:.3} ms ({:.2} GiB/s)\n  \
             GPU (1 submit/tex)   = {gpu_sync_ms:.3} ms ({:.2} GiB/s)\n  \
             GPU (batched submit) = {gpu_batch_ms:.3} ms ({:.2} GiB/s, CPU record = {record_us:.1} us/tex, speedup = {:.2}x)",
            (mb / 1024.0) / (cpu_ms / 1e3),
            (mb / 1024.0) / (gpu_sync_ms / 1e3),
            (mb / 1024.0) / (gpu_batch_ms / 1e3),
            cpu_ms / gpu_batch_ms,
        );
    }

    #[test]
    fn ring_allocator_wrap_fence_recycling_and_oversized() {
        let mut ring = RingAllocator::new(4096);
        assert_eq!(ring.used_bytes(), 0);
        assert_eq!(ring.oldest_fence(), None);

        // Single request exceeding ring capacity returns Oversized.
        assert_eq!(ring.alloc(4097, RING_ALIGN), RingAlloc::Oversized);

        // Sub-allocate three 1024-byte slices (aligned to 512) under fences 1 and 2.
        assert_eq!(
            ring.alloc(700, RING_ALIGN),
            RingAlloc::Slice {
                offset: 0,
                aligned_size: 1024,
                wrapped: false
            }
        );
        ring.commit_batch(1);
        assert_eq!(
            ring.alloc(1024, RING_ALIGN),
            RingAlloc::Slice {
                offset: 1024,
                aligned_size: 1024,
                wrapped: false
            }
        );
        assert_eq!(
            ring.alloc(1024, RING_ALIGN),
            RingAlloc::Slice {
                offset: 2048,
                aligned_size: 1024,
                wrapped: false
            }
        );
        ring.commit_batch(2);
        assert_eq!(ring.used_bytes(), 3072);
        assert_eq!(ring.oldest_fence(), Some(1));

        // 1536 bytes needs wrap (head=3072 -> tail gap 1024 + 1536 = 2560 > free 1024) -> Full.
        assert_eq!(ring.alloc(1536, RING_ALIGN), RingAlloc::Full);

        // Reclaim fence 1 (frees [0, 1024)): tail=1024, head=3072, used=2048.
        // Still cannot fit 1536 at offset 0 because [1024, 3072) is live!
        assert_eq!(ring.reclaim(1), 1024);
        assert_eq!(ring.used_bytes(), 2048);
        assert_eq!(ring.alloc(1536, RING_ALIGN), RingAlloc::Full);

        // Allocate 512 bytes at [3072, 3584), leaving only a 512-byte tail gap.
        assert_eq!(
            ring.alloc(512, RING_ALIGN),
            RingAlloc::Slice {
                offset: 3072,
                aligned_size: 512,
                wrapped: false
            }
        );
        ring.commit_batch(3);

        // Now 1024 bytes cannot fit in the 512-byte tail gap and wraps cleanly to
        // offset 0 (disjoint from [1024, 3584))!
        assert_eq!(
            ring.alloc(1024, RING_ALIGN),
            RingAlloc::Slice {
                offset: 0,
                aligned_size: 1024,
                wrapped: true
            }
        );
        ring.commit_batch(4);
        assert_eq!(ring.used_bytes(), 4096);
        assert_eq!(ring.alloc(512, RING_ALIGN), RingAlloc::Full);

        // Reclaim fence 2 (frees [1024, 3072)): live is [3072, 4096) + [0, 1024); free [1024, 3072).
        assert_eq!(ring.reclaim(2), 2048);
        assert_eq!(
            ring.alloc(1536, RING_ALIGN),
            RingAlloc::Slice {
                offset: 1024,
                aligned_size: 1536,
                wrapped: false
            }
        );
        ring.commit_batch(5);

        // Reclaim everything -> resets head/tail cleanly to 0.
        ring.reclaim(5);
        assert_eq!(ring.used_bytes(), 0);
        assert_eq!(ring.oldest_fence(), None);
        assert_eq!(
            ring.alloc(4096, RING_ALIGN),
            RingAlloc::Slice {
                offset: 0,
                aligned_size: 4096,
                wrapped: false
            }
        );
    }

    #[test]
    fn parse_config_defaults_and_opt_out() {
        assert!(parse_compute_enabled(None));
        assert!(parse_compute_enabled(Some("1")));
        assert!(!parse_compute_enabled(Some("0")));
        assert!(!parse_compute_enabled(Some(" 0 ")));

        assert_eq!(parse_ring_bytes(None), DEFAULT_RING_MB << 20);
        assert_eq!(parse_ring_bytes(Some("4")), MIN_RING_MB << 20);
        assert_eq!(parse_ring_bytes(Some("32")), 32 << 20);
        assert_eq!(parse_ring_bytes(Some("512")), MAX_RING_MB << 20);
    }

    #[test]
    fn compute_queue_batched_detile_matches_cpu_and_returns_common() {
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let Some(host) = host() else {
            return;
        };

        struct TexFixture {
            layout: Layout,
            tiled: Vec<u8>,
            src_buf: ID3D12Resource,
            dst_tex: ID3D12Resource,
            footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
            rows: Vec<u32>,
            jobs: Vec<SubresourceJob>,
            indices: Vec<usize>,
            total: u64,
            bpe: usize,
        }

        let specs = [
            (256u32, 256u32, 9u32, 16u32, 4u32, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_64KB_D, 101u64),
            (128u32, 96u32, 7u32, 4u32, 1u32, DXGI_FORMAT_R8G8B8A8_UNORM, Swizzle::SW_64KB_S, 202u64),
            (192u32, 128u32, 5u32, 16u32, 4u32, DXGI_FORMAT_BC7_UNORM, Swizzle::SW_4KB_S, 303u64),
        ];

        let mut fixtures = Vec::new();
        for (w, h, mips, bpe, block, fmt, swizzle, seed) in specs {
            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex2d,
                width: w,
                height: h,
                depth: 1,
                mips,
                element_bytes: bpe,
                block,
                samples: 1,
                swizzle,
            })
            .unwrap();
            let tiled = fill_pattern(layout.size as usize, seed);
            let (src_buf, src_ptr) = host.upload_buffer(layout.size.next_multiple_of(4)).unwrap();
            // SAFETY: mapped upload buffer covers `tiled.len()`.
            unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };

            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: mips as u16,
                Format: fmt,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mips as usize];
            let mut rows = vec![0u32; mips as usize];
            let mut total = 0u64;
            // SAFETY: valid queries on live device.
            unsafe {
                host.device.GetCopyableFootprints(
                    &tex_desc,
                    0,
                    mips,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    None,
                    Some(&mut total),
                );
            }
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut tex_opt: Option<ID3D12Resource> = None;
            // Destination enters ComputeQ strictly in COMMON!
            // SAFETY: plain committed texture creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &tex_desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut tex_opt,
                    )
                    .unwrap();
            }
            let dst_tex = tex_opt.unwrap();
            let mut jobs = Vec::with_capacity(mips as usize);
            let mut indices = Vec::with_capacity(mips as usize);
            for i in 0..mips as usize {
                let row_pitch = footprints[i].Footprint.RowPitch;
                jobs.push(SubresourceJob {
                    mip: i as u32,
                    slice: 0,
                    dst_offset: footprints[i].Offset,
                    row_pitch,
                    depth_pitch: row_pitch * rows[i],
                    zero: false,
                });
                indices.push(i);
            }
            fixtures.push(TexFixture {
                layout,
                tiled,
                src_buf,
                dst_tex,
                footprints,
                rows,
                jobs,
                indices,
                total,
                bpe: bpe as usize,
            });
        }

        let items: Vec<BatchItem<'_>> = fixtures
            .iter()
            .map(|f| BatchItem {
                layout: &f.layout,
                // SAFETY: live upload buffer.
                src_gpu_va: unsafe { f.src_buf.GetGPUVirtualAddress() },
                src_len: f.tiled.len() as u64,
                dst_texture: &f.dst_tex,
                sub_offset: 0,
                footprints: &f.footprints,
                jobs: &f.jobs,
                indices: &f.indices,
                linear_bytes: f.total,
                need_aliasing: false,
            })
            .collect();

        // Execute all 3 textures in ONE batched ComputeQ submission (async), then
        // synchronize Host::queue via GPU-side `wait_on_queue` before reading back!
        let fence_val = execute_compute_batch(host, &items, false)
            .expect("execute_compute_batch failed");
        assert!(fence_val > 0);
        assert!(wait_on_queue(host, host.queue(), fence_val));

        for (tidx, f) in fixtures.iter().enumerate() {
            for mip in [0usize, 1, f.footprints.len() - 1] {
                let fp = f.footprints[mip];
                let pitch = fp.Footprint.RowPitch as usize;
                let h = f.rows[mip] as usize;
                let len = pitch * h;
                let rb = host.read_back_buffer(len as u64).unwrap();
                let list = host.new_list().unwrap();
                // Texture MUST have exited ComputeQ in COMMON so COMMON -> COPY_SOURCE is legal!
                // SAFETY: open direct list, live texture and readback buffer.
                unsafe {
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &f.dst_tex,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&rb),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, ..fp },
                        },
                    };
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&f.dst_tex),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: mip as u32 },
                    };
                    list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &f.dst_tex,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                list.close();
                host.execute_sync_internal(&list);

                let mut ptr = std::ptr::null_mut();
                // SAFETY: readback completed.
                let got = unsafe {
                    rb.Map(0, None, Some(&mut ptr)).unwrap();
                    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
                    rb.Unmap(0, None);
                    bytes
                };
                let mut expected = vec![0u8; len];
                f.layout
                    .detile(&f.tiled, mip as u32, 0, &mut expected, pitch, len)
                    .unwrap();
                let active = f.layout.mips[mip].width as usize * f.bpe;
                for y in 0..h {
                    assert_eq!(
                        &got[y * pitch..y * pitch + active],
                        &expected[y * pitch..y * pitch + active],
                        "ComputeQ batch tex={tidx} mip={mip} row={y} mismatch"
                    );
                }
            }
        }
        assert!(wait_compute_fence(host, fence_val));
    }

    #[test]
    fn record_batch_on_list_matches_cpu() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC};
        let Some(host) = host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 128,
            height: 128,
            depth: 1,
            mips: 5,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        })
        .unwrap();
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 128,
            Height: 128,
            DepthOrArraySize: 1,
            MipLevels: 5,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 5];
        let mut rows = vec![0u32; 5];
        let mut total = 0u64;
        // SAFETY: valid device query.
        unsafe {
            host.device.GetCopyableFootprints(
                &tex_desc,
                0,
                5,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                None,
                Some(&mut total),
            );
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let make_tex = |seed: u64| {
            let tiled = fill_pattern(layout.size as usize, seed);
            let (src_buf, src_ptr) = host.upload_buffer(layout.size.next_multiple_of(4)).unwrap();
            // SAFETY: mapped upload buffer.
            unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), src_ptr, tiled.len()) };
            let mut tex_opt: Option<ID3D12Resource> = None;
            // SAFETY: plain texture creation in COPY_DEST.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &tex_desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut tex_opt,
                    )
                    .unwrap();
            }
            (tiled, src_buf, tex_opt.unwrap())
        };
        let (t0, s0, dst0) = make_tex(777);
        let (t1, s1, dst1) = make_tex(888);
        let jobs: Vec<SubresourceJob> = (0..5usize)
            .map(|i| SubresourceJob {
                mip: i as u32,
                slice: 0,
                dst_offset: footprints[i].Offset,
                row_pitch: footprints[i].Footprint.RowPitch,
                depth_pitch: footprints[i].Footprint.RowPitch * rows[i],
                zero: false,
            })
            .collect();
        let indices: Vec<usize> = (0..5).collect();
        let items = [
            BatchItem {
                layout: &layout,
                // SAFETY: live upload buffer.
                src_gpu_va: unsafe { s0.GetGPUVirtualAddress() },
                src_len: t0.len() as u64,
                dst_texture: &dst0,
                sub_offset: 0,
                footprints: &footprints,
                jobs: &jobs,
                indices: &indices,
                linear_bytes: total,
                need_aliasing: false,
            },
            BatchItem {
                layout: &layout,
                // SAFETY: live upload buffer.
                src_gpu_va: unsafe { s1.GetGPUVirtualAddress() },
                src_len: t1.len() as u64,
                dst_texture: &dst1,
                sub_offset: 0,
                footprints: &footprints,
                jobs: &jobs,
                indices: &indices,
                linear_bytes: total,
                need_aliasing: false,
            },
        ];

        let list = host.new_list().unwrap();
        assert!(record_batch_on_list(host, &list.list, &items));
        list.close();
        host.execute_sync_internal(&list);

        for (idx, (tiled, dst)) in [(t0, dst0), (t1, dst1)].into_iter().enumerate() {
            let fp = footprints[0];
            let pitch = fp.Footprint.RowPitch as usize;
            let h = rows[0] as usize;
            let len = pitch * h;
            let rb = host.read_back_buffer(len as u64).unwrap();
            let list = host.new_list().unwrap();
            // SAFETY: open direct list, live resources.
            unsafe {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &dst,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&rb),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT { Offset: 0, ..fp },
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&dst),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
            list.close();
            host.execute_sync_internal(&list);

            let mut ptr = std::ptr::null_mut();
            // SAFETY: readback completed.
            let got = unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
                rb.Unmap(0, None);
                bytes
            };
            let mut expected = vec![0u8; len];
            layout.detile(&tiled, 0, 0, &mut expected, pitch, len).unwrap();
            let active = layout.mips[0].width as usize * 16;
            for y in 0..h {
                assert_eq!(
                    &got[y * pitch..y * pitch + active],
                    &expected[y * pitch..y * pitch + active],
                    "direct batch tex={idx} row={y} mismatch"
                );
            }
        }
    }
}
