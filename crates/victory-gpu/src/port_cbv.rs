//! Renderer port #5: the title's constant buffers read by the card where the
//! title has them (`VICTORY_PORT_CBV=1`, off by default; read in [`on`]).
//!
//! What the console allows: a constant buffer anywhere on 16 bytes. What
//! D3D12 asks: a constant buffer view on 256. So the layer made every
//! constant buffer a draw's tables name afresh at each submission: a copy of
//! its bytes (out of title memory, on the CPU) into upload memory of its own
//! on 256, and a view of that ([`crate::draw`]'s tables, `defer_constants`):
//! ~23,000 a frame of the city, ~9 MB, and as many steps of the submitting
//! thread. With renderer port 1 (`VICTORY_PORT_CONSTANTS=1`) the memory they
//! are copied out of is the card's to read in place already.
//!
//! What is done here instead: shaders are translated to read constant buffers
//! as raw bytes (`victory_shader::emit::constants_raw`): a table's through
//! the raw twin of its slot, a root one as a root SRV. A raw view starts on 16
//! bytes, a root SRV on 4. So a constant buffer of a table in title memory in
//! place is a raw view of the title's bytes, made in the twin of its slot,
//! and nothing is copied or done at submission; a root one is bound at its
//! address as root SRVs are. Constant buffers anywhere else are copied as
//! before (the twin of their slot is a raw view of the copy, as it was).
//! The twin of a title slot that holds a constant buffer view is a raw view
//! of its bytes too (it was a constant buffer view: a raw read through one
//! on AMD has no end).
//!
//! What it assumes: what the card reads of a constant buffer at a draw is
//! what the title's CPU wrote before it submitted, as for every other title
//! buffer in place (renderer port 1): before, the copy at submission was
//! taken then whatever the title wrote after. The title writes its constants
//! into a ring it waits on its fences for.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether renderer port #5 is on: the shaders' side
/// (`victory_shader::emit::constants_raw`) and the host's views follow the
/// same switch.
pub(crate) fn on() -> bool {
    victory_shader::emit::constants_raw()
}

/// Whether a table's constant buffer in title memory in place is given to the
/// card where it is (on with the port; `VICTORY_PORT_CBV_PLACE=0`: copied as
/// before, and read through the raw twin all the same). With
/// `VICTORY_AB=<s>,VICTORY_PORT_CBV_PLACE` (diagnostics builds) on and off
/// while the title runs: both sides measured in one run.
pub(crate) fn in_place_on() -> bool {
    on() && crate::toggled!("VICTORY_PORT_CBV_PLACE")
}

/// Whether raw descriptor twins are active (`VICTORY_RAW_VIEWS != 0`).
fn raw_twins_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAW_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether tables whose CBVs are all in-place bind directly through their
/// title host-heap slots instead of lock-free batched dynamic snapshots.
/// Opt-in (`VICTORY_PORT_CBV_DIRECT=1`, default OFF): title-slot mirroring
/// takes per-slot locks and `ResourceTable` mutexes across recording threads,
/// whereas batched dynamic snapshots (`TableBatch` + lock-free `DynamicKept`)
/// cut `list::thin` CPU by ~63% (`+60%` FPS on both RT-Off and Native DXR).
pub(crate) fn table_direct_on() -> bool {
    in_place_on()
        && raw_twins_on()
        && std::env::var_os("VICTORY_PORT_CBV_DIRECT").is_some_and(|v| v == "1")
}

/// Pure eligibility gate enforcing all four V1 guards for bypassing dynamic
/// table snapshots on CBV-bearing tables (`CPU-18`).
#[inline]
pub(crate) fn table_direct_eligible(
    direct_on: bool,
    has_half: bool,
    has_indexed: bool,
    fits_host: bool,
    all_cbvs_in_place: bool,
) -> bool {
    direct_on && !has_half && !has_indexed && fits_host && all_cbvs_in_place
}

/// Maximum constant buffer size in D3D12 (64 KiB).
pub(crate) const MAX_CBV_BYTES: u32 = 64 << 10;

/// Validated host view layout for a constant buffer read as a raw ByteAddressBuffer SRV
/// (plus an optional 256-byte-aligned CBV fallback descriptor in the primary slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RawCbvLayout {
    /// `FirstElement` (in 32-bit dwords) for `D3D12_BUFFER_SRV_FLAG_RAW`.
    pub(crate) first_element: u64,
    /// `NumElements` (in 32-bit dwords, > 0) for `D3D12_BUFFER_SRV_FLAG_RAW`.
    pub(crate) num_elements: u32,
    /// `BufferLocation` for the primary CBV slot (`0` if unaligned to 256 or room < 256).
    pub(crate) cbv_location: u64,
    /// `SizeInBytes` (multiple of 256) for the primary CBV slot (`0` when `cbv_location == 0`).
    pub(crate) cbv_size: u32,
}

/// Computes the validated raw-SRV and optional 256-byte-aligned CBV layout for
/// a constant buffer of `size` bytes at host address `gpu` inside a buffer
/// starting at `base` of `width` bytes.
pub(crate) fn raw_cbv_layout(base: u64, width: u64, gpu: u64, size: u32) -> Option<RawCbvLayout> {
    if size == 0 {
        return None;
    }
    let offset = gpu.checked_sub(base)?;
    if offset >= width || !raw_ok(offset) || !raw_ok(gpu) {
        return None;
    }
    let room = width - offset;
    let clamped = size.min(MAX_CBV_BYTES);
    let wanted = clamped.next_multiple_of(16);
    let raw_bytes = u64::from(wanted).min(room) & !3;
    if raw_bytes == 0 {
        return None;
    }
    let first_element = offset / 4;
    let num_elements = (raw_bytes / 4) as u32;
    let cbv_bytes = u64::from(clamped.next_multiple_of(256)).min(room & !255) as u32;
    let (cbv_location, cbv_size) = if gpu.is_multiple_of(256) && cbv_bytes >= 256 {
        (gpu, cbv_bytes)
    } else {
        (0, 0)
    };
    Some(RawCbvLayout { first_element, num_elements, cbv_location, cbv_size })
}

/// Cache-line-aligned per-shard counter bank (`CPU-26`) so concurrent
/// recording threads never bounce a single shared stats cache line.
#[repr(align(64))]
struct PaddedStats([AtomicU64; 7]);

const STAT_SHARDS: usize = 8;

/// Since the last line: tables' constant buffers given in place, their bytes;
/// copied as before because not in place, because not on 16 bytes; title
/// slots' constant buffers whose twin is a raw view; copied because out of
/// bounds or empty; root constant buffers bound in place.
static STATS: [PaddedStats; STAT_SHARDS] =
    [const { PaddedStats([const { AtomicU64::new(0) }; 7]) }; STAT_SHARDS];

static NEXT_SHARD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static SHARD: usize = NEXT_SHARD.fetch_add(1, Relaxed) & (STAT_SHARDS - 1);
}

#[inline]
fn shard_stats() -> &'static [AtomicU64; 7] {
    let idx = SHARD.with(|&i| i);
    &STATS[idx].0
}

/// A table's constant buffer of `size` bytes was given in place.
pub(crate) fn in_place(size: u32) {
    let s = shard_stats();
    s[0].fetch_add(1, Relaxed);
    s[1].fetch_add(u64::from(size), Relaxed);
}

/// A table's constant buffer was copied as before: `aligned` false when it
/// is in place but not on 16 bytes.
pub(crate) fn copied(aligned: bool) {
    shard_stats()[if aligned { 2 } else { 3 }].fetch_add(1, Relaxed);
}

/// A title slot's constant buffer view got a raw twin.
pub(crate) fn twinned() {
    shard_stats()[4].fetch_add(1, Relaxed);
}

/// A table's constant buffer in place fell back to copying because it was
/// out of bounds or empty.
#[allow(dead_code)]
pub(crate) fn copied_oob_or_empty() {
    shard_stats()[5].fetch_add(1, Relaxed);
}

/// A root constant buffer was bound in place.
#[allow(dead_code)]
pub(crate) fn root_in_place() {
    shard_stats()[6].fetch_add(1, Relaxed);
}

/// Drains and sums all sharded stat counters.
fn take_stats() -> [u64; 7] {
    let mut totals = [0u64; 7];
    for shard in &STATS {
        for (dst, src) in totals.iter_mut().zip(&shard.0) {
            *dst += src.swap(0, Relaxed);
        }
    }
    totals
}

/// Whether a constant buffer at host address `gpu` can be a raw view of
/// those bytes: on 16 bytes.
pub(crate) fn raw_ok(gpu: u64) -> bool {
    gpu.is_multiple_of(16)
}

/// Whether a constant buffer at host address `gpu` can be bound as a direct
/// root SRV: on 4 bytes.
#[allow(dead_code)]
pub(crate) fn root_cbv_ok(gpu: u64) -> bool {
    gpu.is_multiple_of(4)
}

/// Whether a constant buffer at host address `gpu` can be bound in a bounded
/// root-SRV table: on 16 bytes.
#[allow(dead_code)]
pub(crate) fn root_table_cbv_ok(gpu: u64) -> bool {
    raw_ok(gpu)
}

/// For the frame-rate line.
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let denom = frames.max(1e-9);
    let s = take_stats().map(|v| v as f64 / denom);
    tracing::info!(
        "renderer port 5 (VICTORY_PORT_CBV=1) per frame: {:.0} tables' constant buffers read by the card in place ({:.1} KB not copied, \
         no step at submission), {:.0} root CBVs in place; copied as before: {:.0} not in place, {:.0} not on 16 bytes, {:.0} OOB/empty; \
         {:.0} title slots' constant buffers with a raw twin",
        s[0],
        s[1] / 1024.0,
        s[6],
        s[2],
        s[3],
        s[5],
        s[4],
    );
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use windows::Win32::Foundation::{CloseHandle, E_FAIL, HANDLE};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::{Interface, s};
    use windows_sys::Win32::Foundation::{CloseHandle as SysCloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory as mem;

    use super::*;

    const K64: usize = 64 << 10;
    const OUT_BYTES: usize = 64;
    const POISON_BYTE: u8 = 0xA5;
    const POISON_WORD: u32 = u32::from_le_bytes([POISON_BYTE; 4]);

    #[test]
    fn raw_views_need_sixteen_bytes() {
        assert!(raw_ok(0x3500_0010));
        assert!(raw_ok(0x3500_0100));
        assert!(!raw_ok(0x3500_0008));
        assert!(!raw_ok(0x3500_0004));
        assert!(root_cbv_ok(0x3500_0004));
        assert!(root_cbv_ok(0x3500_0010));
        assert!(!root_cbv_ok(0x3500_0002));
        assert!(root_table_cbv_ok(0x3500_0010));
        assert!(!root_table_cbv_ok(0x3500_0004));
    }

    #[test]
    fn table_direct_eligible_enforces_all_four_guards() {
        // All guards satisfied -> bypasses dynamic snapshot.
        assert!(table_direct_eligible(true, false, false, true, true));
        // Guard 1: disabled via toggle / raw views off.
        assert!(!table_direct_eligible(false, false, false, true, true));
        // Guard 2: table has half-view slot.
        assert!(!table_direct_eligible(true, true, false, true, true));
        // Guard 3: table has indexed descriptor array.
        assert!(!table_direct_eligible(true, false, true, true, true));
        // Guard 4a: table exceeds host heap count.
        assert!(!table_direct_eligible(true, false, false, false, true));
        // Guard 4b: at least one CBV is not in-place / misaligned / OOB.
        assert!(!table_direct_eligible(true, false, false, true, false));
    }

    #[test]
    fn sharded_padded_stats_avoid_false_sharing_and_sum_across_threads() {
        assert_eq!(std::mem::align_of::<PaddedStats>(), 64);
        let _ = take_stats();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    in_place(256);
                    in_place(64);
                    copied(true);
                    copied(false);
                    twinned();
                    copied_oob_or_empty();
                    root_in_place();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let totals = take_stats();
        assert_eq!(totals[0], 8);
        assert_eq!(totals[1], 4 * (256 + 64));
        assert_eq!(totals[2], 4);
        assert_eq!(totals[3], 4);
        assert_eq!(totals[4], 4);
        assert_eq!(totals[5], 4);
        assert_eq!(totals[6], 4);
        assert_eq!(take_stats(), [0; 7]);
    }

    #[test]
    fn raw_cbv_layout_alignment() {
        let base = 0x1000_0000u64;
        let width = 0x1_0000u64;

        // 256-byte aligned gets both raw SRV and primary CBV descriptor.
        assert_eq!(
            raw_cbv_layout(base, width, base, 256),
            Some(RawCbvLayout { first_element: 0, num_elements: 64, cbv_location: base, cbv_size: 256 }),
        );

        // 16-byte aligned offsets get raw SRV only (primary CBV slot is null).
        for off in [0x010u64, 0x030, 0x0F0] {
            assert_eq!(
                raw_cbv_layout(base, width, base + off, 48),
                Some(RawCbvLayout { first_element: off / 4, num_elements: 12, cbv_location: 0, cbv_size: 0 }),
                "offset {off:#x}",
            );
        }

        // Sub-16-byte misaligned offsets are rejected.
        for off in [4u64, 8, 12, 0x14, 0x18, 0x1C] {
            assert_eq!(raw_cbv_layout(base, width, base + off, 48), None, "offset {off:#x}");
        }

        // Misaligned base: both offset and gpu must be 16-byte aligned.
        assert_eq!(raw_cbv_layout(base + 4, width, base + 0x14, 48), None);
        assert_eq!(raw_cbv_layout(base + 4, width, base + 0x10, 48), None);
    }

    #[test]
    fn raw_cbv_layout_zero_and_tiny_sizes() {
        let base = 0x2000_0000u64;
        let width = 0x1_0000u64;

        assert_eq!(raw_cbv_layout(base, width, base, 0), None);

        for (size, expected_dwords) in [(4u32, 4u32), (16, 4), (17, 8), (48, 12), (64, 16)] {
            assert_eq!(
                raw_cbv_layout(base, width, base, size),
                Some(RawCbvLayout { first_element: 0, num_elements: expected_dwords, cbv_location: base, cbv_size: 256 }),
                "size {size}",
            );
        }
    }

    #[test]
    fn raw_cbv_layout_partial_tail_room() {
        let base = 0x3000_0000u64;

        // 256-aligned gpu with only 48 bytes left in buffer: raw SRV clamped to 48 bytes, CBV null.
        assert_eq!(
            raw_cbv_layout(base, 0x1030, base + 0x1000, 64),
            Some(RawCbvLayout { first_element: 0x1000 / 4, num_elements: 12, cbv_location: 0, cbv_size: 0 }),
        );

        // Room < 4 bytes rounds down to 0 raw bytes -> rejected.
        assert_eq!(raw_cbv_layout(base, 0x1002, base + 0x1000, 16), None);

        // Room = 6 bytes rounds down to 4 bytes (1 dword), CBV null.
        assert_eq!(
            raw_cbv_layout(base, 0x1006, base + 0x1000, 16),
            Some(RawCbvLayout { first_element: 0x1000 / 4, num_elements: 1, cbv_location: 0, cbv_size: 0 }),
        );

        // Room = 256 bytes with larger size: both raw SRV and CBV clamp to 256 bytes.
        assert_eq!(
            raw_cbv_layout(base, 0x1100, base + 0x1000, 512),
            Some(RawCbvLayout { first_element: 0x1000 / 4, num_elements: 64, cbv_location: base + 0x1000, cbv_size: 256 }),
        );
    }

    #[test]
    fn raw_cbv_layout_clamps_above_64k() {
        let base = 0x4000_0000u64;
        let width = 256 << 10;

        assert_eq!(
            raw_cbv_layout(base, width, base, 128 << 10),
            Some(RawCbvLayout { first_element: 0, num_elements: 16384, cbv_location: base, cbv_size: MAX_CBV_BYTES }),
        );
    }

    #[test]
    fn raw_cbv_layout_above_4gib_width_does_not_truncate() {
        let base = 0x1_0000_0000u64;
        let width = 5u64 << 30;
        let offset = 1u64 << 30; // room == 4 GiB (0x1_0000_0000)
        let gpu = base + offset;

        let layout = raw_cbv_layout(base, width, gpu, 256).expect("4 GiB room must not truncate cbv_size");
        assert_eq!(layout.first_element, offset / 4);
        assert_eq!(layout.num_elements, 64);
        assert_eq!(layout.cbv_location, gpu);
        assert_eq!(layout.cbv_size, 256);

        let full = raw_cbv_layout(base, width, gpu, MAX_CBV_BYTES).expect("4 GiB room with 64 KiB CBV");
        assert_eq!(full.num_elements, 16384);
        assert_eq!(full.cbv_location, gpu);
        assert_eq!(full.cbv_size, MAX_CBV_BYTES);
    }

    #[test]
    fn raw_cbv_layout_out_of_bounds() {
        let base = 0x5000_0000u64;
        let width = 0x1000u64;

        assert_eq!(raw_cbv_layout(base, width, base - 16, 64), None);
        assert_eq!(raw_cbv_layout(base, width, base + width, 64), None);
        assert_eq!(raw_cbv_layout(base, width, base + width + 256, 64), None);
    }

    fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        }
    }

    fn compile_compute(device: &ID3D12Device, source: &str) -> windows::core::Result<(ID3D12RootSignature, ID3D12PipelineState)> {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source and entrypoint names stay alive through D3DCompile;
        // the compiled bytecode blob embeds the root signature.
        unsafe {
            if let Err(err) = D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("port_cbv"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&raw mut errors),
            ) {
                if let Some(blob) = errors {
                    let msg = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
                    panic!("D3DCompile failed ({err}): {}", String::from_utf8_lossy(msg));
                }
                return Err(err);
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                ..Default::default()
            })?;
            Ok((root, pso))
        }
    }

    fn transition(
        list: &ID3D12GraphicsCommandList,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        // SAFETY: `resource` outlives the submitted command list; `ManuallyDrop`
        // prevents releasing our COM reference when the barrier struct drops.
        unsafe {
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(resource),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            };
            list.ResourceBarrier(&[barrier]);
        }
    }

    struct Cmd {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        done: u64,
        event: HANDLE,
    }

    impl Drop for Cmd {
        fn drop(&mut self) {
            // SAFETY: event handle owned by this test rig.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }

    impl Cmd {
        fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
            // SAFETY: plain D3D12 command objects owned by this test.
            unsafe {
                let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let event = CreateEventW(None, false, false, None)?;
                Ok(Self { device: device.clone(), queue, allocator, list, fence, done: 0, event })
            }
        }

        fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> windows::core::Result<()> {
            // SAFETY: exclusive test command list, queue, and fence.
            unsafe {
                self.allocator.Reset()?;
                self.list.Reset(&self.allocator, None)?;
                record(&self.list);
                self.list.Close()?;
                self.queue.ExecuteCommandLists(&[Some(self.list.cast()?)]);
                self.done += 1;
                self.queue.Signal(&self.fence, self.done)?;
                self.fence.SetEventOnCompletion(self.done, self.event)?;
                if WaitForSingleObject(self.event, 10_000).0 != 0 {
                    return Err(windows::core::Error::from_hresult(E_FAIL));
                }
                self.device.GetDeviceRemovedReason()
            }
        }
    }

    struct ComputeRig {
        cmd: Cmd,
        srv_heap: ID3D12DescriptorHeap,
        srv_inc: u32,
        table_root: ID3D12RootSignature,
        table_pso: ID3D12PipelineState,
        root_srv_root: ID3D12RootSignature,
        root_srv_pso: ID3D12PipelineState,
        out_buf: ID3D12Resource,
        poison_up: ID3D12Resource,
        readback: ID3D12Resource,
    }

    impl ComputeRig {
        fn new(device: &ID3D12Device, num_descriptors: u32) -> windows::core::Result<Self> {
            const TABLE_CS: &str = "\
                #define RS \"DescriptorTable(SRV(t0)), UAV(u0)\"\n\
                ByteAddressBuffer cb : register(t0);\n\
                RWByteAddressBuffer out_buf : register(u0);\n\
                [RootSignature(RS)]\n\
                [numthreads(1, 1, 1)]\n\
                void main() {\n\
                    out_buf.Store4(0, cb.Load4(0));\n\
                    out_buf.Store4(16, cb.Load4(16));\n\
                    out_buf.Store4(32, cb.Load4(32));\n\
                    out_buf.Store(48, cb.Load(64));\n\
                }\n";

            const ROOT_SRV_CS: &str = "\
                #define RS \"SRV(t0), UAV(u0)\"\n\
                ByteAddressBuffer cb : register(t0);\n\
                RWByteAddressBuffer out_buf : register(u0);\n\
                [RootSignature(RS)]\n\
                [numthreads(1, 1, 1)]\n\
                void main() {\n\
                    out_buf.Store4(0, cb.Load4(0));\n\
                    out_buf.Store4(16, cb.Load4(16));\n\
                    out_buf.Store4(32, cb.Load4(32));\n\
                }\n";

            let cmd = Cmd::new(device)?;
            let (table_root, table_pso) = compile_compute(device, TABLE_CS)?;
            let (root_srv_root, root_srv_pso) = compile_compute(device, ROOT_SRV_CS)?;

            let committed = |kind: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES| -> windows::core::Result<ID3D12Resource> {
                let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
                let mut res: Option<ID3D12Resource> = None;
                // SAFETY: plain committed buffer creation.
                unsafe {
                    device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer_desc(OUT_BYTES as u64, flags), state, None, &mut res)?;
                }
                res.ok_or_else(windows::core::Error::empty)
            };

            let out_buf = committed(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let poison_up = committed(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let readback = committed(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;

            // SAFETY: upload buffer holds OUT_BYTES bytes while mapped; descriptor heap is plain creation.
            let (srv_heap, srv_inc) = unsafe {
                let mut data = std::ptr::null_mut();
                poison_up.Map(0, None, Some(&mut data))?;
                std::ptr::write_bytes(data.cast::<u8>(), POISON_BYTE, OUT_BYTES);
                poison_up.Unmap(0, None);

                let srv_heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: num_descriptors,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
                let srv_inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                (srv_heap, srv_inc)
            };

            Ok(Self { cmd, srv_heap, srv_inc, table_root, table_pso, root_srv_root, root_srv_pso, out_buf, poison_up, readback })
        }

        fn create_raw_srv(&self, slot: u32, buffer: &ID3D12Resource, layout: RawCbvLayout) {
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: layout.first_element,
                        NumElements: layout.num_elements,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: valid slot inside our descriptor heap and live buffer with validated range.
            unsafe {
                let start = self.srv_heap.GetCPUDescriptorHandleForHeapStart();
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start.ptr + (slot as usize) * (self.srv_inc as usize) };
                self.cmd.device.CreateShaderResourceView(buffer, Some(&desc), handle);
            }
        }

        fn read_out(&self) -> windows::core::Result<[u32; 16]> {
            let mut data = std::ptr::null_mut();
            // SAFETY: readback buffer holds OUT_BYTES (16 dwords) while mapped.
            unsafe {
                self.readback.Map(0, None, Some(&mut data))?;
                let bytes = std::slice::from_raw_parts(data.cast::<u8>(), OUT_BYTES);
                let mut words = [0u32; 16];
                for (dst, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
                    *dst = u32::from_le_bytes(chunk.try_into().unwrap());
                }
                self.readback.Unmap(0, None);
                Ok(words)
            }
        }

        fn dispatch_table(&mut self, slot: u32) -> windows::core::Result<[u32; 16]> {
            // SAFETY: live heap, buffers, root signature, and PSO.
            let gpu_handle = unsafe {
                let start = self.srv_heap.GetGPUDescriptorHandleForHeapStart();
                D3D12_GPU_DESCRIPTOR_HANDLE { ptr: start.ptr + u64::from(slot) * u64::from(self.srv_inc) }
            };
            let out_gpu = unsafe { self.out_buf.GetGPUVirtualAddress() };
            self.cmd.run(|list| unsafe {
                list.CopyBufferRegion(&self.out_buf, 0, &self.poison_up, 0, OUT_BYTES as u64);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list.SetDescriptorHeaps(&[Some(self.srv_heap.clone())]);
                list.SetComputeRootSignature(&self.table_root);
                list.SetPipelineState(&self.table_pso);
                list.SetComputeRootDescriptorTable(0, gpu_handle);
                list.SetComputeRootUnorderedAccessView(1, out_gpu);
                list.Dispatch(1, 1, 1);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list.CopyBufferRegion(&self.readback, 0, &self.out_buf, 0, OUT_BYTES as u64);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            })?;
            self.read_out()
        }

        fn dispatch_root_srv(&mut self, srv_gpu: u64) -> windows::core::Result<[u32; 16]> {
            // SAFETY: live buffers, root signature, PSO, and 4-byte-aligned GPU address.
            let out_gpu = unsafe { self.out_buf.GetGPUVirtualAddress() };
            self.cmd.run(|list| unsafe {
                list.CopyBufferRegion(&self.out_buf, 0, &self.poison_up, 0, OUT_BYTES as u64);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                list.SetComputeRootSignature(&self.root_srv_root);
                list.SetPipelineState(&self.root_srv_pso);
                list.SetComputeRootShaderResourceView(0, srv_gpu);
                list.SetComputeRootUnorderedAccessView(1, out_gpu);
                list.Dispatch(1, 1, 1);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list.CopyBufferRegion(&self.readback, 0, &self.out_buf, 0, OUT_BYTES as u64);
                transition(list, &self.out_buf, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            })?;
            self.read_out()
        }
    }

    /// Writes `dwords` at `base + byte_offset` without assuming pointer alignment.
    unsafe fn write_dwords(base: *mut u8, byte_offset: usize, dwords: &[u32]) {
        // SAFETY: caller guarantees `[base + byte_offset, base + byte_offset + dwords.len() * 4)` is writable.
        unsafe {
            let dst = base.add(byte_offset);
            for (i, &w) in dwords.iter().enumerate() {
                std::ptr::copy_nonoverlapping(w.to_le_bytes().as_ptr(), dst.add(i * 4), 4);
            }
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    fn run_private_heap_raw_cbv_test(device: &ID3D12Device) {
        const SIZE: usize = 4 * K64;
        // SAFETY: private allocation owned and released by this test.
        unsafe {
            let cpu_base = mem::VirtualAlloc(std::ptr::null(), SIZE, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE).cast::<u8>();
            assert!(!cpu_base.is_null(), "VirtualAlloc failed");

            let heap = crate::inplace::open(device, crate::inplace::From::Part(cpu_base as usize, SIZE))
                .expect("OpenExistingHeapFromAddress1 failed");
            let buffer = crate::inplace::place(device, &heap, 0, SIZE as u64).expect("CreatePlacedResource failed");
            let gpu_base = buffer.GetGPUVirtualAddress();
            let width = buffer.GetDesc().Width;

            let offsets = [0x000usize, 0x010, 0x030, 0x0F0];
            let mut rig = ComputeRig::new(device, offsets.len() as u32).expect("ComputeRig::new failed");

            for (slot, &off) in offsets.iter().enumerate() {
                let gpu = gpu_base + off as u64;
                let layout = raw_cbv_layout(gpu_base, width, gpu, 48).expect("valid 16-byte-aligned raw CBV layout");
                assert_eq!(layout.first_element, (off / 4) as u64);
                assert_eq!(layout.num_elements, 12);
                if off.is_multiple_of(256) {
                    assert_eq!(layout.cbv_location, gpu);
                    assert_eq!(layout.cbv_size, 256);
                } else {
                    assert_eq!(layout.cbv_location, 0);
                    assert_eq!(layout.cbv_size, 0);
                }
                rig.create_raw_srv(slot as u32, &buffer, layout);

                // Initial CPU write + non-zero poison immediately past the 48-byte CBV (including byte +64).
                let v1: [u32; 12] = std::array::from_fn(|i| 0x1000_0000 + ((slot as u32) << 16) + i as u32 * 0x111 + 1);
                let poison1: [u32; 8] = [0xDEAD_BEEF, 0xFEED_FACE, 0x1337_C0DE, 0xBAAD_F00D, 0xCAFE_BABE, 0x0BAD_F00D, 0x55AA_55AA, 0x9988_7766];
                write_dwords(cpu_base, off, &v1);
                write_dwords(cpu_base, off + 48, &poison1);

                let got1 = rig.dispatch_table(slot as u32).expect("dispatch_table (v1) failed");
                assert_eq!(&got1[..12], &v1, "offset {off:#x} v1 in-bounds loads");
                assert_eq!(got1[12], 0, "offset {off:#x} v1 out-of-bounds Load(64) must return 0 despite poison bytes");
                assert_eq!(got1[15], POISON_WORD, "untouched tail confirms output buffer was pre-poisoned");

                // Mutate CPU memory in place with NO copy and NO descriptor recreation.
                let v2: [u32; 12] = std::array::from_fn(|i| 0x2000_0000 + ((slot as u32) << 16) + (i as u32 + 1) * 0x333);
                let poison2: [u32; 8] = [0xFFFF_1111; 8];
                write_dwords(cpu_base, off, &v2);
                write_dwords(cpu_base, off + 48, &poison2);

                let got2 = rig.dispatch_table(slot as u32).expect("dispatch_table (v2) failed");
                assert_eq!(&got2[..12], &v2, "offset {off:#x} v2 live CPU update visible in place");
                assert_eq!(got2[12], 0, "offset {off:#x} v2 out-of-bounds Load(64) must remain 0");
            }

            drop(rig);
            drop(buffer);
            drop(heap);

            // The opened heap releases its hold on pages shortly after its last reference drops.
            let freed = (0..200).any(|_| {
                let ok = mem::VirtualFree(cpu_base.cast(), 0, mem::MEM_RELEASE) != 0;
                if !ok {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                ok
            });
            assert!(freed, "VirtualFree(MEM_RELEASE) failed after dropping heap");
        }
    }

    #[test]
    fn warp_raw_cbv_in_place_private_heap_16b_aligned_and_live_cpu_updates() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        run_private_heap_raw_cbv_test(&device);
    }

    #[test]
    fn warp_raw_cbv_in_place_committed_section_and_root_srv() {
        let _turn = crate::indirect::tests::warp_turn();
        let Some(device) = crate::inplace::tests::warp() else { return };
        if device.cast::<ID3D12Device3>().is_err() {
            return;
        }

        const SIZE: usize = 2 * K64;
        // SAFETY: committed section and mapped view owned and released by this test.
        unsafe {
            let handle = mem::CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_COMMIT,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!handle.is_null(), "CreateFileMappingW(SEC_COMMIT) failed");

            let view = mem::MapViewOfFile(handle, mem::FILE_MAP_WRITE, 0, 0, SIZE);
            assert!(!view.Value.is_null(), "MapViewOfFile failed");
            let cpu_base = view.Value.cast::<u8>();

            let heap = crate::inplace::open(&device, crate::inplace::From::Section(handle as usize))
                .expect("OpenExistingHeapFromFileMapping failed");
            let buffer = crate::inplace::place(&device, &heap, 0, SIZE as u64).expect("CreatePlacedResource failed");
            let gpu_base = buffer.GetGPUVirtualAddress();
            let width = buffer.GetDesc().Width;

            let mut rig = ComputeRig::new(&device, 1).expect("ComputeRig::new failed");

            // 1. Descriptor-table raw CBV at +0x010 (16-byte aligned).
            let table_off = 0x010usize;
            let table_gpu = gpu_base + table_off as u64;
            assert!(root_table_cbv_ok(table_gpu));
            let layout = raw_cbv_layout(gpu_base, width, table_gpu, 48).expect("section +0x010 layout");
            assert_eq!(layout.first_element, 4);
            assert_eq!(layout.num_elements, 12);
            assert_eq!(layout.cbv_location, 0);
            assert_eq!(layout.cbv_size, 0);
            rig.create_raw_srv(0, &buffer, layout);

            let table_v1: [u32; 12] = std::array::from_fn(|i| 0xA100_0000 + i as u32 * 0x17 + 3);
            let poison: [u32; 8] = [0xDEAD_C0DE; 8];
            write_dwords(cpu_base, table_off, &table_v1);
            write_dwords(cpu_base, table_off + 48, &poison);

            let got_t1 = rig.dispatch_table(0).expect("section table dispatch (v1)");
            assert_eq!(&got_t1[..12], &table_v1, "section table raw CBV v1");
            assert_eq!(got_t1[12], 0, "section table raw CBV OOB Load(64) must return 0");

            let table_v2: [u32; 12] = std::array::from_fn(|i| 0xA200_0000 + i as u32 * 0x29 + 5);
            write_dwords(cpu_base, table_off, &table_v2);
            let got_t2 = rig.dispatch_table(0).expect("section table dispatch (v2)");
            assert_eq!(&got_t2[..12], &table_v2, "section table raw CBV live update");
            assert_eq!(got_t2[12], 0, "section table raw CBV OOB Load(64) must remain 0");

            // 2. Direct root SRV at +0x004 (4-byte aligned, NOT 16-byte aligned).
            let root_off = 0x004usize;
            let root_gpu = gpu_base + root_off as u64;
            assert!(root_cbv_ok(root_gpu));
            assert!(!raw_ok(root_gpu));
            assert!(!root_table_cbv_ok(root_gpu));

            let root_v1: [u32; 12] = std::array::from_fn(|i| 0xB100_0000 + i as u32 * 0x31 + 7);
            write_dwords(cpu_base, root_off, &root_v1);
            let got_r1 = rig.dispatch_root_srv(root_gpu).expect("section root SRV dispatch (v1)");
            assert_eq!(&got_r1[..12], &root_v1, "section root SRV at +0x004 v1");

            let root_v2: [u32; 12] = std::array::from_fn(|i| 0xB200_0000 + i as u32 * 0x43 + 9);
            write_dwords(cpu_base, root_off, &root_v2);
            let got_r2 = rig.dispatch_root_srv(root_gpu).expect("section root SRV dispatch (v2)");
            assert_eq!(&got_r2[..12], &root_v2, "section root SRV at +0x004 live update");

            drop(rig);
            drop(buffer);
            drop(heap);
            mem::UnmapViewOfFile(view);
            SysCloseHandle(handle);
        }
    }

    fn hardware_device() -> Option<(ID3D12Device, String)> {
        // SAFETY: plain DXGI adapter enumeration and D3D12 device creation.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let mut index = 0u32;
            while let Ok(adapter) = factory.EnumAdapters1(index) {
                index += 1;
                let Ok(desc) = adapter.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                    continue;
                }
                let Some(device) = device else { continue };
                if device.cast::<ID3D12Device13>().is_err() {
                    continue;
                }
                let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..end]);
                return Some((device, name));
            }
            None
        }
    }

    #[test]
    fn hw_gpu_raw_cbv_in_place_coherency() {
        let Some((device, name)) = hardware_device() else {
            eprintln!("skipping hw_gpu_raw_cbv_in_place_coherency: no hardware D3D12Device13 adapter available");
            return;
        };
        eprintln!("running hw_gpu_raw_cbv_in_place_coherency on hardware adapter: {name}");
        run_private_heap_raw_cbv_test(&device);
    }
}
