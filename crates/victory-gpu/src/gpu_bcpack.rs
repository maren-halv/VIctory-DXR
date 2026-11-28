//! Native D3D12 compute shader BCPack decoder (`VICTORY_GPU_BCPACK=1`, opt-in, default off).
//!
//! # Algorithmic Parallelism Boundary
//!
//! BCPack decoding splits cleanly into three GPU stages with distinct parallelism:
//!
//! 1. **Stage 1 (`bcpack_lanes`, `[numthreads(16, 1, 1)]`, `Dispatch(num_streams, 1, 1)`)**:
//!    12- or 16-lane interleaved rANS decoding. Sequential *per stream* because the
//!    8-byte-per-round input schedule depends on each lane's `x < LOW` state, but
//!    **stream-parallel across batched requests**, with all 12/16 lanes of a stream
//!    stepping in parallel per round via a groupshared prefix sum.
//! 2. **Stage 2a (`bcpack_resolve_{plain,bc6,bc7}`, `[numthreads(1, 1, 1)]`, `Dispatch(num_streams, 1, 1)`)**:
//!    256-entry ring-buffer history and 255-entry dictionary delta reconstruction.
//!    Sequential recurrence *per stream* because block `i` may reference block `i - 1`,
//!    **stream-parallel across batched requests** with each stream's 256-entry ring
//!    buffer kept in groupshared memory (`Plain`: 10 KiB, `Bc6`: 15 KiB, `Bc7`: 29 KiB,
//!    all within D3D12's 32 KiB limit).
//! 3. **Stage 2b (`bcpack_pack_{plain,bc6,bc7}`, `[numthreads(64, 1, 1)]`, `Dispatch(total_blocks.div_ceil(64), 1, 1)`)**:
//!    BC1/BC3/BC4/BC5/BC6H/BC7 bit-packing. **100% embarrassingly block-parallel**:
//!    one thread per BC block across every block in the batch.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use victory_formats::bcpack::TABLES_SIZE;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, s};

const BUILTIN_TABLES: &[u8] = include_bytes!("../../victory-formats/src/bcpack-tables.bin");

const MODELS: usize = 128;
const BUCKETS: usize = 9;
const DICTIONARY: usize = 255;

// Byte offsets inside the packed GpuTables ByteAddressBuffer (all 4-byte aligned).
const OFF_FREQ: usize = 0; // 128 * 9 * 4 = 4608
const OFF_START: usize = OFF_FREQ + MODELS * BUCKETS * 4; // +5120 = 9728
const OFF_BUCKET: usize = OFF_START + MODELS * (BUCKETS + 1) * 4; // +32768 = 42496
const OFF_RANK1: usize = OFF_BUCKET + MODELS * 256; // +256 = 42752
const OFF_RANK3A: usize = OFF_RANK1 + 256; // +64 = 42816
const OFF_RANK3C: usize = OFF_RANK3A + 64; // +256 = 43072
const OFF_RANK4: usize = OFF_RANK3C + 256; // +64 = 43136
const OFF_RANK5: usize = OFF_RANK4 + 64; // +64 = 43200
const OFF_CTRL7: usize = OFF_RANK5 + 64; // +2048 = 45248
const OFF_DICT_BC1: usize = OFF_CTRL7 + 512 * 4; // +10200 = 55448
const OFF_DICT_BC3: usize = OFF_DICT_BC1 + DICTIONARY * PLAIN_BYTES; // +10200 = 65648
const OFF_DICT_BC4: usize = OFF_DICT_BC3 + DICTIONARY * PLAIN_BYTES; // +10200 = 75848
const OFF_DICT_BC5: usize = OFF_DICT_BC4 + DICTIONARY * PLAIN_BYTES; // +10200 = 86048
const OFF_DICT_BC6: usize = OFF_DICT_BC5 + DICTIONARY * PLAIN_BYTES; // +15300 = 101348
const OFF_DICT_BC7: usize = OFF_DICT_BC6 + DICTIONARY * BC6_BYTES; // +29580 = 130928
const GPU_TABLES_BYTES: usize = OFF_DICT_BC7 + DICTIONARY * BC7_BYTES;

const PLAIN_BYTES: usize = 40; // 10 u32s: ep[8], ix[32]
const BC6_BYTES: usize = 60; // 15 u32s: e[12 u16], ix[16 i16], meta(mode | part<<8)
const BC7_BYTES: usize = 116; // 29 u32s: meta(ctrl | bits0<<16 | bits1<<24), rows[24 u16], ix[32 i16]
const STREAM_DESC_BYTES: usize = 32; // 8 u32s per stream descriptor

/// Whether the native GPU BCPack compute decoder is enabled
/// (`VICTORY_GPU_BCPACK=1`, opt-in, default off).
#[allow(dead_code)]
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_BCPACK").is_some_and(|v| v == "1"))
}

/// One compressed BCPack stream in a batched GPU decode request.
#[allow(dead_code)]
pub struct StreamJob<'a> {
    pub bc: u8,
    pub src: &'a [u8],
    pub blocks: usize,
}

/// Parsed BCPack constant tables uploaded into a GPU-visible `ByteAddressBuffer`.
#[allow(dead_code)]
pub(crate) struct GpuTables {
    resource: ID3D12Resource,
    packed: Vec<u8>,
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for GpuTables {}
// SAFETY: as above.
unsafe impl Sync for GpuTables {}

impl GpuTables {
    /// Parses raw BCPack table bytes (`bcpack-tables.bin` format) and uploads the
    /// GPU-packed representation into a `ByteAddressBuffer`-compatible resource.
    #[allow(dead_code)]
    pub(crate) fn from_bytes(device: &ID3D12Device, bytes: &[u8]) -> Result<Self, String> {
        let packed = pack_tables(bytes)?;
        let resource = create_buffer(
            device,
            packed.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        upload_bytes(&resource, &packed)?;
        Ok(Self { resource, packed })
    }
}

struct ByteReader<'a>(&'a [u8]);

impl ByteReader<'_> {
    fn slice(&mut self, n: usize) -> &[u8] {
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        head
    }
    fn u8(&mut self) -> u8 {
        self.slice(1)[0]
    }
    fn u16(&mut self) -> u16 {
        let b = self.slice(2);
        u16::from_le_bytes([b[0], b[1]])
    }
}

fn write_u32_le(dst: &mut [u8], offset: usize, val: u32) {
    dst[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

fn pack_tables(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len() != TABLES_SIZE {
        return Err("BCPack: table file has the wrong size".to_string());
    }
    let mut r = ByteReader(bytes);
    let mut out = vec![0u8; GPU_TABLES_BYTES];

    let mut freq = [[0u32; BUCKETS]; MODELS];
    let mut start = [[0u32; BUCKETS + 1]; MODELS];
    for m in 0..MODELS {
        for b in 0..BUCKETS {
            freq[m][b] = u32::from(r.u16());
            write_u32_le(&mut out, OFF_FREQ + (m * BUCKETS + b) * 4, freq[m][b]);
        }
        let used = m / 16 + 2;
        let mut at = 0u32;
        for b in 0..used {
            if freq[m][b] == 0 {
                return Err("BCPack: a model has an empty bucket".to_string());
            }
            start[m][b] = at;
            at += freq[m][b] << b.saturating_sub(1);
        }
        if at != 1 << 16 {
            return Err("BCPack: a model does not tile 65536".to_string());
        }
        start[m][used..].fill(1 << 16);
        for b in 0..=BUCKETS {
            write_u32_le(&mut out, OFF_START + (m * (BUCKETS + 1) + b) * 4, start[m][b]);
        }
        for window in 0..256usize {
            let first = (window as u32) << 8;
            let b = start[m][1..].iter().filter(|&&s| s <= first).count() as u8;
            out[OFF_BUCKET + m * 256 + window] = b;
        }
    }

    out[OFF_RANK1..OFF_RANK1 + 256].copy_from_slice(r.slice(256));
    out[OFF_RANK3A..OFF_RANK3A + 64].copy_from_slice(r.slice(64));
    out[OFF_RANK3C..OFF_RANK3C + 256].copy_from_slice(r.slice(256));
    out[OFF_RANK4..OFF_RANK4 + 64].copy_from_slice(r.slice(64));
    out[OFF_RANK5..OFF_RANK5 + 64].copy_from_slice(r.slice(64));

    for c in 0..512usize {
        write_u32_le(&mut out, OFF_CTRL7 + c * 4, u32::from(r.u16()));
    }

    let mut pack_plain_dict = |base: usize, ep_n: usize, ix_n: usize| {
        for i in 0..DICTIONARY {
            let entry = base + i * PLAIN_BYTES;
            out[entry..entry + ep_n].copy_from_slice(r.slice(ep_n));
            out[entry + 8..entry + 8 + ix_n].copy_from_slice(r.slice(ix_n));
        }
    };
    pack_plain_dict(OFF_DICT_BC1, 6, 16);
    pack_plain_dict(OFF_DICT_BC3, 8, 32);
    pack_plain_dict(OFF_DICT_BC4, 2, 16);
    pack_plain_dict(OFF_DICT_BC5, 4, 32);

    for i in 0..DICTIONARY {
        let entry = OFF_DICT_BC6 + i * BC6_BYTES;
        let _ep_mode = r.u8();
        for p in 0..6usize {
            let lo = u32::from(r.u16());
            let hi = u32::from(r.u16());
            write_u32_le(&mut out, entry + p * 4, lo | (hi << 16));
        }
        let ix_mode = u32::from(r.u8());
        for p in 0..8usize {
            let lo = u32::from(r.u16());
            let hi = u32::from(r.u16());
            write_u32_le(&mut out, entry + 24 + p * 4, lo | (hi << 16));
        }
        write_u32_le(&mut out, entry + 56, ix_mode);
    }

    for i in 0..DICTIONARY {
        let entry = OFF_DICT_BC7 + i * BC7_BYTES;
        let ctrl = u32::from(r.u16());
        for p in 0..12usize {
            let lo = u32::from(r.u16());
            let hi = u32::from(r.u16());
            write_u32_le(&mut out, entry + 4 + p * 4, lo | (hi << 16));
        }
        let bits0 = u32::from(r.u8());
        let bits1 = u32::from(r.u8());
        let _unused = r.u16();
        let mut ix_vals = [0u16; 32];
        for v in &mut ix_vals[..31] {
            *v = r.u16();
        }
        for p in 0..16usize {
            let lo = u32::from(ix_vals[2 * p]);
            let hi = u32::from(ix_vals[2 * p + 1]);
            write_u32_le(&mut out, entry + 52 + p * 4, lo | (hi << 16));
        }
        write_u32_le(&mut out, entry, ctrl | (bits0 << 16) | (bits1 << 24));
    }

    Ok(out)
}

const HLSL_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"

cbuffer Params : register(b0) {
    uint stream_base;
    uint stream_count;
    uint total_blocks;
    uint status_offset;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
};

ByteAddressBuffer tables_buf : register(t0);
ByteAddressBuffer input_buf  : register(t1);
RWByteAddressBuffer work_buf : register(u0);
RWByteAddressBuffer out_buf  : register(u1);

static const uint OFF_FREQ     = 0u;
static const uint OFF_START    = 4608u;
static const uint OFF_BUCKET   = 9728u;
static const uint OFF_RANK1    = 42496u;
static const uint OFF_RANK3A   = 42752u;
static const uint OFF_RANK3C   = 42816u;
static const uint OFF_RANK4    = 43072u;
static const uint OFF_RANK5    = 43136u;
static const uint OFF_CTRL7    = 43200u;
static const uint OFF_DICT_BC1 = 45248u;
static const uint OFF_DICT_BC3 = 55448u;
static const uint OFF_DICT_BC4 = 65648u;
static const uint OFF_DICT_BC5 = 75848u;
static const uint OFF_DICT_BC6 = 86048u;
static const uint OFF_DICT_BC7 = 101348u;

static const uint PLAIN_BYTES = 40u;
static const uint BC6_BYTES   = 60u;
static const uint BC7_BYTES   = 116u;
static const uint DESC_BYTES  = 32u;

uint load_u8_tables(uint byte_off) {
    uint w = tables_buf.Load(byte_off & ~3u);
    return (w >> ((byte_off & 3u) * 8u)) & 0xffu;
}

uint load_u8_input(uint byte_off) {
    uint w = input_buf.Load(byte_off & ~3u);
    return (w >> ((byte_off & 3u) * 8u)) & 0xffu;
}

uint load_u8_work(uint byte_off) {
    uint w = work_buf.Load(byte_off & ~3u);
    return (w >> ((byte_off & 3u) * 8u)) & 0xffu;
}

uint spread4(uint v) {
    return (v & 1u) | ((v & 2u) << 1u) | ((v & 4u) << 2u) | ((v & 8u) << 3u);
}

uint interleave4(uint a, uint b) {
    return spread4(a) | (spread4(b) << 1u);
}

void decode_refs(uint s0, uint s1, uint s2, uint s3, bool paired, out uint ep_ref, out uint ix_ref) {
    uint d;
    if (paired) {
        d = interleave4(s0, s1);
        ix_ref = interleave4(s2, s3);
    } else {
        d = s0;
        ix_ref = s1;
    }
    uint delta = ((d >> 1u) ^ (((d & 1u) != 0u) ? 0xffu : 0u)) & 0xffu;
    ep_ref = (ix_ref + delta) & 0xffu;
}

uint nibble4(uint v) {
    return (v & 1u) | ((v >> 1u) & 2u) | ((v >> 2u) & 4u) | ((v >> 3u) & 8u);
}

int z4(uint u) {
    return int(u >> 1u) ^ -int(u & 1u);
}

int2 nibbles(uint b, bool odd_first) {
    int ev = z4(nibble4(b));
    int od = z4(nibble4(b >> 1u));
    return odd_first ? int2(od, ev) : int2(ev, od);
}

int add_index(int cur, uint bits, int refv, uint ref_bits) {
    uint u_cur = (bits == 0u || bits > 16u) ? 0u : (((uint(cur) & 0xffffu) << (16u - bits)) & 0xffffu);
    uint u_ref = (ref_bits == 0u || ref_bits > 16u) ? 0u : (((uint(refv) & 0xffffu) << (16u - ref_bits)) & 0xffffu);
    uint sum16 = (u_cur + u_ref) & 0xffffu;
    int s16 = int(sum16 << 16u) >> 16;
    return (bits == 0u) ? 0 : (s16 >> int(16u - bits));
}

uint pack_u16_pair(uint lo, uint hi) {
    return (lo & 0xffffu) | ((hi & 0xffffu) << 16u);
}

int unpack_i16_lo(uint w) {
    return int(w << 16u) >> 16;
}

int unpack_i16_hi(uint w) {
    return int(w) >> 16;
}

struct BitWriter {
    uint4 v;
    uint at;
};

void bw_put(inout BitWriter w, uint value, uint n) {
    if (n > 0u && w.at < 128u) {
        uint mask = (n >= 32u) ? 0xffffffffu : ((1u << n) - 1u);
        uint val = value & mask;
        uint word_idx = w.at >> 5u;
        uint bit_off = w.at & 31u;
        if (word_idx == 0u)      w.v.x |= (val << bit_off);
        else if (word_idx == 1u) w.v.y |= (val << bit_off);
        else if (word_idx == 2u) w.v.z |= (val << bit_off);
        else                     w.v.w |= (val << bit_off);
        if (bit_off + n > 32u && word_idx < 3u) {
            uint hi = val >> (32u - bit_off);
            if (word_idx == 0u)      w.v.y |= hi;
            else if (word_idx == 1u) w.v.z |= hi;
            else                     w.v.w |= hi;
        }
    }
    w.at += n;
}

uint find_stream(uint gid) {
    uint lo = 0u;
    uint hi = stream_count;
    [loop]
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        uint blk_start = input_buf.Load((stream_base + mid) * DESC_BYTES + 28u);
        if (blk_start <= gid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return stream_base + lo;
}

static const uint ANCHOR2[64] = {
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
    15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
     6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2, 15
};
static const uint ANCHOR3A[64] = {
     3,  3, 15, 15,  8,  3, 15, 15,  8,  8,  6,  6,  6,  5,  3,  3,
     3,  3,  8, 15,  3,  3,  6, 10,  5,  8,  8,  6,  8,  5, 15, 15,
     8, 15,  3,  5,  6, 10,  8, 15, 15,  3, 15,  5, 15, 15, 15, 15,
     3, 15,  5,  5,  5,  8,  5, 10,  5, 10,  8, 13, 15, 12,  3,  3
};
static const uint ANCHOR3B[64] = {
    15,  8,  8,  3, 15, 15,  3,  8, 15, 15, 15, 15, 15, 15, 15,  8,
    15,  8, 15,  3, 15,  8, 15,  8,  3, 15,  6, 10, 15, 15, 10,  8,
    15,  3, 15, 10, 10,  8,  9, 10,  6, 15,  8, 15,  3,  6,  6,  8,
    15,  3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  3, 15, 15,  8
};

// ============================================================================
// PASS 1: 12/16-lane interleaved rANS decoder (1 threadgroup = 1 stream)
// ============================================================================
#ifdef PASS_LANES

static const uint LOW = 16777216u; // 1 << 24

groupshared uint g_want[16];
groupshared uint g_rec[32];
groupshared uint g_cursor;
groupshared uint g_done;
groupshared uint g_pos;
groupshared uint g_idle;
groupshared uint g_moved;
groupshared uint g_status;

static const uint W1[16] = { 7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7, 0, 0, 0, 0 };
static const uint W3[16] = { 3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7, 0, 0, 0, 0 };
static const uint W4[16] = { 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0 };
static const uint W5[16] = { 3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0 };
static const uint W6[16] = { 3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4 };
static const uint W7[16] = { 3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4 };

uint lane_width_for(uint bc, uint j) {
    if (bc == 1u) return W1[j];
    if (bc == 3u) return W3[j];
    if (bc == 4u) return W4[j];
    if (bc == 5u) return W5[j];
    if (bc == 6u) return W6[j];
    return W7[j];
}

uint bucket_of(uint model, uint slot) {
    uint b = load_u8_tables(OFF_BUCKET + model * 256u + (slot >> 8u));
    [loop]
    while (tables_buf.Load(OFF_START + (model * 10u + b + 1u) * 4u) <= slot) {
        b += 1u;
    }
    return b;
}

[RootSignature(RS)]
[numthreads(16, 1, 1)]
void bcpack_lanes(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint sidx = stream_base + gid.x;
    uint j = gtid.x;
    uint d_off = sidx * DESC_BYTES;

    uint bc         = input_buf.Load(d_off + 0u);
    uint blocks     = input_buf.Load(d_off + 4u);
    uint src_offset = input_buf.Load(d_off + 8u);
    uint src_len    = input_buf.Load(d_off + 12u);
    uint rec_base   = input_buf.Load(d_off + 16u);

    uint num_lanes = (bc >= 6u) ? 16u : 12u;
    uint per = (bc == 1u || bc == 4u) ? 1u : 2u;
    uint stride = num_lanes * per;
    uint words = stride >> 2u;
    uint full = (1u << num_lanes) - 1u;
    uint target_pos = blocks * per;
    bool active = j < num_lanes;

    uint width = active ? lane_width_for(bc, j) : 0u;
    uint phase = 0u;
    bool is_long = false;
    bool do_reset = false;
    uint model = 0u;
    bool need = active;
    bool has_latch = false;
    uint latch = 0u;
    bool has_out = false;
    uint out_sym = 0u;
    bool stalled = false;
    uint lane_x = 0u;
    uint left = 0u;

    if (j == 0u) {
        g_cursor = 0u;
        g_done = 0u;
        g_pos = 0u;
        g_idle = 0u;
        g_moved = 0u;
        g_status = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    [loop]
    while (g_pos < target_pos && g_status == 0u) {
        if (active && !stalled) {
            if (phase != 3u) {
                if (has_latch) {
                    uint b = latch;
                    has_latch = false;
                    need = true;
                    if (phase == 0u) {
                        uint group = width - (b >> 6u);
                        model = group * 16u + (b & 15u);
                        do_reset = (b & 0x10u) != 0u;
                        is_long = (b & 0x20u) != 0u;
                        left = 0u;
                        phase = 1u;
                    } else if (phase == 1u) {
                        if (is_long) {
                            left = b;
                            phase = 2u;
                        } else {
                            left = b + 1u;
                            phase = 3u;
                        }
                    } else {
                        left = ((b << 8u) | left) + 257u;
                        phase = 3u;
                    }
                }
            } else {
                uint x = lane_x;
                if (x < LOW && has_latch) {
                    uint b = latch;
                    has_latch = false;
                    left = left - 1u;
                    x = (x << 8u) | b;
                }
                need = (left != 0u) && !has_latch;
                if (x > LOW) {
                    uint slot = x & 0xffffu;
                    uint b = bucket_of(model, slot);
                    uint sym, f, rem;
                    if (b == 0u) {
                        sym = 0u;
                        f = tables_buf.Load(OFF_FREQ + (model * 9u) * 4u) - 1u;
                        rem = slot - 1u;
                    } else {
                        uint st = tables_buf.Load(OFF_START + (model * 10u + b) * 4u);
                        uint fb = tables_buf.Load(OFF_FREQ + (model * 9u + b) * 4u);
                        uint offset = slot - st;
                        uint q = offset / fb;
                        sym = (1u << (b - 1u)) + q;
                        f = fb;
                        rem = offset - q * fb;
                    }
                    lane_x = (x >> 16u) * f + rem;
                    out_sym = sym & 0xffu;
                    has_out = true;
                } else if (left != 0u) {
                    lane_x = x;
                } else {
                    phase = 0u;
                    need = !has_latch;
                    if (do_reset) {
                        lane_x = 0u;
                    }
                }
            }
        }

        uint bit = 1u << j;
        bool waiting = active && has_out && ((g_done & bit) != 0u);
        g_want[j] = (active && need && !waiting) ? 1u : 0u;
        if (j == 0u) {
            g_moved = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        uint p = 0u;
        [unroll]
        for (uint k = 0u; k < 16u; ++k) {
            if (k < j) p += g_want[k];
        }
        uint total_want = p;
        [unroll]
        for (uint k2 = 0u; k2 < 16u; ++k2) {
            if (k2 >= j) total_want += g_want[k2];
        }

        uint cur = g_cursor;
        uint rem_bytes = (src_len > cur) ? (src_len - cur) : 0u;
        uint avail = min(8u, rem_bytes);
        uint taken = min(total_want, avail);
        uint before = min(p, avail);

        if (g_want[j] != 0u && p < avail) {
            latch = load_u8_input(src_offset + cur + p);
            has_latch = true;
        }
        if (active) {
            stalled = (need && before >= 8u) || waiting;
            if (!stalled && has_out) {
                uint sub = (per == 2u) ? (g_pos & 1u) : 0u;
                g_rec[j * per + sub] = out_sym;
                has_out = false;
                InterlockedOr(g_done, bit);
                InterlockedOr(g_moved, 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (g_done == full) {
            uint next_pos = g_pos + 1u;
            if (per == 1u || (next_pos & 1u) == 0u) {
                uint block_idx = (per == 1u) ? g_pos : (g_pos >> 1u);
                if (j < words) {
                    uint w = g_rec[j * 4u]
                           | (g_rec[j * 4u + 1u] << 8u)
                           | (g_rec[j * 4u + 2u] << 16u)
                           | (g_rec[j * 4u + 3u] << 24u);
                    work_buf.Store(rec_base + block_idx * stride + j * 4u, w);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (j == 0u) {
            g_cursor += taken;
            uint moved = g_moved;
            if (g_done == full) {
                g_pos += 1u;
                g_done = 0u;
                moved = 1u;
            }
            g_idle = (taken > 0u || moved != 0u) ? 0u : (g_idle + 1u);
            if (g_idle > 3u) {
                g_status = 1u;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (j == 0u) {
        out_buf.Store(status_offset + sidx * 4u, g_status);
    }
}
#endif

// ============================================================================
// PASS 2a (Plain: BC1, BC3, BC4, BC5): 256-entry ring buffer + dictionary
// ============================================================================
#ifdef PASS_RESOLVE_PLAIN

groupshared uint g_hist_plain[256 * 10]; // 10 KiB

uint zz8(uint v, uint bits) {
    uint mask = (1u << bits) - 1u;
    return ((v >> 1u) ^ (((v & 1u) != 0u) ? mask : 0u)) & mask;
}

uint load_plain_byte(uint i, uint r, uint dict_off, uint byte_in_plain) {
    if (r == 0u) return 0u;
    if (r <= i) {
        uint slot = (i - r) & 255u;
        uint w = g_hist_plain[slot * 10u + (byte_in_plain >> 2u)];
        return (w >> ((byte_in_plain & 3u) * 8u)) & 0xffu;
    }
    return load_u8_tables(dict_off + (r - i - 1u) * PLAIN_BYTES + byte_in_plain);
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void bcpack_resolve_plain(uint3 gid : SV_GroupID) {
    uint sidx = stream_base + gid.x;
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint bc         = input_buf.Load(d_off + 0u);
    uint blocks     = input_buf.Load(d_off + 4u);
    uint rec_base   = input_buf.Load(d_off + 16u);
    uint inter_base = input_buf.Load(d_off + 20u);

    bool paired = (bc == 3u || bc == 5u);
    uint stride = paired ? 24u : 12u;
    uint dict_off = (bc == 1u) ? OFF_DICT_BC1
                  : (bc == 3u) ? OFF_DICT_BC3
                  : (bc == 4u) ? OFF_DICT_BC4
                               : OFF_DICT_BC5;

    uint ep_bits[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    if (bc == 1u) {
        ep_bits[0] = 5u; ep_bits[1] = 6u; ep_bits[2] = 5u;
        ep_bits[3] = 5u; ep_bits[4] = 6u; ep_bits[5] = 5u;
    } else if (bc == 3u) {
        ep_bits[0] = 5u; ep_bits[1] = 6u; ep_bits[2] = 5u;
        ep_bits[3] = 5u; ep_bits[4] = 6u; ep_bits[5] = 5u;
        ep_bits[6] = 8u; ep_bits[7] = 8u;
    } else if (bc == 4u) {
        ep_bits[0] = 8u; ep_bits[1] = 8u;
    } else {
        ep_bits[0] = 8u; ep_bits[1] = 8u; ep_bits[2] = 8u; ep_bits[3] = 8u;
    }

    uint ix3_n = (bc == 1u) ? 0u : ((bc == 5u) ? 32u : 16u);
    uint ix_total = (bc == 1u || bc == 4u) ? 16u : 32u;

    [loop]
    for (uint i = 0u; i < blocks; ++i) {
        uint r_off = rec_base + i * stride;
        uint s[24];
        uint w_cnt = stride >> 2u;
        [loop]
        for (uint wi = 0u; wi < w_cnt; ++wi) {
            uint w = work_buf.Load(r_off + wi * 4u);
            s[wi * 4u + 0u] = (w >> 0u) & 0xffu;
            s[wi * 4u + 1u] = (w >> 8u) & 0xffu;
            s[wi * 4u + 2u] = (w >> 16u) & 0xffu;
            s[wi * 4u + 3u] = (w >> 24u) & 0xffu;
        }

        uint ep_ref, ix_ref;
        decode_refs(s[0], s[1], s[2], s[3], paired, ep_ref, ix_ref);

        uint ep[8] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
        uint ix[32];
        [unroll]
        for (uint z = 0u; z < 32u; ++z) ix[z] = 0u;

        if (bc == 1u) {
            ep[0] = zz8(s[2], 5u); ep[1] = zz8(s[3], 6u); ep[2] = zz8(s[4], 5u);
            ep[3] = zz8(s[5], 5u); ep[4] = zz8(s[6], 6u); ep[5] = zz8(s[7], 5u);
            [unroll]
            for (uint q = 0u; q < 4u; ++q) {
                uint v = load_u8_tables(OFF_RANK1 + s[8u + q]);
                uint base = (q >> 1u) * 8u + (q & 1u) * 2u;
                ix[base + 0u] = (v >> 0u) & 3u;
                ix[base + 1u] = (v >> 2u) & 3u;
                ix[base + 4u] = (v >> 4u) & 3u;
                ix[base + 5u] = (v >> 6u) & 3u;
            }
        } else if (bc == 3u) {
            ep[0] = zz8(s[14], 5u); ep[1] = zz8(s[16], 6u); ep[2] = zz8(s[18], 5u);
            ep[3] = zz8(s[15], 5u); ep[4] = zz8(s[17], 6u); ep[5] = zz8(s[19], 5u);
            ep[6] = zz8(s[4], 8u);  ep[7] = zz8(s[5], 8u);
            [unroll]
            for (uint k = 0u; k < 8u; ++k) {
                uint v = load_u8_tables(OFF_RANK3A + (s[6u + k] & 63u));
                ix[2u * k + 0u] = v & 7u;
                ix[2u * k + 1u] = (v >> 4u) & 7u;
            }
            [unroll]
            for (uint q = 0u; q < 4u; ++q) {
                uint v = load_u8_tables(OFF_RANK3C + s[20u + q]);
                uint base = 16u + (q >> 1u) * 8u + (q & 1u) * 2u;
                ix[base + 0u] = (v >> 0u) & 3u;
                ix[base + 1u] = (v >> 2u) & 3u;
                ix[base + 4u] = (v >> 4u) & 3u;
                ix[base + 5u] = (v >> 6u) & 3u;
            }
        } else if (bc == 4u) {
            ep[0] = zz8(s[2], 8u); ep[1] = zz8(s[3], 8u);
            [unroll]
            for (uint k = 0u; k < 8u; ++k) {
                uint v = load_u8_tables(OFF_RANK4 + (s[4u + k] & 63u));
                ix[2u * k + 0u] = v & 7u;
                ix[2u * k + 1u] = (v >> 4u) & 7u;
            }
        } else {
            ep[0] = zz8(s[4], 8u); ep[1] = zz8(s[5], 8u);
            ep[2] = zz8(s[6], 8u); ep[3] = zz8(s[7], 8u);
            [unroll]
            for (uint k = 0u; k < 16u; ++k) {
                uint v = load_u8_tables(OFF_RANK5 + (s[8u + k] & 63u));
                ix[2u * k + 0u] = v & 7u;
                ix[2u * k + 1u] = (v >> 4u) & 7u;
            }
        }

        [unroll]
        for (uint e = 0u; e < 8u; ++e) {
            uint b = ep_bits[e];
            uint mask = (b == 0u) ? 0u : ((1u << b) - 1u);
            uint rv = load_plain_byte(i, ep_ref, dict_off, e);
            ep[e] = (ep[e] + rv) & mask;
        }
        [unroll]
        for (uint p = 0u; p < 32u; ++p) {
            uint b = (p < ix3_n) ? 3u : ((p < ix_total) ? 2u : 0u);
            uint mask = (b == 0u) ? 0u : ((1u << b) - 1u);
            uint rv = load_plain_byte(i, ix_ref, dict_off, 8u + p);
            ix[p] = (ix[p] + rv) & mask;
        }

        uint hbase = (i & 255u) * 10u;
        uint obase = inter_base + i * PLAIN_BYTES;
        uint w0 = ep[0] | (ep[1] << 8u) | (ep[2] << 16u) | (ep[3] << 24u);
        uint w1 = ep[4] | (ep[5] << 8u) | (ep[6] << 16u) | (ep[7] << 24u);
        g_hist_plain[hbase + 0u] = w0;
        g_hist_plain[hbase + 1u] = w1;
        work_buf.Store(obase + 0u, w0);
        work_buf.Store(obase + 4u, w1);
        [unroll]
        for (uint iw = 0u; iw < 8u; ++iw) {
            uint w = ix[iw * 4u + 0u]
                   | (ix[iw * 4u + 1u] << 8u)
                   | (ix[iw * 4u + 2u] << 16u)
                   | (ix[iw * 4u + 3u] << 24u);
            g_hist_plain[hbase + 2u + iw] = w;
            work_buf.Store(obase + 8u + iw * 4u, w);
        }
    }
}
#endif

// ============================================================================
// PASS 2a (BC6H): 256-entry ring buffer + dictionary
// ============================================================================
#ifdef PASS_RESOLVE_BC6

groupshared uint g_hist_bc6[256 * 15]; // 15 KiB

static const uint BC6_BITS[15][4] = {
    {  0,  0,  0,  0 },
    { 10,  5,  5,  5 },
    {  7,  6,  6,  6 },
    { 11,  5,  4,  4 },
    { 11,  4,  5,  4 },
    { 11,  4,  4,  5 },
    {  9,  5,  5,  5 },
    {  8,  6,  5,  5 },
    {  8,  5,  6,  5 },
    {  8,  5,  5,  6 },
    {  6,  6,  6,  6 },
    { 10, 10, 10, 10 },
    { 11,  9,  9,  9 },
    { 12,  8,  8,  8 },
    { 16,  4,  4,  4 }
};

uint bc6_index_bits(uint mode) {
    if (mode >= 1u && mode <= 10u) return 3u;
    if (mode >= 11u && mode <= 14u) return 4u;
    return 0u;
}

uint wide8(uint a, uint b) {
    uint acc = 0u;
    [unroll]
    for (uint k = 0u; k < 8u; ++k) {
        acc |= (((a >> k) & 1u) << (2u * k)) | (((b >> k) & 1u) << (2u * k + 1u));
    }
    return acc;
}

uint z16(uint v) {
    return ((v >> 1u) ^ (((v & 1u) != 0u) ? 0xffffu : 0u)) & 0xffffu;
}

uint load_bc6_word(uint i, uint r, uint widx) {
    if (r == 0u) {
        return (widx == 14u) ? 11u : 0u;
    }
    if (r <= i) {
        return g_hist_bc6[((i - r) & 255u) * 15u + widx];
    }
    return tables_buf.Load(OFF_DICT_BC6 + (r - i - 1u) * BC6_BYTES + widx * 4u);
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void bcpack_resolve_bc6(uint3 gid : SV_GroupID) {
    uint sidx = stream_base + gid.x;
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint blocks     = input_buf.Load(d_off + 4u);
    uint rec_base   = input_buf.Load(d_off + 16u);
    uint inter_base = input_buf.Load(d_off + 20u);

    [loop]
    for (uint i = 0u; i < blocks; ++i) {
        uint r_off = rec_base + i * 32u;
        uint s[32];
        [unroll]
        for (uint wi = 0u; wi < 8u; ++wi) {
            uint w = work_buf.Load(r_off + wi * 4u);
            s[wi * 4u + 0u] = (w >> 0u) & 0xffu;
            s[wi * 4u + 1u] = (w >> 8u) & 0xffu;
            s[wi * 4u + 2u] = (w >> 16u) & 0xffu;
            s[wi * 4u + 3u] = (w >> 24u) & 0xffu;
        }

        uint ep_ref, ix_ref;
        decode_refs(s[0], s[1], s[2], s[3], true, ep_ref, ix_ref);

        uint mode = (s[30] + 1u) & 0xffu;
        if (mode < 1u || mode > 14u) {
            out_buf.Store(status_offset + sidx * 4u, 2u);
            return;
        }

        bool two = mode < 11u;
        uint sh0 = 16u - BC6_BITS[mode][0];
        uint sh1 = 16u - BC6_BITS[mode][1];
        uint sh2 = 16u - BC6_BITS[mode][2];
        uint sh3 = 16u - BC6_BITS[mode][3];
        uint shc[3] = { sh1, sh2, sh3 };

        uint res[12];
        res[0] = z16(wide8(s[4], s[5]));
        res[1] = z16(wide8(s[6], s[7]));
        res[2] = z16(wide8(s[8], s[9]));
        res[3] = z16(wide8(s[10], s[11]));
        res[4] = z16(wide8(s[12], s[13]));
        res[5] = z16(wide8(s[14], s[15]));
        if (two) {
            res[6]  = z16(s[16]); res[7]  = z16(s[18]); res[8]  = z16(s[20]);
            res[9]  = z16(s[17]); res[10] = z16(s[19]); res[11] = z16(s[21]);
        } else {
            res[6]  = res[0]; res[7]  = res[1]; res[8]  = res[2];
            res[9]  = res[3]; res[10] = res[4]; res[11] = res[5];
        }

        uint re[12];
        [unroll]
        for (uint p = 0u; p < 6u; ++p) {
            uint rw = load_bc6_word(i, ep_ref, p);
            re[2u * p + 0u] = rw & 0xffffu;
            re[2u * p + 1u] = rw >> 16u;
        }

        uint ep[12];
        uint mask0 = (0xffffu << sh0) & 0xffffu;
        [unroll]
        for (uint c0 = 0u; c0 < 3u; ++c0) {
            ep[c0] = (re[c0] + ((res[c0] << sh0) & 0xffffu)) & mask0;
        }
        uint k_lim = two ? 3u : 1u;
        [loop]
        for (uint k = 0u; k < k_lim; ++k) {
            [unroll]
            for (uint c = 0u; c < 3u; ++c) {
                uint idx = (k + 1u) * 3u + c;
                uint sh = shc[c];
                uint mask = (0xffffu << sh) & 0xffffu;
                ep[idx] = (re[idx] + ((res[idx] << sh) & 0xffffu)) & mask;
            }
        }
        if (!two) {
            [unroll]
            for (uint c2 = 0u; c2 < 3u; ++c2) {
                ep[6u + c2] = ((ep[c2] >> sh0) << sh1) & 0xffffu;
                ep[9u + c2] = ep[3u + c2];
            }
        }

        uint rx_mode = load_bc6_word(i, ix_ref, 14u) & 0xffu;
        uint ref_bits = bc6_index_bits(rx_mode);
        uint bits = bc6_index_bits(mode);
        int ix[16];
        [unroll]
        for (uint k2 = 0u; k2 < 8u; ++k2) {
            uint rw = load_bc6_word(i, ix_ref, 6u + k2);
            int rx0 = unpack_i16_lo(rw);
            int rx1 = unpack_i16_hi(rw);
            int2 pair = nibbles(s[22u + k2], k2 < 6u);
            ix[2u * k2 + 0u] = add_index(pair.x, bits, rx0, ref_bits);
            ix[2u * k2 + 1u] = add_index(pair.y, bits, rx1, ref_bits);
        }

        uint part = s[31] & 31u;
        uint meta = mode | (part << 8u);
        uint hbase = (i & 255u) * 15u;
        uint obase = inter_base + i * BC6_BYTES;

        [unroll]
        for (uint pw = 0u; pw < 6u; ++pw) {
            uint w = pack_u16_pair(ep[2u * pw], ep[2u * pw + 1u]);
            g_hist_bc6[hbase + pw] = w;
            work_buf.Store(obase + pw * 4u, w);
        }
        [unroll]
        for (uint iw = 0u; iw < 8u; ++iw) {
            uint w = pack_u16_pair(uint(ix[2u * iw]), uint(ix[2u * iw + 1u]));
            g_hist_bc6[hbase + 6u + iw] = w;
            work_buf.Store(obase + 24u + iw * 4u, w);
        }
        g_hist_bc6[hbase + 14u] = meta;
        work_buf.Store(obase + 56u, meta);
    }
}
#endif

// ============================================================================
// PASS 2a (BC7): 256-entry ring buffer + dictionary
// ============================================================================
#ifdef PASS_RESOLVE_BC7

groupshared uint g_hist_bc7[256 * 29]; // 29 KiB (< 32 KiB D3D12 limit)

static const uint BC7_BITS[8][4] = {
    { 4, 4, 4, 0 },
    { 6, 6, 6, 0 },
    { 5, 5, 5, 0 },
    { 7, 7, 7, 0 },
    { 5, 5, 5, 6 },
    { 7, 7, 7, 8 },
    { 7, 7, 7, 7 },
    { 5, 5, 5, 5 }
};

static const uint BC7_OFF0[4] = { 22u, 24u, 26u, 28u };
static const uint BC7_OFF1[4] = { 12u, 14u, 16u, 20u };

static const uint BC7_INDEX[8][2] = {
    { 3, 0 },
    { 3, 0 },
    { 2, 0 },
    { 2, 0 },
    { 3, 2 },
    { 2, 2 },
    { 4, 0 },
    { 2, 0 }
};

bool has_pbits(uint mode) {
    return mode == 0u || mode == 1u || mode == 3u || mode == 6u || mode == 7u;
}

uint bc7_rotation(uint ctrl) {
    uint m = ctrl & 7u;
    return (m == 4u || m == 5u) ? ((ctrl >> 3u) & 3u) : 0u;
}

uint4 bc7_precision(uint ctrl) {
    uint m = ctrl & 7u;
    uint rot = bc7_rotation(ctrl);
    uint p[4] = { BC7_BITS[m][0], BC7_BITS[m][1], BC7_BITS[m][2], BC7_BITS[m][3] };
    if (rot != 0u) {
        uint tmp = p[3];
        p[3] = p[rot - 1u];
        p[rot - 1u] = tmp;
    } else if (has_pbits(m)) {
        p[0] += 1u;
    }
    return uint4(p[0], p[1], p[2], p[3]);
}

uint load_bc7_word(uint i, uint r, uint widx) {
    if (r == 0u) return 0u;
    if (r <= i) {
        return g_hist_bc7[((i - r) & 255u) * 29u + widx];
    }
    return tables_buf.Load(OFF_DICT_BC7 + (r - i - 1u) * BC7_BYTES + widx * 4u);
}

[RootSignature(RS)]
[numthreads(1, 1, 1)]
void bcpack_resolve_bc7(uint3 gid : SV_GroupID) {
    uint sidx = stream_base + gid.x;
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint blocks     = input_buf.Load(d_off + 4u);
    uint rec_base   = input_buf.Load(d_off + 16u);
    uint inter_base = input_buf.Load(d_off + 20u);

    [loop]
    for (uint i = 0u; i < blocks; ++i) {
        uint r_off = rec_base + i * 32u;
        uint s[32];
        [unroll]
        for (uint wi = 0u; wi < 8u; ++wi) {
            uint w = work_buf.Load(r_off + wi * 4u);
            s[wi * 4u + 0u] = (w >> 0u) & 0xffu;
            s[wi * 4u + 1u] = (w >> 8u) & 0xffu;
            s[wi * 4u + 2u] = (w >> 16u) & 0xffu;
            s[wi * 4u + 3u] = (w >> 24u) & 0xffu;
        }

        uint ep_ref, ix_ref;
        decode_refs(s[0], s[1], s[2], s[3], true, ep_ref, ix_ref);

        uint ctrl_idx = ((s[31] & 15u) << 5u) | (s[30] & 31u);
        uint ctrl = tables_buf.Load(OFF_CTRL7 + ctrl_idx * 4u);
        if (ctrl == 0u) {
            out_buf.Store(status_offset + sidx * 4u, 3u);
            return;
        }

        uint mode = ctrl & 7u;
        uint subsets = (ctrl & 0xffu) >> 6u;
        uint4 prec4 = bc7_precision(ctrl);
        uint prec[4] = { prec4.x, prec4.y, prec4.z, prec4.w };
        bool three = (subsets == 3u);

        uint stored[24];
        stored[0]  = s[4];  stored[1]  = s[5];  stored[2]  = s[6];  stored[3]  = s[7];
        stored[4]  = s[8];  stored[5]  = s[9];  stored[6]  = three ? 0u : s[10]; stored[7]  = three ? 0u : s[11];
        stored[8]  = s[12]; stored[9]  = s[13]; stored[10] = s[14]; stored[11] = s[15];
        stored[12] = s[16]; stored[13] = s[17]; stored[14] = three ? 0u : s[18]; stored[15] = three ? 0u : s[19];
        stored[16] = s[20]; stored[17] = s[21];
        stored[18] = three ? s[10] : 0u; stored[19] = three ? s[11] : 0u;
        stored[20] = three ? s[18] : 0u; stored[21] = three ? s[19] : 0u;
        stored[22] = 0u; stored[23] = 0u;

        uint r_meta = load_bc7_word(i, ep_ref, 0u);
        uint r_ctrl = r_meta & 0xffffu;
        uint r_rows[24];
        [unroll]
        for (uint rw_i = 0u; rw_i < 12u; ++rw_i) {
            uint w = load_bc7_word(i, ep_ref, 1u + rw_i);
            r_rows[2u * rw_i + 0u] = w & 0xffffu;
            r_rows[2u * rw_i + 1u] = w >> 16u;
        }

        uint ep_rows[24];
        [unroll]
        for (uint z = 0u; z < 24u; ++z) ep_rows[z] = 0u;

        uint common = min(subsets, (r_ctrl & 0xffu) >> 6u);
        uint columns = (prec[3] > 0u) ? 8u : 6u;
        [loop]
        for (uint c = 0u; c < columns; ++c) {
            uint p = prec[c >> 1u];
            if (p == 0u) continue;
            uint mask = (0xffffu << (16u - p)) & 0xffffu;
            [loop]
            for (uint row = 0u; row < subsets; ++row) {
                uint v = stored[row * 8u + c];
                uint residual = (((v >> 1u) ^ (((v & 1u) != 0u) ? 0xffffu : 0u)) << (16u - p)) & 0xffffu;
                uint base_row = (row < common) ? row : 0u;
                uint base_val = r_rows[base_row * 8u + c] & mask;
                ep_rows[row * 8u + c] = (base_val + residual) & 0xffffu;
            }
        }

        uint bits0 = BC7_INDEX[mode][0];
        uint bits1 = BC7_INDEX[mode][1];
        uint rx_meta = load_bc7_word(i, ix_ref, 0u);
        uint rx_bits0 = (rx_meta >> 16u) & 0xffu;
        uint rx_bits1 = (rx_meta >> 24u) & 0xffu;

        int ix0[16];
        int ix1[16];
        [unroll]
        for (uint zi = 0u; zi < 16u; ++zi) { ix0[zi] = 0; ix1[zi] = 0; }

        if (bits0 != 0u) {
            [unroll]
            for (uint k = 0u; k < 4u; ++k) {
                uint o = BC7_OFF0[k];
                [unroll]
                for (uint b = 0u; b < 2u; ++b) {
                    uint rw = load_bc7_word(i, ix_ref, 13u + 2u * k + b);
                    int2 pair = nibbles(s[o + b], k < 3u);
                    uint p0 = 4u * k + 2u * b;
                    ix0[p0 + 0u] = add_index(pair.x, bits0, unpack_i16_lo(rw), rx_bits0);
                    ix0[p0 + 1u] = add_index(pair.y, bits0, unpack_i16_hi(rw), rx_bits0);
                }
            }
        }
        if (bits1 != 0u) {
            [unroll]
            for (uint k = 0u; k < 4u; ++k) {
                uint o = BC7_OFF1[k];
                [unroll]
                for (uint b = 0u; b < 2u; ++b) {
                    uint rw = load_bc7_word(i, ix_ref, 21u + 2u * k + b);
                    int2 pair = nibbles(s[o + b], k < 3u);
                    uint p0 = 4u * k + 2u * b;
                    ix1[p0 + 0u] = add_index(pair.x, bits1, unpack_i16_lo(rw), rx_bits1);
                    ix1[p0 + 1u] = add_index(pair.y, bits1, unpack_i16_hi(rw), rx_bits1);
                }
            }
        }

        uint meta = ctrl | (bits0 << 16u) | (bits1 << 24u);
        uint hbase = (i & 255u) * 29u;
        uint obase = inter_base + i * BC7_BYTES;

        g_hist_bc7[hbase + 0u] = meta;
        work_buf.Store(obase + 0u, meta);
        [unroll]
        for (uint pw = 0u; pw < 12u; ++pw) {
            uint w = pack_u16_pair(ep_rows[2u * pw], ep_rows[2u * pw + 1u]);
            g_hist_bc7[hbase + 1u + pw] = w;
            work_buf.Store(obase + 4u + pw * 4u, w);
        }
        [unroll]
        for (uint iw0 = 0u; iw0 < 8u; ++iw0) {
            uint w = pack_u16_pair(uint(ix0[2u * iw0]), uint(ix0[2u * iw0 + 1u]));
            g_hist_bc7[hbase + 13u + iw0] = w;
            work_buf.Store(obase + 52u + iw0 * 4u, w);
        }
        [unroll]
        for (uint iw1 = 0u; iw1 < 8u; ++iw1) {
            uint w = pack_u16_pair(uint(ix1[2u * iw1]), uint(ix1[2u * iw1 + 1u]));
            g_hist_bc7[hbase + 21u + iw1] = w;
            work_buf.Store(obase + 84u + iw1 * 4u, w);
        }
    }
}
#endif

// ============================================================================
// PASS 2b (Plain: BC1, BC3, BC4, BC5): 100% block-parallel packing
// ============================================================================
#ifdef PASS_PACK_PLAIN

static const uint BC1_CODE[4] = { 0u, 2u, 3u, 1u };
static const uint BC4_CODE[8] = { 0u, 2u, 3u, 4u, 5u, 6u, 7u, 1u };

uint2 pack_bc1_color(uint ep0, uint ep1, uint ep2, uint ep3, uint ep4, uint ep5, uint ix[16]) {
    uint c0 = (ep0 << 11u) | (ep1 << 5u) | ep2;
    uint c1 = (ep3 << 11u) | (ep4 << 5u) | ep5;
    uint bits = 0u;
    [unroll]
    for (uint p = 0u; p < 16u; ++p) {
        bits |= BC1_CODE[ix[p] & 3u] << (2u * p);
    }
    return uint2(c0 | (c1 << 16u), bits);
}

uint2 pack_bc4_alpha(uint e0, uint e1, uint ix[16]) {
    BitWriter w;
    w.v = uint4(0u, 0u, 0u, 0u);
    w.at = 0u;
    bw_put(w, e0 ^ 0x80u, 8u);
    bw_put(w, e1 ^ 0x80u, 8u);
    [unroll]
    for (uint p = 0u; p < 16u; ++p) {
        bw_put(w, BC4_CODE[ix[p] & 7u], 3u);
    }
    return w.v.xy;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void bcpack_pack_plain(uint3 dtid : SV_DispatchThreadID) {
    uint gid = dtid.x;
    if (gid >= total_blocks) return;

    uint sidx = find_stream(gid);
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint bc         = input_buf.Load(d_off + 0u);
    uint inter_base = input_buf.Load(d_off + 20u);
    uint out_base   = input_buf.Load(d_off + 24u);
    uint blk_start  = input_buf.Load(d_off + 28u);
    uint local_blk  = gid - blk_start;

    uint ibase = inter_base + local_blk * PLAIN_BYTES;
    uint w0 = work_buf.Load(ibase + 0u);
    uint w1 = work_buf.Load(ibase + 4u);
    uint ep[8];
    ep[0] = (w0 >> 0u) & 0xffu; ep[1] = (w0 >> 8u) & 0xffu; ep[2] = (w0 >> 16u) & 0xffu; ep[3] = (w0 >> 24u) & 0xffu;
    ep[4] = (w1 >> 0u) & 0xffu; ep[5] = (w1 >> 8u) & 0xffu; ep[6] = (w1 >> 16u) & 0xffu; ep[7] = (w1 >> 24u) & 0xffu;

    uint ix_lo[16];
    [unroll]
    for (uint k0 = 0u; k0 < 4u; ++k0) {
        uint w = work_buf.Load(ibase + 8u + k0 * 4u);
        ix_lo[k0 * 4u + 0u] = (w >> 0u) & 0xffu;
        ix_lo[k0 * 4u + 1u] = (w >> 8u) & 0xffu;
        ix_lo[k0 * 4u + 2u] = (w >> 16u) & 0xffu;
        ix_lo[k0 * 4u + 3u] = (w >> 24u) & 0xffu;
    }

    if (bc == 1u) {
        out_buf.Store2(out_base + local_blk * 8u, pack_bc1_color(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], ix_lo));
    } else if (bc == 4u) {
        out_buf.Store2(out_base + local_blk * 8u, pack_bc4_alpha(ep[0], ep[1], ix_lo));
    } else {
        uint ix_hi[16];
        [unroll]
        for (uint k1 = 0u; k1 < 4u; ++k1) {
            uint w = work_buf.Load(ibase + 24u + k1 * 4u);
            ix_hi[k1 * 4u + 0u] = (w >> 0u) & 0xffu;
            ix_hi[k1 * 4u + 1u] = (w >> 8u) & 0xffu;
            ix_hi[k1 * 4u + 2u] = (w >> 16u) & 0xffu;
            ix_hi[k1 * 4u + 3u] = (w >> 24u) & 0xffu;
        }
        if (bc == 3u) {
            uint2 a = pack_bc4_alpha(ep[6], ep[7], ix_lo);
            uint2 c = pack_bc1_color(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], ix_hi);
            out_buf.Store4(out_base + local_blk * 16u, uint4(a.x, a.y, c.x, c.y));
        } else {
            uint2 a0 = pack_bc4_alpha(ep[0], ep[1], ix_lo);
            uint2 a1 = pack_bc4_alpha(ep[2], ep[3], ix_hi);
            out_buf.Store4(out_base + local_blk * 16u, uint4(a0.x, a0.y, a1.x, a1.y));
        }
    }
}
#endif

// ============================================================================
// PASS 2b (BC6H): 100% block-parallel packing (all modes 1..=14)
// ============================================================================
#ifdef PASS_PACK_BC6

static const uint BC6_BITS[15][4] = {
    {  0,  0,  0,  0 },
    { 10,  5,  5,  5 },
    {  7,  6,  6,  6 },
    { 11,  5,  4,  4 },
    { 11,  4,  5,  4 },
    { 11,  4,  4,  5 },
    {  9,  5,  5,  5 },
    {  8,  6,  5,  5 },
    {  8,  5,  6,  5 },
    {  8,  5,  5,  6 },
    {  6,  6,  6,  6 },
    { 10, 10, 10, 10 },
    { 11,  9,  9,  9 },
    { 12,  8,  8,  8 },
    { 16,  4,  4,  4 }
};
static const uint BC6_CODE_VAL[15] = { 0, 0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 3, 7, 11, 15 };
static const uint BC6_CODE_LEN[15] = { 0, 2, 2, 5, 5,  5,  5,  5,  5,  5,  5, 5, 5,  5,  5 };

#define R 0u
#define G 1u
#define B 2u
#define RUN(ch, ep_i, first, len) bw_put(w, f[(ep_i) * 3u + (ch)] >> (first), (len))
#define D bw_put(w, part, 5u)

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void bcpack_pack_bc6(uint3 dtid : SV_DispatchThreadID) {
    uint gid = dtid.x;
    if (gid >= total_blocks) return;

    uint sidx = find_stream(gid);
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint inter_base = input_buf.Load(d_off + 20u);
    uint out_base   = input_buf.Load(d_off + 24u);
    uint blk_start  = input_buf.Load(d_off + 28u);
    uint local_blk  = gid - blk_start;

    uint ibase = inter_base + local_blk * BC6_BYTES;
    uint meta = work_buf.Load(ibase + 56u);
    uint mode = meta & 0xffu;
    uint part = (meta >> 8u) & 31u;
    bool two = mode < 11u;

    uint sh0 = 16u - BC6_BITS[mode][0];
    uint shc[3] = {
        16u - BC6_BITS[mode][1],
        16u - BC6_BITS[mode][2],
        16u - BC6_BITS[mode][3]
    };

    uint ep[12];
    [unroll]
    for (uint pw = 0u; pw < 6u; ++pw) {
        uint wv = work_buf.Load(ibase + pw * 4u);
        ep[2u * pw + 0u] = wv & 0xffffu;
        ep[2u * pw + 1u] = wv >> 16u;
    }

    uint f[12];
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        [unroll]
        for (uint c = 0u; c < 3u; ++c) {
            uint sh = (k == 0u) ? sh0 : shc[c];
            f[k * 3u + c] = (ep[k * 3u + c] ^ 0x8000u) >> sh;
        }
    }

    BitWriter w;
    w.v = uint4(0u, 0u, 0u, 0u);
    w.at = 0u;
    bw_put(w, BC6_CODE_VAL[mode], BC6_CODE_LEN[mode]);

    switch (mode) {
        case 1u:
            RUN(G,2,4,1); RUN(B,2,4,1); RUN(B,3,4,1); RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10);
            RUN(R,1,0,5); RUN(G,3,4,1); RUN(G,2,0,4); RUN(G,1,0,5); RUN(B,3,0,1); RUN(G,3,0,4);
            RUN(B,1,0,5); RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,5); RUN(B,3,2,1); RUN(R,3,0,5);
            RUN(B,3,3,1); D;
            break;
        case 2u:
            RUN(G,2,5,1); RUN(G,3,4,1); RUN(G,3,5,1); RUN(R,0,0,7); RUN(B,3,0,1); RUN(B,3,1,1);
            RUN(B,2,4,1); RUN(G,0,0,7); RUN(B,2,5,1); RUN(B,3,2,1); RUN(G,2,4,1); RUN(B,0,0,7);
            RUN(B,3,3,1); RUN(B,3,5,1); RUN(B,3,4,1); RUN(R,1,0,6); RUN(G,2,0,4); RUN(G,1,0,6);
            RUN(G,3,0,4); RUN(B,1,0,6); RUN(B,2,0,4); RUN(R,2,0,6); RUN(R,3,0,6); D;
            break;
        case 3u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,5); RUN(R,0,10,1); RUN(G,2,0,4);
            RUN(G,1,0,4); RUN(G,0,10,1); RUN(B,3,0,1); RUN(G,3,0,4); RUN(B,1,0,4); RUN(B,0,10,1);
            RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,5); RUN(B,3,2,1); RUN(R,3,0,5); RUN(B,3,3,1); D;
            break;
        case 4u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,4); RUN(R,0,10,1); RUN(G,3,4,1);
            RUN(G,2,0,4); RUN(G,1,0,5); RUN(G,0,10,1); RUN(G,3,0,4); RUN(B,1,0,4); RUN(B,0,10,1);
            RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,4); RUN(B,3,0,1); RUN(B,3,2,1); RUN(R,3,0,4);
            RUN(G,2,4,1); RUN(B,3,3,1); D;
            break;
        case 5u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,4); RUN(R,0,10,1); RUN(B,2,4,1);
            RUN(G,2,0,4); RUN(G,1,0,4); RUN(G,0,10,1); RUN(B,3,0,1); RUN(G,3,0,4); RUN(B,1,0,5);
            RUN(B,0,10,1); RUN(B,2,0,4); RUN(R,2,0,4); RUN(B,3,1,1); RUN(B,3,2,1); RUN(R,3,0,4);
            RUN(B,3,4,1); RUN(B,3,3,1); D;
            break;
        case 6u:
            RUN(R,0,0,9); RUN(B,2,4,1); RUN(G,0,0,9); RUN(G,2,4,1); RUN(B,0,0,9); RUN(B,3,4,1);
            RUN(R,1,0,5); RUN(G,3,4,1); RUN(G,2,0,4); RUN(G,1,0,5); RUN(B,3,0,1); RUN(G,3,0,4);
            RUN(B,1,0,5); RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,5); RUN(B,3,2,1); RUN(R,3,0,5);
            RUN(B,3,3,1); D;
            break;
        case 7u:
            RUN(R,0,0,8); RUN(G,3,4,1); RUN(B,2,4,1); RUN(G,0,0,8); RUN(B,3,2,1); RUN(G,2,4,1);
            RUN(B,0,0,8); RUN(B,3,3,1); RUN(B,3,4,1); RUN(R,1,0,6); RUN(G,2,0,4); RUN(G,1,0,5);
            RUN(B,3,0,1); RUN(G,3,0,4); RUN(B,1,0,5); RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,6);
            RUN(R,3,0,6); D;
            break;
        case 8u:
            RUN(R,0,0,8); RUN(B,3,0,1); RUN(B,2,4,1); RUN(G,0,0,8); RUN(G,2,5,1); RUN(G,2,4,1);
            RUN(B,0,0,8); RUN(G,3,5,1); RUN(B,3,4,1); RUN(R,1,0,5); RUN(G,3,4,1); RUN(G,2,0,4);
            RUN(G,1,0,6); RUN(G,3,0,4); RUN(B,1,0,5); RUN(B,3,1,1); RUN(B,2,0,4); RUN(R,2,0,5);
            RUN(B,3,2,1); RUN(R,3,0,5); RUN(B,3,3,1); D;
            break;
        case 9u:
            RUN(R,0,0,8); RUN(B,3,1,1); RUN(B,2,4,1); RUN(G,0,0,8); RUN(B,2,5,1); RUN(G,2,4,1);
            RUN(B,0,0,8); RUN(B,3,5,1); RUN(B,3,4,1); RUN(R,1,0,5); RUN(G,3,4,1); RUN(G,2,0,4);
            RUN(G,1,0,5); RUN(B,3,0,1); RUN(G,3,0,4); RUN(B,1,0,6); RUN(B,2,0,4); RUN(R,2,0,5);
            RUN(B,3,2,1); RUN(R,3,0,5); RUN(B,3,3,1); D;
            break;
        case 10u:
            RUN(R,0,0,6); RUN(G,3,4,1); RUN(B,3,0,1); RUN(B,3,1,1); RUN(B,2,4,1); RUN(G,0,0,6);
            RUN(G,2,5,1); RUN(B,2,5,1); RUN(B,3,2,1); RUN(G,2,4,1); RUN(B,0,0,6); RUN(G,3,5,1);
            RUN(B,3,3,1); RUN(B,3,5,1); RUN(B,3,4,1); RUN(R,1,0,6); RUN(G,2,0,4); RUN(G,1,0,6);
            RUN(G,3,0,4); RUN(B,1,0,6); RUN(B,2,0,4); RUN(R,2,0,6); RUN(R,3,0,6); D;
            break;
        case 11u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,10); RUN(G,1,0,10); RUN(B,1,0,10);
            break;
        case 12u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,9); RUN(R,0,10,1); RUN(G,1,0,9);
            RUN(G,0,10,1); RUN(B,1,0,9); RUN(B,0,10,1);
            break;
        case 13u:
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,8); RUN(R,0,10,2); RUN(G,1,0,8);
            RUN(G,0,10,2); RUN(B,1,0,8); RUN(B,0,10,2);
            break;
        default: // 14
            RUN(R,0,0,10); RUN(G,0,0,10); RUN(B,0,0,10); RUN(R,1,0,4); RUN(R,0,10,6); RUN(G,1,0,4);
            RUN(G,0,10,6); RUN(B,1,0,4); RUN(B,0,10,6);
            break;
    }

    uint bits = two ? 3u : 4u;
    uint anchor = two ? ANCHOR2[part] : 0u;
    uint bias = 1u << (bits - 1u);
    [unroll]
    for (uint iw = 0u; iw < 8u; ++iw) {
        uint wv = work_buf.Load(ibase + 24u + iw * 4u);
        uint p0 = 2u * iw;
        uint p1 = p0 + 1u;
        uint v0 = ((wv & 0xffffu) + bias) & 0xffffu;
        uint v1 = ((wv >> 16u) + bias) & 0xffffu;
        uint n0 = bits - ((p0 == 0u || p0 == anchor) ? 1u : 0u);
        uint n1 = bits - ((p1 == 0u || p1 == anchor) ? 1u : 0u);
        bw_put(w, v0, n0);
        bw_put(w, v1, n1);
    }

    out_buf.Store4(out_base + local_blk * 16u, w.v);
}
#endif

// ============================================================================
// PASS 2b (BC7): 100% block-parallel packing (modes 0..=7, rotations, p-bits)
// ============================================================================
#ifdef PASS_PACK_BC7

static const uint BC7_BITS[8][4] = {
    { 4, 4, 4, 0 },
    { 6, 6, 6, 0 },
    { 5, 5, 5, 0 },
    { 7, 7, 7, 0 },
    { 5, 5, 5, 6 },
    { 7, 7, 7, 8 },
    { 7, 7, 7, 7 },
    { 5, 5, 5, 5 }
};

bool has_pbits(uint mode) {
    return mode == 0u || mode == 1u || mode == 3u || mode == 6u || mode == 7u;
}

uint bc7_rotation(uint ctrl) {
    uint m = ctrl & 7u;
    return (m == 4u || m == 5u) ? ((ctrl >> 3u) & 3u) : 0u;
}

uint4 bc7_precision(uint ctrl) {
    uint m = ctrl & 7u;
    uint rot = bc7_rotation(ctrl);
    uint p[4] = { BC7_BITS[m][0], BC7_BITS[m][1], BC7_BITS[m][2], BC7_BITS[m][3] };
    if (rot != 0u) {
        uint tmp = p[3];
        p[3] = p[rot - 1u];
        p[rot - 1u] = tmp;
    } else if (has_pbits(m)) {
        p[0] += 1u;
    }
    return uint4(p[0], p[1], p[2], p[3]);
}

void emit_bc7_indices(inout BitWriter w, uint ibase_words, uint bits, uint a1, uint a2) {
    [unroll]
    for (uint iw = 0u; iw < 8u; ++iw) {
        uint wv = work_buf.Load(ibase_words + iw * 4u);
        uint p0 = 2u * iw;
        uint p1 = p0 + 1u;
        uint n0 = bits - ((p0 == 0u || p0 == a1 || p0 == a2) ? 1u : 0u);
        uint n1 = bits - ((p1 == 0u || p1 == a1 || p1 == a2) ? 1u : 0u);
        bw_put(w, wv & 0xffffu, n0);
        bw_put(w, wv >> 16u, n1);
    }
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void bcpack_pack_bc7(uint3 dtid : SV_DispatchThreadID) {
    uint gid = dtid.x;
    if (gid >= total_blocks) return;

    uint sidx = find_stream(gid);
    if (out_buf.Load(status_offset + sidx * 4u) != 0u) return;

    uint d_off = sidx * DESC_BYTES;
    uint inter_base = input_buf.Load(d_off + 20u);
    uint out_base   = input_buf.Load(d_off + 24u);
    uint blk_start  = input_buf.Load(d_off + 28u);
    uint local_blk  = gid - blk_start;

    uint ibase = inter_base + local_blk * BC7_BYTES;
    uint meta = work_buf.Load(ibase + 0u);
    uint ctrl = meta & 0xffffu;
    uint bits0 = (meta >> 16u) & 0xffu;
    uint bits1 = (meta >> 24u) & 0xffu;

    uint mode = ctrl & 7u;
    uint subsets = (ctrl & 0xffu) >> 6u;
    uint part = (ctrl >> 8u) & 63u;
    uint rot = bc7_rotation(ctrl);
    uint4 prec4 = bc7_precision(ctrl);
    uint prec[4] = { prec4.x, prec4.y, prec4.z, prec4.w };

    uint rows[24];
    [unroll]
    for (uint pw = 0u; pw < 12u; ++pw) {
        uint wv = work_buf.Load(ibase + 4u + pw * 4u);
        rows[2u * pw + 0u] = wv & 0xffffu;
        rows[2u * pw + 1u] = wv >> 16u;
    }

    BitWriter w;
    w.v = uint4(0u, 0u, 0u, 0u);
    w.at = 0u;
    bw_put(w, 1u << mode, mode + 1u);

    if (mode == 0u) {
        bw_put(w, part, 4u);
    } else if (mode == 1u || mode == 2u || mode == 3u || mode == 7u) {
        bw_put(w, part, 6u);
    } else if (mode == 4u) {
        bw_put(w, rot, 2u);
        bw_put(w, (ctrl >> 5u) & 1u, 1u);
    } else if (mode == 5u) {
        bw_put(w, rot, 2u);
    }

    uint ends = 2u * subsets;
    uint order[4] = { 0u, 1u, 2u, 3u };
    if (rot != 0u) {
        uint tmp = order[3];
        order[3] = order[rot - 1u];
        order[rot - 1u] = tmp;
    }

    [unroll]
    for (uint oi = 0u; oi < 4u; ++oi) {
        uint ch = order[oi];
        uint p = prec[ch];
        if (p == 0u) continue;
        uint drop = (ch == 0u && has_pbits(mode)) ? 1u : 0u;
        [loop]
        for (uint e = 0u; e < ends; ++e) {
            uint row = e >> 1u;
            uint col = 2u * ch + (e & 1u);
            uint val = rows[row * 8u + col] >> (16u - p);
            bw_put(w, val >> drop, p - drop);
        }
    }

    if (mode == 1u) {
        [loop]
        for (uint row = 0u; row < subsets; ++row) {
            uint val = rows[row * 8u] >> (16u - prec[0]);
            bw_put(w, val & 1u, 1u);
        }
    } else if (has_pbits(mode)) {
        [loop]
        for (uint e = 0u; e < ends; ++e) {
            uint val = rows[(e >> 1u) * 8u + (e & 1u)] >> (16u - prec[0]);
            bw_put(w, val & 1u, 1u);
        }
    }

    uint a1 = (subsets == 2u) ? ANCHOR2[part] : ((subsets == 3u) ? ANCHOR3A[part] : 0u);
    uint a2 = (subsets == 3u) ? ANCHOR3B[part] : 0u;

    if (mode == 4u) {
        emit_bc7_indices(w, ibase + 84u, bits1, a1, a2);
        emit_bc7_indices(w, ibase + 52u, bits0, a1, a2);
    } else if (mode == 5u) {
        emit_bc7_indices(w, ibase + 52u, bits0, a1, a2);
        emit_bc7_indices(w, ibase + 84u, bits1, a1, a2);
    } else {
        emit_bc7_indices(w, ibase + 52u, bits0, a1, a2);
    }

    out_buf.Store4(out_base + local_blk * 16u, w.v);
}
#endif
"#;

// ============================================================================
// D3D12 Compute Runtime (Double-Buffered Async Slots + Scoped Barriers + Coalescing)
// ============================================================================

/// Number of overlapping GPU flight slots so consecutive batches pipeline on
/// the compute queue instead of stalling per stream.
const FLIGHT_SLOTS: usize = 2;
/// Maximum streams coalesced into one GPU batch from concurrent callers.
const MAX_COALESCED_STREAMS: usize = 64;
/// Maximum total compressed input bytes coalesced into one GPU batch.
const MAX_COALESCED_INPUT_BYTES: usize = 16 << 20;
/// Maximum retained per-slot scratch capacity (outlier batches release on fence).
const MAX_RETAINED_SCRATCH: u64 = 16 << 20;

/// Telemetry counters: `[0]` GPU batches, `[1]` total streams decoded,
/// `[2]` coalesced multi-stream batches, `[3]` total output bytes decoded.
static BCPACK_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn enhanced_barriers_opt_in() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VICTORY_GPU_BCPACK_ENHANCED")
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

/// Barrier mode selected for a `FlightSlot` command list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BarrierKind {
    Enhanced,
    LegacyScoped,
}

fn choose_barrier_kind(device_enhanced: bool, has_list7: bool) -> BarrierKind {
    if device_enhanced && has_list7 {
        BarrierKind::Enhanced
    } else {
        BarrierKind::LegacyScoped
    }
}

struct ScratchBuffer {
    res: Option<ID3D12Resource>,
    capacity: u64,
}

impl ScratchBuffer {
    fn new() -> Self {
        Self { res: None, capacity: 0 }
    }

    fn ensure(
        &mut self,
        device: &ID3D12Device,
        needed: u64,
        heap: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<&ID3D12Resource, String> {
        let aligned = needed.max(256).next_multiple_of(256);
        if self.capacity < aligned || self.res.is_none() {
            let cap = aligned.max(self.capacity.saturating_mul(2));
            self.res = Some(create_buffer(device, cap, heap, flags, state)?);
            self.capacity = cap;
        }
        self.res.as_ref().ok_or_else(|| "missing scratch buffer".to_string())
    }

    fn trim_oversized(&mut self) {
        if self.capacity > MAX_RETAINED_SCRATCH {
            self.res = None;
            self.capacity = 0;
        }
    }
}

struct CachedVramTables {
    hash: blake3::Hash,
    resource: ID3D12Resource,
}

struct TablesUploader {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    event: HANDLE,
    cache: Option<CachedVramTables>,
}

// SAFETY: D3D12 objects are free-threaded; event & allocator are serialized behind `tables_lock`.
unsafe impl Send for TablesUploader {}

impl Drop for TablesUploader {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: owned Win32 event created by `CreateEventW`.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

struct FlightSlot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    list7: Option<ID3D12GraphicsCommandList7>,
    event: HANDLE,
    fence_val: u64,
    input_upload: ScratchBuffer,
    input_default: ScratchBuffer,
    work_uav: ScratchBuffer,
    out_uav: ScratchBuffer,
    out_readback: ScratchBuffer,
}

// SAFETY: D3D12 objects are free-threaded; each slot is leased exclusively to one batch at a time.
unsafe impl Send for FlightSlot {}

impl Drop for FlightSlot {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            // SAFETY: owned Win32 event created by `CreateEventW`.
            unsafe {
                let _ = CloseHandle(self.event);
            }
        }
    }
}

impl FlightSlot {
    fn new(device: &ID3D12Device, enhanced: bool) -> Result<Self, String> {
        // SAFETY: plain D3D12 compute allocator, list, and Win32 event creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(|e| format!("CreateCommandAllocator failed: {e}"))?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)
                .map_err(|e| format!("CreateCommandList failed: {e}"))?;
            list.Close().map_err(|e| format!("initial list Close failed: {e}"))?;
            let list7 = enhanced.then(|| list.cast::<ID3D12GraphicsCommandList7>().ok()).flatten();
            let event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW failed: {e}"))?;
            Ok(Self {
                allocator,
                list,
                list7,
                event,
                fence_val: 0,
                input_upload: ScratchBuffer::new(),
                input_default: ScratchBuffer::new(),
                work_uav: ScratchBuffer::new(),
                out_uav: ScratchBuffer::new(),
                out_readback: ScratchBuffer::new(),
            })
        }
    }

    fn trim_oversized(&mut self) {
        self.input_upload.trim_oversized();
        self.input_default.trim_oversized();
        self.work_uav.trim_oversized();
        self.out_uav.trim_oversized();
        self.out_readback.trim_oversized();
    }
}

struct PreparedBatch {
    n_streams: usize,
    n_plain: u32,
    n_bc6: u32,
    n_bc7: u32,
    plain_blocks: u32,
    bc6_blocks: u32,
    bc7_blocks: u32,
    input_host: Vec<u8>,
    work_total: u64,
    status_offset: u32,
    out_total: u64,
    orig_out_ranges: Vec<(usize, usize)>,
}

struct CoalescedItem {
    bc: u8,
    src: Vec<u8>,
    blocks: usize,
    tables_hash: blake3::Hash,
    result: Mutex<Option<Result<Vec<u8>, String>>>,
    done: Condvar,
}

struct PoolState {
    free_slots: VecDeque<usize>,
    pending: VecDeque<Arc<CoalescedItem>>,
}

struct GpuBcpack {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    next_fence: AtomicU64,
    enhanced: bool,
    root_sig: ID3D12RootSignature,
    pso_lanes: ID3D12PipelineState,
    pso_resolve_plain: ID3D12PipelineState,
    pso_resolve_bc6: ID3D12PipelineState,
    pso_resolve_bc7: ID3D12PipelineState,
    pso_pack_plain: ID3D12PipelineState,
    pso_pack_bc6: ID3D12PipelineState,
    pso_pack_bc7: ID3D12PipelineState,
    tables_lock: Mutex<TablesUploader>,
    slots: [Mutex<FlightSlot>; FLIGHT_SLOTS],
    pool: (Mutex<PoolState>, Condvar),
}

// SAFETY: D3D12 device/queue/fence/PSOs are free-threaded; mutable state is guarded by mutexes.
unsafe impl Send for GpuBcpack {}
// SAFETY: as above.
unsafe impl Sync for GpuBcpack {}

fn create_buffer(
    device: &ID3D12Device,
    size: u64,
    heap: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let mut res: Option<ID3D12Resource> = None;
    // SAFETY: plain D3D12 committed buffer creation.
    unsafe {
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size.max(256),
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut res,
            )
            .map_err(|e| format!("D3D12 CreateCommittedResource ({size} B) failed: {e}"))?;
    }
    res.ok_or_else(|| "D3D12 CreateCommittedResource returned null".to_string())
}

fn upload_bytes(res: &ID3D12Resource, data: &[u8]) -> Result<(), String> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let empty_read = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: `res` is an upload buffer large enough for `data`.
    unsafe {
        res.Map(0, Some(&empty_read), Some(&mut ptr))
            .map_err(|e| format!("D3D12 Map upload failed: {e}"))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
        res.Unmap(0, None);
    }
    Ok(())
}

fn readback_bytes(res: &ID3D12Resource, dst: &mut [u8]) -> Result<(), String> {
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let read_range = D3D12_RANGE { Begin: 0, End: dst.len() };
    let no_write = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: `res` is a readback buffer large enough for `dst`, and GPU work completed.
    unsafe {
        res.Map(0, Some(&read_range), Some(&mut ptr))
            .map_err(|e| format!("D3D12 Map readback failed: {e}"))?;
        std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), dst.as_mut_ptr(), dst.len());
        res.Unmap(0, Some(&no_write));
    }
    Ok(())
}

fn transition(
    list: &ID3D12GraphicsCommandList,
    transitions: &[(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)],
) {
    let mut barriers: Vec<D3D12_RESOURCE_BARRIER> = transitions
        .iter()
        .map(|(res, before, after)| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some((*res).clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: *before,
                    StateAfter: *after,
                }),
            },
        })
        .collect();
    // SAFETY: open command list; each Transition owns a cloned COM ref dropped right after.
    unsafe {
        list.ResourceBarrier(&barriers);
        for b in &mut barriers {
            let t = std::mem::ManuallyDrop::take(&mut b.Anonymous.Transition);
            let _ = std::mem::ManuallyDrop::into_inner(t.pResource);
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
    // SAFETY: open command list and live buffer resources.
    unsafe { list7.Barrier(&group) };
}

/// Emits resource-scoped UAV barriers for `work_uav` and `out_uav`, replacing
/// global `UAV(NULL)` pipeline flushes: Enhanced `D3D12_BUFFER_BARRIER` when
/// available, or resource-specific `D3D12_RESOURCE_BARRIER_TYPE_UAV` otherwise.
fn emit_scoped_uav_barriers(
    list: &ID3D12GraphicsCommandList,
    list7: Option<&ID3D12GraphicsCommandList7>,
    enhanced: bool,
    work_uav: &ID3D12Resource,
    out_uav: &ID3D12Resource,
) {
    if choose_barrier_kind(enhanced, list7.is_some()) == BarrierKind::Enhanced
        && let Some(l7) = list7
    {
        emit_enhanced_buf_barriers(
            l7,
            &[
                enhanced_buf_barrier(
                    work_uav,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                ),
                enhanced_buf_barrier(
                    out_uav,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                ),
            ],
        );
        return;
    }
    let mut barriers = [
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(work_uav.clone())),
                }),
            },
        },
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(Some(out_uav.clone())),
                }),
            },
        },
    ];
    // SAFETY: open command list; each UAV barrier owns a cloned COM ref dropped right after.
    unsafe {
        list.ResourceBarrier(&barriers);
        for b in &mut barriers {
            let u = std::mem::ManuallyDrop::take(&mut b.Anonymous.UAV);
            let _ = std::mem::ManuallyDrop::into_inner(u.pResource);
        }
    }
}

fn prepare_batch(jobs: &[StreamJob<'_>]) -> Result<PreparedBatch, String> {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Family {
        Plain = 0,
        Bc6 = 1,
        Bc7 = 2,
    }

    struct JobMeta {
        orig_idx: usize,
        family: Family,
        bc: u8,
        blocks: u32,
        rec_stride: u32,
        inter_stride: u32,
        out_stride: u32,
    }

    let mut metas = Vec::with_capacity(jobs.len());
    for (orig_idx, job) in jobs.iter().enumerate() {
        let (family, rec_stride, inter_stride, out_stride) = match job.bc {
            1 | 4 => (Family::Plain, 12u32, PLAIN_BYTES as u32, 8u32),
            3 | 5 => (Family::Plain, 24u32, PLAIN_BYTES as u32, 16u32),
            6 => (Family::Bc6, 32u32, BC6_BYTES as u32, 16u32),
            7 => (Family::Bc7, 32u32, BC7_BYTES as u32, 16u32),
            _ => return Err("BCPack: unsupported BC format".to_string()),
        };
        if job.blocks == 0 {
            return Err("BCPack: destination is not a whole number of blocks".to_string());
        }
        let blocks = u32::try_from(job.blocks)
            .map_err(|_| "BCPack: block count exceeds GPU 32-bit limit".to_string())?;
        metas.push(JobMeta {
            orig_idx,
            family,
            bc: job.bc,
            blocks,
            rec_stride,
            inter_stride,
            out_stride,
        });
    }

    // Group streams by format family so each family occupies one contiguous range.
    metas.sort_by_key(|m| m.family);

    let n_streams = metas.len();
    let desc_bytes = n_streams
        .checked_mul(STREAM_DESC_BYTES)
        .ok_or_else(|| "BCPack: stream descriptor overflow".to_string())?;

    let mut total_input = desc_bytes;
    for m in &metas {
        let padded = jobs[m.orig_idx].src.len().next_multiple_of(4);
        total_input = total_input
            .checked_add(padded)
            .ok_or_else(|| "BCPack: input buffer overflow".to_string())?;
    }
    // Extra 4 bytes so ByteAddressBuffer.Load on trailing bytes stays in-bounds.
    let input_buf_len = total_input + 4;
    let _ = u32::try_from(input_buf_len)
        .map_err(|_| "BCPack: input batch exceeds 4 GiB".to_string())?;

    let mut input_host = vec![0u8; input_buf_len];
    let mut src_cursor = desc_bytes;

    let mut rec_total = 0u64;
    for m in &metas {
        rec_total = rec_total
            .checked_add(u64::from(m.blocks) * u64::from(m.rec_stride))
            .ok_or_else(|| "BCPack: records buffer overflow".to_string())?;
    }
    let inter_start = rec_total.next_multiple_of(16);

    let mut rec_cursor = 0u64;
    let mut inter_cursor = inter_start;
    let mut out_cursor = 0u64;

    let mut n_plain = 0u32;
    let mut n_bc6 = 0u32;
    let mut n_bc7 = 0u32;
    let mut plain_blocks = 0u32;
    let mut bc6_blocks = 0u32;
    let mut bc7_blocks = 0u32;

    let mut orig_out_ranges = vec![(0usize, 0usize); n_streams];

    for (sidx, m) in metas.iter().enumerate() {
        let src = jobs[m.orig_idx].src;
        let src_off = src_cursor as u32;
        let src_len = src.len() as u32;
        input_host[src_cursor..src_cursor + src.len()].copy_from_slice(src);
        src_cursor += src.len().next_multiple_of(4);

        let rec_off = u32::try_from(rec_cursor)
            .map_err(|_| "BCPack: work buffer exceeds 4 GiB".to_string())?;
        let inter_off = u32::try_from(inter_cursor)
            .map_err(|_| "BCPack: work buffer exceeds 4 GiB".to_string())?;
        let out_off = u32::try_from(out_cursor)
            .map_err(|_| "BCPack: output buffer exceeds 4 GiB".to_string())?;

        let family_block_start = match m.family {
            Family::Plain => {
                let s = plain_blocks;
                plain_blocks = plain_blocks
                    .checked_add(m.blocks)
                    .ok_or_else(|| "BCPack: block count overflow".to_string())?;
                n_plain += 1;
                s
            }
            Family::Bc6 => {
                let s = bc6_blocks;
                bc6_blocks = bc6_blocks
                    .checked_add(m.blocks)
                    .ok_or_else(|| "BCPack: block count overflow".to_string())?;
                n_bc6 += 1;
                s
            }
            Family::Bc7 => {
                let s = bc7_blocks;
                bc7_blocks = bc7_blocks
                    .checked_add(m.blocks)
                    .ok_or_else(|| "BCPack: block count overflow".to_string())?;
                n_bc7 += 1;
                s
            }
        };

        let d = sidx * STREAM_DESC_BYTES;
        write_u32_le(&mut input_host, d, u32::from(m.bc));
        write_u32_le(&mut input_host, d + 4, m.blocks);
        write_u32_le(&mut input_host, d + 8, src_off);
        write_u32_le(&mut input_host, d + 12, src_len);
        write_u32_le(&mut input_host, d + 16, rec_off);
        write_u32_le(&mut input_host, d + 20, inter_off);
        write_u32_le(&mut input_host, d + 24, out_off);
        write_u32_le(&mut input_host, d + 28, family_block_start);

        let out_bytes = (u64::from(m.blocks) * u64::from(m.out_stride)) as usize;
        orig_out_ranges[m.orig_idx] = (out_off as usize, out_bytes);

        rec_cursor += u64::from(m.blocks) * u64::from(m.rec_stride);
        inter_cursor += u64::from(m.blocks) * u64::from(m.inter_stride);
        out_cursor += out_bytes as u64;
    }

    const MAX_D3D12_DISPATCH_GROUPS: u32 = 65_535;
    if n_streams > MAX_D3D12_DISPATCH_GROUPS as usize
        || plain_blocks.div_ceil(64) > MAX_D3D12_DISPATCH_GROUPS
        || bc6_blocks.div_ceil(64) > MAX_D3D12_DISPATCH_GROUPS
        || bc7_blocks.div_ceil(64) > MAX_D3D12_DISPATCH_GROUPS
    {
        return Err(format!(
            "gpu_bcpack: dispatch exceeds D3D12 65535 thread-group limit \
             (streams={n_streams}, plain={plain_blocks}, bc6={bc6_blocks}, bc7={bc7_blocks})"
        ));
    }

    let work_total = inter_cursor.max(256);
    let status_offset = u32::try_from(out_cursor.next_multiple_of(4))
        .map_err(|_| "BCPack: output buffer exceeds 4 GiB".to_string())?;
    let out_total = u64::from(status_offset) + (n_streams as u64) * 4;

    Ok(PreparedBatch {
        n_streams,
        n_plain,
        n_bc6,
        n_bc7,
        plain_blocks,
        bc6_blocks,
        bc7_blocks,
        input_host,
        work_total,
        status_offset,
        out_total,
        orig_out_ranges,
    })
}

fn record_bcpack_telemetry(streams: u64, out_bytes: u64) {
    let batches = BCPACK_STATS[0].fetch_add(1, Ordering::Relaxed) + 1;
    let total_streams = BCPACK_STATS[1].fetch_add(streams, Ordering::Relaxed) + streams;
    let coalesced = if streams > 1 {
        BCPACK_STATS[2].fetch_add(1, Ordering::Relaxed) + 1
    } else {
        BCPACK_STATS[2].load(Ordering::Relaxed)
    };
    let total_bytes = BCPACK_STATS[3].fetch_add(out_bytes, Ordering::Relaxed) + out_bytes;
    if batches == 1 || batches.is_multiple_of(500) {
        tracing::info!(
            batches,
            coalesced_batches = coalesced,
            streams = total_streams,
            avg_streams_per_batch = format_args!("{:.2}", total_streams as f64 / batches.max(1) as f64),
            mib = total_bytes >> 20,
            "gpu_bcpack telemetry"
        );
    }
}

impl GpuBcpack {
    fn new(device: ID3D12Device) -> Result<Self, String> {
        let mut root_sig: Option<ID3D12RootSignature> = None;
        let mut make_pso = |define: &str, entry: PCSTR| -> Result<ID3D12PipelineState, String> {
            let src = format!("#define {define}\n{HLSL_SOURCE}");
            let code = crate::host::compile(&src, entry, s!("cs_5_1"))
                .map_err(|e| format!("HLSL compile {define} failed: {e}"))?;
            // SAFETY: bytecode contains embedded root signature; plain D3D12 PSO creation.
            unsafe {
                if root_sig.is_none() {
                    root_sig = Some(
                        device
                            .CreateRootSignature(0, &code)
                            .map_err(|e| format!("CreateRootSignature failed: {e}"))?,
                    );
                }
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root_sig),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                };
                device
                    .CreateComputePipelineState(&desc)
                    .map_err(|e| format!("CreateComputePipelineState ({define}) failed: {e}"))
            }
        };

        let pso_lanes = make_pso("PASS_LANES", s!("bcpack_lanes"))?;
        let pso_resolve_plain = make_pso("PASS_RESOLVE_PLAIN", s!("bcpack_resolve_plain"))?;
        let pso_resolve_bc6 = make_pso("PASS_RESOLVE_BC6", s!("bcpack_resolve_bc6"))?;
        let pso_resolve_bc7 = make_pso("PASS_RESOLVE_BC7", s!("bcpack_resolve_bc7"))?;
        let pso_pack_plain = make_pso("PASS_PACK_PLAIN", s!("bcpack_pack_plain"))?;
        let pso_pack_bc6 = make_pso("PASS_PACK_BC6", s!("bcpack_pack_bc6"))?;
        let pso_pack_bc7 = make_pso("PASS_PACK_BC7", s!("bcpack_pack_bc7"))?;
        let root_sig = root_sig.ok_or_else(|| "missing root signature".to_string())?;
        let enhanced = device_enhanced_barriers(&device);

        // SAFETY: plain D3D12 compute queue, fence, and table uploader objects.
        let (queue, fence, tables_uploader) = unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    ..Default::default()
                })
                .map_err(|e| format!("CreateCommandQueue failed: {e}"))?;
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(|e| format!("CreateFence failed: {e}"))?;
            let tbl_alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(|e| format!("CreateCommandAllocator (tables) failed: {e}"))?;
            let tbl_list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &tbl_alloc, None)
                .map_err(|e| format!("CreateCommandList (tables) failed: {e}"))?;
            tbl_list
                .Close()
                .map_err(|e| format!("initial tables list Close failed: {e}"))?;
            let tbl_event = CreateEventW(None, false, false, None)
                .map_err(|e| format!("CreateEventW (tables) failed: {e}"))?;
            (
                queue,
                fence,
                TablesUploader {
                    allocator: tbl_alloc,
                    list: tbl_list,
                    event: tbl_event,
                    cache: None,
                },
            )
        };

        let slot0 = FlightSlot::new(&device, enhanced)?;
        let slot1 = FlightSlot::new(&device, enhanced)?;
        let barriers_label = if choose_barrier_kind(enhanced, slot0.list7.is_some()) == BarrierKind::Enhanced {
            "enhanced"
        } else {
            "legacy-scoped"
        };
        tracing::info!(
            slots = FLIGHT_SLOTS,
            barriers = barriers_label,
            "gpu_bcpack compute runtime initialized"
        );

        let mut free_slots = VecDeque::with_capacity(FLIGHT_SLOTS);
        for i in 0..FLIGHT_SLOTS {
            free_slots.push_back(i);
        }

        Ok(Self {
            device,
            queue,
            fence,
            next_fence: AtomicU64::new(0),
            enhanced,
            root_sig,
            pso_lanes,
            pso_resolve_plain,
            pso_resolve_bc6,
            pso_resolve_bc7,
            pso_pack_plain,
            pso_pack_bc6,
            pso_pack_bc7,
            tables_lock: Mutex::new(tables_uploader),
            slots: [Mutex::new(slot0), Mutex::new(slot1)],
            pool: (
                Mutex::new(PoolState {
                    free_slots,
                    pending: VecDeque::new(),
                }),
                Condvar::new(),
            ),
        })
    }

    fn wait_fence_with_event(&self, fence_val: u64, event: HANDLE) -> Result<(), String> {
        if fence_val == 0 {
            return Ok(());
        }
        // SAFETY: live fence and event.
        unsafe {
            if self.fence.GetCompletedValue() < fence_val {
                self.fence
                    .SetEventOnCompletion(fence_val, event)
                    .map_err(|e| format!("SetEventOnCompletion failed: {e}"))?;
                WaitForSingleObject(event, INFINITE);
            }
        }
        Ok(())
    }

    fn ensure_tables_vram(&self, tables_bytes: &[u8]) -> Result<u64, String> {
        let hash = blake3::hash(tables_bytes);
        let mut uploader = self.tables_lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = &uploader.cache
            && cached.hash == hash
        {
            // SAFETY: resource is valid and resident in NON_PIXEL_SHADER_RESOURCE state.
            return Ok(unsafe { cached.resource.GetGPUVirtualAddress() });
        }

        let staging = GpuTables::from_bytes(&self.device, tables_bytes)?;
        let vram = create_buffer(
            &self.device,
            staging.packed.len() as u64,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let fence_val = self.next_fence.fetch_add(1, Ordering::AcqRel) + 1;
        // SAFETY: tables uploader allocator/list are serialized behind `tables_lock`;
        // we wait for `fence_val` before dropping `staging`.
        unsafe {
            uploader
                .allocator
                .Reset()
                .map_err(|e| format!("tables allocator Reset failed: {e}"))?;
            uploader
                .list
                .Reset(&uploader.allocator, None)
                .map_err(|e| format!("tables list Reset failed: {e}"))?;
            uploader
                .list
                .CopyBufferRegion(&vram, 0, &staging.resource, 0, staging.packed.len() as u64);
            transition(
                &uploader.list,
                &[(
                    &vram,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )],
            );
            uploader
                .list
                .Close()
                .map_err(|e| format!("tables list Close failed: {e}"))?;
            let cmd_list: ID3D12CommandList = uploader
                .list
                .cast()
                .map_err(|e| format!("tables list cast failed: {e}"))?;
            self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
            self.queue
                .Signal(&self.fence, fence_val)
                .map_err(|e| format!("tables queue Signal failed: {e}"))?;
        }
        self.wait_fence_with_event(fence_val, uploader.event)?;

        // SAFETY: resource is valid.
        let gpu_va = unsafe { vram.GetGPUVirtualAddress() };
        uploader.cache = Some(CachedVramTables { hash, resource: vram });
        Ok(gpu_va)
    }

    fn acquire_slot(&self) -> usize {
        let mut pool = self.pool.0.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(slot_idx) = pool.free_slots.pop_front() {
                return slot_idx;
            }
            pool = self.pool.1.wait(pool).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn release_slot(&self, slot_idx: usize) {
        let mut pool = self.pool.0.lock().unwrap_or_else(|e| e.into_inner());
        pool.free_slots.push_back(slot_idx);
        self.pool.1.notify_all();
    }

    /// Encodes and submits `batch` onto `slot_idx` without waiting on the GPU fence,
    /// returning the signaled fence value so another batch can overlap on the other slot.
    fn submit_on_slot(
        &self,
        slot_idx: usize,
        batch: &PreparedBatch,
        tables_va: u64,
    ) -> Result<u64, String> {
        let mut slot = self.slots[slot_idx].lock().unwrap_or_else(|e| e.into_inner());
        // Ensure prior submission on this slot has completed before resetting its allocator.
        self.wait_fence_with_event(slot.fence_val, slot.event)?;

        let use_enhanced =
            choose_barrier_kind(self.enhanced, slot.list7.is_some()) == BarrierKind::Enhanced;
        let default_init = if use_enhanced {
            D3D12_RESOURCE_STATE_COMMON
        } else {
            D3D12_RESOURCE_STATE_COPY_DEST
        };
        let uav_init = if use_enhanced {
            D3D12_RESOURCE_STATE_COMMON
        } else {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        };

        let input_upload = slot
            .input_upload
            .ensure(
                &self.device,
                batch.input_host.len() as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?
            .clone();
        upload_bytes(&input_upload, &batch.input_host)?;

        let input_default = slot
            .input_default
            .ensure(
                &self.device,
                batch.input_host.len() as u64,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_NONE,
                default_init,
            )?
            .clone();
        let work_uav = slot
            .work_uav
            .ensure(
                &self.device,
                batch.work_total,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                uav_init,
            )?
            .clone();
        let out_uav = slot
            .out_uav
            .ensure(
                &self.device,
                batch.out_total,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                uav_init,
            )?
            .clone();
        let out_readback = slot
            .out_readback
            .ensure(
                &self.device,
                batch.out_total,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?
            .clone();

        let fence_val = self.next_fence.fetch_add(1, Ordering::AcqRel) + 1;

        // SAFETY: slot allocator/list are idle; all buffers live until `finish_on_slot`
        // waits for `fence_val`; barriers strictly order all phases without `UAV(NULL)`.
        unsafe {
            slot.allocator
                .Reset()
                .map_err(|e| format!("allocator Reset failed: {e}"))?;
            slot.list
                .Reset(&slot.allocator, None)
                .map_err(|e| format!("list Reset failed: {e}"))?;

            let l = &slot.list;
            let l7 = slot.list7.as_ref();

            l.CopyBufferRegion(
                &input_default,
                0,
                &input_upload,
                0,
                batch.input_host.len() as u64,
            );

            if use_enhanced && let Some(l7) = l7 {
                emit_enhanced_buf_barriers(
                    l7,
                    &[enhanced_buf_barrier(
                        &input_default,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                        D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                    )],
                );
            } else {
                transition(
                    l,
                    &[(
                        &input_default,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    )],
                );
            }

            l.SetComputeRootSignature(&self.root_sig);
            l.SetComputeRootShaderResourceView(1, tables_va);
            l.SetComputeRootShaderResourceView(2, input_default.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(3, work_uav.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(4, out_uav.GetGPUVirtualAddress());

            let status_offset = batch.status_offset;
            let dispatch = |pso: &ID3D12PipelineState, sbase: u32, scount: u32, tblks: u32, groups: u32| {
                let consts = [sbase, scount, tblks, status_offset, 0u32, 0, 0, 0];
                l.SetComputeRoot32BitConstants(0, 8, consts.as_ptr().cast(), 0);
                l.SetPipelineState(pso);
                l.Dispatch(groups, 1, 1);
            };

            // Pass 1: stream-parallel 12/16-lane rANS decode across all streams.
            dispatch(
                &self.pso_lanes,
                0,
                batch.n_streams as u32,
                0,
                batch.n_streams as u32,
            );
            emit_scoped_uav_barriers(l, l7, self.enhanced, &work_uav, &out_uav);

            // Pass 2a: stream-parallel ring-buffer history & dictionary resolution.
            if batch.n_plain > 0 {
                dispatch(
                    &self.pso_resolve_plain,
                    0,
                    batch.n_plain,
                    batch.plain_blocks,
                    batch.n_plain,
                );
            }
            if batch.n_bc6 > 0 {
                dispatch(
                    &self.pso_resolve_bc6,
                    batch.n_plain,
                    batch.n_bc6,
                    batch.bc6_blocks,
                    batch.n_bc6,
                );
            }
            if batch.n_bc7 > 0 {
                dispatch(
                    &self.pso_resolve_bc7,
                    batch.n_plain + batch.n_bc6,
                    batch.n_bc7,
                    batch.bc7_blocks,
                    batch.n_bc7,
                );
            }
            emit_scoped_uav_barriers(l, l7, self.enhanced, &work_uav, &out_uav);

            // Pass 2b: 100% block-parallel BC block bit-packing.
            if batch.plain_blocks > 0 {
                dispatch(
                    &self.pso_pack_plain,
                    0,
                    batch.n_plain,
                    batch.plain_blocks,
                    batch.plain_blocks.div_ceil(64),
                );
            }
            if batch.bc6_blocks > 0 {
                dispatch(
                    &self.pso_pack_bc6,
                    batch.n_plain,
                    batch.n_bc6,
                    batch.bc6_blocks,
                    batch.bc6_blocks.div_ceil(64),
                );
            }
            if batch.bc7_blocks > 0 {
                dispatch(
                    &self.pso_pack_bc7,
                    batch.n_plain + batch.n_bc6,
                    batch.n_bc7,
                    batch.bc7_blocks,
                    batch.bc7_blocks.div_ceil(64),
                );
            }

            if use_enhanced && let Some(l7) = l7 {
                emit_enhanced_buf_barriers(
                    l7,
                    &[enhanced_buf_barrier(
                        &out_uav,
                        D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                        D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    )],
                );
            } else {
                transition(
                    l,
                    &[(
                        &out_uav,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )],
                );
            }

            l.CopyBufferRegion(&out_readback, 0, &out_uav, 0, batch.out_total);

            if !use_enhanced {
                // Restore persistent legacy buffer states for the next call.
                transition(
                    l,
                    &[
                        (
                            &out_uav,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        ),
                        (
                            &input_default,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        ),
                    ],
                );
            }

            l.Close().map_err(|e| format!("list Close failed: {e}"))?;
            let cmd_list: ID3D12CommandList =
                l.cast().map_err(|e| format!("list cast failed: {e}"))?;
            self.queue.ExecuteCommandLists(&[Some(cmd_list)]);
            self.queue
                .Signal(&self.fence, fence_val)
                .map_err(|e| format!("queue Signal failed: {e}"))?;
        }

        slot.fence_val = fence_val;
        Ok(fence_val)
    }

    /// Waits for `slot_idx`'s submission to land, reads back decoded streams,
    /// checks per-stream status codes, and trims oversized scratch buffers.
    fn finish_on_slot(
        &self,
        slot_idx: usize,
        fence_val: u64,
        batch: &PreparedBatch,
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut slot = self.slots[slot_idx].lock().unwrap_or_else(|e| e.into_inner());
        self.wait_fence_with_event(fence_val, slot.event)?;

        let out_readback = slot
            .out_readback
            .res
            .as_ref()
            .ok_or_else(|| "missing readback buffer".to_string())?
            .clone();
        let mut readback = vec![0u8; batch.out_total as usize];
        readback_bytes(&out_readback, &mut readback)?;

        // Drop oversized outlier scratch buffers (> 16 MiB) once the fence has
        // landed so transient large batches never permanently pin VRAM.
        slot.trim_oversized();
        drop(slot);

        let st_base = batch.status_offset as usize;
        for sidx in 0..batch.n_streams {
            let code = u32::from_le_bytes(
                readback[st_base + sidx * 4..st_base + sidx * 4 + 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            match code {
                0 => {}
                1 => return Err("BCPack: the stream ends early or is corrupt".to_string()),
                2 => return Err("BCPack: BC6H mode out of range".to_string()),
                3 => return Err("BCPack: unused BC7 control code".to_string()),
                other => return Err(format!("BCPack: GPU decode error ({other})")),
            }
        }

        let mut results = Vec::with_capacity(batch.n_streams);
        for &(off, len) in &batch.orig_out_ranges {
            results.push(readback[off..off + len].to_vec());
        }
        record_bcpack_telemetry(batch.n_streams as u64, batch.out_total);
        Ok(results)
    }

    /// Coalesces single-stream callers that arrive while both flight slots are
    /// busy into one batched GPU dispatch as soon as either slot frees up.
    fn decode_single_coalesced(
        &self,
        bc: u8,
        src: &[u8],
        blocks: usize,
        tables_bytes: &[u8],
    ) -> Result<Vec<u8>, String> {
        let tables_hash = blake3::hash(tables_bytes);
        let tables_va = self.ensure_tables_vram(tables_bytes)?;

        // Fast path: grab a free slot immediately (plus any compatible queued items).
        let (slot_idx, followers) = {
            let mut pool = self.pool.0.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(slot_idx) = pool.free_slots.pop_front() {
                let followers = drain_compatible(&mut pool.pending, tables_hash, src.len());
                (slot_idx, followers)
            } else {
                // Both double-buffered slots are in flight on the GPU: queue this
                // request so the next thread that acquires a slot batches us together.
                let item = Arc::new(CoalescedItem {
                    bc,
                    src: src.to_vec(),
                    blocks,
                    tables_hash,
                    result: Mutex::new(None),
                    done: Condvar::new(),
                });
                pool.pending.push_back(item.clone());
                loop {
                    if let Some(res) = item.result.lock().unwrap_or_else(|e| e.into_inner()).take() {
                        return res;
                    }
                    if let Some(slot_idx) = pool.free_slots.pop_front() {
                        // We acquired a slot; check if another leader already claimed our item.
                        if let Some(pos) = pool.pending.iter().position(|p| Arc::ptr_eq(p, &item)) {
                            pool.pending.remove(pos);
                            let followers = drain_compatible(&mut pool.pending, tables_hash, src.len());
                            break (slot_idx, followers);
                        }
                        // Another leader already popped our item; return slot and wait on item.done.
                        pool.free_slots.push_back(slot_idx);
                        self.pool.1.notify_one();
                        drop(pool);
                        let mut res_guard = item.result.lock().unwrap_or_else(|e| e.into_inner());
                        while res_guard.is_none() {
                            res_guard = item.done.wait(res_guard).unwrap_or_else(|e| e.into_inner());
                        }
                        return res_guard.take().unwrap_or_else(|| Err("BCPack: missing result".into()));
                    }
                    pool = self.pool.1.wait(pool).unwrap_or_else(|e| e.into_inner());
                }
            }
        };

        let mut jobs = Vec::with_capacity(1 + followers.len());
        jobs.push(StreamJob { bc, src, blocks });
        for f in &followers {
            jobs.push(StreamJob {
                bc: f.bc,
                src: &f.src,
                blocks: f.blocks,
            });
        }

        let batch_res = prepare_batch(&jobs).and_then(|batch| {
            let fence_val = self.submit_on_slot(slot_idx, &batch, tables_va)?;
            self.finish_on_slot(slot_idx, fence_val, &batch)
        });
        self.release_slot(slot_idx);

        match batch_res {
            Ok(mut outs) => {
                let mine = outs.remove(0);
                for (f, out) in followers.into_iter().zip(outs) {
                    *f.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(Ok(out));
                    f.done.notify_one();
                }
                self.pool.1.notify_all();
                Ok(mine)
            }
            Err(err) => {
                // If a coalesced batch failed (e.g., one corrupt stream), fall back to
                // decoding each stream individually so a bad stream never poisons healthy siblings.
                if !followers.is_empty() {
                    for f in followers {
                        let single = self.decode_batch_sync(
                            &[StreamJob {
                                bc: f.bc,
                                src: &f.src,
                                blocks: f.blocks,
                            }],
                            tables_bytes,
                        );
                        let mapped = single.map(|mut v| v.pop().unwrap_or_default());
                        *f.result.lock().unwrap_or_else(|e| e.into_inner()) = Some(mapped);
                        f.done.notify_one();
                    }
                    self.pool.1.notify_all();
                    let mine = self.decode_batch_sync(
                        &[StreamJob { bc, src, blocks }],
                        tables_bytes,
                    )?;
                    return Ok(mine.into_iter().next().unwrap_or_default());
                }
                Err(err)
            }
        }
    }

    fn decode_batch_sync(
        &self,
        jobs: &[StreamJob<'_>],
        tables_bytes: &[u8],
    ) -> Result<Vec<Vec<u8>>, String> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let batch = prepare_batch(jobs)?;
        let tables_va = self.ensure_tables_vram(tables_bytes)?;
        let slot_idx = self.acquire_slot();
        let res = self
            .submit_on_slot(slot_idx, &batch, tables_va)
            .and_then(|fence_val| self.finish_on_slot(slot_idx, fence_val, &batch));
        self.release_slot(slot_idx);
        res
    }
}

fn drain_compatible(
    pending: &mut VecDeque<Arc<CoalescedItem>>,
    tables_hash: blake3::Hash,
    initial_bytes: usize,
) -> Vec<Arc<CoalescedItem>> {
    let mut out = Vec::new();
    let mut bytes = initial_bytes;
    let mut i = 0usize;
    while i < pending.len() && out.len() + 1 < MAX_COALESCED_STREAMS {
        let compat = pending[i].tables_hash == tables_hash
            && bytes.saturating_add(pending[i].src.len()) <= MAX_COALESCED_INPUT_BYTES;
        if compat {
            if let Some(item) = pending.remove(i) {
                bytes = bytes.saturating_add(item.src.len());
                out.push(item);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Handle to an asynchronously submitted GPU BCPack batch occupying one of the
/// double-buffered [`FlightSlot`]s. Calling [`Self::wait`] blocks until the
/// compute queue fence completes and returns the decoded BC block buffers.
#[allow(dead_code)]
pub struct PendingBatch {
    rt: &'static GpuBcpack,
    slot_idx: Option<usize>,
    fence_val: u64,
    batch: PreparedBatch,
}

#[allow(dead_code)]
impl PendingBatch {
    /// Which double-buffered flight slot (`0` or `1`) this batch runs on.
    pub fn slot(&self) -> usize {
        self.slot_idx.unwrap_or(0)
    }

    /// Waits for the GPU compute batch to finish and reads back all decoded streams.
    pub fn wait(mut self) -> Result<Vec<Vec<u8>>, String> {
        let Some(slot_idx) = self.slot_idx.take() else {
            return Ok(Vec::new());
        };
        let res = self.rt.finish_on_slot(slot_idx, self.fence_val, &self.batch);
        self.rt.release_slot(slot_idx);
        res
    }
}

impl Drop for PendingBatch {
    fn drop(&mut self) {
        if let Some(slot_idx) = self.slot_idx.take() {
            let slot = self.rt.slots[slot_idx].lock().unwrap_or_else(|e| e.into_inner());
            let _ = self.rt.wait_fence_with_event(self.fence_val, slot.event);
            drop(slot);
            self.rt.release_slot(slot_idx);
        }
    }
}

fn runtime() -> Result<&'static GpuBcpack, String> {
    static RUNTIME: OnceLock<Result<GpuBcpack, String>> = OnceLock::new();
    match RUNTIME.get_or_init(|| {
        let host = crate::host::host()
            .ok_or_else(|| "no host D3D12 device available for GPU BCPack".to_string())?;
        GpuBcpack::new(host.device.clone())
    }) {
        Ok(rt) => Ok(rt),
        Err(e) => Err(e.clone()),
    }
}

/// Decodes a single BCPack stream on the GPU using the shipped `bcpack-tables.bin`.
/// Automatically overlaps across double-buffered flight slots and coalesces
/// concurrent callers into batched compute submissions.
#[allow(dead_code)]
pub fn decode_gpu(bc: u8, src: &[u8], dst: &mut [u8]) -> Result<(), String> {
    decode_gpu_with_tables(bc, src, dst, BUILTIN_TABLES)
}

/// Decodes a single BCPack stream on the GPU using custom table bytes.
#[allow(dead_code)]
pub fn decode_gpu_with_tables(
    bc: u8,
    src: &[u8],
    dst: &mut [u8],
    tables_bytes: &[u8],
) -> Result<(), String> {
    let block_bytes = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err("BCPack: unsupported BC format".to_string()),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block_bytes) {
        return Err("BCPack: destination is not a whole number of blocks".to_string());
    }
    let blocks = dst.len() / block_bytes;
    let rt = runtime()?;
    let out = rt.decode_single_coalesced(bc, src, blocks, tables_bytes)?;
    dst.copy_from_slice(&out);
    Ok(())
}

/// Submits a batch of BCPack streams asynchronously onto one of the double-buffered
/// compute flight slots so consecutive batches overlap on the GPU.
#[allow(dead_code)]
pub fn submit_gpu_batch_async(
    jobs: &[StreamJob<'_>],
    tables_bytes: &[u8],
) -> Result<PendingBatch, String> {
    let rt = runtime()?;
    if jobs.is_empty() {
        return Ok(PendingBatch {
            rt,
            slot_idx: None,
            fence_val: 0,
            batch: PreparedBatch {
                n_streams: 0,
                n_plain: 0,
                n_bc6: 0,
                n_bc7: 0,
                plain_blocks: 0,
                bc6_blocks: 0,
                bc7_blocks: 0,
                input_host: Vec::new(),
                work_total: 0,
                status_offset: 0,
                out_total: 0,
                orig_out_ranges: Vec::new(),
            },
        });
    }
    let batch = prepare_batch(jobs)?;
    let tables_va = rt.ensure_tables_vram(tables_bytes)?;
    let slot_idx = rt.acquire_slot();
    match rt.submit_on_slot(slot_idx, &batch, tables_va) {
        Ok(fence_val) => Ok(PendingBatch {
            rt,
            slot_idx: Some(slot_idx),
            fence_val,
            batch,
        }),
        Err(e) => {
            rt.release_slot(slot_idx);
            Err(e)
        }
    }
}

/// Decodes a batch of BCPack streams (potentially mixed BC formats and sizes) on the GPU.
#[allow(dead_code)]
pub fn decode_gpu_batch(
    jobs: &[StreamJob<'_>],
    tables_bytes: &[u8],
) -> Result<Vec<Vec<u8>>, String> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    submit_gpu_batch_async(jobs, tables_bytes)?.wait()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use victory_formats::bcpack::{self, Tables};

    use super::*;

    const LOW: u32 = 1 << 24;

    fn widths_and_per(bc: u8) -> (&'static [u8], usize, usize) {
        match bc {
            1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1, 8),
            3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2, 16),
            4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1, 8),
            5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2, 16),
            6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2, 16),
            7 => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2, 16),
            _ => unreachable!(),
        }
    }

    /// Synthetic uniform tables (matching `bcpack.rs::tests::tables`, plus ctrl7 entries for all modes).
    fn synthetic_tables() -> Vec<u8> {
        let mut t = Vec::new();
        for m in 0..MODELS {
            let g = m / 16;
            for b in 0..BUCKETS {
                let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                t.extend_from_slice(&f.to_le_bytes());
            }
        }
        t.extend(0..=255u8); // rank1
        t.extend((0..64u8).map(|v| (v & 7) | ((v >> 3) << 4))); // rank3a
        t.extend(0..=255u8); // rank3c
        t.extend((0..64u8).map(|v| (v & 7) | ((v >> 3) << 4))); // rank4
        t.extend((0..64u8).map(|v| (v & 7) | ((v >> 3) << 4))); // rank5

        // Populate ctrl7 codes for all 8 BC7 modes, rotations, indexSelectionBit, and partitions:
        // code 0: mode 6 (1 subset)
        // code 1..=15: mode 0 (3 subsets, part = code)
        // code 16..=31: mode 1 (2 subsets, part = code)
        // code 32..=47: mode 2 (3 subsets, part = code)
        // code 48..=63: mode 3 (2 subsets, part = code)
        // code 64..=71: mode 4 (1 subset, rot = (code-64)&3, idxSel = (code-64)>>2)
        // code 72..=75: mode 5 (1 subset, rot = code-72)
        // code 76..=91: mode 7 (2 subsets, part = code - 40)
        // code 92: mode 0 with non-zero rot bits (to verify rot is ignored outside modes 4/5)
        // code 93: mode 6 with non-zero rot bits
        // code 511: 0 (unused control code, for error testing)
        for code in 0..512u16 {
            let ctrl: u16 = match code {
                0 => (1 << 6) | 6,
                1..=15 => (code << 8) | (3 << 6),
                16..=31 => (code << 8) | (2 << 6) | 1,
                32..=47 => (code << 8) | (3 << 6) | 2,
                48..=63 => (code << 8) | (2 << 6) | 3,
                64..=71 => {
                    let rot = (code - 64) & 3;
                    let idx_sel = ((code - 64) >> 2) & 1;
                    (1 << 6) | (idx_sel << 5) | (rot << 3) | 4
                }
                72..=75 => {
                    let rot = code - 72;
                    (1 << 6) | (rot << 3) | 5
                }
                76..=91 => ((code - 40) << 8) | (2 << 6) | 7,
                92 => (5 << 8) | (3 << 6) | (2 << 3),
                93 => (1 << 6) | (3 << 3) | 6,
                _ => 0,
            };
            t.extend_from_slice(&ctrl.to_le_bytes());
        }
        t.resize(TABLES_SIZE, 0);
        t
    }

    struct EncTables {
        freq: [[u32; BUCKETS]; MODELS],
        start: [[u32; BUCKETS + 1]; MODELS],
    }

    impl EncTables {
        fn parse(bytes: &[u8]) -> Self {
            let mut r = ByteReader(bytes);
            let mut freq = [[0u32; BUCKETS]; MODELS];
            let mut start = [[0u32; BUCKETS + 1]; MODELS];
            for m in 0..MODELS {
                for b in 0..BUCKETS {
                    freq[m][b] = u32::from(r.u16());
                }
                let used = m / 16 + 2;
                let mut at = 0u32;
                for b in 0..used {
                    start[m][b] = at;
                    at += freq[m][b] << b.saturating_sub(1);
                }
                start[m][used..].fill(1 << 16);
            }
            Self { freq, start }
        }

        fn encode_fragment(&self, width: u8, shape: u8, reset: bool, symbols: &[u8]) -> Vec<u8> {
            let model = usize::from(width) * 16 + usize::from(shape & 15);
            let freq = &self.freq[model];
            let start = &self.start[model];
            let mut x = LOW;
            let mut bytes = Vec::new();
            for &s in symbols.iter().rev() {
                let (f, st) = if s == 0 {
                    (freq[0] - 1, 1)
                } else {
                    let b = (8 - s.leading_zeros()) as usize;
                    let f = freq[b];
                    (f, start[b] + (u32::from(s) - (1 << (b - 1))) * f)
                };
                while x >= (f << 16) {
                    bytes.push(x as u8);
                    x >>= 8;
                }
                x = ((x / f) << 16) + (x % f) + st;
            }
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.reverse();
            let mut hdr = shape & 15;
            if reset {
                hdr |= 0x10;
            }
            let mut out = vec![hdr];
            if bytes.len() > 256 {
                out[0] |= 0x20;
                let n = bytes.len() - 257;
                out.extend_from_slice(&[n as u8, (n >> 8) as u8]);
            } else {
                out.push((bytes.len() - 1) as u8);
            }
            out.extend(bytes);
            out
        }
    }

    #[derive(Clone, Copy, Default)]
    struct SimLane {
        width: u8,
        phase: u8,
        long: bool,
        reset: bool,
        model: u8,
        need: bool,
        latch: Option<u8>,
        out: Option<u8>,
        stalled: bool,
        x: u32,
        left: u32,
    }

    impl SimLane {
        fn step(&mut self, t: &EncTables) {
            if self.phase != 3 {
                let Some(b) = self.latch.take() else { return };
                self.need = true;
                match self.phase {
                    0 => {
                        let group = self.width - (b >> 6);
                        self.model = group * 16 + (b & 15);
                        self.reset = b & 0x10 != 0;
                        self.long = b & 0x20 != 0;
                        self.left = 0;
                        self.phase = 1;
                    }
                    1 if self.long => {
                        self.left = b.into();
                        self.phase = 2;
                    }
                    1 => {
                        self.left = u32::from(b) + 1;
                        self.phase = 3;
                    }
                    _ => {
                        self.left = (u32::from(b) << 8 | self.left) + 257;
                        self.phase = 3;
                    }
                }
                return;
            }
            let mut x = self.x;
            if x < LOW
                && let Some(b) = self.latch.take()
            {
                self.left = self.left.wrapping_sub(1);
                x = x << 8 | u32::from(b);
            }
            self.need = self.left != 0 && self.latch.is_none();
            if x > LOW {
                let freq = &t.freq[usize::from(self.model)];
                let start = &t.start[usize::from(self.model)];
                let slot = x & 0xffff;
                let b = start[1..BUCKETS].iter().filter(|&&s| s <= slot).count();
                let (sym, f, rem) = if b == 0 {
                    (0, freq[0] - 1, slot.wrapping_sub(1))
                } else {
                    let off = slot - start[b];
                    let q = off / freq[b];
                    ((1u32 << (b - 1)) + q, freq[b], off - q * freq[b])
                };
                self.x = (x >> 16).wrapping_mul(f).wrapping_add(rem);
                self.out = Some(sym as u8);
            } else if self.left != 0 {
                self.x = x;
            } else {
                self.phase = 0;
                self.need = self.latch.is_none();
                if self.reset {
                    self.x = 0;
                }
            }
        }
    }

    /// Encodes `records` into a valid BCPack stream for any table bytes (synthetic or shipped).
    fn encode_stream(table_bytes: &[u8], bc: u8, records: &[Vec<u8>], split_fragments: bool) -> Vec<u8> {
        let t = EncTables::parse(table_bytes);
        let (widths, per, _) = widths_and_per(bc);
        let mut queues: Vec<VecDeque<u8>> = (0..widths.len())
            .map(|j| {
                let syms: Vec<u8> = records
                    .iter()
                    .flat_map(|r| r[j * per..j * per + per].iter().copied())
                    .collect();
                let mut bytes = Vec::new();
                if split_fragments && syms.len() > 20 {
                    let mid = syms.len() / 2;
                    bytes.extend(t.encode_fragment(widths[j], (j % 16) as u8, true, &syms[..mid]));
                    bytes.extend(t.encode_fragment(widths[j], ((j + 3) % 16) as u8, false, &syms[mid..]));
                } else {
                    bytes.extend(t.encode_fragment(widths[j], (j % 16) as u8, false, &syms));
                }
                bytes.extend([0, 1, 0, 0]);
                bytes.into()
            })
            .collect();

        let mut lanes = vec![SimLane::default(); widths.len()];
        for (lane, &w) in lanes.iter_mut().zip(widths) {
            lane.width = w;
            lane.need = true;
        }
        let full = (1u32 << widths.len()) - 1;
        let mut pos = 0usize;
        let mut done = 0u32;
        let mut out = Vec::new();
        while pos < records.len() * per {
            let mut taken = 0usize;
            for (j, lane) in lanes.iter_mut().enumerate() {
                let bit = 1u32 << j;
                let before = taken;
                if !lane.stalled {
                    lane.step(&t);
                }
                let waiting = lane.out.is_some() && done & bit != 0;
                if lane.need && !waiting && taken < 8 {
                    let b = queues[j].pop_front().expect("lane queue underrun");
                    out.push(b);
                    lane.latch = Some(b);
                    taken += 1;
                }
                lane.stalled = (lane.need && before >= 8) || waiting;
                if !lane.stalled
                    && lane.out.take().is_some()
                {
                    done |= bit;
                }
            }
            if done == full {
                pos += 1;
                done = 0;
            }
        }
        out
    }

    /// Generates valid records for format `bc` with controlled references (zero, history, dictionary).
    fn make_records(bc: u8, blocks: usize, seed_init: u64, ctrl7_codes: &[u16]) -> Vec<Vec<u8>> {
        let (widths, per, _) = widths_and_per(bc);
        let mut seed = seed_init;
        let mut next = || -> u32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };

        (0..blocks)
            .map(|i| {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| {
                        let alpha = 1u32 << (widths[k / per] + 1);
                        (next() % alpha) as u8
                    })
                    .collect();

                // Exercise zero ref, history ref (<= i), and dictionary ref (> i) explicitly.
                if bc == 1 || bc == 4 {
                    if i == 0 {
                        // Force dictionary reference r > 0 on block 0.
                        r[0] = 0; // delta = 0
                        r[1] = 17; // ix_ref = ep_ref = 17 -> dict[16]
                    } else if i == 1 {
                        // Zero reference.
                        r[0] = 0;
                        r[1] = 0;
                    } else if i == 2 {
                        // History reference to block 0 and dictionary reference.
                        r[0] = 4; // delta = +2
                        r[1] = 2; // ix_ref = 2 (history[0]), ep_ref = 4 (dict[1])
                    }
                } else {
                    // Paired references (s[0..4] are 4-bit symbols):
                    // For block 0, ensure non-zero ix_ref so dictionary entries are hit.
                    if i == 0 {
                        r[0] = 0;
                        r[1] = 0;
                        r[2] = 5;
                        r[3] = 3;
                    }
                }

                if bc == 6 {
                    // Cycle through all 14 valid BC6H modes (s[30] in 0..=13 -> mode 1..=14).
                    r[30] = (i % 14) as u8;
                    r[31] = (next() % 32) as u8;
                } else if bc == 7 {
                    let code = ctrl7_codes[i % ctrl7_codes.len()];
                    r[30] = (code & 31) as u8;
                    r[31] = ((code >> 5) & 15) as u8;
                }
                r
            })
            .collect()
    }

    /// Collects all non-zero BC7 control code indices from raw table bytes.
    fn valid_ctrl7_codes(table_bytes: &[u8]) -> Vec<u16> {
        const RAW_CTRL7_OFF: usize = MODELS * BUCKETS * 2 + 256 + 64 + 256 + 64 + 64;
        (0..512u16)
            .filter(|&c| {
                let off = RAW_CTRL7_OFF + usize::from(c) * 2;
                u16::from_le_bytes([table_bytes[off], table_bytes[off + 1]]) != 0
            })
            .collect()
    }

    #[test]
    fn gpu_tables_from_bytes_validates() {
        let host = crate::host::host().expect("host D3D12 device");
        assert!(GpuTables::from_bytes(&host.device, BUILTIN_TABLES).is_ok());
        assert!(GpuTables::from_bytes(&host.device, &synthetic_tables()).is_ok());
        assert!(GpuTables::from_bytes(&host.device, &[0u8; 32]).is_err());

        let mut bad_tiling = synthetic_tables();
        bad_tiling[0] = 1;
        assert!(GpuTables::from_bytes(&host.device, &bad_tiling).is_err());
    }

    #[test]
    fn all_six_formats_synthetic_tables() {
        let raw_tables = synthetic_tables();
        let cpu_tables = Tables::parse(&raw_tables).unwrap();
        let ctrl_codes = valid_ctrl7_codes(&raw_tables);

        for bc in [1u8, 3, 4, 5, 6, 7] {
            let (_, _, block_bytes) = widths_and_per(bc);
            let blocks = 64;
            let records = make_records(bc, blocks, 0x1234_5678_9abc_def0 + u64::from(bc), &ctrl_codes);
            let src = encode_stream(&raw_tables, bc, &records, true);

            let mut cpu_out = vec![0u8; blocks * block_bytes];
            let mut gpu_out = vec![0u8; blocks * block_bytes];
            bcpack::decode(bc, &src, &mut cpu_out, &cpu_tables).unwrap();
            decode_gpu_with_tables(bc, &src, &mut gpu_out, &raw_tables).unwrap();
            assert_eq!(gpu_out, cpu_out, "BC{bc} mismatch on synthetic tables");
        }
    }

    #[test]
    fn all_six_formats_real_shipped_tables_and_dictionaries() {
        let cpu_tables = Tables::builtin().unwrap();
        let ctrl_codes = valid_ctrl7_codes(BUILTIN_TABLES);
        assert!(!ctrl_codes.is_empty());

        for bc in [1u8, 3, 4, 5, 6, 7] {
            let (_, _, block_bytes) = widths_and_per(bc);
            let blocks = 80;
            let records = make_records(bc, blocks, 0xfedc_ba98_7654_3210 + u64::from(bc) * 17, &ctrl_codes);
            let src = encode_stream(BUILTIN_TABLES, bc, &records, false);

            let mut cpu_out = vec![0u8; blocks * block_bytes];
            let mut gpu_out = vec![0u8; blocks * block_bytes];
            bcpack::decode(bc, &src, &mut cpu_out, &cpu_tables).unwrap();
            decode_gpu(bc, &src, &mut gpu_out).unwrap();
            assert_eq!(gpu_out, cpu_out, "BC{bc} mismatch on shipped bcpack-tables.bin");
        }
    }

    #[test]
    fn bc6h_all_14_modes() {
        let cpu_tables = Tables::builtin().unwrap();
        let mut records = Vec::new();
        let mut seed = 0x55aa_33cc_9966_1122u64;
        let mut next = || -> u32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };
        let (widths, per, block_bytes) = widths_and_per(6);

        // 14 modes * 10 variations (partitions, 1-subset <-> 2-subset history refs, dictionary refs).
        for round in 0..10usize {
            for mode_m1 in 0..14u8 {
                let mut r: Vec<u8> = (0..widths.len() * per)
                    .map(|k| (next() % (1u32 << (widths[k / per] + 1))) as u8)
                    .collect();
                r[30] = mode_m1;
                r[31] = ((round * 3 + usize::from(mode_m1)) % 32) as u8;
                records.push(r);
            }
        }

        let src = encode_stream(BUILTIN_TABLES, 6, &records, true);
        let mut cpu_out = vec![0u8; records.len() * block_bytes];
        let mut gpu_out = vec![0u8; records.len() * block_bytes];
        bcpack::decode(6, &src, &mut cpu_out, &cpu_tables).unwrap();
        decode_gpu(6, &src, &mut gpu_out).unwrap();
        assert_eq!(gpu_out, cpu_out, "BC6H modes 1..=14 mismatch");
    }

    #[test]
    fn bc7_all_8_modes_rotations_pbits_subsets() {
        let raw_tables = synthetic_tables();
        let cpu_tables = Tables::parse(&raw_tables).unwrap();
        let ctrl_codes = valid_ctrl7_codes(&raw_tables);
        // Includes modes 0, 1, 2, 3, 4 (rot 0..=3, idxSel 0..=1), 5 (rot 0..=3), 6, 7,
        // plus non-zero rot bits on modes 0 and 6.
        let blocks = ctrl_codes.len() * 2;
        let records = make_records(7, blocks, 0x7777_1234_5678_9abcu64, &ctrl_codes);
        let src = encode_stream(&raw_tables, 7, &records, true);

        let mut cpu_out = vec![0u8; blocks * 16];
        let mut gpu_out = vec![0u8; blocks * 16];
        bcpack::decode(7, &src, &mut cpu_out, &cpu_tables).unwrap();
        decode_gpu_with_tables(7, &src, &mut gpu_out, &raw_tables).unwrap();
        assert_eq!(gpu_out, cpu_out, "BC7 modes 0..=7 / rotations / p-bits mismatch");
    }

    #[test]
    fn history_wraparound_and_long_fragments() {
        let raw_tables = synthetic_tables();
        let cpu_tables = Tables::parse(&raw_tables).unwrap();
        let ctrl_codes = valid_ctrl7_codes(&raw_tables);

        // 360 blocks > 256 (ring buffer wraparound) and > 256 payload bytes (long fragments).
        let blocks = 360;
        for bc in [1u8, 3, 4, 5, 6, 7] {
            let (_, _, block_bytes) = widths_and_per(bc);
            let records = make_records(bc, blocks, 0xdead_beef_cafe_0000 + u64::from(bc), &ctrl_codes);
            let src = encode_stream(&raw_tables, bc, &records, false);

            let mut cpu_out = vec![0u8; blocks * block_bytes];
            let mut gpu_out = vec![0u8; blocks * block_bytes];
            bcpack::decode(bc, &src, &mut cpu_out, &cpu_tables).unwrap();
            decode_gpu_with_tables(bc, &src, &mut gpu_out, &raw_tables).unwrap();
            assert_eq!(gpu_out, cpu_out, "BC{bc} history wraparound / long fragment mismatch");
        }
    }

    #[test]
    fn multi_stream_batched_decode() {
        let cpu_tables = Tables::builtin().unwrap();
        let ctrl_codes = valid_ctrl7_codes(BUILTIN_TABLES);

        let specs: &[(u8, usize)] = &[
            (7, 17),
            (1, 300),
            (6, 42),
            (4, 1),
            (3, 128),
            (5, 65),
            (7, 260),
            (6, 14),
            (1, 3),
        ];

        let mut streams = Vec::new();
        let mut expected = Vec::new();
        for (idx, &(bc, blocks)) in specs.iter().enumerate() {
            let (_, _, block_bytes) = widths_and_per(bc);
            let records = make_records(bc, blocks, 0x1000_0001 * (idx as u64 + 1), &ctrl_codes);
            let src = encode_stream(BUILTIN_TABLES, bc, &records, idx % 2 == 0);
            let mut cpu_out = vec![0u8; blocks * block_bytes];
            bcpack::decode(bc, &src, &mut cpu_out, &cpu_tables).unwrap();
            streams.push((bc, blocks, src));
            expected.push(cpu_out);
        }

        let jobs: Vec<StreamJob<'_>> = streams
            .iter()
            .map(|(bc, blocks, src)| StreamJob { bc: *bc, src, blocks: *blocks })
            .collect();
        let got = decode_gpu_batch(&jobs, BUILTIN_TABLES).unwrap();
        assert_eq!(got.len(), expected.len());
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(g, e, "batch stream {i} (BC{}, {} blocks) mismatch", specs[i].0, specs[i].1);
        }
    }

    #[test]
    fn malformed_and_error_streams_do_not_hang() {
        let raw_tables = synthetic_tables();
        let ctrl_codes = valid_ctrl7_codes(&raw_tables);

        let records = make_records(4, 64, 0x4444_5555, &ctrl_codes);
        let valid_src = encode_stream(&raw_tables, 4, &records, false);
        let mut dst = vec![0u8; 64 * 8];

        // Truncated stream -> Err, no hang.
        assert!(decode_gpu_with_tables(4, &valid_src[..valid_src.len() / 2], &mut dst, &raw_tables).is_err());
        // Empty stream -> Err, no hang.
        assert!(decode_gpu_with_tables(4, &[], &mut dst, &raw_tables).is_err());
        // Headers promising huge payload -> Err, no hang.
        let bad_headers = [[0x30u8; 8], [0xffu8; 8]].concat();
        assert!(decode_gpu_with_tables(4, &bad_headers, &mut dst[..8], &raw_tables).is_err());
        // Invalid format / bad dst sizes.
        assert!(decode_gpu_with_tables(2, &valid_src, &mut dst, &raw_tables).is_err());
        assert!(decode_gpu_with_tables(4, &valid_src, &mut dst[..12], &raw_tables).is_err());
        assert!(decode_gpu_with_tables(4, &valid_src, &mut [], &raw_tables).is_err());

        // BC6H mode out of range (s[30] = 14 -> mode 15) -> Err.
        let mut bad_bc6 = vec![0u8; 32];
        bad_bc6[30] = 14;
        let bad_bc6_src = encode_stream(&raw_tables, 6, &[bad_bc6], false);
        let mut bc6_dst = [0u8; 16];
        assert!(decode_gpu_with_tables(6, &bad_bc6_src, &mut bc6_dst, &raw_tables).is_err());

        // BC7 unused control code (code 511 -> ctrl = 0) -> Err.
        let mut bad_bc7 = vec![0u8; 32];
        bad_bc7[30] = 31;
        bad_bc7[31] = 15;
        let bad_bc7_src = encode_stream(&raw_tables, 7, &[bad_bc7], false);
        let mut bc7_dst = [0u8; 16];
        assert!(decode_gpu_with_tables(7, &bad_bc7_src, &mut bc7_dst, &raw_tables).is_err());

        // GPU runtime remains healthy after errors.
        let cpu_tables = Tables::parse(&raw_tables).unwrap();
        let mut cpu_out = vec![0u8; 64 * 8];
        bcpack::decode(4, &valid_src, &mut cpu_out, &cpu_tables).unwrap();
        decode_gpu_with_tables(4, &valid_src, &mut dst, &raw_tables).unwrap();
        assert_eq!(dst, cpu_out);
    }

    /// Batched GPU vs CPU throughput benchmark:
    /// `cargo test --release -p victory-gpu gpu_bcpack::tests::gpu_bcpack_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "benchmark"]
    fn gpu_bcpack_bench() {
        let cpu_tables = Tables::builtin().unwrap();
        let ctrl_codes = valid_ctrl7_codes(BUILTIN_TABLES);
        let num_streams = 64usize;
        let blocks_per_stream = 2048usize;
        let runs = 10u32;

        for bc in [1u8, 4, 6, 7] {
            let (_, _, block_bytes) = widths_and_per(bc);
            let total_bytes = num_streams * blocks_per_stream * block_bytes;
            let streams: Vec<Vec<u8>> = (0..num_streams)
                .map(|s| {
                    let recs = make_records(
                        bc,
                        blocks_per_stream,
                        0x9e37_79b9_7f4a_7c15 ^ ((s as u64 + 1) << 32) ^ u64::from(bc),
                        &ctrl_codes,
                    );
                    encode_stream(BUILTIN_TABLES, bc, &recs, false)
                })
                .collect();
            let jobs: Vec<StreamJob<'_>> = streams
                .iter()
                .map(|src| StreamJob { bc, src, blocks: blocks_per_stream })
                .collect();

            // Warmup & verify bit-for-bit equality before timing.
            let gpu_check = decode_gpu_batch(&jobs, BUILTIN_TABLES).unwrap();
            let mut cpu_buf = vec![0u8; blocks_per_stream * block_bytes];
            for (i, src) in streams.iter().enumerate() {
                bcpack::decode(bc, src, &mut cpu_buf, &cpu_tables).unwrap();
                assert_eq!(gpu_check[i], cpu_buf);
            }

            let t_cpu = std::time::Instant::now();
            for _ in 0..runs {
                for src in &streams {
                    bcpack::decode(bc, std::hint::black_box(src), &mut cpu_buf, &cpu_tables).unwrap();
                }
            }
            let cpu_secs = t_cpu.elapsed().as_secs_f64() / f64::from(runs);

            let t_gpu = std::time::Instant::now();
            for _ in 0..runs {
                let out = decode_gpu_batch(std::hint::black_box(&jobs), BUILTIN_TABLES).unwrap();
                std::hint::black_box(out);
            }
            let gpu_secs = t_gpu.elapsed().as_secs_f64() / f64::from(runs);

            let cpu_mbs = total_bytes as f64 / cpu_secs / 1e6;
            let gpu_mbs = total_bytes as f64 / gpu_secs / 1e6;
            println!(
                "BC{bc} ({num_streams}x{blocks_per_stream} blks, {:.2} MB): CPU {cpu_mbs:.1} MB/s ({:.2} ms) | GPU {gpu_mbs:.1} MB/s ({:.2} ms) | {:.2}x",
                total_bytes as f64 / 1e6,
                cpu_secs * 1e3,
                gpu_secs * 1e3,
                gpu_mbs / cpu_mbs,
            );
        }
    }

    #[test]
    fn barrier_scoping_selects_enhanced_or_legacy_scoped() {
        assert_eq!(choose_barrier_kind(true, true), BarrierKind::Enhanced);
        assert_eq!(choose_barrier_kind(true, false), BarrierKind::LegacyScoped);
        assert_eq!(choose_barrier_kind(false, true), BarrierKind::LegacyScoped);
        assert_eq!(choose_barrier_kind(false, false), BarrierKind::LegacyScoped);
    }

    #[test]
    fn double_buffered_async_batches_overlap_across_both_slots() {
        let cpu_tables = Tables::builtin().unwrap();
        let ctrl_codes = valid_ctrl7_codes(BUILTIN_TABLES);

        let make_stream = |bc: u8, blocks: usize, seed: u64| {
            let (_, _, block_bytes) = widths_and_per(bc);
            let recs = make_records(bc, blocks, seed, &ctrl_codes);
            let src = encode_stream(BUILTIN_TABLES, bc, &recs, true);
            let mut cpu_out = vec![0u8; blocks * block_bytes];
            bcpack::decode(bc, &src, &mut cpu_out, &cpu_tables).unwrap();
            (bc, blocks, src, cpu_out)
        };

        let a0 = make_stream(7, 96, 0x1111_2222);
        let a1 = make_stream(1, 128, 0x3333_4444);
        let b0 = make_stream(6, 70, 0x5555_6666);
        let b1 = make_stream(5, 80, 0x7777_8888);

        let jobs_a = [
            StreamJob { bc: a0.0, src: &a0.2, blocks: a0.1 },
            StreamJob { bc: a1.0, src: &a1.2, blocks: a1.1 },
        ];
        let jobs_b = [
            StreamJob { bc: b0.0, src: &b0.2, blocks: b0.1 },
            StreamJob { bc: b1.0, src: &b1.2, blocks: b1.1 },
        ];

        // Submit BOTH batches before waiting on either -> proves double-buffered overlap.
        let pending_a = submit_gpu_batch_async(&jobs_a, BUILTIN_TABLES).unwrap();
        let pending_b = submit_gpu_batch_async(&jobs_b, BUILTIN_TABLES).unwrap();
        assert_ne!(
            pending_a.slot(),
            pending_b.slot(),
            "consecutive async batches must occupy distinct double-buffered flight slots"
        );

        let out_a = pending_a.wait().unwrap();
        let out_b = pending_b.wait().unwrap();
        assert_eq!(out_a[0], a0.3);
        assert_eq!(out_a[1], a1.3);
        assert_eq!(out_b[0], b0.3);
        assert_eq!(out_b[1], b1.3);
    }

    #[test]
    fn concurrent_single_stream_callers_coalesce_and_isolate_errors() {
        let cpu_tables = Tables::builtin().unwrap();
        let ctrl_codes = valid_ctrl7_codes(BUILTIN_TABLES);

        let specs: &[(u8, usize, bool)] = &[
            (1, 120, false),
            (3, 96, false),
            (4, 150, false),
            (5, 84, false),
            (6, 56, false),
            (7, 110, false),
            (4, 64, true), // intentionally truncated -> must Err without poisoning siblings
            (7, 90, false),
        ];

        let payloads: Vec<(u8, usize, Vec<u8>, Option<Vec<u8>>)> = specs
            .iter()
            .enumerate()
            .map(|(i, &(bc, blocks, corrupt))| {
                let (_, _, block_bytes) = widths_and_per(bc);
                let recs = make_records(bc, blocks, 0xaaaa_0001 * (i as u64 + 1), &ctrl_codes);
                let mut src = encode_stream(BUILTIN_TABLES, bc, &recs, i % 2 == 0);
                if corrupt {
                    src.truncate(src.len() / 3);
                    (bc, blocks * block_bytes, src, None)
                } else {
                    let mut cpu = vec![0u8; blocks * block_bytes];
                    bcpack::decode(bc, &src, &mut cpu, &cpu_tables).unwrap();
                    (bc, blocks * block_bytes, src, Some(cpu))
                }
            })
            .collect();

        std::thread::scope(|s| {
            let handles: Vec<_> = payloads
                .iter()
                .map(|(bc, out_len, src, expected)| {
                    let bc = *bc;
                    let out_len = *out_len;
                    s.spawn(move || {
                        let mut dst = vec![0u8; out_len];
                        let res = decode_gpu(bc, src, &mut dst);
                        match expected {
                            Some(exp) => {
                                res.expect("valid concurrent stream must succeed");
                                assert_eq!(&dst, exp, "BC{bc} concurrent decode mismatch");
                            }
                            None => {
                                assert!(res.is_err(), "corrupt stream must fail cleanly");
                            }
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    }
}
