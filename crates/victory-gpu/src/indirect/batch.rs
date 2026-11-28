//! Clamps of many executions in one pass: the arguments of every indirect
//! execution between two points where they may change are bounded together,
//! at the first of them.
//!
//! The executions that come after it are not known yet when the pass is
//! recorded, so it reads what to clamp from a table of jobs in upload
//! memory, which the CPU goes on filling while the list is recorded: the
//! pass is as wide as the table and its threads past the count do nothing.
//! An execution that asks for a record already in the table (the title's
//! tile classes) reads the same clamped one.
//!
//! What the records charge the list's budget is as for records clamped
//! alone, times the executions that read them.

use windows::Win32::Graphics::Direct3D12::*;

use super::{
    Clamp, ClampBarrierPlan, PreInputUav, enhanced_buffer_barrier, uav_barrier, uav_barrier_for,
    uav_capable,
};
use crate::host::transition;

/// The jobs of one pass.
pub(crate) const JOBS: u32 = 512;
/// The bytes of its table in upload memory: the count, then the jobs.
pub(crate) const TABLE: u64 = 16 + 16 * JOBS as u64;
/// The bytes of a clamped record's place in the scratch buffer.
pub(crate) const SLOT: u32 = 32;
/// The source buffers a pass can read.
pub(crate) const SOURCES: usize = 8;
/// The most executions charged by one job: the most a record costs (2^26)
/// as often still fits 32 bits.
pub(crate) const TIMES: u32 = 31;

pub(crate) const SOURCE: &str = r#"
#if BUDGET_TRACK
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), SRV(t5), SRV(t6), SRV(t7), SRV(t8), UAV(u0), UAV(u1)"
#else
#define RS "SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), SRV(t5), SRV(t6), SRV(t7), SRV(t8), UAV(u0)"
#endif
ByteAddressBuffer src0 : register(t0);
ByteAddressBuffer src1 : register(t1);
ByteAddressBuffer src2 : register(t2);
ByteAddressBuffer src3 : register(t3);
ByteAddressBuffer src4 : register(t4);
ByteAddressBuffer src5 : register(t5);
ByteAddressBuffer src6 : register(t6);
ByteAddressBuffer src7 : register(t7);
// The count, 12 bytes of nothing, then per job: source (bits 0-2), kind
// (4-7), words (8-11) and the executions that read it (16-31); the
// record's offset in the source; its place in the scratch buffer; nothing.
ByteAddressBuffer jobs : register(t8);
RWByteAddressBuffer dst_args : register(u0);
#if BUDGET_TRACK
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
#endif
static const uint GROUPS = 1u << 20;
#if BUDGET_TRACK
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
#endif
uint load(uint source, uint at) {
    if (source == 0) return src0.Load(at);
    if (source == 1) return src1.Load(at);
    if (source == 2) return src2.Load(at);
    if (source == 3) return src3.Load(at);
    if (source == 4) return src4.Load(at);
    if (source == 5) return src5.Load(at);
    if (source == 6) return src6.Load(at);
    return src7.Load(at);
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    uint count = jobs.Load(0);
#if !BUDGET_TRACK
    if (id.x >= count) {
        return;
    }
    uint4 job = jobs.Load4(16 + id.x * 16);
    uint kind = (job.x >> 4) & 15;
    uint words = min((job.x >> 8) & 15, 5);
    uint record[5] = {0, 0, 0, 0, 0};
    for (uint w = 0; w < words; w++) {
        record[w] = load(job.x & 7, job.y + w * 4);
    }
    if (kind <= 1) {
        if (record[0] >= 1u << 23) {
            record[0] = 0;
            record[1] = 0;
        }
        record[1] = min(record[1], (1u << 26) / max(record[0], 1u));
    } else if (kind == 2) {
        uint3 raw = uint3(record[0], record[1], record[2]);
        uint3 g = min(raw, uint3(raw.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        record[0] = g.x;
        record[1] = g.y;
        record[2] = g.z;
    } else if (kind == 11) {
        // 1D indirect mesh dispatch (`DispatchMeshX`): 1 word in, 3 words (x, 1, 1) out.
        uint x = record[0];
        if (x == 0u || x > 65535u) {
            record[0] = 0;
            record[1] = 0;
            record[2] = 0;
        } else {
            record[0] = x;
            record[1] = 1;
            record[2] = 1;
        }
    } else if (kind == 12) {
        // Spread indirect compute dispatch: 3 words in, 5 words [row, asked_x, gx, gy, gz] out.
        uint3 raw = uint3(record[0], record[1], record[2]);
        uint row = 0;
        uint asked_x = raw.x;
        uint3 g;
        if (raw.x > 65535u && raw.y >= 1u && raw.y <= 65535u && raw.z == 1u) {
            uint rows = (raw.x + 65534u) / 65535u;
            row = (raw.x + rows - 1u) / rows;
            g = uint3(row, raw.y, rows);
        } else {
            g = min(raw, uint3(65535u, 65535u, 65535u));
        }
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        record[0] = row;
        record[1] = asked_x;
        record[2] = g.x;
        record[3] = g.y;
        record[4] = g.z;
    } else {
        if (record[0] > 65535u) {
            record[0] = 0;
            record[1] = 0;
            record[2] = 0;
        }
        record[1] = min(record[1], 1u);
        record[2] = min(record[2], 1u);
    }
    uint out_words = (kind == 11) ? 3u : (kind == 12) ? 5u : words;
    for (uint s = 0; s < out_words; s++) {
        dst_args.Store(job.z + s * 4, record[s]);
    }
#else
    uint first = gid.x * 64u;
    if (first >= count) {
        return;
    }
    bool active = id.x < count;
    uint4 job = uint4(0, 0, 0, 0);
    uint record[5] = {0, 0, 0, 0, 0};
    uint cost = 0;
    uint asked = 0;
    uint kind = 0;
    uint words = 0;
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    // As the clamp of one execution (`super::SOURCE`) bounds them.
    if (active) {
        job = jobs.Load4(16 + id.x * 16);
        kind = (job.x >> 4) & 15;
        words = min((job.x >> 8) & 15, 5);
        for (uint w = 0; w < words; w++) {
            record[w] = load(job.x & 7, job.y + w * 4);
        }
        if (kind <= 1) {
            if (record[0] >= 1u << 23) {
                record[0] = 0;
                record[1] = 0;
            }
            record[1] = min(record[1], (1u << 26) / max(record[0], 1u));
            cost = record[0] * record[1];
        } else if (kind == 2) {
            uint3 raw = uint3(record[0], record[1], record[2]);
            asked = max(raw.x, max(raw.y, raw.z));
            uint3 g = min(raw, uint3(raw.x <= WIDEST ? WIDEST : 65535u, 65535u, 65535u));
            if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            record[0] = g.x;
            record[1] = g.y;
            record[2] = g.z;
        } else if (kind == 11) {
            // 1D indirect mesh dispatch (`DispatchMeshX`): 1 word in, 3 words (x, 1, 1) out.
            uint x = record[0];
            if (x == 0u || x > 65535u) {
                record[0] = 0;
                record[1] = 0;
                record[2] = 0;
            } else {
                asked = x;
                record[0] = x;
                record[1] = 1;
                record[2] = 1;
            }
            cost = record[0] * 128u;
        } else if (kind == 12) {
            // Spread indirect compute dispatch: 3 words in, 5 words [row, asked_x, gx, gy, gz] out.
            uint3 raw = uint3(record[0], record[1], record[2]);
            asked = max(raw.x, max(raw.y, raw.z));
            uint row = 0;
            uint asked_x = raw.x;
            uint3 g;
            if (raw.x > 65535u && raw.y >= 1u && raw.y <= 65535u && raw.z == 1u) {
                uint rows = (raw.x + 65534u) / 65535u;
                row = (raw.x + rows - 1u) / rows;
                g = uint3(row, raw.y, rows);
            } else {
                g = min(raw, uint3(65535u, 65535u, 65535u));
            }
            if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
                g = uint3(0, 0, 0);
            }
            cost = g.x * g.y * g.z * 64u;
            record[0] = row;
            record[1] = asked_x;
            record[2] = g.x;
            record[3] = g.y;
            record[4] = g.z;
        } else {
            if (record[0] > 65535u) {
                record[0] = 0;
                record[1] = 0;
                record[2] = 0;
            }
            asked = record[0];
            record[1] = min(record[1], 1u);
            record[2] = min(record[2], 1u);
            cost = record[0] * record[1] * record[2] * 128u;
        }
        cost *= job.x >> 16;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind >= 2) {
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    uint n = min(64u, count - first);
    if (gi == 0) {
        uint group_cost = 0;
        for (uint ci = 0; ci < n; ci++) {
            group_cost |= costs[ci];
        }
        if (group_cost == 0) {
            for (uint k = 0; k < n; k++) {
                fits[k] = 1;
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
    if (gi == 0 && all_asked > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    if (fits[gi] == 0) {
        if (kind == 12) {
            record[2] = 0;
            record[3] = 0;
            record[4] = 0;
        } else {
            record[0] = 0;
            record[1] = 0;
            if (kind > 1) {
                record[2] = 0;
            }
        }
    }
    uint out_words = (kind == 11) ? 3u : (kind == 12) ? 5u : words;
    for (uint s = 0; s < out_words; s++) {
        dst_args.Store(job.z + s * 4, record[s]);
    }
#endif
}
"#;

/// The words of a record of `kind` (PC argument numbering).
pub(crate) fn words(kind: u32) -> Option<u32> {
    super::min_stride(kind).map(|bytes| bytes / 4)
}

/// A job's first word.
pub(crate) fn job(source: u32, kind: u32, times: u32) -> u32 {
    source | (kind << 4) | (words(kind).unwrap_or(0) << 8) | (times << 16)
}

/// Packs `(source, offset, kind)` into a single key for inline dedup scans.
#[inline]
fn pack_key(source: u32, offset: u32, kind: u32) -> u64 {
    (u64::from(source) << 40) | (u64::from(kind) << 32) | u64::from(offset)
}

/// The jobs of a pass as the CPU fills them ([`SOURCE`]), and where it
/// keeps which record is where. Uses inline fixed-capacity arrays so opening
/// a batch incurs zero heap allocations.
pub(crate) struct Table {
    /// The table in upload memory ([`TABLE`] bytes).
    at: usize,
    /// The place of the first clamped record in the scratch buffer.
    base: u32,
    /// The host copies it can read, by number ([`crate::shadow::Shadow::id`];
    /// 0: none).
    sources: [u64; SOURCES],
    /// Number of active jobs (`0..=JOBS`).
    len: u32,
    /// Last job count flushed to WC upload memory `table[0]`.
    flushed_len: u32,
    /// Packed `(source, kind, offset)` key for each active job `0..len`.
    keys: [u64; JOBS as usize],
    /// How many executions each job is charged for so far (`1..=TIMES`).
    times: [u8; JOBS as usize],
}

impl Default for Table {
    fn default() -> Self {
        Self {
            at: 0,
            base: 0,
            sources: [0; SOURCES],
            len: 0,
            flushed_len: 0,
            keys: [0; JOBS as usize],
            times: [0; JOBS as usize],
        }
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        self.flush();
    }
}

impl Table {
    /// A table at `at`, with no job yet.
    ///
    /// # Safety
    /// `at` is [`TABLE`] bytes of mapped upload memory, 4-byte aligned, that
    /// live until the list that reads them is recorded and is not run
    /// before.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) unsafe fn new(at: *mut u8, base: u32, sources: [u64; SOURCES]) -> Table {
        // SAFETY: the caller's.
        unsafe { at.cast::<u32>().write_volatile(0) };
        Table { at: at as usize, base, sources, ..Table::default() }
    }

    /// Flushes any prior batch count and rebinds this `Table` in place to a new
    /// upload-memory slot without re-zeroing the 4.6 KiB `keys`/`times` arrays.
    ///
    /// # Safety
    /// Same safety contract as [`Table::new`].
    pub(crate) unsafe fn reset(&mut self, at: *mut u8, base: u32, sources: [u64; SOURCES]) {
        self.flush();
        // SAFETY: the caller's.
        unsafe { at.cast::<u32>().write_volatile(0) };
        self.at = at as usize;
        self.base = base;
        self.sources = sources;
        self.len = 0;
        self.flushed_len = 0;
    }

    /// Flushes the final job count to WC upload memory `table[0]` once.
    #[inline]
    pub(crate) fn flush(&mut self) {
        if self.at != 0 && self.len != self.flushed_len {
            // SAFETY: `self.at` is mapped upload memory alive until the list executes.
            unsafe { (self.at as *mut u32).write_volatile(self.len) };
            self.flushed_len = self.len;
        }
    }

    /// Detaches this table from its current upload-memory backing after
    /// flushing the final job count (so the struct can be kept idle for reuse).
    #[inline]
    pub(crate) fn finish(&mut self) {
        self.flush();
        self.at = 0;
        self.len = 0;
        self.flushed_len = 0;
    }

    /// Appends or deduplicates a job without flushing WC `table[0]` on every
    /// call (`CPU-29`); callers MUST call [`Table::flush`], [`Table::finish`],
    /// or drop the table before the command list executes on the GPU.
    pub(crate) fn place_deferred(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        let source = self.sources.iter().position(|&s| s == id)? as u32;
        let offset = u32::try_from(offset).ok().filter(|o| o.is_multiple_of(4))?;
        offset.checked_add(words(kind)? * 4)?;
        let key = pack_key(source, offset, kind);
        let table = self.at as *mut u32;
        let place = |i: u32| u64::from(self.base + i * SLOT);
        let track_budget = super::budget_enabled();
        // Reverse scan hits the most recently allocated job for `key` first so
        // once a slot reaches `TIMES`, subsequent calls find the overflow slot.
        for i in (0..self.len as usize).rev() {
            if self.keys[i] == key {
                if !track_budget {
                    // Release shader (`!BUDGET_TRACK`) ignores `times`, so
                    // duplicate jobs need zero WC memory writes.
                    return Some(place(i as u32));
                }
                if u32::from(self.times[i]) < TIMES {
                    self.times[i] += 1;
                    // SAFETY: the job's own word of the table ([`Table::new`]).
                    unsafe {
                        table
                            .add(4 + 4 * i)
                            .write_volatile(job(source, kind, u32::from(self.times[i])));
                    }
                    return Some(place(i as u32));
                }
                break;
            }
        }
        let i = self.len;
        if i == JOBS {
            return None;
        }
        self.keys[i as usize] = key;
        self.times[i as usize] = 1;
        self.len = i + 1;
        // SAFETY: inside the table: fewer than [`JOBS`] jobs before it.
        unsafe {
            let to = table.add(4 + 4 * i as usize);
            to.write_volatile(job(source, kind, 1));
            to.add(1).write_volatile(offset);
            to.add(2).write_volatile(self.base + i * SLOT);
            to.add(3).write_volatile(0);
        }
        Some(place(i))
    }

    /// The place in the scratch buffer of the record of `kind` at `offset`
    /// of the host copy `id`, once the pass has run: a job of it is added if
    /// there is none. `None` if the pass cannot read that copy or has no
    /// room for another job. Immediately flushes `table[0]` for callers that
    /// submit before dropping `Table`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn place(&mut self, id: u64, offset: u64, kind: u32) -> Option<u64> {
        let res = self.place_deferred(id, offset, kind)?;
        self.flush();
        Some(res)
    }
}

/// Records the pass over the table at `jobs` (a host address) into `dst`,
/// which is in state `before` and left readable as indirect arguments: after
/// earlier shader writes (the title's argument passes), before what reads the
/// clamped records. When `enhanced` is `Some`, uses Enhanced Barriers keeping
/// `dst` in `COMMON` and scoping pre-clamp sync to `COMPUTE_SHADING` (or
/// `ALL_SHADING` when `graphics_uav_dirty` is true).
#[allow(clippy::too_many_arguments)]
pub(super) fn record(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    enhanced: Option<&ID3D12GraphicsCommandList7>,
    graphics_uav_dirty: bool,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: Option<(Option<&ID3D12Resource>, u64)>,
    sources: [u64; SOURCES],
    source_resources: &[&ID3D12Resource],
    jobs: u64,
    barrier_plan: ClampBarrierPlan,
) {
    if let Some(list7) = enhanced {
        let sync_before = if graphics_uav_dirty {
            D3D12_BARRIER_SYNC_ALL_SHADING
        } else {
            D3D12_BARRIER_SYNC_COMPUTE_SHADING
        };
        let mut buf_barriers: [D3D12_BUFFER_BARRIER; SOURCES + 2] =
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
                for &res in source_resources {
                    if uav_capable(res) {
                        buf_barriers[buf_len] = enhanced_buffer_barrier(
                            res,
                            sync_before,
                            D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                            D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                        );
                        buf_len += 1;
                    }
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
            for (i, source) in sources.into_iter().enumerate() {
                l.SetComputeRootShaderResourceView(i as u32, source);
            }
            l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
            l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
            if let Some((_, budget_gpu)) = budget {
                l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget_gpu);
            }
            l.Dispatch(JOBS.div_ceil(64), 1, 1);
            list7.Barrier(&post_group);
        }
        return;
    }

    let mut first: [D3D12_RESOURCE_BARRIER; SOURCES + 2] =
        std::array::from_fn(|_| D3D12_RESOURCE_BARRIER::default());
    let mut first_len = 0usize;
    match barrier_plan.pre_inputs {
        PreInputUav::None => {}
        PreInputUav::Global => {
            first[first_len] = uav_barrier();
            first_len += 1;
        }
        PreInputUav::Scoped => {
            for &res in source_resources {
                first[first_len] = uav_barrier_for(res);
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
    let after = transition(
        dst,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    );
    // SAFETY: an open list; our root signature and pipeline; live buffers:
    // the sources and the table are read in the common state (promoted).
    unsafe {
        l.ResourceBarrier(&first[..first_len]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for (i, source) in sources.into_iter().enumerate() {
            l.SetComputeRootShaderResourceView(i as u32, source);
        }
        l.SetComputeRootShaderResourceView(SOURCES as u32, jobs);
        l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 1, dst.GetGPUVirtualAddress());
        if let Some((_, budget_gpu)) = budget {
            l.SetComputeRootUnorderedAccessView(SOURCES as u32 + 2, budget_gpu);
        }
        l.Dispatch(JOBS.div_ceil(64), 1, 1);
        // Only the clamped records have to transition to INDIRECT_ARGUMENT;
        // the next clamp in this list orders `budget` via a scoped UAV barrier.
        l.ResourceBarrier(&[after]);
    }
}
