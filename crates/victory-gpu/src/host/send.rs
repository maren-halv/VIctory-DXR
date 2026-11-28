//! What goes up to copies in the card's memory goes up as it was when its
//! submission was prepared (`VICTORY_SEND_OWN`, on by default; `0`: as
//! before).
//!
//! The sends were copies out of the copies' CPU sides, which the card makes
//! when it comes to the submission. By then later submissions are prepared
//! (two frames of them with `VICTORY_RELAY`), and their refreshes have
//! written the same CPU sides: the earlier submission's shaders then read a
//! later one's bytes. For what the title's CPU writes that is as on the
//! console (it does not write what the card still reads). But the title's
//! own buffer copies are commands in its lists, and those are done by us
//! when a submission is prepared: "a value to a buffer, a draw that reads
//! it" in one submission and the same with another value in the next is
//! ordinary, and both draws read the second value.
//!
//! So each submission's bytes are taken out of the CPU sides into memory of
//! the submission's own when they are taken from the dirty lists, and the
//! card copies them from there. Runs of a few words, which a refresh that
//! sends only what changed makes many of, go as pairs of offset and value,
//! stored by one dispatch per copy; the others as a copy each.

use super::*;
use crate::shadow::Shadow;

pub(super) fn own() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_OWN").is_none_or(|v| v != "0"))
}

/// Whether card-copy CPU sides can live in plain CPU memory instead of D3D12
/// `shared_buffer`s: true when every upload goes through `send::take` and
/// `upcheck` does not need a GPU-readable `Card::staging`.
pub(crate) fn cpu_staging_safe(host: &Host) -> bool {
    own() && !upcheck::on() && pass(host).is_some()
}

/// Runs up to this many bytes go as pairs.
const SHORT: usize = 256;

/// Whether a run of `len` bytes at offset `at` of a copy goes as pairs.
fn short(at: usize, len: usize) -> bool {
    len <= SHORT && at % 4 == 0 && len % 4 == 0
}

/// Whether a run of `len` bytes at offset `at` of a copy in the card's memory
/// is no copy command of its own, but pairs of one dispatch.
pub(crate) fn goes_as_words(at: usize, len: usize) -> bool {
    own() && short(at, len)
}

/// Whether the title's bytes that go in under flights' writes
/// (`Shadow::merge_under_writes`) go in by whole words
/// (`VICTORY_UNDER_WORDS=0`: by the bytes that changed, as before).
///
/// By bytes nearly every run (a float's three low bytes) was a copy command
/// of its own, and a submission with more than 4,096 of them stood until the
/// flights had landed: 4 to 14 ms of a steady frame (session play-1125-0203:
/// "more changed runs than a submission sends up"), for some 20 KB. Whole
/// words are pairs of one dispatch, and are not counted.
///
/// A word is taken to be the title's or the shaders', not shared: where the
/// title changed a byte of a word of which a flight's shader writes another
/// byte, that byte is as it was before the flight. The landing rests on the
/// same ([`crate::shadow::exact_back`]).
pub(crate) fn words_under() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| own() && std::env::var_os("VICTORY_UNDER_WORDS").is_none_or(|v| v != "0"))
}

const SCATTER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) { uint at; uint count; };
ByteAddressBuffer pairs : register(t0);
RWByteAddressBuffer card : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint2 p = pairs.Load2(at + id.x * 8u);
    card.Store(p.x, p.y);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(SCATTER, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        // SAFETY: the code lives through the calls; the root signature is
        // inside it.
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
            name(&pso, || "victory: changed words sent to the card".into());
            Some(Pass { root, pso })
        }
    })
    .as_ref()
}

/// Memory a submission's sends come from, mapped.
pub(super) struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
    last_used: u64,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, with one owner.
unsafe impl Send for Memory {}

/// What submissions that have run gave back.
static SPARE: Mutex<Vec<Memory>> = Mutex::new(Vec::new());
/// Monotonic recency counter for size-bucketed LRU eviction in [`SPARE`].
static SPARE_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Total bytes currently retained in [`SPARE`] (lock-free diagnostic accessor).
static SPARE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Number of buffers currently retained in [`SPARE`].
static SPARE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Default [`SPARE`] retention cap in MiB (`192 MiB`, holding two in-flight
/// frames of 40–75 MiB submissions so `send::memory` never churns BAR/L0
/// `CreateCommittedResource` allocations or triggers OS page-fault storms).
const DEFAULT_SPARE_MB: u64 = 192;

fn resolve_kept_bytes(env: Option<&str>) -> u64 {
    let mb = env
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_SPARE_MB);
    mb.saturating_mul(1 << 20)
}

/// At most this much is kept (of the card's memory with [`on_card`], or L0 upload memory).
/// Override with `VICTORY_SEND_SPARE_MB` (default `64` MiB).
fn kept() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        let env = std::env::var("VICTORY_SEND_SPARE_MB").ok();
        resolve_kept_bytes(env.as_deref())
    })
}

/// Total bytes currently held in `send::SPARE` (for `host.rs` `memory summary:`).
#[allow(dead_code)]
pub(crate) fn spare_bytes() -> u64 {
    SPARE_BYTES.load(Ordering::Relaxed)
}

/// `(bytes, buffer_count)` currently held in `send::SPARE`.
#[allow(dead_code)]
pub(crate) fn spare_info() -> (u64, usize) {
    (
        SPARE_BYTES.load(Ordering::Relaxed),
        SPARE_COUNT.load(Ordering::Relaxed),
    )
}

/// Selects indices to evict from `items` (`(size, last_used)`) until the
/// remaining total size fits within `cap`, using size-bucketed LRU retention:
/// excess duplicates within a power-of-two size bucket are evicted oldest-first
/// before evicting the sole resident of any active size bucket (globally LRU),
/// avoiding largest-first SHR sawtooths (`ARCH-05`).
fn select_spare_evictions(items: &[(u64, u64)], cap: u64) -> Vec<usize> {
    let mut total: u64 = items.iter().map(|&(s, _)| s).sum();
    if total <= cap {
        return Vec::new();
    }
    let mut alive = vec![true; items.len()];
    let mut evicted = Vec::new();
    while total > cap {
        // Count surviving entries per exact size bucket.
        let mut victim: Option<(usize, bool, u64)> = None;
        for (i, &(size, tick)) in items.iter().enumerate() {
            if !alive[i] {
                continue;
            }
            let dup = items
                .iter()
                .enumerate()
                .any(|(j, &(s2, _))| j != i && alive[j] && s2 == size);
            let better = match victim {
                None => true,
                // Prefer evicting duplicates over sole bucket residents, then oldest `last_used`.
                Some((_, v_dup, v_tick)) => (dup && !v_dup) || (dup == v_dup && tick < v_tick),
            };
            if better {
                victim = Some((i, dup, tick));
            }
        }
        let Some((idx, _, _)) = victim else {
            break;
        };
        alive[idx] = false;
        total = total.saturating_sub(items[idx].0);
        evicted.push(idx);
    }
    evicted
}

/// Whether the memory lies in the card's memory, written by the CPU through
/// the bus (`VICTORY_SEND_BAR=1`): the copies on the card's drawing queue, which
/// nothing overlaps, then cross no bus. In a captured frame they took 5.4 ms of
/// the card's 30 (342 copies), and take 0.4 this way; but filling it takes the
/// submitting thread 8-9 ms a frame instead of 3 (73-93 MB, run card-3 against
/// card-1), so it is a loss while the card waits for that thread: off by
/// default.
fn on_card() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SEND_BAR").is_some_and(|v| v == "1"))
}

fn bucket_size(size: u64) -> u64 {
    // Power-of-two below 4 MiB (512 KiB, 1, 2, 4 MiB), then 4 MiB steps so a
    // 33 MiB submission allocates 36 MiB instead of rounding up to 64 MiB.
    if size <= 4 << 20 {
        size.max(512 << 10).next_power_of_two()
    } else {
        size.next_multiple_of(4 << 20)
    }
}

fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        // Smallest fitting size bucket; break ties by most-recently-used for cache warmth.
        let fit = spare
            .iter()
            .enumerate()
            .filter(|(_, m)| m.size >= size)
            .min_by_key(|(_, m)| (m.size, std::cmp::Reverse(m.last_used)))
            .map(|(i, _)| i);
        if let Some(i) = fit {
            let mem = spare.swap_remove(i);
            SPARE_BYTES.fetch_sub(mem.size, Ordering::Relaxed);
            SPARE_COUNT.store(spare.len(), Ordering::Relaxed);
            return Some(mem);
        }
    }
    let alloc_t = std::time::Instant::now();
    let size = bucket_size(size);
    // Internal non-bindless BAR staging: skips `CREATE_NOT_RESIDENT` /
    // `MakeResident` / paging-fence stalls (identical to `upload_buffer`).
    let mapped = if on_card() { host.internal_mapped_card_buffer(size) } else { None };
    let (buffer, data) = match mapped {
        Some(made) => made,
        None => host.upload_buffer(size)?,
    };
    name(&buffer, || "bytes a submission sends up".into());
    STATS[4].fetch_add(1, Ordering::Relaxed);
    STATS[5].fetch_add(alloc_t.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Memory {
        buffer,
        data: data as usize,
        size,
        last_used: SPARE_TICK.fetch_add(1, Ordering::Relaxed),
    })
}

/// The submission has run.
pub(super) fn give_back(mut memory: Memory) {
    memory.last_used = SPARE_TICK.fetch_add(1, Ordering::Relaxed);
    let evicted: Vec<Memory> = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let added = memory.size;
        spare.push(memory);
        let mut total = SPARE_BYTES.fetch_add(added, Ordering::Relaxed).saturating_add(added);
        let cap = kept();
        let mut out = Vec::new();
        if total > cap {
            let meta: Vec<(u64, u64)> = spare.iter().map(|m| (m.size, m.last_used)).collect();
            let mut victims = select_spare_evictions(&meta, cap);
            victims.sort_unstable_by(|a, b| b.cmp(a));
            for idx in victims {
                let removed = spare.swap_remove(idx);
                total = total.saturating_sub(removed.size);
                SPARE_BYTES.fetch_sub(removed.size, Ordering::Relaxed);
                out.push(removed);
            }
        }
        SPARE_COUNT.store(spare.len(), Ordering::Relaxed);
        out
    };
    if !evicted.is_empty() {
        crate::host::retire_com_async(evicted);
    }
}

/// Since the last frame-rate line: `[0]` bytes copied, `[1]` runs copied,
/// `[2]` words stored, `[3]` microseconds taking them out of CPU sides,
/// `[4]` `memory()` cache-miss allocations, `[5]` microseconds spent allocating.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// One copy's sends: ranges copied (offset in the copy, in the memory,
/// bytes), and its pairs (where in the memory, how many).
struct Copy {
    runs: Vec<(usize, u64, usize)>,
    pairs: (u64, u32),
}

/// Where a submission's bytes are taken to.
pub(super) struct Own {
    memory: Memory,
    copies: Vec<Copy>,
}

/// Minimum total staging payload bytes (`long copies + scatter pairs * 8`)
/// required to fan out [`take`] across Tier-1 workers.
const PARALLEL_MIN_BYTES: u64 = 256 << 10;
/// Minimum scatter word-pair count required to fan out [`take`] across Tier-1 workers.
const PARALLEL_MIN_PAIRS: u64 = 4_096;
/// Target scatter pairs per parallel worker task (16 KiB output per task).
const WORDS_PER_TASK: usize = 2_048;

#[inline]
fn should_parallelize_take(total_bytes: u64, total_words: u64) -> bool {
    total_bytes.saturating_add(total_words.saturating_mul(8)) >= PARALLEL_MIN_BYTES
        || total_words >= PARALLEL_MIN_PAIRS
}

enum StagingTask {
    /// Pack `all_shorts[range_start..range_end]` from `from` into `to` as `[offset, word]` pairs.
    Pairs {
        from: usize,
        range_start: usize,
        range_end: usize,
        to: usize,
    },
}

/// Fills disjoint destination slices in the mapped staging buffer from long
/// byte-copy `moves` and short word-pair `pair_groups`. Runs inline when
/// `!parallel`, or fans out both copies and scatter-pair packing in a single
/// Tier-1 worker dispatch when `parallel` is true.
fn fill_staging(
    moves: &[(usize, usize, usize)],
    all_shorts: &[(usize, usize)],
    pair_groups: &[(usize, std::ops::Range<usize>, usize)],
    parallel: bool,
) {
    let pack_pairs = |from: usize, ranges: &[(usize, usize)], mut to: usize| {
        let mut dst = to as *mut [u32; 2];
        for &(a, b) in ranges {
            for offset in (a..b).step_by(4) {
                // SAFETY: inside the copy's CPU staging (sole writer during `take`)
                // and inside this task's disjoint slice of the mapped staging buffer.
                unsafe {
                    let word = ((from + offset) as *const u32).read_unaligned();
                    dst.write([offset as u32, word]);
                    dst = dst.add(1);
                }
            }
            to += ((b - a) / 4) * 8;
        }
        let _ = to;
    };

    if !parallel {
        for &(from, ref span, dst) in pair_groups {
            pack_pairs(from, &all_shorts[span.clone()], dst);
        }
        for &(src, dst, len) in moves {
            if len > 0 {
                // SAFETY: caller guarantees disjoint valid source and destination ranges.
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
            }
        }
        return;
    }

    // Stream long copies sequentially on the caller thread so Write-Combining
    // PCIe BAR bursts coalesce at full bus speed (splitting 64 KiB WC writes
    // across `land::each_under` workers regresses `send::take` 2 ms -> 24 ms).
    for &(src, dst, len) in moves {
        if len > 0 {
            // SAFETY: caller guarantees disjoint valid source and destination ranges.
            unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
        }
    }

    let mut tasks: Vec<StagingTask> = Vec::with_capacity(pair_groups.len());
    for &(from, ref span, dst_base) in pair_groups {
        let mut cursor = span.start;
        let mut to = dst_base;
        while cursor < span.end {
            let chunk_start = cursor;
            let mut words = 0usize;
            while cursor < span.end && (words == 0 || words < WORDS_PER_TASK) {
                let (a, b) = all_shorts[cursor];
                words += (b - a) / 4;
                cursor += 1;
            }
            tasks.push(StagingTask::Pairs {
                from,
                range_start: chunk_start,
                range_end: cursor,
                to,
            });
            to += words * 8;
        }
    }

    if tasks.len() < 2 {
        for StagingTask::Pairs { from, range_start, range_end, to } in tasks {
            pack_pairs(from, &all_shorts[range_start..range_end], to);
        }
        return;
    }

    let run_task = |i: usize| {
        let StagingTask::Pairs { from, range_start, range_end, to } = tasks[i];
        pack_pairs(from, &all_shorts[range_start..range_end], to);
    };
    crate::land::each_under(tasks.len(), &run_task);
}

/// Takes what `dirty` sends up out of the CPU sides.
pub(super) fn take(host: &Host, dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)]) -> Option<Own> {
    let began = std::time::Instant::now();
    pass(host)?;
    // Classify short (word-pair) vs long (copy) runs in a single pass per copy
    // using flat buffers to avoid nested per-buffer `Vec` allocations.
    let mut at = 0u64;
    let mut total_bytes = 0u64;
    let mut total_words = 0u64;
    let mut copies = Vec::with_capacity(dirty.len());
    let mut all_shorts: Vec<(usize, usize)> = Vec::new();
    let mut copy_short_spans: Vec<std::ops::Range<usize>> = Vec::with_capacity(dirty.len());
    for (_, ranges) in dirty {
        let mut runs = Vec::new();
        let short_start = all_shorts.len();
        let mut words = 0usize;
        for &(a, b) in ranges {
            let len = b - a;
            if short(a, len) {
                words += len / 4;
                all_shorts.push((a, b));
            } else {
                runs.push((a, at, len));
                total_bytes += len as u64;
                at += (len as u64).next_multiple_of(16);
            }
        }
        total_words += words as u64;
        copies.push(Copy { runs, pairs: (at, words as u32) });
        copy_short_spans.push(short_start..all_shorts.len());
        at += (words as u64 * 8).next_multiple_of(16);
    }
    if at == 0 {
        return None;
    }
    let memory = memory(host, at)?;
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut pair_groups: Vec<(usize, std::ops::Range<usize>, usize)> = Vec::new();
    for (((shadow, _), copy), short_span) in dirty.iter().zip(&copies).zip(copy_short_spans) {
        for &(a, _, n) in &copy.runs {
            if !shadow.ensure_staging_committed(a, a + n) {
                return None;
            }
        }
        for &(a, b) in &all_shorts[short_span.clone()] {
            if !shadow.ensure_staging_committed(a, b) {
                return None;
            }
        }
        let from = shadow.card_data() as usize;
        moves.extend(copy.runs.iter().map(|&(a, to, n)| (from + a, memory.data + to as usize, n)));
        if !short_span.is_empty() {
            pair_groups.push((from, short_span, memory.data + copy.pairs.0 as usize));
        }
    }
    let parallel = should_parallelize_take(total_bytes, total_words);
    fill_staging(&moves, &all_shorts, &pair_groups, parallel);
    STATS[0].fetch_add(total_bytes, Ordering::Relaxed);
    STATS[1].fetch_add(moves.len() as u64, Ordering::Relaxed);
    STATS[2].fetch_add(total_words, Ordering::Relaxed);
    STATS[3].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Own { memory, copies })
}

impl Own {
    /// Records the sends into `list`; the copies that `looked` says are in
    /// `COPY_DEST`, the others in the common state. Copies for which
    /// `needs_read` is true are left in `NON_PIXEL_SHADER_RESOURCE` for
    /// immediate `CARD_COMPARE` re-hashing; all other touched copies are
    /// transitioned back to `COMMON` so `sync` can run inside the submission's
    /// single `ExecuteCommandLists`.
    ///
    /// # Safety
    /// An open list of ours, whose compute state is set anew behind this.
    pub(super) unsafe fn record(
        &self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
        looked: impl Fn(&Shadow) -> bool,
        needs_read: impl Fn(&Shadow) -> bool,
    ) {
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        let write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let common = D3D12_RESOURCE_STATE_COMMON;
        let (mut first, mut last) = (Vec::new(), Vec::new());
        for ((shadow, _), copy) in dirty.iter().zip(&self.copies) {
            for &(a, from, n) in &copy.runs {
                // SAFETY: the caller's; live buffers, ranges inside them.
                unsafe { list.CopyBufferRegion(&shadow.buffer, a as u64, &self.memory.buffer, from, n as u64) };
            }
            let mut state = (looked(shadow) || !copy.runs.is_empty()).then_some(dest);
            if copy.pairs.1 > 0 {
                if let Some(was) = state {
                    first.push(transition(&shadow.buffer, was, write));
                }
                state = Some(write);
            }
            if let Some(was) = state {
                let target = if needs_read(shadow) { read } else { common };
                last.push(transition(&shadow.buffer, was, target));
            }
        }
        // SAFETY: the caller's; live objects.
        unsafe {
            if !first.is_empty() {
                list.ResourceBarrier(&first);
            }
            if let Some(pass) = pass(host)
                && self.copies.iter().any(|c| c.pairs.1 > 0)
            {
                list.SetComputeRootSignature(&pass.root);
                list.SetPipelineState(&pass.pso);
                list.SetComputeRootShaderResourceView(1, self.memory.buffer.GetGPUVirtualAddress());
                for ((shadow, _), copy) in dirty.iter().zip(&self.copies).filter(|(_, c)| c.pairs.1 > 0) {
                    list.SetComputeRootUnorderedAccessView(2, shadow.buffer.GetGPUVirtualAddress());
                    const MOST: u32 = 64 * D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
                    let (at, words) = copy.pairs;
                    let mut done = 0;
                    while done < words {
                        let n = (words - done).min(MOST);
                        let constants = [at as u32 + done * 8, n];
                        list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.Dispatch(n.div_ceil(64), 1, 1);
                        done += n;
                    }
                }
            }
            if !last.is_empty() {
                list.ResourceBarrier(&last);
            }
        }
    }

    /// The memory, to be kept until the submission has run.
    pub(super) fn memory(self) -> Memory {
        self.memory
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_SPARE_MB, PARALLEL_MIN_BYTES, PARALLEL_MIN_PAIRS, fill_staging, resolve_kept_bytes,
        select_spare_evictions, short, should_parallelize_take,
    };

    #[test]
    fn spare_cap_defaults_to_192_mib_and_evicts_duplicates_then_lru() {
        const MIB: u64 = 1 << 20;
        assert_eq!(resolve_kept_bytes(None), DEFAULT_SPARE_MB * MIB);
        assert_eq!(DEFAULT_SPARE_MB, 192);
        assert_eq!(resolve_kept_bytes(Some("128")), 128 * MIB);
        assert_eq!(resolve_kept_bytes(Some("0")), 0);
        assert_eq!(resolve_kept_bytes(Some("bad")), 192 * MIB);
        assert_eq!(super::bucket_size(100 << 10), 512 << 10);
        assert_eq!(super::bucket_size(3 << 20), 4 << 20);
        assert_eq!(super::bucket_size(33 << 20), 36 << 20);

        // Two 16 MiB duplicates (ticks 1, 5), one recent 32 MiB bucket (tick 9),
        // and one 8 MiB bucket (tick 2): total 72 MiB > 64 MiB cap.
        // Largest-first would wrongly evict the fresh 32 MiB buffer; size-bucketed
        // LRU evicts the oldest duplicate 16 MiB (index 0) and retains all 3 active buckets!
        let items = [
            (16 * MIB, 1),
            (8 * MIB, 2),
            (16 * MIB, 5),
            (32 * MIB, 9),
        ];
        assert_eq!(select_spare_evictions(&items, 64 * MIB), vec![0]);

        // When all remaining buckets are unique and still exceed cap (32 + 16 + 8 = 56 > 40),
        // evict globally oldest (LRU: 8 MiB at tick 2, then 16 MiB at tick 5), keeping recent 32 MiB!
        let unique = [
            (8 * MIB, 2),
            (16 * MIB, 5),
            (32 * MIB, 9),
        ];
        assert_eq!(select_spare_evictions(&unique, 40 * MIB), vec![0, 1]);
    }

    #[test]
    fn parallel_send_take_matches_serial_bit_for_bit_and_honors_thresholds() {
        assert!(!should_parallelize_take(PARALLEL_MIN_BYTES - 1, 0));
        assert!(should_parallelize_take(PARALLEL_MIN_BYTES, 0));
        assert!(!should_parallelize_take(0, PARALLEL_MIN_PAIRS - 1));
        assert!(should_parallelize_take(0, PARALLEL_MIN_PAIRS));
        assert!(should_parallelize_take(200 << 10, 8_192));

        // Build 3 deterministic source buffers with mixed long copies (>256 B & unaligned)
        // and >8,000 short scatter word pairs across multiple runs.
        let make_src = |len: usize, seed: u8| -> Vec<u8> {
            (0..len)
                .map(|i| ((i.wrapping_mul(37).wrapping_add(seed as usize * 13)) ^ (i >> 5)) as u8)
                .collect()
        };
        let src0 = make_src(512 << 10, 1);
        let src1 = make_src(256 << 10, 2);
        let src2 = make_src(128 << 10, 3);
        let srcs = [&src0, &src1, &src2];

        let mut per_buf_ranges: [Vec<(usize, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        // Buf 0: one large 300 KiB long copy + one odd unaligned 273 B copy + 200 short runs of 128 B (6,400 words).
        per_buf_ranges[0].push((0, 300 << 10));
        per_buf_ranges[0].push((301 << 10 | 3, (301 << 10 | 3) + 273));
        for k in 0..200usize {
            let off = (320 << 10) + k * 256;
            per_buf_ranges[0].push((off, off + 128));
        }
        // Buf 1: 400 tiny 4-byte/16-byte/64-byte short runs + 2 medium long runs (1024 B, 70 KiB).
        for k in 0..400usize {
            let off = k * 128;
            let len = match k % 3 {
                0 => 4,
                1 => 16,
                _ => 64,
            };
            per_buf_ranges[1].push((off, off + len));
        }
        per_buf_ranges[1].push((100 << 10, (100 << 10) + 1024));
        per_buf_ranges[1].push((120 << 10, (120 << 10) + (70 << 10)));
        // Buf 2: 50 short runs of 256 B (3,200 words) + 1 unaligned 1001 B long copy.
        for k in 0..50usize {
            let off = k * 512;
            per_buf_ranges[2].push((off, off + 256));
        }
        per_buf_ranges[2].push((64 << 10 | 1, (64 << 10 | 1) + 1001));

        // Lay out destination offsets identically to `take()`.
        let mut at = 0usize;
        let mut all_shorts: Vec<(usize, usize)> = Vec::new();
        let mut run_plans: Vec<Vec<(usize, usize, usize)>> = Vec::new();
        let mut short_plans: Vec<(std::ops::Range<usize>, usize, usize)> = Vec::new();
        let mut total_bytes = 0u64;
        let mut total_words = 0u64;
        for ranges in &per_buf_ranges {
            let mut runs = Vec::new();
            let short_start = all_shorts.len();
            let mut words = 0usize;
            for &(a, b) in ranges {
                let len = b - a;
                if short(a, len) {
                    words += len / 4;
                    all_shorts.push((a, b));
                } else {
                    runs.push((a, at, len));
                    total_bytes += len as u64;
                    at += len.next_multiple_of(16);
                }
            }
            total_words += words as u64;
            let pairs_at = at;
            at += (words * 8).next_multiple_of(16);
            run_plans.push(runs);
            short_plans.push((short_start..all_shorts.len(), pairs_at, words));
        }
        assert!(should_parallelize_take(total_bytes, total_words));

        let mut serial_buf = vec![0xCCu8; at];
        let mut parallel_buf = vec![0xCCu8; at];

        let run_into = |dst_buf: &mut [u8], parallel: bool| {
            let dst_base = dst_buf.as_mut_ptr() as usize;
            let mut moves = Vec::new();
            let mut pair_groups = Vec::new();
            for ((src, runs), (span, pairs_at, _)) in srcs.iter().zip(&run_plans).zip(&short_plans) {
                let from = src.as_ptr() as usize;
                for &(a, to, n) in runs {
                    moves.push((from + a, dst_base + to, n));
                }
                if !span.is_empty() {
                    pair_groups.push((from, span.clone(), dst_base + *pairs_at));
                }
            }
            fill_staging(&moves, &all_shorts, &pair_groups, parallel);
        };

        run_into(&mut serial_buf, false);
        run_into(&mut parallel_buf, true);

        assert_eq!(parallel_buf, serial_buf, "parallel send::take must be bit-identical to serial");

        // Spot-check that long copies and word pairs actually contain expected bytes/pairs.
        for ((src, runs), (span, pairs_at, words)) in srcs.iter().zip(&run_plans).zip(&short_plans) {
            for &(a, to, n) in runs {
                assert_eq!(&parallel_buf[to..to + n], &src[a..a + n]);
            }
            let mut cursor = *pairs_at;
            let mut seen_words = 0usize;
            for &(a, b) in &all_shorts[span.clone()] {
                for off in (a..b).step_by(4) {
                    let got_off = u32::from_le_bytes(parallel_buf[cursor..cursor + 4].try_into().unwrap());
                    let got_val = u32::from_le_bytes(parallel_buf[cursor + 4..cursor + 8].try_into().unwrap());
                    let want_val = u32::from_le_bytes(src[off..off + 4].try_into().unwrap());
                    assert_eq!(got_off, off as u32);
                    assert_eq!(got_val, want_val);
                    cursor += 8;
                    seen_words += 1;
                }
            }
            assert_eq!(seen_words, *words);
        }
    }
}
