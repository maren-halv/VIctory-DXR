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

/// Default [`SPARE`] retention cap in MiB (`64 MiB` for both BAR and L0,
/// replacing the `192–256 MiB` largest-first cache; `ARCH-05`).
const DEFAULT_SPARE_MB: u64 = 64;

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
    // By doubles from 512 KiB (average flight sends ~1-2 MiB), so small flights
    // don't pin 8 MiB BAR buffers and overflow `kept()`.
    let size = size.max(512 << 10).next_power_of_two();
    let mapped = if on_card() { host.mapped_card_buffer(size, false) } else { None };
    let (buffer, data) = match mapped {
        Some(made) => made,
        None => host.upload_buffer(size)?,
    };
    name(&buffer, || "bytes a submission sends up".into());
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

/// Since the last frame-rate line: bytes copied, runs copied, words stored,
/// microseconds of taking them out of the CPU sides.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

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

/// Runs whose pairs are one piece of work.
const RUNS_AT_A_TIME: usize = 4096;

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
    let mut all_chunk_words: Vec<usize> = Vec::new();
    let mut copy_short_spans: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> =
        Vec::with_capacity(dirty.len());
    for (_, ranges) in dirty {
        let mut runs = Vec::new();
        let short_start = all_shorts.len();
        let chunk_start = all_chunk_words.len();
        let mut cur_chunk_words = 0usize;
        let mut words = 0usize;
        for &(a, b) in ranges {
            let len = b - a;
            if short(a, len) {
                let w = len / 4;
                words += w;
                cur_chunk_words += w;
                all_shorts.push((a, b));
                if (all_shorts.len() - short_start) % RUNS_AT_A_TIME == 0 {
                    all_chunk_words.push(cur_chunk_words);
                    cur_chunk_words = 0;
                }
            } else {
                runs.push((a, at, len));
                total_bytes += len as u64;
                at += (len as u64).next_multiple_of(16);
            }
        }
        if cur_chunk_words > 0 {
            all_chunk_words.push(cur_chunk_words);
        }
        total_words += words as u64;
        copies.push(Copy { runs, pairs: (at, words as u32) });
        copy_short_spans.push((short_start..all_shorts.len(), chunk_start..all_chunk_words.len()));
        at += (words as u64 * 8).next_multiple_of(16);
    }
    if at == 0 {
        return None;
    }
    let memory = memory(host, at)?;
    let mut moves: Vec<(usize, usize, usize)> = Vec::new();
    let mut work: Vec<(usize, &[(usize, usize)], usize)> = Vec::new();
    for (((shadow, _), copy), (short_span, chunk_span)) in
        dirty.iter().zip(&copies).zip(copy_short_spans)
    {
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
            // (80,000 runs of some words a frame: in pieces of work.)
            let mut to = memory.data + copy.pairs.0 as usize;
            for (chunk, &words) in all_shorts[short_span].chunks(RUNS_AT_A_TIME).zip(&all_chunk_words[chunk_span]) {
                work.push((from, chunk, to));
                to += words * 8;
            }
        }
    }
    let one = |i: usize| {
        let (from, ranges, to) = work[i];
        let mut to = to as *mut [u32; 2];
        for &(a, b) in ranges {
            for offset in (a..b).step_by(4) {
                // SAFETY: inside the copy's mapping (its CPU side, which
                // the thread that asked, and waits, alone writes in these
                // ranges) and inside the memory, which has room for every
                // pair, a piece of work's behind the last one's.
                unsafe {
                    let word = ((from + offset) as *const u32).read_unaligned();
                    to.write([offset as u32, word]);
                    to = to.add(1);
                }
            }
        }
    };
    crate::land::each_under(work.len(), &one);
    STATS[0].fetch_add(total_bytes, Ordering::Relaxed);
    STATS[1].fetch_add(moves.len() as u64, Ordering::Relaxed);
    STATS[2].fetch_add(total_words, Ordering::Relaxed);
    crate::shadow::copy_all(moves);
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
    use super::{DEFAULT_SPARE_MB, resolve_kept_bytes, select_spare_evictions};

    #[test]
    fn spare_cap_defaults_to_64_mib_and_evicts_duplicates_then_lru() {
        const MIB: u64 = 1 << 20;
        assert_eq!(resolve_kept_bytes(None), DEFAULT_SPARE_MB * MIB);
        assert_eq!(DEFAULT_SPARE_MB, 64);
        assert_eq!(resolve_kept_bytes(Some("128")), 128 * MIB);
        assert_eq!(resolve_kept_bytes(Some("0")), 0);
        assert_eq!(resolve_kept_bytes(Some("bad")), 64 * MIB);

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
}
