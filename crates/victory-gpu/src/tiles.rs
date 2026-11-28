//! Levels on demand: host copies of title textures that take the card's
//! memory level by level, as the title holds the levels.
//!
//! The title places a texture's whole chain of levels in its address space
//! and holds only some of them: the smallest first, at the start of the
//! texture's memory. The memory of the larger ones is either not mapped,
//! or belongs to the next texture, which the title packs where they would
//! be (run 188). A host copy was a committed texture: every level of it
//! took the card's memory from the first upload on. Run 189's accounting
//! line counted 2.2 to 3.0 GB of 4.3 to 5.5 GB of host copies in levels
//! whose title memory was not mapped, on some 1,600 copies; with about
//! 2.2 to 3.1 GB of render targets that filled the card, and textures
//! that needed a new host copy waited for room (the main character's face
//! among them, runs 188 and 189).
//!
//! Here such a texture is a reserved one (`CreateReservedResource`): it has
//! every level, so level numbers and sizes are what they were, but no
//! memory. The rules:
//! - A level is given tiles of a heap of ours (`UpdateTileMappings`, on the
//!   queue) only in a submission-time step of a list that also writes all
//!   of that level: the upload of the texture (`texture.rs` `upload`, which
//!   copies every level from the staging buffer every time), or, before
//!   the card itself writes the texture (a copy, a resolve), an upload of
//!   every level with every level given (`texture.rs` `make_whole`). The
//!   step runs when the list is submitted, before the list is executed, so
//!   the mapping is ahead of the copy on the queue. Tiles hold what their
//!   last owner left; no level is shown before it is written whole.
//! - Which levels are held is said by one function, at the moment the
//!   staging buffer is filled (`Staging::held_levels`): the levels whose
//!   title memory is mapped and lies before the start of the next texture
//!   placed in this one's span. The staging buffer holds zeros for the
//!   others, and the step gives tiles by that same answer.
//! - The small levels the card packs into shared tiles are given together
//!   when any of them is held; the upload writes all of them (zeros for
//!   the ones not held).
//! - A shader must not reach a level without memory (it reads zeros: a
//!   black texture). The host view of a slot is made from the first level
//!   that is held and has tiles ([`show_from`], `descriptor/host.rs`
//!   `make`), and the slots of a texture are made again when that level
//!   changes (`HostHeap::texture_views_remade`).
//! - Deferred tile reclamation (`VICTORY_TILES_RECLAIM`, enabled by default;
//!   opt out with `VICTORY_TILES_RECLAIM=0`): when a full upload runs and a
//!   mapped level is no longer held by the title and was never written by the
//!   card (`held | card`), those subresources are unmapped with
//!   `D3D12_TILE_RANGE_FLAG_NULL` in one batched `UpdateTileMappings` call,
//!   their claims are cleared, the first shown level is updated, and the
//!   unmapped tile runs are wrapped in a [`RetiredTiles`] guard handed to
//!   `crate::host::retire` so they return to the pool only after in-flight GPU
//!   work finishes.
//!
//! Only for a texture the title places in memory and fills, two-dimensional,
//! one layer, more than one level, of [`floor`] bytes or more, whose memory
//! is not all its own when the host copy is made (`resource.rs` `hosted`);
//! every other texture is made as before. Off unless
//! `VICTORY_LEVELS_ON_DEMAND=1`.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::host;

/// The card's tile: 64 KiB.
const TILE: u64 = 64 << 10;

/// Tiles in one heap of the pool (64 MiB). The largest level given is a
/// 4096x4096 one of one byte a texel: 256 tiles.
const HEAP_TILES: u32 = 1024;

/// Stack buffer size for `UpdateTileMappings` ranges/regions on the common path.
const STACK_RANGES: usize = 16;

/// Since the start, or now: such textures alive; the bytes all their
/// levels would take; the bytes of the tiles given to them; the bytes of
/// the pool's heaps; such textures made in all; levels given; calls of
/// `UpdateTileMappings`; their microseconds; heaps given back to the card;
/// uploads that left a held level without tiles (the card had no room, or
/// no heap could be had); uploads recorded again for such a level.
pub(crate) static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

/// Now: such textures alive that show no level (no level is both held and
/// given tiles, so every view of them reads zeros: a black texture), and
/// the bytes all their levels would take; since the start: levels given
/// tiles on a full card by [`small_when_full_mb`].
pub(crate) static BLANK: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Reclamation metrics: `[0]` levels (subresources) reclaimed, `[1]` bytes
/// unmapped total, `[2]` bytes currently awaiting GPU fence before returning
/// to the pool, `[3]` bytes returned to the pool.
pub(crate) static RECLAIMED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Tier-2 pressure-driven top-mip demotion / re-promotion metrics (`ARCH-01`):
/// `[0]` textures demoted (Phase 2 committed), `[1]` top-mip levels demoted,
/// `[2]` VRAM bytes reclaimed by demotion, `[3]` re-promotions triggered.
pub(crate) static DEMOTED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Default local VRAM threshold (%) above which idle large tiled textures begin
/// fence-gated top-mip demotion (`VICTORY_TILES_DEMOTE_VRAM_PERCENT`).
const DEFAULT_DEMOTE_VRAM_PERCENT: u64 = 72;
/// Severe VRAM pressure threshold (%) permitting deeper (up to 2 top-mip) demotion
/// after a shorter idle grace period (still never clamping active close-up textures!).
const SEVERE_DEMOTE_VRAM_PERCENT: u64 = 85;
/// Comfort band (%) below which demoted textures proactively re-promote.
const REPROMOTE_VRAM_PERCENT: u64 = 68;
/// Idle grace period for moderate pressure (15 seconds + 60 frames) so active
/// close-up Series X textures are NEVER clamped (`ARCH-01`, Spec-Trap #7).
const DEMOTE_IDLE_MODERATE_MS: u64 = 15_000;
const DEMOTE_IDLE_MODERATE_FRAMES: u64 = 60;
/// Idle grace period for severe pressure (5 seconds + 30 frames).
const DEMOTE_IDLE_SEVERE_MS: u64 = 5_000;
const DEMOTE_IDLE_SEVERE_FRAMES: u64 = 30;
/// Submissions / frames required after descriptor rewrite (`levels_changed`)
/// before Phase 2 issues `UpdateTileMappings(..., NULL)`.
const DEMOTE_FENCE_LAG: u64 = 2;
const DEMOTE_FRAME_LAG: u64 = 2;
/// Maximum demotion/re-promotion actions per 1-Hz sweep tick.
const DEMOTE_PER_SWEEP: usize = 16;

fn parse_reclaim(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Whether unheld, non-card-written levels are unmapped and their tiles
/// reclaimed once in-flight GPU work finishes (`VICTORY_TILES_RECLAIM`,
/// enabled by default; set to `0` to opt out).
pub(crate) fn reclaim_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_reclaim(std::env::var("VICTORY_TILES_RECLAIM").ok().as_deref()))
}

fn parse_demote_wanted(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

/// Whether idle large Tier-2 tiled textures demote top mips under VRAM/host
/// memory pressure (`VICTORY_TILES_DEMOTE`, enabled by default when
/// `VICTORY_LEVELS_ON_DEMAND=1`; set `0` to opt out).
pub(crate) fn demote_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_demote_wanted(std::env::var("VICTORY_TILES_DEMOTE").ok().as_deref()))
}

fn parse_demote_vram_percent(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(DEFAULT_DEMOTE_VRAM_PERCENT, |p| p.clamp(50, 95))
}

fn demote_vram_percent() -> u64 {
    static PCT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PCT.get_or_init(|| {
        parse_demote_vram_percent(std::env::var("VICTORY_TILES_DEMOTE_VRAM_PERCENT").ok().as_deref())
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DemotionPressure {
    /// Below pressure threshold (`can_repromote` true when below comfort band).
    None { can_repromote: bool },
    /// `VRAM > 72%` or `residency::pressured()`: demote 1 top mip on idle (`>= 15s`) textures.
    Moderate { vram_pct: u64 },
    /// `VRAM >= 85%`: demote up to 2 top mips on idle (`>= 5s`) textures.
    Severe { vram_pct: u64 },
}

fn evaluate_demotion_pressure(
    memory: Option<[u64; 4]>,
    empty_pool_bytes: u64,
    host_pressured: bool,
    threshold_pct: u64,
) -> DemotionPressure {
    let vram_pct = match memory {
        Some([budget, used, ..]) if budget > 0 => {
            let used_net = used.saturating_sub(empty_pool_bytes);
            used_net.saturating_mul(100) / budget
        }
        _ => 0,
    };
    let severe_pct = threshold_pct.max(SEVERE_DEMOTE_VRAM_PERCENT);
    if vram_pct >= severe_pct {
        DemotionPressure::Severe { vram_pct }
    } else if vram_pct > threshold_pct || host_pressured {
        DemotionPressure::Moderate { vram_pct }
    } else {
        let repromote_pct = threshold_pct.saturating_sub(4).min(REPROMOTE_VRAM_PERCENT);
        DemotionPressure::None {
            can_repromote: !host_pressured && vram_pct < repromote_pct,
        }
    }
}

/// Pure policy deciding whether an idle tiled texture should demote its top mips
/// and which `target_shown` mip level it should clamp to.
///
/// Strictly enforces:
/// - Never demotes active/recent textures (`!is_idle`) — preserving full Series X fidelity.
/// - Never demotes small textures (`full_bytes < floor_bytes`) or single-standard-mip textures.
/// - Never demotes any level written by the host GPU (`card` bits).
/// - Always leaves at least one backed mip level `>= target_shown` so the texture never goes black.
#[allow(clippy::too_many_arguments)]
fn choose_demote_target(
    pressure: DemotionPressure,
    full_bytes: u64,
    floor_bytes: u64,
    standard_levels: u32,
    levels: u32,
    current_shown: u32,
    backed: u32,
    card: u32,
    is_idle: bool,
) -> Option<u32> {
    let max_demote = match pressure {
        DemotionPressure::None { .. } => return None,
        DemotionPressure::Moderate { .. } => 1u32,
        DemotionPressure::Severe { .. } => 2u32,
    };
    if !is_idle || full_bytes < floor_bytes || standard_levels < 2 || current_shown >= levels {
        return None;
    }
    // Cap target at `standard_levels` so packed mip-tail stays intact and visible.
    let target = current_shown
        .saturating_add(max_demote)
        .min(standard_levels)
        .min(levels.saturating_sub(1));
    if target <= current_shown || target >= 32 {
        return None;
    }
    let below_mask = (1u32 << target) - 1;
    // Never demote any card-written level.
    if card & below_mask != 0 {
        return None;
    }
    // Must actually reclaim at least one currently-backed top mip.
    if backed & below_mask == 0 {
        return None;
    }
    // Must retain at least one backed level at or below `target` (never blank!).
    let every = if levels >= 32 { u32::MAX } else { (1u32 << levels) - 1 };
    let remaining = backed & every & !below_mask;
    if remaining == 0 {
        return None;
    }
    Some(target)
}

/// `VICTORY_TILES_SMALL_WHEN_FULL_MB` (0, off, by default): while the card
/// has no room, an upload still gives tiles to the smallest held levels of
/// its texture (the packed ones, then standard ones from the smallest up,
/// at most [`SMALL_TILES`] tiles a texture), as long as all the tiles
/// given stand under this many megabytes. A texture then shows blurred
/// from its small levels instead of black; its larger levels wait for room
/// as before.
fn small_when_full_mb() -> u64 {
    static MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_TILES_SMALL_WHEN_FULL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(0, |mb| mb.min(1024))
    })
}

/// Tiles a texture gets at most on a full card ([`small_when_full_mb`]): a
/// megabyte.
const SMALL_TILES: u32 = 16;

/// Whether textures are made this way: the switch
/// (`VICTORY_LEVELS_ON_DEMAND=1`; off by default), and a card that reads
/// zeros from, and drops writes to, levels without memory (tiled resources
/// tier 2).
pub(crate) fn wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if !std::env::var_os("VICTORY_LEVELS_ON_DEMAND").is_some_and(|v| v == "1") {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        // SAFETY: a plain query into a struct of its own size.
        let asked = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        let tier = options.TiledResourcesTier.0;
        let on = asked.is_ok() && tier >= 2;
        tracing::info!(
            tier,
            on,
            floor_kb = floor() >> 10,
            "host copies of title textures take the card's memory level by level (VICTORY_LEVELS_ON_DEMAND=1)"
        );
        on
    })
}

fn parse_floor(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map_or(512 << 10, |kb| kb.max(64).saturating_mul(1024))
}

/// The bytes from which a texture is made this way: 512 KiB (`ARCH-01`
/// 256-512 KiB Tier-2 threshold), or `VICTORY_LEVELS_MIN_KB` (min 64 KiB).
pub(crate) fn floor() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| parse_floor(std::env::var("VICTORY_LEVELS_MIN_KB").ok().as_deref()))
}

/// The bytes of the tiles given to textures now: the card's memory they
/// hold (`resource.rs` `room` counts it with the other host copies).
pub(crate) fn given_total() -> u64 {
    STATS[2].load(Ordering::Relaxed)
}

/// One heap of the pool and its free runs of tiles (first tile, tiles), in
/// order, no two touching.
struct Heap {
    heap: ID3D12Heap,
    free: VecDeque<(u32, u32)>,
    free_count: u32,
}

// SAFETY: a heap of the device, named only in queue and device calls, which
// any thread may make; its free runs are plain numbers behind `POOL`'s lock.
unsafe impl Send for Heap {}

impl Heap {
    fn take_runs(&mut self, count: u32) -> Vec<(u32, u32)> {
        let mut runs = Vec::new();
        let mut left = count;
        while left > 0 {
            let Some(front) = self.free.front_mut() else {
                break;
            };
            let (start, tiles) = *front;
            let part = tiles.min(left);
            runs.push((start, part));
            if part == tiles {
                self.free.pop_front();
            } else {
                *front = (start + part, tiles - part);
            }
            left -= part;
        }
        debug_assert_eq!(left, 0);
        self.free_count -= count - left;
        debug_assert_eq!(self.free.iter().map(|r| r.1).sum::<u32>(), self.free_count);
        runs
    }

    fn coalesce(&mut self) {
        let slice = self.free.make_contiguous();
        if slice.len() <= 1 {
            debug_assert_eq!(self.free.iter().map(|r| r.1).sum::<u32>(), self.free_count);
            return;
        }
        slice.sort_unstable_by_key(|r| r.0);
        let mut w = 0usize;
        for r in 1..slice.len() {
            let (from, tiles) = slice[r];
            debug_assert!(slice[w].0 + slice[w].1 <= from, "overlapping tile runs returned to pool");
            if slice[w].0 + slice[w].1 == from {
                slice[w].1 += tiles;
            } else {
                w += 1;
                slice[w] = (from, tiles);
            }
        }
        self.free.truncate(w + 1);
        debug_assert_eq!(self.free.iter().map(|r| r.1).sum::<u32>(), self.free_count);
    }
}

/// The pool's heaps by place; a place is empty after its heap went back to
/// the card (a texture names its heaps by place).
static POOL: Mutex<Vec<Option<Heap>>> = Mutex::new(Vec::new());

/// `count` tiles of one heap, in as many runs as it takes, the lowest free
/// tiles first (single free tiles are used up before a long run is cut):
/// the heap, its place in the pool and the runs (first tile, tiles). A new
/// heap is made outside `POOL`'s lock when no heap has that many free.
fn take(device: &ID3D12Device, count: u32) -> Option<(ID3D12Heap, usize, Vec<(u32, u32)>)> {
    if count == 0 || count > HEAP_TILES {
        return None;
    }
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = pool.iter().position(|h| h.as_ref().is_some_and(|h| h.free_count >= count)) {
            let h = pool[index].as_mut()?;
            let runs = h.take_runs(count);
            return Some((h.heap.clone(), index, runs));
        }
    }
    // Create the 64 MiB heap without holding POOL.lock().
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: u64::from(HEAP_TILES) * TILE,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    };
    let mut made: Option<ID3D12Heap> = None;
    // SAFETY: plain creation; the out pointer is ours.
    if let Err(e) = unsafe { device.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut made) } {
        tracing::warn!(error = %e, "no heap for texture levels given on demand: those levels stay empty");
        return None;
    }
    let made = made?;

    let mut unused_heap: Option<ID3D12Heap> = None;
    let out = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let index = if let Some(existing) =
            pool.iter().position(|h| h.as_ref().is_some_and(|h| h.free_count >= count))
        {
            unused_heap = Some(made);
            existing
        } else {
            let mut free = VecDeque::with_capacity(8);
            free.push_back((0, HEAP_TILES));
            let fresh = Heap {
                heap: made,
                free,
                free_count: HEAP_TILES,
            };
            STATS[3].fetch_add(u64::from(HEAP_TILES) * TILE, Ordering::Relaxed);
            match pool.iter().position(Option::is_none) {
                Some(index) => {
                    pool[index] = Some(fresh);
                    index
                }
                None => {
                    pool.push(Some(fresh));
                    pool.len() - 1
                }
            }
        };
        let h = pool[index].as_mut()?;
        let runs = h.take_runs(count);
        (h.heap.clone(), index, runs)
    };
    drop(unused_heap);
    Some(out)
}

/// Gives the runs `runs` of the pool's heap at `index` back via [`return_runs`].
#[cfg(test)]
fn give_back(index: usize, runs: &[(u32, u32)]) {
    let tagged: Vec<(usize, u32, u32)> = runs.iter().map(|&(start, count)| (index, start, count)).collect();
    return_runs(&tagged);
}

/// Returns a batch of `(heap_index, start, count)` runs across one or more
/// heaps under a single `POOL` lock acquisition, dropping any emptied
/// `ID3D12Heap`s after releasing the lock.
fn return_runs(runs: &[(usize, u32, u32)]) {
    if runs.is_empty() {
        return;
    }
    let mut dropped_heaps: Vec<ID3D12Heap> = Vec::new();
    {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let mut touched = [0usize; STACK_RANGES];
        let mut touched_len = 0usize;
        let mut touched_overflow: Vec<usize> = Vec::new();

        for &(index, start, count) in runs {
            if count == 0 {
                continue;
            }
            let Some(Some(h)) = pool.get_mut(index) else {
                continue;
            };
            h.free.push_back((start, count));
            h.free_count += count;
            if !touched[..touched_len].contains(&index) && !touched_overflow.contains(&index) {
                if touched_len < touched.len() {
                    touched[touched_len] = index;
                    touched_len += 1;
                } else {
                    touched_overflow.push(index);
                }
            }
        }

        let mut live_heaps = pool.iter().flatten().count();
        for &index in touched[..touched_len].iter().chain(touched_overflow.iter()) {
            let Some(Some(h)) = pool.get_mut(index) else {
                continue;
            };
            h.coalesce();
            if h.free_count == HEAP_TILES && live_heaps > 1 {
                if let Some(removed) = pool[index].take() {
                    live_heaps -= 1;
                    STATS[3].fetch_sub(u64::from(HEAP_TILES) * TILE, Ordering::Relaxed);
                    STATS[8].fetch_add(1, Ordering::Relaxed);
                    dropped_heaps.push(removed.heap);
                }
            }
        }
    }
    drop(dropped_heaps);
}

/// Tile runs unmapped from a reserved texture, held alive until in-flight GPU
/// work completes (`crate::host::retire`) before returning them to [`POOL`].
pub(crate) struct RetiredTiles {
    runs: Vec<(usize, u32, u32)>,
    bytes: u64,
}

impl RetiredTiles {
    fn new(runs: Vec<(usize, u32, u32)>, bytes: u64) -> Self {
        RECLAIMED[2].fetch_add(bytes, Ordering::Relaxed);
        Self { runs, bytes }
    }

    #[cfg(test)]
    fn bytes(&self) -> u64 {
        self.bytes
    }

    #[cfg(test)]
    fn runs(&self) -> &[(usize, u32, u32)] {
        &self.runs
    }
}

impl Drop for RetiredTiles {
    fn drop(&mut self) {
        return_runs(&self.runs);
        STATS[2].fetch_sub(self.bytes, Ordering::Relaxed);
        RECLAIMED[2].fetch_sub(self.bytes, Ordering::Relaxed);
        RECLAIMED[3].fetch_add(self.bytes, Ordering::Relaxed);
    }
}

/// What is given so far to one texture, tracked per subresource so unheld
/// levels can be unmapped independently.
struct Given {
    /// Per level with tiles of its own: its allocated `(heap_index, start, count)` runs.
    levels: Vec<Option<Vec<(usize, u32, u32)>>>,
    /// For the small levels the card packs together: their allocated runs.
    packed: Option<Vec<(usize, u32, u32)>>,
}

impl Given {
    #[cfg(test)]
    fn all_runs(&self) -> Vec<(usize, u32, u32)> {
        let mut out = Vec::new();
        for runs in self.levels.iter().flatten() {
            out.extend_from_slice(runs);
        }
        if let Some(runs) = &self.packed {
            out.extend_from_slice(runs);
        }
        out
    }

    fn total_tiles(&self) -> u64 {
        let standard: u64 = self
            .levels
            .iter()
            .flatten()
            .flat_map(|runs| runs.iter())
            .map(|run| u64::from(run.2))
            .sum();
        let packed: u64 = self
            .packed
            .iter()
            .flat_map(|runs| runs.iter())
            .map(|run| u64::from(run.2))
            .sum();
        standard + packed
    }
}

/// Pending Phase-2 tile unmapping awaiting GPU fence & frame completion so no
/// in-flight draw can sample tiles that are about to be unmapped with `NULL`.
#[derive(Clone, Copy, Debug)]
struct PendingDemotion {
    target_shown: u32,
    fence_gate: u64,
    frame_gate: u64,
}

/// A host texture made without memory, and the tiles its levels have been
/// given.
pub(crate) struct Tiled {
    /// Tiles of each level that has tiles of its own (level 0 first).
    standard: Vec<u32>,
    /// The tiles of the levels the card packs together (0: there are none);
    /// they are the levels from `standard.len()` on.
    packed: u32,
    /// Levels in all.
    levels: u32,
    /// The bytes all its tiles would take.
    pub(crate) full: u64,
    given: Mutex<Given>,
    /// Bitmask of levels currently backed by tiles (kept in sync under `given`).
    backed: AtomicU32,
    /// The first level a view of the texture may show: the first one that
    /// is held or was written by the card and has tiles (`u32::MAX`: none
    /// yet).
    shown: AtomicU32,
    /// The levels held at the last upload (a bit a level).
    held: AtomicU32,
    /// The levels the card itself wrote, in lists submitted so far.
    card: AtomicU32,
    /// The levels a recorded list of the card's own writes has already
    /// written with zeros first ([`Tiled::claim`]).
    claimed: AtomicU32,
    /// Last moment (`crate::host::moment()`) a draw/upload touched this texture.
    last_used: AtomicU64,
    /// Minimum mip level allowed while demoted under VRAM pressure (`0` = full res).
    demote_floor: AtomicU32,
    /// Phase-1 -> Phase-2 fence-gated top-mip unmapping state.
    pending_demotion: Mutex<Option<PendingDemotion>>,
}

impl Tiled {
    /// For `texture`, just made reserved with `levels` levels and one
    /// layer: how the card cuts it into tiles. `None` if the card will not
    /// say, or cuts it in a way this does not handle; the caller then makes
    /// the texture the old way.
    pub(crate) fn make(device: &ID3D12Device, texture: &ID3D12Resource, levels: u32) -> Option<Tiled> {
        if levels == 0 || levels > 16 {
            return None;
        }
        let mut total = 0u32;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut shape = D3D12_TILE_SHAPE::default();
        let mut count = levels;
        let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); levels as usize];
        // SAFETY: outputs of our own, `count` tilings of them.
        unsafe {
            device.GetResourceTiling(
                texture,
                Some(&raw mut total),
                Some(&raw mut packed),
                Some(&raw mut shape),
                Some(&raw mut count),
                0,
                tilings.as_mut_ptr(),
            );
        }
        let standard_levels = u32::from(packed.NumStandardMips);
        if total == 0 || standard_levels + u32::from(packed.NumPackedMips) != levels || standard_levels > count {
            return None;
        }
        let mut standard = Vec::with_capacity(standard_levels as usize);
        for tiling in &tilings[..standard_levels as usize] {
            let tiles = tiling
                .WidthInTiles
                .checked_mul(u32::from(tiling.HeightInTiles))
                .and_then(|n| n.checked_mul(u32::from(tiling.DepthInTiles)))
                .unwrap_or(0);
            if tiles == 0 || tiles > HEAP_TILES || tiling.StartTileIndexInOverallResource == D3D12_PACKED_TILE {
                return None;
            }
            standard.push(tiles);
        }
        let packed_tiles = if packed.NumPackedMips > 0 { packed.NumTilesForPackedMips } else { 0 };
        if (packed.NumPackedMips > 0 && packed_tiles == 0) || packed_tiles > HEAP_TILES {
            return None;
        }
        let full = u64::from(total) * TILE;
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(full, Ordering::Relaxed);
        STATS[4].fetch_add(1, Ordering::Relaxed);
        // Made with no level shown.
        BLANK[0].fetch_add(1, Ordering::Relaxed);
        BLANK[1].fetch_add(full, Ordering::Relaxed);
        Some(Tiled {
            given: Mutex::new(Given {
                levels: vec![None; standard.len()],
                packed: None,
            }),
            standard,
            packed: packed_tiles,
            levels,
            full,
            backed: AtomicU32::new(0),
            shown: AtomicU32::new(u32::MAX),
            held: AtomicU32::new(0),
            card: AtomicU32::new(0),
            claimed: AtomicU32::new(0),
            last_used: AtomicU64::new(crate::host::moment()),
            demote_floor: AtomicU32::new(0),
            pending_demotion: Mutex::new(None),
        })
    }

    /// At the submission of a list that uploads every level of `texture`
    /// from its staging buffer, of which `held` are the title's now (the
    /// others are zeros there): tiles for the held ones that have none, and
    /// deferred reclamation for unheld, non-card-written levels when
    /// [`reclaim_wanted`] is enabled.
    /// `room`: whether the card has room for more host copies
    /// (`resource.rs` `room`). Without room no held level gets tiles: the
    /// list still brings the levels that have theirs up to date (which
    /// takes none of the card's memory), and its copies into the others go
    /// nowhere.
    /// Returns whether the first level a view may show changed, and
    /// whether a level named is still without tiles (no room, or no heap
    /// could be had): the caller has the upload recorded again later, or
    /// the level would stay empty until the title writes the texture.
    pub(crate) fn uploaded(&self, texture: &ID3D12Resource, held: u32, room: bool) -> (bool, bool) {
        match host() {
            Some(host) => self.uploaded_on(host.queue(), &host.device, texture, held, room),
            None => (false, false),
        }
    }

    /// [`Tiled::uploaded`] on a queue and a device.
    pub(crate) fn uploaded_on(
        &self,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        held: u32,
        room: bool,
    ) -> (bool, bool) {
        let (changed, missing, retired) =
            self.uploaded_on_reclaim(queue, device, texture, held, room, reclaim_wanted());
        if let Some(retired) = retired {
            crate::host::retire(retired);
        }
        (changed, missing)
    }

    /// [`Tiled::uploaded_on`] with an explicit reclamation toggle, returning any
    /// unmapped tile runs in [`RetiredTiles`] so the caller retires them after
    /// GPU work completes.
    pub(crate) fn uploaded_on_reclaim(
        &self,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        held: u32,
        room: bool,
        reclaim: bool,
    ) -> (bool, bool, Option<RetiredTiles>) {
        let held = held & self.every_level();
        let mut given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        self.held.store(held, Ordering::Relaxed);
        let card = self.card.load(Ordering::Relaxed);
        let named = held | card;
        // The levels the card itself writes are few and are not kept
        // waiting: a write of the card's into no memory is lost.
        // On a full card, with `VICTORY_TILES_SMALL_WHEN_FULL_MB`, the
        // smallest held levels too: this list writes every level whole.
        let small = if room { 0 } else { self.small_levels(held) };
        if small != 0 {
            BLANK[2].fetch_add(u64::from(small.count_ones()), Ordering::Relaxed);
        }
        let (changed, retired) = self.give_locked(
            &mut given,
            queue,
            device,
            texture,
            named,
            if room { named } else { card | small },
            reclaim,
        );
        let missing = self.without(named) != 0;
        (changed, missing, retired)
    }

    /// Of the levels `held` (a bit a level), the smallest ones that may get
    /// tiles on a full card ([`small_when_full_mb`]): the packed levels if
    /// any of them is held, then standard levels from the smallest up while
    /// each is held, within [`SMALL_TILES`] tiles; none if the switch is
    /// off or the tiles given already reach its megabytes. Contiguous from
    /// the smallest level, so that a view shown from the first of them
    /// reads no level without memory.
    fn small_levels(&self, held: u32) -> u32 {
        let cap = small_when_full_mb() << 20;
        if cap == 0 || STATS[2].load(Ordering::Relaxed) >= cap {
            return 0;
        }
        let first_packed = self.standard.len() as u32;
        let mut bits = 0u32;
        let mut tiles = 0u32;
        if self.packed > 0 {
            let packed_bits = self.every_level() & !((1u32 << first_packed) - 1);
            if held & packed_bits == 0 {
                return 0;
            }
            bits |= packed_bits;
            tiles += self.packed;
        }
        for level in (0..first_packed).rev() {
            let tiles_here = self.standard[level as usize];
            if held & (1 << level) == 0 || tiles + tiles_here > SMALL_TILES {
                break;
            }
            tiles += tiles_here;
            bits |= 1 << level;
        }
        if tiles > SMALL_TILES { 0 } else { bits }
    }

    /// At the submission of a list in which the card itself writes the
    /// levels `levels` of `texture` (after zeros, if they were new:
    /// [`Tiled::claim`]): tiles for those that have none. Returns whether
    /// the first level a view may show changed.
    /// Only the levels this list writes get tiles here: a held level that
    /// is waiting for room is written by no copy of this list, and tiles
    /// hold what their last owner left.
    pub(crate) fn card_wrote(&self, texture: &ID3D12Resource, levels: u32) -> bool {
        match host() {
            Some(host) => self.card_wrote_on(host.queue(), &host.device, texture, levels),
            None => false,
        }
    }

    /// [`Tiled::card_wrote`] on a queue and a device.
    pub(crate) fn card_wrote_on(
        &self,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        levels: u32,
    ) -> bool {
        let levels = levels & self.every_level();
        let mut given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        let prev = self.card.load(Ordering::Relaxed);
        let (changed, retired) = self.give_locked(
            &mut given,
            queue,
            device,
            texture,
            prev | levels | self.held.load(Ordering::Relaxed),
            levels,
            false,
        );
        debug_assert!(retired.is_none());
        // Only levels that actually received tiles (or already had them) saw
        // the card's write; a level that got no tiles saw its zeros and write
        // dropped, so neither `card` nor `claimed` should keep it.
        let without = self.without(levels);
        let backed = levels & !without;
        if backed != 0 {
            self.card.fetch_or(backed, Ordering::Relaxed);
        }
        if without != 0 {
            self.unclaim(without);
        }
        changed
    }

    fn packed_bits(&self) -> u32 {
        let first_packed = self.standard.len() as u32;
        if self.packed > 0 && first_packed < self.levels {
            self.every_level() & !((1u32 << first_packed) - 1)
        } else {
            0
        }
    }

    fn sync_backed(&self, given: &Given) -> u32 {
        let mut mask = 0u32;
        for (level, slot) in given.levels.iter().enumerate() {
            if level < 32 && slot.is_some() {
                mask |= 1 << level;
            }
        }
        if given.packed.is_some() {
            mask |= self.packed_bits();
        }
        self.backed.store(mask, Ordering::Release);
        mask
    }

    /// The bits of the levels that get their tiles with level `level`:
    /// itself, or every level the card packs together if it is one of them.
    pub(crate) fn with_level(&self, level: u32) -> u32 {
        let first_packed = self.standard.len() as u32;
        if level < first_packed {
            1 << level
        } else if level < self.levels {
            self.every_level() & !((1u32 << first_packed) - 1)
        } else {
            0
        }
    }

    /// Of `levels`, the ones that have no tiles now and that no list
    /// recorded before has claimed: the caller's list writes them with
    /// zeros before the card's own write, so that what the write does not
    /// cover is not what the tiles' last owner left.
    pub(crate) fn claim(&self, levels: u32) -> u32 {
        let fresh = self.without(levels);
        fresh & !self.claimed.fetch_or(fresh, Ordering::Relaxed)
    }

    /// Gives up the claim on `levels` ([`Tiled::claim`]): the zeros of the
    /// list that claimed them never reached tiles of theirs (the list was
    /// reset without being submitted, the zeros could not be recorded, or
    /// no tiles could be had), so the next list that writes them sends its
    /// own.
    /// shortcut: a second list recorded between the claim and this sends
    /// no zeros; if it is the first whose write gets tiles, what that
    /// write does not cover is what the tiles' last owner left.
    pub(crate) fn unclaim(&self, levels: u32) {
        self.claimed.fetch_and(!levels, Ordering::Relaxed);
    }

    /// Of `levels` (a bit a level), the ones that have no tiles now (lock-free).
    pub(crate) fn without(&self, levels: u32) -> u32 {
        let backed = self.backed.load(Ordering::Acquire);
        levels & self.every_level() & !backed
    }

    /// Gives tiles to the levels of `texture` that both `named` and `may`
    /// name (a bit a level, level 0 the lowest) and that have none yet;
    /// the levels the card packs together get theirs when any of them is
    /// so named. `named`: the levels that are the title's or the card's
    /// own, by which the first level a view may show is told; `may`: the
    /// ones of them that may get tiles in this call. To be called only
    /// from a submission-time step of a list that writes every level of
    /// `may`, whole: the call is ahead of that list on `queue`, and tiles
    /// hold what their last owner left.
    /// Returns whether the first level a view may show changed
    /// ([`Tiled::first_shown`]): the caller has the texture's slots made
    /// again.
    #[cfg(test)]
    pub(crate) fn give_on(
        &self,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        named: u32,
        may: u32,
    ) -> bool {
        let mut given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        let (changed, retired) = self.give_locked(&mut given, queue, device, texture, named, may, false);
        debug_assert!(retired.is_none());
        changed
    }

    fn give_locked(
        &self,
        given: &mut Given,
        queue: &ID3D12CommandQueue,
        device: &ID3D12Device,
        texture: &ID3D12Resource,
        named: u32,
        may: u32,
        reclaim: bool,
    ) -> (bool, Option<RetiredTiles>) {
        let has = |level: u32| level < 32 && named & (1 << level) != 0;
        let now = |level: u32| level < 32 && named & may & (1 << level) != 0;
        let first_packed = self.standard.len() as u32;

        let mut retired = None;
        if reclaim {
            let mut unmap: Vec<(u32, u32, Option<usize>)> = Vec::new();
            for level in 0..first_packed {
                if given.levels[level as usize].is_some() && !has(level) {
                    unmap.push((level, self.standard[level as usize], Some(level as usize)));
                }
            }
            if self.packed > 0 && given.packed.is_some() && !(first_packed..self.levels).any(has) {
                unmap.push((first_packed, self.packed, None));
            }
            if !unmap.is_empty() {
                let n = unmap.len();
                let began = std::time::Instant::now();
                if n <= STACK_RANGES {
                    let mut coords = [D3D12_TILED_RESOURCE_COORDINATE::default(); STACK_RANGES];
                    let mut sizes = [D3D12_TILE_REGION_SIZE::default(); STACK_RANGES];
                    let flags = [D3D12_TILE_RANGE_FLAG_NULL; STACK_RANGES];
                    let mut counts = [0u32; STACK_RANGES];
                    for (i, &(subresource, tiles, _)) in unmap.iter().enumerate() {
                        coords[i] = D3D12_TILED_RESOURCE_COORDINATE {
                            X: 0,
                            Y: 0,
                            Z: 0,
                            Subresource: subresource,
                        };
                        sizes[i] = D3D12_TILE_REGION_SIZE {
                            NumTiles: tiles,
                            ..Default::default()
                        };
                        counts[i] = tiles;
                    }
                    // SAFETY: a live reserved texture; `n` regions covering each
                    // unmapped subresource, paired with `n` NULL ranges of matching
                    // tile counts (`pHeap = None`, `pHeapRangeStartOffsets = None`).
                    unsafe {
                        queue.UpdateTileMappings(
                            texture,
                            n as u32,
                            Some(coords.as_ptr()),
                            Some(sizes.as_ptr()),
                            None::<&ID3D12Heap>,
                            n as u32,
                            Some(flags.as_ptr()),
                            None,
                            Some(counts.as_ptr()),
                            D3D12_TILE_MAPPING_FLAG_NONE,
                        );
                    }
                } else {
                    let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = unmap
                        .iter()
                        .map(|&(subresource, _, _)| D3D12_TILED_RESOURCE_COORDINATE {
                            X: 0,
                            Y: 0,
                            Z: 0,
                            Subresource: subresource,
                        })
                        .collect();
                    let sizes: Vec<D3D12_TILE_REGION_SIZE> = unmap
                        .iter()
                        .map(|&(_, tiles, _)| D3D12_TILE_REGION_SIZE {
                            NumTiles: tiles,
                            ..Default::default()
                        })
                        .collect();
                    let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; n];
                    let counts: Vec<u32> = unmap.iter().map(|&(_, tiles, _)| tiles).collect();
                    // SAFETY: same as above, with heap-allocated arrays for >16 regions.
                    unsafe {
                        queue.UpdateTileMappings(
                            texture,
                            n as u32,
                            Some(coords.as_ptr()),
                            Some(sizes.as_ptr()),
                            None::<&ID3D12Heap>,
                            n as u32,
                            Some(flags.as_ptr()),
                            None,
                            Some(counts.as_ptr()),
                            D3D12_TILE_MAPPING_FLAG_NONE,
                        );
                    }
                }
                STATS[6].fetch_add(1, Ordering::Relaxed);
                STATS[7].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);

                let mut retired_runs = Vec::new();
                let mut reclaimed_bytes = 0u64;
                let mut reclaimed_bits = 0u32;
                for &(_, _, level) in &unmap {
                    let taken = match level {
                        Some(l) => {
                            if l < 32 {
                                reclaimed_bits |= 1 << l;
                            }
                            given.levels[l].take()
                        }
                        None => {
                            reclaimed_bits |= self.packed_bits();
                            given.packed.take()
                        }
                    };
                    if let Some(runs) = taken {
                        for &(heap_index, start, count) in &runs {
                            retired_runs.push((heap_index, start, count));
                            reclaimed_bytes += u64::from(count) * TILE;
                        }
                    }
                }
                // Clear claim bits before exposing the unmapped bits in `backed`
                // so a concurrent card-write claim always zeroes reclaimed levels.
                if reclaimed_bits != 0 {
                    self.unclaim(reclaimed_bits);
                }
                self.sync_backed(given);
                RECLAIMED[0].fetch_add(n as u64, Ordering::Relaxed);
                RECLAIMED[1].fetch_add(reclaimed_bytes, Ordering::Relaxed);
                retired = (!retired_runs.is_empty()).then(|| RetiredTiles::new(retired_runs, reclaimed_bytes));
            }
        }

        // (subresource, tiles, the level if it has tiles of its own) to map now.
        let mut wanted: Vec<(u32, u32, Option<usize>)> = Vec::new();
        for level in 0..first_packed {
            if given.levels[level as usize].is_none() && now(level) {
                wanted.push((level, self.standard[level as usize], Some(level as usize)));
            }
        }
        if self.packed > 0 && given.packed.is_none() && (first_packed..self.levels).any(now) {
            wanted.push((first_packed, self.packed, None));
        }
        let mut mapped_any = false;
        for (subresource, tiles, level) in wanted {
            let Some((heap, index, runs)) = take(device, tiles) else {
                continue;
            };
            let at = D3D12_TILED_RESOURCE_COORDINATE {
                X: 0,
                Y: 0,
                Z: 0,
                Subresource: subresource,
            };
            let size = D3D12_TILE_REGION_SIZE {
                NumTiles: tiles,
                ..Default::default()
            };
            let began = std::time::Instant::now();
            // One region, the subresource's tiles in order, on as many
            // ranges of the heap as the runs taken.
            if runs.len() <= STACK_RANGES {
                let flags = [D3D12_TILE_RANGE_FLAG_NONE; STACK_RANGES];
                let mut starts = [0u32; STACK_RANGES];
                let mut counts = [0u32; STACK_RANGES];
                for (i, &(start, count)) in runs.iter().enumerate() {
                    starts[i] = start;
                    counts[i] = count;
                }
                // SAFETY: a live reserved texture and a live heap of ours; one
                // region of `tiles` tiles from the subresource's first tile;
                // `runs.len()` ranges whose counts add up to `tiles`, in three
                // stack arrays of at least that length that live through the call.
                unsafe {
                    queue.UpdateTileMappings(
                        texture,
                        1,
                        Some(&raw const at),
                        Some(&raw const size),
                        &heap,
                        runs.len() as u32,
                        Some(flags.as_ptr()),
                        Some(starts.as_ptr()),
                        Some(counts.as_ptr()),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
            } else {
                let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; runs.len()];
                let starts: Vec<u32> = runs.iter().map(|run| run.0).collect();
                let counts: Vec<u32> = runs.iter().map(|run| run.1).collect();
                // SAFETY: same as above, with heap-allocated arrays for >16 ranges.
                unsafe {
                    queue.UpdateTileMappings(
                        texture,
                        1,
                        Some(&raw const at),
                        Some(&raw const size),
                        &heap,
                        runs.len() as u32,
                        Some(flags.as_ptr()),
                        Some(starts.as_ptr()),
                        Some(counts.as_ptr()),
                        D3D12_TILE_MAPPING_FLAG_NONE,
                    );
                }
            }
            STATS[6].fetch_add(1, Ordering::Relaxed);
            STATS[7].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
            let level_runs: Vec<(usize, u32, u32)> =
                runs.into_iter().map(|(start, count)| (index, start, count)).collect();
            match level {
                Some(level) => given.levels[level] = Some(level_runs),
                None => given.packed = Some(level_runs),
            }
            mapped_any = true;
            STATS[2].fetch_add(u64::from(tiles) * TILE, Ordering::Relaxed);
            STATS[5].fetch_add(1, Ordering::Relaxed);
        }
        let backed = if mapped_any {
            self.sync_backed(given)
        } else {
            self.backed.load(Ordering::Relaxed)
        };
        let raw_visible = named & backed & self.every_level();
        let floor = self.demote_floor.load(Ordering::Acquire);
        let visible = if floor > 0 && floor < 32 {
            let masked = raw_visible & !((1u32 << floor) - 1);
            if masked != 0 { masked } else { raw_visible }
        } else {
            raw_visible
        };
        let shown = if visible == 0 { u32::MAX } else { visible.trailing_zeros() };
        let old = self.shown.swap(shown, Ordering::Relaxed);
        // (Under `given`'s lock: one thread at a time moves a texture in or
        // out of the blank count.)
        if old == u32::MAX && shown != u32::MAX {
            BLANK[0].fetch_sub(1, Ordering::Relaxed);
            BLANK[1].fetch_sub(self.full, Ordering::Relaxed);
        } else if old != u32::MAX && shown == u32::MAX {
            BLANK[0].fetch_add(1, Ordering::Relaxed);
            BLANK[1].fetch_add(self.full, Ordering::Relaxed);
        }
        (old != shown, retired)
    }

    /// The first level a view of the texture may show; `None` while no
    /// level is both held and given (the views are then as the title made
    /// them, and read zeros).
    pub(crate) fn first_shown(&self) -> Option<u32> {
        let shown = self.shown.load(Ordering::Relaxed);
        (shown != u32::MAX).then_some(shown)
    }

    /// Whether level `level` has tiles (lock-free).
    #[cfg(test)]
    fn has_tiles(&self, level: u32) -> bool {
        level < self.levels && level < 32 && (self.backed.load(Ordering::Acquire) & (1 << level)) != 0
    }

    /// Whether every level has tiles (lock-free).
    #[cfg(test)]
    fn whole(&self) -> bool {
        let every = self.every_level();
        (self.backed.load(Ordering::Acquire) & every) == every
    }

    /// A bit for every level of the texture.
    pub(crate) fn every_level(&self) -> u32 {
        if self.levels >= 32 { u32::MAX } else { (1 << self.levels) - 1 }
    }

    /// The bytes of the tiles it has now.
    pub(crate) fn given_bytes(&self) -> u64 {
        let given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        given.total_tiles() * TILE
    }

    /// Records that this tiled texture was actively used at `moment`.
    pub(crate) fn touch(&self, moment: u64) {
        self.last_used.store(moment, Ordering::Relaxed);
    }

    /// Last moment (`crate::host::moment()`) this texture was touched.
    pub(crate) fn last_used(&self) -> u64 {
        self.last_used.load(Ordering::Relaxed)
    }

    /// Current top-mip demotion floor (`0` means full resolution).
    pub(crate) fn demote_floor(&self) -> u32 {
        self.demote_floor.load(Ordering::Acquire)
    }

    /// Phase 1 of fence-gated top-mip demotion: clamps `first_shown()` to
    /// `target_shown` FIRST so the caller rewrites live GPU SRV descriptors
    /// (`MostDetailedMip = new_shown`) and records `(fence_gate, frame_gate)`.
    /// Does NOT unmap any tiles yet — [`Self::commit_demotion_on`] performs
    /// `UpdateTileMappings(..., NULL)` strictly after the GPU fence completes.
    pub(crate) fn begin_demotion(&self, target_shown: u32, fence_gate: u64, frame_gate: u64) -> bool {
        if target_shown == 0 || target_shown >= self.levels || target_shown >= 32 {
            return false;
        }
        let _given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        let backed = self.backed.load(Ordering::Acquire);
        let card = self.card.load(Ordering::Relaxed);
        let held = self.held.load(Ordering::Relaxed);
        let named = held | card;
        let below_mask = (1u32 << target_shown) - 1;
        if card & below_mask != 0 || backed & below_mask == 0 {
            return false;
        }
        let visible_after = named & backed & self.every_level() & !below_mask;
        if visible_after == 0 {
            return false;
        }
        let new_shown = visible_after.trailing_zeros();
        let old_shown = self.shown.load(Ordering::Relaxed);
        if old_shown != u32::MAX && new_shown <= old_shown {
            return false;
        }
        self.demote_floor.store(target_shown, Ordering::Release);
        self.shown.store(new_shown, Ordering::Release);
        if old_shown == u32::MAX {
            BLANK[0].fetch_sub(1, Ordering::Relaxed);
            BLANK[1].fetch_sub(self.full, Ordering::Relaxed);
        }
        *self.pending_demotion.lock().unwrap_or_else(|e| e.into_inner()) = Some(PendingDemotion {
            target_shown,
            fence_gate,
            frame_gate,
        });
        true
    }

    /// Phase 2 of fence-gated top-mip demotion: unmaps top mips `< target_shown`
    /// via `UpdateTileMappings(..., D3D12_TILE_RANGE_FLAG_NULL)` ONLY after
    /// `finished_fence >= fence_gate && current_frame >= frame_gate` (proving
    /// all prior draws that could read the old `MostDetailedMip` descriptor have
    /// completed on the GPU). Returns `(levels_unmapped, bytes, RetiredTiles)`.
    pub(crate) fn commit_demotion_on(
        &self,
        queue: &ID3D12CommandQueue,
        texture: &ID3D12Resource,
        finished_fence: u64,
        current_frame: u64,
    ) -> Option<(u32, u64, RetiredTiles)> {
        let pending = {
            let mut slot = self.pending_demotion.lock().unwrap_or_else(|e| e.into_inner());
            let p = (*slot)?;
            if self.demote_floor.load(Ordering::Acquire) < p.target_shown {
                *slot = None;
                return None;
            }
            if finished_fence < p.fence_gate || current_frame < p.frame_gate {
                return None;
            }
            slot.take()?
        };

        let mut given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        // Re-check under `given` lock in case `promote()` or `card_wrote()` raced.
        if self.demote_floor.load(Ordering::Acquire) < pending.target_shown {
            return None;
        }
        let card = self.card.load(Ordering::Relaxed);
        let first_packed = self.standard.len() as u32;
        let limit = pending.target_shown.min(first_packed);
        let mut unmap: Vec<(u32, u32, usize)> = Vec::new();
        for level in 0..limit {
            if card & (1 << level) != 0 {
                continue;
            }
            if given.levels[level as usize].is_some() {
                unmap.push((level, self.standard[level as usize], level as usize));
            }
        }
        if unmap.is_empty() {
            return None;
        }

        let n = unmap.len();
        let began = std::time::Instant::now();
        if n <= STACK_RANGES {
            let mut coords = [D3D12_TILED_RESOURCE_COORDINATE::default(); STACK_RANGES];
            let mut sizes = [D3D12_TILE_REGION_SIZE::default(); STACK_RANGES];
            let flags = [D3D12_TILE_RANGE_FLAG_NULL; STACK_RANGES];
            let mut counts = [0u32; STACK_RANGES];
            for (i, &(subresource, tiles, _)) in unmap.iter().enumerate() {
                coords[i] = D3D12_TILED_RESOURCE_COORDINATE {
                    X: 0,
                    Y: 0,
                    Z: 0,
                    Subresource: subresource,
                };
                sizes[i] = D3D12_TILE_REGION_SIZE {
                    NumTiles: tiles,
                    ..Default::default()
                };
                counts[i] = tiles;
            }
            // SAFETY: live reserved texture; descriptor views already clamped and
            // prior GPU submissions completed (`finished_fence >= fence_gate`).
            unsafe {
                queue.UpdateTileMappings(
                    texture,
                    n as u32,
                    Some(coords.as_ptr()),
                    Some(sizes.as_ptr()),
                    None::<&ID3D12Heap>,
                    n as u32,
                    Some(flags.as_ptr()),
                    None,
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        } else {
            let coords: Vec<D3D12_TILED_RESOURCE_COORDINATE> = unmap
                .iter()
                .map(|&(subresource, _, _)| D3D12_TILED_RESOURCE_COORDINATE {
                    X: 0,
                    Y: 0,
                    Z: 0,
                    Subresource: subresource,
                })
                .collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> = unmap
                .iter()
                .map(|&(_, tiles, _)| D3D12_TILE_REGION_SIZE {
                    NumTiles: tiles,
                    ..Default::default()
                })
                .collect();
            let flags = vec![D3D12_TILE_RANGE_FLAG_NULL; n];
            let counts: Vec<u32> = unmap.iter().map(|&(_, tiles, _)| tiles).collect();
            // SAFETY: same as above, heap-allocated for >16 regions.
            unsafe {
                queue.UpdateTileMappings(
                    texture,
                    n as u32,
                    Some(coords.as_ptr()),
                    Some(sizes.as_ptr()),
                    None::<&ID3D12Heap>,
                    n as u32,
                    Some(flags.as_ptr()),
                    None,
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
        }
        STATS[6].fetch_add(1, Ordering::Relaxed);
        STATS[7].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);

        let mut retired_runs = Vec::new();
        let mut reclaimed_bytes = 0u64;
        let mut reclaimed_bits = 0u32;
        for &(_, _, level) in &unmap {
            if level < 32 {
                reclaimed_bits |= 1 << level;
            }
            if let Some(runs) = given.levels[level].take() {
                for &(heap_index, start, count) in &runs {
                    retired_runs.push((heap_index, start, count));
                    reclaimed_bytes += u64::from(count) * TILE;
                }
            }
        }
        if reclaimed_bits != 0 {
            self.unclaim(reclaimed_bits);
        }
        self.sync_backed(&given);
        RECLAIMED[0].fetch_add(n as u64, Ordering::Relaxed);
        RECLAIMED[1].fetch_add(reclaimed_bytes, Ordering::Relaxed);
        DEMOTED[0].fetch_add(1, Ordering::Relaxed);
        DEMOTED[1].fetch_add(n as u64, Ordering::Relaxed);
        DEMOTED[2].fetch_add(reclaimed_bytes, Ordering::Relaxed);
        (!retired_runs.is_empty()).then(|| (n as u32, reclaimed_bytes, RetiredTiles::new(retired_runs, reclaimed_bytes)))
    }

    /// Re-promotes a demoted texture back to full resolution (`demote_floor = 0`),
    /// cancelling any uncommitted Phase-1 demotion. Returns `true` if the texture
    /// was demoted and needs its top mips restored/re-uploaded.
    pub(crate) fn promote(&self) -> bool {
        let prev = self.demote_floor.swap(0, Ordering::AcqRel);
        if prev == 0 {
            return false;
        }
        *self.pending_demotion.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let _given = self.given.lock().unwrap_or_else(|e| e.into_inner());
        let backed = self.backed.load(Ordering::Acquire);
        let named = self.held.load(Ordering::Relaxed) | self.card.load(Ordering::Relaxed);
        let visible = named & backed & self.every_level();
        if visible != 0 {
            let restored = visible.trailing_zeros();
            self.shown.store(restored, Ordering::Release);
        }
        DEMOTED[3].fetch_add(1, Ordering::Relaxed);
        true
    }

    fn has_pending_demotion(&self) -> bool {
        self.pending_demotion
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

struct DemotionCandidate {
    copy: Weak<crate::resource::HostTexture>,
    memory: u64,
    resource_key: usize,
}

// SAFETY: `Weak<HostTexture>` is `Send` (fields of `HostTexture` are `Send + Sync`).
unsafe impl Send for DemotionCandidate {}

static DEMOTION_CANDIDATES: Mutex<Option<HashMap<u64, DemotionCandidate>>> = Mutex::new(None);

/// Registers a live Tier-2 tiled texture as a weak candidate for pressure-driven
/// idle top-mip demotion (`Weak` ensures zero impact on texture eviction).
pub(crate) fn register_candidate(
    serial: u64,
    copy: &Arc<crate::resource::HostTexture>,
    memory: u64,
    resource_key: usize,
) {
    if !demote_wanted() || copy.tiled.is_none() {
        return;
    }
    let mut guard = DEMOTION_CANDIDATES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    map.entry(serial).or_insert_with(|| DemotionCandidate {
        copy: Arc::downgrade(copy),
        memory,
        resource_key,
    });
}

/// Periodic 1-Hz sweep (invoked from `texture::let_go_idle`) that:
/// 1. Commits Phase-2 fence-gated tile unmapping for textures whose descriptor
///    rewrite (`MostDetailedMip = target_shown`) has cleared the GPU fence.
/// 2. Under VRAM pressure (`VRAM > 72%` or `residency::pressured()`), initiates
///    Phase-1 top-mip demotion on idle large (`>= floor()`) tiled textures.
/// 3. Below comfort band (`VRAM < 68%` and not host-pressured), proactively
///    re-promotes previously demoted textures back to full Series X resolution.
pub(crate) fn sweep_demotions() {
    if !wanted() || !demote_wanted() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let live: Vec<(Arc<crate::resource::HostTexture>, u64, usize)> = {
        let mut guard = DEMOTION_CANDIDATES.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = guard.as_mut() else {
            return;
        };
        if map.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(map.len());
        map.retain(|_, c| {
            if let Some(copy) = c.copy.upgrade() {
                out.push((copy, c.memory, c.resource_key));
                true
            } else {
                false
            }
        });
        out
    };
    if live.is_empty() {
        return;
    }

    let (started, finished) = crate::host::submissions();
    let current_frame = crate::frame::current();

    // Phase 2: commit fence-gated unmappings whose descriptor clamps have cleared the GPU.
    let mut committed_tex = 0u64;
    let mut committed_levels = 0u64;
    let mut committed_bytes = 0u64;
    for (copy, _, _) in &live {
        let Some(tiled) = copy.tiled.as_ref() else {
            continue;
        };
        if !tiled.has_pending_demotion() {
            continue;
        }
        if let Some((levels, bytes, retired)) =
            tiled.commit_demotion_on(host.queue(), &copy.texture, finished, current_frame)
        {
            crate::host::retire(retired);
            committed_tex += 1;
            committed_levels += u64::from(levels);
            committed_bytes += bytes;
        }
    }

    let memory = host.video_memory();
    let pressure = evaluate_demotion_pressure(
        memory,
        crate::pool::empty_block_bytes(),
        crate::residency::pressured(),
        demote_vram_percent(),
    );

    // Phase 1 or Proactive Re-Promotion.
    let mut begun_tex = 0usize;
    let mut repromoted_tex = 0usize;
    match pressure {
        DemotionPressure::Moderate { vram_pct } | DemotionPressure::Severe { vram_pct } => {
            let (idle_ms, idle_frames) = if matches!(pressure, DemotionPressure::Severe { .. }) {
                (DEMOTE_IDLE_SEVERE_MS, DEMOTE_IDLE_SEVERE_FRAMES)
            } else {
                (DEMOTE_IDLE_MODERATE_MS, DEMOTE_IDLE_MODERATE_FRAMES)
            };
            let min_bytes = floor();
            let fence_gate = started.saturating_add(DEMOTE_FENCE_LAG);
            let frame_gate = current_frame.saturating_add(DEMOTE_FRAME_LAG);
            for (copy, mem_va, res_key) in &live {
                if begun_tex >= DEMOTE_PER_SWEEP {
                    break;
                }
                if crate::texture::host_wrote(*res_key) {
                    continue;
                }
                let Some(tiled) = copy.tiled.as_ref() else {
                    continue;
                };
                if tiled.has_pending_demotion() {
                    continue;
                }
                let is_idle = crate::host::long_ago(tiled.last_used(), idle_ms, idle_frames);
                let Some(current_shown) = tiled.first_shown() else {
                    continue;
                };
                let Some(target) = choose_demote_target(
                    pressure,
                    tiled.full,
                    min_bytes,
                    tiled.standard.len() as u32,
                    tiled.levels,
                    current_shown,
                    tiled.backed.load(Ordering::Acquire),
                    tiled.card.load(Ordering::Relaxed),
                    is_idle,
                ) else {
                    continue;
                };
                if tiled.begin_demotion(target, fence_gate, frame_gate) {
                    // Rewrite live SRV descriptors to `MostDetailedMip = target`
                    // BEFORE Phase 2 unmaps tiles after `fence_gate` completes.
                    crate::descriptor::host::levels_changed(*mem_va);
                    begun_tex += 1;
                }
            }
            if committed_tex > 0 || begun_tex > 0 {
                tracing::info!(
                    vram_pct,
                    demoted_committed = committed_tex,
                    levels_unmapped = committed_levels,
                    reclaimed_mib = format_args!("{:.2}", committed_bytes as f64 / (1024.0 * 1024.0)),
                    pending_fence_gate = begun_tex,
                    total_reclaimed_mib = DEMOTED[2].load(Ordering::Relaxed) >> 20,
                    "tier-2 tiled mip-tail demotion sweep"
                );
            }
        }
        DemotionPressure::None { can_repromote } => {
            if committed_tex > 0 {
                tracing::info!(
                    demoted_committed = committed_tex,
                    levels_unmapped = committed_levels,
                    reclaimed_mib = format_args!("{:.2}", committed_bytes as f64 / (1024.0 * 1024.0)),
                    total_reclaimed_mib = DEMOTED[2].load(Ordering::Relaxed) >> 20,
                    "tier-2 tiled mip-tail demotion committed"
                );
            }
            if can_repromote {
                for (copy, mem_va, _) in &live {
                    if repromoted_tex >= DEMOTE_PER_SWEEP {
                        break;
                    }
                    let Some(tiled) = copy.tiled.as_ref() else {
                        continue;
                    };
                    if tiled.promote() {
                        crate::descriptor::host::levels_waiting(*mem_va);
                        crate::descriptor::host::levels_changed(*mem_va);
                        repromoted_tex += 1;
                    }
                }
                if repromoted_tex > 0 {
                    tracing::info!(
                        repromoted = repromoted_tex,
                        total_repromotions = DEMOTED[3].load(Ordering::Relaxed),
                        "tier-2 tiled textures re-promoted to full mip resolution"
                    );
                }
            }
        }
    }
}

impl Drop for Tiled {
    fn drop(&mut self) {
        let given = self.given.get_mut().unwrap_or_else(|e| e.into_inner());
        let mut runs = Vec::new();
        let mut bytes = 0u64;
        for level_runs in given.levels.iter().flatten() {
            for &(index, start, tiles) in level_runs {
                runs.push((index, start, tiles));
                bytes += u64::from(tiles) * TILE;
            }
        }
        if let Some(packed_runs) = &given.packed {
            for &(index, start, tiles) in packed_runs {
                runs.push((index, start, tiles));
                bytes += u64::from(tiles) * TILE;
            }
        }
        return_runs(&runs);
        STATS[0].fetch_sub(1, Ordering::Relaxed);
        STATS[1].fetch_sub(self.full, Ordering::Relaxed);
        STATS[2].fetch_sub(bytes, Ordering::Relaxed);
        if *self.shown.get_mut() == u32::MAX {
            BLANK[0].fetch_sub(1, Ordering::Relaxed);
            BLANK[1].fetch_sub(self.full, Ordering::Relaxed);
        }
    }
}

/// Keeps a two-dimensional shader view (2D or 2D array) from the levels
/// before `first` of a texture of `levels` levels: it starts at `first` if
/// it started before it, shows as many levels fewer (clamped to the
/// remaining levels), and reduces `ResourceMinLODClamp` by the same shift so
/// the effective minimum LOD stays at the same resource level.
pub(crate) fn show_from(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, first: u32, levels: u32) {
    if first >= levels {
        return;
    }
    let clamp = |most: &mut u32, mips: &mut u32, min_lod: &mut f32| {
        if first <= *most {
            return;
        }
        let fewer = first - *most;
        *most = first;
        if *mips != u32::MAX {
            *mips = mips.saturating_sub(fewer).clamp(1, levels - first);
        }
        if *min_lod > 0.0 {
            *min_lod = (*min_lod - fewer as f32).max(0.0);
        }
    };
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: `ViewDimension` selects `Texture2D`.
            let view = unsafe { &mut desc.Anonymous.Texture2D };
            clamp(&mut view.MostDetailedMip, &mut view.MipLevels, &mut view.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: `ViewDimension` selects `Texture2DArray`.
            let view = unsafe { &mut desc.Anonymous.Texture2DArray };
            clamp(&mut view.MostDetailedMip, &mut view.MipLevels, &mut view.ResourceMinLODClamp);
        }
        _ => {}
    }
}

/// For the log, with the texture queue's line: what this saves, and what
/// the mapping calls cost since the line before.
pub(crate) fn status() {
    static BEFORE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    if s[4] == 0 {
        return;
    }
    let calls = s[6] - BEFORE[0].swap(s[6], Ordering::Relaxed);
    let micros = s[7] - BEFORE[1].swap(s[7], Ordering::Relaxed);
    tracing::info!(
        "textures with levels on demand: {} alive ({} made so far), {} MB of the card given to {} levels in all, \
         {} MB of the card not allocated for levels the title does not hold ({} MB if every level had memory); \
         the pool's heaps {} MB ({} given back to the card so far); since the line before: {} calls of \
         UpdateTileMappings in {} microseconds; in all {} calls, {} microseconds; {} uploads left a held \
         level without memory (no room on the card, or no heap), {} uploads were recorded again for one",
        s[0],
        s[4],
        s[2] >> 20,
        s[5],
        s[1].saturating_sub(s[2]) >> 20,
        s[1] >> 20,
        s[3] >> 20,
        s[8],
        calls,
        micros,
        s[6],
        s[7],
        s[9],
        s[10],
    );
    let blank: Vec<u64> = BLANK.iter().map(|b| b.load(Ordering::Relaxed)).collect();
    tracing::info!(
        "textures with levels on demand that show no level now (no level both held and given tiles: their views read zeros, \
         a black texture): {} of {} alive, {} MB if all their levels had memory; levels given tiles on a full card since the start \
         (VICTORY_TILES_SMALL_WHEN_FULL_MB={}): {}",
        blank[0],
        s[0],
        blank[1] >> 20,
        small_when_full_mb(),
        blank[2],
    );
    let reclaimed: [u64; 4] = std::array::from_fn(|i| RECLAIMED[i].load(Ordering::Relaxed));
    let demoted: [u64; 4] = std::array::from_fn(|i| DEMOTED[i].load(Ordering::Relaxed));
    tracing::info!(
        "tile reclamation (VICTORY_TILES_RECLAIM={}): {} levels reclaimed, {} MB unmapped total, \
         {} MB awaiting GPU fence, {} MB returned to pool; \
         tier-2 mip demotion (VICTORY_TILES_DEMOTE={}, threshold={}%): {} textures / {} levels demoted ({} MB reclaimed), {} re-promotions",
        if reclaim_wanted() { 1 } else { 0 },
        reclaimed[0],
        reclaimed[1] >> 20,
        reclaimed[2] >> 20,
        reclaimed[3] >> 20,
        if demote_wanted() { 1 } else { 0 },
        demote_vram_percent(),
        demoted[0],
        demoted[1],
        demoted[2] >> 20,
        demoted[3],
    );
}

/// On the card, without the title: what levels on demand relies on, with
/// the pool and the mapping call of this module on a device of the test's
/// own. One test, one device: the pool is one for the process, and its
/// heaps belong to the device that made them.
///
/// Run by name, alone, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored levels_on_demand --nocapture`
///
/// It runs on the SOFTWARE adapter unless `VICTORY_TILES_TEST_CARD=1`: on
/// that adapter nothing reaches the card or its driver, so nothing it does
/// can reset the driver or touch the screen. Wherever it runs, the debug
/// layer is switched on if Windows has it (the optional feature "Graphics
/// Tools"), its messages are printed after every step, and the test stops
/// at the first error or lost device with the step's name, before
/// anything else is sent. On the card it does not run without the layer,
/// and no wait is longer than 1.5 s.
#[cfg(test)]
mod tests {
    use windows::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::{Interface, s};

    use super::*;
    use crate::host::{compile, transition, warp_adapter};

    const SIDE: u32 = 1024;
    const LEVELS: u16 = 11;

    /// Two views of one texture read at the same place and level: the
    /// first as the layer makes it for a texture whose level 0 has no
    /// memory, the second as the title asked.
    const LOOK: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors = 2)), UAV(u0), StaticSampler(s0, filter = FILTER_MIN_MAG_MIP_POINT)"
Texture2D<float4> clamped : register(t0);
Texture2D<float4> whole : register(t1);
SamplerState nearest : register(s0);
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void look() {
    float4 a = clamped.SampleLevel(nearest, float2(0.3, 0.3), 0);
    float4 b = whole.SampleLevel(nearest, float2(0.3, 0.3), 0);
    words.Store(0, (uint)(a.x * 255.0 + 0.5));
    words.Store(4, (uint)(b.x * 255.0 + 0.5));
}
"#;

    /// Where each level of a texture lies in a buffer of all of them: its
    /// place, its rows, the bytes of a row that are the level's own, and
    /// the buffer's bytes.
    struct Laid {
        footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
        rows: Vec<u32>,
        wide: Vec<u64>,
        total: u64,
    }

    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        event: HANDLE,
        submitted: u64,
        /// The debug layer's messages, if Windows has the layer.
        messages: Option<ID3D12InfoQueue>,
        /// What the test is doing, for the message when it stops.
        step: &'static str,
        /// On the card, not on the software adapter.
        card: bool,
    }

    impl Rig {
        /// The software adapter, or the default one with
        /// `VICTORY_TILES_TEST_CARD=1`; `None` without one, without tiled
        /// resources of tier 2, or on the card without the debug layer.
        fn new() -> Option<Rig> {
            // SAFETY: plain object creation on a device of our own.
            unsafe {
                let card = std::env::var_os("VICTORY_TILES_TEST_CARD").is_some_and(|v| v == "1");
                // The debug layer, before the device is made: a call that
                // is not well formed is then refused and named by the
                // runtime, not handed to the driver.
                let mut layer: Option<ID3D12Debug> = None;
                let layered = D3D12GetDebugInterface(&mut layer).is_ok() && layer.is_some();
                layer.iter().for_each(|l| l.EnableDebugLayer());
                if card && !layered {
                    eprintln!("no D3D12 debug layer (Windows optional feature \"Graphics Tools\"): not run on the card without it");
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                if card {
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                } else {
                    let adapter = warp_adapter().ok()?;
                    D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                }
                let device = device?;
                let messages = device.cast::<ID3D12InfoQueue>().ok();
                let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
                device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .ok()?;
                eprintln!(
                    "levels on demand, {}: tiled resources tier {}, debug layer {}",
                    if card { "ON THE CARD" } else { "on the software adapter" },
                    options.TiledResourcesTier.0,
                    if messages.is_some() { "on" } else { "not installed" }
                );
                if options.TiledResourcesTier.0 < 2 {
                    return None;
                }
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
                Some(Rig {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    event,
                    submitted: 0,
                    messages,
                    step: "making the device",
                    card,
                })
            }
        }

        /// Prints what the debug layer said since the last call; the first
        /// error among it (as `host.rs` `log_messages` reads them).
        fn said(&self) -> Option<String> {
            let queue = self.messages.as_ref()?;
            let mut first_error = None;
            // SAFETY: each message is read into a buffer of the size the
            // queue reports, then the queue is emptied.
            unsafe {
                for i in 0..queue.GetNumStoredMessages() {
                    let mut size = 0;
                    if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                        continue;
                    }
                    let mut buffer = vec![0u64; size.div_ceil(8)];
                    let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                        let m = &*message;
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        eprintln!("D3D12 says (severity {}, id {}): {text}", m.Severity.0, m.ID.0);
                        if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                            first_error = Some(text.to_owned());
                        }
                    }
                }
                queue.ClearStoredMessages();
            }
            first_error
        }

        /// Stops the test, naming the step, if the device is lost or the
        /// debug layer reported an error: nothing more is sent after it.
        fn check(&self, what: &str) {
            let error = self.said();
            // SAFETY: a plain query.
            let removed = unsafe { self.device.GetDeviceRemovedReason() };
            assert!(removed.is_ok(), "{}, {what}: the device is lost: {removed:?}", self.step);
            assert!(error.is_none(), "{}, {what}: the debug layer reports an error: {}", self.step, error.unwrap_or_default());
        }

        /// Gives tiles as a submission would, and stops if the mapping
        /// call was refused.
        fn give(&self, tiled: &Tiled, texture: &ID3D12Resource, levels: u32) {
            tiled.give_on(&self.queue, &self.device, texture, levels, levels);
            self.check("tiles given");
        }

        /// Records with `record`, runs the list behind what is already on
        /// the queue (tile mappings among it), waits for the card and
        /// opens the list again.
        fn run(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) {
            record(&self.list);
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                // Never without end: a step is a few copies of a few
                // megabytes. On the card the limit stays under Windows' own
                // two seconds; the software adapter, which compiles its
                // shader on first use and resets nothing, gets longer.
                let limit = if self.card { 1500 } else { 20_000 };
                let waited = WaitForSingleObject(self.event, limit);
                self.check("a submission");
                assert!(
                    waited == WAIT_OBJECT_0,
                    "{}: a submission did not finish within {limit} ms; stopped before another is sent",
                    self.step
                );
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        fn buffer(&self, kind: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES, bytes: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
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
                self.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                    .unwrap();
            }
            out.unwrap()
        }

        /// A square texture description of `side` texels and `levels`
        /// levels; `reserved`: in the layout a texture without memory of
        /// its own must have.
        fn desc(format: DXGI_FORMAT, side: u32, levels: u16, reserved: bool) -> D3D12_RESOURCE_DESC {
            D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(side),
                Height: side,
                DepthOrArraySize: 1,
                MipLevels: levels,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: if reserved { D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE } else { D3D12_TEXTURE_LAYOUT_UNKNOWN },
                ..Default::default()
            }
        }

        /// A texture without memory, in the state copies into it need, and
        /// how the card cuts it.
        fn reserved(&self, desc: &D3D12_RESOURCE_DESC) -> (ID3D12Resource, Tiled) {
            let mut made: Option<ID3D12Resource> = None;
            // SAFETY: plain creation; a texture that is no target has no clear value.
            let asked = unsafe {
                self.device
                    .CreateReservedResource(desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut made)
            };
            // The layer's reason first, if it refused.
            self.check("a texture without memory made");
            asked.expect("a reserved texture");
            let texture = made.expect("a reserved texture");
            let tiled = Tiled::make(&self.device, &texture, u32::from(desc.MipLevels)).expect("the card says how it cuts the texture");
            (texture, tiled)
        }

        fn footprints(&self, desc: &D3D12_RESOURCE_DESC) -> Laid {
            let levels = usize::from(desc.MipLevels);
            let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); levels];
            let mut rows = vec![0u32; levels];
            let mut wide = vec![0u64; levels];
            let mut total = 0u64;
            // SAFETY: outputs for `levels` subresources.
            unsafe {
                self.device.GetCopyableFootprints(
                    desc,
                    0,
                    levels as u32,
                    0,
                    Some(footprints.as_mut_ptr()),
                    Some(rows.as_mut_ptr()),
                    Some(wide.as_mut_ptr()),
                    Some(&mut total),
                );
            }
            Laid {
                footprints,
                rows,
                wide,
                total,
            }
        }

        /// An upload buffer of `total` bytes in which the bytes of each
        /// level are `byte(level)`.
        fn filled(&self, layout: &Laid, byte: impl Fn(usize) -> u8) -> ID3D12Resource {
            let (footprints, rows) = (&layout.footprints, &layout.rows);
            let upload = self.buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, layout.total);
            // SAFETY: the buffer is `total` bytes; each level's rows lie inside it.
            unsafe {
                let mut data = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut data)).unwrap();
                for (level, f) in footprints.iter().enumerate() {
                    let len = rows[level] as usize * f.Footprint.RowPitch as usize;
                    std::slice::from_raw_parts_mut(data.cast::<u8>().add(f.Offset as usize), len).fill(byte(level));
                }
                upload.Unmap(0, None);
            }
            upload
        }

        /// Copies every level of `texture` from `upload`, then every level
        /// back, and gives for each level the first sixteen bytes read of
        /// its first row and of the row and column half way (another tile
        /// of a level of several).
        fn round_trip(
            &mut self,
            texture: &ID3D12Resource,
            layout: &Laid,
            upload: Option<&ID3D12Resource>,
        ) -> Vec<[Vec<u8>; 2]> {
            let (footprints, rows) = (&layout.footprints, &layout.rows);
            let back = self.buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, layout.total);
            self.run(|list| {
                // SAFETY: an open list; the texture and buffers outlive the run.
                unsafe {
                    if let Some(upload) = upload {
                        for (level, f) in footprints.iter().enumerate() {
                            list.CopyTextureRegion(&sub(texture, level as u32), 0, 0, 0, &placed(upload, *f), None);
                        }
                    }
                    list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    for (level, f) in footprints.iter().enumerate() {
                        list.CopyTextureRegion(&placed(&back, *f), 0, 0, 0, &sub(texture, level as u32), None);
                    }
                    list.ResourceBarrier(&[transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
                }
            });
            let mut read = Vec::new();
            // SAFETY: a read-back buffer of `total` bytes the finished list copied into.
            unsafe {
                let mut data = std::ptr::null_mut();
                back.Map(0, None, Some(&mut data)).unwrap();
                for (level, f) in footprints.iter().enumerate() {
                    let pitch = f.Footprint.RowPitch as usize;
                    let len = rows[level] as usize * pitch;
                    let bytes = std::slice::from_raw_parts(data.cast::<u8>().cast_const().add(f.Offset as usize), len);
                    // The bytes of a row that are the level's own (its
                    // blocks or texels; the rest of the pitch is padding).
                    let own = layout.wide[level] as usize;
                    let wide = own.min(len).min(16);
                    let first = bytes[..wide].to_vec();
                    let middle = (rows[level] as usize / 2) * pitch + ((own / 2) & !15);
                    let far = if rows[level] >= 2 && own >= 32 && middle + wide <= len {
                        bytes[middle..middle + wide].to_vec()
                    } else {
                        first.clone()
                    };
                    read.push([first, far]);
                }
                back.Unmap(0, None);
            }
            read
        }
    }

    fn sub(texture: &ID3D12Resource, index: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
        }
    }

    fn placed(buffer: &ID3D12Resource, footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        }
    }

    fn all(bytes: &[Vec<u8>; 2], want: u8) -> bool {
        bytes.iter().all(|part| part.iter().all(|&b| b == want))
    }

    /// The levels from `first` on, a bit a level.
    fn from_level(first: u32) -> u32 {
        (u32::MAX << first) & ((1 << u32::from(LEVELS)) - 1)
    }

    /// (i) Levels arrive one by one: after each step every level with
    /// tiles reads back its own bytes and every level without reads zeros.
    fn levels_arrive(rig: &mut Rig, format: DXGI_FORMAT) {
        rig.step = if format == DXGI_FORMAT_BC7_UNORM { "step 1, levels arrive (BC7)" } else { "step 1, levels arrive (BC3)" };
        let desc = Rig::desc(format, SIDE, LEVELS, true);
        let (texture, tiled) = rig.reserved(&desc);
        // (The card packs the levels smaller than a tile into shared
        // tiles; the software adapter may give each its own.)
        eprintln!(
            "format {}: {} levels with tiles of their own ({:?} tiles), {} tiles for the packed ones",
            format.0,
            tiled.standard.len(),
            tiled.standard,
            tiled.packed
        );
        assert!(!rig.card || tiled.packed > 0, "on the card a 1024 texture has levels smaller than a tile");
        let layout = rig.footprints(&desc);
        let upload = rig.filled(&layout, |level| level as u8 + 1);
        for first in [3u32, 1, 0] {
            // Ahead of the list on the queue, as at a submission.
            rig.give(&tiled, &texture, from_level(first));
            let read = rig.round_trip(&texture, &layout, Some(&upload));
            for (level, bytes) in read.iter().enumerate() {
                let want = if tiled.has_tiles(level as u32) { level as u8 + 1 } else { 0 };
                assert!(all(bytes, want), "format {} from level {first}: level {level} reads {bytes:?}, wanted {want}", format.0);
            }
        }
        assert!(tiled.whole());
        assert_eq!(tiled.first_shown(), Some(0));
    }

    /// (ii) A copy on the card into a level without tiles is dropped; the
    /// same copy in a list that runs after the level was given lands.
    fn a_card_copy_lands_after_the_give(rig: &mut Rig) {
        rig.step = "step 2, a copy on the card into a level without tiles";
        let format = DXGI_FORMAT_BC7_UNORM;
        let desc = Rig::desc(format, SIDE, LEVELS, true);
        let (texture, tiled) = rig.reserved(&desc);
        let layout = rig.footprints(&desc);
        // The last level with tiles of its own, and a texture of its size
        // with every byte 0x5a to copy from.
        let level = tiled.standard.len() as u32 - 1;
        let source_desc = Rig::desc(format, SIDE >> level, 1, false);
        let source_layout = rig.footprints(&source_desc);
        let source_bytes = rig.filled(&source_layout, |_| 0x5a);
        let mut made: Option<ID3D12Resource> = None;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        // SAFETY: plain texture creation.
        unsafe {
            rig.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &source_desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut made)
                .unwrap();
        }
        let source = made.unwrap();
        rig.run(|list| {
            // SAFETY: an open list and live resources.
            unsafe {
                list.CopyTextureRegion(&sub(&source, 0), 0, 0, 0, &placed(&source_bytes, source_layout.footprints[0]), None);
                list.ResourceBarrier(&[transition(&source, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                // No tiles yet: this copy goes nowhere.
                list.CopyTextureRegion(&sub(&texture, level), 0, 0, 0, &sub(&source, 0), None);
            }
        });
        let before = rig.round_trip(&texture, &layout, None);
        assert!(all(&before[level as usize], 0), "a copy into a level without tiles left {:?}", before[level as usize]);
        // As `texture.rs` `card_writes` has it: the give at the list's
        // submission, the copy in the list.
        assert_ne!(tiled.claim(tiled.with_level(level)), 0, "the level is new");
        rig.give(&tiled, &texture, tiled.with_level(level));
        rig.run(|list| {
            // SAFETY: as above.
            unsafe { list.CopyTextureRegion(&sub(&texture, level), 0, 0, 0, &sub(&source, 0), None) };
        });
        let after = rig.round_trip(&texture, &layout, None);
        assert!(all(&after[level as usize], 0x5a), "the copy after the give left {:?}", after[level as usize]);
        assert_eq!(tiled.claim(tiled.with_level(level)), 0, "claimed once");
    }

    /// (iii) Tiles a texture gave back go to the next one, which reads
    /// what was written into it and zeros where it has no tiles; what a
    /// level given and not written reads is printed (the reason no level
    /// is given outside a list that writes it).
    fn a_reused_tile_reads_as_written(rig: &mut Rig) {
        rig.step = "step 3, a reused tile";
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let layout = rig.footprints(&desc);
        let (first, first_tiles) = rig.reserved(&desc);
        let level = first_tiles.standard.len() as u32 - 1;
        rig.give(&first_tiles, &first, 1 << level);
        let upload = rig.filled(&layout, |_| 0xa5);
        let read = rig.round_trip(&first, &layout, Some(&upload));
        assert!(all(&read[level as usize], 0xa5));
        let runs = first_tiles.given.lock().unwrap().all_runs();
        drop(first_tiles);
        drop(first);
        let (second, second_tiles) = rig.reserved(&desc);
        rig.give(&second_tiles, &second, 1 << level);
        let reused = second_tiles.given.lock().unwrap().all_runs() == runs;
        let unwritten = rig.round_trip(&second, &layout, None);
        eprintln!(
            "a level given tiles and not written reads {:?} (its tiles are the first texture's: {reused})",
            &unwritten[level as usize][0][..4]
        );
        let upload = rig.filled(&layout, |_| 0x3c);
        let read = rig.round_trip(&second, &layout, Some(&upload));
        assert!(all(&read[level as usize], 0x3c), "the reused level reads {:?}", read[level as usize]);
        assert!(all(&read[0], 0), "level 0 has no tiles and reads {:?}", read[0]);
    }

    /// (iv) A level whose tiles come from several runs of the heap (single
    /// free tiles between taken ones) reads as written in every tile.
    fn a_level_on_several_ranges(rig: &mut Rig) {
        rig.step = "step 4, a level on several ranges of the heap";
        // Three single tiles; the first and the last go back, the middle
        // one stays taken.
        let singles: Vec<_> = (0..3).map(|_| take(&rig.device, 1).expect("a tile")).collect();
        give_back(singles[0].1, &singles[0].2);
        give_back(singles[2].1, &singles[2].2);
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let layout = rig.footprints(&desc);
        let (texture, tiled) = rig.reserved(&desc);
        // Level 1 of a 1024 texture: four tiles.
        assert!(tiled.standard.len() > 1 && tiled.standard[1] >= 2);
        rig.give(&tiled, &texture, 1 << 1);
        let runs = tiled.given.lock().unwrap().all_runs().len();
        eprintln!("level 1 ({} tiles) was mapped on {runs} ranges", tiled.standard[1]);
        assert!(runs >= 2, "the single free tiles were not used first");
        let upload = rig.filled(&layout, |_| 0x77);
        let read = rig.round_trip(&texture, &layout, Some(&upload));
        assert!(all(&read[1], 0x77), "level 1 reads {:?}", read[1]);
        give_back(singles[1].1, &singles[1].2);
    }

    /// (v) A view made by [`show_from`] of a texture whose level 0 has no
    /// tiles returns level 1's colour for a sample at level 0; the view as
    /// the title asked returns zero.
    fn the_view_starts_at_the_first_level_with_tiles(rig: &mut Rig) -> Option<()> {
        rig.step = "step 5, the view from the first level with tiles";
        let cs = compile(LOOK, s!("look"), s!("cs_5_1")).ok()?;
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let desc = Rig::desc(format, SIDE, LEVELS, true);
        let layout = rig.footprints(&desc);
        let (texture, tiled) = rig.reserved(&desc);
        // Every level but the largest; each level's bytes are sixteen
        // times its number plus one.
        rig.give(&tiled, &texture, from_level(1));
        assert_eq!(tiled.first_shown(), Some(1));
        let upload = rig.filled(&layout, |level| (level as u8 + 1) * 16);
        rig.round_trip(&texture, &layout, Some(&upload));
        // SAFETY: plain object creation; the views are made in a heap
        // that lives to the end of the function, of a live texture.
        let (root, pipeline, views, words, back) = unsafe {
            let root: ID3D12RootSignature = rig.device.CreateRootSignature(0, &cs[..]).ok()?;
            let pipeline: ID3D12PipelineState = rig
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.as_ptr().cast(),
                        BytecodeLength: cs.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            let views: ID3D12DescriptorHeap = rig
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            let as_asked = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: 5768,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: u32::MAX,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            let mut shown = as_asked;
            show_from(&mut shown, tiled.first_shown()?, u32::from(LEVELS));
            assert_eq!(shown.Anonymous.Texture2D.MostDetailedMip, 1);
            let step = rig.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let mut at = views.GetCPUDescriptorHandleForHeapStart();
            rig.device.CreateShaderResourceView(&texture, Some(&raw const shown), at);
            at.ptr += step;
            rig.device.CreateShaderResourceView(&texture, Some(&raw const as_asked), at);
            let words = rig.buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                256,
            );
            let back = rig.buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, 256);
            (root, pipeline, views, words, back)
        };
        // Before the one dispatch: the pipeline and both views are sound.
        rig.check("the pipeline and the views made");
        rig.run(|list| {
            // SAFETY: an open list and live objects.
            unsafe {
                list.ResourceBarrier(&[transition(&texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                list.SetDescriptorHeaps(&[Some(views.clone())]);
                list.SetComputeRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
                list.SetComputeRootUnorderedAccessView(1, words.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);
                list.ResourceBarrier(&[transition(&words, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                list.CopyBufferRegion(&back, 0, &words, 0, 16);
            }
        });
        // SAFETY: a read-back buffer the finished list copied sixteen bytes into.
        let read = unsafe {
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            back.Map(0, Some(&range), Some(&mut data)).unwrap();
            let read = std::slice::from_raw_parts(data.cast::<u32>(), 2).to_vec();
            back.Unmap(0, None);
            read
        };
        eprintln!("a sample at level 0: {} through the view from level 1, {} through the view as asked", read[0], read[1]);
        assert_eq!(read[0], 32, "the view from the first level with tiles reads level 1");
        assert_eq!(read[1], 0, "the view from level 0 reads a level without tiles");
        Some(())
    }

    fn a_full_card_keeps_new_levels_waiting(rig: &mut Rig) {
        rig.step = "step 6, no room on the card: held levels wait, and get their tiles when asked again";
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let (texture, tiled) = rig.reserved(&desc);
        let layout = rig.footprints(&desc);
        let upload = rig.filled(&layout, |level| level as u8 + 1);
        let read_as_given = |rig: &mut Rig, what: &str| {
            let read = rig.round_trip(&texture, &layout, Some(&upload));
            for (level, bytes) in read.iter().enumerate() {
                let want = if tiled.has_tiles(level as u32) { level as u8 + 1 } else { 0 };
                assert!(all(bytes, want), "{what}: level {level} reads {bytes:?}, wanted {want}");
            }
        };
        // No room: nothing is given, the held levels are said to wait,
        // nothing is shown, every level reads zeros.
        let (changed, waiting) = tiled.uploaded_on(&rig.queue, &rig.device, &texture, from_level(3), false);
        rig.check("an upload without room");
        assert!(!changed && waiting, "without room: changed {changed}, waiting {waiting}");
        assert_eq!(tiled.given_bytes(), 0, "without room no tile is given");
        assert_eq!(tiled.first_shown(), None);
        read_as_given(rig, "without room");
        // Asked again with room: the same levels get theirs.
        let (changed, waiting) = tiled.uploaded_on(&rig.queue, &rig.device, &texture, from_level(3), true);
        rig.check("the upload again, with room");
        assert!(changed && !waiting, "with room: changed {changed}, waiting {waiting}");
        assert!(tiled.has_tiles(3) && !tiled.has_tiles(2));
        assert_eq!(tiled.first_shown(), Some(3));
        read_as_given(rig, "with room");
        // More levels held and no room again: the ones that have tiles
        // are brought up to date, the new ones wait, the view stays.
        let before = tiled.given_bytes();
        let (changed, waiting) = tiled.uploaded_on(&rig.queue, &rig.device, &texture, from_level(1), false);
        rig.check("more levels held, without room");
        assert!(!changed && waiting, "more levels without room: changed {changed}, waiting {waiting}");
        assert_eq!(tiled.given_bytes(), before);
        assert_eq!(tiled.first_shown(), Some(3));
        read_as_given(rig, "more levels without room");
        // The card's own write is not kept waiting, and brings only its
        // own level: level 1 still waits although it is held.
        assert_eq!(tiled.claim(1 << 0), 1 << 0);
        let changed = tiled.card_wrote_on(&rig.queue, &rig.device, &texture, 1 << 0);
        rig.check("the card's own write, without room");
        assert!(changed && tiled.has_tiles(0) && !tiled.has_tiles(1) && !tiled.has_tiles(2));
        assert_eq!(tiled.first_shown(), Some(0));
        assert_eq!(tiled.claim(1 << 0), 0, "a level with tiles is not claimed again");
        // With room everything held gets its tiles.
        let (_, waiting) = tiled.uploaded_on(&rig.queue, &rig.device, &texture, from_level(1), true);
        rig.check("the last upload, with room");
        assert!(!waiting && tiled.whole());
        read_as_given(rig, "everything given");
    }

    fn a_claim_given_up_is_made_again(rig: &mut Rig) {
        rig.step = "step 7, a claim whose zeros never landed is given up";
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let (_texture, tiled) = rig.reserved(&desc);
        let level = tiled.with_level(2);
        assert_eq!(tiled.claim(level), level, "the level is new");
        assert_eq!(tiled.claim(level), 0, "claimed once");
        // The list that claimed it was reset without being submitted.
        tiled.unclaim(level);
        assert_eq!(tiled.claim(level), level, "claimed again after it was given up");
        // The small levels the card packs together are claimed together.
        let small = tiled.with_level(u32::from(LEVELS) - 1);
        assert!(!rig.card || small.count_ones() > 1, "on the card a 1024 texture has several levels in one tile");
        assert_eq!(tiled.claim(1 << (u32::from(LEVELS) - 1)), 1 << (u32::from(LEVELS) - 1));
        tiled.unclaim(small);
        assert_eq!(tiled.claim(small), small);
    }

    struct TestPoolGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl TestPoolGuard {
        fn acquire() -> Self {
            static LOCK: Mutex<()> = Mutex::new(());
            let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
            clear_pool_for_test();
            Self(guard)
        }
    }

    impl Drop for TestPoolGuard {
        fn drop(&mut self) {
            clear_pool_for_test();
        }
    }

    fn clear_pool_for_test() {
        let drained: Vec<Option<Heap>> = {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            let live = pool.iter().flatten().count() as u64;
            if live > 0 {
                STATS[3].fetch_sub(live * u64::from(HEAP_TILES) * TILE, Ordering::Relaxed);
            }
            std::mem::take(&mut *pool)
        };
        drop(drained);
    }

    fn runs_overlap(a: &[(usize, u32, u32)], b: &[(usize, u32, u32)]) -> bool {
        a.iter().any(|&(ha, sa, ca)| {
            b.iter()
                .any(|&(hb, sb, cb)| ha == hb && sa < sb + cb && sb < sa + ca)
        })
    }

    /// (viii) Deferred tile reclamation: unmapping unheld upper mip levels
    /// via `D3D12_TILE_RANGE_FLAG_NULL` zeroes GPU reads for those levels,
    /// keeps the unmapped runs reserved while `RetiredTiles` is alive
    /// (preventing premature reuse), returns them to `POOL` on drop after GPU
    /// completion so the next texture reuses the exact runs without growing
    /// the heap pool, never reclaims card-written levels, and honors
    /// `reclaim = false`.
    fn reclamation_unmaps_defers_and_reuses_tiles(rig: &mut Rig) {
        rig.step = "step 8, deferred tile reclamation and reuse";
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let layout = rig.footprints(&desc);
        let (tex_a, tiled_a) = rig.reserved(&desc);
        assert!(tiled_a.standard.len() >= 2, "need at least 2 standard levels for reclamation test");

        // 1) Allocate all levels, upload distinct pattern, verify readback.
        let (changed, missing, ret) =
            tiled_a.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_a, from_level(0), true, true);
        rig.check("allocate all levels on tex_a");
        assert!(changed && !missing && ret.is_none());
        assert!(tiled_a.whole());
        assert_eq!(tiled_a.first_shown(), Some(0));
        let upload_a = rig.filled(&layout, |level| (level as u8 + 1) * 17);
        let read_a = rig.round_trip(&tex_a, &layout, Some(&upload_a));
        for (level, bytes) in read_a.iter().enumerate() {
            let want = (level as u8 + 1) * 17;
            assert!(all(bytes, want), "initial upload: level {level} reads {bytes:?}, wanted {want}");
        }

        // 2) Unmap upper mip levels (0 and 1) via `uploaded_on_reclaim(..., true)`.
        let expected_runs: Vec<(usize, u32, u32)> = {
            let g = tiled_a.given.lock().unwrap();
            let mut r = Vec::new();
            r.extend_from_slice(g.levels[0].as_ref().unwrap());
            r.extend_from_slice(g.levels[1].as_ref().unwrap());
            r
        };
        let expected_bytes = u64::from(tiled_a.standard[0] + tiled_a.standard[1]) * TILE;
        let stats2_before = STATS[2].load(Ordering::Relaxed);
        let rec_levels_before = RECLAIMED[0].load(Ordering::Relaxed);
        let rec_unmapped_before = RECLAIMED[1].load(Ordering::Relaxed);
        let rec_pending_before = RECLAIMED[2].load(Ordering::Relaxed);
        let rec_returned_before = RECLAIMED[3].load(Ordering::Relaxed);

        let (changed, missing, retired) =
            tiled_a.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_a, from_level(2), true, true);
        rig.check("unmap levels 0 and 1 via NULL UpdateTileMappings");
        assert!(changed && !missing, "unmapping levels 0..2 should advance first_shown without missing held levels");
        let retired = retired.expect("unmapping levels 0 and 1 must return RetiredTiles");
        assert_eq!(retired.bytes(), expected_bytes);
        assert_eq!(retired.runs(), expected_runs.as_slice());
        assert!(!tiled_a.has_tiles(0) && !tiled_a.has_tiles(1));
        assert!(tiled_a.has_tiles(2));
        assert_eq!(tiled_a.without(0b111), 0b011);
        assert_eq!(tiled_a.first_shown(), Some(2));
        // Unmapped levels must be unclaimed so a later card write zeroes them.
        assert_eq!(tiled_a.claim(1 << 0), 1 << 0);
        tiled_a.unclaim(1 << 0);

        // Verify readback: unmapped levels now read zeros, remaining levels keep data.
        let read_unmapped = rig.round_trip(&tex_a, &layout, None);
        assert!(all(&read_unmapped[0], 0), "unmapped level 0 must read zeros, got {:?}", read_unmapped[0]);
        assert!(all(&read_unmapped[1], 0), "unmapped level 1 must read zeros, got {:?}", read_unmapped[1]);
        for (level, bytes) in read_unmapped.iter().enumerate().skip(2) {
            let want = (level as u8 + 1) * 17;
            assert!(all(bytes, want), "retained level {level} reads {bytes:?}, wanted {want}");
        }

        // 3) While `RetiredTiles` is held alive (in-flight GPU work), unmapped runs
        //    must NOT be reused by `POOL` and pending-fence bytes must reflect them.
        assert_eq!(RECLAIMED[0].load(Ordering::Relaxed) - rec_levels_before, 2);
        assert_eq!(RECLAIMED[1].load(Ordering::Relaxed) - rec_unmapped_before, expected_bytes);
        assert!(RECLAIMED[2].load(Ordering::Relaxed) >= rec_pending_before + expected_bytes);
        assert_eq!(RECLAIMED[3].load(Ordering::Relaxed), rec_returned_before);
        assert_eq!(
            STATS[2].load(Ordering::Relaxed),
            stats2_before,
            "STATS[2] must stay unchanged while RetiredTiles awaits GPU fence"
        );

        let (tex_probe, tiled_probe) = rig.reserved(&desc);
        let (_, _, probe_ret) =
            tiled_probe.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_probe, 1 << 0, true, true);
        rig.check("probe allocation while RetiredTiles is alive");
        assert!(probe_ret.is_none());
        let probe_runs = tiled_probe.given.lock().unwrap().all_runs();
        assert!(
            !runs_overlap(&probe_runs, retired.runs()),
            "POOL prematurely reused retired runs {expected_runs:?} as {probe_runs:?}"
        );
        // Drain queue before releasing probe texture and retired tiles.
        rig.run(|_| {});
        drop(tiled_probe);
        drop(tex_probe);

        // 4) After GPU fence completes, drop `RetiredTiles` and verify tiles return to
        //    `POOL`, `STATS[2]` decreases, and a new texture reuses those exact runs
        //    without allocating a second 64 MiB heap.
        let stats2_before_drop = STATS[2].load(Ordering::Relaxed);
        let heap_bytes_before = STATS[3].load(Ordering::Relaxed);
        drop(retired);
        assert_eq!(STATS[2].load(Ordering::Relaxed), stats2_before_drop - expected_bytes);
        assert_eq!(RECLAIMED[2].load(Ordering::Relaxed), rec_pending_before);
        assert_eq!(RECLAIMED[3].load(Ordering::Relaxed) - rec_returned_before, expected_bytes);

        let (tex_b, tiled_b) = rig.reserved(&desc);
        let (changed_b, missing_b, ret_b) = tiled_b.uploaded_on_reclaim(
            &rig.queue,
            &rig.device,
            &tex_b,
            (1 << 0) | (1 << 1),
            true,
            true,
        );
        rig.check("tex_b reuses reclaimed tile runs");
        assert!(changed_b && !missing_b && ret_b.is_none());
        let reused_runs = tiled_b.given.lock().unwrap().all_runs();
        assert_eq!(
            reused_runs, expected_runs,
            "subsequent reserved texture should reuse the exact reclaimed tile runs"
        );
        assert_eq!(
            STATS[3].load(Ordering::Relaxed),
            heap_bytes_before,
            "reusing reclaimed tiles must not allocate a second 64 MiB heap"
        );
        let upload_b = rig.filled(&layout, |_| 0x42);
        let read_b = rig.round_trip(&tex_b, &layout, Some(&upload_b));
        assert!(all(&read_b[0], 0x42), "reused level 0 reads {:?}", read_b[0]);
        assert!(all(&read_b[1], 0x42), "reused level 1 reads {:?}", read_b[1]);
        assert!(all(&read_b[2], 0), "unmapped level 2 on tex_b reads {:?}", read_b[2]);

        // 5) Verify `reclaim = false` keeps unheld tiles mapped, and card-written
        //    levels are NEVER reclaimed when `held` drops them.
        let (changed_no_rec, missing_no_rec, ret_no_rec) =
            tiled_b.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_b, 1 << 1, true, false);
        rig.check("reclaim = false keeps tiles mapped");
        assert!(changed_no_rec && !missing_no_rec && ret_no_rec.is_none());
        assert!(tiled_b.has_tiles(0), "reclaim = false must keep level 0 mapped");
        assert_eq!(tiled_b.first_shown(), Some(1));

        // Mark level 0 as card-written, then drop all `held` levels with `reclaim = true`.
        let card_changed = tiled_b.card_wrote_on(&rig.queue, &rig.device, &tex_b, 1 << 0);
        rig.check("card_wrote_on level 0");
        assert!(card_changed);
        assert_eq!(tiled_b.first_shown(), Some(0));

        let (changed_card, missing_card, ret_card) =
            tiled_b.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_b, 0, true, true);
        rig.check("reclaim preserves card-written level 0 while unmapping level 1");
        assert!(!changed_card && !missing_card);
        assert!(tiled_b.has_tiles(0), "card-written level 0 must NEVER be reclaimed");
        assert!(!tiled_b.has_tiles(1), "unheld non-card level 1 must be reclaimed");
        assert_eq!(tiled_b.first_shown(), Some(0));
        let ret_card = ret_card.expect("level 1 should be retired");
        assert_eq!(ret_card.bytes(), u64::from(tiled_b.standard[1]) * TILE);

        // Also reclaim every remaining level (including packed mips if present) on tex_a.
        let (changed_all, missing_all, ret_all) =
            tiled_a.uploaded_on_reclaim(&rig.queue, &rig.device, &tex_a, 0, true, true);
        rig.check("reclaim all remaining levels on tex_a");
        assert!(changed_all && !missing_all);
        assert_eq!(tiled_a.first_shown(), None);
        assert_eq!(tiled_a.given_bytes(), 0);
        assert!(ret_all.is_some());

        // Wait for GPU before dropping retired tiles and resources.
        rig.run(|_| {});
        drop(ret_card);
        drop(ret_all);
        drop(tiled_b);
        drop(tex_b);
        drop(tiled_a);
        drop(tex_a);
    }

    /// Non-ignored WARP end-to-end test for deferred tile reclamation and reuse.
    #[test]
    fn deferred_tile_reclamation_on_warp() {
        let _pool_guard = TestPoolGuard::acquire();
        let Some(mut rig) = Rig::new() else {
            eprintln!("no adapter with tiled resources of tier 2, or no debug layer; skipped");
            return;
        };
        rig.check("the device made");
        reclamation_unmaps_defers_and_reuses_tiles(&mut rig);
        rig.step = "the end";
        rig.check("reclamation test done");
    }

    /// On the software adapter unless `VICTORY_TILES_TEST_CARD=1`.
    #[test]
    #[ignore = "on the card: run it by name with --ignored --nocapture"]
    fn levels_on_demand_on_the_card() {
        let _pool_guard = TestPoolGuard::acquire();
        let Some(mut rig) = Rig::new() else {
            eprintln!("no adapter with tiled resources of tier 2, or no debug layer for a run on the card; skipped");
            return;
        };
        rig.check("the device made");
        levels_arrive(&mut rig, DXGI_FORMAT_BC7_UNORM);
        levels_arrive(&mut rig, DXGI_FORMAT_BC3_UNORM);
        a_card_copy_lands_after_the_give(&mut rig);
        a_reused_tile_reads_as_written(&mut rig);
        a_level_on_several_ranges(&mut rig);
        a_full_card_keeps_new_levels_waiting(&mut rig);
        a_claim_given_up_is_made_again(&mut rig);
        reclamation_unmaps_defers_and_reuses_tiles(&mut rig);
        if the_view_starts_at_the_first_level_with_tiles(&mut rig).is_none() {
            eprintln!("the view test's shader or objects could not be made; that part was skipped");
        }
        rig.step = "the end";
        rig.check("every step done");
        eprintln!(
            "all steps passed {}",
            if rig.card { "ON THE CARD" } else { "on the software adapter; VICTORY_TILES_TEST_CARD=1 is the run on the card" }
        );
        let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
        eprintln!(
            "{} calls of UpdateTileMappings took {} microseconds in all ({} a call); the pool's heaps {} MB, {} given back",
            s[6],
            s[7],
            s[7] / s[6].max(1),
            s[3] >> 20,
            s[8]
        );
    }

    #[test]
    fn parse_reclaim_defaults_on_and_honors_opt_out() {
        assert!(parse_reclaim(None));
        assert!(parse_reclaim(Some("1")));
        assert!(parse_reclaim(Some("")));
        assert!(!parse_reclaim(Some("0")));
        assert!(!parse_reclaim(Some(" 0 ")));
    }

    #[test]
    fn floor_saturates_instead_of_wrapping() {
        assert_eq!(parse_floor(None), 512 << 10);
        assert_eq!(parse_floor(Some("0")), 64 << 10);
        assert_eq!(parse_floor(Some("2048")), 2048 << 10);
        assert_eq!(parse_floor(Some(&(1u64 << 54).to_string())), u64::MAX);
        assert_eq!(parse_floor(Some(&u64::MAX.to_string())), u64::MAX);
    }

    #[test]
    fn show_from_clamps_2d_and_2d_array_and_adjusts_min_lod() {
        let mut srv2d = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 12,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 3.5,
                },
            },
        };
        show_from(&mut srv2d, 3, 11);
        // SAFETY: `ViewDimension` is `TEXTURE2D`.
        let v2d = unsafe { srv2d.Anonymous.Texture2D };
        assert_eq!(v2d.MostDetailedMip, 3);
        // Clamped to remaining levels (11 - 3 = 8), not 12 - 3 = 9.
        assert_eq!(v2d.MipLevels, 8);
        assert_eq!(v2d.ResourceMinLODClamp, 0.5);

        let mut srv_arr = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: 5768,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 6,
                    FirstArraySlice: 0,
                    ArraySize: 1,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 1.5,
                },
            },
        };
        show_from(&mut srv_arr, 2, 11);
        // SAFETY: `ViewDimension` is `TEXTURE2DARRAY`.
        let varr = unsafe { srv_arr.Anonymous.Texture2DArray };
        assert_eq!(varr.MostDetailedMip, 2);
        assert_eq!(varr.MipLevels, 4);
        assert_eq!(varr.ResourceMinLODClamp, 0.0);
    }

    #[test]
    fn card_wrote_failure_does_not_poison_card_mask() {
        let Some(host) = host() else {
            return;
        };
        // A dummy live resource (never mapped because `take()` rejects `HEAP_TILES + 1`).
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = Rig::desc(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 2, false);
        let mut tex: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation on the host device.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut tex,
                )
                .unwrap();
        }
        let tex = tex.unwrap();
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(TILE, Ordering::Relaxed);
        BLANK[0].fetch_add(1, Ordering::Relaxed);
        BLANK[1].fetch_add(TILE, Ordering::Relaxed);
        let mut tiled = Tiled {
            // Level 0 demands > HEAP_TILES so `take()` fails deterministically.
            standard: vec![HEAP_TILES + 1, 1],
            packed: 0,
            levels: 2,
            full: TILE,
            given: Mutex::new(Given {
                levels: vec![None, None],
                packed: None,
            }),
            backed: AtomicU32::new(0),
            shown: AtomicU32::new(u32::MAX),
            held: AtomicU32::new(0),
            card: AtomicU32::new(0),
            claimed: AtomicU32::new(0),
            last_used: AtomicU64::new(0),
            demote_floor: AtomicU32::new(0),
            pending_demotion: Mutex::new(None),
        };
        assert_eq!(tiled.claim(1 << 0), 1 << 0);
        let changed = tiled.card_wrote_on(host.queue(), &host.device, &tex, 1 << 0);
        assert!(!changed);
        assert_eq!(tiled.card.load(Ordering::Relaxed), 0, "failed tile allocation must not poison `card`");
        assert_eq!(tiled.first_shown(), None);
        assert_eq!(tiled.claim(1 << 0), 1 << 0, "failed level must be unclaimed");
        // Even if level 0 later becomes allocatable, an upload with `room == false`
        // that does not hold level 0 must not treat level 0 as card-owned.
        tiled.standard[0] = 1;
        let (changed, waiting) = tiled.uploaded_on(host.queue(), &host.device, &tex, 1 << 1, false);
        assert!(!changed && waiting);
        assert!(!tiled.has_tiles(0));
        assert_eq!(tiled.given_bytes(), 0);
    }

    #[test]
    fn demotion_pressure_and_target_thresholds() {
        let budget = 10_000u64 << 20;
        // 70% used, no empty pool blocks, not host pressured -> None { can_repromote: false } (>= 68%)
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 7_000 << 20, 0, 0]), 0, false, 72),
            DemotionPressure::None { can_repromote: false }
        );
        // 65% used -> None { can_repromote: true } (< 68%)
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 6_500 << 20, 0, 0]), 0, false, 72),
            DemotionPressure::None { can_repromote: true }
        );
        // 75% gross minus 10% empty pool blocks = 65% net -> does NOT falsely demote!
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 7_500 << 20, 0, 0]), 1_000 << 20, false, 72),
            DemotionPressure::None { can_repromote: true }
        );
        // 74% net -> Moderate
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 7_400 << 20, 0, 0]), 0, false, 72),
            DemotionPressure::Moderate { vram_pct: 74 }
        );
        // 50% VRAM but host `residency::pressured()` -> Moderate
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 5_000 << 20, 0, 0]), 0, true, 72),
            DemotionPressure::Moderate { vram_pct: 50 }
        );
        // 86% net -> Severe
        assert_eq!(
            evaluate_demotion_pressure(Some([budget, 8_600 << 20, 0, 0]), 0, false, 72),
            DemotionPressure::Severe { vram_pct: 86 }
        );

        let all_11 = (1u32 << 11) - 1;
        let mod_p = DemotionPressure::Moderate { vram_pct: 75 };
        let sev_p = DemotionPressure::Severe { vram_pct: 88 };

        // Active close-up texture (`is_idle = false`) MUST NEVER be clamped (Spec-Trap #7)!
        assert_eq!(choose_demote_target(mod_p, 1 << 20, 512 << 10, 4, 11, 0, all_11, 0, false), None);
        assert_eq!(choose_demote_target(sev_p, 1 << 20, 512 << 10, 4, 11, 0, all_11, 0, false), None);

        // Below 512 KiB floor -> never demoted.
        assert_eq!(choose_demote_target(mod_p, 256 << 10, 512 << 10, 4, 11, 0, all_11, 0, true), None);

        // Card-written mip 0 -> never demoted.
        assert_eq!(choose_demote_target(mod_p, 1 << 20, 512 << 10, 4, 11, 0, all_11, 1 << 0, true), None);

        // Only mip 0 backed (no fallback mip below it) -> never blanked!
        assert_eq!(choose_demote_target(mod_p, 1 << 20, 512 << 10, 4, 11, 0, 1 << 0, 0, true), None);

        // Idle large texture under Moderate -> demotes 1 top mip (0 -> 1).
        assert_eq!(choose_demote_target(mod_p, 1 << 20, 512 << 10, 4, 11, 0, all_11, 0, true), Some(1));
        // Under Severe -> demotes up to 2 top mips (0 -> 2).
        assert_eq!(choose_demote_target(sev_p, 1 << 20, 512 << 10, 4, 11, 0, all_11, 0, true), Some(2));
    }

    #[test]
    fn fence_gated_top_mip_demotion_and_repromotion_on_warp() {
        let _pool_guard = TestPoolGuard::acquire();
        let Some(mut rig) = Rig::new() else {
            eprintln!("no adapter with tiled resources of tier 2, or no debug layer; skipped");
            return;
        };
        rig.step = "tier-2 fence-gated top-mip demotion and re-promotion";
        let desc = Rig::desc(DXGI_FORMAT_BC7_UNORM, SIDE, LEVELS, true);
        let layout = rig.footprints(&desc);
        let (texture, tiled) = rig.reserved(&desc);
        assert!(tiled.standard.len() >= 2);

        // 1) Allocate all mips and upload distinct byte per mip.
        let (changed, missing, _) =
            tiled.uploaded_on_reclaim(&rig.queue, &rig.device, &texture, from_level(0), true, true);
        rig.check("initial full allocation");
        assert!(changed && !missing && tiled.whole());
        assert_eq!(tiled.first_shown(), Some(0));
        let upload = rig.filled(&layout, |level| (level as u8 + 1) * 19);
        let read = rig.round_trip(&texture, &layout, Some(&upload));
        assert!(all(&read[0], 19));
        assert!(all(&read[1], 38));

        // 2) Phase 1: begin_demotion clamps first_shown() FIRST without unmapping tiles yet!
        assert!(tiled.begin_demotion(1, 10, 20));
        assert_eq!(tiled.first_shown(), Some(1), "descriptor clamp must advance immediately");
        assert_eq!(tiled.demote_floor(), 1);
        assert!(tiled.has_tiles(0), "tiles MUST remain mapped before GPU fence completes");

        // 3) Phase 2 BEFORE fence/frame gate -> MUST NOT unmap!
        assert!(tiled.commit_demotion_on(&rig.queue, &texture, 9, 20).is_none());
        assert!(tiled.commit_demotion_on(&rig.queue, &texture, 10, 19).is_none());
        assert!(tiled.has_tiles(0), "level 0 must stay mapped until fence & frame gates pass");
        let still_mapped = rig.round_trip(&texture, &layout, None);
        assert!(all(&still_mapped[0], 19), "in-flight draws still read valid mip 0 before fence");

        // 4) Phase 2 AFTER fence & frame gate -> unmaps mip 0 cleanly and returns RetiredTiles.
        let expected_bytes = u64::from(tiled.standard[0]) * TILE;
        let (levels_unmapped, bytes, retired) = tiled
            .commit_demotion_on(&rig.queue, &texture, 10, 20)
            .expect("commit_demotion_on must unmap once fence & frame gates pass");
        rig.check("Phase-2 NULL UpdateTileMappings");
        assert_eq!(levels_unmapped, 1);
        assert_eq!(bytes, expected_bytes);
        assert!(!tiled.has_tiles(0));
        assert!(tiled.has_tiles(1));
        assert_eq!(tiled.first_shown(), Some(1));

        let after_demote = rig.round_trip(&texture, &layout, None);
        assert!(all(&after_demote[0], 0), "demoted mip 0 must read zeros");
        assert!(all(&after_demote[1], 38), "retained mip 1 must keep data");

        // Wait for GPU work before returning tiles to pool.
        rig.run(|_| {});
        drop(retired);

        // 5) Re-promote -> demote_floor clears, next upload re-maps and restores mip 0!
        assert!(tiled.promote());
        assert_eq!(tiled.demote_floor(), 0);
        assert!(!tiled.promote(), "second promote is a no-op");
        let (changed, missing, _) =
            tiled.uploaded_on_reclaim(&rig.queue, &rig.device, &texture, from_level(0), true, true);
        rig.check("re-promoted full allocation");
        assert!(changed && !missing && tiled.whole());
        assert_eq!(tiled.first_shown(), Some(0));
        let restored = rig.round_trip(&texture, &layout, Some(&upload));
        assert!(all(&restored[0], 19), "re-promoted mip 0 must read restored bytes");

        // 6) Cancel uncommitted Phase-1 demotion if texture becomes active before fence gate!
        assert!(tiled.begin_demotion(1, 50, 60));
        assert_eq!(tiled.first_shown(), Some(1));
        assert!(tiled.promote());
        assert_eq!(tiled.first_shown(), Some(0), "cancelling pending demotion restores mip 0 immediately");
        assert!(tiled.commit_demotion_on(&rig.queue, &texture, 100, 100).is_none());
        assert!(tiled.has_tiles(0), "cancelled demotion must leave mip 0 tiles intact");
        rig.check("end of fence-gated demotion test");
    }
}
