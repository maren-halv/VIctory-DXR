//! Textures shaders write that the title's own code then reads on the CPU.
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (R32_FLOAT) and a displacement map (R32G32_FLOAT), and
//! its CPU code samples both, which is what boats float on; it read memory
//! nothing had written (run rw1, `VICTORY_READ_WATCH=1`: about 30 places in
//! the title read those two kinds), and boats sank.
//!
//! For textures of the kinds in `VICTORY_TEXTURE_BACK` (comma-separated
//! `<width>x<height>f<format>`; unset or `1`: the four water formats found by
//! the upstream trace; `0`: none), a list in which shaders write one copies it
//! into a read-back buffer of ours when it closes, and once the list has run
//! the texels are tiled into the texture's title memory in the console's
//! layout, as the console's GPU would have left them.
//!
//! Read-back buffers are pooled (`VICTORY_TEXBACK_POOL`, default on, `=0` off)
//! and landed texels are diffed against the last committed frame so unchanged
//! textures/subresources skip retiling and unchanged 4 KiB pages skip title
//! writes (`VICTORY_TEXBACK_DIFF`, default on, `=0` off).
//!
//! shortcut: by kind, not by texture: textures of those kinds the title never
//! reads are copied back too (about a megabyte per such list). The copy shows
//! what the list's last write left; a title read in between sees the frame
//! before, as it would before the console's GPU has run.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::object::This;

#[cfg(test)]
thread_local! {
    pub(crate) static TEST_KINDS: std::cell::Cell<Option<&'static [(u64, u32, u32)]>> =
        const { std::cell::Cell::new(None) };
}

const POOL_ALIGN: u64 = 64 << 10;
const POOL_MIN_MAX_CAP: u64 = 256 << 10;
const POOL_MAX_BUFFERS: usize = 16;
const POOL_MAX_BYTES: u64 = 16 << 20;
const POOL_PRESSURED_BYTES: u64 = 4 << 20;

const FOOTPRINT_CACHE_MAX: usize = 64;

const DIFF_CACHE_MAX_ENTRIES: usize = 24;
const DIFF_CACHE_MAX_BYTES: usize = 16 << 20;
const DIFF_CACHE_PRESSURED_BYTES: usize = 4 << 20;

const DIRTY_PAGE: usize = 4096;
const SCRATCH_SHRINK_THRESHOLD: usize = 16 << 20;
const SCRATCH_SHRINK_KEEP: usize = 2 << 20;

static POOLED_HITS: AtomicU64 = AtomicU64::new(0);
static POOLED_MISSES: AtomicU64 = AtomicU64::new(0);
static DIFF_SKIPS: AtomicU64 = AtomicU64::new(0);
static DIRTY_BYTES_WRITTEN: AtomicU64 = AtomicU64::new(0);
static FULL_BYTES_SAVED: AtomicU64 = AtomicU64::new(0);

fn pool_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXBACK_POOL").is_none_or(|v| v != "0"))
}

fn diff_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXBACK_DIFF").is_none_or(|v| v != "0"))
}

/// The kinds copied back: (width, height, format).
fn kinds() -> &'static [(u64, u32, u32)] {
    #[cfg(test)]
    if let Some(kinds) = TEST_KINDS.get() {
        return kinds;
    }
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        const DEFAULT: &str = "256x256f41,256x256f16,64x64f16,128x128f36";
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") | Err(_) => DEFAULT.to_owned(),
            Ok(spec) => spec.to_owned(),
        };
        let kinds: Vec<(u64, u32, u32)> = list
            .split(',')
            .filter_map(|k| {
                let (size, format) = k.trim().split_once('f')?;
                let (w, h) = size.split_once('x')?;
                Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
            })
            .collect();
        if !kinds.is_empty() {
            tracing::info!(?kinds, "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)");
        }
        kinds
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let kinds = kinds();
    !kinds.is_empty()
        && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket { owner, at, end, valid: AtomicBool::new(true) });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

/// Lock order: `TICKETS` -> `DIFF_CACHE`. Evicting snapshots before releasing
/// `TICKETS` prevents a concurrent `land()` from caching stale bytes for an
/// allocation that was invalidated.
fn invalidate(
    ticket_matches: impl Fn(&Ticket) -> bool,
    diff_matches: impl Fn(&DiffEntry) -> bool,
) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    let evicted = {
        let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
        tickets.retain(|weak| {
            let Some(ticket) = weak.upgrade() else { return false };
            if ticket_matches(&ticket) {
                ticket.valid.store(false, Ordering::Release);
            }
            true
        });
        evict_diff_locked(diff_matches)
    };
    drop(evicted);
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner, |entry| entry.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at, |entry| entry.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(
        |ticket| at < ticket.end && ticket.at < end,
        |entry| at < entry.end && entry.at < end,
    );
}

// --- Readback buffer pool ---

struct PooledBuffer {
    device_key: usize,
    capacity: u64,
    resource: ID3D12Resource,
}

// SAFETY: D3D12 resources are free-threaded COM objects.
unsafe impl Send for PooledBuffer {}

struct BufferPool {
    entries: Vec<PooledBuffer>,
    total_bytes: u64,
}

impl BufferPool {
    const fn new() -> Self {
        Self { entries: Vec::new(), total_bytes: 0 }
    }
}

static BUFFER_POOL: Mutex<BufferPool> = Mutex::new(BufferPool::new());

fn pool_alloc_size(total: u64) -> u64 {
    total.max(1).next_multiple_of(POOL_ALIGN)
}

fn pool_max_match_capacity(total: u64) -> u64 {
    total.saturating_mul(2).max(POOL_MIN_MAX_CAP)
}

fn acquire_buffer(
    device_key: usize,
    total: u64,
    use_pool: bool,
    alloc: impl FnOnce(u64) -> Option<ID3D12Resource>,
) -> Option<(ID3D12Resource, u64)> {
    if !use_pool {
        return Some((alloc(total)?, total));
    }
    let max_cap = pool_max_match_capacity(total);
    {
        let mut pool = BUFFER_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let best = pool
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.device_key == device_key && e.capacity >= total && e.capacity <= max_cap)
            .min_by_key(|(_, e)| e.capacity)
            .map(|(idx, _)| idx);
        if let Some(idx) = best {
            let entry = pool.entries.remove(idx);
            pool.total_bytes = pool.total_bytes.saturating_sub(entry.capacity);
            POOLED_HITS.fetch_add(1, Ordering::Relaxed);
            return Some((entry.resource, entry.capacity));
        }
    }
    POOLED_MISSES.fetch_add(1, Ordering::Relaxed);
    let alloc_size = pool_alloc_size(total);
    Some((alloc(alloc_size)?, alloc_size))
}

fn release_buffer(device_key: usize, capacity: u64, resource: ID3D12Resource) {
    let max_bytes = if crate::residency::critical_commit() {
        0
    } else if crate::residency::pressured() {
        POOL_PRESSURED_BYTES
    } else {
        POOL_MAX_BYTES
    };
    if capacity > max_bytes {
        crate::host::retire_com_async(PooledBuffer { device_key, capacity, resource });
        return;
    }
    let evicted = {
        let mut pool = BUFFER_POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.entries.push(PooledBuffer { device_key, capacity, resource });
        pool.total_bytes = pool.total_bytes.saturating_add(capacity);
        let mut evicted = Vec::new();
        while pool.entries.len() > POOL_MAX_BUFFERS || pool.total_bytes > max_bytes {
            let old = pool.entries.remove(0);
            pool.total_bytes = pool.total_bytes.saturating_sub(old.capacity);
            evicted.push(old);
        }
        evicted
    };
    if !evicted.is_empty() {
        crate::host::retire_com_async(evicted);
    }
}

/// Proactively trims pooled readback buffers and diff snapshots under RAM pressure.
pub(crate) fn trim_pool(pressured: bool, critical: bool) {
    if !pressured && !critical {
        return;
    }
    let (pool_cap, diff_cap) = if critical {
        (0u64, 0usize)
    } else {
        (POOL_PRESSURED_BYTES, DIFF_CACHE_PRESSURED_BYTES)
    };
    let evicted_bufs = {
        let mut pool = BUFFER_POOL.lock().unwrap_or_else(|e| e.into_inner());
        let mut evicted = Vec::new();
        while pool.total_bytes > pool_cap {
            let Some(old) = pool.entries.first() else { break };
            let _ = old;
            let old = pool.entries.remove(0);
            pool.total_bytes = pool.total_bytes.saturating_sub(old.capacity);
            evicted.push(old);
        }
        evicted
    };
    if !evicted_bufs.is_empty() {
        crate::host::retire_com_async(evicted_bufs);
    }
    let evicted_diffs = {
        let mut cache = DIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let mut evicted = Vec::new();
        while cache.total_bytes > diff_cap {
            let Some(_) = cache.entries.first() else { break };
            let old = cache.entries.remove(0);
            cache.total_bytes = cache.total_bytes.saturating_sub(old.linear.len());
            evicted.push(old);
        }
        evicted
    };
    drop(evicted_diffs);
}

// --- Copyable footprint cache ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FootprintKey {
    device_key: usize,
    dimension: i32,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mips: u16,
    format: i32,
    sample_count: u32,
    sample_quality: u32,
    layout: i32,
}

impl FootprintKey {
    fn from_desc(device_key: usize, desc: &D3D12_RESOURCE_DESC) -> Self {
        Self {
            device_key,
            dimension: desc.Dimension.0,
            width: desc.Width,
            height: desc.Height,
            depth_or_array: desc.DepthOrArraySize,
            mips: desc.MipLevels,
            format: desc.Format.0,
            sample_count: desc.SampleDesc.Count,
            sample_quality: desc.SampleDesc.Quality,
            layout: desc.Layout.0,
        }
    }
}

struct FootprintPlan {
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    total: u64,
}

static FOOTPRINT_CACHE: Mutex<Vec<(FootprintKey, Arc<FootprintPlan>)>> = Mutex::new(Vec::new());

fn footprint_plan(
    device: &ID3D12Device,
    device_key: usize,
    desc: &D3D12_RESOURCE_DESC,
) -> Option<Arc<FootprintPlan>> {
    let key = FootprintKey::from_desc(device_key, desc);
    {
        let mut cache = FOOTPRINT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(idx) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(idx);
            let plan = Arc::clone(&entry.1);
            cache.push(entry);
            return Some(plan);
        }
    }

    let mips = u32::from(desc.MipLevels.max(1));
    let array_size = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize.max(1))
    };
    let subresources = mips.checked_mul(array_size)?;
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays of `n` entries matching `subresources`.
    unsafe {
        device.GetCopyableFootprints(
            desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // The last footprint's advertised extent can omit row-end padding.
    // The tiler reads a pitched slice, so the mapped buffer must include it.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let plan = Arc::new(FootprintPlan { footprints, rows, mips, total });
    {
        let mut cache = FOOTPRINT_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(idx) = cache.iter().position(|(k, _)| *k == key) {
            cache.remove(idx);
        } else if cache.len() >= FOOTPRINT_CACHE_MAX {
            cache.remove(0);
        }
        cache.push((key, Arc::clone(&plan)));
    }
    Some(plan)
}

// --- Diff snapshot cache ---

struct DiffEntry {
    owner: usize,
    at: usize,
    end: usize,
    desc: victory_formats::texture::Desc,
    /// Compact active linear texel bytes for all subresources in order
    /// (excluding D3D12 row-pitch padding).
    linear: Vec<u8>,
}

struct DiffCache {
    entries: Vec<DiffEntry>,
    total_bytes: usize,
}

impl DiffCache {
    const fn new() -> Self {
        Self { entries: Vec::new(), total_bytes: 0 }
    }
}

static DIFF_CACHE: Mutex<DiffCache> = Mutex::new(DiffCache::new());

fn evict_diff_locked(matches: impl Fn(&DiffEntry) -> bool) -> Vec<DiffEntry> {
    let mut cache = DIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let mut evicted = Vec::new();
    let mut i = 0;
    while i < cache.entries.len() {
        if matches(&cache.entries[i]) {
            let entry = cache.entries.remove(i);
            cache.total_bytes = cache.total_bytes.saturating_sub(entry.linear.len());
            evicted.push(entry);
        } else {
            i += 1;
        }
    }
    evicted
}

fn take_diff(
    ticket: &Ticket,
    desc: &victory_formats::texture::Desc,
    packed_len: usize,
) -> Option<DiffEntry> {
    let mut cache = DIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let idx = cache.entries.iter().position(|e| {
        e.owner == ticket.owner
            && e.at == ticket.at
            && e.end == ticket.end
            && e.desc == *desc
            && e.linear.len() == packed_len
    })?;
    let entry = cache.entries.remove(idx);
    cache.total_bytes = cache.total_bytes.saturating_sub(entry.linear.len());
    Some(entry)
}

fn store_diff(ticket: &Ticket, entry: DiffEntry) {
    if entry.linear.len() > DIFF_CACHE_MAX_BYTES {
        return;
    }
    let evicted = {
        let mut cache = DIFF_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if !ticket.valid.load(Ordering::Acquire) {
            return;
        }
        let mut evicted = Vec::new();
        let mut i = 0;
        while i < cache.entries.len() {
            let existing = &cache.entries[i];
            if existing.owner == entry.owner && existing.at == entry.at && existing.end == entry.end {
                let old = cache.entries.remove(i);
                cache.total_bytes = cache.total_bytes.saturating_sub(old.linear.len());
                evicted.push(old);
            } else {
                i += 1;
            }
        }
        cache.total_bytes = cache.total_bytes.saturating_add(entry.linear.len());
        cache.entries.push(entry);
        while cache.entries.len() > DIFF_CACHE_MAX_ENTRIES || cache.total_bytes > DIFF_CACHE_MAX_BYTES {
            let old = cache.entries.remove(0);
            cache.total_bytes = cache.total_bytes.saturating_sub(old.linear.len());
            evicted.push(old);
        }
        evicted
    };
    drop(evicted);
}

// --- Thread-local scratch buffers ---

#[derive(Clone, Copy)]
struct SubGeom {
    mip: u32,
    slice: u32,
    src_offset: usize,
    src_len: usize,
    pitch: usize,
    depth_pitch: usize,
    row_bytes: usize,
    height: usize,
    depth: usize,
    packed_offset: usize,
    packed_len: usize,
    contiguous: bool,
}

#[derive(Default)]
struct LandScratch {
    before_tiled: Vec<u8>,
    tiled: Vec<u8>,
    sub_geoms: Vec<SubGeom>,
    changed: Vec<bool>,
    dirty_spans: Vec<(usize, usize)>,
}

impl LandScratch {
    fn maybe_shrink(&mut self, last_size: usize) {
        if last_size <= SCRATCH_SHRINK_KEEP {
            if self.before_tiled.capacity() > SCRATCH_SHRINK_THRESHOLD {
                self.before_tiled.clear();
                self.before_tiled.shrink_to(SCRATCH_SHRINK_KEEP);
            }
            if self.tiled.capacity() > SCRATCH_SHRINK_THRESHOLD {
                self.tiled.clear();
                self.tiled.shrink_to(SCRATCH_SHRINK_KEEP);
            }
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<LandScratch> = RefCell::new(LandScratch::default());
}

fn with_scratch<R>(size: usize, f: impl FnOnce(&mut LandScratch) -> R) -> R {
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => {
            let result = f(&mut scratch);
            scratch.maybe_shrink(size);
            result
        }
        Err(_) => {
            let mut local = LandScratch::default();
            f(&mut local)
        }
    })
}

// --- Public crate API ---

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy independently of the title's resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        // Register the ticket while no mapping change is in progress so a
        // concurrent remap cannot miss it.
        let ticket = {
            let _mapping = crate::watch::title_memory_reading();
            track(resource as usize, at, end)
        };
        Some(Self { copy, layout, ticket })
    }

    pub(crate) fn owner(&self) -> usize { self.ticket.owner }

    pub(crate) fn fresh(&self) -> bool { self.ticket.valid.load(Ordering::Acquire) }
}

/// The readback resources and the allocation the pixels belong to.
pub(crate) struct Back {
    buffer: Option<ID3D12Resource>,
    capacity: u64,
    device_key: usize,
    pool_eligible: bool,
    landed: AtomicBool,
    plan: Arc<FootprintPlan>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

impl Drop for Back {
    fn drop(&mut self) {
        let Some(resource) = self.buffer.take() else { return };
        if self.pool_eligible && self.landed.load(Ordering::Acquire) {
            release_buffer(self.device_key, self.capacity, resource);
        }
    }
}

fn record_on_device(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
    use_pool: bool,
    alloc: impl FnOnce(u64) -> Option<ID3D12Resource>,
) -> Option<Back> {
    if !ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let device_key = device.as_raw() as usize;
    // SAFETY: a live texture kept through the command list's state tracking.
    let desc = unsafe { texture.GetDesc() };
    let plan = footprint_plan(device, device_key, &desc)?;
    let (buffer, capacity) = acquire_buffer(device_key, plan.total, use_pool, alloc)?;
    for (i, footprint) in plan.footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        // SAFETY: an open list; the texture is in the copy-source state.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer: Some(buffer),
        capacity,
        device_key,
        pool_eligible: use_pool,
        landed: AtomicBool::new(false),
        plan,
        layout,
        ticket,
    })
}

/// Records a host copy from a texture already in COPY_SOURCE.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    record_on_device(
        &host.device,
        list,
        &target.copy.texture,
        target.layout,
        target.ticket,
        pool_on(),
        |size| host.read_back_buffer(size),
    )
}

fn note_stale() {
    static STALE: AtomicU64 = AtomicU64::new(0);
    let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
    if skipped.is_power_of_two() {
        tracing::info!(skipped, "skipping texture readbacks whose title allocation changed");
    }
}

/// Runs `write` on the title slice `[at, end)` only if `ticket` is still valid,
/// holding both the title-memory read lock and the ticket lock across the write
/// so no remap or resource release can race with it.
fn commit_if_valid(ticket: &Ticket, write: impl FnOnce(usize, usize)) -> bool {
    let _mapping = crate::watch::title_memory_reading();
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !ticket.valid.load(Ordering::Acquire) {
        return false;
    }
    write(ticket.at, ticket.end - ticket.at);
    true
}

fn build_sub_geoms(
    layout: &victory_formats::texture::Layout,
    plan: &FootprintPlan,
    mapped_len: usize,
    out: &mut Vec<SubGeom>,
) -> Option<usize> {
    out.clear();
    let bpe = layout.desc.element_bytes as usize;
    let mut packed_offset = 0usize;
    for (i, (footprint, &rows)) in plan.footprints.iter().zip(&plan.rows).enumerate() {
        let mip = i as u32 % plan.mips;
        let slice = i as u32 / plan.mips;
        let m = layout.mips.get(mip as usize)?;
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let rows = rows as usize;
        let f_depth = f.Depth as usize;
        let depth_pitch = pitch.checked_mul(rows)?;
        let src_len = depth_pitch.checked_mul(f_depth)?;
        let src_offset = usize::try_from(footprint.Offset).ok()?;
        if src_offset.checked_add(src_len)? > mapped_len {
            return None;
        }
        let row_bytes = (m.width as usize).checked_mul(bpe)?;
        let height = m.height as usize;
        let depth = m.depth as usize;
        if row_bytes > pitch || height > rows || depth > f_depth {
            return None;
        }
        let packed_len = row_bytes.checked_mul(height)?.checked_mul(depth)?;
        let next_packed = packed_offset.checked_add(packed_len)?;
        let contiguous = pitch == row_bytes && (depth == 1 || rows == height);
        out.push(SubGeom {
            mip,
            slice,
            src_offset,
            src_len,
            pitch,
            depth_pitch,
            row_bytes,
            height,
            depth,
            packed_offset,
            packed_len,
            contiguous,
        });
        packed_offset = next_packed;
    }
    Some(packed_offset)
}

fn copy_active_to_packed(src: &[u8], g: &SubGeom, dst: &mut [u8]) {
    if g.contiguous {
        dst.copy_from_slice(&src[..g.packed_len]);
        return;
    }
    let mut at = 0;
    for z in 0..g.depth {
        let z_base = z * g.depth_pitch;
        for y in 0..g.height {
            let row = &src[z_base + y * g.pitch..][..g.row_bytes];
            dst[at..at + g.row_bytes].copy_from_slice(row);
            at += g.row_bytes;
        }
    }
}

fn diff_and_update_subresource(src: &[u8], g: &SubGeom, dst: &mut [u8]) -> bool {
    if g.contiguous {
        let active = &src[..g.packed_len];
        if active == dst {
            return false;
        }
        dst.copy_from_slice(active);
        return true;
    }
    let mut changed = false;
    let mut at = 0;
    for z in 0..g.depth {
        let z_base = z * g.depth_pitch;
        for y in 0..g.height {
            let row_src = &src[z_base + y * g.pitch..][..g.row_bytes];
            let row_dst = &mut dst[at..at + g.row_bytes];
            if row_src != row_dst {
                row_dst.copy_from_slice(row_src);
                changed = true;
            }
            at += g.row_bytes;
        }
    }
    changed
}

/// Finds dirty byte spans by scanning Windows 4 KiB page-aligned chunks,
/// coalescing adjacent dirty pages, and trimming identical leading/trailing
/// bytes on each coalesced span.
fn compute_dirty_spans(base_addr: usize, before: &[u8], after: &[u8], out: &mut Vec<(usize, usize)>) {
    debug_assert_eq!(before.len(), after.len());
    out.clear();
    let len = before.len();
    if len == 0 {
        return;
    }
    let push_trimmed = |s: usize, e: usize, out: &mut Vec<(usize, usize)>| {
        let a = &before[s..e];
        let b = &after[s..e];
        if let Some(first) = a.iter().zip(b).position(|(x, y)| x != y) {
            let last = a.iter().zip(b).rposition(|(x, y)| x != y).unwrap();
            out.push((s + first, s + last + 1));
        }
    };
    let mut current: Option<(usize, usize)> = None;
    let mut offset = 0;
    while offset < len {
        let page_rem = DIRTY_PAGE - (base_addr.wrapping_add(offset) & (DIRTY_PAGE - 1));
        let end = (offset + page_rem).min(len);
        if before[offset..end] != after[offset..end] {
            match &mut current {
                Some((_, cur_end)) => *cur_end = end,
                None => current = Some((offset, end)),
            }
        } else if let Some((s, e)) = current.take() {
            push_trimmed(s, e, out);
        }
        offset = end;
    }
    if let Some((s, e)) = current {
        push_trimmed(s, e, out);
    }
}

enum ReadTitle {
    Ok,
    Stale,
    Unreadable,
}

enum PreparedLand {
    Stale,
    Abort,
    Skipped { entry: DiffEntry },
    Write { entry: Option<DiffEntry> },
}

fn prepare_land(
    ticket: &Ticket,
    layout: &victory_formats::texture::Layout,
    plan: &FootprintPlan,
    mapped: &[u8],
    diff_enabled: bool,
    scratch: &mut LandScratch,
    read_title: impl FnOnce(&mut [u8]) -> ReadTitle,
) -> PreparedLand {
    let size = ticket.end - ticket.at;
    let sub_geoms_ok = diff_enabled
        && build_sub_geoms(layout, plan, mapped.len(), &mut scratch.sub_geoms).is_some();

    if !sub_geoms_ok {
        scratch.tiled.resize(size, 0);
        match read_title(&mut scratch.tiled) {
            ReadTitle::Ok => {}
            ReadTitle::Stale => return PreparedLand::Stale,
            ReadTitle::Unreadable => return PreparedLand::Abort,
        }
        for (i, (footprint, &rows)) in plan.footprints.iter().zip(&plan.rows).enumerate() {
            let (mip, slice) = (i as u32 % plan.mips, i as u32 / plan.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            let offset = footprint.Offset as usize;
            let Some(linear) = mapped.get(offset..offset + len) else {
                return PreparedLand::Abort;
            };
            if let Err(e) = layout.tile(&mut scratch.tiled, mip, slice, linear, pitch, pitch * rows as usize) {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled"));
                return PreparedLand::Abort;
            }
        }
        scratch.changed.clear();
        scratch.changed.resize(plan.footprints.len(), true);
        scratch.dirty_spans.clear();
        if size > 0 {
            scratch.dirty_spans.push((0, size));
        }
        return PreparedLand::Write { entry: None };
    }

    let total_packed = scratch
        .sub_geoms
        .last()
        .map_or(0, |g| g.packed_offset + g.packed_len);

    let entry = if let Some(mut entry) = take_diff(ticket, &layout.desc, total_packed) {
        scratch.changed.clear();
        let mut any_changed = false;
        for g in &scratch.sub_geoms {
            let src = &mapped[g.src_offset..g.src_offset + g.src_len];
            let dst = &mut entry.linear[g.packed_offset..g.packed_offset + g.packed_len];
            let sub_changed = diff_and_update_subresource(src, g, dst);
            scratch.changed.push(sub_changed);
            any_changed |= sub_changed;
        }
        if !any_changed {
            return PreparedLand::Skipped { entry };
        }
        entry
    } else {
        let mut linear = vec![0u8; total_packed];
        for g in &scratch.sub_geoms {
            let src = &mapped[g.src_offset..g.src_offset + g.src_len];
            let dst = &mut linear[g.packed_offset..g.packed_offset + g.packed_len];
            copy_active_to_packed(src, g, dst);
        }
        scratch.changed.clear();
        scratch.changed.resize(scratch.sub_geoms.len(), true);
        DiffEntry {
            owner: ticket.owner,
            at: ticket.at,
            end: ticket.end,
            desc: layout.desc,
            linear,
        }
    };

    scratch.before_tiled.resize(size, 0);
    match read_title(&mut scratch.before_tiled) {
        ReadTitle::Ok => {}
        ReadTitle::Stale => return PreparedLand::Stale,
        ReadTitle::Unreadable => return PreparedLand::Abort,
    }
    scratch.tiled.clear();
    scratch.tiled.extend_from_slice(&scratch.before_tiled);

    for (g, &is_changed) in scratch.sub_geoms.iter().zip(&scratch.changed) {
        if !is_changed {
            continue;
        }
        let linear = &mapped[g.src_offset..g.src_offset + g.src_len];
        if let Err(e) = layout.tile(&mut scratch.tiled, g.mip, g.slice, linear, g.pitch, g.depth_pitch) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(?e, mip = g.mip, slice = g.slice, "a texture copied back could not be tiled"));
            return PreparedLand::Abort;
        }
    }

    compute_dirty_spans(ticket.at, &scratch.before_tiled, &scratch.tiled, &mut scratch.dirty_spans);
    PreparedLand::Write { entry: Some(entry) }
}

fn note_landed(size: usize) {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(
            copied,
            bytes = size,
            pooled_hits = POOLED_HITS.load(Ordering::Relaxed),
            pooled_misses = POOLED_MISSES.load(Ordering::Relaxed),
            diff_skips = DIFF_SKIPS.load(Ordering::Relaxed),
            dirty_bytes_written = DIRTY_BYTES_WRITTEN.load(Ordering::Relaxed),
            full_bytes_saved = FULL_BYTES_SAVED.load(Ordering::Relaxed),
            "texture readback landed"
        );
    }
}

struct MapGuard<'a>(&'a ID3D12Resource);

impl Drop for MapGuard<'_> {
    fn drop(&mut self) {
        // Tell D3D12 the CPU did not modify the readback buffer.
        let written = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: paired with `Map` on this readback buffer.
        unsafe { self.0.Unmap(0, Some(&written)) };
    }
}

/// Once the list has run, tile its texels back into the same title allocation.
pub(crate) fn land(back: &Back) {
    // `host.rs::read_backs` calls `land` only after the command list's GPU fence
    // has completed, so this buffer is safe to return to the pool on Drop.
    back.landed.store(true, Ordering::Release);
    if !back.ticket.valid.load(Ordering::Acquire) {
        note_stale();
        return;
    }
    let Some(buffer) = back.buffer.as_ref() else {
        return;
    };
    let Ok(total) = usize::try_from(back.plan.total) else {
        return;
    };
    let (at, end) = back.range();
    let size = end - at;

    with_scratch(size, |scratch| {
        let prepared = {
            let read_range = D3D12_RANGE { Begin: 0, End: total };
            let mut data = std::ptr::null_mut();
            // SAFETY: our read-back buffer, written by the list that has run.
            unsafe {
                if buffer.Map(0, Some(&read_range), Some(&mut data)).is_err() || data.is_null() {
                    return;
                }
            }
            let _unmap = MapGuard(buffer);
            // SAFETY: `[0, total)` is within the mapped read-back buffer.
            let mapped = unsafe { std::slice::from_raw_parts(data.cast::<u8>().cast_const(), total) };
            prepare_land(
                &back.ticket,
                &back.layout,
                &back.plan,
                mapped,
                diff_on(),
                scratch,
                |buf| {
                    // Hold the mapping read lock only while snapshotting existing title bytes
                    // so uncovered subresource padding is preserved without stalling remaps
                    // across CPU retiling.
                    let _mapping = crate::watch::title_memory_reading();
                    if !back.ticket.valid.load(Ordering::Acquire) {
                        return ReadTitle::Stale;
                    }
                    if crate::host::read_into(at as *const u8, buf) {
                        ReadTitle::Ok
                    } else {
                        ReadTitle::Unreadable
                    }
                },
            )
        };

        match prepared {
            PreparedLand::Stale => note_stale(),
            PreparedLand::Abort => {}
            PreparedLand::Skipped { entry } => {
                store_diff(&back.ticket, entry);
                DIFF_SKIPS.fetch_add(1, Ordering::Relaxed);
                FULL_BYTES_SAVED.fetch_add(size as u64, Ordering::Relaxed);
                note_landed(size);
            }
            PreparedLand::Write { entry } => {
                let dirty_bytes: usize = scratch.dirty_spans.iter().map(|&(s, e)| e - s).sum();
                let committed = commit_if_valid(&back.ticket, |at, _size| {
                    for &(s, e) in &scratch.dirty_spans {
                        // SAFETY: `s <= e <= scratch.tiled.len()`.
                        let src = unsafe { scratch.tiled.as_ptr().add(s) };
                        crate::host::copy_mapped((at + s) as *mut u8, src, e - s, true);
                    }
                });
                if !committed {
                    note_stale();
                    return;
                }
                if let Some(entry) = entry {
                    store_diff(&back.ticket, entry);
                }
                DIRTY_BYTES_WRITTEN.fetch_add(dirty_bytes as u64, Ordering::Relaxed);
                FULL_BYTES_SAVED.fetch_add(size.saturating_sub(dirty_bytes) as u64, Ordering::Relaxed);
                note_landed(size);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32G32_FLOAT, DXGI_SAMPLE_DESC,
    };
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

    fn synthetic_plan(layout: &Layout, format: DXGI_FORMAT) -> FootprintPlan {
        let mut footprints = Vec::new();
        let mut rows = Vec::new();
        let mut offset = 0u64;
        for _slice in 0..layout.slices {
            for m in &layout.mips {
                let row_bytes = m.width * layout.desc.element_bytes;
                let row_pitch = row_bytes.next_multiple_of(256);
                let aligned_offset = offset.next_multiple_of(512);
                footprints.push(D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: aligned_offset,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: m.width,
                        Height: m.height,
                        Depth: m.depth,
                        RowPitch: row_pitch,
                    },
                });
                rows.push(m.height);
                offset = aligned_offset + u64::from(row_pitch) * u64::from(m.height) * u64::from(m.depth);
            }
        }
        FootprintPlan {
            footprints,
            rows,
            mips: layout.desc.mips,
            total: offset,
        }
    }

    fn naive_tile_all(layout: &Layout, plan: &FootprintPlan, mapped: &[u8], initial: &[u8]) -> Vec<u8> {
        let mut out = initial.to_vec();
        for (i, (fp, &rows)) in plan.footprints.iter().zip(&plan.rows).enumerate() {
            let mip = i as u32 % plan.mips;
            let slice = i as u32 / plan.mips;
            let pitch = fp.Footprint.RowPitch as usize;
            let len = pitch * rows as usize * fp.Footprint.Depth as usize;
            let linear = &mapped[fp.Offset as usize..fp.Offset as usize + len];
            layout
                .tile(&mut out, mip, slice, linear, pitch, pitch * rows as usize)
                .unwrap();
        }
        out
    }

    struct LandRunStats {
        skipped: bool,
        retiled: usize,
        span_count: usize,
        dirty_bytes: usize,
    }

    fn run_land_in_memory(
        ticket: &Ticket,
        layout: &Layout,
        plan: &FootprintPlan,
        mapped: &[u8],
        title_mem: &mut [u8],
        diff_enabled: bool,
    ) -> LandRunStats {
        let mut scratch = LandScratch::default();
        let prepared = prepare_land(
            ticket,
            layout,
            plan,
            mapped,
            diff_enabled,
            &mut scratch,
            |buf| {
                buf.copy_from_slice(title_mem);
                ReadTitle::Ok
            },
        );
        match prepared {
            PreparedLand::Skipped { entry } => {
                store_diff(ticket, entry);
                DIFF_SKIPS.fetch_add(1, Ordering::Relaxed);
                LandRunStats {
                    skipped: true,
                    retiled: 0,
                    span_count: 0,
                    dirty_bytes: 0,
                }
            }
            PreparedLand::Write { entry } => {
                let retiled = scratch.changed.iter().filter(|&&c| c).count();
                let mut dirty_bytes = 0usize;
                for &(s, e) in &scratch.dirty_spans {
                    title_mem[s..e].copy_from_slice(&scratch.tiled[s..e]);
                    dirty_bytes += e - s;
                }
                let span_count = scratch.dirty_spans.len();
                if let Some(entry) = entry {
                    store_diff(ticket, entry);
                }
                LandRunStats {
                    skipped: false,
                    retiled,
                    span_count,
                    dirty_bytes,
                }
            }
            PreparedLand::Stale | PreparedLand::Abort => panic!("unexpected stale/abort in test"),
        }
    }

    fn fill_subresource(plan: &FootprintPlan, mapped: &mut [u8], sub: usize, bpe: usize, seed: u8, pad_byte: u8) {
        let fp = &plan.footprints[sub];
        let rows = plan.rows[sub] as usize;
        let pitch = fp.Footprint.RowPitch as usize;
        let row_bytes = fp.Footprint.Width as usize * bpe;
        let base = fp.Offset as usize;
        for y in 0..rows {
            let row = &mut mapped[base + y * pitch..base + (y + 1) * pitch];
            for (x, byte) in row[..row_bytes].iter_mut().enumerate() {
                *byte = seed.wrapping_add((y.wrapping_mul(31) ^ x.wrapping_mul(17)) as u8);
            }
            row[row_bytes..].fill(pad_byte);
        }
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn commit_skips_write_if_invalidated_before_final_lock() {
        let ticket = track(101, 0x8000, 0x9000);
        let mut wrote = false;
        assert!(commit_if_valid(&ticket, |at, size| {
            assert_eq!((at, size), (0x8000, 0x1000));
            wrote = true;
        }));
        assert!(wrote);

        // Simulate a remap/release arriving while tiling ran unlocked.
        remapped(0x8800, 0x100);
        let mut wrote_stale = false;
        assert!(!commit_if_valid(&ticket, |_, _| wrote_stale = true));
        assert!(!wrote_stale);
    }

    #[test]
    fn diff_skip_and_dirty_span_parity_single_and_multi_mip() {
        // 1) Contiguous 256x256 R32_FLOAT (water height map: 1024 B/row == RowPitch).
        let layout_256 = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 256,
            height: 256,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let plan_256 = synthetic_plan(&layout_256, DXGI_FORMAT_R32_FLOAT);
        let size_256 = layout_256.size as usize;
        let base_256 = 0x1000_0000usize;
        let ticket_256 = track(7001, base_256, base_256 + size_256);

        let mut mapped = vec![0u8; plan_256.total as usize];
        fill_subresource(&plan_256, &mut mapped, 0, 4, 0x11, 0x00);

        let initial_title = vec![0x5Au8; size_256];
        let mut title = initial_title.clone();

        // Initial frame writes bit-exact.
        let r1 = run_land_in_memory(&ticket_256, &layout_256, &plan_256, &mapped, &mut title, true);
        let expected_1 = naive_tile_all(&layout_256, &plan_256, &mapped, &initial_title);
        assert!(!r1.skipped);
        assert_eq!(r1.retiled, 1);
        assert!(r1.dirty_bytes > 0);
        assert_eq!(title, expected_1);

        // Identical second frame skips retiling and title writes completely.
        let skips_before = DIFF_SKIPS.load(Ordering::Relaxed);
        let r2 = run_land_in_memory(&ticket_256, &layout_256, &plan_256, &mapped, &mut title, true);
        assert!(r2.skipped);
        assert_eq!(r2.retiled, 0);
        assert_eq!(r2.span_count, 0);
        assert_eq!(r2.dirty_bytes, 0);
        assert!(DIFF_SKIPS.load(Ordering::Relaxed) > skips_before);
        assert_eq!(title, expected_1);

        // Single-bit flip in 1 texel -> detected, bit-exact, minimal dirty span!
        let texel_off = plan_256.footprints[0].Offset as usize + 123 * 1024 + 45 * 4;
        mapped[texel_off] ^= 0x01;
        let r3 = run_land_in_memory(&ticket_256, &layout_256, &plan_256, &mapped, &mut title, true);
        let expected_3 = naive_tile_all(&layout_256, &plan_256, &mapped, &initial_title);
        assert!(!r3.skipped);
        assert_eq!(r3.retiled, 1);
        assert_eq!(r3.span_count, 1);
        assert_eq!(r3.dirty_bytes, 1);
        assert_eq!(title, expected_3);

        // 2) Multi-mip 64x64 R32G32_FLOAT (7 mips: mip 0 is 512 B/row contiguous,
        //    mips 2..6 have RowPitch padding and share the mip tail!).
        let layout_mm = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 7,
            element_bytes: 8,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let plan_mm = synthetic_plan(&layout_mm, DXGI_FORMAT_R32G32_FLOAT);
        let size_mm = layout_mm.size as usize;
        let base_mm = 0x2000_0000usize;
        let ticket_mm = track(7002, base_mm, base_mm + size_mm);

        let mut mapped_mm = vec![0u8; plan_mm.total as usize];
        for mip in 0..7 {
            fill_subresource(&plan_mm, &mut mapped_mm, mip, 8, (0x20 + mip * 7) as u8, 0xAA);
        }
        let initial_mm = vec![0xCCu8; size_mm];
        let mut title_mm = initial_mm.clone();

        let m1 = run_land_in_memory(&ticket_mm, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        let expected_mm1 = naive_tile_all(&layout_mm, &plan_mm, &mapped_mm, &initial_mm);
        assert!(!m1.skipped);
        assert_eq!(m1.retiled, 7);
        assert_eq!(title_mm, expected_mm1);

        // Changing ONLY D3D12 row-pitch padding bytes in padded mips must still diff-skip!
        for mip in 2..7 {
            let fp = &plan_mm.footprints[mip];
            let pitch = fp.Footprint.RowPitch as usize;
            let row_bytes = fp.Footprint.Width as usize * 8;
            assert!(row_bytes < pitch);
            for y in 0..plan_mm.rows[mip] as usize {
                let row_start = fp.Offset as usize + y * pitch;
                mapped_mm[row_start + row_bytes..row_start + pitch].fill(0x55);
            }
        }
        let m_pad = run_land_in_memory(&ticket_mm, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        assert!(m_pad.skipped);
        assert_eq!(m_pad.retiled, 0);
        assert_eq!(m_pad.dirty_bytes, 0);

        // Modify ONLY mip 1 -> mip 0 and mips 2..6 skipped, only 1 subresource retiled, bit-exact!
        let mip1_off = plan_mm.footprints[1].Offset as usize + 5 * plan_mm.footprints[1].Footprint.RowPitch as usize + 3 * 8;
        mapped_mm[mip1_off + 2] ^= 0x80;
        let m2 = run_land_in_memory(&ticket_mm, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        let expected_mm2 = naive_tile_all(&layout_mm, &plan_mm, &mapped_mm, &initial_mm);
        assert!(!m2.skipped);
        assert_eq!(m2.retiled, 1);
        assert_eq!(m2.dirty_bytes, 1);
        assert_eq!(title_mm, expected_mm2);

        // 3) Invalidation via remapped / replaced / released evicts snapshot so
        //    identical data at the same address is retiled & written again!
        // Test remapped:
        title_mm.fill(0x00);
        remapped(base_mm, 4096);
        let ticket_mm_re = track(7002, base_mm, base_mm + size_mm);
        let m_re = run_land_in_memory(&ticket_mm_re, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        assert!(!m_re.skipped);
        assert_eq!(m_re.retiled, 7);
        assert!(m_re.dirty_bytes > 0);
        assert_eq!(title_mm, naive_tile_all(&layout_mm, &plan_mm, &mapped_mm, &vec![0x00; size_mm]));

        // Test replaced:
        title_mm.fill(0x11);
        replaced(base_mm);
        let ticket_mm_rep = track(7002, base_mm, base_mm + size_mm);
        let m_rep = run_land_in_memory(&ticket_mm_rep, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        assert!(!m_rep.skipped);
        assert_eq!(m_rep.retiled, 7);
        assert_eq!(title_mm, naive_tile_all(&layout_mm, &plan_mm, &mapped_mm, &vec![0x11; size_mm]));

        // Test released:
        title_mm.fill(0x22);
        released(7002);
        let ticket_mm_rel = track(7002, base_mm, base_mm + size_mm);
        let m_rel = run_land_in_memory(&ticket_mm_rel, &layout_mm, &plan_mm, &mapped_mm, &mut title_mm, true);
        assert!(!m_rel.skipped);
        assert_eq!(m_rel.retiled, 7);
        assert_eq!(title_mm, naive_tile_all(&layout_mm, &plan_mm, &mapped_mm, &vec![0x22; size_mm]));

        // Cleanup test entries.
        released(7001);
        released(7002);
    }

    #[test]
    fn warp_readback_pool_reuses_buffer_and_lands_bit_exact() {
        let _turn = crate::indirect::tests::warp_turn();
        let Ok(adapter) = crate::host::warp_adapter() else {
            return;
        };

        // Enable D3D12 debug layer when available.
        let mut debug: Option<ID3D12Debug> = None;
        // SAFETY: optional debug interface probe before device creation.
        if unsafe { D3D12GetDebugInterface(&mut debug) }.is_ok()
            && let Some(d) = debug
        {
            // SAFETY: valid debug interface.
            unsafe { d.EnableDebugLayer() };
        }

        let mut device: Option<ID3D12Device> = None;
        // SAFETY: creating a WARP test device.
        if unsafe {
            D3D12CreateDevice(
                &adapter,
                windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0,
                &mut device,
            )
        }
        .is_err()
        {
            return;
        }
        let Some(device) = device else { return };
        let device_key = device.as_raw() as usize;
        let info_queue = device.cast::<ID3D12InfoQueue>().ok();

        let create_readback = |dev: &ID3D12Device, size: u64| -> Option<ID3D12Resource> {
            let mut buf: Option<ID3D12Resource> = None;
            // SAFETY: creating a D3D12 readback buffer on the WARP test device.
            unsafe {
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: size.max(1),
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buf,
                )
                .ok()?;
            }
            buf
        };

        // SAFETY: D3D12 queue/allocator/list/fence/resource setup on WARP.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let allocator: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .unwrap();
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            let mut fence_val = 0u64;

            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 256,
                Height: 256,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &tex_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut texture,
                )
                .unwrap();
            let texture = texture.unwrap();

            let layout = Layout::new(&Desc {
                dimension: Dimension::Tex2d,
                width: 256,
                height: 256,
                depth: 1,
                mips: 1,
                element_bytes: 4,
                block: 1,
                samples: 1,
                swizzle: Swizzle::SW_64KB_S,
            })
            .unwrap();
            let plan = footprint_plan(&device, device_key, &tex_desc).unwrap();

            let mut upload: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_UPLOAD,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: plan.total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut upload,
                )
                .unwrap();
            let upload = upload.unwrap();

            let mut expected_linear = vec![0u8; plan.total as usize];
            fill_subresource(&plan, &mut expected_linear, 0, 4, 0x37, 0x00);
            let mut up_ptr = std::ptr::null_mut();
            upload.Map(0, None, Some(&mut up_ptr)).unwrap();
            std::ptr::copy_nonoverlapping(expected_linear.as_ptr(), up_ptr.cast::<u8>(), expected_linear.len());

            // Upload into texture and transition COPY_DEST -> COPY_SOURCE.
            let upload_to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let upload_from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: plan.footprints[0],
                },
            };
            list.CopyTextureRegion(&upload_to, 0, 0, 0, &upload_from, None);
            let to_src = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&texture),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            };
            list.ResourceBarrier(&[to_src]);

            let mut title_mem = vec![0u8; layout.size as usize];
            let title_at = title_mem.as_mut_ptr() as usize;
            let owner = 0xDEAD_BEEFusize;
            let ticket = track(owner, title_at, title_at + title_mem.len());

            // Run 1: allocate new pooled readback buffer, execute on WARP, land, drop -> returns to pool.
            let back1 = record_on_device(
                &device,
                &list,
                &texture,
                layout.clone(),
                Arc::clone(&ticket),
                true,
                |sz| create_readback(&device, sz),
            )
            .unwrap();
            let raw_ptr_1 = back1.buffer.as_ref().unwrap().as_raw();
            list.Close().unwrap();
            let cmd: ID3D12CommandList = list.cast().unwrap();
            queue.ExecuteCommandLists(&[Some(cmd.clone())]);
            fence_val += 1;
            queue.Signal(&fence, fence_val).unwrap();
            if fence.GetCompletedValue() < fence_val {
                fence.SetEventOnCompletion(fence_val, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            }

            land(&back1);
            let expected_tiled = naive_tile_all(&layout, &plan, &expected_linear, &vec![0u8; layout.size as usize]);
            assert_eq!(title_mem, expected_tiled);
            drop(back1);

            // Run 2: record again -> must reuse the EXACT same ID3D12Resource pointer!
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            let hits_before = POOLED_HITS.load(Ordering::Relaxed);
            let skips_before = DIFF_SKIPS.load(Ordering::Relaxed);
            let back2 = record_on_device(
                &device,
                &list,
                &texture,
                layout.clone(),
                Arc::clone(&ticket),
                true,
                |sz| create_readback(&device, sz),
            )
            .unwrap();
            let raw_ptr_2 = back2.buffer.as_ref().unwrap().as_raw();
            assert_eq!(raw_ptr_1, raw_ptr_2, "pooled readback buffer must be reused");
            assert!(POOLED_HITS.load(Ordering::Relaxed) > hits_before);

            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(cmd.clone())]);
            fence_val += 1;
            queue.Signal(&fence, fence_val).unwrap();
            if fence.GetCompletedValue() < fence_val {
                fence.SetEventOnCompletion(fence_val, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            }
            land(&back2);
            if diff_on() {
                assert!(DIFF_SKIPS.load(Ordering::Relaxed) > skips_before);
            }
            assert_eq!(title_mem, expected_tiled);
            drop(back2);

            // Run 3: record an abandoned Back (never landed) -> Drop must NOT return it to the pool.
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            let abandoned = record_on_device(
                &device,
                &list,
                &texture,
                layout.clone(),
                Arc::clone(&ticket),
                true,
                |sz| create_readback(&device, sz),
            )
            .unwrap();
            assert_eq!(abandoned.buffer.as_ref().unwrap().as_raw(), raw_ptr_1);
            list.Close().unwrap();
            drop(abandoned);

            // Pool must now be empty for `device_key`, so next record misses and allocates fresh.
            let misses_before = POOLED_MISSES.load(Ordering::Relaxed);
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            let back4 = record_on_device(
                &device,
                &list,
                &texture,
                layout.clone(),
                Arc::clone(&ticket),
                true,
                |sz| create_readback(&device, sz),
            )
            .unwrap();
            assert!(POOLED_MISSES.load(Ordering::Relaxed) > misses_before);
            list.Close().unwrap();
            drop(back4);

            upload.Unmap(0, None);
            let _ = windows::Win32::Foundation::CloseHandle(event);

            // Check D3D12 debug layer info queue for zero errors/corruptions.
            if let Some(iq) = info_queue {
                let count = iq.GetNumStoredMessages();
                let mut errors = Vec::new();
                for i in 0..count {
                    let mut len = 0usize;
                    if iq.GetMessage(i, None, &mut len).is_ok() && len > 0 {
                        let mut buf = vec![0u8; len];
                        let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if iq.GetMessage(i, Some(msg), &mut len).is_ok() {
                            let m = &*msg;
                            if matches!(
                                m.Severity,
                                D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
                            ) {
                                let text = std::ffi::CStr::from_ptr(m.pDescription.cast())
                                    .to_string_lossy()
                                    .into_owned();
                                errors.push(text);
                            }
                        }
                    }
                }
                assert!(errors.is_empty(), "D3D12 debug layer errors: {errors:?}");
            }

            // Clean up test state from global caches so WARP device shuts down cleanly.
            released(owner);
            {
                let mut pool = BUFFER_POOL.lock().unwrap_or_else(|e| e.into_inner());
                let mut removed = 0u64;
                pool.entries.retain(|e| {
                    if e.device_key == device_key {
                        removed = removed.saturating_add(e.capacity);
                        false
                    } else {
                        true
                    }
                });
                pool.total_bytes = pool.total_bytes.saturating_sub(removed);
            }
            FOOTPRINT_CACHE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .retain(|(k, _)| k.device_key != device_key);
        }
    }
}
