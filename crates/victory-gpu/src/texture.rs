//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! Dirty-page and per-subresource tracking (`VICTORY_PARTIAL_DETILE`, on by
//! default; `=0` detiles/uploads the whole texture) restricts detiling and
//! `CopyTextureRegion` uploads to only the mips and slices whose tiled bytes
//! changed. When `VICTORY_GPU_DETILE=1` is set, tiled textures skip CPU
//! detiling and run a D3D12 compute shader on the prelude direct command list
//! (`crate::gpu_detile`).
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

pub(crate) mod ahead;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    /// A buffer of its own, or the block [`Staging::piece`] is part of.
    buffer: ID3D12Resource,
    /// Where in `buffer` it begins (the footprints count from there).
    base: u64,
    /// Back in the pool with the last list that names it.
    _piece: Option<crate::pool::StagingPiece>,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice, of the title's texture.
    mips: u32,
    /// The first of them that is read and uploaded
    /// (`crate::resource::held_only`), the one the host copy begins at, and
    /// the bytes of title memory read.
    first: u32,
    level0: u32,
    read: usize,
    /// Which host copy it uploads into, and what that is told to hold.
    serial: u64,
    holds: Arc<crate::resource::Holds>,
    /// Made by the pass over textures picked by index, for one upload
    /// ([`once`]): read when the list is submitted, and never looked at again.
    once: std::sync::atomic::AtomicBool,
    /// The staging buffer's bytes (counted in `STAGED` while it lives).
    /// With `gpu_detile`, this is the tiled upload buffer size (`read` rounded up
    /// to 4 bytes); otherwise it equals `linear_bytes`.
    bytes: u64,
    /// Total bytes of the linear subresource footprints (`GetCopyableFootprints`).
    linear_bytes: u64,
    /// Whether this staging buffer holds raw tiled bytes to be detiled on the
    /// GPU during `PreludeUpload::record` (`VICTORY_GPU_DETILE=1`).
    gpu_detile: bool,
    /// Whether `record_gpu_detiled` has already emitted the first-upload
    /// aliasing barrier for this pooled texture.
    gpu_detile_aliased: std::sync::atomic::AtomicBool,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a list's prelude last
    /// uploaded it, and whether one has uploaded it at all: a list's prelude
    /// records `CopyTextureRegion` at submission time (after `refresh()`)
    /// only when `!uploaded || dirty`.
    dirty: std::sync::atomic::AtomicBool,
    /// Bitmask of subresources (`0..footprints.len().min(64)`) modified since
    /// the last prelude upload (`u64::MAX` means every subresource).
    dirty_subs: std::sync::atomic::AtomicU64,
    uploaded: std::sync::atomic::AtomicBool,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// How many bytes of the texture's memory were mapped when a list last
    /// recorded an upload of it ([`fill_unlocked`]; 0: not asked).
    expected: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
    /// Recorded for a one-shot decoration bake (`BAKING`): submission-time
    /// refresh must wait for title memory to stop remapping
    /// (`title_memory_reading`) instead of taking `no_wait()` /
    /// `refresh_unlocked()` or the 30-frame partial-read throttle.
    exact: std::sync::atomic::AtomicBool,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Digest>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// Until this frame a guarded texture is hashed at every check: its
    /// first checks, and after the hash found a change the guard had not
    /// seen ([`hot_textures`]). 0: not checked under the guard yet.
    hot_until: u64,
    /// The frame it was read in ahead of its submission
    /// ([`Staging::refresh_ahead`]).
    ahead: Option<u64>,
    /// Per-subresource content digests for partial detile/upload on hash-checked
    /// textures (`VICTORY_PARTIAL_DETILE`).
    sub_hashes: Vec<Option<Digest>>,
    /// Whether the full tiled upload buffer has been populated at least once
    /// (`gpu_detile` mode).
    gpu_seeded: bool,
}

/// Frames a guarded texture is hashed at every check from its first check,
/// and after the hash found a change the guard did not see.
const HOT_FRAMES: u64 = 64;

/// Whether a guarded texture whose memory the hash found changed although
/// the guard saw no write is hashed at every check for a while
/// ([`HOT_FRAMES`]), not every eighth frame. A title that writes such a
/// texture through another view of its pages (the guard sees none of it)
/// every frame, a video's frames in a ring of three, showed each one up
/// to eight frames late and in turns: the boot logo video and the loading
/// animation flickered between old and new pictures (fx-flicker run 1441:
/// the presented picture cycled through three, a new one every eight
/// frames). A texture also starts hot, for the ones written that way from
/// the start (the six-star screen of the first seconds, which the hash
/// never caught changing). `VICTORY_TEXTURE_HOT=0`: every eighth frame
/// only, as before.
fn hot_textures() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_HOT").is_none_or(|v| v != "0"))
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: RwLock<Option<HashMap<usize, Arc<Staging>>>> = RwLock::new(None);
/// Exact entry count of [`STAGING`], maintained under `STAGING.write()` so empty
/// checks and cold misses avoid locking.
static STAGING_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Generation bumped (with `Release` under `STAGING.write()`) on every insert,
/// removal, or idle sweep in [`STAGING`], invalidating [`STAGING_TLS`].
static STAGING_EPOCH: AtomicU64 = AtomicU64::new(1);

const STAGING_TLS_CAP: usize = 512;

thread_local! {
    /// Per-thread direct-mapped `Weak<Staging>` lookup cache (`(epoch, resource, weak)`).
    /// Uses `Weak` so [`let_go_idle`]'s `Arc::strong_count(s) > 1` check remains exact.
    static STAGING_TLS: std::cell::RefCell<Vec<(u64, usize, Weak<Staging>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn staging_tls_slot(key: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15usize) >> (usize::BITS - 9)) & (STAGING_TLS_CAP - 1)
}

fn staging_tls_get(key: usize) -> Option<Arc<Staging>> {
    let epoch = STAGING_EPOCH.load(Ordering::Acquire);
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    STAGING_TLS.with(|cell| {
        let slots = cell.borrow();
        let &(at, k, ref weak) = slots.get(staging_tls_slot(key))?;
        (at == epoch && k == key).then(|| weak.upgrade()).flatten()
    })
}

fn staging_tls_put(epoch: u64, key: usize, s: &Arc<Staging>) {
    if STAGING_EPOCH.load(Ordering::Acquire) != epoch {
        return;
    }
    STAGING_TLS.with(|cell| {
        let mut slots = cell.borrow_mut();
        if slots.is_empty() {
            slots.resize_with(STAGING_TLS_CAP, || (0, 0, Weak::new()));
        }
        slots[staging_tls_slot(key)] = (epoch, key, Arc::downgrade(s));
    });
}

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
pub(crate) struct Through {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
    /// Last `source.write_gen` synced into the placed texture in submission
    /// order (starts at 0 while `write_gen` starts at 1, so the first read
    /// always syncs).
    pub(crate) synced_gen: std::sync::atomic::AtomicU64,
}

/// Whether format-alias copies track source texture write generations across
/// command lists and defer cross-list syncs to list preludes
/// (`VICTORY_ALIAS_DIRTY=0` reverts to legacy per-list copying).
fn alias_dirty_tracking() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_DIRTY").is_none_or(|v| v != "0"))
}

/// Records the `from -> through.buffer -> to` subresource copies into `list`.
/// Both `from` (`COPY_SOURCE`) and `to` (`COPY_DEST`) must already be in their
/// copy states; `through.buffer` begins and ends in `COMMON`.
pub(crate) fn record_alias_copy(
    list: &ID3D12GraphicsCommandList,
    from: &ID3D12Resource,
    to: &ID3D12Resource,
    through: &Through,
) {
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: RwLock<Option<HashMap<usize, Arc<Through>>>> = RwLock::new(None);
static THROUGH_COUNT: AtomicUsize = AtomicUsize::new(0);
static THROUGH_EPOCH: AtomicU64 = AtomicU64::new(1);

const THROUGH_TLS_CAP: usize = 64;

thread_local! {
    static THROUGH_TLS: std::cell::RefCell<Vec<(u64, usize, Weak<Through>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn through_tls_slot(key: usize) -> usize {
    (key.wrapping_mul(0x9E37_79B9_7F4A_7C15usize) >> (usize::BITS - 6)) & (THROUGH_TLS_CAP - 1)
}

fn through_tls_get(key: usize) -> Option<Arc<Through>> {
    let epoch = THROUGH_EPOCH.load(Ordering::Acquire);
    if THROUGH_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    THROUGH_TLS.with(|cell| {
        let slots = cell.borrow();
        let &(at, k, ref weak) = slots.get(through_tls_slot(key))?;
        (at == epoch && k == key).then(|| weak.upgrade()).flatten()
    })
}

fn through_tls_put(epoch: u64, key: usize, t: &Arc<Through>) {
    if THROUGH_EPOCH.load(Ordering::Acquire) != epoch {
        return;
    }
    THROUGH_TLS.with(|cell| {
        let mut slots = cell.borrow_mut();
        if slots.is_empty() {
            slots.resize_with(THROUGH_TLS_CAP, || (0, 0, Weak::new()));
        }
        slots[through_tls_slot(key)] = (epoch, key, Arc::downgrade(t));
    });
}

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    // (All of its levels: the copy is level by level.)
    crate::resource::want_level(placed, 0);
    let (Some(to_copy), Some(from_copy)) = (
        crate::resource::host_copy(placed),
        crate::resource::host_copy(source),
    ) else {
        return false;
    };
    let (to, from) = (&to_copy.texture, &from_copy.texture);
    let Some(host) = host() else { return false };
    let through = if let Some(through) = through_tls_get(placed as usize) {
        through
    } else if let Some(through) = {
        if THROUGH_COUNT.load(Ordering::Acquire) > 0 {
            let map = THROUGH.read().unwrap_or_else(|e| e.into_inner());
            let epoch = THROUGH_EPOCH.load(Ordering::Acquire);
            map.as_ref()
                .and_then(|m| m.get(&(placed as usize)).cloned())
                .map(|t| (epoch, t))
        } else {
            None
        }
    } {
        through_tls_put(through.0, placed as usize, &through.1);
        through.1
    } else {
        let mut guard = THROUGH.write().unwrap_or_else(|e| e.into_inner());
        let map = guard.get_or_insert_with(HashMap::new);
        let (epoch, through) = match map.get(&(placed as usize)) {
            Some(through) => (THROUGH_EPOCH.load(Ordering::Acquire), through.clone()),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                    synced_gen: std::sync::atomic::AtomicU64::new(0),
                });
                map.insert(placed as usize, through.clone());
                THROUGH_COUNT.store(map.len(), Ordering::Release);
                let epoch = THROUGH_EPOCH.fetch_add(1, Ordering::Release) + 1;
                (epoch, through)
            }
        };
        drop(guard);
        through_tls_put(epoch, placed as usize, &through);
        through
    };
    if alias_dirty_tracking() && crate::resource::host_tiled(placed).is_none() {
        if !list.alias_use(&from_copy, &to_copy, &through) {
            return true;
        }
        let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
            let state = crate::host::legal_state(texture, COPY_SOURCE);
            let barrier = crate::host::transition(
                texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(state as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
            before
        });
        crate::list::ensure_state(list, placed, COPY_DEST);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
        record_alias_copy(&list.list, from, to, &through);
        list.keep(&through.buffer);
        // A draw that also writes the source (a view of it handled before
        // this one) needs it in that state again; restoring the write state
        // bumps `local_writes` so the next alias read copies once more.
        if let Some(before) = before
            && before & TARGET_OR_UNORDERED != 0
        {
            crate::list::ensure_state(list, source, before);
        }
        return true;
    }
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    record_alias_copy(&list.list, from, to, &through);
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. A staging buffer that holds nothing
/// yet is filled with what is there all the same ([`fill_unlocked`]): a list
/// has recorded the upload from it by then.
fn no_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_NO_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a refresh that does not wait ([`no_wait`]) and finds a page
/// missing fills a staging buffer that was never filled from the pages that
/// are there (`VICTORY_TEXTURE_FILL=0`: it was left as it was made).
///
/// Left for the next refresh, it was uploaded as it was: the upload is
/// recorded before the refresh, and a texture picked by index has no next
/// refresh (its staging buffer goes with the upload). A texture the title
/// holds only the smaller levels of is never all there, so whenever the
/// runtime was remapping at that moment, its levels became zeros on the
/// card, good ones too: of 16,135 textures a recorded frame's slots show,
/// 2,382 had zeros in the first level shown; 137 of that frame's 787 repeated
/// uploads put zeros over a filled level (run pic1-1; 68,700 refreshes left
/// in the 150 s of world of run play1).
fn fill_unlocked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FILL").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there; staging buffers
/// filled from the pages there ([`fill_unlocked`]), and of them those that
/// got less than was mapped (their slots are looked at again).
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
#[inline]
fn waited<T>(_slot: usize, f: impl FnOnce() -> T) -> T {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let out = f();
    #[cfg(feature = "diagnostics")]
    REFRESH_WAITS[_slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds) and their
/// bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) fn live_staging_bytes() -> u64 {
    STAGED[1].load(std::sync::atomic::Ordering::Relaxed)
}

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
    }
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    let kept = STAGING
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| map.len() as u64);
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    if STAGING_COUNT.load(Ordering::Acquire) > 0 {
        let mut guard = STAGING.write().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = guard.as_mut()
            && !map.is_empty()
        {
            let idle_ms = crate::host::ram_idle_ms(crate::residency::headroom());
            // A list that recorded an upload or a check holds the staging until it
            // is reset: one only the table holds is named by no list, recorded or
            // running.
            let (mut went, mut bytes) = (0u64, 0u64);
            map.retain(|_, s| {
                let keep = Arc::strong_count(s) > 1
                    || !crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES);
                if !keep {
                    LET_GO[0].fetch_add(1, Relaxed);
                    LET_GO[1].fetch_add(s.bytes, Relaxed);
                    went += 1;
                    bytes += s.bytes;
                }
                keep
            });
            if went > 0 {
                STAGING_COUNT.store(map.len(), Ordering::Release);
                STAGING_EPOCH.fetch_add(1, Ordering::Release);
                crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
            }
        }
    }
    crate::tiles::sweep_demotions();
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
/// Exact cardinality of [`HOST_WRITTEN`], updated under its lock before bumping
/// [`HOST_WRITTEN_EPOCH`].
static HOST_WRITTEN_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Monotonic generation bumped (with `Release` under `HOST_WRITTEN.lock()`) on
/// every insert or remove, invalidating per-thread [`HOST_WRITTEN_TLS`] snapshots.
static HOST_WRITTEN_EPOCH: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// Per-thread sorted snapshot of [`HOST_WRITTEN`] tagged with [`HOST_WRITTEN_EPOCH`],
    /// making [`host_wrote`] and [`upload_from`]'s host-written check lock-free on the hot path.
    static HOST_WRITTEN_TLS: std::cell::RefCell<(u64, Vec<usize>)> =
        const { std::cell::RefCell::new((0, Vec::new())) };
}

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    if host_wrote(resource as usize) {
        return;
    }
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        let set = set.get_or_insert_with(Default::default);
        let inserted = set.insert(resource as usize);
        if inserted {
            HOST_WRITTEN_COUNT.store(set.len(), Ordering::Release);
            HOST_WRITTEN_EPOCH.fetch_add(1, Ordering::Release);
        }
        inserted
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    #[cfg(feature = "diagnostics")]
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            let mut map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
            map.insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
            HOST_WRITTEN_AT_COUNT.store(map.len(), Ordering::Release);
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
#[cfg(feature = "diagnostics")]
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());
#[cfg(feature = "diagnostics")]
static HOST_WRITTEN_AT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
#[cfg(feature = "diagnostics")]
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    if HOST_WRITTEN_AT_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

#[cfg(not(feature = "diagnostics"))]
#[inline]
pub(crate) fn placed_in_host_written(_address: u64, _desc: &[u32; 14]) {}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    let epoch = HOST_WRITTEN_EPOCH.load(Ordering::Acquire);
    if HOST_WRITTEN_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    HOST_WRITTEN_TLS.with(|cell| {
        let mut cache = cell.borrow_mut();
        if cache.0 != epoch {
            let set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            let cur_epoch = HOST_WRITTEN_EPOCH.load(Ordering::Acquire);
            cache.1.clear();
            if let Some(set) = set.as_ref() {
                cache.1.extend(set.iter().copied());
                cache.1.sort_unstable();
            }
            cache.0 = cur_epoch;
        }
        cache.1.binary_search(&resource).is_ok()
    })
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    if STAGING_COUNT.load(Ordering::Acquire) > 0 {
        let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map.remove(&resource).is_some()
        {
            STAGING_COUNT.store(map.len(), Ordering::Release);
            STAGING_EPOCH.fetch_add(1, Ordering::Release);
        }
    }
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let removed_host_written = if host_wrote(resource) {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(set) = set.as_mut()
            && set.remove(&resource)
        {
            HOST_WRITTEN_COUNT.store(set.len(), Ordering::Release);
            HOST_WRITTEN_EPOCH.fetch_add(1, Ordering::Release);
            true
        } else {
            false
        }
    } else {
        false
    };
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    #[cfg(feature = "diagnostics")]
    if removed_host_written && HOST_WRITTEN_AT_COUNT.load(Ordering::Acquire) > 0 {
        let mut map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, d| d.5 != resource);
        HOST_WRITTEN_AT_COUNT.store(map.len(), Ordering::Release);
    }
    if THROUGH_COUNT.load(Ordering::Acquire) > 0 {
        let mut through = THROUGH.write().unwrap_or_else(|e| e.into_inner());
        if let Some(through) = through.as_mut()
            && through.remove(&resource).is_some()
        {
            THROUGH_COUNT.store(through.len(), Ordering::Release);
            THROUGH_EPOCH.fetch_add(1, Ordering::Release);
        }
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Registers, once per list, `resource`'s staging buffer so that at submission
/// time its memory is refreshed and, if changed (or never uploaded), copied
/// into its host texture on the list's prelude command list.
pub(crate) fn upload(list: &HostList, resource: This, _texture: &ID3D12Resource) {
    upload_from(list, resource, 0);
}

thread_local! {
    /// The thread records uploads for a draw that bakes from them.
    pub(crate) static BAKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The thread records the upload of a texture picked by index, of which
    /// that many bytes were found mapped just now (`u64::MAX`: not asked).
    pub(crate) static PICKING: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    /// The thread is staging a texture for a copy-only queue (`texture::ahead`),
    /// which cannot dispatch GPU compute detile shaders.
    pub(crate) static COPY_QUEUE_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Active while `heap_worker::Batch::record` collects `Staging` handles
    /// deferred into its upload list for parallel burst detiling.
    static AHEAD_COLLECT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static AHEAD_STAGED: std::cell::RefCell<Vec<Arc<Staging>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs `f` while capturing any [`Staging`] handles passed to
/// `HostList::defer_texture` on this thread into `out`.
pub(crate) fn capture_deferred_staging<R>(out: &mut Vec<Arc<Staging>>, f: impl FnOnce() -> R) -> R {
    struct Guard<'a>(&'a mut Vec<Arc<Staging>>, bool);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            AHEAD_COLLECT.set(self.1);
            AHEAD_STAGED.with(|staged| self.0.extend(staged.borrow_mut().drain(..)));
        }
    }
    let prev = AHEAD_COLLECT.replace(true);
    let _guard = Guard(out, prev);
    f()
}

/// Whether a staging buffer made for one upload of a texture picked by index
/// is filled without asking the write watch and the guards
/// (`VICTORY_TEXTURE_ONCE=0`: with, as before). What they say is for the next
/// refresh, and there is none: the buffer goes with the upload. Asking guards
/// the texture's blocks (a change of protection, under the lock every other
/// refresh wants), and the title's next write there faults to open them.
fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ONCE").is_none_or(|v| v != "0"))
}

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. (Not a
/// cryptographic one, which took 0.27 ms a megabyte, four times the detiling
/// it was to spare; nobody makes up textures to collide.)
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane).wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41)) ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block.try_into().unwrap_or(&[0; 64]));
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len) ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// [`upload`] of the levels from `first` on, the ones a view shows: of a
/// texture the title placed there may be no memory of the larger ones, and
/// what lies there is another texture's. (After victory-amd 4436f8f1.)
pub(crate) fn upload_from(list: &HostList, resource: This, first: u32) {
    let first = if crate::resource::held_only() { first } else { 0 };
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_wrote(resource as usize) {
        return;
    }
    // Once a list for each first level; not again for one that the list's
    // upload of larger ones held too.
    if !list.first_use(1 << 63 | (resource as usize) << 4 | first as usize & 15) {
        return;
    }
    if !list.first_use(resource as usize) && staged_from(resource) <= first {
        return;
    }
    // Under `VICTORY_HEAP_WORKER=1`, `host.mirror()` runs at batch publish
    // rather than before upload recording; request the needed mip level here
    // so a clamped or lower-mip host copy grows before staging.
    if crate::resource::held_only() {
        crate::resource::want_level(resource, first);
    }
    let Some(copy) = crate::resource::host_copy(resource) else {
        return;
    };
    let texture = &copy.texture;
    ahead::listed(&copy);
    let Some(staging) = staging(resource, &copy, first) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    let baking = BAKING.get();
    if baking && !staging.exact.load(Relaxed) {
        staging.exact.store(true, Relaxed);
    }
    let picking = PICKING.get();
    if picking.is_none() && staging.once.load(Relaxed) {
        // A draw samples it through a table: kept and checked from here on.
        staging.once.store(false, Relaxed);
    }
    let now = crate::host::moment();
    if staging.used.load(Relaxed) != now {
        staging.used.store(now, Relaxed);
    }
    // Levels on demand (`crate::tiles`): touch recency for tier-2 mip-tail
    // demotion and immediately re-promote if this texture was demoted while idle
    // so active close-up Series X textures always render at full resolution.
    if staging.on_demand
        && let Some(tiled) = copy.tiled.as_ref()
    {
        tiled.touch(now);
        crate::tiles::register_candidate(copy.serial, &copy, staging.memory as u64, resource as usize);
        if tiled.promote() {
            staging.dirty.store(true, Relaxed);
        }
    }
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes.
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The video decoder cycles CPU-written linear plane buffers. Keep a
    // dedicated per-list upload buffer through the list's GPU fence so
    // concurrent frames never overwrite each other or stall on staging hashes.
    if staging.movie_plane() {
        let (at, len) = staging.memory();
        let footprint = staging.footprints[0].Footprint;
        let pitch = footprint.RowPitch as usize;
        let bytes = pitch * staging.rows[0] as usize;
        let plane = staging.clone();
        let Some(source) = list.plane_upload(bytes as u64, (at, len), move |dst| {
            fill_plane(&plane.layout, at, len, dst, bytes, pitch);
        }) else {
            once!("no upload buffer for a CPU-written movie plane");
            return;
        };
        record_upload(list, resource, texture, &staging, &source, 0);
        staging.uploaded.store(true, Relaxed);
        return;
    }
    if no_wait() && fill_unlocked() {
        let there = match picking {
            Some(there) if there != u64::MAX => there,
            _ => mapped_bytes(staging.memory, staging.read),
        };
        staging.expected.store(there, Relaxed);
    }
    // At submission time, `staging.refresh()` runs first (in parallel via
    // `Batch::refresh_front` / `check_textures` / `refresh_textures`), and only
    // then does `PreludeUpload::finish_refresh` decide whether the list's
    // prelude must record `CopyTextureRegion`. This guarantees zero frames of
    // latency when title memory changes (including 3-buffer video rings) and
    // avoids stale-frame races with joined submissions / `VICTORY_THIN`.
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    if AHEAD_COLLECT.get() {
        AHEAD_STAGED.with(|staged| staged.borrow_mut().push(staging.clone()));
    }
    list.defer_texture(PreludeUpload {
        staging,
        copy,
        on_demand,
        force: baking,
    });
}

/// Whether partial dirty-page / per-subresource texture detiling and upload is
/// enabled (`VICTORY_PARTIAL_DETILE=0` disables it and falls back to full
/// texture detile/upload). Enabled by default.
fn partial_detile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PARTIAL_DETILE").is_none_or(|v| v != "0"))
}

/// Bitmask covering every subresource (`0..count`), or `u64::MAX` when `count >= 64`.
fn all_subs_mask(count: usize) -> u64 {
    if count == 0 {
        0
    } else if count >= 64 {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}

/// Returns whether subresource `i` is selected by `sub_mask` (always true when
/// the texture has more than 64 subresources).
#[inline]
fn sub_selected(sub_mask: u64, i: usize, sub_count: usize) -> bool {
    sub_count > 64 || (i < 64 && (sub_mask & (1u64 << i)) != 0)
}

/// Maps dirty 64 KiB write-watch/guard pages (`w > since`) inside `[start, start + read)`
/// to the bitmask of subresources (`0..sub_count`) whose tiled byte spans overlap them.
#[allow(clippy::too_many_arguments)]
fn dirty_subs_from_pages(
    layout: &victory_formats::texture::Layout,
    first: u32,
    mips: u32,
    sub_count: usize,
    start: usize,
    read: usize,
    written: &[(usize, u64)],
    since: u64,
) -> u64 {
    let all = all_subs_mask(sub_count);
    if !partial_detile() || sub_count <= 1 || sub_count > 64 || read == 0 {
        return all;
    }
    let page_bytes = crate::watch::PAGE;
    // Merge adjacent dirty 64 KiB pages into minimal relative byte ranges.
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for &(page, epoch) in written {
        if epoch <= since {
            continue;
        }
        let page_start = page.saturating_mul(page_bytes);
        let page_end = page_start.saturating_add(page_bytes);
        if page_end <= start || page_start >= start.saturating_add(read) {
            continue;
        }
        let lo = page_start.saturating_sub(start) as u64;
        let hi = page_end.saturating_sub(start).min(read) as u64;
        if lo >= hi {
            continue;
        }
        if let Some(last) = ranges.last_mut()
            && lo <= last.1
        {
            last.1 = last.1.max(hi);
        } else {
            ranges.push((lo, hi));
        }
    }
    if ranges.is_empty() {
        return 0;
    }
    let per_slice = mips - first;
    let mut mask = 0u64;
    for i in 0..sub_count {
        let mip = first + (i as u32) % per_slice;
        let slice = (i as u32) / per_slice;
        if ranges
            .iter()
            .any(|&(lo, hi)| layout.subresource_overlaps(mip, slice, lo, hi))
        {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// Upload instructions produced by [`PreludeUpload::finish_refresh`] for
/// `HostList::prelude`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UploadPlan {
    pub(crate) first_upload: bool,
    pub(crate) sub_mask: u64,
}

/// Pure claim helper for [`PreludeUpload::finish_refresh`]: returns
/// `Some(first_upload)` when the host texture has never been uploaded or the
/// staging buffer was detiled since the last upload, and `None` when the host
/// texture already holds the staging buffer's current contents.
fn claim_upload(
    uploaded: &std::sync::atomic::AtomicBool,
    dirty: &std::sync::atomic::AtomicBool,
) -> Option<bool> {
    use std::sync::atomic::Ordering::AcqRel;
    let was_uploaded = uploaded.swap(true, AcqRel);
    let was_dirty = dirty.swap(false, AcqRel);
    (!was_uploaded || was_dirty).then_some(!was_uploaded)
}

/// A title texture deferred to submission time so `staging.refresh()` runs
/// before deciding whether `HostList::prelude` must record `CopyTextureRegion`.
#[derive(Clone)]
pub(crate) struct PreludeUpload {
    pub(crate) staging: Arc<Staging>,
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    on_demand: Option<crate::resource::OnDemand>,
    /// One-shot decoration bake (`BAKING`): always upload on the prelude so
    /// exact submission-time bytes land in VRAM before the bake draw executes.
    force: bool,
}

impl PreludeUpload {
    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        self.staging.memory()
    }

    /// Called on the submitting thread after `self.staging.refresh()` has
    /// completed for this submission. Returns `Some(UploadPlan)` when the
    /// list's prelude must record `CopyTextureRegion` into `self.copy.texture`.
    pub(crate) fn finish_refresh(&self) -> Option<UploadPlan> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
        // Hold `last` while claiming `dirty` / `dirty_subs` and reading `detiled`
        // so a concurrent `refresh_ahead()` cannot overlap writing `self.data`
        // with our prelude `CopyTextureRegion` claim.
        let (claimed, pending_subs, detiled) = {
            let _last = self.staging.last.lock().unwrap_or_else(|e| e.into_inner());
            let claimed = claim_upload(&self.staging.uploaded, &self.staging.dirty);
            let pending_subs = self.staging.dirty_subs.swap(0, AcqRel);
            let detiled = self.staging.detiled.load(Acquire);
            (claimed, pending_subs, detiled)
        };
        let Some(first_upload) = claimed.or(self.force.then_some(false)) else {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexSkipped, 1);
            return None;
        };
        let all = all_subs_mask(self.staging.footprints.len());
        let sub_mask = if first_upload
            || self.force
            || self.on_demand.is_some()
            || !partial_detile()
            || pending_subs == 0
        {
            all
        } else {
            pending_subs & all
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexUploads, 1);
        // A host copy that took the place of one with fewer levels: the slots
        // that show that one are made again once this upload has run.
        if self.staging.first == self.staging.level0 && self.copy.grown.swap(false, Relaxed) {
            crate::descriptor::host::levels_changed_after(self.staging.memory as u64);
        }
        // Levels on demand (`crate::tiles`): after `refresh()` has detiled,
        // give the held levels their memory on the queue before the prelude
        // copies into them; otherwise record what the host copy now holds.
        if let Some(on_demand) = &self.on_demand {
            let waiting = on_demand.uploaded(detiled, self.staging.memory as u64);
            self.staging.waiting.store(
                if waiting { crate::host::moment().max(1) } else { 0 },
                Relaxed,
            );
            // A texture picked by index has no staging buffer after this
            // upload: its slots are made due again by the texture pass.
            if waiting {
                crate::descriptor::host::levels_waiting(self.staging.memory as u64);
            }
        } else {
            self.staging.note_held();
        }
        Some(UploadPlan {
            first_upload,
            sub_mask,
        })
    }

    /// Records `CopyTextureRegion` (and GPU compute detiling when
    /// `VICTORY_GPU_DETILE=1`) onto `cmd_list` (the prelude command list) for
    /// the subresources selected in `sub_mask`. `self.copy.texture` must
    /// already be in `D3D12_RESOURCE_STATE_COPY_DEST`.
    pub(crate) fn record(&self, cmd_list: &ID3D12GraphicsCommandList, sub_mask: u64) {
        let texture = &self.copy.texture;
        let sub_count = self.staging.footprints.len();
        if self.staging.gpu_detile {
            self.record_gpu_detiled(cmd_list, texture, sub_mask, sub_count);
            return;
        }
        // The staging buffer lives as long as `self.staging`, which the list's
        // deferred steps hold until the list is reset after GPU completion.
        for (i, footprint) in self.staging.footprints.iter().enumerate() {
            if !sub_selected(sub_mask, i, sub_count) {
                continue;
            }
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: self.staging.first - self.staging.level0 + i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(&self.staging.buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: self.staging.base + footprint.Offset,
                        ..*footprint
                    },
                },
            };
            // SAFETY: an open prelude list; both locations name live resources.
            unsafe { cmd_list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }

    /// Dispatches the native compute detiler from `self.staging.buffer` (raw
    /// tiled bytes) into a ring-buffered device-local UAV scratch slice, then
    /// copies the selected subresources VRAM-to-VRAM into `texture`.
    fn record_gpu_detiled(
        &self,
        cmd_list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
        sub_mask: u64,
        sub_count: usize,
    ) {
        let Some(host) = host() else {
            return;
        };
        let held = if self.staging.on_demand {
            self.staging
                .detiled
                .load(std::sync::atomic::Ordering::Acquire)
        } else {
            u32::MAX
        };
        let mut jobs = Vec::with_capacity(sub_count);
        let mut indices = Vec::with_capacity(sub_count);
        for (i, (footprint, &rows)) in self
            .staging
            .footprints
            .iter()
            .zip(&self.staging.rows)
            .enumerate()
        {
            if !sub_selected(sub_mask, i, sub_count) {
                continue;
            }
            let (mip, slice) = self.staging.part(i);
            let row_pitch = footprint.Footprint.RowPitch;
            let depth_pitch = row_pitch.saturating_mul(rows);
            jobs.push(crate::gpu_detile::SubresourceJob {
                mip,
                slice,
                dst_offset: footprint.Offset,
                row_pitch,
                depth_pitch,
                zero: mip < 32 && held & (1 << mip) == 0,
            });
            indices.push(i);
        }
        if jobs.is_empty() {
            return;
        }
        // SAFETY: live staging upload buffer.
        let src_va = unsafe { self.staging.buffer.GetGPUVirtualAddress() } + self.staging.base;
        let need_alias = crate::pool::on()
            && !self
                .staging
                .gpu_detile_aliased
                .swap(true, std::sync::atomic::Ordering::AcqRel);
        let ok = crate::gpu_detile::record_prelude_detile(
            host,
            cmd_list,
            &self.staging.layout,
            src_va,
            self.staging.read as u64,
            texture,
            self.staging.first - self.staging.level0,
            &self.staging.footprints,
            &jobs,
            &indices,
            self.staging.linear_bytes,
            need_alias,
        );
        if !ok {
            once!("gpu detile dispatch validation failed; skipping texture upload");
        }
    }
}

fn record_upload(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    source: &ID3D12Resource,
    base: u64,
) {
    use std::sync::atomic::Ordering::Relaxed;
    const COPY_DEST: u32 = 0x400;
    // A texture cut from a block may lie where another did: what the card
    // holds of that one is done with before the first upload into this one.
    if crate::pool::on() && !staging.uploaded.load(Relaxed) {
        let aliasing = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    // SAFETY: a borrowed pointer the barrier never releases.
                    pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
                }),
            },
        };
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[aliasing]) };
    }
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging/plane upload buffer lives until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (source.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        base + footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + base + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: staging.first - staging.level0 + i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(source) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: base + footprint.Offset,
                    ..*footprint
                },
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

fn fill_plane(
    layout: &victory_formats::texture::Layout,
    at: usize,
    len: usize,
    dst: usize,
    bytes: usize,
    pitch: usize,
) {
    let _pin = crate::watch::title_memory_reading();
    // SAFETY: the list owns this full upload allocation until its GPU work completes.
    let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes) };
    let mip = &layout.mips[0];
    if mip.offset == 0
        && mip.pitch as usize * layout.desc.element_bytes as usize == pitch
        && bytes <= len
        && crate::host::read_into(at as *const _, out)
    {
        return;
    }
    with_read(len, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), at as *const u8, len, false);
        if let Err(error) = layout.detile(read, 0, 0, out, pitch, bytes) {
            out.fill(0);
            once!("could not copy the linear plane into the list's upload buffer: {error:?}");
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Records copies of zeros into every subresource of `texture`, which is in
/// the copy destination state.
pub(crate) fn zeros(list: &HostList, host: &crate::host::Host, texture: &ID3D12Resource) {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { 1 } else { u32::from(d.DepthOrArraySize) };
    let count = u32::from(d.MipLevels) * layers;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut bytes = 0u64;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device
            .GetCopyableFootprints(&d, 0, count, 0, Some(footprints.as_mut_ptr()), None, None, Some(&mut bytes));
    }
    if bytes == 0 || bytes == u64::MAX {
        return;
    }
    let Some((buffer, data)) = list.upload(host, bytes) else {
        return;
    };
    // SAFETY: a mapped buffer of `bytes` bytes, just made.
    unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; the list holds the buffer until it is reset.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *footprint },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

impl Staging {
    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
        self.dirty_subs.store(
            all_subs_mask(self.footprints.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.read)
    }

    /// The title's level and the array slice of the staging buffer's part
    /// `i`.
    fn part(&self, i: usize) -> (u32, u32) {
        let mips = self.mips - self.first;
        (self.first + i as u32 % mips, i as u32 / mips)
    }

    /// The host copy holds what the staging buffer was filled from
    /// ([`unchanged`]).
    fn note_held(&self) {
        let hash = self.last.lock().unwrap_or_else(|e| e.into_inner()).hash;
        *self.holds.lock().unwrap_or_else(|e| e.into_inner()) = hash.map(|hash| (self.first, hash));
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _texture = crate::timeline::span("a texture's refresh");
        let read = self.refresh_now();
        #[cfg(feature = "diagnostics")]
        {
            use std::sync::atomic::Ordering::Relaxed;
            let preparing = &crate::host::PREPARING;
            preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
            preparing[3].fetch_add(read as u64, Relaxed);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = read;
    }

    fn movie_plane(&self) -> bool {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UNORM;
        let desc = &self.layout.desc;
        !self.on_demand
            && desc.dimension == victory_formats::texture::Dimension::Tex2d
            && desc.swizzle.0 == 0
            && desc.depth == 1
            && desc.mips == 1
            && desc.samples == 1
            && self.footprints.len() == 1
            && self.footprints[0].Offset == 0
            && self.footprints[0].Footprint.Format == DXGI_FORMAT_R8_UNORM
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        #[cfg(feature = "diagnostics")]
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        #[cfg(feature = "diagnostics")]
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    /// Copies only the tiled byte spans of dirty subresources into `read` at
    /// their exact offsets when `sub_mask` is a strict subset and detiling will
    /// not read outside those spans; falls back to copying all `read.len()` bytes.
    fn copy_dirty_mapped(&self, read: &mut [u8], sub_mask: u64, gpu_seeded: bool) -> usize {
        let sub_count = self.footprints.len();
        let all = all_subs_mask(sub_count);
        if !self.on_demand
            && (!self.gpu_detile || gpu_seeded)
            && sub_count > 1
            && sub_count <= 64
            && (sub_mask & all) != all
        {
            let mut spans: Vec<(usize, usize)> = Vec::new();
            let mut ok = true;
            for i in 0..sub_count {
                if !sub_selected(sub_mask, i, sub_count) {
                    continue;
                }
                let (mip, slice) = self.part(i);
                let Some((s, e)) = self.layout.subresource_byte_span(mip, slice) else {
                    ok = false;
                    break;
                };
                let (s, e) = (s as usize, e as usize);
                if s > e || e > read.len() {
                    ok = false;
                    break;
                }
                if e > s {
                    spans.push((s, e));
                }
            }
            if ok && !spans.is_empty() {
                spans.sort_unstable();
                let mut merged: Vec<(usize, usize)> = Vec::with_capacity(spans.len());
                let mut total = 0usize;
                for (s, e) in spans {
                    if let Some(last) = merged.last_mut()
                        && s <= last.1
                    {
                        if e > last.1 {
                            total += e - last.1;
                            last.1 = e;
                        }
                    } else {
                        total += e - s;
                        merged.push((s, e));
                    }
                }
                if total < read.len() {
                    for (s, e) in merged {
                        // SAFETY: `s < e <= read.len() == self.read`; writing
                        // into `read[s..e]` matches `self.memory + s`.
                        unsafe {
                            crate::host::copy_mapped(
                                read.as_mut_ptr().add(s),
                                self.memory.wrapping_add(s),
                                e - s,
                                false,
                            );
                        }
                    }
                    return total;
                }
            }
        }
        crate::host::copy_mapped(read.as_mut_ptr(), self.memory, read.len(), false);
        read.len()
    }

    fn refresh_timed(&self) -> usize {
        let size = self.read;
        let start = self.memory as usize;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        let last = &mut *last;
        let exact = self.exact.swap(false, std::sync::atomic::Ordering::Relaxed);
        let alone = self.once.load(std::sync::atomic::Ordering::Relaxed);
        if alone && !exact && last.hash.is_some() {
            return 0;
        }
        if !alone && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, false)) {
            let prev_asked = last.asked;
            if !exact && prev_asked != 0 && written.iter().all(|&(_, w)| w <= prev_asked) {
                return 0;
            }
            let sub_mask = if !exact && prev_asked != 0 {
                dirty_subs_from_pages(
                    &self.layout,
                    self.first,
                    self.mips,
                    self.footprints.len(),
                    start,
                    size,
                    &written,
                    prev_asked,
                )
            } else {
                all_subs_mask(self.footprints.len())
            };
            last.asked = now;
            if sub_mask == 0 {
                return 0;
            }
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            let copied = with_read(size, |read| {
                let copied = self.copy_dirty_mapped(read, sub_mask, last.gpu_seeded);
                self.detile_masked(read, last, sub_mask);
                copied
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, copied as u64);
            return copied;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken periodically by the texture's address so
        // textures take turns.
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // The guard saw no write: a change the hash finds came another way.
        let mut blind = false;
        if !alone
            && *GUARD.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) = waited(1, || crate::watch::writes(start, start + size, true))
        {
            const HASH_EVERY: u64 = 8;
            let prev_asked = last.asked;
            let unwritten = prev_asked != 0 && written.iter().all(|&(_, w)| w <= prev_asked);
            let guarded_dirty_mask = if !unwritten && prev_asked != 0 && last.hash.is_some() {
                Some(dirty_subs_from_pages(
                    &self.layout,
                    self.first,
                    self.mips,
                    self.footprints.len(),
                    start,
                    size,
                    &written,
                    prev_asked,
                ))
            } else {
                None
            };
            last.asked = now;
            let frame = crate::frame::current();
            let recent_ahead = last.ahead.is_some_and(|at| frame <= at + 1);
            let turn = frame
                .wrapping_add((start >> 16) as u64)
                .is_multiple_of(HASH_EVERY)
                && !recent_ahead;
            // A texture starts hot: one written through an alias view from its
            // first frames on changes before the periodic hash catches it.
            if last.hot_until == 0 {
                last.hot_until = frame + HOT_FRAMES;
            }
            let hot = hot_textures() && frame < last.hot_until && !recent_ahead;
            if !exact && (unwritten || guarded_dirty_mask == Some(0)) && !turn && !hot {
                return 0;
            }
            blind = unwritten || guarded_dirty_mask == Some(0);
        }
        // A change the guard did not see: hashed at every check for a while.
        let found_blind = |last: &mut Last| {
            if blind && hot_textures() {
                last.hot_until = crate::frame::current() + HOT_FRAMES;
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHot, 1);
            }
        };
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() && !exact {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now() else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else {
            waited(2, crate::watch::title_memory_reading)
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // shortcut: read again at most every 30 frames (reading around
            // unmapped pages is slow, and nothing tells us when pages
            // arrive): a mip that streams in shows up to 30 frames late.
            let frame = crate::frame::current();
            if !exact && last.partial.is_some_and(|at| frame < at + 30) {
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                let hash = digest(read);
                if last.hash == Some(hash) {
                    return false;
                }
                let had_prev = last.hash.is_some();
                last.hash = Some(hash);
                if read.iter().all(|&b| b == 0) {
                    return false;
                }
                let sub_mask = self.changed_subs_by_hash(read, last, had_prev);
                self.detile_masked(read, last, sub_mask);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            if changed {
                found_blind(last);
            }
            return size;
        }
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = digest(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if last.hash == Some(hash) {
                    return size;
                }
                let had_prev = last.hash.is_some();
                last.hash = Some(hash);
                found_blind(last);
                let sub_mask = self.changed_subs_by_hash(tiled, last, had_prev);
                self.detile_masked(tiled, last, sub_mask);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = digest(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        let had_prev = last.hash.is_some();
        last.hash = Some(hash);
        found_blind(last);
        let sub_mask = self.changed_subs_by_hash(tiled, last, had_prev);
        self.detile_masked(tiled, last, sub_mask);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles the texture memory into the staging buffer ahead of the
    /// submission, on the thread of `crate::heap_worker`: read through the
    /// kernel, so that the runtime's remapping never waits for it, whole or
    /// not at all (the submission then does it, as ever). False if it was
    /// not read. What the title writes after this is found at submission.
    pub(crate) fn refresh_ahead(&self) -> bool {
        let size = self.read;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if last.hash.is_some() || last.asked != 0 {
            // Detiled before: the submission's check is cheap.
            return true;
        }
        let asked = crate::watch::writes(start, start + size, false)
            .or_else(|| crate::watch::writes(start, start + size, true));
        let Some((now, _)) = asked else {
            return false;
        };
        with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            last.hash = Some(digest(read));
            let sub_mask = self.changed_subs_by_hash(read, last, false);
            self.detile_masked(read, last, sub_mask);
            last.asked = now;
            last.ahead = Some(crate::frame::current());
            true
        })
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then a staging buffer that was filled
    /// before keeps what it holds and the texture is looked at again at its
    /// next refresh (what was learnt about writes to it is forgotten, so
    /// that it is not passed over as unwritten); one never filled takes the
    /// pages that are there ([`Staging::fill`]). Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            let hash = digest(read);
            if last.hash != Some(hash) {
                let had_prev = last.hash.is_some();
                last.hash = Some(hash);
                let sub_mask = self.changed_subs_by_hash(read, last, had_prev);
                self.detile_masked(read, last, sub_mask);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else if last.hash.is_none() && fill_unlocked() {
            self.fill(last, size)
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// The first filling of the staging buffer while the runtime is
    /// remapping title memory, of a texture not all of which can be read:
    /// with the pages that are there, zeros for the others, as a partly
    /// mapped texture is filled under the lock. Pages that are away only
    /// for the moment of a change (the runtime unmaps the rest of a view to
    /// change one page of it) are read when that change is over. If less
    /// was read than was mapped when the upload was recorded, or is mapped
    /// now, the texture's slots are looked at again.
    fn fill(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let start = self.memory as u64;
        let mut there = 0;
        let got = with_read(size, |read| {
            let mut got = 0;
            for _ in 0..4 {
                got = crate::host::read_there(self.memory, read) as u64;
                there = mapped_bytes(self.memory, size);
                if got >= there.max(self.expected.load(Relaxed)) {
                    break;
                }
                crate::watch::change_over();
            }
            if got != 0 {
                let had_prev = last.hash.is_some();
                last.hash = Some(digest(read));
                let sub_mask = self.changed_subs_by_hash(read, last, had_prev);
                self.detile_masked(read, last, sub_mask);
            }
            got
        });
        last.partial = Some(crate::frame::current());
        UNLOCKED[2].fetch_add(1, Relaxed);
        if got < there.max(self.expected.load(Relaxed)) {
            UNLOCKED[3].fetch_add(1, Relaxed);
            last.asked = 0;
            crate::descriptor::host::read_short(start);
        } else {
            crate::descriptor::host::read_all(start);
        }
        size
    }

    /// Computes per-subresource digests against `last.sub_hashes` and returns
    /// the bitmask of subresources whose tiled bytes changed (always all
    /// subresources on first fill or when `partial_detile()` is off).
    fn changed_subs_by_hash(&self, tiled: &[u8], last: &mut Last, had_prev: bool) -> u64 {
        let sub_count = self.footprints.len();
        let all = all_subs_mask(sub_count);
        if !partial_detile() || sub_count <= 1 || sub_count > 64 {
            return all;
        }
        if last.sub_hashes.len() != sub_count {
            last.sub_hashes.resize(sub_count, None);
        }
        let mut mask = if had_prev { 0u64 } else { all };
        for i in 0..sub_count {
            let (mip, slice) = self.part(i);
            let Some((s, e)) = self.layout.subresource_byte_span(mip, slice) else {
                last.sub_hashes[i] = None;
                mask |= 1u64 << i;
                continue;
            };
            let (s, e) = (s as usize, e as usize);
            if e > tiled.len() {
                last.sub_hashes[i] = None;
                mask |= 1u64 << i;
                continue;
            }
            let h = digest(&tiled[s..e]);
            if !had_prev || last.sub_hashes[i] != Some(h) {
                last.sub_hashes[i] = Some(h);
                mask |= 1u64 << i;
            }
        }
        mask
    }

    /// Detiles (or stages raw tiled bytes for GPU compute detiling) the
    /// subresources selected by `sub_mask` from `tiled` into the staging buffer.
    fn detile_masked(&self, tiled: &[u8], last: &mut Last, mut sub_mask: u64) {
        let sub_count = self.footprints.len();
        let all = all_subs_mask(sub_count);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::{AcqRel, Relaxed};
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            let prev_held = self.detiled.swap(held, AcqRel);
            let flipped = held ^ prev_held;
            if flipped != 0 && sub_count <= 64 {
                for i in 0..sub_count {
                    let (mip, _) = self.part(i);
                    if mip < 32 && (flipped & (1 << mip)) != 0 {
                        sub_mask |= 1u64 << i;
                    }
                }
            } else if flipped != 0 {
                sub_mask = all;
            }
            held
        } else {
            u32::MAX
        };
        if sub_mask == 0 {
            return;
        }
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        if self.gpu_detile {
            let copy_len = self.read.min(tiled.len()).min(self.bytes as usize);
            let full_copy = !last.gpu_seeded || sub_count > 64 || (sub_mask & all) == all;
            let mut copied_partial = false;
            if !full_copy {
                copied_partial = true;
                for i in 0..sub_count {
                    if !sub_selected(sub_mask, i, sub_count) {
                        continue;
                    }
                    let (mip, slice) = self.part(i);
                    let Some((s, e)) = self.layout.subresource_byte_span(mip, slice) else {
                        copied_partial = false;
                        break;
                    };
                    let (s, e) = (s as usize, e as usize);
                    if e > copy_len {
                        copied_partial = false;
                        break;
                    }
                    // SAFETY: `self.data` is mapped for `self.bytes >= copy_len >= e` bytes.
                    unsafe {
                        std::ptr::copy_nonoverlapping(tiled.as_ptr().add(s), self.data.add(s), e - s);
                    }
                }
            }
            if !copied_partial {
                // SAFETY: `self.data` is mapped for `self.bytes >= copy_len` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(tiled.as_ptr(), self.data, copy_len);
                }
                last.gpu_seeded = true;
            }
            self.dirty_subs
                .fetch_or(sub_mask & all, std::sync::atomic::Ordering::Release);
            self.dirty.store(true, std::sync::atomic::Ordering::Release);
            return;
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            if !sub_selected(sub_mask, i, sub_count) {
                continue;
            }
            let (mip, slice) = self.part(i);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            // The last row of the last subresource is not padded to its pitch
            // (VIctory 4fc9d70): to the end of the buffer at most, where a
            // block of the pool goes on with another texture's.
            let len = (pitch * rows as usize * f.Depth as usize)
                .min(self.linear_bytes.saturating_sub(footprint.Offset) as usize);
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
        // Publish dirty only after `detiled` and every texel in `self.data`
        // have been written so a prelude upload never reads half-detiled bytes.
        self.dirty_subs
            .fetch_or(sub_mask & all, std::sync::atomic::Ordering::Release);
        self.dirty.store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let mut map = STAGING.write().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut()
        && map.remove(&(resource as usize)).is_some()
    {
        STAGING_COUNT.store(map.len(), Ordering::Release);
        STAGING_EPOCH.fetch_add(1, Ordering::Release);
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of
/// it up to 48 KiB, else of 16 KiB at its start, its middle and its end
/// (a texture streams in whole mips; the small ones lie first). Read
/// through the kernel, so pages not there count as zeros.
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const WINDOW: usize = 16 << 10;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    let mut print = 0u128;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(3 * WINDOW, 0);
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            print = print.rotate_left(37) ^ digest(&read[..bytes]).0;
        };
        if len <= 3 * WINDOW {
            take(0, len);
        } else {
            take(0, WINDOW);
            take((len / 2) & !0xfff, WINDOW);
            take(len - WINDOW, WINDOW);
        }
    });
    let first = (print as u64 ^ (print >> 64) as u64).to_le_bytes();
    u64::from_le_bytes(first) | 1
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

/// The first level a title texture's staging buffer holds; `u32::MAX`
/// without one.
fn staged_from(resource: This) -> u32 {
    if let Some(s) = staging_tls_get(resource as usize) {
        return s.first;
    }
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return u32::MAX;
    }
    let hit = {
        let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
        let epoch = STAGING_EPOCH.load(Ordering::Acquire);
        map.as_ref()
            .and_then(|map| map.get(&(resource as usize)))
            .map(|s| (epoch, s.clone()))
    };
    if let Some((epoch, s)) = hit {
        let first = s.first;
        staging_tls_put(epoch, resource as usize, &s);
        first
    } else {
        u32::MAX
    }
}

/// Whether the levels from `first` on of `resource`'s host copy were filled
/// from the bytes title memory holds now: an upload would bring nothing new.
/// (Of 787 repeated uploads of a frame that streams 398 were equal in every
/// byte, and a texture uploaded again costs a staging buffer, a detiling and
/// a copy more than this look.) `VICTORY_TEXTURE_SAME=0`: not asked.
pub(crate) fn unchanged(resource: This, first: u32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_SAME").is_none_or(|v| v != "0")) {
        return false;
    }
    let Some(copy) = crate::resource::made_copy(resource) else {
        return false;
    };
    let Some((from, held)) = *copy.holds.lock().unwrap_or_else(|e| e.into_inner()) else {
        return false;
    };
    let Some((memory, desc)) = crate::resource::texture_memory(resource) else {
        return false;
    };
    let Some(layout) = crate::resource::texture_layout(&desc) else {
        return false;
    };
    let layout = &layout.planes[0].layout;
    let wanted = crate::resource::read_from(layout, first);
    let target = crate::resource::base_level_for(&desc, first);
    if copy.base > target || from > wanted.max(target) {
        return false;
    }
    let size = layout.bytes_from(from) as usize;
    with_read(size, |read| {
        if crate::host::read_there(memory, read) == 0 {
            return false;
        }
        digest(read) == held
    })
}

/// Where the levels from `first` on of a title texture are: address and
/// bytes.
pub(crate) fn title_span(resource: This, first: u32) -> Option<(u64, u64)> {
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?;
    let layout = &layout.planes[0].layout;
    Some((memory as u64, layout.bytes_from(crate::resource::read_from(layout, first))))
}

pub(crate) fn is_staged(resource: This) -> bool {
    if STAGING_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    if staging_tls_get(resource as usize).is_some() {
        return true;
    }
    let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
/// (Whether it is mapped is asked only if `ask`; else it counts as mapped.)
pub(crate) fn staged(resource: This, ask: bool) -> Option<(u64, bool)> {
    let s = if let Some(s) = staging_tls_get(resource as usize) {
        s
    } else {
        if STAGING_COUNT.load(Ordering::Acquire) == 0 {
            return None;
        }
        let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    Some((s.read as u64, !ask || committed(s.memory, s.read)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    if crate::safecopy::on() {
        // SAFETY: `safecopy::on()` installed the fault handler; non-mutating probe.
        return unsafe { crate::safecopy::probe_readable(at, len, 4096) };
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let (first, last) = (at / KERNEL_PAGE, at.saturating_add(len - 1) / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

fn mapped_by_views() -> bool {
    static BY_VIEWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BY_VIEWS.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_VIEWS").is_none_or(|v| v != "0"))
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. A texture can also have only interior mips present while
/// streaming; Windows skips each uncommitted run in a single query.
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    let start = at as usize;
    let end = start.saturating_add(len);
    if start == end {
        return 0;
    }
    // Streamed textures lie in views of physical pages, of which the runtime
    // knows which are there: one look, where the host was asked for every
    // 64 KiB of the texture (and once more for each that is not there).
    // `VICTORY_MAPPED_BY_VIEWS=0`: the host is asked, as before.
    if mapped_by_views()
        && let Some(there) = crate::inplace::shown_bytes(start, len)
    {
        return there;
    }
    let query = by_query();
    let ends_missing = !query && !committed(at, 1) && !committed((end - 1) as *const u8, 1);
    if ends_missing && (end - 1) / 4096 - start / 4096 <= 1 {
        return 0;
    }
    if !query && !ends_missing {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            if crate::safecopy::on() {
                // SAFETY: `safecopy::on()` installed the fault handler; non-mutating probe.
                return unsafe { crate::safecopy::readable_bytes(at, len, 4096) } as u64;
            }
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let fast = crate::safecopy::on();
        // Fast path: all 64 KiB blocks present in one zero-syscall probe.
        // SAFETY: `safecopy::on()` installed the fault handler; non-mutating probe.
        if fast && unsafe { crate::safecopy::probe_readable(at, len, BLOCK) } {
            return len as u64;
        }
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            // SAFETY: `safecopy::on()` installed the fault handler; non-mutating probe.
            let present = if fast {
                unsafe { crate::safecopy::probe_readable(lo as *const u8, 1, 1) }
            } else {
                readable(lo, 1, true).0 == 1
            };
            if present {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    mapped_regions(start, end, false)
}

/// Whether `[start, start + len)` has any readable committed page.
pub(crate) fn any_mapped(start: usize, len: usize) -> bool {
    let end = start.saturating_add(len);
    if start == end {
        return false;
    }
    if mapped_by_views()
        && let Some(there) = crate::inplace::shown_bytes(start, len)
    {
        return there > 0;
    }
    if committed(start as *const u8, 1) {
        return true;
    }
    (end - 1) / 4096 > start / 4096 && mapped_regions(start, end, true) > 0
}

fn mapped_regions(start: usize, end: usize, first_only: bool) -> u64 {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if info.State == MEM_COMMIT && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 {
            there += (next - p) as u64;
            if first_only {
                break;
            }
        }
        p = next;
    }
    there
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, copy: &crate::resource::HostTexture, first: u32) -> Option<Arc<Staging>> {
    let key = resource as usize;
    let need_first = first.max(copy.base);
    if let Some(s) = staging_tls_get(key)
        && s.first <= need_first
        && s.serial == copy.serial
    {
        return Some(s);
    }
    if STAGING_COUNT.load(Ordering::Acquire) > 0 {
        let hit = {
            let map = STAGING.read().unwrap_or_else(|e| e.into_inner());
            let epoch = STAGING_EPOCH.load(Ordering::Acquire);
            map.as_ref()
                .and_then(|m| m.get(&key))
                .filter(|s| s.first <= need_first && s.serial == copy.serial)
                .map(|s| (epoch, s.clone()))
        };
        if let Some((epoch, s)) = hit {
            staging_tls_put(epoch, key, &s);
            return Some(s);
        }
    }
    let texture = &copy.texture;
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut guard = STAGING.write().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&key)
        && s.first <= need_first
        && s.serial == copy.serial
    {
        let epoch = STAGING_EPOCH.load(Ordering::Acquire);
        let out = s.clone();
        drop(guard);
        staging_tls_put(epoch, key, &out);
        return Some(out);
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let level0 = copy.base;
    let mips = level0 + u32::from(d.MipLevels);
    let first = if copy.grown.load(Ordering::Relaxed) {
        level0
    } else {
        crate::resource::read_from(&layout, first).max(level0).min(mips - 1)
    };
    let read = layout.bytes_from(first) as usize;
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        mips - first
    } else {
        (mips - first) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            first - level0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let use_gpu_detile =
        !COPY_QUEUE_ONLY.get() && crate::gpu_detile::can_use(&layout, read, total);
    let alloc_bytes = if use_gpu_detile {
        (read as u64).next_multiple_of(4).max(4)
    } else {
        total
    };
    let (piece, buffer, base, data) = match crate::pool::staging(host, alloc_bytes) {
        Some((piece, buffer, base, data)) => (Some(piece), buffer, base, data),
        None => {
            let speculative = PICKING.get().is_some() && !BAKING.get();
            if crate::residency::upload_refused()
                || (speculative && !crate::residency::allow_staging_growth(false))
            {
                crate::residency::note_staging_deferred();
                return None;
            }
            let (buffer, data) = host.upload_buffer(alloc_bytes)?;
            crate::host::name(&buffer, || {
                format!(
                    "staging of {resource:?} {}x{}x{} f{}",
                    d.Width, d.Height, d.DepthOrArraySize, d.Format.0
                )
            });
            (None, buffer, 0, data)
        }
    };
    tracing::debug!(?desc, total, alloc_bytes, use_gpu_detile, "texture contents uploaded from title memory");
    STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    STAGED[1].fetch_add(alloc_bytes, std::sync::atomic::Ordering::Relaxed);
    let init_dirty_subs = all_subs_mask(footprints.len());
    let s = Arc::new(Staging {
        buffer,
        base,
        _piece: piece,
        #[cfg(feature = "diagnostics")]
        size: alloc_bytes,
        data,
        footprints,
        rows,
        layout,
        mips,
        first,
        level0,
        read,
        serial: copy.serial,
        holds: copy.holds.clone(),
        once: std::sync::atomic::AtomicBool::new(once() && !on_demand && PICKING.get().is_some()),
        bytes: alloc_bytes,
        linear_bytes: total,
        gpu_detile: use_gpu_detile,
        gpu_detile_aliased: std::sync::atomic::AtomicBool::new(false),
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        dirty_subs: std::sync::atomic::AtomicU64::new(init_dirty_subs),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        expected: std::sync::atomic::AtomicU64::new(0),
        on_demand,
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
        exact: std::sync::atomic::AtomicBool::new(BAKING.get()),
    });
    map.insert(key, s.clone());
    STAGING_COUNT.store(map.len(), Ordering::Release);
    let epoch = STAGING_EPOCH.fetch_add(1, Ordering::Release) + 1;
    drop(guard);
    staging_tls_put(epoch, key, &s);
    Some(s)
}

#[cfg(test)]
mod title_memory_tests {
    use super::with_title;

    #[test]
    fn a_texture_with_only_interior_pages_counts_its_resident_bytes() {
        use windows_sys::Win32::System::Memory::*;
        const BLOCK: usize = 64 << 10;
        // SAFETY: this private reservation is ours throughout the test.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 5 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let middle = base.add(2 * BLOCK);
            assert_eq!(VirtualAlloc(middle.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE), middle.cast());
            assert_eq!(super::mapped_bytes(base, 5 * BLOCK), BLOCK as u64);
            assert!(super::any_mapped(base as usize, 5 * BLOCK));
            assert_eq!(super::mapped_bytes(base.add(BLOCK - 3), 2 * BLOCK + 9), BLOCK as u64);
            assert_eq!(super::mapped_bytes(base, 2 * BLOCK), 0);
            assert_ne!(VirtualFree(middle.cast(), BLOCK, MEM_DECOMMIT), 0);
            assert_eq!(super::mapped_bytes(base, 5 * BLOCK), 0);
            assert!(!super::any_mapped(base as usize, 5 * BLOCK));
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }
}

#[cfg(test)]
mod prelude_upload_tests {
    use super::claim_upload;
    use std::sync::atomic::{AtomicBool, Ordering::Release};

    #[test]
    fn first_upload_clean_skip_and_immediate_same_frame_reupload() {
        let uploaded = AtomicBool::new(false);
        let dirty = AtomicBool::new(true);

        // Frame 1: first upload runs after refresh() detiled into staging.
        assert_eq!(claim_upload(&uploaded, &dirty), Some(true));
        // Second list in the same submission skips redundant copy.
        assert_eq!(claim_upload(&uploaded, &dirty), None);

        // Frame 2: texture unchanged — refresh() does not detile, prelude skips.
        assert_eq!(claim_upload(&uploaded, &dirty), None);

        // Frame 3: texture changed — refresh()/detile() runs BEFORE claim_upload,
        // so upload is claimed on the exact same frame (0 frames late).
        dirty.store(true, Release);
        assert_eq!(claim_upload(&uploaded, &dirty), Some(false));

        // Frame 4: consecutive change (e.g. refresh_front() on every frame)
        // must also upload immediately, never alternating upload/skip.
        dirty.store(true, Release);
        assert_eq!(claim_upload(&uploaded, &dirty), Some(false));
    }

    #[test]
    fn three_buffer_ring_never_displays_stale_frame() {
        let ring: [(AtomicBool, AtomicBool); 3] =
            std::array::from_fn(|_| (AtomicBool::new(false), AtomicBool::new(true)));
        let mut gpu_frame = [0u32; 3];

        // 12 decoded frames cycling through the 3-buffer ring (like Bink intro),
        // with frame 5 repeated once to exercise the clean-frame transition.
        let schedule = [
            (0, 1, true),
            (1, 2, true),
            (2, 3, true),
            (0, 4, true),
            (1, 5, true),
            (1, 5, false), // held frame: unchanged
            (2, 6, true),
            (0, 7, true),
            (1, 8, true), // immediately after held frame: must NOT be stale!
            (2, 9, true),
            (0, 10, true),
            (1, 11, true),
        ];

        for (slot, video_frame, changed) in schedule {
            let (uploaded, dirty) = &ring[slot];
            if changed {
                dirty.store(true, Release);
            }
            if claim_upload(uploaded, dirty).is_some() {
                gpu_frame[slot] = video_frame;
            }
            assert_eq!(
                gpu_frame[slot], video_frame,
                "slot {slot} displayed stale frame {} instead of {video_frame}",
                gpu_frame[slot]
            );
        }
    }

    #[test]
    fn dirty_pages_map_to_only_affected_subresources() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        // 1024x1024 BC7, 11 mips, 64 KiB display swizzle:
        // tail = mips 3..11 in [0, 64 KiB), mip 2 in [64 KiB, 128 KiB),
        // mip 1 in [128 KiB, 384 KiB), mip 0 in [384 KiB, 1408 KiB).
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 1024,
            height: 1024,
            depth: 1,
            mips: 11,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        })
        .unwrap();
        let page_bytes = crate::watch::PAGE;
        let base_page = 0x1000usize;
        let start = base_page * page_bytes;
        let read = layout.size as usize;

        // Dirty page inside mip 0 only -> ONLY bit 0 set.
        let (mip0_lo, _) = layout.subresource_byte_span(0, 0).unwrap();
        let mip0_page = base_page + (mip0_lo as usize / page_bytes) + 1;
        let mask0 = super::dirty_subs_from_pages(&layout, 0, 11, 11, start, read, &[(mip0_page, 5)], 4);
        assert_eq!(mask0, 1u64 << 0, "mip 0 page should only dirty mip 0");

        // Dirty page inside mip 1 only -> ONLY bit 1 set.
        let (mip1_lo, _) = layout.subresource_byte_span(1, 0).unwrap();
        let mip1_page = base_page + (mip1_lo as usize / page_bytes);
        let mask1 = super::dirty_subs_from_pages(&layout, 0, 11, 11, start, read, &[(mip1_page, 5)], 4);
        assert_eq!(mask1, 1u64 << 1, "mip 1 page should only dirty mip 1");

        // Dirty page in the mip tail -> only tail mips (first_tail_mip..11) set.
        let tail_mask = super::dirty_subs_from_pages(&layout, 0, 11, 11, start, read, &[(base_page, 5)], 4);
        let expected_tail: u64 = (layout.first_tail_mip..11).fold(0, |m, mip| m | (1u64 << mip));
        assert_eq!(tail_mask, expected_tail, "tail page should only dirty tail mips");

        // Stale epoch (<= since) produces 0 dirty subresources.
        let clean = super::dirty_subs_from_pages(&layout, 0, 11, 11, start, read, &[(mip0_page, 4)], 4);
        assert_eq!(clean, 0);

        // 2D array (4 slices, 4 mips): dirty page in slice 2 mip 0 -> ONLY subresource 2*4 + 0.
        let array = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 512,
            height: 512,
            depth: 4,
            mips: 4,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_S,
        })
        .unwrap();
        let (s2m0_lo, _) = array.subresource_byte_span(0, 2).unwrap();
        let s2m0_page = base_page + (s2m0_lo as usize / page_bytes);
        let mask_arr = super::dirty_subs_from_pages(
            &array,
            0,
            4,
            16,
            start,
            array.size as usize,
            &[(s2m0_page, 9)],
            8,
        );
        assert_eq!(mask_arr, 1u64 << (2 * 4));
    }

    #[test]
    fn gpu_detile_end_to_end_and_partial_mip_upload_match_cpu() {
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC7_UNORM, DXGI_SAMPLE_DESC};
        let Some(host) = crate::host::host() else {
            return;
        };
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 512,
            height: 512,
            depth: 1,
            mips: 10,
            element_bytes: 16,
            block: 4,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        })
        .unwrap();
        let mut tiled: Vec<u8> = (0..layout.size as usize)
            .map(|i| ((i.wrapping_mul(2654435761)) >> 19) as u8)
            .collect();

        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 512,
            Height: 512,
            DepthOrArraySize: 1,
            MipLevels: 10,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 10];
        let mut rows = vec![0u32; 10];
        let mut total = 0u64;
        // SAFETY: valid queries on live device.
        unsafe {
            host.device.GetCopyableFootprints(
                &tex_desc,
                0,
                10,
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
        // SAFETY: plain texture creation.
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
        let texture = tex_opt.unwrap();

        let upload_bytes = layout.size.next_multiple_of(4);
        let (tiled_buf, tiled_ptr) = host.upload_buffer(upload_bytes).unwrap();
        let scratch = crate::gpu_detile::shared_scratch(host, total.next_multiple_of(4)).unwrap();

        let run_upload = |tiled_src: &[u8], sub_mask: u64| {
            // SAFETY: `tiled_ptr` is mapped for `upload_bytes >= tiled_src.len()`.
            unsafe {
                std::ptr::copy_nonoverlapping(tiled_src.as_ptr(), tiled_ptr, tiled_src.len());
            }
            let jobs: Vec<_> = (0..10usize)
                .filter(|&i| super::sub_selected(sub_mask, i, 10))
                .map(|i| crate::gpu_detile::SubresourceJob {
                    mip: i as u32,
                    slice: 0,
                    dst_offset: footprints[i].Offset,
                    row_pitch: footprints[i].Footprint.RowPitch,
                    depth_pitch: footprints[i].Footprint.RowPitch * rows[i],
                    zero: false,
                })
                .collect();
            let list = host.new_list().unwrap();
            let common = D3D12_RESOURCE_STATE_COMMON;
            let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            let copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
            // SAFETY: open direct list, live resources.
            unsafe {
                list.list
                    .ResourceBarrier(&[crate::host::transition(&scratch, common, uav)]);
                assert!(crate::gpu_detile::record_detile_subresources(
                    &list.list,
                    &layout,
                    tiled_buf.GetGPUVirtualAddress(),
                    tiled_src.len() as u64,
                    &scratch,
                    total,
                    &jobs,
                ));
                list.list
                    .ResourceBarrier(&[crate::host::transition(&scratch, uav, copy_src)]);
                for i in (0..10usize).filter(|&i| super::sub_selected(sub_mask, i, 10)) {
                    let to = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&texture),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
                    };
                    let from = D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&scratch),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprints[i] },
                    };
                    list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                }
                list.list
                    .ResourceBarrier(&[crate::host::transition(&scratch, copy_src, common)]);
            }
            list.close();
            host.execute(&[&list]);
        };

        let readback_mip = |mip: usize| -> Vec<u8> {
            let fp = footprints[mip];
            let pitch = fp.Footprint.RowPitch as usize;
            let len = pitch * rows[mip] as usize;
            let rb = host.read_back_buffer(len as u64).unwrap();
            let list = host.new_list().unwrap();
            // SAFETY: open direct list, transition texture COPY_DEST -> COPY_SOURCE -> COPY_DEST.
            unsafe {
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
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
                    pResource: std::mem::transmute_copy(&texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: mip as u32 },
                };
                list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: readback buffer mapped for `len` bytes.
            unsafe {
                rb.Map(0, None, Some(&mut ptr)).unwrap();
                let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
                rb.Unmap(0, None);
                bytes
            }
        };

        // Pass 1: full GPU-detiled upload of all 10 mips.
        run_upload(&tiled, super::all_subs_mask(10));
        for mip in [0usize, 1, 2, 5, 9] {
            let pitch = footprints[mip].Footprint.RowPitch as usize;
            let h = rows[mip] as usize;
            let mut expected = vec![0u8; pitch * h];
            layout
                .detile(&tiled, mip as u32, 0, &mut expected, pitch, pitch * h)
                .unwrap();
            let got = readback_mip(mip);
            let active = layout.mips[mip].width as usize * 16;
            for y in 0..h {
                assert_eq!(
                    &got[y * pitch..y * pitch + active],
                    &expected[y * pitch..y * pitch + active],
                    "full upload mip {mip} row {y}"
                );
            }
        }

        // Pass 2: mutate ONLY mip 0 in tiled memory and upload ONLY subresource 0.
        let (mip0_s, mip0_e) = layout.subresource_byte_span(0, 0).unwrap();
        for b in &mut tiled[mip0_s as usize..mip0_e as usize] {
            *b ^= 0x5a;
        }
        run_upload(&tiled, 1u64 << 0);

        // Verify mip 0 updated bit-for-bit AND mip 1 stayed intact.
        for mip in [0usize, 1] {
            let pitch = footprints[mip].Footprint.RowPitch as usize;
            let h = rows[mip] as usize;
            let mut expected = vec![0u8; pitch * h];
            layout
                .detile(&tiled, mip as u32, 0, &mut expected, pitch, pitch * h)
                .unwrap();
            let got = readback_mip(mip);
            let active = layout.mips[mip].width as usize * 16;
            for y in 0..h {
                assert_eq!(
                    &got[y * pitch..y * pitch + active],
                    &expected[y * pitch..y * pitch + active],
                    "partial upload mip {mip} row {y}"
                );
            }
        }
    }

    static FAST_PATH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn host_written_lock_free_fast_path_and_epoch_invalidation() {
        use std::sync::atomic::Ordering;
        let _guard = FAST_PATH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let res_a = 0xDEAD_BEE0usize;
        let res_b = 0xDEAD_BEE8usize;
        super::forget(res_a);
        super::forget(res_b);

        assert!(!super::host_wrote(res_a));
        assert!(!super::host_wrote(res_b));

        // Insert `res_a` under `HOST_WRITTEN` (mirrors `host_written` without COM deref).
        let epoch0 = super::HOST_WRITTEN_EPOCH.load(Ordering::Acquire);
        {
            let mut set = super::HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            let set = set.get_or_insert_with(Default::default);
            assert!(set.insert(res_a));
            super::HOST_WRITTEN_COUNT.store(set.len(), Ordering::Release);
            super::HOST_WRITTEN_EPOCH.fetch_add(1, Ordering::Release);
        }
        let epoch1 = super::HOST_WRITTEN_EPOCH.load(Ordering::Acquire);
        assert!(epoch1 > epoch0);

        // First call populates TLS snapshot; second call hits TLS with identical epoch.
        assert!(super::host_wrote(res_a));
        assert!(!super::host_wrote(res_b));
        super::HOST_WRITTEN_TLS.with(|cell| {
            let cache = cell.borrow();
            assert_eq!(cache.0, epoch1);
            assert!(cache.1.binary_search(&res_a).is_ok());
        });

        // Cross-thread visibility: a fresh thread observes `res_a` via Acquire/Release.
        let seen_other_thread = std::thread::spawn(move || super::host_wrote(res_a) && !super::host_wrote(res_b))
            .join()
            .unwrap();
        assert!(seen_other_thread);

        // `forget` removes `res_a`, decrements count, and bumps epoch to invalidate TLS.
        super::forget(res_a);
        assert!(super::HOST_WRITTEN_EPOCH.load(Ordering::Acquire) > epoch1);
        assert!(!super::host_wrote(res_a));
    }

    #[test]
    fn staging_and_through_tls_caches_invalidate_on_unstage_and_forget_without_pinning_arc() {
        use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
        use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
        let _guard = FAST_PATH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let key = 0xCAFE_0010usize;
        let res = key as crate::object::This;
        super::forget(key);

        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex2d,
            width: 64,
            height: 64,
            depth: 1,
            mips: 1,
            element_bytes: 4,
            block: 1,
            samples: 1,
            swizzle: Swizzle::SW_64KB_D,
        })
        .unwrap();
        // SAFETY: dummy interface pointers never dereferenced; leaked once via `mem::forget`
        // so `ID3D12Resource::drop` never runs.
        let dummy_buf = || unsafe { std::mem::transmute::<usize, ID3D12Resource>(8) };
        let staging = std::sync::Arc::new(super::Staging {
            buffer: dummy_buf(),
            base: 0,
            _piece: None,
            #[cfg(feature = "diagnostics")]
            size: 0,
            data: std::ptr::null_mut(),
            footprints: Vec::new(),
            rows: Vec::new(),
            layout,
            mips: 1,
            first: 2,
            level0: 0,
            read: 0,
            serial: 7,
            holds: std::sync::Arc::new(std::sync::Mutex::new(None)),
            once: AtomicBool::new(false),
            bytes: 0,
            linear_bytes: 0,
            gpu_detile: false,
            gpu_detile_aliased: AtomicBool::new(false),
            memory: std::ptr::null(),
            last: std::sync::Mutex::default(),
            dirty: AtomicBool::new(false),
            dirty_subs: AtomicU64::new(0),
            uploaded: AtomicBool::new(true),
            used: AtomicU64::new(0),
            expected: AtomicU64::new(0),
            on_demand: false,
            detiled: AtomicU32::new(0),
            judged: [AtomicU64::new(0), AtomicU64::new(0)],
            waiting: AtomicU64::new(0),
            exact: AtomicBool::new(false),
        });
        std::mem::forget(staging.clone());

        let epoch = {
            let mut map = super::STAGING.write().unwrap_or_else(|e| e.into_inner());
            let map = map.get_or_insert_with(std::collections::HashMap::new);
            map.insert(key, staging.clone());
            super::STAGING_COUNT.store(map.len(), Ordering::Release);
            super::STAGING_EPOCH.fetch_add(1, Ordering::Release) + 1
        };

        // Populate TLS (`Weak<Staging>`) and confirm strong_count is NOT inflated by TLS.
        let strong_before = std::sync::Arc::strong_count(&staging);
        super::staging_tls_put(epoch, key, &staging);
        assert_eq!(std::sync::Arc::strong_count(&staging), strong_before);
        assert!(super::staging_tls_get(key).is_some());
        assert_eq!(super::staged_from(res), 2);
        assert!(super::is_staged(res));

        // `unstage` bumps `STAGING_EPOCH` and immediately invalidates TLS.
        super::unstage(res);
        assert!(super::STAGING_EPOCH.load(Ordering::Acquire) > epoch);
        assert!(super::staging_tls_get(key).is_none());
        assert_eq!(super::staged_from(res), u32::MAX);
        assert!(!super::is_staged(res));

        // Same check for `THROUGH` + `forget`.
        let through = std::sync::Arc::new(super::Through {
            buffer: dummy_buf(),
            parts: Vec::new(),
            synced_gen: AtomicU64::new(0),
        });
        std::mem::forget(through.clone());
        let t_epoch = {
            let mut map = super::THROUGH.write().unwrap_or_else(|e| e.into_inner());
            let map = map.get_or_insert_with(std::collections::HashMap::new);
            map.insert(key, through.clone());
            super::THROUGH_COUNT.store(map.len(), Ordering::Release);
            super::THROUGH_EPOCH.fetch_add(1, Ordering::Release) + 1
        };
        let t_strong = std::sync::Arc::strong_count(&through);
        super::through_tls_put(t_epoch, key, &through);
        assert_eq!(std::sync::Arc::strong_count(&through), t_strong);
        assert!(super::through_tls_get(key).is_some());

        super::forget(key);
        assert!(super::THROUGH_EPOCH.load(Ordering::Acquire) > t_epoch);
        assert!(super::through_tls_get(key).is_none());
    }
}

/// How long this PC's card takes to make the allocations a staged texture needs, for
/// `cargo test -p victory-gpu allocation_costs -- --ignored --nocapture`.
#[cfg(test)]
mod allocation_costs {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    fn texture(size: u32) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(size),
            Height: size,
            DepthOrArraySize: 1,
            MipLevels: 0,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        }
    }

    fn buffer(size: u64) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        }
    }

    fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, flags: D3D12_HEAP_FLAGS, desc: &D3D12_RESOURCE_DESC) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let state = if kind == D3D12_HEAP_TYPE_UPLOAD { D3D12_RESOURCE_STATE_GENERIC_READ } else { D3D12_RESOURCE_STATE_COMMON };
        let mut out = None;
        // SAFETY: plain creation.
        unsafe { device.CreateCommittedResource(&heap, flags, desc, state, None, &mut out) }.expect("committed");
        out.unwrap()
    }

    fn timed(what: &str, n: usize, threads: usize, make: &(dyn Fn(usize) -> ID3D12Resource + Sync)) {
        let began = std::time::Instant::now();
        let all: Vec<_> = std::thread::scope(|s| {
            (0..threads)
                .map(|t| s.spawn(move || (t..n).step_by(threads).map(make).collect::<Vec<_>>()))
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let took = began.elapsed().as_secs_f64() * 1e3;
        println!("{what}, {threads} thread(s): {n} in {took:.1} ms, {:.0} us each", took * 1e3 / n as f64);
        drop(all);
    }

    /// What a look at how much of a texture is there costs: by questions to
    /// the host, and by the runtime's record of its views.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn look_costs() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const BLOCK: usize = 64 << 10;
        for (name, blocks, there) in [("350 KB, all there", 6, 6), ("22 MB, all there", 350, 350), ("22 MB, 1 MB there", 350, 16)] {
            let make = || {
                // SAFETY: a reservation of our own, of which the first part
                // is committed and touched.
                unsafe {
                    let at = VirtualAlloc(std::ptr::null(), blocks * BLOCK, MEM_RESERVE, PAGE_READWRITE);
                    VirtualAlloc(at, there * BLOCK, MEM_COMMIT, PAGE_READWRITE);
                    std::ptr::write_bytes(at.cast::<u8>(), 1, there * BLOCK);
                    at as usize
                }
            };
            let (asked, noted) = (make(), make());
            for block in 0..there {
                crate::title_memory_view(noted + block * BLOCK, BLOCK, 9, 0);
            }
            for (how, at) in [("asking the host", asked), ("by the views", noted)] {
                const N: u32 = 200;
                let began = std::time::Instant::now();
                for _ in 0..N {
                    assert_eq!(super::mapped_bytes(at as *const u8, blocks * BLOCK), (there * BLOCK) as u64);
                }
                println!("{name}, {how}: {:.1} us a look", began.elapsed().as_secs_f64() * 1e6 / f64::from(N));
            }
            crate::inplace::views_gone(noted, blocks * BLOCK);
        }
    }

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        use super::digest;
        let zeros = [0u8; 400];
        let mut seen: Vec<super::Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| (i * 37 >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }

    /// What a texture met for the first time costs, by step, in
    /// milliseconds a megabyte on one thread: for whether a store of
    /// detiled copies on disk could be worth it (its key is a hash of the
    /// title's bytes, so reading and hashing them stay; `VICTORY_BENCH_FILE`
    /// is a big file to read past the system's cache).
    #[test]
    #[ignore = "a measurement, not a check"]
    fn cold_texture_costs() {
        let Some(host) = crate::host::host() else { return };
        let desc = [3u32, 0, 0, 0, 1024, 0, 1024, 0xb_0001, 98, 1, 0, 0x105, 0, 0];
        // SAFETY: 56 bytes in the description's layout.
        let desc = unsafe { desc.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let layout = crate::resource::texture_layout(&desc).unwrap().planes[0].layout.clone();
        let size = layout.size as usize;
        let title: Vec<u8> = (0..size).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let (_buffer, up) = host.upload_buffer(2 * layout.size).unwrap();
        // SAFETY: the mapping.
        let up = unsafe { std::slice::from_raw_parts_mut(up, 2 * size) };
        let mut read = vec![0u8; size];
        const N: u32 = 200;
        let per_mb = |t: std::time::Duration| t.as_secs_f64() * 1e3 / (f64::from(N) * size as f64 / 1e6);
        let timed = |what: &str, f: &mut dyn FnMut()| {
            f();
            let started = std::time::Instant::now();
            (0..N).for_each(|_| f());
            println!("{what}: {:.3} ms/MB", per_mb(started.elapsed()));
        };
        timed("read through the kernel or a safe copy", &mut || {
            std::hint::black_box(crate::host::read_into(title.as_ptr(), &mut read));
        });
        timed("blake3", &mut || {
            std::hint::black_box(blake3::hash(&title));
        });
        timed("digest", &mut || {
            std::hint::black_box(super::digest(&title));
        });
        timed("detiled into upload memory", &mut || {
            let mut at = 0;
            for (mip, m) in layout.mips.iter().enumerate() {
                let pitch = (m.width as usize * 16).next_multiple_of(256);
                let len = pitch * m.height as usize;
                layout.detile(&title, mip as u32, 0, &mut up[at..at + len], pitch, len).unwrap();
                at += len.next_multiple_of(512);
            }
        });
        timed("copied into upload memory", &mut || up[..size].copy_from_slice(&title));
        if let Some(file) = std::env::var_os("VICTORY_BENCH_FILE") {
            use std::os::windows::fs::{FileExt, OpenOptionsExt};
            const NO_BUFFERING: u32 = 0x2000_0000;
            let file = std::fs::OpenOptions::new().read(true).custom_flags(NO_BUFFERING).open(file).unwrap();
            let len = file.metadata().unwrap().len();
            let mut block = vec![0u8; size + 4096];
            let skew = block.as_ptr().align_offset(4096);
            let n = size / 4096 * 4096;
            let mut at = 0u64;
            timed("read from the disk, one request at a time", &mut || {
                at = (at + 7919 * 4096 * 64) % (len - n as u64) / 4096 * 4096;
                file.seek_read(&mut block[skew..skew + n], at).unwrap();
            });
        }
    }

    #[test]
    #[ignore = "a measurement, not a check"]
    fn allocation_costs() {
        let Some(host) = crate::host::host() else { return };
        let device = &host.device;
        let not_zeroed = D3D12_HEAP_FLAG_CREATE_NOT_ZEROED;
        const N: usize = 1000;
        for size in [128u32, 512, 1024] {
            let (t, b) = (texture(size), buffer(u64::from(size * size) * 4 / 3 + 65536));
            println!("-- BC7 {size}x{size}");
            for threads in [1, 4, 8] {
                timed("texture committed", N, threads, &|_| committed(device, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, &t));
                timed("texture committed not zeroed", N, threads, &|_| committed(device, D3D12_HEAP_TYPE_DEFAULT, not_zeroed, &t));
                let mapped = |flags| {
                    let b = &b;
                    move |_| {
                        let r = committed(device, D3D12_HEAP_TYPE_UPLOAD, flags, b);
                        let mut data = std::ptr::null_mut();
                        // SAFETY: a live upload buffer.
                        unsafe { r.Map(0, None, Some(&mut data)) }.unwrap();
                        // SAFETY: the first byte of the mapping (its first touch).
                        unsafe { data.cast::<u8>().write_volatile(1) };
                        r
                    }
                };
                timed("staging committed and mapped", N, threads, &mapped(D3D12_HEAP_FLAG_NONE));
                timed("staging committed and mapped, not zeroed", N, threads, &mapped(not_zeroed));
            }
            // SAFETY: plain query.
            let info = unsafe { device.GetResourceAllocationInfo(0, &[t]) };
            let per_heap = ((64u64 << 20) / info.SizeInBytes).max(1) as usize;
            let heaps: Vec<ID3D12Heap> = (0..N.div_ceil(per_heap))
                .map(|_| {
                    let desc = D3D12_HEAP_DESC {
                        SizeInBytes: per_heap as u64 * info.SizeInBytes,
                        Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | not_zeroed,
                        ..Default::default()
                    };
                    let mut heap = None;
                    // SAFETY: plain creation.
                    unsafe { device.CreateHeap(&desc, &mut heap) }.expect("heap");
                    heap.unwrap()
                })
                .collect();
            let place = |i: usize| {
                let mut out: Option<ID3D12Resource> = None;
                let at = (i % per_heap) as u64 * info.SizeInBytes;
                // SAFETY: a part of a live heap no other texture has.
                unsafe { device.CreatePlacedResource(&heaps[i / per_heap], at, &t, D3D12_RESOURCE_STATE_COMMON, None, &mut out) }.expect("placed");
                out.unwrap()
            };
            timed("texture placed in 64 MB heaps", N, 1, &place);
            timed("texture placed in 64 MB heaps", N, 8, &place);
        }
    }
}
