//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! A 64 KiB page in a title buffer that holds all of it is in that buffer's
//! copy ([`crate::shadow`]), which every other access uses. A page outside
//! buffers has a page of its own in an **arena**: memory of ours with a copy
//! on the card ([`crate::shadow::by_address`]). The streamed structures lie
//! anywhere between other data (6.6 GB of title memory in 4 MiB windows, run
//! light-1; 8 GB in windows of 256 KiB, light-2), so a page is what is kept.
//!
//! This module keeps what a shader needs to find the pages:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one for all lists, entries only ever go from empty to filled.
//!   The CPU writes it in memory of its own and shaders read a copy of that
//!   on the card, brought up to date as title memory is (over the bus, every
//!   step of every ray waited for it);
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the pages are there from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before one of its dispatches the host makes ready what it can
//! see the dispatch name:
//!
//! - 256 KiB from each root view outside buffers, and where title memory has a
//!   packed ("dehydrated") structure there, all of it: the title ships its
//!   structures packed, and the builder's rehydration stages unpack each in
//!   place once it is streamed in;
//! - 4 MiB from each unordered access view outside buffers (structures built);
//! - the page of each pair of root constants that is a title address and of
//!   each root constant that is one in 256-byte units;
//! - the first page of every structure a list's upload memory names
//!   (a top-level build's instances).
//!
//! **What comes back to title memory**: what the builder stores in title
//! buffers of at most [`BACK`] bytes it names (the sizes the title reads to
//! make a structure smaller). Structures and scratch memory stay on the card
//! (the first whole run of the builder wrote garbage over the title's
//! objects). So an arena's page is the only place a structure is whole, and
//! title memory must not go over it: of a page the title wrote only the
//! dwords that changed since they were last taken go up. A packed structure
//! goes up whole when the builder comes to it, whatever is known of it: the
//! same bytes streamed in again do not differ from those last taken, and the
//! card holds them unpacked (run light-3: a quarter of the structures
//! unpacked a second time, into garbage all over their neighbours).
//!
//! **Stores are bounded**: a structure the builder misreads makes it store
//! anywhere. Pages of title buffers take its stores only once one of its
//! dispatches was given that buffer as an unordered access view (its scratch
//! memory, its results); what else a store reaches is an arena's.
//!
//! A packed structure is gone when title memory has no header of it any more
//! (looked at over [`SWEEP`] frames); the pages it alone has are given back.
//!
//! shortcut: a page held for anything else is kept for the rest of the run.
//! shortcut: pages a list does not name are looked at again in turn, over
//! [`SWEEP`] frames.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_READ_ONLY};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// A directory page.
const PAGE: u64 = 1 << 16;
/// An arena's bytes for builder (`Use::Builds`) workloads (16 MiB = 256 × 64 KiB pages).
const ARENA: u64 = 16 << 20;
/// Number of 64 KiB pages per builder arena (`256`).
const ARENA_PAGES: usize = (ARENA / PAGE) as usize;
/// Initial arena size for read-only (`Use::Reads`) workloads (4 MiB = 64 pages),
/// saving Local VRAM + CPU RAM in RT-Off when only a few pages are touched.
const READ_ARENA: u64 = 4 << 20;
/// Upper arena (`idx >= 1`) occupancy threshold (%) below which live pages are
/// incrementally compacted into lower arenas so the upper arena can be reclaimed.
const COMPACT_MAX_OCCUPANCY_PCT: usize = 35;
/// Minimum frames between incremental upper-arena compaction sweeps in `ready()`.
const COMPACT_INTERVAL_FRAMES: u64 = 60;
/// Shared null UAV buffer size replacing retired empty arena slots in `copies`.
const NULL_COPY_BYTES: u64 = 4096;
/// Frames before unowned, unpinned arena pages are swept when DXR ephemeral release is active.
const STALE_EXTRACTED_FRAMES: u64 = 120;
/// What is compared at a time.
const PIECE: usize = 4096;
/// Pages held, at most (3 GiB).
const MOST: usize = 3 << 14;
/// Pages of title memory with small title buffers in them, at most.
const WINDOWS: usize = 512;
/// The biggest packed structure taken for one.
const PACKED: u64 = 64 << 20;
/// For so many frames what a page holds may still be on its way.
const YOUNG: u64 = 4;
/// Free pages there are, at least, when a list is recorded (1 full 16 MiB arena).
const SPARE: usize = ARENA_PAGES;
/// A packed structure not named for so many frames is another one.
const AGAIN: u64 = 8;
/// From a root view; from one that is written (a structure being built).
const VIEW: u64 = 256 << 10;
const BUILT: u64 = 4 << 20;
/// The biggest title buffer whose builder results come back.
const BACK: u64 = 256 << 10;
/// Pages taken from the faults, of one frame and in all, at most: a ray in a
/// structure that is not right goes anywhere, and 2,048 a frame took all the
/// pages there are in the first seconds (run light-4).
const FAULTS: usize = 128;
const FAULTS_ALL: usize = 2048;
/// Frames over which every page is looked at once.
const SWEEP: usize = 128;

/// What a draw or dispatch does with title memory by address.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Use {
    Reads,
    /// Stores by address, as the structure builder; `back`: its results in
    /// small title buffers come back.
    Builds { back: bool },
}

/// A page of title memory in an arena.
struct Held {
    /// Where in the arena's memory, and the arena's place in `copies`.
    at: usize,
    copy: usize,
    /// The write-watch asking after which it was last taken (0: never).
    taken: u64,
    /// The frame it was last looked at in, and the one it came in.
    looked: u64,
    born: u64,
    /// Packed structures that have bytes in it, and whether it is held for
    /// them alone.
    owners: u32,
    theirs: bool,
    /// In-flight Native DXR `ExtractStaticBlas` submissions currently reading this page.
    pinned: u32,
}

/// One backing arena (`4 MiB` for `Use::Reads`, `16 MiB` for `Use::Builds`).
struct ArenaInfo {
    /// Index in `Memory::copies`.
    copy: usize,
    /// Host CPU virtual address returned by `crate::upload::reserve`.
    base: usize,
    /// Arena byte length (`READ_ARENA` or `ARENA`).
    size: u64,
    /// Count of pages currently mapped in `Memory::pages` pointing into this arena.
    live_pages: usize,
    /// Free page host CPU addresses inside this arena.
    free: Vec<usize>,
    /// True once empty and replaced in `Memory::copies` by the shared 4 KiB null UAV.
    retired: bool,
}

/// Drops a retired arena's `Shadow` (freeing VRAM + staging) and then releases its
/// `VirtualAlloc2` CPU backing once all GPU flights and `bury_lag()` have completed.
struct RetiredArena {
    shadow: Option<Arc<Shadow>>,
    cpu_base: usize,
}

// SAFETY: `Arc<Shadow>` is `Send` and `cpu_base` is owned exclusively by this retirement record.
unsafe impl Send for RetiredArena {}

impl Drop for RetiredArena {
    fn drop(&mut self) {
        if let Some(shadow) = self.shadow.take()
            && let Ok(owned) = Arc::try_unwrap(shadow)
        {
            drop(owned);
            if self.cpu_base != 0 {
                use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
                // SAFETY: `cpu_base` came from `crate::upload::reserve` and no `Shadow`
                // or GPU command list references it anymore.
                unsafe {
                    VirtualFree(self.cpu_base as *mut _, 0, MEM_RELEASE);
                }
            }
        }
    }
}

/// A packed structure the builder came to.
struct Packed {
    /// The frame it last did.
    frame: u64,
    len: u64,
    /// Its first two dwords in title memory.
    header: u64,
}

struct Memory {
    /// The CPU's entries (number, offset), and the copy shaders read.
    directory: (*mut u32, Arc<Shadow>),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`. Retired arena slots are replaced with `null_copy`.
    copies: Vec<Arc<Shadow>>,
    /// Shadow IDs are unique for the lifetime of a backing, including after
    /// the guest reuses its address.
    numbers: HashMap<u64, usize>,
    /// Backing arenas (active and retired).
    arenas: Vec<ArenaInfo>,
    /// Indices in `copies` that have been retired to the shared 4 KiB null UAV.
    retired_copies: std::collections::HashSet<usize>,
    /// Shared 4 KiB null UAV `Shadow` substituting for retired empty arena slots.
    null_copy: Option<Arc<Shadow>>,
    /// Empty arena shadows waiting for `Arc::strong_count == 1` before `host::retire_vram`.
    draining_arenas: Vec<(Arc<Shadow>, usize, u64)>,
    /// Total free pages across all non-retired arenas.
    free_pages: usize,
    /// Whether any `Use::Builds` dispatch has been seen (switches new arenas from 4 MiB to 16 MiB).
    saw_builds: bool,
    /// Last frame on which incremental upper-arena compaction ran.
    last_compact_frame: u64,
    /// By title page.
    pages: HashMap<u64, Held>,
    /// Their numbers in the order they came, and where the sweep is.
    order: Vec<u64>,
    sweep: usize,
    /// Windows of title memory itself ([`Memory::map`]).
    windows: usize,
    /// Packed structures the builder came to: the frame it last did.
    packed: HashMap<u64, Packed>,
    /// Their addresses in the order they came, and where the sweep is.
    packed_order: Vec<u64>,
    packed_sweep: usize,
    /// Copies of title buffers the builder may store in.
    open: std::collections::HashSet<usize>,
    /// Pages taken from faults.
    faulted: usize,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    stats: Stats,
    #[cfg(feature = "diagnostics")]
    dump: dump::Dump,
    #[cfg(feature = "diagnostics")]
    trace: trace::Trace,
}

/// Since the last report.
#[derive(Default)]
struct Stats {
    from: Option<(std::time::Instant, u64)>,
    uses: [u64; 2],
    faults: u64,
    unknown: u64,
    back: u64,
    /// Pages looked at, bytes that changed, packed structures sent.
    looked: u64,
    changed: u64,
    packed: u64,
    packed_bytes: u64,
    gone: u64,
    /// Pages that came for an address in constants, a written view, another
    /// view, a structure named in upload memory.
    came: [u64; 4],
    micros: u64,
}

// SAFETY: the mappings are of memory the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
    /// Last numbered copy this draw prepared. Other recording threads can
    /// subsequently add mappings to the same directory.
    pub(crate) count: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let bytes = u64::from(MEMORY_PAGES) * 8;
        // Zeroed, and watched for writes as the title's memory is.
        let entries = crate::upload::reserve(bytes as usize)?;
        let directory = crate::shadow::by_address(entries as u64, bytes)?;
        crate::host::name(&directory.buffer, || "title memory by address: directory".into());
        let (faults, data) = host.shared_buffer(u64::from(MEMORY_FAULTS) * 4)?;
        crate::host::name(&faults, || "title memory by address: faults".into());
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, MEMORY_FAULTS as usize * 4) };
        // SAFETY: a live buffer.
        let gpu = unsafe { faults.GetGPUVirtualAddress() };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: (entries as *mut u32, directory),
            faults: (faults, data.cast::<u32>(), gpu),
            copies: Vec::new(),
            numbers: HashMap::new(),
            arenas: Vec::new(),
            retired_copies: Default::default(),
            null_copy: None,
            draining_arenas: Vec::new(),
            free_pages: 0,
            saw_builds: false,
            last_compact_frame: 0,
            pages: HashMap::new(),
            order: Vec::new(),
            sweep: 0,
            windows: 0,
            packed: HashMap::new(),
            packed_order: Vec::new(),
            packed_sweep: 0,
            open: Default::default(),
            faulted: 0,
            frame: u64::MAX,
            refused: Default::default(),
            stats: Stats::default(),
            #[cfg(feature = "diagnostics")]
            dump: Default::default(),
            #[cfg(feature = "diagnostics")]
            trace: Default::default(),
        })
    }

    /// The directory's entry of `page`.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES));
        // SAFETY: inside the directory.
        unsafe { self.directory.0.add(page as usize * 2) }
    }

    /// The place in `copies` of the copy that holds `page`, if one does.
    fn copy_of(&self, page: u64) -> Option<usize> {
        if page >= u64::from(MEMORY_PAGES) {
            return None;
        }
        // SAFETY: an entry of the directory.
        ((unsafe { self.entry(page).read() } & !MEMORY_READ_ONLY) as usize).checked_sub(1)
    }

    /// Whether `page` can be given a place: in the directory, not turned down
    /// this frame, and the title's GPU memory. (A shader's pointer, or a
    /// constant taken for an address, may be anything.)
    fn possible(&mut self, page: u64) -> bool {
        // Not the last entry: addresses past the directory read it, and it
        // stays empty.
        if page < 0x1_0000 || page >= u64::from(MEMORY_PAGES) - 1 || self.refused.contains(&page) {
            return false;
        }
        let at = (page * PAGE) as usize;
        let ours = crate::texture::committed(at as *const u8, PAGE as usize)
            && crate::watch::writes(at, at + PAGE as usize, true).is_some();
        if !ours {
            self.refused.insert(page);
            self.stats.unknown += 1;
        }
        ours
    }

    /// Assigns a stable one-based copy number, keeping earlier numbers valid
    /// for previously recorded command lists.
    fn number(&mut self, shadow: Arc<Shadow>) -> Option<usize> {
        let key = shadow.id();
        if let Some(&n) = self.numbers.get(&key) {
            return Some(n);
        }
        // Directory stores a one-based index in the low 31 bits (bit 31 is
        // MEMORY_READ_ONLY, 0 is the null view).
        let next = u32::try_from(self.copies.len()).ok()?.checked_add(1)?;
        if next & MEMORY_READ_ONLY != 0 {
            return None;
        }
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        self.copies.push(shadow);
        let n = self.copies.len() - 1;
        self.numbers.insert(key, n);
        Some(n)
    }

    /// Gives `page` a place, if it has none and the memory is the title's;
    /// whether it has one then.
    fn map(&mut self, page: u64) -> bool {
        if self.copy_of(page).is_some() {
            return true;
        }
        if !self.possible(page) {
            return false;
        }
        let start = page * PAGE;
        if crate::shadow::in_buffer(start, PAGE).is_some() {
            let Some(shadow) = crate::shadow::find(start, PAGE) else {
                return false;
            };
            let fresh = !self.numbers.contains_key(&shadow.id());
            let Some(n) = self.number(shadow.clone()) else {
                return false;
            };
            // Every page the copy holds whole, not only the one asked for.
            // shortcut: a page of a title buffer inside this one that has a
            // copy of its own is given this copy too.
            if fresh {
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    let entry = self.entry(at / PAGE);
                    // SAFETY: an entry of the directory.
                    if unsafe { entry.read() } == 0 {
                        close(entry, n, at - first);
                    }
                    at += PAGE;
                }
            }
            // SAFETY: as above.
            if unsafe { self.entry(page).read() } == 0 {
                close(self.entry(page), n, start - shadow.span().0);
            }
            return true;
        }
        // Small title buffers in it: what the builder stores in them comes
        // back, so title memory itself.
        if self.windows < WINDOWS && crate::shadow::small_buffer_in(start, PAGE, BACK) {
            if let Some(shadow) = crate::shadow::by_address(start, PAGE)
                && let Some(n) = self.number(shadow)
            {
                self.windows += 1;
                fill(self.entry(page), n, 0);
                return true;
            }
            return false;
        }
        if self.pages.len() >= MOST {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(MOST, "title memory by address: no more pages are taken; others read zeros"));
            return false;
        }
        if let Some((copy, at)) = self.pop_free_page() {
            self.hold(page, at, copy);
            return true;
        }
        let bytes = next_arena_bytes(self.saw_builds);
        if !self.another(bytes) {
            return false;
        }
        let Some((copy, at)) = self.pop_free_page() else {
            return false;
        };
        self.hold(page, at, copy);
        true
    }

    /// Pops a free page from the lowest-indexed active arena so upper arenas drain
    /// and can be reclaimed when empty.
    fn pop_free_page(&mut self) -> Option<(usize, usize)> {
        for arena in &mut self.arenas {
            if !arena.retired
                && let Some(at) = arena.free.pop()
            {
                arena.live_pages += 1;
                self.free_pages = self.free_pages.saturating_sub(1);
                return Some((arena.copy, at));
            }
        }
        None
    }

    /// Returns a page to its owning arena and clears its directory entry.
    fn release_page(&mut self, page: u64) -> bool {
        let Some(held) = self.pages.remove(&page) else {
            return false;
        };
        if let Some(arena) = self.arenas.iter_mut().find(|a| a.copy == held.copy) {
            arena.live_pages = arena.live_pages.saturating_sub(1);
            if !arena.retired {
                arena.free.push(held.at);
                self.free_pages += 1;
            }
        }
        // SAFETY: an entry of the directory, under the mutex.
        unsafe { self.entry(page).write_volatile(0) };
        crate::watch::wrote(self.entry(page) as usize, 8);
        true
    }

    /// Lazily allocates the single shared 4 KiB null UAV shadow used to replace
    /// retired empty arenas in `copies` so descriptor slots remain valid.
    fn ensure_null_copy(&mut self) -> Option<Arc<Shadow>> {
        if let Some(null) = &self.null_copy {
            return Some(null.clone());
        }
        let at = crate::upload::reserve(NULL_COPY_BYTES as usize)?;
        let Some(shadow) = crate::shadow::by_address(at as u64, NULL_COPY_BYTES) else {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: freshly reserved and unused.
            unsafe {
                VirtualFree(at as *mut _, 0, MEM_RELEASE);
            }
            return None;
        };
        crate::host::name(&shadow.buffer, || "title memory by address: retired arena null uav".into());
        self.null_copy = Some(shadow.clone());
        Some(shadow)
    }

    /// Retires empty arenas (keeping at most `keep_spare` warm empty arenas),
    /// replacing their `copies[n]` entry with the shared 4 KiB null UAV.
    fn reclaim_empty_arenas(&mut self, keep_spare: usize) -> usize {
        let empty_indices: Vec<usize> = self
            .arenas
            .iter()
            .enumerate()
            .filter_map(|(idx, a)| (!a.retired && a.live_pages == 0).then_some(idx))
            .collect();
        if empty_indices.len() <= keep_spare {
            self.drain_retired_arenas();
            return 0;
        }
        let Some(null_shadow) = self.ensure_null_copy() else {
            return 0;
        };
        let mut retired = 0usize;
        for &idx in &empty_indices[keep_spare..] {
            let copy = self.arenas[idx].copy;
            let base = self.arenas[idx].base;
            let size = self.arenas[idx].size;
            let freed_slots = self.arenas[idx].free.len();
            self.arenas[idx].free.clear();
            self.arenas[idx].retired = true;
            self.free_pages = self.free_pages.saturating_sub(freed_slots);
            self.retired_copies.insert(copy);
            if let Some(slot) = self.copies.get_mut(copy) {
                let old = std::mem::replace(slot, null_shadow.clone());
                self.numbers.remove(&old.id());
                self.draining_arenas.push((old, base, size));
            }
            retired += 1;
        }
        self.drain_retired_arenas();
        retired
    }

    /// Hands any draining empty arena whose `Arc<Shadow>` is no longer held by any
    /// command list over to `host::retire_vram` for fence-safe async VRAM + RAM release.
    fn drain_retired_arenas(&mut self) {
        let mut i = 0;
        while i < self.draining_arenas.len() {
            if Arc::strong_count(&self.draining_arenas[i].0) == 1 {
                let (shadow, cpu_base, size) = self.draining_arenas.swap_remove(i);
                crate::host::retire_vram(
                    RetiredArena {
                        shadow: Some(shadow),
                        cpu_base,
                    },
                    size,
                );
            } else {
                i += 1;
            }
        }
    }

    /// Releases rehydrated BVH4 arena pages covering `[dest_va, dest_va + size)` and
    /// reclaims completely empty arenas (keeping at most `usize::from(self.saw_builds)` spare).
    fn release_extracted_range(&mut self, dest_va: u64, size: u64) -> usize {
        if dest_va == 0 || size == 0 {
            return 0;
        }
        let mut release_len = size;
        if let Some(packed) = self.packed.remove(&dest_va) {
            release_len = release_len.max(packed.len);
            let p_first = dest_va / PAGE;
            let p_last = (dest_va.saturating_add(packed.len) - 1) / PAGE;
            for page in p_first..=p_last {
                if let Some(held) = self.pages.get_mut(&page) {
                    held.owners = held.owners.saturating_sub(1);
                }
            }
            self.stats.gone += 1;
        }
        let first = dest_va / PAGE;
        let last = (dest_va.saturating_add(release_len) - 1) / PAGE;
        let mut released_any = false;
        for page in first..=last {
            let can_release = match self.pages.get_mut(&page) {
                Some(held) => {
                    held.pinned = held.pinned.saturating_sub(1);
                    held.owners == 0 && held.pinned == 0
                }
                None => false,
            };
            if can_release && self.release_page(page) {
                released_any = true;
            }
        }
        if released_any {
            self.reclaim_empty_arenas(usize::from(self.saw_builds))
        } else {
            self.drain_retired_arenas();
            0
        }
    }

    /// Incrementally compacts sparse upper arenas (`idx >= 1`, occupancy `< 35%`) into
    /// free slots in strictly lower active arenas (`0..idx`) when no page in the upper
    /// arena is pinned and lower arenas can absorb every live page.
    ///
    /// Copies exact 64 KiB page contents in both CPU backing (`upload::reserve` + `resend`)
    /// and GPU backing (via `CopyBufferRegion` on `list` when provided), atomically updates
    /// directory entries, and calls [`Self::reclaim_empty_arenas`] so emptied upper arenas
    /// are retired to the shared 4 KiB null UAV and freed via `host::retire_vram`.
    fn compact_sparse_arenas(&mut self, list: Option<&HostList>) -> usize {
        if self.arenas.len() < 2 {
            return 0;
        }
        let mut total_moved = 0usize;
        let mut emptied_any = false;
        for idx in (1..self.arenas.len()).rev() {
            let (src_copy, src_base, src_size, live_pages, retired) = {
                let a = &self.arenas[idx];
                (a.copy, a.base, a.size, a.live_pages, a.retired)
            };
            if retired || live_pages == 0 {
                continue;
            }
            let cap = (src_size / PAGE) as usize;
            if cap == 0 || live_pages.saturating_mul(100) >= cap.saturating_mul(COMPACT_MAX_OCCUPANCY_PCT) {
                continue;
            }
            let lower_free: usize = self.arenas[..idx]
                .iter()
                .filter(|a| !a.retired)
                .map(|a| a.free.len())
                .sum();
            if lower_free < live_pages {
                continue;
            }
            let mut pages_to_move: Vec<u64> = Vec::with_capacity(live_pages);
            let mut has_pinned = false;
            for (&page, held) in &self.pages {
                if held.copy == src_copy {
                    if held.pinned > 0 {
                        has_pinned = true;
                        break;
                    }
                    pages_to_move.push(page);
                }
            }
            if has_pinned || pages_to_move.is_empty() || pages_to_move.len() > lower_free {
                continue;
            }
            // Move every live page from arena `idx` into strictly lower active arenas `0..idx`.
            let mut gpu_copies: Vec<(u64, usize, u64)> = Vec::with_capacity(pages_to_move.len());
            let mut dst_copies_touched: Vec<usize> = Vec::new();
            for page in pages_to_move {
                let Some((dst_copy, dst_at, dst_base)) = self.pop_free_page_below(idx) else {
                    break;
                };
                let Some(held) = self.pages.get_mut(&page) else {
                    // Should not happen; return popped slot.
                    if let Some(a) = self.arenas[..idx].iter_mut().find(|a| a.copy == dst_copy) {
                        a.live_pages = a.live_pages.saturating_sub(1);
                        a.free.push(dst_at);
                        self.free_pages += 1;
                    }
                    continue;
                };
                let src_at = held.at;
                let src_off = (src_at - src_base) as u64;
                let dst_off = (dst_at - dst_base) as u64;
                // SAFETY: both `src_at` and `dst_at` are valid, disjoint 64 KiB arena pages
                // owned by `Memory` under the `MEMORY` mutex.
                unsafe {
                    std::ptr::copy_nonoverlapping(src_at as *const u8, dst_at as *mut u8, PAGE as usize);
                }
                crate::watch::wrote(dst_at, PAGE as usize);
                self.copies[dst_copy].resend(dst_off as usize, dst_off as usize + PAGE as usize);
                held.copy = dst_copy;
                held.at = dst_at;
                self.arenas[idx].live_pages = self.arenas[idx].live_pages.saturating_sub(1);
                self.arenas[idx].free.push(src_at);
                self.free_pages += 1;
                // Atomically update directory entry (offset first, then copy number) so shaders
                // see the relocated page consistently.
                let dir_off = dst_at as u64 - self.copies[dst_copy].span().0;
                let entry = self.entry(page);
                fill(entry, dst_copy, dir_off);
                crate::watch::wrote(entry as usize, 8);
                gpu_copies.push((src_off, dst_copy, dst_off));
                if !dst_copies_touched.contains(&dst_copy) {
                    dst_copies_touched.push(dst_copy);
                }
                total_moved += 1;
            }
            if let Some(list) = list
                && !gpu_copies.is_empty()
            {
                let src_shadow = self.copies[src_copy].clone();
                let mut pre_barriers = Vec::with_capacity(1 + dst_copies_touched.len());
                let mut post_barriers = Vec::with_capacity(1 + dst_copies_touched.len());
                pre_barriers.push(arena_transition(
                    &src_shadow.buffer,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                ));
                post_barriers.push(arena_transition(
                    &src_shadow.buffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
                list.keep_shadow(src_shadow.clone());
                for &dc in &dst_copies_touched {
                    let dst_shadow = self.copies[dc].clone();
                    pre_barriers.push(arena_transition(
                        &dst_shadow.buffer,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    ));
                    post_barriers.push(arena_transition(
                        &dst_shadow.buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    ));
                    list.keep_shadow(dst_shadow);
                }
                // SAFETY: open command list; src and dst are strictly disjoint arenas kept alive
                // on `list` and start/end in `COMMON`.
                unsafe {
                    list.list.ResourceBarrier(&pre_barriers);
                    for &(src_off, dc, dst_off) in &gpu_copies {
                        list.list.CopyBufferRegion(
                            &self.copies[dc].buffer,
                            dst_off,
                            &src_shadow.buffer,
                            src_off,
                            PAGE,
                        );
                    }
                    list.list.ResourceBarrier(&post_barriers);
                }
            }
            if self.arenas[idx].live_pages == 0 {
                emptied_any = true;
            }
        }
        if emptied_any {
            self.reclaim_empty_arenas(0);
        } else {
            self.drain_retired_arenas();
        }
        total_moved
    }

    /// Pops a free page from the lowest-indexed non-retired arena strictly below `upper_idx`.
    fn pop_free_page_below(&mut self, upper_idx: usize) -> Option<(usize, usize, usize)> {
        for arena in &mut self.arenas[..upper_idx] {
            if !arena.retired
                && let Some(at) = arena.free.pop()
            {
                arena.live_pages += 1;
                self.free_pages = self.free_pages.saturating_sub(1);
                return Some((arena.copy, at, arena.base));
            }
        }
        None
    }

    /// Another arena's pages (`READ_ARENA` = 4 MiB for read-only workloads, `ARENA` = 16 MiB for builds).
    fn another(&mut self, bytes: u64) -> bool {
        let Some(at) = crate::upload::reserve(bytes as usize) else {
            return false;
        };
        let Some(n) = crate::shadow::by_address(at as u64, bytes).and_then(|s| self.number(s)) else {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: freshly reserved and failed to create shadow.
            unsafe {
                VirtualFree(at as *mut _, 0, MEM_RELEASE);
            }
            return false;
        };
        let page_count = (bytes / PAGE) as usize;
        // Lowest address popped first.
        let free: Vec<usize> = (0..page_count).rev().map(|i| at + i * PAGE as usize).collect();
        self.free_pages += free.len();
        self.arenas.push(ArenaInfo {
            copy: n,
            base: at,
            size: bytes,
            live_pages: 0,
            free,
            retired: false,
        });
        true
    }

    /// `page` is held at `at` in the arena that is `copy`.
    fn hold(&mut self, page: u64, at: usize, copy: usize) {
        self.pages.insert(
            page,
            Held {
                at,
                copy,
                taken: 0,
                looked: u64::MAX,
                born: self.frame,
                owners: 0,
                theirs: false,
                pinned: 0,
            },
        );
        self.order.push(page);
        fill(self.entry(page), copy, at as u64 - self.copies[copy].span().0);
    }

    /// The packed structure at `address` is gone: the pages it alone had are
    /// given back.
    fn gone(&mut self, address: u64) {
        let Some(packed) = self.packed.remove(&address) else { return };
        for page in address / PAGE..=(address + packed.len - 1) / PAGE {
            let release = match self.pages.get_mut(&page) {
                Some(held) => {
                    held.owners = held.owners.saturating_sub(1);
                    held.owners == 0 && held.theirs && held.pinned == 0
                }
                None => false,
            };
            if release {
                self.release_page(page);
            }
        }
        self.stats.gone += 1;
    }

    /// Whether the packed structure at `address` is still in title memory.
    fn there(&self, address: u64) -> bool {
        let mut header = [0u8; 8];
        crate::texture::committed(address as *const u8, 8) && {
            crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
            self.packed.get(&address).is_some_and(|p| p.header == u64::from_le_bytes(header))
        }
    }

    /// Looks at some of the pages and packed structures again.
    fn look_again(&mut self) {
        let ephemeral = ephemeral_release_allowed();
        let mut released_any = false;
        for _ in 0..self.order.len().div_ceil(SWEEP) {
            if self.order.is_empty() {
                break;
            }
            self.sweep = (self.sweep + 1) % self.order.len();
            let page = self.order[self.sweep];
            if let Some(held) = self.pages.get(&page) {
                if ephemeral
                    && held.owners == 0
                    && held.pinned == 0
                    && self.frame.saturating_sub(held.born) > STALE_EXTRACTED_FRAMES
                {
                    self.release_page(page);
                    self.order.swap_remove(self.sweep);
                    released_any = true;
                } else {
                    self.bring(page, page + 1);
                }
            } else {
                self.order.swap_remove(self.sweep);
            }
        }
        for _ in 0..self.packed_order.len().div_ceil(SWEEP) {
            if self.packed_order.is_empty() {
                break;
            }
            self.packed_sweep = (self.packed_sweep + 1) % self.packed_order.len();
            let address = self.packed_order[self.packed_sweep];
            if !self.packed.contains_key(&address) {
                self.packed_order.swap_remove(self.packed_sweep);
            } else if !self.there(address) {
                self.gone(address);
                released_any = true;
            }
        }
        if released_any && ephemeral {
            self.reclaim_empty_arenas(usize::from(self.saw_builds));
        } else {
            self.drain_retired_arenas();
        }
    }

    /// A dispatch of the builder was given `address` to write: if it is in a
    /// title buffer's copy, stores by address reach that from here on.
    fn opens(&mut self, address: u64) {
        let Some(n) = self.map(address / PAGE).then(|| self.copy_of(address / PAGE)).flatten() else {
            return;
        };
        if self.pages.contains_key(&(address / PAGE)) || !self.open.insert(n) {
            return;
        }
        // (Its pages hold what the CPU writes beside the builder's scratch.)
        self.copies[n].keep_stores();
        let (first, len) = self.copies[n].span();
        tracing::info!(
            start = format_args!("{first:#x}"),
            len,
            "title memory by address: the structure builder may store in a title buffer"
        );
        for page in first.div_ceil(PAGE)..(first + len) / PAGE {
            if self.copy_of(page) == Some(n) {
                fill(self.entry(page), n, page * PAGE - first);
            }
        }
    }

    /// [`Memory::map`] of the pages of `[address, address + len)`, up to the
    /// first that is not the title's, and what the title wrote there since
    /// they were taken goes into the arenas.
    fn ready(&mut self, address: u64, len: u64) {
        let first = address / PAGE;
        let mut end = first;
        while end * PAGE < address + len && self.map(end) {
            end += 1;
        }
        self.bring(first, end);
    }

    /// Takes the pages `[first, end)` that are in arenas from title memory
    /// where it wrote them.
    fn bring(&mut self, first: u64, end: u64) {
        // Once a frame: the builder names the same pages list after list.
        let frame = self.frame;
        if !(first..end).any(|page| self.pages.get(&page).is_some_and(|h| h.looked != frame)) {
            return;
        }
        let started = std::time::Instant::now();
        let answer = crate::watch::writes((first * PAGE) as usize, (end * PAGE) as usize, true);
        for page in first..end {
            let Some(held) = self.pages.get_mut(&page) else {
                continue;
            };
            if held.looked == frame {
                continue;
            }
            held.looked = frame;
            self.stats.looked += 1;
            let wrote = answer
                .as_ref()
                .map(|(_, w)| w.binary_search_by_key(&(page as usize), |w| w.0).map_or(0, |i| w[i].1));
            let now = answer.as_ref().map_or(1, |a| a.0);
            let stale = held.taken == 0 || wrote.is_none_or(|w| w > held.taken);
            if !stale {
                continue;
            }
            let never = std::mem::replace(&mut held.taken, now.max(1)) == 0;
            let shadow = &self.copies[held.copy];
            let base = held.at - shadow.span().0 as usize;
            let mut title = [0u8; PIECE];
            for piece in (0..PAGE as usize).step_by(PIECE) {
                let from = (page * PAGE) as usize + piece;
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, PIECE, false);
                // SAFETY: a page of an arena of ours, under the mutex.
                let ours = unsafe { std::slice::from_raw_parts_mut((held.at + piece) as *mut u8, PIECE) };
                self.stats.changed += merge(&title, ours, never) as u64 * 4;
            }
            if never {
                shadow.resend(base, base + PAGE as usize);
            }
        }
        self.stats.micros += started.elapsed().as_micros() as u64;
    }

    /// The bytes of a packed structure title memory has at `address` by now,
    /// and its first two dwords.
    fn packed_at(&self, address: u64) -> Option<(u64, u64)> {
        if address & 0xFF != 0 {
            return None;
        }
        let mut header = [0u8; 8];
        crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
        let header = u64::from_le_bytes(header);
        Some((packed_len(header)?, header))
    }

    /// At the submission of a list whose builder dispatch was given `address`
    /// outside buffers, in the order of its copies: what title memory has
    /// there now goes up. When the dispatch is recorded it may not be there
    /// yet: the title's queue waits for its streaming. (Nothing is asked of
    /// the write watch here: the submission's asking stands.)
    fn land(&mut self, address: u64) {
        let page = address / PAGE;
        if !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize) {
            return;
        }
        let Some((len, header)) = self.packed_at(address) else {
            // Anything else: what came since its page was first taken.
            for page in page..=(address + VIEW - 1) / PAGE {
                if self.pages.get(&page).is_some_and(|h| h.born + YOUNG >= self.frame) {
                    self.take(page, 0, PAGE as usize, false);
                }
            }
            return;
        };
        if let Some(known) = self.packed.get_mut(&address) {
            let soon = known.frame + AGAIN >= self.frame;
            known.frame = self.frame;
            if known.header == header && soon {
                return;
            }
            if known.header != header {
                self.gone(address);
            }
        }
        let last = (address + len - 1) / PAGE;
        if !self.packed.contains_key(&address) {
            self.packed.insert(address, Packed { frame: self.frame, len, header });
            self.packed_order.push(address);
            for page in page..=last {
                let new = self.copy_of(page).is_none();
                if new && !self.place(page) {
                    continue;
                }
                if let Some(held) = self.pages.get_mut(&page) {
                    held.owners += 1;
                    held.theirs |= new;
                }
            }
            self.stats.packed_bytes += len;
        }
        for page in page..=last {
            let from = address.max(page * PAGE) - page * PAGE;
            let to = (address + len).min((page + 1) * PAGE) - page * PAGE;
            self.take(page, from as usize, to as usize, true);
        }
        self.stats.packed += 1;
    }

    /// A place in an arena for a page of committed title memory, at a
    /// submission: only one there is already (the list brings up the arenas
    /// it knew when it was recorded).
    fn place(&mut self, page: u64) -> bool {
        if page < 0x1_0000
            || page >= u64::from(MEMORY_PAGES) - 1
            || !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize)
            || crate::shadow::in_buffer(page * PAGE, PAGE).is_some()
        {
            return false;
        }
        let Some((copy, at)) = self.pop_free_page() else {
            return false;
        };
        self.hold(page, at, copy);
        crate::watch::wrote(self.entry(page) as usize, 8);
        self.take(page, 0, PAGE as usize, true);
        true
    }

    /// Bytes `from..to` of `page` from title memory into its arena's page, at
    /// a submission; `whole`: sent whether they differ from those last taken
    /// or not.
    fn take(&mut self, page: u64, from: usize, to: usize, whole: bool) {
        let Some(held) = self.pages.get(&page) else { return };
        let shadow = &self.copies[held.copy];
        let base = held.at - shadow.span().0 as usize;
        let mut title = [0u8; PIECE];
        let mut at = from;
        while at < to {
            let n = (PIECE - at % PIECE).min(to - at);
            crate::host::copy_mapped(title.as_mut_ptr(), (page as usize * PAGE as usize + at) as *const u8, n, false);
            // SAFETY: a page of an arena of ours, under the mutex.
            let ours = unsafe { std::slice::from_raw_parts_mut((held.at + at) as *mut u8, n) };
            if *ours != title[..n] {
                ours.copy_from_slice(&title[..n]);
                crate::watch::wrote(held.at + at, n);
            }
            at += n;
        }
        if whole {
            shadow.resend(base + from, base + to);
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(u64::from(page));
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if pages.is_empty() {
            return;
        }
        pages.sort_unstable();
        let mut made = 0;
        for &page in &pages {
            if made < FAULTS && self.faulted < FAULTS_ALL && self.copy_of(page).is_none() && self.map(page) {
                made += 1;
                self.faulted += 1;
                self.bring(page, page + 1);
            }
        }
        self.stats.faults += pages.len() as u64;
        // Each is an access that read 0 or was dropped.
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let said = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if said < 40 || said.is_multiple_of(100) {
            let some: Vec<String> = pages
                .iter()
                .step_by(pages.len().div_ceil(12))
                .map(|&p| format!("{:#x}", p * PAGE))
                .collect();
            tracing::warn!(
                frame = self.frame,
                pages = pages.len(),
                made,
                held = self.pages.len(),
                ?some,
                "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); those that are the title's are there now"
            );
        }
    }

    /// The structures named in a list's upload memory around `address`
    /// (instances of a top-level build: a structure's address is their last
    /// 8 bytes): the first page of each.
    fn named_in_upload(&mut self, address: u64) {
        let Some((from, len)) = crate::upload::block(address as usize) else {
            return;
        };
        // SAFETY: the block is handed out and stays mapped; the title wrote
        // it before it recorded the dispatch.
        let words = unsafe { std::slice::from_raw_parts(from as *const u64, len / 8) };
        let mut seen = std::collections::HashSet::new();
        for &word in words {
            if word & 0xFF == 0 && (1 << 32..1 << 37).contains(&word) && seen.insert(word / PAGE) && self.map(word / PAGE) {
                let page = word / PAGE;
                self.bring(page, page + 1);
            }
        }
    }

    /// A small title buffer whose results come back, if its pages are
    /// windows of title memory itself (or a copy of the buffer's own).
    fn small(&mut self, list: &HostList, start: u64, len: u64) {
        let mut at = start;
        while at < start + len {
            let page = at / PAGE;
            let n = ((page + 1) * PAGE).min(start + len) - at;
            self.map(page);
            if let Some(k) = self.copy_of(page)
                && !self.pages.contains_key(&page)
            {
                self.copies[k].refresh(list, at, n);
                self.copies[k].gpu_writes(list, at, n);
                self.stats.back += 1;
            }
            at += n;
        }
    }

    /// Every addressable copy is brought up to date when `list` is submitted,
    /// after what it deferred so far (once until a copy of the title's intervenes).
    fn bring_up(&self, list: &HostList, count: usize) {
        let (start, len) = self.directory.1.span();
        self.directory.1.refresh(list, start, len);
        for (n, shadow) in self.copies[..count.min(self.copies.len())].iter().enumerate() {
            if self.retired_copies.contains(&n) {
                continue;
            }
            let (start, len) = shadow.span();
            shadow.refresh(list, start, len);
        }
    }

    /// Every ten seconds or so, what was used.
    fn report(&mut self) {
        let now = std::time::Instant::now();
        let (since, frame) = *self.stats.from.get_or_insert((now, self.frame));
        if now.duration_since(since).as_secs_f64() < 10.0 {
            return;
        }
        let frames = self.frame.wrapping_sub(frame).max(1) as f64;
        let s = std::mem::take(&mut self.stats);
        let active_arenas = self.arenas.iter().filter(|a| !a.retired).count();
        tracing::info!(
            "title memory by address: {} pages ({} MB) in {} arenas, {} copies in all; per frame {:.0} dispatches that read, {:.0} that build, \
             {:.1} pages that had no copy, {:.1} pages turned down, {:.1} results brought back; {:.0} pages looked at, {:.1} KB changed, \
             {:.1} packed structures ({} there), {:.2} ms; {} windows; since the last line {} KB of packed structures came and {} of them went, \
             and pages for constants {}, views written {}, others {}, upload memory {}, faults in all {}",
            self.pages.len(),
            (self.pages.len() as u64 * PAGE) >> 20,
            active_arenas,
            self.copies.len(),
            s.uses[0] as f64 / frames,
            s.uses[1] as f64 / frames,
            s.faults as f64 / frames,
            s.unknown as f64 / frames,
            s.back as f64 / frames,
            s.looked as f64 / frames,
            s.changed as f64 / 1024.0 / frames,
            s.packed as f64 / frames,
            self.packed.len(),
            s.micros as f64 / 1000.0 / frames,
            self.windows,
            s.packed_bytes >> 10,
            s.gone,
            s.came[0],
            s.came[1],
            s.came[2],
            s.came[3],
            self.faulted
        );
        self.stats.from = Some((now, self.frame));
    }
}

/// A list with this key ([`HostList::used`]) asks the builder for sizes.
const SIZES: usize = usize::MAX - (9 << 33);

/// Whether a submission that asks for the sizes of structures is waited for,
/// and landed, before the title goes on (`VICTORY_BVH_SIZES_WAIT=1`; off).
///
/// Off: it does not help. In run light-15, with it, 33 of 154 copies had the
/// slot's old size still (all of them recorded one frame after the size was
/// asked for) against 53 of 230 in light-11, 48 of 275 dynamic structures
/// were good, and the title asks in every frame, so every frame waits (109 ms
/// against 90). The title reads the slot before our submission of the asking
/// list is through, or reads it for another purpose; not known.
///
/// The title asks for a new structure's compacted size (a program of one
/// thread stores it into a small buffer), reads it a frame or two later
/// without looking whether it is there, takes memory of that size and copies
/// the structure into it. With submissions that land up to two frames late it
/// read what the slot held before, some other structure's size, and the copy
/// wrote over its neighbours: of 230 such copies of run light-11, 99 were
/// asked for two frames earlier and landed in that frame, none of those with
/// the right size.
fn sizes_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_SIZES_WAIT").is_some_and(|v| v == "1"))
}

/// Whether a dispatch of the builder is its program of a structure's size:
/// one thread, the structure's address and what to store (1: the compacted
/// size) as constants, a small title buffer to store into.
fn asks_size(written: &[u64], constants: &[u32], threads: Option<[u32; 3]>) -> bool {
    threads == Some([1, 1, 1])
        && matches!(*constants, [_, hi, 1..=3, ..] if (1..0x20).contains(&hi))
        && written.first().is_some_and(|&at| crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK))
}

/// Whether one of `lists` asks for sizes.
pub(crate) fn asked_sizes<'a>(lists: impl Iterator<Item = &'a HostList>) -> bool {
    let mut lists = lists;
    let asked = sizes_wait() && lists.any(|l| l.used(SIZES));
    if asked {
        static TIMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n.is_power_of_two() || n.is_multiple_of(100) {
            tracing::info!(n, frame = crate::frame::current(), "submissions that ask for structures' sizes, waited for");
        }
    }
    asked
}

/// [`fill`] of a page stores by address do not reach.
fn close(entry: *mut u32, n: usize, offset: u64) {
    fill(entry, n | MEMORY_READ_ONLY as usize, offset);
}

/// The offset first: a shader that sees the number sees it too.
fn fill(entry: *mut u32, n: usize, offset: u64) {
    // SAFETY: an entry of the directory, under the mutex.
    unsafe {
        entry.add(1).write_volatile(offset as u32);
        entry.write_volatile(n as u32 + 1);
    }
}

/// The bytes of a packed structure whose first two dwords these are: the
/// version (6) with the marks of a bottom level and of one that is packed,
/// and its size unpacked in 64-byte nodes.
fn packed_len(header: u64) -> Option<u64> {
    let (version, nodes) = (header as u32, header >> 32);
    let marks = version >> 16;
    // The header counts its nodes of 64 bytes without the last one: the
    // builder's own program of a structure's size stores one more.
    let len = (nodes + 1) * 64;
    (version & 0xFFFF == 6 && marks & 0x8100 == 0x8100 && nodes > 0 && len <= PACKED).then_some(len)
}

/// Puts what `title` holds into `ours` (what was last taken): the dwords that
/// differ. How many.
fn merge(title: &[u8; PIECE], ours: &mut [u8], never: bool) -> usize {
    if never {
        ours.copy_from_slice(title);
        return PIECE / 4;
    }
    if ours == title {
        return 0;
    }
    let mut changed = 0;
    for (t, o) in title.chunks_exact(4).zip(ours.chunks_exact_mut(4)) {
        if t != o {
            o.copy_from_slice(t);
            changed += 1;
        }
    }
    changed
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name is made ready (`addresses`: its root views, of them
/// `written` those it writes; `constants`: its root constants, see the
/// module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted. None if the host cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    written: &[u64],
    constants: &[u32],
    use_: Use,
    threads: Option<[u32; 3]>,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
        m.look_again();
        if m.frame.saturating_sub(m.last_compact_frame) >= COMPACT_INTERVAL_FRAMES {
            m.last_compact_frame = frame;
            if m.compact_sparse_arenas(Some(list)) > 0 {
                let (d_start, d_len) = m.directory.1.span();
                m.directory.1.refresh(list, d_start, d_len);
            }
        }
        m.report();
        #[cfg(feature = "diagnostics")]
        m.trace.look(frame);
    }
    #[cfg(feature = "diagnostics")]
    if use_ != Use::Reads {
        m.trace.dispatch(frame, list, addresses, written, constants, threads);
    }
    let builds = use_ != Use::Reads;
    if builds {
        m.saw_builds = true;
    }
    if builds && sizes_wait() && asks_size(written, constants, threads) {
        list.first_use(SIZES);
    }
    while builds && m.free_pages < SPARE && m.pages.len() + m.free_pages < MOST && m.another(ARENA) {}
    m.stats.uses[usize::from(builds)] += 1;
    // Once per list and page named: the builder dispatches hundreds of times
    // a frame with the same arguments.
    const READ: usize = usize::MAX - 0x6d;
    let named = |kind: usize, at: u64| list.first_use(usize::MAX - (kind << 26) - (at / PAGE) as usize);
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units (the builder's alone). A constant that only
    // looks like one names memory nothing reads; one that is not the title's
    // memory is refused.
    let held = |m: &Memory| m.pages.len() as u64;
    let before = held(m);
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0x1F).contains(&high)
            && c >= 0x1_0000
        {
            let at = u64::from(c) | u64::from(high) << 32;
            if named(1, at) {
                m.ready(at, 1);
            }
        }
        if builds && (0x0100_0000..0x2000_0000).contains(&c) {
            let at = u64::from(c) << 8;
            if named(1, at) {
                m.ready(at, 1);
            }
        }
    }
    let mut came = [held(m) - before, 0, 0, 0];
    for &at in addresses.iter().filter(|&&a| a >= 1 << 32) {
        // (Several structures to a page.)
        let mut packed = false;
        if builds
            && list.first_use(usize::MAX - (3 << 33) - (at >> 8) as usize)
            && crate::shadow::in_buffer(at, 1).is_none()
            && m.possible(at / PAGE)
        {
            // What it needs, as far as it is known now; the rest when the
            // list is submitted ([`Memory::land`]).
            if let Some((len, _)) = m.packed_at(at) {
                packed = true;
                for page in at / PAGE..=(at + len - 1) / PAGE {
                    let new = m.copy_of(page).is_none();
                    if m.map(page)
                        && new
                        && let Some(held) = m.pages.get_mut(&page)
                    {
                        held.theirs = true;
                    }
                }
                m.bring(at / PAGE, (at + len - 1) / PAGE + 1);
            }
            list.defer_host(move || {
                if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                    m.land(at);
                }
            });
            m.bring_up(list, m.copies.len());
        }
        if builds && written.contains(&at) && list.first_use(usize::MAX - (4 << 33) - (at >> 2 & 0x7FFF_FFFF) as usize) {
            if let Some((start, len)) = crate::shadow::in_buffer(at, 1)
                && len <= BACK
                && use_ == (Use::Builds { back: true })
                && list.first_use(usize::MAX - (5 << 33) - (start >> 2 & 0x7FFF_FFFF) as usize)
            {
                m.small(list, start, len);
            }
            m.opens(at);
            if !packed && crate::shadow::in_buffer(at, 1).is_none() {
                #[cfg(feature = "diagnostics")]
                {
                    let built = m.dump.built.entry(at).or_insert((frame, frame, 0));
                    (built.1, built.2) = (frame, built.2 + 1);
                }
                let before = held(m);
                m.ready(at, BUILT);
                came[1] += held(m).saturating_sub(before);
            }
        }
        if packed || !named(2, at) {
            continue;
        }
        let before = held(m);
        match crate::shadow::in_buffer(at, 1) {
            Some(_) => {
                m.ready(at, 1);
                m.named_in_upload(at);
                came[3] += held(m).saturating_sub(before);
            }
            None => {
                m.ready(at, if builds { VIEW } else { PAGE });
                came[2] += held(m).saturating_sub(before);
            }
        }
    }
    for (sum, n) in m.stats.came.iter_mut().zip(came) {
        *sum += n;
    }
    let (base, count) = heap.raw_page_views(&m.copies)?;
    // Every copy by every list that has come to by-address memory, whoever
    // numbered it: an arena another list made while this one was being
    // recorded was brought up to date by that list's submission alone, which
    // may run after this one's (structures unpacked from zeros, and then
    // written over by the arena's first upload: in frames that stream much in
    // half of them, run light-8).
    let first_read = list.first_use(READ);
    if list.first_use(usize::MAX - (6 << 33) - count as usize) {
        m.bring_up(list, count as usize);
    }
    if first_read {
        list.keep(&m.faults.0);
        #[cfg(feature = "diagnostics")]
        m.dump(list, addresses);
    }
    #[cfg(feature = "diagnostics")]
    if use_ == Use::Reads {
        m.dump.walked(addresses);
    }
    let directory = &m.directory.1;
    Some(Ready {
        directory: directory.gpu(directory.span().0),
        faults: m.faults.2,
        base,
        count,
    })
}

/// Returns the next arena byte size: `4 MiB` for `Use::Reads`-only workloads (RT-Off),
/// `16 MiB` (`256` × 64 KiB pages) once `Use::Builds` has been seen (`R2-F`).
fn next_arena_bytes(saw_builds: bool) -> u64 {
    if saw_builds { ARENA } else { READ_ARENA }
}

/// Builds a resource transition barrier for arena buffer copies.
fn arena_transition(
    buffer: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: non-owning copy valid for the duration of the `ResourceBarrier` call.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Runs incremental upper-arena compaction on idle/sweep ticks (CPU-backed / unpinned pages).
#[allow(dead_code)]
pub(crate) fn compact_idle() -> usize {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return 0;
    };
    m.compact_sparse_arenas(None)
}

/// Pure gate for ephemeral BVH4 rehydration arena reclamation: ONLY permitted when
/// Native DXR is active with builder suppression and `VICTORY_BVH_BUILDER` is OFF
/// (`VICTORY_NATIVE_RT=1 && VICTORY_BVH_BUILDER=0`). NEVER runs when software BVH
/// traversal shaders may still read arenas (`ARCH-10`).
fn ephemeral_release_gate(native_rt: bool, suppress_builder: bool, bvh_builder_env: bool) -> bool {
    native_rt && suppress_builder && !bvh_builder_env
}

fn bvh_builder_active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1"))
}

fn ephemeral_release_allowed() -> bool {
    ephemeral_release_gate(
        crate::dxr::enabled(),
        crate::dxr::suppress_builder(),
        bvh_builder_active(),
    )
}

/// Pins arena pages covering `[dest_va, dest_va + size)` while a Native DXR
/// `ExtractStaticBlas` submission is in flight so concurrent completions sharing a
/// 64 KiB boundary page never free it prematurely.
pub(crate) fn pin_extracted(dest_va: u64, size: u64) {
    if dest_va == 0 || size == 0 || !ephemeral_release_allowed() {
        return;
    }
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else { return };
    let first = dest_va / PAGE;
    let last = (dest_va.saturating_add(size) - 1) / PAGE;
    for page in first..=last {
        if let Some(held) = m.pages.get_mut(&page) {
            held.pinned = held.pinned.saturating_add(1);
        }
    }
}

/// Releases rehydrated BVH4 arena pages covering `[dest_va, dest_va + size)` once
/// Native DXR `ExtractStaticBlas` + node readback has completed on the GPU, and
/// retires completely empty arenas (substituting a shared 4 KiB null UAV in `copies`)
/// so DXR rehydration stays bounded to 1–2 arenas (`64–128 MiB` instead of `912 MiB`).
///
/// Strictly gated on `crate::dxr::enabled() && !bvh_builder_active` (`VICTORY_NATIVE_RT=1`
/// and `VICTORY_BVH_BUILDER=0`); never runs when software BVH shaders may read arenas.
pub(crate) fn release_extracted(dest_va: u64, size: u64) {
    if dest_va == 0 || size == 0 || !ephemeral_release_allowed() {
        return;
    }
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else { return };
    m.release_extracted_range(dest_va, size);
}

/// `VICTORY_BVH_TRACE=1`: the sizes the builder is asked for (its program of
/// one thread that stores a structure's compacted size) and when title memory
/// has them, and the copies to a place of that size.
#[cfg(feature = "diagnostics")]
mod trace {
    use super::*;

    #[derive(Default)]
    pub(super) struct Trace {
        /// Frame, where the size goes, the structure, what title memory had.
        asked: Vec<(u64, u64, u64, u64)>,
        /// By structure, what its size was when last seen to change.
        sizes: HashMap<u64, (u64, u64)>,
        said: u32,
    }

    fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_TRACE").is_some_and(|v| v == "1"))
    }

    fn value(at: u64) -> u64 {
        let mut b = [0u8; 8];
        crate::host::copy_mapped(b.as_mut_ptr(), at as *const u8, 8, false);
        u64::from_le_bytes(b)
    }

    impl Trace {
        fn say(&mut self, text: String) {
            self.said += 1;
            if self.said < 6000 {
                tracing::info!("bvh trace: {text}");
            }
        }

        pub(super) fn look(&mut self, frame: u64) {
            if !on() {
                return;
            }
            let mut gone = Vec::new();
            for i in 0..self.asked.len() {
                let (then, at, structure, was) = self.asked[i];
                let now = value(at);
                if now != was {
                    self.sizes.insert(structure, (now, frame));
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: {was:#x} -> {now:#x}"));
                } else if frame > then + 60 {
                    gone.push(i);
                    self.say(format!("frame {frame}: size of {structure:#x} at {at:#x} asked in {then}: still {was:#x}"));
                }
            }
            for i in gone.into_iter().rev() {
                self.asked.swap_remove(i);
            }
        }

        pub(super) fn dispatch(
            &mut self,
            frame: u64,
            list: &HostList,
            addresses: &[u64],
            written: &[u64],
            constants: &[u32],
            threads: Option<[u32; 3]>,
        ) {
            if !on() {
                return;
            }
            let _ = addresses;
            let list = std::ptr::from_ref(list) as usize & 0xFFFF_FFFF;
            if threads == Some([1, 1, 1])
                && let [lo, hi, mode, ..] = *constants
                && (1..=3).contains(&mode)
                && hi < 0x20
                && let Some(&at) = written.first()
                && crate::shadow::in_buffer(at, 1).is_some_and(|b| b.1 <= BACK)
            {
                let structure = u64::from(lo) | u64::from(hi) << 32;
                let was = value(at);
                self.say(format!("frame {frame} list {list:x}: asks mode {mode} of {structure:#x} to {at:#x}, there {was:#x}"));
                if mode == 1 {
                    self.asked.push((frame, at, structure, was));
                }
            } else if threads == Some([32, 1, 1])
                && let [from, to] = *written
                && [from, to].iter().all(|&a| a >= 1 << 32 && crate::shadow::in_buffer(a, 1).is_none())
            {
                let known = self.sizes.get(&from).copied();
                self.say(format!("frame {frame} list {list:x}: copies {from:#x} to {to:#x} constants {:x?}, its size was {known:x?}", &constants[..constants.len().min(4)]));
            }
        }
    }
}

/// `VICTORY_MEMORY_DUMP=<folder>`: when a file `go` appears there, the arenas
/// as the card holds them are copied out and written (`arena-<n>.bin`), with
/// `pages.txt` (title page, arena, offset) and the structures the walking
/// programs were given: to look at the structures offline.
#[cfg(feature = "diagnostics")]
mod dump {
    use super::*;

    #[derive(Default)]
    pub(super) struct Dump {
        walked: std::collections::BTreeSet<u64>,
        /// Views the builder was given to write outside buffers: the first
        /// and the last frame, how often.
        pub(super) built: std::collections::BTreeMap<u64, (u64, u64, u64)>,
        /// The frame it is written at, and the copies.
        due: Option<(u64, Vec<(ID3D12Resource, usize, u64)>)>,
        done: bool,
    }

    impl Dump {
        pub(super) fn walked(&mut self, addresses: &[u64]) {
            if !self.done && self.walked.len() < 64 {
                self.walked.extend(addresses);
            }
        }
    }

    fn folder() -> Option<&'static std::path::PathBuf> {
        static FOLDER: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
        FOLDER.get_or_init(|| std::env::var_os("VICTORY_MEMORY_DUMP").map(Into::into)).as_ref()
    }

    fn turn(buffer: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: the same pointer, not counted; the barrier is
                    // used while the buffer lives.
                    pResource: unsafe { std::mem::transmute_copy(buffer) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    impl Memory {
        /// At a list's first use: the copies are recorded into it.
        pub(super) fn dump(&mut self, list: &HostList, _: &[u64]) {
            let Some(folder) = folder() else { return };
            if let Some((due, _)) = &self.dump.due {
                if self.frame < *due {
                    return;
                }
                let copies = self.dump.due.take().map(|d| d.1).unwrap_or_default();
                let mut text = String::new();
                for a in &self.dump.walked {
                    text += &format!("walked {a:#x}\n");
                }
                for (a, (first, last, n)) in &self.dump.built {
                    text += &format!("built {a:#x} {first} {last} {n}\n");
                }
                for (a, p) in &self.packed {
                    text += &format!("packed {a:#x} {} {}\n", p.len, p.frame);
                }
                text += &format!("frame {}\n", self.frame);
                for (page, held) in &self.pages {
                    let Some(arena) = self.arenas.iter().position(|a| !a.retired && a.copy == held.copy) else {
                        continue;
                    };
                    let at = held.at - self.arenas[arena].base;
                    text += &format!("{:#x} {arena} {at:#x}\n", page * PAGE);
                }
                let _ = std::fs::write(folder.join("pages.txt"), text);
                for (n, (_, data, size)) in copies.iter().enumerate() {
                    // SAFETY: a mapping of an arena's bytes; the copies ran
                    // frames ago.
                    let bytes = unsafe { std::slice::from_raw_parts(*data as *const u8, *size as usize) };
                    let _ = std::fs::write(folder.join(format!("arena-{n}.bin")), bytes);
                }
                tracing::info!(arenas = copies.len(), "title memory by address: the arenas are written out");
                return;
            }
            if self.dump.done || !folder.join("go").exists() {
                return;
            }
            self.dump.done = true;
            let Some(host) = host() else { return };
            let mut copies = Vec::new();
            for arena in &self.arenas {
                if arena.retired {
                    continue;
                }
                let buffer = &self.copies[arena.copy].buffer;
                let Some((to, data)) = host.shared_buffer(arena.size) else { break };
                // SAFETY: an open list; live buffers of those sizes. The list
                // has not used the arena yet.
                unsafe {
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    list.list.CopyBufferRegion(&to, 0, buffer, 0, arena.size);
                    list.list.ResourceBarrier(&[turn(buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
                }
                list.keep(&to);
                copies.push((to, data as usize, arena.size));
            }
            self.dump.due = Some((self.frame + 12, copies));
        }
    }
}

#[cfg(test)]
mod producer_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_piece_takes_what_changed() {
        let mut title = [7u8; PIECE];
        let mut ours = vec![0u8; PIECE];
        assert_eq!(merge(&title, &mut ours, true), PIECE / 4);
        assert_eq!(ours, title);
        assert_eq!(merge(&title, &mut ours, false), 0);
        // A dword of it.
        title[41] = 8;
        assert_eq!(merge(&title, &mut ours, false), 1);
        assert_eq!(ours, title);
    }

    #[test]
    fn a_packed_structure_is_known_by_its_header() {
        let header = |version: u32, nodes: u32| u64::from(version) | u64::from(nodes) << 32;
        assert_eq!(packed_len(header(0x8102_0006, 0x9a1a)), Some(0x9a1b * 64));
        // Unpacked, a top level, another version, no size, too big.
        assert_eq!(packed_len(header(0x8002_0006, 0x9a1a)), None);
        assert_eq!(packed_len(header(0x0000_0006, 5)), None);
        assert_eq!(packed_len(header(0x8102_0007, 5)), None);
        assert_eq!(packed_len(header(0x8102_0006, 0)), None);
        assert_eq!(packed_len(header(0x8102_0006, 1 << 21)), None);
    }

    #[test]
    fn ephemeral_release_gate_and_initial_read_arena_size() {
        // Initial arena for Use::Reads is 4 MiB; Use::Builds is 16 MiB (256 × 64 KiB pages).
        assert_eq!(ARENA, 16 << 20);
        assert_eq!(ARENA_PAGES, 256);
        assert_eq!(SPARE, 256);
        assert_eq!(next_arena_bytes(false), 4 << 20);
        assert_eq!(next_arena_bytes(true), 16 << 20);

        // Strictly allowed ONLY when VICTORY_NATIVE_RT=1, suppress_builder=1, VICTORY_BVH_BUILDER=0.
        assert!(ephemeral_release_gate(true, true, false));
        assert!(!ephemeral_release_gate(true, true, true));
        assert!(!ephemeral_release_gate(true, false, false));
        assert!(!ephemeral_release_gate(false, true, false));
        assert!(!ephemeral_release_gate(false, false, true));
    }

    #[test]
    fn compact_sparse_arenas_preserves_page_contents_and_reclaims_upper_arena() {
        let Some(mut m) = Memory::new() else {
            return;
        };
        m.saw_builds = true;
        // Allocate two 16 MiB builder arenas (256 pages each) and verify exact sizing.
        assert!(m.another(ARENA));
        assert!(m.another(ARENA));
        assert_eq!(m.arenas.len(), 2);
        assert_eq!(m.arenas[0].size, 16 << 20);
        assert_eq!(m.arenas[0].free.len(), 256);
        assert_eq!(m.arenas[1].size, 16 << 20);
        assert_eq!(m.arenas[1].free.len(), 256);

        // Place 2 pages in arena 0 and 3 pages in arena 1 (occupancy 3/256 = 1.17% < 35%).
        let copy0 = m.arenas[0].copy;
        for i in 0..2u64 {
            let at = m.arenas[0].free.pop().unwrap();
            m.arenas[0].live_pages += 1;
            m.free_pages -= 1;
            m.hold(0x3_0000 + i, at, copy0);
        }

        let copy1 = m.arenas[1].copy;
        let moved_pages = [0x3_0100u64, 0x3_0101, 0x3_0102];
        for (idx, &p) in moved_pages.iter().enumerate() {
            let at = m.arenas[1].free.pop().unwrap();
            m.arenas[1].live_pages += 1;
            m.free_pages -= 1;
            m.hold(p, at, copy1);
            // Write distinct deterministic pattern across the full 64 KiB page.
            let slice = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, PAGE as usize) };
            for (b_idx, byte) in slice.iter_mut().enumerate() {
                *byte = ((idx as u8).wrapping_mul(53)) ^ ((b_idx as u8).wrapping_add(17));
            }
        }

        // Pin one page in arena 1 -> compaction must skip arena 1 while any page is pinned.
        m.pages.get_mut(&moved_pages[0]).unwrap().pinned = 1;
        assert_eq!(m.compact_sparse_arenas(None), 0);
        assert!(!m.arenas[1].retired);
        assert_eq!(m.arenas[1].live_pages, 3);

        // Unpin -> compaction moves all 3 pages into arena 0, preserves exact 64 KiB contents,
        // updates directory entries cleanly, and retires upper arena 1 to the 4 KiB null copy.
        m.pages.get_mut(&moved_pages[0]).unwrap().pinned = 0;
        assert_eq!(m.compact_sparse_arenas(None), 3);
        assert_eq!(m.arenas[0].live_pages, 5);
        assert_eq!(m.arenas[1].live_pages, 0);
        assert!(m.arenas[1].retired);
        assert!(m.retired_copies.contains(&copy1));
        assert_eq!(m.copies[copy1].span().1, NULL_COPY_BYTES);

        for (idx, &p) in moved_pages.iter().enumerate() {
            let held = m.pages.get(&p).expect("moved page stays tracked");
            assert_eq!(held.copy, copy0);
            let dir_copy = unsafe { m.entry(p).read() };
            let dir_off = unsafe { m.entry(p).add(1).read() };
            assert_eq!(dir_copy, copy0 as u32 + 1);
            assert_eq!(u64::from(dir_off), held.at as u64 - m.copies[copy0].span().0);

            let slice = unsafe { std::slice::from_raw_parts(held.at as *const u8, PAGE as usize) };
            for (b_idx, &byte) in slice.iter().enumerate() {
                let expected = ((idx as u8).wrapping_mul(53)) ^ ((b_idx as u8).wrapping_add(17));
                assert_eq!(byte, expected, "page {p:#x} byte {b_idx} corrupted across compaction");
            }
        }
    }

    #[test]
    fn release_extracted_reclaims_empty_arenas_and_substitutes_null_copy() {
        let Some(mut m) = Memory::new() else {
            return;
        };
        m.saw_builds = true;
        // Allocate two small test arenas (4 MiB each) so the test stays fast and light.
        assert!(m.another(READ_ARENA));
        assert!(m.another(READ_ARENA));
        assert_eq!(m.arenas.len(), 2);
        assert_eq!(m.arenas.iter().filter(|a| !a.retired).count(), 2);

        // Map one page in arena 0 and one page in arena 1 directly.
        let copy0 = m.arenas[0].copy;
        let at0 = m.arenas[0].free.pop().unwrap();
        m.arenas[0].live_pages += 1;
        m.free_pages -= 1;
        let page0 = 0x2_0000u64;
        m.hold(page0, at0, copy0);

        let copy1 = m.arenas[1].copy;
        let at1 = m.arenas[1].free.pop().unwrap();
        m.arenas[1].live_pages += 1;
        m.free_pages -= 1;
        let page1 = 0x2_0001u64;
        m.hold(page1, at1, copy1);

        // Pin page1 twice, simulate first extract completion: stays pinned & live.
        m.pages.get_mut(&page1).unwrap().pinned = 2;
        assert_eq!(m.release_extracted_range(page1 * PAGE, PAGE), 0);
        assert_eq!(m.arenas[1].live_pages, 1);

        // Release page0 (unpinned) and page1 (last pin drops to 0): both pages free,
        // directory entries zero out, and arena 1 retires while arena 0 stays as 1 warm spare.
        assert_eq!(m.release_extracted_range(page0 * PAGE, PAGE), 0);
        assert_eq!(m.release_extracted_range(page1 * PAGE, PAGE), 1);
        assert_eq!(unsafe { m.entry(page0).read() }, 0);
        assert_eq!(unsafe { m.entry(page1).read() }, 0);
        assert_eq!(m.arenas[0].live_pages, 0);
        assert_eq!(m.arenas[1].live_pages, 0);
        assert!(!m.arenas[0].retired);
        assert!(m.arenas[1].retired);
        assert!(m.retired_copies.contains(&copy1));
        // `copies[copy1]` is now replaced with the shared 4 KiB null UAV so descriptors stay valid.
        assert_eq!(m.copies[copy1].span().1, NULL_COPY_BYTES);
    }
}
