//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! The title shows one physical page at several addresses, and writes it
//! through one while the card reads it at another. So a block is clean, and
//! known by its bit, only while every view of its page is read-only
//! ([`cover`]): a write through any of them faults, opens them all and takes
//! clean and bit from all of them ([`open`]). A view mapped anew of a page
//! that is guarded elsewhere has the others looked at again
//! ([`views_mapped`]). What we write ourselves is told at one address and
//! counts at all ([`writing`], [`landed_beside`], [`wrote`]). Not so for a
//! page shown in more than [`MANY_VIEWS`] places (what the title shows where
//! it has nothing). `VICTORY_VIEW_GUARD=0`: each address for itself, as
//! before.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Bound, RangeBounds};
use std::sync::{Condvar, Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// The parts of a page.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// All of them.
const WHOLE: u16 = u16::MAX;

/// Whether the parts of a page that the kernel names are kept, and a refresh
/// copies those (`VICTORY_PAGE_PARTS=0`: the 64 KiB page, as before).
///
/// The kernel says which 4 KiB pages were written, and a copy took the 64 KiB
/// around each: 29 MB a frame written so were most of 114 MB copied, and as
/// much sent up and hashed again on the card (session play-1125-0203, run
/// cpu-2). What is checked is as before, a page.
fn parts_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_PAGE_PARTS").is_none_or(|v| v != "0")
            && std::env::var_os("VICTORY_DIRTY_4K").is_none_or(|v| v != "0")
    })
}

/// `VICTORY_DIRTY_4K`: 0 unset or off, 1 on (with stats), 2 in turn (`alt`).
fn dirty4k_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_DIRTY_4K").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether `VICTORY_DIRTY_4K` telemetry is active (`1` or `alt`).
pub(crate) fn dirty4k_kept() -> bool {
    dirty4k_mode() != 0
}

/// Whether `VICTORY_DIRTY_4K` currently requests 4 KiB narrowing (`1` always,
/// `alt` alternating 20s on / 20s off).
pub(crate) fn dirty4k() -> bool {
    match dirty4k_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "refreshes of write-watched memory copy only the 4 KiB pages written from now (VICTORY_DIRTY_4K=alt): {}",
                    if on { "ON" } else { "off, whole 64 KiB pages as usual" }
                );
            }
            on
        }
    }
}

/// Whether refreshes narrow stale 64 KiB pages to their written 4 KiB parts
/// right now (`VICTORY_PAGE_PARTS` default on, `VICTORY_DIRTY_4K=0|1|alt`).
pub(crate) fn dirty4k_narrow() -> bool {
    if !parts_on() {
        return false;
    }
    match dirty4k_mode() {
        2 => dirty4k(),
        _ => true,
    }
}

/// Since the start, with `VICTORY_DIRTY_4K`: stale pages a refresh could narrow
/// to their 4 KiB pages, the 4 KiB pages of them, those narrowed in fact, and
/// stale pages copied whole (written another way or taken too long ago).
pub(crate) static DIRTY4K: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Counts a stale page of a refresh ([`DIRTY4K`]): `found`, its 4 KiB mask if
/// it could be narrowed; `narrowed`, whether it was.
pub(crate) fn dirty4k_count(found: Option<u16>, narrowed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(bits) = found else {
        DIRTY4K[3].fetch_add(1, Relaxed);
        return;
    };
    DIRTY4K[1].fetch_add(u64::from(bits.count_ones()), Relaxed);
    DIRTY4K[2].fetch_add(u64::from(narrowed), Relaxed);
    let n = DIRTY4K[0].fetch_add(1, Relaxed) + 1;
    if n.is_power_of_two() {
        let pieces = DIRTY4K[1].load(Relaxed);
        tracing::info!(
            "stale 64 KiB pages of write-watched memory (VICTORY_DIRTY_4K), since the start: {n} could be copied as their {pieces} written 4 KiB pages ({:.1} MB instead of {:.1} MB), {} were; {} copied whole (written another way, or taken too long ago)",
            pieces as f64 * 4096.0 / 1e6,
            n as f64 * 65536.0 / 1e6,
            DIRTY4K[2].load(Relaxed),
            DIRTY4K[3].load(Relaxed)
        );
    }
}

/// Pages in one journal piece (64 MiB).
const PIECE_PAGES: usize = 1 << SHARD_SHIFT;
const PIECE_WORDS: usize = PIECE_PAGES / 64;

/// Notes for one 64 MiB piece of title address space (`piece = page >> SHARD_SHIFT`).
struct NotePiece {
    piece: usize,
    max_written: u64,
    present: [u64; PIECE_WORDS],
    at: Box<[u64; PIECE_PAGES]>,
}

impl NotePiece {
    fn new(piece: usize) -> NotePiece {
        NotePiece {
            piece,
            max_written: 0,
            present: [0; PIECE_WORDS],
            at: Box::new([0; PIECE_PAGES]),
        }
    }

    fn note(&mut self, idx: usize, now: u64) -> (u64, u64) {
        let before = self.at[idx];
        let latest = before.max(now);
        self.at[idx] = latest;
        self.present[idx / 64] |= 1u64 << (idx % 64);
        if latest > self.max_written {
            self.max_written = latest;
        }
        (before, latest)
    }

    fn for_each(&self, lo: usize, hi: usize, f: &mut impl FnMut(usize, u64) -> bool) -> bool {
        if lo >= hi {
            return true;
        }
        let base = self.piece << SHARD_SHIFT;
        for w in lo / 64..=(hi - 1) / 64 {
            let mut bits = self.present[w] & in_word(w * 64, lo, hi);
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let idx = w * 64 + bit;
                if !f(base + idx, self.at[idx]) {
                    return false;
                }
            }
        }
        true
    }

    fn all_below(&self, lo: usize, hi: usize, floor: u64, pred: &mut impl FnMut(usize, u64) -> bool) -> bool {
        if self.max_written <= floor || lo >= hi {
            return true;
        }
        let base = self.piece << SHARD_SHIFT;
        for w in lo / 64..=(hi - 1) / 64 {
            let mut bits = self.present[w] & in_word(w * 64, lo, hi);
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let idx = w * 64 + bit;
                let at = self.at[idx];
                if at > floor && !pred(base + idx, at) {
                    return false;
                }
            }
        }
        true
    }
}

impl Shard {
    const fn new() -> Shard {
        Shard {
            pieces: Vec::new(),
            max_written: 0,
            parts: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()),
        }
    }

    fn find_piece(&self, piece: usize) -> Option<&NotePiece> {
        if self.pieces.len() <= 8 {
            self.pieces.iter().find(|p| p.piece == piece)
        } else {
            self.pieces.binary_search_by_key(&piece, |p| p.piece).ok().map(|i| &self.pieces[i])
        }
    }

    fn piece_mut(&mut self, piece: usize) -> &mut NotePiece {
        let pos = if self.pieces.len() <= 8 {
            match self.pieces.iter().position(|p| p.piece >= piece) {
                Some(i) if self.pieces[i].piece == piece => return &mut self.pieces[i],
                Some(i) => i,
                None => self.pieces.len(),
            }
        } else {
            match self.pieces.binary_search_by_key(&piece, |p| p.piece) {
                Ok(i) => return &mut self.pieces[i],
                Err(i) => i,
            }
        };
        self.pieces.insert(pos, NotePiece::new(piece));
        &mut self.pieces[pos]
    }

    #[cfg(test)]
    fn written_at(&self, page: usize) -> Option<u64> {
        let p = self.find_piece(page >> SHARD_SHIFT)?;
        let idx = page & (PIECE_PAGES - 1);
        ((p.present[idx / 64] >> (idx % 64)) & 1 != 0).then(|| p.at[idx])
    }

    /// Page `page` was found written by asking `now`: the parts in `parts`
    /// (a bit each). The asking that found it written before (0: none).
    ///
    /// A page's notes only go up: with the journal held alone `now` is the
    /// newest number there is, as it always was; noted under the shared lock
    /// (renderer port #7) another asking with a higher number may have
    /// noted it first.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let (before, latest) = self.piece_mut(page >> SHARD_SHIFT).note(page & (PIECE_PAGES - 1), now);
        if latest > self.max_written {
            self.max_written = latest;
        }
        if parts == WHOLE || !parts_on() {
            if now >= before {
                self.parts.remove(&page);
            } else if let Some(all) = self.parts.get_mut(&page) {
                all.iter_mut().for_each(|at| *at = (*at).max(now));
            }
            return before;
        }
        let all = self.parts.entry(page).or_insert([before; PARTS]);
        for (k, at) in all.iter_mut().enumerate() {
            if parts >> k & 1 != 0 {
                *at = (*at).max(now);
            }
        }
        before
    }

    /// The parts of `page` written by an asking after `taken`.
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        match self.parts.get(&page) {
            Some(all) => all.iter().enumerate().filter(|(_, at)| **at > taken).fold(0, |m, (k, _)| m | 1 << k),
            None => WHOLE,
        }
    }

    fn for_each_piece(&self, piece: usize, lo: usize, hi: usize, f: &mut impl FnMut(usize, u64) -> bool) -> bool {
        self.find_piece(piece).is_none_or(|p| p.for_each(lo, hi, f))
    }

    fn all_below_piece(
        &self,
        piece: usize,
        lo: usize,
        hi: usize,
        floor: u64,
        pred: &mut impl FnMut(usize, u64) -> bool,
    ) -> bool {
        self.max_written <= floor || self.find_piece(piece).is_none_or(|p| p.all_below(lo, hi, floor, pred))
    }

    fn for_each_range(&self, first: usize, last: usize, f: &mut impl FnMut(usize, u64) -> bool) -> bool {
        if first >= last || self.pieces.is_empty() {
            return true;
        }
        let (p_first, p_last) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        let start = self.pieces.partition_point(|p| p.piece < p_first);
        for p in &self.pieces[start..] {
            if p.piece > p_last {
                break;
            }
            let base = p.piece << SHARD_SHIFT;
            let lo = first.saturating_sub(base);
            let hi = (last - base).min(PIECE_PAGES);
            if !p.for_each(lo, hi, f) {
                return false;
            }
        }
        true
    }

    fn all_below_range(&self, first: usize, last: usize, floor: u64, pred: &mut impl FnMut(usize, u64) -> bool) -> bool {
        if self.max_written <= floor || first >= last || self.pieces.is_empty() {
            return true;
        }
        let (p_first, p_last) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        let start = self.pieces.partition_point(|p| p.piece < p_first);
        for p in &self.pieces[start..] {
            if p.piece > p_last {
                break;
            }
            let base = p.piece << SHARD_SHIFT;
            let lo = first.saturating_sub(base);
            let hi = (last - base).min(PIECE_PAGES);
            if !p.all_below(lo, hi, floor, pred) {
                return false;
            }
        }
        true
    }
}

/// The pages of a part of the journal ([`Notes`]): pieces of
/// `1 << SHARD_SHIFT` pages (64 MiB), every [`SHARDS`]th piece in one part.
const SHARD_SHIFT: u32 = 10;
/// The parts: a bit each in a `u64` ([`shards_of`]).
const SHARDS: usize = 64;

fn shard_of(page: usize) -> usize {
    (page >> SHARD_SHIFT) % SHARDS
}

/// The parts that hold pages `[first, last)`, a bit each.
fn shards_of(first: usize, last: usize) -> u64 {
    if first >= last {
        return 0;
    }
    let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
    if b - a + 1 >= SHARDS {
        return u64::MAX;
    }
    (a..=b).fold(0, |m, piece| m | 1 << (piece % SHARDS))
}

/// The parts that hold `pages`, a bit each.
fn shards_of_pages(pages: impl IntoIterator<Item = usize>) -> u64 {
    pages.into_iter().fold(0, |m, page| m | 1 << shard_of(page))
}

/// The pages' notes, in parts of their own, each under a lock of its own
/// (renderer port #7). Only reached through [`JOURNAL`]: held alone, its
/// holder changes them with no part's lock; held shared, a reader reads a
/// part at a time, and who notes or needs a number that counts holds the
/// parts it touches ([`Notes::hold`]).
struct Notes {
    shards: [RwLock<Shard>; SHARDS],
    /// Lock-free upper bound of each shard's `max_written`, updated under the
    /// shard write lock (or exclusive journal lock) before/during noting.
    shard_max: [std::sync::atomic::AtomicU64; SHARDS],
}

/// A part of [`Notes`].
struct Shard {
    /// Active 64 MiB pieces sorted by `piece` (`page >> SHARD_SHIFT`).
    pieces: Vec<NotePiece>,
    /// Largest asking recorded in `pieces`, for O(1) clean-shard skipping.
    max_written: u64,
    /// The same for each of a page's sixteen parts ([`KERNEL_PAGE`]), where
    /// the parts are known ([`parts_on`]); a page that is not here was
    /// written whole, as far as anyone knows.
    parts: std::collections::HashMap<usize, [u64; PARTS], std::hash::BuildHasherDefault<PageHash>>,
}

impl Notes {
    const fn new() -> Notes {
        Notes {
            shards: [const { RwLock::new(Shard::new()) }; SHARDS],
            shard_max: [const { std::sync::atomic::AtomicU64::new(0) }; SHARDS],
        }
    }

    /// A part to read, the journal held shared or alone.
    fn read(&self, shard: usize) -> std::sync::RwLockReadGuard<'_, Shard> {
        if let Ok(s) = self.shards[shard].try_read() {
            return s;
        }
        let began = std::time::Instant::now();
        let s = self.shards[shard].read().unwrap_or_else(|e| e.into_inner());
        port7_waited(P7_PART_WAITS, began);
        s
    }

    #[inline]
    fn shard_max(&self, shard: usize) -> u64 {
        self.shard_max[shard].load(std::sync::atomic::Ordering::Acquire)
    }

    /// See [`Shard::note`]; the journal held alone.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let shard = shard_of(page);
        let s = self.shards[shard].get_mut().unwrap_or_else(|e| e.into_inner());
        let before = s.note(page, parts, now);
        self.shard_max[shard].store(s.max_written, std::sync::atomic::Ordering::Release);
        before
    }

    /// The notes of pages `[first, last)` in page order, `each` of them
    /// until it says no: whether it never did. A part at a time.
    fn each(&self, first: usize, last: usize, mut each: impl FnMut(usize, u64) -> bool) -> bool {
        if first >= last {
            return true;
        }
        let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        if b - a < 4 * SHARDS {
            for piece in a..=b {
                let shard = piece % SHARDS;
                if self.shard_max(shard) == 0 {
                    continue;
                }
                let base = piece << SHARD_SHIFT;
                let lo = first.saturating_sub(base);
                let hi = (last - base).min(PIECE_PAGES);
                let s = self.read(shard);
                if !s.for_each_piece(piece, lo, hi, &mut each) {
                    return false;
                }
            }
            true
        } else {
            let mut all: Vec<(usize, u64)> = Vec::new();
            for shard in 0..SHARDS {
                if self.shard_max(shard) == 0 {
                    continue;
                }
                self.read(shard).for_each_range(first, last, &mut |p, w| {
                    all.push((p, w));
                    true
                });
            }
            all.sort_unstable();
            all.into_iter().all(|(p, w)| each(p, w))
        }
    }

    /// Order-independent predicate over notes in `[first, last)`, skipping any
    /// shard whose `shard_max <= floor` lock-free in O(1).
    #[cfg(test)]
    fn all_below(&self, first: usize, last: usize, floor: u64, mut pred: impl FnMut(usize, u64) -> bool) -> bool {
        if first >= last {
            return true;
        }
        let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        if b - a < 4 * SHARDS {
            for piece in a..=b {
                let shard = piece % SHARDS;
                if self.shard_max(shard) <= floor {
                    continue;
                }
                let base = piece << SHARD_SHIFT;
                let lo = first.saturating_sub(base);
                let hi = (last - base).min(PIECE_PAGES);
                let s = self.read(shard);
                if !s.all_below_piece(piece, lo, hi, floor, &mut pred) {
                    return false;
                }
            }
            true
        } else {
            for shard in 0..SHARDS {
                if self.shard_max(shard) <= floor {
                    continue;
                }
                let s = self.read(shard);
                if !s.all_below_range(first, last, floor, &mut pred) {
                    return false;
                }
            }
            true
        }
    }

    /// The notes of pages `[first, last)` in page order. A part at a time.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        let mut out = Vec::new();
        self.each(first, last, |p, w| {
            out.push((p, w));
            true
        });
        out
    }

    /// The parts `shards` (a bit each), held until dropped: for writing
    /// (`write`) or reading. Taken in order: whoever waits for a part holds
    /// only lower ones, so no two holders wait for each other.
    fn hold(&self, shards: u64, write: bool) -> Held<'_> {
        let mut held = Held { notes: self, shards, read: Vec::new(), write: Vec::new() };
        let mut m = shards;
        while m != 0 {
            let shard = m.trailing_zeros() as usize;
            m &= m - 1;
            let lock = &self.shards[shard];
            if write {
                let guard = match lock.try_write() {
                    Ok(g) => g,
                    Err(_) => {
                        let began = std::time::Instant::now();
                        let g = lock.write().unwrap_or_else(|e| e.into_inner());
                        port7_waited(P7_PART_WAITS, began);
                        g
                    }
                };
                held.write.push(guard);
            } else {
                held.read.push(self.read(shard));
            }
        }
        held
    }
}

/// The pieces of pages `[first, last)` in page order, each in one part;
/// `None` for a range of more pieces than is worth walking (then each part
/// is looked at once).
#[cfg(test)]
fn pieces(first: usize, last: usize) -> Option<Vec<(usize, usize)>> {
    if first >= last {
        return Some(Vec::new());
    }
    let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
    if b - a >= 4 * SHARDS {
        return None;
    }
    Some((a..=b).map(|piece| (first.max(piece << SHARD_SHIFT), last.min((piece + 1) << SHARD_SHIFT))).collect())
}

/// Parts of [`Notes`] held ([`Notes::hold`]).
struct Held<'a> {
    notes: &'a Notes,
    shards: u64,
    read: Vec<std::sync::RwLockReadGuard<'a, Shard>>,
    write: Vec<std::sync::RwLockWriteGuard<'a, Shard>>,
}

impl Held<'_> {
    /// The place among those held of the part of `page`.
    fn place(&self, page: usize) -> usize {
        let shard = shard_of(page);
        assert!(self.shards >> shard & 1 != 0, "page {page:#x}: its part is not held");
        (self.shards & ((1u64 << shard) - 1)).count_ones() as usize
    }

    fn shard(&self, page: usize) -> &Shard {
        let at = self.place(page);
        if self.write.is_empty() { &self.read[at] } else { &self.write[at] }
    }

    /// See [`Shard::note`]; held for writing.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let shard = shard_of(page);
        // Publish upper bound before inserting while holding the shard write lock
        // so lock-free `shard_max` readers linearize cleanly.
        self.notes.shard_max[shard].fetch_max(now, std::sync::atomic::Ordering::Release);
        let at = self.place(page);
        self.write[at].note(page, parts, now)
    }

    /// The notes of pages `[first, last)` in page order; their parts held.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        if first >= last {
            return Vec::new();
        }
        let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        let mut all = Vec::new();
        if b - a < 4 * SHARDS {
            for piece in a..=b {
                let base = piece << SHARD_SHIFT;
                let lo = first.saturating_sub(base);
                let hi = (last - base).min(PIECE_PAGES);
                self.shard(base).for_each_piece(piece, lo, hi, &mut |p, w| {
                    all.push((p, w));
                    true
                });
            }
        } else {
            if self.write.is_empty() {
                for s in &self.read {
                    s.for_each_range(first, last, &mut |p, w| {
                        all.push((p, w));
                        true
                    });
                }
            } else {
                for s in &self.write {
                    s.for_each_range(first, last, &mut |p, w| {
                        all.push((p, w));
                        true
                    });
                }
            }
            all.sort_unstable();
        }
        all
    }
}

/// Of the pages of a copy that were written after it took them (page number,
/// the asking it took the page at): the parts to take again, a bit for each
/// 4 KiB, in the same order.
pub(crate) fn parts_written(pages: &[(usize, u64)]) -> Vec<u16> {
    let j = journal_read();
    parts_written_with_journal(&j, pages)
}

/// Same as [`parts_written`], reusing an already-held [`JournalRead`] so callers
/// holding `Shadow::copied` never re-acquire `JOURNAL.read()` out of lock order.
pub(crate) fn parts_written_with_journal(j: &JournalRead, pages: &[(usize, u64)]) -> Vec<u16> {
    let mut cached: Option<(usize, std::sync::RwLockReadGuard<'_, Shard>)> = None;
    let mut out = Vec::with_capacity(pages.len());
    for &(page, taken) in pages {
        let shard = shard_of(page);
        if cached.as_ref().is_none_or(|(id, _)| *id != shard) {
            drop(cached.take());
            cached = Some((shard, j.0.notes.read(shard)));
        }
        out.push(cached.as_ref().unwrap().1.parts_since(page, taken));
    }
    out
}

/// Numbers the askings; 0 is "never". Counted up with the journal held
/// (shared will do: what notes an answer holds it alone).
static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_asking() -> u64 {
    ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Per page, the asking that last found it written, and its parts'
    /// ([`Shard`]).
    notes: Notes,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

/// Read by many threads at once (a submission's refreshes run on several),
/// written only to note an answer: the kernel is asked, and view memory
/// guarded, with the range held in [`ASKING`] and not under this lock. Under
/// one lock for all of it those threads ran one at a time: 22 of 96 looks at
/// the submitting thread, and one in six of those at the six threads beside
/// it, found it waiting for that lock (run pooloff1).
static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    notes: Notes::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// Shared read guard over [`JOURNAL`] for batching [`nothing_new_in`] checks
/// across a chunk of spans with a single `RwLock::read` acquisition.
pub(crate) struct JournalRead(std::sync::RwLockReadGuard<'static, Journal>);

/// Acquires a shared read lock on [`JOURNAL`] for [`nothing_new_in`].
pub(crate) fn journal_read() -> JournalRead {
    JournalRead(journal())
}

/// Reusable scanner over a held [`JournalRead`] that caches at most one shard
/// read guard across spans (avoiding per-span SRWLock RMW contention across
/// worker threads while never holding two shard locks simultaneously).
pub(crate) struct JournalScan<'a> {
    journal: &'a JournalRead,
    cached: Option<(usize, std::sync::RwLockReadGuard<'a, Shard>)>,
}

impl JournalRead {
    pub(crate) fn scan(&self) -> JournalScan<'_> {
        JournalScan {
            journal: self,
            cached: None,
        }
    }
}

impl<'a> JournalScan<'a> {
    #[inline]
    fn shard(&mut self, shard: usize) -> &Shard {
        if self.cached.as_ref().is_none_or(|(id, _)| *id != shard) {
            // Drop any previously held shard guard before locking another.
            drop(self.cached.take());
            self.cached = Some((shard, self.journal.0.notes.read(shard)));
        }
        &self.cached.as_ref().unwrap().1
    }

    fn all_below(&mut self, first: usize, last: usize, floor: u64, mut pred: impl FnMut(usize, u64) -> bool) -> bool {
        if first >= last {
            return true;
        }
        let notes = &self.journal.0.notes;
        let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        if b - a < 4 * SHARDS {
            for piece in a..=b {
                let shard = piece % SHARDS;
                if notes.shard_max(shard) <= floor {
                    continue;
                }
                let base = piece << SHARD_SHIFT;
                let lo = first.saturating_sub(base);
                let hi = (last - base).min(PIECE_PAGES);
                if !self.shard(shard).all_below_piece(piece, lo, hi, floor, &mut pred) {
                    return false;
                }
            }
            true
        } else {
            for shard in 0..SHARDS {
                if notes.shard_max(shard) <= floor {
                    continue;
                }
                if !self.shard(shard).all_below_range(first, last, floor, &mut pred) {
                    return false;
                }
            }
            true
        }
    }
}

/// Sorts and merges overlapping/adjacent `(first, last)` page ranges so
/// [`ahead_covers`] can binary-search in `O(log N)`.
fn sort_coalesce_ahead(ahead: &mut Vec<(usize, usize)>) {
    if ahead.len() <= 1 {
        return;
    }
    ahead.sort_unstable();
    let mut write = 0usize;
    for read in 1..ahead.len() {
        let (a, b) = ahead[read];
        if a <= ahead[write].1 {
            ahead[write].1 = ahead[write].1.max(b);
        } else {
            write += 1;
            ahead[write] = (a, b);
        }
    }
    ahead.truncate(write + 1);
}

/// Whether sorted, coalesced `ahead` contains a range covering `[first, last)`.
fn ahead_covers(ahead: &[(usize, usize)], first: usize, last: usize) -> bool {
    let i = ahead.partition_point(|&(a, _)| a <= first);
    i > 0 && ahead[i - 1].1 >= last
}

fn journal() -> std::sync::RwLockReadGuard<'static, Journal> {
    if let Ok(j) = JOURNAL.try_read() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(1, || JOURNAL.read().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_READ_WAITS, began);
    j
}

fn journal_mut() -> std::sync::RwLockWriteGuard<'static, Journal> {
    port7_count(P7_ALONE, 1);
    if let Ok(j) = JOURNAL.try_write() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(2, || JOURNAL.write().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_ALONE_WAITS, began);
    j
}

/// Renderer port #7 (`VICTORY_PORT_JOURNAL=1`, off by default; read here,
/// and A/B-able in one run with `VICTORY_AB=<s>,VICTORY_PORT_JOURNAL`): the
/// journal's lock no longer makes the threads that read it wait for those
/// that note in it, nor those for each other, unless the standing asking
/// changes. (The switch also covers `crate::port_journal`'s heaps' records.)
pub(crate) fn port_journal() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_JOURNAL") {
        return on;
    }
    #[cfg(test)]
    if PORT_JOURNAL_IN_TEST.get() {
        return true;
    }
    // Read here: the switch of renderer port #7.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_JOURNAL").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// [`port_journal`], for what this thread does.
    pub(crate) static PORT_JOURNAL_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// Whether this thread is the submitting lane ("victory-submit").
    static ON_LANE: bool = std::thread::current().name() == Some("victory-submit");
}

/// Renderer port #7's counts since the last line: takings of the journal
/// alone; of them, those that waited and nanoseconds; waits for it shared
/// and nanoseconds; the lane's waits for it (both kinds) and nanoseconds;
/// takings alone found needless and left out; notings done shared; waits for
/// a part and nanoseconds.
static PORT7: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
const P7_ALONE: usize = 0;
/// (A count and its nanoseconds next to it, as the others below.)
const P7_ALONE_WAITS: usize = 1;
const P7_READ_WAITS: usize = 3;
const P7_LANE_WAITS: usize = 5;
const P7_SKIPPED: usize = 7;
const P7_SHARED_NOTES: usize = 8;
const P7_PART_WAITS: usize = 9;

fn port7_count(what: usize, n: u64) {
    PORT7[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// A wait of kind `what` (a count, its nanoseconds next) that began then.
fn port7_waited(what: usize, began: std::time::Instant) {
    let ns = began.elapsed().as_nanos() as u64;
    port7_count(what, 1);
    port7_count(what + 1, ns);
    if what != P7_PART_WAITS && ON_LANE.with(|l| *l) {
        port7_count(P7_LANE_WAITS, 1);
        port7_count(P7_LANE_WAITS + 1, ns);
    }
}

/// Renderer port #7's journal counts, taken ([`crate::port_journal::line`]).
pub(crate) struct Port7Counts([u64; 12]);

pub(crate) fn port_journal_counts() -> Port7Counts {
    Port7Counts(std::array::from_fn(|i| PORT7[i].swap(0, std::sync::atomic::Ordering::Relaxed)))
}

impl Port7Counts {
    /// For the line, per frame.
    pub(crate) fn per_frame(&self, frames: f64) -> String {
        let n = |i: usize| self.0[i] as f64 / frames;
        let ms = |i: usize| self.0[i] as f64 / 1e6 / frames;
        format!(
            "the lane waited for its lock {:.1} times, {:.2} ms; all threads waited to take it alone {:.1} times ({:.2} ms) of {:.1} \
             takings alone, shared {:.1} times ({:.2} ms); {:.1} takings alone found needless, {:.1} notings done shared, {:.1} waits \
             for a part ({:.2} ms)",
            n(P7_LANE_WAITS),
            ms(P7_LANE_WAITS + 1),
            n(P7_ALONE_WAITS),
            ms(P7_ALONE_WAITS + 1),
            n(P7_ALONE),
            n(P7_READ_WAITS),
            ms(P7_READ_WAITS + 1),
            n(P7_SKIPPED),
            n(P7_SHARED_NOTES),
            n(P7_PART_WAITS),
            ms(P7_PART_WAITS + 1),
        )
    }
}

/// Since the last line: waits and nanoseconds of waiting, of all threads, for
/// the ranges held ([`ASKING`]), the journal to read it, the journal to note
/// in it, and the guards (faults of the title's among them).
#[cfg(feature = "diagnostics")]
static WAITS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

#[cfg(feature = "diagnostics")]
fn waited<T>(what: usize, f: impl FnOnce() -> T) -> T {
    use std::sync::atomic::Ordering::Relaxed;
    let began = std::time::Instant::now();
    let got = f();
    WAITS[what][0].fetch_add(1, Relaxed);
    WAITS[what][1].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    got
}

#[cfg(not(feature = "diagnostics"))]
fn waited<T>(_: usize, f: impl FnOnce() -> T) -> T {
    f()
}

/// [`WAITS`], taken, for the log: per frame.
#[cfg(feature = "diagnostics")]
pub(crate) fn lock_waits(frames: f64) -> String {
    let of = |what: usize| {
        let [n, ns] = [0, 1].map(|i| WAITS[what][i].swap(0, std::sync::atomic::Ordering::Relaxed));
        format!("{:.0} {:.2} ms", n as f64 / frames, ns as f64 / 1e6 / frames)
    };
    format!("ranges held {}, journal read {}, journal noted in {}, guards {}", of(0), of(1), of(2), of(3))
}

/// The page ranges some thread is asking about: asking forgets what it
/// answers, so until the answer is in the journal no other thread may ask
/// about any of the same pages (it would be told they were not written).
/// Taken before the journal's lock, never under it.
static ASKING: (Mutex<Vec<(usize, usize)>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

/// Pages `[first, last)` held in [`ASKING`] until dropped.
struct Asking(usize, usize);

impl Asking {
    fn hold(first: usize, last: usize) -> Asking {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if held.iter().any(|&(a, b)| a < last && first < b) {
            held = waited(0, || {
                let mut held = held;
                while held.iter().any(|&(a, b)| a < last && first < b) {
                    held = ASKING.1.wait(held).unwrap_or_else(|e| e.into_inner());
                }
                held
            });
        }
        held.push((first, last));
        Asking(first, last)
    }

    /// Holds `[first, last)` in [`ASKING`] only if no overlapping asking is
    /// already active; never blocks.
    fn try_hold(first: usize, last: usize) -> Option<Asking> {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if held.iter().any(|&(a, b)| a < last && first < b) {
            return None;
        }
        held.push((first, last));
        Some(Asking(first, last))
    }
}

impl Drop for Asking {
    fn drop(&mut self) {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = held.iter().position(|&r| r == (self.0, self.1)) {
            held.swap_remove(i);
        }
        drop(held);
        ASKING.1.notify_all();
    }
}

/// Multiple disjoint `(first, last)` page ranges held atomically in [`ASKING`]
/// until dropped so unrelated ranges (e.g. descriptor heaps on
/// `victory-heap-sync`) never stall behind a submission's asking.
struct AskingMany(Vec<(usize, usize)>);

impl AskingMany {
    fn hold(ranges: Vec<(usize, usize)>) -> AskingMany {
        if ranges.is_empty() {
            return AskingMany(ranges);
        }
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        let overlaps = |held: &[(usize, usize)]| {
            ranges
                .iter()
                .any(|&(first, last)| held.iter().any(|&(a, b)| a < last && first < b))
        };
        if overlaps(&held) {
            held = waited(0, || {
                let mut held = held;
                while overlaps(&held) {
                    held = ASKING.1.wait(held).unwrap_or_else(|e| e.into_inner());
                }
                held
            });
        }
        held.extend_from_slice(&ranges);
        AskingMany(ranges)
    }
}

impl Drop for AskingMany {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        for &range in &self.0 {
            if let Some(i) = held.iter().position(|&r| r == range) {
                held.swap_remove(i);
            }
        }
        drop(held);
        ASKING.1.notify_all();
    }
}

thread_local! {
    /// Room for the kernel's answer.
    static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Monotonic write-epoch bumped whenever any title page is noted written,
/// unguarded, landed, or remapped. Combined with [`EPOCH_TABLE`] so submit-time
/// front planning can skip unchanged spans in O(1) without locking `JOURNAL`.
static WATCH_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// Floor epoch used when a single invalidation covers more than `EPOCH_BUCKETS`
/// 64 MiB buckets (`>= 1 TiB`).
static GLOBAL_EPOCH_FLOOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
const EPOCH_BUCKET_SHIFT: usize = 26; // 64 MiB per bucket
const EPOCH_BUCKETS: usize = 16_384; // 1 TiB direct-mapped
const EPOCH_BUCKET_MASK: usize = EPOCH_BUCKETS - 1;
static EPOCH_TABLE: [std::sync::atomic::AtomicU64; EPOCH_BUCKETS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; EPOCH_BUCKETS];

/// Current global write epoch (Acquire-ordered with all preceding journal/guard updates).
#[inline]
pub(crate) fn current_write_epoch() -> u64 {
    WATCH_EPOCH.load(std::sync::atomic::Ordering::Acquire)
}

/// Highest write epoch recorded for any 64 MiB VA bucket overlapping `[start, end)`.
#[inline]
pub(crate) fn range_write_epoch(start: usize, end: usize) -> u64 {
    use std::sync::atomic::Ordering::Acquire;
    if start >= end {
        return 0;
    }
    let first_b = start >> EPOCH_BUCKET_SHIFT;
    let last_b = (end - 1) >> EPOCH_BUCKET_SHIFT;
    if last_b - first_b >= EPOCH_BUCKETS {
        return WATCH_EPOCH.load(Acquire);
    }
    let mut max = GLOBAL_EPOCH_FLOOR.load(Acquire);
    for b in first_b..=last_b {
        let e = EPOCH_TABLE[b & EPOCH_BUCKET_MASK].load(Acquire);
        if e > max {
            max = e;
        }
    }
    max
}

/// Marks the 64 MiB VA buckets containing 64 KiB `pages` dirty at a fresh epoch.
/// Must be called strictly AFTER the corresponding `JOURNAL`/`PUBLISHED`/`GUARDS` updates.
fn mark_write_epoch_pages(pages: impl Iterator<Item = usize>) {
    use std::sync::atomic::Ordering::Release;
    let mut iter = pages.peekable();
    if iter.peek().is_none() {
        return;
    }
    let epoch = WATCH_EPOCH.fetch_add(1, Release) + 1;
    let mut last_bucket = usize::MAX;
    for page in iter {
        let b = ((page.saturating_mul(PAGE)) >> EPOCH_BUCKET_SHIFT) & EPOCH_BUCKET_MASK;
        if b != last_bucket {
            EPOCH_TABLE[b].fetch_max(epoch, Release);
            last_bucket = b;
        }
    }
}

/// Marks all 64 MiB VA buckets overlapping byte range `[start, end)` dirty at a fresh epoch.
/// Must be called strictly AFTER the corresponding `JOURNAL`/`PUBLISHED`/`GUARDS` updates.
fn mark_write_epoch_range(start: usize, end: usize) {
    use std::sync::atomic::Ordering::Release;
    if start >= end {
        return;
    }
    let epoch = WATCH_EPOCH.fetch_add(1, Release) + 1;
    let first_b = start >> EPOCH_BUCKET_SHIFT;
    let last_b = (end - 1) >> EPOCH_BUCKET_SHIFT;
    if last_b - first_b >= EPOCH_BUCKETS {
        GLOBAL_EPOCH_FLOOR.fetch_max(epoch, Release);
        return;
    }
    for b in first_b..=last_b {
        EPOCH_TABLE[b & EPOCH_BUCKET_MASK].fetch_max(epoch, Release);
    }
}

/// Whether adaptive hot-range write-watch reset suppression is enabled
/// (`VICTORY_ADAPTIVE_WATCH=1` opts in; default OFF because `flags = 0` keeps
/// previously dirty pages reported on every mid-frame submission, inflating
/// upload copy bytes across multi-submit frames).
fn adaptive_watch() -> bool {
    #[cfg(test)]
    if let Some(on) = ADAPTIVE_WATCH_IN_TEST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ADAPTIVE_WATCH").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    static ADAPTIVE_WATCH_IN_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[derive(Clone, Copy, Default)]
struct HotWatchEntry {
    /// 8-bit sliding window of recent submissions that reported dirty pages.
    hot_submits: u8,
    /// True once `WRITE_WATCH_FLAG_RESET` was suppressed for this range in the current frame.
    pending_reset: bool,
}

impl HotWatchEntry {
    #[inline]
    fn is_hot(self) -> bool {
        self.hot_submits.count_ones() >= 4
    }
}

const HOT_WATCH_MAX_ENTRIES: usize = 4_096;

/// Per-watched-range `(first_page, last_page)` adaptive reset state.
/// Lock ordering: leaf lock — never held across `ASKING`, `JOURNAL`, or `GetWriteWatch`.
static HOT_WATCH: Mutex<BTreeMap<(usize, usize), HotWatchEntry>> = Mutex::new(BTreeMap::new());

fn hot_watch_flags_for_groups(groups: &[(usize, usize, usize, usize)]) -> Vec<bool> {
    if !adaptive_watch() || groups.is_empty() {
        return vec![false; groups.len()];
    }
    let state = HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
    groups
        .iter()
        .map(|&(_, _, first, last)| state.get(&(first, last)).is_some_and(|e| e.is_hot()))
        .collect()
}

fn hot_watch_record(outcomes: &[(usize, usize, bool, bool)]) {
    if !adaptive_watch() || outcomes.is_empty() {
        return;
    }
    let mut state = HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
    if state.len() + outcomes.len() > HOT_WATCH_MAX_ENTRIES {
        state.retain(|_, e| e.pending_reset || e.is_hot());
        if state.len() > HOT_WATCH_MAX_ENTRIES {
            state.retain(|_, e| e.pending_reset);
        }
    }
    for &(first, last, had_dirty, suppressed) in outcomes {
        let entry = state.entry((first, last)).or_default();
        if !suppressed {
            entry.hot_submits = (entry.hot_submits << 1) | u8::from(had_dirty);
            entry.pending_reset = false;
        } else if !had_dirty {
            entry.hot_submits <<= 1;
            entry.pending_reset = false;
        } else if !entry.pending_reset {
            entry.hot_submits = (entry.hot_submits << 1) | 1;
            entry.pending_reset = true;
        }
    }
}

fn hot_watch_invalidate(first: usize, last: usize) {
    if first >= last {
        return;
    }
    let mut state = HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
    if !state.is_empty() {
        state.retain(|&(a, b), _| b <= first || a >= last);
    }
}

/// Resets write-watch bits once per frame at `Present` for all hot ranges whose
/// mid-frame `WRITE_WATCH_FLAG_RESET` was suppressed, recording any pages written
/// since the last submission into `JOURNAL` so no CPU write is ever dropped.
pub(crate) fn present_reset_hot_watches() {
    if !adaptive_watch() {
        return;
    }
    let mut to_reset: Vec<(usize, usize)> = {
        let mut state = HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
        if state.is_empty() {
            return;
        }
        let mut out = Vec::new();
        for (&range, entry) in state.iter_mut() {
            if entry.pending_reset {
                entry.pending_reset = false;
                out.push(range);
            } else if entry.is_hot() {
                entry.hot_submits <<= 1;
            }
        }
        if state.len() > HOT_WATCH_MAX_ENTRIES / 2 {
            state.retain(|_, e| e.hot_submits != 0 || e.pending_reset);
        }
        out
    };
    if to_reset.is_empty() {
        return;
    }
    // Sort by `(first, Reverse(last))` and drop ranges strictly contained in an earlier range
    // (never merging disjoint/touching ranges that might belong to distinct reservations).
    to_reset.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
    let mut filtered: Vec<(usize, usize)> = Vec::with_capacity(to_reset.len());
    for (first, last) in to_reset {
        if filtered.last().is_some_and(|&(pf, pl)| pf <= first && last <= pl) {
            continue;
        }
        filtered.push((first, last));
    }
    let _held = AskingMany::hold(filtered.clone());
    let mut written: Vec<(usize, u16)> = Vec::new();
    for (first, last) in filtered {
        if !kernel_written_into_flags(first, last, 1, &mut written) {
            invalidate_reservations(first, last);
        }
    }
    if written.is_empty() {
        return;
    }
    if port_journal() {
        let _ = note_shared(&written);
    } else {
        let mut j = journal_mut();
        let now = next_asking();
        for &(page, parts) in &written {
            j.notes.note(page, parts, now);
        }
        drop(j);
        mark_write_epoch_pages(written.iter().map(|&(p, _)| p));
    }
}

/// Appends the pages of `[first, last)` the kernel says were written since it
/// was last asked to `out`, returning `false` if the range is not write-watched
/// memory of one reservation. The caller holds the range ([`Asking`]).
fn kernel_written_into(first: usize, last: usize, out: &mut Vec<(usize, u16)>) -> bool {
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    kernel_written_into_flags(first, last, WRITE_WATCH_FLAG_RESET, out)
}

/// Same as [`kernel_written_into`], with explicit `GetWriteWatch` `flags`
/// (`WRITE_WATCH_FLAG_RESET = 1`, or `0` to query without clearing PTE bits).
fn kernel_written_into_flags(first: usize, last: usize, flags: u32, out: &mut Vec<(usize, u16)>) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    FOUND.with(|found| {
        let mut found = found.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                flags,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        #[cfg(feature = "diagnostics")]
        {
            tally(10, u64::from(failed));
            tally(11, started.elapsed().as_nanos() as u64);
        }
        if failed {
            return false;
        }
        #[cfg(feature = "diagnostics")]
        {
            tally(12, count as u64);
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
            slow(started.elapsed().as_micros() as u64);
            tally(0, 1);
            tally(1, ((last - first) * (PAGE >> 10)) as u64);
        }
        for &at in &found[..count] {
            let (page, part) = (at / PAGE, 1u16 << (at % PAGE / KERNEL_PAGE));
            match out.last_mut() {
                Some((last, parts)) if *last == page => *parts |= part,
                _ => out.push((page, part)),
            }
        }
        true
    })
}

/// The pages of `[first, last)` the kernel says were written since it was
/// last asked, in order, each once, with their written parts; `None` if the range is not write-watched
/// memory of one reservation. The caller holds the range ([`Asking`]).
fn kernel_written(first: usize, last: usize) -> Option<Vec<(usize, u16)>> {
    let mut pages = Vec::new();
    kernel_written_into(first, last, &mut pages).then_some(pages)
}

/// Merges sorted page ranges that strictly overlap (sharing at least one 64 KiB
/// page, which guarantees they lie in the same VirtualAlloc reservation).
/// Touching ranges (`a.1 == b.0`) are kept separate so `ask_all_run`/`ask_run`
/// can still split across adjacent reservations.
fn coalesce_overlapping_pieces(pieces: &mut Vec<(usize, usize)>) {
    if pieces.len() <= 1 {
        return;
    }
    let mut write = 0usize;
    for read in 1..pieces.len() {
        let (a, b) = pieces[read];
        if a < pieces[write].1 {
            pieces[write].1 = pieces[write].1.max(b);
        } else {
            write += 1;
            pieces[write] = (a, b);
        }
    }
    pieces.truncate(write + 1);
}

fn parse_watch_threads(raw: Option<&str>, cpus: usize) -> usize {
    let default = if cpus >= 8 { 4 } else { 1 };
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(1, 8)
}

/// Thread count for parallel `GetWriteWatch` across disjoint groups in
/// `ask_once` (`VICTORY_WATCH_THREADS=<N>`, default `4` on `>= 8`-core CPUs, else `1`).
fn watch_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        let raw = std::env::var("VICTORY_WATCH_THREADS").ok();
        let cpus = std::thread::available_parallelism().map(usize::from).unwrap_or(4);
        parse_watch_threads(raw.as_deref(), cpus)
    })
}

/// Dedicated lazy worker pool for `VICTORY_WATCH_THREADS >= 2`.
struct WatchPool {
    busy: Mutex<()>,
    state: Mutex<WatchPoolState>,
    more: Condvar,
    idle: Condvar,
    threads: usize,
    next: std::sync::atomic::AtomicUsize,
    running: std::sync::atomic::AtomicUsize,
}

struct WatchPoolState {
    work: Option<WatchWork>,
    count: usize,
    generation: u64,
}

#[derive(Clone, Copy)]
struct WatchWork(*const (dyn Fn(usize, usize) + Sync));

// SAFETY: `WatchPool::each_indexed` waits for all helpers to leave `work` before returning.
unsafe impl Send for WatchWork {}

impl WatchPool {
    fn get(threads: usize) -> &'static WatchPool {
        static POOL: std::sync::OnceLock<&'static WatchPool> = std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            let pool: &'static WatchPool = Box::leak(Box::new(WatchPool {
                busy: Mutex::new(()),
                state: Mutex::new(WatchPoolState {
                    work: None,
                    count: 0,
                    generation: 0,
                }),
                more: Condvar::new(),
                idle: Condvar::new(),
                threads,
                next: std::sync::atomic::AtomicUsize::new(0),
                running: std::sync::atomic::AtomicUsize::new(0),
            }));
            for i in 1..threads {
                let _ = std::thread::Builder::new().name(format!("victory-watch-{i}")).spawn(move || {
                    crate::cores::keep_helper_on_fast_cores("victory-watch");
                    pool.help(i);
                });
            }
            pool
        })
    }

    fn help(&self, worker_id: usize) {
        use std::sync::atomic::Ordering;
        let mut seen = 0u64;
        loop {
            let (work, count) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                while state.generation == seen || state.work.is_none() {
                    state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                }
                seen = state.generation;
                let work = state.work.unwrap();
                let count = state.count;
                self.running.fetch_add(1, Ordering::AcqRel);
                (work, count)
            };
            loop {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                if i >= count {
                    break;
                }
                // SAFETY: counted in `running`, so `each_indexed` has not returned.
                unsafe { (*work.0)(worker_id, i) };
            }
            if self.running.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                self.idle.notify_all();
            }
        }
    }

    #[cfg(test)]
    fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        self.each_indexed(count, &|_, i| f(i));
    }

    /// Runs `f(worker_id, item_idx)` for `item_idx in 0..count`, passing a unique
    /// `worker_id in 0..self.threads` (`0` for the caller, `1..self.threads` for helpers)
    /// so workers can accumulate into disjoint per-worker buffers with zero locking.
    fn each_indexed(&self, count: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        use std::sync::atomic::Ordering;
        if count < 2 || self.threads < 2 {
            for i in 0..count {
                f(0, i);
            }
            return;
        }
        let _busy = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: `_busy` serializes callers and `each_indexed` waits for `running == 0`
        // before returning, so the erased lifetime never escapes.
        let work = WatchWork(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize, usize) + Sync + '_), *const (dyn Fn(usize, usize) + Sync + 'static)>(f)
        });
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            self.next.store(0, Ordering::Relaxed);
            state.work = Some(work);
            state.count = count;
            state.generation = state.generation.wrapping_add(1);
        }
        if count >= self.threads {
            self.more.notify_all();
        } else {
            (1..count).for_each(|_| self.more.notify_one());
        }
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= count {
                break;
            }
            f(0, i);
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.work = None;
        while self.running.load(Ordering::Acquire) > 0 {
            state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Lock-free per-worker output slots for [`WatchPool::each_indexed`].
struct WorkerOutputs<T>(Vec<std::cell::UnsafeCell<T>>);

// SAFETY: Each `WatchPool` thread has a distinct `worker_id` in `0..len` and only
// accesses its own slot; `WatchPool::each_indexed` synchronizes all workers before
// the caller consumes the outputs.
unsafe impl<T: Send> Sync for WorkerOutputs<T> {}

impl<T: Default> WorkerOutputs<T> {
    fn new(workers: usize) -> Self {
        Self((0..workers).map(|_| std::cell::UnsafeCell::new(T::default())).collect())
    }

    /// # Safety
    /// Must only be called with a `worker_id` unique to the calling thread for the
    /// duration of the enclosing `WatchPool::each_indexed` batch.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slot_mut(&self, worker_id: usize) -> &mut T {
        // SAFETY: caller guarantees exclusive access per `worker_id`.
        unsafe { &mut *self.0[worker_id].get() }
    }

    fn into_vec(self) -> Vec<T> {
        self.0.into_iter().map(std::cell::UnsafeCell::into_inner).collect()
    }
}

/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Lock-free mirror of `Journal::kept` so `ask_early` yields immediately while
/// a submission's standing asking is active.
static JOURNAL_KEPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Cached address span `[base, end)` in 64 KiB pages: either a confirmed
/// `MEM_PRIVATE` VirtualAlloc reservation (`private: true`) or a negative-cached
/// non-private span such as `MEM_MAPPED`/`MEM_IMAGE` (`private: false`) so
/// `group_pieces` does not repeat `VirtualQuery` every submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Reservation {
    base: usize,
    end: usize,
    private: bool,
}

static RESERVATIONS: RwLock<Vec<Reservation>> = RwLock::new(Vec::new());

const DEFAULT_WATCH_GAP_KB: usize = 8 * 1024;
const MAX_WATCH_GAP_KB: usize = 1024 * 1024;

fn parse_watch_gap_pages(kb_raw: Option<&str>, mb_raw: Option<&str>) -> usize {
    let kb = if let Some(kb) = kb_raw.and_then(|v| v.trim().parse::<usize>().ok()) {
        kb.min(MAX_WATCH_GAP_KB)
    } else if let Some(mb) = mb_raw.and_then(|v| v.trim().parse::<usize>().ok()) {
        mb.min(1024) << 10
    } else {
        DEFAULT_WATCH_GAP_KB
    };
    (kb << 10) / PAGE
}

/// Maximum intra-reservation gap (in 64 KiB pages) bridged between ranges in
/// the same confirmed `MEM_PRIVATE` reservation (`VICTORY_WATCH_GAP_KB=<KiB>`
/// or `VICTORY_WATCH_GAP_MB=<MiB>`, default `8192` KiB / `8` MiB; `0`: only
/// touching/overlapping ranges merge).
fn watch_gap_pages() -> usize {
    #[cfg(test)]
    if let Some(pages) = WATCH_GAP_IN_TEST.get() {
        return pages;
    }
    static PAGES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGES.get_or_init(|| {
        let kb = std::env::var("VICTORY_WATCH_GAP_KB").ok();
        let mb = std::env::var("VICTORY_WATCH_GAP_MB").ok();
        parse_watch_gap_pages(kb.as_deref(), mb.as_deref())
    })
}

#[cfg(test)]
thread_local! {
    static WATCH_GAP_IN_TEST: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Queries Windows for the reservation span containing `page`: returns
/// `private: true` for `MEM_PRIVATE` VirtualAlloc reservations, `private: false`
/// for non-private mapped/image regions (negative-cached), or `None` for free
/// or sub-64 KiB regions.
fn query_reservation(page: usize) -> Option<Reservation> {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by `VirtualQuery`.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer of the right size.
    let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State == MEM_FREE {
        return None;
    }
    if info.Type != MEM_PRIVATE {
        let base_page = (info.BaseAddress as usize).div_ceil(PAGE);
        let end_page = (info.BaseAddress as usize).saturating_add(info.RegionSize) / PAGE;
        return (base_page <= page && page < end_page).then_some(Reservation {
            base: base_page,
            end: end_page,
            private: false,
        });
    }
    let alloc_base = info.AllocationBase as usize;
    if alloc_base == 0 {
        return None;
    }
    let mut end = (info.BaseAddress as usize).saturating_add(info.RegionSize);
    for _ in 0..256 {
        // SAFETY: plain data filled in by `VirtualQuery`.
        let mut next: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer of the right size.
        let m = unsafe { VirtualQuery(end as *const _, &mut next, size_of_val(&next)) };
        if m == 0
            || next.State == MEM_FREE
            || next.Type != MEM_PRIVATE
            || next.AllocationBase as usize != alloc_base
        {
            break;
        }
        let next_end = (next.BaseAddress as usize).saturating_add(next.RegionSize);
        if next_end <= end {
            break;
        }
        end = next_end;
    }
    let base_page = alloc_base / PAGE;
    let end_page = end / PAGE;
    (base_page <= page && page < end_page).then_some(Reservation {
        base: base_page,
        end: end_page,
        private: true,
    })
}

fn find_cached_reservation(res: &[Reservation], page: usize) -> Option<Reservation> {
    let idx = res.partition_point(|r| r.end <= page);
    res.get(idx).copied().filter(|r| r.base <= page && page < r.end)
}

fn cache_reservation(found: Reservation) {
    let mut res = RESERVATIONS.write().unwrap_or_else(|e| e.into_inner());
    res.retain(|r| !(r.base < found.end && found.base < r.end));
    let pos = res.partition_point(|r| r.base < found.base);
    res.insert(pos, found);
}

fn invalidate_reservations(first: usize, last: usize) {
    if first >= last {
        return;
    }
    hot_watch_invalidate(first, last);
    mark_write_epoch_range(first.saturating_mul(PAGE), last.saturating_mul(PAGE));
    {
        let res = RESERVATIONS.read().unwrap_or_else(|e| e.into_inner());
        let idx = res.partition_point(|r| r.end <= first);
        if res.get(idx).is_none_or(|r| r.base >= last) {
            return;
        }
    }
    let mut res = RESERVATIONS.write().unwrap_or_else(|e| e.into_inner());
    res.retain(|r| !(r.base < last && first < r.end));
}

/// Groups sorted, overlap-coalesced `pieces` into `(start_idx, end_idx, first, last)`
/// `GetWriteWatch` spans that never cross `MEM_PRIVATE` reservation boundaries,
/// skip confirmed non-private regions (`MEM_MAPPED`/`MEM_IMAGE`) so failing
/// `GetWriteWatch` calls do not thrash the negative cache, and bridge gaps up
/// to [`watch_gap_pages`] inside the same reservation.
fn group_pieces(pieces: &[(usize, usize)], allow_singletons: bool) -> Vec<(usize, usize, usize, usize)> {
    let gap = watch_gap_pages();
    let mut res_of: Vec<Option<Reservation>> = vec![None; pieces.len()];
    let mut non_private: Vec<bool> = vec![false; pieces.len()];
    let mut misses: Vec<usize> = Vec::new();
    {
        let cached = RESERVATIONS.read().unwrap_or_else(|e| e.into_inner());
        for (i, &(first, last)) in pieces.iter().enumerate() {
            if let Some(r) = find_cached_reservation(&cached, first) {
                if r.private && last <= r.end {
                    res_of[i] = Some(r);
                } else if !r.private && last <= r.end {
                    non_private[i] = true;
                }
            } else {
                misses.push(i);
            }
        }
    }
    if !misses.is_empty() {
        let mut last_found: Option<Reservation> = None;
        for i in misses {
            let (first, last) = pieces[i];
            let found = if let Some(r) = last_found
                && r.base <= first
                && first < r.end
            {
                Some(r)
            } else {
                let q = query_reservation(first);
                if let Some(r) = q {
                    cache_reservation(r);
                    last_found = Some(r);
                }
                q
            };
            if let Some(r) = found && last <= r.end {
                if r.private {
                    res_of[i] = Some(r);
                } else {
                    non_private[i] = true;
                }
            }
        }
    }
    let mut groups: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        if non_private[i] {
            i += 1;
            continue;
        }
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        if let Some(r) = res_of[i] {
            while k < pieces.len()
                && !non_private[k]
                && pieces[k].1 <= r.end
                && pieces[k].0 <= last.saturating_add(gap)
                && last.max(pieces[k].1) - first <= AHEAD_MAX
            {
                last = last.max(pieces[k].1);
                k += 1;
            }
        }
        if allow_singletons || k - i >= 2 {
            groups.push((i, k, first, last));
        }
        i = k;
    }
    groups
}

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`. Check before filtering/allocating/sorting.
    let existing_ahead = {
        let j = journal();
        if j.kept {
            return;
        }
        j.ahead.clone()
    };
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        // Not view memory known by its bits or already covered by `ahead`:
        // avoids redundant `GetWriteWatch(RESET)` calls.
        .filter(|r| {
            r.0 < r.1
                && !published(r.0, r.1)
                && !refused_as_view(r.0, r.1)
                && !ahead_covers(&existing_ahead, r.0 / PAGE, r.1.div_ceil(PAGE))
        })
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    coalesce_overlapping_pieces(&mut pieces);
    if pieces.len() < 2 {
        return;
    }
    let groups = group_pieces(&pieces, false);
    if groups.is_empty() {
        return;
    }
    let _held = AskingMany::hold(groups.iter().map(|&(_, _, first, last)| (first, last)).collect());
    let (mut asked, mut written) = (Vec::new(), Vec::new());
    for (i, k, _, _) in groups {
        ask_run(&pieces[i..k], &mut asked, &mut written);
    }
    sort_coalesce_ahead(&mut asked);
    note_hot(&written);
    if port_journal() {
        // Renderer port #7: the pages noted shared, under their parts; then
        // the number and the ranges, alone (no shared holder with a lower
        // number is left by then).
        let now = note_shared(&written);
        let mut j = journal_mut();
        j.ahead_asked = now;
        j.ahead = asked;
        return;
    }
    let mut j = journal_mut();
    j.ahead_asked = next_asking();
    j.ahead = asked;
    let now = j.ahead_asked;
    for &(page, parts) in &written {
        j.notes.note(page, parts, now);
    }
    drop(j);
    mark_write_epoch_pages(written.iter().map(|&(p, _)| p));
}

/// Renderer port #7: notes `written` (pages and their parts) as found by an
/// asking with a number taken now, holding the journal shared and their
/// parts for writing; the number.
fn note_shared(written: &[(usize, u16)]) -> u64 {
    let j = journal();
    let mut held = j.notes.hold(shards_of_pages(written.iter().map(|w| w.0)), true);
    let now = next_asking();
    for &(page, parts) in written {
        held.note(page, parts, now);
    }
    drop(held);
    drop(j);
    mark_write_epoch_pages(written.iter().map(|&(p, _)| p));
    port7_count(P7_SHARED_NOTES, 1);
    now
}

/// Whether the kernel would refuse to ask about `[start, end)` for its view
/// memory (renderer port #6): read-only by our hand, or known a view
/// ([`VIEWS`]). Such a range is left to [`writes`], which goes to the guards
/// at once; asked with others it failed their asking too, which was then
/// asked again by halves.
fn refused_as_view(start: usize, end: usize) -> bool {
    if !port_guard() || !light() {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let refused = PROTECTED.any(first, last) || VIEWS.any(first, last);
    if refused {
        port6_count(P6_NOT_ASKED, 1);
    }
    refused
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Whether a thread asks, between a submission's askings, about the memory
/// they found written (`VICTORY_ASK_EARLY=<ms>` between its rounds; off
/// unless given: in run cpu-2 the submissions' own askings were none the
/// cheaper for its 650 a frame, 12 to 18 ms where 9 were).
///
/// An asking's cost is the pages it finds written and resets, some 1.4
/// microseconds each, and the title writes 7,000 a frame: the first
/// submission that refreshes its 174 MB heap stood 8 to 10 ms in one call
/// (session play-1125-0203). Asked a little at a time while the title
/// records, the same pages are in the journal when the submission comes, and
/// its own asking finds those written since.
///
/// Nothing is seen later than before: a page goes into the journal as found
/// written by an asking of its own number, as from any thread that asks, and
/// the submission still asks for itself. A page the title writes again after
/// such a round faults once more.
fn early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = match std::env::var("VICTORY_ASK_EARLY") {
            Ok(v) => v.trim().parse::<f64>().unwrap_or(2.0),
            Err(_) => 0.0,
        };
        std::time::Duration::try_from_secs_f64(ms / 1000.0).ok().filter(|d| !d.is_zero())
    })
}

/// The pieces of memory (by [`EARLY_PIECE`] pages) that submissions' askings
/// found written lately: how many pages each, halved every round of 64.
static HOT: Mutex<BTreeMap<usize, u32>> = Mutex::new(BTreeMap::new());
/// The pieces asked in a round, at most.
const EARLY_MOST: usize = 24;
/// The pages of a piece: 2 MiB.
const EARLY_PIECE: usize = (2 << 20) / PAGE;
/// Told when a submission asked.
static HOT_MOVED: Condvar = Condvar::new();

/// Since the last frame line: pieces asked early, the microseconds of it.
static EARLY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`EARLY`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn early_asks() -> [u64; 2] {
    [0, 1].map(|i| EARLY[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// A submission's asking found `written` (pages).
fn note_hot(written: &[(usize, u16)]) {
    let Some(every) = early() else {
        return;
    };
    let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
    for (page, _) in written {
        let n = hot.entry(page / EARLY_PIECE).or_default();
        *n = (*n + 1).min(1 << 10);
    }
    HOT_MOVED.notify_one();
    drop(hot);
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-ask-early".into()).spawn(move || ask_early(every));
    });
}

/// Clamps hot 2 MiB `pieces` to cached `MEM_PRIVATE` reservations, merges
/// adjacent slices within the same reservation, and drains write-watch bits
/// into the journal without blocking active submissions. Returns the number
/// of 2 MiB piece-equivalents asked.
fn drain_early_pieces(pieces: &mut Vec<usize>) -> u64 {
    use std::sync::atomic::Ordering::Acquire;
    if pieces.is_empty() || JOURNAL_KEPT.load(Acquire) {
        return 0;
    }
    pieces.sort_unstable();
    pieces.dedup();
    // Clamp each hot 2 MiB bucket against known `MEM_PRIVATE` reservations and
    // merge adjacent spans inside the same reservation into one `GetWriteWatch`.
    let mut spans: Vec<(usize, usize, u64)> = Vec::new();
    {
        let cached = RESERVATIONS.read().unwrap_or_else(|e| e.into_inner());
        for &piece in pieces.iter() {
            let p_lo = piece * EARLY_PIECE;
            let p_hi = (piece + 1) * EARLY_PIECE;
            let mut idx = cached.partition_point(|r| r.end <= p_lo);
            while let Some(&r) = cached.get(idx) {
                if r.base >= p_hi {
                    break;
                }
                if !r.private {
                    idx += 1;
                    continue;
                }
                let lo = p_lo.max(r.base);
                let hi = p_hi.min(r.end);
                if lo < hi
                    && !published(lo * PAGE, hi * PAGE)
                    && !refused_as_view(lo * PAGE, hi * PAGE)
                {
                    match spans.last_mut() {
                        Some((prev_lo, prev_hi, count))
                            if *prev_hi == lo && hi - *prev_lo <= AHEAD_MAX && *prev_lo >= r.base =>
                        {
                            *prev_hi = hi;
                            *count += 1;
                        }
                        _ => spans.push((lo, hi, 1)),
                    }
                }
                idx += 1;
            }
        }
    }
    let mut asked = 0u64;
    let mut written: Vec<(usize, u16)> = Vec::new();
    for (from, to, count) in spans {
        if JOURNAL_KEPT.load(Acquire) {
            break;
        }
        let Some(_held) = Asking::try_hold(from, to) else {
            continue;
        };
        written.clear();
        if !kernel_written_into(from, to, &mut written) {
            invalidate_reservations(from, to);
            continue;
        }
        asked += count;
        if written.is_empty() {
            continue;
        }
        if port_journal() {
            let _ = note_shared(&written);
        } else {
            let mut j = journal_mut();
            let now = next_asking();
            for &(page, parts) in &written {
                j.notes.note(page, parts, now);
            }
            drop(j);
            mark_write_epoch_pages(written.iter().map(|&(p, _)| p));
        }
    }
    asked
}

fn ask_early(every: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    // So many rounds without a submission's asking: wait for one.
    const IDLE: u32 = 100;
    let mut round = 0u32;
    let mut idle = 0;
    let mut last = 0u64;
    loop {
        std::thread::sleep(every);
        let mut pieces: Vec<usize> = {
            let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
            let now = ASKED.load(Relaxed);
            idle = if now == last { idle + 1 } else { 0 };
            while idle > IDLE {
                hot = HOT_MOVED.wait(hot).unwrap_or_else(|e| e.into_inner());
                idle = 0;
            }
            round += 1;
            if round % 64 == 0 {
                hot.retain(|_, n| {
                    *n /= 2;
                    *n > 0
                });
            }
            // The most written: a piece with a few pages is as dear to ask as
            // one with many.
            let mut most: Vec<(u32, usize)> = hot.iter().filter(|h| *h.1 >= 16).map(|(&p, &n)| (n, p)).collect();
            most.sort_unstable_by(|a, b| b.cmp(a));
            most.into_iter().take(EARLY_MOST).map(|(_, p)| p).collect()
        };
        let began = std::time::Instant::now();
        let asked = drain_early_pieces(&mut pieces);
        EARLY[0].fetch_add(asked, Relaxed);
        EARLY[1].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
        last = ASKED.load(Relaxed);
    }
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    // Renderer port #7: nothing to change is seen shared.
    if port_journal() {
        let j = journal();
        if j.kept || j.ahead.is_empty() {
            port7_count(P7_SKIPPED, 1);
            return;
        }
    }
    let mut j = journal_mut();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) in a single pass, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1 && !published(r.0, r.1) && !refused_as_view(r.0, r.1))
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.is_empty() {
        ask_once_end();
        return;
    }
    pieces.sort_unstable();
    coalesce_overlapping_pieces(&mut pieces);
    let mut groups = group_pieces(&pieces, true);
    let _held = AskingMany::hold(groups.iter().map(|&(_, _, first, last)| (first, last)).collect());
    let (mut asked, mut written) = (Vec::with_capacity(groups.len()), Vec::new());
    let mut outcomes: Vec<(usize, usize, bool, bool)> = Vec::with_capacity(groups.len());
    let threads = watch_threads();
    if threads >= 2 && groups.len() >= 2 {
        // Dispatch largest spans first to minimize parallel straggler tail.
        groups.sort_unstable_by(|a, b| (b.3 - b.2).cmp(&(a.3 - a.2)));
        let suppress = hot_watch_flags_for_groups(&groups);
        let pool = WatchPool::get(threads);
        type WorkerOut = (Vec<(usize, usize)>, Vec<(usize, u16)>, Vec<(usize, usize, bool, bool)>);
        let outputs: WorkerOutputs<WorkerOut> = WorkerOutputs::new(pool.threads);
        pool.each_indexed(groups.len(), &|worker_id, g| {
            let (i, k, first, last) = groups[g];
            let sup = suppress[g];
            let flags = u32::from(!sup);
            // SAFETY: `worker_id` is unique per thread inside `each_indexed`.
            let (g_asked, g_written, g_outcomes) = unsafe { outputs.slot_mut(worker_id) };
            if k - i >= 2 {
                ask_all_run_adaptive(&pieces[i..k], flags, sup, g_asked, g_written, g_outcomes);
            } else {
                let before = g_written.len();
                if kernel_written_into_flags(first, last, flags, g_written) {
                    g_asked.push((first, last));
                    g_outcomes.push((first, last, g_written.len() > before, sup));
                } else {
                    invalidate_reservations(first, last);
                }
            }
        });
        for (g_asked, g_written, g_outcomes) in outputs.into_vec() {
            asked.extend(g_asked);
            written.extend(g_written);
            outcomes.extend(g_outcomes);
        }
    } else {
        let suppress = hot_watch_flags_for_groups(&groups);
        for (&(i, k, first, last), &sup) in groups.iter().zip(&suppress) {
            let flags = u32::from(!sup);
            if k - i >= 2 {
                ask_all_run_adaptive(&pieces[i..k], flags, sup, &mut asked, &mut written, &mut outcomes);
            } else {
                let before = written.len();
                if kernel_written_into_flags(first, last, flags, &mut written) {
                    asked.push((first, last));
                    outcomes.push((first, last, written.len() > before, sup));
                } else {
                    invalidate_reservations(first, last);
                }
            }
        }
    }
    hot_watch_record(&outcomes);
    sort_coalesce_ahead(&mut asked);
    note_hot(&written);
    if port_journal() {
        let now = note_shared(&written);
        let mut j = journal_mut();
        j.ahead_asked = now;
        let kept = !asked.is_empty();
        j.kept = kept;
        JOURNAL_KEPT.store(kept, std::sync::atomic::Ordering::Release);
        j.ahead = asked;
        return;
    }
    let mut j = journal_mut();
    let now = next_asking();
    j.ahead_asked = now;
    for &(page, parts) in &written {
        j.notes.note(page, parts, now);
    }
    let kept = !asked.is_empty();
    j.kept = kept;
    JOURNAL_KEPT.store(kept, std::sync::atomic::Ordering::Release);
    j.ahead = asked;
    drop(j);
    mark_write_epoch_pages(written.iter().map(|&(p, _)| p));
}

/// Like [`ask_run`], but also asks singleton ranges left after a split so
/// `ask_once` finishes every write-watched range in one pass, honoring
/// adaptive `flags` on the unsplit group and falling back to reset flags on splits.
fn ask_all_run_adaptive(
    pieces: &[(usize, usize)],
    flags: u32,
    suppressed: bool,
    asked: &mut Vec<(usize, usize)>,
    written: &mut Vec<(usize, u16)>,
    outcomes: &mut Vec<(usize, usize, bool, bool)>,
) {
    if pieces.is_empty() {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    let before = written.len();
    if kernel_written_into_flags(first, last, flags, written) {
        asked.push((first, last));
        outcomes.push((first, last, written.len() > before, suppressed));
        return;
    }
    invalidate_reservations(first, last);
    if pieces.len() < 2 {
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_all_run_adaptive(a, 1, false, asked, written, outcomes);
    ask_all_run_adaptive(b, 1, false, asked, written, outcomes);
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    // Renderer port #7: none standing is seen shared.
    if port_journal() && !journal().kept {
        port7_count(P7_SKIPPED, 1);
        return;
    }
    let mut j = journal_mut();
    if j.kept {
        j.kept = false;
        JOURNAL_KEPT.store(false, std::sync::atomic::Ordering::Release);
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    // Renderer port #7: with no asking standing there is nothing to note,
    // which is seen shared. (Standing, its number moves: alone, as before.)
    if port_journal() && !journal().kept {
        port7_count(P7_SKIPPED, 1);
        mark_write_epoch_range(at, at.saturating_add(len));
        return;
    }
    let mut j = journal_mut();
    if !j.kept {
        drop(j);
        mark_write_epoch_range(at, at.saturating_add(len));
        return;
    }
    let now = next_asking();
    // What is copied from now on counts as copied after these writes (fork
    // PR 410): answers older than the writes in them made every refresh
    // until the next asking copy the pages again.
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.notes.note(page, WHOLE, now);
    }
    let aliases = elsewhere(at, len);
    for (_, others) in &aliases {
        for &other in others {
            j.notes.note(other / PAGE, WHOLE, now);
        }
    }
    drop(j);
    mark_write_epoch_range(at, at.saturating_add(len));
    for (_, others) in aliases {
        for other in others {
            mark_write_epoch_range(other, other.saturating_add(PAGE));
        }
    }
}

/// The blocks of `[at, at + len)` whose pages the title shows at other
/// addresses too, and there.
fn elsewhere(at: usize, len: usize) -> Vec<(usize, Vec<usize>)> {
    if !view_guard() || len == 0 {
        return Vec::new();
    }
    crate::inplace::shown_elsewhere(at / BLOCK * BLOCK, at.saturating_add(len))
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`]. Adds what was
/// asked to `asked` and the pages found written to `written`.
fn ask_run(pieces: &[(usize, usize)], asked: &mut Vec<(usize, usize)>, written: &mut Vec<(usize, u16)>) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if kernel_written_into(first, last, written) {
        asked.push((first, last));
        return;
    }
    invalidate_reservations(first, last);
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(a, asked, written);
    ask_run(b, asked, written);
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal();
    ahead_covers(&j.ahead, first, last).then(|| (j.ahead_asked, j.notes.range(first, last)))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: the journal has it.
    // (The standing number changes only with the journal held alone, so its
    // notes are all in: a part at a time will do.)
    let ahead = |j: &Journal| ahead_covers(&j.ahead, first, last).then(|| (j.ahead_asked, j.notes.range(first, last)));
    // A fresh number and the notes of the range, its parts held meanwhile
    // (renderer port #7, switch or not: who notes shared takes its number
    // holding the parts it notes in).
    let fresh = || {
        let j = journal();
        let held = j.notes.hold(shards_of(first, last), false);
        (next_asking(), held.range(first, last))
    };
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    // Read-only since an asking noted it: the bits first, then the journal.
    if guarded && published(start, end) {
        #[cfg(feature = "diagnostics")]
        {
            tally(2, 1);
            crate::faultbook::note(crate::faultbook::What::Known, first, last, 0);
        }
        return Some(fresh());
    }
    // Renderer port #6: view memory, which the kernel would refuse, and the
    // caller does not ask the guards (textures, which then guard by a way of
    // their own): no kernel call, ~300 a frame (run guard-2).
    if !guarded && refused_as_view(start, end) {
        return None;
    }
    // Asked, and the answer noted, with the range held: nothing else asks
    // about any of it in between.
    let _range = Asking::hold(first, last);
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    let mut guards_asked = false;
    // View memory read-only by our hand or known as a view (`VIEWS`): the
    // kernel would refuse `GetWriteWatch`, so skip straight to `guard()`.
    let protected = guarded && light() && PROTECTED.any_acquire(first, last);
    let viewed = guarded && !protected && light() && VIEWS.any_acquire(first, last);
    if viewed && port_guard() {
        port6_count(P6_NOT_ASKED, 1);
    }
    let refused = protected || viewed;
    let pages = match if refused { None } else { kernel_written(first, last) } {
        Some(pages) => pages,
        None => {
            guards_asked = true;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let mut written = guard(first * PAGE, last * PAGE).map(|pages| pages.into_iter().map(|p| (p, WHOLE)).collect::<Vec<_>>());
            if written.is_none() && viewed {
                // Not a view after all (other memory there now): forgotten as
                // one, and the kernel asked as before.
                (first..last).for_each(|block| VIEWS.clear(block));
                if let Some(pages) = kernel_written(first, last) {
                    guards_asked = false;
                    written = Some(pages);
                }
            }
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let us = guarding.elapsed().as_micros() as u64;
                add(C::GuardUs, us);
                if written.is_some() {
                    add(C::Watches, 1);
                    add(C::WatchUs, us);
                    slow(us);
                }
            }
            written?
        }
    };
    if pages.is_empty() {
        // Nothing to note: the others may go on reading.
        return Some(fresh());
    }
    let answer = if port_journal() {
        // Renderer port #7: shared, the range's parts held for writing (the
        // range is held too: no other asking notes any of it meanwhile).
        let j = journal();
        let mut held = j.notes.hold(shards_of(first, last) | shards_of_pages(pages.iter().map(|p| p.0)), true);
        let now = next_asking();
        for &(page, parts) in &pages {
            held.note(page, parts, now);
        }
        port7_count(P7_SHARED_NOTES, 1);
        let r = held.range(first, last);
        drop(held);
        drop(j);
        mark_write_epoch_pages(pages.iter().map(|&(p, _)| p));
        (now, r)
    } else {
        let mut j = journal_mut();
        let now = next_asking();
        for &(page, parts) in &pages {
            j.notes.note(page, parts, now);
        }
        let r = j.notes.range(first, last);
        drop(j);
        mark_write_epoch_pages(pages.iter().map(|&(p, _)| p));
        (now, r)
    };
    // Noted: what the guards made clean of it is known by its bits from
    // now on. (The range is still held: no other asking of any of it came
    // between.)
    if guards_asked && light() {
        publish(&pages.iter().map(|p| p.0).collect::<Vec<_>>());
    }
    Some(answer)
}

fn covered_in_journal(j: &Journal, start: usize, end: usize, first: usize, last: usize, prefer_published: bool) -> bool {
    if prefer_published {
        if published(start, end) {
            #[cfg(feature = "diagnostics")]
            crate::faultbook::note(crate::faultbook::What::Known, first, last, 1);
            return true;
        }
        return ahead_covers(&j.ahead, first, last);
    }
    if ahead_covers(&j.ahead, first, last) {
        return true;
    }
    if published(start, end) {
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Known, first, last, 1);
        return true;
    }
    false
}

/// Fast path for [`nothing_new_scan`] when the caller already proved lock-free
/// that `[start, end)` lies inside a range copied whole at asking `ranged > 0`.
pub(crate) fn nothing_new_ranged_scan(
    scan: &mut JournalScan<'_>,
    start: usize,
    end: usize,
    ranged: u64,
    prefer_published: bool,
) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if !covered_in_journal(&scan.journal.0, start, end, first, last, prefer_published) {
        return false;
    }
    scan.all_below(first, last, ranged, |_, write| write <= ranged)
}

/// Whether nothing of `[start, end)` was written that a copy with `copied` and
/// `ranged` does not hold, reusing a [`JournalScan`] to cache the active shard
/// read lock across spans.
pub(crate) fn nothing_new_scan(
    scan: &mut JournalScan<'_>,
    start: usize,
    end: usize,
    base: usize,
    copied: &[u64],
    ranged: u64,
    prefer_published: bool,
) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if !covered_in_journal(&scan.journal.0, start, end, first, last, prefer_published) {
        return false;
    }
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    let floor = if ranged != 0 {
        ranged
    } else {
        let mut min_taken = u64::MAX;
        for &at in taken {
            if at == 0 {
                return false;
            }
            if at < min_taken {
                min_taken = at;
            }
        }
        min_taken
    };
    scan.all_below(first, last, floor, |page, write| write <= taken[page - first].max(ranged))
}

/// Directly finds stale `(page, taken)` pairs in `[start, end)` from an
/// already-held [`JournalRead`] when covered by `ahead` or `PUBLISHED`,
/// avoiding an intermediate `written` Vec allocation and binary search.
pub(crate) fn stale_pages_in_journal(
    j: &JournalRead,
    start: usize,
    end: usize,
    base: usize,
    copied: &[u64],
) -> Option<(u64, Vec<(usize, u64)>)> {
    let (req_first, req_last) = (start / PAGE, end.div_ceil(PAGE));
    let in_ahead = ahead_covers(&j.0.ahead, req_first, req_last);
    if !in_ahead && !published(start, end) {
        return None;
    }
    let first = req_first.max(base);
    let last = req_last.min(base + copied.len());
    let mut out = Vec::new();
    let scan_piece = |shard: &Shard, piece: usize, p_lo: usize, p_hi: usize, out: &mut Vec<(usize, u64)>| {
        if let Some(np) = shard.find_piece(piece) {
            let base_p = piece << SHARD_SHIFT;
            for p in p_lo..p_hi {
                let at = copied[p - base];
                if at == 0 || np.at[p - base_p] > at {
                    out.push((p, at));
                }
            }
        } else {
            for p in p_lo..p_hi {
                if copied[p - base] == 0 {
                    out.push((p, 0));
                }
            }
        }
    };
    if in_ahead {
        if first < last {
            let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
            for piece in a..=b {
                let p_lo = first.max(piece << SHARD_SHIFT);
                let p_hi = last.min((piece + 1) << SHARD_SHIFT);
                let shard = piece % SHARDS;
                if j.0.notes.shard_max(shard) == 0 {
                    for p in p_lo..p_hi {
                        if copied[p - base] == 0 {
                            out.push((p, 0));
                        }
                    }
                } else {
                    let s = j.0.notes.read(shard);
                    scan_piece(&s, piece, p_lo, p_hi, &mut out);
                }
            }
        }
        return Some((j.0.ahead_asked, out));
    }
    #[cfg(feature = "diagnostics")]
    {
        tally(2, 1);
        crate::faultbook::note(crate::faultbook::What::Known, req_first, req_last, 0);
    }
    let held = j.0.notes.hold(shards_of(req_first, req_last), false);
    let now = next_asking();
    if first < last {
        let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
        for piece in a..=b {
            let p_lo = first.max(piece << SHARD_SHIFT);
            let p_hi = last.min((piece + 1) << SHARD_SHIFT);
            scan_piece(held.shard(p_lo), piece, p_lo, p_hi, &mut out);
        }
    }
    Some((now, out))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut new: Vec<(usize, u64)> = Vec::new();
        for (page, write) in pages {
            let last = seen.get(&page).copied();
            if last.is_some_and(|last| last >= write) || (page * PAGE).max(from) >= ((page + 1) * PAGE).min(to) {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            new.push((page, last.unwrap_or(0)));
        }
        // Of a page taken before, the parts written since, where they are
        // known (`VICTORY_TAKE_PARTS=0`: the page).
        let parts = if crate::toggled!("VICTORY_TAKE_PARTS") { parts_written(&new) } else { vec![WHOLE; new.len()] };
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (&(page, last), &parts) in new.iter().zip(&parts) {
            let mut parts = if last == 0 { WHOLE } else { parts };
            while parts != 0 {
                let first = parts.trailing_zeros() as usize;
                let past = first + (!(parts >> first)).trailing_zeros() as usize;
                parts &= if past >= PARTS { 0 } else { u16::MAX << past };
                let (start, end) = ((page * PAGE + first * KERNEL_PAGE).max(from), (page * PAGE + past.min(PARTS) * KERNEL_PAGE).min(to));
                if start >= end {
                    continue;
                }
                match ranges.last_mut() {
                    Some((_, last)) if *last == start => *last = end,
                    _ => ranges.push((start, end)),
                }
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the refreshes of host copies go the light way
/// (`VICTORY_LIGHT_REFRESH=0`: as before): view memory nothing wrote since
/// it was last asked about is known by its bits ([`PUBLISHED`]), a copy
/// nothing of whose range changed is told so before any thread is woken
/// for it (`crate::host`, `Shadow::nothing_new`), and the ends of a range,
/// pages the range holds in part, are not copied again at every refresh
/// (`Shadow::copy_stale`).
///
/// A frame asked about view memory some 1,500 times, for nearly all of it
/// to hear that nothing was written, on up to seven threads at once, each
/// asking under the guards' one lock and behind [`ASKING`]'s: 94 ms a frame
/// of all threads' time (run record-1), most of it waiting.
pub(crate) fn light() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIGHT_REFRESH").is_none_or(|v| v != "0"))
}

/// Renderer port #6 (`VICTORY_PORT_GUARD=1`, off by default; read here, and
/// A/B-able in one run with `VICTORY_AB=<s>,VICTORY_PORT_GUARD`): the guards'
/// one lock no longer makes the title's write faults, and the refreshes'
/// looks at view memory, wait for one another.
///
/// What it is with the switch: [`GUARDS`] is a lock for readers and one
/// writer. The sets of blocks in it are of atomic words, so what changes a
/// block's state needs no lock of all of them: a block's own claim does
/// ([`BUSY`]). A fault, a refresh guarding its range and a block made known
/// by its bits ([`publish`]) take the lock shared and the claims of the
/// blocks they change, so they wait only for those who change the same
/// block (a `VirtualProtect` of it) and for the few things that still take
/// the lock alone (a mapping changed, a kernel read, memory freed, the holes
/// looked at, ...), which take no claims: holding the lock alone, nobody
/// holds one. Who waits for a claim holds no other claim, or only claims of
/// lower blocks (faults take theirs in order); who holds claims and wants
/// more only tries for them (a refresh: what it cannot claim it leaves as it
/// is, written, told so at the next asking), so no two wait for each other.
///
/// Also with the switch:
/// - view memory is not asked of the kernel, which refuses it
///   ([`VIEWS`]; ~300 refused askings a frame);
/// - the planning of a submission's refreshes is made beside the submitting
///   thread ([`port_guard_plan`]);
/// - only if asked for (`VICTORY_GUARD_HOT=<ms>`, [`hot_rest`]): a block the
///   title writes again soon after each time it was taken rests longer.
///   Measured worse (run guard-6: 45.6 -> 47.8 ms a frame): off by default.
pub(crate) fn port_guard() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_GUARD") {
        return on;
    }
    #[cfg(test)]
    if PORT_GUARD_IN_TEST.get() {
        return true;
    }
    // Read here: the switch of renderer port #6.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_GUARD").is_ok_and(|v| v.trim() == "1"))
}

/// With renderer port #6: whether a submission's look at which refreshes
/// copy nothing (`Shadow::nothing_new`, before the threads are woken for the
/// others) is made on the threads beside the submitting one, as it already
/// was for more than 256 refreshes (`VICTORY_PORT_GUARD_PLAN=0`: on the
/// submitting thread, as before; A/B-able by that name).
#[allow(dead_code)]
pub(crate) fn port_guard_plan() -> bool {
    if !port_guard() {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_GUARD_PLAN") {
        return on;
    }
    // Read here: renderer port #6's planning beside the submitting thread.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_GUARD_PLAN").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// [`hot_rest`] in milliseconds, for what this thread does (`None`: env/default, `Some(0)`: off).
    pub(crate) static HOT_IN_TEST: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
    /// [`port_guard`], for what this thread does.
    pub(crate) static PORT_GUARD_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The blocks a thread holding [`GUARDS`] shared is changing the state of
/// (renderer port #6, [`port_guard`]): a bit each, set by who changes it
/// ([`Claims`]).
static BUSY: Bits = Bits::new();

/// Claims of blocks ([`BUSY`]), let go when dropped.
/// Stores up to 8 claims inline to avoid per-fault heap allocations.
struct Claims {
    inline: [usize; 8],
    len: usize,
    overflow: Vec<usize>,
}

impl Claims {
    fn push(&mut self, block: usize) {
        if self.len < self.inline.len() {
            self.inline[self.len] = block;
            self.len += 1;
        } else {
            self.overflow.push(block);
        }
    }

    /// The claims of `blocks` (in order, each once), waited for one after
    /// the other: who waits holds only claims of lower blocks.
    fn wait_for(blocks: &[usize]) -> Claims {
        debug_assert!(blocks.windows(2).all(|w| w[0] < w[1]));
        let mut claims = Claims::none();
        if blocks.len() > claims.inline.len() {
            claims.overflow.reserve(blocks.len() - claims.inline.len());
        }
        let mut spun = 0u64;
        for &block in blocks {
            let mut tries = 0u32;
            while !BUSY.try_set(block) {
                tries += 1;
                spun += 1;
                if tries < 64 {
                    std::hint::spin_loop();
                } else {
                    // Another thread's `VirtualProtect` of the block.
                    std::thread::yield_now();
                }
            }
            claims.push(block);
        }
        if spun != 0 {
            port6_count(P6_CLAIM_SPINS, spun);
        }
        claims
    }

    /// The claim of `block` if it is free; nothing waited for.
    fn try_one(&mut self, block: usize) -> bool {
        let got = BUSY.try_set(block);
        if got {
            self.push(block);
        } else {
            port6_count(P6_CLAIMS_MISSED, 1);
        }
        got
    }

    fn none() -> Claims {
        Claims { inline: [0; 8], len: 0, overflow: Vec::new() }
    }

    fn holds(&self, block: usize) -> bool {
        self.inline[..self.len].contains(&block) || self.overflow.contains(&block)
    }
}

impl Drop for Claims {
    fn drop(&mut self) {
        for &block in &self.inline[..self.len] {
            BUSY.clear(block);
        }
        for &block in &self.overflow {
            BUSY.clear(block);
        }
    }
}

/// Renderer port #6's counts since the last line: faults opened under the
/// shared lock; nanoseconds they waited for the lock; refreshes' guards
/// done shared; blocks a refresh could not claim (left written); spins for
/// claims; blocks rested longer for being hot; kernel askings left out for
/// view memory; blocks found hot.
static PORT6: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
const P6_FAULTS: usize = 0;
const P6_FAULT_WAIT_NS: usize = 1;
const P6_GUARDS: usize = 2;
const P6_CLAIMS_MISSED: usize = 3;
const P6_CLAIM_SPINS: usize = 4;
const P6_RESTED_HOT: usize = 5;
const P6_NOT_ASKED: usize = 6;
const P6_HOT: usize = 7;

fn port6_count(what: usize, n: u64) {
    PORT6[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Renderer port #6's line of the frame lines, per frame.
pub(crate) fn port_guard_line(frames: f64) {
    let n: Vec<f64> = PORT6.iter().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames).collect();
    if !port_guard() && n.iter().all(|&v| v == 0.0) {
        return;
    }
    tracing::info!(
        "renderer port 6 (VICTORY_PORT_GUARD=1) per frame: faults opened under the shared lock {:.1} ({:.2} ms waiting for it), \
         refreshes' guards done shared {:.1}, blocks left written for want of their claim {:.1}, spins for claims {:.0}; \
         blocks found hot {:.1}, rests longer for it {:.1} (VICTORY_GUARD_HOT); kernel askings left out for view memory {:.1}",
        n[P6_FAULTS],
        n[P6_FAULT_WAIT_NS] / 1e6,
        n[P6_GUARDS],
        n[P6_CLAIMS_MISSED],
        n[P6_CLAIM_SPINS],
        n[P6_HOT],
        n[P6_RESTED_HOT],
        n[P6_NOT_ASKED],
    );
}

/// The blocks known to be view memory (renderer port #6): guarded by us
/// once, or found a view by [`guard`]. The kernel refuses to ask about them
/// (its write watch is not kept for views), and with the switch such a
/// range goes to the guards at once. Forgotten when the memory is unmapped
/// or freed; a block wrongly here is only looked at by the guards, which
/// find it no view ([`is_view`]) and say the range is not watched.
static VIEWS: Bits = Bits::new();

/// A bit per block of the address space, in pieces made when first needed
/// and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; Bits::PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; Bits::WORDS];

impl Bits {
    const WORDS: usize = 1 << 10;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 15;

    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Bits::PIECES])
    }

    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / Bits::WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / Bits::WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { std::sync::atomic::AtomicU64::new(0) }; Bits::WORDS]);
            let piece = Box::into_raw(piece);
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now, and another thread's is there.
                drop(unsafe { Box::from_raw(piece) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % Bits::WORDS].fetch_or(1 << (block % 64), SeqCst);
        }
    }

    /// Sets it if it was not set: whether it was not.
    fn try_set(&self, block: usize) -> bool {
        self.set_was(block) == Some(false)
    }

    /// Sets it; whether it was set before (`None`: out of the address space).
    fn set_was(&self, block: usize) -> Option<bool> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let place = self.0.get(word / Bits::WORDS)?;
        if place.load(Acquire).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { std::sync::atomic::AtomicU64::new(0) }; Bits::WORDS]);
            let piece = Box::into_raw(piece);
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now, and another thread's is there.
                drop(unsafe { Box::from_raw(piece) });
            }
        }
        let bit = 1 << (block % 64);
        self.piece(word).map(|piece| piece[word % Bits::WORDS].fetch_or(bit, SeqCst) & bit != 0)
    }

    /// Whether it was set; it is not now.
    fn clear_was(&self, block: usize) -> bool {
        let word = block / 64;
        let bit = 1 << (block % 64);
        self.piece(word)
            .is_some_and(|piece| piece[word % Bits::WORDS].fetch_and(!bit, std::sync::atomic::Ordering::SeqCst) & bit != 0)
    }

    /// Those set of `[first, last)`, in order (`last` at most the address
    /// space's end).
    fn set_in(&self, first: usize, last: usize) -> impl Iterator<Item = usize> + '_ {
        let last = last.min(Bits::PIECES * Bits::WORDS * 64).max(first);
        let mut word = first / 64;
        let end = last.div_ceil(64);
        let mut bits = 0u64;
        std::iter::from_fn(move || {
            loop {
                if bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    return Some((word - 1) * 64 + bit);
                }
                if word >= end {
                    return None;
                }
                match self.piece(word) {
                    Some(piece) => {
                        bits = piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst) & in_word(word * 64, first, last);
                        word += 1;
                    }
                    None => word = (word / Bits::WORDS + 1) * Bits::WORDS,
                }
            }
        })
    }

    /// How many are set.
    #[cfg(test)]
    fn count(&self) -> usize {
        self.0
            .iter()
            .filter_map(|p| {
                // SAFETY: null, or a piece made by `set`, which is never freed.
                unsafe { p.load(std::sync::atomic::Ordering::Acquire).as_ref() }
            })
            .map(|piece| piece.iter().map(|w| w.load(std::sync::atomic::Ordering::SeqCst).count_ones() as usize).sum::<usize>())
            .sum()
    }

    /// Sixty-four of them.
    fn word(&self, word: usize) -> u64 {
        self.piece(word).map_or(0, |piece| piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Whether it was set, and is no more.
    fn take(&self, block: usize) -> bool {
        self.any(block, block + 1) && {
            let (word, bit) = (block / 64, 1 << (block % 64));
            self.piece(word)
                .is_some_and(|piece| piece[word % Bits::WORDS].fetch_and(!bit, std::sync::atomic::Ordering::SeqCst) & bit != 0)
        }
    }

    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            // Ahead of the protection's change, and seen by whoever the
            // writer tells of its write.
            piece[word % Bits::WORDS].fetch_and(!(1 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether a bit of `[first, last)` is set.
    fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                // None of the piece's.
                None => block = (word / Bits::WORDS + 1) * Bits::WORDS * 64,
            }
        }
        false
    }

    /// Lock-free `Acquire`-ordered check whether any bit of `[first, last)` is
    /// set (pairs with `Release`/`SeqCst` stores in `Bits::set` before
    /// `VirtualProtect(PAGE_READONLY)`).
    #[inline]
    fn any_acquire(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::Acquire) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                None => block = (word / Bits::WORDS + 1) * Bits::WORDS * 64,
            }
        }
        false
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::Acquire) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks that are read-only by our hand, not written since, and of
/// which the journal holds every write from before that: for those, what
/// [`writes`] answers is in the journal, and neither the kernel, nor the
/// guards' lock, nor [`ASKING`] need be asked.
///
/// Why that is so: a block's bit is set under the guards' lock while the
/// block is clean, by the asking that made it clean, after that asking noted
/// in the journal what it found ([`publish`]). It is cleared under the same
/// lock, before the block becomes writable again ([`open`]) or is forgotten
/// ([`title_memory_remapped`]). So while the bit is set nothing can have
/// written the block through this address, and who sees the bit and looks
/// at the journal after that finds every earlier write in it. A write that
/// comes while the bit is looked at is no different from one that comes just
/// after an asking: the next asking finds the block open.
///
/// So too for the blocks of placeholders that were copied as zeros
/// (`Guards::held`): nothing can be there before the runtime maps it, and
/// it tells of that ([`title_memory_remapped`]) before the title is.
static PUBLISHED: Bits = Bits::new();

/// The blocks read-only by our hand, or about to be: set under the guards'
/// lock ahead of the protection, cleared under it when the block is open
/// again. Who finds none set has no guard to open and need not wait for the
/// lock ([`writing`]): some 700 pieces a frame are landed in memory that is
/// no view.
static PROTECTED: Bits = Bits::new();

/// The blocks the title's bytes were put in beside their guards
/// ([`put_beside`]) and that no asking has told as written since. Such a
/// block is as it was else: read-only, clean, and an asking that finds its
/// bit here takes the bit and tells it as written.
///
/// The bit is set when the bytes are in, at every view of the page, and
/// [`PUBLISHED`]'s is cleared after that. So an asking either finds the bit
/// or came before the title's thread went on from its write; and who takes
/// the bit reads the bytes after.
static BESIDE: Bits = Bits::new();

/// How much was put into a block beside its guard since it was last told as
/// written (`crate::titlewrite::MOST`), by a few bits of its number.
static PUT: [std::sync::atomic::AtomicU32; 1 << 13] = [const { std::sync::atomic::AtomicU32::new(0) }; 1 << 13];

fn put_count(block: usize) -> &'static std::sync::atomic::AtomicU32 {
    &PUT[block % PUT.len()]
}

/// Whether [`put_beside`] put it there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Beside {
    Done,
    /// The block has had its share.
    Full,
    /// Not guarded, or there is no other way to its page, or the runtime is
    /// changing a mapping.
    Cannot,
}

/// Whether a block of `[at, at + len)` is (or is about to be) read-only by
/// our hand. `len` is not 0.
#[inline]
pub(crate) fn guards_any(at: usize, len: usize) -> bool {
    PROTECTED.any_acquire(at / BLOCK, (at.saturating_add(len) - 1) / BLOCK + 1)
}

/// The title's bytes for a guarded block, put in by `put` at our own view of
/// the same page: it is given what to add to an address of the block (as by
/// `wrapping_add`). `weight`: how much of the block's share it is.
pub(crate) fn put_beside(block: usize, weight: u32, put: impl FnOnce(usize)) -> Beside {
    use std::sync::atomic::Ordering::Relaxed;
    if !light() || !view_guard() || !PROTECTED.any(block, block + 1) {
        return Beside::Cannot;
    }
    let count = put_count(block);
    if count.load(Relaxed) + weight > crate::titlewrite::MOST {
        return Beside::Full;
    }
    // No mapping changes until all is told: a view of the page that came
    // after the look at them would not be.
    let Some(_reading) = title_memory_reading_now() else {
        return Beside::Cannot;
    };
    let Some((page, shown)) = crate::inplace::beside_page(block * BLOCK) else {
        return Beside::Cannot;
    };
    put(page.wrapping_sub(block * BLOCK));
    count.fetch_add(weight, Relaxed);
    for view in shown.iter().filter(|&&view| view != 0) {
        BESIDE.set(view / BLOCK);
    }
    for view in shown.iter().filter(|&&view| view != 0) {
        PUBLISHED.clear(view / BLOCK);
    }
    Beside::Done
}

/// A faulting store of the title's leaves `block` as it is: whether it may be
/// left out (until that has been so often, since the block was taken, that
/// the faults cost more than opening it).
pub(crate) fn nothing_put(block: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let count = put_count(block);
    count.load(Relaxed) < crate::titlewrite::MOST && count.fetch_add(1, Relaxed) < crate::titlewrite::MOST
}

/// The title is about to write block by block around `at`: as by a fault.
pub(crate) fn opened_for_writing(at: usize) {
    open_faulted(at);
}

/// Opens the guarded block at `at` for a write of the title's that faulted
/// there (or is about to be made), and notes it opened: how many blocks were
/// opened (0: it was not guarded, or another thread opened it first), and how
/// long the guards' lock was waited for.
///
/// Uses per-16-MiB sharded `GUARD_SHARD_LOCKS` (sorted ascending when aliases
/// span multiple shards) so cross-region VEH faults never block one another.
/// When `>= 3` adjacent 64 KiB guarded blocks fault sequentially within a
/// frame window, unprotects the contiguous ahead span (`1.5 MiB`) as
/// write-through for `120 ms` while tracking it dirty via `wrote()`/journal.
fn open_faulted(at: usize) -> (usize, std::time::Duration) {
    let block = at / BLOCK;
    let began = std::time::Instant::now();
    let (opened, waited) = if !port_guard() {
        let mut mask = 1u64 << guard_shard_of(block);
        let mut waited = std::time::Duration::ZERO;
        let mut first_lock = true;
        let opened = loop {
            let g = guard_shards_write(mask);
            if first_lock {
                waited = began.elapsed();
                first_lock = false;
            }
            if !GUARDS_STATE.protected.contains(block) {
                break 0;
            }
            let given = if GUARDS_STATE.aliased.contains(block) {
                let mut now: Vec<usize> = crate::inplace::pages_of(block * BLOCK).into_iter().map(|p| p / BLOCK).collect();
                now.push(block);
                now.sort_unstable();
                now.dedup();
                let needed = mask | guard_shards_of_blocks(now.iter().copied());
                if needed != mask {
                    drop(g);
                    mask = needed;
                    continue;
                }
                Some(now)
            } else {
                None
            };
            let opened = open_given(g.as_ref(), at, at + 1, given.as_deref());
            if opened != 0 {
                g.opened(block);
            }
            break opened;
        };
        (opened, waited)
    } else {
        let mut views: Vec<usize> = vec![block];
        let mut mask = 1u64 << guard_shard_of(block);
        let mut waited = std::time::Duration::ZERO;
        let mut first_lock = true;
        let opened = loop {
            let g = guard_shards_read(mask);
            if first_lock {
                waited = began.elapsed();
                port6_count(P6_FAULT_WAIT_NS, waited.as_nanos() as u64);
                first_lock = false;
            }
            let _claims = Claims::wait_for(&views);
            if !GUARDS_STATE.protected.contains(block) {
                break 0;
            }
            let given = if GUARDS_STATE.aliased.contains(block) {
                let mut now: Vec<usize> = crate::inplace::pages_of(block * BLOCK).into_iter().map(|p| p / BLOCK).collect();
                now.push(block);
                now.sort_unstable();
                now.dedup();
                let needed = mask | guard_shards_of_blocks(now.iter().copied());
                if needed != mask || now.iter().any(|b| views.binary_search(b).is_err()) {
                    drop(_claims);
                    drop(g);
                    views = now;
                    mask = needed;
                    continue;
                }
                Some(now)
            } else {
                None
            };
            let opened = open_given(g.as_ref(), at, at + 1, given.as_deref());
            if opened != 0 {
                g.opened(block);
                port6_count(P6_FAULTS, 1);
            }
            break opened;
        };
        (opened, waited)
    };
    if opened != 0 && note_sequential_fault(block) {
        unprotect_streaming_span(block);
    }
    (opened, waited)
}

/// An asking made blocks of `pages` clean and noted them in the journal.
fn publish(pages: &[usize]) {
    let mask = guard_shards_of_blocks(pages.iter().copied());
    if mask == 0 {
        return;
    }
    if port_guard() {
        // A block's claim while its bit is set: a fault opening it meanwhile
        // clears the bit after this sets it, never before.
        let _g = guard_shards_read(mask);
        for &page in pages {
            let _claim = Claims::wait_for(&[page]);
            if GUARDS_STATE.clean.contains(page) || GUARDS_STATE.held.contains(page) {
                PUBLISHED.set(page);
                if BESIDE.any(page, page + 1) {
                    PUBLISHED.clear(page);
                }
            }
        }
        return;
    }
    let _g = guard_shards_write(mask);
    for &page in pages {
        if GUARDS_STATE.clean.contains(page) || GUARDS_STATE.held.contains(page) {
            PUBLISHED.set(page);
            // (Put there since: whoever did clears the bit after this is
            // set, or its own bit is found here.)
            if BESIDE.any(page, page + 1) {
                PUBLISHED.clear(page);
            }
        }
    }
}

/// Whether placeholders are known for what they are (`VICTORY_KNOWN_HOLES=0`:
/// every hole is looked at at every asking, as before).
///
/// Most ranges asked about have holes, reservations the title maps its pages
/// into as it streams (231,000 regions, 1,125 of a frame's 1,530 asked view
/// ranges, run next1-1). Such a range was never known whole: each asking put
/// a question to Windows for each of its holes, under the guards' lock, and
/// 111 ms a frame of all threads' time went into waiting for that lock.
fn known_holes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KNOWN_HOLES").is_none_or(|v| v != "0") && light())
}

/// If `block` lies in a placeholder reservation, returns the exclusive end
/// block of that placeholder region (`>= block + 1`); otherwise `None`.
fn placeholder_end(block: usize) -> Option<usize> {
    /// `MEMORY_REGION_INFORMATION` (ntddk).
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryVirtualMemory(
            process: *mut std::ffi::c_void,
            at: *const std::ffi::c_void,
            class: u32,
            out: *mut std::ffi::c_void,
            len: usize,
            returned: *mut usize,
        ) -> i32;
    }
    // (Class 3 answers with the kind of memory in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    if status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0 {
        let end = region.allocation_base.saturating_add(region.region_size) / BLOCK;
        Some(end.max(block + 1))
    } else {
        None
    }
}

/// Whether `block` lies in a placeholder: a reservation in which pages come
/// to be only by mapping a view (or an allocation) in its place, which an
/// ordinary commit cannot do.
fn placeholder(block: usize) -> bool {
    placeholder_end(block).is_some()
}

/// The runtime gave the title memory at `[at, at + len)`, allocated or
/// committed: what was known of holes there is asked again.
pub fn title_memory_allocated(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    invalidate_reservations(first, last);
    if !known_holes() {
        return;
    }
    let g = guard_shards_write(guard_shards_of(first, last));
    let blocks: Vec<usize> = g.held.range(first..last).collect();
    for block in blocks {
        g.remove_held(block);
        PUBLISHED.clear(block);
    }
}

/// Once a frame: a hundredth of the known holes are looked at, in case one
/// was filled with nobody telling ([`title_memory_allocated`],
/// [`title_memory_remapped`]), which no title we know does.
#[cfg(feature = "diagnostics")]
pub(crate) fn look_at_holes() {
    const PARTS: usize = 100;
    if !known_holes() {
        return;
    }
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let blocks: Vec<usize> = {
        let g = guards_shared();
        let total = g.held_count.load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return;
        }
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.held.range(from..).take(total / PARTS + 1).collect();
        FROM.store(part.last().map_or(0, |b| b + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    let mut until = 0;
    for block in blocks {
        if block >= until {
            until = uncommitted_end(block);
        }
        if block >= until {
            // (Or unmapped memory was mapped since the part was taken: told
            // already, and forgotten.)
            let g = guard_shards_write(1u64 << guard_shard_of(block));
            if g.remove_held(block) {
                PUBLISHED.clear(block);
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{:#x}", block * BLOCK),
                        "a placeholder of the title's was filled with nobody telling (VICTORY_KNOWN_HOLES=0 asks every time)"
                    );
                });
            }
        }
    }
}

/// Once a frame: speculative hole probe disabled outside `diagnostics` builds
/// (no title fills placeholders without notifying the runtime).
#[cfg(not(feature = "diagnostics"))]
#[inline]
pub(crate) fn look_at_holes() {}

/// Whether a block of `[start, end)` is view memory read-only by our hand:
/// the kernel keeps no written pages of it and would refuse an asking.
pub(crate) fn guarded(start: usize, end: usize) -> bool {
    PROTECTED.any_acquire(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether nothing of `[start, end)` can have been written since an asking
/// noted it in the journal ([`PUBLISHED`]).
fn published(start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // (In this order: who puts something in beside a guard sets the one bit,
    // then clears the other, which an asking that is publishing the block may
    // just be setting.)
    light() && PUBLISHED.all(first, last) && !BESIDE.any(first, last)
}

/// Since the last frame line: what asking about title memory was. Kernel
/// askings and the KiB they walked; view ranges answered by their bits, by
/// the guards whole clean, and block by block; blocks made read-only; blocks
/// opened by ourselves; nanoseconds waiting for the guards' lock and
/// nanoseconds of changing protections (all threads); questions to Windows
/// about holes; kernel askings refused; nanoseconds in kernel askings; the
/// 4 KiB pages they found written.
#[cfg(feature = "diagnostics")]
static ASKS: [std::sync::atomic::AtomicU64; 13] = [const { std::sync::atomic::AtomicU64::new(0) }; 13];

#[cfg(feature = "diagnostics")]
fn tally(what: usize, n: u64) {
    ASKS[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// [`ASKS`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn asks() -> [u64; 13] {
    std::array::from_fn(|i| ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The guards' lock on `mask`, the wait for it counted.
fn guard_shards_write_timed(mask: u64) -> GuardsWrite {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let g = guard_shards_write(mask);
    #[cfg(feature = "diagnostics")]
    tally(7, started.elapsed().as_nanos() as u64);
    g
}

/// Shared guards' lock on `mask`, the wait for it counted.
fn guard_shards_read_timed(mask: u64) -> GuardsRead {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let g = guard_shards_read(mask);
    #[cfg(feature = "diagnostics")]
    tally(7, started.elapsed().as_nanos() as u64);
    g
}

/// A set of blocks, a bit each: a fault, or a look at a range of thousands of
/// blocks, is a few words of it and no walk through a tree of every guarded
/// block there is. Of atomic words (renderer port #6): a thread holding
/// [`GUARDS`] shared changes the bits of the blocks it holds the claims of
/// ([`BUSY`]), while others read and change other blocks' bits of the same
/// words. Who holds the lock alone sees and changes them as before.
struct Blocks(Bits);

impl Blocks {
    /// The blocks of 4 GiB (a piece of [`Bits`]).
    #[cfg(test)]
    const IN_PIECE: usize = Bits::WORDS * 64;

    const fn new() -> Blocks {
        Blocks(Bits::new())
    }

    fn word(&self, word: usize) -> u64 {
        self.0.word(word)
    }

    fn contains(&self, block: usize) -> bool {
        self.word(block / 64) >> (block % 64) & 1 != 0
    }

    fn any(&self, first: usize, last: usize) -> bool {
        self.0.any(first, last)
    }

    fn all(&self, first: usize, last: usize) -> bool {
        self.0.all(first, last)
    }

    /// Whether it was not there.
    fn insert(&self, block: usize) -> bool {
        self.0.set_was(block) == Some(false)
    }

    /// Whether it was there.
    fn remove(&self, block: usize) -> bool {
        self.0.clear_was(block)
    }

    fn insert_all(&self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// Those of `range`, in order.
    fn range(&self, range: impl RangeBounds<usize>) -> impl Iterator<Item = usize> + '_ {
        let first = match range.start_bound() {
            Bound::Included(&first) => first,
            Bound::Excluded(&before) => before.saturating_add(1),
            Bound::Unbounded => 0,
        };
        // (Exclusive. No block's number is near the top.)
        let last = match range.end_bound() {
            Bound::Included(&last) => last.saturating_add(1),
            Bound::Excluded(&last) => last,
            Bound::Unbounded => usize::MAX,
        }
        .max(first);
        self.0.set_in(first, last)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.count()
    }
}

/// Of the 64 blocks from `at` on, those in `[from, to)`, a bit each.
fn in_word(at: usize, from: usize, to: usize) -> u64 {
    let low = from.saturating_sub(at).min(64);
    let high = to.saturating_sub(at).min(64);
    if low >= high { 0 } else { (u64::MAX >> (64 - (high - low))) << low }
}

/// The bits of a word that are set, by number.
struct BitsOf(u64);

impl Iterator for BitsOf {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let bit = (self.0 != 0).then(|| self.0.trailing_zeros() as usize)?;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

/// Whether a write fault opens the view that was written and leaves the
/// page's other views read-only, not clean any more
/// (`VICTORY_OPEN_ONE_VIEW=0`: opens them all, as before).
///
/// The others were opened (a change of protection each, under the guards'
/// lock, with other faults waiting) and made read-only again when the page
/// was next taken, though the title writes through one of them: 586 faults a
/// frame opened 880 blocks (run faults-1). A write through one left as it was
/// faults, and opens it.
fn open_one_view() -> bool {
    crate::toggled!("VICTORY_OPEN_ONE_VIEW")
}

/// How long a block the title has just opened by a write is left open by
/// those that take it (`VICTORY_GUARD_REST=<microseconds>`, 3,000 unless
/// given; `0`: not at all, as before).
///
/// 220 of a frame's 586 faults came within 3 ms after a refresh had taken the
/// block and made it read-only, while the title was still writing it, and
/// what a refresh takes is asked about again by the next within a millisecond
/// (run faults-1): made read-only, written, opened, up to a dozen times a
/// frame. A block that rests counts as written at each asking, so nothing is
/// missed; it is read-only at the first asking after its rest, which the
/// title's later writes are seen by.
fn rest() -> std::time::Duration {
    #[cfg(test)]
    {
        REST_IN_TEST.get()
    }
    #[cfg(not(test))]
    {
        static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        if !crate::toggled!("VICTORY_GUARD_REST") {
            return std::time::Duration::ZERO;
        }
        let us = *US.get_or_init(|| {
            std::env::var("VICTORY_GUARD_REST").ok().and_then(|v| v.trim().parse().ok()).filter(|us| *us > 1).unwrap_or(3000)
        });
        std::time::Duration::from_micros(us)
    }
}

#[cfg(test)]
thread_local! {
    /// [`rest`], for what this thread asks.
    static REST_IN_TEST: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

/// 256 blocks of 64 KiB = 16 MiB VA bucket per guard shard (`ARCH-11`).
const GUARD_SHARD_SHIFT: u32 = 8;
/// 64 VA shards for `GUARDS` (`[RwLock<GuardShard>; 64]`), always acquired in
/// strictly ascending shard-index order `0..64` so deadlock is impossible.
const GUARD_SHARDS: usize = 64;

#[inline]
fn guard_shard_of(block: usize) -> usize {
    (block >> GUARD_SHARD_SHIFT) & (GUARD_SHARDS - 1)
}

#[inline]
fn guard_shards_of(first: usize, last: usize) -> u64 {
    if first >= last {
        return 0;
    }
    let (a, b) = (first >> GUARD_SHARD_SHIFT, (last - 1) >> GUARD_SHARD_SHIFT);
    if b - a + 1 >= GUARD_SHARDS {
        return u64::MAX;
    }
    (a..=b).fold(0u64, |m, bucket| m | (1u64 << (bucket & (GUARD_SHARDS - 1))))
}

#[inline]
fn guard_shards_of_blocks(blocks: impl IntoIterator<Item = usize>) -> u64 {
    blocks.into_iter().fold(0u64, |m, block| m | (1u64 << guard_shard_of(block)))
}

/// Shard bitmask covering all alias views of aliased blocks in `[first, last)`.
fn alias_shards_in(first: usize, last: usize) -> u64 {
    let mut mask = 0u64;
    for block in GUARDS_STATE.aliased.range(first..last) {
        for other in crate::inplace::pages_of(block * BLOCK) {
            mask |= 1u64 << guard_shard_of(other / BLOCK);
        }
    }
    mask
}

/// Global atomic block-state bitsets shared across all 64 guard shards.
struct GuardsState {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them.
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Blocks,
    /// Of those, the ones in placeholders ([`known_holes`]).
    held: Blocks,
    #[cfg(feature = "diagnostics")]
    held_count: std::sync::atomic::AtomicUsize,
    /// Blocks whose page is shown at other addresses too and that were made
    /// read-only together with those (which are in it as well): what opens
    /// one opens them all.
    aliased: Blocks,
    /// Fast lock-free hint of blocks currently resting in a shard's `opened`.
    resting: Blocks,
    /// Fast lock-free hint of blocks in an active sequential-write streaming span.
    streaming: Blocks,
}

static GUARDS_STATE: GuardsState = GuardsState {
    protected: Blocks::new(),
    clean: Blocks::new(),
    absent: Blocks::new(),
    held: Blocks::new(),
    #[cfg(feature = "diagnostics")]
    held_count: std::sync::atomic::AtomicUsize::new(0),
    aliased: Blocks::new(),
    resting: Blocks::new(),
    streaming: Blocks::new(),
};

/// Per-16-MiB VA bucket mutable guard state.
struct GuardShard {
    /// Blocks that were absent and turned out to be other memory than views
    /// when something was there again: never guarded, copied every time.
    foreign: BTreeSet<usize>,
    /// Blocks (first, past the last) the kernel writes now ([`kernel_writing`]).
    /// Left as they are by [`guard`], which reports them written.
    kernel: Vec<(usize, usize)>,
    /// Blocks opened by a write fault, and when ([`rest`]); and how hot each
    /// is ([`hot_rest`]). Behind a shard-local lock so faults holding the
    /// shard shared never contend across regions.
    opened: Mutex<std::collections::HashMap<usize, Opened, std::hash::BuildHasherDefault<PageHash>>>,
    /// Active streaming write-through spans `(first, last, until)` in this shard.
    streaming: Mutex<Vec<(usize, usize, std::time::Instant)>>,
}

#[repr(align(64))]
struct GuardShardSlot(RwLock<GuardShard>);

static GUARD_SHARD_LOCKS: [GuardShardSlot; GUARD_SHARDS] =
    [const { GuardShardSlot(RwLock::new(GuardShard::new())) }; GUARD_SHARDS];

/// When a write fault opened a block, and how often it had been written again
/// soon after it was taken ([`hot_rest`]).
#[derive(Clone, Copy)]
struct Opened {
    at: std::time::Instant,
    hot: u8,
}

/// Default ceiling (ms) for hot-block rest extension (`VICTORY_GUARD_HOT`).
/// Roughly one 60 FPS frame: keeps blocks written repeatedly within a frame
/// writable without re-protecting mid-burst, while re-protecting promptly
/// once writes stop. `VICTORY_GUARD_HOT=0` disables hot-rest extension.
const DEFAULT_HOT_REST_MS: u64 = 16;
/// Repeat-fault window past the previous rest before a block cools down
/// (covers per-frame faults at ~20–60 FPS so steady per-frame blocks promote).
const HOT_WINDOW: std::time::Duration = std::time::Duration::from_millis(55);
/// Repeat-fault streak required before extending rest (probation avoids
/// extending rest for one-off 2-write streaming bursts).
const HOT_PROMOTE_AT: u8 = 2;

/// Sequential-write streaming write-through thresholds (`ARCH-11`):
/// when `>= 3` adjacent 64 KiB guarded blocks fault within `STREAM_WINDOW_MS`,
/// unprotect a contiguous ahead span (`1.5 MiB`) as write-through for `120 ms`.
const STREAM_TRIGGER_COUNT: u8 = 3;
const STREAM_WINDOW_MS: u64 = 16;
const STREAM_SPAN_BLOCKS: usize = 24; // 1.5 MiB (within 1-2 MiB target)
const DEFAULT_STREAM_TTL_MS: u64 = 120;

static STREAM_SPANS_OPENED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static STREAM_BLOCKS_OPENED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static STREAM_TTL_IN_TEST: std::cell::Cell<Option<std::time::Duration>> = const { std::cell::Cell::new(None) };
}

fn stream_ttl() -> std::time::Duration {
    #[cfg(test)]
    if let Some(d) = STREAM_TTL_IN_TEST.get() {
        return d;
    }
    static TTL: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *TTL.get_or_init(|| {
        if std::env::var_os("VICTORY_STREAM_WATCH").is_some_and(|v| v == "0") {
            return std::time::Duration::ZERO;
        }
        let ms = std::env::var("VICTORY_STREAM_WATCH_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_STREAM_TTL_MS)
            .min(1000);
        std::time::Duration::from_millis(ms)
    })
}

#[derive(Clone, Copy)]
struct SeqFault {
    block: usize,
    streak: u8,
    at: Option<std::time::Instant>,
}

thread_local! {
    static LAST_SEQ_FAULT: std::cell::Cell<SeqFault> = const {
        std::cell::Cell::new(SeqFault { block: 0, streak: 0, at: None })
    };
}

/// Records a write fault on `block` on the faulting thread; returns `true` when
/// `>= STREAM_TRIGGER_COUNT` strictly adjacent 64 KiB guarded blocks have
/// faulted sequentially within `STREAM_WINDOW_MS`.
fn note_sequential_fault(block: usize) -> bool {
    if stream_ttl().is_zero() {
        return false;
    }
    let now = std::time::Instant::now();
    let window = std::time::Duration::from_millis(STREAM_WINDOW_MS);
    let prev = LAST_SEQ_FAULT.get();
    let streak = match prev.at {
        Some(t) if now.duration_since(t) <= window && block == prev.block.wrapping_add(1) => {
            prev.streak.saturating_add(1)
        }
        _ => 1,
    };
    if streak >= STREAM_TRIGGER_COUNT {
        LAST_SEQ_FAULT.set(SeqFault { block, streak: 0, at: Some(now) });
        true
    } else {
        LAST_SEQ_FAULT.set(SeqFault { block, streak, at: Some(now) });
        false
    }
}

/// Unprotects a contiguous guarded span ahead of `trigger_block` (`1.5 MiB`)
/// as write-through for `stream_ttl()` (`120 ms`), leaving its blocks open and
/// reported dirty in the journal (`wrote()`) so refreshes never read stale data.
fn unprotect_streaming_span(trigger_block: usize) {
    let ttl = stream_ttl();
    if ttl.is_zero() {
        return;
    }
    let open_first = trigger_block.saturating_add(1);
    let open_limit = open_first.saturating_add(STREAM_SPAN_BLOCKS);
    let mut open_last = open_first;
    while open_last < open_limit && PROTECTED.any_acquire(open_last, open_last + 1) {
        open_last += 1;
    }
    if open_last <= open_first {
        return;
    }
    let ttl_first = trigger_block.saturating_sub(usize::from(STREAM_TRIGGER_COUNT) - 1);
    let mut mask = guard_shards_of(ttl_first, open_last);
    let opened = loop {
        let g = guard_shards_write(mask);
        if GUARDS_STATE.aliased.any(open_first, open_last) {
            let needed = mask | alias_shards_in(open_first, open_last);
            if needed != mask {
                drop(g);
                mask = needed;
                continue;
            }
        }
        let opened = open(g.as_ref(), open_first * BLOCK, open_last * BLOCK);
        if opened != 0 {
            let until = std::time::Instant::now() + ttl;
            g.note_streaming(ttl_first, open_last, until);
            STREAM_SPANS_OPENED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            STREAM_BLOCKS_OPENED.fetch_add(opened as u64, std::sync::atomic::Ordering::Relaxed);
        }
        break opened;
    };
    // Publish dirtiness to any standing `ask_once` journal AFTER dropping guard locks.
    if opened != 0 {
        wrote(open_first * BLOCK, (open_last - open_first) * BLOCK);
    }
}

/// How long a block the title writes again soon after each time it is taken
/// rests at most (`VICTORY_GUARD_HOT=<ms>`; default 16 ms; `0`: off, [`rest`]
/// for all). A block opened by a fault repeatedly within its rest plus
/// [`HOT_WINDOW`] rests twice as long per streak past [`HOT_PROMOTE_AT`], up
/// to this ceiling; one that stayed clean longer rests [`rest`] again.
fn hot_rest() -> std::time::Duration {
    #[cfg(test)]
    if let Some(ms) = HOT_IN_TEST.get() {
        return std::time::Duration::from_millis(ms);
    }
    #[cfg(feature = "diagnostics")]
    if crate::ab::pick("VICTORY_GUARD_HOT") == Some(false) {
        return std::time::Duration::ZERO;
    }
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| {
        std::env::var("VICTORY_GUARD_HOT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_HOT_REST_MS)
            .min(1000)
    });
    if ms == 0 {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::from_millis(ms)
}

impl Opened {
    /// How long a block opened so rests.
    fn rest(&self, rest: std::time::Duration) -> std::time::Duration {
        if self.hot < HOT_PROMOTE_AT || rest.is_zero() {
            return rest;
        }
        let most = hot_rest();
        if most.is_zero() {
            return rest;
        }
        rest.saturating_mul(1 << (self.hot - HOT_PROMOTE_AT + 1).min(16)).min(most.max(rest))
    }
}

/// Blocks the runtime has unmapped or split in an active [`Remapping`]
/// ([`title_memory_unmapping`], [`title_memory_splitting`]). Kept out of
/// [`GUARD_SHARD_LOCKS`] behind a `SeqCst` active flag so unmapping cold ranges
/// never takes guard shards exclusively while preserving Dekker ordering with
/// `VIEWS` and `GUARDS_STATE.aliased`.
static CHANGING_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static CHANGING_RANGES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

fn note_changing(first: usize, last: usize) {
    if first >= last {
        return;
    }
    let mut changing = CHANGING_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    match changing.last_mut() {
        Some(before) if before.1 == first => before.1 = last,
        _ => changing.push((first, last)),
    }
    CHANGING_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn clear_changing() {
    let mut changing = CHANGING_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    changing.clear();
    CHANGING_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
}

fn is_changing(block: usize) -> bool {
    if !CHANGING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) {
        return false;
    }
    CHANGING_RANGES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|&(first, last)| (first..last).contains(&block))
}

impl GuardsState {
    fn insert_held(&self, block: usize) -> bool {
        let added = self.held.insert(block);
        #[cfg(feature = "diagnostics")]
        if added {
            self.held_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        added
    }

    fn remove_held(&self, block: usize) -> bool {
        let removed = self.held.remove(block);
        #[cfg(feature = "diagnostics")]
        if removed {
            self.held_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        removed
    }
}

impl GuardShard {
    const fn new() -> GuardShard {
        GuardShard {
            foreign: BTreeSet::new(),
            kernel: Vec::new(),
            opened: Mutex::new(std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new())),
            streaming: Mutex::new(Vec::new()),
        }
    }

    fn left_alone(&self, block: usize) -> bool {
        self.kernel.iter().any(|&(first, last)| (first..last).contains(&block)) || is_changing(block)
    }

    /// Whether `block` is left open for now: the title has just written it.
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        if rest.is_zero() || !GUARDS_STATE.resting.contains(block) {
            return false;
        }
        let opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
        let active = opened.get(&block).is_some_and(|o| {
            let elapsed = o.at.elapsed();
            let longer = o.rest(rest);
            if elapsed >= rest && elapsed < longer {
                port6_count(P6_RESTED_HOT, 1);
            }
            elapsed < longer
        });
        if !active {
            GUARDS_STATE.resting.remove(block);
        }
        active
    }

    /// Whether `block` lies in an active sequential-write streaming write-through span.
    fn is_streaming(&self, block: usize) -> bool {
        if !GUARDS_STATE.streaming.contains(block) {
            return false;
        }
        let now = std::time::Instant::now();
        let mut spans = self.streaming.lock().unwrap_or_else(|e| e.into_inner());
        let mut active = false;
        let mut i = 0;
        while i < spans.len() {
            let (a, b, until) = spans[i];
            if now >= until {
                spans.swap_remove(i);
                for blk in a..b {
                    if !spans.iter().any(|&(oa, ob, ou)| now < ou && (oa..ob).contains(&blk)) {
                        GUARDS_STATE.streaming.remove(blk);
                    }
                }
            } else {
                if (a..b).contains(&block) {
                    active = true;
                }
                i += 1;
            }
        }
        if !active {
            GUARDS_STATE.streaming.remove(block);
        }
        active
    }

    fn note_streaming(&self, first: usize, last: usize, until: std::time::Instant) {
        if first >= last {
            return;
        }
        let now = std::time::Instant::now();
        let mut spans = self.streaming.lock().unwrap_or_else(|e| e.into_inner());
        spans.retain(|&(_, _, u)| u > now);
        // Cap per-shard active streaming spans so metadata stays bounded.
        if spans.len() >= 64 {
            let (old_a, old_b, _) = spans.remove(0);
            for blk in old_a..old_b {
                if !spans.iter().any(|&(oa, ob, _)| (oa..ob).contains(&blk)) {
                    GUARDS_STATE.streaming.remove(blk);
                }
            }
        }
        spans.push((first, last, until));
        for blk in first..last {
            GUARDS_STATE.streaming.insert(blk);
        }
    }

    fn clear_streaming(&self, first: usize, last: usize) {
        if first >= last || !GUARDS_STATE.streaming.any(first, last) {
            return;
        }
        let mut spans = self.streaming.lock().unwrap_or_else(|e| e.into_inner());
        let mut updated = Vec::with_capacity(spans.len());
        for &(a, b, until) in spans.iter() {
            if b <= first || a >= last {
                updated.push((a, b, until));
            } else {
                if a < first {
                    updated.push((a, first, until));
                }
                if last < b {
                    updated.push((last, b, until));
                }
            }
        }
        *spans = updated;
        for blk in first..last {
            GUARDS_STATE.streaming.remove(blk);
        }
    }

    fn clear_opened(&self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        let mut opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
        if opened.is_empty() {
            return;
        }
        if last - first <= opened.len() {
            for b in first..last {
                if opened.remove(&b).is_some() {
                    GUARDS_STATE.resting.remove(b);
                }
            }
        } else {
            opened.retain(|&b, _| {
                if (first..last).contains(&b) {
                    GUARDS_STATE.resting.remove(b);
                    false
                } else {
                    true
                }
            });
        }
    }

    /// A write fault opened `block`.
    fn opened(&self, block: usize) {
        const MANY: usize = 1 << 10;
        let rest = rest();
        let most = hot_rest();
        let mut opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
        if opened.len() >= MANY {
            let keep = HOT_WINDOW.max(most + HOT_WINDOW);
            opened.retain(|&b, o| {
                let alive = o.at.elapsed() < keep;
                if !alive {
                    GUARDS_STATE.resting.remove(b);
                }
                alive
            });
            if opened.len() >= MANY {
                for &b in opened.keys() {
                    GUARDS_STATE.resting.remove(b);
                }
                opened.clear();
            }
        }
        put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        let now = std::time::Instant::now();
        let hot = match opened.get(&block) {
            // Written again within its rest and HOT_WINDOW: hotter; extends rest once streak >= HOT_PROMOTE_AT.
            Some(o) if !most.is_zero() && now.duration_since(o.at) < o.rest(rest) + HOT_WINDOW => {
                let next = o.hot.saturating_add(1).min(16);
                if next >= HOT_PROMOTE_AT {
                    port6_count(P6_HOT, 1);
                }
                next
            }
            _ => 0,
        };
        opened.insert(block, Opened { at: now, hot });
        GUARDS_STATE.resting.insert(block);
    }
}

/// Exclusive guards held on the subset of `GUARD_SHARD_LOCKS` named by `mask`,
/// always acquired in strictly ascending shard order `0..64` (zero heap alloc).
struct GuardsWrite {
    mask: u64,
    len: u8,
    guards: [std::mem::MaybeUninit<std::sync::RwLockWriteGuard<'static, GuardShard>>; GUARD_SHARDS],
}

/// Shared guards held on the subset of `GUARD_SHARD_LOCKS` named by `mask`,
/// always acquired in strictly ascending shard order `0..64` (zero heap alloc).
struct GuardsRead {
    mask: u64,
    len: u8,
    guards: [std::mem::MaybeUninit<std::sync::RwLockReadGuard<'static, GuardShard>>; GUARD_SHARDS],
}

/// Non-blocking extra shard guard acquired during `cover()` when an alias block
/// falls outside the primary range's shard mask.
enum ExtraGuard {
    Write(std::sync::RwLockWriteGuard<'static, GuardShard>),
    Read(std::sync::RwLockReadGuard<'static, GuardShard>),
}

impl std::ops::Deref for ExtraGuard {
    type Target = GuardShard;
    fn deref(&self) -> &GuardShard {
        match self {
            ExtraGuard::Write(g) => g,
            ExtraGuard::Read(g) => g,
        }
    }
}

#[derive(Clone, Copy)]
enum GuardsRef<'a> {
    Write(&'a GuardsWrite),
    Read(&'a GuardsRead),
}

#[inline]
fn shard_place(mask: u64, shard: usize) -> usize {
    debug_assert!((mask >> shard) & 1 != 0, "shard {shard} not held in mask {mask:#x}");
    (mask & ((1u64 << shard) - 1)).count_ones() as usize
}

impl GuardsWrite {
    fn acquire(mask: u64) -> GuardsWrite {
        let mut out = GuardsWrite {
            mask,
            len: 0,
            guards: [const { std::mem::MaybeUninit::uninit() }; GUARD_SHARDS],
        };
        let mut m = mask;
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            let lock = &GUARD_SHARD_LOCKS[s].0;
            let g = match lock.try_write() {
                Ok(g) => g,
                Err(_) => waited(3, || lock.write().unwrap_or_else(|e| e.into_inner())),
            };
            out.guards[usize::from(out.len)].write(g);
            out.len += 1;
        }
        out
    }

    #[inline]
    fn has_shard(&self, shard: usize) -> bool {
        (self.mask >> shard) & 1 != 0
    }

    #[inline]
    fn shard(&self, shard: usize) -> &GuardShard {
        let at = shard_place(self.mask, shard);
        // SAFETY: `at < self.len` because `shard` is set in `self.mask`.
        unsafe { self.guards[at].assume_init_ref() }
    }

    #[inline]
    fn shard_mut(&mut self, shard: usize) -> &mut GuardShard {
        let at = shard_place(self.mask, shard);
        // SAFETY: `at < self.len` because `shard` is set in `self.mask`.
        unsafe { self.guards[at].assume_init_mut() }
    }

    #[inline]
    fn as_ref(&self) -> GuardsRef<'_> {
        GuardsRef::Write(self)
    }

    fn left_alone(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).left_alone(block)
    }

    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        self.shard(guard_shard_of(block)).rests(block, rest)
    }

    fn is_streaming(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).is_streaming(block)
    }

    fn is_foreign(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).foreign.contains(&block)
    }

    fn opened(&self, block: usize) {
        self.shard(guard_shard_of(block)).opened(block);
    }

    fn note_streaming(&self, first: usize, last: usize, until: std::time::Instant) {
        let mut m = guard_shards_of(first, last);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            self.shard(s).note_streaming(first, last, until);
        }
    }

    fn clear_opened(&self, first: usize, last: usize) {
        let mut m = guard_shards_of(first, last);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            self.shard(s).clear_opened(first, last);
        }
    }

    fn clear_streaming(&self, first: usize, last: usize) {
        if !GUARDS_STATE.streaming.any(first, last) {
            return;
        }
        let mut m = guard_shards_of(first, last);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            self.shard(s).clear_streaming(first, last);
        }
    }
}

impl Drop for GuardsWrite {
    fn drop(&mut self) {
        for i in 0..usize::from(self.len) {
            // SAFETY: indices `0..self.len` were initialized in `acquire`.
            unsafe { self.guards[i].assume_init_drop() };
        }
    }
}

impl std::ops::Deref for GuardsWrite {
    type Target = GuardsState;
    fn deref(&self) -> &GuardsState {
        &GUARDS_STATE
    }
}

impl GuardsRead {
    fn acquire(mask: u64) -> GuardsRead {
        let mut out = GuardsRead {
            mask,
            len: 0,
            guards: [const { std::mem::MaybeUninit::uninit() }; GUARD_SHARDS],
        };
        let mut m = mask;
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            let lock = &GUARD_SHARD_LOCKS[s].0;
            let g = match lock.try_read() {
                Ok(g) => g,
                Err(_) => waited(3, || lock.read().unwrap_or_else(|e| e.into_inner())),
            };
            out.guards[usize::from(out.len)].write(g);
            out.len += 1;
        }
        out
    }

    #[inline]
    fn has_shard(&self, shard: usize) -> bool {
        (self.mask >> shard) & 1 != 0
    }

    #[inline]
    fn shard(&self, shard: usize) -> &GuardShard {
        let at = shard_place(self.mask, shard);
        // SAFETY: `at < self.len` because `shard` is set in `self.mask`.
        unsafe { self.guards[at].assume_init_ref() }
    }

    #[inline]
    fn as_ref(&self) -> GuardsRef<'_> {
        GuardsRef::Read(self)
    }

    fn left_alone(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).left_alone(block)
    }

    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        self.shard(guard_shard_of(block)).rests(block, rest)
    }

    fn is_streaming(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).is_streaming(block)
    }

    fn is_foreign(&self, block: usize) -> bool {
        self.shard(guard_shard_of(block)).foreign.contains(&block)
    }

    fn opened(&self, block: usize) {
        self.shard(guard_shard_of(block)).opened(block);
    }
}

impl Drop for GuardsRead {
    fn drop(&mut self) {
        for i in 0..usize::from(self.len) {
            // SAFETY: indices `0..self.len` were initialized in `acquire`.
            unsafe { self.guards[i].assume_init_drop() };
        }
    }
}

impl std::ops::Deref for GuardsRead {
    type Target = GuardsState;
    fn deref(&self) -> &GuardsState {
        &GUARDS_STATE
    }
}

impl<'a> GuardsRef<'a> {
    #[inline]
    fn has_shard(&self, shard: usize) -> bool {
        match self {
            GuardsRef::Write(g) => g.has_shard(shard),
            GuardsRef::Read(g) => g.has_shard(shard),
        }
    }

    #[inline]
    fn shard(&self, shard: usize) -> &'a GuardShard {
        match self {
            GuardsRef::Write(g) => g.shard(shard),
            GuardsRef::Read(g) => g.shard(shard),
        }
    }

    /// Non-blocking attempt to lock an external shard in the same mode as
    /// `self` (`try_write` or `try_read`); never blocks, so deadlock is impossible.
    fn try_lock_extra(&self, shard: usize) -> Option<ExtraGuard> {
        let lock = &GUARD_SHARD_LOCKS[shard].0;
        match self {
            GuardsRef::Write(_) => lock.try_write().ok().map(ExtraGuard::Write),
            GuardsRef::Read(_) => lock.try_read().ok().map(ExtraGuard::Read),
        }
    }
}

fn guard_shards_write(mask: u64) -> GuardsWrite {
    GuardsWrite::acquire(mask)
}

fn guard_shards_read(mask: u64) -> GuardsRead {
    GuardsRead::acquire(mask)
}

/// All 64 guard shards, exclusively (tests).
#[cfg(test)]
fn guards() -> GuardsWrite {
    guard_shards_write(u64::MAX)
}

/// All 64 guard shards, shared (tests & diagnostics `look_at_holes`).
#[cfg(any(test, feature = "diagnostics"))]
fn guards_shared() -> GuardsRead {
    guard_shards_read(u64::MAX)
}

/// The 4 KiB pages of title memory that what shaders wrote was put into,
/// and the frame it last was: in pieces made when first needed and never let
/// go. Some 20,000 pages a frame are noted by the relay and the landing
/// threads: in a map behind one lock that was 309 waits for it and 280 looks
/// at its table in the 40 s of a profile, as many as all else of the landings'
/// notes (run faults-5).
struct Landed([std::sync::atomic::AtomicPtr<LandedPiece>; Landed::PIECES]);

/// Of 256 MiB: the frame (its low 31 bits and one), 0 for never.
type LandedPiece = [std::sync::atomic::AtomicU32; Landed::PAGES];

impl Landed {
    const PAGES: usize = 1 << 16;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 19;

    fn stamp(frame: u64) -> u32 {
        (frame as u32 & 0x7fff_ffff) + 1
    }

    fn note(&self, page: usize, frame: u64) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return;
        };
        let mut piece = place.load(Acquire);
        if piece.is_null() {
            let made: Box<LandedPiece> = Box::new([const { std::sync::atomic::AtomicU32::new(0) }; Landed::PAGES]);
            let made = Box::into_raw(made);
            piece = match place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire) {
                Ok(_) => made,
                Err(there) => {
                    // SAFETY: made just now, and another thread's is there.
                    drop(unsafe { Box::from_raw(made) });
                    there
                }
            };
        }
        // SAFETY: a piece, which is never freed.
        let at = unsafe { &(&*piece)[page % Landed::PAGES] };
        let stamp = Landed::stamp(frame);
        // (Most are noted several times a frame: no write then.)
        if at.load(Relaxed) != stamp {
            at.store(stamp, Relaxed);
        }
    }

    /// Whether `page` was noted in a frame from `since` on, up to `now`.
    fn since(&self, page: usize, since: u64, now: u64) -> bool {
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return false;
        };
        // SAFETY: null, or a piece, which is never freed.
        let Some(piece) = (unsafe { place.load(std::sync::atomic::Ordering::Acquire).as_ref() }) else {
            return false;
        };
        let stamp = piece[page % Landed::PAGES].load(std::sync::atomic::Ordering::Relaxed);
        stamp != 0 && u64::from(Landed::stamp(now).wrapping_sub(stamp) & 0x7fff_ffff) <= now.saturating_sub(since)
    }
}

static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);

/// A page's number, spread.
#[derive(Default)]
struct PageHash(u64);

impl std::hash::Hasher for PageHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(26).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// What shaders wrote is about to go into title memory at `[at, at + len)`.
pub(crate) fn landing(at: usize, len: usize) {
    {
        let frame = crate::frame::current();
        let (first, last) = (at / KERNEL_PAGE, (at + len.max(1) - 1) / KERNEL_PAGE);
        (first..=last).for_each(|page| LANDED.note(page, frame));
        for (block, others) in elsewhere(at, len) {
            let block = block / KERNEL_PAGE;
            let here = first.max(block)..=last.min(block + BLOCK / KERNEL_PAGE - 1);
            for other in others {
                here.clone().for_each(|page| LANDED.note(page - block + other / KERNEL_PAGE, frame));
            }
        }
    }
    writing(at, len);
}

/// Records 4 KiB pages landed directly into write-watched `MEM_PRIVATE` title
/// memory (no section views or guards exist there).
pub(crate) fn note_landed_pages(pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    let frame = crate::frame::current();
    for &page in pages {
        LANDED.note(page, frame);
    }
    mark_write_epoch_pages(pages.iter().map(|&p| p / PARTS));
}


/// What shaders wrote went into title memory through another view of the
/// same pages (`crate::inplace::beside`): `pages` (of 4 KiB, by number)
/// hold some of it, all of them in `spans` (address, bytes).
///
/// Nothing faulted and no guard was opened. A guarded block stays read-only,
/// clean, and known by its bit ([`PUBLISHED`]), all of which say that the
/// title wrote nothing there: our write is an entry of the journal, which
/// every answer about the block comes from, the quick ones and the guards'
/// alike (a block whose guard is off they report anyway). So neither the
/// guards' lock nor a protection is touched here, or by the next refresh.
/// Noted after the bytes are in, never before: a copy that read the block
/// before they were did so with the number of an asking made before that,
/// which is under the one given here, and takes it again; noted before, a
/// copy could read the older bytes with a number above the note's.
///
/// Returns the note's number, and for each 64 KiB block noted the number of
/// the block's last write before (0: none is known).
pub(crate) fn landed_beside(pages: &[usize], spans: &[(usize, usize)]) -> (u64, Vec<(usize, u64)>) {
    let mut before = Vec::with_capacity(spans.len());
    let noted;
    // The same pages at other addresses: copies of the memory there have not
    // what was put in either.
    let elsewhere: Vec<(usize, Vec<usize>)> = spans.iter().flat_map(|&(at, len)| elsewhere(at, len)).collect();
    {
        let frame = crate::frame::current();
        for &page in pages {
            LANDED.note(page, frame);
        }
        for (block, others) in &elsewhere {
            let first = block / KERNEL_PAGE;
            for &page in pages.iter().filter(|&&page| (first..first + BLOCK / KERNEL_PAGE).contains(&page)) {
                others.iter().for_each(|other| LANDED.note(page - first + other / KERNEL_PAGE, frame));
            }
        }
    }
    {
        let mut blocks: Vec<(usize, u16)> = Vec::with_capacity(spans.len());
        for &page in pages {
            let (block, part) = (page / PARTS, 1u16 << (page % PARTS));
            match blocks.last_mut() {
                Some((last, parts)) if *last == block => *parts |= part,
                _ => blocks.push((block, part)),
            }
        }
        let others = || elsewhere.iter().flat_map(|(_, others)| others.iter().map(|other| other / PAGE));
        if port_journal() {
            // Renderer port #7: shared, the parts of its pages held.
            let j = journal();
            let mut held = j.notes.hold(shards_of_pages(blocks.iter().map(|b| b.0).chain(others())), true);
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, held.note(block, parts, now)));
            }
            for other in others() {
                held.note(other, WHOLE, now);
            }
            port7_count(P7_SHARED_NOTES, 1);
        } else {
            let mut j = journal_mut();
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, j.notes.note(block, parts, now)));
            }
            for other in others() {
                j.notes.note(other, WHOLE, now);
            }
        }
    }
    for &(at, len) in spans {
        mark_write_epoch_range(at, at.saturating_add(len));
        // A placed texture there changes with no descriptor written.
        crate::descriptor::host::remapped(at, len);
    }
    for other in elsewhere.into_iter().flat_map(|(_, others)| others) {
        mark_write_epoch_range(other, other.saturating_add(BLOCK));
        crate::descriptor::host::remapped(other, BLOCK);
    }
    (noted, before)
}

/// Whether every write of the title's to 64 KiB block `block` so far is in
/// the journal ([`PUBLISHED`]).
pub(crate) fn block_published(block: usize) -> bool {
    published(block * BLOCK, (block + 1) * BLOCK)
}

/// Whether shaders' writes went into a page of `[at, at + len)` in the last
/// `frames` frames.
pub(crate) fn landed_lately(at: usize, len: usize, frames: u64) -> bool {
    let now = crate::frame::current();
    (at / 4096..=(at + len.max(1) - 1) / 4096).any(|page| LANDED.since(page, now.saturating_sub(frames), now))
}

/// Whether every block of `[at, at + len)` is view memory written since a
/// host copy last took it: guarded then, and the guard is off. (Written by
/// the title, or by us putting a flight's shader writes into it, if not
/// beside the guard, [`landed_beside`].) Not for
/// blocks no copy ever took, which were never guarded: the caller's range
/// lies in a copy a flight was given, which took all of it. (Nor for blocks
/// read-only by way of another view of their page: whether they were written
/// since is not known.)
pub(crate) fn written_since_taken(at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let (first, last) = (at / BLOCK, (at + len - 1) / BLOCK + 1);
    let g = guard_shards_read(guard_shards_of(first, last));
    (first..last).all(|b| !g.clean.contains(b) && !g.absent.contains(b) && !g.protected.contains(b))
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mask = guard_shards_of(first, last);
    let ask = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    let check_clean_and_unsure = |g: &GuardsState| -> Option<Vec<usize>> {
        if (first >= last || g.clean.all(first, last)) && !BESIDE.any(first, last) {
            return None;
        }
        Some(g.absent.range(first..last).filter(|block| !g.held.contains(*block)).collect())
    };
    let fast_protected = !ask && PROTECTED.any_acquire(first, first + 1);
    let initial: Option<Option<Vec<usize>>> = if !ask {
        // Merge view-known check and clean/absent scan into a single sharded lock acquisition.
        let of = |g: &GuardsState| -> Option<Option<Vec<usize>>> {
            let known = fast_protected || g.protected.contains(first) || g.absent.contains(first);
            known.then(|| check_clean_and_unsure(g))
        };
        if port_guard() { of(&guard_shards_read_timed(mask)) } else { of(&guard_shards_write_timed(mask)) }
    } else {
        None
    };
    let unsure: Vec<usize> = match initial {
        Some(Some(unsure)) => {
            install();
            unsure
        }
        Some(None) => {
            install();
            // Nothing written, mapped or new since the range was last taken.
            #[cfg(feature = "diagnostics")]
            tally(3, 1);
            return Some(Vec::new());
        }
        None => {
            if !is_view(start, end) {
                return None;
            }
            install();
            // The holes that an ordinary commit may have filled, with no word of the
            // runtime's. Windows is asked with the lock not held (the range is, no
            // other thread asks about it: `ASKING`); what the runtime tells meanwhile
            // takes the block out of the set, and it is looked at as any other.
            let unsure = if port_guard() {
                check_clean_and_unsure(&guard_shards_read_timed(mask))
            } else {
                check_clean_and_unsure(&guard_shards_write_timed(mask))
            };
            match unsure {
                Some(unsure) => unsure,
                None => {
                    // Nothing written, mapped or new since the range was last taken.
                    #[cfg(feature = "diagnostics")]
                    tally(3, 1);
                    return Some(Vec::new());
                }
            }
        }
    };
    let mut came: Vec<(usize, bool)> = Vec::new();
    let mut uncommitted_until = first;
    for block in unsure {
        // One question for the blocks of a region.
        if block >= uncommitted_until {
            uncommitted_until = uncommitted_end(block);
            #[cfg(feature = "diagnostics")]
            tally(9, 1);
        }
        if block >= uncommitted_until {
            // The hole may have been given up, and its address be the
            // title's heap's, or code's, now: not ours to guard.
            came.push((block, is_view(block * BLOCK, (block + 1) * BLOCK)));
        }
    }
    let rest = rest();
    if port_guard() {
        return Some(guard_shared(first, last, &came, rest));
    }
    let mut g = guard_shards_write_timed(mask);
    // Mark confirmed view blocks in `VIEWS` so `kernel_writing` and `writes`
    // recognize view memory even when `!port_guard()`.
    if !VIEWS.all(first, last) {
        (first..last).for_each(|block| VIEWS.set(block));
    }
    #[cfg(feature = "diagnostics")]
    tally(4, 1);
    for &(block, view) in &came {
        if g.absent.remove(block) && !view {
            g.shard_mut(guard_shard_of(block)).foreign.insert(block);
        }
    }
    // The range's blocks that are neither clean nor known to be holes. A
    // frame asks about some 10 GB of view memory, 150,000 blocks, nearly all
    // clean.
    let mut written = Vec::new();
    for word in first / 64..last.div_ceil(64) {
        let known = g.clean.word(word) & !BESIDE.word(word) | g.absent.word(word);
        written.extend(BitsOf(!known & in_word(word * 64, first, last)).map(|bit| word * 64 + bit));
    }
    let mut run: Option<(usize, usize)> = None;
    for &block in &written {
        if let Some((from, to)) = run
            && to != block
        {
            protect(g.as_ref(), from, to);
            run = None;
            // Faults of the title's wait for the lock: each block is done on
            // its own, and looked at anew below.
            drop(g);
            g = guard_shards_write_timed(mask);
        }
        // Told as written by this, and read by the caller after.
        if BESIDE.take(block) {
            put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        }
        if g.protected.contains(block) {
            // Read-only by way of another view of its page, or not clean
            // because one could not be made so. (Or clean, and bytes were put
            // in beside the guard: it stays.)
            if let Some((from, to)) = run.take() {
                protect(g.as_ref(), from, to);
            }
            if g.clean.contains(block) {
                continue;
            }
            if cover(g.as_ref(), block, block + 1, None) {
                g.clean.insert(block);
            }
        } else if !g.is_foreign(block)
            && !g.left_alone(block)
            && !g.rests(block, rest)
            && !g.is_streaming(block)
        {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            protect(g.as_ref(), from, to);
        }
    }
    if let Some((from, to)) = run {
        protect(g.as_ref(), from, to);
    }
    drop(g);
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Asked, first, last, written.len() as u64);
    Some(written)
}

/// The rest of [`guard`] with renderer port #6: the lock shared, and a
/// block's claim held while its state changes. What a fault opens meanwhile
/// is opened before or after the block's state is looked at and changed,
/// never in between; a block whose claim another holds (a fault, a cover of
/// another range's) is left as it is, written: told so now, and looked at
/// again at the next asking.
fn guard_shared(first: usize, last: usize, came: &[(usize, bool)], rest: std::time::Duration) -> Vec<usize> {
    if !came.is_empty() {
        let came_mask = guard_shards_of_blocks(came.iter().map(|&(b, _)| b));
        let mut g = guard_shards_write_timed(came_mask);
        for &(block, view) in came {
            if g.absent.remove(block) && !view {
                g.shard_mut(guard_shard_of(block)).foreign.insert(block);
            }
        }
    }
    port6_count(P6_GUARDS, 1);
    let mask = guard_shards_of(first, last);
    let g = guard_shards_read_timed(mask);
    // Set VIEWS while holding shard read lock (before left_alone/is_changing)
    // so concurrent title_memory_unmapping never misses this guard attempt.
    if !VIEWS.all(first, last) {
        (first..last).for_each(|block| VIEWS.set(block));
    }
    #[cfg(feature = "diagnostics")]
    tally(4, 1);
    let mut written = Vec::new();
    for word in first / 64..last.div_ceil(64) {
        let known = g.clean.word(word) & !BESIDE.word(word) | g.absent.word(word);
        written.extend(BitsOf(!known & in_word(word * 64, first, last)).map(|bit| word * 64 + bit));
    }
    let mut run: Option<(usize, usize)> = None;
    for &block in &written {
        if let Some((from, to)) = run
            && to != block
        {
            protect_claimed(g.as_ref(), from, to);
            run = None;
        }
        // Told as written by this, and read by the caller after.
        if BESIDE.take(block) {
            put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        }
        if g.protected.contains(block) {
            if let Some((from, to)) = run.take() {
                protect_claimed(g.as_ref(), from, to);
            }
            if g.clean.contains(block) {
                continue;
            }
            // (No other claim is held here.)
            let mut claims = Claims::wait_for(&[block]);
            if g.protected.contains(block)
                && !g.clean.contains(block)
                && cover(g.as_ref(), block, block + 1, Some(&mut claims))
            {
                g.clean.insert(block);
            }
        } else if !g.is_foreign(block)
            && !g.left_alone(block)
            && !g.rests(block, rest)
            && !g.is_streaming(block)
        {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            protect_claimed(g.as_ref(), from, to);
        }
    }
    if let Some((from, to)) = run {
        protect_claimed(g.as_ref(), from, to);
    }
    drop(g);
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Asked, first, last, written.len() as u64);
    written
}

/// [`protect`] of `[from, to)` with the lock shared: of the blocks whose
/// claims can be had now, each run, its other views' blocks only where their
/// claims can be had too.
fn protect_claimed(g: GuardsRef<'_>, from: usize, to: usize) {
    let mut claims = Claims::none();
    let mut start = None;
    for block in from..to {
        if claims.try_one(block) {
            start.get_or_insert(block);
        } else if let Some(s) = start.take() {
            protect_run(g, s, block, &mut claims);
        }
    }
    if let Some(s) = start {
        protect_run(g, s, to, &mut claims);
    }
}

fn protect_run(g: GuardsRef<'_>, from: usize, to: usize, claims: &mut Claims) {
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    protect_here(&GUARDS_STATE, from, to);
    cover(g, from, to, Some(claims));
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: GuardsRef<'_>, from: usize, to: usize) {
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    protect_here(&GUARDS_STATE, from, to);
    cover(g, from, to, None);
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the blocks `[from, to)`, which are read-only by our hand where they
/// can be: every other view of their pages is made read-only too. Where one
/// cannot be (the kernel writes through it now, or the runtime is changing
/// it), a write through it would not be seen: that block is not clean, and
/// told as written at every asking, until it can. Whether all are covered.
/// With the lock shared (renderer port #6), `claims` holds the claims of
/// `[from, to)`'s blocks, and gets those of the other views' blocks it can
/// have now; a view whose block's claim cannot be had is not covered.
fn cover(g: GuardsRef<'_>, from: usize, to: usize, mut claims: Option<&mut Claims>) -> bool {
    let entries = elsewhere(from * BLOCK, (to - from) * BLOCK);
    if entries.is_empty() {
        return true;
    }
    let rest = rest();
    let mut all = true;
    let mut pending: Vec<usize> = Vec::new();
    let mut deferred: Vec<(usize, Vec<usize>)> = Vec::new();
    // Non-blocking locks for any alias shards outside `g`'s primary shard mask.
    let mut extra: Vec<(usize, ExtraGuard)> = Vec::new();
    for (page, others) in entries {
        let block = page / BLOCK;
        if !GUARDS_STATE.protected.contains(block) {
            continue; // not there, or not ours to guard
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GUARDS_STATE.aliased.insert(block);
        let mut covered = true;
        let mut needed: Vec<usize> = Vec::new();
        for other in others {
            let ob = other / BLOCK;
            if let Some(claims) = claims.as_deref_mut()
                && !claims.holds(ob)
                && !claims.try_one(ob)
            {
                covered = false;
                continue;
            }
            GUARDS_STATE.aliased.insert(ob);
            if GUARDS_STATE.protected.contains(ob) {
                continue;
            }
            let os = guard_shard_of(ob);
            let blocked = if g.has_shard(os) {
                let s = g.shard(os);
                s.left_alone(ob) || s.rests(ob, rest) || s.is_streaming(ob)
            } else {
                let idx = if let Some(pos) = extra.iter().position(|(id, _)| *id == os) {
                    Some(pos)
                } else if let Some(eg) = g.try_lock_extra(os) {
                    extra.push((os, eg));
                    Some(extra.len() - 1)
                } else {
                    None
                };
                match idx {
                    Some(pos) => {
                        let s = &*extra[pos].1;
                        s.left_alone(ob) || s.rests(ob, rest) || s.is_streaming(ob)
                    }
                    // Shard contended: leave `block` unclean so the next asking retries (zero deadlock).
                    None => true,
                }
            };
            if blocked {
                covered = false;
                continue;
            }
            pending.push(ob);
            needed.push(ob);
        }
        if !covered {
            all = false;
            GUARDS_STATE.clean.remove(block);
        } else if !needed.is_empty() {
            deferred.push((block, needed));
        }
    }
    if pending.len() == 1 {
        protect_alias_one(&GUARDS_STATE, pending[0]);
    } else if pending.len() > 1 {
        pending.sort_unstable();
        pending.dedup();
        let mut i = 0;
        while i < pending.len() {
            let run_from = pending[i];
            let mut run_to = run_from + 1;
            i += 1;
            while i < pending.len() && pending[i] == run_to {
                run_to += 1;
                i += 1;
            }
            if run_to - run_from == 1 {
                protect_alias_one(&GUARDS_STATE, run_from);
            } else {
                protect_alias_run(&GUARDS_STATE, run_from, run_to);
            }
        }
    }
    for (block, needed) in deferred {
        if needed.iter().any(|&ob| !GUARDS_STATE.protected.contains(ob)) {
            all = false;
            GUARDS_STATE.clean.remove(block);
        }
    }
    drop(extra);
    all
}

/// Makes one claimed, non-resting alias `block` read-only.
fn protect_alias_one(g: &GuardsState, block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if tripwire_overlaps_block(block) {
        return false;
    }
    // Ahead of the protection, with SeqCst Dekker re-check against `title_memory_unmapping`.
    PROTECTED.set(block);
    if CHANGING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) && is_changing(block) {
        PROTECTED.clear(block);
        return false;
    }
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    #[cfg(feature = "diagnostics")]
    tally(8, started.elapsed().as_nanos() as u64);
    if done && matches!(old, PAGE_READWRITE | PAGE_READONLY) {
        #[cfg(feature = "diagnostics")]
        tally(5, 1);
        g.protected.insert(block);
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Alias, block, 0, 0);
        return true;
    }
    if done {
        // Not for us to guard: as it was.
        // SAFETY: as above.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, old, &mut old) };
    }
    PROTECTED.clear(block);
    false
}

/// Batches `VirtualProtect(PAGE_READONLY)` across a contiguous run of claimed
/// alias blocks `[from, to)`, falling back per block if views or protections differ.
fn protect_alias_run(g: &GuardsState, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) != 0
        && (from..to).any(tripwire_overlaps_block)
    {
        for b in from..to {
            protect_alias_one(g, b);
        }
        return;
    }
    (from..to).for_each(|b| PROTECTED.set(b));
    if CHANGING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst) && (from..to).any(is_changing) {
        for b in from..to {
            if is_changing(b) {
                PROTECTED.clear(b);
            } else {
                protect_alias_one(g, b);
            }
        }
        return;
    }
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: contiguous title view memory; only its protection changes.
    let done = unsafe {
        VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) != 0
    };
    #[cfg(feature = "diagnostics")]
    tally(8, started.elapsed().as_nanos() as u64);
    if done && matches!(old, PAGE_READWRITE | PAGE_READONLY) {
        #[cfg(feature = "diagnostics")]
        tally(5, (to - from) as u64);
        g.protected.insert_all(from..to);
        #[cfg(feature = "diagnostics")]
        for b in from..to {
            crate::faultbook::note(crate::faultbook::What::Alias, b, 0, 0);
        }
        return;
    }
    if done {
        // SAFETY: restore unexpected protection before checking block by block.
        unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, old, &mut old) };
    }
    for b in from..to {
        protect_alias_one(g, b);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &GuardsState, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) != 0
        && (from..to).any(tripwire_overlaps_block)
    {
        for b in from..to {
            if !tripwire_overlaps_block(b) {
                protect_here(g, b, b + 1);
            }
        }
        return;
    }
    let set = |from: usize, to: usize| {
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: title view memory; only its protection changes.
        let mut done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        if done && !matches!(old, PAGE_READWRITE | PAGE_READONLY) {
            // What is there is not for us to guard (code, or memory the
            // title protects itself): as it was.
            // SAFETY: as above.
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, old, &mut old) };
            done = false;
        }
        #[cfg(feature = "diagnostics")]
        {
            if done {
                tally(5, (to - from) as u64);
            }
            tally(8, started.elapsed().as_nanos() as u64);
        }
        done
    };
    let note = |g: &GuardsState, from: usize, to: usize| {
        g.protected.insert_all(from..to);
        g.clean.insert_all(from..to);
    };
    // Ahead of the protection (`PROTECTED`), then re-check `is_changing`
    // (SeqCst Dekker pair with `title_memory_unmapping`) so we never call
    // `VirtualProtect` concurrently with an active unmap/remap under shard read lock.
    (from..to).for_each(|block| PROTECTED.set(block));
    if CHANGING_ACTIVE.load(std::sync::atomic::Ordering::SeqCst)
        && (from..to).any(is_changing)
    {
        for block in from..to {
            if is_changing(block) {
                PROTECTED.clear(block);
            } else if set(block, block + 1) {
                note(g, block, block + 1);
            } else {
                PROTECTED.clear(block);
            }
        }
        return;
    }
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut block = from;
    while block < to {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
        #[cfg(feature = "diagnostics")]
        tally(9, 1);
        if n != 0 && info.State == MEM_COMMIT && matches!(info.Protect, PAGE_READWRITE | PAGE_READONLY) {
            let until = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).min(to);
            if until > block && set(block, until) {
                note(g, block, until);
                block = until;
                continue;
            }
        } else if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
            let until = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).min(to);
            if until > block {
                let holes = known_holes();
                let mut ph_until = 0usize;
                for b in block..until {
                    PROTECTED.clear(b);
                    g.absent.insert(b);
                    if holes {
                        if b < ph_until {
                            g.insert_held(b);
                        } else if let Some(end) = placeholder_end(b) {
                            ph_until = end;
                            g.insert_held(b);
                        }
                    }
                }
                block = until;
                continue;
            }
        }
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else {
            PROTECTED.clear(block);
            if uncommitted_end(block) > block {
                g.absent.insert(block);
                if known_holes() && placeholder(block) {
                    g.insert_held(block);
                }
            }
        }
        block += 1;
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; how many there were. The same physical pages at
/// every other view of them are written by that too (the write that faulted,
/// or is ours, `crate::host::copy_mapped`, goes through one view): they are
/// clean no longer, whether they are opened as well ([`open_one_view`]) or
/// not.
fn open(g: GuardsRef<'_>, start: usize, end: usize) -> usize {
    open_given(g, start, end, None)
}

/// [`open`]; `given`: the blocks of the page's views, for a fault opening one
/// block with the lock shared, which holds their claims ([`open_faulted`]).
fn open_given(_g: GuardsRef<'_>, start: usize, end: usize, given: Option<&[usize]>) -> usize {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return 0;
    }
    let mut blocks: Vec<usize> = GUARDS_STATE.protected.range(first..last).collect();
    let here = blocks.len();
    let aliased: Vec<usize> = GUARDS_STATE.aliased.range(first..last).collect();
    if !aliased.is_empty() {
        let one = open_one_view();
        for block in aliased {
            GUARDS_STATE.aliased.remove(block);
            let others: Vec<usize> = match given {
                Some(given) => given.to_vec(),
                None => crate::inplace::pages_of(block * BLOCK).into_iter().map(|p| p / BLOCK).collect(),
            };
            for other in others {
                if one {
                    // Before the first can be written.
                    GUARDS_STATE.clean.remove(other);
                    PUBLISHED.clear(other);
                    mark_write_epoch_pages(std::iter::once(other));
                    continue;
                }
                GUARDS_STATE.aliased.remove(other);
                if !(first..last).contains(&other) && GUARDS_STATE.protected.contains(other) {
                    blocks.push(other);
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[1].fetch_add((blocks.len() - here) as u64, std::sync::atomic::Ordering::Relaxed);
    }
    unguard(&GUARDS_STATE, &blocks);
    here
}

/// Acquires all shards covering `[start, end)` plus any alias shards in
/// strictly ascending shard order, then opens the range.
fn open_range(start: usize, end: usize, timed: bool) -> usize {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return 0;
    }
    let mut mask = guard_shards_of(first, last);
    loop {
        let g = if timed { guard_shards_write_timed(mask) } else { guard_shards_write(mask) };
        if GUARDS_STATE.aliased.any(first, last) {
            let needed = mask | alias_shards_in(first, last);
            if needed != mask {
                drop(g);
                mask = needed;
                continue;
            }
        }
        return open(g.as_ref(), start, end);
    }
}

/// Gives write access back to `blocks`, guarded ones.
fn unguard(g: &GuardsState, blocks: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if blocks.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // All of them before the first can be written: a write through one is a
    // write of the others' bytes.
    for &block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PUBLISHED.clear(block);
    }
    let unguard_one = |block: usize| {
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe {
            if VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) != 0
                && !matches!(old, PAGE_READONLY | PAGE_READWRITE)
            {
                // Not what we left there: the address is other memory's
                // (code's, say), which keeps its protection.
                VirtualProtect((block * BLOCK) as *const _, BLOCK, old, &mut old);
            }
        }
        PROTECTED.clear(block);
    };
    let mut i = 0;
    while i < blocks.len() {
        let start = blocks[i];
        let mut end = start + 1;
        i += 1;
        while i < blocks.len() && blocks[i] == end {
            end += 1;
            i += 1;
        }
        if end - start == 1 {
            unguard_one(start);
            continue;
        }
        // Only batch VirtualProtect across homogeneous committed RO/RW regions
        // returned by VirtualQuery so mixed protections (e.g. PAGE_EXECUTE_READ)
        // are never overwritten or mis-restored.
        let mut cur = start;
        while cur < end {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery,
            };
            // SAFETY: plain data filled in by VirtualQuery.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: valid out pointer of the right size.
            let n = unsafe { VirtualQuery((cur * BLOCK) as *const _, &mut info, size_of_val(&info)) };
            if n != 0
                && info.State == MEM_COMMIT
                && matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE)
            {
                let until = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).min(end);
                if until > cur + 1 {
                    let mut old = 0;
                    // SAFETY: homogeneous committed RO/RW run we made read-only.
                    let done = unsafe {
                        VirtualProtect(
                            (cur * BLOCK) as *const _,
                            (until - cur) * BLOCK,
                            PAGE_READWRITE,
                            &mut old,
                        ) != 0
                    };
                    if done && matches!(old, PAGE_READONLY | PAGE_READWRITE) {
                        for block in cur..until {
                            PROTECTED.clear(block);
                        }
                        cur = until;
                        continue;
                    }
                    if done {
                        // SAFETY: restore unexpected protection before checking block by block.
                        unsafe {
                            VirtualProtect((cur * BLOCK) as *const _, (until - cur) * BLOCK, old, &mut old);
                        }
                    }
                    for block in cur..until {
                        unguard_one(block);
                    }
                    cur = until;
                    continue;
                }
            }
            unguard_one(cur);
            cur += 1;
        }
    }
    mark_write_epoch_pages(blocks.iter().copied());
    #[cfg(feature = "diagnostics")]
    tally(8, started.elapsed().as_nanos() as u64);
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    tripwire_touch(at, len, false);
    // The length is the title's (`fread`'s size times count, saturated).
    let last = at.saturating_add(len).div_ceil(BLOCK);
    if !light() || PROTECTED.any_acquire(at / BLOCK, last) {
        let opened = open_range(at, at.saturating_add(len), true);
        #[cfg(feature = "diagnostics")]
        {
            tally(6, opened as u64);
            crate::faultbook::note(crate::faultbook::What::Ours, at / BLOCK, last, opened as u64);
        }
        let _ = opened;
    }
    mark_write_epoch_range(at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// The kernel writes title memory `[at, at + len)` until what this returns
/// is dropped ([`kernel_writing`]).
pub struct KernelWrite {
    first: usize,
    last: usize,
    /// Shard bitmask where `(first, last)` was registered; `0` on the lock-free fast-path.
    shards: u64,
}

/// The runtime is about to have the kernel write title memory
/// `[at, at + len)` (a file read): its guarded blocks are opened, as by
/// [`writing`], and no refresh guards them again before what this returns is
/// dropped, at the end of that call. One that did would make the kernel's
/// write fail (the kernel does not fault into our handler): a read cut
/// short, which the title takes for what the file holds.
///
/// Uses an `Acquire`-ordered lock-free fast-path (`!PROTECTED.any_acquire` &&
/// `!absent` && `!VIEWS`, matching `Release`/`SeqCst` stores strictly before
/// `VirtualProtect(PAGE_READONLY)`) so ordinary host/private buffers never take
/// any guard lock on entry or drop.
pub fn kernel_writing(at: usize, len: usize) -> KernelWrite {
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    if len == 0 || first >= last {
        return KernelWrite { first, last: first, shards: 0 };
    }
    tripwire_touch(at, len, false);
    if light()
        && !PROTECTED.any_acquire(first, last)
        && !GUARDS_STATE.absent.0.any_acquire(first, last)
        && !VIEWS.any_acquire(first, last)
    {
        mark_write_epoch_range(at, at.saturating_add(len));
        crate::descriptor::host::remapped(at, len);
        return KernelWrite { first, last: first, shards: 0 };
    }
    let range_shards = guard_shards_of(first, last);
    let mut mask = range_shards;
    loop {
        let mut g = guard_shards_write_timed(mask);
        if GUARDS_STATE.aliased.any(first, last) {
            let needed = mask | alias_shards_in(first, last);
            if needed != mask {
                drop(g);
                mask = needed;
                continue;
            }
        }
        open(g.as_ref(), at, at.saturating_add(len));
        // Nothing known of them stands while they are left alone.
        let absent: Vec<usize> = g.absent.range(first..last).collect();
        for block in absent {
            g.absent.remove(block);
            if g.remove_held(block) {
                PUBLISHED.clear(block);
            }
        }
        let mut m = range_shards;
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            g.shard_mut(s).kernel.push((first, last));
        }
        break;
    }
    mark_write_epoch_range(at, at.saturating_add(len));
    crate::descriptor::host::remapped(at, len);
    KernelWrite { first, last, shards: range_shards }
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if self.shards == 0 || self.first == self.last {
            return;
        }
        let mut g = guard_shards_write(self.shards);
        let mut m = self.shards;
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            let kernel = &mut g.shard_mut(s).kernel;
            if let Some(i) = kernel.iter().position(|&r| r == (self.first, self.last)) {
                kernel.swap_remove(i);
            }
        }
    }
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished;
/// 8, for reads to plain addresses the textures alone (bit 2's look).
/// Unset: bits 1, 2 and 8. `guards`: bit 1 alone. `0`: nothing, counted only.
/// `1`: bits 1 and 2. `slots`: bit 2 alone. `all`: everything.
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("guards") => 1,
        Ok("all") => 15,
        _ => 1 | 2 | 8,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished; of those, the ones passed to the texture
/// pass (bit 8 or 4 of [`page_writes`]).
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & (4 | 8) == 0 {
            mark_write_epoch_range(at, end);
            return;
        }
        WRITTEN[4].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            // (Bit 8 alone: the placed textures under it, not the guards.)
            mark_write_epoch_range(at, end);
            crate::descriptor::host::page_written(at, len);
            return;
        }
    }
    let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
    if PROTECTED.any_acquire(first, last) {
        let mut mask = guard_shards_of(first, last);
        loop {
            let g = guard_shards_write(mask);
            if GUARDS_STATE.aliased.any(first, last) {
                let needed = mask | alias_shards_in(first, last);
                if needed != mask {
                    drop(g);
                    mask = needed;
                    continue;
                }
            }
            let guarded = g.protected.range(first..last).count() as u64;
            if pages {
                WRITTEN[2].fetch_add(guarded, Relaxed);
            }
            if mode & 1 != 0 && guarded != 0 {
                open(g.as_ref(), at, end);
            }
            break;
        }
    }
    mark_write_epoch_range(at, end);
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again - and are, the pages noted written, which also has the next check
/// guard the new view ([`cover`]).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    // (A block is noted in `PROTECTED` ahead of the look at its page's views
    // that did not find this one yet.)
    if !ranges.iter().any(|&(at, n)| PROTECTED.any_acquire(at / BLOCK, (at + n).div_ceil(BLOCK))) {
        return;
    }
    // All of them, once one is guarded: another may have been made clean
    // since that look, relying on those that are opened here. And all
    // together: none may be writable while another is clean (who asks about
    // that one by its bits takes no lock).
    let mask = ranges.iter().fold(0u64, |m, &(at, n)| m | guard_shards_of(at / BLOCK, (at + n).div_ceil(BLOCK)));
    let g = guard_shards_write(mask);
    let blocks: Vec<usize> =
        ranges.iter().flat_map(|&(at, n)| g.protected.range(at / BLOCK..(at + n).div_ceil(BLOCK)).collect::<Vec<_>>()).collect();
    #[cfg(feature = "diagnostics")]
    ALIASES[2].fetch_add(blocks.len() as u64, std::sync::atomic::Ordering::Relaxed);
    unguard(&g, &blocks);
}

/// The runtime is about to unmap a view of which it maps a part again at
/// once ([`title_memory_unmapping`] of all of it comes next): the title's
/// threads may write through that part before we are told of it
/// ([`views_mapped`]). What is guarded of its pages at other views is opened,
/// and not guarded again before the change is done ([`Remapping`]).
pub fn title_memory_splitting(at: usize, len: usize) {
    let others: Vec<usize> = elsewhere(at, len).into_iter().flat_map(|(_, others)| others).collect();
    if others.is_empty() {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut mask = guard_shards_of(first, last) | guard_shards_of_blocks(others.iter().map(|&o| o / BLOCK));
    loop {
        let g = guard_shards_write(mask);
        let mut needed = mask;
        if GUARDS_STATE.aliased.any(first, last) {
            needed |= alias_shards_in(first, last);
        }
        for &other in &others {
            let ob = other / BLOCK;
            if GUARDS_STATE.aliased.contains(ob) {
                needed |= alias_shards_in(ob, ob + 1);
            }
        }
        if needed != mask {
            drop(g);
            mask = needed;
            continue;
        }
        // (The view itself too: the part that stays is read-only until then. And
        // all of them told as written before the first is writable.)
        for &other in &others {
            g.clean.remove(other / BLOCK);
            PUBLISHED.clear(other / BLOCK);
        }
        open(g.as_ref(), at, at.saturating_add(len));
        for other in others {
            open(g.as_ref(), other, other + BLOCK);
            if CHANGING.get().0 {
                CHANGING.set((true, true));
                note_changing(other / BLOCK, other / BLOCK + 1);
            }
        }
        break;
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

impl Drop for Remapping {
    fn drop(&mut self) {
        // The change is done: what is there now may be guarded.
        if CHANGING.replace((false, false)).1 {
            clear_changing();
        }
        CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// How often the runtime began or ended a change of what title memory
/// shows: odd while one is under way.
static CHANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// This thread holds [`Remapping`]; and put blocks in
    /// [`Guards::changing`].
    static CHANGING: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((false, false)) };
}

/// For [`again_after_change`]: to be asked before what may fail.
pub(crate) fn changes() -> u64 {
    CHANGES.load(std::sync::atomic::Ordering::SeqCst)
}

/// `f` failed to read or write title memory, at some time since [`changes`]
/// said `since`. If the runtime changed a mapping in that time, the pages
/// may have been away for that moment only ([`MAPPING`]): again, when the
/// change is done. Whether `f` did it.
pub(crate) fn again_after_change(since: u64, mut f: impl FnMut() -> bool) -> bool {
    let mut since = since;
    for _ in 0..4 {
        let now = change_done();
        if now == since {
            return false;
        }
        since = now;
        if f() {
            return true;
        }
    }
    false
}

/// Returns when the change of a mapping that is under way, if one is, is
/// over (or has been going on for 2 ms): pages it took away for its moment
/// are back. Another may have begun.
pub(crate) fn change_over() {
    let now = changes();
    if now.is_multiple_of(2) {
        return;
    }
    let started = std::time::Instant::now();
    while changes() == now && started.elapsed().as_millis() < 2 {
        for _ in 0..50 {
            std::hint::spin_loop();
        }
    }
}

/// [`changes`], when no change is under way (or one has been for a second).
fn change_done() -> u64 {
    let mut tries = 0;
    let mut started = None;
    loop {
        let now = changes();
        if now.is_multiple_of(2) {
            return now;
        }
        tries += 1;
        if tries < 100 {
            std::hint::spin_loop();
            continue;
        }
        if started.get_or_insert_with(std::time::Instant::now).elapsed().as_secs() >= 1 {
            return now;
        }
        // SAFETY: gives the processor to another thread.
        unsafe { windows_sys::Win32::System::Threading::Sleep(u32::from(tries > 200)) };
    }
}

/// Faults of title threads that met a change of mapping and went on after
/// it ([`came_back`]).
static MET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A title thread's access to `at` (`write`: a write) faulted. Whether it
/// can be made now: if a change of mapping is under way, after it.
///
/// The console changes one page's mapping and no other's. Here the rest of
/// a view is unmapped and mapped again ([`MAPPING`]), and a title thread that
/// uses a page of it just then, which the title never gave up, faulted
/// ("reserved, never committed", or read-only for the moment between our
/// forgetting a guarded block and its unmapping). It waits here instead.
/// Memory that is not there with no change under way is not waited for.
fn came_back(at: usize, write: bool) -> bool {
    let there = || {
        if !write {
            return readable_view(at);
        }
        // (A refresh may have guarded the page that came back.)
        open_faulted(at).0 != 0 || writable(at)
    };
    let before = changes();
    if there() || again_after_change(before, there) {
        MET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    false
}

/// Whether `at` is in a committed page of a view that can be read.
fn readable_view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Type == MEM_MAPPED
        && matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE)
}

/// Title memory that could not be written with what was to go there (what a
/// shader wrote, a title's copy): counted, and said at the 1st, 2nd, 4th, 8th
/// and so on. A page the title unmapped loses nothing by it; a page that is
/// there does.
pub(crate) fn lost(what: &'static str, at: usize, len: usize) {
    static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = LOST.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let there = writable(at);
        tracing::warn!(n, what, at = format_args!("{at:#x}"), len, there, "title memory not written");
    }
}


/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    // For the title's own threads ([`came_back`]).
    install();
    let met = MET.load(std::sync::atomic::Ordering::Relaxed);
    if met != 0 {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if met >= 2 * SAID.load(std::sync::atomic::Ordering::Relaxed) {
            SAID.store(met, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(met, "accesses of title threads that waited for a change of mapping");
        }
    }
    let changing = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    CHANGING.set((true, false));
    Remapping(changing)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped title memory `[at, at + len)`: what it shows is new,
/// so it is no longer guarded (nor known unmapped), and read-write.
///
/// The view is there before we are told of it, and a copy's refresh on
/// another thread may have guarded its blocks in between (it finds blocks
/// it knew absent committed, and holds [`GUARDS`], which we wait for).
/// Forgetting such a block and leaving it read-only would make the title's
/// next write to it an access violation nobody takes (`on_write` finds it
/// neither guarded nor writable) and a kernel read into it fail, so the
/// write access is given back.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    tripwire_touch(at, len, false);
    crate::texback::remapped(at, len);
    notify_title_memory_changed(at, len);
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    invalidate_reservations(first, last);
    if !range_needs_guard_update(first, last, false) {
        return;
    }
    let mut mask = guard_shards_of(first, last);
    loop {
        let mut g = guard_shards_write(mask);
        if GUARDS_STATE.aliased.any(first, last) {
            let needed = mask | alias_shards_in(first, last);
            if needed != mask {
                drop(g);
                mask = needed;
                continue;
            }
        }
        clear_changed_guards(&mut g, first, last);
        open(g.as_ref(), at, at.saturating_add(len));
        break;
    }
}

/// The runtime is about to unmap title memory `[at, at + len)`: it is
/// nothing of ours any more. Its protection goes with it. Until the change
/// is done ([`Remapping`]) no refresh guards what is there: one that did
/// before the unmapping would leave the address guarded, clean and known by
/// its bits ([`PUBLISHED`]) with no view behind it.
pub fn title_memory_unmapping(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    tripwire_touch(at, len, true);
    crate::texback::remapped(at, len);
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    invalidate_reservations(first, last);
    // Record `changing` ahead of checking `VIEWS`/`GUARDS_STATE.aliased` so any
    // concurrent `guard`/`guard_shared`/`protect_alias` either observes
    // `CHANGING_ACTIVE` or is observed below and serialized through its shard.
    if CHANGING.get().0 {
        CHANGING.set((true, true));
        note_changing(first, last);
    }
    let had_views = VIEWS.any_acquire(first, last);
    if had_views {
        forget_views(first, last);
    }
    notify_title_memory_changed(at, len);
    if !range_needs_guard_update(first, last, had_views) {
        return;
    }
    let mut g = guard_shards_write(guard_shards_of(first, last));
    clear_changed_guards(&mut g, first, last);
    if g.protected.any(first, last) {
        let blocks: Vec<usize> = g.protected.range(first..last).collect();
        for block in blocks {
            g.protected.remove(block);
            g.clean.remove(block);
            PUBLISHED.clear(block);
            PROTECTED.clear(block);
        }
        drop(g);
        mark_write_epoch_range(at, at.saturating_add(len));
    }
}

/// The title is about to free or decommit memory of its own,
/// `[at, at + len)`, or with no length the whole allocation at `at`: what
/// was noted of its blocks (guarded, clean, absent) would be taken for true
/// of the next memory at the address.
pub fn title_memory_freed(at: usize, len: usize) {
    let freed_len = if len != 0 {
        len
    } else {
        allocation_end(at, at).saturating_sub(at)
    };
    tripwire_touch(at, freed_len.max(1), false);
    if freed_len != 0 {
        crate::inplace::title_memory_freeing(at, freed_len);
        invalidate_reservations(at / PAGE, at.saturating_add(freed_len).div_ceil(PAGE));
    } else {
        invalidate_reservations(at / PAGE, at / PAGE + 1);
    }
    if len != 0 {
        forget_views(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    } else if VIEWS.any_acquire(at / BLOCK, at / BLOCK + 1) {
        forget_views(at / BLOCK, at.saturating_add(freed_len).div_ceil(BLOCK).max(at / BLOCK + 1));
    }
    let first = at / BLOCK;
    let end = if len != 0 {
        at.saturating_add(len).div_ceil(BLOCK)
    } else if freed_len != 0 {
        at.saturating_add(freed_len).div_ceil(BLOCK)
    } else {
        let noted = [
            GUARDS_STATE.protected.range(first..).next(),
            GUARDS_STATE.absent.range(first..).next(),
        ]
        .into_iter()
        .flatten()
        .min();
        match noted {
            Some(n) => allocation_end(at, n * BLOCK).div_ceil(BLOCK),
            None => return,
        }
    };
    if first >= end {
        return;
    }
    let mut mask = guard_shards_of(first, end);
    loop {
        let mut g = guard_shards_write(mask);
        if GUARDS_STATE.aliased.any(first, end) {
            let needed = mask | alias_shards_in(first, end);
            if needed != mask {
                drop(g);
                mask = needed;
                continue;
            }
        }
        let absent: Vec<usize> = g.absent.range(first..end).collect();
        for block in absent {
            g.absent.remove(block);
            if g.remove_held(block) {
                PUBLISHED.clear(block);
            }
        }
        let mut m = guard_shards_of(first, end);
        while m != 0 {
            let s = m.trailing_zeros() as usize;
            m &= m - 1;
            let foreign: Vec<usize> = g.shard(s).foreign.range(first..end).copied().collect();
            for block in foreign {
                g.shard_mut(s).foreign.remove(&block);
            }
        }
        g.clear_opened(first, end);
        g.clear_streaming(first, end);
        // (Memory that is still a view is not freed by that call, which fails:
        // it must not stay read-only.)
        open(g.as_ref(), at, end * BLOCK);
        break;
    }
}

/// The end of the allocation at `at`, if `inside` lies in it; else `at`.
fn allocation_end(at: usize, inside: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let ask = |address: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
        (n != 0 && info.AllocationBase as usize == at)
            .then(|| (info.BaseAddress as usize).saturating_add(info.RegionSize))
    };
    let mut end = at;
    let mut next = ask(inside.max(at));
    while let Some(to) = next {
        end = to;
        next = ask(to);
    }
    end
}

/// What is known of `[first, last)` as view memory ([`VIEWS`]) is forgotten.
fn forget_views(first: usize, last: usize) {
    if !VIEWS.any(first, last) {
        return;
    }
    let set: Vec<usize> = VIEWS.set_in(first, last).collect();
    for block in set {
        VIEWS.clear(block);
    }
}

/// Tells subsystems that keep track of what title memory shows.
fn notify_title_memory_changed(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
}

/// Fast-path check under the shared shard locks: whether `[first, last)` touches
/// any guarded, absent, aliased, foreign, streaming, or view-tracked block
/// (avoiding exclusive shard write locks on cold ranges).
fn range_needs_guard_update(first: usize, last: usize, had_views: bool) -> bool {
    if first >= last {
        return false;
    }
    if had_views
        || PROTECTED.any_acquire(first, last)
        || VIEWS.any_acquire(first, last)
        || GUARDS_STATE.streaming.0.any_acquire(first, last)
    {
        return true;
    }
    let mask = guard_shards_of(first, last);
    let g = guard_shards_read(mask);
    if g.protected.any(first, last) || g.absent.any(first, last) || g.aliased.any(first, last) {
        return true;
    }
    let mut m = mask;
    while m != 0 {
        let s = m.trailing_zeros() as usize;
        m &= m - 1;
        let foreign = &g.shard(s).foreign;
        if !foreign.is_empty() && foreign.range(first..last).next().is_some() {
            return true;
        }
    }
    false
}

/// Clears absent/held/foreign/aliased/opened/streaming guard metadata in `[first, last)`.
fn clear_changed_guards(g: &mut GuardsWrite, first: usize, last: usize) {
    let range = first..last;
    if g.absent.any(first, last) {
        let absent: Vec<usize> = g.absent.range(range.clone()).collect();
        for block in absent {
            g.absent.remove(block);
            if g.remove_held(block) {
                PUBLISHED.clear(block);
            }
        }
    }
    let mut m = guard_shards_of(first, last);
    while m != 0 {
        let s = m.trailing_zeros() as usize;
        m &= m - 1;
        let foreign: Vec<usize> = g.shard(s).foreign.range(range.clone()).copied().collect();
        for block in foreign {
            g.shard_mut(s).foreign.remove(&block);
        }
    }
    if g.aliased.any(first, last) {
        let aliased: Vec<usize> = g.aliased.range(range).collect();
        for block in aliased {
            g.aliased.remove(block);
        }
    }
    g.clear_opened(first, last);
    g.clear_streaming(first, last);
}

/// Mode-1 GPU-only copyback one-shot `PAGE_NOACCESS` VEH tripwire (`ARCH-06`).
///
/// Promoted Mode-1 pure-GPU card buffers arm their interior 4 KiB-aligned guest
/// pages with `PAGE_NOACCESS`. True pure-GPU buffers incur 0 faults/frame. If
/// any title CPU thread (or `safecopy`/kernel path) ever touches the range, the
/// single fault immediately restores `PAGE_READWRITE`, demotes the buffer back
/// to normal copyback, and retries cleanly.
struct TripwireEntry {
    id: u64,
    start: usize,
    end: usize,
    /// Exact committed sub-ranges whose protection was changed to `PAGE_NOACCESS`.
    protected: Vec<(usize, usize)>,
}

struct TripwireState {
    active: Vec<TripwireEntry>,
    /// Bounded ring of recently disarmed `(start, end)` ranges so a second
    /// concurrent thread faulting at the exact instant of disarm still retries.
    disarmed: Vec<(usize, usize)>,
}

static TRIPWIRE: Mutex<TripwireState> = Mutex::new(TripwireState {
    active: Vec::new(),
    disarmed: Vec::new(),
});
static TRIPWIRE_ACTIVE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TRIPWIRE_LO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
static TRIPWIRE_HI: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TRIPWIRE_DISARMED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static TRIPWIRE_TRIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
pub(crate) fn tripwire_active() -> bool {
    TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) != 0
}

pub(crate) fn tripwire_stats() -> (u64, u64) {
    (
        TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) as u64,
        TRIPWIRE_TRIPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

fn tripwire_refresh_bounds(state: &TripwireState) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for e in &state.active {
        lo = lo.min(e.start);
        hi = hi.max(e.end);
    }
    TRIPWIRE_LO.store(lo, std::sync::atomic::Ordering::Relaxed);
    TRIPWIRE_HI.store(hi, std::sync::atomic::Ordering::Relaxed);
    TRIPWIRE_ACTIVE.store(state.active.len(), std::sync::atomic::Ordering::Release);
    TRIPWIRE_DISARMED.store(state.disarmed.len(), std::sync::atomic::Ordering::Release);
}

fn tripwire_push_disarmed(state: &mut TripwireState, start: usize, end: usize) {
    const MAX_DISARMED: usize = 64;
    if state.disarmed.len() >= MAX_DISARMED {
        state.disarmed.remove(0);
    }
    state.disarmed.push((start, end));
}

fn tripwire_restore(protected: &[(usize, usize)]) {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    for &(base, len) in protected {
        let mut old = 0u32;
        // SAFETY: sub-range previously changed from RW/RO to PAGE_NOACCESS by `tripwire_arm`.
        unsafe {
            if VirtualProtect(base as *const _, len, PAGE_READWRITE, &mut old) != 0
                && !matches!(old, PAGE_NOACCESS | PAGE_READWRITE | PAGE_READONLY)
            {
                VirtualProtect(base as *const _, len, old, &mut old);
            }
        }
    }
}

fn tripwire_overlaps_block(block: usize) -> bool {
    if TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return false;
    }
    let start = block * BLOCK;
    let end = start.saturating_add(BLOCK);
    if end <= TRIPWIRE_LO.load(std::sync::atomic::Ordering::Relaxed)
        || start >= TRIPWIRE_HI.load(std::sync::atomic::Ordering::Relaxed)
    {
        return false;
    }
    let state = TRIPWIRE.lock().unwrap_or_else(|e| e.into_inner());
    state.active.iter().any(|e| e.start < end && e.end > start)
}

/// Arms a one-shot `PAGE_NOACCESS` tripwire over the interior 4 KiB-aligned
/// committed pages of `[start, start + len)` owned by shadow `id`.
pub(crate) fn tripwire_arm(id: u64, start: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect,
        VirtualQuery,
    };
    install();
    let Some(raw_end) = start.checked_add(len) else {
        return false;
    };
    let arm_start = start.next_multiple_of(KERNEL_PAGE);
    let arm_end = raw_end & !(KERNEL_PAGE - 1);
    if arm_start >= arm_end {
        return false;
    }
    // Check guard shards BEFORE taking `MAPPING` so we never hold `MAPPING`
    // across `GUARD_SHARD_LOCKS` (avoids lock-order inversion with callers
    // holding guard shards across `title_memory_remapping`).
    let (b_first, b_last) = (arm_start / BLOCK, arm_end.div_ceil(BLOCK));
    {
        let range_shards = guard_shards_of(b_first, b_last);
        let mut mask = range_shards;
        loop {
            let g = guard_shards_write(mask);
            if GUARDS_STATE.aliased.any(b_first, b_last) {
                let needed = mask | alias_shards_in(b_first, b_last);
                if needed != mask {
                    drop(g);
                    mask = needed;
                    continue;
                }
            }
            let mut m = range_shards;
            while m != 0 {
                let s = m.trailing_zeros() as usize;
                m &= m - 1;
                if g.shard(s).kernel.iter().any(|&(a, b)| a < b_last && b > b_first) {
                    return false;
                }
            }
            if g.protected.any(b_first, b_last) {
                open(g.as_ref(), arm_start, arm_end);
            }
            break;
        }
    }
    let Some(_reading) = title_memory_reading_now() else {
        return false;
    };
    // If another thread re-guarded a block in `[b_first, b_last)` between dropping
    // `GUARDS` and taking `MAPPING`, bail cleanly rather than fighting over RO guards.
    if PROTECTED.any(b_first, b_last) {
        return false;
    }
    // Collect committed RW/RO sub-ranges without touching code, guard, or unmapped pages.
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let mut cur = arm_start;
    while cur < arm_end {
        // SAFETY: plain out buffer for VirtualQuery.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(cur as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            break;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(arm_end);
        if region_end <= cur {
            break;
        }
        if info.State == MEM_COMMIT && matches!(info.Protect, PAGE_READWRITE | PAGE_READONLY) {
            candidates.push((cur, region_end - cur));
        }
        cur = region_end;
    }
    if candidates.is_empty() {
        return false;
    }
    // Hold `TRIPWIRE` while protecting so any concurrent fault on the first
    // sub-range blocks until arming completes and then disarms all sub-ranges.
    let mut state = TRIPWIRE.lock().unwrap_or_else(|e| e.into_inner());
    if state.active.iter().any(|e| e.id == id || (e.start < arm_end && e.end > arm_start)) {
        return false;
    }
    let mut protected = Vec::with_capacity(candidates.len());
    for (sub_start, sub_len) in candidates {
        let mut old = 0u32;
        // SAFETY: committed RW/RO title memory verified above under `title_memory_reading_now`.
        let ok = unsafe { VirtualProtect(sub_start as *const _, sub_len, PAGE_NOACCESS, &mut old) != 0 };
        if ok && matches!(old, PAGE_READWRITE | PAGE_READONLY) {
            protected.push((sub_start, sub_len));
        } else if ok {
            // SAFETY: restore unexpected protection unchanged.
            unsafe { VirtualProtect(sub_start as *const _, sub_len, old, &mut old) };
        }
    }
    if protected.is_empty() {
        return false;
    }
    state.active.push(TripwireEntry {
        id,
        start: arm_start,
        end: arm_end,
        protected,
    });
    tripwire_refresh_bounds(&state);
    true
}

/// Disarms and restores `PAGE_READWRITE` for any active tripwire owned by `id`.
pub(crate) fn tripwire_disarm_id(id: u64) {
    if TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return;
    }
    let mut state = TRIPWIRE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = state.active.iter().position(|e| e.id == id) {
        let entry = state.active.swap_remove(i);
        tripwire_restore(&entry.protected);
        tripwire_push_disarmed(&mut state, entry.start, entry.end);
        tripwire_refresh_bounds(&state);
        mark_write_epoch_range(entry.start, entry.end);
    }
}

/// Disarms and demotes any active tripwire overlapping `[at, at + len)`.
/// Called by `safecopy` and kernel/remap entry points so tripwire pages are
/// never treated as unmapped zeros or failed kernel reads.
pub(crate) fn tripwire_touch(at: usize, len: usize, unmapping: bool) -> bool {
    if len == 0 || TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return false;
    }
    let end = at.saturating_add(len);
    if end <= TRIPWIRE_LO.load(std::sync::atomic::Ordering::Relaxed)
        || at >= TRIPWIRE_HI.load(std::sync::atomic::Ordering::Relaxed)
    {
        return false;
    }
    let mut hit_ids: Vec<u64> = Vec::new();
    {
        let mut state = TRIPWIRE.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = 0;
        while i < state.active.len() {
            if state.active[i].start < end && state.active[i].end > at {
                let entry = state.active.swap_remove(i);
                if !unmapping {
                    tripwire_restore(&entry.protected);
                }
                tripwire_push_disarmed(&mut state, entry.start, entry.end);
                mark_write_epoch_range(entry.start, entry.end);
                hit_ids.push(entry.id);
            } else {
                i += 1;
            }
        }
        if !hit_ids.is_empty() {
            tripwire_refresh_bounds(&state);
        }
    }
    if hit_ids.is_empty() {
        return false;
    }
    let reason = if unmapping {
        "title memory unmap/remap"
    } else {
        "safecopy/kernel access"
    };
    for id in hit_ids {
        crate::shadow::tripwire_demote(id, at, 0, 0, reason);
    }
    true
}

fn accessible_for(at: usize, access: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain out struct.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return false;
    }
    if access == 0 {
        matches!(info.Protect, PAGE_READWRITE | PAGE_READONLY)
    } else {
        info.Protect == PAGE_READWRITE
    }
}

/// Checks whether a read/write access violation at `at` hit an active (or
/// concurrently disarmed) Mode-1 GPU-only tripwire. Restores `PAGE_READWRITE`,
/// triggers one-shot permanent demotion, and returns `true` to resume execution.
fn handle_tripwire_fault(at: usize, access: usize, rip: usize, is_ours: bool) -> bool {
    let active = TRIPWIRE_ACTIVE.load(std::sync::atomic::Ordering::Acquire);
    let disarmed = TRIPWIRE_DISARMED.load(std::sync::atomic::Ordering::Acquire);
    if active == 0 && disarmed == 0 {
        return false;
    }
    let maybe_active = active != 0
        && at >= TRIPWIRE_LO.load(std::sync::atomic::Ordering::Relaxed)
        && at < TRIPWIRE_HI.load(std::sync::atomic::Ordering::Relaxed);
    if !maybe_active && disarmed == 0 {
        return false;
    }
    let tripped = {
        let mut state = TRIPWIRE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = state.active.iter().position(|e| e.start <= at && at < e.end) {
            let entry = state.active.swap_remove(i);
            tripwire_restore(&entry.protected);
            tripwire_push_disarmed(&mut state, entry.start, entry.end);
            tripwire_refresh_bounds(&state);
            Some(entry.id)
        } else if state.disarmed.iter().any(|&(s, e)| s <= at && at < e) {
            drop(state);
            return accessible_for(at, access);
        } else {
            None
        }
    };
    if let Some(id) = tripped {
        TRIPWIRE_TRIPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let reason = if is_ours {
            "safecopy tripwire fault"
        } else {
            "cpu tripwire fault"
        };
        crate::shadow::tripwire_demote(id, at, access, rip, reason);
        return true;
    }
    false
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        let _fault = crate::timeline::span("a fault looked at");
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const READ: usize = 0;
        const WRITE: usize = 1;
        // A fault of our own copies (`crate::safecopy`): whatever it is, if
        // it is not a block we guard, the copy fails and goes on from there.
        let ours = crate::safecopy::failing_exit(r.ExceptionAddress as usize);
        // Mode-1 GPU-only one-shot `PAGE_NOACCESS` tripwire: must run BEFORE
        // the `ours` failing-exit and `WRITE`-only checks so neither title CPU
        // reads nor `safecopy` ever treat tripwired pages as unmapped/crashes.
        if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.NumberParameters >= 2
            && matches!(r.ExceptionInformation[0], READ | WRITE)
            && handle_tripwire_fault(
                r.ExceptionInformation[1],
                r.ExceptionInformation[0],
                r.ExceptionAddress as usize,
                ours.is_some(),
            )
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // SAFETY: as above.
        let context = unsafe { (*info).ContextRecord };
        let fail = move |exit: usize| {
            // SAFETY: the context the system passes, to go on from.
            unsafe { (*context).Rip = exit as u64 };
            EXCEPTION_CONTINUE_EXECUTION
        };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return match ours {
                Some(exit) => fail(exit),
                None if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                    && r.NumberParameters >= 2
                    && r.ExceptionInformation[0] == READ
                    && came_back(r.ExceptionInformation[1], false) =>
                {
                    EXCEPTION_CONTINUE_EXECUTION
                }
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        let at = r.ExceptionInformation[1];
        let block = at / BLOCK;
        if ours.is_none()
            && PROTECTED.any(block, block + 1)
            // SAFETY: the context the system passes, to go on from.
            && crate::titlewrite::store(unsafe { &mut *context }, at)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Another thread may have opened the block since this one faulted
        // (then nothing waits for the lock), and a copy may have guarded it
        // again since that (then it is ours again, not read-only for some
        // other reason).
        for _ in 0..3 {
            if !PROTECTED.any(block, block + 1) && writable(at) {
                #[cfg(feature = "diagnostics")]
                crate::faultbook::note(crate::faultbook::What::Open, block, 0, 0);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            #[cfg(feature = "diagnostics")]
            let began = std::time::Instant::now();
            // (Let go again before the next look: the runtime wants the
            // guards to end a change.)
            let (opened, _waited) = open_faulted(at);
            #[cfg(feature = "diagnostics")]
            let waited = _waited;
            if opened != 0 {
                crate::port_descriptors::faulted(at, r.ExceptionAddress as usize);
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    let ns = waited.as_nanos().min(u128::from(u32::MAX)) as usize;
                    let held = began.elapsed().saturating_sub(waited).as_nanos() as u64;
                    crate::faultbook::note(crate::faultbook::What::Fault, block, opened << 32 | ns, held);
                    crate::faultbook::note(crate::faultbook::What::Site, r.ExceptionAddress as usize, block, 0);
                }
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        match ours {
            Some(exit) => fail(exit),
            None if came_back(at, true) => EXCEPTION_CONTINUE_EXECUTION,
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod parts_tests {
    use super::*;

    #[test]
    fn the_parts_of_a_page_written_since_it_was_taken() {
        let mut j = Shard::new();
        assert_eq!(j.parts_since(7, 0), WHOLE);
        assert_eq!(j.note(7, WHOLE, 3), 0);
        // Parts 0 and 5 at 4, part 5 again at 6: at 2 the whole page is due.
        assert_eq!(j.note(7, 0b10_0001, 4), 3);
        assert_eq!(j.note(7, 0b10_0000, 6), 4);
        assert_eq!(j.parts_since(7, 2), WHOLE);
        assert_eq!(j.parts_since(7, 3), 0b10_0001);
        assert_eq!(j.parts_since(7, 4), 0b10_0000);
        assert_eq!(j.parts_since(7, 5), 0b10_0000);
        assert_eq!(j.parts_since(7, 6), 0);
        // Written whole: all of it, whoever took it before.
        assert_eq!(j.note(7, WHOLE, 8), 6);
        assert_eq!(j.parts_since(7, 7), WHOLE);
        // A page never written before: what was not named was never written.
        assert_eq!(j.note(9, 0b100, 9), 0);
        assert_eq!(j.parts_since(9, 5), 0b100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal may hold pages of another test's memory that was here.
        let (none, before) = writes(start, end, false).expect("watched");
        assert!(before.iter().all(|&(_, w)| w < none), "{before:?}");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let new = |w: Vec<(usize, u64)>| w.into_iter().filter(|w| w.1 > none).collect::<Vec<_>>();
        assert_eq!(new(after), [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(new(again), [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // Before the view goes, as the runtime does: the address may be
        // another test's the moment it is free.
        title_memory_unmapping(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    /// A placeholder of `blocks` blocks, each its own.
    fn placeholder(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder, split.
        unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            for block in 0..blocks - 1 {
                assert_ne!(
                    VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                    0
                );
            }
            p
        }
    }

    /// Private memory of `protect` in place of the placeholder of one block.
    fn private_at(at: usize, protect: u32) {
        use windows_sys::Win32::System::Memory::{MEM_REPLACE_PLACEHOLDER, VirtualAlloc2};
        // SAFETY: our placeholder.
        let got = unsafe {
            VirtualAlloc2(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                at as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                protect,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(got as usize, at);
    }

    fn protection(at: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data the query fills in.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(at as *const _, &mut info, size_of_val(&info));
            info.Protect
        }
    }

    /// A refresh between the opening of a kernel read's destination and the
    /// read itself leaves it open: the read lands, and is seen.
    #[test]
    fn a_refresh_leaves_open_what_the_kernel_is_writing() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        let pages = |block: usize| at / PAGE + block..at / PAGE + block + 1;
        assert!(writes(at, at + 2 * BLOCK, true).is_some());
        assert!(published(at, at + 2 * BLOCK) || !light());
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        file.rewind().unwrap();
        let writing = kernel_writing(at + 100, 13);
        // The refresh: the block being written is new to it every time, and
        // stays open; the other is as it was.
        for _ in 0..2 {
            assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
            let (now, found) = writes(at, at + BLOCK, true).expect("a view");
            assert!(found.contains(&(at / PAGE, now)));
            assert!(writable(at) && !writable(at + BLOCK));
            assert!(!published(at, at + BLOCK));
        }
        // SAFETY: 13 bytes of our view, nothing else refers to them.
        let into = unsafe { std::slice::from_raw_parts_mut((at + 100) as *mut u8, 13) };
        assert_eq!(file.read(into).expect("the kernel's write lands"), 13);
        assert_eq!(into, b"from the disk");
        drop(writing);
        // Seen by the next refresh, and guarded again.
        assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
        assert!(!writable(at));
        assert_eq!(guard(at, at + 2 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// A hole the kernel is to write is no known hole meanwhile, nor is one
    /// the runtime has begun to change; after that it is again.
    #[test]
    fn what_is_left_alone_is_no_known_hole() {
        let p = placeholder(3);
        let known = |block: usize| published(p + block * BLOCK, p + (block + 1) * BLOCK);
        let ask = || writes(p, p + 3 * BLOCK, true).expect("as a view");
        ask();
        let holes = known_holes();
        assert!((0..3).all(known) == holes);
        let writing = kernel_writing(p + BLOCK + 5, 100);
        for _ in 0..2 {
            assert!(known(0) == holes && !known(1) && known(2) == holes);
            let (now, written) = ask();
            assert!(written.contains(&(p / PAGE + 1, now)));
            assert!(written.iter().all(|w| w.0 == p / PAGE + 1 || w.1 < now));
        }
        assert!(!guards().held.contains(p / BLOCK + 1));
        drop(writing);
        ask();
        assert!((0..3).all(known) == holes);
        {
            let _changing = title_memory_remapping();
            title_memory_unmapping(p, BLOCK);
            ask();
            assert!(!known(0) && !guards().held.contains(p / BLOCK));
        }
        ask();
        assert!((0..3).all(known) == holes);
        title_memory_freed(p, BLOCK);
        assert!(!known(0) && !guards().held.contains(p / BLOCK));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A refresh during a change of mapping guards nothing the runtime has
    /// begun to unmap: no note of it stays behind.
    #[test]
    fn nothing_is_guarded_between_our_forgetting_it_and_its_unmapping() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        assert!(writes(at, at + BLOCK, true).is_some());
        let changing = title_memory_remapping();
        title_memory_unmapping(at, BLOCK);
        assert!(writes(at, at + BLOCK, true).is_some());
        {
            let g = guards();
            assert!(!g.protected.contains(at / BLOCK) && !g.clean.contains(at / BLOCK));
        }
        assert!(!published(at, at + BLOCK));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
        drop(changing);
        assert!(!guards().left_alone(at / BLOCK));
    }

    /// Memory at an address where blocks were guarded or absent is not
    /// taken for them: code stays code, private memory is not guarded.
    #[test]
    fn other_memory_at_a_guarded_address_keeps_its_protection() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3, PAGE_EXECUTE_READ,
            PAGE_EXECUTE_READWRITE, UnmapViewOfFile2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(3);
        // SAFETY: a section's block in place of the first placeholder.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, p);
        // One block guarded, two absent.
        assert_eq!(guard(p, p + 3 * BLOCK).map(|w| w.len()), Some(3));
        // Not told of what comes: the view goes, code is there; and private
        // memory in the holes.
        // SAFETY: our view.
        assert_ne!(unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) }, 0);
        private_at(p, PAGE_EXECUTE_READ);
        private_at(p + BLOCK, PAGE_READWRITE);
        private_at(p + 2 * BLOCK, PAGE_EXECUTE_READWRITE);
        // (The runtime's allocations it tells of, or the holes stay known.)
        title_memory_allocated(p + BLOCK, 2 * BLOCK);
        let as_made = || {
            assert_eq!(protection(p), PAGE_EXECUTE_READ);
            assert_eq!(protection(p + BLOCK), PAGE_READWRITE);
            assert_eq!(protection(p + 2 * BLOCK), PAGE_EXECUTE_READWRITE);
        };
        let holes = [p / PAGE + 1, p / PAGE + 2];
        for _ in 0..2 {
            let written = guard(p, p + 3 * BLOCK).expect("as a view");
            assert!(holes.iter().all(|page| written.contains(page)), "copied every time");
            as_made();
            let g = guards();
            assert!(holes.iter().all(|&block| !g.held.contains(block) && !g.absent.contains(block)));
        }
        writing(p, BLOCK);
        as_made();
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our allocations.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// Memory the title frees leaves no note of its blocks, whether it names
    /// its length or the allocation.
    #[test]
    fn freed_memory_leaves_no_notes() {
        let p = placeholder(3);
        let noted = |block: usize| {
            let g = guards();
            let block = p / BLOCK + block;
            g.protected.contains(block) || g.clean.contains(block) || g.absent.contains(block)
        };
        assert!(guard(p, p + 3 * BLOCK).is_some());
        assert!(noted(0) && noted(1) && noted(2));
        // The middle allocation.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(0) && !noted(1) && noted(2));
        title_memory_freed(p, BLOCK);
        assert!(!noted(0) && noted(2));
        // Nothing noted there: nothing to do, and nothing beside it.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(2));
        title_memory_freed(p + 2 * BLOCK, 0);
        assert!(!noted(2));
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A write between another thread's asking about a range and the second
    /// part of a submission's one asking is later than what that thread
    /// copied.
    #[test]
    fn the_rest_of_the_one_asking_has_a_number_of_its_own() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        // One range: `ask_ahead` leaves it; `ask_once` asks it with a newer number.
        let ranges = [(p, p + 4 * PAGE)];
        ask_ahead(&ranges);
        let (copied, _) = writes(p, p + 4 * PAGE, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write_volatile(1) };
        ask_once(&ranges);
        let last = journal().notes.range(p / PAGE + 1, p / PAGE + 2).first().map(|n| n.1);
        ask_once_end();
        assert!(last.is_some_and(|last| last > copied), "{last:?} is not after {copied}");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn ahead_covers_binary_searches_coalesced_ranges() {
        let mut ahead = vec![(20, 25), (10, 15), (14, 18), (18, 20), (30, 35)];
        sort_coalesce_ahead(&mut ahead);
        assert_eq!(ahead, [(10, 25), (30, 35)]);
        assert!(ahead_covers(&ahead, 10, 11));
        assert!(ahead_covers(&ahead, 10, 25));
        assert!(ahead_covers(&ahead, 15, 22));
        assert!(ahead_covers(&ahead, 30, 35));
        assert!(!ahead_covers(&ahead, 9, 11));
        assert!(!ahead_covers(&ahead, 24, 26));
        assert!(!ahead_covers(&ahead, 25, 30));
        assert!(!ahead_covers(&ahead, 34, 36));
        assert!(!ahead_covers(&[], 0, 1));
    }

    #[test]
    fn asking_many_does_not_block_disjoint_ranges() {
        // Hold two disjoint ranges and verify a third disjoint range (e.g. a
        // descriptor heap on `victory-heap-sync`) can be held immediately.
        let held = AskingMany::hold(vec![(100, 120), (200, 220)]);
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _disjoint = Asking::hold(300, 310);
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(2))
            .expect("disjoint Asking::hold must not block behind AskingMany");
        t.join().unwrap();
        drop(held);
    }

    /// The runtime maps a view and then tells us; a copy's refresh on another
    /// thread can guard it in between. Told, it is the title's to write: a
    /// block left read-only and forgotten would be an access violation in
    /// title code.
    #[test]
    fn a_block_guarded_before_we_were_told_of_its_view_can_be_written() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write; the
        // view holds it.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK);
            CloseHandle(section);
            view
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        // The refresh, then the runtime's word.
        assert!(guard(at, at + 2 * BLOCK).is_some());
        assert!(!writable(at) && !writable(at + BLOCK));
        title_memory_remapped(at, 2 * BLOCK);
        assert!(writable(at) && writable(at + BLOCK));
        // SAFETY: inside the view; a fault would end the process.
        unsafe { ((at + 8) as *mut u8).write_volatile(1) };
        // Looked at again as new, and guarded again.
        assert_eq!(
            guard(at, at + 2 * BLOCK),
            Some((at / PAGE..at / PAGE + 2).collect())
        );
        assert!(!writable(at));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_unmapping(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A write opens the view it goes through and no other: the others stay
    /// read-only, are told as written, and take writes of their own.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(writable(two) && !writable(one));
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view that was left as it was, after the other.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]), "not clean, though never opened");
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // Of a page that was told at one view, a kernel write to the other
        // finds it open.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block the title has just written is not made read-only, and is told
    /// as written at each asking, at every view of it, until it has rested.
    #[test]
    fn a_block_just_written_rests() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        let rests = |rests: bool| REST_IN_TEST.set(std::time::Duration::from_secs(if rests { 3600 } else { 0 }));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        rests(true);
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()), "nothing written: as it was");
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(writable(two), "no fault for the next write");
            assert!(writes(two, two + BLOCK, true).is_some() && !published(two, two + BLOCK) && !published(one, one + BLOCK));
            unsafe { ((two + 8) as *mut u8).write(1) };
        }
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What is put in beside a guard is told at every view, once, and the
    /// block stays read-only; until it has had its share.
    #[test]
    fn what_the_title_puts_in_beside_a_guard_is_told() {
        let (section, one, two) = two_views(BLOCK);
        crate::title_memory_section(section, BLOCK as u64, true);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        assert!(writes(one, one + BLOCK, true).is_some() && writes(two, two + BLOCK, true).is_some());
        put_count(one / BLOCK).store(0, std::sync::atomic::Ordering::Relaxed);
        let put = |byte: u8| loop {
            // SAFETY: a byte of the section's only page.
            let done = put_beside(one / BLOCK, 1, |delta| unsafe { ((one + 100).wrapping_add(delta) as *mut u8).write(byte) });
            // (Other tests hold changes of mapping.)
            if done != Beside::Cannot {
                return done;
            }
        };
        for round in 1..4 {
            assert_eq!(put(round), Beside::Done);
            // SAFETY: the view the title reads it by.
            assert_eq!(unsafe { ((two + 100) as *const u8).read_volatile() }, round);
            assert!(!writable(one) && !writable(two) && !published(one, one + BLOCK) && !published(two, two + BLOCK));
            for view in [two, one] {
                assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
                assert_eq!(guard(view, view + BLOCK), Some(Vec::new()));
            }
        }
        for _ in 0..crate::titlewrite::MOST {
            assert_eq!(put(9), Beside::Done);
        }
        assert_eq!(put(9), Beside::Full);
        // A refresh between gives it its share again.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(put(9), Beside::Done);
        // Opened by a fault, it is told by that.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        // SAFETY: a view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(put_beside(one / BLOCK, 1, |_| panic!("it is not guarded")), Beside::Cannot);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
        crate::title_memory_section_gone(section);
    }

    /// Stores that fault, and copies, as the title makes them: what is there
    /// after, the block's guard and what the next asking says.
    #[test]
    fn stores_and_copies_of_the_title_leave_the_guard() {
        use std::sync::atomic::{AtomicU32, Ordering};
        unsafe extern "system-unwind" fn real(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
            // SAFETY: by the caller.
            unsafe { std::ptr::copy(from, to, n) };
            to
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::title_memory_section(section, SIZE as u64, true);
        let pages = |view: usize| vec![view / PAGE, view / PAGE + 1];
        assert_eq!(guard(one, one + SIZE), Some(pages(one)));
        assert_eq!(guard(two, two + SIZE), Some(pages(two)));
        let copy = crate::titlewrite::import("memcpy", real as *const () as usize);
        crate::titlewrite::ON_IN_TEST.set(true);
        let copy = if copy == real as *const () as usize {
            crate::titlewrite::copy_for_test(real as *const () as usize)
        } else {
            copy
        };
        // SAFETY: of that signature.
        let copy: unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8 = unsafe { std::mem::transmute(copy) };
        // No change of mapping meanwhile but the other tests' moments of it:
        // a store that meets one opens the block, as before.
        let mut kept = 0;
        let quiet = || {
            while title_memory_reading_now().is_none() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        for round in 1..200u32 {
            quiet();
            // SAFETY (all): words of the view.
            unsafe { ((one + 64) as *mut u32).write_volatile(round) };
            let was = unsafe { AtomicU32::from_ptr((two + 64) as *mut u32) }.fetch_sub(1, Ordering::SeqCst);
            assert_eq!((was, unsafe { ((one + 64) as *const u32).read_volatile() }), (round, round - 1));
            let bytes = vec![round as u8; 3000 + round as usize];
            // Over the end of the first block.
            assert_eq!(unsafe { copy((one + BLOCK - 1000) as *mut u8, bytes.as_ptr(), bytes.len()) } as usize, one + BLOCK - 1000);
            let there = unsafe { std::slice::from_raw_parts((two + BLOCK - 1000) as *const u8, bytes.len()) };
            assert!(there == bytes && unsafe { ((two + BLOCK + 2000 + round as usize) as *const u8).read_volatile() } != round as u8);
            kept += usize::from(!writable(one) && !writable(two));
            for view in [one, two] {
                assert_eq!(guard(view, view + SIZE), Some(pages(view)), "round {round}");
                assert_eq!(guard(view, view + SIZE), Some(Vec::new()), "round {round}");
            }
        }
        assert!(kept > 10, "the guard stayed {kept} times: {:?}", crate::titlewrite::DONE);
        // What is there already is not written: nothing to tell.
        for k in 0..10 {
            // SAFETY (all): as above.
            unsafe { ((one + 64) as *mut u32).write_volatile(198) };
            let same = unsafe { std::slice::from_raw_parts((one + BLOCK - 900) as *const u8, 1500) }.to_vec();
            unsafe { copy((one + BLOCK - 900) as *mut u8, same.as_ptr(), same.len()) };
            assert!(!writable(one) && !writable(two), "{k}");
        }
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        // Many stores to a block: it is opened.
        quiet();
        for k in 0..20 {
            unsafe { ((one + 128 + k * 4) as *mut u32).write_volatile(7) };
        }
        assert!(writable(one));
        crate::titlewrite::ON_IN_TEST.set(false);
        drop_views(section, &[one, two], SIZE);
        crate::title_memory_section_gone(section);
    }

    /// Of a page taken before, what is taken again is the kernel's pages
    /// written since; where they are not known, all of it.
    #[test]
    fn what_is_taken_again_is_what_was_written() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let write = |at: usize| {
            // SAFETY: committed pages of ours.
            unsafe { ((p + at) as *mut u8).write_volatile(1) }
        };
        let tracked = track(p, 4 * PAGE, false).expect("watched");
        // (Other tests' memory was here.)
        (0..4).for_each(|page| write(page * PAGE + 5));
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)], "never taken");
        assert!(tracked.take(p, p + 4 * PAGE).is_empty());
        write(PAGE + 5000);
        write(PAGE + 8192);
        write(PAGE + 65535);
        write(2 * PAGE);
        assert_eq!(
            tracked.take(p, p + 4 * PAGE),
            [(p + PAGE + 4096, p + PAGE + 3 * 4096), (p + 2 * PAGE - 4096, p + 2 * PAGE + 4096)]
        );
        write(2 * PAGE + 70);
        // Two askings' writes are in what is taken after both.
        assert!(writes(p, p + 4 * PAGE, false).is_some());
        write(2 * PAGE + 9000);
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p + 2 * PAGE, p + 2 * PAGE + 4096), (p + 2 * PAGE + 8192, p + 2 * PAGE + 12288)]);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The frames pages were last landed in.
    #[test]
    fn landed_pages_are_known_by_their_frame() {
        static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);
        let landed = &LANDED;
        for page in [0, 5, Landed::PAGES - 1, Landed::PAGES, 7 * Landed::PAGES + 3] {
            assert!(!landed.since(page, 0, 100));
            landed.note(page, 100);
            assert!(landed.since(page, 100, 100) && landed.since(page, 0, 100) && landed.since(page, 90, 110));
            assert!(!landed.since(page, 101, 110));
            landed.note(page, 0x1_0000_0005);
            assert!(landed.since(page, 0x1_0000_0000, 0x1_0000_0007) && !landed.since(page, 0x1_0000_0006, 0x1_0000_0007));
            assert!(!landed.since(page + 1, 0, 0x1_0000_0007));
        }
        landed.note(usize::MAX / 4096, 1);
        assert!(!landed.since(usize::MAX / 4096, 0, 1));
    }

    /// The set of blocks against another of the same.
    #[test]
    fn blocks_are_a_set() {
        let (set, mut other) = (Blocks::new(), BTreeSet::new());
        let mut seed = 1u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for round in 0..40_000 {
            // About the ends of pieces and words.
            let block = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
            if next(3) == 0 {
                assert_eq!(set.remove(block), other.remove(&block));
            } else {
                assert_eq!(set.insert(block), other.insert(block));
            }
            assert_eq!(set.contains(block), other.contains(&block));
            if round % 64 == 0 {
                let a = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
                let b = a + [next(10), next(300), 2 * Blocks::IN_PIECE + next(100)][next(3)];
                assert!(set.range(a..b).eq(other.range(a..b).copied()), "{a}..{b}");
                assert!(set.range(a..).eq(other.range(a..).copied()) && set.range(..).eq(other.iter().copied()));
                assert!(set.range(a..=b).eq(other.range(a..=b).copied()));
                assert_eq!(set.len(), other.len());
            }
        }
        assert_eq!(set.range(5..usize::MAX).next(), other.range(5..).next().copied());
    }

    /// What is known of a block by its bit is so at every view of its page:
    /// the write through one takes the bits of all.
    #[test]
    fn bits_go_at_every_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let ask = |at: usize| writes(at, at + SIZE, true).expect("a view");
        for at in [one, two, one, two] {
            ask(at);
        }
        assert!(published(one, one + SIZE) && published(two, two + SIZE));
        let (before, _) = ask(two);
        // SAFETY: the first view, opened by the handler.
        unsafe { ((one + BLOCK + 8) as *mut u8).write(1) };
        assert!(published(one, one + BLOCK) && published(two, two + BLOCK), "the other block is as it was");
        assert!(!published(one + BLOCK, one + SIZE) && !published(two + BLOCK, two + SIZE));
        let told = |at: usize| ask(at).1.iter().find(|w| w.0 == (at + BLOCK) / PAGE).map_or(0, |w| w.1);
        assert!(told(two) > before && told(one) > before);
        drop_views(section, &[one, two], SIZE);
    }

    /// While the kernel writes through one view, the same page at another
    /// is told as written at every asking, and the first is not guarded.
    #[test]
    fn what_the_kernel_writes_through_is_not_guarded_for_another_view() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        let writing = kernel_writing(two, 100);
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]), "the kernel may be writing it");
            assert!(writable(two));
        }
        assert!(!written_since_taken(one, 8), "by whom is not known");
        drop(writing);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // SAFETY: opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What shaders wrote, put in past the guards and told at one view, is
    /// told at the others, whose copies have not got it; and so is what we
    /// wrote while a submission's one asking stands.
    #[test]
    fn what_is_put_in_beside_the_guards_is_told_at_every_view() {
        let (section, one, two) = two_views(BLOCK);
        writes(two, two + BLOCK, true).expect("a view");
        let (before, _) = writes(two, two + BLOCK, true).expect("a view");
        assert!(published(two, two + BLOCK));
        landed_beside(&[one / 4096 + 3], &[(one + 3 * 4096, 16)]);
        let (_, told) = writes(two, two + BLOCK, true).expect("a view");
        assert!(told.iter().any(|w| w.0 == two / PAGE && w.1 > before), "{told:?} after {before}");
        assert!(landed_lately(two + 3 * 4096, 16, 1) && !landed_lately(two + 4 * 4096, 16, 1));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A view is cut in two by the runtime: until that is done, its pages
    /// count as written wherever else they are shown.
    #[test]
    fn a_view_being_split_leaves_its_pages_unguarded_elsewhere() {
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages: Vec<usize> = (0..3).map(|k| one / PAGE + k).collect();
        assert_eq!(guard(one, one + SIZE), Some(pages.clone()));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        let changing = title_memory_remapping();
        title_memory_splitting(two, SIZE);
        title_memory_unmapping(two, SIZE);
        for _ in 0..2 {
            assert_eq!(guard(one, one + SIZE), Some(pages.clone()), "written through the part that stays");
        }
        // The first and last page stay mapped.
        for k in [0, 2] {
            title_memory_remapped(two + k * BLOCK, BLOCK);
            crate::inplace::title_memory_view(two + k * BLOCK, BLOCK, section, (k * BLOCK) as u64);
        }
        drop(changing);
        assert_eq!(guard(one, one + SIZE), Some(pages));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert!(!writable(two) && writable(two + BLOCK) && !writable(two + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod many_askers_tests {
    use super::{PAGE, writes};
    use windows_sys::Win32::System::Memory::*;

    /// Threads that write pages and ask about overlapping ranges at the same
    /// time: whoever wrote a page before asking is told of that write (by
    /// its own asking or by one that took the kernel's answer before it).
    #[test]
    fn no_write_is_lost_between_askers() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const THREADS: usize = 8;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        std::thread::scope(|s| {
            for t in 0..THREADS {
                s.spawn(move || {
                    let mut last = [0u64; PAGES];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..4000 {
                        let page = next(PAGES);
                        let from = page - next(page.min(12) + 1);
                        let to = page + 1 + next((PAGES - page).min(12));
                        // SAFETY: committed memory of ours; what the racing
                        // writers leave in it does not matter.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// The same of view memory, which is guarded: what its bits answer
    /// misses no write either, however often blocks are opened and guarded
    /// again between.
    #[test]
    fn no_write_to_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(0, 0);
    }

    /// The same with renderer port #6 ([`super::port_guard`]): faults and
    /// refreshes beside one another under the shared lock, each block's state
    /// changed by the holder of its claim.
    #[test]
    fn no_write_to_views_is_lost_between_askers_with_port_6() {
        writes_to_views_are_not_lost::<1>(0, 1);
    }

    /// Pages at three addresses, a view coming and going, with port #6; and
    /// with half the threads under the shared lock and half under the lock
    /// alone (the switch changes in an A/B run).
    #[test]
    fn no_write_through_another_view_is_lost_with_port_6() {
        writes_to_views_are_not_lost::<3>(0, 1);
        writes_to_views_are_not_lost::<3>(0, 2);
    }

    /// Resting blocks with port #6, hot ones resting longer.
    #[test]
    fn no_write_to_resting_views_is_lost_with_port_6() {
        writes_to_views_are_not_lost::<3>(50, 1);
        writes_to_views_are_not_lost::<1>(50, 2);
    }

    /// The same with blocks that rest after a write ([`super::rest`]).
    #[test]
    fn no_write_to_resting_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(50, 0);
        writes_to_views_are_not_lost::<3>(50, 0);
    }

    /// The same of pages shown at three addresses, each of which has copies
    /// of its own: a write through any of them, or put past the guards, is
    /// told at all. A fourth view comes and goes meanwhile, and is written
    /// too. (The first assertion fails within a few rounds if a guard holds
    /// for one address only, `VICTORY_VIEW_GUARD=0`.)
    #[test]
    fn no_write_through_another_view_is_lost_between_askers() {
        writes_to_views_are_not_lost::<3>(0, 0);
    }

    /// `port`: 0, renderer port #6 off; 1, on in all threads; 2, on in every
    /// other thread.
    fn writes_to_views_are_not_lost<const VIEWS: usize>(rest: u64, port: u8) {
        let rest = std::time::Duration::from_micros(rest);
        const PAGES: usize = 12;
        const THREADS: usize = 8;
        // SAFETY: views of a section of our own.
        let (section, all): (usize, [usize; VIEWS]) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (PAGES * PAGE) as u32,
                std::ptr::null(),
            );
            crate::title_memory_section(section as usize, (PAGES * PAGE) as u64, true);
            let all = std::array::from_fn(|_| {
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize;
                assert!(at != 0 && at.is_multiple_of(PAGE));
                super::title_memory_remapped(at, PAGES * PAGE);
                crate::title_memory_view(at, PAGES * PAGE, section as usize, 0);
                at
            });
            (section as usize, all)
        };
        let by_bits = std::sync::atomic::AtomicUsize::new(0);
        let port6 = |what: usize| super::PORT6[what].load(std::sync::atomic::Ordering::Relaxed);
        let port6_before = [super::P6_FAULTS, super::P6_GUARDS].map(port6);
        let past = std::sync::atomic::AtomicUsize::new(0);
        let title_beside = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            if VIEWS > 1 {
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    super::PORT_GUARD_IN_TEST.set(port == 1);
                    let mut last = [0u64; VIEWS];
                    for round in 0..300usize {
                        let (page, asked) = (round * 5 % PAGES, round % VIEWS);
                        // SAFETY (all): a view of one page of the section,
                        // this thread's alone.
                        let one = unsafe {
                            MapViewOfFile(section as _, FILE_MAP_WRITE, 0, (page * PAGE) as u32, PAGE).Value as usize
                        };
                        assert_ne!(one, 0);
                        super::title_memory_remapped(one, PAGE);
                        crate::title_memory_view(one, PAGE, section, (page * PAGE) as u64);
                        unsafe { ((one + THREADS * 8) as *mut u64).write_volatile(round as u64) };
                        let (now, written) =
                            writes(all[asked] + page * PAGE, all[asked] + (page + 1) * PAGE, true).expect("a view");
                        let number = written.first().map_or(0, |w| w.1);
                        assert!(number > last[asked], "page {page}, through a new view: {number} is not after {}", last[asked]);
                        last[asked] = now.max(last[asked]);
                        {
                            let _changing = super::title_memory_remapping();
                            super::title_memory_unmapping(one, PAGE);
                            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: one as *mut _ }) };
                        }
                        last = [last[asked]; VIEWS];
                    }
                });
            }
            for t in 0..THREADS {
                let (by_bits, past, title_beside) = (&by_bits, &past, &title_beside);
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    super::PORT_GUARD_IN_TEST.set(port == 1 || port == 2 && t % 2 == 1);
                    super::HOT_IN_TEST.set(Some(if rest.is_zero() { 0 } else { 20 }));
                    let mut last = [[0u64; PAGES]; VIEWS];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..500 {
                        let page = next(PAGES);
                        let from = page - next(page.min(3) + 1);
                        let to = page + 1 + next((PAGES - page).min(3));
                        // Written through one view, asked about at another.
                        let (through, asked) = (all[next(VIEWS)], next(VIEWS));
                        #[allow(unused_assignments)]
                        let mut how = 0;
                        let (at, last) = (all[asked], &mut last[asked]);
                        if next(4) != 0 {
                            // As in the test above, or after a word of ours,
                            // or past the guard.
                            let word = through + page * PAGE + t * 8;
                            // SAFETY (both): as above.
                            // (Waited for: other tests hold changes of mapping.)
                            // (Some as a copy of the title's, or a store of it, does.)
                            if next(3) == 0
                                && super::put_beside(word / PAGE, 1, |delta| unsafe {
                                    (word.wrapping_add(delta) as *mut u64).write_volatile(3);
                                }) == super::Beside::Done
                            {
                                title_beside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                                let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                                assert!(number > last[page], "page {page}, beside: {number} is not after {}", last[page]);
                                for l in &mut last[from..to] {
                                    *l = now;
                                }
                                continue;
                            }
                            let reading = (next(3) == 0).then(super::title_memory_reading);
                            how = usize::from(reading.is_some());
                            let beside = reading.as_ref().and_then(|_| crate::inplace::beside(word, 8));
                            if let Some(to) = beside {
                                unsafe { (to as *mut u64).write_volatile(2) };
                                super::landed_beside(&[word / 4096], &[(word, 8)]);
                                past.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                assert_eq!(unsafe { (word as *const u64).read_volatile() }, 2, "put elsewhere");
                            } else {
                                how += 2;
                                if next(4) == 0 {
                                    super::writing(word, 8);
                                    how += 4;
                                }
                                unsafe { (word as *mut u64).write_volatile(1) };
                            }
                        } else {
                            // Nothing written: what is asked counts as before.
                            by_bits.fetch_add(
                                usize::from(super::published(at + from * PAGE, at + to * PAGE)),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            let (now, _) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                            for l in &mut last[from..to] {
                                *l = now;
                            }
                            continue;
                        }
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(
                            number > last[page],
                            "{VIEWS} views, page {page}, through {:x} ({how}), asked at {:x}: {number} is not after {}",
                            through / PAGE + page,
                            at / PAGE + page,
                            last[page]
                        );
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        if port != 0 {
            let after = [super::P6_FAULTS, super::P6_GUARDS].map(port6);
            assert!(after[0] > port6_before[0] && after[1] > port6_before[1], "port 6: no fault or guard under the shared lock");
        }
        assert!(past.into_inner() > 0, "nothing was put past the guards");
        assert!(title_beside.into_inner() > 0, "nothing of the title's was put in beside the guards");
        // (The view stays: at its address other tests' memory would be told
        // of these writes.)
    }

    /// A range of a placeholder with views in it is known whole, its holes
    /// among it, until something is mapped into one, unmapped from it or
    /// allocated there. A hole that is no placeholder's is asked about.
    #[test]
    fn holes_are_known_until_the_runtime_tells() {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 8;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section)
        };
        let map = |block: usize| unsafe {
            let to = at + block * PAGE;
            VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                to as *const _,
                (block * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, to);
            super::title_memory_remapped(to, PAGE);
        };
        map(1);
        map(5);
        assert!(super::placeholder(at / PAGE) && super::placeholder(at / PAGE + 7) && !super::placeholder(at / PAGE + 5));
        let end = at + BLOCKS * PAGE;
        let ask = |since: u64| -> (u64, Vec<usize>) {
            let (now, written) = writes(at, end, true).expect("a view");
            (now, written.into_iter().filter(|w| w.1 > since).map(|w| w.0 - at / PAGE).collect())
        };
        let (first, all) = ask(0);
        assert_eq!(all, (0..BLOCKS).collect::<Vec<_>>());
        assert!(super::published(at, end), "with its holes");
        // Written; mapped into; unmapped.
        unsafe { ((at + 5 * PAGE) as *mut u8).write_volatile(1) };
        assert!(!super::published(at, end));
        let (second, all) = ask(first);
        assert_eq!(all, [5]);
        map(2);
        assert!(!super::published(at, end));
        let (third, all) = ask(second);
        assert_eq!(all, [2]);
        unsafe { ((at + 2 * PAGE) as *mut u8).write_volatile(1) };
        let (fourth, all) = ask(third);
        assert_eq!(all, [2]);
        super::title_memory_unmapping(at + 5 * PAGE, PAGE);
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: (at + 5 * PAGE) as *mut _ };
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
        }
        let (fifth, all) = ask(fourth);
        assert_eq!(all, [5]);
        assert!(super::published(at, end));
        // Told of memory there, though there is none: asked about again.
        super::title_memory_allocated(at, PAGE);
        assert!(!super::published(at, end));
        let (_, all) = ask(fifth);
        assert_eq!(all, []);
        // A reservation of the ordinary kind is no placeholder.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(!super::placeholder(plain / PAGE));
        super::look_at_holes();
        assert!(!super::published(at, end), "block 0 stays asked about");
        assert!(super::published(at + PAGE, end), "the others are as they were");
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// Pages mapped into a placeholder, written and unmapped again, each by a
    /// thread of its own as the runtime does it, while others ask about all
    /// of it: what was written, and that a page came or went, is told to
    /// whoever asks after it, and blocks are known by their bits meanwhile.
    #[test]
    fn no_mapping_is_lost_between_askers() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 12;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            for block in 0..BLOCKS - 1 {
                VirtualFree((at + block * PAGE) as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section as usize)
        };
        let end = at + BLOCKS * PAGE;
        let (stop, by_bits) = (AtomicBool::new(false), AtomicUsize::new(0));
        let number = |written: &[(usize, u64)], block: usize| written.iter().find(|w| w.0 == at / PAGE + block).map_or(0, |w| w.1);
        std::thread::scope(|s| {
            let mappers: Vec<_> = (0..4usize)
                .map(|t| {
                    let number = &number;
                    s.spawn(move || {
                        for round in 0..120 {
                            let block = t * 3 + round % 3;
                            let to = at + block * PAGE;
                            let (before, _) = writes(at, end, true).expect("a view");
                            unsafe {
                                let view = MapViewOfFile3(
                                    section as _,
                                    GetCurrentProcess(),
                                    to as *const _,
                                    (block * PAGE) as u64,
                                    PAGE,
                                    MEM_REPLACE_PLACEHOLDER,
                                    PAGE_READWRITE,
                                    std::ptr::null_mut(),
                                    0,
                                );
                                assert_eq!(view.Value as usize, to);
                                super::title_memory_remapped(to, PAGE);
                                (to as *mut u8).write_volatile(round as u8);
                            }
                            let (mapped, written) = writes(at, end, true).expect("a view");
                            assert!(number(&written, block) > before, "mapped and written: not told");
                            unsafe { ((to + 9) as *mut u8).write_volatile(round as u8) };
                            let (again, written) = writes(to, to + PAGE, true).expect("a view");
                            assert!(number(&written, block) > mapped, "written again: not told");
                            super::title_memory_unmapping(to, PAGE);
                            unsafe {
                                let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: to as *mut _ };
                                assert_ne!(UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER), 0);
                            }
                            let _ = again;
                        }
                    })
                })
                .collect();
            for t in 0..3usize {
                let (stop, by_bits) = (&stop, &by_bits);
                s.spawn(move || {
                    let mut k = t;
                    while !stop.load(Ordering::Relaxed) {
                        k = (k * 5 + 3) % 64;
                        let (from, to) = (at + k % 6 * PAGE, end - k / 8 % 5 * PAGE);
                        by_bits.fetch_add(usize::from(super::published(from, to)), Ordering::Relaxed);
                        assert!(writes(from, to, true).is_some());
                    }
                });
            }
            let failed = mappers.into_iter().map(|m| m.join()).filter(Result::is_err).count();
            stop.store(true, Ordering::Relaxed);
            assert_eq!(failed, 0);
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        // All unmapped: known whole, and nothing but holes.
        writes(at, end, true).expect("a view");
        assert!(super::published(at, end));
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// A block's bit goes with everything that makes it the title's to write,
    /// or forgets it.
    #[test]
    fn bits_go_with_the_guard() {
        // SAFETY: a view of a section of our own, which stays.
        let at = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (2 * PAGE) as u32, std::ptr::null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        let end = at + 2 * PAGE;
        let takes: [&dyn Fn(); 4] = [
            &|| super::title_memory_remapped(at, PAGE),
            // SAFETY: the view.
            &|| unsafe { ((at + 8) as *mut u8).write_volatile(1) },
            &|| super::writing(at, 1),
            // (Told again, as of a view mapped there next: it is read-only.)
            &|| {
                super::title_memory_unmapping(at, PAGE);
                assert!(!super::published(at, end), "forgotten");
                super::title_memory_remapped(at, PAGE);
                let mut old = 0;
                // SAFETY: the view.
                unsafe { VirtualProtect(at as *const _, PAGE, PAGE_READWRITE, &mut old) };
            },
        ];
        for (i, take) in takes.iter().enumerate() {
            writes(at, end, true).expect("a view");
            assert!(super::published(at, end), "{i}");
            take();
            assert!(!super::published(at, end), "{i}");
            assert!(super::published(at + PAGE, end), "{i}: the other block");
        }
        super::title_memory_unmapping(at, 2 * PAGE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: super::Bits = super::Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }
}

/// Renderer port #7 ([`port_journal`]): the journal's notes in parts.
#[cfg(test)]
mod port7_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use windows_sys::Win32::System::Memory::*;

    fn rng(seed: &mut u64, n: usize) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as usize % n.max(1)
    }

    /// The parts of a range, its pieces, and what is read of it a part at a
    /// time or holding its parts, over ranges across the parts' bounds.
    #[test]
    fn parts_pieces_and_ranges_agree() {
        const P: usize = 1 << SHARD_SHIFT;
        assert_eq!(shards_of(0, 0), 0);
        assert_eq!(shards_of(0, 1), 1);
        assert_eq!(shards_of(P - 1, P + 1), 0b11);
        assert_eq!(shards_of(63 * P, 65 * P), 1 << 63 | 1);
        assert_eq!(shards_of(5, 5 + SHARDS * P), u64::MAX);
        assert_eq!(pieces(P - 2, 2 * P + 3), Some(vec![(P - 2, P), (P, 2 * P), (2 * P, 2 * P + 3)]));
        assert_eq!(pieces(0, 1 << 40), None);
        let notes = Notes::new();
        let mut want: BTreeMap<usize, u64> = BTreeMap::new();
        let mut seed = 7;
        for n in 1..4000u64 {
            let page = rng(&mut seed, 70 * P);
            let mut held = notes.hold(shards_of_pages([page]), true);
            held.note(page, WHOLE, n);
            want.insert(page, n);
        }
        for _ in 0..500 {
            let first = rng(&mut seed, 70 * P);
            let last = first + rng(&mut seed, 3 * P);
            let expected: Vec<(usize, u64)> = want.range(first..last).map(|(&p, &w)| (p, w)).collect();
            assert_eq!(notes.range(first, last), expected, "{first}..{last} a part at a time");
            assert_eq!(notes.hold(shards_of(first, last), false).range(first, last), expected, "{first}..{last} held");
        }
        // The whole space: each part once.
        assert_eq!(notes.range(0, usize::MAX / 2), want.iter().map(|(&p, &w)| (p, w)).collect::<Vec<_>>());
        assert_eq!(notes.hold(u64::MAX, false).range(0, usize::MAX / 2).len(), want.len());
    }

    /// Noted out of order (shared noters' numbers may come in any order): a
    /// page's note, and its parts', only go up.
    #[test]
    fn notes_only_go_up() {
        let mut s = Shard::new();
        assert_eq!(s.note(3, WHOLE, 9), 0);
        assert_eq!(s.note(3, 0b11, 5), 9);
        assert_eq!(s.written_at(3), Some(9));
        assert_eq!(s.parts_since(3, 8), WHOLE, "all of it at 9");
        assert_eq!(s.note(3, 0b100, 12), 9);
        assert_eq!(s.parts_since(3, 9), 0b100);
        // An older whole note leaves the newer part as it is.
        assert_eq!(s.note(3, WHOLE, 10), 12);
        assert_eq!(s.parts_since(3, 10), 0b100);
        assert_eq!(s.parts_since(3, 9), WHOLE);
        assert_eq!(s.written_at(3), Some(12));
    }

    /// Many threads note pages (holding their parts for writing, a number
    /// taken under them) and read ranges with a fresh number (holding their
    /// parts for reading), the parts taken in any mix: every reader sees
    /// every note of a lower number of its range's pages (what keeps a write
    /// from being lost), and nobody waits for ever. (A watchdog fails the
    /// test after 60 s.)
    #[test]
    fn a_fresh_number_sees_every_lower_note_and_nobody_deadlocks() {
        const P: usize = 1 << SHARD_SHIFT;
        const THREADS: usize = 8;
        const ROUNDS: usize = 3000;
        static NOTES: Notes = Notes::new();
        static NUMBER: AtomicU64 = AtomicU64::new(0);
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Per thread: what it noted (page, number) and what it read
            // (first, last, number, the notes read).
            type Read = (usize, usize, u64, Vec<(usize, u64)>);
            let logs: Vec<(Vec<(usize, u64)>, Vec<Read>)> = std::thread::scope(|s| {
                let all: Vec<_> = (0..THREADS)
                    .map(|t| {
                        s.spawn(move || {
                            let mut seed = t as u64 * 104_729 + 3;
                            let (mut noted, mut read) = (Vec::new(), Vec::new());
                            for _ in 0..ROUNDS {
                                let first = rng(&mut seed, 200 * P);
                                if rng(&mut seed, 2) == 0 {
                                    // A few pages, some far apart.
                                    let mut pages: Vec<usize> =
                                        (0..1 + rng(&mut seed, 6)).map(|_| first + rng(&mut seed, 3 * P)).collect();
                                    if rng(&mut seed, 4) == 0 {
                                        pages.push(rng(&mut seed, 200 * P));
                                    }
                                    let mut held = NOTES.hold(shards_of_pages(pages.iter().copied()), true);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    for &p in &pages {
                                        held.note(p, WHOLE, now);
                                        noted.push((p, now));
                                    }
                                } else {
                                    let last = first + 1 + rng(&mut seed, 3 * P);
                                    let held = NOTES.hold(shards_of(first, last), false);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    read.push((first, last, now, held.range(first, last)));
                                }
                            }
                            (noted, read)
                        })
                    })
                    .collect();
                all.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut by_page: std::collections::HashMap<usize, Vec<u64>> = Default::default();
            for (noted, _) in &logs {
                for &(p, n) in noted {
                    by_page.entry(p).or_default().push(n);
                }
            }
            by_page.values_mut().for_each(|v| v.sort_unstable());
            let mut checked = 0;
            for (_, read) in &logs {
                for (first, last, now, seen) in read {
                    let seen: std::collections::HashMap<usize, u64> = seen.iter().copied().collect();
                    for (&p, ns) in by_page.iter().filter(|(p, _)| (*first..*last).contains(*p)) {
                        // The highest note below the reader's number.
                        let below = ns.partition_point(|n| n < now);
                        if below > 0 {
                            let wanted = ns[below - 1];
                            let got = seen.get(&p).copied().unwrap_or(0);
                            assert!(got >= wanted, "page {p}: read {got} with number {now}, note {wanted} missed");
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 1000, "only {checked} checks");
            done.send(()).unwrap();
        });
        wait.recv_timeout(std::time::Duration::from_secs(60)).expect("the parts' holders deadlocked (or a check failed)");
    }

    /// The journal's functions themselves, with the switch on in all threads
    /// and in every other one (as in an A/B run): writers and askers of one
    /// region; a submission's one asking standing over another region, its
    /// thread writing there (told with `wrote` or found by its asking), and
    /// other threads asking about that region meanwhile. Whoever wrote a page
    /// before asking is told of that write.
    #[test]
    fn no_write_is_lost_with_port_7() {
        askers_and_a_submission(1);
        askers_and_a_submission(2);
    }

    /// `port`: 1, renderer port #7 on in all threads; 2, in every other.
    fn askers_and_a_submission(port: u8) {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const HALF: usize = PAGES / 2;
        const THREADS: usize = 6;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        let shared_before = PORT7[P7_SHARED_NOTES].load(Relaxed);
        let skipped_before = PORT7[P7_SKIPPED].load(Relaxed);
        std::thread::scope(|s| {
            // Writers and askers of the first half.
            for t in 0..THREADS {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t % 2 == 0);
                    let mut last = [0u64; HALF];
                    let mut seed = t as u64 * 7919 + 1;
                    for _ in 0..2500 {
                        let page = rng(&mut seed, HALF);
                        let from = page - rng(&mut seed, page.min(8) + 1);
                        let to = page + 1 + rng(&mut seed, (HALF - page).min(8));
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                        // Now and then a no-op of the standing asking's.
                        if rng(&mut seed, 16) == 0 {
                            wrote(at + page * PAGE, 8);
                            ask_ahead_done();
                        }
                    }
                });
            }
            // Askers of the second half that write nothing there: they take
            // the kernel's answers from the submission's thread.
            for t in 0..2 {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t == 0);
                    let mut seed = t as u64 * 31 + 5;
                    for _ in 0..3000 {
                        let page = HALF + rng(&mut seed, HALF);
                        let to = (page + 1 + rng(&mut seed, 4)).min(PAGES);
                        let _ = writes(at + page * PAGE, at + to * PAGE, false);
                    }
                });
            }
            // The submission: writes pages of the second half, asks once over
            // all of it, writes more and tells `wrote`, reads everything from
            // the standing asking, ends it.
            s.spawn(move || {
                let mut last = [0u64; HALF];
                let mut seed = 99;
                for round in 0..400 {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || round % 2 == 0);
                    let before: Vec<usize> = (0..3).map(|_| rng(&mut seed, HALF)).collect();
                    for &p in &before {
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + (HALF + p) * PAGE) as *mut u64).write_volatile(round) };
                    }
                    let ranges: Vec<(usize, usize)> =
                        (0..4).map(|k| (at + (HALF + k * 6) * PAGE, at + (HALF + k * 6 + 6) * PAGE)).collect();
                    ask_once(&ranges);
                    let told = rng(&mut seed, HALF);
                    // SAFETY: committed memory of ours.
                    unsafe { ((at + (HALF + told) * PAGE + 64) as *mut u64).write_volatile(round) };
                    wrote(at + (HALF + told) * PAGE + 64, 8);
                    let mut read: Vec<usize> = before.iter().copied().chain([told]).collect();
                    read.sort_unstable();
                    read.dedup();
                    for p in read {
                        let (now, written) =
                            writes(at + (HALF + p) * PAGE, at + (HALF + p + 1) * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + HALF + p).map_or(0, |w| w.1);
                        assert!(number > last[p], "round {round}, page {p}: {number} is not after {}", last[p]);
                        last[p] = now;
                    }
                    ask_ahead_done();
                    ask_once_end();
                }
            });
        });
        assert!(PORT7[P7_SHARED_NOTES].load(Relaxed) > shared_before, "the shared notings ran");
        assert!(PORT7[P7_SKIPPED].load(Relaxed) > skipped_before, "needless takings alone were left out");
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn cold_unmap_and_remap_fast_path_avoid_exclusive_guards_lock() {
        let cold = 0x3000_0000_0000usize;
        // Hold GUARDS shared while unmapping and remapping an unwatched range:
        // if either tried to take GUARDS exclusively, this would deadlock.
        let _shared = super::guards_shared();
        {
            let _remapping = super::title_memory_remapping();
            super::title_memory_unmapping(cold, 4 * PAGE);
            assert!(super::is_changing(cold / BLOCK));
            assert!(super::is_changing(cold / BLOCK + 3));
            assert!(!super::is_changing(cold / BLOCK + 4));
            super::title_memory_remapped(cold, 4 * PAGE);
        }
        assert!(!super::is_changing(cold / BLOCK));
    }

    #[test]
    fn batched_unguard_opens_contiguous_and_split_runs() {
        const PAGES: usize = 6;
        // SAFETY: private allocation freed before return.
        let at = unsafe { VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        let first = at / BLOCK;
        {
            let g = super::guards();
            super::protect_here(&g, first, first + PAGES);
            for b in first..first + PAGES {
                assert!(g.protected.contains(b));
                assert!(super::PROTECTED.any(b, b + 1));
            }
            // Open two contiguous runs: [0..3) and [4..6), leaving block 3 guarded.
            let blocks = [first, first + 1, first + 2, first + 4, first + 5];
            super::unguard(&g, &blocks);
            for &b in &blocks {
                assert!(!g.protected.contains(b));
                assert!(!super::PROTECTED.any(b, b + 1));
            }
            assert!(g.protected.contains(first + 3));
            super::unguard(&g, &[first + 3]);
            assert!(!g.protected.contains(first + 3));
        }
        // SAFETY: all pages are PAGE_READWRITE again.
        for page in 0..PAGES {
            unsafe { ((at + page * PAGE) as *mut u8).write_volatile(0x5a) };
        }
        // SAFETY: private allocation from above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn coalesce_overlapping_pieces_merges_overlaps_and_preserves_touching() {
        let mut pieces = vec![(1, 3), (2, 5), (4, 4), (5, 7), (7, 10), (15, 20), (16, 18)];
        super::coalesce_overlapping_pieces(&mut pieces);
        // (1,3)+(2,5)+(4,4) -> (1,5); touching (5,7) and (7,10) stay separate; (15,20)+(16,18) -> (15,20).
        assert_eq!(pieces, vec![(1, 5), (5, 7), (7, 10), (15, 20)]);
    }

    #[test]
    fn notes_piece_bitmap_all_below_matches_range() {
        const P: usize = 1 << super::SHARD_SHIFT;
        let mut notes = super::Notes::new();
        notes.note(5, super::WHOLE, 10);
        notes.note(P + 2, 0b0101, 20);
        notes.note(P + 900, super::WHOLE, 15);
        notes.note(65 * P + 7, super::WHOLE, 30); // same shard (1) as P+2
        assert!(notes.all_below(0, 5, 0, |_, _| false));
        assert!(notes.all_below(0, 10, 10, |_, _| false));
        assert!(!notes.all_below(0, 10, 9, |_, _| false));
        assert!(notes.all_below(P + 3, P + 900, 10, |_, _| false));
        assert!(notes.all_below(P, P + 1000, 20, |_, _| false));
        assert!(!notes.all_below(P, P + 1000, 19, |_, _| false));
        assert_eq!(
            notes.range(0, 66 * P),
            vec![(5, 10), (P + 2, 20), (P + 900, 15), (65 * P + 7, 30)]
        );
    }

    #[test]
    fn watch_pool_runs_all_items_exactly_once() {
        let pool = super::WatchPool::get(3);
        let hits: Vec<std::sync::atomic::AtomicUsize> = (0..64).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
        pool.each(hits.len(), &|i| {
            hits[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(std::sync::atomic::Ordering::Relaxed) == 1));
    }

    #[test]
    fn hot_block_promotes_skips_reprotect_and_reports_every_write() {
        for use_port_guard in [false, true] {
            super::PORT_GUARD_IN_TEST.set(use_port_guard);
            super::REST_IN_TEST.set(std::time::Duration::from_millis(15));
            super::HOT_IN_TEST.set(Some(120));
            // SAFETY: section and views owned by this test, cleaned up below.
            let (section, one, two) = unsafe {
                let section = CreateFileMappingW(
                    -1isize as _,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    BLOCK as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
                let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
                assert!(one != 0 && two != 0);
                super::title_memory_remapped(one, BLOCK);
                super::title_memory_remapped(two, BLOCK);
                crate::title_memory_view(one, BLOCK, section as usize, 0);
                crate::title_memory_view(two, BLOCK, section as usize, 0);
                (section, one, two)
            };
            let p1 = one / PAGE;
            let p2 = two / PAGE;
            assert_eq!(super::guard(one, one + BLOCK), Some(vec![p1]));
            assert_eq!(super::guard(two, two + BLOCK), Some(vec![p2]));
            assert!(!super::writable(one) && !super::writable(two));

            let hot_before = super::PORT6[super::P6_HOT].load(Relaxed);
            let rested_before = super::PORT6[super::P6_RESTED_HOT].load(Relaxed);

            // Fault 1 (hot=0) and Fault 2 (hot=1, probation): both rest only `rest` (15 ms).
            // Window is rest (15 ms) + HOT_WINDOW (25 ms) = 40 ms, so sleeping 22 ms expires rest
            // while staying well inside the hot-promotion window.
            for round in 0..2u8 {
                // SAFETY: view opened by vectored exception handler.
                unsafe { ((one + 8) as *mut u8).write_volatile(round + 1) };
                assert!(super::writable(one));
                std::thread::sleep(std::time::Duration::from_millis(22));
                assert_eq!(super::guard(one, one + BLOCK), Some(vec![p1]));
                assert!(!super::writable(one), "port_guard={use_port_guard} round {round} should re-protect after base rest");
                assert_eq!(super::guard(two, two + BLOCK), Some(vec![p2]));
            }

            // Fault 3 (hot=2, promoted!) -> rests 2*rest = 30 ms; wait 38 ms (< 30+25=55 ms) to re-protect, then Fault 4 -> rests 60 ms!
            // SAFETY: view opened by handler.
            unsafe { ((one + 8) as *mut u8).write_volatile(3) };
            std::thread::sleep(std::time::Duration::from_millis(38));
            assert_eq!(super::guard(one, one + BLOCK), Some(vec![p1]));
            assert!(!super::writable(one));
            // Fault 4 -> hot=3, rests 4*rest = 60 ms!
            unsafe { ((one + 8) as *mut u8).write_volatile(4) };
            assert!(super::PORT6[super::P6_HOT].load(Relaxed) > hot_before, "hot counter must increment");

            // Sleep 25 ms (> base rest 15 ms, < promoted rest 60 ms): must stay writable without VirtualProtect thrash,
            // while still reporting every write accurately on both views!
            std::thread::sleep(std::time::Duration::from_millis(25));
            let (before, _) = super::writes(two, two + BLOCK, true).expect("view");
            assert!(super::writable(one), "hot block must stay writable past base rest");
            assert!(super::PORT6[super::P6_RESTED_HOT].load(Relaxed) > rested_before, "hot rest counter must increment");
            unsafe { ((one + 16) as *mut u8).write_volatile(99) };
            let (_, told1) = super::writes(one, one + BLOCK, true).expect("view");
            let (_, told2) = super::writes(two, two + BLOCK, true).expect("view");
            assert!(told1.iter().any(|&(p, at)| p == p1 && at > before), "primary view must report write");
            assert!(told2.iter().any(|&(p, at)| p == p2 && at > before), "alias view must report write");
            assert!(super::writable(one), "still resting hot");

            // Wait past the promoted rest (60 ms total -> 45 ms more): block cools and re-protects cleanly.
            std::thread::sleep(std::time::Duration::from_millis(45));
            assert_eq!(super::guard(one, one + BLOCK), Some(vec![p1]));
            assert!(!super::writable(one) && !super::writable(two), "must re-protect once rest expires");
            assert_eq!(super::guard(one, one + BLOCK), Some(Vec::new()));

            super::title_memory_unmapping(one, BLOCK);
            super::title_memory_unmapping(two, BLOCK);
            // SAFETY: views and section created above.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: one as *mut _ });
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: two as *mut _ });
                windows_sys::Win32::Foundation::CloseHandle(section);
            }
        }
        super::REST_IN_TEST.set(std::time::Duration::ZERO);
        super::HOT_IN_TEST.set(None);
        super::PORT_GUARD_IN_TEST.set(false);
    }

    #[test]
    fn batched_alias_cover_protects_contiguous_views() {
        const BLOCKS: usize = 4;
        const SIZE: usize = BLOCKS * BLOCK;
        // SAFETY: views of a test-owned section, cleaned up below.
        let (section, one, two) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            assert!(one != 0 && two != 0);
            super::title_memory_remapped(one, SIZE);
            super::title_memory_remapped(two, SIZE);
            crate::title_memory_view(one, SIZE, section as usize, 0);
            crate::title_memory_view(two, SIZE, section as usize, 0);
            (section, one, two)
        };
        let pages: Vec<usize> = (0..BLOCKS).map(|k| one / PAGE + k).collect();
        assert_eq!(super::guard(one, one + SIZE), Some(pages));
        for k in 0..BLOCKS {
            assert!(!super::writable(one + k * BLOCK));
            assert!(!super::writable(two + k * BLOCK), "alias block {k} must be protected");
        }
        assert_eq!(super::guard(one, one + SIZE), Some(Vec::new()));
        // Write through alias block 2 -> primary view reports block 2 only.
        // SAFETY: view opened by handler.
        unsafe { ((two + 2 * BLOCK + 12) as *mut u8).write_volatile(0x42) };
        assert_eq!(super::guard(one, one + SIZE), Some(vec![one / PAGE + 2]));
        assert!(!super::writable(two + 2 * BLOCK));

        super::title_memory_unmapping(one, SIZE);
        super::title_memory_unmapping(two, SIZE);
        // SAFETY: views and section created above.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: one as *mut _ });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: two as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    #[test]
    fn group_pieces_coalesces_same_reservation_and_splits_across_reservations() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const RES_A_PAGES: usize = 320; // 20 MiB (> old 8 MiB AHEAD_GAP)
        const RES_B_PAGES: usize = 32;  // 2 MiB
        // Reserve 20 MiB with MEM_WRITE_WATCH, commit first 2 MiB and last 2 MiB (leaving a 16 MiB reserved hole in the middle!).
        // SAFETY: private MEM_WRITE_WATCH reservations freed before return.
        let res_a = unsafe {
            VirtualAlloc(std::ptr::null(), RES_A_PAGES * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(res_a != 0 && res_a.is_multiple_of(PAGE));
        unsafe {
            assert!(!VirtualAlloc(res_a as *const _, 32 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            assert!(!VirtualAlloc((res_a + (RES_A_PAGES - 32) * PAGE) as *const _, 32 * PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
        }
        let res_b = unsafe {
            VirtualAlloc(std::ptr::null(), RES_B_PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(res_b != 0 && res_b.is_multiple_of(PAGE));

        let a_base = res_a / PAGE;
        let b_base = res_b / PAGE;
        // Pieces 18 MiB apart inside res_a (> old 8 MiB gap) plus a piece in res_b.
        let mut pieces = vec![
            (a_base, a_base + 4),
            (a_base + RES_A_PAGES - 4, a_base + RES_A_PAGES),
            (b_base, b_base + 4),
        ];
        pieces.sort_unstable();
        // With an 8 MiB gap limit, pieces 19.5 MiB apart inside res_a stay split (3 groups).
        super::WATCH_GAP_IN_TEST.set(Some((8 << 20) / PAGE));
        let split_groups = super::group_pieces(&pieces, true);
        assert_eq!(split_groups.len(), 3, "expected 3 groups under 8 MiB gap, got {split_groups:?}");

        // With a >= 20 MiB gap limit, the two pieces inside res_a coalesce into ONE group while res_b stays separate.
        super::WATCH_GAP_IN_TEST.set(Some((64 << 20) / PAGE));
        let groups = super::group_pieces(&pieces, true);
        // Must coalesce the two pieces inside res_a into ONE group, and keep res_b separate!
        assert_eq!(groups.len(), 2, "expected 2 groups (1 for res_a, 1 for res_b), got {groups:?}");
        assert!(groups.iter().any(|&(_, _, f, l)| f == a_base && l == a_base + RES_A_PAGES));
        assert!(groups.iter().any(|&(_, _, f, l)| f == b_base && l == b_base + 4));

        // Synthetic touching reservations must NEVER merge into one group.
        let syn_a = 0x5000_0000usize;
        let syn_b = syn_a + 16;
        super::cache_reservation(super::Reservation { base: syn_a, end: syn_b, private: true });
        super::cache_reservation(super::Reservation { base: syn_b, end: syn_b + 16, private: true });
        let syn_groups = super::group_pieces(&[(syn_a + 14, syn_b), (syn_b, syn_b + 2)], true);
        assert_eq!(syn_groups.len(), 2, "touching separate reservations must stay split: {syn_groups:?}");
        super::invalidate_reservations(syn_a, syn_b + 16);

        // Now verify real Windows GetWriteWatch across res_a's uncommitted middle hole + res_b via ask_once!
        let (before_a, _) = super::writes(res_a, res_a + 4 * PAGE, false).expect("res_a watched");
        let (before_b, _) = super::writes(res_b, res_b + 4 * PAGE, false).expect("res_b watched");
        // SAFETY: committed pages of res_a (start + end) and res_b.
        unsafe {
            ((res_a + PAGE) as *mut u64).write_volatile(0x1111);
            ((res_a + (RES_A_PAGES - 2) * PAGE) as *mut u64).write_volatile(0x2222);
            ((res_b + 2 * PAGE) as *mut u64).write_volatile(0x3333);
        }
        super::ask_once(&[
            (res_a, res_a + 4 * PAGE),
            (res_a + (RES_A_PAGES - 4) * PAGE, res_a + RES_A_PAGES * PAGE),
            (res_b, res_b + 4 * PAGE),
        ]);
        let (_, wa1) = super::asked(res_a, res_a + 4 * PAGE).expect("res_a start covered by ask_once");
        let (_, wa2) = super::asked(res_a + (RES_A_PAGES - 4) * PAGE, res_a + RES_A_PAGES * PAGE)
            .expect("res_a end covered by ask_once");
        let (_, wb) = super::asked(res_b, res_b + 4 * PAGE).expect("res_b covered by ask_once");
        assert!(wa1.iter().any(|&(p, at)| p == a_base + 1 && at > before_a));
        assert!(wa2.iter().any(|&(p, at)| p == a_base + RES_A_PAGES - 2 && at > before_a));
        assert!(wb.iter().any(|&(p, at)| p == b_base + 2 && at > before_b));
        super::ask_once_end();
        super::WATCH_GAP_IN_TEST.set(None);

        super::title_memory_freed(res_a, RES_A_PAGES * PAGE);
        super::title_memory_freed(res_b, RES_B_PAGES * PAGE);
        // SAFETY: reservations allocated above.
        unsafe {
            VirtualFree(res_a as *mut _, 0, MEM_RELEASE);
            VirtualFree(res_b as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn drain_early_pieces_coalesces_respects_kept_and_never_loses_writes() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 3 * super::EARLY_PIECE; // 6 MiB = 3 buckets of 2 MiB
        // SAFETY: private MEM_WRITE_WATCH reservation freed before return.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));

        // Populate reservation cache and establish baseline asking numbers.
        super::ask_once(&[(at, at + PAGES * PAGE)]);
        let (baseline, _) = super::asked(at, at + PAGES * PAGE).expect("covered");
        // While ask_once is standing (`JOURNAL_KEPT == true`), early drain must yield immediately (0 asked).
        let first_bucket = (at / PAGE) / super::EARLY_PIECE;
        let last_bucket = ((at / PAGE) + PAGES - 1) / super::EARLY_PIECE;
        let mut buckets: Vec<usize> = (first_bucket..=last_bucket).collect();
        assert_eq!(super::drain_early_pieces(&mut buckets), 0, "must skip while ask_once stands");
        super::ask_once_end();

        // While Asking holds the range, try_hold must skip without blocking.
        {
            let _held = super::Asking::hold(at / PAGE, at / PAGE + PAGES);
            assert_eq!(super::drain_early_pieces(&mut buckets), 0, "must skip while range is held");
        }

        // Write pages across all 3 buckets, then drain early (both default and port_journal modes).
        for use_port_journal in [false, true] {
            super::PORT_JOURNAL_IN_TEST.set(use_port_journal);
            let (before, _) = super::writes(at, at + PAGES * PAGE, false).expect("watched");
            let targets = [1usize, super::EARLY_PIECE + 5, 2 * super::EARLY_PIECE + 9];
            for &off in &targets {
                // SAFETY: committed pages inside our allocation.
                unsafe { ((at + off * PAGE) as *mut u64).write_volatile(0xdead_beef) };
            }
            let asked = super::drain_early_pieces(&mut buckets);
            assert!(asked >= 3, "expected >= 3 piece-equivalents drained, got {asked}");

            // Submission's ask_once after early drain must still see every write with asking > before!
            super::ask_once(&[(at, at + PAGES * PAGE)]);
            let (_, seen) = super::asked(at, at + PAGES * PAGE).expect("covered");
            for &off in &targets {
                let page = at / PAGE + off;
                assert!(
                    seen.iter().any(|&(p, w)| p == page && w > before && w > baseline),
                    "port_journal={use_port_journal}: early-drained write on page {page} lost"
                );
            }
            super::ask_once_end();
        }
        super::PORT_JOURNAL_IN_TEST.set(false);

        super::title_memory_freed(at, PAGES * PAGE);
        // SAFETY: reservation allocated above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn watch_threads_and_gap_parsing_honor_defaults_and_overrides() {
        assert_eq!(super::parse_watch_threads(None, 16), 4);
        assert_eq!(super::parse_watch_threads(None, 8), 4);
        assert_eq!(super::parse_watch_threads(None, 4), 1);
        assert_eq!(super::parse_watch_threads(None, 2), 1);
        assert_eq!(super::parse_watch_threads(Some("1"), 16), 1);
        assert_eq!(super::parse_watch_threads(Some("4"), 16), 4);
        assert_eq!(super::parse_watch_threads(Some("0"), 16), 1);
        assert_eq!(super::parse_watch_threads(Some("12"), 16), 8);

        assert_eq!(super::parse_watch_gap_pages(None, None), (8 << 20) / PAGE);
        assert_eq!(super::parse_watch_gap_pages(Some("2048"), Some("8")), (2 << 20) / PAGE);
        assert_eq!(super::parse_watch_gap_pages(None, Some("2")), (2 << 20) / PAGE);
        assert_eq!(super::parse_watch_gap_pages(Some("0"), None), 0);
        assert_eq!(super::parse_watch_gap_pages(Some("999999999"), None), (1024 << 20) / PAGE);
    }

    #[test]
    fn adaptive_hot_range_suppression_engages_resets_at_present_and_never_drops_writes() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 4;
        const SIZE: usize = PAGES * PAGE;
        // SAFETY: private MEM_WRITE_WATCH reservation freed at end of test.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(base != 0 && base.is_multiple_of(PAGE));
        let key = (base / PAGE, base / PAGE + PAGES);

        super::ADAPTIVE_WATCH_IN_TEST.set(Some(true));
        super::hot_watch_invalidate(key.0, key.1);

        // Warm up / establish clean baseline (clears initial commit write-watch bits).
        let _ = super::writes(base, base + SIZE, false).expect("watched");

        // Submits 1..3 dirty: history has 1, 2, 3 bits set (< 4/8), so suppression stays OFF.
        for round in 1..=3usize {
            // SAFETY: committed page inside our test allocation.
            unsafe { ((base + 64) as *mut u64).write_volatile(round as u64) };
            let ep_before = super::range_write_epoch(base, base + SIZE);
            super::ask_once(&[(base, base + SIZE)]);
            assert!(
                super::range_write_epoch(base, base + SIZE) > ep_before,
                "dirty ask_once round {round} must bump VA-bucket write epoch"
            );
            super::ask_once_end();
            let entry = {
                let state = super::HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
                state.get(&key).copied().expect("range tracked in HOT_WATCH")
            };
            assert_eq!(entry.hot_submits.count_ones(), round as u32);
            assert!(!entry.is_hot(), "must not engage before 4/8 dirty submits");
            assert!(!entry.pending_reset);
        }

        // Submit 4 dirty: hits 4/8 (50%) -> transitions to hot!
        // SAFETY: committed page inside our test allocation.
        unsafe { ((base + 64) as *mut u64).write_volatile(4) };
        super::ask_once(&[(base, base + SIZE)]);
        super::ask_once_end();
        {
            let state = super::HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
            let entry = state.get(&key).copied().expect("tracked");
            assert!(entry.is_hot(), "4/8 dirty submits must mark range hot");
            assert!(!entry.pending_reset, "4th submit still used reset; suppression starts on 5th");
        }

        // Submit 5 dirty (page 0): now hot -> suppresses WRITE_WATCH_FLAG_RESET (`flags = 0`)!
        // SAFETY: committed page inside our test allocation.
        unsafe { ((base + 64) as *mut u64).write_volatile(5) };
        super::ask_once(&[(base, base + SIZE)]);
        let (ask5, seen5) = super::asked(base, base + SIZE).expect("covered");
        assert!(seen5.iter().any(|&(p, at)| p == base / PAGE && at == ask5));
        super::ask_once_end();
        {
            let state = super::HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
            assert!(state[&key].pending_reset, "hot submit must suppress mid-frame reset");
        }

        // Submit 6: do NOT write page 0, write page 2 AFTER submit 5!
        // With `flags = 0`, page 0 stays dirty in Windows (zero MiWriteWatchFault on rewrite)
        // AND newly written page 2 is caught immediately!
        // SAFETY: committed page 2 inside our test allocation.
        unsafe { ((base + 2 * PAGE + 128) as *mut u64).write_volatile(6) };
        super::ask_once(&[(base, base + SIZE)]);
        let (ask6, seen6) = super::asked(base, base + SIZE).expect("covered");
        assert!(
            seen6.iter().any(|&(p, at)| p == base / PAGE && at == ask6),
            "suppressed page 0 must remain reported while flags=0"
        );
        assert!(
            seen6.iter().any(|&(p, at)| p == base / PAGE + 2 && at == ask6),
            "newly written page 2 mid-frame must never be dropped"
        );
        super::ask_once_end();

        // Simulate a CPU write to page 3 AFTER the last submission and BEFORE Present!
        // SAFETY: committed page 3 inside our test allocation.
        unsafe { ((base + 3 * PAGE + 256) as *mut u64).write_volatile(7) };
        let ep_pre_present = super::range_write_epoch(base, base + SIZE);
        super::present_reset_hot_watches();
        assert!(
            super::range_write_epoch(base, base + SIZE) > ep_pre_present,
            "present_reset_hot_watches must record late writes and bump write epoch"
        );
        {
            let state = super::HOT_WATCH.lock().unwrap_or_else(|e| e.into_inner());
            assert!(!state[&key].pending_reset, "Present must clear pending_reset");
        }
        // Verify page 3's late write was recorded into JOURNAL with asking > ask6!
        super::ask_once(&[(base, base + SIZE)]);
        let (ask7, seen7) = super::asked(base, base + SIZE).expect("covered");
        assert!(
            seen7.iter().any(|&(p, at)| p == base / PAGE + 3 && at > ask6 && at < ask7),
            "late pre-Present write on page 3 must be preserved in JOURNAL across Present reset"
        );
        // Because Present reset PTE bits and no CPU write happened after Present,
        // ask7 itself saw 0 newly dirty pages!
        assert!(
            !seen7.iter().any(|&(_, at)| at == ask7),
            "no pages should be newly dirty after Present reset without a new CPU write"
        );
        super::ask_once_end();

        super::ADAPTIVE_WATCH_IN_TEST.set(None);
        super::title_memory_freed(base, SIZE);
        // SAFETY: reservation allocated above.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn write_epoch_table_isolates_buckets_and_invalidates_on_wrote_and_landed() {
        let bucket_a = 0x1000_0000_0000usize; // 64 MiB aligned
        let bucket_b = bucket_a + (64 << 20); // adjacent 64 MiB bucket
        let ep_a0 = super::range_write_epoch(bucket_a, bucket_a + 4096);
        let ep_b0 = super::range_write_epoch(bucket_b, bucket_b + 4096);

        super::wrote(bucket_a + 128, 256);
        let ep_a1 = super::range_write_epoch(bucket_a, bucket_a + 4096);
        let ep_b1 = super::range_write_epoch(bucket_b, bucket_b + 4096);
        assert!(ep_a1 > ep_a0, "wrote() must bump touched bucket epoch");
        assert_eq!(ep_b1, ep_b0, "wrote() must not dirty disjoint 64 MiB bucket");

        // `note_landed_pages` takes 4 KiB page indices.
        super::note_landed_pages(&[(bucket_b + 4096) / super::KERNEL_PAGE]);
        let ep_a2 = super::range_write_epoch(bucket_a, bucket_a + 4096);
        let ep_b2 = super::range_write_epoch(bucket_b, bucket_b + 4096);
        assert_eq!(ep_a2, ep_a1, "note_landed_pages must not dirty disjoint bucket");
        assert!(ep_b2 > ep_b1, "note_landed_pages must bump touched bucket epoch");
    }

    #[test]
    fn mode1_tripwire_catches_direct_cpu_read_restores_rw_and_disarms_one_shot() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let len = 4 * 4096;
        // SAFETY: plain committed allocation freed at end of test.
        let base = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
        assert!(!base.is_null());
        // SAFETY: inside our committed allocation.
        unsafe {
            base.add(4096 + 17).write(0x5A);
        }
        let id = 0x7190_0001u64;
        let tripped_before = super::tripwire_stats().1;
        // Wait out any concurrent test holding `MAPPING.write()` (`title_memory_remapping`).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !super::tripwire_arm(id, base as usize, len) {
            assert!(std::time::Instant::now() < deadline, "tripwire_arm timed out waiting for MAPPING");
            std::thread::yield_now();
        }
        assert!(super::tripwire_active());
        // Direct CPU read faults into VEH once, restores PAGE_READWRITE, disarms, and returns real byte.
        // SAFETY: live allocation armed with our VEH tripwire.
        let got = unsafe { base.add(4096 + 17).read_volatile() };
        assert_eq!(got, 0x5A);
        let tripped_after = super::tripwire_stats().1;
        assert_eq!(tripped_after, tripped_before + 1);
        // Second read and write incur zero additional tripwire faults (one-shot).
        // SAFETY: page was restored to PAGE_READWRITE by the single tripwire fault.
        unsafe {
            assert_eq!(base.add(4096 + 17).read_volatile(), 0x5A);
            base.add(4096 + 17).write_volatile(0xA5);
            assert_eq!(base.add(4096 + 17).read_volatile(), 0xA5);
        }
        assert_eq!(super::tripwire_stats().1, tripped_after);
        super::tripwire_disarm_id(id);
        // SAFETY: freeing our test allocation.
        unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
    }

    /// Lock-free `!PROTECTED.any_acquire()` fast-path in `kernel_writing`:
    /// even when all 64 guard shards are exclusively held, `kernel_writing`
    /// and its `Drop` guard complete immediately without blocking on non-view memory.
    #[test]
    fn kernel_writing_fast_path_is_lock_free_on_unguarded_memory() {
        let cold = 0x3100_0000_0000usize;
        let _all = super::guards();
        let writing = super::kernel_writing(cold, 4 * BLOCK);
        assert_eq!(writing.shards, 0, "unguarded range must take lock-free fast-path");
        drop(writing);
    }

    /// 64-shard `GUARDS` allows concurrent disjoint 16-MiB shard operations
    /// without blocking, and sorted ascending multi-shard locking is deadlock-free
    /// under overlapping multi-thread stress.
    #[test]
    fn sharded_guards_disjoint_concurrency_and_sorted_locking_no_deadlock() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        const SHARD_BYTES: usize = (1 << super::GUARD_SHARD_SHIFT) * BLOCK; // 16 MiB
        // SAFETY: test-owned pagefile section mapped into a 2-shard (16 MiB + 1 block) view.
        let (section, base) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (SHARD_BYTES + BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SHARD_BYTES + BLOCK).Value as usize;
            assert_ne!(base, 0);
            (section, base)
        };
        let b0 = base / BLOCK;
        let b1 = (base + SHARD_BYTES) / BLOCK;
        let s0 = super::guard_shard_of(b0);
        let s1 = super::guard_shard_of(b1);
        assert_ne!(s0, s1, "blocks 16 MiB apart must map to distinct shards");

        // Guard both blocks, then hold shard s0 exclusively while faulting on shard s1.
        assert!(super::writes(base, base + BLOCK, true).is_some());
        assert!(super::writes(base + SHARD_BYTES, base + SHARD_BYTES + BLOCK, true).is_some());
        assert!(!super::writable(base) && !super::writable(base + SHARD_BYTES));
        {
            let _hold_s0 = super::guard_shards_write(1u64 << s0);
            // SAFETY: VEH opens block b1 in shard s1 without touching locked shard s0!
            unsafe { ((base + SHARD_BYTES + 16) as *mut u8).write_volatile(0x77) };
            assert!(super::writable(base + SHARD_BYTES));
        }

        // Overlapping multi-shard sorted lock stress across 8 threads (never deadlocks).
        std::thread::scope(|scope| {
            for tid in 0..8usize {
                scope.spawn(move || {
                    for round in 0..200usize {
                        let a = (tid * 7 + round) & 63;
                        let b = (tid * 13 + round * 3 + 5) & 63;
                        let c = (tid * 19 + round * 5 + 11) & 63;
                        let mask = (1u64 << a) | (1u64 << b) | (1u64 << c);
                        if round % 3 == 0 {
                            let _w = super::guard_shards_write(mask);
                        } else {
                            let _r = super::guard_shards_read(mask);
                        }
                    }
                });
            }
        });

        super::title_memory_unmapping(base, SHARD_BYTES + BLOCK);
        // SAFETY: section and view created above.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Region-level sequential-write detection: when >= 3 adjacent 64 KiB
    /// guarded blocks fault sequentially, the ahead span unprotects as
    /// write-through for `stream_ttl()` while still reporting all writes as dirty
    /// (zero stale copies), then re-protects cleanly after TTL expires.
    #[test]
    fn sequential_write_streaming_unprotects_ahead_tracks_dirty_and_reprotects() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        const BLOCKS: usize = 10;
        const SIZE: usize = BLOCKS * BLOCK;
        super::STREAM_TTL_IN_TEST.set(Some(std::time::Duration::from_millis(45)));
        super::LAST_SEQ_FAULT.set(super::SeqFault { block: 0, streak: 0, at: None });

        // SAFETY: test-owned pagefile section & view, cleaned up below.
        let (section, base) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let base = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            assert_ne!(base, 0);
            super::title_memory_remapped(base, SIZE);
            (section, base)
        };

        assert!(super::writes(base, base + SIZE, true).is_some());
        for k in 0..BLOCKS {
            assert!(!super::writable(base + k * BLOCK), "block {k} should start protected");
        }
        let spans_before = super::STREAM_SPANS_OPENED.load(std::sync::atomic::Ordering::Relaxed);

        // Fault sequentially on blocks 0, 1, 2 (3 adjacent 64 KiB blocks).
        for k in 0..3usize {
            // SAFETY: view opened by VEH.
            unsafe { ((base + k * BLOCK + 8) as *mut u8).write_volatile((k + 1) as u8) };
        }
        assert!(
            super::STREAM_SPANS_OPENED.load(std::sync::atomic::Ordering::Relaxed) > spans_before,
            "3 sequential faults must trigger streaming write-through"
        );
        // Ahead blocks 3..10 must already be writable (zero further VEH faults).
        for k in 3..BLOCKS {
            assert!(super::writable(base + k * BLOCK), "ahead block {k} must be unprotected");
        }

        // Refreshing during the TTL keeps the streaming span writable AND reports every block dirty.
        let (before, first_seen) = super::writes(base, base + SIZE, true).expect("view");
        for k in 0..BLOCKS {
            assert!(first_seen.iter().any(|&(p, at)| p == base / PAGE + k && at == before));
            assert!(super::writable(base + k * BLOCK), "streaming block {k} stays write-through during TTL");
        }
        // Writing to an ahead block without faulting is still reported dirty on the next asking.
        // SAFETY: writable streaming block.
        unsafe { ((base + 7 * BLOCK + 12) as *mut u8).write_volatile(0xEE) };
        let (after, second_seen) = super::writes(base, base + SIZE, true).expect("view");
        assert!(second_seen.iter().any(|&(p, at)| p == base / PAGE + 7 && at > before && at == after));

        // After TTL expires, the next refresh cleanly re-protects all blocks.
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(super::writes(base, base + SIZE, true).is_some());
        for k in 0..BLOCKS {
            assert!(!super::writable(base + k * BLOCK), "block {k} must re-protect after TTL expiry");
        }

        super::STREAM_TTL_IN_TEST.set(None);
        super::title_memory_unmapping(base, SIZE);
        // SAFETY: section and view created above.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: base as *mut _ });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Cached non-private regions (`MEM_MAPPED`/`MEM_IMAGE`) are skipped by
    /// `group_pieces` without issuing failing `GetWriteWatch` calls or wiping
    /// the negative cache.
    #[test]
    fn group_pieces_skips_cached_non_private_regions() {
        let base = 0x6200_0000usize;
        super::cache_reservation(super::Reservation { base, end: base + 16, private: false });
        let groups = super::group_pieces(&[(base, base + 4), (base + 8, base + 12)], true);
        assert!(groups.is_empty(), "non-private regions must not emit GetWriteWatch groups: {groups:?}");
        super::invalidate_reservations(base, base + 16);
    }
}
