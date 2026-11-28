//! What shaders wrote goes into title memory on a few threads
//! (`VICTORY_LAND`, on by default; `=0` merges piece by piece on the thread
//! that lands flights, as before).
//!
//! The relay's thread (`crate::host::relay`) was 58-60 ms of a 104 ms frame
//! landing (run record-1): some 1,500 pieces of 64 KiB, 98 MB, fetched from
//! copies in the card's memory, and each of them searched from both ends,
//! cleared room for, read from title memory, copied, merged and compared
//! again while it was put, one after the other; a lock and a protection
//! call to open the guard of each, and the kernel's fault on the first
//! write to each 4 KiB page of memory just made writable again.
//!
//! Here a piece is merged in one pass ([`crate::safecopy::merge`]): what
//! the card holds is compared with what it held 32 bytes at a time, title
//! memory is read only where they differ, just before the dwords that
//! change are stored. The pieces of a landing are cut into jobs for a few
//! threads kept for it ([`Pool`]).
//!
//! What stays in order: pieces in the same 64 KiB of title memory are in
//! one job, in the order they were given (two copies may hold the same
//! title memory); a landing's copies in the card's memory come before its
//! other copies; a landing is done, whole, before the title is told
//! anything and before the next one starts (`Host::relay_land`).
//!
//! With `VICTORY_LAND_BESIDE` (on by default) title memory that is a view of
//! the title's physical pages is written through a view of our own of the
//! same pages ([`crate::inplace::beside`]). The guards (`crate::watch`) are
//! read-only title views: they stay as they are, no protection changes, and
//! [`crate::watch::landed_beside`] notes the blocks written afterwards, so
//! that the next refresh takes them as before.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

/// The unit pieces are cut in (of title memory), and the guards' blocks.
pub(crate) const BLOCK: usize = 64 << 10;
/// The unit title memory is there or not in.
const PAGE: usize = 4096;
/// About this many bytes to a job.
const JOB: usize = 256 << 10;

/// See the file.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_LAND").is_none_or(|v| v != "0") && crate::safecopy::on() && crate::shadow::exact_back()
    })
}

/// How many threads land, the relay's own counted (`VICTORY_LAND_THREADS`).
fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let asked = std::env::var("VICTORY_LAND_THREADS").ok().and_then(|v| v.trim().parse().ok());
        asked.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| n.get() / 4).clamp(1, 4)).clamp(1, 16)
    })
}

/// Whether views of the title's pages are written through one of ours.
fn beside_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_BESIDE").is_none_or(|v| v != "0"))
}

/// Whether pieces an earlier fetch holds as they are now are left out
/// ([`Had`]; `VICTORY_LAND_SKIP=1`, off by default: counted either way).
pub(crate) fn skip_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_SKIP").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line; microseconds first.
pub(crate) mod stat {
    /// Which pieces to fetch; their list recorded and sent.
    pub(crate) const GATHER: usize = 0;
    /// Looking for what shaders wrote to copies in the PC's memory, the part
    /// done while the fetch was on its way.
    pub(crate) const SCAN_BESIDE: usize = 1;
    /// The same, with nothing else to wait for.
    pub(crate) const SCAN: usize = 2;
    /// Merging what was fetched: until the last job is done, and the jobs'
    /// time on all threads.
    pub(crate) const CARD: usize = 3;
    pub(crate) const CARD_ALL: usize = 4;
    /// Merging what the scan found.
    pub(crate) const BACK: usize = 5;
    pub(crate) const BACK_ALL: usize = 6;
    /// In the jobs: waiting for the lock against the title's bytes going in.
    pub(crate) const LOCK: usize = 7;
    /// In the jobs: guards opened before a write, or noted after one.
    pub(crate) const NOTES: usize = 8;
    /// The flights' ranges done with; reading back, resets; what the title
    /// is told; the rest (the line's account).
    pub(crate) const DONE: usize = 9;
    pub(crate) const READ_BACKS: usize = 10;
    pub(crate) const TOLD: usize = 11;
    pub(crate) const REST: usize = 12;
    /// Counts: pieces merged, those with a new byte, their 4 KiB pages,
    /// those with a new byte, blocks of 32 bytes with a new byte, dwords put.
    pub(crate) const PIECES: usize = 13;
    pub(crate) const NEW_PIECES: usize = 14;
    pub(crate) const PAGES: usize = 15;
    pub(crate) const NEW_PAGES: usize = 16;
    pub(crate) const NEW_BLOCKS: usize = 17;
    pub(crate) const DWORDS: usize = 18;
    /// Pieces put through a view of ours, through the title's; pages where
    /// title memory was not there.
    pub(crate) const BESIDE: usize = 19;
    pub(crate) const DIRECT: usize = 20;
    pub(crate) const MISSING: usize = 21;
    /// Bytes asked for twice in a landing (fetched once); pieces and bytes
    /// an earlier fetch held already.
    pub(crate) const TWICE: usize = 22;
    pub(crate) const HAD: usize = 23;
    pub(crate) const HAD_BYTES: usize = 24;
    /// Jobs.
    pub(crate) const JOBS: usize = 25;
    pub(crate) const COUNT: usize = 26;
}

static STATS: [AtomicU64; stat::COUNT] = [const { AtomicU64::new(0) }; stat::COUNT];

pub(crate) fn count(which: usize, n: u64) {
    STATS[which].fetch_add(n, Ordering::Relaxed);
}

/// Adds the microseconds since `from` to `which`; now.
pub(crate) fn spent(which: usize, from: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    count(which, (now - from).as_micros() as u64);
    now
}

/// For the frame-rate line: where landing went, per frame.
pub(crate) fn log(frames: f64) {
    let t: Vec<f64> = TAKEN.iter().map(|t| t.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if t[0] > 0.0 {
        tracing::info!(
            "blocks landed in beside the guards per frame: {:.0}, {:.0} of them held by the copy they were landed through, which takes \
             them no more (VICTORY_LAND_TAKEN={})",
            t[0],
            t[1],
            u8::from(taken()),
        );
    }
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if s[stat::PIECES] == 0.0 && s[stat::REST] == 0.0 {
        return;
    }
    let ms = |i: usize| s[i] / 1000.0;
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "landing per frame (VICTORY_LAND={}, {} threads): which pieces to fetch {:.2} ms, then the fetches (the relay's line); \
         looking through copies in the PC's memory {:.2} ms beside them and {:.2} ms more; merging what was fetched {:.2} ms \
         ({:.2} ms on all threads, {:.0} jobs), merging the others {:.2} ms ({:.2} on all threads); of the threads' time \
         {:.2} ms waiting for the title's bytes going in, {:.2} ms opening or noting guards; then ranges done with {:.2} ms, \
         read-backs and lists {:.2}, the title told {:.2}, the rest {:.2}",
        u8::from(on()),
        threads(),
        ms(stat::GATHER),
        ms(stat::SCAN_BESIDE),
        ms(stat::SCAN),
        ms(stat::CARD),
        ms(stat::CARD_ALL),
        s[stat::JOBS],
        ms(stat::BACK),
        ms(stat::BACK_ALL),
        ms(stat::LOCK),
        ms(stat::NOTES),
        ms(stat::DONE),
        ms(stat::READ_BACKS),
        ms(stat::TOLD),
        ms(stat::REST),
    );
    tracing::info!(
        "landed per frame: {:.0} pieces, {:.0} with a new byte; of their {:.0} pages of 4 KiB {:.0} with a new byte ({:.1} MB), \
         {:.1} MB in blocks of 32 bytes with one, {:.2} MB put as dwords; {:.0} pieces through a view of ours \
         (VICTORY_LAND_BESIDE={}), {:.0} through the title's; {:.0} pages not there; asked for twice in a landing {:.1} MB; \
         held already by an earlier fetch {:.0} pieces {:.1} MB (VICTORY_LAND_SKIP={})",
        s[stat::PIECES],
        s[stat::NEW_PIECES],
        s[stat::PAGES],
        s[stat::NEW_PAGES],
        mb(s[stat::NEW_PAGES] * PAGE as f64),
        mb(s[stat::NEW_BLOCKS] * 32.0),
        mb(s[stat::DWORDS] * 4.0),
        s[stat::BESIDE],
        u8::from(beside_on()),
        s[stat::DIRECT],
        s[stat::MISSING],
        mb(s[stat::TWICE]),
        s[stat::HAD],
        mb(s[stat::HAD_BYTES]),
        u8::from(skip_on()),
    );
}

/// Cache-line padded wrapper to eliminate cross-core false sharing on hot atomics.
#[repr(align(64))]
struct CachePadded<T>(T);

/// One active parallel batch submitted to a [`Pool`]. Lives on the caller's
/// stack until every helper has exited it (`active_helpers == 0`).
struct Batch {
    work: Work,
    count: usize,
    next: CachePadded<std::sync::atomic::AtomicUsize>,
    active_helpers: CachePadded<std::sync::atomic::AtomicUsize>,
    panicked: std::sync::atomic::AtomicBool,
}

/// Shared bounded Tier-1 worker pool. Supports concurrent external batches
/// without a global `turn` lock and falls back to inline execution on any
/// re-entrant call from inside a Tier-1 worker so deadlock is impossible.
struct Pool {
    state: Mutex<State>,
    more: Condvar,
    idle: Condvar,
    threads: usize,
    pending: CachePadded<std::sync::atomic::AtomicUsize>,
}

struct State {
    /// Active batches in LIFO order so the most recent batch drains first.
    active: Vec<usize>,
}

#[derive(Clone, Copy)]
struct Work(*const (dyn Fn(usize) + Sync));

// SAFETY: the closure may be called from any thread (`Sync`), and
// `Pool::each` keeps it alive until no helper is inside a call.
unsafe impl Send for Work {}

thread_local! {
    /// True while executing on a Tier-1 worker thread so any re-entrant call
    /// to `each` / `each_under` / `each_view` runs inline instead of blocking
    /// on other Tier-1 workers.
    static IN_TIER1_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Pool {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn new(name: &'static str, threads: usize) -> &'static Pool {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State { active: Vec::with_capacity(8) }),
            more: Condvar::new(),
            idle: Condvar::new(),
            threads,
            pending: CachePadded(std::sync::atomic::AtomicUsize::new(0)),
        }));
        for i in 1..threads {
            // (A helper that is not there is only missed.)
            let _ = std::thread::Builder::new().name(format!("{name}-{i}")).spawn(move || {
                // Tier-1 burst workers always run at THREAD_PRIORITY_NORMAL.
                crate::cores::keep_helper_on_fast_cores(name);
                IN_TIER1_WORKER.set(true);
                pool.help()
            });
        }
        pool
    }

    fn help(&self) {
        use std::sync::atomic::Ordering;
        loop {
            let batch_ptr = {
                let mut state = self.state();
                loop {
                    // Pick the newest (LIFO) batch with unclaimed work.
                    let found = state.active.iter().rev().copied().find(|&ptr| {
                        // SAFETY: every pointer in `state.active` points to a live stack `Batch`
                        // whose caller holds `state` to remove it and waits for `active_helpers == 0`.
                        let b = unsafe { &*(ptr as *const Batch) };
                        b.next.0.load(Ordering::Relaxed) < b.count
                    });
                    if let Some(ptr) = found {
                        // SAFETY: valid while `state` is held; incrementing `active_helpers`
                        // pins the `Batch` until we decrement it below.
                        let b = unsafe { &*(ptr as *const Batch) };
                        b.active_helpers.0.fetch_add(1, Ordering::AcqRel);
                        break ptr;
                    }
                    state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            };
            // SAFETY: pinned by `active_helpers` until we decrement it below.
            let batch = unsafe { &*(batch_ptr as *const Batch) };
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                loop {
                    let i = batch.next.0.fetch_add(1, Ordering::Relaxed);
                    if i >= batch.count {
                        break;
                    }
                    let working = crate::timeline::span("a landing's job");
                    // SAFETY: pinned by `active_helpers`, so `each` has not returned.
                    unsafe { (*batch.work.0)(i) };
                    working.end();
                }
            }));
            if ran.is_err() {
                batch.panicked.store(true, Ordering::Release);
            }
            if batch.active_helpers.0.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _state = self.state();
                self.idle.notify_all();
            }
        }
    }

    /// Calls `f` with every number below `count`, on the pool's threads and
    /// this one, and returns when every call has. Re-entrant calls from inside
    /// a Tier-1 worker execute inline to guarantee zero deadlock.
    fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        use std::sync::atomic::Ordering;
        if count < 2 || self.threads < 2 || IN_TIER1_WORKER.get() {
            (0..count).for_each(f);
            return;
        }
        let _spread = crate::timeline::span("landing: spread over threads");
        // SAFETY: the lifetime is left out only until this call returns, and
        // it does not before `active_helpers == 0` and `batch` is removed from `state.active`.
        let work = Work(unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(f) });
        let batch = Batch {
            work,
            count,
            next: CachePadded(std::sync::atomic::AtomicUsize::new(0)),
            active_helpers: CachePadded(std::sync::atomic::AtomicUsize::new(0)),
            panicked: std::sync::atomic::AtomicBool::new(false),
        };
        let batch_ptr = (&raw const batch) as usize;
        {
            let mut state = self.state();
            state.active.push(batch_ptr);
            self.pending.0.fetch_add(1, Ordering::Release);
        }
        if count >= self.threads {
            self.more.notify_all();
        } else {
            (1..count).for_each(|_| self.more.notify_one());
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let i = batch.next.0.fetch_add(1, Ordering::Relaxed);
                if i >= count {
                    break;
                }
                f(i);
            }
        }));
        let mut state = self.state();
        if let Some(pos) = state.active.iter().rposition(|&p| p == batch_ptr) {
            state.active.remove(pos);
        }
        self.pending.0.fetch_sub(1, Ordering::AcqRel);
        while batch.active_helpers.0.load(Ordering::Acquire) > 0 {
            state = self.idle.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        let panicked = batch.panicked.load(Ordering::Acquire);
        drop(state);
        if let Err(panic) = ran {
            std::panic::resume_unwind(panic);
        }
        assert!(!panicked, "a job of landing panicked");
    }
}

/// Unified bounded Tier-1 worker pool (`6` `THREAD_PRIORITY_NORMAL` workers by
/// default) shared by `land`, `under`, `records`, and `views`.
fn tier1_pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new("victory-tier1", crate::cores::tier1_worker_count() + 1))
}

fn single_threaded_override(var: &'static str) -> bool {
    std::env::var(var).ok().and_then(|v| v.trim().parse::<usize>().ok()) == Some(1)
}

/// [`Pool::each`] on the shared bounded Tier-1 pool for landing.
pub(crate) fn each(count: usize, f: &(dyn Fn(usize) + Sync)) {
    if threads() < 2 {
        (0..count).for_each(f);
        return;
    }
    tier1_pool().each(count, f);
}

/// [`Pool::each`] on the shared bounded Tier-1 pool for the title's bytes
/// under flights' writes (`VICTORY_UNDER_THREADS=1`: inline on caller).
pub(crate) fn each_under(count: usize, f: &(dyn Fn(usize) + Sync)) {
    static INLINE: OnceLock<bool> = OnceLock::new();
    if *INLINE.get_or_init(|| single_threaded_override("VICTORY_UNDER_THREADS")) {
        (0..count).for_each(f);
        return;
    }
    tier1_pool().each(count, f);
}

/// Calls `f` with every number below `count` on this thread and the shared
/// bounded Tier-1 pool (`VICTORY_VIEW_THREADS=1`: inline on caller).
/// Concurrent `records` and `views` callers run simultaneously on the shared
/// multi-batch pool without serializing behind one another.
pub(crate) fn each_view(_records: bool, count: usize, f: &(dyn Fn(usize) + Sync)) {
    static INLINE: OnceLock<bool> = OnceLock::new();
    if *INLINE.get_or_init(|| single_threaded_override("VICTORY_VIEW_THREADS")) {
        (0..count).for_each(f);
        return;
    }
    tier1_pool().each(count, f);
}

/// A piece to land: the copy's number, the title address, its bytes.
pub(crate) type Piece = (u64, usize, usize);

/// `pieces` in the order they are landed: by the 64 KiB block of title
/// memory they lie in, those in one block in the order given (each is
/// within one block). Where one copy alone has pieces in a block, those
/// that overlap or touch are one piece: a landing's flights declare parts
/// of the same memory, and what overlapped came down and was merged twice.
/// Each with the number of the piece given that it begins with.
pub(crate) fn ordered(pieces: &[Piece]) -> Vec<(usize, usize, usize)> {
    // Skip pieces already covered by an active Mode-1/Mode-2 GPU-only tripwire
    // before queuing GPU readbacks (`host::relay_card_back` filters through here).
    let mut order: Vec<usize> = (0..pieces.len())
        .filter(|&i| !crate::shadow::is_gpu_only_piece_va(pieces[i].0, pieces[i].1, pieces[i].2))
        .collect();
    order.sort_by_key(|&i| pieces[i].1 / BLOCK);
    let mut out: Vec<(usize, usize, usize)> = Vec::with_capacity(order.len());
    for group in order.chunk_by(|&a, &b| pieces[a].1 / BLOCK == pieces[b].1 / BLOCK) {
        let alone = group.len() > 1 && group.iter().all(|&i| pieces[i].0 == pieces[group[0]].0);
        if !alone {
            out.extend(group.iter().map(|&i| (i, pieces[i].1, pieces[i].2)));
            continue;
        }
        let mut sorted = group.to_vec();
        sorted.sort_by_key(|&i| pieces[i].1);
        let begun = out.len();
        for i in sorted {
            let (_, at, len) = pieces[i];
            match out[begun..].last_mut() {
                Some(last) if at <= last.1 + last.2 => {
                    let end = (last.1 + last.2).max(at + len);
                    count(stat::TWICE, (last.2 + len - (end - last.1)) as u64);
                    last.2 = end - last.1;
                }
                _ => out.push((i, at, len)),
            }
        }
    }
    out
}

/// Pieces (title address, bytes; in [`ordered`]'s order) as jobs: runs of
/// about [`JOB`] bytes that end where a block ends.
pub(crate) fn jobs(pieces: &[(usize, usize)]) -> Vec<std::ops::Range<usize>> {
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    let (mut from, mut bytes) = (0, 0);
    for (i, &(at, len)) in pieces.iter().enumerate() {
        let new_block = i == 0 || pieces[i - 1].0 / BLOCK != at / BLOCK;
        if bytes >= JOB && new_block {
            out.push(from..i);
            (from, bytes) = (i, 0);
        }
        bytes += len;
    }
    if from < pieces.len() {
        out.push(from..pieces.len());
    }
    out
}

/// What earlier fetches brought, for as long as a flight they ran behind
/// has not landed: per fetch, the value the host's fence had been given
/// behind the last flight sent before it, and by copy (its number) and
/// piece the offsets it brought.
///
/// A fetch on the drawing queue runs behind everything sent before it, so
/// it brings what those flights' shaders wrote too, and the merge of it put
/// that into title memory. Their own landings flag the same pieces and
/// fetched them again, to find every byte merged. A piece that a fetch held
/// whole, behind the flight that flags it, has nothing more of that flight.
/// (Bytes left out of that merge, the title's that went up behind the
/// fetch, went up behind that flight too, and stand over what it wrote.)
#[derive(Default)]
pub(crate) struct Had(Vec<(u64, std::collections::HashMap<(u64, usize), (usize, usize)>)>);

impl Had {
    /// Flights from the one behind which `fence` is signalled on are landed
    /// next: what is of no use to them is dropped.
    pub(crate) fn landing(&mut self, fence: u64) {
        self.0.retain(|had| had.0 >= fence);
    }

    /// A fetch that ran behind every flight up to `fence` brought (copy,
    /// title address, bytes) `pieces`. They are merged before the next ask.
    pub(crate) fn fetched(&mut self, fence: u64, pieces: impl Iterator<Item = Piece>) {
        let mut all = std::collections::HashMap::new();
        for (copy, at, len) in pieces {
            // (Two parts of a block that do not touch: the longer one.)
            let had = all.entry((copy, at / BLOCK)).or_insert((at, at + len));
            if len > had.1 - had.0 {
                *had = (at, at + len);
            }
        }
        self.0.push((fence, all));
    }

    /// Whether a fetch behind the flight of `fence` brought all of the
    /// piece.
    pub(crate) fn holds(&self, fence: u64, (copy, at, len): Piece) -> bool {
        self.0.iter().any(|(behind, all)| {
            *behind >= fence && all.get(&(copy, at / BLOCK)).is_some_and(|had| had.0 <= at && at + len <= had.1)
        })
    }
}

/// Whether a piece is put through a view of ours while the runtime changes
/// a mapping too (`VICTORY_LAND_ANYTIME=0`: through the title's address, as
/// before).
///
/// While the world streams a change is under way nearly all the time (pages
/// are mapped in groups, eight threads side by side: 28 of 1,620 pieces a
/// frame went through a view of ours in run textures-4, 630 in run n03). Put
/// through the title's address a piece opens the guards of its blocks, 5,000
/// changes of protection a frame, which wait for the address-space lock the
/// mapping wants as well, and every refresh guards them again: the frame went
/// from 105 to 140 ms. The record names a run of a section only while it is
/// mapped there (it is taken out before a view goes and put in when one is
/// there), so the change is one of other memory, or one the title makes of
/// memory its shaders are writing at that moment, which it may not (its
/// fences are signalled after the landing).
fn anytime() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_ANYTIME").is_none_or(|v| v != "0"))
}

/// What a job wrote beside the guards (or into write-watched private memory),
/// to be noted ([`Notes::flush`]).
#[derive(Default)]
pub(crate) struct Notes {
    /// Pages of 4 KiB written through `beside`, by number.
    pages: Vec<usize>,
    /// Per piece, from the first such page to the last: address, bytes.
    spans: Vec<(usize, usize)>,
    /// Per piece put beside the guards, all of it: its 64 KiB block and the
    /// copy it was landed through (which the job holds until the flush).
    landed: Vec<(usize, *const crate::shadow::Shadow)>,
    /// Pages of 4 KiB written directly into write-watched `MEM_PRIVATE` memory.
    private_pages: Vec<usize>,
}

// SAFETY: the pointers are used by the job that has the notes, in `flush`.
unsafe impl Send for Notes {}

/// Whether a copy a landing went through is noted as up to date with it
/// (`VICTORY_LAND_TAKEN=0`: it is not, and the next refresh copies the 64
/// KiB again and sends them up to the card, as before).
///
/// What a landing puts into title memory the copy holds already: the card
/// wrote it there, and a copy in the card's memory was given it on the CPU's
/// side by the same merge. But the block counted as written, for that copy
/// as for every other one, so the next refresh copied the whole block, sent
/// it up and had the card hash it again: some 900 blocks a frame landed
/// beside the guards, of the 103 MB a frame that went up (run record-1).
///
/// Why it may be left out ([`crate::shadow::Shadow::landed`]): the copy held
/// title memory's bytes of the block before the landing, if it had taken it
/// after the block's last noted write and no write was to be noted (the
/// block's bit, `crate::watch::block_published`), and the landing put the
/// same bytes into both. Only where one copy alone landed in the block.
fn taken() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAND_TAKEN").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: blocks landed in beside the guards, and
/// those of them the copy was noted as up to date with.
static TAKEN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

impl Notes {
    /// At the end of a job: what it wrote beside the guards is noted.
    pub(crate) fn flush(&mut self) {
        if !self.private_pages.is_empty() {
            crate::watch::note_landed_pages(&self.private_pages);
            self.private_pages.clear();
        }
        if self.pages.is_empty() {
            return;
        }
        let noting = std::time::Instant::now();
        let (now, blocks) = crate::watch::landed_beside(&self.pages, &self.spans);
        TAKEN[0].fetch_add(blocks.len() as u64, Ordering::Relaxed);
        if taken() {
            for (block, before) in blocks {
                let mut through = self.landed.iter().filter(|l| l.0 == block).map(|l| l.1);
                let Some(copy) = through.next() else {
                    continue;
                };
                if !copy.is_null() && through.all(|other| other == copy) {
                    // SAFETY: the job holds its copies until this is done.
                    let copy = unsafe { &*copy };
                    TAKEN[1].fetch_add(u64::from(copy.landed(block, before, now)), Ordering::Relaxed);
                }
            }
        }
        self.pages.clear();
        self.spans.clear();
        self.landed.clear();
        spent(stat::NOTES, noting);
    }
}

/// Lands one piece: `parts` (offsets from and to) of the bytes at `now`,
/// which the card wrote, go into title memory from `title` on and into
/// `held` ([`crate::safecopy::merge`] says how, and what `take` is). The
/// piece lies in one 64 KiB block of title memory. Returns the bytes put.
///
/// # Safety
/// `now` and `held` are ours from 0 to the last part's end, and nothing but
/// the card writes `now` there, nothing but this call `held`.
pub(crate) unsafe fn piece(
    title: usize,
    now: *const u8,
    held: *mut u8,
    parts: &[(usize, usize)],
    take: bool,
    notes: &mut Notes,
    copy: *const crate::shadow::Shadow,
) -> u64 {
    // Mode-1/Mode-2 GPU-only promoted piece: never touch guest `PAGE_NOACCESS`
    // memory or merge back into title RAM.
    if !copy.is_null() && unsafe { (&*copy).is_va_gpu_only(title) } {
        return 0;
    }
    // Write-watched `MEM_PRIVATE` copies have neither section views nor guards,
    // so skip `beside()` lookups, the redundant `differ()` pre-pass, and guard
    // opening and merge directly into title memory in one pass.
    let private_watched = !copy.is_null() && unsafe { (&*copy).is_private_watched() };
    // SAFETY: the caller's.
    unsafe { piece_at(title, now, held, parts, take, notes, beside_on(), private_watched, copy) }
}

/// [`piece`], and whether a view of ours is looked for.
///
/// # Safety
/// [`piece`]'s.
#[allow(clippy::too_many_arguments)]
unsafe fn piece_at(
    title: usize,
    now: *const u8,
    held: *mut u8,
    parts: &[(usize, usize)],
    take: bool,
    notes: &mut Notes,
    beside: bool,
    private_watched: bool,
    copy: *const crate::shadow::Shadow,
) -> u64 {
    let Some(end) = parts.last().map(|p| p.1) else {
        return 0;
    };
    // From here until the piece is put the runtime changes no mapping: our
    // view's address stays where the title sees `title`. Not waited for: if
    // a change is under way, what the record of the views says now does
    // ([`anytime`]), or the title's own address.
    let reading = (beside && !private_watched).then(crate::watch::title_memory_reading_now).flatten();
    let beside = if !private_watched && (reading.is_some() || beside && anytime()) {
        crate::inplace::beside(title, end)
    } else {
        None
    };
    count(stat::PIECES, 1);
    count(if beside.is_some() { stat::BESIDE } else { stat::DIRECT }, 1);
    // The pages that differ, before any is written: none, and the guard
    // stays; else it is opened once for the piece.
    let mut differing = [true; BLOCK / PAGE + 1];
    let base = title / PAGE;
    if beside.is_none() && !private_watched {
        drop(reading);
        differing.fill(false);
        for_pages(title, parts, |from, n| {
            // SAFETY: the caller's.
            if unsafe { crate::safecopy::differ(now.add(from), held.add(from), n) } {
                differing[(title + from) / PAGE - base] = true;
            }
        });
        let Some(first) = differing.iter().position(|&d| d) else {
            for_pages(title, parts, |_, _| count(stat::PAGES, 1));
            return 0;
        };
        let last = differing.iter().rposition(|&d| d).unwrap_or(first);
        let (from, to) = (((base + first) * PAGE).max(title), ((base + last + 1) * PAGE).min(title + end));
        let opening = std::time::Instant::now();
        crate::watch::landing(from, to - from);
        spent(stat::NOTES, opening);
    }
    let into = beside.unwrap_or(title);
    let since = crate::watch::changes();
    let (mut pages, mut new_pages, mut blocks, mut dwords, mut missing) = (0, 0u64, 0, 0, 0);
    let mut span: Option<(usize, usize)> = None;
    for_pages(title, parts, |from, n| {
        pages += 1;
        if !differing[(title + from) / PAGE - base] {
            return;
        }
        // SAFETY: the caller's; title memory, by an address that shows it
        // (`beside`) or the title's own.
        let merge = |take| unsafe { crate::safecopy::merge(into.wrapping_add(from) as *mut u8, now.add(from), held.add(from), n, take) };
        let mut done = merge(take && beside.is_some());
        if !done.there && beside.is_none() {
            // Away only while the runtime changes a mapping beside it: the
            // rest of it when that is done. What is still not there then is
            // taken as the caller says.
            let mut put = done.dwords;
            let again = crate::watch::again_after_change(since, || {
                done = merge(false);
                put += done.dwords;
                done.there
            });
            if !again && take {
                done = merge(true);
                put += done.dwords;
            }
            done.dwords = put;
            // (What was new to the first try was new.)
            done.new = done.new.max(1);
        }
        blocks += done.new;
        dwords += done.dwords;
        missing += u64::from(!done.there);
        if done.new > 0 {
            new_pages += 1;
            if beside.is_some() {
                let page = (title + from) / PAGE;
                // (Two parts in one page.)
                if notes.pages.last() != Some(&page) {
                    notes.pages.push(page);
                }
                let s = span.get_or_insert((title + from, title + from + n));
                s.1 = title + from + n;
            } else if private_watched {
                let page = (title + from) / PAGE;
                if notes.private_pages.last() != Some(&page) {
                    notes.private_pages.push(page);
                }
            }
        }
    });
    if let Some((from, to)) = span {
        notes.spans.push((from, to - from));
        // Not through a copy that is not to be noted, if that is asked: the
        // block then has two.
        let through = if missing == 0 { copy } else { std::ptr::null() };
        notes.landed.push((title / BLOCK, through));
    }
    count(stat::NEW_PIECES, u64::from(new_pages > 0));
    count(stat::PAGES, pages);
    count(stat::NEW_PAGES, new_pages);
    count(stat::NEW_BLOCKS, blocks);
    count(stat::DWORDS, dwords);
    count(stat::MISSING, missing);
    crate::shadow::PUT[0].fetch_add(u64::from(dwords > 0), Ordering::Relaxed);
    crate::shadow::PUT[2].fetch_add(dwords * 4, Ordering::Relaxed);
    dwords * 4
}

/// `parts` of a piece that begins at `title`, cut where 4 KiB pages of title
/// memory end: offset and bytes.
fn for_pages(title: usize, parts: &[(usize, usize)], mut f: impl FnMut(usize, usize)) {
    for &(from, to) in parts {
        let mut at = from;
        while at < to {
            let n = (PAGE - (title + at) % PAGE).min(to - at);
            f(at, n);
            at += n;
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tier1_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Re-entrant nested calls across `each`, `each_under`, and `each_view`
    /// on the shared Tier-1 pool never deadlock and execute every item once.
    #[test]
    fn reentrant_and_concurrent_tier1_pool_calls_never_deadlock() {
        let outer = AtomicUsize::new(0);
        let inner_under = AtomicUsize::new(0);
        let inner_view = AtomicUsize::new(0);

        each(24, &|i| {
            outer.fetch_add(1, Ordering::Relaxed);
            each_under(8, &|j| {
                inner_under.fetch_add(i + j + 1, Ordering::Relaxed);
                each_view(i % 2 == 0, 4, &|_| {
                    inner_view.fetch_add(1, Ordering::Relaxed);
                });
            });
        });

        assert_eq!(outer.load(Ordering::Relaxed), 24);
        assert_eq!(inner_view.load(Ordering::Relaxed), 24 * 8 * 4);
        let expected_under: usize = (0..24).flat_map(|i| (0..8).map(move |j| i + j + 1)).sum();
        assert_eq!(inner_under.load(Ordering::Relaxed), expected_under);

        // Concurrent external callers across all 4 entrypoints share the single Tier-1 pool.
        let hits = [const { AtomicUsize::new(0) }; 4];
        std::thread::scope(|s| {
            s.spawn(|| each(64, &|_| { hits[0].fetch_add(1, Ordering::Relaxed); }));
            s.spawn(|| each_under(64, &|_| { hits[1].fetch_add(1, Ordering::Relaxed); }));
            s.spawn(|| each_view(true, 64, &|_| { hits[2].fetch_add(1, Ordering::Relaxed); }));
            s.spawn(|| each_view(false, 64, &|_| { hits[3].fetch_add(1, Ordering::Relaxed); }));
        });
        for (idx, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 64, "caller {idx}");
        }
    }

    #[test]
    fn tier1_workers_run_at_normal_priority() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_NORMAL};
        let bad = AtomicUsize::new(0);
        each(32, &|_| {
            if IN_TIER1_WORKER.get() {
                // SAFETY: queries only the calling worker thread's priority.
                let p = unsafe { GetThreadPriority(GetCurrentThread()) };
                if p != THREAD_PRIORITY_NORMAL {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        assert_eq!(bad.load(Ordering::Relaxed), 0);
    }
}
