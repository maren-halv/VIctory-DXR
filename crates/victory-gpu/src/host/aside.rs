//! Work for a list that a thread of ours does while the title records on:
//! the pass over the heap's buffer views (`crate::draw`), which took 12 to 17
//! ms of one recording thread at the frame's first list that needs it, with
//! the submission of that list, and of those behind it, waiting.
//! `VICTORY_HEAP_PASS_ASIDE=0`: done where it is needed, as before.
//!
//! What it finds are refreshes of host copies, which are steps of the list
//! for its submission, and what it makes are descriptors, which the card
//! reads when the list runs: both are there when the list is closed
//! (`HostList::close` waits), and the refreshes stand at the place in the
//! list's steps where the pass was begun.

use super::*;

/// A host copy, and the title memory of it (from, to).
pub(crate) type Refresh = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// What is being found, until it is.
pub(crate) struct Coming(Mutex<Option<Vec<Refresh>>>, std::sync::Condvar);

/// Since the last frame line: microseconds lists waited at their close.
pub(crate) static WAITED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Coming {
    /// What was found.
    pub(crate) fn wait(&self) -> Vec<Refresh> {
        let began = std::time::Instant::now();
        let _waiting = crate::timeline::span("a list's close waits for the heap's views");
        let mut found = self.0.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(found) = found.take() {
                WAITED.fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
                return found;
            }
            found = self.1.wait(found).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn found(&self, what: Vec<Refresh>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(what);
        self.1.notify_all();
    }
}

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_ASIDE").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

fn resolve_worker_count(env: Option<&str>, logical_cpus: Option<usize>) -> usize {
    if let Some(n) = env.and_then(|v| v.trim().parse::<usize>().ok()) {
        return n.clamp(1, 4);
    }
    if logical_cpus.is_some_and(|n| n > 8) { 3 } else { 2 }
}

struct Pool {
    queue: Mutex<std::collections::VecDeque<Job>>,
    wake: std::sync::Condvar,
}

/// `work` begins on a `victory-aside` worker pool (2-3 threads; per-list ordering
/// is enforced by `HostList::take_coming` and per-heap ordering by `buffer_sweep`),
/// falling back to inline execution if no worker spawned.
pub(crate) fn begin(work: impl FnOnce() -> Vec<Refresh> + Send + 'static) -> std::sync::Arc<Coming> {
    static POOL: OnceLock<Option<std::sync::Arc<Pool>>> = OnceLock::new();
    let coming = std::sync::Arc::new(Coming(Mutex::new(None), std::sync::Condvar::new()));
    let told = coming.clone();
    let job: Job = Box::new(move || {
        // (However it ends, the list that waits goes on.)
        let _aside = crate::timeline::span("beside: the heap's views");
        let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_default();
        told.found(found);
    });
    let pool = POOL.get_or_init(|| {
        let count = resolve_worker_count(
            std::env::var("VICTORY_ASIDE_THREADS").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        );
        let pool = std::sync::Arc::new(Pool {
            queue: Mutex::new(std::collections::VecDeque::new()),
            wake: std::sync::Condvar::new(),
        });
        let mut spawned = 0usize;
        for idx in 0..count {
            let shared = pool.clone();
            let name = format!("victory-aside-{idx}");
            let thread_name = name.clone();
            if std::thread::Builder::new()
                .name(name)
                .spawn(move || {
                    crate::cores::keep_on_fast_cores(&thread_name);
                    loop {
                        let job = {
                            let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
                            while q.is_empty() {
                                q = shared.wake.wait(q).unwrap_or_else(|e| e.into_inner());
                            }
                            q.pop_front()
                        };
                        if let Some(job) = job {
                            job();
                        }
                    }
                })
                .is_ok()
            {
                spawned += 1;
            }
        }
        (spawned > 0).then_some(pool)
    });
    match pool {
        Some(pool) => {
            pool.queue.lock().unwrap_or_else(|e| e.into_inner()).push_back(job);
            pool.wake.notify_one();
        }
        None => job(),
    }
    coming
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_count_scales_and_honors_override() {
        assert_eq!(resolve_worker_count(None, Some(8)), 2);
        assert_eq!(resolve_worker_count(None, Some(16)), 3);
        assert_eq!(resolve_worker_count(None, None), 2);
        assert_eq!(resolve_worker_count(Some("1"), Some(16)), 1);
        assert_eq!(resolve_worker_count(Some("3"), Some(4)), 3);
        assert_eq!(resolve_worker_count(Some("0"), Some(16)), 1);
        assert_eq!(resolve_worker_count(Some("9"), Some(16)), 4);
    }

    #[test]
    fn what_is_found_is_waited_for_and_a_panic_lets_the_list_go_on() {
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let coming = begin(move || {
            let _ = wait.recv();
            Vec::new()
        });
        assert!(coming.0.lock().unwrap().is_none());
        // A second job can finish on another worker while the first is still blocked.
        let fast = begin(Vec::new);
        assert!(fast.wait().is_empty());
        go.send(()).unwrap();
        assert!(coming.wait().is_empty());
        assert!(begin(|| panic!("a panic of the work")).wait().is_empty());
        assert!(begin(Vec::new).wait().is_empty());
    }
}
