//! The title's buffer copies, done on the card at their place in the list
//! (`VICTORY_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them (where its host copy
//! has one side only, `crate::shadow::card_all`).
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. So are those of more than 8 MiB out of
//! memory shaders do not write (streaming).

use super::*;
use crate::shadow::Shadow;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPIES_IN_PLACE").is_none_or(|v| v != "0"))
}

/// The most bytes copied out of memory shaders do not write.
const MOST: usize = 8 << 20;

/// Since the last frame-rate line: copies at their place, their bytes, those
/// out of a host copy, and texture copies out of one.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether the card copies the bytes the CPU takes out of the source at
    /// submission (out of the list's upload memory), not a host copy that
    /// shaders write.
    pub(super) fn from_upload(&self) -> bool {
        self.from.is_some()
    }

    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        let source = touches(written, self.src, self.src + self.len);
        if source && self.from.is_some() {
            return Waits::Yes;
        }
        match source || touches(written, self.dst, self.dst + self.len) {
            true => Waits::ForReaders,
            false => Waits::No,
        }
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. Gives
    /// the bytes if it took them.
    pub(super) fn prepare(&self, free: bool) -> Option<Vec<u8>> {
        if !free {
            if let Some(to) = self.from {
                copy_mapped(to as *mut u8, self.src as *const u8, self.len, false);
            }
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        let written = copy_mapped_ok(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        if written {
            return Some(bytes);
        }
        // Title memory that is not all there was left alone; verify fallback writes.
        let mut there = vec![0u8; self.len];
        (read_into(self.dst as *const u8, &mut there) && there == bytes).then_some(bytes)
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Whether a step of `runs` reads one of `ranges` on the CPU.
pub(super) fn read_by(runs: &[&HostList], mut ranges: Vec<(usize, usize)>) -> bool {
    join_ranges(&mut ranges, []);
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| match step {
            &Step::Copy(_, src, len) | &Step::Fill(_, src, len, _) => touches(&ranges, src, src + len),
            Step::Texture(staging) => {
                let (at, len) = staging.memory();
                touches(&ranges, at, at + len)
            }
            &Step::Plane((at, len), _) => touches(&ranges, at, at.saturating_add(len)),
            Step::Host(_, (from, to)) => touches(&ranges, *from, *to),
            Step::Placed(p) => p.from.is_some() && touches(&ranges, p.src, p.src + p.len),
            _ => false,
        })
    })
}

/// All GPU stages that can read or write a shadow buffer (excludes rasterizer
/// RTV/DSV/resolve stages).
const SHADOW_BUFFER_SYNC: D3D12_BARRIER_SYNC = D3D12_BARRIER_SYNC(
    D3D12_BARRIER_SYNC_ALL_SHADING.0
        | D3D12_BARRIER_SYNC_INDEX_INPUT.0
        | D3D12_BARRIER_SYNC_EXECUTE_INDIRECT.0
        | D3D12_BARRIER_SYNC_COPY.0
        | D3D12_BARRIER_SYNC_PREDICATION.0,
);

/// GPU stages that can write a shadow buffer (shaders or copies).
const SHADOW_WRITE_SYNC: D3D12_BARRIER_SYNC = D3D12_BARRIER_SYNC(
    D3D12_BARRIER_SYNC_ALL_SHADING.0 | D3D12_BARRIER_SYNC_COPY.0,
);

/// Whether the host device supports D3D12 Enhanced Barriers.
pub(super) fn enhanced_barriers(host: &Host) -> bool {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    })
}

/// Builds a whole-buffer Enhanced Barrier referencing `resource` without
/// transferring COM ownership.
fn buffer_barrier(
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
) -> D3D12_BUFFER_BARRIER {
    D3D12_BUFFER_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
        // SAFETY: borrowed COM pointer; the barrier slice never releases it.
        pResource: unsafe { std::mem::transmute_copy(resource) },
        Offset: 0,
        Size: u64::MAX,
    }
}

/// Emits a batched `D3D12_BARRIER_TYPE_BUFFER` group via Enhanced Barriers.
/// Returns `false` if Enhanced Barriers or `ID3D12GraphicsCommandList7` are
/// unavailable (caller falls back to legacy barriers).
///
/// # Safety
/// An open list and live buffer resources.
unsafe fn emit_buffer_barriers(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    barriers: &[D3D12_BUFFER_BARRIER],
) -> bool {
    if !enhanced_barriers(host) {
        return false;
    }
    let Ok(list7) = list.cast::<ID3D12GraphicsCommandList7>() else {
        return false;
    };
    if !barriers.is_empty() {
        // SAFETY: the caller's; `barriers` stays live for the `Barrier` call.
        unsafe {
            list7.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_BUFFER,
                NumBarriers: barriers.len() as u32,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pBufferBarriers: barriers.as_ptr(),
                },
            }]);
        }
    }
    true
}

/// Legacy full-pipeline fallback when Enhanced Barriers are unavailable.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all_legacy(list: &ID3D12GraphicsCommandList) {
    let alias = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                pResourceAfter: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    let uav = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    };
    // SAFETY: the caller's.
    unsafe { list.ResourceBarrier(&[alias, uav]) };
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Whether the host device supports Enhanced Barriers.
    pub(crate) fn enhanced_barriers_supported(&self) -> bool {
        host().is_some_and(enhanced_barriers)
    }

    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
        let scratch = match &own {
            Some(_) => None,
            None if len <= MOST => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
            None => return false,
        };
        let self_copy = own
            .as_ref()
            .is_some_and(|from| from.buffer.as_raw() == to.buffer.as_raw());
        let between = if self_copy {
            let Some(buf) = self.between(host, len as u64) else {
                return false;
            };
            Some(buf)
        } else {
            None
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if let Some(between) = &between {
                    // SAFETY: an open list; live buffers, ranges inside them.
                    unsafe {
                        let pre = [buffer_barrier(
                            &from.buffer,
                            SHADOW_BUFFER_SYNC,
                            D3D12_BARRIER_SYNC_COPY,
                            D3D12_BARRIER_ACCESS_COMMON,
                            D3D12_BARRIER_ACCESS_COPY_SOURCE,
                        )];
                        if !emit_buffer_barriers(host, &self.list, &pre) {
                            wait_for_all_legacy(&self.list);
                        }
                        self.list.CopyBufferRegion(between, 0, &from.buffer, offset, len as u64);
                        let mid = [
                            buffer_barrier(
                                between,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_ACCESS_COPY_DEST,
                                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                            ),
                            buffer_barrier(
                                &to.buffer,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                                D3D12_BARRIER_ACCESS_COPY_DEST,
                            ),
                        ];
                        if !emit_buffer_barriers(host, &self.list, &mid) {
                            self.list.ResourceBarrier(&[transition(
                                between,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_COPY_SOURCE,
                            )]);
                            wait_for_all_legacy(&self.list);
                        }
                        self.list.CopyBufferRegion(&to.buffer, at, between, 0, len as u64);
                        let post = [buffer_barrier(
                            &to.buffer,
                            D3D12_BARRIER_SYNC_COPY,
                            SHADOW_BUFFER_SYNC,
                            D3D12_BARRIER_ACCESS_COPY_DEST,
                            D3D12_BARRIER_ACCESS_COMMON,
                        )];
                        if !emit_buffer_barriers(host, &self.list, &post) {
                            wait_for_all_legacy(&self.list);
                        }
                    }
                } else {
                    // SAFETY: an open list; live buffers, ranges inside them.
                    unsafe {
                        let pre = [
                            buffer_barrier(
                                &to.buffer,
                                SHADOW_BUFFER_SYNC,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_ACCESS_COMMON,
                                D3D12_BARRIER_ACCESS_COPY_DEST,
                            ),
                            buffer_barrier(
                                &from.buffer,
                                SHADOW_WRITE_SYNC,
                                D3D12_BARRIER_SYNC_COPY,
                                D3D12_BARRIER_ACCESS_COMMON,
                                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                            ),
                        ];
                        if !emit_buffer_barriers(host, &self.list, &pre) {
                            wait_for_all_legacy(&self.list);
                        }
                        self.list.CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64);
                        let post = [
                            buffer_barrier(
                                &to.buffer,
                                D3D12_BARRIER_SYNC_COPY,
                                SHADOW_BUFFER_SYNC,
                                D3D12_BARRIER_ACCESS_COPY_DEST,
                                D3D12_BARRIER_ACCESS_COMMON,
                            ),
                            // Prevent WAR if a subsequent shader/copy overwrites `from.buffer`.
                            buffer_barrier(
                                &from.buffer,
                                D3D12_BARRIER_SYNC_COPY,
                                SHADOW_WRITE_SYNC,
                                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                                D3D12_BARRIER_ACCESS_COMMON,
                            ),
                        ];
                        if !emit_buffer_barriers(host, &self.list, &post) {
                            wait_for_all_legacy(&self.list);
                        }
                    }
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => {
                // SAFETY: an open list; live buffers, ranges inside them.
                unsafe {
                    let pre = [buffer_barrier(
                        &to.buffer,
                        SHADOW_BUFFER_SYNC,
                        D3D12_BARRIER_SYNC_COPY,
                        D3D12_BARRIER_ACCESS_COMMON,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                    )];
                    if !emit_buffer_barriers(host, &self.list, &pre) {
                        wait_for_all_legacy(&self.list);
                    }
                    self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
                    let post = [buffer_barrier(
                        &to.buffer,
                        D3D12_BARRIER_SYNC_COPY,
                        SHADOW_BUFFER_SYNC,
                        D3D12_BARRIER_ACCESS_COPY_DEST,
                        D3D12_BARRIER_ACCESS_COMMON,
                    )];
                    if !emit_buffer_barriers(host, &self.list, &post) {
                        wait_for_all_legacy(&self.list);
                    }
                }
            }
            (None, None) => return false,
        }
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.has_placed = true;
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            from: scratch.map(|s| s.2 as usize),
        }));
        deferred.title_written();
        self.changed_something();
        true
    }

    /// For a copy of `[src, src + len)` into a texture: a buffer (readable by
    /// copies) that holds what shaders wrote there, at this place of the list.
    /// `None` where shaders do not write it.
    pub(crate) fn place_source(&self, src: usize, len: usize) -> Option<ID3D12Resource> {
        if !on() {
            return None;
        }
        let host = host()?;
        let from = shaders_own(src, len)?;
        let between = self.between(host, len as u64)?;
        from.refresh(self, src as u64, len as u64);
        STATS[3].fetch_add(1, Ordering::Relaxed);
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            let pre = [buffer_barrier(
                &from.buffer,
                SHADOW_WRITE_SYNC,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_COMMON,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
            )];
            if !emit_buffer_barriers(host, &self.list, &pre) {
                wait_for_all_legacy(&self.list);
            }
            self.list.CopyBufferRegion(&between, 0, &from.buffer, from.offset(src as u64), len as u64);
            let post = [
                buffer_barrier(
                    &between,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_COPY_DEST,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                ),
                // Prevent WAR if a subsequent shader/copy overwrites `from.buffer`.
                buffer_barrier(
                    &from.buffer,
                    D3D12_BARRIER_SYNC_COPY,
                    SHADOW_WRITE_SYNC,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_ACCESS_COMMON,
                ),
            ];
            if !emit_buffer_barriers(host, &self.list, &post) {
                self.list.ResourceBarrier(&[transition(
                    &between,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                wait_for_all_legacy(&self.list);
            }
        }
        Some(between)
    }

    /// A device-local scratch buffer for this recording alone. Uses
    /// `internal_uav_buffer` (non-bindless, skipping `MakeResident`/`mapped`
    /// stalls) and reuses idle power-of-two buffers up to 8 MiB once no list
    /// holds a reference (`references == 1`, decayed back to `COMMON`).
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        const MAX_POOLED_LEN: u64 = 8 << 20;
        const MAX_POOLED_COUNT: usize = 8;
        const MAX_POOLED_BYTES: u64 = 16 << 20;
        struct ScratchPool(Vec<(u64, ID3D12Resource)>);
        // SAFETY: D3D12 resources are free-threaded.
        unsafe impl Send for ScratchPool {}
        static POOL: Mutex<ScratchPool> = Mutex::new(ScratchPool(Vec::new()));

        let aligned = len.next_multiple_of(1 << 16).max(1 << 16);
        if aligned > MAX_POOLED_LEN {
            let buffer = host.internal_uav_buffer(aligned)?;
            name(&buffer, || "bytes between copies".into());
            self.keep(&buffer);
            return Some(buffer);
        }
        let class = aligned.checked_next_power_of_two().unwrap_or(aligned);
        {
            let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, existing)) = pool
                .0
                .iter()
                .find(|(size, buf)| *size == class && crate::host::references(buf) == 1)
            {
                let buffer = existing.clone();
                self.keep(&buffer);
                return Some(buffer);
            }
        }
        let buffer = host.internal_uav_buffer(class)?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            let mut total: u64 = pool.0.iter().map(|(s, _)| *s).sum();
            while (pool.0.len() >= MAX_POOLED_COUNT || total.saturating_add(class) > MAX_POOLED_BYTES)
                && let Some(pos) = pool.0.iter().position(|(_, buf)| crate::host::references(buf) == 1)
            {
                let (removed_size, _) = pool.0.swap_remove(pos);
                total = total.saturating_sub(removed_size);
            }
            if pool.0.len() < MAX_POOLED_COUNT && total.saturating_add(class) <= MAX_POOLED_BYTES {
                pool.0.push((class, buffer.clone()));
            }
        }
        Some(buffer)
    }
}

#[cfg(test)]
mod tests {
    /// What copies with two sides cost to make (ignored: a measurement for
    /// `VICTORY_CARD_ALL`): buffers of 64 KiB on the card and in the PC's
    /// memory.
    #[test]
    #[ignore]
    fn small_buffers_cost() {
        let Some(host) = super::host() else {
            return;
        };
        const N: usize = 1000;
        let used = || host.video_memory().map_or(0, |m| m[1] >> 20);
        let before = used();
        let t = std::time::Instant::now();
        let card: Vec<_> = (0..N).map(|_| host.uav_buffer(1 << 16)).collect();
        let card_us = t.elapsed().as_micros() as f64 / N as f64;
        let after = used();
        let t = std::time::Instant::now();
        let pc: Vec<_> = (0..N).map(|_| host.shared_buffer(1 << 16)).collect();
        let pc_us = t.elapsed().as_micros() as f64 / N as f64;
        println!(
            "{N} buffers of 64 KiB: on the card {card_us:.0} us each, {} MB of its memory; in the PC's memory {pc_us:.0} us each",
            after.saturating_sub(before)
        );
        drop((card, pc));
    }
}
