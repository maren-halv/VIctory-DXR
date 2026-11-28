//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

/// Each host heap's identity, handed out in order: a range of dynamic slots
/// is given back to the heap that took it, and a heap made later at the same
/// address is not that heap ([`HostHeap::release`]).
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// A set of a heap's slots, a bit for each: no lock (a pass that went through
/// a set of 70,000 under its lock kept everyone who writes slots waiting for
/// 5 ms of a frame, timeline thin-6).
struct Slots {
    bits: Vec<std::sync::atomic::AtomicU64>,
    dirty: std::sync::atomic::AtomicBool,
}

impl Slots {
    fn new(count: usize) -> Self {
        let mut bits = Vec::new();
        bits.resize_with(count.div_ceil(64), Default::default);
        Slots { bits, dirty: std::sync::atomic::AtomicBool::new(false) }
    }

    fn any(&self) -> bool {
        self.dirty.load(std::sync::atomic::Ordering::Acquire)
    }

    fn contains(&self, slot: u32) -> bool {
        let bit = 1u64 << (slot % 64);
        self.bits
            .get(slot as usize / 64)
            .is_some_and(|bits| bits.load(std::sync::atomic::Ordering::Acquire) & bit != 0)
    }

    fn put(&self, slot: u32) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Release};
        let bit = 1u64 << (slot % 64);
        if let Some(bits) = self.bits.get(slot as usize / 64)
            && bits.load(Acquire) & bit == 0
        {
            bits.fetch_or(bit, AcqRel);
            self.dirty.store(true, Release);
        }
    }

    /// Whether it was there.
    fn remove(&self, slot: u32) -> bool {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let bit = 1 << (slot % 64);
        self.bits.get(slot as usize / 64).is_some_and(|bits| {
            bits.load(Acquire) & bit != 0 && bits.fetch_and(!bit, AcqRel) & bit != 0
        })
    }

    fn of(n: usize, mut bits: u64, out: &mut Vec<u32>, which: &impl Fn(u32) -> bool) {
        while bits != 0 {
            let slot = n as u32 * 64 + bits.trailing_zeros();
            bits &= bits - 1;
            if which(slot) {
                out.push(slot);
            }
        }
    }

    /// Those that `which` says, in order.
    fn all(&self, which: impl Fn(u32) -> bool) -> Vec<u32> {
        let mut out = Vec::new();
        for (n, bits) in self.bits.iter().enumerate() {
            Self::of(n, bits.load(std::sync::atomic::Ordering::Acquire), &mut out, &which);
        }
        out
    }

    /// All of them, and none is left.
    fn take(&self) -> Vec<u32> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        if !self.dirty.swap(false, AcqRel) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for (n, bits) in self.bits.iter().enumerate() {
            if bits.load(Acquire) != 0 {
                Self::of(n, bits.swap(0, AcqRel), &mut out, &|_| true);
            }
        }
        out
    }
}

/// Two-level demand-paged chunk shift (`4096` slots per chunk) for sparse
/// `HostHeap` sidecars so untouched title/dynamic ranges allocate zero RAM (`ARCH-05`).
const SPARSE_SHIFT: usize = 12;
const SPARSE_CHUNK: usize = 1 << SPARSE_SHIFT;
const SPARSE_MASK: usize = SPARSE_CHUNK - 1;

struct SparseU64 {
    len: usize,
    chunks: Box<[std::sync::OnceLock<Box<[AtomicU64; SPARSE_CHUNK]>>]>,
}

impl SparseU64 {
    fn new(len: usize) -> Self {
        let count = len.div_ceil(SPARSE_CHUNK);
        let chunks = (0..count).map(|_| std::sync::OnceLock::new()).collect();
        Self { len, chunks }
    }

    #[inline]
    fn get(&self, idx: usize, order: Ordering) -> Option<u64> {
        if idx >= self.len {
            return None;
        }
        Some(
            self.chunks[idx >> SPARSE_SHIFT]
                .get()
                .map_or(0, |c| c[idx & SPARSE_MASK].load(order)),
        )
    }

    #[inline]
    fn set(&self, idx: usize, val: u64, order: Ordering) {
        if idx >= self.len {
            return;
        }
        let slot = &self.chunks[idx >> SPARSE_SHIFT];
        if val == 0 {
            if let Some(c) = slot.get() {
                c[idx & SPARSE_MASK].store(0, order);
            }
            return;
        }
        let c = slot.get_or_init(|| Box::new([const { AtomicU64::new(0) }; SPARSE_CHUNK]));
        c[idx & SPARSE_MASK].store(val, order);
    }

    #[cfg(test)]
    fn allocated_chunks(&self) -> usize {
        self.chunks.iter().filter(|c| c.get().is_some()).count()
    }
}

struct SparseU32 {
    len: usize,
    chunks: Box<[std::sync::OnceLock<Box<[std::sync::atomic::AtomicU32; SPARSE_CHUNK]>>]>,
}

impl SparseU32 {
    fn new(len: usize) -> Self {
        let count = len.div_ceil(SPARSE_CHUNK);
        let chunks = (0..count).map(|_| std::sync::OnceLock::new()).collect();
        Self { len, chunks }
    }

    #[inline]
    fn get(&self, idx: usize) -> u32 {
        if idx >= self.len {
            return 0;
        }
        self.chunks[idx >> SPARSE_SHIFT]
            .get()
            .map_or(0, |c| c[idx & SPARSE_MASK].load(Ordering::Acquire))
    }

    #[inline]
    fn set(&self, idx: usize, val: u32) {
        if idx >= self.len {
            return;
        }
        let slot = &self.chunks[idx >> SPARSE_SHIFT];
        if val == 0 {
            if let Some(c) = slot.get() {
                c[idx & SPARSE_MASK].store(0, Ordering::Release);
            }
            return;
        }
        let c = slot.get_or_init(|| Box::new([const { std::sync::atomic::AtomicU32::new(0) }; SPARSE_CHUNK]));
        c[idx & SPARSE_MASK].store(val, Ordering::Release);
    }
}

struct SparseU8 {
    len: usize,
    default: u8,
    chunks: Box<[std::sync::OnceLock<Box<[std::sync::atomic::AtomicU8; SPARSE_CHUNK]>>]>,
}

impl SparseU8 {
    fn new(len: usize, default: u8) -> Self {
        let count = len.div_ceil(SPARSE_CHUNK);
        let chunks = (0..count).map(|_| std::sync::OnceLock::new()).collect();
        Self { len, default, chunks }
    }

    #[inline]
    fn get(&self, idx: usize) -> Option<u8> {
        if idx >= self.len {
            return None;
        }
        Some(
            self.chunks[idx >> SPARSE_SHIFT]
                .get()
                .map_or(self.default, |c| c[idx & SPARSE_MASK].load(Ordering::Relaxed)),
        )
    }

    #[inline]
    fn set(&self, idx: usize, val: u8) {
        if idx >= self.len {
            return;
        }
        let slot = &self.chunks[idx >> SPARSE_SHIFT];
        if val == self.default {
            if let Some(c) = slot.get() {
                c[idx & SPARSE_MASK].store(val, Ordering::Relaxed);
            }
            return;
        }
        let default = self.default;
        let c = slot.get_or_init(|| {
            let mut v = Vec::with_capacity(SPARSE_CHUNK);
            v.resize_with(SPARSE_CHUNK, || std::sync::atomic::AtomicU8::new(default));
            v.into_boxed_slice().try_into().ok().unwrap()
        });
        c[idx & SPARSE_MASK].store(val, Ordering::Relaxed);
    }
}

/// Demand-paged atomic bitset in 4096-slot (`64 * 64`) chunks (512 bytes/chunk).
struct SparseBits {
    len: usize,
    chunks: Box<[std::sync::OnceLock<Box<[AtomicU64; SPARSE_CHUNK / 64]>>]>,
}

impl SparseBits {
    fn new(len: usize) -> Self {
        let count = len.div_ceil(SPARSE_CHUNK);
        let chunks = (0..count).map(|_| std::sync::OnceLock::new()).collect();
        Self { len, chunks }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn get(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        let Some(c) = self.chunks[idx >> SPARSE_SHIFT].get() else {
            return false;
        };
        let local = idx & SPARSE_MASK;
        (c[local >> 6].load(Ordering::Acquire) >> (local & 63)) & 1 != 0
    }

    #[inline]
    fn set(&self, idx: usize, val: bool) {
        if idx >= self.len {
            return;
        }
        let slot = &self.chunks[idx >> SPARSE_SHIFT];
        let local = idx & SPARSE_MASK;
        let bit = 1u64 << (local & 63);
        if !val {
            if let Some(c) = slot.get()
                && c[local >> 6].load(Ordering::Relaxed) & bit != 0
            {
                c[local >> 6].fetch_and(!bit, Ordering::AcqRel);
            }
            return;
        }
        let c = slot.get_or_init(|| Box::new([const { AtomicU64::new(0) }; SPARSE_CHUNK / 64]));
        if c[local >> 6].load(Ordering::Relaxed) & bit == 0 {
            c[local >> 6].fetch_or(bit, Ordering::AcqRel);
        }
    }

    #[inline]
    fn clear_if_set(&self, idx: usize) -> bool {
        if idx >= self.len {
            return false;
        }
        let Some(c) = self.chunks[idx >> SPARSE_SHIFT].get() else {
            return false;
        };
        let local = idx & SPARSE_MASK;
        let bit = 1u64 << (local & 63);
        c[local >> 6].load(Ordering::Relaxed) & bit != 0
            && c[local >> 6].fetch_and(!bit, Ordering::AcqRel) & bit != 0
    }
}

struct ResourceEntry {
    raw: usize,
    resource: ID3D12Resource,
    refs: u32,
}

/// Deduplicated host COM resource table backing title descriptor slots (`0..count`),
/// replacing `1.245M` dense `Mutex<Option<ID3D12Resource>>` slots with compact
/// `u32` table IDs (`0` = none) and exact per-resource reference counts (`ARCH-05`).
struct ResourceTable {
    /// Index `0` is reserved for `None`.
    entries: Vec<Option<ResourceEntry>>,
    free: Vec<u32>,
    by_raw: std::collections::HashMap<usize, u32, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
}

impl ResourceTable {
    fn new() -> Self {
        Self {
            entries: vec![None],
            free: Vec::new(),
            by_raw: Default::default(),
        }
    }

    #[inline]
    fn raw_of(&self, id: u32) -> usize {
        self.entries
            .get(id as usize)
            .and_then(|e| e.as_ref())
            .map_or(0, |e| e.raw)
    }

    fn clone_of(&self, id: u32) -> Option<ID3D12Resource> {
        self.entries
            .get(id as usize)
            .and_then(|e| e.as_ref())
            .map(|e| e.resource.clone())
    }

    fn acquire_owned(&mut self, resource: ID3D12Resource) -> (u32, Option<ID3D12Resource>) {
        use windows::core::Interface;
        let raw = resource.as_raw() as usize;
        if let Some(&id) = self.by_raw.get(&raw) {
            if let Some(entry) = self.entries[id as usize].as_mut() {
                entry.refs += 1;
            }
            return (id, Some(resource));
        }
        let id = self.alloc_slot(ResourceEntry { raw, resource, refs: 1 });
        self.by_raw.insert(raw, id);
        (id, None)
    }

    fn acquire_ref(&mut self, resource: &ID3D12Resource) -> u32 {
        use windows::core::Interface;
        let raw = resource.as_raw() as usize;
        if let Some(&id) = self.by_raw.get(&raw) {
            if let Some(entry) = self.entries[id as usize].as_mut() {
                entry.refs += 1;
            }
            return id;
        }
        let id = self.alloc_slot(ResourceEntry {
            raw,
            resource: resource.clone(),
            refs: 1,
        });
        self.by_raw.insert(raw, id);
        id
    }

    fn alloc_slot(&mut self, entry: ResourceEntry) -> u32 {
        if let Some(id) = self.free.pop() {
            self.entries[id as usize] = Some(entry);
            id
        } else {
            let id = self.entries.len() as u32;
            self.entries.push(Some(entry));
            id
        }
    }

    fn release(&mut self, id: u32) -> Option<ID3D12Resource> {
        if id == 0 {
            return None;
        }
        let entry = self.entries.get_mut(id as usize)?.as_mut()?;
        entry.refs -= 1;
        if entry.refs > 0 {
            return None;
        }
        let entry = self.entries[id as usize].take()?;
        self.by_raw.remove(&entry.raw);
        self.free.push(id);
        Some(entry.resource)
    }
}

/// The record each slot of a heap was last made from, and a lock for every
/// few hundredth of the slots: whoever makes a slot holds its lock, and no
/// other.
///
/// One lock for the heap was what everything of a frame waited for once the
/// lists were translated by threads of ours: half of what those did, two
/// thirds of the heaps' records and half of the pass over the buffer views
/// (profile thin-2).
struct Made {
    keys: SparseU64,
    locks: [Mutex<()>; Self::LOCKS],
}

impl Made {
    const LOCKS: usize = 512;

    fn new(count: usize) -> Self {
        Made {
            keys: SparseU64::new(count),
            locks: [const { Mutex::new(()) }; Self::LOCKS],
        }
    }

    /// The lock slot `index` is made under. Neighbours have different ones.
    fn lock(&self, index: u32) -> std::sync::MutexGuard<'_, ()> {
        self.locks[index as usize % Self::LOCKS].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Non-blocking acquire of the lock slot `index` is made under.
    fn try_lock(&self, index: u32) -> Option<std::sync::MutexGuard<'_, ()>> {
        match self.locks[index as usize % Self::LOCKS].try_lock() {
            Ok(g) => Some(g),
            Err(std::sync::TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }

    fn get(&self, index: u32) -> Option<u64> {
        self.keys.get(index as usize, Ordering::Acquire)
    }

    /// Under the slot's lock.
    fn set(&self, index: u32, key: u64) {
        self.keys.set(index as usize, key, Ordering::Release);
    }
}

/// Initial dynamic descriptor reservation for raw memory page views.
pub(crate) const FIRST_RAW_PAGE_SLOTS: u32 = 4096;

struct RawPageRun {
    base: u32,
    slots: u32,
    populated: u32,
}

/// Word-granular bitmap allocator for dynamic descriptor slots.
/// Bit `i` is `1` when taken and `0` when free; padding bits beyond `cap` in
/// the final word are initialized to `1` so word scans never cross `cap`.
pub(crate) struct DynamicPool {
    cap: u32,
    cursor: u32,
    bits: Vec<u64>,
}

impl DynamicPool {
    pub(crate) fn new(cap: u32) -> Self {
        let words = (cap as usize).div_ceil(64);
        let mut bits = vec![0u64; words];
        let rem = (cap as usize) & 63;
        if rem != 0 {
            bits[words - 1] = !((1u64 << rem) - 1);
        }
        Self { cap, cursor: 0, bits }
    }

    fn find_in_range(&self, start: usize, end: usize, n: usize) -> Option<usize> {
        if n == 0 || end <= start || end - start < n {
            return None;
        }
        let mut run = 0usize;
        let mut pos = start;
        while pos < end {
            let word_idx = pos >> 6;
            let bit = pos & 63;
            let w = self.bits[word_idx];
            let word_end = ((word_idx + 1) << 6).min(end);
            let avail = word_end - pos;
            if bit == 0 && avail == 64 {
                if w == 0 {
                    run += 64;
                    if run >= n {
                        return Some(pos + 64 - run);
                    }
                    pos += 64;
                    continue;
                }
                if w == u64::MAX {
                    run = 0;
                    pos += 64;
                    continue;
                }
            }
            let mask = if avail == 64 { u64::MAX } else { (1u64 << avail) - 1 };
            let mut rem = (w >> bit) | !mask;
            let mut offset = 0usize;
            while offset < avail {
                let zeros = (rem.trailing_zeros() as usize).min(avail - offset);
                if zeros > 0 {
                    run += zeros;
                    if run >= n {
                        return Some(pos + offset + zeros - run);
                    }
                    offset += zeros;
                    if zeros == 64 {
                        break;
                    }
                    rem >>= zeros;
                }
                if offset >= avail {
                    break;
                }
                run = 0;
                let ones = (rem.trailing_ones() as usize).min(avail - offset);
                offset += ones;
                if ones == 64 {
                    break;
                }
                rem >>= ones;
            }
            pos = word_end;
        }
        None
    }

    /// Finds a contiguous run of `n` free slots starting from `cursor`, then
    /// wrapping from `0` (extending past `cursor` by `n - 1` so runs crossing
    /// `cursor` are found, while never wrapping across `cap`).
    pub(crate) fn find(&self, n: u32) -> Option<u32> {
        let n = n as usize;
        let cap = self.cap as usize;
        if n == 0 || n > cap {
            return None;
        }
        let cursor = self.cursor as usize;
        self.find_in_range(cursor, cap, n)
            .or_else(|| {
                let second_end = cursor.saturating_add(n.saturating_sub(1)).min(cap);
                self.find_in_range(0, second_end, n)
            })
            .map(|first| first as u32)
    }

    pub(crate) fn mark_taken(&mut self, first: u32, n: u32) {
        let mut pos = first as usize;
        let end = first.saturating_add(n).min(self.cap) as usize;
        while pos < end {
            let w = pos >> 6;
            let bit = pos & 63;
            let count = (64 - bit).min(end - pos);
            let mask = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << bit };
            self.bits[w] |= mask;
            pos += count;
        }
        if self.cap > 0 {
            self.cursor = (first.saturating_add(n) % self.cap).min(self.cap - 1);
        }
    }

    /// Clears `[first, first + n)` and returns how many bits were previously set.
    pub(crate) fn mark_free(&mut self, first: u32, n: u32) -> u32 {
        let mut pos = first as usize;
        let end = first.saturating_add(n).min(self.cap) as usize;
        let mut freed = 0u32;
        while pos < end {
            let w = pos >> 6;
            let bit = pos & 63;
            let count = (64 - bit).min(end - pos);
            let mask = if count == 64 { u64::MAX } else { ((1u64 << count) - 1) << bit };
            let old = self.bits[w];
            freed += (old & mask).count_ones();
            self.bits[w] = old & !mask;
            pos += count;
        }
        freed
    }

    /// Rewinds the bump cursor to `first` if `[first, first + n)` ends right at
    /// the current cursor, keeping recycled slab tails immediately reusable.
    pub(crate) fn rewind_if_tail(&mut self, first: u32, n: u32) {
        if self.cap > 0 && first.saturating_add(n) % self.cap == self.cursor {
            self.cursor = first;
        }
    }

    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) fn is_taken(&self, index: u32) -> Option<bool> {
        (index < self.cap).then(|| (self.bits[(index >> 6) as usize] >> (index & 63)) & 1 != 0)
    }
}

/// Stack-buffered batch of single-descriptor CPU-to-shader-visible copies.
/// Flushes up to 64 pairs in one `ID3D12Device::CopyDescriptors` call.
pub(crate) struct DescriptorCopyBatch {
    dsts: [D3D12_CPU_DESCRIPTOR_HANDLE; Self::CAP],
    srcs: [D3D12_CPU_DESCRIPTOR_HANDLE; Self::CAP],
    len: usize,
}

impl DescriptorCopyBatch {
    const CAP: usize = 64;

    pub(crate) fn new() -> Self {
        Self {
            dsts: [D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }; Self::CAP],
            srcs: [D3D12_CPU_DESCRIPTOR_HANDLE { ptr: 0 }; Self::CAP],
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn remaining(&self) -> usize {
        Self::CAP - self.len
    }

    #[inline]
    pub(crate) fn push(&mut self, dst: D3D12_CPU_DESCRIPTOR_HANDLE, src: D3D12_CPU_DESCRIPTOR_HANDLE) {
        self.dsts[self.len] = dst;
        self.srcs[self.len] = src;
        self.len += 1;
        if self.len == Self::CAP {
            self.flush();
        }
    }

    #[inline]
    pub(crate) fn flush(&mut self) {
        if self.len == 0 {
            return;
        }
        let n = self.len;
        self.len = 0;
        let Some(host) = host() else { return };
        // SAFETY: `dsts[..n]` are live shader-visible CBV/SRV/UAV handles and
        // `srcs[..n]` are live non-shader-visible CPU handles (ListViews or CachedNulls).
        unsafe {
            if n == 1 {
                host.device.CopyDescriptorsSimple(
                    1,
                    self.dsts[0],
                    self.srcs[0],
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                );
            } else {
                host.device.CopyDescriptors(
                    n as u32,
                    self.dsts.as_ptr(),
                    None,
                    n as u32,
                    self.srcs.as_ptr(),
                    None,
                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                );
            }
        }
    }
}

impl Drop for DescriptorCopyBatch {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Exclusive per-slot resource retention for dynamic descriptor slots
/// (`index >= count`). Each dynamic slot is leased to a single `HostList` /
/// recording thread via `HostHeap::next` (`Mutex<DynamicPool>`) and returned
/// only after list reset, establishing a happens-before edge between owners
/// with zero per-slot mutex overhead.
struct DynamicKept(Box<[std::cell::UnsafeCell<Option<ID3D12Resource>>]>);

// SAFETY: Every slot `i` is accessed exclusively by the single thread that
// currently owns dynamic slot `count + i` between `DynamicPool::mark_taken`
// and `DynamicPool::mark_free` (synchronized by `HostHeap::next`).
unsafe impl Sync for DynamicKept {}

impl DynamicKept {
    fn new(len: usize) -> Self {
        Self((0..len).map(|_| std::cell::UnsafeCell::new(None)).collect())
    }

    #[inline]
    fn replace_ref(&self, idx: usize, resource: Option<&ID3D12Resource>) {
        use windows::core::Interface;
        let Some(cell) = self.0.get(idx) else {
            return;
        };
        // SAFETY: exclusive owner of leased dynamic slot `idx`.
        let slot = unsafe { &mut *cell.get() };
        if slot.as_ref().map(Interface::as_raw) == resource.map(Interface::as_raw) {
            return;
        }
        if let Some(old) = std::mem::replace(slot, resource.cloned()) {
            crate::host::retire_resource(old);
        }
    }

    #[inline]
    fn replace(&self, idx: usize, resource: Option<ID3D12Resource>) {
        use windows::core::Interface;
        let Some(cell) = self.0.get(idx) else {
            return;
        };
        // SAFETY: exclusive owner of leased dynamic slot `idx`.
        let slot = unsafe { &mut *cell.get() };
        if slot.as_ref().map(Interface::as_raw) == resource.as_ref().map(Interface::as_raw) {
            return;
        }
        if let Some(old) = std::mem::replace(slot, resource) {
            crate::host::retire_resource(old);
        }
    }
}

pub(crate) struct HostHeap {
    pub(crate) buffer_sweep: Mutex<crate::draw::BufferSweep>,
    pub(crate) buffer_frame: std::sync::atomic::AtomicU64,
    pub(crate) heap: ID3D12DescriptorHeap,
    /// Guest base address of slot 0 in the title's view heap (lazily captured
    /// from `mirror` / `mirror_batch`; 0 until seen). Used by
    /// [`HostHeap::unpin_stale_window_slots`] to safely revalidate guest
    /// descriptor bytes before unpinning dead loose windows (Spec-Trap #16).
    title_start: std::sync::atomic::AtomicU64,
    cpu: usize,
    gpu: u64,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count`: [`DYNAMIC`] in view heaps, none in
    /// sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    sampler: bool,
    /// Where the raw twins of the title's slots start
    /// (`victory_shader::emit::raw_views`; 0: there are none), and which of
    /// them are not null (demand-paged 4K-slot bitset; covers `0..count + dynamic`
    /// because `port_cbv` table CBVs and Wave 2 TLAS twins read `self.raw + index`).
    raw: u32,
    twins: SparseBits,
    /// When each slot was written last (with `crate::book`; else empty).
    #[cfg(feature = "diagnostics")]
    written: Vec<std::sync::atomic::AtomicU32>,
    /// The record (view index and generation) each slot was last made from.
    made: Made,
    /// The same for a slot whose descriptor is made and stays (else 0): what
    /// [`HostHeap::mirror`] asks first, without the lock. Changed under it.
    quick: SparseU64,
    /// The records of the title's heap as the pass over records copied in
    /// as bytes last saw them ([`Copied::changes`]). Made on first
    /// use.
    copied: Mutex<Vec<u8>>,
    /// New ones of them that wait for a draw ([`HostHeap::copied_wait`]),
    /// keyed by the frame at which they become due so deferred `copied_tabled`
    /// slots (`stamp + COPIED_FRAMES + 1`) never block earlier `COPIED_GRACE`
    /// batches.
    waiting: Mutex<std::collections::BTreeMap<u64, Vec<u32>>>,
    /// Non-view slot unmakes deferred off `sync_heaps_marked` onto maintenance
    /// so `victory-submit` does not wait for `~4k/f` unmakes before `SYNC_DONE`.
    unmakes: Mutex<Vec<u32>>,
    /// The slots made from a texture whose memory was found away, with what
    /// they were made from and when that was
    /// ([`HostHeap::texture_slot_gone`]).
    away: Mutex<std::collections::HashMap<u32, (u64, u64)>>,
    /// The host resource each title slot (`0..count`) views, stored as compact
    /// `u32` IDs (`0` = none) into deduplicated [`ResourceTable`], with striped
    /// per-slot locks for concurrent updates. Dynamic slots (`count..`) use
    /// lock-free [`DynamicKept`] because each slot has an exclusive owner.
    kept_ids: SparseU32,
    kept_locks: [Mutex<()>; 64],
    resources: Mutex<ResourceTable>,
    /// Title slots (`0..count`) referencing `window == true` loose shadow
    /// buffers, grouped by COM raw pointer for O(evicted) weak-pin eviction
    /// (`ARCH-05`).
    window_slots: Mutex<std::collections::HashMap<usize, Vec<u32>, std::hash::BuildHasherDefault<crate::host::AddressHash>>>,
    /// Lock-free filter of title slots present in `window_slots` so non-window
    /// updates never touch `window_slots.lock()`.
    window_slot_bits: SparseBits,
    dynamic_kept: DynamicKept,
    /// Word-granular bitmap allocator for dynamic slots (`count..count + dynamic`).
    next: Mutex<DynamicPool>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `2 * SLOT_STRIDES_COUNT` dwords (lower half stride|kind, upper half
    /// element count at fixed `SLOT_STRIDES_COUNT` offset required by shader
    /// root-SRV indexing in `lower.rs`), its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// Sparse 4K-chunk CPU-RAM shadow of packed `(word | count << 32)` per slot
    /// to skip unchanged scattered WC `write_volatile` stores (`CPU-30`, `ARCH-05`).
    stride_cache: SparseU64,
    /// Runs of raw-memory copy UAVs grow geometrically. Older reservations
    /// stay valid until this heap is destroyed because recorded command lists
    /// still use their base.
    raw_pages: Mutex<Option<RawPageRun>>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    buffers: Slots,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Slots,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// Lock-free filter for slots currently in `textures.slots`, avoiding
    /// `textures.lock()` contention on non-texture and unqueued slots.
    queued_textures: Slots,
    /// Per main slot (demand-paged 4K chunks, default `1`): `0` = live/other
    /// descriptor, `1..=17` = cached null kind already written there, skipping
    /// redundant `CopyDescriptorsSimple` calls.
    null_kind: SparseU8,
    /// Per title slot (demand-paged 4K chunks), the 64-bit monotonic frame epoch
    /// (plus one; 0: never) in which a draw last read it through a table, or made
    /// a table it lies in (with [`TABLED_READ`] set in bit 63 if the draw read it).
    tabled: SparseU64,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// The slots that may be in `placed` (a slot noted there is put here
    /// under its lock, and taken out when it is forgotten there; a slot
    /// another's view made stale stays, which says only "may"): a write of
    /// any other view needs nothing of `placed` ([`spans_quick`]).
    spanned: Slots,
    /// [`NEXT_ID`]'s number for this heap ([`HostHeap::release`]).
    id: u64,
}

/// Whether a slot's write of no placed texture view leaves `placed` alone
/// when the slot is not in it ([`HostHeap::spanned`]; `VICTORY_SPANS_QUICK=0`:
/// its lock is taken for every write, as before).
///
/// Every record the pass over records copied as bytes mirrors went by that
/// one lock of the heap, from six threads, with the title's own threads and
/// the texture passes: 4 of the 16 ms a frame the submitting thread waited
/// for that pass were its threads waiting there (run city4, timeline 113).
/// What is done is the same: forgetting a slot that is not there does
/// nothing.
/// Since the last frame line: writes that left `placed` alone
/// ([`spans_quick`]), writes that took its lock.
pub(crate) static SPANS_QUICK: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

fn spans_quick() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SPANS_QUICK") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPANS_QUICK").is_none_or(|v| v != "0"))
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    bump_view_cache_epoch();
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
}

/// [`levels_changed`] once the submissions begun so far (that of the list
/// whose step calls this among them) have run.
pub(crate) fn levels_changed_after(start: u64) {
    bump_view_cache_epoch();
    let after = crate::host::submissions().0;
    RESHOWN_AFTER.lock().unwrap_or_else(|e| e.into_inner()).push((after, start));
}

static RESHOWN_AFTER: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Slots that show a placed texture from another level than the title's view
/// in them says now, and are made from that view when the upload it needs
/// has run ([`HostHeap::texture_slot_flip`]).
struct Flip {
    /// The submission to wait for.
    after: u64,
    /// The heap (its address), the slot, where its record is and which
    /// record it was.
    heap: usize,
    index: u32,
    handle: u64,
    key: u64,
    /// The frame it was asked in, of a slot that shows nothing of the
    /// texture until then ([`HostHeap::texture_slot_show_after`]).
    late: Option<u64>,
}

/// Whether a title slot shows a texture that is new to it only once the
/// submission with its upload has finished (`VICTORY_SHOW_AFTER=1`; off).
/// Shown at once, the lists already on the card read it by index while it is
/// made: all three losses whose report had the address book (n08, n11, n15)
/// were page faults in a texture made and shown in the last tenth of a
/// second.
pub(crate) fn show_after() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHOW_AFTER").is_some_and(|v| v != "0"))
}

/// Slots shown late ([`show_after`]), the frames they waited, the longest;
/// those that were given the neutral texture meanwhile; those not shown any
/// more when their time came.
static LATE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a texture slot showed before its last write: its record (0: null or
/// the neutral texture), when (ms), and the submissions begun and finished.
#[cfg(feature = "diagnostics")]
static BEFORE: Mutex<std::collections::BTreeMap<u32, (u64, u64, (u64, u64), &'static str)>> =
    Mutex::new(std::collections::BTreeMap::new());

#[cfg(feature = "diagnostics")]
fn note_before(index: u32, was: u64, by: &'static str) {
    if crate::book::on() {
        let entry = (was, crate::host::now_ms(), crate::host::submissions(), by);
        BEFORE.lock().unwrap_or_else(|e| e.into_inner()).insert(index, entry);
    }
}

/// A 1x1 gray texture, in the place of one that is not there yet.
fn neutral() -> Option<&'static ID3D12Resource> {
    static TEXTURE: std::sync::OnceLock<Option<ID3D12Resource>> = std::sync::OnceLock::new();
    TEXTURE
        .get_or_init(|| {
            let host = host()?;
            let device = &host.device;
            let (from, data) = host.upload_buffer(256)?;
            // SAFETY: the buffer's mapping; plain creation and recording,
            // waited for before what it names goes.
            unsafe {
                std::ptr::write_bytes(data, 0x80, 256);
                let mut made: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: 1,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut made,
                    )
                    .ok()?;
                let made = made?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
                let list: ID3D12GraphicsCommandList =
                    device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None).ok()?;
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&made),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                let footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 1,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                };
                let source = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&from),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
                };
                list.CopyTextureRegion(&to, 0, 0, 0, &source, None);
                list.Close().ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).ok()?)]);
                queue.Signal(&fence, 1).ok()?;
                // (Without an event: this waits.)
                fence.SetEventOnCompletion(1, windows::Win32::Foundation::HANDLE::default()).ok()?;
                Some(made)
            }
        })
        .as_ref()
}

/// Makes the descriptor of [`neutral`] for a slot of `view` at `at`, if the
/// view is a 2D texture's or an array's; else a null one of its kind.
fn neutral_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    const TEXTURE2D: u32 = 4;
    const TEXTURE2D_ARRAY: u32 = 5;
    let dimension = match view {
        View::Shader(_, Some(desc)) => desc[1],
        View::Placed(false, resource, desc) if resource[0] != 1 => desc[1],
        _ => 0,
    };
    let desc = match dimension {
        TEXTURE2D => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
            },
        },
        TEXTURE2D_ARRAY => D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV { MipLevels: 1, ArraySize: 1, ..Default::default() },
            },
        },
        _ => {
            null_for(view, at);
            return None;
        }
    };
    let (Some(texture), Some(host)) = (neutral(), host()) else {
        null_for(view, at);
        return None;
    };
    // SAFETY: a slot of our heap, a live texture.
    unsafe { host.device.CreateShaderResourceView(texture, Some(&desc), at) };
    Some(texture.clone())
}

static FLIPS: Mutex<Vec<Flip>> = Mutex::new(Vec::new());

/// Deferred slot flips recorded into an open or not-yet-submitted command
/// list (`HostHeap::flip`), tracked so a one-shot decoration bake draw can
/// promote them out of the `neutral()` gray stand-in before sampling them.
pub(crate) struct PendingListFlip {
    id: u64,
    pub(crate) origin: (usize, u64),
    flip: Flip,
}

impl PendingListFlip {
    pub(crate) fn index(&self) -> u32 {
        self.flip.index
    }

    pub(crate) fn handle(&self) -> u64 {
        self.flip.handle
    }

    pub(crate) fn key(&self) -> u64 {
        self.flip.key
    }
}

static NEXT_PENDING_FLIP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static PENDING_LIST_FLIPS: Mutex<Vec<PendingListFlip>> = Mutex::new(Vec::new());
static PENDING_LIST_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn take_pending_list_flip(id: u64) -> Option<Flip> {
    if PENDING_LIST_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
        return None;
    }
    let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
    let pos = pending.iter().position(|p| p.id == id)?;
    let entry = pending.swap_remove(pos);
    PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
    Some(entry.flip)
}

fn remove_pending_list_flip(id: u64) {
    let _ = take_pending_list_flip(id);
}

/// Makes the slots whose new texture's upload has run
/// ([`HostHeap::texture_slots_flip`]); on any thread. Not if the heap is at
/// something else: the next texture pass does.
pub(crate) fn flip_now() {
    bump_view_cache_epoch();
    let _ = super::with_view_heap(HostHeap::texture_slots_flip);
}

/// Since the start: slots made from their new view after its upload had run,
/// and those the title had written again by then.
pub(crate) static FLIPPED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The key of the record at `handle`, a view's descriptor slot; 0: none.
fn record_key(handle: u64) -> u64 {
    // SAFETY: the handle is a descriptor slot in title memory.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag == VIEW_TAG { u64::from(generation) << 32 | u64::from(i) } else { 0 }
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Placed textures (by the start of their memory) filled while the runtime
/// was remapping with less than was mapped of them
/// (`crate::texture::fill_unlocked`), whose slots the next texture pass
/// makes due again; and how often in a row that was, which ends it: memory
/// that is shown and cannot be read would be uploaded again every frame.
static SHORT: Mutex<(std::collections::BTreeSet<u64>, std::collections::BTreeMap<u64, u32>)> =
    Mutex::new((std::collections::BTreeSet::new(), std::collections::BTreeMap::new()));

/// Fillings in a row of one texture that got less than was mapped, after
/// which its slots wait for the next notice of its memory.
const SHORT_IN_A_ROW: u32 = 8;

/// The placed texture at `start` was filled with less than was mapped.
pub(crate) fn read_short(start: u64) {
    let mut short = SHORT.lock().unwrap_or_else(|e| e.into_inner());
    let times = short.1.entry(start).or_insert(0);
    *times += 1;
    if *times <= SHORT_IN_A_ROW {
        short.0.insert(start);
    }
}

/// The placed texture at `start` was filled with all that was mapped.
pub(crate) fn read_all(start: u64) {
    let mut short = SHORT.lock().unwrap_or_else(|e| e.into_inner());
    if !short.1.is_empty() {
        short.1.remove(&start);
    }
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The level each slot's view starts at, and where what it is read from
    /// ends ([`Spans::shows`]).
    first: std::collections::HashMap<u32, (u32, u64)>,
    /// Event-driven missing-mip VA index: maps `missing_start` -> `(missing_end, slot)`
    /// for waiting/partial placed textures so `remapped`/`page_written` wakes them in
    /// 1 frame without waiting on 30f/256f exponential backoff (`ARCH-08`).
    missing_by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    missing_by_slot: std::collections::HashMap<u32, u64>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if start == 0 || end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// The view in slot `index` starts at level `first` of its texture; the
    /// memory of that and the smaller levels ends at `end`.
    fn shows(&mut self, index: u32, first: u32, end: u64) {
        self.first.insert(index, (first, end));
    }

    /// Whether slot `index`'s view is read from memory at or past `start`.
    fn reaches(&self, index: u32, start: u64) -> bool {
        self.first.get(&index).is_none_or(|&(_, end)| end > start)
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).map(|&(first, _)| first).min()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        self.by_start.range(start.saturating_add(HEAD)..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        self.first.remove(&index);
        self.clear_missing_mip(index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// Indexes slot `index` as waiting on missing-mip VA range `[start, end)`.
    fn note_missing_mip(&mut self, index: u32, start: u64, end: u64) {
        if start == 0 || end <= start {
            return;
        }
        if let Some(&prev) = self.missing_by_slot.get(&index) {
            if prev == start
                && self.missing_by_start.get(&prev).is_some_and(|(e, _)| *e >= end)
            {
                return;
            }
            self.clear_missing_mip(index);
        }
        let entry = self.missing_by_start.entry(start).or_insert_with(|| (end, Default::default()));
        entry.0 = entry.0.max(end);
        entry.1.insert(index);
        self.missing_by_slot.insert(index, start);
    }

    /// Removes slot `index` from the missing-mip VA index.
    fn clear_missing_mip(&mut self, index: u32) {
        let Some(start) = self.missing_by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.missing_by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.missing_by_start.remove(&start);
            }
        }
    }

    /// Returns waiting/partial texture slots whose missing-mip VA range overlaps `[start, end)`.
    fn waiting_slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start || self.missing_by_start.is_empty() {
            return Vec::new();
        }
        let before = self
            .missing_by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(_, span)| span);
        before
            .into_iter()
            .chain(self.missing_by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .filter(|&slot| self.reaches(slot, start))
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if start == 0 || end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .filter(|&slot| self.reaches(slot, start))
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped; `crate::watch::title_memory_written`), since a texture
/// pass last took the ranges ([`page_written`]). Kept apart from
/// [`REMAPPED`] so that what such writes do is counted by itself.
static PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
/// Physical-page writes retained specifically for one-shot decoration/ped
/// texture bakes when `VICTORY_PAGE_WRITES` bit 2 is off (the default): normal
/// frames stay free of queue flooding while every bake still sees freshly
/// DirectStorage-streamed texture bytes immediately instead of waiting on
/// `RECHECK_FRAMES`.
static BAKE_PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose texture had been uploaded
/// before the write (out of the queue, or in it after an upload): they are
/// the ones that kept the bytes from before, and the ones made due when
/// `VICTORY_PAGE_WRITES` asks for it.
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Frames from the pass that takes a written range to the look at its
/// slots: the levels of one texture arrive in a few requests, and a look
/// two frames on takes those of the next frame with it.
const PAGE_WRITE_AFTER: u64 = 2;

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges; locks one mutex and calls nothing.
pub(crate) fn page_written(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Disjoint ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    let dropped = push_remapped(&mut ranges, (start, end), MOST);
    if dropped > 0 {
        PAGE_WRITES[1].fetch_add(dropped, Relaxed);
    }
}

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// Appends `range` to `ranges`, coalescing adjacent/overlapping entries on
/// overflow before ever dropping anything. Returns how many ranges were dropped
/// (0 in normal play; if `ranges` still reaches `most` disjoint ranges after
/// coalescing, drops only the oldest half rather than clearing the whole queue).
fn push_remapped(ranges: &mut Vec<(u64, u64)>, (start, end): (u64, u64), most: usize) -> u64 {
    if let Some(last) = ranges.last_mut()
        && last.1 >= start
        && last.0 <= end
    {
        last.0 = last.0.min(start);
        last.1 = last.1.max(end);
        return 0;
    }
    let mut dropped = 0u64;
    if ranges.len() >= most {
        join(ranges);
        if ranges.len() >= most {
            let keep = (most / 2).min(most.saturating_sub(1));
            let drop = ranges.len() - keep;
            ranges.drain(..drop);
            dropped = drop as u64;
        }
    }
    ranges.push((start, end));
    dropped
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Disjoint ranges kept until a pass takes them.
    const MOST: usize = 1 << 16;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    let dropped = push_remapped(&mut ranges, (start, end), MOST);
    if dropped > 0 {
        STALE[2].fetch_add(dropped, Relaxed);
    }
}

/// Ranges of memory, those that touch made one (DexNom/VIctory 921d8d3b:
/// several threads map and write, some 5,000 ranges a frame, the same
/// textures' over and over, and each was looked up).
fn join(ranges: &mut Vec<(u64, u64)>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, last| {
        if next.0 > last.1 {
            return false;
        }
        last.1 = last.1.max(next.1);
        true
    });
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 120;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 4;

/// Frames for which a record copied into a slot a draw read through a
/// table is left to such a draw ([`HostHeap::copied_tabled`]). Raised from 8
/// to 48 (`CPU-02`) to eliminate frame-ring wrap expiry cascades.
const COPIED_FRAMES: u64 = 48;

/// Frames a record copied into any other slot waits for one, before it is
/// taken for one shaders pick by index ([`HostHeap::copied_wait`]).
const COPIED_GRACE: u64 = 4;

/// In [`HostHeap::tabled`]: bit 63 indicates a draw read the slot in that frame.
const TABLED_READ: u64 = 1 << 63;

/// Encodes monotonic 64-bit `frame` as a non-zero tabled stamp (without [`TABLED_READ`]).
#[inline]
fn tabled_now(frame: u64) -> u64 {
    frame.saturating_add(1) & !TABLED_READ
}

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
}

thread_local! {
    /// This thread is inside [`HostHeap::publish`].
    static PUBLISHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
}

impl TextureQueue {
    /// Maximum already-uploaded `RECHECK_FRAMES` slots handed out per `due()`
    /// call so burst rechecks never preempt fresh uploads or trigger false
    /// catch-up passes.
    const RECHECK_CAP: usize = 256;

    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in three
    /// priority tiers:
    /// 1. partial uploads (newest write first) and fresh / mip-upgrade writes
    ///    (newest and oldest alternating to avoid starving either end of a
    ///    scene-load backlog);
    /// 2. unmapped-memory retries (`looks > 0`, never uploaded), newest first;
    /// 3. already-uploaded `RECHECK_FRAMES` zombies (`checked.2 != 0`), oldest
    ///    due first and capped at [`Self::RECHECK_CAP`].
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let mut partial: Vec<(u32, u64)> = Vec::new();
        let mut fresh: Vec<(u32, u64)> = Vec::new();
        let mut unmapped: Vec<(u32, u64)> = Vec::new();
        let mut recheck: Vec<(u64, u32, u64)> = Vec::new();
        for (&index, w) in self.slots.iter().filter(|(_, w)| w.due <= frame) {
            if w.checked.2 != 0 {
                recheck.push((w.due, index, w.write));
            } else if w.mapped > 0 {
                partial.push((index, w.write));
            } else if w.looks == 0 {
                fresh.push((index, w.write));
            } else {
                unmapped.push((index, w.write));
            }
        }
        let newest_first = |a: &(u32, u64), b: &(u32, u64)| b.1.cmp(&a.1);
        partial.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);
        unmapped.sort_unstable_by(newest_first);
        // Most overdue rechecks first so every uploaded slot rotates under the cap.
        recheck.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(&b.2)));

        let mut order: Vec<(u32, u64)> = Vec::with_capacity(
            (partial.len() + fresh.len() + unmapped.len() + recheck.len().min(Self::RECHECK_CAP)).min(max),
        );
        // Tier 1: partial uploads, then fresh / mip-upgrade writes (newest/oldest interleaved).
        order.extend(partial.into_iter().take(max));
        let (mut new, mut old) = (0, fresh.len());
        while new < old && order.len() < max {
            order.push(fresh[new]);
            new += 1;
            if new < old && order.len() < max {
                old -= 1;
                order.push(fresh[old]);
            }
        }
        // Tier 2: unmapped-memory retries.
        if order.len() < max {
            let rem = max - order.len();
            order.extend(unmapped.into_iter().take(rem));
        }
        // Tier 3: capped recheck zombies.
        if order.len() < max {
            let rem = (max - order.len()).min(Self::RECHECK_CAP);
            order.extend(recheck.into_iter().take(rem).map(|(_, index, write)| (index, write)));
        }
        order
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.slots.remove(&index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
            w.checked = (0, 0, 0, 0, 0);
        }
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
            self.held.insert(index, write);
        }
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            if let Some(w) = self.same(index, write) {
                w.due = w.due.min(frame);
                back += 1;
            }
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    #[allow(clippy::too_many_arguments)]
    fn recheck(&mut self, index: u32, write: u64, due: u64, at: (u64, u64), print: u64, looks: u8, uploaded: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
        }
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look.
    fn checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.slots.get(&index).map(|w| w.checked).filter(|c| c.2 != 0)
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    /// Wakes a waiting/partial slot on missing-mip page arrival in 1 frame
    /// without resetting `w.mapped` (preserving already-uploaded mip progress).
    fn wake_missing(&mut self, index: u32, due: u64) -> bool {
        if let Some(w) = self.slots.get_mut(&index) {
            w.due = w.due.min(due);
            w.looks = 0;
            true
        } else {
            false
        }
    }

    /// Returns `(index, write)` for every slot in `wanted` currently present
    /// in the queue (regardless of future `due` frame), for scoped bakes.
    fn for_slots(&self, wanted: &[u32]) -> Vec<(u32, u64)> {
        wanted
            .iter()
            .filter_map(|&idx| self.slots.get(&idx).map(|w| (idx, w.write)))
            .collect()
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded; excludes
    /// already-uploaded rechecks); uploaded in part; and the frames the
    /// oldest due non-recheck slot has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 && w.checked.2 == 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                if w.checked.2 == 0 {
                    c[4] = c[4].max(frame.saturating_sub(w.written));
                }
            }
        }
        c
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    const STRUCTURE: u32 = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, desc) => {
            if *r == 0 && desc.is_some_and(|d| d[1] == STRUCTURE) && crate::dxr::enabled() {
                false
            } else {
                crate::resource::buffer_bytes(*r as crate::object::This).is_none()
            }
        }
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element and typed integer format bits of a buffer view, decoded
/// in a single pass over the view descriptor; `(None, 0)` if not a buffer view.
fn view_stride_and_kind(view: &View) -> (Option<u32>, u32) {
    use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
    if let View::Shader(0, Some(d)) = view
        && d[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
        && crate::dxr::enabled()
    {
        let guest_tlas_va = u64::from(d[4]) | (u64::from(d[5]) << 32);
        return (None, crate::dxr::tlas_meta_base(guest_tlas_va));
    }
    let Some((stride, raw, format)) = view_elements(view) else {
        return (None, 0);
    };
    let bytes = if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format).map_or(0, |(bytes, _)| bytes as u32)
    };
    let kind = if stride == 0 && !raw {
        match format {
            3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => VIEW_INTEGER,
            4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => VIEW_INTEGER | VIEW_SIGNED,
            _ => 0,
        }
    } else {
        0
    };
    (Some(bytes), kind)
}

/// Packed [`victory_shader::emit::SLOT_STRIDES`] word for `view`.
fn view_stride_word(view: &View) -> u32 {
    let (stride, kind) = view_stride_and_kind(view);
    stride.unwrap_or(0) | kind
}

/// Element count for buffer views (`NumElements`, or byte count `NumElements * 4`
/// for raw views), stored in the upper half of [`HostHeap::strides`] so shaders
/// read buffer sizes without emitting `dx.op.getDimensions(72)` on heap/table
/// buffers (avoids an AMD RDNA 4 `amdxc64.dll` compiler crash).
fn view_count(view: &View) -> u32 {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            let Ok(w) = d[..10].try_into() else {
                return 0;
            };
            (w, *unordered)
        }
        _ => return 0,
    };
    // SAFETY: buffer views' union members.
    unsafe {
        let (num, raw) = if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.NumElements, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.NumElements, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0)
        };
        if raw { num.saturating_mul(4) } else { num }
    }
}

#[cfg(test)]
fn view_stride(view: &View) -> Option<u32> {
    view_stride_and_kind(view).0
}

#[cfg(test)]
fn view_kind(view: &View) -> u32 {
    view_stride_and_kind(view).1
}

/// A buffer view's structure stride, whether it is raw, and its format;
/// `None` if `view` is not a buffer view.
fn view_elements(view: &View) -> Option<(u32, bool, u32)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    Some(unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format.0 as u32)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format.0 as u32)
        }
    })
}

/// Whether a buffer view that begins `offset` bytes into its host copy begins
/// where no host view can: raw views begin on 16 bytes, others on a multiple
/// of their elements (`stride` of a structured one, which is made raw where it
/// can be, `element` of a typed one). The console's begin on any dword, and
/// the title puts parameter blocks at ring positions: a raw view of 128 bytes
/// at 8 (the colour table bakes'), structured ones of elements that are no
/// divisor of their place (the culling's 456-byte constants). Such a view gets
/// a copy of its own bytes ([`crate::shadow::exact`]). (A typed view of
/// 12-byte elements at a multiple of 16 that is none of 12 is one of them: it
/// was passed over, and stayed null.)
fn off_elements(offset: u64, stride: u32, raw: bool, element: u64) -> bool {
    if raw {
        !offset.is_multiple_of(16)
    } else if stride > 0 {
        !offset.is_multiple_of(16) && !offset.is_multiple_of(u64::from(stride))
    } else {
        element > 0 && !offset.is_multiple_of(element)
    }
}

/// Whether `view` is a raytracing acceleration structure SRV descriptor (regardless of DXR flag).
#[inline]
fn is_tlas_srv_desc(view: &View) -> bool {
    matches!(view, View::Shader(0, Some(d)) if d[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32)
}

/// Whether `view` is a native DXR raytracing acceleration structure SRV.
#[inline]
fn is_tlas_srv(view: &View) -> bool {
    is_tlas_srv_desc(view) && crate::dxr::enabled()
}

/// Resolves the mirror (cull-enabled) TLAS GPU VA for `guest_tlas_va` stored in
/// the HostHeap twin slot (`self.raw + index`) for 2-pass hardware
/// `FrontBackTrack` (`ARCH-13`).
#[inline]
fn twin_tlas_va(guest_tlas_va: u64) -> Option<u64> {
    crate::dxr::resolve_mirror_tlas_va(guest_tlas_va)
}

#[inline]
fn twinned_for(view: &View, dxr_enabled: bool) -> bool {
    matches!(view, View::Constant(..)) || view_elements(view).is_some() || (dxr_enabled && is_tlas_srv_desc(view))
}

/// Whether `view` has a twin slot when its host slot is made ([`raw_twins`]):
/// a buffer view, a constant buffer's, or a native DXR TLAS SRV (`ARCH-13`).
fn twinned(view: &View) -> bool {
    twinned_for(view, crate::dxr::enabled())
}

/// Whether views off their copy's elements get a copy of their own bytes. On by
/// default; `VICTORY_VIEW_BYTES=0` leaves them null, as before.
fn view_bytes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_BYTES").is_none_or(|v| v != "0"))
}

/// The copy the buffer view of `[start, start + len)` is made on: `shadow`, the one
/// its buffer has, or one of the view's own bytes ([`off_elements`]).
fn view_home(
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    start: u64,
    len: u64,
    (stride, raw, format): (u32, bool, u32),
) -> std::sync::Arc<crate::shadow::Shadow> {
    let Some(offset) = shadow.offset_in(start).map(|_| shadow.offset(start)) else {
        return shadow;
    };
    if !view_bytes() {
        return shadow;
    }
    let element = crate::resource::element(format).map_or(0, |(bytes, _)| bytes);
    if !off_elements(offset, stride, raw, element) {
        return shadow;
    }
    crate::shadow::exact(start, len).unwrap_or(shadow)
}

/// For device-removed reports: what the slots of the title's view heap that
/// the words of user data may name hold now ([`crate::host::indirects`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn describe_slots(words: &[u32]) -> Vec<(usize, u32, String)> {
    super::with_view_heap(|host| {
        crate::host::indirects::slots(words, host.count, 6)
            .into_iter()
            .take(24)
            .map(|(word, slot)| (word, slot, host.describe_slot(slot).unwrap_or_default()))
            .collect()
    })
    .unwrap_or_default()
}

/// For `crate::book`: the bytes of up to `n` descriptors of the view heap from
/// slot `at` on (from the first twin on, where `at` lies before it and past
/// the others), as the card's heap has them: the first slot, the bytes, and
/// the slot behind the last one worth reading.
#[cfg(feature = "diagnostics")]
pub(crate) fn shown_bytes(at: u32, n: u32) -> Option<(u32, Vec<u8>, u32)> {
    super::with_view_heap(|host| {
        let own = host.count + host.dynamic;
        let end = if host.raw == 0 { own } else { host.raw + own };
        let at = if at >= own && at < host.raw { host.raw } else { at };
        let stop = if at < own { own } else { end };
        let n = n.min(stop.saturating_sub(at));
        if n == 0 || host.step != 32 {
            return None;
        }
        let mut bytes = vec![0u8; n as usize * 32];
        crate::host::read_into(host.cpu(at).ptr as *const u8, &mut bytes).then_some((at, bytes, end))
    })?
}

/// For `crate::book`: what our books say of a slot of the view heap.
#[cfg(feature = "diagnostics")]
pub(crate) fn slot_story(slot: u32) -> String {
    use std::sync::atomic::Ordering::Relaxed;
    use windows::core::Interface;
    super::with_view_heap(|host| {
        let (index, twin) = if host.raw != 0 && slot >= host.raw { (slot - host.raw, true) } else { (slot, false) };
        let mut text = if index < host.count {
            format!("title slot {index}, made from {}", host.describe_slot(index).unwrap_or_default())
        } else {
            let taken = host.next.lock().unwrap_or_else(|e| e.into_inner()).is_taken(index - host.count);
            format!("dynamic slot {} (taken {taken:?})", index - host.count)
        };
        if twin {
            let made = (index < host.twins.len() as u32).then(|| host.twins.get(index as usize));
            text.push_str(&format!(", its raw twin (made {made:?})"));
        }
        if let Some(at) = host.written.get(slot as usize) {
            text.push_str(&format!(", written last at {:.1} s", f64::from(at.load(Relaxed)) / 1e3));
        }
        let before = BEFORE.lock().unwrap_or_else(|e| e.into_inner()).get(&index).copied();
        if let Some((was, at, (started, finished), by)) = before {
            let what = match view_at(was as u32, (was >> 32) as u32) {
                _ if was == 0 => "nothing (or the neutral texture)".into(),
                None => "a view that is gone".into(),
                Some(view) => format!("{view:?}").chars().take(160).collect::<String>(),
            };
            text.push_str(&format!(
                ", changed ({by}) at {:.3} s with submissions {started} begun and {finished} finished, showed before: {what}",
                at as f64 / 1e3
            ));
        }
        let kept = host.kept_in(index);
        match kept {
            None => text.push_str(", holds nothing"),
            Some(resource) => {
                // SAFETY: a resource the slot holds.
                let d = unsafe { resource.GetDesc() };
                let gpu = if d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
                    // SAFETY: as above.
                    unsafe { resource.GetGPUVirtualAddress() }
                } else {
                    0
                };
                text.push_str(&format!(
                    ", holds {:#x}: dimension {} {}x{}x{} levels {} f{} at {gpu:#x}, {} references",
                    resource.as_raw() as usize,
                    d.Dimension.0,
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.MipLevels,
                    d.Format.0,
                    crate::host::references(&resource)
                ));
            }
        }
        text
    })
    .unwrap_or_default()
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The size of every host view heap on a host that takes more than
/// [`PORTABLE_VIEWS`] (AMD): the title's heap (a little over 1,000,000
/// views in the retail game) plus [`DYNAMIC`] fits with room to spare, and
/// the last slot is the null slot every view index at or past it reads.
const UNCAPPED_VIEWS: u32 = 1 << 21;

/// Where the raw twins of the title's slots start in a heap that has them
/// ([`raw_twins`]): past every slot a title's heap and the dynamic ones take.
const RAW_TWINS: u32 = 1 << 21;

/// Whether view heaps keep a raw twin of every buffer view of the title's
/// slots, which shaders read that pick buffers by an index they compute
/// (`victory_shader::emit::raw_views` says why; `VICTORY_RAW_VIEWS=0`: they
/// read the title's slots themselves, as before). Needs a host that takes
/// heaps of 2^22 views.
fn raw_twins() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAW_VIEWS").is_none_or(|v| v != "0"))
}

/// Picks, once, the null slot at the end of every host view heap and tells
/// the translator now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. Called before a shader is translated and before a
/// view heap is made.
///
/// Every host clamps. A host that caps view heaps at [`PORTABLE_VIEWS`]
/// (NVIDIA) hung on a view index past the heap; one that does not (AMD
/// RDNA3, RX 7900 XT) took a page fault and removed the device on entering
/// the world: a compute program read view slot 16,777,088 (0xffff80) of a
/// heap of 1,507,328, past its end. Unclamped, that index is undefined on
/// any host.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        let fits = |count: u32| {
            // SAFETY: plain heap creation; the heap is released at once.
            let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            made.is_ok()
        };
        let twins = raw_twins() && fits(2 * RAW_TWINS);
        if twins {
            victory_shader::emit::set_raw_views(RAW_TWINS);
        }
        // (The twins lie past the null slot.)
        let views = if twins {
            RAW_TWINS
        } else if fits(UNCAPPED_VIEWS) {
            UNCAPPED_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            raw_twins = twins,
            "host view heaps hold {views} views; view indices are clamped to the null slot at the end from the first shader on"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

/// [`DYNAMIC`] for a heap of `count` title slots whose null slot is `null`
/// (0: none): up to four times as many where the host heap has the room (a
/// host that takes more than [`PORTABLE_VIEWS`]; `VICTORY_DYNAMIC_SLOTS`
/// sets the most). 2^18 ran out in the title's bigger places (session
/// play-1125-0127: 27,000 to 35,000 tables a frame, up to 3,285 of them
/// refused, with 2,639 indirect draws and dispatches).
fn dynamic_slots(count: u32, null: u32) -> u32 {
    static MOST: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| {
        std::env::var("VICTORY_DYNAMIC_SLOTS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DYNAMIC << 2)
    });
    if null < PORTABLE_VIEWS {
        return DYNAMIC;
    }
    null.saturating_sub(count).min(most).max(DYNAMIC)
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // A view heap ends at the null slot shaders clamp view indices to
        // (`probe_view_limit`), whatever the title's own count: the title's
        // slots, then the dynamic ones, then null up to and with that slot.
        if !sampler {
            probe_view_limit();
        }
        let null = if sampler {
            0
        } else {
            victory_shader::emit::view_heap_limit()
        };
        let raw = if sampler { 0 } else { victory_shader::emit::raw_views() };
        let extra = if sampler { 0 } else { dynamic_slots(count, null) };
        let (heap, count, size) = if null == 0 {
            match make(count + extra) {
                Ok(heap) => (heap, count, count + extra),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            let size = if raw == 0 { null + 1 } else { 2 * raw };
            let fitted = if count + extra <= null {
                count
            } else {
                // shortcut: descriptors past the host's limit are left out; a title that reaches
                // them needs remapping. They read the null slot instead.
                tracing::warn!(count, "host view heap capped at {size}; views past it read a null slot");
                null - extra
            };
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        let h = HostHeap {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            buffer_sweep: Mutex::default(),
            buffer_frame: std::sync::atomic::AtomicU64::new(u64::MAX),
            heap,
            title_start: std::sync::atomic::AtomicU64::new(0),
            cpu,
            gpu,
            step,
            count,
            dynamic: extra,
            sampler,
            raw,
            twins: SparseBits::new(if raw == 0 { 0 } else { (count + extra) as usize }),
            #[cfg(feature = "diagnostics")]
            written: (0..if sampler || !crate::book::on() { 0 } else { size })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            made: Made::new(count as usize),
            quick: SparseU64::new(count as usize),
            copied: Mutex::new(Vec::new()),
            waiting: Mutex::default(),
            unmakes: Mutex::default(),
            away: Mutex::default(),
            kept_ids: SparseU32::new(count as usize),
            kept_locks: [const { Mutex::new(()) }; 64],
            resources: Mutex::new(ResourceTable::new()),
            window_slots: Mutex::default(),
            window_slot_bits: SparseBits::new(if sampler { 0 } else { count as usize }),
            dynamic_kept: DynamicKept::new(extra as usize),
            next: Mutex::new(DynamicPool::new(extra)),
            free: std::sync::atomic::AtomicU32::new(extra),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            stride_cache: if sampler {
                SparseU64::new(0)
            } else {
                let cap = (count + extra).min(victory_shader::emit::SLOT_STRIDES_COUNT) as usize;
                SparseU64::new(cap)
            },
            raw_pages: Mutex::new(None),
            buffers: Slots::new(count as usize),
            changed: Slots::new(count as usize),
            textures: Mutex::default(),
            queued_textures: Slots::new(if sampler { 0 } else { count as usize }),
            null_kind: SparseU8::new(if sampler { 0 } else { (count + extra) as usize }, 1),
            tabled: SparseU64::new(if sampler { 0 } else { count as usize }),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            spanned: Slots::new((count + extra) as usize),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            // (The twins are null buffer views.)
            let ends = if self.raw == 0 { [total, total] } else { [self.raw.min(total), total] };
            let mut done = 0;
            for (twin, end) in ends.into_iter().enumerate() {
                if done == end {
                    continue;
                }
                if self.sampler {
                    null_sampler(start);
                } else if twin == 1 {
                    null_buffer_view(start);
                } else {
                    null_view(start);
                }
                let mut filled = 1;
                while filled < BLOCK {
                    let n = filled.min(BLOCK - filled);
                    host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                    filled += n;
                }
                while done < end {
                    let n = BLOCK.min(end - done);
                    host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                    done += n;
                }
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.at(index));
        self.put(index);
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        self.dynamic_chunk(n, n, None).map(|(first, _)| first)
    }

    /// Allocates at least `min_n` (and up to `want_n` when plenty of headroom
    /// remains) consecutive free dynamic slots, optionally releasing `recycle`
    /// `(first, count)` under the same lock first. Returns `(first, granted)`.
    pub(crate) fn dynamic_chunk(
        &self,
        min_n: u32,
        want_n: u32,
        recycle: Option<(u32, u32)>,
    ) -> Option<(u32, u32)> {
        if self.sampler || min_n == 0 || min_n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut pool = self.next.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((old_first, old_n)) = recycle
            && old_n > 0
            && let Some(at) = old_first.checked_sub(self.count)
        {
            let given = pool.mark_free(at, old_n);
            if given > 0 {
                self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
                self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
                pool.rewind_if_tail(at, old_n);
            }
        }
        let free = self.free.load(std::sync::atomic::Ordering::Relaxed);
        let too_long = self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        const SLAB_HEADROOM: u32 = 8192;
        let try_want = want_n > min_n
            && want_n <= self.dynamic
            && free >= want_n.saturating_add(SLAB_HEADROOM)
            && want_n < too_long;
        let mut granted = min_n;
        let mut found = if try_want {
            pool.find(want_n).map(|first| {
                granted = want_n;
                first
            })
        } else {
            None
        };
        let enough = free >= min_n && min_n < too_long;
        if found.is_none() && enough {
            found = pool.find(min_n);
            granted = min_n;
        }
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(min_n, std::sync::atomic::Ordering::Relaxed);
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n = min_n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        pool.mark_taken(first, granted);
        self.free.fetch_sub(granted, std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        Some((self.count + first, granted))
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// This heap's identity ([`NEXT_ID`]): dynamic slots are given back only
    /// to it, never to a heap made later at the same address.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Lock-free check whether title slot `index` is currently queued in `textures`.
    #[inline]
    pub(crate) fn is_texture_queued(&self, index: u32) -> bool {
        !self.sampler && self.queued_textures.contains(index)
    }

    /// Fast atomic check whether any deferred list-flip slots are currently pending.
    #[inline]
    pub(crate) fn has_pending_list_flips(&self) -> bool {
        !self.sampler && PENDING_LIST_COUNT.load(std::sync::atomic::Ordering::Acquire) != 0
    }

    /// Returns `(index, write)` for every slot in `wanted` currently queued in
    /// `textures` (regardless of future `due` frame), for scoped decoration bakes.
    pub(crate) fn texture_slots_for_bake(&self, wanted: &[u32]) -> Vec<(u32, u64)> {
        if self.sampler || wanted.is_empty() {
            return Vec::new();
        }
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).for_slots(wanted)
    }

    /// Indexes slot `index` by its missing-mip VA range `[start, end)` so
    /// `remapped`/`page_written` wakes it in 1 frame (`ARCH-08`).
    pub(crate) fn texture_slot_missing_mip(&self, index: u32, start: u64, end: u64) {
        if self.sampler || start == 0 || end <= start {
            return;
        }
        SPANNED[0].fetch_min(start, std::sync::atomic::Ordering::Relaxed);
        SPANNED[1].fetch_max(end, std::sync::atomic::Ordering::Relaxed);
        self.placed.lock().unwrap_or_else(|e| e.into_inner()).note_missing_mip(index, start, end);
    }

    /// Removes slot `index` from the missing-mip VA index once fully uploaded.
    pub(crate) fn texture_slot_clear_missing_mip(&self, index: u32) {
        if self.sampler {
            return;
        }
        self.placed.lock().unwrap_or_else(|e| e.into_inner()).clear_missing_mip(index);
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, textures.census(frame))
    }

    /// A number that changes when a slot is written or made due anew, and
    /// whether memory was noted since a pass last took the notes.
    pub(crate) fn texture_news(&self) -> (u64, bool) {
        let writes = self.textures.lock().unwrap_or_else(|e| e.into_inner()).writes;
        let noted = |ranges: &Mutex<Vec<(u64, u64)>>| !ranges.lock().unwrap_or_else(|e| e.into_inner()).is_empty();
        (writes, noted(&REMAPPED) || noted(&PAGE_WRITTEN) || noted(&BAKE_PAGE_WRITTEN))
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => {
                textures.done(index, write);
                if !textures.slots.contains_key(&index) {
                    self.queued_textures.remove(index);
                }
            }
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.recheck(index, write, due, at, print, looks, uploaded);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.textures.lock().unwrap_or_else(|e| e.into_inner()).mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        self.texture_slots_page_written(frame);
        let mut ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        join(&mut ranges);
        let (mut slots, mut missing): (Vec<u32>, Vec<u32>) = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let slots = ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect();
            let missing = ranges.iter().flat_map(|&(start, end)| placed.waiting_slots_in(start, end)).collect();
            (slots, missing)
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        missing.sort_unstable();
        missing.dedup();
        missing.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
            self.queued_textures.put(slot);
        }
        for &slot in &missing {
            if !textures.wake_missing(slot, frame + 1) {
                textures.changed(slot, frame, frame + 1);
            }
            self.queued_textures.put(slot);
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// Makes the slots of the textures filled with less than was mapped
    /// ([`read_short`]) due from the next frame on. Returns how many.
    pub(crate) fn texture_slots_short(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts = std::mem::take(&mut SHORT.lock().unwrap_or_else(|e| e.into_inner()).0);
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
            self.queued_textures.put(slot);
        }
        slots.len()
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and finds the slots whose
    /// placed texture holds bytes of them ([`Spans::slots_holding`]). A
    /// slot whose texture was uploaded before the write shows the bytes
    /// from before: with `VICTORY_PAGE_WRITES` (`crate::watch::page_writes`,
    /// bit 2 for DirectStorage page writes or bit 8 for plain-read
    /// completions) it is due again [`PAGE_WRITE_AFTER`] frames on, as memory
    /// mapped under it would make it ([`HostHeap::texture_slots_remapped`]);
    /// its host copy exists, so the upload needs no room on the card. A
    /// slot still waiting for its first upload is left as it is: that
    /// upload takes the bytes that are there by then. Slots tables read
    /// are left to their draws. Counted either way; returns how many slots
    /// it made due.
    fn texture_slots_page_written(&self, frame: u64) -> usize {
        let mut ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        join(&mut ranges);
        // Always wake waiting/partial textures whose missing-mip VA arrived via
        // DirectStorage/page writes in 1 frame (kills 30f/256f backoff without
        // re-uploading already-complete textures).
        let mut missing: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.waiting_slots_in(start, end)).collect()
        };
        if !missing.is_empty() {
            missing.sort_unstable();
            missing.dedup();
            missing.retain(|&slot| !self.tabled_lately(slot, frame));
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            for &slot in &missing {
                if !textures.wake_missing(slot, frame + 1) {
                    textures.changed(slot, frame, frame + 1);
                }
                self.queued_textures.put(slot);
            }
        }
        let due = crate::watch::page_writes() & (2 | 8) != 0;
        let made_due = self.slots_written_under(&ranges, frame, due)[2];
        if !due {
            let mut kept = BAKE_PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
            kept.extend(ranges);
            if kept.len() > 256 {
                join(&mut kept);
                const MOST: usize = 1 << 14;
                if kept.len() > MOST {
                    let drop_n = kept.len() - MOST;
                    kept.drain(..drop_n);
                }
            }
        }
        made_due
    }

    /// Wakes uploaded placed-texture slots touched by recent physical-page
    /// writes (DirectStorage completions) immediately (`due = frame`) right
    /// before a one-shot ped/decoration bake draw.
    pub(crate) fn texture_slots_bake_written(&self, frame: u64) -> usize {
        let mut ranges = std::mem::take(&mut *BAKE_PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        join(&mut ranges);
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        if slots.is_empty() {
            return 0;
        }
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        for &slot in &slots {
            textures.changed(slot, frame, frame);
            self.queued_textures.put(slot);
        }
        slots.len()
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end):
    /// the slots that held bytes of them, those of them uploaded before,
    /// and those made due (the latter, if `due`).
    fn slots_written_under(&self, ranges: &[(u64, u64)], frame: u64, due: bool) -> [usize; 3] {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let held = slots.len();
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        // Uploaded before: out of the queue, or in it after an upload (in
        // part, or whole and being looked at again).
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        PAGE_WRITES[2].fetch_add(held as u64, Relaxed);
        PAGE_WRITES[3].fetch_add(slots.len() as u64, Relaxed);
        if !due {
            return [held, slots.len(), 0];
        }
        for &slot in &slots {
            textures.changed(slot, frame, frame + PAGE_WRITE_AFTER);
            self.queued_textures.put(slot);
        }
        [held, slots.len(), slots.len()]
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full.
        if key == 0 || self.made.get(index) != Some(key) {
            return false;
        }
        let _made = self.made.lock(index);
        self.made.get(index) == Some(key) && self.kept_in(index).is_some()
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
            self.queued_textures.put(slot);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        if self.sampler {
            return 0;
        }
        {
            let mut after = RESHOWN_AFTER.lock().unwrap_or_else(|e| e.into_inner());
            if !after.is_empty() {
                let finished = crate::host::submissions().1;
                let mut now = RESHOWN.lock().unwrap_or_else(|e| e.into_inner());
                after.retain(|&(at, start)| {
                    if at <= finished {
                        now.push(start);
                    }
                    at > finished
                });
            }
        }
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        let mut remade = 0;
        for slot in slots {
            let _made = self.made.lock(slot);
            let Some(key) = self.made.get(slot).filter(|&key| key != 0) else {
                continue;
            };
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.at(slot)) {
                self.put(slot);
                self.keep(slot, kept);
                remade += 1;
            }
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    ///
    /// Not at the first look: a title that moves a texture in its streaming
    /// memory unmaps it before it maps it again, and what was shown went
    /// black for the pictures in between (session play-1125-0127: faces,
    /// books, posters and sleeves for 0.3 to 2 s). Only when it is still
    /// away `AWAY_MS` after it was first found away
    /// (`VICTORY_TEXTURE_AWAY_MS`; 0: at once, as before).
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        const AWAY_MS: u64 = 1000;
        static WAIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let wait = *WAIT.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_AWAY_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(AWAY_MS)
        });
        let _made = self.made.lock(index);
        let Some(key) = self.made.get(index).filter(|&key| key != 0) else {
            return false;
        };
        let now = crate::host::now_ms();
        {
            let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
            match away.get(&index) {
                Some(&(was, since)) if was == key && now.saturating_sub(since) < wait => return false,
                Some(&(was, _)) if was == key => {
                    away.remove(&index);
                }
                _ if wait == 0 => {}
                _ => {
                    away.insert(index, (key, now));
                    return false;
                }
            }
        }
        self.unmake(index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// Whether host slot `index` shows the placed texture the title's record
    /// at `handle` views, by an earlier view of it.
    pub(crate) fn texture_slot_shows_other(&self, index: u32, handle: u64) -> bool {
        let key = record_key(handle);
        let Some(shown) = self.made.get(index) else {
            return false;
        };
        if key == 0 || shown == 0 || shown == key {
            return false;
        }
        let view = |key: u64| view_at(key as u32, (key >> 32) as u32);
        matches!((view(shown), view(key)), (Some(a), Some(b)) if same_placed_texture(&a, &b))
    }

    /// Slot `index` is made from the record at `handle` (what it is now)
    /// when `list`, which holds the upload of what that view shows, has run:
    /// until then it shows what it showed. (Made at once, the frames on the
    /// card read the new view before its levels were filled.)
    pub(crate) fn texture_slot_flip(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        self.flip(list, index, handle, None);
    }

    /// The same of a slot that shows nothing of the texture the record at
    /// `handle` views, or something else: until then it shows the neutral
    /// texture ([`show_after`]).
    pub(crate) fn texture_slot_show_after(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        if self.show_later(index, handle) {
            self.flip(list, index, handle, Some(crate::frame::current()));
        }
    }

    /// What slot `index` shows until the texture the record at `handle` views
    /// is shown. False: it shows it, or there is none.
    fn show_later(&self, index: u32, handle: u64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        let key = record_key(handle);
        let view = view_at(key as u32, (key >> 32) as u32);
        let (true, Some(view)) = (key != 0, view) else {
            return false;
        };
        let stale = {
            let _made = self.made.lock(index);
            let Some(shown) = self.made.get(index) else {
                return false;
            };
            if shown == key {
                return false;
            }
            // (As `mirror`.)
            let stale = self.note_buffer(index, Some(&view), true);
            let old = view_at(shown as u32, (shown >> 32) as u32);
            let kept = old.is_some_and(|old| match kin(&old, &view) {
                Kin::Same => true,
                kin => crate::resource::room_lately() && kin.kept(keep_shown()),
            });
            if shown == 0 || !kept {
                #[cfg(feature = "diagnostics")]
                note_before(index, shown, "neutral");
                self.quick_is(index, 0);
                self.made.set(index, 0);
                let kept = neutral_for(&view, self.at(index));
                self.put(index);
                self.clear_twin(index);
                LATE[3].fetch_add(u64::from(kept.is_some()), Relaxed);
                self.keep(index, kept);
            }
            stale
        };
        self.unmake_stale(index, stale);
        true
    }

    /// [`HostHeap::texture_slot_show_after`] of a texture whose upload `job`
    /// makes: shown when that has run.
    pub(crate) fn texture_slot_show_ahead(&self, job: &crate::texture::ahead::Job, index: u32, handle: u64) {
        if !self.show_later(index, handle) {
            return;
        }
        let (heap, key) = (std::ptr::from_ref(self) as usize, record_key(handle));
        let late = Some(crate::frame::current());
        job.then(move || {
            let flip = Flip { after: 0, heap, index, handle, key, late };
            FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
        });
    }

    /// [`HostHeap::mirror`] of a slot whose table a draw read: a texture made
    /// a moment ago is shown after the draw's list ([`show_after`]; the draw
    /// reads the table's dynamic slots).
    pub(crate) fn mirror_after(&self, list: &crate::host::HostList, index: u32, handle: u64) {
        if show_after()
            && !crate::texture::BAKING.get()
            && index < self.count
            && !self.texture_slot_made_from(index, handle)
            && let Some(view) = super::view(handle)
            && let Some(View::Shader(resource, _)) = placed_texture(&view).or(Some(view))
            && crate::resource::buffer_bytes(resource as crate::object::This).is_none()
            && crate::resource::host_resource(resource as crate::object::This).is_some_and(crate::pool::fresh)
            && !crate::texture::ahead::there(resource as crate::object::This)
        {
            return self.texture_slot_show_after(list, index, handle);
        }
        self.mirror(index, handle, true);
    }

    fn flip(&self, list: &crate::host::HostList, index: u32, handle: u64, late: Option<u64>) {
        use std::sync::atomic::Ordering::{Relaxed, Release};
        let (heap, key) = (std::ptr::from_ref(self) as usize, record_key(handle));
        let id = NEXT_PENDING_FLIP.fetch_add(1, Relaxed);
        let origin = (std::ptr::from_ref(list) as usize, list.epoch());
        {
            let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
            pending.push(PendingListFlip {
                id,
                origin,
                flip: Flip { after: 0, heap, index, handle, key, late },
            });
            PENDING_LIST_COUNT.store(pending.len(), Release);
        }
        struct PendingGuard(u64);
        impl PendingGuard {
            fn id(&self) -> u64 {
                self.0
            }
        }
        impl Drop for PendingGuard {
            fn drop(&mut self) {
                remove_pending_list_flip(self.0);
            }
        }
        let guard = PendingGuard(id);
        list.defer_host(move || {
            // Call a `&self` method so Rust 2021+ moves the whole `guard`
            // into this closure rather than copying `guard.0` and dropping
            // `guard` immediately at the end of `flip()`.
            let Some(mut flip) = take_pending_list_flip(guard.id()) else {
                return;
            };
            flip.after = crate::host::submissions().0;
            FLIPS.lock().unwrap_or_else(|e| e.into_inner()).push(flip);
        });
    }

    /// Drains deferred flips for this heap whose slot index is in `wanted`
    /// (leaving unrelated world-texture pending flips untouched).
    pub(crate) fn take_pending_list_flips_for(&self, wanted: &[u32]) -> Vec<PendingListFlip> {
        if wanted.is_empty() || PENDING_LIST_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let own = std::ptr::from_ref(self) as usize;
        let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
        let (mine, rest): (Vec<PendingListFlip>, Vec<PendingListFlip>) = pending
            .drain(..)
            .partition(|p| p.flip.heap == own && wanted.binary_search(&p.flip.index).is_ok());
        *pending = rest;
        PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
        mine
    }

    /// Drains deferred flips still sitting in unsubmitted command lists for
    /// this heap so a decoration bake can mirror them out of `neutral()`.
    /// Entries that cannot be promoted yet (unmapped memory or deferred
    /// staging) must be returned via [`HostHeap::restore_pending_list_flip`].
    pub(crate) fn take_pending_list_flips(&self) -> Vec<PendingListFlip> {
        if PENDING_LIST_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let own = std::ptr::from_ref(self) as usize;
        let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
        let (mine, rest): (Vec<PendingListFlip>, Vec<PendingListFlip>) =
            pending.drain(..).partition(|p| p.flip.heap == own);
        *pending = rest;
        PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
        mine
    }

    /// Re-queues an unpromoted deferred flip drained by
    /// [`HostHeap::take_pending_list_flips`] so its originating command list
    /// still flips it out of `neutral()` once submitted.
    pub(crate) fn restore_pending_list_flip(&self, item: PendingListFlip) {
        let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
        pending.push(item);
        PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
    }

    /// Whether title descriptor slot `handle` still holds record `key`.
    pub(crate) fn slot_still_at(&self, handle: u64, key: u64) -> bool {
        key != 0 && record_key(handle) == key
    }

    /// Looks up `(view, key)` for slot `index` prior to a synchronous bake
    /// upload without replacing its current host descriptor with `neutral()`;
    /// [`HostHeap::publish_uploaded_slot`] exposes the real descriptor and
    /// updates `Spans`/stride tracking once the GPU copy finishes.
    pub(crate) fn prepare_bake_slot(&self, index: u32, handle: u64) -> Option<(View, u64)> {
        if index >= self.count {
            return None;
        }
        let key = record_key(handle);
        let view = (key != 0).then(|| view_at(key as u32, (key >> 32) as u32)).flatten()?;
        Some((view, key))
    }

    /// Exposes the real host descriptor for slot `index` after its synchronous
    /// bake upload has completed on the GPU, updating `Spans`/stride tracking
    /// without re-queuing the slot in `TextureQueue`.
    pub(crate) fn publish_uploaded_slot(&self, index: u32, handle: u64, key: u64) -> bool {
        if index >= self.count || key == 0 {
            return false;
        }
        let (published, stale) = {
            let _made = self.made.lock(index);
            if self.made.get(index) == Some(key) {
                return false;
            }
            if record_key(handle) != key {
                return false;
            }
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                return false;
            };
            let stale = self.note_buffer(index, Some(&view), true);
            if let Some(kept) = make(&view, self.at(index)) {
                self.put(index);
                self.quick_is(index, 0);
                self.clear_twin(index);
                self.made.set(index, key);
                self.keep(index, kept);
                self.quick_is(index, key);
                (true, stale)
            } else {
                (false, stale)
            }
        };
        self.unmake_stale(index, stale);
        published
    }

    /// Makes the slots whose new view's upload has run
    /// ([`HostHeap::texture_slot_flip`]); returns how many.
    pub(crate) fn texture_slots_flip(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let finished = crate::host::submissions().1;
        let ready: Vec<Flip> = {
            let mut flips = FLIPS.lock().unwrap_or_else(|e| e.into_inner());
            if flips.is_empty() {
                return 0;
            }
            let own = std::ptr::from_ref(self) as usize;
            let (ready, wait): (Vec<Flip>, Vec<Flip>) =
                flips.drain(..).partition(|f| f.after <= finished && f.heap == own);
            *flips = wait;
            ready
        };
        let mut flipped = 0;
        let mut late_flipped = 0;
        for f in ready {
            let _made = self.made.lock(f.index);
            let Some(shown) = self.made.get(f.index) else {
                continue;
            };
            if shown == f.key {
                continue;
            }
            // Written again since, or made null: that write is in the queue.
            // (A slot shown late is null until now.)
            let view = view_at(f.key as u32, (f.key >> 32) as u32);
            let there = shown != 0 || f.late.is_some();
            let (true, true, Some(view)) = (f.key != 0 && record_key(f.handle) == f.key, there, view) else {
                FLIPPED[1].fetch_add(1, Relaxed);
                LATE[4].fetch_add(u64::from(f.late.is_some()), Relaxed);
                continue;
            };
            // Its memory went meanwhile: the texture pass finds that.
            if f.late.is_some()
                && let View::Placed(_, resource, desc) = &view
            {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                if !crate::texture::any_mapped(address as usize, len as usize) {
                    LATE[4].fetch_add(1, Relaxed);
                    continue;
                }
            }
            if let Some(kept) = make(&view, self.at(f.index)) {
                #[cfg(feature = "diagnostics")]
                note_before(f.index, shown, if f.late.is_some() { "shown after its upload" } else { "flip" });
                self.put(f.index);
                #[cfg(feature = "diagnostics")]
                let was = shown;
                self.quick_is(f.index, 0);
                self.made.set(f.index, f.key);
                self.keep(f.index, kept);
                self.quick_is(f.index, f.key);
                #[cfg(feature = "diagnostics")]
                if f.late.is_some() && was != 0 {
                    same_bytes(was, f.key);
                }
                if let Some(since) = f.late {
                    late_flipped += 1;
                    let frames = crate::frame::current().saturating_sub(since);
                    let n = LATE[0].fetch_add(1, Relaxed) + 1;
                    let all = LATE[1].fetch_add(frames, Relaxed) + frames;
                    let most = LATE[2].fetch_max(frames, Relaxed).max(frames);
                    if n == 1 || n.is_multiple_of(5000) {
                        tracing::info!(
                            "slots shown after their texture's upload (VICTORY_SHOW_AFTER=1), since the start: {n}, {:.2} frames after they were asked for, the longest {most}; the neutral texture meanwhile in {}; {} not shown any more by then",
                            all as f64 / n as f64,
                            LATE[3].load(Relaxed),
                            LATE[4].load(Relaxed)
                        );
                    }
                } else {
                    flipped += 1;
                }
            }
        }
        FLIPPED[0].fetch_add(flipped as u64, Relaxed);
        flipped + late_flipped
    }

    /// The memory of the texture title slot `index` views is there (again):
    /// see [`HostHeap::texture_slot_gone`].
    pub(crate) fn texture_slot_here(&self, index: u32) {
        let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
        if !away.is_empty() {
            away.remove(&index);
        }
    }

    /// Makes host slot `index` null again, as it is before its view is
    /// made: a null view of the kind it held. Under its lock ([`Made`]).
    fn unmake(&self, index: u32) {
        self.quick_is(index, 0);
        let Some(key) = self.made.get(index) else {
            return;
        };
        self.made.set(index, 0);
        if key == 0 {
            return;
        }
        let view = view_at(key as u32, (key >> 32) as u32);
        let put_done = match &view {
            Some(view) => {
                if let Some(kind) = cached_null_kind(view)
                    && self.copy_cached_null(index, kind)
                {
                    true
                } else {
                    null_for(view, self.at(index));
                    false
                }
            }
            None => self.copy_cached_null(index, 0) || {
                null_view(self.at(index));
                false
            },
        };
        if !put_done {
            self.put(index);
        }
        if let Some(view) = &view {
            self.clear_or_null_tlas_twin(index, view);
        } else {
            self.clear_twin(index);
        }
        self.write_stride(index, 0, 0);
        self.keep(index, None);
    }

    /// The slots [`HostHeap::note_buffer`] found stale for slot `index`, with
    /// what they were made from then, become null. None of their locks is
    /// held here; one made from another record meanwhile is left as it is.
    fn unmake_stale(&self, index: u32, stale: Vec<(u32, u64)>) {
        for (slot, key) in stale {
            if slot == index || key == 0 {
                continue;
            }
            let _made = self.made.lock(slot);
            if self.made.get(slot) == Some(key) {
                self.unmake(slot);
            }
        }
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(old) = self.tabled.get(index as usize, Relaxed) else {
            return;
        };
        let now = tabled_now(frame);
        if !read {
            if old & !TABLED_READ != now {
                self.tabled.set(index as usize, now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            self.tabled.set(index as usize, now | TABLED_READ, Relaxed);
            if tabled_out() && self.queued_textures.contains(index) {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                if textures.slots.remove(&index).is_some() {
                    self.queued_textures.remove(index);
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = tabled_now(frame);
        tabled_out() && stamp != 0 && now.saturating_sub(stamp) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u64 {
        self.tabled
            .get(index as usize, std::sync::atomic::Ordering::Relaxed)
            .unwrap_or(0)
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`]; `same_texture`, whether
    /// the write is a `Kin::Same` mip upgrade of the texture already shown
    /// (zero table grace so resolution upgrades aren't delayed).
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, same_texture: bool, frame: u64) {
        if PUBLISHING.get() {
            return;
        }
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = tabled_now(frame);
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw (except `Kin::Same` mip
        // upgrades of an existing texture, which need zero grace).
        let lately = stamp != 0 && now.saturating_sub(stamp & !TABLED_READ) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        if texture && !read_now {
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            let due = if lately && !seen && !same_texture {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            textures.written(index, frame, due, seen);
            self.queued_textures.put(index);
        } else if self.queued_textures.contains(index) {
            let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
            textures.remove(index);
            self.queued_textures.remove(index);
        }
    }

    /// Title slot `index` lies in a table a draw reads or had made afresh:
    /// what it holds is the draw's to make ([`HostHeap::copied_tabled`]).
    pub(crate) fn slot_tabled(&self, index: u32) {
        self.tabled_in(index, false, crate::frame::current());
    }

    /// Whether slot `index` still mirrors `key` on the lock-free quick table.
    #[inline]
    pub(crate) fn quick_matches(&self, index: u32, key: u64) -> bool {
        key != 0
            && quick_mirror()
            && self.quick.get(index as usize, std::sync::atomic::Ordering::Acquire) == Some(key)
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        self.buffers.all(|_| true)
    }

    /// Of [`HostHeap::buffer_slots`], those of one of `parts` parts: every
    /// slot is in the same part each time.
    pub(crate) fn buffer_slots_of(&self, part: u32, parts: u32) -> Vec<u32> {
        self.buffers.all(|slot| slot % parts == part)
    }

    /// Removes title slot `index` from the tracked heap-buffer set (e.g. once
    /// confirmed tabled or without a backing buffer copy).
    #[inline]
    pub(crate) fn untrack_buffer(&self, index: u32) {
        self.buffers.remove(index);
    }

    /// Lock-free O(1) check for whether any buffer view slot changed since the
    /// last [`HostHeap::buffer_changes`] call.
    #[inline]
    pub(crate) fn has_buffer_changes(&self) -> bool {
        self.changed.any()
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        self.changed.take()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: its element stride in the table, and whether
    /// it is a buffer view at all.
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held.
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]), each with the record its host slot is made from:
    /// out of the queue of textures already, their host slots are the
    /// caller's to make null when it has let go of its own slot's lock
    /// ([`HostHeap::unmake_stale`]).
    fn note_buffer(&self, index: u32, view: Option<&View>, seen: bool) -> Vec<(u32, u64)> {
        self.note_buffer_inner(index, view, seen, true, false)
    }

    fn note_buffer_inner(
        &self,
        index: u32,
        view: Option<&View>,
        seen: bool,
        record_change: bool,
        same_texture: bool,
    ) -> Vec<(u32, u64)> {
        if self.sampler {
            return Vec::new();
        }
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, desc))
                if !(*r == 0
                    && desc.is_some_and(|d| d[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32)
                    && crate::dxr::enabled())
                    && crate::resource::buffer_bytes(*r as crate::object::This).is_none() =>
            {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        self.queue_texture(index, texture, seen, same_texture, crate::frame::current());
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, crate::frame::current()) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                let at = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                // Guard against null-address placed descriptors (`at == 0`), which
                // otherwise clash in `Spans::by_start` and trigger 124k+ fake stale storms.
                if at == 0 {
                    None
                } else {
                    let held = crate::resource::placed_bytes_from(resource, crate::resource::first_mip(desc));
                    Some((resource, at, desc[4], held.map_or(u64::MAX, |held| at.saturating_add(held))))
                }
            }
            _ => None,
        };
        // (Under the slot's lock, as every write of it: whether it is in
        // `spanned` changes only here.)
        let in_spans = |index: u32| {
            let bit = 1u64 << (index % 64);
            self.spanned.bits.get(index as usize / 64).is_none_or(|w| w.load(std::sync::atomic::Ordering::Acquire) & bit != 0)
        };
        let stale = if placed_view.is_none() && !in_spans(index) && spans_quick() {
            SPANS_QUICK[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Vec::new()
        } else {
            use std::sync::atomic::Ordering::Relaxed;
            SPANS_QUICK[1].fetch_add(1, Relaxed);
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            let span = placed_view.and_then(|(resource, start, first, held)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first, held))
            });
            match span {
                Some((start, end, first, held)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    self.spanned.put(index);
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first, held);
                    // (Under the lock of `placed`: a later write of one of
                    // them comes by here before it makes the slot.)
                    stale.into_iter().map(|slot| (slot, self.made.get(slot).unwrap_or(0))).collect()
                }
                None => {
                    placed.forget(index);
                    self.spanned.remove(index);
                    Vec::new()
                }
            }
        };
        if !stale.is_empty() {
            STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
            if stale.iter().any(|&(slot, _)| self.queued_textures.contains(slot)) {
                let mut textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
                for &(slot, _) in &stale {
                    if self.queued_textures.contains(slot) {
                        textures.remove(slot);
                        self.queued_textures.remove(slot);
                    }
                }
            }
        }
        let (stride, kind) = view.map_or((None, 0), view_stride_and_kind);
        self.write_stride(index, stride.unwrap_or(0) | kind, view.map_or(0, view_count));
        let tabled = self.copied_tabled(index, crate::frame::current());
        let was = if stride.is_some() && !tabled {
            self.buffers.put(index);
            true
        } else {
            self.buffers.remove(index)
        };
        if was && record_change && !tabled {
            self.changed.put(index);
        }
        stale
    }

    /// Writes `word` (stride | format kind) and `count` (element/byte count) for
    /// view-heap slot `index` into the host's [`victory_shader::emit::SLOT_STRIDES`] table,
    /// skipping unchanged scattered WC stores via `stride_cache` (`CPU-30`).
    #[inline]
    fn write_stride(&self, index: u32, word: u32, count: u32) {
        let cap = victory_shader::emit::SLOT_STRIDES_COUNT;
        if let Some((_, _, data)) = &self.strides
            && index < cap
        {
            let packed = u64::from(word) | (u64::from(count) << 32);
            if let Some(cached) = self.stride_cache.get(index as usize, std::sync::atomic::Ordering::Relaxed) {
                if cached == packed {
                    return;
                }
                self.stride_cache.set(index as usize, packed, std::sync::atomic::Ordering::Relaxed);
            }
            // SAFETY: inside the `2 * SLOT_STRIDES_COUNT` dword mapping; shaders
            // read aligned dwords concurrently while we write.
            unsafe {
                let base = *data as *mut u32;
                base.add(index as usize).write_volatile(word);
                base.add((cap + index) as usize).write_volatile(count);
            }
        }
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned), if they were taken by this heap: `id` is [`HostHeap::id`]'s.
    /// The title can release the heap a list's slots came from, and a later
    /// heap can be made at the same address; the list's reset then names the
    /// old heap, and must not free the new one's slots ([`NEXT_ID`]).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, id: u64, first: u32, n: u32) {
        if id != self.id || n == 0 {
            return;
        }
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut pool = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let given = pool.mark_free(at, n);
        if given > 0 {
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Gives back a slice of `(heap, id, first, n)` ranges belonging to this
    /// heap under one lock acquisition.
    pub(crate) fn release_batch(&self, id: u64, ranges: &[(usize, u64, u32, u32)]) {
        if id != self.id || ranges.is_empty() {
            return;
        }
        let mut pool = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let mut given = 0u32;
        for &(_, rid, first, n) in ranges {
            if rid != self.id || n == 0 {
                continue;
            }
            if let Some(at) = first.checked_sub(self.count) {
                given += pool.mark_free(at, n);
            }
        }
        if given > 0 {
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let prev_slot = CURRENT_SLOT.replace(Some(index));
        let made = make_twinned_info(view, self.at(index), self.twin(index));
        CURRENT_SLOT.set(prev_slot);
        let mut put_done = false;
        let (kept, same_twin, is_window) = match made {
            Some((kept, same_twin, is_window)) => (Some(kept), same_twin, is_window),
            None => (None, false, false),
        };
        if kept.is_none() {
            if let Some(kind) = cached_null_kind(view)
                && self.copy_cached_null(index, kind)
            {
                put_done = true;
            } else {
                null_for(view, self.at(index));
            }
        }
        if !put_done {
            self.put(index);
        }
        if kept.is_some() && twinned(view) {
            if same_twin {
                self.twin_made_same(index);
            } else {
                self.twin_made(index, true);
            }
        } else {
            self.clear_or_null_tlas_twin(index, view);
        }
        let (word, count) = if kept.is_some() {
            (view_stride_word(view), view_count(view))
        } else {
            (0, 0)
        };
        self.write_stride(index, word, count);
        self.keep_window(index, kept.flatten(), is_window);
    }

    /// Single-slot wrapper around [`HostHeap::make_listed_cached`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn make_listed(&self, _recording: u64, words: [u32; 3], index: u32) -> Option<Listed> {
        let mut batch = DescriptorCopyBatch::new();
        let (kind, texture, _) = self.make_listed_cached(words, index, &mut batch, |_| false)?;
        Some(Listed { kind, texture, ready: (0, 0, false), tabled: None })
    }

    /// Copies the view named by `words` from this thread's persistent CPU
    /// descriptor cache ([`ListViews`], split into independent texture and
    /// buffer rings invalidated by [`texture_view_cache_epoch`] and
    /// [`buffer_view_cache_epoch`]) into dynamic slot `index`, creating it on a
    /// miss and queueing its shader-visible copy(ies) into `batch`.
    ///
    /// Performs a single TLS lookup and clones `View` only when `need_view`
    /// returns `true` (e.g. per-list readiness miss).
    pub(crate) fn make_listed_cached(
        &self,
        words: [u32; 3],
        index: u32,
        batch: &mut DescriptorCopyBatch,
        need_view: impl FnOnce(Option<victory_shader::native::ResourceType>) -> bool,
    ) -> Option<(Option<victory_shader::native::ResourceType>, bool, Option<View>)> {
        if self.sampler {
            return None;
        }
        let tex_epoch = texture_view_cache_epoch();
        let buf_epoch = buffer_view_cache_epoch();
        LIST_VIEWS.with(|views| {
            let mut views = views.borrow_mut();
            if views.is_none() {
                *views = Some(ListViews::new()?);
            }
            let views = views.as_mut()?;
            if views.tex_epoch != tex_epoch {
                batch.flush();
                views.textures.clear();
                views.tex_epoch = tex_epoch;
            }
            if views.buf_epoch != buf_epoch {
                batch.flush();
                views.buffers.clear();
                views.buf_epoch = buf_epoch;
            }
            let twins = (index as usize) < self.twins.len();
            let (start, step) = (views.start, views.step);
            let slot_of = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + i as usize * step };
            let made: &ListView = if let Some(m) = views.buffers.get(&words) {
                m
            } else if let Some(m) = views.textures.get(&words) {
                m
            } else {
                let view = super::view_at(words[1], words[2])?;
                // (A constant buffer's bytes are copied for each draw.)
                if matches!(view, View::Constant(..)) {
                    return None;
                }
                let texture = is_texture_view(&view);
                let (map, base_slot, cap_slots) = if texture {
                    (&mut views.textures, 0u32, ListViews::TEX_SLOTS)
                } else {
                    (&mut views.buffers, ListViews::TEX_SLOTS, ListViews::BUF_SLOTS)
                };
                if 2 * map.len() as u32 >= cap_slots {
                    // Flush queued copies referencing old CPU slots before wrapping the ring.
                    batch.flush();
                    map.clear();
                }
                let next_at = base_slot + 2 * map.len() as u32;
                let twin = twinned(&view);
                let prev_slot = CURRENT_SLOT.replace(Some(index));
                let made_info = make_twinned_info(&view, slot_of(next_at), twin.then(|| slot_of(next_at + 1)));
                CURRENT_SLOT.set(prev_slot);
                let (kept, twin_same, _is_window) = made_info?;
                let stride_word = view_stride_word(&view);
                let count = view_count(&view);
                let kind = crate::draw::view_kind(&view);
                map.entry(words).or_insert(ListView {
                    view,
                    at: next_at,
                    twin,
                    twin_same,
                    stride_word,
                    count,
                    kept,
                    kind,
                    texture,
                })
            };
            self.queue_copy_from(index, slot_of(made.at), batch);
            if made.twin && twins {
                let twin_src = if made.twin_same { slot_of(made.at) } else { slot_of(made.at + 1) };
                self.queue_copy_from(self.raw + index, twin_src, batch);
                self.twins.set(index as usize, true);
            } else {
                self.clear_twin_batched(index, batch);
            }
            self.write_stride(index, made.stride_word, made.count);
            self.keep_ref(index, made.kept.as_ref());
            let view = need_view(made.kind).then(|| made.view.clone());
            Some((made.kind, made.texture, view))
        })
    }

    #[inline]
    fn queue_copy_from(&self, to: u32, from: D3D12_CPU_DESCRIPTOR_HANDLE, batch: &mut DescriptorCopyBatch) {
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        self.null_kind.set(to as usize, 0);
        batch.push(self.cpu(to), from);
    }

    /// Makes slot `index` read nothing, queueing any needed null copy into `batch`.
    pub(crate) fn make_null_batched(&self, index: u32, batch: &mut DescriptorCopyBatch) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            if !self.queue_cached_null(index, 0, batch) {
                null_view(self.at(index));
                self.put(index);
            }
            self.write_stride(index, 0, 0);
        }
        self.clear_twin_batched(index, batch);
        self.keep_ref(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            if !self.copy_cached_null(index, 0) {
                null_view(self.at(index));
                self.put(index);
            }
            self.write_stride(index, 0, 0);
        }
        self.clear_twin(index);
        self.keep_ref(index, None);
    }

    /// Where the raw twin of slot `index` is made.
    fn twin(&self, index: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        (index < self.twins.len() as u32).then(|| staged(self.sampler, 1).unwrap_or_else(|| self.cpu(self.raw + index)))
    }

    /// The raw twin of slot `index` was made by this thread ([`make_twinned`]),
    /// or is a null view if it was not.
    fn twin_made(&self, index: u32, made: bool) {
        if !made {
            return self.clear_twin(index);
        }
        if (index as usize) < self.twins.len() {
            self.put_twin(index);
            self.twins.set(index as usize, true);
        }
    }

    /// The raw twin of slot `index` is bit-identical to the primary raw descriptor
    /// in `staged(0)` (`CPU-23`): copy `staged(0)` directly to `self.raw + index`
    /// without a duplicate `CreateSRV`/`CreateUAV`.
    fn twin_made_same(&self, index: u32) {
        if (index as usize) < self.twins.len() {
            self.copy_to(self.raw + index, 0);
            self.twins.set(index as usize, true);
        }
    }

    /// Updates the twin slot of `index` when `view` is nulled: for DXR TLAS
    /// views writes an empty-TLAS SRV to `self.raw + index` (`ARCH-13`) so
    /// 2-pass `FrontBackTrack` never reads a buffer SRV on a nulled TLAS slot;
    /// for all other views clears the twin to a null buffer view.
    fn clear_or_null_tlas_twin(&self, index: u32, view: &View) {
        if is_tlas_srv(view)
            && let Some(twin) = self.twin(index)
        {
            null_for(view, twin);
            self.twin_made(index, true);
        } else {
            self.clear_twin(index);
        }
    }

    /// Batched variant of [`HostHeap::clear_or_null_tlas_twin`].
    fn clear_or_null_tlas_twin_batched(
        &self,
        index: u32,
        view: &View,
        tlas_null_slot: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
        batch: &mut DescriptorCopyBatch,
    ) {
        if is_tlas_srv(view)
            && (index as usize) < self.twins.len()
            && let Some(src) = tlas_null_slot
        {
            self.queue_copy_from(self.raw + index, src, batch);
            self.twins.set(index as usize, true);
        } else {
            self.clear_twin_batched(index, batch);
        }
    }

    /// The raw twin of slot `index` is a null view again, queued into `batch`.
    fn clear_twin_batched(&self, index: u32, batch: &mut DescriptorCopyBatch) {
        if self.twins.clear_if_set(index as usize)
            && !self.queue_cached_null(self.raw + index, 1, batch)
        {
            null_buffer_view(self.at(self.raw + index));
            self.put(self.raw + index);
        }
    }

    /// The raw twin of slot `index` is a null view again.
    fn clear_twin(&self, index: u32) {
        if self.twins.clear_if_set(index as usize) {
            if !self.copy_cached_null(self.raw + index, 1) {
                null_buffer_view(self.at(self.raw + index));
                self.put(self.raw + index);
            }
        }
    }

    /// Returns the immutable CPU-only handle for a pre-built null descriptor:
    /// `0` = 2D SRV, `1` = raw-buffer SRV, `2` = CBV, `3..=10` = SRV texture dims
    /// (`1D`, `1DArray`, `2DArray`, `2DMS`, `2DMSArray`, `3D`, `Cube`, `CubeArray`),
    /// `11` = buffer UAV, `12..=16` = texture UAV dims (`1D`, `1DArray`, `2D`, `2DArray`, `3D`).
    fn cached_null_handle(kind: usize) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        const COUNT: usize = 17;
        if kind >= COUNT {
            return None;
        }
        struct CachedNulls {
            _heap: ID3D12DescriptorHeap,
            handles: [usize; COUNT],
        }
        // SAFETY: CPU-only read-only descriptor heap with immutable descriptors.
        unsafe impl Send for CachedNulls {}
        unsafe impl Sync for CachedNulls {}
        static CACHE: std::sync::OnceLock<Option<CachedNulls>> = std::sync::OnceLock::new();
        let host = host()?;
        let cache = CACHE.get_or_init(|| {
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: COUNT as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            };
            // SAFETY: creates a tiny 17-slot CPU-only descriptor heap once.
            let heap: ID3D12DescriptorHeap = unsafe { host.device.CreateDescriptorHeap(&desc).ok()? };
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as usize;
            let base = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;
            let handles = std::array::from_fn(|i| base + i * step);
            let h = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: handles[i] };
            null_view(h(0));
            null_buffer_view(h(1));
            let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: null CBV in slot 2 of our CPU-only heap.
            unsafe {
                host.device.CreateConstantBufferView(Some(&cbv), h(2));
            }
            for (slot, dim) in [(3, 2u32), (4, 3), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)] {
                let mut words = [0u32; 10];
                words[1] = dim;
                null_like(&words, h(slot));
            }
            let buf_uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                ..Default::default()
            };
            // SAFETY: immutable null UAV descriptors in slots 11..17 of our CPU-only heap.
            unsafe {
                host.device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&buf_uav), h(11));
                for (slot, dim) in [(12, Some(2u32)), (13, Some(3)), (14, None), (15, Some(5)), (16, Some(8))] {
                    let uav = null_uav_desc(dim);
                    host.device.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&uav), h(slot));
                }
            }
            Some(CachedNulls { _heap: heap, handles })
        })
        .as_ref()?;
        Some(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cache.handles[kind] })
    }

    fn queue_cached_null(&self, to: u32, kind: usize, batch: &mut DescriptorCopyBatch) -> bool {
        let tag = (kind + 1) as u8;
        if self.null_kind.get(to as usize) == Some(tag) {
            return true;
        }
        let Some(from) = Self::cached_null_handle(kind) else {
            return false;
        };
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        batch.push(self.cpu(to), from);
        self.null_kind.set(to as usize, tag);
        true
    }

    /// Copies a pre-built constant null descriptor (`0` = 2D SRV, `1` = raw-buffer
    /// SRV, `2` = CBV) directly into shader-visible slot `to`.
    fn copy_cached_null(&self, to: u32, kind: usize) -> bool {
        let tag = (kind + 1) as u8;
        if self.null_kind.get(to as usize) == Some(tag) {
            return true;
        }
        let (Some(host), Some(from)) = (host(), Self::cached_null_handle(kind)) else {
            return false;
        };
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: immutable CPU-only source descriptor -> live shader-visible slot.
        unsafe {
            host.device
                .CopyDescriptorsSimple(1, self.cpu(to), from, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }
        self.null_kind.set(to as usize, tag);
        true
    }

    /// Slot `index` now views `resource` (or nothing of the host's), deduplicating
    /// COM refs in [`ResourceTable`] for title slots (`0..count`) and using
    /// lock-free [`DynamicKept`] for dynamic slots (`count..`).
    fn keep_ref(&self, index: u32, resource: Option<&ID3D12Resource>) {
        if let Some(dyn_idx) = index.checked_sub(self.count) {
            self.dynamic_kept.replace_ref(dyn_idx as usize, resource);
            return;
        }
        self.keep_title_ref(index, resource, false);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        self.keep_window(index, resource, false);
    }

    /// Slot `index` now views `resource`, recording title-slot window-shadow
    /// back-references when `is_window` is true so [`HostHeap::unpin_window_buffers`]
    /// can safely evict idle loose windows (`ARCH-05`).
    fn keep_window(&self, index: u32, resource: Option<ID3D12Resource>, is_window: bool) {
        if let Some(dyn_idx) = index.checked_sub(self.count) {
            self.dynamic_kept.replace(dyn_idx as usize, resource);
            return;
        }
        self.keep_title_owned(index, resource, is_window);
    }

    fn keep_title_ref(&self, index: u32, resource: Option<&ID3D12Resource>, is_window: bool) {
        use windows::core::Interface;
        let idx = index as usize;
        if idx >= self.count as usize {
            return;
        }
        let new_raw = resource.map_or(0usize, |r| r.as_raw() as usize);
        if new_raw == 0 && self.kept_ids.get(idx) == 0 && !self.window_slot_bits.get(idx) {
            return;
        }
        let _slot_guard = self.kept_locks[idx % self.kept_locks.len()].lock().unwrap_or_else(|e| e.into_inner());
        let old_id = self.kept_ids.get(idx);
        let was_window = self.window_slot_bits.get(idx);
        let (old_raw, retired) = {
            let mut table = self.resources.lock().unwrap_or_else(|e| e.into_inner());
            let old_raw = table.raw_of(old_id);
            if old_raw == new_raw {
                let need_window = new_raw != 0 && is_window;
                if was_window == need_window {
                    return;
                }
                (old_raw, None)
            } else {
                let new_id = match resource {
                    Some(res) => table.acquire_ref(res),
                    None => 0,
                };
                self.kept_ids.set(idx, new_id);
                let retired = table.release(old_id);
                (old_raw, retired)
            }
        };
        self.sync_window_slot(index, old_raw, new_raw, was_window, new_raw != 0 && is_window);
        drop(_slot_guard);
        if let Some(old) = retired {
            crate::host::retire_resource(old);
        }
    }

    fn keep_title_owned(&self, index: u32, resource: Option<ID3D12Resource>, is_window: bool) {
        use windows::core::Interface;
        let idx = index as usize;
        if idx >= self.count as usize {
            return;
        }
        let new_raw = resource.as_ref().map_or(0usize, |r| r.as_raw() as usize);
        if new_raw == 0 && self.kept_ids.get(idx) == 0 && !self.window_slot_bits.get(idx) {
            return;
        }
        let _slot_guard = self.kept_locks[idx % self.kept_locks.len()].lock().unwrap_or_else(|e| e.into_inner());
        let old_id = self.kept_ids.get(idx);
        let was_window = self.window_slot_bits.get(idx);
        let (old_raw, retired, redundant) = {
            let mut table = self.resources.lock().unwrap_or_else(|e| e.into_inner());
            let old_raw = table.raw_of(old_id);
            if old_raw == new_raw {
                let need_window = new_raw != 0 && is_window;
                if was_window == need_window {
                    drop(table);
                    drop(_slot_guard);
                    drop(resource);
                    return;
                }
                (old_raw, None, resource)
            } else {
                let (new_id, redundant) = match resource {
                    Some(res) => table.acquire_owned(res),
                    None => (0, None),
                };
                self.kept_ids.set(idx, new_id);
                let retired = table.release(old_id);
                (old_raw, retired, redundant)
            }
        };
        self.sync_window_slot(index, old_raw, new_raw, was_window, new_raw != 0 && is_window);
        drop(_slot_guard);
        drop(redundant);
        // A running submission may still read the old resource through this slot.
        if let Some(old) = retired {
            crate::host::retire_resource(old);
        }
    }

    /// Maintains the secondary `window_slots` index for title slots holding
    /// `window == true` loose shadow buffers (leaf lock, never called under `resources`).
    fn sync_window_slot(&self, index: u32, old_raw: usize, new_raw: usize, was_window: bool, need_window: bool) {
        if !was_window && !need_window {
            return;
        }
        let idx = index as usize;
        let mut ws = self.window_slots.lock().unwrap_or_else(|e| e.into_inner());
        if was_window && (old_raw != new_raw || !need_window) {
            if let Some(list) = ws.get_mut(&old_raw) {
                if let Some(pos) = list.iter().position(|&s| s == index) {
                    list.swap_remove(pos);
                }
                if list.is_empty() {
                    ws.remove(&old_raw);
                }
            }
            if !need_window {
                self.window_slot_bits.set(idx, false);
            }
        }
        if need_window && (!was_window || old_raw != new_raw) {
            let list = ws.entry(new_raw).or_default();
            if !list.contains(&index) {
                list.push(index);
            }
            self.window_slot_bits.set(idx, true);
        }
    }

    /// Raw COM pointer currently held by title slot `index` (`0` = None).
    fn kept_raw_at(&self, index: u32) -> usize {
        let id = self.kept_ids.get(index as usize);
        if id == 0 {
            return 0;
        }
        self.resources.lock().unwrap_or_else(|e| e.into_inner()).raw_of(id)
    }

    /// What title slot `index` views (dynamic slots are exclusively owned and
    /// not inspected cross-thread).
    fn kept_in(&self, index: u32) -> Option<ID3D12Resource> {
        let id = self.kept_ids.get(index as usize);
        if id == 0 {
            return None;
        }
        self.resources.lock().unwrap_or_else(|e| e.into_inner()).clone_of(id)
    }

    /// Records the guest base address of slot 0 on first view-heap mirror.
    #[inline]
    fn note_title_start(&self, index: u32, handle: u64) {
        if !self.sampler && self.title_start.load(Ordering::Relaxed) == 0
            && let Some(base) = handle.checked_sub(u64::from(index) * 32)
            && base != 0
        {
            self.title_start.store(base, Ordering::Release);
        }
    }

    /// Safe stale-slot revalidation unpinning (`Spec-Trap #16` & `Spec-Trap #4`):
    ///
    /// For each candidate `(raw_buffer, win_start, win_len, last_used_frame)` whose
    /// window has been idle for `>= STALE_WINDOW_IDLE_FRAMES` (`300` frames), inspects
    /// only title-heap slots (`0..self.count`) currently pinning `raw_buffer`.
    /// Reads the guest descriptor's current bytes in Xbox RAM via `safecopy`
    /// (bypassing / never tripping `PAGE_NOACCESS` tripwires per Spec-Trap #4) and
    /// compares the current guest descriptor's VA/length against `[win_start, win_start + win_len)`.
    ///
    /// ONLY if the guest has ALREADY overwritten or zeroed that descriptor slot so
    /// it no longer references `[win_start, win_start + win_len)` (a provably dead
    /// slot) does this method null the host descriptor slot + raw twin and retire
    /// the COM reference (Null-Before-Retire). Live bindless slots are NEVER touched.
    pub(crate) fn unpin_stale_window_slots(
        &self,
        candidates: &[(usize, u64, u64, u64)],
        now_frame: u64,
        heap_start_override: Option<u64>,
    ) -> usize {
        if self.sampler || candidates.is_empty() {
            return 0;
        }
        let heap_start = heap_start_override.unwrap_or_else(|| self.title_start.load(Ordering::Acquire));
        if heap_start == 0 {
            return 0;
        }
        let work: Vec<(usize, u64, u64, Vec<u32>)> = {
            let ws = self.window_slots.lock().unwrap_or_else(|e| e.into_inner());
            if ws.is_empty() {
                return 0;
            }
            candidates
                .iter()
                .filter_map(|&(raw, win_start, win_len, last_used_frame)| {
                    if win_len == 0
                        || now_frame.saturating_sub(last_used_frame) < STALE_WINDOW_IDLE_FRAMES
                    {
                        return None;
                    }
                    ws.get(&raw)
                        .map(|slots| (raw, win_start, win_len, slots.clone()))
                })
                .collect()
        };
        if work.is_empty() {
            return 0;
        }
        let mut unpinned = 0usize;
        for (raw, win_start, win_len, slots) in work {
            for slot in slots {
                if slot >= self.count {
                    continue;
                }
                let Some(guest_handle) = heap_start.checked_add(u64::from(slot) * 32) else {
                    continue;
                };
                // Quick unlocked pre-check before acquiring the per-slot lock:
                // skip immediately if guest RAM cannot be safely read or still references the window.
                if !guest_slot_proven_stale(guest_handle, win_start, win_len) {
                    continue;
                }
                let _made = self.made.lock(slot);
                if !self.window_slot_bits.get(slot as usize) || self.kept_raw_at(slot) != raw {
                    continue;
                }
                // Re-verify under the slot lock so a concurrent overwrite back to
                // this window is never erroneously nulled (Spec-Trap #16).
                if !guest_slot_proven_stale(guest_handle, win_start, win_len) {
                    continue;
                }
                // MANDATORY NULL-BEFORE-RETIRE (Spec-Trap #16):
                // Clear quick/made caches, overwrite shader-visible slot + raw twin
                // with `null_buffer_view` FIRST, then retire COM ref via `keep_title_owned`.
                self.quick_is(slot, 0);
                self.made.set(slot, 0);
                if !self.copy_cached_null(slot, 1) {
                    null_buffer_view(self.at(slot));
                    self.put(slot);
                }
                self.clear_twin(slot);
                self.write_stride(slot, 0, 0);
                self.keep_title_owned(slot, None, false);
                unpinned += 1;
            }
        }
        if unpinned > 0 {
            bump_buffer_view_cache_epoch();
        }
        unpinned
    }

    /// Unpins title-heap descriptor slots (`0..self.count`) referencing any of
    /// `raw_buffers` (`window == true` loose shadows) using the mandatory
    /// Null-Before-Retire protocol (`ARCH-05`, Spec-Trap #16):
    /// 1. Overwrites the shader-visible descriptor slot AND its raw twin with
    ///    `null_buffer_view` FIRST,
    /// 2. Clears `made`, `quick`, and `strides` so future uses re-materialize,
    /// 3. Releases the `ResourceTable` entry so the COM ref retires through
    ///    `host::retire_resource` (keeping in-flight lists alive until fence
    ///    completion — zero GPU page faults).
    /// NEVER touches `dynamic_kept` slots or non-window shadows.
    pub(crate) fn unpin_window_buffers(&self, raw_buffers: &[usize]) -> usize {
        if self.sampler || raw_buffers.is_empty() {
            return 0;
        }
        let work: Vec<(usize, Vec<u32>)> = {
            let ws = self.window_slots.lock().unwrap_or_else(|e| e.into_inner());
            if ws.is_empty() {
                return 0;
            }
            raw_buffers
                .iter()
                .filter_map(|&raw| ws.get(&raw).map(|slots| (raw, slots.clone())))
                .collect()
        };
        if work.is_empty() {
            return 0;
        }
        let mut unpinned = 0usize;
        for (raw, slots) in work {
            for slot in slots {
                if slot >= self.count {
                    continue;
                }
                let _made = self.made.lock(slot);
                if !self.window_slot_bits.get(slot as usize) || self.kept_raw_at(slot) != raw {
                    continue;
                }
                // MANDATORY NULL-BEFORE-RETIRE (Spec-Trap #16):
                // Clear quick/made caches, overwrite shader-visible slot + raw twin
                // with `null_buffer_view` FIRST, then retire COM ref via `keep_title_owned`.
                self.quick_is(slot, 0);
                self.made.set(slot, 0);
                if !self.copy_cached_null(slot, 1) {
                    null_buffer_view(self.at(slot));
                    self.put(slot);
                }
                self.clear_twin(slot);
                self.write_stride(slot, 0, 0);
                self.keep_title_owned(slot, None, false);
                unpinned += 1;
            }
        }
        if unpinned > 0 {
            bump_buffer_view_cache_epoch();
        }
        unpinned
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.at(index))
        };
        self.put(index);
        // (Bound by its handle, not read by index.)
        self.clear_twin(index);
        self.write_stride(index, 1, elements.saturating_mul(4));
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.at(index));
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 0, 0);
        self.keep(index, None);
    }

    /// Ensures a contiguous run of raw-memory copy UAVs in this heap covering
    /// `copies`, growing geometrically as needed while keeping older bases
    /// reserved for recorded command lists. Returns `(base, count)` where slot
    /// `base` is the null raw UAV and `base + 1..=base + count` view `copies`.
    pub(crate) fn raw_page_views(
        &self,
        copies: &[std::sync::Arc<crate::shadow::Shadow>],
    ) -> Option<(u32, u32)> {
        let mut run = self.raw_pages.lock().unwrap_or_else(|e| e.into_inner());
        let requested = u32::try_from(copies.len()).ok()?;
        let needed = requested.saturating_add(1);
        if run.as_ref().is_none_or(|r| needed > r.slots && r.slots < self.dynamic) {
            let slots = needed
                .checked_next_power_of_two()
                .unwrap_or(u32::MAX)
                .max(FIRST_RAW_PAGE_SLOTS)
                .min(self.dynamic);
            let reservation = self.dynamic(slots).map(|base| (base, slots)).or_else(|| {
                // An already fragmented heap can still expose a prefix on its
                // first bind. Subsequent binds retry growth as lists release slots.
                let small = needed.min(FIRST_RAW_PAGE_SLOTS);
                if run.is_some() || small >= slots {
                    return None;
                }
                self.dynamic(small).map(|base| (base, small))
            });
            if let Some((base, slots)) = reservation {
                // Every slot of the run: a number shows no view of another kind.
                for n in 0..slots {
                    self.null_raw_uav_at(base + n);
                }
                if let Some(old) = run.as_ref() {
                    tracing::info!(previous = old.slots, slots, "grew raw memory descriptor run");
                }
                *run = Some(RawPageRun { base, slots, populated: 0 });
            }
        }
        let run = run.as_mut()?;
        let count = requested.min(run.slots - 1);
        if count < requested {
            static WARN: std::sync::Once = std::sync::Once::new();
            WARN.call_once(|| {
                tracing::warn!(
                    requested,
                    count,
                    "raw memory bindings are limited by the available descriptor heap"
                );
            });
        }
        for (n, shadow) in copies[..count as usize].iter().enumerate().skip(run.populated as usize) {
            let (_, len) = shadow.span();
            self.raw_uav_at(run.base + 1 + n as u32, &shadow.buffer, 0, len);
        }
        run.populated = run.populated.max(count);
        Some((run.base, count))
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.at(index),
            )
        };
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 1, elements.saturating_mul(4));
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        let Some(host) = host() else { return };
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: 1,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: a slot of our heap; a null resource is allowed with a desc.
        unsafe {
            host.device.CreateUnorderedAccessView(
                None::<&ID3D12Resource>,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.at(index),
            )
        };
        self.put(index);
        self.clear_twin(index);
        self.write_stride(index, 0, 0);
        self.keep(index, None);
    }

    /// The shader-visible handle of slot `index`.
    #[inline]
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: self.gpu + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory, `size` bytes at `gpu` of `buffer`, which the slot holds.
    ///
    /// The list holds it too, until it is reset; the slot shows it until it
    /// is made again, and a shader that picks the slot by an index it
    /// computes from wrong data reads it through there. The list's scratch
    /// chunks go back to the pool and are freed past its limit: session n02
    /// lost the device to a page fault in the memory of three freed ones.
    /// `VICTORY_SLOT_HOLDS=0`: only the list holds it, as before.
    pub(crate) fn constant_at(&self, index: u32, buffer: &ID3D12Resource, gpu: u64, size: u32) {
        static HOLDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let holds = *HOLDS.get_or_init(|| std::env::var_os("VICTORY_SLOT_HOLDS").is_none_or(|v| v != "0"));
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.at(index))
        };
        self.put(index);
        // A program that reads it as raw bytes reads its twin.
        let twin = self.twin(index).filter(|_| desc.SizeInBytes > 0);
        if let Some(twin) = twin {
            // SAFETY: a live buffer that holds the range.
            let first = (gpu - unsafe { buffer.GetGPUVirtualAddress() }) / 4;
            let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: first,
                        NumElements: desc.SizeInBytes / 4,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            // SAFETY: as above, and a slot of our heaps.
            unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), twin) };
        }
        self.twin_made(index, twin.is_some());
        self.write_stride(index, 0, 0);
        // After the descriptor: what it showed is let go when nothing shows
        // it any more.
        self.keep_ref(index, holds.then_some(buffer));
    }

    /// Batched variant of [`HostHeap::constant_at`] using `TableBatch`
    /// (`BatchStage` + `DescriptorCopyBatch`) and a precomputed buffer base GPU
    /// address (zero `GetGPUVirtualAddress()` COM call). When `port_cbv::on()`
    /// and raw twins are active, translated table CBV loads read only the raw
    /// twin so the primary slot reuses a cached null CBV instead of calling
    /// `CreateConstantBufferView`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn constant_at_batched(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        base: u64,
        gpu: u64,
        size: u32,
        tb: &mut TableBatch<'_>,
    ) {
        static HOLDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let holds = *HOLDS.get_or_init(|| std::env::var_os("VICTORY_SLOT_HOLDS").is_none_or(|v| v != "0"));
        let Some(host) = host() else { return };
        let cbv_bytes = constant_bytes(size);
        let has_twin = self.twin(index).is_some() && cbv_bytes > 0;
        let raw_only = crate::port_cbv::on() && has_twin;
        if raw_only {
            if !self.queue_cached_null(index, 2, &mut tb.copy) {
                let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                if let Some(s0) = tb.alloc_stage(1) {
                    // SAFETY: slot of our CPU staging heap; null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&cbv), s0) };
                    self.queue_copy_from(index, s0, &mut tb.copy);
                } else {
                    // SAFETY: slot of our heap; null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&cbv), self.at(index)) };
                    self.put(index);
                }
            }
            let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: (gpu - base) / 4,
                        NumElements: cbv_bytes / 4,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            if let Some(s_twin) = tb.alloc_stage(1) {
                // SAFETY: live buffer covering range; CPU staging slot.
                unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), s_twin) };
                self.queue_copy_from(self.raw + index, s_twin, &mut tb.copy);
                self.twins.set(index as usize, true);
            } else if let Some(twin) = self.twin(index) {
                // SAFETY: live buffer covering range; heap slot.
                unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), twin) };
                self.twin_made(index, true);
            }
        } else {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: cbv_bytes,
            };
            let need = 1 + u32::from(has_twin);
            if let Some((s0, s1)) = tb.alloc_stage_pair(need) {
                // SAFETY: CPU staging slot; live buffer.
                unsafe { host.device.CreateConstantBufferView(Some(&desc), s0) };
                self.queue_copy_from(index, s0, &mut tb.copy);
                if has_twin {
                    let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: (gpu - base) / 4,
                                NumElements: cbv_bytes / 4,
                                StructureByteStride: 0,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            },
                        },
                    };
                    // SAFETY: CPU staging slot; live buffer.
                    unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), s1) };
                    self.queue_copy_from(self.raw + index, s1, &mut tb.copy);
                    self.twins.set(index as usize, true);
                } else {
                    self.clear_twin_batched(index, &mut tb.copy);
                }
            } else {
                // SAFETY: slot of our heap; live buffer.
                unsafe { host.device.CreateConstantBufferView(Some(&desc), self.at(index)) };
                self.put(index);
                if let Some(twin) = self.twin(index).filter(|_| cbv_bytes > 0) {
                    let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_R32_TYPELESS,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV {
                                FirstElement: (gpu - base) / 4,
                                NumElements: cbv_bytes / 4,
                                StructureByteStride: 0,
                                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                            },
                        },
                    };
                    // SAFETY: live buffer covering range.
                    unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), twin) };
                    self.twin_made(index, true);
                } else {
                    self.clear_twin_batched(index, &mut tb.copy);
                }
            }
        }
        self.write_stride(index, 0, 0);
        self.keep_ref(index, holds.then_some(buffer));
    }

    /// Renderer port 5 (`crate::port_cbv`): makes slot `index` (a dynamic
    /// one) show the constant buffer of `size` bytes at `gpu` of `buffer`
    /// (title memory in place) with nothing copied: its raw twin a raw view
    /// of those bytes, which translated shaders read; the slot itself reuses a
    /// cached null CBV under `port_cbv::on()` (since shaders only read the raw
    /// twin), batched through `TableBatch` (`BatchStage` + `DescriptorCopyBatch`).
    /// Takes cached `(base, width)` from `Shadow` to avoid per-CBV COM queries.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn constant_raw_at_known(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        base: u64,
        width: u64,
        gpu: u64,
        size: u32,
        tb: &mut TableBatch<'_>,
    ) -> bool {
        let Some(host) = host() else { return false };
        let Some(twin) = self.twin(index) else { return false };
        let Some(layout) = crate::port_cbv::raw_cbv_layout(base, width, gpu, size) else {
            return false;
        };
        let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: layout.first_element,
                    NumElements: layout.num_elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        if crate::port_cbv::on() || layout.cbv_location == 0 {
            if !self.queue_cached_null(index, 2, &mut tb.copy) {
                let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                if let Some(s0) = tb.alloc_stage(1) {
                    // SAFETY: CPU staging slot; null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&cbv), s0) };
                    self.queue_copy_from(index, s0, &mut tb.copy);
                } else {
                    // SAFETY: slot of our heap; null CBV.
                    unsafe { host.device.CreateConstantBufferView(Some(&cbv), self.at(index)) };
                    self.put(index);
                }
            }
        } else {
            let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: layout.cbv_location,
                SizeInBytes: layout.cbv_size,
            };
            if let Some(s0) = tb.alloc_stage(1) {
                // SAFETY: CPU staging slot; range lies in live buffer.
                unsafe { host.device.CreateConstantBufferView(Some(&cbv), s0) };
                self.queue_copy_from(index, s0, &mut tb.copy);
            } else {
                // SAFETY: slot of our heap; range lies in live buffer.
                unsafe { host.device.CreateConstantBufferView(Some(&cbv), self.at(index)) };
                self.put(index);
            }
        }
        if let Some(s_twin) = tb.alloc_stage(1) {
            // SAFETY: CPU staging slot; range lies in live buffer.
            unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), s_twin) };
            self.queue_copy_from(self.raw + index, s_twin, &mut tb.copy);
            self.twins.set(index as usize, true);
        } else {
            // SAFETY: slot of our heap; range lies in live buffer.
            unsafe { host.device.CreateShaderResourceView(buffer, Some(&raw), twin) };
            self.twin_made(index, true);
        }
        self.write_stride(index, 0, 0);
        self.keep_ref(index, Some(buffer));
        true
    }

    /// Where a descriptor for slot `index` is made, then [`HostHeap::put`]:
    /// a slot of this thread's in a heap the card does not see
    /// ([`staged`]), or the slot itself.
    fn at(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        staged(self.sampler, 0).unwrap_or_else(|| self.cpu(index))
    }

    /// The descriptor this thread made last ([`HostHeap::at`]) goes to slot
    /// `index`.
    fn put(&self, index: u32) {
        self.copy_to(index, 0);
    }

    /// The raw twin this thread made last goes to that of slot `index`.
    fn put_twin(&self, index: u32) {
        if self.twin(index).is_some() {
            self.copy_to(self.raw + index, 1);
        }
    }

    fn copy_to(&self, to: u32, which: u32) {
        self.null_kind.set(to as usize, 0);
        #[cfg(feature = "diagnostics")]
        if let Some(written) = self.written.get(to as usize) {
            written.store(crate::host::coarse_ms() as u32, std::sync::atomic::Ordering::Relaxed);
        }
        let (Some(from), Some(host)) = (staged(self.sampler, which), host()) else {
            return; // made in place
        };
        let kind = if self.sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        // SAFETY: a slot of our heap, and one of this thread's heap slots.
        unsafe { host.device.CopyDescriptorsSimple(1, self.cpu(to), from, kind) };
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// The records last seen of the title's heap, `step` bytes each, held for
    /// threads that compare what they read of its changed pages beside one
    /// another ([`Copied::changes`]; `super::sync_heaps`: the title copies
    /// records into its heap as bytes).
    ///
    /// The page says only that something on it was written, and `mirror`
    /// by itself would take every buffer or placed-texture record on it for
    /// newly written each time (it leaves such a slot unmade until a draw
    /// or a pass uses it): the texture queue's entry for the slot would
    /// start again, and the buffer pass would look at 2,048 slots for one
    /// that changed.
    pub(crate) fn copied_begin(&self, step: usize) -> Copied<'_> {
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        if copied.is_empty() {
            // Nothing there: a slot never looked at that holds nothing is
            // as it was.
            copied.resize(self.count as usize * step, 0);
        }
        let (at, len) = (copied.as_mut_ptr(), copied.len());
        Copied { at, len, step, _held: copied }
    }

    /// Whether new records of draws' tables are left to the draws
    /// (`VICTORY_HEAP_SYNC=all`: every new record is mirrored at once, as
    /// it first was).
    ///
    /// The title builds each draw's tables in its heap by copying records
    /// there as bytes: 60,000 to 210,000 a frame in its world (session
    /// play-1125-0127), 0.9 microseconds each on the submitting thread and
    /// 1.4 more in the buffer pass they were all handed to, 120 to 600 ms
    /// of every frame, and the draws make what they read of them anyway.
    pub(crate) fn copied_waits(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "all"))
    }

    /// If title slot `index` was read through a table within [`COPIED_FRAMES`]
    /// frames of `frame`, returns the frame (`stamp_frame + COPIED_FRAMES + 1`)
    /// at which it should be re-checked in case it was repurposed to bindless
    /// after `self.copied` was updated (`ARCH-08`).
    pub(crate) fn copied_tabled_until(&self, index: u32, frame: u64) -> Option<u64> {
        if !self.copied_waits() {
            return None;
        }
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        if stamp == 0 {
            return None;
        }
        let stamp_frame = stamp - 1;
        (frame.saturating_sub(stamp_frame) <= COPIED_FRAMES)
            .then(|| stamp_frame.saturating_add(COPIED_FRAMES + 1))
    }

    /// Whether a draw read title slot `index` through a table within the
    /// last few frames before `frame` ([`HostHeap::tabled`]): what it holds
    /// is such a draw's to make.
    pub(crate) fn copied_tabled(&self, index: u32, frame: u64) -> bool {
        self.copied_tabled_until(index, frame).is_some()
    }

    /// New records the title copied into slots `slots` in `frame` wait for
    /// the draw that may read them through a table (`due = frame + COPIED_GRACE`).
    pub(crate) fn copied_wait(&self, frame: u64, slots: Vec<u32>) {
        self.copied_wait_at(frame.saturating_add(COPIED_GRACE), slots);
    }

    /// Defers `slots` to become due at `due_frame` (used both by `copied_wait`
    /// and by `copied_tabled_until` re-checks so no bindless slot is silently
    /// dropped after `self.copied` was updated).
    pub(crate) fn copied_wait_at(&self, due_frame: u64, slots: Vec<u32>) {
        if slots.is_empty() {
            return;
        }
        self.waiting
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(due_frame)
            .or_default()
            .extend(slots);
    }

    /// Defers non-view slot unmakes off the submission-critical path onto
    /// maintenance (`CPU-02`), where each slot re-checks `word(0) != VIEW_TAG`.
    pub(crate) fn defer_unmakes(&self, slots: Vec<u32>) {
        if slots.is_empty() {
            return;
        }
        self.unmakes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(slots);
    }

    /// Takes up to `max` deferred non-view unmake slots for maintenance.
    pub(crate) fn take_deferred_unmakes(&self, max: usize) -> Vec<u32> {
        let mut unmakes = self.unmakes.lock().unwrap_or_else(|e| e.into_inner());
        if unmakes.is_empty() {
            return Vec::new();
        }
        let mut out = if unmakes.len() <= max {
            std::mem::take(&mut *unmakes)
        } else {
            let split = unmakes.len() - max;
            unmakes.split_off(split)
        };
        drop(unmakes);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The slots that have waited long enough at `frame`
    /// ([`HostHeap::copied_wait`] / [`HostHeap::copied_wait_at`]), each once.
    /// Paced per round when the queue is healthy so a single submission round
    /// does not stall on thousands of deferred slots at once.
    pub(crate) fn copied_before(&self, frame: u64) -> Vec<u32> {
        const PER_ROUND: usize = 1024;
        let mut waiting = self.waiting.lock().unwrap_or_else(|e| e.into_inner());
        let drain_all = waiting.len() > 64;
        let mut out = Vec::new();
        while let Some((&due, _)) = waiting.first_key_value() {
            if due > frame {
                break;
            }
            if !drain_all && out.len() >= PER_ROUND {
                break;
            }
            let Some((_, mut slots)) = waiting.pop_first() else {
                break;
            };
            if !drain_all && out.len() + slots.len() > PER_ROUND {
                let take = PER_ROUND.saturating_sub(out.len()).max(1);
                let rest = slots.split_off(take);
                out.extend(slots);
                if !rest.is_empty() {
                    waiting.insert(due, rest);
                }
                break;
            }
            out.extend(slots);
        }
        drop(waiting);
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Makes host slot `index` from the record at `handle` if it is still
    /// the one with `key`, whose texture's upload is in the queue now
    /// (`crate::heap_worker`). The slot's place in the queue of textures
    /// picked by index was settled when that upload was recorded, and stays
    /// as it is. False if the title wrote the slot again.
    pub(crate) fn publish(&self, index: u32, handle: u64, key: u64) -> bool {
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || u64::from(generation) << 32 | u64::from(i) != key {
            return false;
        }
        PUBLISHING.set(true);
        let shown = self.mirror(index, handle, true).is_some();
        PUBLISHING.set(false);
        shown
    }

    /// Batched [`HostHeap::publish`] for `heap_worker`: verifies each slot still
    /// holds `key`, then mirrors all still-matching slots through
    /// `mirror_batch_inner` under `PUBLISHING = true`. Returns `(shown, stale)`.
    pub(crate) fn publish_batch(&self, items: &[(u32, u64, u64)]) -> (u64, u64) {
        if items.is_empty() {
            return (0, 0);
        }
        if self.sampler || items.len() == 1 || !batch_stage_available() {
            let (mut shown, mut stale) = (0u64, 0u64);
            for &(index, handle, key) in items {
                if self.publish(index, handle, key) {
                    shown += 1;
                } else {
                    stale += 1;
                }
            }
            return (shown, stale);
        }
        let mut valid: Vec<(u32, u64, Option<[u32; 3]>)> = Vec::with_capacity(items.len());
        let mut stale = 0u64;
        for &(index, handle, key) in items {
            // SAFETY: the handle is a descriptor slot in title memory.
            let rec = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            if rec[0] != VIEW_TAG || (u64::from(rec[2]) << 32 | u64::from(rec[1])) != key {
                stale += 1;
                continue;
            }
            valid.push((index, handle, Some(rec)));
        }
        if valid.is_empty() {
            return (0, stale);
        }
        struct ResetPublishing;
        impl Drop for ResetPublishing {
            fn drop(&mut self) {
                PUBLISHING.set(false);
            }
        }
        PUBLISHING.set(true);
        let _reset = ResetPublishing;
        let mirrored = self.mirror_batch_inner(&valid, true, true, true);
        let shown = mirrored.iter().filter(|(_, v, _, _)| v.is_some()).count() as u64;
        stale += mirrored.len() as u64 - shown;
        (shown, stale)
    }

    /// Batched [`HostHeap::mirror`] with `for_draw = true` for `make_waiting`.
    pub(crate) fn mirror_batch_handles(&self, items: &[(u32, u64, Option<[u32; 3]>)]) {
        if items.is_empty() {
            return;
        }
        if self.sampler || items.len() == 1 || !batch_stage_available() {
            for &(index, handle, _) in items {
                let _ = self.mirror_inner(index, handle, true, true);
            }
            return;
        }
        let _ = self.mirror_batch_inner(items, true, true, false);
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        self.mirror_inner(index, handle, for_draw, true)
    }

    /// Same as [`HostHeap::mirror`] with `for_draw = true`, for `heap_buffer_pass`:
    /// avoids re-queuing slots the heap pass itself just mirrored back into `self.changed`.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mirror_heap_pass(&self, index: u32, handle: u64) -> Option<View> {
        self.mirror_inner(index, handle, true, false)
    }

    /// Batched [`HostHeap::mirror`] for `sync_heaps`: decodes records under one
    /// `VIEWS.read()` per mini-batch and flushes descriptor/null copies via
    /// `DescriptorCopyBatch` while holding the affected stripe locks.
    pub(crate) fn mirror_batch(&self, slots: &[u32], heap_start: u64, step: u64, for_draw: bool) {
        if slots.is_empty() {
            return;
        }
        if self.sampler || slots.len() == 1 || !batch_stage_available() {
            for &index in slots {
                self.mirror_inner(index, heap_start + u64::from(index) * step, for_draw, true);
            }
            return;
        }
        let items: Vec<(u32, u64, Option<[u32; 3]>)> = slots
            .iter()
            .map(|&index| (index, heap_start + u64::from(index) * step, None))
            .collect();
        let _ = self.mirror_batch_inner(&items, for_draw, true, false);
    }

    /// Batched [`HostHeap::mirror_heap_pass`] for `heap_buffer_pass`: decodes
    /// records under one `VIEWS.read()` per mini-batch, flushes descriptor/null
    /// copies in batched `CopyDescriptors` calls, and returns any resolved
    /// `(Shadow, address, len)` directly to avoid a second `shadow::find`.
    pub(crate) fn mirror_heap_pass_batch(
        &self,
        misses: &[(u32, u64, Option<[u32; 3]>)],
    ) -> Vec<(u32, Option<View>, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>, bool)> {
        if misses.is_empty() {
            return Vec::new();
        }
        if self.sampler || !batch_stage_available() {
            return misses
                .iter()
                .map(|&(index, handle, _)| (index, self.mirror_inner(index, handle, true, false), None, false))
                .collect();
        }
        self.mirror_batch_inner(misses, true, false, true)
    }

    fn mirror_batch_inner(
        &self,
        items: &[(u32, u64, Option<[u32; 3]>)],
        for_draw: bool,
        record_change: bool,
        want_results: bool,
    ) -> Vec<(u32, Option<View>, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>, bool)> {
        const MINI: usize = 32;
        let mut results = if want_results {
            Vec::with_capacity(items.len())
        } else {
            Vec::new()
        };
        if want_results {
            results.resize_with(items.len(), || (0, None, None, false));
        }
        let quick_on = quick_mirror();
        if let Some(&(index, handle, _)) = items.first() {
            self.note_title_start(index, handle);
        }
        for (chunk_start, chunk) in items.chunks(MINI).enumerate().map(|(c, s)| (c * MINI, s)) {
            // Decode records, pre-fetch old views for non-draw overwrites, and handle
            // lock-free quick hits under a single `VIEWS.read()`.
            let mut decoded: [(u32, u64, u64, Option<View>, usize, u64, Option<View>); MINI] =
                std::array::from_fn(|_| (0, 0, 0, None, 0, 0, None));
            let mut pending_len = 0usize;
            {
                let views = super::lock_views_read();
                for (offset, &(index, handle, pre_record)) in chunk.iter().enumerate() {
                    let out_idx = chunk_start + offset;
                    if want_results {
                        results[out_idx].0 = index;
                    }
                    if index >= self.count {
                        static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                            tracing::warn!(
                                index,
                                count = self.count,
                                for_draw,
                                "title slot past the host heap"
                            );
                        }
                        continue;
                    }
                    let [tag, i, generation] = match pre_record {
                        Some(r) => r,
                        // SAFETY: the handle is a descriptor slot in title memory.
                        None => unsafe { (handle as *const [u32; 3]).read_unaligned() },
                    };
                    let key = u64::from(generation) << 32 | u64::from(i);
                    if tag == VIEW_TAG
                        && quick_on
                        && self.quick.get(index as usize, std::sync::atomic::Ordering::Acquire) == Some(key)
                        && let Some(view) = views.view_at(i, generation)
                    {
                        if want_results {
                            results[out_idx] = (index, Some(view), None, false);
                        }
                        continue;
                    }
                    let shown_hint = if !for_draw { self.made.get(index).unwrap_or(0) } else { 0 };
                    if tag != VIEW_TAG && !for_draw && shown_hint == 0 {
                        continue;
                    }
                    let view = (tag == VIEW_TAG).then(|| views.view_at(i, generation)).flatten();
                    let old_hint = if !for_draw
                        && shown_hint != 0
                        && shown_hint != key
                        && view.as_ref().is_some_and(|v| !v.is_buffer())
                    {
                        views.view_at(shown_hint as u32, (shown_hint >> 32) as u32)
                    } else {
                        None
                    };
                    decoded[pending_len] = (index, handle, key, view, out_idx, shown_hint, old_hint);
                    pending_len += 1;
                }
            }
            if pending_len == 0 {
                continue;
            }
            with_batch_stage(|stage| {
                let mut held_guards: Vec<std::sync::MutexGuard<'_, ()>> = Vec::with_capacity(MINI);
                let mut held_stripes = [0u64; 8]; // 512 bits == Made::LOCKS
                let mut seen_slots: Vec<u32> = Vec::with_capacity(MINI);
                let mut copy_batch = DescriptorCopyBatch::new();
                let mut stage_used = 0u32;
                let mut publish: Vec<(u32, u64, Option<ID3D12Resource>, bool, bool)> = Vec::with_capacity(MINI);
                let mut stale_all: Vec<(u32, Vec<(u32, u64)>)> = Vec::new();

                let flush_mini = |copy_batch: &mut DescriptorCopyBatch,
                                  publish: &mut Vec<(u32, u64, Option<ID3D12Resource>, bool, bool)>,
                                  held_guards: &mut Vec<std::sync::MutexGuard<'_, ()>>,
                                  held_stripes: &mut [u64; 8],
                                  seen_slots: &mut Vec<u32>,
                                  stage_used: &mut u32,
                                  stale_all: &mut Vec<(u32, Vec<(u32, u64)>)>| {
                    copy_batch.flush();
                    for (idx, new_key, kept_res, update_kept, is_window) in publish.drain(..) {
                        if update_kept {
                            self.keep_window(idx, kept_res, is_window);
                        }
                        self.made.set(idx, new_key);
                        self.quick_is(idx, new_key);
                    }
                    held_guards.clear();
                    *held_stripes = [0u64; 8];
                    seen_slots.clear();
                    *stage_used = 0;
                    for (idx, stale) in stale_all.drain(..) {
                        self.unmake_stale(idx, stale);
                    }
                };

                for p in 0..pending_len {
                    let index = decoded[p].0;
                    let handle = decoded[p].1;
                    let key = decoded[p].2;
                    let view = decoded[p].3.take();
                    let out_idx = decoded[p].4;
                    let prefetched_shown = decoded[p].5;
                    let prefetched_old = decoded[p].6.take();

                    if seen_slots.contains(&index) || stage_used + 2 > BatchStage::CAP || copy_batch.remaining() < 2 {
                        flush_mini(
                            &mut copy_batch,
                            &mut publish,
                            &mut held_guards,
                            &mut held_stripes,
                            &mut seen_slots,
                            &mut stage_used,
                            &mut stale_all,
                        );
                    }
                    let stripe = (index as usize) % Made::LOCKS;
                    let bit = 1u64 << (stripe & 63);
                    if held_stripes[stripe >> 6] & bit == 0 {
                        if held_guards.is_empty() {
                            held_guards.push(self.made.lock(index));
                        } else if let Some(g) = self.made.try_lock(index) {
                            held_guards.push(g);
                        } else {
                            flush_mini(
                                &mut copy_batch,
                                &mut publish,
                                &mut held_guards,
                                &mut held_stripes,
                                &mut seen_slots,
                                &mut stage_used,
                                &mut stale_all,
                            );
                            held_guards.push(self.made.lock(index));
                        }
                        held_stripes[stripe >> 6] |= bit;
                    }
                    seen_slots.push(index);

                    let shown = self.made.get(index).unwrap_or(0);
                    let Some(view) = view else {
                        if shown != 0 {
                            self.quick_is(index, 0);
                            if !self.queue_cached_null(index, 0, &mut copy_batch) {
                                let s0 = stage.slot(stage_used);
                                stage_used += 1;
                                null_view(s0);
                                self.queue_copy_from(index, s0, &mut copy_batch);
                            }
                            self.write_stride(index, 0, 0);
                            self.clear_twin_batched(index, &mut copy_batch);
                            let _ = self.note_buffer_inner(index, None, false, record_change, false);
                            publish.push((index, 0, None, true, false));
                        } else if for_draw {
                            let _ = self.note_buffer_inner(index, None, false, record_change, false);
                        }
                        continue;
                    };

                    let mut direct_copy = None;
                    let mut made_now = false;
                    if shown != key {
                        let waiting = if for_draw { None } else { view.waiting_texture() };
                        if let Some(texture) = waiting {
                            super::wait_for(texture, handle);
                        }
                        let later = !for_draw && (view.is_buffer() || waiting.is_some());
                        let was_buffer = later && view.is_buffer() && self.buffers.contains(index);
                        let placed_tex = matches!(view, View::Placed(false, r, _) if r[0] != 1);
                        let old = if shown != 0 && !for_draw && (placed_tex || (later && keep_views())) {
                            if shown == prefetched_shown && !view.is_buffer() {
                                prefetched_old
                            } else {
                                view_at(shown as u32, (shown >> 32) as u32)
                            }
                        } else {
                            None
                        };
                        let old_kin = old.as_ref().map(|old| kin(old, &view));
                        let stale = self.note_buffer_inner(
                            index,
                            Some(&view),
                            for_draw,
                            record_change,
                            old_kin == Some(Kin::Same),
                        );
                        // Deferred buffer views built by `heap_buffer_pass` (or on draw):
                        // skip writing a throwaway null when the slot is already null or
                        // already holds a valid buffer descriptor; `made`/`quick` keep
                        // `shown != key` so the next draw or pass builds the real view.
                        if later && view.is_buffer() && (shown == 0 || was_buffer) {
                            if !stale.is_empty() {
                                stale_all.push((index, stale));
                            }
                            if want_results {
                                results[out_idx] = (index, Some(view), None, false);
                            }
                            continue;
                        }
                        let mut kept_old = false;
                        if later && keep_views() && let (Some(old), Some(k)) = (old.as_ref(), old_kin) {
                            if k == Kin::Same {
                                KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                kept_old = true;
                            } else if is_texture_view(old) {
                                let kept = crate::resource::room_lately() && k.kept(keep_shown());
                                rewritten(k, kept);
                                if kept {
                                    kept_old = true;
                                }
                            }
                        }
                        if kept_old {
                            if !stale.is_empty() {
                                stale_all.push((index, stale));
                            }
                            if want_results {
                                results[out_idx] = (index, Some(view), None, false);
                            }
                            continue;
                        }
                        #[cfg(feature = "diagnostics")]
                        if is_texture_view(&view) {
                            note_before(index, shown, if later { "null, waits" } else { "at once" });
                        }
                        self.quick_is(index, 0);
                        let has_twin = (index as usize) < self.twins.len();
                        let mut reserved_s0 = None;
                        let made = if later {
                            None
                        } else {
                            let s0 = stage.slot(stage_used);
                            let s1 = stage.slot(stage_used + 1);
                            stage_used += 2;
                            reserved_s0 = Some(s0);
                            let prev_slot = CURRENT_SLOT.replace(Some(index));
                            let res = make_twinned_with_copy(&view, s0, has_twin.then_some(s1))
                                .map(|(kept, sc, same_twin, is_window)| (kept, sc, s0, s1, same_twin, is_window));
                            CURRENT_SLOT.set(prev_slot);
                            res
                        };
                        match made {
                            Some((kept, sc, s0, s1, same_twin, is_window)) => {
                                self.queue_copy_from(index, s0, &mut copy_batch);
                                if twinned(&view) && has_twin {
                                    let twin_src = if same_twin { s0 } else { s1 };
                                    self.queue_copy_from(self.raw + index, twin_src, &mut copy_batch);
                                    self.twins.set(index as usize, true);
                                } else {
                                    self.clear_twin_batched(index, &mut copy_batch);
                                }
                                direct_copy = sc;
                                made_now = true;
                                publish.push((index, key, kept, true, is_window));
                            }
                            None => {
                                let mut tlas_null_src = None;
                                if let Some(kind) = cached_null_kind(&view)
                                    && self.queue_cached_null(index, kind, &mut copy_batch)
                                {
                                    // Queued from cached null heap.
                                } else {
                                    let s0 = reserved_s0.unwrap_or_else(|| {
                                        let s = stage.slot(stage_used);
                                        stage_used += 1;
                                        s
                                    });
                                    null_for(&view, s0);
                                    self.queue_copy_from(index, s0, &mut copy_batch);
                                    tlas_null_src = Some(s0);
                                }
                                self.clear_or_null_tlas_twin_batched(index, &view, tlas_null_src, &mut copy_batch);
                                publish.push((index, if later { 0 } else { key }, None, shown != 0, false));
                            }
                        }
                        if !stale.is_empty() {
                            stale_all.push((index, stale));
                        }
                    }
                    if want_results {
                        results[out_idx] = (index, Some(view), direct_copy, made_now);
                    }
                }
                flush_mini(
                    &mut copy_batch,
                    &mut publish,
                    &mut held_guards,
                    &mut held_stripes,
                    &mut seen_slots,
                    &mut stage_used,
                    &mut stale_all,
                );
            });
        }
        results
    }

    fn mirror_inner(&self, index: u32, handle: u64, for_draw: bool, record_change: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        self.note_title_start(index, handle);
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        // Made from this record, and nothing has changed it: most of the
        // asks, from five recording threads, behind one lock while another
        // held it to make a descriptor (15 % of a recording thread's time in
        // the opening, profile cpu-1).
        if tag == VIEW_TAG
            && quick_mirror()
            && self.quick.get(index as usize, std::sync::atomic::Ordering::Acquire) == Some(key)
            && let Some(view) = view_at(i, generation)
        {
            return Some(view);
        }
        if tag != VIEW_TAG && !for_draw && self.made.get(index).unwrap_or(0) == 0 {
            return None;
        }
        let held = self.made.lock(index);
        let shown = self.made.get(index).unwrap_or(0);
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if shown != 0 {
                self.quick_is(index, 0);
                self.made.set(index, 0);
                self.make_null(index);
                let _ = self.note_buffer_inner(index, None, false, record_change, false);
            } else if for_draw {
                // Left for later and stale since: out of the sets and the stride table.
                let _ = self.note_buffer_inner(index, None, false, record_change, false);
            }
            return None;
        };
        if shown != key {
            // A view of a texture object that has no host copy yet waits
            // for it, as buffer views wait for a draw: the slot is made
            // when the texture is first used (`super::make_waiting`), or
            // here when a draw reads the slot through a table.
            let waiting = if for_draw { None } else { view.waiting_texture() };
            if let Some(texture) = waiting {
                super::wait_for(texture, handle);
            }
            let later = !for_draw && (view.is_buffer() || waiting.is_some());
            let was_buffer = later && view.is_buffer() && self.buffers.contains(index);
            let placed_tex = matches!(view, View::Placed(false, r, _) if r[0] != 1);
            let old = if shown != 0 && !for_draw && (placed_tex || (later && keep_views())) {
                view_at(shown as u32, (shown >> 32) as u32)
            } else {
                None
            };
            let old_kin = old.as_ref().map(|old| kin(old, &view));
            // Views of other textures in the memory this one describes
            // are stale: their host slots read nothing from here on, and
            // let go of the host copies they held.
            // (When this slot's lock is let go: a lock at a time.)
            let stale = self.note_buffer_inner(
                index,
                Some(&view),
                for_draw,
                record_change,
                old_kin == Some(Kin::Same),
            );
            if later && view.is_buffer() && (shown == 0 || was_buffer) {
                drop(held);
                self.unmake_stale(index, stale);
                return Some(view);
            }
            // Another view of the placed texture the slot shows already (a
            // title that streams a texture in writes its view again with
            // more mips each time some arrive): the slot goes on showing
            // the old view until a draw or the texture pass makes the new
            // one and uploads what it shows. Made null here, the texture
            // read as nothing for as long as it waited in the queue, every
            // time it got sharper.
            if later && keep_views() && let (Some(old), Some(kin)) = (old.as_ref(), old_kin) {
                if kin == Kin::Same {
                    KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    drop(held);
                    self.unmake_stale(index, stale);
                    return Some(view);
                }
                // And of another texture that can stand for it
                // ([`keep_shown`]): black, then grey for a frame or two is
                // what popped.
                if is_texture_view(old) {
                    let kept = crate::resource::room_lately() && kin.kept(keep_shown());
                    rewritten(kin, kept);
                    if kept {
                        drop(held);
                        self.unmake_stale(index, stale);
                        return Some(view);
                    }
                }
            }
            #[cfg(feature = "diagnostics")]
            if !self.sampler && is_texture_view(&view) {
                note_before(index, shown, if later { "null, waits" } else { "at once" });
            }
            self.quick_is(index, 0);
            self.made.set(index, if later { 0 } else { key });
            let made = if later {
                None
            } else {
                let prev_slot = CURRENT_SLOT.replace(Some(index));
                let res = make_twinned_info(&view, self.at(index), self.twin(index));
                CURRENT_SLOT.set(prev_slot);
                res
            };
            let (kept, same_twin, is_window) = match made {
                Some((kept, same_twin, is_window)) => (Some(kept), same_twin, is_window),
                None => (None, false, false),
            };
            let mut put_done = false;
            if kept.is_none() && !self.sampler {
                if let Some(raw_buf) = cached_null_kind(&view)
                    && self.copy_cached_null(index, raw_buf)
                {
                    put_done = true;
                } else {
                    null_for(&view, self.at(index));
                }
            }
            if !put_done && (kept.is_some() || !self.sampler) {
                self.put(index);
            }
            if kept.is_some() && twinned(&view) {
                if same_twin {
                    self.twin_made_same(index);
                } else {
                    self.twin_made(index, true);
                }
            } else {
                self.clear_or_null_tlas_twin(index, &view);
            }
            if kept.is_some() || shown != 0 {
                self.keep_window(index, kept.flatten(), is_window);
            }
            self.quick_is(index, if later { 0 } else { key });
            drop(held);
            self.unmake_stale(index, stale);
        }
        Some(view)
    }

    /// Whether slot `index` was made from no record: what [`HostHeap::mirror`]
    /// would find before it did nothing for one that holds no view.
    pub(crate) fn shows_none(&self, index: u32) -> bool {
        self.made.get(index).unwrap_or(0) == 0
    }

    /// Slot `index` is made from record `key` and stays so (0: not). Under
    /// the lock of `made`.
    fn quick_is(&self, index: u32, key: u64) {
        self.quick.set(index as usize, key, std::sync::atomic::Ordering::Release);
    }

    /// Whether slot `index` is backed by a made host descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.made.get(index).is_some_and(|key| key != 0)
    }

    /// Whether host slot `index` shows `buffer`; `None` if it shows no
    /// resource (diagnostics: a buffer view's slot must show the host copy
    /// that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let shown = self.kept_in(index)?;
        Some(windows::core::Interface::as_raw(&shown) == windows::core::Interface::as_raw(buffer))
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self.made.get(index)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Slots of this thread's own in heaps the card does not see, two of views
/// and two of samplers (`which` 0 or 1): descriptors are made there and
/// copied whole into the heaps it does see. Made in place they are seen half
/// written: an RX 7900 XTX saw a texture of no size in 4 % of its reads of
/// slots whose views a thread made over and over (the driver clears the
/// descriptor and fills it step by step), and none in 8,500 million with the
/// same descriptors copied, three times as fast
/// (`indirect::tests::descriptors_written_under_readers`). Title threads and
/// our passes rewrite slots while submissions run whose shaders pick them;
/// three device losses of 2026-11-25 (play-1125-0203, n02, n03) were reads of
/// memory no view names. `VICTORY_VIEW_COPY=0`: made in place, as before.
fn staged(sampler: bool, which: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
    /// Threads at a time that have slots; more make views in place.
    const THREADS: u32 = 4096;
    struct Stage {
        starts: [usize; 2],
        steps: [usize; 2],
        _heaps: [ID3D12DescriptorHeap; 2],
    }
    // SAFETY: descriptor heaps are free-threaded; each thread has slots of
    // its own.
    unsafe impl Send for Stage {}
    // SAFETY: as above.
    unsafe impl Sync for Stage {}
    static STAGE: std::sync::OnceLock<Option<Stage>> = std::sync::OnceLock::new();
    /// Places given back, and how many were ever given out.
    static PLACES: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));
    struct Place(Option<u32>);
    impl Drop for Place {
        fn drop(&mut self) {
            if let Some(place) = self.0 {
                PLACES.lock().unwrap_or_else(|e| e.into_inner()).0.push(place);
            }
        }
    }
    thread_local! {
        static PLACE: Place = {
            let mut places = PLACES.lock().unwrap_or_else(|e| e.into_inner());
            Place(places.0.pop().or_else(|| {
                (places.1 < THREADS).then(|| {
                    places.1 += 1;
                    places.1 - 1
                })
            }))
        };
    }
    let stage = STAGE
        .get_or_init(|| {
            if std::env::var_os("VICTORY_VIEW_COPY").is_some_and(|v| v == "0") {
                return None;
            }
            let host = host()?;
            let make = |kind: D3D12_DESCRIPTOR_HEAP_TYPE| -> Option<ID3D12DescriptorHeap> {
                // SAFETY: plain heap creation.
                unsafe {
                    host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: 2 * THREADS,
                        ..Default::default()
                    })
                }
                .inspect_err(|e| tracing::warn!(error = %e, "no heap to make views in: they are made in place"))
                .ok()
            };
            let kinds = [D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER];
            let heaps = [make(kinds[0])?, make(kinds[1])?];
            // SAFETY: live heaps, plain queries.
            let (starts, steps) = unsafe {
                (
                    [0, 1].map(|i: usize| heaps[i].GetCPUDescriptorHandleForHeapStart().ptr),
                    kinds.map(|kind| host.device.GetDescriptorHandleIncrementSize(kind) as usize),
                )
            };
            Some(Stage { starts, steps, _heaps: heaps })
        })
        .as_ref()?;
    // (A thread that is ending has none any more.)
    let place = PLACE.try_with(|p| p.0).ok().flatten()?;
    let kind = usize::from(sampler);
    if !sampler {
        let base = stage.starts[0] + (2 * place) as usize * stage.steps[0];
        STAGED_RANGE.with(|r| r.set((base, base + 2 * stage.steps[0])));
    }
    Some(D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: stage.starts[kind] + (2 * place + which) as usize * stage.steps[kind],
    })
}

thread_local! {
    /// Per-thread CPU-only descriptor address ranges (`[start, end)`) for
    /// `staged()`, `BatchStage`, and `ListViews`, allowing lock-free O(1)
    /// validation in [`is_staged_twin`] (even while `LIST_VIEWS` is borrowed)
    /// and fixing 93%-failing `move_raw` (`CPU-23`).
    static STAGED_RANGE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static BATCH_STAGE_RANGE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    static LIST_VIEWS_RANGE: std::cell::Cell<(usize, usize)> = const { std::cell::Cell::new((0, 0)) };
    /// Active descriptor heap slot index being created on this thread (for Probe C logging).
    static CURRENT_SLOT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Per-thread CPU-only CBV/SRV/UAV descriptor staging block (`64` slots = up to
/// 32 main+twin pairs) for batched `mirror_batch` / `mirror_heap_pass_batch`.
struct BatchStage {
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: usize,
}

impl BatchStage {
    const CAP: u32 = 64;

    fn new() -> Option<Self> {
        if std::env::var_os("VICTORY_VIEW_COPY").is_some_and(|v| v == "0") {
            return None;
        }
        let host = host()?;
        // SAFETY: plain CPU-only descriptor heap owned by this thread.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: Self::CAP,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .ok()?
        };
        // SAFETY: live heap, plain queries.
        let (start, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize,
            )
        };
        BATCH_STAGE_RANGE.with(|r| r.set((start, start + Self::CAP as usize * step)));
        Some(Self { _heap: heap, start, step })
    }

    #[inline]
    fn slot(&self, i: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.start + i as usize * self.step,
        }
    }
}

thread_local! {
    static BATCH_STAGE: std::cell::RefCell<Option<Option<BatchStage>>> = const { std::cell::RefCell::new(None) };
}

fn batch_stage_available() -> bool {
    BATCH_STAGE
        .try_with(|cell| {
            let mut slot = cell.borrow_mut();
            slot.get_or_insert_with(BatchStage::new).is_some()
        })
        .unwrap_or(false)
}

/// Scoped descriptor-table batcher pairing a [`DescriptorCopyBatch`] with a
/// cursor into this thread's 64-slot [`BatchStage`]. Flushes automatically when
/// staging slots or copy capacity run out, and on drop.
pub(crate) struct TableBatch<'a> {
    pub(crate) copy: DescriptorCopyBatch,
    stage: Option<&'a BatchStage>,
    used: u32,
}

impl TableBatch<'_> {
    #[inline]
    fn alloc_stage(&mut self, slots: u32) -> Option<D3D12_CPU_DESCRIPTOR_HANDLE> {
        let stage = self.stage?;
        if self.used + slots > BatchStage::CAP || self.copy.remaining() < slots as usize {
            self.copy.flush();
            self.used = 0;
        }
        let at = self.used;
        self.used += slots;
        Some(stage.slot(at))
    }

    #[inline]
    fn alloc_stage_pair(&mut self, slots: u32) -> Option<(D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE)> {
        let stage = self.stage?;
        if self.used + slots > BatchStage::CAP || self.copy.remaining() < slots as usize {
            self.copy.flush();
            self.used = 0;
        }
        let at = self.used;
        self.used += slots;
        Some((stage.slot(at), stage.slot(at + slots.saturating_sub(1))))
    }
}

pub(crate) fn with_table_batch<R>(f: impl FnOnce(&mut TableBatch<'_>) -> R) -> R {
    let mut f = Some(f);
    BATCH_STAGE
        .try_with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(BatchStage::new());
            }
            let borrow = cell.borrow();
            let mut batch = TableBatch {
                copy: DescriptorCopyBatch::new(),
                stage: borrow.as_ref().and_then(|s| s.as_ref()),
                used: 0,
            };
            let out = (f.take().unwrap())(&mut batch);
            batch.copy.flush();
            out
        })
        .unwrap_or_else(|_| {
            let mut batch = TableBatch {
                copy: DescriptorCopyBatch::new(),
                stage: None,
                used: 0,
            };
            let out = (f.take().unwrap())(&mut batch);
            batch.copy.flush();
            out
        })
}

fn with_batch_stage<R>(f: impl FnOnce(&BatchStage) -> R) -> Option<R> {
    BATCH_STAGE
        .try_with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(BatchStage::new());
            }
            let borrow = cell.borrow();
            borrow.as_ref().and_then(|s| s.as_ref()).map(f)
        })
        .ok()
        .flatten()
}

#[inline]
fn is_staged_twin(twin: D3D12_CPU_DESCRIPTOR_HANDLE) -> bool {
    let p = twin.ptr;
    let in_range = |cell: &'static std::thread::LocalKey<std::cell::Cell<(usize, usize)>>| {
        cell.try_with(|r| {
            let (lo, hi) = r.get();
            p >= lo && p < hi
        })
        .unwrap_or(false)
    };
    in_range(&LIST_VIEWS_RANGE) || in_range(&BATCH_STAGE_RANGE) || in_range(&STAGED_RANGE)
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The unsigned integer format with the layout of a 16-bit float or
/// signed integer format (one, two or four channels), if `format` is one.
pub(crate) fn half_format(format: u32) -> Option<u32> {
    match format {
        // R16G16B16A16_FLOAT and _SINT: R16G16B16A16_UINT.
        10 | 14 => Some(12),
        // R16G16_FLOAT and _SINT: R16G16_UINT.
        34 | 38 => Some(36),
        // R16_FLOAT and _SINT: R16_UINT.
        54 | 59 => Some(57),
        _ => None,
    }
}

/// `view` as a shader is shown it where it moves a buffer's 16-bit
/// channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`): a buffer view in a
/// 16-bit float or signed format, in the unsigned format of the same
/// layout, through which every 16-bit pattern is read and stored as it
/// is. `None` for every other view: the title's own serves (an unsigned
/// one already carries the bits; a texture can only be viewed in the
/// formats it was made for).
pub(crate) fn half_view(view: &View) -> Option<View> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    match view {
        View::Shader(resource, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Shader(*resource, Some(desc)))
        }
        View::Unordered(resource, counter, Some(desc)) if desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Unordered(*resource, *counter, Some(desc)))
        }
        View::Placed(unordered, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => {
            let mut desc = *desc;
            desc[0] = half_format(desc[0])?;
            Some(View::Placed(*unordered, *resource, desc))
        }
        _ => None,
    }
}

/// The format of `view`, a buffer view through which a shader moves
/// 16-bit channels as unsigned integers
/// (`victory_shader::emit::HostShader::half_views`), if those integers
/// are not what the console gives: a format of wider channels, a
/// normalised one or a signed 8-bit one, whose values the console converts
/// to and from halves. `None` for the formats the integers serve exactly
/// (16 bits a channel of any kind, through [`half_view`] where needed, and
/// unsigned channels of fewer bits; the title's are 16-bit unsigned) and
/// for any other view.
pub(crate) fn half_inexact(view: &View) -> Option<u32> {
    // Buffer views: dimension 1 in the title's description of either kind.
    const BUFFER: u32 = 1;
    // R16G16B16A16, R16G16 and R16 (float, unsigned, signed), then
    // R10G10B10A2, R8G8B8A8, R8G8 and R8 unsigned.
    const EXACT: [u32; 13] = [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62];
    let format = match view {
        View::Shader(_, Some(desc)) | View::Unordered(_, _, Some(desc)) if desc[1] == BUFFER => {
            desc[0]
        }
        View::Placed(_, resource, desc) if resource[0] == BUFFER && desc[1] == BUFFER => desc[0],
        _ => return None,
    };
    (!EXACT.contains(&format)).then_some(format)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The address a shader finds in a descriptor of title memory: where a
/// buffer view starts, or the ray-tracing structure of a structure's view
/// (no resource: the view's description holds the address, and the shader
/// walks the structure from there itself).
pub(crate) fn view_address(view: &View) -> Option<u64> {
    const STRUCTURE: u32 = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
    match view {
        View::Shader(_, Some(desc)) if desc[1] == STRUCTURE => {
            Some(u64::from(desc[4]) | u64::from(desc[5]) << 32)
        }
        _ => buffer_range(view).map(|range| range.0),
    }
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((view_home(shadow, address, len, view_elements(view)?), address, len));
    }
    let shadow = crate::shadow::find(address, len)?;
    Some((view_home(shadow, address, len, view_elements(view)?), address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// Raw twins ([`raw_twins`]) made since the last frame line, and those of them that
/// do not begin on 16 bytes, as a raw view must ([`move_raw`]), and those of
/// these that could not be moved there (null).
pub(crate) static TWINS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Moves the raw view at `at`, a slot of our heaps the card does not see,
/// which shows `bytes` from `start` on (GPU address), `by` bytes on: for a
/// twin whose view begins between two multiples of 16, where the host makes
/// no raw view. An AMD descriptor of a raw buffer is its address (48 bits),
/// its bytes and flags; only one that says just that is changed, and only
/// where what the handle points at is the descriptor. False: not moved.
pub(crate) fn move_raw(at: D3D12_CPU_DESCRIPTOR_HANDLE, start: u64, bytes: u32, by: u32) -> bool {
    // (Off dwords too, as the typed view it stands for begins there: a view
    // of 16 bit values at an odd one.)
    if by >= bytes {
        return false;
    }
    let mut words = [0u8; 16];
    if !crate::host::read_into(at.ptr as *const u8, &mut words) {
        return false;
    }
    let word = |i: usize| u32::from_le_bytes([words[4 * i], words[4 * i + 1], words[4 * i + 2], words[4 * i + 3]]);
    let address = u64::from(word(0)) | u64::from(word(1)) << 32;
    if address != start || word(2) != bytes || start >> 48 != 0 {
        return false;
    }
    let moved = start + u64::from(by);
    // SAFETY: this thread's own slot (it has slots of its own: the caller
    // asks only for those), read just now.
    unsafe {
        let words = at.ptr as *mut u32;
        words.write(moved as u32);
        words.add(1).write((moved >> 32) as u32);
        words.add(2).write(bytes - by);
    }
    true
}

/// The bytes a buffer view the host takes ([`buffer_view_ok`]) shows as a raw
/// view (first dword, dwords, and the bytes past that dword it begins: a raw
/// view begins on 16 bytes); of a buffer of `width` bytes.
fn twin_window(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> Option<(u64, u32, u32)> {
    use std::sync::atomic::Ordering::Relaxed;
    TWINS[0].fetch_add(1, Relaxed);
    let element = if raw {
        4
    } else if stride > 0 {
        u64::from(stride)
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let (start, end) = (first * element, ((first + u64::from(count)) * element).next_multiple_of(4));
    let end = end.min(width & !3);
    let from = start & !15;
    if end <= start {
        return None;
    }
    let dwords = u32::try_from((end - from) / 4).ok()?;
    if from != start {
        TWINS[1].fetch_add(1, Relaxed);
    }
    Some((from / 4, dwords, (start - from) as u32))
}

/// Probe C (`ARCH-16`): logs full descriptor metadata on the first 16
/// `offset % element != 0` warnings to distinguish sub-4-byte views from torn reads.
fn log_probe_c_misaligned(
    unordered: bool,
    address: u64,
    len: u64,
    format: u32,
    stride: u32,
    raw: bool,
    first_element: u64,
    num_elements: u32,
    offset: u64,
    element: u64,
) {
    static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n < 16 {
        let slot = CURRENT_SLOT.try_with(|c| c.get()).ok().flatten();
        tracing::warn!(
            probe_c = n + 1,
            unordered,
            ?slot,
            address = format_args!("{address:#x}"),
            len,
            format,
            stride,
            raw,
            first_element,
            num_elements,
            offset,
            element,
            "Probe C: buffer view not on an element of its host copy (offset % element != 0)"
        );
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<(Kept, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>, bool, bool)> {
    let mut shadow = crate::shadow::find(address, len)?;
    let mut offset = shadow.offset(address);
    let unit = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let own = unsafe { &mut desc.Anonymous.Buffer };
    let start = address + own.FirstElement * unit;
    let range_len = u64::from(own.NumElements) * unit;
    let kind = (own.StructureByteStride, own.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, desc.Format.0 as u32);
    let home = view_home(shadow.clone(), start, range_len, kind);
    if !std::sync::Arc::ptr_eq(&home, &shadow) {
        // Its own bytes: it begins where they do.
        (shadow, offset) = (home, 0);
        own.FirstElement = 0;
    }
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            log_probe_c_misaligned(
                false,
                address,
                len,
                desc.Format.0 as u32,
                buffer.StructureByteStride,
                buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0,
                buffer.FirstElement,
                buffer.NumElements,
                offset,
                element,
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    let width = shadow.width();
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    // Already-raw & `raw_window`-converted views have bit-identical primary and
    // raw-twin descriptors (`CPU-23`): skip the duplicate `CreateShaderResourceView`
    // when `at` lives in a CPU-only staging heap so callers copy directly from `at`.
    let mut same_twin = false;
    if let Some(twin) = twin {
        if raw && is_staged_twin(at) {
            TWINS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            same_twin = true;
        } else {
            let window = twin_window(desc.Format, first, count, stride, raw, width);
            if let Some((first, count, by)) = window {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
                    FirstElement: first,
                    NumElements: count,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                };
                // SAFETY: a live buffer and a slot of our heap.
                unsafe {
                    host.device
                        .CreateShaderResourceView(&shadow.buffer, Some(&desc), twin);
                }
                let start = shadow.buffer_gpu() + first * 4;
                if by > 0 && !(is_staged_twin(twin) && move_raw(twin, start, count.saturating_mul(4), by)) {
                    TWINS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    null_buffer_view(twin);
                }
            } else {
                null_buffer_view(twin);
            }
        }
    }
    let kept = Some(shadow.buffer.clone());
    let is_window = shadow.is_window();
    let copy = (range_len > 0 && view_extent() && start >= address && start - address < len.max(1))
        .then_some((shadow, start, range_len));
    Some((kept, copy, same_twin, is_window))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<(Kept, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>, bool, bool)> {
    let orig_address = address;
    let mut shadow = crate::shadow::find(address, len)?;
    let unit = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    };
    // SAFETY: a buffer view's union member.
    let own = unsafe { &mut desc.Anonymous.Buffer };
    let start = address + own.FirstElement * unit;
    let range_len = u64::from(own.NumElements) * unit;
    let kind = (own.StructureByteStride, own.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, desc.Format.0 as u32);
    let home = view_home(shadow.clone(), start, range_len, kind);
    let mut address = address;
    if !std::sync::Arc::ptr_eq(&home, &shadow) {
        // Its own bytes: it begins where they do.
        (shadow, address) = (home, start);
        own.FirstElement = 0;
    }
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            log_probe_c_misaligned(
                true,
                address,
                len,
                desc.Format.0 as u32,
                buffer.StructureByteStride,
                buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0,
                buffer.FirstElement,
                buffer.NumElements,
                offset,
                element,
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    let width = shadow.width();
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    let mut same_twin = false;
    if let Some(twin) = twin {
        if raw && is_staged_twin(at) {
            TWINS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            same_twin = true;
        } else {
            let window = twin_window(desc.Format, first, count, stride, raw, width);
            if let Some((first, count, by)) = window {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Anonymous.Buffer = D3D12_BUFFER_UAV {
                    FirstElement: first,
                    NumElements: count,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                };
                // SAFETY: a live buffer and a slot of our heap.
                unsafe {
                    host.device
                        .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), twin);
                }
                let start = shadow.buffer_gpu() + first * 4;
                if by > 0 && !(is_staged_twin(twin) && move_raw(twin, start, count.saturating_mul(4), by)) {
                    TWINS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    null_buffer_view(twin);
                }
            } else {
                null_buffer_view(twin);
            }
        }
    }
    let kept = Some(shadow.buffer.clone());
    let is_window = shadow.is_window();
    let copy = (range_len > 0 && view_extent() && start >= orig_address && start - orig_address < len.max(1))
        .then_some((shadow, start, range_len));
    Some((kept, copy, same_twin, is_window))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(
    texture: &ID3D12Resource,
    castable: &[DXGI_FORMAT],
    mut format: Option<&mut DXGI_FORMAT>,
    unordered: bool,
) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    // A texture created with D3D12 relaxed format casting allows views in any
    // format in its castable list, keeping the requested view format intact
    // (PC forbids R9G9B9E5_SHAREDEXP UAVs, so keep it SRV-only).
    if (!unordered || view != DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0 as u32)
        && castable.iter().any(|f| f.0 as u32 == view)
    {
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The bytes of the host's constant buffer view for a title's of `size`:
/// whole 256s, and at most the 64 KiB a shader can index. The host removes
/// the device for a larger view, and the title's heap holds such sizes in
/// slots it does not use.
fn constant_bytes(size: u32) -> u32 {
    const LARGEST: u32 = 64 << 10;
    size.min(LARGEST).next_multiple_of(256)
}

/// Renderer port 5 ([`crate::port_cbv`]): the constant buffer of `size` bytes
/// at `gpu` of `buffer` as a constant buffer view at `at` where it can be (on
/// 256 bytes; else a null one) and a raw view of its bytes at `twin`, which
/// translated shaders read. `false` (nothing made): not on 16 bytes.
fn constant_twinned(
    buffer: &ID3D12Resource,
    gpu: u64,
    size: u32,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> bool {
    let Some(host) = host() else { return false };
    // SAFETY: a live buffer.
    let (base, width) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc().Width) };
    let Some(layout) = crate::port_cbv::raw_cbv_layout(base, width, gpu, size) else {
        return false;
    };
    let cbv = D3D12_CONSTANT_BUFFER_VIEW_DESC {
        BufferLocation: layout.cbv_location,
        SizeInBytes: layout.cbv_size,
    };
    // SAFETY: slots of our heaps; the ranges lie in the live buffer.
    unsafe {
        host.device.CreateConstantBufferView(Some(&cbv), at);
        if let Some(twin) = twin {
            let raw = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: layout.first_element,
                        NumElements: layout.num_elements,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            };
            host.device.CreateShaderResourceView(buffer, Some(&raw), twin);
            crate::port_cbv::twinned();
        }
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    make_twinned(view, at, None)
}

/// [`make`]; `twin`: where the twin slot goes ([`raw_twins`]: raw buffer twin,
/// constant buffer twin, or mirror TLAS SRV for `ARCH-13`).
fn make_twinned(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<Kept> {
    make_twinned_with_copy(view, at, twin).map(|(kept, _, _, _)| kept)
}

/// [`make_twinned`], also returning `same_twin` (`true` when the twin descriptor
/// is bit-identical to `at` in a CPU-only staging heap and needs no second creation)
/// and `is_window` (`true` when backed by a `window == true` loose shadow).
fn make_twinned_info(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<(Kept, bool, bool)> {
    make_twinned_with_copy(view, at, twin).map(|(kept, _, same_twin, is_window)| (kept, same_twin, is_window))
}

/// [`make_twinned`], also returning any resolved `(Shadow, address, len)` for
/// buffer views so `heap_buffer_pass` skips a duplicate `buffer_copy` lookup,
/// `same_twin` (`true` when callers can copy the twin directly from `at`), and
/// `is_window` (`true` when backed by a `window == true` loose shadow).
fn make_twinned_with_copy(
    view: &View,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
    twin: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
) -> Option<(Kept, Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>, bool, bool)> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            let is_window = shadow.is_window();
            if crate::port_cbv::on() {
                // Renderer port 5: shaders read constant buffers as raw bytes
                // through the twin; a constant buffer view there would be read
                // raw, with no end on AMD.
                return constant_twinned(&shadow.buffer, gpu, size, at, twin)
                    .then(|| (Some(shadow.buffer.clone()), None, false, is_window));
            }
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_bytes(size),
            };
            // SAFETY: slots of our heap; the range lies in the host copy.
            unsafe {
                device.CreateConstantBufferView(Some(&desc), at);
                if let Some(twin) = twin {
                    device.CreateConstantBufferView(Some(&desc), twin);
                }
            }
            Some((Some(shadow.buffer.clone()), None, false, is_window))
        }
        View::Shader(resource, desc) => {
            if *resource == 0
                && let Some(desc) = desc
                && desc[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32
                && crate::dxr::enabled()
            {
                let guest_tlas_va = u64::from(desc[4]) | (u64::from(desc[5]) << 32);
                let Some((host_tlas_gpu_va, slab_buf)) = crate::dxr::tlas_srv(guest_tlas_va) else {
                    return None;
                };
                let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_UNKNOWN,
                    ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                            Location: host_tlas_gpu_va,
                        },
                    },
                };
                // SAFETY: a slot of our heap; `slab_buf` is kept alive via `self.keep`.
                unsafe {
                    device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&srv_desc), at);
                }
                if let Some(twin) = twin {
                    let twin_va = twin_tlas_va(guest_tlas_va).unwrap_or(host_tlas_gpu_va);
                    let twin_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: DXGI_FORMAT_UNKNOWN,
                        ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV {
                                Location: twin_va,
                            },
                        },
                    };
                    // SAFETY: twin slot of our heap for 2-pass hardware FrontBackTrack (`ARCH-13`).
                    unsafe {
                        device.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&twin_desc), twin);
                    }
                }
                return Some((Some(slab_buf), None, false, false));
            }
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                let texture = &copy.texture;
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(
                    texture,
                    copy.castable_formats(),
                    desc.as_mut().map(|d| &mut d.Format),
                    false,
                ) {
                    return None;
                }
                if let Some(first) = crate::resource::first_shown(resource)
                    && first > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
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
                    crate::tiles::show_from(desc.get_or_insert(whole), first, u32::from(own.MipLevels));
                }
                if copy.base > 0
                    && let Some(desc) = &mut desc
                {
                    // SAFETY: a live resource.
                    let own_mips = u32::from(unsafe { texture.GetDesc() }.MipLevels);
                    from_base(desc, copy.base, own_mips);
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some((Some(texture.clone()), None, false, false))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at, twin)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at, twin)
                    }
                    _ => None,
                };
            };
            let texture = &copy.texture;
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let (texture, castable) = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    (twin, &[][..])
                }
                None => (texture, copy.castable_formats()),
            };
            if !fits(texture, castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some((Some(texture.clone()), None, false, false))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some((None, None, false, false))
        }
        View::ConsoleSampler(words) => {
            let desc = console_sampler(words, aniso_point_mip(device));
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some((None, None, false, false))
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at).map(|kept| (kept, None, false, false));
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                let at = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                if *unordered || at != 0 {
                    static SAID: Mutex<std::collections::BTreeSet<u64>> = Mutex::new(std::collections::BTreeSet::new());
                    static TIMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let times = TIMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
                    if (said.len() < 64 && said.insert(at)) || times.is_power_of_two() {
                        tracing::warn!(
                            address = format_args!("{at:#x}"),
                            unordered = *unordered,
                            times,
                            "placed texture view not mirrored (null): resource {resource:?}, view {:?}",
                            &desc[..10]
                        );
                    }
                }
                return None;
            }
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at, twin)
            } else {
                buffer_srv(address, len, srv_desc(words), at, twin)
            }
        }
    }
}

/// Whether the host GPU filters anisotropically with point mips
/// (`D3D12_OPTIONS19`, optional): asked once. Older runtimes do not know
/// the query, which means no.
fn aniso_point_mip(device: &ID3D12Device) -> bool {
    static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *SUPPORTED.get_or_init(|| {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        // SAFETY: the feature query writes the structure of the given size.
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.AnisoFilterWithPointMipSupported.as_bool()
    })
}

/// Console sampler description for the host, keeping point mips on anisotropic
/// samplers where the host supports it (`#2031`, matching static samplers).
fn console_sampler(words: &[u32; 27], aniso_point_mip: bool) -> D3D12_SAMPLER_DESC {
    let mut desc = crate::sampler::console(words);
    let aniso = words[1] >= 2 || words[2] >= 2;
    if aniso && aniso_point_mip && words[3] != 2 {
        desc.Filter.0 &= !1;
    }
    desc
}

/// The view `desc` of a texture, of a host copy of its levels from `base`
/// on (`crate::resource::held_only`). What it shows of larger ones, if any,
/// is left out, and the shifted range is clamped to `own_mips`.
fn from_base(desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC, base: u32, own_mips: u32) {
    let clamp_mips = |most: &mut u32, levels: &mut u32, lod: &mut f32| {
        let lost = base.saturating_sub(*most);
        *most = most.saturating_sub(base);
        if own_mips > 0 {
            *most = (*most).min(own_mips - 1);
        }
        if *levels != u32::MAX {
            *levels = levels.saturating_sub(lost).max(1);
            if own_mips > 0 {
                *levels = (*levels).min(own_mips.saturating_sub(*most)).max(1);
            }
        }
        *lod = (*lod - base as f32).max(0.0);
    };
    match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: the member its dimension says.
            let t = unsafe { &mut desc.Anonymous.Texture2D };
            clamp_mips(&mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: the member its dimension says.
            let t = unsafe { &mut desc.Anonymous.Texture2DArray };
            clamp_mips(&mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURECUBE => {
            // SAFETY: the member its dimension says.
            let t = unsafe { &mut desc.Anonymous.TextureCube };
            clamp_mips(&mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
        }
        D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => {
            // SAFETY: the member its dimension says.
            let t = unsafe { &mut desc.Anonymous.TextureCubeArray };
            clamp_mips(&mut t.MostDetailedMip, &mut t.MipLevels, &mut t.ResourceMinLODClamp);
        }
        _ => {}
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// What the texture view a slot is written with has to do with the one it
/// shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kin {
    /// A view of the same placed texture ([`same_placed_texture`]).
    Same,
    /// A texture of the same description placed elsewhere (moved, or one of
    /// its kind).
    Moved,
    /// One of the same format and shape in another size: the same texture
    /// with more or fewer levels, as a title streams it.
    Sized,
    /// A view of the same dimension, of another texture.
    Other,
    /// A view no shader that reads the one can read.
    None,
}

impl Kin {
    /// Whether the slot goes on showing what it shows, by [`keep_shown`].
    fn kept(self, keep: u8) -> bool {
        match self {
            Kin::Same => true,
            Kin::Moved | Kin::Sized => keep >= 1,
            Kin::Other => keep >= 2,
            Kin::None => false,
        }
    }
}

/// [`Kin`] of the view `old`, which a slot shows, and `new`.
fn kin(old: &View, new: &View) -> Kin {
    if same_placed_texture(old, new) {
        return Kin::Same;
    }
    if let (View::Placed(false, a, from), View::Placed(false, b, to)) = (old, new) {
        if a[0] == 1 || b[0] == 1 || from[1] != to[1] {
            return Kin::None;
        }
        // (Width, 64 bits, height, depth and levels, format: words 4 to 8.)
        if a[0] == b[0] && a[2..13] == b[2..13] {
            return Kin::Moved;
        }
        let (wide, high) = ([a[4], b[4]].map(u64::from), [a[6], b[6]].map(u64::from));
        return if from[0] == to[0] && a[0] == b[0] && wide[0] * high[1] == wide[1] * high[0] {
            Kin::Sized
        } else {
            Kin::Other
        };
    }
    let dimension = |view: &View| match view {
        View::Placed(false, resource, desc) if resource[0] != 1 => desc[1],
        View::Shader(r, Some(desc)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => desc[1],
        _ => 0,
    };
    match (dimension(old), dimension(new)) {
        (0, _) | (_, 0) => Kin::None,
        (a, b) if a == b => Kin::Other,
        _ => Kin::None,
    }
}

/// Whether a slot that is written with a view of another texture goes on
/// showing the one it shows until the new one is made and its upload has run
/// (`VICTORY_KEEP_SHOWN`): 0: never, it reads as zeros, then as the neutral
/// texture; 1: one of the same kind in another place or size; 2: any of the
/// same dimension.
fn keep_shown() -> u8 {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_KEEP_SHOWN") {
        return if on { 2 } else { 0 };
    }
    static KEEP: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var("VICTORY_KEEP_SHOWN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1))
}

/// Slots that showed a texture and were written with a view of another,
/// since the start, by [`Kin`] (moved, sized, other, none), and those of
/// them that kept what they showed.
static REWRITTEN: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

fn rewritten(kin: Kin, kept: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let class = match kin {
        Kin::Moved | Kin::Same => 0,
        Kin::Sized => 1,
        Kin::Other => 2,
        Kin::None => 3,
    };
    REWRITTEN[class].fetch_add(1, Relaxed);
    REWRITTEN[4].fetch_add(u64::from(kept), Relaxed);
    let n: u64 = REWRITTEN[..4].iter().map(|n| n.load(Relaxed)).sum();
    if n.is_multiple_of(200_000) {
        tracing::info!(
            "slots that showed a texture and were written with another, since the start: {} of the same kind elsewhere, {} of another size, {} others, {} of another dimension; {} of them kept what they showed meanwhile (VICTORY_KEEP_SHOWN)",
            REWRITTEN[0].load(Relaxed),
            REWRITTEN[1].load(Relaxed),
            REWRITTEN[2].load(Relaxed),
            REWRITTEN[3].load(Relaxed),
            REWRITTEN[4].load(Relaxed)
        );
    }
}

/// Diagnostics: whether a texture that a slot showed until its new one came
/// holds the new one's bytes (the title moved it), by the bytes their host
/// copies were filled from: equal, different, not known.
#[cfg(feature = "diagnostics")]
fn same_bytes(old: u64, new: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static SEEN: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
    let view = |key: u64| view_at(key as u32, (key >> 32) as u32);
    let (Some(old), Some(new)) = (view(old), view(new)) else {
        return;
    };
    let held = |view: &View| {
        let View::Placed(false, resource, desc) = view else {
            return None;
        };
        let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
        crate::resource::texture_held(crate::resource::texture_at_made(address, resource)?)
    };
    let class = match kin(&old, &new) {
        Kin::Moved => 0,
        Kin::Same | Kin::None => return,
        _ => 3,
    };
    let which = match (held(&old), held(&new)) {
        (Some(a), Some(b)) if a == b => 0,
        (Some(a), Some(b)) if a.0 == b.0 => 1,
        _ => 2,
    };
    SEEN[class + which].fetch_add(1, Relaxed);
    let n: u64 = SEEN.iter().map(|n| n.load(Relaxed)).sum();
    if n.is_multiple_of(1000) {
        let seen: Vec<u64> = SEEN.iter().map(|n| n.load(Relaxed)).collect();
        tracing::info!(
            "slots that went on showing another texture until the new one came, since the start: of one of the same kind elsewhere {} had the new one's bytes, {} others, {} not known; of others {}, {}, {}",
            seen[0], seen[1], seen[2], seen[3], seen[4], seen[5]
        );
    }
}

/// For `crate::popwatch`, since the start: slots made null as they were
/// written with another texture, such that kept the old one, slots given the
/// neutral texture, slots made null as their memory was another's.
#[cfg(feature = "diagnostics")]
pub(crate) fn pop_causes() -> [u64; 4] {
    use std::sync::atomic::Ordering::Relaxed;
    let all: u64 = REWRITTEN[..4].iter().map(|n| n.load(Relaxed)).sum();
    let kept = REWRITTEN[4].load(Relaxed);
    [all.saturating_sub(kept), kept, LATE[3].load(Relaxed), STALE[0].load(Relaxed)]
}

/// Parses `VICTORY_HEAP_LOCKS`: `Some(0)` forces the lock-free mirror path off,
/// `Some(1)` forces it on, `Some(2)` alternates every 20 seconds (`alt`),
/// `None` defers to `VICTORY_QUICK_MIRROR` (default on; `HostHeap::made` is
/// already 512-way sharded and `HostHeap::kept` is per-slot with outside-lock
/// retirement).
fn parse_heap_locks(val: Option<&str>) -> Option<u8> {
    match val.map(str::trim) {
        Some("0") => Some(0),
        Some("1") => Some(1),
        Some("alt") => Some(2),
        _ => None,
    }
}

/// Whether a slot already made from its record is told so without the lock
/// (`VICTORY_QUICK_MIRROR=0` or `VICTORY_HEAP_LOCKS=0`: with it;
/// `VICTORY_HEAP_LOCKS=alt`: twenty seconds with, twenty without).
fn quick_mirror() -> bool {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| {
        if let Some(m) = parse_heap_locks(std::env::var("VICTORY_HEAP_LOCKS").ok().as_deref()) {
            return m;
        }
        u8::from(std::env::var_os("VICTORY_QUICK_MIRROR").is_none_or(|v| v != "0"))
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the host heap's slots are found made without their lock from now (VICTORY_HEAP_LOCKS=alt): {}",
                    if on { "ON" } else { "off, lock as usual" }
                );
            }
            on
        }
    }
}

/// The records last seen of a heap, held ([`HostHeap::copied_begin`]).
pub(crate) struct Copied<'a> {
    at: *mut u8,
    len: usize,
    step: usize,
    _held: std::sync::MutexGuard<'a, Vec<u8>>,
}

// SAFETY: the bytes are reached through `changes` alone, whose callers take
// parts apart from one another; the guard stays with the thread that has it.
unsafe impl Sync for Copied<'_> {}

impl Copied<'_> {
    /// The slots of `records` (as the title has them now, from slot `first`
    /// on) that are not the ones last seen, which they are from here on.
    /// (`records`: as the caller read them, through the kernel, which fails on
    /// a page the title unmapped meanwhile where a read in place would fault:
    /// the title unmaps heap pages too.)
    ///
    /// # Safety
    /// No other call beside this one has any of these slots.
    pub(crate) unsafe fn changes(&self, first: u32, records: &[u8], out: &mut Vec<u32>) {
        let at = first as usize * self.step;
        let end = (at + records.len()).min(self.len);
        if at >= end {
            return;
        }
        // SAFETY: within the records held; the caller's slots alone.
        let last = unsafe { std::slice::from_raw_parts_mut(self.at.add(at), end - at) };
        changed_records(last, &records[..end - at], self.step, first, out);
    }

    /// As [`Copied::changes`] for the `count` records of title memory at
    /// `title` (slot `first` on), 32 bytes each, read where they are, each
    /// once ([`crate::safecopy::changed`]); `bits` is room. False when a page
    /// of them could not be read: those before it are done, the others
    /// stay as last seen.
    ///
    /// # Safety
    /// As [`Copied::changes`]; `title` is title memory (read by a copy that
    /// fails where a page is not there).
    pub(crate) unsafe fn changes_in_place(&self, first: u32, title: *const u8, count: usize, bits: &mut Vec<u64>, out: &mut Vec<u32>) -> bool {
        let at = first as usize * self.step;
        let count = count.min(self.len.saturating_sub(at) / 32);
        if self.step != 32 || count == 0 {
            return true;
        }
        bits.clear();
        bits.resize(count.div_ceil(64), 0);
        // SAFETY: within the records held; the caller's slots alone.
        let whole = unsafe { crate::safecopy::changed(title, self.at.add(at), count, bits) };
        for (n, &word) in bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                out.push(first + (n * 64) as u32 + word.trailing_zeros());
                word &= word - 1;
            }
        }
        whole
    }

    /// The `count` records last seen from slot `first` on (fewer where the
    /// heap ends).
    ///
    /// # Safety
    /// As [`Copied::changes`]: no other call beside this one has any of
    /// these slots.
    pub(crate) unsafe fn seen(&self, first: u32, count: usize) -> &[u8] {
        let at = (first as usize * self.step).min(self.len);
        let len = (count * self.step).min(self.len - at);
        // SAFETY: within the records held; the caller's slots alone.
        unsafe { std::slice::from_raw_parts(self.at.add(at), len) }
    }
}

/// The records (`step` bytes each, the first is slot `first`) that differ
/// between `last` and `records` (of one length) go into `out` and `last`.
/// Most of a changed page is as it was: it is looked at sixteen records at a
/// time.
fn changed_records(last: &mut [u8], records: &[u8], step: usize, first: u32, out: &mut Vec<u32>) {
    const AT_A_TIME: usize = 16;
    let block = AT_A_TIME * step;
    for (b, (old, new)) in last.chunks_mut(block).zip(records.chunks(block)).enumerate() {
        if old == new {
            continue;
        }
        for (n, (old, new)) in old.chunks_exact_mut(step).zip(new.chunks_exact(step)).enumerate() {
            if old != new {
                old.copy_from_slice(new);
                out.push(first + (b * AT_A_TIME + n) as u32);
            }
        }
    }
}

#[cfg(test)]
mod changed_records_tests {
    use super::changed_records;

    #[test]
    fn changed_records_are_those_that_differ_one_by_one() {
        for (step, count) in [(32usize, 128usize), (32, 2048), (16, 77), (32, 5), (32, 17)] {
            let mut seed = 7u64;
            let mut random = move || {
                seed = seed.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
                (seed >> 33) as usize
            };
            let mut last: Vec<u8> = (0..step * count).map(|_| random() as u8).collect();
            let mut now = last.clone();
            let mut differ = Vec::new();
            for slot in 0..count {
                if random() % 9 == 0 {
                    now[slot * step + random() % step] ^= 1 + (random() % 255) as u8;
                    differ.push(1000 + slot as u32);
                }
            }
            let mut out = Vec::new();
            changed_records(&mut last, &now, step, 1000, &mut out);
            assert_eq!(out, differ);
            assert_eq!(last, now);
            out.clear();
            changed_records(&mut last, &now, step, 1000, &mut out);
            assert!(out.is_empty());
        }
    }
}

/// Whether both are shader views of one texture the title placed in
/// memory: the same address and the same resource description, whatever
/// part of it each shows.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            // (Words 1 and 13 are the filler of the description's layout, not
            // part of what a texture is: `crate::resource::texture_at`.)
            let same = |from: usize, to: usize| resource[from..to] == other[from..to];
            resource[0] != 1 && resource[0] == other[0] && same(2, 13) && desc[10..12] == other_desc[10..12]
        }
        _ => false,
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use). When a live
/// host-written texture at `address` supports D3D12 relaxed format casting to
/// the requested SRV format, redirects directly to that source resource so no
/// alias twin texture or copy is needed.
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    if let Some(source) = crate::resource::relaxed_alias_target(address, resource, desc[0]) {
        let mut srv: [u32; 10] = desc[..10].try_into().ok()?;
        // Format=0 (DXGI_FORMAT_UNKNOWN) meant the placed texture's own format;
        // set it explicitly when redirecting to `source` (which has another format).
        if srv[0] == 0 {
            srv[0] = resource[8];
        }
        return Some(View::Shader(source as usize, Some(srv)));
    }
    let texture = crate::resource::texture_at(address, resource)?;
    crate::resource::want_level(texture, crate::resource::first_mip(desc));
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
/// Maps a constant null texture SRV dimension to its `CachedNulls` index, or
/// `None` when DXR requires a dynamic empty-TLAS acceleration-structure view.
fn cached_null_srv_dim(dim: u32) -> Option<usize> {
    if dim == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32 && crate::dxr::enabled() {
        return None;
    }
    Some(match dim {
        2 => 3,
        3 => 4,
        4 => 0,
        5 => 5,
        6 => 6,
        7 => 7,
        8 => 8,
        9 => 9,
        10 => 10,
        _ => 0,
    })
}

/// Maps a null texture UAV dimension to its `CachedNulls` index.
fn cached_null_uav_dim(dim: Option<u32>) -> usize {
    match dim.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => 12,
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => 13,
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => 15,
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => 16,
        _ => 14,
    }
}

/// Returns the `CachedNulls` index matching `null_for(view, ..)`, or `None`
/// only for DXR acceleration-structure views whose null descriptor points at
/// the live empty TLAS.
fn cached_null_kind(view: &View) -> Option<usize> {
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    match view {
        View::Unordered(r, _, desc) => {
            if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) {
                Some(11)
            } else {
                Some(cached_null_uav_dim(desc.map(|d| d[1])))
            }
        }
        View::Placed(true, resource, desc) => {
            if resource[0] == 1 {
                Some(11)
            } else {
                Some(cached_null_uav_dim(Some(desc[1])))
            }
        }
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => Some(1),
        View::Placed(false, resource, _) if resource[0] == 1 => Some(1),
        View::Shader(_, Some(desc)) => cached_null_srv_dim(desc[1]),
        View::Placed(false, _, desc) => cached_null_srv_dim(desc[1]),
        View::Constant(_, _) => Some(2),
        _ => Some(0),
    }
}

fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    if words[1] == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32 && crate::dxr::enabled() {
        let Some(host) = host() else { return };
        let location = crate::dxr::empty_tlas_srv().map_or(0, |(va, _)| va);
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: location },
            },
        };
        // SAFETY: a slot of our heap; points to our permanent 0-instance empty TLAS (or 0 if unavailable).
        unsafe {
            host.device
                .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at);
        }
        return;
    }
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_put_and_made_try_lock_pure_cpu() {
        let slots = Slots::new(130);
        assert!(!slots.contains(65));
        slots.put(65);
        slots.put(65);
        assert!(slots.contains(65));
        assert!(slots.remove(65));
        assert!(!slots.remove(65));

        let made = Made::new(1024);
        let g = made.lock(7);
        assert!(made.try_lock(7).is_none(), "same stripe must return WouldBlock");
        assert!(made.try_lock(7 + Made::LOCKS as u32).is_none(), "aliased stripe must return WouldBlock");
        assert!(made.try_lock(8).is_some(), "neighbour stripe must acquire lock-free");
        drop(g);
        assert!(made.try_lock(7).is_some(), "stripe must be free after drop");
    }

    /// A range of dynamic slots is given back only to the heap that took it:
    /// the title can release a heap while a list still holds slots of it,
    /// and a later heap can be made at the same address before that list is
    /// reset.
    #[cfg(windows)]
    #[test]
    fn dynamic_slots_are_given_back_only_to_the_heap_that_took_them() {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(a) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let Some(b) = HostHeap::new(false, 4) else {
            return;
        };
        assert_ne!(a.id(), b.id());
        let first = a.dynamic(2).expect("two dynamic slots");
        let free = a.free.load(Relaxed);
        a.release(b.id(), first, 2);
        assert_eq!(
            a.free.load(Relaxed),
            free,
            "another heap's identity gave the range back"
        );
        a.release(a.id(), first, 2);
        assert_eq!(
            a.free.load(Relaxed),
            free + 2,
            "their own heap did not get the range back"
        );
    }

    /// A view of a host copy that begins at a later level shows the same
    /// levels of it, across 2D, 2DArray, and Cube views, clamped to `own_mips`.
    #[test]
    fn views_count_from_the_first_level_of_the_host_copy() {
        let view = |first, levels, clamp, own_mips| {
            let mut desc = srv_desc(&[98, 4, 5768, 0, first, levels, 0, 0, 0, 0]);
            desc.Anonymous.Texture2D.ResourceMinLODClamp = clamp;
            from_base(&mut desc, 3, own_mips);
            // SAFETY: the member its dimension says.
            let t = unsafe { desc.Anonymous.Texture2D };
            (t.MostDetailedMip, t.MipLevels, t.ResourceMinLODClamp)
        };
        assert_eq!(view(3, 4, 0.0, 10), (0, 4, 0.0));
        assert_eq!(view(5, u32::MAX, 5.5, 10), (2, u32::MAX, 2.5));
        // (Two levels the copy has not.)
        assert_eq!(view(1, 4, 0.0, 10), (0, 2, 0.0));
        // Clamps MostDetailedMip and MipLevels to `own_mips`.
        assert_eq!(view(9, 4, 0.0, 4), (3, 1, 0.0));
        assert_eq!(view(4, 8, 0.0, 4), (1, 3, 0.0));

        // 2DArray (dimension 5) and Cube (dimension 9) shift & clamp identically.
        let mut array = srv_desc(&[98, 5, 5768, 0, 4, 6, 0, 2, 0, 0]);
        array.Anonymous.Texture2DArray.ResourceMinLODClamp = 4.5;
        from_base(&mut array, 3, 4);
        // SAFETY: the member its dimension says.
        let ta = unsafe { array.Anonymous.Texture2DArray };
        assert_eq!((ta.MostDetailedMip, ta.MipLevels, ta.ResourceMinLODClamp), (1, 3, 1.5));

        let mut cube = srv_desc(&[98, 9, 5768, 0, 2, 6, 0, 0, 0, 0]);
        cube.Anonymous.TextureCube.ResourceMinLODClamp = 3.25;
        from_base(&mut cube, 3, 4);
        // SAFETY: the member its dimension says.
        let tc = unsafe { cube.Anonymous.TextureCube };
        assert_eq!((tc.MostDetailedMip, tc.MipLevels, tc.ResourceMinLODClamp), (0, 4, 0.25));
    }

    #[test]
    fn remapped_coalesces_before_dropping_and_never_clears_all() {
        // Interleaved multi-threaded adjacent ranges coalesce with 0 drops.
        let mut ranges = Vec::new();
        let mut dropped = 0;
        for i in 0..16u64 {
            dropped += push_remapped(&mut ranges, (i * 0x1000, (i + 1) * 0x1000), 8);
            dropped += push_remapped(&mut ranges, (0x10_0000 + i * 0x1000, 0x10_0000 + (i + 1) * 0x1000), 8);
        }
        assert_eq!(dropped, 0);
        join(&mut ranges);
        assert_eq!(ranges, vec![(0, 0x1_0000), (0x10_0000, 0x11_0000)]);

        // Truly disjoint overflow keeps half + newest instead of clearing all.
        let mut disjoint = Vec::new();
        let mut total_dropped = 0;
        for i in 0..10u64 {
            total_dropped += push_remapped(&mut disjoint, (i * 0x10_0000, i * 0x10_0000 + 0x1000), 8);
        }
        assert_eq!(total_dropped, 4);
        assert_eq!(disjoint.len(), 6);
        assert_eq!(disjoint.last().copied(), Some((9 * 0x10_0000, 9 * 0x10_0000 + 0x1000)));
    }

    #[test]
    fn anisotropic_samplers_keep_point_mips_where_the_host_can() {
        let mut w = [0u32; 27];
        w[1] = 3; // anisotropic bilinear magnification
        w[2] = 3; // and minification
        w[13] = 8;
        for (mip, supported, filter) in [
            (1, true, 0x54),
            (1, false, 0x55),
            (2, true, 0x55),
            (2, false, 0x55),
            (0, true, 0x54),
        ] {
            w[3] = mip;
            let d = console_sampler(&w, supported);
            assert_eq!(d.Filter.0, filter, "mip {mip}, supported {supported}");
            assert_eq!(d.MaxAnisotropy, 8);
        }
        w[3] = 1;
        w[16] = 4; // comparison
        assert_eq!(console_sampler(&w, true).Filter.0, 0xd4);
        assert_eq!(console_sampler(&w, false).Filter.0, 0xd5);
        w[16] = 0;
        w[5] = 2; // maximum reduction
        assert_eq!(console_sampler(&w, true).Filter.0, 0x1d4);
        assert_eq!(console_sampler(&w, false).Filter.0, 0x1d5);
        w[5] = 1; // minimum reduction
        assert_eq!(console_sampler(&w, true).Filter.0, 0x154);
        assert_eq!(console_sampler(&w, false).Filter.0, 0x155);
    }

    /// 2D textures and arrays have a neutral stand-in, others a null view.
    #[test]
    fn a_texture_not_there_yet_has_a_neutral_stand_in() {
        let Some(host) = host() else { return };
        // SAFETY: plain heap creation and queries.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let at = heap.GetCPUDescriptorHandleForHeapStart();
            let view = |dimension| View::Shader(0, Some([98, dimension, 5768, 0, 0, 1, 0, 1, 0, 0]));
            assert!(neutral_for(&view(4), at).is_some());
            assert!(neutral_for(&view(5), at).is_some());
            assert!(neutral_for(&view(9), at).is_none());
            assert!(neutral_for(&view(8), at).is_none());
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }

    /// Memory past the levels a slot's view shows is not its texture's.
    #[test]
    fn a_view_of_smaller_levels_is_not_told_of_memory_past_them() {
        let mut spans = Spans::default();
        spans.note(1, 0x10000, 0x50000, true);
        spans.shows(1, 3, 0x12000);
        spans.note(2, 0x10000, 0x50000, true);
        spans.shows(2, 0, 0x50000);
        assert_eq!(spans.slots_holding(0x11000, 0x11800), vec![1, 2]);
        assert_eq!(spans.slots_holding(0x20000, 0x21000), vec![2]);
        assert_eq!(spans.slots_in(0x20000, 0x21000), vec![2]);
        spans.forget(2);
        assert!(spans.slots_in(0x20000, 0x21000).is_empty());
        assert_eq!(spans.slots_in(0x0, 0x10001), vec![1]);
    }

    /// A texture filled with less than was mapped is looked at again, but
    /// not for ever: memory that is shown and cannot be read ends it, and a
    /// filling that got everything starts the count anew.
    #[test]
    fn a_short_filling_is_looked_at_again_a_few_times() {
        // An address no other test names.
        let start = 0x7fee_1234_0000u64;
        let pending = || super::SHORT.lock().unwrap().0.remove(&start);
        for _ in 0..super::SHORT_IN_A_ROW {
            super::read_short(start);
            assert!(pending());
        }
        super::read_short(start);
        assert!(!pending());
        super::read_all(start);
        super::read_short(start);
        assert!(pending());
        super::read_all(start);
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3, u64::MAX);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1, u64::MAX);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
    }

    #[test]
    fn ranges_that_touch_are_one() {
        let mut ranges = vec![(40, 50), (30, 40), (35, 40), (10, 20), (45, 60), (60, 61), (1, 1)];
        join(&mut ranges);
        assert_eq!(ranges, [(1, 1), (10, 20), (30, 61)]);
        join(&mut ranges);
        assert_eq!(ranges.len(), 3);
    }

    /// Typed buffer views of integer formats are marked in the table of
    /// strides; float, normalized, raw and structured ones are not.
    #[test]
    fn integer_typed_buffer_views_are_marked_in_the_table_of_strides() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        // Format, dimension (buffer), first element, count, stride, counter, flags.
        let uav = |format, stride, flags| View::Unordered(0, 0, Some([format, 1, 0, 0, 64, stride, 0, 0, flags, 0]));
        assert_eq!(view_kind(&uav(12, 0, 0)), VIEW_INTEGER, "R16G16B16A16_UINT");
        assert_eq!(view_kind(&uav(59, 0, 0)), VIEW_INTEGER | VIEW_SIGNED, "R16_SINT");
        assert_eq!(view_kind(&uav(10, 0, 0)), 0, "R16G16B16A16_FLOAT");
        assert_eq!(view_kind(&uav(58, 0, 0)), 0, "R16_SNORM");
        assert_eq!(view_kind(&uav(39, 0, 1)), 0, "raw");
        assert_eq!(view_kind(&uav(0, 16, 0)), 0, "structured");
        assert_eq!(view_stride(&uav(12, 0, 0)), Some(8));
    }

    /// What may stand for a texture that is not there yet: one of its kind in
    /// another place or size; another one only if asked; nothing a shader
    /// cannot read in its place.
    #[test]
    fn a_slot_shows_on_what_can_stand_for_its_new_texture() {
        let texture = [3, 0, 0, 0, 512, 0, 256, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], format: u32, dimension: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[10], desc[11]) = (format, dimension, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let old = view(texture, 98, 4, 0x1000_0000);
        let mut twice = texture;
        (twice[4], twice[6], twice[7]) = (1024, 512, 0x9_0001);
        let mut square = texture;
        square[6] = 512;
        let mut buffer = texture;
        buffer[0] = 1;
        assert_eq!(kin(&old, &view(texture, 98, 4, 0x1000_0000)), Kin::Same);
        assert_eq!(kin(&old, &view(texture, 98, 4, 0x2000_0000)), Kin::Moved);
        assert_eq!(kin(&old, &view(twice, 98, 4, 0x2000_0000)), Kin::Sized);
        assert_eq!(kin(&old, &view(square, 98, 4, 0x2000_0000)), Kin::Other);
        assert_eq!(kin(&old, &view(twice, 83, 4, 0x2000_0000)), Kin::Other);
        assert_eq!(kin(&old, &view(texture, 98, 9, 0x2000_0000)), Kin::None);
        assert_eq!(kin(&old, &view(buffer, 98, 1, 0x2000_0000)), Kin::None);
        for (kin, kept) in [(Kin::Same, [true; 3]), (Kin::Moved, [false, true, true]), (Kin::Sized, [false, true, true]), (Kin::Other, [false, false, true]), (Kin::None, [false; 3])] {
            assert_eq!([0, 1, 2].map(|keep| kin.kept(keep)), kept, "{kin:?}");
        }
    }

    /// A constant buffer view larger than the host takes is cut to what it
    /// takes: the device, which such a view removes, stays.
    #[test]
    fn an_oversized_constant_buffer_view_leaves_the_device_alive() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // SAFETY: a plain device query.
        let alive = || unsafe { host.device.GetDeviceRemovedReason() }.is_ok();
        // Made first: an allocation that fails on a removed device must not
        // end the test early.
        let (buffer, _) = host.upload_buffer(1 << 20).expect("an upload buffer");
        let memory = vec![0u8; 2 << 20];
        let at = (memory.as_ptr() as u64).next_multiple_of(64 << 10);
        for size in [969_472u32, 896_256] {
            assert!(make(&View::Constant(at, size), heap.cpu(0)).is_some());
            assert!(alive(), "a view of {size} bytes");
        }
        // SAFETY: a live upload buffer.
        heap.constant_at(1, &buffer, unsafe { buffer.GetGPUVirtualAddress() }, 969_472);
        assert!(alive(), "a view of the list's own memory");
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The raw twin of a view shows the same bytes: typed and structured ones
    /// that begin on 16 bytes, whole dwords, none past the buffer.
    #[test]
    fn raw_twins_show_the_views_bytes() {
        let (u16s, u32s, none) = (DXGI_FORMAT(57), DXGI_FORMAT(42), DXGI_FORMAT_UNKNOWN);
        assert_eq!(twin_window(u32s, 4, 10, 0, false, 4096), Some((4, 10, 0)));
        assert_eq!(twin_window(u16s, 8, 7, 0, false, 4096), Some((4, 4, 0)));
        // the last dword would reach past the buffer
        assert_eq!(twin_window(u16s, 8, 7, 0, false, 30), Some((4, 3, 0)));
        assert_eq!(twin_window(none, 2, 3, 24, false, 4096), Some((12, 18, 0)));
        assert_eq!(twin_window(DXGI_FORMAT_R32_TYPELESS, 8, 5, 0, true, 4096), Some((8, 5, 0)));
        // where no raw view begins: from the 16 bytes below
        assert_eq!(twin_window(none, 1, 3, 24, false, 4096), Some((4, 20, 8)));
        assert_eq!(twin_window(u32s, 1, 10, 0, false, 4096), Some((0, 11, 4)));
        assert_eq!(twin_window(u32s, 1, 10, 0, false, 4), None);
    }

    /// What a slot is made as is made beside the heap and copied in whole: the
    /// slot's descriptor is the one the view has, and each thread makes its
    /// own.
    #[test]
    fn views_are_made_beside_the_heap_and_copied_in() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let (Some(a), Some(b)) = (staged(false, 0), staged(false, 1)) else {
            return;
        };
        assert_ne!(a.ptr, b.ptr);
        assert_ne!(Some(a.ptr), std::thread::spawn(|| staged(false, 0).map(|h| h.ptr)).join().unwrap());
        let (buffer, _) = host.shared_buffer(1 << 16).unwrap();
        let index = heap.dynamic(1).unwrap();
        let bytes = |at: D3D12_CPU_DESCRIPTOR_HANDLE| {
            let mut bytes = [0u8; 32];
            crate::host::read_into(at.ptr as *const u8, &mut bytes).then_some(bytes)
        };
        heap.raw_at(index, &buffer, 64, 1024);
        let (slot, beside) = (bytes(heap.cpu(index)), bytes(staged(false, 0).unwrap()));
        assert_eq!(slot, beside, "copied as it was made");
        heap.null_raw_at(index);
        assert_ne!(bytes(heap.cpu(index)), slot, "another view since");
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A raw view moved on by some dwords says so in its descriptor, on a
    /// card whose descriptors are what we take them for; on another nothing
    /// is changed.
    #[test]
    fn a_raw_view_is_moved_only_where_its_descriptor_says_where_it_is() {
        let (Some(host), Some(at)) = (host(), staged(false, 1)) else {
            return; // no host GPU here
        };
        let (buffer, _) = host.shared_buffer(1 << 16).unwrap();
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 8,
                    NumElements: 100,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: this thread's slot, a live buffer.
        let start = unsafe {
            host.device.CreateShaderResourceView(&buffer, Some(&desc), at);
            buffer.GetGPUVirtualAddress() + 32
        };
        let words = || {
            let mut bytes = [0u8; 32];
            assert!(crate::host::read_into(at.ptr as *const u8, &mut bytes));
            bytes
        };
        let before = words();
        assert!(!move_raw(at, start + 16, 400, 8), "another address");
        assert!(!move_raw(at, start, 404, 8), "another size");
        assert!(!move_raw(at, start, 400, 400), "nothing left");
        assert_eq!(words(), before);
        if move_raw(at, start, 400, 8) {
            let moved = words();
            assert_eq!(moved[..8], (start + 8).to_le_bytes());
            assert_eq!(moved[8..12], 392u32.to_le_bytes());
            assert_eq!(moved[12..], before[12..]);
        } else {
            assert_eq!(words(), before);
        }
    }

    /// Views that begin where no host view can: raw ones off 16 bytes,
    /// structured ones off their elements (and off 16 bytes, where they are
    /// made raw), typed ones off theirs.
    #[test]
    fn views_off_their_elements_are_told() {
        // the colour table bakes' parameter block: raw, 8 bytes into its window
        assert!(off_elements(8, 0, true, 4));
        assert!(!off_elements(0, 0, true, 4));
        assert!(!off_elements(48, 0, true, 4));
        // 456-byte elements: at one of them, on 16 bytes, at neither
        assert!(!off_elements(456, 456, false, 0));
        assert!(!off_elements(32, 456, false, 0));
        assert!(off_elements(100, 456, false, 0));
        // A typed view of 12-byte elements on 16 bytes and not on an element.
        assert!(off_elements(23_702_896, 0, false, 12));
        assert!(!off_elements(48, 0, false, 12));
        assert!(!off_elements(12, 0, false, 4));
        assert!(off_elements(12, 0, false, 8));
    }

    /// Such a view is made on a copy of its own bytes, and that copy is the one
    /// brought up to date for it; another's is its buffer's.
    #[test]
    fn views_off_their_elements_get_a_copy_of_their_bytes() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let memory = vec![0u64; 1 << 13];
        let start = (memory.as_ptr() as u64).next_multiple_of(4096);
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Memory placed at `address`, seen as the title passes it: raw, 32
        // dwords, or four 456-byte elements.
        let view = |address: u64, raw: bool| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            let words = if raw {
                [39, 1, 5768, 0, 0, 0, 32, 0, 1, 0]
            } else {
                [0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]
            };
            desc[..10].copy_from_slice(&words);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |v: &View| {
            let made = make(v, at).expect("a view").expect("a buffer");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        for (off, raw, len) in [(8, true, 128u64), (8, false, 4 * 456)] {
            let v = view(start + off, raw);
            assert_eq!(width(&v), len.next_multiple_of(256));
            let (copy, address, bytes) = buffer_copy(&v).unwrap();
            assert_eq!((copy.span(), address, bytes), ((start + off, len), start + off, len));
        }
        for (off, raw) in [(16, true), (32, false)] {
            let v = view(start + off, raw);
            let (copy, ..) = buffer_copy(&v).unwrap();
            assert_ne!(copy.span().0, start + off, "its buffer's copy");
            // SAFETY: a live buffer.
            assert_eq!(width(&v), unsafe { copy.buffer.GetDesc() }.Width);
        }
    }

    /// The queue of textures picked by index serves partial and fresh/mip-upgrade
    /// uploads first, unmapped-memory retries second, and capped recheck zombies
    /// last; a slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again in Tier 2, after fresh Tier 1 slots (never preempting fresh uploads).
            let due = q.due(frame, 100);
            assert_eq!(due.last().unwrap().0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept in Tier 1 ahead of fresh.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    #[test]
    fn texture_queue_tiers_cap_rechecks_and_keep_census_accurate() {
        let mut q = TextureQueue::default();
        // Tier 1 partial, Tier 1 fresh, Tier 2 unmapped retry, Tier 3 rechecks.
        q.written(1, 10, 0, false);
        let w1 = q.due(10, 10)[0].1;
        q.partly(1, w1, 20, 4096);

        q.written(2, 12, 0, false); // fresh
        q.written(3, 14, 0, false); // fresh

        q.written(4, 11, 0, false);
        let w4 = q.due(11, 10).iter().find(|d| d.0 == 4).unwrap().1;
        q.later(4, w4, 19, 4); // due at 20, unmapped retry (Tier 2)

        // 300 already-uploaded recheck zombies due across frames 15..20.
        for i in 0..300u32 {
            let slot = 100 + i;
            q.written(slot, 1, 0, false);
            let w = q.slots[&slot].write;
            let due_at = 15 + u64::from(i % 6);
            q.recheck(slot, w, due_at, (0x1000 + u64::from(i) * 0x100, 4096), 0xdead_beef | 1, 0, 1);
        }

        // Census must NOT count recheck zombies as "waiting for their memory" (c[2])
        // nor let their old write frame inflate oldest non-recheck age (c[4]).
        let c = q.census(20);
        assert_eq!(c[0], 304); // total slots
        assert_eq!(c[1], 304); // all due at frame 20
        assert_eq!(c[2], 1, "only unmapped slot 4 waits for memory");
        assert_eq!(c[3], 1, "only partial slot 1 is uploaded in part");
        assert_eq!(c[4], 10, "oldest non-recheck slot (1) written at frame 10 -> age 10");

        let due = q.due(20, 4096);
        // 4 non-recheck slots + RECHECK_CAP (256) capped zombies.
        assert_eq!(due.len(), 4 + TextureQueue::RECHECK_CAP);
        // Tier 1 first: partial (1), then fresh (3, 2).
        assert_eq!(due[0].0, 1);
        assert_eq!(due[1].0, 3);
        assert_eq!(due[2].0, 2);
        // Tier 2 second: unmapped retry (4).
        assert_eq!(due[3].0, 4);
        // Tier 3 last: oldest due rechecks (due_at == 15) come before later ones (due_at == 20).
        let first_recheck_due = q.slots[&due[4].0].due;
        let last_recheck_due = q.slots[&due.last().unwrap().0].due;
        assert_eq!(first_recheck_due, 15);
        assert!(last_recheck_due >= first_recheck_due);

        // If a recheck zombie later turns unmapped (`later()`), it leaves Tier 3 cleanly.
        let (r_slot, r_write) = due[4];
        q.later(r_slot, r_write, 20, 4);
        assert!(q.checked(r_slot).is_none());
        assert_eq!(q.census(20)[2], 2);
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture: a slot
    /// waiting for its first upload is left alone, one uploaded before is
    /// counted, and made due two frames on when that is asked for. (The
    /// ranges are handed over here, not through the list all heaps share.)
    #[test]
    fn a_page_write_makes_an_uploaded_texture_slot_due_when_asked() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false).is_empty());
        let written = [(at, at + 0x1000)];
        // Waiting for its first upload: held, not uploaded, nothing done.
        assert_eq!(heap.slots_written_under(&written, 50, true), [1, 0, 0]);
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Uploaded and out of the queue. Memory elsewhere: nothing.
        assert_eq!(heap.slots_written_under(&[(at + 0x10_0000, at + 0x10_1000)], 60, true), [0, 0, 0]);
        // Under it, counted only: it stays out of the queue.
        assert_eq!(heap.slots_written_under(&written, 60, false), [1, 1, 0]);
        assert_eq!(slots(u64::MAX), []);
        // Under it, asked for: due two frames on, not before.
        assert_eq!(heap.slots_written_under(&written, 60, true), [1, 1, 1]);
        assert_eq!(slots(61), []);
        assert_eq!(slots(62), [1]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0]).unwrap();
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [(1, 0)]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// A write of no placed texture view forgets the slot in the spans
    /// whether it goes by their lock or not (`spans_quick`): slots that are
    /// in them stay marked, also one another's view made stale.
    #[test]
    fn writes_of_other_views_leave_the_spans_as_under_their_lock() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let marked = |slot: u32| heap.spanned.all(|_| true).contains(&slot);
        let at = 0x7fed_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        assert!(heap.note_buffer(1, Some(&view(at, 64)), false).is_empty());
        assert!(marked(1) && !marked(2));
        // Slot 1 made stale by slot 2: out of `by_slot`, still in `first`.
        assert_eq!(heap.note_buffer(2, Some(&view(at, 128)), false), [(1, 0)]);
        assert!(marked(1) && marked(2));
        assert!(heap.placed.lock().unwrap().first.contains_key(&1));
        // Written with no view: forgotten, under the lock, and unmarked.
        assert!(heap.note_buffer(1, None, false).is_empty());
        assert!(!marked(1) && !heap.placed.lock().unwrap().first.contains_key(&1));
        // A slot never in the spans: nothing there to change.
        assert!(heap.note_buffer(3, None, false).is_empty());
        assert!(heap.note_buffer(2, None, false).is_empty());
        let spans = heap.placed.lock().unwrap();
        assert!(spans.by_slot.is_empty() && spans.by_start.is_empty() && spans.first.is_empty());
        assert!(heap.spanned.all(|_| true).is_empty());
    }

    #[test]
    fn heap_locks_switch_parses_modes() {
        assert_eq!(parse_heap_locks(None), None);
        assert_eq!(parse_heap_locks(Some("")), None);
        assert_eq!(parse_heap_locks(Some("0")), Some(0));
        assert_eq!(parse_heap_locks(Some("1")), Some(1));
        assert_eq!(parse_heap_locks(Some(" alt ")), Some(2));
    }

    #[test]
    fn a_set_of_slots_holds_what_was_put_in() {
        let set = Slots::new(200);
        for slot in [199, 0, 64, 63, 130, 64] {
            set.put(slot);
        }
        assert_eq!(set.all(|_| true), [0, 63, 64, 130, 199]);
        assert_eq!(set.all(|slot| slot % 2 == 0), [0, 64, 130]);
        assert!(set.remove(64) && !set.remove(64) && !set.remove(5));
        assert_eq!(set.take(), [0, 63, 130, 199]);
        assert!(set.take().is_empty());
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, false, frame);
        heap.queue_texture(3, true, false, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, false, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // A `Kin::Same` mip upgrade on a recently-tabled slot takes zero grace.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        heap.queue_texture(3, true, false, true, frame + 2);
        assert_eq!(slots(frame + 2), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 2);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(heap.id(), a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(heap.id(), left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(heap.id(), left, 2);
        heap.release(heap.id(), right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(heap.id(), first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(heap.id(), last, 1);
        assert_eq!(heap.dynamic(2), None);
    }

    #[test]
    fn dynamic_slots_populate_and_clear_stride_table() {
        use victory_shader::emit::{VIEW_INTEGER, VIEW_SIGNED};
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let Some((_, _, data)) = heap.strides else {
            panic!("view heap should allocate strides table");
        };
        let read_stride = |slot: u32| -> u32 {
            // SAFETY: inside `SLOT_STRIDES_COUNT` mapping of `heap.strides`.
            unsafe { *(data as *const u32).add(slot as usize) }
        };
        let memory = vec![0u64; 1 << 11];
        let addr = (memory.as_ptr() as u64).next_multiple_of(4096);
        let placed_typed = |uav: bool, dxgi_fmt: u32, elems: u32| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4096, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            // [format, dim=1 (buffer), mapping/pad, ..., num_elements @6, stride @7, flags @8]
            let words = if uav {
                [dxgi_fmt, 1, 0, 0, elems, 0, 0, 0, 0, 0]
            } else {
                [dxgi_fmt, 1, 5768, 0, 0, 0, elems, 0, 0, 0]
            };
            desc[..10].copy_from_slice(&words);
            (desc[10], desc[11]) = (addr as u32, (addr >> 32) as u32);
            View::Placed(uav, resource, desc)
        };

        let s0 = heap.dynamic(4).unwrap();
        // R16G16B16A16_UINT (12): 8 bytes + VIEW_INTEGER
        let u16x4 = placed_typed(true, 12, 64);
        heap.make_at(s0, &u16x4);
        assert_eq!(read_stride(s0), 8 | VIEW_INTEGER);

        // R16_SINT (59): 2 bytes + VIEW_INTEGER | VIEW_SIGNED
        let s16 = placed_typed(false, 59, 64);
        heap.make_at(s0 + 1, &s16);
        assert_eq!(read_stride(s0 + 1), 2 | VIEW_INTEGER | VIEW_SIGNED);

        // R16G16B16A16_FLOAT (10): 8 bytes, no integer bits
        let f16x4 = placed_typed(true, 10, 64);
        heap.make_at(s0 + 2, &f16x4);
        assert_eq!(read_stride(s0 + 2), 8);

        // Clearing / nulling / raw views overwrite stale bits cleanly.
        heap.make_null(s0);
        assert_eq!(read_stride(s0), 0);
        heap.null_raw_at(s0 + 1);
        assert_eq!(read_stride(s0 + 1), 0);
        heap.null_raw_uav_at(s0 + 2);
        assert_eq!(read_stride(s0 + 2), 0);

        // `make_listed` (both cold creation and cached copy hit) must populate strides.
        let mut slot_mem = [0u32; 8];
        crate::descriptor::record(slot_mem.as_mut_ptr() as u64, u16x4);
        let rec = [slot_mem[0], slot_mem[1], slot_mem[2]];
        assert!(heap.make_listed(0x1234, rec, s0).is_some());
        assert_eq!(read_stride(s0), 8 | VIEW_INTEGER);
        heap.make_null(s0 + 3);
        assert_eq!(read_stride(s0 + 3), 0);
        assert!(heap.make_listed(0x1234, rec, s0 + 3).is_some());
        assert_eq!(read_stride(s0 + 3), 8 | VIEW_INTEGER);
    }
}

/// Minimum idle age (in frames) before a loose window's pinning descriptor
/// slots are eligible for safe guest-RAM stale-slot revalidation (`Spec-Trap #16`).
pub(crate) const STALE_WINDOW_IDLE_FRAMES: u64 = 300;

#[inline]
fn ranges_overlap(a_start: u64, a_len: u64, b_start: u64, b_len: u64) -> bool {
    if b_len == 0 {
        return false;
    }
    let a_end = a_start.saturating_add(a_len.max(1));
    let b_end = b_start.saturating_add(b_len);
    a_start < b_end && b_start < a_end
}

/// Safely reads the first 12 bytes (`[tag, index, generation]`) of a 32-byte
/// guest descriptor record in Xbox RAM without faulting or tripping a Mode-1/2
/// `PAGE_NOACCESS` tripwire (`Spec-Trap #4`). Returns `None` if the address is
/// null, tripwired, or unreadable.
fn safe_read_guest_descriptor_words(addr: u64) -> Option<[u32; 3]> {
    if addr == 0 {
        return None;
    }
    if crate::shadow::is_tripwired(addr as usize, 32) {
        return None;
    }
    let mut bytes = [0u8; 32];
    if crate::safecopy::on() {
        // SAFETY: `bytes` is a valid stack buffer; `safecopy::copy` catches
        // any fault reading `addr` and never disarms shadow tripwires.
        if !unsafe { crate::safecopy::copy(bytes.as_mut_ptr(), addr as *const u8, bytes.len()) } {
            return None;
        }
    } else if !crate::host::read_into(addr as *const u8, &mut bytes) {
        return None;
    }
    Some([
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
    ])
}

/// Returns `true` if `view` references any address overlapping
/// `[win_start, win_start + win_len)`.
fn view_references_window(view: &View, win_start: u64, win_len: u64) -> bool {
    if win_len == 0 {
        return false;
    }
    if let Some((addr, len)) = buffer_range(view)
        && ranges_overlap(addr, len, win_start, win_len)
    {
        return true;
    }
    if let Some((base, width)) = buffer_extent(view)
        && ranges_overlap(base, width, win_start, win_len)
    {
        return true;
    }
    match view {
        View::Constant(addr, size) => ranges_overlap(*addr, u64::from(*size), win_start, win_len),
        View::Placed(_, _, desc) => {
            let at = u64::from(desc[10]) | (u64::from(desc[11]) << 32);
            at != 0 && ranges_overlap(at, 1, win_start, win_len)
        }
        _ => view_address(view)
            .is_some_and(|at| at != 0 && ranges_overlap(at, 1, win_start, win_len)),
    }
}

/// Returns `true` ONLY when `guest_handle` is safely readable in guest RAM AND
/// has provably been zeroed, freed, or overwritten so it no longer references
/// `[win_start, win_start + win_len)`. Any read failure or tripwire returns
/// `false` (conservative: never speculatively null a slot).
fn guest_slot_proven_stale(guest_handle: u64, win_start: u64, win_len: u64) -> bool {
    let Some([tag, i, generation]) = safe_read_guest_descriptor_words(guest_handle) else {
        return false;
    };
    if tag != VIEW_TAG {
        return true;
    }
    match super::view_at(i, generation) {
        None => true,
        Some(view) => !view_references_window(&view, win_start, win_len),
    }
}

/// Unpins title-heap descriptor slots referencing idle `window == true` loose
/// shadow buffers (`raw_buffers` = `buffer.as_raw() as usize`), enforcing the
/// Null-Before-Retire protocol (`ARCH-05`, Spec-Trap #16) so `shadow::let_go_idle`
/// can reclaim them safely once retired COM refs drain.
pub(crate) fn unpin_idle_window_slots(raw_buffers: &[usize]) -> usize {
    if raw_buffers.is_empty() {
        return 0;
    }
    super::with_view_heap(|host| host.unpin_window_buffers(raw_buffers)).unwrap_or(0)
}

/// Safely revalidates and unpins provably stale title-heap descriptor slots
/// pinning idle (`>= 300` frames) loose windows (`(raw_buffer, win_start,
/// win_len, last_used_frame)`). Callable when `l0_loose` exceeds
/// `VICTORY_WINDOW_BUDGET_MB` (`Spec-Trap #16` & `Spec-Trap #4`).
#[allow(dead_code)]
pub(crate) fn unpin_stale_window_slots(
    candidates: &[(usize, u64, u64, u64)],
    now_frame: u64,
) -> usize {
    if candidates.is_empty() {
        return 0;
    }
    super::with_view_heap(|host| host.unpin_stale_window_slots(candidates, now_frame, None))
        .unwrap_or(0)
}

/// Global epoch invalidating per-thread [`ListViews`] texture CPU descriptor
/// caches whenever host backing textures/views change (mip growth, ahead flip).
static VIEW_CACHE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Separate epoch for buffer/TLAS entries in [`ListViews`] so texture mip flips
/// and `resource::let_go_idle` sweeps do not wipe cached buffer descriptors (`CPU-23`).
static BUFFER_VIEW_CACHE_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Current `(texture_view_epoch, buffer_view_epoch)` pair used by
/// `draw::State::last_stage_tables` to detect descriptor-heap invalidations.
#[inline]
pub(crate) fn view_cache_epochs() -> (u64, u64) {
    (
        VIEW_CACHE_EPOCH.load(std::sync::atomic::Ordering::Acquire),
        BUFFER_VIEW_CACHE_EPOCH.load(std::sync::atomic::Ordering::Acquire),
    )
}

#[inline]
pub(crate) fn bump_view_cache_epoch() {
    VIEW_CACHE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Release);
}

#[inline]
#[allow(dead_code)]
pub(crate) fn bump_buffer_view_cache_epoch() {
    BUFFER_VIEW_CACHE_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Release);
}

#[inline]
fn texture_view_cache_epoch() -> (u64, u64, u64) {
    (
        VIEW_CACHE_EPOCH.load(std::sync::atomic::Ordering::Acquire),
        crate::resource::SWEPT_US[0].load(std::sync::atomic::Ordering::Relaxed),
        crate::frame::current() >> 4,
    )
}

#[inline]
fn buffer_view_cache_epoch() -> (u64, u64) {
    (
        BUFFER_VIEW_CACHE_EPOCH.load(std::sync::atomic::Ordering::Acquire),
        crate::frame::current() >> 6,
    )
}

/// What is known of a record a list's tables hold ([`HostHeap::make_listed`]).
#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Listed {
    pub(crate) kind: Option<victory_shader::native::ResourceType>,
    pub(crate) texture: bool,
    /// Read at this epoch of the list ([`crate::host::HostList::epoch`]) with
    /// this depth target tested: up to date; and whether it is that target's.
    pub(crate) ready: (u64, usize, bool),
    /// The title's slot last noted for it, and whether the draw read it.
    pub(crate) tabled: Option<(u32, bool)>,
}

struct ListView {
    view: View,
    /// Its descriptor's slot in [`ListViews`]; its raw twin's is the next (unless `twin_same`).
    at: u32,
    twin: bool,
    twin_same: bool,
    stride_word: u32,
    count: u32,
    kept: Kept,
    kind: Option<victory_shader::native::ResourceType>,
    texture: bool,
}

/// Persistent per-thread CPU descriptor cache by record key `[VIEW_TAG, i, gen]`
/// in a non-shader-visible heap, partitioned into independent texture (`0..4096`)
/// and buffer/TLAS (`4096..8192`) rings invalidated separately by
/// [`texture_view_cache_epoch`] and [`buffer_view_cache_epoch`] (`CPU-23`).
struct ListViews {
    tex_epoch: (u64, u64, u64),
    buf_epoch: (u64, u64),
    // Kept alive so CPU descriptor handles (`start + i * step`) remain valid.
    _heap: ID3D12DescriptorHeap,
    step: usize,
    start: usize,
    textures: std::collections::HashMap<[u32; 3], ListView, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
    buffers: std::collections::HashMap<[u32; 3], ListView, std::hash::BuildHasherDefault<crate::host::AddressHash>>,
}

impl ListViews {
    const TEX_SLOTS: u32 = 4096;
    const BUF_SLOTS: u32 = 4096;
    const SLOTS: u32 = Self::TEX_SLOTS + Self::BUF_SLOTS;

    fn new() -> Option<ListViews> {
        let host = host()?;
        // SAFETY: plain object creation.
        unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: Self::SLOTS,
                    ..Default::default()
                })
                .ok()?;
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            LIST_VIEWS_RANGE.with(|r| r.set((start, start + Self::SLOTS as usize * step)));
            Some(ListViews {
                tex_epoch: texture_view_cache_epoch(),
                buf_epoch: buffer_view_cache_epoch(),
                _heap: heap,
                step,
                start,
                textures: Default::default(),
                buffers: Default::default(),
            })
        }
    }
}

thread_local! {
    static LIST_VIEWS: std::cell::RefCell<Option<ListViews>> = const { std::cell::RefCell::new(None) };
}

/// Whether a list makes a view its tables hold once (`VICTORY_LIST_VIEWS=0`:
/// for every table).
pub(crate) fn list_views() -> bool {
    crate::toggled!("VICTORY_LIST_VIEWS")
}

#[cfg(test)]
mod bake_flip_tests {
    use super::*;

    #[test]
    fn pending_list_flips_promote_once_and_clean_up() {
        let heap = 0xdead_beefusize;
        let id1 = NEXT_PENDING_FLIP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id2 = NEXT_PENDING_FLIP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        {
            let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
            pending.push(PendingListFlip {
                id: id1,
                origin: (1, 1),
                flip: Flip { after: 0, heap, index: 7, handle: 0, key: 11, late: Some(1) },
            });
            pending.push(PendingListFlip {
                id: id2,
                origin: (2, 1),
                flip: Flip { after: 0, heap, index: 8, handle: 0, key: 12, late: Some(1) },
            });
            PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
        }
        let taken = take_pending_list_flip(id1).expect("id1 should be present");
        assert_eq!(taken.index, 7);
        assert!(take_pending_list_flip(id1).is_none(), "second take must be a no-op");
        remove_pending_list_flip(id2);
        assert!(take_pending_list_flip(id2).is_none(), "removed entry must be gone");
    }

    #[test]
    fn deferred_flip_closure_keeps_pending_entry_alive_until_called_or_dropped() {
        let heap = 0xcafe_babeusize;
        let id1 = NEXT_PENDING_FLIP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id2 = NEXT_PENDING_FLIP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        {
            let mut pending = PENDING_LIST_FLIPS.lock().unwrap_or_else(|e| e.into_inner());
            for (id, idx) in [(id1, 10), (id2, 11)] {
                pending.push(PendingListFlip {
                    id,
                    origin: (3, 1),
                    flip: Flip { after: 0, heap, index: idx, handle: 0, key: 20, late: Some(1) },
                });
            }
            PENDING_LIST_COUNT.store(pending.len(), std::sync::atomic::Ordering::Release);
        }
        struct TestGuard(u64);
        impl TestGuard {
            fn id(&self) -> u64 {
                self.0
            }
        }
        impl Drop for TestGuard {
            fn drop(&mut self) {
                remove_pending_list_flip(self.0);
            }
        }
        let make_cb = |id: u64| -> Box<dyn Fn() -> Option<u32> + Send + 'static> {
            let guard = TestGuard(id);
            Box::new(move || take_pending_list_flip(guard.id()).map(|f| f.index))
        };
        let cb1 = make_cb(id1);
        let cb2 = make_cb(id2);
        // Must still be present after make_cb returns (guard moved, not dropped).
        assert_eq!(cb1(), Some(10));
        assert_eq!(cb1(), None);
        drop(cb2);
        assert!(take_pending_list_flip(id2).is_none(), "dropping uncalled closure cleans up entry");
    }

    /// The unsigned twin of a 16-bit float or signed format has its layout,
    /// and only buffer views in such a format are shown as another view.
    #[test]
    fn only_16_bit_float_and_signed_buffer_views_get_an_unsigned_twin() {
        for (format, unsigned) in [(10, 12), (14, 12), (34, 36), (38, 36), (54, 57), (59, 57)] {
            assert_eq!(half_format(format), Some(unsigned));
            assert_eq!(
                crate::resource::element(format),
                crate::resource::element(unsigned),
                "format {format}"
            );
        }
        // The unsigned formats themselves, normalised 16-bit formats, and
        // wider and narrower ones.
        for other in [0, 12, 36, 57, 11, 13, 35, 37, 56, 58, 42, 17, 30, 61] {
            assert_eq!(half_format(other), None, "format {other}");
        }
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(10, BUFFER)))),
            Some(View::Unordered(7, 0, Some(words(12, BUFFER))))
        );
        assert_eq!(
            half_view(&View::Shader(7, Some(words(54, BUFFER)))),
            Some(View::Shader(7, Some(words(57, BUFFER))))
        );
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(38, BUFFER));
        placed[10] = 0x1234_0000;
        let mut twin = placed;
        twin[0] = 36;
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(
            half_view(&View::Placed(true, memory, placed)),
            Some(View::Placed(true, memory, twin))
        );
        assert_eq!(
            half_view(&View::Unordered(7, 0, Some(words(12, BUFFER)))),
            None
        );
        assert_eq!(half_view(&View::Shader(7, Some(words(57, BUFFER)))), None);
        assert_eq!(half_view(&View::Unordered(7, 0, Some(words(10, 4)))), None);
        assert_eq!(half_view(&View::Shader(7, None)), None);
        assert_eq!(half_view(&View::Constant(0x1000, 256)), None);
    }

    /// The buffer views through which the integers a shader moves halves
    /// as are not what the console gives are named; the title's are not.
    #[test]
    fn views_whose_values_the_console_would_convert_are_named() {
        const BUFFER: u32 = 1;
        let words = |format: u32, dimension: u32| [format, dimension, 5768, 0, 0, 0, 64, 0, 0, 0];
        for exact in [10, 12, 14, 34, 36, 38, 54, 57, 59, 25, 30, 50, 62] {
            let view = View::Unordered(7, 0, Some(words(exact, BUFFER)));
            assert_eq!(half_inexact(&view), None, "format {exact}");
        }
        for inexact in [41, 42, 11, 28, 64, 0] {
            let view = View::Shader(7, Some(words(inexact, BUFFER)));
            assert_eq!(half_inexact(&view), Some(inexact));
        }
        let mut placed = [0u32; 16];
        placed[..10].copy_from_slice(&words(41, BUFFER));
        let memory = [BUFFER, 0, 0, 0, 512, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0];
        assert_eq!(half_inexact(&View::Placed(true, memory, placed)), Some(41));
        assert_eq!(half_inexact(&View::Shader(7, Some(words(41, 4)))), None);
        assert_eq!(half_inexact(&View::Constant(0x1000, 256)), None);
    }

    /// A slot whose texture was uploaded ahead shows it when it is
    /// published, and does not go back into the queue of textures picked by
    /// index for that; one the title wrote again is left alone.
    #[test]
    fn a_published_slot_stays_out_of_the_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let view = |at: u64| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[2], desc[5]) = (98, 4, 5768, 1);
            (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let title = [0u32; 8];
        let handle = title.as_ptr() as u64;
        crate::descriptor::record(handle, view(0x7fec_0000_0000));
        let (first, key) = crate::descriptor::view_and_key(handle).unwrap();
        assert!(heap.note_buffer(1, Some(&first), false).is_empty());
        let (index, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(index, 1);
        // Its upload was recorded: out of the queue, the slot not made yet.
        heap.texture_slot_done(1, write, None);
        assert!(!heap.made(1));
        assert!(heap.publish(1, handle, key));
        assert!(heap.made(1));
        assert!(heap.texture_slots_due(u64::MAX, 8).0.is_empty());
        // Written again since the upload was recorded.
        crate::descriptor::record(handle, view(0x7feb_0000_0000));
        assert!(!heap.publish(1, handle, key));
        assert_ne!(crate::descriptor::view_and_key(handle).unwrap().1, key);
    }
}

#[cfg(test)]
mod dynamic_pool_tests {
    use super::DynamicPool;

    #[test]
    fn dynamic_pool_padding_wrap_and_cursor_crossing() {
        for cap in [1u32, 63, 64, 65, 130, 195] {
            let mut pool = DynamicPool::new(cap);
            assert_eq!(pool.find(cap), Some(0), "full alloc for cap={cap}");
            pool.mark_taken(0, cap);
            assert_eq!(pool.find(1), None, "exhausted for cap={cap}");
            assert_eq!(pool.mark_free(0, cap), cap);
            assert_eq!(pool.mark_free(0, cap), 0, "double free reports 0");
            assert_eq!(pool.find(cap + 1), None, "over-cap rejected for cap={cap}");
        }

        // Verify cursor-crossing runs are found, while physical-end wrap is rejected.
        let mut pool = DynamicPool::new(130);
        pool.mark_taken(0, 1);
        pool.mark_taken(1, 2);
        pool.mark_taken(3, 2);
        pool.mark_taken(5, 125);
        assert_eq!(pool.find(1), None);

        assert_eq!(pool.mark_free(1, 2), 2);
        assert_eq!(pool.find(2), Some(1));
        pool.mark_taken(1, 2); // cursor is now at 3, right between [1..3) and [3..5)
        assert_eq!(pool.mark_free(1, 2), 2);
        assert_eq!(pool.mark_free(3, 2), 2);
        assert_eq!(pool.find(4), Some(1), "run crossing cursor must be found");
        pool.mark_taken(1, 4);
        assert_eq!(pool.find(1), None);

        // One slot free at physical start and one at physical end must not merge.
        assert_eq!(pool.mark_free(0, 1), 1);
        assert_eq!(pool.mark_free(129, 1), 1);
        assert_eq!(pool.find(2), None);
        // Cursor is at 5, so search finds 129 first, then wraps to 0.
        assert_eq!(pool.find(1), Some(129));
        pool.mark_taken(129, 1);
        assert_eq!(pool.find(1), Some(0));
    }

    #[test]
    fn dynamic_pool_matches_vec_bool_oracle_across_word_boundaries() {
        let cap = 317u32;
        let mut pool = DynamicPool::new(cap);
        let mut ref_taken = vec![false; cap as usize];
        let mut ref_cursor = 0usize;
        let mut live: Vec<(u32, u32)> = Vec::new();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next_u32 = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 32) as u32
        };

        for _ in 0..800 {
            if live.len() > 24 || (!live.is_empty() && next_u32() % 3 == 0) {
                let idx = (next_u32() as usize) % live.len();
                let (first, n) = live.swap_remove(idx);
                let got_freed = pool.mark_free(first, n);
                let mut ref_freed = 0u32;
                for slot in &mut ref_taken[first as usize..(first + n) as usize] {
                    if *slot {
                        ref_freed += 1;
                        *slot = false;
                    }
                }
                assert_eq!(got_freed, ref_freed);
                continue;
            }
            let n = (next_u32() % 90) + 1;
            let got = pool.find(n);
            let n_us = n as usize;
            let end = cap as usize;
            let second_end = ref_cursor.saturating_add(n_us.saturating_sub(1)).min(end);
            let mut run = 0usize;
            let mut last = usize::MAX;
            let mut expected = None;
            for at in (ref_cursor..end).chain(0..second_end) {
                if at != last.wrapping_add(1) {
                    run = 0;
                }
                last = at;
                run = if ref_taken[at] { 0 } else { run + 1 };
                if run == n_us {
                    expected = Some((at + 1 - n_us) as u32);
                    break;
                }
            }
            assert_eq!(got, expected, "mismatch for n={n}, cursor={ref_cursor}");
            if let Some(first) = got {
                pool.mark_taken(first, n);
                ref_taken[first as usize..(first + n) as usize].fill(true);
                ref_cursor = ((first + n) as usize) % end;
                live.push((first, n));
            }
        }
        // Rewind-if-tail restores cursor to recycled slab tail.
        let mut tail_pool = DynamicPool::new(256);
        let a = tail_pool.find(64).unwrap();
        tail_pool.mark_taken(a, 64);
        assert_eq!(tail_pool.mark_free(20, 44), 44);
        tail_pool.rewind_if_tail(20, 44);
        assert_eq!(tail_pool.find(16), Some(20));
    }

    #[test]
    fn cached_null_kind_maps_all_static_srv_uav_and_cbv_dims() {
        use super::{View, cached_null_kind};
        let srv_desc = |dim: u32| {
            let mut w = [0u32; 10];
            w[1] = dim;
            w
        };
        assert_eq!(cached_null_kind(&View::Constant(0, 256)), Some(2));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(1)))), Some(1));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(4)))), Some(0));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(2)))), Some(3));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(3)))), Some(4));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(5)))), Some(5));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(6)))), Some(6));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(7)))), Some(7));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(8)))), Some(8));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(9)))), Some(9));
        assert_eq!(cached_null_kind(&View::Shader(0, Some(srv_desc(10)))), Some(10));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(1)))), Some(11));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(2)))), Some(12));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(3)))), Some(13));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(4)))), Some(14));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(5)))), Some(15));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, Some(srv_desc(8)))), Some(16));
        assert_eq!(cached_null_kind(&View::Unordered(0, 0, None)), Some(14));
    }

    #[test]
    fn relaxed_cast_fits_and_placed_texture_redirects_without_alias_copy() {
        use super::{View, fits, host, make, placed_texture};
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;

        let Some(h) = host() else {
            return;
        };
        if !crate::resource::relaxed_casting_supported(&h.device) {
            return;
        }
        let (heap_props, src_desc_template) = crate::resource::readback_test_descriptions();
        // Build a 64x64 R32_UINT UAV texture through `create_committed`.
        let mut src_words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 0x0001_0001, 42, 1, 0, 0, 4, 0];
        let _ = src_desc_template;
        // SAFETY: 14 words = 56-byte `ResourceDesc`.
        let src_desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(src_words) };
        let mut src = std::ptr::null();
        // `IID_ID3D12Resource` = {696442be-a72e-4059-bc79-5b5c98040fad}
        let iid = crate::object::guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                &heap_props,
                0,
                &src_desc,
                0x8,
                std::ptr::null(),
                &iid,
                &mut src,
            ),
            0
        );
        let copy = crate::resource::host_copy(src).unwrap();
        let castable = copy.castable_formats();
        assert!(!castable.is_empty());

        // `fits()` permits R9G9B9E5_SHAREDEXP SRV without overwriting format,
        // permits R32_UINT UAV, forbids R9G9B9E5_SHAREDEXP UAV, and rejects when non-castable.
        let mut fmt = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        assert!(fits(&copy.texture, castable, Some(&mut fmt), false));
        assert_eq!(fmt, DXGI_FORMAT_R9G9B9E5_SHAREDEXP);

        let mut uav_ok = DXGI_FORMAT_R32_UINT;
        assert!(fits(&copy.texture, castable, Some(&mut uav_ok), true));
        assert_eq!(uav_ok, DXGI_FORMAT_R32_UINT);

        let mut uav_bad = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        assert!(!fits(&copy.texture, castable, Some(&mut uav_bad), true));

        let mut no_cast = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
        assert!(!fits(&copy.texture, &[], Some(&mut no_cast), false));

        // Placed R9G9B9E5 SRV at `src`'s title address redirects directly to `src`,
        // preserving format 67 even when the placed SRV descriptor leaves Format=0.
        let (addr_ptr, _) = crate::resource::texture_memory(src).unwrap();
        let addr = addr_ptr as u64;
        src_words[8] = 67; // R9G9B9E5_SHAREDEXP
        src_words[12] = 0; // flags = 0 (fills)
        let mut view_desc = [0u32; 16];
        view_desc[0] = 0; // UNKNOWN -> must become 67 on redirect
        view_desc[1] = 4; // TEXTURE2D
        view_desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view_desc[5] = 1; // MipLevels
        view_desc[10] = addr as u32;
        view_desc[11] = (addr >> 32) as u32;
        let placed_view = View::Placed(false, src_words, view_desc);
        let redirected = placed_texture(&placed_view).unwrap();
        match redirected {
            View::Shader(res, Some(srv)) => {
                assert_eq!(res, src as usize);
                assert_eq!(srv[0], 67);
            }
            other => panic!("expected redirected View::Shader, got {other:?}"),
        }

        // Making the placed view directly into a descriptor heap binds `src`'s host texture.
        // SAFETY: plain heap creation on live test device.
        let heap: ID3D12DescriptorHeap = unsafe {
            h.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .unwrap()
        };
        // SAFETY: live heap.
        let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let kept = make(&placed_view, at).unwrap().unwrap();
        assert_eq!(
            windows::core::Interface::as_raw(&kept),
            windows::core::Interface::as_raw(&copy.texture)
        );
        // Release `src` via COM `IUnknown::Release` (vtable slot 2).
        type Release = extern "system" fn(crate::object::This) -> u32;
        // SAFETY: `src` is a live COM resource whose first word is its vtable pointer.
        let release: Release = unsafe {
            let vtbl = *src.cast::<*const *const ()>();
            std::mem::transmute(*vtbl.add(2))
        };
        assert_eq!(release(src), 0);
    }

    #[test]
    fn tlas_srv_is_twinned_when_dxr_enabled() {
        use super::{View, twinned, twinned_for};
        use windows::Win32::Graphics::Direct3D12::D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;

        let mut tlas_desc = [0u32; 10];
        tlas_desc[1] = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
        tlas_desc[2] = 0x1000;
        let tlas_view = View::Shader(0, Some(tlas_desc));
        assert!(twinned_for(&tlas_view, true));
        assert!(!twinned_for(&tlas_view, false));
        assert_eq!(twinned(&tlas_view), crate::dxr::enabled());

        let mut tex_desc = [0u32; 10];
        tex_desc[1] = 4; // TEXTURE2D
        let tex_view = View::Shader(0x1234, Some(tex_desc));
        assert!(!twinned_for(&tex_view, true));
        assert!(!twinned_for(&tex_view, false));
    }

    #[test]
    fn spans_note_zero_address_produces_zero_stale_slots() {
        use super::Spans;

        let mut spans = Spans::default();
        assert!(spans.note(1, 0, 0x1000, true).is_empty());
        assert!(spans.note(2, 0, 0x2000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x1000, true).is_empty());
        assert!(spans.by_slot.is_empty());
        assert!(spans.by_start.is_empty());
    }

    #[test]
    fn spans_missing_mip_wakes_partial_slot_preserving_mapped() {
        use super::{Spans, TextureQueue};

        let mut spans = Spans::default();
        spans.note_missing_mip(42, 0x2000_0000, 0x2001_0000);
        spans.note_missing_mip(43, 0x3000_0000, 0x3001_0000);
        assert_eq!(spans.waiting_slots_in(0x2000_4000, 0x2000_8000), vec![42]);
        assert!(spans.waiting_slots_in(0x4000_0000, 0x4000_1000).is_empty());

        let mut queue = TextureQueue::default();
        queue.written(42, 10, 0, false);
        // Simulate partial upload (`mapped = 4096`), then scheduled far in the future.
        let entry = queue.slots.get_mut(&42).unwrap();
        entry.due = 100;
        entry.mapped = 4096;
        assert_eq!(queue.slots[&42].due, 100);
        assert_eq!(queue.slots[&42].mapped, 4096);
        // Missing-mip wakeup moves `due` to `frame` while preserving `mapped`.
        queue.wake_missing(42, 11);
        assert_eq!(queue.slots[&42].due, 11);
        assert_eq!(queue.slots[&42].mapped, 4096);

        spans.clear_missing_mip(42);
        assert!(spans.waiting_slots_in(0x2000_4000, 0x2000_8000).is_empty());
    }

    #[test]
    fn copied_tabled_uses_64bit_epochs_and_defers_without_silent_drop() {
        use super::{COPIED_FRAMES, TABLED_READ, tabled_now};
        use std::collections::BTreeMap;

        // Cross-thread skew (`frame < stamp`) must saturate to 0, never wrap near u32::MAX.
        let stamp = tabled_now(100);
        let earlier = tabled_now(99);
        assert_eq!(earlier.saturating_sub(stamp), 0);
        assert!(earlier.saturating_sub(stamp) <= COPIED_FRAMES);
        assert_eq!((stamp | TABLED_READ) & !TABLED_READ, stamp);

        // Deferral target is strictly `stamp_frame + COPIED_FRAMES + 1` (past the tabled window).
        let stamp_frame = stamp - 1;
        let due = stamp_frame.saturating_add(COPIED_FRAMES + 1);
        assert!(tabled_now(due).saturating_sub(stamp) > COPIED_FRAMES);

        // BTreeMap waiting queue preserves deferred slots across arbitrary 64-bit frames.
        let mut waiting: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        waiting.entry(due).or_default().push(7);
        let not_ready = waiting.first_key_value().is_some_and(|(&f, _)| f <= due - 1);
        assert!(!not_ready);
        let ready = waiting.first_key_value().is_some_and(|(&f, _)| f <= due);
        assert!(ready);
        assert_eq!(waiting.remove(&due), Some(vec![7]));
    }

    #[test]
    fn sparse_sidecars_demand_page_4k_chunks_without_allocating_untouched_slots() {
        use super::{SPARSE_CHUNK, SparseBits, SparseU8, SparseU64};
        use std::sync::atomic::Ordering;

        let u64s = SparseU64::new(1_048_576);
        assert_eq!(u64s.allocated_chunks(), 0);
        // Reads and zero-writes allocate zero chunks.
        assert_eq!(u64s.get(500_000, Ordering::Acquire), Some(0));
        u64s.set(500_000, 0, Ordering::Release);
        assert_eq!(u64s.allocated_chunks(), 0);
        assert_eq!(u64s.get(1_048_576, Ordering::Acquire), None);

        // Non-zero write allocates exactly one 4K chunk.
        u64s.set(SPARSE_CHUNK + 17, 0xdead_beef, Ordering::Release);
        assert_eq!(u64s.allocated_chunks(), 1);
        assert_eq!(u64s.get(SPARSE_CHUNK + 17, Ordering::Acquire), Some(0xdead_beef));
        assert_eq!(u64s.get(SPARSE_CHUNK + 18, Ordering::Acquire), Some(0));

        // SparseU8 default=1 matches `null_kind` semantics.
        let nk = SparseU8::new(100_000, 1);
        assert_eq!(nk.get(12_345), Some(1));
        nk.set(12_345, 1);
        assert_eq!(nk.get(12_345), Some(1));
        nk.set(12_345, 0);
        assert_eq!(nk.get(12_345), Some(0));
        nk.set(12_345, 2);
        assert_eq!(nk.get(12_345), Some(2));

        // SparseBits tracks twins/window_slot_bits cleanly across chunk boundaries.
        let bits = SparseBits::new(20_000);
        assert!(!bits.get(9_000));
        assert!(!bits.clear_if_set(9_000));
        bits.set(9_000, true);
        assert!(bits.get(9_000));
        assert!(bits.clear_if_set(9_000));
        assert!(!bits.get(9_000));
        assert!(!bits.clear_if_set(9_000));
    }

    #[test]
    fn unpin_window_buffers_nulls_before_retire_and_preserves_non_window_and_dynamic_slots() {
        use super::{HostHeap, Ordering, host};
        use windows::core::Interface;

        let Some(h) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 16) else {
            return;
        };
        let Some((win_buf, _)) = h.shared_buffer(4096) else {
            return;
        };
        let Some((keep_buf, _)) = h.shared_buffer(4096) else {
            return;
        };
        let win_raw = win_buf.as_raw() as usize;
        let keep_raw = keep_buf.as_raw() as usize;

        // Pin two title slots to `win_buf` (`is_window = true`), one title slot to `keep_buf`
        // (`is_window = false`), and one dynamic slot to `win_buf`.
        heap.made.set(2, 0x1002);
        heap.quick_is(2, 0x1002);
        heap.twins.set(2, true);
        heap.keep_window(2, Some(win_buf.clone()), true);

        heap.made.set(5, 0x1005);
        heap.quick_is(5, 0x1005);
        heap.keep_window(5, Some(win_buf.clone()), true);

        heap.made.set(6, 0x1006);
        heap.quick_is(6, 0x1006);
        heap.keep_window(6, Some(keep_buf.clone()), false);

        let dyn_slot = heap.dynamic(1).unwrap();
        heap.keep_window(dyn_slot, Some(win_buf.clone()), true);

        // `ResourceTable` deduplicates the two title slots into a single COM ref.
        assert_eq!(heap.kept_raw_at(2), win_raw);
        assert_eq!(heap.kept_raw_at(5), win_raw);
        assert_eq!(heap.kept_raw_at(6), keep_raw);

        // Unpinning `win_raw` + `keep_raw` must ONLY unpin the two window title slots (2 and 5):
        // clears `made`/`quick`/`twins`/`kept_ids` and writes `null_buffer_view` (tag 2),
        // while leaving non-window slot 6 and `dynamic_kept` untouched!
        let epoch_before = super::BUFFER_VIEW_CACHE_EPOCH.load(Ordering::Acquire);
        let unpinned = heap.unpin_window_buffers(&[win_raw, keep_raw]);
        assert_eq!(unpinned, 2);
        assert!(super::BUFFER_VIEW_CACHE_EPOCH.load(Ordering::Acquire) > epoch_before);

        for slot in [2u32, 5] {
            assert_eq!(heap.made.get(slot), Some(0));
            assert_eq!(heap.quick.get(slot as usize, Ordering::Acquire), Some(0));
            assert_eq!(heap.kept_raw_at(slot), 0);
            assert!(!heap.twins.get(slot as usize));
            assert!(!heap.window_slot_bits.get(slot as usize));
            // Cached null kind `1` (raw-buffer SRV) -> tag `2`.
            assert_eq!(heap.null_kind.get(slot as usize), Some(2));
        }
        // Non-window title slot 6 untouched.
        assert_eq!(heap.made.get(6), Some(0x1006));
        assert_eq!(heap.kept_raw_at(6), keep_raw);
        // Second call is an O(1) no-op.
        assert_eq!(heap.unpin_window_buffers(&[win_raw]), 0);
        heap.release(heap.id(), dyn_slot, 1);
    }

    #[test]
    fn unpin_stale_window_slots_never_nulls_live_guest_slot_and_unpins_overwritten_or_zeroed_slots() {
        use super::{HostHeap, Ordering, STALE_WINDOW_IDLE_FRAMES, View, host};
        use windows::core::Interface;

        let Some(h) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 16) else {
            return;
        };
        let Some((win_buf, _)) = h.shared_buffer(4096) else {
            return;
        };
        let Some((keep_buf, _)) = h.shared_buffer(4096) else {
            return;
        };
        let win_raw = win_buf.as_raw() as usize;
        let keep_raw = keep_buf.as_raw() as usize;

        // Backing guest descriptor heap memory (16 x 32-byte slots).
        let mut guest_heap = vec![0u32; 16 * 8];
        let guest_base = guest_heap.as_mut_ptr() as u64;

        let win_start = 0x7000_1000_0000u64;
        let win_len = 0x40_0000u64;
        let other_start = 0x7000_9000_0000u64;

        let placed_buf_view = |addr: u64, bytes: u32| {
            let mut desc = [0u32; 16];
            desc[0] = 0; // UNKNOWN (structured)
            desc[1] = 1; // BUFFER
            desc[2] = 5768;
            desc[5] = bytes / 16; // NumElements
            desc[6] = 16; // StructureByteStride
            desc[10] = addr as u32;
            desc[11] = (addr >> 32) as u32;
            let mut res = [0u32; 14];
            res[0] = 1; // BUFFER
            res[4] = bytes;
            View::Placed(false, res, desc)
        };

        // Slot 2: LIVE guest descriptor still pointing into `[win_start, win_start + win_len)`.
        crate::descriptor::record(guest_base + 2 * 32, placed_buf_view(win_start + 0x1000, 4096));
        heap.made.set(2, 0x2002);
        heap.quick_is(2, 0x2002);
        heap.keep_window(2, Some(win_buf.clone()), true);

        // Slot 3: STALE guest descriptor ZEROED by the title.
        // (`guest_heap` bytes at slot 3 are already all 0 != VIEW_TAG.)
        heap.made.set(3, 0x2003);
        heap.quick_is(3, 0x2003);
        heap.twins.set(3, true);
        heap.keep_window(3, Some(win_buf.clone()), true);

        // Slot 4: STALE guest descriptor OVERWRITTEN to point to a different VA outside the window.
        crate::descriptor::record(guest_base + 4 * 32, placed_buf_view(other_start, 4096));
        heap.made.set(4, 0x2004);
        heap.quick_is(4, 0x2004);
        heap.keep_window(4, Some(win_buf.clone()), true);

        // Slot 5: non-window slot (must never be touched).
        heap.made.set(5, 0x2005);
        heap.quick_is(5, 0x2005);
        heap.keep_window(5, Some(keep_buf.clone()), false);

        // 1) Window idle for < 300 frames: NOTHING is unpinned (even zeroed slot 3).
        let young = [(win_raw, win_start, win_len, 500u64)];
        assert_eq!(
            heap.unpin_stale_window_slots(&young, 500 + STALE_WINDOW_IDLE_FRAMES - 1, Some(guest_base)),
            0
        );
        assert_eq!(heap.kept_raw_at(3), win_raw);
        assert_eq!(heap.kept_raw_at(4), win_raw);

        // 2) Window idle for >= 300 frames:
        //    - Slot 2 (still references `win_start + 0x1000`) MUST stay intact! (Spec-Trap #16)
        //    - Slot 3 (zeroed) and Slot 4 (overwritten to `other_start`) MUST be unpinned cleanly!
        let epoch_before = super::BUFFER_VIEW_CACHE_EPOCH.load(Ordering::Acquire);
        let idle = [
            (win_raw, win_start, win_len, 100u64),
            (keep_raw, win_start, win_len, 100u64),
        ];
        let unpinned = heap.unpin_stale_window_slots(&idle, 100 + STALE_WINDOW_IDLE_FRAMES, Some(guest_base));
        assert_eq!(unpinned, 2);
        assert!(super::BUFFER_VIEW_CACHE_EPOCH.load(Ordering::Acquire) > epoch_before);

        // Live bindless slot 2 MUST NEVER be nulled.
        assert_eq!(heap.made.get(2), Some(0x2002));
        assert_eq!(heap.quick.get(2, Ordering::Acquire), Some(0x2002));
        assert_eq!(heap.kept_raw_at(2), win_raw);
        assert!(heap.window_slot_bits.get(2));

        // Provably stale slots 3 (zeroed) and 4 (overwritten) MUST be nulled-before-retire.
        for slot in [3u32, 4] {
            assert_eq!(heap.made.get(slot), Some(0));
            assert_eq!(heap.quick.get(slot as usize, Ordering::Acquire), Some(0));
            assert_eq!(heap.kept_raw_at(slot), 0);
            assert!(!heap.twins.get(slot as usize));
            assert!(!heap.window_slot_bits.get(slot as usize));
            assert_eq!(heap.null_kind.get(slot as usize), Some(2));
        }

        // Non-window slot 5 untouched.
        assert_eq!(heap.made.get(5), Some(0x2005));
        assert_eq!(heap.kept_raw_at(5), keep_raw);

        // Repeating revalidation with slot 2 still live does nothing.
        assert_eq!(
            heap.unpin_stale_window_slots(&idle, 1000, Some(guest_base)),
            0
        );
        assert_eq!(heap.kept_raw_at(2), win_raw);

        // Once guest overwrites slot 2 to zero, revalidation unpins it cleanly too.
        guest_heap[2 * 8..2 * 8 + 8].fill(0);
        assert_eq!(
            heap.unpin_stale_window_slots(&idle, 1000, Some(guest_base)),
            1
        );
        assert_eq!(heap.kept_raw_at(2), 0);
    }
}
