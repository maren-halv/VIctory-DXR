//! Copies between our memory and the title's that fail instead of faulting.
//!
//! Title memory may be unmapped, remapped or made read-only by the title's
//! other threads while we copy it, so copies went through the kernel
//! (`ReadProcessMemory`, `NtWriteVirtualMemory`), which reports a missing
//! page instead of faulting: some 35,000 calls a frame, each of which locks
//! the pages it copies, 39 ms of kernel time on the submitting thread alone.
//!
//! [`copy`] is one `rep movsb`. A fault in it is taken by the handler of
//! [`crate::watch`] (`on_write`, the first in the process), which opens a
//! block we guard ourselves and lets the copy go on, and otherwise sends it
//! to its failing exit. `VICTORY_SAFE_COPY=0`: through the kernel, as before.

use std::sync::OnceLock;

std::arch::global_asm!(
    ".globl victory_safe_copy",
    ".globl victory_safe_copy_at",
    ".globl victory_safe_copy_after",
    ".globl victory_safe_copy_failed",
    // rcx destination, rdx source, r8 bytes; eax: whether all were copied.
    // A leaf that leaves the stack alone (the two registers it must give
    // back are kept in volatile ones), so a stack walk through it works.
    "victory_safe_copy:",
    "mov r10, rsi",
    "mov r11, rdi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "victory_safe_copy_at:",
    "rep movsb",
    "victory_safe_copy_after:",
    "mov eax, 1",
    "mov rsi, r10",
    "mov rdi, r11",
    "ret",
    "victory_safe_copy_failed:",
    "xor eax, eax",
    "mov rsi, r10",
    "mov rdi, r11",
    "ret",
    ".globl victory_safe_put",
    ".globl victory_safe_put_at",
    ".globl victory_safe_put_after",
    ".globl victory_safe_put_failed",
    // rcx destination, rdx what was there, r8 what is to be there, r9 blocks
    // of eight dwords: stores the dwords in which the two differ, and no
    // other byte. Uses caller shadow space [rsp + 8 .. rsp + 40) with scalar
    // tzcnt/blsr stores for partial blocks (avoiding microcoded vpmaskmovd).
    // eax: whether all could be stored.
    "victory_safe_put:",
    "test r9, r9",
    "jz 4f",
    "2:",
    "vmovdqu ymm1, ymmword ptr [r8]",
    "vpcmpeqd ymm2, ymm1, ymmword ptr [rdx]",
    "vmovmskps eax, ymm2",
    "xor eax, 255",
    "jz 3f",
    "cmp eax, 255",
    "je 5f",
    "vmovdqu ymmword ptr [rsp + 8], ymm1",
    "6:",
    "tzcnt r10d, eax",
    "mov r11d, dword ptr [rsp + 8 + r10 * 4]",
    "victory_safe_put_at:",
    "mov dword ptr [rcx + r10 * 4], r11d",
    "blsr eax, eax",
    "jnz 6b",
    "jmp 3f",
    "5:",
    "vmovdqu ymmword ptr [rcx], ymm1",
    "victory_safe_put_after:",
    "3:",
    "add rcx, 32",
    "add rdx, 32",
    "add r8, 32",
    "dec r9",
    "jnz 2b",
    "vzeroupper",
    "4:",
    "mov eax, 1",
    "ret",
    "victory_safe_put_failed:",
    "vzeroupper",
    "xor eax, eax",
    "ret",
    ".globl victory_safe_merge",
    ".globl victory_safe_merge_at",
    ".globl victory_safe_merge_after",
    ".globl victory_safe_merge_failed",
    // rcx title memory, rdx what the card holds, r8 what it held when last
    // looked at, r9 blocks of 32 bytes, fifth ([rsp + 40]) where the two
    // counts go. Where a byte of the second differs from the third, it goes
    // into title memory (as whole aligned dwords, and only dwords that then
    // differ from what is there) and into the third; each block of title
    // memory is read just before it is written. Partial blocks use a tight
    // scalar tzcnt/blsr loop over the 8-bit dword mask via caller shadow space
    // [rsp + 8 .. rsp + 40) so rsp/r9/r10/r11 stay exact across any SEH fault.
    // rax: blocks left on fault (0: all done). Counts: changed blocks, dwords stored.
    "victory_safe_merge:",
    "xor r10d, r10d",
    "xor r11d, r11d",
    "test r9, r9",
    "jz 6f",
    "2:",
    "vmovdqu ymm0, ymmword ptr [rdx]",
    "vpcmpeqb ymm1, ymm0, ymmword ptr [r8]",
    "vpmovmskb eax, ymm1",
    "cmp eax, -1",
    "je 5f",
    "victory_safe_merge_at:",
    "vmovdqu ymm2, ymmword ptr [rcx]",
    "vpblendvb ymm3, ymm0, ymm2, ymm1",
    "vpcmpeqd ymm4, ymm3, ymm2",
    "vmovmskps eax, ymm4",
    "xor eax, 255",
    "jz 4f",
    "cmp eax, 255",
    "je 3f",
    "vmovdqu ymmword ptr [rsp + 8], ymm3",
    "vmovq xmm4, rdx",
    "vmovq xmm5, r8",
    "8:",
    "tzcnt edx, eax",
    "mov r8d, dword ptr [rsp + 8 + rdx * 4]",
    "mov dword ptr [rcx + rdx * 4], r8d",
    "inc r11",
    "blsr eax, eax",
    "jnz 8b",
    "vmovq rdx, xmm4",
    "vmovq r8, xmm5",
    "jmp 4f",
    "3:",
    "vmovdqu ymmword ptr [rcx], ymm3",
    "victory_safe_merge_after:",
    "add r11, 8",
    "4:",
    "inc r10",
    "vmovdqu ymmword ptr [r8], ymm0",
    "5:",
    "add rcx, 32",
    "add rdx, 32",
    "add r8, 32",
    "dec r9",
    "jnz 2b",
    "6:",
    "victory_safe_merge_failed:",
    "vzeroupper",
    "mov rax, qword ptr [rsp + 40]",
    "mov qword ptr [rax], r10",
    "mov qword ptr [rax + 8], r11",
    "mov rax, r9",
    "ret",
    ".globl victory_safe_changed",
    ".globl victory_safe_changed_at",
    ".globl victory_safe_changed_after",
    ".globl victory_safe_changed_failed",
    // rcx title memory, rdx records last seen (32 bytes each), r8 record
    // count, r9 bitmap of changed records (bit n: record n). Reads each
    // record of title memory once, compares it with the one last seen, and
    // if it differs stores the new one over it and sets its bit. eax: 1 if
    // every record was read, 0 if a page of title memory faulted (the
    // records before it are done).
    "victory_safe_changed:",
    "xor r10d, r10d",
    "test r8, r8",
    "jz 3f",
    "2:",
    "victory_safe_changed_at:",
    "vmovdqu ymm0, ymmword ptr [rcx]",
    "victory_safe_changed_after:",
    "vpcmpeqb ymm1, ymm0, ymmword ptr [rdx]",
    "vpmovmskb eax, ymm1",
    "cmp eax, -1",
    "je 4f",
    "vmovdqu ymmword ptr [rdx], ymm0",
    "bts qword ptr [r9], r10",
    "4:",
    "add rcx, 32",
    "add rdx, 32",
    "inc r10",
    "cmp r10, r8",
    "jb 2b",
    "3:",
    "vzeroupper",
    "mov eax, 1",
    "ret",
    "victory_safe_changed_failed:",
    "vzeroupper",
    "xor eax, eax",
    "ret",
    ".globl victory_safe_probe_readable",
    ".globl victory_safe_probe_at",
    ".globl victory_safe_probe_after",
    ".globl victory_safe_probe_failed",
    // rcx address, rdx length in bytes, r8 stride (>= 1); eax: 1 if every
    // stride-spaced probe byte and the final byte were readable, 0 on fault.
    // Leaf using only volatile registers so stack unwinding works with zero
    // prologue or epilogue.
    "victory_safe_probe_readable:",
    "test rdx, rdx",
    "jz 3f",
    "2:",
    "victory_safe_probe_at:",
    "movzx eax, byte ptr [rcx]",
    "victory_safe_probe_after:",
    "cmp rdx, 1",
    "jbe 3f",
    "cmp rdx, r8",
    "jbe 4f",
    "add rcx, r8",
    "sub rdx, r8",
    "jmp 2b",
    "4:",
    "lea rcx, [rcx + rdx - 1]",
    "mov rdx, 1",
    "jmp 2b",
    "3:",
    "mov eax, 1",
    "ret",
    "victory_safe_probe_failed:",
    "xor eax, eax",
    "ret",
);

unsafe extern "system" {
    fn victory_safe_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    fn victory_safe_copy_at();
    fn victory_safe_copy_after();
    fn victory_safe_copy_failed();
    fn victory_safe_put(dst: *mut u8, was: *const u8, now: *const u8, blocks: usize) -> u32;
    fn victory_safe_put_at();
    fn victory_safe_put_after();
    fn victory_safe_put_failed();
    fn victory_safe_merge(title: *mut u8, now: *const u8, held: *mut u8, blocks: usize, counts: *mut [u64; 2]) -> usize;
    fn victory_safe_merge_at();
    fn victory_safe_merge_after();
    fn victory_safe_merge_failed();
    fn victory_safe_changed(title: *const u8, seen: *mut u8, records: usize, bits: *mut u64) -> u32;
    fn victory_safe_changed_at();
    fn victory_safe_changed_after();
    fn victory_safe_changed_failed();
    fn victory_safe_probe_readable(addr: *const u8, len: usize, stride: usize) -> u32;
    fn victory_safe_probe_at();
    fn victory_safe_probe_after();
    fn victory_safe_probe_failed();
}

/// Ensures any active Mode-1 GPU-only `PAGE_NOACCESS` tripwire overlapping
/// `[addr, addr + len)` is disarmed and demoted before `safecopy` touches it,
/// so tripwired pages are never misclassified as unmapped zeros (including by
/// `host::read_sparse` fallback paths).
#[inline]
fn ensure_not_tripwired(addr: *const u8, len: usize) {
    if len != 0 && crate::watch::tripwire_active() {
        crate::watch::tripwire_touch(addr as usize, len, false);
    }
}

/// The records of 32 bytes at `title` (`records` of them) that differ from
/// those in `seen`: each is read once, goes over the one in `seen`, and gets
/// its bit in `bits` (record n: bit n % 64 of word n / 64). False if a page
/// of title memory could not be read: then the records before it are done,
/// and the others are as they were.
///
/// # Safety
/// `seen` holds `records * 32` bytes of ours that nothing else reads or
/// writes meanwhile, and `bits` at least `records.div_ceil(64)` words, all
/// 0; `title` is as [`copy`]'s source. [`on`] must have been asked.
pub(crate) unsafe fn changed(title: *const u8, seen: *mut u8, records: usize, bits: &mut [u64]) -> bool {
    assert!(bits.len() >= records.div_ceil(64));
    ensure_not_tripwired(title, records.saturating_mul(32));
    // SAFETY: a fault reading title memory leaves through the failing exit
    // ([`failing_exit`]); the rest is the caller's.
    unsafe { victory_safe_changed(title, seen, records, bits.as_mut_ptr()) != 0 }
}

/// What [`merge`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Merged {
    /// Blocks of 32 bytes in which the card's bytes were new.
    pub(crate) new: u64,
    /// Dwords stored in title memory.
    pub(crate) dwords: u64,
    /// Whether title memory could be read and written wherever it had to be.
    pub(crate) there: bool,
}

/// What shaders wrote goes into title memory: every byte of `now` that
/// differs from the one in `held` goes to `title` and into `held`, every
/// other byte of both stays as it is. Title memory is read 32 bytes at a
/// time, each just before what changes of them is stored, and only the
/// dwords that change are stored (at ends that are no dword's, bytes): the
/// title's own threads may be writing those beside them.
///
/// Where title memory cannot be read or written, the rest of `held` takes
/// `now` if `take` says so (the caller has the card's bytes nowhere else),
/// else it stays.
///
/// # Safety
/// `now` and `held` are `len` bytes of ours, not overlapping, that nothing
/// else writes (but for `now` the card: it is read once); `title` is as
/// [`copy`]'s. The stretch lies in one page of 4 KiB of title memory. [`on`]
/// must have been asked.
pub(crate) unsafe fn merge(title: *mut u8, now: *const u8, held: *mut u8, len: usize, take: bool) -> Merged {
    ensure_not_tripwired(title.cast_const(), len);
    let mut out = Merged { there: true, ..Merged::default() };
    let mut at = 0;
    while at < len {
        let address = title.addr() + at;
        let left = len - at;
        if !address.is_multiple_of(4) || left < 4 {
            // A dword only partly in the stretch: its bytes one by one.
            // SAFETY: inside the stretch.
            unsafe {
                let byte = now.add(at).read_volatile();
                let mut there = 0u8;
                if byte != held.add(at).read() {
                    out.new += 1;
                    let mut ok = copy(&mut there, title.add(at), 1);
                    if ok && there != byte {
                        ok = copy(title.add(at), &byte, 1);
                        out.dwords += u64::from(ok);
                    }
                    out.there &= ok;
                    if ok || take {
                        held.add(at).write(byte);
                    }
                }
            }
            at += 1;
            continue;
        }
        let into = address % 32;
        let mut counts = [0u64; 2];
        if into != 0 || left < 32 {
            // Part of a block: in a block of its own, equal beside the part.
            let n = (32 - into).min(left) & !3;
            let (mut new, mut old) = ([0u8; 32], [0u8; 32]);
            // SAFETY: inside the stretch; the block of title memory around
            // it lies in the same page, and nothing of it outside the part
            // is stored (the two arrays are equal there).
            unsafe {
                std::ptr::copy_nonoverlapping(now.add(at), new.as_mut_ptr().add(into), n);
                std::ptr::copy_nonoverlapping(held.add(at), old.as_mut_ptr().add(into), n);
                let stuck = victory_safe_merge(title.wrapping_add(at).wrapping_sub(into), new.as_ptr(), old.as_mut_ptr(), 1, &mut counts);
                out.there &= stuck == 0;
                let from = if stuck != 0 && take { new.as_ptr() } else { old.as_ptr() };
                std::ptr::copy_nonoverlapping(from.add(into), held.add(at), n);
            }
            at += n;
        } else {
            let blocks = left / 32;
            // SAFETY: whole blocks of the stretch.
            let stuck = unsafe { victory_safe_merge(title.add(at), now.add(at), held.add(at), blocks, &mut counts) };
            if stuck != 0 {
                out.there = false;
                if take {
                    let from = at + (blocks - stuck) * 32;
                    // SAFETY: the blocks that were not done.
                    unsafe { std::ptr::copy_nonoverlapping(now.add(from), held.add(from), stuck * 32) };
                }
            }
            at += blocks * 32;
        }
        out.new += counts[0];
        out.dwords += counts[1];
    }
    out
}

/// Whether the `len` bytes at `a` and `b` differ.
///
/// # Safety
/// Both are `len` bytes that can be read; the card may be writing them.
pub(crate) unsafe fn differ(a: *const u8, b: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256};
    /// # Safety
    /// [`differ`]'s; the processor has AVX2 (as the code above assumes).
    #[target_feature(enable = "avx2")]
    unsafe fn blocks(a: *const u8, b: *const u8, n: usize) -> bool {
        let mut at = 0;
        while at < n {
            let mut any = _mm256_setzero_si256();
            for k in at..n.min(at + 4) {
                // SAFETY: block `k`, inside both.
                let (x, y) = unsafe {
                    (_mm256_loadu_si256(a.add(k * 32).cast::<__m256i>()), _mm256_loadu_si256(b.add(k * 32).cast::<__m256i>()))
                };
                any = _mm256_or_si256(any, _mm256_xor_si256(x, y));
            }
            if _mm256_testz_si256(any, any) == 0 {
                return true;
            }
            at += 4;
        }
        false
    }
    // SAFETY: the caller's.
    if unsafe { blocks(a, b, len / 32) } {
        return true;
    }
    // SAFETY: the bytes behind the blocks.
    (len / 32 * 32..len).any(|i| unsafe { a.add(i).read_volatile() != b.add(i).read_volatile() })
}

/// Puts `now` at `dst` where it differs from `was`, whole dwords (as `dst`
/// is aligned) and not a byte beside them: other threads may be writing
/// those. At the ends, single bytes of a dword only partly in the stretch.
/// False if a page could not be written; then some may have been.
///
/// # Safety
/// As [`copy`]'s.
pub(crate) unsafe fn put_changed(dst: *mut u8, was: &[u8], now: &[u8]) -> bool {
    assert_eq!(was.len(), now.len());
    let n = now.len();
    ensure_not_tripwired(dst.cast_const(), n);
    let head = dst.addr().wrapping_neg() & 3;
    let head = head.min(n);
    let blocks = (n - head) / 32;
    let tail = head + blocks * 32;
    let mut all = true;
    if blocks > 0 {
        // SAFETY: the two slices hold the blocks; the destination's faults
        // leave through the failing exit.
        all &= unsafe { victory_safe_put(dst.add(head), was[head..].as_ptr(), now[head..].as_ptr(), blocks) != 0 };
    }
    let single = |from: usize, to: usize| {
        let mut all = true;
        let mut i = from;
        while i < to {
            // A whole aligned dword, or the bytes there are of one.
            let len = (4 - (dst.addr() + i) % 4).min(to - i);
            if was[i..i + len] != now[i..i + len] {
                // SAFETY: inside the stretch.
                all &= unsafe { copy(dst.add(i), now[i..].as_ptr(), len) };
            }
            i += len;
        }
        all
    };
    all &= single(0, head);
    all &= single(tail, n);
    all
}

/// Whether copies are made here and not by the kernel.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SAFE_COPY").is_none_or(|v| v != "0");
        if on {
            crate::watch::install();
        }
        on
    })
}

/// Copies `len` bytes. False if a page of the source cannot be read or one
/// of the destination cannot be written; then some of the bytes may have
/// been copied.
///
/// # Safety
/// Of the two ranges, what is not title memory must be valid, and no Rust
/// reference may cover the destination. [`on`] must have been asked.
pub(crate) unsafe fn copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    ensure_not_tripwired(src, len);
    ensure_not_tripwired(dst.cast_const(), len);
    // SAFETY: a fault on either side leaves through the failing exit
    // ([`failing_exit`]); the rest is the caller's.
    unsafe { victory_safe_copy(dst, src, len) != 0 }
}

/// Probes `[addr, addr + len)` for readability at `stride`-byte intervals plus
/// the final byte, without mutating or dirtying any page. Returns `true` if
/// `len == 0` or every probed byte was readable without faulting.
///
/// # Safety
/// [`on`] must have been asked (which installs the [`crate::watch`] fault
/// handler). Non-mutating read-only probe; never dirties pages.
pub(crate) unsafe fn probe_readable(addr: *const u8, len: usize, stride: usize) -> bool {
    if len == 0 {
        return true;
    }
    ensure_not_tripwired(addr, len);
    let stride = stride.max(1);
    // SAFETY: a fault reading `addr` leaves through the failing exit
    // ([`failing_exit`]); `stride >= 1` guarantees termination.
    unsafe { victory_safe_probe_readable(addr, len, stride) != 0 }
}

/// Counts how many `stride`-spaced chunks covering `[addr, addr + len)` (where
/// chunk `i` covers `[addr + i * stride, (addr + (i + 1) * stride).min(addr + len))`)
/// are readable.
///
/// # Safety
/// [`on`] must have been asked (which installs the [`crate::watch`] fault
/// handler). Non-mutating read-only probe; never dirties pages.
#[allow(dead_code)]
pub(crate) unsafe fn count_readable(addr: *const u8, len: usize, stride: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let stride = stride.max(1);
    // SAFETY: caller's contract.
    if unsafe { probe_readable(addr, len, stride) } {
        return len.div_ceil(stride);
    }
    let mut count = 0;
    let mut off = 0;
    while off < len {
        let chunk_len = (len - off).min(stride);
        // SAFETY: `probe_readable` traps faults; `wrapping_add` avoids UB on unmapped addresses.
        if unsafe { probe_readable(addr.wrapping_add(off), chunk_len, stride) } {
            count += 1;
        }
        off += chunk_len;
    }
    count
}

/// Returns the total byte length of readable `stride`-spaced chunks covering
/// `[addr, addr + len)`.
///
/// # Safety
/// [`on`] must have been asked (which installs the [`crate::watch`] fault
/// handler). Non-mutating read-only probe; never dirties pages.
pub(crate) unsafe fn readable_bytes(addr: *const u8, len: usize, stride: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let stride = stride.max(1);
    // SAFETY: caller's contract.
    if unsafe { probe_readable(addr, len, stride) } {
        return len;
    }
    let mut total = 0;
    let mut off = 0;
    while off < len {
        let chunk_len = (len - off).min(stride);
        // SAFETY: `probe_readable` traps faults; `wrapping_add` avoids UB on unmapped addresses.
        if unsafe { probe_readable(addr.wrapping_add(off), chunk_len, stride) } {
            total += chunk_len;
        }
        off += chunk_len;
    }
    total
}

/// For the fault handler: where a fault at instruction `at` is to go on, if
/// it is [`copy`]'s.
pub(crate) fn failing_exit(at: usize) -> Option<usize> {
    let (from, to) = (victory_safe_copy_at as *const () as usize, victory_safe_copy_after as *const () as usize);
    if (from..to).contains(&at) {
        return Some(victory_safe_copy_failed as *const () as usize);
    }
    let (from, to) = (victory_safe_put_at as *const () as usize, victory_safe_put_after as *const () as usize);
    if (from..to).contains(&at) {
        return Some(victory_safe_put_failed as *const () as usize);
    }
    let (from, to) = (victory_safe_changed_at as *const () as usize, victory_safe_changed_after as *const () as usize);
    if (from..to).contains(&at) {
        return Some(victory_safe_changed_failed as *const () as usize);
    }
    let (from, to) = (victory_safe_probe_at as *const () as usize, victory_safe_probe_after as *const () as usize);
    if (from..to).contains(&at) {
        return Some(victory_safe_probe_failed as *const () as usize);
    }
    let (from, to) = (victory_safe_merge_at as *const () as usize, victory_safe_merge_after as *const () as usize);
    (from..to).contains(&at).then_some(victory_safe_merge_failed as *const () as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect,
    };

    struct Pages(*mut u8, usize);

    impl Pages {
        fn new(len: usize) -> Pages {
            // SAFETY: plain allocation.
            let at = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
            assert!(!at.is_null());
            Pages(at.cast(), len)
        }
        fn fill(&self) {
            for i in 0..self.1 {
                // SAFETY: inside the allocation.
                unsafe { self.0.add(i).write((i * 31 + 7) as u8) };
            }
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn copies_every_length_and_alignment() {
        assert!(on());
        let (a, b) = (Pages::new(1 << 16), Pages::new(1 << 16));
        a.fill();
        for len in [0usize, 1, 3, 4, 8, 31, 64, 255, 4095, 4096, 4097, 60_000] {
            for (da, sa) in [(0usize, 0usize), (1, 0), (0, 3), (5, 2)] {
                // SAFETY: inside the two allocations.
                unsafe {
                    std::ptr::write_bytes(b.0, 0xEE, b.1);
                    assert!(copy(b.0.add(da), a.0.add(sa), len));
                    let (to, from) = (std::slice::from_raw_parts(b.0, b.1), std::slice::from_raw_parts(a.0, a.1));
                    assert_eq!(to[da..da + len], from[sa..sa + len]);
                    assert!(to[..da].iter().chain(&to[da + len..]).all(|&x| x == 0xEE));
                }
            }
        }
    }

    /// The records that differ get their bits and go over those last seen;
    /// a page that is not there fails, with the records before it done.
    #[test]
    fn changed_records_are_found_and_taken() {
        assert!(on());
        let pages = Pages::new(3 << 12);
        pages.fill();
        let records: usize = (3 << 12) / 32;
        // SAFETY: inside the allocation.
        let title = unsafe { std::slice::from_raw_parts(pages.0, pages.1) }.to_vec();
        let mut seen = title.clone();
        let mut differ = Vec::new();
        for n in [0usize, 1, 63, 64, 65, 200, 383] {
            seen[n * 32 + n % 32] ^= 0x11;
            differ.push(n);
        }
        let mut bits = vec![0u64; records.div_ceil(64)];
        // SAFETY: inside the allocation; `seen` and `bits` are ours.
        assert!(unsafe { changed(pages.0, seen.as_mut_ptr(), records, &mut bits) });
        let found: Vec<usize> = (0..records).filter(|n| bits[n / 64] >> (n % 64) & 1 != 0).collect();
        assert_eq!(found, differ);
        assert_eq!(seen, title);
        bits.fill(0);
        // SAFETY: as above.
        assert!(unsafe { changed(pages.0, seen.as_mut_ptr(), records, &mut bits) });
        assert!(bits.iter().all(|&b| b == 0));
        // The second page is not there: records 0 and 200 (on the third) differ.
        for n in [0usize, 200] {
            seen[n * 32] ^= 0x22;
        }
        // SAFETY: a page of our allocation.
        unsafe { VirtualFree(pages.0.add(4096).cast(), 4096, MEM_DECOMMIT) };
        let mut bits = vec![0u64; records.div_ceil(64)];
        // SAFETY: as above; the page that is not there fails the call.
        assert!(!unsafe { changed(pages.0, seen.as_mut_ptr(), records, &mut bits) });
        let found: Vec<usize> = (0..records).filter(|n| bits[n / 64] >> (n % 64) & 1 != 0).collect();
        assert_eq!(found, [0]);
        assert_eq!(seen[..32], title[..32]);
        assert_ne!(seen[200 * 32..201 * 32], title[200 * 32..201 * 32]);
    }

    /// The dwords that changed are stored, and no byte of any other, at
    /// every alignment and length; a page that is not there fails.
    #[test]
    fn only_changed_dwords_are_put() {
        assert!(on());
        let pages = Pages::new(3 << 12);
        for at in [0usize, 1, 2, 3, 4, 13, 4090] {
            for len in [0usize, 1, 3, 4, 5, 31, 32, 33, 64, 100, 1000, 4096] {
                let was: Vec<u8> = (0..len).map(|i| (i * 13 + 1) as u8).collect();
                for step in [1usize, 5, 17, 64, 10_000] {
                    let mut now = was.clone();
                    for i in (0..len).step_by(step) {
                        now[i] ^= 0x5A;
                    }
                    // Other threads changed everything meanwhile.
                    // SAFETY: inside the allocation.
                    unsafe {
                        std::ptr::write_bytes(pages.0, 0xC3, pages.1);
                        assert!(put_changed(pages.0.add(at), &was, &now));
                        let there = std::slice::from_raw_parts(pages.0, pages.1);
                        let base = pages.0.addr();
                        for (k, &byte) in there.iter().enumerate() {
                            let expect = if (at..at + len).contains(&k) {
                                let dword = (base + k) / 4;
                                let changed = (0..len).any(|i| now[i] != was[i] && (base + at + i) / 4 == dword);
                                if changed { now[k - at] } else { 0xC3 }
                            } else {
                                0xC3
                            };
                            if byte != expect {
                                panic!("at {at} len {len} step {step} byte {k}: {byte:#x}, not {expect:#x}");
                            }
                            if k > at + len + 64 {
                                break;
                            }
                        }
                    }
                }
            }
        }
        // SAFETY: pages of our allocation.
        unsafe {
            VirtualFree(pages.0.add(4096).cast(), 4096, MEM_DECOMMIT);
            let (was, now) = (vec![0u8; 8192], vec![1u8; 8192]);
            assert!(!put_changed(pages.0, &was, &now));
            // Nothing differs on the page that is not there: nothing fails.
            let mut now = was.clone();
            now[0] = 1;
            assert!(put_changed(pages.0, &was, &now));
        }
    }

    /// [`merge`] against what it is to do, byte by byte, for every kind of
    /// start, end and change: the bytes the card changed arrive in title
    /// memory and in what it held, no dword without one is stored.
    #[test]
    fn a_merge_puts_what_the_card_changed_and_nothing_else() {
        assert!(on());
        let pages = Pages::new(1 << 12);
        let mut seed = 1u32;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..4000u32 {
            let at = random() as usize % 100;
            let len = random() as usize % 300;
            let (from, to) = (at, (at + len).min(400));
            let n = to - from;
            let base: Vec<u8> = (0..512).map(|_| random() as u8).collect();
            let was: Vec<u8> = base[from..to].to_vec();
            let mut held = was.clone();
            let mut now = was.clone();
            let mut title = base.clone();
            // The card's writes: dwords and single bytes; the title's: some
            // in the same dwords, some the same values.
            for _ in 0..random() % 12 {
                if n > 0 {
                    let i = random() as usize % n;
                    now[i] = random() as u8;
                }
                let i = random() as usize % 512;
                title[i] = if (from..to).contains(&i) && round % 3 == 0 { now[i - from] } else { random() as u8 };
            }
            if round % 7 == 0 {
                now.iter_mut().for_each(|b| *b = b.wrapping_add(1));
            }
            // SAFETY: inside the allocation and the vectors.
            let done = unsafe {
                std::ptr::copy_nonoverlapping(title.as_ptr(), pages.0, 512);
                merge(pages.0.add(from), now.as_ptr(), held.as_mut_ptr(), n, round % 2 == 0)
            };
            assert!(done.there);
            // SAFETY: inside the allocation.
            let there = unsafe { std::slice::from_raw_parts(pages.0, 512) };
            for i in 0..512 {
                let wrote = (from..to).contains(&i) && now[i - from] != was[i - from];
                let expect = if wrote { now[i - from] } else { title[i] };
                assert_eq!(there[i], expect, "round {round} byte {i} of {from}..{to}");
            }
            assert_eq!(held, now, "round {round}");
            assert_eq!(done.new == 0, was == now);
            // Found merged already.
            // SAFETY: as above.
            let again = unsafe { merge(pages.0.add(from), now.as_ptr(), held.as_mut_ptr(), n, false) };
            assert_eq!(again, Merged { new: 0, dwords: 0, there: true });
        }
    }

    /// Title memory that is not there: what the card wrote stays to be
    /// looked at again, or is taken, as asked; a page that only cannot be
    /// written counts the same.
    #[test]
    fn a_merge_into_missing_memory_fails() {
        assert!(on());
        let pages = Pages::new(1 << 13);
        let mut old = 0;
        for (missing, take) in [(true, false), (true, true), (false, false), (false, true)] {
            for (at, len) in [(0usize, 4096usize), (4, 64), (7, 1), (40, 100)] {
                let was = vec![1u8; len];
                let mut now = was.clone();
                now[len - 1] = 9;
                let mut held = was.clone();
                // SAFETY: our allocation.
                let done = unsafe {
                    if missing {
                        VirtualFree(pages.0.cast(), 4096, MEM_DECOMMIT);
                    } else {
                        VirtualProtect(pages.0.cast(), 4096, PAGE_READONLY, &mut old);
                    }
                    let done = merge(pages.0.add(at), now.as_ptr(), held.as_mut_ptr(), len, take);
                    VirtualAlloc(pages.0.cast(), 4096, MEM_COMMIT, PAGE_READWRITE);
                    VirtualProtect(pages.0.cast(), 4096, PAGE_READWRITE, &mut old);
                    done
                };
                assert!(!done.there);
                assert_eq!(held, if take { now.clone() } else { was.clone() }, "{missing} {take} {at} {len}");
                // Nothing to put: nothing fails.
                let mut held = now.clone();
                // SAFETY: as above; the addresses are no one's.
                let done = unsafe { merge(std::ptr::without_provenance_mut(0x1000 + at), now.as_ptr(), held.as_mut_ptr(), len, take) };
                assert!(done.there && done.new == 0);
            }
        }
    }

    /// [`merge`] beside threads that write other dwords of the same memory:
    /// none of their writes is lost or undone.
    #[test]
    fn merges_undo_no_write_of_other_threads() {
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        assert!(on());
        let pages = Pages::new(1 << 12);
        let base = pages.0 as usize;
        // SAFETY: committed, aligned memory, only touched as atomics or by
        // the merge's stores of whole dwords.
        let dword = |i: usize| unsafe { &*((base + i * 4) as *const AtomicU32) };
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writers: Vec<_> = (0..2usize)
                .map(|w| {
                    let stop = &stop;
                    s.spawn(move || {
                        let mut value = 1u32;
                        while !stop.load(Ordering::Relaxed) {
                            for i in (w..1024).step_by(4) {
                                // This thread's own dword: nobody else's value.
                                assert_eq!(dword(i).load(Ordering::Relaxed), value - 1, "dword {i}");
                                dword(i).store(value, Ordering::Relaxed);
                            }
                            value += 1;
                        }
                    })
                })
                .collect();
            // The card's dwords: every fourth, from the third on.
            let mut held = vec![0u8; 4096];
            for round in 1..3000u32 {
                let mut now = held.clone();
                for i in (2..1024).step_by(4) {
                    if i * 7 % 5 != round as usize % 5 {
                        now[i * 4..i * 4 + 4].copy_from_slice(&(round + i as u32).to_le_bytes());
                    }
                }
                // SAFETY: the allocation, whole.
                let done = unsafe { merge(base as *mut u8, now.as_ptr(), held.as_mut_ptr(), 4096, false) };
                assert!(done.there);
                for i in (2..1024).step_by(4) {
                    let expect = u32::from_le_bytes(now[i * 4..i * 4 + 4].try_into().unwrap());
                    assert_eq!(dword(i).load(Ordering::Relaxed), expect);
                }
                for i in (3..1024).step_by(4) {
                    assert_eq!(dword(i).load(Ordering::Relaxed), 0);
                }
            }
            stop.store(true, Ordering::Relaxed);
            for w in writers {
                w.join().expect("no write was undone");
            }
        });
    }

    #[test]
    fn differences_are_seen_wherever_they_are() {
        for len in [0usize, 1, 31, 32, 33, 127, 128, 129, 200] {
            let a = vec![7u8; len];
            // SAFETY: the vectors' bytes.
            assert!(!unsafe { differ(a.as_ptr(), a.clone().as_ptr(), len) });
            for at in 0..len {
                let mut b = a.clone();
                b[at] = 8;
                // SAFETY: as above.
                assert!(unsafe { differ(a.as_ptr(), b.as_ptr(), len) }, "{len} {at}");
            }
        }
    }

    /// What the copies cost (ignored: a measurement): 64 KiB pieces of
    /// scattered memory into a buffer the card reads, by the kernel and here.
    #[test]
    #[ignore]
    fn copy_costs() {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        assert!(on());
        let Some(host) = crate::host::host() else { return };
        const N: usize = 128 << 20;
        let from = Pages::new(N);
        from.fill();
        let (_buffer, shared) = host.shared_buffer(N as u64).expect("a shared buffer");
        let plain = Pages::new(N);
        plain.fill();
        for (what, to) in [("a buffer the card reads", shared), ("plain memory", plain.0)] {
            for piece in [256usize, 4 << 10, 64 << 10] {
                let count = (N / piece).min(4096);
                // A different, scattered piece each time.
                let at = |i: usize| (i * 7919 % (N / piece)) * piece;
                let time = |f: &dyn Fn(usize) -> bool| {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let t = std::time::Instant::now();
                        for i in 0..count {
                            assert!(f(at(i)));
                        }
                        best = best.min(t.elapsed().as_secs_f64());
                    }
                    (best * 1e6 / count as f64, count as f64 * piece as f64 / best / 1e9)
                };
                // SAFETY: inside the buffers.
                let kernel = time(&|o| unsafe {
                    let mut done = 0;
                    ReadProcessMemory(GetCurrentProcess(), from.0.add(o).cast(), to.add(o).cast(), piece, &mut done) != 0
                });
                // SAFETY: as above.
                let here = time(&|o| unsafe { copy(to.add(o), from.0.add(o), piece) });
                // SAFETY: as above.
                let std_copy = time(&|o| unsafe {
                    std::ptr::copy_nonoverlapping(from.0.add(o), to.add(o), piece);
                    true
                });
                println!(
                    "{piece:>6} bytes into {what}: kernel {:.2} us ({:.1} GB/s), rep movsb {:.2} us ({:.1} GB/s), memcpy {:.2} us ({:.1} GB/s)",
                    kernel.0, kernel.1, here.0, here.1, std_copy.0, std_copy.1
                );
            }
        }
    }

    /// Missing, forbidden and read-only pages: the copy fails, at any byte,
    /// and the process goes on.
    #[test]
    fn faults_are_failures() {
        assert!(on());
        let (a, b) = (Pages::new(3 << 12), Pages::new(3 << 12));
        a.fill();
        let mut old = 0;
        // SAFETY: pages of our allocations.
        unsafe {
            assert!(copy(b.0, a.0, b.1));
            VirtualProtect(a.0.add(4096).cast(), 4096, PAGE_NOACCESS, &mut old);
            assert!(!copy(b.0, a.0, b.1));
            assert!(copy(b.0, a.0, 4096));
            assert!(!copy(b.0, a.0.add(8191), 1));
            assert!(copy(b.0, a.0.add(8192), 4096));
            VirtualProtect(a.0.add(4096).cast(), 4096, PAGE_READWRITE, &mut old);
            VirtualFree(a.0.add(8192).cast(), 4096, MEM_DECOMMIT);
            assert!(!copy(b.0, a.0, b.1));
            VirtualProtect(b.0.add(4096).cast(), 4096, PAGE_READONLY, &mut old);
            assert!(!copy(b.0, a.0, 8192));
            assert!(!copy(std::ptr::without_provenance_mut(0x10), a.0, 4));
            assert!(!copy(b.0, std::ptr::without_provenance(usize::MAX - 7), 4));
            VirtualProtect(b.0.add(4096).cast(), 4096, PAGE_READWRITE, &mut old);
        }
        // Many faults, on several threads at once.
        let at = a.0 as usize;
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(move || {
                    let mut to = [0u8; 64];
                    for _ in 0..500 {
                        // SAFETY: a decommitted page and our own buffer.
                        assert!(!unsafe { copy(to.as_mut_ptr(), (at + 8192) as *const u8, 64) });
                    }
                });
            }
        });
    }

    #[test]
    fn probe_readable_handles_committed_decommitted_noaccess_and_readonly() {
        assert!(on());
        let pages = Pages::new(1 << 16);
        pages.fill();
        let unmapped: *const u8 = std::ptr::without_provenance(0x1000);
        let mut old = 0;

        // SAFETY: `on()` installed the fault handler; all offsets lie within `pages` or test unmapped trapping.
        unsafe {
            // Zero-length always succeeds, even on unmapped addresses or zero stride.
            assert!(probe_readable(pages.0, 0, 4096));
            assert!(probe_readable(pages.0, 0, 0));
            assert!(probe_readable(unmapped, 0, 4096));

            // Single byte, unaligned offsets, and exact page boundaries on committed memory.
            for off in [0usize, 1, 7, 4095] {
                assert!(probe_readable(pages.0.add(off), 1, 4096));
                assert!(probe_readable(pages.0.add(off), 1, 0));
                assert!(probe_readable(pages.0.add(off), 4095, 4096));
                assert!(probe_readable(pages.0.add(off), 4096, 4096));
                assert!(probe_readable(pages.0.add(off), 4097, 4096));
            }
            assert!(probe_readable(pages.0, 1 << 16, 4096));

            // Unmapped address fails.
            assert!(!probe_readable(unmapped, 1, 4096));
            assert!(!probe_readable(unmapped, 4096, 4096));

            // PAGE_READONLY is readable.
            assert_ne!(VirtualProtect(pages.0.add(4096).cast(), 4096, PAGE_READONLY, &mut old), 0);
            assert!(probe_readable(pages.0, 1 << 16, 4096));
            assert!(probe_readable(pages.0.add(4096), 4096, 4096));
            assert_ne!(VirtualProtect(pages.0.add(4096).cast(), 4096, PAGE_READWRITE, &mut old), 0);

            // PAGE_NOACCESS fails whenever touched (start, middle, or exact last byte).
            assert_ne!(VirtualProtect(pages.0.add(2 * 4096).cast(), 4096, PAGE_NOACCESS, &mut old), 0);
            assert!(!probe_readable(pages.0, 1 << 16, 4096));
            assert!(!probe_readable(pages.0.add(2 * 4096), 1, 4096));
            assert!(!probe_readable(pages.0.add(4096), 4097, 4096));
            assert!(probe_readable(pages.0, 2 * 4096, 4096));
            assert!(probe_readable(pages.0.add(3 * 4096), 4096, 4096));
            assert_ne!(VirtualProtect(pages.0.add(2 * 4096).cast(), 4096, PAGE_READWRITE, &mut old), 0);

            // MEM_DECOMMIT in middle.
            assert_ne!(VirtualFree(pages.0.add(8 * 4096).cast(), 4096, MEM_DECOMMIT), 0);
            assert!(!probe_readable(pages.0, 1 << 16, 4096));
            assert!(!probe_readable(pages.0.add(7 * 4096 + 1), 4096, 4096));
            assert!(probe_readable(pages.0, 8 * 4096, 4096));
            assert!(probe_readable(pages.0.add(9 * 4096), 7 * 4096, 4096));

            // MEM_DECOMMIT at start.
            assert_ne!(VirtualFree(pages.0.cast(), 4096, MEM_DECOMMIT), 0);
            assert!(!probe_readable(pages.0, 4096, 4096));
            assert!(!probe_readable(pages.0, 2 * 4096, 4096));
            assert!(probe_readable(pages.0.add(4096), 7 * 4096, 4096));

            // MEM_DECOMMIT at end.
            assert_ne!(VirtualFree(pages.0.add(15 * 4096).cast(), 4096, MEM_DECOMMIT), 0);
            assert!(!probe_readable(pages.0.add(9 * 4096), 7 * 4096, 4096));
            assert!(!probe_readable(pages.0.add(14 * 4096 + 1), 4096, 4096));
            assert!(probe_readable(pages.0.add(9 * 4096), 6 * 4096, 4096));
        }
    }

    #[test]
    fn count_readable_and_readable_bytes_skip_holes_accurately() {
        assert!(on());
        let pages = Pages::new(8 * 4096);
        pages.fill();
        let mut old = 0;

        // SAFETY: `on()` installed the fault handler; all operations are on `pages`.
        unsafe {
            // Zero length and fully-readable fast path.
            assert_eq!(count_readable(pages.0, 0, 4096), 0);
            assert_eq!(readable_bytes(pages.0, 0, 4096), 0);
            assert_eq!(count_readable(pages.0, 8 * 4096, 4096), 8);
            assert_eq!(readable_bytes(pages.0, 8 * 4096, 4096), 8 * 4096);

            // Punch holes: decommit pages 2 and 5, NOACCESS page 4, READONLY page 6.
            assert_ne!(VirtualFree(pages.0.add(2 * 4096).cast(), 4096, MEM_DECOMMIT), 0);
            assert_ne!(VirtualFree(pages.0.add(5 * 4096).cast(), 4096, MEM_DECOMMIT), 0);
            assert_ne!(VirtualProtect(pages.0.add(4 * 4096).cast(), 4096, PAGE_NOACCESS, &mut old), 0);
            assert_ne!(VirtualProtect(pages.0.add(6 * 4096).cast(), 4096, PAGE_READONLY, &mut old), 0);

            // Readable pages: 0, 1, 3, 6 (readonly), 7 => 5 pages.
            assert_eq!(count_readable(pages.0, 8 * 4096, 4096), 5);
            assert_eq!(readable_bytes(pages.0, 8 * 4096, 4096), 5 * 4096);

            // Unaligned tail on readable page 7.
            assert_eq!(count_readable(pages.0, 7 * 4096 + 321, 4096), 5);
            assert_eq!(readable_bytes(pages.0, 7 * 4096 + 321, 4096), 4 * 4096 + 321);

            // Unaligned tail on decommitted page 5.
            assert_eq!(count_readable(pages.0, 5 * 4096 + 321, 4096), 3);
            assert_eq!(readable_bytes(pages.0, 5 * 4096 + 321, 4096), 3 * 4096);

            // Unaligned start + tail: chunk 0 (p0..p1), chunk 6 (p6..p7), and tail chunk 7 (p7) are readable.
            assert_eq!(count_readable(pages.0.add(100), 7 * 4096 + 123, 4096), 3);
            assert_eq!(readable_bytes(pages.0.add(100), 7 * 4096 + 123, 4096), 2 * 4096 + 123);
        }
    }

    #[test]
    fn safecopy_recognizes_tripwire_pages_and_never_zeroes_them() {
        assert!(on());
        let src = Pages::new(4 * 4096);
        let dst = Pages::new(4 * 4096);
        src.fill();
        let id = 0x5AFE_C091u64;
        let arm_retry = |tid: u64| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while !crate::watch::tripwire_arm(tid, src.0 as usize, src.1) {
                assert!(std::time::Instant::now() < deadline, "tripwire_arm timed out");
                std::thread::yield_now();
            }
        };
        arm_retry(id);
        assert!(crate::watch::tripwire_active());
        // SAFETY: both allocations are live; safecopy must disarm the tripwire
        // and read the true bytes rather than treating PAGE_NOACCESS as unmapped.
        unsafe {
            assert!(probe_readable(src.0, src.1, 4096));
        }
        // Re-arm and test `copy()` directly as well.
        arm_retry(id + 1);
        unsafe {
            std::ptr::write_bytes(dst.0, 0, dst.1);
            assert!(copy(dst.0, src.0, src.1));
            let s = std::slice::from_raw_parts(src.0, src.1);
            let d = std::slice::from_raw_parts(dst.0, dst.1);
            assert_eq!(d, s);
        }
        crate::watch::tripwire_disarm_id(id);
        crate::watch::tripwire_disarm_id(id + 1);
    }

    /// Exhaustive parity check over all 256 8-dword bitmasks for scalar
    /// `tzcnt`/`blsr` `victory_safe_merge` & `victory_safe_put`, including
    /// `title == now` no-op store skip on read-only pages.
    #[test]
    fn tzcnt_blsr_merge_and_put_cover_all_256_dword_masks_and_noop_skip() {
        assert!(on());
        let pages = Pages::new(4096);
        let mut old = 0u32;
        for mask in 0u16..=255 {
            let mut title_init = [0x11u8; 32];
            let mut held = [0x22u8; 32];
            let mut now = held;
            for d in 0..8usize {
                title_init[d * 4..d * 4 + 4].copy_from_slice(&(0x1000u32 + d as u32).to_le_bytes());
                if (mask >> d) & 1 != 0 {
                    // Mutate only 1 byte inside the dword (always high-bit set so it never
                    // equals `held` (0x22) or `title_init` (bytes <= 0x10)).
                    let byte_off = d * 4 + (d % 4);
                    now[byte_off] = 0x80 | ((d as u8).wrapping_mul(13).wrapping_add(mask as u8) & 0x7F);
                }
            }
            // SAFETY: inside our committed 4 KiB page.
            let done = unsafe {
                std::ptr::copy_nonoverlapping(title_init.as_ptr(), pages.0, 32);
                merge(pages.0, now.as_ptr(), held.as_mut_ptr(), 32, false)
            };
            assert!(done.there);
            assert_eq!(done.new, u64::from(mask != 0), "mask {mask:#04x}");
            assert_eq!(done.dwords, u64::from(mask.count_ones()), "mask {mask:#04x}");
            assert_eq!(held, now);
            // SAFETY: inside our committed page.
            let got = unsafe { std::slice::from_raw_parts(pages.0, 32) };
            for i in 0..32 {
                let d = i / 4;
                let expect = if (mask >> d) & 1 != 0 && i == d * 4 + (d % 4) {
                    now[i]
                } else {
                    title_init[i]
                };
                assert_eq!(got[i], expect, "mask {mask:#04x} byte {i}");
            }

            // `put_changed` parity on the same 8-dword mask.
            let was_put = [0x33u8; 32];
            let mut now_put = was_put;
            for d in 0..8usize {
                if (mask >> d) & 1 != 0 {
                    now_put[d * 4..d * 4 + 4].copy_from_slice(&(0x5000u32 + d as u32).to_le_bytes());
                }
            }
            // SAFETY: inside our committed page.
            unsafe {
                std::ptr::write_bytes(pages.0, 0xCC, 32);
                assert!(put_changed(pages.0, &was_put, &now_put));
                let got_put = std::slice::from_raw_parts(pages.0, 32);
                for d in 0..8usize {
                    let expect = if (mask >> d) & 1 != 0 {
                        &now_put[d * 4..d * 4 + 4]
                    } else {
                        &[0xCC; 4][..]
                    };
                    assert_eq!(&got_put[d * 4..d * 4 + 4], expect, "put mask {mask:#04x} dword {d}");
                }
            }
        }

        // `title == now` no-op skip invariant (V2): even when `now != held`, if
        // title memory already holds `now`, `merge` must issue ZERO stores (so
        // it succeeds without faulting on a `PAGE_READONLY` page!).
        let mut held = [0x01u8; 32];
        let now = [0x77u8; 32];
        // SAFETY: our committed page, protected RO during merge and restored to RW.
        unsafe {
            std::ptr::copy_nonoverlapping(now.as_ptr(), pages.0, 32);
            assert_ne!(VirtualProtect(pages.0.cast(), 4096, PAGE_READONLY, &mut old), 0);
            let done = merge(pages.0, now.as_ptr(), held.as_mut_ptr(), 32, false);
            assert_ne!(VirtualProtect(pages.0.cast(), 4096, PAGE_READWRITE, &mut old), 0);
            assert_eq!(done, Merged { new: 1, dwords: 0, there: true });
            assert_eq!(held, now);
        }
    }
}
