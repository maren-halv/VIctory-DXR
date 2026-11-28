//! Renderer port #2 (`VICTORY_PORT_DESCRIPTORS=1`, read in `victory-gpu`'s
//! `port_descriptors`): the title's own code that copies descriptor records
//! into its shader-visible heap as bytes tells the layer which slots it wrote.
//!
//! What the title does (the target executable): each draw's and each
//! dispatch's descriptor tables are allocated from its frame allocator
//! (image +0x4d9f400; the allocator's memory is the placed shader-visible
//! view heap) and filled by two functions, +0x4d9efd0 (graphics) and
//! +0x4eec8e0 (compute), three loops each, one record at a time:
//! `and r, 7FE0h` (the slot's byte offset in the table), `vmovups ymm0,
//! [record]`, `vmovups [table + r], ymm0`. Found with `dumpbin /disasm` from
//! the write faults of run faults-6 (book records "the instruction at a made
//! the fault"); a search of all of `.text` for the same instruction shape
//! finds these six stores and no other.
//!
//! Each store becomes a jump to code of ours that does the store, then, if
//! the address lies in the heap `victory-gpu` names in [`Marks`], sets the
//! slot's byte in a map of slots and its 4 KiB chunk's byte in a map of
//! chunks, and jumps back. The flags are clobbered: at every site the next
//! instructions are a `movzx` and a `cmp`, which sets them before any use
//! (checked as part of the original bytes). Every register is put back.
//!
//! Each site's original bytes (from the `and` to the `cmp`) are compared
//! before anything is written; a site whose bytes differ is left alone and
//! said so (another build of the title).

/// The cells the stubs read, `#[repr(C)]` in `victory-gpu`: the heap's first
/// byte, its length in bytes (0: none, nothing is marked), the map of slots
/// (a byte each), the map of chunks (a byte for 128 slots).
pub const MARKS_LEN: usize = 32;

/// A store of the title's to hook: the window's offset in the image, the
/// store's offset in the window and its length, and the window's bytes.
pub struct Site {
    pub window: usize,
    pub store: usize,
    pub len: usize,
    pub bytes: &'static [u8],
}

/// The ten record/allocator stores of the title's heap writers
/// (the target executable 1.0.2.1, image offsets):
/// - 6 per-record `vmovups [base+index], ymm0` table fillers (`+0x4d9efd0`, `+0x4eec8e0`)
/// - 1 4x32B block `vmovups [rbx+0x60..0x00], ymm3..0` store (`+0x4d9f25f`)
/// - 3 heap-resident allocator/header stores (`+0x18849c`, `+0x4e9b4e2`, `+0x2474afc`)
pub const SITES: [Site; 10] = [
    // +0x4d9efd0, loop 1: [rcx+rsi]
    Site { window: 0x4d9f04f, store: 10, len: 5, bytes: &[0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6, 0x47, 0x0b, 0x39, 0xc5] },
    // loop 2: [rcx+rsi]
    Site { window: 0x4d9f10d, store: 10, len: 5, bytes: &[0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc5, 0xfc, 0x11, 0x04, 0x31, 0x0f, 0xb6, 0x47, 0x0f, 0x41, 0x39, 0xc5] },
    // loop 3: [r9+rsi]
    Site { window: 0x4d9f19c, store: 12, len: 6, bytes: &[0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc4, 0xc1, 0x7c, 0x10, 0x02, 0xc4, 0xc1, 0x7c, 0x11, 0x04, 0x31, 0x44, 0x0f, 0xb6, 0x4f, 0x0d, 0x44, 0x39, 0xc8] },
    // +0x4eec8e0, loop 1: [r9+r8]
    Site { window: 0x4eec95d, store: 11, len: 6, bytes: &[0x41, 0x81, 0xe1, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x01, 0x0f, 0xb6, 0x42, 0x0b, 0x39, 0xc5] },
    // loop 2: [rdi+r8]
    Site { window: 0x4eeca08, store: 10, len: 6, bytes: &[0x81, 0xe7, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x06, 0xc4, 0xa1, 0x7c, 0x11, 0x04, 0x07, 0x0f, 0xb6, 0x72, 0x0d, 0x39, 0xf0] },
    // loop 3: [r10+r8]
    Site { window: 0x4eeca82, store: 11, len: 6, bytes: &[0x41, 0x81, 0xe2, 0xe0, 0x7f, 0x00, 0x00, 0xc5, 0xfc, 0x10, 0x00, 0xc4, 0x81, 0x7c, 0x11, 0x04, 0x02, 0x0f, 0xb6, 0x42, 0x0f, 0x41, 0x39, 0xc6] },
    // +0x4d9f25f: 4x32B block copy `vmovups [rbx+0x60..0x00], ymm3..0` (4 contiguous slots)
    Site {
        window: 0x4d9f24a,
        store: 21,
        len: 19,
        bytes: &[
            0xc4, 0xc1, 0x7c, 0x10, 0x4c, 0x24, 0x20, // vmovups ymm1, [r12+0x20]
            0xc4, 0xc1, 0x7c, 0x10, 0x54, 0x24, 0x40, // vmovups ymm2, [r12+0x40]
            0xc4, 0xc1, 0x7c, 0x10, 0x5c, 0x24, 0x60, // vmovups ymm3, [r12+0x60]
            0xc5, 0xfc, 0x11, 0x5b, 0x60, // vmovups [rbx+0x60], ymm3
            0xc5, 0xfc, 0x11, 0x53, 0x40, // vmovups [rbx+0x40], ymm2
            0xc5, 0xfc, 0x11, 0x4b, 0x20, // vmovups [rbx+0x20], ymm1
            0xc5, 0xfc, 0x11, 0x03, // vmovups [rbx], ymm0
            0x4c, 0x89, 0xf9, // mov rcx, r15
            0x48, 0x89, 0xf2, // mov rdx, rsi
            0x49, 0x89, 0xd8, // mov r8, rbx
            0x45, 0x31, 0xc9, // xor r9d, r9d
        ],
    },
    // +0x18849c: `mov [rbx], rdx ; mov [rcx+rax+0x48], rbx`
    Site {
        window: 0x188493,
        store: 9,
        len: 8,
        bytes: &[
            0x48, 0x89, 0x57, 0x10, // mov [rdi+0x10], rdx
            0x48, 0x8b, 0x54, 0x01, 0x48, // mov rdx, [rcx+rax+0x48]
            0x48, 0x89, 0x13, // mov [rbx], rdx
            0x48, 0x89, 0x5c, 0x01, 0x48, // mov [rcx+rax+0x48], rbx
            0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, // movabs rdx, 0xffffffff00000000
        ],
    },
    // +0x4e9b4e2: `mov qword ptr [r14], 0`
    Site {
        window: 0x4e9b4da,
        store: 8,
        len: 7,
        bytes: &[
            0x4d, 0x8d, 0x74, 0x07, 0xe8, // lea r14, [r15+rax-0x18]
            0x48, 0x89, 0xd9, // mov rcx, rbx
            0x49, 0xc7, 0x06, 0x00, 0x00, 0x00, 0x00, // mov qword ptr [r14], 0
            0x49, 0x89, 0x5e, 0x08, // mov [r14+8], rbx
        ],
    },
    // +0x2474afc: `mov [rbx], rdx ; mov [rcx+rax+0x48], rbx`
    Site {
        window: 0x2474af3,
        store: 9,
        len: 8,
        bytes: &[
            0x48, 0x89, 0x57, 0x10, // mov [rdi+0x10], rdx
            0x48, 0x8b, 0x54, 0x01, 0x48, // mov rdx, [rcx+rax+0x48]
            0x48, 0x89, 0x13, // mov [rbx], rdx
            0x48, 0x89, 0x5c, 0x01, 0x48, // mov [rcx+rax+0x48], rbx
            0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, // movabs rdx, 0xffffffff00000000
        ],
    },
];

/// `lea rax, [base+index]` for a `vmovups [base+index], ymm0` (VEX, ModRM
/// 04h with a SIB byte, no displacement, the stack pointer not its base):
/// `None` for another form.
fn lea_of(store: &[u8]) -> Option<[u8; 4]> {
    let (x, b, op_at) = match *store.first()? {
        0xc5 => (false, false, 2),
        0xc4 => {
            let p = *store.get(1)?;
            if p & 0x1f != 1 {
                return None;
            }
            (p & 0x40 == 0, p & 0x20 == 0, 3)
        }
        _ => return None,
    };
    let (op, modrm, sib) = (*store.get(op_at)?, *store.get(op_at + 1)?, *store.get(op_at + 2)?);
    if op != 0x11 || modrm != 0x04 || store.len() != op_at + 3 {
        return None;
    }
    // No base (mode 0, base 101), or the stack pointer as base, or rsp as index (none).
    if sib & 7 == 5 || (sib & 7 == 4 && !b) || (sib >> 3 & 7 == 4 && !x) {
        return None;
    }
    Some([0x48 | u8::from(x) << 1 | u8::from(b), 0x8d, 0x04, sib])
}

/// Decoded target address expression for a hooked store instruction sequence.
enum StoreTarget {
    /// `vmovups [base+index], ymm0`: 4-byte `lea rax, [base+index]` (flags may be clobbered).
    Lea([u8; 4]),
    /// Direct base register (`mov rax, reg`, 3 bytes) writing `slots` consecutive 32-byte slots
    /// (preserves `rflags` via `pushfq`/`popfq`).
    BaseReg { mov_rax: [u8; 3], slots: u8 },
}

fn target_of(store: &[u8]) -> Option<StoreTarget> {
    if let Some(lea) = lea_of(store) {
        return Some(StoreTarget::Lea(lea));
    }
    match store {
        // +0x4d9f25f: 4x32B `vmovups [rbx+0x60..0x00], ymm3..0`
        [
            0xc5, 0xfc, 0x11, 0x5b, 0x60,
            0xc5, 0xfc, 0x11, 0x53, 0x40,
            0xc5, 0xfc, 0x11, 0x4b, 0x20,
            0xc5, 0xfc, 0x11, 0x03,
        ] => Some(StoreTarget::BaseReg { mov_rax: [0x48, 0x89, 0xd8], slots: 4 }), // mov rax, rbx
        // +0x18849c & +0x2474afc: `mov [rbx], rdx ; mov [rcx+rax+0x48], rbx`
        [0x48, 0x89, 0x13, 0x48, 0x89, 0x5c, 0x01, 0x48] => {
            Some(StoreTarget::BaseReg { mov_rax: [0x48, 0x89, 0xd8], slots: 1 }) // mov rax, rbx
        }
        // +0x4e9b4e2: `mov qword ptr [r14], 0`
        [0x49, 0xc7, 0x06, 0x00, 0x00, 0x00, 0x00] => {
            Some(StoreTarget::BaseReg { mov_rax: [0x4c, 0x89, 0xf0], slots: 1 }) // mov rax, r14
        }
        _ => None,
    }
}

/// The code at `to` for the store `store` at `at`: the store, the marks, back
/// to `at + store.len()`.
///
/// TSO invariant: every stub writes `slots[s] = 1` followed UNCONDITIONALLY by
/// `chunks[s / 128] = 1` (never test-before-write on `chunks[]`).
fn stub(store: &[u8], at: usize, to: usize, marks: usize) -> Option<Vec<u8>> {
    let target = target_of(store)?;
    let mut out = store.to_vec();
    match target {
        StoreTarget::Lea(lea) => {
            out.push(0x50); // push rax
            out.extend(lea); // lea rax, [base+index]
            out.extend([0x52, 0x51]); // push rdx ; push rcx
            out.extend([0x48, 0xba]); // mov rdx, marks
            out.extend((marks as u64).to_le_bytes());
            out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
            out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
            out.extend([0x73, 0x18]); // jae past the marks
            out.extend([0x48, 0xc1, 0xe8, 0x05]); // shr rax, 5: the slot
            out.extend([0x48, 0x8b, 0x4a, 0x10]); // mov rcx, [rdx+16]
            out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
            out.extend([0x48, 0xc1, 0xe8, 0x07]); // shr rax, 7: the chunk
            out.extend([0x48, 0x8b, 0x4a, 0x18]); // mov rcx, [rdx+24]
            out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
            out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
        }
        StoreTarget::BaseReg { mov_rax, slots: 1 } => {
            out.push(0x9c); // pushfq
            out.push(0x50); // push rax
            out.extend(mov_rax); // mov rax, base_reg
            out.extend([0x52, 0x51]); // push rdx ; push rcx
            out.extend([0x48, 0xba]); // mov rdx, marks
            out.extend((marks as u64).to_le_bytes());
            out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
            out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
            out.extend([0x73, 0x18]); // jae past the marks
            out.extend([0x48, 0xc1, 0xe8, 0x05]); // shr rax, 5: the slot
            out.extend([0x48, 0x8b, 0x4a, 0x10]); // mov rcx, [rdx+16]
            out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
            out.extend([0x48, 0xc1, 0xe8, 0x07]); // shr rax, 7: the chunk
            out.extend([0x48, 0x8b, 0x4a, 0x18]); // mov rcx, [rdx+24]
            out.extend([0xc6, 0x04, 0x01, 0x01]); // mov byte [rcx+rax], 1
            out.extend([0x59, 0x5a, 0x58]); // pop rcx ; pop rdx ; pop rax
            out.push(0x9d); // popfq
        }
        StoreTarget::BaseReg { mov_rax, slots } => {
            out.push(0x9c); // pushfq
            out.extend([0x50, 0x52, 0x51, 0x56]); // push rax ; push rdx ; push rcx ; push rsi
            out.extend([0x48, 0xba]); // mov rdx, marks
            out.extend((marks as u64).to_le_bytes());
            out.extend(mov_rax); // mov rax, base_reg
            out.extend([0x48, 0x2b, 0x02]); // sub rax, [rdx]
            out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
            let jae_outer = out.len();
            out.extend([0x73, 0x00]); // jae done (patched below)
            out.extend([0xbe, slots, 0x00, 0x00, 0x00]); // mov esi, slots
            let loop_top = out.len();
            out.extend([0x48, 0x3b, 0x42, 0x08]); // cmp rax, [rdx+8]
            let jae_inner = out.len();
            out.extend([0x73, 0x00]); // jae done (patched below)
            out.extend([0x48, 0x89, 0xc1]); // mov rcx, rax
            out.extend([0x48, 0xc1, 0xe9, 0x05]); // shr rcx, 5: the slot
            out.push(0x50); // push rax
            out.extend([0x48, 0x8b, 0x42, 0x10]); // mov rax, [rdx+16]
            out.extend([0xc6, 0x04, 0x08, 0x01]); // mov byte [rax+rcx], 1
            out.extend([0x48, 0xc1, 0xe9, 0x07]); // shr rcx, 7: the chunk
            out.extend([0x48, 0x8b, 0x42, 0x18]); // mov rax, [rdx+24]
            out.extend([0xc6, 0x04, 0x08, 0x01]); // mov byte [rax+rcx], 1
            out.push(0x58); // pop rax
            out.extend([0x48, 0x83, 0xc0, 0x20]); // add rax, 32
            out.extend([0xff, 0xce]); // dec esi
            let jnz_at = out.len();
            let back_rel = (loop_top as isize) - ((jnz_at + 2) as isize);
            out.extend([0x75, i8::try_from(back_rel).ok()? as u8]); // jnz loop_top
            let done_at = out.len();
            out[jae_outer + 1] = u8::try_from(done_at - (jae_outer + 2)).ok()?;
            out[jae_inner + 1] = u8::try_from(done_at - (jae_inner + 2)).ok()?;
            out.extend([0x5e, 0x59, 0x5a, 0x58]); // pop rsi ; pop rcx ; pop rdx ; pop rax
            out.push(0x9d); // popfq
        }
    }
    let back = crate::console_insn::jump(to + out.len(), at + store.len())?;
    out.extend(back);
    Some(out)
}

/// What became of a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hooked {
    Done,
    /// Its bytes are not the ones expected: left alone.
    Differs,
    /// Outside the image, no room for code near it, or Windows refused.
    Failed(&'static str),
}

/// Hooks `sites` in the image of `size` bytes at `base`; the stubs read the
/// cells at `marks` ([`MARKS_LEN`] bytes). What became of each, said in the
/// log too.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet; `marks` stays
/// valid for as long as the image runs.
pub unsafe fn apply(base: usize, size: usize, marks: usize, sites: &[Site]) -> Vec<Hooked> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut room: Option<(usize, usize)> = None;
    let mut out = Vec::new();
    for site in sites {
        let done = (|| {
            if site.window.checked_add(site.bytes.len()).is_none_or(|end| end > size) || site.store + site.len > site.bytes.len() {
                return Hooked::Failed("outside the image");
            }
            let window = base + site.window;
            // SAFETY: inside the mapped image (checked above).
            let there = unsafe { std::slice::from_raw_parts(window as *const u8, site.bytes.len()) };
            if there != site.bytes {
                return Hooked::Differs;
            }
            let at = window + site.store;
            if room.is_none_or(|(place, used)| crate::console_insn::jump(at, place + used).is_none()) {
                room = crate::console_insn::near(at).map(|place| (place, 0));
            }
            let Some((place, used)) = room.as_mut() else {
                return Hooked::Failed("no room for code near the title's");
            };
            let to = *place + *used;
            let store = &site.bytes[site.store..site.store + site.len];
            let (Some(body), Some(jump)) = (stub(store, at, to, marks), crate::console_insn::jump(at, to)) else {
                return Hooked::Failed("not a store of the expected form, or out of reach");
            };
            if *used + body.len() > crate::console_insn::STUBS_LEN {
                return Hooked::Failed("no room for code near the title's");
            }
            let mut old = 0;
            // SAFETY: our own code memory, which nothing runs yet; then the
            // store, whose page is made writable for it and put back, and
            // which nothing runs either.
            unsafe {
                std::ptr::copy_nonoverlapping(body.as_ptr(), to as *mut u8, body.len());
                if VirtualProtect(at as *const _, site.len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
                    return Hooked::Failed("Windows would not make it writable");
                }
                std::ptr::write_bytes(at as *mut u8, 0x90, site.len);
                std::ptr::copy_nonoverlapping(jump.as_ptr(), at as *mut u8, jump.len());
                VirtualProtect(at as *const _, site.len, old, &mut old);
                FlushInstructionCache(GetCurrentProcess(), at as *const _, site.len);
            }
            *used += body.len().next_multiple_of(16);
            Hooked::Done
        })();
        let offset = format_args!("{:#x}", site.window + site.store);
        match done {
            Hooked::Done => tracing::info!(site = %offset, "renderer port 2: the title's descriptor record store at title+site patched to tell the layer its slot (VICTORY_PORT_DESCRIPTORS)"),
            Hooked::Differs => tracing::warn!(site = %offset, "renderer port 2: not patched, the title's bytes there are not the ones expected (another build?)"),
            Hooked::Failed(why) => tracing::warn!(site = %offset, why, "renderer port 2: not patched"),
        }
        out.push(done);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sites_lea_their_stores_address() {
        let leas: Vec<[u8; 4]> = SITES[..6].iter().map(|s| lea_of(&s.bytes[s.store..s.store + s.len]).unwrap()).collect();
        assert_eq!(leas[0], [0x48, 0x8d, 0x04, 0x31]); // lea rax, [rcx+rsi]
        assert_eq!(leas[1], [0x48, 0x8d, 0x04, 0x31]);
        assert_eq!(leas[2], [0x49, 0x8d, 0x04, 0x31]); // [r9+rsi]
        assert_eq!(leas[3], [0x4b, 0x8d, 0x04, 0x01]); // [r9+r8]
        assert_eq!(leas[4], [0x4a, 0x8d, 0x04, 0x07]); // [rdi+r8]
        assert_eq!(leas[5], [0x4b, 0x8d, 0x04, 0x02]); // [r10+r8]
        // Each table-filler window: the store follows the record's load and is
        // followed by a movzx and the cmp that sets the flags.
        for s in &SITES[..6] {
            let after = &s.bytes[s.store + s.len..];
            let movzx = if after[0] == 0x0f { 0 } else { 1 };
            assert_eq!(&after[movzx..movzx + 2], &[0x0f, 0xb6]);
            assert_eq!(after[after.len() - 2], 0x39, "cmp");
        }
        // All 10 sites decode cleanly via `target_of`.
        for s in &SITES {
            assert!(target_of(&s.bytes[s.store..s.store + s.len]).is_some());
        }
        // A store with a displacement, of the stack, of another kind: not taken.
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x44, 0x31, 0x08]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x11, 0x04, 0x24]).is_none());
        assert!(lea_of(&[0xc5, 0xfc, 0x10, 0x04, 0x31]).is_none());
    }

    /// The marks `victory-gpu` keeps, for a test.
    #[repr(C)]
    struct Marks {
        start: u64,
        len: u64,
        slots: *mut u8,
        chunks: *mut u8,
    }

    /// Each site's window in code of a test's, twice, one copy hooked, both
    /// run: the record is stored as before, the registers the window leaves
    /// are the same, and the slot and its chunk are marked when the table is
    /// in the heap, not otherwise.
    #[test]
    fn a_hooked_site_stores_and_marks_its_slot() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // mov to, from (64 bits)
        let mov = |to: u8, from: u8| [0x48 | (from >> 3) << 2 | (to >> 3), 0x89, 0xc0 | (from & 7) << 3 | (to & 7)];
        for site in &SITES[..6] {
            let store = &site.bytes[site.store..site.store + site.len];
            let lea = lea_of(store).unwrap();
            let base = lea[3] & 7 | (lea[0] & 1) << 3;
            let index = lea[3] >> 3 & 7 | (lea[0] & 2) << 2;
            let load_at = site.store - if site.bytes[site.store - 5] == 0xc4 { 5 } else { 4 };
            let load = &site.bytes[load_at..site.store];
            let load_reg = if load[0] == 0xc4 { load[4] & 7 | 8 } else { load[3] & 7 };
            // fn(offset: rcx, table: rdx, record: r8, scratch: r9) -> rax:
            // the store's address as the window leaves its registers.
            let mut code: Vec<u8> = vec![0x56, 0x57, 0x53]; // push rsi ; push rdi ; push rbx
            code.extend(mov(11, 1)); // r11 = offset
            code.extend(mov(3, 2)); // rbx = table
            code.extend(mov(6, 8)); // rsi = record
            for scratch in [7, 10, 2] {
                code.extend(mov(scratch, 9)); // rdi, r10, rdx: what the movzx reads
            }
            // The window's `and` masks the offset's register; the table is in
            // the other. (The record first: rsi may be one of them.)
            let rex = usize::from(site.bytes[0] == 0x41);
            let offset = site.bytes[rex + 1] & 7 | if rex == 1 { 8 } else { 0 };
            assert!(offset == base || offset == index);
            let table = if offset == base { index } else { base };
            code.extend(mov(load_reg, 6));
            code.extend(mov(table, 3));
            code.extend(mov(offset, 11));
            let at = code.len();
            code.extend(site.bytes);
            code.extend(lea); // lea rax, [base+index]
            code.extend([0x5b, 0x5f, 0x5e]); // pop rbx ; pop rdi ; pop rsi
            code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
            type F = unsafe extern "system" fn(u64, *mut u8, *const u8, *mut u8) -> u64;
            // SAFETY: fresh memory for code and data, one copy changed before it runs.
            unsafe {
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                assert_ne!(place, 0);
                let (plain, hooked) = (place + 0x100 - at, place + 0x1100 - at);
                std::ptr::copy_nonoverlapping(code.as_ptr(), plain as *mut u8, code.len());
                std::ptr::copy_nonoverlapping(code.as_ptr(), hooked as *mut u8, code.len());
                let heap = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
                let scratch = [0u8; 64];
                let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
                let marks = Box::new(Marks { start: heap as u64 + 0x2000, len: 0x8000, slots: slots.as_mut_ptr(), chunks: chunks.as_mut_ptr() });
                let mine = [Site { window: 0x1100, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                assert_eq!(*((place + 0x1100 + site.store) as *const u8), 0xe9);
                let (plain, hooked): (F, F) = (std::mem::transmute(plain), std::mem::transmute(hooked));
                let record: [u8; 32] = std::array::from_fn(|k| k as u8 ^ 0x5a);
                // In the heap: slot 0x40 + 3 of its start.
                let table = heap.add(0x2000 + 0x40 * 32);
                let first = plain(0x60 | 0x8000_0000, table.add(0x4000), record.as_ptr(), scratch.as_ptr().cast_mut());
                let second = hooked(0x60 | 0x8000_0000, table, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(first.wrapping_sub(0x4000), second, "the registers the window leaves");
                assert_eq!(std::slice::from_raw_parts(table.add(0x60), 32), &record);
                assert_eq!(std::slice::from_raw_parts(table.add(0x4060), 32), &record);
                let slot = 0x40 + 3;
                assert_eq!(slots[slot], 1);
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 1);
                assert_eq!(chunks[slot / 128], 1);
                // Before the heap: stored, not marked.
                slots.fill(0);
                chunks.fill(0);
                let below = heap.add(0x1000);
                hooked(0x20, below, record.as_ptr(), scratch.as_ptr().cast_mut());
                assert_eq!(std::slice::from_raw_parts(below.add(0x20), 32), &record);
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
                drop(marks);
            }
        }
    }

    #[test]
    fn other_bytes_are_left_alone() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // SAFETY: fresh memory of our own.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            let mut bytes = SITES[0].bytes.to_vec();
            bytes[3] = 0x7e; // and ecx, 7EE0h
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), place.add(0x100), bytes.len());
            let mine = [Site { window: 0x100, ..SITES[0] }];
            assert_eq!(apply(place as usize, 0x10000, 0, &mine), [Hooked::Differs]);
            assert_eq!(std::slice::from_raw_parts(place.add(0x100), bytes.len()), &bytes[..]);
            let outside = [Site { window: 0xfff0, ..SITES[0] }];
            assert!(matches!(apply(place as usize, 0x10000, 0, &outside)[..], [Hooked::Failed(_)]));
        }
    }

    /// Executes the four newly added writer sites (+0x4d9f25f 4x32B block store,
    /// +0x18849c / +0x2474afc [rbx] store, +0x4e9b4e2 [r14] store) in live
    /// JIT buffers and verifies exact writes, mark bits, chunk boundaries, and
    /// bounds safety.
    #[test]
    fn new_sites_store_and_mark_slots() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
        // SAFETY: allocates fresh RWX buffers exclusively for this test.
        unsafe {
            let heap = VirtualAlloc(std::ptr::null(), 0x20000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            assert!(!heap.is_null());
            let (mut slots, mut chunks) = (vec![0u8; 0x8000 / 32], vec![0u8; 0x8000 / 32 / 128]);
            let marks = Box::new(Marks {
                start: heap as u64 + 0x2000,
                len: 0x8000,
                slots: slots.as_mut_ptr(),
                chunks: chunks.as_mut_ptr(),
            });

            // 1) Site 6 (+0x4d9f25f): 4x32B vmovups [rbx+0x60..0x00] crossing chunk boundary (slots 126..130).
            {
                let site = &SITES[6];
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                let mut code: Vec<u8> = Vec::new();
                // fn(dst_rbx: rcx, src_r12: rdx)
                code.extend([0x53, 0x41, 0x54]); // push rbx ; push r12
                code.extend([0x48, 0x89, 0xcb]); // mov rbx, rcx
                code.extend([0x49, 0x89, 0xd4]); // mov r12, rdx
                code.extend([0xc4, 0xc1, 0x7c, 0x10, 0x04, 0x24]); // vmovups ymm0, [r12]
                let win_at = code.len();
                code.extend(site.bytes);
                code.extend([0x41, 0x5c, 0x5b]); // pop r12 ; pop rbx
                code.extend([0xc5, 0xf8, 0x77, 0xc3]); // vzeroupper ; ret
                let entry = place + 0x200 - win_at;
                std::ptr::copy_nonoverlapping(code.as_ptr(), entry as *mut u8, code.len());
                let mine = [Site { window: 0x200, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                let f: unsafe extern "system" fn(*mut u8, *const u8) = std::mem::transmute(entry);
                let src: [u8; 128] = std::array::from_fn(|k| (k as u8).wrapping_mul(3) ^ 0xa5);
                let dst = heap.add(0x2000 + 126 * 32);
                f(dst, src.as_ptr());
                assert_eq!(std::slice::from_raw_parts(dst, 128), &src);
                for s in 126..130 {
                    assert_eq!(slots[s], 1, "slot {s} must be marked");
                }
                assert_eq!(slots.iter().filter(|&&b| b != 0).count(), 4);
                assert_eq!(chunks[0], 1);
                assert_eq!(chunks[1], 1);
                // Out-of-bounds below heap: stores cleanly, marks nothing.
                slots.fill(0);
                chunks.fill(0);
                let below = heap.add(0x1000);
                f(below, src.as_ptr());
                assert_eq!(std::slice::from_raw_parts(below, 128), &src);
                assert!(slots.iter().chain(&chunks).all(|&b| b == 0));
            }

            // 2) Sites 7 & 9 (+0x18849c / +0x2474afc): `mov [rbx], rdx ; mov [rcx+rax+0x48], rbx`.
            for &idx in &[7usize, 9usize] {
                slots.fill(0);
                chunks.fill(0);
                let site = &SITES[idx];
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                let mut code: Vec<u8> = Vec::new();
                // fn(dst_rbx: rcx, node_buf: rdx)
                code.extend([0x53, 0x57]); // push rbx ; push rdi
                code.extend([0x48, 0x89, 0xcb]); // mov rbx, rcx
                code.extend([0x48, 0x89, 0xd7]); // mov rdi, rdx
                code.extend([0x48, 0x89, 0xd1]); // mov rcx, rdx
                code.extend([0x31, 0xc0]); // xor eax, eax
                let win_at = code.len();
                code.extend(site.bytes);
                code.extend([0x5f, 0x5b, 0xc3]); // pop rdi ; pop rbx ; ret
                let entry = place + 0x200 - win_at;
                std::ptr::copy_nonoverlapping(code.as_ptr(), entry as *mut u8, code.len());
                let mine = [Site { window: 0x200, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                let f: unsafe extern "system" fn(*mut u8, *mut u64) = std::mem::transmute(entry);
                let mut node = [0u64; 16];
                node[9] = 0x1122_3344_5566_7788; // offset 0x48
                let dst = heap.add(0x2000 + 200 * 32);
                f(dst, node.as_mut_ptr());
                assert_eq!((dst as *const u64).read_unaligned(), 0x1122_3344_5566_7788);
                assert_eq!(node[9], dst as u64);
                assert_eq!(slots[200], 1);
                assert_eq!(chunks[200 / 128], 1);
            }

            // 3) Site 8 (+0x4e9b4e2): `mov qword ptr [r14], 0`.
            {
                slots.fill(0);
                chunks.fill(0);
                let site = &SITES[8];
                let place = VirtualAlloc(std::ptr::null(), 0x10000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as usize;
                let mut code: Vec<u8> = Vec::new();
                // fn(dst_base_plus_18: rcx, rbx_val: rdx)
                code.extend([0x53, 0x41, 0x56, 0x41, 0x57]); // push rbx ; push r14 ; push r15
                code.extend([0x48, 0x89, 0xd3]); // mov rbx, rdx
                code.extend([0x49, 0x89, 0xcf]); // mov r15, rcx
                code.extend([0x31, 0xc0]); // xor eax, eax
                let win_at = code.len();
                code.extend(site.bytes);
                code.extend([0x41, 0x5f, 0x41, 0x5e, 0x5b, 0xc3]); // pop r15 ; pop r14 ; pop rbx ; ret
                let entry = place + 0x200 - win_at;
                std::ptr::copy_nonoverlapping(code.as_ptr(), entry as *mut u8, code.len());
                let mine = [Site { window: 0x200, ..*site }];
                assert_eq!(apply(place, 0x10000, &raw const *marks as usize, &mine), [Hooked::Done]);
                let f: unsafe extern "system" fn(*mut u8, u64) = std::mem::transmute(entry);
                let dst = heap.add(0x2000 + 300 * 32);
                (dst as *mut u64).write_unaligned(0xdead_beef_cafe_babe);
                f(dst.add(0x18), 0x1234_5678);
                assert_eq!((dst as *const u64).read_unaligned(), 0);
                assert_eq!((dst.add(8) as *const u64).read_unaligned(), 0x1234_5678);
                assert_eq!(slots[300], 1);
                assert_eq!(chunks[300 / 128], 1);
            }
            drop(marks);
        }
    }
}
