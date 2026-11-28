//! Static TLS of an executable mapped as a DLL.
//!
//! An exe always gets TLS index 0, so title code reaches its thread-local
//! block as `mov r64, gs:[0x58]` (the thread's TLS array) followed by
//! `[r64]`, without reading `_tls_index`. Mapped as a DLL, the image gets
//! another index. Every such load is turned into a call to a thunk that
//! yields `&array[index]` instead, and the image's `_tls_index` (which CRT
//! code does read, and adds to the array pointer) is set to 0.

use std::ffi::c_void;

use anyhow::{Result, bail};
use victory_pe::ImageInfo;
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
    VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const TLS64_ADDRESS_OF_INDEX: usize = 16;

/// Redirects the image's TLS-array loads if the OS gave it an index other
/// than 0. When `vtls_rva` is `Some`, all `mov r64, gs:[0x58]` sites were
/// pre-patched on disk in `prepare()`, so only the single 4 KiB `.vtls` page
/// is updated (zero `.text` copy-on-write private pages!). Returns the number
/// of runtime `.text` loads patched (0 when `.vtls` was used).
///
/// # Safety
/// `base` is the image `info` describes, mapped and not yet running.
pub(crate) unsafe fn redirect(base: usize, info: &ImageInfo, vtls_rva: Option<u32>) -> Result<usize> {
    let Some(tls) = &info.tls else { return Ok(0) };
    // SAFETY: the TLS directory and the index it points to are in the image;
    // the OS relocated the pointer and wrote the index.
    let index_at = unsafe {
        ((base + tls.directory_rva as usize + TLS64_ADDRESS_OF_INDEX) as *const usize).read()
    } as *mut u32;
    // SAFETY: as above.
    let index = unsafe { index_at.read() };
    if let Some(vtls_rva) = vtls_rva {
        let vtls = (base + vtls_rva as usize) as *mut u8;
        let mut old = 0;
        // SAFETY: `.vtls` is a mapped 4 KiB section of the image.
        if unsafe { VirtualProtect(vtls as _, 4096, PAGE_READWRITE, &mut old) } == 0 {
            bail!(
                "VirtualProtect(.vtls rw) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: `.vtls` is writable now and holds at least `TLS_THUNK_BYTES`.
        let slice = unsafe { std::slice::from_raw_parts_mut(vtls, crate::prepare::TLS_THUNK_BYTES) };
        crate::prepare::write_tls_thunks(slice, index);
        // SAFETY: restoring execute protection and flushing the single `.vtls` page.
        if unsafe { VirtualProtect(vtls as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
            bail!(
                "VirtualProtect(.vtls rx) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: flushing our own updated thunk page.
        unsafe { FlushInstructionCache(GetCurrentProcess(), vtls as _, 4096) };
        if index != 0 {
            // SAFETY: see `index_at`; `.data` is writable.
            unsafe { index_at.write(0) };
        }
        tracing::info!(
            index,
            vtls_rva = format_args!("{vtls_rva:#x}"),
            "static TLS redirected via pre-patched .vtls"
        );
        return Ok(0);
    }
    if index == 0 {
        return Ok(0);
    }
    let thunks = thunks(near(base, info.size_of_image as usize)?, index)?;
    let mut patched = 0;
    for section in info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
    {
        let start = base + section.rva as usize;
        let len = section.virtual_size as usize;
        let mut old = 0;
        // SAFETY: the section is part of the mapped image.
        if unsafe { VirtualProtect(start as _, len, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            bail!(
                "VirtualProtect(code) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: writable now, and no title code runs yet.
        let code = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) };
        patched += crate::prepare::patch_tls_loads(code, start, thunks);
        // SAFETY: restoring the protection we changed; flushing our own code.
        unsafe {
            VirtualProtect(start as _, len, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), start as _, len);
        }
    }
    // SAFETY: see `index_at`; `.data` is writable.
    unsafe { index_at.write(0) };
    tracing::info!(index, patched, "static TLS redirected");
    Ok(patched)
}

/// Fills `block` with one thunk per register: `mov r, gs:[0x58];
/// lea r, [r + index * 8]; ret` (flags untouched, like the load it replaces).
fn thunks(block: *mut u8, index: u32) -> Result<usize> {
    // SAFETY: `block` is a writable 4 KiB page allocated by `near`.
    let slice = unsafe { std::slice::from_raw_parts_mut(block, crate::prepare::TLS_THUNK_BYTES) };
    crate::prepare::write_tls_thunks(slice, index);
    let mut old = 0;
    // SAFETY: `block` is our allocation of one page.
    if unsafe { VirtualProtect(block as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
        bail!(
            "VirtualProtect(TLS thunks) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(block as usize)
}

/// One writable page within `rel32` reach of `[base, base + size)`. Leaked:
/// title code calls into it for the life of the process.
fn near(base: usize, size: usize) -> Result<*mut u8> {
    const GRANULE: usize = 0x1_0000;
    // The search spans 1 GiB each side, so every call site stays in reach.
    if size >= 1 << 30 {
        bail!("image too large for TLS thunks within rel32 reach");
    }
    let above = (base + size).next_multiple_of(GRANULE);
    for k in 0..(1usize << 14) {
        for hint in [
            base.checked_sub((k + 1) * GRANULE),
            Some(above + k * GRANULE),
        ] {
            let Some(hint) = hint else { continue };
            // SAFETY: a fresh allocation at a free address, or null.
            let p = unsafe {
                VirtualAlloc(
                    hint as *const c_void,
                    4096,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };
            if !p.is_null() {
                return Ok(p.cast());
            }
        }
    }
    bail!("no free page near the image for TLS thunks")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the thread's TLS array pointer the way title code does.
    fn tls_array() -> usize {
        let array: usize;
        // SAFETY: reads TEB.ThreadLocalStoragePointer.
        unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array) };
        array
    }

    #[test]
    fn loads_become_calls_that_yield_the_title_slot() {
        let page = near(tls_array as *const () as usize, 0).unwrap();
        let thunks = thunks(near(page as usize, 4096).unwrap(), 3).unwrap();
        // (before the load, REX, ModRM, after the load): each case returns
        // the loaded register in rax; r12 is saved, being callee-saved.
        let cases: [(&[u8], u8, u8, &[u8]); 3] = [
            (&[], 0x48, 0x04, &[0xC3]),                   // rax
            (&[], 0x4C, 0x04, &[0x4C, 0x89, 0xC0, 0xC3]), // r8
            (
                &[0x41, 0x54],
                0x4C,
                0x24,
                &[0x4C, 0x89, 0xE0, 0x41, 0x5C, 0xC3],
            ), // r12
        ];
        for (before, rex, modrm, after) in cases {
            let mut code = before.to_vec();
            code.extend([0x65, rex, 0x8B, modrm, 0x25, 0x58, 0, 0, 0]);
            code.extend(after);
            // SAFETY: the page is ours and writable.
            let bytes = unsafe { std::slice::from_raw_parts_mut(page, code.len()) };
            bytes.copy_from_slice(&code);
            assert_eq!(crate::prepare::patch_tls_loads(bytes, page as usize, thunks), 1);
            let mut old = 0;
            // SAFETY: our page, made executable to run the patched code.
            unsafe { VirtualProtect(page as _, 4096, PAGE_EXECUTE_READ, &mut old) };
            // SAFETY: the page holds the patched load and a return.
            let f: extern "system" fn() -> usize = unsafe { std::mem::transmute(page) };
            assert_eq!(f(), tls_array() + 3 * 8, "REX {rex:#x} ModRM {modrm:#x}");
            // SAFETY: our page again.
            unsafe { VirtualProtect(page as _, 4096, PAGE_READWRITE, &mut old) };
        }
    }
}
