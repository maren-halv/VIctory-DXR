//! Turns a title's image into something the Windows loader maps but does
//! not run. Pure byte manipulation, so it is testable on any host.

use victory_pe::layout::{self, dir, dll_flags, file_flags};
use victory_pe::{ImageInfo, PeError};

#[derive(Debug, thiserror::Error)]
pub enum PrepareError {
    #[error(transparent)]
    Pe(#[from] PeError),
    #[error("cannot patch image: {0}")]
    Patch(&'static str),
}

/// A patched copy of an image plus everything we need from the original.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Parsed from the *original* bytes: imports, TLS callbacks and entry
    /// point as the title expects them.
    pub info: ImageInfo,
    /// The patched bytes to hand to the Windows loader.
    pub bytes: Vec<u8>,
    /// BLAKE3 of the original bytes: identifies the exact build.
    pub original_hash: blake3::Hash,
    /// RVA of the appended `.vtls` thunk section when executable TLS loads
    /// were pre-patched on disk (avoids ~31 MB `.text` copy-on-write RAM).
    pub(crate) vtls_rva: Option<u32>,
}

impl PreparedImage {
    /// File name for the patched copy, unique per original build.
    pub fn file_name(&self) -> String {
        format!("victory-image-{}.dll", &self.original_hash.to_hex()[..16])
    }
}

/// Produces the patched copy. The changes are:
///
/// * `IMAGE_FILE_DLL` set, entry point cleared: the OS maps it like a DLL
///   and calls nothing;
/// * import directory points to its empty terminator and the bound-import
///   directory is cleared: we bind the original slots using `info.imports`;
/// * TLS callback pointer cleared: the OS still allocates static TLS for
///   the image, but callbacks only run when we call them (after binding);
///   its base relocation is made a no-op so the 0 survives a rebase;
/// * `FORCE_INTEGRITY` cleared, checksum zeroed: the copy is unsigned.
pub fn prepare(original: &[u8]) -> Result<PreparedImage, PrepareError> {
    let info = ImageInfo::parse(original)?;
    let mut bytes = original.to_vec();

    let fh = layout::file_header_offset(&bytes).ok_or(PrepareError::Patch("no file header"))?;
    let opt =
        layout::optional_header_offset(&bytes).ok_or(PrepareError::Patch("no optional header"))?;

    let characteristics = info.characteristics | file_flags::DLL;
    write_u16(
        &mut bytes,
        fh + layout::FILE_HEADER_CHARACTERISTICS,
        characteristics,
    )?;
    write_u32(&mut bytes, opt + layout::OPT_ADDRESS_OF_ENTRY_POINT, 0)?;
    write_u32(&mut bytes, opt + layout::OPT_CHECKSUM, 0)?;
    let dll_characteristics = info.dll_characteristics & !dll_flags::FORCE_INTEGRITY;
    write_u16(
        &mut bytes,
        opt + layout::OPT_DLL_CHARACTERISTICS,
        dll_characteristics,
    )?;

    // Windows 10 assigns static TLS to the prepared DLL when its empty
    // import table is present. The parser has already walked one descriptor
    // per module up to the null terminator, regardless of the recorded Size.
    if let Some(at) = layout::data_directory_offset(&bytes, dir::IMPORT) {
        let rva = layout::read_u32(&bytes, at).ok_or(PrepareError::Patch("no import RVA"))?;
        if rva != 0 {
            let terminator = u32::try_from(info.imports.len())
                .ok()
                .and_then(|count| count.checked_mul(IMPORT_DESCRIPTOR_SIZE))
                .and_then(|offset| rva.checked_add(offset))
                .ok_or(PrepareError::Patch("import terminator RVA overflow"))?;
            write_u32(&mut bytes, at, terminator)?;
            write_u32(&mut bytes, at + 4, IMPORT_DESCRIPTOR_SIZE)?;
        }
    }
    if let Some(at) = layout::data_directory_offset(&bytes, dir::BOUND_IMPORT) {
        write(&mut bytes, at, &[0; layout::DATA_DIRECTORY_SIZE])?;
    }

    if let Some(tls) = &info.tls
        && tls.callbacks_va != 0
    {
        let at = info
            .rva_to_file_offset(tls.directory_rva)
            .ok_or(PrepareError::Patch("TLS directory is not file-backed"))?;
        write_u64(
            &mut bytes,
            at as usize + layout::TLS64_ADDRESS_OF_CALLBACKS,
            0,
        )?;
        // The field's base relocation would turn the 0 into `delta` whenever
        // the loader maps the copy away from its preferred base (Wine always
        // does for DYNAMIC_BASE images), and the loader then reads callbacks
        // from there. Make that entry a no-op.
        let field_rva = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
        for (entry, target) in dir64_relocations(&bytes, &info) {
            if target == field_rva {
                write_u16(&mut bytes, entry, 0)?; // IMAGE_REL_BASED_ABSOLUTE
            }
        }
    }

    let vtls_rva = append_vtls_section(&mut bytes, &info, fh, opt);

    Ok(PreparedImage {
        info,
        bytes,
        original_hash: blake3::hash(original),
        vtls_rva,
    })
}

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const SCN_CNT_CODE: u32 = 0x0000_0020;
const SCN_MEM_READ: u32 = 0x4000_0000;
const SECTION_HEADER_SIZE: usize = 40;
/// `mov r64, gs:[0x58]` is `65 REX 8B ModRM 25 58 00 00 00`.
const TLS_LOAD_LEN: usize = 9;
pub(crate) const TLS_THUNK_SIZE: usize = 32;
pub(crate) const TLS_THUNK_BYTES: usize = 16 * TLS_THUNK_SIZE;

/// Rewrites each `mov r64, gs:[0x58]` in `code` (at address/RVA `at`) as
/// `call thunks[r]; nop4`. `thunks` must be within `rel32` reach of `code`.
/// Because `call rel32` is RIP-relative, patching in RVA space inside
/// `prepare()` is 100% invariant to ASLR / rebasing at `LoadLibraryExW` time.
pub(crate) fn patch_tls_loads(code: &mut [u8], at: usize, thunks: usize) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i + TLS_LOAD_LEN <= code.len() {
        let c = &code[i..i + TLS_LOAD_LEN];
        let rex = c[1];
        let reg = usize::from((c[3] >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
        // shortcut: a byte match, not a decode. In the title all
        // 53,246 hits are instruction starts and none is missed (checked
        // against a full disassembly); decode .text if a title shows one.
        if c[0] == 0x65
            && (rex == 0x48 || rex == 0x4C)
            && c[2] == 0x8B
            && c[3] & 0xC7 == 0x04
            && c[4..] == [0x25, 0x58, 0, 0, 0]
            && reg != 4
        {
            let rel = (thunks + reg * TLS_THUNK_SIZE) as i64 - (at + i + 5) as i64;
            code[i] = 0xE8;
            code[i + 1..i + 5].copy_from_slice(&(rel as i32).to_le_bytes());
            code[i + 5..i + 9].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);
            count += 1;
            i += TLS_LOAD_LEN;
        } else {
            i += 1;
        }
    }
    count
}

/// Fills `buf` (at least [`TLS_THUNK_BYTES`] bytes) with one thunk per GPR:
/// `mov r, gs:[0x58]; lea r, [r + index * 8]; ret` (flags untouched).
pub(crate) fn write_tls_thunks(buf: &mut [u8], index: u32) {
    assert!(buf.len() >= TLS_THUNK_BYTES);
    buf[..TLS_THUNK_BYTES].fill(0xCC);
    let disp = (index * 8).to_le_bytes();
    for reg in 0..16u8 {
        let (r, low) = (reg >> 3, reg & 7);
        let mut code = [0u8; TLS_THUNK_SIZE];
        let mut n = 0;
        let base = [0x65, 0x48 | (r << 2), 0x8B, (low << 3) | 4, 0x25, 0x58, 0, 0, 0];
        code[..base.len()].copy_from_slice(&base);
        n += base.len();
        let lea = [0x48 | (r << 2) | r, 0x8D, 0x80 | (low << 3) | low];
        code[n..n + lea.len()].copy_from_slice(&lea);
        n += lea.len();
        if low == 4 {
            code[n] = 0x24; // SIB: [r12 + disp32]
            n += 1;
        }
        code[n..n + 4].copy_from_slice(&disp);
        n += 4;
        code[n] = 0xC3;
        n += 1;
        let dst = usize::from(reg) * TLS_THUNK_SIZE;
        buf[dst..dst + n].copy_from_slice(&code[..n]);
    }
}

fn read_u16(bytes: &[u8], at: usize) -> Option<u16> {
    let end = at.checked_add(2)?;
    Some(u16::from_le_bytes(bytes.get(at..end)?.try_into().ok()?))
}

fn align_up(value: u32, align: u32) -> Option<u32> {
    if align == 0 || !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    value.checked_add(mask).map(|v| v & !mask)
}

/// Appends a 1-page `.vtls` code section to a prepared EXE image with static
/// TLS and pre-patches every `mov r64, gs:[0x58]` site in executable sections
/// on disk. At load time, `tls::redirect` only writes 512 bytes into `.vtls`
/// instead of copy-on-writing ~31 MB of `.text` pages into private RAM.
fn append_vtls_section(
    bytes: &mut Vec<u8>,
    info: &ImageInfo,
    fh: usize,
    opt: usize,
) -> Option<u32> {
    if info.is_dll() || info.tls.is_none() {
        return None;
    }
    if std::env::var_os("VICTORY_VTLS").is_some_and(|v| v == "0") {
        return None;
    }
    if info.sections.iter().any(|s| s.name == ".vtls") {
        return None;
    }
    let num_sections = read_u16(bytes, fh + 2)?;
    let opt_size = read_u16(bytes, fh + 16)? as usize;
    let section_align = layout::read_u32(bytes, opt + 32)?;
    let file_align = layout::read_u32(bytes, opt + 36)?;
    let size_of_headers = layout::read_u32(bytes, opt + 60)? as usize;
    if section_align < 4096 || file_align < 512 {
        return None;
    }
    let new_header_at = opt
        .checked_add(opt_size)?
        .checked_add(usize::from(num_sections).checked_mul(SECTION_HEADER_SIZE)?)?;
    let new_header_end = new_header_at.checked_add(SECTION_HEADER_SIZE)?;
    if new_header_end > size_of_headers || new_header_end > bytes.len() {
        return None;
    }
    if !bytes[new_header_at..new_header_end].iter().all(|&b| b == 0) {
        return None;
    }
    let max_va_end = info
        .sections
        .iter()
        .map(|s| s.rva.saturating_add(s.virtual_size.max(s.file_size)))
        .max()
        .unwrap_or(info.size_of_image)
        .max(info.size_of_image);
    let vtls_rva = align_up(max_va_end, section_align)?;
    // Ensure every `call rel32` from executable code into `.vtls` stays well in range.
    if vtls_rva >= (1 << 30) {
        return None;
    }
    let vtls_virtual_size = 4096u32;
    let vtls_raw_size = align_up(TLS_THUNK_BYTES as u32, file_align)?;
    let vtls_raw_offset = align_up(u32::try_from(bytes.len()).ok()?, file_align)?;
    let new_size_of_image = align_up(vtls_rva.checked_add(vtls_virtual_size)?, section_align)?;
    let new_file_len = usize::try_from(vtls_raw_offset.checked_add(vtls_raw_size)?).ok()?;

    // Clear Authenticode certificate directory (index 4) if present because
    // the prepared DLL is unsigned and appending `.vtls` extends the file.
    if let Some(sec_dir) = layout::data_directory_offset(bytes, 4) {
        let _ = write(bytes, sec_dir, &[0; layout::DATA_DIRECTORY_SIZE]);
    }

    // Write the 40-byte IMAGE_SECTION_HEADER for `.vtls`.
    let mut sh = [0u8; SECTION_HEADER_SIZE];
    sh[0..8].copy_from_slice(b".vtls\0\0\0");
    sh[8..12].copy_from_slice(&vtls_virtual_size.to_le_bytes());
    sh[12..16].copy_from_slice(&vtls_rva.to_le_bytes());
    sh[16..20].copy_from_slice(&vtls_raw_size.to_le_bytes());
    sh[20..24].copy_from_slice(&vtls_raw_offset.to_le_bytes());
    sh[36..40].copy_from_slice(&(SCN_CNT_CODE | SCN_MEM_EXECUTE | SCN_MEM_READ).to_le_bytes());
    write(bytes, new_header_at, &sh).ok()?;
    write_u16(bytes, fh + 2, num_sections.checked_add(1)?).ok()?;
    write_u32(bytes, opt + 56, new_size_of_image).ok()?;

    // Append the `.vtls` raw bytes initialized with index-0 identity thunks.
    bytes.resize(new_file_len, 0);
    let raw_start = vtls_raw_offset as usize;
    write_tls_thunks(&mut bytes[raw_start..raw_start + TLS_THUNK_BYTES], 0);

    // Pre-patch all `mov r64, gs:[0x58]` sites across executable sections on disk.
    let mut patched = 0usize;
    for section in info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
    {
        let start = section.file_offset as usize;
        let len = section.file_size.min(section.virtual_size) as usize;
        if let Some(code) = bytes.get_mut(start..start.saturating_add(len)) {
            patched += patch_tls_loads(code, section.rva as usize, vtls_rva as usize);
        }
    }
    tracing::debug!(
        vtls_rva = format_args!("{vtls_rva:#x}"),
        patched,
        "pre-patched static TLS loads into .vtls section"
    );
    Some(vtls_rva)
}

/// Size of an IMAGE_IMPORT_DESCRIPTOR in the PE file.
const IMPORT_DESCRIPTOR_SIZE: u32 = 20;

/// Every `IMAGE_REL_BASED_DIR64` base relocation: (file offset of its entry,
/// RVA it patches). Empty if the image has no readable relocation table.
fn dir64_relocations(bytes: &[u8], info: &ImageInfo) -> Vec<(usize, u32)> {
    const DIR64: u16 = 10;
    let mut out = Vec::new();
    let Some(d) = layout::data_directory_offset(bytes, dir::BASERELOC) else {
        return out;
    };
    let (Some(rva), Some(size)) = (layout::read_u32(bytes, d), layout::read_u32(bytes, d + 4))
    else {
        return out;
    };
    let Some(start) = info.rva_to_file_offset(rva).filter(|_| size != 0) else {
        return out;
    };
    let (mut at, end) = (start as usize, start as usize + size as usize);
    while at + 8 <= end {
        let (Some(page), Some(block)) =
            (layout::read_u32(bytes, at), layout::read_u32(bytes, at + 4))
        else {
            break;
        };
        let block = block as usize;
        if block < 8 || at + block > end {
            break;
        }
        for entry in (at + 8..at + block).step_by(2) {
            let Some(e) = bytes.get(entry..entry + 2) else {
                break;
            };
            let e = u16::from_le_bytes([e[0], e[1]]);
            if e >> 12 == DIR64 {
                out.push((entry, page + u32::from(e & 0xFFF)));
            }
        }
        at += block;
    }
    out
}

fn write_u16(bytes: &mut [u8], at: usize, v: u16) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u32(bytes: &mut [u8], at: usize, v: u32) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write_u64(bytes: &mut [u8], at: usize, v: u64) -> Result<(), PrepareError> {
    write(bytes, at, &v.to_le_bytes())
}
fn write(bytes: &mut [u8], at: usize, v: &[u8]) -> Result<(), PrepareError> {
    bytes
        .get_mut(at..at + v.len())
        .ok_or(PrepareError::Patch("header field outside of file"))?
        .copy_from_slice(v);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_own_test_binary() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll(), "info describes the original");
        assert!(prepared.info.import_count() > 0);

        let patched = ImageInfo::parse(&prepared.bytes).unwrap();
        assert!(patched.is_dll());
        assert_eq!(patched.entry_rva, 0);
        assert!(patched.imports.is_empty());
        assert_eq!(patched.dll_characteristics & dll_flags::FORCE_INTEGRITY, 0);
        if let Some(tls) = &patched.tls {
            assert_eq!(tls.callbacks_va, 0);
            let field = tls.directory_rva + layout::TLS64_ADDRESS_OF_CALLBACKS as u32;
            assert!(
                dir64_relocations(&prepared.bytes, &patched)
                    .iter()
                    .all(|&(_, target)| target != field),
                "the cleared callback pointer must not be relocated"
            );
        }
        // Original sections stay at exact same RVAs/file offsets; EXEs with
        // static TLS append one `.vtls` thunk section at the end.
        if let Some(vtls_rva) = prepared.vtls_rva {
            assert_eq!(patched.sections.len(), prepared.info.sections.len() + 1);
            let vtls = patched.sections.last().unwrap();
            assert_eq!(vtls.name, ".vtls");
            assert_eq!(vtls.rva, vtls_rva);
            assert!(prepared.bytes.len() > original.len());
        } else {
            assert_eq!(patched.sections.len(), prepared.info.sections.len());
            assert_eq!(prepared.bytes.len(), original.len());
        }
        for (after, before) in patched.sections.iter().zip(&prepared.info.sections) {
            assert_eq!(after.name, before.name);
            assert_eq!(after.rva, before.rva);
            assert_eq!(after.file_offset, before.file_offset);
        }
    }

    #[test]
    fn keeps_iat_slots_when_import_size_excludes_the_terminator() {
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let info = ImageInfo::parse(&original).unwrap();
        let directory = layout::data_directory_offset(&original, dir::IMPORT).unwrap();
        // This linker layout counts the descriptors for real modules only.
        let descriptor_bytes = u32::try_from(info.imports.len()).unwrap() * IMPORT_DESCRIPTOR_SIZE;
        write_u32(&mut original, directory + 4, descriptor_bytes).unwrap();
        let prepared = prepare(&original).unwrap();
        let empty = layout::read_u32(&prepared.bytes, directory).unwrap();
        assert_eq!(
            empty,
            layout::read_u32(&original, directory).unwrap() + descriptor_bytes
        );
        assert_eq!(
            layout::read_u32(&prepared.bytes, directory + 4),
            Some(IMPORT_DESCRIPTOR_SIZE)
        );
        let at = info.rva_to_file_offset(empty).unwrap() as usize;
        assert_eq!(
            &prepared.bytes[at..at + IMPORT_DESCRIPTOR_SIZE as usize],
            &[0; 20]
        );
        let iat = layout::data_directory_offset(&original, dir::IAT).unwrap();
        assert_eq!(
            &prepared.bytes[iat..iat + layout::DATA_DIRECTORY_SIZE],
            &original[iat..iat + layout::DATA_DIRECTORY_SIZE]
        );
        assert_eq!(prepared.info.import_count(), info.import_count());
        for module in &prepared.info.imports {
            for entry in &module.entries {
                let at = info.rva_to_file_offset(entry.iat_slot_rva).unwrap() as usize;
                assert_eq!(&prepared.bytes[at..at + 8], &original[at..at + 8]);
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_assigns_static_tls_to_a_prepared_dll() {
        // Model a title DLL using our own linked image. Loading a DLL leaves
        // its assigned index intact; the executable-specific redirect resets it.
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let flags =
            layout::file_header_offset(&original).unwrap() + layout::FILE_HEADER_CHARACTERISTICS;
        let characteristics =
            ImageInfo::parse(&original).unwrap().characteristics | file_flags::DLL;
        write_u16(&mut original, flags, characteristics).unwrap();
        let prepared = prepare(&original).unwrap();
        let info = &prepared.info;
        let tls = info.tls.as_ref().expect("the test binary uses static TLS");
        let directory = info.rva_to_file_offset(tls.directory_rva).unwrap() as usize;
        let index_va = layout::read_u64(&original, directory + 16).unwrap();
        let index_rva = index_va.checked_sub(info.image_base).unwrap();
        assert!(index_rva + 4 <= u64::from(info.size_of_image));

        let folder = tempfile::tempdir().unwrap();
        let image = crate::load(&prepared, folder.path()).unwrap();
        // SAFETY: the mapped test image contains the checked DWORD above.
        let assigned =
            unsafe { ((image.base() + index_rva as usize) as *const u32).read_unaligned() };
        // SAFETY: the prepared DLL has never been attached or given to a caller.
        unsafe { image.unload() };
        assert_ne!(
            assigned, 0,
            "the process executable already owns TLS slot zero"
        );
    }

    #[test]
    fn file_name_is_stable_per_build() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let a = prepare(&original).unwrap();
        let b = prepare(&original).unwrap();
        assert_eq!(a.file_name(), b.file_name());
    }

    #[cfg(windows)]
    #[test]
    fn vtls_prepared_exe_loads_and_redirects_via_vtls_section() {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let prepared = prepare(&original).unwrap();
        let vtls_rva = prepared.vtls_rva.expect("test exe has static TLS and header room for .vtls");
        let tls = prepared.info.tls.as_ref().unwrap();
        let directory = prepared.info.rva_to_file_offset(tls.directory_rva).unwrap() as usize;
        let index_va = layout::read_u64(&original, directory + 16).unwrap();
        let index_rva = index_va.checked_sub(prepared.info.image_base).unwrap();

        let folder = tempfile::tempdir().unwrap();
        let image = crate::load(&prepared, folder.path()).unwrap();
        // SAFETY: `tls::redirect` reset `_tls_index` to 0 after updating `.vtls` thunks.
        let index_after = unsafe { ((image.base() + index_rva as usize) as *const u32).read_unaligned() };
        // Verify `.vtls` thunk 0 (rax) starts with `mov rax, gs:[0x58]` (`65 48 8B 04 25 58 00 00 00`).
        // SAFETY: `.vtls` is mapped at `image.base() + vtls_rva`.
        let thunk0 = unsafe { std::slice::from_raw_parts((image.base() + vtls_rva as usize) as *const u8, 9) };
        assert_eq!(thunk0, &[0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0]);
        // SAFETY: never attached or called.
        unsafe { image.unload() };
        assert_eq!(index_after, 0, "_tls_index must be reset to 0 after .vtls redirect");
    }
}
