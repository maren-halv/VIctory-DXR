//! Diagnostics overlay: performance HUD, live rolling frametime graph, and
//! interactive F4 debug menu (`VICTORY_PANEL=1` for Compact, `2`/`full` for
//! Full; off by default; `F3` cycles Off -> Compact -> Full, `F4` toggles the
//! debug menu while our window has focus).
//!
//! Zero-cost when hidden (only polls `F3`/`F4` key state and returns). While
//! active, renders to a single reusable max-sized buffer per `(format, scale)`
//! and copies only the active sub-rectangle onto the swap chain's back buffer
//! behind the title's picture (invisible to screenshots and frame recording).

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_DOWN, VK_F3, VK_F4, VK_LEFT, VK_RETURN, VK_RIGHT, VK_UP,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{Host, HostList, transition};

/// Averaging window for numeric counters (keeps text readable without jitter).
const STATS_EVERY: Duration = Duration::from_millis(500);
/// Refresh cadence for the rolling frametime graph (~12.5 Hz).
const GRAPH_EVERY: Duration = Duration::from_millis(80);

/// 5x7 bitmap font geometry and cell stride in dots.
const GLYPH: (u32, u32) = (5, 7);
const CELL: (u32, u32) = (6, 9);

/// Panel inner text width (40 chars * 6 dots = 240 dots = 120 graph samples * 2).
const COLUMNS: u32 = 40;
const PAD_X: u32 = 6;
const PANEL_W: u32 = COLUMNS * CELL.0 + 2 * PAD_X; // 252 dots
const GRAPH_SAMPLES: usize = 120;
const GRAPH_INNER_W: u32 = (GRAPH_SAMPLES as u32) * 2; // 240 dots
const GRAPH_INNER_H: u32 = 34;

/// Maximum possible height in dots (Full HUD + Graph + F4 Menu + borders).
const MAX_PANEL_H: u32 = 288;

/// Upload ring depth: plenty of headroom even during rapid menu navigation.
const UPLOADS: usize = 6;
/// Frames to hold a retired `Buffers` before dropping (past swapchain latency).
const RETIRE_FRAMES: u32 = 16;

const MENU_ITEMS: usize = 7;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum HudMode {
    Off,
    Compact,
    Full,
}

impl HudMode {
    fn from_env() -> Self {
        let Some(raw) = std::env::var_os("VICTORY_PANEL") else {
            return Self::Off;
        };
        let s = raw.to_string_lossy();
        match s.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "false" => Self::Off,
            "2" | "full" => Self::Full,
            _ => Self::Compact,
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Off => Self::Compact,
            Self::Compact => Self::Full,
            Self::Full => Self::Off,
        }
    }

    const fn prev(self) -> Self {
        match self {
            Self::Off => Self::Full,
            Self::Compact => Self::Off,
            Self::Full => Self::Compact,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Off => "OFF",
            Self::Compact => "COMPACT",
            Self::Full => "FULL",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GraphScale {
    Ms33,
    Ms50,
    Ms100,
}

impl GraphScale {
    const fn max_ms(self) -> f32 {
        match self {
            Self::Ms33 => 33.3,
            Self::Ms50 => 50.0,
            Self::Ms100 => 100.0,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Ms33 => "33 MS",
            Self::Ms50 => "50 MS",
            Self::Ms100 => "100 MS",
        }
    }

    const fn tag(self) -> &'static str {
        match self {
            Self::Ms33 => "33MS",
            Self::Ms50 => "50MS",
            Self::Ms100 => "100MS",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Ms33 => Self::Ms50,
            Self::Ms50 => Self::Ms100,
            Self::Ms100 => Self::Ms33,
        }
    }

    const fn prev(self) -> Self {
        match self {
            Self::Ms33 => Self::Ms100,
            Self::Ms50 => Self::Ms33,
            Self::Ms100 => Self::Ms50,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Corner {
    TopLeft,
    TopRight,
    BotLeft,
    BotRight,
}

impl Corner {
    const fn label(self) -> &'static str {
        match self {
            Self::TopLeft => "TOP-LEFT",
            Self::TopRight => "TOP-RIGHT",
            Self::BotLeft => "BOT-LEFT",
            Self::BotRight => "BOT-RIGHT",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::TopLeft => Self::TopRight,
            Self::TopRight => Self::BotLeft,
            Self::BotLeft => Self::BotRight,
            Self::BotRight => Self::TopLeft,
        }
    }

    const fn prev(self) -> Self {
        match self {
            Self::TopLeft => Self::BotRight,
            Self::TopRight => Self::TopLeft,
            Self::BotLeft => Self::TopRight,
            Self::BotRight => Self::BotLeft,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum UiScale {
    Auto,
    X1,
    X2,
    X3,
}

impl UiScale {
    fn resolve(self, height: u32) -> u32 {
        match self {
            Self::Auto => (height / 540).clamp(1, 3),
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X3 => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Auto => "AUTO",
            Self::X1 => "1X",
            Self::X2 => "2X",
            Self::X3 => "3X",
        }
    }

    const fn next(self) -> Self {
        match self {
            Self::Auto => Self::X1,
            Self::X1 => Self::X2,
            Self::X2 => Self::X3,
            Self::X3 => Self::Auto,
        }
    }

    const fn prev(self) -> Self {
        match self {
            Self::Auto => Self::X3,
            Self::X1 => Self::Auto,
            Self::X2 => Self::X1,
            Self::X3 => Self::X2,
        }
    }
}

/// Edge-triggered key state (only polls navigation keys when menu is open).
#[derive(Default)]
struct Keys {
    down: [bool; 7],
}

#[derive(Default)]
struct KeyEvents {
    f3: bool,
    f4: bool,
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    enter: bool,
}

impl Keys {
    fn poll(&mut self, menu_open: bool) -> KeyEvents {
        const VKS: [u16; 7] = [VK_F3, VK_F4, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_RETURN];
        let count = if menu_open { VKS.len() } else { 2 };
        let mut rising = [false; 7];
        let mut any = false;
        for i in 0..count {
            // SAFETY: plain key-state query.
            let is_down = unsafe { GetAsyncKeyState(i32::from(VKS[i])) } < 0;
            let was_down = self.down[i];
            self.down[i] = is_down;
            if is_down && !was_down {
                rising[i] = true;
                any = true;
            }
        }
        if !menu_open {
            self.down[2..].fill(false);
        }
        if !any || !foreground_is_ours() {
            return KeyEvents::default();
        }
        KeyEvents {
            f3: rising[0],
            f4: rising[1],
            up: rising[2],
            down: rising[3],
            left: rising[4],
            right: rising[5],
            enter: rising[6],
        }
    }
}

fn foreground_is_ours() -> bool {
    let mut process = 0;
    // SAFETY: plain window query; `process` has room for the PID.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// Seven rows of five dots, the leftmost the highest bit.
#[rustfmt::skip]
const fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        b'1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        b'3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        b'4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        b'5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        b'6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        b'7' => [0b11111, 0b00001, 0b00010, 0b00010, 0b00100, 0b00100, 0b00100],
        b'8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        b'9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        b'A' => [0b00100, 0b01010, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001],
        b'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        b'C' => [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
        b'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        b'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        b'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        b'G' => [0b01111, 0b10000, 0b10000, 0b10011, 0b10001, 0b10001, 0b01111],
        b'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        b'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        b'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        b'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        b'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        b'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        b'N' => [0b10001, 0b11001, 0b11001, 0b10101, 0b10011, 0b10011, 0b10001],
        b'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        b'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        b'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        b'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        b'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        b'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        b'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        b'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        b'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        b'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        b'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        b'.' => [0, 0, 0, 0, 0, 0b01100, 0b01100],
        b'/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        b'-' => [0, 0, 0, 0b11111, 0, 0, 0],
        b':' => [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
        b'%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        b'+' => [0, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0],
        b'<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        b'>' => [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
        b'[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        b']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        b'(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        b')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        b',' => [0, 0, 0, 0, 0b00110, 0b00100, 0b01000],
        b'_' => [0, 0, 0, 0, 0, 0, 0b11111],
        b'=' => [0, 0, 0b11111, 0, 0b11111, 0, 0],
        b' ' => [0; 7],
        _ => [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111],
    }
}

/// Theme colors mapped onto the swap chain's pixel format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Color {
    Bg = 0,
    Border,
    Header,
    PlotBg,
    Highlight,
    Cyan,
    Green,
    Mint,
    Amber,
    Red,
    Muted,
    White,
    RefGreen,
    RefAmber,
}

const COLOR_COUNT: usize = 14;

const THEME_RGBA: [[u8; 4]; COLOR_COUNT] = [
    [0x0E, 0x11, 0x17, 0xFF], // Bg (#0E1117)
    [0x2D, 0x37, 0x48, 0xFF], // Border (#2D3748)
    [0x16, 0x1B, 0x22, 0xFF], // Header (#161B22)
    [0x0A, 0x0D, 0x12, 0xFF], // PlotBg (#0A0D12)
    [0x1E, 0x29, 0x3B, 0xFF], // Highlight / bar track (#1E293B)
    [0x38, 0xBD, 0xF8, 0xFF], // Cyan (#38BDF8)
    [0x4A, 0xDE, 0x80, 0xFF], // Green (#4ADE80)
    [0x2D, 0xD4, 0xBF, 0xFF], // Mint (#2DD4BF)
    [0xFB, 0xBF, 0x24, 0xFF], // Amber (#FBBF24)
    [0xF8, 0x71, 0x71, 0xFF], // Red (#F87171)
    [0x94, 0xA3, 0xB8, 0xFF], // Muted (#94A3B8)
    [0xF8, 0xFA, 0xFC, 0xFF], // White (#F8FAFC)
    [0x1F, 0x4D, 0x36, 0xFF], // RefGreen (subtle 16.7ms line)
    [0x52, 0x3B, 0x18, 0xFF], // RefAmber (subtle 33.3ms line)
];

/// IEEE-754 binary32 -> binary16 (round-to-nearest-even).
const fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if frac != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    if unbiased < -14 {
        if unbiased < -24 {
            return sign;
        }
        let mant = frac | 0x0080_0000;
        let shift = (-14 - unbiased) as u32 + 13;
        let half_bit = 1u32 << (shift - 1);
        let mut sub = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        if rem > half_bit || (rem == half_bit && (sub & 1) != 0) {
            sub += 1;
        }
        return sign | (sub as u16);
    }
    let h_exp = ((unbiased + 15) as u32) << 10;
    let mut h_frac = frac >> 13;
    let rem = frac & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && (h_frac & 1) != 0) {
        h_frac += 1;
        if h_frac == 0x0400 {
            return sign | ((h_exp + 0x0400) as u16);
        }
    }
    sign | (h_exp as u16) | (h_frac as u16)
}

fn srgb_to_linear(c: u8) -> f32 {
    let s = f32::from(c) / 255.0;
    if s <= 0.04045 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
}

/// Encodes one RGBA8 color into the swap chain's pixel format (`bpp` bytes valid).
fn encode_pixel(format: DXGI_FORMAT, [r, g, b, a]: [u8; 4]) -> Option<([u8; 8], usize)> {
    let mut px = [0u8; 8];
    let bpp = match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => {
            px[..4].copy_from_slice(&[r, g, b, a]);
            4
        }
        DXGI_FORMAT_B8G8R8A8_UNORM => {
            px[..4].copy_from_slice(&[b, g, r, a]);
            4
        }
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let r10 = (u32::from(r) * 1023 + 127) / 255;
            let g10 = (u32::from(g) * 1023 + 127) / 255;
            let b10 = (u32::from(b) * 1023 + 127) / 255;
            let a2 = (u32::from(a) * 3 + 127) / 255;
            let packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
            px[..4].copy_from_slice(&packed.to_le_bytes());
            4
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            let channels = [
                f32_to_f16(srgb_to_linear(r)),
                f32_to_f16(srgb_to_linear(g)),
                f32_to_f16(srgb_to_linear(b)),
                f32_to_f16(f32::from(a) / 255.0),
            ];
            for (i, h) in channels.into_iter().enumerate() {
                px[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
            }
            8
        }
        _ => return None,
    };
    Some((px, bpp))
}

struct Palette {
    entries: [[u8; 8]; COLOR_COUNT],
    bpp: usize,
}

impl Palette {
    fn for_format(format: DXGI_FORMAT) -> Option<Self> {
        let mut entries = [[0u8; 8]; COLOR_COUNT];
        let mut bpp = 0;
        for (slot, rgba) in entries.iter_mut().zip(THEME_RGBA) {
            let (px, n) = encode_pixel(format, rgba)?;
            *slot = px;
            bpp = n;
        }
        Some(Self { entries, bpp })
    }

    #[inline]
    fn bytes(&self, c: Color) -> &[u8] {
        &self.entries[c as usize][..self.bpp]
    }
}

/// Dot-resolution panel surface (1 byte per dot, scaled on upload).
struct Canvas {
    w: u32,
    h: u32,
    dots: Vec<Color>,
}

impl Canvas {
    fn new(w: u32, max_h: u32) -> Self {
        Self { w, h: 0, dots: vec![Color::Bg; (w * max_h) as usize] }
    }

    fn clear(&mut self, h: u32) {
        let h = h.min(MAX_PANEL_H);
        self.h = h;
        let len = (self.w * h) as usize;
        self.dots[..len].fill(Color::Bg);
    }

    #[inline]
    fn set(&mut self, x: u32, y: u32, c: Color) {
        if x < self.w && y < self.h {
            self.dots[(y * self.w + x) as usize] = c;
        }
    }

    fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        let x1 = (x + w).min(self.w);
        let y1 = (y + h).min(self.h);
        if x >= x1 || y >= y1 {
            return;
        }
        for row in y..y1 {
            let start = (row * self.w + x) as usize;
            let end = (row * self.w + x1) as usize;
            self.dots[start..end].fill(c);
        }
    }

    fn stroke_rect(&mut self, x: u32, y: u32, w: u32, h: u32, c: Color) {
        if w == 0 || h == 0 {
            return;
        }
        self.fill_rect(x, y, w, 1, c);
        self.fill_rect(x, y + h - 1, w, 1, c);
        self.fill_rect(x, y, 1, h, c);
        self.fill_rect(x + w - 1, y, 1, h, c);
    }

    fn dashed_hline(&mut self, x: u32, y: u32, w: u32, c: Color) {
        for dx in 0..w {
            if (dx / 3) % 2 == 0 {
                self.set(x + dx, y, c);
            }
        }
    }

    fn text(&mut self, mut x: u32, y: u32, s: &str, c: Color) -> u32 {
        for &b in s.as_bytes() {
            let g = glyph(b.to_ascii_uppercase());
            for (row, &bits) in g.iter().enumerate() {
                if bits == 0 {
                    continue;
                }
                for col in 0..GLYPH.0 {
                    if (bits >> (GLYPH.0 - 1 - col)) & 1 != 0 {
                        self.set(x + col, y + row as u32, c);
                    }
                }
            }
            x += CELL.0;
        }
        x
    }

    fn text_right(&mut self, right_x: u32, y: u32, s: &str, c: Color) {
        let width = (s.len() as u32) * CELL.0;
        let x = right_x.saturating_sub(width);
        self.text(x, y, s, c);
    }

    fn progress_bar(&mut self, x: u32, y: u32, w: u32, h: u32, ratio: f64, fill_color: Color) {
        self.fill_rect(x, y, w, h, Color::Highlight);
        let fill_w = ((ratio.clamp(0.0, 1.0)) * f64::from(w)).round() as u32;
        if fill_w > 0 {
            self.fill_rect(x, y, fill_w.min(w), h, fill_color);
        }
    }

    /// Blits the active `(self.w, self.h)` dot grid scaled by `scale` into `out`.
    fn blit(&self, scale: u32, palette: &Palette, pitch: usize, out: &mut [u8]) {
        let bpp = palette.bpp;
        let s = scale as usize;
        for y in 0..self.h {
            let top = pitch * (y * scale) as usize;
            let row = &self.dots[(y * self.w) as usize..((y + 1) * self.w) as usize];
            for (x, &color) in row.iter().enumerate() {
                let px = palette.bytes(color);
                let base = top + x * s * bpp;
                for i in 0..s {
                    let at = base + i * bpp;
                    out[at..at + bpp].copy_from_slice(px);
                }
            }
            for i in 1..s {
                out.copy_within(top..top + pitch, top + i * pitch);
            }
        }
    }
}

/// Rolling ring buffer of the last 120 per-frame durations (in milliseconds).
struct FrameRing {
    samples: [f32; GRAPH_SAMPLES],
    head: usize,
    len: usize,
}

impl FrameRing {
    fn new() -> Self {
        Self { samples: [0.0; GRAPH_SAMPLES], head: 0, len: 0 }
    }

    fn push(&mut self, ms: f32) {
        self.samples[self.head] = ms;
        self.head = (self.head + 1) % GRAPH_SAMPLES;
        if self.len < GRAPH_SAMPLES {
            self.len += 1;
        }
    }

    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    fn sample(&self, i: usize) -> f32 {
        let start = if self.len == GRAPH_SAMPLES { self.head } else { 0 };
        self.samples[(start + i) % GRAPH_SAMPLES]
    }

    /// 1% low FPS: 1000 / mean of the worst ~1% frames in the 120-frame window.
    fn one_percent_low_fps(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let mut sorted = [0.0f32; GRAPH_SAMPLES];
        for i in 0..self.len {
            sorted[i] = self.sample(i);
        }
        let slice = &mut sorted[..self.len];
        slice.sort_unstable_by(f32::total_cmp);
        let k = self.len.div_ceil(100);
        let sum: f64 = slice[self.len - k..].iter().map(|&v| f64::from(v)).sum();
        let avg_ms = sum / k as f64;
        1e3 / avg_ms.max(0.001)
    }
}

/// The counters read for it. They grow, and go back to 0 with the frame-rate
/// line (right behind a look of ours).
#[derive(Clone, Copy, Default)]
struct Counters([u64; COUNTERS]);

const COUNTERS: usize = 11;

impl Counters {
    fn now() -> Counters {
        use std::sync::atomic::Ordering;
        use super::{CARD_CLOCK_US, COPY_BACK, PREPARING};
        let get = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let record_ns = crate::copystat::peek(crate::copystat::C::RecordNs);
        #[cfg(not(feature = "diagnostics"))]
        let record_ns = 0;
        Counters([
            super::worked(),
            get(&CARD_CLOCK_US[0]),
            get(&CARD_CLOCK_US[1]),
            get(&CARD_CLOCK_US[2]),
            get(&PREPARING[3]),
            get(&COPY_BACK[0]),
            get(&COPY_BACK[1]),
            get(&crate::shadow::CARD_STATS[0]),
            crate::draw::dropped_now(),
            record_ns,
            crate::draw::drawn(),
        ])
    }

    /// What came since `before`, per frame.
    fn since(self, before: Counters, frames: f64) -> [f64; COUNTERS] {
        let mut out = [0.0; COUNTERS];
        for (i, out) in out.iter_mut().enumerate() {
            let (was, is) = (before.0[i], self.0[i]);
            *out = if is >= was { is - was } else { is } as f64 / frames;
        }
        out
    }
}

#[derive(Clone, Copy, Default)]
struct Stats {
    valid: bool,
    fps: f64,
    frame_ms: f64,
    gpu_busy_ms: f64,
    idle_behind_ms: f64,
    submit_busy_ms: f64,
    gpu_wait_ms: f64,
    recording_ms: f64,
    copy_back_ms: f64,
    draws: f64,
    dropped: f64,
    mb_up: f64,
    mb_tex: f64,
    mb_back: f64,
    cost_us: f64,
    ram_private_gb: f64,
    ram_working_gb: f64,
    ram_total_gb: f64,
    vram_used_gb: f64,
    vram_budget_gb: f64,
    vram_shared_gb: f64,
}

fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.WorkingSetSize as u64, c.PagefileUsage as u64)
}

fn total_ram_bytes() -> u64 {
    static TOTAL: OnceLock<u64> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        // SAFETY: zero-initializing a plain C struct before setting `dwLength`.
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: `dwLength` matches the struct size.
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 { status.ullTotalPhys } else { 0 }
    })
}

fn active_rt() -> (&'static str, Color) {
    if crate::dxr::enabled() {
        ("DXR", Color::Green)
    } else if crate::pipeline::builder_runs() {
        ("BVH", Color::Cyan)
    } else {
        ("OFF", Color::Muted)
    }
}

fn active_fsr() -> (&'static str, Color) {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    let forced = *ENV.get_or_init(|| {
        if let Ok(v) = std::env::var("VICTORY_UPSCALER") {
            match v.trim() {
                "7" => return Some(true),
                "6" => return Some(false),
                _ => {}
            }
        }
        if std::env::var_os("VICTORY_FSR").is_some() {
            return Some(crate::switch::on("VICTORY_FSR"));
        }
        if std::env::var_os("VICTORY_FSR_1440P").is_some() {
            return Some(crate::switch::on("VICTORY_FSR_1440P"));
        }
        if std::env::var_os("VICTORY_FSR_4K").is_some() {
            return Some(crate::switch::on("VICTORY_FSR_4K"));
        }
        None
    });
    let on = forced.unwrap_or_else(|| crate::console::current() == crate::console::Console::SeriesS);
    if on { ("ON", Color::Green) } else { ("OFF", Color::Muted) }
}

fn active_ambient() -> (String, Color) {
    if !crate::ambient::on() {
        return ("OFF".to_owned(), Color::Muted);
    }
    match crate::ambient::current_level() {
        Some(v) => (format!("{v:.1}"), Color::Cyan),
        None => ("OFF".to_owned(), Color::Muted),
    }
}

fn fps_color(fps: f64) -> Color {
    if fps >= 58.0 {
        Color::Green
    } else if fps >= 40.0 {
        Color::Mint
    } else if fps >= 30.0 {
        Color::Amber
    } else {
        Color::Red
    }
}

fn frametime_color(ms: f32) -> Color {
    if ms <= 16.7 {
        Color::Green
    } else if ms <= 25.0 {
        Color::Mint
    } else if ms <= 33.3 {
        Color::Amber
    } else {
        Color::Red
    }
}

fn mem_bar_color(ratio: f64) -> Color {
    if ratio >= 0.90 {
        Color::Red
    } else if ratio >= 0.75 {
        Color::Amber
    } else {
        Color::Green
    }
}

/// Computes the exact active height in dots for the current view state.
fn layout_height(hud: HudMode, graph_on: bool, menu_open: bool) -> u32 {
    if hud == HudMode::Off && !menu_open {
        return 0;
    }
    let mut h = 1 + 13 + 1; // top border + header strip + divider
    if hud != HudMode::Off {
        h += 4 + 9; // top pad + FPS/1%L/ms/max row
        if graph_on {
            h += 2 + 9 + (GRAPH_INNER_H + 2); // gap + caption + framed plot
        }
        h += 4 + 16 + 16 + 9 + 3; // gap + RAM + VRAM + mode summary + pad
        if hud == HudMode::Full {
            h += 1 + 3 + 6 * 9 + 2; // divider + pad + 6 metric rows + pad
        }
    }
    if menu_open {
        let sep = u32::from(hud != HudMode::Off);
        h += sep + 3 + 10 + (MENU_ITEMS as u32) * 11 + 3; // optional divider + header + 7 rows + pad
    }
    h + 1 // bottom border
}

struct RenderState<'a> {
    hud: HudMode,
    graph_on: bool,
    graph_scale: GraphScale,
    corner: Corner,
    ui_scale: UiScale,
    resolved_scale: u32,
    menu_open: bool,
    menu_sel: usize,
    stats: &'a Stats,
    worst_ms: f64,
    ring: &'a FrameRing,
    ambient_level: Option<f32>,
    rt: (&'static str, Color),
    fsr: (&'static str, Color),
    ambient_badge: (String, Color),
}

fn render_panel(canvas: &mut Canvas, st: &RenderState<'_>) {
    let h = layout_height(st.hud, st.graph_on, st.menu_open);
    canvas.clear(h);
    if h == 0 {
        return;
    }
    let right_x = PANEL_W - PAD_X;

    // Header strip.
    canvas.fill_rect(1, 1, PANEL_W - 2, 13, Color::Header);
    let title = if st.hud == HudMode::Off { "VICTORY DEBUG" } else { "VICTORY HUD" };
    canvas.text(PAD_X, 4, title, Color::Cyan);
    canvas.text_right(right_x, 4, "[F3] HUD [F4] MENU", Color::Muted);
    canvas.fill_rect(1, 14, PANEL_W - 2, 1, Color::Border);

    let mut y = 15;

    if st.hud != HudMode::Off {
        y += 4;
        // Row 1: FPS, 1% Low, Frame ms, Longest ms.
        let low1 = st.ring.one_percent_low_fps();
        let mut x = PAD_X;
        x = canvas.text(x, y, "FPS ", Color::Muted);
        if st.stats.valid {
            x = canvas.text(x, y, &format!("{:5.1}", st.stats.fps), fps_color(st.stats.fps));
        } else {
            x = canvas.text(x, y, "  -- ", Color::Muted);
        }
        x = canvas.text(x, y, " 1%L ", Color::Muted);
        if st.ring.len > 0 {
            x = canvas.text(x, y, &format!("{:4.0}", low1), fps_color(low1));
        } else {
            x = canvas.text(x, y, " -- ", Color::Muted);
        }
        x = canvas.text(x, y, " ", Color::Muted);
        if st.stats.valid {
            x = canvas.text(x, y, &format!("{:4.1}MS", st.stats.frame_ms), Color::White);
        } else {
            x = canvas.text(x, y, " -- MS", Color::Muted);
        }
        let _ = x;
        let worst_col = if st.worst_ms > 50.0 {
            Color::Red
        } else if st.worst_ms > 33.3 {
            Color::Amber
        } else {
            Color::White
        };
        let worst_str = format!("MAX {:3.0}", st.worst_ms);
        canvas.text_right(right_x, y, &worst_str, worst_col);
        y += 9;

        // Rolling frametime graph.
        if st.graph_on {
            y += 2;
            canvas.text(PAD_X, y, "FRAMETIME", Color::Muted);
            let legend = format!("60/30FPS  {}", st.graph_scale.tag());
            canvas.text_right(right_x, y, &legend, Color::Muted);
            y += 9;

            let gx = PAD_X - 1;
            let gw = GRAPH_INNER_W + 2;
            let gh = GRAPH_INNER_H + 2;
            canvas.fill_rect(gx + 1, y + 1, GRAPH_INNER_W, GRAPH_INNER_H, Color::PlotBg);
            canvas.stroke_rect(gx, y, gw, gh, Color::Border);

            let max_ms = st.graph_scale.max_ms();
            let plot_x = gx + 1;
            let plot_y = y + 1;
            let plot_bottom = plot_y + GRAPH_INNER_H - 1;

            // Dashed reference lines at 16.7 ms (60 FPS) and 33.3 ms (30 FPS).
            for (ref_ms, ref_col) in [(16.7f32, Color::RefGreen), (33.3f32, Color::RefAmber)] {
                if ref_ms <= max_ms + 0.05 {
                    let frac = (ref_ms / max_ms).clamp(0.0, 1.0);
                    let ry = plot_bottom - ((frac * (GRAPH_INNER_H - 1) as f32).round() as u32);
                    canvas.dashed_hline(plot_x, ry, GRAPH_INNER_W, ref_col);
                }
            }

            // Per-frame bars (oldest on left, newest flush right).
            let count = st.ring.len;
            let offset = GRAPH_SAMPLES - count;
            for i in 0..count {
                let ms = st.ring.sample(i);
                let frac = (ms / max_ms).clamp(0.0, 1.0);
                let bar_h = ((frac * GRAPH_INNER_H as f32).round() as u32).clamp(1, GRAPH_INNER_H);
                let bx = plot_x + ((offset + i) as u32) * 2;
                let by = plot_y + GRAPH_INNER_H - bar_h;
                canvas.fill_rect(bx, by, 2, bar_h, frametime_color(ms));
            }

            y += gh;
        }

        // RAM + bar.
        y += 4;
        let ram_ratio = if st.stats.ram_total_gb > 0.0 {
            (st.stats.ram_private_gb / st.stats.ram_total_gb).clamp(0.0, 1.0)
        } else if st.stats.ram_private_gb > 0.0 {
            (st.stats.ram_working_gb / st.stats.ram_private_gb).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut x = canvas.text(PAD_X, y, "RAM  ", Color::Muted);
        x = canvas.text(x, y, &format!("{:4.1} GB", st.stats.ram_private_gb), Color::White);
        let _ = canvas.text(x, y, &format!("  WORK {:4.1} GB", st.stats.ram_working_gb), Color::Muted);
        if st.stats.ram_total_gb > 0.0 {
            canvas.text_right(right_x, y, &format!("{:3.0}%", ram_ratio * 100.0), mem_bar_color(ram_ratio));
        }
        canvas.progress_bar(PAD_X, y + 9, COLUMNS * CELL.0, 4, ram_ratio, mem_bar_color(ram_ratio));
        y += 16;

        // VRAM + bar.
        let vram_ratio = if st.stats.vram_budget_gb > 0.0 {
            (st.stats.vram_used_gb / st.stats.vram_budget_gb).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut x = canvas.text(PAD_X, y, "VRAM ", Color::Muted);
        x = canvas.text(
            x,
            y,
            &format!("{:4.1}/{:.1} GB", st.stats.vram_used_gb, st.stats.vram_budget_gb),
            Color::White,
        );
        let _ = canvas.text(x, y, &format!("  SHR {:.1}", st.stats.vram_shared_gb), Color::Muted);
        if st.stats.vram_budget_gb > 0.0 {
            canvas.text_right(right_x, y, &format!("{:3.0}%", vram_ratio * 100.0), mem_bar_color(vram_ratio));
        }
        canvas.progress_bar(PAD_X, y + 9, COLUMNS * CELL.0, 4, vram_ratio, mem_bar_color(vram_ratio));
        y += 16;

        // Active mode summary: RT / FSR / AMBIENT.
        let mut x = canvas.text(PAD_X, y, "RT:", Color::Muted);
        x = canvas.text(x, y, st.rt.0, st.rt.1);
        x = canvas.text(x, y, "   FSR:", Color::Muted);
        x = canvas.text(x, y, st.fsr.0, st.fsr.1);
        x = canvas.text(x, y, "   AMBIENT:", Color::Muted);
        let _ = canvas.text(x, y, &st.ambient_badge.0, st.ambient_badge.1);
        y += 9 + 3;

        // Full HUD extra telemetry.
        if st.hud == HudMode::Full {
            canvas.fill_rect(1, y, PANEL_W - 2, 1, Color::Border);
            y += 1 + 3;

            let rows: [(&str, String, &str, String); 4] = [
                ("GPU BUSY ", format!("{:4.1}MS", st.stats.gpu_busy_ms), "IDLE BEHIND ", format!("{:4.1}MS", st.stats.idle_behind_ms)),
                ("SUB BUSY ", format!("{:4.1}MS", st.stats.submit_busy_ms), "GPU WAIT    ", format!("{:4.1}MS", st.stats.gpu_wait_ms)),
                ("RECORD   ", format!("{:4.0}MS", st.stats.recording_ms), "COPY BACK   ", format!("{:4.1}MS", st.stats.copy_back_ms)),
                ("DRAWS    ", format!("{:5.0}", st.stats.draws), "DROPPED     ", format!("{:5.0}", st.stats.dropped)),
            ];
            for (l1, v1, l2, v2) in rows {
                let x = canvas.text(PAD_X, y, l1, Color::Muted);
                let _ = canvas.text(x, y, &v1, Color::White);
                let mid = PAD_X + 19 * CELL.0;
                let x2 = canvas.text(mid, y, l2, Color::Muted);
                let _ = canvas.text(x2, y, &v2, Color::White);
                y += 9;
            }
            let mut x = canvas.text(PAD_X, y, "MB UP ", Color::Muted);
            x = canvas.text(x, y, &format!("{:4.0}", st.stats.mb_up), Color::White);
            x = canvas.text(x, y, "  TEX ", Color::Muted);
            x = canvas.text(x, y, &format!("{:4.0}", st.stats.mb_tex), Color::White);
            x = canvas.text(x, y, "  BACK ", Color::Muted);
            let _ = canvas.text(x, y, &format!("{:4.0}", st.stats.mb_back), Color::White);
            y += 9;

            let x = canvas.text(PAD_X, y, "PANEL COST ", Color::Muted);
            let _ = canvas.text(x, y, &format!("{:.0} US/F", st.stats.cost_us), Color::Cyan);
            y += 9 + 2;
        }
    }

    // F4 interactive debug menu.
    if st.menu_open {
        if st.hud != HudMode::Off {
            canvas.fill_rect(1, y, PANEL_W - 2, 1, Color::Border);
            y += 1;
        }
        y += 3;
        canvas.text(PAD_X, y, "DEBUG MENU", Color::Cyan);
        canvas.text_right(right_x, y, "[UP/DN] [LT/RT]", Color::Muted);
        y += 10;

        let scale_val = match st.ui_scale {
            UiScale::Auto => format!("AUTO ({}X)", st.resolved_scale),
            other => other.label().to_owned(),
        };
        let amb_val = match st.ambient_level {
            Some(v) if v > 0.0 => format!("{v:.1}"),
            _ => "0.0 (OFF)".to_owned(),
        };
        let items: [(&str, String, Color); MENU_ITEMS] = [
            ("HUD MODE", st.hud.label().to_owned(), Color::Cyan),
            ("FRAMETIME GRAPH", if st.graph_on { "ON" } else { "OFF" }.to_owned(), if st.graph_on { Color::Green } else { Color::Muted }),
            ("GRAPH SCALE", st.graph_scale.label().to_owned(), Color::Cyan),
            ("CORNER", st.corner.label().to_owned(), Color::Cyan),
            ("UI SCALE", scale_val, Color::Cyan),
            ("AMBIENT", amb_val, if st.ambient_level.is_some_and(|v| v > 0.0) { Color::Green } else { Color::Muted }),
            ("RESET STATS", "[ENTER]".to_owned(), Color::Amber),
        ];

        for (i, (label, val, val_col)) in items.into_iter().enumerate() {
            let selected = i == st.menu_sel;
            if selected {
                canvas.fill_rect(2, y, PANEL_W - 4, 10, Color::Highlight);
                canvas.text(PAD_X, y + 2, ">", Color::Cyan);
            }
            canvas.text(PAD_X + 2 * CELL.0, y + 2, label, if selected { Color::White } else { Color::Muted });
            canvas.text_right(right_x, y + 2, &val, if selected { Color::Cyan } else { val_col });
            y += 11;
        }
    }

    // Crisp outer 1px border.
    canvas.stroke_rect(0, 0, PANEL_W, h, Color::Border);
}

/// Computes the clamped destination `(dst_x, dst_y)` on the back buffer.
fn placement(corner: Corner, (width, height): (u32, u32), (copy_w, copy_h): (u32, u32), scale: u32) -> (u32, u32) {
    let pad = 6 * scale;
    let max_x = width.saturating_sub(copy_w);
    let max_y = height.saturating_sub(copy_h);
    let x = match corner {
        Corner::TopLeft | Corner::BotLeft => pad.min(max_x),
        Corner::TopRight | Corner::BotRight => max_x.saturating_sub(pad),
    };
    let y = match corner {
        Corner::TopLeft | Corner::TopRight => pad.min(max_y),
        Corner::BotLeft | Corner::BotRight => max_y.saturating_sub(pad),
    };
    (x, y)
}

/// Reusable max-sized GPU buffers for a given `(format, scale)`.
/// Toggling Compact / Full / F4 / Graph copies only the active sub-rectangle
/// with zero reallocations and zero VRAM leaks.
struct Buffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    max_w_px: u32,
    max_h_px: u32,
    palette: Palette,
    uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    turn: usize,
    fresh: bool,
    active_h_px: u32,
}

// SAFETY: free-threaded D3D12 resources; upload memory is only written under the presenter's lock.
unsafe impl Send for Buffers {}

impl Buffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<Buffers> {
        let palette = Palette::for_format(format)?;
        let max_w_px = PANEL_W * scale;
        let max_h_px = MAX_PANEL_H * scale;
        let pitch = (max_w_px * palette.bpp as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(max_h_px);
        let uploads = (0..UPLOADS).map(|_| host.upload_buffer(bytes)).collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain default-heap buffer creation.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(Buffers {
            format,
            scale,
            pitch,
            max_w_px,
            max_h_px,
            palette,
            uploads,
            card: card?,
            turn: 0,
            fresh: false,
            active_h_px: 0,
        })
    }

    fn write(&mut self, canvas: &Canvas) {
        let active_h_px = (canvas.h * self.scale).min(self.max_h_px);
        self.active_h_px = active_h_px;
        if active_h_px == 0 {
            return;
        }
        self.turn = (self.turn + 1) % UPLOADS;
        let len = self.pitch as usize * active_h_px as usize;
        // SAFETY: mapped upload buffer holds `pitch * max_h_px >= len` bytes and is unused by GPU.
        let dst = unsafe { std::slice::from_raw_parts_mut(self.uploads[self.turn].1, len) };
        canvas.blit(self.scale, &self.palette, self.pitch as usize, dst);
        self.fresh = true;
    }
}

/// The panel of a presenter.
pub(super) struct Panel {
    hud: HudMode,
    menu_open: bool,
    menu_sel: usize,
    graph_on: bool,
    graph_scale: GraphScale,
    corner: Corner,
    ui_scale: UiScale,
    keys: Keys,
    was_visible: bool,
    since: Instant,
    last: Instant,
    last_graph_paint: Instant,
    frames: u32,
    worst_ms: f64,
    cost: Duration,
    before: Counters,
    ring: FrameRing,
    stats: Stats,
    canvas: Canvas,
    dirty: bool,
    buffers: Option<Buffers>,
    retired: Vec<(Buffers, u32)>,
    failed: bool,
}

impl Panel {
    pub(super) fn new() -> Panel {
        let now = Instant::now();
        let hud = HudMode::from_env();
        Panel {
            hud,
            menu_open: false,
            menu_sel: 0,
            graph_on: true,
            graph_scale: GraphScale::Ms50,
            corner: Corner::TopLeft,
            ui_scale: UiScale::Auto,
            keys: Keys::default(),
            was_visible: hud != HudMode::Off,
            since: now,
            last: now,
            last_graph_paint: now,
            frames: 0,
            worst_ms: 0.0,
            cost: Duration::ZERO,
            before: Counters::default(),
            ring: FrameRing::new(),
            stats: Stats::default(),
            canvas: Canvas::new(PANEL_W, MAX_PANEL_H),
            dirty: hud != HudMode::Off,
            buffers: None,
            retired: Vec::new(),
            failed: false,
        }
    }

    pub(super) fn visible(&self) -> bool {
        self.hud != HudMode::Off || self.menu_open
    }

    fn reset_stats(&mut self) {
        self.worst_ms = 0.0;
        self.ring.clear();
        self.dirty = true;
    }

    fn apply_keys(&mut self, ev: KeyEvents) -> bool {
        let mut changed = false;
        if ev.f3 {
            self.hud = self.hud.next();
            changed = true;
        }
        if ev.f4 {
            self.menu_open = !self.menu_open;
            changed = true;
        }
        if !self.menu_open {
            return changed;
        }
        if ev.up {
            self.menu_sel = (self.menu_sel + MENU_ITEMS - 1) % MENU_ITEMS;
            changed = true;
        }
        if ev.down {
            self.menu_sel = (self.menu_sel + 1) % MENU_ITEMS;
            changed = true;
        }
        let step = if ev.left {
            -1
        } else if ev.right || ev.enter {
            1
        } else {
            0
        };
        if step != 0 {
            changed = true;
            match self.menu_sel {
                0 => self.hud = if step > 0 { self.hud.next() } else { self.hud.prev() },
                1 => self.graph_on = !self.graph_on,
                2 => self.graph_scale = if step > 0 { self.graph_scale.next() } else { self.graph_scale.prev() },
                3 => self.corner = if step > 0 { self.corner.next() } else { self.corner.prev() },
                4 => self.ui_scale = if step > 0 { self.ui_scale.next() } else { self.ui_scale.prev() },
                5 => {
                    let cur = crate::ambient::current_level().unwrap_or(0.0);
                    let ticks = ((cur * 5.0).round() as i32 + step).clamp(0, 15);
                    crate::ambient::set_live_level(ticks as f32 * 0.2);
                }
                6 => self.reset_stats(),
                _ => {}
            }
        }
        changed
    }

    fn refresh_memory(&mut self, host: &Host) {
        let gb = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let (working, private) = process_memory();
        let [budget, used, _, shared] = host.video_memory().unwrap_or_default();
        self.stats.ram_private_gb = gb(private);
        self.stats.ram_working_gb = gb(working);
        self.stats.ram_total_gb = gb(total_ram_bytes());
        self.stats.vram_used_gb = gb(used);
        self.stats.vram_budget_gb = gb(budget);
        self.stats.vram_shared_gb = gb(shared);
    }

    /// A frame was presented; `line`: the counters go back to 0 right behind
    /// this call. When hidden, only polls `F3`/`F4` and returns immediately.
    pub(super) fn frame(&mut self, line: bool) {
        // Tick retired buffer countdowns so old scale/format buffers drop cleanly.
        if !self.retired.is_empty() {
            self.retired.retain_mut(|(_, left)| {
                *left = left.saturating_sub(1);
                *left > 0
            });
        }

        let ev = self.keys.poll(self.menu_open);
        if self.apply_keys(ev) {
            self.dirty = true;
        }
        if !self.visible() {
            self.was_visible = false;
            self.frames = 0;
            return;
        }

        let now = Instant::now();
        if !self.was_visible || (self.frames == 0 && now.saturating_duration_since(self.last) > 2 * STATS_EVERY) {
            self.was_visible = true;
            self.since = now;
            self.last = now;
            self.last_graph_paint = now;
            self.frames = 0;
            self.cost = Duration::ZERO;
            self.before = Counters::now();
            if let Some(host) = super::host() {
                self.refresh_memory(host);
            }
            self.dirty = true;
            return;
        }

        let dt_ms = now.saturating_duration_since(self.last).as_secs_f64() * 1e3;
        self.last = now;
        self.frames += 1;
        if dt_ms.is_finite() && dt_ms > 0.0 {
            self.worst_ms = self.worst_ms.max(dt_ms);
            self.ring.push(dt_ms as f32);
        }

        let elapsed = now.saturating_duration_since(self.since);
        if elapsed >= STATS_EVERY || (line && elapsed * 2 >= STATS_EVERY) {
            if let Some(host) = super::host() {
                let frames = f64::from(self.frames);
                let counters = Counters::now();
                let c = counters.since(self.before, frames);
                self.before = counters;
                let ms = elapsed.as_secs_f64() * 1e3 / frames;
                let idle = (c[1] / 1e3).min(ms);
                self.stats.valid = true;
                self.stats.fps = 1e3 / ms.max(0.001);
                self.stats.frame_ms = ms;
                self.stats.gpu_busy_ms = (ms - idle).max(0.0);
                self.stats.idle_behind_ms = c[2] / 1e3;
                self.stats.submit_busy_ms = c[0] / 1e3;
                self.stats.gpu_wait_ms = c[3] / 1e3;
                self.stats.recording_ms = c[9] / 1e6;
                self.stats.copy_back_ms = c[5] / 1e3;
                self.stats.draws = c[10];
                self.stats.dropped = c[8];
                self.stats.mb_up = c[7] / 1e6;
                self.stats.mb_tex = c[4] / 1e6;
                self.stats.mb_back = c[6] / 1e6;
                self.stats.cost_us = self.cost.as_secs_f64() * 1e6 / frames;
                self.refresh_memory(host);
                self.dirty = true;
                self.last_graph_paint = now;
                self.since = now;
                self.frames = 0;
                self.cost = now.elapsed();
            }
            return;
        }

        // Smoothly refresh the frametime graph at ~12.5 Hz without jittering 500 ms counters.
        if self.hud != HudMode::Off && self.graph_on && now.saturating_duration_since(self.last_graph_paint) >= GRAPH_EVERY {
            self.dirty = true;
            self.last_graph_paint = now;
        }
    }

    /// Records the copy of the active panel onto `back`, a swap chain buffer
    /// in `D3D12_RESOURCE_STATE_COPY_DEST`.
    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
    ) {
        if !self.visible() || self.failed || width == 0 || height == 0 {
            return;
        }
        let began = Instant::now();
        // Not the title's: the frame recorder leaves it out.
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let scale = self.ui_scale.resolve(height);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            if let Some(old) = self.buffers.take() {
                self.retired.push((old, RETIRE_FRAMES));
            }
            self.buffers = Buffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.dirty = true;
        }
        if !self.stats.valid && self.stats.vram_budget_gb == 0.0 {
            self.refresh_memory(host);
        }
        let Some(buffers) = &mut self.buffers else { return };
        if std::mem::take(&mut self.dirty) || buffers.active_h_px == 0 {
            let st = RenderState {
                hud: self.hud,
                graph_on: self.graph_on,
                graph_scale: self.graph_scale,
                corner: self.corner,
                ui_scale: self.ui_scale,
                resolved_scale: scale,
                menu_open: self.menu_open,
                menu_sel: self.menu_sel,
                stats: &self.stats,
                worst_ms: self.worst_ms,
                ring: &self.ring,
                ambient_level: crate::ambient::current_level(),
                rt: active_rt(),
                fsr: active_fsr(),
                ambient_badge: active_ambient(),
            };
            render_panel(&mut self.canvas, &st);
            buffers.write(&self.canvas);
        }

        let active_w = buffers.max_w_px;
        let active_h = buffers.active_h_px;
        let copy_w = active_w.min(width);
        let copy_h = active_h.min(height);
        if copy_w == 0 || copy_h == 0 {
            return;
        }
        let (dst_x, dst_y) = placement(self.corner, (width, height), (copy_w, copy_h), scale);

        let list = &list.list;
        // SAFETY: an open list and live resources; active sub-rect lies within both.
        unsafe {
            if std::mem::take(&mut buffers.fresh) {
                let copy_bytes = u64::from(buffers.pitch) * u64::from(active_h);
                list.CopyBufferRegion(&buffers.card, 0, &buffers.uploads[buffers.turn].0, 0, copy_bytes);
                list.ResourceBarrier(&[transition(
                    &buffers.card,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffers.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: active_w,
                            Height: active_h,
                            Depth: 1,
                            RowPitch: buffers.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX {
                left: 0,
                top: 0,
                front: 0,
                right: copy_w,
                bottom: copy_h,
                back: 1,
            };
            list.CopyTextureRegion(&to, dst_x, dst_y, 0, &from, Some(&part));
        }
        self.cost += began.elapsed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character used by the HUD and menu has its own distinct 5x7 glyph.
    #[test]
    fn characters_differ() {
        let used: Vec<u8> = (b'0'..=b'9')
            .chain(b'A'..=b'Z')
            .chain(*b"./- :%+<>[](),_=")
            .collect();
        for (i, a) in used.iter().enumerate() {
            assert_ne!(glyph(*a), glyph(b'#'), "{} has a picture", *a as char);
            assert!(glyph(*a).iter().all(|row| row >> GLYPH.0 == 0));
            for b in &used[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{} {}", *a as char, *b as char);
            }
        }
    }

    #[test]
    fn format_encoders_match_bit_layouts() {
        // R8G8B8A8 and B8G8R8A8 channel order.
        let (rgba, bpp) = encode_pixel(DXGI_FORMAT_R8G8B8A8_UNORM, [0x11, 0x22, 0x33, 0xFF]).unwrap();
        assert_eq!(bpp, 4);
        assert_eq!(&rgba[..4], &[0x11, 0x22, 0x33, 0xFF]);
        let (bgra, _) = encode_pixel(DXGI_FORMAT_B8G8R8A8_UNORM, [0x11, 0x22, 0x33, 0xFF]).unwrap();
        assert_eq!(&bgra[..4], &[0x33, 0x22, 0x11, 0xFF]);

        // R10G10B10A2: R in 0..9, G in 10..19, B in 20..29, A in 30..31.
        let (red10, _) = encode_pixel(DXGI_FORMAT_R10G10B10A2_UNORM, [255, 0, 0, 255]).unwrap();
        assert_eq!(u32::from_le_bytes(red10[..4].try_into().unwrap()), 1023 | (3 << 30));
        let (green10, _) = encode_pixel(DXGI_FORMAT_R10G10B10A2_UNORM, [0, 255, 0, 255]).unwrap();
        assert_eq!(u32::from_le_bytes(green10[..4].try_into().unwrap()), (1023 << 10) | (3 << 30));
        let (blue10, _) = encode_pixel(DXGI_FORMAT_R10G10B10A2_UNORM, [0, 0, 255, 255]).unwrap();
        assert_eq!(u32::from_le_bytes(blue10[..4].try_into().unwrap()), (1023 << 20) | (3 << 30));

        // R16G16B16A16_FLOAT: black = (0,0,0,1), white = (1,1,1,1).
        let (blk, bpp16) = encode_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT, [0, 0, 0, 255]).unwrap();
        assert_eq!(bpp16, 8);
        assert_eq!(blk, [0, 0, 0, 0, 0, 0, 0, 0x3C]);
        let (wht, _) = encode_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT, [255, 255, 255, 255]).unwrap();
        assert_eq!(wht, [0, 0x3C, 0, 0x3C, 0, 0x3C, 0, 0x3C]);
        assert_eq!(f32_to_f16(0.5), 0x3800);
    }

    #[test]
    fn panel_renders_and_blits_within_max_height() {
        let mut ring = FrameRing::new();
        for i in 0..120 {
            ring.push(if i % 30 == 0 { 40.0 } else if i % 10 == 0 { 22.0 } else { 15.0 });
        }
        assert!(ring.one_percent_low_fps() > 20.0 && ring.one_percent_low_fps() < 30.0);

        let stats = Stats {
            valid: true,
            fps: 60.0,
            frame_ms: 16.6,
            ram_private_gb: 8.2,
            ram_working_gb: 6.4,
            ram_total_gb: 32.0,
            vram_used_gb: 6.0,
            vram_budget_gb: 12.0,
            ..Stats::default()
        };
        let mut canvas = Canvas::new(PANEL_W, MAX_PANEL_H);
        let st = RenderState {
            hud: HudMode::Full,
            graph_on: true,
            graph_scale: GraphScale::Ms50,
            corner: Corner::TopLeft,
            ui_scale: UiScale::Auto,
            resolved_scale: 2,
            menu_open: true,
            menu_sel: 2,
            stats: &stats,
            worst_ms: 40.0,
            ring: &ring,
            ambient_level: Some(1.8),
            rt: ("DXR", Color::Green),
            fsr: ("ON", Color::Green),
            ambient_badge: ("1.8".to_owned(), Color::Cyan),
        };
        render_panel(&mut canvas, &st);
        assert!(canvas.h > 0 && canvas.h <= MAX_PANEL_H);

        for fmt in [
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
        ] {
            let palette = Palette::for_format(fmt).unwrap();
            let scale = 2;
            let pitch = (PANEL_W * scale * palette.bpp as u32).next_multiple_of(256) as usize;
            let mut out = vec![0u8; pitch * (canvas.h * scale) as usize];
            canvas.blit(scale, &palette, pitch, &mut out);
            // Top-left corner pixel is the border color.
            assert_eq!(&out[..palette.bpp], palette.bytes(Color::Border));
            // Header strip interior uses the header background color.
            let hdr_off = pitch * (2 * scale) as usize + (2 * scale) as usize * palette.bpp;
            assert_eq!(&out[hdr_off..hdr_off + palette.bpp], palette.bytes(Color::Header));
        }
    }

    #[test]
    fn corner_placement_stays_in_bounds() {
        for corner in [Corner::TopLeft, Corner::TopRight, Corner::BotLeft, Corner::BotRight] {
            let (x, y) = placement(corner, (1920, 1080), (504, 300), 2);
            assert!(x + 504 <= 1920 && y + 300 <= 1080);
            // Tiny back buffer still clamps cleanly.
            let (x2, y2) = placement(corner, (100, 80), (100, 80), 2);
            assert_eq!((x2, y2), (0, 0));
        }
    }

    #[test]
    fn menu_navigation_and_reset_stats() {
        let mut p = Panel::new();
        p.hud = HudMode::Off;
        p.menu_open = false;

        // F3 cycles Off -> Compact -> Full -> Off.
        p.apply_keys(KeyEvents { f3: true, ..Default::default() });
        assert_eq!(p.hud, HudMode::Compact);
        p.apply_keys(KeyEvents { f3: true, ..Default::default() });
        assert_eq!(p.hud, HudMode::Full);
        p.apply_keys(KeyEvents { f3: true, ..Default::default() });
        assert_eq!(p.hud, HudMode::Off);

        // F4 opens menu even when HUD is Off.
        p.apply_keys(KeyEvents { f4: true, ..Default::default() });
        assert!(p.menu_open && p.visible());
        assert_eq!(layout_height(p.hud, p.graph_on, p.menu_open), 109);

        // Navigate down through items and toggle values.
        p.apply_keys(KeyEvents { right: true, ..Default::default() }); // item 0: HUD MODE
        assert_eq!(p.hud, HudMode::Compact);
        p.apply_keys(KeyEvents { down: true, ..Default::default() }); // item 1: GRAPH
        p.apply_keys(KeyEvents { right: true, ..Default::default() });
        assert!(!p.graph_on);
        p.apply_keys(KeyEvents { down: true, ..Default::default() }); // item 2: GRAPH SCALE
        p.apply_keys(KeyEvents { right: true, ..Default::default() });
        assert_eq!(p.graph_scale, GraphScale::Ms100);
        p.apply_keys(KeyEvents { down: true, ..Default::default() }); // item 3: CORNER
        p.apply_keys(KeyEvents { right: true, ..Default::default() });
        assert_eq!(p.corner, Corner::TopRight);
        p.apply_keys(KeyEvents { down: true, ..Default::default() }); // item 4: UI SCALE
        p.apply_keys(KeyEvents { right: true, ..Default::default() });
        assert_eq!(p.ui_scale, UiScale::X1);

        // Item 6: RESET STATS clears worst_ms and ring.
        p.worst_ms = 88.0;
        p.ring.push(88.0);
        p.menu_sel = 6;
        p.apply_keys(KeyEvents { enter: true, ..Default::default() });
        assert_eq!(p.worst_ms, 0.0);
        assert_eq!(p.ring.len, 0);
    }

    #[test]
    fn counters_that_went_back_to_nothing_count_from_there() {
        let (mut a, mut b) = (Counters::default(), Counters::default());
        (a.0[0], b.0[0]) = (10, 30);
        (a.0[1], b.0[1]) = (50, 20);
        let c = b.since(a, 10.0);
        assert_eq!((c[0], c[1]), (2.0, 2.0));
    }
}
