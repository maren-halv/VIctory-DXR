//! `gameinput.dll`: input devices from the host. On Windows, prefers the
//! native PC `GameInput.dll` backend (low-latency gamepads with 4-motor
//! impulse triggers, native mouse, and Unicode keyboard state) and falls back
//! to XInput + Win32 keyboard/mouse when `VICTORY_GAMEINPUT_NATIVE=0` or the
//! system DLL is unavailable. While no gamepad is connected, the keyboard
//! drives one stand-in gamepad (`FALLBACK_BUTTONS`). The keyboard and mouse
//! are devices of their own too. A poll thread turns host state into readings
//! (a new one per change) and runs device and reading callbacks.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{
    Condvar, LazyLock, Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const MOUSE_KIND: u32 = 0x20;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// `GameInputRumbleMotors` bitmasks.
const RUMBLE_MOTORS_XINPUT: u32 = 0x3;
const RUMBLE_MOTORS_ALL: u32 = 0xF;

/// `GameInputMouseButtons` bitmasks.
const MOUSE_LEFT: u32 = 0x01;
const MOUSE_RIGHT: u32 = 0x02;
const MOUSE_MIDDLE: u32 = 0x04;
const MOUSE_BUTTON4: u32 = 0x08;
const MOUSE_BUTTON5: u32 = 0x10;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
const MENU: u32 = 0x1;
const VIEW: u32 = 0x2;
const A: u32 = 0x4;
const B: u32 = 0x8;
const X: u32 = 0x10;
const Y: u32 = 0x20;
const UP: u32 = 0x40;
const DOWN: u32 = 0x80;
const LEFT: u32 = 0x100;
const RIGHT: u32 = 0x200;
const LEFT_SHOULDER: u32 = 0x400;
const RIGHT_SHOULDER: u32 = 0x800;
const LEFT_STICK: u32 = 0x1000;
const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard, the mouse.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const MOUSE: usize = PADS + 2;
const DEVICES: usize = PADS + 3;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// The keyboard's stand-in gamepad (virtual key, button); README lists it.
/// Sticks: W/A/S/D left, I/J/K/L right; triggers Z and C.
const FALLBACK_BUTTONS: [(u8, u32); 14] = [
    (0x1B, MENU), // Esc
    (0x09, VIEW), // Tab
    (0x0D, A),    // Enter
    (0x08, B),    // Backspace
    (b'X', X),
    (b'Y', Y),
    (0x26, UP), // arrows
    (0x28, DOWN),
    (0x25, LEFT),
    (0x27, RIGHT),
    (b'Q', LEFT_SHOULDER),
    (b'E', RIGHT_SHOULDER),
    (b'F', LEFT_STICK),
    (b'H', RIGHT_STICK),
];

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PadState {
    buttons: u32,
    left_trigger: f32,
    right_trigger: f32,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

/// A mouse reading (`GameInputMouseState`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MouseState {
    buttons: u32,
    position_x: i64,
    position_y: i64,
    wheel_x: i64,
    wheel_y: i64,
}

const _: () = assert!(
    size_of::<KeyState>() == 12
        && size_of::<PadState>() == 28
        && size_of::<MouseState>() == 40
        && align_of::<MouseState>() == 8
        && std::mem::offset_of!(MouseState, buttons) == 0
        && std::mem::offset_of!(MouseState, position_x) == 8
        && std::mem::offset_of!(MouseState, position_y) == 16
        && std::mem::offset_of!(MouseState, wheel_x) == 24
        && std::mem::offset_of!(MouseState, wheel_y) == 32
);

/// Host input, or a fake in tests.
trait Source: Send {
    /// Supported `GameInputRumbleMotors` bitmask advertised on pad devices.
    fn rumble_motors(&self) -> u32 {
        RUMBLE_MOTORS_XINPUT
    }
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Writes held keys into `out` (reusing its capacity); empty while unfocused.
    fn keys(&mut self, out: &mut Vec<KeyState>);
    /// Current mouse state, `None` when no mouse is present.
    fn mouse(&mut self) -> Option<MouseState>;
    /// Motor levels (`0.0..=1.0`): `[low, high, left_trigger, right_trigger]`.
    fn rumble(&mut self, slot: usize, motors: [f32; 4]);
}

fn fallback_with(on: impl Fn(u8) -> f32) -> PadState {
    PadState {
        buttons: FALLBACK_BUTTONS
            .iter()
            .filter(|(vk, _)| on(*vk) != 0.0)
            .fold(0, |all, (_, bit)| all | bit),
        left_trigger: on(b'Z'),
        right_trigger: on(b'C'),
        left_x: on(b'D') - on(b'A'),
        left_y: on(b'W') - on(b'S'),
        right_x: on(b'L') - on(b'J'),
        right_y: on(b'I') - on(b'K'),
    }
}

fn fallback(keys: &[KeyState]) -> PadState {
    fallback_with(|vk| {
        if keys.iter().any(|k| k.virtual_key == vk) {
            1.0
        } else {
            0.0
        }
    })
}

fn fallback_vks(vks: &[u8]) -> PadState {
    fallback_with(|vk| if vks.contains(&vk) { 1.0 } else { 0.0 })
}

/// Two pads as one: buttons of both, and of each axis and trigger the one
/// pushed further.
fn merged(a: PadState, b: PadState) -> PadState {
    let further = |x: f32, y: f32| if y.abs() > x.abs() { y } else { x };
    PadState {
        buttons: a.buttons | b.buttons,
        left_trigger: further(a.left_trigger, b.left_trigger),
        right_trigger: further(a.right_trigger, b.right_trigger),
        left_x: further(a.left_x, b.left_x),
        left_y: further(a.left_y, b.left_y),
        right_x: further(a.right_x, b.right_x),
        right_y: further(a.right_y, b.right_y),
    }
}

fn clamp_motor(level: f32) -> f32 {
    if level.is_finite() {
        level.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Whether the title sees one pad that is always there, fed by the keyboard
/// and by every host pad (on by default; `VICTORY_ONE_PAD=0` shows each host
/// pad as a device of its own, and the stand-in only while there is none).
/// The title keeps to the pad it started with: with the stand-in taken away
/// when a pad was switched on, it showed its controller-disconnected screen
/// until the pad was off again. It also keeps a pad the host lists twice (its
/// own slot and a remapper's) from being two players.
fn one_pad() -> bool {
    std::env::var_os("VICTORY_ONE_PAD").is_none_or(|v| v != "0")
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(
    std::mem::offset_of!(DeviceInfo, id) == 0x20
        && std::mem::offset_of!(DeviceInfo, supported_input) == 0x68
        && std::mem::offset_of!(DeviceInfo, rumble_motors) == 0x6c
);

fn kind(index: usize) -> u32 {
    match index {
        KEYBOARD => KEYBOARD_KIND,
        MOUSE => MOUSE_KIND,
        _ => GAMEPAD_KIND,
    }
}

impl Device {
    fn new(index: usize, pad_rumble_motors: u32) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            rumble_motors: if index <= FALLBACK {
                pad_rumble_motors
            } else {
                0
            },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
    Mouse(MouseState),
}

impl Value {
    fn matches_keys(&self, keys: &[KeyState]) -> bool {
        matches!(self, Value::Keys(prev) if prev.as_slice() == keys)
    }
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
            Value::Mouse(_) => MOUSE_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

// SAFETY: `Reading` is immutable after creation except for its atomic refcount,
// and `device` points into a permanently live `Hub`.
unsafe impl Send for Ref {}
// SAFETY: as above.
unsafe impl Sync for Ref {}

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Clone for Ref {
    fn clone(&self) -> Ref {
        reading_add_ref(self.0.as_ptr());
        Ref(self.0)
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);
type ReadingCallback = extern "system" fn(u64, *mut c_void, *const Reading, bool);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> DeviceEvent {
        DeviceEvent {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct ReadingRegistration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    threshold: f32,
    context: usize,
    callback: ReadingCallback,
    last_pad: [Option<PadState>; PADS + 1],
}

fn pad_exceeds_threshold(prev: PadState, curr: PadState, threshold: f32) -> bool {
    prev.buttons != curr.buttons
        || (curr.left_trigger - prev.left_trigger).abs() >= threshold
        || (curr.right_trigger - prev.right_trigger).abs() >= threshold
        || (curr.left_x - prev.left_x).abs() >= threshold
        || (curr.left_y - prev.left_y).abs() >= threshold
        || (curr.right_x - prev.right_x).abs() >= threshold
        || (curr.right_y - prev.right_y).abs() >= threshold
}

impl ReadingRegistration {
    fn matches(&mut self, devices: &[Device; DEVICES], index: usize, reading: &Reading) -> bool {
        if reading.kind() & self.kind == 0 {
            return false;
        }
        if self.device != 0 && self.device != &devices[index] as *const Device as usize {
            return false;
        }
        if let Value::Pad(pad) = reading.value
            && index <= FALLBACK
        {
            if self.threshold > 0.0
                && let Some(prev) = self.last_pad[index]
                && !pad_exceeds_threshold(prev, pad, self.threshold)
            {
                return false;
            }
            self.last_pad[index] = Some(pad);
        }
        true
    }
}

struct DeviceEvent {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

struct ReadingEvent {
    token: u64,
    context: usize,
    callback: ReadingCallback,
    reading: Ref,
}

enum Event {
    Device(DeviceEvent),
    Reading(ReadingEvent),
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
}

#[derive(Default)]
struct CallbackState {
    callbacks: Vec<Registration>,
    reading_callbacks: Vec<ReadingRegistration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<DeviceEvent>,
    /// Callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
}

struct Poller {
    source: Box<dyn Source>,
    keys_buf: Vec<KeyState>,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Lock order: `poller` -> `slots` -> `callbacks`.
    /// Apart from `slots`: host polling can be slow, and the title reads
    /// readings every frame.
    poller: Mutex<Poller>,
    /// The title sees one pad ([`one_pad`]).
    one_pad: bool,
    /// Per-device reading history; read-shared on the per-frame hot path.
    slots: RwLock<[Slot; DEVICES]>,
    /// Lock-free mirror of `slots[i].connected` for `GetDeviceStatus`.
    connected: [AtomicBool; DEVICES],
    /// Set when `callbacks.pending` is non-empty so idle polls skip locking `callbacks`.
    has_pending: AtomicBool,
    callbacks: Mutex<CallbackState>,
    /// A callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn read_lock<T: ?Sized>(rw: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rw.read().unwrap_or_else(PoisonError::into_inner)
}

fn write_lock<T: ?Sized>(rw: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rw.write().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind locks or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>, one_pad: bool) -> &'static Hub {
        let motors = source.rumble_motors();
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(|i| Device::new(i, motors)),
            poller: Mutex::new(Poller {
                source,
                keys_buf: Vec::new(),
            }),
            one_pad,
            slots: RwLock::new(Default::default()),
            connected: std::array::from_fn(|_| AtomicBool::new(false)),
            has_pending: AtomicBool::new(false),
            callbacks: Mutex::new(CallbackState::default()),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>) -> &'static Hub {
        let hub = Hub::new(source, one_pad());
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn slots(&self) -> RwLockReadGuard<'_, [Slot; DEVICES]> {
        read_lock(&self.slots)
    }

    fn callbacks(&self) -> MutexGuard<'_, CallbackState> {
        lock(&self.callbacks)
    }

    /// Reads the host once; device and reading callbacks run on the calling thread.
    fn poll(&self) {
        let events = {
            let mut poller = lock(&self.poller);
            let Poller { source, keys_buf } = &mut *poller;
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            source.keys(keys_buf);
            let mouse = source.mouse();

            let pad_targets: [Option<PadState>; PADS + 1] = if self.one_pad {
                let combined = pads.into_iter().flatten().fold(fallback(keys_buf), merged);
                [None, None, None, None, Some(combined)]
            } else {
                let any_pad = pads.iter().any(Option::is_some);
                let fb = (!any_pad).then(|| fallback(keys_buf));
                [pads[0], pads[1], pads[2], pads[3], fb]
            };

            // Fast path: when no device connected/disconnected, no reading value
            // changed, and no async enumeration is pending, avoid write locks,
            // callback locks, and allocations entirely.
            if !self.has_pending.load(Ordering::Acquire) {
                let slots = self.slots();
                let pads_same = (0..=FALLBACK).all(|i| slot_pad_unchanged(&slots[i], pad_targets[i]));
                let keys_same = slots[KEYBOARD].connected
                    && slots[KEYBOARD]
                        .history
                        .back()
                        .is_some_and(|r| r.value.matches_keys(keys_buf));
                let mouse_same = slot_mouse_unchanged(&slots[MOUSE], mouse);
                if pads_same && keys_same && mouse_same {
                    return;
                }
            }

            let now = now();
            let mut slots = write_lock(&self.slots);
            let mut transitions: Vec<(usize, u32, u32)> = Vec::new();
            let mut new_readings: Vec<(usize, Ref)> = Vec::new();

            for (i, pad) in pad_targets.into_iter().enumerate() {
                update_slot(
                    &self.devices,
                    &self.connected[i],
                    &mut slots[i],
                    i,
                    pad.map(Value::Pad),
                    now,
                    &mut transitions,
                    &mut new_readings,
                );
            }

            let keys_changed = !slots[KEYBOARD].connected
                || slots[KEYBOARD]
                    .history
                    .back()
                    .is_none_or(|r| !r.value.matches_keys(keys_buf));
            if keys_changed {
                update_slot(
                    &self.devices,
                    &self.connected[KEYBOARD],
                    &mut slots[KEYBOARD],
                    KEYBOARD,
                    Some(Value::Keys(keys_buf.clone())),
                    now,
                    &mut transitions,
                    &mut new_readings,
                );
            }

            update_slot(
                &self.devices,
                &self.connected[MOUSE],
                &mut slots[MOUSE],
                MOUSE,
                mouse.map(Value::Mouse),
                now,
                &mut transitions,
                &mut new_readings,
            );

            let mut events = Vec::new();
            let need_cb = !transitions.is_empty()
                || !new_readings.is_empty()
                || self.has_pending.load(Ordering::Acquire);
            if need_cb {
                let mut cb = self.callbacks();
                if !cb.pending.is_empty() {
                    events.extend(cb.pending.drain(..).map(Event::Device));
                    self.has_pending.store(false, Ordering::Release);
                }
                for (index, current, previous) in transitions {
                    events.extend(
                        cb.callbacks
                            .iter()
                            .filter(|r| {
                                r.matches(&self.devices, index) && r.status & STATUS_CONNECTED != 0
                            })
                            .map(|r| Event::Device(r.event(index, now, current, previous))),
                    );
                }
                if !cb.reading_callbacks.is_empty() {
                    for (index, reading_ref) in &new_readings {
                        for r in &mut cb.reading_callbacks {
                            if r.matches(&self.devices, *index, reading_ref) {
                                events.push(Event::Reading(ReadingEvent {
                                    token: r.token,
                                    context: r.context,
                                    callback: r.callback,
                                    reading: reading_ref.clone(),
                                }));
                            }
                        }
                    }
                }
            }
            events
        };
        if !events.is_empty() {
            self.deliver(events);
        }
    }

    /// Runs callbacks, never under our locks: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for event in events {
            match event {
                Event::Device(e) => {
                    let r = e.registration;
                    {
                        let mut cb = self.callbacks();
                        if !cb.callbacks.iter().any(|c| c.token == r.token) {
                            continue;
                        }
                        cb.running.push((r.token, me));
                    }
                    (r.callback)(
                        r.token,
                        r.context as *mut c_void,
                        &self.devices[e.device],
                        e.timestamp,
                        e.current,
                        e.previous,
                    );
                    let mut cb = self.callbacks();
                    if let Some(i) = cb.running.iter().position(|&x| x == (r.token, me)) {
                        cb.running.swap_remove(i);
                    }
                    self.returned.notify_all();
                }
                Event::Reading(e) => {
                    {
                        let mut cb = self.callbacks();
                        if !cb.reading_callbacks.iter().any(|c| c.token == e.token) {
                            continue;
                        }
                        cb.running.push((e.token, me));
                    }
                    (e.callback)(
                        e.token,
                        e.context as *mut c_void,
                        e.reading.0.as_ptr(),
                        false,
                    );
                    drop(e.reading);
                    let mut cb = self.callbacks();
                    if let Some(i) = cb.running.iter().position(|&x| x == (e.token, me)) {
                        cb.running.swap_remove(i);
                    }
                    self.returned.notify_all();
                }
            }
        }
    }

    /// O(1) lookup of a device pointer belonging to this hub.
    fn index(&self, device: *const Device) -> Option<usize> {
        let base = self.devices.as_ptr() as usize;
        let diff = (device as usize).checked_sub(base)?;
        let stride = size_of::<Device>();
        if diff.is_multiple_of(stride) {
            let index = diff / stride;
            if index < DEVICES {
                return Some(index);
            }
        }
        None
    }
}

fn slot_pad_unchanged(slot: &Slot, target: Option<PadState>) -> bool {
    match target {
        None => !slot.connected,
        Some(pad) => {
            slot.connected
                && slot
                    .history
                    .back()
                    .is_some_and(|r| matches!(r.value, Value::Pad(p) if p == pad))
        }
    }
}

fn slot_mouse_unchanged(slot: &Slot, target: Option<MouseState>) -> bool {
    match target {
        None => !slot.connected,
        Some(mouse) => {
            slot.connected
                && slot
                    .history
                    .back()
                    .is_some_and(|r| matches!(r.value, Value::Mouse(m) if m == mouse))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_slot(
    devices: &[Device; DEVICES],
    connected_flag: &AtomicBool,
    slot: &mut Slot,
    index: usize,
    value: Option<Value>,
    now: u64,
    transitions: &mut Vec<(usize, u32, u32)>,
    new_readings: &mut Vec<(usize, Ref)>,
) {
    let was = slot.connected;
    slot.connected = value.is_some();
    if was != slot.connected {
        connected_flag.store(slot.connected, Ordering::Release);
    }
    if let Some(value) = value
        && (!was || slot.history.back().is_none_or(|r| r.value != value))
    {
        slot.sequence += 1;
        if slot.history.len() == HISTORY {
            slot.history.pop_front();
        }
        let reading = Ref::new(Reading {
            vtable: &READING,
            refs: AtomicU32::new(1),
            device: &devices[index],
            sequence: slot.sequence,
            timestamp: now,
            value,
        });
        new_readings.push((index, reading.clone()));
        slot.history.push_back(reading);
    }
    if was != slot.connected {
        tracing::info!(device = index, connected = !was, "input device");
        let (current, previous) = if was {
            (0, STATUS_CONNECTED)
        } else {
            (STATUS_CONNECTED, 0)
        };
        transitions.push((index, current, previous));
    }
}

#[cfg(windows)]
fn host() -> Box<dyn Source> {
    host::create_source(host::native_enabled())
}

#[cfg(not(windows))]
fn host() -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self, out: &mut Vec<KeyState>) {
            out.clear();
        }
        fn mouse(&mut self) -> Option<MouseState> {
            None
        }
        fn rumble(&mut self, _: usize, _: [f32; 4]) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| Hub::start(host()));

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    input_kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let dev_index = if device.is_null() {
        None
    } else {
        match hub.index(device) {
            Some(i) => Some(i),
            None => return fail(out, E_INVALIDARG),
        }
    };
    let slots = hub.slots();
    let found = match dev_index {
        None => (0..DEVICES)
            .filter(|&i| kind(i) & input_kind != 0 && slots[i].connected)
            .filter_map(|i| slots[i].history.back())
            .max_by_key(|r| r.timestamp),
        Some(i) => {
            if !slots[i].connected {
                return fail(out, DEVICE_DISCONNECTED);
            }
            slots[i].history.back().filter(|r| r.kind() & input_kind != 0)
        }
    };
    tracing::trace!(
        kind = format_args!("{input_kind:#x}"),
        device = ?dev_index,
        sequence = found.map(|r| r.sequence),
        "current reading"
    );
    hand(out, found)
}

/// The reading after (`forward`) or before `reference` on its device.
fn relative(
    this: This,
    reference: *const Reading,
    input_kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    // shortcut: with no device given only the reference's device is walked,
    // not every device's readings in time order; the title walks
    // only keyboard readings (one device) and the current pad's previous one.
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let slots = hub.slots();
    let history = &slots[index].history;
    if history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    let wanted = |r: &&Ref| r.kind() & input_kind != 0;
    let found = if forward {
        history
            .iter()
            .filter(|r| r.sequence > reference.sequence)
            .find(wanted)
    } else {
        history
            .iter()
            .rev()
            .filter(|r| r.sequence < reference.sequence)
            .find(wanted)
    };
    hand(out, found)
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    input_kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, input_kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    input_kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, input_kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let slots = hub.slots();
    let history = &slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

fn fail_token(out: *mut u64, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(0) };
    }
    hr
}

extern "system" fn register_reading(
    this: This,
    device: *const Device,
    input_kind: u32,
    threshold: f32,
    context: *mut c_void,
    callback: Option<ReadingCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return fail_token(out, E_INVALIDARG);
    };
    if input_kind == 0 || !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return fail_token(out, E_INVALIDARG);
    }
    let hub = hub(this);
    if !device.is_null() && hub.index(device).is_none() {
        return fail_token(out, E_INVALIDARG);
    }
    let tok = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let registration = ReadingRegistration {
        token: tok,
        device: device as usize,
        kind: input_kind,
        threshold,
        context: context as usize,
        callback,
        last_pad: [None; PADS + 1],
    };
    hub.callbacks().reading_callbacks.push(registration);
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(tok) };
    }
    S_OK
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    input_kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return fail_token(out, E_INVALIDARG);
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind: input_kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        // Respect lock order: `slots` -> `callbacks`.
        let slots = hub.slots();
        let mut cb = hub.callbacks();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{input_kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        cb.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION && !found.is_empty() {
            cb.pending.append(&mut found);
            hub.has_pending.store(true, Ordering::Release);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    if !found.is_empty() {
        hub.deliver(found.into_iter().map(Event::Device).collect());
    }
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let hub = hub(this);
    let mut cb = hub.callbacks();
    cb.callbacks.retain(|r| r.token != token);
    cb.reading_callbacks.retain(|r| r.token != token);
    cb.pending.retain(|e| e.registration.token != token);
    if cb.pending.is_empty() {
        hub.has_pending.store(false, Ordering::Release);
    }
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut cb = hub.callbacks();
    loop {
        let running = cb.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        cb = match deadline {
            None => hub.returned.wait(cb).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(cb, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().connected[device.index].load(Ordering::Acquire) {
        STATUS_CONNECTED
    } else {
        0
    }
}
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    let one_pad = device.hub().one_pad;
    if device.index >= PADS && !(one_pad && device.index == FALLBACK) {
        return;
    }
    let raw = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let motors = [
        clamp_motor(raw[0]),
        clamp_motor(raw[1]),
        clamp_motor(raw[2]),
        clamp_motor(raw[3]),
    ];
    let mut poller = lock(&device.hub().poller);
    if one_pad {
        // The one pad is every host pad.
        for slot in 0..PADS {
            if poller.source.pad(slot).is_some() {
                poller.source.rumble(slot, motors);
            }
        }
    } else {
        poller.source.rumble(device.index, motors);
    }
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, input_kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & input_kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) | Value::Mouse(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn mouse_state(this: *const Reading, out: *mut MouseState) -> bool {
    match &reading(this).value {
        Value::Mouse(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    mouse_state as *const (),       // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// Host input: native PC `GameInput.dll` when available, else XInput + Win32.
#[cfg(windows)]
mod host {
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, GetKeyState, GetKeyboardLayout, GetKeyboardState, HKL,
        MAPVK_VK_TO_VSC_EX, MapVirtualKeyW, ToUnicodeEx, VK_CAPITAL, VK_CONTROL, VK_LBUTTON,
        VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_MBUTTON, VK_MENU, VK_NUMLOCK, VK_RBUTTON,
        VK_RCONTROL, VK_RMENU, VK_RSHIFT, VK_SHIFT, VK_XBUTTON1, VK_XBUTTON2,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetCursorPos, GetForegroundWindow, GetWindowThreadProcessId,
    };

    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    /// Whether the native PC `GameInput.dll` backend is enabled (default true;
    /// set `VICTORY_GAMEINPUT_NATIVE=0` to force XInput + Win32 fallback).
    pub(super) fn native_enabled() -> bool {
        std::env::var_os("VICTORY_GAMEINPUT_NATIVE").is_none_or(|v| v != "0")
    }

    pub(super) fn create_source(use_native: bool) -> Box<dyn Source> {
        if use_native && let Some(native) = NativeGameInput::try_new() {
            return Box::new(native);
        }
        Box::new(Host::default())
    }

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    /// Translates a virtual key + scan code into `(code_point, is_dead_key)`
    /// without mutating OS dead-key state (`uFlags = 0x4`, Windows 10 1607+).
    pub(super) fn translate_key(
        vk: u8,
        scan_code: u32,
        key_state: &[u8; 256],
        hkl: HKL,
    ) -> (u32, bool) {
        let mut buf = [0u16; 8];
        // SAFETY: valid 256-byte key state and output buffer.
        let ret = unsafe {
            ToUnicodeEx(
                u32::from(vk),
                scan_code & 0xFF,
                key_state.as_ptr(),
                buf.as_mut_ptr(),
                buf.len() as i32,
                0x4,
                hkl,
            )
        };
        if ret == -1 {
            let cp = char::decode_utf16(buf.iter().take_while(|&&u| u != 0).copied())
                .next()
                .and_then(Result::ok)
                .map_or(0, u32::from);
            (cp, true)
        } else if ret > 0 {
            let len = (ret as usize).min(buf.len());
            let cp = char::decode_utf16(buf[..len].iter().copied())
                .next()
                .and_then(Result::ok)
                .map_or(0, u32::from);
            (cp, false)
        } else {
            (0, false)
        }
    }

    fn build_keyboard_state(held_vks: &[u8]) -> [u8; 256] {
        let mut state = [0u8; 256];
        // SAFETY: 256-byte output buffer; poll threads have no message queue
        // so we overlay held keys and global toggle bits below.
        unsafe {
            let _ = GetKeyboardState(state.as_mut_ptr());
            if GetKeyState(i32::from(VK_CAPITAL)) & 1 != 0 {
                state[usize::from(VK_CAPITAL)] |= 0x01;
            }
            if GetKeyState(i32::from(VK_NUMLOCK)) & 1 != 0 {
                state[usize::from(VK_NUMLOCK)] |= 0x01;
            }
        }
        for &vk in held_vks {
            state[usize::from(vk)] |= 0x80;
            match u16::from(vk) {
                VK_LSHIFT | VK_RSHIFT => state[usize::from(VK_SHIFT)] |= 0x80,
                VK_LCONTROL | VK_RCONTROL => state[usize::from(VK_CONTROL)] |= 0x80,
                VK_LMENU | VK_RMENU => state[usize::from(VK_MENU)] |= 0x80,
                _ => {}
            }
        }
        state
    }

    fn push_win32_keys(vks: &[u8], fg_tid: Option<u32>, out: &mut Vec<KeyState>) {
        if vks.is_empty() {
            return;
        }
        let kbd_state = build_keyboard_state(vks);
        // SAFETY: valid thread id or 0 for current thread.
        let hkl = unsafe { GetKeyboardLayout(fg_tid.unwrap_or(0)) };
        for &vk in vks {
            // SAFETY: plain query.
            let scan_code = unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) };
            let (code_point, dead_key) = translate_key(vk, scan_code, &kbd_state, hkl);
            out.push(KeyState {
                scan_code,
                code_point,
                virtual_key: vk,
                dead_key,
            });
        }
    }

    fn apply_scripted_buttons(pad: &mut PadState, buf: &mut Vec<u8>) {
        scripted_into(buf);
        if !buf.is_empty() {
            *pad = super::merged(*pad, super::fallback_vks(buf));
        }
    }

    // -------------------------------------------------------------------------
    // Native PC `GameInput.dll` COM backend (System32, SDK 10.0.26100.0 ABI)
    // -------------------------------------------------------------------------

    type NativeGameInputCreate = unsafe extern "system" fn(*mut *mut c_void) -> HRESULT;
    type NativeDeviceCallback =
        unsafe extern "system" fn(u64, *mut c_void, *mut c_void, u64, u32, u32);

    /// Raw `GameInputKeyState` using `u8` for C `bool` to avoid invalid-bool UB.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct RawKeyState {
        scan_code: u32,
        code_point: u32,
        virtual_key: u8,
        dead_key: u8,
        _pad: [u8; 2],
    }

    const _: () = assert!(size_of::<RawKeyState>() == size_of::<KeyState>());

    /// RAII guard for native COM pointers returned by `GameInput.dll`.
    struct ComPtr(*mut c_void);

    impl ComPtr {
        fn as_ptr(&self) -> *mut c_void {
            self.0
        }
    }

    impl Drop for ComPtr {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: owned COM reference.
                unsafe { com_release(self.0) };
            }
        }
    }

    unsafe fn vtable_fn<F: Copy>(obj: *mut c_void, slot: usize) -> F {
        // SAFETY: COM object begins with a vtable pointer containing `slot`.
        let vtable = unsafe { *(obj as *const *const *const ()) };
        let f = unsafe { *vtable.add(slot) };
        unsafe { std::mem::transmute_copy(&f) }
    }

    unsafe fn com_add_ref(obj: *mut c_void) -> u32 {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = unsafe { vtable_fn(obj, 1) };
        unsafe { f(obj) }
    }

    unsafe fn com_release(obj: *mut c_void) -> u32 {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = unsafe { vtable_fn(obj, 2) };
        unsafe { f(obj) }
    }

    unsafe fn gi_get_current_reading(
        gi: *mut c_void,
        input_kind: u32,
        device: *mut c_void,
    ) -> Result<ComPtr, HRESULT> {
        type Fn = unsafe extern "system" fn(
            *mut c_void,
            u32,
            *mut c_void,
            *mut *mut c_void,
        ) -> HRESULT;
        let f: Fn = unsafe { vtable_fn(gi, 4) };
        let mut reading = std::ptr::null_mut();
        let hr = unsafe { f(gi, input_kind, device, &mut reading) };
        if hr == S_OK && !reading.is_null() {
            Ok(ComPtr(reading))
        } else {
            Err(hr)
        }
    }

    unsafe fn gi_register_device_callback(
        gi: *mut c_void,
        input_kind: u32,
        status_filter: u32,
        enumeration_kind: u32,
        context: *mut c_void,
        callback: NativeDeviceCallback,
    ) -> Result<u64, HRESULT> {
        type Fn = unsafe extern "system" fn(
            *mut c_void,
            *mut c_void,
            u32,
            u32,
            u32,
            *mut c_void,
            NativeDeviceCallback,
            *mut u64,
        ) -> HRESULT;
        let f: Fn = unsafe { vtable_fn(gi, 9) };
        let mut token = 0u64;
        let hr = unsafe {
            f(
                gi,
                std::ptr::null_mut(),
                input_kind,
                status_filter,
                enumeration_kind,
                context,
                callback,
                &mut token,
            )
        };
        if hr == S_OK { Ok(token) } else { Err(hr) }
    }

    unsafe fn gi_unregister_callback(gi: *mut c_void, token: u64, timeout_us: u64) -> bool {
        type Fn = unsafe extern "system" fn(*mut c_void, u64, u64) -> bool;
        let f: Fn = unsafe { vtable_fn(gi, 13) };
        unsafe { f(gi, token, timeout_us) }
    }

    unsafe fn reading_get_key_count(reading: *mut c_void) -> u32 {
        type Fn = unsafe extern "system" fn(*mut c_void) -> u32;
        let f: Fn = unsafe { vtable_fn(reading, 14) };
        unsafe { f(reading) }
    }

    unsafe fn reading_get_key_state(
        reading: *mut c_void,
        count: u32,
        out: *mut RawKeyState,
    ) -> u32 {
        type Fn = unsafe extern "system" fn(*mut c_void, u32, *mut RawKeyState) -> u32;
        let f: Fn = unsafe { vtable_fn(reading, 15) };
        unsafe { f(reading, count, out) }
    }

    unsafe fn reading_get_mouse_state(reading: *mut c_void, out: *mut MouseState) -> bool {
        type Fn = unsafe extern "system" fn(*mut c_void, *mut MouseState) -> bool;
        let f: Fn = unsafe { vtable_fn(reading, 16) };
        unsafe { f(reading, out) }
    }

    unsafe fn reading_get_gamepad_state(reading: *mut c_void, out: *mut PadState) -> bool {
        type Fn = unsafe extern "system" fn(*mut c_void, *mut PadState) -> bool;
        let f: Fn = unsafe { vtable_fn(reading, 22) };
        unsafe { f(reading, out) }
    }

    unsafe fn device_get_status(device: *mut c_void) -> u32 {
        type Fn = unsafe extern "system" fn(*mut c_void) -> u32;
        let f: Fn = unsafe { vtable_fn(device, 4) };
        unsafe { f(device) }
    }

    unsafe fn device_set_rumble(device: *mut c_void, params: *const [f32; 4]) {
        type Fn = unsafe extern "system" fn(*mut c_void, *const [f32; 4]);
        let f: Fn = unsafe { vtable_fn(device, 10) };
        unsafe { f(device, params) };
    }

    fn load_native_gameinput_create() -> Option<NativeGameInputCreate> {
        static FN: OnceLock<Option<NativeGameInputCreate>> = OnceLock::new();
        *FN.get_or_init(|| {
            // Load strictly from System32 so we never resolve a local shim DLL.
            let name: Vec<u16> = "GameInput.dll\0".encode_utf16().collect();
            // SAFETY: null-terminated UTF-16 DLL name; handle stays loaded for process lifetime.
            let hmod = unsafe {
                LoadLibraryExW(
                    name.as_ptr(),
                    std::ptr::null_mut(),
                    LOAD_LIBRARY_SEARCH_SYSTEM32,
                )
            };
            if hmod.is_null() {
                tracing::debug!("System32 GameInput.dll not available; using XInput fallback");
                return None;
            }
            // SAFETY: valid module handle and null-terminated ASCII export name.
            let proc = unsafe { GetProcAddress(hmod, c"GameInputCreate".as_ptr().cast()) }?;
            // SAFETY: GameInputCreate export signature matches `NativeGameInputCreate`.
            Some(unsafe {
                std::mem::transmute::<unsafe extern "system" fn() -> isize, NativeGameInputCreate>(
                    proc,
                )
            })
        })
    }

    struct NativePadSlot {
        device: *mut c_void,
        last_state: PadState,
    }

    struct NativePads {
        slots: Mutex<[Option<NativePadSlot>; PADS]>,
    }

    unsafe extern "system" fn on_native_pad_device(
        _token: u64,
        context: *mut c_void,
        device: *mut c_void,
        _timestamp: u64,
        current_status: u32,
        _previous_status: u32,
    ) {
        if context.is_null() || device.is_null() {
            return;
        }
        // SAFETY: `context` points to `NativePads`, kept alive until after
        // `UnregisterCallback` returns in `NativeGameInput::drop`.
        let pads = unsafe { &*(context as *const NativePads) };
        let mut slots = lock(&pads.slots);
        let connected = current_status & 0x1 != 0;
        if connected {
            if slots
                .iter()
                .any(|s| s.as_ref().is_some_and(|p| p.device == device))
            {
                return;
            }
            if let Some(empty) = slots.iter_mut().find(|s| s.is_none()) {
                // SAFETY: `device` is a valid `IGameInputDevice` for the duration of this callback.
                unsafe { com_add_ref(device) };
                *empty = Some(NativePadSlot {
                    device,
                    last_state: PadState::default(),
                });
            }
        } else if let Some(slot) = slots
            .iter_mut()
            .find(|s| s.as_ref().is_some_and(|p| p.device == device))
            && let Some(removed) = slot.take()
        {
            // SAFETY: releasing the ref added on connect.
            unsafe { com_release(removed.device) };
        }
    }

    pub(super) struct NativeGameInput {
        gi: ComPtr,
        cb_token: u64,
        /// Heap-allocated context for `on_native_pad_device`; freed in `Drop`
        /// only after `UnregisterCallback` completes.
        pads: *mut NativePads,
        last_mouse: MouseState,
        vk_buf: Vec<u8>,
    }

    // SAFETY: `IGameInput` and `IGameInputDevice` are free-threaded COM objects,
    // and `NativeGameInput` is accessed through `Hub`'s `poller` mutex.
    unsafe impl Send for NativeGameInput {}

    impl NativeGameInput {
        pub(super) fn try_new() -> Option<NativeGameInput> {
            let create = load_native_gameinput_create()?;
            let mut raw = std::ptr::null_mut();
            // SAFETY: valid out-pointer to receive `IGameInput*`.
            let hr = unsafe { create(&mut raw) };
            if hr != S_OK || raw.is_null() {
                tracing::warn!(
                    hr = format_args!("{hr:#010x}"),
                    "native GameInputCreate failed; falling back to XInput"
                );
                return None;
            }
            let gi = ComPtr(raw);
            let pads = Box::into_raw(Box::new(NativePads {
                slots: Mutex::new(Default::default()),
            }));
            // Blocking enumeration (2) populates already-connected pads synchronously.
            // SAFETY: `gi` is valid and `pads` remains valid until after `UnregisterCallback`.
            let cb_token = match unsafe {
                gi_register_device_callback(
                    gi.as_ptr(),
                    GAMEPAD_KIND,
                    0x1, // GameInputDeviceConnected
                    2,   // GameInputBlockingEnumeration
                    pads.cast(),
                    on_native_pad_device,
                )
            } {
                Ok(tok) => tok,
                Err(hr) => {
                    tracing::warn!(
                        hr = format_args!("{hr:#010x}"),
                        "native GameInput RegisterDeviceCallback failed; falling back to XInput"
                    );
                    // SAFETY: registration failed, no callback can run.
                    drop(unsafe { Box::from_raw(pads) });
                    return None;
                }
            };
            tracing::info!("native PC GameInput.dll backend active (4-motor impulse triggers)");
            Some(NativeGameInput {
                gi,
                cb_token,
                pads,
                last_mouse: MouseState::default(),
                vk_buf: Vec::new(),
            })
        }

        fn pads(&self) -> &NativePads {
            // SAFETY: allocated in `try_new` and freed only in `Drop`.
            unsafe { &*self.pads }
        }
    }

    impl Drop for NativeGameInput {
        fn drop(&mut self) {
            let unregistered = if self.cb_token != 0 {
                // Wait for any in-flight device callback before releasing devices/context.
                // SAFETY: `self.gi` and `self.cb_token` are valid.
                unsafe { gi_unregister_callback(self.gi.as_ptr(), self.cb_token, 5_000_000) }
            } else {
                true
            };
            if !unregistered {
                // Keep `self.pads` leaked if the callback did not drain in time so an
                // in-flight callback can never dereference freed memory.
                tracing::warn!("GameInput: device callback did not finish unregistering before drop");
                return;
            }
            // SAFETY: callback is unregistered, so no concurrent access to `self.pads`.
            let pads = unsafe { Box::from_raw(self.pads) };
            let mut slots = lock(&pads.slots);
            for slot in slots.iter_mut() {
                if let Some(removed) = slot.take() {
                    // SAFETY: releasing the ref added in `on_native_pad_device`.
                    unsafe { com_release(removed.device) };
                }
            }
        }
    }

    impl Source for NativeGameInput {
        fn rumble_motors(&self) -> u32 {
            RUMBLE_MOTORS_ALL
        }

        fn pad(&mut self, slot: usize) -> Option<PadState> {
            let (device, last_state) = {
                let slots = lock(&self.pads().slots);
                let entry = slots[slot].as_ref()?;
                // AddRef under lock, then drop lock before calling into GameInput.dll
                // to avoid AB-BA deadlock with `on_native_pad_device`.
                // SAFETY: `entry.device` is live while `slots` is held.
                unsafe { com_add_ref(entry.device) };
                (entry.device, entry.last_state)
            };
            // SAFETY: we own a COM ref to `device` and `self.gi` is live.
            let reading_res = unsafe { gi_get_current_reading(self.gi.as_ptr(), GAMEPAD_KIND, device) };
            let mut updated_state = None;
            let mut pad = match reading_res {
                Ok(reading) => {
                    let mut state = PadState::default();
                    if unsafe { reading_get_gamepad_state(reading.as_ptr(), &mut state) } {
                        updated_state = Some(state);
                        state
                    } else {
                        last_state
                    }
                }
                Err(READING_NOT_FOUND) => {
                    if unsafe { device_get_status(device) } & 0x1 != 0 {
                        last_state
                    } else {
                        // SAFETY: releasing our temporary COM ref.
                        unsafe { com_release(device) };
                        return None;
                    }
                }
                Err(_) => {
                    // SAFETY: releasing our temporary COM ref.
                    unsafe { com_release(device) };
                    return None;
                }
            };
            if let Some(state) = updated_state {
                let mut slots = lock(&self.pads().slots);
                if let Some(entry) = slots[slot].as_mut()
                    && entry.device == device
                {
                    entry.last_state = state;
                }
            }
            // SAFETY: releasing our temporary COM ref.
            unsafe { com_release(device) };
            if slot == 0 {
                apply_scripted_buttons(&mut pad, &mut self.vk_buf);
            }
            Some(pad)
        }

        fn keys(&mut self, out: &mut Vec<KeyState>) {
            out.clear();
            let fg_tid = foreground_thread();
            if fg_tid.is_some()
                && let Ok(reading) = unsafe {
                    gi_get_current_reading(self.gi.as_ptr(), KEYBOARD_KIND, std::ptr::null_mut())
                }
            {
                let count = unsafe { reading_get_key_count(reading.as_ptr()) } as usize;
                if count > 0 {
                    let mut raw = [RawKeyState::default(); 64];
                    let cap = count.min(raw.len());
                    let n = unsafe {
                        reading_get_key_state(reading.as_ptr(), cap as u32, raw.as_mut_ptr())
                    } as usize;
                    for k in &raw[..n.min(cap)] {
                        out.push(KeyState {
                            scan_code: k.scan_code,
                            code_point: k.code_point,
                            virtual_key: k.virtual_key,
                            dead_key: k.dead_key != 0,
                        });
                    }
                }
            }
            scripted_into(&mut self.vk_buf);
            if !self.vk_buf.is_empty() {
                self.vk_buf
                    .retain(|vk| !out.iter().any(|k| k.virtual_key == *vk));
                push_win32_keys(&self.vk_buf, fg_tid, out);
            }
        }

        fn mouse(&mut self) -> Option<MouseState> {
            if foreground_thread().is_some() {
                if let Ok(reading) = unsafe {
                    gi_get_current_reading(self.gi.as_ptr(), MOUSE_KIND, std::ptr::null_mut())
                } {
                    let mut state = MouseState::default();
                    if unsafe { reading_get_mouse_state(reading.as_ptr(), &mut state) } {
                        self.last_mouse = state;
                    }
                }
            } else {
                self.last_mouse.buttons = 0;
            }
            Some(self.last_mouse)
        }

        fn rumble(&mut self, slot: usize, motors: [f32; 4]) {
            let device = {
                let slots = lock(&self.pads().slots);
                let Some(entry) = slots[slot].as_ref() else {
                    return;
                };
                // SAFETY: `entry.device` is live while `slots` is held.
                unsafe { com_add_ref(entry.device) };
                entry.device
            };
            let params = if motors == [0.0; 4] {
                std::ptr::null()
            } else {
                &motors as *const [f32; 4]
            };
            // SAFETY: we own a COM ref to `device`; `params` is null or valid 4 floats.
            unsafe {
                device_set_rumble(device, params);
                com_release(device);
            }
        }
    }

    // -------------------------------------------------------------------------
    // XInput + Win32 keyboard/mouse fallback backend
    // -------------------------------------------------------------------------

    #[derive(Default)]
    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        last_mouse: MouseState,
        vk_buf: Vec<u8>,
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            let mut pad = convert(&state.Gamepad);
            // The scripted keys (`VICTORY_KEYS`) also press the first
            // pad's buttons, by the stand-in pad's table: with a real pad
            // connected the stand-in is not there, and a title that listens
            // to the pad (the title's start screen) saw nothing.
            if slot == 0 {
                apply_scripted_buttons(&mut pad, &mut self.vk_buf);
            }
            Some(pad)
        }

        fn keys(&mut self, out: &mut Vec<KeyState>) {
            out.clear();
            scripted_into(&mut self.vk_buf);
            let fg_tid = foreground_thread();
            if fg_tid.is_some() {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                for vk in 0x08..=0xFE_u8 {
                    if (0x10..=0x12).contains(&vk) || self.vk_buf.contains(&vk) {
                        continue;
                    }
                    // SAFETY: plain query.
                    if unsafe { GetAsyncKeyState(i32::from(vk)) } < 0 {
                        self.vk_buf.push(vk);
                    }
                }
            }
            if !self.vk_buf.is_empty() {
                push_win32_keys(&self.vk_buf, fg_tid, out);
            }
        }

        fn mouse(&mut self) -> Option<MouseState> {
            if foreground_thread().is_some() {
                const MOUSE_BUTTONS: [(i32, u32); 5] = [
                    (VK_LBUTTON as i32, MOUSE_LEFT),
                    (VK_RBUTTON as i32, MOUSE_RIGHT),
                    (VK_MBUTTON as i32, MOUSE_MIDDLE),
                    (VK_XBUTTON1 as i32, MOUSE_BUTTON4),
                    (VK_XBUTTON2 as i32, MOUSE_BUTTON5),
                ];
                let mut buttons = 0;
                for (vk, bit) in MOUSE_BUTTONS {
                    // SAFETY: plain query.
                    if unsafe { GetAsyncKeyState(vk) } < 0 {
                        buttons |= bit;
                    }
                }
                let mut pt = POINT::default();
                // SAFETY: valid output pointer.
                if unsafe { GetCursorPos(&mut pt) } != 0 {
                    self.last_mouse.position_x = i64::from(pt.x);
                    self.last_mouse.position_y = i64::from(pt.y);
                }
                self.last_mouse.buttons = buttons;
            } else {
                self.last_mouse.buttons = 0;
            }
            Some(self.last_mouse)
        }

        fn rumble(&mut self, slot: usize, motors: [f32; 4]) {
            let speed = |level: f32| (clamp_motor(level) * f32::from(u16::MAX)) as u16;
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: speed(motors[0]),
                wRightMotorSpeed: speed(motors[1]),
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts (`<from>-<to>:<key>`: all that
    /// time), with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the stand-in pad's table maps them to buttons).
    fn scripted_into(out: &mut Vec<u8>) {
        out.clear();
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                let time = |t: &str| Duration::try_from_secs_f64(t.trim().parse().ok()?).ok();
                let (from, to) = match at.split_once('-') {
                    Some((from, to)) => (time(from)?, time(to)?),
                    None => (time(at)?, time(at)? + Duration::from_millis(200)),
                };
                Some((from, to, vk))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        if keys.is_empty() {
            return;
        }
        let now = start.elapsed();
        for &(from, to, vk) in keys {
            if (from..to).contains(&now) {
                out.push(vk);
            }
        }
    }

    /// Returns the foreground window's thread ID if it belongs to our process.
    fn foreground_thread() -> Option<u32> {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            let tid = GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            (process == GetCurrentProcessId()).then_some(tid)
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }

    #[test]
    fn scripted_fallback_vks_drive_sticks_triggers_and_buttons() {
        let physical = PadState {
            buttons: B,
            left_x: 0.25,
            ..Default::default()
        };
        let scripted = super::fallback_vks(&[0x0D, b'W', b'L', b'Z', b'C']);
        let combined = super::merged(physical, scripted);
        assert_eq!(combined.buttons, A | B);
        assert_eq!(combined.left_x, 0.25);
        assert_eq!(combined.left_y, 1.0);
        assert_eq!(combined.right_x, 1.0);
        assert_eq!(combined.left_trigger, 1.0);
        assert_eq!(combined.right_trigger, 1.0);
    }

    #[test]
    fn win32_translate_key_maps_unicode_and_modifiers() {
        // SAFETY: thread 0 returns the current thread's active keyboard layout.
        let hkl = unsafe { GetKeyboardLayout(0) };
        let scan_a = unsafe { MapVirtualKeyW(u32::from(b'A'), MAPVK_VK_TO_VSC_EX) };
        let plain = [0u8; 256];
        let (cp, dead) = translate_key(b'A', scan_a, &plain, hkl);
        assert_eq!(cp, u32::from(b'a'));
        assert!(!dead);

        let mut shifted = [0u8; 256];
        shifted[usize::from(VK_SHIFT)] = 0x80;
        let (cp_shift, dead_shift) = translate_key(b'A', scan_a, &shifted, hkl);
        assert_eq!(cp_shift, u32::from(b'A'));
        assert!(!dead_shift);

        // Non-character function key produces code point 0.
        let scan_f1 = unsafe { MapVirtualKeyW(0x70, MAPVK_VK_TO_VSC_EX) };
        assert_eq!(translate_key(0x70, scan_f1, &plain, hkl), (0, false));
    }

    #[test]
    fn native_gameinput_loads_polls_and_fallback_works() {
        let mut native = NativeGameInput::try_new()
            .expect("System32 GameInput.dll should initialize on this Windows PC");
        assert_eq!(native.rumble_motors(), RUMBLE_MOTORS_ALL);

        let mut keys = Vec::new();
        for slot in 0..PADS {
            let _ = native.pad(slot);
        }
        native.keys(&mut keys);
        let mouse = native.mouse();
        assert!(mouse.is_some());
        native.rumble(0, [0.0; 4]);
        drop(native);

        // Forced fallback uses XInput + Win32 with 2-motor rumble advertisement.
        let mut fallback_src = create_source(false);
        assert_eq!(fallback_src.rumble_motors(), RUMBLE_MOTORS_XINPUT);
        fallback_src.keys(&mut keys);
        assert!(fallback_src.mouse().is_some());
        fallback_src.rumble(0, [0.0; 4]);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        mouse: Option<MouseState>,
        rumble: Vec<(usize, [f32; 4])>,
        rumble_motors: u32,
    }

    impl Default for FakeHost {
        fn default() -> Self {
            Self {
                pads: Default::default(),
                keys: Vec::new(),
                mouse: None,
                rumble: Vec::new(),
                rumble_motors: RUMBLE_MOTORS_XINPUT,
            }
        }
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn rumble_motors(&self) -> u32 {
            self.0.lock().unwrap().rumble_motors
        }
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self, out: &mut Vec<KeyState>) {
            out.clear();
            out.extend_from_slice(&self.0.lock().unwrap().keys);
        }
        fn mouse(&mut self) -> Option<MouseState> {
            self.0.lock().unwrap().mouse
        }
        fn rumble(&mut self, slot: usize, motors: [f32; 4]) {
            self.0.lock().unwrap().rumble.push((slot, motors));
        }
    }

    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with(false)
    }

    fn fake_with(one_pad: bool) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with_motors(one_pad, RUMBLE_MOTORS_XINPUT)
    }

    fn fake_with_motors(one_pad: bool, motors: u32) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost {
            rumble_motors: motors,
            ..Default::default()
        }));
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))), one_pad);
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, input_kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            input_kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, input_kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            input_kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk_kind(
        hub: &Hub,
        offset: usize,
        from: *const Reading,
        input_kind: u32,
    ) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, input_kind, device, &mut out);
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        walk_kind(hub, offset, from, GAMEPAD_KIND)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn mouse(reading: *const Reading) -> MouseState {
        let mut state = MouseState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut MouseState) -> bool, _>(reading, 0x80);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn a_host_pad_feeds_the_one_pad_which_never_goes() {
        let (hub, host) = fake_with(true);
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);
        let pad = |hub: &Hub| match &hub.slots()[FALLBACK].history.back().unwrap().value {
            Value::Pad(p) => *p,
            Value::Keys(_) | Value::Mouse(_) => unreachable!(),
        };
        host.lock().unwrap().keys = vec![key(b'D')];
        hub.poll();
        assert_eq!(pad(hub).left_x, 1.0);
        // A pad is switched on, in any slot: its input arrives, beside the
        // keyboard's, and no device comes or goes.
        host.lock().unwrap().pads[2] = Some(PadState {
            buttons: A,
            left_y: -0.5,
            ..Default::default()
        });
        hub.poll();
        let p = pad(hub);
        assert_eq!((p.buttons, p.left_x, p.left_y), (A, 1.0, -0.5));
        let device: *const Device = &hub.devices[FALLBACK];
        set_rumble(device, &[1.0, 0.0, 0.25, 0.75]);
        assert_eq!(host.lock().unwrap().rumble, [(2, [1.0, 0.0, 0.25, 0.75])]);
        // And off again.
        host.lock().unwrap().pads[2] = None;
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(pad(hub), PadState::default());
        assert_eq!(log.lock().unwrap().len(), 1);
        assert!(!hub.slots()[2].connected);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(
                &hub.object,
                std::ptr::null(),
                GAMEPAD_KIND,
                1,
                0,
                context,
                Some(slow),
                &mut token
            ),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState::default());
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(
            !unregister(&hub.object, token, 5000),
            "times out while it runs"
        );
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 0.25, 0.75]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(
            host.lock().unwrap().rumble,
            [(2, [0.5, 1.0, 0.25, 0.75]), (2, [0.0; 4])]
        );
    }

    #[test]
    fn impulse_trigger_rumble_forwards_four_motors_and_advertises_capabilities() {
        // Native 4-motor hub in one_pad = false mode.
        let (hub_multi, host_multi) = fake_with_motors(false, RUMBLE_MOTORS_ALL);
        assert_eq!(hub_multi.devices[0].info.rumble_motors, RUMBLE_MOTORS_ALL);
        assert_eq!(hub_multi.devices[KEYBOARD].info.rumble_motors, 0);
        assert_eq!(hub_multi.devices[MOUSE].info.rumble_motors, 0);

        host_multi.lock().unwrap().pads[1] = Some(PadState::default());
        hub_multi.poll();
        set_rumble(&hub_multi.devices[1], &[0.2, 0.4, 0.6, 0.8]);
        set_rumble(&hub_multi.devices[1], &[1.5, -0.5, f32::NAN, 0.3]);
        assert_eq!(
            host_multi.lock().unwrap().rumble,
            [(1, [0.2, 0.4, 0.6, 0.8]), (1, [1.0, 0.0, 0.0, 0.3])]
        );

        // Native 4-motor hub in one_pad = true mode forwards to all connected pads.
        let (hub_one, host_one) = fake_with_motors(true, RUMBLE_MOTORS_ALL);
        assert_eq!(
            hub_one.devices[FALLBACK].info.rumble_motors,
            RUMBLE_MOTORS_ALL
        );
        host_one.lock().unwrap().pads[0] = Some(PadState::default());
        host_one.lock().unwrap().pads[3] = Some(PadState::default());
        hub_one.poll();
        set_rumble(&hub_one.devices[FALLBACK], &[0.1, 0.2, 0.9, 1.0]);
        assert_eq!(
            host_one.lock().unwrap().rumble,
            [(0, [0.1, 0.2, 0.9, 1.0]), (3, [0.1, 0.2, 0.9, 1.0])]
        );
    }

    #[test]
    fn mouse_connects_reads_walks_history_and_disconnects() {
        assert_eq!(size_of::<MouseState>(), 40);
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, MOUSE_KIND, 2, &log);
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(current(hub, MOUSE_KIND, MOUSE), Err(DEVICE_DISCONNECTED));

        let m1 = MouseState {
            buttons: MOUSE_LEFT,
            position_x: 120,
            position_y: -45,
            wheel_x: 0,
            wheel_y: 120,
        };
        host.lock().unwrap().mouse = Some(m1);
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(MOUSE, STATUS_CONNECTED, 0)]);

        let r1 = current(hub, MOUSE_KIND, MOUSE).unwrap();
        assert_eq!(mouse(r1), m1);
        // Wildcard device lookup also finds the mouse.
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        assert_eq!(
            slot::<Current, _>(&hub.object, 0x20)(&hub.object, MOUSE_KIND, std::ptr::null(), &mut out),
            S_OK
        );
        assert_eq!(out.cast::<Reading>(), r1);
        release(out.cast());

        // Mouse reading is not a gamepad reading.
        let mut dummy_pad = PadState::default();
        assert!(!slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(r1, 0xb0)(
            r1,
            &mut dummy_pad
        ));

        let m2 = MouseState {
            buttons: MOUSE_RIGHT | MOUSE_BUTTON4,
            position_x: 300,
            position_y: 500,
            wheel_x: -1,
            wheel_y: 240,
        };
        host.lock().unwrap().mouse = Some(m2);
        hub.poll();
        let r2 = current(hub, MOUSE_KIND, MOUSE).unwrap();
        assert_eq!(mouse(r2), m2);
        assert_eq!(walk_kind(hub, 0x30, r2, MOUSE_KIND), Ok(r1));
        assert_eq!(walk_kind(hub, 0x28, r1, MOUSE_KIND), Ok(r2));
        for r in [r1, r1, r2, r2] {
            release(r);
        }

        host.lock().unwrap().mouse = None;
        hub.poll();
        assert_eq!(log.lock().unwrap()[1..], [(MOUSE, 0, STATUS_CONNECTED)]);
        assert_eq!(current(hub, MOUSE_KIND, MOUSE), Err(DEVICE_DISCONNECTED));
    }

    #[test]
    fn reading_callbacks_deliver_filter_stop_and_synchronize_unregister() {
        type RegisterReading = extern "system" fn(
            This,
            *const Device,
            u32,
            f32,
            *mut c_void,
            Option<ReadingCallback>,
            *mut u64,
        ) -> HRESULT;
        type ReadingLog = Mutex<Vec<(u64, u32, u64)>>;

        extern "system" fn record_reading(
            token: u64,
            context: *mut c_void,
            r: *const Reading,
            overrun: bool,
        ) {
            assert!(!overrun);
            // SAFETY: test passes `&ReadingLog` and valid `Reading`.
            let log = unsafe { &*(context as *const ReadingLog) };
            let r = reading(r);
            log.lock().unwrap().push((token, r.kind(), r.sequence));
        }

        let (hub, host) = fake();
        let reg = slot::<RegisterReading, _>(&hub.object, 0x40);
        let stop = slot::<extern "system" fn(This, u64), _>(&hub.object, 0x60);
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);

        // Invalid arguments fail cleanly and zero out token.
        let mut tok = 99;
        assert_eq!(
            reg(&hub.object, std::ptr::null(), GAMEPAD_KIND, 0.0, std::ptr::null_mut(), None, &mut tok),
            E_INVALIDARG
        );
        assert_eq!(tok, 0);

        let pad_log = ReadingLog::default();
        let mouse_log = ReadingLog::default();
        let mut pad_tok = 0;
        let mut mouse_tok = 0;
        // Pad callback on slot 0 with analog threshold 0.25.
        assert_eq!(
            reg(
                &hub.object,
                &hub.devices[0],
                GAMEPAD_KIND,
                0.25,
                (&pad_log as *const ReadingLog).cast_mut().cast(),
                Some(record_reading),
                &mut pad_tok,
            ),
            S_OK
        );
        // Mouse callback on any device.
        assert_eq!(
            reg(
                &hub.object,
                std::ptr::null(),
                MOUSE_KIND,
                0.0,
                (&mouse_log as *const ReadingLog).cast_mut().cast(),
                Some(record_reading),
                &mut mouse_tok,
            ),
            S_OK
        );

        // Pad 0 connects: delivered once.
        host.lock().unwrap().pads[0] = Some(PadState {
            left_x: 0.0,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(pad_log.lock().unwrap().len(), 1);
        assert!(mouse_log.lock().unwrap().is_empty());

        // Analog change below threshold (0.1 < 0.25): filtered out.
        host.lock().unwrap().pads[0] = Some(PadState {
            left_x: 0.1,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(pad_log.lock().unwrap().len(), 1);

        // Analog change meeting threshold (0.3 >= 0.25): delivered.
        host.lock().unwrap().pads[0] = Some(PadState {
            left_x: 0.3,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(pad_log.lock().unwrap().len(), 2);

        // Pad 1 change does not trigger slot 0's callback.
        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(pad_log.lock().unwrap().len(), 2);

        // Mouse reading triggers mouse callback only.
        host.lock().unwrap().mouse = Some(MouseState {
            buttons: MOUSE_LEFT,
            position_x: 10,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(mouse_log.lock().unwrap().len(), 1);
        assert_eq!(mouse_log.lock().unwrap()[0].1, MOUSE_KIND);

        // StopCallback halts future delivery.
        stop(&hub.object, mouse_tok);
        host.lock().unwrap().mouse = Some(MouseState {
            buttons: MOUSE_RIGHT,
            position_x: 20,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(mouse_log.lock().unwrap().len(), 1);

        assert!(unregister(&hub.object, pad_tok, 5000));

        // Verify UnregisterCallback waits for an in-flight reading callback.
        static R_INSIDE: AtomicBool = AtomicBool::new(false);
        static R_LEAVE: AtomicBool = AtomicBool::new(false);
        static R_SELF_FAIL: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow_reading(
            token: u64,
            context: *mut c_void,
            _r: *const Reading,
            _overrun: bool,
        ) {
            let unreg = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unreg(context.cast(), token, 0) {
                R_SELF_FAIL.fetch_add(1, Ordering::SeqCst);
            }
            R_INSIDE.store(true, Ordering::SeqCst);
            while !R_LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let mut slow_tok = 0;
        let ctx = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            reg(
                &hub.object,
                std::ptr::null(),
                MOUSE_KIND,
                0.0,
                ctx,
                Some(slow_reading),
                &mut slow_tok,
            ),
            S_OK
        );
        host.lock().unwrap().mouse = Some(MouseState {
            position_x: 99,
            ..Default::default()
        });
        let poller = std::thread::spawn(|| hub.poll());
        while !R_INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!unregister(&hub.object, slow_tok, 5000));
        let releaser = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(40));
            R_LEAVE.store(true, Ordering::SeqCst);
        });
        let t0 = Instant::now();
        assert!(unregister(&hub.object, slow_tok, u64::MAX));
        assert!(t0.elapsed() >= Duration::from_millis(30));
        assert_eq!(R_SELF_FAIL.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        releaser.join().unwrap();
    }
}
