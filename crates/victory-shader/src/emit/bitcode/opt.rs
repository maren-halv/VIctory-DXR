//! Clean-up of a finished module before it is written: the lowering emits
//! one instruction after another as the console code names them, and
//! nothing stands between it and the driver's compiler. Every pass here is
//! exact: the module computes the same values and has the same effects for
//! every input.
//!
//! - [`DEAD`]: instructions without an effect whose result nothing reads
//!   are left out (the lowering computes every dword of every register a
//!   console instruction writes; a guarded write reads the register's old
//!   value whether anything reads the new one or not).
//! - [`FOLD`]: integer arithmetic on constants is done here, and identities
//!   (`x + 0`, `x & -1`, a select between a value and itself, a cast there
//!   and back, a join of one value) give their operand. Float arithmetic is
//!   left alone: `x + 0.0` and `x * 1.0` are not `x` for every `x`.
//! - [`REUSE`]: an instruction without an effect whose operands are those
//!   of an earlier one that every path to it has run gives that one's
//!   result (a descriptor handle made again in each block, an index clamped
//!   again, a register read as a float again).
//! - [`FACTS`]: in the blocks only one side of a branch leads to, the
//!   branch's condition is that side's constant, and so is what the
//!   condition was made of (both terms of an `and` that held, the opposite
//!   comparison). The lowering branches on a lane's condition and then
//!   guards each instruction behind the branch with the same condition.
//! - [`LOADS`]: within a block, a read of the same place as an earlier read
//!   gives that one's result while no store, atomic, barrier or other call
//!   with an effect stands between them (another thread's unsynchronised
//!   store may fall between any two reads or before both: the title's code
//!   cannot tell which).
//!
//! Which passes run is `emit::shader_opt()` (`VICTORY_SHADER_OPT`), part of
//! the translation and so of the shader cache's key.

use std::collections::{HashMap, HashSet};

use super::{
    Attr, Bin, Cast, Const, Func, MdDef, Module, Op, Term, Ty, TypeDef, V, for_each_operand, pred,
    successors,
};

/// Dead-code elimination.
pub const DEAD: u32 = 1;
/// Constants folded, identities removed.
pub const FOLD: u32 = 2;
/// One instruction for one value.
pub const REUSE: u32 = 4;
/// What a branch taken says about its condition.
pub const FACTS: u32 = 8;
/// One read for one place within a block, while nothing is stored.
pub const LOADS: u32 = 16;
/// Not a pass over the module: the lowering of a first-lane read in a
/// wave64 program run as two host waves skips the exchange that asks which
/// half has a lane where every lane is in EXEC (`lower.rs`,
/// `v_readfirstlane_b32`).
pub const FIRST: u32 = 32;
/// Not a pass over the module either: the lowering reads the dwords of one
/// console load (`s_buffer_load_dwordx2` to `x16` from an SRV, a
/// `global_load_dwordx2` to `x4`) by one call for every four of them, not
/// by a call each (`lower.rs`, `raw_dwords`). The same bytes are read.
/// From fxceless/victory-fork c36971b2. On an RX 7900 XTX it is worth nothing:
/// the yard's frame replayed turn by turn, dispatches and draws 18.47 -> 18.45
/// ms (12 loops) and 18.23 -> 18.16 (5 loops): the driver joins the loads.
pub const WIDE: u32 = 64;
/// The passes over the module.
pub const MODULE: u32 = DEAD | FOLD | REUSE | FACTS | LOADS;
/// Every pass.
pub const ALL: u32 = MODULE | FIRST | WIDE;
/// The passes that run when nothing says which: all passes including bit-exact
/// [`WIDE`] loads (`127`; `VICTORY_SHADER_OPT=63` restores the pre-wide default).
pub const DEFAULT: u32 = ALL;

/// What the instructions a module writes use, of the features a shader
/// declares by flag (the validator's "flags must match usage").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Usage {
    /// A wave or quad operation.
    pub wave_ops: bool,
    /// A rasterizer ordered view.
    pub rovs: bool,
    /// A 64-bit integer, as a result or an operand.
    pub int64: bool,
    /// `dx.op.tertiary.i32(50)` (`Msad` / D3D11.1 shader extensions).
    pub msad: bool,
    /// A typed load of several components from a UAV.
    pub typed_uav_loads: bool,
    /// A handle from the view heap.
    pub heap_views: bool,
    /// A handle from the sampler heap.
    pub heap_samplers: bool,
    /// A 64-bit atomic.
    pub atomic64: bool,
    /// Inline ray tracing (`RayQuery` or `RaytracingAccelerationStructure`).
    pub raytracing_tier_1_1: bool,
}

/// What makes two instructions without an effect the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Bin(u8, u8, V, V),
    Cast(u8, Ty, V),
    Cmp(u8, V, V),
    Select(V, V, V),
    Extract(V, u32),
    Call(u32, Vec<V>),
    Gep(Ty, V, Vec<V>),
    Load(V),
}

/// The blocks reached from the entry, each after the blocks that dominate
/// it, and each block's children in the dominator tree.
struct Tree {
    order: Vec<usize>,
    children: Vec<Vec<usize>>,
    /// Per block: the reached blocks that branch to it, once per edge.
    preds: Vec<Vec<usize>>,
    /// Immediate dominator of each block (`usize::MAX` if unreachable).
    idom: Vec<usize>,
}

/// The comparison that holds exactly when `p` does not (floats 0-15,
/// integers 32-41).
fn opposite(p: u8) -> u8 {
    match p {
        0..=15 => 15 - p,
        pred::EQ => pred::NE,
        pred::NE => pred::EQ,
        pred::IUGT => pred::IULE,
        pred::IULE => pred::IUGT,
        pred::IUGE => pred::IULT,
        pred::IULT => pred::IUGE,
        pred::SGT => pred::SLE,
        pred::SLE => pred::SGT,
        pred::SGE => pred::SLT,
        _ => pred::SGE,
    }
}

/// What holds in the block being walked: values that are a known bit there,
/// and comparisons (by their operands) with a known outcome.
#[derive(Default)]
struct Facts {
    bits: HashMap<V, bool>,
    compares: HashMap<(u8, V, V), bool>,
}

/// The facts a block added, to forget when its part of the tree is done.
#[derive(Default)]
struct Learned {
    bits: Vec<V>,
    compares: Vec<(u8, V, V)>,
}

fn mask(bits: u32) -> u64 {
    if bits >= 64 { !0 } else { (1 << bits) - 1 }
}

/// `x` of `bits` bits as a signed number.
fn signed(bits: u32, x: u64) -> i64 {
    if bits < 64 && x >> (bits - 1) & 1 != 0 {
        (x | !mask(bits)) as i64
    } else {
        x as i64
    }
}

/// What `v` stands for after the replacements so far.
fn resolve(repl: &[Option<V>], mut v: V) -> V {
    while let V::Inst(i) = v {
        match repl[i as usize] {
            Some(r) => v = r,
            None => break,
        }
    }
    v
}

/// Replaces every value `op` reads by `f` of it.
fn map_operands(op: &mut Op, mut f: impl FnMut(V) -> V) {
    match op {
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) | Op::Store(a, b) | Op::AtomicRmw(_, a, b) => {
            *a = f(*a);
            *b = f(*b);
        }
        Op::Cast(_, v) | Op::Extract(v, _) | Op::Alloca(_, v) | Op::Load(v) => *v = f(*v),
        Op::Select(a, b, c) | Op::CmpXchg(a, b, c) => {
            *a = f(*a);
            *b = f(*b);
            *c = f(*c);
        }
        Op::Call(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
        Op::Phi(inc) => inc.iter_mut().for_each(|x| x.0 = f(x.0)),
        Op::Gep(_, p, idx) => {
            *p = f(*p);
            idx.iter_mut().for_each(|i| *i = f(*i));
        }
    }
}

/// What a call does besides returning its result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Nothing, and the result depends on the arguments alone for the whole
    /// run of the shader (arithmetic, handles, inputs, constant buffers,
    /// resource sizes).
    Pure,
    /// Nothing, but the result depends on memory a store may change
    /// (buffer and texture reads).
    Reads,
    /// `dx.op.rayQuery_State*`: reads the state of its `RayQuery` handle (`args[1]`).
    /// Unchanged by memory stores; invalidated only by `RayQueryMut` on the same handle.
    RayQueryRead,
    /// `dx.op.rayQuery_*` mutators (`TraceRayInline`, `Proceed`, `Abort`, `Commit*`):
    /// mutates `args[1]` (`rq`), always kept.
    RayQueryMut,
    /// Nothing, but the result depends on which lanes run the instruction
    /// (wave and quad operations).
    Lanes,
    /// Stores, atomics, barriers, outputs, discards, handle allocations: always kept.
    Writes,
}

pub(super) fn effect(f: &Func) -> Effect {
    let name = f.name.strip_prefix("dx.op.").unwrap_or(&f.name);
    match f.attr {
        Some(Attr::ReadNone) => Effect::Pure,
        // A shader cannot write a constant buffer or resize a resource;
        // `lower.rs` only emits `discard` at shader exit so `isHelperLane` is
        // invariant across the entire shader body.
        Some(Attr::ReadOnly)
            if name.starts_with("cbufferLoadLegacy")
                || name.starts_with("getDimensions")
                || (super::super::rdna_fp16_opt_on() && name.starts_with("isHelperLane")) =>
        {
            Effect::Pure
        }
        Some(Attr::ReadOnly) if name.starts_with("rayQuery_State") => Effect::RayQueryRead,
        Some(Attr::ReadOnly) if name.starts_with("wave") => Effect::Lanes,
        Some(Attr::ReadOnly) => Effect::Reads,
        // Declared with the attribute of calls that have effects, and have
        // none: they read the other lanes' values.
        _ if name.starts_with("wave") || name.starts_with("quad") => Effect::Lanes,
        _ if name.starts_with("rayQuery_") => Effect::RayQueryMut,
        _ => Effect::Writes,
    }
}

impl Module {
    /// Runs the clean-up passes named in `passes` ([`ALL`]: every one).
    pub(crate) fn optimise(&mut self, passes: u32) {
        if passes & (FOLD | REUSE | FACTS | LOADS) != 0 {
            // Values merged in one round can make two joins the same in
            // the next (a loop's registers that only go round), and a
            // branch that goes one way leaves blocks with one way in.
            for _ in 0..4 {
                let mut changed = self.merge(passes);
                if passes & FOLD != 0 {
                    changed |= self.prune();
                }
                if passes & LOADS != 0 {
                    changed |= self.licm();
                }
                if !changed {
                    break;
                }
            }
        }
        if passes & DEAD != 0 {
            self.drop_dead();
        }
    }

    /// Per function: what a call of it does.
    pub(super) fn effects(&self) -> Vec<Effect> {
        self.funcs.iter().map(effect).collect()
    }

    /// The blocks reached from the entry.
    pub(super) fn reached(&self) -> Vec<bool> {
        let mut seen = vec![false; self.blocks.len()];
        if self.blocks.is_empty() {
            return seen;
        }
        let mut stack = vec![0usize];
        seen[0] = true;
        while let Some(b) = stack.pop() {
            for &s in successors(self.blocks[b].term).iter() {
                if !seen[s] {
                    seen[s] = true;
                    stack.push(s);
                }
            }
        }
        seen
    }

    fn dominator_tree(&self) -> Tree {
        let n = self.blocks.len();
        let succs = |b: usize| successors(self.blocks[b].term);
        // Reverse postorder, as `write` lays the blocks out.
        let mut seen = vec![false; n];
        let mut post = Vec::new();
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some(top) = stack.last_mut() {
            let (b, k) = *top;
            top.1 += 1;
            match succs(b).get(k) {
                Some(&s) if !seen[s] => {
                    seen[s] = true;
                    stack.push((s, 0));
                }
                Some(_) => {}
                None => {
                    post.push(b);
                    stack.pop();
                }
            }
        }
        post.reverse();
        let order = post;
        let mut number = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            number[b] = k;
        }
        let mut preds = vec![Vec::new(); n];
        for &b in &order {
            for &s in succs(b).iter() {
                preds[s].push(b);
            }
        }
        // Immediate dominators (Cooper, Harvey and Kennedy's iteration).
        let mut idom = vec![usize::MAX; n];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &order[1..] {
                let mut new = usize::MAX;
                for &p in &preds[b] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    if new == usize::MAX {
                        new = p;
                        continue;
                    }
                    let (mut x, mut y) = (p, new);
                    while x != y {
                        while number[x] > number[y] {
                            x = idom[x];
                        }
                        while number[y] > number[x] {
                            y = idom[y];
                        }
                    }
                    new = x;
                }
                if idom[b] != new {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        let mut children = vec![Vec::new(); n];
        for &b in &order[1..] {
            children[idom[b]].push(b);
        }
        Tree {
            order,
            children,
            preds,
            idom,
        }
    }

    /// Notes that `v` (an i1) is `holds` in the block being walked, and what
    /// follows from it for the values `v` was made of.
    fn learn(&self, v: V, holds: bool, facts: &mut Facts, learned: &mut Learned, depth: u32) {
        let V::Inst(i) = v else { return };
        if facts.bits.contains_key(&v) {
            return;
        }
        facts.bits.insert(v, holds);
        learned.bits.push(v);
        if depth == 0 {
            return;
        }
        match self.insts[i as usize].op {
            // Both terms of an `and` that holds hold; neither of an `or`
            // that does not.
            Op::Bin(Bin::And, a, b) if holds => {
                self.learn(a, true, facts, learned, depth - 1);
                self.learn(b, true, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Or, a, b) if !holds => {
                self.learn(a, false, facts, learned, depth - 1);
                self.learn(b, false, facts, learned, depth - 1);
            }
            Op::Bin(Bin::Xor, a, b) => {
                let (x, k) = match (self.int_k(a), self.int_k(b)) {
                    (None, Some((1, k))) => (a, k),
                    (Some((1, k)), None) => (b, k),
                    _ => return,
                };
                self.learn(x, holds != (k != 0), facts, learned, depth - 1);
            }
            // The opposite comparison of the same operands.
            Op::Cmp(p, a, b) => {
                let key = (opposite(p), a, b);
                if let std::collections::hash_map::Entry::Vacant(e) = facts.compares.entry(key) {
                    e.insert(!holds);
                    learned.compares.push(key);
                }
            }
            _ => {}
        }
    }

    fn int_bits(&self, ty: Ty) -> Option<u32> {
        match self.types[ty.0 as usize] {
            TypeDef::Int(bits) => Some(bits),
            _ => None,
        }
    }

    /// An integer constant's width and value (the bits of that width).
    pub(crate) fn int_k(&self, v: V) -> Option<(u32, u64)> {
        let V::Const(i) = v else { return None };
        let (ty, Const::Int(x)) = &self.consts[i as usize] else {
            return None;
        };
        let bits = self.int_bits(*ty)?;
        Some((bits, *x as u64 & mask(bits)))
    }

    /// The bits of a 32-bit float that is a constant, given as one or as a
    /// cast of an integer constant.
    pub(crate) fn float_k(&self, v: V) -> Option<f32> {
        let (v, cast) = match v {
            V::Inst(i) => match self.insts[i as usize].op {
                Op::Cast(Cast::Bitcast, x) => (x, true),
                _ => return None,
            },
            _ => (v, false),
        };
        let V::Const(i) = v else { return None };
        match &self.consts[i as usize].1 {
            Const::Float(bits) if !cast => Some(f32::from_bits(*bits)),
            Const::Int(_) if cast => {
                let (bits, x) = self.int_k(v)?;
                (bits == 32).then_some(f32::from_bits(x as u32))
            }
            _ => None,
        }
    }

    /// For a float that is a number and not an infinity whatever the program
    /// reads: a power of two its magnitude does not pass.
    pub(crate) fn bounded(&self, v: V, depth: u32) -> Option<f32> {
        self.bounded_ext(v, depth, true)
    }

    /// As [`Module::bounded`]; when `extended` is `false`, restricts analysis
    /// to the original pre-N3 rules (`VICTORY_LEGACY_MUL_OPT=0`).
    pub(crate) fn bounded_ext(&self, v: V, depth: u32, extended: bool) -> Option<f32> {
        let mut budget = if extended { 128u32 } else { u32::MAX };
        self.bounded_inner(v, depth, extended, &mut budget)
    }

    fn sign_bit_only(&self, v: V, depth: u32) -> bool {
        if let Some((32, k)) = self.int_k(v) {
            return k & 0x7FFF_FFFF == 0;
        }
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let V::Inst(i) = v else { return false };
        match self.insts[i as usize].op {
            Op::Bin(Bin::And, a, b) => {
                self.int_k(a).is_some_and(|(w, k)| w == 32 && k & 0x7FFF_FFFF == 0)
                    || self.int_k(b).is_some_and(|(w, k)| w == 32 && k & 0x7FFF_FFFF == 0)
            }
            Op::Select(_, a, b) => {
                self.sign_bit_only(a, depth) && self.sign_bit_only(b, depth)
            }
            _ => false,
        }
    }

    /// Peels any `bitcast f32 <-> i32` roundtrips around a float value.
    fn peel_float_bitcasts(&self, mut v: V) -> V {
        for _ in 0..4 {
            let V::Inst(i) = v else { break };
            let Op::Cast(Cast::Bitcast, x) = self.insts[i as usize].op else {
                break;
            };
            let V::Inst(j) = x else { break };
            let Op::Cast(Cast::Bitcast, y) = self.insts[j as usize].op else {
                break;
            };
            if self.type_of(y) != self.type_of(v) {
                break;
            }
            v = y;
        }
        v
    }

    /// Bound on an `i32` value interpreted as IEEE-754 `f32` bits.
    fn bounded_bits(&self, v: V, depth: u32, budget: &mut u32) -> Option<f32> {
        if let Some((32, k)) = self.int_k(v) {
            let f = f32::from_bits(k as u32);
            return f.is_finite().then(|| f.abs().max(1e-30).log2().ceil());
        }
        let V::Inst(i) = v else { return None };
        if self.unit_floats.contains(&i) {
            return Some(0.0);
        }
        *budget = budget.checked_sub(1)?;
        let depth = depth.checked_sub(1)?;
        match self.insts[i as usize].op {
            Op::Cast(Cast::Bitcast, y)
                if matches!(self.types[self.type_of(y).0 as usize], TypeDef::Float) =>
            {
                self.bounded_inner(y, depth, true, budget)
            }
            Op::Select(_, a, b) => {
                Some(self.bounded_bits(a, depth, budget)?.max(self.bounded_bits(b, depth, budget)?))
            }
            Op::Bin(Bin::And, a, b) => {
                let check = |mask_v: V, other: V, budget: &mut u32| -> Option<f32> {
                    let (32, m) = self.int_k(mask_v)? else {
                        return None;
                    };
                    let m = m as u32;
                    if m == 0x7FFF_FFFF {
                        return self.bounded_bits(other, depth, budget);
                    }
                    // If at least one exponent bit is cleared, the exponent can never reach 255.
                    if m & 0x7F80_0000 != 0x7F80_0000 {
                        let max_exp = ((m >> 23) & 0xFF) as i32 - 127 + 1;
                        return Some(max_exp.max(-100) as f32);
                    }
                    None
                };
                check(b, a, budget).or_else(|| check(a, b, budget))
            }
            Op::Bin(Bin::Or | Bin::Xor, a, b) => {
                if self.sign_bit_only(b, 3) {
                    self.bounded_bits(a, depth, budget)
                } else if self.sign_bit_only(a, 3) {
                    self.bounded_bits(b, depth, budget)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn bounded_inner(&self, v: V, depth: u32, extended: bool, budget: &mut u32) -> Option<f32> {
        if let Some(k) = self.float_k(v) {
            return k.is_finite().then(|| k.abs().max(1e-30).log2().ceil());
        }
        let V::Inst(i) = v else { return None };
        if extended && self.unit_floats.contains(&i) {
            return Some(0.0);
        }
        if extended {
            *budget = budget.checked_sub(1)?;
        }
        let depth = depth.checked_sub(1)?;
        // (A power that overflows, or might with rounding, is no bound.)
        let held = |b: f32| (b < 126.0).then_some(b);
        match &self.insts[i as usize].op {
            Op::Cast(Cast::UiToFp, x) => {
                if extended
                    && let V::Inst(j) = *x
                    && let Op::Bin(Bin::And, a, b) = self.insts[j as usize].op
                {
                    let mask = self
                        .int_k(b)
                        .or_else(|| self.int_k(a))
                        .filter(|&(w, _)| w == 32)
                        .map(|(_, k)| k as u32);
                    if let Some(m) = mask {
                        return Some(if m == 0 {
                            -100.0
                        } else {
                            (32 - m.leading_zeros()) as f32
                        });
                    }
                }
                Some(32.0)
            }
            Op::Cast(Cast::SiToFp, _) => Some(32.0),
            Op::Cast(Cast::Bitcast, x) => {
                if extended {
                    self.bounded_bits(*x, depth, budget)
                } else {
                    match *x {
                        V::Inst(j) => match self.insts[j as usize].op {
                            Op::Cast(Cast::Bitcast, y) if self.type_of(y) == self.type_of(v) => {
                                self.bounded_inner(y, depth, extended, budget)
                            }
                            _ => None,
                        },
                        _ => None,
                    }
                }
            }
            Op::Bin(Bin::Mul, a, b) => {
                let ba = self.bounded_inner(*a, depth, extended, budget)?;
                let bb = self.bounded_inner(*b, depth, extended, budget)?;
                held(ba + bb)
            }
            Op::Bin(Bin::Sub, a, b)
                if extended && self.float_k(*a).is_some_and(|k| k == 0.0) =>
            {
                self.bounded_inner(*b, depth, extended, budget)
            }
            Op::Bin(Bin::Add | Bin::Sub, a, b) => {
                let ba = self.bounded_inner(*a, depth, extended, budget)?;
                let bb = self.bounded_inner(*b, depth, extended, budget)?;
                held(ba.max(bb) + 1.0)
            }
            Op::Select(_, a, b) => {
                let ba = self.bounded_inner(*a, depth, extended, budget)?;
                let bb = self.bounded_inner(*b, depth, extended, budget)?;
                Some(ba.max(bb))
            }
            Op::Call(_, args) => {
                let (_, code) = self.int_k(*args.first()?)?;
                match (code, &args[1..]) {
                    // Saturated, a fraction, a cosine or sine (of a number).
                    (7, [_]) => Some(0.0),
                    (6, [x]) => self.bounded_inner(*x, depth, extended, budget),
                    (22 | 12 | 13, [x]) => {
                        self.bounded_inner(*x, depth, extended, budget).map(|_| 0.0)
                    }
                    (26..=29, [x]) => held(self.bounded_inner(*x, depth, extended, budget)? + 1.0),
                    // Of two numbers; or one of them the other way round: the
                    // minimum (36) of a finite bound and a maximum (35) of another.
                    (35 | 36, [a, b]) => {
                        let ba = self.bounded_inner(*a, depth, extended, budget);
                        let bb = self.bounded_inner(*b, depth, extended, budget);
                        match (ba, bb) {
                            (Some(x), Some(y)) => Some(x.max(y)),
                            _ if extended => {
                                let other = if code == 35 { 36 } else { 35 };
                                let (inner, kb) = match (ba, bb) {
                                    (None, Some(y)) => (*a, y),
                                    (Some(x), None) => (*b, x),
                                    _ => return None,
                                };
                                let inner = self.peel_float_bitcasts(inner);
                                let V::Inst(j) = inner else { return None };
                                let Op::Call(_, inner_args) = &self.insts[j as usize].op else {
                                    return None;
                                };
                                let (_, inner_code) = self.int_k(*inner_args.first()?)?;
                                (inner_code == other && inner_args.len() == 3).then_some(())?;
                                let kl = self
                                    .bounded_inner(inner_args[1], depth, extended, budget)
                                    .or_else(|| {
                                        self.bounded_inner(inner_args[2], depth, extended, budget)
                                    })?;
                                Some(kb.max(kl))
                            }
                            _ => {
                                let other = if code == 35 { 36 } else { 35 };
                                let (inner, k) =
                                    if self.float_k(*b).is_some() { (*a, *b) } else { (*b, *a) };
                                let k = self
                                    .float_k(k)
                                    .and_then(|_| self.bounded_inner(k, depth, extended, budget))?;
                                let V::Inst(j) = inner else { return None };
                                let Op::Call(_, inner) = &self.insts[j as usize].op else {
                                    return None;
                                };
                                let (_, code) = self.int_k(*inner.first()?)?;
                                let low = inner[1..].iter().find(|x| self.float_k(**x).is_some())?;
                                (code == other).then_some(())?;
                                Some(k.max(self.bounded_inner(*low, depth, extended, budget)?))
                            }
                        }
                    }
                    (46, [a, b, c]) => {
                        let ba = self.bounded_inner(*a, depth, extended, budget)?;
                        let bb = self.bounded_inner(*b, depth, extended, budget)?;
                        let bc = self.bounded_inner(*c, depth, extended, budget)?;
                        held((ba + bb).max(bc) + 1.0)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Post-folding pass for AMD RDNA `rdna::LEGACY` zero-guards on
    /// `v_mul_legacy_f32` and `v_mac/mad_legacy_f32`. Collapses the 5-op or
    /// 2-op zero-guard to plain `Bin::Mul` / `FMad` whenever extended
    /// [`Module::bounded`] proves no Inf/NaN can meet `0.0`, and narrows a
    /// 5-op guard to a 2-op guard when one operand is now bounded.
    pub(crate) fn elide_legacy_mul_guards(&mut self) -> bool {
        let mut any_changed = false;
        let is_dx_op = |m: &Module, f: u32, prefix: &str, args: &[V], want: u64| -> bool {
            m.funcs[f as usize].name == prefix
                && args.first().is_some_and(|&v| m.int_k(v) == Some((32, want)))
        };
        let fabs_arg = |m: &Module, v: V| -> Option<V> {
            let V::Inst(i) = v else { return None };
            let Op::Call(f, args) = &m.insts[i as usize].op else {
                return None;
            };
            if m.funcs[*f as usize].name == "dx.op.unary.f32"
                && args.len() == 2
                && m.int_k(args[0]) == Some((32, 6))
            {
                Some(args[1])
            } else {
                None
            }
        };
        let fmin_fabs_pair = |m: &Module, v: V| -> Option<(V, V)> {
            let V::Inst(i) = v else { return None };
            let Op::Call(f, args) = &m.insts[i as usize].op else {
                return None;
            };
            if m.funcs[*f as usize].name == "dx.op.binary.f32"
                && args.len() == 3
                && m.int_k(args[0]) == Some((32, 36))
            {
                Some((fabs_arg(m, args[1])?, fabs_arg(m, args[2])?))
            } else {
                None
            }
        };
        for _ in 0..3 {
            let reached = self.reached();
            let mut uses = vec![0u32; self.insts.len()];
            for (b, blk) in self.blocks.iter().enumerate() {
                if !reached[b] {
                    continue;
                }
                for &i in blk.phis.iter().chain(&blk.body) {
                    for_each_operand(&self.insts[i as usize].op, |v| {
                        if let V::Inst(u) = v {
                            uses[u as usize] = uses[u as usize].saturating_add(1);
                        }
                    });
                }
                if let Some(Term::CondBr(c, ..)) = blk.term
                    && let V::Inst(u) = c
                {
                    uses[u as usize] = uses[u as usize].saturating_add(1);
                }
            }
            let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
            let mut narrow: Vec<(u32, V, V)> = Vec::new();
            let mut round_changed = false;
            for (b, blk) in self.blocks.iter().enumerate() {
                if !reached[b] {
                    continue;
                }
                for &i in &blk.body {
                    let Op::Select(z, true_v, false_v) = self.insts[i as usize].op else {
                        continue;
                    };
                    if !matches!(self.types[self.insts[i as usize].ty.0 as usize], TypeDef::Float) {
                        continue;
                    }
                    let V::Inst(z_id) = z else { continue };
                    let Op::Cmp(pred::OEQ, asked, z_zero) = self.insts[z_id as usize].op else {
                        continue;
                    };
                    if self.float_k(z_zero) != Some(0.0) {
                        continue;
                    }
                    let V::Inst(f_id) = false_v else { continue };
                    let pair = match &self.insts[f_id as usize].op {
                        // legacy_mul: Select(OEQ(asked, 0.0), 0.0, Mul(a, b))
                        Op::Bin(Bin::Mul, a, b)
                            if self.float_k(true_v).is_some_and(|k| k == 0.0) =>
                        {
                            Some((*a, *b))
                        }
                        // legacy_fmad: Select(OEQ(asked, 0.0), Add(0.0, c), FMad(a, b, c))
                        Op::Call(f, args)
                            if args.len() == 4
                                && is_dx_op(self, *f, "dx.op.tertiary.f32", args, 46) =>
                        {
                            let (a, b, add_c) = (args[1], args[2], args[3]);
                            let addend_ok = match true_v {
                                V::Inst(t_id) => match self.insts[t_id as usize].op {
                                    Op::Bin(Bin::Add, z1, c2) => {
                                        self.float_k(z1).is_some_and(|k| k == 0.0) && c2 == add_c
                                    }
                                    _ => false,
                                },
                                _ => false,
                            };
                            addend_ok.then_some((a, b))
                        }
                        _ => None,
                    };
                    let Some((a, b)) = pair else { continue };
                    let is_five_op = fmin_fabs_pair(self, asked)
                        .is_some_and(|(x, y)| (x == a && y == b) || (x == b && y == a));
                    let is_two_op = asked == a || asked == b;
                    if !is_five_op && !is_two_op {
                        continue;
                    }
                    let fa = self.bounded(a, 20).is_some();
                    let fb = self.bounded(b, 20).is_some();
                    let not_zero =
                        |x: V| self.float_k(x).is_some_and(|k| k != 0.0 && k.is_finite());
                    if a == b || (fa && fb) || not_zero(a) || not_zero(b) {
                        repl[i as usize] = Some(false_v);
                        round_changed = true;
                    } else if is_five_op && (fa ^ fb) && uses[z_id as usize] == 1 {
                        narrow.push((z_id, if fa { a } else { b }, z_zero));
                        round_changed = true;
                    }
                }
            }
            if !round_changed {
                break;
            }
            any_changed = true;
            for (z_id, op, z_zero) in narrow {
                self.insts[z_id as usize].op = Op::Cmp(pred::OEQ, op, z_zero);
            }
            for blk in &mut self.blocks {
                blk.body.retain(|&i| repl[i as usize].is_none());
                if let Some(Term::CondBr(c, t, f)) = blk.term {
                    blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
                }
            }
            let (blocks, insts) = (&self.blocks, &mut self.insts);
            for blk in blocks {
                for &i in blk.phis.iter().chain(&blk.body) {
                    map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
                }
            }
        }
        any_changed
    }

    /// Whether an `i32` value is guaranteed even (bit 0 is 0).
    pub(crate) fn is_even_i32(&self, v: V, depth: u32) -> bool {
        if let Some((_, k)) = self.int_k(v) {
            return k & 1 == 0;
        }
        let Some(depth) = depth.checked_sub(1) else {
            return false;
        };
        let V::Inst(i) = v else { return false };
        match self.insts[i as usize].op {
            Op::Bin(Bin::Shl, _, sh) => self.int_k(sh).is_some_and(|(_, k)| k >= 1),
            Op::Bin(Bin::And, a, b) | Op::Bin(Bin::Mul, a, b) => {
                self.is_even_i32(a, depth) || self.is_even_i32(b, depth)
            }
            Op::Bin(Bin::Add | Bin::Sub | Bin::Or | Bin::Xor, a, b)
            | Op::Select(_, a, b) => self.is_even_i32(a, depth) && self.is_even_i32(b, depth),
            _ => false,
        }
    }

    /// Whether `v` is a 24-bit unsigned `RayQuery` state scalar (`InstanceId`,
    /// `GeometryIndex`, `InstanceContribution`, candidate or committed).
    fn is_rq_u24(&self, v: V) -> bool {
        let V::Inst(id) = v else { return false };
        let Op::Call(f, args) = &self.insts[id as usize].op else {
            return false;
        };
        self.funcs[*f as usize].name == "dx.op.rayQuery_StateScalar.i32"
            && args
                .first()
                .and_then(|&op| self.int_k(op))
                .is_some_and(|(w, op)| w == 32 && matches!(op, 202 | 203 | 208 | 209 | 214 | 215))
    }

    /// Whether `v` is bitwise NOT of `x` (`xor(x, all_ones)` for width `bits`).
    fn is_bitnot_of(&self, v: V, x: V, bits: u32) -> bool {
        let V::Inst(id) = v else { return false };
        let Op::Bin(Bin::Xor, a, b) = self.insts[id as usize].op else {
            return false;
        };
        let all = mask(bits);
        (a == x && self.int_k(b) == Some((bits, all)))
            || (b == x && self.int_k(a) == Some((bits, all)))
    }

    /// The constant `x` of the integer type `ty`, kept sign-extended as
    /// [`Module::i32`] and [`Module::bool`] keep theirs.
    fn k(&mut self, ty: Ty, bits: u32, x: u64) -> V {
        self.int_const(ty, signed(bits, x & mask(bits)))
    }

    /// The value instruction `i` is known to have without it, if any: a
    /// constant or an earlier value. May instead give `i` simpler operands
    /// (`changed` says so).
    fn simplify(&mut self, i: u32, changed: &mut bool) -> Option<V> {
        let ty = self.insts[i as usize].ty;
        match self.insts[i as usize].op {
            Op::Bin(op, a, b) => {
                // Floats: `x + 0.0`, `x * 1.0` and the like are not `x` for
                // every `x` (a negative zero, a signalling NaN).
                let bits = self.int_bits(ty)?;
                let (ka, kb) = (self.int_k(a).map(|k| k.1), self.int_k(b).map(|k| k.1));
                let all = mask(bits);
                if let (Some(x), Some(y)) = (ka, kb) {
                    // A shift by the width or more and a division by zero
                    // have no defined result: left as they are.
                    let amount = u32::try_from(y).ok().filter(|&y| y < bits);
                    let r = match op {
                        Bin::Add => x.wrapping_add(y),
                        Bin::Sub => x.wrapping_sub(y),
                        Bin::Mul => x.wrapping_mul(y),
                        Bin::And => x & y,
                        Bin::Or => x | y,
                        Bin::Xor => x ^ y,
                        Bin::Shl => x << amount?,
                        Bin::LShr => x >> amount?,
                        Bin::AShr => (signed(bits, x) >> amount?) as u64,
                        Bin::UDiv if y != 0 => x / y,
                        Bin::URem if y != 0 => x % y,
                        _ => return None,
                    };
                    return Some(self.k(ty, bits, r));
                }
                match (op, ka, kb) {
                    (
                        Bin::Add | Bin::Sub | Bin::Or | Bin::Xor | Bin::Shl | Bin::LShr | Bin::AShr,
                        _,
                        Some(0),
                    ) => Some(a),
                    (Bin::Add | Bin::Or | Bin::Xor, Some(0), _) => Some(b),
                    (Bin::Mul | Bin::UDiv, _, Some(1)) => Some(a),
                    (Bin::Mul, Some(1), _) => Some(b),
                    (Bin::Mul | Bin::And, _, Some(0)) => Some(b),
                    (Bin::Mul | Bin::And, Some(0), _) => Some(a),
                    (Bin::And, _, Some(m)) if m == all => Some(a),
                    (Bin::And, Some(m), _) if m == all => Some(b),
                    (Bin::Or, _, Some(m)) if m == all => Some(b),
                    (Bin::Or, Some(m), _) if m == all => Some(a),
                    (Bin::And | Bin::Or, _, _) if a == b => Some(a),
                    (Bin::Sub | Bin::Xor, _, _) if a == b => Some(self.k(ty, bits, 0)),
                    // `x & ~x = 0`, `x | ~x = -1`, `(x ^ k) ^ k = x`.
                    (Bin::And, _, _)
                        if self.is_bitnot_of(a, b, bits) || self.is_bitnot_of(b, a, bits) =>
                    {
                        Some(self.k(ty, bits, 0))
                    }
                    (Bin::Or, _, _)
                        if self.is_bitnot_of(a, b, bits) || self.is_bitnot_of(b, a, bits) =>
                    {
                        Some(self.k(ty, bits, all))
                    }
                    (Bin::Xor, None, Some(k)) => {
                        if let V::Inst(x) = a
                            && let Op::Bin(Bin::Xor, x0, x1) = self.insts[x as usize].op
                        {
                            if self.int_k(x1) == Some((bits, k)) {
                                return Some(x0);
                            }
                            if self.int_k(x0) == Some((bits, k)) {
                                return Some(x1);
                            }
                        }
                        None
                    }
                    (Bin::Xor, Some(k), None) => {
                        if let V::Inst(x) = b
                            && let Op::Bin(Bin::Xor, x0, x1) = self.insts[x as usize].op
                        {
                            if self.int_k(x1) == Some((bits, k)) {
                                return Some(x0);
                            }
                            if self.int_k(x0) == Some((bits, k)) {
                                return Some(x1);
                            }
                        }
                        None
                    }
                    // 24-bit unsigned RayQuery scalars: `x & 0xFFFFFF -> x`, `(x << k) >>u k -> x`.
                    (Bin::And, _, Some(m))
                        if bits == 32 && (m & 0x00FF_FFFF) == 0x00FF_FFFF && self.is_rq_u24(a) =>
                    {
                        Some(a)
                    }
                    (Bin::And, Some(m), _)
                        if bits == 32 && (m & 0x00FF_FFFF) == 0x00FF_FFFF && self.is_rq_u24(b) =>
                    {
                        Some(b)
                    }
                    (Bin::LShr, _, Some(k)) if bits == 32 && (1..=8).contains(&k) => {
                        if let V::Inst(sh) = a
                            && let Op::Bin(Bin::Shl, x, sh_k) = self.insts[sh as usize].op
                            && self.int_k(sh_k) == Some((32, k))
                            && self.is_rq_u24(x)
                        {
                            return Some(x);
                        }
                        None
                    }
                    _ => None,
                }
            }
            Op::Cmp(p, a, b) if p <= pred::TRUE => {
                // Two float constants compare one way (`v_mul_legacy` asks
                // whether its constant factor is zero).
                let (V::Const(x), V::Const(y)) = (a, b) else {
                    return None;
                };
                let (Const::Float(x), Const::Float(y)) =
                    (&self.consts[x as usize].1, &self.consts[y as usize].1)
                else {
                    return None;
                };
                let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                let unordered = x.is_nan() || y.is_nan();
                // Bits 0-2 of a predicate: equal, greater, less; bit 3:
                // or unordered.
                let holds = if unordered {
                    p & 8 != 0
                } else {
                    (p & 1 != 0 && x == y) || (p & 2 != 0 && x > y) || (p & 4 != 0 && x < y)
                };
                Some(self.bool(holds))
            }
            Op::Cmp(p, a, b) => {
                if !(pred::EQ..=pred::SLE).contains(&p) {
                    return None;
                }
                let holds = |x: u64, y: u64, bits: u32| {
                    let (sx, sy) = (signed(bits, x), signed(bits, y));
                    match p {
                        pred::EQ => x == y,
                        pred::NE => x != y,
                        pred::IUGT => x > y,
                        pred::IUGE => x >= y,
                        pred::IULT => x < y,
                        pred::IULE => x <= y,
                        pred::SGT => sx > sy,
                        pred::SGE => sx >= sy,
                        pred::SLT => sx < sy,
                        _ => sx <= sy,
                    }
                };
                if let (Some((bits, x)), Some((_, y))) = (self.int_k(a), self.int_k(b)) {
                    return Some(self.bool(holds(x, y, bits)));
                }
                if a == b {
                    return Some(self.bool(holds(0, 0, 32)));
                }
                // A bit widened and tested again is the bit.
                if p == pred::NE
                    && self.int_k(b).is_some_and(|k| k.1 == 0)
                    && let V::Inst(z) = a
                    && let Op::Cast(Cast::ZExt, x) = self.insts[z as usize].op
                    && self.type_of(x) == ty
                {
                    return Some(x);
                }
                None
            }
            Op::Select(c, mut t, mut f) => {
                if let Some((_, c)) = self.int_k(c) {
                    return Some(if c != 0 { t } else { f });
                }
                // Under the same condition an inner select is one of its
                // operands (two guarded writes of one register in a row).
                let mut inner = false;
                while let V::Inst(x) = t
                    && let Op::Select(c2, t2, _) = self.insts[x as usize].op
                    && c2 == c
                {
                    (t, inner) = (t2, true);
                }
                while let V::Inst(x) = f
                    && let Op::Select(c2, _, f2) = self.insts[x as usize].op
                    && c2 == c
                {
                    (f, inner) = (f2, true);
                }
                if t == f {
                    return Some(t);
                }
                if self.int_bits(ty) == Some(1) {
                    let t_eff = if let Some((_, k)) = self.int_k(t) {
                        Some(k != 0)
                    } else if t == c {
                        Some(true)
                    } else if self.is_bitnot_of(t, c, 1) {
                        Some(false)
                    } else {
                        None
                    };
                    let f_eff = if let Some((_, k)) = self.int_k(f) {
                        Some(k != 0)
                    } else if f == c {
                        Some(false)
                    } else if self.is_bitnot_of(f, c, 1) {
                        Some(true)
                    } else {
                        None
                    };
                    match (t_eff, f_eff) {
                        (Some(true), Some(true)) => return Some(self.bool(true)),
                        (Some(false), Some(false)) => return Some(self.bool(false)),
                        (Some(true), Some(false)) => return Some(c),
                        (Some(false), Some(true)) => {
                            if self.is_bitnot_of(t, c, 1) {
                                return Some(t);
                            }
                            if self.is_bitnot_of(f, c, 1) {
                                return Some(f);
                            }
                            if let V::Inst(ci) = c
                                && let Op::Bin(Bin::Xor, x0, x1) = self.insts[ci as usize].op
                            {
                                if self.int_k(x1) == Some((1, 1)) {
                                    return Some(x0);
                                }
                                if self.int_k(x0) == Some((1, 1)) {
                                    return Some(x1);
                                }
                            }
                            let yes = self.bool(true);
                            self.insts[i as usize].op = Op::Bin(Bin::Xor, c, yes);
                            *changed = true;
                            return None;
                        }
                        _ => {}
                    }
                }
                if inner {
                    self.insts[i as usize].op = Op::Select(c, t, f);
                    *changed = true;
                }
                None
            }
            Op::Cast(op, x) => {
                let from = self.type_of(x);
                let through = match x {
                    V::Inst(y) => match self.insts[y as usize].op {
                        Op::Cast(op2, z) => Some((op2, z)),
                        _ => None,
                    },
                    _ => None,
                };
                match op {
                    Cast::Bitcast => {
                        if from == ty {
                            return Some(x);
                        }
                        // There and back: the bits are the same.
                        if let Some((Cast::Bitcast, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        let V::Const(c) = x else { return None };
                        let to_float = matches!(self.types[ty.0 as usize], TypeDef::Float);
                        match self.consts[c as usize].1 {
                            Const::Int(bits) if to_float && self.int_bits(from) == Some(32) => {
                                Some(self.f32(bits as u32))
                            }
                            Const::Float(bits) if self.int_bits(ty) == Some(32) => {
                                Some(self.int_const(ty, i64::from(bits as i32)))
                            }
                            _ => None,
                        }
                    }
                    Cast::ZExt | Cast::SExt | Cast::Trunc => {
                        let to = self.int_bits(ty)?;
                        if let Some((bits, k)) = self.int_k(x) {
                            let k = if op == Cast::SExt {
                                signed(bits, k) as u64
                            } else {
                                k
                            };
                            return Some(self.k(ty, to, k));
                        }
                        // Widened and narrowed again.
                        if op == Cast::Trunc
                            && let Some((Cast::ZExt | Cast::SExt, z)) = through
                            && self.type_of(z) == ty
                        {
                            return Some(z);
                        }
                        None
                    }
                    _ => None,
                }
            }
            Op::Call(f, ref args)
                if super::super::rdna_fp16_opt_on()
                    && self.funcs[f as usize].name == "dx.op.binary.i32"
                    && args.len() == 3 =>
            {
                let (op_v, a, b) = (args[0], args[1], args[2]);
                let Some((32, op)) = self.int_k(op_v) else {
                    return None;
                };
                if !(37..=40).contains(&op) {
                    return None;
                }
                if a == b {
                    return Some(a);
                }
                let ka = self.int_k(a).filter(|&(w, _)| w == 32).map(|(_, k)| k as u32);
                let kb = self.int_k(b).filter(|&(w, _)| w == 32).map(|(_, k)| k as u32);
                if let (Some(x), Some(y)) = (ka, kb) {
                    let r = match op {
                        37 => (x as i32).max(y as i32) as u32, // IMax
                        38 => (x as i32).min(y as i32) as u32, // IMin
                        39 => x.max(y),                        // UMax
                        _ => x.min(y),                         // UMin
                    };
                    return Some(self.k(ty, 32, u64::from(r)));
                }
                match (op, ka, kb) {
                    (37, Some(x), _) if x == i32::MIN as u32 => Some(b),
                    (37, _, Some(y)) if y == i32::MIN as u32 => Some(a),
                    (37, Some(x), _) if x == i32::MAX as u32 => Some(a),
                    (37, _, Some(y)) if y == i32::MAX as u32 => Some(b),
                    (38, Some(x), _) if x == i32::MAX as u32 => Some(b),
                    (38, _, Some(y)) if y == i32::MAX as u32 => Some(a),
                    (38, Some(x), _) if x == i32::MIN as u32 => Some(a),
                    (38, _, Some(y)) if y == i32::MIN as u32 => Some(b),
                    (39, Some(0), _) => Some(b),
                    (39, _, Some(0)) => Some(a),
                    (39, Some(u32::MAX), _) => Some(a),
                    (39, _, Some(u32::MAX)) => Some(b),
                    (40, Some(u32::MAX), _) => Some(b),
                    (40, _, Some(u32::MAX)) => Some(a),
                    (40, Some(0), _) => Some(a),
                    (40, _, Some(0)) => Some(b),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// What kind of instruction `i` is for merging.
    fn class(&self, effects: &[Effect], i: u32) -> Effect {
        match &self.insts[i as usize].op {
            Op::Bin(..)
            | Op::Cast(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Extract(..)
            | Op::Gep(..) => Effect::Pure,
            Op::Load(p) => {
                // A constant table reads the same whatever is stored.
                let base = match *p {
                    V::Inst(g) => match &self.insts[g as usize].op {
                        Op::Gep(_, base, _) => *base,
                        _ => *p,
                    },
                    _ => *p,
                };
                match base {
                    V::Global(g) if self.globals[g as usize].init.is_some() => Effect::Pure,
                    _ => Effect::Reads,
                }
            }
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => Effect::Writes,
            Op::Phi(_) | Op::Alloca(..) => Effect::Lanes,
            Op::Call(f, _) => effects[*f as usize],
        }
    }

    fn key(&self, i: u32) -> Option<Key> {
        let inst = &self.insts[i as usize];
        Some(match &inst.op {
            Op::Bin(op, a, b) => {
                // Integer sums, products and bit operations whichever way
                // round; float ones as written (which NaN an operation of
                // two NaNs gives is the hardware's choice by operand).
                let commutes = matches!(op, Bin::Add | Bin::Mul | Bin::And | Bin::Or | Bin::Xor)
                    && self.int_bits(inst.ty).is_some();
                let (a, b) = if commutes && b < a {
                    (*b, *a)
                } else {
                    (*a, *b)
                };
                let flags = self.fast_math.get(&i).copied().unwrap_or(0);
                Key::Bin(*op as u8, flags, a, b)
            }
            Op::Cast(op, v) => Key::Cast(*op as u8, inst.ty, *v),
            Op::Cmp(p, a, b) => Key::Cmp(*p, *a, *b),
            Op::Select(c, t, f) => Key::Select(*c, *t, *f),
            Op::Extract(a, n) => Key::Extract(*a, *n),
            Op::Call(f, args) => Key::Call(*f, args.clone()),
            Op::Gep(ty, p, idx) => Key::Gep(*ty, *p, idx.clone()),
            Op::Load(p) => Key::Load(*p),
            _ => return None,
        })
    }

    /// [`FOLD`], [`REUSE`], [`FACTS`] and [`LOADS`]: one walk over the
    /// dominator tree. Returns whether anything changed.
    fn merge(&mut self, passes: u32) -> bool {
        if self.blocks.is_empty() {
            return false;
        }
        let effects = self.effects();
        let tree = self.dominator_tree();
        let mut repl: Vec<Option<V>> = vec![None; self.insts.len()];
        let mut changed = false;
        // Joins of one value: until none is left (one gone can leave
        // another with one value).
        if passes & FOLD != 0 {
            let mut again = true;
            while again {
                again = false;
                for &b in &tree.order {
                    for &p in &self.blocks[b].phis {
                        if repl[p as usize].is_some() {
                            continue;
                        }
                        let Op::Phi(incoming) = &self.insts[p as usize].op else {
                            continue;
                        };
                        let mut one = None;
                        let mut same = true;
                        for &(v, _) in incoming {
                            let v = resolve(&repl, v);
                            if v == V::Inst(p) {
                                continue;
                            }
                            match one {
                                None => one = Some(v),
                                Some(o) if o == v => {}
                                Some(_) => same = false,
                            }
                        }
                        if let (true, Some(v)) = (same, one) {
                            repl[p as usize] = Some(v);
                            again = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        // Per-block `RayQueryMut` handles (`None` = unknown/wildcard) for cross-block
        // dominated `RayQueryRead` CSE.
        let has_rq = passes & LOADS != 0
            && effects
                .iter()
                .any(|&e| matches!(e, Effect::RayQueryRead | Effect::RayQueryMut));
        let mut block_rq_muts: Vec<Vec<Option<V>>> = if has_rq {
            vec![Vec::new(); self.blocks.len()]
        } else {
            Vec::new()
        };
        if has_rq {
            for &b in &tree.order {
                for &i in &self.blocks[b].body {
                    if self.class(&effects, i) == Effect::RayQueryMut {
                        let rq = match &self.insts[i as usize].op {
                            Op::Call(_, args) => args.get(1).map(|&v| resolve(&repl, v)),
                            _ => None,
                        };
                        if !block_rq_muts[b].contains(&rq) {
                            block_rq_muts[b].push(rq);
                        }
                    }
                }
            }
        }
        let mut rq_clean_cache: HashMap<(Option<V>, usize, usize), bool> = HashMap::new();
        let mut rq_dom_known: HashMap<Key, (V, usize)> = HashMap::new();
        let mut rq_local: HashMap<Key, (V, Option<V>)> = HashMap::new();
        let mut rq_mut_in_cur: Vec<Option<V>> = Vec::new();
        // The values every path to the block being walked has computed
        // (those of the blocks above it in the tree, and its own so far).
        let mut known: HashMap<Key, V> = HashMap::new();
        // The reads of the block being walked since its last store.
        let mut read: HashMap<Key, V> = HashMap::new();
        let mut facts = Facts::default();
        let (yes, no) = (self.bool(true), self.bool(false));
        enum Step {
            Enter(usize),
            Leave(Vec<Key>, Vec<(Key, Option<(V, usize)>)>, Learned),
        }
        let mut steps = vec![Step::Enter(0)];
        while let Some(step) = steps.pop() {
            let b = match step {
                Step::Enter(b) => b,
                Step::Leave(keys, rq_undo, learned) => {
                    for k in keys {
                        known.remove(&k);
                    }
                    for (k, prev) in rq_undo.into_iter().rev() {
                        match prev {
                            Some(v) => {
                                rq_dom_known.insert(k, v);
                            }
                            None => {
                                rq_dom_known.remove(&k);
                            }
                        }
                    }
                    for v in learned.bits {
                        facts.bits.remove(&v);
                    }
                    for k in learned.compares {
                        facts.compares.remove(&k);
                    }
                    continue;
                }
            };
            let mut added = Vec::new();
            // Reached from one block alone, by one side of its branch: the
            // condition is that side's, here and in every block below.
            let mut learned = Learned::default();
            if passes & FACTS != 0
                && let [p] = tree.preds[b][..]
                && let Some(Term::CondBr(c, t, f)) = self.blocks[p].term
                && t != f
            {
                self.learn(resolve(&repl, c), b == t, &mut facts, &mut learned, 8);
            }
            let known_bit = |facts: &Facts, v: V| match facts.bits.get(&v) {
                Some(true) => yes,
                Some(false) => no,
                None => v,
            };
            read.clear();
            rq_local.clear();
            rq_mut_in_cur.clear();
            // Two joins of the same values from the same blocks are one.
            if passes & REUSE != 0 && self.blocks[b].phis.len() > 1 {
                let mut joins: HashMap<Vec<(usize, V)>, V> = HashMap::new();
                for &p in &self.blocks[b].phis {
                    if repl[p as usize].is_some() {
                        continue;
                    }
                    let ty = self.insts[p as usize].ty;
                    let Op::Phi(incoming) = &self.insts[p as usize].op else {
                        continue;
                    };
                    let mut key: Vec<(usize, V)> = incoming
                        .iter()
                        .map(|&(v, from)| (from, resolve(&repl, v)))
                        .collect();
                    key.sort();
                    // The type too: an undefined value of one type is not
                    // that of another.
                    key.push((ty.0 as usize, V::Global(u32::MAX)));
                    match joins.get(&key) {
                        Some(&first) => {
                            repl[p as usize] = Some(first);
                            changed = true;
                        }
                        None => {
                            joins.insert(key, V::Inst(p));
                        }
                    }
                }
            }
            let body = std::mem::take(&mut self.blocks[b].body);
            let mut kept = Vec::with_capacity(body.len());
            for i in body {
                map_operands(&mut self.insts[i as usize].op, |v| {
                    let v = resolve(&repl, v);
                    let bit = known_bit(&facts, v);
                    changed |= bit != v;
                    bit
                });
                if let Op::Cmp(p, x, y) = self.insts[i as usize].op
                    && let Some(&holds) = facts.compares.get(&(p, x, y))
                {
                    repl[i as usize] = Some(if holds { yes } else { no });
                    changed = true;
                    continue;
                }
                if passes & FOLD != 0
                    && let Some(v) = self.simplify(i, &mut changed)
                {
                    repl[i as usize] = Some(v);
                    changed = true;
                    continue;
                }
                let class = self.class(&effects, i);
                if passes & REUSE != 0
                    && class == Effect::Pure
                    && let Some(key) = self.key(i)
                {
                    if let Some(&first) = known.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    known.insert(key.clone(), V::Inst(i));
                    added.push(key);
                }
                if passes & LOADS != 0
                    && class == Effect::RayQueryRead
                    && let Some(key) = self.key(i)
                {
                    let rq = match &self.insts[i as usize].op {
                        Op::Call(_, args) => args.get(1).copied(),
                        _ => None,
                    };
                    if let Some(&(first, _)) = rq_local.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    let mut_in_cur = rq_mut_in_cur
                        .iter()
                        .any(|m| m.is_none() || rq.is_none() || *m == rq);
                    if !mut_in_cur
                        && let Some(&(first, def_blk)) = rq_dom_known.get(&key)
                        && Self::rq_path_clean(
                            rq,
                            def_blk,
                            b,
                            &tree.preds,
                            &block_rq_muts,
                            &mut rq_clean_cache,
                        )
                    {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    rq_local.insert(key, (V::Inst(i), rq));
                }
                if passes & LOADS != 0 && class == Effect::RayQueryMut {
                    let rq = match &self.insts[i as usize].op {
                        Op::Call(_, args) => args.get(1).copied(),
                        _ => None,
                    };
                    rq_mut_in_cur.push(rq);
                    rq_local.retain(|_, (_, r)| !(rq.is_none() || r.is_none() || *r == rq));
                }
                // Reads of memory, and wave operations (the same lanes run
                // the whole block): the block's own earlier one.
                if passes & LOADS != 0
                    && matches!(class, Effect::Reads | Effect::Lanes)
                    && let Some(key) = self.key(i)
                {
                    if let Some(&first) = read.get(&key) {
                        repl[i as usize] = Some(first);
                        changed = true;
                        continue;
                    }
                    read.insert(key, V::Inst(i));
                }
                if class == Effect::Writes {
                    read.clear();
                }
                kept.push(i);
            }
            self.blocks[b].body = kept;
            if let Some(Term::CondBr(c, t, f)) = self.blocks[b].term {
                let c = known_bit(&facts, resolve(&repl, c));
                self.blocks[b].term = Some(Term::CondBr(c, t, f));
            }
            let mut rq_undo = Vec::new();
            for (k, (v, _)) in rq_local.drain() {
                let prev = rq_dom_known.insert(k.clone(), (v, b));
                rq_undo.push((k, prev));
            }
            steps.push(Step::Leave(added, rq_undo, learned));
            for &c in tree.children[b].iter().rev() {
                steps.push(Step::Enter(c));
            }
        }
        if !changed {
            return false;
        }
        // Everything that read a replaced value reads its replacement
        // (joins read values from blocks walked after them).
        for blk in &mut self.blocks {
            blk.phis.retain(|&p| repl[p as usize].is_none());
            if let Some(Term::CondBr(c, t, f)) = blk.term {
                blk.term = Some(Term::CondBr(resolve(&repl, c), t, f));
            }
        }
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for blk in blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                map_operands(&mut insts[i as usize].op, |v| resolve(&repl, v));
            }
        }
        true
    }

    /// Whether every CFG path from `def_blk` to the entry of `cur_blk` (including loop
    /// back-edges through `cur_blk`) is free of `RayQueryMut` calls affecting `rq`.
    fn rq_path_clean(
        rq: Option<V>,
        def_blk: usize,
        cur_blk: usize,
        preds: &[Vec<usize>],
        block_rq_muts: &[Vec<Option<V>>],
        cache: &mut HashMap<(Option<V>, usize, usize), bool>,
    ) -> bool {
        if def_blk == cur_blk {
            return true;
        }
        if let Some(&ok) = cache.get(&(rq, def_blk, cur_blk)) {
            return ok;
        }
        let mut visited = vec![false; preds.len()];
        // Intentionally do NOT pre-mark `cur_blk` so cycles back into `cur_blk` check `cur_blk`.
        let mut stack: Vec<usize> = preds[cur_blk].clone();
        while let Some(x) = stack.pop() {
            if x == def_blk || visited[x] {
                continue;
            }
            visited[x] = true;
            if block_rq_muts[x]
                .iter()
                .any(|m| m.is_none() || rq.is_none() || *m == rq)
            {
                cache.insert((rq, def_blk, cur_blk), false);
                return false;
            }
            stack.extend_from_slice(&preds[x]);
        }
        cache.insert((rq, def_blk, cur_blk), true);
        true
    }

    /// Whether `h` is an annotated read-only SRV buffer handle (`kind & 0x1000 == 0`
    /// and `kind & 0xFF in 10..=12`), and whether it is one of the always-bound immutable
    /// host stride root SRVs (`t255` `SLOT_STRIDES` or `t252` `SLOT_TABLE_STRIDES`).
    fn srv_handle_kind(&self, h: V) -> (bool, bool) {
        let V::Inst(a) = h else {
            return (false, false);
        };
        let Op::Call(f, args) = &self.insts[a as usize].op else {
            return (false, false);
        };
        if self.funcs[*f as usize].name != "dx.op.annotateHandle" || args.len() < 3 {
            return (false, false);
        }
        let V::Const(props) = args[2] else {
            return (false, false);
        };
        let Const::Aggregate(fields) = &self.consts[props as usize].1 else {
            return (false, false);
        };
        let Some(&Const::Int(kind)) = fields.first().map(|&c| &self.consts[c as usize].1) else {
            return (false, false);
        };
        let is_ro_buf = (kind & 0x1000) == 0 && matches!(kind & 0xFF, 10..=12);
        if !is_ro_buf {
            return (false, false);
        }
        let mut is_stride_srv = false;
        if (kind & 0xFF) == 11
            && let V::Inst(c) = args[1]
            && let Op::Call(cf, cargs) = &self.insts[c as usize].op
            && self.funcs[*cf as usize].name == "dx.op.createHandleFromBinding"
            && cargs.len() >= 2
            && let V::Const(bc) = cargs[1]
            && let Const::Aggregate(bf) = &self.consts[bc as usize].1
            && bf.len() >= 4
        {
            if let (
                Const::Int(lo),
                Const::Int(hi),
                Const::Int(space),
                Const::Int(class),
            ) = (
                &self.consts[bf[0] as usize].1,
                &self.consts[bf[1] as usize].1,
                &self.consts[bf[2] as usize].1,
                &self.consts[bf[3] as usize].1,
            ) {
                is_stride_srv = *space == i64::from(super::super::shader::ROOT_CBV_SPACE)
                    && *class == 0
                    && *lo == *hi
                    && (*lo == i64::from(super::super::shader::SLOT_STRIDES)
                        || *lo == i64::from(super::super::shader::SLOT_TABLE_STRIDES));
            }
        }
        (true, is_stride_srv)
    }

    fn dominates(idom: &[usize], a: usize, mut b: usize) -> bool {
        while b != a {
            let p = idom[b];
            if p == b || p == usize::MAX {
                return false;
            }
            b = p;
        }
        true
    }

    /// [`LOADS`] LICM: hoists loop-invariant immutable stride SRV loads (`t255`/`t252`)
    /// from anywhere in a natural loop, and must-execute (`B == header`) read-only SRV
    /// buffer loads in loops without memory writes, into the loop's dedicated preheader
    /// along with their loop-invariant pure operands and `Extract`s.
    fn licm(&mut self) -> bool {
        if self.blocks.len() < 2 {
            return false;
        }
        let effects = self.effects();
        let tree = self.dominator_tree();
        // Discover natural loops by back-edges `latch -> header` where `header dom latch`.
        let mut back_edges: HashMap<usize, Vec<usize>> = HashMap::new();
        for &b in &tree.order {
            for &s in successors(self.blocks[b].term).iter() {
                if Self::dominates(&tree.idom, s, b) {
                    back_edges.entry(s).or_default().push(b);
                }
            }
        }
        if back_edges.is_empty() {
            return false;
        }
        let mut loops: Vec<(usize, HashSet<usize>)> = Vec::with_capacity(back_edges.len());
        for (h, latches) in back_edges {
            let mut body: HashSet<usize> = HashSet::new();
            body.insert(h);
            let mut stack = latches;
            while let Some(x) = stack.pop() {
                if body.insert(x) {
                    stack.extend_from_slice(&tree.preds[x]);
                }
            }
            // Require reducible natural loop (`h` dominates every block in `body`).
            if body.iter().all(|&b| Self::dominates(&tree.idom, h, b)) {
                loops.push((h, body));
            }
        }
        // Process innermost loops first so hoisted ops cascade outward.
        loops.sort_by_key(|(_, body)| body.len());

        let mut def_block = vec![usize::MAX; self.insts.len()];
        for &b in &tree.order {
            for &i in self.blocks[b].phis.iter().chain(&self.blocks[b].body) {
                def_block[i as usize] = b;
            }
        }

        let mut any_changed = false;
        for (h, loop_blocks) in loops {
            let outside_preds: Vec<usize> = tree.preds[h]
                .iter()
                .copied()
                .filter(|p| !loop_blocks.contains(p))
                .collect();
            if outside_preds.len() != 1 {
                continue;
            }
            let pre = outside_preds[0];
            if !matches!(self.blocks[pre].term, Some(Term::Br(t)) if t == h) {
                continue;
            }

            let mut loop_has_mem_writes = false;
            for &b in &loop_blocks {
                for &i in &self.blocks[b].body {
                    let op = &self.insts[i as usize].op;
                    if matches!(op, Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..))
                        || self.class(&effects, i) == Effect::Writes
                    {
                        loop_has_mem_writes = true;
                        break;
                    }
                }
                if loop_has_mem_writes {
                    break;
                }
            }

            // Determine whether an instruction `i` inside `loop_blocks` can be safely
            // computed in `pre` as a loop-invariant dependency (no div/rem speculation).
            let safe_pure_dep = |m: &Module, i: u32| -> bool {
                match &m.insts[i as usize].op {
                    Op::Bin(Bin::UDiv | Bin::URem, _, _) => false,
                    Op::Phi(_) | Op::Alloca(..) => false,
                    _ => m.class(&effects, i) == Effect::Pure,
                }
            };
            // Recursive check (with memoization) for whether every loop-defined operand
            // of `i` is a safe hoistable pure op.
            let mut inv_cache: HashMap<u32, bool> = HashMap::new();
            fn is_inv_dep(
                m: &Module,
                i: u32,
                loop_blocks: &HashSet<usize>,
                def_block: &[usize],
                safe_pure: &impl Fn(&Module, u32) -> bool,
                cache: &mut HashMap<u32, bool>,
            ) -> bool {
                let db = def_block[i as usize];
                if db == usize::MAX || !loop_blocks.contains(&db) {
                    return true;
                }
                if let Some(&ok) = cache.get(&i) {
                    return ok;
                }
                if !safe_pure(m, i) {
                    cache.insert(i, false);
                    return false;
                }
                cache.insert(i, false); // cycle guard
                let mut ok = true;
                for_each_operand(&m.insts[i as usize].op, |v| {
                    if ok && let V::Inst(u) = v {
                        ok = is_inv_dep(m, u, loop_blocks, def_block, safe_pure, cache);
                    }
                });
                cache.insert(i, ok);
                ok
            }

            let mut hoist_list: Vec<u32> = Vec::new();
            let mut hoist_set: HashSet<u32> = HashSet::new();
            let mut hoisted_loads: HashSet<u32> = HashSet::new();

            fn collect_deps(
                m: &Module,
                i: u32,
                loop_blocks: &HashSet<usize>,
                def_block: &[usize],
                hoist_set: &mut HashSet<u32>,
                hoist_list: &mut Vec<u32>,
            ) {
                for_each_operand(&m.insts[i as usize].op, |v| {
                    if let V::Inst(u) = v {
                        let db = def_block[u as usize];
                        if db != usize::MAX && loop_blocks.contains(&db) && !hoist_set.contains(&u)
                        {
                            collect_deps(m, u, loop_blocks, def_block, hoist_set, hoist_list);
                            if hoist_set.insert(u) {
                                hoist_list.push(u);
                            }
                        }
                    }
                });
            }

            // Walk loop blocks in RPO order (`tree.order` filtered by `loop_blocks`).
            let ordered_loop: Vec<usize> = tree
                .order
                .iter()
                .copied()
                .filter(|b| loop_blocks.contains(b))
                .collect();
            for &b in &ordered_loop {
                for &i in &self.blocks[b].body {
                    let Op::Call(f, args) = &self.insts[i as usize].op else {
                        continue;
                    };
                    let name = self.funcs[*f as usize].name.as_str();
                    if !(name.starts_with("dx.op.rawBufferLoad.")
                        || name.starts_with("dx.op.bufferLoad."))
                        || args.len() < 2
                    {
                        continue;
                    }
                    let (is_ro_srv, is_stride_srv) = self.srv_handle_kind(args[1]);
                    let eligible = is_stride_srv || (is_ro_srv && !loop_has_mem_writes && b == h);
                    if !eligible {
                        continue;
                    }
                    let mut operands_inv = true;
                    for_each_operand(&self.insts[i as usize].op, |v| {
                        if operands_inv && let V::Inst(u) = v {
                            operands_inv = is_inv_dep(
                                self,
                                u,
                                &loop_blocks,
                                &def_block,
                                &safe_pure_dep,
                                &mut inv_cache,
                            );
                        }
                    });
                    if !operands_inv {
                        continue;
                    }
                    collect_deps(
                        self,
                        i,
                        &loop_blocks,
                        &def_block,
                        &mut hoist_set,
                        &mut hoist_list,
                    );
                    if hoist_set.insert(i) {
                        hoist_list.push(i);
                        hoisted_loads.insert(i);
                    }
                }
            }

            if hoist_list.is_empty() {
                continue;
            }
            // Also hoist `Extract` of hoisted loads so only the extracted scalar crosses the loop.
            for &b in &ordered_loop {
                for &i in &self.blocks[b].body {
                    if let Op::Extract(V::Inst(src), _) = self.insts[i as usize].op
                        && hoisted_loads.contains(&src)
                        && hoist_set.insert(i)
                    {
                        hoist_list.push(i);
                    }
                }
            }
            for &b in &ordered_loop {
                self.blocks[b].body.retain(|i| !hoist_set.contains(i));
            }
            for &i in &hoist_list {
                def_block[i as usize] = pre;
                self.blocks[pre].body.push(i);
            }
            any_changed = true;
        }
        any_changed
    }

    /// [`FOLD`]: a branch on a constant goes one way, and trivial empty diamonds
    /// collapse directly into their join block. The side not taken loses the
    /// branching block's values at its joins, and so do the joins of blocks that
    /// nothing reaches any more. Returns whether any branch changed.
    fn prune(&mut self) -> bool {
        let mut changed = false;
        for b in 0..self.blocks.len() {
            let Some(Term::CondBr(c, t, f)) = self.blocks[b].term else {
                continue;
            };
            if t == f {
                if self.blocks[t].phis.is_empty() {
                    self.blocks[b].term = Some(Term::Br(t));
                    changed = true;
                }
                continue;
            }
            let Some((_, k)) = self.int_k(c) else {
                continue;
            };
            let (taken, not) = if k != 0 { (t, f) } else { (f, t) };
            self.blocks[b].term = Some(Term::Br(taken));
            changed = true;
            for n in 0..self.blocks[not].phis.len() {
                let p = self.blocks[not].phis[n];
                if let Op::Phi(incoming) = &mut self.insts[p as usize].op {
                    incoming.retain(|&(_, from)| from != b);
                }
            }
        }
        // Collapse empty diamonds (`b -> {t, f} -> j` and asymmetric `b -> {empty -> j, j}`).
        let reached_now = self.reached();
        let mut preds = vec![Vec::new(); self.blocks.len()];
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached_now[b] {
                continue;
            }
            for &s in successors(blk.term).iter() {
                preds[s].push(b);
            }
        }
        let mut touched = vec![false; self.blocks.len()];
        for b in 0..self.blocks.len() {
            if !reached_now[b] || touched[b] {
                continue;
            }
            let Some(Term::CondBr(_, t, f)) = self.blocks[b].term else {
                continue;
            };
            if t == f || touched[t] || touched[f] {
                continue;
            }
            let is_empty_sole = |x: usize| -> Option<usize> {
                if x == 0 || x == b || preds[x].as_slice() != [b] {
                    return None;
                }
                let blk = &self.blocks[x];
                if !blk.phis.is_empty() || !blk.body.is_empty() {
                    return None;
                }
                match blk.term {
                    Some(Term::Br(j)) if j != x => Some(j),
                    _ => None,
                }
            };
            // Case 1: Symmetric empty diamond `b -> {t, f}`, both empty `Br(j)`.
            if let (Some(jt), Some(jf)) = (is_empty_sole(t), is_empty_sole(f))
                && jt == jf
                && !touched[jt]
                && !preds[jt].contains(&b)
            {
                let j = jt;
                let phis_match = self.blocks[j].phis.iter().all(|&p| {
                    let Op::Phi(inc) = &self.insts[p as usize].op else {
                        return false;
                    };
                    let vt = inc.iter().find(|&&(_, from)| from == t).map(|&(v, _)| v);
                    let vf = inc.iter().find(|&&(_, from)| from == f).map(|&(v, _)| v);
                    vt.is_some() && vt == vf
                });
                if phis_match {
                    self.blocks[b].term = Some(Term::Br(j));
                    for n in 0..self.blocks[j].phis.len() {
                        let p = self.blocks[j].phis[n];
                        if let Op::Phi(inc) = &mut self.insts[p as usize].op {
                            let v = inc.iter().find(|&&(_, from)| from == t).unwrap().0;
                            inc.retain(|&(_, from)| from != t && from != f);
                            inc.push((v, b));
                        }
                    }
                    touched[b] = true;
                    touched[t] = true;
                    touched[f] = true;
                    touched[j] = true;
                    changed = true;
                    continue;
                }
            }
            // Case 2: Asymmetric empty diamond `b -> {empty -> j, j}`.
            let asym = if is_empty_sole(t) == Some(f) {
                Some((t, f))
            } else if is_empty_sole(f) == Some(t) {
                Some((f, t))
            } else {
                None
            };
            if let Some((empty, j)) = asym
                && !touched[j]
            {
                let phis_match = self.blocks[j].phis.iter().all(|&p| {
                    let Op::Phi(inc) = &self.insts[p as usize].op else {
                        return false;
                    };
                    let ve = inc.iter().find(|&&(_, from)| from == empty).map(|&(v, _)| v);
                    let vb = inc.iter().find(|&&(_, from)| from == b).map(|&(v, _)| v);
                    ve.is_some() && ve == vb
                });
                if phis_match {
                    self.blocks[b].term = Some(Term::Br(j));
                    for n in 0..self.blocks[j].phis.len() {
                        let p = self.blocks[j].phis[n];
                        if let Op::Phi(inc) = &mut self.insts[p as usize].op {
                            inc.retain(|&(_, from)| from != empty);
                        }
                    }
                    touched[b] = true;
                    touched[empty] = true;
                    touched[j] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            return false;
        }
        let reached = self.reached();
        let (blocks, insts) = (&self.blocks, &mut self.insts);
        for (_, blk) in blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &p in &blk.phis {
                if let Op::Phi(incoming) = &mut insts[p as usize].op {
                    incoming.retain(|&(_, from)| reached[from]);
                }
            }
        }
        true
    }

    /// Whether instruction `i` must stay whatever reads it.
    fn has_effect(&self, effects: &[Effect], i: u32) -> bool {
        match &self.insts[i as usize].op {
            Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => true,
            Op::Call(f, _) => matches!(effects[*f as usize], Effect::Writes | Effect::RayQueryMut),
            _ => false,
        }
    }

    /// [`DEAD`]: keeps the instructions with an effect, the branch
    /// conditions, and what they read, transitively; the rest leave their
    /// blocks. Blocks no branch reaches are emptied (they are not written).
    fn drop_dead(&mut self) {
        let effects = self.effects();
        let reached = self.reached();
        let mut live = vec![false; self.insts.len()];
        let mut work: Vec<u32> = Vec::new();
        let mark = |live: &mut [bool], work: &mut Vec<u32>, v: V| {
            if let V::Inst(i) = v
                && !live[i as usize]
            {
                live[i as usize] = true;
                work.push(i);
            }
        };
        for (b, blk) in self.blocks.iter().enumerate() {
            if !reached[b] {
                continue;
            }
            for &i in &blk.body {
                if self.has_effect(&effects, i) {
                    mark(&mut live, &mut work, V::Inst(i));
                }
            }
            if let Some(Term::CondBr(c, ..)) = blk.term {
                mark(&mut live, &mut work, c);
            }
        }
        while let Some(i) = work.pop() {
            for_each_operand(&self.insts[i as usize].op, |v| {
                mark(&mut live, &mut work, v)
            });
        }
        for (b, blk) in self.blocks.iter_mut().enumerate() {
            if reached[b] {
                blk.body.retain(|&i| live[i as usize]);
                blk.phis.retain(|&i| live[i as usize]);
            } else {
                blk.body.clear();
                blk.phis.clear();
            }
        }
        self.drop_unused_functions();
        self.drop_unused_tables();
    }

    /// See [`Usage`]: over the blocks reached from the entry.
    pub(crate) fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let reached = self.reached();
        let wide = |ty: Ty| self.types[ty.0 as usize] == TypeDef::Int(64);
        // The first field of a handle's annotation: its kind, with the UAV
        // bit; the second: the component type and count.
        let annotation = |h: V| -> Option<(i64, i64)> {
            let V::Inst(a) = h else { return None };
            let Op::Call(f, args) = &self.insts[a as usize].op else {
                return None;
            };
            if self.funcs[*f as usize].name != "dx.op.annotateHandle" {
                return None;
            }
            let V::Const(props) = *args.get(2)? else {
                return None;
            };
            let Const::Aggregate(fields) = &self.consts[props as usize].1 else {
                return None;
            };
            match (
                &self.consts[fields[0] as usize].1,
                &self.consts[fields[1] as usize].1,
            ) {
                (Const::Int(kind), Const::Int(components)) => Some((*kind, *components)),
                _ => None,
            }
        };
        for (_, blk) in self.blocks.iter().enumerate().filter(|(b, _)| reached[*b]) {
            for &i in blk.phis.iter().chain(&blk.body) {
                let inst = &self.insts[i as usize];
                u.int64 |= wide(inst.ty);
                for_each_operand(&inst.op, |v| u.int64 |= wide(self.type_of(v)));
                let Op::Call(f, args) = &inst.op else {
                    continue;
                };
                let name = self.funcs[*f as usize].name.as_str();
                let name = name.strip_prefix("dx.op.").unwrap_or(name);
                if name.starts_with("wave") || name.starts_with("quad") {
                    u.wave_ops = true;
                } else if name.starts_with("allocateRayQuery") || name.starts_with("rayQuery_") {
                    u.raytracing_tier_1_1 = true;
                } else if name == "createHandleFromHeap" {
                    match args.get(2).and_then(|&s| self.int_k(s)) {
                        Some((_, 0)) => u.heap_views = true,
                        _ => u.heap_samplers = true,
                    }
                } else if name == "annotateHandle" {
                    if let Some((kind, _)) = annotation(V::Inst(i)) {
                        u.rovs |= kind & 0x2000 != 0;
                        u.raytracing_tier_1_1 |= (kind & 0xff) == 16;
                    }
                } else if name.starts_with("atomic") && name.ends_with(".i64") {
                    u.atomic64 = true;
                } else if name == "tertiary.i32"
                    && args.first().is_some_and(|&op| self.int_k(op) == Some((32, 50)))
                {
                    u.msad = true;
                } else if name.starts_with("textureLoad") || name.starts_with("bufferLoad") {
                    u.typed_uav_loads |= args.get(1).and_then(|&h| annotation(h)).is_none_or(
                        |(kind, components)| kind & 0x1000 != 0 && components >> 8 & 0xff > 1,
                    );
                }
            }
        }
        u
    }

    /// A constant table no remaining load reads leaves the module (the
    /// validator refuses an internal declaration nothing uses). Shared
    /// memory stays declared whether it is read or not.
    fn drop_unused_tables(&mut self) {
        let mut used: Vec<bool> = self.globals.iter().map(|g| g.init.is_none()).collect();
        if used.iter().all(|&u| u) {
            return;
        }
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                for_each_operand(&self.insts[i as usize].op, |v| {
                    if let V::Global(g) = v {
                        used[g as usize] = true;
                    }
                });
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Global(g)) = md {
                used[*g as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.globals.retain(|_| {
            old += 1;
            used[old - 1]
        });
        let renumber = |v: V| match v {
            V::Global(g) if used[g as usize] => V::Global(new[g as usize]),
            v => v,
        };
        for inst in &mut self.insts {
            map_operands(&mut inst.op, renumber);
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, v) = md {
                *v = renumber(*v);
            }
        }
    }

    /// Declarations no remaining call names leave the module (the validator
    /// refuses an external function nothing uses); the others keep their
    /// order, so `main` stays function 0.
    fn drop_unused_functions(&mut self) {
        let mut used: Vec<bool> = self.funcs.iter().map(|f| f.body).collect();
        for blk in &self.blocks {
            for &i in blk.phis.iter().chain(&blk.body) {
                if let Op::Call(f, _) = self.insts[i as usize].op {
                    used[f as usize] = true;
                }
            }
        }
        for md in &self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                used[*f as usize] = true;
            }
        }
        if used.iter().all(|&u| u) {
            return;
        }
        let mut new = vec![u32::MAX; used.len()];
        for (next, (old, _)) in used.iter().enumerate().filter(|u| *u.1).enumerate() {
            new[old] = next as u32;
        }
        let mut old = 0;
        self.funcs.retain(|_| {
            old += 1;
            used[old - 1]
        });
        self.func_ids = self
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i as u32))
            .collect();
        // Instructions that left their block keep a stale number: nothing
        // reads them again.
        for inst in &mut self.insts {
            if let Op::Call(f, _) = &mut inst.op
                && used[*f as usize]
            {
                *f = new[*f as usize];
            }
        }
        for md in &mut self.mds {
            if let MdDef::Value(_, V::Func(f)) = md {
                *f = new[*f as usize];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `main` with one block and a function of each kind of effect.
    struct Kit {
        m: Module,
        b: usize,
        pure: V,
        reads: V,
        writes: V,
    }

    fn kit() -> Kit {
        let mut m = Module::default();
        let i32t = m.int(32);
        let void = m.void();
        let fty = m.func_ty(i32t, vec![i32t, i32t]);
        m.ptr(fty);
        let pure = m.function("dx.op.binary.i32", fty, Some(Attr::ReadNone), false);
        let reads = m.function("dx.op.bufferLoad.i32", fty, Some(Attr::ReadOnly), false);
        let writes = m.function("dx.op.bufferStore.i32", fty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        Kit {
            m,
            b,
            pure,
            reads,
            writes,
        }
    }

    fn body(m: &Module, b: usize) -> usize {
        m.blocks[b].body.len()
    }

    #[test]
    fn unread_results_without_effect_are_dropped() {
        let Kit {
            mut m,
            b,
            pure,
            reads,
            writes,
        } = kit();
        let (one, two) = (m.i32(1), m.i32(2));
        let used = m.call(b, reads, vec![one, two]);
        let dead = m.call(b, reads, vec![two, one]);
        let chain = m.bin(b, Bin::Add, dead, one);
        let _unread = m.call(b, pure, vec![chain, two]);
        // A store's result is unread and stays, with what it reads.
        let kept = m.bin(b, Bin::Add, used, two);
        m.call(b, writes, vec![kept, one]);
        m.terminate(b, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(body(&m, b), 3, "the load, the add and the store");
        m.write();
    }

    #[test]
    fn branch_conditions_and_phi_cycles() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let i32t = m.int(32);
        let (zero, one) = (m.i32(0), m.i32(1));
        let (head, exit) = (m.new_block(), m.new_block());
        m.terminate(b, Term::Br(head));
        // A counter that decides the branch, and a register that only
        // feeds itself round the loop.
        let count = m.phi(head, i32t);
        let idle = m.phi(head, i32t);
        let next = m.bin(head, Bin::Add, count, one);
        let idle_next = m.call(head, reads, vec![idle, one]);
        let more = m.cmp(head, pred::IULT, next, one);
        m.terminate(head, Term::CondBr(more, head, exit));
        m.set_phi(count, vec![(zero, b), (next, head)]);
        m.set_phi(idle, vec![(zero, b), (idle_next, head)]);
        m.call(exit, writes, vec![zero, one]);
        m.terminate(exit, Term::Ret);
        m.optimise(DEAD);
        assert_eq!(m.blocks[head].phis.len(), 1, "the counter");
        assert_eq!(body(&m, head), 2, "the step and the test");
        assert_eq!(body(&m, exit), 1);
        m.write();
    }

    #[test]
    fn wave_operations_have_no_effect_of_their_own() {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let first = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let barrier =
            k.m.function("dx.op.barrier", fty, Some(Attr::NoDuplicate), false);
        let one = k.m.i32(1);
        k.m.call(k.b, first, vec![one, one]);
        k.m.call(k.b, barrier, vec![one, one]);
        k.m.terminate(k.b, Term::Ret);
        k.m.optimise(DEAD);
        assert_eq!(body(&k.m, k.b), 1, "the barrier");
        // The declarations nothing calls any more are gone, and the call
        // that stayed names its function by the new number.
        let names: Vec<&str> = k.m.funcs.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["main", "dx.op.barrier"]);
        let Op::Call(f, _) = k.m.insts[k.m.blocks[k.b].body[0] as usize].op else {
            panic!("a call");
        };
        assert_eq!(k.m.funcs[f as usize].name, "dx.op.barrier");
        k.m.write();
    }

    /// The operands of each store left in block `b`.
    fn stores(m: &Module, b: usize) -> Vec<Vec<V>> {
        m.blocks[b]
            .body
            .iter()
            .filter_map(|&i| match &m.insts[i as usize].op {
                Op::Call(g, args) if m.funcs[*g as usize].name.contains("Store") => {
                    Some(args.clone())
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn constants_and_identities_fold() {
        let Kit {
            mut m,
            b,
            reads,
            writes,
            ..
        } = kit();
        let (i32t, f32t) = (m.int(32), m.float());
        let (zero, one, all) = (m.i32(0), m.i32(1), m.i32(u32::MAX));
        let x = m.call(b, reads, vec![one, one]);
        // x + 0, -1 & x, x >> 0, x * 1, x | x: all x.
        let a = m.bin(b, Bin::Add, x, zero);
        let a = m.bin(b, Bin::And, all, a);
        let a = m.bin(b, Bin::LShr, a, zero);
        let a = m.bin(b, Bin::Mul, a, one);
        let a = m.bin(b, Bin::Or, a, a);
        // There and back.
        let f = m.cast(b, Cast::Bitcast, a, f32t);
        let a = m.cast(b, Cast::Bitcast, f, i32t);
        // Constants: (262168 >> 16) & 127 = 4, 7 & 31 = 7, -8 >>s 1 = -4.
        let k = m.i32(262168);
        let sixteen = m.i32(16);
        let k = m.bin(b, Bin::LShr, k, sixteen);
        let k127 = m.i32(127);
        let k = m.bin(b, Bin::And, k, k127);
        let minus8 = m.i32(-8i32 as u32);
        let s = m.bin(b, Bin::AShr, minus8, one);
        // A shift by the width is left alone.
        let k32 = m.i32(32);
        let wide = m.bin(b, Bin::Shl, one, k32);
        // A select between a value and itself, and on a constant.
        let c = m.cmp(b, pred::IULT, zero, one);
        let sel = m.select(b, c, a, wide);
        let same = m.select(b, c, sel, sel);
        m.call(b, writes, vec![same, k]);
        m.call(b, writes, vec![s, wide]);
        m.terminate(b, Term::Ret);
        m.optimise(FOLD | DEAD);
        assert_eq!(body(&m, b), 4, "the read, the wide shift and two stores");
        let four = m.i32(4);
        let minus4 = m.i32(-4i32 as u32);
        let stores = stores(&m, b);
        assert_eq!(stores, [vec![x, four], vec![minus4, wide]]);
        m.write();
    }

    #[test]
    fn float_arithmetic_is_not_folded() {
        let Kit {
            mut m, b, writes, ..
        } = kit();
        let f32t = m.float();
        let fty = m.func_ty(f32t, vec![f32t, f32t]);
        let input = m.function("dx.op.loadInput.f32", fty, Some(Attr::ReadNone), false);
        let (zero, one, two) = (m.f32(0), m.f32(1.0f32.to_bits()), m.f32(2.0f32.to_bits()));
        let x = m.call(b, input, vec![zero, zero]);
        let y = m.call(b, input, vec![one, one]);
        let a = m.bin(b, Bin::Add, x, zero);
        let p = m.bin(b, Bin::Mul, a, one);
        // The same sum the other way round is another instruction.
        let xy = m.bin(b, Bin::Add, x, y);
        let yx = m.bin(b, Bin::Add, y, x);
        // Constants compare here: 1 < 2, and nothing equals a NaN.
        let nan = m.f32(f32::NAN.to_bits());
        let lt = m.cmp(b, pred::OLT, one, two);
        let eq = m.cmp(b, pred::OEQ, nan, nan);
        let une = m.cmp(b, pred::UNE, nan, one);
        let sel = m.select(b, lt, xy, yx);
        let sel = m.select(b, eq, p, sel);
        let sel = m.select(b, une, sel, p);
        m.call(b, writes, vec![sel, p]);
        m.terminate(b, Term::Ret);
        m.optimise(ALL);
        // Two inputs, x + 0.0, * 1.0, x + y and the store: y + x and the
        // selects are gone.
        assert_eq!(body(&m, b), 6);
        assert_eq!(stores(&m, b), [[xy, p]]);
        m.write();
    }

    /// entry: `c = x < y`, branch to `then` or `other`, which join.
    struct Diamond {
        k: Kit,
        x: V,
        y: V,
        c: V,
        then: usize,
        other: usize,
        join: usize,
    }

    fn diamond() -> Diamond {
        let mut k = kit();
        let (one, two) = (k.m.i32(1), k.m.i32(2));
        let x = k.m.call(k.b, k.reads, vec![one, one]);
        let y = k.m.call(k.b, k.reads, vec![two, two]);
        let c = k.m.cmp(k.b, pred::IULT, x, y);
        let (then, other, join) = (k.m.new_block(), k.m.new_block(), k.m.new_block());
        k.m.terminate(k.b, Term::CondBr(c, then, other));
        k.m.terminate(then, Term::Br(join));
        k.m.terminate(other, Term::Br(join));
        k.m.terminate(join, Term::Ret);
        Diamond {
            k,
            x,
            y,
            c,
            then,
            other,
            join,
        }
    }

    #[test]
    fn a_value_is_reused_below_its_block_and_not_beside_it() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    pure,
                    writes,
                    ..
                },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        // The entry's value again in a branch; a branch's value again in
        // the other branch and in the join (neither is on every path).
        let top = m.call(b, pure, vec![x, y]);
        m.call(b, writes, vec![top, top]);
        let again = m.call(then, pure, vec![x, y]);
        let side = m.bin(then, Bin::Add, y, x);
        m.call(then, writes, vec![again, side]);
        let beside = m.bin(other, Bin::Add, x, y);
        m.call(other, writes, vec![beside, beside]);
        let after = m.bin(join, Bin::Add, x, y);
        m.call(join, writes, vec![after, after]);
        m.optimise(REUSE);
        assert_eq!(stores(&m, then), [[top, side]]);
        assert_eq!(body(&m, then), 2);
        assert_eq!(body(&m, other), 2);
        assert_eq!(body(&m, join), 2);
        m.write();
    }

    #[test]
    fn a_branch_taken_says_what_its_condition_is() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        // Behind the branch: the condition again (a guard), its opposite
        // comparison, and a guarded write.
        let guard = m.bin(then, Bin::And, c, d);
        let old = m.select(then, guard, x, y);
        let not = m.cmp(then, pred::IUGE, x, y);
        let new = m.select(then, not, y, old);
        m.call(then, writes, vec![new, new]);
        let kept = m.select(other, c, x, y);
        m.call(other, writes, vec![kept, kept]);
        // After the join nothing is known.
        let open = m.select(join, c, x, y);
        m.call(join, writes, vec![open, open]);
        m.optimise(FOLD | FACTS);
        // then: select(d, x, y) and the store; other: y is stored.
        assert_eq!(body(&m, then), 2);
        let V::Inst(sel) = stores(&m, then)[0][0] else {
            panic!()
        };
        assert!(matches!(m.insts[sel as usize].op, Op::Select(g, a, z) if (g, a, z) == (d, x, y)));
        assert_eq!(stores(&m, other), [[y, y]]);
        assert_eq!(body(&m, join), 2);
        let _ = both;
        m.write();

        // Both terms of an `and` that held hold behind it.
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            c,
            ..
        } = diamond();
        let three = m.i32(3);
        let d = m.call(b, reads, vec![three, three]);
        let d = m.cmp(b, pred::NE, d, three);
        let both = m.bin(b, Bin::And, c, d);
        let (inner, out) = (m.new_block(), m.new_block());
        m.terminate(b, Term::CondBr(both, inner, out));
        let a = m.select(inner, c, x, y);
        let z = m.select(inner, d, a, y);
        m.call(inner, writes, vec![z, z]);
        m.terminate(inner, Term::Br(out));
        // Where the `and` did not hold either term may.
        let w = m.select(out, d, x, y);
        m.call(out, writes, vec![w, w]);
        m.terminate(out, Term::Ret);
        m.optimise(FOLD | FACTS);
        assert_eq!(stores(&m, inner), [[x, x]]);
        assert_eq!(body(&m, out), 2);
        m.write();
    }

    #[test]
    fn a_branch_on_a_constant_goes_one_way() {
        let Diamond {
            k: Kit {
                mut m, b, writes, ..
            },
            x,
            y,
            c,
            then,
            other,
            join,
        } = diamond();
        let i32t = m.int(32);
        // Behind the branch the same test again: its other side is never
        // run, and the join behind it has one value.
        let (deep, never, meet) = (m.new_block(), m.new_block(), m.new_block());
        m.terminate(then, Term::CondBr(c, deep, never));
        m.terminate(deep, Term::Br(meet));
        m.call(never, writes, vec![y, y]);
        m.terminate(never, Term::Br(meet));
        let p = m.phi(meet, i32t);
        m.set_phi(p, vec![(x, deep), (y, never)]);
        m.call(meet, writes, vec![p, p]);
        m.terminate(meet, Term::Br(join));
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(p, meet), (y, other)]);
        m.call(join, writes, vec![q, q]);
        m.optimise(ALL);
        assert!(matches!(m.blocks[then].term, Some(Term::Br(t)) if t == deep));
        assert_eq!(stores(&m, meet), [[x, x]]);
        assert!(m.blocks[meet].phis.is_empty());
        assert_eq!(body(&m, never), 0, "emptied: nothing reaches it");
        let Op::Phi(incoming) = &m.insts[m.blocks[join].phis[0] as usize].op else {
            panic!()
        };
        assert_eq!(incoming, &vec![(x, meet), (y, other)]);
        let _ = b;
        m.write();
    }

    #[test]
    fn a_read_is_reused_within_its_block_until_a_store() {
        let Diamond {
            k:
                Kit {
                    mut m,
                    b,
                    reads,
                    writes,
                    ..
                },
            x,
            y,
            then,
            ..
        } = diamond();
        let i32t = m.int(32);
        // The entry reads (1, 1) for x: again in the entry it is x; in the
        // next block it is read again.
        let one = m.i32(1);
        let again = m.call(b, reads, vec![one, one]);
        m.call(b, writes, vec![again, y]);
        let after = m.call(b, reads, vec![one, one]);
        let twice = m.call(b, reads, vec![one, one]);
        let sum = m.bin(b, Bin::Add, after, twice);
        m.call(b, writes, vec![sum, sum]);
        let below = m.call(then, reads, vec![one, one]);
        m.call(then, writes, vec![below, below]);
        // A constant buffer cannot change: one load for the function.
        let cfty = m.func_ty(i32t, vec![i32t, i32t]);
        let cb = m.function(
            "dx.op.cbufferLoadLegacy.i32",
            cfty,
            Some(Attr::ReadOnly),
            false,
        );
        let top = m.call(b, cb, vec![one, one]);
        m.call(b, writes, vec![top, top]);
        let under = m.call(then, cb, vec![one, one]);
        m.call(then, writes, vec![under, under]);
        m.optimise(REUSE | LOADS);
        let stores = stores(&m, b);
        assert_eq!(stores[0], [x, y], "the second read is the first");
        let V::Inst(sum) = stores[1][0] else { panic!() };
        assert!(
            matches!(m.insts[sum as usize].op, Op::Bin(Bin::Add, a, z) if a == after && z == after),
            "read again after the store, once"
        );
        assert_eq!(stores[2], [top, top]);
        assert_eq!(
            super::tests::stores(&m, then),
            [vec![below, below], vec![top, top]]
        );
        m.write();
    }

    #[test]
    fn a_join_of_one_value_and_two_equal_joins() {
        let Diamond {
            k: Kit { mut m, writes, .. },
            x,
            y,
            then,
            other,
            join,
            ..
        } = diamond();
        let i32t = m.int(32);
        let one = m.phi(join, i32t);
        m.set_phi(one, vec![(x, then), (x, other)]);
        let p = m.phi(join, i32t);
        m.set_phi(p, vec![(x, then), (y, other)]);
        let q = m.phi(join, i32t);
        m.set_phi(q, vec![(y, other), (x, then)]);
        m.call(join, writes, vec![one, p]);
        let sum = m.bin(join, Bin::Add, p, q);
        m.call(join, writes, vec![sum, q]);
        m.optimise(FOLD | REUSE);
        assert_eq!(
            m.blocks[join].phis,
            [match p {
                V::Inst(i) => i,
                _ => unreachable!(),
            }]
        );
        let stores = stores(&m, join);
        assert_eq!(stores[0], [x, p]);
        assert_eq!(stores[1][1], p);
        m.write();
    }

    // ---- the passes against an interpreter, on random programs ----

    /// Runs `m` on `inputs`: the stores it makes, in order (and how it
    /// ended). Reads a value before its definition ran: a panic.
    fn run(m: &Module, inputs: &[u32]) -> Vec<(String, Vec<u64>)> {
        let mut vals: Vec<Option<u64>> = vec![None; m.insts.len()];
        let mut memory = [0u64; 16];
        let mut trace = Vec::new();
        let eval = |vals: &[Option<u64>], v: V| -> u64 {
            match v {
                V::Inst(i) => vals[i as usize].expect("a value defined before its use"),
                V::Const(c) => {
                    match (&m.consts[c as usize].1, m.int_bits(m.consts[c as usize].0)) {
                        (Const::Int(x), Some(bits)) => *x as u64 & mask(bits),
                        (Const::Float(bits), _) => u64::from(*bits),
                        _ => 0,
                    }
                }
                _ => panic!("a global or function as a value"),
            }
        };
        let (mut b, mut from) = (0usize, usize::MAX);
        for _ in 0..4000 {
            let blk = &m.blocks[b];
            // Joins take their values together, before any is assigned.
            let joined: Vec<(u32, u64)> = blk
                .phis
                .iter()
                .map(|&p| {
                    let Op::Phi(incoming) = &m.insts[p as usize].op else {
                        panic!("a join")
                    };
                    let v = incoming
                        .iter()
                        .find(|x| x.1 == from)
                        .expect("a value for the edge taken");
                    (p, eval(&vals, v.0))
                })
                .collect();
            for (p, x) in joined {
                vals[p as usize] = Some(x);
            }
            for &i in &blk.body {
                let inst = &m.insts[i as usize];
                let bits = m.int_bits(inst.ty);
                let float = |x: u64| f32::from_bits(x as u32);
                let r = match &inst.op {
                    Op::Bin(op, x, y) => {
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        match bits {
                            Some(bits) => {
                                let r = match op {
                                    Bin::Add => x.wrapping_add(y),
                                    Bin::Sub => x.wrapping_sub(y),
                                    Bin::Mul => x.wrapping_mul(y),
                                    Bin::And => x & y,
                                    Bin::Or => x | y,
                                    Bin::Xor => x ^ y,
                                    Bin::Shl => x << y,
                                    Bin::LShr => x >> y,
                                    Bin::AShr => (signed(bits, x) >> y) as u64,
                                    _ => panic!("no divisions here"),
                                };
                                r & mask(bits)
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                let r = match op {
                                    Bin::Add => x + y,
                                    Bin::Sub => x - y,
                                    _ => x * y,
                                };
                                u64::from(r.to_bits())
                            }
                        }
                    }
                    Op::Cmp(p, x, y) => {
                        let wide = m.int_bits(m.type_of(*x));
                        let (x, y) = (eval(&vals, *x), eval(&vals, *y));
                        let holds = match wide {
                            Some(bits) => {
                                let (sx, sy) = (signed(bits, x), signed(bits, y));
                                match *p {
                                    pred::EQ => x == y,
                                    pred::NE => x != y,
                                    pred::IUGT => x > y,
                                    pred::IUGE => x >= y,
                                    pred::IULT => x < y,
                                    pred::IULE => x <= y,
                                    pred::SGT => sx > sy,
                                    pred::SGE => sx >= sy,
                                    pred::SLT => sx < sy,
                                    _ => sx <= sy,
                                }
                            }
                            None => {
                                let (x, y) = (float(x), float(y));
                                if x.is_nan() || y.is_nan() {
                                    p & 8 != 0
                                } else {
                                    (p & 1 != 0 && x == y)
                                        || (p & 2 != 0 && x > y)
                                        || (p & 4 != 0 && x < y)
                                }
                            }
                        };
                        u64::from(holds)
                    }
                    Op::Select(c, t, f) => {
                        if eval(&vals, *c) != 0 {
                            eval(&vals, *t)
                        } else {
                            eval(&vals, *f)
                        }
                    }
                    Op::Cast(op, x) => {
                        let from_bits = m.int_bits(m.type_of(*x));
                        let x = eval(&vals, *x);
                        match (op, from_bits, bits) {
                            (Cast::Bitcast, _, _) => x,
                            (Cast::ZExt, _, _) => x,
                            (Cast::SExt, Some(f), Some(t)) => signed(f, x) as u64 & mask(t),
                            (Cast::Trunc, _, Some(t)) => x & mask(t),
                            _ => panic!("a cast the generator does not make"),
                        }
                    }
                    Op::Call(f, args) => {
                        let a: Vec<u64> = args.iter().map(|&v| eval(&vals, v)).collect();
                        let name = m.funcs[*f as usize].name.as_str();
                        match name {
                            "dx.op.loadInput.i32" => {
                                u64::from(inputs[(a[0] ^ a[1]) as usize % inputs.len()])
                            }
                            "dx.op.binary.i32" => a[0].min(a[1]),
                            "dx.op.cbufferLoadLegacy.i32" => u64::from(
                                inputs[a[0] as usize % inputs.len()].rotate_left(a[1] as u32 & 31),
                            ),
                            "dx.op.bufferLoad.i32" => memory[(a[0] ^ a[1]) as usize % 16],
                            "dx.op.waveReadLaneFirst.i32" => a[0] ^ a[1],
                            _ => {
                                memory[a[0] as usize % 16] = a[1];
                                trace.push((name.to_owned(), a));
                                0
                            }
                        }
                    }
                    _ => panic!("an instruction the generator does not make"),
                };
                vals[i as usize] = Some(r);
            }
            from = b;
            b = match blk.term.expect("a terminator") {
                Term::Ret => {
                    trace.push(("ret".into(), Vec::new()));
                    return trace;
                }
                Term::Br(t) => t,
                Term::CondBr(c, t, f) => {
                    if eval(&vals, c) != 0 {
                        t
                    } else {
                        f
                    }
                }
            };
        }
        trace.push(("still running".into(), Vec::new()));
        trace
    }

    /// A random structured program over a few registers, as the lowering
    /// makes them: every register joined at every join, conditions tested
    /// again behind their branch, values computed again.
    struct Gen {
        k: Kit,
        rng: u64,
        regs: Vec<V>,
        conds: Vec<V>,
        cur: usize,
        input: V,
        table: V,
        wave: V,
        /// The operands of the block's last read (to read the place again,
        /// and to store to it).
        last: Option<(V, V)>,
    }

    impl Gen {
        fn next(&mut self, n: u64) -> u64 {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            (self.rng >> 11) % n
        }

        fn reg(&mut self) -> V {
            let n = self.next(self.regs.len() as u64) as usize;
            self.regs[n]
        }

        /// A register or a small constant.
        fn value(&mut self) -> V {
            match self.next(4) {
                0 => {
                    let k = [0u32, 1, 2, 31, u32::MAX, 0x8000_0000, 7][self.next(7) as usize];
                    self.k.m.i32(k)
                }
                _ => self.reg(),
            }
        }

        fn cond(&mut self) -> V {
            let c = self.cur;
            let made = match self.next(6) {
                0 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    return self.conds[n];
                }
                1 if self.conds.len() > 1 => {
                    let (a, b) = (
                        self.next(self.conds.len() as u64),
                        self.next(self.conds.len() as u64),
                    );
                    let op = [Bin::And, Bin::Or, Bin::Xor][self.next(3) as usize];
                    self.k
                        .m
                        .bin(c, op, self.conds[a as usize], self.conds[b as usize])
                }
                2 if !self.conds.is_empty() => {
                    let n = self.next(self.conds.len() as u64) as usize;
                    let t = self.k.m.bool(true);
                    self.k.m.bin(c, Bin::Xor, self.conds[n], t)
                }
                _ => {
                    let p = pred::EQ + self.next(10) as u8;
                    let (x, y) = (self.reg(), self.value());
                    self.k.m.cmp(c, p, x, y)
                }
            };
            self.conds.push(made);
            made
        }

        fn statement(&mut self) {
            let c = self.cur;
            let d = self.next(self.regs.len() as u64) as usize;
            let i32t = self.k.m.int(32);
            let f32t = self.k.m.float();
            let r = match self.next(14) {
                0..=3 => {
                    let op = [Bin::Add, Bin::Sub, Bin::Mul, Bin::And, Bin::Or, Bin::Xor]
                        [self.next(6) as usize];
                    let (x, y) = (self.value(), self.value());
                    self.k.m.bin(c, op, x, y)
                }
                4 => {
                    let op = [Bin::Shl, Bin::LShr, Bin::AShr][self.next(3) as usize];
                    let x = self.value();
                    let by = self.next(32) as u32;
                    let by = self.k.m.i32(by);
                    self.k.m.bin(c, op, x, by)
                }
                5 | 6 => {
                    // A guarded write: the old value where the guard fails.
                    let g = self.cond();
                    let x = self.value();
                    self.k.m.select(self.cur, g, x, self.regs[d])
                }
                7 => {
                    let (x, y) = (self.value(), self.value());
                    let f = [self.k.pure, self.input, self.table, self.wave][self.next(4) as usize];
                    self.k.m.call(c, f, vec![x, y])
                }
                8 | 9 => {
                    let again = self.next(2) == 0;
                    let (x, y) = match self.last {
                        Some(last) if again => last,
                        _ => (self.value(), self.value()),
                    };
                    self.last = Some((x, y));
                    self.k.m.call(c, self.k.reads, vec![x, y])
                }
                10 => {
                    // A store, half the time to the place read last.
                    let (mut x, y) = (self.value(), self.value());
                    let there = self.next(2) == 0;
                    if let Some((a, b)) = self.last
                        && there
                    {
                        x = self.k.m.bin(c, Bin::Xor, a, b);
                    }
                    self.k.m.call(c, self.k.writes, vec![x, y]);
                    return;
                }
                11 => {
                    // Float arithmetic, read back as bits.
                    let (x, y) = (self.value(), self.value());
                    let (x, y) = (
                        self.k.m.cast(c, Cast::Bitcast, x, f32t),
                        self.k.m.cast(c, Cast::Bitcast, y, f32t),
                    );
                    let op = [Bin::Add, Bin::Sub, Bin::Mul][self.next(3) as usize];
                    let f = self.k.m.bin(c, op, x, y);
                    if self.next(2) == 0 {
                        let p = self.next(16) as u8;
                        let t = self.k.m.cmp(c, p, f, y);
                        self.conds.push(t);
                    }
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
                12 => {
                    let g = self.cond();
                    self.k.m.cast(self.cur, Cast::ZExt, g, i32t)
                }
                _ => {
                    let x = self.value();
                    let f = self.k.m.cast(c, Cast::Bitcast, x, f32t);
                    self.k.m.cast(c, Cast::Bitcast, f, i32t)
                }
            };
            self.regs[d] = r;
        }

        fn code(&mut self, depth: u32) {
            for _ in 0..1 + self.next(6) {
                match self.next(8) {
                    0 | 1 if depth > 0 => self.branch(depth - 1),
                    2 if depth > 0 => self.repeat(depth - 1),
                    _ => self.statement(),
                }
            }
        }

        /// if / else, every register joined.
        fn branch(&mut self, depth: u32) {
            let c = self.cond();
            let i32t = self.k.m.int(32);
            let (then, other, join) = (
                self.k.m.new_block(),
                self.k.m.new_block(),
                self.k.m.new_block(),
            );
            self.k.m.terminate(self.cur, Term::CondBr(c, then, other));
            let (before, known) = (self.regs.clone(), self.conds.len());
            self.cur = then;
            self.last = None;
            self.code(depth);
            let (then_regs, then_end) = (std::mem::replace(&mut self.regs, before), self.cur);
            self.k.m.terminate(then_end, Term::Br(join));
            self.conds.truncate(known);
            self.cur = other;
            self.last = None;
            if self.next(3) != 0 {
                self.code(depth);
            }
            let other_end = self.cur;
            self.k.m.terminate(other_end, Term::Br(join));
            self.conds.truncate(known);
            for (r, &t) in then_regs.iter().enumerate() {
                let p = self.k.m.phi(join, i32t);
                self.k
                    .m
                    .set_phi(p, vec![(t, then_end), (self.regs[r], other_end)]);
                self.regs[r] = p;
            }
            self.cur = join;
            self.last = None;
        }

        /// A loop that runs at most a few times, every register joined at
        /// its head.
        fn repeat(&mut self, depth: u32) {
            let i32t = self.k.m.int(32);
            let (head, exit) = (self.k.m.new_block(), self.k.m.new_block());
            let before = self.cur;
            self.k.m.terminate(before, Term::Br(head));
            let entering = self.regs.clone();
            let joins: Vec<V> = (0..self.regs.len())
                .map(|_| self.k.m.phi(head, i32t))
                .collect();
            let count = self.k.m.phi(head, i32t);
            self.regs = joins.clone();
            let known = self.conds.len();
            self.cur = head;
            self.last = None;
            self.code(depth);
            let one = self.k.m.i32(1);
            let next = self.k.m.bin(self.cur, Bin::Add, count, one);
            let limit = 1 + self.next(3) as u32;
            let limit = self.k.m.i32(limit);
            let mut more = self.k.m.cmp(self.cur, pred::IULT, next, limit);
            if self.next(2) == 0 {
                let c = self.cond();
                more = self.k.m.bin(self.cur, Bin::And, more, c);
            }
            let latch = self.cur;
            self.k.m.terminate(latch, Term::CondBr(more, head, exit));
            for (r, &p) in joins.iter().enumerate() {
                self.k
                    .m
                    .set_phi(p, vec![(entering[r], before), (self.regs[r], latch)]);
            }
            let zero = self.k.m.i32(0);
            self.k.m.set_phi(count, vec![(zero, before), (next, latch)]);
            self.conds.truncate(known);
            self.cur = exit;
            self.last = None;
        }
    }

    fn random_program(seed: u64) -> Module {
        let mut k = kit();
        let i32t = k.m.int(32);
        let fty = k.m.func_ty(i32t, vec![i32t, i32t]);
        let input =
            k.m.function("dx.op.loadInput.i32", fty, Some(Attr::ReadNone), false);
        let table = k.m.function(
            "dx.op.cbufferLoadLegacy.i32",
            fty,
            Some(Attr::ReadOnly),
            false,
        );
        let wave = k.m.function(
            "dx.op.waveReadLaneFirst.i32",
            fty,
            Some(Attr::NoUnwind),
            false,
        );
        let b = k.b;
        let mut g = Gen {
            k,
            rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1,
            regs: Vec::new(),
            conds: Vec::new(),
            cur: b,
            input,
            table,
            wave,
            last: None,
        };
        for r in 0..5 {
            let (x, y) = (g.k.m.i32(r), g.k.m.i32(r * 3));
            let v = g.k.m.call(b, input, vec![x, y]);
            g.regs.push(v);
        }
        g.code(3);
        g.code(2);
        // Every register is read at the end.
        for r in 0..g.regs.len() {
            let at = g.k.m.i32(r as u32);
            g.k.m.call(g.cur, g.k.writes, vec![at, g.regs[r]]);
        }
        g.k.m.terminate(g.cur, Term::Ret);
        g.k.m
    }

    #[test]
    fn every_pass_leaves_what_random_programs_store() {
        let mut removed = 0;
        for seed in 1..=400u64 {
            let plain = random_program(seed);
            let inputs: Vec<Vec<u32>> = (0..6u32)
                .map(|n| {
                    (0..7u32)
                        .map(|i| match n {
                            0 => 0,
                            1 => i,
                            2 => u32::MAX - i,
                            _ => {
                                (seed as u32 ^ n)
                                    .wrapping_mul(0x9E37_79B9)
                                    .rotate_left(i * 5)
                                    ^ (i << 28)
                            }
                        })
                        .collect()
                })
                .collect();
            let expected: Vec<_> = inputs.iter().map(|i| run(&plain, i)).collect();
            let written: usize = plain
                .blocks
                .iter()
                .map(|b| b.body.len() + b.phis.len())
                .sum();
            for passes in [
                DEAD,
                FOLD,
                REUSE,
                FACTS,
                LOADS,
                FOLD | FACTS,
                FOLD | REUSE | DEAD,
                ALL,
            ] {
                let mut m = random_program(seed);
                m.optimise(passes);
                for (i, want) in inputs.iter().zip(&expected) {
                    assert_eq!(&run(&m, i), want, "seed {seed}, passes {passes}");
                }
                if passes == ALL {
                    let left: usize = m.blocks.iter().map(|b| b.body.len() + b.phis.len()).sum();
                    removed += written - left;
                }
                m.write();
            }
        }
        // The programs are of the wasteful kind: the passes find work.
        assert!(removed > 4000, "{removed} instructions removed");
    }

    #[test]
    fn rayquery_effects_and_usage_are_tracked() {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i32t = m.int(32);
        let alloc_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let proceed_ty = m.func_ty(i1, vec![i32t, i32t]);
        let state_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [alloc_ty, proceed_ty, state_ty, store_ty, main_ty] {
            m.ptr(ty);
        }
        let alloc = m.function("dx.op.allocateRayQuery", alloc_ty, Some(Attr::NoUnwind), false);
        let proceed = m.function(
            "dx.op.rayQuery_Proceed.i1",
            proceed_ty,
            Some(Attr::NoUnwind),
            false,
        );
        let state = m.function(
            "dx.op.rayQuery_StateScalar.i32",
            state_ty,
            Some(Attr::ReadOnly),
            false,
        );
        let store = m.function("dx.op.rawBufferStore.i32", store_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);
        let b = m.new_block();
        let op_alloc = m.i32(178);
        let op_proceed = m.i32(180);
        let op_status = m.i32(184);
        let zero = m.i32(0);
        let rq1 = m.call(b, alloc, vec![op_alloc, zero]);
        let rq2 = m.call(b, alloc, vec![op_alloc, zero]);
        let s_before = m.call(b, state, vec![op_status, rq1]);
        let s_same = m.call(b, state, vec![op_status, rq1]);
        let _p = m.call(b, proceed, vec![op_proceed, rq1]);
        let s_after = m.call(b, state, vec![op_status, rq1]);
        let _dead_state = m.call(b, state, vec![op_status, rq2]);
        let sum = m.bin(b, Bin::Add, s_before, s_same);
        m.call(b, store, vec![sum, s_after]);
        m.terminate(b, Term::Ret);
        m.optimise(MODULE);
        assert!(m.usage().raytracing_tier_1_1);
        // Two distinct allocateRayQuery calls, one proceed, two kept state reads
        // (before and after proceed; s_same reused s_before, _dead_state dropped),
        // one add, and one store = 7 instructions.
        assert_eq!(body(&m, b), 7);
        let s = stores(&m, b);
        let V::Inst(add_i) = s[0][0] else { panic!() };
        assert!(
            matches!(m.insts[add_i as usize].op, Op::Bin(Bin::Add, x, y) if x == s_before && y == s_before),
            "identical state reads before proceed merge"
        );
        assert_ne!(s[0][1], s_before, "proceed invalidates cached state reads");
        m.write();
    }

    #[test]
    fn extended_bounded_and_post_fold_legacy_mul_elision() {
        let mut m = Module::default();
        let void = m.void();
        let i32t = m.int(32);
        let f32t = m.float();
        let u1_ty = m.func_ty(f32t, vec![i32t, f32t]);
        let b2_ty = m.func_ty(f32t, vec![i32t, f32t, f32t]);
        let rd_ty = m.func_ty(f32t, vec![i32t, i32t]);
        let st_ty = m.func_ty(void, vec![f32t, f32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [u1_ty, b2_ty, rd_ty, st_ty, main_ty] {
            m.ptr(ty);
        }
        let unary = m.function("dx.op.unary.f32", u1_ty, Some(Attr::ReadNone), false);
        let binary = m.function("dx.op.binary.f32", b2_ty, Some(Attr::ReadNone), false);
        let read = m.function("dx.op.loadInput.f32", rd_ty, Some(Attr::ReadNone), false);
        let store = m.function("dx.op.bufferStore.f32", st_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let entry = m.new_block();
        let body_blk = m.new_block();
        let (k0, k1, k6, k7, k35, k36) = (
            m.i32(0),
            m.i32(1),
            m.i32(6),
            m.i32(7),
            m.i32(35),
            m.i32(36),
        );
        let zero = m.f32(0);
        let one = m.f32(1.0f32.to_bits());
        let neg_two = m.f32((-2.0f32).to_bits());
        let sign_bit = m.i32(0x8000_0000);
        let c_true = m.bool(true);

        // Unbounded input x, plus a norm_quotient-marked value through sign-bit Or + bitcast + Select.
        let x = m.call(entry, read, vec![k0, k0]);
        let y_unk = m.call(entry, read, vec![k0, k1]);
        let norm_i = m.cast(entry, Cast::Bitcast, x, i32t);
        m.mark_unit_float(norm_i);
        let signed_i = m.bin(entry, Bin::Or, norm_i, sign_bit);
        let one_i = m.i32(1.0f32.to_bits());
        let sel_i = m.select(entry, c_true, signed_i, one_i);
        let from_norm = m.cast(entry, Cast::Bitcast, sel_i, f32t);
        assert!(m.bounded_ext(from_norm, 20, true).is_some());
        assert!(m.bounded_ext(from_norm, 12, false).is_none());

        // Clamped FMin(bitcast(bitcast(FMax(x, -2.0))), 1.0) and saturate(x).
        let fmax = m.call(entry, binary, vec![k35, x, neg_two]);
        let fmax_i = m.cast(entry, Cast::Bitcast, fmax, i32t);
        let fmax_f = m.cast(entry, Cast::Bitcast, fmax_i, f32t);
        let clamped = m.call(entry, binary, vec![k36, fmax_f, one]);
        let sat = m.call(entry, unary, vec![k7, x]);
        assert!(m.bounded_ext(clamped, 20, true).is_some());
        assert!(m.bounded_ext(sat, 20, true).is_some());

        // Pass `from_norm` through a 1-incoming phi so pre-fold `bounded()` misses it,
        // then emit the full 5-op legacy_mul guard in `body_blk`.
        m.terminate(entry, Term::Br(body_blk));
        let phi_a = m.phi(body_blk, f32t);
        let phi_b = m.phi(body_blk, f32t);
        m.set_phi(phi_a, vec![(from_norm, entry)]);
        m.set_phi(phi_b, vec![(clamped, entry)]);

        // Guard 1: both operands become bounded after phi folding -> collapses to plain Mul!
        let prod1 = m.bin(body_blk, Bin::Mul, phi_a, phi_b);
        let abs1a = m.call(body_blk, unary, vec![k6, phi_a]);
        let abs1b = m.call(body_blk, unary, vec![k6, phi_b]);
        let fmin1 = m.call(body_blk, binary, vec![k36, abs1a, abs1b]);
        let cmp1 = m.cmp(body_blk, pred::OEQ, fmin1, zero);
        let guard1 = m.select(body_blk, cmp1, zero, prod1);

        // Guard 2: one operand becomes bounded (phi_a), one stays unknown (y_unk) -> narrows to 2-op!
        let prod2 = m.bin(body_blk, Bin::Mul, phi_a, y_unk);
        let abs2a = m.call(body_blk, unary, vec![k6, phi_a]);
        let abs2b = m.call(body_blk, unary, vec![k6, y_unk]);
        let fmin2 = m.call(body_blk, binary, vec![k36, abs2a, abs2b]);
        let cmp2 = m.cmp(body_blk, pred::OEQ, fmin2, zero);
        let guard2 = m.select(body_blk, cmp2, zero, prod2);

        m.call(body_blk, store, vec![guard1, guard2]);
        m.terminate(body_blk, Term::Ret);

        m.optimise(MODULE);
        assert!(m.elide_legacy_mul_guards(), "post-fold pass must simplify both guards");
        m.optimise(MODULE);

        let st = stores(&m, body_blk);
        // Guard 1 collapsed directly to Mul(from_norm, clamped).
        let V::Inst(g1) = st[0][0] else { panic!() };
        assert!(
            matches!(m.insts[g1 as usize].op, Op::Bin(Bin::Mul, _, _)),
            "guard1 must collapse to plain Bin::Mul, got {:?}",
            m.insts[g1 as usize].op
        );
        // Guard 2 narrowed its Cmp to OEQ(from_norm, 0.0) without FAbs/FMin.
        let V::Inst(g2) = st[0][1] else { panic!() };
        let Op::Select(V::Inst(c2), _, _) = m.insts[g2 as usize].op else {
            panic!("guard2 should remain a 2-op select");
        };
        assert!(
            matches!(m.insts[c2 as usize].op, Op::Cmp(pred::OEQ, op, _) if op != fmin2),
            "guard2 cmp must narrow away FMin(FAbs, FAbs), got {:?}",
            m.insts[c2 as usize].op
        );
        // No FAbs calls should remain in body_blk.
        let has_fabs = m.blocks[body_blk].body.iter().any(|&i| {
            matches!(&m.insts[i as usize].op, Op::Call(f, args) if m.funcs[*f as usize].name == "dx.op.unary.f32" && m.int_k(args[0]) == Some((32, 6)))
        });
        assert!(!has_fabs, "all FAbs calls in body_blk should be dead and removed");
        m.write();
    }

    #[test]
    fn binary_i32_minmax_folds_and_is_helper_lane_is_pure() {
        let mut m = Module::default();
        let void = m.void();
        let i1t = m.int(1);
        let i32t = m.int(32);
        let b2_ty = m.func_ty(i32t, vec![i32t, i32t, i32t]);
        let rd_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let hl_ty = m.func_ty(i1t, vec![i32t]);
        let st_ty = m.func_ty(void, vec![i32t, i32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [b2_ty, rd_ty, hl_ty, st_ty, main_ty] {
            m.ptr(ty);
        }
        let binary = m.function("dx.op.binary.i32", b2_ty, Some(Attr::ReadNone), false);
        let read = m.function("dx.op.bufferLoad.i32", rd_ty, Some(Attr::ReadOnly), false);
        let helper = m.function("dx.op.isHelperLane.i1", hl_ty, Some(Attr::ReadOnly), false);
        let store = m.function("dx.op.bufferStore.i32", st_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let b = m.new_block();
        let (k37, k38, k39, k40) = (m.i32(37), m.i32(38), m.i32(39), m.i32(40));
        let (k0, k5, k10, k221) = (m.i32(0), m.i32(5), m.i32(10), m.i32(221));
        let neg1 = m.i32(u32::MAX);
        let x = m.call(b, read, vec![k0, k5]);

        // Constant UMin(5, 10) -> 5; IMax(-1, 5) -> 5; UMax(-1, 5) -> -1; IMin(-1, 5) -> -1.
        let c_umin = m.call(b, binary, vec![k40, k5, k10]);
        let c_imax = m.call(b, binary, vec![k37, neg1, k5]);
        let c_umax = m.call(b, binary, vec![k39, neg1, k5]);
        let c_imin = m.call(b, binary, vec![k38, neg1, k5]);
        // Idempotent UMin(x, x) -> x, and bound UMin(x, u32::MAX) -> x.
        let idem = m.call(b, binary, vec![k40, x, x]);
        let bound = m.call(b, binary, vec![k40, idem, neg1]);

        // Two isHelperLane calls separated by a store: Effect::Pure merges them!
        let h1 = m.call(b, helper, vec![k221]);
        let h1_i = m.cast(b, Cast::ZExt, h1, i32t);
        m.call(b, store, vec![c_umin, c_imax]);
        m.call(b, store, vec![c_umax, c_imin]);
        let h2 = m.call(b, helper, vec![k221]);
        let h2_i = m.cast(b, Cast::ZExt, h2, i32t);
        let hsum = m.bin(b, Bin::Add, h1_i, h2_i);
        m.call(b, store, vec![bound, hsum]);
        m.terminate(b, Term::Ret);

        let prev = crate::emit::lower::TEST_RDNA_FP16_OPT.replace(Some(true));
        m.optimise(MODULE);
        crate::emit::lower::TEST_RDNA_FP16_OPT.set(prev);

        let st = stores(&m, b);
        assert_eq!(st[0], vec![k5, k5]);
        assert_eq!(st[1], vec![neg1, neg1]);
        assert_eq!(st[2][0], x, "idempotent and max-bound UMin must fold to x");
        let helper_calls = m.blocks[b]
            .body
            .iter()
            .filter(|&&i| {
                matches!(&m.insts[i as usize].op, Op::Call(f, _) if m.funcs[*f as usize].name == "dx.op.isHelperLane.i1")
            })
            .count();
        assert_eq!(helper_calls, 1, "isHelperLane must deduplicate across stores as Effect::Pure");
        m.write();
    }

    #[test]
    fn rayquery_cross_block_cse_respects_mutations_and_ignores_memory_stores() {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i32t = m.int(32);
        let alloc_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let proceed_ty = m.func_ty(i1, vec![i32t, i32t]);
        let state_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [alloc_ty, proceed_ty, state_ty, store_ty, main_ty] {
            m.ptr(ty);
        }
        let alloc = m.function("dx.op.allocateRayQuery", alloc_ty, Some(Attr::NoUnwind), false);
        let proceed = m.function(
            "dx.op.rayQuery_Proceed.i1",
            proceed_ty,
            Some(Attr::NoUnwind),
            false,
        );
        let state = m.function(
            "dx.op.rayQuery_StateScalar.i32",
            state_ty,
            Some(Attr::ReadOnly),
            false,
        );
        let store = m.function("dx.op.rawBufferStore.i32", store_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let entry = m.new_block();
        let b_clean = m.new_block();
        let b_mut = m.new_block();
        let b_join = m.new_block();
        let b_loop = m.new_block();
        let b_exit = m.new_block();

        let (k178, k180, k209, k0, k1) = (m.i32(178), m.i32(180), m.i32(209), m.i32(0), m.i32(1));
        let rq = m.call(entry, alloc, vec![k178, k0]);
        let g_entry = m.call(entry, state, vec![k209, rq]);
        // A memory store does NOT invalidate RayQuery state reads.
        m.call(entry, store, vec![g_entry, k0]);
        let g_after_store = m.call(entry, state, vec![k209, rq]);
        let cond = m.cmp(entry, pred::NE, g_after_store, k0);
        m.terminate(entry, Term::CondBr(cond, b_clean, b_mut));

        // b_clean: dominated by entry with no RayQueryMut -> must reuse g_entry.
        let g_clean = m.call(b_clean, state, vec![k209, rq]);
        m.call(b_clean, store, vec![g_clean, k1]);
        m.terminate(b_clean, Term::Br(b_join));

        // b_mut: mutates rq -> join must NOT reuse g_entry.
        let _p1 = m.call(b_mut, proceed, vec![k180, rq]);
        m.terminate(b_mut, Term::Br(b_join));

        let g_join = m.call(b_join, state, vec![k209, rq]);
        m.call(b_join, store, vec![g_join, k1]);
        m.terminate(b_join, Term::Br(b_loop));

        // b_loop: reads state BEFORE proceed on back-edge -> must NOT reuse g_join.
        let g_loop = m.call(b_loop, state, vec![k209, rq]);
        m.call(b_loop, store, vec![g_loop, k0]);
        let cont = m.call(b_loop, proceed, vec![k180, rq]);
        m.terminate(b_loop, Term::CondBr(cont, b_loop, b_exit));
        m.terminate(b_exit, Term::Ret);

        m.optimise(MODULE);
        assert_eq!(stores(&m, b_clean)[0][0], g_entry, "b_clean must reuse entry read across store");
        assert_ne!(stores(&m, b_join)[0][0], g_entry, "b_join must not reuse across b_mut proceed");
        assert_ne!(stores(&m, b_loop)[0][0], g_join, "loop header must not reuse across self-loop proceed");
        let entry_states = m.blocks[entry]
            .body
            .iter()
            .filter(|&&i| matches!(&m.insts[i as usize].op, Op::Call(f, _) if m.funcs[*f as usize].name == "dx.op.rayQuery_StateScalar.i32"))
            .count();
        assert_eq!(entry_states, 1, "store between two state reads must not prevent reuse");
        m.write();
    }

    #[test]
    fn rayquery_bitfields_bool_selects_and_empty_diamonds_fold() {
        let mut m = Module::default();
        let void = m.void();
        let i32t = m.int(32);
        let state_ty = m.func_ty(i32t, vec![i32t, i32t]);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let main_ty = m.func_ty(void, vec![]);
        for ty in [state_ty, store_ty, main_ty] {
            m.ptr(ty);
        }
        let state = m.function(
            "dx.op.rayQuery_StateScalar.i32",
            state_ty,
            Some(Attr::ReadOnly),
            false,
        );
        let store = m.function("dx.op.bufferStore.i32", store_ty, Some(Attr::NoUnwind), false);
        m.function("main", main_ty, None, true);

        let entry = m.new_block();
        let d_left = m.new_block();
        let d_right = m.new_block();
        let d_join = m.new_block();

        let (k0, k8, k209, k215, mask24) = (
            m.i32(0),
            m.i32(8),
            m.i32(209), // COMMITTED_GEOMETRY_INDEX
            m.i32(215), // COMMITTED_INSTANCE_CONTRIBUTION
            m.i32(0x00FF_FFFF),
        );
        let (t1, f1) = (m.bool(true), m.bool(false));

        let geom = m.call(entry, state, vec![k209, k0]);
        let shl = m.bin(entry, Bin::Shl, geom, k8);
        let lshr = m.bin(entry, Bin::LShr, shl, k8);
        let contrib = m.call(entry, state, vec![k215, k0]);
        let masked = m.bin(entry, Bin::And, contrib, mask24);

        let c = m.cmp(entry, pred::NE, geom, k0);
        let not_c = m.bin(entry, Bin::Xor, c, t1);
        // select(c, !c, true) -> !c; select(c, false, !c) -> !c; select(c, !c, false) -> false
        let s1 = m.select(entry, c, not_c, t1);
        let s2 = m.select(entry, c, f1, not_c);
        let s3 = m.select(entry, c, not_c, f1);
        let z1 = m.cast(entry, Cast::ZExt, s1, i32t);
        let z2 = m.cast(entry, Cast::ZExt, s2, i32t);
        let z3 = m.cast(entry, Cast::ZExt, s3, i32t);

        // Symmetric empty diamond whose join phi receives the same value on both arms.
        m.terminate(entry, Term::CondBr(c, d_left, d_right));
        m.terminate(d_left, Term::Br(d_join));
        m.terminate(d_right, Term::Br(d_join));
        let p = m.phi(d_join, i32t);
        m.set_phi(p, vec![(lshr, d_left), (lshr, d_right)]);
        m.call(d_join, store, vec![p, masked]);
        let sum = m.bin(d_join, Bin::Add, z1, z2);
        m.call(d_join, store, vec![sum, z3]);
        m.terminate(d_join, Term::Ret);

        m.optimise(MODULE);
        assert!(
            matches!(m.blocks[entry].term, Some(Term::Br(t)) if t == d_join),
            "empty diamond must collapse"
        );
        let st = stores(&m, d_join);
        assert_eq!(st[0][0], geom, "(GeometryIndex << 8) >> 8 must fold to GeometryIndex");
        assert_eq!(st[0][1], contrib, "InstanceContribution & 0xFFFFFF must fold to InstanceContribution");
        assert_eq!(st[1][1], k0, "select(c, !c, false) must fold to 0");
        let V::Inst(sum_i) = st[1][0] else { panic!() };
        assert!(
            matches!(m.insts[sum_i as usize].op, Op::Bin(Bin::Add, a, b) if a == b),
            "select(c, !c, true) and select(c, false, !c) must both fold to !c and CSE"
        );
        m.write();
    }

    #[test]
    fn licm_hoists_stride_srv_and_readonly_header_loads_but_not_uav_or_written_loops() {
        let mut s = crate::emit::shader::Shader::new(false);
        let entry = s.entry;
        let pre = s.m.new_block();
        let hdr = s.m.new_block();
        let inner = s.m.new_block();
        let exit = s.m.new_block();
        let i32t = s.t.i32;

        let stride_h = s.root_srv_handle(crate::emit::shader::SLOT_STRIDES);
        let ro_h = s.root_srv_handle(0);
        let uav_h = s.root_uav_handle(0);
        let idx = s.op(entry, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        s.m.terminate(entry, Term::Br(pre));
        s.m.terminate(pre, Term::Br(hdr));

        let cnt = s.m.phi(hdr, i32t);
        let acc = s.m.phi(hdr, i32t);
        let zero = s.m.i32(0);
        let one = s.m.i32(1);
        let four = s.m.i32(4);
        let undef = s.m.undef(i32t);
        let mask1 = s.m.i8(1);
        let align4 = s.m.i32(4);
        let c139 = s.m.i32(139);
        let i8t = s.m.int(8);
        let res_i32 = s
            .m
            .named_struct("dx.types.ResRet.i32", vec![i32t, i32t, i32t, i32t, i32t]);
        let load_ty = s.m.func_ty(
            res_i32,
            vec![i32t, s.t.handle, i32t, i32t, i8t, i32t],
        );
        s.m.ptr(load_ty);
        let raw_load = s.m.function("dx.op.rawBufferLoad.i32", load_ty, Some(Attr::ReadOnly), false);

        // Must-execute read-only SRV load in header + UAV load in header (must NOT hoist!).
        let c2 = s.m.i32(2);
        let byte_off = s.m.bin(hdr, Bin::Shl, idx, c2);
        let ro_ret = s.m.call(hdr, raw_load, vec![c139, ro_h, byte_off, undef, mask1, align4]);
        let ro_val = s.m.extract(hdr, ro_ret, 0);
        let uav_ret = s.m.call(hdr, raw_load, vec![c139, uav_h, byte_off, undef, mask1, align4]);
        let uav_val = s.m.extract(hdr, uav_ret, 0);
        let cond = s.m.cmp(hdr, pred::IULT, cnt, four);
        s.m.terminate(hdr, Term::CondBr(cond, inner, exit));

        // Conditional block inside loop reading t255 stride table -> MUST hoist to `pre`!
        let st_off = s.m.bin(inner, Bin::Shl, idx, c2);
        let st_ret = s.m.call(inner, raw_load, vec![c139, stride_h, st_off, undef, mask1, align4]);
        let st_val = s.m.extract(inner, st_ret, 0);
        let sum1 = s.m.bin(inner, Bin::Add, acc, ro_val);
        let sum2 = s.m.bin(inner, Bin::Add, sum1, uav_val);
        let sum3 = s.m.bin(inner, Bin::Add, sum2, st_val);
        let next = s.m.bin(inner, Bin::Add, cnt, one);
        s.m.terminate(inner, Term::Br(hdr));
        s.m.set_phi(cnt, vec![(zero, pre), (next, inner)]);
        s.m.set_phi(acc, vec![(zero, pre), (sum3, inner)]);

        // Store final acc in exit so everything stays live.
        let store_ty = s.m.func_ty(
            s.t.void,
            vec![i32t, s.t.handle, i32t, i32t, i32t, i32t, i32t, i32t, i8t, i32t],
        );
        s.m.ptr(store_ty);
        let raw_store = s.m.function("dx.op.rawBufferStore.i32", store_ty, Some(Attr::NoUnwind), false);
        let c140 = s.m.i32(140);
        s.m.call(
            exit,
            raw_store,
            vec![c140, uav_h, zero, undef, acc, undef, undef, undef, mask1, align4],
        );
        s.m.terminate(exit, Term::Ret);

        s.m.optimise(MODULE);
        let count_loads = |blk: usize| {
            s.m.blocks[blk]
                .body
                .iter()
                .filter(|&&i| {
                    matches!(&s.m.insts[i as usize].op, Op::Call(f, _) if s.m.funcs[*f as usize].name == "dx.op.rawBufferLoad.i32")
                })
                .count()
        };
        assert_eq!(count_loads(pre), 2, "t255 stride load + header read-only SRV load must hoist to preheader");
        assert_eq!(count_loads(hdr), 1, "UAV load must remain in header");
        assert_eq!(count_loads(inner), 0, "t255 stride load must leave inner block");
        s.m.write();
    }
}
