//! Certified discrete Gaussian sampler (paper theory source `sampling`,
//! "Bounded scalar Gaussian arithmetic and exact first-stage certificates").
//!
//! A port of the packet's `bounded_gaussian_reference/reference.py`
//! (fallback stages, correction) and `fast_gaussian_certificate/
//! certificate.py` (profile 1: the recommended P128 first stage). The target
//! is D_σ(k) ∝ exp(-k²/(2σ²)) for integer 1 <= σ <= 2^40.
//!
//! Proposal: X = 2V1-1, Y = 2V2-1, S = X²+Y², Z = σX·sqrt(-2 log S / S),
//! rounded to the nearest integer; then accept with probability
//! a_k = e^{-b}/I_k. Every decision is made by exact integer comparisons on
//! outward-rounded dyadic intervals, so it holds for every continuation of
//! the retained bit prefixes. Ambiguity extends the same prefixes; it never
//! draws fresh randomness. Cap failures are explicit `Abort`s.
//!
//! The fast stage certifies |k| = m by
//!     (2m-1)² S_hi < T_lo  and  T_hi < (2m+1)² S_lo,   T = -8σ²X² log S,
//! omitting the lower test for m = 0, with no square root and no division by S.

use super::coins::{BitStream, CoinsExhausted};
use num_bigint::BigInt;
use num_integer::{Integer, Roots};
#[allow(unused_imports)]
use num_traits::{One, Signed, ToPrimitive, Zero};
use once_cell::sync::Lazy;
use std::collections::HashMap;

type Z = BigInt;

/// Terminal guard width g (α = 2^-g).
pub const G: usize = 8704;
pub const BMAX: usize = 26172;
pub const PMAX: u32 = 65536;
pub const LOGMAX: u32 = 8725;
pub const IMAX: u32 = 16384;
pub const EMAX: u32 = 4096;
pub const OUTER: u32 = 3072;
pub const TRIALS: u32 = 8192;

/// First-stage profiles (coordinate bits, P, log terms); index 1 is the
/// recommended P128 certificate.
pub const FAST_PROFILES: [(usize, u32, u32); 2] = [(64, 96, 32), (80, 128, 43)];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Abort {
    UniformEndpointGuard,
    PolarSGuard,
    RoundingUnresolved,
    PolarTrialCap,
    CorrectionComparisonCap,
    OuterProposalCap,
    Coins(CoinsExhausted),
}

impl From<CoinsExhausted> for Abort {
    fn from(e: CoinsExhausted) -> Self { Abort::Coins(e) }
}

#[inline]
fn floor_shr(a: &Z, p: u32) -> Z { a.div_floor(&(Z::one() << p)) }
#[inline]
fn ceil_shr(a: &Z, p: u32) -> Z { -floor_shr(&-a, p) }
#[inline]
fn cdiv(a: &Z, b: &Z) -> Z { -((-a).div_floor(b)) }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iv {
    pub lo: Z,
    pub hi: Z,
}

/// Append-only uniform prefix: value in [prefix, prefix+1) · 2^-bits.
pub struct LazyUniform {
    src: Box<dyn BitStream>,
    pub prefix: Z,
    pub bits: usize,
}

impl LazyUniform {
    pub fn new(src: Box<dyn BitStream>) -> Self { LazyUniform { src, prefix: Z::zero(), bits: 0 } }
    pub fn refine(&mut self, bits: usize) -> Result<(), Abort> {
        assert!(self.bits <= bits && bits <= BMAX, "append-only uniform prefix cap");
        while self.bits < bits {
            let b = self.src.next_bit()?;
            self.prefix = (&self.prefix << 1u32) + Z::from(b);
            self.bits += 1;
        }
        Ok(())
    }
    fn endpoint_guard_known_safe(&self) -> bool {
        !self.prefix.is_zero() && self.prefix != (Z::one() << self.bits) - 1
    }
    fn terminal_endpoint_bad(&self) -> bool {
        if self.bits < G { return false; }
        let head: Z = &self.prefix >> (self.bits - G);
        head.is_zero() || head == (Z::one() << G) - 1
    }
}

/// Dyadic interval arithmetic on the grid 2^-p.
pub struct Dyadic {
    pub p: u32,
    one: Z,
    ln2: Option<(u32, Iv)>,
}

impl Dyadic {
    pub fn new(p: u32) -> Self {
        assert!((32..=PMAX).contains(&p), "precision outside public allocation");
        Dyadic { p, one: Z::one() << p, ln2: None }
    }
    fn iv(&self, lo: Z, hi: Z) -> Iv {
        assert!(lo <= hi, "reversed interval");
        debug_assert!(lo.bits().max(hi.bits()) <= (self.p + 9000) as u64, "endpoint magnitude guard");
        Iv { lo, hi }
    }
    fn point(&self, v: Z) -> Iv { Iv { lo: v.clone(), hi: v } }
    fn integer(&self, k: i64) -> Iv { self.point(&self.one * k) }
    fn rational(&self, n: &Z, d: &Z) -> Iv {
        assert!(d.is_positive());
        let v = &self.one * n;
        self.iv(v.div_floor(d), cdiv(&v, d))
    }
    fn add(&self, a: &Iv, b: &Iv) -> Iv { self.iv(&a.lo + &b.lo, &a.hi + &b.hi) }
    fn neg(&self, a: &Iv) -> Iv { self.iv(-&a.hi, -&a.lo) }
    fn sub(&self, a: &Iv, b: &Iv) -> Iv { self.iv(&a.lo - &b.hi, &a.hi - &b.lo) }
    fn scale(&self, a: &Iv, k: &Z) -> Iv {
        if !k.is_negative() { self.iv(&a.lo * k, &a.hi * k) } else { self.iv(&a.hi * k, &a.lo * k) }
    }
    fn div_int(&self, a: &Iv, k: &Z) -> Iv {
        assert!(k.is_positive());
        self.iv(a.lo.div_floor(k), cdiv(&a.hi, k))
    }
    fn mul(&self, a: &Iv, b: &Iv) -> Iv {
        let mut low = &a.lo * &b.lo;
        let mut high = low.clone();
        for (x, y) in [(&a.lo, &b.hi), (&a.hi, &b.lo), (&a.hi, &b.hi)] {
            let v = x * y;
            if v < low { low = v.clone(); }
            if v > high { high = v; }
        }
        self.iv(floor_shr(&low, self.p), ceil_shr(&high, self.p))
    }
    fn square(&self, a: &Iv) -> Iv {
        let l = &a.lo * &a.lo;
        let h = &a.hi * &a.hi;
        let low = if !a.lo.is_positive() && !a.hi.is_negative() { Z::zero() } else { l.clone().min(h.clone()) };
        let high = l.max(h);
        self.iv(floor_shr(&low, self.p), ceil_shr(&high, self.p))
    }
    fn positive_mul(&self, a: &Iv, b: &Iv) -> Iv {
        assert!(!a.lo.is_negative() && !b.lo.is_negative(), "nonnegative operands required");
        self.iv(floor_shr(&(&a.lo * &b.lo), self.p), ceil_shr(&(&a.hi * &b.hi), self.p))
    }
    fn div_positive(&self, a: &Iv, b: &Iv) -> Iv {
        assert!(!a.lo.is_negative() && b.lo.is_positive(), "positive interval division required");
        self.iv((&a.lo << self.p).div_floor(&b.hi), cdiv(&(&a.hi << self.p), &b.lo))
    }
    fn sqrt(&self, a: &Iv) -> Iv {
        assert!(!a.lo.is_negative());
        let low = (&a.lo << self.p).sqrt();
        let rad = &a.hi << self.p;
        let mut high = rad.sqrt();
        if &high * &high != rad { high += 1; }
        self.iv(low, high)
    }

    /// log(m) for m in [1, 2] (reference series, general intervals).
    fn log_unit(&self, m: &Iv, terms: u32) -> Iv {
        assert!(m.lo >= self.one && m.hi <= &self.one * 2, "log range-reduction invariant");
        let one = self.integer(1);
        let z = self.div_positive(&self.sub(m, &one), &self.add(m, &one));
        let zz = self.square(&z);
        let mut term = z;
        let mut total = self.integer(0);
        let two = Z::from(2);
        for j in 0..terms {
            total = self.add(&total, &self.div_int(&self.scale(&term, &two), &Z::from(2 * j + 1)));
            term = self.mul(&term, &zz);
        }
        let t = log_tail(self.p, terms);
        self.iv(total.lo, total.hi + t)
    }
    fn log_exact(&mut self, s: &Z, terms: u32) -> Iv {
        assert!(s.is_positive());
        let e = s.bits() as i64 - 1 - self.p as i64;
        assert!(e <= 0, "polar logarithm range exceeded");
        let m = self.point(s << (-e) as u32);
        if self.ln2.as_ref().map(|(t, _)| *t) != Some(terms) {
            let two = self.integer(2);
            self.ln2 = Some((terms, self.log_unit(&two, terms)));
        }
        let ans = self.log_unit(&m, terms);
        let ln2 = self.ln2.as_ref().unwrap().1.clone();
        self.add(&ans, &self.scale(&ln2, &Z::from(e)))
    }
    fn radial_exact(&mut self, s: &Z, terms: u32) -> Iv {
        let ell = self.log_exact(s, terms);
        let t = self.iv((&self.one - s).max(-&ell.hi), -&ell.lo);
        let two = Z::from(2);
        let r = self.div_positive(&self.scale(&t, &two), &self.point(s.clone()));
        self.sqrt(&r)
    }
    fn rounding(&self, a: &Iv) -> Option<Z> {
        let half: Z = &self.one >> 1u32;
        let k = floor_shr(&(&a.lo + &half), self.p);
        let center: Z = &k << self.p;
        if &center - &half < a.lo && a.hi < &center + &half { Some(k) } else { None }
    }
    fn prefix(&self, u: &LazyUniform, signed: bool) -> Iv {
        assert!(u.bits <= self.p as usize, "coordinate prefix cannot exceed dyadic precision");
        let sh = self.p as usize - u.bits;
        let low: Z = &u.prefix << sh;
        let high: Z = (&u.prefix + 1) << sh;
        if signed { self.iv(low * 2 - &self.one, high * 2 - &self.one) } else { self.iv(low, high) }
    }

    /// Enclosure of a_k = e^{-b}/I_k, a = k/(2σ²), b = 1/(8σ²).
    fn correction(&self, k: &Z, sigma: u64, degree: u32, exp_degree: u32) -> Iv {
        let sig = Z::from(sigma);
        assert!(sigma >= 1 && sigma <= 1 << 40 && k.abs() <= &sig * 256, "public correction magnitude guard");
        assert!((1..=IMAX).contains(&degree) && (1..=EMAX).contains(&exp_degree));
        let s2 = &sig * &sig;
        let a = self.rational(k, &(&s2 * 2));
        let b = self.rational(&Z::one(), &(&s2 * 8));
        let two_b = self.scale(&b, &Z::from(2));
        let (mut c0, mut c1) = (self.integer(1), self.neg(&a));
        let mut total = c0.clone();
        for n in 2..=degree {
            let cn = self.neg(&self.div_int(&self.add(&self.mul(&a, &c1), &self.mul(&two_b, &c0)), &Z::from(n)));
            if n % 2 == 0 { total = self.add(&total, &self.div_int(&cn, &Z::from(n + 1))); }
            c0 = c1;
            c1 = cn;
        }
        // Cauchy on |v| = 2: exp(2|a|+4b) < 2^m.
        let ka: Z = k.abs();
        let four_s2: Z = &s2 * Z::from(4);
        let numer: Z = Z::from(3) * (ka * Z::from(2) + Z::one()) + &four_s2 - Z::one();
        let m: Z = numer.div_floor(&four_s2);
        let shift = Z::from(self.p) + m - Z::from(degree);
        let tail = if shift.is_positive() { Z::one() << shift.to_u64().unwrap() } else { Z::one() };
        let lo = (&total.lo - &tail).max((&self.one * Z::from(7)).div_floor(&Z::from(8)));
        let integ = self.iv(lo, &total.hi + &tail);
        let (mut term, mut num) = (self.integer(1), self.integer(1));
        let neg_b = self.neg(&b);
        for n in 1..=exp_degree {
            term = self.div_int(&self.mul(&term, &neg_b), &Z::from(n));
            num = self.add(&num, &term);
        }
        let shift = self.p as i64 - 3 * (exp_degree as i64 + 1);
        let tail = Z::one() << shift.max(0) as u32;
        let num = self.iv((&num.lo - &tail).max(Z::zero()), (&num.hi + &tail).min(self.one.clone()));
        let ratio = self.div_positive(&num, &integ);
        self.iv(ratio.lo.max(Z::zero()), ratio.hi.min(self.one.clone()))
    }
}

/// ceil(3 · 2^(p-3n)) as the reference computes it.
fn log_tail(p: u32, terms: u32) -> Z {
    if 3 * terms > p {
        if 3 * terms == p + 1 { Z::from(2) } else { Z::one() }
    } else {
        Z::from(3) << (p - 3 * terms)
    }
}

/// Exact rational constants of a fast profile: coefficient intervals of
/// 1/(2j+1), the ln2 enclosure (atanh series at 1/3 plus its tail) and the
/// rounded tail ceil(3q/2^{3n}).
pub struct FastConstants {
    pub p: u32,
    pub terms: u32,
    pub coeff: Vec<(Z, Z)>,
    pub ln2: (Z, Z),
    pub tail: Z,
}

impl FastConstants {
    pub fn make(p: u32, n: u32) -> Self {
        let q = Z::one() << p;
        let coeff = (0..n).map(|j| {
            let d = Z::from(2 * j + 1);
            (q.div_floor(&d), cdiv(&q, &d))
        }).collect();
        // lower = 2 Σ_{j<n} 1/((2j+1) 3^{2j+1}) as an exact fraction num/den.
        let (mut num, mut den) = (Z::zero(), Z::one());
        for j in 0..n {
            let d = Z::from(2 * j + 1) * num_traits::pow(Z::from(3), (2 * j + 1) as usize);
            num = &num * &d + &den;
            den = &den * &d;
            let g = num.gcd(&den);
            num /= &g;
            den /= &g;
        }
        num *= 2;
        let lo = (&num * &q).div_floor(&den);
        let t = Z::one() << (3 * n);
        let hi = cdiv(&((&num * &t + &den * 3) * &q), &(&den * &t));
        let tail = cdiv(&(&q * 3), &t);
        FastConstants { p, terms: n, coeff, ln2: (lo, hi), tail }
    }
}

static FAST_CONSTANTS: Lazy<Vec<FastConstants>> =
    Lazy::new(|| FAST_PROFILES.iter().map(|&(_, p, n)| FastConstants::make(p, n)).collect());

/// Public fallback schedule: (coordinate bits, P, log terms).
pub fn stages() -> Vec<(usize, u32, u32)> {
    let mut v = vec![(128usize, 192u32, 64u32)];
    for bits in [256usize, 512, 1024, 2048, 4096, 8192, 16384] {
        let p = 2 * bits as u32 + 64;
        v.push((bits, p, LOGMAX.min((p + 2) / 3)));
    }
    v.push((BMAX, PMAX, LOGMAX));
    v
}

enum Trial {
    DiskRejected,
    Refine,
    Rounded(Z),
    Abort(Abort),
}

/// Fast-stage log enclosure for an exact grid point 0 < s <= 1.
fn fast_log(d: &Dyadic, c: &FastConstants, s: &Z) -> Iv {
    assert!(s.is_positive() && s <= &d.one, "s outside (0,1]");
    let e = s.bits() as i64 - 1 - d.p as i64;
    let m: Z = s << (-e) as u32;
    let num: Z = (&m - &d.one) << d.p;
    let den: Z = &m + &d.one;
    let z = d.iv(num.div_floor(&den), cdiv(&num, &den));
    let zz = d.positive_mul(&z, &z);
    let (l, h) = c.coeff.last().unwrap();
    let mut acc = d.iv(l.clone(), h.clone());
    for j in (0..c.coeff.len() - 1).rev() {
        let (l, h) = &c.coeff[j];
        acc = d.add(&d.positive_mul(&acc, &zz), &d.iv(l.clone(), h.clone()));
    }
    let ans = d.scale(&d.positive_mul(&z, &acc), &Z::from(2));
    let ans = d.iv(ans.lo, ans.hi + &c.tail);
    let ln2 = d.iv(c.ln2.0.clone(), c.ln2.1.clone());
    d.add(&ans, &d.scale(&ln2, &Z::from(e)))
}

/// Untrusted float hint for |k|.
pub(crate) fn candidate_from_float(u: &LazyUniform, v: &LazyUniform, sigma: u64) -> Option<i64> {
    let f = |l: &LazyUniform| {
        let p = l.prefix.to_f64()?;
        Some((p + 0.5) * 2f64.powi(1 - l.bits as i32) - 1.0)
    };
    let (x, y) = (f(u)?, f(v)?);
    let s = x * x + y * y;
    let r = (sigma as f64 * x * (-2.0 * s.ln() / s).sqrt()).abs();
    if !r.is_finite() { return None; }
    Some((r + 0.5).floor() as i64)
}

fn certify_m(t: &Iv, s: &Iv, m: &Z) -> bool {
    let up: Z = m * 2 + 1;
    let lower_ok = m.is_zero() || {
        let dn: Z = m * 2 - 1;
        &dn * &dn * &s.hi < t.lo
    };
    lower_ok && t.hi < &up * &up * &s.lo
}

fn find_candidate(t: &Iv, s: &Iv, sigma: u64) -> Z {
    let (mut lo, mut hi) = (Z::zero(), Z::from(sigma) * 256);
    while lo < hi {
        let mid: Z = (&lo + &hi) >> 1u32;
        let w: Z = &mid * 2 + 1;
        if t.lo < &w * &w * &s.hi { hi = mid } else { lo = mid + 1 }
    }
    lo
}

fn fast_trial(prof: usize, u: &LazyUniform, v: &LazyUniform, sigma: u64, st: &mut Stats) -> Trial {
    let c = &FAST_CONSTANTS[prof];
    let d = Dyadic::new(c.p);
    let x = d.prefix(u, true);
    let y = d.prefix(v, true);
    let xx = d.square(&x);
    let s = d.add(&xx, &d.square(&y));
    if s.lo >= d.one { return Trial::DiskRejected; }
    if u.terminal_endpoint_bad() || v.terminal_endpoint_bad() { return Trial::Abort(Abort::UniformEndpointGuard); }
    if !(u.endpoint_guard_known_safe() && v.endpoint_guard_known_safe()) { return Trial::Refine; }
    let alpha = Z::one() << (d.p as i64 - G as i64).max(0) as u32;
    if s.lo < alpha || s.hi > &d.one - &alpha { return Trial::Refine; }
    let loglo = fast_log(&d, c, &s.lo);
    let loghi = fast_log(&d, c, &s.hi);
    let minuslog = d.iv((&d.one - &s.hi).max(-&loghi.hi), -&loglo.lo);
    let t = d.scale(&d.positive_mul(&xx, &minuslog), &(Z::from(sigma) * Z::from(sigma) * 8));
    let bound = Z::from(sigma) * 256;
    let mut m: Option<Z> = None;
    if let Some(h) = candidate_from_float(u, v, sigma) {
        for k in [h, h - 1, h + 1] {
            let kz = Z::from(k);
            if k >= 0 && kz < bound && certify_m(&t, &s, &kz) { m = Some(kz); st.hint_certified += 1; break; }
        }
    }
    let m = match m {
        Some(m) => m,
        None => {
            st.integer_search += 1;
            let m = find_candidate(&t, &s, sigma);
            if !certify_m(&t, &s, &m) { return Trial::Refine; }
            m
        }
    };
    if m.is_zero() { return Trial::Rounded(m); }
    let sign = if !x.lo.is_negative() { 1 } else if !x.hi.is_positive() { -1 } else { return Trial::Refine };
    assert!(m < bound, "output invariant");
    Trial::Rounded(m * sign)
}

fn ref_trial(d: &mut Dyadic, u: &LazyUniform, v: &LazyUniform, sigma: u64, terms: u32) -> Trial {
    let x = d.prefix(u, true);
    let y = d.prefix(v, true);
    let s = d.add(&d.square(&x), &d.square(&y));
    if s.lo >= d.one { return Trial::DiskRejected; }
    if u.terminal_endpoint_bad() || v.terminal_endpoint_bad() { return Trial::Abort(Abort::UniformEndpointGuard); }
    if !(u.endpoint_guard_known_safe() && v.endpoint_guard_known_safe()) { return Trial::Refine; }
    let alpha = Z::one() << (d.p as i64 - G as i64).max(0) as u32;
    let terminal = u.bits == BMAX;
    if s.lo < alpha || s.hi > &d.one - &alpha {
        return if terminal { Trial::Abort(Abort::PolarSGuard) } else { Trial::Refine };
    }
    let lower = d.radial_exact(&s.hi, terms);
    let upper = d.radial_exact(&s.lo, terms);
    let r = d.iv(lower.lo, upper.hi);
    let normal = d.scale(&d.mul(&x, &r), &Z::from(sigma));
    match d.rounding(&normal) {
        None => if terminal { Trial::Abort(Abort::RoundingUnresolved) } else { Trial::Refine },
        Some(k) => {
            assert!(k.abs() < Z::from(sigma) * 256, "certified Gaussian overflow invariant failed");
            Trial::Rounded(k)
        }
    }
}

/// 64-bit downward-rounded exact lower bound on a_k.
pub fn gate_threshold(k: &Z, sigma: u64) -> u64 {
    let m = Z::one() << 64u32;
    let s2 = Z::from(sigma) * Z::from(sigma);
    let n = (&s2).max(&(k * k)).clone();
    let den: Z = &s2 * &s2 * 8;
    if n >= den { return 0; }
    (&m - cdiv(&(&m * &n), &den)).to_u64().unwrap()
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict { AcceptGate, Accept, Reject, Refine, Abort }

fn acceptance_at(d: &Dyadic, u: &mut LazyUniform, k: &Z, sigma: u64, gate: u64, degree: u32, exp_degree: u32) -> Result<Verdict, Abort> {
    u.refine(u.bits.max(64))?;
    let first: Z = &u.prefix >> (u.bits - 64);
    if first < Z::from(gate) { return Ok(Verdict::AcceptGate); }
    let a = d.correction(k, sigma, degree, exp_degree);
    let ui = d.prefix(u, false);
    if ui.hi < a.lo { return Ok(Verdict::Accept); }
    if ui.lo > a.hi { return Ok(Verdict::Reject); }
    Ok(if u.bits >= G + 4 && d.p == PMAX { Verdict::Abort } else { Verdict::Refine })
}

/// Where the sampler gets its three kinds of streams: the polar pair of
/// (outer proposal, raw trial) and the correction uniform of an outer proposal.
pub trait GaussianStreams {
    fn polar(&mut self, outer: u32, trial: u32) -> (Box<dyn BitStream>, Box<dyn BitStream>);
    fn correction(&mut self, outer: u32) -> Box<dyn BitStream>;
    /// First `bits` (<= 128) bits of the polar pair, as integers.
    fn polar_prefix(&mut self, outer: u32, trial: u32, bits: usize) -> Result<(u128, u128), CoinsExhausted> {
        let (mut x, mut y) = self.polar(outer, trial);
        let (mut a, mut b) = (0u128, 0u128);
        for _ in 0..bits {
            a = (a << 1) | x.next_bit()? as u128;
            b = (b << 1) | y.next_bit()? as u128;
        }
        Ok((a, b))
    }
    /// First 64 bits of the correction uniform.
    fn correction_prefix64(&mut self, outer: u32) -> Result<u64, CoinsExhausted> {
        let mut c = self.correction(outer);
        let mut a = 0u64;
        for _ in 0..64 { a = (a << 1) | c.next_bit()? as u64; }
        Ok(a)
    }
    /// Position marker so a fast pass can be replayed by the exact sampler.
    fn mark(&self) -> u64 { 0 }
    fn rewind(&mut self, _mark: u64) {}
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub raw_trials: u64,
    pub disk_rejected: u64,
    pub fast_refine: u64,
    pub fallback_stages: u64,
    pub hint_certified: u64,
    pub integer_search: u64,
    pub accept_gate: u64,
    pub accept_correction: u64,
    pub reject_correction: u64,
    pub correction_stages: u64,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.raw_trials += o.raw_trials;
        self.disk_rejected += o.disk_rejected;
        self.fast_refine += o.fast_refine;
        self.fallback_stages += o.fallback_stages;
        self.hint_certified += o.hint_certified;
        self.integer_search += o.integer_search;
        self.accept_gate += o.accept_gate;
        self.accept_correction += o.accept_correction;
        self.reject_correction += o.reject_correction;
        self.correction_stages += o.correction_stages;
    }
}

/// One certified sample of D_σ. `fast` selects the first-stage profile
/// (Some(1) = recommended P128; None = reference schedule only).
pub fn sample_one(sigma: u64, streams: &mut dyn GaussianStreams, fast: Option<usize>, st: &mut Stats) -> Result<i64, Abort> {
    assert!(sigma >= 1 && sigma <= 1 << 40);
    let sched = stages();
    for outer in 0..OUTER {
        let mut k: Option<Z> = None;
        for trial in 0..TRIALS {
            st.raw_trials += 1;
            let (sx, sy) = streams.polar(outer, trial);
            let (mut u, mut v) = (LazyUniform::new(sx), LazyUniform::new(sy));
            let mut res = Trial::Refine;
            if let Some(prof) = fast {
                let b = FAST_PROFILES[prof].0;
                u.refine(b)?;
                v.refine(b)?;
                res = fast_trial(prof, &u, &v, sigma, st);
                if matches!(res, Trial::Refine) { st.fast_refine += 1; }
            }
            if matches!(res, Trial::Refine) {
                for &(b, p, n) in &sched {
                    u.refine(b)?;
                    v.refine(b)?;
                    let mut d = Dyadic::new(p);
                    res = ref_trial(&mut d, &u, &v, sigma, n);
                    st.fallback_stages += 1;
                    if !matches!(res, Trial::Refine) { break; }
                }
            }
            match res {
                Trial::Abort(a) => return Err(a),
                Trial::Refine => return Err(Abort::RoundingUnresolved),
                Trial::DiskRejected => { st.disk_rejected += 1; }
                Trial::Rounded(r) => { k = Some(r); break; }
            }
        }
        let k = k.ok_or(Abort::PolarTrialCap)?;
        let gate = gate_threshold(&k, sigma);
        let mut acc = LazyUniform::new(streams.correction(outer));
        let mut verdict = Verdict::Refine;
        for (level, &(b, p, _)) in sched.iter().enumerate() {
            acc.refine(if level == 0 { 64 } else { b.min(G + 4) })?;
            let d = Dyadic::new(p);
            let degree = if p == PMAX { IMAX } else { IMAX.min(p + 512) };
            let exp_degree = if p == PMAX { EMAX } else { EMAX.min((p + 2) / 3) };
            verdict = acceptance_at(&d, &mut acc, &k, sigma, gate, degree, exp_degree)?;
            if level > 0 { st.correction_stages += 1; }
            if verdict != Verdict::Refine { break; }
        }
        match verdict {
            Verdict::AcceptGate => { st.accept_gate += 1; return Ok(k.to_i64().unwrap()); }
            Verdict::Accept => { st.accept_correction += 1; return Ok(k.to_i64().unwrap()); }
            Verdict::Reject => { st.reject_correction += 1; }
            _ => return Err(Abort::CorrectionComparisonCap),
        }
    }
    Err(Abort::OuterProposalCap)
}

/// Semantic-mode streams of one Gaussian coordinate (paper private-coin
/// addresses: opening, attempt, coordinate, outer proposal, raw trial).
pub struct CoordinateStreams<'a> {
    pub coins: &'a super::coins::Coins,
    pub opening: u64,
    pub attempt: u16,
    pub coordinate: u32,
}

impl<'a> GaussianStreams for CoordinateStreams<'a> {
    fn polar(&mut self, outer: u32, trial: u32) -> (Box<dyn BitStream>, Box<dyn BitStream>) {
        use super::coins::Purpose;
        let mk = |p| Box::new(self.coins.stream(self.polar_address(p, outer, trial))) as Box<dyn BitStream>;
        (mk(Purpose::PolarX), mk(Purpose::PolarY))
    }
    fn correction(&mut self, outer: u32) -> Box<dyn BitStream> {
        Box::new(self.coins.stream(self.correction_address(outer)))
    }
    fn polar_prefix(&mut self, outer: u32, trial: u32, bits: usize) -> Result<(u128, u128), CoinsExhausted> {
        use super::coins::Purpose;
        assert!(bits <= 128);
        let f = |p| {
            let b = self.coins.block(&self.polar_address(p, outer, trial), 0);
            u128::from_be_bytes(b[..16].try_into().unwrap()) >> (128 - bits)
        };
        Ok((f(Purpose::PolarX), f(Purpose::PolarY)))
    }
    fn correction_prefix64(&mut self, outer: u32) -> Result<u64, CoinsExhausted> {
        let b = self.coins.block(&self.correction_address(outer), 0);
        Ok(u64::from_be_bytes(b[..8].try_into().unwrap()))
    }
}

impl<'a> CoordinateStreams<'a> {
    fn polar_address(&self, p: super::coins::Purpose, outer: u32, trial: u32) -> super::coins::Address {
        let mut a = super::coins::Address::new(p, self.opening);
        a.attempt = self.attempt;
        a.coordinate = self.coordinate;
        a.outer = outer as u16;
        a.trial = trial as u16;
        a
    }
    fn correction_address(&self, outer: u32) -> super::coins::Address {
        let mut a = super::coins::Address::new(super::coins::Purpose::Correction, self.opening);
        a.attempt = self.attempt;
        a.coordinate = self.coordinate;
        a.outer = outer as u16;
        a
    }
}

/// Fixture streams: the i-th requested stream is `FixtureStream(label, i)`,
/// in request order (polar X, polar Y per trial, then correction).
pub struct FixtureStreams { pub label: Vec<u8>, pub next: u64 }

impl GaussianStreams for FixtureStreams {
    fn polar(&mut self, _o: u32, _t: u32) -> (Box<dyn BitStream>, Box<dyn BitStream>) {
        let a = Box::new(super::coins::FixtureStream::new(&self.label, self.next));
        let b = Box::new(super::coins::FixtureStream::new(&self.label, self.next + 1));
        self.next += 2;
        (a, b)
    }
    fn correction(&mut self, _o: u32) -> Box<dyn BitStream> {
        let a = Box::new(super::coins::FixtureStream::new(&self.label, self.next));
        self.next += 1;
        a
    }
    fn mark(&self) -> u64 { self.next }
    fn rewind(&mut self, mark: u64) { self.next = mark; }
}

/// The interval reference's P128 trial on given 80-bit prefixes, rendered as
/// "disk", "refine" or the certified k (for differential tests).
#[cfg(test)]
pub(crate) fn reference_fast_trial(u: u128, v: u128, sigma: u64) -> String {
    struct Never;
    impl BitStream for Never { fn next_bit(&mut self) -> Result<u8, CoinsExhausted> { Err(CoinsExhausted) } }
    let mk = |p: u128| LazyUniform { src: Box::new(Never), prefix: Z::from(p), bits: FAST_PROFILES[1].0 };
    match fast_trial(1, &mk(u), &mk(v), sigma, &mut Stats::default()) {
        Trial::DiskRejected => "disk".into(),
        Trial::Refine => "refine".into(),
        Trial::Rounded(k) => k.to_string(),
        Trial::Abort(a) => format!("{a:?}"),
    }
}

/// Exact discrete-Gaussian probabilities (normalized over |k| <= kmax) for tests.
pub fn reference_pmf(sigma: f64, kmax: i64) -> HashMap<i64, f64> {
    let w: Vec<(i64, f64)> = (-kmax..=kmax).map(|k| (k, (-(k * k) as f64 / (2.0 * sigma * sigma)).exp())).collect();
    let tot: f64 = w.iter().map(|x| x.1).sum();
    w.into_iter().map(|(k, x)| (k, x / tot)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shifts_floor_and_constants_match_the_packet() {
        assert_eq!(floor_shr(&Z::from(-5), 1), Z::from(-3));
        assert_eq!(ceil_shr(&Z::from(-5), 1), Z::from(-2));
        assert_eq!(ceil_shr(&Z::from(5), 1), Z::from(3));
        // Values from fast_gaussian_certificate/constants.json.
        let c = &FAST_CONSTANTS[1];
        assert_eq!(c.ln2.0.to_string(), "235865763225513294137944142764154484399");
        assert_eq!(c.ln2.1.to_string(), "235865763225513294137944142764154484401");
        assert_eq!(c.tail, Z::from(2));
        assert_eq!(c.coeff[1].0.to_string(), "113427455640312821154458202477256070485");
        assert_eq!(c.coeff[42].1.to_string(), "4003321963775746628980877734491390724");
        let c0 = &FAST_CONSTANTS[0];
        assert_eq!(c0.ln2.0.to_string(), "54916777467707473351141471128");
        assert_eq!(c0.ln2.1.to_string(), "54916777467707473351141471132");
        assert_eq!(c0.tail, Z::from(3));
    }

    #[test]
    fn log_enclosures_contain_the_float_log() {
        let c = &FAST_CONSTANTS[1];
        let d = Dyadic::new(c.p);
        let mut dr = Dyadic::new(192);
        for &s in &[0.001f64, 0.1, 0.25, 0.5, 0.7071, 0.999] {
            let sz = Z::from((s * 2f64.powi(60)) as u64) << 68u32;
            let l = fast_log(&d, c, &sz);
            let f = |z: &Z| z.to_f64().unwrap() / 2f64.powi(128);
            assert!(f(&l.lo) <= s.ln() + 1e-12 && s.ln() - 1e-12 <= f(&l.hi), "{s}");
            assert!(&l.hi - &l.lo < Z::from(64));
            let sr = Z::from((s * 2f64.powi(60)) as u64) << 132u32;
            let lr = dr.log_exact(&sr, 64);
            // Same point at two precisions: the enclosures must intersect.
            assert!((&lr.lo >> 64u32) <= l.hi && l.lo <= (&lr.hi >> 64u32) + 1);
        }
    }

    fn chi2(sigma: u64, n: usize, fast: Option<usize>) -> (f64, Stats) {
        let mut st = Stats::default();
        let mut fx = FixtureStreams { label: format!("chi2-{sigma}-{fast:?}").into_bytes(), next: 0 };
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for _ in 0..n { *counts.entry(sample_one(sigma, &mut fx, fast, &mut st).unwrap()).or_default() += 1; }
        let kmax = (6 * sigma) as i64;
        let pmf = reference_pmf(sigma as f64, kmax + 40);
        let (mut x2, mut tail_e, mut tail_o) = (0.0, 0.0, 0usize);
        for k in -kmax - 40..=kmax + 40 {
            let e = pmf[&k] * n as f64;
            let o = *counts.get(&k).unwrap_or(&0);
            if e >= 5.0 { x2 += (o as f64 - e).powi(2) / e; } else { tail_e += e; tail_o += o; }
        }
        assert!(counts.keys().all(|k| k.abs() <= kmax + 40));
        if tail_e > 0.0 { x2 += (tail_o as f64 - tail_e).powi(2) / tail_e.max(1.0); }
        (x2, st)
    }

    #[test]
    fn small_sigma_distribution_and_correction_path() {
        // σ = 1: correction rejects are frequent, so the exact correction
        // enclosure is exercised. ~10 cells: chi² p=1e-6 critical ≈ 40.
        let (x2, st) = chi2(1, 20_000, Some(1));
        assert!(x2 < 40.0, "chi2 {x2} {st:?}");
        assert!(st.reject_correction > 0 && st.accept_correction > 0, "{st:?}");
        // σ = 3 through the reference schedule only (no fast stage).
        let (x2, st) = chi2(3, 6_000, None);
        assert!(x2 < 70.0, "chi2 {x2} {st:?}");
        assert!(st.fallback_stages > 0);
    }

    #[test]
    fn response_sigma_moments() {
        let sigma = super::super::params::Profile::ternary().sigma;
        let mut st = Stats::default();
        let mut fx = FixtureStreams { label: b"moments".to_vec(), next: 0 };
        let n = 4000;
        let xs: Vec<f64> = (0..n).map(|_| sample_one(sigma, &mut fx, Some(1), &mut st).unwrap() as f64 / sigma as f64).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| x * x).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.08 && (var - 1.0).abs() < 0.08, "mean {mean} var {var}");
        assert_eq!(st.fast_refine, 0, "{st:?}");
        assert_eq!(st.reject_correction, 0);
    }

    #[test]
    fn bit_identical_to_the_python_packet() {
        // Outputs of the packet's certificate.sample_one(profile=1) and
        // reference.sample_one on identical SHAKE-256 fixture streams.
        let cases: [(u64, Option<usize>, &[i64]); 5] = [
            (1, Some(1), &[-1, -1, -1, 1, 0, 0, -1, 1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 2, 0, -1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, -1, 1, -1, 0, 0, 0, 1]),
            (3, Some(1), &[10, -3, 0, 3, -4, -2, 4, 3, 2, -6, 4, 2, 0, -6, -2, 3, -3, -1, -1, 2, -1, -1, 1, 6, -3, -2, 1, 2, 0, 1, -1, -1, -2, -3, -1, -4, -4, 4, 0, 1]),
            (56, Some(1), &[-84, 54, -93, -66, -49, 51, -51, 34, -10, -27, -21, 11, 53, -50, -66, 78, -127, 3, -2, 14, -86, -100, 6, 95, -30, 85, -23, -75, -30, -88]),
            (3076346150, Some(1), &[2167383052, 7323372862, -6973072843, -5211926799, -1887725873, -543901378, -215149183, -1845136578, 43810557, 1151286122, -1699561882, -4028104731, -4676965865, 2518865519, 91347264, 1516389143, 1342313113, -2337450259, 1637750063, -3468562084, -2699007286, 3201631496, 287754937, -4030868433, -313129578, -989497882, -760683852, -1998298868, -658516519, -2247085928]),
            (3, None, &[10, -3, 0, 3, -4, -2, 4, 3, 2, -6]),
        ];
        for (sigma, fast, want) in cases {
            let mut fx = FixtureStreams { label: format!("xcheck-{sigma}").into_bytes(), next: 0 };
            let mut st = Stats::default();
            let got: Vec<i64> = (0..want.len()).map(|_| sample_one(sigma, &mut fx, fast, &mut st).unwrap()).collect();
            assert_eq!(got, want, "sigma {sigma} fast {fast:?}");
        }
    }

    #[test]
    #[ignore]
    fn throughput_response_sigma() {
        let sigma = super::super::params::Profile::ternary().sigma;
        let coins = super::super::coins::Coins::new([3u8; 64], super::super::coins::CoinProfile::Ternary);
        let n = 20_000u32;
        let t = std::time::Instant::now();
        let mut st = Stats::default();
        for c in 0..n {
            let mut s = CoordinateStreams { coins: &coins, opening: 0, attempt: 0, coordinate: c };
            sample_one(sigma, &mut s, Some(1), &mut st).unwrap();
        }
        let dt = t.elapsed().as_secs_f64();
        println!("{n} samples {:.3}s = {:.2} us/sample (1 thread) {st:?}", dt, dt * 1e6 / n as f64);
    }
}
