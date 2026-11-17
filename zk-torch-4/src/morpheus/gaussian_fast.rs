//! Fixed-width P128 first stage of the certified Gaussian sampler.
//!
//! Same decisions as `gaussian::fast_trial` with profile 1 (80 coordinate
//! bits, P = 128, 43 log terms), computed in 384-bit fixed registers with no
//! allocation. Any outcome other than "disk rejected" or "certified and
//! accepted by the 64-bit correction gate" returns `None`, and the caller
//! replays the whole scalar through the exact sampler on the same address
//! streams. The certified output is unique (rounding cells are disjoint), so
//! the replay returns the identical value; only the work differs.
//!
//! Two simplifications against the interval reference, both exact:
//! only the lower log endpoint at S_lo and the upper one at S_hi are used
//! by the certificate, so each trial evaluates one rounded-down and one
//! rounded-up Horner chain instead of two full intervals; and signed
//! quantities are carried as sign plus magnitude.

use super::gaussian::{self, Abort, FastConstants, GaussianStreams, Stats, FAST_PROFILES, OUTER, TRIALS};
use num_traits::ToPrimitive;

#[allow(clippy::all)]
mod u384 {
    uint::construct_uint! { pub struct U384(6); }
}
pub use u384::U384;

const P: usize = 128;
const BC: usize = 80;

#[inline]
fn ceil_shr(x: U384, s: usize) -> U384 {
    let r = x >> s;
    if (r << s) == x { r } else { r + 1 }
}

fn from_z(z: &num_bigint::BigInt) -> U384 {
    let (sign, bytes) = z.to_bytes_be();
    assert!(sign != num_bigint::Sign::Minus);
    U384::from_big_endian(&bytes)
}

/// Per-σ constants for the fixed stage.
pub struct FastSigma {
    pub sigma: u64,
    pub(crate) d4: U384,
    pub(crate) s8: U384,
    pub(crate) bound: U384,
    pub(crate) coeff_lo: Vec<U384>,
    pub(crate) coeff_hi: Vec<U384>,
    pub(crate) ln2_lo: U384,
    pub(crate) ln2_hi: U384,
    pub(crate) tail: U384,
    one: U384,
    pub(crate) gate_small: u64,
}

pub enum Fixed { Disk, Rounded(i64), Refine }

impl FastSigma {
    pub fn new(sigma: u64) -> Self {
        assert!(sigma >= 1 && sigma <= 1 << 40);
        assert_eq!(FAST_PROFILES[1], (BC, P as u32, 43));
        let c = FastConstants::make(P as u32, 43);
        let s = U384::from(sigma);
        FastSigma {
            sigma,
            s8: s * s * 8,
            d4: s * s * s * s * 8,
            bound: s * 256,
            coeff_lo: c.coeff.iter().map(|x| from_z(&x.0)).collect(),
            coeff_hi: c.coeff.iter().map(|x| from_z(&x.1)).collect(),
            ln2_lo: from_z(&c.ln2.0),
            ln2_hi: from_z(&c.ln2.1),
            tail: from_z(&c.tail),
            one: U384::one() << P,
            gate_small: gaussian::gate_threshold(&num_bigint::BigInt::from(0), sigma),
        }
    }

    /// Lower bound of log(M) (rounded down) for the exact point m in [one, 2·one).
    fn ans_lo(&self, m: U384) -> U384 {
        let num = (m - self.one) << P;
        let den = m + self.one;
        let z = num / den;
        let zz = (z * z) >> P;
        let n = self.coeff_lo.len();
        let mut acc = self.coeff_lo[n - 1];
        for j in (0..n - 1).rev() { acc = ((acc * zz) >> P) + self.coeff_lo[j]; }
        ((z * acc) >> P) * 2
    }
    /// Upper bound of log(M) including the series tail.
    fn ans_hi(&self, m: U384) -> U384 {
        let num = (m - self.one) << P;
        let den = m + self.one;
        let (q, r) = num.div_mod(den);
        let z = if r.is_zero() { q } else { q + 1 };
        let zz = ceil_shr(z * z, P);
        let n = self.coeff_hi.len();
        let mut acc = self.coeff_hi[n - 1];
        for j in (0..n - 1).rev() { acc = ceil_shr(acc * zz, P) + self.coeff_hi[j]; }
        ceil_shr(z * acc, P) * 2 + self.tail
    }

    /// Square interval of X = 2V-1 for an 80-bit prefix u; sign = +1/-1/0 (straddle).
    fn square(&self, u: u128) -> (U384, U384, i8) {
        let a = U384::from(u) << (P - BC + 1);
        let b = U384::from(u + 1) << (P - BC + 1);
        let (ml, sl) = if a >= self.one { (a - self.one, 1i8) } else { (self.one - a, -1) };
        let (mh, sh) = if b >= self.one { (b - self.one, 1i8) } else { (self.one - b, -1) };
        let (l, h) = (ml * ml, mh * mh);
        let straddle = sl < 0 && sh > 0 || ml.is_zero() || mh.is_zero();
        let low = if straddle { U384::zero() } else { l.min(h) };
        let sign = if sl > 0 || ml.is_zero() { 1 } else if sh < 0 || mh.is_zero() { -1 } else { 0 };
        (low >> P, ceil_shr(l.max(h), P), sign)
    }

    fn certify(&self, t_lo: U384, t_hi: U384, s_lo: U384, s_hi: U384, m: U384) -> bool {
        let up = m * 2 + 1;
        (m.is_zero() || { let dn = m * 2 - 1; dn * dn * s_hi < t_lo }) && t_hi < up * up * s_lo
    }

    /// The P128 trial on prefixes (u, v) of 80 bits each.
    pub fn trial(&self, u: u128, v: u128) -> Fixed {
        let (xl, xh, xsign) = self.square(u);
        let (yl, yh, _) = self.square(v);
        let (s_lo, s_hi) = (xl + yl, xh + yh);
        if s_lo >= self.one { return Fixed::Disk; }
        let top = (1u128 << BC) - 1;
        if u == 0 || v == 0 || u == top || v == top { return Fixed::Refine; }
        if s_lo.is_zero() || s_hi > self.one - 1 { return Fixed::Refine; }
        // -log S upper bound from S_lo, lower bound from S_hi.
        let e_lo = 129 - s_lo.bits();
        let nl_hi = self.ln2_hi * e_lo - self.ans_lo(s_lo << e_lo);
        let e_hi = 129 - s_hi.bits();
        let a = self.ln2_lo * e_hi;
        let b = self.ans_hi(s_hi << e_hi);
        let floor = self.one - s_hi;
        let nl_lo = if a > b && a - b > floor { a - b } else { floor };
        let t_lo = ((xl * nl_lo) >> P) * self.s8;
        let t_hi = ceil_shr(xh * nl_hi, P) * self.s8;
        // Untrusted float hint, then exact binary search on the same intervals.
        let hint = {
            let f = |p: u128| (p as f64 + 0.5) * 2f64.powi(1 - BC as i32) - 1.0;
            let (x, y) = (f(u), f(v));
            let s = x * x + y * y;
            let r = (self.sigma as f64 * x * (-2.0 * s.ln() / s).sqrt()).abs();
            if r.is_finite() { Some((r + 0.5).floor() as i64) } else { None }
        };
        let mut m = None;
        if let Some(h) = hint {
            for k in [h, h - 1, h + 1] {
                if k >= 0 {
                    let kz = U384::from(k as u64);
                    if kz < self.bound && self.certify(t_lo, t_hi, s_lo, s_hi, kz) { m = Some(kz); break; }
                }
            }
        }
        let m = match m {
            Some(m) => m,
            None => {
                let (mut lo, mut hi) = (U384::zero(), self.bound);
                while lo < hi {
                    let mid = (lo + hi) >> 1;
                    let w = mid * 2 + 1;
                    if t_lo < w * w * s_hi { hi = mid } else { lo = mid + 1 }
                }
                if !self.certify(t_lo, t_hi, s_lo, s_hi, lo) { return Fixed::Refine; }
                lo
            }
        };
        let m = m.low_u64() as i64;
        if m == 0 { return Fixed::Rounded(0); }
        match xsign { 1 => Fixed::Rounded(m), -1 => Fixed::Rounded(-m), _ => Fixed::Refine }
    }

    /// 64-bit correction gate threshold for k.
    fn gate(&self, k: i64) -> u64 {
        if k.unsigned_abs() <= self.sigma { self.gate_small } else { gaussian::gate_threshold(&num_bigint::BigInt::from(k), self.sigma) }
    }
}

/// One certified sample: fixed fast pass, exact replay on anything else.
pub fn sample(fs: &FastSigma, streams: &mut dyn GaussianStreams, st: &mut Stats) -> Result<i64, Abort> {
    if super::coins::no_zk() { return Ok(0); }
    let mark = streams.mark();
    if let Some(k) = sample_fixed(fs, streams, st)? { return Ok(k); }
    streams.rewind(mark);
    st.fast_refine += 1;
    gaussian::sample_one(fs.sigma, streams, Some(1), st)
}

fn sample_fixed(fs: &FastSigma, streams: &mut dyn GaussianStreams, st: &mut Stats) -> Result<Option<i64>, Abort> {
    // The exact sampler starts a new outer proposal after a correction
    // reject; the fixed pass only handles the first proposal.
    let _ = OUTER;
    for trial in 0..TRIALS {
        st.raw_trials += 1;
        let (u, v) = streams.polar_prefix(0, trial, BC)?;
        match fs.trial(u, v) {
            Fixed::Disk => { st.disk_rejected += 1; }
            Fixed::Refine => return Ok(None),
            Fixed::Rounded(k) => {
                st.hint_certified += 1;
                let g = streams.correction_prefix64(0)?;
                if g < fs.gate(k) { st.accept_gate += 1; return Ok(Some(k)); }
                return Ok(None);
            }
        }
    }
    Ok(None)
}

#[allow(dead_code)]
fn _unused(z: &num_bigint::BigInt) -> Option<u64> { z.to_u64() }

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::coins::{CoinProfile, Coins};
    use super::super::gaussian::{CoordinateStreams, FixtureStreams};
    use rand::{Rng, SeedableRng};

    /// The fixed trial agrees with the interval reference trial, including
    /// prefixes near the axes, the unit circle and the endpoints.
    #[test]
    fn fixed_trial_matches_reference_trial() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let top = (1u128 << BC) - 1;
        for &sigma in &[1u64, 3, 56, 1344, 3_076_346_150, 1 << 40] {
            let fs = FastSigma::new(sigma);
            let mut cases: Vec<(u128, u128)> = (0..3000).map(|_| (rng.gen::<u128>() & top, rng.gen::<u128>() & top)).collect();
            let half = 1u128 << (BC - 1);
            for d in [0u128, 1, 2, 1000] {
                cases.push((half + d, half + 7));
                cases.push((half - d, half + 7));
                cases.push((half + d, rng.gen::<u128>() & top));
            }
            cases.extend([(0, 5), (top, 5), (5, top), (1, 1), (top - 1, half), (half, top - 1)]);
            // Just inside the unit circle (|X| ≈ 1/sqrt 2 on both axes).
            let r = ((half as f64) * (1.0 + std::f64::consts::FRAC_1_SQRT_2)) as u128;
            for d in 0..50u128 { cases.push((r - d, r)); cases.push((r + d, r)); }
            for (u, v) in cases {
                let got = match fs.trial(u, v) { Fixed::Disk => "disk".to_string(), Fixed::Refine => "refine".into(), Fixed::Rounded(k) => k.to_string() };
                let want = gaussian::reference_fast_trial(u, v, sigma);
                assert_eq!(got, want, "sigma {sigma} u {u} v {v}");
            }
        }
    }

    #[test]
    fn fixed_sampler_reproduces_exact_sampler() {
        for &sigma in &[1u64, 3, 1344, 3_076_346_150] {
            let fs = FastSigma::new(sigma);
            let (mut a, mut b) = (FixtureStreams { label: b"fx".to_vec(), next: 0 }, FixtureStreams { label: b"fx".to_vec(), next: 0 });
            let (mut s1, mut s2) = (Stats::default(), Stats::default());
            for _ in 0..300 {
                assert_eq!(sample(&fs, &mut a, &mut s1).unwrap(), gaussian::sample_one(sigma, &mut b, Some(1), &mut s2).unwrap());
            }
            let coins = Coins::new([1u8; 64], CoinProfile::Ternary);
            for c in 0..200u32 {
                let mut x = CoordinateStreams { coins: &coins, opening: 2, attempt: 1, coordinate: c };
                let mut y = CoordinateStreams { coins: &coins, opening: 2, attempt: 1, coordinate: c };
                assert_eq!(sample(&fs, &mut x, &mut s1).unwrap(), gaussian::sample_one(sigma, &mut y, Some(1), &mut s2).unwrap());
            }
        }
    }

    #[test]
    #[ignore]
    fn throughput_fixed() {
        use rayon::prelude::*;
        let sigma = super::super::params::Profile::ternary().sigma;
        let fs = FastSigma::new(sigma);
        let coins = Coins::new([3u8; 64], CoinProfile::Ternary);
        let n = 2_000_000u32;
        let t = std::time::Instant::now();
        let st = (0..n).into_par_iter().fold(Stats::default, |mut st, c| {
            let mut s = CoordinateStreams { coins: &coins, opening: 0, attempt: 0, coordinate: c };
            sample(&fs, &mut s, &mut st).unwrap();
            st
        }).reduce(Stats::default, |mut a, b| { a.add(&b); a });
        let dt = t.elapsed().as_secs_f64();
        println!("{n} samples {dt:.3}s on {} threads = {:.3} us/sample/thread, {:.1} M/s {st:?}",
            rayon::current_num_threads(), dt * 1e6 * rayon::current_num_threads() as f64 / n as f64, n as f64 / dt / 1e6);
        let t = std::time::Instant::now();
        let m = 200_000u32;
        (0..m).into_par_iter().for_each(|c| {
            let mut s = CoordinateStreams { coins: &coins, opening: 0, attempt: 0, coordinate: c };
            let _ = s.polar_prefix(0, 0, 80);
            let _ = s.correction_prefix64(0);
        });
        let dt = t.elapsed().as_secs_f64();
        println!("coins only (3 blocks): {:.3} us/sample/thread", dt * 1e6 * rayon::current_num_threads() as f64 / m as f64);
    }
}
