//! Exact clipped rejection coin: Bernoulli(min{1, exp(x)/3}) for rational x
//! (theory source candidate_primary64/clipped_rejection_bernoulli), used once
//! per authentication attempt with x = (‖U‖² − ‖Z‖²)/(2σ²) and the 8708-bit
//! protocol coin of that attempt.
//!
//! `decide_exact` ports the reference (g = 8704, b = g+4 uniform bits,
//! p = g+48, 14 squarings, Taylor degree 1024). `decide` first tries a
//! P = 256 bracket; a decision there is the exact decision, because both
//! compare the same uniform with an enclosure of the same number and the
//! cheap enclosure only decides when the uniform's cell clears it by at
//! least 2^-256, far more than the reference bracket width (< 2^-8731).

use num_bigint::{BigInt, BigUint};
use num_integer::Integer;
use num_traits::{One, Signed, Zero};

pub const G: u32 = 8704;
pub const B: u32 = G + 4;
const PREC: u32 = G + 48;
const M: u32 = 14;
const DEG: u32 = 1024;
const P_CHEAP: u32 = 256;
const DEG_CHEAP: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coin { Accept, Reject, Abort }

/// x = num/den, den > 0.
#[derive(Clone, Debug)]
pub struct Ratio { pub num: BigInt, pub den: BigInt }

fn cdiv(a: &BigInt, b: &BigInt) -> BigInt { -((-a).div_floor(b)) }

fn ge_int(x: &Ratio, k: i64) -> bool { x.num >= &x.den * k }
fn le_int(x: &Ratio, k: i64) -> bool { x.num <= &x.den * k }

/// Reference decision on the B-bit uniform u.
pub fn decide_exact(x: &Ratio, u: &BigUint) -> Coin {
    assert!(x.num.bits() <= 128 && x.den.bits() <= 128 && x.den.is_positive());
    let u = BigInt::from(u.clone());
    if ge_int(x, 2) { return Coin::Accept; }
    if le_int(x, -(G as i64)) {
        // Reject once the uniform's lower endpoint reaches alpha/2 = 2^-(g+1).
        return if u >= BigInt::one() << (B - G - 1) { Coin::Reject } else { Coin::Abort };
    }
    let scale = BigInt::one() << PREC;
    // y = |x| / 2^14 = a / c.
    let a = x.num.abs();
    let c = &x.den << M;
    let (mut p, mut den, mut prev) = (BigInt::one(), BigInt::one(), BigInt::one());
    let mut power = BigInt::one();
    for j in 1..=DEG {
        prev = p.clone();
        power = -power * &a;
        let cj = &c * j;
        p = &cj * &p + &power;
        den *= &cj;
    }
    let lower = prev * &c * DEG;
    let mut lo = (lower * &scale).div_floor(&den);
    let mut hi = cdiv(&(&p * &scale), &den);
    for _ in 0..M {
        lo = (&lo * &lo).div_floor(&scale);
        hi = cdiv(&(&hi * &hi), &scale).min(scale.clone());
    }
    // Bracket [ln/ld, hn/hd] of the acceptance probability.
    let three = BigInt::from(3);
    let (ln, ld, hn, hd) = if x.num <= BigInt::zero() {
        (lo, &three * &scale, hi, &three * &scale)
    } else {
        let lo2 = lo.max(&scale / 8);
        let (mut ln, mut ld) = (scale.clone(), &three * &hi);
        if ln > ld { ln = BigInt::one(); ld = BigInt::one(); }
        let (mut hn, mut hd) = (scale.clone(), &three * &lo2);
        if hn > hd { hn = BigInt::one(); hd = BigInt::one(); }
        (ln, ld, hn, hd)
    };
    let two_b = BigInt::one() << B;
    // (u+1)/2^B < ln/ld  ->  accept; u/2^B > hn/hd -> reject.
    if (&u + 1) * &ld < &ln * &two_b { return Coin::Accept; }
    if &u * &hd > &hn * &two_b { return Coin::Reject; }
    Coin::Abort
}

/// Cheap P = 256 bracket; None when the uniform's cell is not separated.
fn decide_cheap(x: &Ratio, u: &BigUint) -> Option<Coin> {
    if ge_int(x, 2) { return Some(Coin::Accept); }
    if le_int(x, -(G as i64)) { return None; }
    let p = P_CHEAP;
    let one = BigInt::one() << p;
    let a = x.num.abs() << p;
    let c = &x.den << M;
    let (y_lo, y_hi) = (a.div_floor(&c), cdiv(&a, &c));
    // exp(-y) for y in [y_lo, y_hi] ⊂ [0, 1): lower bound at y_hi with the
    // series truncated after an odd term, upper bound at y_lo after an even
    // term; positive terms rounded in the bound's direction, negative ones
    // against it.
    let series = |y: &BigInt, upper: bool| -> BigInt {
        let (mut dn, mut up) = (one.clone(), one.clone());
        let mut s = one.clone();
        let last = if upper { DEG_CHEAP } else { DEG_CHEAP - 1 };
        for j in 1..=last {
            let d = BigInt::from(j) << p;
            dn = (&dn * y).div_floor(&d);
            up = cdiv(&(&up * y), &d);
            let odd = j % 2 == 1;
            s = match (odd, upper) {
                (true, true) => s - &dn,
                (true, false) => s - &up,
                (false, true) => s + &up,
                (false, false) => s + &dn,
            };
        }
        s
    };
    let mut lo = series(&y_hi, false).max(BigInt::zero());
    let mut hi = series(&y_lo, true).min(one.clone());
    for _ in 0..M {
        lo = (&lo * &lo) >> p;
        hi = cdiv(&(&hi * &hi), &one).min(one.clone());
    }
    let three = BigInt::from(3);
    let (alo, ahi) = if x.num <= BigInt::zero() {
        (lo.div_floor(&three), cdiv(&hi, &three))
    } else {
        let lo2 = lo.max(&one / 8);
        let alo = if hi.is_zero() { one.clone() } else { ((&one * &one).div_floor(&(&three * &hi))).min(one.clone()) };
        let ahi = cdiv(&(&one * &one), &(&three * &lo2)).min(one.clone());
        (alo, ahi)
    };
    let u = BigInt::from(u.clone());
    let two_b = BigInt::one() << B;
    let shift = BigInt::one() << p;
    if (&u + 1) * &shift < &alo * &two_b { return Some(Coin::Accept); }
    if &u * &shift > &ahi * &two_b { return Some(Coin::Reject); }
    None
}

pub fn decide(x: &Ratio, u: &BigUint) -> Coin {
    if super::coins::no_zk() { return Coin::Accept; }
    decide_cheap(x, u).unwrap_or_else(|| decide_exact(x, u))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn cheap_decisions_equal_the_reference_and_fixtures_hold() {
        let r = |n: i64, d: i64| Ratio { num: BigInt::from(n), den: BigInt::from(d) };
        let half = BigUint::one() << (B - 1);
        // Packet fixtures: decide(0, 0) = accept, decide(0, 1/2) = reject.
        assert_eq!(decide(&r(0, 1), &BigUint::zero()), Coin::Accept);
        assert_eq!(decide(&r(0, 1), &half), Coin::Reject);
        assert_eq!(decide(&r(2, 1), &BigUint::zero()), Coin::Accept);
        assert_eq!(decide(&r(-(G as i64), 1), &BigUint::zero()), Coin::Abort);
        assert_eq!(decide(&r(-(G as i64), 1), &(BigUint::one() << (B - G - 1))), Coin::Reject);
        assert_eq!(decide(&r(-(G as i64), 1), &((BigUint::one() << (B - G - 1)) - 1u32)), Coin::Abort);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(4);
        let xs = [r(-100, 1), r(-2, 1), r(-1, 1), r(1, 3), r(1, 1), r(1098612288, 1000000000), r(11, 10), r(1999, 1000), r(-12345678901234567, 98765432101234567)];
        for x in xs.iter() {
            for _ in 0..3 {
                let u = BigUint::from_bytes_be(&(0..(B as usize + 7) / 8).map(|_| rng.gen::<u8>()).collect::<Vec<_>>()) >> (8 * ((B as usize + 7) / 8) - B as usize);
                let want = decide_exact(x, &u);
                if let Some(c) = decide_cheap(x, &u) { assert_eq!(c, want, "{x:?}"); }
                assert_eq!(decide(x, &u), want);
            }
        }
        // ln 3 ≈ 1.0986: just below, exp(x)/3 < 1; a uniform near 1 rejects.
        let near_one = (BigUint::one() << B) - 1u32;
        assert_eq!(decide(&r(1098612288, 1000000000), &near_one), Coin::Reject);
    }
}
