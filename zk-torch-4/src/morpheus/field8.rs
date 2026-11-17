//! The scalar challenge field K = F_q[Θ]/(Θ^8 − 3) of the Morpheus paper
//! (§Preliminaries, "Common algebra"): degree-8 extension of the almost-
//! Goldilocks prime q = 2^64 − 2^32 − 31.
//!
//! Irreducibility of Θ^8 − 3 (checked when this module was written): 8 | q − 1,
//! 3 is a quadratic non-residue mod q and −3/4 is not a fourth power, which
//! are Capelli's conditions for x^8 − a.
//!
//! Elements are stored as eight canonical base coefficients, a_0 + a_1 Θ + …
//! + a_7 Θ^7. Host arithmetic canonicalizes, so equal elements are equal words.

use almost_goldilocks_cuda::field::{reduce128, AlmostGoldilocksField as F, ALMOST_GOLDILOCKS_PRIME as Q};
use serde::{Deserialize, Serialize};

/// Θ^8 = W.
pub const W: u64 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct K8(pub [u64; 8]);

#[inline]
fn canon(v: u64) -> u64 { if v >= Q { v - Q } else { v } }
#[inline]
fn addq(a: u64, b: u64) -> u64 {
    let (s, c) = canon(a).overflowing_add(canon(b));
    if c || s >= Q { s.wrapping_sub(Q) } else { s }
}
#[inline]
fn subq(a: u64, b: u64) -> u64 {
    let (a, b) = (canon(a), canon(b));
    if a >= b { a - b } else { Q - (b - a) }
}
#[inline]
fn mulq(a: u64, b: u64) -> u64 { reduce128(canon(a) as u128 * canon(b) as u128) }

/// Base-field exponentiation (canonical).
pub fn powq(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    b = canon(b);
    while e > 0 {
        if e & 1 == 1 { r = mulq(r, b); }
        b = mulq(b, b);
        e >>= 1;
    }
    r
}
/// Base-field inverse by Fermat (0 maps to 0).
pub fn invq(a: u64) -> u64 { powq(a, Q - 2) }

impl K8 {
    pub const ZERO: K8 = K8([0; 8]);
    pub const ONE: K8 = K8([1, 0, 0, 0, 0, 0, 0, 0]);

    pub fn from_base(a: u64) -> Self { let mut c = [0u64; 8]; c[0] = canon(a); K8(c) }
    pub fn from_field(a: F) -> Self { Self::from_base(a.0) }
    pub fn is_zero(&self) -> bool { self.0.iter().all(|&c| canon(c) == 0) }
    /// The constant (base-field) coefficient.
    pub fn c0(&self) -> u64 { canon(self.0[0]) }

    pub fn add(&self, o: &K8) -> K8 {
        let mut r = [0u64; 8];
        for i in 0..8 { r[i] = addq(self.0[i], o.0[i]); }
        K8(r)
    }
    pub fn sub(&self, o: &K8) -> K8 {
        let mut r = [0u64; 8];
        for i in 0..8 { r[i] = subq(self.0[i], o.0[i]); }
        K8(r)
    }
    pub fn neg(&self) -> K8 { K8::ZERO.sub(self) }

    /// Multiplication modulo Θ^8 − 3. Products are reduced once each and the
    /// at most eight (canonical, < 2^64) terms per output coefficient are
    /// summed in u128 before one more reduction; the wrapped half is scaled
    /// by W = 3 the same way, so no intermediate overflows (8 * 2^64 * 3 < 2^128).
    pub fn mul(&self, o: &K8) -> K8 {
        // Unreduced 128-bit products summed in 192 bits per output coefficient
        // (the Θ^8-wrapped half scaled by W = 3), one reduction each.
        let a: [u64; 8] = self.0.map(canon);
        let b: [u64; 8] = o.0.map(canon);
        let mut lo = [0u128; 8];
        let mut ov = [0u64; 8];
        let mut hl = [0u128; 8];
        let mut hov = [0u64; 8];
        for i in 0..8 {
            for j in 0..8 {
                let p = a[i] as u128 * b[j] as u128;
                let k = i + j;
                if k < 8 { let (s, c) = lo[k].overflowing_add(p); lo[k] = s; ov[k] += c as u64; }
                else { let (s, c) = hl[k - 8].overflowing_add(p); hl[k - 8] = s; hov[k - 8] += c as u64; }
            }
        }
        let mut r = [0u64; 8];
        for k in 0..8 {
            // x = lo + ov·2^128 + 3·(hl + hov·2^128) mod q, 2^128 ≡ C² (mod q).
            let h = reduce128(hl[k]) as u128 * W as u128;                       // < 2^66
            let (s, c) = lo[k].overflowing_add(h);
            let top = ov[k] + c as u64 + W * hov[k];                              // small
            const C2: u64 = 63 * (1u64 << 32) + 992;                            // 2^128 mod q
            r[k] = reduce128(reduce128(s) as u128 + top as u128 * C2 as u128);
        }
        K8(r)
    }
    #[cfg(test)]
    pub fn mul_ref(&self, o: &K8) -> K8 {
        let a: [u64; 8] = self.0.map(canon);
        let b: [u64; 8] = o.0.map(canon);
        let mut lo = [0u128; 8];
        let mut hi = [0u128; 8];
        for i in 0..8 {
            for j in 0..8 {
                let p = reduce128(a[i] as u128 * b[j] as u128) as u128;
                let k = i + j;
                if k < 8 { lo[k] += p } else { hi[k - 8] += p }
            }
        }
        let mut r = [0u64; 8];
        for k in 0..8 {
            r[k] = reduce128(lo[k] + (W as u128) * (reduce128(hi[k]) as u128));
        }
        K8(r)
    }
    pub fn square(&self) -> K8 { self.mul(self) }
    /// Multiply by a base-field scalar.
    pub fn scale(&self, s: u64) -> K8 { K8(self.0.map(|c| mulq(c, s))) }

    /// Frobenius power φ^k: Θ^q = ω Θ with ω = 3^((q−1)/8) ∈ F_q, so
    /// φ^k(Σ a_i Θ^i) = Σ a_i ω^{ik} Θ^i.
    pub fn frobenius(&self, k: u32) -> K8 {
        let omega = powq(W, (Q - 1) / 8);
        let wk = powq(omega, k as u64 % 8);
        let mut r = [0u64; 8];
        let mut f = 1u64;
        for i in 0..8 { r[i] = mulq(self.0[i], f); f = mulq(f, wk); }
        K8(r)
    }
    /// Field norm N(a) = Π_k φ^k(a) ∈ F_q.
    pub fn norm(&self) -> u64 {
        let mut p = *self;
        for k in 1..8 { p = p.mul(&self.frobenius(k)); }
        debug_assert!(p.0[1..].iter().all(|&c| canon(c) == 0));
        p.c0()
    }
    /// Inverse a^{-1} = (Π_{k=1..7} φ^k(a)) / N(a). Zero maps to zero.
    pub fn inv(&self) -> K8 {
        if self.is_zero() { return K8::ZERO; }
        let mut conj = K8::ONE;
        for k in 1..8 { conj = conj.mul(&self.frobenius(k)); }
        let n = self.mul(&conj).c0();
        conj.scale(invq(n))
    }
    pub fn pow(&self, mut e: u128) -> K8 {
        let (mut b, mut r) = (*self, K8::ONE);
        while e > 0 { if e & 1 == 1 { r = r.mul(&b); } b = b.square(); e >>= 1; }
        r
    }
    /// Flat u64 words (for transcripts / serialization), canonical.
    pub fn words(&self) -> [u64; 8] { self.0.map(canon) }
}

/// eq(x, r) = Π_j (x_j r_j + (1 − x_j)(1 − r_j)) table over {0,1}^n, with
/// bit j of the index matching r[j] (the same little-endian convention as
/// `poly::evaluate_lagrange_basis_ext2`).
pub fn eq_table(r: &[K8]) -> Vec<K8> {
    let mut t = vec![K8::ZERO; 1usize << r.len()];
    t[0] = K8::ONE;
    for (i, ri) in r.iter().enumerate() {
        let half = 1usize << i;
        let omr = K8::ONE.sub(ri);
        if half >= 1 << 12 {
            use rayon::prelude::*;
            let (lo, hi) = t.split_at_mut(half);
            lo.par_iter_mut().zip(hi[..half].par_iter_mut()).for_each(|(a, b)| { *b = a.mul(ri); *a = a.mul(&omr); });
        } else {
            for j in (0..half).rev() {
                t[j | half] = t[j].mul(ri);
                t[j] = t[j].mul(&omr);
            }
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rnd(seed: &mut u64) -> K8 {
        let mut c = [0u64; 8];
        for v in c.iter_mut() { *seed ^= *seed << 13; *seed ^= *seed >> 7; *seed ^= *seed << 17; *v = *seed % Q; }
        K8(c)
    }
    #[test]
    fn ring_axioms_and_inverse() {
        let mut s = 0x1234_5678_9abc_def1u64;
        for _ in 0..200 {
            let (a, b, c) = (rnd(&mut s), rnd(&mut s), rnd(&mut s));
            assert_eq!(a.mul(&b), b.mul(&a));
            assert_eq!(a.mul(&b.mul(&c)), a.mul(&b).mul(&c));
            assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
            assert_eq!(a.mul(&a.inv()), K8::ONE);
            assert_eq!(a.sub(&a), K8::ZERO);
        }
    }
    #[test]
    fn theta_relations() {
        let mut th = K8::ZERO; th.0[1] = 1;
        assert_eq!(th.pow(8), K8::from_base(W));
        // Frobenius is the q-th power map.
        let mut s = 99u64;
        let a = rnd(&mut s);
        assert_eq!(a.frobenius(1), a.pow(Q as u128));
        // Norm is multiplicative and lands in F_q.
        let b = rnd(&mut s);
        assert_eq!(a.mul(&b).norm(), mulq(a.norm(), b.norm()));
    }
    #[test]
    fn field_has_no_zero_divisors_sampled() {
        // x^8 - 3 irreducible <=> every nonzero element invertible; spot check
        // elements with small coefficients, which would expose a factor.
        for a in 0..4u64 { for b in 0..4u64 { for c in 0..4u64 {
            let x = K8([a, b, c, 1, 0, 0, 0, 0]);
            assert_eq!(x.mul(&x.inv()), K8::ONE);
        }}}
    }
    #[test]
    fn eq_table_matches_product() {
        let mut s = 7u64;
        let r: Vec<K8> = (0..5).map(|_| rnd(&mut s)).collect();
        let t = eq_table(&r);
        for idx in 0..32usize {
            let mut p = K8::ONE;
            for (j, rj) in r.iter().enumerate() {
                p = p.mul(&if (idx >> j) & 1 == 1 { *rj } else { K8::ONE.sub(rj) });
            }
            assert_eq!(t[idx], p);
        }
        let s_all = t.iter().fold(K8::ZERO, |acc, x| acc.add(x));
        assert_eq!(s_all, K8::ONE);
    }
}

#[cfg(test)]
mod mul_tests {
    use super::*;
    #[test]
    fn fast_mul_matches_reference() {
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
        let edge = [0u64, 1, 2, Q - 1, Q - 2, Q, Q + 5, u64::MAX, 1 << 63, (1 << 32) + 31];
        for it in 0..20000 {
            let gen = |next: &mut dyn FnMut() -> u64, i: usize| -> u64 { if it % 3 == 0 { edge[(next() as usize + i) % edge.len()] } else { next() } };
            let a = K8(std::array::from_fn(|i| gen(&mut next, i)));
            let b = K8(std::array::from_fn(|i| gen(&mut next, i)));
            assert_eq!(a.mul(&b), a.mul_ref(&b), "{a:?} * {b:?}");
        }
    }
}
