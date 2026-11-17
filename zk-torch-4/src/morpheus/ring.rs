//! Physical ring R = F_q[Y]/(Y^4096 + 1), the semantic ring
//! S = F_q[X]/(X^64 + 1) embedded by X = Y^64, and the authenticated endpoint
//! module A = K ⊗ S (paper §Preliminaries, proof supplement "The algebraic
//! lift" and "Exact arithmetic implementation").
//!
//! q has no degree-4096 NTT (Y^4096 + 1 splits into 16 factors of degree 256),
//! so products are computed EXACTLY over the integers: centered inputs, a
//! negacyclic NTT in each auxiliary 31-bit prime, pointwise accumulation,
//! inverse NTT, Garner CRT to the centered integer, and only then reduction
//! mod q. The caller picks how many primes from a magnitude bound B on any
//! output coefficient (the guard is 2B < Π p_i). Reducing an unreconstructed
//! RNS accumulator mod q would be wrong.

use super::field8::K8;
use almost_goldilocks_cuda::field::{reduce128, ALMOST_GOLDILOCKS_PRIME as Q};
use once_cell::sync::Lazy;

pub const N: usize = 4096; // physical degree
pub const DS: usize = 64; // semantic degree
pub const T: usize = 64; // N / DS
pub const LOG_N: usize = 12;

/// Auxiliary primes p ≡ 1 (mod 8192) and primitive order-8192 roots ψ
/// (ψ^4096 = −1), from the proof supplement.
pub const NTT_PRIMES: [(u64, u64); 5] = [
    (2147377153, 1202012506),
    (2147352577, 760495213),
    (2147295233, 920713129),
    (2147205121, 2061834542),
    (2147196929, 1321772815),
];

#[inline]
fn pw(mut b: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1u64;
    b %= p;
    while e > 0 {
        if e & 1 == 1 { r = r * b % p; }
        b = b * b % p;
        e >>= 1;
    }
    r
}

/// Per-prime tables for the length-4096 negacyclic NTT, in the standard
/// Kyber/Dilithium formulation: zetas[k] = ψ^{bitrev_12(k)} with ψ a
/// primitive 8192-th root, so the negacyclic twist is built into the
/// twiddles. Forward: Cooley-Tukey, natural order in, bit-reversed out.
/// Inverse: Gentleman-Sande with the inverse twiddles, then × N^{-1}.
pub struct NttTables {
    pub p: u64,
    zetas: Vec<u64>,
    zetas_inv: Vec<u64>,
    n_inv: u64,
}

fn bitrev(x: usize, bits: usize) -> usize { x.reverse_bits() >> (usize::BITS as usize - bits) }

impl NttTables {
    fn new(p: u64, psi: u64) -> Self {
        assert_eq!(pw(psi, 4096, p), p - 1, "psi must have order 8192");
        let psi_inv = pw(psi, p - 2, p);
        let zetas = (0..N).map(|k| pw(psi, bitrev(k, LOG_N) as u64, p)).collect();
        let zetas_inv = (0..N).map(|k| pw(psi_inv, bitrev(k, LOG_N) as u64, p)).collect();
        NttTables { p, zetas, zetas_inv, n_inv: pw(N as u64, p - 2, p) }
    }

    /// In-place forward negacyclic NTT of residues in [0, p).
    pub fn forward(&self, a: &mut [u64]) {
        let p = self.p;
        let mut k = 1usize;
        let mut len = N / 2;
        while len >= 1 {
            let mut start = 0;
            while start < N {
                let z = self.zetas[k];
                k += 1;
                for j in start..start + len {
                    let t = a[j + len] * z % p;
                    let u = a[j];
                    a[j + len] = if u >= t { u - t } else { u + p - t };
                    a[j] = if u + t >= p { u + t - p } else { u + t };
                }
                start += 2 * len;
            }
            len >>= 1;
        }
    }

    /// Inverse of `forward`, including the N^{-1} scale.
    pub fn inverse(&self, a: &mut [u64]) {
        let p = self.p;
        let mut k = N - 1;
        let mut len = 1usize;
        while len < N {
            let mut start = 0;
            // Groups at this level are visited in the reverse of the forward
            // order, so walk the twiddle index down from the top of its range.
            let groups = N / (2 * len);
            let mut idx = k + 1 - groups;
            while start < N {
                let z = self.zetas_inv[idx];
                idx += 1;
                for j in start..start + len {
                    let u = a[j];
                    let v = a[j + len];
                    a[j] = if u + v >= p { u + v - p } else { u + v };
                    let d = if u >= v { u - v } else { u + p - v };
                    a[j + len] = d * z % p;
                }
                start += 2 * len;
            }
            k -= groups;
            len <<= 1;
        }
        for x in a.iter_mut() { *x = *x * self.n_inv % p; }
    }
}

pub static TABLES: Lazy<Vec<NttTables>> =
    Lazy::new(|| NTT_PRIMES.iter().map(|&(p, psi)| NttTables::new(p, psi)).collect());

/// Garner/CRT helper for the first `k` primes: mixed-radix digits, folded
/// mod q, centered by a lexicographic comparison with ⌊P/2⌋.
pub struct Crt {
    k: usize,
    p: Vec<u64>,
    inv: Vec<Vec<u64>>,  // inv[i][j] = p_j^{-1} mod p_i (j < i)
    half_digits: Vec<u64>, // mixed-radix digits of ⌊P/2⌋
    p_mod_q: u64,
}

impl Crt {
    pub fn new(k: usize) -> Self {
        assert!((1..=5).contains(&k));
        let p: Vec<u64> = NTT_PRIMES[..k].iter().map(|x| x.0).collect();
        let mut inv = vec![vec![0u64; k]; k];
        for i in 0..k { for j in 0..i { inv[i][j] = pw(p[j] % p[i], p[i] - 2, p[i]); } }
        // ⌊P/2⌋ in mixed radix: P is odd, so ⌊P/2⌋ = (P − 1)/2, whose digits are
        // obtained by halving the mixed-radix representation of P − 1, i.e.
        // digits (p_i − 1) for all i, from the top down with borrow.
        let mut digits: Vec<u64> = p.iter().map(|&x| x - 1).collect(); // P - 1
        let mut rem = 0u64; // remainder carried down, in units of the next radix
        for i in (0..k).rev() {
            let cur = rem * p[i] + digits[i];
            digits[i] = cur / 2;
            rem = cur % 2;
        }
        let mut pmq = 1u64;
        for &x in &p { pmq = reduce128(pmq as u128 * x as u128); }
        Crt { k, p, inv, half_digits: digits, p_mod_q: pmq }
    }
    pub fn modulus_log2(&self) -> f64 { self.p.iter().map(|&x| (x as f64).log2()).sum() }

    /// Residues r[i] mod p_i of an integer X with |X| < P/2 -> X mod q.
    #[inline]
    pub fn reconstruct(&self, r: &[u64]) -> u64 {
        let k = self.k;
        let mut d = [0u64; 5];
        for i in 0..k {
            let pi = self.p[i];
            let mut v = r[i] % pi;
            for j in 0..i {
                // v = (v - d_j) * p_j^{-1} mod p_i
                let dj = d[j] % pi;
                v = if v >= dj { v - dj } else { v + pi - dj };
                v = v * self.inv[i][j] % pi;
            }
            d[i] = v;
        }
        // X mod q by Horner from the top digit.
        let mut x = 0u64;
        for i in (0..k).rev() {
            x = reduce128(x as u128 * self.p[i] as u128 + d[i] as u128);
        }
        // X > ⌊P/2⌋ means the centered value is X − P.
        let mut greater = false;
        for i in (0..k).rev() {
            if d[i] != self.half_digits[i] { greater = d[i] > self.half_digits[i]; break; }
        }
        if greater {
            if x >= self.p_mod_q { x - self.p_mod_q } else { Q - (self.p_mod_q - x) }
        } else { x }
    }
}

/// Smallest number of primes with 2·bound < Π p_i, where `bound` bounds every
/// output integer coefficient (given as log2).
pub fn primes_for_bound(log2_bound: f64) -> usize {
    for k in 1..=5 {
        let lp: f64 = NTT_PRIMES[..k].iter().map(|x| (x.0 as f64).log2()).sum();
        if log2_bound + 1.0 < lp - 1e-9 { return k; }
    }
    panic!("bound 2^{log2_bound} exceeds the 5-prime CRT range");
}

/// Center a canonical field element: (−q/2, q/2].
#[inline]
pub fn center(a: u64) -> i64 {
    let a = if a >= Q { a - Q } else { a };
    if a > (Q - 1) / 2 { -((Q - a) as i64) } else { a as i64 }
}
#[inline]
pub fn from_signed(x: i64) -> u64 {
    if x >= 0 { (x as u64) % Q } else { Q - ((x.unsigned_abs()) % Q) }
}

/// Residue of a signed 64-bit integer modulo a 31-bit prime.
#[inline]
fn res(x: i64, p: u64) -> u64 { x.rem_euclid(p as i64) as u64 }

/// Exact matrix-vector product over R: out[i] = Σ_c a(i, c) · x[c] mod q,
/// where `a(i, c)` yields CENTERED key coefficients and `x[c]` are CENTERED
/// input coefficients. `log2_bound` must bound every output coefficient of
/// the integer product (e.g. log2(cols · 4096 · H · M)).
pub fn matvec(
    rows: usize,
    cols: usize,
    key: &dyn Fn(usize, usize) -> Vec<i64>,
    x: &[Vec<i64>],
    log2_bound: f64,
) -> Vec<Vec<u64>> {
    assert_eq!(x.len(), cols);
    let k = primes_for_bound(log2_bound);
    let crt = Crt::new(k);
    let tabs = &TABLES[..k];
    // Transform inputs once per prime (reused across output rows).
    let xh: Vec<Vec<Vec<u64>>> = x.iter().map(|xc| {
        tabs.iter().map(|t| {
            let mut v: Vec<u64> = xc.iter().map(|&c| res(c, t.p)).collect();
            t.forward(&mut v);
            v
        }).collect()
    }).collect();
    let mut out = Vec::with_capacity(rows);
    for i in 0..rows {
        let mut acc: Vec<Vec<u64>> = vec![vec![0u64; N]; k];
        for c in 0..cols {
            let a = key(i, c);
            for (pi, t) in tabs.iter().enumerate() {
                let mut ah: Vec<u64> = a.iter().map(|&v| res(v, t.p)).collect();
                t.forward(&mut ah);
                let accp = &mut acc[pi];
                let xp = &xh[c][pi];
                for j in 0..N { accp[j] = (accp[j] + ah[j] * xp[j]) % t.p; }
            }
        }
        for (pi, t) in tabs.iter().enumerate() { t.inverse(&mut acc[pi]); }
        let mut row = vec![0u64; N];
        let mut r = [0u64; 5];
        for j in 0..N {
            for pi in 0..k { r[pi] = acc[pi][j]; }
            row[j] = crt.reconstruct(&r[..k]);
        }
        out.push(row);
    }
    out
}

/// Schoolbook negacyclic product mod q (test oracle).
pub fn mul_schoolbook(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut acc = vec![(0u128, 0u128); N]; // (positive, negative) sums of reduced products
    for i in 0..N {
        if a[i] == 0 { continue; }
        for j in 0..N {
            let p = reduce128(a[i] as u128 * b[j] as u128) as u128;
            let k = i + j;
            if k < N { acc[k].0 += p } else { acc[k - N].1 += p }
        }
    }
    acc.iter().map(|&(pos, neg)| {
        let p = reduce128(pos);
        let n = reduce128(neg);
        if p >= n { p - n } else { Q - (n - p) }
    }).collect()
}

// ---------------------------------------------------------------------------
// Coefficient layout, projections and the S-linear evaluation map.
// ---------------------------------------------------------------------------

/// Logical position within a 4096 block (semantic column j, coefficient k:
/// 64j + k) -> physical coefficient j + 64k. An involution.
#[inline]
pub fn phys_of_logical(l: usize) -> usize { let (j, k) = (l / DS, l % DS); j + T * k }

/// π_j(w): the S element with coefficients [w]_{j + 64k}, k = 0..63.
pub fn pi_j(w: &[u64], j: usize) -> [u64; DS] {
    let mut s = [0u64; DS];
    for k in 0..DS { s[k] = w[j + T * k]; }
    s
}

/// An element of A = K ⊗ S: a polynomial in X of degree < 64 with K coefficients.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AElem(pub Vec<K8>);

impl AElem {
    pub fn zero() -> Self { AElem(vec![K8::ZERO; DS]) }
    pub fn add(&self, o: &AElem) -> AElem { AElem(self.0.iter().zip(&o.0).map(|(a, b)| a.add(b)).collect()) }
    pub fn sub(&self, o: &AElem) -> AElem { AElem(self.0.iter().zip(&o.0).map(|(a, b)| a.sub(b)).collect()) }
    pub fn scale_k(&self, s: &K8) -> AElem { AElem(self.0.iter().map(|a| a.mul(s)).collect()) }
    /// Constant S coefficient: ct_S.
    pub fn ct(&self) -> K8 { self.0[0] }
    /// Negacyclic product with an S element (base coefficients).
    pub fn mul_s(&self, s: &[u64; DS]) -> AElem {
        let mut r = vec![K8::ZERO; DS];
        for i in 0..DS {
            if self.0[i].is_zero() { continue; }
            for j in 0..DS {
                if s[j] == 0 { continue; }
                let t = self.0[i].scale(s[j]);
                let k = i + j;
                if k < DS { r[k] = r[k].add(&t) } else { r[k - DS] = r[k - DS].sub(&t) }
            }
        }
        AElem(r)
    }
}

/// g(X) = Σ_k a_k X^{-k} in A, with X^{-k} = −X^{64−k} for k > 0.
pub fn g_of_weights(a: &[K8; DS]) -> AElem {
    let mut g = vec![K8::ZERO; DS];
    g[0] = a[0];
    for k in 1..DS { g[DS - k] = a[k].neg(); }
    AElem(g)
}

/// Ev_a(W) = Σ_{l,j} g_{l,j} π_j(W_l) for physical columns W_l (each 4096
/// canonical coefficients) and weights a[l][j][k] ∈ K. Its constant S
/// coefficient equals Σ a_{l,j,k} [W_l]_{j+64k}.
pub fn ev(w: &[Vec<u64>], a: &dyn Fn(usize, usize, usize) -> K8) -> AElem {
    let mut acc = AElem::zero();
    for (l, wl) in w.iter().enumerate() {
        for j in 0..T {
            let mut wt = [K8::ZERO; DS];
            for k in 0..DS { wt[k] = a(l, j, k); }
            if wt.iter().all(|x| x.is_zero()) { continue; }
            acc = acc.add(&g_of_weights(&wt).mul_s(&pi_j(wl, j)));
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    fn xs(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn ntt_roundtrip_and_negacyclic_product() {
        let mut s = 42u64;
        for t in TABLES.iter() {
            let a: Vec<u64> = (0..N).map(|_| xs(&mut s) % t.p).collect();
            let mut b = a.clone();
            t.forward(&mut b);
            t.inverse(&mut b);
            assert_eq!(a, b, "roundtrip p={}", t.p);
        }
        // X^4095 * X = -1 in each prime.
        let t = &TABLES[0];
        let mut x1 = vec![0u64; N]; x1[1] = 1;
        let mut xl = vec![0u64; N]; xl[N - 1] = 1;
        t.forward(&mut x1); t.forward(&mut xl);
        let mut prod: Vec<u64> = (0..N).map(|i| x1[i] * xl[i] % t.p).collect();
        t.inverse(&mut prod);
        assert_eq!(prod[0], t.p - 1);
        assert!(prod[1..].iter().all(|&v| v == 0));
    }

    #[test]
    fn crt_matvec_matches_schoolbook() {
        let mut s = 7u64;
        // Uniform key (centered ~2^63) times ternary inputs: 3 primes suffice
        // for 2 columns; times Gaussian-sized inputs (~2^36) needs 5.
        for &(m_bits, cols) in &[(1u32, 2usize), (36, 2)] {
            let keys: Vec<Vec<Vec<u64>>> = (0..2).map(|_| (0..cols).map(|_| (0..N).map(|_| xs(&mut s) % Q).collect()).collect()).collect();
            let x: Vec<Vec<i64>> = (0..cols).map(|_| (0..N).map(|_| {
                let v = (xs(&mut s) % (1u64 << m_bits)) as i64;
                if xs(&mut s) & 1 == 1 { -v } else { v }
            }).collect()).collect();
            let log2b = (cols as f64).log2() + 12.0 + 63.0 + m_bits as f64;
            let out = matvec(2, cols, &|i, c| keys[i][c].iter().map(|&v| center(v)).collect(), &x, log2b);
            for i in 0..2 {
                let mut want = vec![0u64; N];
                for c in 0..cols {
                    let xc: Vec<u64> = x[c].iter().map(|&v| from_signed(v)).collect();
                    let p = mul_schoolbook(&keys[i][c], &xc);
                    for j in 0..N { want[j] = { let t = want[j] + p[j]; if t >= Q || t < want[j] { t.wrapping_sub(Q) } else { t } }; }
                }
                assert_eq!(out[i], want, "m_bits={m_bits}");
            }
        }
    }

    #[test]
    fn layout_is_an_involution_and_pi_matches() {
        for l in 0..N { assert_eq!(phys_of_logical(phys_of_logical(l)), l); }
        let w: Vec<u64> = (0..N as u64).collect();
        // logical (j, k) sits at physical j + 64k, which is what π_j reads.
        for j in [0usize, 5, 63] { let p = pi_j(&w, j); for k in 0..DS { assert_eq!(p[k] as usize, j + T * k); } }
    }

    #[test]
    fn ev_constant_term_and_s_linearity() {
        let mut s = 99u64;
        let w: Vec<Vec<u64>> = (0..2).map(|_| (0..N).map(|_| xs(&mut s) % 3).collect()).collect();
        let wts: Vec<K8> = (0..2 * N).map(|_| { let mut c = [0u64; 8]; for v in c.iter_mut() { *v = xs(&mut s) % Q; } K8(c) }).collect();
        let a = |l: usize, j: usize, k: usize| wts[l * N + j * DS + k];
        let e = ev(&w, &a);
        let mut want = K8::ZERO;
        for l in 0..2 { for j in 0..T { for k in 0..DS { want = want.add(&a(l, j, k).scale(w[l][j + T * k])); } } }
        assert_eq!(e.ct(), want);
        // S-linearity: Ev(X·W) = X·Ev(W), where X = Y^64 acts on the physical column.
        let shift = |v: &Vec<u64>| -> Vec<u64> {
            let mut r = vec![0u64; N];
            for i in 0..N { let k = i + T; if k < N { r[k] = v[i] } else { r[k - N] = (Q - v[i]) % Q } }
            r
        };
        let wx: Vec<Vec<u64>> = w.iter().map(shift).collect();
        let mut xpoly = [0u64; DS]; xpoly[1] = 1;
        assert_eq!(ev(&wx, &a), e.mul_s(&xpoly));
    }
}
