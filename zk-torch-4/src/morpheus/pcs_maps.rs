//! Public linear maps of the PCS (paper §PCS, Eqs. pcs_link_eval and
//! opening_auxiliary), CPU reference.
//!
//! * `enc` / `decode_h`: the A-element digit encoding and the total S-linear
//!   decoder H(P, s, e) = Σ_{a,h} b_a 2^h π_h(P_a), defined on arbitrary
//!   integer inputs (responses are not binary).
//! * `ev_raw`: Ev_raw(W; ξ) with the MLE weights eq(ξ, u) of the raw index
//!   u = 4096·l + 64·j + k (column l, logical in-column position 64j + k,
//!   physical j + 64k). The weights factor as e_l·e_j·e_k, so
//!       Ev = g_k(X) · P(X),   P(X) = Σ_{l,j} e_l e_j π_j(W_l),
//!   one K-weighted pass over the coefficients plus one A product.
//! * `e_o`: t_O = Σ_{i,k} a_ik(s_O) (x_i^k − 1/2).

use super::field8::{eq_table, K8};
use super::ring::{from_signed, g_of_weights, AElem, DS, N, T};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;

/// Ring columns of one encoded A element.
pub const ENC_COLS: usize = 8;

/// Product in A = K ⊗ S (negacyclic in X, K coefficients).
pub fn mul_a(a: &AElem, b: &AElem) -> AElem {
    let mut r = vec![K8::ZERO; DS];
    for i in 0..DS {
        if a.0[i].is_zero() { continue; }
        for j in 0..DS {
            let t = a.0[i].mul(&b.0[j]);
            let k = i + j;
            if k < DS { r[k] = r[k].add(&t) } else { r[k - DS] = r[k - DS].sub(&t) }
        }
    }
    AElem(r)
}

/// S action of a C0 challenge (coefficients in {±1, ±2}) on an A element.
pub fn act_c0_a(beta: &[i8; 64], a: &AElem) -> AElem {
    let s: [u64; DS] = std::array::from_fn(|k| from_signed(beta[k] as i64));
    a.mul_s(&s)
}

/// S action of β on one ring element R (X = Y^64), over the integers.
pub fn act_c0_ring_int(beta: &[i8; 64], v: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; N];
    for (m, &b) in beta.iter().enumerate() {
        let sh = T * m; // Y^{64m}
        let b = b as i64;
        for i in 0..N {
            let k = i + sh;
            if k < N { out[k] += b * v[i] } else { out[k - N] -= b * v[i] }
        }
    }
    out
}

/// Same action on a canonical ring element mod q.
pub fn act_c0_ring_q(beta: &[i8; 64], v: &[u64]) -> Vec<u64> {
    let mut out = vec![0u128; N];
    let mut neg = vec![0u128; N];
    for (m, &b) in beta.iter().enumerate() {
        let sh = T * m;
        let (mag, sgn) = (b.unsigned_abs() as u128, b < 0);
        for i in 0..N {
            let k = i + sh;
            let (pos_side, idx) = if k < N { (!sgn, k) } else { (sgn, k - N) };
            if pos_side { out[idx] += mag * v[i] as u128 } else { neg[idx] += mag * v[i] as u128 }
        }
    }
    (0..N).map(|i| {
        let p = (out[i] % Q as u128) as u64;
        let n = (neg[i] % Q as u128) as u64;
        if p >= n { p - n } else { Q - (n - p) }
    }).collect()
}

/// enc(u): 8 binary ring columns; column a holds bit h of residue
/// coordinate a of semantic coefficient k at physical position h + 64k.
pub fn enc(u: &AElem) -> Vec<Vec<i64>> {
    (0..ENC_COLS).map(|a| {
        let mut col = vec![0i64; N];
        for k in 0..DS {
            let w = u.0[k].0[a];
            for h in 0..64 { col[h + T * k] = ((w >> h) & 1) as i64; }
        }
        col
    }).collect()
}

/// H on the first 8 columns (any integers); the suffix is ignored.
pub fn decode_h(cols: &[Vec<i64>]) -> AElem {
    let mut out = vec![K8::ZERO; DS];
    let mut pow2 = [0u64; 64];
    for h in 0..64 { pow2[h] = ((1u128 << h) % Q as u128) as u64; }
    for a in 0..ENC_COLS {
        for k in 0..DS {
            let mut acc: u128 = 0;
            let mut neg: u128 = 0;
            for h in 0..64 {
                let v = cols[a][h + T * k];
                let t = pow2[h] as u128 * v.unsigned_abs() as u128;
                if v >= 0 { acc += t } else { neg += t }
            }
            let p = (acc % Q as u128) as u64;
            let n = (neg % Q as u128) as u64;
            out[k].0[a] = if p >= n { p - n } else { Q - (n - p) };
        }
    }
    AElem(out)
}

/// Ev_raw(W; ξ) for physical columns W (centered integers) with
/// ξ = (ξ_0 .. ξ_{ℓ-1}), bit i of the raw index u little-endian.
pub fn ev_raw(w: &[Vec<i64>], xi: &[K8]) -> AElem {
    assert!(xi.len() >= 12 && (1usize << xi.len()) == w.len() * N, "ℓ must cover the columns");
    let ek = eq_table(&xi[..6]);
    let ej = eq_table(&xi[6..12]);
    let el = eq_table(&xi[12..]);
    let mut p = vec![K8::ZERO; DS];
    for (l, col) in w.iter().enumerate() {
        for j in 0..T {
            let wt = el[l].mul(&ej[j]);
            for m in 0..DS {
                let v = col[j + T * m];
                if v != 0 { p[m] = p[m].add(&wt.scale(from_signed(v))); }
            }
        }
    }
    let wk: [K8; DS] = std::array::from_fn(|k| ek[k]);
    mul_a(&g_of_weights(&wk), &AElem(p))
}

/// Aux layout: semantic coefficient (i, k), k = 1..=d_i, i over variables,
/// row-major, each 8 digit columns.
pub fn aux_index(degs: &[usize], i: usize, k: usize) -> usize {
    degs[..i].iter().sum::<usize>() + (k - 1)
}

/// t_O = Σ_{i,k} a_ik (x_i^k − 1/2) from the aux digit columns.
pub fn e_o(aux_cols: &[Vec<i64>], degs: &[usize], x: &[K8]) -> AElem {
    assert_eq!(degs.len(), x.len());
    let half = K8::ONE.scale((Q + 1) / 2);
    let mut acc = AElem::zero();
    for (i, &d) in degs.iter().enumerate() {
        let mut xp = K8::ONE;
        for k in 1..=d {
            xp = xp.mul(&x[i]);
            let idx = aux_index(degs, i, k);
            let a = decode_h(&aux_cols[ENC_COLS * idx..ENC_COLS * idx + ENC_COLS]);
            acc = acc.add(&a.scale_k(&xp.sub(&half)));
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ring::ev;
    fn xs(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }
    fn rk(s: &mut u64) -> K8 { let mut c = [0u64; 8]; for v in c.iter_mut() { *v = xs(s) % Q; } K8(c) }

    #[test]
    fn ev_raw_matches_the_generic_ev_with_eq_weights() {
        let mut s = 3u64;
        let cols = 2;
        let w: Vec<Vec<i64>> = (0..cols).map(|_| (0..N).map(|_| (xs(&mut s) % 7) as i64 - 3).collect()).collect();
        let xi: Vec<K8> = (0..13).map(|_| rk(&mut s)).collect();
        let eqw = eq_table(&xi);
        let wq: Vec<Vec<u64>> = w.iter().map(|c| c.iter().map(|&v| from_signed(v)).collect()).collect();
        let want = ev(&wq, &|l, j, k| eqw[l * N + DS * j + k]);
        assert_eq!(ev_raw(&w, &xi), want);
        // ct_S of Ev_raw is the MLE of the raw vector (logical order) at ξ.
        let mut mle = K8::ZERO;
        for l in 0..cols { for j in 0..T { for k in 0..DS { mle = mle.add(&eqw[l * N + DS * j + k].scale(wq[l][j + T * k])); } } }
        assert_eq!(want.ct(), mle);
    }

    #[test]
    fn decoder_inverts_the_encoder_and_is_s_linear() {
        let mut s = 11u64;
        let u = AElem((0..DS).map(|_| rk(&mut s)).collect());
        assert_eq!(decode_h(&enc(&u)), u);
        // H(β·P) = β·H(P) for a C0 challenge, on non-binary integer inputs.
        let p: Vec<Vec<i64>> = (0..ENC_COLS).map(|_| (0..N).map(|_| (xs(&mut s) % 2001) as i64 - 1000).collect()).collect();
        let beta: [i8; 64] = std::array::from_fn(|i| [-2i8, -1, 1, 2][(xs(&mut s) % 4) as usize] * if i % 5 == 0 { 1 } else { 1 });
        let bp: Vec<Vec<i64>> = p.iter().map(|c| act_c0_ring_int(&beta, c)).collect();
        assert_eq!(decode_h(&bp), act_c0_a(&beta, &decode_h(&p)));
        // Ev_raw is S-linear too.
        let xi: Vec<K8> = (0..13).map(|_| rk(&mut s)).collect();
        let w: Vec<Vec<i64>> = (0..2).map(|_| (0..N).map(|_| (xs(&mut s) % 3) as i64 - 1).collect()).collect();
        let bw: Vec<Vec<i64>> = w.iter().map(|c| act_c0_ring_int(&beta, c)).collect();
        assert_eq!(ev_raw(&bw, &xi), act_c0_a(&beta, &ev_raw(&w, &xi)));
        // Ring action mod q agrees with the integer action.
        let v: Vec<u64> = (0..N).map(|_| xs(&mut s) % Q).collect();
        let vi: Vec<i64> = v.iter().map(|&x| super::super::ring::center(x)).collect();
        let a = act_c0_ring_q(&beta, &v);
        let b: Vec<u64> = act_c0_ring_int(&beta, &vi.iter().map(|&x| x / 1024).collect::<Vec<_>>()).iter().map(|&x| from_signed(x)).collect();
        let c = act_c0_ring_q(&beta, &vi.iter().map(|&x| from_signed(x / 1024)).collect::<Vec<_>>());
        assert_eq!(b, c);
        assert_eq!(a.len(), N);
    }
}
