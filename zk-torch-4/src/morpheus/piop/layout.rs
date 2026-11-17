//! Source layout: how committed tensors become aligned regions of raw
//! digit vectors (paper §PCS "source relation", Appendix numeric layout).
//!
//! A committed tensor has 2^n index positions and 2^p digit planes:
//!   * value tensors: signed binary planes, value = Σ_b 2^b d_b, d_b ∈ {−1,0,1};
//!   * bit tensors (one-hot inputs, lookup selector chunks, the graph-mask
//!     bank W_G): digits in {0,1}, declared to the link as bit blocks.
//! A tensor larger than the per-piece cap is split by planes, then by the
//! high index bits; a piece (index bits [0, m), planes [plane0, plane0+2^p'))
//! sits at raw base + i·2^p' + b (planes low) and contributes the term
//! scale·eq(r_low, i)·pw(b) with scale = eq(r_high, hb).
//! Pieces are packed first-fit decreasing, so every region is aligned to its
//! size. The layout is public (part of the statement).

use crate::morpheus::field8::K8;
use crate::morpheus::link::{Region, Term};
use crate::morpheus::params::Profile;
use crate::morpheus::ring::N;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TKey {
    /// A committed DAG edge (weights, inputs, dense advice, advice outputs).
    Edge(usize),
    /// Chunk `chunk` of the one-hot selector of lookup node `node`.
    Sel { node: usize, chunk: usize },
    /// The graph-mask bank W_G.
    Wg,
    /// Deferred fixed-weight opening: the per-request carry bank (hidden
    /// weight evaluations, same slot encoding as W_G).
    Carry,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Piece { pub region: Region, pub plane0: usize, pub hb: usize, pub hbits: usize }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorLayout { pub key: TKey, pub n: usize, pub p: usize, pub bits: bool, pub pieces: Vec<Piece> }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    pub tensors: Vec<TensorLayout>,
    pub n_sources: usize,
    pub ell: usize,
    #[serde(default)]
    pub min_lg: usize,
    #[serde(skip)]
    index: HashMap<TKey, usize>,
}

/// Region serde helpers (link::Region is a plain struct).
impl Serialize for Region {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.source, self.base, self.m, self.p).serialize(s)
    }
}
impl<'de> Deserialize<'de> for Region {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (source, base, m, p) = <(usize, usize, usize, usize)>::deserialize(d)?;
        Ok(Region { source, base, m, p })
    }
}

fn eq_bits(r: &[K8], bits: usize) -> K8 {
    r.iter().enumerate().fold(K8::ONE, |a, (j, x)| a.mul(&if (bits >> j) & 1 == 1 { *x } else { K8::ONE.sub(x) }))
}

impl Layout {
    /// Pack tensors `(key, n, p, bits)` into sources of the profile. Every
    /// piece gets at least 2^`min_lg` raw positions (index bits beyond the
    /// tensor's are zero digits; terms zero-extend the point), so a block of
    /// that size never straddles two regions.
    pub fn pack_min(profile: &Profile, specs: &[(TKey, usize, usize, bool)], min_lg: usize) -> Self {
        let specs: Vec<(TKey, usize, usize, bool)> = specs.iter().map(|&(k, n, p, b)| {
            assert!(p <= min_lg || min_lg == 0, "plane bits exceed the block size");
            (k, n, p, b)
        }).collect();
        let mut lay = Self::pack_inner(profile, &specs, min_lg);
        lay.min_lg = min_lg;
        lay
    }
    pub fn pack(profile: &Profile, specs: &[(TKey, usize, usize, bool)]) -> Self { Self::pack_min(profile, specs, 0) }

    fn pack_inner(profile: &Profile, specs: &[(TKey, usize, usize, bool)], min_lg: usize) -> Self {
        let ell = profile.ell();
        let cap = profile.source_payload_cols * N;
        let max_lg = (usize::BITS - 1 - cap.leading_zeros()) as usize; // floor(log2 cap)
        // Split into pieces of log-size <= max_lg.
        let mut tensors: Vec<TensorLayout> = Vec::new();
        let mut pieces: Vec<(usize, usize, usize, usize, usize)> = Vec::new(); // (tensor, m, p', plane0, hb)
        for (ti, &(key, n, p, bits)) in specs.iter().enumerate() {
            tensors.push(TensorLayout { key, n, p, bits, pieces: Vec::new() });
            if n + p <= max_lg {
                // Grow small tensors to the minimum piece size with zero index bits.
                let m = n.max(min_lg.saturating_sub(p));
                pieces.push((ti, m, p, 0, 0));
            } else if n <= max_lg {
                let pp = max_lg - n;
                for c in 0..1usize << (p - pp) { pieces.push((ti, n, pp, c << pp, 0)); }
            } else {
                for b in 0..1usize << p { for hb in 0..1usize << (n - max_lg) { pieces.push((ti, max_lg, 0, b, hb)); } }
            }
        }
        // First-fit decreasing.
        let mut order: Vec<usize> = (0..pieces.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(pieces[i].1 + pieces[i].2));
        let mut fill: Vec<usize> = Vec::new();
        for i in order {
            let (ti, m, pp, plane0, hb) = pieces[i];
            let size = 1usize << (m + pp);
            let s = match fill.iter().position(|&f| f + size <= cap) { Some(s) => s, None => { fill.push(0); fill.len() - 1 } };
            let base = fill[s];
            debug_assert_eq!(base % size, 0);
            fill[s] += size;
            let n = specs[ti].1;
            tensors[ti].pieces.push(Piece { region: Region { source: s, base, m, p: pp }, plane0, hb, hbits: n.saturating_sub(m) });
        }
        for t in tensors.iter_mut() { t.pieces.sort_by_key(|p| (p.plane0, p.hb)); }
        let index = tensors.iter().enumerate().map(|(i, t)| (t.key, i)).collect();
        Layout { tensors, n_sources: fill.len().max(1), ell, min_lg: 0, index }
    }

    /// Pack groups of tensors into disjoint, consecutive source ranges
    /// (group g gets sources ranges[g]). Used to keep fixed model weights,
    /// per-request tensors and the carry bank in separate sources.
    pub fn pack_groups(profile: &Profile, groups: &[Vec<(TKey, usize, usize, bool)>], min_lg: usize) -> (Self, Vec<std::ops::Range<usize>>) {
        let mut tensors = Vec::new();
        let mut ranges = Vec::new();
        let mut off = 0;
        for g in groups {
            let mut l = Self::pack_min(profile, g, min_lg);
            for t in l.tensors.iter_mut() { for pc in t.pieces.iter_mut() { pc.region.source += off; } }
            let n = if g.is_empty() { 0 } else { l.n_sources };
            ranges.push(off..off + n);
            off += n;
            tensors.extend(l.tensors);
        }
        let index = tensors.iter().enumerate().map(|(i, t)| (t.key, i)).collect();
        (Layout { tensors, n_sources: off.max(1), ell: profile.ell(), min_lg, index }, ranges)
    }

    /// Declared bit blocks of sources `range`, renumbered from 0.
    pub fn bit_blocks_in(&self, range: &std::ops::Range<usize>) -> Vec<Vec<(usize, usize)>> {
        let all = self.bit_blocks();
        all[range.clone()].to_vec()
    }

    pub fn rebuild_index(&mut self) { self.index = self.tensors.iter().enumerate().map(|(i, t)| (t.key, i)).collect(); }
    pub fn get(&self, k: &TKey) -> &TensorLayout { &self.tensors[*self.index.get(k).unwrap_or_else(|| panic!("tensor {k:?} not in layout"))] }
    pub fn has(&self, k: &TKey) -> bool { self.index.contains_key(k) }

    /// Terms of coeff·Σ_{b,i} eq(r, i)·pw[b]·digit_b(i) for tensor `k`.
    pub fn terms(&self, k: &TKey, r: &[K8], pw: &[K8], coeff: K8) -> Vec<Term> {
        let t = self.get(k);
        assert_eq!(r.len(), t.n, "point arity for {k:?}");
        assert!(pw.len() <= 1usize << t.p);
        t.pieces.iter().filter_map(|pc| {
            let np = 1usize << pc.region.p;
            if pc.plane0 >= pw.len() { return None; }
            let mut w: Vec<K8> = (0..np).map(|b| *pw.get(pc.plane0 + b).unwrap_or(&K8::ZERO)).collect();
            if w.iter().all(|x| x.is_zero()) { return None; }
            for x in w.iter_mut() { *x = x.mul(&coeff); }
            let m = pc.region.m;
            let (lo, scale) = if m >= t.n {
                // Grown piece: index bits beyond the tensor's are zero.
                let mut lo = r.to_vec();
                lo.resize(m, K8::ZERO);
                (lo, K8::ONE)
            } else {
                (r[..m].to_vec(), eq_bits(&r[m..], pc.hb))
            };
            Some(Term { region: pc.region.clone(), r: lo, pw: std::sync::Arc::new(w), scale })
        }).collect()
    }

    /// Same terms as `terms(k, r, pw, coeff)`, sharing `pw` (coeff goes into
    /// the scale) for pieces that cover all of its planes.
    pub fn terms_shared(&self, k: &TKey, r: &[K8], pw: &std::sync::Arc<Vec<K8>>, coeff: K8) -> Vec<Term> {
        let t = self.get(k);
        if t.pieces.iter().any(|pc| pc.plane0 + (1usize << pc.region.p) > pw.len()) { return self.terms(k, r, pw, coeff); }
        assert_eq!(r.len(), t.n, "point arity for {k:?}");
        // Plane-split pieces share cached slices of `pw`, keyed by its
        // address: callers pass process-lifetime vectors (the W_G monomials).
        type SubKey = (usize, usize, usize);
        static SUB: once_cell::sync::Lazy<std::sync::Mutex<HashMap<SubKey, std::sync::Arc<Vec<K8>>>>> = once_cell::sync::Lazy::new(Default::default);
        let sub = |plane0: usize, np: usize| -> std::sync::Arc<Vec<K8>> {
            if plane0 == 0 && np == pw.len() { return pw.clone(); }
            let key = (std::sync::Arc::as_ptr(pw) as usize, plane0, np);
            SUB.lock().unwrap().entry(key).or_insert_with(|| std::sync::Arc::new(pw[plane0..plane0 + np].to_vec())).clone()
        };
        t.pieces.iter().filter_map(|pc| {
            let m = pc.region.m;
            let scale = if m >= t.n { coeff } else { eq_bits(&r[m..], pc.hb).mul(&coeff) };
            // An exact zero coefficient contributes nothing to the same row.
            // Test before cloning point/weights: split Boolean W_G slots have
            // many zero-support pieces. Never approximate a field coefficient.
            let pwp = sub(pc.plane0, 1usize << pc.region.p);
            // CUDA v2 derives its bit-region census from retained terms, so
            // retain the original GPU representation until that path is audited.
            #[cfg(not(feature = "cuda"))]
            {
                // Do not hide malformed metadata by deleting an invalid zero term:
                // retain it for the same downstream statement-shape rejection.
                let cap = u32::try_from(self.ell).ok().and_then(|e| 1usize.checked_shl(e));
                let size = m.checked_add(pc.region.p).filter(|&e| e <= self.ell)
                    .and_then(|e| u32::try_from(e).ok()).and_then(|e| 1usize.checked_shl(e));
                let valid_region = match (cap, size) {
                    (Some(cap), Some(size)) => pc.region.source < self.n_sources
                        && pc.region.base % size == 0
                        && pc.region.base.checked_add(size).map_or(false, |end| end <= cap),
                    _ => false,
                };
                if scale.is_zero() && crate::morpheus::wire::canonical_k(&scale) && valid_region
                    && r[..m.min(t.n)].iter().chain(pwp.iter()).all(crate::morpheus::wire::canonical_k)
                { return None; }
            }
            let lo = if m >= t.n {
                let mut lo = r.to_vec();
                lo.resize(m, K8::ZERO);
                lo
            } else {
                r[..m].to_vec()
            };
            Some(Term { region: pc.region.clone(), r: lo, pw: pwp, scale })
        }).collect()
    }

    /// Declared bit blocks per source.
    pub fn bit_blocks(&self) -> Vec<Vec<(usize, usize)>> {
        let mut v = vec![Vec::new(); self.n_sources];
        for t in &self.tensors {
            if !t.bits { continue; }
            for pc in &t.pieces { v[pc.region.source].push((pc.region.base, pc.region.m + pc.region.p)); }
        }
        v
    }

    /// Write tensor `k`'s digits into the raw sources: `digit(plane, index)`.
    pub fn fill(&self, k: &TKey, raw: &mut [Vec<i8>], digit: &(dyn Fn(usize, usize) -> i8 + Sync)) {
        let t = self.get(k);
        for pc in &t.pieces {
            let (m, pp) = (pc.region.m, pc.region.p);
            let dst = &mut raw[pc.region.source][pc.region.base..pc.region.base + (1usize << (m + pp))];
            use rayon::prelude::*;
            let n = t.n;
            dst.par_chunks_mut(1usize << pp).enumerate().for_each(|(i, row)| {
                if m > n && i >= 1usize << n { return; }
                for (b, d) in row.iter_mut().enumerate() { *d = digit(pc.plane0 + b, (pc.hb << m) | i); }
            });
        }
    }
}

/// Disjoint-region writes into the sources from several threads.
#[derive(Clone, Copy)]
pub struct RawPtrs(pub *mut i8, pub usize);
unsafe impl Send for RawPtrs {}
unsafe impl Sync for RawPtrs {}
impl RawPtrs { #[inline] pub fn ptr(&self) -> *mut i8 { self.0 } }

impl Layout {
    /// Same as `fill`, writing through raw pointers: callers may fill
    /// different tensors concurrently (pieces never overlap).
    /// fill_ptr into packed sources (crate::morpheus::packed): each row of
    /// 2^p digits is built then stored (byte-aligned rows directly).
    pub fn fill_packed<F: Fn(usize, usize) -> i8 + Sync>(&self, k: &TKey, ptrs: &[Option<crate::morpheus::packed::PackedPtr>], digit: &F) {
        use crate::morpheus::packed::code;
        self.fill_packed4(k, ptrs, digit, &|b: usize, i: usize| (0..4).fold(0u8, |v, r| v | code(digit(b + r, i)) << (2 * r)));
    }
    /// fill_packed with `quad(b, i)`: the packed byte of planes b..b+4 of
    /// element i (b a multiple of 4), used for rows of 4 or more planes.
    pub fn fill_packed4<F: Fn(usize, usize) -> i8 + Sync, G: Fn(usize, usize) -> u8 + Sync>(&self, k: &TKey, ptrs: &[Option<crate::morpheus::packed::PackedPtr>], digit: &F, quad: &G) {
        use crate::morpheus::packed::code;
        use rayon::prelude::*;
        let t = self.get(k);
        for pc in &t.pieces {
            let (m, pp) = (pc.region.m, pc.region.p);
            let Some(base) = ptrs[pc.region.source] else { continue };
            let n = t.n;
            // Rows past 2^n stay zero.
            let rows = if m > n { 1usize << n } else { 1usize << m };
            let digits = rows << pp;
            let (b0, hb, plane0, mask) = (pc.region.base, pc.hb << m, pc.plane0, (1usize << pp) - 1);
            let at = |j: usize| digit(plane0 + (j & mask), hb | (j >> pp));
            if b0 % 4 != 0 || (1usize << (m + pp)) < 4 {
                // Region smaller than a byte: shares its bytes, atomic OR.
                let row: Vec<i8> = (0..digits).map(at).collect();
                base.write(b0, &row);
                continue;
            }
            // The region owns whole bytes: build each byte and store it.
            assert!((b0 + digits).div_ceil(4) <= base.1);
            let out = unsafe { std::slice::from_raw_parts_mut(base.0.add(b0 / 4), digits.div_ceil(4)) };
            if pp >= 2 && plane0 % 4 == 0 {
                out.par_chunks_mut(1 << 12).enumerate().for_each(|(c, o)| {
                    let j0 = c << 14;
                    for (q, x) in o.iter_mut().enumerate() {
                        let j = j0 + 4 * q;
                        *x = quad(plane0 + (j & mask), hb | (j >> pp));
                    }
                });
                continue;
            }
            out.par_chunks_mut(1 << 12).enumerate().for_each(|(c, o)| {
                let j0 = c << 14;
                for (q, x) in o.iter_mut().enumerate() {
                    let j = j0 + 4 * q;
                    let mut v = 0u8;
                    for r in 0..4 { if j + r < digits { v |= code(at(j + r)) << (2 * r); } }
                    *x = v;
                }
            });
        }
    }
    pub fn fill_ptr(&self, k: &TKey, ptrs: &[RawPtrs], digit: &(dyn Fn(usize, usize) -> i8 + Sync)) {
        let t = self.get(k);
        for pc in &t.pieces {
            let (m, pp) = (pc.region.m, pc.region.p);
            let base = ptrs[pc.region.source];
            let n = t.n;
            use rayon::prelude::*;
            (0..1usize << m).into_par_iter().for_each(|i| {
                if m > n && i >= 1usize << n { return; }
                let row = unsafe { std::slice::from_raw_parts_mut(base.ptr().add(pc.region.base + (i << pp)), 1usize << pp) };
                for (b, d) in row.iter_mut().enumerate() { *d = digit(pc.plane0 + b, (pc.hb << m) | i); }
            });
        }
    }
}

/// Plane weights 2^b of a signed value tensor with 2^p planes.
pub fn value_pw(p: usize) -> Vec<K8> { crate::morpheus::link::value_planes(p) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morpheus::link::term_mle;
    use rand::{Rng, SeedableRng};
    use crate::morpheus::field8::eq_table;
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;

    #[test]
    fn split_pieces_reproduce_the_tensor_mle() {
        let p = Profile::toy(); // cap 12288 -> pieces of at most 2^13
        let specs = vec![(TKey::Edge(1), 12, 3, false), (TKey::Edge(2), 14, 1, false), (TKey::Wg, 5, 9, true), (TKey::Edge(3), 0, 2, false)];
        let lay = Layout::pack_min(&p, &specs, 10);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(3);
        let mut raw = vec![vec![0i8; 1 << p.ell()]; lay.n_sources];
        for &(k, n, pl, _) in &specs {
            let digits: Vec<Vec<i8>> = (0..1usize << pl).map(|_| (0..1usize << n).map(|_| rng.gen_range(-1i8..=1)).collect()).collect();
            lay.fill(&k, &mut raw, &|b, i| digits[b][i]);
            let r: Vec<K8> = (0..n).map(|_| K8(std::array::from_fn(|_| rng.gen::<u64>() % Q))).collect();
            let pw: Vec<K8> = (0..1usize << pl).map(|_| K8(std::array::from_fn(|_| rng.gen::<u64>() % Q))).collect();
            // Direct value.
            let e = eq_table(&r);
            let mut want = K8::ZERO;
            for b in 0..1usize << pl { for i in 0..1usize << n { let d = digits[b][i] as i64; if d != 0 { let v = e[i].mul(&pw[b]); want = if d > 0 { want.add(&v) } else { want.sub(&v) }; } } }
            // Through the raw sources and the terms' MLE weights.
            let mut got = K8::ZERO;
            for t in lay.terms(&k, &r, &pw, K8::ONE) {
                let tb = eq_table(&t.r);
                for b in 0..1usize << t.region.p { for i in 0..1usize << t.region.m.min(n.max(0)).max(if t.region.m > n { n } else { t.region.m }) {
                    let d = raw[t.region.source][t.region.base + (i << t.region.p) + b] as i64;
                    if d != 0 { let v = tb[i].mul(&t.pw[b]).mul(&t.scale); got = if d > 0 { got.add(&v) } else { got.sub(&v) }; }
                } }
                let _ = term_mle;
            }
            assert_eq!(got, want, "{k:?}");
        }
    }
}
