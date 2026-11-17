//! Witness-recovering link Π_link (paper §PCS, Eqs. pcs_link_norm,
//! opening_auxiliary, pcs_link_eval), CPU reference.
//!
//! Sources are raw digit vectors w_e ∈ {−1,0,1}^{2^ℓ} in logical raw order
//! u = 4096·l + 64·j + k (physical position j + 64k of column l). The payload
//! holds aligned *regions*: a tensor with 2^m entries and 2^p signed digit
//! planes occupies raw indices base + i·2^p + b (value_i = Σ_b 2^b d_{b,i});
//! planes are the low bits, so the link binds each entry's planes first.
//! The suffix holds the source's hiding and identity-error coefficients.
//!
//! Opening claims are mixed linear rows Σ_terms scale·Σ_{b,i} eq(r, i)·pw_b
//! ·raw(W_e)_{base + i 2^p + b} = y. Each term's MLE over the raw index is
//! scale·(Σ_b pw_b eq(b, ξ_low))·eq(r, ξ_mid)·eq(base, ξ_high), so the verifier
//! evaluates every row in O(m + 2^p).
//!
//! The masked sumcheck runs over U = (X, z), X ∈ {0,1}^ℓ first (raw-index bit
//! 0 first), z last, on F_{W,u} + M_O with degree caps (deg_old, deg_final).

use super::auth::{f_ox, f_x, AuxTuple, Point, SourceVec, Tuple};
use super::coins::{Address, BitStream, Coins, Purpose};
use super::field8::{eq_table, invq, K8};
use super::keys::KeySeed;
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::enc;
use super::ring::{phys_of_logical, AElem, DS, N};
use super::transcript::{Item, MTranscript, Stage};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;

/// An aligned tensor region inside a source.
#[derive(Clone, Debug)]
pub struct Region { pub source: usize, pub base: usize, pub m: usize, pub p: usize }

/// One term of a mixed linear row.
#[derive(Clone, Debug)]
pub struct Term { pub region: Region, pub r: Vec<K8>, pub pw: std::sync::Arc<Vec<K8>>, pub scale: K8 }

#[derive(Clone, Debug)]
pub struct Row { pub terms: Vec<Term>, pub y: K8 }

/// Plane weights of a signed-digit value region: 2^b.
pub fn value_planes(p: usize) -> Vec<K8> { (0..1usize << p).map(|b| K8::from_base(((1u128 << b) % Q as u128) as u64)).collect() }

/// The public statement of one opening.
#[derive(Clone, Debug)]
pub struct Statement {
    pub n: usize,
    /// Per source: aligned bit blocks (base, log2 size) declared binary.
    pub bits: Vec<Vec<(usize, usize)>>,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkProof {
    pub g_commit: Vec<Vec<Vec<u64>>>,
    pub c_o: Vec<Vec<u64>>,
    pub rounds: Vec<Vec<K8>>,
    pub a_hat: Vec<AElem>,
    pub t_o: AElem,
}

// ---------------------------------------------------------------- helpers

fn kb(x: i64) -> K8 { K8::from_base(if x >= 0 { x as u64 } else { Q - x.unsigned_abs() }) }

/// MLE of the indicator of an aligned block (base, lg) at ξ.
fn block_mle(base: usize, lg: usize, xi: &[K8]) -> K8 {
    let mut acc = K8::ONE;
    for (i, x) in xi.iter().enumerate().skip(lg) {
        acc = acc.mul(&if (base >> i) & 1 == 1 { *x } else { K8::ONE.sub(x) });
    }
    acc
}

fn eq_point(r: &[K8], x: &[K8]) -> K8 {
    r.iter().zip(x).fold(K8::ONE, |a, (ri, xi)| a.mul(&ri.mul(xi).add(&K8::ONE.sub(ri).mul(&K8::ONE.sub(xi)))))
}

/// MLE of a term's weights over the raw index at ξ.
pub fn term_mle(t: &Term, xi: &[K8]) -> K8 {
    let (m, p) = (t.region.m, t.region.p);
    let eb = eq_table(&xi[..p]);
    let e_r = eq_point(&t.r, &xi[p..p + m]);
    let pw = t.pw.iter().zip(&eb).fold(K8::ZERO, |a, (w, e)| a.add(&w.mul(e)));
    t.scale.mul(&e_r).mul(&pw).mul(&block_mle(t.region.base, m + p, xi))
}

/// Dense weight table of a term (prover side, reference).
fn add_term_table(t: &Term, coef: &K8, table: &mut [K8]) {
    let (m, p) = (t.region.m, t.region.p);
    // A Boolean point selects one index: touch only its 2^p planes.
    if t.r.iter().all(|x| *x == K8::ZERO || *x == K8::ONE) {
        let i = t.r.iter().enumerate().fold(0usize, |a, (j, x)| a | (((*x == K8::ONE) as usize) << j));
        for b in 0..1usize << p {
            let u = t.region.base + (i << p) + b;
            table[u] = table[u].add(&coef.mul(&t.scale).mul(&t.pw[b]));
        }
        return;
    }
    let er = eq_table(&t.r);
    for b in 0..1usize << p {
        let wb = coef.mul(&t.scale).mul(&t.pw[b]);
        for i in 0..1usize << m {
            let u = t.region.base + (i << p) + b;
            table[u] = table[u].add(&wb.mul(&er[i]));
        }
    }
}

/// g(r) from values at 0..=d.
pub fn interpolate(vals: &[K8], r: &K8) -> K8 {
    let d = vals.len();
    let mut acc = K8::ZERO;
    for i in 0..d {
        let mut num = K8::ONE;
        let mut den: u64 = 1;
        for j in 0..d {
            if i == j { continue; }
            num = num.mul(&r.sub(&K8::from_base(j as u64)));
            let diff = i as i64 - j as i64;
            den = ((den as u128 * kb(diff).0[0] as u128) % Q as u128) as u64;
        }
        acc = acc.add(&vals[i].mul(&num).scale(invq(den)));
    }
    acc
}

fn q_of(v: &K8) -> K8 { v.mul(v).mul(v).sub(v) }            // R_0(v) = v(v²−1)
fn b_of(v: &K8) -> K8 { v.mul(v).sub(v) }                   // v(v−1)
fn half() -> K8 { K8::from_base((Q + 1) / 2) }

/// m_i(u) = Σ_k ct(a_ik)(u^k − ½).
fn m_term(a: &[K8], u: &K8) -> K8 {
    let mut acc = K8::ZERO;
    let mut up = K8::ONE;
    for ak in a { up = up.mul(u); acc = acc.add(&ak.mul(&up.sub(&half()))); }
    acc
}

// ---------------------------------------------------------------- coins

fn ternary_coin(coins: &Coins, a: Address) -> i64 {
    if super::coins::no_zk() { return 0; }
    let mut s = coins.stream(a);
    loop {
        let v = (s.next_bit().expect("ternary cap") << 1) | s.next_bit().expect("ternary cap");
        if v < 3 { return v as i64 - 1; }
    }
}
pub fn uniform_q_coin(coins: &Coins, purpose: Purpose, owner: u64, coord: u32) -> u64 {
    if super::coins::no_zk() { return 0; }
    for trial in 0..512u16 {
        let mut a = Address::new(purpose, owner);
        a.coordinate = coord;
        a.trial = trial;
        let mut s = coins.stream(a);
        let mut v = 0u64;
        for _ in 0..64 { v = (v << 1) | s.next_bit().unwrap() as u64; }
        if v < Q { return v; }
    }
    panic!("uniform-q cap");
}
pub fn uniform_a_pub(coins: &Coins, purpose: Purpose, owner: u64, coord0: u32) -> AElem { uniform_a(coins, purpose, owner, coord0) }
pub fn ternary_cols_pub(coins: &Coins, purpose: Purpose, owner: u64, cols: usize) -> Vec<Vec<i64>> { ternary_cols(coins, purpose, owner, cols) }
fn uniform_a(coins: &Coins, purpose: Purpose, owner: u64, coord0: u32) -> AElem {
    AElem((0..DS).map(|m| K8(std::array::from_fn(|a| uniform_q_coin(coins, purpose, owner, coord0 + (m * 8 + a) as u32)))).collect())
}
fn ternary_cols(coins: &Coins, purpose: Purpose, owner: u64, cols: usize) -> Vec<Vec<i64>> {
    (0..cols).map(|c| (0..N).map(|i| {
        let mut a = Address::new(purpose, owner);
        a.coordinate = (c * N + i) as u32;
        ternary_coin(coins, a)
    }).collect()).collect()
}

/// Raw digit vector -> physical source columns.
pub fn raw_to_cols(raw: &[i8]) -> Vec<Vec<i64>> {
    raw.chunks(N).map(|blk| {
        let mut col = vec![0i64; N];
        for (l, &v) in blk.iter().enumerate() { col[phys_of_logical(l)] = v as i64; }
        col
    }).collect()
}

/// Same as `fill_source_suffix`, sampling the coins on the GPU.
pub fn fill_source_suffix_gpu(p: &Profile, coins: &Coins, source_id: u64, raw: &mut [i8]) {
    let start = p.source_payload_cols * N;
    let suf = source_suffix_gpu(p, coins, source_id);
    raw[start..start + suf.len()].copy_from_slice(&suf);
}

/// The suffix digits (hiding then identity error columns, logical order) of
/// source `source_id`, which start at digit `source_payload_cols · N`.
pub fn source_suffix_gpu(p: &Profile, coins: &Coins, source_id: u64) -> Vec<i8> {
    let hid = super::pcs_gpu::ternary_gpu(coins, Purpose::SourceSecret, source_id, p.hiding * N);
    let err = super::pcs_gpu::ternary_gpu(coins, Purpose::SourceError, source_id, p.rank * N);
    let mut out = vec![0i8; (p.hiding + p.rank) * N];
    for (c, col) in hid.chunks(N).chain(err.chunks(N)).enumerate() {
        for l in 0..N { out[c * N + l] = col[phys_of_logical(l)]; }
    }
    out
}

/// Fill a source's suffix (hiding then identity error) from its coins.
pub fn fill_source_suffix(p: &Profile, coins: &Coins, source_id: u64, raw: &mut [i8]) {
    let start = p.source_payload_cols * N;
    let hid = ternary_cols(coins, Purpose::SourceSecret, source_id, p.hiding);
    let err = ternary_cols(coins, Purpose::SourceError, source_id, p.rank);
    for (c, col) in hid.iter().chain(&err).enumerate() {
        for l in 0..N { raw[start + c * N + l] = col[phys_of_logical(l)] as i8; }
    }
}

// ---------------------------------------------------------------- prover

pub struct LinkOutput {
    pub proof: LinkProof,
    pub point: Point,
    pub sources: Vec<SourceVec>,
    /// Int8 link witnesses (GPU services); `sources[e].r` then holds only
    /// the encoded-u columns.
    pub r8: Option<Vec<Vec<i8>>>,
    pub s_o: Vec<Vec<i64>>,
    pub targets: Vec<Tuple>,
    pub target_o: AuxTuple,
}

/// The initial bundle (alpha, gamma, eta_O): one indexed query of type
/// K^{l+2}, after every source, link and auxiliary commitment.
pub(crate) fn challenges_scalar(tr: &mut MTranscript, ell: usize) -> (Vec<K8>, K8, K8) {
    let mut b = tr.challenge_k_bundle(Stage::LinkInit, 0, ell + 2);
    let eta = b.pop().unwrap();
    let gamma = b.pop().unwrap();
    (b, gamma, eta)
}

fn pows(g: &K8, n: usize) -> Vec<K8> {
    let mut v = Vec::with_capacity(n);
    let mut x = K8::ONE;
    for _ in 0..n { v.push(x); x = x.mul(g); }
    v
}

/// Prover. `raw[e]` are complete sources (payload + suffix); `tr` already
/// holds the statement and the source commitments.
pub fn prove(
    p: &Profile, seed: &KeySeed, coins: &Coins, opening: u64, tr: &mut MTranscript,
    st: &Statement, raw: &[Vec<i8>],
) -> LinkOutput {
    let ell = p.ell();
    let n = st.n;
    let degs = p.degs();
    assert_eq!(raw.len(), n);
    // Link masks and auxiliary witness from addressed coins.
    let owner = |e: usize| opening * p.n_max as u64 + e as u64;
    let u: Vec<AElem> = (0..n).map(|e| uniform_a(coins, Purpose::LinkPayload, owner(e), 0)).collect();
    let r_cols: Vec<Vec<Vec<i64>>> = (0..n).map(|e| [enc(&u[e]), ternary_cols(coins, Purpose::LinkSecret, owner(e), p.hiding), ternary_cols(coins, Purpose::LinkError, owner(e), p.rank)].concat()).collect();
    let a_aux: Vec<AElem> = (0..p.aux_coeffs).map(|i| uniform_a(coins, Purpose::AuxPayload, opening, (i * 512) as u32)).collect();
    let s_o: Vec<Vec<i64>> = [a_aux.iter().flat_map(enc).collect::<Vec<_>>(), ternary_cols(coins, Purpose::AuxSecret, opening, p.hiding), ternary_cols(coins, Purpose::AuxError, opening, p.rank)].concat();
    // Commitments G_e and C_O, then the scalar challenges.
    let g_commit: Vec<Vec<Vec<u64>>> = r_cols.iter().map(|r| super::auth::lin(p, seed, super::keys::KeyKind::Link, A_ENC_COLS, r)).collect();
    let c_o = super::auth::lin(p, seed, super::keys::KeyKind::Aux, A_ENC_COLS * p.aux_coeffs, &s_o);
    for g in &g_commit { for c in g { tr.absorb(Item::LinkCommitment, c); } }
    for c in &c_o { tr.absorb(Item::AuxCommitment, c); }
    let (alpha, gamma, eta) = challenges_scalar(tr, ell);
    let gp = pows(&gamma, 2 * n + st.rows.len());

    // Dense tables (reference): w_e, eq(α), χ_b,e, q_e.
    let size = 1usize << ell;
    let mut w: Vec<Vec<K8>> = raw.iter().map(|r| r.iter().map(|&d| kb(d as i64)).collect()).collect();
    let mut eqa = eq_table(&alpha);
    let mut chi: Vec<Vec<K8>> = st.bits.iter().map(|bl| {
        let mut t = vec![K8::ZERO; size];
        for &(b, lg) in bl { for i in 0..1usize << lg { t[b + i] = K8::ONE; } }
        t
    }).collect();
    let mut q: Vec<Vec<K8>> = vec![vec![K8::ZERO; size]; n];
    for (j, row) in st.rows.iter().enumerate() {
        let c = eta.mul(&gp[2 * n + j]);
        for t in &row.terms { add_term_table(t, &c, &mut q[t.region.source]); }
    }
    let ct_a: Vec<Vec<K8>> = {
        let mut v = Vec::new();
        let mut idx = 0;
        for &d in &degs { v.push((0..d).map(|k| a_aux[idx + k].0[0]).collect()); idx += d; }
        v
    };

    let nvars = ell + 1;
    let mut rounds = Vec::new();
    let mut xi: Vec<K8> = Vec::new();
    let mut mo_prefix = K8::ZERO; // Σ_{j<i} m_j(r_j)
    for i in 0..ell {
        let d = degs[i];
        let half_len = w[0].len() / 2;
        let scale_rest = K8::from_base(1u64 << (nvars - 1 - i));
        let mut g = vec![K8::ZERO; d + 1];
        for (t, gt) in g.iter_mut().enumerate() {
            let tk = K8::from_base(t as u64);
            let lerp = |a: &K8, b: &K8| a.add(&tk.mul(&b.sub(a)));
            use rayon::prelude::*;
            let acc = (0..half_len).into_par_iter().map(|k| {
                let mut acc = K8::ZERO;
                let e = lerp(&eqa[2 * k], &eqa[2 * k + 1]);
                for s in 0..n {
                    if w[s][2 * k].is_zero() && w[s][2 * k + 1].is_zero() { continue; }
                    let wv = lerp(&w[s][2 * k], &w[s][2 * k + 1]);
                    let cv = lerp(&chi[s][2 * k], &chi[s][2 * k + 1]);
                    let qv = lerp(&q[s][2 * k], &q[s][2 * k + 1]);
                    let range = gp[s].mul(&q_of(&wv)).add(&gp[n + s].mul(&cv).mul(&b_of(&wv)));
                    acc = acc.add(&e.mul(&range)).add(&qv.mul(&wv));
                }
                acc
            }).reduce(|| K8::ZERO, |a, b| a.add(&b));
            *gt = acc.add(&scale_rest.mul(&mo_prefix.add(&m_term(&ct_a[i], &tk))));
        }
        tr.absorb_k(Item::RoundPoly, &g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        rounds.push(g);
        mo_prefix = mo_prefix.add(&m_term(&ct_a[i], &r));
        let fold = |t: &mut Vec<K8>| { use rayon::prelude::*; let h = t.len() / 2; *t = (0..h).into_par_iter().map(|k| t[2 * k].add(&r.mul(&t[2 * k + 1].sub(&t[2 * k])))).collect(); };
        fold(&mut eqa);
        for s in 0..n { fold(&mut w[s]); fold(&mut chi[s]); fold(&mut q[s]); }
        xi.push(r);
    }
    // Final variable z.
    let d = degs[ell];
    let mut g = vec![K8::ZERO; d + 1];
    for (t, gt) in g.iter_mut().enumerate() {
        let tk = K8::from_base(t as u64);
        let zz = tk.mul(&K8::ONE.sub(&tk));
        let mut acc = K8::ZERO;
        for s in 0..n {
            let v = w[s][0].add(&zz.mul(&u[s].0[0]));
            let range = gp[s].mul(&q_of(&v)).add(&gp[n + s].mul(&chi[s][0]).mul(&b_of(&v)));
            acc = acc.add(&eqa[0].mul(&range)).add(&q[s][0].mul(&v));
        }
        *gt = acc.mul(&half()).add(&mo_prefix.add(&m_term(&ct_a[ell], &tk)));
    }
    tr.absorb_k(Item::RoundPoly, &g);
    let rho = tr.challenge_k_not01(Stage::LinkRound, 0, ell as u64);
    rounds.push(g);

    let point = Point { xi, rho };
    let sources: Vec<SourceVec> = raw.iter().zip(&r_cols).map(|(rw, r)| SourceVec { w: raw_to_cols(rw), r: r.clone() }).collect();
    let targets: Vec<Tuple> = sources.iter().map(|s| f_x(p, seed, &point, &s.w, &s.r)).collect();
    let target_o = f_ox(p, seed, &point, &s_o);
    let a_hat: Vec<AElem> = targets.iter().map(|t| t.a.clone()).collect();
    for a in &a_hat { tr.absorb_k(Item::Endpoint, &a.0); }
    tr.absorb_k(Item::Endpoint, &target_o.t.0);
    let proof = LinkProof { g_commit, c_o, rounds, a_hat, t_o: target_o.t.clone() };
    LinkOutput { proof, point, sources, s_o, targets, target_o, r8: None }
}

// ---------------------------------------------------------------- verifier

/// Verifies the link and returns the point and the authentication targets
/// (with the commitments taken from the statement and the proof).
fn statement_shape_ok(p: &Profile, st: &Statement) -> bool {
    let ell = p.ell();
    let Some(cap) = 1usize.checked_shl(ell as u32) else { return false; };
    let block_ok = |base: usize, lg: usize| -> bool {
        if lg > ell { return false; }
        let Some(size) = 1usize.checked_shl(lg as u32) else { return false; };
        base % size == 0 && base.checked_add(size).map_or(false, |end| end <= cap)
    };
    st.n <= p.n_max && st.bits.len() == st.n
        && st.bits.iter().flatten().all(|&(base, lg)| block_ok(base, lg))
        && st.rows.iter().all(|row| super::wire::canonical_k(&row.y) && row.terms.iter().all(|t| {
            let Some(lg) = t.region.m.checked_add(t.region.p) else { return false; };
            t.region.source < st.n && block_ok(t.region.base, lg)
                && t.r.len() == t.region.m
                && 1usize.checked_shl(t.region.p as u32) == Some(t.pw.len())
                && t.r.iter().chain(t.pw.iter()).all(super::wire::canonical_k)
                && super::wire::canonical_k(&t.scale)
        }))
}

pub fn verify(
    p: &Profile, tr: &mut MTranscript, st: &Statement, c_src: &[Vec<Vec<u64>>], pf: &LinkProof,
) -> Option<(Point, Vec<Tuple>, AuxTuple)> {
    // Validate all externally supplied shapes before indexing or transcript work.
    if !super::wire::link_shape_ok(p, st.n, pf)
        || c_src.len() != st.n || !c_src.iter().all(|c| super::wire::commitment_ok(p, c))
        || !statement_shape_ok(p, st) { return None; }
    let ell = p.ell();
    let n = st.n;
    let degs = p.degs();
    if pf.rounds.len() != ell + 1 || pf.a_hat.len() != n || pf.g_commit.len() != n || c_src.len() != n { return None; }
    // The setup cap n <= n_max is part of the statement's validity.
    if n > p.n_max { return None; }
    for g in &pf.g_commit { for c in g { tr.absorb(Item::LinkCommitment, c); } }
    for c in &pf.c_o { tr.absorb(Item::AuxCommitment, c); }
    let (alpha, gamma, eta) = challenges_scalar(tr, ell);
    let gp = pows(&gamma, 2 * n + st.rows.len());
    let mut claim = st.rows.iter().enumerate().fold(K8::ZERO, |a, (j, row)| a.add(&eta.mul(&gp[2 * n + j]).mul(&row.y)));
    let mut xi = Vec::new();
    for i in 0..ell {
        let g = &pf.rounds[i];
        if g.len() != degs[i] + 1 || g[0].add(&g[1]) != claim { eprintln!("link: round {i} sum mismatch"); return None; }
        tr.absorb_k(Item::RoundPoly, g);
        let r = tr.challenge_k(Stage::LinkRound, 0, i as u64);
        claim = interpolate(g, &r);
        xi.push(r);
    }
    let g = &pf.rounds[ell];
    if g.len() != degs[ell] + 1 || g[0].add(&g[1]) != claim { return None; }
    tr.absorb_k(Item::RoundPoly, g);
    let rho = tr.challenge_k_not01(Stage::LinkRound, 0, ell as u64);
    let final_claim = interpolate(g, &rho);
    // Φ_x(â) + ct_S(t_O).
    let eqa = eq_point(&alpha, &xi);
    let mut acc = K8::ZERO;
    for e in 0..n {
        let v = pf.a_hat[e].0[0];
        let chi = st.bits[e].iter().fold(K8::ZERO, |a, &(b, lg)| a.add(&block_mle(b, lg, &xi)));
        acc = acc.add(&eqa.mul(&gp[e].mul(&q_of(&v)).add(&gp[n + e].mul(&chi).mul(&b_of(&v)))));
    }
    // Σ_rows η γ^j Σ_terms term_mle·â: eq(ξ_[0..p)) per plane count and
    // <pw, eq> per distinct weight vector are shared (W_G terms share pw).
    {
        use rayon::prelude::*;
        use std::collections::HashMap;
        let maxp = st.rows.iter().flat_map(|r| r.terms.iter().map(|t| t.region.p)).max().unwrap_or(0);
        let ebs: Vec<Vec<K8>> = (0..=maxp).map(|p| eq_table(&xi[..p])).collect();
        let mut keys: Vec<(usize, usize)> = st.rows.iter().flat_map(|r| r.terms.iter().map(|t| (std::sync::Arc::as_ptr(&t.pw) as usize, t.region.p))).collect();
        keys.sort_unstable();
        keys.dedup();
        let mut by_key: HashMap<(usize, usize), &Term> = HashMap::new();
        for r in &st.rows { for t in &r.terms { by_key.entry((std::sync::Arc::as_ptr(&t.pw) as usize, t.region.p)).or_insert(t); } }
        let dots: HashMap<(usize, usize), K8> = keys.par_iter().map(|&k| {
            let t = by_key[&k];
            (k, t.pw.iter().zip(&ebs[k.1]).fold(K8::ZERO, |a, (w, e)| a.add(&w.mul(e))))
        }).collect();
        let s = st.rows.par_iter().enumerate().map(|(j, row)| {
            let c = eta.mul(&gp[2 * n + j]);
            row.terms.iter().fold(K8::ZERO, |a, t| {
                let (m, p) = (t.region.m, t.region.p);
                let pw = dots[&(std::sync::Arc::as_ptr(&t.pw) as usize, p)];
                let v = t.scale.mul(&eq_point(&t.r, &xi[p..p + m])).mul(&pw).mul(&block_mle(t.region.base, m + p, &xi));
                a.add(&c.mul(&v).mul(&pf.a_hat[t.region.source].0[0]))
            })
        }).reduce(|| K8::ZERO, |a, b| a.add(&b));
        acc = acc.add(&s);
    }
    if acc.mul(&half()).add(&pf.t_o.0[0]) != final_claim { eprintln!("link: final check mismatch"); return None; }
    for a in &pf.a_hat { tr.absorb_k(Item::Endpoint, &a.0); }
    tr.absorb_k(Item::Endpoint, &pf.t_o.0);
    let point = Point { xi, rho };
    let targets = (0..n).map(|e| Tuple { c: c_src[e].clone(), g: pf.g_commit[e].clone(), a: pf.a_hat[e].clone() }).collect();
    Some((point, targets, AuxTuple { c: pf.c_o.clone(), t: pf.t_o.clone() }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::auth;
    use super::super::coins::CoinProfile;
    use rand::{Rng, SeedableRng};

    /// Two sources with value tensors, a bit block, and rows (one mixing
    /// both sources); the full PCS (link + authentication) verifies, and a
    /// false claim or a non-ternary digit is rejected.
    #[test]
    fn link_and_authentication_end_to_end() {
        let p = Profile::toy();
        let seed = KeySeed([5; 8]);
        let coins = Coins::new([4u8; 64], CoinProfile::Ternary);
        let ell = p.ell();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(7);
        let n = 2;
        let mut raw = vec![vec![0i8; 1 << ell]; n];
        // Source 0: tensor A (2^8 entries, 4 planes) at 0, tensor B (2^6, 2 planes) at 1024.
        // Source 1: tensor C (2^9, 4 planes) at 0 and a bit block (2^7) at 4096.
        let regs = [Region { source: 0, base: 0, m: 8, p: 2 }, Region { source: 0, base: 1024, m: 6, p: 1 }, Region { source: 1, base: 0, m: 9, p: 2 }];
        let mut vals: Vec<Vec<i64>> = Vec::new();
        for r in &regs {
            let planes = 1usize << r.p;
            let mut v = vec![0i64; 1 << r.m];
            for (i, vi) in v.iter_mut().enumerate() {
                let x: i64 = rng.gen_range(-(1i64 << planes) + 1..(1i64 << planes));
                *vi = x;
                for b in 0..planes {
                    let d = (x.abs() >> b) & 1;
                    raw[r.source][r.base + (i << r.p) + b] = (d * x.signum()) as i8;
                }
            }
            vals.push(v);
        }
        for i in 0..128 { raw[1][4096 + i] = rng.gen_range(0..2); }
        for e in 0..n { fill_source_suffix(&p, &coins, e as u64, &mut raw[e]); }
        let rk = |rng: &mut rand_chacha::ChaCha20Rng| K8(std::array::from_fn(|_| rng.gen::<u64>() % Q));
        let mle = |v: &[i64], r: &[K8]| -> K8 { let e = eq_table(r); v.iter().zip(&e).fold(K8::ZERO, |a, (&x, w)| a.add(&w.mul(&kb(x)))) };
        let mut rows = Vec::new();
        for (ri, r) in regs.iter().enumerate() {
            let pt: Vec<K8> = (0..r.m).map(|_| rk(&mut rng)).collect();
            let y = mle(&vals[ri], &pt);
            rows.push(Row { terms: vec![Term { region: r.clone(), r: pt, pw: std::sync::Arc::new(value_planes(r.p)), scale: K8::ONE }], y });
        }
        // Mixed row: 3·A(r) − 2·C(r') spanning both sources.
        let (pa, pc): (Vec<K8>, Vec<K8>) = ((0..8).map(|_| rk(&mut rng)).collect(), (0..9).map(|_| rk(&mut rng)).collect());
        let y = mle(&vals[0], &pa).scale(3).sub(&mle(&vals[2], &pc).scale(2));
        rows.push(Row { terms: vec![
            Term { region: regs[0].clone(), r: pa, pw: std::sync::Arc::new(value_planes(2)), scale: K8::from_base(3) },
            Term { region: regs[2].clone(), r: pc, pw: std::sync::Arc::new(value_planes(2)), scale: kb(-2) },
        ], y });
        let st = Statement { n, bits: vec![vec![], vec![(4096, 7)]], rows };

        let c_src: Vec<Vec<Vec<u64>>> = raw.iter().map(|r| auth::lin(&p, &seed, super::super::keys::KeyKind::Source, p.source_payload_cols, &raw_to_cols(r))).collect();
        let mk_tr = || { let mut t = MTranscript::new(p.name, &coins.salt(0)); for c in &c_src { for x in c { t.absorb(Item::SourceCommitment, x); } } t };
        let mut tp = mk_tr();
        let out = prove(&p, &seed, &coins, 0, &mut tp, &st, &raw);
        // The structured prover produces the identical proof and transcript.
        for b0 in [6usize, 7] {
            let mut tf = mk_tr();
            let fast = super::super::link_fast::prove(&p, &seed, &coins, 0, &mut tf, &st, &raw, b0);
            assert_eq!(fast.proof.rounds, out.proof.rounds, "B0 = {b0}");
            assert_eq!(fast.proof.a_hat, out.proof.a_hat);
            assert_eq!(fast.point.rho, out.point.rho);
        }
        let apf = auth::prove(&p, &seed, &coins, 0, &tp, &out.point, &out.sources, &out.s_o).unwrap();

        let mut tv = mk_tr();
        let (pt, targets, target_o) = verify(&p, &mut tv, &st, &c_src, &out.proof).expect("link verifies");
        assert_eq!(pt.xi, out.point.xi);
        assert!(auth::verify(&p, &seed, &tv, &pt, &targets, &target_o, &apf));

        // A false claim value fails the link.
        let mut bad = st.clone();
        bad.rows[1].y = bad.rows[1].y.add(&K8::ONE);
        let mut tv = mk_tr();
        assert!(verify(&p, &mut tv, &bad, &c_src, &out.proof).is_none());
        // A non-ternary digit makes the honest-looking prover's sum wrong.
        let mut raw2 = raw.clone();
        raw2[0][3] = 2;
        let mut tp2 = mk_tr();
        let out2 = prove(&p, &seed, &coins, 0, &mut tp2, &st, &raw2);
        let mut tv = mk_tr();
        assert!(verify(&p, &mut tv, &st, &c_src, &out2.proof).is_none());
    }
}
