//! Masked same-point batching / authentication (paper §PCS, Eq.
//! pcs_masked_rlc_checks), CPU reference.
//!
//! For each source e, V_e = (W_e, r_e); the auxiliary witness is s_O. With
//! public point x = (ξ, ρ):
//!     F_x(V)    = (A_W W, A_L r, Ev_raw(W; ξ) + ρ(1−ρ) H(r)),
//!     F_O,x(s)  = (A_O s, E_O(s; x)).
//! In each of τ rows: Gaussian masks U_r, U_O,r (addressed private coins);
//! first messages D = F(U); one C0 column β_{r,e} per source and β_{r,O};
//! responses Z_r = U_r + Σ_e β_{r,e} V_e and Z_O,r = U_O,r + β_{r,O} s_O over
//! the integers. One joint decision per attempt: caps first (|Z| < B_Z and
//! per-block Euclidean caps), then the exact clipped coin with
//! x = (‖U‖² − ‖Z‖²)/(2σ²). A rejected attempt is discarded (its first
//! messages never enter the published transcript); the next attempt uses
//! fresh masks at new addresses and attempt-indexed challenges.

use super::coins::{Address, BitStream, Coins, Purpose};
use super::commit::{commit, BlockWitness};
use super::field8::K8;
use super::gaussian::{CoordinateStreams, Stats};
use super::gaussian_fast::{self, FastSigma};
use super::keys::{KeyKind, KeySeed};
use super::params::{Profile, A_ENC_COLS};
use super::pcs_maps::{act_c0_a, act_c0_ring_int, act_c0_ring_q, decode_h, e_o, ev_raw};
use super::rejection::{self, Coin, Ratio};
use super::ring::{AElem, N};
use super::transcript::{Item, MTranscript, Stage};
use num_bigint::{BigInt, BigUint};
use rayon::prelude::*;

/// Opening point x = (ξ, ρ).
#[derive(Clone, Debug)]
pub struct Point { pub xi: Vec<K8>, pub rho: K8 }

impl Point {
    pub fn aux_vars(&self) -> Vec<K8> { let mut v = self.xi.clone(); v.push(self.rho); v }
    fn rho_factor(&self) -> K8 { self.rho.mul(&K8::ONE.sub(&self.rho)) }
}

/// Target / first-message tuple of a source-type block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tuple { pub c: Vec<Vec<u64>>, pub g: Vec<Vec<u64>>, pub a: AElem }
/// Target / first-message tuple of the auxiliary block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuxTuple { pub c: Vec<Vec<u64>>, pub t: AElem }

/// A·cols under key `kind` with `pc` payload columns, then hiding, then the
/// identity error (the [P | B | Id] normal form).
pub fn lin(p: &Profile, seed: &KeySeed, kind: KeyKind, pc: usize, cols: &[Vec<i64>]) -> Vec<Vec<u64>> {
    assert_eq!(cols.len(), pc + p.hiding + p.rank);
    let w = BlockWitness {
        payload: cols[..pc].iter().map(|c| Some(c.clone())).collect(),
        hiding: cols[pc..pc + p.hiding].to_vec(),
        error: cols[pc + p.hiding..].to_vec(),
    };
    commit(p, seed, kind, pc, &w)
}

pub fn f_x(p: &Profile, seed: &KeySeed, pt: &Point, w: &[Vec<i64>], r: &[Vec<i64>]) -> Tuple {
    let c = lin(p, seed, KeyKind::Source, p.source_payload_cols, w);
    let g = lin(p, seed, KeyKind::Link, A_ENC_COLS, r);
    let a = ev_raw(w, &pt.xi).add(&decode_h(&r[..A_ENC_COLS]).scale_k(&pt.rho_factor()));
    Tuple { c, g, a }
}

pub fn f_ox(p: &Profile, seed: &KeySeed, pt: &Point, s: &[Vec<i64>]) -> AuxTuple {
    let c = lin(p, seed, KeyKind::Aux, A_ENC_COLS * p.aux_coeffs, s);
    let t = e_o(&s[..A_ENC_COLS * p.aux_coeffs], &p.degs(), &pt.aux_vars());
    AuxTuple { c, t }
}

pub(crate) fn act_tuple(beta: &[i8; 64], t: &Tuple) -> Tuple {
    Tuple {
        c: t.c.iter().map(|x| act_c0_ring_q(beta, x)).collect(),
        g: t.g.iter().map(|x| act_c0_ring_q(beta, x)).collect(),
        a: act_c0_a(beta, &t.a),
    }
}
pub(crate) fn add_ring(a: &[Vec<u64>], b: &[Vec<u64>]) -> Vec<Vec<u64>> {
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
    a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(&u, &v)| ((u as u128 + v as u128) % Q as u128) as u64).collect()).collect()
}
pub(crate) fn add_tuple(a: &Tuple, b: &Tuple) -> Tuple { Tuple { c: add_ring(&a.c, &b.c), g: add_ring(&a.g, &b.g), a: a.a.add(&b.a) } }

/// A source's witness: full source columns W (payload, hiding, error) and
/// link columns r (8 encoded digit columns, hiding, error).
#[derive(Clone, Debug)]
pub struct SourceVec { pub w: Vec<Vec<i64>>, pub r: Vec<Vec<i64>> }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthProof {
    pub attempt: u16,
    pub d: Vec<Tuple>,
    pub d_o: Vec<AuxTuple>,
    /// Responses in their wire encoding (37 bits per coefficient).
    pub z: Responses,
}

/// Packed responses of the source (W), link (L) and auxiliary (O) blocks,
/// each τ rows of [col][N] coefficients (`wire::pack37`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Responses { pub w: Vec<u8>, pub l: Vec<u8>, pub o: Vec<u8> }

impl Responses {
    /// Per row: source-block columns then link-block columns; auxiliary rows.
    /// None if a length or a code is outside the profile's support.
    pub fn rows(&self, p: &Profile) -> Option<(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)> {
        let (sw, sl, so) = block_cols(p);
        let w = super::wire::rows_of(p, &self.w, sw)?;
        let l = super::wire::rows_of(p, &self.l, sl)?;
        let o = super::wire::rows_of(p, &self.o, so)?;
        Some((w.into_iter().zip(l).map(|(a, b)| [a, b].concat()).collect(), o))
    }
    pub fn from_rows(p: &Profile, z: &[Vec<Vec<i64>>], z_o: &[Vec<Vec<i64>>]) -> Self {
        let sw = p.source_cols();
        let w: Vec<Vec<Vec<i64>>> = z.iter().map(|r| r[..sw].to_vec()).collect();
        let l: Vec<Vec<Vec<i64>>> = z.iter().map(|r| r[sw..].to_vec()).collect();
        Responses { w: super::wire::pack_rows(p, &w), l: super::wire::pack_rows(p, &l), o: super::wire::pack_rows(p, z_o) }
    }
}

#[derive(Debug)]
pub enum AuthError { Exhausted, Sampler(super::gaussian::Abort), CoinAbort }

/// Block dimensions in ring columns: (source, link, aux).
fn block_cols(p: &Profile) -> (usize, usize, usize) { (p.source_cols(), p.link_cols(), p.aux_cols()) }

/// Gaussian columns for coordinates [off, off + cols·N) of row `row`.
fn gaussian_cols(coins: &Coins, opening: u64, attempt: u16, fs: &FastSigma, coord0: u64, cols: usize) -> Result<Vec<Vec<i64>>, AuthError> {
    let flat: Vec<i64> = (0..(cols * N) as u64).into_par_iter().map(|i| {
        let mut s = CoordinateStreams { coins, opening, attempt, coordinate: (coord0 + i) as u32 };
        gaussian_fast::sample(fs, &mut s, &mut Stats::default())
    }).collect::<Result<_, _>>().map_err(AuthError::Sampler)?;
    Ok(flat.chunks(N).map(|c| c.to_vec()).collect())
}

pub(crate) fn challenges_pub(tr: &MTranscript, d: &[Tuple], d_o: &[AuxTuple], attempt: u16, n: usize) -> Vec<Vec<[i8; 64]>> { challenges(tr, d, d_o, attempt, n) }

fn challenges(tr: &MTranscript, d: &[Tuple], d_o: &[AuxTuple], attempt: u16, n: usize) -> Vec<Vec<[i8; 64]>> {
    let mut t = tr.fork();
    for (x, y) in d.iter().zip(d_o) {
        for v in x.c.iter().chain(&x.g).chain(&y.c) { t.absorb(Item::FirstMessage, v); }
        t.absorb_k(Item::FirstMessage, &x.a.0);
        t.absorb_k(Item::FirstMessage, &y.t.0);
    }
    // One query of type (C0^tau)^{n+1}: column e, row r is element e*tau + r.
    let tau = d.len();
    let all = t.challenge_c0_tuple(Stage::AuthColumn, attempt as u64, tau * (n + 1));
    (0..tau).map(|r| (0..=n).map(|e| all[e * tau + r]).collect()).collect()
}

fn sq_norm(cols: &[Vec<i64>]) -> u128 { cols.iter().flatten().map(|&v| (v as i128 * v as i128) as u128).sum() }

/// Caps on one row's blocks; true if all pass.
fn caps_ok(p: &Profile, z: &[Vec<i64>], z_o: &[Vec<i64>]) -> bool {
    let (sw, sl, _) = block_cols(p);
    let bz = p.b_z as i64;
    if z.iter().chain(z_o).flatten().any(|&v| v.abs() >= bz) { return false; }
    let blocks: [(&[Vec<i64>], usize); 3] = [(&z[..sw], sw), (&z[sw..sw + sl], sl), (z_o, z_o.len())];
    blocks.iter().all(|(b, cols)| sq_norm(b) <= p.r2(cols * N))
}

fn combine(beta: &[i8; 64], acc: &mut [Vec<i64>], v: &[Vec<i64>]) {
    for (a, x) in acc.iter_mut().zip(v) {
        let t = act_c0_ring_int(beta, x);
        for (u, w) in a.iter_mut().zip(t) { *u += w; }
    }
}

/// Prover. `tr` holds the complete prefix (statement, commitments, link).
pub fn prove(
    p: &Profile, seed: &KeySeed, coins: &Coins, opening: u64, tr: &MTranscript,
    pt: &Point, sources: &[SourceVec], s_o: &[Vec<i64>],
) -> Result<AuthProof, AuthError> {
    let fs = FastSigma::new(p.sigma);
    let (sw, sl, so) = block_cols(p);
    let row_coords = ((sw + sl + so) * N) as u64;
    let n = sources.len();
    for attempt in 0..p.attempts as u16 {
        let mut u = Vec::new();
        let mut u_o = Vec::new();
        let (mut d, mut d_o) = (Vec::new(), Vec::new());
        for row in 0..p.tau {
            let base = row as u64 * row_coords;
            let uw = gaussian_cols(coins, opening, attempt, &fs, base, sw)?;
            let ul = gaussian_cols(coins, opening, attempt, &fs, base + (sw * N) as u64, sl)?;
            let uo = gaussian_cols(coins, opening, attempt, &fs, base + ((sw + sl) * N) as u64, so)?;
            d.push(f_x(p, seed, pt, &uw, &ul));
            d_o.push(f_ox(p, seed, pt, &uo));
            u.push([uw, ul].concat());
            u_o.push(uo);
        }
        let beta = challenges(tr, &d, &d_o, attempt, n);
        let mut z = u.clone();
        let mut z_o = u_o.clone();
        for row in 0..p.tau {
            for (e, s) in sources.iter().enumerate() {
                combine(&beta[row][e], &mut z[row], &[s.w.clone(), s.r.clone()].concat());
            }
            combine(&beta[row][n], &mut z_o[row], s_o);
        }
        let caps = (0..p.tau).all(|r| caps_ok(p, &z[r], &z_o[r]));
        if caps {
            let nu: u128 = u.iter().chain(&u_o).map(|c| sq_norm(c)).sum();
            let nz: u128 = z.iter().chain(&z_o).map(|c| sq_norm(c)).sum();
            let x = Ratio { num: BigInt::from(nu as i128) - BigInt::from(nz as i128), den: BigInt::from(2u128 * p.sigma as u128 * p.sigma as u128) };
            let mut a = Address::new(Purpose::ProtocolCoin, opening);
            a.attempt = attempt;
            let mut st = coins.stream(a);
            let mut ub = BigUint::from(0u8);
            for _ in 0..rejection::B { ub = (ub << 1u32) + BigUint::from(st.next_bit().map_err(|_| AuthError::CoinAbort)?); }
            match rejection::decide(&x, &ub) {
                Coin::Accept => return Ok(AuthProof { attempt, d, d_o, z: Responses::from_rows(p, &z, &z_o) }),
                Coin::Reject => {}
                Coin::Abort => return Err(AuthError::CoinAbort),
            }
        }
    }
    Err(AuthError::Exhausted)
}

/// Verifier: challenges, caps and the linear authentication equations.
pub fn verify(
    p: &Profile, seed: &KeySeed, tr: &MTranscript, pt: &Point,
    targets: &[Tuple], target_o: &AuxTuple, pf: &AuthProof,
) -> bool {
    if !super::wire::auth_shape_ok(p, pf) || targets.len() > p.n_max
        || pt.xi.len() != p.ell() || !pt.xi.iter().all(super::wire::canonical_k)
        || !super::wire::canonical_k(&pt.rho) || pt.rho == K8::ZERO || pt.rho == K8::ONE
        || !targets.iter().all(|t| super::wire::commitment_ok(p, &t.c)
            && super::wire::commitment_ok(p, &t.g) && super::wire::endpoint_ok(&t.a))
        || !super::wire::commitment_ok(p, &target_o.c) || !super::wire::endpoint_ok(&target_o.t) { return false; }
    let (sw, sl, so) = block_cols(p);
    if (pf.attempt as usize) >= p.attempts || pf.d.len() != p.tau || pf.d_o.len() != p.tau {
        return false;
    }
    let (z, z_o) = match pf.z.rows(p) { Some(x) => x, None => return false };
    let n = targets.len();
    let beta = challenges(tr, &pf.d, &pf.d_o, pf.attempt, n);
    for r in 0..p.tau {
        if z[r].len() != sw + sl || z_o[r].len() != so { return false; }
        if !caps_ok(p, &z[r], &z_o[r]) { return false; }
        let lhs = f_x(p, seed, pt, &z[r][..sw], &z[r][sw..]);
        let mut rhs = pf.d[r].clone();
        for (e, t) in targets.iter().enumerate() { rhs = add_tuple(&rhs, &act_tuple(&beta[r][e], t)); }
        if lhs != rhs { return false; }
        let lo = f_ox(p, seed, pt, &z_o[r]);
        let bt = &beta[r][n];
        let ro = AuxTuple {
            c: add_ring(&pf.d_o[r].c, &target_o.c.iter().map(|x| act_c0_ring_q(bt, x)).collect::<Vec<_>>()),
            t: pf.d_o[r].t.add(&act_c0_a(bt, &target_o.t)),
        };
        if lo != ro { return false; }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::coins::CoinProfile;
    use super::super::pcs_maps::enc;
    use super::super::sampler::ternary_vec;
    use rand::SeedableRng;

    fn rk<R: rand::Rng>(rng: &mut R) -> K8 { use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q; K8(std::array::from_fn(|_| rng.gen::<u64>() % Q)) }

    #[test]
    fn honest_authentication_verifies_and_tampering_fails() {
        let p = Profile::toy();
        let seed = KeySeed([3; 8]);
        let coins = Coins::new([9u8; 64], CoinProfile::Ternary);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(2);
        let tern = |rng: &mut rand_chacha::ChaCha20Rng, cols: usize| -> Vec<Vec<i64>> { (0..cols).map(|_| ternary_vec(rng, N).unwrap()).collect() };
        let sources: Vec<SourceVec> = (0..2).map(|_| {
            let w = tern(&mut rng, p.source_cols());
            let u = AElem((0..64).map(|_| rk(&mut rng)).collect());
            let r = [enc(&u), tern(&mut rng, p.hiding + p.rank)].concat();
            SourceVec { w, r }
        }).collect();
        let aux_payload: Vec<Vec<i64>> = (0..p.aux_coeffs).flat_map(|_| enc(&AElem((0..64).map(|_| rk(&mut rng)).collect()))).collect();
        let s_o = [aux_payload, tern(&mut rng, p.hiding + p.rank)].concat();
        let pt = Point { xi: (0..p.ell()).map(|_| rk(&mut rng)).collect(), rho: rk(&mut rng) };
        let targets: Vec<Tuple> = sources.iter().map(|s| f_x(&p, &seed, &pt, &s.w, &s.r)).collect();
        let target_o = f_ox(&p, &seed, &pt, &s_o);
        let mut tr = MTranscript::new(p.name, &coins.salt(0));
        for t in &targets { for c in &t.c { tr.absorb(Item::SourceCommitment, c); } }

        let pf = prove(&p, &seed, &coins, 0, &tr, &pt, &sources, &s_o).unwrap();
        eprintln!("accepted at attempt {}", pf.attempt);
        assert!(verify(&p, &seed, &tr, &pt, &targets, &target_o, &pf));
        // Wrong endpoint, wrong response, wrong prefix.
        let mut bad = targets.clone();
        bad[1].a.0[0] = bad[1].a.0[0].add(&K8::ONE);
        assert!(!verify(&p, &seed, &tr, &pt, &bad, &target_o, &pf));
        let mut pz = pf.clone();
        // Response coefficient 0 of row 0 moves by one.
        pz.z.w[0] ^= 1;
        assert!(!verify(&p, &seed, &tr, &pt, &targets, &target_o, &pz));
        let mut tr2 = tr.clone();
        tr2.absorb(Item::Other, &[1]);
        assert!(!verify(&p, &seed, &tr2, &pt, &targets, &target_o, &pf));
    }
}
