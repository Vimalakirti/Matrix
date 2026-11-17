//! Proof wire format v4: SHAKE256 typed-prefix/public-key profile. The reported proof size is the length of this
//! encoding, not an estimate.
//!
//! * Field elements of F_q and ring coefficients of commitments: 8 bytes
//!   little-endian, canonical (< q). A K element is 8 of them.
//! * Authentication responses: |Z| < B_Z, so a coefficient has 2·B_Z − 1
//!   permitted values. In memory a block holds u = Z + (B_Z − 1) in 37 bits,
//!   LSB first, eight coefficients per 37 bytes (`pack37`), in the order
//!   block (W, L, O), row, column, coefficient. On the wire each block is a
//!   u32 length and a lossless entropy code of the same coefficients
//!   (`zenc`/`zdec`): Z = (h << 30) + low, the 30 low bits raw and h
//!   rANS-coded under a fixed table for the profile's Gaussian (σ ≈ 3.08e9),
//!   in independent chunks of 2^16 coefficients. About 33.6 bits per
//!   coefficient instead of 37. The decoder requires every |Z| < B_Z and
//!   restores the packed bytes exactly; the transcript never sees this code.
//! * Every variable-length list carries a u32 length; the decoder rejects
//!   trailing bytes and any length that does not match the profile.

use super::auth::{AuthProof, AuxTuple, Responses, Tuple};
use super::field8::K8;
use super::link::LinkProof;
use super::params::Profile;
use super::ring::{AElem, DS, N};
use rayon::prelude::*;

/// Request/finalization semantics version, also absorbed before challenges.
pub use super::protocol::{PROTOCOL_VERSION, MAGIC};
use super::protocol::accepts_wire_magic;
pub const Z_BITS: usize = 37;

/// Offset and largest code of the response codec.
pub fn z_off(p: &Profile) -> i64 { p.b_z as i64 - 1 }
pub fn z_umax(p: &Profile) -> u64 { 2 * p.b_z - 2 }
/// Bytes of `n` packed coefficients (n a multiple of 8).
pub fn packed_len(n: usize) -> usize { assert_eq!(n % 8, 0); n / 8 * Z_BITS }

fn pack_group(g: &[i64], off: i64, o: &mut [u8]) {
    let mut w = [0u64; 5];
    for (t, &v) in g.iter().enumerate() {
        let u = ((v + off) as u64) & ((1 << Z_BITS) - 1);
        let (bp, i) = (Z_BITS * t, Z_BITS * t / 64);
        let sh = bp % 64;
        w[i] |= u << sh;
        if sh + Z_BITS > 64 { w[i + 1] |= u >> (64 - sh); }
    }
    for (b, x) in o.iter_mut().enumerate() { *x = (w[b / 8] >> (8 * (b % 8))) as u8; }
}
/// The eight raw 37-bit codes of one packed group (no range check).
fn unpack_group(p: &[u8], g: &mut [u64; 8]) {
    let mut w = [0u64; 5];
    for (i, &x) in p.iter().enumerate() { w[i / 8] |= (x as u64) << (8 * (i % 8)); }
    for (t, u) in g.iter_mut().enumerate() {
        let (bp, i) = (Z_BITS * t, Z_BITS * t / 64);
        let sh = bp % 64;
        *u = w[i] >> sh;
        if sh + Z_BITS > 64 { *u |= w[i + 1] << (64 - sh); }
        *u &= (1 << Z_BITS) - 1;
    }
}

pub fn pack37(z: &[i64], off: i64) -> Vec<u8> {
    let mut out = vec![0u8; packed_len(z.len())];
    out.par_chunks_mut(Z_BITS).zip(z.par_chunks(8)).for_each(|(o, g)| pack_group(g, off, o));
    out
}

/// Inverse of `pack37`; None if a code exceeds `umax` or the length is off.
pub fn unpack37(b: &[u8], n: usize, off: i64, umax: u64) -> Option<Vec<i64>> {
    if n % 8 != 0 || b.len() != packed_len(n) { return None; }
    let mut z = vec![0i64; n];
    let ok = z.par_chunks_mut(8).zip(b.par_chunks(Z_BITS)).all(|(g, p)| {
        let mut u = [0u64; 8];
        unpack_group(p, &mut u);
        for (v, &u) in g.iter_mut().zip(&u) { *v = u as i64 - off; }
        u.iter().all(|&u| u <= umax)
    });
    ok.then_some(z)
}

// ---------------------------------------------------------- response codec
//
// A 37-bit code u is split as Z = u - off = (h << 30) + low, low in [0, 2^30),
// h in [-128, 128). This covers every 37-bit code, so encoding is total and the
// range check u <= umax happens on decode. Per chunk of ZC_CHUNK coefficients:
// u32 rANS byte length, the rANS stream of the h (u64 final state, then u32
// words, LE), then the 30-bit lows packed LSB first (no padding: the chunk
// size is a multiple of 8). Chunk count and low length follow from n.

const ZC_LOW: u32 = 30;
const ZC_CHUNK: usize = 1 << 16;
const ZC_P: u32 = 16;
const RANS_L: u64 = 1 << 31;
/// Static frequencies (sum 2^16) of h = Z >> 30 for Z ~ D_sigma, sigma =
/// 3_076_346_150: round(2^16 Pr[h]) floored at 1, the -230 rounding surplus
/// taken evenly from h = -1, 0. Only h in [-12, 11] exceed 1.
const ZC_MID: [u32; 24] = [3, 12, 39, 116, 305, 711, 1465, 2678, 4338, 6229, 7927, 8829,
    8829, 7927, 6229, 4338, 2678, 1465, 711, 305, 116, 39, 12, 3];
/// Encoder entries [x_max, rcp, bias, 2^P - f, shift]: x / f by multiply-high (ryg rans64).
struct Tab { f: [u32; 256], c: [u32; 257], s: Vec<u8>, e: [[u64; 5]; 256] }
fn tab() -> &'static Tab {
    static T: std::sync::OnceLock<Tab> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut f = [1u32; 256];
        f[116..140].copy_from_slice(&ZC_MID);
        let mut c = [0u32; 257];
        for i in 0..256 { c[i + 1] = c[i] + f[i]; }
        assert_eq!(c[256], 1 << ZC_P);
        let mut s = vec![0u8; 1 << ZC_P];
        for i in 0..256 { s[c[i] as usize..c[i + 1] as usize].fill(i as u8); }
        let e = std::array::from_fn(|i| {
            let (f, c) = (f[i] as u64, c[i] as u64);
            let xm = ((RANS_L >> ZC_P) << 32) * f;
            if f < 2 { return [xm, !0, c + (1 << ZC_P) - 1, (1 << ZC_P) - f, 0]; }
            let sh = 64 - (f - 1).leading_zeros() as u64;
            [xm, (((1u128 << (sh + 63)) + f as u128 - 1) / f as u128) as u64, c, (1 << ZC_P) - f, sh - 1]
        });
        Tab { f, c, s, e }
    })
}

fn zenc_chunk(t: &Tab, b: &[u8], off: i64) -> Vec<u8> {
    let n = b.len() / Z_BITS * 8;
    let (mut sym, mut low) = (Vec::with_capacity(n), Vec::with_capacity(n / 8 * ZC_LOW as usize));
    let (mut acc, mut nb) = (0u64, 0u32);
    for p in b.chunks(Z_BITS) {
        let mut u = [0u64; 8];
        unpack_group(p, &mut u);
        for u in u {
            let z = u as i64 - off;
            sym.push(((z >> ZC_LOW) + 128) as u8);
            acc |= (z as u64 & ((1 << ZC_LOW) - 1)) << nb;
            nb += ZC_LOW;
            while nb >= 8 { low.push(acc as u8); acc >>= 8; nb -= 8; }
        }
    }
    let (mut x, mut words) = (RANS_L, Vec::<u32>::with_capacity(n / 8));
    for &s in sym.iter().rev() {
        let [xm, rcp, bias, cmpl, sh] = t.e[s as usize];
        if x >= xm { words.push(x as u32); x >>= 32; }
        x += bias + ((((x as u128 * rcp as u128) >> 64) as u64) >> sh) * cmpl;
    }
    let mut o = Vec::with_capacity(12 + 4 * words.len() + low.len());
    o.extend_from_slice(&(8 + 4 * words.len() as u32).to_le_bytes());
    o.extend_from_slice(&x.to_le_bytes());
    for w in words.iter().rev() { o.extend_from_slice(&w.to_le_bytes()); }
    o.extend_from_slice(&low);
    o
}

fn zdec_chunk(t: &Tab, r: &[u8], low: &[u8], off: i64, umax: u64, o: &mut [u8]) -> Option<()> {
    if r.len() < 8 || r.len() % 4 != 0 { return None; }
    let mut x = u64::from_le_bytes(r[..8].try_into().ok()?);
    // The state stays in [L, 2^63), so no step overflows on any input.
    if !(RANS_L..1 << 63).contains(&x) { return None; }
    let (mut ws, mut lb) = (r[8..].chunks_exact(4), low.iter());
    let (mut acc, mut nb) = (0u64, 0u32);
    for g in o.chunks_mut(Z_BITS) {
        let mut z = [0i64; 8];
        for v in z.iter_mut() {
            let slot = x & ((1 << ZC_P) - 1);
            let s = t.s[slot as usize] as usize;
            x = t.f[s] as u64 * (x >> ZC_P) + slot - t.c[s] as u64;
            if x < RANS_L { x = x << 32 | u32::from_le_bytes(ws.next()?.try_into().ok()?) as u64; }
            while nb < ZC_LOW { acc |= (*lb.next()? as u64) << nb; nb += 8; }
            *v = ((s as i64 - 128) << ZC_LOW) + (acc & ((1 << ZC_LOW) - 1)) as i64;
            acc >>= ZC_LOW;
            nb -= ZC_LOW;
            let u = *v + off;
            if u < 0 || u as u64 > umax { return None; }
        }
        pack_group(&z, off, g);
    }
    (x == RANS_L && ws.next().is_none() && lb.next().is_none()).then_some(())
}

/// Wire form of packed responses (any 37-bit codes; the length must be whole groups).
pub fn zenc(b: &[u8], off: i64) -> Vec<u8> { zenc_chunks(b, off).concat() }
fn zenc_chunks(b: &[u8], off: i64) -> Vec<Vec<u8>> {
    assert_eq!(b.len() % Z_BITS, 0, "response bytes");
    let t = tab();
    b.par_chunks(packed_len(ZC_CHUNK)).map(|c| zenc_chunk(t, c, off)).collect()
}
/// Packed responses of `n` coefficients from their wire form; None unless the
/// stream is exactly well formed and every code is at most `umax`.
pub fn zdec(e: &[u8], n: usize, off: i64, umax: u64) -> Option<Vec<u8>> {
    if n % 8 != 0 { return None; }
    let mut r = R::new(e);
    let segs = (0..n).step_by(ZC_CHUNK).map(|i| {
        let rans = r.blob()?;
        Some((rans, r.take((n - i).min(ZC_CHUNK) / 8 * ZC_LOW as usize)?))
    }).collect::<Option<Vec<_>>>()?;
    if !r.done() { return None; }
    let (t, mut out) = (tab(), vec![0u8; packed_len(n)]);
    out.par_chunks_mut(packed_len(ZC_CHUNK)).zip(segs.par_iter())
        .all(|(o, &(a, l))| zdec_chunk(t, a, l, off, umax, o).is_some()).then_some(out)
}

/// Rows [row][col][N] of a block from its packed bytes.
pub fn rows_of(p: &Profile, bytes: &[u8], cols: usize) -> Option<Vec<Vec<Vec<i64>>>> {
    let z = unpack37(bytes, p.tau * cols * N, z_off(p), z_umax(p))?;
    Some(z.chunks(cols * N).map(|r| r.chunks(N).map(|c| c.to_vec()).collect()).collect())
}
/// Packed bytes of rows [row][col][N].
pub fn pack_rows(p: &Profile, rows: &[Vec<Vec<i64>>]) -> Vec<u8> {
    let flat: Vec<i64> = rows.iter().flatten().flatten().copied().collect();
    pack37(&flat, z_off(p))
}

// ------------------------------------------------------------------ writer

#[derive(Default)]
pub struct W { pub b: Vec<u8> }

impl W {
    pub fn u8(&mut self, x: u8) { self.b.push(x); }
    pub fn u16(&mut self, x: u16) { self.b.extend_from_slice(&x.to_le_bytes()); }
    pub fn u32(&mut self, x: u32) { self.b.extend_from_slice(&x.to_le_bytes()); }
    pub fn len(&mut self, n: usize) { self.u32(u32::try_from(n).expect("wire length")); }
    pub fn u64s(&mut self, v: &[u64]) {
        self.len(v.len());
        self.b.reserve(8 * v.len());
        for x in v { self.b.extend_from_slice(&x.to_le_bytes()); }
    }
    pub fn ring_vec(&mut self, v: &[Vec<u64>]) { self.len(v.len()); for x in v { self.u64s(x); } }
    pub fn k8(&mut self, x: &K8) { for c in &x.0 { self.b.extend_from_slice(&c.to_le_bytes()); } }
    pub fn k8s(&mut self, v: &[K8]) { self.len(v.len()); for x in v { self.k8(x); } }
    pub fn a(&mut self, a: &AElem) { self.k8s(&a.0); }
    pub fn bytes(&mut self, v: &[u8]) { self.len(v.len()); self.b.extend_from_slice(v); }
}

pub struct R<'a> { b: &'a [u8], pub pos: usize }

impl<'a> R<'a> {
    pub fn new(b: &'a [u8]) -> Self { R { b, pos: 0 } }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    pub fn done(&self) -> bool { self.pos == self.b.len() }
    pub fn u8(&mut self) -> Option<u8> { Some(self.take(1)?[0]) }
    pub fn u16(&mut self) -> Option<u16> { Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?)) }
    pub fn u32(&mut self) -> Option<u32> { Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?)) }
    pub fn len(&mut self) -> Option<usize> { Some(self.u32()? as usize) }
    fn canon(x: u64) -> Option<u64> { (x < almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME).then_some(x) }
    pub fn u64s(&mut self) -> Option<Vec<u64>> {
        let n = self.len()?;
        let s = self.take(n.checked_mul(8)?)?;
        s.chunks(8).map(|c| Self::canon(u64::from_le_bytes(c.try_into().unwrap()))).collect()
    }
    // Bound counts against bytes before creating collection iterators.
    fn count(&mut self, max: usize, min_bytes: usize) -> Option<usize> {
        let n = self.len()?;
        if n > max || n.checked_mul(min_bytes)? > self.b.len().checked_sub(self.pos)? { return None; }
        Some(n)
    }
    fn exact_len(&mut self, want: usize) -> Option<()> { (self.len()? == want).then_some(()) }
    fn commitment(&mut self, p: &Profile) -> Option<Vec<Vec<u64>>> {
        self.exact_len(p.rank)?;
        let row_bytes = N.checked_mul(8)?.checked_add(4)?;
        if p.rank.checked_mul(row_bytes)? > self.b.len().checked_sub(self.pos)? { return None; }
        (0..p.rank).map(|_| {
            self.exact_len(N)?;
            self.take(N * 8)?.chunks_exact(8).map(|c| Self::canon(u64::from_le_bytes(c.try_into().ok()?))).collect()
        }).collect()
    }
    fn bytes_exact(&mut self, n: usize) -> Option<Vec<u8>> {
        self.exact_len(n)?; Some(self.take(n)?.to_vec())
    }
    pub fn k8(&mut self) -> Option<K8> {
        let s = self.take(64)?;
        let mut c = [0u64; 8];
        for (i, x) in c.iter_mut().enumerate() { *x = Self::canon(u64::from_le_bytes(s[8 * i..8 * i + 8].try_into().unwrap()))?; }
        Some(K8(c))
    }
    pub fn k8s(&mut self) -> Option<Vec<K8>> { let n = self.count(usize::MAX, 64)?; (0..n).map(|_| self.k8()).collect() }
    fn k8s_exact(&mut self, n: usize) -> Option<Vec<K8>> {
        self.exact_len(n)?;
        if n.checked_mul(64)? > self.b.len().checked_sub(self.pos)? { return None; }
        (0..n).map(|_| self.k8()).collect()
    }
    pub fn a(&mut self) -> Option<AElem> { Some(AElem(self.k8s_exact(DS)?)) }
    pub fn bytes(&mut self) -> Option<Vec<u8>> { Some(self.blob()?.to_vec()) }
    fn blob(&mut self) -> Option<&'a [u8]> { let n = self.len()?; self.take(n) }
}

// Public verifier inputs must use a registered, locally validated Profile.
// These validators also protect callers passing in-memory proof structures.
pub(crate) fn canonical_k(k: &K8) -> bool {
    k.0.iter().all(|&v| v < almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME)
}
pub(crate) fn endpoint_ok(a: &AElem) -> bool {
    a.0.len() == DS && a.0.iter().all(canonical_k)
}
pub(crate) fn commitment_ok(p: &Profile, c: &[Vec<u64>]) -> bool {
    c.len() == p.rank && c.iter().all(|r| r.len() == N && r.iter().all(|&v| v < almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME))
}
pub(crate) fn link_shape_ok(p: &Profile, n: usize, l: &LinkProof) -> bool {
    n <= p.n_max && l.g_commit.len() == n && l.a_hat.len() == n
        && l.g_commit.iter().all(|c| commitment_ok(p, c))
        && commitment_ok(p, &l.c_o) && l.a_hat.iter().all(endpoint_ok) && endpoint_ok(&l.t_o)
        && l.rounds.len() == p.ell() + 1
        && l.rounds.iter().zip(p.degs()).all(|(r, d)| r.len() == d + 1 && r.iter().all(canonical_k))
}
fn response_bytes(p: &Profile, cols: usize) -> Option<usize> {
    let n = p.tau.checked_mul(cols)?.checked_mul(N)?;
    if n % 8 != 0 { return None; }
    (n / 8).checked_mul(Z_BITS)
}
pub(crate) fn auth_shape_ok(p: &Profile, a: &AuthProof) -> bool {
    (a.attempt as usize) < p.attempts && a.d.len() == p.tau && a.d_o.len() == p.tau
        && a.d.iter().all(|t| commitment_ok(p, &t.c) && commitment_ok(p, &t.g) && endpoint_ok(&t.a))
        && a.d_o.iter().all(|t| commitment_ok(p, &t.c) && endpoint_ok(&t.t))
        && response_bytes(p, p.source_cols()) == Some(a.z.w.len())
        && response_bytes(p, p.link_cols()) == Some(a.z.l.len())
        && response_bytes(p, p.aux_cols()) == Some(a.z.o.len())
}

// ------------------------------------------------------------ proof parts

pub fn put_link(w: &mut W, l: &LinkProof) {
    w.len(l.g_commit.len());
    for g in &l.g_commit { w.ring_vec(g); }
    w.ring_vec(&l.c_o);
    w.len(l.rounds.len());
    for r in &l.rounds { w.k8s(r); }
    w.len(l.a_hat.len());
    for a in &l.a_hat { w.a(a); }
    w.a(&l.t_o);
}
pub fn get_link(p: &Profile, r: &mut R) -> Option<LinkProof> {
    let min_commit = p.rank.checked_mul(N.checked_mul(8)?.checked_add(4)?)?.checked_add(4)?;
    let n = r.count(p.n_max, min_commit)?;
    let g_commit = (0..n).map(|_| r.commitment(p)).collect::<Option<_>>()?;
    let c_o = r.commitment(p)?;
    r.exact_len(p.ell() + 1)?;
    let rounds = p.degs().into_iter().map(|d| r.k8s_exact(d + 1)).collect::<Option<_>>()?;
    r.exact_len(n)?;
    let a_hat = (0..n).map(|_| r.a()).collect::<Option<_>>()?;
    let t_o = r.a()?;
    Some(LinkProof { g_commit, c_o, rounds, a_hat, t_o })
}

fn put_first(w: &mut W, a: &AuthProof) {
    w.u16(a.attempt);
    w.len(a.d.len());
    for t in &a.d { w.ring_vec(&t.c); w.ring_vec(&t.g); w.a(&t.a); }
    w.len(a.d_o.len());
    for t in &a.d_o { w.ring_vec(&t.c); w.a(&t.t); }
}
/// Authentication proof; the response coefficient counts are fixed by the
/// profile and each response is sent through the response codec.
pub fn put_auth(p: &Profile, w: &mut W, a: &AuthProof) {
    put_first(w, a);
    for z in [&a.z.w, &a.z.l, &a.z.o] {
        let c = zenc_chunks(z, z_off(p));
        w.len(c.iter().map(Vec::len).sum());
        for c in c { w.b.extend_from_slice(&c); }
    }
}
pub fn get_auth(p: &Profile, r: &mut R) -> Option<AuthProof> { get_auth_cols(p, r, p.source_cols()) }
fn get_auth_cols(p: &Profile, r: &mut R, source_cols: usize) -> Option<AuthProof> {
    let attempt = r.u16()?;
    if attempt as usize >= p.attempts { return None; }
    r.exact_len(p.tau)?;
    let d = (0..p.tau).map(|_| Some(Tuple { c: r.commitment(p)?, g: r.commitment(p)?, a: r.a()? })).collect::<Option<Vec<_>>>()?;
    r.exact_len(p.tau)?;
    let d_o = (0..p.tau).map(|_| Some(AuxTuple { c: r.commitment(p)?, t: r.a()? })).collect::<Option<Vec<_>>>()?;
    let mut resp = |cols: usize| zdec(r.blob()?, p.tau.checked_mul(cols)?.checked_mul(N)?, z_off(p), z_umax(p));
    let z = Responses { w: resp(source_cols)?, l: resp(p.link_cols())?, o: resp(p.aux_cols())? };
    Some(AuthProof { attempt, d, d_o, z })
}

fn put_opt_auth(p: &Profile, w: &mut W, a: &Option<AuthProof>) {
    match a { Some(a) => { w.u8(1); put_auth(p, w, a); } None => w.u8(0) }
}
fn get_opt_auth(p: &Profile, r: &mut R) -> Option<Option<AuthProof>> {
    match r.u8()? { 0 => Some(None), 1 => Some(Some(get_auth(p, r)?)), _ => None }
}

/// The transmitted part of a request proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestProof {
    pub salt: [u8; 64],
    pub piop: Vec<K8>,
    pub c_src: Vec<Vec<Vec<u64>>>,
    pub link: LinkProof,
    pub auth: Option<AuthProof>,
}

pub fn encode_request(p: &Profile, pf: &RequestProof) -> Vec<u8> {
    let mut w = W::default();
    w.b.extend_from_slice(MAGIC);
    w.bytes(p.name.as_bytes());
    w.b.extend_from_slice(&pf.salt);
    w.k8s(&pf.piop);
    w.len(pf.c_src.len());
    for c in &pf.c_src { w.ring_vec(c); }
    put_link(&mut w, &pf.link);
    put_opt_auth(p, &mut w, &pf.auth);
    w.b
}

pub fn decode_request(p: &Profile, b: &[u8]) -> Option<RequestProof> {
    let mut r = R::new(b);
    if !accepts_wire_magic(r.take(8)?) || r.bytes_exact(p.name.len())? != p.name.as_bytes() { return None; }
    let salt: [u8; 64] = r.take(64)?.try_into().ok()?;
    let piop = r.k8s()?;
    let min_commit = p.rank.checked_mul(N.checked_mul(8)?.checked_add(4)?)?.checked_add(4)?;
    let n = r.count(p.n_max, min_commit)?;
    let c_src = (0..n).map(|_| r.commitment(p)).collect::<Option<_>>()?;
    let link = get_link(p, &mut r)?;
    let auth = get_opt_auth(p, &mut r)?;
    (r.done() && link.g_commit.len() == n).then_some(RequestProof { salt, piop, c_src, link, auth })
}

/// Encoded bytes by part: (header+salt, PIOP messages, source commitments,
/// link round messages, link commitments and endpoints, authentication
/// first messages, responses).
pub fn parts(p: &Profile, pf: &RequestProof) -> [(&'static str, usize); 7] {
    let total = encode_request(p, pf).len();
    let mut w = W::default(); w.k8s(&pf.piop); let piop = w.b.len();
    let mut w = W::default(); w.len(pf.c_src.len()); for c in &pf.c_src { w.ring_vec(c); } let csrc = w.b.len();
    let mut w = W::default(); w.len(pf.link.rounds.len()); for r in &pf.link.rounds { w.k8s(r); } let rounds = w.b.len();
    let mut w = W::default(); put_link(&mut w, &pf.link); let link = w.b.len() - rounds;
    let first = pf.auth.as_ref().map_or(0, |a| { let mut w = W::default(); put_first(&mut w, a); w.b.len() });
    // Magic, name, salt and the auth presence byte; the coded responses are the rest.
    let head = MAGIC.len() + 4 + p.name.len() + 64 + 1;
    let resp = total - head - piop - csrc - rounds - link - first;
    [("header", head), ("piop", piop), ("source commitments", csrc), ("link rounds", rounds), ("link commitments", link), ("auth first messages", first), ("responses", resp)]
}

/// The transmitted part of a finalization proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalProof { pub salt: [u8; 64], pub link: LinkProof, pub auth: Option<AuthProof> }

pub fn encode_final(p: &Profile, pf: &FinalProof) -> Vec<u8> {
    let mut w = W::default();
    w.b.extend_from_slice(MAGIC);
    w.bytes(p.name.as_bytes());
    w.b.extend_from_slice(&pf.salt);
    put_link(&mut w, &pf.link);
    put_opt_auth(p, &mut w, &pf.auth);
    w.b
}

pub fn decode_final(p: &Profile, b: &[u8]) -> Option<FinalProof> {
    let mut r = R::new(b);
    if !accepts_wire_magic(r.take(8)?) || r.bytes_exact(p.name.len())? != p.name.as_bytes() { return None; }
    let salt: [u8; 64] = r.take(64)?.try_into().ok()?;
    let link = get_link(p, &mut r)?;
    let auth = get_opt_auth(p, &mut r)?;
    r.done().then_some(FinalProof { salt, link, auth })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn typed_reader_rejects_malformed_shapes_before_field_reads() {
        for n in [0usize, DS - 1, DS + 1, u32::MAX as usize] {
            let mut w = W::default(); w.len(n);
            assert!(R::new(&w.b).a().is_none());
        }
        let mut w = W::default(); w.a(&AElem(vec![K8::ZERO; DS]));
        assert!(R::new(&w.b).a().is_some());
        let p = Profile::toy();
        for n in [0usize, p.rank - 1, p.rank + 1, u32::MAX as usize] {
            let mut w = W::default(); w.len(n);
            assert!(R::new(&w.b).commitment(&p).is_none());
        }
        let mut w = W::default(); w.len(u32::MAX as usize);
        assert!(R::new(&w.b).k8s().is_none());
        let mut w = W::default(); w.u8(2);
        assert!(get_opt_auth(&p, &mut R::new(&w.b)).is_none());
    }

    #[test]
    fn in_memory_shapes_are_checked_not_just_decoded_bytes() {
        let p = Profile::toy();
        for n in [0usize, DS - 1, DS, DS + 1] {
            assert_eq!(endpoint_ok(&AElem(vec![K8::ZERO; n])), n == DS);
        }
        for n in [0usize, N - 1, N, N + 1] {
            assert_eq!(commitment_ok(&p, &vec![vec![0; n]; p.rank]), n == N);
        }
        let mut a = AElem(vec![K8::ZERO; DS]);
        a.0[0].0[0] = almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;
        assert!(!endpoint_ok(&a));
    }

    #[test]
    fn pack37_round_trips_the_full_support_and_rejects_out_of_range_codes() {
        let p = Profile::ternary();
        assert_eq!(64 - (2 * p.b_z - 2).leading_zeros() as usize, Z_BITS);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(5);
        let bz = p.b_z as i64;
        let mut z: Vec<i64> = (0..4096).map(|_| rng.gen_range(-(bz - 1)..bz)).collect();
        z[..4].copy_from_slice(&[-(bz - 1), bz - 1, 0, -1]);
        let b = pack37(&z, z_off(&p));
        assert_eq!(b.len(), 4096 * 37 / 8);
        assert_eq!(unpack37(&b, z.len(), z_off(&p), z_umax(&p)).unwrap(), z);
        // Code 2·B_Z − 1 (Z = B_Z) is outside the support.
        let mut bad = z.clone();
        bad[7] = bz;
        assert!(unpack37(&pack37(&bad, z_off(&p)), z.len(), z_off(&p), z_umax(&p)).is_none());
        assert!(unpack37(&b[1..], z.len(), z_off(&p), z_umax(&p)).is_none());
    }

    /// Centered rounded Gaussian (Box-Muller) clipped to |Z| < B_Z.
    fn gauss(n: usize, seed: u64, bz: i64) -> Vec<i64> {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let s = Profile::ternary().sigma as f64;
        (0..n).map(|_| {
            let (a, b): (f64, f64) = (rng.gen::<f64>().max(1e-300), rng.gen());
            ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos() * s).round().clamp(-(bz - 1) as f64, (bz - 1) as f64) as i64
        }).collect()
    }

    #[test]
    fn response_codec_round_trips_and_compresses() {
        let p = Profile::ternary();
        let (off, umax, bz) = (z_off(&p), z_umax(&p), p.b_z as i64);
        // Several chunks, a partial last chunk; edge values in both.
        let n = 3 * ZC_CHUNK + 8 * 1001;
        let mut z = gauss(n, 7, bz);
        for (i, v) in [-(bz - 1), bz - 1, 0, -1, 1, -(1 << 30), (1 << 30) - 1].into_iter().enumerate() { z[i] = v; z[n - 1 - i] = v; }
        let b = pack37(&z, off);
        let e = zenc(&b, off);
        assert_eq!(zdec(&e, n, off, umax).unwrap(), b);
        let bits = e.len() as f64 * 8.0 / n as f64;
        eprintln!("response codec: {bits:.3} bits/coefficient (raw {Z_BITS})");
        assert!(bits < 33.7, "{bits}");
        // All-extreme and all-zero blocks also round-trip.
        for v in [-(bz - 1), bz - 1, 0] {
            let b = pack37(&vec![v; 4096], off);
            assert_eq!(zdec(&zenc(&b, off), 4096, off, umax).unwrap(), b);
        }
        assert!(zdec(&[], 0, off, umax).unwrap().is_empty());
    }

    #[test]
    fn response_codec_rejects_malformed_streams() {
        let p = Profile::ternary();
        let (off, umax, bz) = (z_off(&p), z_umax(&p), p.b_z as i64);
        let n = ZC_CHUNK + 4096;
        let z = gauss(n, 9, bz);
        let e = zenc(&pack37(&z, off), off);
        assert!(zdec(&e, n, off, umax).is_some());
        // Truncation, trailing bytes, wrong counts.
        for k in [1, 4, 8, e.len() / 2, e.len()] { assert!(zdec(&e[..e.len() - k], n, off, umax).is_none()); }
        let mut t = e.clone(); t.push(0);
        assert!(zdec(&t, n, off, umax).is_none());
        for m in [n - 8, n + 8, n + 1, 0] { assert!(zdec(&e, m, off, umax).is_none()); }
        // A rANS length that moves the chunk boundary.
        for d in [-4i64, 4, 1] {
            let mut t = e.clone();
            let k = u32::from_le_bytes(t[..4].try_into().unwrap()) as i64 + d;
            t[..4].copy_from_slice(&(k as u32).to_le_bytes());
            assert!(zdec(&t, n, off, umax).is_none());
        }
        // A corrupted rANS state or word must not panic (it may decode to other values).
        for i in [4, 11, 12, 20, 100] { let mut t = e.clone(); t[i] ^= 0x5a; let _ = zdec(&t, n, off, umax); }
        // Out-of-range values: Z = ±B_Z and any 37-bit code above umax encode, but never decode.
        for v in [bz, -bz, (1i64 << 37) - 1 - off] {
            let mut bad = z.clone(); bad[ZC_CHUNK + 5] = v;
            let b = pack37(&bad, off);
            assert!(zdec(&zenc(&b, off), n, off, umax).is_none());
        }
        // Flipping a low bit changes one coefficient and still decodes.
        let mut t = e.clone(); let l = t.len() - 1; t[l] ^= 1;
        let d = unpack37(&zdec(&t, n, off, umax).unwrap(), n, off, umax).unwrap();
        assert_eq!(d.iter().zip(&z).filter(|(a, b)| a != b).count(), 1);
    }

    #[test]
    #[ignore]
    fn response_codec_throughput() {
        let p = Profile::ternary();
        let (off, umax) = (z_off(&p), z_umax(&p));
        let n = 50 << 20;
        let b = pack37(&gauss(n, 11, p.b_z as i64), off);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let c = zenc_chunks(&b, off);
            eprintln!("encode chunks only: {:.3}s", t.elapsed().as_secs_f64());
            drop(c);
            let t = std::time::Instant::now();
            let e = zenc(&b, off);
            let te = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let d = zdec(&e, n, off, umax).unwrap();
            let td = t.elapsed().as_secs_f64();
            assert_eq!(d, b);
            eprintln!("n={n} threads={} bits/coef={:.4} encode {:.1} Mcoef/s ({te:.3}s) decode {:.1} Mcoef/s ({td:.3}s)",
                rayon::current_num_threads(), e.len() as f64 * 8.0 / n as f64, n as f64 / te / 1e6, n as f64 / td / 1e6);
        }
    }
}

/// Transport only: this discriminator is never a semantic transcript item.
pub const COMPACT_MAGIC:&[u8;8]=b"MCOLv001";
fn valid_codes(p:&Profile,bytes:&[u8])->bool{
    // One physical column at a time, bounded temporary decode storage.
    use rayon::prelude::*;let n=packed_len(N);bytes.len()%n==0&&bytes.par_chunks(n).all(|b|unpack37(b,N,z_off(p),z_umax(p)).is_some())
}
pub(crate) fn auth_shape_ok_cols(p:&Profile,a:&AuthProof,cols:usize)->bool{
    (a.attempt as usize)<p.attempts&&a.d.len()==p.tau&&a.d_o.len()==p.tau
    &&a.d.iter().all(|t|commitment_ok(p,&t.c)&&commitment_ok(p,&t.g)&&endpoint_ok(&t.a))
    &&a.d_o.iter().all(|t|commitment_ok(p,&t.c)&&endpoint_ok(&t.t))
    &&response_bytes(p,cols)==Some(a.z.w.len())&&response_bytes(p,p.link_cols())==Some(a.z.l.len())&&response_bytes(p,p.aux_cols())==Some(a.z.o.len())
}
fn compact_auth_ok(p:&Profile,s:&super::auth_compact::Support,a:&Option<AuthProof>)->bool{
    s.matches(p)&&a.as_ref().map(|a|auth_shape_ok_cols(p,a,s.len())&&[&a.z.w,&a.z.l,&a.z.o].iter().all(|b|valid_codes(p,b))).unwrap_or(false)
}
fn get_compact_auth(p:&Profile,s:&super::auth_compact::Support,r:&mut R)->Option<Option<AuthProof>>{
    // Compact transport is defined for a complete authenticated opening.
    if r.u8()?!=1{return None}let a=get_auth_cols(p,r,s.len())?;
    if ![&a.z.w,&a.z.l,&a.z.o].iter().all(|b|valid_codes(p,b)){return None}Some(Some(a))
}
pub fn encode_request_compact(p:&Profile,s:&super::auth_compact::Support,pf:&RequestProof)->Option<Vec<u8>>{
    if !compact_auth_ok(p,s,&pf.auth){return None}let mut w=W::default();w.b.extend_from_slice(COMPACT_MAGIC);w.b.extend_from_slice(MAGIC);w.bytes(p.name.as_bytes());w.b.extend_from_slice(&pf.salt);w.k8s(&pf.piop);w.len(pf.c_src.len());for c in &pf.c_src{w.ring_vec(c)}put_link(&mut w,&pf.link);put_opt_auth(p,&mut w,&pf.auth);Some(w.b)
}
pub fn decode_request_compact(p:&Profile,s:&super::auth_compact::Support,b:&[u8])->Option<RequestProof>{
    if !s.matches(p){return None}let mut r=R::new(b);if r.take(8)?!=COMPACT_MAGIC||!accepts_wire_magic(r.take(8)?)||r.bytes_exact(p.name.len())?!=p.name.as_bytes(){return None}
    let salt=r.take(64)?.try_into().ok()?;let piop=r.k8s()?;let min_commit=p.rank.checked_mul(N.checked_mul(8)?.checked_add(4)?)?.checked_add(4)?;let n=r.count(p.n_max,min_commit)?;
    let c_src=(0..n).map(|_|r.commitment(p)).collect::<Option<_>>()?;let link=get_link(p,&mut r)?;let auth=get_compact_auth(p,s,&mut r)?;
    (r.done()&&link.g_commit.len()==n).then_some(RequestProof{salt,piop,c_src,link,auth})
}
pub fn encode_final_compact(p:&Profile,s:&super::auth_compact::Support,pf:&FinalProof)->Option<Vec<u8>>{
    if !compact_auth_ok(p,s,&pf.auth){return None}let mut w=W::default();w.b.extend_from_slice(COMPACT_MAGIC);w.b.extend_from_slice(MAGIC);w.bytes(p.name.as_bytes());w.b.extend_from_slice(&pf.salt);put_link(&mut w,&pf.link);put_opt_auth(p,&mut w,&pf.auth);Some(w.b)
}
pub fn decode_final_compact(p:&Profile,s:&super::auth_compact::Support,b:&[u8])->Option<FinalProof>{
    if !s.matches(p){return None}let mut r=R::new(b);if r.take(8)?!=COMPACT_MAGIC||!accepts_wire_magic(r.take(8)?)||r.bytes_exact(p.name.len())?!=p.name.as_bytes(){return None}
    let salt=r.take(64)?.try_into().ok()?;let link=get_link(p,&mut r)?;let auth=get_compact_auth(p,s,&mut r)?;r.done().then_some(FinalProof{salt,link,auth})
}
/// Stream canonical legacy W-response bytes, including offset-binary omitted zeros.
/// Only for semantic continuation/legacy comparison, not compact wire framing.
pub fn visit_expanded_w(p:&Profile,s:&super::auth_compact::Support,bytes:&[u8],mut visit:impl FnMut(&[u8]))->Option<()>{
    if !s.matches(p)||response_bytes(p,s.len())?!=bytes.len()||!valid_codes(p,bytes){return None}
    let n=packed_len(N);let zero=pack37(&vec![0;N],z_off(p));
    for row in 0..p.tau{let mut k=0;for c in 0..p.source_cols(){if s.columns().get(k)==Some(&c){let off=(row*s.len()+k)*n;visit(&bytes[off..off+n]);k+=1}else{visit(&zero)}}}Some(())
}
#[cfg(any(test,feature="cpu-replay"))] pub mod compact_tests{
use super::*;use super::super::auth_compact::Support;
fn fixture(p:&Profile,s:&Support)->RequestProof{
 let c=vec![vec![0;N];p.rank];let a=AElem::zero();let z=Responses{w:pack37(&vec![0;p.tau*s.len()*N],z_off(p)),l:pack37(&vec![0;p.tau*p.link_cols()*N],z_off(p)),o:pack37(&vec![0;p.tau*p.aux_cols()*N],z_off(p))};
 RequestProof{salt:[0;64],piop:vec![K8::ZERO],c_src:vec![c.clone()],link:LinkProof{g_commit:vec![c.clone()],c_o:c.clone(),rounds:p.degs().iter().map(|&d|vec![K8::ZERO;d+1]).collect(),a_hat:vec![a.clone()],t_o:a.clone()},auth:Some(AuthProof{attempt:0,d:vec![Tuple{c:c.clone(),g:c.clone(),a:a.clone()};p.tau],d_o:vec![AuxTuple{c,t:a};p.tau],z})}
}
#[cfg_attr(test,test)] pub fn compact_framing_codes_and_virtual_zero_extension(){
 let p=Profile::toy();let s=Support::from_public_columns(&p,vec![0,2,3,4,5,6,7]).unwrap();assert!(Support::from_public_columns(&p,vec![0,2,3,4,5,6]).is_none());let mut pf=fixture(&p,&s);
 pf.salt[..8].copy_from_slice(COMPACT_MAGIC);pf.auth.as_mut().unwrap().z.w[..8].copy_from_slice(COMPACT_MAGIC);
 let bytes=encode_request_compact(&p,&s,&pf).unwrap();assert_eq!(decode_request_compact(&p,&s,&bytes),Some(pf.clone()));assert!(decode_request(&p,&bytes).is_none());assert!(decode_request(&p,&bytes[8..]).is_none());assert!(decode_request_compact(&p,&s,&bytes[8..]).is_none());
 let az=&pf.auth.as_ref().unwrap().z;let enc=|z:&[u8]|zenc(z,z_off(&p)).len();let wlen=bytes.len()-enc(&az.w)-enc(&az.l)-enc(&az.o)-12;let mut wronglen=bytes.clone();wronglen[wlen..wlen+4].copy_from_slice(&(enc(&az.w) as u32+1).to_le_bytes());assert!(decode_request_compact(&p,&s,&wronglen).is_none());
 let mut bad=bytes.clone();bad.push(0);assert!(decode_request_compact(&p,&s,&bad).is_none());let mut bad=bytes.clone();bad[8]=b'!';assert!(decode_request_compact(&p,&s,&bad).is_none());assert!(decode_request_compact(&p,&s,&bytes[..bytes.len()-1]).is_none());
 let mut badpf=pf.clone();badpf.auth.as_mut().unwrap().z.w[..37].fill(255);assert!(encode_request_compact(&p,&s,&badpf).is_none());let mut invalid=COMPACT_MAGIC.to_vec();invalid.extend(encode_request(&p,&badpf));assert!(decode_request_compact(&p,&s,&invalid).is_none());
 let mut dense=vec![];visit_expanded_w(&p,&s,&pf.auth.as_ref().unwrap().z.w,|b|dense.extend_from_slice(b)).unwrap();let z=rows_of(&p,&dense,p.source_cols()).unwrap();assert!(z.iter().all(|r|r[1].iter().all(|&v|v==0)));assert_eq!(&dense[packed_len(N)..packed_len(2*N)],pack37(&vec![0;N],z_off(&p)));assert_ne!(&dense[packed_len(N)..packed_len(2*N)],vec![0;packed_len(N)]);
 let compact=pf.auth.as_ref().unwrap();let mut legacy=pf.clone();legacy.auth.as_mut().unwrap().z.w=dense;let encoded=encode_request(&p,&legacy);assert_eq!(decode_request(&p,&encoded),Some(legacy));assert_eq!(compact.z.w.len(),response_bytes(&p,s.len()).unwrap());
 let finalpf=FinalProof{salt:pf.salt,link:pf.link.clone(),auth:pf.auth.clone()};let fb=encode_final_compact(&p,&s,&finalpf).unwrap();assert_eq!(decode_final_compact(&p,&s,&fb),Some(finalpf));assert!(decode_final(&p,&fb).is_none());
}
}
