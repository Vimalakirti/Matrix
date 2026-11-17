//! Candidate SHAKE256 honest typed-prefix interface, protocol-incompatible with v1.
//! This implements the TO1 *codec*, not an instantiation theorem for SHAKE.
//! No full quantum-oracle simulation, history erasure, or statistical PRG claim.
use super::field8::K8;
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use sha3::digest::{ExtendableOutput, Update, XofReader};
/// Item types (absorbed before each item).
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum Item {
    Profile = 1,
    Salt = 2,
    Statement = 3,
    SourceCommitment = 4,
    LinkCommitment = 5,
    AuxCommitment = 6,
    RoundPoly = 7,
    Endpoint = 8,
    FirstMessage = 9,
    GraphClaim = 10,
    Layout = 11,
    Other = 99,
}

/// Challenge result types.
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum ChalType { K = 1, KNot01 = 2, C0 = 3, KBundle = 4, C0Tuple = 5 }

/// Protocol stages (part of every challenge address).
#[repr(u64)]
#[derive(Clone, Copy, Debug)]
pub enum Stage {
    GraphMask = 1,
    GraphReduction = 2,
    GraphMerge = 3,
    LinkRange = 4,
    LinkCombine = 5,
    LinkRound = 6,
    AuthColumn = 7,
    LinkInit = 8,
}


pub const RAW_SLACK: usize = 8192;
/// (queries, nanoseconds) over the process, for timing reports.
pub static QUERY_STATS: (std::sync::atomic::AtomicU64, std::sync::atomic::AtomicU64) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
pub const FS_DOMAIN: &[u8] = b"MORPHEUS/FS/SHAKE256/v2\0";
const MAX_RANK_BITS: u64 = 1 << 30;
// All u64-length byte strings: no smaller arbitrary protocol restriction.
const MAX_ADDRESS_BYTES: u64 = u64::MAX;

/// Literal, injectively framed prefix. It is deliberately not a hash digest.
#[derive(Clone)]
pub struct MTranscript {
    prefix: Vec<u8>,
    /// SHAKE256 state after absorbing prefix[..absorbed]. Absorbing the
    /// prefix incrementally and cloning the state per query gives exactly
    /// SHAKE256(prefix) without rehashing the whole prefix every time.
    sponge: sha3::Shake256,
    absorbed: usize,
    /// Incremental state of `registered_cardinality` over prefix[..parsed].
    parse: ParseState,
    /// Bytes of the address held by the transcript this one was forked from
    /// (`fork`): `prefix` holds address bytes base.. only; `absorbed` and
    /// `parse.parsed` stay absolute.
    base: usize,
}

/// `registered_cardinality`'s parser, resumable over appended frames.
#[derive(Clone)]
struct ParseState { parsed: usize, frame_number: usize, pending: Option<Card>, failed: bool }

/// A pending query cardinality: q^8 (the K challenge, by far the most frequent,
/// kept symbolic so no big integer is built per query) or any other value.
#[derive(Clone)]
enum Card { Q8, Big(BigUint) }
impl Card {
    /// rank >= cardinality for a minimal big-endian rank encoding.
    fn le_rank(&self, bytes: &[u8]) -> bool {
        match self {
            Card::Q8 => { let n = &krank().q8_be; if bytes.len() != n.len() { bytes.len() > n.len() } else { bytes >= &n[..] } }
            Card::Big(n) => BigUint::from_bytes_be(bytes) >= *n,
        }
    }
    fn value(&self) -> BigUint { match self { Card::Q8 => krank().q8.clone(), Card::Big(n) => n.clone() } }
    fn is_q8(&self) -> bool { match self { Card::Q8 => true, Card::Big(n) => *n == krank().q8 } }
}

impl ParseState {
    fn new() -> Self { ParseState { parsed: 0, frame_number: 0, pending: None, failed: false } }
    /// Consume the complete frames of address[self.parsed..] (same checks as
    /// `registered_cardinality::parse`, frame by frame).
    fn advance(&mut self, address: &[u8]) { self.advance_at(address, 0) }
    /// Same over address bytes base.. (`tail`).
    fn advance_at(&mut self, tail: &[u8], base: usize) {
        if self.failed { return; }
        if self.parsed == 0 {
            assert_eq!(base, 0, "parser state of a forked transcript");
            let address = tail;
            if address.len() as u64 > MAX_ADDRESS_BYTES || !address.starts_with(FS_DOMAIN) { self.failed = true; return; }
            self.parsed = FS_DOMAIN.len();
        }
        while self.parsed < base + tail.len() {
            let rest = &tail[self.parsed - base..];
            match parse_frame(rest, self.frame_number, &mut self.pending) {
                Some(used) => { self.parsed += used; self.frame_number += 1; }
                None => { self.failed = true; return; }
            }
        }
    }
    fn cardinality(&self) -> BigUint {
        if self.failed || self.frame_number < 3 { return BigUint::one(); }
        self.pending.as_ref().map(Card::value).unwrap_or_else(BigUint::one)
    }
    /// cardinality() == q^8, without building it.
    fn is_q8(&self) -> bool {
        !self.failed && self.frame_number >= 3 && self.pending.as_ref().map_or(false, Card::is_q8)
    }
}

/// One frame of the registered-domain parser: bytes used, or None if malformed.
fn parse_frame(rest: &[u8], frame_number: usize, pending: &mut Option<Card>) -> Option<usize> {
    if rest.len()<17 { return None; }
    let enc=rest[0];
    let item=u64::from_be_bytes(rest[1..9].try_into().ok()?);
    let len=usize::try_from(u64::from_be_bytes(rest[9..17].try_into().ok()?)).ok()?;
    let end=17usize.checked_add(len)?;
    let bytes=rest.get(17..end)?;
    if frame_number==0 {
        if enc!=2 || item!=Item::Profile as u64 { return None; }
    } else if frame_number==1 {
        if enc!=2 || item!=Item::Salt as u64 || len!=64 { return None; }
    } else if let Some(n)=pending.take() {
        if enc!=5 || item!=0 || bytes.is_empty() || (bytes.len()>1 && bytes[0]==0) || n.le_rank(bytes) { return None; }
    } else {
        match enc {
            1|2|3|6 => {
                if !matches!(item,1..=11|99) || ((enc==1 || enc==6) && len%8!=0) || (enc==3 && len%64!=0) { return None; }
                if matches!(enc,1|3) && bytes.chunks_exact(8).any(|b|u64::from_be_bytes(b.try_into().unwrap())>=Q) { return None; }
            },
            4 => {
                if item!=0 || len!=40 { return None; }
                let mut fields=[0u64;5];
                for (w,b) in fields.iter_mut().zip(bytes.chunks_exact(8)) { *w=u64::from_be_bytes(b.try_into().ok()?); }
                let [ty,stage,_attempt,index,arity]=fields;
                if matches!(ty,4|5) && index!=0 { return None; }
                if !(1..=8).contains(&stage) { return None; }
                let n=match ty {
                    1 if arity==1=>{ *pending=Some(Card::Q8); return Some(end); },
                    2 if arity==1=>&krank().q8-2u32,
                    4=> {
                        let exp=arity.checked_mul(8)?;
                        if exp.checked_mul(64)?>MAX_RANK_BITS { return None; }
                        BigUint::from(Q).pow(u32::try_from(exp).ok()?)
                    },
                    3|5 if ty==5 || arity==1=> {
                        let bits=arity.checked_mul(128)?;
                        if bits>MAX_RANK_BITS { return None; }
                        BigUint::one()<<usize::try_from(bits).ok()?
                    },
                    _=>return None
                };
                *pending=Some(Card::Big(n));
            },
            _=>return None
        }
    }
    Some(end)
}

/// `typed_rank` from a sponge that has absorbed the address.
fn typed_rank_from(h: sha3::Shake256, cardinality: &BigUint) -> BigUint {
    assert!(!cardinality.is_zero());
    let bits = (cardinality - BigUint::one()).bits();
    assert!(bits <= MAX_RANK_BITS, "unregistered typed width");
    let ell = usize::try_from(bits).unwrap() + RAW_SLACK;
    let mut bytes = vec![0u8; ell.div_ceil(8)];
    h.finalize_xof().read(&mut bytes);
    let unused = (8 - ell % 8) % 8;
    (BigUint::from_bytes_be(&bytes) >> unused) % cardinality
}

fn frame(dst: &mut Vec<u8>, encoding: u8, item: u64, bytes: &[u8]) {
    let total=dst.len().checked_add(17).and_then(|n|n.checked_add(bytes.len())).expect("address length overflow");
    assert!(u64::try_from(total).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
    dst.push(encoding);
    dst.extend_from_slice(&item.to_be_bytes());
    dst.extend_from_slice(&u64::try_from(bytes.len()).expect("frame length").to_be_bytes());
    dst.extend_from_slice(bytes);
}

/// TO1: first ell MSB-first bits, interpreted as an unsigned integer, modulo N.
/// The output length is derived from the registered domain, never caller-selected.
fn typed_rank(address: &[u8], cardinality: &BigUint) -> BigUint {
    assert!(!cardinality.is_zero());
    let bits = (cardinality - BigUint::one()).bits();
    assert!(bits <= MAX_RANK_BITS, "unregistered typed width");
    let ell = usize::try_from(bits).unwrap() + RAW_SLACK;
    let mut h = sha3::Shake256::default();
    h.update(address);
    let mut bytes = vec![0u8; ell.div_ceil(8)];
    h.finalize_xof().read(&mut bytes);
    let unused = (8 - ell % 8) % 8;
    (BigUint::from_bytes_be(&bytes) >> unused) % cardinality
}

/// Total registered-domain parser. Malformed byte strings still identify distinct
/// SHAKE inputs; only their typed decoding defaults to a singleton. This parser
/// does not validate the protocol statement: that remains the verifier's job.
pub fn registered_cardinality(address: &[u8]) -> BigUint {
    fn parse(address: &[u8]) -> Option<BigUint> {
        if address.len() as u64 > MAX_ADDRESS_BYTES { return None; }
        let mut rest=address.strip_prefix(FS_DOMAIN)?;
        let mut frame_number=0usize;
        let mut pending: Option<BigUint>=None;
        while !rest.is_empty() {
            if rest.len()<17 { return None; }
            let enc=rest[0];
            let item=u64::from_be_bytes(rest[1..9].try_into().ok()?);
            let len=usize::try_from(u64::from_be_bytes(rest[9..17].try_into().ok()?)).ok()?;
            let end=17usize.checked_add(len)?;
            let bytes=rest.get(17..end)?;
            rest=rest.get(end..)?;
            if frame_number==0 {
                if enc!=2 || item!=Item::Profile as u64 { return None; }
            } else if frame_number==1 {
                if enc!=2 || item!=Item::Salt as u64 || len!=64 { return None; }
            } else if let Some(n)=pending.take() {
                if enc!=5 || item!=0 || bytes.is_empty() || (bytes.len()>1 && bytes[0]==0) || BigUint::from_bytes_be(bytes)>=n { return None; }
            } else {
                match enc {
                    1|2|3|6 => {
                        if !matches!(item,1..=11|99) || ((enc==1 || enc==6) && len%8!=0) || (enc==3 && len%64!=0) { return None; }
                        if matches!(enc,1|3) && bytes.chunks_exact(8).any(|b|u64::from_be_bytes(b.try_into().unwrap())>=Q) { return None; }
                    },
                    4 => {
                        if item!=0 || len!=40 { return None; }
                        let mut fields=[0u64;5];
                        for (w,b) in fields.iter_mut().zip(bytes.chunks_exact(8)) { *w=u64::from_be_bytes(b.try_into().ok()?); }
                        let [ty,stage,_attempt,index,arity]=fields;
                        if matches!(ty,4|5) && index!=0 { return None; }
                        if !(1..=8).contains(&stage) { return None; }
                        let n=match ty {
                            1 if arity==1=>BigUint::from(Q).pow(8),
                            2 if arity==1=>BigUint::from(Q).pow(8)-2u32,
                            4=> {
                                let exp=arity.checked_mul(8)?;
                                if exp.checked_mul(64)?>MAX_RANK_BITS { return None; }
                                BigUint::from(Q).pow(u32::try_from(exp).ok()?)
                            },
                            3|5 if ty==5 || arity==1=> {
                                let bits=arity.checked_mul(128)?;
                                if bits>MAX_RANK_BITS { return None; }
                                BigUint::one()<<usize::try_from(bits).ok()?
                            },
                            _=>return None
                        };
                        pending=Some(n);
                    },
                    _=>return None
                }
            }
            frame_number+=1;
        }
        if frame_number<3 { return None; }
        pending
    }
    parse(address).unwrap_or_else(BigUint::one)
}

/// Fixed-width `typed_rank` for K challenges (cardinality Q^8): the raw
/// sample X (ell = bits(Q^8 − 1) + RAW_SLACK bits, a whole number of bytes)
/// is reduced as Σ_j w_j·(2^{64j} mod Q^8) with each table entry in base-Q
/// digits, then carried in base Q; the low 8 base-Q digits of X are the K8
/// limbs of X mod Q^8 (digits beyond Q^8 are dropped). Same result as
/// `typed_rank_from` + `unrank_k` without big-integer division.
struct KRank { q8: BigUint, q8_be: Vec<u8>, nbytes: usize, t: Vec<[u64; 8]> }
fn krank() -> &'static KRank {
    static K: std::sync::OnceLock<KRank> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let q8 = BigUint::from(Q).pow(8);
        let ell = usize::try_from((&q8 - BigUint::one()).bits()).unwrap() + RAW_SLACK;
        assert_eq!(ell % 64, 0, "fast K rank needs whole words");
        let nbytes = ell / 8;
        let q = BigUint::from(Q);
        let t = (0..nbytes / 8).map(|j| {
            let mut v = (BigUint::one() << (64 * j)) % &q8;
            let mut d = [0u64; 8];
            for x in &mut d { *x = (&v % &q).to_u64().unwrap(); v /= &q; }
            d
        }).collect();
        let q8_be = q8.to_bytes_be();
        KRank { q8, q8_be, nbytes, t }
    })
}
/// Minimal big-endian bytes of Σ_k d[k]·q^k (BigUint::to_bytes_be), in
/// fixed-width limbs.
fn rank_be(d: &[u64; 8]) -> Vec<u8> {
    let mut acc = [0u64; 9];   // little-endian limbs
    for &x in d.iter().rev() {
        let mut carry = x as u128;
        for l in acc.iter_mut() { let v = (*l as u128) * (Q as u128) + carry; *l = v as u64; carry = v >> 64; }
        debug_assert_eq!(carry, 0);
    }
    let mut out: Vec<u8> = acc.iter().rev().flat_map(|l| l.to_be_bytes()).collect();
    let nz = out.iter().position(|&b| b != 0).unwrap_or(out.len() - 1);
    out.drain(..nz);
    out
}
fn k_from_sponge(h: sha3::Shake256) -> [u64; 8] {
    let kr = krank();
    let mut bytes = vec![0u8; kr.nbytes];
    h.finalize_xof().read(&mut bytes);
    let nw = kr.nbytes / 8;
    // acc[k] = Σ_j w_j T_j[k] as (lo: u128, hi: u64).
    let mut lo = [0u128; 8];
    let mut hi = [0u64; 8];
    for j in 0..nw {
        let w = u64::from_be_bytes(bytes[kr.nbytes - 8 * (j + 1)..kr.nbytes - 8 * j].try_into().unwrap()) as u128;
        if w == 0 { continue; }
        let tj = &kr.t[j];
        for k in 0..8 {
            let (s, c) = lo[k].overflowing_add(w * tj[k] as u128);
            lo[k] = s; hi[k] += c as u64;
        }
    }
    // Base-Q carry (a 3-limb carry: hi·2^128 + lo + carry < 2^192).
    let mut d = [0u64; 8];
    let mut carry = [0u64; 3];   // little-endian limbs
    for k in 0..8 {
        // v = (hi[k], lo[k]) + carry
        let l0 = lo[k] as u64 as u128 + carry[0] as u128;
        let l1 = (lo[k] >> 64) + carry[1] as u128 + (l0 >> 64);
        let l2 = hi[k] as u128 + carry[2] as u128 + (l1 >> 64);
        let limbs = [l2 as u64, l1 as u64, l0 as u64];
        let (mut rem, mut quo) = (0u128, [0u64; 3]);
        for (i, &x) in limbs.iter().enumerate() {
            let cur = (rem << 64) | x as u128;
            quo[2 - i] = (cur / Q as u128) as u64;
            rem = cur % Q as u128;
        }
        d[k] = rem as u64;
        carry = quo;
    }
    d
}

impl MTranscript {
    pub fn new(profile: &str, salt: &[u8; 64]) -> Self {
        let mut t = Self { prefix: FS_DOMAIN.to_vec(), sponge: sha3::Shake256::default(), absorbed: 0, parse: ParseState::new(), base: 0 };
        t.absorb_bytes(Item::Profile, profile.as_bytes());
        t.absorb_bytes(Item::Salt, salt);
        t
    }
    /// Complete public prefix for verifier-owned deferred finalization state.
    pub fn literal_prefix(&self) -> &[u8] { assert_eq!(self.base, 0, "literal prefix of a forked transcript"); &self.prefix }
    /// A transcript that continues this one exactly like a clone, without
    /// copying its (possibly GB-scale) prefix: the sponge and the cardinality
    /// parser are caught up and only bytes appended after the fork are kept.
    /// Hash every absorbed byte now (SHAKE absorbs a stream, so later
    /// queries are unchanged); lets a caller overlap hashing with other work.
    pub fn hash_pending(&mut self) {
        self.parse.advance_at(&self.prefix, self.base);
        self.sponge.update(&self.prefix[self.absorbed - self.base..]);
        self.absorbed = self.base + self.prefix.len();
    }
    /// `absorb_bytes` followed by `hash_pending`, without keeping the bytes:
    /// the transcript continues like a fork (no literal prefix), with the
    /// same sponge and parser state.
    pub fn absorb_bytes_hashed(&mut self, item: Item, bytes: &[u8]) {
        self.hash_pending();
        let mut head = Vec::with_capacity(17);
        head.push(2u8);
        head.extend_from_slice(&(item as u64).to_be_bytes());
        head.extend_from_slice(&u64::try_from(bytes.len()).expect("frame length").to_be_bytes());
        let end = self.absorbed.checked_add(17).and_then(|n| n.checked_add(bytes.len())).expect("address length overflow");
        assert!(u64::try_from(end).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
        // The parser's checks for an encoding-2 frame (header only).
        let p = &mut self.parse;
        if !p.failed {
            if p.frame_number < 2 || p.pending.is_some() || !matches!(item as u64, 1..=11 | 99) { p.failed = true; }
            else { p.parsed = end; p.frame_number += 1; }
        }
        self.sponge.update(&head);
        self.sponge.update(bytes);
        self.prefix.clear();
        self.absorbed = end;
        self.base = end;
    }
    /// Bytes absorbed but not hashed yet.
    pub fn pending_bytes(&self) -> usize { self.base + self.prefix.len() - self.absorbed }
    pub fn fork(&self) -> Self {
        let end = self.base + self.prefix.len();
        let mut sponge = self.sponge.clone();
        sponge.update(&self.prefix[self.absorbed - self.base..]);
        let mut parse = self.parse.clone();
        parse.advance_at(&self.prefix, self.base);
        MTranscript { prefix: Vec::new(), sponge, absorbed: end, parse, base: end }
    }
    /// Canonical field residues, except Statement/Layout which encode arbitrary
    /// public u64 metadata exactly (including q itself), with a separate tag.
    pub fn absorb(&mut self, item: Item, words: &[u64]) {
        let mut bytes = Vec::with_capacity(words.len().checked_mul(8).unwrap());
        let metadata=matches!(item,Item::Statement|Item::Layout);
        for &w in words { bytes.extend_from_slice(&(if metadata {w} else {w % Q}).to_be_bytes()); }
        frame(&mut self.prefix, if metadata {6} else {1}, item as u64, &bytes);
    }
    pub fn absorb_bytes(&mut self, item: Item, bytes: &[u8]) {
        frame(&mut self.prefix, 2, item as u64, bytes);
    }
    pub fn absorb_k(&mut self, item: Item, ks: &[K8]) {
        let words: Vec<u64> = ks.iter().flat_map(|k| k.0).collect();
        let mut bytes = Vec::with_capacity(words.len().checked_mul(8).unwrap());
        for w in words { bytes.extend_from_slice(&(w % Q).to_be_bytes()); }
        frame(&mut self.prefix, 3, item as u64, &bytes);
    }
    fn query(&mut self, ty: ChalType, stage: Stage, attempt: u64, index: u64, arity: usize) -> BigUint {
        let t_q = std::time::Instant::now();
        let r = self.query_inner(ty, stage, attempt, index, arity);
        QUERY_STATS.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        QUERY_STATS.1.fetch_add(t_q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        r
    }
    fn query_inner(&mut self, ty: ChalType, stage: Stage, attempt: u64, index: u64, arity: usize) -> BigUint {
        let arity64 = u64::try_from(arity).expect("arity overflow");
        let n = match ty {
            ChalType::K => { assert_eq!(arity, 1); BigUint::from(Q).pow(8) },
            ChalType::KNot01 => { assert_eq!(arity, 1); BigUint::from(Q).pow(8) - 2u32 },
            ChalType::KBundle => {
                let exp = arity.checked_mul(8).and_then(|v| u32::try_from(v).ok()).expect("arity overflow");
                assert!(u64::from(exp) * 64 <= MAX_RANK_BITS);
                BigUint::from(Q).pow(exp)
            },
            ChalType::C0 | ChalType::C0Tuple => {
                if matches!(ty, ChalType::C0) { assert_eq!(arity, 1); }
                let bits = arity.checked_mul(128).expect("arity overflow");
                assert!(u64::try_from(bits).unwrap() <= MAX_RANK_BITS);
                BigUint::one() << bits
            }
        };
        let mut a = Vec::with_capacity(40);
        for w in [ty as u64, stage as u64, attempt, index, arity64] { a.extend_from_slice(&w.to_be_bytes()); }
        frame(&mut self.prefix, 4, 0, &a);
        assert!(u64::try_from(self.base + self.prefix.len()).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
        self.parse.advance_at(&self.prefix, self.base);
        assert_eq!(self.parse.cardinality(), n, "builder/parser mismatch");
        self.sponge.update(&self.prefix[self.absorbed - self.base..]);
        self.absorbed = self.base + self.prefix.len();
        let r = typed_rank_from(self.sponge.clone(), &n);
        #[cfg(test)]
        if self.base == 0 {
            assert_eq!(registered_cardinality(&self.prefix), n, "incremental parser differs");
            assert_eq!(typed_rank(&self.prefix, &n), r, "incremental sponge differs");
        }
        // Subsequent challenges bind this decoded result and its exact address.
        // Unused raw bits are NEVER read into a future address or transcript.
        frame(&mut self.prefix, 5, 0, &r.to_bytes_be());
        r
    }
    pub fn challenge_k(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        let t_q = std::time::Instant::now();
        let k = self.query_k(stage, attempt, index);
        QUERY_STATS.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        QUERY_STATS.1.fetch_add(t_q.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        k
    }
    /// `query(ChalType::K, ..)` + `unrank_k` with the fixed-width rank.
    fn query_k(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        let mut a = Vec::with_capacity(40);
        for w in [ChalType::K as u64, stage as u64, attempt, index, 1] { a.extend_from_slice(&w.to_be_bytes()); }
        frame(&mut self.prefix, 4, 0, &a);
        assert!(u64::try_from(self.base + self.prefix.len()).unwrap() <= MAX_ADDRESS_BYTES, "address cap exceeded");
        self.parse.advance_at(&self.prefix, self.base);
        assert!(self.parse.is_q8(), "builder/parser mismatch");
        self.sponge.update(&self.prefix[self.absorbed - self.base..]);
        self.absorbed = self.base + self.prefix.len();
        let d = k_from_sponge(self.sponge.clone());
        let r_be = rank_be(&d);
        #[cfg(test)]
        if self.base == 0 {
            let n = krank().q8.clone();
            assert_eq!(registered_cardinality(&self.prefix), n, "incremental parser differs");
            let slow = typed_rank(&self.prefix, &n);
            assert_eq!(slow.to_bytes_be(), r_be, "fixed-width rank differs");
        }
        frame(&mut self.prefix, 5, 0, &r_be);
        K8(d)
    }
    pub fn challenge_k_not01(&mut self, stage: Stage, attempt: u64, index: u64) -> K8 {
        let r = self.query(ChalType::KNot01, stage, attempt, index, 1) + 2u32;
        unrank_k(r, 1).remove(0)
    }
    pub fn challenge_k_bundle(&mut self, stage: Stage, attempt: u64, k: usize) -> Vec<K8> {
        let r = self.query(ChalType::KBundle, stage, attempt, 0, k);
        unrank_k(r, k)
    }
    pub fn challenge_c0_tuple(&mut self, stage: Stage, attempt: u64, count: usize) -> Vec<[i8; 64]> {
        let r = self.query(ChalType::C0Tuple, stage, attempt, 0, count);
        // Base-4 digits, least significant first: digit j is bits 2j..2j+2
        // of the rank (read from its limbs; shifting the rank each digit is
        // quadratic in the count).
        let limbs = r.to_u64_digits();
        (0..count).map(|i| {
            let mut out = [0i8; 64];
            for (k, v) in out.iter_mut().enumerate() {
                let j = i * 64 + k;
                let d = limbs.get(j / 32).map_or(0, |w| (w >> (2 * (j % 32))) & 3);
                *v = [-2, -1, 1, 2][d as usize];
            }
            out
        }).collect()
    }
}
fn unrank_k(mut r: BigUint, count: usize) -> Vec<K8> {
    let q = BigUint::from(Q);
    (0..count).map(|_| {
        let mut k = [0u64; 8];
        for c in &mut k { *c = (&r % &q).to_u64().unwrap(); r /= &q; }
        K8(k)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fork_equals_clone() {
        let mut t = MTranscript::new("p", &[7u8; 64]);
        t.absorb(Item::FirstMessage, &[1, 2, 3]);
        let _ = t.challenge_k(Stage::LinkRound, 0, 0);
        t.absorb(Item::FirstMessage, &[4, 5]);
        for _ in 0..2 {
            let (mut a, mut b) = (t.clone(), t.fork());
            for x in [&mut a, &mut b] { x.absorb(Item::FirstMessage, &[9; 300]); x.absorb_k(Item::FirstMessage, &[K8([3; 8])]); }
            assert_eq!(a.challenge_k(Stage::AuthColumn, 1, 2), b.challenge_k(Stage::AuthColumn, 1, 2));
            assert_eq!(a.challenge_c0_tuple(Stage::AuthColumn, 1, 5), b.challenge_c0_tuple(Stage::AuthColumn, 1, 5));
            let (mut c, mut d) = (a.clone(), b.fork());
            c.absorb(Item::FirstMessage, &[1]); d.absorb(Item::FirstMessage, &[1]);
            assert_eq!(c.challenge_k_not01(Stage::LinkRound, 0, 3), d.challenge_k_not01(Stage::LinkRound, 0, 3));
            t.absorb(Item::FirstMessage, &[6]);
        }
    }
    #[test]
    fn hash_pending_is_transparent() {
        let mut a = MTranscript::new("p", &[3u8; 64]);
        let mut b = a.clone();
        for i in 0..5u64 {
            for t in [&mut a, &mut b] { t.absorb(Item::FirstMessage, &[i, i * 7, 9]); }
            b.hash_pending();
            if i % 2 == 1 { assert_eq!(a.challenge_k(Stage::LinkRound, 0, i), b.challenge_k(Stage::LinkRound, 0, i)); }
        }
        assert_eq!(a.challenge_k_bundle(Stage::LinkInit, 0, 3), b.challenge_k_bundle(Stage::LinkInit, 0, 3));
        assert_eq!(a.prefix, b.prefix);
    }
    #[test]
    fn absorb_bytes_hashed_is_transparent() {
        let mut a = MTranscript::new("p", &[4u8; 64]);
        let mut b = a.clone();
        let big: Vec<u8> = (0..5000u32).map(|i| (i * 31) as u8).collect();
        for i in 0..3u64 {
            for t in [&mut a, &mut b] { t.absorb(Item::Statement, &[i]); }
            a.absorb_bytes(Item::Other, &big[..1000 * i as usize]);
            b.absorb_bytes_hashed(Item::Other, &big[..1000 * i as usize]);
            for t in [&mut a, &mut b] { t.absorb_k(Item::Other, &[K8::ONE]); }
            assert_eq!(a.challenge_k(Stage::LinkRound, 0, i), b.challenge_k(Stage::LinkRound, 0, i));
        }
        assert_eq!(a.challenge_k_bundle(Stage::LinkInit, 0, 3), b.challenge_k_bundle(Stage::LinkInit, 0, 3));
        // A frame the parser rejects (pending query result) fails the same way.
        let (mut c, mut d) = (a.clone(), b.clone());
        c.parse.pending = Some(Card::Q8); d.parse.pending = Some(Card::Q8);
        c.absorb_bytes(Item::Other, &big[..10]); c.hash_pending(); c.parse.advance_at(&c.prefix, c.base);
        d.absorb_bytes_hashed(Item::Other, &big[..10]);
        assert!(c.parse.failed && d.parse.failed);
    }
    #[test]
    fn c0_tuple_digits_equal_shifting() {
        let mut t = MTranscript::new("p", &[5u8; 64]);
        t.absorb(Item::FirstMessage, &[1, 2, 3]);
        for count in [1usize, 3, 37] {
            let mut a = t.clone();
            let mut b = t.clone();
            let fast = a.challenge_c0_tuple(Stage::AuthColumn, 2, count);
            let mut r = b.query(ChalType::C0Tuple, Stage::AuthColumn, 2, 0, count);
            let slow: Vec<[i8; 64]> = (0..count).map(|_| {
                let mut out = [0i8; 64];
                for v in &mut out { *v = [-2, -1, 1, 2][(&r % 4u32).to_usize().unwrap()]; r >>= 2; }
                out
            }).collect();
            assert_eq!(fast, slow);
            assert_eq!(a.prefix, b.prefix);
        }
    }
    #[test]
    fn rank_be_equals_biguint() {
        let mut x = 1u64;
        let mut cases: Vec<[u64; 8]> = vec![[0; 8], [Q - 1; 8], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]];
        for _ in 0..2000 { cases.push(std::array::from_fn(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); x % Q })); }
        for d in cases {
            let r = d.iter().rev().fold(BigUint::zero(), |acc, &v| acc * Q + v);
            assert_eq!(rank_be(&d), r.to_bytes_be());
            // The parser's q^8 comparison equals the big-integer one.
            assert_eq!(Card::Q8.le_rank(&r.to_bytes_be()), r >= krank().q8);
        }
        for b in [krank().q8_be.clone(), (&krank().q8 - 1u32).to_bytes_be(), (&krank().q8 + 1u32).to_bytes_be(), vec![0]] {
            assert_eq!(Card::Q8.le_rank(&b), BigUint::from_bytes_be(&b) >= krank().q8);
        }
    }
    #[test]
    fn framed_encodings_cannot_alias() {
        let mut a = MTranscript::new("test", &[0;64]);
        let mut b = a.clone();
        a.absorb(Item::Other, &[1]); b.absorb_bytes(Item::Other, &1u64.to_be_bytes());
        assert_ne!(a.prefix, b.prefix);
        let mut c = a.clone(); let mut d = a.clone();
        c.query(ChalType::K, Stage::LinkRound, 0, 0, 1);
        d.query(ChalType::K, Stage::LinkRound, Q, 0, 1);
        assert_ne!(c.prefix, d.prefix); // Integer addresses are not field-reduced.
    }
    #[test]
    fn singleton_and_non_byte_aligned_codec() {
        assert_eq!(typed_rank(b"kat", &BigUint::one()), BigUint::zero());
        assert_eq!(typed_rank(b"kat", &BigUint::from(257u32)), BigUint::from(176u32));
    }
    #[test]
    fn shake_standard_empty_kat() {
        let mut h = sha3::Shake256::default(); h.update(b"");
        let mut out = [0;32]; h.finalize_xof().read(&mut out);
        assert_eq!(out, [0x46,0xb9,0xdd,0x2b,0x0b,0xa8,0x8d,0x13,0x23,0x3b,0x3f,0xeb,0x74,0x3e,0xeb,0x24,0x3f,0xcd,0x52,0xea,0x62,0xb8,0x1b,0x82,0xb5,0x0c,0x27,0x64,0x6e,0xd5,0x76,0x2f]);
    }
}

#[cfg(test)]
mod candidate_tests {
 use super::*;
 #[test] fn python_chronological_known_answers() {
   let mut t=MTranscript::new("test",&[0;64]);
   assert_eq!(t.challenge_k(Stage::LinkRound,0,0).0,[7797489697268215020, 10771494639500149469, 9585284757969161066, 8519815873561415581, 14459922377744084684, 5078317245461511668, 2955893371482872527, 4638795857135829957]);
   assert_eq!(t.challenge_k_not01(Stage::LinkRound,0,1).0,[12117270624391117241, 900558997749131788, 1659691970300626709, 12516491091674694104, 12860766057750762003, 15388120984586741696, 14910131183502859555, 18118523938667695142]);
 }
 #[test] fn branch_retry_is_addressed_without_rewriting_prior_queries() {
   let t=MTranscript::new("test",&[0;64]);
   let frozen=t.prefix.clone();
   let a=t.clone().challenge_c0_tuple(Stage::AuthColumn,0,2);
   let b=t.clone().challenge_c0_tuple(Stage::AuthColumn,1,2);
   assert_ne!(a,b); assert_eq!(t.prefix,frozen);
   assert_eq!(a,t.clone().challenge_c0_tuple(Stage::AuthColumn,0,2));
 }
 #[test] fn empty_bundles_and_canonical_field_words() {
   let mut t=MTranscript::new("test",&[0;64]);
   assert!(t.challenge_k_bundle(Stage::LinkInit,0,0).is_empty());
   assert!(t.challenge_c0_tuple(Stage::AuthColumn,0,0).is_empty());
   let mut a=MTranscript::new("test",&[0;64]);let mut b=a.clone();
   a.absorb(Item::Other,&[Q]);b.absorb(Item::Other,&[0]);
   assert_eq!(a.prefix,b.prefix);
 }
}

#[cfg(test)] mod parser_tests {
 use super::*;
 fn address(ty:u64, stage:u64, arity:u64)->Vec<u8> {
  let mut t=MTranscript::new("test",&[0;64]);let mut a=vec![];
  for w in [ty,stage,0,0,arity] { a.extend_from_slice(&w.to_be_bytes()); }
  frame(&mut t.prefix,4,0,&a);t.prefix
 }
 #[test] fn total_defaults_and_registered_domains() {
  assert_eq!(registered_cardinality(b"invalid"),BigUint::one());
  assert_eq!(registered_cardinality(&address(1,6,1)),BigUint::from(Q).pow(8));
  assert_eq!(registered_cardinality(&address(2,6,1)),BigUint::from(Q).pow(8)-2u32);
  assert_eq!(registered_cardinality(&address(5,7,3)),BigUint::one()<<384usize);
  for a in [address(1,6,2),address(99,6,1),address(4,6,u64::MAX),address(4,0,1)] {
   assert_eq!(registered_cardinality(&a),BigUint::one());
  }
 }
 #[test] fn malformed_history_cannot_select_nontrivial_domain() {
  let mut t=MTranscript::new("test",&[0;64]);
  frame(&mut t.prefix,5,0,&[0]); // unsolicited output
  let mut a=vec![];for w in [1u64,6,0,0,1] {a.extend_from_slice(&w.to_be_bytes());}
  frame(&mut t.prefix,4,0,&a);
  assert_eq!(registered_cardinality(&t.prefix),BigUint::one());
 }
}

#[cfg(test)] mod incremental_tests {
 use super::*;
 #[test] fn incremental_queries_equal_full_rehash() {
  // query() asserts equality with the full-prefix parse and hash under cfg(test).
  let mut t=MTranscript::new("inc",&[7;64]);
  for i in 0..50u64 {
   t.absorb(Item::Other,&(0..(i*37%300)).collect::<Vec<u64>>());
   if i%3==0 { t.absorb_bytes(Item::Other,&vec![i as u8; (i*1001%5000) as usize]); }
   if i%5==0 { t.absorb_k(Item::Other,&[K8([i;8])]); }
   let mut c=t.clone();
   let _=c.challenge_c0_tuple(Stage::AuthColumn,i,3);
   let _=t.challenge_k(Stage::LinkRound,0,i);
   let _=t.challenge_k_bundle(Stage::LinkInit,0,(i%4) as usize);
  }
 }
}

#[cfg(test)] mod metadata_tests {
 use super::*;
 #[test] fn literal_setup_modulus_is_not_zero() {
  let mut a=MTranscript::new("test",&[0;64]);let mut b=a.clone();
  a.absorb(Item::Statement,&[Q]);b.absorb(Item::Statement,&[0]);
  assert_ne!(a.literal_prefix(),b.literal_prefix());
  assert_ne!(a.challenge_k(Stage::LinkInit,0,0),b.challenge_k(Stage::LinkInit,0,0));
 }
}
