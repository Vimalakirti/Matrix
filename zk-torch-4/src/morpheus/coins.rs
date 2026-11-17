//! Private-coin backend (paper theory source `private_coins`, "A bounded
//! one-seed private-coin backend"). Every honest private draw has a
//! canonical address; its bits are the MSB-first expansion of the 512-bit
//! blocks
//!     B_S(a, j) = F(TAG || S || profile || purpose || owner || attempt ||
//!                   coordinate || outer || trial || stream || j),
//! a fixed 106-byte message. Refinement appends bits from the same address
//! and never restarts a stream, so a GPU first pass and a CPU fallback that
//! re-reads the same address see the same prefix.
//!
//! F is SHAKE-256 truncated to 64 bytes, the reference packet's test
//! instantiation. The packet states this is not a proved random-oracle
//! instantiation; swapping F changes only `block`.
//!
//! The packet's profile byte covers PRIMARY64 (0) and causal1024 (1). The
//! ternary candidate is registered here as profile 2 with the same field
//! layout. Address uniqueness is structural (addresses are built from loop
//! indices that never repeat). A bounded registry reserves opening IDs and
//! registration namespaces, not individual draws (10^8-10^9 per opening).

use sha3::digest::{ExtendableOutput, Update, XofReader};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

pub const TAG: &[u8; 17] = b"MORPHEUS/COIN/v1\0";
pub const MESSAGE_BYTES: usize = 106;
pub const OPENINGS: u64 = 1 << 20;
/// Explicit per-backend budget for registered private-input source suffixes.
pub const INPUT_SOURCES: u64 = 1 << 20;
pub const INPUT_OWNER_BASE: u64 = 1 << 62;
pub const ATTEMPTS: u16 = 512;
/// Gaussian outer proposals R and raw polar trials H.
pub const OUTER: u16 = 3072;
pub const TRIALS: u16 = 8192;
pub const POLAR_BITS: usize = 26172;
pub const COIN_BITS: usize = 8708;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CoinProfile { Primary64 = 0, Causal1024 = 1, Ternary = 2 }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Purpose {
    Salt = 1,
    SourceSecret = 2,
    SourceError = 3,
    LinkPayload = 4,
    LinkSecret = 5,
    LinkError = 6,
    AuxPayload = 7,
    AuxSecret = 8,
    AuxError = 9,
    PolarX = 10,
    PolarY = 11,
    Correction = 12,
    ProtocolCoin = 13,
    GaussianCompat = 14,
    /// Graph-mask bank W_G (not in the packet's table; registered here as
    /// 15 with the auxiliary-payload layout: opening owner, coordinate =
    /// slot·8 + residue, uniform-q rejection trials).
    GraphMask = 15,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Address {
    pub purpose: Purpose,
    pub owner: u64,
    pub attempt: u16,
    pub coordinate: u32,
    pub outer: u16,
    pub trial: u16,
    pub stream: u32,
}

impl Address {
    pub fn new(purpose: Purpose, owner: u64) -> Self {
        Address { purpose, owner, attempt: 0, coordinate: 0, outer: 0, trial: 0, stream: 0 }
    }
    /// Bit cap of the stream at this address.
    pub fn cap(&self) -> usize {
        match self.purpose {
            Purpose::Salt => 512,
            Purpose::LinkPayload | Purpose::AuxPayload | Purpose::GraphMask => 64,
            Purpose::PolarX | Purpose::PolarY | Purpose::GaussianCompat => POLAR_BITS,
            Purpose::Correction | Purpose::ProtocolCoin => COIN_BITS,
            // Hiding secrets/errors: six bits for PRIMARY64; the ternary
            // sampler reads two-bit proposals, up to 8192 of them.
            _ => 2 * super::sampler::TERNARY_PROPOSALS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoinsExhausted;

/// The fixed 106-byte message for block `j` of address `a`.
/// MORPHEUS_NO_ZK=1: a same-protocol ablation that removes zero knowledge
/// (never a paper configuration). Every hiding sample (ternary suffixes,
/// uniform masks, Gaussian masks) is zero and every attempt is accepted; the
/// proof keeps its shape and still verifies.
pub fn no_zk() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var_os("MORPHEUS_NO_ZK").is_some())
}

pub fn encode(profile: CoinProfile, seed: &[u8; 64], a: &Address, j: u8) -> [u8; MESSAGE_BYTES] {
    let mut m = [0u8; MESSAGE_BYTES];
    let mut o = 0;
    let mut put = |b: &[u8]| { m[o..o + b.len()].copy_from_slice(b); o += b.len(); };
    put(TAG);
    put(seed);
    put(&[profile as u8, a.purpose as u8]);
    put(&a.owner.to_be_bytes());
    put(&a.attempt.to_be_bytes());
    put(&a.coordinate.to_be_bytes());
    put(&a.outer.to_be_bytes());
    put(&a.trial.to_be_bytes());
    put(&a.stream.to_be_bytes());
    put(&[j]);
    debug_assert_eq!(o, MESSAGE_BYTES);
    m
}

/// F: one 512-bit block.
pub fn block(msg: &[u8; MESSAGE_BYTES]) -> [u8; 64] {
    let mut h = sha3::Shake256::default();
    h.update(msg);
    let mut out = [0u8; 64];
    h.finalize_xof().read(&mut out);
    out
}

/// An append-only bit stream.
pub trait BitStream {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted>;
}

/// Lifecycle failures are fail-closed and contain no private seed material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoinLifecycleError {
    DuplicateOpening,
    OpeningLimit,
    ModelAlreadyRegistered,
    InputSourceLimit,
    LifecyclePoisoned,
}

impl std::fmt::Display for CoinLifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::DuplicateOpening => "private-coin opening ID already consumed",
            Self::OpeningLimit => "private-coin opening lifetime cap exceeded",
            Self::ModelAlreadyRegistered => "private-coin model registration already consumed",
            Self::InputSourceLimit => "private-coin input source lifetime cap exceeded",
            Self::LifecyclePoisoned => "private-coin lifecycle lock poisoned",
        })
    }
}
impl std::error::Error for CoinLifecycleError {}

#[derive(Default)]
struct Lifecycle {
    openings: HashSet<u64>,
    model_registered: bool,
    input_sources: u64,
}

/// The private seed and profile; hands out address streams.
///
/// Clones share a single lifecycle. Independently constructing two backends
/// with the same seed is NOT safe: this in-process guard cannot detect seed
/// reuse across processes or independent constructors. Production callers
/// must use fresh OS entropy and retain this backend with registered sources.
///
/// The lifecycle guarantee applies to callers that reserve via the registration
/// and opening methods before drawing (the production request/register APIs).
/// Low-level `stream`, `block`, and `salt` deliberately permit deterministic
/// rereads for prefix refinement and GPU/CPU fallback; they are not standalone
/// opening/registration APIs and must not be used to bypass reservation.
#[derive(Clone)]
pub struct Coins {
    seed: [u8; 64],
    pub profile: CoinProfile,
    lifecycle: Arc<Mutex<Lifecycle>>,
}

impl Coins {
    /// Explicit seeds are for deterministic fixtures or externally managed
    /// unique seed lifetimes; never recreate a production seed to reset limits.
    pub fn new(seed: [u8; 64], profile: CoinProfile) -> Self {
        Coins { seed, profile, lifecycle: Arc::new(Mutex::new(Lifecycle::default())) }
    }
    /// Reserve an opening ID before sampling any opening-local coins. A
    /// failed proof still consumes its ID. There is deliberately no rollback.
    /// The mutex makes duplicate checking and the lifetime cap atomic across
    /// clones, while holding at most OPENINGS IDs rather than per-draw state.
    pub fn begin_opening(&self, opening: u64) -> Result<[u8; 64], CoinLifecycleError> {
        {
            let mut state = self.lifecycle.lock().map_err(|_| CoinLifecycleError::LifecyclePoisoned)?;
            if state.openings.contains(&opening) { return Err(CoinLifecycleError::DuplicateOpening); }
            if state.openings.len() as u64 >= OPENINGS { return Err(CoinLifecycleError::OpeningLimit); }
            state.openings.insert(opening);
        }
        Ok(self.salt(opening))
    }
    /// Model source owners are fixed, so permit exactly one registration per
    /// backend lifetime, even if that registration fails. Repeated proofs of
    /// the SAME saved raw sources remain valid; they do not call this method.
    pub fn claim_model_registration(&self) -> Result<(), CoinLifecycleError> {
        let mut state = self.lifecycle.lock().map_err(|_| CoinLifecycleError::LifecyclePoisoned)?;
        if state.model_registered { return Err(CoinLifecycleError::ModelAlreadyRegistered); }
        state.model_registered = true;
        Ok(())
    }
    /// Allocate distinct SourceSecret/SourceError owners for private-input
    /// registrations. Ranges are burned on issue, including failed commits.
    /// Empty registrations consume no budget because they draw no coins.
    ///
    /// Source-owner domains: request owners ((j+1)<<24)|e with j<2^32-1
    /// and e<2^24 lie below 2^56; input owners lie in [2^62,2^62+2^20);
    /// model owners (1<<63)|e with e<2^24 start at 2^63. Hence disjoint
    /// even though all three use the same source-secret/error purposes.
    pub fn allocate_input_registration(&self, n_sources: usize) -> Result<std::ops::Range<u64>, CoinLifecycleError> {
        let n = u64::try_from(n_sources).map_err(|_| CoinLifecycleError::InputSourceLimit)?;
        let mut state = self.lifecycle.lock().map_err(|_| CoinLifecycleError::LifecyclePoisoned)?;
        let end = state.input_sources.checked_add(n).filter(|&end| end <= INPUT_SOURCES)
            .ok_or(CoinLifecycleError::InputSourceLimit)?;
        let start = state.input_sources;
        state.input_sources = end;
        Ok((INPUT_OWNER_BASE + start)..(INPUT_OWNER_BASE + end))
    }
    /// A fresh seed from the OS.
    pub fn from_entropy(profile: CoinProfile) -> Self {
        use rand::RngCore;
        let mut seed = [0u8; 64];
        rand::rngs::OsRng.fill_bytes(&mut seed);
        Coins::new(seed, profile)
    }
    pub fn stream(&self, a: Address) -> Prefix {
        Prefix { coins: self.clone(), cap: a.cap(), address: a, bits: 0, idx: usize::MAX, blk: [0; 64] }
    }
    pub fn block(&self, a: &Address, j: u8) -> [u8; 64] { block(&encode(self.profile, &self.seed, a, j)) }
    /// Keccak rate lanes of the padded Gaussian coin message for (opening,
    /// attempt) with purpose, coordinate, outer and trial zeroed; the GPU
    /// sampler XORs those fields in per thread.
    pub fn gaussian_base_lanes(&self, opening: u64, attempt: u16) -> [u64; 17] {
        let mut a = Address::new(Purpose::PolarX, opening);
        a.attempt = attempt;
        let mut m = [0u8; 136];
        m[..MESSAGE_BYTES].copy_from_slice(&encode(self.profile, &self.seed, &a, 0));
        m[82] = 0;
        m[MESSAGE_BYTES] ^= 0x1f;
        m[135] ^= 0x80;
        let mut l = [0u64; 17];
        for i in 0..17 { l[i] = u64::from_le_bytes(m[8 * i..8 * i + 8].try_into().unwrap()); }
        l
    }
    /// Public salt for an opening (512 bits at its own index).
    pub fn salt(&self, opening: u64) -> [u8; 64] { self.block(&Address::new(Purpose::Salt, opening), 0) }
}

/// MSB-first prefix of one address stream.
pub struct Prefix {
    coins: Coins,
    pub address: Address,
    cap: usize,
    pub bits: usize,
    idx: usize,
    blk: [u8; 64],
}

impl BitStream for Prefix {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted> {
        if self.bits >= self.cap { return Err(CoinsExhausted); }
        let (j, off) = (self.bits / 512, self.bits % 512);
        if j != self.idx {
            self.blk = self.coins.block(&self.address, j as u8);
            self.idx = j;
        }
        self.bits += 1;
        Ok((self.blk[off / 8] >> (7 - off % 8)) & 1)
    }
}

/// Fixture streams for tests: stream i is SHAKE-256(label || i) read
/// MSB-first. Not a private-coin backend.
pub struct FixtureStream { reader: sha3::Shake256Reader, byte: u8, left: u32 }

impl FixtureStream {
    pub fn new(label: &[u8], i: u64) -> Self {
        let mut h = sha3::Shake256::default();
        h.update(label);
        h.update(&i.to_be_bytes());
        FixtureStream { reader: h.finalize_xof(), byte: 0, left: 0 }
    }
}

impl BitStream for FixtureStream {
    fn next_bit(&mut self) -> Result<u8, CoinsExhausted> {
        if self.left == 0 {
            let mut b = [0u8; 1];
            self.reader.read(&mut b);
            self.byte = b[0];
            self.left = 8;
        }
        self.left -= 1;
        Ok((self.byte >> self.left) & 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opening_ids_are_consumed_across_clones_and_failed_proofs() {
        let c = Coins::new([1; 64], CoinProfile::Ternary);
        let clone = c.clone();
        let salt = c.begin_opening(7).unwrap();
        assert_eq!(salt, c.salt(7)); // Still private-derived, no public nonce replacement.
        // Simulate failure after reservation: no completion/rollback is needed.
        assert_eq!(clone.begin_opening(7), Err(CoinLifecycleError::DuplicateOpening));
        assert_ne!(salt, clone.begin_opening(8).unwrap());
    }

    #[test]
    fn opening_limit_has_no_overflow_or_reset() {
        let c = Coins::new([2; 64], CoinProfile::Ternary);
        // Populate consumed IDs directly, avoiding a million SHAKE calls.
        c.lifecycle.lock().unwrap().openings.extend(0..OPENINGS - 1);
        c.begin_opening(OPENINGS - 1).unwrap();
        assert_eq!(c.begin_opening(OPENINGS), Err(CoinLifecycleError::OpeningLimit));
        assert_eq!(c.clone().begin_opening(u64::MAX), Err(CoinLifecycleError::OpeningLimit));
        assert_eq!(c.begin_opening(0), Err(CoinLifecycleError::DuplicateOpening));
        assert_eq!(c.lifecycle.lock().unwrap().openings.len() as u64, OPENINGS);
    }

    #[test]
    fn model_registration_once_but_saved_sources_support_multiple_proofs() {
        let c = Coins::new([3; 64], CoinProfile::Ternary);
        c.claim_model_registration().unwrap();
        assert_eq!(c.clone().claim_model_registration(), Err(CoinLifecycleError::ModelAlreadyRegistered));
        // Reading the same saved suffix is intentional, not a new registration.
        let a = Address::new(Purpose::SourceSecret, 1 << 63);
        assert_eq!(c.block(&a, 0), c.clone().block(&a, 0));
        c.begin_opening(1).unwrap();
        c.begin_opening(2).unwrap();
    }

    #[test]
    fn input_allocations_burn_ranges_and_are_bounded_and_disjoint() {
        let c = Coins::new([4; 64], CoinProfile::Ternary);
        assert_eq!(c.allocate_input_registration(0).unwrap(), INPUT_OWNER_BASE..INPUT_OWNER_BASE);
        let first = c.allocate_input_registration(2).unwrap();
        // Even if committing first fails, the clone gets a fresh range.
        let second = c.clone().allocate_input_registration(3).unwrap();
        assert_eq!(first, INPUT_OWNER_BASE..INPUT_OWNER_BASE + 2);
        assert_eq!(second, INPUT_OWNER_BASE + 2..INPUT_OWNER_BASE + 5);
        assert_ne!(c.block(&Address::new(Purpose::SourceSecret, first.start), 0),
                   c.block(&Address::new(Purpose::SourceSecret, second.start), 0));
        c.allocate_input_registration((INPUT_SOURCES - 5) as usize).unwrap();
        assert_eq!(c.allocate_input_registration(1), Err(CoinLifecycleError::InputSourceLimit));
        assert!(c.allocate_input_registration(0).unwrap().is_empty());
        assert_eq!(c.allocate_input_registration(usize::MAX), Err(CoinLifecycleError::InputSourceLimit));
        let max_request = (((1u64 << 32) - 1) << 24) | ((1u64 << 24) - 1);
        assert!(max_request < INPUT_OWNER_BASE);
        assert!(INPUT_OWNER_BASE + INPUT_SOURCES <= 1u64 << 63);
        // Widths in the fixed-width message are unchanged.
        let a = Address::new(Purpose::SourceSecret, INPUT_OWNER_BASE + INPUT_SOURCES - 1);
        assert_eq!(&encode(c.profile, &[4; 64], &a, 0)[83..91], &a.owner.to_be_bytes());
    }

    #[test]
    fn concurrent_clones_accept_exactly_one_duplicate_id_and_model_claim() {
        let c = Coins::new([5; 64], CoinProfile::Ternary);
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16).map(|_| {
            let c = c.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                (c.begin_opening(99).is_ok(), c.claim_model_registration().is_ok(),
                 c.allocate_input_registration(1).unwrap().start)
            })
        }).collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.0).count(), 1);
        assert_eq!(results.iter().filter(|r| r.1).count(), 1);
        let owners: HashSet<_> = results.iter().map(|r| r.2).collect();
        assert_eq!(owners.len(), 16);
        assert!(owners.iter().all(|&owner| (INPUT_OWNER_BASE..INPUT_OWNER_BASE + 16).contains(&owner)));
    }

    #[test]
    fn concurrent_input_allocations_cannot_cross_lifetime_cap() {
        let c = Coins::new([8; 64], CoinProfile::Ternary);
        c.allocate_input_registration((INPUT_SOURCES - 1) as usize).unwrap();
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let threads: Vec<_> = (0..16).map(|_| {
            let c = c.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                c.allocate_input_registration(1)
            })
        }).collect();
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|r| **r == Err(CoinLifecycleError::InputSourceLimit)).count(), 15);
        assert_eq!(c.lifecycle.lock().unwrap().input_sources, INPUT_SOURCES);
    }

    #[test]
    fn poisoned_lifecycle_is_fail_closed() {
        let c = Coins::new([6; 64], CoinProfile::Ternary);
        let clone = c.clone();
        assert!(std::thread::spawn(move || {
            let _guard = clone.lifecycle.lock().unwrap();
            panic!("simulate interrupted lifecycle mutation");
        }).join().is_err());
        assert_eq!(c.begin_opening(0), Err(CoinLifecycleError::LifecyclePoisoned));
        assert_eq!(c.claim_model_registration(), Err(CoinLifecycleError::LifecyclePoisoned));
        assert_eq!(c.allocate_input_registration(1), Err(CoinLifecycleError::LifecyclePoisoned));
    }

    #[test]
    fn codec_is_fixed_width_and_streams_are_prefix_stable() {
        let c = Coins::new([7u8; 64], CoinProfile::Ternary);
        let mut a = Address::new(Purpose::PolarX, 3);
        a.attempt = 1;
        a.coordinate = 99;
        a.outer = 2;
        a.trial = 5;
        let m = encode(c.profile, &[7u8; 64], &a, 1);
        assert_eq!(m.len(), 106);
        assert_eq!(&m[..17], TAG);
        assert_eq!(m[81], 2);
        assert_eq!(m[82], 10);
        // Bits across a block boundary, and a re-read of the same address.
        let mut s = c.stream(a);
        let v: Vec<u8> = (0..1100).map(|_| s.next_bit().unwrap()).collect();
        let mut s2 = c.stream(a);
        let w: Vec<u8> = (0..1100).map(|_| s2.next_bit().unwrap()).collect();
        assert_eq!(v, w);
        let b1 = c.block(&a, 1);
        for t in 0..512 { assert_eq!(v[512 + t], (b1[t / 8] >> (7 - t % 8)) & 1); }
        // Different trial, different stream.
        let mut b = a;
        b.trial = 6;
        let mut s3 = c.stream(b);
        let x: Vec<u8> = (0..256).map(|_| s3.next_bit().unwrap()).collect();
        assert_ne!(&v[..256], &x[..]);
        // Cap.
        let mut s4 = c.stream(Address::new(Purpose::LinkPayload, 0));
        for _ in 0..64 { s4.next_bit().unwrap(); }
        assert!(s4.next_bit().is_err());
    }
}
