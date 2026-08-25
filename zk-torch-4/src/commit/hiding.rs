//! Randomized (hiding) Ajtai commitment.
//!
//! `C = A_msg·x + A_hid·s` where `s` is fresh ternary randomness sampled per
//! commitment. `A_hid` is an independent matrix, obtained by committing the
//! hiding block against a **domain-separated seed**, so the two matrices are
//! independent even though both come from the same ChaCha8 construction.
//!
//! Binding is unaffected: Module-SIS applies to the concatenated `[A_msg|A_hid]`
//! at bound `2·B_x`. Hiding is statistical by the leftover-hash bound, sized in
//! [`super::layout::hiding_block_coeffs`].
//!
//! ## Where the hiding block lives in the evaluation domain
//!
//! The write-up's construction zeroes the public evaluation vector on the
//! hiding coordinates, so `Ev(W,r) = T̃(r)` exactly. That keeps evaluation
//! semantics unchanged, but it also means the terminal values `a_e` handed to
//! the masked RLC are *true* evaluations of the witness — not simulatable, so
//! HVZK cannot be proved for them.
//!
//! [`HidingMode`] makes this a choice. Under [`HidingMode::InEvaluation`] the
//! hiding block is simply another aligned block of the packed domain, so a
//! random `ξ` blends it in and `a_e = T̃(ξ) + <s, ψ_ξ>` is blinded by the
//! commitment's own randomness. Everything stays linear, so both masked-RLC
//! checks are unchanged; the DAG's own claims sit at Boolean-prefixed points
//! that never touch the hiding block; and the range polynomial passes natively
//! because `R_2(u) = (u+1)u(u-1)` vanishes on `{-1,0,1}`.
//!
//! Each opening leaks one linear equation in `s`, so a security statement must
//! bound openings per commitment. With deferred weight opening that bound is
//! (total requests / flush size), far below the block's symbol count.

use almost_goldilocks_cuda::ajtai::{self, RingCommitment, Seed, KAPPA, RING_DIM};

use crate::transcript::Transcript;

/// Modulus `q` for Almost-Goldilocks, as `u128` for host-side ring arithmetic.
const Q: u128 = ((1u128 << 64) - (1u128 << 32) + 1) - 32;

/// Whether the hiding block contributes to polynomial evaluations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HidingMode {
    /// Evaluation vector is zero on the hiding coordinates. `Ev(W,r) = T̃(r)`
    /// exactly; terminal values are unblinded.
    ZeroExtended,
    /// Hiding block is part of the evaluation domain, so terminal values at a
    /// random point are blinded by `s`.
    InEvaluation,
}

/// Public parameters for the randomized commitment.
#[derive(Clone, Copy, Debug)]
pub struct HidingKey {
    /// Seed for `A_msg` — the same seed the non-hiding commit path uses.
    pub msg_seed: Seed,
    /// Seed for `A_hid`, derived from `msg_seed` under a fixed domain tag.
    pub hid_seed: Seed,
    /// Hiding block length in coefficients.
    pub block_coeffs: usize,
    /// Ternary symbols carried per ring coordinate (see
    /// [`super::layout::HIDING_SYMBOLS_PER_COORD`]).
    pub symbols_per_coord: usize,
    pub mode: HidingMode,
}

impl HidingKey {
    /// Derive `A_hid`'s seed from the message seed with domain separation.
    ///
    /// Uses the protocol's own sponge so the derivation is deterministic and
    /// reproducible by the verifier without pulling in another hash.
    pub fn derive(msg_seed: Seed, mode: HidingMode) -> Self {
        let mut t = Transcript::new(b"zk4-ajtai-hiding-matrix");
        for w in msg_seed.0.iter() {
            t.append_u64(b"seed", *w as u64);
        }
        t.append_u64(b"kappa", KAPPA as u64);
        t.append_u64(b"ring_dim", RING_DIM as u64);
        let words = t.challenge_vector(b"A_hid", 8);
        let mut out = [0u32; 8];
        for (i, w) in words.iter().enumerate() {
            // Fold the field element to 32 bits; the sponge output is already
            // uniform, and A_hid only needs a PRG key.
            out[i] = (w.0 ^ (w.0 >> 32)) as u32;
        }
        Self {
            msg_seed,
            hid_seed: Seed(out),
            block_coeffs: super::layout::hiding_block_coeffs(),
            symbols_per_coord: super::layout::HIDING_SYMBOLS_PER_COORD,
            mode,
        }
    }

    /// Number of ring coordinates in the hiding block.
    pub fn coords(&self) -> usize {
        self.block_coeffs / RING_DIM
    }
}

/// Per-commitment hiding randomness. This is opening state: it must be retained
/// alongside the witness for the life of the commitment, and persisted with
/// offline weight commitments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HidingRandomness {
    /// Ternary values in `{-1, 0, 1}`, one per used slot, length
    /// `coords * symbols_per_coord`. Stored as `i8` rather than field elements
    /// so the norm bound is visible in the type.
    pub symbols: Vec<i8>,
}

impl HidingRandomness {
    /// Sample fresh randomness for one commitment.
    ///
    /// `rng_words` supplies uniform 64-bit words; the caller owns the RNG so
    /// this stays deterministic under test and unpredictable in production.
    pub fn sample<F: FnMut() -> u64>(key: &HidingKey, mut rng_words: F) -> Self {
        let total = key.coords() * key.symbols_per_coord;
        let mut symbols = Vec::with_capacity(total);
        // Rejection-free ternary: take 2 bits, map {0,1,2}->{-1,0,1}, redraw on 3.
        let mut bits: u64 = 0;
        let mut have = 0u32;
        while symbols.len() < total {
            if have < 2 {
                bits = rng_words();
                have = 64;
            }
            let two = (bits & 3) as i8;
            bits >>= 2;
            have -= 2;
            if two < 3 {
                symbols.push(two - 1);
            }
        }
        Self { symbols }
    }

    /// `||s||_inf`. Must be 1 (or 0 for an all-zero draw) for the norm budget.
    pub fn norm_inf(&self) -> u8 {
        self.symbols.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
    }

    /// Expand to canonical field elements laid out over the hiding block's
    /// coefficient slots, ready for [`ajtai::commit_wide`].
    ///
    /// With `symbols_per_coord == RING_DIM` every slot is used; with `1` only
    /// the constant term of each ring coordinate is, and the rest are zero
    /// (the conservative layout).
    pub fn to_field_slots(&self, key: &HidingKey) -> Vec<u64> {
        let mut out = vec![0u64; key.block_coeffs];
        let spc = key.symbols_per_coord;
        for (i, v) in self.symbols.iter().enumerate() {
            let coord = i / spc;
            let slot = i % spc;
            let idx = coord * RING_DIM + slot;
            out[idx] = match v {
                1 => 1u64,
                -1 => (Q - 1) as u64,
                _ => 0u64,
            };
        }
        out
    }
}

/// Ring-add two commitments (the Ajtai map is linear, so this composes blocks).
pub fn ring_add(a: &RingCommitment, b: &RingCommitment) -> RingCommitment {
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            out.rows[i][r] = ((a.rows[i][r] as u128 + b.rows[i][r] as u128) % Q) as u64;
        }
    }
    out
}

/// Reduce every coefficient into `[0, q)` so two congruent commitments compare
/// equal. The kernels return canonical values; host-side sums need this.
pub fn canonicalize(c: &RingCommitment) -> RingCommitment {
    let mut out = RingCommitment::zero();
    for i in 0..KAPPA {
        for r in 0..RING_DIM {
            out.rows[i][r] = ((c.rows[i][r] as u128) % Q) as u64;
        }
    }
    out
}

/// `A_hid · s` — the hiding term of the commitment.
///
/// Committed at `col_offset` so the hiding block occupies its own column window,
/// matching its position in the packed layout.
pub fn commit_hiding_term(
    key: &HidingKey,
    s: &HidingRandomness,
    col_offset: u64,
) -> Result<RingCommitment, String> {
    let slots = s.to_field_slots(key);
    ajtai::commit_wide(key.hid_seed, &slots, col_offset, None)
        .map_err(|e| format!("hiding commit failed: {:?}", e))
}

/// `C = A_msg·x + A_hid·s` for a message already committed elsewhere.
///
/// Takes the message commitment rather than the witness so it composes with all
/// three message paths (sparse, binary planes, wide) without duplicating them.
pub fn randomize(
    message_commitment: &RingCommitment,
    key: &HidingKey,
    s: &HidingRandomness,
    hiding_col_offset: u64,
) -> Result<RingCommitment, String> {
    let h = commit_hiding_term(key, s, hiding_col_offset)?;
    Ok(ring_add(message_commitment, &h))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commit::layout::{hiding_entropy_bits, PackLayout, LeafKey, LeafSpec};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    fn key() -> HidingKey {
        HidingKey::derive(Seed([9, 8, 7, 6, 5, 4, 3, 2]), HidingMode::InEvaluation)
    }

    #[test]
    fn hiding_seed_is_domain_separated_and_deterministic() {
        let k1 = key();
        let k2 = key();
        assert_eq!(k1.hid_seed.0, k2.hid_seed.0, "derivation must be reproducible");
        assert_ne!(
            k1.hid_seed.0, k1.msg_seed.0,
            "A_hid must not equal A_msg or binding of the pair is not MSIS"
        );
        let other = HidingKey::derive(Seed([1, 1, 1, 1, 1, 1, 1, 1]), HidingMode::InEvaluation);
        assert_ne!(k1.hid_seed.0, other.hid_seed.0, "distinct seeds give distinct A_hid");
    }

    #[test]
    fn sampled_randomness_is_ternary_and_full_length() {
        let k = key();
        let mut rng = Rng(0xDEADBEEF12345678);
        let s = HidingRandomness::sample(&k, || rng.next());
        assert_eq!(s.symbols.len(), k.coords() * k.symbols_per_coord);
        assert!(s.norm_inf() <= 1, "||s||_inf must be 1 for the norm budget");
        assert!(s.symbols.iter().all(|v| (-1..=1).contains(v)));
        // all three values should actually occur
        for want in [-1i8, 0, 1] {
            assert!(s.symbols.contains(&want), "value {} never sampled", want);
        }
    }

    #[test]
    fn field_slots_encode_minus_one_as_q_minus_one() {
        let k = key();
        let s = HidingRandomness { symbols: {
            let mut v = vec![0i8; k.coords() * k.symbols_per_coord];
            v[0] = 1; v[1] = -1;
            v
        }};
        let slots = s.to_field_slots(&k);
        assert_eq!(slots[0], 1);
        assert_eq!(slots[1] as u128, Q - 1);
        assert_eq!(slots.len(), k.block_coeffs);
    }

    #[test]
    fn entropy_exceeds_the_leftover_hash_requirement() {
        let need = (KAPPA * RING_DIM) as f64 * 64.0 + 2.0 * 128.0;
        assert!(hiding_entropy_bits() > need);
    }

    #[test]
    fn hiding_block_fits_above_the_message_capacity() {
        let leaves: Vec<LeafSpec> = (0..64)
            .map(|e| LeafSpec { key: LeafKey { edge: e, plane: 0 }, arity: 14 })
            .collect();
        let layout = PackLayout::build(&leaves, 22).expect("layout");
        let k = key();
        assert_eq!(layout.hiding_coeffs, k.block_coeffs);
        assert_eq!(
            layout.hiding_offset() + layout.hiding_coeffs,
            1usize << layout.ambient_arity,
            "hiding block must occupy the top of the ambient domain"
        );
        // and it must not overlap any message block
        for (_, p) in &layout.placements {
            assert!(p.offset + (1usize << p.arity) <= layout.hiding_offset());
        }
    }

    // ---- GPU tests ----

    #[test]
    fn randomized_commitment_hides_and_stays_additive() {
        let k = key();
        let mut rng = Rng(0x243F6A8885A308D3);
        let arity = 14usize;
        let msg: Vec<u64> = (0..(1usize << arity)).map(|_| rng.next() & 1).collect();

        let c_msg = ajtai::commit_wide(k.msg_seed, &msg, 0, None).expect("msg");
        let hid_off = ((1usize << arity) / RING_DIM) as u64;

        let s1 = HidingRandomness::sample(&k, || rng.next());
        let s2 = HidingRandomness::sample(&k, || rng.next());
        assert_ne!(s1, s2, "fresh randomness per commitment");

        let c1 = randomize(&c_msg, &k, &s1, hid_off).expect("c1");
        let c2 = randomize(&c_msg, &k, &s2, hid_off).expect("c2");

        // Same message, different randomness => different commitment.
        assert_ne!(
            canonicalize(&c1).rows[0], canonicalize(&c2).rows[0],
            "commitment must depend on the hiding randomness"
        );

        // Removing the hiding term recovers the message commitment exactly,
        // which is what keeps the opening's linear checks intact.
        let h1 = commit_hiding_term(&k, &s1, hid_off).expect("h1");
        let mut neg = RingCommitment::zero();
        for i in 0..KAPPA {
            for r in 0..RING_DIM {
                neg.rows[i][r] = ((Q - h1.rows[i][r] as u128) % Q) as u64;
            }
        }
        let recovered = ring_add(&c1, &neg);
        assert_eq!(
            canonicalize(&recovered).rows, canonicalize(&c_msg).rows,
            "C - A_hid*s must equal A_msg*x"
        );
    }

    #[test]
    fn hiding_term_is_linear_in_the_randomness() {
        let k = key();
        let mut rng = Rng(0x13198A2E03707344);
        let a = HidingRandomness::sample(&k, || rng.next());
        let b = HidingRandomness::sample(&k, || rng.next());

        // s_sum = a + b, taken over the integers; entries land in [-2, 2] so it
        // is not itself a legal hiding draw, but linearity of L must still hold.
        let sum_slots: Vec<u64> = a
            .to_field_slots(&k)
            .iter()
            .zip(b.to_field_slots(&k).iter())
            .map(|(x, y)| ((*x as u128 + *y as u128) % Q) as u64)
            .collect();

        let ca = commit_hiding_term(&k, &a, 0).expect("a");
        let cb = commit_hiding_term(&k, &b, 0).expect("b");
        let csum = ajtai::commit_wide(k.hid_seed, &sum_slots, 0, None).expect("sum");

        assert_eq!(
            canonicalize(&csum).rows,
            canonicalize(&ring_add(&ca, &cb)).rows,
            "A_hid must be linear in s"
        );
    }
}
