//! Fiat-Shamir transcript over AlmostGoldilocks, backed by the Monolith
//! permutation from [`crate::crypto::monolith`].
//!
//! Structure mirrors zk-torch-3's `Transcript`:
//! - DuplexChallenger sponge with overwrite-mode absorption.
//! - Rate / capacity split matches Monolith's WIDTH=12 — RATE=8, CAPACITY=4
//!   (yielding ~4·log2(q) ≈ 252-bit security level, well above the 128 bits
//!   we target for the protocol).
//! - Public API mirrors zk-torch-3 verbatim (`append_*`, `challenge_*`,
//!   `fork`, `fingerprint`, `get_state`) so the rest of the prover/verifier
//!   port stays mechanical.

use almost_goldilocks_cuda::extension::AlmostGoldilocksExt2;
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::crypto::digest::Digest;
use crate::crypto::monolith::{monolith_permute, WIDTH};

pub const RATE: usize = 8;
pub const CAPACITY: usize = WIDTH - RATE; // = 4

/// Fiat-Shamir transcript.
#[derive(Clone, Debug)]
pub struct Transcript {
    sponge_state: [u64; WIDTH],
    input_buffer: [u64; RATE],
    output_buffer: [u64; RATE],
    input_count: usize,
    output_count: usize,
}

impl Transcript {
    /// Create a fresh transcript and absorb a domain-separation label
    /// (byte-for-byte, no length prefix). Matches zk-torch-3 behaviour so
    /// downstream ports don't need to think about label semantics here.
    pub fn new(label: &[u8]) -> Self {
        let mut t = Self {
            sponge_state: [0u64; WIDTH],
            input_buffer: [0u64; RATE],
            output_buffer: [0u64; RATE],
            input_count: 0,
            output_count: 0,
        };
        for &b in label {
            t.absorb_u64(b as u64);
        }
        t
    }

    // ------------------------------------------------------------------
    // Public API — mirrors zk-torch-3
    // ------------------------------------------------------------------

    /// Length-prefixed absorption of a sub-label used to namespace the
    /// subsequent append/squeeze call. Empty labels are skipped (matches
    /// zk-torch-3 / Basefold convention).
    fn absorb_label(&mut self, label: &[u8]) {
        if label.is_empty() {
            return;
        }
        self.absorb_u64(label.len() as u64);
        for &b in label {
            self.absorb_u64(b as u64);
        }
    }

    pub fn append_scalar(&mut self, label: &[u8], scalar: &AlmostGoldilocksField) {
        self.absorb_label(label);
        self.absorb_u64(scalar.reduce().0);
    }

    pub fn append_scalars(&mut self, label: &[u8], scalars: &[AlmostGoldilocksField]) {
        self.absorb_label(label);
        for s in scalars {
            self.absorb_u64(s.reduce().0);
        }
    }

    pub fn append_u64(&mut self, label: &[u8], value: u64) {
        self.absorb_label(label);
        // Reduce to canonical form so absorption is independent of the
        // caller's representation choice.
        self.absorb_u64(AlmostGoldilocksField(value).reduce().0);
    }

    /// Batched version of [`append_u64`] — absorbs the label once, then
    /// all values. Saves ≥ `(label_bytes + 1)·(N − 1)` sponge absorbs
    /// vs calling `append_u64` in a loop (which re-absorbs the label
    /// each iteration). Used by `absorb_group_commitments` in the
    /// fold-tree verifier where each ring commitment is 960 u64s with
    /// a fixed per-call label.
    pub fn append_u64_slice(&mut self, label: &[u8], values: &[u64]) {
        self.absorb_label(label);
        for &v in values {
            self.absorb_u64(AlmostGoldilocksField(v).reduce().0);
        }
    }

    pub fn append_ext2(&mut self, label: &[u8], val: &AlmostGoldilocksExt2) {
        self.absorb_label(label);
        self.absorb_u64(val.c0.reduce().0);
        self.absorb_u64(val.c1.reduce().0);
    }

    pub fn append_digest(&mut self, label: &[u8], digest: &Digest) {
        self.absorb_label(label);
        for e in &digest.elements {
            self.absorb_u64(e.reduce().0);
        }
    }

    pub fn challenge_scalar(&mut self, label: &[u8]) -> AlmostGoldilocksField {
        self.absorb_label(label);
        AlmostGoldilocksField(self.squeeze_u64())
    }

    pub fn challenge_ext2(&mut self, label: &[u8]) -> AlmostGoldilocksExt2 {
        self.absorb_label(label);
        let c0 = AlmostGoldilocksField(self.squeeze_u64());
        let c1 = AlmostGoldilocksField(self.squeeze_u64());
        AlmostGoldilocksExt2::new(c0, c1)
    }

    pub fn challenge_vector(&mut self, label: &[u8], len: usize) -> Vec<AlmostGoldilocksField> {
        self.absorb_label(label);
        (0..len)
            .map(|_| AlmostGoldilocksField(self.squeeze_u64()))
            .collect()
    }

    /// Cheap structural hash of internal state, for debug logging only. Not
    /// used in proof semantics.
    pub fn fingerprint(&self) -> u64 {
        let mut h = (self.input_count * 100 + self.output_count) as u64;
        for &v in &self.sponge_state {
            h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(v);
        }
        h
    }

    /// Snapshot the sponge state (12 u64s) for diagnostic inspection.
    pub fn get_state(&self) -> Vec<u64> {
        self.sponge_state.to_vec()
    }

    /// Domain-separated fork: clone this transcript and absorb a `family`
    /// label plus a `branch_id`, yielding an independent child transcript.
    ///
    /// Both arguments MUST be public functions of data already absorbed into
    /// this transcript, and the caller must absorb each branch's messages
    /// back into the parent in ascending `branch_id` order once every branch
    /// completes. Under those conditions a branch's challenges depend only on
    /// (parent state, family, branch_id), so a hybrid over branches gives a
    /// composed knowledge error equal to the sum of the per-branch errors.
    /// Violate either and the prover picks its branch structure *after*
    /// seeing the branch challenges.
    ///
    /// `family` exists to keep distinct fork families from colliding. Absorbing
    /// only the id (the previous behavior) meant that forking one parent state
    /// by partition index 3 and by edge id 3 produced the *same* child, so two
    /// unrelated branches would share every challenge. Nothing in the current
    /// call graph forks one state under two families, but that was a property
    /// of where the parent happened to have advanced rather than of the
    /// construction. Every call site passes a distinct literal, and the
    /// verifier mirror must pass the same one.
    pub fn fork(&self, family: &[u8], branch_id: usize) -> Transcript {
        let mut forked = self.clone();
        forked.absorb_label(family);
        forked.absorb_u64(branch_id as u64);
        forked
    }

    // ------------------------------------------------------------------
    // Internal sponge plumbing — DuplexChallenger overwrite-mode
    // ------------------------------------------------------------------

    /// One duplexing step:
    /// 1. Overwrite the rate portion of the sponge state with the buffered
    ///    inputs (zero-padding the tail if fewer inputs were buffered).
    /// 2. Apply the Monolith permutation.
    /// 3. Refill the output buffer with the new rate portion (reversed so
    ///    `squeeze` can pop LIFO-style and produce values in absorption
    ///    order).
    fn duplexing(&mut self) {
        for i in 0..self.input_count {
            self.sponge_state[i] = self.input_buffer[i];
        }
        for i in self.input_count..RATE {
            self.sponge_state[i] = 0;
        }
        self.input_count = 0;

        monolith_permute(&mut self.sponge_state);

        for i in 0..RATE {
            self.output_buffer[RATE - 1 - i] = self.sponge_state[i];
        }
        self.output_count = RATE;
    }

    fn absorb_u64(&mut self, value: u64) {
        // Pending challenges become stale once new input arrives — same
        // semantics as zk-torch-3's `absorb`.
        self.output_count = 0;

        self.input_buffer[self.input_count] = value;
        self.input_count += 1;

        if self.input_count == RATE {
            self.duplexing();
        }
    }

    fn squeeze_u64(&mut self) -> u64 {
        if self.output_count == 0 {
            self.duplexing();
        }
        self.output_count -= 1;
        self.output_buffer[self.output_count]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_inputs_same_challenges() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");
        t1.append_scalar(b"val", &AlmostGoldilocksField(42));
        t2.append_scalar(b"val", &AlmostGoldilocksField(42));
        let c1 = t1.challenge_scalar(b"c");
        let c2 = t2.challenge_scalar(b"c");
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_inputs_different_challenges() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");
        t1.append_scalar(b"val", &AlmostGoldilocksField(42));
        t2.append_scalar(b"val", &AlmostGoldilocksField(43));
        assert_ne!(t1.challenge_scalar(b"c"), t2.challenge_scalar(b"c"));
    }

    #[test]
    fn different_labels_different_challenges() {
        let mut t1 = Transcript::new(b"a");
        let mut t2 = Transcript::new(b"b");
        assert_ne!(t1.challenge_scalar(b"c"), t2.challenge_scalar(b"c"));
    }

    /// Label-length prefix prevents `"ab" || "c"` from colliding with
    /// `"a" || "bc"`. Verifies our absorb_label includes the length byte.
    #[test]
    fn label_length_prefix_prevents_collision() {
        let mut t1 = Transcript::new(b"x");
        let mut t2 = Transcript::new(b"x");
        t1.append_scalar(b"ab", &AlmostGoldilocksField(0));
        t1.append_scalar(b"c", &AlmostGoldilocksField(0));
        t2.append_scalar(b"a", &AlmostGoldilocksField(0));
        t2.append_scalar(b"bc", &AlmostGoldilocksField(0));
        assert_ne!(t1.challenge_scalar(b""), t2.challenge_scalar(b""));
    }

    /// Two consecutive squeezes within one duplexing round must produce
    /// different values (otherwise the output buffer reuse is wrong).
    #[test]
    fn consecutive_squeezes_differ() {
        let mut t = Transcript::new(b"test");
        let a = t.challenge_scalar(b"x");
        let b = t.challenge_scalar(b"x");
        assert_ne!(a, b);
    }

    /// Squeezing across a duplexing boundary (more than RATE = 8 values)
    /// also produces distinct values. Exercises the duplexing trigger.
    #[test]
    fn squeezes_across_duplexing_boundary() {
        let mut t = Transcript::new(b"test");
        let n = RATE * 3;
        let vals = t.challenge_vector(b"x", n);
        let unique: std::collections::HashSet<u64> = vals.iter().map(|v| v.0).collect();
        assert_eq!(unique.len(), n, "challenge_vector produced duplicates");
    }

    /// New input absorption must invalidate any pending squeezed outputs —
    /// re-squeezing after an append yields a different value.
    #[test]
    fn absorb_invalidates_pending_squeezes() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = t1.clone();
        let _ = t1.challenge_scalar(b"a"); // triggers a duplexing, fills output_buffer
        t1.append_scalar(b"b", &AlmostGoldilocksField(7));
        let after = t1.challenge_scalar(b"c");

        let _ = t2.challenge_scalar(b"a");
        t2.append_scalar(b"b", &AlmostGoldilocksField(7));
        let after2 = t2.challenge_scalar(b"c");
        assert_eq!(after, after2, "deterministic after absorb");

        // And same setup without the intermediate append should differ.
        let mut t3 = Transcript::new(b"test");
        let _ = t3.challenge_scalar(b"a");
        let no_append = t3.challenge_scalar(b"c");
        assert_ne!(after, no_append, "absorption must affect subsequent squeezes");
    }

    #[test]
    fn fork_diverges_per_branch_id() {
        let t = Transcript::new(b"root");
        let mut f0 = t.fork(b"fam", 0);
        let mut f1 = t.fork(b"fam", 1);
        assert_ne!(f0.challenge_scalar(b"c"), f1.challenge_scalar(b"c"));
        // But two forks at the same (family, id) agree (deterministic) —
        // this is what lets the verifier replay a branch without replaying
        // the prover's device schedule.
        let mut g0 = t.fork(b"fam", 0);
        let mut h0 = t.fork(b"fam", 0);
        assert_eq!(g0.challenge_scalar(b"c"), h0.challenge_scalar(b"c"));
    }

    /// Distinct fork FAMILIES must not collide at the same branch id. Before
    /// the family label existed, forking one parent state by partition index
    /// 3 and by edge id 3 produced identical children, so two unrelated
    /// branches shared every challenge.
    #[test]
    fn fork_diverges_per_family() {
        let t = Transcript::new(b"root");
        let mut a = t.fork(b"dag_partition", 3);
        let mut b = t.fork(b"open_reducer", 3);
        assert_ne!(a.challenge_scalar(b"c"), b.challenge_scalar(b"c"),
                   "distinct fork families must not collide at the same id");
        // A family label must not be confusable with a longer/shorter one
        // (absorb_label length-prefixes, so this is structural, not luck).
        let mut c = t.fork(b"ft_bucket", 0);
        let mut d = t.fork(b"ft_bucket0", 0);
        assert_ne!(c.challenge_scalar(b"c"), d.challenge_scalar(b"c"));
    }

    #[test]
    fn ext2_challenge_components_differ() {
        let mut t = Transcript::new(b"test");
        let e = t.challenge_ext2(b"x");
        // Vanishingly unlikely that both components collide on a 252-bit
        // sponge; if this test ever fires you've broken the squeeze loop.
        assert_ne!(e.c0, e.c1);
    }

    /// Frozen golden vector for the transcript itself. The exact value
    /// depends on Monolith RC, MDS, the permutation, the label-length-prefix
    /// scheme, and the duplexing rate — so it pins all of those at once.
    #[test]
    fn transcript_golden_squeeze() {
        let mut t = Transcript::new(b"zk-torch-4-test");
        t.append_scalar(b"a", &AlmostGoldilocksField(1));
        t.append_scalar(b"b", &AlmostGoldilocksField(2));
        t.append_u64(b"c", 0xDEAD_BEEF);
        let c = t.challenge_scalar(b"out");
        const EXPECTED: u64 = 0x5df33cda688d0bce;
        assert_eq!(
            c.0, EXPECTED,
            "Transcript golden squeeze drifted: got {:#018x}, expected {:#018x}",
            c.0, EXPECTED
        );
    }

    /// Diagnostic helper: print the transcript golden output so you can paste
    /// the new value when bumping protocol parameters intentionally.
    #[test]
    #[ignore]
    fn dump_transcript_golden() {
        let mut t = Transcript::new(b"zk-torch-4-test");
        t.append_scalar(b"a", &AlmostGoldilocksField(1));
        t.append_scalar(b"b", &AlmostGoldilocksField(2));
        t.append_u64(b"c", 0xDEAD_BEEF);
        let c = t.challenge_scalar(b"out");
        println!("transcript_golden = 0x{:016x}", c.0);
    }
}
