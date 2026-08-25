use goldilocks_cuda::{GoldilocksField, GoldilocksExt2};
use goldilocks_cuda::basefold::BasefoldTranscript;
use goldilocks_cuda::poseidon2::Poseidon2Hash;

// ============================================================================
// Poseidon2 Constants (Width 8, from Plonky3)
// ============================================================================

const WIDTH: usize = 8;
const RATE: usize = 4;

/// External round constants: 8 rounds x 8 elements
/// First 4 rounds are initial, last 4 are terminal
const RC_EXT: [[u64; 8]; 8] = [
    // Initial round 0
    [0xdd5743e7f2a5a5d9, 0xcb3a864e58ada44b, 0xd5a3726889dedc0d, 0x79365a7f967583c1,
     0xb7820c177b0a3c30, 0x68a60479943b7240, 0x0a22eeab67b97c41, 0xbee9d90e037fa7d4],
    // Initial round 1
    [0xf1dda5b9259dfcb4, 0x27515210be112d59, 0x654a4db8f15f6c3a, 0x5a9bfff8c29db0c4,
     0xb72f2628c95a885b, 0xffcb54fc166beadd, 0xe32f91d59c463772, 0x9e0ff2a9bbea79db],
    // Initial round 2
    [0xce57d6245ddca6b2, 0xb1fc8d402bba1eb1, 0x64df974fb666d528, 0x59e4e5b237e52e78,
     0x5e255e2742caa8fc, 0xc4a17304e330ef9a, 0x60e3a513cdd9f0b1, 0xbf29a6a5a0c9c8ba],
    // Initial round 3
    [0xcea721cce82fb11b, 0xe5b55eb8098ece81, 0x4ec66bfb89f7c380, 0xc33be12b6ef7e4fa,
     0xa41b5f5b84c57d7d, 0xb526206b2138a936, 0x9f7a5ac62f16eb4e, 0x7de77f405f683aa5],
    // Terminal round 0
    [0x014ef1197d341346, 0x9725e20825d07394, 0xc8ff2a22516f3604, 0xde1b51f7cf493493,
     0xa7e3af1a53eadb78, 0x9b7b7ee0ddf9f229, 0xf0a460f2dc649e7d, 0xdc4448d02c2cb823],
    // Terminal round 1
    [0xaa62c88e0b294011, 0x058eb9d810ce9f74, 0x4e3d7a5403566d24, 0xe8ae66da2f8b7a63,
     0x3fdad3b08dae2e0b, 0xec81e61303dd409b, 0x3cbaa1cd35d31fc4, 0x847f7806eec88ffa],
    // Terminal round 2
    [0x98ae09a325893803, 0xf8a6475077968838, 0xdf55b7419443de5b, 0xdbe15699e696ca70,
     0x8e54a27d8db00424, 0x7679a45cd9d1e12a, 0x2844f52be73e0e2f, 0xd41f8eb31ada34f8],
    // Terminal round 3
    [0xe9dd318bae1f3961, 0xf7462137299efe1a, 0x6dbbe06779e1d573, 0xfe35b05cbe707632,
     0x5d8896b12654fd8c, 0x6f96ef47c32d4ae2, 0xb0caa221dbbfc0da, 0x0bc2a5bf1f238d3f],
];

/// Internal round constants (22 values, applied only to first element)
const RC_INT: [u64; 22] = [
    0x488897d85ff51f56, 0x1140737ccb162218, 0xa7eeb9215866ed35,
    0x9bd2976fee49fcc9, 0xc0c8f0de580a3fcc, 0x4fb2dae6ee8fc793,
    0x343a89f35f37395b, 0x223b525a77ca72c8, 0x56ccb62574aaa918,
    0xc4d507d8027af9ed, 0xa080673cf0b7e95c, 0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69, 0xe9e3f7993938f186, 0x1b761c80e772f459,
    0x606cec607a1b5fac, 0x14a0c2e1d45f03cd, 0x4eace8855398574f,
    0xf905ca7103eff3e6, 0xf8c8f8d20862c059, 0xb524fe8bdd678e5a,
    0xfbb7865901a1ec41,
];

/// Internal diffusion diagonal for width 8
const DIAG_8: [u64; 8] = [
    0xa98811a1fed4e3a5, 0x1cc48b54f377e2a0, 0xe40cd4f6c5609a26,
    0x11de79ebca97a4a3, 0x9177c73d8b7e929c, 0x2a6fe8085797e791,
    0x3de6e93329f8d5ad, 0x3f7af9125da962fe,
];

// ============================================================================
// Poseidon2 Permutation (CPU)
// ============================================================================

#[inline]
fn gl(v: u64) -> GoldilocksField {
    GoldilocksField(v)
}

/// S-box: x^7 via addition chain x^2 -> x^4 -> x^3 -> x^7
#[inline]
fn sbox(x: GoldilocksField) -> GoldilocksField {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x2 * x;
    x4 * x3
}

/// Apply 4x4 MDS circulant matrix [2,3,1,1; 1,2,3,1; 1,1,2,3; 3,1,1,2]
#[inline]
fn mds4(s: &mut [GoldilocksField]) {
    let t01 = s[0] + s[1];
    let t23 = s[2] + s[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + s[1];
    let t01233 = t0123 + s[3];

    let s0 = t01123 + t01;              // 2*x0 + 3*x1 + x2 + x3
    let s1 = t01123 + (s[2] + s[2]);    // x0 + 2*x1 + 3*x2 + x3
    let s2 = t01233 + t23;              // x0 + x1 + 2*x2 + 3*x3
    let s3 = t01233 + (s[0] + s[0]);    // 3*x0 + x1 + x2 + 2*x3

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

/// Full MDS transformation for width-8 state:
/// 1. Apply 4x4 MDS to each 4-element chunk
/// 2. Add cross-chunk sums
#[inline]
fn mds_light_8(state: &mut [GoldilocksField; WIDTH]) {
    mds4(&mut state[0..4]);
    mds4(&mut state[4..8]);

    let sum0 = state[0] + state[4];
    let sum1 = state[1] + state[5];
    let sum2 = state[2] + state[6];
    let sum3 = state[3] + state[7];

    state[0] = state[0] + sum0;
    state[1] = state[1] + sum1;
    state[2] = state[2] + sum2;
    state[3] = state[3] + sum3;
    state[4] = state[4] + sum0;
    state[5] = state[5] + sum1;
    state[6] = state[6] + sum2;
    state[7] = state[7] + sum3;
}

/// Internal diffusion: state[i] = state[i] * diag[i] + sum(state)
#[inline]
fn diffusion_8(state: &mut [GoldilocksField; WIDTH]) {
    let mut sum = state[0];
    for i in 1..WIDTH {
        sum = sum + state[i];
    }
    for i in 0..WIDTH {
        let prod = state[i] * gl(DIAG_8[i]);
        state[i] = prod + sum;
    }
}

/// Full Poseidon2 permutation for width 8.
/// Structure: 4 initial external rounds + 22 internal rounds + 4 terminal external rounds.
fn poseidon2_permute(state: &mut [GoldilocksField; WIDTH]) {
    // Initial external rounds (4 rounds)
    for r in 0..4 {
        for i in 0..WIDTH {
            state[i] = state[i] + gl(RC_EXT[r][i]);
            state[i] = sbox(state[i]);
        }
        mds_light_8(state);
    }

    // Internal rounds (22 rounds)
    for r in 0..22 {
        state[0] = state[0] + gl(RC_INT[r]);
        state[0] = sbox(state[0]);
        diffusion_8(state);
    }

    // Terminal external rounds (4 rounds)
    for r in 0..4 {
        for i in 0..WIDTH {
            state[i] = state[i] + gl(RC_EXT[4 + r][i]);
            state[i] = sbox(state[i]);
        }
        mds_light_8(state);
    }
}

// ============================================================================
// DuplexChallenger Transcript (Plonky3-compatible sponge)
// ============================================================================

/// Poseidon2-based Fiat-Shamir transcript using DuplexChallenger sponge.
///
/// Matches the Plonky3 DuplexChallenger pattern:
/// - Width 8, Rate 4, Capacity 4
/// - Overwrite-mode sponge (not XOR)
/// - Buffered input/output with duplexing
#[derive(Clone, Debug)]
pub struct Transcript {
    sponge_state: [GoldilocksField; WIDTH],
    input_buffer: [GoldilocksField; RATE],
    output_buffer: [GoldilocksField; RATE],
    input_count: usize,
    output_count: usize,
}

impl Transcript {
    pub fn new(label: &[u8]) -> Self {
        let mut t = Self {
            sponge_state: [GoldilocksField::zero(); WIDTH],
            input_buffer: [GoldilocksField::zero(); RATE],
            output_buffer: [GoldilocksField::zero(); RATE],
            input_count: 0,
            output_count: 0,
        };
        // Domain separation: absorb label bytes as field elements
        for &b in label {
            t.absorb(GoldilocksField(b as u64));
        }
        t
    }

    /// Absorb a label as domain separator: length prefix + label bytes.
    /// Length prefix prevents collisions (e.g., "ab"+"c" vs "a"+"bc").
    /// Empty labels are skipped for backward compatibility with BasefoldTranscript.
    fn absorb_label(&mut self, label: &[u8]) {
        if label.is_empty() {
            return;
        }
        self.absorb(GoldilocksField(label.len() as u64));
        for &b in label {
            self.absorb(GoldilocksField(b as u64));
        }
    }

    /// Absorb a field element into the transcript.
    pub fn append_scalar(&mut self, label: &[u8], scalar: &GoldilocksField) {
        self.absorb_label(label);
        self.absorb(*scalar);
    }

    /// Absorb multiple field elements.
    pub fn append_scalars(&mut self, label: &[u8], scalars: &[GoldilocksField]) {
        self.absorb_label(label);
        for s in scalars {
            self.absorb(*s);
        }
    }

    /// Absorb a u64 value.
    pub fn append_u64(&mut self, label: &[u8], value: u64) {
        self.absorb_label(label);
        self.absorb(GoldilocksField(value));
    }

    /// Squeeze a field element challenge.
    pub fn challenge_scalar(&mut self, label: &[u8]) -> GoldilocksField {
        self.absorb_label(label);
        self.squeeze()
    }

    /// Get a fingerprint of the current transcript state for debugging.
    pub fn fingerprint(&self) -> u64 {
        let mut h = (self.input_count * 100 + self.output_count) as u64;
        for &v in &self.sponge_state {
            h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(v.0);
        }
        h
    }

    /// Squeeze an Ext2 challenge (two base field elements).
    pub fn challenge_ext2(&mut self, label: &[u8]) -> GoldilocksExt2 {
        self.absorb_label(label);
        let c0 = self.squeeze();
        let c1 = self.squeeze();
        GoldilocksExt2::new(c0, c1)
    }

    /// Absorb an Ext2 value into the transcript.
    pub fn append_ext2(&mut self, label: &[u8], val: &GoldilocksExt2) {
        self.absorb_label(label);
        self.absorb(val.c0);
        self.absorb(val.c1);
    }

    /// Squeeze multiple field element challenges.
    pub fn challenge_vector(&mut self, label: &[u8], len: usize) -> Vec<GoldilocksField> {
        self.absorb_label(label);
        (0..len).map(|_| self.squeeze()).collect()
    }

    /// Perform duplexing:
    /// 1. Overwrite rate portion with input buffer
    /// 2. Zero remaining rate positions
    /// 3. Apply Poseidon2 permutation
    /// 4. Extract rate portion to output buffer
    fn duplexing(&mut self) {
        // Overwrite rate portion with buffered inputs
        for i in 0..self.input_count {
            self.sponge_state[i] = self.input_buffer[i];
        }
        // Zero remaining rate positions
        for i in self.input_count..RATE {
            self.sponge_state[i] = GoldilocksField::zero();
        }
        self.input_count = 0;

        // Apply Poseidon2 permutation
        poseidon2_permute(&mut self.sponge_state);

        // Extract rate portion to output buffer (reversed for LIFO pop = FIFO read)
        for i in 0..RATE {
            self.output_buffer[RATE - 1 - i] = self.sponge_state[i];
        }
        self.output_count = RATE;
    }

    /// Absorb a field element (observe in DuplexChallenger terminology)
    fn absorb(&mut self, value: GoldilocksField) {
        // Invalidate output buffer — new data means old challenges are stale
        self.output_count = 0;

        self.input_buffer[self.input_count] = value;
        self.input_count += 1;

        // If input buffer is full, perform duplexing
        if self.input_count == RATE {
            self.duplexing();
        }
    }

    /// Squeeze a field element (sample in DuplexChallenger terminology)
    fn squeeze(&mut self) -> GoldilocksField {
        // If no buffered outputs, perform duplexing
        if self.output_count == 0 {
            self.duplexing();
        }

        // Pop from output buffer (LIFO)
        self.output_count -= 1;
        self.output_buffer[self.output_count]
    }

    /// Get a copy of the internal state for external use.
    pub fn get_state(&self) -> Vec<u64> {
        self.sponge_state.iter().map(|f| f.0).collect()
    }

    /// Fork this transcript into a partition-specific sub-transcript.
    pub fn fork(&self, partition_id: usize) -> Transcript {
        let mut forked = self.clone();
        forked.absorb(GoldilocksField(partition_id as u64));
        forked
    }
}

impl BasefoldTranscript for Transcript {
    fn observe_field(&mut self, value: GoldilocksField) {
        self.append_scalar(b"", &value);
    }
    fn observe_ext2(&mut self, value: GoldilocksExt2) {
        self.append_ext2(b"", &value);
    }
    fn observe_hash(&mut self, hash: &Poseidon2Hash) {
        for e in &hash.elements {
            self.append_scalar(b"", e);
        }
    }
    fn sample_challenge(&mut self) -> GoldilocksField {
        self.challenge_scalar(b"")
    }
    fn sample_challenge_ext2(&mut self) -> GoldilocksExt2 {
        self.challenge_ext2(b"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_deterministic() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");

        t1.append_scalar(b"val", &GoldilocksField(42));
        t2.append_scalar(b"val", &GoldilocksField(42));

        let c1 = t1.challenge_scalar(b"challenge");
        let c2 = t2.challenge_scalar(b"challenge");
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_transcript_different_inputs() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");

        t1.append_scalar(b"val", &GoldilocksField(42));
        t2.append_scalar(b"val", &GoldilocksField(43));

        let c1 = t1.challenge_scalar(b"challenge");
        let c2 = t2.challenge_scalar(b"challenge");
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_transcript_different_labels() {
        let mut t1 = Transcript::new(b"protocol-a");
        let mut t2 = Transcript::new(b"protocol-b");

        t1.append_scalar(b"", &GoldilocksField(100));
        t2.append_scalar(b"", &GoldilocksField(100));

        let c1 = t1.challenge_scalar(b"");
        let c2 = t2.challenge_scalar(b"");
        assert_ne!(c1, c2, "Different labels should produce different challenges");
    }

    #[test]
    fn test_poseidon2_permute_nonzero() {
        // Permuting all-zero state should produce non-zero output
        let mut state = [GoldilocksField::zero(); WIDTH];
        poseidon2_permute(&mut state);
        let any_nonzero = state.iter().any(|f| f.0 != 0);
        assert!(any_nonzero, "Poseidon2 permutation of zero should produce nonzero");
    }

    #[test]
    fn test_poseidon2_permute_deterministic() {
        let mut s1 = [GoldilocksField::zero(); WIDTH];
        let mut s2 = [GoldilocksField::zero(); WIDTH];
        s1[0] = GoldilocksField(42);
        s2[0] = GoldilocksField(42);
        poseidon2_permute(&mut s1);
        poseidon2_permute(&mut s2);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_sponge_multiple_squeezes() {
        let mut t = Transcript::new(b"test");
        t.append_scalar(b"", &GoldilocksField(1));

        // Squeeze 8 challenges — should trigger 2 duplexings (4 per)
        let challenges: Vec<GoldilocksField> = (0..8)
            .map(|_| t.challenge_scalar(b""))
            .collect();

        // All should be non-zero (with overwhelming probability)
        for (i, c) in challenges.iter().enumerate() {
            assert_ne!(c.0, 0, "Challenge {} should be non-zero", i);
        }

        // All should be distinct
        for i in 0..challenges.len() {
            for j in (i + 1)..challenges.len() {
                assert_ne!(challenges[i], challenges[j],
                    "Challenges {} and {} should differ", i, j);
            }
        }
    }

    #[test]
    fn test_fork_independence() {
        let mut t = Transcript::new(b"test");
        t.append_scalar(b"", &GoldilocksField(99));

        let mut f1 = t.fork(0);
        let mut f2 = t.fork(1);

        let c1 = f1.challenge_scalar(b"");
        let c2 = f2.challenge_scalar(b"");
        assert_ne!(c1, c2, "Forked transcripts with different IDs should diverge");
    }

    #[test]
    fn test_label_domain_separation() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");

        t1.append_scalar(b"foo", &GoldilocksField(100));
        t2.append_scalar(b"bar", &GoldilocksField(100));

        let c1 = t1.challenge_scalar(b"ch");
        let c2 = t2.challenge_scalar(b"ch");
        assert_ne!(c1, c2, "Different labels should produce different challenges");
    }

    #[test]
    fn test_label_length_prefix_collision_resistance() {
        let mut t1 = Transcript::new(b"test");
        let mut t2 = Transcript::new(b"test");

        t1.append_scalar(b"ab", &GoldilocksField(100));
        t1.append_scalar(b"c", &GoldilocksField(200));

        t2.append_scalar(b"a", &GoldilocksField(100));
        t2.append_scalar(b"bc", &GoldilocksField(200));

        let c1 = t1.challenge_scalar(b"ch");
        let c2 = t2.challenge_scalar(b"ch");
        assert_ne!(c1, c2, "Length-prefixed labels should prevent collisions");
    }
}
