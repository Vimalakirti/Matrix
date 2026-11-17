//! Parameter profiles. The production profile is the paper's ternary-hiding
//! candidate (§Concrete Implementation Parameters, "Ternary-hiding candidate
//! used for the estimates", and the proof supplement's ternary bookkeeping).
//! Tests use the same code at reduced dimensions via `Profile::toy`.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// Identifier bound into transcripts.
    pub name: &'static str,
    /// Physical output rank (rows of R).
    pub rank: usize,
    /// Authentication rows τ.
    pub tau: usize,
    /// Hiding ring columns per block (k_W = k_L = k_O).
    pub hiding: usize,
    /// Payload ring columns per source (source dimension minus the suffix).
    pub source_payload_cols: usize,
    /// Semantic A-coefficients encoded by the auxiliary (D_sc).
    pub aux_coeffs: usize,
    /// Old / final variable degree caps of the link sumcheck.
    pub deg_old: usize,
    pub deg_final: usize,
    /// Response width σ and strict coefficient cutoff B_Z.
    pub sigma: u64,
    pub b_z: u64,
    /// Raw proof noises (σ_raw,W, σ_raw,L, σ_raw,O).
    pub raw_noise: [u64; 3],
    /// Rejection constant M, attempt cap, salt bits.
    pub rej_m: u64,
    pub attempts: usize,
    pub salt_bits: usize,
    /// Source count cap n_max and payload energy cap S_max (log2).
    pub n_max: usize,
    pub s_max_log2: u32,
}

/// Ring columns of one encoded A element (8 K-coordinates x 64 bits).
pub const A_ENC_COLS: usize = 8;

impl Profile {
    /// The ternary-hiding candidate (production).
    pub fn ternary() -> Self {
        Profile {
            name: "morpheus-ternary-shake256-v2",
            rank: 16,
            tau: 2,
            hiding: 648,
            source_payload_cols: 32104, // 131497984 / 4096
            aux_coeffs: 114,            // D_sc = 4 * 27 + 6
            deg_old: 4,
            deg_final: 6,
            sigma: 3_076_346_150,
            b_z: 52_297_884_550,
            raw_noise: [56, 1344, 80],
            rej_m: 3,
            attempts: 512,
            salt_bits: 512,
            n_max: 1 << 18,
            s_max_log2: 38,
        }
    }
    /// Reduced dimensions for tests (same algebra and code paths).
    pub fn toy() -> Self {
        // 3 + 3 + 2 = 8 source columns (ℓ = 15). The link degree caps stay
        // 4 / 6 (they must cover the link polynomial's true degrees), so the
        // auxiliary block has 15·4 + 6 = 66 coefficients.
        Profile { name: "morpheus-toy-shake256-v2", rank: 2, hiding: 3, source_payload_cols: 3, aux_coeffs: 66, ..Self::ternary() }
    }

    /// Mid-size test profile: 2^20 coordinates per source (251 + 3 + 2
    /// columns), true link degree caps (20·4 + 6 = 86 aux coefficients).
    pub fn mini() -> Self {
        Profile { name: "morpheus-mini-shake256-v2", rank: 2, hiding: 3, source_payload_cols: 251, aux_coeffs: 86, ..Self::ternary() }
    }

    /// Identity-error ring columns = rank.
    pub fn identity_cols(&self) -> usize { self.rank }
    pub fn source_cols(&self) -> usize { self.source_payload_cols + self.hiding + self.rank }
    pub fn link_cols(&self) -> usize { A_ENC_COLS + self.hiding + self.rank }
    pub fn aux_cols(&self) -> usize { A_ENC_COLS * self.aux_coeffs + self.hiding + self.rank }
    /// Physical scalar dimensions D_W, D_L, D_O.
    pub fn dims(&self) -> (usize, usize, usize) {
        let n = super::ring::N;
        (n * self.source_cols(), n * self.link_cols(), n * self.aux_cols())
    }
    /// Raw index width ℓ of a source (2^ℓ = D_W).
    pub fn ell(&self) -> usize {
        let d = self.dims().0;
        assert!(d.is_power_of_two());
        d.trailing_zeros() as usize
    }
    /// Link-sumcheck degree caps d_i of the N = ℓ + 1 variables.
    pub fn degs(&self) -> Vec<usize> {
        let mut v = vec![self.deg_old; self.ell()];
        v.push(self.deg_final);
        assert_eq!(v.iter().sum::<usize>(), self.aux_coeffs);
        v
    }
    /// Setup caps of one opening: at most n_max sources and payload energy
    /// Σ‖payload‖² ≤ 2^s_max. Payload digits are ternary, so the energy is
    /// at most the payload capacity, and the digits are counted only when
    /// the capacity exceeds the cap (more than 2091 full sources).
    pub fn check_caps(&self, raw: &dyn super::packed::Sources) {
        use rayon::prelude::*;
        let n = raw.count();
        assert!(n <= self.n_max, "{} sources exceed n_max = {}", n, self.n_max);
        let cap = (self.source_payload_cols * super::ring::N) as u128;
        if n as u128 * cap <= 1u128 << self.s_max_log2 { return; }
        let pl = (cap as usize).min(raw.src_len());
        // Every digit is in {-1, 0, 1}: the energy counts nonzero digits.
        let e: u128 = (0..n).into_par_iter().map(|s| raw.nonzeros(s, 0, pl).0 as u128).sum();
        assert!(e <= 1u128 << self.s_max_log2, "payload energy {e} exceeds 2^{}", self.s_max_log2);
    }
    /// Exact integer Euclidean cap R_P^2 = σ^2 [D + 2⌈sqrt(114 D)⌉ + 228].
    pub fn r2(&self, d: usize) -> u128 {
        let s2 = (self.sigma as u128) * (self.sigma as u128);
        let root = ((114u128 * d as u128) as f64).sqrt().ceil() as u128;
        // Correct the float ceil to an exact integer ceiling of sqrt(114 D).
        let mut r = root.saturating_sub(2);
        while r * r < 114 * d as u128 { r += 1; }
        s2 * (d as u128 + 2 * r + 228)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ternary_dimensions_match_the_paper() {
        let p = Profile::ternary();
        let (dw, dl, d_o) = p.dims();
        assert_eq!(dw, 1 << 27);
        assert_eq!(dl, 2_752_512);
        assert_eq!(d_o, 6_455_296);
        assert_eq!(dw + dl + d_o, 143_425_536); // response coordinates per row
        assert_eq!(p.source_payload_cols * 4096, 131_497_984);
        assert_eq!((p.hiding + p.rank) * 4096, 2_719_744); // suffix incl. identity error
        assert_eq!(p.b_z, 17 * p.sigma);
        assert!(2048u128 * p.b_z as u128 <= 18446744069414584289u128);
        assert_eq!(p.deg_old * 27 + p.deg_final, p.aux_coeffs);
    }
}
