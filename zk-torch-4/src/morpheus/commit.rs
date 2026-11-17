//! Randomized Ajtai commitments (paper §PCS, "Randomized Ajtai commitments"):
//! for block kind P with normal-form key A_P = [P_P | B_P | Id],
//!     C = P_P · x + B_P · s + e,
//! x the payload (ternary digits / bits / encoded masks), s the hiding
//! secret and e the identity error, both fresh uniform ternary in the ternary
//! candidate. Payload columns that are entirely zero are skipped, so the
//! payload part is support-sensitive at ring-column granularity; hiding and
//! error parts are dense by design.
//!
//! This is the CPU reference (exact CRT matvec, see ring.rs). The production
//! path will run on the GPU with NTT-domain key caches.

use super::keys::{key_element, KeyKind, KeySeed};
use super::params::Profile;
use super::ring::{center, from_signed, matvec, N};
use super::sampler::{ternary_vec, SamplerExhausted};
use rand::RngCore;

/// A block witness in physical layout: payload ring columns (None = zero),
/// hiding ring columns, identity-error ring columns (one per output row).
#[derive(Clone, Debug)]
pub struct BlockWitness {
    pub payload: Vec<Option<Vec<i64>>>,
    pub hiding: Vec<Vec<i64>>,
    pub error: Vec<Vec<i64>>,
}

impl BlockWitness {
    /// Fresh ternary hiding and error for `payload`.
    pub fn with_fresh_hiding<R: RngCore>(
        profile: &Profile,
        payload: Vec<Option<Vec<i64>>>,
        rng: &mut R,
    ) -> Result<Self, SamplerExhausted> {
        let hiding = (0..profile.hiding).map(|_| ternary_vec(rng, N)).collect::<Result<_, _>>()?;
        let error = (0..profile.rank).map(|_| ternary_vec(rng, N)).collect::<Result<_, _>>()?;
        Ok(BlockWitness { payload, hiding, error })
    }
    /// Largest |coefficient| over payload and hiding (bounds the CRT product).
    fn max_abs(&self) -> u64 {
        let p = self.payload.iter().flatten().flat_map(|c| c.iter()).map(|v| v.unsigned_abs()).max().unwrap_or(0);
        let h = self.hiding.iter().flat_map(|c| c.iter()).map(|v| v.unsigned_abs()).max().unwrap_or(0);
        p.max(h).max(1)
    }
}

/// Commit to `w` under the `kind` key with `payload_cols` payload columns.
/// Returns `profile.rank` ring elements (canonical coefficients).
pub fn commit(profile: &Profile, seed: &KeySeed, kind: KeyKind, payload_cols: usize, w: &BlockWitness) -> Vec<Vec<u64>> {
    assert!(w.payload.len() <= payload_cols);
    assert_eq!(w.hiding.len(), profile.hiding);
    assert_eq!(w.error.len(), profile.rank);
    // Columns that contribute: nonzero payload columns, then all hiding columns.
    let mut cols: Vec<u64> = Vec::new();
    let mut xs: Vec<Vec<i64>> = Vec::new();
    for (c, col) in w.payload.iter().enumerate() {
        if let Some(v) = col { if v.iter().any(|&t| t != 0) { cols.push(c as u64); xs.push(v.clone()); } }
    }
    for (h, col) in w.hiding.iter().enumerate() { cols.push((payload_cols + h) as u64); xs.push(col.clone()); }
    // Bound on every integer output coefficient: (#cols) · N · H · M.
    let log2_bound = (xs.len().max(1) as f64).log2() + 12.0 + 63.0 + (w.max_abs() as f64).log2().max(0.0);
    let key = |i: usize, c: usize| -> Vec<i64> {
        key_element(seed, kind, i as u32, cols[c]).iter().map(|&v| center(v)).collect()
    };
    let mut out = matvec(profile.rank, xs.len(), &key, &xs, log2_bound);
    // Identity error.
    for (i, row) in out.iter_mut().enumerate() {
        for j in 0..N {
            let e = from_signed(w.error[i][j]);
            let s = row[j] as u128 + e as u128;
            row[j] = (s % almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as u128) as u64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ring::mul_schoolbook;
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
    use rand::SeedableRng;

    fn direct(profile: &Profile, seed: &KeySeed, kind: KeyKind, payload_cols: usize, w: &BlockWitness) -> Vec<Vec<u64>> {
        let mut out = vec![vec![0u64; N]; profile.rank];
        let mut add = |row: &mut Vec<u64>, v: &[u64]| for j in 0..N { row[j] = ((row[j] as u128 + v[j] as u128) % Q as u128) as u64 };
        for i in 0..profile.rank {
            for (c, col) in w.payload.iter().enumerate() {
                if let Some(v) = col {
                    let x: Vec<u64> = v.iter().map(|&t| from_signed(t)).collect();
                    let p = mul_schoolbook(&key_element(seed, kind, i as u32, c as u64), &x);
                    add(&mut out[i], &p);
                }
            }
            for (h, col) in w.hiding.iter().enumerate() {
                let x: Vec<u64> = col.iter().map(|&t| from_signed(t)).collect();
                let p = mul_schoolbook(&key_element(seed, kind, i as u32, (payload_cols + h) as u64), &x);
                add(&mut out[i], &p);
            }
            let e: Vec<u64> = w.error[i].iter().map(|&t| from_signed(t)).collect();
            add(&mut out[i], &e);
        }
        out
    }

    #[test]
    fn commitment_matches_schoolbook_and_is_linear() {
        let profile = Profile::toy();
        let seed = KeySeed([9, 8, 7, 6, 5, 4, 3, 2]);
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(3);
        let pc = profile.source_payload_cols;
        let payload: Vec<Option<Vec<i64>>> = (0..pc)
            .map(|c| if c == 1 { None } else { Some(ternary_vec(&mut rng, N).unwrap()) })
            .collect();
        let w = BlockWitness::with_fresh_hiding(&profile, payload, &mut rng).unwrap();
        let c = commit(&profile, &seed, KeyKind::Source, pc, &w);
        assert_eq!(c, direct(&profile, &seed, KeyKind::Source, pc, &w));

        // Linearity in the full witness (payload, hiding, error together).
        let w2 = BlockWitness::with_fresh_hiding(&profile, (0..pc).map(|_| Some(ternary_vec(&mut rng, N).unwrap())).collect(), &mut rng).unwrap();
        let sum = |a: &Vec<i64>, b: &Vec<i64>| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<i64>>();
        let w3 = BlockWitness {
            payload: w.payload.iter().zip(&w2.payload).map(|(a, b)| Some(sum(a.as_ref().unwrap_or(&vec![0; N]), b.as_ref().unwrap()))).collect(),
            hiding: w.hiding.iter().zip(&w2.hiding).map(|(a, b)| sum(a, b)).collect(),
            error: w.error.iter().zip(&w2.error).map(|(a, b)| sum(a, b)).collect(),
        };
        let c2 = commit(&profile, &seed, KeyKind::Source, pc, &w2);
        let c3 = commit(&profile, &seed, KeyKind::Source, pc, &w3);
        for i in 0..profile.rank {
            for j in 0..N { assert_eq!(c3[i][j], ((c[i][j] as u128 + c2[i][j] as u128) % Q as u128) as u64); }
        }
    }
}
