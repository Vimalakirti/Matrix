//! Honest samplers for the prover's private coins.
//!
//! Ternary hiding coefficients follow the ternary candidate's specified
//! sampler (proof supplement, "Finite ternary sampling"): per coefficient,
//! up to 8192 two-bit proposals, accept the first value below three and
//! translate it to {-1, 0, 1}; exhaustion (probability 4^-8192) is an
//! explicit failure, never a silent default. The accepted law is exactly
//! uniform, and the stopping time is independent of the accepted value.
//!
//! The certified discrete Gaussian sampler lives in `gaussian.rs`.

use rand::RngCore;

pub const TERNARY_PROPOSALS: usize = 8192;

#[derive(Debug)]
pub struct SamplerExhausted;

/// Buffered two-bit proposals from a CSPRNG.
pub struct BitSource<'a, R: RngCore> {
    rng: &'a mut R,
    buf: u64,
    left: u32,
}

impl<'a, R: RngCore> BitSource<'a, R> {
    pub fn new(rng: &'a mut R) -> Self { BitSource { rng, buf: 0, left: 0 } }
    #[inline]
    pub fn two_bits(&mut self) -> u64 {
        if self.left < 2 { self.buf = self.rng.next_u64(); self.left = 64; }
        let v = self.buf & 3;
        self.buf >>= 2;
        self.left -= 2;
        v
    }
}

/// One uniform trit in {-1, 0, 1}.
#[inline]
pub fn ternary<R: RngCore>(bits: &mut BitSource<R>) -> Result<i64, SamplerExhausted> {
    for _ in 0..TERNARY_PROPOSALS {
        let v = bits.two_bits();
        if v < 3 { return Ok(v as i64 - 1); }
    }
    Err(SamplerExhausted)
}

/// `n` uniform trits.
pub fn ternary_vec<R: RngCore>(rng: &mut R, n: usize) -> Result<Vec<i64>, SamplerExhausted> {
    let mut bits = BitSource::new(rng);
    (0..n).map(|_| ternary(&mut bits)).collect()
}

/// The prover's coin generator: ChaCha20 seeded from the OS. Tests seed it
/// deterministically.
pub fn prover_rng() -> rand_chacha::ChaCha20Rng {
    use rand::SeedableRng;
    rand_chacha::ChaCha20Rng::from_entropy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn ternary_is_balanced() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(1);
        let v = ternary_vec(&mut rng, 300_000).unwrap();
        let mut c = [0usize; 3];
        for x in v { c[(x + 1) as usize] += 1; }
        for k in c { assert!((99_000..101_000).contains(&k), "{c:?}"); }
    }
}
