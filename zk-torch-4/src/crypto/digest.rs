//! 4-element hash digest, matching the output rate of [`crate::crypto::monolith`].

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

pub const DIGEST_SIZE: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Digest {
    pub elements: [AlmostGoldilocksField; DIGEST_SIZE],
}

impl Digest {
    pub const fn zero() -> Self {
        Self {
            elements: [
                AlmostGoldilocksField::zero(),
                AlmostGoldilocksField::zero(),
                AlmostGoldilocksField::zero(),
                AlmostGoldilocksField::zero(),
            ],
        }
    }

    /// Construct from four raw `u64` values. Inputs are normalized to canonical
    /// form via `AlmostGoldilocksField::reduce`.
    pub fn from_raw(raw: [u64; DIGEST_SIZE]) -> Self {
        Self {
            elements: [
                AlmostGoldilocksField::new(raw[0]).reduce(),
                AlmostGoldilocksField::new(raw[1]).reduce(),
                AlmostGoldilocksField::new(raw[2]).reduce(),
                AlmostGoldilocksField::new(raw[3]).reduce(),
            ],
        }
    }

    /// Canonicalize every element and dump the raw `u64`s.
    pub fn to_raw(self) -> [u64; DIGEST_SIZE] {
        [
            self.elements[0].reduce().0,
            self.elements[1].reduce().0,
            self.elements[2].reduce().0,
            self.elements[3].reduce().0,
        ]
    }
}
