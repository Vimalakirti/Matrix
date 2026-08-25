//! Cryptographic primitives over the almost-Goldilocks field.
//!
//! Currently houses:
//! - [`digest::Digest`]: a 4-element hash output.
//! - [`monolith`]: the Monolith permutation (CPU) tuned for `F_q`, plus the
//!   `(round, position) -> AlmostGoldilocksField` round-constant table
//!   derived deterministically from SHA-256.

pub mod digest;
pub mod monolith;

pub use digest::Digest;
