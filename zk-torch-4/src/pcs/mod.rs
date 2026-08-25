//! The packed polynomial commitment scheme (`ZK4_PCS=packed`).
//!
//! Three layers, in the order the opening runs them:
//!
//! 1. **Packing + randomized commitment** ([`crate::commit::layout`],
//!    [`crate::commit::hiding`]) — leaves become aligned blocks of a larger
//!    committed polynomial, `C = A_msg·x + A_hid·s`.
//! 2. **Link** ([`link`]) — one tagged sumcheck that certifies every source
//!    claim and every coefficient's range bound, terminating at a shared point
//!    `ξ` with one claim per commitment.
//! 3. **Masked RLC** (next phase) — merges those same-point claims into `τ`
//!    Gaussian-masked responses.
//!
//! The order is forced: linearity can only merge claims that already share an
//! evaluation point, so the link has to run first.

pub mod integration;
pub mod link;
pub mod mrlc;
