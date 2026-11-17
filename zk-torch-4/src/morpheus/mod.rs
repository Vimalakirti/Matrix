//! Reimplementation of the prover to the IEEE S&P'27 Morpheus construction:
//! hiding lattice PCS (randomized Ajtai commitments over R = F_q[Y]/(Y^4096+1),
//! masked range/query link, Gaussian authentication) and the masked graph PIOP
//! over K = F_q[Θ]/(Θ^8 − 3). Built bottom-up; the previous prover stays in
//! place until the switch.

pub mod field8;
pub mod packed;
pub mod ring;
pub mod params;
pub mod keys;
pub mod sampler;
pub mod commit;
pub mod coins;
pub mod gaussian;
pub mod gaussian_fast;
pub mod gaussian_gpu;
pub mod commit_gpu;
pub mod transcript;
pub mod pcs_maps;
pub mod rejection;
pub mod auth;
pub mod auth_gpu;
pub mod link;
pub mod link_fast;
pub mod link_gpu;
pub mod pcs_gpu;
pub mod piop;
pub mod request;
pub mod models;
pub mod driver;
pub mod wire;
pub mod ar_statement;

pub mod statement;

pub mod protocol;

pub mod auth_compact;

pub mod public_support;
pub mod bounds;
