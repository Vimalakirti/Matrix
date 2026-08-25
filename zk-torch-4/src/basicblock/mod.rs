//! BasicBlock — the per-node ZKML operation interface.
//!
//! Each block exposes:
//! - `run(inputs)`: forward-pass evaluation (CPU host-resident outputs).
//! - `run_gpu(inputs)`: optional GPU-resident fast path. Default forwards to
//!   `run`. Per philosophy rule #7 we document the default explicitly: blocks
//!   without a GPU kernel (Reducer, lookup auxiliaries, etc.) keep
//!   `run_gpu == run`. When a GPU kernel exists for the op the override is a
//!   strict perf improvement, never a soundness divergence.
//! - `prove(...)`: backward-pass claim transform / sumcheck.
//! - `verify(...)`: verifier-side check of the prove output.
//!
//! Blocks are dispatched by [`BasicBlockType`], an enum that wires the trait
//! to a concrete instance.

pub mod add;
pub mod concat;
pub mod conv;
pub mod einsum;
pub mod exp;
pub mod instancenorm;
pub mod llama;
pub mod maxpool;
pub mod pad;
pub mod permute;
pub mod pointpillar;
pub mod range;
pub mod reducer;
pub mod relu;
pub mod scale;
pub mod shape;
pub mod subsample;

pub use add::{Add, Sub};
pub use concat::{ChannelSlice, Concat};
pub use conv::{
    Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D,
    DepthwiseConv2D, FlattenKernel, FlattenKernel3D,
};
pub use einsum::Einsum;
pub use exp::{ExpHelper, TwoPow};
pub use instancenorm::{InstanceNorm3D, InstanceNormHelper};
pub use llama::{DivConst, RMSReciprocal, SigmoidConst, SoftmaxConst};
pub use maxpool::{GeneralMaxPoolHelper, MaxPoolHelper, Replicate2x2};
pub use pad::{ZeroPad, ZeroPad3D, ZeroPadAsym};
pub use permute::Permute;
pub use pointpillar::{GatherFromGrid, PillarMaxPool, ScatterToBEV};
pub use range::NonNegative;
pub use reducer::Reducer;
pub use relu::{ProductZeroCheck, ReLUHelper};
pub use scale::{ScaleDown, ScaleUp};
pub use shape::ChangeShape;
pub use subsample::SubSample2D;

use crate::dag::{Claim, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;

pub trait BasicBlock: std::fmt::Debug + Send + Sync {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness>;

    /// GPU-resident path. Default forwards to `run`; individual blocks
    /// override when an AGL CUDA kernel exists. **Do not** introduce a
    /// runtime threshold-based dispatch here — that would silently fall back
    /// to CPU per philosophy rule #7. If a block needs a threshold, the
    /// caller (typically `Dag::run`) makes the choice explicitly.
    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        self.run(inputs)
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>);

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool;
}

/// Heterogeneous op enum. As basicblocks are ported, new variants are added
/// here. Until task #9 finishes, downstream code that pattern-matches on
/// `BasicBlockType` must keep its `match` exhaustive — this enum is the
/// single source of truth for which ops zk-torch-4 currently supports.
#[derive(Clone, Debug)]
pub enum BasicBlockType {
    Add(Add),
    Sub(Sub),
    ChangeShape(ChangeShape),
    ChannelSlice(ChannelSlice),
    Concat(Concat),
    Conv1D(Conv1D),
    Conv2D(Conv2D),
    Conv3D(Conv3D),
    ConvTranspose1D(ConvTranspose1D),
    ConvTranspose2D(ConvTranspose2D),
    ConvTranspose3D(ConvTranspose3D),
    DepthwiseConv2D(DepthwiseConv2D),
    DivConst(DivConst),
    Einsum(Einsum),
    FlattenKernel(FlattenKernel),
    FlattenKernel3D(FlattenKernel3D),
    ExpHelper(ExpHelper),
    GatherFromGrid(GatherFromGrid),
    GeneralMaxPoolHelper(GeneralMaxPoolHelper),
    InstanceNorm3D(InstanceNorm3D),
    MaxPoolHelper(MaxPoolHelper),
    NonNegative(NonNegative),
    Permute(Permute),
    PillarMaxPool(PillarMaxPool),
    ProductZeroCheck(ProductZeroCheck),
    Reducer(Reducer),
    ReLUHelper(ReLUHelper),
    Replicate2x2(Replicate2x2),
    RMSReciprocal(RMSReciprocal),
    ScaleDown(ScaleDown),
    ScaleUp(ScaleUp),
    ScatterToBEV(ScatterToBEV),
    SigmoidConst(SigmoidConst),
    SoftmaxConst(SoftmaxConst),
    SubSample2D(SubSample2D),
    TwoPow(TwoPow),
    ZeroPad(ZeroPad),
    ZeroPad3D(ZeroPad3D),
    ZeroPadAsym(ZeroPadAsym),
}

impl BasicBlockType {
    pub fn out_arity(&self) -> usize {
        match self {
            BasicBlockType::Permute(_) => 2,
            BasicBlockType::ScaleDown(_) => 2,
            BasicBlockType::ScaleUp(_) => 2,
            BasicBlockType::ExpHelper(_) => 2,
            // Y + aux Y_full (full 1D conv, committed advice that binds
            // Y = conv(X, W) via the α-sum + masked-view sumchecks).
            // ConvTranspose* stay at 1: their full polynomial product IS the
            // padded output (junk-free, identity crop map), so the α-sum is
            // bound directly to Y with no aux edge. Junk-free 1×1 stride-1
            // Conv2D likewise binds Y directly (bit-complement crop map).
            BasicBlockType::Conv1D(_) => 2,
            // Grand-product mode adds a third output K (leftover advice) for the
            // VerfCNN-style output binding: Y, Y_full, K.
            BasicBlockType::Conv2D(c) => {
                if c.junk_free() { 1 } else if c.grand_product_mode() { 3 } else { 2 }
            }
            BasicBlockType::Conv3D(_) => 2,
            BasicBlockType::DepthwiseConv2D(_) => 2,
            _ => 1,
        }
    }

    pub fn check_inputs(&self, n: usize) {
        match self {
            BasicBlockType::Add(_) => assert_eq!(n, 2, "Add expects 2 inputs, got {n}"),
            BasicBlockType::Sub(_) => assert_eq!(n, 2, "Sub expects 2 inputs, got {n}"),
            BasicBlockType::Concat(_) => assert_eq!(n, 2, "Concat expects 2 inputs, got {n}"),
            BasicBlockType::Conv1D(_)
            | BasicBlockType::Conv2D(_)
            | BasicBlockType::Conv3D(_)
            | BasicBlockType::ConvTranspose1D(_)
            | BasicBlockType::ConvTranspose2D(_)
            | BasicBlockType::ConvTranspose3D(_)
            | BasicBlockType::DepthwiseConv2D(_) => {
                assert_eq!(n, 2, "Conv* expects 2 inputs (X, W), got {n}")
            }
            BasicBlockType::Einsum(_) => assert!(n >= 1, "Einsum expects at least 1 input, got {n}"),
            BasicBlockType::GatherFromGrid(_) => assert_eq!(n, 2, "GatherFromGrid expects 2 inputs, got {n}"),
            BasicBlockType::InstanceNorm3D(_) => assert_eq!(n, 3, "InstanceNorm3D expects 3 inputs, got {n}"),
            BasicBlockType::ProductZeroCheck(_) => {
                assert_eq!(n, 2, "ProductZeroCheck expects 2 inputs, got {n}")
            }
            BasicBlockType::ScatterToBEV(_) => assert_eq!(n, 2, "ScatterToBEV expects 2 inputs, got {n}"),
            _ => assert_eq!(n, 1, "{:?} expects 1 input, got {n}", self),
        }
    }
}

/// Per-variant dispatch macro: expands to one `match` arm per BasicBlockType
/// variant calling `$method` on the inner block.
macro_rules! dispatch_basicblock {
    ($self:ident, $method:ident($($arg:expr),*)) => {
        match $self {
            BasicBlockType::Add(b)                  => b.$method($($arg),*),
            BasicBlockType::Sub(b)                  => b.$method($($arg),*),
            BasicBlockType::ChangeShape(b)          => b.$method($($arg),*),
            BasicBlockType::ChannelSlice(b)         => b.$method($($arg),*),
            BasicBlockType::Concat(b)               => b.$method($($arg),*),
            BasicBlockType::Conv1D(b)               => b.$method($($arg),*),
            BasicBlockType::Conv2D(b)               => b.$method($($arg),*),
            BasicBlockType::Conv3D(b)               => b.$method($($arg),*),
            BasicBlockType::ConvTranspose1D(b)      => b.$method($($arg),*),
            BasicBlockType::ConvTranspose2D(b)      => b.$method($($arg),*),
            BasicBlockType::ConvTranspose3D(b)      => b.$method($($arg),*),
            BasicBlockType::DepthwiseConv2D(b)      => b.$method($($arg),*),
            BasicBlockType::DivConst(b)             => b.$method($($arg),*),
            BasicBlockType::Einsum(b)               => b.$method($($arg),*),
            BasicBlockType::FlattenKernel(b)        => b.$method($($arg),*),
            BasicBlockType::FlattenKernel3D(b)      => b.$method($($arg),*),
            BasicBlockType::ExpHelper(b)            => b.$method($($arg),*),
            BasicBlockType::GatherFromGrid(b)       => b.$method($($arg),*),
            BasicBlockType::GeneralMaxPoolHelper(b) => b.$method($($arg),*),
            BasicBlockType::InstanceNorm3D(b)       => b.$method($($arg),*),
            BasicBlockType::MaxPoolHelper(b)        => b.$method($($arg),*),
            BasicBlockType::NonNegative(b)          => b.$method($($arg),*),
            BasicBlockType::Permute(b)              => b.$method($($arg),*),
            BasicBlockType::PillarMaxPool(b)        => b.$method($($arg),*),
            BasicBlockType::ProductZeroCheck(b)     => b.$method($($arg),*),
            BasicBlockType::Reducer(b)              => b.$method($($arg),*),
            BasicBlockType::ReLUHelper(b)           => b.$method($($arg),*),
            BasicBlockType::Replicate2x2(b)         => b.$method($($arg),*),
            BasicBlockType::RMSReciprocal(b)        => b.$method($($arg),*),
            BasicBlockType::ScaleDown(b)            => b.$method($($arg),*),
            BasicBlockType::ScaleUp(b)              => b.$method($($arg),*),
            BasicBlockType::ScatterToBEV(b)         => b.$method($($arg),*),
            BasicBlockType::SigmoidConst(b)         => b.$method($($arg),*),
            BasicBlockType::SoftmaxConst(b)         => b.$method($($arg),*),
            BasicBlockType::SubSample2D(b)          => b.$method($($arg),*),
            BasicBlockType::TwoPow(b)               => b.$method($($arg),*),
            BasicBlockType::ZeroPad(b)              => b.$method($($arg),*),
            BasicBlockType::ZeroPad3D(b)            => b.$method($($arg),*),
            BasicBlockType::ZeroPadAsym(b)          => b.$method($($arg),*),
        }
    };
}

impl BasicBlock for BasicBlockType {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        dispatch_basicblock!(self, run(inputs))
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        dispatch_basicblock!(self, run_gpu(inputs))
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        dispatch_basicblock!(self, prove(witnesses, edge_ids, out_claims, transcript))
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        dispatch_basicblock!(self, verify(witnesses, claims, sumcheck_proofs, transcript))
    }
}
