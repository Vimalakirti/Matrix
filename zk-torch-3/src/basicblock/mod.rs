pub mod add;
pub mod concat;
pub mod conv;
pub mod einsum;
pub mod exp;
pub mod instancenorm;
pub mod maxpool;
pub mod pad;
pub mod permute;
pub mod range;
pub mod relu;
pub mod reducer;
pub mod scale;
pub mod shape;
pub mod subsample;
pub mod llama;
pub mod pointpillar;

pub use add::{Add, Sub};
pub use concat::{Concat, ChannelSlice};
pub use conv::{Conv1D, Conv2D, Conv3D, ConvTranspose1D, ConvTranspose2D, ConvTranspose3D, DepthwiseConv2D, FlattenKernel, FlattenKernel3D};
pub use einsum::Einsum;
pub use exp::{ExpHelper, TwoPow};
pub use instancenorm::{InstanceNormHelper, InstanceNorm3D};
pub use maxpool::{GeneralMaxPoolHelper, MaxPoolHelper, Replicate2x2};
pub use pad::{ZeroPad, ZeroPadAsym, ZeroPad3D};
pub use permute::Permute;
pub use range::NonNegative;
pub use relu::{ReLUHelper, ProductZeroCheck};
pub use reducer::Reducer;
pub use scale::{ScaleDown, ScaleUp};
pub use shape::ChangeShape;
pub use subsample::SubSample2D;
pub use llama::{DivConst, RMSReciprocal, SigmoidConst, SoftmaxConst};
pub use pointpillar::{PillarMaxPool, ScatterToBEV, GatherFromGrid};


use crate::dag::{Claim, Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;

/// The basic block trait — core interface for ZKML operations.
pub trait BasicBlock: std::fmt::Debug + Send + Sync {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness>;

    /// GPU-resident witness generation. Default impl falls back to CPU `run`,
    /// which transparently triggers any on-demand device→host download via
    /// `DeviceDenseMLPoly`'s lazy cache. Override per op as kernels land.
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

/// Heterogeneous op enum — all block types.
#[derive(Clone, Debug)]
pub enum BasicBlockType {
    Add(Add),
    Sub(Sub),
    Conv1D(Conv1D),
    Conv2D(Conv2D),
    Conv3D(Conv3D),
    ConvTranspose1D(ConvTranspose1D),
    ConvTranspose2D(ConvTranspose2D),
    ConvTranspose3D(ConvTranspose3D),
    Einsum(Einsum),
    ExpHelper(ExpHelper),
    FlattenKernel(FlattenKernel),
    FlattenKernel3D(FlattenKernel3D),
    MaxPoolHelper(MaxPoolHelper),
    Replicate2x2(Replicate2x2),
    TwoPow(TwoPow),
    ZeroPad(ZeroPad),
    ChangeShape(ChangeShape),
    ScaleDown(ScaleDown),
    ScaleUp(ScaleUp),
    NonNegative(NonNegative),
    Reducer(Reducer),
    Permute(Permute),
    RMSReciprocal(RMSReciprocal),
    DivConst(DivConst),
    SoftmaxConst(SoftmaxConst),
    SigmoidConst(SigmoidConst),
    ReLUHelper(ReLUHelper),
    SubSample2D(SubSample2D),
    ZeroPadAsym(ZeroPadAsym),
    GeneralMaxPoolHelper(GeneralMaxPoolHelper),
    Concat(Concat),
    ChannelSlice(ChannelSlice),
    DepthwiseConv2D(DepthwiseConv2D),
    InstanceNorm3D(InstanceNorm3D),
    ZeroPad3D(ZeroPad3D),
    PillarMaxPool(PillarMaxPool),
    ScatterToBEV(ScatterToBEV),
    GatherFromGrid(GatherFromGrid),
    ProductZeroCheck(ProductZeroCheck),
}

impl BasicBlockType {
    pub fn out_arity(&self) -> usize {
        match self {
            BasicBlockType::Permute(_) => 2,
            BasicBlockType::ScaleDown(_) => 2,
            BasicBlockType::ScaleUp(_) => 2,
            BasicBlockType::ExpHelper(_) => 2,
            _ => 1,
        }
    }

    pub fn check_inputs(&self, n: usize) {
        match self {
            BasicBlockType::Add(_) => assert!(n == 2, "Add expects 2 inputs, got {n}"),
            BasicBlockType::Sub(_) => assert!(n == 2, "Sub expects 2 inputs, got {n}"),
            BasicBlockType::Conv1D(_) => assert!(n == 2, "Conv1D expects 2 inputs, got {n}"),
            BasicBlockType::Conv2D(_) => assert!(n == 2, "Conv2D expects 2 inputs, got {n}"),
            BasicBlockType::DepthwiseConv2D(_) => assert!(n == 2, "DepthwiseConv2D expects 2 inputs, got {n}"),
            BasicBlockType::Conv3D(_) => assert!(n == 2, "Conv3D expects 2 inputs, got {n}"),
            BasicBlockType::ConvTranspose1D(_) => assert!(n == 2, "ConvTranspose1D expects 2 inputs, got {n}"),
            BasicBlockType::ConvTranspose2D(_) => assert!(n == 2, "ConvTranspose2D expects 2 inputs, got {n}"),
            BasicBlockType::ConvTranspose3D(_) => assert!(n == 2, "ConvTranspose3D expects 2 inputs, got {n}"),
            BasicBlockType::Einsum(_) => assert!(n >= 1, "Einsum expects at least 1 input, got {n}"),
            BasicBlockType::Concat(_) => assert!(n == 2, "Concat expects 2 inputs, got {n}"),
            BasicBlockType::InstanceNorm3D(_) => assert!(n == 3, "InstanceNorm3D expects 3 inputs, got {n}"),
            BasicBlockType::ScatterToBEV(_) => assert!(n == 2, "ScatterToBEV expects 2 inputs, got {n}"),
            BasicBlockType::GatherFromGrid(_) => assert!(n == 2, "GatherFromGrid expects 2 inputs, got {n}"),
            BasicBlockType::ProductZeroCheck(_) => assert!(n == 2, "ProductZeroCheck expects 2 inputs, got {n}"),
            _ => assert!(n == 1, "Unary op expects 1 input, got {n} ({:?})", self),
        }
    }
}

impl BasicBlock for BasicBlockType {
    fn run(&self, inputs: &[&Witness]) -> Vec<Witness> {
        match self {
            BasicBlockType::Add(b) => b.run(inputs),
            BasicBlockType::Sub(b) => b.run(inputs),
            BasicBlockType::Conv1D(b) => b.run(inputs),
            BasicBlockType::Conv2D(b) => b.run(inputs),
            BasicBlockType::Conv3D(b) => b.run(inputs),
            BasicBlockType::ConvTranspose1D(b) => b.run(inputs),
            BasicBlockType::ConvTranspose2D(b) => b.run(inputs),
            BasicBlockType::ConvTranspose3D(b) => b.run(inputs),
            BasicBlockType::Einsum(b) => b.run(inputs),
            BasicBlockType::ExpHelper(b) => b.run(inputs),
            BasicBlockType::FlattenKernel(b) => b.run(inputs),
            BasicBlockType::FlattenKernel3D(b) => b.run(inputs),
            BasicBlockType::MaxPoolHelper(b) => b.run(inputs),
            BasicBlockType::Replicate2x2(b) => b.run(inputs),
            BasicBlockType::TwoPow(b) => b.run(inputs),
            BasicBlockType::ZeroPad(b) => b.run(inputs),
            BasicBlockType::ChangeShape(b) => b.run(inputs),
            BasicBlockType::ScaleDown(b) => b.run(inputs),
            BasicBlockType::ScaleUp(b) => b.run(inputs),
            BasicBlockType::NonNegative(b) => b.run(inputs),
            BasicBlockType::Permute(b) => b.run(inputs),
            BasicBlockType::Reducer(b) => b.run(inputs),
            BasicBlockType::RMSReciprocal(b) => b.run(inputs),
            BasicBlockType::DivConst(b) => b.run(inputs),
            BasicBlockType::SoftmaxConst(b) => b.run(inputs),
            BasicBlockType::SigmoidConst(b) => b.run(inputs),
            BasicBlockType::ReLUHelper(b) => b.run(inputs),
            BasicBlockType::SubSample2D(b) => b.run(inputs),
            BasicBlockType::ZeroPadAsym(b) => b.run(inputs),
            BasicBlockType::GeneralMaxPoolHelper(b) => b.run(inputs),
            BasicBlockType::Concat(b) => b.run(inputs),
            BasicBlockType::ChannelSlice(b) => b.run(inputs),
            BasicBlockType::DepthwiseConv2D(b) => b.run(inputs),
            BasicBlockType::InstanceNorm3D(b) => b.run(inputs),
            BasicBlockType::ZeroPad3D(b) => b.run(inputs),
            BasicBlockType::PillarMaxPool(b) => b.run(inputs),
            BasicBlockType::ScatterToBEV(b) => b.run(inputs),
            BasicBlockType::GatherFromGrid(b) => b.run(inputs),
            BasicBlockType::ProductZeroCheck(b) => b.run(inputs),
        }
    }

    fn run_gpu(&self, inputs: &[&Witness]) -> Vec<Witness> {
        match self {
            BasicBlockType::Add(b) => b.run_gpu(inputs),
            BasicBlockType::Sub(b) => b.run_gpu(inputs),
            BasicBlockType::Conv1D(b) => b.run_gpu(inputs),
            BasicBlockType::Conv2D(b) => b.run_gpu(inputs),
            BasicBlockType::Conv3D(b) => b.run_gpu(inputs),
            BasicBlockType::ConvTranspose1D(b) => b.run_gpu(inputs),
            BasicBlockType::ConvTranspose2D(b) => b.run_gpu(inputs),
            BasicBlockType::ConvTranspose3D(b) => b.run_gpu(inputs),
            BasicBlockType::Einsum(b) => b.run_gpu(inputs),
            BasicBlockType::ExpHelper(b) => b.run_gpu(inputs),
            BasicBlockType::FlattenKernel(b) => b.run_gpu(inputs),
            BasicBlockType::FlattenKernel3D(b) => b.run_gpu(inputs),
            BasicBlockType::MaxPoolHelper(b) => b.run_gpu(inputs),
            BasicBlockType::Replicate2x2(b) => b.run_gpu(inputs),
            BasicBlockType::TwoPow(b) => b.run_gpu(inputs),
            BasicBlockType::ZeroPad(b) => b.run_gpu(inputs),
            BasicBlockType::ChangeShape(b) => b.run_gpu(inputs),
            BasicBlockType::ScaleDown(b) => b.run_gpu(inputs),
            BasicBlockType::ScaleUp(b) => b.run_gpu(inputs),
            BasicBlockType::NonNegative(b) => b.run_gpu(inputs),
            BasicBlockType::Permute(b) => b.run_gpu(inputs),
            BasicBlockType::Reducer(b) => b.run_gpu(inputs),
            BasicBlockType::RMSReciprocal(b) => b.run_gpu(inputs),
            BasicBlockType::DivConst(b) => b.run_gpu(inputs),
            BasicBlockType::SoftmaxConst(b) => b.run_gpu(inputs),
            BasicBlockType::SigmoidConst(b) => b.run_gpu(inputs),
            BasicBlockType::ReLUHelper(b) => b.run_gpu(inputs),
            BasicBlockType::SubSample2D(b) => b.run_gpu(inputs),
            BasicBlockType::ZeroPadAsym(b) => b.run_gpu(inputs),
            BasicBlockType::GeneralMaxPoolHelper(b) => b.run_gpu(inputs),
            BasicBlockType::Concat(b) => b.run_gpu(inputs),
            BasicBlockType::ChannelSlice(b) => b.run_gpu(inputs),
            BasicBlockType::DepthwiseConv2D(b) => b.run_gpu(inputs),
            BasicBlockType::InstanceNorm3D(b) => b.run_gpu(inputs),
            BasicBlockType::ZeroPad3D(b) => b.run_gpu(inputs),
            BasicBlockType::PillarMaxPool(b) => b.run_gpu(inputs),
            BasicBlockType::ScatterToBEV(b) => b.run_gpu(inputs),
            BasicBlockType::GatherFromGrid(b) => b.run_gpu(inputs),
            BasicBlockType::ProductZeroCheck(b) => b.run_gpu(inputs),
        }
    }

    fn prove(
        &self,
        witnesses: &[&Witness],
        edge_ids: &[usize],
        out_claims: &[&Claim],
        transcript: &mut Transcript,
    ) -> (Vec<SumcheckProof>, Vec<Claim>) {
        match self {
            BasicBlockType::Add(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Sub(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Conv1D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Conv2D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Conv3D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ConvTranspose1D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ConvTranspose2D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ConvTranspose3D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Einsum(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ExpHelper(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::FlattenKernel(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::FlattenKernel3D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::MaxPoolHelper(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Replicate2x2(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::TwoPow(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ZeroPad(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ChangeShape(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ScaleDown(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ScaleUp(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::NonNegative(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Permute(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Reducer(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::RMSReciprocal(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::DivConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::SoftmaxConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::SigmoidConst(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ReLUHelper(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::SubSample2D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ZeroPadAsym(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::GeneralMaxPoolHelper(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::Concat(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ChannelSlice(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::DepthwiseConv2D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::InstanceNorm3D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ZeroPad3D(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::PillarMaxPool(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ScatterToBEV(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::GatherFromGrid(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
            BasicBlockType::ProductZeroCheck(b) => b.prove(witnesses, edge_ids, out_claims, transcript),
        }
    }

    fn verify(
        &self,
        witnesses: &[&Witness],
        claims: &[&Claim],
        sumcheck_proofs: &[&SumcheckProof],
        transcript: &mut Transcript,
    ) -> bool {
        match self {
            BasicBlockType::Add(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Sub(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Conv1D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Conv2D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Conv3D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ConvTranspose1D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ConvTranspose2D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ConvTranspose3D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Einsum(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ExpHelper(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::FlattenKernel(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::FlattenKernel3D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::MaxPoolHelper(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Replicate2x2(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::TwoPow(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ZeroPad(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ChangeShape(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ScaleDown(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ScaleUp(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::NonNegative(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Permute(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Reducer(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::RMSReciprocal(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::DivConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::SoftmaxConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::SigmoidConst(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ReLUHelper(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::SubSample2D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ZeroPadAsym(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::GeneralMaxPoolHelper(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::Concat(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ChannelSlice(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::DepthwiseConv2D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::InstanceNorm3D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ZeroPad3D(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::PillarMaxPool(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ScatterToBEV(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::GatherFromGrid(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
            BasicBlockType::ProductZeroCheck(b) => b.verify(witnesses, claims, sumcheck_proofs, transcript),
        }
    }
}
