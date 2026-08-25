//! Fold-tree verifier (§7). Replays the per-level same-point + γ-multifold
//! transcript, derives the chunk commitments homomorphically when needed,
//! and finally checks `c* = M · f*` + `f*(R*) = y*`. Implementation lands
//! in task #34.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FoldTreeError {
    LevelMismatch { level: usize, reason: String },
    SamePointFailed { level: usize, group: usize },
    MultifoldFailed { level: usize, group: usize },
    SplitFailed { level: usize, group: usize },
    FinalCommitmentMismatch,
    FinalEvaluationMismatch,
}
