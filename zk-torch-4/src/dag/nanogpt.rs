//! nanoGPT exactly as EZKL ships it in `examples/onnx/nanoGPT`.
//!
//! Config read from that directory's `gen.py` and confirmed against the
//! exported graph:
//!
//!   block_size 64, vocab 65, n_layer 4, n_head 4, n_embd 64, bias FALSE
//!   input  [1, 64] token ids
//!   output [1, 64, 65] logits, for every position
//!
//! Structurally this is GPT-2, so the blocks reuse `gpt2_block` rather than
//! restating attention. Three differences from the GPT-2 rows here, all read
//! off the ONNX rather than assumed:
//!
//! `bias = False` everywhere. The exported graph has `ln_1.weight [64]` with no
//! matching bias tensor and its MatMuls take no Add. Passing an all-zero bias
//! is arithmetically identical to having none, so the shared block is reused
//! rather than forked.
//!
//! `c_attn` is FUSED as one `[64, 192]` matrix that the graph then `Split`s
//! into Q, K, V. Splitting it here at build time produces the same three
//! `[64, 64]` projections `gpt2_block` expects.
//!
//! `lm_head` is TIED to `wte`: `gen.py` assigns
//! `self.transformer.wte.weight = self.lm_head.weight`. The same committed
//! tensor therefore backs both the embedding lookup and the output projection,
//! which is also why it is one deferred weight rather than two.
//!
//! One DEVIATION, which the paper should state. nanoGPT's activation is the
//! tanh approximation to GELU; `gpt2_mlp` here uses the sigmoid approximation
//! `x * sigmoid(1.702x)`. The two differ by well under a percent over the range
//! these activations take, and the prover cost is identical because both are
//! one lookup-backed nonlinearity per element, but it is not bit-exact against
//! EZKL's graph.

use crate::dag::gpt2::gpt2_block;
use crate::dag::{DagBuilder, EdgeId, Role, Witness};

/// EZKL's nanoGPT hyperparameters.
pub const NANOGPT_BLOCK_SIZE: usize = 64;
pub const NANOGPT_VOCAB: usize = 65;
pub const NANOGPT_N_LAYER: usize = 4;
pub const NANOGPT_N_HEAD: usize = 4;
pub const NANOGPT_N_EMBD: usize = 64;
pub const NANOGPT_HEAD_DIM: usize = NANOGPT_N_EMBD / NANOGPT_N_HEAD;

/// Per-block weights, in the order `gpt2_block` consumes them. `bias = False`
/// in EZKL's config, so every `*_b` here is expected to be an all-zero tensor.
pub struct NanoGptBlockWeights {
    pub ln1_w: Witness,
    pub q_w: Witness,
    pub k_w: Witness,
    pub v_w: Witness,
    pub o_w: Witness,
    pub ln1_b: Witness,
    pub q_b: Witness,
    pub k_b: Witness,
    pub v_b: Witness,
    pub o_b: Witness,
    pub ln2_w: Witness,
    pub fc_w: Witness,
    pub proj_w: Witness,
    pub ln2_b: Witness,
    pub fc_b: Witness,
    pub proj_b: Witness,
}

/// nanoGPT forward pass: token ids -> logits for every position.
///
/// `token_ids` seeds the one-hot embedding selector; it is `Role::Input`, so it
/// is committed and opened per proof and is never deferred as a shared weight.
/// `wte` is `Role::Constant` and backs both the embedding and the tied output
/// projection.
pub fn nanogpt(
    wte: Witness,
    wpe: Witness,
    blocks: Vec<NanoGptBlockWeights>,
    ln_f_w: Witness,
    ln_f_b: Witness,
    seq_len: usize,
    token_ids: Vec<usize>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, _x| {
        assert_eq!(blocks.len(), NANOGPT_N_LAYER, "nanoGPT has 4 blocks");

        // Token embedding, then the learned positional embedding added in.
        // wpe is a plain [seq, n_embd] constant, so it is an add, not a lookup.
        let wte_e = g.param(wte);
        let (mut h, _sel) =
            g.embedding_lookup(wte_e, seq_len, NANOGPT_VOCAB, &token_ids, Role::Input);
        let wpe_e = g.param(wpe);
        h = g.add(h, wpe_e)[0];
        // gpt2_block works in [batch, seq, hidden]; the embedding returns
        // [seq, hidden].
        h = g.change_shape(h, vec![1, seq_len, NANOGPT_N_EMBD]);

        for b in blocks {
            h = g.pipe(
                &[h],
                gpt2_block(
                    b.ln1_w, b.q_w, b.k_w, b.v_w, b.o_w,
                    b.ln1_b, b.q_b, b.k_b, b.v_b, b.o_b,
                    b.ln2_w, b.fc_w, b.proj_w,
                    b.ln2_b, b.fc_b, b.proj_b,
                    NANOGPT_N_HEAD, NANOGPT_HEAD_DIM, seq_len,
                ),
            )[0];
            g.layer_boundaries.push(h);
        }

        // Final LayerNorm, then the tied output projection. wte is [vocab,
        // n_embd] and the head contracts over n_embd, so the equation states
        // that directly instead of transposing the shared tensor.
        let ln_f_w_e = g.param(ln_f_w);
        let ln_f_b_e = g.param(ln_f_b);
        h = g.pipe(&[h], crate::dag::gpt2::gpt2_layer_norm(ln_f_w_e, ln_f_b_e))[0];
        let logits = g.einsum("bsi,vi->bsv".to_string(), vec![h, wte_e], false)[0];
        vec![logits]
    }
}
