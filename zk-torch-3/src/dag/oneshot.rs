//! One-shot autoregressive proving primitives (zkAgent §4.1 style).
//!
//! A causal-masked transformer over a length-K sequence already produces K
//! logit vectors in one forward pass. If the input sequence is autoregressive
//! (`t_{i+1} = argmax(logits[i])`), proving K next-token predictions reduces
//! to one circuit run plus K argmax constraints, dwarfing the cost of K
//! independent transformer proofs.
//!
//! In-circuit, we prove `next_token_ids[i] == argmax(logits[i,:])` for every
//! position. The *public* shift constraint
//!   `token_ids[i+1] == next_token_ids[i]   for i ∈ [prompt_len-1, seq_len-1)`
//! is checked by the verifier off-circuit, on the public selector parameters.
//! Together these enforce `argmax(logits[i,:]) == token_ids[i+1]` at every
//! generated position — exactly the autoregressive recurrence.
//!
//! Both selectors `S` (embedding) and `S'` (argmax) are param edges so the
//! driver can overwrite their witnesses between forward passes during the
//! AR generation loop without rebuilding the DAG.

use goldilocks_cuda::GoldilocksField;

use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::arith::f_to_int;

impl DagBuilder {
    /// One-hot selector matrix `S[i, indices[i]] = 1`, others 0.
    /// Shape `(len, table_size)`. MLE col-major flat index: `(i, v) → i + v * len_pad`.
    pub fn build_one_hot_selector(
        len: usize,
        table_size: usize,
        indices: &[usize],
    ) -> Witness {
        assert_eq!(indices.len(), len);
        let len_pad = len.next_power_of_two();
        let table_pad = table_size.next_power_of_two();
        let total = len_pad * table_pad;
        let mut data = vec![GoldilocksField(0); total];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < table_size, "index {} out of range [0, {})", idx, table_size);
            data[i + idx * len_pad] = GoldilocksField(1);
        }
        Witness::new(
            vec![len, table_size],
            data,
            DataType::Float,
            0,
            Role::Constant,
        )
    }

    /// Embedding lookup `H_0[i,:] = W_E[token_ids[i], :]` proven via one-hot einsum.
    /// `w_e` has shape `(vocab_size, hidden_dim)`, sf = SF_LOG.
    /// Returns `(h0_edge, selector_edge)`. `selector_edge` is exposed so the
    /// driver can overwrite it during AR witness generation.
    ///
    /// Output `H_0` shape: `(seq_len, hidden_dim)`, sf = SF_LOG.
    pub fn embedding_lookup(
        &mut self,
        w_e: EdgeId,
        seq_len: usize,
        vocab_size: usize,
        token_ids: &[usize],
    ) -> (EdgeId, EdgeId) {
        let s = Self::build_one_hot_selector(seq_len, vocab_size, token_ids);
        let s_id = self.param(s);
        // S has sf=0, W_E has sf=SF_LOG. scale_back=false keeps output sf=SF_LOG.
        let h0 = self.einsum("sv,vd->sd".to_string(), vec![s_id, w_e], false)[0];
        (h0, s_id)
    }

    /// Add learned positional embeddings (GPT-2 style absolute PE).
    /// Both inputs have shape `(seq_len, hidden_dim)`, sf = SF_LOG.
    pub fn add_positional_encoding(&mut self, h0: EdgeId, pos_embed: Witness) -> EdgeId {
        let p = self.param(pos_embed);
        self.add(h0, p)[0]
    }

    /// Weight-tied LM head: `logits = einsum("bsd,vd->bsv", hidden, W_E)`.
    /// `hidden`: `(1, seq_len, hidden_dim)`, `w_e`: `(vocab_size, hidden_dim)`.
    /// Output: `(seq_len, vocab_size)` (after change_shape).
    pub fn lm_head_weight_tied(
        &mut self,
        hidden: EdgeId,
        w_e: EdgeId,
        seq_len: usize,
        vocab_size: usize,
    ) -> EdgeId {
        let logits = self.einsum("bsd,vd->bsv".to_string(), vec![hidden, w_e], true)[0];
        self.change_shape(logits, vec![seq_len, vocab_size])
    }

    /// Separate-weight LM head matching the existing zk-torch-3 convention:
    /// `lm_head_w` shape `(hidden_dim, vocab_size)`, einsum `"bsi,ij->bsj"`.
    /// Output: `(seq_len, vocab_size)`.
    pub fn lm_head(
        &mut self,
        hidden: EdgeId,
        lm_head_w: EdgeId,
        seq_len: usize,
        vocab_size: usize,
    ) -> EdgeId {
        let logits = self
            .einsum("bsi,ij->bsj".to_string(), vec![hidden, lm_head_w], true)[0];
        self.change_shape(logits, vec![seq_len, vocab_size])
    }

    /// If `logits` is already shape `(seq_len, vocab_size)` (e.g. from an
    /// existing model that flattened batch into 1), this is the no-op variant.
    pub fn argmax_check(
        &mut self,
        logits: EdgeId,
        seq_len: usize,
        vocab_size: usize,
        token_ids: &[usize],
    ) -> EdgeId {
        let s = Self::build_one_hot_selector(seq_len, vocab_size, token_ids);
        let s_id = self.param(s);

        // selected[i] = logits[i, token_ids[i]]
        // logits has sf=SF_LOG, S' has sf=0, scale_back=false → output sf=SF_LOG.
        let selected = self
            .einsum("sv,sv->s".to_string(), vec![logits, s_id], false)[0];

        // diffs[i, j] = selected[i] - logits[i, j]   (broadcast over j)
        let selected_broad = self.change_shape(selected, vec![seq_len, 1]);
        let diffs = self.sub(selected_broad, logits)[0];

        // diffs >= 0 ⇒ selected[i] is ≥ every logits[i, j] ⇒ argmax.
        self.add_nonneg_node(diffs);

        s_id
    }
}

/// Read `argmax(logits[pos, :])` from a forward-pass-populated witness array.
/// Assumes `logits` has shape `(seq_len, vocab_size)` and is dense.
/// Field elements are interpreted as signed integers (canonical
/// SF-quantized representation).
pub fn extract_argmax_at(
    witnesses: &[Vec<Witness>],
    logits_edge: EdgeId,
    pos: usize,
    seq_len: usize,
    vocab_size: usize,
) -> usize {
    let logits_w = &witnesses[logits_edge][0];
    let evals = logits_w
        .data
        .as_ref()
        .expect("logits must have data after forward pass")
        .evaluations_ref();
    let seq_pad = seq_len.next_power_of_two();
    let mut best_v = 0usize;
    let mut best_val = i128::MIN;
    for v in 0..vocab_size {
        let val = f_to_int(evals[pos + v * seq_pad]);
        if val > best_val {
            best_val = val;
            best_v = v;
        }
    }
    best_v
}

/// Read argmax for every row `i ∈ [0, seq_len)`.
pub fn extract_argmax_all(
    witnesses: &[Vec<Witness>],
    logits_edge: EdgeId,
    seq_len: usize,
    vocab_size: usize,
) -> Vec<usize> {
    (0..seq_len)
        .map(|i| extract_argmax_at(witnesses, logits_edge, i, seq_len, vocab_size))
        .collect()
}
