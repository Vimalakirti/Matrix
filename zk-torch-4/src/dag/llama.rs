use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use ndarray::Array2;
use std::f64::consts::PI;

use crate::basicblock::BasicBlockType;
use crate::basicblock::{RMSReciprocal, SoftmaxConst};
use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::SF_FLOAT;
use crate::SF_LOG;

pub fn pair_swap_perm_matrix(d: usize) -> Vec<AlmostGoldilocksField> {
    assert!(d % 2 == 0, "dimension d must be even");

    let mut m = Array2::from_elem((d, d), AlmostGoldilocksField(0));

    for i in (0..d).step_by(2) {
        m[[i, i + 1]] = AlmostGoldilocksField(1);
        m[[i + 1, i]] = AlmostGoldilocksField(1);
    }

    m.into_dyn()
        .view()
        .reversed_axes()
        .iter()
        .copied()
        .collect::<Vec<AlmostGoldilocksField>>()
}

/// Generate the cosine vector for RoPE.
pub fn rope_cos_vec(d: usize, m: f64, base: f64) -> Vec<AlmostGoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let mut v = Vec::with_capacity(d);

    for i in 0..half {
        let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
        let angle = m * theta_i;
        let c = angle.cos();
        let c = (c * *SF_FLOAT as f64).round() as i64;
        let c = if c > 0 {
            AlmostGoldilocksField(c as u64)
        } else {
            AlmostGoldilocksField(0) - AlmostGoldilocksField((-c) as u64)
        };
        v.push(c);
        v.push(c);
    }

    v
}

/// Generate the sine vector for RoPE.
pub fn rope_sin_vec(d: usize, m: f64, base: f64) -> Vec<AlmostGoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let mut raw = Vec::with_capacity(d);

    for i in 0..half {
        let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
        let angle = m * theta_i;
        let s = angle.sin();
        raw.push((s * *SF_FLOAT as f64).round() as i64);
        raw.push((-s * *SF_FLOAT as f64).round() as i64);
    }

    raw.iter()
        .map(|f| {
            if *f > 0 {
                AlmostGoldilocksField(*f as u64)
            } else {
                AlmostGoldilocksField(0) - AlmostGoldilocksField((-*f) as u64)
            }
        })
        .collect()
}

impl DagBuilder {
    pub fn rms_reciprocal(&mut self, x: EdgeId) -> Vec<EdgeId> {
        let inp = self.init_values[x].as_ref().unwrap();
        let dim = inp.shape[inp.shape.len() - 1];
        let rms_reciprocal_basicblock =
            BasicBlockType::RMSReciprocal(RMSReciprocal { dim });

        if self.init_values[x].is_some() {
            let mut shape = self.init_values[x].as_ref().unwrap().shape.clone();
            let shape_len = shape.len();
            let data_type = self.init_values[x].as_ref().unwrap().data_type;
            let sf = self.init_values[x].as_ref().unwrap().sf;
            shape[shape_len - 1] = 1;
            let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Auxiliary);
            self.init_values.push(Some(out_value));
        } else {
            self.init_values.push(None);
        }

        self.add_gkr_node(vec![x], rms_reciprocal_basicblock)
    }

    // `div_const` lives in `builder.rs` (already promoted during the
    // step-4 builder port). Keeping it out of `llama.rs` here to avoid
    // duplicate-impl errors.

    pub fn softmax_const(&mut self, a: EdgeId) -> Vec<EdgeId> {
        let inp = self.init_values[a].as_ref().unwrap();
        let dim = inp.shape[inp.shape.len() - 1];
        let softmax_const_basicblock =
            BasicBlockType::SoftmaxConst(SoftmaxConst { dim });
        let shape = inp.shape.clone();
        let sf = inp.sf;
        let data_type = inp.data_type;
        let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
        self.init_values.push(Some(out_value));
        self.add_gkr_node(vec![a], softmax_const_basicblock)
    }

    pub fn rope(&mut self, a: EdgeId, pos: usize) -> Vec<EdgeId> {
        assert!(self.init_values[a].is_some(), "Input must be initialized");
        let inp_value = self.init_values[a].as_ref().unwrap();
        let d = inp_value.shape[inp_value.shape.len() - 1];
        let perm_matrix = pair_swap_perm_matrix(d);
        let perm_matrix = Witness::new(
            vec![d, d],
            perm_matrix,
            DataType::Float,
            0,
            Role::Constant,
        );
        let perm_matrix = self.param(perm_matrix);
        let sin_branch =
            self.einsum("bshd,de->bshe".to_string(), vec![a, perm_matrix], false)[0];
        let cos_branch = a;

        let sin_param = rope_sin_vec(d, pos as f64, 10000.0);
        let cos_param = rope_cos_vec(d, pos as f64, 10000.0);
        let sin_param = Witness::new(
            vec![1, d],
            sin_param,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        );
        let cos_param = Witness::new(
            vec![1, d],
            cos_param,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        );
        let sin_param = self.param(sin_param);
        let cos_param = self.param(cos_param);
        let sin_branch = self.einsum(
            "bshd,sd->bshd".to_string(),
            vec![sin_branch, sin_param],
            true,
        )[0];
        let cos_branch = self.einsum(
            "bshd,sd->bshd".to_string(),
            vec![cos_branch, cos_param],
            true,
        )[0];
        let out = self.add(sin_branch, cos_branch)[0];
        vec![out]
    }

    /// RoPE with precomputed cos/sin edge IDs.
    /// cos_param and sin_param should have shape [seq_len, d] (matching the input's seq dim).
    pub fn rope_with_vecs(&mut self, a: EdgeId, cos_param: EdgeId, sin_param: EdgeId) -> Vec<EdgeId> {
        assert!(self.init_values[a].is_some(), "Input must be initialized");
        let inp_value = self.init_values[a].as_ref().unwrap();
        let d = inp_value.shape[inp_value.shape.len() - 1];
        let perm_matrix = pair_swap_perm_matrix(d);
        let perm_matrix = Witness::new(
            vec![d, d],
            perm_matrix,
            DataType::Float,
            0,
            Role::Constant,
        );
        let perm_matrix = self.param(perm_matrix);
        let sin_branch =
            self.einsum("bshd,de->bshe".to_string(), vec![a, perm_matrix], false)[0];
        let cos_branch = a;

        let sin_branch = self.einsum(
            "bshd,sd->bshd".to_string(),
            vec![sin_branch, sin_param],
            true,
        )[0];
        let cos_branch = self.einsum(
            "bshd,sd->bshd".to_string(),
            vec![cos_branch, cos_param],
            true,
        )[0];
        let out = self.add(sin_branch, cos_branch)[0];
        vec![out]
    }
}

pub fn llama_rms_norm(
    w_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom RMSNorm layer expects 1 input");
        let x = x[0];
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let seq = x_shape[1];
        let n = x_shape[x_shape.len() - 1];
        let r = g.rms_reciprocal(x)[0];

        // Prove r is correctly computed
        let sf = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(*SF_FLOAT as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let tolerance = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(2)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![x, x], true)[0];
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0];
        let x_mean = g.div_const(x_sum, n)[0];
        let x_mean = g.change_shape(x_mean, vec![1, seq, 1]);
        let n_param: usize = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(n as u64)],
            DataType::Float,
            0,
            Role::Constant,
        ));
        let mean_tolerance = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField((n / 2) as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_mean_mul_n =
            g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

        // `x_sum` is [1, seq] but `x_mean_mul_n` is [1, seq, 1]. Subtracting them
        // directly broadcasts to a [1, seq, seq] OUTER PRODUCT whose entry (i, j)
        // is `x_sum[j] - n*x_mean[i]`, a difference between two DIFFERENT rows'
        // sums. Only the diagonal is the intended rounding remainder; off-diagonal
        // entries are unbounded in both signs and break the NonNegative check as
        // soon as row energies differ. A power-of-two padding tail guarantees they
        // differ, since a pad row has zero energy while real rows do not - which
        // is why non-power-of-two sequence lengths (BERT-Large 384, Whisper
        // large-v3 1500/448) failed while power-of-two ones (GPT-2 1024) happened
        // to stay in range. Reshape first so the sub stays element-wise per row.
        // Same fix as dag/bert.rs and dag/whisper.rs::whisper_layer_norm.
        let x_sum_reshaped = g.change_shape(x_sum, vec![1, seq, 1]);
        let x_sum_sub_x_mean_mul_n = g.sub(x_sum_reshaped, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0];
        let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
        let z_sf_diff = g.sub(z, sf)[0];
        // The reciprocity gate |mean(x^2)*r^2 - 1| <= tolerance is only satisfiable
        // where a row carries data. On a zero padding row mean(x^2) = 0, so z = 0
        // and z_sf_diff = -sf: positive_3 becomes -sf + tolerance (-1022 at SF=10,
        // tolerance 2) and positive_4 becomes sf + tolerance (1026) - exactly the
        // negative-and-overflow pair measured on BERT and Whisper. Mask the pad
        // rows so they read as |0| <= tolerance. Nothing is lost: r on a pad row
        // only ever multiplies a zero row, so it is unconstrained by construction.
        // Only mask when the sequence axis is ACTUALLY padded. The gate exists to
        // neutralise all-zero padding rows; when `seq` is already a power of two
        // there are none, and the mask is a semantic no-op that still costs a
        // constant witness plus an einsum per norm. Measured on llama2 8L/seq64
        // (seq already 64): masking unconditionally added 357 fold-tree leaves and
        // cost ~15% prove (36.4-38.3s -> 42.3-43.4s). Models with a non-power-of-two
        // sequence or context length still get it, which is what BERT-Large (384)
        // and Whisper large-v3 (1500/448) need.
        let z_sf_diff = if seq.next_power_of_two() != seq {
            g.mask(z_sf_diff, vec![1, seq, 1])
        } else {
            z_sf_diff
        };
        let positive_3 = g.add(z_sf_diff, tolerance)[0];
        let positive_4 = g.sub(tolerance, z_sf_diff)[0];
        g.add_nonneg_node(positive_3);
        g.add_nonneg_node(positive_4);

        // Compute RMSNorm
        let r = g.change_shape(r, vec![1, seq]);
        let h = g.einsum("bsi,bs->bsi".to_string(), vec![x, r], true)[0];
        let out = g.einsum("bsi,i->bsi".to_string(), vec![h, w_e], true)[0];
        vec![out]
    }
}

/// SwiGLU MLP, sharded along the `ffn` axis for arity control.
///
/// For each shard `s`:
///   gate_s = x @ w_1[:, ffn_s], up_s = x @ w_2[:, ffn_s]
///   swish_s = gate_s · sigmoid(gate_s)
///   partial_s = (swish_s · up_s) @ w_3[ffn_s, :]
///
/// Output = Σ_s partial_s  (elementwise add of `[b, s, hidden]` tensors).
///
/// With N = 16 shards on Llama-2-7B (ffn = 16384), each einsum's
/// wide-poly arity drops from 26 to 22 → GPU same-point + multifold
/// kernels run instead of falling back to CPU. Pass a single-element
/// vec for the un-sharded behavior.
pub fn llama_mlp(
    w_1_shards: Vec<EdgeId>,
    w_2_shards: Vec<EdgeId>,
    w_3_shards: Vec<EdgeId>,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom MLP layer expects 1 input");
        assert_eq!(w_1_shards.len(), w_2_shards.len(), "FFN shard count mismatch");
        assert_eq!(w_1_shards.len(), w_3_shards.len(), "FFN shard count mismatch");
        assert!(!w_1_shards.is_empty(), "Need at least one FFN shard");
        let x = x[0];

        let mut partials: Vec<EdgeId> = Vec::with_capacity(w_1_shards.len());
        for ((w1, w2), w3) in w_1_shards.iter().zip(w_2_shards.iter()).zip(w_3_shards.iter()) {
            let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, *w1], true)[0];
            let sigmoid = g.sigmoid(h_1)[0];
            let swish = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, sigmoid], true)[0];
            let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![x, *w2], true)[0];
            let mul = g.einsum("bsi,bsi->bsi".to_string(), vec![swish, h_2], true)[0];
            let partial = g.einsum("bsi,ij->bsj".to_string(), vec![mul, *w3], true)[0];
            partials.push(partial);
        }

        // Sum partials elementwise. Each Add is `[b, s, hidden]`-sized
        // (arity log2(hidden) ≤ 12 — trivial vs the matmuls).
        let mut acc = partials[0];
        for i in 1..partials.len() {
            acc = g.add(acc, partials[i])[0];
        }
        vec![acc]
    }
}

pub fn llama_attention(
    w_q_e: EdgeId,
    w_k_e: EdgeId,
    w_v_e: EdgeId,
    w_o_e: EdgeId,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    cos_param: EdgeId,
    sin_param: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(
            x.len() == 1,
            "This custom Attention layer expects 1 input"
        );

        let inp = x[0];

        let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0];
        let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0];
        let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0];

        let q = g.reshape(q, vec![1, seq_len, num_heads, head_dim])[0];
        let k = g.reshape(k, vec![1, seq_len, num_heads, head_dim])[0];
        let v = g.reshape(v, vec![1, seq_len, num_heads, head_dim])[0];

        let q = g.rope_with_vecs(q, cos_param, sin_param)[0];
        let k = g.rope_with_vecs(k, cos_param, sin_param)[0];

        let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
        let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
        let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

        let d_sqrt_recip = ((*SF_FLOAT as f64) / ((head_dim as f64).sqrt())).round() as u64;
        let d_sqrt_recip = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(d_sqrt_recip)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let scores =
            g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
        let scores = g.einsum(
            "bhst,z->bhst".to_string(),
            vec![scores, d_sqrt_recip],
            true,
        )[0];
        let scores = if seq_len > 1 { g.causal_mask(scores, seq_len) } else { scores };
        let softmax_c = g.softmax_const(scores)[0];
        let scores = g.add(scores, softmax_c)[0];
        let scores = g.exp(scores)[0];

        let out =
            g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
        let out = g.change_shape(out, vec![1, seq_len, num_heads, head_dim]);
        let out = g.reshape(out, vec![1, seq_len, num_heads * head_dim])[0];

        let out =
            g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0];
        vec![out]
    }
}

pub fn llama_block(
    attn_norm_w: Witness,
    attn_q_w: Witness,
    attn_k_w: Witness,
    attn_v_w: Witness,
    attn_o_w: Witness,
    proj_norm_w: Witness,
    proj_1_w_shards: Vec<Witness>,
    proj_2_w_shards: Vec<Witness>,
    proj_3_w_shards: Vec<Witness>,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    cos_param: EdgeId,
    sin_param: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom Block layer expects 1 input");
        let x = x[0];
        let attn_norm = g.param(attn_norm_w);
        let attn_q = g.param(attn_q_w);
        let attn_k = g.param(attn_k_w);
        let attn_v = g.param(attn_v_w);
        let attn_o = g.param(attn_o_w);
        let proj_norm = g.param(proj_norm_w);
        // Shard FFN matrices: each (gate, up, down) shard becomes its
        // own param edge; `llama_mlp` builds a per-shard pipeline and
        // sums the per-shard `[b, s, hidden]` partial outputs.
        let proj_1_e: Vec<EdgeId> = proj_1_w_shards.into_iter().map(|w| g.param(w)).collect();
        let proj_2_e: Vec<EdgeId> = proj_2_w_shards.into_iter().map(|w| g.param(w)).collect();
        let proj_3_e: Vec<EdgeId> = proj_3_w_shards.into_iter().map(|w| g.param(w)).collect();

        let attn_norm_out = g.pipe(&[x], llama_rms_norm(attn_norm));
        let attn_out = g.pipe(
            &[attn_norm_out[0]],
            llama_attention(attn_q, attn_k, attn_v, attn_o, num_heads, head_dim, seq_len, cos_param, sin_param),
        );
        let residual_attn = g.add(attn_out[0], x)[0];

        let proj_norm_out = g.pipe(&[residual_attn], llama_rms_norm(proj_norm));
        let proj_out = g.pipe(&proj_norm_out, llama_mlp(proj_1_e, proj_2_e, proj_3_e));
        let residual_proj = g.add(proj_out[0], residual_attn)[0];

        vec![residual_proj]
    }
}

/// Llama-2-7B transformer BODY only: RoPE attention + FFN layers +
/// final RMSNorm, returning the hidden state `[1, seq_len, hidden_dim]`
/// (NO logits head). For one-shot AR proving the caller attaches its own
/// `lm_head` + `argmax_check` (the bundled head in [`llama_2_7b`] folds
/// the seq axis away assuming seq_len == 1). Same layer args as
/// `llama_2_7b` minus `logits_w_shards`.
pub fn llama_2_7b_hidden(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    proj_1_w_vec: Vec<Vec<Witness>>,
    proj_2_w_vec: Vec<Vec<Witness>>,
    proj_3_w_vec: Vec<Vec<Witness>>,
    layer_norm_w: Witness,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "llama_2_7b_hidden expects 1 input");
        let mut x = x[0];
        let theta = 10000.0;
        let cos_param = g.param(Witness::new(
            vec![seq_len, head_dim], rope_cos_mat_llama3(head_dim, seq_len, theta),
            DataType::Float, *SF_LOG, Role::Constant));
        let sin_param = g.param(Witness::new(
            vec![seq_len, head_dim], rope_sin_mat_llama3(head_dim, seq_len, theta),
            DataType::Float, *SF_LOG, Role::Constant));
        let num_layers = attn_norm_w_vec.len();
        for i in 0..num_layers {
            let block = g.pipe(&[x], llama_block(
                attn_norm_w_vec[i].clone(), attn_q_w_vec[i].clone(),
                attn_k_w_vec[i].clone(), attn_v_w_vec[i].clone(),
                attn_o_w_vec[i].clone(), proj_norm_w_vec[i].clone(),
                proj_1_w_vec[i].clone(), proj_2_w_vec[i].clone(),
                proj_3_w_vec[i].clone(), num_heads, head_dim, seq_len,
                cos_param, sin_param));
            x = block[0];
            if i < num_layers - 1 { g.layer_boundaries.push(x); }
        }
        let layer_norm_w = g.param(layer_norm_w);
        g.pipe(&[x], llama_rms_norm(layer_norm_w))
    }
}

/// Llama-2-7B end-to-end DAG.
///
/// `logits_w_shards` is a vector of N weight matrices each of shape
/// `[hidden_dim, vocab / N]` — the logits head's W is sharded along
/// the vocab axis so each einsum's wide-poly arity stays ≤ 22 (= max
/// GPU same-point bucket size). Passing a single-element vec
/// reproduces the un-sharded behavior. The DAG produces N output
/// edges (one per shard); the prover/verifier sees them as
/// independent output_ports.
pub fn llama_2_7b(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    // FFN matrices, outer = layer, inner = ffn-axis shards (gate / up / down).
    proj_1_w_vec: Vec<Vec<Witness>>,
    proj_2_w_vec: Vec<Vec<Witness>>,
    proj_3_w_vec: Vec<Vec<Witness>>,
    layer_norm_w: Witness,
    logits_w_shards: Vec<Witness>,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(
            x.len() == 1,
            "This custom LLaMA-2-7B layer expects 1 input"
        );
        let mut x = x[0];

        // Precompute RoPE cos/sin matrices: shape [seq_len, head_dim]
        let theta = 10000.0;
        let cos_data = rope_cos_mat_llama3(head_dim, seq_len, theta);
        let sin_data = rope_sin_mat_llama3(head_dim, seq_len, theta);
        let cos_param = g.param(Witness::new(
            vec![seq_len, head_dim],
            cos_data,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let sin_param = g.param(Witness::new(
            vec![seq_len, head_dim],
            sin_data,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));

        let num_layers = attn_norm_w_vec.len();
        for i in 0..num_layers {
            let block = g.pipe(
                &[x],
                llama_block(
                    attn_norm_w_vec[i].clone(),
                    attn_q_w_vec[i].clone(),
                    attn_k_w_vec[i].clone(),
                    attn_v_w_vec[i].clone(),
                    attn_o_w_vec[i].clone(),
                    proj_norm_w_vec[i].clone(),
                    proj_1_w_vec[i].clone(),
                    proj_2_w_vec[i].clone(),
                    proj_3_w_vec[i].clone(),
                    num_heads,
                    head_dim,
                    seq_len,
                    cos_param,
                    sin_param,
                ),
            );
            x = block[0];
            if i < num_layers - 1 {
                g.layer_boundaries.push(x);
            }
        }
        let layer_norm_w = g.param(layer_norm_w);
        let layer_norm_out = g.pipe(&[x], llama_rms_norm(layer_norm_w));
        // One einsum per logits shard. The vocab dim is split across N
        // shards (caller's responsibility), each producing a separate
        // output edge of shape `[1, vocab / N]`.
        let mut outs = Vec::with_capacity(logits_w_shards.len());
        for shard in logits_w_shards.into_iter() {
            let shard_vocab = shard.shape[1];
            let logits_w = g.param(shard);
            let raw = g.einsum(
                "bij,jk->ik".to_string(),
                vec![layer_norm_out[0], logits_w],
                true,
            )[0];
            let out = g.change_shape(raw, vec![1, shard_vocab]);
            outs.push(out);
        }
        outs
    }
}

// ============================================================================
// Llama 3.1 8B support (GQA + llama3 RoPE)
// ============================================================================

/// Llama 3 frequency scaling: adjusts RoPE frequencies based on wavelength.
fn llama3_adjusted_freq(
    freq: f64,
    factor: f64,
    low_freq_factor: f64,
    high_freq_factor: f64,
    orig_ctx: usize,
) -> f64 {
    let wavelength = 2.0 * PI / freq;
    let low_freq_wavelen = orig_ctx as f64 / low_freq_factor;
    let high_freq_wavelen = orig_ctx as f64 / high_freq_factor;

    if wavelength < high_freq_wavelen {
        freq // high frequency: keep original
    } else if wavelength > low_freq_wavelen {
        freq / factor // low frequency: scale down
    } else {
        // smooth interpolation
        let smooth =
            (orig_ctx as f64 / wavelength - low_freq_factor) / (high_freq_factor - low_freq_factor);
        (1.0 - smooth) * freq / factor + smooth * freq
    }
}

fn field_from_i64(v: i64) -> AlmostGoldilocksField {
    if v >= 0 {
        AlmostGoldilocksField(v as u64)
    } else {
        AlmostGoldilocksField(0) - AlmostGoldilocksField((-v) as u64)
    }
}

/// Generate cos matrix for Llama 3 RoPE: shape [seq_len, d].
/// Data is in MLE order (s has stride 1, d has stride seq_padded).
pub fn rope_cos_mat_llama3(d: usize, seq_len: usize, theta: f64) -> Vec<AlmostGoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let seq_padded = seq_len.next_power_of_two().max(1);
    let total = seq_padded * d; // d=128 already pow2
    let mut data = vec![AlmostGoldilocksField(0); total];

    let factor = 8.0;
    let low_freq_factor = 1.0;
    let high_freq_factor = 4.0;
    let orig_ctx = 8192;

    for i in 0..half {
        let freq = theta.powf(-2.0 * (i as f64) / (d as f64));
        let adj_freq = llama3_adjusted_freq(freq, factor, low_freq_factor, high_freq_factor, orig_ctx);

        for m in 0..seq_len {
            let angle = m as f64 * adj_freq;
            let c = field_from_i64((angle.cos() * *SF_FLOAT as f64).round() as i64);
            // cos[2i] = cos[2i+1] = cos(angle)
            data[m + (2 * i) * seq_padded] = c;
            data[m + (2 * i + 1) * seq_padded] = c;
        }
    }

    data
}

/// Generate sin matrix for Llama 3 RoPE: shape [seq_len, d].
/// Data is in MLE order (s has stride 1, d has stride seq_padded).
pub fn rope_sin_mat_llama3(d: usize, seq_len: usize, theta: f64) -> Vec<AlmostGoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let seq_padded = seq_len.next_power_of_two().max(1);
    let total = seq_padded * d;
    let mut data = vec![AlmostGoldilocksField(0); total];

    let factor = 8.0;
    let low_freq_factor = 1.0;
    let high_freq_factor = 4.0;
    let orig_ctx = 8192;

    for i in 0..half {
        let freq = theta.powf(-2.0 * (i as f64) / (d as f64));
        let adj_freq = llama3_adjusted_freq(freq, factor, low_freq_factor, high_freq_factor, orig_ctx);

        for m in 0..seq_len {
            let angle = m as f64 * adj_freq;
            let s = angle.sin();
            // sin[2i] = +sin(angle), sin[2i+1] = -sin(angle)
            data[m + (2 * i) * seq_padded] = field_from_i64((s * *SF_FLOAT as f64).round() as i64);
            data[m + (2 * i + 1) * seq_padded] =
                field_from_i64((-s * *SF_FLOAT as f64).round() as i64);
        }
    }

    data
}

/// RMSNorm that handles seq > 1 correctly.
fn llama3_rms_norm(
    w_e: EdgeId,
    seq: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "RMSNorm expects 1 input");
        let x = x[0];
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let n = x_shape[x_shape.len() - 1];
        let r = g.rms_reciprocal(x)[0];

        let sf = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(*SF_FLOAT as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let tolerance = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(2)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![x, x], true)[0];
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0];
        let x_mean = g.div_const(x_sum, n)[0];
        let x_mean = g.change_shape(x_mean, vec![1, seq, 1]);
        let n_param: usize = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(n as u64)],
            DataType::Float,
            0,
            Role::Constant,
        ));
        let mean_tolerance = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField((n / 2) as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_mean_mul_n =
            g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

        // `x_sum` is [1, seq] but `x_mean_mul_n` is [1, seq, 1]. Subtracting them
        // directly broadcasts to a [1, seq, seq] OUTER PRODUCT whose entry (i, j)
        // is `x_sum[j] - n*x_mean[i]`, a difference between two DIFFERENT rows'
        // sums. Only the diagonal is the intended rounding remainder; off-diagonal
        // entries are unbounded in both signs and break the NonNegative check as
        // soon as row energies differ. A power-of-two padding tail guarantees they
        // differ, since a pad row has zero energy while real rows do not - which
        // is why non-power-of-two sequence lengths (BERT-Large 384, Whisper
        // large-v3 1500/448) failed while power-of-two ones (GPT-2 1024) happened
        // to stay in range. Reshape first so the sub stays element-wise per row.
        // Same fix as dag/bert.rs and dag/whisper.rs::whisper_layer_norm.
        let x_sum_reshaped = g.change_shape(x_sum, vec![1, seq, 1]);
        let x_sum_sub_x_mean_mul_n = g.sub(x_sum_reshaped, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0];
        let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
        let z_sf_diff = g.sub(z, sf)[0];
        // The reciprocity gate |mean(x^2)*r^2 - 1| <= tolerance is only satisfiable
        // where a row carries data. On a zero padding row mean(x^2) = 0, so z = 0
        // and z_sf_diff = -sf: positive_3 becomes -sf + tolerance (-1022 at SF=10,
        // tolerance 2) and positive_4 becomes sf + tolerance (1026) - exactly the
        // negative-and-overflow pair measured on BERT and Whisper. Mask the pad
        // rows so they read as |0| <= tolerance. Nothing is lost: r on a pad row
        // only ever multiplies a zero row, so it is unconstrained by construction.
        // Only mask when the sequence axis is ACTUALLY padded. The gate exists to
        // neutralise all-zero padding rows; when `seq` is already a power of two
        // there are none, and the mask is a semantic no-op that still costs a
        // constant witness plus an einsum per norm. Measured on llama2 8L/seq64
        // (seq already 64): masking unconditionally added 357 fold-tree leaves and
        // cost ~15% prove (36.4-38.3s -> 42.3-43.4s). Models with a non-power-of-two
        // sequence or context length still get it, which is what BERT-Large (384)
        // and Whisper large-v3 (1500/448) need.
        let z_sf_diff = if seq.next_power_of_two() != seq {
            g.mask(z_sf_diff, vec![1, seq, 1])
        } else {
            z_sf_diff
        };
        let positive_3 = g.add(z_sf_diff, tolerance)[0];
        let positive_4 = g.sub(tolerance, z_sf_diff)[0];
        g.add_nonneg_node(positive_3);
        g.add_nonneg_node(positive_4);

        let r = g.change_shape(r, vec![1, seq]);
        let h = g.einsum("bsi,bs->bsi".to_string(), vec![x, r], true)[0];
        let out = g.einsum("bsi,i->bsi".to_string(), vec![h, w_e], true)[0];
        vec![out]
    }
}

/// GQA attention for Llama 3.1 8B.
/// Supports num_heads Q heads with num_kv_heads KV heads (num_heads / num_kv_heads queries per group).
pub fn llama3_attention(
    w_q_e: EdgeId,
    w_k_e: EdgeId,
    w_v_e: EdgeId,
    w_o_e: EdgeId,
    cos_param: EdgeId,
    sin_param: EdgeId,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "GQA attention expects 1 input");
        let inp = x[0];
        let hidden_dim = num_heads * head_dim;
        let num_queries_per_group = num_heads / num_kv_heads;

        // Linear projections
        let q = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_q_e], true)[0];
        let k = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_k_e], true)[0];
        let v = g.einsum("bsi,ij->bsj".to_string(), vec![inp, w_v_e], true)[0];

        // Reshape for multi-head: split last dim into [heads, head_dim]
        let q = g.reshape(q, vec![1, seq_len, num_heads, head_dim])[0]; // [1, s, 32, 128]
        let k = g.reshape(k, vec![1, seq_len, num_kv_heads, head_dim])[0]; // [1, s, 8, 128]
        let v = g.reshape(v, vec![1, seq_len, num_kv_heads, head_dim])[0]; // [1, s, 8, 128]

        // Apply RoPE
        let q = g.rope_with_vecs(q, cos_param, sin_param)[0];
        let k = g.rope_with_vecs(k, cos_param, sin_param)[0];

        // Split Q heads into groups for GQA: [1, s, 32, 128] → [1, s, 4, 8, 128]
        // In MLE, low bits of head dim = within-group query index (q),
        // high bits = group index (g). So shape is [b, s, q, g, d].
        let q = g.change_shape(
            q,
            vec![1, seq_len, num_queries_per_group, num_kv_heads, head_dim],
        );

        // Transpose for attention
        // Q: [1, s, 4, 8, 128] (bsqgd) → [1, 8, 4, s, 128] (bgqsd)
        let q = g.einsum("bsqgd->bgqsd".to_string(), vec![q], false)[0];
        // K: [1, s, 8, 128] (bsgd) → [1, 8, s, 128] (bgsd)
        let k = g.einsum("bsgd->bgsd".to_string(), vec![k], false)[0];
        // V: [1, s, 8, 128] (bsgd) → [1, 8, s, 128] (bgsd)
        let v = g.einsum("bsgd->bgsd".to_string(), vec![v], false)[0];

        // Attention scores: Q·K^T with GQA broadcasting
        // q: [1, 8, 4, s, 128], k: [1, 8, s, 128]
        // K broadcasts over q dimension
        let scores =
            g.einsum("bgqsd,bgtd->bgqst".to_string(), vec![q, k], true)[0];

        // Scale by 1/sqrt(d)
        let d_sqrt_recip = ((*SF_FLOAT as f64) / (head_dim as f64).sqrt()).round() as u64;
        let d_sqrt_recip = g.param(Witness::new(
            vec![1],
            vec![AlmostGoldilocksField(d_sqrt_recip)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let scores = g.einsum(
            "bgqst,z->bgqst".to_string(),
            vec![scores, d_sqrt_recip],
            true,
        )[0];

        // Causal mask + normalized softmax
        let scores = if seq_len > 1 {
            g.causal_mask(scores, seq_len)
        } else {
            scores
        };
        let softmax_c = g.softmax_const(scores)[0];
        let scores = g.add(scores, softmax_c)[0];
        let scores = g.exp(scores)[0];

        // Weighted values: scores · V with GQA broadcasting
        // V broadcasts over q dimension
        let out =
            g.einsum("bgqst,bgtd->bgqsd".to_string(), vec![scores, v], true)[0];

        // Flatten back: [1, 8, 4, s, 128] → [1, s, 4, 8, 128] → [1, s, 32, 128] → [1, s, 4096]
        let out = g.einsum("bgqsd->bsqgd".to_string(), vec![out], false)[0];
        let out = g.change_shape(out, vec![1, seq_len, num_heads, head_dim]);
        let out = g.reshape(out, vec![1, seq_len, hidden_dim])[0];

        // Output projection
        let out =
            g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0];
        vec![out]
    }
}

/// Llama 3.1 transformer block (GQA attention + SwiGLU MLP).
pub fn llama3_block(
    attn_norm_w: Witness,
    attn_q_w: Witness,
    attn_k_w: Witness,
    attn_v_w: Witness,
    attn_o_w: Witness,
    proj_norm_w: Witness,
    // FFN weights, sharded along the `ffn` axis for arity control. A
    // single-element vec is the un-sharded behaviour. Measured on Llama-2:
    // sharding cuts the proof 391.6 -> 222.6 MB and prove 285 -> 227s, because
    // many small committed edges fold better than a few huge ones.
    proj_1_w: Vec<Witness>,
    proj_2_w: Vec<Witness>,
    proj_3_w: Vec<Witness>,
    cos_param: EdgeId,
    sin_param: EdgeId,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "Block expects 1 input");
        let x = x[0];
        let attn_norm = g.param(attn_norm_w);
        let attn_q = g.param(attn_q_w);
        let attn_k = g.param(attn_k_w);
        let attn_v = g.param(attn_v_w);
        let attn_o = g.param(attn_o_w);
        let proj_norm = g.param(proj_norm_w);
        let proj_1: Vec<_> = proj_1_w.into_iter().map(|w| g.param(w)).collect();
        let proj_2: Vec<_> = proj_2_w.into_iter().map(|w| g.param(w)).collect();
        let proj_3: Vec<_> = proj_3_w.into_iter().map(|w| g.param(w)).collect();

        let attn_norm_out = g.pipe(&[x], llama3_rms_norm(attn_norm, seq_len));
        let attn_out = g.pipe(
            &[attn_norm_out[0]],
            llama3_attention(
                attn_q, attn_k, attn_v, attn_o, cos_param, sin_param, num_heads,
                num_kv_heads, head_dim, seq_len,
            ),
        );
        let residual_attn = g.add(attn_out[0], x)[0];

        let proj_norm_out = g.pipe(&[residual_attn], llama3_rms_norm(proj_norm, seq_len));
        let proj_out = g.pipe(
            &proj_norm_out,
            llama_mlp(proj_1, proj_2, proj_3),
        );
        let residual_proj = g.add(proj_out[0], residual_attn)[0];

        vec![residual_proj]
    }
}

/// Llama 3.1 8B model builder.
pub fn llama3_8b(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    // Outer = layer, inner = ffn-axis shards. A single-element inner vec is
    // the un-sharded behaviour; see `llama_mlp`.
    proj_1_w_vec: Vec<Vec<Witness>>,
    proj_2_w_vec: Vec<Vec<Witness>>,
    proj_3_w_vec: Vec<Vec<Witness>>,
    layer_norm_w: Witness,
    // Vocab-axis shards of the lm_head; a single-element vec is unsharded.
    logits_w_shards: Vec<Witness>,
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "Llama 3.1 8B expects 1 input");
        let mut x = x[0];

        // Precompute RoPE cos/sin matrices: shape [seq_len, head_dim]
        let theta = 500000.0;
        let cos_data = rope_cos_mat_llama3(head_dim, seq_len, theta);
        let sin_data = rope_sin_mat_llama3(head_dim, seq_len, theta);
        let cos_param = g.param(Witness::new(
            vec![seq_len, head_dim],
            cos_data,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let sin_param = g.param(Witness::new(
            vec![seq_len, head_dim],
            sin_data,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));

        let num_layers = attn_norm_w_vec.len();
        for i in 0..num_layers {
            let block = g.pipe(
                &[x],
                llama3_block(
                    attn_norm_w_vec[i].clone(),
                    attn_q_w_vec[i].clone(),
                    attn_k_w_vec[i].clone(),
                    attn_v_w_vec[i].clone(),
                    attn_o_w_vec[i].clone(),
                    proj_norm_w_vec[i].clone(),
                    proj_1_w_vec[i].clone(),
                    proj_2_w_vec[i].clone(),
                    proj_3_w_vec[i].clone(),
                    cos_param,
                    sin_param,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    seq_len,
                ),
            );
            x = block[0];
            if i < num_layers - 1 {
                g.layer_boundaries.push(x);
            }
        }
        let layer_norm_w = g.param(layer_norm_w);
        let layer_norm_out = g.pipe(&[x], llama3_rms_norm(layer_norm_w, seq_len));
        // One einsum per logits shard. The vocab dim is split across N shards
        // (caller's responsibility), each producing an output of shape
        // [1, seq, vocab/N]. Unsharded, Llama-3's head is 4096 x 131072 = 2^29
        // in ONE committed edge, which is the single largest arity in the
        // model and the reason its proof is ~4.6x Llama-2's.
        let mut outs = Vec::with_capacity(logits_w_shards.len());
        for shard in logits_w_shards.into_iter() {
            let shard_vocab = shard.shape[1];
            let logits_w = g.param(shard);
            let raw = g.einsum(
                "bsi,ij->bsj".to_string(),
                vec![layer_norm_out[0], logits_w],
                true,
            )[0];
            outs.push(g.change_shape(raw, vec![1, seq_len, shard_vocab]));
        }
        let _ = vocab_size;
        outs
    }
}

/// Llama-3.1-8B transformer BODY (blocks + final RMSNorm), returning the
/// hidden state `[1, seq, hidden]` — no logits head. Mirror of `llama3_8b`
/// without the vocab einsum, for one-shot AR proving where the head is the
/// shared `lm_head` (+ `argmax_check`) from `oneshot.rs`. Same as
/// `llama_2_7b_hidden` but GQA (`num_kv_heads`) + RoPE θ=500000.
#[allow(clippy::too_many_arguments)]
pub fn llama3_8b_hidden(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    // Outer = layer, inner = ffn-axis shards. A single-element inner vec is
    // the un-sharded behaviour; see `llama_mlp`.
    proj_1_w_vec: Vec<Vec<Witness>>,
    proj_2_w_vec: Vec<Vec<Witness>>,
    proj_3_w_vec: Vec<Vec<Witness>>,
    layer_norm_w: Witness,
    seq_len: usize,
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "llama3_8b_hidden expects 1 input");
        let mut x = x[0];
        let theta = 500000.0;
        let cos_param = g.param(Witness::new(
            vec![seq_len, head_dim], rope_cos_mat_llama3(head_dim, seq_len, theta),
            DataType::Float, *SF_LOG, Role::Constant));
        let sin_param = g.param(Witness::new(
            vec![seq_len, head_dim], rope_sin_mat_llama3(head_dim, seq_len, theta),
            DataType::Float, *SF_LOG, Role::Constant));
        let num_layers = attn_norm_w_vec.len();
        for i in 0..num_layers {
            let block = g.pipe(&[x], llama3_block(
                attn_norm_w_vec[i].clone(), attn_q_w_vec[i].clone(),
                attn_k_w_vec[i].clone(), attn_v_w_vec[i].clone(),
                attn_o_w_vec[i].clone(), proj_norm_w_vec[i].clone(),
                proj_1_w_vec[i].clone(), proj_2_w_vec[i].clone(),
                proj_3_w_vec[i].clone(), cos_param, sin_param,
                num_heads, num_kv_heads, head_dim, seq_len));
            x = block[0];
            if i < num_layers - 1 { g.layer_boundaries.push(x); }
        }
        let layer_norm_w = g.param(layer_norm_w);
        g.pipe(&[x], llama3_rms_norm(layer_norm_w, seq_len))
    }
}
