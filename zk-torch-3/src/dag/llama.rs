use goldilocks_cuda::GoldilocksField;
use ndarray::Array2;
use std::f64::consts::PI;

use crate::basicblock::BasicBlockType;
use crate::basicblock::{DivConst, RMSReciprocal, SoftmaxConst};
use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::SF_FLOAT;
use crate::SF_LOG;

pub fn pair_swap_perm_matrix(d: usize) -> Vec<GoldilocksField> {
    assert!(d % 2 == 0, "dimension d must be even");

    let mut m = Array2::from_elem((d, d), GoldilocksField(0));

    for i in (0..d).step_by(2) {
        m[[i, i + 1]] = GoldilocksField(1);
        m[[i + 1, i]] = GoldilocksField(1);
    }

    m.into_dyn()
        .view()
        .reversed_axes()
        .iter()
        .copied()
        .collect::<Vec<GoldilocksField>>()
}

/// Generate the cosine vector for RoPE.
pub fn rope_cos_vec(d: usize, m: f64, base: f64) -> Vec<GoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let mut v = Vec::with_capacity(d);

    for i in 0..half {
        let theta_i = base.powf(-2.0 * (i as f64) / (d as f64));
        let angle = m * theta_i;
        let c = angle.cos();
        let c = (c * *SF_FLOAT as f64).round() as i64;
        let c = if c > 0 {
            GoldilocksField(c as u64)
        } else {
            GoldilocksField(0) - GoldilocksField((-c) as u64)
        };
        v.push(c);
        v.push(c);
    }

    v
}

/// Generate the sine vector for RoPE.
pub fn rope_sin_vec(d: usize, m: f64, base: f64) -> Vec<GoldilocksField> {
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
                GoldilocksField(*f as u64)
            } else {
                GoldilocksField(0) - GoldilocksField((-*f) as u64)
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

    pub fn div_const(&mut self, a: EdgeId, c: usize) -> Vec<EdgeId> {
        let div_const_basicblock =
            BasicBlockType::DivConst(DivConst { divisor: c as u64 });
        assert!(self.init_values[a].is_some(), "Input must be initialized");
        let inp_value = self.init_values[a].as_ref().unwrap();
        let shape = inp_value.shape.clone();
        let sf = inp_value.sf;
        let data_type = inp_value.data_type;
        let out_value = Witness::new_wo_data(shape, data_type, sf, Role::Output);
        self.init_values.push(Some(out_value));
        self.add_gkr_node(vec![a], div_const_basicblock)
    }

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
            vec![GoldilocksField(*SF_FLOAT as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let tolerance = g.param(Witness::new(
            vec![1],
            vec![GoldilocksField(2)],
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
            vec![GoldilocksField(n as u64)],
            DataType::Float,
            0,
            Role::Constant,
        ));
        let mean_tolerance = g.param(Witness::new(
            vec![1],
            vec![GoldilocksField((n / 2) as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_mean_mul_n =
            g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

        let x_sum_sub_x_mean_mul_n = g.sub(x_sum, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0];
        let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
        let z_sf_diff = g.sub(z, sf)[0];
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

pub fn llama_mlp(
    w_1_e: EdgeId,
    w_2_e: EdgeId,
    w_3_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom MLP layer expects 1 input");
        let x = x[0];
        let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0];
        let sigmoid = g.sigmoid(h_1)[0];
        let swish = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, sigmoid], true)[0];
        let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_2_e], true)[0];
        let mul = g.einsum("bsi,bsi->bsi".to_string(), vec![swish, h_2], true)[0];
        let out = g.einsum("bsi,ij->bsj".to_string(), vec![mul, w_3_e], true)[0];
        vec![out]
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
            vec![GoldilocksField(d_sqrt_recip)],
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
    proj_1_w: Witness,
    proj_2_w: Witness,
    proj_3_w: Witness,
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
        let proj_1 = g.param(proj_1_w);
        let proj_2 = g.param(proj_2_w);
        let proj_3 = g.param(proj_3_w);

        let attn_norm_out = g.pipe(&[x], llama_rms_norm(attn_norm));
        let attn_out = g.pipe(
            &[attn_norm_out[0]],
            llama_attention(attn_q, attn_k, attn_v, attn_o, num_heads, head_dim, seq_len, cos_param, sin_param),
        );
        let residual_attn = g.add(attn_out[0], x)[0];

        let proj_norm_out = g.pipe(&[residual_attn], llama_rms_norm(proj_norm));
        let proj_out = g.pipe(&proj_norm_out, llama_mlp(proj_1, proj_2, proj_3));
        let residual_proj = g.add(proj_out[0], residual_attn)[0];

        vec![residual_proj]
    }
}

pub fn llama_2_7b(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    proj_1_w_vec: Vec<Witness>,
    proj_2_w_vec: Vec<Witness>,
    proj_3_w_vec: Vec<Witness>,
    layer_norm_w: Witness,
    logits_w: Witness,
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
        let logits_w = g.param(logits_w);
        let out = g.einsum(
            "bij,jk->ik".to_string(),
            vec![layer_norm_out[0], logits_w],
            true,
        )[0];
        let out = g.change_shape(out, vec![1, 32000]);
        vec![out]
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

fn field_from_i64(v: i64) -> GoldilocksField {
    if v >= 0 {
        GoldilocksField(v as u64)
    } else {
        GoldilocksField(0) - GoldilocksField((-v) as u64)
    }
}

/// Generate cos matrix for Llama 3 RoPE: shape [seq_len, d].
/// Data is in MLE order (s has stride 1, d has stride seq_padded).
pub fn rope_cos_mat_llama3(d: usize, seq_len: usize, theta: f64) -> Vec<GoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let seq_padded = seq_len.next_power_of_two().max(1);
    let total = seq_padded * d; // d=128 already pow2
    let mut data = vec![GoldilocksField(0); total];

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
pub fn rope_sin_mat_llama3(d: usize, seq_len: usize, theta: f64) -> Vec<GoldilocksField> {
    assert!(d % 2 == 0, "d must be even");
    let half = d / 2;
    let seq_padded = seq_len.next_power_of_two().max(1);
    let total = seq_padded * d;
    let mut data = vec![GoldilocksField(0); total];

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
            vec![GoldilocksField(*SF_FLOAT as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let tolerance = g.param(Witness::new(
            vec![1],
            vec![GoldilocksField(2)],
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
            vec![GoldilocksField(n as u64)],
            DataType::Float,
            0,
            Role::Constant,
        ));
        let mean_tolerance = g.param(Witness::new(
            vec![1],
            vec![GoldilocksField((n / 2) as u64)],
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        ));
        let x_mean_mul_n =
            g.einsum("bsi,i->bsi".to_string(), vec![x_mean, n_param], false)[0];

        let x_sum_sub_x_mean_mul_n = g.sub(x_sum, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0];
        let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
        let z_sf_diff = g.sub(z, sf)[0];
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
            vec![GoldilocksField(d_sqrt_recip)],
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
    proj_1_w: Witness,
    proj_2_w: Witness,
    proj_3_w: Witness,
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
        let proj_1 = g.param(proj_1_w);
        let proj_2 = g.param(proj_2_w);
        let proj_3 = g.param(proj_3_w);

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
        let proj_out = g.pipe(&proj_norm_out, llama_mlp(proj_1, proj_2, proj_3));
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
    proj_1_w_vec: Vec<Witness>,
    proj_2_w_vec: Vec<Witness>,
    proj_3_w_vec: Vec<Witness>,
    layer_norm_w: Witness,
    logits_w: Witness,
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
        let logits_w = g.param(logits_w);
        let out = g.einsum(
            "bsi,ij->bsj".to_string(),
            vec![layer_norm_out[0], logits_w],
            true,
        )[0];
        let out = g.change_shape(out, vec![1, seq_len, vocab_size]);
        vec![out]
    }
}
