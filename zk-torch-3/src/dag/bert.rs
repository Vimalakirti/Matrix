use goldilocks_cuda::GoldilocksField;
use ndarray::ArrayD;

use crate::dag::llama::llama_rms_norm;
use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::shape::pad_to_pow_of_two;
use crate::SF_FLOAT;
use crate::SF_LOG;

pub fn bert_layer_norm(
    w_e: EdgeId,
    b_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom LayerNorm layer expects 1 input");
        let x = x[0];
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x], false)[0];
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let n = x_shape[x_shape.len() - 1];
        let x_mean = g.div_const(x_sum, n)[0];
        let seq = x_shape[1];
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

        let x_minus_mean = g.sub(x, x_mean)[0];
        let x_minus_mean = g.mask(x_minus_mean, vec![1, seq, n]);

        let x_rms = g.pipe(&[x_minus_mean], llama_rms_norm(w_e))[0];

        let out = g.add(x_rms, b_e)[0];
        vec![out]
    }
}

pub fn bert_mlp(
    w_1_e: EdgeId,
    w_2_e: EdgeId,
    b_1_e: EdgeId,
    b_2_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom MLP layer expects 1 input");
        let x = x[0];
        let h_1 = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1_e], true)[0];
        let h_1 = g.add(h_1, b_1_e)[0];

        // compute Gelu
        let shape = g.init_values[h_1].as_ref().unwrap().shape.clone();
        let val_num: usize = shape.iter().product();
        let vals: Vec<GoldilocksField> = (0..val_num)
            .map(|_| GoldilocksField((1.702 * *SF_FLOAT).round() as u64))
            .collect();
        let vals = ArrayD::from_shape_vec(shape.clone(), vals).unwrap();
        let pad_vals = pad_to_pow_of_two(&vals, &GoldilocksField(0));
        let col_major: Vec<_> = pad_vals.view().reversed_axes().iter().cloned().collect();
        let constant = Witness::new(
            shape,
            col_major,
            DataType::Float,
            *SF_LOG,
            Role::Constant,
        );
        let constant = g.param(constant);
        let h_1 = g.einsum("bsi,bsi->bsi".to_string(), vec![h_1, constant], true)[0];
        let h_1 = g.sigmoid(h_1)[0];

        let h_2 = g.einsum("bsi,ij->bsj".to_string(), vec![h_1, w_2_e], true)[0];
        let h_2 = g.add(h_2, b_2_e)[0];
        vec![h_2]
    }
}

pub fn bert_attention(
    w_q_e: EdgeId,
    w_k_e: EdgeId,
    w_v_e: EdgeId,
    w_o_e: EdgeId,
    b_q_e: EdgeId,
    b_k_e: EdgeId,
    b_v_e: EdgeId,
    b_o_e: EdgeId,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
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

        let q = g.add(q, b_q_e)[0];
        let k = g.add(k, b_k_e)[0];
        let v = g.add(v, b_v_e)[0];

        let q = g.reshape(q, vec![1, seq_len, num_heads, head_dim])[0];
        let k = g.reshape(k, vec![1, seq_len, num_heads, head_dim])[0];
        let v = g.reshape(v, vec![1, seq_len, num_heads, head_dim])[0];

        let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
        let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
        let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

        let d_sqrt_recip = (*SF_FLOAT as f64 / (head_dim as f64).sqrt()).round() as u64;
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
        let softmax_c = g.softmax_const(scores)[0];
        let scores = g.add(scores, softmax_c)[0];
        let scores = g.exp(scores)[0];

        let out =
            g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
        let out = g.change_shape(out, vec![1, seq_len, num_heads, head_dim]);
        let out = g.reshape(out, vec![1, seq_len, num_heads * head_dim])[0];

        let out =
            g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o_e], true)[0];
        let out = g.add(out, b_o_e)[0];
        vec![out]
    }
}

pub fn bert_block(
    attn_norm_w: Witness,
    attn_q_w: Witness,
    attn_k_w: Witness,
    attn_v_w: Witness,
    attn_o_w: Witness,
    attn_norm_b: Witness,
    attn_q_b: Witness,
    attn_k_b: Witness,
    attn_v_b: Witness,
    attn_o_b: Witness,
    proj_norm_w: Witness,
    proj_1_w: Witness,
    proj_2_w: Witness,
    proj_norm_b: Witness,
    proj_1_b: Witness,
    proj_2_b: Witness,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom Block layer expects 1 input");
        let x = x[0];
        let attn_norm_w = g.param(attn_norm_w);
        let attn_q_w = g.param(attn_q_w);
        let attn_k_w = g.param(attn_k_w);
        let attn_v_w = g.param(attn_v_w);
        let attn_o_w = g.param(attn_o_w);
        let attn_norm_b = g.param(attn_norm_b);
        let attn_q_b = g.param(attn_q_b);
        let attn_k_b = g.param(attn_k_b);
        let attn_v_b = g.param(attn_v_b);
        let attn_o_b = g.param(attn_o_b);
        let proj_norm_w = g.param(proj_norm_w);
        let proj_1_w = g.param(proj_1_w);
        let proj_2_w = g.param(proj_2_w);
        let proj_norm_b = g.param(proj_norm_b);
        let proj_1_b = g.param(proj_1_b);
        let proj_2_b = g.param(proj_2_b);

        let attn_out = g.pipe(
            &[x],
            bert_attention(
                attn_q_w,
                attn_k_w,
                attn_v_w,
                attn_o_w,
                attn_q_b,
                attn_k_b,
                attn_v_b,
                attn_o_b,
                num_heads,
                head_dim,
                seq_len,
            ),
        );
        let residual_attn = g.add(attn_out[0], x)[0];
        let attn_norm_out =
            g.pipe(&[residual_attn], bert_layer_norm(attn_norm_w, attn_norm_b));
        let proj_out = g.pipe(
            &attn_norm_out,
            bert_mlp(proj_1_w, proj_2_w, proj_1_b, proj_2_b),
        );
        let residual_proj = g.add(proj_out[0], residual_attn)[0];
        let proj_norm_out =
            g.pipe(&[residual_proj], bert_layer_norm(proj_norm_w, proj_norm_b));
        vec![proj_norm_out[0]]
    }
}

pub fn bert_large(
    attn_norm_w_vec: Vec<Witness>,
    attn_q_w_vec: Vec<Witness>,
    attn_k_w_vec: Vec<Witness>,
    attn_v_w_vec: Vec<Witness>,
    attn_o_w_vec: Vec<Witness>,
    attn_norm_b_vec: Vec<Witness>,
    attn_q_b_vec: Vec<Witness>,
    attn_k_b_vec: Vec<Witness>,
    attn_v_b_vec: Vec<Witness>,
    attn_o_b_vec: Vec<Witness>,
    proj_norm_w_vec: Vec<Witness>,
    proj_1_w_vec: Vec<Witness>,
    proj_2_w_vec: Vec<Witness>,
    proj_norm_b_vec: Vec<Witness>,
    proj_1_b_vec: Vec<Witness>,
    proj_2_b_vec: Vec<Witness>,
    layer_norm_w: Witness,
    layer_norm_b: Witness,
    num_heads: usize,
    head_dim: usize,
    seq_len: usize,
    matmul_w: Witness,
    matmul_b: Witness,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(
            x.len() == 1,
            "This custom BERT-large layer expects 1 input"
        );
        let mut x = x[0];
        let layer_norm_w = g.param(layer_norm_w);
        let layer_norm_b = g.param(layer_norm_b);
        x = g.pipe(&[x], bert_layer_norm(layer_norm_w, layer_norm_b))[0];
        let num_layers = attn_norm_w_vec.len();
        for i in 0..num_layers {
            let block = g.pipe(
                &[x],
                bert_block(
                    attn_norm_w_vec[i].clone(),
                    attn_q_w_vec[i].clone(),
                    attn_k_w_vec[i].clone(),
                    attn_v_w_vec[i].clone(),
                    attn_o_w_vec[i].clone(),
                    attn_norm_b_vec[i].clone(),
                    attn_q_b_vec[i].clone(),
                    attn_k_b_vec[i].clone(),
                    attn_v_b_vec[i].clone(),
                    attn_o_b_vec[i].clone(),
                    proj_norm_w_vec[i].clone(),
                    proj_1_w_vec[i].clone(),
                    proj_2_w_vec[i].clone(),
                    proj_norm_b_vec[i].clone(),
                    proj_1_b_vec[i].clone(),
                    proj_2_b_vec[i].clone(),
                    num_heads,
                    head_dim,
                    seq_len,
                ),
            );
            x = block[0];
            if i < num_layers - 1 {
                g.layer_boundaries.push(x);
            }
        }

        let matmul_w = g.param(matmul_w);
        let matmul_b = g.param(matmul_b);
        let x = g.einsum("bsi,ij->bsj".to_string(), vec![x, matmul_w], true)[0];
        let x = g.add(x, matmul_b)[0];

        vec![x]
    }
}
