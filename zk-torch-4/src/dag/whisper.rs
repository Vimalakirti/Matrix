use almost_goldilocks_cuda::field::AlmostGoldilocksField;
use ndarray::ArrayD;
use std::sync::OnceLock;

use crate::dag::{DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::shape::pad_to_pow_of_two;
use crate::SF_FLOAT;
use crate::SF_LOG;

/// Width (in SF=10 units) of the LayerNorm reciprocity gate
/// `r² · mean(x²) ≈ sf`. `whisper_rms_norm` checks
/// `|z - sf| <= recip_tolerance`. Default 2 keeps the original ±0.002
/// float-band soundness margin; smoke-test bins that drive the LN with
/// low-variance synthetic inputs (where integer rounding in
/// `r² · mean(x²) / sf` exceeds ±2) can widen it via
/// `WHISPER_RECIP_TOL`. The same env knob lifts the gate's strictness
/// uniformly across all whisper LN instances.
fn whisper_recip_tolerance() -> u64 {
    static VAL: OnceLock<u64> = OnceLock::new();
    *VAL.get_or_init(|| {
        std::env::var("WHISPER_RECIP_TOL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2)
    })
}

// ============================================================================
// Whisper-specific LayerNorm (handles arbitrary seq_len)
// ============================================================================

/// LayerNorm for Whisper: works with any sequence length.
/// Input shape: [1, S, D] (batch=1, seq=S, hidden=D).
fn whisper_layer_norm(
    w_e: EdgeId,
    b_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "LayerNorm expects 1 input");
        let x = x[0];
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let seq = x_shape[1];
        let n = x_shape[x_shape.len() - 1];

        // Mean computation
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x], false)[0];
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

        // Reshape x_sum [1, seq] → [1, seq, 1] so the sub against
        // x_mean_mul_n (shape [1, seq, 1]) stays element-wise per row
        // instead of broadcasting to an outer-product [1, seq, seq].
        // (NumPy-style left-pad would otherwise produce [1, 1, seq] vs
        // [1, seq, 1] → [1, seq, seq], mixing rows.)
        let x_sum_reshaped = g.change_shape(x_sum, vec![1, seq, 1]);
        let x_sum_sub_x_mean_mul_n = g.sub(x_sum_reshaped, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let x_minus_mean = g.sub(x, x_mean)[0];
        let x_minus_mean = g.mask(x_minus_mean, vec![1, seq, n]);

        // RMS norm with correct reshaping for seq > 1
        let x_rms = g.pipe(&[x_minus_mean], whisper_rms_norm(w_e, seq))[0];

        let out = g.add(x_rms, b_e)[0];
        vec![out]
    }
}

/// RMS norm that handles seq > 1 correctly.
/// The key difference from llama_rms_norm: change_shape(r, [1, seq]) instead of [1, 1].
fn whisper_rms_norm(
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
            vec![AlmostGoldilocksField(whisper_recip_tolerance())],
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

        // See note in `whisper_layer_norm`: reshape x_sum so the sub is
        // element-wise per row rather than an outer-product broadcast.
        let x_sum_reshaped = g.change_shape(x_sum, vec![1, seq, 1]);
        let x_sum_sub_x_mean_mul_n = g.sub(x_sum_reshaped, x_mean_mul_n)[0];
        let positive_1 = g.add(x_sum_sub_x_mean_mul_n, mean_tolerance)[0];
        let positive_2 = g.sub(mean_tolerance, x_sum_sub_x_mean_mul_n)[0];
        g.add_nonneg_node(positive_1);
        g.add_nonneg_node(positive_2);

        let r_sq = g.einsum("bsi,bsi->bsi".to_string(), vec![r, r], true)[0];
        let z = g.einsum("bsi,bsi->bsi".to_string(), vec![x_mean, r_sq], true)[0];
        let z_sf_diff = g.sub(z, sf)[0];
        // Same pad-row masking as llama_rms_norm: the reciprocity gate
        // |mean(x^2)*r^2 - 1| <= tolerance cannot hold on a zero padding row,
        // where mean(x^2) = 0 makes z_sf_diff = -sf (positive_3 = -1022 at SF=10,
        // tolerance 2). Whisper hits this on its CONTEXT axes rather than the
        // hidden one, which is why n_audio_ctx 1500 -> 2048 and n_text_ctx
        // 448 -> 512 failed while a power-of-two ctx passed. Sound: r on a pad
        // row only ever multiplies a zero row.
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

        // Key fix: change_shape to [1, seq] instead of [1, 1]
        let r = g.change_shape(r, vec![1, seq]);
        let h = g.einsum("bsi,bs->bsi".to_string(), vec![x, r], true)[0];
        let out = g.einsum("bsi,i->bsi".to_string(), vec![h, w_e], true)[0];
        vec![out]
    }
}

// ============================================================================
// Whisper model graph
// ============================================================================

/// Weights for one encoder transformer block.
pub struct EncoderBlockWeights {
    pub attn_ln_w: Witness,
    pub attn_ln_b: Witness,
    pub w_q: Witness,
    pub w_k: Witness,
    pub w_v: Witness,
    pub w_o: Witness,
    pub b_q: Witness,
    pub b_k: Witness,
    pub b_v: Witness,
    pub b_o: Witness,
    pub mlp_ln_w: Witness,
    pub mlp_ln_b: Witness,
    pub w_mlp1: Witness,
    pub w_mlp2: Witness,
    pub b_mlp1: Witness,
    pub b_mlp2: Witness,
}

/// Weights for one decoder transformer block (self-attn + cross-attn + MLP).
pub struct DecoderBlockWeights {
    // Self-attention
    pub attn_ln_w: Witness,
    pub attn_ln_b: Witness,
    pub w_q: Witness,
    pub w_k: Witness,
    pub w_v: Witness,
    pub w_o: Witness,
    pub b_q: Witness,
    pub b_k: Witness,
    pub b_v: Witness,
    pub b_o: Witness,
    // Cross-attention
    pub cross_ln_w: Witness,
    pub cross_ln_b: Witness,
    pub xw_q: Witness,
    pub xw_k: Witness,
    pub xw_v: Witness,
    pub xw_o: Witness,
    pub xb_q: Witness,
    pub xb_k: Witness,
    pub xb_v: Witness,
    pub xb_o: Witness,
    // MLP
    pub mlp_ln_w: Witness,
    pub mlp_ln_b: Witness,
    pub w_mlp1: Witness,
    pub w_mlp2: Witness,
    pub b_mlp1: Witness,
    pub b_mlp2: Witness,
}

/// GELU activation: x * sigmoid(1.702 * x).
/// Works for any tensor shape by building the einsum string dynamically.
fn whisper_gelu(g: &mut DagBuilder, x: EdgeId) -> EdgeId {
    let shape = g.init_values[x].as_ref().unwrap().shape.clone();
    let ndim = shape.len();
    let letters: String = (0..ndim).map(|i| (b'a' + i as u8) as char).collect();
    let einsum_str = format!("{},{}->{}", letters, letters, letters);

    let val_num: usize = shape.iter().product();
    let vals: Vec<AlmostGoldilocksField> = (0..val_num)
        .map(|_| AlmostGoldilocksField((1.702 * *SF_FLOAT).round() as u64))
        .collect();
    let vals = ArrayD::from_shape_vec(shape.clone(), vals).unwrap();
    let pad_vals = pad_to_pow_of_two(&vals, &AlmostGoldilocksField(0));
    let col_major: Vec<_> = pad_vals.view().reversed_axes().iter().cloned().collect();
    let constant = Witness::new(
        shape,
        col_major,
        DataType::Float,
        *SF_LOG,
        Role::Constant,
    );
    let constant = g.param(constant);
    let h = g.einsum(einsum_str.clone(), vec![x, constant], true)[0];
    let h = g.sigmoid(h)[0];
    g.einsum(einsum_str, vec![x, h], true)[0]
}

/// Self-attention (no causal mask). Parameterized by n_head and head_dim.
fn whisper_self_attention(
    g: &mut DagBuilder,
    x: EdgeId,
    w_q: EdgeId,
    w_k: EdgeId,
    w_v: EdgeId,
    w_o: EdgeId,
    b_q: EdgeId,
    b_k: EdgeId,
    b_v: EdgeId,
    b_o: EdgeId,
    n_head: usize,
    head_dim: usize,
    n_state: usize,
) -> EdgeId {
    // x shape: [1, seq, n_state]
    let seq = g.init_values[x].as_ref().unwrap().shape[1];

    let q = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_q], true)[0];
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_k], true)[0];
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_v], true)[0];

    let q = g.add(q, b_q)[0];
    let k = g.add(k, b_k)[0];
    let v = g.add(v, b_v)[0];

    let q = g.reshape(q, vec![1, seq, n_head, head_dim])[0];
    let k = g.reshape(k, vec![1, seq, n_head, head_dim])[0];
    let v = g.reshape(v, vec![1, seq, n_head, head_dim])[0];

    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    let d_sqrt_recip = (*SF_FLOAT as f64 / (head_dim as f64).sqrt()).round() as u64;
    let d_sqrt_recip = g.param(Witness::new(
        vec![1],
        vec![AlmostGoldilocksField(d_sqrt_recip)],
        DataType::Float,
        *SF_LOG,
        Role::Constant,
    ));
    let scores = g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
    // Use 'z' for scalar to avoid overwriting 's' dimension when seq > 1
    let scores = g.einsum("bhst,z->bhst".to_string(), vec![scores, d_sqrt_recip], true)[0];
    let softmax_c = g.softmax_const(scores)[0];
    let scores = g.add(scores, softmax_c)[0];
    let scores = g.exp(scores)[0];

    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    let out = g.change_shape(out, vec![1, seq, n_head, head_dim]);
    let out = g.reshape(out, vec![1, seq, n_state])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, w_o], true)[0];
    g.add(out, b_o)[0]
}

/// Cross-attention: Q from decoder, K/V from encoder output.
/// seq_q and seq_kv may differ.
fn whisper_cross_attention(
    g: &mut DagBuilder,
    x_dec: EdgeId,       // decoder state [1, S_dec, n_state]
    enc_out: EdgeId,     // encoder output [1, S_enc, n_state]
    xw_q: EdgeId,
    xw_k: EdgeId,
    xw_v: EdgeId,
    xw_o: EdgeId,
    xb_q: EdgeId,
    xb_k: EdgeId,
    xb_v: EdgeId,
    xb_o: EdgeId,
    n_head: usize,
    head_dim: usize,
    n_state: usize,
    seq_dec: usize,
    seq_enc: usize,
) -> EdgeId {
    // Q from decoder
    let q = g.einsum("bsi,ij->bsj".to_string(), vec![x_dec, xw_q], true)[0];
    let q = g.add(q, xb_q)[0];

    // K, V from encoder output
    let k = g.einsum("bsi,ij->bsj".to_string(), vec![enc_out, xw_k], true)[0];
    let v = g.einsum("bsi,ij->bsj".to_string(), vec![enc_out, xw_v], true)[0];
    let k = g.add(k, xb_k)[0];
    let v = g.add(v, xb_v)[0];

    // Reshape to multi-head
    let q = g.reshape(q, vec![1, seq_dec, n_head, head_dim])[0];
    let k = g.reshape(k, vec![1, seq_enc, n_head, head_dim])[0];
    let v = g.reshape(v, vec![1, seq_enc, n_head, head_dim])[0];

    let q = g.einsum("bshd->bhsd".to_string(), vec![q], false)[0];
    let k = g.einsum("bshd->bhsd".to_string(), vec![k], false)[0];
    let v = g.einsum("bshd->bhsd".to_string(), vec![v], false)[0];

    let d_sqrt_recip = (*SF_FLOAT as f64 / (head_dim as f64).sqrt()).round() as u64;
    let d_sqrt_recip = g.param(Witness::new(
        vec![1],
        vec![AlmostGoldilocksField(d_sqrt_recip)],
        DataType::Float,
        *SF_LOG,
        Role::Constant,
    ));
    // scores[b,h,s,t] where s indexes decoder positions, t indexes encoder positions
    let scores = g.einsum("bhsd,bhtd->bhst".to_string(), vec![q, k], true)[0];
    // Use 'z' for scalar to avoid overwriting 's' dimension when seq > 1
    let scores = g.einsum("bhst,z->bhst".to_string(), vec![scores, d_sqrt_recip], true)[0];
    let softmax_c = g.softmax_const(scores)[0];
    let scores = g.add(scores, softmax_c)[0];
    let scores = g.exp(scores)[0];

    let out = g.einsum("bhst,bhtd->bhsd".to_string(), vec![scores, v], true)[0];
    let out = g.change_shape(out, vec![1, seq_dec, n_head, head_dim]);
    let out = g.reshape(out, vec![1, seq_dec, n_state])[0];

    let out = g.einsum("bsi,ij->bsj".to_string(), vec![out, xw_o], true)[0];
    g.add(out, xb_o)[0]
}

/// Whisper MLP: Linear → GELU → Linear.
fn whisper_mlp(
    g: &mut DagBuilder,
    x: EdgeId,
    w_1: EdgeId,
    w_2: EdgeId,
    b_1: EdgeId,
    b_2: EdgeId,
) -> EdgeId {
    let h = g.einsum("bsi,ij->bsj".to_string(), vec![x, w_1], true)[0];
    let h = g.add(h, b_1)[0];
    let h = whisper_gelu(g, h);
    let h = g.einsum("bsi,ij->bsj".to_string(), vec![h, w_2], true)[0];
    g.add(h, b_2)[0]
}

/// Build the audio encoder pipeline.
/// Input: mel spectrogram [n_mels, 2*n_audio_ctx]
/// Output: [1, n_audio_ctx, n_state]
pub fn whisper_encoder(
    g: &mut DagBuilder,
    mel_input: EdgeId,
    conv1_w: EdgeId,       // [n_state, n_mels, 3]
    conv1_bias: EdgeId,    // [n_state, conv1_out_len]
    conv2_w: EdgeId,       // [n_state, n_state, 3]
    conv2_bias: EdgeId,    // [n_state, n_audio_ctx]
    pos_emb: EdgeId,       // [1, n_audio_ctx, n_state]
    enc_blocks: Vec<EncoderBlockWeights>,
    final_ln_w: Witness,
    final_ln_b: Witness,
    n_head: usize,
    head_dim: usize,
    n_state: usize,
    n_audio_ctx: usize,
) -> EdgeId {
    // Conv1: pad(1,1) → conv1d(k=3, s=1) → add_bias → ReLU
    // Conv1D output: [n_state, 2*n_audio_ctx] (Uint, sf=0)
    let padded1 = g.pad1d(mel_input, 1, 1);
    let conv1_out = g.conv1d(padded1, conv1_w, 3)[0];
    let conv1_out = g.add(conv1_out, conv1_bias)[0];
    let conv1_out = g.relu(conv1_out);

    // Conv2: pad(1,1) → conv1d_strided(k=3, s=2) → add_bias → ReLU
    // Conv1D output: [n_state, n_audio_ctx] (Uint, sf=0)
    let padded2 = g.pad1d(conv1_out, 1, 1);
    let conv2_out = g.conv1d_strided(padded2, conv2_w, 3, 2)[0];
    let conv2_out = g.add(conv2_out, conv2_bias)[0];
    let conv2_out = g.relu(conv2_out);

    // Permute [n_state, n_audio_ctx] → [n_audio_ctx, n_state]
    let transposed = g.einsum("ct->tc".to_string(), vec![conv2_out], false)[0];

    // Reshape to [1, n_audio_ctx, n_state]
    let x = g.change_shape(transposed, vec![1, n_audio_ctx, n_state]);

    // Scale up from Uint,sf=0 to Float,SF_LOG for transformer blocks
    let x = g.scale(x, 0, *SF_LOG)[0];

    // Add sinusoidal positional embedding (Float, SF_LOG)
    let mut x = g.add(x, pos_emb)[0];

    // Encoder transformer blocks
    let num_enc_blocks = enc_blocks.len();
    for (i, bw) in enc_blocks.into_iter().enumerate() {
        let attn_ln_w = g.param(bw.attn_ln_w);
        let attn_ln_b = g.param(bw.attn_ln_b);
        let w_q = g.param(bw.w_q);
        let w_k = g.param(bw.w_k);
        let w_v = g.param(bw.w_v);
        let w_o = g.param(bw.w_o);
        let b_q = g.param(bw.b_q);
        let b_k = g.param(bw.b_k);
        let b_v = g.param(bw.b_v);
        let b_o = g.param(bw.b_o);
        let mlp_ln_w = g.param(bw.mlp_ln_w);
        let mlp_ln_b = g.param(bw.mlp_ln_b);
        let w_mlp1 = g.param(bw.w_mlp1);
        let w_mlp2 = g.param(bw.w_mlp2);
        let b_mlp1 = g.param(bw.b_mlp1);
        let b_mlp2 = g.param(bw.b_mlp2);

        // x = x + self_attn(layer_norm(x))
        let ln_out = g.pipe(&[x], whisper_layer_norm(attn_ln_w, attn_ln_b));
        let attn_out = whisper_self_attention(
            g, ln_out[0], w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o,
            n_head, head_dim, n_state,
        );
        let residual = g.add(attn_out, x)[0];

        // x = x + mlp(layer_norm(x))
        let ln_out = g.pipe(&[residual], whisper_layer_norm(mlp_ln_w, mlp_ln_b));
        let mlp_out = whisper_mlp(g, ln_out[0], w_mlp1, w_mlp2, b_mlp1, b_mlp2);
        x = g.add(mlp_out, residual)[0];

        if i < num_enc_blocks - 1 {
            g.layer_boundaries.push(x);
        }
    }

    // Final layer norm
    let final_ln_w = g.param(final_ln_w);
    let final_ln_b = g.param(final_ln_b);
    let out = g.pipe(&[x], whisper_layer_norm(final_ln_w, final_ln_b));
    out[0]
}

/// Build the text decoder pipeline.
/// Input: pre-embedded decoder [1, S_dec, n_state], encoder output [1, S_enc, n_state]
/// Output: logits [1, S_dec, vocab_size] (or just [1, S_dec, n_state] before final projection)
pub fn whisper_decoder(
    g: &mut DagBuilder,
    dec_input: EdgeId,     // [1, n_text_ctx, n_state]
    encoder_output: EdgeId, // [1, n_audio_ctx, n_state]
    pos_emb: EdgeId,       // [1, n_text_ctx, n_state]
    dec_blocks: Vec<DecoderBlockWeights>,
    final_ln_w: Witness,
    final_ln_b: Witness,
    n_head: usize,
    head_dim: usize,
    n_state: usize,
    n_text_ctx: usize,
    n_audio_ctx: usize,
) -> EdgeId {
    // Add positional embedding
    let mut x = g.add(dec_input, pos_emb)[0];

    // Decoder transformer blocks
    let num_blocks = dec_blocks.len();
    for (i, bw) in dec_blocks.into_iter().enumerate() {
        let attn_ln_w = g.param(bw.attn_ln_w);
        let attn_ln_b = g.param(bw.attn_ln_b);
        let w_q = g.param(bw.w_q);
        let w_k = g.param(bw.w_k);
        let w_v = g.param(bw.w_v);
        let w_o = g.param(bw.w_o);
        let b_q = g.param(bw.b_q);
        let b_k = g.param(bw.b_k);
        let b_v = g.param(bw.b_v);
        let b_o = g.param(bw.b_o);

        let cross_ln_w = g.param(bw.cross_ln_w);
        let cross_ln_b = g.param(bw.cross_ln_b);
        let xw_q = g.param(bw.xw_q);
        let xw_k = g.param(bw.xw_k);
        let xw_v = g.param(bw.xw_v);
        let xw_o = g.param(bw.xw_o);
        let xb_q = g.param(bw.xb_q);
        let xb_k = g.param(bw.xb_k);
        let xb_v = g.param(bw.xb_v);
        let xb_o = g.param(bw.xb_o);

        let mlp_ln_w = g.param(bw.mlp_ln_w);
        let mlp_ln_b = g.param(bw.mlp_ln_b);
        let w_mlp1 = g.param(bw.w_mlp1);
        let w_mlp2 = g.param(bw.w_mlp2);
        let b_mlp1 = g.param(bw.b_mlp1);
        let b_mlp2 = g.param(bw.b_mlp2);

        // x = x + self_attn(layer_norm(x))
        let ln_out = g.pipe(&[x], whisper_layer_norm(attn_ln_w, attn_ln_b));
        let attn_out = whisper_self_attention(
            g, ln_out[0], w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o,
            n_head, head_dim, n_state,
        );
        let residual = g.add(attn_out, x)[0];

        // x = x + cross_attn(layer_norm(x), encoder_output)
        let ln_out = g.pipe(&[residual], whisper_layer_norm(cross_ln_w, cross_ln_b));
        let cross_out = whisper_cross_attention(
            g, ln_out[0], encoder_output,
            xw_q, xw_k, xw_v, xw_o, xb_q, xb_k, xb_v, xb_o,
            n_head, head_dim, n_state, n_text_ctx, n_audio_ctx,
        );
        let residual = g.add(cross_out, residual)[0];

        // x = x + mlp(layer_norm(x))
        let ln_out = g.pipe(&[residual], whisper_layer_norm(mlp_ln_w, mlp_ln_b));
        let mlp_out = whisper_mlp(g, ln_out[0], w_mlp1, w_mlp2, b_mlp1, b_mlp2);
        x = g.add(mlp_out, residual)[0];

        if i < num_blocks - 1 {
            g.layer_boundaries.push(x);
        }
    }

    // Final layer norm
    let final_ln_w = g.param(final_ln_w);
    let final_ln_b = g.param(final_ln_b);
    let out = g.pipe(&[x], whisper_layer_norm(final_ln_w, final_ln_b));
    out[0]
}

/// Build the full Whisper model.
/// Returns the output edge.
pub fn whisper_model(
    g: &mut DagBuilder,
    mel_input: EdgeId,
    dec_input: EdgeId,
    // Encoder conv weights
    conv1_w: EdgeId,
    conv1_bias: EdgeId,
    conv2_w: EdgeId,
    conv2_bias: EdgeId,
    enc_pos_emb: EdgeId,
    // Encoder blocks
    enc_blocks: Vec<EncoderBlockWeights>,
    enc_final_ln_w: Witness,
    enc_final_ln_b: Witness,
    // Decoder weights
    dec_pos_emb: EdgeId,
    dec_blocks: Vec<DecoderBlockWeights>,
    dec_final_ln_w: Witness,
    dec_final_ln_b: Witness,
    // Dimensions
    n_head: usize,
    head_dim: usize,
    n_state: usize,
    n_audio_ctx: usize,
    n_text_ctx: usize,
) -> EdgeId {
    // Encoder
    let encoder_output = whisper_encoder(
        g, mel_input, conv1_w, conv1_bias, conv2_w, conv2_bias, enc_pos_emb,
        enc_blocks, enc_final_ln_w, enc_final_ln_b,
        n_head, head_dim, n_state, n_audio_ctx,
    );

    // Decoder
    whisper_decoder(
        g, dec_input, encoder_output, dec_pos_emb,
        dec_blocks, dec_final_ln_w, dec_final_ln_b,
        n_head, head_dim, n_state, n_text_ctx, n_audio_ctx,
    )
}
