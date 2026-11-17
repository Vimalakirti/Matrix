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

    /// Exact RMS reciprocal advice of the in-circuit mean `x_mean`
    /// (shape [.., 1]); see `rms_reciprocal_gate`.
    pub fn rms_reciprocal_from_mean(&mut self, x_mean: EdgeId) -> EdgeId {
        let inp = self.init_values[x_mean].as_ref().unwrap();
        let (shape, dt, sf) = (inp.shape.clone(), inp.data_type, inp.sf);
        self.init_values.push(Some(Witness::new_wo_data(shape, dt, sf, Role::Auxiliary)));
        self.add_gkr_node(vec![x_mean], BasicBlockType::RMSReciprocal(RMSReciprocal { dim: 0 }))[0]
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

/// There is one deterministic RMS relation; the old environment-controlled
/// tolerance relation is no longer selected by any named model.
pub fn exact_rms() -> bool { true }

/// Scalar exact reciprocal with independently bounded linked intermediates.
/// Epsilon is exactly zero, and a zero statistic deterministically returns zero.
pub(crate) fn rms_reciprocal_gate(g: &mut DagBuilder, x_mean: EdgeId) -> EdgeId {
    g.rms_reciprocal_from_mean(x_mean)
}

pub(crate) fn rms_reciprocal_constraints(g: &mut DagBuilder, x_mean: EdgeId, r: EdgeId) {
    let input=g.init_values[x_mean].as_ref().unwrap();
    let output=g.init_values[r].as_ref().unwrap();
    assert_eq!(input.shape,output.shape,"RMS advice shape mismatch");
    assert_eq!(input.sf,output.sf,"RMS advice scale mismatch");
    assert!(!input.shape.is_empty() && input.shape.iter().all(|&d|d>0),"invalid RMS shape");
    use crate::basicblock::llama::{rms_bounds, RMS_MEAN_MAX};
    let sf=g.init_values[x_mean].as_ref().unwrap().sf;
    let (rmax,umax,pmax)=rms_bounds(sf);
    let shape=g.init_values[x_mean].as_ref().unwrap().shape.clone();
    let len=shape.iter().map(|x| x.next_power_of_two()).product();
    let m=g.change_shape(x_mean,vec![len]);
    let rf=g.change_shape(r,vec![len]);
    g.integer_interval(m,0,RMS_MEAN_MAX);
    g.integer_interval(rf,0,rmax);
    let c=|g: &mut DagBuilder,v| g.integer_literal(vec![len],v,0);
    let mul=|g: &mut DagBuilder,a,b| g.einsum("i,i->i".into(),vec![a,b],false)[0];
    let one=c(g,1); let three=c(g,3); let target=c(g,1i128 << (3*sf));
    let bucket=g.add_gkr_node(vec![m],BasicBlockType::RMSBucket(crate::basicblock::llama::RMSBucket))[0];
    g.init_values.push(Some(Witness::new_wo_data(vec![len,32],DataType::Bool,0,Role::Auxiliary)));
    let bounds=crate::basicblock::llama::rms_bucket_bounds(sf);
    let rsel=rms_select(g,bucket,bounds.iter().map(|b|b.2).collect());
    let usel=rms_select(g,bucket,bounds.iter().map(|b|b.3).collect());
    let rd=g.sub(rsel,rf)[0];g.integer_interval(rd,0,rmax);
    // ReLUHelper(v-1) supplies the zero flag on nonnegative integer v.
    // Its hint is independently defined by three range/equality constraints.
    let zero_flag=|g: &mut DagBuilder,v| {
        let vm1=g.sub(v,one)[0];
        let z=g.add_gkr_node(vec![vm1],BasicBlockType::ReLUHelper(crate::basicblock::relu::ReLUHelper))[0];
        g.init_values.push(Some(Witness::new_wo_data(vec![len],DataType::Bool,0,Role::Output)));
        g.integer_interval(z,0,1);
        let zv=mul(g,z,v); g.integer_interval(zv,0,0);
        let vpz=g.add(v,z)[0]; g.integer_interval(vpz,1,RMS_MEAN_MAX.max(rmax)+1);
        z
    };
    let z=zero_flag(g,m); let b=zero_flag(g,rf);
    let zr=mul(g,z,rf); g.integer_interval(zr,0,0);
    let u=mul(g,m,rf); g.integer_interval(u,0,umax);
    let ud=g.sub(usel,u)[0];g.integer_interval(ud,0,umax);
    let t=mul(g,u,rf);
    // h=RD(m-3,4)=floor((m-1)/4); ceil(m/4)=h+1, even at m=0.
    let mm3=g.sub(m,three)[0];let h=g.div_const(mm3,4)[0];
    let ct=g.sub(target,t)[0];let ctu=g.add(ct,u)[0];
    let dl=g.sub(ctu,h)[0];let dl=g.sub(dl,one)[0];
    let tu=g.add(t,u)[0];let tuh=g.add(tu,h)[0];let dh=g.sub(tuh,target)[0];
    let active_low=g.sub(one,b)[0]; let active_high=g.sub(one,z)[0];
    let dl=mul(g,active_low,dl); let dh=mul(g,active_high,dh);
    g.integer_interval(dl,0,pmax);
    g.integer_interval(dh,0,pmax);

}

/// Select from an immutable public table without publishing the private bucket.
fn rms_select(g:&mut DagBuilder,bucket:EdgeId,values:Vec<i128>)->EdgeId {
    let table=g.public_literal(Witness::new(vec![32],values.into_iter().map(crate::util::arith::int_to_f).collect(),DataType::Int,0,Role::Input));
    g.einsum("ij,j->i".into(),vec![bucket,table],false)[0]
}

pub(crate) fn rms_bucket_constraints(g:&mut DagBuilder,m:EdgeId,bucket:EdgeId) {
    use crate::basicblock::llama::{rms_bucket_bounds,RMS_MEAN_MAX};
    let input=g.init_values[m].as_ref().unwrap();
    assert_eq!(input.shape.len(),1,"bucket input must be a flat row tensor");
    let len=input.shape[0];let sf=input.sf;
    assert!(len.is_power_of_two());
    assert_eq!(g.init_values[bucket].as_ref().unwrap().shape,vec![len,32]);
    assert_eq!(g.init_values[bucket].as_ref().unwrap().sf,0);
    g.integer_interval(m,0,RMS_MEAN_MAX);
    g.integer_interval(bucket,0,1);
    let sum=g.einsum("ij->i".into(),vec![bucket],false)[0];g.integer_interval(sum,1,1);
    let bounds=rms_bucket_bounds(sf);
    let lo=rms_select(g,bucket,bounds.iter().map(|b|b.0).collect());
    let hi=rms_select(g,bucket,bounds.iter().map(|b|b.1).collect());
    let a=g.sub(m,lo)[0];let b=g.sub(hi,m)[0];
    g.integer_interval(a,0,RMS_MEAN_MAX+1);g.integer_interval(b,0,RMS_MEAN_MAX+1);
}

/// Bound and mask the RMS input before squaring. Bounding the square's
/// ScaleDown quotient makes this local rescale an integer rounding function;
/// this does not certify unrelated ScaleDown instances elsewhere in the DAG.
pub(crate) fn rms_square(g: &mut DagBuilder, x: EdgeId) -> EdgeId {
    use crate::basicblock::llama::{floor_sqrt,rms_bounds,RMS_MEAN_MAX};
    let sf=g.init_values[x].as_ref().unwrap().sf;
    let _=rms_bounds(sf);
    let shape=g.init_values[x].as_ref().unwrap().shape.clone();
    assert!((1..=(1<<20)).contains(shape.last().unwrap()), "unsupported RMS reduction dimension");
    let x=g.mask(x,shape);
    let bound=floor_sqrt(RMS_MEAN_MAX*(1i128<<sf));
    g.integer_interval(x,-bound,bound);
    let sq=g.einsum("bsi,bsi->bsi".into(),vec![x,x],true)[0];
    g.integer_interval(sq,0,RMS_MEAN_MAX);
    sq
}

/// The two RMS output rescales are bounded locally as well; unrelated graph
/// rescales remain outside this normalization certificate.
pub(crate) fn rms_apply(g: &mut DagBuilder,x: EdgeId,r: EdgeId,w: EdgeId) -> EdgeId {
    use crate::basicblock::llama::{floor_sqrt,rms_bounds,RMS_MEAN_MAX};
    let sf=g.init_values[x].as_ref().unwrap().sf;
    assert_eq!(g.init_values[w].as_ref().unwrap().sf,sf,"RMS gain scale mismatch");
    let (rmax,_,_)=rms_bounds(sf); let q=1i128<<sf;
    let xmax=floor_sqrt(RMS_MEAN_MAX*q);
    // Mean/square roots also imply |h|<2^31 across s<=20,n<=2^20.
    // See the profile proof; this extra bound is independent of final output.
    let hmax=((xmax*rmax+q-1)/q).min((1i128<<31)-1);
    let wmax=(1i128<<30)-1;
    g.integer_interval(w,-wmax,wmax);
    let h=g.einsum("bsi,bs->bsi".into(),vec![x,r],true)[0];
    g.integer_interval(h,-hmax,hmax);
    let out=g.einsum("bsi,i->bsi".into(),vec![h,w],true)[0];
    let ymax=(hmax*wmax+q-1)/q;
    g.integer_interval(out,-ymax,ymax);
    out
}

pub fn llama_rms_norm(
    w_e: EdgeId,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This custom RMSNorm layer expects 1 input");
        let x = x[0];
        let shape=g.init_values[x].as_ref().unwrap().shape.clone();
        let x=g.mean_input(x,*shape.last().unwrap());
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let seq = x_shape[1];
        let n = x_shape[x_shape.len() - 1];

        let x_sq = rms_square(g,x);
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0];
        let x_mean = g.div_const(x_sum, n)[0];
        let x_mean = g.change_shape(x_mean, vec![1, seq, 1]);
        // div_const emits exact deterministic rounding roots.

        let r = rms_reciprocal_gate(g, x_mean);

        // Compute RMSNorm
        let r = g.change_shape(r, vec![1, seq]);
        let out = rms_apply(g,x,r,w_e);
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
pub(crate) fn llama3_rms_norm(
    w_e: EdgeId,
    seq: usize,
) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "RMSNorm expects 1 input");
        let x = x[0];
        let shape=g.init_values[x].as_ref().unwrap().shape.clone();
        let x=g.mean_input(x,*shape.last().unwrap());
        let x_shape = g.init_values[x].as_ref().unwrap().shape.clone();
        let n = x_shape[x_shape.len() - 1];

        let x_sq = rms_square(g,x);
        let x_sum = g.einsum("bsi->bs".to_string(), vec![x_sq], false)[0];
        let x_mean = g.div_const(x_sum, n)[0];
        let x_mean = g.change_shape(x_mean, vec![1, seq, 1]);
        // div_const emits exact deterministic rounding roots.

        let r = rms_reciprocal_gate(g, x_mean);

        let r = g.change_shape(r, vec![1, seq]);
        let out = rms_apply(g,x,r,w_e);
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
