//! Canonical PUBLIC statement codec, version 2. This binds a verifier-trusted
//! DAG; it is NOT a proof that a compiler implements the intended model.
//! Private tensor payloads, advice, and Coins never enter this module. Fixed
//! private model values must be identified separately by pinned commitments.
//! All lengths and usize values are checked; sequences are length delimited.
//! Operator tags/field order and IEEE-754 finite f64 bit semantics are versioned.
use crate::basicblock::BasicBlockType;
use crate::dag::{DataType, PolyType, Role};
use super::{keys::KeySeed, params::Profile, piop::Graph};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;

#[derive(Default)]
struct Words(Vec<u64>);
impl Words {
    fn u(&mut self, x: u64) { self.0.push(x); }
    fn us(&mut self, x: usize) -> Result<(), String> {
        self.u(u64::try_from(x).map_err(|_| "integer exceeds u64")?); Ok(())
    }
    fn bytes(&mut self, xs: &[u8]) -> Result<(), String> {
        self.us(xs.len())?; for &x in xs { self.u(u64::from(x)); } Ok(())
    }
    fn usizes(&mut self, xs: &[usize]) -> Result<(), String> {
        self.us(xs.len())?; for &x in xs { self.us(x)?; } Ok(())
    }
    fn nested(&mut self, xs: &[Vec<usize>]) -> Result<(), String> {
        self.us(xs.len())?; for x in xs { self.usizes(x)?; } Ok(())
    }
    fn pairs(&mut self, xs: &[(usize, usize)]) -> Result<(), String> {
        self.us(xs.len())?; for &(a,b) in xs { self.us(a)?; self.us(b)?; } Ok(())
    }
    fn field_values(&mut self, xs: &[u64]) -> Result<(), String> {
        if xs.iter().any(|&v| v >= Q) { return Err("noncanonical public field value".into()); }
        self.us(xs.len())?; self.0.extend_from_slice(xs); Ok(())
    }
}

fn operator(k: &BasicBlockType) -> Result<Vec<u64>, String> {
    use BasicBlockType::*;
    let mut w = Words::default();
    match k {
        Add(_) => { w.u(1); },
        Sub(_) => { w.u(2); },
        ChangeShape(b) => { w.u(3); w.usizes(&b.target_shape)?; },
        ChannelSlice(b) => { w.u(4); w.us(b.channels_in)?; w.us(b.channels_out)?; w.us(b.channel_start)?; w.usizes(&b.spatial_dims)?; },
        Concat(b) => { w.u(5); w.us(b.channels_a)?; w.usizes(&b.spatial_dims)?; },
        Conv1D(b) => { w.u(6); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_size)?; w.us(b.input_len)?; w.us(b.s_in)?; w.us(b.s_kernel)?; w.us(b.l_out)?; w.us(b.conv_stride)?; },
        Conv2D(b) => { w.u(7); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.stride_w)?; w.us(b.s_in)?; w.us(b.s_kernel)?; w.us(b.h_out)?; w.us(b.w_out)?; w.us(b.conv_stride_h)?; w.us(b.conv_stride_w)?; w.us(b.dilation_h)?; w.us(b.dilation_w)?; w.us(b.batch)?; },
        Conv3D(b) => { w.u(8); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_d)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.input_d)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.stride_h)?; w.us(b.stride_w)?; w.us(b.s_in)?; w.us(b.s_kernel)?; w.us(b.d_out)?; w.us(b.h_out)?; w.us(b.w_out)?; w.us(b.conv_stride_d)?; w.us(b.conv_stride_h)?; w.us(b.conv_stride_w)?; },
        ConvTranspose1D(b) => { w.u(9); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_size)?; w.us(b.input_len)?; w.us(b.stride)?; w.us(b.s_in)?; w.us(b.l_out)?; },
        ConvTranspose2D(b) => { w.u(10); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.stride_h)?; w.us(b.stride_w)?; w.us(b.flat_stride)?; w.us(b.s_kernel)?; w.us(b.h_out)?; w.us(b.w_out)?; },
        ConvTranspose3D(b) => { w.u(11); w.us(b.c_in)?; w.us(b.c_out)?; w.us(b.kernel_d)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.input_d)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.stride_d)?; w.us(b.stride_h)?; w.us(b.stride_w)?; w.us(b.flat_stride_h)?; w.us(b.flat_stride_w)?; w.us(b.s_kernel)?; w.us(b.d_out)?; w.us(b.h_out)?; w.us(b.w_out)?; },
        DepthwiseConv2D(b) => { w.u(12); w.us(b.channels)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.stride_w)?; w.us(b.s_in)?; w.us(b.s_kernel)?; w.us(b.h_out)?; w.us(b.w_out)?; w.us(b.conv_stride_h)?; w.us(b.conv_stride_w)?; },
        DivConst(b) => { w.u(13); w.u(b.divisor); },
        Einsum(b) => { w.u(14); w.bytes(b.equation.as_bytes())?; w.nested(&b.input_shapes)?; w.usizes(&b.output_shape)?; let (perms, rounds) = b.public_schedule(); w.us(perms.len())?; for p in perms { w.pairs(p)?; } w.us(rounds)?; },
        FlattenKernel(b) => { w.u(15); w.us(b.s_w)?; w.us(b.kh)?; w.us(b.kw)?; w.us(b.c_out)?; w.us(b.c_in)?; w.us(b.dilation_h)?; w.us(b.dilation_w)?; },
        FlattenKernel3D(b) => { w.u(16); w.us(b.stride_h)?; w.us(b.stride_w)?; w.us(b.kd)?; w.us(b.kh)?; w.us(b.kw)?; w.us(b.c_out)?; w.us(b.c_in)?; },
        ExpHelper(b) => {
            if !(1..=8).contains(&b.num_bits) || *crate::SF_LOG > 30 { return Err("unsupported exp profile".into()); }
            w.u(17); w.us(b.num_bits)?;
            w.u(crate::basicblock::exp::exp_ln2(*crate::SF_LOG) as u64);
            w.bytes(b"canonical-nonpositive-unclipped-exp-v1")?;
        },
        GatherFromGrid(b) => { w.u(18); w.us(b.n_points)?; w.us(b.channels)?; w.us(b.grid_h)?; w.us(b.grid_w)?; },
        GeneralMaxPoolHelper(b) => { w.u(19); w.us(b.channels)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.kernel_h)?; w.us(b.kernel_w)?; w.us(b.stride_h)?; w.us(b.stride_w)?; },
        InstanceNorm3D(b) => { w.u(20); w.us(b.channels)?; w.us(b.depth)?; w.us(b.height)?; w.us(b.width)?; if !b.eps.is_finite() { return Err("nonfinite InstanceNorm3D.eps".into()); } w.u(b.eps.to_bits()); },
        MaxPoolHelper(b) => { w.u(21); w.us(b.channels)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.pool_h)?; w.us(b.pool_w)?; },
        NonNegative(b) => { w.u(22); w.us(b.table_size_log)?; },
        Permute(b) => { w.u(23); w.pairs(&b.ranges)?; },
        PillarMaxPool(b) => { w.u(24); w.us(b.n_pillars)?; w.us(b.max_points)?; w.us(b.features)?; },
        ProductZeroCheck(_) => { w.u(25); },
        Reducer(_) => { w.u(26); },
        ReLUHelper(_) => { w.u(27); },
        Replicate2x2(b) => { w.u(28); w.us(b.channels)?; w.us(b.out_h)?; w.us(b.out_w)?; },
        RMSReciprocal(b) => { w.u(29); w.us(b.dim)?; },
        RMSBucket(_) => { w.u(40); w.bytes(b"private-dyadic-rms-bucket-v1")?; },
        ScaleDown(b) => { w.u(30); w.us(b.output_sf)?; },
        ScaleUp(b) => { w.u(31); w.us(b.output_sf)?; },
        ScatterToBEV(b) => { w.u(32); w.us(b.n_pillars)?; w.us(b.features)?; w.us(b.ny)?; w.us(b.nx)?; },
        SigmoidConst(b) => { w.u(33); w.us(b.segments)?; },
        SoftmaxConst(b) => { w.u(34); w.us(b.dim)?; },
        SubSample2D(b) => { w.u(35); w.us(b.channels)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.out_h)?; w.us(b.out_w)?; w.us(b.stride_h)?; w.us(b.stride_w)?; w.us(b.offset_h)?; w.us(b.offset_w)?; },
        TwoPow(_) => { w.u(36); },
        ZeroPad(b) => { w.u(37); w.us(b.channels)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.pad_h)?; w.us(b.pad_w)?; },
        ZeroPad3D(b) => { w.u(38); w.us(b.channels)?; w.us(b.input_d)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.pad_d)?; w.us(b.pad_h)?; w.us(b.pad_w)?; },
        ZeroPadAsym(b) => { w.u(39); w.us(b.channels)?; w.us(b.input_h)?; w.us(b.input_w)?; w.us(b.pad_h_top)?; w.us(b.pad_h_bottom)?; w.us(b.pad_w_left)?; w.us(b.pad_w_right)?; },
    }
    Ok(w.0)
}

/// Public protocol setup identity (no private randomness).
fn setup(w: &mut Words, p: &Profile, seed: &KeySeed, b0: usize, defer: bool, carry_log: usize) -> Result<(), String> {
    w.bytes(b"morpheus-public-statement")?; w.u(2);
    w.bytes(p.name.as_bytes())?;
    w.usizes(&[p.rank,p.tau,p.hiding,p.source_payload_cols,p.aux_coeffs,p.deg_old,p.deg_final])?;
    w.u(p.sigma); w.u(p.b_z); for v in p.raw_noise { w.u(v); }
    w.u(p.rej_m); w.usizes(&[p.attempts,p.salt_bits,p.n_max])?; w.u(u64::from(p.s_max_log2));
    w.u(Q); w.usizes(&[super::ring::N,super::ring::DS,super::ring::T,super::ring::LOG_N,super::params::A_ENC_COLS])?;
    // Both rings are negacyclic, extension polynomial is Theta^8 - W.
    w.u(1); w.u(1); w.u(8); w.u(super::field8::W);
    w.us(super::ring::NTT_PRIMES.len())?;
    for (p,r) in super::ring::NTT_PRIMES { w.u(p); w.u(r); }
    w.bytes(b"chacha8-public-keys-v1")?;
    for s in seed.0 { w.u(u64::from(s)); }
    w.us(b0)?; w.u(u64::from(defer)); w.us(carry_log)?;
    Ok(())
}

fn encode(g: &Graph, p: &Profile, seed: &KeySeed, b0: usize, defer: bool, carry_log: usize, values: bool) -> Result<Vec<u64>, String> {
    let d = g.dag; let ne = d.num_edges; let nn = d.nodes.len();
    if [g.shapes.len(),g.nvars.len(),g.committed.len(),g.model.len(),g.bool_typed.len(),g.edge_meta.len(),d.consumers.len(),d.producers.len(),d.edge_aliases.len()].iter().any(|&n| n != ne) {
        return Err("edge descriptor length mismatch".into());
    }
    if d.topo.len() != nn { return Err("topological order length mismatch".into()); }
    let mut pos = vec![None; nn];
    for (i,&n) in d.topo.iter().enumerate() {
        if n >= nn || pos[n].replace(i).is_some() { return Err("invalid topological order".into()); }
    }
    for (i,n) in d.nodes.iter().enumerate() {
        if n.id != i || n.inputs.iter().chain(&n.outputs).any(|&e| e >= ne) { return Err("invalid node wiring".into()); }
        for &e in &n.inputs {
            if let Some(prod) = d.producers[e] {
                if prod >= nn || pos[prod] >= pos[i] { return Err("non-topological producer".into()); }
            }
            if !d.consumers[e].contains(&i) { return Err("consumer wiring mismatch".into()); }
        }
        for &e in &n.outputs { if d.producers[e] != Some(i) { return Err("producer wiring mismatch".into()); } }
    }
    for e in 0..ne {
        if let Some(n) = d.producers[e] { if n >= nn || !d.nodes[n].outputs.contains(&e) { return Err("invalid producer".into()); } }
        if d.consumers[e].iter().any(|&n| n >= nn || !d.nodes[n].inputs.contains(&e)) { return Err("invalid consumer".into()); }
    }
    if d.range.iter().chain(&d.two_pow).any(|&n| n >= nn)
        || d.topo_levels.iter().flatten().any(|&n| n >= nn) {
        return Err("invalid auxiliary node index".into());
    }
    let na = d.alias_to_edge.len();
    if d.alias_to_consumer.len() != na || d.alias_input_slot.len() != na {
        return Err("alias descriptor length mismatch".into());
    }
    let mut aliases_seen = vec![false; na];
    for (e, aliases) in d.edge_aliases.iter().enumerate() {
        for alias in aliases {
            let a = alias.0;
            if a >= na || aliases_seen[a] || d.alias_to_edge[a] != e {
                return Err("invalid or duplicate edge alias".into());
            }
            aliases_seen[a] = true;
        }
    }
    for a in 0..na {
        let n = d.alias_to_consumer[a]; let slot = d.alias_input_slot[a];
        if !aliases_seen[a] || n >= nn || d.nodes[n].inputs.get(slot) != Some(&d.alias_to_edge[a]) {
            return Err("alias wiring mismatch".into());
        }
    }
    if g.cfg.tcl == 0 || g.cfg.tcl >= usize::BITS as usize || g.cfg.slots_log >= usize::BITS as usize || g.cfg.req_planes_log >= usize::BITS as usize {
        return Err("invalid graph configuration width".into());
    }
    g.validate_exp_profile()?;
    let mut w = Words::default(); setup(&mut w,p,seed,b0,defer,carry_log)?;
    w.u(u64::from(values));
    w.usizes(&[g.cfg.tcl,g.cfg.slots_log,g.cfg.req_planes_log])?;
    w.u(u64::from(g.protocol_conv_pp));
    // SF_FLOAT currently evaluates an i32 shift; keep its positive finite domain.
    if *crate::SF_LOG > 30 { return Err("fixed point scale width (supported: 0..=30)".into()); }
    w.us(*crate::SF_LOG)?; w.u(u64::from(crate::SF_FLOAT.to_bits()));
    w.us(ne)?; w.us(nn)?;
    for n in &d.nodes {
        w.us(n.id)?; let op = operator(&n.kind)?; w.us(op.len())?; w.0.extend(op);
        w.usizes(&n.inputs)?; w.usizes(&n.outputs)?;
    }
    w.us(d.exp_guard_nodes.len())?;
    for (&n,ids) in &d.exp_guard_nodes { w.us(n)?; w.usizes(ids)?; }
    w.usizes(&d.topo)?; w.nested(&d.topo_levels)?;
    w.usizes(&d.range)?; w.usizes(&d.two_pow)?;
    w.nested(&d.consumers)?;
    w.us(d.producers.len())?;
    for n in &d.producers { match n { None => w.u(0), Some(n) => { w.u(1); w.us(*n)?; } } }
    for edges in [&d.input_ports,&d.output_ports,&d.layer_boundaries,&d.boundary_edges,&d.crossing_edges] {
        if edges.iter().any(|&e| e >= ne) { return Err("invalid port/boundary edge".into()); } w.usizes(edges)?;
    }
    let mut claims: Vec<_> = d.self_claim_edges.iter().copied().collect(); claims.sort_unstable();
    if claims.iter().any(|&e| e >= ne) { return Err("invalid self claim edge".into()); } w.usizes(&claims)?;
    w.us(d.edge_aliases.len())?;
    for a in &d.edge_aliases { w.us(a.len())?; for x in a { w.us(x.0)?; } }
    w.usizes(&d.alias_to_edge)?; w.usizes(&d.alias_to_consumer)?; w.usizes(&d.alias_input_slot)?;
    for e in 0..ne {
        w.usizes(&g.shapes[e])?;
        let mut padded = Vec::new(); let mut nv = 0usize;
        for &s in &g.shapes[e] {
            let x = s.max(1).checked_next_power_of_two().ok_or("shape padding overflow")?;
            nv = nv.checked_add(x.trailing_zeros() as usize).ok_or("shape rank overflow")?; padded.push(x);
        }
        if nv != g.nvars[e] || nv >= usize::BITS as usize { return Err("inconsistent padded shape".into()); }
        w.usizes(&padded)?; w.us(nv)?;
        let (dt,pt,sf,role) = g.edge_meta[e];
        w.u(match dt { DataType::Uint=>1,DataType::Int=>2,DataType::Bool=>3,DataType::Float=>4 });
        w.u(match pt { PolyType::Dense=>1,PolyType::Sparse=>2 }); w.us(sf)?;
        w.u(match role { Role::Auxiliary=>1,Role::Constant=>2,Role::Input=>3,Role::Output=>4 });
        if g.bool_typed[e] != (d.producers[e].is_none() && dt == DataType::Bool) { return Err("boolean type mismatch".into()); }
        if g.model[e] != (role == Role::Constant) { return Err("model role mismatch".into()); }
        for flag in [g.committed[e],g.model[e],g.bool_typed[e]] { w.u(u64::from(flag)); }
    }
    w.us(g.lookups.len())?;
    for (&n,&bits) in &g.lookups { if n >= nn || bits >= usize::BITS as usize { return Err("invalid lookup descriptor".into()); } w.us(n)?; w.us(bits)?; }
    // Registry membership and VALUES belong to program identity, unlike request
    // inputs. Refuse even request-time replacement of an immutable literal.
    w.us(d.program_literals.len())?;
    for (&e, v) in &d.program_literals {
        if e >= ne || d.producers[e].is_some() || g.edge_meta[e].3 != Role::Input
            || g.public.get(&e).map(|x| x.as_slice()) != Some(v.as_slice()) {
            return Err("program literal missing, overwritten, or not an input leaf".into());
        }
        public_array(&mut w,g,e,v,true)?;
    }
    let mut public: Vec<_> = g.public.iter().collect(); public.sort_unstable_by_key(|(e,_)| **e);
    w.us(public.len())?;
    for (&e,v) in public { public_array(&mut w,g,e,v,values)?; }
    // Output order is meaningful (root-claim challenge schedule); preserve it.
    w.us(g.outputs.len())?; let mut seen = std::collections::HashSet::new();
    for (e,v) in &g.outputs {
        if !seen.insert(*e) { return Err("duplicate claimed output".into()); }
        public_array(&mut w,g,*e,v,values)?;
    }
    // MTranscript absorbs field elements. Split EVERY raw word into two
    // 32-bit limbs so values >= q cannot alias after field reduction.
    Ok(w.0.into_iter().flat_map(|v| [v & 0xffff_ffff, v >> 32]).collect())
}

fn public_array(w: &mut Words, g: &Graph, e: usize, v: &[u64], values: bool) -> Result<(), String> {
    if e >= g.nvars.len() || v.len() != 1usize.checked_shl(u32::try_from(g.nvars[e]).map_err(|_| "public shape width")?).ok_or("public shape overflow")? {
        return Err("public array shape mismatch".into());
    }
    if v.iter().any(|&v| v >= Q) { return Err("noncanonical public field value".into()); }
    w.us(e)?; if values { w.field_values(v)?; } else { w.us(v.len())?; } Ok(())
}

/// Encode the complete public request statement before any challenge.
pub fn words(g: &Graph, p: &Profile, seed: &KeySeed, b0: usize, defer: bool, carry_log: usize) -> Result<Vec<u64>, String> {
    encode(g,p,seed,b0,defer,carry_log,true)
}

/// Stable public graph/setup identity, excluding request input/output VALUES.
/// The caller must additionally pin the model commitments and source layout.
pub fn model_words(g: &Graph, p: &Profile, seed: &KeySeed, b0: usize, defer: bool, carry_log: usize) -> Result<Vec<u64>, String> {
    encode(g,p,seed,b0,defer,carry_log,false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basicblock::{ChangeShape, DivConst, Permute};
    use crate::dag::{DagBuilder, Witness};
    use super::super::piop::Config;
    use std::sync::Arc;

    fn encoded(g: &Graph) -> Vec<u64> {
        words(g,&Profile::toy(),&KeySeed([1;8]),11,false,4).unwrap()
    }
    #[test]
    fn typed_operator_fields_are_not_debug_or_product_only() {
        let a = BasicBlockType::ChangeShape(ChangeShape { target_shape: vec![2,4] });
        let b = BasicBlockType::ChangeShape(ChangeShape { target_shape: vec![4,2] });
        assert_ne!(operator(&a).unwrap(),operator(&b).unwrap());
        let a = BasicBlockType::Permute(Permute { ranges: vec![(0,1),(1,3)] });
        let b = BasicBlockType::Permute(Permute { ranges: vec![(1,3),(0,1)] });
        assert_ne!(operator(&a).unwrap(),operator(&b).unwrap());
        assert_ne!(operator(&BasicBlockType::DivConst(DivConst { divisor: 1 })).unwrap(),
                   operator(&BasicBlockType::DivConst(DivConst { divisor: Q+1 })).unwrap());
    }
    #[test]
    fn public_values_map_order_shapes_roles_and_outputs_bound() {
        let mut b = DagBuilder::new(); b.input(vec![2,2],DataType::Uint); b.input(vec![2,2],DataType::Uint);
        let (d, wit) = b.compile(); let mut g = Graph::new(&d,&wit,Config::default());
        g.public.insert(0,Arc::new(vec![1,2,3,4])); g.public.insert(1,Arc::new(vec![4,3,2,1]));
        let base = encoded(&g);
        let a=g.public.remove(&0).unwrap(); let b=g.public.remove(&1).unwrap();
        g.public.insert(1,b); g.public.insert(0,a); assert_eq!(base,encoded(&g));
        g.public.insert(0,Arc::new(vec![1,2,4,3])); assert_ne!(base,encoded(&g));
        g.public.insert(0,Arc::new(vec![1,2,3,4]));
        g.shapes[0]=vec![1,4]; assert_ne!(base,encoded(&g)); g.shapes[0]=vec![2,2];
        g.edge_meta[0].3=Role::Auxiliary; assert_ne!(base,encoded(&g)); g.edge_meta[0].3=Role::Input;
        g.edge_meta[0].0=DataType::Int; assert_ne!(base,encoded(&g)); g.edge_meta[0].0=DataType::Uint;
        g.outputs=vec![(0,Arc::new(vec![1,2,3,4])),(1,Arc::new(vec![4,3,2,1]))];
        let ordered=encoded(&g); g.outputs.reverse(); assert_ne!(ordered,encoded(&g));
        g.public.insert(0,Arc::new(vec![Q,2,3,4]));
        assert!(words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    }
    #[test]
    fn private_payload_is_not_exported_and_raw_u64s_do_not_alias_mod_q() {
        use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;
        let make = |v| {
            let mut b=DagBuilder::new();
            b.param(Witness::new(vec![2],vec![F(v),F(v)],DataType::Uint,0,Role::Constant));
            b.compile()
        };
        let (d1,w1)=make(17); let (d2,w2)=make(981);
        let g1=Graph::new(&d1,&w1,Config::default()); let g2=Graph::new(&d2,&w2,Config::default());
        assert_eq!(encoded(&g1),encoded(&g2));
        let mut p=Profile::toy(); p.sigma=1;
        let x=words(&g1,&p,&KeySeed([1;8]),11,false,4).unwrap(); p.sigma=Q+1;
        let y=words(&g1,&p,&KeySeed([1;8]),11,false,4).unwrap();
        assert_ne!(x,y); assert!(x.iter().chain(&y).all(|&v| v<=u32::MAX as u64));
    }
    #[test]
    fn model_identity_allows_new_request_values_but_binds_metadata() {
        let mut b = DagBuilder::new(); b.input(vec![2,2],DataType::Uint);
        let (d,wit) = b.compile(); let mut g = Graph::new(&d,&wit,Config::default());
        g.public.insert(0,Arc::new(vec![1,2,3,4]));
        g.outputs = vec![(0,Arc::new(vec![4,3,2,1]))];
        let model = |g: &Graph| model_words(g,&Profile::toy(),&KeySeed([1;8]),11,true,4).unwrap();
        let registered = model(&g); let request = encoded(&g);
        g.public.insert(0,Arc::new(vec![5,6,7,8]));
        g.outputs = vec![(0,Arc::new(vec![8,7,6,5]))];
        assert_eq!(registered,model(&g)); assert_ne!(request,encoded(&g));
        g.shapes[0] = vec![1,4]; assert_ne!(registered,model(&g)); g.shapes[0] = vec![2,2];
        g.edge_meta[0].3 = Role::Auxiliary; assert_ne!(registered,model(&g)); g.edge_meta[0].3 = Role::Input;
        g.cfg.tcl += 1; assert_ne!(registered,model(&g));
    }
    #[test]
    fn program_literal_values_frozen_but_private_weights_remain_hidden() {
        use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;
        let make = |literal, weight| {
            let mut b=DagBuilder::new();
            b.public_literal(Witness::new(vec![2],vec![F(literal);2],DataType::Uint,0,Role::Constant));
            b.param(Witness::new(vec![2],vec![F(weight);2],DataType::Uint,0,Role::Constant));
            b.compile()
        };
        let (d1,w1)=make(17,99); let (d2,w2)=make(18,99); let (d3,w3)=make(17,100);
        let g1=Graph::new(&d1,&w1,Config::default());
        let g2=Graph::new(&d2,&w2,Config::default());
        let g3=Graph::new(&d3,&w3,Config::default());
        let id=|g:&Graph| model_words(g,&Profile::toy(),&KeySeed([1;8]),11,true,4).unwrap();
        assert_ne!(id(&g1),id(&g2)); assert_eq!(id(&g1),id(&g3));
        assert!(g1.public.contains_key(&0)); assert!(!g1.public.contains_key(&1));
        assert!(!g1.model[0]); assert!(g1.model[1]);
        let mut g=Graph::new(&d1,&w1,Config::default()); g.public.remove(&0);
        assert!(words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    }
    #[test]
    fn reject_nonfinite_public_parameter() {
        use crate::basicblock::InstanceNorm3D;
        let k=BasicBlockType::InstanceNorm3D(InstanceNorm3D { channels:1,depth:1,height:1,width:1,eps:f64::NAN });
        assert!(operator(&k).is_err());
    }
}
