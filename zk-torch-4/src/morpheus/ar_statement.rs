//! Public AR metadata for a trusted, locally constructed DAG.
//!
//! This validates descriptor values, public selectors, shift, and logit bounds;
//! it is NOT a canonical-program validator. The verifier must independently pin
//! the intended DAG (including embedding/rank wiring, causal masks and numerical
//! operators). A caller-supplied arbitrary DAG is outside this API's guarantee.
//! Callers must explicitly select the AR relation; `ar: None` proves no AR claim.
use super::piop::Graph;

#[derive(Clone, Debug)]
pub struct ArStatement {
    pub prompt_len: usize,
    pub seq: usize,
    pub vocab: usize,
    pub embedding: usize,
    pub selectors: Vec<usize>,
    pub ranges: Vec<(usize, usize)>,
    pub logits: Vec<usize>,
    pub rank_constants: Vec<[usize;3]>,
    pub logit_bound: u64,
}

impl ArStatement {
    pub fn words(&self) -> Vec<u64> {
        // Version 1 declares lowest-index argmax and contiguous positions 0..seq.
        let mut w=vec![1,self.prompt_len as u64,self.seq as u64,self.vocab as u64,self.embedding as u64,self.selectors.len() as u64];
        for (&e,&(s,n)) in self.selectors.iter().zip(&self.ranges) { w.extend([e as u64,s as u64,n as u64]); }
        w.push(self.logit_bound);
        w.extend(self.logits.iter().map(|&e|e as u64));
        for ids in &self.rank_constants { w.extend(ids.iter().map(|&e|e as u64)); }
        w
    }
    pub fn validate(&self,g: &Graph) -> bool {
        if self.seq==0 || self.vocab==0 || self.prompt_len==0 || self.prompt_len>self.seq || self.selectors.len()!=self.ranges.len() || self.selectors.is_empty() { return false; }
        let mut end=0usize;
        for &(start,len) in &self.ranges {
            if start!=end || len==0 { return false; }
            end=match end.checked_add(len) { Some(n)=>n,None=>return false };
        }
        if end!=self.vocab || self.logits.len()!=self.ranges.len() || self.rank_constants.len()!=self.ranges.len() { return false; }
        use crate::basicblock::BasicBlockType;
        use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;
        let bp=match self.logit_bound.checked_add(1) { Some(v)=>v,None=>return false };
        if !bp.is_power_of_two() || bp<2 || bp >= 1u64<<62 { return false; }
        let bound_bits=bp.trailing_zeros() as usize+1;
        let q=ALMOST_GOLDILOCKS_PRIME as u128;
        let rm=(1u128<<bound_bits)-1;
        if 2*rm>=q { return false; }
        let d=match (self.logit_bound as u128).checked_mul(2).and_then(|n|n.checked_mul(self.vocab as u128)).and_then(|n|n.checked_add(self.vocab as u128-1)) { Some(v)=>v,None=>return false };
        let k=(u128::BITS-d.leading_zeros()) as usize;
        if k>=64 || d+((1u128<<k)-1)>=q { return false; }
        // Check the signed-bound roots refer to THESE public bound sources
        // and THESE logits, with exactly the declared range width.
        for (&logit,ids) in self.logits.iter().zip(&self.rank_constants) {
            let range_on=|out:usize| g.dag.nodes.iter().any(|n| n.inputs==vec![out] && matches!(&n.kind, BasicBlockType::NonNegative(nn) if nn.table_size_log==bound_bits));
            let lower=g.dag.nodes.iter().find(|n| n.inputs==vec![logit,ids[0]] && matches!(&n.kind,BasicBlockType::Add(_)));
            let upper=g.dag.nodes.iter().find(|n| n.inputs==vec![ids[0],logit] && matches!(&n.kind,BasicBlockType::Sub(_)));
            if !lower.and_then(|n|n.outputs.first()).map(|&e|range_on(e)).unwrap_or(false) || !upper.and_then(|n|n.outputs.first()).map(|&e|range_on(e)).unwrap_or(false) { return false; }
        }
        // Structural arithmetic constants are public request sources, not
        // unconstrained model tensors. Validate before graph Fiat-Shamir.
        for (ids,&(start,width)) in self.rank_constants.iter().zip(&self.ranges) {
            for (edge,value) in [(ids[0],self.logit_bound),(ids[1],self.vocab as u64)] {
                if g.shapes.get(edge)!=Some(&vec![1]) || g.public.get(&edge).map(|v|v.as_slice())!=Some(&[value][..]) { return false; }
            }
            let vp=match width.checked_next_power_of_two() { Some(v)=>v,None=>return false };
            if g.shapes.get(ids[2])!=Some(&vec![1,width]) { return false; }
            let data=match g.public.get(&ids[2]) { Some(v)=>v,None=>return false };
            if data.len()!=vp || data.iter().enumerate().any(|(j,&v)|v!=if j<width {(start+j) as u64} else {0}) { return false; }
        }
        let decode=|edge:usize, width:usize| -> Option<Vec<Option<usize>>> {
            if g.shapes.get(edge)? != &vec![self.seq,width] { return None; }
            let v=g.public.get(&edge)?;
            let sp=self.seq.checked_next_power_of_two()?;
            let vp=width.checked_next_power_of_two()?;
            if v.len()!=sp.checked_mul(vp)? { return None; }
            let mut rows=vec![None; self.seq];
            for j in 0..vp { for i in 0..sp {
                match v[i+j*sp] {
                    0=>{},
                    1 if i<self.seq && j<width && rows[i].is_none()=>rows[i]=Some(j),
                    _=>return None,
                }
            }}
            Some(rows)
        };
        let input=match decode(self.embedding,self.vocab) { Some(v)=>v,None=>return false };
        if input.iter().any(Option::is_none) { return false; }
        let mut predicted=vec![None;self.seq];
        for (&e,&(start,width)) in self.selectors.iter().zip(&self.ranges) {
            let rows=match decode(e,width) { Some(v)=>v,None=>return false };
            for (i,v) in rows.into_iter().enumerate() { if let Some(j)=v {
                if predicted[i].is_some() { return false; }
                predicted[i]=Some(start+j);
            }}
        }
        predicted.iter().all(Option::is_some) && (self.prompt_len-1..self.seq-1).all(|i| input[i+1]==predicted[i])
    }
}
