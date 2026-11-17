//! Canonical max over authenticated support, finite exp, exact integer division.
//! No log/softplus advice, additive negative sentinel, or secret-dependent layout.
use crate::basicblock::{BasicBlockType,MaskedMaxSelector,NormalizedRatio,ProductZeroCheck};
use crate::basicblock::normalization::ratio_budget;
use crate::dag::{DagBuilder,DataType,EdgeId,Role,Witness};
use crate::util::arith::int_to_f;

// Scores are rescale quotients, so their bound enters Eq. rounding_nowrap
// as |x| + S*SCORE_BOUND + S/2 < q; 2^54 broke it at S = 2^10.
const SCORE_BOUND:i128=(1i128<<40)-1;
fn mul(b:&mut DagBuilder,a:EdgeId,c:EdgeId)->EdgeId {
    let ndim=b.init_values[a].as_ref().unwrap().shape.len();
    let s:String=(0..ndim).map(|i|(b'a'+i as u8)as char).collect();
    b.einsum(format!("{s},{s}->{s}"),vec![a,c],false)[0]
}
fn zero(b:&mut DagBuilder,x:EdgeId) {
    let w=b.init_values[x].as_ref().unwrap();let shape=w.shape.clone();
    let one=b.integer_literal(shape.clone(),1,0);
    let out=b.add_gkr_node(vec![x,one],BasicBlockType::ProductZeroCheck(ProductZeroCheck))[0];
    b.init_values.push(Some(Witness::new_wo_data(shape,DataType::Uint,0,Role::Output)));
    b.self_claim_edges.push(out);
}
fn maximum(b:&mut DagBuilder,s:EdgeId,h:EdgeId)->EdgeId {
    let prod=mul(b,s,h);b.einsum("ab->a".into(),vec![prod],false)[0]
}
fn support_rows(b:&DagBuilder,mask:EdgeId,shape:&[usize])->Vec<i128> {
    assert_eq!(shape.len(),2);assert!(shape.iter().all(|d|d.is_power_of_two()));
    let values=b.program_literals.get(&mask).expect("normalization support must be an immutable program literal");
    assert_eq!(values.len(),shape[0]*shape[1]);assert!(values.iter().all(|&v|v<=1));
    (0..shape[0]).map(|r|i128::from((0..shape[1]).any(|c|values[r+shape[0]*c]==1))).collect()
}

pub(crate) fn max_selector_constraints(b:&mut DagBuilder,s:EdgeId,a:EdgeId,h:EdgeId) {
    let sw=b.init_values[s].as_ref().unwrap();let shape=sw.shape.clone();let sf=sw.sf;
    assert_eq!(b.init_values[a].as_ref().unwrap().shape,shape);assert_eq!(b.init_values[h].as_ref().unwrap().shape,shape);
    assert_eq!(b.init_values[a].as_ref().unwrap().sf,0);assert_eq!(b.init_values[h].as_ref().unwrap().sf,0);
    let active=support_rows(b,a,&shape);assert!(shape[1]<(1usize<<54));
    // Boolean H + sum H=V + H supported on A proves a selected member,
    // not merely an arbitrary upper bound. Empty rows have H=0 and m=0.
    b.integer_interval(h,0,1);
    let one=b.integer_literal(shape.clone(),1,0);let forbidden=b.sub(one,a)[0];
    let bad=mul(b,h,forbidden);zero(b,bad);
    let count=b.einsum("ab->a".into(),vec![h],false)[0];
    let v=b.public_literal(Witness::new(vec![shape[0]],active.into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Input));
    let diff=b.sub(count,v)[0];zero(b,diff);
    let selected=mul(b,s,a);b.integer_interval(selected,-SCORE_BOUND,SCORE_BOUND);
    let m=maximum(b,s,h);let mb=b.change_shape(m,vec![shape[0],1]);
    let shifted=b.sub(s,mb)[0];let shifted=mul(b,shifted,a);
    let l=crate::basicblock::exp::exp_ln2(sf);let bound=(1i128<<crate::basicblock::exp::exp_bits())*l-l/2-1;
    // 2*SCORE_BOUND+bound and both paired range envelopes are <p.
    b.integer_interval(shifted,-bound,0);
}

pub(crate) fn ratio_constraints(b:&mut DagBuilder,e:EdgeId,d:EdgeId,w:EdgeId,max_terms:usize) {
    let ew=b.init_values[e].as_ref().unwrap();let shape=ew.shape.clone();let sf=ew.sf;
    assert!(!shape.is_empty() && shape.len()<=26 && shape.iter().all(|&d|d>0),"ratio requires explicit nonzero supported extents");
    assert_eq!(b.init_values[d].as_ref().unwrap().shape,shape);assert_eq!(b.init_values[w].as_ref().unwrap().shape,shape);
    assert_eq!(b.init_values[d].as_ref().unwrap().sf,sf);assert_eq!(b.init_values[w].as_ref().unwrap().sf,sf);
    let(q,dm,bits)=ratio_budget(sf,max_terms);
    b.integer_interval(e,0,q);b.integer_interval(d,q,dm);b.integer_interval(w,0,q);
    let twiceq=b.integer_literal(shape.clone(),2*q,0);
    let n2=mul(b,e,twiceq);let nd=b.add(n2,d)[0];
    let dw=mul(b,d,w);let twice=b.integer_literal(shape.clone(),2,0);let dw2=mul(b,dw,twice);
    let rho=b.sub(nd,dw2)[0];
    let d2=mul(b,d,twice);let one=b.integer_literal(shape,1,sf);let dm1=b.sub(d2,one)[0];
    let tau=b.sub(dm1,rho)[0];
    b.add_nonneg_node_bits(rho,bits);b.add_nonneg_node_bits(tau,bits);
}

impl DagBuilder {
    /// Explicit public support in the original column-major padded layout.
    /// Every logical query row must have a key; padding rows must have none.
    pub fn selected_softmax(&mut self,scores:EdgeId,support:Vec<bool>)->EdgeId {
        let w=self.init_values[scores].as_ref().unwrap();let shape=w.shape.clone();let sf=w.sf;
        assert!(!shape.is_empty() && shape.len()<=26 && shape.iter().all(|&d|d>0));
        assert_eq!(sf,*crate::SF_LOG);assert!(sf<=30);
        let pads:Vec<_>=shape.iter().map(|&d|d.checked_next_power_of_two().expect("normalization shape overflow")).collect();
        let cols=*pads.last().unwrap();let rows=pads[..pads.len()-1].iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("normalization row overflow");
        assert_eq!(support.len(),rows.checked_mul(cols).expect("normalization shape overflow"));
        let mut active=vec![0;rows];let mut max_terms=0;
        for r in 0..rows {
            let mut index=r;let mut logical=true;
            for (&extent,&pad) in shape[..shape.len()-1].iter().zip(&pads) {logical &= index%pad<extent;index/=pad;}
            let mut count=0;
            for c in 0..cols {if support[r+rows*c] {assert!(logical && c<*shape.last().unwrap(),"support selects padding");count+=1;}}
            assert_eq!(count>0,logical,"all-masked logical row is unsupported");
            active[r]=i128::from(logical);max_terms=max_terms.max(count);
        }
        ratio_budget(sf,max_terms);
        let scores=self.change_shape(scores,vec![rows,cols]);
        let mask=self.public_literal(Witness::new(vec![rows,cols],support.into_iter().map(|a|int_to_f(i128::from(a))).collect(),DataType::Float,0,Role::Input));
        // Mask raw excluded scores before *any* numerical-domain constraint.
        let s=mul(self,scores,mask);
        let h=self.add_gkr_node(vec![s,mask],BasicBlockType::MaskedMaxSelector(MaskedMaxSelector))[0];
        self.init_values.push(Some(Witness::new_wo_data(vec![rows,cols],DataType::Float,0,Role::Auxiliary)));
        let m=maximum(self,s,h);let m=self.change_shape(m,vec![rows,1]);
        let t=self.sub(s,m)[0];let t=mul(self,t,mask);
        let e=self.exp(t)[0];let e=mul(self,e,mask);
        let d=self.einsum("ab->a".into(),vec![e],false)[0];
        // Empty padded rows get a fixed positive divisor; numerator remains0.
        let q=1i128<<sf;
        let dummy=self.public_literal(Witness::new(vec![rows],active.into_iter().map(|v|int_to_f((1-v)*q)).collect(),DataType::Float,sf,Role::Input));
        let d=self.add(d,dummy)[0];
        let ones=self.integer_literal(vec![rows,cols],1,0);
        let d=self.einsum("a,ab->ab".into(),vec![d,ones],false)[0];
        let out=self.add_gkr_node(vec![e,d],BasicBlockType::NormalizedRatio(NormalizedRatio{max_terms}))[0];
        self.init_values.push(Some(Witness::new_wo_data(vec![rows,cols],DataType::Float,sf,Role::Auxiliary)));
        self.change_shape(out,shape)
    }
    fn normalization_support(&self,scores:EdgeId,causal:bool)->Vec<bool> {
        let shape=&self.init_values[scores].as_ref().unwrap().shape;
        assert!(!shape.is_empty() && shape.iter().all(|&d|d>0));
        assert!(!causal || shape.len()>=2);
        let pads:Vec<_>=shape.iter().map(|&d|d.checked_next_power_of_two().expect("shape overflow")).collect();
        let n=pads.iter().try_fold(1usize,|n,&d|n.checked_mul(d)).expect("shape overflow");
        (0..n).map(|mut i| {let mut c=vec![];let mut valid=true;for (&d,&p)in shape.iter().zip(&pads){c.push(i%p);valid &=i%p<d;i/=p;} valid && (!causal || c[c.len()-1]<=c[c.len()-2])}).collect()
    }
    pub fn softmax(&mut self,scores:EdgeId)->EdgeId {let a=self.normalization_support(scores,false);self.selected_softmax(scores,a)}
    pub fn causal_softmax(&mut self,scores:EdgeId,seq_len:usize)->EdgeId {
        let shape=&self.init_values[scores].as_ref().unwrap().shape;
        assert!(shape.len()>=2);assert_eq!(&shape[shape.len()-2..],&[seq_len,seq_len]);
        let a=self.normalization_support(scores,true);self.selected_softmax(scores,a)
    }
    pub(crate) fn finite_sigmoid(&mut self,x:EdgeId)->EdgeId {
        let w=self.init_values[x].as_ref().unwrap();let shape=w.shape.clone();let sf=w.sf;
        let mask=self.normalization_support(x,false);
        let n=mask.len();let x=self.change_shape(x,vec![n]);
        let valid=self.public_literal(Witness::new(vec![n],mask.into_iter().map(|v|int_to_f(i128::from(v))).collect(),DataType::Float,0,Role::Input));
        let x=mul(self,x,valid);
        let selector=self.public_literal(Witness::new(vec![2],vec![int_to_f(0),int_to_f(1)],DataType::Float,0,Role::Input));
        let pair=self.einsum("a,b->ab".into(),vec![x,selector],false)[0];
        // Reinterpret the true logical cells as active rows, preserving original
        // padding support. selected_softmax requires its *own* logical rows live,
        // so use a full safe pair for pads then explicitly zero their output.
        let out=self.softmax(pair);
        let chosen=self.einsum("ab,b->ab".into(),vec![out,selector],false)[0];
        let chosen=self.einsum("ab->a".into(),vec![chosen],false)[0];
        let chosen=mul(self,chosen,valid);
        assert_eq!(self.init_values[chosen].as_ref().unwrap().sf,sf);
        self.change_shape(chosen,shape)
    }
}
