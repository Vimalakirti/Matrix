//! Hints for the registered selected finite-exp normalization program.
//! Both receive defining arithmetic/range roots at DagBuilder::compile.
use crate::basicblock::BasicBlock;
use crate::dag::{Claim,DataType,Role,Witness};
use crate::sumcheck::SumcheckProof;
use crate::transcript::Transcript;
use crate::util::arith::{f_to_int,int_to_f};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;

/// Public no-wrap budget; does not inspect any secret value.
/// Returns (Q, Dmax, residual lookup width).
pub fn ratio_budget(sf:usize,max_terms:usize)->(i128,i128,usize) {
    assert!(sf<=30 && max_terms>0,"unsupported ratio profile");
    let q=1i128<<sf;
    let dm=q.checked_mul(i128::try_from(max_terms).expect("term count overflow")).expect("denominator budget overflow");
    let rmax=dm.checked_mul(2).and_then(|n|n.checked_sub(1)).expect("residual budget overflow");
    let bits=(128-(rmax as u128).leading_zeros() as usize).max(1);
    assert!(bits<62,"ratio residual lookup too wide");
    let t=(1i128<<bits)-1;
    let positive=2*q*q+dm;
    let negative=dm.checked_mul(2*q-1).and_then(|n|n.checked_add(t)).expect("ratio product budget overflow");
    let p=ALMOST_GOLDILOCKS_PRIME as i128;
    assert!(2*t<p && positive<p && negative<p,"ratio residual can wrap; use a smaller public profile or linked limbs");
    (q,dm,bits)
}

#[derive(Clone,Debug)]
pub struct MaskedMaxSelector;
impl BasicBlock for MaskedMaxSelector {
    fn run(&self,inputs:&[&Witness])->Vec<Witness> {
        assert_eq!(inputs.len(),2);let x=inputs[0];let mask=inputs[1];
        assert_eq!(x.shape.len(),2);assert_eq!(x.shape,mask.shape);
        let rows=x.shape[0];let cols=x.shape[1];
        assert!(rows.is_power_of_two() && cols.is_power_of_two());
        let mut h=vec![int_to_f(0);rows*cols];
        for row in 0..rows {
            let mut best:Option<(i128,usize)>=None;
            for col in 0..cols {
                let i=row+rows*col;let a=f_to_int(mask.data.as_ref().unwrap().index(i));
                assert!(a==0 || a==1,"mask must be public Boolean data");
                if a==1 {let v=f_to_int(x.data.as_ref().unwrap().index(i));if best.map(|(b,_)|v>b).unwrap_or(true) {best=Some((v,i));}}
            }
            if let Some((_,i))=best {h[i]=int_to_f(1);}
        }
        vec![Witness::new(x.shape.clone(),h,DataType::Float,0,Role::Auxiliary)]
    }
    fn prove(&self,_:&[&Witness],_:&[usize],_:&[&Claim],_:&mut Transcript)->(Vec<SumcheckProof>,Vec<Claim>){(vec![],vec![])}
    fn verify(&self,_:&[&Witness],_:&[&Claim],_:&[&SumcheckProof],_:&mut Transcript)->bool{true}
}

#[derive(Clone,Debug)]
pub struct NormalizedRatio { pub max_terms:usize }
impl BasicBlock for NormalizedRatio {
    fn run(&self,inputs:&[&Witness])->Vec<Witness> {
        assert_eq!(inputs.len(),2);let e=inputs[0];let d=inputs[1];
        assert_eq!(e.shape,d.shape);assert_eq!(e.sf,d.sf);
        let(q,dm,_)=ratio_budget(e.sf,self.max_terms);
        let data=e.data.as_ref().unwrap().evaluations_ref().iter().enumerate().map(|(i,&x)| {
            let n=f_to_int(x);let den=f_to_int(d.data.as_ref().unwrap().index(i));
            assert!((0..=q).contains(&n) && (q..=dm).contains(&den),"normalization ratio outside profile");
            int_to_f((2*q*n+den).div_euclid(2*den))
        }).collect();
        vec![Witness::new(e.shape.clone(),data,e.data_type,e.sf,Role::Auxiliary)]
    }
    fn prove(&self,_:&[&Witness],_:&[usize],_:&[&Claim],_:&mut Transcript)->(Vec<SumcheckProof>,Vec<Claim>){(vec![],vec![])}
    fn verify(&self,_:&[&Witness],_:&[&Claim],_:&[&SumcheckProof],_:&mut Transcript)->bool{true}
}
