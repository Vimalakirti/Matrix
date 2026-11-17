//! Requires the selected finite-sigmoid production increment. This is a joint
//! integration test, not a claim that legacy log-advice sigmoid is repaired by
//! layout metadata alone.
use super::*;
use crate::basicblock::{BasicBlock,BasicBlockType};
use crate::util::arith::{f_to_int,int_to_f};
fn forward(d:&Dag,w:&mut [Vec<Witness>]) {
    for &n in &d.topo {
        let nd=&d.nodes[n];
        if let BasicBlockType::NonNegative(r)=&nd.kind {
            for v in w[nd.inputs[0]][0].data.as_ref().unwrap().evaluations_ref() {
                let z=f_to_int(*v);assert!(z>=0 && z<(1i128<<r.table_size_log),"range {n}: {z}");
            }
        }
        if matches!(nd.kind,BasicBlockType::ProductZeroCheck(_)) {
            let a=w[nd.inputs[0]][0].data.as_ref().unwrap();let b=w[nd.inputs[1]][0].data.as_ref().unwrap();
            for i in 0..a.evaluations_ref().len() {assert_eq!((a.index(i)*b.index(i)).reduce().0,0);}
        }
        let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outs=nd.kind.run(&ins);
        for (&e,out) in nd.outputs.iter().zip(outs) {w[e]=vec![out];}
    }
}
#[test]
fn cnn_silu_row_major_finite_profile_matches_exact_dyadic_reference() {
    let sf=*crate::SF_LOG;assert!(sf<=20);
    let q=1i128<<sf;let l=crate::basicblock::exp::exp_ln2(sf);
    // At -k*Ls (k<=2), canonical remainder is exactly0. Internal Qe=2^15
    // therefore gives E_s=RD(2^(30-k),2^(30-s)), independently of the DAG.
    let exp_dyadic=|k:usize| {let den=1i128<<(30-sf);((1i128<<(30-k))+den/2)/den};
    for shape in [vec![3usize,3,5],vec![3,3,2,5]] {
        let n=shape.iter().map(|d|d.next_power_of_two()).product::<usize>();
        let logical=shape.iter().product::<usize>();
        for poison in [1i128<<60,-(1i128<<60)] {
            let mut data=vec![poison;n];let mut expected=vec![0i128;n];
            for i in 0..logical {
                let (mut rem,mut raw,mut stride)=(i,0,1);
                for d in (0..shape.len()).rev() {raw+=(rem%shape[d])*stride;rem/=shape[d];stride*=shape[d].next_power_of_two();}
                let k=(i%5) as i128-2;let x=k*l;data[raw]=x;
                let e0=if k>0 {exp_dyadic(k as usize)} else {q};
                let ex=if k<0 {exp_dyadic((-k) as usize)} else {q};
                let d=e0+ex;let sig=(2*q*ex+d)/(2*d);
                expected[raw]=(x*sig+q/2).div_euclid(q);
            }
            let mut b=DagBuilder::new();let x=b.committed_input(Witness::new(shape.clone(),data.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Input));
            let y=b.silu_row_major(x);let(d,mut w)=b.compile();forward(&d,&mut w);
            assert_eq!(w[y][0].shape,shape);
            assert_eq!(w[y][0].data.as_ref().unwrap().evaluations_ref().iter().map(|v|f_to_int(*v)).collect::<Vec<_>>(),expected);
            assert!(d.nodes.iter().all(|n|!matches!(n.kind,BasicBlockType::SigmoidConst(_) | BasicBlockType::SoftmaxConst(_))));
        }
    }
}
