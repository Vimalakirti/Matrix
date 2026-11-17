//! Original CPU operator bodies and actual emitted range/PZC predicate checks.
//! Not cryptographic verifier acceptance/rejection tests.
use super::*;
use crate::basicblock::{BasicBlock,BasicBlockType,NormalizedRatio};
use crate::basicblock::normalization::ratio_budget;
use crate::util::arith::{f_to_int,int_to_f};
fn input(b:&mut DagBuilder,shape:Vec<usize>,v:Vec<i128>,sf:usize)->EdgeId {
    b.committed_input(Witness::new(shape,v.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Input))
}
fn forward(d:&Dag,w:&mut [Vec<Witness>],bad:Option<(EdgeId,Vec<i128>)>)->bool {
    let mut good=true;
    for &n in &d.topo {
        let nd=&d.nodes[n];
        if let BasicBlockType::NonNegative(b)=&nd.kind {
            good &= w[nd.inputs[0]][0].data.as_ref().unwrap().evaluations_ref().iter().all(|v| {let z=f_to_int(*v);z>=0 && z<(1i128<<b.table_size_log)});
        }
        if matches!(nd.kind,BasicBlockType::ProductZeroCheck(_)) {
            let a=w[nd.inputs[0]][0].data.as_ref().unwrap();let b=w[nd.inputs[1]][0].data.as_ref().unwrap();
            good &= a.evaluations_ref().iter().enumerate().all(|(i,v)|(*v*b.index(i)).reduce().0==0);
        }
        let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outputs=nd.kind.run(&ins);
        for (&e,mut v) in nd.outputs.iter().zip(outputs) {
            if let Some((target,values))=&bad {if e==*target {v=Witness::new(v.shape,values.iter().copied().map(int_to_f).collect(),v.data_type,v.sf,v.role);}}
            w[e]=vec![v];
        }
    }
    good
}
fn vals(w:&[Vec<Witness>],e:EdgeId)->Vec<i128>{w[e][0].data.as_ref().unwrap().evaluations_ref().iter().map(|v|f_to_int(*v)).collect()}
#[test]
fn normalized_ratio_actual_roots_ties_and_modular_mutations() {
    for sf in [0,10,16,20,30] {
        let q=1i128<<sf;let terms=if sf==30 {2}else{4096};let ds=vec![q,2*q,2*q,(terms as i128)*q];let es=vec![0,q,q/2,q];
        let mut b=DagBuilder::new();let e=input(&mut b,vec![4],es.clone(),sf);let d=input(&mut b,vec![4],ds.clone(),sf);
        let y=b.add_gkr_node(vec![e,d],BasicBlockType::NormalizedRatio(NormalizedRatio{max_terms:terms}))[0];
        b.init_values.push(Some(Witness::new_wo_data(vec![4],DataType::Float,sf,Role::Auxiliary)));
        let(dag,w)=b.compile();let mut honest=w.clone();assert!(forward(&dag,&mut honest,None));
        let expected:Vec<_>=es.iter().zip(&ds).map(|(&e,&d)|(2*q*e+d)/(2*d)).collect();assert_eq!(vals(&honest,y),expected);
        for delta in [-1,1] {let mut bad=expected.clone();bad[0]+=delta;assert!(!forward(&dag,&mut w.clone(),Some((y,bad))));}
        // Old remainder-only equations admit a huge modular quotient; explicit
        // range roots reject it without relying on a host range assertion.
        let mut bad=expected.clone();bad[0]=(1i128<<54)-1;assert!(!forward(&dag,&mut w.clone(),Some((y,bad))));
    }
}
#[test]
fn causal_softmax_future_and_padding_are_structurally_excluded() {
    let sf=*crate::SF_LOG;let q=1i128<<sf;let l=crate::basicblock::exp::exp_ln2(sf);
    let mut baseline=None;
    for poison in [0,100*q,15*l,(1i128<<60)-1,-((1i128<<60)-1)] {
        let mut x=vec![poison;16];for t in 0..3 {for s in t..3 {x[s+4*t]=-((s-t)as i128)*l;}}
        let mut b=DagBuilder::new();let x=input(&mut b,vec![3,3],x,sf);let y=b.causal_softmax(x,3);let(d,mut w)=b.compile();
        assert!(forward(&d,&mut w,None));let actual=vals(&w,y);
        assert_eq!(actual[0],q);
        for t in 0..4 {for s in 0..4 {if t>s || t>=3 || s>=3 {assert_eq!(actual[s+4*t],0);}}}
        if let Some(ref expected)=baseline {assert_eq!(&actual,expected);}else{baseline=Some(actual);}
        assert!(d.nodes.iter().all(|n|!matches!(n.kind,BasicBlockType::SoftmaxConst(_) | BasicBlockType::SigmoidConst(_))));
    }
}
#[test]
fn selected_max_onehot_and_all_masked_mutations() {
    let sf=*crate::SF_LOG;let q=1i128<<sf;let l=crate::basicblock::exp::exp_ln2(sf);
    let mut b=DagBuilder::new();let x=input(&mut b,vec![1,3],vec![0,-l,100*q,1i128<<60],sf);let y=b.selected_softmax(x,vec![true,true,false,false]);let(d,w)=b.compile();
    let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaskedMaxSelector(_))).unwrap().outputs[0];
    let mut honest=w.clone();assert!(forward(&d,&mut honest,None));assert_eq!(vals(&honest,y)[2..],[0,0]);
    for bad in [vec![0,0,0,0],vec![1,1,0,0],vec![2,0,0,0],vec![0,0,1,0],vec![0,1,0,0]] {
        let accepted=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||forward(&d,&mut w.clone(),Some((h,bad))))).unwrap_or(false);assert!(!accepted);
    }
    assert!(std::panic::catch_unwind(||{let mut b=DagBuilder::new();let x=input(&mut b,vec![1,3],vec![0;4],sf);b.selected_softmax(x,vec![false;4]);}).is_err());
}
#[test]
fn sigmoid_two_logits_and_padding_are_exact() {
    let sf=*crate::SF_LOG;let q=1i128<<sf;let l=crate::basicblock::exp::exp_ln2(sf);
    let mut baseline=None;
    for poison in [0,1i128<<60,-(1i128<<60)] {
        let mut b=DagBuilder::new();let x=input(&mut b,vec![3],vec![0,l,-l,poison],sf);let y=b.sigmoid(x)[0];let(d,mut w)=b.compile();assert!(forward(&d,&mut w,None));let got=vals(&w,y);
        assert_eq!(got[0],(q+1)/2);assert_eq!(got[3],0);assert!(got[..3].iter().all(|&v|v>=0&&v<=q));
        if let Some(ref expected)=baseline {assert_eq!(&got,expected);}else{baseline=Some(got);}
    }
}
#[test]
fn practical_contexts_and_sf30_wrap_profile_fail_closed() {
    for sf in [16,20] {for terms in [512,4096,1<<20] {ratio_budget(sf,terms);}}
    ratio_budget(30,7);
    assert!(std::panic::catch_unwind(||ratio_budget(30,9)).is_err());
    let q=1i128<<30;let d=8*q;let p=almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as i128;
    let rho=(d-2*d*q).rem_euclid(p);let tau=2*d-1-rho;
    assert_eq!((rho,tau),(4*q-31,12*q+30));assert!(rho>=0&&rho<2*d&&tau>=0&&tau<2*d);
    for terms in [512usize,4096] {
        let sf=*crate::SF_LOG;if std::panic::catch_unwind(||ratio_budget(sf,terms)).is_err(){continue;}
        let mut b=DagBuilder::new();let x=input(&mut b,vec![1,terms],vec![0;terms],sf);let y=b.softmax(x);let(d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
        let q=1i128<<sf;let expected=(2*q+terms as i128)/(2*terms as i128);assert!(vals(&w,y).iter().all(|&v|v==expected));
    }
}

#[test]
fn whisper_decoder_encoder_cross_attention_selection_and_dependency() {
    let sf=*crate::SF_LOG;let q=1i128<<sf;
    let mut encoder_first=vec![];let mut cross_first=vec![];let mut decoder_prefix:Option<Vec<i128>>=None;
    assert!(6*q*q<(almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as i128)/2);
    // Raw attention products stay below p/2 throughSF30: max self score4Q^2.
    for future in [-2*q,2*q] {
        for causal in [false,true] {
            let mut b=DagBuilder::new();let x=input(&mut b,vec![1,3,1],vec![q,2*q,future,777],sf);
            let weight=b.param(Witness::new(vec![1,1],vec![int_to_f(q)],DataType::Float,sf,Role::Constant));
            let bias=b.param(Witness::new(vec![1],vec![int_to_f(0)],DataType::Float,sf,Role::Constant));
            let y=super::whisper::whisper_self_attention(&mut b,x,weight,weight,weight,weight,bias,bias,bias,bias,1,1,1,causal);
            let(d,mut w)=b.compile();assert!(forward(&d,&mut w,None));let got=vals(&w,y);
            if causal {assert_eq!(got[0],q);let prefix=got[..2].to_vec();if let Some(ref expected)=decoder_prefix{assert_eq!(&prefix,expected);}else{decoder_prefix=Some(prefix);}}else{encoder_first.push(got[0]);}
            let selector=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaskedMaxSelector(_))).unwrap();
            let support=&d.program_literals[&selector.inputs[1]];
            for t in 0..4 {for s in 0..4 {assert_eq!(support[s+4*t],u64::from(s<3&&t<3&&(!causal||t<=s)));}}
        }
        let mut b=DagBuilder::new();let xd=input(&mut b,vec![1,2,1],vec![q,q],sf);let xe=input(&mut b,vec![1,3,1],vec![q,2*q,future,777],sf);
        let weight=b.param(Witness::new(vec![1,1],vec![int_to_f(q)],DataType::Float,sf,Role::Constant));
        let bias=b.param(Witness::new(vec![1],vec![int_to_f(0)],DataType::Float,sf,Role::Constant));
        let y=super::whisper::whisper_cross_attention(&mut b,xd,xe,weight,weight,weight,weight,bias,bias,bias,bias,1,1,1,2,3);
        let(d,mut w)=b.compile();assert!(forward(&d,&mut w,None));cross_first.push(vals(&w,y)[0]);
        let selector=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaskedMaxSelector(_))).unwrap();
        let support=&d.program_literals[&selector.inputs[1]];assert_eq!(support,&vec![1,1,1,1,1,1,0,0]);
    }
    assert_ne!(encoder_first[0],encoder_first[1],"encoder must retain later-key dependency");
    assert_ne!(cross_first[0],cross_first[1],"cross attention must include all encoder keys, not decoder-triangular keys");
}
