//! CPU graph/range-predicate tests, NOT cryptographic verification receipts.
use super::*;
use crate::basicblock::{BasicBlock, BasicBlockType, InstanceNormHelper};
use crate::basicblock::llama::{floor_sqrt, RMS_MEAN_MAX};
use crate::util::arith::{f_to_int, int_to_f};
use super::instancenorm::{epsilon_units, instance_norm_bounds};

fn source(b: &mut DagBuilder, shape: Vec<usize>, vals: Vec<i128>, sf: usize, role: Role) -> EdgeId {
    let w = Witness::new(shape, vals.into_iter().map(int_to_f).collect(), DataType::Float, sf, role);
    if role == Role::Constant { b.param(w) } else { b.committed_input(w) }
}
fn forward(d: &Dag, w: &mut [Vec<Witness>], mutation: Option<(EdgeId,i128)>) -> bool {
    let mut valid = true;
    for &n in &d.topo {
        let nd = &d.nodes[n];
        if let BasicBlockType::NonNegative(b) = &nd.kind {
            for v in w[nd.inputs[0]][0].data.as_ref().unwrap().evaluations_ref() {
                let x = f_to_int(*v); valid &= x >= 0 && x < (1i128 << b.table_size_log);
            }
        }
        let ins = nd.inputs.iter().map(|&e| &w[e][0]).collect::<Vec<_>>();
        let outs = nd.kind.run(&ins);
        for (&e, mut v) in nd.outputs.iter().zip(outs) {
            if let Some((target,delta)) = mutation { if target == e {
                let values = v.data.as_ref().unwrap().evaluations_ref().iter().map(|x| int_to_f(f_to_int(*x)+delta)).collect();
                v = Witness::new(v.shape, values, v.data_type, v.sf, v.role);
            }}
            w[e] = vec![v];
        }
        // Check the actual local rounded-rescale relation too, rather than
        // mistaking successful witness generation for constraint satisfaction.
        if let BasicBlockType::ScaleDown(b) = &nd.kind {
            let a = &w[nd.inputs[0]][0]; let y = &w[nd.outputs[0]][0];
            let q = 1i128 << (a.sf-b.output_sf);
            for (x,y) in a.data.as_ref().unwrap().evaluations_ref().iter().zip(y.data.as_ref().unwrap().evaluations_ref()) {
                let rho = f_to_int(*x)-q*f_to_int(*y)+q/2;
                valid &= 0 <= rho && rho < q;
            }
        }
    }
    valid
}
fn rd(a: i128, d: i128) -> i128 { (a+d/2).div_euclid(d) }
fn reciprocal(m: i128, sf: usize) -> i128 {
    if m == 0 { return 0; }
    let c = 1i128 << (3*sf); let r = floor_sqrt(c/m);
    if 4*c >= m*(2*r+1)*(2*r+1) {r+1} else {r}
}
fn reference(x: &[i128], gamma: i128, beta: i128, sf: usize, e: i128) -> Vec<i128> {
    let q = 1i128 << sf; let n = x.len() as i128;
    let mean = rd(x.iter().sum(),n);
    let variance = rd(x.iter().map(|x| rd((x-mean)*(x-mean),q)).sum(),n);
    let a = rd(gamma*reciprocal(variance+e,sf),q);
    let offset = beta-rd(a*mean,q);
    x.iter().map(|x| rd(a*x,q)+offset).collect()
}

#[test]
fn instance_norm_exact_epsilon_conversion_and_public_bounds() {
    for sf in 0..=20 {
        let half = 0.5f64/(1u64 << sf) as f64;
        assert_eq!(epsilon_units(half,sf),1);
        assert_eq!(epsilon_units(f64::from_bits(half.to_bits()-1),sf),0);
        assert_eq!(epsilon_units(f64::from_bits(half.to_bits()+1),sf),1);
        assert_eq!(epsilon_units(0.0,sf),0);
        assert_eq!(epsilon_units(-0.0,sf),0);
        assert_eq!(epsilon_units(f64::from_bits(1),sf),0);
        assert_eq!(epsilon_units((RMS_MEAN_MAX as f64)/(1u64 << sf) as f64,sf),RMS_MEAN_MAX);
        for e in [0,1,RMS_MEAN_MAX] { let _ = instance_norm_bounds(sf,e); }
    }
    assert_eq!(epsilon_units(1e-5,10),0);
    assert_eq!(epsilon_units(1e-5,16),1);
    assert_eq!(epsilon_units(1e-5,20),10);
    for e in [-1.0, f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(std::panic::catch_unwind(||epsilon_units(e,20)).is_err());
    }
}

#[test]
fn instance_norm_row_major_layout_padding_and_parameter_dependencies() {
    let sf = *crate::SF_LOG; if sf>20 {return;}
    let q = 1i128 << sf;
    for epsilon in [0,1,epsilon_units(1e-5,sf)] {
        let shape=vec![3usize,3,2,5];let cp=4;let dp=4;let hp=2;let wp=8;
        let mut values=vec![777777;cp*dp*hp*wp];let mut rows=vec![Vec::new();3];
        for c in 0..3 {for d in 0..3 {for h in 0..2 {for x in 0..5 {
            let v=if c==0 {q} else if c==1 {(x as i128-2)*q} else {((d+h+x)%3) as i128-q/2};
            values[x+wp*(h+hp*(d+dp*c))]=v;rows[c].push(v);
        }}}}
        let mut b=DagBuilder::new();let x=source(&mut b,shape.clone(),values,sf,Role::Input);
        let gain=source(&mut b,vec![3],vec![q,2*q,-q,999999],sf,Role::Constant);
        let bias=source(&mut b,vec![3],vec![1,-2,3,888888],sf,Role::Constant);
        let y=b.instancenorm3d_units(x,gain,bias,epsilon);
        let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
        assert!(!d.nodes.iter().any(|n|matches!(n.kind,BasicBlockType::InstanceNorm3D(_))));
        let expected:Vec<_>=(0..3).map(|c|reference(&rows[c],[q,2*q,-q][c],[1,-2,3][c],sf,epsilon)).collect();
        let got=w[y][0].data.as_ref().unwrap().evaluations_ref();
        for c in 0..cp {let mut at=0;for dd in 0..dp {for h in 0..hp {for xx in 0..wp {
            let logical=c<3&&dd<3&&h<2&&xx<5;
            let v=if logical {let z=expected[c][at];at+=1;z}else{0};
            assert_eq!(f_to_int(got[xx+wp*(h+hp*(dd+dp*c))]),v,"sf={sf} e={epsilon} c={c} d={dd} h={h} w={xx}");
        }}}}
        let mut graph=crate::morpheus::piop::Graph::new(&d,&w,crate::morpheus::piop::Config{tcl:4,slots_log:12,req_planes_log:5});
        graph.prune_unclaimed();
        for (e,committed) in graph.committed.iter().enumerate() {
            if *committed && !graph.public.contains_key(&e) {
                for v in w[e][0].data.as_ref().unwrap().evaluations_ref() {
                    assert!(f_to_int(*v).abs() <= (1i128<<32)-1, "private source needs more than32 planes: edge={e}");
                }
            }
        }
        assert!(graph.committed[gain] && graph.committed[bias]);
        assert!(d.consumers[gain].iter().any(|&n|matches!(d.nodes[n].kind,BasicBlockType::Einsum(_))));
        assert!(d.consumers[bias].iter().any(|&n|matches!(d.nodes[n].kind,BasicBlockType::Einsum(_))));
    }
}

#[test]
fn instance_norm_domain_extremes_constant_channels_and_zero_epsilon() {
    let sf = *crate::SF_LOG; if sf>20 {return;}
    for epsilon in [0,1,RMS_MEAN_MAX] {
        let bounds=instance_norm_bounds(sf,epsilon);let xbound=bounds.input;let gain=bounds.parameter;
        for values in [vec![xbound,xbound],vec![-xbound,xbound],vec![0,0]] {
            let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,2],values.clone(),sf,Role::Input);
            let g=source(&mut b,vec![1],vec![gain],sf,Role::Constant);
            let beta=source(&mut b,vec![1],vec![-gain],sf,Role::Constant);
            let y=b.instancenorm3d_units(x,g,beta,epsilon);let (d,mut w)=b.compile();
            assert!(forward(&d,&mut w,None),"sf={sf} e={epsilon}");
            let got=w[y][0].data.as_ref().unwrap().evaluations_ref().iter().map(|x|f_to_int(*x)).collect::<Vec<_>>();
            assert_eq!(got,reference(&values,gain,-gain,sf,epsilon));
            if values[0]==values[1] {assert_eq!(got,vec![-gain,-gain]);}
        }
    }
}

#[test]
fn instance_norm_changed_statistics_reciprocal_and_affine_quotients_fail_constraints() {
    let sf = *crate::SF_LOG; if sf>20 {return;}let q=1i128<<sf;
    let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,2],vec![-q,q],sf,Role::Input);
    let g=source(&mut b,vec![1],vec![q],sf,Role::Constant);let beta=source(&mut b,vec![1],vec![3],sf,Role::Constant);
    b.instancenorm3d_units(x,g,beta,1);let (d,w)=b.compile();assert!(forward(&d,&mut w.clone(),None));
    let means=d.nodes.iter().filter(|n|matches!(n.kind,BasicBlockType::DivConst(_))).take(2).map(|n|n.outputs[0]).collect::<Vec<_>>();
    let r=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::RMSReciprocal(_))).unwrap().outputs[0];
    for e in means.into_iter().chain(std::iter::once(r)) {
        for delta in [-1,1] {assert!(!forward(&d,&mut w.clone(),Some((e,delta))),"edge={e} delta={delta}");}
    }
    // SF0 has no ScaleDown (unit divisor); its ScaleUp relation is exact.
    for n in &d.nodes {if matches!(n.kind,BasicBlockType::ScaleDown(_)) {
        for delta in [-1,1] {assert!(!forward(&d,&mut w.clone(),Some((n.outputs[0],delta))));}
    }}
}

#[test]
fn instance_norm_rejects_legacy_raw_advice_and_invalid_public_profiles() {
    let rejected=std::panic::catch_unwind(|| {
        let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,1],vec![0],10,Role::Input);
        let g=source(&mut b,vec![1],vec![1],10,Role::Constant);let beta=source(&mut b,vec![1],vec![0],10,Role::Constant);
        b.add_gkr_node(vec![x,g,beta],BasicBlockType::InstanceNorm3D(InstanceNormHelper{channels:1,depth:1,height:1,width:1,eps:1e-5}));
        b.init_values.push(Some(Witness::new_wo_data(vec![2,1],DataType::Float,10,Role::Output)));b.compile();
    });assert!(rejected.is_err());
    for (sf,e) in [(21,0),(20,-1),(20,RMS_MEAN_MAX+1)] {
        assert!(std::panic::catch_unwind(||instance_norm_bounds(sf,e)).is_err());
    }
    for bad in 0..3 {
        assert!(std::panic::catch_unwind(|| {
            let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,1],vec![0],10,Role::Input);
            let gain=source(&mut b,if bad==0 {vec![2]} else {vec![1]},if bad==0 {vec![1,1]}else{vec![1]},if bad==1 {11}else{10},Role::Constant);
            let beta=source(&mut b,vec![1],vec![0],10,Role::Constant);
            b.instancenorm3d_units(x,gain,beta,if bad==2 {-1}else{0});
        }).is_err());
    }
}

#[test]
fn instance_norm_out_of_domain_rejects_without_clipping_and_gains_are_used() {
    let sf=*crate::SF_LOG;if sf>20 {return;}let q=1i128<<sf;
    let bounds=instance_norm_bounds(sf,1);
    for bad in 0..3 {
        let mut b=DagBuilder::new();
        let input=if bad==0 {bounds.input+1}else{q};
        let x=source(&mut b,vec![1,1,1,2],vec![input,input],sf,Role::Input);
        let g=source(&mut b,vec![1],vec![if bad==1 {bounds.parameter+1}else{q}],sf,Role::Constant);
        let beta=source(&mut b,vec![1],vec![if bad==2 {bounds.parameter+1}else{0}],sf,Role::Constant);
        b.instancenorm3d_units(x,g,beta,1);let (d,mut w)=b.compile();
        assert!(!forward(&d,&mut w,None),"accepted out-of-domain case{bad}");
    }
    let mut results=Vec::new();
    for (gain,bias) in [(q,0),(2*q,0),(q,1)] {
        let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,2],vec![-q,q],sf,Role::Input);
        let g=source(&mut b,vec![1],vec![gain],sf,Role::Constant);let beta=source(&mut b,vec![1],vec![bias],sf,Role::Constant);
        let y=b.instancenorm3d_units(x,g,beta,0);let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
        results.push(w[y][0].data.as_ref().unwrap().evaluations_ref().iter().map(|x|f_to_int(*x)).collect::<Vec<_>>());
    }
    assert_ne!(results[0],results[1]);assert_ne!(results[0],results[2]);
}

#[test]
fn instance_norm_large_spatial_divisors_have_exact_bounded_scalar_guards() {
    // Boundary arithmetic only: this does NOT allocate/run128³ or256³ models.
    let bound=(1i128<<54)-1;
    for c in [1usize<<20,(1<<20)+1,1<<21,1<<24] {
        for a in [-bound,-(c as i128)/2,-1,0,1,c as i128/2,bound] {
            let mut b=DagBuilder::new();let x=source(&mut b,vec![1],vec![a],0,Role::Input);
            let y=b.div_const(x,c)[0];let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
            assert_eq!(f_to_int(w[y][0].data.as_ref().unwrap().index(0)),rd(a,c as i128));
            for delta in [-1,1] {assert!(!forward(&d,&mut w.clone(),Some((y,delta))));}
            let ymax=bound/c as i128+1;
            assert!(bound+c as i128*ymax+c as i128/2+((c-1) as i128) < (1i128<<56));
        }
    }
    assert!(std::panic::catch_unwind(|| {
        let mut b=DagBuilder::new();let x=source(&mut b,vec![1],vec![0],0,Role::Input);
        b.div_const(x,(1<<24)+1);b.compile();
    }).is_err());
}

#[test]
fn instance_norm_manual_legacy_dag_is_rejected_at_registration() {
    let mut b=DagBuilder::new();let x=source(&mut b,vec![1,1,1,1],vec![0],10,Role::Input);
    let g=source(&mut b,vec![1],vec![1],10,Role::Constant);let beta=source(&mut b,vec![1],vec![0],10,Role::Constant);
    b.instancenorm3d_units(x,g,beta,1);let (mut d,w)=b.compile();
    d.nodes[0].kind=BasicBlockType::InstanceNorm3D(InstanceNormHelper{channels:1,depth:1,height:1,width:1,eps:1e-5});
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::morpheus::piop::Graph::new(&d,&w,crate::morpheus::piop::Config::default());
    })).is_err());
}
