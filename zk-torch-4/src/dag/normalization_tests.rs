//! Real CPU operator/graph tests, plus direct evaluation of emitted range roots.
//! These are NOT a substitute for cryptographic prover/verifier replay.
use super::*;
use crate::basicblock::{BasicBlock, BasicBlockType, DivConst, RMSReciprocal};
use crate::basicblock::llama::{rounded_rms_reciprocal, RMS_MEAN_MAX};
use crate::util::arith::{f_to_int,int_to_f};

fn forward(d: &Dag,w: &mut [Vec<Witness>],override_value: Option<(EdgeId,i128)>) -> bool {
    let values=override_value.map(|(e,v)| (e,vec![v;w[e][0].shape.iter().map(|n|n.next_power_of_two()).product()]));
    forward_values(d,w,values)
}
fn forward_values(d: &Dag,w: &mut [Vec<Witness>],override_value: Option<(EdgeId,Vec<i128>)>) -> bool {
    let mut valid=true;
    for &n in &d.topo {
        let nd=&d.nodes[n];
        if let BasicBlockType::NonNegative(b)=&nd.kind {
            for v in w[nd.inputs[0]][0].data.as_ref().unwrap().evaluations_ref() {
                let x=f_to_int(*v);
                valid &= x>=0 && x<(1i128<<b.table_size_log);
            }
        }
        let inputs=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outputs=nd.kind.run(&inputs);
        for (&e,mut v) in nd.outputs.iter().zip(outputs) {
            if let Some((target,values))=&override_value {
                if e==*target {
                    v=Witness::new(v.shape,values.iter().copied().map(int_to_f).collect(),v.data_type,v.sf,v.role);
                }
            }
            w[e]=vec![v];
        }
    }
    valid
}
fn input(b:&mut DagBuilder,shape:Vec<usize>,values:Vec<i128>,sf:usize)->EdgeId {
    b.committed_input(Witness::new(shape,values.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Input))
}

#[test]
fn division_actual_graph_signed_ties_and_mutations() {
    for c in [1,2,3,768,1024,8192,1<<20] {
        for a in [-(c as i128)/2,-1,0,1,c as i128/2,(1i128<<54)-1,-((1i128<<54)-1)] {
            let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![a],0);
            // Raw advice must acquire the same constraints at compile time.
            let y=b.add_gkr_node(vec![x],BasicBlockType::DivConst(DivConst{divisor:c}))[0];
            b.init_values.push(Some(Witness::new_wo_data(vec![1],DataType::Float,0,Role::Output)));
            let (d,w)=b.compile();let expected=(a+c as i128/2).div_euclid(c as i128);
            let mut honest=w.clone();assert!(forward(&d,&mut honest,None));
            assert_eq!(f_to_int(honest[y][0].data.as_ref().unwrap().index(0)),expected);
            for bad in [expected-1,expected+1] {assert!(!forward(&d,&mut w.clone(),Some((y,bad))));}
        }
    }
}

#[test]
fn rms_actual_graph_zero_ties_sign_and_adjacent_mutations() {
    for sf in [0,1,10,15,16,20] {
        let q=1i128<<sf;
        for m in [0,1,2,q,4*(1i128<<(3*sf)),RMS_MEAN_MAX] {
            if m>RMS_MEAN_MAX {continue;}
            let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![m],sf);
            let r=b.rms_reciprocal_from_mean(x);
            let (d,w)=b.compile();let expected=rounded_rms_reciprocal(m,sf);
            let mut honest=w.clone();assert!(forward(&d,&mut honest,None),"sf={sf} m={m}");
            assert_eq!(f_to_int(honest[r][0].data.as_ref().unwrap().index(0)),expected);
            for bad in [expected-1,expected+1,-expected,1i128<<40] {
                if bad!=expected {assert!(!forward(&d,&mut w.clone(),Some((r,bad))),"sf={sf} m={m} r={bad}");}
            }
        }
    }
}

#[test]
fn means_ignore_poisoned_padding_and_use_logical_count() {
    for pad in [-100000,777777] {
        let mut b=DagBuilder::new();
        // Col-major [3,3]: axis0 varies fastest, both axes have padding.
        let mut v=vec![pad;16];for j in 0..3 {for i in 0..3 {v[i+4*j]=(i+j*3) as i128;}}
        let x=input(&mut b,vec![3,3],v,0);let y=b.reduce_mean(x,&[1]);
        let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
        let got=w[y][0].data.as_ref().unwrap().evaluations_ref().iter().map(|v|f_to_int(*v)).collect::<Vec<_>>();
        assert_eq!(got,vec![3,4,5,0]);
        let mut b=DagBuilder::new();
        let x=input(&mut b,vec![3],vec![1,2,3,pad],0);let y=b.reduce_mean(x,&[0]);
        let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None));
        assert!(w[y][0].shape.is_empty());
        assert_eq!(f_to_int(w[y][0].data.as_ref().unwrap().index(0)),2);
    }
}

#[test]
fn named_transformer_norm_paths_cpu_and_zero_extension() {
    for n in [3usize,768,1024,4096,8192] {
        let sf=*crate::SF_LOG;if sf>20 {return;}
        let q=1i128<<sf;
        for kind in 0..6 {
            let mut b=DagBuilder::new();let len=4*n.next_power_of_two();
            let mut v=vec![12345;len];
            for j in 0..n {for i in 0..3 {v[i+4*j]=if i==0 {0} else {q+(j%3) as i128};}}
            let x=input(&mut b,vec![1,3,n],v,sf);
            let gain=b.param(Witness::new(vec![n],vec![int_to_f(q);n.next_power_of_two()],DataType::Float,sf,Role::Constant));
            let bias=b.param(Witness::new(vec![n],vec![int_to_f(0);n.next_power_of_two()],DataType::Float,sf,Role::Constant));
            let y=match kind {
                0=>b.pipe(&[x],super::llama::llama_rms_norm(gain))[0],
                1=>b.pipe(&[x],super::gpt2::gpt2_layer_norm(gain,bias))[0],
                2=>b.pipe(&[x],super::gptj::gptj_layer_norm(gain,bias))[0],
                3=>b.pipe(&[x],super::bert::bert_layer_norm(gain,bias))[0],
                4=>b.pipe(&[x],super::whisper::whisper_layer_norm(gain,bias))[0],
                _=>b.pipe(&[x],super::llama::llama3_rms_norm(gain,3))[0],
            };
            let (d,mut w)=b.compile();assert!(forward(&d,&mut w,None),"kind={kind} n={n}");
            for j in 0..n {assert_eq!(f_to_int(w[y][0].data.as_ref().unwrap().index(4*j)),0);}
        }
    }
}

#[test]
fn normalization_rejects_unsupported_scale_and_legacy_advice() {
    let bad_scale=std::panic::catch_unwind(|| {let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![1],21);b.rms_reciprocal_from_mean(x);b.compile();});
    assert!(bad_scale.is_err());
    let legacy=std::panic::catch_unwind(|| {let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![1],10);let _=b.add_gkr_node(vec![x],BasicBlockType::RMSReciprocal(RMSReciprocal{dim:1}));b.init_values.push(Some(Witness::new_wo_data(vec![1],DataType::Float,10,Role::Output)));b.compile();});
    assert!(legacy.is_err());
    let over_mean=std::panic::catch_unwind(|| rounded_rms_reciprocal(RMS_MEAN_MAX+1,10));
    assert!(over_mean.is_err());
    let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![1i128<<54],0);
    b.div_const(x,2);let (d,mut w)=b.compile();assert!(!forward(&d,&mut w,None));
}

#[test]
fn private_rms_bucket_rejects_wrong_valid_onehot_and_padding() {
    for m in [0,1,2,3,127,128,RMS_MEAN_MAX] {
        let mut b=DagBuilder::new();let x=input(&mut b,vec![1],vec![m],20);
        b.rms_reciprocal_from_mean(x);let (d,w)=b.compile();
        let bucket=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::RMSBucket(_))).unwrap().outputs[0];
        assert!(forward(&d,&mut w.clone(),None));
        let correct=if m==0 {0} else {128-(m as u128).leading_zeros() as usize};
        for wrong in [0,1,2,28,29,31] {
            if wrong==correct {continue;}
            let mut hint=vec![0;32];hint[wrong]=1;
            assert!(!forward_values(&d,&mut w.clone(),Some((bucket,hint))),"m={m},wrong={wrong}");
        }
        assert!(!forward_values(&d,&mut w.clone(),Some((bucket,vec![0;32]))));
        let mut two=vec![0;32];two[correct]=1;two[(correct+1)%32]=1;
        assert!(!forward_values(&d,&mut w.clone(),Some((bucket,two))));
        let mut nonbinary=vec![0;32];nonbinary[correct]=2;
        assert!(!forward_values(&d,&mut w.clone(),Some((bucket,nonbinary))));
        let graph=crate::morpheus::piop::Graph::new(&d,&w,crate::morpheus::piop::Config{tcl:4,slots_log:10,req_planes_log:5});
        assert!(graph.committed[bucket]);assert!(!graph.public.contains_key(&bucket));
        assert_eq!(graph.shapes[bucket],vec![1,32]);
    }
}

#[test]
fn private_rms_all_dyadic_boundaries_sf16_sf20() {
    let mut means=vec![0];
    for j in 0..28 {let lo=1i128<<j;means.extend([lo-1,lo,2*lo-1]);}
    means.resize(128,0);
    for sf in [16,20] {
        let mut b=DagBuilder::new();let x=input(&mut b,vec![128],means.clone(),sf);
        let r=b.rms_reciprocal_from_mean(x);let (d,mut w)=b.compile();
        assert!(forward(&d,&mut w,None));
        let bucket=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::RMSBucket(_))).unwrap().outputs[0];
        assert_eq!(w[bucket][0].shape,vec![128,32]);
        for (i,&m) in means.iter().enumerate() {
            assert_eq!(f_to_int(w[r][0].data.as_ref().unwrap().index(i)),rounded_rms_reciprocal(m,sf));
            let expected=if m==0 {0} else {128-(m as u128).leading_zeros() as usize};
            for j in 0..32 {assert_eq!(f_to_int(w[bucket][0].data.as_ref().unwrap().index(i+128*j)),if j==expected {1}else{0});}
        }
    }
}

#[test]
fn lenet_average_pool_exact_signed_ties_layout_and_padding() {
    for (c,h,w) in [(6usize,28usize,28usize),(16,10,10),(3,4,6)] {
        for batch in [1usize,3] {
            let cp=c.next_power_of_two();let hp=h.next_power_of_two();let wp=w.next_power_of_two();
            let lead=if batch==1 {c}else{batch.next_power_of_two()*cp};
            let len=lead.next_power_of_two()*hp*wp;
            let mut values=vec![999999;len];
            for b in 0..batch {for ch in 0..c {for ih in 0..h {for iw in 0..w {
                values[iw+wp*(ih+hp*(ch+b*cp))]=((iw+ih+ch+b)%7) as i128-3;
            }}}}
            let shape=if batch==1 {vec![c,h,w]}else{vec![batch,c,h,w]};
            let mut builder=DagBuilder::new();let x=input(&mut builder,shape,values.clone(),*crate::SF_LOG);
            let y=super::lenet::avg_pool2x2(&mut builder,x,c);let (d,mut witnesses)=builder.compile();
            assert!(forward(&d,&mut witnesses,None));
            let out=witnesses[y][0].data.as_ref().unwrap();let op=(h/2).next_power_of_two();let oq=(w/2).next_power_of_two();
            for b in 0..batch {for ch in 0..c {for ih in 0..h/2 {for iw in 0..w/2 {
                let sum=(0..2).flat_map(|dh|(0..2).map(move|dw|(dh,dw))).map(|(dh,dw)|values[2*iw+dw+wp*(2*ih+dh+hp*(ch+b*cp))]).sum::<i128>();
                assert_eq!(f_to_int(out.index(iw+oq*(ih+op*(ch+b*cp)))),(sum+2).div_euclid(4));
            }}}}
            for b in 0..batch.next_power_of_two() {for ch in 0..cp {for ih in 0..op {for iw in 0..oq {
                if b>=batch || ch>=c || ih>=h/2 || iw>=w/2 {assert_eq!(f_to_int(out.index(iw+oq*(ih+op*(ch+b*cp)))),0);}
            }}}}
        }
    }
}

#[test]
fn resnet_global_mean_uses_row_major_channels_and_spatial_padding() {
    for shape in [vec![2usize,2,2],vec![3,3,5],vec![3,7,7],vec![2,3,3,5]] {
        let ndim=shape.len();let c=shape[ndim-3];let h=shape[ndim-2];let w=shape[ndim-1];
        let batch=if ndim==4 {shape[0]}else{1};
        let cp=c.next_power_of_two();let hp=h.next_power_of_two();let wp=w.next_power_of_two();
        let mut values=vec![999999;batch.next_power_of_two()*cp*hp*wp];
        for b in 0..batch {for ch in 0..c {for ih in 0..h {for iw in 0..w {
            values[iw+wp*(ih+hp*(ch+cp*b))]=(100*ch+1000*b+ih*w+iw) as i128;
        }}}}
        let mut builder=DagBuilder::new();let x=input(&mut builder,shape,values.clone(),*crate::SF_LOG);
        let y=super::resnet::global_mean_row_major(&mut builder,x);let (d,mut witnesses)=builder.compile();
        assert!(forward(&d,&mut witnesses,None));
        let out=witnesses[y][0].data.as_ref().unwrap();
        for b in 0..batch {for ch in 0..c {
            let sum=(0..h).flat_map(|ih|(0..w).map(move|iw|(ih,iw))).map(|(ih,iw)|values[iw+wp*(ih+hp*(ch+cp*b))]).sum::<i128>();
            let n=(h*w) as i128;assert_eq!(f_to_int(out.index(ch+cp*b)),(sum+n/2).div_euclid(n));
        }}
        for b in 0..batch.next_power_of_two() {for ch in 0..cp {
            if b>=batch || ch>=c {assert_eq!(f_to_int(out.index(ch+cp*b)),0);}
        }}
    }
}
