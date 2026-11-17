//! CPU execution of original operators and direct evaluation of the emitted
//! NonNegative/PZC roots. These are not cryptographic verifier replay receipts.
use super::*;
use crate::basicblock::{BasicBlock,BasicBlockType,GeneralMaxPoolHelper,MaxPoolHelper};
use crate::util::arith::{f_to_int,int_to_f};
fn input(b:&mut DagBuilder,s:Vec<usize>,v:Vec<i128>,sf:usize)->EdgeId {
    b.committed_input(Witness::new(s,v.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Input))
}
fn values(w:&[Vec<Witness>],e:EdgeId)->Vec<i128> {
    w[e][0].data.as_ref().unwrap().evaluations_ref().iter().map(|v|f_to_int(*v)).collect()
}
fn forward(d:&Dag,w:&mut [Vec<Witness>],overrides:&[(EdgeId,Vec<i128>)])->bool {
    let mut ok=true;
    for &n in &d.topo {
        let nd=&d.nodes[n];
        if let BasicBlockType::NonNegative(r)=&nd.kind {
            ok &= values(w,nd.inputs[0]).iter().all(|&v|v>=0 && v<1i128<<r.table_size_log);
        }
        if matches!(nd.kind,BasicBlockType::ProductZeroCheck(_)) {
            let a=w[nd.inputs[0]][0].data.as_ref().unwrap();let b=w[nd.inputs[1]][0].data.as_ref().unwrap();
            ok &= a.evaluations_ref().iter().enumerate().all(|(i,v)|(*v*b.index(i)).reduce().0==0);
        }
        let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outs=nd.kind.run(&ins);
        for (&e,mut v) in nd.outputs.iter().zip(outs) {
            if let Some((_,vals))=overrides.iter().find(|(target,_)|*target==e) {
                v=Witness::new(v.shape,vals.iter().copied().map(int_to_f).collect(),v.data_type,v.sf,v.role);
            }
            w[e]=vec![v];
        }
    }
    ok
}
fn raw_case(general:bool,xs:Vec<i128>)->(Dag,Vec<Vec<Witness>>,EdgeId,EdgeId) {
    let mut b=DagBuilder::new();let x=input(&mut b,vec![1,2,2],xs,*crate::SF_LOG);
    let kind=if general {BasicBlockType::GeneralMaxPoolHelper(GeneralMaxPoolHelper {channels:1,input_h:2,input_w:2,kernel_h:2,kernel_w:2,stride_h:2,stride_w:2})}
    else {BasicBlockType::MaxPoolHelper(MaxPoolHelper {channels:1,input_h:2,input_w:2,pool_h:2,pool_w:2})};
    let y=b.add_gkr_node(vec![x],kind)[0];
    b.init_values.push(Some(Witness::new_wo_data(vec![1,1,1],DataType::Uint,*crate::SF_LOG,Role::Output)));
    let(d,w)=b.compile();let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
    (d,w,y,h)
}
#[test]
fn maxpool_raw_helpers_reject_forged_above_max_and_selector_mutations() {
    for general in [false,true] {
        let(d,w,y,h)=raw_case(general,vec![0;4]);
        assert!(forward(&d,&mut w.clone(),&[]));
        // The old dominance-only graph accepts this exact all-zero forgery.
        assert!(!forward(&d,&mut w.clone(),&[(y,vec![1])]));
        for bad in [vec![0,0,0,0],vec![1,1,0,0],vec![2,0,0,0],vec![-1,1,1,0]] {
            assert!(!forward(&d,&mut w.clone(),&[(h,bad)]));
        }
        // Any tied maximizer is valid; selected index is private and unexposed.
        assert!(forward(&d,&mut w.clone(),&[(h,vec![0,0,0,1])]));
        let(d,w,y,h)=raw_case(general,vec![-7,-3,-3,-9]);
        let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));assert_eq!(values(&honest,y),[-3]);
        assert!(!forward(&d,&mut w.clone(),&[(h,vec![1,0,0,0])]));
        // Mutating both output and selection to a real but nonmax member must
        // fail dominance, independently of the attainment equality.
        assert!(!forward(&d,&mut w.clone(),&[(y,vec![-7]),(h,vec![1,0,0,0])]));
    }
}
#[test]
fn maxpool_nonpower_batch_channel_overlap_stride_and_edge_reference() {
    for (batch,channels,ih,iw,kh,kw,sh,sw,regular) in [
        (3usize,3usize,5usize,7usize,2usize,2usize,2usize,2usize,true),
        (3,3,5,7,3,2,1,2,false),(1,3,6,5,3,2,2,3,false),(1,1,3,5,3,5,1,1,false),
    ] {
        for poison in [1i128<<60,-(1i128<<60)] {
            let (bp,cp,hp,wp)=(batch.next_power_of_two(),channels.next_power_of_two(),ih.next_power_of_two(),iw.next_power_of_two());
            let (oh,ow)=((ih-kh)/sh+1,(iw-kw)/sw+1);let (ohp,owp)=(oh.next_power_of_two(),ow.next_power_of_two());
            let mut x=vec![poison;bp*cp*hp*wp];let mut expected=vec![0;bp*cp*ohp*owp];
            for b in 0..batch {for c in 0..channels {for h in 0..ih {for w in 0..iw {
                x[w+wp*(h+hp*(c+cp*b))]=((17*b+11*c+3*h+5*w)%29) as i128-25;
            }}}}
            for b in 0..batch {for c in 0..channels {for h in 0..oh {for w in 0..ow {
                let mut m=i128::MIN;
                for a in 0..kh {for z in 0..kw {m=m.max(x[w*sw+z+wp*(h*sh+a+hp*(c+cp*b))]);}}
                expected[w+owp*(h+ohp*(c+cp*b))]=m;
            }}}}
            let shape=if batch==1 {vec![channels,ih,iw]}else{vec![batch,channels,ih,iw]};
            let mut g=DagBuilder::new();let x=input(&mut g,shape,x,*crate::SF_LOG);
            let y=if regular {g.maxpool2d(x,kh,kw)}else{g.maxpool_general(x,kh,kw,sh,sw)};
            let(d,mut w)=g.compile();assert!(forward(&d,&mut w,&[]));assert_eq!(values(&w,y),expected);
            assert_eq!(w[y][0].sf,*crate::SF_LOG);
        }
    }
}
#[test]
fn maxpool_storage_padding_selector_and_claimed_output_are_constrained() {
    let mut g=DagBuilder::new();let mut x=vec![1i128<<60;64];
    for c in 0..3 {for h in 0..3 {for w in 0..3 {x[w+4*(h+4*c)]=-1;}}}
    let x=input(&mut g,vec![3,3,3],x,0);let y=g.maxpool_general(x,1,1,1,1);
    let(d,w)=g.compile();let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
    let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));
    let mut bad=values(&honest,h);bad[3]=1;assert!(!forward(&d,&mut w.clone(),&[(h,bad)]));
    let advice=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::GeneralMaxPoolHelper(_))).unwrap().outputs[0];
    let mut bad=values(&honest,advice);bad[3]=1;assert!(!forward(&d,&mut w.clone(),&[(advice,bad)]));
    // Nonpower kernel has an unused selector column, also explicitly excluded.
    let mut g=DagBuilder::new();let x=input(&mut g,vec![1,1,3],vec![-3,-2,-1,1i128<<60],0);
    let y=g.maxpool_general(x,1,3,1,1);let(d,w)=g.compile();
    let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
    assert!(!forward(&d,&mut w.clone(),&[(h,vec![0,0,0,1]),(y,vec![0])]));
}
#[test]
fn maxpool_explicit_zero_pad_is_not_negative_infinity() {
    let mut g=DagBuilder::new();let x=input(&mut g,vec![1,2,2],vec![-5,-4,-3,-2],0);
    let p=g.pad(x,1,1);let y=g.maxpool_general(p,2,2,1,1);let(d,mut w)=g.compile();
    assert!(forward(&d,&mut w,&[]));let v=values(&w,y);
    assert_eq!((v[0],v[1],v[4],v[5],v[10]),(0,0,0,-2,0));
}
#[test]
fn maxpool_signed_domain_boundaries_and_invalid_geometry() {
    let b=super::maxpool::VALUE_BOUND;
    let(d,w,y,_)=raw_case(false,vec![-b,b,0,b-1]);let mut good=w.clone();
    assert!(forward(&d,&mut good,&[]));assert_eq!(values(&good,y),[b]);
    let(d,w,_,_)=raw_case(false,vec![-b,b+1,0,1]);assert!(!forward(&d,&mut w.clone(),&[]));
    for (kh,kw,sh,sw) in [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0),(3,1,1,1)] {
        assert!(std::panic::catch_unwind(||{let mut g=DagBuilder::new();let x=g.input(vec![1,2,2],DataType::Float);g.maxpool_general(x,kh,kw,sh,sw);}).is_err());
    }
}
#[test]
fn maxpool_registered_public_support_bit_sources_and_cost_census() {
    use crate::morpheus::{piop::{Graph,Config},statement,params::Profile,keys::KeySeed};
    let mut g=DagBuilder::new();let x=input(&mut g,vec![1,3,5],vec![0;32],0);let y=g.maxpool_general(x,2,3,1,2);
    let(d,mut w)=g.compile();assert!(forward(&d,&mut w,&[]));
    let mut graph=Graph::new(&d,&w,Config{tcl:4,slots_log:12,req_planes_log:6});
    let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
    assert_eq!(w[h][0].data_type,DataType::Bool);
    assert!(graph.outputs.iter().any(|(e,_)|*e==y));assert!(!graph.committed[y]);assert!(graph.committed[h]);assert!(!graph.model[h]);
    for (&e,v) in &d.program_literals {assert_eq!(&**graph.public.get(&e).unwrap(),v);assert!(!graph.model[e]);}
    let before=statement::model_words(&graph,&Profile::toy(),&KeySeed([7;8]),11,false,4).unwrap();
    let (&mask,v)=d.program_literals.iter().find(|(_,v)|v.len()==32 && v.iter().all(|&v|v<=1)).unwrap();
    let mut bad=v.clone();bad[0]^=1;graph.public.insert(mask,std::sync::Arc::new(bad));
    assert!(statement::model_words(&graph,&Profile::toy(),&KeySeed([7;8]),11,false,4).is_err());
    let pzc=d.nodes.iter().filter(|n|matches!(n.kind,BasicBlockType::ProductZeroCheck(_))).count();
    let ranges=d.nodes.iter().filter(|n|matches!(n.kind,BasicBlockType::NonNegative(_))).count();
    println!("MAXPOOL_CENSUS K=6 padded_output=4 selector_cells=32 nodes={} edges={} range_roots={} pzc_roots={} public_literals={} public_cells={} private_dense_sources={} bit_sources={} statement_words={}",
        d.nodes.len(),d.num_edges(),ranges,pzc,d.program_literals.len(),d.program_literals.values().map(Vec::len).sum::<usize>(),
        (0..d.num_edges()).filter(|&e|graph.committed[e] && !graph.public.contains_key(&e)).count(),
        (0..d.num_edges()).filter(|&e|graph.committed[e] && w[e][0].data_type==DataType::Bool).count(),before.len());
    assert_eq!(ranges,18);assert_eq!(pzc,4);
}

#[test]
fn maxpool_valid_padding_negative_borders_odd_shapes_batch_and_poison() {
    for (batch,channels,ih,iw,kh,kw,sh,sw,ph,pw) in [
        (1usize,1usize,3usize,5usize,5usize,5usize,1usize,1usize,2usize,2usize),
        (3,3,3,5,3,3,1,2,1,1),(1,3,5,7,2,3,2,2,1,1),
    ] {
        for poison in [1i128<<60,-(1i128<<60)] {
            let (bp,cp,hp,wp)=(batch.next_power_of_two(),channels.next_power_of_two(),ih.next_power_of_two(),iw.next_power_of_two());
            let (oh,ow)=((ih+2*ph-kh)/sh+1,(iw+2*pw-kw)/sw+1);
            let (ohp,owp)=(oh.next_power_of_two(),ow.next_power_of_two());
            let mut xs=vec![poison;bp*cp*hp*wp];let mut expected=vec![0;bp*cp*ohp*owp];
            for b in 0..batch {for c in 0..channels {for h in 0..ih {for w in 0..iw {
                xs[w+wp*(h+hp*(c+cp*b))]=-100+((11*b+7*c+3*h+w)%31) as i128;
            }}}}
            for b in 0..batch {for c in 0..channels {for h in 0..oh {for w in 0..ow {
                let mut m=None;
                for a in 0..kh {for z in 0..kw {
                    let hh=(h*sh+a) as isize-ph as isize;let ww=(w*sw+z) as isize-pw as isize;
                    if hh>=0 && hh<ih as isize && ww>=0 && ww<iw as isize {
                        let v=xs[ww as usize+wp*(hh as usize+hp*(c+cp*b))];m=Some(m.map(|m:i128|m.max(v)).unwrap_or(v));
                    }
                }}
                expected[w+owp*(h+ohp*(c+cp*b))]=m.unwrap();
            }}}}
            let mut g=DagBuilder::new();let shape=if batch==1 {vec![channels,ih,iw]} else {vec![batch,channels,ih,iw]};
            let x=input(&mut g,shape,xs,*crate::SF_LOG);let y=g.maxpool_general_padded(x,kh,kw,sh,sw,ph,pw);
            let(d,mut w)=g.compile();assert!(forward(&d,&mut w,&[]));assert_eq!(values(&w,y),expected);
            assert_eq!(w[y][0].sf,*crate::SF_LOG);assert!(values(&w,y)[0]<0);
        }
    }
    // A border-only window is not represented by an invented zero or sentinel.
    assert!(std::panic::catch_unwind(||{let mut g=DagBuilder::new();let x=g.input(vec![1,1,1],DataType::Float);g.maxpool_general_padded(x,1,1,1,1,1,1);g.compile();}).is_err());
    for shape in [vec![0,3,2,2],vec![3,0,2,2],vec![3,3,0,2]] {
        assert!(std::panic::catch_unwind(||{let mut g=DagBuilder::new();let x=g.input(shape,DataType::Float);g.maxpool2d(x,1,1);}).is_err());
    }
}
#[test]
fn maxpool_raw_public_window_support_excludes_poison_and_rejects_private_masks() {
    use crate::basicblock::MaskedMaxPoolHelper;
    for poison in [1i128<<60,-(1i128<<60)] {
        let mut g=DagBuilder::new();let x=input(&mut g,vec![1,2,2],vec![poison,-3,-7,poison],0);
        let mask=g.public_literal(Witness::new(vec![1,4],vec![0,1,1,0].into_iter().map(int_to_f).collect(),DataType::Bool,0,Role::Input));
        let p=GeneralMaxPoolHelper{channels:1,input_h:2,input_w:2,kernel_h:2,kernel_w:2,stride_h:1,stride_w:1};
        let y=g.add_gkr_node(vec![x,mask],BasicBlockType::MaskedMaxPoolHelper(MaskedMaxPoolHelper{pool:p}))[0];
        g.init_values.push(Some(Witness::new_wo_data(vec![1,1,1],DataType::Uint,0,Role::Output)));
        let(d,w)=g.compile();let mut good=w.clone();assert!(forward(&d,&mut good,&[]));assert_eq!(values(&good,y),[-3]);
        let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
        assert!(!forward(&d,&mut w.clone(),&[(y,vec![poison]),(h,vec![1,0,0,0])]));
    }
    assert!(std::panic::catch_unwind(||{
        let mut g=DagBuilder::new();let x=g.input(vec![1,2,2],DataType::Float);let mask=g.input(vec![1,4],DataType::Bool);
        let p=GeneralMaxPoolHelper{channels:1,input_h:2,input_w:2,kernel_h:2,kernel_w:2,stride_h:1,stride_w:1};
        g.add_gkr_node(vec![x,mask],BasicBlockType::MaskedMaxPoolHelper(MaskedMaxPoolHelper{pool:p}));
        g.init_values.push(Some(Witness::new_wo_data(vec![1,1,1],DataType::Uint,*crate::SF_LOG,Role::Output)));g.compile();
    }).is_err());
}

#[test]
fn yolo_actual_sppf_pooling_stage_negative_border_reference_and_scale_metadata() {
    let sf=*crate::SF_LOG;let q=1i128<<sf;
    let mut g=DagBuilder::new();let mut raw=vec![1i128<<60;32];let mut reference=vec![0;15];
    for h in 0..3 {for w in 0..5 {let v=(-19+(h*5+w) as i128)*q;raw[w+8*h]=v;reference[w+5*h]=v;}}
    let x=input(&mut g,vec![1,3,5],raw,sf);let stages=super::yolo::sppf_pool(&mut g,x);
    let mut expected=Vec::new();
    for _ in 0..3 {
        let previous=reference.clone();
        for h in 0usize..3 {for w in 0usize..5 {
            reference[w+5*h]=(h.saturating_sub(2)..(h+3).min(3)).flat_map(|ih|
                (w.saturating_sub(2)..(w+3).min(5)).map(move |iw|iw+5*ih)).map(|i|previous[i]).max().unwrap();
        }}
        let mut padded=vec![0;32];for h in 0..3 {for w in 0..5 {padded[w+8*h]=reference[w+5*h];}}expected.push(padded);
    }
    for &e in &stages {assert_eq!(g.init_values[e].as_ref().unwrap().sf,sf);}
    let(d,mut w)=g.compile();assert!(forward(&d,&mut w,&[]));
    for (i,&e) in stages.iter().enumerate(){assert_eq!(values(&w,e),expected[i]);assert!(values(&w,e)[0]<0);}
    for asym in [false,true] {
        let mut g=DagBuilder::new();let x=input(&mut g,vec![3,3,2,2],vec![3*q;64],sf);
        let p=if asym {g.pad_asym(x,1,2,2,1)}else{g.pad(x,1,1)};
        assert_eq!(g.init_values[p].as_ref().unwrap().sf,sf);
        let y=g.maxpool_general(p,2,2,1,1);assert_eq!(g.init_values[y].as_ref().unwrap().sf,sf);
        let two=g.integer_literal(g.init_values[y].as_ref().unwrap().shape.clone(),2,0);
        let z=g.einsum("abcd,abcd->abcd".into(),vec![y,two],false)[0];
        assert_eq!(g.init_values[z].as_ref().unwrap().sf,sf);
        let(d,mut w)=g.compile();assert!(forward(&d,&mut w,&[]));assert_eq!(w[z][0].sf,sf);
    }
}

#[test]
fn pillar_maxpool_raw_and_wrapper_exact_attainment_and_source_type() {
    use crate::basicblock::PillarMaxPool;
    use crate::morpheus::piop::{Graph,Config,request_source_is_bits};
    for raw in [false,true] {
        let mut b=DagBuilder::new();let x=input(&mut b,vec![1,3,2],vec![0;8],*crate::SF_LOG);
        let out=if raw {
            let y=b.add_gkr_node(vec![x],BasicBlockType::PillarMaxPool(PillarMaxPool {n_pillars:1,max_points:3,features:2}))[0];
            b.init_values.push(Some(Witness::new_wo_data(vec![1,2],DataType::Uint,*crate::SF_LOG,Role::Output)));y
        }else {b.pillar_maxpool(x,1,3,2)};
        let(d,w)=b.compile();let y=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::PillarMaxPool(_))).unwrap().outputs[0];
        let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
        let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));assert_eq!(values(&honest,out),[0,0]);
        assert!(!forward(&d,&mut w.clone(),&[(y,vec![1,1])]));
        assert!(!forward(&d,&mut w.clone(),&[(h,vec![0;8])]));
        let graph=Graph::new(&d,&honest,Config{tcl:4,slots_log:12,req_planes_log:6});
        assert!(graph.committed[y]);assert!(!request_source_is_bits(&graph,y));assert!(request_source_is_bits(&graph,h));
        if !raw {assert!(graph.outputs.iter().any(|(e,_)|*e==out));assert!(!graph.committed[out]);}
    }
}
#[test]
fn pillar_maxpool_nonpower_negative_padding_scale_and_geometry() {
    for poison in [1i128<<60,-(1i128<<60)] {
        let mut xs=vec![poison;64];let mut expected=vec![0;16];
        for n in 0..3 {for d in 0..3 {let mut m=i128::MIN;for t in 0..3 {
            let v=-1-(n*11+t*7+d*3) as i128;xs[d+4*(t+4*n)]=v;m=m.max(v);
        }expected[d+4*n]=m;}}
        let mut b=DagBuilder::new();let x=input(&mut b,vec![3,3,3],xs,*crate::SF_LOG);let y=b.pillar_maxpool(x,3,3,3);
        assert_eq!(b.init_values[y].as_ref().unwrap().sf,*crate::SF_LOG);
        let(d,mut w)=b.compile();assert!(forward(&d,&mut w,&[]));assert_eq!(values(&w,y),expected);assert_eq!(w[y][0].sf,*crate::SF_LOG);
    }
    for (n,t,d) in [(0,3,3),(3,0,3),(3,3,0)] {assert!(std::panic::catch_unwind(|| {
        let mut b=DagBuilder::new();let x=b.input(vec![n,t,d],DataType::Float);b.pillar_maxpool(x,n,t,d);
    }).is_err());}
}
