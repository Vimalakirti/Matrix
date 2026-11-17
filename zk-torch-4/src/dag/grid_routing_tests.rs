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

#[test]
fn private_grid_scatter_last_write_invalid_padding_and_scale() {
    for sf in [0,10,20] { for raw in [false,true] {
        let mut b=DagBuilder::new();
        let mut vals=vec![1i128<<60;32];
        for p in 0..5 {for c in 0..3 {vals[c+4*p]=100*p as i128+c as i128-13;}}
        let x=input(&mut b,vec![5,3],vals,sf);
        let coords=input(&mut b,vec![5,2],vec![0,1,2,2,0,1,-1,0,3,1,9,9,9,9,9,9],0);
        let out=if raw {let y=b.add_gkr_node(vec![x,coords],BasicBlockType::ScatterToBEV(crate::basicblock::ScatterToBEV{n_pillars:5,features:3,ny:3,nx:3}))[0];
            b.init_values.push(Some(Witness::new_wo_data(vec![3,3,3],DataType::Uint,sf,Role::Output)));y}
            else {b.scatter_to_bev(x,coords,5,3,3,3)};
        let(d,w)=b.compile();let y=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::ScatterToBEV(_))).unwrap().outputs[0];
        let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));
        let mut want=vec![0;64];for c in 0..3 {want[1+16*c]=187+c as i128;want[2+4*2+16*c]=87+c as i128;}
        assert_eq!(values(&honest,out),want);assert_eq!(honest[out][0].sf,sf);
        let mut bad=want.clone();bad[1]=-13;assert!(!forward(&d,&mut w.clone(),&[(y,bad)]));
        let mut bad=want.clone();bad[3]=1;assert!(!forward(&d,&mut w.clone(),&[(y,bad)]));
        let mut bad=want.clone();bad[16*3]=1;assert!(!forward(&d,&mut w.clone(),&[(y,bad)]));
        let graph=crate::morpheus::piop::Graph::new(&d,&honest,crate::morpheus::piop::Config{tcl:4,slots_log:16,req_planes_log:6});
        assert!(!crate::morpheus::piop::request_source_is_bits(&graph,y));
        if !raw {assert!(graph.outputs.iter().any(|(e,_)|*e==out));assert!(!graph.committed[out]);}
    }}
}
#[test]
fn private_grid_gather_invalid_negative_nonpower_and_padding() {
    for raw in [false,true] {
        let mut b=DagBuilder::new();let mut vals=vec![-(1i128<<60);64];
        for c in 0..3 {for h in 0..3 {for w in 0..3 {vals[w+4*h+16*c]=(c*100+h*10+w) as i128-50;}}}
        let g=input(&mut b,vec![3,3,3],vals,16);
        let coords=input(&mut b,vec![5,2],vec![0,1,2,2,-1,0,0,3,3,1,0,0,0,0,0,0],0);
        let out=if raw {let y=b.add_gkr_node(vec![g,coords],BasicBlockType::GatherFromGrid(crate::basicblock::GatherFromGrid{n_points:5,channels:3,grid_h:3,grid_w:3}))[0];
            b.init_values.push(Some(Witness::new_wo_data(vec![5,3],DataType::Uint,16,Role::Output)));y}
            else {b.gather_from_grid(g,coords,5,3,3,3)};
        let(d,w)=b.compile();let y=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::GatherFromGrid(_))).unwrap().outputs[0];
        let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));let mut want=vec![0;32];
        for c in 0..3 {want[c]=c as i128*100-49;want[4+c]=c as i128*100-28;}
        assert_eq!(values(&honest,out),want);assert_eq!(honest[out][0].sf,16);
        for i in [0,8,3,20] {let mut bad=want.clone();bad[i]+=1;assert!(!forward(&d,&mut w.clone(),&[(y,bad)]));}
    }
}
#[test]
fn private_grid_budget_preflight_rejects_before_large_allocation() {
    assert!(std::panic::catch_unwind(||crate::dag::grid_routing::geometry(12000,64,512,512)).is_err());
    for d in [(0,1,1,1),(1,0,1,1),(1,1,0,1),(1,1,1,0)] {
        assert!(std::panic::catch_unwind(||crate::dag::grid_routing::geometry(d.0,d.1,d.2,d.3)).is_err());
    }
}

#[test]
fn private_grid_all_centered_coordinate_extremes() {
    let q=almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as i128;
    for invalid in [-q/2,-1,2,q/2] {
        let mut b=DagBuilder::new();let x=input(&mut b,vec![2,1],vec![-7,9],20);
        let co=input(&mut b,vec![2,2],vec![invalid,0,0,invalid],0);
        let s=b.scatter_to_bev(x,co,2,1,2,2);
        let g=b.gather_from_grid(s,co,2,1,2,2);
        let(d,w)=b.compile();let mut honest=w.clone();assert!(forward(&d,&mut honest,&[]));
        assert_eq!(values(&honest,s),[0,0,0,0]);assert_eq!(values(&honest,g),[0,0]);
    }
}
