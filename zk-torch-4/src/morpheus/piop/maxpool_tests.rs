//! Authored masked-PIOP positive/negative replay, not an end-to-end PCS proof.
//! Native CUDA linkage may block execution; typecheck is not replay acceptance.
use super::*;
use crate::basicblock::{BasicBlock,BasicBlockType};
use crate::dag::{DagBuilder,DataType};
use crate::util::arith::int_to_f;
use crate::morpheus::params::Profile;
fn rows_hold(rows:&[Row],raw:&[Vec<i8>])->bool {
    rows.iter().all(|row| {
        let mut acc=K8::ZERO;
        for t in &row.terms {
            let eq=crate::morpheus::field8::eq_table(&t.r);let mut sum=K8::ZERO;
            for (i,e) in eq.iter().enumerate() {
                let mut plane_sum=K8::ZERO;
                for (b,pw) in t.pw.iter().enumerate() {
                    let digit=raw[t.region.source][t.region.base+(i<<t.region.p)+b];
                    plane_sum=match digit {0=>plane_sum,1=>plane_sum.add(pw),-1=>plane_sum.sub(pw),_=>plane_sum.add(&pw.mul(&kb(digit as i128)))};
                }
                sum=sum.add(&e.mul(&plane_sum));
            }
            acc=acc.add(&sum.mul(&t.scale));
        }
        acc==row.y
    })
}
fn prove(g:&Graph,pd:&ProverData,lay:&Layout)->Vec<K8> {
    let tr=MTranscript::new("exact-valid-window-maxpool-test",&[0;64]);
    let mut pc=Ctx::prover(tr,&pd.bank,lay,g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);
    pc.proof
}
// Negative cases reuse an honest transcript; no prover abort is verifier rejection.
fn verify(g:&Graph,pd:&ProverData,lay:&Layout,proof:&[K8])->bool {
    let p=Profile::toy();let raw=fill_sources(g,pd,lay,p.ell());
    let tr=MTranscript::new("exact-valid-window-maxpool-test",&[0;64]);
    let mut vc=Ctx::verifier(tr,proof.to_vec(),lay,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}
fn case(padded:bool)->(Dag,Vec<Vec<Witness>>,usize,usize,usize) {
    let mut b=DagBuilder::new();let x=b.committed_input(Witness::new(vec![1,2,2],
        vec![-4,-3,-2,-1].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Input));
    let out=if padded {b.maxpool_general_padded(x,3,3,1,1,1,1)}else{b.maxpool2d(x,2,2)};
    let (d,mut w)=b.compile();
    for &n in &d.topo {let nd=&d.nodes[n];let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);for (&e,v) in nd.outputs.iter().zip(outs){w[e]=vec![v];}}
    let selector=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap();
    let (h,y)=(selector.outputs[0],selector.inputs[1]);(d,w,out,y,h)
}
#[cfg_attr(test, test)]
pub fn exact_maxpool_piop_binds_advice_output_selector_and_terminal_public_root() {
    for padded in [false,true] {
        let(d,w,out,y,h)=case(padded);
        let mut g=Graph::new(&d,&w,Config{tcl:4,slots_log:16,req_planes_log:6});
        assert!(g.outputs.iter().any(|(e,_)|*e==out));
        let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<16]);
        let specs=tensor_specs(&g,&pd);
        assert!(specs.iter().any(|(k,_,p,b)|*k==TKey::Edge(h) && *p==0 && *b));
        let lay=Layout::pack(&Profile::toy(),&specs);let honest_proof=prove(&g,&pd,&lay);assert!(verify(&g,&pd,&lay,&honest_proof));
        let honest_y=pd.vals[y].clone();let honest_h=pd.vals[h].clone();
        let mut bad=honest_y.as_ref().unwrap().as_ref().clone();bad[0]=0;
        pd.vals[y]=Some(Arc::new(bad));assert!(!verify(&g,&pd,&lay,&honest_proof));pd.vals[y]=honest_y;
        for mode in 0..3 {
            let mut bad=honest_h.as_ref().unwrap().as_ref().clone();
            if mode==0 {bad.fill(0);}else if mode==1 {bad[0]=1;}else{bad[0]=2;}
            pd.vals[h]=Some(Arc::new(bad));
            if mode==2 {
                let err=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||tensor_specs(&g,&pd))).expect_err("non-bit source admission must reject");
                let message=err.downcast_ref::<String>().map(String::as_str).or_else(||err.downcast_ref::<&str>().copied()).unwrap_or("");
                assert!(message.contains("declared bit tensor holds a non-bit value"));
                println!("OBSERVED source-format admission rejection for h=2; not verifier rejection");
                continue;
            }
            // Keep the HONEST registered layout, including h's one bit plane.
            // No tensor_specs host assertion is counted as verifier rejection.
            assert!(!verify(&g,&pd,&lay,&honest_proof),"selector mutation {mode}, padded={padded}");
        }
        pd.vals[h]=honest_h;
        let mut bad=pd.vals[out].as_ref().unwrap().as_ref().clone();bad[0]=0;
        let bad=Arc::new(bad);pd.vals[out]=Some(bad.clone());g.outputs=vec![(out,bad)];
        assert!(!verify(&g,&pd,&lay,&honest_proof));
    }
}
#[cfg_attr(test, test)]
pub fn exact_maxpool_piop_rejects_changed_registered_support() {
    let(d,w,_,_,h)=case(true);let g=Graph::new(&d,&w,Config{tcl:4,slots_log:16,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<16]);
    let lay=Layout::pack(&Profile::toy(),&tensor_specs(&g,&pd));let honest_proof=prove(&g,&pd,&lay);assert!(verify(&g,&pd,&lay,&honest_proof));
    let sel=d.producers[h].unwrap();let mask=d.nodes[sel].inputs[2];
    let mut bad=pd.vals[mask].as_ref().unwrap().as_ref().clone();bad[0]^=1;
    pd.vals[mask]=Some(Arc::new(bad));assert!(!verify(&g,&pd,&lay,&honest_proof));
}
#[cfg_attr(test,test)]
pub fn honest_prover_refuses_recomputed_false_max() {
    // Recompute every downstream edge after forging y=1 on an all-zero
    // window. Dominance remains valid, isolating the missing attainment root.
    let mut b=DagBuilder::new();let x=b.committed_input(Witness::new(vec![1,2,2],vec![int_to_f(0);4],DataType::Float,0,Role::Input));
    let out=b.maxpool2d(x,2,2);let(d,mut w)=b.compile();
    let y=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolHelper(_))).unwrap().outputs[0];
    for &n in &d.topo {let nd=&d.nodes[n];let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outs=nd.kind.run(&ins);
        for (&e,mut v) in nd.outputs.iter().zip(outs){if e==y {v=Witness::new(v.shape,vec![int_to_f(1)],v.data_type,v.sf,v.role);}w[e]=vec![v];}}
    let g=Graph::new(&d,&w,Config{tcl:4,slots_log:16,req_planes_log:6});
    assert_eq!(g.outputs.iter().find(|(e,_)|*e==out).unwrap().1[0],1);
    let pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<16]);
    let lay=Layout::pack(&Profile::toy(),&tensor_specs(&g,&pd));
    let abort=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||prove(&g,&pd,&lay)));
    let err=abort.expect_err("honest prover refusal, NOT verifier rejection");
    let message=err.downcast_ref::<String>().map(String::as_str).or_else(||err.downcast_ref::<&str>().copied()).unwrap_or("");
    assert!(message.starts_with("prover self-check failed:"),"unexpected unrelated panic: {message}");
    println!("OBSERVED honest-prover abort for recomputed false max; not verifier rejection");
}

#[cfg_attr(test, test)]
pub fn pillar_maxpool_piop_attainment_source_width_and_public_root() {
    let mut b=DagBuilder::new();let x=b.committed_input(Witness::new(vec![1,3,2],
        vec![-4,-8,-3,-7,-2,-6,0,0].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Input));
    let out=b.pillar_maxpool(x,1,3,2);let(d,mut w)=b.compile();
    for &n in &d.topo {let nd=&d.nodes[n];let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);for (&e,v) in nd.outputs.iter().zip(outs){w[e]=vec![v];}}
    let y=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::PillarMaxPool(_))).unwrap().outputs[0];
    let h=d.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::MaxPoolSelector(_))).unwrap().outputs[0];
    let mut g=Graph::new(&d,&w,Config{tcl:4,slots_log:16,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<16]);let specs=tensor_specs(&g,&pd);
    assert!(specs.iter().any(|(k,_,p,b)|*k==TKey::Edge(y) && *p==6 && !*b));
    assert!(specs.iter().any(|(k,_,p,b)|*k==TKey::Edge(h) && *p==0 && *b));
    let lay=Layout::pack(&Profile::toy(),&specs);let honest_proof=prove(&g,&pd,&lay);assert!(verify(&g,&pd,&lay,&honest_proof));
    let old=pd.vals[y].clone();pd.vals[y]=Some(Arc::new(vec![0,0]));assert!(!verify(&g,&pd,&lay,&honest_proof));pd.vals[y]=old;
    let bad=Arc::new(vec![0,0]);pd.vals[out]=Some(bad.clone());g.outputs=vec![(out,bad)];assert!(!verify(&g,&pd,&lay,&honest_proof));
}
