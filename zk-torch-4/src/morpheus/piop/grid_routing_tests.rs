//! Authored masked-PIOP positive/negative replay, not an end-to-end PCS proof.
//! Native CUDA linkage may block execution; typecheck is not replay acceptance.
use super::*;
use crate::basicblock::{BasicBlock,BasicBlockType};
use crate::dag::{DagBuilder,DataType};
use crate::util::arith::int_to_f;
use crate::morpheus::params::Profile;
fn rows_hold(rows:&[Row],raw:&[Vec<i8>])->bool {
    // Exact zero-source identity, checked over every digit before skipping.
    let zero_source:Vec<bool>=raw.iter().map(|s|s.iter().all(|&d|d==0)).collect();
    rows.iter().all(|row| {
        let mut acc=K8::ZERO;
        for t in &row.terms {
            // Preserve the original indexed evaluator's bounds checks even
            // when the entire source is zero.
            let count=1usize.checked_shl(t.r.len() as u32).unwrap();
            let end=t.region.base.checked_add((count-1).checked_shl(t.region.p as u32).unwrap()).unwrap().checked_add(t.pw.len()).unwrap();
            assert!(end<=raw[t.region.source].len());
            if zero_source[t.region.source] { continue; }
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
    let tr=MTranscript::new("exact-private-grid-test",&[0;64]);
    let mut pc=Ctx::prover(tr,&pd.bank,lay,g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);println!("GRID prover {}",pc.census_line());pc.proof
}
fn verify(g:&Graph,pd:&ProverData,lay:&Layout,proof:&[K8])->bool {
    let raw=fill_sources(g,pd,lay,lay.ell);
    let mut vc=Ctx::verifier(MTranscript::new("exact-private-grid-test",&[0;64]),proof.to_vec(),lay,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);println!("GRID verifier {}",vc.census_line());vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}
#[cfg_attr(test, test)]
pub fn private_grid_piop_exact_last_write_and_zero_invalid() { grid_piop_fixture(17,Profile::toy()); }
/// Separate toy smoke with the same relation, inputs and mutations; a smaller
/// public mask bank and existing mini source layout, not the original toy
/// 17-bit-bank test. No PCS authentication or security-level claim.
#[cfg_attr(test, test)]
pub fn private_grid_piop_small_bank_smoke() { grid_piop_fixture(14,Profile::mini()); }
fn grid_piop_fixture(slots_log:usize,profile:Profile) {
    for gather in [false,true] {for forged in [false,true] {
        let mut b=DagBuilder::new();
        let x=b.committed_input(Witness::new(vec![3,1],vec![int_to_f(-3),int_to_f(7),int_to_f(99),int_to_f(0)],DataType::Float,0,Role::Input));
        let coords=b.committed_input(Witness::new(vec![3,2],vec![int_to_f(0),int_to_f(0),int_to_f(0),int_to_f(0),int_to_f(-1),int_to_f(0),int_to_f(0),int_to_f(0)],DataType::Int,0,Role::Input));
        let s=b.scatter_to_bev(x,coords,3,1,1,1);
        let out=if gather {b.gather_from_grid(s,coords,3,1,1,1)}else{s};
        let (d,mut w)=b.compile();
        let y=d.nodes.iter().filter(|n|if gather {matches!(n.kind,BasicBlockType::GatherFromGrid(_))}else{matches!(n.kind,BasicBlockType::ScatterToBEV(_))}).last().unwrap().outputs[0];
        // Coherent forged-output trace: every downstream computed witness is
        // recomputed from the forged advice, not retained honest by accident.
        for &n in &d.topo {let nd=&d.nodes[n];let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outs=nd.kind.run(&ins);
            for (&e,mut v) in nd.outputs.iter().zip(outs){if forged && e==y {let len=v.data.as_ref().unwrap().evaluations_ref().len();v=Witness::new(v.shape,vec![int_to_f(-3);len],v.data_type,v.sf,v.role);}w[e]=vec![v];}}
        if !forged {assert_eq!(witness_vals(&w[out][0]).as_ref(),&if gather {vec![7,7,0,0]}else{vec![7]});}
        let mut g=Graph::new(&d,&w,Config{tcl:4,slots_log,req_planes_log:6});
        assert!(g.outputs.iter().any(|(e,_)|*e==out));
        let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<slots_log]);let lay=Layout::pack(&profile,&tensor_specs(&g,&pd));
        if forged {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(||prove(&g,&pd,&lay))) {
                Ok(proof)=>assert!(!verify(&g,&pd,&lay,&proof)),
                Err(err)=>{let msg=err.downcast_ref::<String>().map(String::as_str).or_else(||err.downcast_ref::<&str>().copied()).unwrap_or("");
                    assert!(msg.contains("prover self-check failed"),"unexpected abort: {msg}");
                    println!("OBSERVED coherent grid forgery refused by prover self-check, not verifier rejection");}
            }
        } else {
            let proof=prove(&g,&pd,&lay);assert!(verify(&g,&pd,&lay,&proof));
            let mut bad=proof.clone();assert!(!bad.is_empty());bad[0]=bad[0].add(&K8::ONE);
            assert!(!verify(&g,&pd,&lay,&bad),"actual mutated proof rejected");
            let orig=pd.vals[y].clone();let mut bad=orig.as_ref().unwrap().as_ref().clone();bad[0]=int_to_f(-3).reduce().0;
            pd.vals[y]=Some(Arc::new(bad));assert!(!verify(&g,&pd,&lay,&proof),"retained honest proof cannot bind changed source");pd.vals[y]=orig;
            let mut bad=pd.vals[out].as_ref().unwrap().as_ref().clone();bad[0]=int_to_f(-3).reduce().0;
            let bad=Arc::new(bad);pd.vals[out]=Some(bad.clone());g.outputs=vec![(out,bad)];
            assert!(!verify(&g,&pd,&lay,&proof),"actual public output mutation rejected");
        }
    }}
}
