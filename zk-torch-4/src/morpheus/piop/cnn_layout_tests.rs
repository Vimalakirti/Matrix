//! Actual masked PIOP replay tests. These are authored/typechecked even when
//! native CUDA symbols prevent execution; no proof acceptance is inferred.
use super::*;
use crate::basicblock::BasicBlock;
use crate::dag::{DagBuilder,DataType};
use crate::util::arith::int_to_f;
use crate::morpheus::params::Profile;

fn rows_hold(rows: &[Row], raw: &[Vec<i8>]) -> bool {
    rows.iter().all(|row| {
        let mut acc=K8::ZERO;
        for t in &row.terms {
            let eq=crate::morpheus::field8::eq_table(&t.r);let mut sum=K8::ZERO;
            for (i,e) in eq.iter().enumerate() {for (b,pw) in t.pw.iter().enumerate() {
                let digit=raw[t.region.source][t.region.base+(i<<t.region.p)+b];
                sum=sum.add(&e.mul(pw).mul(&kb(digit as i128)));
            }}
            acc=acc.add(&sum.mul(&t.scale));
        }
        acc==row.y
    })
}
fn replay(g:&Graph,pd:&ProverData)->bool {
    let p=Profile::toy();let lay=Layout::pack(&p,&tensor_specs(g,pd));
    let raw=fill_sources(g,pd,&lay,p.ell());
    let tr=MTranscript::new("cnn-logical-layout-test",&[0;64]);
    let mut pc=Ctx::prover(tr.clone(),&pd.bank,&lay,g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);
    let mut vc=Ctx::verifier(tr,pc.proof.clone(),&lay,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}
fn flatten_case()->(Dag,Vec<Vec<Witness>>,usize) {
    let mut b=DagBuilder::new();let mut raw=vec![int_to_f(777);32];
    for c in 0..3 {for h in 0..2 {for w in 0..3 {raw[w+4*(h+2*c)]=int_to_f((w+3*(h+2*c)) as i128-9);}}}
    let x=b.committed_input(Witness::new(vec![3,2,3],raw,DataType::Float,0,Role::Input));
    let out=b.flatten_row_major(x);
    let (d,mut w)=b.compile();
    for &n in &d.topo {let nd=&d.nodes[n];let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();let outs=nd.kind.run(&ins);for (&e,v) in nd.outputs.iter().zip(outs) {w[e]=vec![v];}}
    (d,w,out)
}
#[test]
fn logical_flatten_piop_rejects_wrong_logical_and_padding_outputs() {
    let (d,w,out)=flatten_case();
    let mut g=Graph::new(&d,&w,Config{tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert!(replay(&g,&pd));
    let honest=pd.vals[out].as_ref().unwrap().clone();
    for index in [5,18,31] {
        let mut bad=honest.as_ref().clone();bad[index]=from_i128(123);
        let bad=Arc::new(bad);pd.vals[out]=Some(bad.clone());g.outputs=vec![(out,bad)];
        assert!(!replay(&g,&pd),"mutated output {index}");
    }
}
#[test]
fn logical_flatten_piop_rejects_changed_public_gather_factor() {
    let (d,w,_)=flatten_case();let g=Graph::new(&d,&w,Config{tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert!(replay(&g,&pd));
    let (&map,values)=d.program_literals.iter().next().unwrap();
    let mut bad=values.clone();bad[0]^=1;pd.vals[map]=Some(Arc::new(bad));
    assert!(!replay(&g,&pd));
}
