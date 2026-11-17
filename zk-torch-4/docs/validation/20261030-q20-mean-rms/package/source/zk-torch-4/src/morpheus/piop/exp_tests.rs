//! CPU PIOP prover/verifier replay tests. These check the actual emitted link
//! rows against raw sources; they do NOT stand in for an end-to-end PCS proof.
use super::*;
use crate::basicblock::{BasicBlock, ExpHelper};
use crate::dag::{DagBuilder, DataType};
use crate::morpheus::{keys::KeySeed, params::Profile, statement};
use crate::util::arith::int_to_f;

fn standalone() -> (Dag, Vec<Vec<Witness>>) {
    let mut b = DagBuilder::new();
    let l = exp_ln2();
    let x = b.committed_input(Witness::new(vec![1], vec![int_to_f(-l)], DataType::Float, *crate::SF_LOG, Role::Input));
    // Deliberately bypass b.exp(): compile must guard the raw primitive too.
    b.add_gkr_node(vec![x], BasicBlockType::ExpHelper(ExpHelper::new(8)));
    b.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, *crate::SF_LOG, Role::Output)));
    b.init_values.push(Some(Witness::new_wo_data(vec![1], DataType::Float, 0, Role::Auxiliary)));
    let (d, mut w) = b.compile();
    for &n in &d.topo {
        let node = &d.nodes[n];
        let ins = node.inputs.iter().map(|&e| &w[e][0]).collect::<Vec<_>>();
        let outs = node.kind.run(&ins);
        for (&e, v) in node.outputs.iter().zip(outs) { w[e] = vec![v]; }
    }
    (d,w)
}

fn rows_hold(rows: &[Row], raw: &[Vec<i8>]) -> bool {
    rows.iter().all(|row| {
        let mut acc = K8::ZERO;
        for t in &row.terms {
            let eq = crate::morpheus::field8::eq_table(&t.r);
            let mut sum = K8::ZERO;
            for (i, e) in eq.iter().enumerate() {
                for (b, pw) in t.pw.iter().enumerate() {
                    let digit = raw[t.region.source][t.region.base + (i << t.region.p) + b];
                    sum = sum.add(&e.mul(pw).mul(&kb(digit as i128)));
                }
            }
            acc = acc.add(&sum.mul(&t.scale));
        }
        acc == row.y
    })
}

fn replay(g: &Graph, pd: &ProverData) -> bool {
    let p = Profile::toy();
    let lay = Layout::pack(&p, &tensor_specs(g,pd));
    let raw = fill_sources(g,pd,&lay,p.ell());
    let tr = MTranscript::new("exp-guard-test", &[0;64]);
    let mut pc = Ctx::prover(tr.clone(), &pd.bank, &lay, g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);
    let mut vc = Ctx::verifier(tr,pc.proof.clone(),&lay,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}

#[test]
fn standalone_exp_has_canonical_verifier_roots_and_rejects_compensated_k_r() {
    let (d,w) = standalone();
    assert_eq!(d.exp_guard_nodes.len(),1);
    let mut g = Graph::new(&d,&w,Config {tcl:4,slots_log:10,req_planes_log:6});
    let mut pd = ProverData::new(&g,&w,vec![K8::ZERO;1<<10]);
    assert!(replay(&g,&pd));
    let n = &d.nodes[0]; let r=n.outputs[0];
    // Old equation still holds: x=r-Lk with (k,r)=(2,L).
    let l=exp_ln2(); pd.sel.insert(0,vec![2]);
    pd.vals[r]=Some(Arc::new(vec![l as u64]));
    g.outputs=vec![(r,Arc::new(vec![l as u64]))];
    let ids=d.exp_guard_nodes[&0];
    let rho=d.nodes[ids[2]].inputs[0]; let slack=d.nodes[ids[3]].inputs[0];
    pd.vals[rho]=Some(Arc::new(vec![from_i128(l/2-l)]));
    pd.vals[slack]=Some(Arc::new(vec![from_i128(l-1-(l/2-l))]));
    // A malicious prover supplies valid bounded selectors, not a constructor
    // panic. No selection can satisfy the negative canonical remainder root.
    pd.sel.insert(ids[2],vec![0]);
    pd.sel.insert(ids[3],vec![((l-1-(l/2-l)) as u64) % (1u64 << g.lookups[&ids[3]])]);
    assert!(!replay(&g,&pd));
}

#[test]
fn immutable_literal_source_mutation_fails_verifier_rows() {
    let (d,w)=standalone();
    let g=Graph::new(&d,&w,Config {tcl:4,slots_log:10,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<10]);
    assert!(replay(&g,&pd));
    let (&e,v)=d.program_literals.iter().find(|(_,v)|v[0]>1).unwrap();
    pd.vals[e]=Some(Arc::new(vec![v[0]-1]));
    assert!(!replay(&g,&pd));
}

#[test]
fn verifier_statement_rejects_removed_guard_or_changed_literal() {
    let (mut d,w)=standalone();
    let guard=d.exp_guard_nodes[&0][2];
    d.exp_guard_nodes.remove(&0);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Graph::new(&d,&w,Config::default()))).is_err());
    let (d,w)=standalone();
    let mut g=Graph::new(&d,&w,Config::default());
    let e=d.nodes[guard].inputs[0];
    // Turning a derived root into a committed edge would bypass reduction.
    g.committed[e]=true;
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    g.committed[e]=false;
    g.lookups.insert(0,40);
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
    g.lookups.insert(0,8);
    let (&literal,values)=d.program_literals.iter().next().unwrap();
    g.public.insert(literal,Arc::new(vec![values[0]+1]));
    assert!(statement::words(&g,&Profile::toy(),&KeySeed([1;8]),11,false,4).is_err());
}

#[test]
fn finite_exp_rejects_modular_scaled_quotient_counterfeit() {
    let mut b=DagBuilder::new();
    let x=b.committed_input(Witness::new(vec![1],vec![int_to_f(0)],DataType::Float,*crate::SF_LOG,Role::Input));
    let out=b.exp(x)[0]; let (d,mut w)=b.compile();
    for &n in &d.topo {
        let nd=&d.nodes[n];
        let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);
        for (&e,v) in nd.outputs.iter().zip(outs) { w[e]=vec![v]; }
    }
    let mut g=Graph::new(&d,&w,Config {tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert!(replay(&g,&pd));
    let add=|a:u64,b:u64| ((a as u128+b as u128)%Q as u128) as u64;
    let sub=|a:u64,b:u64| ((a as u128+Q as u128-b as u128)%Q as u128) as u64;
    let mul=|a:u64,b:u64| ((a as u128*b as u128)%Q as u128) as u64;
    let mut forged=false;
    // Construct a whole field-consistent false execution. Every scale selector
    // is in range; only the new integer interval roots exclude the counterfeit.
    for &n in &d.topo {
        let nd=&d.nodes[n];
        let v=|e:usize| pd.vals[e].as_ref().unwrap()[0];
        let y=match &nd.kind {
            BasicBlockType::ExpHelper(_) => Some(0),
            BasicBlockType::TwoPow(_) => Some(1<<15),
            BasicBlockType::ChangeShape(_) => Some(v(nd.inputs[0])),
            BasicBlockType::Add(_) => Some(add(v(nd.inputs[0]),v(nd.inputs[1]))),
            BasicBlockType::Sub(_) => Some(sub(v(nd.inputs[0]),v(nd.inputs[1]))),
            BasicBlockType::Einsum(_) => Some(nd.inputs.iter().fold(1,|a,&e|mul(a,v(e)))),
            BasicBlockType::ScaleUp(_) => Some(mul(v(nd.inputs[0]),1<<g.lookups[&n])),
            BasicBlockType::ScaleDown(_) => {
                let s=1u64<<g.lookups[&n]; let mut rem=s/2;
                let input=nd.inputs[0];
                let square=d.producers[input].map(|p| {
                    let p=&d.nodes[p]; matches!(p.kind,BasicBlockType::Einsum(_))
                        && p.inputs.len()==2 && p.inputs[0]==p.inputs[1]
                }).unwrap_or(false);
                if square && !forged { rem+=1; forged=true; }
                let y=mul(sub(add(v(input),s/2),rem),invq(s));
                pd.sel.insert(n,vec![rem]); Some(y)
            }
            BasicBlockType::NonNegative(_) => {
                let a=v(nd.inputs[0]); let limit=1u64<<g.lookups[&n];
                pd.sel.insert(n,vec![if a<limit {a} else {0}]); None
            }
            _=>panic!("unexpected exp operator"),
        };
        if let Some(y)=y { pd.vals[nd.outputs[0]]=Some(Arc::new(vec![y])); }
    }
    assert!(forged);
    g.outputs=vec![(out,pd.vals[out].as_ref().unwrap().clone())];
    assert!(!replay(&g,&pd));
}

#[test]
fn padded_exp_verifier_rejects_nonzero_claimed_padding() {
    let mut b=DagBuilder::new();
    // Positive padding must be masked BEFORE the nonpositive helper guard.
    let mut values=vec![int_to_f(0);5]; values.extend(vec![int_to_f(400);3]);
    let x=b.committed_input(Witness::new(vec![5],values,DataType::Float,*crate::SF_LOG,Role::Input));
    let out=b.exp(x)[0]; let (d,mut w)=b.compile();
    for &n in &d.topo {
        let nd=&d.nodes[n]; let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
        let outs=nd.kind.run(&ins);
        for (&e,v) in nd.outputs.iter().zip(outs) {w[e]=vec![v];}
    }
    let mut g=Graph::new(&d,&w,Config {tcl:4,slots_log:12,req_planes_log:6});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<12]);
    assert_eq!(&pd.vals[out].as_ref().unwrap()[5..],&[0;3]);
    assert!(replay(&g,&pd));
    let mut bad=pd.vals[out].as_ref().unwrap().as_ref().clone(); bad[5]=1;
    let bad=Arc::new(bad); pd.vals[out]=Some(bad.clone()); g.outputs=vec![(out,bad)];
    assert!(!replay(&g,&pd));
}

/// Real PIOP replay coverage for the next numerical repair. These tests are
/// authored/typechecked even when missing CUDA linkage blocks execution.
fn normalization_case(is_rms:bool, input:i128, override_advice:Option<i128>) -> bool {
    normalization_case_with_bucket(is_rms,input,override_advice,None)
}
fn normalization_case_with_bucket(is_rms:bool,input:i128,override_advice:Option<i128>,bucket_override:Option<usize>)->bool {
    let mut b=DagBuilder::new();
    let x=b.committed_input(Witness::new(vec![1],vec![int_to_f(input)],DataType::Float,10,Role::Input));
    let y=if is_rms {b.rms_reciprocal_from_mean(x)} else {b.div_const(x,2)[0]};
    let (d,mut w)=b.compile();
    let forward=|w:&mut Vec<Vec<Witness>>,bad:Option<i128>,bad_bucket:Option<usize>| {
        for &n in &d.topo {
            let nd=&d.nodes[n];
            let ins=nd.inputs.iter().map(|&e|&w[e][0]).collect::<Vec<_>>();
            let outs=nd.kind.run(&ins);
            for (&e,mut v) in nd.outputs.iter().zip(outs) {
                if e==y {if let Some(bad)=bad {v=Witness::new(vec![1],vec![int_to_f(bad)],v.data_type,v.sf,v.role);}}
                if matches!(nd.kind,BasicBlockType::RMSBucket(_)) {
                    if let Some(index)=bad_bucket {
                        let mut vals=vec![int_to_f(0);32];vals[index]=int_to_f(1);
                        v=Witness::new(vec![1,32],vals,v.data_type,v.sf,v.role);
                    }
                }
                if bad.is_some() && matches!(nd.kind,BasicBlockType::ReLUHelper(_)) {
                    // Malicious source remains well-typed: a negative r would
                    // make the honest helper emit >1, but that must not turn
                    // this verifier test into a source-fit constructor panic.
                    let bit=if v.data.as_ref().unwrap().index(0).reduce().0==0 {0} else {1};
                    v=Witness::new(vec![1],vec![int_to_f(bit)],v.data_type,v.sf,v.role);
                }
                w[e]=vec![v];
            }
        }
    };
    forward(&mut w,None,None);
    let g=Graph::new(&d,&w,Config{tcl:4,slots_log:10,req_planes_log:5});
    let mut pd=ProverData::new(&g,&w,vec![K8::ZERO;1<<10]);
    if override_advice.is_some() || bucket_override.is_some() {
        forward(&mut w,override_advice,bucket_override);
        // Mutate AFTER honest prover preparation. A constructor panic is not
        // verifier rejection: supply legal selectors even for negative roots.
        for (e,ws) in w.iter().enumerate() {
            if ws[0].poly_type==crate::dag::PolyType::Dense {
                pd.vals[e]=Some(Arc::new(ws[0].data.as_ref().unwrap().evaluations_ref().iter().map(|v|v.reduce().0).collect()));
            }
        }
        for (&node,&bits) in &g.lookups {
            if matches!(d.nodes[node].kind,BasicBlockType::NonNegative(_)) {
                let vals=pd.v(d.nodes[node].inputs[0]);
                pd.sel.insert(node,vals.iter().map(|&v|if v<(1u64<<bits) {v} else {0}).collect());
            }
        }
    }
    replay(&g,&pd)
}

#[test]
fn exact_mean_piop_rejects_both_old_half_tie_counterexamples() {
    for (input,correct,wrong) in [(1,1,0),(-1,0,-1)] {
        assert!(normalization_case(false,input,None));
        assert!(normalization_case(false,input,Some(correct)));
        assert!(!normalization_case(false,input,Some(wrong)));
    }
}

#[test]
fn exact_rms_piop_rejects_sign_zero_and_neighbor_counterexamples() {
    assert!(normalization_case(true,1024,None));
    for bad in [-1024,0,1023,1025] {assert!(!normalization_case(true,1024,Some(bad)));}
    assert!(normalization_case(true,0,None));
    assert!(!normalization_case(true,0,Some(1)));
}

#[test]
fn exact_rms_piop_rejects_wrong_private_bucket_and_forbidden_padding() {
    assert!(normalization_case_with_bucket(true,1024,None,None));
    for wrong in [0,1,10,12,29,31] {
        assert!(!normalization_case_with_bucket(true,1024,None,Some(wrong)));
    }
}

#[test]
fn nonnegative_partial_chunks_preserve_61_62_63_bit_widths() {
    for bits in [61,62,63] {
        let mut b=DagBuilder::new();
        let x=b.committed_input(Witness::new(vec![1],vec![int_to_f(0)],DataType::Int,0,Role::Input));
        let node=b.nodes.len();b.add_nonneg_node_bits(x,bits);
        let (d,w)=b.compile();let g=Graph::new(&d,&w,Config{tcl:4,slots_log:10,req_planes_log:5});
        let chunks=g.chunks(node);
        assert_eq!(chunks.iter().sum::<usize>(),bits);
        assert_eq!(chunks.last(),Some(&(bits%4)));
        assert!(chunks[..chunks.len()-1].iter().all(|&w|w==4));
        // Actual selector table maxima reconstruct exactly2^bits−1, never
        // the rounded-up2^64−1 domain that would invalidate signed lifting.
        let mut shift=0;let mut max=0u128;
        for w in chunks {max+=((1u128<<w)-1)<<shift;shift+=w;}
        assert_eq!(max,(1u128<<bits)-1);
    }
}
