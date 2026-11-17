//! Executed CPU forward and production-helper identities are distinct from
//! authored masked-PIOP replay tests (which require the genuine CUDA link).
use super::*;
use crate::basicblock::{BasicBlock,conv::*};
use crate::dag::{DagBuilder,DataType};
use crate::util::arith::int_to_f;

fn point(n: usize, seed:u64) -> Vec<K8> { (0..n).map(|i| K8([seed+i as u64,2,3,5,7,11,13,17])).collect() }
fn run(kind: &BasicBlockType, ins:&[&Witness]) -> Vec<Witness> {
    match kind {
        BasicBlockType::Conv3D(f)=>f.run(ins), BasicBlockType::DepthwiseConv2D(f)=>f.run(ins),
        BasicBlockType::ConvTranspose2D(f)=>f.run(ins), BasicBlockType::ConvTranspose3D(f)=>f.run(ins),
        BasicBlockType::FlattenKernel(f)=>f.run(ins), BasicBlockType::FlattenKernel3D(f)=>f.run(ins),
        BasicBlockType::ChangeShape(f)=>f.run(ins), _=>panic!("unexpected fixture block"),
    }
}
fn add(b:&mut DagBuilder, kind:BasicBlockType, inputs:Vec<usize>) -> Vec<usize> {
    let ins:Vec<_> = inputs.iter().map(|&e|b.init_values[e].as_ref().unwrap()).collect();
    let outs=run(&kind,&ins);let edges=b.add_gkr_node(inputs,kind);
    assert_eq!(edges.len(),outs.len());
    for w in outs { b.init_values.push(Some(w)); }
    edges
}
fn reshape(b:&mut DagBuilder, e:usize, shape:Vec<usize>) -> usize {
    let out=b.change_shape(e,shape);let n=b.nodes.last().unwrap();
    let w=run(&n.kind,&[b.init_values[e].as_ref().unwrap()]).remove(0);b.init_values[out]=Some(w);out
}
fn witness(shape:Vec<usize>,role:Role,seed:usize) -> Witness {
    let len:usize=shape.iter().map(|n|n.next_power_of_two()).product();
    Witness::new(shape,(0..len).map(|i|int_to_f(((i*7+seed)%23) as i128-11)).collect(),DataType::Float,0,role)
}
struct Fixture { dag:Dag, wit:Vec<Vec<Witness>>, x:usize, raw:usize, flat:usize, y:usize, kind:BasicBlockType, ins:Vec<usize>, outs:Vec<usize>, steps:Vec<usize>, ci:usize, co:usize, wk:WKind }
fn fixture(family:usize, mode:usize) -> Fixture {
    let (ci,co) = if family==1 {(3,3)} else {(3,5)};
    let (kind,ins,outs,steps,wk)=match family {
        0=>{let f=Conv3D::new_strided(ci,co,2,2,3,3,4,5,1,2,2);let o=vec![f.w_out,f.h_out,f.d_out];(BasicBlockType::Conv3D(f),vec![5,4,3],o,vec![2,2,1],WKind::Normal)},
        1=>{let f=DepthwiseConv2D::new_strided(ci,2,3,4,5,2,2);let o=vec![f.w_out,f.h_out];(BasicBlockType::DepthwiseConv2D(f),vec![5,4],o,vec![2,2],WKind::Depthwise)},
        2=>{let f=ConvTranspose2D::new(ci,co,2,3,3,2,2,3);let o=vec![f.w_out,f.h_out];(BasicBlockType::ConvTranspose2D(f),vec![2,3],o,vec![3,2],WKind::Transpose)},
        3=>{let f=ConvTranspose3D::new(ci,co,2,2,3,2,3,2,2,2,3);let o=vec![f.w_out,f.h_out,f.d_out];(BasicBlockType::ConvTranspose3D(f),vec![2,3,2],o,vec![3,2,2],WKind::Transpose)},
        _=>unreachable!(),
    };
    let spec=family_geometry(&kind);
    let mut b=DagBuilder::new();
    let mut xs=vec![ci];xs.extend(ins.iter().rev());
    let x=b.committed_input(witness(xs,Role::Input,13));
    let flat_shape=vec![spec.outer,spec.inner,spec.extent];
    let (raw,flat)=if mode==0 {
        let w=b.param(witness(flat_shape,Role::Constant,17));(w,w)
    } else {
        let outer=spec.outer+if mode==5 {1} else {0};
        assert_eq!(lb(outer),lb(spec.outer),"channel mismatch stays in the same padded arity");
        let mut shape=vec![outer,spec.inner];shape.extend(spec.dims.iter().rev());
        let raw=b.param(witness(shape,Role::Constant,17));
        let fk=if spec.dims.len()==3 { BasicBlockType::FlattenKernel3D(FlattenKernel3D {stride_h:spec.strides[2],stride_w:spec.strides[1]+usize::from(mode==4),kd:spec.dims[2],kh:spec.dims[1],kw:spec.dims[0],c_out:outer,c_in:spec.inner}) }
        else {BasicBlockType::FlattenKernel(FlattenKernel {s_w:spec.strides[1],kh:spec.dims[1],kw:spec.dims[0],c_out:outer,c_in:spec.inner,dilation_h:1,dilation_w:if mode==4 {2}else{1}})};
        let mut flat=add(&mut b,fk,vec![raw])[0];
        if mode==2 {let n:usize=flat_shape.iter().map(|&d|lb(d)).sum();flat=reshape(&mut b,flat,vec![1usize<<n]);flat=reshape(&mut b,flat,flat_shape.clone());}
        if mode==3 {flat=reshape(&mut b,flat,vec![1]);flat=reshape(&mut b,flat,flat_shape.clone());}
        (raw,flat)
    };
    let y=add(&mut b,kind.clone(),vec![x,flat])[0];
    let (dag,wit)=b.compile();
    Fixture {dag,wit,x,raw,flat,y,kind,ins,outs,steps,ci,co,wk}
}
fn check(f:&Fixture,mode:usize) {
    let g=Graph::new(&f.dag,&f.wit,Config {tcl:4,slots_log:12,req_planes_log:6});
    let (source,flat)=family_kernel(&g,f.flat,&f.kind);
    if mode==1 || mode==2 {assert_eq!(source,f.raw);assert!(flat.is_none());}
    else {assert_eq!(source,f.flat);assert!(flat.is_some());}
    let spec=family_geometry(&f.kind);
    let (sk,si,so)=(spec.dims.iter().map(|&n|lb(n)).sum::<usize>(),f.ins.iter().map(|&n|lb(n)).sum::<usize>(),f.outs.iter().map(|&n|lb(n)).sum::<usize>());
    let (lci,lco)=(lb(f.ci),lb(f.co));let r=point(so+lco,23);let rd=&r[so..];
    let rels:Vec<_>=(0..spec.dims.len()).map(|a|Rel {ob:lb(f.outs[a]),kb:lb(spec.dims[a]),ib:lb(f.ins[a]),t:if f.wk==WKind::Transpose {trip_tconv(f.outs[a],spec.dims[a],f.ins[a],f.steps[a])} else {trip_conv(f.outs[a],spec.dims[a],f.ins[a],f.steps[a],1)}}).collect();
    let mut off=0;let ro:Vec<_>=rels.iter().map(|d|{let v=r[off..off+d.ob].to_vec();off+=d.ob;v}).collect();
    let xv=witness_vals(&f.wit[f.x][0]);let wv=witness_vals(&f.wit[source][0]);let yf=witness_vals(&f.wit[f.y][0]);
    let taps=flat.as_ref().map(|r|r.ib).unwrap_or(sk);
    let (cin_off,cout_off,nw)=conv_weight_offsets(f.wk,taps,lci,lco);
    assert_eq!(nw,g.nvars[source]);
    let wt=conv_weight_table(&wv,nw,sk,lci,f.wk,cout_off,rd,f.co,flat.as_ref());
    let at=conv_patch_table(&xv,&rels,&ro,lci,1,&[]);
    let ed=eq_table(rd);
    let rhs=(0..f.ci).flat_map(|c|(0..1usize<<sk).map(move |k|k+(c<<sk))).fold(K8::ZERO,|a,i| {
        let z=at[i].mul(&wt[i]);a.add(&if f.wk==WKind::Depthwise {z.mul(&ed[i>>sk])} else {z})
    });
    assert_eq!(mle(&yf,&r),rhs,"forward versus production table identity family={:?} mode={mode}",f.wk);
    // Exact terminal weight map identity at independently non-base challenges.
    let (xk,xc)=(point(sk,51),point(lci,71));let mut kr=xk.clone();kr.extend_from_slice(&xc);
    let ek=eq_table(&xk);let ec=eq_table(&xc);let mut linked=K8::ZERO;
    let map=flat.clone().unwrap_or_else(||Rel::map(sk,sk,(0..1usize<<sk).map(|k|(k,k))));
    for c in 0..1usize<<lci {for &(k,_,j) in &map.t {
        if f.wk==WKind::Depthwise {linked=linked.add(&ek[k].mul(&ec[c]).scale(wv[j+(c<<cout_off)]));}
        else {for d in 0..f.co {linked=linked.add(&ek[k].mul(&ec[c]).mul(&ed[d]).scale(wv[j+(c<<cin_off)+(d<<cout_off)]));}}
    }}
    assert_eq!(tensor::mle_k(&wt,&kr),linked,"actual source weight map");
    // Patch contraction -> actual input: each sparse table is the production map.
    let mut ko=0;let tabs:Vec<_>=rels.iter().zip(&ro).map(|(rel,ro)|{let t=rel.table(ro,&xk[ko..ko+rel.kb]);ko+=rel.kb;t}).collect();
    let mut linked=K8::ZERO;
    for c in 0..1usize<<lci {for i in 0..1usize<<si {
        let mut rem=i;let mut p=ec[c];for (a,rel) in rels.iter().enumerate() {p=p.mul(&tabs[a][rem&((1<<rel.ib)-1)]);rem>>=rel.ib;}
        linked=linked.add(&p.scale(xv[i+(c<<si)]));
    }}
    assert_eq!(tensor::mle_k(&at,&kr),linked,"actual source input map");
    // Evaluate every nonidentity flatten producer's actual sparse map, including
    // its logical channel masks; this is the fallback's next reduction step.
    for n in &f.dag.nodes {
        let (rel,inner,outer)=match &n.kind {
            BasicBlockType::FlattenKernel(fk)=>(flatten_kernel_rel(fk,g.nvars[n.outputs[0]]-lb(fk.c_in)-lb(fk.c_out)),fk.c_in,fk.c_out),
            BasicBlockType::FlattenKernel3D(fk)=>(flatten_kernel3d_rel(fk,g.nvars[n.outputs[0]]-lb(fk.c_in)-lb(fk.c_out)),fk.c_in,fk.c_out),_=>continue,
        };
        let rr=point(g.nvars[n.outputs[0]],97);let (ei,eo)=(eq_table(&rr[rel.ob..rel.ob+lb(inner)]),eq_table(&rr[rel.ob+lb(inner)..]));
        let t=rel.table(&rr[..rel.ob],&[]);let v=witness_vals(&f.wit[n.inputs[0]][0]);let mut sum=K8::ZERO;
        for o in 0..outer {for i in 0..inner {for k in 0..1usize<<rel.ib {sum=sum.add(&eo[o].mul(&ei[i]).mul(&t[k]).scale(v[k+(i<<rel.ib)+(o<<(rel.ib+lb(inner)))]));}}}
        assert_eq!(mle(&witness_vals(&f.wit[n.outputs[0]][0]),&rr),sum,"actual flatten producer mapping");
    }
}
#[cfg_attr(test, test)]
pub fn family_sources_and_field_maps() {
    for family in 0..4 {for mode in 0..6 {check(&fixture(family,mode),mode);}}
}
#[cfg_attr(test, test)]
pub fn family_geometry_and_collision_guards() {
    let f=FlattenKernel3D {stride_h:1,stride_w:1,kd:2,kh:2,kw:2,c_out:1,c_in:1};
    assert!(std::panic::catch_unwind(||flatten_kernel3d_rel(&f,3)).is_err());
    let f=FlattenKernel3D {stride_h:usize::MAX,stride_w:2,kd:2,kh:2,kw:1,c_out:1,c_in:1};
    assert!(std::panic::catch_unwind(||flatten_kernel3d_rel(&f,3)).is_err());
    for family in 0..4 {
        let mut f=fixture(family,0);match &mut f.kind {
            BasicBlockType::Conv3D(c)=>c.stride_h+=1,
            BasicBlockType::DepthwiseConv2D(c)=>c.stride_w+=1,
            BasicBlockType::ConvTranspose2D(c)=>c.flat_stride+=1,
            BasicBlockType::ConvTranspose3D(c)=>c.flat_stride_h+=1,_=>unreachable!(),
        }
        assert!(std::panic::catch_unwind(||family_geometry(&f.kind)).is_err());
    }
}

fn tiny_counterexample(family:usize) -> Fixture {
    let mut b=DagBuilder::new();let transposed=family>=2;let nd=if family==0 || family==3 {3}else{2};
    let mut ins=vec![if transposed {2}else{3}];ins.resize(nd,1);
    let (kind,wk)=match family {
        0=>(BasicBlockType::Conv3D(Conv3D::new(1,1,1,1,2,1,1,3)),WKind::Normal),
        1=>(BasicBlockType::DepthwiseConv2D(DepthwiseConv2D::new(1,1,2,1,3)),WKind::Depthwise),
        2=>(BasicBlockType::ConvTranspose2D(ConvTranspose2D::new(1,1,1,2,1,2,1,1)),WKind::Transpose),
        3=>(BasicBlockType::ConvTranspose3D(ConvTranspose3D::new(1,1,1,1,2,1,1,2,1,1,1)),WKind::Transpose),_=>unreachable!(),
    };
    let spec=family_geometry(&kind);let mut xs=vec![1];xs.extend(ins.iter().rev());
    let mut v:Vec<_>=(1..=ins[0]).map(|i|int_to_f(i as i128)).collect();v.resize(padded(ins[0]),int_to_f(999));
    let x=b.committed_input(Witness::new(xs,v,DataType::Uint,0,Role::Input));
    let mut ws=vec![1,1];ws.extend(spec.dims.iter().rev());
    let raw=b.param(Witness::new(ws,[2,3].into_iter().map(int_to_f).collect(),DataType::Uint,0,Role::Constant));
    let fk=if nd==3 {BasicBlockType::FlattenKernel3D(FlattenKernel3D {stride_h:spec.strides[2],stride_w:spec.strides[1],kd:1,kh:1,kw:2,c_out:1,c_in:1})}
        else {BasicBlockType::FlattenKernel(FlattenKernel {s_w:spec.strides[1],kh:1,kw:2,c_out:1,c_in:1,dilation_h:1,dilation_w:1})};
    let original_flat=add(&mut b,fk,vec![raw])[0];
    let flat=reshape(&mut b,original_flat,vec![1]);let flat=reshape(&mut b,flat,vec![1,1,2]);
    let wrong=run(&kind,&[b.init_values[x].as_ref().unwrap(),b.init_values[original_flat].as_ref().unwrap()]);
    let old:Vec<_>=witness_vals(&wrong[0]).as_ref().clone();
    assert_eq!(old,if transposed {vec![2,7,6,0]}else{vec![8,13]},"old bypass output");
    let y=add(&mut b,kind.clone(),vec![x,flat])[0];let (dag,wit)=b.compile();
    assert_eq!(witness_vals(&wit[y][0]).as_ref(),&if transposed {vec![2,4,0,0]}else{vec![2,4]},"actual truncated source output");
    let mut outs=vec![if transposed {3}else{2}];outs.resize(nd,1);
    Fixture {dag,wit,x,raw,flat,y,kind,ins,outs,steps:vec![1;nd],ci:1,co:1,wk}
}
#[cfg_attr(test, test)]
pub fn family_truncate_regrow_counterexamples() {
    for family in 0..4 {check(&tiny_counterexample(family),3);}
}

fn prove(g:&Graph,pd:&ProverData)->(Layout,Vec<K8>) {
    let p=crate::morpheus::params::Profile::toy();let layout=Layout::pack(&p,&tensor_specs(g,pd));
    let tr=MTranscript::new("convolution-family-source-test",&[0;64]);
    let mut pc=Ctx::prover(tr,&pd.bank,&layout,g.cfg.slots_log);Walk::new(g,Some(pd)).run(&mut pc);
    { let proof=std::mem::take(&mut pc.proof); drop(pc); (layout,proof) }
}
// Negative cases reuse an honest transcript; no prover abort is verifier rejection.
fn verify(g:&Graph,pd:&ProverData,proof:&(Layout,Vec<K8>))->bool {
    let p=crate::morpheus::params::Profile::toy();let layout=&proof.0;
    let raw=fill_sources(g,pd,&layout,p.ell());let tr=MTranscript::new("convolution-family-source-test",&[0;64]);
    let mut vc=Ctx::verifier(tr,proof.1.clone(),&layout,g.cfg.slots_log);Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && vc.rows.iter().all(|row| {
        let mut acc=K8::ZERO;
        for t in &row.terms {
            let eq=eq_table(&t.r);let mut sum=K8::ZERO;
            for (i,q) in eq.iter().enumerate() {
                let mut plane_sum=K8::ZERO;
                for (b,pw) in t.pw.iter().enumerate() {
                    let digit=raw[t.region.source][t.region.base+(i<<t.region.p)+b];
                    plane_sum=match digit {0=>plane_sum,1=>plane_sum.add(pw),-1=>plane_sum.sub(pw),_=>plane_sum.add(&pw.mul(&kb(digit as i128)))};
                }
                sum=sum.add(&q.mul(&plane_sum));
            }
            acc=acc.add(&sum.mul(&t.scale));
        }acc==row.y
    })
}
#[cfg_attr(test, test)]
pub fn family_masked_piop_source_replay() {
    for family in 0..4 {for mode in 0..6 {
        let f=fixture(family,mode);let mut g=Graph::new(&f.dag,&f.wit,Config {tcl:4,slots_log:12,req_planes_log:6});
        let mut pd=ProverData::new(&g,&f.wit,(0..4096).map(|i|K8::from_base(17*i+9)).collect());
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof),"honest family={family} mode={mode}");
        let mut bad=pd.v(f.y).as_ref().clone();bad[0]=from_i128(1234567);pd.vals[f.y]=Some(Arc::new(bad.clone()));g.outputs=vec![(f.y,Arc::new(bad))];
        assert!(!verify(&g,&pd,&honest_proof),"counterfeit family={family} mode={mode}");
    }}
    for family in 0..4 {
        let f=tiny_counterexample(family);let mut g=Graph::new(&f.dag,&f.wit,Config {tcl:4,slots_log:12,req_planes_log:6});
        let mut pd=ProverData::new(&g,&f.wit,(0..4096).map(|i|K8::from_base(17*i+9)).collect());
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof));
        let bad=if family>=2 {vec![2,7,6,0]}else{vec![8,13]};pd.vals[f.y]=Some(Arc::new(bad.clone()));g.outputs=vec![(f.y,Arc::new(bad))];
        assert!(!verify(&g,&pd,&honest_proof),"original-kernel bypass counterfeit family={family}");
    }
}
