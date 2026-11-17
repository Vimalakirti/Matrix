//! Authored CPU-only semantic and masked-PIOP replay tests. A replay checks
//! verifier messages and emitted source/link rows, not the PCS or CUDA path.
use super::*;
use crate::basicblock::BasicBlock;
use crate::dag::{DagBuilder, DataType};
use crate::morpheus::params::Profile;
use crate::util::arith::int_to_f;

fn fixture(b: usize, ci: usize, co: usize, h: usize, w: usize, k: (usize,usize), stride: (usize,usize), dilation: (usize,usize), poison: i128) -> (Dag, Vec<Vec<Witness>>, usize, usize, usize) {
    let (bp,cp,dp,hp,wp,kp,lp) = (b.next_power_of_two(),ci.next_power_of_two(),co.next_power_of_two(),h.next_power_of_two(),w.next_power_of_two(),k.0.next_power_of_two(),k.1.next_power_of_two());
    let mut xv = vec![int_to_f(poison); bp*cp*hp*wp];
    let mut wv = vec![int_to_f(-poison); dp*cp*kp*lp];
    for bb in 0..b { for c in 0..ci { for hh in 0..h { for ww in 0..w {
        xv[ww+wp*(hh+hp*(c+cp*bb))] = int_to_f(((bb*31+c*7+hh*5+ww*3)%23) as i128-11);
    } } } }
    for d in 0..co { for c in 0..ci { for kh in 0..k.0 { for kw in 0..k.1 {
        wv[kw+lp*(kh+kp*(c+cp*d))] = int_to_f(((d*13+c*5+kh*3+kw)%7) as i128-3);
    } } } }
    let mut builder = DagBuilder::new();
    let x = builder.committed_input(Witness::new(vec![b,ci,h,w],xv.clone(),DataType::Float,0,Role::Input));
    let weights = builder.param(Witness::new(vec![co,ci,k.0,k.1],wv.clone(),DataType::Float,0,Role::Constant));
    let y = if dilation != (1,1) { builder.conv2d_dilated(x,weights,k,stride,dilation)[0] }
        else if stride != (1,1) { builder.conv2d_strided(x,weights,k,stride)[0] }
        else { builder.conv2d(x,weights,k)[0] };
    let (dag,mut wit) = builder.compile();
    // Match only the three actual CPU blocks, not a substituted implementation.
    for &n in &dag.topo {
        let node = &dag.nodes[n];
        let ins: Vec<_> = node.inputs.iter().map(|&e| &wit[e][0]).collect();
        let outs = match &node.kind {
            BasicBlockType::FlattenKernel(f) => f.run(&ins),
            BasicBlockType::Conv2D(f) => { assert_eq!(f.batch,b); f.run(&ins) },
            BasicBlockType::ChangeShape(f) => f.run(&ins),
            _ => panic!("unexpected block in Conv2D fixture"),
        };
        for (&e,v) in node.outputs.iter().zip(outs) { wit[e]=vec![v]; }
    }
    let (ho,wo) = ((h-dilation.0*(k.0-1)-1)/stride.0+1,(w-dilation.1*(k.1-1)-1)/stride.1+1);
    assert_eq!(wit[y][0].shape,vec![b,co,ho,wo]);
    let (hop,wop) = (ho.next_power_of_two(),wo.next_power_of_two());
    let mut expected = vec![int_to_f(0); bp*dp*hop*wop];
    for bb in 0..b { for d in 0..co { for oh in 0..ho { for ow in 0..wo {
        let mut sum = 0i128;
        for c in 0..ci { for kh in 0..k.0 { for kw in 0..k.1 {
            let xi = ow*stride.1+kw*dilation.1+wp*(oh*stride.0+kh*dilation.0+hp*(c+cp*bb));
            let wi = kw+lp*(kh+kp*(c+cp*d));
            sum += f_to_int(xv[xi])*f_to_int(wv[wi]);
        } } }
        expected[ow+wop*(oh+hop*(d+dp*bb))]=int_to_f(sum);
    } } } }
    assert_eq!(wit[y][0].data.as_ref().unwrap().evaluations_ref(),expected.as_slice(),"direct integer convolution including all zero padding");
    (dag,wit,x,weights,y)
}

fn rows_hold(rows: &[Row], raw: &[Vec<i8>]) -> bool {
    rows.iter().all(|row| {
        let mut acc=K8::ZERO;
        for t in &row.terms {
            let e=eq_table(&t.r); let mut sum=K8::ZERO;
            for (i,q) in e.iter().enumerate() {
                let mut plane_sum=K8::ZERO;
                for (b,pw) in t.pw.iter().enumerate() {
                    let digit=raw[t.region.source][t.region.base+(i<<t.region.p)+b];
                    plane_sum=match digit {0=>plane_sum,1=>plane_sum.add(pw),-1=>plane_sum.sub(pw),_=>plane_sum.add(&pw.mul(&kb(digit as i128)))};
                }
                sum=sum.add(&q.mul(&plane_sum));
            }
            acc=acc.add(&sum.mul(&t.scale));
        }
        acc==row.y
    })
}

fn prove(g:&Graph,pd:&ProverData)->(Layout,Vec<K8>) {
    let p=crate::morpheus::params::Profile::toy();let layout=Layout::pack(&p,&tensor_specs(g,pd));
    let tr=MTranscript::new("batched-convolution-test",&[0;64]);
    let mut pc=Ctx::prover(tr,&pd.bank,&layout,g.cfg.slots_log);
    Walk::new(g,Some(pd)).run(&mut pc);
    { let proof=std::mem::take(&mut pc.proof); drop(pc); (layout,proof) }
}
// Negative cases reuse an honest transcript; no prover abort is verifier rejection.
fn verify(g:&Graph,pd:&ProverData,proof:&(Layout,Vec<K8>))->bool {
    let p=Profile::toy();
    let layout=&proof.0;
    let raw=fill_sources(g,pd,&layout,p.ell());
    let tr=MTranscript::new("batched-convolution-test",&[0;64]);
    let mut vc=Ctx::verifier(tr,proof.1.clone(),&layout,g.cfg.slots_log);
    Walk::new(g,None).run(&mut vc);
    vc.ok && vc.consumed() && rows_hold(&vc.rows,&raw)
}

fn graph<'a>(dag: &'a Dag,wit: &[Vec<Witness>],pp: bool) -> Graph<'a> {
    let mut g=Graph::new(dag,wit,Config {tcl:4,slots_log:12,req_planes_log:6});
    // Explicit test protocol choice avoids mutating the process environment.
    g.protocol_conv_pp=pp;
    for n in &dag.nodes { if matches!(n.kind,BasicBlockType::Conv2D(_)) && n.outputs.len()>1 { g.committed[n.outputs[1]]=pp; } }
    g
}
fn data(g: &Graph,w: &[Vec<Witness>]) -> ProverData {
    ProverData::new(g,w,(0..4096).map(|i|K8::from_base(17*i+9)).collect())
}

#[cfg_attr(test, test)]
pub fn batched_conv_cpu_semantics_and_explicit_shape() {
    for b in [1,2,3,5] { for (ci,co) in [(1,1),(3,3)] {
        for (h,w,k,s,d) in [(3,5,(2,2),(1,1),(1,1)),(5,6,(2,2),(2,3),(1,1)),(5,6,(2,2),(1,2),(2,2)),(3,5,(1,1),(1,1),(1,1))] {
            fixture(b,ci,co,h,w,k,s,d,1<<35);
        }
    } }
}

#[cfg_attr(test, test)]
pub fn batched_conv_piop_masks_padding_and_rejects_false_outputs() {
    for pp in [false,true] { for b in [1,2,3,5] {
        let (dag,wit,_,_,y)=fixture(b,3,3,3,5,(2,2),(1,1),(1,1),1<<35);
        let mut g=graph(&dag,&wit,pp); let mut pd=data(&g,&wit);
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof),"honest B={b} pp={pp}");
        // Change an active output in the last logical batch, not just b=0.
        let mut bad=pd.v(y).as_ref().clone();
        let index=(b-1)*4*2*4; bad[index]=from_i128(1234567);
        pd.vals[y]=Some(Arc::new(bad.clone())); g.outputs=vec![(y,Arc::new(bad))];
        assert!(!verify(&g,&pd,&honest_proof),"false last-batch output B={b} pp={pp}");
    } }
}

#[cfg_attr(test, test)]
pub fn batched_conv_piop_rejects_padded_output_and_full_advice() {
    for pp in [false,true] {
        let (dag,wit,_,_,y)=fixture(3,3,3,3,5,(2,2),(1,1),(1,1),-(1<<35));
        let mut g=graph(&dag,&wit,pp); let mut pd=data(&g,&wit);
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof));
        let mut bad=pd.v(y).as_ref().clone();
        // b=3 is outside logical B=3, but inside Bp=4.
        bad[3*4*2*4]=7;
        pd.vals[y]=Some(Arc::new(bad.clone()));g.outputs=vec![(y,Arc::new(bad))];
        assert!(!verify(&g,&pd,&honest_proof),"nonzero padded output pp={pp}");
        if pp {
            let g=graph(&dag,&wit,pp);let mut pd=data(&g,&wit);
            let node=dag.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::Conv2D(_))).unwrap();
            let yf=node.outputs[1];let mut bad=pd.v(yf).as_ref().clone();
            let last=bad.len()-1;bad[last]=11;pd.vals[yf]=Some(Arc::new(bad));
            assert!(!verify(&g,&pd,&honest_proof),"nonzero padded full-convolution advice");
        }
    }
}

#[cfg_attr(test, test)]
pub fn batched_conv_piop_stride_dilation_and_shared_kernel() {
    for pp in [false,true] {
        let (dag,wit,_,_,_)=fixture(3,3,3,5,6,(2,2),(2,3),(2,2),1<<35);
        let g=graph(&dag,&wit,pp);let pd=data(&g,&wit);
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof),"non-power-of-two stride, dilation, B=C=D=3 pp={pp}");
    }
    let bad=std::panic::catch_unwind(|| {
        let mut b=DagBuilder::new();
        let x=b.input(vec![3,3,4,4],DataType::Uint);
        let w=b.input(vec![3,3,3,2,2],DataType::Uint);
        b.conv2d(x,w,(2,2));
    });
    assert!(bad.is_err(),"per-batch kernels are not shared kernels");
}

#[cfg_attr(test, test)]
pub fn batched_conv_actual_field_tables_match_forward() {
    use cv::{conv_patch_table,conv_pp_powers,masked_axis_fold,trip_conv,Rel};
    let lb=|x:usize|log2_ceil(x.max(1));
    let point=|n:usize,seed:u64| (0..n).map(|i|K8([seed+i as u64,2,3,5,7,11,13,17])).collect::<Vec<_>>();
    for b in [1,2,3,5] { for (h,w,k,stride,dilation) in [
        (4,5,(2,3),(1,1),(1,1)),(5,8,(2,2),(2,3),(2,2))] {
        let (dag,wit,x,weights,y)=fixture(b,3,3,h,w,k,stride,dilation,1<<35);
        let node=dag.nodes.iter().find(|n|matches!(n.kind,BasicBlockType::Conv2D(_))).unwrap();
        let f=if let BasicBlockType::Conv2D(f)=&node.kind { f } else { unreachable!() };
        let (lw,lh,lci,lco,lbatch,lk,lkh,lkw)=(lb(w),lb(h),lb(3),lb(3),lb(b),lb(f.s_kernel),lb(k.0),lb(k.1));
        let (lwo,lho)=(lb(f.w_out),lb(f.h_out));
        let (si,sk)=(lw+lh,lkh+lkw);
        let r=point(lwo+lho+lco+lbatch,23);
        let (rd,rb)=(&r[lwo+lho..lwo+lho+lco],&r[lwo+lho+lco..]);
        let rels=vec![
            Rel {ob:lwo,kb:lkw,ib:lw,t:trip_conv(f.w_out,k.1,w,stride.1,dilation.1)},
            Rel {ob:lho,kb:lkh,ib:lh,t:trip_conv(f.h_out,k.0,h,stride.0,dilation.0)},
        ];
        let ro=vec![r[..lwo].to_vec(),r[lwo..lwo+lho].to_vec()];
        let (xv,wv,yv)=(witness_vals(&wit[x][0]),witness_vals(&wit[weights][0]),witness_vals(&wit[y][0]));
        let at=conv_patch_table(&xv,&rels,&ro,lci,b,rb);
        let wt=masked_axis_fold(&wv,sk+lci+lco,sk+lci,rd,3);
        let sum=(0..3usize).flat_map(|c|(0..1usize<<sk).map(move |k|k+(c<<sk)))
            .fold(K8::ZERO,|a,i|a.add(&at[i].mul(&wt[i])));
        assert_eq!(mle(&yv,&r),sum,"actual normal table identity B={b}");
        // The actual sparse relation tables must also link the patch MLE to
        // the original input with the SAME masked batch equality.
        let xk=point(sk,51); let xc=point(lci,71);
        let (tw,th)=(rels[0].table(&ro[0],&xk[..lkw]),rels[1].table(&ro[1],&xk[lkw..]));
        let (ec,eb)=(eq_table(&xc),eq_table(rb));
        let mut linked=K8::ZERO;
        for bb in 0..b { for cc in 0..1usize<<lci { for hh in 0..1usize<<lh { for ww in 0..1usize<<lw {
            let i=ww+(hh<<lw)+(cc<<si)+(bb<<(si+lci));
            linked=linked.add(&tw[ww].mul(&th[hh]).mul(&ec[cc]).mul(&eb[bb]).scale(xv[i]));
        } } } }
        let mut patch_point=xk;patch_point.extend_from_slice(&xc);
        assert_eq!(tensor::mle_k(&at,&patch_point),linked,"actual normal input-map identity");

        let (wf,yf)=(witness_vals(&wit[node.inputs[1]][0]),witness_vals(&wit[node.outputs[1]][0]));
        let lf=lb(f.s_in+f.s_kernel-1);
        for alpha in [K8::ZERO,K8::ONE,K8([37,2,3,5,7,11,13,17])] {
            let (am,ap,aj)=conv_pp_powers(f,alpha);
            let fixed:Vec<_>=(0..lco).map(|i|(lf+i,rd[i])).chain((0..lbatch).map(|i|(lf+lco+i,rb[i]))).collect();
            let ft=partial_eval(&yf,lf+lco+lbatch,&fixed);
            let lhs=am.iter().zip(ft).fold(K8::ZERO,|a,(u,v)|a.add(&u.mul(&v)));
            let xb=masked_axis_fold(&xv,si+lci+lbatch,si+lci,rb,b);
            let wd=masked_axis_fold(&wf,lk+lci+lco,lk+lci,rd,3);
            let rhs=(0..3usize).fold(K8::ZERO,|a,c| {
                let px=(0..1usize<<si).fold(K8::ZERO,|a,p|a.add(&ap[p].mul(&xb[p+(c<<si)])));
                let pw=(0..1usize<<lk).fold(K8::ZERO,|a,j|a.add(&aj[j].mul(&wd[j+(c<<lk)])));
                a.add(&px.mul(&pw))
            });
            assert_eq!(lhs,rhs,"actual PP powers and masked-axis table identity B={b} alpha={alpha:?}");
        }
        // Exact crop relation at a non-Boolean K point (including non-power
        // of two stride); all omitted b/d/output slots must contribute zero.
        let er=eq_table(&r);let mut crop=K8::ZERO;
        for bb in 0..b { for d in 0..3 { for oh in 0..f.h_out { for ow in 0..f.w_out {
            let yi=ow+(oh<<lwo)+(d<<(lwo+lho))+(bb<<(lwo+lho+lco));
            let m=f.s_in-1-oh*stride.0*f.stride_w-ow*stride.1;
            crop=crop.add(&er[yi].scale(yf[m+(d<<lf)+(bb<<(lf+lco))]));
        } } } }
        assert_eq!(mle(&yv,&r),crop,"exact PP crop map identity");
    } }
}

fn kernel_source_fixture(mode: usize) -> (Dag,Vec<Vec<Witness>>,usize) {
    use crate::basicblock::conv::{Conv2D,FlattenKernel};
    let mut b=DagBuilder::new();
    let x=b.committed_input(Witness::new(vec![1,1,8],(1..=8).map(int_to_f).collect(),DataType::Uint,0,Role::Input));
    let wf=if mode==0 {
        b.param(Witness::new(vec![1,1,3],[2,0,3,0].into_iter().map(int_to_f).collect(),DataType::Uint,0,Role::Constant))
    } else {
        let w=b.param(Witness::new(vec![1,1,1,2],[2,3].into_iter().map(int_to_f).collect(),DataType::Uint,0,Role::Constant));
        let dw=if mode==1 {3}else{2};
        let wf=b.add_gkr_node(vec![w],BasicBlockType::FlattenKernel(FlattenKernel {s_w:8,kh:1,kw:2,c_out:1,c_in:1,dilation_h:1,dilation_w:dw}))[0];
        b.init_values.push(Some(Witness::new_wo_data(vec![1,1,dw+1],DataType::Uint,0,Role::Output)));
        if mode==2 {let truncated=b.change_shape(wf,vec![1,1,2]);b.change_shape(truncated,vec![1,1,3])} else {wf}
    };
    let cv=Conv2D::new_dilated(1,1,1,2,1,8,1,1,1,2);
    let outs=b.add_gkr_node(vec![x,wf],BasicBlockType::Conv2D(cv));let y=outs[0];
    b.init_values.push(Some(Witness::new_wo_data(vec![1,1,6],DataType::Uint,0,Role::Output)));
    b.init_values.push(Some(Witness::new_wo_data(vec![1,10],DataType::Uint,0,Role::Auxiliary)));
    let (dag,mut wit)=b.compile();
    for &id in &dag.topo {
        let n=&dag.nodes[id];let ins:Vec<_>=n.inputs.iter().map(|&e|&wit[e][0]).collect();
        let out=match &n.kind {
            BasicBlockType::FlattenKernel(f)=>f.run(&ins),BasicBlockType::Conv2D(f)=>f.run(&ins),BasicBlockType::ChangeShape(f)=>f.run(&ins),_=>unreachable!(),
        };
        for (&e,w) in n.outputs.iter().zip(out) {wit[e]=vec![w];}
    }
    let actual=witness_vals(&wit[y][0]);
    assert_eq!(&actual[..6],if mode==0 {&[11,16,21,26,31,36]} else {&[2,4,6,8,10,12]});
    (dag,wit,y)
}

#[cfg_attr(test, test)]
pub fn conv_kernel_source_cpu_and_collision_guards() {
    for mode in 0..3 {kernel_source_fixture(mode);}
    let collision=crate::basicblock::conv::FlattenKernel {s_w:1,kh:2,kw:2,c_out:1,c_in:1,dilation_h:1,dilation_w:1};
    assert!(std::panic::catch_unwind(||cv::flatten_kernel_rel(&collision,2)).is_err(),"ambiguous scatter cannot be summed");
    let overflow=crate::basicblock::conv::FlattenKernel {s_w:2,kh:2,kw:1,c_out:1,c_in:1,dilation_h:(usize::MAX/2)+2,dilation_w:1};
    assert!(std::panic::catch_unwind(||cv::flatten_kernel_rel(&overflow,2)).is_err(),"tap indices cannot wrap machine words");
}

#[cfg_attr(test, test)]
pub fn conv_kernel_source_piop_preserves_nonidentity_producers() {
    for mode in 0..3 {for pp in [false,true] {
        let (dag,wit,y)=kernel_source_fixture(mode);
        let mut g=graph(&dag,&wit,pp);let mut pd=data(&g,&wit);
        let honest_proof=prove(&g,&pd);assert!(verify(&g,&pd,&honest_proof),"actual flat kernel source mode={mode} pp={pp}");
        if mode>0 {
            let mut bad=pd.v(y).as_ref().clone();bad[..6].copy_from_slice(&[11,16,21,26,31,36]);
            pd.vals[y]=Some(Arc::new(bad.clone()));g.outputs=vec![(y,Arc::new(bad))];
            assert!(!verify(&g,&pd,&honest_proof),"raw-kernel bypass counterfeit mode={mode} pp={pp}");
        }
    }}
}

#[cfg_attr(test,test)]
pub fn cpu_replay_rejects_tampered_transcript_and_zero_skip_is_exact() {
    for i in 0..100u64 {
        let q=K8([i+1,2,3,5,7,11,13,17]);let w=K8([19,23,i+2,29,31,37,41,43]);
        for d in [-1i8,0,1] {
            let dense=q.mul(&w).mul(&kb(d as i128));
            let sparse=if d==0 {K8::ZERO}else{q.mul(&w).mul(&kb(d as i128))};
            assert_eq!(dense,sparse);
        }
    }
    // Independent dense-vs-factored dot products with full-extension points,
    // multiple bitplanes/terms and mixed nonternary signed test digits.
    for seed in 0..64u64 {
        let mut dense=K8::ZERO;let mut factored=K8::ZERO;
        for term in 0..3u64 {
            let scale=K8([term+1,seed+2,3,5,7,11,13,17]);
            let mut dsum=K8::ZERO;let mut fsum=K8::ZERO;
            for i in 0..5u64 {
                let q=K8([seed+i+1,2,3,5,7,11,13,17]);let mut plane=K8::ZERO;
                for b in 0..17u64 {
                    let pw=K8([b+1,term+2,seed+3,5,7,11,13,17]);let digit=((seed+term+i+b)%5) as i8-2;
                    dsum=dsum.add(&q.mul(&pw).mul(&kb(digit as i128)));
                    plane=match digit {0=>plane,1=>plane.add(&pw),-1=>plane.sub(&pw),_=>plane.add(&pw.mul(&kb(digit as i128)))};
                }
                fsum=fsum.add(&q.mul(&plane));
            }
            dense=dense.add(&dsum.mul(&scale));factored=factored.add(&fsum.mul(&scale));
        }
        assert_eq!(dense,factored,"exact row factoring seed={seed}");
    }
    let(d,w,_,_,_)=fixture(1,1,1,2,2,(1,1),(1,1),(1,1),0);
    let g=graph(&d,&w,false);let pd=data(&g,&w);let proof=prove(&g,&pd);
    assert!(verify(&g,&pd,&proof));assert!(!proof.1.is_empty());
    for index in [0,proof.1.len()-1] {let mut bad=(proof.0.clone(),proof.1.clone());bad.1[index]=bad.1[index].add(&K8::ONE);assert!(!verify(&g,&pd,&bad));}
    assert!(!verify(&g,&pd,&(proof.0.clone(),proof.1[..proof.1.len()-1].to_vec())));
    let mut bad=(proof.0.clone(),proof.1.clone());bad.1.push(K8::ZERO);assert!(!verify(&g,&pd,&bad));
}
