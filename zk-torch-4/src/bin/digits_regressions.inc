fn edge_case_regressions(){
 use zk_torch_4::basicblock::{BasicBlock,BasicBlockType as B,scale::ScaleDown};
 let sd=ScaleDown{output_sf:10};let inp=w(vec![8],vec![-1537,-1536,-1535,-513,-512,-511,511,512],20,Role::Input);let got=sd.run(&[&inp]);let got:Vec<i128>=got[0].data.as_ref().unwrap().evaluations().iter().map(|&x|f_to_int(x)).collect();assert_eq!(got,vec![-2,-1,-1,-1,0,0,0,1]);
 for case in 0..3 {
 let mut m=Model{weights:vec![vec![vec![0;32];64],vec![vec![0;10];32]],biases:vec![vec![-1024;32],vec![-2048;10]]};let class=if case==0{0}else if case==1{3}else{9};if case==1 {m.biases[1][3]=-1024;m.biases[1][7]=-1024;}if case==2 {m.biases[1][9]=-1024;}
 let(d,mut wi,_,lg,_,_)=build(&m,&[0;64],class);run(&d,&mut wi,&[]);let vals=wi[lg][0].data.as_ref().unwrap().evaluations();assert_eq!((0..10).max_by_key(|&i|(f_to_int(vals[i]),-(i as i128))).unwrap(),class);
 for n in &d.nodes {if let B::NonNegative(nn)=&n.kind {for x in wi[n.inputs[0]][0].data.as_ref().unwrap().evaluations(){let v=f_to_int(x);assert!(v>=0 && (v as u128)<(1u128<<nn.table_size_log),"case {case}, node {} value {v}",n.id);}}}
 println!("PASS actual_DAG_all_negative_tied_or_last_class case={case} class={class} padded_vocab=16 logical_vocab=10");
 }
 println!("PASS actual_ScaleDown_half_ties_toward_positive_infinity");
}
