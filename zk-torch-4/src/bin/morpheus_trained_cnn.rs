//! Fixed, trained Q10 CNN using existing authenticated operators.
use zk_torch_4::{dag::{DagBuilder,Dag,Witness,DataType,Role},util::arith::{int_to_f,f_to_int}};
use zk_torch_4::morpheus::{request::{self,Opts},params::Profile,coins::{Coins,CoinProfile}};
use serde::Deserialize;
#[derive(Deserialize)] struct Model {conv1_weight:Vec<Vec<Vec<Vec<i128>>>>,conv1_bias:Vec<i128>,conv2_weight:Vec<Vec<Vec<Vec<i128>>>>,conv2_bias:Vec<i128>,fc_weight:Vec<Vec<i128>>,fc_bias:Vec<i128>}
impl Model {fn zero()->Self{Self{conv1_weight:vec![vec![vec![vec![0;3];3];1];8],conv1_bias:vec![0;8],conv2_weight:vec![vec![vec![vec![0;2];2];8];16],conv2_bias:vec![0;16],fc_weight:vec![vec![0;16];10],fc_bias:vec![0;10]}}}
#[derive(Deserialize)] struct Input {dataset_index:usize,input_q10:Vec<i128>}
#[derive(Deserialize)] struct Output {dataset_index:usize,logits:Vec<i128>,prediction:usize}
fn read<T:serde::de::DeserializeOwned>(name:&str)->T {let dir=std::env::var("CNN_FIXTURE").unwrap_or("/tmp/morpheus-trained-cnn".into());serde_yaml::from_str(&std::fs::read_to_string(format!("{dir}/{name}.json")).unwrap()).unwrap()}
fn w(shape:Vec<usize>,values:Vec<i128>,sf:usize,role:Role)->Witness {Witness::new(shape,values.into_iter().map(int_to_f).collect(),DataType::Float,sf,role)}
fn interval(b:&mut DagBuilder,e:usize,lo:i128,hi:i128) {
 let wi=b.init_values[e].as_ref().unwrap();let shape=wi.shape.clone();let sf=wi.sf;let n=shape.iter().map(|x|x.next_power_of_two()).product();
 let l=b.public_literal(w(shape.clone(),vec![lo;n],sf,Role::Input));let h=b.public_literal(w(shape,vec![hi;n],sf,Role::Input));
 let a=b.sub(e,l)[0];let z=b.sub(h,e)[0];let bits=(128-((hi-lo) as u128).leading_zeros() as usize).max(1);
 b.add_nonneg_node_bits(a,bits);b.add_nonneg_node_bits(z,bits);
}
fn pad_conv(a:&[Vec<Vec<Vec<i128>>>],co:usize,ci:usize,k:usize)->Vec<i128>{assert_eq!(a.len(),co);let kp=k.next_power_of_two();let mut v=vec![0;co*ci*kp*kp];for o in 0..co {assert_eq!(a[o].len(),ci);for i in 0..ci{assert_eq!(a[o][i].len(),k);for h in 0..k{assert_eq!(a[o][i][h].len(),k);for x in 0..k{v[((o*ci+i)*kp+h)*kp+x]=a[o][i][h][x];}}}}v}
fn build(m:&Model,x:&[i128],batch:usize,classes:&[usize])->(Dag,Vec<Vec<Witness>>,usize,usize,usize,usize){
 assert!(matches!(batch,1|4|16));assert_eq!(x.len(),batch*64);assert_eq!(classes.len(),batch);
 let mut b=DagBuilder::new();let input=b.committed_input(w(vec![batch,1,8,8],x.to_vec(),10,Role::Input));interval(&mut b,input,0,1024);
 let cw1=b.param(w(vec![8,1,3,3],pad_conv(&m.conv1_weight,8,1,3),10,Role::Constant));interval(&mut b,cw1,-16383,16383);
 assert_eq!(m.conv1_bias.len(),8);let cb1=b.param(w(vec![8],m.conv1_bias.clone(),10,Role::Constant));interval(&mut b,cb1,-16383,16383);
 let cw2=b.param(w(vec![16,8,2,2],pad_conv(&m.conv2_weight,16,8,2),10,Role::Constant));interval(&mut b,cw2,-16383,16383);
 assert_eq!(m.conv2_bias.len(),16);let cb2=b.param(w(vec![16],m.conv2_bias.clone(),10,Role::Constant));interval(&mut b,cb2,-16383,16383);
 assert_eq!(m.fc_weight.len(),10);let mut fw=vec![0;16*16];for o in 0..10{assert_eq!(m.fc_weight[o].len(),16);for i in 0..16{fw[i+16*o]=m.fc_weight[o][i];}}
 let fw=b.param(w(vec![16,10],fw,10,Role::Constant));interval(&mut b,fw,-16383,16383);
 assert_eq!(m.fc_bias.len(),10);let mut fb=m.fc_bias.clone();fb.resize(16,0);let fb=b.param(w(vec![1,10],fb,10,Role::Constant));interval(&mut b,fb,-16383,16383);
 let d1=b.conv2d(input,cw1,(3,3))[0];let q1=b.scale(d1,20,10)[0];interval(&mut b,q1,-147447,147447);
 let z1=b.add_channel_bias_row_major(q1,cb1,8,2);let h1=b.relu(z1);let p1=b.maxpool2d(h1,2,2);
 let d2=b.conv2d(p1,cw2,(2,2))[0];let q2=b.scale(d2,20,10)[0];interval(&mut b,q2,-83875840,83875840);
 let z2=b.add_channel_bias_row_major(q2,cb2,16,2);let h2=b.relu(z2);let p2=b.maxpool2d(h2,2,2);
 // All final spatial dimensions are one: this is an exact storage view,
 // not a padded logical flatten or data transpose.
 let flat=b.change_shape(p2,vec![16,batch]);let dot=b.einsum("ib,ij->bj".into(),vec![flat,fw],false)[0];let q=b.scale(dot,20,10)[0];interval(&mut b,q,-21475098272,21475098272);
 let logits=b.add(q,fb)[0];let selector=b.argmax_check(logits,batch,10,classes,Role::Input);let out=b.change_shape(selector,vec![batch,10]);
 let(d,wi)=b.compile();(d,wi,input,logits,selector,out)
}
fn run(d:&Dag,wi:&mut [Vec<Witness>],feed:&[(usize,Witness)]){
 use zk_torch_4::basicblock::BasicBlock;
 for(e,w)in feed{wi[*e]=vec![w.clone()];}for &n in &d.topo{let nd=&d.nodes[n];let refs:Vec<_>=nd.inputs.iter().map(|&e|&wi[e][0]).collect();let result=nd.kind.run(&refs);assert_eq!(result.len(),nd.outputs.len());for(&e,w)in nd.outputs.iter().zip(result){wi[e]=vec![w];}}
}
include!("cnn_budget.inc");
mod proof{include!("cnn_wire.inc");}
fn measure(stage:&str,start:std::time::Instant){if std::env::var_os("CNN_MEASURE").is_some(){println!("MEASURE stage={} seconds={:.9}",stage,start.elapsed().as_secs_f64());}}
fn selected_mode()->bool{std::env::var_os("CNN_SELECTED_COMPACT").is_some()}
fn compact_mode()->bool{true}
fn validate_case(batch:usize,pp:bool){assert!(matches!((batch,pp),(1,false)|(4,false)|(16,false)|(4,true)),"not one of four predeclared cases");}
fn set_conv(pp:bool){if pp{std::env::set_var("MORPHEUS_CONV","pp");}else{std::env::remove_var("MORPHEUS_CONV");}std::env::remove_var("ZK4_CONV_GRANDPRODUCT");}
fn main(){assert!(!cfg!(feature="cuda"));std::env::set_var("ZK4_SF_LOG","10");std::env::set_var("ZK4_TABLE_SIZE_LOG","27");std::env::set_var("ARGMAX_LOGIT_BITS","35");std::env::remove_var("MORPHEUS_AT_RUN");
 if let Ok(dir)=std::env::var("CNN_VERIFY"){proof::verify(&dir);return;}
 let batch=std::env::var("CNN_BATCH").unwrap_or("1".into()).parse::<usize>().unwrap();let pp=std::env::var("CNN_CONV_MODE").unwrap_or("default".into());assert!(pp=="default"||pp=="pp");let pp=pp=="pp";validate_case(batch,pp);set_conv(pp);
 let clock=std::time::Instant::now();let m:Model=read("model");let xs:Vec<Input>=read("test_inputs");let ys:Vec<Output>=read("integer_test_outputs");assert_eq!(xs.len(),450);assert_eq!(ys.len(),450);measure("fixture_read",clock);
 let classes:Vec<_>=ys[..batch].iter().map(|y|y.prediction).collect();let first:Vec<_>=xs[..batch].iter().flat_map(|x|x.input_q10.clone()).collect();let clock=std::time::Instant::now();let(d,mut wi,input,logits,selector,out)=build(&m,&first,batch,&classes);measure("dag_and_initial_witness_build",clock);
 println!("DAG batch={batch} pp={pp} nodes={} edges={} input={input} logits={logits} selector={selector} output={out}",d.nodes.len(),d.num_edges());
 let clock=std::time::Instant::now();for start in (0..xs.len()).step_by(batch){let ids:Vec<_>=(0..batch).map(|j|(start+j)%xs.len()).collect();let cls:Vec<_>=ids.iter().map(|&i|ys[i].prediction).collect();let feed:Vec<_>=ids.iter().flat_map(|&i|xs[i].input_q10.clone()).collect();run(&d,&mut wi,&[(input,w(vec![batch,1,8,8],feed,10,Role::Input)),(selector,DagBuilder::build_one_hot_selector_witness(batch,10,&cls,Role::Input))]);let values=wi[logits][0].data.as_ref().unwrap().evaluations();for(bi,&i)in ids.iter().enumerate(){assert_eq!(xs[i].dataset_index,ys[i].dataset_index);let actual:Vec<_>=(0..10).map(|c|f_to_int(values[bi+batch*c])).collect();assert_eq!(actual,ys[i].logits,"sample{i} dataset{}",xs[i].dataset_index);}if start%50<batch{println!("FORWARD checked={}",(start+batch).min(xs.len()));}}
 measure("all_450_validation",clock);println!("PASS ALL_450_RUST_CNN_LOGITS_AND_PREDICTIONS_EXACT");
 let clock=std::time::Instant::now();run(&d,&mut wi,&[(input,w(vec![batch,1,8,8],first,10,Role::Input)),(selector,DagBuilder::build_one_hot_selector_witness(batch,10,&classes,Role::Input))]);measure("selected_batch_witness_existing_dag",clock);
 let mut o=Opts::from_env();o.profile=if selected_mode(){Profile::ternary()}else{Profile::mini()};o.b0=9;o.cfg.slots_log=mask_budget(&d,&wi);o.cfg.tcl=4;o.cfg.req_planes_log=6;o.defer=false;o.auth=true;o.verify=true;o.coins=if selected_mode(){assert!(std::env::var_os("MORPHEUS_COIN_SEED").is_none());Coins::from_entropy(CoinProfile::Ternary)}else{Coins::new([29;64],CoinProfile::Ternary)};
 let clock=std::time::Instant::now();let mut prepared=request::prepare(&d,&wi,&[selector],&mut o);proof::normalize_layout(&mut prepared,&d,&o);measure("graph_cache_layout_prepare",clock);assert!(!prepared.g.public.contains_key(&input));assert_eq!(prepared.g.outputs.len(),1);assert_eq!(prepared.g.outputs[0].0,out);assert_eq!(prepared.g.protocol_conv_pp,pp);
 let support=request::request_support(&prepared,&o).expect("public support admission");let n=support.len();let response=o.profile.tau*(n+o.profile.link_cols()+o.profile.aux_cols())*4096/8*37;
 println!("PUBLIC_PREFLIGHT batch={batch} pp={pp} nodes={} edges={} sources={} model={:?} input={:?} request={:?} slots_log={} active_W_columns={} response_bytes={} raw_source_bytes={} auth_mask_bytes={}",d.nodes.len(),d.num_edges(),prepared.lay.n_sources,prepared.model,prepared.input,prepared.req,o.cfg.slots_log,n,response,prepared.lay.n_sources*(1usize<<o.profile.ell()),o.profile.tau*(n+o.profile.link_cols()+o.profile.aux_cols())*4096*8);
 if std::env::var_os("CNN_PREFLIGHT_ONLY").is_some(){return;}
 if selected_mode(){assert!(std::env::var_os("CNN_SELECTED_RUN_ADMITTED").is_some(),"selected run requires independent audit and resource admission");}
 if let Ok(dir)=std::env::var("CNN_PROVE"){proof::prove(&mut prepared,&o,batch,&classes,pp,&dir);}
}
