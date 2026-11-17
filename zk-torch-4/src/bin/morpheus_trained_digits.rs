//! Trained digits 64-32-10, guarded integer semantics. Functional profiles only.
use zk_torch_4::{dag::{DagBuilder,Dag,Witness,DataType,Role},util::arith::{int_to_f,f_to_int}};
use zk_torch_4::morpheus::{request::{self,Opts},params::Profile,coins::{Coins,CoinProfile}};
use serde::Deserialize;
#[derive(Deserialize)] struct Model {weights:Vec<Vec<Vec<i128>>>,biases:Vec<Vec<i128>>}
#[derive(Deserialize)] struct Input {dataset_index:usize,input_q10:Vec<i128>}
#[derive(Deserialize)] struct Output {dataset_index:usize,logits:Vec<i128>,prediction:usize}
fn read<T:serde::de::DeserializeOwned>(name:&str)->T {serde_yaml::from_str(&std::fs::read_to_string(format!("/tmp/morpheus-trained-digits/{name}.json")).unwrap()).unwrap()}
fn w(shape:Vec<usize>,values:Vec<i128>,sf:usize,role:Role)->Witness {Witness::new(shape,values.into_iter().map(int_to_f).collect(),DataType::Float,sf,role)}
fn interval(b:&mut DagBuilder,e:usize,lo:i128,hi:i128) {
 let wi=b.init_values[e].as_ref().unwrap();let shape=wi.shape.clone();let sf=wi.sf;let n=shape.iter().map(|x|x.next_power_of_two()).product();
 let l=b.public_literal(w(shape.clone(),vec![lo;n],sf,Role::Input));let h=b.public_literal(w(shape,vec![hi;n],sf,Role::Input));
 let a=b.sub(e,l)[0];let z=b.sub(h,e)[0];let bits=(128-((hi-lo) as u128).leading_zeros() as usize).max(1);
 b.add_nonneg_node_bits(a,bits);b.add_nonneg_node_bits(z,bits);
}
fn build(m:&Model,x:&[i128],class:usize)->(Dag,Vec<Vec<Witness>>,usize,usize,usize,usize) {
 let mut b=DagBuilder::new();let input=b.committed_input(w(vec![1,64],x.to_vec(),10,Role::Input));interval(&mut b,input,0,1024);
 let mut params=Vec::new();for layer in 0..2 {let ni=if layer==0 {64}else{32};let no:usize=if layer==0 {32}else{10};let mut values=vec![0;ni*no.next_power_of_two()];for j in 0..no {for i in 0..ni {values[i+j*ni]=m.weights[layer][i][j];}}
 let weight=b.param(w(vec![ni,no],values,10,Role::Constant));interval(&mut b,weight,-16383,16383);
 let mut bias=m.biases[layer].clone();bias.resize(no.next_power_of_two(),0);let bias=b.param(w(vec![1,no],bias,10,Role::Constant));interval(&mut b,bias,-16383,16383);params.push((weight,bias));}
 let dot1=b.einsum("si,ij->sj".into(),vec![input,params[0].0],false)[0];let q1=b.scale(dot1,20,10)[0];interval(&mut b,q1,-1048512,1048512);
 let z1=b.add(q1,params[0].1)[0];let h=b.relu(z1);
 let dot2=b.einsum("si,ij->sj".into(),vec![h,params[1].0],false)[0];let q2=b.scale(dot2,20,10)[0];interval(&mut b,q2,-545192962,545192962);
 let logits=b.add(q2,params[1].1)[0];let selector=b.argmax_check(logits,1,10,&[class],Role::Input);let out=b.change_shape(selector,vec![1,10]);
 let (d,wi)=b.compile();(d,wi,input,logits,selector,out)
}
// Execute the actual compiled DAG's original BasicBlock::run kernels in topo order.
// Dag::run's optional GPU run_hook references are intentionally not used.
fn run(d:&Dag,wi:&mut [Vec<Witness>],feed:&[(usize,Witness)]) {
 use zk_torch_4::basicblock::BasicBlock;
 for (e,w) in feed {wi[*e]=vec![w.clone()];}
 for &n in &d.topo {let nd=&d.nodes[n];let refs:Vec<_>=nd.inputs.iter().map(|&e|&wi[e][0]).collect();let result=nd.kind.run(&refs);assert_eq!(result.len(),nd.outputs.len());for (&e,w) in nd.outputs.iter().zip(result){wi[e]=vec![w];}}
}
include!("digits_budget.inc");
include!("digits_regressions.inc");
mod proof { include!("digits_wire.inc"); }
mod census { include!("digits_census.inc"); }
// Measurement-only observations; never absorbed into the proof transcript.
fn measure(stage:&str,start:std::time::Instant){if std::env::var_os("DIGITS_MEASURE").is_some(){println!("MEASURE stage={} seconds={:.9}",stage,start.elapsed().as_secs_f64());}}
fn compact_mode()->bool{std::env::var_os("DIGITS_COMPACT").is_some() || selected_mode()}
fn selected_mode()->bool{std::env::var_os("DIGITS_SELECTED_COMPACT").is_some()}
fn main(){assert!(!cfg!(feature="cuda"));std::env::set_var("ZK4_SF_LOG","10");std::env::set_var("ZK4_TABLE_SIZE_LOG","21");std::env::set_var("ARGMAX_LOGIT_BITS","30");std::env::remove_var("MORPHEUS_AT_RUN");
 if let Ok(dir)=std::env::var("DIGITS_VERIFY"){proof::verify(&dir);return;}
 edge_case_regressions();
 let clock_load=std::time::Instant::now();
 let m:Model=read("model");let xs:Vec<Input>=read("test_inputs");let ys:Vec<Output>=read("integer_test_outputs");measure("fixture_read",clock_load);
 let clock_build=std::time::Instant::now();
 let(d,mut wi,input,logits,selector,out)=build(&m,&xs[0].input_q10,ys[0].prediction);measure("dag_and_initial_witness_build",clock_build);
 println!("DAG nodes={} edges={} input={} logits={} selector={} class_out={}",d.nodes.len(),d.num_edges(),input,logits,selector,out);
 let clock_validation=std::time::Instant::now();
 for (i,(x,y)) in xs.iter().zip(&ys).enumerate(){assert_eq!(x.dataset_index,y.dataset_index);let sel=DagBuilder::build_one_hot_selector_witness(1,10,&[y.prediction],Role::Input);run(&d,&mut wi,&[(input,w(vec![1,64],x.input_q10.clone(),10,Role::Input)),(selector,sel)]);
 let actual:Vec<i128>=wi[logits][0].data.as_ref().unwrap().evaluations().iter().take(10).map(|&x|f_to_int(x)).collect();assert_eq!(actual,y.logits,"sample {i} dataset {}",x.dataset_index);let predicted=(0..10).max_by_key(|&j|(actual[j],-(j as i128))).unwrap();assert_eq!(predicted,y.prediction);if i%50==0 {println!("FORWARD checked={}",i+1);}}
 measure("all_450_validation",clock_validation);
 println!("PASS ALL_450_RUST_DAG_LOGITS_AND_PREDICTIONS_EXACT");
 let clock_witness=std::time::Instant::now();
 run(&d,&mut wi,&[(input,w(vec![1,64],xs[0].input_q10.clone(),10,Role::Input)),(selector,DagBuilder::build_one_hot_selector_witness(1,10,&[ys[0].prediction],Role::Input))]);
 measure("single_case_witness_existing_dag",clock_witness);
 let clock_options=std::time::Instant::now();
 let mut o=Opts::from_env();o.profile=if selected_mode(){Profile::ternary()}else{Profile::mini()};o.b0=9;o.cfg.slots_log=mask_budget(&d,&wi);o.cfg.tcl=4;o.cfg.req_planes_log=6;o.defer=false;o.auth=true;o.verify=true;o.coins=if selected_mode(){assert!(std::env::var_os("MORPHEUS_COIN_SEED").is_none(),"selected private tape must not be fixed");Coins::from_entropy(CoinProfile::Ternary)}else{Coins::new([29;64],CoinProfile::Ternary)};
 measure("options_entropy_and_mask_budget",clock_options);
 let clock_prepare=std::time::Instant::now();
 let mut pp=request::prepare(&d,&wi,&[selector],&mut o);measure("graph_cache_layout_prepare",clock_prepare);assert!(!pp.g.public.contains_key(&input));assert_eq!(pp.g.outputs.len(),1);assert_eq!(pp.g.outputs[0].0,out);
 println!("CENSUS profile={} security_claim=false fixed_test_seed={} sources={} model={:?} input={:?} request={:?} slots_log={} specs={:?}",o.profile.name,!selected_mode(),pp.lay.n_sources,pp.model,pp.input,pp.req,o.cfg.slots_log,pp.specs);
 if compact_mode(){let support=request::request_support(&pp,&o).expect("public support admission");let n=support.len();let response=o.profile.tau*(n+o.profile.link_cols()+o.profile.aux_cols())*4096/8*37;println!("COMPACT_PREFLIGHT active_W_columns={} original_columns={:?} response_bytes={} raw_source_bytes={} auth_mask_bytes={}",n,support.columns(),response,pp.lay.n_sources*(1usize<<o.profile.ell()),o.profile.tau*(n+o.profile.link_cols()+o.profile.aux_cols())*4096*8);}
 if std::env::var_os("DIGITS_PREFLIGHT_ONLY").is_some(){return;}
 if selected_mode(){assert!(std::env::var_os("DIGITS_SELECTED_RUN_ADMITTED").is_some(),"selected execution requires independent native audit and safe resource admission");}
 if std::env::var_os("DIGITS_CENSUS").is_some(){census::run(&mut pp,&o);}

 if let Ok(dir)=std::env::var("DIGITS_PROVE"){proof::prove(&mut pp,&o,ys[0].prediction,&dir);}

}
