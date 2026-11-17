use super::*;
use serde::{Serialize,Deserialize};
use std::{collections::{BTreeMap,HashMap},sync::Arc,ops::Range};
use zk_torch_4::morpheus::{piop::{Graph,ProverData,layout::{Layout,TKey}},request::{Prepared,Times,RegisteredModel,RegisteredInput},wire,field8::K8};
#[derive(Serialize,Deserialize)]
struct Snapshot {
 shapes:Vec<Vec<usize>>,edge_meta:Vec<(DataType,zk_torch_4::dag::PolyType,usize,Role)>,nvars:Vec<usize>,committed:Vec<bool>,model:Vec<bool>,bool_typed:Vec<bool>,lookups:BTreeMap<usize,usize>,public:Vec<(usize,Vec<u64>)>,outputs:Vec<(usize,Vec<u64>)>,layout:Layout,specs:Vec<(TKey,usize,usize,bool)>,groups:[Range<usize>;4],
 mc:Vec<Vec<Vec<u64>>>,ms:Vec<u64>,ml:Vec<u64>,ic:Vec<Vec<Vec<u64>>>,is:Vec<u64>,il:Vec<u64>,slots_log:usize,class:usize,
}
pub fn prove(pp:&mut Prepared,o:&Opts,class:usize,dir:&str) {
 std::fs::create_dir_all(dir).unwrap();
 let clock_model=std::time::Instant::now();let model=request::register_cpu(pp,o);measure("model_registration",clock_model);
 let clock_input=std::time::Instant::now();let input=request::register_inputs_cpu(pp,o);measure("input_registration",clock_input);
 let clock_registry=std::time::Instant::now();
 let mr=model.public();let ir=input.public();let g=&pp.g;
 let s=Snapshot{shapes:g.shapes.clone(),edge_meta:g.edge_meta.clone(),nvars:g.nvars.clone(),committed:g.committed.clone(),model:g.model.clone(),bool_typed:g.bool_typed.clone(),lookups:g.lookups.clone(),public:g.public.iter().map(|(&e,v)|(e,(**v).clone())).collect(),outputs:g.outputs.iter().map(|(e,v)|(*e,(**v).clone())).collect(),layout:pp.lay.clone(),specs:pp.specs.clone(),groups:[pp.model.clone(),pp.input.clone(),pp.req.clone(),pp.carry.clone()],mc:mr.commitments.to_vec(),ms:mr.statement.to_vec(),ml:mr.layout.to_vec(),ic:ir.commitments.to_vec(),is:ir.statement.to_vec(),il:ir.layout.to_vec(),slots_log:o.cfg.slots_log,class};
 let registry_bytes=bincode::serialize(&s).unwrap();measure("public_registry_prepare_encode",clock_registry);
 let clock_registry_write=std::time::Instant::now();std::fs::write(format!("{dir}/public-registry.bin"),&registry_bytes).unwrap();measure("public_registry_write",clock_registry_write);drop(registry_bytes);
 let clock_prove=std::time::Instant::now();
 let pf=if selected_mode(){
  // Local machine admission, not protocol parameter weakening. Counts are
  // public layout/statement structure and exceeding the budget stops BEFORE link.
  let allocated:usize=pp.lay.tensors.iter().flat_map(|t|&t.pieces).map(|p|1usize.checked_shl((p.region.m+p.region.p).try_into().unwrap()).unwrap()).sum();
  let gate=|p:&Profile,st:&zk_torch_4::morpheus::link::Statement,b0:usize|{
   assert_eq!(p,&Profile::ternary());assert_eq!(b0,9);assert_eq!(st.n,3);
   let plan=zk_torch_4::morpheus::link_fast::cpu_memory_plan(p,st,b0).expect("selected public link memory plan");
   println!("SELECTED_PUBLIC_LINK_PLAN {:?} allocated_digits={}",plan,allocated);
   assert!(plan.dense_query_bytes<=allocated.checked_mul(64).unwrap(),"dense query budget");
   assert!(plan.block_term_entries<=1_000_000,"block term metadata budget");
   assert!(plan.max_round_tables_bytes<=64*1024*1024,"round table budget");
   assert!(plan.declared_bit_blocks<=allocated/512,"declared bit block budget");
  };
  request::prove_request_cpu_compact_guarded(pp,o,0,&model,&input,&gate).proof
 }else if compact_mode(){request::prove_request_cpu_compact(pp,o,0,&model,&input).proof}else{request::prove_request_cpu(pp,o,0,&model,&input)};measure("prove_api_including_source_pack_commit_and_link_selfcheck",clock_prove);
 let clock_encode=std::time::Instant::now();let bytes=encode(pp,o,&pf);measure("proof_encode",clock_encode);
 let clock_write=std::time::Instant::now();std::fs::write(format!("{dir}/request-0.bin"),&bytes).unwrap();measure("proof_write",clock_write);println!("PROOF_SERIALIZED bytes={} attempt={}",bytes.len(),pf.auth.as_ref().unwrap().attempt);
}
pub fn verify(dir:&str) {
 let clock_admission=std::time::Instant::now();
 // No model.json, test inputs, expected logits, raw registrations, masks or witness data are read.
 let mut s:Snapshot=bincode::deserialize(&std::fs::read(format!("{dir}/public-registry.bin")).unwrap()).unwrap();s.layout.rebuild_index();assert!(s.class<10,"classification outside logical vocabulary");assert_eq!(s.slots_log,11,"fixed workload slot capacity");
 let mut m=Model{weights:vec![vec![vec![0;32];64],vec![vec![0;10];32]],biases:vec![vec![0;32],vec![0;10]]};m.biases[1][s.class]=1;let(d,mut wi,_,_,selector,_)=build(&m,&[0;64],s.class);run(&d,&mut wi,&[]);drop(m);
 let mut o=Opts::from_env();o.profile=if selected_mode(){Profile::ternary()}else{Profile::mini()};o.b0=9;o.cfg.slots_log=s.slots_log;o.cfg.tcl=4;o.cfg.req_planes_log=6;o.defer=false;o.auth=true;o.verify=true;o.coins=Coins::new([77;64],CoinProfile::Ternary);
 // Reconstruct every public semantic field from the typed DAG, not the supplied snapshot.
 let expected=request::prepare(&d,&wi,&[selector],&mut o);
 assert_eq!(s.slots_log,o.cfg.slots_log);assert_eq!(s.shapes,expected.g.shapes);assert_eq!(s.edge_meta,expected.g.edge_meta);assert_eq!(s.nvars,expected.g.nvars);assert_eq!(s.committed,expected.g.committed);assert_eq!(s.model,expected.g.model);assert_eq!(s.bool_typed,expected.g.bool_typed);assert_eq!(s.lookups,expected.g.lookups);
 let public:HashMap<_,_>=s.public.iter().map(|(e,v)|(*e,v.clone())).collect();assert_eq!(public.len(),s.public.len());let canonical:HashMap<_,_>=expected.g.public.iter().map(|(&e,v)|(e,(**v).clone())).collect();assert_eq!(public,canonical,"noncanonical public literals or classification selector");
 let outputs:Vec<_>=expected.g.outputs.iter().map(|(e,v)|(*e,(**v).clone())).collect();assert_eq!(s.outputs,outputs,"noncanonical classification output");
 let specs:Vec<_>=expected.specs.iter().map(|&(key,n,p,bits)|match key {TKey::Edge(e) if expected.g.model[e]=>(key,n,4,false),_=>(key,n,p,bits)}).collect();assert_eq!(s.specs,specs,"not the declared public source types");
 let mut groups=[Vec::new(),Vec::new(),Vec::new()];for &sp in &specs {let group=match sp.0 {TKey::Edge(e) if expected.g.model[e]=>0,TKey::Edge(e) if d.producers[e].is_none() && expected.g.edge_meta[e].3==Role::Input && !expected.g.public.contains_key(&e)=>1,_=>2};groups[group].push(sp);}
 let (layout,ranges)=Layout::pack_groups(&o.profile,&groups,o.b0);assert_eq!(bincode::serialize(&s.layout).unwrap(),bincode::serialize(&layout).unwrap(),"noncanonical source layout");assert_eq!(&s.groups[..3],ranges.as_slice());assert_eq!(s.groups[3],layout.n_sources..layout.n_sources);drop(expected);drop(wi);measure("public_registry_read_reconstruction_admission",clock_admission);println!("PASS canonical_public_statement_and_layout_admission");if std::env::var_os("DIGITS_REGISTRY_ONLY").is_some(){return;}
 let g=Graph{dag:&d,shapes:s.shapes,edge_meta:s.edge_meta,nvars:s.nvars,committed:s.committed,model:s.model,bool_typed:s.bool_typed,lookups:s.lookups,public:s.public.into_iter().map(|(e,v)|(e,Arc::new(v))).collect(),outputs:s.outputs.into_iter().map(|(e,v)|(e,Arc::new(v))).collect(),cfg:o.cfg.clone(),protocol_conv_pp:false};
 let pd=ProverData{vals:vec![],sel:HashMap::new(),bank:vec![],#[cfg(feature="cuda")] cache:Default::default()};let mut pp=Prepared{g,pd,lay:s.layout,model:s.groups[0].clone(),input:s.groups[1].clone(),req:s.groups[2].clone(),carry:s.groups[3].clone(),specs:s.specs,ar:None};
 let mr=RegisteredModel{commitments:&s.mc,statement:&s.ms,layout:&s.ml};let ir=RegisteredInput{commitments:&s.ic,statement:&s.is,layout:&s.il};
 let clock_read=std::time::Instant::now();let bytes=std::fs::read(format!("{dir}/request-0.bin")).unwrap();measure("proof_read",clock_read);
 let clock_decode=std::time::Instant::now();let mut pf=decode(&pp,&o,&bytes).expect("canonical request wire");measure("proof_decode",clock_decode);
 let clock_canonical=std::time::Instant::now();assert_eq!(encode(&pp,&o,&pf),bytes);assert!(pf.auth.is_some());measure("proof_canonical_roundtrip",clock_canonical);
 let accepted=|p:&wire::RequestProof|verify_request(&pp,&o,0,Some(mr),ir,p,&mut Times::new()).is_some();
 let clock_verify=std::time::Instant::now();let mut valid_times=Times::new();assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut valid_times).is_some());measure("single_valid_verify_api",clock_verify);
 if std::env::var_os("DIGITS_MEASURE").is_some(){for (stage,seconds) in valid_times{println!("MEASURE stage=valid_{} seconds={:.9}",stage.replace(' ',"_"),seconds);}}
 println!("PASS SECRET_FREE_SAVED_WIRE_VERIFY");let clock_negatives=std::time::Instant::now();
 pf.auth.as_mut().unwrap().z.w[0]^=1;assert!(!accepted(&pf));pf.auth.as_mut().unwrap().z.w[0]^=1;println!("PASS response_mutation_rejected");
 let old=pf.piop[0];pf.piop[0]=old.add(&K8::ONE);assert!(!accepted(&pf));pf.piop[0]=old;println!("PASS piop_mutation_rejected");
 pf.piop.push(K8::ZERO);assert!(!accepted(&pf));pf.piop.pop();println!("PASS extra_piop_rejected");
 let old=pf.link.a_hat[0].0[0];pf.link.a_hat[0].0[0]=old.add(&K8::ONE);assert!(!accepted(&pf));pf.link.a_hat[0].0[0]=old;println!("PASS link_mutation_rejected");
 let auth=pf.auth.take();assert!(!accepted(&pf));pf.auth=auth;println!("PASS missing_auth_rejected");
 assert!(verify_request(&pp,&o,1,Some(mr),ir,&pf,&mut Times::new()).is_none());println!("PASS request_id_rejected");
 let mut mc=s.mc.clone();mc[0][0][0]^=1;assert!(verify_request(&pp,&o,0,Some(RegisteredModel{commitments:&mc,..mr}),ir,&pf,&mut Times::new()).is_none());println!("PASS registry_model_mutation_rejected");
 let mut ic=s.ic.clone();ic[0][0][0]^=1;assert!(verify_request(&pp,&o,0,Some(mr),RegisteredInput{commitments:&ic,..ir},&pf,&mut Times::new()).is_none());println!("PASS registry_input_mutation_rejected");
 assert!(decode(&pp,&o,&bytes[..bytes.len()-1]).is_none());let mut tail=bytes.clone();tail.push(0);assert!(decode(&pp,&o,&tail).is_none());println!("PASS malformed_wire_rejected");
 let old_public=pp.g.public[&selector].clone();let old_output=pp.g.outputs[0].clone();let mut wrong=vec![0;16];wrong[(s.class+1)%10]=1;let wrong=Arc::new(wrong);pp.g.public.insert(selector,wrong.clone());pp.g.outputs[0]=(old_output.0,wrong);assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());pp.g.public.insert(selector,old_public);pp.g.outputs[0]=old_output;println!("PASS canonical_wrong_classification_rejected");
 let (e,v)=pp.g.outputs[0].clone();let mut v=(*v).clone();v[0]^=1;pp.g.outputs[0]=(e,Arc::new(v));assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());println!("PASS public_classification_mutation_rejected");
 measure("all_proof_negative_controls",clock_negatives);
 println!("PASS COMPLETE_AUTHENTICATED_TRAINED_DIGITS profile={} compact={} native_security_audit_separate=true",o.profile.name,compact_mode());
}

fn encode(pp:&Prepared,o:&Opts,pf:&wire::RequestProof)->Vec<u8>{if compact_mode(){wire::encode_request_compact(&o.profile,&request::request_support(pp,o).unwrap(),pf).unwrap()}else{wire::encode_request(&o.profile,pf)}}
fn decode(pp:&Prepared,o:&Opts,b:&[u8])->Option<wire::RequestProof>{if compact_mode(){wire::decode_request_compact(&o.profile,&request::request_support(pp,o)?,b)}else{wire::decode_request(&o.profile,b)}}
fn verify_request(pp:&Prepared,o:&Opts,j:u64,m:Option<RegisteredModel<'_>>,i:RegisteredInput<'_>,pf:&wire::RequestProof,t:&mut Times)->Option<(Vec<zk_torch_4::morpheus::link::Row>,K8,Vec<u8>)>{if compact_mode(){request::verify_request_cpu_compact(pp,o,j,m,i,pf,t)}else{request::verify_request_cpu(pp,o,j,m,i,pf,t)}}
