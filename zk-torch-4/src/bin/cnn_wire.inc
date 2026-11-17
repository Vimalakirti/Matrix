use super::*;
use serde::{Serialize,Deserialize};
use std::{collections::{BTreeMap,HashMap},sync::Arc,ops::Range};
use zk_torch_4::morpheus::{piop::{Graph,ProverData,layout::{Layout,TKey}},request::{Prepared,Times,RegisteredModel,RegisteredInput},wire,field8::K8};

/// Fixed public model type: every registered coefficient has 16 magnitude
/// planes, independently of observed weights or biases. The graph's explicit
/// interval guards bound the decoded model value to [-16383,16383].
pub fn normalize_layout(pp:&mut Prepared,d:&Dag,o:&Opts) {
 for sp in &mut pp.specs {if let TKey::Edge(e)=sp.0 {if pp.g.model[e] {assert!(sp.2<=4,"model magnitude exceeds fixed 16-plane public type");sp.2=4;sp.3=false;}}}
 let mut groups=[Vec::new(),Vec::new(),Vec::new()];
 for &sp in &pp.specs {let group=match sp.0 {TKey::Edge(e) if pp.g.model[e]=>0,TKey::Edge(e) if d.producers[e].is_none() && pp.g.edge_meta[e].3==Role::Input && !pp.g.public.contains_key(&e)=>1,_=>2};groups[group].push(sp);}
 let (layout,ranges)=Layout::pack_groups(&o.profile,&groups,o.b0);assert_eq!(ranges.len(),3);
 pp.model=ranges[0].clone();pp.input=ranges[1].clone();pp.req=ranges[2].clone();pp.carry=layout.n_sources..layout.n_sources;pp.lay=layout;
}
// `prepare` constructs lookup witnesses even for its metadata-only use. A zero
// CNN need not predict every claimed class in a batch. Neutralize produced
// private dense values, retaining every shape/type/scale/role and preserving all
// public leaf literals, the public selector, and its terminal output. This only
// enables canonical descriptor reconstruction; no dummy proof is created.
fn erase_dummy_private_intermediates(d:&Dag,wi:&mut [Vec<Witness>],out:usize) {
 // Snapshot every leaf (including all public literals and class selector) and
 // the public terminal output. No operator/DAG mutation is possible through
 // this shared `&Dag`; these are dummy descriptor values, never prover data.
 let preserved:Vec<_>=(0..wi.len()).filter(|&e|d.producers[e].is_none() || e==out)
  .map(|e|(e,wi[e][0].data.as_ref().expect("dummy leaf/output data").evaluations())).collect();
 let metadata:Vec<_>=wi.iter().map(|w|{let w=&w[0];(w.shape.clone(),w.data_type,w.poly_type,w.sf,w.role)}).collect();
 for e in 0..wi.len() {
  if d.producers[e].is_none() || e==out {continue;}
  let old=&wi[e][0];
  if old.poly_type!=zk_torch_4::dag::PolyType::Dense {continue;}
  let size:usize=old.shape.iter().map(|n|n.next_power_of_two()).product();
  wi[e]=vec![Witness::new(old.shape.clone(),vec![int_to_f(0);size],old.data_type,old.sf,old.role)];
 }
 for (e,before) in preserved {assert_eq!(wi[e][0].data.as_ref().unwrap().evaluations(),before,"metadata reconstruction modified public/leaf edge {e}");}
 for (e,before) in metadata.iter().enumerate(){let w=&wi[e][0];assert_eq!(&(w.shape.clone(),w.data_type,w.poly_type,w.sf,w.role),before,"metadata reconstruction changed tensor type {e}");}
 // DagBuilder::public_literal creates leaf edges; every leaf was checked above.
}
fn admit_case(batch:usize,classes:&[usize],conv_pp:bool) {
 assert!(matches!(batch,1|4|16),"not a predeclared CNN batch");
 assert!(!conv_pp || batch==4,"pp baseline is admitted only at batch 4");
 assert_eq!(classes.len(),batch);assert!(classes.iter().all(|&c|c<10),"classification outside logical vocabulary");
}
#[derive(Serialize,Deserialize)]
struct Snapshot {
 shapes:Vec<Vec<usize>>,edge_meta:Vec<(DataType,zk_torch_4::dag::PolyType,usize,Role)>,nvars:Vec<usize>,committed:Vec<bool>,model:Vec<bool>,bool_typed:Vec<bool>,lookups:BTreeMap<usize,usize>,public:Vec<(usize,Vec<u64>)>,outputs:Vec<(usize,Vec<u64>)>,layout:Layout,specs:Vec<(TKey,usize,usize,bool)>,groups:[Range<usize>;4],
 mc:Vec<Vec<Vec<u64>>>,ms:Vec<u64>,ml:Vec<u64>,ic:Vec<Vec<Vec<u64>>>,is:Vec<u64>,il:Vec<u64>,slots_log:usize,batch:usize,classes:Vec<usize>,conv_pp:bool,
}
pub fn prove(pp:&mut Prepared,o:&Opts,batch:usize,classes:&[usize],conv_pp:bool,dir:&str) {
 admit_case(batch,classes,conv_pp);assert_eq!(pp.g.protocol_conv_pp,conv_pp);
 std::fs::create_dir_all(dir).unwrap();
 let clock_model=std::time::Instant::now();let model=request::register_cpu(pp,o);measure("model_registration",clock_model);
 let clock_input=std::time::Instant::now();let input=request::register_inputs_cpu(pp,o);measure("input_registration",clock_input);
 let clock_registry=std::time::Instant::now();
 let mr=model.public();let ir=input.public();let g=&pp.g;
 let s=Snapshot{shapes:g.shapes.clone(),edge_meta:g.edge_meta.clone(),nvars:g.nvars.clone(),committed:g.committed.clone(),model:g.model.clone(),bool_typed:g.bool_typed.clone(),lookups:g.lookups.clone(),public:g.public.iter().map(|(&e,v)|(e,(**v).clone())).collect(),outputs:g.outputs.iter().map(|(e,v)|(*e,(**v).clone())).collect(),layout:pp.lay.clone(),specs:pp.specs.clone(),groups:[pp.model.clone(),pp.input.clone(),pp.req.clone(),pp.carry.clone()],mc:mr.commitments.to_vec(),ms:mr.statement.to_vec(),ml:mr.layout.to_vec(),ic:ir.commitments.to_vec(),is:ir.statement.to_vec(),il:ir.layout.to_vec(),slots_log:o.cfg.slots_log,batch,classes:classes.to_vec(),conv_pp};
 let registry_bytes=bincode::serialize(&s).unwrap();measure("public_registry_prepare_encode",clock_registry);
 let clock_registry_write=std::time::Instant::now();std::fs::write(format!("{dir}/public-registry.bin"),&registry_bytes).unwrap();measure("public_registry_write",clock_registry_write);drop(registry_bytes);
 let clock_prove=std::time::Instant::now();
 let pf=if selected_mode(){
  // Local machine admission, not protocol parameter weakening. Counts are
  // public layout/statement structure and exceeding the budget stops BEFORE link.
  let allocated:usize=pp.lay.tensors.iter().flat_map(|t|&t.pieces).map(|p|1usize.checked_shl((p.region.m+p.region.p).try_into().unwrap()).unwrap()).sum();
  let expected_sources=pp.lay.n_sources;
  let gate=|p:&Profile,st:&zk_torch_4::morpheus::link::Statement,b0:usize|{
   assert_eq!(p,&Profile::ternary());assert_eq!(b0,9);assert_eq!(st.n,expected_sources);
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
 let mut s:Snapshot=bincode::deserialize(&std::fs::read(format!("{dir}/public-registry.bin")).unwrap()).unwrap();s.layout.rebuild_index();admit_case(s.batch,&s.classes,s.conv_pp);
 // The admitted public descriptor selects the statement's existing convolution route.
 // It never chooses private data or removes a relation.
 set_conv(s.conv_pp);
 let m=Model::zero();let(d,mut wi,_,_,selector,out)=build(&m,&vec![0;s.batch*64],s.batch,&s.classes);run(&d,&mut wi,&[]);drop(m);
 erase_dummy_private_intermediates(&d,&mut wi,out);
 // Dummy values need not satisfy the claimed classes: they only construct typed
 // shapes and public literals. Private produced values are zeroed solely to
 // construct in-range dummy lookup indices; this is NOT an accepted witness.
 // Their openings/selectors/caches are discarded before the mandatory verifier.
 let canonical_slots=mask_budget(&d,&wi);
 let mut o=Opts::from_env();o.profile=if selected_mode(){Profile::ternary()}else{Profile::mini()};o.b0=9;o.cfg.slots_log=canonical_slots;o.cfg.tcl=4;o.cfg.req_planes_log=6;o.defer=false;o.auth=true;o.verify=true;o.coins=Coins::new([77;64],CoinProfile::Ternary);
 // Reconstruct every public semantic field from the typed DAG, not the supplied snapshot.
 let mut expected=request::prepare(&d,&wi,&[selector],&mut o);normalize_layout(&mut expected,&d,&o);assert_eq!(expected.g.protocol_conv_pp,s.conv_pp);
 assert_eq!(s.slots_log,o.cfg.slots_log);assert_eq!(s.shapes,expected.g.shapes);assert_eq!(s.edge_meta,expected.g.edge_meta);assert_eq!(s.nvars,expected.g.nvars);assert_eq!(s.committed,expected.g.committed);assert_eq!(s.model,expected.g.model);assert_eq!(s.bool_typed,expected.g.bool_typed);assert_eq!(s.lookups,expected.g.lookups);
 let public:HashMap<_,_>=s.public.iter().map(|(e,v)|(*e,v.clone())).collect();assert_eq!(public.len(),s.public.len());let canonical:HashMap<_,_>=expected.g.public.iter().map(|(&e,v)|(e,(**v).clone())).collect();assert_eq!(public,canonical,"noncanonical public literals or classification selector");
 let outputs:Vec<_>=expected.g.outputs.iter().map(|(e,v)|(*e,(**v).clone())).collect();assert_eq!(s.outputs,outputs,"noncanonical classification output");
 let specs=expected.specs.clone();assert_eq!(s.specs,specs,"not the declared public source types");
 let mut groups=[Vec::new(),Vec::new(),Vec::new()];for &sp in &specs {let group=match sp.0 {TKey::Edge(e) if expected.g.model[e]=>0,TKey::Edge(e) if d.producers[e].is_none() && expected.g.edge_meta[e].3==Role::Input && !expected.g.public.contains_key(&e)=>1,_=>2};groups[group].push(sp);}
 let (layout,ranges)=Layout::pack_groups(&o.profile,&groups,o.b0);assert_eq!(bincode::serialize(&s.layout).unwrap(),bincode::serialize(&layout).unwrap(),"noncanonical source layout");assert_eq!(&s.groups[..3],ranges.as_slice());assert_eq!(s.groups[3],layout.n_sources..layout.n_sources);drop(expected);drop(wi);measure("public_registry_read_reconstruction_admission",clock_admission);println!("PASS canonical_public_statement_and_layout_admission");if std::env::var_os("CNN_REGISTRY_ONLY").is_some(){return;}
 let g=Graph{dag:&d,shapes:s.shapes,edge_meta:s.edge_meta,nvars:s.nvars,committed:s.committed,model:s.model,bool_typed:s.bool_typed,lookups:s.lookups,public:s.public.into_iter().map(|(e,v)|(e,Arc::new(v))).collect(),outputs:s.outputs.into_iter().map(|(e,v)|(e,Arc::new(v))).collect(),cfg:o.cfg.clone(),protocol_conv_pp:s.conv_pp};
 let pd=ProverData{vals:vec![],sel:HashMap::new(),bank:vec![],#[cfg(feature="cuda")] cache:Default::default()};let mut pp=Prepared{g,pd,lay:s.layout,model:s.groups[0].clone(),input:s.groups[1].clone(),req:s.groups[2].clone(),carry:s.groups[3].clone(),specs:s.specs,ar:None};
 let mr=RegisteredModel{commitments:&s.mc,statement:&s.ms,layout:&s.ml};let ir=RegisteredInput{commitments:&s.ic,statement:&s.is,layout:&s.il};
 let clock_read=std::time::Instant::now();let bytes=std::fs::read(format!("{dir}/request-0.bin")).unwrap();measure("proof_read",clock_read);
 let clock_decode=std::time::Instant::now();let mut pf=decode(&pp,&o,&bytes).expect("canonical request wire");measure("proof_decode",clock_decode);
 let clock_canonical=std::time::Instant::now();assert_eq!(encode(&pp,&o,&pf),bytes);assert!(pf.auth.is_some());measure("proof_canonical_roundtrip",clock_canonical);
 let accepted=|p:&wire::RequestProof|verify_request(&pp,&o,0,Some(mr),ir,p,&mut Times::new()).is_some();
 let clock_verify=std::time::Instant::now();let mut valid_times=Times::new();assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut valid_times).is_some());measure("single_valid_verify_api",clock_verify);
 if std::env::var_os("CNN_MEASURE").is_some(){for (stage,seconds) in valid_times{println!("MEASURE stage=valid_{} seconds={:.9}",stage.replace(' ',"_"),seconds);}}
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
 let old_public=pp.g.public[&selector].clone();let old_output=pp.g.outputs[0].clone();let mut wrong=(*old_public).clone();wrong[s.classes[0]*s.batch]=0;wrong[((s.classes[0]+1)%10)*s.batch]=1;let wrong=Arc::new(wrong);pp.g.public.insert(selector,wrong.clone());pp.g.outputs[0]=(old_output.0,wrong);assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());pp.g.public.insert(selector,old_public);pp.g.outputs[0]=old_output;println!("PASS canonical_wrong_classification_rejected");
 // Batch-tail controls begin from the restored, valid public statement.
 if s.batch>1 {
  let last=s.batch-1;let mut wrong_classes=s.classes.clone();wrong_classes[last]=(wrong_classes[last]+1)%10;
  assert_eq!(&wrong_classes[..last],&s.classes[..last]);admit_case(s.batch,&wrong_classes,s.conv_pp);
  // This is a coherent alternative class descriptor and canonical one-hot
  // selector/output, not merely malformed metadata. The saved proof must fail.
  let wrong=Arc::new(DagBuilder::build_one_hot_selector_witness(s.batch,10,&wrong_classes,Role::Input).data.as_ref().unwrap().evaluations().iter().map(|&x|{let v=f_to_int(x);assert!(v==0||v==1);v as u64}).collect::<Vec<_>>());
  let old_public=pp.g.public[&selector].clone();let old_output=pp.g.outputs[0].clone();
  pp.g.public.insert(selector,wrong.clone());pp.g.outputs[0]=(old_output.0,wrong);
  assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());
  pp.g.public.insert(selector,old_public);pp.g.outputs[0]=old_output;
  println!("PASS canonical_last_row_wrong_classification_rejected");
  let old_output=pp.g.outputs[0].clone();let mut bad=(*old_output.1).clone();bad[last]^=1;
  pp.g.outputs[0]=(old_output.0,Arc::new(bad));
  assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());
  pp.g.outputs[0]=old_output;println!("PASS last_row_output_mutation_rejected");
 }
 let old_output=pp.g.outputs[0].clone();let mut v=(*old_output.1).clone();v[0]^=1;pp.g.outputs[0]=(old_output.0,Arc::new(v));assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());pp.g.outputs[0]=old_output;println!("PASS public_classification_mutation_rejected");
 let old_mode=pp.g.protocol_conv_pp;pp.g.protocol_conv_pp=!old_mode;assert!(verify_request(&pp,&o,0,Some(mr),ir,&pf,&mut Times::new()).is_none());pp.g.protocol_conv_pp=old_mode;println!("PASS convolution_mode_mutation_rejected");
 measure("all_proof_negative_controls",clock_negatives);
 println!("PASS COMPLETE_AUTHENTICATED_TRAINED_CNN profile={} compact={} native_security_audit_separate=true",o.profile.name,compact_mode());
}

fn encode(pp:&Prepared,o:&Opts,pf:&wire::RequestProof)->Vec<u8>{if compact_mode(){wire::encode_request_compact(&o.profile,&request::request_support(pp,o).unwrap(),pf).unwrap()}else{wire::encode_request(&o.profile,pf)}}
fn decode(pp:&Prepared,o:&Opts,b:&[u8])->Option<wire::RequestProof>{if compact_mode(){wire::decode_request_compact(&o.profile,&request::request_support(pp,o)?,b)}else{wire::decode_request(&o.profile,b)}}
fn verify_request(pp:&Prepared,o:&Opts,j:u64,m:Option<RegisteredModel<'_>>,i:RegisteredInput<'_>,pf:&wire::RequestProof,t:&mut Times)->Option<(Vec<zk_torch_4::morpheus::link::Row>,K8,Vec<u8>)>{if compact_mode(){request::verify_request_cpu_compact(pp,o,j,m,i,pf,t)}else{request::verify_request_cpu(pp,o,j,m,i,pf,t)}}
