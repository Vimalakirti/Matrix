//! Complete compact-transport toy-profile CPU proof, NOT selected-profile security evidence.
use zk_torch_4::{dag::{DagBuilder,Witness,DataType,Role},util::arith::int_to_f};
use zk_torch_4::morpheus::{request::{self,Opts,Times,RegisteredModel,RegisteredInput},wire,params::Profile,coins::{Coins,CoinProfile},field8::K8};
fn main() {
 assert!(!cfg!(feature="cuda"));
 // Independent signed-CRT boundary oracle, with overflow checks enabled.
 { use zk_torch_4::morpheus::{ring,pcs_maps};
   let q=almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as i128;
   let crt=ring::Crt::new(3);
   for z in [-1i128,-(1i128<<90),-q/2,0,1,1i128<<90] {
     let r:Vec<u64>=ring::NTT_PRIMES[..3].iter().map(|(p,_)|z.rem_euclid(*p as i128) as u64).collect();
     assert_eq!(crt.reconstruct(&r),z.rem_euclid(q) as u64);
   }
   let mut cols=vec![vec![0i64;4096];8];cols[0][0]=-1;cols[0][1]=1;
   assert_eq!(pcs_maps::decode_h(&cols).0[0].0[0],1);
   let mut beta=[0i8;64];beta[0]=-1;let mut v=vec![0u64;4096];v[0]=(q-1) as u64;
   assert_eq!(pcs_maps::act_c0_ring_q(&beta,&v)[0],1);
   println!("PASS signed_modular_boundary_regressions");
 }

 let mut b=DagBuilder::new();
 let x=b.committed_input(Witness::new(vec![4],vec![1,2,3,4].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Input));
 let m=b.param(Witness::new(vec![4],vec![5,6,7,8].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Constant));
 let _y=b.add(x,m)[0]; let(d,w)=b.compile();
 let mut o=Opts::from_env(); o.profile=Profile::toy();o.b0=9;o.cfg.slots_log=6;o.cfg.tcl=4;o.cfg.req_planes_log=4;o.defer=false;o.auth=true;o.verify=true;
 o.coins=Coins::new([19;64],CoinProfile::Ternary);
 println!("PROFILE={} TOY_ALGEBRA_ONLY=true SECURITY_100BIT_CLAIM=false FIXED_TEST_SEED=true",o.profile.name);
 let mut pp=request::prepare(&d,&w,&[],&mut o);
 assert!(!pp.g.public.contains_key(&x));assert!(!pp.model.is_empty());assert!(!pp.input.is_empty());assert!(!pp.g.outputs.is_empty());
 println!("GROUPS model={:?} private_input={:?} request={:?}",pp.model,pp.input,pp.req);
 let support=request::request_support(&pp,&o).expect("registered request support");
 let model=request::register_cpu(&pp,&o);let input=request::register_inputs_cpu(&pp,&o);
 let bytes=if let Some(path)=std::env::var_os("MORPHEUS_COMPACT_PROOF_IN") {
   println!("VERIFY_ONLY existing serialized proof; no proving");std::fs::read(path).expect("read proof")
 } else {
   let pf=request::prove_request_cpu_compact(&mut pp,&o,0,&model,&input).proof;
   assert!(request::verify_request_cpu_compact(&pp,&o,0,Some(model.public()),input.public(),&pf,&mut Times::new()).is_some());println!("PASS raw_full_verify");
   let bytes=wire::encode_request_compact(&o.profile,&support,&pf).expect("compact request encode");
   if let Some(path)=std::env::var_os("MORPHEUS_COMPACT_PROOF_OUT") {std::fs::write(path,&bytes).expect("persist proof");println!("PASS proof_bytes_persisted");}
   bytes
 };
 // The verifier must work without the prover's masks, tensor cache or coins.
 pp.pd.bank.clear();pp.pd.vals.clear();pp.pd.sel.clear();o.coins=Coins::new([77;64],CoinProfile::Ternary);
 let mut decoded=wire::decode_request_compact(&o.profile,&support,&bytes).expect("decode");assert!(decoded.auth.is_some());
 assert!(request::verify_request_cpu_compact(&pp,&o,0,Some(model.public()),input.public(),&decoded,&mut Times::new()).is_some());println!("PASS serialized_full_verify bytes={} attempt={}",bytes.len(),decoded.auth.as_ref().unwrap().attempt);
 let reject=|p:&wire::RequestProof|request::verify_request_cpu_compact(&pp,&o,0,Some(model.public()),input.public(),p,&mut Times::new()).is_none();
 decoded.auth.as_mut().unwrap().z.w[0]^=1;assert!(reject(&decoded));decoded.auth.as_mut().unwrap().z.w[0]^=1;println!("PASS response_mutation_rejected");
 let old_piop=decoded.piop[0];decoded.piop[0]=old_piop.add(&K8::ONE);assert!(reject(&decoded));decoded.piop[0]=old_piop;println!("PASS piop_mutation_rejected");
 decoded.piop.push(K8::ZERO);assert!(reject(&decoded));decoded.piop.pop();println!("PASS extra_piop_rejected");
 let old=decoded.link.a_hat[0].0[0];decoded.link.a_hat[0].0[0]=old.add(&K8::ONE);assert!(reject(&decoded));decoded.link.a_hat[0].0[0]=old;println!("PASS link_mutation_rejected");
 let auth=decoded.auth.take();assert!(reject(&decoded));decoded.auth=auth;println!("PASS missing_auth_rejected");
 assert!(request::verify_request_cpu_compact(&pp,&o,1,Some(model.public()),input.public(),&decoded,&mut Times::new()).is_none());println!("PASS request_id_mutation_rejected");
 let mr=model.public();let mut mc=mr.commitments.to_vec();mc[0][0][0]^=1;
 assert!(request::verify_request_cpu_compact(&pp,&o,0,Some(RegisteredModel{commitments:&mc,..mr}),input.public(),&decoded,&mut Times::new()).is_none());println!("PASS registered_model_mutation_rejected");
 let ir=input.public();let mut ic=ir.commitments.to_vec();ic[0][0][0]^=1;
 assert!(request::verify_request_cpu_compact(&pp,&o,0,Some(model.public()),RegisteredInput{commitments:&ic,..ir},&decoded,&mut Times::new()).is_none());println!("PASS registered_private_input_mutation_rejected");
 assert!(wire::decode_request_compact(&o.profile,&support,&bytes[..bytes.len()-1]).is_none());let mut trailing=bytes.clone();trailing.push(0);assert!(wire::decode_request_compact(&o.profile,&support,&trailing).is_none());println!("PASS malformed_wire_rejected");
 use sha3::{Digest,Sha3_256};println!("PROOF_SHA3_256={:x}",Sha3_256::digest(&bytes));

 let mut mutated=bytes.clone();let last=mutated.len()-1;mutated[last]^=1;
 let bad=wire::decode_request_compact(&o.profile,&support,&mutated).expect("canonical byte mutation must decode");assert!(reject(&bad));println!("PASS serialized_response_mutation_rejected");
 let (e,vals)=pp.g.outputs[0].clone();let mut vals=(*vals).clone();vals[0]+=1;pp.g.outputs[0]=(e,std::sync::Arc::new(vals));
 assert!(request::verify_request_cpu_compact(&pp,&o,0,Some(model.public()),input.public(),&decoded,&mut Times::new()).is_none());println!("PASS public_output_mutation_rejected");
 println!("PASS COMPLETE_COMPACT_AUTHENTICATED_TOY_PROOF_ONLY");
}
