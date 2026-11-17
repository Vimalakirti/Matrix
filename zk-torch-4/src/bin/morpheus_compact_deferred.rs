//! Complete two-request compact-transport deferred toy proof; never selected security evidence.
use zk_torch_4::{dag::{DagBuilder,Witness,DataType,Role},util::arith::int_to_f};
use zk_torch_4::morpheus::{request::{self,Opts,Times,QueueEntry,RegisteredModel},wire,params::Profile,coins::{Coins,CoinProfile},field8::K8};
use std::{path::PathBuf,panic::{catch_unwind,AssertUnwindSafe}};
fn main() {
 assert!(!cfg!(feature="cuda"));
 // Sparse all-zero declared bit blocks must survive final masking. Compare
 // byte-for-byte with the independent dense reference and unchanged verifier.
 { use zk_torch_4::morpheus::{link,link_fast,auth,keys::{KeySeed,KeyKind},transcript::{MTranscript,Item}};
   let p=Profile::toy();let seed=KeySeed([1;8]);let coins=Coins::new([23;64],CoinProfile::Ternary);
   for one in [false,true] {
     let mut raw=vec![vec![0i8;1<<p.ell()]];if one {raw[0][0]=1;}
     link::fill_source_suffix(&p,&coins,12,&mut raw[0]);
     let cs=vec![auth::lin(&p,&seed,KeyKind::Source,p.source_payload_cols,&link::raw_to_cols(&raw[0]))];
     let st=link::Statement{n:1,bits:vec![vec![(0,12)]],rows:vec![]};
     let mut tr=MTranscript::new(p.name,&coins.salt(91));for c in &cs {for r in c {tr.absorb(Item::SourceCommitment,r);}}
     let mut ft=tr.clone();let mut rt=tr.clone();let mut vt=tr.clone();
     let f=link_fast::prove_with(&p,&seed,&coins,91,&mut ft,&st,&raw,9,&link_fast::cpu_backend,&mut link_fast::CpuServices,Some(&cs),false);
     let r=link::prove(&p,&seed,&coins,91,&mut rt,&st,&raw);
     let mut fw=wire::W::default();let mut rw=wire::W::default();wire::put_link(&mut fw,&f.proof);wire::put_link(&mut rw,&r.proof);assert_eq!(fw.b,rw.b,"sparse/dense exact link bytes");
     assert!(link::verify(&p,&mut vt,&st,&cs,&f.proof).is_some());
     println!("PASS sparse_declared_zero_bit_blocks_match_dense_reference one_nonzero={}",one);
   }
 }
 if std::env::var_os("MORPHEUS_COMPACT_LINK_PREFLIGHT_ONLY").is_some(){return;}

 let dir=PathBuf::from(std::env::var_os("MORPHEUS_COMPACT_DEFERRED_DIR").expect("set private local evidence output directory"));
 std::fs::create_dir_all(&dir).unwrap();
 let replay=std::env::var_os("MORPHEUS_COMPACT_DEFERRED_REPLAY").is_some();
 let mut b=DagBuilder::new();
 let x=b.committed_input(Witness::new(vec![4],vec![1,2,3,4].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Input));
 let m=b.param(Witness::new(vec![4],vec![5,6,7,8].into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Constant));
 let _y=b.add(x,m)[0];let(d,w)=b.compile();
 let mut o=Opts::from_env();o.profile=Profile::toy();o.b0=9;o.cfg.slots_log=6;o.cfg.tcl=4;o.cfg.req_planes_log=4;o.defer=true;o.auth=true;o.verify=true;
 o.coins=Coins::new([19;64],CoinProfile::Ternary);
 println!("PROFILE={} TOY_ALGEBRA_ONLY=true SECURITY_100BIT_CLAIM=false FIXED_TEST_SEED=true DEFERRED_TWO_REQUESTS=true",o.profile.name);
 let mut pp=request::prepare(&d,&w,&[],&mut o);assert!(!pp.g.public.contains_key(&x));assert!(!pp.model.is_empty());assert!(!pp.input.is_empty());assert_eq!(pp.carry.len(),1);
 let request_support=request::request_support(&pp,&o).expect("registered request support");
 let final_support=request::final_support(&pp,&o).expect("registered final support");
 let model=request::register_cpu(&pp,&o);let input=request::register_inputs_cpu(&pp,&o);
 let model_commitments=model.public().commitments.to_vec();let model_count=model.source_count();
 let mut accepted=Vec::new();let mut request_proofs=Vec::new();let mut carries=Vec::new();
 for j in 0..2u64 {
   let path=dir.join(format!("request-{j}.bin"));
   let bytes=if replay {std::fs::read(&path).unwrap()}else{
     let out=request::prove_request_cpu_compact(&mut pp,&o,j,&model,&input);
     assert!(request::verify_request_cpu_compact(&pp,&o,j,Some(model.public()),input.public(),&out.proof,&mut Times::new()).is_some());
     carries.push(out.carry_raw.expect("private carry witness"));wire::encode_request_compact(&o.profile,&request_support,&out.proof).expect("compact request encode")
   };
   let decoded=wire::decode_request_compact(&o.profile,&request_support,&bytes).expect("request wire decode");assert_eq!(wire::encode_request_compact(&o.profile,&request_support,&decoded).expect("compact request reencode"),bytes);
   let verdict=request::verify_request_cpu_compact(&pp,&o,j,Some(model.public()),input.public(),&decoded,&mut Times::new()).expect("request verification");
   assert!(!verdict.0.is_empty());assert!(!verdict.2.is_empty());
   println!("PASS request_wire_authenticated_PENDING_FINAL j={} bytes={} attempt={} deferred_rows={}",j,bytes.len(),decoded.auth.as_ref().unwrap().attempt,verdict.0.len());
   if !replay {std::fs::write(path,&bytes).unwrap();}
   accepted.push(verdict);request_proofs.push(decoded);
 }
 // Entries are constructed only after successful independent verifier replay
 // of each decoded request, never from prover-declared rows/digests/prefixes.
 let q:Vec<QueueEntry>=accepted.iter().enumerate().map(|(j,(rows,digest,prefix))|QueueEntry{j:j as u64,deferred:rows,c_carry:request_proofs[j].c_src.last().unwrap(),digest:*digest,literal_prefix:prefix}).collect();
 let bytes=if replay {std::fs::read(dir.join("final.bin")).unwrap()}else{
   let pf=request::finalize_cpu_compact(&pp,&o,0,&model,&q,&carries);
   assert!(request::verify_final_cpu_compact(&pp,&o,0,model.public(),&q,&pf,&mut Times::new()));
   let bytes=wire::encode_final_compact(&o.profile,&final_support,&pf).expect("compact final encode");std::fs::write(dir.join("final.bin"),&bytes).unwrap();bytes
 };
 assert_eq!(model.source_count(),model_count);assert_eq!(model.public().commitments,&model_commitments);println!("PASS immutable_registered_model_preserved");
 if !replay {
   assert!(catch_unwind(AssertUnwindSafe(||request::finalize_cpu_compact(&pp,&o,0,&model,&q,&carries))).is_err());
   assert_eq!(model.source_count(),model_count);assert_eq!(model.public().commitments,&model_commitments);
   assert!(catch_unwind(AssertUnwindSafe(||request::prove_request_cpu_compact(&mut pp,&o,0,&model,&input))).is_err());
   assert_eq!(model.source_count(),model_count);println!("PASS duplicate_request_and_final_opening_fail_closed_model_preserved");
 }
 pp.pd.bank.clear();pp.pd.vals.clear();pp.pd.sel.clear();o.coins=Coins::new([77;64],CoinProfile::Ternary);
 let mut decoded=wire::decode_final_compact(&o.profile,&final_support,&bytes).expect("final wire decode");assert_eq!(wire::encode_final_compact(&o.profile,&final_support,&decoded).expect("compact final reencode"),bytes);
 assert!(request::verify_final_cpu_compact(&pp,&o,0,model.public(),&q,&decoded,&mut Times::new()));
 println!("PASS final_wire_authenticated bytes={} attempt={} queue=2",bytes.len(),decoded.auth.as_ref().unwrap().attempt);
 let reject=|queue:&[QueueEntry],pf:&wire::FinalProof|!request::verify_final_cpu_compact(&pp,&o,0,model.public(),queue,pf,&mut Times::new());
 let mut reverse=q.clone();reverse.reverse();assert!(reject(&reverse,&decoded));println!("PASS queue_reorder_rejected");
 let mut bad=q.clone();bad[0].digest=bad[0].digest.add(&K8::ONE);assert!(reject(&bad,&decoded));println!("PASS queue_digest_mutation_rejected");
 let mut prefix=q[0].literal_prefix.to_vec();prefix[0]^=1;let mut bad=q.clone();bad[0].literal_prefix=&prefix;assert!(reject(&bad,&decoded));println!("PASS queue_literal_prefix_mutation_rejected");
 let mut carry=q[0].c_carry.to_vec();carry[0][0]^=1;let mut bad=q.clone();bad[0].c_carry=&carry;assert!(reject(&bad,&decoded));println!("PASS queue_carry_mutation_rejected");
 let mut bad=q.clone();bad[0].j=2;assert!(reject(&bad,&decoded));println!("PASS queue_request_id_mutation_rejected");
 let mut rows=q[0].deferred.to_vec();rows[0].y=rows[0].y.add(&K8::ONE);let mut bad=q.clone();bad[0].deferred=&rows;assert!(reject(&bad,&decoded));println!("PASS queue_rows_mutation_rejected");
 let mr=model.public();let mut mc=mr.commitments.to_vec();mc[0][0][0]^=1;assert!(!request::verify_final_cpu_compact(&pp,&o,0,RegisteredModel{commitments:&mc,..mr},&q,&decoded,&mut Times::new()));println!("PASS wrong_registered_model_rejected");
 let apf=decoded.auth.take();assert!(reject(&q,&decoded));decoded.auth=apf;println!("PASS missing_final_auth_rejected");
 decoded.auth.as_mut().unwrap().z.w[0]^=1;assert!(reject(&q,&decoded));decoded.auth.as_mut().unwrap().z.w[0]^=1;println!("PASS final_response_mutation_rejected");
 let old=decoded.link.a_hat[0].0[0];decoded.link.a_hat[0].0[0]=old.add(&K8::ONE);assert!(reject(&q,&decoded));decoded.link.a_hat[0].0[0]=old;println!("PASS final_endpoint_mutation_rejected");
 assert!(!request::verify_final_cpu_compact(&pp,&o,1,model.public(),&q,&decoded,&mut Times::new()));println!("PASS wrong_batch_id_rejected");
 assert!(wire::decode_final_compact(&o.profile,&final_support,&bytes[..bytes.len()-1]).is_none());let mut trailing=bytes.clone();trailing.push(0);assert!(wire::decode_final_compact(&o.profile,&final_support,&trailing).is_none());println!("PASS malformed_final_wire_rejected");
 use sha3::{Digest,Sha3_256};println!("FINAL_SHA3_256={:x}",Sha3_256::digest(&bytes));
 println!("PASS COMPLETE_COMPACT_AUTHENTICATED_TWO_REQUEST_DEFERRED_TOY_BATCH_ONLY");
}
