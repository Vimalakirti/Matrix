//! Structural codec/version tests, not an accepting authenticated proof.
use zk_torch_4::morpheus::{params::Profile, wire, link::LinkProof, ring::{AElem,N}, field8::K8, transcript::{MTranscript,Item,Stage}};
fn main() {
 assert!(!cfg!(feature="cuda"));
 let p=Profile::toy();
 let link=LinkProof {g_commit:vec![],c_o:vec![vec![0;N];p.rank],rounds:p.degs().into_iter().map(|d|vec![K8::ZERO;d+1]).collect(),a_hat:vec![],t_o:AElem::zero()};
 let req=wire::RequestProof {salt:[0;64],piop:vec![],c_src:vec![],link:link.clone(),auth:None};
 let fin=wire::FinalProof {salt:[0;64],link,auth:None};
 let rb=wire::encode_request(&p,&req);let fb=wire::encode_final(&p,&fin);
 assert_eq!(wire::decode_request(&p,&rb),Some(req));assert_eq!(wire::decode_final(&p,&fb),Some(fin));
 for old in [b"MRPHSv1\0",b"MRPHSv2\0",b"MRPHSv3\0",b"MRPHSv5\0"] {
  let mut r=rb.clone();r[..8].copy_from_slice(old);assert!(wire::decode_request(&p,&r).is_none());
  let mut f=fb.clone();f[..8].copy_from_slice(old);assert!(wire::decode_final(&p,&f).is_none());
 }
 let mut old_profile=p.clone();old_profile.name="morpheus-toy";
 assert!(wire::decode_request(&old_profile,&rb).is_none());assert!(wire::decode_final(&old_profile,&fb).is_none());
 let mut t=MTranscript::new(p.name,&[0;64]);t.absorb(Item::Statement,&[wire::PROTOCOL_VERSION,u64::MAX]);
 let before=t.literal_prefix().to_vec();t.challenge_k(Stage::LinkRound,0,0);
 assert_ne!(before,t.literal_prefix());
 println!("PASS actual request/final wire roundtrips; old/future magic and old profile rejected; literal prefix chronology. Structurally valid unauthenticated fixtures, NOT accepted proofs.");
}
