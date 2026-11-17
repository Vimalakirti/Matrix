use zk_torch_4::morpheus::{self,request::{self,Opts,Prepared},piop::{self,Ctx,Walk,layout::{Layout,TKey}},transcript::{MTranscript,Item},wire,params::Profile,keys::KeyKind,link};
fn absorb_model(tr: &mut MTranscript, c_model: &[Vec<Vec<u64>>]) {
    tr.absorb(Item::Statement, &[c_model.len() as u64]);
    for c in c_model { for x in c { tr.absorb(Item::SourceCommitment, x); } }
}

/// The registry's public layout, literally: every tensor's key, index
/// width, planes, bit type and pieces (source, base, size, planes, plane
/// offset, block split).
fn absorb_layout(tr: &mut MTranscript, lay: &Layout) {
    let mut w = vec![lay.n_sources as u64, lay.ell as u64, lay.min_lg as u64, lay.tensors.len() as u64];
    for t in &lay.tensors {
        match t.key {
            TKey::Edge(e) => w.extend([1, e as u64]),
            TKey::Sel { node, chunk } => w.extend([2, node as u64, chunk as u64]),
            TKey::Wg => w.push(3),
            TKey::Carry => w.push(4),
        }
        w.extend([t.n as u64, t.p as u64, t.bits as u64, t.pieces.len() as u64]);
        for pc in &t.pieces {
            let r = &pc.region;
            w.extend([r.source as u64, r.base as u64, r.m as u64, r.p as u64, pc.plane0 as u64, pc.hb as u64, pc.hbits as u64]);
        }
    }
    tr.absorb(Item::Layout, &w);
}

fn request_transcript(o: &Opts, pp: &Prepared, j: u64, salt: &[u8; 64], c_model: &[Vec<Vec<u64>>], c_input: &[Vec<Vec<u64>>], c_req: &[Vec<Vec<u64>>]) -> Option<MTranscript> {
    let words = morpheus::statement::words(&pp.g, &o.profile, &o.seed, o.b0, o.defer, o.carry_log).ok()?;
    let mut tr = MTranscript::new(o.profile.name, salt);
    tr.absorb(Item::Statement, &[wire::PROTOCOL_VERSION, 0, o.defer as u64]);
    tr.absorb(Item::Statement, &words);
    absorb_layout(&mut tr, &pp.lay);
    if let Some(ar) = &pp.ar {
        assert!(ar.validate(&pp.g), "invalid public AR statement");
        tr.absorb(Item::Statement, &ar.words());
        // Bind exact canonical public selector arrays before graph challenges.
        for e in std::iter::once(&ar.embedding).chain(ar.selectors.iter()).chain(ar.rank_constants.iter().flatten()) {
            tr.absorb(Item::Statement, &[*e as u64]);
            tr.absorb(Item::Statement, &pp.g.public[e]);
        }
    }
    tr.absorb(Item::Statement, &[1]); absorb_model(&mut tr, c_model);
    tr.absorb(Item::Statement, &[2]); absorb_model(&mut tr, c_input);
    tr.absorb(Item::Statement, &[3, c_req.len() as u64, j]);
    for c in c_req { for x in c { tr.absorb(Item::SourceCommitment, x); } }
    Some(tr)
}

pub fn run(pp:&mut Prepared,o:&Opts){
 let mut total=0u128;let mut raw;
 pp.pd.bank=piop::sample_bank(&o.coins,request::req_opening(0),1<<o.cfg.slots_log);
 raw=piop::fill_sources(&pp.g,&pp.pd,&pp.lay,o.profile.ell());
 for (e,r) in raw.iter_mut().enumerate(){let owner=if pp.model.contains(&e){(1u64<<63)+(e-pp.model.start) as u64}else if pp.input.contains(&e){(1u64<<62)+(e-pp.input.start) as u64}else{(1u64<<24)+(e-pp.req.start) as u64};link::fill_source_suffix(&o.profile,&o.coins,owner,r);
 assert!(r.iter().all(|&v|(-1..=1).contains(&v)));let pe:u128=r[..o.profile.source_payload_cols*morpheus::ring::N].iter().map(|&v|(v as i128*v as i128) as u128).sum();total+=pe;println!("ENERGY source={e} payload={pe} coordinates={}",r.len());}
 o.profile.check_caps(&raw);println!("ENERGY total={total} cap={}",1u128<<o.profile.s_max_log2);
 for t in &pp.lay.tensors {println!("LAYOUT {:?}",t);}
 let c:Vec<_>=raw.iter().map(|r|morpheus::auth::lin(&o.profile,&o.seed,KeyKind::Source,o.profile.source_payload_cols,&link::raw_to_cols(r))).collect();
 let tr=request_transcript(o,pp,0,&o.coins.salt(request::req_opening(0)),&c[pp.model.clone()],&c[pp.input.clone()],&c[pp.req.clone()]).unwrap();
 let mut pc=Ctx::prover(tr.clone(),&pp.pd.bank,&pp.lay,o.cfg.slots_log);Walk::new(&pp.g,Some(&pp.pd)).run(&mut pc);
 let mut vc=Ctx::verifier(tr,pc.proof.clone(),&pp.lay,o.cfg.slots_log);Walk::new(&pp.g,None).run(&mut vc);assert!(vc.ok);assert!(vc.consumed());assert_eq!(pc.slots_used(),vc.slots_used());println!("MASK {} rows={} messages={}",pc.census_line(),pc.rows.len(),pc.proof.len());
 println!("PASS ACTUAL_PIOP_CENSUS_ONLY_NOT_AUTHENTICATED_PROOF");
}
