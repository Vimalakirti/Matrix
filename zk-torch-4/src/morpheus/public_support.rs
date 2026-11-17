//! Deterministic public allocation support. No witness or challenge is an input.
//! This structural admission supplements, and does not replace, authentication
//! and semantic/canonical reconstruction of the immutable registered layout.
use super::{auth_compact::Support,params::Profile,piop::layout::Layout,ring::N};
use std::{collections::HashSet,ops::Range};
fn pow(n:usize)->Option<usize>{1usize.checked_shl(u32::try_from(n).ok()?)}
pub fn derive(p:&Profile,lay:&Layout,ranges:&[Range<usize>])->Option<Support>{
    if lay.ell!=p.ell()||lay.n_sources==0||lay.n_sources>p.n_max{return None}
    let cap=p.source_payload_cols.checked_mul(N)?;
    let mut chosen=vec![false;lay.n_sources];
    for r in ranges{if r.start>r.end||r.end>lay.n_sources{return None}for e in r.clone(){if chosen[e]{return None}chosen[e]=true;}}
    if !chosen.iter().any(|&v|v){return None}
    let mut allocation=vec![Vec::<(usize,usize)>::new();lay.n_sources];let mut columns=vec![false;p.source_cols()];let mut keys=HashSet::new();
    for t in &lay.tensors{
        if !keys.insert(t.key)||t.pieces.is_empty(){return None}
        let planes=pow(t.p)?;let width=pow(t.n)?;let total=planes.checked_mul(width)?;
        let mut coverage=Vec::<(usize,usize,usize,usize)>::new();let mut volume=0usize;
        for piece in &t.pieces{let r=&piece.region;
            if r.source>=lay.n_sources||piece.hbits!=t.n.saturating_sub(r.m){return None}
            let size=pow(r.m.checked_add(r.p)?)?;let end=r.base.checked_add(size)?;
            if r.base%size!=0||end>cap{return None}
            let plane_size=pow(r.p)?;let plane_end=piece.plane0.checked_add(plane_size)?;
            if plane_end>planes||piece.plane0%plane_size!=0||piece.hb>=pow(piece.hbits)?{return None}
            let ixsize=pow(r.m.min(t.n))?;let ix0=piece.hb.checked_mul(ixsize)?;let ixend=ix0.checked_add(ixsize)?;if ixend>width{return None}
            for &(a,b,c,d)in &coverage{if piece.plane0<b&&a<plane_end&&ix0<d&&c<ixend{return None}}
            coverage.push((piece.plane0,plane_end,ix0,ixend));volume=volume.checked_add(plane_size.checked_mul(ixsize)?)?;
            allocation[r.source].push((r.base,end));
            if chosen[r.source]{for c in r.base/N..=(end-1)/N{columns[c]=true;}}
        }
        if volume!=total{return None}
    }
    for intervals in &mut allocation{intervals.sort_unstable();if intervals.windows(2).any(|w|w[0].1>w[1].0){return None}}
    for c in p.source_payload_cols..p.source_cols(){columns[c]=true}
    Support::from_public_columns(p,columns.iter().enumerate().filter_map(|(c,&yes)|yes.then_some(c)).collect())
}
#[cfg(any(test,feature="cpu-replay"))] pub mod functional_tests{
use super::*;use super::super::piop::layout::TKey;
#[cfg_attr(test,test)] pub fn all_allocated_pieces_and_suffix_are_included(){let p=Profile::toy();let(lay,rs)=Layout::pack_groups(&p,&[vec![(TKey::Edge(1),8,2,false)],vec![(TKey::Wg,12,0,true),(TKey::Carry,11,0,true)]],0);let s=derive(&p,&lay,&rs).unwrap();assert_eq!(s.columns(),&[0,1,3,4,5,6,7]);let second=derive(&p,&lay,&[rs[1].clone()]).unwrap();assert_eq!(second,s);let mut bad=lay.clone();bad.tensors[0].pieces[0].region.base=1;assert!(derive(&p,&bad,&rs).is_none());let mut bad=lay.clone();bad.tensors[0].pieces[0].region.source=lay.n_sources;assert!(derive(&p,&bad,&rs).is_none());let mut bad=lay.clone();bad.tensors[1].pieces[0].region.base=p.source_payload_cols*N;assert!(derive(&p,&bad,&rs).is_none());let mut bad=lay.clone();bad.tensors[2].pieces[0].region.base=0;assert!(derive(&p,&bad,&rs).is_none());let mut bad=lay.clone();bad.tensors[0].pieces.clear();assert!(derive(&p,&bad,&rs).is_none());}
}
