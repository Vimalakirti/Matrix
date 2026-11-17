//! Public-column restricted CPU authentication. Original key and coin addresses.
//! The caller must derive support from the authenticated public layout, never data.
use super::{auth::{self, AuthError, AuthProof, AuxTuple, Point, Responses, Tuple}, coins::{Address, BitStream, Coins, Purpose}, field8::{eq_table,K8}, gaussian::{CoordinateStreams, Stats}, gaussian_fast::{self,FastSigma}, keys::{key_element,KeyKind,KeySeed}, params::{Profile,A_ENC_COLS}, pcs_maps::{act_c0_ring_int,decode_h,e_o,mul_a}, rejection::{self,Coin,Ratio}, ring::{self,AElem,N,DS,T}, transcript::MTranscript};
use num_bigint::{BigInt,BigUint};
use rayon::prelude::*;

/// Explicit original physical column indices. Construction is registry-owned.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct Support { columns: Vec<usize>, full: usize }
impl Support {
    pub fn from_public_columns(p:&Profile, columns:Vec<usize>)->Option<Self> {
        if columns.is_empty() || columns.windows(2).any(|w|w[0]>=w[1]) || columns.last().copied()? >= p.source_cols() {return None}
        if !(p.source_payload_cols..p.source_cols()).all(|c|columns.binary_search(&c).is_ok()) {return None}
        Some(Self{columns,full:p.source_cols()})
    }
    pub fn columns(&self)->&[usize]{&self.columns}
    pub fn len(&self)->usize{self.columns.len()}
    pub fn matches(&self,p:&Profile)->bool{self.full==p.source_cols() && Self::from_public_columns(p,self.columns.clone()).is_some()}
}

/// Bounded-memory original-index matvec: transform one column, reuse over rows.
/// No input clone or all-column NTT cache. The CRT bound is the original bound.
pub fn lin_indexed(p:&Profile,seed:&KeySeed,kind:KeyKind,pc:usize,indices:&[usize],get:&dyn Fn(usize)->Vec<i64>)->Vec<Vec<u64>> {
    assert!(indices.windows(2).all(|w|w[0]<w[1]));
    assert!(indices.iter().all(|&c|c<pc+p.hiding+p.rank));
    let mut maximum=1u64;
    for &c in indices {if c<pc+p.hiding {maximum=maximum.max(get(c).iter().map(|x|x.unsigned_abs()).max().unwrap_or(0));}}
    let count=indices.iter().filter(|&&c|c<pc+p.hiding).count();
    let bound=(count.max(1) as f64).log2()+12.0+63.0+(maximum as f64).log2();
    let k=ring::primes_for_bound(bound);let tabs=&ring::TABLES[..k];let crt=ring::Crt::new(k);
    let mut acc=vec![vec![vec![0u64;N];k];p.rank];
    for &c in indices {if c>=pc+p.hiding {continue}
        let x=get(c);assert_eq!(x.len(),N);
        let xh:Vec<Vec<u64>>=tabs.iter().map(|t|{let mut v:Vec<u64>=x.iter().map(|&v|v.rem_euclid(t.p as i64) as u64).collect();t.forward(&mut v);v}).collect();
        acc.par_iter_mut().enumerate().for_each(|(row,row_acc)|{let key=key_element(seed,kind,row as u32,c as u64);
            for (pi,t) in tabs.iter().enumerate(){let mut a:Vec<u64>=key.iter().map(|&v|ring::center(v).rem_euclid(t.p as i64) as u64).collect();t.forward(&mut a);for j in 0..N{row_acc[pi][j]=(row_acc[pi][j]+a[j]*xh[pi][j])%t.p;}}
        });
    }
    let mut out=vec![vec![0;N];p.rank];
    for row in 0..p.rank {for (pi,t) in tabs.iter().enumerate(){t.inverse(&mut acc[row][pi]);} let ec=pc+p.hiding+row;let e=if indices.binary_search(&ec).is_ok(){get(ec)}else{vec![0;N]};
        for j in 0..N{let mut r=[0;5];for pi in 0..k{r[pi]=acc[row][pi][j];}out[row][j]=((crt.reconstruct(&r[..k]) as u128+ring::from_signed(e[j]) as u128)%almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as u128) as u64;}
    }out
}
pub fn ev_indexed(xi:&[K8],indices:&[usize],get:&dyn Fn(usize)->Vec<i64>)->AElem{
    let ek=eq_table(&xi[..6]);let ej=eq_table(&xi[6..12]);let el=eq_table(&xi[12..]);let mut a=vec![K8::ZERO;DS];
    for &c in indices{let col=get(c);for j in 0..T{let w=el[c].mul(&ej[j]);for m in 0..DS{let v=col[j+T*m];if v!=0{a[m]=a[m].add(&w.scale(ring::from_signed(v)));}}}}
    mul_a(&ring::g_of_weights(&std::array::from_fn(|i|ek[i])),&AElem(a))
}
fn fx(p:&Profile,seed:&KeySeed,pt:&Point,s:&Support,w:&[Vec<i64>],l:&[Vec<i64>])->Tuple{
    let get=|c|w[s.columns.binary_search(&c).unwrap()].clone();
    Tuple{c:lin_indexed(p,seed,KeyKind::Source,p.source_payload_cols,&s.columns,&get),g:lin_dense(p,seed,KeyKind::Link,A_ENC_COLS,l),a:ev_indexed(&pt.xi,&s.columns,&get).add(&decode_h(&l[..A_ENC_COLS]).scale_k(&pt.rho.mul(&K8::ONE.sub(&pt.rho))))}
}
fn lin_dense(p:&Profile,seed:&KeySeed,kind:KeyKind,pc:usize,x:&[Vec<i64>])->Vec<Vec<u64>>{lin_indexed(p,seed,kind,pc,&(0..x.len()).collect::<Vec<_>>(),&|c|x[c].clone())}
fn fo(p:&Profile,seed:&KeySeed,pt:&Point,x:&[Vec<i64>])->AuxTuple{AuxTuple{c:lin_dense(p,seed,KeyKind::Aux,A_ENC_COLS*p.aux_coeffs,x),t:e_o(&x[..A_ENC_COLS*p.aux_coeffs],&p.degs(),&pt.aux_vars())}}
fn norm(x:&[Vec<i64>])->u128{x.iter().flatten().map(|&v|(v as i128*v as i128) as u128).sum()}
fn cap(p:&Profile,x:&[Vec<i64>],fullcols:usize)->bool{x.iter().flatten().all(|v|v.unsigned_abs()<p.b_z)&&norm(x)<=p.r2(fullcols*N)}
fn gaussian(coins:&Coins,opening:u64,attempt:u16,fs:&FastSigma,base:u64,indices:&[usize])->Result<Vec<Vec<i64>>,AuthError>{indices.par_iter().map(|&c|(0..N).map(|i|{let mut stream=CoordinateStreams{coins,opening,attempt,coordinate:(base+(c*N+i) as u64) as u32};gaussian_fast::sample(fs,&mut stream,&mut Stats::default()).map_err(AuthError::Sampler)}).collect()).collect()}
fn add(beta:&[i8;64],a:&mut[Vec<i64>],get:&dyn Fn(usize)->Vec<i64>){for(c,a)in a.iter_mut().enumerate(){let x=get(c);for (u,v)in a.iter_mut().zip(act_c0_ring_int(beta,&x)){*u+=v;}}}
/// Raw i8 sources are logical-order full domains. Only registered columns read.
pub fn raw_column(raw:&[i8],c:usize)->Vec<i64>{let mut x=vec![0;N];for i in 0..N{x[ring::phys_of_logical(i)]=raw[c*N+i] as i64;}x}
fn pack(p:&Profile,rows:&[Vec<Vec<i64>>])->Vec<u8>{let mut out=Vec::new();for row in rows{for c in row{out.extend(super::wire::pack37(c,super::wire::z_off(p)));}}out}
pub fn prove(p:&Profile,seed:&KeySeed,coins:&Coins,opening:u64,tr:&MTranscript,pt:&Point,s:&Support,raw:&[Vec<i8>],links:&[Vec<Vec<i64>>],aux:&[Vec<i64>])->Result<AuthProof,AuthError>{
    assert!(s.matches(p));assert_eq!(raw.len(),links.len());assert!(raw.iter().all(|r|r.len()==p.source_cols()*N));
    check_raw_support(p,s,raw);
    let fs=FastSigma::new(p.sigma);let(sw,sl,so)=(p.source_cols(),p.link_cols(),p.aux_cols());let stride=((sw+sl+so)*N)as u64;
    for attempt in 0..p.attempts as u16{let(mut uw,mut ul,mut uo)=(vec![],vec![],vec![]);let(mut d,mut d_o)=(vec![],vec![]);
        for row in 0..p.tau{let base=row as u64*stride;let w=gaussian(coins,opening,attempt,&fs,base,&s.columns)?;let l=gaussian(coins,opening,attempt,&fs,base+(sw*N)as u64,&(0..sl).collect::<Vec<_>>())?;let o=gaussian(coins,opening,attempt,&fs,base+((sw+sl)*N)as u64,&(0..so).collect::<Vec<_>>())?;d.push(fx(p,seed,pt,s,&w,&l));d_o.push(fo(p,seed,pt,&o));uw.push(w);ul.push(l);uo.push(o);}
        let nu:u128=uw.iter().chain(&ul).chain(&uo).map(|r|norm(r)).sum();let beta=auth::challenges_pub(tr,&d,&d_o,attempt,raw.len());
        // In-place U→Z after retaining exact U squared norm: no second mask array.
        for row in 0..p.tau{for e in 0..raw.len(){add(&beta[row][e],&mut uw[row],&|i|raw_column(&raw[e],s.columns[i]));add(&beta[row][e],&mut ul[row],&|i|links[e][i].clone());}add(&beta[row][raw.len()],&mut uo[row],&|i|aux[i].clone());}
        if !(0..p.tau).all(|r|cap(p,&uw[r],sw)&&cap(p,&ul[r],sl)&&cap(p,&uo[r],so)){continue}
        let nz:u128=uw.iter().chain(&ul).chain(&uo).map(|r|norm(r)).sum();let x=Ratio{num:BigInt::from(nu as i128)-BigInt::from(nz as i128),den:BigInt::from(2u128*p.sigma as u128*p.sigma as u128)};let mut a=Address::new(Purpose::ProtocolCoin,opening);a.attempt=attempt;let mut st=coins.stream(a);let mut ub=BigUint::from(0u8);for _ in 0..rejection::B{ub=(ub<<1u32)+BigUint::from(st.next_bit().map_err(|_|AuthError::CoinAbort)?);}
        match rejection::decide(&x,&ub){Coin::Accept=>return Ok(AuthProof{attempt,d,d_o,z:Responses{w:pack(p,&uw),l:pack(p,&ul),o:pack(p,&uo)}}),Coin::Reject=>{},Coin::Abort=>return Err(AuthError::CoinAbort)}
    }Err(AuthError::Exhausted)
}
/// Secret-free original authentication equations with virtual zero extension.
pub fn verify(p:&Profile,seed:&KeySeed,tr:&MTranscript,pt:&Point,s:&Support,targets:&[Tuple],target_o:&AuxTuple,pf:&AuthProof)->bool{
    if !s.matches(p)||targets.len()>p.n_max||pt.xi.len()!=p.ell()||!pt.xi.iter().all(super::wire::canonical_k)||!super::wire::canonical_k(&pt.rho)||pt.rho==K8::ZERO||pt.rho==K8::ONE||pf.attempt as usize>=p.attempts||pf.d.len()!=p.tau||pf.d_o.len()!=p.tau{return false}
    if !targets.iter().chain(&pf.d).all(|t|super::wire::commitment_ok(p,&t.c)&&super::wire::commitment_ok(p,&t.g)&&super::wire::endpoint_ok(&t.a))||!std::iter::once(target_o).chain(&pf.d_o).all(|t|super::wire::commitment_ok(p,&t.c)&&super::wire::endpoint_ok(&t.t)){return false}
    let sizes=[s.len(),p.link_cols(),p.aux_cols()];let bytes=[&pf.z.w,&pf.z.l,&pf.z.o];for i in 0..3{if bytes[i].len()!=super::wire::packed_len(p.tau*sizes[i]*N){return false}}
    let beta=auth::challenges_pub(tr,&pf.d,&pf.d_o,pf.attempt,targets.len());
    for row in 0..p.tau{let mut blocks=vec![];for i in 0..3{let n=super::wire::packed_len(sizes[i]*N);let Some(x)=super::wire::unpack37(&bytes[i][row*n..(row+1)*n],sizes[i]*N,super::wire::z_off(p),super::wire::z_umax(p))else{return false};blocks.push(x.chunks(N).map(|c|c.to_vec()).collect::<Vec<_>>());}
        if !cap(p,&blocks[0],p.source_cols())||!cap(p,&blocks[1],p.link_cols())||!cap(p,&blocks[2],p.aux_cols()){return false}
        let lhs=fx(p,seed,pt,s,&blocks[0],&blocks[1]);let mut rhs=pf.d[row].clone();for(e,t)in targets.iter().enumerate(){rhs=auth::add_tuple(&rhs,&auth::act_tuple(&beta[row][e],t));}if lhs!=rhs{return false}
        let lo=fo(p,seed,pt,&blocks[2]);let b=&beta[row][targets.len()];let ro=AuxTuple{c:auth::add_ring(&pf.d_o[row].c,&target_o.c.iter().map(|x|super::pcs_maps::act_c0_ring_q(b,x)).collect::<Vec<_>>()),t:pf.d_o[row].t.add(&super::pcs_maps::act_c0_a(b,&target_o.t))};if lo!=ro{return false}
    }true
}
#[cfg(any(test,feature="cpu-replay"))]
pub mod functional_tests{
use super::*;use super::super::coins::CoinProfile;
#[cfg_attr(test,test)] pub fn original_index_algebra_and_sampler_equivalence(){
 let p=Profile::toy();let seed=KeySeed([71;8]);let indices=vec![0,2,3,4,5,6,7];let mut dense=vec![vec![0i64;N];p.source_cols()];for &c in &indices{for i in 0..N{dense[c][i]=((i*13+c*19)%7)as i64-3;}}
 let get=|c:usize|dense[c].clone();assert_eq!(lin_indexed(&p,&seed,KeyKind::Source,p.source_payload_cols,&indices,&get),auth::lin(&p,&seed,KeyKind::Source,p.source_payload_cols,&dense));
 let xi=vec![K8::ONE.scale(7);p.ell()];assert_eq!(ev_indexed(&xi,&indices,&get),super::super::pcs_maps::ev_raw(&dense,&xi));
 let b=std::array::from_fn(|i|[-2,-1,1,2][i%4]);let shifted:Vec<_>=dense.iter().map(|v|act_c0_ring_int(&b,v)).collect();assert!(shifted[1].iter().all(|&v|v==0));assert_eq!(ev_indexed(&xi,&indices,&|c|shifted[c].clone()),super::super::pcs_maps::act_c0_a(&b,&ev_indexed(&xi,&indices,&get)));
 let coins=Coins::new([37;64],CoinProfile::Ternary);let fs=FastSigma::new(p.sigma);let stride=((p.source_cols()+p.link_cols()+p.aux_cols())*N)as u64;for row in 0..2{let base=row*stride;let samples=gaussian(&coins,29,3,&fs,base,&indices).unwrap();for (k,&c)in indices.iter().enumerate(){for i in [0,1,63,64,4095]{let mut streams=CoordinateStreams{coins:&coins,opening:29,attempt:3,coordinate:(base+(c*N+i)as u64)as u32};assert_eq!(samples[k][i],gaussian_fast::sample(&fs,&mut streams,&mut Stats::default()).unwrap());}}}
}
#[cfg_attr(test,test)] pub fn full_support_proof_matches_dense_and_real_negatives(){
 let p=Profile::toy();let seed=KeySeed([43;8]);let coins=Coins::new([18;64],CoinProfile::Ternary);let s=Support::from_public_columns(&p,(0..p.source_cols()).collect()).unwrap();let raw=vec![(0..p.source_cols()*N).map(|i|(i%3)as i8-1).collect::<Vec<_>>()];let links=vec![vec![vec![0;N];p.link_cols()]];let aux=vec![vec![0;N];p.aux_cols()];let dense=vec![auth::SourceVec{w:(0..p.source_cols()).map(|c|raw_column(&raw[0],c)).collect(),r:links[0].clone()}];let pt=Point{xi:vec![K8::ONE.scale(5);p.ell()],rho:K8::ONE.scale(3)};let tr=MTranscript::new(p.name,&coins.salt(71));let targets=vec![auth::f_x(&p,&seed,&pt,&dense[0].w,&dense[0].r)];let target_o=auth::f_ox(&p,&seed,&pt,&aux);
 let pf=prove(&p,&seed,&coins,71,&tr,&pt,&s,&raw,&links,&aux).unwrap();let legacy=auth::prove(&p,&seed,&coins,71,&tr,&pt,&dense,&aux).unwrap();assert_eq!(pf,legacy);assert!(verify(&p,&seed,&tr,&pt,&s,&targets,&target_o,&pf));assert!(auth::verify(&p,&seed,&tr,&pt,&targets,&target_o,&pf));
 let mut bad=pf.clone();bad.z.w[0]^=1;assert!(!verify(&p,&seed,&tr,&pt,&s,&targets,&target_o,&bad));let mut bad=pf.clone();bad.z.w.pop();assert!(!verify(&p,&seed,&tr,&pt,&s,&targets,&target_o,&bad));let mut targets2=targets.clone();targets2[0].a.0[0]=targets2[0].a.0[0].add(&K8::ONE);assert!(!verify(&p,&seed,&tr,&pt,&s,&targets2,&target_o,&pf));
}
#[cfg_attr(test,test)] pub fn restricted_proof_zero_extension_verifies(){
 let p=Profile::toy();let seed=KeySeed([21;8]);let coins=Coins::new([61;64],CoinProfile::Ternary);
 let support=Support::from_public_columns(&p,vec![0,2,3,4,5,6,7]).unwrap();
 let mut raw=vec![vec![0i8;p.source_cols()*N]];for &c in support.columns(){for i in 0..N{raw[0][c*N+i]=((c+i)%3)as i8-1;}}
 let links=vec![vec![vec![0;N];p.link_cols()]];let aux=vec![vec![0;N];p.aux_cols()];let pt=Point{xi:vec![K8::ONE.scale(11);p.ell()],rho:K8::ONE.scale(7)};let tr=MTranscript::new(p.name,&coins.salt(93));
 let dense:Vec<_>=(0..p.source_cols()).map(|c|raw_column(&raw[0],c)).collect();let targets=vec![auth::f_x(&p,&seed,&pt,&dense,&links[0])];let target_o=auth::f_ox(&p,&seed,&pt,&aux);
 let pf=prove(&p,&seed,&coins,93,&tr,&pt,&support,&raw,&links,&aux).unwrap();assert!(verify(&p,&seed,&tr,&pt,&support,&targets,&target_o,&pf));
 let mut expanded=pf.clone();let compact=super::super::wire::rows_of(&p,&pf.z.w,support.len()).unwrap();let mut rows=vec![vec![vec![0;N];p.source_cols()];p.tau];for r in 0..p.tau{for(i,&c)in support.columns().iter().enumerate(){rows[r][c]=compact[r][i].clone();}}expanded.z.w=pack(&p,&rows);assert_ne!(&expanded.z.w[super::super::wire::packed_len(N)..2*super::super::wire::packed_len(N)],vec![0u8;super::super::wire::packed_len(N)].as_slice());assert!(auth::verify(&p,&seed,&tr,&pt,&targets,&target_o,&expanded));
 let mut bad=pf.clone();bad.z.w[super::super::wire::packed_len(2*N)]^=1;assert!(!verify(&p,&seed,&tr,&pt,&support,&targets,&target_o,&bad));assert!(!auth::verify(&p,&seed,&tr,&pt,&targets,&target_o,&pf));
}

}

/// Check the public submodule premise without deriving support from values.
/// Used before block-link allocation as well as before Gaussian authentication.
pub fn check_raw_support(p:&Profile,s:&Support,raw:&[Vec<i8>]){check_support(p,s,&super::packed::I8s(raw))}
/// Same, over int8 or packed sources.
pub fn check_support(p:&Profile,s:&Support,raw:&dyn super::packed::Sources){
    assert!(s.matches(p));assert!(raw.count()==0||raw.src_len()==p.source_cols()*N);
    use rayon::prelude::*;
    (0..raw.count()).into_par_iter().for_each(|e|{for c in 0..p.source_cols(){if s.columns.binary_search(&c).is_err(){assert!(raw.is_zero(e,c*N,N),"source outside registered support");}}});
}
