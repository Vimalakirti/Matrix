//! Exact private grid routing baseline using only existing authenticated arithmetic.
//! This favors a simple complete relation over sparse performance: no routing
//! hint, coordinate, or private-derived map is exposed as public program data.
use crate::basicblock::{BasicBlockType,ProductZeroCheck};
use crate::dag::{DagBuilder,DataType,EdgeId,Role,Witness};
use crate::util::arith::int_to_f;
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
fn product(b:&mut DagBuilder,a:EdgeId,c:EdgeId)->EdgeId {
    let k=b.init_values[a].as_ref().unwrap().shape.len();
    let s:String=(0..k).map(|i|(b'a'+i as u8) as char).collect();
    b.einsum(format!("{s},{s}->{s}"),vec![a,c],false)[0]
}
fn literal(b:&mut DagBuilder,shape:Vec<usize>,values:Vec<i128>)->EdgeId {
    b.public_literal(Witness::new(shape,values.into_iter().map(int_to_f).collect(),DataType::Uint,0,Role::Input))
}
fn zero(b:&mut DagBuilder,x:EdgeId) {
    let shape=b.init_values[x].as_ref().unwrap().shape.clone();
    let one=b.integer_literal(shape.clone(),1,0);
    let e=b.add_gkr_node(vec![x,one],BasicBlockType::ProductZeroCheck(ProductZeroCheck))[0];
    b.init_values.push(Some(Witness::new_wo_data(shape,DataType::Uint,0,Role::Output)));
    b.self_claim_edges.push(e);
}
pub(crate) fn geometry(n:usize,c:usize,h:usize,w:usize) -> (usize,usize,usize,usize) {
    assert_eq!(usize::BITS,64,"grid routing matches the existing 64-bit centered-to-usize ABI");
    assert!(n>0 && c>0 && h>0 && w>0,"grid routing requires positive public extents");
    assert!((h as u128)<Q as u128/2 && (w as u128)<Q as u128/2,
        "grid extents must fit the positive centered coordinate domain");
    let p=(n.checked_next_power_of_two().unwrap(),c.checked_next_power_of_two().unwrap(),
        h.checked_next_power_of_two().unwrap(),w.checked_next_power_of_two().unwrap());
    p.0.checked_mul(p.2).and_then(|v|v.checked_mul(p.3)).and_then(|v|v.checked_mul(p.1))
        .expect("grid routing tensor size overflow");
    // Conservative declared witness-cell estimate, checked before constructing
    // any equality matrix or private-route graph. This is a resource limit,
    // never a coordinate-value bound or an unchecked fallback.
    let estimate=(p.0 as u128)*(p.2 as u128+p.3 as u128)*320
        +(n as u128)*((p.0 as u128)+(p.2 as u128)*(p.3 as u128)*(p.1 as u128+8)*4)
        +16*(p.0 as u128)*(p.1 as u128)+16*(p.2 as u128)*(p.3 as u128)*(p.1 as u128);
    let budget=std::env::var("MORPHEUS_GRID_ROUTING_MAX_CELLS").ok()
        .map(|s|s.parse::<u128>().expect("invalid grid routing budget"))
        .unwrap_or(1u128<<26);
    assert!(estimate<=budget,"exact private grid routing needs at most {estimate} witness cells, budget {budget}; explicitly raise MORPHEUS_GRID_ROUTING_MAX_CELLS only with adequate resources; no unconstrained fallback");
    p
}
/// E[p,j] = [centered(coords[p,axis])=j] on logical p,j; zero on padding.
/// Fermat gives exact equality over F_q, including all signed invalid values.
/// The deterministic exponent is fixed public metadata, not witness advice.
fn axis_equal(b:&mut DagBuilder,coords:EdgeId,n:usize,len:usize,axis:usize)->EdgeId {
    assert_eq!(b.init_values[coords].as_ref().unwrap().shape,vec![n,2]);
    assert_eq!(b.init_values[coords].as_ref().unwrap().sf,0,"grid coordinates must have integer scale zero");
    let (np,lp)=(n.next_power_of_two(),len.next_power_of_two());
    let cv=b.change_shape(coords,vec![2,np]);
    let basis=literal(b,vec![2],if axis==0 {vec![1,0]} else {vec![0,1]});
    let cv=b.einsum("ap,a->p".into(),vec![cv,basis],false)[0];
    let ones=b.integer_literal(vec![lp],1,0);
    let expanded=b.einsum("p,l->pl".into(),vec![cv,ones],false)[0];
    let indices=literal(b,vec![np,lp],(0..np*lp).map(|i|(i/np) as i128).collect());
    let base=b.sub(expanded,indices)[0];
    let mut acc=b.integer_literal(vec![np,lp],1,0);let mut power=base;let mut e=Q-1;
    while e>0 {if e&1==1 {acc=product(b,acc,power);} e>>=1;if e>0 {power=product(b,power,power);}}
    let one=b.integer_literal(vec![np,lp],1,0);let eq=b.sub(one,acc)[0];
    let support=literal(b,vec![np,lp],(0..np*lp).map(|i|i128::from(i%np<n && i/np<len)).collect());
    product(b,eq,support)
}
pub(crate) fn attach_gather(b:&mut DagBuilder,grid:EdgeId,coords:EdgeId,y:EdgeId,n:usize,c:usize,h:usize,w:usize) {
    let (np,cp,hp,wp)=geometry(n,c,h,w);
    assert_eq!(b.init_values[grid].as_ref().unwrap().shape,vec![c,h,w]);
    assert_eq!(b.init_values[y].as_ref().unwrap().shape,vec![n,c]);
    assert_eq!(b.init_values[grid].as_ref().unwrap().sf,b.init_values[y].as_ref().unwrap().sf);
    let ey=axis_equal(b,coords,n,h,0);let ex=axis_equal(b,coords,n,w,1);
    let grid=b.change_shape(grid,vec![wp,hp,cp]);
    let expected=b.einsum("ph,pw,whc->cp".into(),vec![ey,ex,grid],false)[0];
    let mask=literal(b,vec![cp,np],(0..cp*np).map(|i|i128::from(i%cp<c && i/cp<n)).collect());
    let expected=product(b,expected,mask);let y=b.change_shape(y,vec![cp,np]);
    let error=b.sub(y,expected)[0];zero(b,error);
}
pub(crate) fn attach_scatter(b:&mut DagBuilder,x:EdgeId,coords:EdgeId,y:EdgeId,n:usize,c:usize,h:usize,w:usize) {
    let (np,cp,hp,wp)=geometry(n,c,h,w);
    assert_eq!(b.init_values[x].as_ref().unwrap().shape,vec![n,c]);
    assert_eq!(b.init_values[y].as_ref().unwrap().shape,vec![c,h,w]);
    let sf=b.init_values[x].as_ref().unwrap().sf;assert_eq!(sf,b.init_values[y].as_ref().unwrap().sf);
    let ey=axis_equal(b,coords,n,h,0);let ex=axis_equal(b,coords,n,w,1);
    let x=b.change_shape(x,vec![cp,np]);
    let mut survival=literal(b,vec![wp,hp],(0..wp*hp).map(|i|i128::from(i%wp<w && i/wp<h)).collect());
    let channels=literal(b,vec![cp],(0..cp).map(|i|i128::from(i<c)).collect());
    let mut expected=b.integer_literal(vec![wp,hp,cp],0,sf);
    for p in (0..n).rev() {
        let mut select=vec![0;np];select[p]=1;let select=literal(b,vec![np],select);
        let hy=b.einsum("ph,p->h".into(),vec![ey,select],false)[0];
        let hx=b.einsum("pw,p->w".into(),vec![ex,select],false)[0];
        let hit=b.einsum("w,h->wh".into(),vec![hx,hy],false)[0];
        let live=product(b,hit,survival);
        let value=b.einsum("cp,p->c".into(),vec![x,select],false)[0];let value=product(b,value,channels);
        let term=b.einsum("wh,c->whc".into(),vec![live,value],false)[0];
        expected=b.add(expected,term)[0];
        let one=b.integer_literal(vec![wp,hp],1,0);let miss=b.sub(one,hit)[0];survival=product(b,survival,miss);
    }
    let y=b.change_shape(y,vec![wp,hp,cp]);let error=b.sub(y,expected)[0];zero(b,error);
}
