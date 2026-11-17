//! Authenticated exact window-maximum relation. Advice supplies only a private
//! Boolean selection. All geometry, support and extraction coefficients are
//! public registered program data; no host assertion substitutes for a root.
use crate::basicblock::{BasicBlockType,GeneralMaxPoolHelper,MaxPoolSelector,ProductZeroCheck,SubSample2D};
use crate::dag::{DagBuilder,DataType,EdgeId,Role,Witness};
use crate::util::arith::int_to_f;

/// Signed integer profile, independent of the request source-plane profile.
/// Full-domain completeness additionally needs req_planes_log=6 (64 planes).
/// Smaller registered source widths support only their representable subset.
pub const VALUE_BOUND:i128=(1i128<<54)-1;
fn zero(b:&mut DagBuilder,x:EdgeId) {
    let shape=b.init_values[x].as_ref().unwrap().shape.clone();
    let one=b.integer_literal(shape.clone(),1,0);
    let out=b.add_gkr_node(vec![x,one],BasicBlockType::ProductZeroCheck(ProductZeroCheck))[0];
    b.init_values.push(Some(Witness::new_wo_data(shape,DataType::Uint,0,Role::Output)));
    b.self_claim_edges.push(out);
}
fn product(b:&mut DagBuilder,a:EdgeId,c:EdgeId)->EdgeId {
    let ndim=b.init_values[a].as_ref().unwrap().shape.len();
    let s:String=(0..ndim).map(|i|(b'a'+i as u8)as char).collect();
    b.einsum(format!("{s},{s}->{s}"),vec![a,c],false)[0]
}
fn validate(b:&DagBuilder,x:EdgeId,y:EdgeId,p:&GeneralMaxPoolHelper)->(Vec<usize>,usize,usize) {
    let shape=p.output_shape();
    assert_eq!(b.init_values[x].as_ref().unwrap().shape,vec![p.channels,p.input_h,p.input_w]);
    assert_eq!(b.init_values[y].as_ref().unwrap().shape,shape);
    assert_eq!(b.init_values[x].as_ref().unwrap().sf,b.init_values[y].as_ref().unwrap().sf);
    let rows=shape.iter().map(|n|n.next_power_of_two()).product::<usize>();
    let cols=(p.kernel_h*p.kernel_w).next_power_of_two();
    rows.checked_mul(cols).expect("maxpool selection size overflow");
    // Cardinality cannot wrap in the field. This is public geometry validation.
    assert!((cols as u128)<almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as u128);
    (shape,rows,cols)
}
pub(crate) fn window_support(b:&mut DagBuilder,p:&GeneralMaxPoolHelper,
    valid:impl Fn(usize,usize,usize,usize)->bool)->EdgeId {
    let shape=p.output_shape();let (hp,wp)=(shape[1].next_power_of_two(),shape[2].next_power_of_two());
    let rows=shape[0].next_power_of_two()*hp*wp;let cols=(p.kernel_h*p.kernel_w).next_power_of_two();
    let mut mask=vec![int_to_f(0);rows.checked_mul(cols).expect("maxpool support size overflow")];
    for c in 0..shape[0] {for oh in 0..shape[1] {for ow in 0..shape[2] {for kh in 0..p.kernel_h {for kw in 0..p.kernel_w {
        if valid(oh,ow,kh,kw) {mask[ow+wp*(oh+hp*c)+rows*(kh*p.kernel_w+kw)]=int_to_f(1);}
    }}}}}
    b.public_literal(Witness::new(vec![rows,cols],mask,DataType::Bool,0,Role::Input))
}
pub(crate) fn attach_selector(b:&mut DagBuilder,x:EdgeId,y:EdgeId,p:GeneralMaxPoolHelper,support:Option<EdgeId>) {
    let (_,rows,cols)=validate(b,x,y,&p);
    let support=support.unwrap_or_else(||window_support(b,&p,|_,_,_,_|true));
    b.add_gkr_node(vec![x,y,support],BasicBlockType::MaxPoolSelector(MaxPoolSelector{pool:p}));
    b.init_values.push(Some(Witness::new_wo_data(vec![rows,cols],DataType::Bool,0,Role::Auxiliary)));
}
pub(crate) fn selector_constraints(b:&mut DagBuilder,x:EdgeId,y:EdgeId,h:EdgeId,support:EdgeId,p:GeneralMaxPoolHelper) {
    let (shape,rows,cols)=validate(b,x,y,&p);
    assert_eq!(b.init_values[h].as_ref().unwrap().shape,vec![rows,cols]);
    assert_eq!(b.init_values[h].as_ref().unwrap().sf,0);
    assert_eq!(b.init_values[support].as_ref().unwrap().shape,vec![rows,cols]);
    assert_eq!(b.init_values[support].as_ref().unwrap().sf,0);
    let masks=b.program_literals.get(&support).expect("maxpool support must be immutable public program data");
    assert_eq!(masks.len(),rows*cols);assert!(masks.iter().all(|&v|v<=1));
    let (hp,wp)=(shape[1].next_power_of_two(),shape[2].next_power_of_two());
    let mut active=vec![0i128;rows];
    for row in 0..rows {
        let logical=row%wp<shape[2] && row/wp%hp<shape[1] && row/(wp*hp)<shape[0];
        let mut any=false;
        for col in 0..cols {if masks[row+rows*col]==1 {
            assert!(logical && col<p.kernel_h*p.kernel_w,"maxpool support selects storage padding");any=true;
        }}
        assert_eq!(any,logical,"maxpool logical window must have a valid member");
        active[row]=i128::from(logical);
    }
    let one=b.integer_literal(vec![rows,cols],1,0);
    let complement=b.sub(one,h)[0];let nonbit=product(b,h,complement);zero(b,nonbit);
    let forbidden=b.sub(one,support)[0];let excluded=product(b,h,forbidden);zero(b,excluded);
    let count=b.einsum("ab->a".into(),vec![h],false)[0];
    let active=b.public_literal(Witness::new(vec![rows],active.into_iter().map(int_to_f).collect(),DataType::Bool,0,Role::Input));
    let count_error=b.sub(count,active)[0];zero(b,count_error);
    let yv=b.change_shape(y,vec![rows]);
    let sf=b.init_values[x].as_ref().unwrap().sf;
    let mut selected=b.integer_literal(vec![rows],0,sf);
    for kh in 0..p.kernel_h {for kw in 0..p.kernel_w {
        let sample=b.add_gkr_node(vec![x],BasicBlockType::SubSample2D(SubSample2D::new_with_output_size(
            p.channels,p.input_h,p.input_w,p.stride_h,p.stride_w,kh,kw,shape[1],shape[2])))[0];
        b.init_values.push(Some(Witness::new_wo_data(shape.clone(),DataType::Uint,sf,Role::Output)));
        let sample=b.change_shape(sample,vec![rows]);
        let mut basis=vec![int_to_f(0);cols];basis[kh*p.kernel_w+kw]=int_to_f(1);
        let basis=b.public_literal(Witness::new(vec![cols],basis,DataType::Bool,0,Role::Input));
        let ak=b.einsum("ab,b->a".into(),vec![support,basis],false)[0];
        // Excluded boundary values (including poisoned zero-pad witnesses) are
        // masked BEFORE any integer-domain or order predicate is applied.
        let sample=product(b,sample,ak);
        // Paired roots are <2^55 with authenticated sum 2B<2^55. Their entire
        // sum envelope is <2^56<<p, so even malicious witnesses cannot wrap.
        b.integer_interval(sample,-VALUE_BOUND,VALUE_BOUND);
        let diff=b.sub(yv,sample)[0];let diff=product(b,diff,ak);
        // One-hot attainment bounds y by B. Thus each live y-x is in [-2B,2B],
        // and p-2B>2^55-1 excludes every negative lift from this range root.
        b.add_nonneg_node_bits(diff,55);
        let hk=b.einsum("ab,b->a".into(),vec![h,basis],false)[0];
        let value=product(b,hk,sample);
        selected=b.add(selected,value)[0];
    }}
    // Exactly one selector is 1 on a live row; every selector is 0 on padding.
    // The product sum hence has one term (or zero), so no large-sum wrap exists.
    // This also fixes y=0 on all output storage padding, not just honest runs.
    let attainment=b.sub(yv,selected)[0];zero(b,attainment);
}
