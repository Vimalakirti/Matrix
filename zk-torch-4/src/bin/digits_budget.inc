fn mask_budget(d:&Dag,wi:&[Vec<Witness>])->usize {
 use zk_torch_4::basicblock::BasicBlockType as B;
 use zk_torch_4::util::arith::log2_ceil;
 let nv:Vec<usize>=wi.iter().map(|ws|ws[0].shape.iter().map(|&x|{assert!(x>0);log2_ceil(x)}).sum()).collect();
 let mut ids=d.topo.clone();ids.sort();assert_eq!(ids,(0..d.nodes.len()).collect::<Vec<_>>());
 let sc=|n:usize,p:usize|->u128 { (n as u128).checked_mul((p+1) as u128).unwrap().checked_add((3*p) as u128).unwrap() };
 let mut total=0u128;
 for n in &d.nodes {let cost=match &n.kind {
 B::NonNegative(_)=>1,
 B::ProductZeroCheck(_)=>sc(nv[n.inputs[0]],2),
 B::ReLUHelper(_)=>0,
 k=>{let red=match k {B::Add(_)|B::Sub(_)=>2,B::ScaleDown(_)=>1,B::ChangeShape(_)=>0,B::Einsum(_)=>sc(n.inputs.iter().map(|&e|nv[e]).sum(),n.inputs.len()),x=>panic!("unreviewed budget path {x:?}")};red+sc(nv[n.outputs[0]],1)}
 };total=total.checked_add(cost).unwrap();println!("MASK_BOUND node={} kind={:?} slots={cost}",n.id,n.kind);}
 let rounded=usize::try_from(total).unwrap().max(1).next_power_of_two();println!("MASK_BOUND total={total} rounded={rounded} digits={}",rounded*512);rounded.trailing_zeros() as usize
}
