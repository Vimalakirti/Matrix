fn s(n:usize,p:usize,h:bool)->u128 { n as u128*(p+h as usize).max(1) as u128+(2*p).max(1) as u128+p as u128 }
fn map(n:usize)->u128{s(n,1,true)}
fn mask_budget(d:&Dag,w:&[Vec<Witness>])->usize {
 use zk_torch_4::{basicblock::BasicBlockType as B,util::arith::log2_ceil};
 let label="CNN";let tcl=4;
 assert!(tcl>0);
 let nv:Vec<usize>=w.iter().map(|x|x[0].shape.iter().map(|&v|log2_ceil(v)).sum()).collect();
 let m=*nv.iter().max().unwrap_or(&0);
 let mut total=0u128;let mut roots=0;let mut merges=0;let mut reductions=0;
 for nd in &d.nodes {
  let ni=|j:usize|nv[nd.inputs[j]];
  let (root,red,merge,kind):(u128,u128,u128,&str)=match &nd.kind{
   B::NonNegative(_)=>(1,0,0,"nonnegative_root"),
   B::ProductZeroCheck(_)=>(s(ni(0),2,true),0,0,"pzc_root"),
   B::MaskedMaxPoolHelper(_)|B::MaxPoolSelector(_)|B::MaskedMaxSelector(_)|B::NormalizedRatio(_)|B::DivConst(_)|B::SoftmaxConst(_)|B::SigmoidConst(_)|B::RMSReciprocal(_)|B::RMSBucket(_)|B::ReLUHelper(_)|B::MaxPoolHelper(_)|B::GeneralMaxPoolHelper(_)|B::PillarMaxPool(_)|B::InstanceNorm3D(_)|B::ScatterToBEV(_)|B::GatherFromGrid(_)=>(0,0,0,"advice"),
   k=>{let red=match k{
    B::Add(_)|B::Sub(_)|B::Concat(_)=>2,
    B::ChangeShape(_)=>0,
    B::ScaleDown(_)|B::ScaleUp(_)|B::ExpHelper(_)=>1,
    B::TwoPow(_)=>{let en=d.producers[nd.inputs[0]].unwrap();let bits=match &d.nodes[en].kind{B::ExpHelper(e)=>e.num_bits,_=>panic!("TwoPow producer")};let p=bits.div_ceil(tcl).max(1);if p==1{0}else{s(nv[nd.outputs[0]],p,true)}},
    B::Einsum(_)=>s(nd.inputs.iter().map(|&e|nv[e]).sum(),nd.inputs.len(),true),
    B::ZeroPad(_)|B::ZeroPadAsym(_)|B::ZeroPad3D(_)|B::SubSample2D(_)|B::Replicate2x2(_)|B::ChannelSlice(_)|B::FlattenKernel(_)|B::FlattenKernel3D(_)=>map(ni(0)),
    // All conv alternatives: n1 <= sum input axis logs + kernel geometry logs.
    B::Conv1D(c)=>s(log2_ceil(c.kernel_size)+log2_ceil(c.c_in),2,true)+2*map(m),
    B::Conv2D(c)=>{let ordinary=s(log2_ceil(c.kernel_w)+log2_ceil(c.kernel_h)+log2_ceil(c.c_in),2,true)+2*map(m);let pp=11*m as u128+19; ordinary.max(pp)},
    B::Conv3D(c)=>s(log2_ceil(c.kernel_w)+log2_ceil(c.kernel_h)+log2_ceil(c.kernel_d)+log2_ceil(c.c_in),2,true)+2*map(m),
    B::DepthwiseConv2D(c)=>s(log2_ceil(c.kernel_w)+log2_ceil(c.kernel_h)+log2_ceil(c.channels),2,true)+2*map(m),
    B::ConvTranspose2D(c)=>s(log2_ceil(c.kernel_w)+log2_ceil(c.kernel_h)+log2_ceil(c.c_in),2,true)+2*map(m),
    B::ConvTranspose3D(c)=>s(log2_ceil(c.kernel_w)+log2_ceil(c.kernel_h)+log2_ceil(c.kernel_d)+log2_ceil(c.c_in),2,true)+2*map(m),
    other=>panic!("unsupported budget kind {other:?}"),
   };(0,red,map(nv[nd.outputs[0]]),"compute")}
  };
  println!("NODE label={label} id={} kind={kind} root={root} reduction={red} merge={merge}",nd.id);
  roots+=root;reductions+=red;merges+=merge;total+=root+red+merge;
 }
 let cap=total.max(1).next_power_of_two();
 println!("BUDGET label={label} sf={} nodes={} edges={} max_nvars={m} roots={roots} reductions={reductions} merges={merges} upper={total} rounded_slots={cap} rounded_digits={} default_slots=131072 default_digits=67108864 within_default={} within_cap_2pow20={}",*zk_torch_4::SF_LOG,d.nodes.len(),d.num_edges(),cap*512,total<=131072,total<=1048576);
 usize::try_from(cap).unwrap().trailing_zeros() as usize
}
