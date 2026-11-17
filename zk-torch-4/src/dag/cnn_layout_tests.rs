//! Real CPU graph execution and independently indexed logical references.
//! Direct range predicates/statement checks are not cryptographic replay.
use super::*;
use crate::basicblock::{BasicBlock, BasicBlockType, ChangeShape};
use crate::util::arith::{f_to_int, int_to_f};

fn input(g: &mut DagBuilder, shape: Vec<usize>, values: Vec<i128>, sf: usize) -> EdgeId {
    g.committed_input(Witness::new(shape, values.into_iter().map(int_to_f).collect(), DataType::Float, sf, Role::Input))
}
fn forward(d: &Dag, w: &mut [Vec<Witness>]) {
    for &n in &d.topo {
        let nd = &d.nodes[n];
        if let BasicBlockType::NonNegative(r) = &nd.kind {
            for value in w[nd.inputs[0]][0].data.as_ref().unwrap().evaluations_ref() {
                let v = f_to_int(*value);
                assert!(v >= 0 && v < 1i128 << r.table_size_log, "range root {n}: {v}");
            }
        }
        let inputs = nd.inputs.iter().map(|&e| &w[e][0]).collect::<Vec<_>>();
        let outputs = nd.kind.run(&inputs);
        for (&e, out) in nd.outputs.iter().zip(outputs) { w[e] = vec![out]; }
    }
}
fn values(w: &Witness) -> Vec<i128> {
    w.data.as_ref().unwrap().evaluations_ref().iter().map(|v| f_to_int(*v)).collect()
}
fn row_major(shape: &[usize], poison: i128) -> (Vec<i128>, Vec<i128>) {
    let count = shape.iter().product::<usize>();
    let padded = shape.iter().map(|s| s.next_power_of_two()).collect::<Vec<_>>();
    let mut data = vec![poison; padded.iter().product()];
    let logical = (0..count).map(|i| i as i128 - (count / 2) as i128).collect::<Vec<_>>();
    for i in 0..count {
        let (mut rem, mut raw, mut stride) = (i, 0, 1);
        for d in (0..shape.len()).rev() {
            raw += (rem % shape[d]) * stride;
            rem /= shape[d]; stride *= padded[d];
        }
        data[raw] = logical[i];
    }
    (data, logical)
}

#[test]
fn flatten_row_major_actual_graph_matches_logical_reference_with_poison() {
    for shape in [vec![16,5,5], vec![3,3,5], vec![2,4,8], vec![3], vec![3,2,3,5]] {
        for poison in [-777777, 888888] {
            let (data, mut expected) = row_major(&shape, poison);
            let mut g = DagBuilder::new();
            let x = input(&mut g, shape.clone(), data, *crate::SF_LOG);
            let y = g.flatten_row_major(x);
            let (d, mut w) = g.compile();
            forward(&d, &mut w);
            expected.resize(expected.len().next_power_of_two(), 0);
            assert_eq!(values(&w[y][0]), expected, "shape={shape:?}");
            assert_eq!(w[y][0].sf, *crate::SF_LOG);
            if shape == [16,5,5] {
                let old = ChangeShape { target_shape: vec![400] }.run(&[&w[x][0]]);
                assert_ne!(values(&old[0]), expected, "old truncation must be caught");
                assert_eq!(old[0].data.as_ref().unwrap().evaluations_ref().len(), 512);
                assert_eq!(d.program_literals.values().map(Vec::len).sum::<usize>(), 16384);
                assert_eq!(d.program_literals.values().flatten().filter(|&&v| v == 1).count(), 1200);
            }
        }
    }
}

#[test]
fn lenet_actual_classifier_ignores_poisoned_weight_bias_and_activation_padding() {
    let sf = *crate::SF_LOG;
    assert!(sf <= 20);
    let q = 1i128 << sf;
    for poison in [-12345, 333333] {
        let mut g = DagBuilder::new();
        let mut expected = (0..400).map(|i| (i as i128 % 23 - 11)*q).collect::<Vec<_>>();
        let mut data = vec![poison; 1024];
        for c in 0..16 { for h in 0..5 { for w in 0..5 {
            data[w+8*(h+8*c)] = expected[w+5*(h+5*c)];
        }}}
        let x = input(&mut g, vec![16,5,5], data, sf);
        let x = g.flatten_row_major(x);
        let mut weights = Vec::new(); let mut biases = Vec::new();
        for (ni, no) in [(400usize,120usize),(120,84),(84,10)] {
            let mut w = vec![poison; ni.next_power_of_two()*no.next_power_of_two()];
            let mut b = vec![poison; no.next_power_of_two()];
            let mut next = vec![0i128; no];
            for o in 0..no {
                for i in 0..ni {
                    let v = if i == (o*7)%ni { q } else if i == (o*7+1)%ni { -q } else { 0 };
                    w[o+no.next_power_of_two()*i] = v;
                    next[o] += expected[i]*v;
                }
                b[o] = (o as i128 % 5 - 2)*q;
                next[o] = (next[o]+q/2).div_euclid(q)+b[o];
            }
            weights.push(Witness::new(vec![no,ni], w.into_iter().map(int_to_f).collect(), DataType::Float, sf, Role::Constant));
            biases.push(Witness::new(vec![no], b.into_iter().map(int_to_f).collect(), DataType::Float, sf, Role::Constant));
            expected = next;
        }
        let y = super::lenet::classifier(&mut g, x, &weights, &biases);
        let (d, mut w) = g.compile(); forward(&d, &mut w);
        expected.resize(16,0);
        assert_eq!(values(&w[y][0]), expected);
        for &edge in &d.layer_boundaries {
            let v = values(&w[edge][0]); let logical = w[edge][0].shape[0];
            assert!(v[logical..].iter().all(|&v| v == 0));
        }
    }
}

#[test]
fn resnet_actual_gap_fc_matches_nonpower_logical_batch_and_bias_reference() {
    let sf = *crate::SF_LOG; assert!(sf <= 20); let q = 1i128 << sf;
    for (batch, channels, h, width) in [(1usize,3usize,3usize,5usize),(3,3,3,5),(5,5,2,3)] {
        for explicit in [true,false] {
            for poison in [-98765,222222] {
                let (cp,bp,hp,wp) = (channels.next_power_of_two(),batch.next_power_of_two(),h.next_power_of_two(),width.next_power_of_two());
                let mut data = vec![poison; bp*cp*hp*wp];
                let mut means = vec![vec![0i128; channels]; batch];
                for b in 0..batch { for c in 0..channels {
                    let mut sum = 0;
                    for ih in 0..h { for iw in 0..width {
                        let v = ((b*7+c*3+iw) as i128-ih as i128-5)*q + (iw%2) as i128;
                        data[iw+wp*(ih+hp*(c+cp*b))] = v; sum += v;
                    }}
                    let n = (h*width) as i128; means[b][c] = (sum+n/2).div_euclid(n);
                }}
                let shape = if explicit {vec![batch,channels,h,width]} else if batch==1 {vec![channels,h,width]} else {vec![bp*cp,h,width]};
                let mut g = DagBuilder::new(); let x = input(&mut g,shape,data,sf);
                let classes = 5usize; let op = classes.next_power_of_two();
                let mut w = vec![poison; cp*op]; let mut bv = vec![poison;op];
                let mut expected = vec![0i128; op*bp];
                for o in 0..classes {
                    bv[o] = (o as i128-2)*q;
                    for c in 0..channels { w[c+cp*o] = ((c+o)%3) as i128*q-q; }
                    for b in 0..batch {
                        let sum = (0..channels).map(|c| means[b][c]*w[c+cp*o]).sum::<i128>();
                        expected[o+op*b] = (sum+q/2).div_euclid(q)+bv[o];
                    }
                }
                let weight = g.param(Witness::new(vec![channels,classes],w.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Constant));
                // Public API documents a vector bias even for batched FC.
                let bias = g.param(Witness::new(vec![classes],bv.into_iter().map(int_to_f).collect(),DataType::Float,sf,Role::Constant));
                let y = super::resnet::classifier_head(&mut g,x,weight,Some(bias),batch,sf);
                let (d,mut w) = g.compile(); forward(&d,&mut w);
                assert_eq!(values(&w[y][0]),expected,"B={batch} C={channels} explicit={explicit}");
                assert_eq!(w[y][0].shape,if batch==1 {vec![classes]} else {vec![classes,batch]});
            }
        }
    }
}

#[test]
fn flatten_maps_are_immutable_public_program_data_not_private_sources() {
    use crate::morpheus::{piop::{Graph,Config},statement,params::Profile,keys::KeySeed};
    let mut statements = Vec::new();
    for poison in [-77,88] {
        let mut g = DagBuilder::new();
        let (v,_) = row_major(&[3,2,3],poison);
        let x = input(&mut g,vec![3,2,3],v,0); g.flatten_row_major(x);
        let (d,mut w) = g.compile(); forward(&d,&mut w);
        let mut graph = Graph::new(&d,&w,Config::default());
        for (&e,v) in &d.program_literals {
            assert_eq!(&**graph.public.get(&e).unwrap(),v);
            assert!(!graph.model[e]);
        }
        statements.push(statement::model_words(&graph,&Profile::toy(),&KeySeed([7;8]),11,false,4).unwrap());
        let (&e,v) = d.program_literals.iter().next().unwrap();
        let mut changed = v.clone(); changed[0] ^= 1;
        graph.public.insert(e,std::sync::Arc::new(changed));
        assert!(statement::words(&graph,&Profile::toy(),&KeySeed([7;8]),11,false,4).is_err());
    }
    assert_eq!(statements[0],statements[1],"private padding cannot change the program");
}

#[test]
fn row_major_channel_bias_distinct_channels_basis_and_poisoned_padding() {
    use crate::basicblock::Add;
    for (batch, spatial, explicit) in [(1usize,vec![3usize,5usize],false),(3,vec![3,5],true),(3,vec![3,5],false),(1,vec![2,3,5],false)] {
        let channels = 3usize; let cp = 4; let bp = batch.next_power_of_two();
        let sp = spatial.iter().map(|s|s.next_power_of_two()).product::<usize>();
        let logical_sp = spatial.iter().product::<usize>();
        let mut shape = if explicit {vec![batch,channels]} else if batch==1 {vec![channels]} else {vec![batch*cp]};
        shape.extend(&spatial);
        for bias_form in 0..3 {
            for poison in [-123456,777777] {
                let mut data = vec![poison;bp*cp*sp];
                let mut expected = vec![0;data.len()];
                let mut bias_data = if bias_form<2 {vec![poison;cp]} else {vec![poison;cp*sp]};
                for c in 0..channels { if bias_form<2 {bias_data[c] = 101*(c as i128+1);} }
                for b in 0..batch { for c in 0..channels { for pos in 0..logical_sp {
                    let (mut rem,mut raw,mut stride)=(pos,0,1);
                    for d in (0..spatial.len()).rev() {raw+=(rem%spatial[d])*stride;rem/=spatial[d];stride*=spatial[d].next_power_of_two();}
                    let i = raw+sp*(c+cp*b);
                    let base = if c==1 && pos==logical_sp/2 {1000*(b as i128+1)} else {0};
                    data[i] = base;
                    let v = 101*(c as i128+1)+if bias_form==2 {pos as i128} else {0};
                    expected[i] = base+v;
                    if bias_form==2 {bias_data[raw+sp*c]=v;}
                }}}
                let bias_shape = match bias_form {0=>vec![channels],1=>{let mut s=vec![channels];s.extend(vec![1;spatial.len()]);s},_=>{let mut s=vec![channels];s.extend(&spatial);s}};
                let mut g=DagBuilder::new();let x=input(&mut g,shape.clone(),data,0);
                let bias=g.param(Witness::new(bias_shape,bias_data.into_iter().map(int_to_f).collect(),DataType::Float,0,Role::Constant));
                let y=g.add_channel_bias_row_major(x,bias,channels,spatial.len());
                let (d,mut w)=g.compile();forward(&d,&mut w);
                assert_eq!(values(&w[y][0]),expected,"B={batch} spatial={spatial:?} explicit={explicit} form={bias_form}");
                if batch==1 && spatial.len()==2 && bias_form==1 {
                    let old=Add.run(&[&w[x][0],&w[bias][0]]);
                    assert_ne!(values(&old[0]),expected,"old column-major broadcast must fail");
                }
            }
        }
    }
}

#[test]
fn resnet_head_rejects_same_padded_size_but_wrong_logical_extents() {
    for (shape,features,batch) in [(vec![3usize,3,2,3],4usize,3usize),(vec![3,3,2,3],3,4),(vec![3,2,3],4,1),(vec![8,2,3],3,3)] {
        assert!(std::panic::catch_unwind(|| {
            let mut g=DagBuilder::new();let x=g.input(shape,DataType::Float);
            let w=g.param(Witness::new(vec![features,5],vec![int_to_f(0);features.next_power_of_two()*8],DataType::Float,0,Role::Constant));
            super::resnet::classifier_head(&mut g,x,w,None,batch,0);
        }).is_err());
    }
}

#[test]
fn cnn_public_literal_source_census_representative_graphs() {
    use crate::morpheus::{piop::{Graph,Config,layout::{TKey,Layout}},params::Profile};
    fn census(label:&str,g:DagBuilder) {
        let(d,mut w)=g.compile();forward(&d,&mut w);
        let graph=Graph::new(&d,&w,Config{tcl:4,slots_log:17,req_planes_log:5});
        let mut specs=Vec::new();let mut slots=0usize;let mut coords=0usize;
        for (&e,values) in &d.program_literals {
            assert!(graph.committed[e] && graph.public.contains_key(&e) && !graph.model[e]);
            let max_bits=values.iter().map(|&v| {let p=almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME;let a=if v>p/2 {p-v}else{v};(64-a.leading_zeros() as usize).max(1)}).max().unwrap();
            let planes=max_bits.next_power_of_two();let bits=values.iter().all(|&v|v<=1);
            let raw=values.len()*planes;slots+=values.len();coords+=raw;
            specs.push((TKey::Edge(e),graph.nvars[e],planes.trailing_zeros() as usize,bits));
            println!("CENSUS {label} edge={e} shape={:?} field_slots={} planes={planes} bit_typed={bits} raw_coordinates={raw}",w[e][0].shape,values.len());
        }
        let profile=Profile::ternary();let layout=Layout::pack(&profile,&specs);
        let packed=layout.tensors.iter().flat_map(|t|&t.pieces).map(|p|1usize<<(p.region.m+p.region.p)).sum::<usize>();
        assert_eq!(packed,coords);
        println!("CENSUS_TOTAL {label} literals={} field_slots={slots} raw_coordinates={coords} public_only_sources={} source_payload_cap={} n_max={} energy_cap_log2={} whole_workload_compatibility=UNVERIFIED",d.program_literals.len(),layout.n_sources,profile.source_payload_cols*crate::morpheus::ring::N,profile.n_max,profile.s_max_log2);
    }
    let sf=*crate::SF_LOG;
    let zero=|shape:Vec<usize>| {let n=shape.iter().map(|s|s.next_power_of_two()).product();Witness::new(shape,vec![int_to_f(0);n],DataType::Float,sf,Role::Constant)};
    let mut g=DagBuilder::new();let x=input(&mut g,vec![1,32,32],vec![0;1024],sf);
    let cw=vec![zero(vec![6,1,5,5]),zero(vec![16,6,5,5])];
    let cb=vec![zero(vec![6,28,28]),zero(vec![16,10,10])];
    let fw=vec![zero(vec![120,400]),zero(vec![84,120]),zero(vec![10,84])];let fb=vec![zero(vec![120]),zero(vec![84]),zero(vec![10])];
    g.pipe(&[x],super::lenet::lenet5(cw,cb,fw,fb));census("LeNet5_full_graph",g);
    let mut g=DagBuilder::new();let x=input(&mut g,vec![3,2048,7,7],vec![0;4*2048*8*8],sf);
    let w=g.param(zero(vec![2048,1000]));let b=g.param(zero(vec![1000]));
    super::resnet::classifier_head(&mut g,x,w,Some(b),3,sf);census("ResNet_head_B3_C2048_7x7_O1000",g);
}

#[test]
fn explicit_nonpower_batch_survives_pad_and_pool_registered_views() {
    let b=3usize;let c=3usize;let (h,w)=(4usize,6usize);let (cp,bp,hp,wp)=(4,4,4,8);
    for poison in [1i128<<40,-(1i128<<40)] {
        let mut raw=vec![poison;bp*cp*hp*wp];
        for ib in 0..b {for ic in 0..c {for ih in 0..h {for iw in 0..w {
            raw[iw+wp*(ih+hp*(ic+cp*ib))]=(ib*100+ic*10+ih*2+iw) as i128;
        }}}}
        for general in [false,true] {
            let mut g=DagBuilder::new();let x=input(&mut g,vec![b,c,h,w],raw.clone(),0);
            let padded=g.pad(x,1,1);
            assert_eq!(g.init_values[padded].as_ref().unwrap().shape,vec![b,c,6,8]);
            let y=if general {g.maxpool_general(padded,2,2,2,2)}else{g.maxpool2d(padded,2,2)};
            assert_eq!(g.init_values[y].as_ref().unwrap().shape,vec![b,c,3,4]);
            let (d,mut wit)=g.compile();forward(&d,&mut wit);
            assert_eq!(wit[y][0].shape,vec![b,c,3,4]);
            let got=values(&wit[y][0]);let mut expected=vec![0;bp*cp*4*4];
            for ib in 0..b {for ic in 0..c {for oh in 0..3 {for ow in 0..4 {
                let mut maximum=i128::MIN;
                for dh in 0..2 {for dw in 0..2 {
                    let ih=(oh*2+dh) as isize-1;let iw=(ow*2+dw) as isize-1;
                    let v=if ih>=0 && ih<h as isize && iw>=0 && iw<w as isize {(ib*100+ic*10+ih as usize*2+iw as usize) as i128}else{0};
                    maximum=maximum.max(v);
                }}
                expected[ow+4*(oh+4*(ic+cp*ib))]=maximum;
            }}}}
            assert_eq!(got,expected,"general={general}");
        }
    }
}

#[test]
fn explicit_batch_pad_conv_bias_pool_chain_matches_logical_reference() {
    let (batch,ci,co,h,w)=(3usize,3usize,5usize,4usize,6usize);
    for mode in 0..3 {
        let (kernel,stride,dilation)=match mode {0=>(1usize,1usize,1usize),1=>(1,2,1),_=>(2,1,2)};
        let (ph,pw)=(h+2,w+2);let oh=(ph-dilation*(kernel-1)-1)/stride+1;let ow=(pw-dilation*(kernel-1)-1)/stride+1;
        let khp=kernel.next_power_of_two();let mut raw=vec![int_to_f(1i128<<40);4*4*4*8];
        for b in 0..batch {for c in 0..ci {for ih in 0..h {for iw in 0..w {raw[iw+8*(ih+4*(c+4*b))]=int_to_f((b*40+c*10+ih*2+iw)as i128);}}}}
        let mut weights=vec![int_to_f(-55555);8*4*khp*khp];
        for d in 0..co {for c in 0..ci {for kh in 0..kernel {for kw in 0..kernel {
            weights[kw+khp*(kh+khp*(c+4*d))]=int_to_f(if c==d%ci {(1+kh+kw)as i128}else{0});
        }}}}
        let mut g=DagBuilder::new();let x=g.committed_input(Witness::new(vec![batch,ci,h,w],raw,DataType::Float,0,Role::Input));
        let weight=g.param(Witness::new(vec![co,ci,kernel,kernel],weights,DataType::Float,0,Role::Constant));
        let padded=g.pad(x,1,1);
        let convolved=match mode {0=>g.conv2d(padded,weight,(kernel,kernel))[0],1=>g.conv2d_strided(padded,weight,(kernel,kernel),(stride,stride))[0],_=>g.conv2d_dilated(padded,weight,(kernel,kernel),(stride,stride),(dilation,dilation))[0]};
        assert_eq!(g.init_values[convolved].as_ref().unwrap().shape,vec![batch,co,oh,ow]);
        let mut bias_values=vec![int_to_f(777);8];for c in 0..co {bias_values[c]=int_to_f((c+1)as i128*11);}
        let bias=g.param(Witness::new(vec![co,1,1],bias_values,DataType::Float,0,Role::Constant));
        let biased=g.add_channel_bias_row_major(convolved,bias,co,2);
        let y=g.maxpool_general(biased,2,2,1,1);
        let (d,mut wit)=g.compile();forward(&d,&mut wit);
        assert!(d.nodes.iter().any(|n|matches!(&n.kind,BasicBlockType::Conv2D(c) if c.batch==batch)));
        let out_h=oh-1;let out_w=ow-1;let hp=out_h.next_power_of_two();let wp=out_w.next_power_of_two();
        let mut expected=vec![0;4*8*hp*wp];
        let cv=|b:usize,d:usize,ho:usize,wo:usize| {
            let mut total=(d+1)as i128*11;let c=d%ci;
            for kh in 0..kernel {for kw in 0..kernel {
                let ih=(ho*stride+kh*dilation)as isize-1;let iw=(wo*stride+kw*dilation)as isize-1;
                if ih>=0 && ih<h as isize && iw>=0 && iw<w as isize {total+=(b*40+c*10+ih as usize*2+iw as usize)as i128*(1+kh+kw)as i128;}
            }} total
        };
        for b in 0..batch {for c in 0..co {for ih in 0..out_h {for iw in 0..out_w {
            let mut maximum=i128::MIN;for dh in 0..2 {for dw in 0..2 {maximum=maximum.max(cv(b,c,ih+dh,iw+dw));}}
            expected[iw+wp*(ih+hp*(c+8*b))]=maximum;
        }}}}
        assert_eq!(wit[y][0].shape,vec![batch,co,out_h,out_w]);assert_eq!(values(&wit[y][0]),expected,"mode={mode}");
    }
}
