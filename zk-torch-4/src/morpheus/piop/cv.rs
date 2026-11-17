//! Convolution-family and spatial index-map reductions of the masked graph
//! PIOP (vision and audio front ends).
//!
//! Layout. These blocks store [C, (D,) H, W] tensors padded per axis with
//! the LAST axis in the lowest index bits (`x[c,h,w]` at `w + Wp(h + Hp c)`).
//!
//! Index maps (ZeroPad*, SubSample2D, Replicate2x2, ChannelSlice) read
//! Y[.., o_d, ..] = X[.., i_d, ..] along mapped axes d, with a relation
//! R_d ∋ (o, i). A claim Ỹ(r) = z becomes the masked sumcheck
//!   z = Σ_i X̃(r_fixed, i) · Π_d T_d(i_d),  T_d(i) = Σ_{(o,i)∈R_d} eq(r_o, o),
//! over the mapped input bits. T_d is public: the verifier builds it from
//! R_d (|R_d| ≤ the axis length) and evaluates its MLE.
//!
//! Convolutions (Conv1D/2D/3D, DepthwiseConv2D, ConvTranspose2D/3D) relate
//! output o, kernel tap k and input i per spatial axis by R_d ∋ (o, k, i)
//! (i = s·o + δ·k, or o = s·i + k when transposed). Two masked sumchecks:
//!   1. z = Σ_{k,c} W̃(.., k, c) · A(k, c),  A(k, c) = Σ_o Π_d eq(r_o, o_d) X(c, i(o, k)),
//!      over the kernel taps and input channels (an eq factor for the shared
//!      channel of a depthwise conv);
//!   2. the claim on A at (x_k, x_c) is the index map above with
//!      T_d(i) = Σ_{(o,k,i)∈R_d} eq(r_o, o)·eq(x_k, k).
//! A weight producer is bypassed only if its public scatter geometry matches
//! the consumer and every skipped reshape preserves padded arity. Otherwise
//! a public tap map binds the actual flat edge, including nonidentity views.

use super::*;

fn lb(n: usize) -> usize { log2_ceil(n.max(1)) }

/// MORPHEUS_CONV=pp: prove Conv2D with the polynomial-product construction
/// and a committed full convolution (ablation baseline).

pub(super) fn conv_pp() -> bool { std::env::var("MORPHEUS_CONV").ok().as_deref() == Some("pp") }

/// One mapped axis: relation triples (o, k, i) and bit widths.
#[derive(Clone, Debug)]
pub(super) struct Rel { pub ob: usize, pub kb: usize, pub ib: usize, pub t: Vec<(usize, usize, usize)> }

impl Rel {
    fn map(ob: usize, ib: usize, pairs: impl Iterator<Item = (usize, usize)>) -> Self {
        Rel { ob, kb: 0, ib, t: pairs.map(|(o, i)| (o, 0, i)).collect() }
    }
    /// T(i) = Σ eq(r_o, o)·eq(r_k, k) over the triples, dense over i.
    pub(super) fn table(&self, ro: &[K8], rk: &[K8]) -> Vec<K8> {
        let (eo, ek) = (eq_table(ro), eq_table(rk));
        let mut t = vec![K8::ZERO; 1usize << self.ib];
        for &(o, k, i) in &self.t { t[i] = t[i].add(&eo[o].mul(&ek[k])); }
        t
    }
    /// Sparse kernel-contraction matrix M[k][i] = Σ_o eq(r_o, o).
    fn kmat(&self, ro: &[K8]) -> Vec<Vec<(usize, K8)>> {
        let eo = eq_table(ro);
        let mut m: Vec<HashMap<usize, K8>> = vec![HashMap::new(); 1usize << self.kb];
        for &(o, k, i) in &self.t { let e = m[k].entry(i).or_insert(K8::ZERO); *e = e.add(&eo[o]); }
        m.into_iter().map(|h| { let mut v: Vec<(usize, K8)> = h.into_iter().collect(); v.sort_by_key(|x| x.0); v }).collect()
    }
}

/// A mapped axis of the input: its first variable and relation, with the
/// output point r_o (and kernel point r_k).
pub(super) struct MapAxis { pub off: usize, pub rel: Rel, pub ro: Vec<K8>, pub rk: Vec<K8> }

/// MLE of the indicator [c < n] over Boolean c (lowest bit first) at x.
fn prefix_mle(n: usize, x: &[K8]) -> K8 {
    if n >= 1usize << x.len() { return K8::ONE; }
    // Σ_{c<n} eq(x, c): walk the bits of n from the top.
    let (mut acc, mut pre) = (K8::ZERO, K8::ONE);
    for b in (0..x.len()).rev() {
        let one = x[b];
        let zero = K8::ONE.sub(&one);
        if (n >> b) & 1 == 1 {
            // c agrees with n above b and has 0 at b: every lower suffix counts.
            acc = acc.add(&pre.mul(&zero));
            pre = pre.mul(&one);
        } else {
            pre = pre.mul(&zero);
        }
    }
    acc
}

/// MLE at y of the table c ↦ eq(r, c)·[c < n]: Σ_{c<n} eq(r, c)·eq(y, c).
fn masked_eq(r: &[K8], y: &[K8], n: usize) -> K8 {
    if n >= 1usize << r.len() { return eq_pt(r, y); }
    let (er, ey) = (eq_table(r), eq_table(y));
    (0..n).fold(K8::ZERO, |a, c| a.add(&er[c].mul(&ey[c])))
}

/// Contract one logical axis, ignoring adversarial values in padded slots.
/// Remaining variables retain their original low-to-high order.
pub(super) fn masked_axis_fold(v: &[u64], n: usize, off: usize, r: &[K8], logical: usize) -> Vec<K8> {
    assert_eq!(v.len(), 1usize << n);
    assert!(off + r.len() <= n && logical > 0 && logical <= 1usize << r.len());
    if logical == 1usize << r.len() {
        return partial_eval(v, n, &(0..r.len()).map(|j| (off + j, r[j])).collect::<Vec<_>>());
    }
    let e = eq_table(r);
    let low_mask = (1usize << off) - 1;
    (0..1usize << (n - r.len())).into_par_iter().map(|i| {
        let base = (i & low_mask) | ((i >> off) << (off + r.len()));
        (0..logical).fold(K8::ZERO, |a, c| a.add(&e[c].scale(v[base | (c << off)])))
    }).collect()
}

/// Actual normal-reduction patch table, isolated for direct CPU identity tests.
pub(super) fn conv_patch_table(xv: &[u64], rels: &[Rel], ro: &[Vec<K8>], lci: usize, batch: usize, rb: &[K8]) -> Vec<K8> {
    let si: usize = rels.iter().map(|d| d.ib).sum();
    let lbatch = lb(batch);
    let mut sizes: Vec<usize> = rels.iter().map(|d| d.ib).collect();
    sizes.push(lci);
    let mut cur = if lbatch == 0 { None } else {
        Some(masked_axis_fold(xv, si + lci + lbatch, si + lci, &rb, batch))
    };
    for (a, d) in rels.iter().enumerate() {
        let m = d.kmat(&ro[a]);
        let pre: usize = 1usize << sizes[..a].iter().sum::<usize>();
        let post: usize = 1usize << sizes[a + 1..].iter().sum::<usize>();
        let (ni, nk) = (1usize << d.ib, 1usize << d.kb);
        let out: Vec<K8> = (0..post * nk).into_par_iter().flat_map_iter(|pk| {
            let (ph, k) = (pk / nk, pk % nk);
            let row = &m[k];
            let cur = &cur;
            (0..pre).map(move |pl| {
                let mut acc = K8::ZERO;
                for &(i, coef) in row {
                    let idx = (ph * ni + i) * pre + pl;
                    let v = match cur { Some(t) => t[idx].mul(&coef), None => { let u = xv[idx]; if u == 0 { continue } else { coef.scale(u) } } };
                    acc = acc.add(&v);
                }
                acc
            })
        }).collect();
        sizes[a] = d.kb;
        cur = Some(out);
    }
    cur.expect("conv with no spatial axis")
}

/// Actual PP exponent masks. Exponents are nonnegative integer indices;
/// alpha=0 is supported without inverse or modular exponent-index wrapping.
pub(super) fn conv_pp_powers(cv: &crate::basicblock::conv::Conv2D, alpha: K8) -> (Vec<K8>, Vec<K8>, Vec<K8>) {
    let (lw, lf, lk) = (lb(cv.input_w), lb(cv.s_in + cv.s_kernel - 1), lb(cv.s_kernel));
    let pows = |n: usize| -> Vec<K8> {
        let mut v = Vec::with_capacity(n); let mut a = K8::ONE;
        for _ in 0..n { v.push(a); a = a.mul(&alpha); } v
    };
    let am = pows(1usize << lf);
    // No inverse: alpha=0 is a well-defined evaluation too.
    let ap: Vec<K8> = (0..cv.s_in).map(|p| {
        if (p & ((1 << lw) - 1)) < cv.input_w && (p >> lw) < cv.input_h { am[cv.s_in - 1 - p] } else { K8::ZERO }
    }).collect();
    let mut aj = vec![K8::ZERO; 1usize << lk];
    for h in 0..cv.kernel_h { for w in 0..cv.kernel_w {
        let j = h * cv.dilation_h * cv.stride_w + w * cv.dilation_w;
        aj[j] = am[j];
    } }
    (am, ap, aj)
}

/// Build the actual flatten producer map. Scatter collisions are rejected:
/// CPU last-write and GPU racing writes cannot define one portable relation.
pub(super) fn flatten_kernel_rel(fk: &crate::basicblock::conv::FlattenKernel, lk: usize) -> Rel {
    let (lkw, lkh) = (lb(fk.kw), lb(fk.kh));
    assert!([fk.kh, fk.kw, fk.c_in, fk.c_out, fk.s_w, fk.dilation_h, fk.dilation_w].iter().all(|&v| v > 0), "positive FlattenKernel geometry");
    let pairs: Vec<_> = (0..fk.kh).flat_map(|h| (0..fk.kw).map(move |w|
        (h.checked_mul(fk.dilation_h).and_then(|v| v.checked_mul(fk.s_w))
            .and_then(|v| w.checked_mul(fk.dilation_w).and_then(|u| v.checked_add(u)))
            .expect("FlattenKernel tap index overflow"), w + (h << lkw)))).collect();
    let mut seen = std::collections::HashSet::new();
    assert!(pairs.iter().all(|&(j, _)| j < 1usize << lk && seen.insert(j)),
        "FlattenKernel tap scatter must be in bounds and injective");
    Rel::map(lk, lkw + lkh, pairs.into_iter())
}

/// Conv kinds by weight layout.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum WKind { Normal, Transpose, Depthwise }

/// Public tap-index layout, lowest spatial axis first. Channel counts are in
/// storage order [outer, inner, flat taps], NOT necessarily [C_out, C_in].
struct KernelGeometry { dims: Vec<usize>, strides: Vec<usize>, inner: usize, outer: usize, extent: usize }

fn padded(n: usize) -> usize { n.checked_next_power_of_two().expect("spatial extent overflow") }

/// Validate the descriptor against the same canonical indexing used by the
/// CPU/GPU forward blocks. These four descriptors have no batch coordinate.
fn family_geometry(kind: &BasicBlockType) -> KernelGeometry {
    use crate::basicblock::conv::*;
    let (ins, dims, steps, outs, transposed, inner, outer, declared_strides, extent, s_in) = match kind {
        BasicBlockType::Conv3D(Conv3D { c_in,c_out,kernel_d,kernel_h,kernel_w,input_d,input_h,input_w,
            stride_h,stride_w,s_in,s_kernel,d_out,h_out,w_out,conv_stride_d,conv_stride_h,conv_stride_w }) =>
            (vec![*input_w,*input_h,*input_d],vec![*kernel_w,*kernel_h,*kernel_d],vec![*conv_stride_w,*conv_stride_h,*conv_stride_d],
             vec![*w_out,*h_out,*d_out],false,*c_in,*c_out,vec![1,*stride_w,*stride_h],*s_kernel,Some(*s_in)),
        BasicBlockType::DepthwiseConv2D(DepthwiseConv2D { channels,kernel_h,kernel_w,input_h,input_w,stride_w,s_in,s_kernel,h_out,w_out,conv_stride_h,conv_stride_w }) =>
            (vec![*input_w,*input_h],vec![*kernel_w,*kernel_h],vec![*conv_stride_w,*conv_stride_h],
             vec![*w_out,*h_out],false,1,*channels,vec![1,*stride_w],*s_kernel,Some(*s_in)),
        BasicBlockType::ConvTranspose2D(ConvTranspose2D { c_in,c_out,kernel_h,kernel_w,input_h,input_w,stride_h,stride_w,flat_stride,s_kernel,h_out,w_out }) =>
            (vec![*input_w,*input_h],vec![*kernel_w,*kernel_h],vec![*stride_w,*stride_h],
             vec![*w_out,*h_out],true,*c_out,*c_in,vec![1,*flat_stride],*s_kernel,None),
        BasicBlockType::ConvTranspose3D(ConvTranspose3D { c_in,c_out,kernel_d,kernel_h,kernel_w,input_d,input_h,input_w,
            stride_d,stride_h,stride_w,flat_stride_h,flat_stride_w,s_kernel,d_out,h_out,w_out }) =>
            (vec![*input_w,*input_h,*input_d],vec![*kernel_w,*kernel_h,*kernel_d],vec![*stride_w,*stride_h,*stride_d],
             vec![*w_out,*h_out,*d_out],true,*c_out,*c_in,vec![1,*flat_stride_w,*flat_stride_h],*s_kernel,None),
        _ => panic!("not an unbatched flat-kernel convolution family"),
    };
    assert!(inner > 0 && outer > 0 && ins.iter().chain(&dims).chain(&steps).all(|&v| v > 0), "positive convolution geometry");
    for a in 0..ins.len() {
        let out = if transposed { (ins[a]-1).checked_mul(steps[a]).and_then(|v| v.checked_add(dims[a])).expect("transposed output overflow") }
            else { assert!(dims[a] <= ins[a], "convolution kernel must fit input"); (ins[a]-dims[a])/steps[a]+1 };
        assert_eq!(outs[a], out, "convolution output geometry");
    }
    let grid = if transposed { &outs } else { &ins };
    let mut stride = 1usize;
    let mut strides = Vec::new();
    for &n in grid { strides.push(stride); stride = stride.checked_mul(padded(n)).expect("spatial volume overflow"); }
    assert_eq!(strides, declared_strides, "convolution flat spatial strides");
    if let Some(s) = s_in { assert_eq!(s, stride, "convolution padded input volume"); }
    let want_extent = dims.iter().zip(&strides).try_fold(1usize, |a,(&k,&s)| (k-1).checked_mul(s).and_then(|v| a.checked_add(v))).expect("kernel extent overflow");
    assert_eq!(extent,want_extent,"convolution flat kernel extent");
    padded(extent);
    KernelGeometry { dims,strides,inner,outer,extent }
}

/// Exact public gather k -> j. Bounds and injectivity reject ambiguous
/// producer scatters; no finite-word wrapping is accepted as geometry.
fn kernel_tap_rel(dims: &[usize], strides: &[usize], lk: usize) -> Rel {
    assert_eq!(dims.len(),strides.len());
    assert!(dims.iter().chain(strides).all(|&v| v > 0),"positive kernel scatter geometry");
    let bits: Vec<_> = dims.iter().map(|&d| lb(d)).collect();
    let sk: usize = bits.iter().sum();
    let size = 1usize.checked_shl(lk.try_into().expect("kernel bit width")).expect("kernel flat arity overflow");
    let mut pairs = vec![(0usize,0usize)];
    let mut off = 0;
    for ((&d,&stride),&b) in dims.iter().zip(strides).zip(&bits) {
        let mut next = Vec::new();
        for &(k,j) in &pairs { for tap in 0..d {
            let k = tap.checked_shl(off).and_then(|v| k.checked_add(v)).expect("kernel tap arity overflow");
            let j = tap.checked_mul(stride).and_then(|v| j.checked_add(v)).expect("kernel tap index overflow");
            next.push((k,j));
        } }
        pairs = next; off += b as u32;
    }
    let mut seen = std::collections::HashSet::new();
    assert!(pairs.iter().all(|&(_,j)| j < size && seen.insert(j)),"kernel tap scatter must be in bounds and injective");
    Rel::map(sk,lk,pairs.into_iter())
}

pub(super) fn flatten_kernel3d_rel(fk: &crate::basicblock::conv::FlattenKernel3D, lk: usize) -> Rel {
    assert!(fk.c_in > 0 && fk.c_out > 0,"positive FlattenKernel3D channels");
    let r = kernel_tap_rel(&[fk.kw,fk.kh,fk.kd],&[1,fk.stride_w,fk.stride_h],lk);
    Rel::map(r.ib,r.ob,r.t.into_iter().map(|(k,_,j)|(j,k)))
}

/// Return the raw kernel only when the whole skipped path is an identity on
/// the taps used by this consumer. A truncation followed by regrowth remains
/// bound to the actual final edge; matching endpoint arity is insufficient.
pub(super) fn family_kernel(g: &Graph, edge: usize, kind: &BasicBlockType) -> (usize, Option<Rel>) {
    let spec = family_geometry(kind);
    let lk = lb(spec.extent);
    assert_eq!(g.nvars[edge],lk+lb(spec.inner)+lb(spec.outer),"flat kernel extent/channel layout");
    let tap = kernel_tap_rel(&spec.dims,&spec.strides,lk);
    let mut e = edge;
    loop {
        match g.dag.producers[e].map(|p| &g.dag.nodes[p]) {
            Some(n) if matches!(n.kind,BasicBlockType::ChangeShape(_)) && g.nvars[n.inputs[0]] == g.nvars[e] => e=n.inputs[0],
            Some(n) => {
                let matches = match &n.kind {
                    BasicBlockType::FlattenKernel(f) => spec.dims == [f.kw,f.kh] && spec.strides == [f.dilation_w,f.dilation_h.checked_mul(f.s_w).expect("flatten stride overflow")]
                        && (f.c_in,f.c_out) == (spec.inner,spec.outer),
                    BasicBlockType::FlattenKernel3D(f) => spec.dims == [f.kw,f.kh,f.kd] && spec.strides == [1,f.stride_w,f.stride_h]
                        && (f.c_in,f.c_out) == (spec.inner,spec.outer),
                    _ => false,
                };
                if matches {
                    assert_eq!(g.nvars[n.inputs[0]],tap.ob+lb(spec.inner)+lb(spec.outer),"unflattened source kernel shape");
                    return (n.inputs[0],None);
                }
                break;
            }
            None => break,
        }
    }
    (edge,Some(tap))
}

/// Channel-axis offsets in the actual weight storage, including depthwise's
/// single shared channel axis and transposed [C_in,C_out,taps] ordering.
pub(super) fn conv_weight_offsets(kind: WKind, taps: usize, lci: usize, lco: usize) -> (usize,usize,usize) {
    match kind {
        WKind::Normal => (taps,taps+lci,taps+lci+lco),
        WKind::Transpose => (taps+lco,taps,taps+lci+lco),
        WKind::Depthwise => (taps,taps,taps+lci),
    }
}

/// The production stage-1 kernel table, also exercised by CPU identity tests.
pub(super) fn conv_weight_table(wv: &[u64], nw: usize, sk: usize, lci: usize, kind: WKind, cout_off: usize, rd: &[K8], channels_out: usize, flat: Option<&Rel>) -> Vec<K8> {
    let folded = if kind == WKind::Depthwise { partial_eval(wv,nw,&[]) }
        else { masked_axis_fold(wv,nw,cout_off,rd,channels_out) };
    if let Some(rel) = flat {
        let mut tab = vec![K8::ZERO;1usize << (sk+lci)];
        for cc in 0..1usize << lci { for &(k,_,j) in &rel.t { tab[k+(cc<<sk)]=folded[j+(cc<<rel.ib)]; } }
        tab
    } else { folded }
}

impl<'g, 'a> Walk<'g, 'a> {
    /// Index-map reduction of a claim (z0, η0) on X̃(fixed, mapped) where
    /// `fixed` assigns every unmapped variable of X.
    pub(super) fn map_reduce(&mut self, ctx: &mut Ctx, x: usize, mut fixed: Vec<(usize, K8)>, axes: Vec<MapAxis>, z0: K8, eta0: &Lin) {
        // Trivial axes become fixed variables: an identity over the whole
        // padded range (T = eq(r_o, ·)), or a map onto one input index i0
        // (T = c·[i = i0], the scale c folded into the claim).
        let mut scale = K8::ONE;
        let mut rest = Vec::new();
        for a in axes {
            let full_id = a.rel.kb == 0 && a.rel.ob == a.rel.ib && a.rel.t.len() == 1usize << a.rel.ib
                && a.rel.t.iter().all(|&(o, _, i)| o == i) && { let mut seen = vec![false; 1usize << a.rel.ib]; a.rel.t.iter().all(|&(_, _, i)| !std::mem::replace(&mut seen[i], true)) };
            let i0 = a.rel.t.first().map(|t| t.2);
            if full_id {
                for b in 0..a.rel.ib { fixed.push((a.off + b, a.ro[b])); }
            } else if i0.is_some() && a.rel.t.iter().all(|t| t.2 == i0.unwrap()) && a.rel.table(&a.ro, &a.rk)[i0.unwrap()] != K8::ZERO {
                // (A zero scale cannot be divided out; such an axis takes the
                // sumcheck path below.)
                let c = a.rel.table(&a.ro, &a.rk)[i0.unwrap()];
                scale = scale.mul(&c);
                for b in 0..a.rel.ib { fixed.push((a.off + b, if (i0.unwrap() >> b) & 1 == 1 { K8::ONE } else { K8::ZERO })); }
            } else {
                rest.push(a);
            }
        }
        let mut axes = rest;
        let inv = scale.inv();
        let (z0, eta0) = (z0.mul(&inv), lin_scaled(eta0, &inv));
        let nx = self.g.nvars[x];
        if axes.is_empty() {
            assert_eq!(fixed.len(), nx, "index map must assign every variable of edge {x}");
            let mut point = vec![K8::ZERO; nx];
            for &(v, c) in &fixed { point[v] = c; }
            self.push(ctx, x, AClaim { point, pad: eta0, z: z0 });
            return;
        }
        let eta0 = &eta0;
        axes.sort_by_key(|a| a.off);
        let n: usize = axes.iter().map(|a| a.rel.ib).sum();
        assert_eq!(n + fixed.len(), nx, "index map must assign every variable of edge {x}");
        let tabs: Vec<Vec<K8>> = axes.iter().map(|a| a.rel.table(&a.ro, &a.rk)).collect();
        let bits: Vec<usize> = axes.iter().map(|a| a.rel.ib).collect();
        let tables = if ctx.prover {
            // Public factor Π_d T_d over the mapped bits (earlier axes low).
            let offs: Vec<usize> = bits.iter().scan(0, |acc, &b| { let o = *acc; *acc += b; Some(o) }).collect();
            let p: Vec<K8> = (0..1usize << n).into_par_iter().with_min_len(1 << 12).map(|x| {
                tabs.iter().zip(&offs).zip(&bits).fold(K8::ONE, |acc, ((t, &o), &b)| acc.mul(&t[(x >> o) & ((1usize << b) - 1)]))
            }).collect();
            let pd = self.pd.unwrap();
            backend_tables!(gpu::enabled() && (1usize << n) >= gpu::threshold(), {
                let xp = gpu::partial_eval(&pd.dev(x), nx, &fixed);
                Some(Tables::Dev(vec![gpu::KTab::from_host(&p), xp]))
            }, {
                Some(Tables::Host(vec![p, partial_eval(pd.v(x), nx, &fixed)]))
            })
        } else { None };
        if ctx.prover && selfcheck() {
            if let Some(Tables::Host(t)) = tables.as_ref() {
                let sum = t[0].iter().zip(&t[1]).fold(K8::ZERO, |a, (p, q)| a.add(&p.mul(q)));
                let want = z0.sub(&ctx.lin_val(eta0));
                eprintln!("[selfcheck] map_reduce edge {x}: n {n} fixed {} identity {}", fixed.len(), if sum == want { "ok" } else { "FALSE" });
            }
        }
        let out = ctx.masked_sc(n, 1, true, z0, eta0, tables, &|xs: &[K8]| {
            let mut off = 0;
            tabs.iter().zip(&bits).fold(K8::ONE, |acc, (t, &b)| { let v = tensor::mle_k(t, &xs[off..off + b]); off += b; acc.mul(&v) })
        });
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        let mut point = vec![K8::ZERO; nx];
        for &(v, c) in &fixed { point[v] = c; }
        let mut j = 0;
        for a in &axes { for b in 0..a.rel.ib { point[a.off + b] = out.x[j]; j += 1; } }
        self.push(ctx, x, AClaim { point, pad: lin1(out.pads[0], rr), z: out.v[0] });
    }

    /// The channel axis of an index map: fixed to the output point when the
    /// channel count is a power of two, else an identity map over the real
    /// channels (padded input channels need not be zero).
    fn chan_axis(channels: usize, xoff: usize, nc: usize, rc: &[K8], fixed: &mut Vec<(usize, K8)>, axes: &mut Vec<MapAxis>) {
        assert_eq!(rc.len(), nc);
        if channels.is_power_of_two() || nc == 0 {
            for j in 0..nc { fixed.push((xoff + j, rc[j])); }
        } else {
            axes.push(MapAxis { off: xoff, rel: Rel::map(nc, nc, (0..channels).map(|c| (c, c))), ro: rc.to_vec(), rk: vec![] });
        }
    }

    pub(super) fn reduce_cv(&mut self, ctx: &mut Ctx, node: usize, c: AClaim) -> Option<&'static str> {
        let nd = self.g.dag.nodes[node].clone();
        let (x, y) = (nd.inputs[0], nd.outputs[0]);
        let r = &c.point;
        match &nd.kind {
            BasicBlockType::ZeroPad(p) => {
                let (hp, wp) = (p.input_h + 2 * p.pad_h, p.input_w + 2 * p.pad_w);
                self.pad_like(ctx, x, y, p.channels, &[p.input_h, p.input_w], &[hp, wp], &[p.pad_h, p.pad_w], c);
                Some("pad")
            }
            BasicBlockType::ZeroPadAsym(p) => {
                let (hp, wp) = (p.input_h + p.pad_h_top + p.pad_h_bottom, p.input_w + p.pad_w_left + p.pad_w_right);
                self.pad_like(ctx, x, y, p.channels, &[p.input_h, p.input_w], &[hp, wp], &[p.pad_h_top, p.pad_w_left], c);
                Some("pad")
            }
            BasicBlockType::ZeroPad3D(p) => {
                let o = [p.input_d + 2 * p.pad_d, p.input_h + 2 * p.pad_h, p.input_w + 2 * p.pad_w];
                self.pad_like(ctx, x, y, p.channels, &[p.input_d, p.input_h, p.input_w], &o, &[p.pad_d, p.pad_h, p.pad_w], c);
                Some("pad")
            }
            BasicBlockType::SubSample2D(s) => {
                let (lwo, lho) = (lb(s.out_w), lb(s.out_h));
                let (lw, lh) = (lb(s.input_w), lb(s.input_h));
                let nc = self.g.nvars[x] - lw - lh;
                assert_eq!(self.g.nvars[y], lwo + lho + nc, "SubSample2D channel bits");
                let rw = Rel::map(lwo, lw, (0..s.out_w).map(|o| (o, o * s.stride_w + s.offset_w)).filter(|&(_, i)| i < s.input_w));
                let rh = Rel::map(lho, lh, (0..s.out_h).map(|o| (o, o * s.stride_h + s.offset_h)).filter(|&(_, i)| i < s.input_h));
                let mut axes = vec![MapAxis { off: 0, rel: rw, ro: r[..lwo].to_vec(), rk: vec![] }, MapAxis { off: lw, rel: rh, ro: r[lwo..lwo + lho].to_vec(), rk: vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(s.channels, lw + lh, nc, &r[lwo + lho..], &mut fixed, &mut axes);
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("subsample")
            }
            BasicBlockType::Replicate2x2(rp) => {
                let (lwo, lho) = (lb(rp.out_w), lb(rp.out_h));
                let (iw, ih) = (rp.out_w / 2, rp.out_h / 2);
                let (lw, lh) = (lb(iw), lb(ih));
                let nc = self.g.nvars[x] - lw - lh;
                assert_eq!(self.g.nvars[y], lwo + lho + nc, "Replicate2x2 channel bits");
                let rw = Rel::map(lwo, lw, (0..rp.out_w).map(|o| (o, o / 2)).filter(|&(_, i)| i < iw));
                let rh = Rel::map(lho, lh, (0..rp.out_h).map(|o| (o, o / 2)).filter(|&(_, i)| i < ih));
                let mut axes = vec![MapAxis { off: 0, rel: rw, ro: r[..lwo].to_vec(), rk: vec![] }, MapAxis { off: lw, rel: rh, ro: r[lwo..lwo + lho].to_vec(), rk: vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(rp.channels, lw + lh, nc, &r[lwo + lho..], &mut fixed, &mut axes);
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("replicate")
            }
            BasicBlockType::ChannelSlice(cs) => {
                let s: usize = cs.spatial_dims.iter().map(|&d| lb(d)).sum();
                let (lci, lco) = (lb(cs.channels_in), lb(cs.channels_out));
                assert_eq!(self.g.nvars[x], s + lci);
                assert_eq!(self.g.nvars[y], s + lco);
                let fixed: Vec<(usize, K8)> = (0..s).map(|j| (j, r[j])).collect();
                let rel = Rel::map(lco, lci, (0..cs.channels_out).map(|o| (o, o + cs.channel_start)));
                self.map_reduce(ctx, x, fixed, vec![MapAxis { off: s, rel, ro: r[s..].to_vec(), rk: vec![] }], c.z, &c.pad);
                Some("slice")
            }
            BasicBlockType::Concat(cc) => {
                let b = nd.inputs[1];
                assert!(cc.channels_a.is_power_of_two(), "Concat needs a power-of-two first operand");
                let na = self.g.nvars[x];
                assert_eq!(self.g.nvars[b], na);
                assert_eq!(self.g.nvars[y], na + 1);
                let t = r[na];
                let p = r[..na].to_vec();
                let (sa, va) = self.child_lin(ctx, x, &p);
                let (sb, vb) = self.child_lin(ctx, b, &p);
                // z0 − (1−t) v_a − t v_b = (η0 − (1−t) a_a − t a_b)(W_G).
                let omt = K8::ONE.sub(&t);
                let mut l = c.pad.clone();
                lin_add(&mut l, &lin1(sa, K8::ONE), &omt.neg());
                lin_add(&mut l, &lin1(sb, K8::ONE), &t.neg());
                ctx.row(vec![], &l, c.z.sub(&omt.mul(&va)).sub(&t.mul(&vb)));
                self.push(ctx, x, AClaim { point: p.clone(), pad: lin1(sa, K8::ONE), z: va });
                self.push(ctx, b, AClaim { point: p, pad: lin1(sb, K8::ONE), z: vb });
                Some("concat")
            }
            BasicBlockType::Conv1D(cv) => {
                let w = nd.inputs[1];
                let rel = |_| Rel { ob: lb(cv.l_out), kb: lb(cv.kernel_size), ib: lb(cv.input_len), t: trip_conv(cv.l_out, cv.kernel_size, cv.input_len, cv.conv_stride, 1) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_size], vec![rel(0)], WKind::Normal, lb(cv.c_in), lb(cv.c_out), 1, None, c);
                Some("conv")
            }
            BasicBlockType::Conv2D(cv) if self.g.protocol_conv_pp && nd.outputs.len() > 1 => {
                validate_conv2d_geometry(cv);
                self.conv2d_pp(ctx, &nd, cv, c);
                Some("conv")
            }
            BasicBlockType::FlattenKernel(fk) => {
                let (lkw, lkh, lci, lco) = (lb(fk.kw), lb(fk.kh), lb(fk.c_in), lb(fk.c_out));
                let lk = self.g.nvars[y] - lci - lco;
                assert_eq!(self.g.nvars[x], lkw + lkh + lci + lco, "shared kernel shape");
                let mut axes = vec![MapAxis { off: 0, rel: flatten_kernel_rel(fk, lk), ro: r[..lk].to_vec(), rk: vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(fk.c_in, lkw + lkh, lci, &r[lk..lk + lci], &mut fixed, &mut axes);
                Self::chan_axis(fk.c_out, lkw + lkh + lci, lco, &r[lk + lci..], &mut fixed, &mut axes);
                self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
                Some("flatten")
            }
            BasicBlockType::FlattenKernel3D(fk) => {
                let (sk,lci,lco) = (lb(fk.kw)+lb(fk.kh)+lb(fk.kd),lb(fk.c_in),lb(fk.c_out));
                let lk = self.g.nvars[y].checked_sub(lci+lco).expect("FlattenKernel3D channel bits");
                assert_eq!(self.g.nvars[x],sk+lci+lco,"FlattenKernel3D source shape");
                let mut axes = vec![MapAxis { off:0,rel:flatten_kernel3d_rel(fk,lk),ro:r[..lk].to_vec(),rk:vec![] }];
                let mut fixed = Vec::new();
                Self::chan_axis(fk.c_in,sk,lci,&r[lk..lk+lci],&mut fixed,&mut axes);
                Self::chan_axis(fk.c_out,sk+lci,lco,&r[lk+lci..],&mut fixed,&mut axes);
                self.map_reduce(ctx,x,fixed,axes,c.z,&c.pad);
                Some("flatten")
            }
            BasicBlockType::Conv2D(cv) => {
                validate_conv2d_geometry(cv);
                let (w, flat_kernel) = self.conv2d_kernel(nd.inputs[1], cv);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, cv.dilation_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, cv.dilation_h) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Normal, lb(cv.c_in), lb(cv.c_out), cv.batch, flat_kernel, c);
                Some("conv")
            }
            BasicBlockType::Conv3D(cv) => {
                let (w,flat_kernel) = family_kernel(self.g,nd.inputs[1],&nd.kind);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, 1) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, 1) };
                let rd = Rel { ob: lb(cv.d_out), kb: lb(cv.kernel_d), ib: lb(cv.input_d), t: trip_conv(cv.d_out, cv.kernel_d, cv.input_d, cv.conv_stride_d, 1) };
                self.conv(ctx, x, w, &[cv.c_out, cv.c_in, cv.kernel_d, cv.kernel_h, cv.kernel_w], vec![rw, rh, rd], WKind::Normal, lb(cv.c_in), lb(cv.c_out), 1, flat_kernel, c);
                Some("conv")
            }
            BasicBlockType::DepthwiseConv2D(cv) => {
                let (w,flat_kernel) = family_kernel(self.g,nd.inputs[1],&nd.kind);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_conv(cv.w_out, cv.kernel_w, cv.input_w, cv.conv_stride_w, 1) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_conv(cv.h_out, cv.kernel_h, cv.input_h, cv.conv_stride_h, 1) };
                self.conv(ctx, x, w, &[cv.channels, 1, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Depthwise, lb(cv.channels), lb(cv.channels), 1, flat_kernel, c);
                Some("conv")
            }
            BasicBlockType::ConvTranspose2D(cv) => {
                let (w,flat_kernel) = family_kernel(self.g,nd.inputs[1],&nd.kind);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_tconv(cv.w_out, cv.kernel_w, cv.input_w, cv.stride_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_tconv(cv.h_out, cv.kernel_h, cv.input_h, cv.stride_h) };
                self.conv(ctx, x, w, &[cv.c_in, cv.c_out, cv.kernel_h, cv.kernel_w], vec![rw, rh], WKind::Transpose, lb(cv.c_in), lb(cv.c_out), 1, flat_kernel, c);
                Some("conv")
            }
            BasicBlockType::ConvTranspose3D(cv) => {
                let (w,flat_kernel) = family_kernel(self.g,nd.inputs[1],&nd.kind);
                let rw = Rel { ob: lb(cv.w_out), kb: lb(cv.kernel_w), ib: lb(cv.input_w), t: trip_tconv(cv.w_out, cv.kernel_w, cv.input_w, cv.stride_w) };
                let rh = Rel { ob: lb(cv.h_out), kb: lb(cv.kernel_h), ib: lb(cv.input_h), t: trip_tconv(cv.h_out, cv.kernel_h, cv.input_h, cv.stride_h) };
                let rd = Rel { ob: lb(cv.d_out), kb: lb(cv.kernel_d), ib: lb(cv.input_d), t: trip_tconv(cv.d_out, cv.kernel_d, cv.input_d, cv.stride_d) };
                self.conv(ctx, x, w, &[cv.c_in, cv.c_out, cv.kernel_d, cv.kernel_h, cv.kernel_w], vec![rw, rh, rd], WKind::Transpose, lb(cv.c_in), lb(cv.c_out), 1, flat_kernel, c);
                Some("conv")
            }
            _ => None,
        }
    }

    /// Polynomial-product convolution with a shared (batch-independent) kernel.
    /// For every logical b,d, F[b,d,m] is the coefficient of T^m in
    /// Σ_c (Σ_p X[b,c,p] T^(S-1-p)) (Σ_j W_flat[d,c,j] T^j).
    /// The public masks exclude padded b,c,d,p and non-tap j. Sampling an
    /// independent batch point binds every batch slice, not a sum of slices.
    fn conv2d_pp(&mut self, ctx: &mut Ctx, nd: &crate::dag::Node, cv: &crate::basicblock::conv::Conv2D, c: AClaim) {
        let (x, wf, yf) = (nd.inputs[0], nd.inputs[1], nd.outputs[1]);
        assert!(cv.batch > 0, "conv logical batch must be positive");
        let (lw, lh, lci, lco, lbatch) = (lb(cv.input_w), lb(cv.input_h), lb(cv.c_in), lb(cv.c_out), lb(cv.batch));
        let (lwo, lho) = (lb(cv.w_out), lb(cv.h_out));
        let l_si = lw + lh;
        let lf = lb(cv.s_in + cv.s_kernel - 1);
        let lk = lb(cv.s_kernel);
        assert_eq!(self.g.nvars[x], l_si + lci + lbatch, "conv input bits");
        assert_eq!(self.g.nvars[wf], lk + lci + lco, "shared kernel must have no batch variables");
        assert_eq!(self.g.nvars[yf], lf + lco + lbatch, "full-convolution bits");
        assert_eq!(c.point.len(), lwo + lho + lco + lbatch, "conv output bits");
        assert_eq!(cv.s_in, 1usize << l_si, "padded spatial stride");
        assert_eq!(cv.stride_w, 1usize << lw, "padded width stride");
        let r = &c.point;
        // Exact public crop relation. Unlike bit substitution, this also
        // supports non-power-of-two strides, without a carry-free assumption.
        let pairs = (0..cv.h_out).flat_map(|h| (0..cv.w_out).map(move |w| {
            (w + (h << lwo), cv.s_in - 1 - h * cv.conv_stride_h * cv.stride_w - w * cv.conv_stride_w)
        }));
        let mut axes = vec![MapAxis { off: 0, rel: Rel::map(lwo + lho, lf, pairs), ro: r[..lwo + lho].to_vec(), rk: vec![] }];
        let mut fixed = Vec::new();
        Self::chan_axis(cv.c_out, lf, lco, &r[lwo + lho..lwo + lho + lco], &mut fixed, &mut axes);
        Self::chan_axis(cv.batch, lf + lco, lbatch, &r[lwo + lho + lco..], &mut fixed, &mut axes);
        self.map_reduce(ctx, yf, fixed, axes, c.z, &c.pad);

        let alpha = ctx.chal();
        let rd = ctx.chals(lco);
        let rb = ctx.chals(lbatch);
        let (am, ap, aj) = conv_pp_powers(cv, alpha);
        let (mut eb, mut ed) = (eq_table(&rb), eq_table(&rd));
        eb[cv.batch..].fill(K8::ZERO);
        ed[cv.c_out..].fill(K8::ZERO);
        let (yf2, px, pw, val) = if ctx.prover {
            let pd = self.pd.unwrap();
            // Y_full is NOT masked here: the random-point polynomial identity
            // also forces all padded batch/channel/coefficient slots to zero.
            let yf_fixed: Vec<_> = (0..lco).map(|j| (lf + j, rd[j]))
                .chain((0..lbatch).map(|j| (lf + lco + j, rb[j]))).collect();
            let yf2 = partial_eval(pd.v(yf), lf + lco + lbatch, &yf_fixed);
            let xv = masked_axis_fold(pd.v(x), l_si + lci + lbatch, l_si + lci, &rb, cv.batch);
            let px: Vec<K8> = (0..1usize << lci).into_par_iter().map(|cc|
                (0..cv.s_in).fold(K8::ZERO, |a,p| a.add(&ap[p].mul(&xv[p + (cc << l_si)])))).collect();
            let wd = masked_axis_fold(pd.v(wf), lk + lci + lco, lk + lci, &rd, cv.c_out);
            let pw: Vec<K8> = (0..1usize << lci).map(|cc|
                (0..1usize << lk).fold(K8::ZERO, |a,j| a.add(&aj[j].mul(&wd[j + (cc << lk)])))).collect();
            let val = am.iter().zip(&yf2).fold(K8::ZERO, |a,(u,v)| a.add(&u.mul(v)));
            if selfcheck() {
                let rhs = (0..cv.c_in).fold(K8::ZERO, |a,cc| a.add(&px[cc].mul(&pw[cc])));
                assert_eq!(val, rhs, "batched conv polynomial-product identity");
            }
            (yf2, px, pw, Some(val))
        } else { (vec![],vec![],vec![],None) };
        let a = ctx.alloc("convolution A pad");
        let z = ctx.send(val.map(|v| v.add(&ctx.bank[a as usize])));
        let eta = lin1(a, K8::ONE);
        let am_eval = am.clone();
        let o1 = ctx.masked_sc(lf, 1, true, z, &eta,
            if ctx.prover { Some(Tables::Host(vec![am, yf2])) } else { None },
            &|xs: &[K8]| tensor::mle_k(&am_eval, xs));
        let rr1 = o1.rho.mul(&o1.rho.sub(&K8::ONE));
        let mut p1 = o1.x.clone(); p1.extend_from_slice(&rd); p1.extend_from_slice(&rb);
        self.push(ctx, yf, AClaim { point: p1, pad: lin1(o1.pads[0], rr1), z: o1.v[0] });
        let ci_masked = !cv.c_in.is_power_of_two();
        let tables2 = if ctx.prover {
            let mut t = Vec::new();
            if ci_masked { t.push((0..1usize << lci).map(|cc| if cc < cv.c_in { K8::ONE } else { K8::ZERO }).collect()); }
            t.push(px); t.push(pw); Some(Tables::Host(t))
        } else { None };
        let o2 = ctx.masked_sc(lci, 2, ci_masked, z, &eta, tables2, &|xs: &[K8]| prefix_mle(cv.c_in, xs));
        let rr2 = o2.rho.mul(&o2.rho.sub(&K8::ONE));
        let xc = &o2.x;
        // X_rev(x_c,alpha,r_b): retain batch as an explicit masked variable
        // in this reduction, so poisoned padded batches cannot enter the fold.
        let tables3 = if ctx.prover {
            let px: Vec<K8> = (0..1usize << (l_si + lbatch)).map(|i| ap[i & (cv.s_in - 1)].mul(&eb[i >> l_si])).collect();
            let fixed: Vec<_> = (0..lci).map(|j| (l_si + j, xc[j])).collect();
            Some(Tables::Host(vec![px, partial_eval(self.pd.unwrap().v(x), l_si + lci + lbatch, &fixed)]))
        } else { None };
        let o3 = ctx.masked_sc(l_si + lbatch, 1, true, o2.v[0], &lin1(o2.pads[0], rr2), tables3,
            &|xs: &[K8]| tensor::mle_k(&ap, &xs[..l_si]).mul(&tensor::mle_k(&eb, &xs[l_si..])));
        let rr3 = o3.rho.mul(&o3.rho.sub(&K8::ONE));
        let mut xp = o3.x[..l_si].to_vec(); xp.extend_from_slice(xc); xp.extend_from_slice(&o3.x[l_si..]);
        self.push(ctx, x, AClaim { point: xp, pad: lin1(o3.pads[0], rr3), z: o3.v[0] });
        // Shared W_flat has only j,c,d variables, never a batch coordinate.
        let tables4 = if ctx.prover {
            let pw: Vec<K8> = (0..1usize << (lk + lco)).map(|i| aj[i & ((1 << lk) - 1)].mul(&ed[i >> lk])).collect();
            let fixed: Vec<_> = (0..lci).map(|j| (lk + j, xc[j])).collect();
            Some(Tables::Host(vec![pw, partial_eval(self.pd.unwrap().v(wf), lk + lci + lco, &fixed)]))
        } else { None };
        let o4 = ctx.masked_sc(lk + lco, 1, true, o2.v[1], &lin1(o2.pads[1], rr2), tables4,
            &|xs: &[K8]| tensor::mle_k(&aj, &xs[..lk]).mul(&tensor::mle_k(&ed, &xs[lk..])));
        let rr4 = o4.rho.mul(&o4.rho.sub(&K8::ONE));
        let mut wp = o4.x[..lk].to_vec(); wp.extend_from_slice(xc); wp.extend_from_slice(&o4.x[lk..]);
        self.push(ctx, wf, AClaim { point: wp, pad: lin1(o4.pads[0], rr4), z: o4.v[0] });
    }

    /// Look through a FlattenKernel only when its public geometry agrees
    /// exactly with this Conv2D. Otherwise bind the ACTUAL flat weight edge
    /// with a tap map; blindly bypassing a different flatten is unsound.
    fn conv2d_kernel(&self, edge: usize, cv: &crate::basicblock::conv::Conv2D) -> (usize, Option<Rel>) {
        assert_eq!(self.g.nvars[edge], lb(cv.s_kernel) + lb(cv.c_in) + lb(cv.c_out),
            "Conv2D flat kernel extent must match its public indexing strides");
        let mut e = edge;
        loop {
            match self.g.dag.producers[e].map(|p| &self.g.dag.nodes[p]) {
                Some(n) if matches!(n.kind, BasicBlockType::ChangeShape(_))
                    && self.g.nvars[n.inputs[0]] == self.g.nvars[e] => e = n.inputs[0],
                Some(n) => {
                    if let BasicBlockType::FlattenKernel(fk) = &n.kind {
                        if (fk.s_w, fk.kh, fk.kw, fk.c_in, fk.c_out, fk.dilation_h, fk.dilation_w)
                            == (cv.stride_w, cv.kernel_h, cv.kernel_w, cv.c_in, cv.c_out, cv.dilation_h, cv.dilation_w) {
                            return (n.inputs[0], None);
                        }
                    }
                    break;
                }
                None => break,
            }
        }
        let sk = lb(cv.kernel_w) + lb(cv.kernel_h);
        let lk = self.g.nvars[edge].checked_sub(lb(cv.c_in) + lb(cv.c_out)).expect("flat kernel channels");
        assert_eq!(lk, lb(cv.s_kernel), "flat kernel spatial stride");
        let pairs = (0..cv.kernel_h).flat_map(|h| (0..cv.kernel_w).map(move |w|
            (w + (h << lb(cv.kernel_w)), h * cv.dilation_h * cv.stride_w + w * cv.dilation_w)));
        (edge, Some(Rel::map(sk, lk, pairs)))
    }

    /// ZeroPad-style map: Y[c, o_d] = X[c, o_d − off_d] for o_d − off_d < in_d.
    #[allow(clippy::too_many_arguments)]
    fn pad_like(&mut self, ctx: &mut Ctx, x: usize, y: usize, channels: usize, ins: &[usize], outs: &[usize], off: &[usize], c: AClaim) {
        // Axes are given outermost-first ([d,] h, w); the lowest bits are the last axis.
        let k = ins.len();
        let ib: Vec<usize> = ins.iter().map(|&v| lb(v)).collect();
        let ob: Vec<usize> = outs.iter().map(|&v| lb(v)).collect();
        let (si, so): (usize, usize) = (ib.iter().sum(), ob.iter().sum());
        let nc = self.g.nvars[x] - si;
        assert_eq!(self.g.nvars[y], so + nc, "pad channel bits");
        let r = &c.point;
        let (mut xo, mut yo) = (0, 0);
        let mut axes = Vec::new();
        let mut fixed = Vec::new();
        Self::chan_axis(channels, si, nc, &r[so..], &mut fixed, &mut axes);
        for d in (0..k).rev() {
            let rel = Rel::map(ob[d], ib[d], (0..ins[d]).map(|i| (i + off[d], i)));
            axes.push(MapAxis { off: xo, rel, ro: r[yo..yo + ob[d]].to_vec(), rk: vec![] });
            xo += ib[d]; yo += ob[d];
        }
        self.map_reduce(ctx, x, fixed, axes, c.z, &c.pad);
    }

    /// Two-stage convolution reduction. `wshape` is the unflattened kernel
    /// shape ([C_out, C_in, k..] normal/depthwise, [C_in, C_out, k..]
    /// transposed); `rels` are the spatial axes, lowest first.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, ctx: &mut Ctx, x: usize, w: usize, wshape: &[usize], rels: Vec<Rel>, kind: WKind, lci: usize, lco: usize, batch: usize, flat_kernel: Option<Rel>, c: AClaim) {
        let r = c.point.clone();
        let so: usize = rels.iter().map(|d| d.ob).sum();
        let si: usize = rels.iter().map(|d| d.ib).sum();
        let sk: usize = rels.iter().map(|d| d.kb).sum();
        assert!(batch > 0, "conv logical batch must be positive");
        let lbatch = lb(batch);
        assert_eq!(self.g.nvars[x], si + lci + lbatch, "conv input bits (edge {x})");
        assert_eq!(r.len(), so + lco + lbatch, "conv output bits");
        let weight_taps = flat_kernel.as_ref().map(|rel| rel.ib).unwrap_or(sk);
        let (cin_off,cout_off,expected_weight_bits) = conv_weight_offsets(kind,weight_taps,lci,lco);
        assert_eq!(self.g.nvars[w], expected_weight_bits, "shared conv weight bits (edge {w})");
        let ro: Vec<Vec<K8>> = { let mut off = 0; rels.iter().map(|d| { let v = r[off..off + d.ob].to_vec(); off += d.ob; v }).collect() };
        let rd = r[so..so + lco].to_vec();
        let rb = r[so + lco..].to_vec();
        let (channels_in, channels_out) = match kind {
            WKind::Normal => (wshape[1], wshape[0]),
            WKind::Transpose => (wshape[0], wshape[1]),
            WKind::Depthwise => (wshape[0], wshape[0]),
        };
        // Stage 1 variables: kernel taps then the summed channel.
        let n1 = sk + lci;
        let tables = if ctx.prover {
            let pd = self.pd.unwrap();
            let a_tab = conv_patch_table(pd.v(x), &rels, &ro, lci, batch, &rb);
            // W restricted to the output channel point (or kept, depthwise).
            let wv = pd.v(w);
            let nw = self.g.nvars[w];
            let w_tab = conv_weight_table(wv,nw,sk,lci,kind,cout_off,&rd,channels_out,flat_kernel.as_ref());
            assert_eq!(w_tab.len(), 1usize << n1, "kernel table size");
            assert_eq!(a_tab.len(), 1usize << n1, "patch table size");
            let mut t = Vec::new();
            if kind == WKind::Depthwise {
                // Public channel relation G(d, c) = [d = c]·[c < C]: zero on
                // padded channels, independent of the committed padding.
                let e = eq_table(&rd);
                let nch = wshape[0];
                t.push((0..1usize << n1).map(|i| if (i >> sk) < nch { e[i >> sk] } else { K8::ZERO }).collect::<Vec<_>>());
            } else if !channels_in.is_power_of_two() {
                t.push((0..1usize << n1).map(|i| if (i >> sk) < channels_in { K8::ONE } else { K8::ZERO }).collect());
            }
            t.push(w_tab);
            t.push(a_tab);
            Some(Tables::Host(t))
        } else { None };
        let dw = kind == WKind::Depthwise;
        let rd2 = rd.clone();
        let nch = wshape[0];
        let out = ctx.masked_sc(n1, 2, dw || !channels_in.is_power_of_two(), c.z, &c.pad, tables,
            &|xs: &[K8]| if dw { masked_eq(&rd2, &xs[sk..], nch) } else { prefix_mle(channels_in, &xs[sk..]) });
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        let (xk, xc) = (out.x[..sk].to_vec(), out.x[sk..].to_vec());
        // The kernel has no batch variables: one shared kernel is bound.
        // Non-power-of-two output channels must be masked, not simply fixed.
        let channel_off = if dw { cout_off } else { cin_off };
        let mut fixed: Vec<_> = (0..lci).map(|j| (channel_off+j,xc[j])).collect();
        let mut axes = Vec::new();
        if let Some(rel) = flat_kernel {
            axes.push(MapAxis { off:0,rel,ro:xk.clone(),rk:vec![] });
        } else { fixed.extend((0..sk).map(|j| (j,xk[j]))); }
        if !dw { Self::chan_axis(channels_out,cout_off,lco,&rd,&mut fixed,&mut axes); }
        self.map_reduce(ctx,w,fixed,axes,out.v[0],&lin1(out.pads[0],rr));
        // Patch claim -> input map, preserving the logical batch identity.
        let mut fixed: Vec<(usize, K8)> = (0..lci).map(|j| (si + j, xc[j])).collect();
        let (mut xo, mut ko) = (0, 0);
        let mut axes = Vec::new();
        for (a, d) in rels.into_iter().enumerate() {
            let (ib, kb) = (d.ib, d.kb);
            axes.push(MapAxis { off: xo, rel: d, ro: ro[a].clone(), rk: xk[ko..ko + kb].to_vec() });
            xo += ib; ko += kb;
        }
        Self::chan_axis(batch, si + lci, lbatch, &rb, &mut fixed, &mut axes);
        self.map_reduce(ctx, x, fixed, axes, out.v[1], &lin1(out.pads[1], rr));
    }

    /// ProductZeroCheck roots: Σ_x eq(r, x)·A(x)·B(x) = 0.
    pub(super) fn pzc_roots(&mut self, ctx: &mut Ctx) {
        let nodes: Vec<usize> = self.g.dag.nodes.iter().filter(|n| matches!(n.kind, BasicBlockType::ProductZeroCheck(_))).map(|n| n.id).collect();
        for node in nodes {
            let nd = self.g.dag.nodes[node].clone();
            let (a, b) = (nd.inputs[0], nd.inputs[1]);
            let n = self.g.nvars[a];
            assert_eq!(self.g.nvars[b], n);
            let r = ctx.chals(n);
            let tables = if ctx.prover {
                let pd = self.pd.unwrap();
                backend_tables!(gpu::enabled() && (1usize << n) >= gpu::threshold(), {
                    Some(Tables::Dev(vec![gpu::eq(&r), gpu::lift(&pd.dev(a), 1 << n), gpu::lift(&pd.dev(b), 1 << n)]))
                }, {
                    let f = |e: usize| pd.v(e).iter().map(|&v| K8::from_base(v)).collect::<Vec<_>>();
                    Some(Tables::Host(vec![eq_table(&r), f(a), f(b)]))
                })
            } else { None };
            let rc = r.clone();
            let out = ctx.masked_sc(n, 2, true, K8::ZERO, &Lin::new(), tables, &|xs: &[K8]| eq_pt(&rc, xs));
            let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
            self.push(ctx, a, AClaim { point: out.x.clone(), pad: lin1(out.pads[0], rr), z: out.v[0] });
            self.push(ctx, b, AClaim { point: out.x, pad: lin1(out.pads[1], rr), z: out.v[1] });
        }
    }
}

/// The public descriptor must encode the same canonical geometry as the
/// forward block. In particular no inconsistent flatten stride is inferred
/// from just a total tensor arity.
fn validate_conv2d_geometry(cv: &crate::basicblock::conv::Conv2D) {
    assert!([cv.batch, cv.c_in, cv.c_out, cv.kernel_h, cv.kernel_w, cv.input_h, cv.input_w,
        cv.conv_stride_h, cv.conv_stride_w, cv.dilation_h, cv.dilation_w].iter().all(|&v| v > 0), "positive Conv2D geometry");
    let eh = (cv.kernel_h - 1).checked_mul(cv.dilation_h).and_then(|v| v.checked_add(1)).expect("Conv2D height overflow");
    let ew = (cv.kernel_w - 1).checked_mul(cv.dilation_w).and_then(|v| v.checked_add(1)).expect("Conv2D width overflow");
    assert!(eh <= cv.input_h && ew <= cv.input_w, "Conv2D effective kernel must fit input");
    let wp = cv.input_w.checked_next_power_of_two().expect("Conv2D width extent");
    let hp = cv.input_h.checked_next_power_of_two().expect("Conv2D height extent");
    assert_eq!(cv.stride_w, wp, "Conv2D flat width stride");
    assert_eq!(cv.s_in, wp.checked_mul(hp).expect("Conv2D spatial extent"));
    assert_eq!(cv.s_kernel, (eh - 1).checked_mul(wp).and_then(|v| v.checked_add(ew)).expect("Conv2D kernel extent"));
    assert_eq!(cv.h_out, (cv.input_h - eh) / cv.conv_stride_h + 1);
    assert_eq!(cv.w_out, (cv.input_w - ew) / cv.conv_stride_w + 1);
}

/// Convolution relation along one axis: i = s·o + δ·k (o < O, k < K, i < I).
pub(super) fn trip_conv(o_n: usize, k_n: usize, i_n: usize, s: usize, dl: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for o in 0..o_n { for k in 0..k_n { let i = o * s + dl * k; if i < i_n { v.push((o, k, i)); } } }
    v
}

/// Transposed convolution relation: o = s·i + k (o < O).
fn trip_tconv(o_n: usize, k_n: usize, i_n: usize, s: usize) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for i in 0..i_n { for k in 0..k_n { let o = i * s + k; if o < o_n { v.push((o, k, i)); } } }
    v
}

#[cfg(test)]
mod prefix_tests {
    use super::*;
    #[test]
    fn prefix_mle_matches_the_boolean_sum() {
        let x: Vec<K8> = (0..5).map(|i| K8::from_base(7 + 13 * i as u64)).collect();
        for n in 0..=33usize {
            let e = eq_table(&x);
            let want = (0..32usize.min(n)).fold(K8::ZERO, |a, c| a.add(&e[c]));
            assert_eq!(prefix_mle(n, &x), want, "n = {n}");
        }
    }
}

#[cfg(any(test, feature = "cpu-replay"))]
#[path = "cv_family_tests.rs"]
pub mod family_tests;
