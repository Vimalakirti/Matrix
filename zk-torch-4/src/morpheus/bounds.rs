//! Certified integer bounds and no-wrap checks for a registered graph
//! (paper Appendix A, Eq. rounding_nowrap and the accumulator/product and
//! source-decoding conditions next to it).
//!
//! The pass reads only public statement data: the DAG, shapes, roles, the
//! declared digit planes of every committed source (model tensors: their
//! registered width; request tensors: 2^REQ_PLANES_LOG planes; bit sources:
//! {0,1}), immutable program literals, and the lookup/range tables. It never
//! reads a request's computed values.
//!
//! An edge bound [lo, hi] is *certified* when every field assignment that
//! satisfies the graph's constraints has a centered integer lift in [lo, hi].
//! Sources of certified bounds:
//!   * committed tensors: the PCS decoder radius (B_x-1)(2^m-1) = 2^m - 1
//!     for ternary digits and m planes ({0,1} for bit sources);
//!   * program literals: their exact values;
//!   * lookups: a NonNegative(b) root proves its input is a table index in
//!     [0, 2^b); TwoPow outputs lie in [0, 2^15]; ExpHelper quotients in
//!     [0, 2^b);
//!   * linear propagation: c = a +/- b, with certified a, b whose magnitudes
//!     sum below q/2, has lift(c) = lift(a) +/- lift(b) (forward); and
//!     a = c -/+ b (backward) likewise; contractions forward only (sum of
//!     |products| below q/2); rescale x = S*y + r - floor(S/2) gives x from y;
//!     a ProductZeroCheck against a nonzero literal forces 0;
//!   * views: forward (and backward where every input entry appears).
//! Nothing is propagated forward through a rescale: its output
//! y = (x + floor(S/2) - r)/S is a field quotient, an integer only for the
//! right remainder, so y is certified only by a constraint downstream.
//!
//! Every derived bound is intersected into the edge's interval; rules are
//! iterated (forward topological pass + backward pass) to a fixed point.
//! Entry: `MORPHEUS_BOUNDS=<csv prefix>` in `Dag::run` (before the forward
//! pass; nothing is computed and no GPU is touched).

use crate::basicblock::BasicBlockType as K;
use crate::dag::{Dag, Role, Witness};
use crate::morpheus::piop::{self, Config, Graph};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use std::io::Write;

type Iv = Option<(i128, i128)>;
const CAP: i128 = 1i128 << 120;
fn hq() -> i128 { ((Q - 1) / 2) as i128 }
fn mk(lo: i128, hi: i128) -> Iv { if lo < -CAP || hi > CAP { None } else { Some((lo, hi)) } }
fn mag(a: Iv) -> Option<i128> { a.map(|(l, h)| l.abs().max(h.abs())) }
fn hull(a: Iv, b: Iv) -> Iv { match (a, b) { (Some(x), Some(y)) => Some((x.0.min(y.0), x.1.max(y.1))), _ => None } }
fn with0(a: Iv) -> Iv { hull(a, Some((0, 0))) }
/// a + s*b, valid in the integers when the magnitudes stay below q/2.
fn lin(a: Iv, b: Iv, s: i128) -> Iv {
    let (x, y) = (a?, b?);
    if mag(a)? + mag(b)? > hq() { return None; }
    if s > 0 { mk(x.0 + y.0, x.1 + y.1) } else { mk(x.0 - y.1, x.1 - y.0) }
}
fn mul(a: (i128, i128), b: (i128, i128)) -> Iv {
    let c = [a.0.checked_mul(b.0), a.0.checked_mul(b.1), a.1.checked_mul(b.0), a.1.checked_mul(b.1)];
    if c.iter().any(|x| x.is_none()) { return None; }
    let c: Vec<i128> = c.iter().map(|x| x.unwrap()).collect();
    mk(*c.iter().min().unwrap(), *c.iter().max().unwrap())
}
fn times(a: Iv, n: i128) -> Iv { let (l, h) = a?; mk(l.checked_mul(n)?, h.checked_mul(n)?) }
fn lg(x: Option<i128>) -> String { match x { None => "inf".into(), Some(v) => format!("{:.2}", ((v.max(0) as f64) + 1.0).log2()) } }
fn pad(d: usize) -> i128 { d.max(1).next_power_of_two() as i128 }

pub struct Bounds {
    pub iv: Vec<Iv>, pub why: Vec<String>,
    /// Support bounds: (edge, mask id) -> interval of the entries where the
    /// public 0/1 mask is one (from x*M range constraints).
    pub sup: std::collections::HashMap<(usize, usize), (i128, i128)>,
    /// Interned masks (flat padded 0/1 vectors) and their literal edges.
    pub masks: Vec<(Vec<bool>, usize)>,
    sup_idx: std::collections::HashMap<usize, std::collections::BTreeSet<usize>>,
    /// Two-operand contractions with a public literal operand: (other input
    /// index, max positive row sum P, min negative row sum N, remaining
    /// summed extent) where rows are the literal's non-summed coordinates.
    lit_cache: std::collections::HashMap<usize, Option<(usize, i128, i128, i128)>>,
    mask_of_lit: std::collections::HashMap<usize, Option<usize>>,
}

impl Bounds {
    /// Mask id of a 0/1 program literal edge (None if not 0/1 or all ones).
    fn mask_id(&mut self, dag: &Dag, e: usize) -> Option<usize> {
        if let Some(m) = self.mask_of_lit.get(&e) { return *m; }
        let r = dag.program_literals.get(&e).and_then(|v| {
            if !v.iter().all(|&x| x <= 1) || v.iter().all(|&x| x == 1) { return None; }
            let bits: Vec<bool> = v.iter().map(|&x| x == 1).collect();
            match self.masks.iter().position(|(b, _)| *b == bits) { Some(i) => Some(i), None => { self.masks.push((bits, e)); Some(self.masks.len() - 1) } }
        });
        self.mask_of_lit.insert(e, r);
        r
    }
    /// Intern the intersection of two masks (same length), if not empty.
    fn meet_mask(&mut self, a: usize, c: usize) -> Option<usize> {
        if a == c { return Some(a); }
        if self.masks[a].0.len() != self.masks[c].0.len() { return None; }
        let bits: Vec<bool> = self.masks[a].0.iter().zip(&self.masks[c].0).map(|(x, y)| *x && *y).collect();
        if !bits.iter().any(|&t| t) { return None; }
        let lit = self.masks[a].1;
        Some(match self.masks.iter().position(|(b, _)| *b == bits) { Some(i) => i, None => { self.masks.push((bits, lit)); self.masks.len() - 1 } })
    }
    /// Bound of edge e on the support of mask m (full bound intersected).
    fn on(&self, e: usize, m: usize) -> Iv {
        match (self.iv[e], self.sup.get(&(e, m))) {
            (Some(a), Some(b)) => Some((a.0.max(b.0), a.1.min(b.1))),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(*b),
            _ => None,
        }
    }
    fn set_sup(&mut self, e: usize, m: usize, v: Iv) -> bool {
        let Some((l, h)) = v else { return false };
        if let Some((fl, fh)) = self.iv[e] { if fl >= l && fh <= h { return false; } }
        match self.sup.get(&(e, m)).copied() {
            None => { self.sup.insert((e, m), (l, h)); self.sup_idx.entry(e).or_default().insert(m); true }
            Some((ol, oh)) => {
                let (nl, nh) = (ol.max(l), oh.min(h));
                if nl > nh || (nl, nh) == (ol, oh) { false } else { self.sup.insert((e, m), (nl, nh)); true }
            }
        }
    }
    fn masks_on(&self, es: &[usize]) -> Vec<usize> {
        let mut v: Vec<usize> = es.iter().flat_map(|e| self.sup_idx.get(e).into_iter().flatten().copied()).collect();
        v.sort(); v.dedup(); v
    }
    fn set(&mut self, e: usize, v: Iv, why: &str) -> bool {
        let Some((l, h)) = v else { return false };
        match self.iv[e] {
            None => { self.iv[e] = Some((l, h)); self.why[e] = why.into(); true }
            Some((ol, oh)) => {
                let (nl, nh) = (ol.max(l), oh.min(h));
                if nl > nh { eprintln!("[bounds] empty interval on edge {e} ({why}); keeping [{ol},{oh}]"); return false; }
                if (nl, nh) != (ol, oh) { self.iv[e] = Some((nl, nh)); self.why[e] = why.into(); true } else { false }
            }
        }
    }
}

fn centered(v: u64) -> i128 { if v > Q / 2 { v as i128 - Q as i128 } else { v as i128 } }

/// Einsum: (summed-term count over padded extents, is pure view).
fn einsum_terms(eq: &str, in_shapes: &[Vec<usize>]) -> (i128, bool) {
    let (lhs, rhs) = eq.split_once("->").unwrap();
    let specs: Vec<Vec<char>> = lhs.split(',').map(|s| s.trim().chars().collect()).collect();
    let out: Vec<char> = rhs.trim().chars().collect();
    let mut ext = std::collections::BTreeMap::new();
    for (s, sh) in specs.iter().zip(in_shapes) { for (c, d) in s.iter().zip(sh) { ext.insert(*c, pad(*d)); } }
    let n: i128 = ext.iter().filter(|(c, _)| !out.contains(c)).map(|(_, d)| *d).product();
    (n, specs.len() == 1 && n == 1)
}

/// Convolutions: summed terms per output (padded channel x kernel volume).
fn conv_terms(k: &K) -> Option<i128> {
    Some(match k {
        K::Conv1D(c) => pad(c.c_in) * pad(c.kernel_size),
        K::Conv2D(c) => pad(c.c_in) * pad(c.kernel_h) * pad(c.kernel_w),
        K::Conv3D(c) => pad(c.c_in) * pad(c.kernel_d) * pad(c.kernel_h) * pad(c.kernel_w),
        K::DepthwiseConv2D(c) => pad(c.kernel_h) * pad(c.kernel_w),
        K::ConvTranspose1D(c) => pad(c.c_in) * pad(c.kernel_size),
        K::ConvTranspose2D(c) => pad(c.c_in) * pad(c.kernel_h) * pad(c.kernel_w),
        K::ConvTranspose3D(c) => pad(c.c_in) * pad(c.kernel_d) * pad(c.kernel_h) * pad(c.kernel_w),
        _ => return None,
    })
}

fn is_view(k: &K) -> bool {
    matches!(k, K::ZeroPad(_) | K::ZeroPadAsym(_) | K::ZeroPad3D(_) | K::SubSample2D(_) | K::Replicate2x2(_) | K::ChannelSlice(_) | K::Concat(_) | K::FlattenKernel(_) | K::FlattenKernel3D(_) | K::Permute(_))
}

pub fn kind_name(k: &K) -> String { format!("{k:?}").split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string() }

/// Committed sources and their declared planes, as in request::prepare and
/// piop::tensor_specs.
fn sources<'a>(dag: &'a Dag, wit: &[Vec<Witness>]) -> (Graph<'a>, Vec<Iv>, Vec<String>) {
    let cfg = Config { tcl: std::env::var("TCL").ok().and_then(|s| s.parse().ok()).unwrap_or(4), slots_log: 10,
        req_planes_log: std::env::var("REQ_PLANES_LOG").ok().and_then(|s| s.parse().ok()).unwrap_or(4) };
    let mut g = Graph::new(dag, wit, cfg);
    let ne = dag.num_edges();
    // Public request inputs: leaf Role::Input with values (driver::run_hook).
    for e in 0..ne {
        if dag.producers[e].is_none() && wit[e][0].role == Role::Input && wit[e][0].data.is_some() && !g.public.contains_key(&e) {
            g.public.insert(e, piop::witness_vals(&wit[e][0]));
        }
    }
    let private: Vec<usize> = (0..ne).filter(|&e| dag.producers[e].is_none() && wit[e][0].role == Role::Input && !g.public.contains_key(&e)).collect();
    g.prune_unclaimed();
    for &e in &private { g.committed[e] = true; }
    for e in 0..ne { if g.model[e] { g.committed[e] = true; } }
    let mut iv = vec![None; ne];
    let mut why = vec![String::new(); ne];
    for e in 0..ne {
        if let Some(v) = dag.program_literals.get(&e) {
            let (mut l, mut h) = (i128::MAX, i128::MIN);
            for &x in v { let c = centered(x); l = l.min(c); h = h.max(c); }
            iv[e] = Some((l, h)); why[e] = "literal".into();
            continue;
        }
        if !g.committed[e] { continue; }
        let (planes, bits) = if g.model[e] || g.public.contains_key(&e) {
            let v = match g.public.get(&e) { Some(v) => v.clone(), None => piop::witness_vals(&wit[e][0]) };
            let maxb = v.iter().map(|&x| { let a = if x > Q / 2 { Q - x } else { x }; 64 - a.leading_zeros() }).max().unwrap_or(0).max(1) as usize;
            (1usize << crate::util::arith::log2_ceil(maxb), v.iter().all(|&x| x <= 1))
        } else if piop::request_source_is_bits(&g, e) { (1, true) } else { (1usize << g.cfg.req_planes_log, false) };
        if bits { iv[e] = Some((0, 1)); why[e] = "bit source".into(); }
        else { let r = (1i128 << planes) - 1; iv[e] = Some((-r, r)); why[e] = format!("decode {planes} ternary planes"); }
    }
    // Terminal outputs are public (bound by a root claim to the published
    // values), so the verifier knows them exactly.
    for (e, v) in &g.outputs {
        let (mut l, mut h) = (i128::MAX, i128::MIN);
        for &x in v.iter() { let c = centered(x); l = l.min(c); h = h.max(c); }
        if l <= h { iv[*e] = Some((l, h)); why[*e] = "public output".into(); }
    }
    (g, iv, why)
}

pub fn run(dag: &Dag, wit: &[Vec<Witness>], prefix: &str) {
    let t0 = std::time::Instant::now();
    let model = std::env::var("MORPHEUS_NAME").or_else(|_| std::env::var("MODEL")).unwrap_or_else(|_| "model".into());
    let (g, iv, why) = sources(dag, wit);
    let ne = dag.num_edges();
    let committed = g.committed.clone();
    let mut b = Bounds { iv, why, sup: Default::default(), masks: Vec::new(), mask_of_lit: Default::default(), sup_idx: Default::default(), lit_cache: Default::default() };
    // Field-native power chains (grid routing's Fermat equality test): an
    // elementwise product tree computing x^(q-1) is {0,1}-valued in F_q for
    // every x, so its intermediate products need no integer lift.
    let fermat = fermat_powers(dag, &g);
    for (&e, &(_, k)) in &fermat.1 { if k == (Q - 1) as u128 { b.iv[e] = Some((0, 1)); b.why[e] = "Fermat x^(q-1) in F_q".into(); } }
    let sf = *crate::SF_LOG;
    let ln2 = crate::basicblock::exp::exp_ln2(sf);
    let lookups = g.lookups.clone();
    // Fixed point.
    let mut rounds = 0;
    loop {
        rounds += 1;
        let mut ch = false;
        for &n in &dag.topo { ch |= step(dag, &g, &mut b, n, ln2, &lookups, true); }
        for &n in dag.topo.iter().rev() { ch |= step(dag, &g, &mut b, n, ln2, &lookups, false); }
        if !ch || rounds >= 64 { break; }
    }
    eprintln!("[bounds] support bounds {} on {} masks", b.sup.len(), b.masks.len());
    if std::env::var_os("BOUNDS_GRAPH").is_some() { for nd in &dag.nodes { let ins: Vec<String> = nd.inputs.iter().map(|&i| format!("e{i}{:?}{}", g.shapes[i], if dag.program_literals.contains_key(&i) {"L"} else if committed[i] {"C"} else {""})).collect(); eprintln!("[node] {} {:?} in {:?} out {:?} iv_out {:?} sup {:?}", nd.id, format!("{:?}", nd.kind).chars().take(90).collect::<String>(), ins, nd.outputs, b.iv[nd.outputs[0]], b.sup_idx.get(&nd.outputs[0])); } }
    if std::env::var_os("BOUNDS_DEBUG").is_some() { for ((e, mi), v) in &b.sup { eprintln!("[sup] e{e} prod {:?} mask {mi} {:?}", dag.producers[*e].map(|p| (p, kind_name(&dag.nodes[p].kind))), v); } }
    eprintln!("[bounds] fixed point after {rounds} rounds ({:.1}s)", t0.elapsed().as_secs_f64());

    // Checks.
    let qf = (Q as f64).log2();
    let mut rows: Vec<String> = Vec::new();
    let mut summ: std::collections::BTreeMap<String, (usize, usize, f64)> = Default::default();
    let mut sup_count: std::collections::BTreeMap<String, usize> = Default::default();
    let mut emit = |rows: &mut Vec<String>, node: usize, kind: &str, check: &str, s: i128, z: Option<i128>, y: Option<i128>, be: Option<i128>, lhs: Option<i128>, note: &str| {
        let support_only = note.starts_with("SUPPORT-ONLY");
        let (status, margin) = if note.starts_with("FIELD-NATIVE") { ("field-exact", f64::NAN) } else { match lhs { Some(l) if l < Q as i128 => (if support_only { "SUPPORT-ONLY" } else { "certified" }, qf - ((l.max(1)) as f64).log2()), Some(_) => ("VIOLATED", f64::NAN), None => ("UNCERTIFIED", f64::NAN) } };
        let e = summ.entry(check.to_string()).or_insert((0, 0, f64::INFINITY));
        e.0 += 1; if status == "field-exact" { e.1 += 1; } if status == "certified" { e.1 += 1; if margin < e.2 { e.2 = margin; } }
        if status == "SUPPORT-ONLY" { *sup_count.entry(check.to_string()).or_insert(0) += 1; }
        rows.push(format!("{model},{node},{kind},{check},{s},{},{},{},{},{},{status},{note}", lg(z), lg(y), lg(be), lg(lhs), if margin.is_nan() { "".into() } else { format!("{margin:.2}") }));
    };
    let m = |e: usize, b: &Bounds| mag(b.iv[e]);
    let prod_name = |e: usize| dag.producers[e].map(|p| kind_name(&dag.nodes[p].kind)).unwrap_or_else(|| if committed[e] { "source".into() } else { "leaf".into() });
    for &n in &dag.topo {
        let nd = &dag.nodes[n];
        let kn = kind_name(&nd.kind);
        let o = nd.outputs[0];
        match &nd.kind {
            K::ScaleDown(_) | K::ScaleUp(_) => {
                let l = lookups[&n] as u32;
                let s = 1i128 << l;
                let (x, y) = (nd.inputs[0], o);
                let up = matches!(nd.kind, K::ScaleUp(_));
                let (z, yy) = (m(x, &b), m(y, &b));
                let mut lhs = if up { (|| Some(s * z? + yy? + s / 2))() } else { (|| Some(z? + s * yy? + s / 2))() };
                let mut note = format!("x<-{} [{}]; y bound via: {}", prod_name(x), b.why[x], if b.iv[y].is_some() { b.why[y].clone() } else { "none".to_string() }) + &format!("; x from: {}; y fate: {}", prod_fate(dag, x), fate(dag, y));
                let (mut z, mut yy) = (z, yy);
                if lhs.is_none() && !up {
                    // Certified only where a public 0/1 mask is one?
                    for mi in b.masks_on(&[y]) {
                        if let (Some(z2), Some(y2)) = (mag(b.on(x, mi)), mag(b.on(y, mi))) {
                            let l2 = z2 + s * y2 + s / 2;
                            if lhs.map(|l| l2 < l).unwrap_or(true) {
                                let (bits, lit) = &b.masks[mi];
                                lhs = Some(l2); z = Some(z2); yy = Some(y2);
                                note = format!("SUPPORT-ONLY: y bounded only where public mask e{lit} is 1 ({} of {} entries); {note}", bits.iter().filter(|&&t| t).count(), bits.len());
                            }
                        }
                    }
                }
                emit(&mut rows, n, &kn, if up { "upscale" } else { "rescale" }, s, z, yy, None, lhs, &note.replace(',', ";"));
            }
            K::DivConst(d) => {
                let c = d.divisor as i128;
                let (z, yy) = (m(nd.inputs[0], &b), m(o, &b));
                let lhs = (|| Some(z? + c * yy? + c / 2))();
                emit(&mut rows, n, &kn, "rational-rescale", c, z, yy, None, lhs, &format!("y via {}", b.why[o]).replace(',', ";"));
            }
            K::Einsum(es) => {
                let (nt, view) = einsum_terms(&es.equation, &es.input_shapes);
                if view { continue; }
                if fermat.0.contains(&n) {
                    let k = fermat.1.get(&o).map(|x| x.1).unwrap_or(0);
                    emit(&mut rows, n, &kn, "field-power", nt, None, None, None, Some(1), &format!("FIELD-NATIVE: x^{k} of a Fermat x^(q-1) equality chain; exact in F_q, no integer lift needed"));
                    continue;
                }
                let be = einsum_fwd(&b, &nd.inputs, nt).and_then(|v| mag(Some(v)));
                let mut raw = einsum_raw(&b, &nd.inputs, nt);
                if let Some(Some((k, p, nn, rest))) = b.lit_cache.get(&n).copied() {
                    if let Some(xm) = m(nd.inputs[k], &b) {
                        let r2 = (p - nn).checked_mul(xm).and_then(|v| v.checked_mul(rest));
                        if let Some(r2) = r2 { if raw.map(|r| r2 < r).unwrap_or(true) { raw = Some(r2); } }
                    }
                }
                let z = m(o, &b);
                let lhs = (|| Some(z? + raw?))();
                let check = if nt > 1 { "accumulator" } else if nd.inputs.len() > 1 { "product" } else { "accumulator" };
                let ins: Vec<String> = nd.inputs.iter().map(|&i| format!("{}:{}", prod_name(i), lg(m(i, &b)))).collect();
                let _ = be;
                emit(&mut rows, n, &kn, check, nt, z, None, raw, lhs, &format!("{} terms={nt} in=[{}]", es.equation, ins.join(" ")).replace(',', ";"));
            }
            K::Add(_) | K::Sub(_) => {
                let raw = (|| Some(m(nd.inputs[0], &b)? + m(nd.inputs[1], &b)?))();
                let z = m(o, &b);
                let lhs = (|| Some(z? + raw?))();
                emit(&mut rows, n, &kn, "add/sub", 1, z, None, raw, lhs, &format!("in=[{}:{} {}:{}]", prod_name(nd.inputs[0]), lg(m(nd.inputs[0], &b)), prod_name(nd.inputs[1]), lg(m(nd.inputs[1], &b))).replace(',', ";"));
            }
            k if conv_terms(k).is_some() => {
                let nt = conv_terms(k).unwrap();
                let raw = (|| m(nd.inputs[0], &b)?.checked_mul(m(nd.inputs[1], &b)?)?.checked_mul(nt))();
                let z = m(o, &b);
                let lhs = (|| Some(z? + raw?))();
                emit(&mut rows, n, &kn, "accumulator", nt, z, None, raw, lhs, &format!("conv terms={nt} x:{} w:{}", lg(m(nd.inputs[0], &b)), lg(m(nd.inputs[1], &b))));
            }
            K::NonNegative(nn) => {
                let r = (1i128 << nn.table_size_log) - 1;
                emit(&mut rows, n, &kn, "range-lookup", nn.table_size_log as i128, Some(r), None, None, Some(2 * r), &format!("on {}", prod_name(nd.inputs[0])));
            }
            K::ExpHelper(eh) => {
                let kmax = ln2 * ((1i128 << eh.num_bits) - 1);
                let (x, r) = (m(nd.inputs[0], &b), m(o, &b));
                let lhs = (|| Some(x? + r? + kmax))();
                emit(&mut rows, n, &kn, "exp-quotient", ln2, x, r, Some(kmax), lhs, "x = r - ln2*k");
            }
            K::TwoPow(_) => { emit(&mut rows, n, &kn, "two-pow-lookup", 1 << 15, m(o, &b), None, None, Some(2 * (1 << 15)), ""); }
            K::ProductZeroCheck(_) => {
                let (a, c) = (nd.inputs[0], nd.inputs[1]);
                let lit_nz = dag.program_literals.get(&c).map(|v| v.iter().all(|&x| x != 0)).unwrap_or(false);
                let raw = (|| m(a, &b)?.checked_mul(m(c, &b)?))();
                let lhs = if lit_nz { Some(raw.unwrap_or(0)) } else { raw };
                emit(&mut rows, n, &kn, "zero-product", 1, m(a, &b), m(c, &b), raw, if lit_nz && raw.is_none() { Some(1) } else { lhs }, if lit_nz { "against nonzero literal: field-exact" } else { "" });
            }
            _ => {}
        }
    }
    // Source decoding: q > 2 (B_x - 1)(2^m - 1).
    for e in 0..ne {
        if !committed[e] || dag.program_literals.contains_key(&e) { continue; }
        let r = mag(b.iv[e]);
        emit(&mut rows, usize::MAX, &prod_name(e), "source-decode", e as i128, r, None, None, r.map(|r| 2 * r), &b.why[e]);
    }
    let path = format!("{prefix}.csv");
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "model,node,op,check,param,log2_Z,log2_Y,log2_BE,log2_lhs,margin_bits,status,note").unwrap();
    for r in &rows { writeln!(f, "{r}").unwrap(); }
    let mut s = std::fs::File::create(format!("{prefix}.summary.txt")).unwrap();
    writeln!(s, "model {model}: {} nodes, {} edges, q = {Q} (log2 {:.4}), fixed point {rounds} rounds", dag.nodes.len(), ne, qf).unwrap();
    for (k, (n, c, mm)) in &summ { let sc = sup_count.get(k).copied().unwrap_or(0); writeln!(s, "{k:18} checks {n:7}  certified {c:7}  support-only {sc:6}  uncertified/violated {:7}  min margin bits {}", n - c - sc, if mm.is_finite() { format!("{mm:.2}") } else { "-".into() }).unwrap(); }
    eprintln!("[bounds] wrote {path} ({} rows)", rows.len());
    print!("{}", std::fs::read_to_string(format!("{prefix}.summary.txt")).unwrap());
}

/// (nodes in Fermat power chains, edge -> (base edge, exponent)).
fn fermat_powers(dag: &Dag, g: &Graph) -> (std::collections::HashSet<usize>, std::collections::HashMap<usize, (usize, u128)>) {
    let mut pw: std::collections::HashMap<usize, (usize, u128)> = Default::default();
    let ones = |e: usize| dag.program_literals.get(&e).map(|v| v.iter().all(|&x| x == 1)).unwrap_or(false);
    for &n in &dag.topo {
        let nd = &dag.nodes[n];
        let K::Einsum(es) = &nd.kind else { continue };
        let (lhs, rhs) = es.equation.split_once("->").unwrap();
        let specs: Vec<&str> = lhs.split(',').map(|t| t.trim()).collect();
        if specs.len() != 2 || !specs.iter().all(|t| *t == rhs.trim()) || g.shapes[nd.inputs[0]] != g.shapes[nd.inputs[1]] { continue; }
        let term = |e: usize| -> Option<(Option<usize>, u128)> { if ones(e) { Some((None, 0)) } else if let Some(&(b, k)) = pw.get(&e) { Some((Some(b), k)) } else { Some((Some(e), 1)) } };
        let (a, c) = (term(nd.inputs[0]).unwrap(), term(nd.inputs[1]).unwrap());
        let base = match (a.0, c.0) { (Some(x), Some(y)) if x == y => x, (Some(x), None) | (None, Some(x)) => x, _ => continue };
        pw.insert(nd.outputs[0], (base, a.1 + c.1));
    }
    // Keep only chains that reach x^(q-1) (walk back through their products).
    let target = (Q - 1) as u128;
    let mut nodes = std::collections::HashSet::new();
    let mut keep: std::collections::HashMap<usize, (usize, u128)> = Default::default();
    let mut stack: Vec<usize> = pw.iter().filter(|(_, v)| v.1 == target).map(|(e, _)| *e).collect();
    while let Some(e) = stack.pop() {
        if keep.contains_key(&e) { continue; }
        let Some(&v) = pw.get(&e) else { continue };
        keep.insert(e, v);
        if let Some(p) = dag.producers[e] { nodes.insert(p); for &i in &dag.nodes[p].inputs { if pw.contains_key(&i) { stack.push(i); } } }
    }
    (nodes, keep)
}

/// Where an edge's value goes: follow views and bias/elementwise adds while
/// the edge has one consumer, up to the first contraction, lookup or fan-out.
fn fate(dag: &Dag, mut e: usize) -> String {
    let mut path = Vec::new();
    for _ in 0..8 {
        let cs = &dag.consumers[e];
        if cs.len() != 1 { path.push(format!("fanout{}", cs.len())); break; }
        let nd = &dag.nodes[cs[0]];
        let k = match &nd.kind { K::Einsum(es) => format!("Einsum[{}]", es.equation.replace(',', ";")), k => kind_name(k) };
        let pass = matches!(nd.kind, K::ChangeShape(_) | K::Add(_) | K::Sub(_)) || matches!(&nd.kind, K::Einsum(es) if einsum_terms(&es.equation, &es.input_shapes).1);
        path.push(k);
        if !pass { break; }
        e = nd.outputs[0];
    }
    path.join(">")
}

/// The producer of x (einsum equation or kind).
fn prod_fate(dag: &Dag, e: usize) -> String {
    match dag.producers[e].map(|p| &dag.nodes[p].kind) { Some(K::Einsum(es)) => format!("Einsum[{}]", es.equation.replace(',', ";")), Some(k) => kind_name(k), None => "leaf".into() }
}

fn consumers(dag: &Dag, e: usize) -> String {
    let mut v: Vec<String> = dag.consumers[e].iter().map(|&c| kind_name(&dag.nodes[c].kind)).collect();
    v.sort(); v.dedup(); v.join("|")
}

/// Raw magnitude bound sum |prod| (B_E), from certified inputs.
fn lit_contract(dag: &Dag, nd: &crate::dag::Node, es: &crate::basicblock::einsum::Einsum) -> Option<(usize, i128, i128, i128)> {
    if nd.inputs.len() != 2 { return None; }
    let (lhs, rhs) = es.equation.split_once("->").unwrap();
    let specs: Vec<Vec<char>> = lhs.split(',').map(|t| t.trim().chars().collect()).collect();
    let out: Vec<char> = rhs.trim().chars().collect();
    let k = (0..2).find(|&k| dag.program_literals.contains_key(&nd.inputs[k]))?;
    let vals = &dag.program_literals[&nd.inputs[k]];
    let spec = &specs[k];
    let pads: Vec<usize> = es.input_shapes[k].iter().map(|&d| d.max(1).next_power_of_two()).collect();
    if pads.iter().product::<usize>() != vals.len() { return None; }
    // Row key: the literal's coordinates on non-summed chars.
    let summed: Vec<bool> = spec.iter().map(|c| !out.contains(c)).collect();
    let mut rows: std::collections::HashMap<usize, (i128, i128)> = Default::default();
    for (i, &v) in vals.iter().enumerate() {
        let (mut rem, mut key, mut kstride) = (i, 0usize, 1usize);
        for d in 0..pads.len() {
            let c = rem % pads[d]; rem /= pads[d];
            if !summed[d] { key += c * kstride; kstride *= pads[d]; }
        }
        let x = centered(v);
        let r = rows.entry(key).or_insert((0, 0));
        if x > 0 { r.0 += x } else { r.1 += x }
    }
    let p = rows.values().map(|r| r.0).max().unwrap_or(0);
    let n = rows.values().map(|r| r.1).min().unwrap_or(0);
    // Summed chars of the other operand not covered by the literal.
    let mut rest: i128 = 1;
    let other = &specs[1 - k];
    for (c, &d) in other.iter().zip(&es.input_shapes[1 - k]) { if !out.contains(c) && !spec.contains(c) { rest *= pad(d); } }
    Some((1 - k, p, n, rest))
}

/// Signed range and magnitude bound of a literal contraction (see lit_cache).
fn lit_bound(b: &Bounds, ins: &[usize], lc: Option<(usize, i128, i128, i128)>) -> Option<(i128, i128)> {
    let (k, p, n, rest) = lc?;
    let (xl, xh) = b.iv[ins[k]]?;
    let lo = p.checked_mul(xl)?.checked_add(n.checked_mul(xh)?)?.checked_mul(rest)?;
    let hi = p.checked_mul(xh)?.checked_add(n.checked_mul(xl)?)?.checked_mul(rest)?;
    let (lo, hi) = (lo.min(hi), lo.max(hi));
    if lo.abs().max(hi.abs()) > hq() { return None; }
    Some((lo, hi))
}

fn einsum_raw(b: &Bounds, ins: &[usize], nt: i128) -> Option<i128> {
    let mut p: i128 = 1;
    for &i in ins { p = p.checked_mul(mag(b.iv[i])?)?; }
    p.checked_mul(nt)
}
fn einsum_fwd(b: &Bounds, ins: &[usize], nt: i128) -> Iv {
    let mut acc = Some((1i128, 1i128));
    for &i in ins { acc = mul(acc?, b.iv[i]?); }
    let r = times(acc, nt)?;
    if mag(Some(r))? > hq() { return None; }
    Some(r)
}

/// One rule application at node n (forward or backward); true if changed.
fn step(dag: &Dag, g: &Graph, b: &mut Bounds, n: usize, ln2: i128, lookups: &std::collections::BTreeMap<usize, usize>, fwd: bool) -> bool {
    let nd = &dag.nodes[n];
    let o = nd.outputs[0];
    let mut ch = false;
    match &nd.kind {
        K::Add(_) | K::Sub(_) => {
            let s = if matches!(nd.kind, K::Sub(_)) { -1 } else { 1 };
            let (a, c) = (nd.inputs[0], nd.inputs[1]);
            if fwd { ch |= b.set(o, lin(b.iv[a], b.iv[c], s), if s > 0 { "fwd add" } else { "fwd sub" }); }
            else {
                // a = o - s*c ; c = s*(o - a)
                ch |= b.set(a, lin(b.iv[o], b.iv[c], -s), "bwd add/sub");
                let t = lin(b.iv[o], b.iv[a], -1);
                let t = if s > 0 { t } else { t.map(|(l, h)| (-h, -l)) };
                ch |= b.set(c, t, "bwd add/sub");
            }
        }
        K::Einsum(es) => {
            let (nt, view) = einsum_terms(&es.equation, &es.input_shapes);
            if view {
                if fwd { ch |= b.set(o, b.iv[nd.inputs[0]], "view"); } else { ch |= b.set(nd.inputs[0], b.iv[o], "view (bwd)"); }
            } else if fwd {
                ch |= b.set(o, einsum_fwd(b, &nd.inputs, nt), "fwd contraction");
                if !b.lit_cache.contains_key(&n) { let v = lit_contract(dag, nd, es); b.lit_cache.insert(n, v); }
                if let Some(v) = lit_bound(b, &nd.inputs, b.lit_cache[&n]) { ch |= b.set(o, Some(v), "fwd contraction (literal rows)"); }
            }
        }
        K::ChangeShape(_) => {
            let (i, ni, no) = (nd.inputs[0], g.nvars[nd.inputs[0]], g.nvars[o]);
            if fwd { ch |= b.set(o, if no > ni { with0(b.iv[i]) } else { b.iv[i] }, "view"); }
            else if no >= ni { ch |= b.set(i, b.iv[o], "view (bwd)"); }
        }
        K::ScaleDown(_) | K::ScaleUp(_) => {
            let l = lookups[&n] as u32;
            let s = 1i128 << l;
            let (x, y) = (nd.inputs[0], o);
            if matches!(nd.kind, K::ScaleUp(_)) {
                // y = S x exactly (remainder authenticated to S/2).
                if fwd { if let Some(v) = times(b.iv[x], s) { if mag(Some(v)).unwrap() <= hq() { ch |= b.set(y, Some(v), "fwd upscale"); } } }
            } else if fwd {
                // Once x and y are certified and Eq. rounding_nowrap holds, the
                // rounding identity is an integer one: y = floor((x + floor(S/2))/S).
                if let (Some((xl, xh)), Some(ym)) = (b.iv[x], mag(b.iv[y])) {
                    let z = xl.abs().max(xh.abs());
                    if z + s * ym + s / 2 < Q as i128 {
                        ch |= b.set(y, mk((xl + s / 2).div_euclid(s), (xh + s / 2).div_euclid(s)), "rounding of certified x");
                    }
                }
            } else {
                // x = S y + r - floor(S/2), r in [0, S).
                if let Some((yl, yh)) = b.iv[y] {
                    if (yl.abs().max(yh.abs())) * s + s <= hq() { ch |= b.set(x, mk(s * yl - s / 2, s * yh + s - 1 - s / 2), "bwd rescale"); }
                }
                if l == 0 { ch |= b.set(y, b.iv[x], "S=1 rescale"); }
            }
        }
        K::ExpHelper(eh) => {
            let kmax = ln2 * ((1i128 << eh.num_bits) - 1);
            let x = nd.inputs[0];
            if fwd {
                // r = x + ln2*k, k in [0, 2^b).
                ch |= b.set(o, lin(b.iv[x], Some((0, kmax)), 1), "fwd exp");
                if nd.outputs.len() > 1 { ch |= b.set(nd.outputs[1], Some((0, (1i128 << eh.num_bits) - 1)), "exp selector index"); }
            } else { ch |= b.set(x, lin(b.iv[o], Some((0, kmax)), -1), "bwd exp"); }
        }
        K::TwoPow(_) => { if fwd { ch |= b.set(o, Some((0, 1 << 15)), "two-pow table"); } }
        K::NonNegative(nn) => { if fwd { ch |= b.set(nd.inputs[0], Some((0, (1i128 << nn.table_size_log) - 1)), &format!("range table 2^{}", nn.table_size_log)); } }
        K::ProductZeroCheck(_) => {
            let (a, c) = (nd.inputs[0], nd.inputs[1]);
            if dag.program_literals.get(&c).map(|v| v.iter().all(|&x| x != 0)).unwrap_or(false) { ch |= b.set(a, Some((0, 0)), "zero check"); }
            if dag.program_literals.get(&a).map(|v| v.iter().all(|&x| x != 0)).unwrap_or(false) { ch |= b.set(c, Some((0, 0)), "zero check"); }
        }
        k if is_view(k) => {
            if fwd {
                let mut v = Some((0i128, 0i128));
                for &i in &nd.inputs { v = hull(v, b.iv[i]); }
                ch |= b.set(o, v, "view");
            } else if matches!(k, K::ZeroPad(_) | K::ZeroPadAsym(_) | K::ZeroPad3D(_) | K::Concat(_) | K::Replicate2x2(_)) {
                for &i in &nd.inputs.clone() { ch |= b.set(i, b.iv[o], "view (bwd)"); }
            }
        }
        k if conv_terms(k).is_some() => {
            if fwd {
                let nt = conv_terms(k).unwrap();
                let v = (|| times(mul(b.iv[nd.inputs[0]]?, b.iv[nd.inputs[1]]?), nt))();
                if let Some(v) = v { if mag(Some(v)).unwrap() <= hq() { ch |= b.set(o, Some(v), "fwd conv"); } }
            }
        }
        _ => {}
    }
    ch | step_sup(dag, g, b, n, fwd)
}

/// Support-restricted rules (entries where a public 0/1 mask is one).
fn step_sup(dag: &Dag, g: &Graph, b: &mut Bounds, n: usize, fwd: bool) -> bool {
    let nd = &dag.nodes[n];
    let o = nd.outputs[0];
    let same = |x: usize| g.nvars[x] == g.nvars[o] && g.shapes[x] == g.shapes[o];
    let mut ch = false;
    match &nd.kind {
        K::Einsum(es) => {
            // o = x * M elementwise, M a public 0/1 literal: x on supp(M) = o.
            let (lhs, rhs) = es.equation.split_once("->").unwrap();
            let specs: Vec<&str> = lhs.split(',').map(|t| t.trim()).collect();
            if !fwd && specs.len() == 2 && specs.iter().all(|t| *t == rhs.trim()) && nd.inputs.iter().all(|&i| same(i)) {
                for k in 0..2 {
                    let (x, mlit) = (nd.inputs[k], nd.inputs[1 - k]);
                    if dag.program_literals.get(&mlit).map(|v| v.iter().all(|&t| t == 1)).unwrap_or(false) {
                        // All-ones literal: x = o entrywise.
                        let v = b.iv[o]; ch |= b.set(x, v, "bwd x*1");
                        for m2 in b.masks_on(&[o]) { let v = b.on(o, m2); ch |= b.set_sup(x, m2, v); }
                    } else if let Some(m) = b.mask_id(dag, mlit) {
                        let v = b.iv[o]; ch |= b.set_sup(x, m, v);
                        // o bounded on supp(m2) gives x on supp(m) and supp(m2).
                        for m2 in b.masks_on(&[o]) {
                            if let Some(mm) = b.meet_mask(m, m2) { let v = b.on(o, m2); ch |= b.set_sup(x, mm, v); }
                        }
                    }
                }
            }
        }
        K::Add(_) | K::Sub(_) => {
            // A broadcast operand contributes its full bound (valid on any
            // subset of entries); only same-shape operands get support bounds.
            let (a, c) = (nd.inputs[0], nd.inputs[1]);
            let (sa, sc) = (same(a), same(c));
            let s = if matches!(nd.kind, K::Sub(_)) { -1 } else { 1 };
            let ms = b.masks_on(&[a, c, o]);
            for m in ms {
                let on = |b: &Bounds, e: usize, ok: bool| if ok { b.on(e, m) } else { b.iv[e] };
                if fwd { let v = lin(on(b, a, sa), on(b, c, sc), s); ch |= b.set_sup(o, m, v); }
                else {
                    if sa { let v = lin(b.on(o, m), on(b, c, sc), -s); ch |= b.set_sup(a, m, v); }
                    if sc {
                        let t = lin(b.on(o, m), on(b, a, sa), -1);
                        let t = if s > 0 { t } else { t.map(|(l, h)| (-h, -l)) };
                        ch |= b.set_sup(c, m, t);
                    }
                }
            }
        }
        K::ChangeShape(_) => {
            let i = nd.inputs[0];
            if g.nvars[i] == g.nvars[o] {
                for m in b.masks_on(&[i, o]) {
                    if fwd { let v = b.on(i, m); ch |= b.set_sup(o, m, v); } else { let v = b.on(o, m); ch |= b.set_sup(i, m, v); }
                }
            }
        }
        K::ScaleDown(_) => {
            let x = nd.inputs[0];
            if !fwd && same(x) {
                let l = g.lookups[&n] as u32; let s = 1i128 << l;
                for m in b.masks_on(&[o]) {
                    if let Some((yl, yh)) = b.on(o, m) {
                        if yl.abs().max(yh.abs()) * s + s <= hq() { ch |= b.set_sup(x, m, mk(s * yl - s / 2, s * yh + s - 1 - s / 2)); }
                    }
                }
            }
        }
        _ => {}
    }
    ch
}

/// MORPHEUS_RESCALE_STATS=1 (after a forward pass): the widest |x| and |y| of
/// every ScaleDown, grouped by producer of x and fate of y. Reads request
/// values; a sizing aid for the rescale bound, not part of the statement.
pub fn rescale_stats(dag: &Dag, wit: &[Vec<Witness>]) {
    let w = |e: usize| -> u32 {
        if wit[e].is_empty() || wit[e][0].data.is_none() { return 0; }
        crate::morpheus::piop::witness_vals(&wit[e][0]).iter().map(|&v| { let a = if v > Q / 2 { Q - v } else { v }; 64 - a.leading_zeros() }).max().unwrap_or(0)
    };
    let mut groups: std::collections::BTreeMap<String, (usize, u32, u32)> = Default::default();
    for &n in &dag.topo {
        let nd = &dag.nodes[n];
        if !matches!(nd.kind, K::ScaleDown(_)) { continue; }
        let (x, y) = (nd.inputs[0], nd.outputs[0]);
        // Quotient bound added by DagBuilder::certify_rescales (y + B -> NonNegative).
        let added = dag.consumers[y].iter().any(|&c| matches!(dag.nodes[c].kind, K::Add(_))
            && dag.consumers[dag.nodes[c].outputs[0]].iter().any(|&d| matches!(dag.nodes[d].kind, K::NonNegative(_))));
        let key = format!("{} x from {} ; y -> {}", if added { "[added]" } else { "[own]  " }, prod_fate(dag, x), fate(dag, y));
        let g = groups.entry(key).or_insert((0, 0, 0));
        g.0 += 1; g.1 = g.1.max(w(x)); g.2 = g.2.max(w(y));
    }
    let (mut mx, mut my) = (0, 0);
    for (k, (c, bx, by)) in &groups { println!("[rescale-stats] {c:4} |x|<2^{bx:2} |y|<2^{by:2}  {k}"); mx = mx.max(*bx); my = my.max(*by); }
    println!("[rescale-stats] widest |x| 2^{mx}, |y| 2^{my}");
}
