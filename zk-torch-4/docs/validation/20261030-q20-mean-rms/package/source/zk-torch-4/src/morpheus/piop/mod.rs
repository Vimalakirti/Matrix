//! Masked graph PIOP (paper §"A Unified PIOP Compiler for AI Inference"),
//! over K = F_q[Θ]/(Θ^8 − 3), on zk-torch-4 DAGs.
//!
//! Obligations are augmented claims (e, r, η, z): z = T̃_e(r) + η(W_G), with
//! η a linear functional on the graph-mask bank W_G (a committed bit source
//! of K slots, each 8 residues × 64 bits).
//!
//! Reductions, in reverse topological order, each producer once:
//!   * Einsum: the masked contraction sumcheck of Eq. graph_mask: free
//!     indices used by one operand are fixed by partial evaluation, shared
//!     free indices keep an eq factor, summed indices are summed; one extra
//!     variable Z, fresh pads a_s per operand, mask polynomial M_c, one
//!     combined correction row q_c (Eq. combined_correction).
//!   * Add / Sub: masked linear check with one correction row.
//!   * Views (ChangeShape, pure permutation einsums): point transport.
//!   * Lookups (ScaleDown/Up, ExpHelper, TwoPow, NonNegative): the selector
//!     is a committed one-hot (Shout chunks of 2^tcl, bit-typed in the link);
//!     every read is linear in it, so value/row-sum-one checks are linear rows
//!     the PCS proves directly.
//!   * Multi-consumer merges: same-point claims by a pad-difference row,
//!     others by the masked β-merge sumcheck.
//! Committed tensors (weights, inputs, advice, selectors, W_G) end as mixed
//! linear rows for the joint opening.
//!
//! Prover and verifier run the same driver: the prover computes and records
//! every message, the verifier replays them; both derive identical
//! transcripts and rows.

pub mod cv;
pub mod gpu;
pub mod layout;
pub mod tensor;
#[cfg(test)]
mod exp_tests;

use crate::basicblock::BasicBlockType;
use crate::dag::{Dag, PolyType, Role, Witness};
use crate::morpheus::field8::{eq_table, invq, K8};
use crate::morpheus::link::{interpolate, Row, Term};
use crate::morpheus::transcript::{Item, MTranscript, Stage};
use crate::util::arith::{f_to_int, log2_ceil};
use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
use almost_goldilocks_cuda::DeviceBuffer;
use layout::{value_pw, Layout, TKey};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tensor::{from_i128, mle, partial_eval};

pub type Lin = BTreeMap<u32, K8>;

fn lin1(slot: u32, c: K8) -> Lin { let mut m = Lin::new(); m.insert(slot, c); m }
fn lin_add(a: &mut Lin, b: &Lin, c: &K8) {
    for (k, v) in b { let e = a.entry(*k).or_insert(K8::ZERO); *e = e.add(&v.mul(c)); }
}
fn lin_scaled(b: &Lin, c: &K8) -> Lin { let mut a = Lin::new(); lin_add(&mut a, b, c); a }
fn kb(x: i128) -> K8 { K8::from_base(from_i128(x)) }
fn inv2() -> K8 { K8::from_base((Q + 1) / 2) }

/// Prover self-checks of every emitted claim (MORPHEUS_SELFCHECK=1): an
/// independent MLE evaluation per claim, for tests and debugging.
pub fn selfcheck() -> bool {
    static ON: once_cell::sync::Lazy<bool> = once_cell::sync::Lazy::new(|| std::env::var("MORPHEUS_SELFCHECK").ok().as_deref() == Some("1"));
    *ON
}

#[derive(Clone, Debug)]
pub struct AClaim { pub point: Vec<K8>, pub pad: Lin, pub z: K8 }

#[derive(Clone, Debug)]
pub struct Config {
    /// Shout chunk width (bits) of lookup selectors.
    pub tcl: usize,
    /// log2 of the graph-mask slot capacity.
    pub slots_log: usize,
    /// log2 of the declared digit planes of every non-bit request tensor
    /// (public; 4 = values below 2^16 in magnitude).
    pub req_planes_log: usize,
}

impl Default for Config { fn default() -> Self { Config { tcl: 4, slots_log: 17, req_planes_log: 4 } } }

// ---------------------------------------------------------------- statement

/// Public description of the graph instance (shapes, roles, lookup ranges).
pub struct Graph<'a> {
    pub dag: &'a Dag,
    pub shapes: Vec<Vec<usize>>,
    /// Full public tensor types retained for canonical statement encoding.
    pub edge_meta: Vec<(crate::dag::DataType, crate::dag::PolyType, usize, crate::dag::Role)>,
    pub nvars: Vec<usize>,
    pub committed: Vec<bool>,
    /// Fixed model tensors (Role::Constant): shared by every request, the
    /// candidates for deferred fixed-weight opening.
    pub model: Vec<bool>,
    /// Graph inputs whose declared data type is Bool (one-hot selector inputs).
    pub bool_typed: Vec<bool>,
    /// Lookup nodes and their table bit widths.
    pub lookups: BTreeMap<usize, usize>,
    /// Edges whose values are public (request inputs), with their values.
    pub public: HashMap<usize, Arc<Vec<u64>>>,
    /// Terminal outputs of the graph (public, bound by a root claim).
    pub outputs: Vec<(usize, Arc<Vec<u64>>)>,
    pub cfg: Config,
    pub protocol_conv_pp: bool,
}

fn advice_kind(k: &BasicBlockType) -> bool {
    matches!(k, BasicBlockType::DivConst(_) | BasicBlockType::SoftmaxConst(_) | BasicBlockType::SigmoidConst(_) | BasicBlockType::RMSReciprocal(_) | BasicBlockType::RMSBucket(_)
        // Vision advice, constrained by range checks / product zero checks
        // wired by the builder (the same relation as the previous prover).
        | BasicBlockType::ReLUHelper(_) | BasicBlockType::MaxPoolHelper(_) | BasicBlockType::GeneralMaxPoolHelper(_)
        | BasicBlockType::PillarMaxPool(_) | BasicBlockType::InstanceNorm3D(_) | BasicBlockType::ScatterToBEV(_) | BasicBlockType::GatherFromGrid(_))
}

/// ln2 in the ExpHelper fixed point.
fn exp_ln2() -> i128 { crate::basicblock::exp::exp_ln2(*crate::SF_LOG) }

/// 2^h·Θ^a for W_G plane b = 64a + h.
fn wg_monomials() -> &'static Arc<Vec<K8>> {
    static M: once_cell::sync::Lazy<Arc<Vec<K8>>> = once_cell::sync::Lazy::new(|| Arc::new((0..512).map(|b| {
        let mut e = [0u64; 8];
        e[b / 64] = ((1u128 << (b % 64)) % Q as u128) as u64;
        K8(e)
    }).collect()));
    &M
}

pub fn witness_vals(w: &Witness) -> Arc<Vec<u64>> {
    // Canonical residues (= from_i128(f_to_int(f))) without the i128 detour.
    let d = w.data.as_ref().expect("witness without data");
    match d.try_evaluations_ref() {
        Some(ev) => Arc::new(ev.par_iter().map(|f| f.reduce().0).collect()),
        None => Arc::new(d.evaluations().par_iter().map(|f| f.reduce().0).collect()),
    }
}

impl<'a> Graph<'a> {
    pub fn new(dag: &'a Dag, wit: &[Vec<Witness>], cfg: Config) -> Self {
        let protocol_conv_pp = cv::conv_pp();
        let ne = dag.num_edges();
        let shapes: Vec<Vec<usize>> = (0..ne).map(|e| wit[e][0].shape.clone()).collect();
        let edge_meta = (0..ne).map(|e| { let w = &wit[e][0]; (w.data_type, w.poly_type, w.sf, w.role) }).collect();
        let nvars: Vec<usize> = shapes.iter().map(|s| s.iter().map(|&d| log2_ceil(d)).sum()).collect();
        let mut committed = vec![false; ne];
        let model: Vec<bool> = (0..ne).map(|e| wit[e][0].role == Role::Constant).collect();
        // Declared Bool graph inputs (computed edges may inherit the tag).
        let bool_typed: Vec<bool> = (0..ne).map(|e| dag.producers[e].is_none() && wit[e][0].data_type == crate::dag::DataType::Bool).collect();
        for e in 0..ne {
            let w = &wit[e][0];
            committed[e] = match w.role {
                Role::Constant | Role::Input => true,
                Role::Auxiliary => w.poly_type == PolyType::Dense,
                Role::Output => false,
            };
        }
        let mut lookups = BTreeMap::new();
        for nd in &dag.nodes {
            if advice_kind(&nd.kind) { committed[nd.outputs[0]] = true; }
            // Full-convolution / grand-product advice of the previous conv
            // proofs is not part of this reduction.
            if matches!(nd.kind, BasicBlockType::Conv1D(_) | BasicBlockType::Conv2D(_) | BasicBlockType::Conv3D(_) | BasicBlockType::DepthwiseConv2D(_)) {
                for &o in &nd.outputs[1..] { committed[o] = false; }
                // The polynomial-product baseline commits Y_full.
                if protocol_conv_pp && matches!(nd.kind, BasicBlockType::Conv2D(_)) && nd.outputs.len() == 2 { committed[nd.outputs[1]] = true; }
            }
            match &nd.kind {
                BasicBlockType::NonNegative(nn) => { lookups.insert(nd.id, nn.table_size_log); }
                BasicBlockType::ScaleDown(sd) => { lookups.insert(nd.id, wit[nd.inputs[0]][0].sf - sd.output_sf); }
                BasicBlockType::ScaleUp(su) => { lookups.insert(nd.id, su.output_sf - wit[nd.inputs[0]][0].sf); }
                BasicBlockType::ExpHelper(eh) => { lookups.insert(nd.id, eh.num_bits); }
                _ => {}
            }
        }
        let public = dag.program_literals.iter().map(|(&e,v)| {
            assert!(dag.producers[e].is_none() && wit[e][0].role == Role::Input,
                "program literal must be a public input leaf");
            (e, Arc::new(v.clone()))
        }).collect();
        // Terminal outputs: unconsumed first outputs of compute nodes (not the
        // zero certificate of a product check, not lookup gates).
        let outputs: Vec<(usize, Arc<Vec<u64>>)> = (0..ne).filter(|&e| {
            let w = &wit[e][0];
            dag.consumers[e].is_empty() && w.role == Role::Output && w.data.is_some() && !committed[e]
                && dag.producers[e].map(|p| { let nd = &dag.nodes[p]; nd.outputs[0] == e && !matches!(nd.kind, BasicBlockType::ProductZeroCheck(_) | BasicBlockType::NonNegative(_)) }).unwrap_or(false)
        }).map(|e| (e, witness_vals(&wit[e][0]))).collect();
        let g = Graph { dag, shapes, edge_meta, nvars, committed, model, bool_typed, lookups, public, outputs, cfg, protocol_conv_pp };
        g.validate_exp_profile().expect("invalid canonical exp guard graph");
        g
    }

    /// Fail closed when a manually assembled/mutated graph omits any canonical
    /// exp gate. Called by both construction and the verifier statement codec.
    pub(crate) fn validate_exp_profile(&self) -> Result<(), String> {
        let d = self.dag;
        let helpers: Vec<_> = d.nodes.iter().filter(|n| matches!(n.kind, BasicBlockType::ExpHelper(_))).collect();
        if helpers.len() != d.exp_guard_nodes.len() { return Err("exp guard registry mismatch".into()); }
        let sub = |e: usize| -> Result<(usize,usize),String> {
            let n = d.producers.get(e).and_then(|x| *x).and_then(|n| d.nodes.get(n)).ok_or("exp guard missing subtraction")?;
            if !matches!(n.kind, BasicBlockType::Sub(_)) || n.inputs.len()!=2 || n.outputs.as_slice()!=[e] {
                return Err("exp guard must be subtraction".into());
            }
            Ok((n.inputs[0],n.inputs[1]))
        };
        for n in helpers {
            let bits = match &n.kind { BasicBlockType::ExpHelper(h) => h.num_bits, _=>unreachable!() };
            if !(1..=8).contains(&bits) || *crate::SF_LOG>30 || n.inputs.len()!=1 || n.outputs.len()!=2 {
                return Err("unsupported exp profile".into());
            }
            if self.lookups.get(&n.id) != Some(&bits) || self.cfg.tcl==0 {
                return Err("exp selector width/configuration mismatch".into());
            }
            if self.cfg.tcl<4 && bits>self.cfg.tcl && d.nodes.iter().any(|tp|
                matches!(tp.kind,BasicBlockType::TwoPow(_)) && tp.inputs.first()==n.outputs.get(1)) {
                return Err("chunked TwoPow requires first chunk width at least four".into());
            }
            let x=n.inputs[0]; let r=n.outputs[0];
            if self.edge_meta[x].2 != *crate::SF_LOG || self.edge_meta[r].2 != *crate::SF_LOG || self.shapes[x]!=self.shapes[r] || self.committed[r] {
                return Err("exp scale/shape mismatch".into());
            }
            let l=exp_ln2(); let bound=(1i128<<bits)*l-l/2-1;
            let width=|v:i128| (128-(v as u128).leading_zeros() as usize).max(1);
            let ids=d.exp_guard_nodes.get(&n.id).ok_or("missing exp guards")?;
            let mut es=[0;4];
            for j in 0..4 {
                let guard=d.nodes.get(ids[j]).ok_or("missing exp range node")?;
                let expected=if j<2 {width(bound)} else {width(l-1)};
                if !matches!(&guard.kind, BasicBlockType::NonNegative(b) if b.table_size_log==expected)
                    || guard.inputs.len()!=1 || !d.range.contains(&ids[j]) || self.lookups.get(&ids[j])!=Some(&expected) {
                    return Err("wrong exp range guard".into());
                }
                es[j]=guard.inputs[0];
                if self.shapes[es[j]]!=self.shapes[x] || self.committed[es[j]] { return Err("exp guard shape mismatch".into()); }
            }
            let (z,a)=sub(es[0])?; let (b,nx)=sub(es[1])?;
            let (h,rr)=sub(es[2])?; let (lm1,rho)=sub(es[3])?;
            if (a,nx,rr,rho)!=(x,es[0],r,es[2]) {return Err("disconnected exp guard".into());}
            for (e,v) in [(z,0),(b,bound),(h,l/2),(lm1,l-1)] {
                let vals=d.program_literals.get(&e).ok_or("exp bound is not an immutable literal")?;
                if self.shapes[e]!=self.shapes[x] || self.edge_meta[e].2!=*crate::SF_LOG
                    || vals.len()!=1usize.checked_shl(self.nvars[x] as u32).ok_or("exp shape overflow")?
                    || vals.iter().any(|&a| a!=v as u64) || self.public.get(&e).map(|a|a.as_slice())!=Some(vals.as_slice()) {
                    return Err("wrong exp literal bound".into());
                }
            }
        }
        Ok(())
    }

    /// Stop committing tensors the PIOP never claims: non-public edges whose
    /// every consumer is an advice node (e.g. InstanceNorm γ/β feeding only
    /// the statistics advice). Call after `public` is set.
    pub fn prune_unclaimed(&mut self) {
        for e in 0..self.dag.num_edges() {
            if !self.committed[e] || self.public.contains_key(&e) { continue; }
            if self.dag.producers[e].map(|p| advice_kind(&self.dag.nodes[p].kind)).unwrap_or(false) { continue; }
            // The polynomial-product baseline's committed full convolution.
            if self.protocol_conv_pp && self.dag.producers[e].map(|p| matches!(self.dag.nodes[p].kind, BasicBlockType::Conv2D(_))).unwrap_or(false) { continue; }
            if self.dag.consumers[e].iter().all(|&c| advice_kind(&self.dag.nodes[c].kind)) { self.committed[e] = false; }
        }
    }

    fn chunks(&self, node: usize) -> Vec<usize> {
        let l = self.lookups[&node];
        let mut v = Vec::new();
        let mut done = 0;
        while done < l { let w = self.cfg.tcl.min(l - done); v.push(w); done += w; }
        if v.is_empty() { v.push(1); }
        v
    }

    /// Lookup node whose selector a TwoPow reads (the producer of its input).
    fn exp_node_of(&self, two_pow_node: usize) -> usize {
        let e = self.dag.nodes[two_pow_node].inputs[0];
        self.dag.producers[e].expect("TwoPow input has a producer")
    }
}

// ---------------------------------------------------------------- prover data

/// Prover-only data: dense edge values and lookup selector indices.
pub struct ProverData {
    pub vals: Vec<Option<Arc<Vec<u64>>>>,
    pub sel: HashMap<usize, Vec<u64>>,
    pub bank: Vec<K8>,
    pub cache: gpu::EdgeCache,
}

impl ProverData {
    pub fn new(g: &Graph, wit: &[Vec<Witness>], bank: Vec<K8>) -> Self {
        let ne = g.dag.num_edges();
        let vals: Vec<Option<Arc<Vec<u64>>>> = (0..ne).into_par_iter().map(|e| {
            let w = &wit[e][0];
            if w.poly_type == PolyType::Dense && w.data.is_some() { Some(witness_vals(w)) } else { None }
        }).collect();
        let iv = |e: usize, i: usize| -> i128 { let x = vals[e].as_ref().unwrap()[i]; if x > Q / 2 { x as i128 - Q as i128 } else { x as i128 } };
        let ln2 = exp_ln2();
        let sel: HashMap<usize, Vec<u64>> = g.lookups.par_iter().map(|(&node, &l)| {
            let nd = &g.dag.nodes[node];
            let n = 1usize << g.nvars[nd.inputs[0]];
            let t: Vec<u64> = (0..n).into_par_iter().map(|i| {
                let t: i128 = match &nd.kind {
                    BasicBlockType::NonNegative(_) => iv(nd.inputs[0], i),
                    BasicBlockType::ScaleDown(_) => { let s = 1i128 << l; iv(nd.inputs[0], i) - s * iv(nd.outputs[0], i) + s / 2 }
                    BasicBlockType::ScaleUp(_) => { let s = 1i128 << l; s * iv(nd.inputs[0], i) - iv(nd.outputs[0], i) + s / 2 }
                    BasicBlockType::ExpHelper(_) => { let d = iv(nd.outputs[0], i) - iv(nd.inputs[0], i); assert_eq!(d % ln2, 0); d / ln2 }
                    _ => unreachable!(),
                };
                if !(t >= 0 && t < (1i128 << l)) {
                    let src = g.dag.producers[nd.inputs[0]].map(|p| { let pn = &g.dag.nodes[p]; format!("{:?} <- [{}]", pn.kind, pn.inputs.iter().map(|&e| format!("{:?}", g.dag.producers[e].map(|q| format!("{:?}", g.dag.nodes[q].kind).chars().take(40).collect::<String>()))).collect::<Vec<_>>().join(", ")) }).unwrap_or_default();
                    let sv = |e: usize| -> i128 { vals[e].as_ref().map(|v| { let x = v[i.min(v.len() - 1)]; if x > Q / 2 { x as i128 - Q as i128 } else { x as i128 } }).unwrap_or(0) };
                    let mut dump = String::new();
                    if let Some(p) = g.dag.producers[nd.inputs[0]] {
                        for &e in &g.dag.nodes[p].inputs {
                            dump += &format!(" | edge {e} {:?} sf {} = {}", wit[e][0].shape, wit[e][0].sf, sv(e));
                            if let Some(q) = g.dag.producers[e] { for &e2 in &g.dag.nodes[q].inputs { dump += &format!(" [in {e2} {:?} sf {} = {}]", wit[e2][0].shape, wit[e2][0].sf, sv(e2)); } }
                        }
                    }
                    panic!("lookup node {node} ({:?}): index {t} outside [0, 2^{l}) at {i}; input from {src};{dump}", nd.kind);
                }
                t as u64
            }).collect();
            (node, t)
        }).collect();
        ProverData { vals, sel, bank, cache: gpu::EdgeCache::default() }
    }
    /// Upload every dense edge's values ahead of the PIOP (they are fixed
    /// before any challenge); the cache serves them later.
    pub fn prefetch(&self) {
        // Reverse topological use order is roughly reverse edge order; stop at
        // half the budget so on-demand uploads do not evict the prefetch.
        for e in (0..self.vals.len()).rev() {
            if let Some(v) = &self.vals[e] {
                if v.len() >= gpu::threshold() {
                    if self.cache.bytes() + v.len() * 8 > gpu::edge_budget() / 2 { break; }
                    let _ = self.cache.get(e, v);
                }
            }
        }
    }
    fn v(&self, e: usize) -> &Arc<Vec<u64>> { self.vals[e].as_ref().unwrap_or_else(|| panic!("no dense values for edge {e}")) }
    fn dev(&self, e: usize) -> Arc<almost_goldilocks_cuda::DeviceBuffer<u64>> { self.cache.get(e, self.v(e)) }
}

/// Layout specs: (key, index vars, log2 planes, bit-typed).
pub fn tensor_specs(g: &Graph, pd: &ProverData) -> Vec<(TKey, usize, usize, bool)> {
    let mut specs = Vec::new();
    for e in 0..g.dag.num_edges() {
        if !g.committed[e] { continue; }
        let v = pd.v(e);
        let maxb = v.par_iter().map(|&x| { let a = if x > Q / 2 { Q - x } else { x }; 64 - a.leading_zeros() }).max().unwrap_or(0).max(1) as usize;
        let (p, is_bits) = if g.model[e] {
            // Model tensors: widths declared once at model registration (part
            // of the registered model's public type), independent of requests.
            (log2_ceil(maxb), v.par_iter().all(|&x| x <= 1))
        } else if g.public.contains_key(&e) {
            // Public structural/request data may declare its exact width:
            // these values are already in the public statement, not secrets.
            (log2_ceil(maxb), v.par_iter().all(|&x| x <= 1))
        } else {
            // Request tensors: a public rule, never the request's values. Bit
            // type is declared by the producer (sign and selection helpers) or
            // by the input's declared Bool type; every other request tensor
            // gets 2^req_planes_log planes. The honest prover checks the fit.
            let bits = g.bool_typed[e] || g.dag.producers[e].map(|n| matches!(g.dag.nodes[n].kind,
                BasicBlockType::ReLUHelper(_) | BasicBlockType::RMSBucket(_) | BasicBlockType::PillarMaxPool(_) | BasicBlockType::ScatterToBEV(_))).unwrap_or(false);
            if bits {
                assert!(v.par_iter().all(|&x| x <= 1), "edge {e}: declared bit tensor holds a non-bit value");
                (0, true)
            } else {
                let p = g.cfg.req_planes_log;
                assert!(maxb <= 1 << p, "edge {e}: value of {maxb} bits exceeds the declared {} request planes", 1 << p);
                (p, false)
            }
        };
        specs.push((TKey::Edge(e), g.nvars[e], p, is_bits));
        if std::env::var("MORPHEUS_SPECS").is_ok() {
            let prod = g.dag.producers[e].map(|n| format!("{:?}", g.dag.nodes[n].kind).split('(').next().unwrap().to_string()).unwrap_or("-".into());
            eprintln!("[spec] e={e} model={} prod={prod} nvars={} maxb={maxb} p={p} bits={is_bits}", g.model[e], g.nvars[e]);
        }
    }
    for &node in g.lookups.keys() {
        let n = g.nvars[g.dag.nodes[node].inputs[0]];
        for (c, &w) in g.chunks(node).iter().enumerate() { specs.push((TKey::Sel { node, chunk: c }, n, w, true)); }
    }
    specs.push((TKey::Wg, g.cfg.slots_log, 9, true));
    specs
}

/// Raw digit sources of the layout.
pub fn fill_sources(g: &Graph, pd: &ProverData, lay: &Layout, ell: usize) -> Vec<Vec<i8>> {
    fill_sources_in(g, pd, lay, ell, 0..lay.n_sources)
}

/// The raw sources `range` of the layout (tensors packed elsewhere are skipped).
pub fn fill_sources_in(g: &Graph, pd: &ProverData, lay: &Layout, ell: usize, range: std::ops::Range<usize>) -> Vec<Vec<i8>> {
    use layout::RawPtrs;
    let mut raw: Vec<Vec<i8>> = range.clone().into_par_iter().map(|_| vec![0i8; 1usize << ell]).collect();
    let mut ptrs: Vec<RawPtrs> = vec![RawPtrs(std::ptr::null_mut(), 0); lay.n_sources];
    for (k, r) in raw.iter_mut().enumerate() { ptrs[range.start + k] = RawPtrs(r.as_mut_ptr(), r.len()); }
    lay.tensors.par_iter().filter(|t| t.pieces.iter().all(|pc| range.contains(&pc.region.source))).for_each(|t| {
        match t.key {
            TKey::Edge(e) => {
                let v = pd.v(e).clone();
                lay.fill_ptr(&t.key, &ptrs, &move |b, i| {
                    let x = v[i];
                    let (a, s) = if x > Q / 2 { (Q - x, -1i8) } else { (x, 1i8) };
                    if (a >> b) & 1 == 1 { s } else { 0 }
                });
            }
            TKey::Sel { node, chunk } => {
                let t = &pd.sel[&node];
                let shift: usize = g.chunks(node)[..chunk].iter().sum();
                let w = g.chunks(node)[chunk];
                lay.fill_ptr(&t_key(node, chunk), &ptrs, &move |b, i| (((t[i] >> shift) & ((1 << w) - 1)) as usize == b) as i8);
            }
            TKey::Wg => {
                let bank = &pd.bank;
                lay.fill_ptr(&TKey::Wg, &ptrs, &move |b, s| {
                    if s >= bank.len() { return 0; }
                    let (a, h) = (b / 64, b % 64);
                    ((bank[s].0[a] >> h) & 1) as i8
                });
            }
            TKey::Carry => {}
        }
    });
    raw
}
/// The carry source (local index 0) of a deferred request: slot k holds the
/// hidden weight evaluation carries[k], 8 residues × 64 bits like W_G.
pub fn fill_carry(lay: &Layout, carries: &[K8], ell: usize, first_source: usize) -> Vec<i8> {
    use layout::RawPtrs;
    let mut raw = vec![0i8; 1usize << ell];
    let t = lay.get(&TKey::Carry);
    assert!(t.pieces.iter().all(|pc| pc.region.source == first_source), "carry bank spans one source");
    let mut ptrs: Vec<RawPtrs> = vec![RawPtrs(std::ptr::null_mut(), 0); first_source + 1];
    ptrs[first_source] = RawPtrs(raw.as_mut_ptr(), raw.len());
    lay.fill_ptr(&TKey::Carry, &ptrs, &move |b, s| {
        if s >= carries.len() { return 0; }
        let (a, h) = (b / 64, b % 64);
        ((carries[s].0[a] >> h) & 1) as i8
    });
    raw
}

fn t_key(node: usize, chunk: usize) -> TKey { TKey::Sel { node, chunk } }

// ---------------------------------------------------------------- driver

pub struct Ctx<'a> {
    prover: bool,
    pub tr: MTranscript,
    pub proof: Vec<K8>,
    pos: usize,
    pub ok: bool,
    pub fail: Option<String>,
    pub rows: Vec<Row>,
    bank: &'a [K8],
    next_slot: u32,
    /// Kind of every allocated slot: 1 = pad, 2 = sumcheck-message mask.
    slot_kind: Vec<u8>,
    chal: u64,
    lay: &'a Layout,
    slots_log: usize,
    pub stats: BTreeMap<&'static str, (usize, f64)>,
    /// Fresh W_G slots by kind.
    pub census: BTreeMap<&'static str, u64>,
    /// Deferred fixed-weight opening (None: monolithic).
    pub defer: Option<Defer>,
}

/// Deferred fixed-weight opening state of one request. A claim
/// (e, r, η, z) on a model tensor e becomes
///   request row:   carry_k + η(W_G) = z            (opened with the request)
///   deferred row:  T̃_e(r) − carry_k = 0             (opened at finalization)
/// where carry_k = T̃_e(r) is a fresh slot of the request's carry bank, a
/// committed source opened with the request and again at finalization.
pub struct Defer {
    pub carry_log: usize,
    pub carries: Vec<K8>,
    pub rows: Vec<Row>,
    next: u32,
}

impl Defer {
    pub fn new(carry_log: usize) -> Self { Defer { carry_log, carries: Vec::new(), rows: Vec::new(), next: 0 } }
    pub fn used(&self) -> u32 { self.next }
}

struct ScOut { x: Vec<K8>, rho: K8, v: Vec<K8>, pads: Vec<u32> }

/// Prover tables of a masked sumcheck: on the host or on the device.
pub enum Tables { Host(Vec<Vec<K8>>), Dev(Vec<gpu::KTab>) }

impl<'a> Ctx<'a> {
    pub fn prover(tr: MTranscript, bank: &'a [K8], lay: &'a Layout, slots_log: usize) -> Self {
        Ctx { prover: true, tr, proof: Vec::new(), pos: 0, ok: true, fail: None, rows: Vec::new(), bank, next_slot: 0, slot_kind: Vec::new(), chal: 0, lay, slots_log, stats: BTreeMap::new(), census: BTreeMap::new(), defer: None }
    }
    pub fn verifier(tr: MTranscript, proof: Vec<K8>, lay: &'a Layout, slots_log: usize) -> Self {
        Ctx { prover: false, tr, proof, pos: 0, ok: true, fail: None, rows: Vec::new(), bank: &[], next_slot: 0, slot_kind: Vec::new(), chal: 0, lay, slots_log, stats: BTreeMap::new(), census: BTreeMap::new(), defer: None }
    }
    fn send(&mut self, v: Option<K8>) -> K8 {
        let x = if self.prover { let x = v.unwrap(); self.proof.push(x); x } else {
            match self.proof.get(self.pos) { Some(&x) => { self.pos += 1; x } None => { self.reject("proof too short"); K8::ZERO } }
        };
        self.tr.absorb_k(Item::GraphClaim, &[x]);
        x
    }
    fn send_vec(&mut self, len: usize, v: Option<Vec<K8>>) -> Vec<K8> {
        let xs = if self.prover { let x = v.unwrap(); assert_eq!(x.len(), len); self.proof.extend_from_slice(&x); x } else {
            if self.pos + len > self.proof.len() { self.reject("proof too short"); vec![K8::ZERO; len] }
            else { let x = self.proof[self.pos..self.pos + len].to_vec(); self.pos += len; x }
        };
        self.tr.absorb_k(Item::RoundPoly, &xs);
        xs
    }
    fn reject(&mut self, why: &str) {
        if self.prover { panic!("prover self-check failed: {why}"); }
        if self.ok { self.fail = Some(why.to_string()); }
        self.ok = false;
    }
    fn check(&mut self, c: bool, why: &str) { if !c { self.reject(why); } }
    /// Every proof message was read (verifier).
    pub fn consumed(&self) -> bool { self.pos == self.proof.len() }
    fn chal(&mut self) -> K8 { self.chal += 1; self.tr.challenge_k(Stage::GraphReduction, 0, self.chal) }
    fn chal01(&mut self) -> K8 { self.chal += 1; self.tr.challenge_k_not01(Stage::GraphReduction, 0, self.chal) }
    fn chals(&mut self, n: usize) -> Vec<K8> { (0..n).map(|_| self.chal()).collect() }
    /// One fresh W_G slot (one K mask = 512 bounded digit coefficients),
    /// counted by kind for the per-workload census.
    fn alloc(&mut self, kind: &'static str) -> u32 {
        *self.census.entry(kind).or_default() += 1;
        let s = self.next_slot;
        assert!((s as usize) < (1usize << self.slots_log), "graph-mask slots exhausted; raise slots_log");
        self.next_slot += 1;
        self.slot_kind.push(if kind == "round/degree mask" { 2 } else { 1 });
        s
    }
    /// Allocation discipline, checked by both parties on every claim row:
    /// it refers only to pads already allocated, never to a future slot or to
    /// a sumcheck-message slot (those appear only in terminal corrections).
    fn check_pad_row(&mut self, pad: &Lin) {
        let ok = pad.keys().all(|&k| self.slot_kind.get(k as usize) == Some(&1));
        self.check(ok, "claim row refers to a future slot or a message slot");
    }
    pub fn slots_used(&self) -> u32 { self.next_slot }
    /// Census line: slots by kind, used / committed slots and digit
    /// coefficients (512 per K mask).
    pub fn census_line(&self) -> String {
        let kinds: Vec<String> = self.census.iter().map(|(k, v)| format!("{k} {v}")).collect();
        let cap = 1u64 << self.slots_log;
        format!("graph-mask census: {} used of {} committed slots ({} of {} digit coefficients) [{}]",
            self.next_slot, cap, self.next_slot as u64 * 512, cap * 512, kinds.join(", "))
    }

    /// Terms of coeff·carry_k (slot k of the carry bank).
    fn carry_terms(&self, k: u32, coeff: K8) -> Vec<Term> {
        let n = self.defer.as_ref().unwrap().carry_log;
        let r: Vec<K8> = (0..n).map(|j| if (k >> j) & 1 == 1 { K8::ONE } else { K8::ZERO }).collect();
        self.lay.terms_shared(&TKey::Carry, &r, wg_monomials(), coeff)
    }

    /// A claim on model tensor e under deferral: split into the request row
    /// and the deferred row through a fresh carry slot.
    fn defer_claim(&mut self, model_terms: Vec<Term>, pad: &Lin, z: K8) {
        let prover = self.prover;
        let v = if prover { Some(z.sub(&self.lin_val(pad))) } else { None };
        let d = self.defer.as_mut().unwrap();
        let k = d.next;
        assert!((k as usize) < (1usize << d.carry_log), "carry slots exhausted; raise the carry bank size");
        d.next += 1;
        if let Some(v) = v { d.carries.push(v); }
        let ct = self.carry_terms(k, K8::ONE);
        self.row(ct, pad, z);
        let mut dt = model_terms;
        dt.extend(self.carry_terms(k, K8::ONE.neg()));
        self.defer.as_mut().unwrap().rows.push(Row { terms: dt, y: K8::ZERO });
    }
    fn tick(&mut self, name: &'static str, t0: std::time::Instant) {
        let e = self.stats.entry(name).or_insert((0, 0.0));
        e.0 += 1; e.1 += t0.elapsed().as_secs_f64();
    }
    fn lin_val(&self, l: &Lin) -> K8 { l.iter().fold(K8::ZERO, |a, (s, c)| a.add(&self.bank[*s as usize].mul(c))) }

    /// Row: Σ terms(sources) + lin(W_G) = y.
    fn row(&mut self, mut terms: Vec<Term>, lin: &Lin, y: K8) {
        let t0 = std::time::Instant::now();
        // Slot plane b = 64a + h carries 2^h·Θ^a: one shared weight vector,
        // the coefficient goes into the term scale.
        let (lay, slots_log) = (self.lay, self.slots_log);
        let mono = wg_monomials();
        let ents: Vec<(u32, K8)> = lin.iter().filter(|(_, c)| !c.is_zero()).map(|(&s, c)| (s, *c)).collect();
        let mk = |&(s, c): &(u32, K8)| -> Vec<Term> {
            let r: Vec<K8> = (0..slots_log).map(|j| if (s >> j) & 1 == 1 { K8::ONE } else { K8::ZERO }).collect();
            lay.terms_shared(&TKey::Wg, &r, mono, c)
        };
        let wg: Vec<Vec<Term>> = if ents.len() >= 64 { ents.par_iter().map(mk).collect() } else { ents.iter().map(mk).collect() };
        for t in wg { terms.extend(t); }
        self.rows.push(Row { terms, y });
        self.tick(".row_wg", t0);
    }

    /// The masked sumcheck of Eqs. graph_mask / combined_correction.
    /// `tables` (prover): [public factor (if has_pub)], private operands; all
    /// of length 2^n. Returns the point, ρ, the child values and pad slots.
    #[allow(clippy::too_many_arguments)]
    fn masked_sc(&mut self, n: usize, npriv: usize, has_pub: bool, z0: K8, eta0: &Lin,
                 mut tables: Option<Tables>, pub_at: &dyn Fn(&[K8]) -> K8) -> ScOut {
        self.check_pad_row(eta0);
        let t0 = std::time::Instant::now();
        let ntab = npriv + has_pub as usize;
        let d_old = ntab.max(1);
        let d_fin = (2 * npriv).max(1);
        let nn = n + 1;
        let degs: Vec<usize> = (0..nn).map(|i| if i < n { d_old } else { d_fin }).collect();
        let b: Vec<Vec<u32>> = degs.iter().map(|&d| (0..d).map(|_| self.alloc("round/degree mask")).collect()).collect();
        let pads: Vec<u32> = (0..npriv).map(|_| self.alloc("operand pad")).collect();
        let bv: Vec<Vec<K8>> = if self.prover { b.iter().map(|v| v.iter().map(|&s| self.bank[s as usize]).collect()).collect() } else { vec![] };
        let eta_v = if self.prover { self.lin_val(eta0) } else { K8::ZERO };
        let half = inv2();
        let m_term = |bk: &[K8], u: &K8| -> K8 {
            let mut acc = K8::ZERO; let mut up = K8::ONE;
            for c in bk { up = up.mul(u); acc = acc.add(&c.mul(&up.sub(&half))); }
            acc
        };
        let mut claim = z0;
        let mut mpref = K8::ZERO;
        let mut x = Vec::with_capacity(n);
        let mut inv2p = K8::ONE; // 2^-(i+1) after update
        for i in 0..n {
            inv2p = inv2p.mul(&half);
            let d = degs[i];
            let tr0 = std::time::Instant::now();
            let on_dev = matches!(tables.as_ref(), Some(Tables::Dev(_)));
            let g = if self.prover {
                let s: Vec<K8> = match tables.as_ref().unwrap() {
                Tables::Dev(t) => gpu::sc_round(t, d),
                Tables::Host(tabs) => {
                let len = tabs[0].len();
                (0..len / 2).into_par_iter().with_min_len(512).fold(|| vec![K8::ZERO; d + 1], |mut acc, k| {
                    let mut vals: Vec<K8> = tabs.iter().map(|t| t[2 * k]).collect();
                    let diffs: Vec<K8> = tabs.iter().map(|t| t[2 * k + 1].sub(&t[2 * k])).collect();
                    for (tt, a) in acc.iter_mut().enumerate() {
                        if tt > 0 { for (v, df) in vals.iter_mut().zip(&diffs) { *v = v.add(df); } }
                        let p = vals.iter().skip(1).fold(vals[0], |a, v| a.mul(v));
                        *a = a.add(&p);
                    }
                    acc
                }).reduce(|| vec![K8::ZERO; d + 1], |a, b| a.iter().zip(&b).map(|(x, y)| x.add(y)).collect())
                }};
                let scale = K8::from_base(1u64 << (nn - 1 - i));
                Some((0..=d).map(|t| {
                    let tk = K8::from_base(t as u64);
                    s[t].add(&inv2p.mul(&eta_v)).add(&scale.mul(&mpref.add(&m_term(&bv[i], &tk))))
                }).collect::<Vec<_>>())
            } else { None };
            self.tick(if on_dev { ".r_dev_eval" } else { ".r_host_eval" }, tr0);
            let tr1 = std::time::Instant::now();
            let g = self.send_vec(d + 1, g);
            self.check(g[0].add(&g[1]) == claim, "sumcheck round sum");
            let r = self.chal();
            claim = interpolate(&g, &r);
            self.tick(".r_transcript", tr1);
            let tr2 = std::time::Instant::now();
            if self.prover {
                mpref = mpref.add(&m_term(&bv[i], &r));
                match tables.as_mut().unwrap() {
                    Tables::Host(ts) => for t in ts.iter_mut() {
                        let h = t.len() / 2;
                        *t = (0..h).into_par_iter().with_min_len(1024).map(|k| t[2 * k].add(&r.mul(&t[2 * k + 1].sub(&t[2 * k])))).collect();
                    },
                    Tables::Dev(ts) => for t in ts.iter_mut() { *t = gpu::sc_fold(t, &r); },
                }
            }
            self.tick(if on_dev { ".r_dev_fold" } else { ".r_host_fold" }, tr2);
            x.push(r);
        }
        self.tick(".sc_rounds", t0);
        let t1 = std::time::Instant::now();
        inv2p = inv2p.mul(&half); // 2^-N
        let d = d_fin;
        let fin_vals: Vec<K8> = if self.prover {
            match tables.as_ref().unwrap() { Tables::Host(ts) => ts.iter().map(|t| t[0]).collect(), Tables::Dev(ts) => ts.iter().map(|t| t.first()).collect() }
        } else { vec![] };
        let g = if self.prover {
            let pubv = if has_pub { fin_vals[0] } else { K8::ONE };
            let priv_v = &fin_vals[has_pub as usize..];
            let av: Vec<K8> = pads.iter().map(|&s| self.bank[s as usize]).collect();
            Some((0..=d).map(|t| {
                let tk = K8::from_base(t as u64);
                let tt = tk.mul(&tk.sub(&K8::ONE));
                let prod = priv_v.iter().zip(&av).fold(pubv, |a, (p, ad)| a.mul(&p.add(&tt.mul(ad))));
                half.mul(&prod).add(&inv2p.mul(&eta_v)).add(&mpref.add(&m_term(&bv[n], &tk)))
            }).collect::<Vec<_>>())
        } else { None };
        let g = self.send_vec(d + 1, g);
        self.check(g[0].add(&g[1]) == claim, "sumcheck final-round sum");
        let rho = self.chal01();
        let rr = rho.mul(&rho.sub(&K8::ONE));
        let v: Vec<K8> = (0..npriv).map(|s| {
            let val = if self.prover { Some(fin_vals[has_pub as usize + s].add(&rr.mul(&self.bank[pads[s] as usize]))) } else { None };
            self.send(val)
        }).collect();
        // Correction q_c = M_c(u) + 2^-N η_0, one authenticated row.
        let mut u = x.clone();
        u.push(rho);
        let mut q_lin = lin_scaled(eta0, &inv2p);
        for (i, bs) in b.iter().enumerate() {
            let mut up = K8::ONE;
            for &s in bs { up = up.mul(&u[i]); lin_add(&mut q_lin, &lin1(s, K8::ONE), &up.sub(&half)); }
        }
        let q = self.send(if self.prover { Some(self.lin_val(&q_lin)) } else { None });
        self.row(vec![], &q_lin, q);
        let pubv = pub_at(&x);
        let expect = half.mul(&v.iter().fold(pubv, |a, y| a.mul(y))).add(&q);
        self.check(interpolate(&g, &rho) == expect, "sumcheck terminal identity");
        self.tick(".sc_final", t1);
        let e = self.stats.entry("masked_sc").or_insert((0, 0.0));
        e.0 += 1; e.1 += t0.elapsed().as_secs_f64();
        ScOut { x, rho, v, pads }
    }
}

// ---------------------------------------------------------------- graph walk

pub struct Walk<'g, 'a> {
    g: &'g Graph<'a>,
    pd: Option<&'g ProverData>,
    pending: Vec<Vec<AClaim>>,
}

fn eq_pt(a: &[K8], b: &[K8]) -> K8 {
    a.iter().zip(b).fold(K8::ONE, |acc, (x, y)| acc.mul(&x.mul(y).add(&K8::ONE.sub(x).mul(&K8::ONE.sub(y)))))
}

impl<'g, 'a> Walk<'g, 'a> {
    pub fn new(g: &'g Graph<'a>, pd: Option<&'g ProverData>) -> Self {
        Walk { g, pd, pending: vec![Vec::new(); g.dag.num_edges()] }
    }

    fn mle_e(&self, e: usize, r: &[K8]) -> K8 {
        let pd = self.pd.unwrap();
        if gpu::enabled() && (1usize << r.len()) >= gpu::threshold() { gpu::mle(&pd.dev(e), r) } else { mle(pd.v(e), r) }
    }

    /// A claim on edge e: a row if e is committed, else pending.
    fn push(&mut self, ctx: &mut Ctx, e: usize, c: AClaim) {
        ctx.check_pad_row(&c.pad);
        if ctx.prover && selfcheck() {
            let truth = self.mle_e(e, &c.point).add(&ctx.lin_val(&c.pad));
            assert!(truth == c.z, "prover produced a false claim on edge {e}");
        }
        if self.g.committed[e] {
            let t0 = std::time::Instant::now();
            let p = ctx.lay.get(&TKey::Edge(e)).p;
            let terms = ctx.lay.terms(&TKey::Edge(e), &c.point, &value_pw(p), K8::ONE);
            if ctx.defer.is_some() && self.g.model[e] { ctx.defer_claim(terms, &c.pad, c.z); } else { ctx.row(terms, &c.pad, c.z); }
            ctx.tick(".push_row", t0);
        } else {
            self.pending[e].push(c);
        }
    }

    /// Selector read Σ_c 2^{shift_c} Σ_{i,t} eq(r,i) A_c(i,t)·t (or a custom table).
    fn sel_terms(&self, ctx: &Ctx, node: usize, r: &[K8], coeff: K8, table: Option<&dyn Fn(usize) -> i128>) -> Vec<Term> {
        let mut terms = Vec::new();
        let mut shift = 0;
        for (c, &w) in self.g.chunks(node).iter().enumerate() {
            let pw: Vec<K8> = (0..1usize << w).map(|t| match table {
                Some(f) => kb(f(t)),
                None => kb((t as i128) << shift),
            }).collect();
            terms.extend(ctx.lay.terms(&t_key(node, c), r, &pw, coeff));
            shift += w;
        }
        terms
    }

    /// Roots: NonNegative gates, selector row-sum-one, public inputs.
    pub fn roots(&mut self, ctx: &mut Ctx) {
        let lookups: Vec<usize> = self.g.lookups.keys().copied().collect();
        for &node in &lookups {
            let nd = &self.g.dag.nodes[node];
            let n = self.g.nvars[nd.inputs[0]];
            for c in 0..self.g.chunks(node).len() {
                let r = ctx.chals(n);
                let w = self.g.chunks(node)[c];
                let terms = ctx.lay.terms(&t_key(node, c), &r, &vec![K8::ONE; 1 << w], K8::ONE);
                ctx.row(terms, &Lin::new(), K8::ONE);
            }
            if let BasicBlockType::NonNegative(_) = nd.kind {
                let x = nd.inputs[0];
                let r = ctx.chals(n);
                let a = ctx.alloc("root pad");
                let z = ctx.send(if ctx.prover { Some(self.mle_e(x, &r).add(&ctx.bank[a as usize])) } else { None });
                let terms = self.sel_terms(ctx, node, &r, K8::ONE, None);
                ctx.row(terms, &lin1(a, K8::ONE), z);
                self.push(ctx, x, AClaim { point: r, pad: lin1(a, K8::ONE), z });
            }
        }
        self.pzc_roots(ctx);
        // Public outputs: Ỹ(r) at a random r.
        let outs = self.g.outputs.clone();
        for (e, vals) in outs {
            let r = ctx.chals(self.g.nvars[e]);
            let y = mle(&vals, &r);
            self.push(ctx, e, AClaim { point: r, pad: Lin::new(), z: y });
        }
        let mut pubs: Vec<usize> = self.g.public.keys().copied().collect();
        pubs.sort();
        for e in pubs {
            let r = ctx.chals(self.g.nvars[e]);
            let y = mle(&self.g.public[&e], &r);
            let p = ctx.lay.get(&TKey::Edge(e)).p;
            let terms = ctx.lay.terms(&TKey::Edge(e), &r, &value_pw(p), K8::ONE);
            ctx.row(terms, &Lin::new(), y);
        }
    }

    /// Merge the pending claims of a non-committed edge into one.
    fn merge(&mut self, ctx: &mut Ctx, e: usize) -> Option<AClaim> {
        let cs = std::mem::take(&mut self.pending[e]);
        if cs.is_empty() { return None; }
        if cs.len() == 1 { return Some(cs.into_iter().next().unwrap()); }
        let n = self.g.nvars[e];
        // Every multi-claim merge is the registered beta merge (also when
        // the points coincide).
        let beta = ctx.chal();
        let mut bp = K8::ONE;
        let (mut z0, mut eta0) = (K8::ZERO, Lin::new());
        let mut weights = Vec::new();
        for c in &cs {
            z0 = z0.add(&bp.mul(&c.z));
            lin_add(&mut eta0, &c.pad, &bp);
            weights.push(bp);
            bp = bp.mul(&beta);
        }
        let tables = if ctx.prover {
            if gpu::enabled() && (1usize << n) >= gpu::threshold() {
                let mut pubt = gpu::KTab::zeroed(1 << n);
                for (c, w) in cs.iter().zip(&weights) { gpu::axpy(&mut pubt, &gpu::eq(&c.point), w); }
                let t = gpu::lift(&self.pd.unwrap().dev(e), 1 << n);
                Some(Tables::Dev(vec![pubt, t]))
            } else {
                let mut pubt = vec![K8::ZERO; 1 << n];
                for (c, w) in cs.iter().zip(&weights) {
                    let et = eq_table(&c.point);
                    pubt.par_iter_mut().zip(et.par_iter()).for_each(|(a, b)| *a = a.add(&b.mul(w)));
                }
                let t: Vec<K8> = self.pd.unwrap().v(e).par_iter().map(|&v| K8::from_base(v)).collect();
                Some(Tables::Host(vec![pubt, t]))
            }
        } else { None };
        let pts: Vec<Vec<K8>> = cs.iter().map(|c| c.point.clone()).collect();
        let out = ctx.masked_sc(n, 1, true, z0, &eta0, tables, &|x: &[K8]| pts.iter().zip(&weights).fold(K8::ZERO, |a, (p, w)| a.add(&w.mul(&eq_pt(p, x)))));
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        Some(AClaim { point: out.x, pad: lin1(out.pads[0], rr), z: out.v[0] })
    }

    /// Process the whole DAG.
    pub fn run(&mut self, ctx: &mut Ctx) {
        let t0 = std::time::Instant::now();
        self.roots(ctx);
        ctx.tick("~roots", t0);
        let topo: Vec<usize> = self.g.dag.topo.clone();
        for &node in topo.iter().rev() {
            let nd = &self.g.dag.nodes[node];
            let kind = &nd.kind;
            if matches!(kind, BasicBlockType::NonNegative(_)) || advice_kind(kind) { continue; }
            let out = nd.outputs[0];
            if self.g.committed[out] { continue; }
            let tm = std::time::Instant::now();
            let c = match self.merge(ctx, out) { Some(c) => c, None => continue };
            ctx.tick("~merge", tm);
            let t0 = std::time::Instant::now();
            if ctx.prover && selfcheck() {
                let truth = self.mle_e(out, &c.point).add(&ctx.lin_val(&c.pad));
                eprintln!("[selfcheck] node {node} {:?} out {out}: claim {}", nd.kind, if truth == c.z { "ok" } else { "FALSE" });
            }
            let label = self.reduce(ctx, node, c);
            let e = ctx.stats.entry(label).or_insert((0, 0.0));
            e.0 += 1; e.1 += t0.elapsed().as_secs_f64();
        }
        for e in 0..self.pending.len() {
            if !self.pending[e].is_empty() {
                panic!("edge {e} has {} unreduced claims (no producer handled it; producer {:?})", self.pending[e].len(), self.g.dag.producers[e].map(|p| format!("{:?}", self.g.dag.nodes[p].kind).chars().take(80).collect::<String>()));
            }
        }
    }

    fn child_lin(&self, ctx: &mut Ctx, e: usize, r: &[K8]) -> (u32, K8) {
        let a = ctx.alloc("linear/merge child pad");
        let v = ctx.send(if ctx.prover { Some(self.mle_e(e, r).add(&ctx.bank[a as usize])) } else { None });
        (a, v)
    }

    fn reduce(&mut self, ctx: &mut Ctx, node: usize, c: AClaim) -> &'static str {
        let nd = self.g.dag.nodes[node].clone();
        match &nd.kind {
            BasicBlockType::Einsum(es) => { self.einsum(ctx, &nd.inputs, &es.equation, &es.input_shapes, &es.output_shape, c); "einsum" }
            BasicBlockType::Add(_) | BasicBlockType::Sub(_) => {
                let sub = matches!(nd.kind, BasicBlockType::Sub(_));
                let (a, b, o) = (nd.inputs[0], nd.inputs[1], nd.outputs[0]);
                let pa = broadcast_point(&self.g.shapes[a], &self.g.shapes[o], &c.point);
                let pb = broadcast_point(&self.g.shapes[b], &self.g.shapes[o], &c.point);
                let (sa, va) = self.child_lin(ctx, a, &pa);
                let (sb, vb) = self.child_lin(ctx, b, &pb);
                let sgn = if sub { K8::ONE.neg() } else { K8::ONE };
                // z0 − v_a ∓ v_b = (η_0 − a_a ∓ a_b)(W_G).
                let mut l = c.pad.clone();
                lin_add(&mut l, &lin1(sa, K8::ONE), &K8::ONE.neg());
                lin_add(&mut l, &lin1(sb, K8::ONE), &sgn.neg());
                ctx.row(vec![], &l, c.z.sub(&va).sub(&sgn.mul(&vb)));
                self.push(ctx, a, AClaim { point: pa, pad: lin1(sa, K8::ONE), z: va });
                self.push(ctx, b, AClaim { point: pb, pad: lin1(sb, K8::ONE), z: vb });
                "add"
            }
            BasicBlockType::ChangeShape(_) => {
                let (i, o) = (nd.inputs[0], nd.outputs[0]);
                let (ni, no) = (self.g.nvars[i], self.g.nvars[o]);
                if no >= ni {
                    let f = c.point[ni..].iter().fold(K8::ONE, |a, x| a.mul(&K8::ONE.sub(x)));
                    let fi = f.inv();
                    self.push(ctx, i, AClaim { point: c.point[..ni].to_vec(), pad: lin_scaled(&c.pad, &fi), z: c.z.mul(&fi) });
                } else {
                    let mut p = c.point.clone();
                    p.resize(ni, K8::ZERO);
                    self.push(ctx, i, AClaim { point: p, pad: c.pad, z: c.z });
                }
                "view"
            }
            BasicBlockType::ScaleDown(_) | BasicBlockType::ScaleUp(_) => {
                let up = matches!(nd.kind, BasicBlockType::ScaleUp(_));
                let l = self.g.lookups[&node];
                let s = kb(1i128 << l);
                let hs = if l == 0 { K8::ZERO } else { kb(1i128 << (l - 1)) };
                let x = nd.inputs[0];
                let (sx, vx) = self.child_lin(ctx, x, &c.point);
                let terms = self.sel_terms(ctx, node, &c.point, K8::ONE, None);
                let (lin, y) = if !up {
                    // v_x − S z0 + S/2 = (a_x − S η0)(W_G) + rem(r).
                    let mut lin = lin1(sx, K8::ONE);
                    lin_add(&mut lin, &c.pad, &s.neg());
                    (lin, vx.sub(&s.mul(&c.z)).add(&hs))
                } else {
                    // S v_x − z0 + S/2 = (S a_x − η0)(W_G) + rem(r).
                    let mut lin = lin1(sx, s);
                    lin_add(&mut lin, &c.pad, &K8::ONE.neg());
                    (lin, s.mul(&vx).sub(&c.z).add(&hs))
                };
                // Exact upscaling: authenticate the remainder, not only its range.
                if up { ctx.row(terms.clone(), &Lin::new(), hs); }
                ctx.row(terms, &lin, y);
                self.push(ctx, x, AClaim { point: c.point, pad: lin1(sx, K8::ONE), z: vx });
                "scale"
            }
            BasicBlockType::ExpHelper(_) => {
                // x = r_out − ln2·k: v_x − z0 = (a_x − η0)(W_G) − ln2·k(r).
                let x = nd.inputs[0];
                let (sx, vx) = self.child_lin(ctx, x, &c.point);
                let terms = self.sel_terms(ctx, node, &c.point, kb(-exp_ln2()), None);
                let mut lin = lin1(sx, K8::ONE);
                lin_add(&mut lin, &c.pad, &K8::ONE.neg());
                ctx.row(terms, &lin, vx.sub(&c.z));
                self.push(ctx, x, AClaim { point: c.point, pad: lin1(sx, K8::ONE), z: vx });
                "exp"
            }
            BasicBlockType::TwoPow(_) => {
                let en = self.g.exp_node_of(node);
                let chunks = self.g.chunks(en);
                let f = |t: usize| -> i128 { if t <= 15 { 1i128 << (15 - t) } else { 0 } };
                if chunks.len() == 1 {
                    // y(r) = Σ_t A(i,t) 2^{15−t}: terminal row.
                    let terms = self.sel_terms(ctx, en, &c.point, K8::ONE, Some(&f));
                    ctx.row(terms, &c.pad, c.z);
                } else {
                    // 2^{15−k} = L(i)·Π_{c≥1} H_c(i), L(i) = Σ_t A_0(i,t) 2^{15−t},
                    // H_c(i) = A_c(i, 0): masked sumcheck of eq(r,·)·L·Π H_c.
                    assert!(chunks[0] >= 4, "TwoPow over chunked selectors needs a first chunk of at least 4 bits");
                    let n = c.point.len();
                    let np = chunks.len();
                    let use_gpu = gpu::enabled() && (1usize << n) >= gpu::threshold();
                    let tables = if ctx.prover {
                        let t = &self.pd.unwrap().sel[&en];
                        let w0 = chunks[0];
                        let f0: Vec<K8> = (0..1usize << w0).map(|lo| kb(f(lo))).collect();
                        if use_gpu {
                            // Selector indices once; L and the H_c built on the device.
                            let ds = DeviceBuffer::from_slice(t).expect("selector upload");
                            let mut d = vec![gpu::eq(&c.point), gpu::sel_lookup(&ds, 0, w0, &f0)];
                            let mut shift = w0;
                            for &w in &chunks[1..] {
                                let ind: Vec<K8> = (0..1usize << w).map(|v| if v == 0 { K8::ONE } else { K8::ZERO }).collect();
                                d.push(gpu::sel_lookup(&ds, shift, w, &ind));
                                shift += w;
                            }
                            Some(Tables::Dev(d))
                        } else {
                            let mut v = vec![eq_table(&c.point)];
                            v.push(t.par_iter().map(|&k| f0[(k & ((1 << w0) - 1)) as usize]).collect());
                            let mut shift = w0;
                            for &w in &chunks[1..] {
                                v.push(t.par_iter().map(|&k| if (k >> shift) & ((1 << w) - 1) == 0 { K8::ONE } else { K8::ZERO }).collect());
                                shift += w;
                            }
                            Some(Tables::Host(v))
                        }
                    } else { None };
                    let r = c.point.clone();
                    let out = ctx.masked_sc(n, np, true, c.z, &c.pad, tables, &|x: &[K8]| eq_pt(&r, x));
                    let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
                    for k in 0..np {
                        let pw: Vec<K8> = (0..1usize << chunks[k]).map(|t| if k == 0 { kb(f(t)) } else if t == 0 { K8::ONE } else { K8::ZERO }).collect();
                        let terms = ctx.lay.terms(&t_key(en, k), &out.x, &pw, K8::ONE);
                        ctx.row(terms, &lin1(out.pads[k], rr), out.v[k]);
                    }
                }
                "two_pow"
            }
            other => match self.reduce_cv(ctx, node, c) {
                Some(l) => l,
                None => panic!("Morpheus PIOP: unsupported node kind {other:?}"),
            },
        }
    }

    /// Masked EinSumcheck.
    fn einsum(&mut self, ctx: &mut Ctx, inputs: &[usize], equation: &str, in_shapes: &[Vec<usize>], out_shape: &[usize], c: AClaim) {
        use crate::basicblock::einsum::{char_to_range, classify_einsum_indices_from_shapes};
        let (cls, specs, out_chars) = classify_einsum_indices_from_shapes(equation);
        let out_rng = char_to_range(&out_chars, out_shape);
        let in_rng: Vec<_> = specs.iter().zip(in_shapes).map(|(s, sh)| char_to_range(s, sh)).collect();
        let bits = |ch: char| -> usize {
            for r in &in_rng { if let Some(&(a, b)) = r.get(&ch) { return b - a; } }
            panic!("einsum char {ch}")
        };
        // Sumcheck variables: free_multi (output order) then summation.
        let mut sc_pos: HashMap<char, usize> = HashMap::new();
        let mut n_sc = 0;
        for &ch in cls.free_multi.iter().chain(cls.summation.iter()) { sc_pos.insert(ch, n_sc); n_sc += bits(ch); }
        let n_multi: usize = cls.free_multi.iter().map(|&ch| bits(ch)).sum();
        let r_multi: Vec<K8> = cls.free_multi.iter().flat_map(|&ch| { let (a, b) = out_rng[&ch]; c.point[a..b].to_vec() }).collect();
        // Pure view: one operand, nothing summed or shared.
        if inputs.len() == 1 && n_sc == 0 {
            let mut p = vec![K8::ZERO; self.g.nvars[inputs[0]]];
            for (&ch, &(a, b)) in &in_rng[0] { let (oa, _) = out_rng[&ch]; p[a..b].copy_from_slice(&c.point[oa..oa + (b - a)]); }
            self.push(ctx, inputs[0], AClaim { point: p, pad: c.pad, z: c.z });
            return;
        }
        let once: std::collections::HashSet<char> = cls.free_once.iter().copied().collect();
        // Child point layout per operand: (var, Some(out coord) | None = sc var index).
        let maps: Vec<Vec<(usize, Result<usize, usize>)>> = in_rng.iter().map(|rng| {
            let mut v = Vec::new();
            for (&ch, &(a, b)) in rng {
                for j in 0..b - a {
                    if once.contains(&ch) { v.push((a + j, Ok(out_rng[&ch].0 + j))); } else { v.push((a + j, Err(sc_pos[&ch] + j))); }
                }
            }
            v.sort_by_key(|x| x.0);
            v
        }).collect();
        let has_pub = n_multi > 0;
        let use_gpu = gpu::enabled() && (1usize << n_sc) >= gpu::threshold();
        let t_tab = std::time::Instant::now();
        let tables = if ctx.prover && use_gpu {
            let mut t = Vec::new();
            if has_pub { t.push(gpu::gather(&gpu::eq(&r_multi), &(0..n_multi).collect::<Vec<_>>(), n_sc)); }
            for (k, &inp) in inputs.iter().enumerate() {
                let dv = self.pd.unwrap().dev(inp);
                let fixed: Vec<(usize, K8)> = maps[k].iter().filter_map(|&(v, m)| m.ok().map(|o| (v, c.point[o]))).collect();
                let pe = gpu::partial_eval(&dv, self.g.nvars[inp], &fixed);
                let rest: Vec<usize> = maps[k].iter().filter_map(|&(_, m)| m.err()).collect();
                t.push(gpu::gather(&pe, &rest, n_sc));
            }
            Some(Tables::Dev(t))
        } else if ctx.prover {
            let mut t = Vec::new();
            if has_pub {
                let e = eq_table(&r_multi);
                let mask = (1usize << n_multi) - 1;
                t.push((0..1usize << n_sc).into_par_iter().map(|x| e[x & mask]).collect::<Vec<_>>());
            }
            for (k, &inp) in inputs.iter().enumerate() {
                let vals = self.pd.unwrap().v(inp);
                let n_t = self.g.nvars[inp];
                let fixed: Vec<(usize, K8)> = maps[k].iter().filter_map(|&(v, m)| m.ok().map(|o| (v, c.point[o]))).collect();
                let pe = partial_eval(vals, n_t, &fixed);
                let rest: Vec<usize> = maps[k].iter().filter_map(|&(_, m)| m.err()).collect(); // sc positions in operand order
                t.push((0..1usize << n_sc).into_par_iter().map(|x| {
                    let idx = rest.iter().enumerate().fold(0usize, |a, (j, &p)| a | (((x >> p) & 1) << j));
                    pe[idx]
                }).collect::<Vec<_>>());
            }
            Some(Tables::Host(t))
        } else { None };
        ctx.tick(".es_tables", t_tab);
        let out = ctx.masked_sc(n_sc, inputs.len(), has_pub, c.z, &c.pad, tables, &|x: &[K8]| eq_pt(&r_multi, &x[..n_multi]));
        let rr = out.rho.mul(&out.rho.sub(&K8::ONE));
        for (k, &inp) in inputs.iter().enumerate() {
            let p: Vec<K8> = maps[k].iter().map(|&(_, m)| match m { Ok(o) => c.point[o], Err(s) => out.x[s] }).collect();
            self.push(ctx, inp, AClaim { point: p, pad: lin1(out.pads[k], rr), z: out.v[k] });
        }
    }
}

/// Project an output point onto a broadcast operand (Add/Sub semantics).
fn broadcast_point(in_shape: &[usize], out_shape: &[usize], point: &[K8]) -> Vec<K8> {
    let matched = crate::util::shape::matched_axes(in_shape, out_shape).expect("broadcast shapes");
    let mut p = Vec::new();
    let mut off = 0;
    for (d, &sz) in out_shape.iter().enumerate() {
        let b = log2_ceil(sz);
        if matched.contains(&d) { p.extend_from_slice(&point[off..off + b]); }
        off += b;
    }
    p
}

/// Honest graph-mask bank: uniform K slots from the addressed coins.
pub fn sample_bank(coins: &crate::morpheus::coins::Coins, opening: u64, slots: usize) -> Vec<K8> {
    use crate::morpheus::coins::Purpose;
    let v = crate::morpheus::pcs_gpu::uniform_q_gpu(coins, Purpose::GraphMask, opening, 0, slots * 8);
    v.chunks(8).map(|c| K8(c.try_into().unwrap())).collect()
}

pub fn invq_pub(x: u64) -> u64 { invq(x) }

#[cfg(test)]
mod wg_tests {
    use super::*;
    #[test]
    fn monomial_plane_weights() {
        let c = K8(std::array::from_fn(|j| (0x9e3779b97f4a7c15u64.wrapping_mul(j as u64 + 7)) % Q));
        for b in [0usize, 1, 63, 64, 130, 300, 511] {
            let (a, h) = (b / 64, b % 64);
            let mut e = [0u64; 8];
            e[a] = ((1u128 << h) % Q as u128) as u64;
            let want = K8(e).mul(&c);
            let rot = K8(std::array::from_fn(|j| if j >= a { c.0[j - a] } else { ((c.0[j + 8 - a] as u128 * 3) % Q as u128) as u64 }));
            let p2 = ((1u128 << h) % Q as u128) as u64;
            assert_eq!(rot.scale(p2), want, "plane {b}");
        }
    }
}
