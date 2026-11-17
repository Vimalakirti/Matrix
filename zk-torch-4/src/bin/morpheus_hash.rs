//! Digest of a Morpheus model instance (graph + every witness value) for
//! byte-identity checks across source versions (CPU only, no CUDA init).
use zk_torch_4::morpheus::models;

fn main() {
    if std::env::var_os("MORPHEUS_FULL_AUX").is_none() { std::env::set_var("MORPHEUS_LEAN_AUX", "1"); }
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let inst = models::build(&model, 0);
    let mut h: u64 = 0xcbf29ce484222325;
    let mut mix = |x: u64| { for b in x.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } };
    for n in &inst.dag.nodes {
        for b in format!("{:?}", n.kind).bytes() { mix(b as u64); }
        for &e in &n.inputs { mix(e as u64); }
        for &e in &n.outputs { mix(e as u64); }
    }
    let mut wh = 0u64;
    for ws in &inst.wit {
        for w in ws {
            for &s in &w.shape { mix(s as u64); }
            mix(w.sf as u64); mix(w.role as u64);
            if let Some(d) = w.data.as_ref() { for f in d.evaluations().iter() { mix(f.0); wh += 1; } }
        }
    }
    for &e in &inst.public { mix(e as u64); }
    println!("{model}: {} nodes {} edges {} values digest {h:016x}", inst.dag.nodes.len(), inst.dag.num_edges(), wh);
}
