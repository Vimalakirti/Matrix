//! Model instances for the Morpheus prover: the same DAGs, weights and
//! inputs as the paper's previous evaluation binaries
//! (bench_streaming_oneshot_{gpt2,gptj,llama2,llama3}, bench_streaming_bert,
//! ...), so measurements compare at matched workloads.
//!
//! Decoder-only transformers prove a one-shot autoregressive generation:
//! embedding of the SEQ_LEN input tokens, the body, the LM head and the
//! argmax check binding every next token to its logits. The token
//! selectors are request inputs and public.

use crate::dag::{gptj::gpt_j_6b_hidden_ext, gpt2::gpt_2_small, llama::{llama3_8b_hidden, llama_2_7b_hidden_rope, rope_mats_plain}, bert::bert_large};
use crate::dag::{Dag, DagBuilder, DataType, EdgeId, Role, Witness};
use crate::util::arith::f_to_int;
use crate::SF_LOG;
use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn ev(k: &str, d: usize) -> usize { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
fn zeros(n: usize) -> Vec<F> { crate::zero_witness_vec(n) }
fn cw(shape: Vec<usize>) -> Witness {
    let n = shape.iter().map(|d| d.next_power_of_two()).product();
    Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant)
}

/// WEIGHTS=dense: quantized-model-like weights instead of the benchmark's
/// zero matrices. Matrices are uniform integers in [-WEIGHT_MAX, WEIGHT_MAX]
/// (default 63, i.e. |w| < 0.0625 at scale 2^10), norm gains are 1.0 and
/// biases 0. The default reproduces the previous evaluation's weights.
fn dense() -> bool { std::env::var("WEIGHTS").ok().as_deref() == Some("dense") }

/// A weight matrix (logical shape, padded storage, last axis lowest).
fn mat(shape: Vec<usize>) -> Witness {
    if !dense() { return cw(shape); }
    static CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let a = ev("WEIGHT_MAX", 63) as i64;
    let mut rng = StdRng::seed_from_u64(7 + CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let pads: Vec<usize> = shape.iter().map(|d| d.next_power_of_two()).collect();
    let n: usize = pads.iter().product();
    let mut data = zeros(n);
    // Fill logical entries only (padding stays zero).
    let mut idx = vec![0usize; shape.len()];
    loop {
        let mut flat = 0usize;
        let mut stride = 1usize;
        for d in (0..shape.len()).rev() { flat += idx[d] * stride; stride *= pads[d]; }
        let v: i64 = rng.gen_range(-a..=a);
        data[flat] = F(crate::morpheus::piop::tensor::from_i128(v as i128));
        let mut d = shape.len();
        loop {
            if d == 0 { break; }
            d -= 1;
            idx[d] += 1;
            if idx[d] < shape[d] { break; }
            idx[d] = 0;
            if d == 0 { d = usize::MAX; break; }
        }
        if d == usize::MAX || shape.is_empty() { break; }
    }
    Witness::new(shape, data, DataType::Float, *SF_LOG, Role::Constant)
}

/// WEIGHTS=real: a pretrained checkpoint exported by
/// scripts/export_transformer.py into REAL_WEIGHTS (one `<name>.bin` per
/// tensor: b"ZKRW", ndim:u32, shape:u32*ndim, row-major i32 values that are
/// already round(w * 2^sf), in the graph's slot layout). The export's
/// `meta.txt` must name the same scale as SF_LOG. Real prompts:
/// `prompts.bin` ([N, 16] token ids); request k uses prompt k mod N.
fn real() -> bool { std::env::var("WEIGHTS").ok().as_deref() == Some("real") }

struct Real { dir: std::path::PathBuf, model: String, hf: Option<Hf> }

/// Direct source: the HF safetensors snapshot named by meta.txt
/// (`source=hf`, `snapshot=<dir>`), converted in memory with exactly the
/// transforms of scripts/export_transformer.py (no on-disk int export; the
/// 7B-8B exports do not fit the available disks).
struct Hf {
    /// tensor name -> (file, dtype, shape, byte range)
    index: std::collections::HashMap<String, (std::path::PathBuf, String, Vec<usize>, (u64, u64))>,
    cfg: serde_yaml::Value,
}

impl Hf {
    fn open(snap: &std::path::Path) -> Hf {
        use std::io::Read;
        let mut index = std::collections::HashMap::new();
        let mut files: Vec<_> = std::fs::read_dir(snap).unwrap_or_else(|e| panic!("{}: {e}", snap.display()))
            .filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false)).collect();
        files.sort();
        assert!(!files.is_empty(), "{}: no safetensors", snap.display());
        for f in files {
            let mut fh = std::fs::File::open(&f).unwrap();
            let mut n = [0u8; 8]; fh.read_exact(&mut n).unwrap();
            let n = u64::from_le_bytes(n);
            let mut h = vec![0u8; n as usize]; fh.read_exact(&mut h).unwrap();
            let v: serde_yaml::Value = serde_yaml::from_slice(&h).expect("safetensors header");
            for (k, e) in v.as_mapping().unwrap() {
                let k = k.as_str().unwrap();
                if k == "__metadata__" { continue; }
                let dt = e["dtype"].as_str().unwrap().to_string();
                let shape = e["shape"].as_sequence().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let o = e["data_offsets"].as_sequence().unwrap();
                index.insert(k.to_string(), (f.clone(), dt, shape, (8 + n + o[0].as_u64().unwrap(), 8 + n + o[1].as_u64().unwrap())));
            }
        }
        let cfg = serde_yaml::from_str(&std::fs::read_to_string(snap.join("config.json")).expect("config.json")).expect("config.json");
        Hf { index, cfg }
    }
    fn cfg(&self, k: &str) -> Option<usize> { self.cfg[k].as_u64().map(|v| v as usize) }
    /// Tensor values as f64 (exact for f16 / bf16 / f32) with its shape.
    fn get(&self, name: &str) -> (Vec<usize>, Vec<f64>) {
        use rayon::prelude::*;
        use std::io::{Read, Seek, SeekFrom};
        let (f, dt, shape, (a, b)) = self.index.get(name).unwrap_or_else(|| panic!("checkpoint has no tensor {name}"));
        let mut fh = std::fs::File::open(f).unwrap();
        fh.seek(SeekFrom::Start(*a)).unwrap();
        let mut buf = vec![0u8; (*b - *a) as usize];
        fh.read_exact(&mut buf).unwrap();
        let v: Vec<f64> = match dt.as_str() {
            "F16" => buf.par_chunks_exact(2).map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => buf.par_chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64).collect(),
            "F32" => buf.par_chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
            d => panic!("{name}: unsupported dtype {d}"),
        };
        assert_eq!(v.len(), shape.iter().product::<usize>());
        (shape.clone(), v)
    }
}

fn f16_to_f64(h: u16) -> f64 {
    let s = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e { 0 => s * m * 2f64.powi(-24), 31 => if m == 0.0 { s * f64::INFINITY } else { f64::NAN }, _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15) }
}

/// round(w * 2^sf) half to even, as torch.round in the exporter.
fn quant(w: f64) -> i32 {
    let q = (w * (1u64 << *SF_LOG) as f64).round_ties_even();
    assert!(q.abs() < 2f64.powi(31), "weight does not fit i32 at scale 2^{}", *SF_LOG);
    q as i32
}

/// Quantize a [rows, cols] tensor, optionally selecting rows (`perm`) and
/// transposing (HF Linear [out, in] -> graph [in, out]).
fn quant_2d(shape: &[usize], v: &[f64], perm: Option<&[usize]>, transpose: bool) -> (Vec<usize>, Vec<i32>) {
    use rayon::prelude::*;
    let (r, c) = (shape[0], shape[1]);
    let row = |i: usize| perm.map(|p| p[i]).unwrap_or(i);
    if !transpose {
        let out = (0..r * c).into_par_iter().map(|k| quant(v[row(k / c) * c + k % c])).collect();
        (vec![r, c], out)
    } else {
        let out = (0..r * c).into_par_iter().map(|k| { let (i, j) = (k / r, k % r); quant(v[row(j) * c + i]) }).collect();
        (vec![c, r], out)
    }
}

/// Row order of a q/k weight so graph slot (h, 2i) holds the pair's second
/// component and (h, 2i+1) its first (see export_transformer.pair_swap_rows).
fn pair_swap_perm(heads: usize, hd: usize, rot: usize, half_split: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..hd).collect();
    for i in 0..rot / 2 {
        let (first, second) = if half_split { (i, i + rot / 2) } else { (2 * i, 2 * i + 1) };
        idx[2 * i] = second; idx[2 * i + 1] = first;
    }
    (0..heads * hd).map(|k| (k / hd) * hd + idx[k % hd]).collect()
}

impl Real {
    fn open(model: &str) -> Option<Real> {
        if !real() { return None; }
        let dir = std::path::PathBuf::from(std::env::var("REAL_WEIGHTS").expect("WEIGHTS=real needs REAL_WEIGHTS=<export dir>"));
        let meta = std::fs::read_to_string(dir.join("meta.txt")).unwrap_or_else(|e| panic!("{}/meta.txt: {e}", dir.display()));
        let get = |k: &str| meta.lines().find_map(|l| l.strip_prefix(&format!("{k}="))).map(|v| v.trim().to_string());
        assert_eq!(get("model").as_deref(), Some(model), "REAL_WEIGHTS holds a different model");
        let sf: usize = get("sf").and_then(|v| v.parse().ok()).expect("meta.txt: sf");
        assert_eq!(sf, *SF_LOG, "REAL_WEIGHTS was exported at scale 2^{sf}, the graph runs at 2^{}", *SF_LOG);
        let hf = (get("source").as_deref() == Some("hf")).then(|| Hf::open(std::path::Path::new(&get("snapshot").expect("meta.txt: snapshot"))));
        let layers: usize = match hf.as_ref() {
            Some(h) => h.cfg("num_hidden_layers").or(h.cfg("n_layer")).expect("config: layers"),
            None => get("layers").and_then(|v| v.parse().ok()).expect("meta.txt: layers"),
        };
        let want = ev("NUM_LAYERS", 0);
        assert!(want <= layers, "NUM_LAYERS={want} but REAL_WEIGHTS has {layers} layers");
        if hf.is_none() { assert!(std::env::var_os("REAL_RESID_SCALE").is_none(), "REAL_RESID_SCALE needs a direct (source=hf) REAL_WEIGHTS"); }
        eprintln!("[weights] real {model} checkpoint from {} (scale 2^{sf}, {layers} layers{})", dir.display(),
            if hf.is_some() { format!(", direct from {}", get("snapshot").unwrap()) } else { " exported".into() });
        Some(Real { dir, model: model.to_string(), hf })
    }
    fn raw(&self, name: &str) -> (Vec<usize>, Vec<i32>) {
        let Some(hf) = self.hf.as_ref().filter(|_| name != "prompts") else { return read_zkrw(&self.dir.join(format!("{name}.bin"))) };
        // Graph slot name -> checkpoint tensor and transform (the exporter's).
        let (layer, slot) = match name.strip_prefix('l').and_then(|r| r.split_once('.')).and_then(|(l, s)| l.parse::<usize>().ok().map(|l| (l, s))) {
            Some((l, s)) => (Some(l), s),
            None => (None, name),
        };
        // REAL_RESID_SCALE=c multiplies every write into the residual stream
        // (embeddings, attention / MLP output projections and their biases)
        // by c. Without an epsilon (the graph's normalizations have none) every
        // RMSNorm / LayerNorm is invariant to it, so the network computes the
        // same function (GPT-2's tied head scales its logits by c, argmax
        // unchanged), but the normalization statistic mean(x^2), formed from
        // per-element squares rounded at scale 2^sf, gets c^2 more resolution:
        // Llama-2's first layers have mean(x^2) ~ 5e-5, below the 2^-10 grid.
        let c: f64 = std::env::var("REAL_RESID_SCALE").ok().map(|v| v.parse().expect("REAL_RESID_SCALE")).unwrap_or(1.0);
        let resid = matches!(slot, "embed" | "wte" | "wpe" | "o_w" | "o_b" | "down_w" | "fc2_w" | "fc2_b" | "fc_out_w" | "fc_out_b");
        let k = if resid { c } else { 1.0 };
        let get = |t: &str| { let (s, mut v) = hf.get(t); if k != 1.0 { v.iter_mut().for_each(|x| *x *= k); } (s, v) };
        let one = |t: &str| { let (s, v) = get(t); if s.len() == 1 { (s, v.iter().map(|&x| quant(x)).collect()) } else { quant_2d(&s, &v, None, false) } };
        let lin = |t: &str, perm: Option<Vec<usize>>| { let (s, v) = get(t); quant_2d(&s, &v, perm.as_deref(), true) };
        match self.model.as_str() {
            "gpt2" => {
                let h = hf.cfg("n_embd").unwrap();
                // Conv1D weights are [in, out]; c_attn [h, 3h] holds q | k | v.
                let qkv = |l: usize, i: usize, bias: bool| {
                    let (s, v) = one(&format!("h.{l}.attn.c_attn.{}", if bias { "bias" } else { "weight" }));
                    if bias { (vec![h], v[i * h..(i + 1) * h].to_vec()) }
                    else { (vec![s[0], h], (0..s[0] * h).map(|k| v[(k / h) * s[1] + i * h + k % h]).collect()) }
                };
                match (layer, slot) {
                    (None, "lnf_w") => one("ln_f.weight"),
                    (None, "lnf_b") => one("ln_f.bias"),
                    (None, "wte") => one("wte.weight"),
                    (None, "wpe") => one("wpe.weight"),
                    (Some(l), s) => {
                        let p = |t: &str| format!("h.{l}.{t}");
                        match s {
                            "ln1_w" => one(&p("ln_1.weight")), "ln1_b" => one(&p("ln_1.bias")),
                            "ln2_w" => one(&p("ln_2.weight")), "ln2_b" => one(&p("ln_2.bias")),
                            "q_w" => qkv(l, 0, false), "k_w" => qkv(l, 1, false), "v_w" => qkv(l, 2, false),
                            "q_b" => qkv(l, 0, true), "k_b" => qkv(l, 1, true), "v_b" => qkv(l, 2, true),
                            "o_w" => one(&p("attn.c_proj.weight")), "o_b" => one(&p("attn.c_proj.bias")),
                            "fc1_w" => one(&p("mlp.c_fc.weight")), "fc1_b" => one(&p("mlp.c_fc.bias")),
                            "fc2_w" => one(&p("mlp.c_proj.weight")), "fc2_b" => one(&p("mlp.c_proj.bias")),
                            _ => panic!("no gpt2 slot {name}"),
                        }
                    }
                    _ => panic!("no gpt2 slot {name}"),
                }
            }
            "llama2" | "llama3" => {
                let nh = hf.cfg("num_attention_heads").unwrap();
                let nkv = hf.cfg("num_key_value_heads").unwrap_or(nh);
                let hd = hf.cfg("hidden_size").unwrap() / nh;
                match (layer, slot) {
                    (None, "final_norm") => one("model.norm.weight"),
                    (None, "embed") => one("model.embed_tokens.weight"),
                    (None, "lm_head") => one("lm_head.weight"),
                    (Some(l), s) => {
                        let p = |t: &str| format!("model.layers.{l}.{t}");
                        match s {
                            "attn_norm" => one(&p("input_layernorm.weight")),
                            "mlp_norm" => one(&p("post_attention_layernorm.weight")),
                            "q_w" => lin(&p("self_attn.q_proj.weight"), Some(pair_swap_perm(nh, hd, hd, true))),
                            "k_w" => lin(&p("self_attn.k_proj.weight"), Some(pair_swap_perm(nkv, hd, hd, true))),
                            "v_w" => lin(&p("self_attn.v_proj.weight"), None),
                            "o_w" => lin(&p("self_attn.o_proj.weight"), None),
                            "gate_w" => lin(&p("mlp.gate_proj.weight"), None),
                            "up_w" => lin(&p("mlp.up_proj.weight"), None),
                            "down_w" => lin(&p("mlp.down_proj.weight"), None),
                            _ => panic!("no llama slot {name}"),
                        }
                    }
                    _ => panic!("no llama slot {name}"),
                }
            }
            "gptj" => {
                let nh = hf.cfg("n_head").unwrap();
                let hd = hf.cfg("n_embd").unwrap() / nh;
                let rot = hf.cfg("rotary_dim").unwrap();
                match (layer, slot) {
                    (None, "lnf_w") => one("transformer.ln_f.weight"),
                    (None, "lnf_b") => one("transformer.ln_f.bias"),
                    (None, "wte") => one("transformer.wte.weight"),
                    (None, "lm_head") => one("lm_head.weight"),
                    (None, "lm_head_b") => one("lm_head.bias"),
                    (Some(l), s) => {
                        let p = |t: &str| format!("transformer.h.{l}.{t}");
                        match s {
                            "ln1_w" => one(&p("ln_1.weight")),
                            "ln1_b" => one(&p("ln_1.bias")),
                            "q_w" => lin(&p("attn.q_proj.weight"), Some(pair_swap_perm(nh, hd, rot, false))),
                            "k_w" => lin(&p("attn.k_proj.weight"), Some(pair_swap_perm(nh, hd, rot, false))),
                            "v_w" => lin(&p("attn.v_proj.weight"), None),
                            "o_w" => lin(&p("attn.out_proj.weight"), None),
                            "fc_in_w" => lin(&p("mlp.fc_in.weight"), None),
                            "fc_in_b" => one(&p("mlp.fc_in.bias")),
                            "fc_out_w" => lin(&p("mlp.fc_out.weight"), None),
                            "fc_out_b" => one(&p("mlp.fc_out.bias")),
                            _ => panic!("no gptj slot {name}"),
                        }
                    }
                    _ => panic!("no gptj slot {name}"),
                }
            }
            m => panic!("direct checkpoint loading is not wired for {m}; use an exported REAL_WEIGHTS"),
        }
    }
    /// Rows [r0, r0 + nr) and columns [c0, c0 + nc) of a 1-D or 2-D tensor
    /// (1-D: columns only), as a padded column-major Constant witness.
    fn block(&self, name: &str, r0: usize, nr: usize, c0: usize, nc: usize) -> Witness {
        let (shape, v) = self.raw(name);
        let w = block_witness(name, &shape, &v, r0, nr, c0, nc);
        self.channels(name, w)
    }
    /// The axis of a 2-D weight that its einsums keep (its output channels):
    /// graph linears are [in, out], the embedding [vocab, hidden] and the LM
    /// head [vocab, hidden] (logits per vocab row). GPT-2's tied table feeds
    /// both, so it stays per tensor, as do 1-D gains and biases.
    fn channel_axis(&self, name: &str) -> Option<usize> {
        let slot = name.split_once('.').map(|(_, s)| s).unwrap_or(name);
        match slot {
            "q_w" | "k_w" | "v_w" | "o_w" | "gate_w" | "up_w" | "down_w" | "fc1_w" | "fc2_w" | "fc_in_w" | "fc_out_w" | "embed" => Some(1),
            "wte" if self.model != "gpt2" => Some(1),
            "lm_head" => Some(0),
            _ => None,
        }
    }
    /// REAL_CHANNEL_SCALE (default on): store a 2-D weight per output channel
    /// with the fewest right shifts that keep every entry within 8 signed
    /// digit planes (|w| <= 255); see dag::builder::register_channel_shift.
    fn channels(&self, name: &str, w: Witness) -> Witness {
        let Some(axis) = self.channel_axis(name).filter(|_| w.shape.len() == 2 && std::env::var("REAL_CHANNEL_SCALE").map(|v| v != "0").unwrap_or(true)) else { return w };
        let (r, c) = (w.shape[0], w.shape[1]);
        let rp = r.next_power_of_two();
        let ev = w.data.as_ref().unwrap().evaluations();
        let iv = |k: usize| crate::util::arith::f_to_int(ev[k]);
        let n = if axis == 0 { r } else { c };
        let at = |ch: usize, k: usize| if axis == 0 { ch + rp * k } else { k + rp * ch };
        let len = if axis == 0 { c } else { r };
        // round(v / 2^s), half to even, as the exporter's torch.round.
        let shr = |v: i128, s: u32| -> i128 { if s == 0 { return v } let d = 1i128 << s; let q = v.div_euclid(d); let rem = v - q * d; if 2 * rem > d || (2 * rem == d && q % 2 != 0) { q + 1 } else { q } };
        use rayon::prelude::*;
        let shifts: Vec<u32> = (0..n).into_par_iter().map(|ch| {
            let m = (0..len).map(|k| iv(at(ch, k)).abs()).max().unwrap_or(0);
            (0u32..).find(|&s| shr(m, s).abs() <= 255 && shr(-m, s).abs() <= 255).unwrap()
        }).collect();
        let mut d = ev.clone();
        for ch in 0..n { let s = shifts[ch]; if s == 0 { continue } for k in 0..len { let i = at(ch, k); d[i] = crate::util::arith::int_to_f(shr(iv(i), s)); } }
        let mut out = Witness::new(w.shape.clone(), d, w.data_type, w.sf, w.role);
        crate::dag::builder::register_channel_shift(&mut out, shifts);
        out
    }
    /// The whole tensor; its shape must be `shape`.
    fn full(&self, name: &str, shape: Vec<usize>) -> Witness {
        let (s, v) = self.raw(name);
        assert_eq!(s, shape, "{name}: exported shape {s:?}, the graph slot needs {shape:?}");
        if s.len() == 1 { block_witness(name, &s, &v, 0, 1, 0, s[0]) } else { self.channels(name, block_witness(name, &s, &v, 0, s[0], 0, s[1])) }
    }
    fn prompt(&self, k: u64, len: usize) -> Vec<usize> {
        let (s, v) = self.raw("prompts");
        assert!(s.len() == 2 && s[0] > 0 && s[1] >= len, "prompts.bin: shape {s:?}");
        let r = (k as usize) % s[0];
        v[r * s[1]..r * s[1] + len].iter().map(|&t| t as usize).collect()
    }
}

fn read_zkrw(path: &std::path::Path) -> (Vec<usize>, Vec<i32>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..4], b"ZKRW", "{}: not a ZKRW tensor", path.display());
    let u = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let nd = u(4);
    let shape: Vec<usize> = (0..nd).map(|i| u(8 + 4 * i)).collect();
    let off = 8 + 4 * nd;
    let n: usize = shape.iter().product();
    assert_eq!(b.len(), off + 4 * n, "{}: truncated", path.display());
    let v = b[off..].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    (shape, v)
}

/// Logical row-major values -> padded column-major storage (element (i, j)
/// of an [r, c] tensor at i + j * r.next_power_of_two(), see Witness::get).
fn block_witness(name: &str, shape: &[usize], v: &[i32], r0: usize, nr: usize, c0: usize, nc: usize) -> Witness {
    use rayon::prelude::*;
    let (rows, cols) = match shape.len() { 1 => (1, shape[0]), 2 => (shape[0], shape[1]), _ => panic!("{name}: only 1-D/2-D tensors") };
    assert!(r0 + nr <= rows && c0 + nc <= cols, "{name}: block [{r0}+{nr}, {c0}+{nc}] outside {shape:?}");
    let to_f = |x: i32| crate::util::arith::int_to_f(x as i128);
    if shape.len() == 1 {
        let mut d = zeros(nc.next_power_of_two());
        for j in 0..nc { d[j] = to_f(v[c0 + j]); }
        return Witness::new(vec![nc], d, DataType::Float, *SF_LOG, Role::Constant);
    }
    let rp = nr.next_power_of_two();
    let mut d = zeros(rp * nc.next_power_of_two());
    d.par_chunks_mut(rp).take(nc).enumerate().for_each(|(j, col)| {
        for i in 0..nr { col[i] = to_f(v[(r0 + i) * cols + c0 + j]); }
    });
    Witness::new(vec![nr, nc], d, DataType::Float, *SF_LOG, Role::Constant)
}

/// A normalization gain: 1.0 (dense) or the benchmark's zeros.
fn norm(dim: usize) -> Witness {
    if !dense() { return cw(vec![dim]); }
    let mut d = zeros(dim.next_power_of_two());
    for x in d.iter_mut().take(dim) { *x = F(1u64 << *SF_LOG); }
    Witness::new(vec![dim], d, DataType::Float, *SF_LOG, Role::Constant)
}

/// A bias: zeros in both modes.
fn bias(dim: usize) -> Witness { cw(vec![dim]) }

/// A built request: the graph, its witness, which edges are public, and the
/// unit count for per-unit reporting (generated tokens, input tokens, 1).
pub struct Instance {
    pub name: String,
    pub dag: Dag,
    pub wit: Vec<Vec<Witness>>,
    pub public: Vec<EdgeId>,
    pub units: usize,
    pub unit: &'static str,
    pub witness_secs: f64,
    pub ar: Option<super::ar_statement::ArStatement>,
}

/// Embedding / head patterns of the previous binaries.
#[derive(Clone, Copy)]
enum Emb { Gpt2, Const1024, Alt2048 }

/// Embedding / head table source: a synthetic pattern, or rows of a real
/// checkpoint tensor ([vocab, hidden]) with an optional head bias ([vocab]).
#[derive(Clone, Copy)]
enum Tab<'a> { Pat(Emb), Real(&'a Real, &'static str, Option<&'static str>) }

impl Tab<'_> {
    /// Rows [start, start + len) as a [len, hidden] Constant witness.
    fn rows(&self, start: usize, len: usize, hidden: usize) -> Witness {
        match *self {
            Tab::Pat(k) => Witness::new(vec![len, hidden], emb_data(k, len, hidden), DataType::Float, *SF_LOG, Role::Constant),
            Tab::Real(r, name, _) => r.block(name, start, len, 0, hidden),
        }
    }
    fn bias(&self, start: usize, len: usize) -> Option<Witness> {
        match *self { Tab::Real(r, _, Some(b)) => Some(r.block(b, 0, 1, start, len)), _ => None }
    }
}

fn emb_data(kind: Emb, rows: usize, hidden: usize) -> Vec<F> {
    let rp = rows.next_power_of_two();
    let hp = hidden.next_power_of_two();
    let mut data = zeros(rp * hp);
    for d in 0..hp {
        let val = match kind { Emb::Gpt2 => F(1000 + 10 * ((d % 16) as u64)), Emb::Const1024 => F(1024), Emb::Alt2048 => F(if d % 2 == 0 { 0 } else { 2048 }) };
        match kind { Emb::Const1024 => for v in 0..rp { data[v + d * rp] = val; }, _ => for v in 0..rows { data[v + d * rp] = val; } }
    }
    data
}

struct Ar { g: DagBuilder, emb_sel: EdgeId, logits: Vec<EdgeId>, sels: Vec<EdgeId>, ranges: Vec<(usize, usize)>, rank_constants: Vec<[usize;3]>, prompt: Option<Vec<usize>> }

/// Embedding, body (closure), LM head (tied for GPT-2), argmax check.
#[allow(clippy::too_many_arguments)]
fn ar_graph(seq: usize, vocab: usize, hidden: usize, emb: Tab, head: Tab, tied: bool, shards: usize, pos: Option<Witness>,
            body: &dyn Fn(&mut DagBuilder, EdgeId) -> EdgeId) -> Ar {
    let init = vec![0usize; seq];
    let mut g = DagBuilder::new();
    g.exact_head_merge = matches!(emb, Tab::Real(..));
    let w_e = g.param(emb.rows(0, vocab, hidden));
    let (h0, emb_sel) = g.embedding_lookup(w_e, seq, vocab, &init, Role::Input);
    let h0 = match pos { Some(p) => g.add_positional_encoding(h0, p), None => h0 };
    let h_in = g.change_shape(h0, vec![1, seq, hidden]);
    let h_out = body(&mut g, h_in);
    if shards > 1 {
        let ranges = DagBuilder::vocab_shard_ranges(vocab, shards);
        let lens: Vec<usize> = ranges.iter().map(|&(_, l)| l).collect();
        let ws: Vec<EdgeId> = ranges.iter().map(|&(s, l)| g.param(head.rows(s, l, hidden))).collect();
        let mut lg = g.lm_head_sharded(h_out, &ws, seq, &lens);
        for (k, &(s, l)) in ranges.iter().enumerate() {
            if let Some(b) = head.bias(s, l) { let b = g.param(b); lg[k] = g.add(lg[k], b)[0]; }
        }
        let first_new = g.num_edges;
        let sels = g.argmax_check_sharded(&lg, &ranges, seq, &init, Role::Input);
        let rank_constants = rank_constants(&g, first_new, ranges.len());
        Ar { g, emb_sel, logits: lg, sels, ranges, rank_constants, prompt: None }
    } else {
        let logits = if tied { g.lm_head_weight_tied(h_out, w_e, seq, vocab) } else {
            let w = g.param(head.rows(0, vocab, hidden));
            g.lm_head(h_out, w, seq, vocab)
        };
        let logits = match head.bias(0, vocab) { Some(b) => { let b = g.param(b); g.add(logits, b)[0] } None => logits };
        let first_new = g.num_edges;
        let sel = g.argmax_check(logits, seq, vocab, &init, Role::Input);
        let rank_constants = rank_constants(&g, first_new, 1);
        Ar { g, emb_sel, logits: vec![logits], sels: vec![sel], ranges: vec![(0, vocab)], rank_constants, prompt: None }
    }
}

/// Run the generation: random input tokens, forward, next tokens = argmax of
/// the logits, set the argmax selectors, forward again.
fn rank_constants(g: &DagBuilder, first: usize, shards: usize) -> Vec<[usize;3]> {
    // Rank constructors emit B,V,index as public inputs before selector inputs.
    let edges: Vec<usize> = (first..g.num_edges).filter(|&e| g.init_values[e].as_ref().map(|w|w.role==Role::Input).unwrap_or(false)).take(3*shards).collect();
    assert_eq!(edges.len(),3*shards);
    edges.chunks_exact(3).map(|e|[e[0],e[1],e[2]]).collect()
}

fn rss_gb() -> f64 { std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())).unwrap_or(0.0) * 4096.0 / 1e9 }
fn ar_run(name: &str, ar: Ar, seq: usize, vocab: usize, request: u64) -> Instance {
    let Ar { g, emb_sel, logits, sels, ranges, rank_constants, prompt } = ar;
    let tr = std::env::var("MORPHEUS_RSS").is_ok();
    if tr { eprintln!("[rss] graph built {:.1} GB", rss_gb()); }
    let (dag, mut wit) = g.compile();
    if tr { eprintln!("[rss] compiled {:.1} GB", rss_gb()); }
    let t = Instant::now();
    let mut rng = StdRng::seed_from_u64(42 + request);
    assert!(seq > 0);
    let prompt_len = 16.min(seq);
    let mut tokens: Vec<usize> = (0..seq).map(|_| rng.gen::<usize>() % vocab).collect();
    // Real checkpoints: the request's prompt is a real text prompt (the
    // random suffix is only the Jacobi starting point).
    if let Some(p) = prompt.as_ref() {
        assert!(p.len() >= prompt_len && p.iter().all(|&t| t < vocab), "real prompt does not fit the vocabulary");
        tokens[..prompt_len].copy_from_slice(&p[..prompt_len]);
    }
    // Real checkpoints only: starting points for the Jacobi passes. Both only
    // choose the initial suffix; the passes below still run to the fixed
    // point, which is the unique fixed-point greedy suffix.
    //  * MORPHEUS_TOKEN_CACHE=<dir>: reuse the suffix a previous run of the
    //    same model / depth / shape / scale / prompt generated (then one
    //    confirming pass), and store this run's suffix.
    //  * MORPHEUS_DRAFT_CMD=<cmd>: a float draft model (scripts/draft_server.py)
    //    proposes the suffix after the confirmed prefix, initially and after
    //    every pass that changed a token: ~one pass per divergence from the
    //    float model instead of ~one per token.
    let cache = prompt.as_ref().and_then(|_| std::env::var("MORPHEUS_TOKEN_CACHE").ok()).map(|d| {
        let _ = std::fs::create_dir_all(&d);
        std::path::PathBuf::from(d).join(format!("{}_L{}_s{seq}_v{vocab}_sf{}_p{request}.tok", name.replace(['/', ' '], "_"), std::env::var("NUM_LAYERS").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "all".into()), *SF_LOG))
    });
    let mut cached = false;
    if let Some(c) = cache.as_ref() {
        if let Ok(txt) = std::fs::read_to_string(c) {
            let t: Vec<usize> = txt.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if t.len() == seq && t[..prompt_len] == tokens[..prompt_len] && t.iter().all(|&x| x < vocab) {
                tokens = t; cached = true;
                eprintln!("[generate] starting from cached suffix {}", c.display());
            }
        }
    }
    let mut drafter = prompt.as_ref().and_then(|_| std::env::var("MORPHEUS_DRAFT_CMD").ok()).map(|cmd| {
        use std::io::BufRead;
        let mut ch = std::process::Command::new("sh").arg("-c").arg(&cmd)
            .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn().expect("MORPHEUS_DRAFT_CMD");
        let mut out = std::io::BufReader::new(ch.stdout.take().unwrap());
        let mut l = String::new(); out.read_line(&mut l).unwrap();
        assert_eq!(l.trim(), "ready", "draft server did not start");
        (ch, out)
    });
    let mut draft = |tokens: &mut Vec<usize>, keep: usize| {
        use std::io::{BufRead, Write};
        let Some((ch, out)) = drafter.as_mut() else { return };
        let line: Vec<String> = std::iter::once(seq).chain(tokens[..keep].iter().copied()).map(|x| x.to_string()).collect();
        writeln!(ch.stdin.as_mut().unwrap(), "{}", line.join(" ")).unwrap();
        let mut l = String::new(); out.read_line(&mut l).unwrap();
        let t: Vec<usize> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        assert!(t.len() == seq && t[..keep] == tokens[..keep], "bad draft");
        for (p, x) in t.into_iter().enumerate().skip(keep) { tokens[p] = x.min(vocab - 1); }
    };
    if !cached { draft(&mut tokens, prompt_len); }
    // Generate an actual suffix. This is deliberately outside proof timing.
    // Jacobi decoding: each pass sets every suffix token to the argmax at the
    // previous position. The model is causal, so after k passes the first k
    // suffix tokens are final, and the unique fixed point is the sequential
    // greedy suffix. Stops at the fixed point (at most seq - prompt_len passes).
    let mut passes = 0;
    for _ in prompt_len..seq {
        wit[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &tokens, Role::Input)];
        dag.run(&mut wit, &[]);
        passes += 1;
        if tr { eprintln!("[rss] pass {passes} {:.1} GB", rss_gb()); }
        if std::env::var_os("MORPHEUS_GEN_TRACE").is_some() { eprintln!("[generate] pass {passes} done at {:.1}s", t.elapsed().as_secs_f64()); }
        let mut changed = false;
        let mut first_change = seq;
        for position in prompt_len..seq {
            let (mut best, mut value) = (0usize, i128::MIN);
            for (k, &(start,len)) in ranges.iter().enumerate() {
                for off in 0..len {
                    let v=f_to_int(wit[logits[k]][0].get(&[position-1,off]));
                    if v>value { value=v; best=start+off; }
                }
            }
            if tokens[position] != best { tokens[position]=best; if !changed { first_change = position; } changed = true; }
        }
        if !changed { break; }
        // Tokens before first_change were confirmed, and first_change is now
        // correct; redraft the rest.
        if first_change + 1 < seq { draft(&mut tokens, first_change + 1); }
    }
    eprintln!("[generate] {} suffix tokens in {passes} forward passes", seq - prompt_len);
    drop(draft);
    if let Some((mut ch, _)) = drafter.take() { drop(ch.stdin.take()); let _ = ch.wait(); }
    if let Some(c) = cache.as_ref() {
        let _ = std::fs::write(c, tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "));
    }
    if tr {
        // Witness memory by producing operator (8 bytes per stored element).
        let mut by: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for e in 0..dag.num_edges() {
            let Some(d) = wit[e][0].data.as_ref() else { continue };
            let k = dag.producers[e].map(|n| format!("{:?}", dag.nodes[n].kind).split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string()).unwrap_or("-".into());
            let x = by.entry(k).or_default(); x.0 += d.len(); x.1 += 1;
        }
        let mut v: Vec<_> = by.into_iter().collect(); v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        for (k, (n, c)) in v.iter().take(15) { eprintln!("[rss]   {k:24} {c:6} tensors {:8.2} GB", *n as f64 * 8.0 / 1e9); }
        if std::env::var_os("MORPHEUS_RSS_EXIT").is_some() { std::process::exit(0); }
    }
    if std::env::var("MORPHEUS_RSS").is_ok() { eprintln!("[rss] after generation {:.1} GB", std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<f64>().ok())).unwrap_or(0.0) * 4096.0 / 1e9); }
    if std::env::var("MORPHEUS_GEN_CHECK").is_ok() {
        // Reference: one position per forward pass.
        let mut seq_tokens = tokens.clone();
        for position in prompt_len..seq { seq_tokens[position] = 0; }
        for position in prompt_len..seq {
            wit[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &seq_tokens, Role::Input)];
            dag.run(&mut wit, &[]);
            let (mut best, mut value) = (0usize, i128::MIN);
            for (k, &(start,len)) in ranges.iter().enumerate() {
                for off in 0..len {
                    let v=f_to_int(wit[logits[k]][0].get(&[position-1,off]));
                    if v>value { value=v; best=start+off; }
                }
            }
            seq_tokens[position]=best;
        }
        assert_eq!(seq_tokens, tokens, "Jacobi suffix differs from sequential generation");
        eprintln!("[generate] matches sequential generation");
    }
    wit[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &tokens, Role::Input)];
    dag.run(&mut wit, &[]);
    let next: Vec<usize> = (0..seq).map(|i| {
        let (mut bt, mut bv) = (0usize, i128::MIN);
        for (k, &(start, len)) in ranges.iter().enumerate() {
            let lg = &wit[logits[k]][0];
            for off in 0..len { let v = f_to_int(lg.get(&[i, off])); if v > bv { bv = v; bt = start + off; } }
        }
        bt
    }).collect();
    if ranges.len() > 1 {
        for (k, &(start, len)) in ranges.iter().enumerate() {
            let local: Vec<Option<usize>> = next.iter().map(|&t| if t >= start && t < start + len { Some(t - start) } else { None }).collect();
            wit[sels[k]] = vec![DagBuilder::build_sharded_one_hot_selector_witness(seq, len, &local, Role::Input)];
        }
    } else {
        wit[sels[0]] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &next, Role::Input)];
    }
    dag.run(&mut wit, &[]);
    let mut public = vec![emb_sel];
    public.extend(&sels);
    public.extend(rank_constants.iter().flatten().copied());
    let ar = Some(super::ar_statement::ArStatement { prompt_len, seq, vocab, embedding: emb_sel, selectors: sels.clone(), ranges: ranges.clone(), rank_constants, logits: logits.clone(), logit_bound: DagBuilder::first_argmax_budget(vocab).0 });
    Instance { name: name.into(), dag, wit, public, units: seq.saturating_sub(16).max(1), unit: "generated token", witness_secs: t.elapsed().as_secs_f64(), ar }
}

pub fn gpt2(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 12), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 50257));
    let (nh, hd) = (ev("NUM_HEADS", 12), ev("HEAD_DIM", 64));
    let h = nh * hd; let ffn = 4 * h;
    let rw = Real::open("gpt2");
    let mut v: [Vec<Witness>; 16] = Default::default();
    if let Some(r) = rw.as_ref() {
        // Slot order of gpt_2_small: ln1 w, q k v o, ln1 b, q k v o biases,
        // ln2 w, fc1, fc2, ln2 b, fc1 b, fc2 b.
        const N: [&str; 16] = ["ln1_w", "q_w", "k_w", "v_w", "o_w", "ln1_b", "q_b", "k_b", "v_b", "o_b", "ln2_w", "fc1_w", "fc2_w", "ln2_b", "fc1_b", "fc2_b"];
        for l in 0..layers {
            for (k, n) in N.iter().enumerate() {
                let shape = match k { 1..=4 => vec![h, h], 11 => vec![h, ffn], 12 => vec![ffn, h], 14 => vec![ffn], _ => vec![h] };
                v[k].push(r.full(&format!("l{l}.{n}"), shape));
            }
        }
    } else {
    for _ in 0..layers {
        v[0].push(norm(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        for k in 5..10 { v[k].push(bias(h)); }
        v[10].push(norm(h));
        v[11].push(mat(vec![h, ffn])); v[12].push(mat(vec![ffn, h]));
        v[13].push(bias(h)); v[14].push(bias(ffn)); v[15].push(bias(h));
    }
    }
    let ln = match rw.as_ref() { Some(r) => (r.full("lnf_w", vec![h]), r.full("lnf_b", vec![h])), None => (norm(h), bias(h)) };
    let pos = match rw.as_ref() {
        Some(r) => r.block("wpe", 0, seq, 0, h),
        None => Witness::new(vec![seq, h], zeros(seq.next_power_of_two() * h.next_power_of_two()), DataType::Float, *SF_LOG, Role::Constant),
    };
    let v = std::cell::RefCell::new(Some((v, ln)));
    let tab = match rw.as_ref() { Some(r) => Tab::Real(r, "wte", None), None => Tab::Pat(Emb::Gpt2) };
    let mut ar = ar_graph(seq, vocab, h, tab, tab, true, 1, Some(pos), &|g, x| {
        let (v, ln) = v.borrow_mut().take().unwrap();
        let [a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p] = v;
        g.pipe(&[x], gpt_2_small(a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p, ln.0, ln.1, nh, hd, seq))[0]
    });
    ar.prompt = rw.as_ref().map(|r| r.prompt(request, 16.min(seq)));
    ar_run("GPT-2", ar, seq, vocab, request)
}

pub fn gptj(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 28), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 50400));
    let (nh, hd, ffn) = (ev("NUM_HEADS", 16), ev("HEAD_DIM", 256), ev("FFN_DIM", 16384));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let rw = Real::open("gptj");
    let mut v: [Vec<Witness>; 10] = Default::default();
    if let Some(r) = rw.as_ref() {
        // Slot order of gpt_j_6b_hidden: ln1 w, q k v o, ln1 b, fc_in, fc_out, fc_in b, fc_out b.
        const N: [&str; 10] = ["ln1_w", "q_w", "k_w", "v_w", "o_w", "ln1_b", "fc_in_w", "fc_out_w", "fc_in_b", "fc_out_b"];
        for l in 0..layers {
            for (k, n) in N.iter().enumerate() {
                let shape = match k { 1..=4 => vec![h, h], 6 => vec![h, ffn], 7 => vec![ffn, h], 8 => vec![ffn], _ => vec![h] };
                v[k].push(r.full(&format!("l{l}.{n}"), shape));
            }
        }
    } else {
    for _ in 0..layers {
        v[0].push(norm(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        v[5].push(bias(h));
        v[6].push(mat(vec![h, ffn])); v[7].push(mat(vec![ffn, h]));
        v[8].push(bias(ffn)); v[9].push(bias(h));
    }
    }
    let fin = match rw.as_ref() { Some(r) => (r.full("lnf_w", vec![h]), r.full("lnf_b", vec![h])), None => (norm(h), bias(h)) };
    // Pretrained GPT-J: rotary on the first ROTARY_DIM (64) dims of each
    // head, no frequency scaling, and GELU(h) = h * sigmoid(1.702 h).
    let rope = rw.as_ref().map(|_| rope_mats_plain(hd, seq, 10000.0, ev("ROTARY_DIM", 64)));
    let gelu_mul = rw.is_some();
    let v = std::cell::RefCell::new(Some((v, fin, rope)));
    let (emb, head) = match rw.as_ref() { Some(r) => (Tab::Real(r, "wte", None), Tab::Real(r, "lm_head", Some("lm_head_b"))), None => (Tab::Pat(Emb::Alt2048), Tab::Pat(Emb::Alt2048)) };
    let mut ar = ar_graph(seq, vocab, h, emb, head, false, shards, None, &|g, x| {
        let ([a, b, c, d, e, f, gg, hh, i, j], fin, rope) = v.borrow_mut().take().unwrap();
        g.pipe(&[x], gpt_j_6b_hidden_ext(a, b, c, d, e, f, gg, hh, i, j, fin.0, fin.1, nh, hd, seq, rope, gelu_mul))[0]
    });
    ar.prompt = rw.as_ref().map(|r| r.prompt(request, 16.min(seq)));
    ar_run("GPT-J-6B", ar, seq, vocab, request)
}

pub fn llama2(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 32), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 32000));
    let (nh, hd, ffn, fs) = (ev("NUM_HEADS", 32), ev("HEAD_DIM", 128), ev("FFN_DIM", 11008), ev("FFN_SHARDS", 1));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let sf = ffn / fs;
    let (mut a, mut b, mut c, mut d, mut e, mut f) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut p1, mut p2, mut p3): (Vec<Vec<Witness>>, Vec<Vec<Witness>>, Vec<Vec<Witness>>) = (vec![], vec![], vec![]);
    let sfp = ffn.next_power_of_two() / fs;
    let cs = |shape: Vec<usize>, n: usize| Witness::new(shape, zeros(n), DataType::Float, *SF_LOG, Role::Constant);
    let rw = Real::open("llama2");
    if let Some(r) = rw.as_ref() {
        assert_eq!(ffn % fs, 0, "real weights: FFN_SHARDS must divide FFN_DIM");
        for l in 0..layers {
            let n = |s: &str| format!("l{l}.{s}");
            a.push(r.full(&n("attn_norm"), vec![h])); b.push(r.full(&n("q_w"), vec![h, h])); c.push(r.full(&n("k_w"), vec![h, h]));
            d.push(r.full(&n("v_w"), vec![h, h])); e.push(r.full(&n("o_w"), vec![h, h])); f.push(r.full(&n("mlp_norm"), vec![h]));
            p1.push((0..fs).map(|s| r.block(&n("gate_w"), 0, h, s * sf, sf)).collect());
            p2.push((0..fs).map(|s| r.block(&n("up_w"), 0, h, s * sf, sf)).collect());
            p3.push((0..fs).map(|s| r.block(&n("down_w"), s * sf, sf, 0, h)).collect());
        }
    }
    for _ in 0..(if rw.is_some() { 0 } else { layers }) {
        a.push(norm(h)); b.push(mat(vec![h, h])); c.push(mat(vec![h, h])); d.push(mat(vec![h, h])); e.push(mat(vec![h, h])); f.push(norm(h));
        let hp = h.next_power_of_two();
        if dense() {
            p1.push((0..fs).map(|_| mat(vec![h, sf])).collect());
            p2.push((0..fs).map(|_| mat(vec![h, sf])).collect());
            p3.push((0..fs).map(|_| mat(vec![sf, h])).collect());
        } else {
            p1.push((0..fs).map(|_| cs(vec![h, sf], hp * sfp)).collect());
            p2.push((0..fs).map(|_| cs(vec![h, sf], hp * sfp)).collect());
            p3.push((0..fs).map(|_| cs(vec![sf, h], sfp * hp)).collect());
        }
    }
    let fin = match rw.as_ref() { Some(r) => r.full("final_norm", vec![h]), None => norm(h) };
    // Pretrained Llama-2: plain RoPE (theta 10000, no Llama-3 scaling).
    let rope = rw.as_ref().map(|_| rope_mats_plain(hd, seq, 10000.0, hd));
    let v = std::cell::RefCell::new(Some((a, b, c, d, e, f, p1, p2, p3, fin, rope)));
    let (emb, head) = match rw.as_ref() { Some(r) => (Tab::Real(r, "embed", None), Tab::Real(r, "lm_head", None)), None => (Tab::Pat(Emb::Const1024), Tab::Pat(Emb::Const1024)) };
    let mut ar = ar_graph(seq, vocab, h, emb, head, false, shards, None, &|g, x| {
        let (a, b, c, d, e, f, p1, p2, p3, fin, rope) = v.borrow_mut().take().unwrap();
        g.pipe(&[x], llama_2_7b_hidden_rope(a, b, c, d, e, f, p1, p2, p3, fin, nh, hd, seq, rope))[0]
    });
    ar.prompt = rw.as_ref().map(|r| r.prompt(request, 16.min(seq)));
    ar_run("LLaMA2-7B", ar, seq, vocab, request)
}

pub fn llama3(request: u64) -> Instance {
    let (layers, seq, vocab) = (ev("NUM_LAYERS", 32), ev("SEQ_LEN", 128), ev("VOCAB_SIZE", 128256));
    let (nh, nkv, hd, ffn) = (ev("NUM_HEADS", 32), ev("NUM_KV_HEADS", 8), ev("HEAD_DIM", 128), ev("FFN_DIM", 14336));
    let shards = ev("VOCAB_SHARDS", 1);
    let h = nh * hd;
    let kv = nkv * hd;
    let rw = Real::open("llama3");
    let mut v: [Vec<Witness>; 9] = Default::default();
    if let Some(r) = rw.as_ref() {
        const N: [&str; 9] = ["attn_norm", "q_w", "k_w", "v_w", "o_w", "mlp_norm", "gate_w", "up_w", "down_w"];
        for l in 0..layers {
            for (k, n) in N.iter().enumerate() {
                let shape = match k { 0 | 5 => vec![h], 1 | 4 => vec![h, h], 2 | 3 => vec![h, kv], 6 | 7 => vec![h, ffn], _ => vec![ffn, h] };
                v[k].push(r.full(&format!("l{l}.{n}"), shape));
            }
        }
    }
    for _ in 0..(if rw.is_some() { 0 } else { layers }) {
        v[0].push(norm(h));
        v[1].push(mat(vec![h, h])); v[2].push(mat(vec![h, kv])); v[3].push(mat(vec![h, kv])); v[4].push(mat(vec![h, h]));
        v[5].push(norm(h));
        v[6].push(mat(vec![h, ffn])); v[7].push(mat(vec![h, ffn])); v[8].push(mat(vec![ffn, h]));
    }
    let fin = match rw.as_ref() { Some(r) => r.full("final_norm", vec![h]), None => norm(h) };
    let v = std::cell::RefCell::new(Some((v, fin)));
    // The default RoPE tables (theta 500000 with the Llama-3.1 frequency
    // scaling) are the checkpoint's.
    let (emb, head) = match rw.as_ref() { Some(r) => (Tab::Real(r, "embed", None), Tab::Real(r, "lm_head", None)), None => (Tab::Pat(Emb::Const1024), Tab::Pat(Emb::Const1024)) };
    let mut ar = ar_graph(seq, vocab, h, emb, head, false, shards, None, &|g, x| {
        let ([a, b, c, d, e, f, p1, p2, p3], fin) = v.borrow_mut().take().unwrap();
        let w = |v: Vec<Witness>| v.into_iter().map(|w| vec![w]).collect();
        g.pipe(&[x], llama3_8b_hidden(a, b, c, d, e, f, w(p1), w(p2), w(p3), fin, seq, hd, nh, nkv))[0]
    });
    ar.prompt = rw.as_ref().map(|r| r.prompt(request, 16.min(seq)));
    ar_run("LLaMA3-8B", ar, seq, vocab, request)
}

pub fn bert(request: u64) -> Instance {
    let (layers, seq) = (ev("NUM_LAYERS", 24), ev("SEQ_LEN", 128));
    let (nh, hd) = (16usize, 64usize);
    let h = nh * hd; let ffn = 4 * h; let ncls = 2;
    let unit = |dim: usize| { let mut d = zeros(dim.next_power_of_two()); for x in d.iter_mut().take(dim) { *x = F(1024); } Witness::new(vec![dim], d, DataType::Float, *SF_LOG, Role::Constant) };
    let mut v: [Vec<Witness>; 16] = Default::default();
    for _ in 0..layers {
        v[0].push(unit(h));
        for k in 1..5 { v[k].push(mat(vec![h, h])); }
        for k in 5..10 { v[k].push(cw(vec![h])); }
        v[10].push(unit(h));
        v[11].push(mat(vec![h, ffn])); v[12].push(mat(vec![ffn, h]));
        v[13].push(cw(vec![h])); v[14].push(cw(vec![ffn])); v[15].push(cw(vec![h]));
    }
    let mut g = DagBuilder::new();
    let x = g.input(vec![1, seq, h], DataType::Float);
    let [a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p] = v;
    let _ = g.pipe(&[x], bert_large(a, b, c, d, e, f, gg, hh, i, j, k, l, m, n, o, p, unit(h), cw(vec![h]), nh, hd, seq, mat(vec![h, ncls]), cw(vec![ncls])));
    let (dag, mut wit) = g.compile();
    let pad: usize = [1usize, seq, h].iter().map(|&s| s.next_power_of_two()).product();
    let stride = seq.next_power_of_two();
    let data: Vec<F> = (0..pad).map(|i| F(if (i / stride + request as usize) % 2 == 0 { 0 } else { 2048 })).collect();
    let input = Witness::new(vec![1, seq, h], data, DataType::Float, *SF_LOG, Role::Input);
    let t = Instant::now();
    dag.run(&mut wit, &[(x, input)]);
    Instance { name: "BERT-Large".into(), dag, wit, public: vec![x], units: seq, unit: "input token", witness_secs: t.elapsed().as_secs_f64(), ar: None }
}

/// MODEL env: gpt2 | gptj | llama2 | llama3 | bert.
pub fn build(name: &str, request: u64) -> Instance {
    match name {
        "gpt2" => gpt2(request),
        "gptj" => gptj(request),
        "llama2" => llama2(request),
        "llama3" => llama3(request),
        "bert" => bert(request),
        other => panic!("unknown MODEL {other}"),
    }
}

/// CPU-only validation of a built transformer instance (no proving, no GPU):
/// prints the generated tokens, every range-check (NonNegative) violation of
/// the witness (the `[range] WARNING` condition, which lean aux skips), and
/// the widest request tensor (to choose REQ_PLANES_LOG). With `dump`, writes
/// the tokens and the dequantized logits of every position:
/// b"ZKLG" | seq:u32 | vocab:u32 | sf:u32 | tokens:u32*seq | logits:f32*seq*vocab.
pub fn check_report(inst: &Instance, dump: Option<&std::path::Path>) {
    use crate::basicblock::BasicBlockType;
    use almost_goldilocks_cuda::field::ALMOST_GOLDILOCKS_PRIME as Q;
    use rayon::prelude::*;
    let (dag, wit) = (&inst.dag, &inst.wit);
    let kind = |e: usize| dag.producers[e].map(|n| format!("{:?}", dag.nodes[n].kind).split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap().to_string()).unwrap_or("leaf".into());
    let signed = |x: u64| -> i128 { if x > Q / 2 { -((Q - x) as i128) } else { x as i128 } };
    let bits = |a: i128| (128 - a.unsigned_abs().leading_zeros()) as usize;
    let sf = *SF_LOG;
    // Tokens and logits.
    if let Some(ar) = inst.ar.as_ref() {
        let sel = &wit[ar.embedding][0];
        let tokens: Vec<usize> = (0..ar.seq).map(|i| (0..ar.vocab).find(|&v| f_to_int(sel.get(&[i, v])) == 1).expect("one-hot row")).collect();
        println!("[check] tokens {tokens:?}");
        let mut lg = vec![0f32; ar.seq * ar.vocab];
        let mut amax = 0i128;
        for (k, &(start, len)) in ar.ranges.iter().enumerate() {
            let w = &wit[ar.logits[k]][0];
            for i in 0..ar.seq { for off in 0..len {
                let v = f_to_int(w.get(&[i, off]));
                amax = amax.max(v.abs());
                lg[i * ar.vocab + start + off] = v as f32 / (1u64 << sf) as f32;
            } }
        }
        println!("[check] max |logit| {:.2} ({} bits at scale 2^{sf}; ARGMAX_LOGIT_BITS must exceed it)", amax as f64 / (1u64 << sf) as f64, bits(amax));
        if let Some(p) = dump {
            let mut b: Vec<u8> = b"ZKLG".to_vec();
            for x in [ar.seq as u32, ar.vocab as u32, sf as u32] { b.extend(x.to_le_bytes()); }
            for &t in &tokens { b.extend((t as u32).to_le_bytes()); }
            for &x in &lg { b.extend(x.to_le_bytes()); }
            std::fs::write(p, b).expect("write dump");
            println!("[check] dumped tokens + logits to {}", p.display());
        }
    }
    // CHECK_EDGES_DUMP=<dir>: every 3-D [1, seq, d] edge, dequantized
    // (row-major f32), for locating a divergence against the float model.
    if let (Some(dir), Some(ar)) = (std::env::var_os("CHECK_EDGES_DUMP"), inst.ar.as_ref()) {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let _ = ar;
        for e in 0..dag.num_edges() {
            let w = &wit[e][0];
            let n: usize = w.shape.iter().product();
            if w.shape.len() < 2 || n > (1 << 23) || w.data.is_none() || dag.producers[e].is_none() { continue; }
            if let Some(d) = std::env::var("CHECK_EDGES_DIM").ok().and_then(|v| v.parse::<usize>().ok()) { if *w.shape.last().unwrap() != d { continue; } }
            // [ndim][shape][sf][row-major logical values]
            let mut b: Vec<u8> = Vec::new();
            b.extend((w.shape.len() as u32).to_le_bytes());
            for &x in &w.shape { b.extend((x as u32).to_le_bytes()); }
            b.extend((w.sf as u32).to_le_bytes());
            let mut idx = vec![0usize; w.shape.len()];
            for _ in 0..n {
                b.extend((f_to_int(w.get(&idx)) as f32 / (1u64 << w.sf.min(60)) as f32).to_le_bytes());
                for d in (0..idx.len()).rev() { idx[d] += 1; if idx[d] < w.shape[d] { break; } idx[d] = 0; }
            }
            std::fs::write(dir.join(format!("e{e}_{}.f32", kind(e))), b).unwrap();
        }
    }
    // Range checks.
    let mut viol = 0usize;
    for nd in &dag.nodes {
        let BasicBlockType::NonNegative(nn) = &nd.kind else { continue };
        let e = nd.inputs[0];
        let Some(_) = wit[e][0].data.as_ref() else { println!("[check] range node {} input e{e} has no data", nd.id); continue };
        let v = crate::morpheus::piop::witness_vals(&wit[e][0]);
        let t = 1i128 << nn.table_size_log;
        let (bad, neg, mx) = v.par_iter().map(|&x| { let s = signed(x); (usize::from(s < 0 || s >= t), usize::from(s < 0), s.abs()) })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2.max(b.2)));
        if bad > 0 {
            viol += 1;
            // The checked tensor: the Sub's first operand (integer_interval /
            // argmax build `x - lo`, `hi - x`).
            let src = dag.producers[e].map(|p| dag.nodes[p].inputs.iter().map(|&i| format!("e{i}:{}", kind(i))).collect::<Vec<_>>().join(",")).unwrap_or_default();
            println!("[check] [range] VIOLATION node {} table 2^{}: {bad}/{} out of range ({neg} negative), max |value| {mx} ({} bits); input e{e} = {} of [{src}]",
                nd.id, nn.table_size_log, v.len(), bits(mx), kind(e));
        }
    }
    println!("[check] range violations: {viol} nodes");
    // Request tensor widths (as Morpheus' request preparation declares them).
    let mut g = crate::morpheus::piop::Graph::new(dag, wit, crate::morpheus::piop::Config::default());
    for &e in &inst.public { g.public.entry(e).or_insert_with(|| crate::morpheus::piop::witness_vals(&wit[e][0])); }
    g.prune_unclaimed();
    let mut req: Vec<(usize, usize, String)> = Vec::new();
    let mut model_b = 0usize;
    for e in 0..dag.num_edges() {
        if !g.committed[e] && !g.model[e] { continue; }
        let Some(_) = wit[e][0].data.as_ref() else { continue };
        let v = crate::morpheus::piop::witness_vals(&wit[e][0]);
        let mb = v.par_iter().map(|&x| bits(signed(x))).max().unwrap_or(0).max(1);
        if g.model[e] { model_b = model_b.max(mb); continue; }
        if g.public.contains_key(&e) || crate::morpheus::piop::request_source_is_bits(&g, e) { continue; }
        req.push((mb, e, kind(e)));
    }
    req.sort_by(|a, b| b.cmp(a));
    let top: Vec<String> = req.iter().take(8).map(|(b, e, k)| format!("e{e}:{k}={b}")).collect();
    let mb = req.first().map(|r| r.0).unwrap_or(0);
    println!("[check] widest request tensor {mb} bits -> REQ_PLANES_LOG >= {} (top: {})", crate::util::arith::log2_ceil(mb), top.join(" "));
    println!("[check] widest model tensor {model_b} bits");
}
