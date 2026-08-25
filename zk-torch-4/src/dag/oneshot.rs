//! One-shot autoregressive (AR) circuit primitives (ported from
//! zk-torch-2 `src/dag/oneshot.rs`, zkAgent §4.1).
//!
//! These build the front-end (token → hidden via embedding lookup) and
//! back-end (hidden → logits → argmax check) around the existing
//! transformer core, so a full T-token autoregressive generation is
//! proved in ONE full-sequence circuit (`seq_len = T`, causal-masked)
//! instead of T per-token proofs. The model's prediction at position `i`
//! depends only on tokens `0..=i` (causal masking), so a single
//! full-sequence forward pass reproduces every AR step; the public
//! shift constraint `argmax(logits[i]) == token[i+1]` (checked by the
//! verifier on public data) ties each position's prediction to the next
//! input token.
//!
//! ## Roles and the streaming accumulator
//!
//! The streaming accumulator defers + amortizes every `Role::Constant`
//! edge across N proofs, treating them as one shared value. So only the
//! genuinely-shared MODEL WEIGHTS may be `Role::Constant` here. The
//! per-generation one-hot SELECTORS (embedding + argmax) change every
//! generation, so they must NOT be `Role::Constant` — they are committed
//! and opened per-proof. `build_one_hot_selector` therefore tags the
//! selector with `selector_role`, which one-shot+streaming callers set to
//! `Role::Input`/`Auxiliary` (per-proof), while standalone single-proof
//! callers may use `Role::Constant`.

use almost_goldilocks_cuda::field::AlmostGoldilocksField;

use crate::dag::builder::DagBuilder;
use crate::dag::{DataType, EdgeId, Role, Witness};

impl DagBuilder {
    /// Build a one-hot selector matrix. `S[i, indices[i]] = 1`, all others
    /// 0. Shape `(len, table_size)`, col-major MLE layout: element `(i, v)`
    /// at flat index `i + v * len_pad` (matches zk-torch-4's 2D tensor
    /// convention — cf. `causal_mask`). `role` controls how the streaming
    /// path treats it (see module docs): `Role::Constant` for a standalone
    /// single proof, a per-proof role (`Input`/`Auxiliary`) when composed
    /// with the streaming accumulator.
    pub fn build_one_hot_selector_witness(
        len: usize,
        table_size: usize,
        indices: &[usize],
        role: Role,
    ) -> Witness {
        assert_eq!(indices.len(), len);
        let len_pad = len.next_power_of_two();
        let table_pad = table_size.next_power_of_two();
        let total = len_pad * table_pad;
        let mut data = vec![AlmostGoldilocksField(0); total];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < table_size, "Index {} out of range [0, {})", idx, table_size);
            data[i + idx * len_pad] = AlmostGoldilocksField(1);
        }
        Witness::new(vec![len, table_size], data, DataType::Float, 0, role)
    }

    /// Public accessor used by the AR loop / argmax fixup to rebuild a
    /// selector after a forward pass (selector edge is overwritten in
    /// place). Defaults to `Role::Constant` to match the zk-torch-2
    /// signature; one-shot+streaming callers should use
    /// [`Self::build_one_hot_selector_witness`] with a per-proof role.
    pub fn build_one_hot_selector_pub(
        len: usize,
        table_size: usize,
        indices: &[usize],
    ) -> Witness {
        Self::build_one_hot_selector_witness(len, table_size, indices, Role::Constant)
    }

    /// Embedding lookup: proves `H_0[i,:] = W_E[token_ids[i],:]`.
    ///
    /// `w_e`: committed embedding matrix edge, shape `(vocab_size, hidden_dim)`.
    /// `token_ids`: public token IDs (length `seq_len`).
    /// `selector_role`: role for the per-generation one-hot selector — use a
    /// per-proof role (`Input`/`Auxiliary`) under the streaming accumulator,
    /// `Role::Constant` for a standalone single proof.
    ///
    /// Returns `(h0_edge, selector_edge)`. The selector edge is exposed so
    /// the caller can overwrite it during autoregressive generation.
    pub fn embedding_lookup(
        &mut self,
        w_e: EdgeId,
        seq_len: usize,
        vocab_size: usize,
        token_ids: &[usize],
        selector_role: Role,
    ) -> (EdgeId, EdgeId) {
        let s = Self::build_one_hot_selector_witness(seq_len, vocab_size, token_ids, selector_role);
        let s_id = if selector_role == Role::Constant {
            self.param(s)
        } else {
            self.committed_input(s)
        };
        // H_0 = einsum("sv,vd->sd", S, W_E) -> (seq_len, hidden_dim).
        // scale_back=false: S has sf=0, W_E has sf=SF_LOG; output keeps
        // sf=SF_LOG. scale_back=true would divide by 2^SF_LOG (output_sf =
        // first input's sf = 0), destroying the values.
        let h0 = self.einsum("sv,vd->sd".to_string(), vec![s_id, w_e], false)[0];
        (h0, s_id)
    }

    /// Add learned positional embeddings (GPT-2 absolute PE).
    /// `H_pe = H_0 + P`. `pos_embed`: shape `(seq_len, hidden_dim)`,
    /// sf set by the caller to match `H_0`.
    pub fn add_positional_encoding(&mut self, h0: EdgeId, pos_embed: Witness) -> EdgeId {
        let p = self.param(pos_embed);
        self.add(h0, p)[0]
    }

    /// LM head using a weight-tied `W_E` (GPT-2). `hidden`: transformer
    /// output edge, shape `(1, seq_len, hidden_dim)`. Returns logits edge
    /// of shape `(seq_len, vocab_size)`.
    pub fn lm_head_weight_tied(
        &mut self,
        hidden: EdgeId,
        w_e: EdgeId,
        seq_len: usize,
        vocab_size: usize,
    ) -> EdgeId {
        // logits = einsum("bsd,vd->bsv", hidden, W_E) -> (1, seq_len, vocab).
        // scale_back=true: input sf = 2·SF_LOG (hidden · weight) → SF_LOG.
        let logits = self.einsum("bsd,vd->bsv".to_string(), vec![hidden, w_e], true)[0];
        self.change_shape(logits, vec![seq_len, vocab_size])
    }

    /// LM head with a separate (un-tied) weight `w_lm` of shape
    /// `(vocab_size, hidden_dim)` (Llama). Same contraction as the tied
    /// variant. Returns logits edge `(seq_len, vocab_size)`.
    pub fn lm_head(
        &mut self,
        hidden: EdgeId,
        w_lm: EdgeId,
        seq_len: usize,
        vocab_size: usize,
    ) -> EdgeId {
        let logits = self.einsum("bsd,vd->bsv".to_string(), vec![hidden, w_lm], true)[0];
        self.change_shape(logits, vec![seq_len, vocab_size])
    }

    /// Sound argmax verification: proves `token_ids[i] = argmax(logits[i,:])`
    /// for each position via logit-difference range checks.
    ///
    /// `logits`: edge of shape `(seq_len, vocab_size)`.
    /// `token_ids`: claimed argmax token per position (length `seq_len`).
    /// `selector_role`: per-proof role under streaming, `Constant` standalone.
    ///
    /// Returns the selector edge ID (overwritable after a forward pass).
    pub fn argmax_check(
        &mut self,
        logits: EdgeId,
        seq_len: usize,
        vocab_size: usize,
        token_ids: &[usize],
        selector_role: Role,
    ) -> EdgeId {
        // 1. One-hot selector for the claimed next tokens.
        let s = Self::build_one_hot_selector_witness(seq_len, vocab_size, token_ids, selector_role);
        let s_id = if selector_role == Role::Constant {
            self.param(s)
        } else {
            self.committed_input(s)
        };

        // 2. Extract selected logits: selected[i] = logits[i, token_ids[i]]
        //    = Σ_v logits[i,v]·S[i,v]. Done as elementwise-multiply then
        //    reduce (two patterns the einsum prover supports — cf.
        //    "bsi,bsi->bsi" and "bsi->bs"); a fused "sv,sv->s" (shared
        //    free index + summation) is not exercised elsewhere and is
        //    avoided here. scale_back=false (logits sf=SF_LOG, S sf=0).
        let prod = self.einsum("sv,sv->sv".to_string(), vec![logits, s_id], false)[0];
        let selected = self.einsum("sv->s".to_string(), vec![prod], false)[0];

        // 3. Broadcast subtract: diffs[i,j] = selected[i] - logits[i,j].
        let selected_broad = self.change_shape(selected, vec![seq_len, 1]);
        let diffs = self.sub(selected_broad, logits)[0];

        // 4. Range check diffs ≥ 0 ⇒ selected[i] ≥ logits[i,j] ∀j ⇒
        //    token_ids[i] is the argmax of row i.
        self.add_nonneg_node(diffs);

        s_id
    }

    /// Contiguous vocab-shard ranges: `n_shards` blocks covering
    /// `[0, vocab_size)`. Block `k` is `(start, len)`. Used to split the LM
    /// head + argmax range-check along the vocab axis so each fold-tree leaf
    /// stays small enough to fit GPU memory at full vocab (the un-sharded
    /// argmax range-check is dense over the whole vocab → one ~2^30 leaf).
    /// May return fewer than `n_shards` blocks when `vocab_size` is small.
    pub fn vocab_shard_ranges(vocab_size: usize, n_shards: usize) -> Vec<(usize, usize)> {
        let n_shards = n_shards.max(1);
        let chunk = vocab_size.div_ceil(n_shards);
        let mut out = Vec::new();
        let mut start = 0;
        while start < vocab_size {
            let len = chunk.min(vocab_size - start);
            out.push((start, len));
            start += len;
        }
        if out.is_empty() {
            out.push((0, vocab_size));
        }
        out
    }

    /// One-hot selector for a single vocab shard. `local[i] = Some(off)` puts
    /// a 1 at column `off` of row `i`; `local[i] = None` leaves row `i` all
    /// zero (the claimed token for position `i` lives in a different shard).
    /// Shape `(len, shard_vocab)`, same col-major layout as
    /// [`Self::build_one_hot_selector_witness`].
    pub fn build_sharded_one_hot_selector_witness(
        len: usize,
        shard_vocab: usize,
        local: &[Option<usize>],
        role: Role,
    ) -> Witness {
        assert_eq!(local.len(), len);
        let len_pad = len.next_power_of_two();
        let table_pad = shard_vocab.next_power_of_two();
        let mut data = vec![AlmostGoldilocksField(0); len_pad * table_pad];
        for (i, off) in local.iter().enumerate() {
            if let Some(off) = off {
                assert!(*off < shard_vocab, "offset {} out of shard [0,{})", off, shard_vocab);
                data[i + off * len_pad] = AlmostGoldilocksField(1);
            }
        }
        Witness::new(vec![len, shard_vocab], data, DataType::Float, 0, role)
    }

    /// Vocab-sharded LM head. `w_lm_shards[k]` has shape
    /// `(shard_vocabs[k], hidden_dim)`; returns one logits edge per shard,
    /// each shape `(seq_len, shard_vocabs[k])`. Same per-shard contraction as
    /// [`Self::lm_head`] (`bsd,vd->bsv`, scale_back). Splitting the vocab axis
    /// keeps each head einsum + downstream argmax leaf small.
    pub fn lm_head_sharded(
        &mut self,
        hidden: EdgeId,
        w_lm_shards: &[EdgeId],
        seq_len: usize,
        shard_vocabs: &[usize],
    ) -> Vec<EdgeId> {
        assert_eq!(w_lm_shards.len(), shard_vocabs.len());
        w_lm_shards
            .iter()
            .zip(shard_vocabs.iter())
            .map(|(&w, &v)| {
                let logits = self.einsum("bsd,vd->bsv".to_string(), vec![hidden, w], true)[0];
                self.change_shape(logits, vec![seq_len, v])
            })
            .collect()
    }

    /// Sharded sound argmax. Mathematically identical to
    /// [`Self::argmax_check`], partitioned over the vocab: the global one-hot's
    /// single 1 lands in exactly one shard, so
    /// `selected[i] = Σ_k Σ_v logits_k[i,v]·S_k[i,v]` equals the selected
    /// logit, and a per-shard range check `diffs_k = selected − logits_k ≥ 0`
    /// proves it dominates every logit in every shard. `shard_ranges` are the
    /// `(start, len)` blocks from [`Self::vocab_shard_ranges`]; `token_ids` are
    /// global vocab indices. Returns the N per-shard selector edges
    /// (overwritable per generation, in shard order).
    pub fn argmax_check_sharded(
        &mut self,
        logits_shards: &[EdgeId],
        shard_ranges: &[(usize, usize)],
        seq_len: usize,
        token_ids: &[usize],
        selector_role: Role,
    ) -> Vec<EdgeId> {
        assert_eq!(logits_shards.len(), shard_ranges.len());

        // 1. Per-shard one-hot selectors + per-shard selected logit.
        let mut sel_ids = Vec::with_capacity(shard_ranges.len());
        let mut selected_k_ids = Vec::with_capacity(shard_ranges.len());
        for (k, &(start, len)) in shard_ranges.iter().enumerate() {
            let local: Vec<Option<usize>> = token_ids
                .iter()
                .map(|&t| if t >= start && t < start + len { Some(t - start) } else { None })
                .collect();
            let s = Self::build_sharded_one_hot_selector_witness(seq_len, len, &local, selector_role);
            let s_id = if selector_role == Role::Constant {
                self.param(s)
            } else {
                self.committed_input(s)
            };
            // selected_k[i] = Σ_v logits_k[i,v]·S_k[i,v] (elementwise then reduce).
            let prod = self.einsum("sv,sv->sv".to_string(), vec![logits_shards[k], s_id], false)[0];
            let selected_k = self.einsum("sv->s".to_string(), vec![prod], false)[0];
            sel_ids.push(s_id);
            selected_k_ids.push(selected_k);
        }

        // 2. selected[i] = Σ_k selected_k[i]  (the 1 is in exactly one shard).
        let mut selected = selected_k_ids[0];
        for &sk in &selected_k_ids[1..] {
            selected = self.add(selected, sk)[0];
        }
        let selected_broad = self.change_shape(selected, vec![seq_len, 1]);

        // 3. Per-shard range check: diffs_k = selected − logits_k ≥ 0.
        for &lg in logits_shards {
            let diffs = self.sub(selected_broad, lg)[0];
            self.add_nonneg_node(diffs);
        }

        sel_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::DagBuilder;
    use crate::util::arith::{f_to_int, int_to_f};
    use almost_goldilocks_cuda::field::AlmostGoldilocksField;

    // Scaled-identity embedding: W_E[v, d] = sf·[v == d]. Then
    // h0[i,:] = W_E[token[i],:] (a scaled basis vector) and
    // logits[i,v] = <h0[i], W_E[v]> / sf = sf·[v == token[i]] — so the
    // argmax of row i is exactly token[i] and the max margin is sf.
    fn identity_w_e(vocab: usize, hidden: usize, sf: usize) -> Witness {
        let vp = vocab.next_power_of_two();
        let hp = hidden.next_power_of_two();
        let scale = int_to_f(1i128 << sf);
        let mut data = vec![AlmostGoldilocksField(0); vp * hp];
        // shape (vocab, hidden), col-major: (v, d) at v + d·vp.
        for v in 0..vocab.min(hidden) {
            data[v + v * vp] = scale;
        }
        Witness::new(vec![vocab, hidden], data, DataType::Float, sf, Role::Constant)
    }

    /// Forward-pass semantics of the ported primitives: embedding_lookup
    /// ("sv,vd->sd"), lm_head_weight_tied ("bsd,vd->bsv"), and the
    /// argmax_check chain ("sv,sv->s" + sub). Validates the einsum
    /// equations + 2D col-major layout without needing the GPU prove.
    #[test]
    fn oneshot_primitives_forward_semantics() {
        let sf = 8usize;
        let vocab = 4usize;
        let hidden = 4usize;
        let seq = 3usize;
        let tokens = vec![2usize, 0, 3];

        let mut g = DagBuilder::new();
        let w_e = g.param(identity_w_e(vocab, hidden, sf));
        let (h0, _sel) = g.embedding_lookup(w_e, seq, vocab, &tokens, Role::Constant);
        let h_in = g.change_shape(h0, vec![1, seq, hidden]);
        let logits = g.lm_head_weight_tied(h_in, w_e, seq, vocab);

        // Inline the argmax chain so we can read the diffs edge.
        let sel2 = g.param(DagBuilder::build_one_hot_selector_witness(
            seq, vocab, &tokens, Role::Constant));
        let selected = g.einsum("sv,sv->s".to_string(), vec![logits, sel2], false)[0];
        let selected_b = g.change_shape(selected, vec![seq, 1]);
        let diffs = g.sub(selected_b, logits)[0];

        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[]);

        // h0[i,:] must equal W_E[token[i],:] = sf·e_{token[i]}.
        let h0_w = &witnesses[h0][0];
        for i in 0..seq {
            for d in 0..hidden {
                let got = f_to_int(h0_w.get(&[i, d]));
                let want = if d == tokens[i] { 1i128 << sf } else { 0 };
                assert_eq!(got, want, "h0[{},{}]", i, d);
            }
        }

        // logits[i,v] must equal sf·[v == token[i]] (diagonal), so argmax
        // of each row is token[i] with margin sf.
        let lg_w = &witnesses[logits][0];
        for i in 0..seq {
            for v in 0..vocab {
                let got = f_to_int(lg_w.get(&[i, v]));
                let want = if v == tokens[i] { 1i128 << sf } else { 0 };
                assert_eq!(got, want, "logits[{},{}]", i, v);
            }
        }

        // diffs[i,j] = selected[i] - logits[i,j] = sf·[token==j? ... ] must
        // be ≥ 0 everywhere (0 at the argmax column, sf elsewhere).
        let df_w = &witnesses[diffs][0];
        for i in 0..seq {
            for j in 0..vocab {
                let got = f_to_int(df_w.get(&[i, j]));
                assert!(got >= 0, "diffs[{},{}] = {} should be ≥ 0", i, j, got);
                let want = if j == tokens[i] { 0 } else { 1i128 << sf };
                assert_eq!(got, want, "diffs[{},{}]", i, j);
            }
        }
    }

    /// Composition with the streaming accumulator (step 3): two
    /// one-shot-style generations (embedding → lm_head → argmax, the parts
    /// that exercise the role routing) where the shared weight W_E is
    /// `Role::Constant` (deferred + accumulated) and the per-generation
    /// one-hot selectors are `Role::Input` (committed/opened per-proof,
    /// NOT deferred). Asserts the deferred claims include W_E but never a
    /// selector edge, both per-proof verifies pass, the accumulator
    /// accepts both, and finalize verifies. (Transformer omitted to keep
    /// the GPU test small; the full pipeline is covered by the
    /// bench_streaming_oneshot_gpt2 bin.)
    #[test]
    fn oneshot_streaming_two_generations_verify() {
        use crate::commit::{AjtaiKey, GpuAjtaiStore};
        use crate::dag::streaming_accumulator::{AccumulatorState, VerifierAccumulator};
        use crate::transcript::Transcript;
        almost_goldilocks_cuda::init().expect("CUDA init");
        let seed = almost_goldilocks_cuda::ajtai::Seed([
            0x0123_4567, 0x89AB_CDEF, 0xFEED_FACE, 0xDEAD_BEEF,
            0xCAFE_BABE, 0x1357_9BDF, 0x2468_ACE0, 0x0BAD_C0DE]);
        let sf = 8usize;
        let (vocab, hidden, seq) = (2usize, 4usize, 2usize);

        // Build once: W_E Constant (tied), selectors Role::Input.
        let mut g = DagBuilder::new();
        let w_e = g.param(identity_w_e(vocab, hidden, sf));
        let init = vec![0usize; seq];
        let (h0, emb_sel) = g.embedding_lookup(w_e, seq, vocab, &init, Role::Input);
        let h_in = g.change_shape(h0, vec![1, seq, hidden]);
        let logits = g.lm_head_weight_tied(h_in, w_e, seq, vocab);
        let argmax_sel = g.argmax_check(logits, seq, vocab, &init, Role::Input);
        let (dag, wt) = g.compile();
        assert_eq!(wt[emb_sel][0].role, Role::Input);
        assert_eq!(wt[argmax_sel][0].role, Role::Input);

        let key = AjtaiKey::new(seed, /*max_num_vars=*/ 24, /*b=*/ 21);
        let label = b"oneshot-stream-test";
        let mut prover_acc = AccumulatorState::new(label);
        let mut verifier_acc = VerifierAccumulator::new(label);
        let gens = [vec![0usize, 1], vec![1usize, 0]];
        let mut last: Option<(GpuAjtaiStore, Vec<Vec<Witness>>)> = None;

        for (gi, tokens) in gens.iter().enumerate() {
            let mut store = GpuAjtaiStore::new(dag.num_edges(), key);
            let mut w = wt.clone();
            w[emb_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, tokens, Role::Input)];
            dag.run(&mut w, &[]);
            // With identity W_E, true argmax of row i is tokens[i].
            let next: Vec<usize> = (0..seq)
                .map(|i| argmax_row_test(&w[logits][0], i, vocab)).collect();
            assert_eq!(&next, tokens, "gen {}: argmax should equal embedding tokens", gi);
            w[argmax_sel] = vec![DagBuilder::build_one_hot_selector_witness(seq, vocab, &next, Role::Input)];
            dag.run(&mut w, &[]);
            dag.commit(&w, &mut store);

            let mut tp = Transcript::new(b"per-gen");
            let (dp, fp) = dag.prove_with_fold_tree_modes(&w, &store, &mut tp, /*defer=*/ true);
            let mut tv = Transcript::new(b"per-gen");
            let r = dag.verify_with_fold_tree_deferred(&w, &store, &dp, &fp, &mut tv);
            assert!(r.ok, "gen {} per-proof verify failed", gi);

            // W_E (Constant) must be deferred; selectors (Input) must not.
            let deferred: Vec<usize> = r.claims.iter().map(|c| c.edge_id).collect();
            assert!(deferred.contains(&w_e), "gen {}: W_E should be deferred", gi);
            assert!(!deferred.contains(&emb_sel) && !deferred.contains(&argmax_sel),
                "gen {}: a per-generation selector was deferred (unsound)", gi);

            let chunk = prover_acc.add_proof(&r, &w);
            assert!(verifier_acc.verify_add_proof(&r, &w, &chunk),
                "gen {} streaming verifier rejected", gi);
            last = Some((store, w));
        }

        let (store, wt_last) = last.unwrap();
        let final_proof = prover_acc.finalize(&wt_last, &store);
        assert!(verifier_acc.verify_finalize(&store, &final_proof),
            "verify_finalize rejected — composition soundness chain broken");
    }

    fn argmax_row_test(logits: &Witness, pos: usize, vocab: usize) -> usize {
        let mut best = 0usize; let mut bv = i128::MIN;
        for v in 0..vocab {
            let val = f_to_int(logits.get(&[pos, v]));
            if val > bv { bv = val; best = v; }
        }
        best
    }

    /// A WRONG claimed argmax token must produce a negative diff (the
    /// range check would reject) — the soundness direction of argmax_check.
    #[test]
    fn oneshot_argmax_wrong_token_makes_negative_diff() {
        let sf = 8usize;
        let (vocab, hidden, seq) = (4usize, 4usize, 2usize);
        let true_tokens = vec![1usize, 3];
        let wrong_tokens = vec![0usize, 3]; // row 0 claims token 0, true argmax is 1

        let mut g = DagBuilder::new();
        let w_e = g.param(identity_w_e(vocab, hidden, sf));
        let (h0, _) = g.embedding_lookup(w_e, seq, vocab, &true_tokens, Role::Constant);
        let h_in = g.change_shape(h0, vec![1, seq, hidden]);
        let logits = g.lm_head_weight_tied(h_in, w_e, seq, vocab);
        let sel = g.param(DagBuilder::build_one_hot_selector_witness(
            seq, vocab, &wrong_tokens, Role::Constant));
        let selected = g.einsum("sv,sv->s".to_string(), vec![logits, sel], false)[0];
        let selected_b = g.change_shape(selected, vec![seq, 1]);
        let diffs = g.sub(selected_b, logits)[0];

        let (dag, mut witnesses) = g.compile();
        dag.run(&mut witnesses, &[]);

        // Row 0 claimed token 0 (logits 0) but true argmax is token 1
        // (logits sf): diffs[0,1] = 0 - sf < 0.
        let df_w = &witnesses[diffs][0];
        assert_eq!(f_to_int(df_w.get(&[0, 1])), -(1i128 << sf),
            "wrong argmax must yield a negative diff at the true-max column");
    }
}
