# ZK-Torch-3 Proving Framework: Pseudo-Algorithm

## Overview

The system proves correctness of a neural network computation organized as a **DAG** (Directed Acyclic Graph). Each node is a **BasicBlock** (Einsum, Add, Conv2D, etc.) that operates on **multilinear polynomials** stored on edges. The proof uses a **backward pass** from outputs to inputs, generating **sumcheck proofs** at each node and **opening proofs** via Basefold PCS.

---

## 1. Setup Phase

```
COMPILE(model_definition):
  Build DAG:
    nodes[]         — operations (Einsum, Add, Conv2D, ...)
    edges[]         — witness polynomials (multilinear over {0,1}^n)
    producers[e]    — which node produces edge e
    consumers[e]    — which nodes consume edge e
    topo[]          — topological order of nodes
    topo_levels[][] — nodes grouped by dependency level (for parallel execution)
    input_ports[]   — model input edges
    output_ports[]  — model output edges
```

## 2. Forward Pass (Witness Generation)

```
RUN(dag, input_data):
  For each level in topo_levels:      // levels are data-independent
    For each node in level:           // parallelizable via rayon
      inputs = [witnesses[e] for e in node.input_edges]
      outputs = node.basic_block.run(inputs)
      For each (e, out) in zip(node.output_edges, outputs):
        witnesses[e] = out
  Return witnesses
```

## 3. Commitment Phase

```
COMMIT(dag, witnesses, basefold_table):
  gpu_store = GpuCommitmentStore::new(basefold_table)
  For each edge e where should_commit(e):   // boundary edges, self-claim edges, etc.
    if 2 ≤ num_vars(e) ≤ 22:               // GPU memory constraints
      commitment = basefold_commit_gpu(witnesses[e])
      gpu_store.store(e, commitment.root, commitment.gpu_data)
  gpu_store.free_commitments()              // release GPU memory for proving phase
  Return gpu_store
```

## 4. Proving — Main Algorithm

```
PROVE(dag, witnesses, commitments, transcript):
  claims[e] = [] for all edges e    // claims waiting to be propagated
  node_proofs = {}
  nodes_to_prove = {}

  //=== PHASE 1: Output Claims ===
  For each output edge e:
    r = [transcript.challenge_ext2() for _ in 0..num_vars(e)]
    v = witnesses[e].evaluate(r)
    claims[e].append(Claim{edge=e, point=r, eval=v})
    nodes_to_prove.add(producers[e])

  //=== PHASE 2: Backward Pass (claim propagation) ===
  While nodes_to_prove is not empty:
    node_id = pick next node (must have all consumers already proved)

    // Collect all claims on this node's output edges
    out_claims = []
    For each output edge e of node_id:
      out_claims.extend(claims[e])

    // REDUCE: if multiple claims on same output, combine via Reducer
    If len(out_claims) > 1:
      reducer_proof, reduced_claim = REDUCER_PROVE(witnesses, out_claims, transcript)
      node_proofs[node_id].reducer = reducer_proof
      out_claims = [reduced_claim]

    // PROVE NODE: invoke BasicBlock-specific sumcheck
    sumcheck_proofs, input_claims = node.basic_block.prove(
        witnesses, out_claims, transcript)
    node_proofs[node_id].sumcheck = sumcheck_proofs

    // PROPAGATE: new claims on input edges flow to their producers
    For each claim c in input_claims:
      claims[c.edge].append(c)
      If producers[c.edge] exists:
        nodes_to_prove.add(producers[c.edge])

  //=== PHASE 3: Lookup Proofs ===
  two_pow_proof  = PROVE_TWO_POW(witnesses, claims, transcript)
  range_proof    = PROVE_RANGE(witnesses, claims, transcript)

  //=== PHASE 4: Opening Proofs ===
  opening_proofs = PROVE_OPENINGS(witnesses, claims, commitments, transcript)

  Return Proof{node_proofs, opening_proofs, two_pow_proof, range_proof}
```

## 5. Reducer — Combining Multiple Claims

When an edge has multiple consumers, each generates a claim. The Reducer combines them into one:

```
REDUCER_PROVE(witness_poly, claims=[c₁,...,cₖ], transcript):
  α = transcript.challenge_ext2()

  // Build combined eq polynomial:
  //   eq_combined(x) = Σᵢ αⁱ · eq(cᵢ.point, x)
  // where eq(r,x) = Π_j (r_j·x_j + (1-r_j)(1-x_j))

  eq_combined = Σᵢ αⁱ · lagrange_basis(cᵢ.point)

  // Expected sum = Σᵢ αⁱ · cᵢ.eval
  expected_sum = Σᵢ αⁱ · cᵢ.eval

  // Sumcheck: prove Σ_x witness(x) · eq_combined(x) = expected_sum
  proof, challenges = SUMCHECK_PROVE([witness_ext2, eq_combined], expected_sum)

  // Output: single reduced claim at the sumcheck challenge point
  reduced_eval = witness.evaluate(challenges)
  Return proof, Claim{edge=c₁.edge, point=challenges, eval=reduced_eval}
```

## 6. Einsum — Tensor Contraction Proof

Einsum handles operations like matrix multiply (`ij,jk->ik`):

```
EINSUM_PROVE(witnesses, equation, out_claim, transcript):
  // Parse equation: classify indices
  free_once  = indices appearing in output only once
  free_multi = indices appearing in multiple inputs (not output)
  summation  = indices not in output

  // Reorder variables: [free_once | free_multi | summation]
  For each input polynomial:
    permuted_input = permute_variables(input, new_order)  // LUT-based for n>16

  // Partial evaluate: fix free_once variables to challenge point
  For each permuted input:
    partial = partial_eval(permuted_input, out_claim.point[free_once_bits])
    // Now polynomial has only (free_multi + summation) variables

  // Broadcast: pad dimensions for any that need broadcasting
  For each partial polynomial:
    broadcast to sumcheck_size via modular indexing (% padded_shape)

  // Build eq polynomial for free_multi dimensions (if any)
  If free_multi is not empty:
    eq_poly = lagrange_basis(out_claim.point[free_multi_bits])
    broadcast eq_poly across summation dimensions

  // SUMCHECK: prove Σ_x (Πᵢ fᵢ(x)) · eq(x) = expected_sum
  If total_rounds ≤ GPU_SUMCHECK_THRESHOLD (14):
    proof = CPU_SUMCHECK([partials..., eq_poly])
  Else:
    proof = GPU_SUMCHECK([partials..., eq_poly])  // CUDA-accelerated

  // Extract input claims: invert permutation to restore original variable order
  For each input i:
    challenges_i = invert_permute(sumcheck_challenges, input_i_order)
    eval_i = final_eval(i)
    input_claims.append(Claim{edge=input_edge_i, point=challenges_i, eval=eval_i})

  Return proof, input_claims
```

## 7. Sumcheck Protocol (Core Primitive)

Proves `Σ_{x∈{0,1}ⁿ} Πᵢ fᵢ(x) = claimed_sum`:

```
SUMCHECK_PROVE(polynomials [f₁,...,fₘ], claimed_sum):
  For round j = 0 to n-1:
    // Compute univariate round polynomial h_j(t):
    //   h_j(t) = Σ_{x_{j+1},...,x_{n-1} ∈ {0,1}} Πᵢ fᵢ(r₀,...,r_{j-1}, t, x_{j+1},...,x_{n-1})

    // Evaluate at t=0 and t=1:
    s₀ = Σ_{x'∈{0,1}^{n-j-1}} Πᵢ fᵢ(r₀,...,r_{j-1}, 0, x')
    s₁ = Σ_{x'∈{0,1}^{n-j-1}} Πᵢ fᵢ(r₀,...,r_{j-1}, 1, x')

    Assert s₀ + s₁ = claimed_sum
    Send oracle message (s₀, s₁) to transcript

    rⱼ = transcript.challenge_ext2()    // Fiat-Shamir challenge
    claimed_sum = h_j(rⱼ)              // Lagrange interpolation at rⱼ

    // Fold polynomials: fᵢ[k] ← fᵢ[2k] + rⱼ·(fᵢ[2k+1] - fᵢ[2k])
    // (halves the polynomial size each round)
    For each fᵢ: fold(fᵢ, rⱼ)

  // After n rounds: each fᵢ is a single value
  final_eval = Πᵢ fᵢ[0]
  Assert final_eval = claimed_sum
  Return SumcheckProof{round_messages, final_eval}
```

**GPU variant**: fold kernel on GPU with double-buffered swap (avoids cross-warp race). Block-reduction for partial sums.

## 8. Lookup Proofs

For range constraints (e.g., proving values are non-negative):

```
PROVE_RANGE(witnesses, claims, transcript):
  // Table: range_table[i] = i for i ∈ [0, 2ⁿ)
  // Each NonNegative/ScaleDown node produces a SparseMLPoly "selection"
  //   selection = {(input_idx, table_idx)} pairs

  α = transcript.challenge_ext2()
  β = transcript.challenge_ext2()

  // For each selection polynomial:
  //   Compute part_aux[j] = Σ_{(inp,tbl)∈selection, tbl=j} eq(claim.point, inp)
  //   middle_claim = Σⱼ part_aux[j] · table[j]
  //   Accumulate: combined_aux += βⁱ · part_aux

  // Table sumcheck proves:
  //   Σ_y combined_aux(y) · (table(y) + α) = Σᵢ βⁱ · (middle_claimᵢ + α · sum_auxᵢ)

  Return sumcheck_proof
```

## 9. Opening Proofs (Basefold PCS)

```
PROVE_OPENINGS(witnesses, claims, commitments, transcript):
  master_seed = transcript.challenge_ext2()

  // Collect all opening tasks: (edge_id, claim_point)
  tasks = []
  For each committed edge e:
    For each claim c on e:
      tasks.append((e, c.point))

  // Deduplicate: group by (edge_id, point_bytes)
  // Only compute proof for first occurrence; clone for duplicates
  unique_tasks = deduplicate(tasks)

  // Partition by polynomial size
  gpu_tasks = {t ∈ unique_tasks : num_vars(t.edge) ≥ 22}
  cpu_tasks = {t ∈ unique_tasks : num_vars(t.edge) < 22}

  // GPU openings: sequential on GPU
  For each task in gpu_tasks:
    t = fork_transcript(master_seed, task_idx)
    proofs[task_idx] = gpu_open_ext2(witnesses[task.edge], task.point,
                                     commitments[task.edge].root, t)

  // CPU openings: parallel via rayon
  For each task in cpu_tasks (parallel):
    t = fork_transcript(master_seed, task_idx)
    proofs[task_idx] = cpu_open_ext2(witnesses[task.edge], task.point,
                                     commitments[task.edge].root, t)

  // Distribute proofs (duplicates get cloned from canonical)
  Return proofs
```

## 10. Verification (Mirrors Prover)

```
VERIFY(dag, proof, commitments, transcript):
  //=== PHASE 1: Re-derive output claims ===
  For each output edge e:
    r = [transcript.challenge_ext2() for _ in 0..num_vars(e)]   // same challenges
    Assert proof.claims[e][0].point == r

  //=== PHASE 2: Backward pass (identical traversal order) ===
  For each node in same order as prover:
    If multiple claims on output:
      α = transcript.challenge_ext2()
      expected = Σᵢ αⁱ · claimᵢ.eval
      Assert SUMCHECK_VERIFY(reducer_proof, expected, transcript)

    Assert node.basic_block.verify(witnesses, claims, sumcheck_proofs, transcript)
    Propagate input claims to producers

  //=== PHASE 3: Verify lookups ===
  Assert VERIFY_TWO_POW(proof.two_pow, transcript)
  Assert VERIFY_RANGE(proof.range, transcript)

  //=== PHASE 4: Verify openings (parallel) ===
  master_seed = transcript.challenge_ext2()
  For each unique opening task (parallel):
    t = fork_transcript(master_seed, task_idx)
    Assert BasefoldVerifier::verify_ext2(
      commitments[e].root, point, proof, basefold_table, t)

  Return true
```

## 11. Key Invariants

| Property | Detail |
|----------|--------|
| **Little-endian** | Variable 0 = bit 0 = LSB. `fix_variables` folds pairs `[2j, 2j+1]` |
| **Deterministic transcript** | Prover and verifier derive identical challenges via Poseidon2 sponge |
| **Backward claim flow** | Claims flow outputs → inputs; only producers of claimed edges are proved |
| **Reducer before node proof** | Multiple consumer claims are combined before the node's own sumcheck |
| **Dedup openings** | Same (edge, point) proven once, proof cloned for duplicates |
| **GPU/CPU threshold** | Sumcheck: ≤14 vars → CPU, >14 → GPU. Openings: ≥22 vars → GPU |

## 12. Multi-GPU Parallel Proving

```
PROVE_PARALLEL(dag, witnesses, num_partitions):
  // Partition DAG at evenly-spaced layer boundaries
  partitions = partition_dag(dag, boundary_edges)

  // Output claims → route to owning partition
  For each partition k (parallel, GPU k % num_devices):
    set_device(k % num_devices)
    transcript_k = transcript.fork(k)    // domain separation
    partition_proofs[k] = prove_partition(partitions[k], claims_for_k, transcript_k)

  // Merge boundary claims, then shared lookup + opening proofs
  merge_boundary_claims()
  lookup_proofs = prove_lookups(transcript)
  opening_proofs = prove_openings(transcript)  // parallel across devices
```
