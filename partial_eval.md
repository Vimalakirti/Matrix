Certainly! Here is the algorithm described as a formal vector-matrix transformation. You can copy and paste this into any Markdown editor.

---

## Algorithm: Partial Evaluation of a Multilinear Polynomial via Vector-Matrix Product

This algorithm performs the partial evaluation of the first $m$ variables of an $N$-variate multilinear polynomial $f$. It transforms the $2^N$ evaluations of $f$ into $2^{N-m}$ evaluations of a new polynomial $g$.

### 1. Mathematical Setup

Given:

* **Polynomial $f(x_1, \dots, x_N)$**: Represented by its evaluation vector $V_f$ of length $2^N$.
* **Evaluation Point $r = (r_1, \dots, r_m)$**: The values for the first $m$ variables.
* **Target $g(x_{m+1}, \dots, x_N) = f(r, x_{m+1}, \dots, x_N)$**: An $(N-m)$-variate polynomial.

### 2. The Vector-Matrix Representation

We define the operation as:


$$V_g = v_{eq(r)} \cdot M_f$$

Where:

* **$v_{eq(r)}$**: A row vector of size $1 \times 2^m$ containing the evaluations of $eq(r, \cdot)$.
* **$M_f$**: A matrix of size $2^m \times 2^{N-m}$ created by reshaping $V_f$.
* **$V_g$**: A row vector of size $1 \times 2^{N-m}$ representing the evaluations of $g$.

---

### 3. Step-by-Step Algorithm

#### Step A: Generate the Equality Vector $v_{eq(r)}$

Use dynamic programming to compute all $2^m$ evaluations of the $m$-variate $eq$ polynomial in $O(2^m)$ time.

1. Initialize `evals = [1.0]`
2. For each $r_i$ in $(r_1, \dots, r_m)$:
* `new_evals` = array of size `2 * len(evals)`
* For `j` from `0` to `len(evals) - 1`:
* `new_evals[2j]` = `evals[j] * (1 - r_i)`
* `new_evals[2j + 1]` = `evals[j] * r_i`


* `evals = new_evals`


3. **Result:** $v_{eq(r)} = \text{evals}$

#### Step B: Reshape the Evaluation Vector $V_f$

Interpret the flat array of $2^N$ evaluations as a matrix $M_f$ with $2^m$ rows and $2^{N-m}$ columns.

* Index mapping: $M_f[i][j] = V_f[i \cdot 2^{N-m} + j]$

#### Step C: Compute the Vector-Matrix Product

For each column $j \in \{0, \dots, 2^{N-m}-1\}$, calculate the dot product of $v_{eq(r)}$ and the $j$-th column of $M_f$:

$$V_g[j] = \sum_{i=0}^{2^m-1} v_{eq(r)}[i] \cdot M_f[i][j]$$

---

### 4. Complexity Analysis

* **Time Complexity:** $O(2^N)$. We perform $2^m \cdot 2^{N-m}$ multiplications and additions.
* **Space Complexity:** $O(2^{N-m})$ to store the resulting vector $V_g$ (excluding the input $V_f$).
* **Efficiency:** This is optimal as it touches each evaluation of $f$ exactly once.

### 5. Sample code in CPU

```rust
fn partial_evaluate_optimized(poly_evals: &[f64], r: &[f64]) -> Vec<f64> {
    let n = poly_evals.len().trailing_zeros() as usize;
    let m = r.len();
    assert!(m <= n);

    // 1. Optimized eq(r, x) generation
    let mut eq_vec = Vec::with_capacity(1 << m);
    eq_vec.push(1.0);

    for &ri in r {
        let current_len = eq_vec.len();
        for i in 0..current_len {
            let prev = eq_vec[i];
            let high = prev * ri;      // Only 1 multiplication
            let low = prev - high;     // 1 subtraction instead of multiplication
            
            eq_vec[i] = low;           // Update bit = 0 in place
            eq_vec.push(high);         // Push bit = 1
        }
    }

    // 2. Vector-Matrix product (LSB-first)
    let num_cols = 1 << (n - m);
    let num_rows = 1 << m;
    let mut result = vec![0.0; num_cols];

    for i in 0..num_rows {
        let eq_val = eq_vec[i];
        if eq_val == 0.0 { continue; }

        // Pointer/Iterator optimization could be used here for speed
        for j in 0..num_cols {
            result[j] += eq_val * poly_evals[i + (j << m)];
        }
    }

    result
}
```
