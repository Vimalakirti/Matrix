# Goldilocks Field CUDA Implementation

CUDA kernels for arithmetic over the Goldilocks prime field and Poseidon2 hash function, based on the Plonky3 reference implementation.

## Field Properties

- **Prime**: P = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
- **Size**: 64-bit (single `uint64_t` per element)
- **Key optimization**: 2^96 ≡ -1 (mod P), enabling fast reduction
- **Non-canonical representation**: Values can be in [0, 2^64), reducing branches

## Files

- `goldilocks.cuh` - Header with field structure, constants, and inline device functions
- `goldilocks_kernels.cu` - CUDA kernels for batch field operations
- `extension.cuh` - Extension fields (quadratic and quintic) header
- `extension_kernels.cu` - CUDA kernels for extension field operations
- `poseidon2.cuh` - Poseidon2 hash function header with constants and permutation
- `poseidon2_kernels.cu` - CUDA kernels for batch hashing and Merkle trees
- `Makefile` - Build configuration

## Building

```bash
# Build all tests
make

# Run all tests
make test

# Run only field tests
make test-field

# Run only Poseidon2 tests
make test-poseidon2

# Run only extension field tests
make test-extension

# Build as static library
make lib

# Build as shared library
make shared

# Clean
make clean
```

Adjust `-arch=sm_70` in the Makefile to match your GPU architecture.

## Usage

### Initialization

```cpp
#include "goldilocks.cuh"

// Initialize constant memory (call once at startup)
cudaError_t err = goldilocks_init();
if (err != cudaSuccess) {
    // Handle error
}
```

### Creating Field Elements

```cpp
// On host or device
GoldilocksField a(12345);
GoldilocksField b(67890);
```

### Device-Side Operations

```cpp
__global__ void my_kernel(GoldilocksField* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    GoldilocksField a = data[idx];
    GoldilocksField b = data[idx + 1];

    // Arithmetic operations
    GoldilocksField sum = gl_add(a, b);
    GoldilocksField diff = gl_sub(a, b);
    GoldilocksField prod = gl_mul(a, b);
    GoldilocksField sq = gl_square(a);
    GoldilocksField neg = gl_neg(a);
    GoldilocksField inv = gl_inverse(a);
    GoldilocksField quot = gl_div(a, b);
    GoldilocksField half = gl_halve(a);
    GoldilocksField power = gl_exp(a, 100);

    // Equality (compares canonical forms)
    bool equal = gl_eq(a, b);
    bool is_zero = gl_is_zero(a);
    bool is_one = gl_is_one(a);

    // Canonicalize (only when needed for output/comparison)
    uint64_t canonical = canonicalize(a.value);
}
```

### Batch Operations

```cpp
// Allocate device memory
GoldilocksField *d_a, *d_b, *d_result;
cudaMalloc(&d_a, N * sizeof(GoldilocksField));
cudaMalloc(&d_b, N * sizeof(GoldilocksField));
cudaMalloc(&d_result, N * sizeof(GoldilocksField));

// Copy data to device
cudaMemcpy(d_a, h_a, N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, h_b, N * sizeof(GoldilocksField), cudaMemcpyHostToDevice);

// Use wrapper functions (automatic grid/block sizing)
gl_batch_add(d_a, d_b, d_result, N);
gl_batch_sub(d_a, d_b, d_result, N);
gl_batch_mul(d_a, d_b, d_result, N);
gl_batch_square(d_a, d_result, N);
gl_batch_inverse(d_a, d_result, N);

// Or launch kernels directly
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
gl_batch_mul_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, N);
```

### Extension Field (Quadratic)

```cpp
// F_{p^2} = F_p[x] / (x^2 - 7)
GoldilocksExtQuad a(GoldilocksField(1), GoldilocksField(2));  // 1 + 2x
GoldilocksExtQuad b(GoldilocksField(3), GoldilocksField(4));  // 3 + 4x

GoldilocksExtQuad sum = gl_ext_add(a, b);
GoldilocksExtQuad diff = gl_ext_sub(a, b);
GoldilocksExtQuad prod = gl_ext_mul(a, b);
GoldilocksExtQuad sq = gl_ext_square(a);
```

### Polynomial Operations

```cpp
// Evaluate polynomial at multiple points
gl_poly_eval_kernel<<<grid, block>>>(d_coeffs, d_points, d_results, n_coeffs, n_points);

// Naive polynomial multiplication (use FFT for large polynomials)
gl_poly_mul_naive_kernel<<<grid, block>>>(d_a, d_b, d_result, n_a, n_b);
```

### Matrix Operations

```cpp
// Matrix-vector multiplication: result = A * v
gl_matrix_vec_mul_kernel<<<grid, block>>>(d_A, d_v, d_result, m, n);

// Matrix-matrix multiplication: C = A * B
dim3 block(16, 16);
dim3 grid((n + 15) / 16, (m + 15) / 16);
gl_matrix_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
```

## API Reference

### Core Operations

| Function | Description |
|----------|-------------|
| `gl_add(a, b)` | Addition: a + b (mod P) |
| `gl_sub(a, b)` | Subtraction: a - b (mod P) |
| `gl_neg(a)` | Negation: -a (mod P) |
| `gl_mul(a, b)` | Multiplication: a * b (mod P) |
| `gl_square(a)` | Squaring: a^2 (mod P) |
| `gl_double(a)` | Doubling: 2*a (mod P) |
| `gl_halve(a)` | Halving: a/2 (mod P) |
| `gl_inverse(a)` | Inverse: 1/a (mod P) |
| `gl_div(a, b)` | Division: a/b (mod P) |
| `gl_exp(a, n)` | Exponentiation: a^n (mod P) |
| `gl_eq(a, b)` | Equality check |
| `gl_is_zero(a)` | Check if zero |
| `gl_is_one(a)` | Check if one |
| `canonicalize(v)` | Reduce to [0, P) |

### Batch Kernels

| Kernel | Description |
|--------|-------------|
| `gl_batch_add_kernel` | Element-wise addition |
| `gl_batch_sub_kernel` | Element-wise subtraction |
| `gl_batch_mul_kernel` | Element-wise multiplication |
| `gl_batch_square_kernel` | Element-wise squaring |
| `gl_batch_neg_kernel` | Element-wise negation |
| `gl_batch_inverse_kernel` | Element-wise inversion |
| `gl_scalar_mul_kernel` | Scalar multiplication |
| `gl_batch_exp_kernel` | Element-wise exponentiation |
| `gl_batch_fma_kernel` | Fused multiply-add |
| `gl_sum_reduce_kernel` | Sum reduction |
| `gl_dot_product_kernel` | Dot product |

## Performance Notes

- **Non-canonical representation** reduces branch divergence
- **PTX assembly** used for 64-bit multiplication
- **Constant memory** stores pre-computed powers and roots
- **Batch operations** maximize GPU utilization
- Consider using **Montgomery batch inversion** when computing many inverses

---

# Poseidon2 Hash Function

CUDA implementation of Poseidon2 for the Goldilocks field.

## Poseidon2 Properties

- **S-box**: x^7 (degree 7)
- **Rounds**: 4 initial external + 22 internal + 4 terminal external
- **Widths**: 8 or 16 field elements
- **Security**: 128-bit

## Poseidon2 Usage

### Initialization

```cpp
#include "poseidon2.cuh"

// Initialize both Goldilocks and Poseidon2 constants
goldilocks_init();
poseidon2_init();
```

### Single Permutation (Device)

```cpp
__global__ void my_kernel() {
    GoldilocksField state[8] = {0};

    // Fill state with data...
    state[0] = GoldilocksField(123);

    // Apply permutation in-place
    poseidon2_permute_8(state);
}
```

### Hashing (Device)

```cpp
__device__ void hash_data() {
    GoldilocksField input[16];  // Input data
    GoldilocksField output[4];  // Hash output

    // Hash using width-8 sponge (rate=4, output=4)
    poseidon2_hash_8_4(input, 16, output);
}
```

### Batch Permutation (Host)

```cpp
// Allocate device memory for batch_size states
GoldilocksField *d_states_in, *d_states_out;
cudaMalloc(&d_states_in, batch_size * 8 * sizeof(GoldilocksField));
cudaMalloc(&d_states_out, batch_size * 8 * sizeof(GoldilocksField));

// Run batch permutation
poseidon2_batch_permute_8(d_states_in, d_states_out, batch_size);
```

### Merkle Tree Construction

```cpp
// Build Merkle tree with 1024 leaves (4 elements each)
const int NUM_LEAVES = 1024;
const int CHUNK_SIZE = 4;
const int TREE_SIZE = (2 * NUM_LEAVES - 1) * CHUNK_SIZE;

GoldilocksField* d_tree;
cudaMalloc(&d_tree, TREE_SIZE * sizeof(GoldilocksField));

// Copy leaves to first portion
cudaMemcpy(d_tree, h_leaves, NUM_LEAVES * CHUNK_SIZE * sizeof(GoldilocksField),
           cudaMemcpyHostToDevice);

// Build tree (computes all parent nodes)
poseidon2_build_merkle_tree_8(d_tree, NUM_LEAVES);

// Root is at the end of the tree
```

### 2-to-1 Compression (Device)

```cpp
__device__ void compress_pair() {
    GoldilocksField left[4], right[4], output[4];

    // Compress two chunks into one
    poseidon2_compress_8(left, right, output);
}
```

## Poseidon2 API Reference

### Permutation Functions

| Function | Description |
|----------|-------------|
| `poseidon2_permute_8(state)` | Apply width-8 permutation in-place |
| `poseidon2_permute_16(state)` | Apply width-16 permutation in-place |

### Hash Functions

| Function | Description |
|----------|-------------|
| `poseidon2_hash_8_4(in, len, out)` | Hash with width-8, rate-4, output-4 |
| `poseidon2_hash_16_8(in, len, out)` | Hash with width-16, rate-8, output-8 |

### Compression Functions

| Function | Description |
|----------|-------------|
| `poseidon2_compress_8(left, right, out)` | 2-to-1 compression (4-element chunks) |
| `poseidon2_compress_16(left, right, out)` | 2-to-1 compression (8-element chunks) |

### Batch Kernels

| Kernel | Description |
|--------|-------------|
| `poseidon2_batch_permute_8_kernel` | Batch width-8 permutation |
| `poseidon2_batch_permute_16_kernel` | Batch width-16 permutation |
| `poseidon2_batch_hash_8_to_4_kernel` | Batch hashing |
| `poseidon2_merkle_layer_8_kernel` | One Merkle tree layer |

### Host Wrappers

| Function | Description |
|----------|-------------|
| `poseidon2_batch_permute_8()` | Batch permutation wrapper |
| `poseidon2_batch_hash_8()` | Batch hash wrapper |
| `poseidon2_merkle_layer_8()` | Merkle layer wrapper |
| `poseidon2_build_merkle_tree_8()` | Build complete Merkle tree |

## Poseidon2 Performance

Typical performance on modern GPUs:
- **Batch permutation**: ~1M+ permutations/second
- **Merkle tree**: Efficient parallel construction layer-by-layer

## Extension Field Hashing

Poseidon2 supports hashing extension field elements via serialization.

### Hashing GF(p²) Elements

```cpp
// Device-side: Hash single element
GoldilocksExt2 input(123, 456);  // 123 + 456*X
GoldilocksField output[4];
poseidon2_hash_ext2(input, output);

// Device-side: Hash array of elements
GoldilocksExt2 inputs[10];
GoldilocksField output[4];
poseidon2_hash_ext2_array(inputs, 10, output);

// Host-side: Batch hash
poseidon2_batch_hash_ext2(d_inputs, d_outputs, batch_size);
```

### Hashing GF(p⁵) Elements

```cpp
// Device-side: Hash single element
GoldilocksExt5 input(1, 2, 3, 4, 5);
GoldilocksField output[4];
poseidon2_hash_ext5(input, output);

// Host-side: Batch hash
poseidon2_batch_hash_ext5(d_inputs, d_outputs, batch_size);
```

### Merkle Trees over Extension Fields

```cpp
// Build Merkle tree with GF(p²) leaves
const int NUM_LEAVES = 1024;
const int TREE_SIZE = 2 * NUM_LEAVES - 1;

GoldilocksExt2* d_tree;
cudaMalloc(&d_tree, TREE_SIZE * sizeof(GoldilocksExt2));

// Copy leaves to d_tree[0..NUM_LEAVES-1]
// ...

// Build tree
poseidon2_build_merkle_tree_ext2(d_tree, NUM_LEAVES);

// Root is at d_tree[TREE_SIZE - 1]
```

### Extension Field Hashing API

| Function | Description |
|----------|-------------|
| `poseidon2_hash_ext2(in, out)` | Hash single GF(p²) element |
| `poseidon2_hash_ext2_array(in, len, out)` | Hash array of GF(p²) elements |
| `poseidon2_hash_ext2_to_ext2(in, len, out)` | Hash and return as GF(p²) |
| `poseidon2_hash_ext5(in, out)` | Hash single GF(p⁵) element |
| `poseidon2_compress_ext2(l, r, out)` | 2-to-1 compression for GF(p²) |
| `poseidon2_compress_ext5(l, r, out)` | 2-to-1 compression for GF(p⁵) |
| `poseidon2_batch_hash_ext2()` | Batch hash GF(p²) elements |
| `poseidon2_batch_hash_ext5()` | Batch hash GF(p⁵) elements |
| `poseidon2_merkle_layer_ext2()` | Merkle tree layer for GF(p²) |
| `poseidon2_build_merkle_tree_ext2()` | Build complete Merkle tree |

---

# Extension Fields

CUDA implementation of Goldilocks extension fields.

## Supported Extensions

| Extension | Degree | Irreducible Polynomial | W | Element Size |
|-----------|--------|------------------------|---|--------------|
| GF(p²) | 2 | X² - 7 | 7 | 128 bits |
| GF(p⁵) | 5 | X⁵ - 3 | 3 | 320 bits |

## Extension Field Usage

### Quadratic Extension GF(p²)

```cpp
#include "extension.cuh"

// Create elements: a = 1 + 2X, b = 3 + 4X
GoldilocksExt2 a(1, 2);
GoldilocksExt2 b(3, 4);

// Arithmetic (device code)
GoldilocksExt2 sum = ext2_add(a, b);
GoldilocksExt2 diff = ext2_sub(a, b);
GoldilocksExt2 prod = ext2_mul(a, b);
GoldilocksExt2 sq = ext2_square(a);
GoldilocksExt2 inv = ext2_inverse(a);
GoldilocksExt2 quot = ext2_div(a, b);

// Frobenius: a -> a^p (conjugation for degree 2)
GoldilocksExt2 frob = ext2_frobenius(a);
GoldilocksExt2 conj = ext2_conjugate(a);  // Same as frobenius

// Norm to base field: a * conj(a)
GoldilocksField norm = ext2_norm(a);

// Exponentiation
GoldilocksExt2 power = ext2_exp(a, 100);
```

### Quintic Extension GF(p⁵)

```cpp
// Create element: a = 1 + X + 2X² + 3X³ + 4X⁴
GoldilocksExt5 a(1, 1, 2, 3, 4);

// Arithmetic (device code)
GoldilocksExt5 sum = ext5_add(a, b);
GoldilocksExt5 prod = ext5_mul(a, b);
GoldilocksExt5 sq = ext5_square(a);
GoldilocksExt5 inv = ext5_inverse(a);  // Uses Frobenius-based algorithm

// Frobenius: a -> a^p
GoldilocksExt5 frob = ext5_frobenius(a);

// Repeated Frobenius: a -> a^(p^count)
GoldilocksExt5 frob_k = ext5_repeated_frobenius(a, 3);  // a^(p³)
```

### Batch Operations

```cpp
// Allocate device memory
GoldilocksExt2 *d_a, *d_b, *d_result;
cudaMalloc(&d_a, n * sizeof(GoldilocksExt2));
cudaMalloc(&d_b, n * sizeof(GoldilocksExt2));
cudaMalloc(&d_result, n * sizeof(GoldilocksExt2));

// Batch operations (host wrappers)
ext2_batch_add(d_a, d_b, d_result, n);
ext2_batch_mul(d_a, d_b, d_result, n);
ext2_batch_square(d_a, d_result, n);
ext2_batch_inverse(d_a, d_result, n);

// Quintic extension
ext5_batch_mul(d_a5, d_b5, d_result5, n);
ext5_batch_inverse(d_a5, d_result5, n);
```

## Extension Field API Reference

### Quadratic Extension (GF(p²))

| Function | Description |
|----------|-------------|
| `ext2_add(a, b)` | Addition |
| `ext2_sub(a, b)` | Subtraction |
| `ext2_neg(a)` | Negation |
| `ext2_mul(a, b)` | Multiplication |
| `ext2_square(a)` | Squaring (optimized) |
| `ext2_inverse(a)` | Inversion using norm |
| `ext2_div(a, b)` | Division |
| `ext2_scalar_mul(s, a)` | Scalar multiplication |
| `ext2_frobenius(a)` | Frobenius: a → a^p |
| `ext2_conjugate(a)` | Conjugation: a₀ + a₁X → a₀ - a₁X |
| `ext2_norm(a)` | Norm to base field |
| `ext2_exp(a, n)` | Exponentiation |

### Quintic Extension (GF(p⁵))

| Function | Description |
|----------|-------------|
| `ext5_add(a, b)` | Addition |
| `ext5_sub(a, b)` | Subtraction |
| `ext5_neg(a)` | Negation |
| `ext5_mul(a, b)` | Multiplication |
| `ext5_square(a)` | Squaring (optimized) |
| `ext5_inverse(a)` | Inversion (Frobenius-based) |
| `ext5_div(a, b)` | Division |
| `ext5_scalar_mul(s, a)` | Scalar multiplication |
| `ext5_frobenius(a)` | Frobenius: a → a^p |
| `ext5_repeated_frobenius(a, k)` | a → a^(p^k) |
| `ext5_exp(a, n)` | Exponentiation |

### Batch Kernels

| Kernel | Description |
|--------|-------------|
| `ext2_batch_add_kernel` | Batch GF(p²) addition |
| `ext2_batch_mul_kernel` | Batch GF(p²) multiplication |
| `ext2_batch_square_kernel` | Batch GF(p²) squaring |
| `ext2_batch_inverse_kernel` | Batch GF(p²) inversion |
| `ext2_batch_frobenius_kernel` | Batch Frobenius |
| `ext5_batch_*_kernel` | Corresponding GF(p⁵) kernels |

## Extension Field Optimizations

- **Squaring**: Uses ~60% fewer operations than general multiplication
- **Inversion (GF(p²))**: Direct formula using norm (1 base field inversion)
- **Inversion (GF(p⁵))**: Frobenius-based logarithmic algorithm (1 base field inversion + few multiplications)
- **Frobenius**: O(D) multiplications by pre-computed DTH_ROOT powers
