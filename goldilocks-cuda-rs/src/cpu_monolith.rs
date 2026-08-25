//! CPU-only Monolith hash functions for Merkle tree verification.
//!
//! Provides leaf hashing and auth path verification matching the GPU kernels
//! in `cuda/monolith.cuh` and `cuda/monolith_kernels.cu`.

use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::poseidon2::Poseidon2Hash;

const WIDTH: usize = 12;
const NUM_BARS: usize = 4;
const N_ROUNDS: usize = 6;

// ============================================================================
// Monolith Constants (LOOKUP_BITS = 8, Goldilocks field)
// ============================================================================

const ROUND_CONSTANTS: [[u64; 12]; 7] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [13596126580325903823, 5676126986831820406, 11349149288412960427,
     3368797843020733411, 16240671731749717664, 9273190757374900239,
     14446552112110239438, 4033077683985131644, 4291229347329361293,
     13231607645683636062, 1383651072186713277, 8898815177417587567],
    [2383619671172821638, 6065528368924797662, 16737578966352303081,
     2661700069680749654, 7414030722730336790, 18124970299993404776,
     9169923000283400738, 15832813151034110977, 16245117847613094506,
     11056181639108379773, 10546400734398052938, 8443860941261719174],
    [15799082741422909885, 13421235861052008152, 15448208253823605561,
     2540286744040770964, 2895626806801935918, 8644593510196221619,
     17722491003064835823, 5166255496419771636, 1015740739405252346,
     4400043467547597488, 5176473243271652644, 4517904634837939508],
    [18341030605319882173, 13366339881666916534, 6291492342503367536,
     10004214885638819819, 4748655089269860551, 1520762444865670308,
     8393589389936386108, 11025183333304586284, 5993305003203422738,
     458912836931247573, 5947003897778655410, 17184667486285295106],
    [15710528677110011358, 8929476121507374707, 2351989866172789037,
     11264145846854799752, 14924075362538455764, 10107004551857451916,
     18325221206052792232, 16751515052585522105, 15305034267720085905,
     15639149412312342017, 14624541102106656564, 3542311898554959098],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

const MDS: [[u64; 12]; 12] = [
    [ 7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8],
    [ 8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22, 21],
    [21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6, 22],
    [22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7,  6],
    [ 6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9,  7],
    [ 7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10,  9],
    [ 9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13, 10],
    [10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26, 13],
    [13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8, 26],
    [26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23,  8],
    [ 8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7, 23],
    [23,  8, 26, 13, 10,  9,  7,  6, 22, 21,  8,  7],
];

// ============================================================================
// Field helpers
// ============================================================================

const P: u64 = 0xFFFFFFFF00000001; // 2^64 - 2^32 + 1

#[inline]
fn from_u96(lo: u64, hi: u64) -> u64 {
    // 2^64 ≡ 2^32 - 1 (mod p)
    // x = lo + hi * 2^64 ≡ lo + hi * (2^32 - 1) (mod p)
    let correction = (hi as u128) * (0xFFFFFFFF_u128);
    let sum = (lo as u128) + correction;
    let lo2 = sum as u64;
    let hi2 = (sum >> 64) as u64;
    if hi2 == 0 {
        if lo2 >= P { lo2 - P } else { lo2 }
    } else {
        from_u96(lo2, hi2)
    }
}

#[inline]
fn gl_mul(a: u64, b: u64) -> u64 {
    let prod = (a as u128) * (b as u128);
    from_u96(prod as u64, (prod >> 64) as u64)
}

#[inline]
fn gl_add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    // If carry or sum >= P, subtract P
    let (r, borrow) = sum.overflowing_sub(P);
    if carry || !borrow { r } else { sum }
}

// ============================================================================
// Monolith Layers
// ============================================================================

/// Bars: bitwise S-box on first 4 elements (LOOKUP_BITS = 8)
#[inline]
fn bar_64(limb: u64) -> u64 {
    let limbl1 = ((!limb & 0x8080808080808080) >> 7) | ((!limb & 0x7F7F7F7F7F7F7F7F) << 1);
    let limbl2 = ((limb & 0xC0C0C0C0C0C0C0C0) >> 6) | ((limb & 0x3F3F3F3F3F3F3F3F) << 2);
    let limbl3 = ((limb & 0xE0E0E0E0E0E0E0E0) >> 5) | ((limb & 0x1F1F1F1F1F1F1F1F) << 3);
    let tmp = limb ^ (limbl1 & limbl2 & limbl3);
    ((tmp & 0x8080808080808080) >> 7) | ((tmp & 0x7F7F7F7F7F7F7F7F) << 1)
}

fn bars(state: &mut [u64; WIDTH]) {
    for i in 0..NUM_BARS {
        state[i] = bar_64(state[i]);
    }
}

/// Bricks: Feistel Type-3 (reverse iteration)
fn bricks(state: &mut [u64; WIDTH]) {
    for i in (1..WIDTH).rev() {
        let sq = gl_mul(state[i - 1], state[i - 1]);
        state[i] = gl_add(state[i], sq);
    }
}

/// Concrete: MDS matrix multiply + round constants
fn concrete(state: &mut [u64; WIDTH], round: usize) {
    let mut result = [0u64; WIDTH];
    for row in 0..WIDTH {
        let mut acc: u128 = 0;
        for col in 0..WIDTH {
            acc += (state[col] as u128) * (MDS[row][col] as u128);
        }
        acc += ROUND_CONSTANTS[round][row] as u128;
        result[row] = from_u96(acc as u64, (acc >> 64) as u64);
    }
    *state = result;
}

// ============================================================================
// Full Monolith Permutation
// ============================================================================

pub fn monolith_permute(state: &mut [u64; WIDTH]) {
    concrete(state, 0);
    for r in 1..=N_ROUNDS {
        bars(state);
        bricks(state);
        concrete(state, r);
    }
}

// ============================================================================
// Merkle Tree Hash Functions
// ============================================================================

/// Compress two digests into one (2-to-1 compression).
/// Matches `monolith_compress` in `cuda/monolith.cuh`.
pub fn monolith_compress(left: &Poseidon2Hash, right: &Poseidon2Hash) -> Poseidon2Hash {
    let mut state = [0u64; WIDTH];
    state[0] = left.elements[0].0;
    state[1] = left.elements[1].0;
    state[2] = left.elements[2].0;
    state[3] = left.elements[3].0;
    state[4] = right.elements[0].0;
    state[5] = right.elements[1].0;
    state[6] = right.elements[2].0;
    state[7] = right.elements[3].0;
    // state[8..11] already zeroed
    monolith_permute(&mut state);
    Poseidon2Hash {
        elements: [
            GoldilocksField(state[0]),
            GoldilocksField(state[1]),
            GoldilocksField(state[2]),
            GoldilocksField(state[3]),
        ],
    }
}

/// Hash a GL codeword leaf pair into a digest.
pub fn hash_gl_leaf(a: GoldilocksField, b: GoldilocksField) -> Poseidon2Hash {
    let mut state = [0u64; WIDTH];
    state[0] = a.0;
    state[1] = b.0;
    monolith_permute(&mut state);
    Poseidon2Hash {
        elements: [
            GoldilocksField(state[0]),
            GoldilocksField(state[1]),
            GoldilocksField(state[2]),
            GoldilocksField(state[3]),
        ],
    }
}

/// Hash an ext2 codeword leaf pair into a digest.
pub fn hash_ext2_leaf(a: GoldilocksExt2, b: GoldilocksExt2) -> Poseidon2Hash {
    let mut state = [0u64; WIDTH];
    state[0] = a.c0.0;
    state[1] = a.c1.0;
    state[2] = b.c0.0;
    state[3] = b.c1.0;
    monolith_permute(&mut state);
    Poseidon2Hash {
        elements: [
            GoldilocksField(state[0]),
            GoldilocksField(state[1]),
            GoldilocksField(state[2]),
            GoldilocksField(state[3]),
        ],
    }
}

/// Verify a Merkle authentication path.
pub fn verify_auth_path(
    leaf_hash: &Poseidon2Hash,
    path: &[Poseidon2Hash],
    leaf_index: usize,
    root: &Poseidon2Hash,
) -> bool {
    let mut current = *leaf_hash;
    let mut idx = leaf_index;

    for sibling in path {
        if idx & 1 == 0 {
            current = monolith_compress(&current, sibling);
        } else {
            current = monolith_compress(sibling, &current);
        }
        idx >>= 1;
    }

    current == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monolith_permute_test_vector() {
        // Reference test vector from research/monolith (LOOKUP_BITS = 8)
        let mut state = [0u64; 12];
        for i in 0..12 {
            state[i] = i as u64;
        }
        monolith_permute(&mut state);

        let expected: [u64; 12] = [
            5867581605548782913, 588867029099903233, 6043817495575026667,
            805786589926590032, 9919982299747097782, 6718641691835914685,
            7951881005429661950, 15453177927755089358, 974633365445157727,
            9654662171963364206, 6281307445101925412, 13745376999934453119,
        ];

        assert_eq!(state, expected, "Monolith CPU permutation does not match reference test vector");
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_permutation_large_value() {
        crate::init().unwrap();
        // Input that causes leaf hash mismatch: state[0]=8, state[1]=p-8, rest=0
        // This is the hash_gl_leaf(8, p-8) case
        let mut input = [0u64; 12];
        input[0] = 8;
        input[1] = 18446744069414584313; // p - 8
        let mut cpu_output = input;
        monolith_permute(&mut cpu_output);

        let mut gpu_output = [0u64; 12];
        let ret = unsafe {
            crate::ffi::monolith_permute_test_ffi(input.as_ptr(), gpu_output.as_mut_ptr())
        };
        assert_eq!(ret, 0, "GPU monolith_permute_test_ffi failed");

        for i in 0..12 {
            if cpu_output[i] != gpu_output[i] {
                eprintln!("[PERMUTE] state[{}] mismatch: CPU={} GPU={}", i, cpu_output[i], gpu_output[i]);
            }
        }
        assert_eq!(cpu_output, gpu_output, "GPU Monolith permute differs from CPU for large input values");
    }

    #[test]
    fn test_monolith_compress_deterministic() {
        let h1 = Poseidon2Hash::from_raw([1, 2, 3, 4]);
        let h2 = Poseidon2Hash::from_raw([5, 6, 7, 8]);
        let r1 = monolith_compress(&h1, &h2);
        let r2 = monolith_compress(&h1, &h2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_merkle() {
        crate::init().unwrap();
        use crate::memory::{DeviceBuffer, synchronize};
        use crate::merkle::DeviceMerkleTree;
        use crate::poseidon2::POSEIDON2_DIGEST_SIZE;

        // Create a small GL codeword: 8 elements → 4 leaves
        let codeword: Vec<u64> = (100..108).collect();
        let n = codeword.len();
        let num_leaves = n / 2;
        let d_cw = DeviceBuffer::from_slice(&codeword).unwrap();

        // GPU tree
        let gpu_tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        synchronize().unwrap();
        let gpu_root = gpu_tree.root().unwrap();

        // CPU tree: hash leaves, then compress
        let leaf0 = hash_gl_leaf(GoldilocksField(100), GoldilocksField(101));
        let leaf1 = hash_gl_leaf(GoldilocksField(102), GoldilocksField(103));
        let leaf2 = hash_gl_leaf(GoldilocksField(104), GoldilocksField(105));
        let leaf3 = hash_gl_leaf(GoldilocksField(106), GoldilocksField(107));
        let mid0 = monolith_compress(&leaf0, &leaf1);
        let mid1 = monolith_compress(&leaf2, &leaf3);
        let cpu_root = monolith_compress(&mid0, &mid1);

        assert_eq!(gpu_root, cpu_root,
            "GPU Monolith Merkle root {:?} != CPU Monolith root {:?}", gpu_root, cpu_root);
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_merkle_large() {
        crate::init().unwrap();
        use crate::memory::{DeviceBuffer, synchronize};
        use crate::merkle::DeviceMerkleTree;

        // 32 elements → 16 leaves → matches basefold test with num_vars=4, log_rate=1
        let codeword: Vec<u64> = (1..=32).collect();
        let n = codeword.len();
        let num_leaves = n / 2;
        let d_cw = DeviceBuffer::from_slice(&codeword).unwrap();

        let gpu_tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        synchronize().unwrap();
        let gpu_root = gpu_tree.root().unwrap();

        // CPU tree
        let mut leaves = Vec::new();
        for i in 0..num_leaves {
            leaves.push(hash_gl_leaf(GoldilocksField(codeword[2*i]), GoldilocksField(codeword[2*i+1])));
        }
        let mut layer = leaves;
        while layer.len() > 1 {
            let next: Vec<_> = (0..layer.len()/2).map(|i| monolith_compress(&layer[2*i], &layer[2*i+1])).collect();
            layer = next;
        }
        let cpu_root = layer[0];

        assert_eq!(gpu_root, cpu_root,
            "GPU Monolith large Merkle root {:?} != CPU root {:?}", gpu_root, cpu_root);

        // Also verify auth paths — test ALL leaf indices
        for leaf_idx in 0..num_leaves {
            let leaf_hash = hash_gl_leaf(
                GoldilocksField(codeword[2*leaf_idx]),
                GoldilocksField(codeword[2*leaf_idx+1]),
            );
            let path = gpu_tree.auth_path(leaf_idx).unwrap();
            assert!(verify_auth_path(&leaf_hash, &path, leaf_idx, &gpu_root),
                "Auth path verification failed for leaf {}", leaf_idx);
        }
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_specific_input() {
        crate::init().unwrap();
        use crate::memory::{DeviceBuffer, synchronize};
        use crate::merkle::DeviceMerkleTree;

        // Exact values from basefold test that cause leaf 1 mismatch
        let cw: Vec<u64> = vec![1, 2, 8, 18446744069414584313, 12, 18446744069414584317, 7, 8];
        let n = cw.len();
        let d_cw = DeviceBuffer::from_slice(&cw).unwrap();
        let gpu_tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        synchronize().unwrap();

        let num_leaves = n / 2;
        for i in 0..num_leaves {
            let gpu_leaf = gpu_tree.leaf_digest(i).unwrap();
            let cpu_leaf = hash_gl_leaf(GoldilocksField(cw[2 * i]), GoldilocksField(cw[2 * i + 1]));
            if gpu_leaf != cpu_leaf {
                eprintln!("[SPECIFIC] leaf {} MISMATCH: inputs=({}, {})", i, cw[2*i], cw[2*i+1]);
                eprintln!("  GPU: {:?}", gpu_leaf.to_raw());
                eprintln!("  CPU: {:?}", cpu_leaf.to_raw());

                // Also test with the value reduced mod p
                let p: u64 = 0xFFFFFFFF00000001;
                let a_reduced = cw[2*i] % p;
                let b_reduced = cw[2*i+1] % p;
                let cpu_reduced = hash_gl_leaf(GoldilocksField(a_reduced), GoldilocksField(b_reduced));
                eprintln!("  CPU_reduced: {:?} (inputs reduced to {} {})", cpu_reduced.to_raw(), a_reduced, b_reduced);
            }
            assert_eq!(gpu_leaf, cpu_leaf, "Leaf {} mismatch for inputs ({}, {})", i, cw[2*i], cw[2*i+1]);
        }
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_merkle_large_random() {
        crate::init().unwrap();
        use crate::memory::{DeviceBuffer, synchronize};
        use crate::merkle::DeviceMerkleTree;

        // Use large field values like basefold codewords
        let codeword: Vec<u64> = vec![
            0xFFFFFFFF00000000, 0x123456789ABCDEF0, 0xDEADBEEFCAFEBABE, 0x0102030405060708,
            0xAAAAAAAABBBBBBBB, 0xCCCCCCCCDDDDDDDD, 0xEEEEEEEEFFFFFFFF, 0x9876543210FEDCBA,
            0x1111111122222222, 0x3333333344444444, 0x5555555566666666, 0x7777777788888888,
            0x99999999AAAAAAAA, 0xBBBBBBBBCCCCCCCC, 0xDDDDDDDDEEEEEEEE, 0xFFFFFFFF11111111,
            0x2222222233333333, 0x4444444455555555, 0x6666666677777777, 0x8888888899999999,
            0xAAAAAAAABBBBBBBB, 0xCCCCCCCCDDDDDDDD, 0xEEEEEEEEFFFFFFFF, 0x0000000011111111,
            0x2222222233333333, 0x4444444455555555, 0x6666666677777777, 0x8888888899999999,
            0xABCDEF0123456789, 0xFEDCBA9876543210, 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0,
        ];
        let n = codeword.len();
        let num_leaves = n / 2;
        let d_cw = DeviceBuffer::from_slice(&codeword).unwrap();

        let gpu_tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        synchronize().unwrap();
        let gpu_root = gpu_tree.root().unwrap();

        // CPU tree
        let mut leaves = Vec::new();
        for i in 0..num_leaves {
            leaves.push(hash_gl_leaf(GoldilocksField(codeword[2*i]), GoldilocksField(codeword[2*i+1])));
        }
        let mut layer = leaves;
        while layer.len() > 1 {
            let next: Vec<_> = (0..layer.len()/2).map(|i| monolith_compress(&layer[2*i], &layer[2*i+1])).collect();
            layer = next;
        }
        let cpu_root = layer[0];

        assert_eq!(gpu_root, cpu_root,
            "GPU Monolith large random root {:?} != CPU root {:?}", gpu_root, cpu_root);

        // Verify auth paths
        for leaf_idx in 0..num_leaves {
            let leaf_hash = hash_gl_leaf(GoldilocksField(codeword[2*leaf_idx]), GoldilocksField(codeword[2*leaf_idx+1]));
            let path = gpu_tree.auth_path(leaf_idx).unwrap();
            assert!(verify_auth_path(&leaf_hash, &path, leaf_idx, &gpu_root),
                "Auth path failed for leaf {}", leaf_idx);
        }
    }

    #[test]
    fn test_monolith_gpu_vs_cpu_merkle_ext2() {
        crate::init().unwrap();
        use crate::memory::{DeviceBuffer, synchronize};
        use crate::merkle::DeviceMerkleTree;
        use crate::extension::GoldilocksExt2;

        // 4 ext2 elements → 2 leaves
        let cw = vec![
            GoldilocksExt2::new(GoldilocksField(10), GoldilocksField(20)),
            GoldilocksExt2::new(GoldilocksField(30), GoldilocksField(40)),
            GoldilocksExt2::new(GoldilocksField(50), GoldilocksField(60)),
            GoldilocksExt2::new(GoldilocksField(70), GoldilocksField(80)),
        ];
        let raw: Vec<u64> = cw.iter().flat_map(|e| vec![e.c0.0, e.c1.0]).collect();
        let d_cw = DeviceBuffer::from_slice(&raw).unwrap();

        let gpu_tree = DeviceMerkleTree::build_from_ext2_codeword(&d_cw, 4).unwrap();
        synchronize().unwrap();
        let gpu_root = gpu_tree.root().unwrap();

        // CPU: hash leaf pairs, then compress
        let leaf0 = hash_ext2_leaf(cw[0], cw[1]);
        let leaf1 = hash_ext2_leaf(cw[2], cw[3]);
        let cpu_root = monolith_compress(&leaf0, &leaf1);

        assert_eq!(gpu_root, cpu_root,
            "GPU Monolith ext2 Merkle root {:?} != CPU root {:?}", gpu_root, cpu_root);
    }
}
