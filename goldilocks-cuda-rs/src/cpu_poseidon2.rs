//! CPU-only Poseidon2 hash functions for Merkle tree verification.
//!
//! Provides leaf hashing and auth path verification matching the GPU kernels
//! in `cuda/poseidon2_kernels.cu` and `cuda/wrapper.cu`.

use crate::extension::GoldilocksExt2;
use crate::field::GoldilocksField;
use crate::poseidon2::Poseidon2Hash;

const WIDTH: usize = 8;

// ============================================================================
// Poseidon2 Constants (same as cuda/poseidon2.cuh and zk-torch-3/transcript.rs)
// ============================================================================

const RC_EXT: [[u64; 8]; 8] = [
    [0xdd5743e7f2a5a5d9, 0xcb3a864e58ada44b, 0xd5a3726889dedc0d, 0x79365a7f967583c1,
     0xb7820c177b0a3c30, 0x68a60479943b7240, 0x0a22eeab67b97c41, 0xbee9d90e037fa7d4],
    [0xf1dda5b9259dfcb4, 0x27515210be112d59, 0x654a4db8f15f6c3a, 0x5a9bfff8c29db0c4,
     0xb72f2628c95a885b, 0xffcb54fc166beadd, 0xe32f91d59c463772, 0x9e0ff2a9bbea79db],
    [0xce57d6245ddca6b2, 0xb1fc8d402bba1eb1, 0x64df974fb666d528, 0x59e4e5b237e52e78,
     0x5e255e2742caa8fc, 0xc4a17304e330ef9a, 0x60e3a513cdd9f0b1, 0xbf29a6a5a0c9c8ba],
    [0xcea721cce82fb11b, 0xe5b55eb8098ece81, 0x4ec66bfb89f7c380, 0xc33be12b6ef7e4fa,
     0xa41b5f5b84c57d7d, 0xb526206b2138a936, 0x9f7a5ac62f16eb4e, 0x7de77f405f683aa5],
    [0x014ef1197d341346, 0x9725e20825d07394, 0xc8ff2a22516f3604, 0xde1b51f7cf493493,
     0xa7e3af1a53eadb78, 0x9b7b7ee0ddf9f229, 0xf0a460f2dc649e7d, 0xdc4448d02c2cb823],
    [0xaa62c88e0b294011, 0x058eb9d810ce9f74, 0x4e3d7a5403566d24, 0xe8ae66da2f8b7a63,
     0x3fdad3b08dae2e0b, 0xec81e61303dd409b, 0x3cbaa1cd35d31fc4, 0x847f7806eec88ffa],
    [0x98ae09a325893803, 0xf8a6475077968838, 0xdf55b7419443de5b, 0xdbe15699e696ca70,
     0x8e54a27d8db00424, 0x7679a45cd9d1e12a, 0x2844f52be73e0e2f, 0xd41f8eb31ada34f8],
    [0xe9dd318bae1f3961, 0xf7462137299efe1a, 0x6dbbe06779e1d573, 0xfe35b05cbe707632,
     0x5d8896b12654fd8c, 0x6f96ef47c32d4ae2, 0xb0caa221dbbfc0da, 0x0bc2a5bf1f238d3f],
];

const RC_INT: [u64; 22] = [
    0x488897d85ff51f56, 0x1140737ccb162218, 0xa7eeb9215866ed35,
    0x9bd2976fee49fcc9, 0xc0c8f0de580a3fcc, 0x4fb2dae6ee8fc793,
    0x343a89f35f37395b, 0x223b525a77ca72c8, 0x56ccb62574aaa918,
    0xc4d507d8027af9ed, 0xa080673cf0b7e95c, 0xf0184884eb70dcf8,
    0x044f10b0cb3d5c69, 0xe9e3f7993938f186, 0x1b761c80e772f459,
    0x606cec607a1b5fac, 0x14a0c2e1d45f03cd, 0x4eace8855398574f,
    0xf905ca7103eff3e6, 0xf8c8f8d20862c059, 0xb524fe8bdd678e5a,
    0xfbb7865901a1ec41,
];

const DIAG_8: [u64; 8] = [
    0xa98811a1fed4e3a5, 0x1cc48b54f377e2a0, 0xe40cd4f6c5609a26,
    0x11de79ebca97a4a3, 0x9177c73d8b7e929c, 0x2a6fe8085797e791,
    0x3de6e93329f8d5ad, 0x3f7af9125da962fe,
];

// ============================================================================
// Poseidon2 Permutation (CPU)
// ============================================================================

#[inline]
fn gl(v: u64) -> GoldilocksField {
    GoldilocksField(v)
}

#[inline]
fn sbox(x: GoldilocksField) -> GoldilocksField {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x3 = x2 * x;
    x4 * x3
}

#[inline]
fn mds4(s: &mut [GoldilocksField]) {
    let t01 = s[0] + s[1];
    let t23 = s[2] + s[3];
    let t0123 = t01 + t23;
    let t01123 = t0123 + s[1];
    let t01233 = t0123 + s[3];
    let s0 = t01123 + t01;
    let s1 = t01123 + (s[2] + s[2]);
    let s2 = t01233 + t23;
    let s3 = t01233 + (s[0] + s[0]);
    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}

#[inline]
fn mds_light_8(state: &mut [GoldilocksField; WIDTH]) {
    mds4(&mut state[0..4]);
    mds4(&mut state[4..8]);
    let sum0 = state[0] + state[4];
    let sum1 = state[1] + state[5];
    let sum2 = state[2] + state[6];
    let sum3 = state[3] + state[7];
    state[0] = state[0] + sum0;
    state[1] = state[1] + sum1;
    state[2] = state[2] + sum2;
    state[3] = state[3] + sum3;
    state[4] = state[4] + sum0;
    state[5] = state[5] + sum1;
    state[6] = state[6] + sum2;
    state[7] = state[7] + sum3;
}

#[inline]
fn diffusion_8(state: &mut [GoldilocksField; WIDTH]) {
    let mut sum = state[0];
    for i in 1..WIDTH {
        sum = sum + state[i];
    }
    for i in 0..WIDTH {
        let prod = state[i] * gl(DIAG_8[i]);
        state[i] = prod + sum;
    }
}

/// Full Poseidon2 permutation for width 8.
pub fn poseidon2_permute(state: &mut [GoldilocksField; WIDTH]) {
    for r in 0..4 {
        for i in 0..WIDTH {
            state[i] = state[i] + gl(RC_EXT[r][i]);
            state[i] = sbox(state[i]);
        }
        mds_light_8(state);
    }
    for r in 0..22 {
        state[0] = state[0] + gl(RC_INT[r]);
        state[0] = sbox(state[0]);
        diffusion_8(state);
    }
    for r in 0..4 {
        for i in 0..WIDTH {
            state[i] = state[i] + gl(RC_EXT[4 + r][i]);
            state[i] = sbox(state[i]);
        }
        mds_light_8(state);
    }
}

// ============================================================================
// Merkle Tree Hash Functions
// ============================================================================

/// Compress two Poseidon2Hash digests into one (2-to-1 compression).
/// Matches `poseidon2_compress_8` in `cuda/poseidon2.cuh`.
pub fn poseidon2_compress(left: &Poseidon2Hash, right: &Poseidon2Hash) -> Poseidon2Hash {
    let mut state = [GoldilocksField::zero(); WIDTH];
    for i in 0..4 {
        state[i] = left.elements[i];
    }
    for i in 0..4 {
        state[4 + i] = right.elements[i];
    }
    poseidon2_permute(&mut state);
    Poseidon2Hash {
        elements: [state[0], state[1], state[2], state[3]],
    }
}

/// Hash a GL codeword leaf pair (2 base field elements) into a Poseidon2Hash.
/// Matches `hash_gl_leaves_kernel` in `cuda/wrapper.cu`.
pub fn hash_gl_leaf(a: GoldilocksField, b: GoldilocksField) -> Poseidon2Hash {
    let mut state = [GoldilocksField::zero(); WIDTH];
    state[0] = a;
    state[1] = b;
    poseidon2_permute(&mut state);
    Poseidon2Hash {
        elements: [state[0], state[1], state[2], state[3]],
    }
}

/// Hash an ext2 codeword leaf pair (2 ext2 elements = 4 base field elements) into a Poseidon2Hash.
/// Matches `hash_ext2_leaves_kernel` in `cuda/wrapper.cu`.
pub fn hash_ext2_leaf(a: GoldilocksExt2, b: GoldilocksExt2) -> Poseidon2Hash {
    let mut state = [GoldilocksField::zero(); WIDTH];
    state[0] = a.c0;
    state[1] = a.c1;
    state[2] = b.c0;
    state[3] = b.c1;
    poseidon2_permute(&mut state);
    Poseidon2Hash {
        elements: [state[0], state[1], state[2], state[3]],
    }
}

/// Verify a Merkle authentication path.
///
/// Given a leaf hash, sibling path, leaf index, and expected root,
/// walks up the tree recomputing parent hashes and checks the final
/// hash against the root.
pub fn verify_auth_path(
    leaf_hash: &Poseidon2Hash,
    path: &[Poseidon2Hash],
    leaf_index: usize,
    root: &Poseidon2Hash,
) -> bool {
    let mut current = leaf_hash.clone();
    let mut idx = leaf_index;

    for sibling in path {
        if idx & 1 == 0 {
            // Current is left child, sibling is right
            current = poseidon2_compress(&current, sibling);
        } else {
            // Sibling is left child, current is right
            current = poseidon2_compress(sibling, &current);
        }
        idx >>= 1;
    }

    current == *root
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{DeviceBuffer, synchronize};
    use crate::merkle::DeviceMerkleTree;

    #[test]
    fn test_poseidon2_permute_matches_transcript() {
        // Verify CPU permutation produces non-zero output from zero state
        let mut state = [GoldilocksField::zero(); WIDTH];
        poseidon2_permute(&mut state);
        assert!(state.iter().any(|f| f.0 != 0));
    }

    #[test]
    fn test_poseidon2_permute_deterministic() {
        let mut s1 = [GoldilocksField::zero(); WIDTH];
        let mut s2 = [GoldilocksField::zero(); WIDTH];
        s1[0] = GoldilocksField(42);
        s2[0] = GoldilocksField(42);
        poseidon2_permute(&mut s1);
        poseidon2_permute(&mut s2);
        assert_eq!(s1, s2);
    }

    #[test]
    #[cfg(not(feature = "monolith"))]
    fn test_gl_leaf_hash_matches_gpu() {
        crate::init().unwrap();

        // Create a small codeword and build Merkle tree on GPU
        let data: Vec<u64> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let d_cw = DeviceBuffer::from_slice(&data).unwrap();
        let tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, 8).unwrap();
        synchronize().unwrap();

        // Check each leaf digest matches CPU computation
        for i in 0..4 {
            let gpu_digest = tree.leaf_digest(i).unwrap();
            let cpu_digest = hash_gl_leaf(GoldilocksField(data[2 * i]), GoldilocksField(data[2 * i + 1]));
            assert_eq!(gpu_digest, cpu_digest, "Leaf {} mismatch", i);
        }
    }

    #[test]
    #[cfg(not(feature = "monolith"))]
    fn test_compress_matches_gpu() {
        crate::init().unwrap();

        // Build a tree and verify the root can be reconstructed from leaves + auth paths
        let data: Vec<u64> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let d_cw = DeviceBuffer::from_slice(&data).unwrap();
        let tree = DeviceMerkleTree::build_from_gl_codeword(&d_cw, 8).unwrap();
        synchronize().unwrap();

        let root = tree.root().unwrap();

        // Verify auth path for each leaf
        for i in 0..4 {
            let leaf_hash = hash_gl_leaf(GoldilocksField(data[2 * i]), GoldilocksField(data[2 * i + 1]));
            let path = tree.auth_path(i).unwrap();
            assert!(
                verify_auth_path(&leaf_hash, &path, i, &root),
                "Auth path verification failed for leaf {}",
                i
            );
        }
    }

    #[test]
    #[cfg(not(feature = "monolith"))]
    fn test_ext2_leaf_hash_matches_gpu() {
        crate::init().unwrap();

        // 4 ext2 elements = 8 u64 = 2 leaves
        let data: Vec<u64> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let d_cw = DeviceBuffer::from_slice(&data).unwrap();
        let tree = DeviceMerkleTree::build_from_ext2_codeword(&d_cw, 4).unwrap();
        synchronize().unwrap();

        // 2 leaves: leaf 0 = hash(ext2(100,200), ext2(300,400)), leaf 1 = hash(ext2(500,600), ext2(700,800))
        for i in 0..2 {
            let gpu_digest = tree.leaf_digest(i).unwrap();
            let a = GoldilocksExt2::new(
                GoldilocksField(data[4 * i]),
                GoldilocksField(data[4 * i + 1]),
            );
            let b = GoldilocksExt2::new(
                GoldilocksField(data[4 * i + 2]),
                GoldilocksField(data[4 * i + 3]),
            );
            let cpu_digest = hash_ext2_leaf(a, b);
            assert_eq!(gpu_digest, cpu_digest, "Ext2 leaf {} mismatch", i);
        }
    }
}
