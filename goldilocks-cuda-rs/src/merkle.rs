//! GPU-resident Merkle tree built entirely on device.
//!
//! The tree is stored as a single contiguous `DeviceBuffer` with layers packed sequentially:
//! `[layer0: N digests] [layer1: N/2 digests] ... [root: 1 digest]`
//!
//! Each digest is 4 × u64 (Poseidon2Hash). Total size = (2N - 1) × 4 u64.
//!
//! Construction does zero host transfers — only the root (32 bytes) is copied to host on demand.

use std::os::raw::c_int;

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::memory::DeviceBuffer;
use crate::poseidon2::{Poseidon2Hash, POSEIDON2_DIGEST_SIZE};

/// A Merkle tree stored entirely on the GPU.
pub struct DeviceMerkleTree {
    /// All layers packed into one contiguous buffer.
    d_tree: DeviceBuffer<u64>,
    num_leaves: usize,
    /// Element offsets for each layer (in u64 units).
    /// layer_offsets[0] = 0 (leaves), layer_offsets[1] = num_leaves * 4, etc.
    layer_offsets: Vec<usize>,
}

impl DeviceMerkleTree {
    /// Build a Merkle tree from base-field codeword data on GPU.
    ///
    /// The codeword is hashed in pairs: every 2 adjacent elements become one leaf digest.
    /// `num_leaves = codeword_len / 2`.
    ///
    /// **Transfers**: 0 during construction. Tree lives entirely on GPU.
    pub fn build_from_gl_codeword(
        d_codeword: &DeviceBuffer<u64>,
        codeword_len: usize,
    ) -> Result<Self> {
        assert!(codeword_len >= 2 && codeword_len.is_power_of_two());
        let num_leaves = codeword_len / 2;
        let tree_nodes = 2 * num_leaves - 1;
        let mut d_tree = DeviceBuffer::<u64>::new(tree_nodes * POSEIDON2_DIGEST_SIZE)?;

        let ret = unsafe {
            #[cfg(feature = "monolith")]
            { ffi::monolith_merkle_tree_gl_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
            #[cfg(not(feature = "monolith"))]
            { ffi::poseidon2_merkle_tree_gl_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let layer_offsets = compute_layer_offsets(num_leaves);
        Ok(Self {
            d_tree,
            num_leaves,
            layer_offsets,
        })
    }

    /// Build a Merkle tree from ext2 codeword data on GPU.
    ///
    /// Every 2 adjacent ext2 elements (= 4 base field elements) become one leaf digest.
    /// `num_leaves = codeword_len / 2` (in ext2 elements).
    pub fn build_from_ext2_codeword(
        d_codeword: &DeviceBuffer<u64>,
        codeword_len_ext2: usize,
    ) -> Result<Self> {
        assert!(codeword_len_ext2 >= 2 && codeword_len_ext2.is_power_of_two());
        let num_leaves = codeword_len_ext2 / 2;
        let tree_nodes = 2 * num_leaves - 1;
        let mut d_tree = DeviceBuffer::<u64>::new(tree_nodes * POSEIDON2_DIGEST_SIZE)?;

        let ret = unsafe {
            #[cfg(feature = "monolith")]
            { ffi::monolith_merkle_tree_ext2_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
            #[cfg(not(feature = "monolith"))]
            { ffi::poseidon2_merkle_tree_ext2_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let layer_offsets = compute_layer_offsets(num_leaves);
        Ok(Self {
            d_tree,
            num_leaves,
            layer_offsets,
        })
    }

    /// Build a Merkle tree from ext2 codeword data into a pre-allocated buffer.
    ///
    /// Same as `build_from_ext2_codeword` but takes ownership of `d_tree` instead of
    /// allocating a new buffer. The buffer must have at least `(2 * num_leaves - 1) * POSEIDON2_DIGEST_SIZE`
    /// u64 elements where `num_leaves = codeword_len_ext2 / 2`.
    ///
    /// This avoids per-round `cudaMalloc` overhead when building multiple trees.
    pub fn build_from_ext2_codeword_into(
        d_codeword: &DeviceBuffer<u64>,
        codeword_len_ext2: usize,
        d_tree: DeviceBuffer<u64>,
    ) -> Result<Self> {
        assert!(codeword_len_ext2 >= 2 && codeword_len_ext2.is_power_of_two());
        let num_leaves = codeword_len_ext2 / 2;
        let tree_nodes = 2 * num_leaves - 1;
        assert!(d_tree.len() >= tree_nodes * POSEIDON2_DIGEST_SIZE,
            "pre-allocated tree buffer too small: {} < {}",
            d_tree.len(), tree_nodes * POSEIDON2_DIGEST_SIZE);
        let mut d_tree = d_tree;

        let ret = unsafe {
            #[cfg(feature = "monolith")]
            { ffi::monolith_merkle_tree_ext2_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
            #[cfg(not(feature = "monolith"))]
            { ffi::poseidon2_merkle_tree_ext2_ffi(d_codeword.as_ptr(), d_tree.as_mut_ptr(), num_leaves as c_int) }
        };
        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }

        let layer_offsets = compute_layer_offsets(num_leaves);
        Ok(Self {
            d_tree,
            num_leaves,
            layer_offsets,
        })
    }

    /// Reclaim the underlying device buffer from this tree.
    /// Useful for reusing the allocation for a subsequent tree build.
    pub fn into_buffer(self) -> DeviceBuffer<u64> {
        self.d_tree
    }

    /// Get the root hash. Copies exactly 32 bytes (4 u64) from device.
    pub fn root(&self) -> Result<Poseidon2Hash> {
        let root_offset = *self.layer_offsets.last().unwrap();
        let raw = self.d_tree.read_slice(root_offset, POSEIDON2_DIGEST_SIZE)?;
        Ok(Poseidon2Hash::from_raw([raw[0], raw[1], raw[2], raw[3]]))
    }

    /// Extract authentication path for a single leaf index.
    /// Copies log₂(N) × 32 bytes from device (one sibling per layer).
    pub fn auth_path(&self, leaf_index: usize) -> Result<Vec<Poseidon2Hash>> {
        let num_layers = self.layer_offsets.len() - 1; // exclude root layer
        let mut path = Vec::with_capacity(num_layers);
        let mut idx = leaf_index;

        for layer in 0..num_layers {
            let sibling_idx = idx ^ 1;
            let offset = self.layer_offsets[layer] + sibling_idx * POSEIDON2_DIGEST_SIZE;
            let raw = self.d_tree.read_slice(offset, POSEIDON2_DIGEST_SIZE)?;
            path.push(Poseidon2Hash::from_raw([raw[0], raw[1], raw[2], raw[3]]));
            idx /= 2;
        }

        Ok(path)
    }

    /// Bulk-download entire tree to host memory.
    pub fn to_host(&self) -> Result<HostMerkleTree> {
        let data = self.d_tree.to_vec()?;
        Ok(HostMerkleTree {
            data,
            layer_offsets: self.layer_offsets.clone(),
            num_leaves: self.num_leaves,
        })
    }

    /// Extract auth paths for multiple queries efficiently.
    /// For large trees: downloads small levels in bulk, uses selective reads for large levels.
    /// For small trees: downloads entire tree in one shot.
    pub fn batch_auth_paths(&self, leaf_indices: &[usize]) -> Result<Vec<Vec<Poseidon2Hash>>> {
        let num_layers = self.layer_offsets.len() - 1;
        if num_layers == 0 {
            return Ok(leaf_indices.iter().map(|_| Vec::new()).collect());
        }

        // Download each level: bulk if small (≤ 16K nodes = 512KB), selective if large
        const LEVEL_BULK_THRESHOLD: usize = 16384; // nodes
        let mut level_data: Vec<Option<Vec<u64>>> = Vec::with_capacity(num_layers);
        let mut level_size = self.num_leaves;
        for layer in 0..num_layers {
            if level_size <= LEVEL_BULK_THRESHOLD {
                let offset = self.layer_offsets[layer];
                let len = level_size * POSEIDON2_DIGEST_SIZE;
                level_data.push(Some(self.d_tree.read_slice(offset, len)?));
            } else {
                level_data.push(None); // will use per-query reads
            }
            level_size /= 2;
        }

        // Extract paths
        let mut all_paths = Vec::with_capacity(leaf_indices.len());
        for &leaf_idx in leaf_indices {
            let mut path = Vec::with_capacity(num_layers);
            let mut idx = leaf_idx;
            for layer in 0..num_layers {
                let sibling_idx = idx ^ 1;
                let raw = if let Some(ref data) = level_data[layer] {
                    // Bulk-downloaded level: direct index
                    let off = sibling_idx * POSEIDON2_DIGEST_SIZE;
                    &data[off..off + POSEIDON2_DIGEST_SIZE]
                } else {
                    // Large level: selective GPU read (returns owned vec, handled below)
                    &[] // placeholder
                };
                if raw.is_empty() {
                    // Selective read from GPU
                    let offset = self.layer_offsets[layer] + sibling_idx * POSEIDON2_DIGEST_SIZE;
                    let r = self.d_tree.read_slice(offset, POSEIDON2_DIGEST_SIZE)?;
                    path.push(Poseidon2Hash::from_raw([r[0], r[1], r[2], r[3]]));
                } else {
                    path.push(Poseidon2Hash::from_raw([raw[0], raw[1], raw[2], raw[3]]));
                }
                idx /= 2;
            }
            all_paths.push(path);
        }

        Ok(all_paths)
    }

    /// Number of leaves in this tree.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Read a leaf digest by index. Copies 32 bytes from device.
    pub fn leaf_digest(&self, leaf_index: usize) -> Result<Poseidon2Hash> {
        let offset = self.layer_offsets[0] + leaf_index * POSEIDON2_DIGEST_SIZE;
        let raw = self.d_tree.read_slice(offset, POSEIDON2_DIGEST_SIZE)?;
        Ok(Poseidon2Hash::from_raw([raw[0], raw[1], raw[2], raw[3]]))
    }
}

/// A Merkle tree downloaded to host memory.
/// Auth path extraction is pure CPU indexing — no GPU transfers.
pub struct HostMerkleTree {
    pub data: Vec<u64>,
    pub layer_offsets: Vec<usize>,
    pub num_leaves: usize,
}

impl HostMerkleTree {
    /// Extract authentication path for a leaf. Pure CPU — no GPU transfers.
    pub fn auth_path(&self, leaf_index: usize) -> Vec<Poseidon2Hash> {
        let num_layers = self.layer_offsets.len() - 1;
        let mut path = Vec::with_capacity(num_layers);
        let mut idx = leaf_index;

        for layer in 0..num_layers {
            let sibling_idx = idx ^ 1;
            let offset = self.layer_offsets[layer] + sibling_idx * POSEIDON2_DIGEST_SIZE;
            let raw = &self.data[offset..offset + POSEIDON2_DIGEST_SIZE];
            path.push(Poseidon2Hash::from_raw([raw[0], raw[1], raw[2], raw[3]]));
            idx /= 2;
        }

        path
    }
}

/// Compute layer offsets (in u64 units) for a tree with `num_leaves` leaves.
fn compute_layer_offsets(num_leaves: usize) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut offset = 0usize;
    let mut layer_size = num_leaves;
    while layer_size >= 1 {
        offsets.push(offset);
        offset += layer_size * POSEIDON2_DIGEST_SIZE;
        if layer_size == 1 {
            break;
        }
        layer_size /= 2;
    }
    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::synchronize;

    #[test]
    fn test_device_merkle_tree_gl() {
        crate::init().unwrap();

        let n = 16usize;
        let data: Vec<u64> = (0..n as u64).collect();
        let d_codeword = DeviceBuffer::from_slice(&data).unwrap();

        let tree = DeviceMerkleTree::build_from_gl_codeword(&d_codeword, n).unwrap();
        synchronize().unwrap();

        assert_eq!(tree.num_leaves(), n / 2);

        let root = tree.root().unwrap();
        // Root should be non-trivial
        assert!(
            root.elements.iter().any(|e| e.0 != 0),
            "Root should be non-zero"
        );

        // Auth path should have log2(num_leaves) entries
        let path = tree.auth_path(0).unwrap();
        assert_eq!(path.len(), (n / 2).trailing_zeros() as usize);
    }

    #[test]
    fn test_device_merkle_tree_consistency() {
        crate::init().unwrap();

        let n = 8usize;
        let data: Vec<u64> = (100..100 + n as u64).collect();
        let d_cw = DeviceBuffer::from_slice(&data).unwrap();

        let tree1 = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        let tree2 = DeviceMerkleTree::build_from_gl_codeword(&d_cw, n).unwrap();
        synchronize().unwrap();

        // Same input should produce same root
        assert_eq!(tree1.root().unwrap(), tree2.root().unwrap());
    }
}
