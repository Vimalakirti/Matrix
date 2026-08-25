//! Poseidon2 hash function on GPU.

use crate::error::{CudaError, Result};
use crate::ffi;
use crate::field::GoldilocksField;
use crate::memory::DeviceBuffer;
use std::os::raw::c_int;

/// Poseidon2 width (number of field elements in state).
pub const POSEIDON2_WIDTH: usize = 8;

/// Poseidon2 rate (number of absorbed elements per permutation).
pub const POSEIDON2_RATE: usize = 4;

/// Poseidon2 digest size.
pub const POSEIDON2_DIGEST_SIZE: usize = 4;

/// A Poseidon2 hash output (4 field elements).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Poseidon2Hash {
    pub elements: [GoldilocksField; POSEIDON2_DIGEST_SIZE],
}

impl Poseidon2Hash {
    /// Create a new hash from field elements.
    pub const fn new(elements: [GoldilocksField; POSEIDON2_DIGEST_SIZE]) -> Self {
        Self { elements }
    }

    /// Create a hash from raw u64 values.
    pub fn from_raw(raw: [u64; POSEIDON2_DIGEST_SIZE]) -> Self {
        Self {
            elements: [
                GoldilocksField(raw[0]),
                GoldilocksField(raw[1]),
                GoldilocksField(raw[2]),
                GoldilocksField(raw[3]),
            ],
        }
    }

    /// Convert to raw u64 values.
    pub fn to_raw(&self) -> [u64; POSEIDON2_DIGEST_SIZE] {
        [
            self.elements[0].0,
            self.elements[1].0,
            self.elements[2].0,
            self.elements[3].0,
        ]
    }
}

/// Low-level batch Poseidon2 operations on device buffers.
pub struct Poseidon2Batch;

impl Poseidon2Batch {
    /// Batch hash: apply Poseidon2 permutation to n states.
    /// Input/output: 8 u64 values per state.
    pub fn permutation(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n = input.len() / POSEIDON2_WIDTH;
        if output.len() != input.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::poseidon2_hash_batch_ffi(input.as_ptr(), output.as_mut_ptr(), n as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Batch compress: compress pairs of 4-element digests.
    /// Input: left (4*n u64), right (4*n u64), output (4*n u64).
    pub fn compress(
        left: &DeviceBuffer<u64>,
        right: &DeviceBuffer<u64>,
        output: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        let n = left.len() / POSEIDON2_DIGEST_SIZE;
        if right.len() != left.len() || output.len() != left.len() {
            return Err(CudaError::InvalidArgument(
                "Buffer lengths must match".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::poseidon2_compress_batch_ffi(
                left.as_ptr(),
                right.as_ptr(),
                output.as_mut_ptr(),
                n as c_int,
            )
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }

    /// Merkle layer: compute parent nodes from child pairs.
    /// Input: 2*n nodes (4 u64 each), output: n nodes.
    pub fn merkle_layer(input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>) -> Result<()> {
        let n_output = output.len() / POSEIDON2_DIGEST_SIZE;
        let n_input = input.len() / POSEIDON2_DIGEST_SIZE;

        if n_input != 2 * n_output {
            return Err(CudaError::InvalidArgument(
                "Input must have 2x the nodes of output".to_string(),
            ));
        }

        let ret = unsafe {
            ffi::poseidon2_merkle_layer_ffi(input.as_ptr(), output.as_mut_ptr(), n_output as c_int)
        };

        if ret != 0 {
            return Err(CudaError::KernelFailed);
        }
        Ok(())
    }
}

/// High-level Poseidon2 operations with automatic memory management.
pub struct Poseidon2Ops;

impl Poseidon2Ops {
    /// Hash a batch of 8-element states.
    pub fn hash_batch(
        inputs: &[[GoldilocksField; POSEIDON2_WIDTH]],
    ) -> Result<Vec<[GoldilocksField; POSEIDON2_WIDTH]>> {
        let n = inputs.len();
        let flat: Vec<u64> = inputs.iter().flat_map(|s| s.iter().map(|f| f.0)).collect();

        let d_input = DeviceBuffer::from_slice(&flat)?;
        let mut d_output = DeviceBuffer::<u64>::new(n * POSEIDON2_WIDTH)?;

        Poseidon2Batch::permutation(&d_input, &mut d_output)?;

        let result_flat = d_output.to_vec()?;
        let result: Vec<[GoldilocksField; POSEIDON2_WIDTH]> = result_flat
            .chunks_exact(POSEIDON2_WIDTH)
            .map(|chunk| {
                let mut arr = [GoldilocksField::zero(); POSEIDON2_WIDTH];
                for (i, &v) in chunk.iter().enumerate() {
                    arr[i] = GoldilocksField(v);
                }
                arr
            })
            .collect();

        Ok(result)
    }

    /// Compress pairs of hashes.
    pub fn compress_batch(
        left: &[Poseidon2Hash],
        right: &[Poseidon2Hash],
    ) -> Result<Vec<Poseidon2Hash>> {
        let n = left.len();
        if right.len() != n {
            return Err(CudaError::InvalidArgument(
                "Input lengths must match".to_string(),
            ));
        }

        let left_flat: Vec<u64> = left
            .iter()
            .flat_map(|h| h.elements.iter().map(|f| f.0))
            .collect();
        let right_flat: Vec<u64> = right
            .iter()
            .flat_map(|h| h.elements.iter().map(|f| f.0))
            .collect();

        let d_left = DeviceBuffer::from_slice(&left_flat)?;
        let d_right = DeviceBuffer::from_slice(&right_flat)?;
        let mut d_output = DeviceBuffer::<u64>::new(n * POSEIDON2_DIGEST_SIZE)?;

        Poseidon2Batch::compress(&d_left, &d_right, &mut d_output)?;

        let result_flat = d_output.to_vec()?;
        let result: Vec<Poseidon2Hash> = result_flat
            .chunks_exact(POSEIDON2_DIGEST_SIZE)
            .map(|chunk| {
                Poseidon2Hash::from_raw([chunk[0], chunk[1], chunk[2], chunk[3]])
            })
            .collect();

        Ok(result)
    }

    /// Build a Merkle tree from leaves.
    /// Returns all tree nodes layer by layer (leaves first, root last).
    pub fn build_merkle_tree(leaves: &[Poseidon2Hash]) -> Result<Vec<Vec<Poseidon2Hash>>> {
        let n = leaves.len();
        if n == 0 || (n & (n - 1)) != 0 {
            return Err(CudaError::InvalidArgument(
                "Number of leaves must be a power of 2".to_string(),
            ));
        }

        let mut layers: Vec<Vec<Poseidon2Hash>> = Vec::new();
        layers.push(leaves.to_vec());

        let mut current: Vec<u64> = leaves
            .iter()
            .flat_map(|h| h.elements.iter().map(|f| f.0))
            .collect();

        let mut num_nodes = n;

        while num_nodes > 1 {
            let d_input = DeviceBuffer::from_slice(&current)?;
            let mut d_output = DeviceBuffer::<u64>::new((num_nodes / 2) * POSEIDON2_DIGEST_SIZE)?;

            Poseidon2Batch::merkle_layer(&d_input, &mut d_output)?;

            current = d_output.to_vec()?;
            num_nodes /= 2;

            let layer: Vec<Poseidon2Hash> = current
                .chunks_exact(POSEIDON2_DIGEST_SIZE)
                .map(|chunk| Poseidon2Hash::from_raw([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            layers.push(layer);
        }

        Ok(layers)
    }

    /// Get the root of a Merkle tree.
    pub fn merkle_root(leaves: &[Poseidon2Hash]) -> Result<Poseidon2Hash> {
        let tree = Self::build_merkle_tree(leaves)?;
        Ok(tree.last().unwrap()[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init;

    #[test]
    fn test_poseidon2_hash() {
        init().unwrap();

        let input = [[GoldilocksField::zero(); POSEIDON2_WIDTH]; 1];
        let output = Poseidon2Ops::hash_batch(&input).unwrap();

        // Just verify it runs without error
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_merkle_tree() {
        init().unwrap();

        // Create 8 leaves
        let leaves: Vec<Poseidon2Hash> = (0..8)
            .map(|i| {
                Poseidon2Hash::from_raw([i as u64, 0, 0, 0])
            })
            .collect();

        let tree = Poseidon2Ops::build_merkle_tree(&leaves).unwrap();

        // Should have 4 layers: 8, 4, 2, 1 nodes
        assert_eq!(tree.len(), 4);
        assert_eq!(tree[0].len(), 8);
        assert_eq!(tree[1].len(), 4);
        assert_eq!(tree[2].len(), 2);
        assert_eq!(tree[3].len(), 1);
    }
}
