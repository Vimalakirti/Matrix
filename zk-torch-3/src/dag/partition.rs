//! DAG partitioning for parallel proving.
//!
//! Splits a DAG at boundary edges into N partitions that can be proved independently.
//! Each partition is a self-contained sub-proof connected by polynomial commitments
//! at the boundaries.

use std::collections::HashSet;

use super::{Dag, EdgeId, NodeId};

/// Describes a partition of the DAG for parallel proving.
#[derive(Debug, Clone)]
pub struct PartitionDesc {
    /// Partition index (0-based).
    pub partition_id: usize,
    /// Nodes belonging to this partition, in topological order.
    pub node_ids: Vec<NodeId>,
    /// All input edges for this partition (from previous partitions or DAG inputs).
    pub input_edges: Vec<EdgeId>,
    /// All output edges for this partition (to next partitions or DAG outputs).
    pub output_edges: Vec<EdgeId>,
    /// Subset of input_edges that are boundary edges (from a previous partition).
    pub boundary_input_edges: Vec<EdgeId>,
    /// Subset of output_edges that are boundary edges (to a next partition).
    pub boundary_output_edges: Vec<EdgeId>,
}

/// Partition a DAG at the given boundary edges.
///
/// `boundary_edges` are the edges where the DAG is "cut". Each boundary edge
/// is the output of one partition and the input of the next. For transformer
/// models, these are the hidden state edges between layers.
///
/// Returns N partitions where N = boundary_edges.len() + 1.
/// Partition 0 contains all nodes from DAG inputs up to (producing) boundary_edges[0].
/// Partition k contains nodes between boundary_edges[k-1] and boundary_edges[k].
/// Partition N-1 contains nodes from boundary_edges[N-2] to DAG outputs.
pub fn partition_dag(dag: &Dag, boundary_edges: &[EdgeId]) -> Vec<PartitionDesc> {
    let num_partitions = boundary_edges.len() + 1;

    if boundary_edges.is_empty() {
        // No split: single partition containing everything
        return vec![PartitionDesc {
            partition_id: 0,
            node_ids: dag.topo.clone(),
            input_edges: dag.input_ports.clone(),
            output_edges: dag.output_ports.clone(),
            boundary_input_edges: vec![],
            boundary_output_edges: vec![],
        }];
    }

    let boundary_set: HashSet<EdgeId> = boundary_edges.iter().copied().collect();

    // Assign each node to a partition.
    // Strategy: walk the topological order. A node belongs to partition k if
    // all of its produced edges are "before" boundary_edges[k].
    //
    // More precisely: a boundary edge B_k separates partition k from partition k+1.
    // The producer of B_k belongs to partition k.
    // A consumer of B_k belongs to partition k+1 (or later).
    //
    // We assign partitions by tracking which "segment" we're in:
    // - Start in partition 0
    // - When we encounter a node that CONSUMES a boundary edge B_k, that node
    //   is in partition k+1 (or later, based on which boundary edge it consumes)

    // First, map each boundary edge to its partition boundary index.
    // boundary_edges[k] separates partition k from partition k+1.
    let mut boundary_edge_idx: std::collections::HashMap<EdgeId, usize> = std::collections::HashMap::new();
    for (k, &e) in boundary_edges.iter().enumerate() {
        boundary_edge_idx.insert(e, k);
    }

    // Assign each node to the maximum partition required by its inputs.
    // A node consuming boundary_edge[k] must be in partition >= k+1.
    // A node producing boundary_edge[k] must be in partition k.
    let mut node_partition = vec![0usize; dag.nodes.len()];

    // First pass: assign based on consumed boundary edges
    for node in &dag.nodes {
        let mut min_partition = 0usize;
        for &e in &node.inputs {
            if let Some(&k) = boundary_edge_idx.get(&e) {
                // Consuming boundary edge k means we're in partition k+1 or later
                min_partition = min_partition.max(k + 1);
            }
        }
        node_partition[node.id] = min_partition;
    }

    // Second pass: propagate forward in topological order.
    // If a node's input comes from a node in partition k, this node must be >= k.
    // (Unless the connecting edge is a boundary edge, which we already handled.)
    for &nid in &dag.topo {
        let node = &dag.nodes[nid];
        let my_partition = node_partition[nid];
        for &out_edge in &node.outputs {
            if boundary_set.contains(&out_edge) {
                // This edge is a boundary; consumers are already assigned to partition k+1
                continue;
            }
            for &consumer_nid in &dag.consumers[out_edge] {
                if node_partition[consumer_nid] < my_partition {
                    node_partition[consumer_nid] = my_partition;
                }
            }
        }
    }

    // Build partition descriptors
    let mut partitions: Vec<PartitionDesc> = (0..num_partitions)
        .map(|k| PartitionDesc {
            partition_id: k,
            node_ids: Vec::new(),
            input_edges: Vec::new(),
            output_edges: Vec::new(),
            boundary_input_edges: Vec::new(),
            boundary_output_edges: Vec::new(),
        })
        .collect();

    // Collect nodes per partition (in topological order)
    for &nid in &dag.topo {
        let k = node_partition[nid];
        partitions[k].node_ids.push(nid);
    }

    // Compute input/output edges for each partition
    let node_partition_set: Vec<HashSet<NodeId>> = (0..num_partitions)
        .map(|k| partitions[k].node_ids.iter().copied().collect())
        .collect();

    for k in 0..num_partitions {
        let mut input_set = HashSet::new();
        let mut output_set = HashSet::new();

        for &nid in &partitions[k].node_ids {
            let node = &dag.nodes[nid];

            // Input edges: edges consumed by this partition but not produced by it
            for &e in &node.inputs {
                match dag.producers[e] {
                    Some(producer_nid) if node_partition_set[k].contains(&producer_nid) => {
                        // Produced within this partition — not an input
                    }
                    _ => {
                        // Produced outside or is a DAG input
                        input_set.insert(e);
                    }
                }
            }

            // Output edges: edges produced by this partition and consumed outside it
            for &e in &node.outputs {
                let consumed_outside = dag.consumers[e].iter().any(|&c| !node_partition_set[k].contains(&c));
                let is_dag_output = dag.consumers[e].is_empty();
                if consumed_outside || is_dag_output {
                    output_set.insert(e);
                }
            }
        }

        partitions[k].input_edges = input_set.iter().copied().collect();
        partitions[k].input_edges.sort();
        partitions[k].output_edges = output_set.iter().copied().collect();
        partitions[k].output_edges.sort();

        // Classify boundary edges
        partitions[k].boundary_input_edges = partitions[k]
            .input_edges
            .iter()
            .filter(|e| boundary_set.contains(e))
            .copied()
            .collect();
        partitions[k].boundary_output_edges = partitions[k]
            .output_edges
            .iter()
            .filter(|e| boundary_set.contains(e))
            .copied()
            .collect();
    }

    partitions
}

/// Map each edge to the partition that "owns" it (producer's partition).
/// Edges with no producer (DAG inputs/weights) are assigned to the first consumer's partition.
/// Returns a Vec of length `dag.num_edges()` with `None` for unassigned edges.
pub fn edge_partition_map(dag: &Dag, partitions: &[PartitionDesc]) -> Vec<Option<usize>> {
    let mut map = vec![None; dag.num_edges()];
    // Build node→partition lookup
    let mut node_partition = vec![None; dag.nodes.len()];
    for p in partitions {
        for &nid in &p.node_ids {
            node_partition[nid] = Some(p.partition_id);
        }
    }
    // Assign each edge to its producer's partition
    for (e, producer) in dag.producers.iter().enumerate() {
        if let Some(pid) = producer {
            map[e] = node_partition[*pid];
        }
    }
    // Edges with no producer (inputs/weights): assign to first consumer's partition
    for (e, consumers) in dag.consumers.iter().enumerate() {
        if map[e].is_none() && !consumers.is_empty() {
            map[e] = node_partition[consumers[0]];
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::builder::DagBuilder;

    #[test]
    fn test_partition_no_split() {
        // Build a simple DAG: input -> Add -> output
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], crate::dag::DataType::Uint);
        let y = g.input(vec![4], crate::dag::DataType::Uint);
        let _z = g.add(x, y);
        let (dag, _witnesses) = g.compile();

        let partitions = partition_dag(&dag, &[]);
        assert_eq!(partitions.len(), 1);
        assert_eq!(partitions[0].node_ids.len(), dag.nodes.len());
        assert!(partitions[0].boundary_input_edges.is_empty());
        assert!(partitions[0].boundary_output_edges.is_empty());
    }

    #[test]
    fn test_partition_two_way_split() {
        // Build: input -> Add(+bias1) -> [boundary] -> Add(+bias2) -> output
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], crate::dag::DataType::Uint);
        let b1 = g.input(vec![4], crate::dag::DataType::Uint);
        let mid = g.add(x, b1);
        let boundary_edge = mid[0]; // This is the edge we'll split on

        let b2 = g.input(vec![4], crate::dag::DataType::Uint);
        let _out = g.add(boundary_edge, b2);

        let (dag, _witnesses) = g.compile();

        let partitions = partition_dag(&dag, &[boundary_edge]);
        assert_eq!(partitions.len(), 2);

        // Partition 0: contains the first Add node
        assert_eq!(partitions[0].node_ids.len(), 1);
        assert!(partitions[0].boundary_output_edges.contains(&boundary_edge));
        assert!(partitions[0].boundary_input_edges.is_empty());

        // Partition 1: contains the second Add node
        assert_eq!(partitions[1].node_ids.len(), 1);
        assert!(partitions[1].boundary_input_edges.contains(&boundary_edge));
        assert!(partitions[1].boundary_output_edges.is_empty());
    }

    #[test]
    fn test_partition_three_way_split() {
        // Build: input -> Add -> [B1] -> Add -> [B2] -> Add -> output
        let mut g = DagBuilder::new();
        let x = g.input(vec![4], crate::dag::DataType::Uint);
        let b1 = g.input(vec![4], crate::dag::DataType::Uint);
        let mid1 = g.add(x, b1);
        let boundary1 = mid1[0];

        let b2 = g.input(vec![4], crate::dag::DataType::Uint);
        let mid2 = g.add(boundary1, b2);
        let boundary2 = mid2[0];

        let b3 = g.input(vec![4], crate::dag::DataType::Uint);
        let _out = g.add(boundary2, b3);

        let (dag, _witnesses) = g.compile();

        let partitions = partition_dag(&dag, &[boundary1, boundary2]);
        assert_eq!(partitions.len(), 3);

        // Check node counts
        assert_eq!(partitions[0].node_ids.len(), 1);
        assert_eq!(partitions[1].node_ids.len(), 1);
        assert_eq!(partitions[2].node_ids.len(), 1);

        // Check boundary edges
        assert!(partitions[0].boundary_output_edges.contains(&boundary1));
        assert!(partitions[1].boundary_input_edges.contains(&boundary1));
        assert!(partitions[1].boundary_output_edges.contains(&boundary2));
        assert!(partitions[2].boundary_input_edges.contains(&boundary2));
    }
}
