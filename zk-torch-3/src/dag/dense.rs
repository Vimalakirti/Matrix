use crate::dag::{DagBuilder, EdgeId, Witness};

pub fn dense_add_relu(w: Witness, b: Witness) -> impl FnOnce(&mut DagBuilder, &[EdgeId]) -> Vec<EdgeId> {
    move |g, x| {
        assert!(x.len() == 1, "This dense layer expects 1 input");
        let x = x[0];
        let w_e = g.param(w);
        let h = g.einsum("i,ij->j".to_string(), vec![x, w_e], true)[0];
        let b_e = g.param(b);
        let h = g.add(h, b_e)[0];
        vec![h]
    }
}
