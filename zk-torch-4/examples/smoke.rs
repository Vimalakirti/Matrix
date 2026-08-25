// End-to-end smoke: build a tiny DAG, run it, check outputs.
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use almost_goldilocks_cuda::field::AlmostGoldilocksField;

fn main() {
    let mut g = DagBuilder::new();
    let x = g.input(vec![4], DataType::Int);
    let y = g.input(vec![4], DataType::Int);
    let z = g.add(x, y)[0];
    let (dag, mut witnesses) = g.compile();
    println!("DAG: {} edges, {} nodes", dag.num_edges(), dag.nodes.len());

    let x_in = Witness::new(
        vec![4],
        (1..=4u64).map(AlmostGoldilocksField).collect(),
        DataType::Int, 0, Role::Input,
    );
    let y_in = Witness::new(
        vec![4],
        (10..=13u64).map(AlmostGoldilocksField).collect(),
        DataType::Int, 0, Role::Input,
    );
    dag.run(&mut witnesses, &[(x, x_in), (y, y_in)]);

    let z_out = witnesses[z][0].data.as_ref().unwrap();
    for i in 0..4 {
        println!("z[{}] = {}", i, z_out.index(i).reduce().0);
    }
}
