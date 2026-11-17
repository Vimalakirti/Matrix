//! Determine conv2d_strided's weight layout empirically, then verify a
//! block-diagonal 1/4 kernel really computes 2x2 average pooling.
use almost_goldilocks_cuda::field::AlmostGoldilocksField as F;
use zk_torch_4::dag::{DagBuilder, DataType, Role, Witness};
use zk_torch_4::SF_LOG;

fn main() {
    let c = 2usize; let hw = 4usize;                    // [2,4,4] -> [2,2,2]
    let (cp, hwp) = (c.next_power_of_two(), hw.next_power_of_two());
    let mut g = DagBuilder::new();
    let x = g.input(vec![c, hw, hw], DataType::Uint);

    // w[co][ci][kh][kw] = 1/4 when co == ci, else 0. Padded row-major, the
    // layout vgg/resnet use for conv weights.
    let (kp, cip) = (2usize.next_power_of_two(), c.next_power_of_two());
    let mut wd = zk_torch_4::zero_witness_vec(cp * cip * kp * kp);
    let quarter = F(1u64 << (*SF_LOG - 2));
    for co in 0..c { for kh in 0..2 { for kw in 0..2 {
        wd[((co * cip + co) * kp + kh) * kp + kw] = quarter;
    }}}
    let w = g.param(Witness::new(vec![c, c, 2, 2], wd, DataType::Uint, *SF_LOG, Role::Constant));
    let y = g.conv2d_strided(x, w, (2, 2), (2, 2))[0];
    let y = g.scale(y, 2 * *SF_LOG, *SF_LOG)[0];
    let (dag, wt) = g.compile();
    println!("out shape {:?}", dag_shape(&wt, y));

    // Input value v at (ch,i,j) = ch*100 + i*10 + j, scaled.
    let mut d = zk_torch_4::zero_witness_vec(cp * hwp * hwp);
    for ch in 0..c { for i in 0..hw { for j in 0..hw {
        d[(ch * hwp + i) * hwp + j] = F(((ch * 100 + i * 10 + j) as u64) << *SF_LOG);
    }}}
    let mut ws = wt.clone();
    let mut dag = dag;
    dag.run(&mut ws, &[(x, Witness::new(vec![c, hw, hw], d, DataType::Uint, *SF_LOG, Role::Input))]);
    let out = ws[y][0].data.as_ref().unwrap();
    let op = 2usize.next_power_of_two();
    for ch in 0..c { for i in 0..2 { for j in 0..2 {
        let got = out.index((ch * op + i) * op + j).0 as f64 / (1u64 << *SF_LOG) as f64;
        let want = (0..2).flat_map(|a| (0..2).map(move |b| (a, b)))
            .map(|(a, b)| (ch * 100 + (2 * i + a) * 10 + (2 * j + b)) as f64).sum::<f64>() / 4.0;
        println!("  ch{} ({},{})  got {:>8.3}  want {:>8.3}  {}", ch, i, j, got, want,
                 if (got - want).abs() < 0.01 { "OK" } else { "MISMATCH" });
    }}}
}

fn dag_shape(wt: &[Vec<Witness>], e: usize) -> Vec<usize> { wt[e][0].shape.clone() }
