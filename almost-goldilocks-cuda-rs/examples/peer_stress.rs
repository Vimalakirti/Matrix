use almost_goldilocks_cuda::memory::DeviceBuffer;
use std::sync::{Arc, Barrier};

fn main() {
    almost_goldilocks_cuda::init().expect("init");
    let n_dev = almost_goldilocks_cuda::device_count().min(4);
    println!("devices: {}", n_dev);
    let n_ring = 1usize << 16; // arity-22 plane
    // Phase 1: each device thread allocates + fills a 13-chunk buffer.
    let barrier = Arc::new(Barrier::new(n_dev as usize));
    let handles: Vec<_> = (0..n_dev).map(|d| {
        let b = barrier.clone();
        std::thread::spawn(move || {
            almost_goldilocks_cuda::set_device(d).unwrap();
            let data: Vec<u64> = (0..13 * n_ring as u64).map(|i| i.wrapping_mul(0x9E37 + d as u64)).collect();
            let buf = DeviceBuffer::<u64>::from_slice(&data).unwrap();
            b.wait();
            (d, buf, data)
        })
    }).collect();
    let sources: Vec<(i32, DeviceBuffer<u64>, Vec<u64>)> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();
    let sources = Arc::new(sources);
    println!("sources ready");

    // Phase 2: each device thread concurrently peer-copies slices from ALL
    // sources into a local concat + pageable uploads, repeatedly.
    let handles: Vec<_> = (0..n_dev).map(|d| {
        let srcs = sources.clone();
        std::thread::spawn(move || {
            almost_goldilocks_cuda::set_device(d).unwrap();
            let host: Vec<u64> = vec![0xABCD; n_ring];
            for round in 0..50 {
                let mut local = DeviceBuffer::<u64>::new(63 * n_ring).unwrap();
                for i in 0..63usize {
                    let (sd, sbuf, _) = &srcs[(i + round) % srcs.len()];
                    let idx = i % 13;
                    if *sd == d {
                        local.copy_range_from_device(i * n_ring, sbuf, idx * n_ring, n_ring).unwrap();
                    } else {
                        local.copy_range_from_device_peer(i * n_ring, d, sbuf, idx * n_ring, *sd, n_ring).unwrap();
                    }
                    if i % 7 == 0 { local.write_slice_at(i * n_ring, &host).unwrap(); }
                }
                // verify one slice
                let got = local.read_slice(0, n_ring).unwrap();
                let (sd0, _, sdata) = &srcs[round % srcs.len()];
                let _ = (got, sd0, sdata);
            }
            println!("dev {} done", d);
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    println!("ALL OK");
}
