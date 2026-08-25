fn main() {
    almost_goldilocks_cuda::init().expect("init");
    let n = almost_goldilocks_cuda::device_count();
    println!("devices: {}", n);
    assert!(n >= 2);
    use almost_goldilocks_cuda::memory::DeviceBuffer;
    // alloc + fill on dev 0
    almost_goldilocks_cuda::set_device(0).unwrap();
    let data: Vec<u64> = (0..1u64 << 16).collect();
    let src = DeviceBuffer::<u64>::from_slice(&data).unwrap();
    println!("src on dev0 ok");
    // alloc on dev 1, peer copy
    almost_goldilocks_cuda::set_device(1).unwrap();
    let mut dst = DeviceBuffer::<u64>::new(1 << 16).unwrap();
    println!("dst on dev1 ok");
    dst.copy_range_from_device_peer(0, 1, &src, 0, 0, 1 << 16).unwrap();
    println!("peer copy ok");
    let back = dst.to_vec().unwrap();
    assert_eq!(back, data);
    println!("verify ok");
    // also plain dtod cross-device
    let mut dst2 = DeviceBuffer::<u64>::new(1 << 16).unwrap();
    dst2.copy_range_from_device(0, &src, 0, 1 << 16).unwrap();
    let back2 = dst2.to_vec().unwrap();
    assert_eq!(back2, data);
    println!("plain dtod cross-device ok too");
}
