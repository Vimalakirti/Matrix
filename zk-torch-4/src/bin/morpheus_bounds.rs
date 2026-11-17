//! E2: certified bound / no-wrap pass for the transformer instances of
//! morpheus::models (CPU only; exits inside Dag::run before the forward
//! pass). Usage: MORPHEUS_BOUNDS=<out prefix> MODEL=gpt2 ... morpheus_bounds
use zk_torch_4::morpheus::models;
fn main() {
    assert!(std::env::var_os("MORPHEUS_BOUNDS").is_some(), "set MORPHEUS_BOUNDS=<out prefix>");
    std::env::set_var("MORPHEUS_LEAN_AUX", "1");
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gpt2".into());
    let _ = models::build(&model, 0);
    panic!("graph built without a forward pass: bounds hook not reached");
}
