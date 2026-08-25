pub mod basicblock;
pub mod commit;
pub mod dag;
pub mod poly;
pub mod sumcheck;
pub mod transcript;
pub mod util;

use once_cell::sync::Lazy;
use std::env;
use std::fs::File;
use std::io::Read;

use crate::util::config::Config;

pub static CONFIG_FILE: Lazy<String> = Lazy::new(|| {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        // Default config for testing
        return "config.yaml".to_string();
    }
    args[1].clone()
});

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    if let Ok(mut file) = File::open(&*CONFIG_FILE) {
        let mut contents = String::new();
        file.read_to_string(&mut contents).expect("Could not read config");
        serde_yaml::from_str(&contents).expect("Could not parse config")
    } else {
        Config::default()
    }
});

pub static SF_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.scale_factor_log);
pub static TABLE_SIZE_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.table_size_log);
pub static TABLE_COMMIT_LOG: Lazy<usize> = Lazy::new(|| CONFIG.sf.table_commit_log);
pub static SF_FLOAT: Lazy<f32> = Lazy::new(|| (1 << *SF_LOG) as f32);
pub static SF_INT: Lazy<usize> = Lazy::new(|| 1 << *SF_LOG);

pub const SIGN_BIT: usize = 63;
pub const FIELD_SIZE: usize = 64;
pub const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

// Re-exports
pub use goldilocks_cuda::{self, GoldilocksField, GoldilocksExt2};
pub use poly::{CryptoField, DenseMLPoly, MLPoly};
pub use transcript::Transcript;
