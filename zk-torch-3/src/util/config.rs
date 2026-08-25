use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaleFactorConfig {
    pub scale_factor_log: usize,
    pub table_size_log: usize,
    pub table_commit_log: usize,
}

impl Default for ScaleFactorConfig {
    fn default() -> Self {
        Self {
            scale_factor_log: 10,
            table_size_log: 20,
            table_commit_log: 16,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub sf: ScaleFactorConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sf: ScaleFactorConfig::default(),
        }
    }
}
