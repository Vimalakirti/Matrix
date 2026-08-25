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
            // Each range-check aux chunk commits at arity `input_n +
            // table_commit_log`, and the fold-tree multifold is dense over
            // `2^(arity-6)` — so this knob sets the fold-tree cliff. It is a
            // PURE PERFORMANCE knob: the verifier reconstructs the full table
            // value as `Σ_j middle_claim_j · 2^(j · table_commit_log)`, so any
            // block size is sound regardless of the value range (that is
            // `table_size_log`'s job; it only adds chunks, linearly). A stale
            // default of 16 here silently pushed CV range auxes to arity ~32
            // → 165 s ResNet-50 proves. 8 keeps every shipped config off the
            // cliff; smaller (6) is marginally faster at the cost of more
            // chunks. See bench sweep in project_zk3_vs_zk4_baselines.
            table_commit_log: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub sf: ScaleFactorConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_present() {
        let c = Config::default();
        assert_eq!(c.sf.scale_factor_log, 10);
        assert_eq!(c.sf.table_size_log, 20);
        assert_eq!(c.sf.table_commit_log, 8);
    }

    #[test]
    fn roundtrip_yaml() {
        let c = Config::default();
        let s = serde_yaml::to_string(&c).unwrap();
        let back: Config = serde_yaml::from_str(&s).unwrap();
        assert_eq!(back.sf.scale_factor_log, c.sf.scale_factor_log);
        assert_eq!(back.sf.table_size_log, c.sf.table_size_log);
        assert_eq!(back.sf.table_commit_log, c.sf.table_commit_log);
    }
}
