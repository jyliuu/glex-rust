// Pure Rust XGBoost JSON parsing (no PyO3 dependencies)

pub mod json_schema;
pub mod parser;

pub use parser::parse_json_tree;

