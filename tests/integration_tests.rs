//! Integration tests for LongCodeZip

use longcodezip::{CodeLanguage, CompressionConfig, LongCodeZip, ProviderConfig};

#[tokio::test]
async fn test_compress_python_code() {
    let code = r#"
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def subtract(a, b):
    return a - b
"#;

    let provider = ProviderConfig::new(
        "test",
        "https://api.example.com",
        "test-key",
        "test-model",
    );

    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);

    let compressor = LongCodeZip::new(config).unwrap();
    let result = compressor.compress_code(code, "", "").await.unwrap();

    assert!(!result.compressed_code.is_empty());
    assert!(result.compression_ratio > 0.0);
    assert!(result.compression_ratio <= 1.0);
}

#[tokio::test]
async fn test_compress_with_query() {
    let code = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"#;

    let provider = ProviderConfig::new(
        "test",
        "https://api.example.com",
        "test-key",
        "test-model",
    );

    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);

    let compressor = LongCodeZip::new(config).unwrap();
    let result = compressor
        .compress_code(code, "how to calculate fibonacci?", "")
        .await
        .unwrap();

    assert!(!result.compressed_code.is_empty());
    // With keyword matching, fibonacci should be ranked higher
    // But without real API, we just check that compression worked
    assert!(result.selected_functions.len() > 0);
    assert!(result.compression_ratio > 0.0);
}

#[tokio::test]
async fn test_compress_rust_code() {
    let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
"#;

    let provider = ProviderConfig::new(
        "test",
        "https://api.example.com",
        "test-key",
        "test-model",
    );

    let config = CompressionConfig::default()
        .with_rate(0.6)
        .with_language(CodeLanguage::Rust)
        .with_provider(provider);

    let compressor = LongCodeZip::new(config).unwrap();
    let result = compressor.compress_code(code, "", "").await.unwrap();

    assert!(!result.compressed_code.is_empty());
    assert!(result.selected_functions.len() > 0);
}

#[test]
fn test_config_builder() {
    let provider = ProviderConfig::new(
        "deepseek",
        "https://api.deepseek.com",
        "test-key",
        "deepseek-chat",
    );

    let config = CompressionConfig::default()
        .with_rate(0.7)
        .with_target_token(1000)
        .with_language(CodeLanguage::TypeScript)
        .with_provider(provider)
        .with_rank_only(true);

    assert_eq!(config.rate, 0.7);
    assert_eq!(config.target_token, 1000);
    assert_eq!(config.language, CodeLanguage::TypeScript);
    assert_eq!(config.rank_only, true);
}

#[test]
fn test_language_comment_markers() {
    assert_eq!(CodeLanguage::Python.comment_marker(), "#");
    assert_eq!(CodeLanguage::Rust.comment_marker(), "//");
    assert_eq!(CodeLanguage::JavaScript.comment_marker(), "//");
    assert_eq!(CodeLanguage::Cpp.comment_marker(), "//");
}
