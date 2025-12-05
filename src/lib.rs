//! # LongCodeZip
//! 
//! A Rust implementation of LongCodeZip - compress long code context for LLMs.
//! 
//! This library provides a two-stage compression method:
//! 1. **Coarse-grained**: Function-level ranking and selection
//! 2. **Fine-grained**: Block-level optimization using importance scoring
//! 
//! ## Example
//! 
//! ```rust,no_run
//! use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig};
//! 
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = CompressionConfig::default()
//!         .with_rate(0.5)
//!         .with_language(CodeLanguage::Python);
//!     
//!     let compressor = LongCodeZip::new(config)?;
//!     
//!     let result = compressor.compress_code(
//!         "def foo(): pass",
//!         "What does foo do?",
//!         "Analyze this code:"
//!     ).await?;
//!     
//!     println!("Compression: {:.2}%", result.compression_ratio * 100.0);
//!     Ok(())
//! }
//! ```

pub mod code_splitter;
pub mod compressor;
pub mod entropy;
pub mod error;
pub mod optimizer;
pub mod provider;
pub mod text_chunker;
pub mod tokenizer;
pub mod types;

pub use error::{Error, Result};
pub use types::{
    CodeLanguage, CompressionConfig, CompressionResult, 
    FunctionCompression, ProviderConfig
};
pub use compressor::LongCodeZip;
pub use tokenizer::{Tokenizer, TokenizerModel, ApproximateTokenizer};
pub use entropy::{EntropyChunker, EntropyChunk, ThresholdMethod};
pub use optimizer::{KnapsackOptimizer, Block, SelectionResult};
pub use text_chunker::{TextChunker, TextChunk, TextChunkingStrategy, ChunkType};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
