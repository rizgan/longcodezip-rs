//! Error types for LongCodeZip

use thiserror::Error;

/// Main error type for LongCodeZip operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),
    
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Code processing error: {0}")]
    ProcessingError(String),
    
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Result type alias for LongCodeZip operations
pub type Result<T> = std::result::Result<T, Error>;
