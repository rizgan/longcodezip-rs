//! Tokenizer module for accurate token counting

use crate::error::{Error, Result};
use lazy_static::lazy_static;
use std::sync::Arc;
use tiktoken_rs::{cl100k_base, o200k_base, p50k_base, r50k_base, CoreBPE};

/// Supported tokenizer models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    Cl100kBase,
    /// GPT-4o models
    O200kBase,
    /// Code models (Codex)
    P50kBase,
    /// GPT-3 models (davinci, curie, etc.)
    R50kBase,
}

impl TokenizerModel {
    /// Get tokenizer model from model name
    pub fn from_model_name(model: &str) -> Self {
        let model_lower = model.to_lowercase();
        
        if model_lower.contains("gpt-4o") {
            Self::O200kBase
        } else if model_lower.contains("gpt-4") || model_lower.contains("gpt-3.5-turbo") {
            Self::Cl100kBase
        } else if model_lower.contains("code") || model_lower.contains("codex") {
            Self::P50kBase
        } else if model_lower.contains("gpt-3") {
            Self::R50kBase
        } else if model_lower.contains("deepseek") || model_lower.contains("claude") {
            // DeepSeek и Claude используют cl100k_base compatible
            Self::Cl100kBase
        } else {
            // Default to cl100k_base (most common)
            Self::Cl100kBase
        }
    }
    
    /// Get the tokenizer name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
            Self::P50kBase => "p50k_base",
            Self::R50kBase => "r50k_base",
        }
    }
}

lazy_static! {
    static ref CL100K_BASE: Arc<CoreBPE> = Arc::new(cl100k_base().unwrap());
    static ref O200K_BASE: Arc<CoreBPE> = Arc::new(o200k_base().unwrap());
    static ref P50K_BASE: Arc<CoreBPE> = Arc::new(p50k_base().unwrap());
    static ref R50K_BASE: Arc<CoreBPE> = Arc::new(r50k_base().unwrap());
}

/// Tokenizer for accurate token counting
pub struct Tokenizer {
    model: TokenizerModel,
    bpe: Arc<CoreBPE>,
}

impl Tokenizer {
    /// Create a new tokenizer with the specified model
    pub fn new(model: TokenizerModel) -> Self {
        let bpe = match model {
            TokenizerModel::Cl100kBase => CL100K_BASE.clone(),
            TokenizerModel::O200kBase => O200K_BASE.clone(),
            TokenizerModel::P50kBase => P50K_BASE.clone(),
            TokenizerModel::R50kBase => R50K_BASE.clone(),
        };
        
        Self { model, bpe }
    }
    
    /// Create a tokenizer from model name
    pub fn from_model_name(model_name: &str) -> Self {
        let model = TokenizerModel::from_model_name(model_name);
        Self::new(model)
    }
    
    /// Get the model type
    pub fn model(&self) -> TokenizerModel {
        self.model
    }
    
    /// Count tokens in text
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let tokens = self.bpe.encode_with_special_tokens(text);
        Ok(tokens.len())
    }
    
    /// Encode text to tokens
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        Ok(self.bpe.encode_with_special_tokens(text))
    }
    
    /// Encode text without special tokens
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<usize>> {
        Ok(self.bpe.encode_ordinary(text))
    }
    
    /// Decode tokens to text
    pub fn decode(&self, tokens: &[usize]) -> Result<String> {
        self.bpe
            .decode(tokens.to_vec())
            .map_err(|e| Error::ProcessingError(format!("Failed to decode tokens: {}", e)))
    }
    
    /// Truncate text to max tokens
    pub fn truncate(&self, text: &str, max_tokens: usize) -> Result<String> {
        let tokens = self.encode(text)?;
        
        if tokens.len() <= max_tokens {
            return Ok(text.to_string());
        }
        
        let truncated_tokens = &tokens[..max_tokens];
        self.decode(truncated_tokens)
    }
    
    /// Count tokens in multiple texts
    pub fn count_tokens_batch(&self, texts: &[&str]) -> Result<Vec<usize>> {
        texts.iter().map(|text| self.count_tokens(text)).collect()
    }
}

/// Approximate token counter (fallback for when tiktoken is not available)
pub struct ApproximateTokenizer {
    chars_per_token: f64,
}

impl ApproximateTokenizer {
    /// Create a new approximate tokenizer
    pub fn new() -> Self {
        Self {
            chars_per_token: 4.0, // Average for English text
        }
    }
    
    /// Count tokens (approximate)
    pub fn count_tokens(&self, text: &str) -> usize {
        let chars = text.chars().count();
        (chars as f64 / self.chars_per_token).ceil() as usize
    }
}

impl Default for ApproximateTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer_model_from_name() {
        assert_eq!(
            TokenizerModel::from_model_name("gpt-4"),
            TokenizerModel::Cl100kBase
        );
        assert_eq!(
            TokenizerModel::from_model_name("gpt-4o"),
            TokenizerModel::O200kBase
        );
        assert_eq!(
            TokenizerModel::from_model_name("gpt-3.5-turbo"),
            TokenizerModel::Cl100kBase
        );
        assert_eq!(
            TokenizerModel::from_model_name("deepseek-chat"),
            TokenizerModel::Cl100kBase
        );
    }
    
    #[test]
    fn test_tokenizer_count() {
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        
        let text = "Hello, world!";
        let count = tokenizer.count_tokens(text).unwrap();
        
        assert!(count > 0);
        assert!(count < 10); // Should be around 3-4 tokens
    }
    
    #[test]
    fn test_tokenizer_encode_decode() {
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        
        let text = "The quick brown fox jumps over the lazy dog";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        
        assert_eq!(text, decoded);
    }
    
    #[test]
    fn test_tokenizer_truncate() {
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        
        let text = "This is a long text that should be truncated to a smaller number of tokens";
        let truncated = tokenizer.truncate(text, 5).unwrap();
        
        let tokens = tokenizer.count_tokens(&truncated).unwrap();
        assert!(tokens <= 5);
    }
    
    #[test]
    fn test_approximate_tokenizer() {
        let tokenizer = ApproximateTokenizer::new();
        
        let text = "Hello world";
        let count = tokenizer.count_tokens(text);
        
        assert!(count > 0);
        assert_eq!(count, 3); // 11 chars / 4 = 2.75 -> 3
    }
    
    #[test]
    fn test_batch_count() {
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        
        let texts = vec!["Hello", "world", "test"];
        let counts = tokenizer.count_tokens_batch(&texts).unwrap();
        
        assert_eq!(counts.len(), 3);
        assert!(counts.iter().all(|&c| c > 0));
    }
}
