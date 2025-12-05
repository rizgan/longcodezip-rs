//! Core types for LongCodeZip

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported programming languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CodeLanguage {
    Python,
    Rust,
    TypeScript,
    JavaScript,
    Cpp,
    Java,
    Go,
}

impl CodeLanguage {
    pub fn comment_marker(&self) -> &'static str {
        match self {
            CodeLanguage::Python => "#",
            CodeLanguage::Rust => "//",
            CodeLanguage::TypeScript => "//",
            CodeLanguage::JavaScript => "//",
            CodeLanguage::Cpp => "//",
            CodeLanguage::Java => "//",
            CodeLanguage::Go => "//",
        }
    }
    
    pub fn as_str(&self) -> &'static str {
        match self {
            CodeLanguage::Python => "python",
            CodeLanguage::Rust => "rust",
            CodeLanguage::TypeScript => "typescript",
            CodeLanguage::JavaScript => "javascript",
            CodeLanguage::Cpp => "cpp",
            CodeLanguage::Java => "java",
            CodeLanguage::Go => "go",
        }
    }
}

/// API provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider name (e.g., "deepseek", "openai")
    pub provider: String,
    /// API endpoint URL
    pub api_url: String,
    /// API key
    pub api_key: String,
    /// Model name
    pub model: String,
    /// Temperature (default: 0.0)
    #[serde(default)]
    pub temperature: f32,
    /// Max tokens for completion
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    2048
}

impl ProviderConfig {
    pub fn new(provider: &str, api_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            provider: provider.to_string(),
            api_url: api_url.to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            temperature: 0.0,
            max_tokens: 2048,
        }
    }
    
    /// Create config for OpenAI
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self::new(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            api_key,
            model,
        )
    }
    
    /// Create config for DeepSeek
    pub fn deepseek(api_key: &str) -> Self {
        Self::new(
            "deepseek",
            "https://api.deepseek.com/chat/completions",
            api_key,
            "deepseek-chat",
        )
    }
    
    /// Create config for Anthropic Claude
    pub fn claude(api_key: &str, model: &str) -> Self {
        Self::new(
            "anthropic",
            "https://api.anthropic.com/v1/messages",
            api_key,
            model,
        )
    }
    
    /// Create config for Azure OpenAI
    pub fn azure_openai(api_key: &str, resource_name: &str, deployment_name: &str, api_version: &str) -> Self {
        Self::new(
            "azure-openai",
            &format!("https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}", 
                    resource_name, deployment_name, api_version),
            api_key,
            deployment_name,
        )
    }
    
    /// Create config for Google Gemini
    pub fn gemini(api_key: &str, model: &str) -> Self {
        Self::new(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/models",
            api_key,
            model,
        )
    }
    
    /// Create config for Qwen (Alibaba)
    pub fn qwen(api_key: &str, model: &str) -> Self {
        Self::new(
            "qwen",
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            api_key,
            model,
        )
    }
    
    /// Create config for Ollama (local)
    pub fn ollama(model: &str, base_url: Option<&str>) -> Self {
        let url = base_url.unwrap_or("http://localhost:11434");
        Self::new(
            "ollama",
            &format!("{}/api/chat", url),
            "", // Ollama doesn't require API key
            model,
        )
    }
    
    /// Create config for LM Studio (local)
    pub fn lm_studio(model: &str, base_url: Option<&str>) -> Self {
        let url = base_url.unwrap_or("http://localhost:1234");
        Self::new(
            "lm-studio",
            &format!("{}/v1/chat/completions", url),
            "", // LM Studio doesn't require API key
            model,
        )
    }
    
    /// Create config for llama.cpp server
    pub fn llama_cpp(model: &str, base_url: Option<&str>) -> Self {
        let url = base_url.unwrap_or("http://localhost:8080");
        Self::new(
            "llama-cpp",
            &format!("{}/v1/chat/completions", url),
            "", // llama.cpp doesn't require API key
            model,
        )
    }
}

/// Configuration for code compression
///
/// Note: The `language` field is only used for `compress_code()` method.
/// When using `compress_text()`, this field is ignored.
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression rate (0.0-1.0)
    pub rate: f64,
    /// Target number of tokens (-1 for auto)
    pub target_token: i32,
    /// Programming language (only used for compress_code, ignored for compress_text)
    pub language: CodeLanguage,
    /// Dynamic compression ratio
    pub dynamic_compression_ratio: f64,
    /// Context budget adjustment (e.g., "+100")
    pub context_budget: String,
    /// If true, only rank functions without fine-grained compression
    pub rank_only: bool,
    /// Fine-grained compression rate
    pub fine_ratio: Option<f64>,
    /// Minimum lines for fine-grained compression
    pub min_lines_for_fine_grained: usize,
    /// Importance weighting factor
    pub importance_beta: f64,
    /// Use knapsack algorithm for block selection
    pub use_knapsack: bool,
    /// Repeat instruction at end of prompt
    pub repeat_instruction_at_end: bool,
    /// Provider configuration
    pub provider: ProviderConfig,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            rate: 0.5,
            target_token: -1,
            language: CodeLanguage::Python,
            dynamic_compression_ratio: 0.2,
            context_budget: "+100".to_string(),
            rank_only: false,
            fine_ratio: None,
            min_lines_for_fine_grained: 5,
            importance_beta: 0.5,
            use_knapsack: true,
            repeat_instruction_at_end: true,
            provider: ProviderConfig::new(
                "deepseek",
                "https://api.deepseek.com/chat/completions",
                "",
                "deepseek-chat"
            ),
        }
    }
}

impl CompressionConfig {
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate;
        self
    }
    
    pub fn with_target_token(mut self, target: i32) -> Self {
        self.target_token = target;
        self
    }
    
    pub fn with_language(mut self, language: CodeLanguage) -> Self {
        self.language = language;
        self
    }
    
    pub fn with_provider(mut self, provider: ProviderConfig) -> Self {
        self.provider = provider;
        self
    }
    
    pub fn with_rank_only(mut self, rank_only: bool) -> Self {
        self.rank_only = rank_only;
        self
    }
}

/// Result of code compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionResult {
    /// Original code
    pub original_code: String,
    /// Compressed code
    pub compressed_code: String,
    /// Full compressed prompt with instruction and query
    pub compressed_prompt: String,
    /// Number of tokens in original code
    pub original_tokens: usize,
    /// Number of tokens in compressed code
    pub compressed_tokens: usize,
    /// Final tokens including instruction and query
    pub final_compressed_tokens: usize,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Per-function compression details
    pub function_compressions: HashMap<usize, FunctionCompression>,
    /// Selected function indices
    pub selected_functions: Vec<usize>,
    /// Fine-grained method used (if any)
    pub fine_grained_method_used: Option<String>,
}

/// Compression details for a single function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCompression {
    /// Original token count
    pub original_tokens: usize,
    /// Compressed token count
    pub compressed_tokens: usize,
    /// Compression ratio for this function
    pub compression_ratio: f64,
    /// Individual fine-grained ratio applied
    pub individual_fine_ratio: Option<f64>,
    /// Note about compression decision
    pub note: Option<String>,
}

/// Response from LLM API for perplexity calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Request to LLM API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
}
