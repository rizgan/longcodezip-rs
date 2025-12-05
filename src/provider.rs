//! LLM provider for calculating perplexity and token counts

use crate::error::{Error, Result};
use crate::types::{LLMRequest, LLMResponse, Message, ProviderConfig};
use crate::tokenizer::Tokenizer;
use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;

/// Trait for LLM providers
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Calculate approximate token count for text
    async fn get_token_count(&self, text: &str) -> Result<usize>;
    
    /// Calculate perplexity or relevance score for text given context
    /// Returns a score where higher means more relevant
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64>;
    
    /// Get completion from the model
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}

/// OpenAI-compatible provider (works with DeepSeek, OpenAI, etc.)
pub struct OpenAICompatibleProvider {
    config: ProviderConfig,
    client: Client,
    tokenizer: Tokenizer,
}

impl OpenAICompatibleProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let tokenizer = Tokenizer::from_model_name(&config.model);
        Self {
            config,
            client: Client::new(),
            tokenizer,
        }
    }
    
    async fn make_request(&self, messages: Vec<Message>, max_tokens: Option<usize>) -> Result<LLMResponse> {
        let request = LLMRequest {
            model: self.config.model.clone(),
            messages,
            temperature: Some(self.config.temperature),
            max_tokens,
            logprobs: None,
            top_logprobs: None,
        };
        
        let response = self.client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::ApiError(format!("API request failed with status {}: {}", status, error_text)));
        }
        
        let llm_response: LLMResponse = response.json().await?;
        Ok(llm_response)
    }
}

#[async_trait]
impl LLMProvider for OpenAICompatibleProvider {
    async fn get_token_count(&self, text: &str) -> Result<usize> {
        // Use accurate tiktoken tokenizer
        self.tokenizer.count_tokens(text)
    }
    
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
        // Calculate relevance by asking the model to rate how helpful the context is
        // for answering the query. Returns a score approximation.
        
        if context.trim().is_empty() {
            return Ok(0.0);
        }
        
        if query.trim().is_empty() {
            // Without a query, we can't determine relevance
            // Return a neutral score based on length
            let tokens = self.get_token_count(context).await?;
            return Ok(tokens as f64 * 0.01); // Small positive score
        }
        
        // For performance, we use a simpler heuristic:
        // 1. Check if context contains keywords from query
        // 2. Calculate a relevance score based on keyword overlap
        
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();
        
        // Extract words from query
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3) // Skip short words
            .collect();
        
        if query_words.is_empty() {
            return Ok(0.0);
        }
        
        // Count keyword matches
        let mut match_count = 0;
        for word in &query_words {
            if context_lower.contains(word) {
                match_count += 1;
            }
        }
        
        // Calculate relevance score
        let match_ratio = match_count as f64 / query_words.len() as f64;
        let token_count = self.get_token_count(context).await? as f64;
        
        // Score combines match ratio with a length penalty/bonus
        // Higher match ratio = more relevant
        // Normalize by log of length to prevent very long contexts from dominating
        let relevance = match_ratio * 10.0 - (token_count.ln() / 10.0);
        
        Ok(relevance)
    }
    
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful code assistant. Provide concise, accurate responses.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        
        let response = self.make_request(messages, Some(max_tokens)).await?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(Error::ApiError("No response from API".to_string()))
        }
    }
}

/// Create a provider from configuration
pub fn create_provider(config: ProviderConfig) -> Box<dyn LLMProvider> {
    match config.provider.as_str() {
        "anthropic" => Box::new(AnthropicProvider::new(config)),
        "gemini" => Box::new(GeminiProvider::new(config)),
        "qwen" => Box::new(QwenProvider::new(config)),
        "azure-openai" => Box::new(AzureOpenAIProvider::new(config)),
        // OpenAI-compatible providers: OpenAI, DeepSeek, Ollama, LM Studio, llama.cpp
        _ => Box::new(OpenAICompatibleProvider::new(config)),
    }
}

/// Anthropic Claude provider
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: Client,
    tokenizer: Tokenizer,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let tokenizer = Tokenizer::from_model_name(&config.model);
        Self {
            config,
            client: Client::new(),
            tokenizer,
        }
    }
    
    async fn make_request(&self, messages: Vec<Message>, max_tokens: Option<usize>) -> Result<LLMResponse> {
        // Anthropic uses a different API format
        let system_message = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone());
        
        let user_messages: Vec<_> = messages.iter()
            .filter(|m| m.role != "system")
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        
        let mut body = json!({
            "model": self.config.model,
            "messages": user_messages,
            "max_tokens": max_tokens.unwrap_or(self.config.max_tokens),
        });
        
        if let Some(system) = system_message {
            body["system"] = json!(system);
        }
        
        let response = self.client
            .post(&self.config.api_url)
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::ApiError(format!("Anthropic API request failed with status {}: {}", status, error_text)));
        }
        
        // Convert Anthropic response to our standard format
        let anthropic_response: serde_json::Value = response.json().await?;
        let content = anthropic_response["content"][0]["text"]
            .as_str()
            .ok_or_else(|| Error::ApiError("Invalid Anthropic response format".to_string()))?
            .to_string();
        
        Ok(LLMResponse {
            choices: vec![crate::types::Choice {
                message: Message {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: anthropic_response["stop_reason"].as_str().map(String::from),
            }],
            usage: Some(crate::types::Usage {
                prompt_tokens: anthropic_response["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize,
                completion_tokens: anthropic_response["usage"]["output_tokens"].as_u64().unwrap_or(0) as usize,
                total_tokens: 0,
            }),
        })
    }
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn get_token_count(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }
    
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
        // Use same heuristic as OpenAI
        if context.trim().is_empty() {
            return Ok(0.0);
        }
        
        if query.trim().is_empty() {
            let tokens = self.get_token_count(context).await?;
            return Ok(tokens as f64 * 0.01);
        }
        
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();
        
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        if query_words.is_empty() {
            return Ok(0.0);
        }
        
        let mut match_count = 0;
        for word in &query_words {
            if context_lower.contains(word) {
                match_count += 1;
            }
        }
        
        let match_ratio = match_count as f64 / query_words.len() as f64;
        let token_count = self.get_token_count(context).await? as f64;
        let relevance = match_ratio * 10.0 - (token_count.ln() / 10.0);
        
        Ok(relevance)
    }
    
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        
        let response = self.make_request(messages, Some(max_tokens)).await?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(Error::ApiError("No response from Anthropic API".to_string()))
        }
    }
}

/// Google Gemini provider
pub struct GeminiProvider {
    config: ProviderConfig,
    client: Client,
    tokenizer: Tokenizer,
}

impl GeminiProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let tokenizer = Tokenizer::from_model_name(&config.model);
        Self {
            config,
            client: Client::new(),
            tokenizer,
        }
    }
    
    async fn make_request(&self, messages: Vec<Message>, max_tokens: Option<usize>) -> Result<LLMResponse> {
        // Convert messages to Gemini format
        let contents: Vec<_> = messages.iter()
            .filter(|m| m.role != "system")
            .map(|m| json!({
                "role": if m.role == "assistant" { "model" } else { "user" },
                "parts": [{"text": m.content}]
            }))
            .collect();
        
        let system_instruction = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| json!({"parts": [{"text": m.content}]}));
        
        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": max_tokens.unwrap_or(self.config.max_tokens),
            }
        });
        
        if let Some(instruction) = system_instruction {
            body["systemInstruction"] = instruction;
        }
        
        let url = format!("{}/{}:generateContent?key={}", 
                         self.config.api_url, self.config.model, self.config.api_key);
        
        let response = self.client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::ApiError(format!("Gemini API request failed with status {}: {}", status, error_text)));
        }
        
        let gemini_response: serde_json::Value = response.json().await?;
        let content = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .ok_or_else(|| Error::ApiError("Invalid Gemini response format".to_string()))?
            .to_string();
        
        Ok(LLMResponse {
            choices: vec![crate::types::Choice {
                message: Message {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: gemini_response["candidates"][0]["finishReason"].as_str().map(String::from),
            }],
            usage: gemini_response["usageMetadata"].as_object().map(|usage| {
                crate::types::Usage {
                    prompt_tokens: usage["promptTokenCount"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: usage["candidatesTokenCount"].as_u64().unwrap_or(0) as usize,
                    total_tokens: usage["totalTokenCount"].as_u64().unwrap_or(0) as usize,
                }
            }),
        })
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    async fn get_token_count(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }
    
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
        // Use same heuristic
        if context.trim().is_empty() {
            return Ok(0.0);
        }
        
        if query.trim().is_empty() {
            let tokens = self.get_token_count(context).await?;
            return Ok(tokens as f64 * 0.01);
        }
        
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();
        
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        if query_words.is_empty() {
            return Ok(0.0);
        }
        
        let mut match_count = 0;
        for word in &query_words {
            if context_lower.contains(word) {
                match_count += 1;
            }
        }
        
        let match_ratio = match_count as f64 / query_words.len() as f64;
        let token_count = self.get_token_count(context).await? as f64;
        let relevance = match_ratio * 10.0 - (token_count.ln() / 10.0);
        
        Ok(relevance)
    }
    
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        
        let response = self.make_request(messages, Some(max_tokens)).await?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(Error::ApiError("No response from Gemini API".to_string()))
        }
    }
}

/// Qwen (Alibaba) provider
pub struct QwenProvider {
    config: ProviderConfig,
    client: Client,
    tokenizer: Tokenizer,
}

impl QwenProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let tokenizer = Tokenizer::from_model_name(&config.model);
        Self {
            config,
            client: Client::new(),
            tokenizer,
        }
    }
    
    async fn make_request(&self, messages: Vec<Message>, max_tokens: Option<usize>) -> Result<LLMResponse> {
        let body = json!({
            "model": self.config.model,
            "input": {
                "messages": messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect::<Vec<_>>()
            },
            "parameters": {
                "result_format": "message",
                "max_tokens": max_tokens.unwrap_or(self.config.max_tokens),
            }
        });
        
        let response = self.client
            .post(&self.config.api_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::ApiError(format!("Qwen API request failed with status {}: {}", status, error_text)));
        }
        
        let qwen_response: serde_json::Value = response.json().await?;
        let content = qwen_response["output"]["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| Error::ApiError("Invalid Qwen response format".to_string()))?
            .to_string();
        
        Ok(LLMResponse {
            choices: vec![crate::types::Choice {
                message: Message {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: qwen_response["output"]["choices"][0]["finish_reason"].as_str().map(String::from),
            }],
            usage: qwen_response["usage"].as_object().map(|usage| {
                crate::types::Usage {
                    prompt_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as usize,
                    completion_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as usize,
                    total_tokens: usage["total_tokens"].as_u64().unwrap_or(0) as usize,
                }
            }),
        })
    }
}

#[async_trait]
impl LLMProvider for QwenProvider {
    async fn get_token_count(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }
    
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
        // Use same heuristic
        if context.trim().is_empty() {
            return Ok(0.0);
        }
        
        if query.trim().is_empty() {
            let tokens = self.get_token_count(context).await?;
            return Ok(tokens as f64 * 0.01);
        }
        
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();
        
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        if query_words.is_empty() {
            return Ok(0.0);
        }
        
        let mut match_count = 0;
        for word in &query_words {
            if context_lower.contains(word) {
                match_count += 1;
            }
        }
        
        let match_ratio = match_count as f64 / query_words.len() as f64;
        let token_count = self.get_token_count(context).await? as f64;
        let relevance = match_ratio * 10.0 - (token_count.ln() / 10.0);
        
        Ok(relevance)
    }
    
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        
        let response = self.make_request(messages, Some(max_tokens)).await?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(Error::ApiError("No response from Qwen API".to_string()))
        }
    }
}

/// Azure OpenAI provider
pub struct AzureOpenAIProvider {
    config: ProviderConfig,
    client: Client,
    tokenizer: Tokenizer,
}

impl AzureOpenAIProvider {
    pub fn new(config: ProviderConfig) -> Self {
        let tokenizer = Tokenizer::from_model_name(&config.model);
        Self {
            config,
            client: Client::new(),
            tokenizer,
        }
    }
    
    async fn make_request(&self, messages: Vec<Message>, max_tokens: Option<usize>) -> Result<LLMResponse> {
        let request = LLMRequest {
            model: self.config.model.clone(),
            messages,
            temperature: Some(self.config.temperature),
            max_tokens,
            logprobs: None,
            top_logprobs: None,
        };
        
        let response = self.client
            .post(&self.config.api_url)
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(Error::ApiError(format!("Azure OpenAI API request failed with status {}: {}", status, error_text)));
        }
        
        let llm_response: LLMResponse = response.json().await?;
        Ok(llm_response)
    }
}

#[async_trait]
impl LLMProvider for AzureOpenAIProvider {
    async fn get_token_count(&self, text: &str) -> Result<usize> {
        self.tokenizer.count_tokens(text)
    }
    
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
        // Use same heuristic as OpenAI
        if context.trim().is_empty() {
            return Ok(0.0);
        }
        
        if query.trim().is_empty() {
            let tokens = self.get_token_count(context).await?;
            return Ok(tokens as f64 * 0.01);
        }
        
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();
        
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        if query_words.is_empty() {
            return Ok(0.0);
        }
        
        let mut match_count = 0;
        for word in &query_words {
            if context_lower.contains(word) {
                match_count += 1;
            }
        }
        
        let match_ratio = match_count as f64 / query_words.len() as f64;
        let token_count = self.get_token_count(context).await? as f64;
        let relevance = match_ratio * 10.0 - (token_count.ln() / 10.0);
        
        Ok(relevance)
    }
    
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful code assistant. Provide concise, accurate responses.".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            },
        ];
        
        let response = self.make_request(messages, Some(max_tokens)).await?;
        
        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(Error::ApiError("No response from Azure OpenAI API".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_token_count() {
        let config = ProviderConfig::new(
            "test",
            "https://api.example.com",
            "test-key",
            "gpt-4" // Use gpt-4 for cl100k_base tokenizer
        );
        let provider = OpenAICompatibleProvider::new(config);
        
        let count = provider.get_token_count("Hello world").await.unwrap();
        assert!(count > 0);
        // With tiktoken, "Hello world" should be around 2-3 tokens
        assert!(count < 5);
    }
    
    #[tokio::test]
    async fn test_relevance_empty_context() {
        let config = ProviderConfig::new(
            "test",
            "https://api.example.com",
            "test-key",
            "test-model"
        );
        let provider = OpenAICompatibleProvider::new(config);
        
        let score = provider.calculate_relevance("", "query").await.unwrap();
        assert_eq!(score, 0.0);
    }
}
