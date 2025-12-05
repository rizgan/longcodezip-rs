# Usage Examples - LongCodeZip-rs

Complete examples showing how to use LongCodeZip in different scenarios, with and without LLM APIs.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Without LLM (Local Analysis)](#without-llm-local-analysis)
- [With LLM (Smart Compression)](#with-llm-smart-compression)
- [Advanced Examples](#advanced-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Basic Usage

### Minimal Example with LLM

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup provider
    let provider = ProviderConfig::deepseek("your-api-key");
    
    // Configure compression
    let config = CompressionConfig::default()
        .with_rate(0.5)  // Keep 50% of tokens
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    // Create compressor
    let compressor = LongCodeZip::new(config)?;
    
    // Compress code
    let code = "def hello(): print('world')";
    let result = compressor.compress_code(code, "", "").await?;
    
    println!("Compressed: {}", result.compressed_code);
    println!("Ratio: {:.1}%", result.compression_ratio * 100.0);
    
    Ok(())
}
```

## Without LLM (Local Analysis)

### Example 1: Token Counting

```rust
use longcodezip::{Tokenizer, TokenizerModel};

fn count_tokens() -> Result<(), Box<dyn std::error::Error>> {
    // Create tokenizer (no API needed)
    let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
    
    let code = r#"
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    "#;
    
    // Count tokens locally
    let tokens = tokenizer.count_tokens(code)?;
    println!("Token count: {}", tokens);
    
    Ok(())
}
```

### Example 2: Compression Planning

```rust
use longcodezip::{Tokenizer, TokenizerModel, ApproximateTokenizer};

fn plan_compression() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
    
    // Read your code
    let code = std::fs::read_to_string("my_project.py")?;
    let total_tokens = tokenizer.count_tokens(&code)?;
    
    println!("Original: {} tokens", total_tokens);
    println!("\nCompression estimates:");
    
    for rate in [0.2, 0.3, 0.5, 0.7] {
        let target = (total_tokens as f64 * rate) as usize;
        let saved = total_tokens - target;
        let saved_pct = ((1.0 - rate) * 100.0) as usize;
        
        println!("  {}%: {} tokens (save {} / {}%)", 
                 (rate * 100.0) as usize, 
                 target, 
                 saved, 
                 saved_pct);
    }
    
    Ok(())
}
```

### Example 3: Approximate Token Counting (Fast)

```rust
use longcodezip::ApproximateTokenizer;

fn fast_estimation() {
    let tokenizer = ApproximateTokenizer::new();
    
    let code = "let x = 42;\nconsole.log(x);";
    let approx_tokens = tokenizer.count_tokens(code);
    
    println!("Approximate tokens: {}", approx_tokens);
    // Fast estimation: ~4 characters per token
}
```

### Example 4: Batch File Analysis (No LLM)

```rust
use longcodezip::Tokenizer;
use std::fs;
use std::path::Path;

fn analyze_project() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
    let mut total_tokens = 0;
    
    let files = vec!["src/main.rs", "src/lib.rs", "src/utils.rs"];
    
    for file_path in files {
        let content = fs::read_to_string(file_path)?;
        let tokens = tokenizer.count_tokens(&content)?;
        
        println!("{}: {} tokens", 
                 Path::new(file_path).file_name().unwrap().to_str().unwrap(),
                 tokens);
        
        total_tokens += tokens;
    }
    
    println!("\nTotal: {} tokens", total_tokens);
    println!("30% compression would save: {} tokens", 
             (total_tokens as f64 * 0.7) as usize);
    
    Ok(())
}
```

## With LLM (Smart Compression)

### Example 5: Query-Based Compression

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn query_compression() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    let code = r#"
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
    "#;
    
    // The query helps prioritize relevant code
    let result = compressor.compress_code(
        code,
        "How to calculate fibonacci?",  // Query
        "Extract relevant functions:"    // Instruction
    ).await?;
    
    println!("Original: {} tokens", result.original_tokens);
    println!("Compressed: {} tokens", result.compressed_tokens);
    println!("\n{}", result.compressed_code);
    
    Ok(())
}
```

### Example 6: Multiple LLM Providers

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn multi_provider() -> Result<(), Box<dyn std::error::Error>> {
    let code = "fn main() { println!(\"Hello\"); }";
    
    // Try DeepSeek
    let provider1 = ProviderConfig::deepseek("deepseek-key");
    let config1 = CompressionConfig::default()
        .with_rate(0.5)
        .with_provider(provider1);
    let compressor1 = LongCodeZip::new(config1)?;
    let result1 = compressor1.compress_code(code, "", "").await?;
    println!("DeepSeek: {:.1}% compression", result1.compression_ratio * 100.0);
    
    // Try OpenAI
    let provider2 = ProviderConfig::openai("openai-key");
    let config2 = CompressionConfig::default()
        .with_rate(0.5)
        .with_provider(provider2);
    let compressor2 = LongCodeZip::new(config2)?;
    let result2 = compressor2.compress_code(code, "", "").await?;
    println!("OpenAI: {:.1}% compression", result2.compression_ratio * 100.0);
    
    // Try Ollama (local, free)
    let provider3 = ProviderConfig::ollama("llama3");
    let config3 = CompressionConfig::default()
        .with_rate(0.5)
        .with_provider(provider3);
    let compressor3 = LongCodeZip::new(config3)?;
    let result3 = compressor3.compress_code(code, "", "").await?;
    println!("Ollama: {:.1}% compression", result3.compression_ratio * 100.0);
    
    Ok(())
}
```

### Example 7: Caching for Performance

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn with_caching() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_provider(provider)
        .with_cache(true)  // Enable caching
        .with_cache_ttl(3600);  // 1 hour cache lifetime
    
    let compressor = LongCodeZip::new(config)?;
    let code = "def test(): pass";
    
    // First call - hits API
    let start = std::time::Instant::now();
    let result1 = compressor.compress_code(code, "test", "").await?;
    println!("First call: {} ms", start.elapsed().as_millis());
    
    // Second call - uses cache (much faster!)
    let start = std::time::Instant::now();
    let result2 = compressor.compress_code(code, "test", "").await?;
    println!("Cached call: {} ms", start.elapsed().as_millis());
    
    // View cache stats
    let stats = compressor.cache_stats();
    println!("Cache entries: {}", stats.valid_entries);
    println!("Hit rate: {:.1}%", stats.hit_rate * 100.0);
    
    Ok(())
}
```

### Example 8: Parallel Processing

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn parallel_processing() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_provider(provider)
        .with_parallel(true)  // Enable parallel processing
        .with_parallel_threads(4);  // Use 4 threads
    
    let compressor = LongCodeZip::new(config)?;
    
    let files = vec![
        std::fs::read_to_string("file1.py")?,
        std::fs::read_to_string("file2.py")?,
        std::fs::read_to_string("file3.py")?,
    ];
    
    let start = std::time::Instant::now();
    
    for (i, code) in files.iter().enumerate() {
        let result = compressor.compress_code(code, "", "").await?;
        println!("File {}: {} → {} tokens", 
                 i + 1, 
                 result.original_tokens, 
                 result.compressed_tokens);
    }
    
    println!("\nTotal time: {} ms", start.elapsed().as_millis());
    
    Ok(())
}
```

## Advanced Examples

### Example 9: Fine-Grained Compression

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn fine_grained() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_provider(provider)
        .with_use_knapsack(true)  // Enable knapsack optimization
        .with_rank_only(false);   // Use fine-grained compression
    
    let compressor = LongCodeZip::new(config)?;
    
    let code = std::fs::read_to_string("large_file.py")?;
    let result = compressor.compress_code(code.as_str(), "", "").await?;
    
    println!("Fine-grained compression results:");
    println!("  Original: {} tokens", result.original_tokens);
    println!("  Compressed: {} tokens", result.compressed_tokens);
    println!("  Ratio: {:.2}%", result.compression_ratio * 100.0);
    println!("  Method: {}", result.fine_grained_method_used.unwrap_or_default());
    
    Ok(())
}
```

### Example 10: Text Compression (Not Code)

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig, TextChunkingStrategy};

#[tokio::main]
async fn text_compression() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    let text = std::fs::read_to_string("article.txt")?;
    
    // Compress by paragraphs
    let result = compressor.compress_text(
        &text,
        "What are the main points?",
        "Summarize:",
        TextChunkingStrategy::Paragraphs
    ).await?;
    
    println!("Text compression:");
    println!("  {} → {} tokens", result.original_tokens, result.compressed_tokens);
    println!("\n{}", result.compressed_code);
    
    Ok(())
}
```

### Example 11: Custom Token Budget

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn custom_budget() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    
    // Instead of percentage, use exact token count
    let config = CompressionConfig::default()
        .with_target_token(500)  // Compress to exactly 500 tokens
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    let code = std::fs::read_to_string("big_file.py")?;
    let result = compressor.compress_code(&code, "", "").await?;
    
    println!("Target: 500 tokens");
    println!("Actual: {} tokens", result.compressed_tokens);
    println!("Original: {} tokens", result.original_tokens);
    
    Ok(())
}
```

### Example 12: Environment Variable Configuration

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};
use std::env;

#[tokio::main]
async fn env_config() -> Result<(), Box<dyn std::error::Error>> {
    // Read from environment
    let api_key = env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY not set");
    
    let provider = ProviderConfig::deepseek(&api_key);
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    // Use compressor...
    Ok(())
}
```

## Error Handling

### Example 13: Proper Error Handling

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig, Error};

#[tokio::main]
async fn handle_errors() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ProviderConfig::deepseek("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_provider(provider);
    
    let compressor = match LongCodeZip::new(config) {
        Ok(c) => c,
        Err(Error::ConfigError(msg)) => {
            eprintln!("Configuration error: {}", msg);
            return Err(msg.into());
        }
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            return Err(e.into());
        }
    };
    
    let code = "def test(): pass";
    
    match compressor.compress_code(code, "", "").await {
        Ok(result) => {
            println!("Success: {} → {} tokens", 
                     result.original_tokens, 
                     result.compressed_tokens);
        }
        Err(Error::ApiError(msg)) => {
            eprintln!("API error: {}", msg);
            eprintln!("Check your API key and network connection");
        }
        Err(Error::RequestError(e)) => {
            eprintln!("Network error: {}", e);
        }
        Err(e) => {
            eprintln!("Compression failed: {}", e);
        }
    }
    
    Ok(())
}
```

## Best Practices

### Example 14: Production-Ready Setup

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};
use std::sync::Arc;

#[tokio::main]
async fn production_setup() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Use environment variables for secrets
    let api_key = std::env::var("LLM_API_KEY")?;
    
    // 2. Configure with sensible defaults
    let provider = ProviderConfig::deepseek(&api_key);
    let config = CompressionConfig::default()
        .with_rate(0.3)
        .with_provider(provider)
        .with_cache(true)           // Always enable cache
        .with_cache_ttl(3600)       // 1 hour
        .with_parallel(true)        // Parallel processing
        .with_parallel_threads(4)   // Match CPU cores
        .with_use_knapsack(true);   // Better quality
    
    // 3. Create compressor once, reuse many times
    let compressor = Arc::new(LongCodeZip::new(config)?);
    
    // 4. Handle errors gracefully
    let code = std::fs::read_to_string("file.py")?;
    
    match compressor.compress_code(&code, "", "").await {
        Ok(result) => {
            // 5. Log important metrics
            println!("Compression: {:.1}%", result.compression_ratio * 100.0);
            
            // 6. Save compressed output
            std::fs::write("file.compressed.py", &result.compressed_code)?;
        }
        Err(e) => {
            eprintln!("Compression failed: {}", e);
            // 7. Fall back to original if compression fails
            std::fs::write("file.compressed.py", &code)?;
        }
    }
    
    // 8. Monitor cache effectiveness
    let stats = compressor.cache_stats();
    println!("Cache hit rate: {:.1}%", stats.hit_rate * 100.0);
    
    Ok(())
}
```

### Example 15: Testing Without API

```rust
#[cfg(test)]
mod tests {
    use longcodezip::{Tokenizer, TokenizerModel};

    #[test]
    fn test_token_counting() {
        // Test without LLM - fast, free, no API needed
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        
        let code = "def test(): pass";
        let tokens = tokenizer.count_tokens(code).unwrap();
        
        assert!(tokens > 0);
        assert!(tokens < 100);
    }
    
    #[test]
    fn test_compression_estimates() {
        let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
        let code = "fn main() { println!(\"test\"); }";
        let tokens = tokenizer.count_tokens(code).unwrap();
        
        let target_30 = (tokens as f64 * 0.3) as usize;
        assert!(target_30 < tokens);
    }
}
```

## Quick Reference

```rust
// WITHOUT LLM (Free, Fast, Local)
let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
let tokens = tokenizer.count_tokens(code)?;

// WITH LLM (Smart, Quality, API Cost)
let provider = ProviderConfig::deepseek("key");
let config = CompressionConfig::default()
    .with_rate(0.3)
    .with_provider(provider);
let compressor = LongCodeZip::new(config)?;
let result = compressor.compress_code(code, query, instruction).await?;

// HYBRID (Best of Both)
let tokens = tokenizer.count_tokens(code)?;
if tokens > 1000 {
    // Use LLM for large files
    let result = compressor.compress_code(code, "", "").await?;
} else {
    // Small file, no compression needed
}
```

## More Information

- [COMPRESSION_BENCHMARKS.md](COMPRESSION_BENCHMARKS.md) - Performance data
- [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md) - LLM provider setup
- [QUICKSTART.md](QUICKSTART.md) - Getting started
- [API Documentation](https://docs.rs/longcodezip) - Full API reference

---

**Last Updated**: December 2025  
**Version**: 0.6.0
