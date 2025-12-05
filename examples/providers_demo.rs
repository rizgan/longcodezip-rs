//! Example demonstrating different LLM provider configurations
//! 
//! This example shows how to configure and use various LLM providers
//! including OpenAI, Anthropic, Azure, Gemini, Qwen, and local models.

use longcodezip::{LongCodeZip, CompressionConfig};
use longcodezip::types::{CodeLanguage, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("LongCodeZip Provider Examples\n");
    println!("This example demonstrates configuration for various LLM providers.\n");
    
    // Example code to compress
    let code = r#"
def calculate_fibonacci(n):
    """Calculate Fibonacci number recursively."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def calculate_factorial(n):
    """Calculate factorial iteratively."""
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"#;

    println!("Original code ({} lines):\n{}\n", code.lines().count(), code);
    
    // Example 1: OpenAI
    println!("=== Example 1: OpenAI ===");
    let openai_config = ProviderConfig::openai(
        "your-openai-api-key",
        "gpt-4"
    );
    println!("Provider: {}", openai_config.provider);
    println!("API URL: {}", openai_config.api_url);
    println!("Model: {}\n", openai_config.model);
    
    // Example 2: DeepSeek
    println!("=== Example 2: DeepSeek ===");
    let deepseek_config = ProviderConfig::deepseek("your-deepseek-api-key");
    println!("Provider: {}", deepseek_config.provider);
    println!("API URL: {}", deepseek_config.api_url);
    println!("Model: {}\n", deepseek_config.model);
    
    // Example 3: Anthropic Claude
    println!("=== Example 3: Anthropic Claude ===");
    let claude_config = ProviderConfig::claude(
        "your-anthropic-api-key",
        "claude-3-5-sonnet-20241022"
    );
    println!("Provider: {}", claude_config.provider);
    println!("API URL: {}", claude_config.api_url);
    println!("Model: {}\n", claude_config.model);
    
    // Example 4: Azure OpenAI
    println!("=== Example 4: Azure OpenAI ===");
    let azure_config = ProviderConfig::azure_openai(
        "your-azure-api-key",
        "your-resource-name",
        "your-deployment-name",
        "2024-02-01"
    );
    println!("Provider: {}", azure_config.provider);
    println!("API URL: {}", azure_config.api_url);
    println!("Model: {}\n", azure_config.model);
    
    // Example 5: Google Gemini
    println!("=== Example 5: Google Gemini ===");
    let gemini_config = ProviderConfig::gemini(
        "your-google-api-key",
        "gemini-pro"
    );
    println!("Provider: {}", gemini_config.provider);
    println!("API URL: {}", gemini_config.api_url);
    println!("Model: {}\n", gemini_config.model);
    
    // Example 6: Qwen (Alibaba)
    println!("=== Example 6: Qwen (Alibaba) ===");
    let qwen_config = ProviderConfig::qwen(
        "your-qwen-api-key",
        "qwen-turbo"
    );
    println!("Provider: {}", qwen_config.provider);
    println!("API URL: {}", qwen_config.api_url);
    println!("Model: {}\n", qwen_config.model);
    
    // Example 7: Ollama (Local)
    println!("=== Example 7: Ollama (Local) ===");
    let ollama_config = ProviderConfig::ollama(
        "llama3.1:8b",
        None // Uses default http://localhost:11434
    );
    println!("Provider: {}", ollama_config.provider);
    println!("API URL: {}", ollama_config.api_url);
    println!("Model: {}", ollama_config.model);
    println!("Note: No API key required for local models\n");
    
    // Example 8: LM Studio (Local)
    println!("=== Example 8: LM Studio (Local) ===");
    let lm_studio_config = ProviderConfig::lm_studio(
        "local-model",
        None // Uses default http://localhost:1234
    );
    println!("Provider: {}", lm_studio_config.provider);
    println!("API URL: {}", lm_studio_config.api_url);
    println!("Model: {}\n", lm_studio_config.model);
    
    // Example 9: llama.cpp server (Local)
    println!("=== Example 9: llama.cpp Server (Local) ===");
    let llama_cpp_config = ProviderConfig::llama_cpp(
        "llama-3-8b",
        Some("http://localhost:8080") // Custom URL
    );
    println!("Provider: {}", llama_cpp_config.provider);
    println!("API URL: {}", llama_cpp_config.api_url);
    println!("Model: {}\n", llama_cpp_config.model);
    
    // Demonstrate actual compression with DeepSeek if API key is available
    if let Ok(api_key) = std::env::var("DEEPSEEK_API_KEY") {
        println!("\n=== Running actual compression with DeepSeek ===");
        
        let config = CompressionConfig::default()
            .with_rate(0.5)
            .with_language(CodeLanguage::Python)
            .with_provider(ProviderConfig::deepseek(&api_key));
        
        let compressor = LongCodeZip::new(config)?;
        
        let query = "Show me the prime number checking function";
        match compressor.compress_code(code, query, "").await {
            Ok(result) => {
                println!("Compression successful!");
                println!("Original tokens: {}", result.original_tokens);
                println!("Compressed tokens: {}", result.compressed_tokens);
                println!("Compression ratio: {:.2}%", result.compression_ratio * 100.0);
                println!("\nCompressed code:\n{}", result.compressed_code);
            }
            Err(e) => {
                println!("Compression failed: {}", e);
            }
        }
    } else {
        println!("\n=== Skipping actual compression ===");
        println!("Set DEEPSEEK_API_KEY environment variable to run actual compression");
    }
    
    println!("\n=== Configuration Tips ===");
    println!("1. Cloud providers (OpenAI, Anthropic, etc.) require API keys");
    println!("2. Local models (Ollama, LM Studio) don't require API keys");
    println!("3. Azure OpenAI requires resource name and deployment name");
    println!("4. All providers support the same LLMProvider trait interface");
    println!("5. Token counting uses tiktoken for accurate estimation");
    
    Ok(())
}
