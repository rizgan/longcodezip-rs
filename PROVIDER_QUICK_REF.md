# Quick Provider Reference

Quick reference for configuring different LLM providers in LongCodeZip.

## Cloud Providers

### OpenAI
```rust
ProviderConfig::openai("sk-...", "gpt-4")
```
- Models: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`
- [Get API Key](https://platform.openai.com/api-keys)

### DeepSeek
```rust
ProviderConfig::deepseek("sk-...")
```
- Model: `deepseek-chat`
- [Get API Key](https://platform.deepseek.com/)

### Anthropic Claude
```rust
ProviderConfig::claude("sk-ant-...", "claude-3-5-sonnet-20241022")
```
- Models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`, `claude-3-haiku-20240307`
- [Get API Key](https://console.anthropic.com/)

### Azure OpenAI
```rust
ProviderConfig::azure_openai(
    "your-api-key",
    "resource-name",
    "deployment-name",
    "2024-02-01"
)
```
- Requires Azure subscription
- [Azure Portal](https://portal.azure.com/)

### Google Gemini
```rust
ProviderConfig::gemini("AI...", "gemini-pro")
```
- Models: `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`
- [Get API Key](https://makersuite.google.com/app/apikey)

### Qwen (Alibaba)
```rust
ProviderConfig::qwen("sk-...", "qwen-turbo")
```
- Models: `qwen-turbo`, `qwen-plus`, `qwen-max`
- [Get API Key](https://dashscope.console.aliyun.com/)

## Local Providers (No API Key!)

### Ollama
```rust
ProviderConfig::ollama("llama3.1:8b", None)
```
Setup:
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.1:8b
```
Popular models: `llama3.1:8b`, `codellama:7b`, `mistral:7b`, `deepseek-coder:6.7b`

### LM Studio
```rust
ProviderConfig::lm_studio("local-model", None)
```
Setup:
1. Download [LM Studio](https://lmstudio.ai/)
2. Download a model
3. Start local server (port 1234)

### llama.cpp
```rust
ProviderConfig::llama_cpp("model", Some("http://localhost:8080"))
```
Setup:
```bash
# Get llama.cpp from https://github.com/ggerganov/llama.cpp
./server -m model.gguf --port 8080
```

## Complete Example

```rust
use longcodezip::{LongCodeZip, CompressionConfig};
use longcodezip::types::{CodeLanguage, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Choose provider
    let provider = ProviderConfig::deepseek(&std::env::var("DEEPSEEK_API_KEY")?);
    // Or local: ProviderConfig::ollama("llama3.1:8b", None)
    
    // Configure
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    // Compress
    let compressor = LongCodeZip::new(config)?;
    let result = compressor.compress_code(code, query, "").await?;
    
    println!("Compressed: {} → {} tokens", 
             result.original_tokens, 
             result.compressed_tokens);
    
    Ok(())
}
```

## Environment Variables

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."

# Use in code
let provider = ProviderConfig::deepseek(&std::env::var("DEEPSEEK_API_KEY")?);
```

## Decision Matrix

**Best for Quality:** OpenAI GPT-4, Claude 3.5 Sonnet  
**Best for Cost:** DeepSeek, Gemini Pro  
**Best for Privacy:** Ollama, LM Studio, llama.cpp (local)  
**Best for Development:** Ollama (free, local, no key)  
**Best for Chinese:** Qwen  
**Best for Code:** DeepSeek, CodeLlama (Ollama)  

## See Also

- [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md) - Complete documentation
- [EXAMPLES.md](EXAMPLES.md) - Usage examples
- [README.md](README.md) - Main documentation
