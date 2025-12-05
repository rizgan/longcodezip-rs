# Provider Guide

LongCodeZip supports multiple LLM providers, both cloud-based and local. This guide explains how to configure and use each provider.

## Supported Providers

### Cloud Providers

1. **OpenAI** - GPT-3.5, GPT-4, and other OpenAI models
2. **DeepSeek** - DeepSeek-V2 and DeepSeek-Chat
3. **Anthropic Claude** - Claude 3 models (Opus, Sonnet, Haiku)
4. **Azure OpenAI** - Microsoft Azure hosted OpenAI models
5. **Google Gemini** - Gemini Pro and other Google AI models
6. **Qwen (Alibaba)** - Qwen Turbo and other Alibaba Cloud models

### Local Providers

7. **Ollama** - Local models via Ollama
8. **LM Studio** - Local models via LM Studio
9. **llama.cpp** - Local models via llama.cpp server

## Configuration Examples

### 1. OpenAI

```rust
use longcodezip::types::ProviderConfig;

let config = ProviderConfig::openai(
    "your-api-key",
    "gpt-4"
);
```

**API Key:** Get from [OpenAI Platform](https://platform.openai.com/api-keys)

**Supported Models:**
- `gpt-4` - Most capable model
- `gpt-4-turbo` - Faster and cheaper GPT-4
- `gpt-3.5-turbo` - Fast and efficient

### 2. DeepSeek

```rust
let config = ProviderConfig::deepseek("your-api-key");
```

**API Key:** Get from [DeepSeek Platform](https://platform.deepseek.com/)

**Default Model:** `deepseek-chat`

**Features:**
- Cost-effective alternative to GPT-4
- Good code understanding capabilities
- Fast response times

### 3. Anthropic Claude

```rust
let config = ProviderConfig::claude(
    "your-api-key",
    "claude-3-5-sonnet-20241022"
);
```

**API Key:** Get from [Anthropic Console](https://console.anthropic.com/)

**Supported Models:**
- `claude-3-5-sonnet-20241022` - Latest Sonnet (recommended)
- `claude-3-opus-20240229` - Most capable
- `claude-3-haiku-20240307` - Fast and efficient

**Special Features:**
- Uses Messages API format
- Requires `anthropic-version` header
- System messages handled separately

### 4. Azure OpenAI

```rust
let config = ProviderConfig::azure_openai(
    "your-api-key",
    "your-resource-name",
    "your-deployment-name",
    "2024-02-01"
);
```

**Setup:**
1. Create Azure OpenAI resource
2. Deploy a model (e.g., gpt-4)
3. Get API key from Azure Portal

**Parameters:**
- `api_key` - From Azure Portal
- `resource_name` - Your Azure resource name
- `deployment_name` - Your model deployment name
- `api_version` - API version (use "2024-02-01")

### 5. Google Gemini

```rust
let config = ProviderConfig::gemini(
    "your-api-key",
    "gemini-pro"
);
```

**API Key:** Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

**Supported Models:**
- `gemini-pro` - Standard Gemini model
- `gemini-1.5-pro` - Latest version with larger context
- `gemini-1.5-flash` - Faster variant

**Special Features:**
- Different API format than OpenAI
- System instructions handled separately
- Good multilingual support

### 6. Qwen (Alibaba)

```rust
let config = ProviderConfig::qwen(
    "your-api-key",
    "qwen-turbo"
);
```

**API Key:** Get from [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/)

**Supported Models:**
- `qwen-turbo` - Fast and efficient
- `qwen-plus` - More capable
- `qwen-max` - Most capable

**Special Features:**
- Uses DashScope API
- Custom request format
- Good for Chinese language tasks

### 7. Ollama (Local)

```rust
let config = ProviderConfig::ollama(
    "llama3.1:8b",
    None  // Uses default http://localhost:11434
);

// Or with custom URL
let config = ProviderConfig::ollama(
    "llama3.1:8b",
    Some("http://192.168.1.100:11434")
);
```

**Setup:**
1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama3.1:8b`
3. Ollama runs on port 11434 by default

**Popular Models:**
- `llama3.1:8b` - Meta Llama 3.1 (8B parameters)
- `codellama:7b` - Code-specialized model
- `mistral:7b` - Mistral AI model
- `deepseek-coder:6.7b` - DeepSeek code model

**No API Key Required!**

### 8. LM Studio (Local)

```rust
let config = ProviderConfig::lm_studio(
    "local-model",
    None  // Uses default http://localhost:1234
);
```

**Setup:**
1. Download [LM Studio](https://lmstudio.ai/)
2. Download a model in LM Studio
3. Start the local server (default port 1234)

**Features:**
- User-friendly GUI
- Easy model management
- OpenAI-compatible API
- No API key required

### 9. llama.cpp Server (Local)

```rust
let config = ProviderConfig::llama_cpp(
    "llama-3-8b",
    Some("http://localhost:8080")
);
```

**Setup:**
1. Build or download [llama.cpp](https://github.com/ggerganov/llama.cpp)
2. Download a GGUF model file
3. Run server: `./server -m model.gguf --port 8080`

**Features:**
- Highly optimized inference
- CPU and GPU support
- Minimal dependencies
- No API key required

## Usage Example

```rust
use longcodezip::{CodeCompressor, CompressionConfig};
use longcodezip::types::{CodeLanguage, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Choose your provider
    let provider = ProviderConfig::deepseek("your-api-key");
    // Or use local Ollama
    // let provider = ProviderConfig::ollama("llama3.1:8b", None);
    
    // Create configuration
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    // Compress code
    let compressor = CodeCompressor::new(config);
    let result = compressor.compress(code, query).await?;
    
    println!("Compressed from {} to {} tokens", 
             result.original_tokens, 
             result.compressed_tokens);
    
    Ok(())
}
```

## Provider Comparison

| Provider | Cost | Speed | Quality | Local | API Key Required |
|----------|------|-------|---------|-------|------------------|
| OpenAI GPT-4 | High | Medium | Excellent | No | Yes |
| DeepSeek | Low | Fast | Very Good | No | Yes |
| Claude 3 | Medium | Fast | Excellent | No | Yes |
| Azure OpenAI | High | Medium | Excellent | No | Yes |
| Gemini Pro | Low | Fast | Very Good | No | Yes |
| Qwen | Low | Fast | Good | No | Yes |
| Ollama | Free | Medium | Good | Yes | No |
| LM Studio | Free | Medium | Good | Yes | No |
| llama.cpp | Free | Fast | Good | Yes | No |

## Environment Variables

You can use environment variables for API keys:

```bash
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Then in code:

```rust
let api_key = std::env::var("DEEPSEEK_API_KEY")
    .expect("DEEPSEEK_API_KEY not set");
let config = ProviderConfig::deepseek(&api_key);
```

## Choosing a Provider

**For production:**
- Use OpenAI GPT-4 or Claude 3.5 for best quality
- Use DeepSeek for cost-effective alternative

**For development:**
- Use Ollama or LM Studio for free local testing
- Use DeepSeek for affordable cloud testing

**For privacy:**
- Use local models (Ollama, LM Studio, llama.cpp)
- No data leaves your machine

**For specific languages:**
- Use Qwen for Chinese language tasks
- Use CodeLlama (via Ollama) for code-heavy tasks

## Troubleshooting

### API Key Issues
```
Error: API request failed with status 401
```
**Solution:** Check your API key is correct and has proper permissions

### Connection Issues
```
Error: connection refused
```
**Solution:** For local models, ensure server is running on correct port

### Model Not Found
```
Error: model not found
```
**Solution:** For Ollama, run `ollama pull <model-name>` first

### Rate Limiting
```
Error: rate limit exceeded
```
**Solution:** Add delays between requests or upgrade your plan

## Best Practices

1. **Use environment variables** for API keys (don't hardcode)
2. **Start with local models** for development
3. **Monitor costs** when using cloud providers
4. **Cache results** to avoid redundant API calls
5. **Handle errors gracefully** with retries
6. **Test with small code snippets** first
7. **Use appropriate models** for your task complexity

## See Also

- [EXAMPLES.md](EXAMPLES.md) - Code examples
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [README.md](README.md) - Main documentation
