# Примеры использования LongCodeZip

## Базовый пример

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Конфигурация провайдера
    let provider = ProviderConfig::new(
        "deepseek",
        "https://api.deepseek.com/chat/completions",
        "your-api-key",
        "deepseek-chat",
    );
    
    // Конфигурация компрессии
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    let code = r#"
    def add(a, b):
        return a + b
    
    def multiply(x, y):
        return x * y
    "#;
    
    let result = compressor.compress_code(code, "how to add numbers?", "").await?;
    
    println!("Compressed: {}", result.compressed_code);
    println!("Ratio: {:.2}%", result.compression_ratio * 100.0);
    
    Ok(())
}
```

## С использованием OpenAI

```rust
let provider = ProviderConfig::new(
    "openai",
    "https://api.openai.com/v1/chat/completions",
    env::var("OPENAI_API_KEY").unwrap(),
    "gpt-4",
);

let config = CompressionConfig::default()
    .with_rate(0.6)
    .with_language(CodeLanguage::TypeScript)
    .with_provider(provider);
```

## Компрессия Rust кода

```rust
let code = r#"
pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}
"#;

let config = CompressionConfig::default()
    .with_rate(0.5)
    .with_language(CodeLanguage::Rust)
    .with_provider(provider);

let compressor = LongCodeZip::new(config)?;
let result = compressor.compress_code(
    code,
    "implement fibonacci",
    "Answer the following question based on the code:"
).await?;
```

## С целевым количеством токенов

```rust
let config = CompressionConfig::default()
    .with_target_token(1000)  // Вместо rate
    .with_language(CodeLanguage::Python)
    .with_provider(provider);
```

## Настройка температуры и max_tokens

```rust
let mut provider = ProviderConfig::new(
    "deepseek",
    "https://api.deepseek.com/chat/completions",
    "your-key",
    "deepseek-chat",
);
provider.temperature = 0.7;
provider.max_tokens = 4096;
```

## Обработка результата

```rust
let result = compressor.compress_code(code, query, instruction).await?;

println!("Original tokens: {}", result.original_tokens);
println!("Compressed tokens: {}", result.compressed_tokens);
println!("Final prompt tokens: {}", result.final_compressed_tokens);
println!("Compression ratio: {:.2}%", result.compression_ratio * 100.0);

// Информация о функциях
for (idx, func) in &result.function_compressions {
    println!("Function {}: {} -> {} tokens", 
        idx, 
        func.original_tokens, 
        func.compressed_tokens
    );
}

// Какие функции были выбраны
println!("Selected functions: {:?}", result.selected_functions);
```

## Использование в библиотеке

```toml
# Cargo.toml
[dependencies]
longcodezip = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

```rust
// lib.rs или main.rs
use longcodezip::LongCodeZip;

pub async fn compress_my_code(code: &str) -> Result<String, Box<dyn std::error::Error>> {
    let config = /* ... */;
    let compressor = LongCodeZip::new(config)?;
    let result = compressor.compress_code(code, "", "").await?;
    Ok(result.compressed_code)
}
```
