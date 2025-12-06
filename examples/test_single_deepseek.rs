use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("Testing DeepSeek API connection...\n");
    
    let content = fs::read_to_string("test_files/user_service.ts")?;
    println!("File loaded: {} characters\n", content.len());
    
    let provider = ProviderConfig::new(
        "deepseek",
        "https://api.deepseek.com/chat/completions",
        "api-key",
        "deepseek-chat"
    );
    
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::TypeScript)
        .with_provider(provider)
        .with_cache(false)
        .with_parallel(false);
    
    println!("Configuration:");
    println!("  Provider: deepseek");
    println!("  Model: deepseek-chat");
    println!("  Compression rate: 50%");
    println!("  Cache: disabled");
    println!("  Parallel: disabled\n");
    
    let compressor = LongCodeZip::new(config)?;
    
    println!("Starting compression...\n");
    let start = std::time::Instant::now();
    
    let result = compressor.compress_code(
        &content,
        "Analyze this TypeScript code",
        "Extract the most important parts:",
    ).await?;
    
    let elapsed = start.elapsed();
    
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    Compression Results                        ");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("Original tokens:     {}", result.original_tokens);
    println!("Compressed tokens:   {}", result.compressed_tokens);
    println!("Compression ratio:   {:.2}%", result.compression_ratio * 100.0);
    println!("Time elapsed:        {:.2}s", elapsed.as_secs_f64());
    println!("\n═══════════════════════════════════════════════════════════════\n");
    
    println!("Compressed code preview (first 500 chars):");
    println!("{}", &result.compressed_code[..result.compressed_code.len().min(500)]);
    
    Ok(())
}
