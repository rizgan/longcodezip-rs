use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, TextChunkingStrategy, ProviderConfig};
use std::fs;
use std::path::Path;
use std::time::Instant;
use std::env;

#[derive(Debug)]
struct CompressionStats {
    filename: String,
    file_type: String,
    provider: String,
    compression_rate: f64,
    original_tokens: usize,
    compressed_tokens: usize,
    actual_ratio: f64,
    time_ms: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_benchmarks())
}

async fn run_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("        LongCodeZip Real LLM Compression Benchmark            ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Configure providers
    let providers = vec![
        (
            "DeepSeek",
            ProviderConfig::new(
                "deepseek",
                "https://api.deepseek.com/chat/completions",
                "sk-b78ab15d637749a9a8c6ae69a919c0a9",
                "deepseek-chat"
            )
        ),
        (
            "Qwen",
            ProviderConfig::new(
                "alibabacloud",
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
                "sk-690a6bc4610c444f9817a163978cb676",
                "qwen-max"
            )
        ),
    ];

    let test_files = vec![
        ("test_files/web_scraper.py", "Python"),
        ("test_files/server_utils.rs", "Rust"),
        ("test_files/user_service.ts", "TypeScript"),
        ("test_files/task_queue.go", "Go"),
        ("test_files/ShoppingCart.java", "Java"),
    ];

    let compression_rates = vec![0.3, 0.5];
    let mut all_stats: Vec<CompressionStats> = Vec::new();

    // Test each provider
    for (provider_name, provider_config) in &providers {
        println!("🤖 Testing with {} LLM", provider_name);
        println!("─────────────────────────────────────────────────────────────\n");

        for &rate in &compression_rates {
            println!("Target compression rate: {:.0}%\n", rate * 100.0);

            for (file_path, file_type) in &test_files {
                match test_compression(
                    file_path,
                    file_type,
                    rate,
                    provider_name,
                    provider_config.clone()
                ).await {
                    Ok(stats) => {
                        print_stats(&stats);
                        all_stats.push(stats);
                    }
                    Err(e) => eprintln!("❌ Error processing {}: {}\n", file_path, e),
                }
            }
            println!();
        }
    }

    // Generate comparison report
    generate_comparison_report(&all_stats);

    Ok(())
}

async fn test_compression(
    file_path: &str,
    file_type: &str,
    rate: f64,
    provider_name: &str,
    provider_config: ProviderConfig,
) -> Result<CompressionStats, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;

    let language = match file_type {
        "Python" => CodeLanguage::Python,
        "Rust" => CodeLanguage::Rust,
        "TypeScript" => CodeLanguage::TypeScript,
        "Go" => CodeLanguage::Go,
        "Java" => CodeLanguage::Java,
        _ => CodeLanguage::Python,
    };

    let config = CompressionConfig::default()
        .with_rate(rate)
        .with_language(language)
        .with_provider(provider_config)
        .with_cache(false); // Disable cache to test real LLM performance

    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();

    let result = compressor.compress_code(
        &content,
        "Analyze and compress this code",
        "Extract the most important parts:",
    ).await?;

    let elapsed = start.elapsed().as_millis();
    let actual_ratio = result.compressed_tokens as f64 / result.original_tokens as f64;

    let stats = CompressionStats {
        filename: get_filename(file_path),
        file_type: file_type.to_string(),
        provider: provider_name.to_string(),
        compression_rate: rate,
        original_tokens: result.original_tokens,
        compressed_tokens: result.compressed_tokens,
        actual_ratio,
        time_ms: elapsed,
    };

    Ok(stats)
}

fn print_stats(stats: &CompressionStats) {
    println!("  📄 {} ({}):", stats.filename, stats.file_type);
    println!("     Original tokens:   {:>6}", stats.original_tokens);
    println!("     Compressed tokens: {:>6}", stats.compressed_tokens);
    println!("     Target rate:       {:>6.0}%", stats.compression_rate * 100.0);
    println!("     Actual ratio:      {:>6.2}%", stats.actual_ratio * 100.0);
    println!("     Time:              {:>6} ms", stats.time_ms);
    println!();
}

fn get_filename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

fn generate_comparison_report(stats: &[CompressionStats]) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                   Comparison Report                          ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Group by provider
    let mut by_provider: std::collections::HashMap<String, Vec<&CompressionStats>> =
        std::collections::HashMap::new();

    for stat in stats {
        by_provider.entry(stat.provider.clone()).or_default().push(stat);
    }

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Provider Performance Comparison                             │");
    println!("├──────────┬──────────┬──────────┬───────────┬────────────────┤");
    println!("│ Provider │ Avg Orig │ Avg Comp │ Avg Ratio │ Avg Time       │");
    println!("├──────────┼──────────┼──────────┼───────────┼────────────────┤");

    for (provider, provider_stats) in by_provider.iter() {
        let avg_original = provider_stats.iter().map(|s| s.original_tokens).sum::<usize>() as f64
            / provider_stats.len() as f64;
        let avg_compressed = provider_stats.iter().map(|s| s.compressed_tokens).sum::<usize>() as f64
            / provider_stats.len() as f64;
        let avg_ratio = provider_stats.iter().map(|s| s.actual_ratio).sum::<f64>()
            / provider_stats.len() as f64;
        let avg_time = provider_stats.iter().map(|s| s.time_ms).sum::<u128>() as f64
            / provider_stats.len() as f64;

        println!(
            "│ {:<8} │ {:>8.0} │ {:>8.0} │ {:>8.2}% │ {:>10.0} ms │",
            provider,
            avg_original,
            avg_compressed,
            avg_ratio * 100.0,
            avg_time
        );
    }

    println!("└──────────┴──────────┴──────────┴───────────┴────────────────┘\n");

    // Detailed comparison by file and rate
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Detailed Results by File                                    │");
    println!("├────────────────────┬──────────┬──────┬────────┬─────────────┤");
    println!("│ File               │ Provider │ Rate │ Actual │ Time (ms)   │");
    println!("├────────────────────┼──────────┼──────┼────────┼─────────────┤");

    for stat in stats {
        println!(
            "│ {:<18} │ {:<8} │ {:>3.0}% │ {:>5.1}% │ {:>11} │",
            &stat.filename[..stat.filename.len().min(18)],
            &stat.provider[..stat.provider.len().min(8)],
            stat.compression_rate * 100.0,
            stat.actual_ratio * 100.0,
            stat.time_ms
        );
    }

    println!("└────────────────────┴──────────┴──────┴────────┴─────────────┘\n");

    // Overall statistics
    let total_original: usize = stats.iter().map(|s| s.original_tokens).sum();
    let total_compressed: usize = stats.iter().map(|s| s.compressed_tokens).sum();
    let overall_ratio = total_compressed as f64 / total_original as f64;
    let total_time: u128 = stats.iter().map(|s| s.time_ms).sum();
    let avg_time = total_time as f64 / stats.len() as f64;

    println!("📈 Overall Statistics:");
    println!("   Total original tokens:   {:>10}", total_original);
    println!("   Total compressed tokens: {:>10}", total_compressed);
    println!("   Overall compression:     {:>9.2}%", overall_ratio * 100.0);
    println!("   Total processing time:   {:>10} ms", total_time);
    println!("   Average time per file:   {:>10.0} ms", avg_time);
    println!("   Number of tests:         {:>10}", stats.len());

    // Best performer
    if !stats.is_empty() {
        let best_compression = stats.iter().min_by(|a, b| 
            a.actual_ratio.partial_cmp(&b.actual_ratio).unwrap()
        ).unwrap();

        let fastest = stats.iter().min_by_key(|s| s.time_ms).unwrap();

        println!("\n🏆 Best compression:");
        println!("   {} with {} - {:.2}% ratio (target {:.0}%)",
            best_compression.filename,
            best_compression.provider,
            best_compression.actual_ratio * 100.0,
            best_compression.compression_rate * 100.0
        );

        println!("\n⚡ Fastest processing:");
        println!("   {} with {} - {} ms",
            fastest.filename,
            fastest.provider,
            fastest.time_ms
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");
}
