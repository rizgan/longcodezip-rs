use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, TextChunkingStrategy};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct CompressionStats {
    filename: String,
    file_type: String,
    original_tokens: usize,
    compressed_tokens: usize,
    compression_ratio: f64,
    time_ms: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_benchmarks())
}

async fn run_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("           LongCodeZip Compression Benchmark Test             ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let test_files = vec![
        ("test_files/web_scraper.py", "Python"),
        ("test_files/server_utils.rs", "Rust"),
        ("test_files/user_service.ts", "TypeScript"),
        ("test_files/ml_pipeline_doc.md", "Markdown"),
        ("test_files/task_queue.go", "Go"),
        ("test_files/ShoppingCart.java", "Java"),
        ("test_files/dashboard.html", "HTML"),
        ("test_files/api_reference.txt", "Text"),
        ("test_files/openapi_spec.json", "JSON"),
        ("test_files/analytics_query.sql", "SQL"),
    ];

    let compression_ratios = vec![0.3, 0.5, 0.7];
    let mut all_stats: Vec<CompressionStats> = Vec::new();

    // Test 1: Code compression with different ratios
    println!("📊 Test 1: Code compression with different ratios");
    println!("─────────────────────────────────────────────────────────────\n");

    for &ratio in &compression_ratios {
        println!("Target compression rate: {:.1}%\n", ratio * 100.0);

        for (file_path, file_type) in &test_files {
            if is_code_file(file_type) {
                match test_code_compression(file_path, file_type, ratio).await {
                    Ok(stats) => all_stats.push(stats),
                    Err(e) => eprintln!("Error processing {}: {}", file_path, e),
                }
            }
        }
        println!();
    }

    // Test 2: Text compression with different strategies
    println!("\n📝 Test 2: Text compression with different chunking strategies");
    println!("─────────────────────────────────────────────────────────────\n");

    let text_strategies = vec![
        ("Paragraphs", TextChunkingStrategy::Paragraphs),
        ("Sentences", TextChunkingStrategy::Sentences),
        ("MarkdownSections", TextChunkingStrategy::MarkdownSections),
    ];

    for (strategy_name, strategy) in &text_strategies {
        println!("Strategy: {}\n", strategy_name);

        for (file_path, file_type) in &test_files {
            if is_text_file(file_type) {
                match test_text_compression(file_path, file_type, 0.5, *strategy).await {
                    Ok(stats) => all_stats.push(stats),
                    Err(e) => eprintln!("Error processing {}: {}", file_path, e),
                }
            }
        }
        println!();
    }

    // Generate summary
    generate_summary(&all_stats);

    Ok(())
}

async fn test_code_compression(
    file_path: &str,
    file_type: &str,
    ratio: f64,
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
        .with_rate(ratio)
        .with_language(language);

    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();

    let result = compressor.compress_code(
        &content,
        "Analyze this code",
        "Extract important parts:",
    ).await?;

    let elapsed = start.elapsed().as_millis();
    let actual_ratio = result.compressed_tokens as f64 / result.original_tokens as f64;

    let stats = CompressionStats {
        filename: get_filename(file_path),
        file_type: file_type.to_string(),
        original_tokens: result.original_tokens,
        compressed_tokens: result.compressed_tokens,
        compression_ratio: actual_ratio,
        time_ms: elapsed,
    };

    print_stats(&stats);
    Ok(stats)
}

async fn test_text_compression(
    file_path: &str,
    file_type: &str,
    ratio: f64,
    strategy: TextChunkingStrategy,
) -> Result<CompressionStats, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;

    let config = CompressionConfig::default()
        .with_rate(ratio);

    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();

    let result = compressor.compress_text(
        &content,
        "Summarize key points",
        "Extract main information:",
        strategy,
    ).await?;

    let elapsed = start.elapsed().as_millis();
    let actual_ratio = result.compressed_tokens as f64 / result.original_tokens as f64;

    let stats = CompressionStats {
        filename: get_filename(file_path),
        file_type: file_type.to_string(),
        original_tokens: result.original_tokens,
        compressed_tokens: result.compressed_tokens,
        compression_ratio: actual_ratio,
        time_ms: elapsed,
    };

    print_stats(&stats);
    Ok(stats)
}

fn print_stats(stats: &CompressionStats) {
    println!("  📄 {} ({}):", stats.filename, stats.file_type);
    println!("     Original tokens:   {:>6}", stats.original_tokens);
    println!("     Compressed tokens: {:>6}", stats.compressed_tokens);
    println!("     Compression ratio: {:>6.2}%", stats.compression_ratio * 100.0);
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

fn is_code_file(file_type: &str) -> bool {
    matches!(file_type, "Python" | "Rust" | "TypeScript" | "Go" | "Java" | "HTML" | "SQL")
}

fn is_text_file(file_type: &str) -> bool {
    matches!(file_type, "Markdown" | "Text" | "JSON")
}

fn generate_summary(stats: &[CompressionStats]) {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                      Summary Report                          ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut by_type: std::collections::HashMap<String, Vec<&CompressionStats>> =
        std::collections::HashMap::new();

    for stat in stats {
        by_type.entry(stat.file_type.clone()).or_default().push(stat);
    }

    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Performance by File Type                                    │");
    println!("├─────────────────┬──────────┬──────────┬──────────┬─────────┤");
    println!("│ File Type       │ Avg Size │ Avg Comp │ Avg Ratio│ Avg Time│");
    println!("├─────────────────┼──────────┼──────────┼──────────┼─────────┤");

    for (file_type, type_stats) in by_type.iter() {
        let avg_original = type_stats.iter().map(|s| s.original_tokens).sum::<usize>() as f64
            / type_stats.len() as f64;
        let avg_compressed = type_stats.iter().map(|s| s.compressed_tokens).sum::<usize>() as f64
            / type_stats.len() as f64;
        let avg_ratio = type_stats.iter().map(|s| s.compression_ratio).sum::<f64>()
            / type_stats.len() as f64;
        let avg_time = type_stats.iter().map(|s| s.time_ms).sum::<u128>() as f64
            / type_stats.len() as f64;

        println!(
            "│ {:<15} │ {:>8} │ {:>8} │ {:>7.1}% │ {:>6.0}ms│",
            file_type,
            avg_original as usize,
            avg_compressed as usize,
            avg_ratio * 100.0,
            avg_time
        );
    }

    println!("└─────────────────┴──────────┴──────────┴──────────┴─────────┘\n");

    let total_original: usize = stats.iter().map(|s| s.original_tokens).sum();
    let total_compressed: usize = stats.iter().map(|s| s.compressed_tokens).sum();
    let overall_ratio = total_compressed as f64 / total_original as f64;
    let total_time: u128 = stats.iter().map(|s| s.time_ms).sum();

    println!("📈 Overall Statistics:");
    println!("   Total original tokens:   {:>10}", total_original);
    println!("   Total compressed tokens: {:>10}", total_compressed);
    println!("   Overall compression:     {:>9.2}%", overall_ratio * 100.0);
    println!("   Total processing time:   {:>10} ms", total_time);
    println!("   Number of tests:         {:>10}", stats.len());

    if !stats.is_empty() {
        let best = stats.iter().min_by(|a, b| 
            a.compression_ratio.partial_cmp(&b.compression_ratio).unwrap()
        ).unwrap();

        let worst = stats.iter().max_by(|a, b|
            a.compression_ratio.partial_cmp(&b.compression_ratio).unwrap()
        ).unwrap();

        println!("\n🏆 Best compression:");
        println!("   {} ({}) - {:.2}% compression ratio",
            best.filename, best.file_type, best.compression_ratio * 100.0);

        println!("\n⚠️  Least compression:");
        println!("   {} ({}) - {:.2}% compression ratio",
            worst.filename, worst.file_type, worst.compression_ratio * 100.0);
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");
}
