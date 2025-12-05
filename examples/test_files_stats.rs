use longcodezip::{Tokenizer, TokenizerModel, ApproximateTokenizer};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
struct FileStats {
    filename: String,
    file_type: String,
    chars: usize,
    tokens_cl100k: usize,
    tokens_o200k: usize,
    chars_per_token_cl100k: f64,
    chars_per_token_o200k: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("           LongCodeZip Test Files Statistics                  ");
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

    // Initialize tokenizers
    println!("📝 Initializing tokenizers...\n");
    let start = Instant::now();
    let tokenizer_cl100k = Tokenizer::new(TokenizerModel::Cl100kBase);
    let tokenizer_o200k = Tokenizer::new(TokenizerModel::O200kBase);
    let approx_tokenizer = ApproximateTokenizer::new();
    println!("✅ Tokenizers initialized in {} ms\n", start.elapsed().as_millis());

    let mut all_stats = Vec::new();

    println!("📊 Analyzing test files...\n");
    println!("┌──────────────────────────────────────────────────────────────┐");
    println!("│ File Analysis                                                │");
    println!("├─────────────────────┬──────┬─────────┬─────────┬────────────┤");
    println!("│ Filename            │ Type │ cl100k  │ o200k   │ Chars/Tok  │");
    println!("├─────────────────────┼──────┼─────────┼─────────┼────────────┤");

    for (file_path, file_type) in &test_files {
        let content = fs::read_to_string(file_path)?;
        
        let chars = content.len();
        let tokens_cl100k = tokenizer_cl100k.count_tokens(&content)?;
        let tokens_o200k = tokenizer_o200k.count_tokens(&content)?;
        let tokens_approx = approx_tokenizer.count_tokens(&content);
        
        let chars_per_token_cl100k = chars as f64 / tokens_cl100k as f64;
        let chars_per_token_o200k = chars as f64 / tokens_o200k as f64;
        
        let filename = Path::new(file_path)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap();
        
        println!(
            "│ {:<19} │ {:<4} │ {:>7} │ {:>7} │ {:>4.2}/{:<4.2}│",
            filename,
            &file_type[..file_type.len().min(4)],
            tokens_cl100k,
            tokens_o200k,
            chars_per_token_cl100k,
            chars_per_token_o200k
        );
        
        let stats = FileStats {
            filename: filename.to_string(),
            file_type: file_type.to_string(),
            chars,
            tokens_cl100k,
            tokens_o200k,
            chars_per_token_cl100k,
            chars_per_token_o200k,
        };
        
        all_stats.push(stats);
    }

    println!("└─────────────────────┴──────┴─────────┴─────────┴────────────┘\n");

    // Calculate compression estimates at different rates
    println!("📈 Estimated Compression Results\n");
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Compression Estimates (using cl100k tokenizer)              │");
    println!("├─────────────────────┬─────────┬──────────────────────────────┤");
    println!("│ Filename            │ Original│ 30%  │ 50%  │ 70%  compression│");
    println!("├─────────────────────┼─────────┼──────┼──────┼─────────────────┤");

    for stats in &all_stats {
        let comp_30 = (stats.tokens_cl100k as f64 * 0.3) as usize;
        let comp_50 = (stats.tokens_cl100k as f64 * 0.5) as usize;
        let comp_70 = (stats.tokens_cl100k as f64 * 0.7) as usize;
        
        println!(
            "│ {:<19} │ {:>7} │ {:>4} │ {:>4} │ {:>4} tokens   │",
            stats.filename,
            stats.tokens_cl100k,
            comp_30,
            comp_50,
            comp_70
        );
    }

    println!("└─────────────────────┴─────────┴──────┴──────┴─────────────────┘\n");

    // Summary by file type
    println!("📋 Summary by File Type\n");
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Type        │ Avg Chars │ Avg Tokens │ Chars/Token │ Files │");
    println!("├─────────────┼───────────┼────────────┼─────────────┼───────┤");

    let mut type_stats: std::collections::HashMap<String, Vec<&FileStats>> =
        std::collections::HashMap::new();

    for stat in &all_stats {
        type_stats
            .entry(stat.file_type.clone())
            .or_default()
            .push(stat);
    }

    for (file_type, stats_list) in type_stats.iter() {
        let avg_chars = stats_list.iter().map(|s| s.chars).sum::<usize>() as f64 
            / stats_list.len() as f64;
        let avg_tokens = stats_list.iter().map(|s| s.tokens_cl100k).sum::<usize>() as f64
            / stats_list.len() as f64;
        let avg_ratio = avg_chars / avg_tokens;
        
        println!(
            "│ {:<11} │ {:>9.0} │ {:>10.0} │ {:>11.2} │ {:>5} │",
            file_type,
            avg_chars,
            avg_tokens,
            avg_ratio,
            stats_list.len()
        );
    }

    println!("└─────────────┴───────────┴────────────┴─────────────┴───────┘\n");

    // Overall statistics
    let total_chars: usize = all_stats.iter().map(|s| s.chars).sum();
    let total_tokens_cl100k: usize = all_stats.iter().map(|s| s.tokens_cl100k).sum();
    let total_tokens_o200k: usize = all_stats.iter().map(|s| s.tokens_o200k).sum();
    let avg_chars_per_token_cl100k = total_chars as f64 / total_tokens_cl100k as f64;
    let avg_chars_per_token_o200k = total_chars as f64 / total_tokens_o200k as f64;

    println!("📈 Overall Statistics:");
    println!("   Total characters:        {:>10}", total_chars);
    println!("   Total tokens (cl100k):   {:>10}", total_tokens_cl100k);
    println!("   Total tokens (o200k):    {:>10}", total_tokens_o200k);
    println!("   Chars/token (cl100k):    {:>10.2}", avg_chars_per_token_cl100k);
    println!("   Chars/token (o200k):     {:>10.2}", avg_chars_per_token_o200k);
    println!("   Number of files:         {:>10}", all_stats.len());

    // Compression estimates for total
    println!("\n💾 Total Compression Estimates (cl100k):");
    for rate in [0.3, 0.5, 0.7] {
        let compressed = (total_tokens_cl100k as f64 * rate) as usize;
        let saved = total_tokens_cl100k - compressed;
        let savings_pct = (1.0 - rate) * 100.0;
        
        println!(
            "   At {:>3.0}% rate: {:>6} tokens → {:>6} tokens (save {:>6} / {:>5.1}%)",
            rate * 100.0,
            total_tokens_cl100k,
            compressed,
            saved,
            savings_pct
        );
    }

    // Best and worst for compression
    let most_tokens = all_stats.iter().max_by_key(|s| s.tokens_cl100k).unwrap();
    let least_tokens = all_stats.iter().min_by_key(|s| s.tokens_cl100k).unwrap();
    
    println!("\n🏆 Largest file:");
    println!("   {} ({}) - {} tokens", most_tokens.filename, most_tokens.file_type, most_tokens.tokens_cl100k);
    
    println!("\n📦 Smallest file:");
    println!("   {} ({}) - {} tokens", least_tokens.filename, least_tokens.file_type, least_tokens.tokens_cl100k);

    // Most efficient tokenization
    let most_efficient = all_stats.iter().max_by(|a, b|
        a.chars_per_token_cl100k.partial_cmp(&b.chars_per_token_cl100k).unwrap()
    ).unwrap();
    
    let least_efficient = all_stats.iter().min_by(|a, b|
        a.chars_per_token_cl100k.partial_cmp(&b.chars_per_token_cl100k).unwrap()
    ).unwrap();
    
    println!("\n✨ Most efficient tokenization:");
    println!("   {} ({}) - {:.2} chars/token",
        most_efficient.filename, most_efficient.file_type, most_efficient.chars_per_token_cl100k);
    
    println!("\n🔍 Least efficient tokenization:");
    println!("   {} ({}) - {:.2} chars/token",
        least_efficient.filename, least_efficient.file_type, least_efficient.chars_per_token_cl100k);

    println!("\n═══════════════════════════════════════════════════════════════\n");
    
    Ok(())
}
