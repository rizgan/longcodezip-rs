//! Fine-grained compression demo
//!
//! Demonstrates the difference between coarse-grained (function-level)
//! and fine-grained (entropy-based + knapsack) compression.

use longcodezip::{CodeLanguage, CompressionConfig, LongCodeZip, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let example_code = r#"
import numpy as np
from typing import List, Dict

def calculate_mean(numbers: List[float]) -> float:
    """Calculate arithmetic mean of numbers"""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def calculate_std(numbers: List[float]) -> float:
    """Calculate standard deviation"""
    if not numbers:
        return 0.0
    mean = calculate_mean(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    return np.sqrt(variance)

def normalize_data(data: List[float]) -> List[float]:
    """Normalize data to zero mean and unit variance"""
    mean = calculate_mean(data)
    std = calculate_std(data)
    if std == 0:
        return [0.0] * len(data)
    return [(x - mean) / std for x in data]

def apply_softmax(logits: List[float]) -> List[float]:
    """Apply softmax function to logits"""
    max_logit = max(logits)
    exp_logits = [np.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

def cross_entropy_loss(predictions: List[float], targets: List[int]) -> float:
    """Calculate cross-entropy loss"""
    epsilon = 1e-10
    loss = 0.0
    for pred, target in zip(predictions, targets):
        loss -= np.log(pred[target] + epsilon)
    return loss / len(targets)
"#;

    let query = "How do I normalize my dataset?";
    let instruction = "Analyze this code and answer the question:";

    println!("=== LongCodeZip Fine-Grained Compression Demo ===\n");
    println!("Code length: {} characters\n", example_code.len());
    println!("Query: {}\n", query);

    // API configuration (using DeepSeek)
    let api_key = "api-key";
    let provider_config = ProviderConfig::deepseek(api_key);

    println!("--- Coarse-Grained Compression (Function-level) ---\n");
    
    let coarse_config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider_config.clone())
        .with_rank_only(true); // Disable fine-grained

    let compressor_coarse = LongCodeZip::new(coarse_config)?;
    let result_coarse = compressor_coarse
        .compress_code(example_code, query, instruction)
        .await?;

    println!("Compression method: {:?}", result_coarse.fine_grained_method_used);
    println!("Original tokens: {}", result_coarse.original_tokens);
    println!("Compressed tokens: {}", result_coarse.compressed_tokens);
    println!(
        "Compression ratio: {:.2}%",
        result_coarse.compression_ratio * 100.0
    );
    println!("Selected functions: {}", result_coarse.selected_functions.len());
    println!(
        "Functions selected: {:?}",
        result_coarse.selected_functions
    );
    println!("\nCompressed code preview:");
    println!("{}", "-".repeat(60));
    let preview_coarse = result_coarse
        .compressed_code
        .lines()
        .take(15)
        .collect::<Vec<_>>()
        .join("\n");
    println!("{}", preview_coarse);
    if result_coarse.compressed_code.lines().count() > 15 {
        println!("... ({} more lines)", result_coarse.compressed_code.lines().count() - 15);
    }
    println!("{}\n", "-".repeat(60));

    println!("\n--- Fine-Grained Compression (Entropy + Knapsack) ---\n");

    let fine_config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider_config);

    let compressor_fine = LongCodeZip::new(fine_config)?;
    let result_fine = compressor_fine
        .compress_code(example_code, query, instruction)
        .await?;

    println!("Compression method: {:?}", result_fine.fine_grained_method_used);
    println!("Original tokens: {}", result_fine.original_tokens);
    println!("Compressed tokens: {}", result_fine.compressed_tokens);
    println!(
        "Compression ratio: {:.2}%",
        result_fine.compression_ratio * 100.0
    );
    println!("Selected blocks: {}", result_fine.selected_functions.len());
    println!(
        "Blocks selected: {:?}",
        result_fine.selected_functions
    );
    println!("\nCompressed code preview:");
    println!("{}", "-".repeat(60));
    let preview_fine = result_fine
        .compressed_code
        .lines()
        .take(15)
        .collect::<Vec<_>>()
        .join("\n");
    println!("{}", preview_fine);
    if result_fine.compressed_code.lines().count() > 15 {
        println!("... ({} more lines)", result_fine.compressed_code.lines().count() - 15);
    }
    println!("{}\n", "-".repeat(60));

    println!("\n=== Comparison ===\n");
    println!("Coarse-grained:");
    println!("  - Tokens: {} ({:.1}% of original)", 
        result_coarse.compressed_tokens,
        result_coarse.compression_ratio * 100.0
    );
    println!("  - Selected chunks: {}", result_coarse.selected_functions.len());
    
    println!("\nFine-grained:");
    println!("  - Tokens: {} ({:.1}% of original)", 
        result_fine.compressed_tokens,
        result_fine.compression_ratio * 100.0
    );
    println!("  - Selected chunks: {}", result_fine.selected_functions.len());
    
    let improvement = (result_coarse.compression_ratio - result_fine.compression_ratio).abs();
    println!("\nDifference: {:.2} percentage points", improvement * 100.0);

    if result_fine.compression_ratio < result_coarse.compression_ratio {
        println!("✅ Fine-grained achieved better compression!");
    } else if result_fine.compression_ratio > result_coarse.compression_ratio {
        println!("ℹ️  Coarse-grained was more aggressive");
    } else {
        println!("ℹ️  Both methods achieved similar compression");
    }

    Ok(())
}
