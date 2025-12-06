//! Demo example for LongCodeZip
//! 
//! This example demonstrates how to use the LongCodeZip library
//! to compress code with a query using the DeepSeek API.

use longcodezip::{CodeLanguage, CompressionConfig, LongCodeZip, ProviderConfig};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Example code to compress
    let code = r#"
def add(a, b):
    """Add two numbers together."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y

def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def quick_sort(arr):
    """Quick sort algorithm implementation."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def binary_search(arr, target):
    """Binary search algorithm."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def calculate(self, a, b, operation):
        """Perform calculation and store in history."""
        result = None
        if operation == 'add':
            result = add(a, b)
        elif operation == 'subtract':
            result = subtract(a, b)
        elif operation == 'multiply':
            result = multiply(a, b)
        elif operation == 'divide':
            result = divide(a, b)
        
        self.history.append((a, b, operation, result))
        return result
"#;

    // Query to focus compression
    let query = "How to implement a quick sort algorithm?";
    
    let instruction = "Given the following code context, answer the question.";
    
    // Get API key from environment variable
    let api_key = env::var("DEEPSEEK_API_KEY")
        .unwrap_or_else(|_| "api-key".to_string());
    
    // Configure the provider (DeepSeek) - now with accurate tokenizer
    let provider = ProviderConfig::deepseek(&api_key);
    
    // Configure compression
    let config = CompressionConfig::default()
        .with_rate(0.5) // Keep 50% of tokens
        .with_language(CodeLanguage::Python)
        .with_provider(provider)
        .with_rank_only(true); // Use only coarse-grained compression
    
    println!("🚀 LongCodeZip Demo");
    println!("==================\n");
    
    println!("Original code length: {} characters", code.len());
    println!("Query: {}\n", query);
    
    // Create compressor
    let compressor = LongCodeZip::new(config)?;
    
    // Compress the code
    println!("⏳ Compressing code...");
    let result = compressor.compress_code(code, query, instruction).await?;
    
    println!("✅ Compression complete!\n");
    
    // Display results
    println!("📊 Statistics:");
    println!("  Original tokens:    {}", result.original_tokens);
    println!("  Compressed tokens:  {}", result.compressed_tokens);
    println!("  Final tokens:       {}", result.final_compressed_tokens);
    println!("  Compression ratio:  {:.2}%", result.compression_ratio * 100.0);
    println!("  Selected functions: {}", result.selected_functions.len());
    println!();
    
    println!("📝 Compressed Code:");
    println!("{}", "=".repeat(80));
    println!("{}", result.compressed_code);
    println!("{}", "=".repeat(80));
    println!();
    
    println!("💬 Full Prompt:");
    println!("{}", "=".repeat(80));
    println!("{}", result.compressed_prompt);
    println!("{}", "=".repeat(80));
    println!();
    
    println!("🎯 Function Compressions:");
    for (idx, func_comp) in &result.function_compressions {
        println!(
            "  Function {}: {} -> {} tokens (ratio: {:.2})",
            idx,
            func_comp.original_tokens,
            func_comp.compressed_tokens,
            func_comp.compression_ratio
        );
        if let Some(note) = &func_comp.note {
            println!("    Note: {}", note);
        }
    }
    
    Ok(())
}
