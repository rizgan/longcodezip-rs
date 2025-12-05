//! Demo of caching and parallel processing features

use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig, CodeLanguage};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    // Get API key from environment
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .expect("Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable");
    
    let provider = if std::env::var("OPENAI_API_KEY").is_ok() {
        ProviderConfig::openai(&api_key, "gpt-3.5-turbo")
    } else {
        ProviderConfig::deepseek(&api_key)
    };
    
    println!("=== LongCodeZip: Caching and Parallel Processing Demo ===\n");
    
    // Sample code with multiple functions
    let code = r#"
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

def is_prime(n):
    """Check if number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def bubble_sort(arr):
    """Sort array using bubble sort."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def binary_search(arr, target):
    """Search for target in sorted array."""
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

def merge_sort(arr):
    """Sort array using merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"#;
    
    let query = "How do the sorting algorithms work?";
    
    // Test 1: Sequential processing without cache
    println!("📊 Test 1: Sequential (no parallel, no cache)");
    let config = CompressionConfig::default()
        .with_provider(provider.clone())
        .with_language(CodeLanguage::Python)
        .with_rate(0.5)
        .with_cache(false)
        .with_parallel(false);
    
    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();
    let result = compressor.compress_code(code, query, "Analyze:").await?;
    let duration1 = start.elapsed();
    
    println!("  ⏱️  Time: {:?}", duration1);
    println!("  📦 Compression: {:.1}% -> {:.1}%",
        result.original_tokens,
        result.compression_ratio * 100.0
    );
    println!("  ✅ Selected {} chunks\n", result.selected_functions.len());
    
    // Test 2: Parallel processing without cache
    println!("📊 Test 2: Parallel processing (no cache)");
    let config = CompressionConfig::default()
        .with_provider(provider.clone())
        .with_language(CodeLanguage::Python)
        .with_rate(0.5)
        .with_cache(false)
        .with_parallel(true)
        .with_parallel_threads(4);
    
    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();
    let result = compressor.compress_code(code, query, "Analyze:").await?;
    let duration2 = start.elapsed();
    
    println!("  ⏱️  Time: {:?}", duration2);
    println!("  📦 Compression: {:.1}% -> {:.1}%",
        result.original_tokens,
        result.compression_ratio * 100.0
    );
    println!("  ✅ Selected {} chunks", result.selected_functions.len());
    println!("  🚀 Speedup: {:.2}x\n", duration1.as_secs_f64() / duration2.as_secs_f64());
    
    // Test 3: First run with cache enabled
    println!("📊 Test 3: With caching (first run - cache miss)");
    let config = CompressionConfig::default()
        .with_provider(provider.clone())
        .with_language(CodeLanguage::Python)
        .with_rate(0.5)
        .with_cache(true)
        .with_parallel(true)
        .with_parallel_threads(4);
    
    let compressor = LongCodeZip::new(config)?;
    let start = Instant::now();
    let result = compressor.compress_code(code, query, "Analyze:").await?;
    let duration3 = start.elapsed();
    
    let stats = compressor.cache_stats();
    println!("  ⏱️  Time: {:?}", duration3);
    println!("  📦 Compression: {:.1}% -> {:.1}%",
        result.original_tokens,
        result.compression_ratio * 100.0
    );
    println!("  💾 Cache: {} entries\n", stats.valid_entries);
    
    // Test 4: Second run with cache (cache hit)
    println!("📊 Test 4: With caching (second run - cache hit)");
    let start = Instant::now();
    let result = compressor.compress_code(code, query, "Analyze:").await?;
    let duration4 = start.elapsed();
    
    let stats = compressor.cache_stats();
    println!("  ⏱️  Time: {:?}", duration4);
    println!("  📦 Compression: {:.1}% -> {:.1}%",
        result.original_tokens,
        result.compression_ratio * 100.0
    );
    println!("  💾 Cache: {} entries", stats.valid_entries);
    println!("  🚀 Speedup from cache: {:.2}x\n", duration3.as_secs_f64() / duration4.as_secs_f64());
    
    // Summary
    println!("=== Summary ===");
    println!("Sequential:          {:?}", duration1);
    println!("Parallel (no cache): {:?} ({:.2}x faster)", duration2, duration1.as_secs_f64() / duration2.as_secs_f64());
    println!("First with cache:    {:?}", duration3);
    println!("Cached:              {:?} ({:.2}x faster)", duration4, duration1.as_secs_f64() / duration4.as_secs_f64());
    
    println!("\n✨ Cache stats:");
    println!("  Valid entries: {}", stats.valid_entries);
    println!("  Total entries: {}", stats.total_entries);
    
    Ok(())
}
