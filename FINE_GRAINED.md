# Fine-Grained Compression Guide

This guide explains the fine-grained compression features in LongCodeZip v0.2.0.

## Overview

LongCodeZip now supports two compression modes:

1. **Coarse-Grained** (v0.1.0): Function-level ranking and selection
2. **Fine-Grained** (v0.2.0): Entropy-based chunking + Knapsack optimization

## Table of Contents

- [What is Fine-Grained Compression?](#what-is-fine-grained-compression)
- [Key Components](#key-components)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Performance](#performance)
- [Best Practices](#best-practices)

## What is Fine-Grained Compression?

Fine-grained compression improves upon coarse-grained by:

- **Entropy Chunking**: Splits code at topic boundaries (high perplexity points) instead of rigid function boundaries
- **Knapsack Optimization**: Uses dynamic programming to select optimal blocks within token budget
- **Better Context**: Preserves more semantically relevant code segments

### Comparison

```rust
// Coarse-Grained
// - Splits by functions (def, fn, class)
// - Greedy selection by relevance ranking
// - May miss important helper code

// Fine-Grained
// - Splits by semantic boundaries (entropy spikes)
// - Optimal selection via knapsack DP
// - Includes context needed for understanding
```

## Key Components

### 1. Entropy Chunker

Detects topic shifts using heuristic perplexity approximation:

```rust
use longcodezip::entropy::{EntropyChunker, ThresholdMethod};

let chunker = EntropyChunker::new();
let chunks = chunker.chunk_text(code)?;

for chunk in chunks {
    println!("Lines {}-{}: {:.2} perplexity", 
        chunk.start_line, 
        chunk.end_line,
        chunk.perplexity
    );
}
```

**Heuristics used:**
- Indentation changes → higher perplexity
- Empty lines → very high perplexity (topic boundary)
- Function/class keywords → high perplexity
- Line length variance → moderate perplexity

**Threshold methods:**
- `Std`: Mean + k×σ (standard deviation)
- `RobustStd`: Median + k×MAD (robust to outliers)
- `Iqr`: Q3 + k×IQR (interquartile range)
- `Mad`: Median + k×MAD (median absolute deviation)

### 2. Knapsack Optimizer

Selects blocks to maximize importance within token budget:

```rust
use longcodezip::optimizer::{KnapsackOptimizer, Block};
use std::collections::HashSet;

let optimizer = KnapsackOptimizer::new();

let blocks = vec![
    Block { index: 0, text: "fn a() {}".into(), tokens: 10, importance: 5.0 },
    Block { index: 1, text: "fn b() {}".into(), tokens: 20, importance: 15.0 },
];

let preserved = HashSet::new();
let result = optimizer.select_blocks(&blocks, 30, &preserved)?;

println!("Selected: {:?}", result.selected_indices);
println!("Total value: {:.2}", result.total_value);
println!("Efficiency: {:.2}", result.efficiency);
```

**Algorithms:**
- **Dynamic Programming**: Exact solution for ≤100 items, ≤2000 tokens
- **Greedy Approximation**: Fast heuristic for larger problems

## Usage

### Basic Example

```rust
use longcodezip::{LongCodeZip, CompressionConfig, CodeLanguage, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(ProviderConfig::deepseek("your-api-key"));
    
    let compressor = LongCodeZip::new(config)?;
    
    let result = compressor.compress_code(
        code,
        "How does normalization work?",
        "Analyze this code:"
    ).await?;
    
    println!("Method: {:?}", result.fine_grained_method_used);
    println!("Ratio: {:.1}%", result.compression_ratio * 100.0);
    
    Ok(())
}
```

### Switching Between Modes

```rust
// Coarse-Grained (Function-level)
let coarse_config = CompressionConfig::default()
    .with_rank_only(true);  // Disable fine-grained

// Fine-Grained (Entropy + Knapsack)
let fine_config = CompressionConfig::default();
// use_knapsack=true by default
```

## Configuration

### CompressionConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_knapsack` | `bool` | `true` | Enable knapsack optimization |
| `rank_only` | `bool` | `false` | Disable fine-grained (coarse mode) |
| `importance_beta` | `f64` | `0.5` | Weight for importance scoring |
| `min_lines_for_fine_grained` | `usize` | `5` | Minimum lines to apply fine-grained |

### Entropy Chunker Configuration

```rust
use longcodezip::entropy::{EntropyChunker, ThresholdMethod};

// Default: Std method, k=0.2
let chunker = EntropyChunker::new();

// Custom configuration
let chunker = EntropyChunker::with_config(
    ThresholdMethod::RobustStd,  // More robust to outliers
    0.3                          // Higher k = fewer chunks
);
```

### Optimizer Configuration

```rust
use longcodezip::optimizer::KnapsackOptimizer;

// Default: max 100 items, 2000 capacity for DP
let optimizer = KnapsackOptimizer::new();

// Custom limits (force greedy for all)
let optimizer = KnapsackOptimizer::with_limits(0, 0);
```

## Examples

### Example 1: Compare Modes

```bash
cargo run --example fine_grained_demo
```

Output:
```
Coarse-grained:
  - Tokens: 69 (21.3% of original)
  - Selected chunks: 1

Fine-grained:
  - Tokens: 134 (41.4% of original)
  - Selected chunks: 2
```

### Example 2: Custom Entropy Threshold

```rust
let chunker = EntropyChunker::with_config(
    ThresholdMethod::Iqr,  // Use IQR method
    0.15                   // Lower k = more chunks
);

let chunks = chunker.chunk_text(code)?;
println!("Created {} chunks", chunks.len());
```

### Example 3: Preserved Blocks

```rust
use std::collections::HashSet;

let mut preserved = HashSet::new();
preserved.insert(0);  // Always include first block

let result = optimizer.select_blocks(&blocks, 500, &preserved)?;
// Block 0 guaranteed to be in result.selected_indices
```

## Performance

### Computational Complexity

| Component | Time | Space |
|-----------|------|-------|
| Entropy Chunking | O(n) | O(n) |
| Knapsack DP | O(n×W) | O(n×W) |
| Knapsack Greedy | O(n log n) | O(1) |

Where:
- n = number of items
- W = token budget capacity

### Benchmarks

On typical code files (500-2000 tokens):

```
Coarse-grained: ~50ms
Fine-grained:   ~150ms (with DP)
                ~80ms (greedy fallback)
```

### Memory Usage

```
Entropy chunker: ~1KB per 100 lines
Knapsack DP:     ~8KB per 1000 items×tokens
Greedy:          Minimal (<1KB)
```

## Best Practices

### 1. Choose the Right Mode

**Use Coarse-Grained when:**
- Code is well-structured with clear functions
- Speed is critical
- Functions align with topics

**Use Fine-Grained when:**
- Code has mixed concerns within functions
- Need optimal token utilization
- Working with procedural code

### 2. Tune Entropy Threshold

```rust
// More chunks (fine-grained splitting)
let chunker = EntropyChunker::with_config(ThresholdMethod::Std, 0.1);

// Fewer chunks (coarser splitting)
let chunker = EntropyChunker::with_config(ThresholdMethod::Std, 0.5);
```

### 3. Set Appropriate Rate

```rust
// Aggressive compression (may lose context)
config.with_rate(0.3);

// Conservative compression (keeps more context)
config.with_rate(0.7);
```

### 4. Monitor Results

```rust
let result = compressor.compress_code(code, query, instruction).await?;

println!("Method: {:?}", result.fine_grained_method_used);
println!("Original: {} tokens", result.original_tokens);
println!("Compressed: {} tokens", result.compressed_tokens);
println!("Ratio: {:.1}%", result.compression_ratio * 100.0);
println!("Chunks: {}", result.selected_functions.len());
```

## Advanced Topics

### Fallback Behavior

Fine-grained automatically falls back to coarse-grained when:

1. Entropy chunking produces <2 chunks
2. Knapsack returns empty selection
3. Code is very short (<5 lines)

```rust
// Automatic fallback in compressor.rs
if entropy_chunks.len() < 2 {
    info!("Falling back to function splitting");
    chunks = split_code_by_functions(code, language)?;
}
```

### Custom Importance Scoring

Extend the `LLMProvider` trait to implement custom relevance scoring:

```rust
async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64> {
    // Custom scoring logic
    // Return 0.0-10.0 importance score
}
```

## Troubleshooting

### Issue: Too Many/Few Chunks

**Solution**: Adjust entropy threshold k-factor

```rust
// Fewer chunks (increase k)
EntropyChunker::with_config(ThresholdMethod::Std, 0.3);

// More chunks (decrease k)
EntropyChunker::with_config(ThresholdMethod::Std, 0.1);
```

### Issue: Suboptimal Selection

**Solution**: Check importance scores

```rust
let importances = compressor.calculate_chunk_importances(&chunks, query).await?;
for (i, score) in importances.iter().enumerate() {
    println!("Chunk {}: importance={:.2}", i, score);
}
```

### Issue: Slow Performance

**Solution**: Force greedy mode for large inputs

```rust
let optimizer = KnapsackOptimizer::with_limits(0, 0);  // Always greedy
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md) - Accurate token counting
- [EXAMPLES.md](EXAMPLES.md) - More usage examples
- [API Docs](https://docs.rs/longcodezip) - Full API reference
