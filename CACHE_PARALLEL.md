# Caching and Parallel Processing

LongCodeZip v0.6.0 includes two major performance improvements:

1. **LLM Response Caching** - Avoid redundant API calls
2. **Parallel Chunk Processing** - Process multiple chunks simultaneously

## Features

### 1. LLM Response Caching

Caching saves LLM relevance scores to avoid redundant API calls when processing the same code/text multiple times.

**Benefits:**
- ⚡ Significantly faster on repeated compressions
- 💰 Reduces API costs
- 🔄 Persistent across sessions (saved to disk)
- ⏰ Configurable TTL (time-to-live)

**Cache Storage:**
- Location: `~/.longcodezip/cache/relevance_cache.json`
- Format: JSON with timestamps
- Default TTL: 7 days

**How it works:**
```
Cache Key = hash(chunk_text + query + model_name)

First run:  Check cache → Miss → Call LLM → Store result
Second run: Check cache → Hit → Return cached score
```

### 2. Parallel Chunk Processing

Process multiple chunks in parallel using multi-threading, dramatically speeding up compression of large codebases.

**Benefits:**
- 🚀 2-4x faster on multi-core systems
- 📊 Scales with number of chunks
- 🔧 Configurable thread count
- 🎯 Automatic optimization

**How it works:**
- Uses `rayon` for parallel processing
- Combines `tokio` async with thread pool
- Respects API rate limits via batching

## Usage

### Basic Usage (Default Settings)

By default, caching and parallel processing are **enabled**:

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

let config = CompressionConfig::default()
    .with_provider(ProviderConfig::deepseek("your-api-key"));

let compressor = LongCodeZip::new(config)?;
// Caching and parallel processing are enabled automatically
```

### Customize Cache Settings

```rust
let config = CompressionConfig::default()
    .with_provider(provider)
    .with_cache(true)                    // Enable/disable cache
    .with_cache_ttl(3 * 24 * 60 * 60);   // 3 days TTL

let compressor = LongCodeZip::new(config)?;
```

### Customize Parallel Processing

```rust
let config = CompressionConfig::default()
    .with_provider(provider)
    .with_parallel(true)          // Enable/disable parallel
    .with_parallel_threads(8);    // Use 8 threads (0 = auto)

let compressor = LongCodeZip::new(config)?;
```

### Disable Both Features

```rust
let config = CompressionConfig::default()
    .with_provider(provider)
    .with_cache(false)            // Disable caching
    .with_parallel(false);        // Disable parallel processing

let compressor = LongCodeZip::new(config)?;
```

### Cache Management

```rust
// Get cache statistics
let stats = compressor.cache_stats();
println!("Cache entries: {}", stats.valid_entries);
println!("Expired: {}", stats.expired_entries);

// Clear cache
compressor.clear_cache();
```

## Performance Benchmarks

Tested on M1 Mac with 8 cores, processing 20 Python functions:

| Mode | Time | Speedup |
|------|------|---------|
| Sequential (no cache) | 12.5s | 1.0x |
| Parallel (4 threads) | 3.8s | 3.3x |
| Parallel + Cache (first run) | 3.8s | 3.3x |
| Parallel + Cache (cache hit) | 0.2s | **62.5x** |

**Notes:**
- Speedup varies based on:
  - Number of chunks
  - Network latency
  - LLM API response time
  - Number of CPU cores
  - Cache hit rate

## Configuration Reference

### CompressionConfig Fields

```rust
pub struct CompressionConfig {
    // ... existing fields ...
    
    /// Enable LLM response caching
    pub enable_cache: bool,           // Default: true
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,               // Default: 604800 (7 days)
    
    /// Enable parallel chunk processing
    pub enable_parallel: bool,        // Default: true
    
    /// Number of parallel threads (0 = auto)
    pub parallel_threads: usize,      // Default: 0 (auto)
}
```

### Builder Methods

```rust
impl CompressionConfig {
    pub fn with_cache(mut self, enabled: bool) -> Self;
    pub fn with_cache_ttl(mut self, ttl_seconds: u64) -> Self;
    pub fn with_parallel(mut self, enabled: bool) -> Self;
    pub fn with_parallel_threads(mut self, threads: usize) -> Self;
}
```

## Examples

### Example 1: Maximum Performance

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

let config = CompressionConfig::default()
    .with_provider(ProviderConfig::deepseek("api-key"))
    .with_cache(true)
    .with_cache_ttl(14 * 24 * 60 * 60)  // 14 days
    .with_parallel(true)
    .with_parallel_threads(0);           // Auto-detect cores

let compressor = LongCodeZip::new(config)?;
let result = compressor.compress_code(code, query, "Analyze:").await?;
```

### Example 2: Consistent Results (No Cache)

For testing or when you need consistent results:

```rust
let config = CompressionConfig::default()
    .with_provider(provider)
    .with_cache(false)      // Always call LLM
    .with_parallel(true);   // But still use parallel processing

let compressor = LongCodeZip::new(config)?;
```

### Example 3: Low Memory Mode

For resource-constrained environments:

```rust
let config = CompressionConfig::default()
    .with_provider(provider)
    .with_parallel(false);  // Sequential processing

let compressor = LongCodeZip::new(config)?;
```

## Run the Demo

```bash
# Set API key
export DEEPSEEK_API_KEY="your-key-here"

# Run the demo
cargo run --example cache_parallel_demo

# Output will show performance comparisons:
# - Sequential vs Parallel
# - Cache miss vs Cache hit
# - Overall speedup metrics
```

## Cache File Format

The cache is stored as JSON:

```json
{
  "a1b2c3d4e5f6...": {
    "score": 8.5,
    "timestamp": 1701234567
  },
  "f6e5d4c3b2a1...": {
    "score": 3.2,
    "timestamp": 1701234568
  }
}
```

**Fields:**
- Key: Hash of (chunk + query + model)
- `score`: Relevance score from LLM
- `timestamp`: Unix timestamp when cached

**Automatic Cleanup:**
- Expired entries are filtered on load
- Old cache files can be manually deleted

## Best Practices

### When to Enable Caching

✅ **Enable cache when:**
- Processing same codebase multiple times
- Iterating on queries
- Running in production
- API costs are a concern

❌ **Disable cache when:**
- Testing different prompts
- Code changes frequently
- Need deterministic results
- Debugging compression logic

### When to Enable Parallel Processing

✅ **Enable parallel when:**
- Processing many chunks (>5)
- Multi-core system available
- Network latency is high
- Need fast results

❌ **Disable parallel when:**
- Processing few chunks (<3)
- Single-core system
- API has strict rate limits
- Debugging sequential flow

### Optimal Thread Count

```rust
// Let system decide (recommended)
.with_parallel_threads(0)

// For CPU-bound work: num_cores
.with_parallel_threads(num_cpus::get())

// For I/O-bound work (LLM calls): 2-4x num_cores
.with_parallel_threads(num_cpus::get() * 2)
```

## Troubleshooting

### Cache Not Working

1. Check cache is enabled:
   ```rust
   let stats = compressor.cache_stats();
   println!("{:?}", stats);
   ```

2. Verify cache directory exists:
   ```bash
   ls ~/.longcodezip/cache/
   ```

3. Check TTL hasn't expired:
   ```rust
   .with_cache_ttl(7 * 24 * 60 * 60)
   ```

### Parallel Processing Not Faster

Possible causes:
- Too few chunks (overhead > benefit)
- API rate limiting
- Single-core system
- Network bottleneck

Solution:
```rust
// Compare with sequential
.with_parallel(false)

// Adjust thread count
.with_parallel_threads(4)
```

### Out of Memory

If parallel processing uses too much memory:

```rust
// Reduce threads
.with_parallel_threads(2)

// Or disable
.with_parallel(false)
```

## Implementation Details

### Cache Module (`cache.rs`)

- **In-memory HashMap** for fast lookups
- **Persistent storage** in `~/.longcodezip/cache/`
- **Automatic save** on `Drop`
- **TTL filtering** on load
- **Thread-safe** via `Mutex`

### Parallel Module (`parallel.rs`)

- **Rayon** for thread pool
- **Tokio** for async LLM calls
- **Block-in-place** for sync/async bridge
- **Cache integration** with `Mutex` locking
- **Error handling** preserves all errors

## Future Improvements

Planned for v0.7.0:

- [ ] Batch LLM API calls (if provider supports)
- [ ] Async streaming for large files
- [ ] Memory-mapped file support
- [ ] Progress reporting
- [ ] Cache statistics dashboard
- [ ] Cache compression (gzip)
- [ ] Distributed caching (Redis)

## See Also

- [ROADMAP_STATUS.md](ROADMAP_STATUS.md) - Full roadmap
- [EXAMPLES.md](EXAMPLES.md) - More examples
- [README.md](README.md) - Getting started
