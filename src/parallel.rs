//! Parallel processing utilities for chunk analysis

use crate::cache::ResponseCache;
use crate::error::Result;
use crate::provider::LLMProvider;
use std::sync::Mutex;

/// Parallel chunk processor
pub struct ParallelProcessor {
    /// Number of threads to use (0 = auto)
    num_threads: usize,
}

impl ParallelProcessor {
    /// Create a new parallel processor
    pub fn new() -> Self {
        Self { num_threads: 0 }
    }
    
    /// Create with specific thread count
    pub fn with_threads(num_threads: usize) -> Self {
        Self { num_threads }
    }
    
    /// Calculate importances for chunks in parallel
    pub async fn calculate_importances_parallel(
        &self,
        provider: &(dyn LLMProvider + Sync),
        chunks: &[String],
        query: &str,
        cache: Option<&Mutex<ResponseCache>>,
        model: &str,
    ) -> Result<Vec<f64>> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        
        // Set up thread pool if specified
        if self.num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .ok(); // Ignore if already initialized
        }
        
        // Process chunks in parallel using tokio
        let query = query.to_string();
        let model = model.to_string();
        
        // Create async tasks for all chunks
        let tasks: Vec<_> = chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| {
                let chunk = chunk.clone();
                let query = query.clone();
                let model = model.clone();
                
                async move {
                    // Check cache first
                    let cache_key = if let Some(cache_mutex) = cache {
                        let cache = cache_mutex.lock().unwrap();
                        let key = ResponseCache::generate_key(&chunk, &query, &model);
                        if let Some(score) = cache.get(&key) {
                            return Ok((idx, score));
                        }
                        Some(key)
                    } else {
                        None
                    };
                    
                    // Calculate score
                    let score = provider.calculate_relevance(&chunk, &query).await?;
                    
                    // Store in cache
                    if let (Some(key), Some(cache_mutex)) = (cache_key, cache) {
                        let mut cache = cache_mutex.lock().unwrap();
                        cache.set(key, score);
                    }
                    
                    Ok::<(usize, f64), crate::error::Error>((idx, score))
                }
            })
            .collect();
        
        // Execute all tasks concurrently
        let results = futures::future::join_all(tasks).await;
        
        // Collect and sort by index
        let mut scores = vec![0.0; chunks.len()];
        for result in results {
            let (idx, score) = result?;
            scores[idx] = score;
        }
        
        Ok(scores)
    }
}

impl Default for ParallelProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderConfig;
    use crate::provider::OpenAICompatibleProvider;
    
    #[tokio::test]
    async fn test_parallel_processor_creation() {
        let processor = ParallelProcessor::new();
        assert_eq!(processor.num_threads, 0);
        
        let processor = ParallelProcessor::with_threads(4);
        assert_eq!(processor.num_threads, 4);
    }
    
    #[tokio::test]
    async fn test_parallel_importances_empty() {
        let processor = ParallelProcessor::new();
        let config = ProviderConfig::new("test", "http://localhost", "key", "model");
        let provider = OpenAICompatibleProvider::new(config);
        
        let result = processor
            .calculate_importances_parallel(&provider, &[], "query", None, "model")
            .await
            .unwrap();
        
        assert_eq!(result.len(), 0);
    }
}
