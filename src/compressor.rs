//! Main compressor implementation

use crate::code_splitter::split_code_by_functions;
use crate::entropy::EntropyChunker;
use crate::error::{Error, Result};
use crate::optimizer::{Block, KnapsackOptimizer};
use crate::provider::{create_provider, LLMProvider};
use crate::types::{CompressionConfig, CompressionResult, FunctionCompression};
use log::{debug, info};
use std::collections::{HashMap, HashSet};

/// Main LongCodeZip compressor
pub struct LongCodeZip {
    config: CompressionConfig,
    provider: Box<dyn LLMProvider>,
    entropy_chunker: EntropyChunker,
    optimizer: KnapsackOptimizer,
}

impl LongCodeZip {
    /// Create a new compressor with the given configuration
    pub fn new(config: CompressionConfig) -> Result<Self> {
        if config.provider.api_key.is_empty() {
            return Err(Error::ConfigError("API key is required".to_string()));
        }
        
        let provider = create_provider(config.provider.clone());
        let entropy_chunker = EntropyChunker::new();
        let optimizer = KnapsackOptimizer::new();
        
        Ok(Self { 
            config, 
            provider,
            entropy_chunker,
            optimizer,
        })
    }
    
    /// Compress code with a query and instruction
    ///
    /// # Arguments
    /// * `code` - The source code to compress
    /// * `query` - Query to determine relevance
    /// * `instruction` - Additional instruction for the prompt
    ///
    /// # Returns
    /// CompressionResult with compressed code and statistics
    pub async fn compress_code(
        &self,
        code: &str,
        query: &str,
        instruction: &str,
    ) -> Result<CompressionResult> {
        info!("Starting code compression");
        debug!("Code length: {} chars", code.len());
        debug!("Query: {}", query);
        
        // Step 1: Split code - use entropy for fine-grained, functions for coarse
        let chunks = if !self.config.rank_only && self.config.use_knapsack {
            // Fine-grained: entropy-based chunking
            info!("Using fine-grained compression with entropy chunking");
            let entropy_chunks = self.entropy_chunker.chunk_text(code)?;
            
            // Fallback to function split if entropy gives too few chunks
            if entropy_chunks.len() < 2 {
                info!("Entropy chunking produced {} chunk(s), falling back to function splitting", entropy_chunks.len());
                split_code_by_functions(code, self.config.language)?
            } else {
                entropy_chunks.into_iter().map(|c| c.text).collect()
            }
        } else {
            // Coarse-grained: function-based chunking
            info!("Using coarse-grained compression with function splitting");
            split_code_by_functions(code, self.config.language)?
        };
        info!("Split code into {} chunks", chunks.len());
        
        // Step 2: Calculate token counts
        let mut chunk_tokens = Vec::new();
        let mut total_tokens = 0;
        
        for chunk in &chunks {
            let tokens = self.provider.get_token_count(chunk).await?;
            chunk_tokens.push(tokens);
            total_tokens += tokens;
        }
        
        debug!("Total tokens: {}", total_tokens);
        
        // Step 3: Determine target tokens
        let target_token = if self.config.target_token > 0 {
            self.config.target_token as usize
        } else {
            (total_tokens as f64 * self.config.rate) as usize
        };
        
        debug!("Target tokens: {}", target_token);
        
        // Step 4: Rank chunks by relevance if query is provided
        let chunk_importances = if !query.trim().is_empty() {
            self.calculate_chunk_importances(&chunks, query).await?
        } else {
            // No query, equal importance
            vec![1.0; chunks.len()]
        };
        
        // Step 5: Select chunks - use knapsack for fine-grained
        let (selected_indices, selected_tokens) = if self.config.use_knapsack {
            self.select_chunks_knapsack(
                &chunks,
                &chunk_tokens,
                &chunk_importances,
                target_token,
            ).await?
        } else {
            // Greedy selection by ranking
            let mut ranked: Vec<usize> = (0..chunks.len()).collect();
            ranked.sort_by(|&a, &b| {
                chunk_importances[b]
                    .partial_cmp(&chunk_importances[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.select_chunks_within_budget(&ranked, &chunk_tokens, target_token)
        };
        
        info!(
            "Selected {} chunks ({} tokens)",
            selected_indices.len(),
            selected_tokens
        );
        
        // Step 6: Build compressed code
        let mut compressed_chunks = Vec::new();
        let mut function_compressions = HashMap::new();
        let comment_marker = self.config.language.comment_marker();
        
        for (i, chunk) in chunks.iter().enumerate() {
            if selected_indices.contains(&i) {
                compressed_chunks.push(chunk.clone());
                
                function_compressions.insert(
                    i,
                    FunctionCompression {
                        original_tokens: chunk_tokens[i],
                        compressed_tokens: chunk_tokens[i],
                        compression_ratio: 1.0,
                        individual_fine_ratio: Some(1.0),
                        note: Some("Selected".to_string()),
                    },
                );
            } else {
                // Add omission marker for skipped chunks
                let omission = format!("{} ... ", comment_marker);
                compressed_chunks.push(omission);
            }
        }
        
        let compressed_code = compressed_chunks.join("\n\n");
        
        // Step 7: Build final prompt
        let compressed_prompt = self.build_prompt(&compressed_code, query, instruction);
        
        let final_tokens = self.provider.get_token_count(&compressed_prompt).await?;
        
        Ok(CompressionResult {
            original_code: code.to_string(),
            compressed_code: compressed_code.clone(),
            compressed_prompt,
            original_tokens: total_tokens,
            compressed_tokens: selected_tokens,
            final_compressed_tokens: final_tokens,
            compression_ratio: selected_tokens as f64 / total_tokens as f64,
            function_compressions,
            selected_functions: selected_indices,
            fine_grained_method_used: if self.config.use_knapsack {
                Some("entropy_knapsack".to_string())
            } else if self.config.rank_only {
                Some("rank_only".to_string())
            } else {
                None
            },
        })
    }
    
    /// Calculate importance scores for chunks
    async fn calculate_chunk_importances(
        &self,
        chunks: &[String],
        query: &str,
    ) -> Result<Vec<f64>> {
        debug!("Calculating importance for {} chunks", chunks.len());
        
        let mut importances = Vec::new();
        
        for chunk in chunks {
            let score = self.provider.calculate_relevance(chunk, query).await?;
            importances.push(score);
        }
        
        Ok(importances)
    }
    
    /// Select chunks using knapsack optimizer
    async fn select_chunks_knapsack(
        &self,
        chunks: &[String],
        chunk_tokens: &[usize],
        chunk_importances: &[f64],
        target_token: usize,
    ) -> Result<(Vec<usize>, usize)> {
        debug!("Using knapsack optimizer for chunk selection");
        
        // Create blocks for optimizer
        let blocks: Vec<Block> = chunks
            .iter()
            .enumerate()
            .map(|(i, text)| Block {
                index: i,
                text: text.clone(),
                tokens: chunk_tokens[i],
                importance: chunk_importances[i],
            })
            .collect();
        
        let preserved = HashSet::new();
        let result = self.optimizer.select_blocks(&blocks, target_token, &preserved)?;
        
        let mut selected: Vec<usize> = result.selected_indices.into_iter().collect();
        
        // Fallback: if nothing was selected, use greedy
        if selected.is_empty() {
            info!("Knapsack returned empty selection, falling back to greedy");
            let mut ranked: Vec<usize> = (0..chunks.len()).collect();
            ranked.sort_by(|&a, &b| {
                chunk_importances[b]
                    .partial_cmp(&chunk_importances[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            return Ok(self.select_chunks_within_budget(&ranked, chunk_tokens, target_token));
        }
        
        selected.sort_unstable();
        
        info!(
            "Knapsack selected {} blocks, value={:.2}, weight={}/{}",
            selected.len(),
            result.total_value,
            result.total_weight,
            target_token
        );
        
        Ok((selected, result.total_weight))
    }
    
    /// Select chunks within token budget
    fn select_chunks_within_budget(
        &self,
        ranked_indices: &[usize],
        chunk_tokens: &[usize],
        target_token: usize,
    ) -> (Vec<usize>, usize) {
        let mut selected = Vec::new();
        let mut current_tokens = 0;
        
        for &idx in ranked_indices {
            let tokens = chunk_tokens[idx];
            
            if current_tokens + tokens <= target_token {
                selected.push(idx);
                current_tokens += tokens;
            } else if selected.is_empty() {
                // Always select at least one chunk
                selected.push(idx);
                current_tokens += tokens;
                break;
            }
        }
        
        // Sort selected indices to maintain original code order
        selected.sort_unstable();
        
        (selected, current_tokens)
    }
    
    /// Build final prompt with instruction and query
    fn build_prompt(&self, compressed_code: &str, query: &str, instruction: &str) -> String {
        let mut parts = Vec::new();
        
        if !instruction.trim().is_empty() {
            parts.push(instruction.trim().to_string());
        }
        
        if !compressed_code.trim().is_empty() {
            parts.push(compressed_code.to_string());
        }
        
        if self.config.repeat_instruction_at_end && !instruction.trim().is_empty() {
            parts.push(instruction.trim().to_string());
        }
        
        if !query.trim().is_empty() {
            parts.push(query.trim().to_string());
        }
        
        parts.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CodeLanguage, ProviderConfig};
    
    #[tokio::test]
    async fn test_compress_simple_code() {
        let code = r#"
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
"#;
        
        let config = CompressionConfig {
            rate: 0.5,
            language: CodeLanguage::Python,
            provider: ProviderConfig::new(
                "test",
                "https://api.example.com",
                "test-key",
                "test-model",
            ),
            ..Default::default()
        };
        
        let compressor = LongCodeZip::new(config).unwrap();
        let result = compressor.compress_code(code, "", "").await.unwrap();
        
        assert!(!result.compressed_code.is_empty());
        assert!(result.compression_ratio <= 1.0);
    }
}
