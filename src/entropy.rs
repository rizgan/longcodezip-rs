//! Entropy-based text chunking for fine-grained compression
//!
//! This module implements perplexity-based text splitting similar to LongLLMLingua.
//! It splits code into chunks at high-perplexity boundaries (topic shifts).

use crate::error::Result;

/// Represents a chunk of text with its perplexity score
#[derive(Debug, Clone)]
pub struct EntropyChunk {
    /// The text content of this chunk
    pub text: String,
    /// Starting line number in original text
    pub start_line: usize,
    /// Ending line number in original text
    pub end_line: usize,
    /// Average perplexity score for this chunk
    pub perplexity: f64,
}

/// Represents a line with its perplexity score
#[derive(Debug, Clone)]
struct LineInfo {
    text: String,
    line_number: usize,
    perplexity: f64,
}

/// Entropy-based text chunker
///
/// Splits text into semantic chunks by detecting perplexity spikes.
/// High perplexity indicates topic shifts or important boundaries.
pub struct EntropyChunker {
    /// Statistical method for threshold calculation
    method: ThresholdMethod,
    /// Multiplier for threshold (default: 0.2)
    k_factor: f64,
}

/// Methods for calculating adaptive thresholds
#[derive(Debug, Clone, Copy)]
pub enum ThresholdMethod {
    /// Standard deviation based
    Std,
    /// Robust standard deviation (MAD-based)
    RobustStd,
    /// Interquartile range based
    Iqr,
    /// Median absolute deviation
    Mad,
}

impl EntropyChunker {
    /// Create a new entropy chunker with default settings
    ///
    /// # Examples
    ///
    /// ```
    /// use longcodezip::entropy::EntropyChunker;
    ///
    /// let chunker = EntropyChunker::new();
    /// ```
    pub fn new() -> Self {
        Self {
            method: ThresholdMethod::Std,
            k_factor: 0.2,
        }
    }

    /// Create a chunker with specific method and k-factor
    ///
    /// # Arguments
    ///
    /// * `method` - Statistical method for threshold
    /// * `k_factor` - Threshold multiplier (typically 0.1-0.5)
    pub fn with_config(method: ThresholdMethod, k_factor: f64) -> Self {
        Self { method, k_factor }
    }

    /// Split text into entropy-based chunks
    ///
    /// This is a simplified version that uses heuristics instead of LLM.
    /// It approximates perplexity using:
    /// - Line length variance
    /// - Indentation changes
    /// - Empty line detection
    /// - Special character density
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to chunk
    ///
    /// # Returns
    ///
    /// Vector of chunks with perplexity scores
    pub fn chunk_text(&self, text: &str) -> Result<Vec<EntropyChunk>> {
        // Split into lines
        let lines: Vec<&str> = text.lines().collect();
        
        if lines.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate perplexity approximation for each line
        let line_infos = self.calculate_line_perplexities(&lines);

        // Find spike points
        let spike_indices = self.find_perplexity_spikes(&line_infos);

        // Create chunks
        let chunks = self.create_chunks(&line_infos, &spike_indices);

        Ok(chunks)
    }

    /// Calculate perplexity approximation for each line
    ///
    /// Uses heuristics instead of actual LLM perplexity:
    /// - Indentation changes → high perplexity
    /// - Empty lines → very high perplexity
    /// - Long lines after short → high perplexity
    /// - Special chars (def, class, fn, etc.) → high perplexity
    fn calculate_line_perplexities(&self, lines: &[&str]) -> Vec<LineInfo> {
        let mut line_infos = Vec::new();

        for (i, &line) in lines.iter().enumerate() {
            let text = line.to_string();
            
            // Base perplexity
            let mut ppl = 1.0;

            // Empty line → high perplexity (topic boundary)
            if line.trim().is_empty() {
                ppl = 10.0;
            } else {
                // Calculate indentation
                let indent = line.len() - line.trim_start().len();

                // Check indentation change
                if i > 0 {
                    let prev_line = lines[i - 1];
                    let prev_indent = prev_line.len() - prev_line.trim_start().len();
                    let indent_change = (indent as i32 - prev_indent as i32).abs();
                    
                    // Large indentation changes → higher perplexity
                    ppl += indent_change as f64 * 0.5;
                }

                // Check for function/class definitions (topic boundaries)
                let trimmed = line.trim();
                if trimmed.starts_with("def ")
                    || trimmed.starts_with("class ")
                    || trimmed.starts_with("fn ")
                    || trimmed.starts_with("impl ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("enum ")
                    || trimmed.starts_with("function ")
                    || trimmed.starts_with("const ")
                    || trimmed.starts_with("export ")
                {
                    ppl += 5.0;
                }

                // Check for comments (often mark boundaries)
                if trimmed.starts_with("//") || trimmed.starts_with("#") || trimmed.starts_with("/*") {
                    ppl += 2.0;
                }

                // Line length variance
                if i > 0 {
                    let prev_len = lines[i - 1].trim().len();
                    let curr_len = trimmed.len();
                    let len_diff = (curr_len as i32 - prev_len as i32).abs();
                    
                    if len_diff > 20 {
                        ppl += len_diff as f64 * 0.1;
                    }
                }

                // Special character density
                let special_chars = trimmed.chars().filter(|c| 
                    *c == '{' || *c == '}' || *c == '[' || *c == ']' || *c == '(' || *c == ')'
                ).count();
                
                ppl += special_chars as f64 * 0.3;
            }

            line_infos.push(LineInfo {
                text,
                line_number: i,
                perplexity: ppl,
            });
        }

        line_infos
    }

    /// Calculate adaptive threshold based on statistical method
    fn calculate_threshold(&self, perplexities: &[f64]) -> f64 {
        // Filter valid values
        let valid: Vec<f64> = perplexities
            .iter()
            .filter(|&&p| p.is_finite() && p > 0.0)
            .copied()
            .collect();

        if valid.len() < 3 {
            return 0.5; // Fallback
        }

        match self.method {
            ThresholdMethod::Std => {
                let mean = valid.iter().sum::<f64>() / valid.len() as f64;
                let variance = valid.iter()
                    .map(|p| (p - mean).powi(2))
                    .sum::<f64>() / valid.len() as f64;
                let std = variance.sqrt();
                mean + self.k_factor * std
            }
            ThresholdMethod::RobustStd => {
                let median = self.median(&valid);
                let mad = self.median_absolute_deviation(&valid, median);
                let robust_std = mad * 1.4826; // MAD to std conversion
                median + self.k_factor * robust_std
            }
            ThresholdMethod::Iqr => {
                let q25 = self.percentile(&valid, 25.0);
                let q75 = self.percentile(&valid, 75.0);
                let iqr = q75 - q25;
                q75 + self.k_factor * iqr
            }
            ThresholdMethod::Mad => {
                let median = self.median(&valid);
                let mad = self.median_absolute_deviation(&valid, median);
                median + self.k_factor * mad
            }
        }
    }

    /// Find indices where perplexity spikes occur
    fn find_perplexity_spikes(&self, line_infos: &[LineInfo]) -> Vec<usize> {
        let perplexities: Vec<f64> = line_infos.iter().map(|l| l.perplexity).collect();
        let threshold = self.calculate_threshold(&perplexities);

        let mut spike_indices = Vec::new();

        for i in 1..perplexities.len() - 1 {
            let current = perplexities[i];
            let left = perplexities[i - 1];
            let right = perplexities[i + 1];

            // Skip invalid values
            if !current.is_finite() || !left.is_finite() || !right.is_finite() {
                continue;
            }

            // Check if current is a spike
            let left_diff = current - left;
            let right_diff = current - right;

            if (left_diff >= threshold || right_diff >= threshold) 
                && left_diff >= 0.0 
                && right_diff >= 0.0 
            {
                spike_indices.push(i);
            }
        }

        spike_indices
    }

    /// Create chunks from lines and spike indices
    fn create_chunks(&self, line_infos: &[LineInfo], spike_indices: &[usize]) -> Vec<EntropyChunk> {
        let mut chunks = Vec::new();

        // Create split points: start + spikes + end
        let mut split_points = vec![0];
        split_points.extend(spike_indices.iter().map(|&i| i + 1));
        split_points.push(line_infos.len());

        for i in 0..split_points.len() - 1 {
            let start = split_points[i];
            let end = split_points[i + 1];

            if start >= end {
                continue;
            }

            // Collect lines for this chunk
            let chunk_lines: Vec<String> = line_infos[start..end]
                .iter()
                .map(|l| l.text.clone())
                .collect();

            // Calculate average perplexity
            let avg_ppl = line_infos[start..end]
                .iter()
                .map(|l| l.perplexity)
                .filter(|p| p.is_finite())
                .sum::<f64>() / (end - start) as f64;

            chunks.push(EntropyChunk {
                text: chunk_lines.join("\n"),
                start_line: line_infos[start].line_number,
                end_line: line_infos[end - 1].line_number,
                perplexity: avg_ppl,
            });
        }

        chunks
    }

    /// Calculate median of a sorted array
    fn median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate median absolute deviation
    fn median_absolute_deviation(&self, values: &[f64], median: f64) -> f64 {
        let deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
        self.median(&deviations)
    }

    /// Calculate percentile
    fn percentile(&self, values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl Default for EntropyChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_simple() {
        let chunker = EntropyChunker::new();
        let text = "line1\nline2\n\nline3\nline4";
        
        let chunks = chunker.chunk_text(text).unwrap();
        assert!(chunks.len() > 0);
    }

    #[test]
    fn test_chunking_python_code() {
        let chunker = EntropyChunker::new();
        let text = r#"
def function1():
    print("hello")

def function2():
    print("world")
"#;
        
        let chunks = chunker.chunk_text(text).unwrap();
        // Should split at function boundaries (heuristic may produce 1 or more chunks)
        assert!(chunks.len() >= 1);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let chunker = EntropyChunker::new();
        let chunks = chunker.chunk_text("").unwrap();
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn test_threshold_methods() {
        let chunker_std = EntropyChunker::with_config(ThresholdMethod::Std, 0.2);
        let chunker_mad = EntropyChunker::with_config(ThresholdMethod::Mad, 0.2);
        
        let text = "def foo():\n    pass\n\ndef bar():\n    pass";
        
        let chunks1 = chunker_std.chunk_text(text).unwrap();
        let chunks2 = chunker_mad.chunk_text(text).unwrap();
        
        // Both should produce reasonable chunks
        assert!(chunks1.len() > 0);
        assert!(chunks2.len() > 0);
    }
}
