//! Text chunking strategies for non-code content
//!
//! Provides specialized chunking for regular text, prose, and documentation.

use crate::error::Result;

/// Text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The text content of this chunk
    pub text: String,
    /// Starting position in original text
    pub start_pos: usize,
    /// Ending position in original text
    pub end_pos: usize,
    /// Chunk type (paragraph, sentence, section)
    pub chunk_type: ChunkType,
    /// Importance score (0.0-10.0)
    pub importance: f64,
}

/// Type of text chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkType {
    /// Paragraph (double newline separated)
    Paragraph,
    /// Sentence (period/question/exclamation separated)
    Sentence,
    /// Section (markdown header or similar)
    Section,
    /// Custom delimiter
    Custom,
}

/// Chunking strategy for text
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextChunkingStrategy {
    /// Split by paragraphs (double newlines)
    Paragraphs,
    /// Split by sentences
    Sentences,
    /// Split by markdown sections (# headers)
    MarkdownSections,
    /// Split by custom delimiter
    Custom,
}

/// Text chunker for non-code content
pub struct TextChunker {
    strategy: TextChunkingStrategy,
    custom_delimiter: Option<String>,
}

impl TextChunker {
    /// Create a new text chunker with paragraph strategy
    ///
    /// # Examples
    ///
    /// ```
    /// use longcodezip::text_chunker::TextChunker;
    ///
    /// let chunker = TextChunker::new();
    /// ```
    pub fn new() -> Self {
        Self {
            strategy: TextChunkingStrategy::Paragraphs,
            custom_delimiter: None,
        }
    }

    /// Create chunker with specific strategy
    pub fn with_strategy(strategy: TextChunkingStrategy) -> Self {
        Self {
            strategy,
            custom_delimiter: None,
        }
    }

    /// Create chunker with custom delimiter
    pub fn with_delimiter(delimiter: &str) -> Self {
        Self {
            strategy: TextChunkingStrategy::Custom,
            custom_delimiter: Some(delimiter.to_string()),
        }
    }

    /// Chunk text according to strategy
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to chunk
    ///
    /// # Returns
    ///
    /// Vector of text chunks with metadata
    pub fn chunk_text(&self, text: &str) -> Result<Vec<TextChunk>> {
        match self.strategy {
            TextChunkingStrategy::Paragraphs => self.chunk_by_paragraphs(text),
            TextChunkingStrategy::Sentences => self.chunk_by_sentences(text),
            TextChunkingStrategy::MarkdownSections => self.chunk_by_markdown(text),
            TextChunkingStrategy::Custom => {
                let delim = self.custom_delimiter.as_deref().unwrap_or("\n\n");
                self.chunk_by_delimiter(text, delim)
            }
        }
    }

    /// Chunk text with a specific strategy (ignoring self.strategy)
    ///
    /// # Arguments
    ///
    /// * `text` - Input text to chunk
    /// * `strategy` - Strategy to use for chunking
    ///
    /// # Returns
    ///
    /// Vector of text chunks with metadata
    pub fn chunk_with_strategy(&self, text: &str, strategy: TextChunkingStrategy) -> Result<Vec<TextChunk>> {
        match strategy {
            TextChunkingStrategy::Paragraphs => self.chunk_by_paragraphs(text),
            TextChunkingStrategy::Sentences => self.chunk_by_sentences(text),
            TextChunkingStrategy::MarkdownSections => self.chunk_by_markdown(text),
            TextChunkingStrategy::Custom => {
                // Use stored delimiter or default
                let delim = self.custom_delimiter.as_deref().unwrap_or("\n\n");
                self.chunk_by_delimiter(text, delim)
            }
        }
    }

    /// Split text into paragraphs
    fn chunk_by_paragraphs(&self, text: &str) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;

        // Split by double newlines or more
        let parts: Vec<&str> = text.split("\n\n").collect();

        for part in parts {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                current_pos += part.len() + 2; // +2 for \n\n
                continue;
            }

            let importance = self.calculate_paragraph_importance(trimmed);

            chunks.push(TextChunk {
                text: trimmed.to_string(),
                start_pos: current_pos,
                end_pos: current_pos + part.len(),
                chunk_type: ChunkType::Paragraph,
                importance,
            });

            current_pos += part.len() + 2;
        }

        Ok(chunks)
    }

    /// Split text into sentences
    fn chunk_by_sentences(&self, text: &str) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let mut sentence_start = 0;
        let mut in_quote = false;

        let chars: Vec<char> = text.chars().collect();
        
        for (i, &ch) in chars.iter().enumerate() {
            // Track quotes
            if ch == '"' || ch == '\'' {
                in_quote = !in_quote;
            }

            // Sentence endings
            if !in_quote && (ch == '.' || ch == '?' || ch == '!') {
                // Check if next char is space or end
                if i + 1 >= chars.len() || chars[i + 1].is_whitespace() {
                    let sentence: String = chars[sentence_start..=i].iter().collect();
                    let trimmed = sentence.trim();

                    if !trimmed.is_empty() && trimmed.len() > 3 {
                        let importance = self.calculate_sentence_importance(trimmed);

                        chunks.push(TextChunk {
                            text: trimmed.to_string(),
                            start_pos: current_pos,
                            end_pos: current_pos + sentence.len(),
                            chunk_type: ChunkType::Sentence,
                            importance,
                        });
                    }

                    sentence_start = i + 1;
                    current_pos += sentence.len();
                }
            }
        }

        // Add remaining text
        if sentence_start < chars.len() {
            let sentence: String = chars[sentence_start..].iter().collect();
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                chunks.push(TextChunk {
                    text: trimmed.to_string(),
                    start_pos: current_pos,
                    end_pos: current_pos + sentence.len(),
                    chunk_type: ChunkType::Sentence,
                    importance: 5.0,
                });
            }
        }

        Ok(chunks)
    }

    /// Split text by markdown headers
    fn chunk_by_markdown(&self, text: &str) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut current_section = String::new();
        let mut section_start = 0;
        let mut current_pos = 0;

        for line in text.lines() {
            let trimmed = line.trim();
            
            // Detect markdown header
            if trimmed.starts_with('#') {
                // Save previous section
                if !current_section.trim().is_empty() {
                    let importance = self.calculate_section_importance(&current_section);
                    
                    chunks.push(TextChunk {
                        text: current_section.trim().to_string(),
                        start_pos: section_start,
                        end_pos: current_pos,
                        chunk_type: ChunkType::Section,
                        importance,
                    });
                }

                // Start new section
                current_section = line.to_string() + "\n";
                section_start = current_pos;
            } else {
                current_section.push_str(line);
                current_section.push('\n');
            }

            current_pos += line.len() + 1;
        }

        // Add last section
        if !current_section.trim().is_empty() {
            let importance = self.calculate_section_importance(&current_section);
            
            chunks.push(TextChunk {
                text: current_section.trim().to_string(),
                start_pos: section_start,
                end_pos: current_pos,
                chunk_type: ChunkType::Section,
                importance,
            });
        }

        Ok(chunks)
    }

    /// Split by custom delimiter
    fn chunk_by_delimiter(&self, text: &str, delimiter: &str) -> Result<Vec<TextChunk>> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;

        let parts: Vec<&str> = text.split(delimiter).collect();

        for part in parts {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                current_pos += part.len() + delimiter.len();
                continue;
            }

            chunks.push(TextChunk {
                text: trimmed.to_string(),
                start_pos: current_pos,
                end_pos: current_pos + part.len(),
                chunk_type: ChunkType::Custom,
                importance: 5.0,
            });

            current_pos += part.len() + delimiter.len();
        }

        Ok(chunks)
    }

    /// Calculate importance for paragraph
    fn calculate_paragraph_importance(&self, text: &str) -> f64 {
        let mut score: f64 = 5.0;

        // Longer paragraphs slightly more important
        if text.len() > 200 {
            score += 1.0;
        }

        // Contains numbers/data
        if text.chars().any(|c| c.is_numeric()) {
            score += 0.5;
        }

        // Contains question marks (important questions)
        if text.contains('?') {
            score += 1.0;
        }

        // Starts with capital letter (proper sentence)
        if text.chars().next().map_or(false, |c| c.is_uppercase()) {
            score += 0.5;
        }

        // Contains keywords
        let keywords = ["important", "key", "main", "critical", "essential", "note"];
        for keyword in keywords {
            if text.to_lowercase().contains(keyword) {
                score += 1.5;
                break;
            }
        }

        score.min(10.0)
    }

    /// Calculate importance for sentence
    fn calculate_sentence_importance(&self, text: &str) -> f64 {
        let mut score: f64 = 5.0;

        // Questions are important
        if text.contains('?') {
            score += 2.0;
        }

        // Contains numbers
        if text.chars().any(|c| c.is_numeric()) {
            score += 1.0;
        }

        // Short sentences less important
        if text.len() < 30 {
            score -= 1.0;
        }

        // Very long sentences more important
        if text.len() > 100 {
            score += 1.0;
        }

        score.max(0.0).min(10.0)
    }

    /// Calculate importance for section
    fn calculate_section_importance(&self, text: &str) -> f64 {
        let mut score: f64 = 5.0;

        // Check header level (more # = less important subsection)
        let first_line = text.lines().next().unwrap_or("");
        let hash_count = first_line.chars().take_while(|&c| c == '#').count();
        
        match hash_count {
            1 => score += 3.0,  // # Main title
            2 => score += 2.0,  // ## Section
            3 => score += 1.0,  // ### Subsection
            _ => {}
        }

        // Contains code blocks
        if text.contains("```") {
            score += 1.5;
        }

        // Contains lists
        if text.contains("- ") || text.contains("* ") {
            score += 0.5;
        }

        score.min(10.0)
    }
}

impl Default for TextChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paragraph_chunking() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunker = TextChunker::new();
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "First paragraph.");
        assert_eq!(chunks[1].text, "Second paragraph.");
    }

    #[test]
    fn test_sentence_chunking() {
        let text = "First sentence. Second sentence? Third sentence!";
        let chunker = TextChunker::with_strategy(TextChunkingStrategy::Sentences);
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert!(chunks.len() >= 3);
        assert!(chunks[0].text.contains("First"));
    }

    #[test]
    fn test_markdown_chunking() {
        let text = "# Title\n\nContent\n\n## Section\n\nMore content";
        let chunker = TextChunker::with_strategy(TextChunkingStrategy::MarkdownSections);
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_custom_delimiter() {
        let text = "Part1|||Part2|||Part3";
        let chunker = TextChunker::with_delimiter("|||");
        let chunks = chunker.chunk_text(text).unwrap();
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Part1");
    }

    #[test]
    fn test_importance_scoring() {
        let chunker = TextChunker::new();
        
        // Important keywords
        let score1 = chunker.calculate_paragraph_importance("This is important information");
        assert!(score1 > 6.0);
        
        // Regular text
        let score2 = chunker.calculate_paragraph_importance("Just regular text here");
        assert!(score2 < 7.0);
    }
}
