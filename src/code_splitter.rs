//! Code splitter for dividing code into function-level chunks

use crate::error::{Error, Result};
use crate::types::CodeLanguage;
use regex::Regex;

/// Split code into function-based chunks
pub fn split_code_by_functions(code: &str, language: CodeLanguage) -> Result<Vec<String>> {
    let pattern = get_pattern_for_language(language)?;
    
    let regex = Regex::new(&pattern)
        .map_err(|e| Error::ProcessingError(format!("Invalid regex pattern: {}", e)))?;
    
    let matches: Vec<_> = regex.find_iter(code).collect();
    
    if matches.is_empty() {
        // No matches found, return the whole code as one chunk
        return Ok(vec![code.to_string()]);
    }
    
    let mut chunks = Vec::new();
    
    // Add code before first match if any
    if matches[0].start() > 0 {
        chunks.push(code[..matches[0].start()].to_string());
    }
    
    // Process each match
    for (i, match_obj) in matches.iter().enumerate() {
        let start = match_obj.start();
        
        // End is either start of next match or end of code
        let end = if i < matches.len() - 1 {
            matches[i + 1].start()
        } else {
            code.len()
        };
        
        chunks.push(code[start..end].to_string());
    }
    
    Ok(chunks)
}

/// Get regex pattern for detecting functions/classes in different languages
fn get_pattern_for_language(language: CodeLanguage) -> Result<String> {
    let pattern = match language {
        CodeLanguage::Python => {
            // Match 'def' or 'class' at start of line or after newline
            r"(?m)^[ \t]*(def|class)\s+\w+"
        }
        CodeLanguage::Rust => {
            // Match 'fn', 'impl', 'struct', 'enum', 'trait'
            r"(?m)^[ \t]*(?:pub\s+)?(?:async\s+)?(?:fn|impl|struct|enum|trait)\s+"
        }
        CodeLanguage::TypeScript | CodeLanguage::JavaScript => {
            // Match 'function', 'class', arrow functions, methods
            r"(?m)^[ \t]*(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+|(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\("
        }
        CodeLanguage::Cpp => {
            // Match class or function definitions
            r"(?m)^[ \t]*(?:class|struct|(?:\w+\s+)*\w+\s+\w+\s*\()"
        }
        CodeLanguage::Java => {
            // Match class or method definitions
            r"(?m)^[ \t]*(?:public|private|protected)?\s*(?:static\s+)?(?:class|(?:\w+\s+)*\w+\s+\w+\s*\()"
        }
        CodeLanguage::Go => {
            // Match 'func' or 'type'
            r"(?m)^[ \t]*(?:func|type)\s+\w+"
        }
    };
    
    Ok(pattern.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_split_python_code() {
        let code = r#"
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

class Calculator:
    def compute(self):
        pass
"#;
        
        let chunks = split_code_by_functions(code, CodeLanguage::Python).unwrap();
        // Should have at least 3 chunks (add, multiply, Calculator class)
        assert!(chunks.len() >= 3);
    }
    
    #[test]
    fn test_split_rust_code() {
        let code = r#"
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
"#;
        
        let chunks = split_code_by_functions(code, CodeLanguage::Rust).unwrap();
        assert!(chunks.len() >= 3);
    }
    
    #[test]
    fn test_empty_code() {
        let code = "";
        let chunks = split_code_by_functions(code, CodeLanguage::Python).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "");
    }
}
