//! Example demonstrating different tokenizers for different models

use longcodezip::{Tokenizer, TokenizerModel};

fn main() {
    println!("🔢 Tokenizer Comparison Demo\n");
    println!("{}", "=".repeat(60));
    
    let sample_text = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"#;
    
    println!("Sample code:\n{}", sample_text);
    println!("{}", "=".repeat(60));
    
    // Test different tokenizers
    let tokenizers = vec![
        ("GPT-4 (cl100k_base)", TokenizerModel::Cl100kBase),
        ("GPT-4o (o200k_base)", TokenizerModel::O200kBase),
        ("Codex (p50k_base)", TokenizerModel::P50kBase),
        ("GPT-3 (r50k_base)", TokenizerModel::R50kBase),
    ];
    
    println!("\n📊 Token counts by model:");
    println!("{:-<60}", "");
    
    for (name, model) in tokenizers {
        let tokenizer = Tokenizer::new(model);
        let count = tokenizer.count_tokens(sample_text).unwrap();
        println!("{:<25} {} tokens", name, count);
    }
    
    println!("{}", "=".repeat(60));
    
    // Demonstrate encoding/decoding
    println!("\n🔍 Encoding example (GPT-4):");
    let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
    
    let simple_text = "Hello, world!";
    let tokens = tokenizer.encode(simple_text).unwrap();
    let decoded = tokenizer.decode(&tokens).unwrap();
    
    println!("Original: {}", simple_text);
    println!("Tokens:   {:?}", tokens);
    println!("Token count: {}", tokens.len());
    println!("Decoded:  {}", decoded);
    
    // Demonstrate model name detection
    println!("\n🎯 Automatic model detection:");
    println!("{:-<60}", "");
    
    let models = vec![
        "gpt-4",
        "gpt-4o",
        "gpt-3.5-turbo",
        "deepseek-chat",
        "claude-3-opus",
        "code-davinci-002",
    ];
    
    for model in models {
        let tokenizer = Tokenizer::from_model_name(model);
        println!("{:<25} -> {}", model, tokenizer.model().name());
    }
    
    // Demonstrate truncation
    println!("\n✂️  Text truncation example:");
    println!("{:-<60}", "");
    
    let long_text = "The quick brown fox jumps over the lazy dog. This is a longer sentence that will be truncated.";
    let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);
    
    let original_tokens = tokenizer.count_tokens(long_text).unwrap();
    println!("Original: {} tokens", original_tokens);
    println!("Text: {}\n", long_text);
    
    for max_tokens in [5, 10, 15] {
        let truncated = tokenizer.truncate(long_text, max_tokens).unwrap();
        let actual_tokens = tokenizer.count_tokens(&truncated).unwrap();
        println!("Truncated to {} tokens ({} actual):", max_tokens, actual_tokens);
        println!("  {}\n", truncated);
    }
    
    // Batch processing
    println!("\n📦 Batch token counting:");
    println!("{:-<60}", "");
    
    let texts = vec![
        "Short text",
        "A bit longer text with more words",
        "def example():\n    return 'code'",
    ];
    
    let counts = tokenizer.count_tokens_batch(&texts).unwrap();
    
    for (text, count) in texts.iter().zip(counts.iter()) {
        println!("{:<40} {} tokens", 
            if text.len() > 37 { 
                format!("{}...", &text[..37]) 
            } else { 
                text.to_string() 
            }, 
            count
        );
    }
    
    println!("\n{}", "=".repeat(60));
    println!("✅ Demo complete!");
}
