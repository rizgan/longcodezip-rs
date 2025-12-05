//! Test text compression with new TextChunker

use longcodezip::{
    CodeLanguage, CompressionConfig, LongCodeZip, ProviderConfig,
    text_chunker::TextChunkingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let scientific_article = r#"
Introduction to Quantum Computing

Quantum computing represents a paradigm shift in computational technology. Unlike classical computers that use bits, quantum computers use qubits that can exist in superposition.

Superposition and Entanglement

The principle of superposition allows qubits to be in multiple states simultaneously. When combined with quantum entanglement, quantum computers can process vast amounts of information in parallel.

This entanglement creates correlations between qubits that have no classical equivalent. Two entangled qubits behave as a single system, regardless of the distance between them.

Quantum Gates and Circuits

Quantum gates manipulate qubits through unitary transformations. Common gates include the Hadamard gate, Pauli gates (X, Y, Z), and the CNOT gate. These gates form the building blocks of quantum circuits.

The Hadamard gate creates superposition, while the CNOT gate creates entanglement. Together, they enable universal quantum computation.

Applications in Cryptography

Quantum computing has revolutionary applications in cryptography. Shor's algorithm can factor large numbers exponentially faster than classical algorithms, threatening current encryption methods.

However, quantum key distribution (QKD) offers unbreakable encryption based on the laws of physics. Any attempt to intercept the key disturbs the quantum state, alerting the parties.

Machine Learning and Optimization

Quantum computers excel at optimization problems. The quantum approximate optimization algorithm (QAOA) can find near-optimal solutions to complex problems.

In machine learning, quantum neural networks process data in higher-dimensional Hilbert spaces, potentially offering advantages in pattern recognition and classification tasks.

Current Challenges

Decoherence remains the primary obstacle. Quantum states are fragile and easily disturbed by environmental noise. Error correction requires significant overhead, as a single logical qubit may need hundreds of physical qubits.

Scaling to large numbers of qubits presents engineering challenges. Maintaining coherence times while increasing qubit count is an active area of research.

Future Outlook

As technology advances, quantum computers may revolutionize fields from drug discovery to financial modeling. The race to achieve quantum advantage continues across industry and academia.
"#;

    let provider = ProviderConfig::deepseek("sk-b78ab15d637749a9a8c6ae69a919c0a9");

    println!("=== Text Compression with TextChunker ===\n");
    println!("Original text: {} chars\n", scientific_article.len());

    let query = "What is quantum entanglement and how does it work?";
    println!("Query: {}\n", query);

    // Test 1: Paragraph strategy
    println!("--- Strategy 1: Paragraphs ---");
    test_compression(
        &scientific_article,
        query,
        TextChunkingStrategy::Paragraphs,
        provider.clone(),
    ).await?;

    println!("\n");

    // Test 2: Sentences strategy
    println!("--- Strategy 2: Sentences ---");
    test_compression(
        &scientific_article,
        query,
        TextChunkingStrategy::Sentences,
        provider.clone(),
    ).await?;

    println!("\n");

    // Test 3: Markdown sections
    let markdown_text = r#"
# Quantum Computing Guide

## Introduction

Quantum computing uses qubits instead of classical bits.

## Key Concepts

### Superposition

Qubits can be in multiple states at once.

### Entanglement

Quantum particles become correlated.

## Applications

- Cryptography
- Drug discovery
- Optimization

## Challenges

Decoherence and scaling remain difficult.
"#;

    println!("--- Strategy 3: Markdown Sections ---");
    test_compression(
        markdown_text,
        "What is superposition?",
        TextChunkingStrategy::MarkdownSections,
        provider.clone(),
    ).await?;

    println!("\n=== Summary ===");
    println!("✅ TextChunker successfully integrated!");
    println!("📊 Strategies available:");
    println!("   - Paragraphs: Best for prose and articles");
    println!("   - Sentences: Fine-grained control");
    println!("   - MarkdownSections: Structured documents");
    println!("   - Custom: Use your own delimiter");

    Ok(())
}

async fn test_compression(
    text: &str,
    query: &str,
    strategy: TextChunkingStrategy,
    provider: ProviderConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = CompressionConfig::default()
        .with_rate(0.4) // Aggressive compression
        .with_language(CodeLanguage::Python) // Not used for text
        .with_provider(provider);

    let compressor = LongCodeZip::new(config)?;
    
    let start = std::time::Instant::now();
    let result = compressor
        .compress_text(text, query, "", strategy)
        .await?;
    let elapsed = start.elapsed();

    println!("Strategy: {:?}", strategy);
    println!("Time: {:.2}s", elapsed.as_secs_f64());
    println!("Original tokens: {}", result.original_tokens);
    println!("Compressed tokens: {}", result.compressed_tokens);
    println!("Compression: {:.1}%", result.compression_ratio * 100.0);
    println!("Chunks selected: {}/{}", 
        result.selected_functions.len(),
        result.selected_functions.len() + result.compressed_code.matches("...").count()
    );

    // Show compressed preview
    println!("\nCompressed text preview:");
    println!("{}", "-".repeat(60));
    let lines: Vec<&str> = result.compressed_code.lines().take(12).collect();
    for line in lines {
        if line.len() > 70 {
            println!("{}...", &line[..70]);
        } else {
            println!("{}", line);
        }
    }
    if result.compressed_code.lines().count() > 12 {
        println!("... ({} more lines)", result.compressed_code.lines().count() - 12);
    }
    println!("{}", "-".repeat(60));

    Ok(())
}
