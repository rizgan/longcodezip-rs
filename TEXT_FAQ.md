# FAQ: Сжатие текста

## Нужно ли менять конфигурацию для текста?

**НЕТ!** Используйте обычный `CompressionConfig`:

```rust
// Та же конфигурация для кода и текста
let config = CompressionConfig::default()
    .with_provider(ProviderConfig::deepseek("your-key"));

let compressor = LongCodeZip::new(config)?;

// Сжатие кода
compressor.compress_code(code, query, "").await?;

// Сжатие текста - НИЧЕГО не меняем!
compressor.compress_text(text, query, "", TextChunkingStrategy::Paragraphs).await?;
```

## Что происходит с полем `language`?

Поле `language` в `CompressionConfig` **игнорируется** при вызове `compress_text()`.
Оно используется только для `compress_code()` для определения синтаксиса функций.

## Какие параметры важны для текста?

Для сжатия текста важны:

1. **`rate`** - коэффициент сжатия (0.0-1.0)
   ```rust
   .with_rate(0.4)  // Оставить 40% токенов
   ```

2. **`provider`** - LLM провайдер для оценки релевантности
   ```rust
   .with_provider(ProviderConfig::deepseek("key"))
   ```

3. **`strategy`** (параметр метода) - как разбивать текст
   ```rust
   TextChunkingStrategy::Paragraphs  // По абзацам
   TextChunkingStrategy::Sentences   // По предложениям
   TextChunkingStrategy::MarkdownSections  // По заголовкам
   ```

## Можно ли использовать один компрессор для кода и текста?

**ДА!** Создайте один `LongCodeZip` и используйте оба метода:

```rust
let compressor = LongCodeZip::new(config)?;

// Сжимаем код
let code_result = compressor
    .compress_code(python_code, "How does login work?", "")
    .await?;

// Сжимаем текст
let text_result = compressor
    .compress_text(article, "What is AI?", "", TextChunkingStrategy::Paragraphs)
    .await?;
```

## Какую стратегию выбрать?

| Тип текста | Рекомендуемая стратегия | Причина |
|-----------|------------------------|---------|
| Научные статьи | `Paragraphs` | Логические блоки по абзацам |
| Техническая документация | `MarkdownSections` | Структура по заголовкам |
| FAQ, Q&A | `Sentences` | Точный контроль по вопросам |
| Логи, данные | `Custom("---")` | Пользовательский разделитель |

## Примеры

### Минимальный пример

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig, text_chunker::TextChunkingStrategy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = CompressionConfig::default()
        .with_provider(ProviderConfig::deepseek("your-key"));
    
    let compressor = LongCodeZip::new(config)?;
    
    let result = compressor.compress_text(
        "Ваш текст...",
        "Ваш вопрос?",
        "",
        TextChunkingStrategy::Paragraphs
    ).await?;
    
    println!("{} → {} токенов", result.original_tokens, result.compressed_tokens);
    Ok(())
}
```

### С настройкой сжатия

```rust
let config = CompressionConfig::default()
    .with_rate(0.3)  // Агрессивное сжатие - 30%
    .with_provider(ProviderConfig::openai("key", "gpt-4"));

let result = compressor.compress_text(
    long_article,
    "Summarize main findings",
    "Focus on methodology",
    TextChunkingStrategy::Paragraphs
).await?;
```

## Смотрите также

- [TEXT_COMPRESSION.md](TEXT_COMPRESSION.md) - Полное руководство
- `examples/simple_text_demo.rs` - Простой пример
- `examples/text_compression_demo.rs` - Все стратегии
