# Text Compression Guide

LongCodeZip теперь поддерживает сжатие не только кода, но и обычного текста!

## Основные возможности

### Стратегии разбиения

Библиотека предлагает 4 стратегии для разбиения текста:

#### 1. Paragraphs (Абзацы)
Разбивает текст по пустым строкам. Лучше всего подходит для прозы и статей.

```rust
use longcodezip::{LongCodeZip, CompressionConfig, text_chunker::TextChunkingStrategy};

let compressor = LongCodeZip::new(config)?;
let result = compressor
    .compress_text(text, query, "", TextChunkingStrategy::Paragraphs)
    .await?;
```

**Когда использовать:**
- Научные статьи
- Блоги и новости
- Документация с абзацами
- Книги и эссе

#### 2. Sentences (Предложения)
Разбивает по предложениям (`.`, `?`, `!`). Обеспечивает более детальный контроль.

```rust
let result = compressor
    .compress_text(text, query, "", TextChunkingStrategy::Sentences)
    .await?;
```

**Когда использовать:**
- Когда нужна высокая точность
- Короткие тексты
- Списки определений
- FAQ

#### 3. MarkdownSections (Секции Markdown)
Разбивает по заголовкам (`#`, `##`, `###`). Идеально для структурированных документов.

```rust
let result = compressor
    .compress_text(text, query, "", TextChunkingStrategy::MarkdownSections)
    .await?;
```

**Когда использовать:**
- README файлы
- Техническая документация
- Wiki страницы
- Структурированные руководства

#### 4. Custom (Пользовательский разделитель)
Используйте свой собственный разделитель.

```rust
let result = compressor
    .compress_text(text, query, "", TextChunkingStrategy::Custom("---".to_string()))
    .await?;
```

**Когда использовать:**
- Специальный формат данных
- Логи с разделителями
- CSV или TSV файлы
- Кастомная структура

## Оценка важности

TextChunker автоматически оценивает важность каждого фрагмента:

### Для абзацев:
- ✅ Ключевые слова: "important", "key", "critical", "essential", "fundamental"
- ✅ Вопросы (содержат `?`)
- ✅ Длина абзаца (длинные = больше информации)
- ❌ Слишком короткие абзацы (<50 символов)

### Для предложений:
- ✅ Первое и последнее предложения
- ✅ Вопросы
- ✅ Числа и статистика
- ✅ Переходные слова: "however", "therefore", "thus"
- ✅ Оптимальная длина (50-200 символов)

### Для секций:
- ✅ Уровень заголовка (h1 > h2 > h3)
- ✅ Наличие кодовых блоков (\`\`\`)
- ✅ Наличие списков (`-`, `*`)
- ✅ Длина секции

## Полный пример

```rust
use longcodezip::{
    CodeLanguage, CompressionConfig, LongCodeZip, ProviderConfig,
    text_chunker::TextChunkingStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let article = r#"
# Machine Learning Basics

## Introduction

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

## Key Concepts

### Supervised Learning

In supervised learning, the model learns from labeled data. Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.
"#;

    let provider = ProviderConfig::openai("your-api-key");
    let config = CompressionConfig::default()
        .with_rate(0.4)
        .with_provider(provider);

    let compressor = LongCodeZip::new(config)?;

    // Используем стратегию MarkdownSections для документации
    let result = compressor
        .compress_text(
            article,
            "What is supervised learning?",
            "",
            TextChunkingStrategy::MarkdownSections
        )
        .await?;

    println!("Original: {} tokens", result.original_tokens);
    println!("Compressed: {} tokens", result.compressed_tokens);
    println!("Ratio: {:.1}%", result.compression_ratio * 100.0);
    println!("\n{}", result.compressed_code);

    Ok(())
}
```

## Сравнение стратегий

| Стратегия | Научная статья | История | Техническая документация |
|-----------|---------------|---------|--------------------------|
| Paragraphs | 39.7% ⭐ | 100% ❌ | 45% |
| Sentences | 38.1% ⭐ | 95% ❌ | 42% ⭐ |
| MarkdownSections | N/A | N/A | 28.6% ⭐⭐ |
| Custom | Зависит от данных | Зависит от данных | Зависит от данных |

## Советы по использованию

1. **Для технических документов**: Используйте `MarkdownSections` - наилучшее сжатие
2. **Для научных текстов**: `Paragraphs` или `Sentences` - баланс точности и сжатия
3. **Для неструктурированного текста**: `Paragraphs` безопаснее всего
4. **Для специальных форматов**: `Custom` с вашим разделителем

## API Reference

```rust
/// Сжать текст с использованием выбранной стратегии
pub async fn compress_text(
    &self,
    text: &str,                          // Исходный текст
    query: &str,                         // Запрос пользователя
    instruction: &str,                   // Дополнительные инструкции (опционально)
    strategy: TextChunkingStrategy,      // Стратегия разбиения
) -> Result<CompressionResult>
```

### TextChunkingStrategy

```rust
pub enum TextChunkingStrategy {
    Paragraphs,                  // Разбить по абзацам
    Sentences,                   // Разбить по предложениям
    MarkdownSections,            // Разбить по заголовкам Markdown
    Custom(String),              // Пользовательский разделитель
}
```

## Производительность

Тесты на различных типах текста:

- **Научная статья** (2387 символов): 413 → 164 токена за 0.01s
- **Markdown документ** (300 символов): 70 → 20 токенов за 0.00s
- **Неструктурированная история**: Низкая эффективность (требует больше контекста)

## Ограничения

1. **Художественная литература**: Может работать плохо, так как большинство контента важно для понимания
2. **Очень короткие тексты**: Сжатие может быть неэффективным
3. **Сильно структурированные данные**: Лучше использовать специализированные форматы (JSON, XML)

## Что дальше?

- Попробуйте разные стратегии на вашем тексте
- Настройте `compression_rate` для баланса качества/размера
- Комбинируйте с инструкциями для лучшего результата
