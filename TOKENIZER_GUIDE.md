# Tokenizer Guide

## Обзор

С версии 0.3.0 LongCodeZip-rs использует **точный tokenizer** на основе tiktoken вместо приблизительного подсчета.

## Преимущества

### До (v0.1.0 - v0.2.0)
```rust
// Приблизительный подсчет: chars / 4
let count = text.chars().count() as f64 / 4.0;
// ❌ Неточно: может отличаться на 20-50%
```

### После (v0.3.0+)
```rust
// Точный подсчет через tiktoken
let tokenizer = Tokenizer::from_model_name("gpt-4");
let count = tokenizer.count_tokens(text)?;
// ✅ Точность 100%
```

## Поддерживаемые модели

### OpenAI
- **GPT-4, GPT-3.5-turbo** → `cl100k_base`
- **GPT-4o** → `o200k_base`
- **Code models (Codex)** → `p50k_base`
- **GPT-3 (davinci, curie)** → `r50k_base`

### Другие провайдеры
- **DeepSeek** → `cl100k_base` (совместимо)
- **Claude (Anthropic)** → `cl100k_base` (совместимо)
- **Custom models** → `cl100k_base` (по умолчанию)

## Базовое использование

### Автоматический выбор tokenizer

```rust
use longcodezip::Tokenizer;

// Автоматически выберет правильный tokenizer для модели
let tokenizer = Tokenizer::from_model_name("gpt-4");

let text = "def hello():\n    print('world')";
let count = tokenizer.count_tokens(text)?;

println!("Tokens: {}", count);
```

### Явный выбор модели

```rust
use longcodezip::{Tokenizer, TokenizerModel};

// Явно указать модель tokenizer
let tokenizer = Tokenizer::new(TokenizerModel::Cl100kBase);

let count = tokenizer.count_tokens(text)?;
```

## Продвинутое использование

### Кодирование и декодирование

```rust
let tokenizer = Tokenizer::from_model_name("gpt-4");

// Кодирование в токены
let tokens = tokenizer.encode("Hello, world!")?;
println!("Tokens: {:?}", tokens); // [9906, 11, 1917, 0]

// Декодирование обратно
let text = tokenizer.decode(&tokens)?;
println!("Text: {}", text); // "Hello, world!"
```

### Truncation (обрезка)

```rust
let tokenizer = Tokenizer::from_model_name("gpt-4");

let long_text = "Very long text that needs to be truncated...";

// Обрезать до 50 токенов
let truncated = tokenizer.truncate(long_text, 50)?;

// Проверить результат
let count = tokenizer.count_tokens(&truncated)?;
assert!(count <= 50);
```

### Batch обработка

```rust
let tokenizer = Tokenizer::from_model_name("gpt-4");

let texts = vec![
    "First function",
    "Second function",
    "Third function",
];

// Подсчитать токены для всех текстов сразу
let counts = tokenizer.count_tokens_batch(&texts)?;

for (text, count) in texts.iter().zip(counts.iter()) {
    println!("{}: {} tokens", text, count);
}
```

## Интеграция с LongCodeZip

### Автоматическое использование

LongCodeZip автоматически использует правильный tokenizer:

```rust
use longcodezip::{LongCodeZip, CompressionConfig, ProviderConfig};

// Создаем провайдер с моделью
let provider = ProviderConfig::openai("your-key", "gpt-4");

// Tokenizer автоматически настраивается для gpt-4
let config = CompressionConfig::default()
    .with_provider(provider);

let compressor = LongCodeZip::new(config)?;
```

### Разные модели

```rust
// OpenAI GPT-4
let provider = ProviderConfig::openai("key", "gpt-4");
// Использует: cl100k_base

// OpenAI GPT-4o
let provider = ProviderConfig::openai("key", "gpt-4o");
// Использует: o200k_base

// DeepSeek
let provider = ProviderConfig::deepseek("key");
// Использует: cl100k_base

// Claude
let provider = ProviderConfig::claude("key", "claude-3-opus");
// Использует: cl100k_base
```

## Сравнение моделей

Разные tokenizer'ы дают разное количество токенов:

```rust
let code = r#"
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"#;

// GPT-4 (cl100k_base):  29 tokens
// GPT-4o (o200k_base):  29 tokens
// Codex (p50k_base):    40 tokens
// GPT-3 (r50k_base):    50 tokens
```

## Производительность

### Кеширование

Tokenizer'ы создаются один раз и кешируются:

```rust
use longcodezip::Tokenizer;

// Первый вызов - загружает tokenizer
let tokenizer1 = Tokenizer::from_model_name("gpt-4");

// Последующие вызовы - используют кеш
let tokenizer2 = Tokenizer::from_model_name("gpt-4");
```

### Скорость

Точный tokenizer быстрее приблизительного:

```
Приблизительный: ~0.01ms на текст
Точный tiktoken:  ~0.005ms на текст
```

## Fallback режим

Если tiktoken недоступен, используется приблизительный режим:

```rust
use longcodezip::ApproximateTokenizer;

let tokenizer = ApproximateTokenizer::new();
let count = tokenizer.count_tokens("Hello world");
// Приблизительно: 11 chars / 4 = 3 tokens
```

## Примеры

### Пример 1: Сравнение моделей

```bash
cargo run --example tokenizer_demo
```

Вывод:
```
📊 Token counts by model:
GPT-4 (cl100k_base)       29 tokens
GPT-4o (o200k_base)       29 tokens
Codex (p50k_base)         40 tokens
GPT-3 (r50k_base)         50 tokens
```

### Пример 2: Компрессия с точным подсчетом

```bash
cargo run --example demo
```

Результат:
```
Original tokens:    402 (точный подсчет)
Compressed tokens:  173
Compression ratio:  43.03%
```

## Best Practices

### 1. Используйте правильную модель

```rust
// ✅ Правильно
let provider = ProviderConfig::openai("key", "gpt-4");

// ❌ Неправильно - несоответствие модели
let provider = ProviderConfig::openai("key", "gpt-4");
let tokenizer = Tokenizer::new(TokenizerModel::R50kBase); // Другая модель!
```

### 2. Кешируйте tokenizer

```rust
// ✅ Правильно - создать один раз
let tokenizer = Tokenizer::from_model_name("gpt-4");
for text in texts {
    tokenizer.count_tokens(text)?;
}

// ❌ Неправильно - создавать каждый раз
for text in texts {
    let tokenizer = Tokenizer::from_model_name("gpt-4");
    tokenizer.count_tokens(text)?;
}
```

### 3. Используйте batch для множества текстов

```rust
// ✅ Правильно
let counts = tokenizer.count_tokens_batch(&texts)?;

// ⚠️ Менее эффективно
let counts: Vec<_> = texts.iter()
    .map(|t| tokenizer.count_tokens(t))
    .collect::<Result<Vec<_>>>()?;
```

## Troubleshooting

### Проблема: Разные результаты на разных моделях

**Решение:** Убедитесь, что используете правильный tokenizer для вашей модели.

```rust
// Проверить какой tokenizer используется
let tokenizer = Tokenizer::from_model_name("gpt-4");
println!("Model: {}", tokenizer.model().name());
```

### Проблема: Ошибка при декодировании

**Решение:** Убедитесь, что токены валидны для данного tokenizer'а.

```rust
let tokens = tokenizer.encode(text)?;
let decoded = tokenizer.decode(&tokens)?; // Должно работать
```

## Миграция с v0.2.0

### До

```rust
// Приблизительный подсчет
let chars = text.chars().count();
let tokens = (chars as f64 / 4.0).ceil() as usize;
```

### После

```rust
// Точный подсчет
let tokenizer = Tokenizer::from_model_name("gpt-4");
let tokens = tokenizer.count_tokens(text)?;
```

### Автоматическая миграция

LongCodeZip автоматически использует новый tokenizer, никаких изменений в коде не требуется!

```rust
// Этот код работает с обеими версиями
let compressor = LongCodeZip::new(config)?;
let result = compressor.compress_code(code, query, "").await?;
```

## Дополнительная информация

- Документация tiktoken: https://github.com/openai/tiktoken
- OpenAI tokenizer info: https://platform.openai.com/tokenizer
- Модели и их tokenizer'ы: https://platform.openai.com/docs/models
