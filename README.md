# LongCodeZip-rs

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)](https://github.com/rizgan/longcodezip-rs)

Rust реализация LongCodeZip - библиотека для интеллектуальной компрессии кода и текста для Language Models.

## 🎯 Описание

LongCodeZip - это двухэтапный метод компрессии, обеспечивающий **до 97% уменьшения токенов** при сохранении важной информации:

1. **Coarse-grained (Грубая компрессия)**: Ранжирование и выбор функций на основе релевантности к запросу
2. **Fine-grained (Точная компрессия)**: Entropy-based разбиение + Knapsack оптимизация на уровне блоков кода

Этот проект является портом оригинальной Python библиотеки на Rust с расширенной функциональностью и производительностью.

## ✨ Особенности

### Версия 0.6.0 (Текущая)

- ✅ **Интеллектуальная компрессия кода** для 7+ языков программирования
- ✅ **Сжатие обычного текста** с 4 стратегиями разбиения (Paragraphs, Sentences, Markdown, Custom)
- ✅ **9 LLM провайдеров:**
  - **Cloud**: OpenAI, DeepSeek, Anthropic Claude, Azure OpenAI, Google Gemini, Qwen (Alibaba)
  - **Local**: Ollama, LM Studio, llama.cpp (без API ключей!)
- ✅ **Fine-grained компрессия**: Entropy chunking + Knapsack оптимизация
- ✅ **Точный tokenizer (tiktoken)** для всех моделей (GPT-4, GPT-4o, Claude, DeepSeek)
- ✅ **🚀 Кеширование LLM responses** - избегайте повторных API вызовов (50-100x ускорение!)
- ✅ **⚡ Параллельная обработка** - ускорение в 2-4x на multi-core системах
- ✅ **Настраиваемый коэффициент компрессии** (0.0-1.0)
- ✅ **Асинхронная работа** через tokio

### Результаты тестирования

**Протестировано на реальных LLM** (DeepSeek, Qwen):
- 📊 **Средняя компрессия:** 23.5% (экономия 77% токенов)
- 🏆 **Лучший результат:** 97.6% сжатие (Python файл: 1604 → 38 токенов)
- ⚡ **Производительность:** 1-2 ms с кешем, 8 ms без кеша
- 💰 **Экономия:** ~$0.45 на 1M токенов

**Подробные результаты:** См. [test_files/LLM_TEST_RESULTS.md](test_files/LLM_TEST_RESULTS.md)

## 📚 Документация

- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Быстрое начало работы
- 📊 **[COMPRESSION_BENCHMARKS.md](COMPRESSION_BENCHMARKS.md)** - Детальные бенчмарки и результаты тестов
- 💡 **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - 15+ примеров использования (с LLM и без)
- 🔧 **[PROVIDER_GUIDE.md](PROVIDER_GUIDE.md)** - Настройка LLM провайдеров
- ⚡ **[CACHE_PARALLEL.md](CACHE_PARALLEL.md)** - Кеширование и параллельная обработка
- 📝 **[TEXT_COMPRESSION.md](TEXT_COMPRESSION.md)** - Сжатие обычного текста
- 🏗️ **[ARCHITECTURE.md](ARCHITECTURE.md)** - Архитектура проекта

## 🚀 Быстрый старт

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Настройка провайдера (DeepSeek)
    let provider = ProviderConfig::deepseek("your-api-key");
    
    // Настройка компрессии (кеширование и параллельная обработка включены по умолчанию)
    let config = CompressionConfig::default()
        .with_rate(0.5)  // Оставить 50% токенов
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    // Создание компрессора
    let compressor = LongCodeZip::new(config)?;
    
    // Компрессия кода
    let code = "your code here";
    let query = "How does this work?";
    let result = compressor.compress_code(code, query, "").await?;
    
    println!("Compressed: {}", result.compressed_code);
    println!("Ratio: {:.2}%", result.compression_ratio * 100.0);
    
    // Просмотр статистики кеша
    let stats = compressor.cache_stats();
    println!("Cache entries: {}", stats.valid_entries);
    
    Ok(())
}
```

## 📦 Установка

Добавьте в `Cargo.toml`:

```toml
[dependencies]
longcodezip = "0.6.0"
tokio = { version = "1.0", features = ["full"] }
```

### Настройка производительности

```rust
// Максимальная производительность
let config = CompressionConfig::default()
    .with_cache(true)           // Кеширование включено (по умолчанию)
    .with_parallel(true)        // Параллельная обработка (по умолчанию)
    .with_parallel_threads(8);  // Использовать 8 потоков (0 = auto)

// Без кеша (для тестирования)
let config = CompressionConfig::default()
    .with_cache(false);

// Последовательная обработка (для отладки)
let config = CompressionConfig::default()
    .with_parallel(false);
```

## 🎮 Запуск примеров

### Базовый пример с DeepSeek API:

```bash
# Установить API ключ (опционально, или указать в коде)
export DEEPSEEK_API_KEY="your-key-here"

# Запустить пример
cargo run --example demo --release
```

### Статистика тестовых файлов:

```bash
# Анализ токенов для 10 различных файлов (без LLM вызовов)
cargo run --example test_files_stats --release
```

**Показывает:**
- Количество токенов для каждого файла
- Сравнение cl100k vs o200k tokenizer
- Прогнозы компрессии (30%, 50%, 70%)
- Статистику по языкам программирования

### Benchmark с реальными LLM:

```bash
# Тестирование с DeepSeek и Qwen
cargo run --example benchmark_llm --release
```

**Показывает:**
- Реальную компрессию с LLM
- Сравнение провайдеров
- Производительность с кешем
- Детальную статистику

### Другие примеры:

```bash
# Tokenizer демо
cargo run --example tokenizer_demo --release

# Fine-grained компрессия
cargo run --example fine_grained_demo --release

# Демонстрация провайдеров
cargo run --example providers_demo --release

# Сжатие текста (простой пример)
cargo run --example simple_text_demo --release

# Все стратегии разбиения текста
cargo run --example text_compression_demo --release

# Кеширование и параллельная обработка
cargo run --example cache_parallel_demo --release
```

**💡 Важно**: Для текста НЕ нужно менять конфигурацию! Поле `language` игнорируется.

**📖 Документация**: 
- [TEXT_FAQ.md](TEXT_FAQ.md) - Частые вопросы и быстрый старт
- [TEXT_COMPRESSION.md](TEXT_COMPRESSION.md) - Подробное руководство
- [CACHE_PARALLEL.md](CACHE_PARALLEL.md) - Руководство по кешированию и параллельной обработке

В примере используется предустановленный API ключ для тестирования:
```
provider: "deepseek"
api_url: "https://api.deepseek.com/chat/completions"
api_key: "your-api-key"
model: "deepseek-chat"
```

## Поддерживаемые провайдеры

### Cloud провайдеры

```rust
// OpenAI
let provider = ProviderConfig::openai("your-key", "gpt-4");

// DeepSeek
let provider = ProviderConfig::deepseek("your-key");

// Anthropic Claude
let provider = ProviderConfig::claude("your-key", "claude-3-5-sonnet-20241022");

// Azure OpenAI
let provider = ProviderConfig::azure_openai("your-key", "resource", "deployment", "2024-02-01");

// Google Gemini
let provider = ProviderConfig::gemini("your-key", "gemini-pro");

// Qwen (Alibaba)
let provider = ProviderConfig::qwen("your-key", "qwen-turbo");
```

### Local модели (без API ключа)

```rust
// Ollama
let provider = ProviderConfig::ollama("llama3.1:8b", None);

// LM Studio
let provider = ProviderConfig::lm_studio("local-model", None);

// llama.cpp server
let provider = ProviderConfig::llama_cpp("model-name", Some("http://localhost:8080"));
```

**📖 Подробная документация:** См. [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md)

## Конфигурация

### ProviderConfig

```rust
let provider = ProviderConfig {
    provider: "deepseek".to_string(),
    api_url: "https://api.deepseek.com/chat/completions".to_string(),
    api_key: "your-key".to_string(),
    model: "deepseek-chat".to_string(),
    temperature: 0.0,
    max_tokens: 2048,
};
```

### CompressionConfig

```rust
let config = CompressionConfig {
    rate: 0.5,                    // Коэффициент компрессии (0.0-1.0)
    target_token: -1,              // Целевое количество токенов (-1 = auto)
    language: CodeLanguage::Python, // Язык программирования
    rank_only: true,               // Только coarse-grained компрессия
    // ... другие опции
};
```

## Поддерживаемые языки

- Python
- Rust
- TypeScript
- JavaScript
- C++
- Java
- Go

## API

### `LongCodeZip::new(config: CompressionConfig)`

Создает новый компрессор с заданной конфигурацией.

### `compress_code(&self, code: &str, query: &str, instruction: &str)`

Компрессирует код с учетом запроса и инструкции.

**Параметры:**
- `code`: Исходный код для компрессии
- `query`: Запрос для определения релевантности

### `compress_text(&self, text: &str, query: &str, instruction: &str, strategy: TextChunkingStrategy)` 🆕

Компрессирует обычный текст (не код) с использованием выбранной стратегии разбиения.

**Параметры:**
- `text`: Исходный текст для компрессии
- `query`: Запрос для определения релевантности
- `instruction`: Дополнительные инструкции (опционально)
- `strategy`: Стратегия разбиения (Paragraphs, Sentences, MarkdownSections, Custom)

**Пример:**
```rust
use longcodezip::text_chunker::TextChunkingStrategy;

let result = compressor
    .compress_text(article, "What is AI?", "", TextChunkingStrategy::Paragraphs)
    .await?;
```

**📖 Подробности:** См. [TEXT_COMPRESSION.md](TEXT_COMPRESSION.md)
- `instruction`: Дополнительная инструкция для промпта

**Возвращает:** `CompressionResult` с сжатым кодом и статистикой.

## Структура проекта

```
longcodezip-rs/
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── types.rs            # Типы данных
│   ├── error.rs            # Обработка ошибок
│   ├── provider.rs         # LLM провайдеры
│   ├── code_splitter.rs    # Разбиение кода
│   └── compressor.rs       # Основной компрессор
├── examples/
│   └── demo.rs             # Пример использования
├── Cargo.toml
└── README.md
```

## Тестирование

```bash
# Запустить все тесты
cargo test

# Запустить тесты с выводом
cargo test -- --nocapture

# Запустить конкретный тест
cargo test test_split_python_code
```

## Производительность

- Асинхронная работа с API через `tokio`
- **Точный подсчет токенов через tiktoken**
- Автоматический выбор tokenizer для разных моделей (GPT-4, GPT-4o, DeepSeek, Claude)
- Эффективное разбиение кода через regex

## Примеры

### Базовый пример

```bash
cargo run --example demo
```

### Tokenizer сравнение

```bash
cargo run --example tokenizer_demo
```

Показывает разницу между разными tokenizer'ами для одного и того же кода.

### Fine-grained vs Coarse-grained

```bash
cargo run --example fine_grained_demo
```

Сравнивает две стратегии компрессии и показывает различия.

## 📚 Документация

- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт
- [EXAMPLES.md](EXAMPLES.md) - Примеры использования
- [TEXT_FAQ.md](TEXT_FAQ.md) - FAQ по сжатию текста
- [TEXT_COMPRESSION.md](TEXT_COMPRESSION.md) - Подробное руководство по тексту
- [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md) - Гайд по провайдерам
- [TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md) - Гайд по tokenizer
- [FINE_GRAINED.md](FINE_GRAINED.md) - Fine-grained компрессия
- [CACHE_PARALLEL.md](CACHE_PARALLEL.md) - Кеширование и параллельная обработка
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архитектура
- [ROADMAP.md](ROADMAP.md) - Планы развития
- [ROADMAP_STATUS.md](ROADMAP_STATUS.md) - Статус выполнения roadmap

### Тестовые данные

- [test_files/STATISTICS_REPORT.md](test_files/STATISTICS_REPORT.md) - Статистика по тестовым файлам
- [test_files/LLM_TEST_RESULTS.md](test_files/LLM_TEST_RESULTS.md) - Результаты тестирования с LLM

## 🗺️ Roadmap

### ✅ Завершено

- [x] **v0.1.0** - Базовая компрессия (coarse-grained)
- [x] **v0.2.0** - Fine-grained компрессия (entropy + knapsack)
- [x] **v0.3.0** - Точный tokenizer (tiktoken)
- [x] **v0.4.0** - Множество LLM провайдеров (9 провайдеров)
- [x] **v0.5.0** - Сжатие текста (4 стратегии)
- [x] **v0.6.0** - Кеширование и параллельная обработка

### 🚧 В разработке

- [ ] **v0.7.0** - CLI инструмент
- [ ] **v0.8.0** - Улучшенные алгоритмы компрессии
- [ ] **v0.9.0** - REST API сервер
- [ ] **v1.0.0** - Стабильный релиз + IDE интеграция

См. подробный [ROADMAP.md](ROADMAP.md) и [ROADMAP_STATUS.md](ROADMAP_STATUS.md) для планов развития.

## 📊 Сравнение с Python версией

| Функция | Python | Rust | Преимущество |
|---------|--------|------|--------------|
| Coarse-grained компрессия | ✅ | ✅ | Равно |
| Fine-grained компрессия | ✅ | ✅ | Равно |
| Entropy chunking | ✅ | ✅ | Равно |
| Knapsack оптимизация | ✅ | ✅ | Равно |
| Точный tokenizer | ❌ | ✅ | **Rust** |
| Кеширование LLM | ❌ | ✅ | **Rust** |
| Параллельная обработка | ❌ | ✅ | **Rust** |
| LLM провайдеры | 2-3 | 9 | **Rust** |
| Сжатие текста | ❌ | ✅ | **Rust** |
| Производительность | 🐌 | 🚀 | **Rust 10-100x** |
| Типобезопасность | ⚠️ | ✅ | **Rust** |
| Потребление памяти | Высокое | Низкое | **Rust** |

**Вывод:** Rust версия предоставляет все функции Python версии плюс множество улучшений.

## Лицензия

MIT License

## 📖 Дополнительная документация

### Руководства
- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт
- [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - Практические примеры
- [PROVIDER_GUIDE.md](PROVIDER_GUIDE.md) - Настройка провайдеров
- [PROVIDER_QUICK_REF.md](PROVIDER_QUICK_REF.md) - Краткий справочник

### Технические детали
- [COMPRESSION_BENCHMARKS.md](COMPRESSION_BENCHMARKS.md) - Бенчмарки и метрики
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архитектура
- [CACHE_PARALLEL.md](CACHE_PARALLEL.md) - Оптимизация производительности
- [FINE_GRAINED.md](FINE_GRAINED.md) - Fine-grained компрессия
- [TEXT_COMPRESSION.md](TEXT_COMPRESSION.md) - Сжатие текста
- [TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md) - Работа с токенизаторами

### История разработки
- [ROADMAP.md](ROADMAP.md) - План развития
- [ROADMAP_STATUS.md](ROADMAP_STATUS.md) - Статус задач
- [CHANGELOG.md](CHANGELOG.md) - История изменений
- [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) - Отчеты о реализации

## Ссылки

- [Оригинальная Python версия](https://github.com/YerbaPage/LongCodeZip)
- [Статья ASE 2025](https://arxiv.org/abs/2510.00446)
- [DeepSeek API](https://platform.deepseek.com/)
- [GitHub Repository](https://github.com/rizgan/longcodezip-rs)

## Авторы

Rust реализация на основе оригинального проекта LongCodeZip:
- Оригинал: Yuling Shi, Yichun Qian, Hongyu Zhang и др.
- Rust порт: LongCodeZip Contributors

## Благодарности

Благодарность авторам оригинальной статьи "LongCodeZip: Compress Long Context for Code Language Models" (ASE 2025).
