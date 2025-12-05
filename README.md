# LongCodeZip-rs

Rust реализация LongCodeZip - библиотека для компрессии длинного кода для Language Models.

## Описание

LongCodeZip - это двухэтапный метод компрессии кода:

1. **Coarse-grained (Грубая компрессия)**: Ранжирование и выбор функций на основе релевантности к запросу
2. **Fine-grained (Точная компрессия)**: Оптимизация на уровне блоков кода (в разработке)

Этот проект является портом оригинальной Python библиотеки на Rust с поддержкой API провайдеров (DeepSeek, OpenAI и другие).

## Особенности

- ✅ Разбиение кода на функции (Python, Rust, TypeScript, JavaScript, C++, Java, Go)
- ✅ Ранжирование функций по релевантности к запросу
- ✅ Поддержка OpenAI-совместимых API (DeepSeek, OpenAI)
- ✅ Настраиваемый коэффициент компрессии
- ✅ Асинхронная работа с API
- ✅ **Точный tokenizer (tiktoken) для всех моделей**
- ✅ **Fine-grained компрессия**: Entropy chunking + Knapsack оптимизация

## Установка

Добавьте в `Cargo.toml`:

```toml
[dependencies]
longcodezip = "0.1.0"
```

## Быстрый старт

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Настройка провайдера (DeepSeek)
    let provider = ProviderConfig::new(
        "deepseek",
        "https://api.deepseek.com/chat/completions",
        "your-api-key",
        "deepseek-chat",
    );
    
    // Настройка компрессии
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
    
    Ok(())
}
```

## Запуск примеров

### Базовый пример с DeepSeek API:

```bash
# Установить API ключ (опционально, или указать в коде)
export DEEPSEEK_API_KEY="your-key-here"

# Запустить пример
cargo run --example demo
```

### Tokenizer демо:

```bash
cargo run --example tokenizer_demo
```

### Fine-grained компрессия:

```bash
# Сравнение coarse vs fine-grained
cargo run --example fine_grained_demo
```

В примере используется предустановленный API ключ для тестирования:
```
provider: "deepseek"
api_url: "https://api.deepseek.com/chat/completions"
api_key: "key"
model: "deepseek-chat"
```

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

## Документация

- [QUICKSTART.md](QUICKSTART.md) - Быстрый старт
- [EXAMPLES.md](EXAMPLES.md) - Примеры использования
- [TOKENIZER_GUIDE.md](TOKENIZER_GUIDE.md) - Гайд по tokenizer
- [FINE_GRAINED.md](FINE_GRAINED.md) - Fine-grained компрессия
- [ARCHITECTURE.md](ARCHITECTURE.md) - Архитектура
- [ROADMAP.md](ROADMAP.md) - Планы развития

## Roadmap

- [x] Базовая компрессия (coarse-grained) - v0.1.0
- [x] Поддержка DeepSeek API - v0.1.0
- [x] Разбиение кода для разных языков - v0.1.0
- [x] **Точный tokenizer (tiktoken) - v0.3.0** ✨
- [x] **Fine-grained компрессия (энтропийное разбиение) - v0.2.0** ✨
- [x] **Knapsack оптимизация для блоков - v0.2.0** ✨
- [ ] Поддержка других провайдеров - v0.4.0
- [ ] CLI инструмент - v0.5.0
- [ ] Бенчмарки - v0.6.0

См. подробный [ROADMAP.md](ROADMAP.md) для планов развития.

## Сравнение с Python версией

| Функция | Python | Rust |
|---------|--------|------|
| Coarse-grained компрессия | ✅ | ✅ |
| Fine-grained компрессия | ✅ | ✅ |
| Transformers модели | ✅ | ❌ (heuristic) |
| API провайдеры | ⚠️ | ✅ |
| Производительность | 🐌 | 🚀 |
| Типобезопасность | ⚠️ | ✅ |

## Лицензия

MIT License

## Ссылки

- [Оригинальная Python версия](https://github.com/YerbaPage/LongCodeZip)
- [Статья ASE 2025](https://arxiv.org/abs/2510.00446)
- [DeepSeek API](https://platform.deepseek.com/)

## Авторы

Rust реализация на основе оригинального проекта LongCodeZip:
- Оригинал: Yuling Shi, Yichun Qian, Hongyu Zhang и др.
- Rust порт: LongCodeZip Contributors

## Благодарности

Благодарность авторам оригинальной статьи "LongCodeZip: Compress Long Context for Code Language Models" (ASE 2025).
