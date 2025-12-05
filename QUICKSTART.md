# Быстрый старт LongCodeZip-rs

## Установка зависимостей

```bash
# Убедитесь, что Rust установлен
rustc --version

# Если нет - установите через rustup
# https://rustup.rs/
```

## Сборка проекта

```bash
# Перейдите в директорию проекта
cd longcodezip-rs

# Проверка синтаксиса
cargo check

# Сборка в debug режиме
cargo build

# Сборка в release режиме (оптимизированная)
cargo build --release
```

## Запуск тестов

```bash
# Запустить все тесты
cargo test

# Запустить тесты с выводом
cargo test -- --nocapture

# Запустить конкретный тест
cargo test test_compress_python_code
```

## Запуск примера (demo)

### Вариант 1: С переменной окружения

```bash
# Установить API ключ
$env:DEEPSEEK_API_KEY="your-api-key-here"

# Запустить пример
cargo run --example demo
```

### Вариант 2: С предустановленным ключом

Пример уже содержит тестовый API ключ DeepSeek:

```bash
# Просто запустите
cargo run --example demo
```

### Вариант 3: Release сборка (быстрее)

```bash
cargo run --example demo --release
```

## Использование в своем проекте

### 1. Добавьте зависимость

В `Cargo.toml`:

```toml
[dependencies]
longcodezip = { path = "../longcodezip-rs" }
tokio = { version = "1", features = ["full"] }
```

Или после публикации на crates.io:

```toml
[dependencies]
longcodezip = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

### 2. Создайте main.rs

```rust
use longcodezip::{LongCodeZip, CodeLanguage, CompressionConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ваш код здесь
    let provider = ProviderConfig::new(
        "deepseek",
        "https://api.deepseek.com/chat/completions",
        "your-api-key",
        "deepseek-chat",
    );
    
    let config = CompressionConfig::default()
        .with_rate(0.5)
        .with_language(CodeLanguage::Python)
        .with_provider(provider);
    
    let compressor = LongCodeZip::new(config)?;
    
    let code = "your code here";
    let result = compressor.compress_code(code, "your query", "").await?;
    
    println!("Compressed: {}", result.compressed_code);
    
    Ok(())
}
```

### 3. Запустите

```bash
cargo run
```

## Настройка API провайдера

### DeepSeek

```rust
let provider = ProviderConfig::new(
    "deepseek",
    "https://api.deepseek.com/chat/completions",
    "sk-xxxxxx",
    "deepseek-chat",
);
```

Получить API ключ: https://platform.deepseek.com/

### OpenAI

```rust
let provider = ProviderConfig::new(
    "openai",
    "https://api.openai.com/v1/chat/completions",
    "sk-xxxxxx",
    "gpt-4",
);
```

### Другие OpenAI-совместимые провайдеры

```rust
let provider = ProviderConfig::new(
    "custom",
    "https://your-api.com/v1/chat/completions",
    "your-key",
    "your-model",
);
```

## Примеры использования

### Базовый пример

```rust
let code = r#"
def add(a, b):
    return a + b
"#;

let result = compressor.compress_code(code, "", "").await?;
println!("{}", result.compressed_code);
```

### С запросом

```rust
let code = "your long code";
let query = "How does this function work?";

let result = compressor.compress_code(code, query, "").await?;
```

### С инструкцией

```rust
let instruction = "Given the following code, answer the question:";
let result = compressor.compress_code(code, query, instruction).await?;
```

## Отладка

### Включить логирование

```rust
// В начале main.rs
env_logger::init();
```

Затем запустите с переменной окружения:

```bash
$env:RUST_LOG="debug"
cargo run --example demo
```

### Проверить ошибки

```rust
match compressor.compress_code(code, query, instruction).await {
    Ok(result) => println!("Success: {}", result.compression_ratio),
    Err(e) => eprintln!("Error: {}", e),
}
```

## Часто встречающиеся проблемы

### API ключ не работает

Проверьте:
- Ключ скопирован полностью
- Нет лишних пробелов
- Ключ действителен и не истек

### Ошибка компиляции

```bash
# Обновите зависимости
cargo update

# Очистите кеш и пересоберите
cargo clean
cargo build
```

### Медленная компиляция

```bash
# Используйте release сборку только когда нужно
cargo build --release

# Для разработки используйте debug
cargo build
```

## Дополнительная информация

- См. `README.md` для общей документации
- См. `EXAMPLES.md` для подробных примеров
- См. `examples/demo.rs` для рабочего примера
- Документация API: `cargo doc --open`

## Поддержка

Если возникли проблемы:
1. Проверьте документацию
2. Запустите тесты: `cargo test`
3. Проверьте логи с `RUST_LOG=debug`
4. Создайте issue на GitHub
