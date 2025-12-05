# Архитектура LongCodeZip-rs

## Обзор

LongCodeZip-rs - это Rust реализация алгоритма компрессии кода для LLM, основанная на двухэтапном подходе.

## Структура проекта

```
longcodezip-rs/
├── src/
│   ├── lib.rs              # Главный модуль библиотеки
│   ├── types.rs            # Определения типов данных
│   ├── error.rs            # Обработка ошибок
│   ├── provider.rs         # Интеграция с LLM API
│   ├── code_splitter.rs    # Парсинг и разбиение кода
│   └── compressor.rs       # Основная логика компрессии
├── examples/
│   └── demo.rs             # Демонстрационный пример
├── tests/
│   └── integration_tests.rs # Интеграционные тесты
└── Cargo.toml              # Конфигурация проекта
```

## Модули

### 1. `types.rs` - Типы данных

**Основные типы:**

- `CodeLanguage` - Поддерживаемые языки программирования
- `ProviderConfig` - Конфигурация API провайдера
- `CompressionConfig` - Настройки компрессии
- `CompressionResult` - Результат компрессии
- `FunctionCompression` - Детали компрессии функции
- `LLMRequest/LLMResponse` - Типы для API запросов

**Ключевые структуры:**

```rust
pub struct CompressionConfig {
    pub rate: f64,                    // Коэффициент компрессии
    pub target_token: i32,            // Целевое количество токенов
    pub language: CodeLanguage,       // Язык кода
    pub provider: ProviderConfig,     // Конфигурация API
    // ...
}

pub struct CompressionResult {
    pub original_code: String,
    pub compressed_code: String,
    pub compression_ratio: f64,
    pub function_compressions: HashMap<usize, FunctionCompression>,
    // ...
}
```

### 2. `error.rs` - Обработка ошибок

Использует `thiserror` для определения типов ошибок:

```rust
pub enum Error {
    ApiError(String),
    RequestError(reqwest::Error),
    JsonError(serde_json::Error),
    ConfigError(String),
    ProcessingError(String),
    InvalidInput(String),
}
```

### 3. `provider.rs` - LLM Провайдер

**Trait `LLMProvider`:**

```rust
#[async_trait]
pub trait LLMProvider: Send + Sync {
    async fn get_token_count(&self, text: &str) -> Result<usize>;
    async fn calculate_relevance(&self, context: &str, query: &str) -> Result<f64>;
    async fn get_completion(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}
```

**Реализация:**

- `OpenAICompatibleProvider` - Работает с DeepSeek, OpenAI, и другими OpenAI-совместимыми API

**Алгоритм расчета релевантности:**

1. Извлечение ключевых слов из запроса (длина > 3)
2. Подсчет совпадений в контексте
3. Расчет score: `match_ratio * 10.0 - ln(token_count) / 10.0`
4. Высокий score = более релевантный контекст

### 4. `code_splitter.rs` - Разбиение кода

**Функция `split_code_by_functions`:**

Использует regex паттерны для разных языков:

```rust
// Python
r"(?m)^[ \t]*(def|class)\s+\w+"

// Rust
r"(?m)^[ \t]*(?:pub\s+)?(?:async\s+)?(?:fn|impl|struct|enum|trait)\s+"

// TypeScript/JavaScript
r"(?m)^[ \t]*(?:export\s+)?(?:async\s+)?(?:function|class)\s+\w+"
```

**Процесс:**

1. Применение regex для поиска функций/классов
2. Разбиение кода на chunks между найденными паттернами
3. Возврат массива фрагментов кода

### 5. `compressor.rs` - Основной компрессор

**Структура `LongCodeZip`:**

```rust
pub struct LongCodeZip {
    config: CompressionConfig,
    provider: Box<dyn LLMProvider>,
}
```

**Алгоритм компрессии `compress_code`:**

```
1. Разбиение кода на функции (code_splitter)
   └─> chunks: Vec<String>

2. Подсчет токенов для каждого chunk
   └─> chunk_tokens: Vec<usize>

3. Определение target_token
   └─> rate * total_tokens или явно заданное значение

4. Ранжирование chunks по релевантности к query
   └─> ranked_chunks: Vec<usize>
   └─> использует provider.calculate_relevance()

5. Выбор chunks в пределах бюджета
   └─> Жадный алгоритм: берем top-ranked пока не превысим target
   └─> Минимум 1 chunk всегда выбирается

6. Сборка compressed_code
   └─> Выбранные chunks + omission markers для пропущенных

7. Формирование финального промпта
   └─> [instruction] + compressed_code + [instruction] + query

8. Возврат CompressionResult
```

**Ключевые методы:**

- `rank_chunks_by_relevance()` - Ранжирование по query
- `select_chunks_within_budget()` - Выбор в пределах бюджета
- `build_prompt()` - Формирование финального промпта

## Алгоритм компрессии (Coarse-grained)

### Шаг 1: Разбиение на функции

```
Исходный код → Regex паттерны → Chunks (функции/классы)
```

### Шаг 2: Подсчет токенов

```
Каждый chunk → Provider.get_token_count() → token_count
```

Текущая реализация: `chars / 4.0` (приближение)

### Шаг 3: Ранжирование

```
Для каждого chunk:
  1. Извлечь ключевые слова из query
  2. Подсчитать совпадения в chunk
  3. Рассчитать score
  4. Сортировать по убыванию score
```

### Шаг 4: Выбор

```
target_token = rate * total_tokens
selected = []
current_tokens = 0

Для каждого ranked_chunk:
  if current_tokens + chunk_tokens <= target_token:
    selected.add(chunk)
    current_tokens += chunk_tokens
  else:
    break
```

### Шаг 5: Сборка

```
result = []
for i, chunk in chunks:
  if i in selected:
    result.append(chunk)
  else:
    result.append("# ... ")
    
compressed_code = "\n\n".join(result)
```

## API взаимодействие

### Запрос к LLM API:

```rust
POST {api_url}
Headers:
  Authorization: Bearer {api_key}
  Content-Type: application/json

Body:
{
  "model": "deepseek-chat",
  "messages": [...],
  "temperature": 0.0,
  "max_tokens": 2048
}
```

### Ответ:

```rust
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

## Асинхронность

Использует `tokio` для асинхронных операций:

- API запросы через `reqwest`
- `async fn` для всех IO операций
- `#[tokio::main]` в примерах

## Производительность

**Оптимизации:**

1. **Простой подсчет токенов**: `chars / 4.0` вместо реального токенизатора
2. **Локальное ранжирование**: Keyword matching вместо API вызовов
3. **Параллельность**: Возможность параллельной обработки chunks (todo)
4. **Кеширование**: Можно добавить кеш для повторяющихся запросов

**Сложность:**

- Разбиение кода: O(n) где n = длина кода
- Ранжирование: O(m * k) где m = количество chunks, k = количество ключевых слов
- Общая: O(n + m*k)

## Расширяемость

### Добавление нового языка:

1. Добавить в `CodeLanguage` enum
2. Добавить regex паттерн в `code_splitter.rs`
3. Добавить comment marker в `CodeLanguage::comment_marker()`

### Добавление нового провайдера:

1. Реализовать `LLMProvider` trait
2. Добавить в `create_provider()` функцию

### Добавление fine-grained компрессии:

1. Добавить модуль `entropy_chunking.rs`
2. Реализовать разбиение на блоки по энтропии
3. Добавить knapsack алгоритм для оптимизации
4. Интегрировать в `compressor.rs`

## Тестирование

**Unit тесты:**

- `code_splitter::tests` - Разбиение кода
- `provider::tests` - Token count, relevance
- `compressor::tests` - Базовая компрессия

**Integration тесты:**

- Полный цикл компрессии
- Разные языки
- С/без query
- Конфигурация

**Запуск:**

```bash
cargo test
cargo test -- --nocapture  # С выводом
```

## Будущие улучшения

1. **Fine-grained компрессия**
   - Entropy-based chunking
   - Knapsack optimization
   - Line-level importance

2. **Улучшенный tokenizer**
   - Интеграция tiktoken или подобного
   - Точный подсчет токенов

3. **Кеширование**
   - In-memory cache для token counts
   - Disk cache для API ответов

4. **Дополнительные провайдеры**
   - Anthropic Claude
   - Local models (llama.cpp)
   - Azure OpenAI

5. **CLI инструмент**
   - Командная строка для компрессии
   - Batch processing
   - Pipeline integration

6. **Метрики**
   - Benchmarking suite
   - Performance profiling
   - Quality metrics

## Литература

- [Оригинальная статья](https://arxiv.org/abs/2510.00446)
- [Python реализация](https://github.com/YerbaPage/LongCodeZip)
- [DeepSeek API Docs](https://platform.deepseek.com/api-docs/)
- [Tokio Async Runtime](https://tokio.rs/)
