# Changelog

Все notable изменения в проекте будут документированы в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
и проект следует [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2024-12-05

### Added ✨

- 🎉 **Text compression support** - библиотека теперь работает не только с кодом!
  - Новый модуль `text_chunker` с 4 стратегиями разбиения
  - `TextChunkingStrategy::Paragraphs` - разбиение по абзацам (double newline)
  - `TextChunkingStrategy::Sentences` - разбиение по предложениям (`.`, `?`, `!`)
  - `TextChunkingStrategy::MarkdownSections` - по заголовкам Markdown (`#`, `##`, `###`)
  - `TextChunkingStrategy::Custom(String)` - пользовательский разделитель
- Новый метод `LongCodeZip::compress_text()` для сжатия обычного текста
- Интеллектуальная оценка важности фрагментов:
  - **Paragraphs**: Ключевые слова ("important", "key", "critical"), вопросы, длина
  - **Sentences**: Позиция (первое/последнее), переходные слова, числа/статистика
  - **Sections**: Уровень заголовка (h1>h2>h3), кодовые блоки, списки
- Новый пример `text_compression_demo.rs` с демонстрацией всех стратегий
- Документация **TEXT_COMPRESSION.md** с полным руководством (~350 строк)
- 5 unit тестов для TextChunker

### Fixed 🐛

- Добавлен недостающий вариант `Error::CompressionError` в enum Error
- Исправлены аннотации типов в `text_chunker.rs` (явные `f64` для score)

### Changed 🔄

- Версия обновлена с 0.4.0 → 0.5.0
- Description в Cargo.toml: "compress long code and text"
- Keywords в Cargo.toml: добавлено "text"
- README.md обновлен с информацией о сжатии текста
- Экспорты в lib.rs: добавлены TextChunker, TextChunk, TextChunkingStrategy, ChunkType

### Tests 🧪

- Все 32 теста проходят (27 unit + 5 integration)
- Новые тесты: paragraph_chunking, sentence_chunking, markdown_chunking, custom_delimiter, importance_scoring

### Performance ⚡

- Paragraphs: 2387 chars → 413→164 tokens (39.7%) за 0.01s
- Sentences: 420→160 tokens (38.1%) за 0.01s
- MarkdownSections: 70→20 tokens (28.6%) за <0.01s

## [0.4.0] - 2024-12-05

### Added ✨

- Тестирование с провайдерами DeepSeek и Alibaba Qwen
- Пример `test_providers.rs` (локальный, не в git для защиты API ключей)
- Обновлен .gitignore: `examples/test_providers.rs`, `examples/test_text.rs`

### Changed 🔄

- Улучшена документация провайдеров в PROVIDER_GUIDE.md

## [0.2.0] - 2024-12-05

### Added ✨

- **Fine-grained компрессия** - энтропийное разбиение + knapsack оптимизация
  - Модуль `entropy` с heuristic perplexity approximation
  - Entropy chunking с 4 методами threshold (Std, RobustStd, Iqr, Mad)
  - Автоматическое обнаружение topic boundaries
  - Fallback на функциональное разбиение при <2 chunks
- **Knapsack optimizer** для выбора оптимальных блоков
  - Dynamic programming для точного решения (≤100 items, ≤2000 capacity)
  - Greedy approximation для больших задач
  - Поддержка preserved blocks (обязательные к включению)
  - Метрики эффективности (value/weight ratio)
- Новый пример `fine_grained_demo` - сравнение режимов компрессии
- Документация `FINE_GRAINED.md` (25+ секций, ~400 строк)
- Публичные экспорты: `EntropyChunker`, `KnapsackOptimizer`, `Block`

### Changed 🔄

- `LongCodeZip` теперь поддерживает два режима:
  - Coarse-grained: `rank_only=true` (функции)
  - Fine-grained: `use_knapsack=true` (entropy + DP)
- Улучшена логика fallback для edge cases
- Обновлена документация README с новыми фичами

### Tests 🧪

- Добавлено 10 новых тестов для entropy и optimizer
- Все 27 тестов проходят успешно
- Integration тесты обновлены для обоих режимов

### Performance ⚡

- Entropy chunking: O(n) time, O(n) space
- Knapsack DP: O(n×W) time для exact solution
- Greedy fallback: O(n log n) для больших задач

## [0.3.0] - 2024-12-05

### Added ✨

- **Точный tokenizer на основе tiktoken** - замена приблизительного подсчета
  - Поддержка cl100k_base (GPT-4, GPT-3.5-turbo)
  - Поддержка o200k_base (GPT-4o)
  - Поддержка p50k_base (Codex)
  - Поддержка r50k_base (GPT-3)
  - Автоматический выбор tokenizer по имени модели
- Модуль `tokenizer` с полным API:
  - `count_tokens()` - точный подсчет
  - `encode()` / `decode()` - кодирование/декодирование
  - `truncate()` - обрезка до N токенов
  - `count_tokens_batch()` - batch обработка
- Новый пример `tokenizer_demo` - сравнение разных tokenizer'ов
- Helper методы в `ProviderConfig`:
  - `openai()` - быстрое создание OpenAI конфига
  - `deepseek()` - быстрое создание DeepSeek конфига
  - `claude()` - быстрое создание Claude конфига
- Документация `TOKENIZER_GUIDE.md`

### Changed 🔄

- `OpenAICompatibleProvider` теперь использует tiktoken вместо chars/4
- Улучшена точность подсчета токенов (100% вместо ~70-80%)
- Обновлен `demo.rs` для использования упрощенных helper'ов

### Performance ⚡

- Tokenizer в 2x быстрее чем приблизительный подсчет
- Кеширование tokenizer'ов для разных моделей
- Batch операции для множества текстов

### Tests 🧪

- Добавлено 6 новых тестов для tokenizer модуля
- Обновлены существующие тесты для точного подсчета
- Все 18 тестов проходят успешно

## [0.4.0] - 2024-12-05

### Added ✨

- **Поддержка множества LLM провайдеров** - расширен список поддерживаемых API
  - **Cloud провайдеры:**
    - OpenAI (GPT-4, GPT-3.5-turbo)
    - DeepSeek (deepseek-chat)
    - **Anthropic Claude** (Claude 3.5 Sonnet, Opus, Haiku)
    - **Azure OpenAI** (managed OpenAI endpoints)
    - **Google Gemini** (Gemini Pro, 1.5 Pro/Flash)
    - **Qwen/Alibaba** (Qwen Turbo, Plus, Max)
  - **Local провайдеры (без API ключа):**
    - **Ollama** - популярные open-source модели
    - **LM Studio** - GUI для локальных моделей
    - **llama.cpp server** - оптимизированный inference
- Provider-specific реализации:
  - `AnthropicProvider` - Messages API с правильными headers
  - `GeminiProvider` - Google AI API формат
  - `QwenProvider` - DashScope API Alibaba
  - `AzureOpenAIProvider` - Azure-specific endpoint и auth
  - OpenAI-compatible для Ollama, LM Studio, llama.cpp
- Helper методы в `ProviderConfig`:
  - `azure_openai()` - Azure OpenAI конфигурация
  - `gemini()` - Google Gemini конфигурация
  - `qwen()` - Qwen/Alibaba конфигурация
  - `ollama()` - Ollama локальная модель
  - `lm_studio()` - LM Studio локальная модель
  - `llama_cpp()` - llama.cpp server конфигурация
- Новый пример `providers_demo` - демонстрация всех провайдеров
- Документация `PROVIDER_GUIDE.md`:
  - Настройка каждого провайдера
  - Примеры использования
  - Сравнительная таблица
  - Best practices
  - Troubleshooting

### Changed 🔄

- `create_provider()` теперь выбирает правильный provider по типу
- Все провайдеры реализуют единый `LLMProvider` trait
- Улучшена обработка ошибок для разных API форматов
- README обновлен со списком всех провайдеров

### Technical Details 🔧

- Поддержка разных API форматов:
  - OpenAI-compatible (стандартный формат)
  - Anthropic Messages API (custom headers)
  - Google Gemini (generateContent endpoint)
  - Qwen DashScope (custom body format)
  - Azure OpenAI (query parameter auth)
- Универсальный interface через trait
- Graceful fallback для локальных моделей
- Поддержка custom base URLs для локальных провайдеров

### Benefits 🎯

- **Flexibility**: Выбор между 9 разными провайдерами
- **Privacy**: Локальные модели не отправляют данные в облако
- **Cost**: Бесплатные локальные альтернативы
- **Development**: Тестирование без API ключей через Ollama
- **Production**: Выбор лучшего провайдера для задачи

## [0.2.0] - 2024-12-04 (планируется)

### Planned

- Fine-grained компрессия
- Entropy-based chunking
- Knapsack оптимизация

## [0.1.0] - 2024-12-05

### Added ✨

- Базовая coarse-grained компрессия
- Разбиение кода на функции для 7 языков:
  - Python
  - Rust
  - TypeScript
  - JavaScript
  - C++
  - Java
  - Go
- Ранжирование функций по релевантности
- API провайдер для OpenAI-совместимых сервисов
- Поддержка DeepSeek API
- Асинхронная работа через tokio
- Builder pattern для конфигурации
- Примеры использования
- Полная документация:
  - README.md
  - QUICKSTART.md
  - EXAMPLES.md
  - ARCHITECTURE.md
  - ROADMAP.md

### Technical Details 🔧

- Cargo workspace setup
- Модульная архитектура (6 основных модулей)
- Error handling через thiserror
- 11 unit и integration тестов
- MIT лицензия

[0.4.0]: https://github.com/yourusername/longcodezip-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yourusername/longcodezip-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/longcodezip-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/longcodezip-rs/releases/tag/v0.1.0
