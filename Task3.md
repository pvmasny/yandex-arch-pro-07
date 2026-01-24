# Векторный индекс базы знаний (Звёздные Войны)


## Модель эмбеддингов
- **Название**: `all-MiniLM-L6-v2`
- **Размер эмбеддинга**: 384 (float32)

## База знаний
- **Путь**: `./Task2/knowledge_base/modified_md`
- **Структура**:
  - `characters/` — персонажи
  - `races/` — расы
  - `states/` — государства
  - `transports/` — транспорт
  - `weapons/` — оружие
- **Формат**: `.md` (Markdown)
- **Общее количество файлов**: 30

## Статистика индекса
- **Количество чанков**: 65
- **Размчер чанка**: 256 символов
- **Размер индекса (FAISS)**: 97,5 КБ


## Время генерации
- **Загрузка и разбиение**: 4.193 сек

# Пример запроса к индексу
```bash
python .\search.py --query="Какой был характер Зарта Кэлакса?" --rerank_cross_encoder cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

## Ответ
```json
[
  {
    "rank": 1,
    "score": 7.716066837310791,
    "id": "Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md::chunk_1",
    "path": "Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md",
    "title": "Зарт Кэлакс.md",
    "chunk_index": 1,
    "snippet": "[Зарт Кэлакс.md] Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md (chunk #1)"
  },
  {
    "rank": 2,
    "score": 5.629749298095703,
    "id": "Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md::chunk_0",
    "path": "Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md",
    "title": "Зарт Кэлакс.md",
    "chunk_index": 0,
    "snippet": "[Зарт Кэлакс.md] Task2/knowledge_base/modified_md/characters/Зарт Кэлакс.md (chunk #0)"
  },
  {
    "rank": 3,
    "score": -0.004993467126041651,
    "id": "Task2/knowledge_base/modified_md/characters/Калрантис.md::chunk_0",
    "path": "Task2/knowledge_base/modified_md/characters/Калрантис.md",
    "title": "Калрантис.md",
    "chunk_index": 0,
    "snippet": "[Калрантис.md] Task2/knowledge_base/modified_md/characters/Калрантис.md (chunk #0)"
  },
  {
    "rank": 4,
    "score": -1.473738431930542,
    "id": "Task2/knowledge_base/modified_md/characters/Веларион.md::chunk_1",
    "path": "Task2/knowledge_base/modified_md/characters/Веларион.md",
    "title": "Веларион.md",
    "chunk_index": 1,
    "snippet": "[Веларион.md] Task2/knowledge_base/modified_md/characters/Веларион.md (chunk #1)"
  },
  {
    "rank": 5,
    "score": -1.5499675273895264,
    "id": "Task2/knowledge_base/modified_md/characters/Веларион.md::chunk_2",
    "path": "Task2/knowledge_base/modified_md/characters/Веларион.md",
    "title": "Веларион.md",
    "chunk_index": 2,
    "snippet": "[Веларион.md] Task2/knowledge_base/modified_md/characters/Веларион.md (chunk #2)"
  }
]
```