from pathlib import Path

# Пути к индексу и метаданным
INDEX_DIR = "index"
FAISS_INDEX_PATH = "index/faiss.index"
METADATA_PATH = "index/metadata.jsonl"
CHUNKS_PATH = "index/chunks.jsonl"
BM25_WEIGHT = 0.3


# Модель для эмбеддингов (должна совпадать с той, что использовалась при индексации)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Модель Ollama
OLLAMA_MODEL = "llama3.1"

# URL Ollama (если локально — по умолчанию)
OLLAMA_BASE_URL = "http://localhost:11434"


# Token Telegram-бота
TELEGRAM_TOKEN = "8541175039:AAHolCXkEu6ouHPFM1TmN3-tCdFQHNlH85E"
