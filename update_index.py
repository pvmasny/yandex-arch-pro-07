#!/usr/bin/env python3

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Конфигурация
KB_PATH = Path("Task2/knowledge_base")
INDEX_PATH = Path("index/index.faiss")
METADATA_PATH = Path("index/metadata.jsonl")
CHUNKS_PATH = Path("index/chunks.jsonl")
LOG_PATH = Path("index/update_log.txt")
LAST_UPDATE_PATH = Path(".last_update.txt")

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 256  # токенов
OVERLAP = 64

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_last_update() -> float:
    if LAST_UPDATE_PATH.exists():
        return float(LAST_UPDATE_PATH.read_text())
    return 0.0

def save_last_update(timestamp: float):
    LAST_UPDATE_PATH.write_text(str(timestamp))

def read_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def generate_embeddings(chunks: List[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)


def update_index(new_embeddings: np.ndarray, index: faiss.Index, metadata: List[Dict]):
    index.add(new_embeddings.astype(np.float32))
    with open(METADATA_PATH, "a", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def main():
    start_time = time.time()
    logger.info("Начало обновления индекса")

    # Загрузка последней даты обновления
    last_update = load_last_update()
    now = time.time()

    # Сбор новых/изменённых файлов
    new_files = []
    for file_path in KB_PATH.rglob("*.*"):
        if file_path.is_file() and file_path.stat().st_mtime > last_update:
            new_files.append(file_path)

    logger.info(f"Найдено новых/изменённых файлов: {len(new_files)}")

    if not new_files:
        logger.info("Нет новых данных. Завершение.")
        return

    # Инициализация модели и индекса
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexFlatL2(dim)

    new_chunks = []
    new_metadata = []

    # Обработка файлов
    for file_path in new_files:
        try:
            text = read_file(file_path)
            chunks = split_into_chunks(text, CHUNK_SIZE, OVERLAP)
            
            for i, chunk in enumerate(chunks):
                new_chunks.append(chunk)
                new_metadata.append({
                    "file": str(file_path),
                    "chunk_id": i,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Ошибка при обработке {file_path}: {e}")

    # Генерация эмбеддингов
    if new_chunks:
        embeddings = generate_embeddings(new_chunks, model)
        update_index(embeddings, index, new_metadata)
        faiss.write_index(index, str(INDEX_PATH))
        
        # Сохранение чанков
        with open(CHUNKS_PATH, "a", encoding="utf-8") as f:
            for chunk in new_chunks:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

        logger.info(f"Добавлено чанков: {len(new_chunks)}")
    else:
        logger.warning("Нет чанков для добавления.")

    # Обновление метки времени
    save_last_update(now)

    end_time = time.time()
    logger.info(
        f"Обновление завершено. Время: {end_time - start_time:.2f} сек. "
        f"Размер индекса: {index.ntotal}"
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}")
