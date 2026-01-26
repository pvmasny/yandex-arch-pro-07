#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
import json
import logging
import gzip
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LC = True
except ImportError:
    _HAS_LC = False



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def read_text_file(path: Path) -> str:
    """Читает текстовый файл с кодировкой UTF‑8."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка чтения файла {path}: {e}")
        return ""



def collect_documents(data_dir: Path, exts: Tuple[str, ...] = (".txt", ".md")) -> List[Path]:
    """Собирает все файлы с указанными расширениями в директории (рекурсивно)."""
    files: List[Path] = []
    for root, _, filenames in os.walk(data_dir):
        for name in filenames:
            if name.lower().endswith(exts):
                files.append(Path(root) / name)
    return sorted(files)



def word_chunk(text: str, chunk_size_words: int = 256, chunk_overlap_words: int = 40) -> List[Tuple[str, int]]:
    """
    Разбивает текст на чанки по количеству слов.
    """
    words = text.split()
    chunks: List[Tuple[str, int]] = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size_words, len(words))
        piece_words = words[i:j]
        if not piece_words:
            break
        chunks.append((" ".join(piece_words), i))
        step = chunk_size_words - chunk_overlap_words if chunk_size_words > chunk_overlap_words else chunk_size_words
        i += max(step, 1)
    return chunks



def lc_chunk(text: str, chunk_size_chars: int = 1200, chunk_overlap_chars: int = 200) -> List[Tuple[str, int]]:
    """
    Разбивает текст на чанки с помощью RecursiveCharacterTextSplitter (по символам).
    """
    if not _HAS_LC:
        raise RuntimeError(
            "RecursiveCharacterTextSplitter недоступен. Установите langchain-text-splitters "
            "или не используйте --use_lc_splitter."
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parts: List[str] = splitter.split_text(text)

    chunks: List[Tuple[str, int]] = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor  # fallback
        chunks.append((part, idx))
        cursor = idx + len(part)
    return chunks




def main():
    parser = argparse.ArgumentParser(description="Индексация БЗ в FAISS с эмбеддингами")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Task2/knowledge_base/modified_md",
        help="Каталог с .txt/.md файлами (по умолчанию: Task2/knowledge_base/modified_md)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="index",
        help="Куда сохранять индекс и метаданные (по умолчанию: Task2/index)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Имя модели эмбеддингов (по умолчанию: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--use_lc_splitter",
        action="store_true",
        help="Использовать RecursiveCharacterTextSplitter (по символам)"
    )
    parser.add_argument(
        "--chunk_size_words",
        type=int,
        default=200,
        help="Размер чанка по словам (если без LC)"
    )
    parser.add_argument(
        "--chunk_overlap_words",
        type=int,
        default=40,
        help="Перекрытие чанков по словам"
    )
    parser.add_argument(
        "--lc_chunk_chars",
        type=int,
        default=1200,
        help="Размер чанка по символам (LC)"
    )
    parser.add_argument(
        "--lc_chunk_overlap_chars",
        type=int,
        default=200,
        help="Перекрытие чанков по символам (LC)"
    )
    parser.add_argument(
        "--store_chunks",
        default=True,
        action="store_true",
        help="Сохранять тексты чанков в index/chunks.jsonl.gz"
    )

    args = parser.parse_args()

    # Валидация аргументов
    assert args.chunk_size_words > 0, "--chunk_size_words должен быть > 0"
    assert args.lc_chunk_chars > 0, "--lc_chunk_chars должен быть > 0"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_index_path = out_dir / "faiss.index"
    meta_path = out_dir / "metadata.jsonl"
    chunks_path = out_dir / "chunks.jsonl.gz"  # Сжатый формат

    # Кешинг: проверяем, существует ли индекс
    if faiss_index_path.exists() and meta_path.exists():
        logger.info(f"Индекс и метаданные уже существуют в {out_dir}. Пропуск генерации.")
        return

    t_total = time.perf_counter()

    files = collect_documents(data_dir)
    if not files:
        raise SystemExit(f"Нет документов в {data_dir.resolve()}")

    logger.info(f"Найдено {len(files)} файлов для обработки.")

    t_chunk = time.perf_counter()
    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()  # Для дедупликации

    for fp in files:
        text = read_text_file(fp)
        if not text or not text.strip():
            logger.warning(f"Файл {fp} пуст или не прочитан. Пропуск.")
            continue

        if args.use_lc_splitter:
            try:
                parts = lc_chunk(text, args.lc_chunk_chars, args.lc_chunk_overlap_chars)
            except Exception as e:
                logger.error(f"Ошибка чанкинга для {fp}: {e}")
                continue
        else:
                        parts = word_chunk(text, args.chunk_size_words, args.chunk_overlap_words)


        for i, (part, start_pos) in enumerate(parts):
            # Дедупликация: пропускаем идентичные чанки
            if part in seen_texts:
                logger.debug(f"Дубликат чанка пропущен для {fp}, чанк {i}")
                continue
            seen_texts.add(part)

            chunks.append(part)
            metadatas.append({
                "id": f"{fp.as_posix()}::chunk_{i}",
                "path": fp.as_posix(),
                "title": fp.name,
                "chunk_index": i,
                "chunk_len_words": len(part.split()),
                "position": {
                    "type": "char" if args.use_lc_splitter else "word",
                    "start": int(start_pos)
                },
                "source": fp.as_posix(),
                "file_mtime": fp.stat().st_mtime  # Время последнего изменения файла
            })

    chunk_time = time.perf_counter() - t_chunk


    if not chunks:
        raise SystemExit("После чанкинга не осталось текста. Проверьте входные файлы и параметры.")


    logger.info(f"Сформировано {len(chunks)} уникальных чанков.")

    t_model = time.perf_counter()
    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {args.model}: {e}")
        raise
    model_time = time.perf_counter() - t_model


    t_emb = time.perf_counter()
    try:
        emb = model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,  # Для cosine similarity через IP
        )
    except Exception as e:
        logger.error(f"Ошибка при кодировании текстов: {e}")
        raise
    emb_time = time.perf_counter() - t_emb

    dim = int(emb.shape[1])

    t_faiss = time.perf_counter()
    index = faiss.IndexFlatIP(dim)  # Inner Product для cosine
    index.add(emb)
    faiss_write_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_write_path))
    faiss_time = time.perf_counter() - t_faiss


    t_write = time.perf_counter()

    # Сохраняем метаданные
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadatas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Сохраняем чанки (сжато)
    if args.store_chunks:
        with gzip.open(chunks_path, "wt", encoding="utf-8") as f:
            for i, text in enumerate(chunks):
                f.write(json.dumps({"i": i, "text": text}, ensure_ascii=False) + "\n")
        logger.info(f"Чанки сохранены в {chunks_path}")


    write_time = time.perf_counter() - t_write


    total_time = time.perf_counter() - t_total

    stats = {
        "model": args.model,
        "embedding_dim": dim,
        "num_files": len(files),
        "num_chunks": len(chunks),
        "num_unique_chunks": len(seen_texts),

        # Тайминги
        "elapsed_chunking_s": round(chunk_time, 3),
        "elapsed_model_load_s": round(model_time, 3),
        "elapsed_embedding_s": round(emb_time, 3),
        "elapsed_faiss_io_s": round(faiss_time, 3),
        "elapsed_write_meta_s": round(write_time, 3),
        "elapsed_total_s": round(total_time, 3),

        # Пути
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "faiss_index": str(faiss_write_path),
        "metadata": str(meta_path),
        "chunks_gz": str(chunks_path) if args.store_chunks else None
    }

    # Сохраняем статистику
    with open(out_dir / "build_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


    logger.info("Индексация завершена.")
    print(json.dumps(stats, ensure_ascii=False))



if __name__ == "__main__":
    main()
