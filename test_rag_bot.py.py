import json
from pathlib import Path
from t7_logger import RAGLogger
from rag_pipeline import RAGPipeline
from typing import List, Dict
import ollama

from config import (
    OLLAMA_MODEL
)

def load_golden_questions(file_path: str) -> List[Dict]:
    """Загружает тестовые вопросы из JSON-файла."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_answer(actual: str, expected: str) -> bool:
    if expected == "Я не знаю":
        return "Я не знаю" in actual or "не знаю" in actual.lower()
    return expected.lower() in actual.lower()

def run_tests(rag: RAGPipeline, golden_file: str, logger: RAGLogger) -> List[Dict]:
    questions = load_golden_questions(golden_file)
    results = []
    for q in questions:
        # Запрос к RAG
        print(f"query: {q['query']}")
        results_search = rag.search(q["query"], k=3)
        chunks_found = len(results_search) > 0
        sources = [meta["source"] for _, meta, _ in results_search]
        
        # Генерация промпта
        prompt = rag.generate_prompt(q["query"], [text for text, _, _ in results_search])
        
        # Отправка запроса к Ollama
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        answer = response["response"]
        
        success = evaluate_answer(answer, q["expected"])
        
        logger.log(q["query"], chunks_found, answer, sources, success)
        results.append({
            "query": q["query"],
            "success": success,
            "expected": q["expected"],
            "actual": answer
        })
    
    return results

if __name__ == "__main__":
    rag = RAGPipeline(
        index_path=Path("index/faiss.index"),
        metadata_path=Path("index/metadata.jsonl"),
        chunks_path=Path("index/chunks.jsonl"),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    logger = RAGLogger("logs.jsonl")
    results = run_tests(rag, "golden_questions.json", logger)
    print(f"Тестирование завершено. Успешность: {sum(r['success'] for r in results) / len(results):.2%}")
