import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class RAGLogger:
    def __init__(self, log_file: str = "logs.jsonl"):
        self.log_file = Path(log_file)

    def log(
        self,
        query: str,
        chunks_found: bool,
        answer: str,
        sources: List[str],
        success: bool
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "chunks_found": chunks_found,
            "answer_length": len(answer),
            "success": success,
            "sources": sources,
            "answer": answer
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def to_csv(self, csv_file: str = "logs.csv"):
        with open(self.log_file, "r", encoding="utf-8") as f_jsonl, \
             open(csv_file, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=[
                "timestamp", "query", "chunks_found", "answer_length",
                "success", "sources", "answer"
            ])
            writer.writeheader()
            for line in f_jsonl:
                writer.writerow(json.loads(line.strip()))
