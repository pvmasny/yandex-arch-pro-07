import json
from collections import Counter
from pathlib import Path

def analyze_logs(log_file: str):
    # Проверка существования файла
    file_path = Path(log_file)
    if not file_path.exists():
        print(f"Ошибка: файл '{log_file}' не найден.")
        return

    logs = []
    error_count = 0

    # Построчное чтение и разбор JSONL
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Пропускаем пустые строки
                continue
            try:
                log_entry = json.loads(line)
                logs.append(log_entry)
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Ошибка JSON в строке {line_num}: {e}")

    # Сообщение о найденных ошибках
    if error_count > 0:
        print(f"\nПредупреждение: обнаружено {error_count} ошибок в формате JSON. Эти строки пропущены.\n")

    # Защита от пустого лога
    if not logs:
        print("Лог пуст или не содержит валидных записей.")
        return

    # 1. Общая успешность
    successful = [log for log in logs if log.get("success")]
    success_rate = len(successful) / len(logs)
    print(f"Общая успешность: {success_rate:.2%}")

    # 2. Частые неудачные запросы
    failed_queries = [log["query"] for log in logs if not log.get("success")]
    if failed_queries:
        print("\nЧастые неудачные запросы:")
        for q, count in Counter(failed_queries).most_common(5):
            print(f"  {q} ({count} раз)")
    else:
        print("\nНеудачных запросов не обнаружено.")

    # 3. Нерелевантные источники (найденные чанки, но неверный ответ)
    suspicious = [
        log for log in logs
        if log.get("chunks_found") and not log.get("success") and log.get("sources")
    ]
    print(f"\nКоличество случаев с найденными источниками, но неверным ответом: {len(suspicious)}")

    # 4. Статистика по источникам (опционально)
    all_sources = [src for log in logs for src in log.get("sources", [])]
    if all_sources:
        print(f"\nВсего упомянуто источников: {len(all_sources)}")
        print("Топ-5 самых частых источников:")
        for src, count in Counter(all_sources).most_common(5):
            print(f"  {src} ({count} раз)")

if __name__ == "__main__":
    analyze_logs("logs.jsonl")
