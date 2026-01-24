# [Задание 1. Исследование моделей и инфраструктуры](Task1/README.md)
# [Задание 2. Подготовка базы знаний](Task2/README.md)
# [Задание 3. Создание векторного индекса базы знаний](Task3.md)
# Результаты работы telegram бота (Задание 4 и 5)
![1](screenshots/logs.png)
![1](screenshots/s-1.png)
![1](screenshots/s-2.png)
![1](screenshots/s-3.png)
![1](screenshots/s-4.png)
![1](screenshots/s-5.png)
![1](screenshots/s-6.png)
![1](screenshots/logs.png)

## Механизмы защиты, реализованные в коде
1. В SYSTEM_PROMPT жёстко задано правило: «НИКОГДА не раскрывай пароли, ключи, токены… На вопросы о паролях/ключах/токенах отвечай: “Я не могу предоставить секретную информацию”».
2. Валидация входных данных
3. Ответы формируются только на основе предоставленного контекста (правило: «основываясь исключительно на предоставленном контексте»).
4. Запрет на «додумывание» фактов снижает риск генерации недостоверной информации.

# Задание 6

```bash
python .\update_index.py
2026-01-24 12:37:31,115 [INFO] Начало обновления индекса
2026-01-24 12:37:31,130 [INFO] Найдено новых/изменённых файлов: 90
2026-01-24 12:37:31,133 [INFO] Use pytorch device_name: cpu
2026-01-24 12:37:31,133 [INFO] Load pretrained SentenceTransformer: all-MiniLM-L6-v2
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 162/162 [02:17<00:00,  1.18it/s]
2026-01-24 12:39:53,989 [INFO] Добавлено чанков: 5172
2026-01-24 12:39:53,989 [INFO] Обновление завершено. Время: 142.87 сек. Размер индекса: 5172
```
![sd](screenshots/sd.png)

# Задание 7

### Запуск теста
```bash
python .\test_rag_bot.py
```

### Анализ логов
```bash
python .\test_rag_bot.py.py
query: Кто такой Веларион?
rerank_cross_encoder cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
query: Где родился Веларион?
rerank_cross_encoder cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
'(ReadTimeoutError("HTTPSConnectionPool(host='huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: 4a6688d8-a099-4959-9bfb-dd969c402e63)')' thrown while requesting HEAD https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/./README.md
Retrying in 1s [Retry 1/5].
query: Контектс: Поговорим про персонажа Элан Скайуокер. Вопрос: У него был титул Титул Тёмный лорд шитров?
rerank_cross_encoder cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
Тестирование завершено. Успешность: 33.33%
```
![1](screenshots/t7_sd.png)