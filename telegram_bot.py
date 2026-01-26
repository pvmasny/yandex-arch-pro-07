import logging
from typing import List
import os
from pathlib import Path
import ollama

from telegram import Update, ForceReply
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


import httpx
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Импорты из локальных модулей (как в FastAPI)
from config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    BM25_WEIGHT,
    TELEGRAM_TOKEN
)
from rag_pipeline import RAGPipeline

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация RAG-пайплайна (как в FastAPI)
try:
    rag = RAGPipeline(
        index_path=Path(FAISS_INDEX_PATH),
        metadata_path=Path(METADATA_PATH),
        chunks_path=Path(CHUNKS_PATH),
        embedding_model=EMBEDDING_MODEL,
        bm25_weight=BM25_WEIGHT
    )
    logger.info("RAG-пайплайн загружен успешно.")
except Exception as e:
    logger.error(f"Ошибка при загрузке RAG-пайплайна: {e}")
    raise

# Токен бота (установите в окружении или прямо здесь)
TOKEN = os.getenv("TELEGRAM_TOKEN", "8541175039:AAHolCXkEu6ouHPFM1TmN3-tCdFQHNlH85E")


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_markdown_v2(
        fr"Привет, {user.mention_markdown_v2()}\! Я RAG‑бот для поиска ответов в базе знаний\.",
        reply_markup=ForceReply(selective=True),
    )

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text
    logger.info(f"Получен запрос от Telegram: '{query}'")


    try:
        # Поиск релевантных чанков (аналогично FastAPI /query)
        results = rag.search(
            query=query,
            k=3,  # по умолчанию 3 чанка
            use_hybrid=False,
            rerank_cross_encoder="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )

        if not results:
            await update.message.reply_text("Я не знаю")
            return

        # Извлекаем тексты и метаданные
        contexts = [text for text, _, _ in results]
        meta_list = [meta for _, meta, _ in results]

        print(f"contexts", contexts)
        print(f"meta_list", meta_list)

        # Формируем промпт для LLM
        prompt = rag.generate_prompt(query, contexts)

        # Отправляем запрос к Ollama
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.3}
        )
        answer = response["response"]

        # Формируем ответ с источниками
        reply = f"{answer}\n\n**Источники:**\n"
        for i, meta in enumerate(meta_list, 1):
            reply += f"{i}. {meta.get('source', 'Неизвестно')}\n"

        await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        await update.message.reply_text("Произошла ошибка. Попробуйте ещё раз.")

def main() -> None:
    """Запуск Telegram-бота."""
    application = Application.builder().token(TOKEN).build()

    # Обработчики команд и сообщений
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


    # Запуск бота
    logger.info("Запуск Telegram-бота...")
    application.run_polling()

if __name__ == "__main__":
    main()
