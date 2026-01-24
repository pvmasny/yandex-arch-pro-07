import os
import re
from bs4 import BeautifulSoup

def clean_html_to_text(html_content):
    """
    Преобразует HTML в чистый текст.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Удаляем скрипты, стили и прочие нетекстовые элементы
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
        element.decompose()

    # Получаем текст
    text = soup.get_text(separator='\n', strip=True)

    # Очищаем от лишних пробелов и переносов
    text = re.sub(r'\n+', '\n', text)      # несколько \n → один
    text = re.sub(r' +', ' ', text)       # несколько пробелов → один
    text = re.sub(r'\n\s*\n', '\n\n', text)  # пустые строки → одна пустая строка
    return text.strip()

def process_html_files(input_root, output_root):
    """
    Обходит все HTML‑файлы в input_root и сохраняет чистый текст в output_root.
    Сохраняет структуру подкаталогов.
    """
    # Создаём корневую выходную папку
    os.makedirs(output_root, exist_ok=True)

    # Обходим все файлы и папки рекурсивно
    for root, dirs, files in os.walk(input_root):
        for filename in files:
            if filename.lower().endswith(('.html', '.htm')):
                # Полный путь к входному файлу
                input_path = os.path.join(root, filename)

                # Формируем относительный путь (без input_root)
                rel_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, rel_path)

                # Создаём выходную папку, если её нет
                os.makedirs(output_dir, exist_ok=True)


                # Имя выходного файла (.txt вместо .html/.htm)
                output_filename = os.path.splitext(filename)[0] + '.txt'
                output_path = os.path.join(output_dir, output_filename)

                try:
                    # Читаем HTML
                    with open(input_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    # Очищаем
                    text_content = clean_html_to_text(html_content)

                    # Сохраняем текст
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)

                    print(f"✅ {input_path} → {output_path}")

                except Exception as e:
                    print(f"❌ Ошибка обработки {input_path}: {e}")

def main():
    INPUT_DIR = "Task2/knowledge_base/original"
    OUTPUT_DIR = "Task2/knowledge_base/original_md"

    print("Начинаем обработку HTML‑файлов...")
    process_html_files(INPUT_DIR, OUTPUT_DIR)
    print("\nГотово! Чистые тексты сохранены в:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
