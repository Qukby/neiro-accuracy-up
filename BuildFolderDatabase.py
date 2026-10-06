import os
import re
import json
import torch
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# НАСТРОЙКА УВЕЛИЧЕННОГО СПЛИТТЕРА (Размер чанка: 1200 символов)
# =====================================================================
class SmartTextSplitter:
    def __init__(self, chunk_size_chars: int = 1200, overlap_sentences: int = 3):
        self.chunk_size_chars = chunk_size_chars
        self.overlap_sentences = overlap_sentences

    def _split_into_sentences(self, text: str) -> list:
        # Паттерн корректно обрабатывает русскую и английскую раскладки
        sentences = re.split(r'(?<=[.!?])\s+(?=[А-ЯA-Z])', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def split_page(self, page_text: str, page_number: int, source_name: str, book_title: str) -> list:
        paragraphs = [p.strip() for p in page_text.split('\n') if p.strip()]
        chunks = []
        current_chunk_text = ""
        current_sentences = []
        
        for para in paragraphs:
            para_sentences = self._split_into_sentences(para)
            for sentence in para_sentences:
                if len(current_chunk_text) + len(sentence) < self.chunk_size_chars:
                    current_sentences.append(sentence)
                    current_chunk_text = " ".join(current_sentences)
                else:
                    if current_sentences:
                        chunks.append({
                            "book": book_title,
                            "source": source_name,
                            "page": page_number,
                            "text": current_chunk_text
                        })
                    overlap_pool = current_sentences[-self.overlap_sentences:] if len(current_sentences) >= self.overlap_sentences else current_sentences
                    current_sentences = list(overlap_pool) + [sentence]
                    current_chunk_text = " ".join(current_sentences)
        
        if current_sentences:
            chunks.append({
                "book": book_title,
                "source": source_name,
                "page": page_number,
                "text": current_chunk_text
            })
        return chunks

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ И СБОРКА КОНВЕЙЕРА ПАРСИНГА ПАПКИ
# =====================================================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Загрузка модели {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device_str)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def extract_topic_with_llm(page_text: str, page_num: int, book_title: str) -> str:
    system_prompt = """Ты — ИИ-ассистент, размечающий учебники. Определи ОДНУ главную техническую тему или заголовок подраздела на этой странице.
Выводи строго только само название темы. Без пояснений, без кавычек и без точек."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Книга: {book_title}\nСтраница: {page_num}\n\nТекст:\n{page_text[:1500]}"}
    ]
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(device_str)

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=40, temperature=0.1)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def parse_all_pdfs_in_folder(folder_path: str):
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_folder_path = os.path.join(current_script_dir, folder_path)
    
    if not os.path.exists(absolute_folder_path):
        os.makedirs(absolute_folder_path)
        print(f"Папка '{folder_path}' не существовала. Создали её рядом со скриптом. Положите туда PDF файлы.")
        return []

    pdf_files = [f for f in os.listdir(absolute_folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"В папочке '{folder_path}' нет PDF файлов для обработки.")
        return []

    splitter = SmartTextSplitter(chunk_size_chars=1200, overlap_sentences=3)
    global_knowledge_base = []

    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(absolute_folder_path, pdf_file)
        print(f"\n Начинаем обработку книги: {pdf_file}")
        reader = PdfReader(full_pdf_path)
        
        # Парсим страницы выборочно или целиком. Для теста ограничим 5-10 страницами, чтобы сберечь время
        # Если нужно парсить книгу целиком — замените range(17, 25) на range(len(reader.pages))
        for page_idx in range(17, min(25, len(reader.pages))):
            real_page_num = page_idx + 1
            raw_text = reader.pages[page_idx].extract_text()
            
            if not raw_text.strip():
                continue
                
            print(f"[{pdf_file} | Стр {real_page_num}] Анализ темы...")
            detected_topic = extract_topic_with_llm(raw_text, real_page_num, pdf_file)
            print(f"==> Тема: {detected_topic}")
            
            page_chunks = splitter.split_page(raw_text, real_page_num, detected_topic, pdf_file)
            global_knowledge_base.extend(page_chunks)

    return global_knowledge_base

if __name__ == "__main__":
    FOLDER_NAME = "books_dir"
    output_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_parsed_data.json")
    
    chunks = parse_all_pdfs_in_folder(FOLDER_NAME)
    
    if chunks:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)
        print(f"\n Успешно! Спарсено {len(chunks)} крупных чанков. Данные в: {output_json_path}")
