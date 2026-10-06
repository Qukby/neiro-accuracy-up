import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# Абсолютные пути
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(current_dir, "folder_vector_db")
FAISS_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.json")

device_str = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# ЗАГРУЗКА РЕСУРСОВ
# =====================================================================
print("Загрузка векторной базы и моделей...")
index = faiss.read_index(FAISS_FILE)
with open(METADATA_FILE, 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

emb_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
emb_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device_str)

# Для первой и второй нейросети мы используем одну и ту же модель Qwen-7B,
# но переключаем её поведение с помощью разных системных промптов!
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map=device_str)
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# =====================================================================
# КОМПОНЕНТЫ СИСТЕМЫ
# =====================================================================

# 1. Первая нейросеть: Оптимизатор запроса
def expand_query(user_query: str) -> str:
    system_prompt = "Ты — ИИ-генератор поискового контекста. На основе короткого вопроса пользователя напиши подробное описание технической проблемы, добавь связанные понятия и ключевые слова на русском и английском языках для поиска в учебниках. Выводи только этот текст."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос пользователя: {user_query}"}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer([prompt], return_tensors="pt").to(device_str)
    with torch.no_grad():
        # Ссылаемся на правильный объект llm_model
        generated_ids = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.3)
    generated_ids = [oid[len(iids):] for iids, oid in zip(inputs.input_ids, generated_ids)]
    return llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# Поиск в базе
def get_context_from_db(expanded_query: str, top_k=4):
    prefixed_query = f"query: {expanded_query}"
    inputs = emb_tokenizer(prefixed_query, return_tensors="pt").to(device_str)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_base[idx] for idx in indices[0] if idx != -1]

# 2. ВТОРАЯ НЕЙРОСЕТЬ: Финальный Ассистент-Преподаватель
def generate_final_answer(user_question: str, found_chunks: list) -> str:
    # Собираем текстовый блок контекста из найденных чанков
    context_str = ""
    for i, chunk in enumerate(found_chunks, 1):
        context_str += f"\n--- ФРАГМЕНТ ИЗ ИСТОЧНИКА №{i} ---\n"
        context_str += f"Книга: {chunk['book']}, Раздел/Тема: {chunk['source']}, Страница: {chunk['page']}\n"
        context_str += f"Текст материала: {chunk['text']}\n"
        context_str += "-"*40 + "\n"

    system_prompt = """Ты — высококвалифицированный ассистент-преподаватель в университете. Твоя задача — дать максимально развернутый, точный и понятный ответ на вопрос студента, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты методических материалов.
Обязательно используй технические термины и логику из предоставленного текста. Не сокращай ответ, расписывай его подробно.
В самом конце твоего ответа обязательно выведи блок 'Использованные источники:' со списком книг и номерами страниц, которые тебе помогли."""

    user_content = f"""Вот найденные методические материалы:
{context_str}

Вопрос студента: {user_question}
Ответь на вопрос развернуто и укажи источники в конце:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer([prompt], return_tensors="pt").to(device_str)
    
    with torch.no_grad():
        # Даем большой лимит на генерацию (1024 токена), чтобы ответ получился обширным и не обрезался
        generated_ids = llm_model.generate(**inputs, max_new_tokens=1024, temperature=0.5)
        
    generated_ids = [oid[len(iids):] for iids, oid in zip(inputs.input_ids, generated_ids)]
    return llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# =====================================================================
# СКВОЗНОЙ ЗАПУСК СИСТЕМЫ
# =====================================================================
if __name__ == "__main__":
    # Запрос студента
    student_query = "На каких сайтах проходят онлайновые соревнования по программированию?"
    print(f"Пользователь: {student_query}")
    
    # Ступень 1: Оптимизация запроса первой нейросетью
    print("\n[Шаг 1] Перевод и расширение запроса первой нейросетью...")
    expanded_search_query = expand_query(student_query)
    
    # Ступень 2: Извлечение контекста из FAISS
    print("[Шаг 2] Поиск релевантных материалов в векторной базе...")
    chunks_context = get_context_from_db(expanded_search_query, top_k=4)
    
    # Ступень 3: Генерация обширного ответа второй нейросетью
    print("[Шаг 3] Формирование развернутого ответа второй нейросетью...")
    final_answer = generate_final_answer(student_query, chunks_context)
    
    print("\n" + "="*40 + " ОТВЕТ АССИСТЕНТА " + "="*40)
    print(final_answer)
    print("="*98)
