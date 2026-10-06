import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# Настройки путей через абсолютное позиционирование в Linux
current_dir = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(current_dir, "folder_vector_db")
FAISS_FILE = os.path.join(DB_DIR, "index.faiss")
METADATA_FILE = os.path.join(DB_DIR, "metadata.json")
INPUT_JSON = os.path.join(current_dir, "folder_parsed_data.json")

device_str = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка моделей
print("Загрузка эмбеддеро-модели...")
emb_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
emb_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device_str)

print("Загрузка Qwen для расширения запросов...")
llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map=device_str)
llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def get_embeddings(texts):
    prefixed_texts = [f"passage: {text}" for text in texts]
    inputs = emb_tokenizer(prefixed_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device_str) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = emb_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def build_vector_store():
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
        
    print(f"Кодируем {len(knowledge_base)} крупных чанков...")
    all_texts = [chunk["text"] for chunk in knowledge_base]
    embeddings = get_embeddings(all_texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    faiss.write_index(index, FAISS_FILE)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)
    print("Векторная база для папки книг успешно сохранена!")
    return index, knowledge_base

# =====================================================================
# ПЕРВАЯ НЕЙРОСЕТЬ: ОБШИРНОЕ РАСШИРЕНИЕ ЗАПРОСА (Генерация HyDE контекста)
# =====================================================================
def expand_query_extensive(user_query: str) -> str:
    # Промпт изменен: теперь мы просим выдать развернутый абзац с терминологией, чтобы зацепить максимум векторов
    system_prompt = """Ты — ИИ-генератор поискового контекста. На основе короткого вопроса пользователя напиши подробное описание технической проблемы, добавь связанные понятия, алгоритмы, структуры данных и ключевые слова на русском и английском языках, которые могут встречаться на страницах профильной литературы. Выводи только этот поисковый текст."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос пользователя: {user_query}"}
    ]
    text_prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = llm_tokenizer([text_prompt], return_tensors="pt").to(device_str)
    
    with torch.no_grad():
        # Увеличиваем число токенов генерации, чтобы запрос не обрезался
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=200, temperature=0.4)
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def search_faiss(index, knowledge_base, query, top_k=4): # Вытаскиваем сразу ТОП-4 крупных куска
    prefixed_query = f"query: {query}"
    inputs = emb_tokenizer(prefixed_query, return_tensors="pt").to(device_str)
    with torch.no_grad():
        outputs = emb_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1).cpu().numpy()
        
    distances, indices = index.search(query_embedding, top_k)
    return [knowledge_base[idx] for idx in indices[0] if idx != -1]

if __name__ == "__main__":
    if not os.path.exists(FAISS_FILE):
        index, knowledge_base = build_vector_store()
    else:
        index = faiss.read_index(FAISS_FILE)
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)

    # Итоговый тест первой ступени
    question = "платформы для участия в онлайн соревнованиях"
    expanded = expand_query_extensive(question)
    print(f"\n[Расширенный поисковый образ]:\n{expanded}\n")
    
    hits = search_faiss(index, knowledge_base, expanded, top_k=4)
    print(f"Найдено релевантных крупных документов: {len(hits)}")
