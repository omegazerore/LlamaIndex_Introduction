from src.ollama_connection import llama_index_ollama

model = "gpt-oss:120b-cloud"

ollama_llm = llama_index_ollama(model=model, temperature=0)