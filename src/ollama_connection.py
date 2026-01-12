import os

from llama_index.llms.ollama import Ollama
from openai import AsyncOpenAI
from ragas.llms import llm_factory

from initialization import credential_init

def llama_index_ollama(model: str, temperature: float):

    """
    llm = llama_index_ollama(model='gpt-oss:120b-cloud') 
    """
    credential_init()
    
    base_url = "https://ollama.com"

    llm = Ollama(model=model, request_timeout=60.0, 
                 base_url=base_url, temperature=temperature)

    return llm


def ragas_ollama(model: str):

    credential_init()

    base_url="https://ollama.com/v1"
    
    client = AsyncOpenAI(
        api_key=os.environ["OLLAMA_API_KEY"],
        base_url=base_url
    )
    llm = llm_factory(model, provider="openai", client=client, max_tokens=4096)

    return llm