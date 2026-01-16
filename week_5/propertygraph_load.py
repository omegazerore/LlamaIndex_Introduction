import os

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.io.path_definition import get_project_dir

# 3. 儲存時，如果是在 Windows，建議顯式檢查目錄
persist_path = os.path.join(get_project_dir(), "week_5", "storage_physics")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=persist_path),
    embed_model=embed_model
)