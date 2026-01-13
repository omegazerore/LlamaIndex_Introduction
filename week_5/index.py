import faiss
import mlflow
from llama_index.core import StorageContext, Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.ollama_connection import llama_index_ollama


model = "gpt-oss:120b-cloud"

ollama_llm = llama_index_ollama(model=model, temperature=0)

loader = PyMuPDFReader()

docs = loader.load("week_1/08物理.pdf")

doc_text = "\n\n".join([d.get_content() for d in docs])
docs = [Document(text=doc_text)]

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
d = 1024 # 必須與 embedding model 的輸出維度一致

faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    [],
    storage_context=storage_context,
    transformations=[SimpleFileNodeParser()],
    embed_model=embed_model
)

# add documents to index
for doc in docs:
    index.insert(doc)

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("LlamaIndex")

mlflow.models.set_model(index)