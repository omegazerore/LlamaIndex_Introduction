import os
os.environ["PYTHONUTF8"] = "1"

from typing import Literal

from llama_index.core import Document
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.openai import OpenAI
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from src.io.path_definition import  get_project_dir
from initialization import credential_init

credential_init()

openai_llm = OpenAI(
    model="gpt-4o-mini",
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

loader = PyMuPDFReader()

docs = loader.load(os.path.join(get_project_dir(), "week_1", "08物理.pdf"))

doc_text = "\n\n".join([d.get_content() for d in docs])
doc_text = doc_text.encode("utf-8", errors="ignore").decode("utf-8")
docs = [Document(text=doc_text)]

PhysicalEntities = Literal["人", "事", "時", "地", "物"]

# 2. 建立驗證規則：定義哪些實體之間可以有哪些關係
PhysicalRelations = Literal["參與", "發生於", "包含", "操作", "規範"]

# 3. 定義驗證架構
# 例如：「人」會參與「事」；「事」會有對應的「物」或「時」
kg_validation_schema = {
    "人": ["參與", "操作"], # 例如：學生 操作 實驗器材
    "事": ["發生於", "包含", "規範"], # 例如：基礎物理一 包含 萬有引力
    "物": ["發生於"] # 例如：示範實驗器材 發生於 物理實驗室
}

# 4. 初始化提取器
kg_extractor = SchemaLLMPathExtractor(
    llm=openai_llm,
    possible_entities=PhysicalEntities,
    possible_relations=PhysicalRelations,
    kg_validation_schema=kg_validation_schema,
    strict=True # 啟動嚴格模式，確保 LLM 僅使用我們定義的標籤
)

index_physics = PropertyGraphIndex.from_documents(
    docs,
    kg_extractors=[kg_extractor],
    embed_model=embed_model
)

index_physics.storage_context.persist(persist_dir=os.path.join(get_project_dir(), "week_5", "storage_physics"))