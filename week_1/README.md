# LlamaIndex 實作：向量檢索（Vector-based RAG）快速上手 (Week 1)

本專案為 RAG（Retrieval-Augmented Generation）教學系列的 Week 1 筆記，示範如何使用 LlamaIndex 架構來建立向量檢索流程，內容涵蓋從 Embeddings、Node 切分、Vector Store、Index 建置到檢索與評估的最小可行範例（MWE）。

---

## 📅 專案資訊

- **初版日期**：2026.01.04
- **最近更新**：2026.01.20
- **筆記版本**：v1.1.0
- **Notebook**：`notebook.ipynb`（Week 1）
---

## 🚀 核心重點

1. 建立一個可運作的 RAG Pipeline 必須處理三件事：
   - 將文字轉為向量（Embeddings）
   - 將文件切分為可檢索的最小單位（Nodes）
   - 使用 Vector Store + Index 進行語意檢索（Retriever / Query Engine）

2. Node Parser 與 Text Splitter 的選擇直接影響檢索品質：
   - Chunk 太大會讓 embedding 混淆主題
   - Chunk 太小會導致語意破碎
   - 選擇要根據資料類型（Markdown、HTML、程式碼、JSON、長文）與應用場景調整

3. 向量資料庫（Vector Store）與 metadata 支援是實務差異點：
   - FAISS：高效、適合本地，但 metadata filter 功能有限
   - Qdrant：支援 metadata 過濾、適合生產與細粒度檢索

---

## 🧩 快速上手（Quick Start）

示範流程大致為：

1. 連接 Ollama（LLM-as-a-Judge / 生成器）
2. 載入 Embedding model（例如 BAAI/bge-small-en-v1.5）
3. 選擇並執行 Node Parser（SentenceSplitter、MarkdownNodeParser、HTMLNodeParser 等）
4. 建立 Vector Store（FAISS / Qdrant）
5. 建立 VectorStoreIndex 並取得 Retriever / Query Engine
6. 執行檢索並把 source nodes 提供給 LLM 生成最終回答

範例程式（省略細節）：

```python
# Ollama LLM（示意）
from llama_index.llms.ollama import Ollama
ollama_llm = Ollama(model="gpt-oss:120b-cloud", request_timeout=60.0)

# Embedding model（示範：BAAI/bge-small-en-v1.5，dim=384）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# FAISS VectorStore（確保 d 與 embedding 維度一致）
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
d = 384
faiss_index = faiss.IndexFlatL2(d)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# 建立 Index（示意）
from llama_index.core import StorageContext, VectorStoreIndex
index = VectorStoreIndex(nodes, embed_model=embed_model, llm=ollama_llm)
retriever = index.as_retriever(similarity_top_k=3, embed_model=embed_model)
```

---

## 🔧 技術棧（Tech Stack）

- Framework：LlamaIndex  
- LLM：Ollama（gpt-oss 系列 for cloud）  
- Embeddings：HuggingFace（範例：BAAI/bge-small-en-v1.5；另有 BAAI/bge-m3 等更高維度模型）  
- Vector Stores：FAISS、Qdrant  
- 檔案處理：FlatReader / PyMuPDFReader / HTMLReader  
- 語言：Python

---

## Node Parser 與 Text Splitter（重點說明）

- Node = 可檢索的最小單位（比 Document 更細）
- 常見 Node Parsers：
  - SentenceSplitter：以句子切分（常用）
  - SentenceWindowNodeParser：每個 node 為單句，但 metadata 保留前後句窗（利於精準 embedding）
  - MarkdownNodeParser / HTMLNodeParser / JSONNodeParser / SimpleFileNodeParser：依檔案類型解析
  - SemanticSplitterNodeParser：在已斷句的基礎上，依語意跳躍（semantic breakpoints）進行切分
  - TokenTextSplitter：依 token 數切分（精細控制）
  - CodeSplitter：針對程式碼檔做語言-aware 的切分

建議：
- 對於長篇說明文，使用 SentenceSplitter 或 SemanticSplitter
- 對於程式碼檔使用 CodeSplitter
- 需要上下文但又要細粒度時，使用 SentenceWindowNodeParser

---

## Vector Store：FAISS 與 Qdrant 比較

- FAISS
  - 優點：本地效能高、輕量
  - 限制：不支援 metadata-level filtering（或較不方便）
  - 注意：建立 Index 時需供入與 embedding 維度一致的 d（例：384 / 1024）

- Qdrant
  - 優點：支援 metadata filter、向量集合管理、可用於遠端/容器化部署
  - 使用時需注意同步 aclient（async）或 client（sync）的差異

---

## Index 建置與 StorageContext（概念補充）

- StorageContext 管理 Vector Store、Docstore、Index Store 與可選的 Property Graph Store（若要做知識圖）
- VectorStoreIndex 提供 as_retriever / as_query_engine 的便利介面
- 建議在建置時同時指定 embed_model 與 llm（可在 as_query_engine 時覆寫）

---

## 檢索（Retrieval）與 Query Engine

- Retriever 的職責：
  - 把 query 轉為 embedding
  - 與 vector store 比對並回傳最相近的 nodes（NodeWithScore）
- Query Engine 則負責更上層的處理：拼接 context、呼叫 LLM 生成最終回答
- 注意：retriever 回傳的是 Node 清單（不是最終文字答案），需由 LLM 進行融合與生成

---

## 問答集生成（Dataset Generation）與參數說明

使用 LlamaIndex 的 DatasetGenerator / QueryResponseDataset 可自動從 nodes 生成問答集。兩個常見參數影響產出量：

- num_questions_per_chunk（深度）  
  - 每個 node 由 LLM 生成的問題數（控制單一 node 的挖掘深度）

- num（廣度）  
  - 從所選 nodes 中總共要處理多少個 node（控制覆蓋範圍）

公式：Total Questions = num * num_questions_per_chunk

實務建議：分批（batch）呼叫 LLM 以降低 API 失敗率並控制成本（例如每批 5 個 node，之後休息 1–2 秒）。

---

## 評估（Evaluation）— LlamaIndex 內建評估器

常見 Evaluator：

- CorrectnessEvaluator（正確性，需要 reference）
- SemanticSimilarityEvaluator（語意相似度，使用 embed model）
- RelevancyEvaluator（回答是否針對問題）
- FaithfulnessEvaluator（是否忠實於被檢索的 Context）

BatchEvalRunner 可並行執行多個 evaluator，並輸出結果表格（get_results_df）方便比較不同 Vector Store / Index 設定的效能。

範例流程（簡化）：

```python
from llama_index.core.evaluation import CorrectnessEvaluator, FaithfulnessEvaluator, RelevancyEvaluator, SemanticSimilarityEvaluator
evaluator_c = CorrectnessEvaluator(llm=ollama_llm)
evaluator_f = FaithfulnessEvaluator(llm=ollama_llm)
evaluator_r = RelevancyEvaluator(llm=ollama_llm)
evaluator_s = SemanticSimilarityEvaluator(embed_model=embed_model)

# Batch run with BatchEvalRunner...
```

---

## 注意事項與實務提示

- Embedding 維度必須一致：向量資料庫的 d 要與 embedding model 輸出維度相符（例：BAAI/bge-small-en-v1.5 → d=384；bge-m3 → d=1024）。
- 小心 token / chunk 大小：避免一次送入過長文本超出 LLM 上下文限制。
- metadata-filter：若需要以 metadata 做精準過濾，建議使用支援 metadata 的 vector store（如 Qdrant）。
- 並發控制：對 cloud LLM 做大量請求時，需設定合適的 batch 與 rate limit（可用 run_config / asyncio 分批）。
- 評估成本：使用 LLM-based evaluator（LLM-as-a-Judge）會增加成本，請在樣本量上做好取捨。

---

## 參考資料與延伸閱讀

- LlamaIndex 官方文件（Node Parsers / Vector Stores / Evaluation）  
- Ollama Cloud（模型連線與使用）  
- FAISS / Qdrant 官方說明

---

如果你想要範例程式或 step-by-step 的執行筆記（含完整 code cell），請參考 notebook.ipynb。需要我幫你把 notebook 中的某段流程拆成可直接複製執行的 script 嗎？