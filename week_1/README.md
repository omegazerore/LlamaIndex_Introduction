# LlamaIndex 實戰：向量檢索與進階切分（Week 1）

本專案紀錄了使用 **LlamaIndex** 構建 **RAG（Retrieval-Augmented Generation，檢索增強生成）** 系統的基礎流程與進階優化技術。  
內容涵蓋從 **環境設定、多樣化向量資料庫整合，到量化評估** 的完整實作。

---

## 📅 更新資訊

- **初版日期**：2026.01.04  
- **最近更新**：2026.01.20  
- **筆記版本**：v1.1.0  

---

## 🚀 核心重點

### 1. 向量檢索最小可行流程（Minimal Working Example）

建立 RAG 系統的基礎架構與核心組件理解：

- **LLM 整合**  
  - 採用 Ollama 雲端服務：`gpt-oss:120b-cloud`

- **Embedding Model**  
  - 本地端：`BAAI/bge-small-en-v1.5`（384 維）  
  - 進階測試：`BAAI/bge-m3`（1024 維）

- **Storage Context**  
  - 統一管理：
    - Vector Store（向量）
    - Docstore（文本）
    - Index Store（結構）

---

### 2. 進階 Node Parsers 與切分技術

針對不同場景，選擇合適的「**最小檢索原子單位**」：

- **基礎切分**
  - `SentenceSplitter`
  - `TokenTextSplitter`

- **檔案感知切分**
  - `SimpleFileNodeParser` 搭配 `FlatReader` 自動識別檔案格式
  - `HTMLNodeParser` 支援指定 HTML 標籤解析

- **程式碼切分**
  - `CodeSplitter`：支援多種程式語言，依邏輯與行數進行切分

- **語意切分**
  - `SemanticSplitterNodeParser`
  - 利用 Embedding 相似度自適應尋找語意斷點，避免斷章取義

---

### 3. 多樣化向量資料庫（Vector Stores）

實作並比較不同特性的向量儲存方案：

- **FAISS**
  - 高效能本地向量檢索
  - 使用 `IndexFlatL2`（歐式距離）
  - 適合快速原型開發

- **Qdrant**
  - 支援非同步操作（`AsyncQdrantClient`）
  - 強大的 **Metadata Filtering**（如 `ExactMatchFilter`）
  - 解決 FAISS 無法在元數據層級進行過濾的限制

---

### 4. 進階檢索策略與管線設計

- **SentenceWindowNodeParser**
  - 精準檢索單一句子
  - 透過 Metadata 替換，提供前後窗口（Window）上下文資訊

- **IngestionPipeline**
  - 將以下流程模組化：
    - Node Parser
    - Embedding
    - 寫入 Vector Store

---

### 5. 量化評估（Evaluation）

利用 **LLM 作為裁判（Judge）**，確保系統整體品質：

- **DatasetGenerator**
  - 自動從現有 Nodes 生成問答資料集
  - 參數控制：
    - `num`：題目廣度
    - `num_questions_per_chunk`：題目深度

- **四大評估指標**

| 指標 | 評估重點 |
| :--- | :--- |
| Correctness | 回答內容是否與標準答案一致 |
| Faithfulness | 回答是否忠實於檢索文本（避免幻覺） |
| Relevancy | 回答是否確實解決使用者問題 |
| Semantic Similarity | 回答與標準答案在向量空間的接近程度 |

---

## 🛠️ 技術棧

- **Framework**：LlamaIndex  
- **Vector DB**：FAISS、Qdrant  
- **Models**：
  - HuggingFace：BGE v1.5 / BGE-M3  
  - Ollama Cloud：GPT-OSS  
- **Tools**：
  - `nest_asyncio`（處理非同步事件循環）
  - `pandas`（評估結果分析與視覺化）
