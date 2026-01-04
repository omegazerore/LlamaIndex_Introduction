# LlamaIndex 實戰：向量檢索與進階切分 (Week 1)

本專案紀錄了使用 LlamaIndex 構建 RAG（檢索增強生成）系統的基礎流程與進階優化技術。內容涵蓋從環境設定、資料處理管線到自動化評估的完整實作。

## 📅 更新資訊
- **初版日期**：2026.01.04
- **筆記版本**：v1.0.0

## 🚀 核心重點

### 1. 向量檢索最小可行流程 (Minimal Working Example)
建立 RAG 系統的基礎架構：
* **Embedding Model**：採用 `BAAI/bge-small-en-v1.5`。
* **Vector Store**：使用 FAISS (Facebook AI Similarity Search) 進行高效能向量檢索。
* **Ingestion Pipeline**：將文件切分、向量化與寫入儲存空間的流程模組化。

### 2. Node Parsers 與文字切分技術
針對不同資料格式與場景的切分策略：
* **基礎切分**：`SentenceSplitter` (保留句子邊界) 與 `TokenTextSplitter`。
* **格式感知**：支援 Markdown、HTML、JSON 及多種程式語言 (CodeSplitter) 的專屬解析器。
* **語意切分**：`SemanticSplitterNodeParser`，依據語意相似度自適應決定切分點。

### 3. 進階檢索策略
* **SentenceWindowNodeParser**：檢索單一句子以確保精確度，並透過 Metadata 替換提供完整的上下文字窗 (Context Window)。
* **HierarchicalNodeParser**：建立父子階層結構，結合 `AutoMergingRetriever` 在子節點被檢索時自動合併為父節點，提供更豐富的脈絡。

### 4. 量化評估 (Evaluation)
利用 LlamaIndex 的評估模組確保系統品質：
* **自動化生成**：使用 `DatasetGenerator` 生成測試問題集。
* **多維度指標**：包含正確性 (Correctness)、忠誠度 (Faithfulness) 與相關性 (Relevancy) 評估。

## 🛠️ 技術棧
- **Framework**: LlamaIndex
- **Vector DB**: FAISS
- **Models**: HuggingFace (BGE Embeddings), OpenAI (GPT-4o-mini)
