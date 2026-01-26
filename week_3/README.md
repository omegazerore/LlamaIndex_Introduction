# LlamaIndex 進階檢索策略與多模態 RAG（Week 3）

本專案紀錄 **LlamaIndex 在進階檢索（Advanced Retrieval）與 Query Transformation** 上的實戰應用，涵蓋從 **文件級摘要索引（DocumentSummaryIndex）**、**HyDE / Multi-step Query Transform**，一路延伸到 **Pandas Query Engine** 與 **多模態（Multi-modal）RAG** 的完整實作。

本週重點不只是「功能展示」，而是聚焦在 **檢索邏輯設計、LLM 角色分工，以及在真實系統中如何避免 Context Explosion 與語意錯配問題**。

---

## 📅 更新資訊

- **初版日期**：2026.01.25  
- **筆記版本**：v1.0.0  
- **教學週次**：Week 3  
- **Notebook**：`notebook.ipynb`

---

## 🚀 核心主題概覽

### 1. DocumentSummaryIndex：以簡馭繁的文件級檢索

**DocumentSummaryIndex 的設計哲學：**

> 不先檢索 Chunk，而是先理解「整份文件在說什麼」。

- 透過 LLM 為**每一份文件生成摘要（Summary Node）**
- 查詢時先比對摘要，再回傳該文件下的所有節點
- 從「全局理解」導向「局部細節」，避免單一 Chunk 斷章取義

**與 Recursive Retrieval 的差異：**

- DocumentSummaryIndex 是 **索引類型**
- Recursive Retrieval 是 **檢索邏輯**
- 兩者可獨立使用，也能互補搭配

---

### 2. Retriever 設計：Document-level 的 similarity_top_k 思維

- `similarity_top_k` 在這裡代表的是 **文件數量，而非 Chunk 數**
- 一旦文件命中，該文件下的所有 Nodes 都會被拉取
- 優點：上下文完整  
- 風險：文件過長會導致 **Context Window 爆炸**

**實務建議：**
- 搭配 `response_mode="tree_summarize"`
- 適合中小量文件，不適合超大型單文件

---

### 3. Query Transformations：把「問問題」變成「設計查詢流程」

LlamaIndex 允許在檢索前、檢索中、甚至檢索後進行 **Query Transformation**：

**常見使用情境**
- 將問題轉成更適合向量搜尋的形式（HyDE）
- 將複雜問題拆解為子問題（Step Decompose）
- 多步推理，逐層縮小搜尋空間

---

### 4. HyDE（Hypothetical Document Embeddings）

HyDE 的核心不是「猜答案」，而是：

> **用「答案的形式」去搜尋「答案」**

解決傳統 RAG 中常見的問題：
- Query 太短
- 文件太長
- 向量語義不對齊（Query–Document Asymmetry）

**優點**
- 顯著提升語意檢索品質
- 對開放式問題特別有效

**限制**
- 額外 LLM 成本
- 不適合精確事實查詢（數值、年份）

---

### 5. Recursive Retriever × HyDE / SDQT（多層檢索架構）

本週示範多種組合策略：

- **HyDE + RecursiveRetriever**
- **StepDecomposeQueryTransform + RecursiveRetriever**

典型結構：
1. 上層：摘要節點（Summary / IndexNode）
2. 下層：全文向量索引
3. Query 先在摘要層過濾，再深入全文層精檢索

這種設計有效解決：
- 文件量大
- 主題分散
- LLM 容易迷失重點的問題

---

### 6. Pandas Query Engine：讓 LLM 直接「操作資料」

- 使用 LLM 將自然語言轉換為 Pandas 可執行的 Python 表達式
- 適合：
  - 數據分析
  - 內部報表查詢
  - 結構化資料問答

⚠️ **安全提醒**
- 內部使用 `eval()`
- 僅適合沙盒或受信任環境
- 不建議直接用於 Production 對外服務

---

### 7. Recursive Retriever × Pandas Query Engine（進階示範）

- 向量索引只負責「找對資料來源」
- 命中後將 Query 轉交給對應的 Pandas Query Engine
- 實現 **文件檢索 + 表格計算** 的混合式 RAG 架構

---

### 8. Multi-modal RAG：文字 × 圖片的語義對齊

- 使用 **Qdrant** 作為 Text / Image 雙向量儲存
- 引入 **CLIP / Jina CLIP / AltCLIP**
- 支援：
  - Text → Image
  - Image → Image
  - Text + Image 聯合檢索

核心理解：

> **CLIP 不是在辨識圖片，而是在對齊語義空間。**

---

### 9. Image Query Engine：圖像檢索 × LLM 合成

- Retriever：找出最相關圖片
- Query Engine：
  - 組合 Prompt
  - 將圖片與文字一併送入多模態 LLM（如 qwen3-vl）
- 實現 Image-based QA 與內容生成

---

## 🛠️ 技術棧

- **Framework**：LlamaIndex  
- **Vector DB**：FAISS、Qdrant  
- **LLM**：Ollama（gpt-oss:120b、qwen3-vl）、OpenAI gpt-4o  
- **Embedding**：
  - BAAI/bge-m3
  - OpenAI CLIP
  - jinaai/jina-clip-v2
  - BAAI/AltCLIP  

---

## 🎯 教學核心精神

這一週的重點不是「API 怎麼用」，  
而是 **如何設計一個不會隨資料規模崩壞的檢索系統**。
