# LlamaIndex 實戰：可持久化 RAG 與進階檢索策略（Week 2）

本專案紀錄了使用 **LlamaIndex + FAISS/Qdrant** 构建一個 **可持久化（Persistent）、可擴展（Scalable）** 且避免常見工程陷阱的 **RAG（Retrieval-Augmented Generation）** 系統的完整實戰流程。

內容不僅涵蓋「如何實作」，更深入解析 **為什麼這樣設計**，並系統性說明在實務中最容易出錯的關鍵環節。

---

## 📅 更新資訊

- **初版日期**：2026.01.06  
- **最新更新**：2026.01.21  
- **筆記版本**：v1.1.0  
- **Notebook**：`notebook.ipynb`  

---

## 🚀 核心重點

### 1. 文件建模與 Metadata 設計（Document Layer）

RAG 系統品質的基礎來自正確的資料建模：

- 使用 Wikipedia（中文）作為示範資料來源，模擬真實世界的非結構化文本  
- 為每份文件設計明確的 **Metadata**（如作者），作為後續檢索與過濾的重要依據  
- 自動化處理：透過 `glob` 與 `Path` 模組自動匹配文件名與對應 Metadata  

---

### 2. 進階切分策略：從固定長度到語意切分

- **SemanticSplitterNodeParser**  
  - 利用 Embedding 模型（`bge-m3`）尋找文本斷句點（Breakpoint）  
  - 確保每個 Node 內部語意一致性  

- **RecursiveCharacterTextSplitter**  
  - 基礎切分工具  
  - 針對中文標點符號（如「。」、「，」）進行優化  

---

### 3. 多樣化的向量儲存支援（Vector Stores）

除了基本的 FAISS，新增對分散式環境更友好的 Qdrant：

- **FAISS**  
  - 適合本地快速原型開發與單機檢索  

- **Qdrant**  
  - 支援非同步操作（`AsyncQdrantClient`）  
  - 更靈活的 Collection 管理，適合生產環境  

---

### 4. 向量索引的持久化與動態更新（Persistence & Update）

- **正確的持久化流程**  
  - 使用 `StorageContext.persist()`  
  - 示範如何從磁碟安全還原索引，避免建立空索引覆蓋舊資料  

- **動態更新索引**  
  - 使用 `insert_nodes()` 或 `ainsert_nodes()` 將新資料（如《銀之匙》）動態加入已存在索引  
  - 無需重新構建整個數據庫  

- **避免常見陷阱**  
  - **Embedding 維度不匹配**：查詢與索引皆使用相同的 `bge-m3`（1024 維）  
  - **StorageContext 邊界**：理解 Vector Store、Doc Store 與 Index Store 的組合關係  

---

### 5. Shared Storage Context 與 Multi-Index 管理（進階）

- **解決「幽靈索引」**  
  - 手動設定 `index_id` 並同步至 `index_store`  
  - 確保同一儲存目錄下管理多部作品（如《東京喰種》、《一拳超人》）不互相混淆  

- **索引清理**  
  - 正確刪除 LlamaIndex 自動產生的 UUID ID  
  - 保持 Metadata 資料乾淨  

---

### 6. 進階檢索策略（Advanced Retrieval）

- **Small-to-Big Retrieval（Sentence Window）**  
  - 使用單句提升精確度  
  - 回傳給 LLM 時透過 `MetadataReplacementPostProcessor` 補回上下文（Window）  

- **Auto-Retrieval**  
  - 透過 LLM 自動分析問題並推斷 Metadata 過濾條件（Metadata Filters）  
  - 實現語意驅動檢索策略  

- **Recursive Retriever（二層結構）**  
  - 建立 **Summary Index（摘要層）** 與 **Vector Index（全文層）** 的映射  
  - 檢索流程：先找出最相關的摘要節點 → 再深入該節點對應全文進行精檢索  

---

## 🛠️ 技術棧

- **Framework**：LlamaIndex  
- **Vector DB**：FAISS、Qdrant  
- **LLM**：OpenAI gpt-4o-mini、Ollama（gpt-oss:120b）  
- **Embedding**：BAAI/bge-m3
