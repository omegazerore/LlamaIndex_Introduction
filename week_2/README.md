# 持久化 RAG：從資料建模到進階檢索（Week 2）

本專案為 RAG（Retrieval-Augmented Generation）工程實作的 **Week 2 教學筆記**，重點在於如何建立「可持久化（Persistent）、可擴展（Scalable）且穩健」的 RAG 系統。範例以 LlamaIndex 結合 FAISS / Qdrant，並示範常見陷阱與修正策略。

---

## 📅 專案資訊

- **初版日期**：2026.01.06
- **最新更新**：2026.01.21
- **筆記版本**：v1.1.0
- **Notebook**：`notebook.ipynb`
---

## 🚀 本週核心重點

1. 為何「持久化」重要  
   - RAG 是狀態型系統：向量/索引的生命週期、index_id 與 storage context 的一致性，直接影響系統可用性與維護成本。
   - 常見錯誤來源：embedding 維度不一致、向量雖然已寫進磁碟但無法被查到、隨資料成長檢索品質下降。

2. 建立可靠的 Document Layer（資料建模）  
   - 在資料輸入階段就規劃好 metadata（例如 author、source、file_name），metadata 不只是裝飾，而是後續 filter / auto-retrieval 的基礎。
   - 合理的 chunking（chunk_size / overlap / sentence splitter）會影響 retrieval 精準度與上下文完整性。

3. StorageContext 與 Index 管理（核心概念）  
   - StorageContext 是索引一致性的邊界（單一 StorageContext 下可管理多個 index）。
   - index_id 必須同步更新到 storage_context.index_store；否則會殘留 unused/ghost index。
   - Persist 與 Load 的正確順序與方式（例如 FaissVectorStore.from_persist_dir / Qdrant 的 Async client 設定）是關鍵。

4. 向量資料庫比較：FAISS vs Qdrant  
   - FAISS：輕量、單機、速度快（需注意向量維度與 index 型別對應）。  
   - Qdrant：支援持久化、過程中可異步操作、適合大規模部署與複雜過濾（metadata filtering / Auto-Retrieval）。

5. 進階檢索策略（提升精準度與上下文還原）
   - Small-to-Big Retrieval（Sentence Window）：單句 embedding + 回傳時擴展 window，適合長文件與精細事實查詢。
   - Auto-Retrieval（Metadata-driven）：LLM 推斷可能的 metadata filter，再在 filter 後的子空間做向量搜尋（降低雜訊）。
   - Hierarchical Nodes + AutoMergingRetriever：Leaf → Parent 的自動合併，平衡精準與上下文完整性。
   - SummaryIndex / RecursiveRetriever / IndexNode：建立多層索引與遞迴檢索管線，實現由粗到細的檢索流程。

---

## 🧠 設計原則（Design Philosophy）

- 系統化地把「工程痛點」變成可檢測的流程（例如：persist → load → query 驗證）。  
- 把 metadata 視為第一階檢索過濾器（在大資料量下先縮小候選集合）。  
- StorageContext = 一致性邊界（所有 index 的資源與 docstore 在此統一管理）。  
- 以可重現、可回滾的方式執行 persist / update（避免在 loop 中每次 persist，改為最後一次 persist）。

---

## 🛠️ 技術棧（Tech Stack）

- Framework：LlamaIndex (最新穩定版)  
- Vector DB：FAISS（IndexFlatL2） / Qdrant（AsyncQdrantClient）  
- Embeddings：HuggingFace Embedding（BAAI/bge-m3）  
- LLM-as-a-Judge & Response：Ollama（gpt-oss:120b-cloud）  
- Data loading：SimpleDirectoryReader / wikipediaapi  
- Text splitter：RecursiveCharacterTextSplitter、SentenceWindowNodeParser、HierarchicalNodeParser  
- 其他：Python、asyncio、shutil、pathlib

---

## ⚙️ Quick Start（重點步驟速覽）

1. 資料蒐集與 metadata 設計（範例使用 Wikipedia 中文）  
2. 建立 nodes（Semantic / Sentence / Hierarchical splitters）並產生 embeddings  
3. 建立 VectorStore（FAISS 或 Qdrant）並以 StorageContext 管理  
4. persist → load → query 驗證索引是否可用  
5. 若加入新資料：為新 nodes 計算 embedding → insert_nodes → 最後 persist storage_context  
6. 若共享 StorageContext 管理多個索引：手動 set_index_id 並同步 storage_context.index_store（避免殘留 auto-generated id）

（Notebook 已包含完整程式碼範例，請依序執行並觀察每一步的輸出）

---

## 📌 常見的陷阱與對策

- 向量維度錯誤（embedding 維度 d 與 FAISS index 建構必須一致）  
  → 將 embedding model 與 index 建立的 d 統一，並在 load 時避免手動初始化空 index。

- Persist 後查不到向量  
  → 使用官方提供的 from_persist_dir / from_persist_dir 方法還原 vector store，不要自行建立空的 index 再覆寫。

- 多次建立 index 導致 storage_context 中遺留多個無用 index_id  
  → 建立 index 後立即 set_index_id 並把原始 auto-generated id 刪除；在 loop 結束後才 persist。

- Auto-Retrieval 無法搭配 FAISS  
  → Auto-Retrieval 需要支援 metadata filtering 的 vector store，Qdrant 更適合此用例。

---

## 🔍 進階檢索速覽（你會在 Notebook 裡實作的幾種模式）

- Sentence Window（Small-to-Big）  
  - Node parser：SentenceWindowNodeParser  
  - Query 時使用 MetadataReplacementPostProcessor 還原 window 上下文

- Auto-Retrieval（Metadata-driven）  
  - 定義 VectorStoreInfo / MetadataInfo  
  - LLM 推斷 filter → 在 filter 子集做向量檢索（減少雜訊）

- HierarchicalNodeParser + AutoMergingRetriever  
  - 建立 leaf / mid / root 節點並把所有 node 存入 docstore（即使只有 leaf 被向量化）  
  - 檢索 leaf 後回溯 parent_id，判定合併條件，回傳父節點以提供完整上下文

- SummaryIndex / RecursiveRetriever / IndexNode  
  - 用 SummaryIndex 做 top-level overview（快速定位領域）  
  - 每個 summary 對應一個 IndexNode（指向下層 index）  
  - RecursiveRetriever 會從 top 層沿著 IndexNode 逐層遞迴檢索，並可選擇每層是否由 QueryEngine 用 LLM 直接合成答案

---

## ✅ Notebook 內容對應（你將學到的實作）

- 資料抓取、metadata 補齊與 Document 物件化  
- 各類 node parser（semantic / sentence / hierarchical / sentence window）的使用與比較  
- FAISS 與 Qdrant 的 StorageContext 建立、persist、load 範例  
- 如何新增 nodes、插入並更新現有 index（包含向量化步驟）  
- Shared StorageContext 的 multi-index 管理與 cleanup 範例（set_index_id、index_store 操作）  
- Auto-Retrieval、AutoMergingRetriever、RecursiveRetriever 與 IndexNode 的實務範例  
- SummaryIndex 與 response_mode（compact / refine / tree_summarize 等）差異實驗

---

## 🏁 回家挑戰（Homework）

- 嘗試把現有的 RecursiveRetriever pipeline persist 下來，並重新從磁碟還原後驗證遞迴檢索是否仍能正確找到下層索引（特別注意 storage_context.docstore 是否包含所有層級的 node）。  
- 將 Auto-Retrieval 的 metadata 設計延伸：加入多個 metadata 欄位（例如 publication_year、genre），觀察 LLM 推斷 filter 的效果與檢索回傳品質的變化。  
- 將 FAISS 改為 HNSW 索引或其他 ANN 結構，測試在大量資料下的查詢效能與 recall 變化。

---

若要進一步實作，請直接打開 notebook.ipynb，按單元格順序執行並觀察每一步的輸出與 persist 檔案（./week_2/ 目錄下的 storage_*）。祝你在建構可持久化的 RAG 系統時事事順利！