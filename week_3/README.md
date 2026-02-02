# 進階檢索策略：DocumentSummaryIndex、查詢轉換與多模態實作 (Week 3)

本專案為「進階檢索策略」系列教學筆記（Week 3），涵蓋 DocumentSummaryIndex 的設計心法、查詢轉換技巧（HyDE / 多步拆解）、以及多模態檢索與 CLIP 類模型的整合實務。

---

## 📅 專案資訊

- **初版日期**：2026.01.25
- **筆記版本**：v1.0.0
- **教學週次**：Week 3
- **Notebook**：`notebook.ipynb`

---

## 🚀 核心重點總覽

### 1) DocumentSummaryIndex：以「摘要」為入口的文件級檢索
- 核心概念：先用 LLM 為整篇文件產生摘要並索引，查詢時在摘要層做匹配，命中文檔後再取出該文檔下所有 chunks，確保上下文完整性（避免單一 chunk 斷章取義）。
- 優點：更容易抓住文件整體語意，適合長文件或 chunk 無法單獨提供足夠上下文的情況。
- 風險：若文件非常長，命中文檔後會回傳大量 chunk，可能造成 LLM context window 或 latency 問題；通常建議搭配 summarize/response_mode（如 tree_summarize）。

DocumentSummaryIndex vs Recursive Retrieval（要點比較）

- DocumentSummaryIndex：為每份文件建立 summary node，查詢先比對摘要 → 回傳該文件所有 nodes。
- Recursive Retrieval：是一種檢索流程（可和多種 index 配合），查詢可在多層節點間遞歸深入篩選。

建議：文件級篩選使用 DocumentSummaryIndex + embedding retriever；實驗性或少量摘要可用 LLM retriever。

---

### 2) Retriever 模式：Embedding vs LLM
- EMBEDDING（向量檢索）  
  - 基於摘要/文件的向量相似度做搜尋，穩定、成本較低、可擴展性高。
- LLM（語意判斷）  
  - 將所有摘要文字交給 LLM 判斷相關性（可能更靈活但成本與延遲高），適合少量摘要或需複雜語意/規則判斷的場景。

similarity_top_k 的意義：決定要選出多少份「摘要文件」。選到的每個文件都會把其全部 chunks 拉出來，請注意回傳總量。

---

### 3) 查詢轉換（Query Transformations）
查詢轉換允許在查詢送入索引前／查詢執行過程中對查詢做改寫或拆解，常見應用：

- HyDE（Hypothetical Document Embeddings）
  - 流程：Query → 用 LLM 生成「假想答案（hypothetical document）」→ 將假想答案做 embedding → 用此向量做檢索。
  - 優點：將「問」轉為「答」，減少 query-document 非對稱性，提升召回語意相符的文件。
  - 缺點：每次檢索額外呼叫 LLM（成本+延遲）、依賴 LLM 生成品質、在精確事實查詢時若假想答案錯誤可能誤導檢索。
- StepDecompose / 多步拆解（Multi-Step）
  - 把複合查詢拆成多個子問題逐步執行（例如 StepDecomposeQueryTransform），能降低一次性多事實處理的混淆。
- HyDE 與 Recursive Retriever 的結合
  - 可先在頂層使用 HyDE 對摘要層做檢索，再遞歸進入各文件的內部 retriever 以取得細節（混合「全域語意」與「局部精準」）。

實務上，視資料量與延遲/成本容忍度選擇是否啟用 HyDE 或 LLM-based retriever。

---

### 4) Pandas Query Engine：用 LLM 做資料分析（含風險提示）
- 功能：把 DataFrame 封裝為一個 Query Engine，讓 LLM 生成可被 eval() 執行的 Python 表達式以回答問題（如統計、過濾、相關性分析）。
- 安全警告：內部使用 eval() 執行模型生成程式碼，存在 prompt-injection 與任意程式執行風險。僅建議在受信任或沙盒化環境使用，生產環境應做額外防護（限制輸出、審核/沙盒執行）。
- 常見用法：玩具範例（小表格、Titanic）、可自訂 prompt 以強制限制 LLM 只回傳純表達式（便於 eval）。
- 延伸：可把 PandasQueryEngine 作為某個「節點」的 query_engine，再用 RecursiveRetriever 將資料庫的檢索結果映射到對應的 pandas engine（適合內部財務／分析型檢索）。

---

### 5) 多模態（Multi-modal）與 CLIP 系列 Embeddings
- 多模態索引：同時支援文字與圖片的向量化，能進行 text→image、image→image 以及 text+image 混合檢索。
- CLIP 概念速覽：
  - 雙塔（image encoder + text encoder）把圖與文投射到同一共享嵌入空間，使用餘弦相似度衡量配對程度。
  - 優勢：零樣本檢索、語義對齊、跨語言圖文比對（取決於訓練資料與模型）。
- 實務選擇：
  - OpenAI CLIP：英文優化、穩定。
  - JinaAI / jina-clip-v2：支援多語言、與 LlamaIndex 兼容性好，適合跨語言檢索。
  - AltCLIP：使用多語言 text encoder（如 XLM-R），對中文與跨語言檢索友好，可作為自定義 MultiModalEmbedding。
- Vector store：可選 FAISS（本地）或 Qdrant（可持久化、支援多模態 collection），依場景選擇。

多模態檢索實務要點：
- 檢索時盡量提供詳細的圖片描述（或用 LLM 先生成描述），有助於找到語意配對的圖片。
- 圖片檢索可以同時接受 image queries（PIL / base64）或文字 query，也可混合兩者（QueryBundle）。
- 對於 image → QA workflow，常見做法是：retrieve 相關圖像 → 將 top-k 圖像與 prompt 組合 → 送入支援多模態的 LLM（如 Qwen3-vl）做合成回答。

安全與道德注意事項：多模態系統會涉及敏感圖像與內容的處理，務必遵守平台政策與法規，並在需要時加入內容審查與過濾機制。

---

## 🧠 設計原則（Design Philosophy）

- 以「文件級語意」為核心：對長文件優先做摘要索引，可提升檢索穩定性與語境完整性。  
- 組合檢索策略：將 embedding 與 LLM-based 判斷視為互補工具，根據資料量、延遲與成本做權衡。  
- 最小信任邊界：在需要執行模型產出的程式碼／多模態內容時，務必採取沙盒與審核流程。  
- 可觀察性與可回溯性：在實驗環境保留檢索 trace（哪些摘要被命中、top-k 檔案），便於故障分析。

---

## 🛠️ 技術棧（Tech Stack / Components）

- 檢索與索引： LlamaIndex（原名），DocumentSummaryIndex、VectorStoreIndex、MultiModalVectorStoreIndex、RecursiveRetriever  
- Embeddings： HuggingFace embeddings（BAAI/bge-m3）、ClipEmbedding、AltCLIP / jina-clip-v2  
- LLM 平台： Ollama（qwen3-vl、gpt-oss）、OpenAI（示例）  
- Vector Stores： FAISS（本地）、Qdrant（本地持久化 / 服務化）  
- 多模態處理： CLIP / AltCLIP / JinaAI  
- 資料處理： pandas（PandasQueryEngine）、SimpleDirectoryReader、SentenceSplitter  
- 可視化與追蹤：（可接 Arize / Phoenix 等監控工具）

---

## ⚡ 快速開始（How to get started）

1. 準備資料：將文本放到 week_3/data 或 images 放到 week_3/images。
2. 建立 DocumentSummaryIndex（或 VectorStoreIndex）並選擇適合的 embed_model 與 llm。
3. 根據需求選擇 retriever 模式：
   - 大量文件：使用 embedding retriever（similarity_top_k 控制命中文檔數）。
   - 少量摘要或需複雜語意判斷：可嘗試 LLM retriever（注意成本）。
4. 若欲提升召回或語意對齊，可在 query pipeline 加入 HyDEQueryTransform 或 StepDecomposeQueryTransform。
5. 多模態檢索：選用適合的 image/text embed model（AltCLIP 或 jina-clip-v2），並將 vector store 換成支持圖片 store 的 Qdrant 或 MultiModalVectorStore。
6. 若使用 PandasQueryEngine，務必在受控環境或加上 sandbox 措施。

---

## 參考與延伸閱讀

- LlamaIndex 文件（DocumentSummaryIndex、HyDE、RecursiveRetriever）  
- CLIP / AltCLIP / Jina-CLIP 官方 repo 與模型卡  
- 多模態 LLM（Qwen3-vl 等）使用說明與內容政策  

---

如需本筆記的範例程式碼、資料集或完整 Notebook（含可執行 cell），請參考同目錄下的 `notebook.ipynb`。若要把範例移植到生產環境，請特別評估成本、延遲與安全風險，並在必要情況下採用沙盒化或二次審核機制。