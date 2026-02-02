# 進階檢索策略 - Part 4 (Week 4)

本專案為 RAG（Retrieval-Augmented Generation）系列的 Week 4 教學筆記，聚焦於「重排（Re-Ranking）」與「混合檢索（Hybrid Retrieval）」的進階實作與策略選擇。內容涵蓋本地開源 reranker（如 BGE-Reranker）、LLM-based reranker（RankGPT / Cohere Rerank 4）、以及 BM25、QueryFusion 等混合式檢索器實作與調校建議。

---

## 📅 專案資訊

- 初版日期：2026.02.02  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`  
- 週次：week_4
---

## 🚀 本週核心重點

1. 理解 ReRank 在兩階段檢索流程中的角色與 trade-off  
2. 在本地部署高品質 Cross-Encoder reranker（FlagEmbeddingReranker / BGE-Reranker）  
3. 使用 LLM 做 Listwise ReRanking（RankGPT / Cohere Rerank 4）以取得更強的語意判別力  
4. 掌握 BM25 的使用方式、metadata filtering 與中文 tokenization（jieba / BPE）  
5. 結合向量檢索與關鍵字檢索的 Hybrid Retrieval（QueryFusionRetriever），並理解多種融合模式（RRF / Relative / Distribution-based / Simple）

---

## 🔍 ReRank（重新排序）概念總結

- 兩階段檢索（Retrieval Funnel）：
  - 第一階段：向量檢索或 BM25 快速召回大量候選（例如 30–100 筆）
  - 第二階段：用高精度 reranker（cross-encoder / LLM）在候選中精選 top_n，提供給 LLM 作為上下文
- top_n：決定最終傳給 LLM 的文件數量，是準確度、成本與延遲間的平衡參數

建議值（參考）
- 一般問答（FAQ、客服）：top_n = 3–5  
- 技術引用、需引用文件：top_n = 5–8  
- 複雜推理、多來源整合：top_n = 10+

---

## 🛠 主要 ReRank 方案比較

| 方案 | 成本 | 速度 | 適用場景 |
|---|---:|---:|---|
| BGE (Local, FlagEmbeddingReranker) | 低（需 GPU） | 中 | 隱私要求高、想在本地執行高品質 rerank |
| RankGPT（LLM listwise） | 高（API token） | 慢 | 對語意理解與長文本排序需求極高 |
| Cohere Rerank 4（託管） | 中（按次計費） | 快 | 企業級生產、多語言與長上下文需求 |

---

## FlagEmbeddingReranker（BAAI/bge-reranker 系列）

- 類型：Cross-Encoder（同時輸入 query 與 doc，計算精準相關性分數）
- 優勢：在少量候選上提供非常精準的排序結果
- 取捨：計算成本高（無法大規模即時對大量候選排序），通常搭配第一階段 Vector Search（粗選）使用
- 實務建議：將前段候選數設定為 30–100，再用 reranker 精選 5–20 筆傳給 LLM

---

## GPTReranker（RankGPT / Listwise）

- 方法：將 query 與整個候選列表一起送入 LLM，要求輸出依相關性排序的文件編號（Listwise）
- 優勢：充分利用大型模型的推理能力，能理解細緻的語意與上下文需求
- 欠缺：成本（token）與延遲較高，適合用作「最後一道精選」或高價值場景

---

## Cohere Rerank 4（託管商業方案）

- 特性：
  - 支援 32k tokens 的長上下文
  - 多語言與跨語言能力優秀
  - 提供 Pro 與 Fast 版本以平衡準確性與延遲
- 適用情境：企業級應用、需要處理跨語言或長文檔的重排序任務

---

## 🔁 多層 / 複數 ReRank 的價值

為何採用多重 ReRank？
- 初層快速篩選（速度與成本優先）→ 後層精排（精度優先）
- 保持高 recall 的同時逐步提升 precision
- 結合不同訊號（向量相似、關鍵字重疊、LLM-based relevance）使排序更穩定
- 最終提供更乾淨、集中且有助 LLM 生成的上下文

範例流程：Vector Search (top_k=50) → BGE-Reranker → Cohere/RankGPT（最終 top_n）

---

## 🔀 Hybrid Retriever（BM25 + Semantic）

當向量檢索出現召回不足（未找到關鍵字或精確說法）時，BM25 與其他關鍵字檢索器能補強召回率。Hybrid Retriever（如 QueryFusionRetriever）結合二者優點，常見做法：

- BM25：精確字詞 / 專有名詞、數字、程式碼匹配
- FAISS（向量）：語意近似、同義改寫捕捉
- QueryFusion：整合多檢索器結果、使用 RRF 或其他融合策略重新排序

---

## BM25 使用方式（快速要點）

兩種常見建立 BM25 的方式：
1. 直接從 nodes 建立（適合小型資料、快速測試）
2. 先存入 docstore（SimpleDocumentStore / Redis / MongoDB），再建立 BM25Retriever（適合大型或分散式系統）

Metadata filtering：可在 BM25 檢索時加入 metadata filters（例如 headquarter=San Francisco）以縮小檢索範圍。

中文處理：需注意 tokenizer。常見工具：jieba（可搭配繁體字典 dict.txt.big），或自行實作 BPE / tokenization pipeline。

---

## QueryFusionRetriever 與 RAG-Fusion 思想

- QueryFusionRetriever 可以同時呼叫多個 retriever（向量 + BM25），並支援延伸查詢（num_queries）由 LLM 生成多個改寫查詢以增加召回多樣性。
- num_queries：會產生除原查詢外的多個擴充查詢（例如設定為 4，則總查詢為原始 + 3 個改寫查詢）
- 融合模式（FUSION_MODES）：
  1. RECIPROCAL_RANK（RRF，預設）— 根據排名加權合併，簡單且穩健  
  2. RELATIVE_SCORE — 將不同檢索器的分數正規化後平均  
  3. DIST_BASED_SCORE — 考慮分數分佈（平均與標準差）  
  4. SIMPLE — 基本串接或加總，適合作為 baseline

RAG-Fusion 的核心：用多角度查詢補足原始查詢不確定性，並用融合策略降低單一檢索偏誤。

---

## 中文 Tokenization / 實務技巧

- 建議使用 jieba 並載入繁體字典（dict.txt.big）以改善繁體中文分詞結果
- 在 BM25 或其他基於詞的檢索下，良好的 tokenizer 是召回率與精準度的關鍵
- 若需更精細控制，可考慮 BPE 或自訂詞彙表（視 embedding model 與向量化策略而定）

---

## 🧰 技術棧（Tech Stack）

- 檢索索引 / Framework：LlamaIndex (舊稱 GPT Index)  
- 向量資料庫：FAISS  
- 本地 reranker：FlagEmbedding / BGE-Reranker（BAAI）  
- LLM（推理 / RankGPT）：Ollama（gpt-oss:120b-cloud）或其他託管 LLM  
- 託管 ReRank：Cohere Rerank v4  
- Keyword Retriever：BM25Retriever（支援 docstore）  
- 中文分詞：jieba（可搭配繁體字典）  
- 觀測 / tracing（選項）：Arize Phoenix (phoenix)  
- Embeddings：BAAI/bge-m3（示例）  
- 開發語言：Python

---

## ✅ 實務建議 (Best Practices)

- 先評估第一階段檢索的召回（Recall），再決定是否要加重排或混合檢索策略  
- 在有限 GPU / CPU 下，用小範圍（top_k 30–100）做 Cross-Encoder 重排，避免一次處理大量候選  
- 結合 metadata filter 可顯著提升精準度（特別在企業文件與結構化資料中）  
- 對於多語言或長輸入場景，考慮 Cohere Rerank 4 或類似長上下文支援的託管方案  
- 使用 QueryFusionRetriever 時，調整 num_queries 與 fusion mode 以找到最適合場景的召回/精準平衡

---

## 開始使用（快速檢查清單）

- 將資料放置：week_4/data/*.txt  
- 若使用中文分詞：準備 jieba 繁體字典（week_4/dict.txt.big）並載入  
- 若使用本地 BGE reranker：確認可用 GPU 並安裝 FlagEmbedding  
- 建立 index（FAISS + HuggingFaceEmbedding），再搭配 reranker / BM25 / QueryFusion Retriever 進行測試

---

若需範例程式片段、Notebook 中的完整實作或範例查詢測試，請開啟同目錄下的 `notebook.ipynb`。