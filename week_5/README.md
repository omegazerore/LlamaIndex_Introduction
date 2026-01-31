# RAGAS 實戰：Automated Evaluation & Knowledge Graph Testset Generation (Week 5)

本專案為 RAG（Retrieval-Augmented Generation）系列實作的 **Week 5 教學筆記**，聚焦於 **RAG 系統的量化評估** 與 **自動化測試集生成**。

---

## 📅 專案資訊

- **初版日期**：2026.01.31   
- **筆記版本**：v1.0.0  
- **Notebook**：`notebook.ipynb`  

---

## 🚀 核心重點總覽

### 1. RAGAS 評估三部曲：從「檢索」到「生成」

RAG 系統的錯誤來源可拆解為「找錯資料」與「生成幻覺」。RAGAS 提供結構化指標，將評估拆分為以下三個層次：

#### 🔍 檢索品質（Retrieval）
- **Context Precision**  
  評估真正相關的片段是否被排在前面，影響模型取得關鍵資訊的效率。
- **Context Recall**  
  檢查檢索結果是否完整覆蓋回答所需的所有事實。

#### ✍️ 生成品質（Generation）
- **Faithfulness（忠實度）**  
  驗證答案是否嚴格依據上下文，有效抑制幻覺（Hallucination）。
- **Answer Relevance（相關性）**  
  衡量回答是否切中問題核心，而非流於表面敘述。

#### 🎯 端到端指標
- **Answer Accuracy（正確性）**  
  在有 Ground Truth 的情況下，直接比對生成答案與標準答案的一致性。

---

### 2. 合成資料生成（Synthetic Data Generation）

為解決缺乏人工標註資料的痛點，本專案實作 **自動化測試集生成流程**，可直接從原始文件產生：

(Question, Context, Ground Truth)

重點設計包含：

- **Single-hop / Multi-hop 問題**  
  從單一文件理解，到跨文件推理的複雜問題生成。
- **Persona-driven 測試集**  
  模擬不同背景與需求的使用者角色，提升資料多樣性與真實性。

---

### 3. 知識圖譜（Knowledge Graph, KG）導向設計

RAGAS v0.4 引入圖結構來驅動合成資料生成，模擬人類的「跨文本推理」過程。

#### 🧩 特徵提取（Extractors）
- Headlines
- Keyphrases
- Summary  
作為圖節點的語意標籤。

#### 🔗 關係建立（Relationship Builders）
- **Jaccard Similarity**  
  依據標題重疊程度建立文件連結。
- **OverlapScore**  
  結合 Jaro-Winkler 距離與關鍵詞重疊，建立語意邊（Semantic Edge），為 Multi-hop 推理的基礎。

---

### 4. 噪音敏感度（Noise Sensitivity）分析

真實檢索情境往往伴隨大量雜訊。本週實作 **NoiseSensitivity** 指標，用以評估模型的抗干擾能力。

- **Relevant Noise**：主題相關但內容錯誤的資訊  
- **Irrelevant Noise**：與問題完全無關的雜訊  

#### 邏輯鏈條分析
透過以下運算邏輯解析模型行為：

(Context ∩ GT) ∧ (Context ∩ Answer) ∧ ¬GT

用以判斷模型是否能在噪音存在下維持正確推理。

---

## 🧠 核心設計原則（Design Philosophy）

- **數據驅動開發（DDE）**  
  優化不再憑直覺，而是以 RAGAS 指標變化為依據。
- **自動化回歸測試**  
  將評估流程納入 CI/CD，確保每次 Pipeline 更新不破壞 Faithfulness。
- **圖結構優於線性結構**  
  利用 KG 捕捉文件間隱性關聯，生成更具挑戰性的測試題。

---

## 🛠️ 技術棧（Tech Stack）

- **Framework**：RAGAS (v0.4+)  
- **LLM-as-a-Judge**：Ollama (`gpt-oss:120b-cloud`)  
- **Embeddings**：HuggingFace (`BAAI/bge-m3`)  
- **KG Components**：  
  - HeadlineSplitter  
  - OverlapScoreBuilder  
  - SummaryExtractor  
- **Data Handling**：  
  - LangChain `DirectoryLoader`  
  - OpenCC（繁簡轉換）  
- **Language**：Python  

---


