# RAGAS 實戰：評估指標深度解析與合成資料生成（Week 4）

本專案聚焦於 **RAG（Retrieval-Augmented Generation）系統開發中的「最後一哩路」：品質評估與保證**。  
核心探討的問題是：

> **如何以可量化、可重現，且不高度依賴人工標註的方式，評估你的 RAG 系統？**

Week 4 以 **RAGAS** 為核心框架，示範如何建立一套完整的 **自動化 RAG 評估與測試資料生成流程**，涵蓋評估指標設計、合成測試集構建，以及與 LlamaIndex 的端到端整合實測。

---

## 📅 更新資訊

- **初版日期**：2026.01.12  
- **筆記版本**：v1.0.0  
- **Notebook**：`notebook.md`（Current Release）

---

## 🚀 核心重點

### 1️⃣ RAGAS 評估框架：LLM-as-a-Judge

傳統指標（如 BLEU、ROUGE）難以衡量 RAG 系統的核心價值。  
RAGAS 採用 **LLM-as-a-Judge** 的設計，從「檢索」與「生成」兩個獨立維度進行量化評估。

#### 🔹 檢索品質（Retrieval）

- **Context Precision（內容精準度）**  
  評估與問題相關的資訊是否被排序在檢索結果的前段。

- **Context Recall（內容召回率）**  
  衡量檢索內容是否完整涵蓋生成正確答案所需的關鍵資訊。

#### 🔹 生成品質（Generation）

- **Faithfulness（忠實度）**  
  評估回答是否完全基於提供的上下文，嚴格偵測與過濾幻覺。

- **Answer Relevance（答案相關性）**  
  判斷回答是否直接回應問題，避免冗餘或偏離主題的內容。

- **Answer Accuracy（答案正確性）**  
  與標準答案（Ground Truth）進行語意層級的端到端一致性比對。

📌 **核心觀念**  
評估必須拆解為「檢索」與「生成」兩個獨立面向，才能精準定位 RAG 系統的實際瓶頸。

---

### 2️⃣ 知識圖譜導向的合成資料生成（Testset Generation）

針對 RAG 系統在測試階段常見的 **「缺乏標註資料」** 問題，RAGAS 透過 **Knowledge Graph（知識圖譜）** 自動生成高品質測試集。

- **Transforms Pipeline**  
  結合 `HeadlinesExtractor`、`KeyphrasesExtractor` 與 `HeadlineSplitter`，  
  將非結構化文件轉換為具備語意關聯的節點。

- **Relationship Builder**  
  使用 **Jaccard Similarity** 與 **Cosine Similarity** 建立節點間的語意連結，  
  支援後續 **Multi-hop 推理問題** 的自動生成。

- **Personas（多樣化角色）**  
  模擬不同使用者情境（如新手旅客、憤怒的商務客），  
  讓生成的測試問題更貼近真實語氣與使用場景。

---

### 3️⃣ 進階評估指標：噪音敏感度（Noise Sensitivity）

RAG 系統常會檢索到與問題無關的雜訊內容，此指標用於衡量：

> **LLM 是否會被上下文中的錯誤資訊誤導？**

- **計算邏輯**  
  當上下文中包含正確答案，但 LLM 卻選擇了  
  「不在 Ground Truth 中的錯誤資訊」作答時，觸發懲罰。

- **重要提醒**  
  必須與 **Faithfulness** 指標一併觀察，  
  避免將「檢索完全錯誤導致的 0 分」誤判為「抗噪能力強」。

---

### 4️⃣ LlamaIndex 整合實戰：SentenceWindowNodeParser

本章節實作了一個 **RAGAS 航空助理範例**，並整合 Week 2 的進階策略：

- **檢索端**  
  使用 `SentenceWindowNodeParser`，確保檢索內容包含足夠的上下文視窗  
  （`Window Size = 2`）。

- **生成端**  
  使用 **Ollama Cloud Service** 提供的 `gpt-oss:120b-cloud` 進行回應生成。

- **自動化流程**  
  從文件載入、Index 建立，到使用 RAGAS 進行  
  **非同步（Async）批次評估**，完成全自動化 Benchmark 跑分流程。

---

## 🧠 核心設計原則（Takeaways）

- **沒量化，就沒優化**  
  透過自動化評估指標，快速驗證檢索策略（如 Chunk Size、Top-K）的改進成效。

- **合成資料的價值在於多樣性**  
  結合 Persona 與 Knowledge Graph，有效覆蓋邊緣案例（Edge Cases）。

- **非同步與平行化（Async / Parallel）**  
  在大規模評估實驗中，可顯著降低整體實驗時間成本。

- **指標需互補解讀**  
  單一指標容易產生誤導（例如 Faithfulness 高但 Answer Relevance 低），  
  應整體觀察 Metrics Radar 以獲得全面視角。

---

## 🛠️ 技術棧

- **Framework**：RAGAS, LlamaIndex  
- **Evaluation**：LLM-as-a-Judge  
  （Faithfulness, Answer Relevance, Context Precision, Context Recall, Noise Sensitivity）  
- **Embedding Models**：HuggingFace（`BAAI/bge-m3`）  
- **Node Parser**：SentenceWindowNodeParser  
- **LLM**：Ollama Cloud Service（`gpt-oss:120b-cloud`）  
- **Data Tech**：Knowledge Graph, Synthetic Data Generation
