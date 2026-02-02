# RAGAS 實作：基於知識圖譜的合成測試集（Week 6）

本專案為 RAG（Retrieval-Augmented Generation）系列教學的 **Week 6 筆記**，重點放在利用 RAGAS 的 Knowledge Graph（KG）來產生 Single-hop 與 Multi-hop 的合成測試資料，並針對生成結果做自動化評估。

---

## 📅 專案資訊

- 初版日期：2026.02.02  
- 筆記版本：v1.0.0  
- Notebook：`notebook.ipynb`（Week 6）  
- 目錄：約定資料夾為 `week_6/`（示範程式碼中使用）

---

## 🚀 本週核心重點摘要

1. Single-hop 與 Multi-hop 的差異與設計取向  
   - Single-hop：每個 QA pair 僅依賴單一節點（node）來完全回答，適合評估檢索精準度與答案忠實度（faithfulness）。  
   - Multi-hop：問題必須結合兩個以上節點的資訊才能完整回答，通常透過節點間的關聯（relations）來定義可用的組合。

2. Persona（角色）在合成資料中的作用  
   - Persona 用於模擬不同使用者的提問視角（語氣、關注點、長度），不改變事實來源，只影響問題的表達方式與細節取向。

3. Prompt 設計要點  
   - QueryCondition 與 QueryAnswerGenerationPrompt 的設計需強調「答案必須忠實於提供的 context」，避免模型加入未出現在 context 的資訊。
   - Multi-hop prompt 要明確標示各背景段落（如 `<1-hop>`、`<2-hop>`），並要求答案必須整合多段資訊。

4. 合成流程與抽樣控制  
   - 透過 TestsetGenerator 與 query_distribution 控制不同 synthesizer（single-hop / multi-hop）與 property（如 `keyphrases`）的採樣比例。

5. 自動化評估（LLM-as-a-Judge）  
   - 使用 RAGAS 提供的指標：Faithfulness、AnswerRelevancy、ContextPrecision、NoiseSensitivity 等，並以異步化（asyncio + semaphore）來控制大量 LLM 請求的併發數量與速率。

6. 節點過濾與關聯品質把關  
   - 對於 Multi-hop，建議在使用 OverlapScoreBuilder 建立關係後，再以節點摘要向量（summary_embedding）做 cosine 相似度過濾，以降低語義上不合理的配對（阈值可視情況調整，例如 0.7）。

---

## 🧩 Single-hop 合成（核心概念）

- 定義：Single-hop 的 QA 必須能從單一 node 的內容完整推導。節點切分應以「一個可獨立回答的完整語意單位」為標準，避免 chunk 太小導致資訊不足或誘發模型幻覺。
- 生成管線重點：Persona → QueryCondition → QueryAnswerGenerationPrompt → SingleHopSpecificQuerySynthesizer → TestsetGenerator。

實務上常見注意事項：
- 保證 context 的資訊足夠支撐答案（否則答案會被迫補推理）。  
- Prompt 明確限制「僅使用 context」，並在 examples 中示範正確範例。

---

## 👥 Persona 設計範例（已在 Notebook 實作）

示例 persona 類型（本週示範）：
- 金庸新手讀者（入門、需明確解釋）
- 資深金庸迷（注重細節與一致性）
- 角色導向讀者（關注人物動機）
- 武俠世界觀研究者（結構性比較）
- 記憶模糊的休閒讀者（澄清與糾錯）

用途：提升測試集多樣性，使得合成題目更貼近真實使用情境。

---

## ✍️ Prompt 設計要點（Query / Multi-hop）

- Single-hop Prompt：強調「答案不得使用 context 以外的資訊」，並給出多樣範例（不同 persona、query_style、query_length）。
- Multi-hop Prompt：要求引用並組合至少兩段以 `<1-hop>`、`<2-hop>` 等標記的背景，說明必須使用多段資訊來回答。

提示：RAGAS 內部還有一組 enum（QueryLength、QueryStyle）用於控制噪音與格式，請注意與你在自然語言 prompt 裡設定的描述（如「正式」、「敘事型」）並非相同層級。

---

## 🗂️ 知識圖（KG）載入與準備

- 範例：從檔案載入 Single-hop KG（`week_5/knowledge_graph_single_hop.json` 的延續或相似檔案）：
  - KnowledgeGraph.load(...)
- 在產生測試集前，確保每個 node 已具備必要屬性（如 `page_content`、`keyphrases`、`summary`、`summary_embedding` 等），並已建立關係（relationship），供 synthesizer 選取。

---

## ⚙️ Testset 生成（範例流程）

1. 建立 LLM 與 Embeddings（例如 Ollama gpt-oss:120b-cloud 與 HuggingFace BAAI/bge-m3）。
2. 建置 TestsetGenerator，傳入 knowledge_graph、persona_list 與 embedding_model。
3. 定義 query_distribution（mapping synthesizer → weight）。
4. 呼叫 generator.generate(testset_size=N, query_distribution=...)，取得合成測試集（Single-hop / Multi-hop）。

輸出建議：
- 將生成的 DataFrame 加上 testcase_id（例如使用 sha256(user_input|reference) 的前 16 bytes），作為測試案例唯一識別。
- 儲存為 Dataset（RAGAS 的 Dataset class）並 persist（local/csv）以便後續回溯與評估。

---

## 🧪 生成回應並做批次評估（範例）

- 可把合成測試集用 LlamaIndex（或其他 RAG pipeline）做檢索並生成回應，示範流程包含：
  - 建立 VectorStoreIndex（Faiss）並用 SentenceWindowNodeParser 做節點切分。
  - 建立 query_engine 並做批次查詢（asyncio + batch size 控制）。
  - 將 response、retrieved_contexts、reference 與 testcase_id 一起存入 Dataset。

評估流程示例（建議）：
- 使用 RAGAS 指標套件：Faithfulness、AnswerRelevancy、ContextPrecision、NoiseSensitivity。
- 以 asyncio.gather 並搭配 Semaphore 控制並發量，避免超量向 Ollama 發送請求。
- 最後輸出 exp_results.to_pandas() 並儲存為 CSV（如：`week_6/test_dataset_synthesized_benchmark_evaluation.csv`）。

---

## 🔁 Multi-hop 合成（核心邏輯與過濾）

1. 關係來源：使用 OverlapScoreBuilder（或其他 relationship builder）來判定哪些 nodes 具備重疊可作為 multi-hop 的候選 pair（關鍵 property 例如 `overlapped_items`）。
2. 篩選條件：relation 必須包含文字型的 overlapped_items，且整體重疊分數需超過 threshold。
3. 二次過濾（推薦實務步驟）：
   - 對 candidate relationships 以節點的 summary_embedding 做 cosine similarity 計算，剔除語意上不夠接近的配對（例如只保留 sim >= 0.7）。
   - 這樣可以降低文字層級相似（false positive）導致的無意義 multi-hop 組合。
4. 使用 MultiHopSpecificQuerySynthesizer 產生多跳題，並同樣以 persona 與 prompt 控制風格與長度。

---

## ⚠️ 常見提醒與實務技巧

- Jupyter 與 asyncio：Notebook 內執行 RAGAS 的 async 任務時，常需要 nest_asyncio.apply() 來避免事件迴圈衝突。
- Embedding 向量維度：若使用 Faiss 等 index，建立索引時的向量維度（d）必須與 embedding model 輸出維度一致。
- QueryStyle enum：RAGAS 內部的 QueryStyle（如 LONG/MEDIUM/SHORT、MISSPELLED、WEB_SEARCH_LIKE）與 prompt 裡的「自然語言風格描述」不屬同一設定層級，使用時請區分。
- NoiseSensitivity 解讀：0 並不一定代表無問題，需搭配 Faithfulness 與 ContextRecall 一起判讀。

---

## 🛠️ 技術棧（本週範例）

- Framework：RAGAS (v0.4+)  
- LLM-as-a-Judge / Generator：Ollama（gpt-oss:120b-cloud）  
- Embeddings：HuggingFace（BAAI/bge-m3）  
- Vector Store：Faiss（示範）  
- KG Components：HeadlinesExtractor、KeyphrasesExtractor、SummaryExtractor、OverlapScoreBuilder、CosineSimilarityBuilder  
- Data Loader：LangChain DirectoryLoader / LlamaIndex SimpleDirectoryReader  
- 工具：OpenCC（繁簡轉換）、nest_asyncio、asyncio  

---

## 設計哲學（Design Principles）

- 以「知識圖（KG）」為中心，模擬跨文本推理路徑，提升 Multi-hop 題目的挑戰度與多樣性。  
- 生成流程強調「答案必須忠實於提供的上下文」，以降低模型幻覺（hallucination）。  
- 評估流程採用 LLM-as-a-Judge，使得在缺乏大量人工標註時仍能進行可量化、可回溯的測試。  
- 把合成測試納入 CI/CD 回歸測試，確保 RAG pipeline 變動不降低 Faithfulness。

---

如需更完整的程式範例、Prompt 範本或實作細節，請參閱同目錄下的 `notebook.ipynb`（Week 6）。若要複製執行，請先確認 Ollama 與對應 embedding model 已正確設定並能在本機或雲端呼叫。祝你在生成與評估合成測試集的實驗中順利！