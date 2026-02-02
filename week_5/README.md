# RAGAS 實作筆記：RAG 評估與單跳知識圖生成 (Week 5)

本專案為 RAG（Retrieval-Augmented Generation）系列的 Week 5 教學範例，重點在於使用 RAGAS 工具鏈進行：
1) RAG 系統的量化評估指標實作與示範；  
2) 從原始文件建構 Knowledge Graph（KG），並準備作為後續合成測試集的基礎（Single-hop 範例）。

---

## 📅 專案資訊

- **初版日期**：2026.01.31
- **筆記版本**：v1.0.0
- **Notebook**：`notebook.ipynb`

---

## 🚀 本週重點摘要

1. 為何要用 RAGAS？  
   - 傳統 LLM 評估無法有效區分「檢索失誤」與「生成幻覺」。RAGAS 採用 LLM-as-a-Judge，自動化量化多個 RAG 關鍵指標，降低人力成本並提升可重現性。

2. 指標體系（Retrieval × Generation）  
   - 檢索面：Context Precision（排序與前置相關性）、Context Recall（事實覆蓋）。  
   - 生成面：Faithfulness（忠實度）、Answer Relevance（答案針對性）、Answer Accuracy（有 Ground Truth 時的正確性）。  
   - 進階：Noise Sensitivity（噪音敏感度），需與 Context Recall / Faithfulness 並行解讀。

3. 合成資料生成（Synthetic Testset）  
   - 自動從文本生成 Question / Context / Ground Truth，支援 Single-hop、Multi-hop 與 reasoning-based 題型。  
   - 可透過 persona-driven 設計提高測試集多樣性（模擬不同使用者場景）。

4. Knowledge Graph（KG）導向設計  
   - 將文件拆成節點（nodes），在節點上抽取各種特徵（headlines、keyphrases、summary、embeddings 等），再以相似度或重疊度建立關係（relationships）。  
   - OverlapScoreBuilder、Jaccard / Cosine Builders 等是形成 Multi-hop 路徑的重要元件。

---

## 🧾 Notebook 中的重點流程（概覽）

- 資料準備  
  - 範例使用 wikipediaapi 與 OpenCC 取得並轉換為繁體中文的文本（本範例抓取多部武俠小說頁面作示例）。
  - 使用 LangChain 的 DirectoryLoader 讀取文本並生成初始 Document 物件清單。

- 建立 Knowledge Graph 節點  
  - 以每份文件建立 Node（NodeType.DOCUMENT），並以 page_content 與 metadata 作為初始屬性。

- 節點特徵擷取（Extractors）  
  - HeadlinesExtractor：擷取可作為分段標題的片段以便切分。  
  - KeyphrasesExtractor：抽取關鍵詞。  
  - SummaryExtractor：摘要（供 SummaryEmbedding 使用）。  
  - EmbeddingExtractor：將指定屬性（如 page_content / summary）轉成向量。

- 關係建立（Relationship Builders）  
  - JaccardSimilarityBuilder / CosineSimilarityBuilder / SummaryCosineSimilarityBuilder / OverlapScoreBuilder：以不同策略建立 node 之間的 edge（例如 keyphrases_overlap）。

- Transforms 與 Pipeline  
  - 可將多個 Extractor 與 Relationship Builder 串聯或並行（Parallel），並用 apply_transforms 套用到 KnowledgeGraph。  
  - RunConfig 可控制並行度、timeout 與重試次數，避免大量 LLM 請求導致不穩定。

- Persist 與載入 Knowledge Graph  
  - 將處理完成的 KG 儲存為 JSON（範例：`week_5/knowledge_graph_single_hop.json`），並可重新載入供後續合成使用。

---

## 🔎 評估指標要點（簡要）

- Context Precision：衡量相關片段是否被排在較前位置（Precision@k 加權）。適用於有 reference 的情況。  
- Context Recall：檢查檢索到的上下文是否涵蓋回答所需的事實點。  
- Faithfulness：判斷回答中每個主張是否能被檢索到的上下文支持（0–1 分）。  
- Answer Relevance：透過從答案逆向生成問題並計算與原問題的語意相似度來判定「針對性」。  
- Answer Accuracy：在有 Ground Truth 時，兩位 LLM 評審（0 / 2 / 4 分制）給分並標準化為 [0,1]。  
- Noise Sensitivity：在 context 含錯誤或無關資訊時，衡量模型被誤導的程度；該指標具條件性，需與 Faithfulness/Recall 一起解讀。

---

## 🛠 快速上手（Quick Start）

- 預備：請先安裝 RAGAS、相關 embeddings 與您使用的 LLM 連線套件（Notebook 中使用 Ollama 範例）。  
- 主要步驟（摘要）：

1. 設定工作目錄
```python
import os
os.chdir("../")
```

2. 載入 LLM 與 Embeddings
```python
from ragas.embeddings import HuggingFaceEmbeddings
from src.ollama_connection import ragas_ollama

ragas_llm = ragas_ollama("gpt-oss:120b-cloud")
embeddings = HuggingFaceEmbeddings("BAAI/bge-m3")
```

3. 建 Dataset（Mock 範例）
```python
from ragas import Dataset
dataset = Dataset(name="test_dataset", backend="local/csv", root_dir="week_5")
# append / save...
```

4. 計算指標（Experiment 範例）
```python
from pydantic import BaseModel
from ragas import experiment
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

class ExperimentResult(BaseModel):
    faithfulness: float
    answer_relevancy: float

@experiment(ExperimentResult)
async def run_evaluation(row):
    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=embeddings)
    # 呼叫 .ascore(...) 並回傳 ExperimentResult
```

5. 建構與儲存 Knowledge Graph（single-hop 範例）
```python
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
kg = KnowledgeGraph()
# 將 docs 轉成 Node 並加入 kg.nodes
# apply_transforms(...) 處理後：
kg.save("week_5/knowledge_graph_single_hop.json")
```

---

## 🔧 技術棧與第三方套件

- Core：RAGAS (v0.4+)  
- LLM-as-a-Judge 範例：Ollama（gpt-oss:120b-cloud）  
- Embeddings：HuggingFace（BAAI/bge-m3）  
- Loader：LangChain DirectoryLoader  
- 繁簡轉換：OpenCC（opencc-python-reimplemented）  
- KG 組件：HeadlinesExtractor、HeadlineSplitter、KeyphrasesExtractor、SummaryExtractor、EmbeddingExtractor、OverlapScoreBuilder、Cosine/Jaccard Builders  
- 開發語言：Python

---

## ✅ 注意事項與建議

- Noise Sensitivity 的 0 分並不總是正向結果，需要與 Faithfulness 和 Context Recall 一併解讀。  
- 在大量呼叫 LLM 的流程中，務必使用 RunConfig 控制併發與重試，避免速率限制或非預期錯誤。  
- 若輸入語言或資料帶有繁簡差異，先行統一（如用 OpenCC）可降低抽取器輸出噪音。  
- 建議先在小型 subset 上測試整個 transforms pipeline，再放大到整個語料庫以節省成本與時間。

---

## 下一步（Week 6 預告）

- 使用已建立的 single-hop KG 作為基礎，實作 Multi-hop 測試集合成流程（跨節點路徑生成、題目難度分級）。  
- 建立自動化回歸測試（將評估流程納入 CI/CD），並比較不同 retriever / reranker / generator 組合的表現。

---

若想直接複現 Notebook 中的步驟，請開啟 `notebook.ipynb`，按照順序執行資料抓取、Node 建立、Extractors/Builders 設定，最後套用 apply_transforms 並儲存 KG（範例檔案：week_5/knowledge_graph_single_hop.json）。祝實驗順利！