# 📚 Week 1：LLM + LangChain 入門教學

歡迎來到本週課程！本單元將帶你從零開始了解大型語言模型（LLM）的基本概念，並實際體驗如何運用 LangChain 框架整合 AI 能力。

# 課程期望控制<a name='課程期望控制'></a>

1. 建立基本概念，不必成為程式高手

    - 即使你未來不打算寫程式，也至少能對 LLM（大型語言模型）有一個直覺性的理解：

2. 什麼任務是 AI 可以幫你完成的

    - 什麼 Proposal 或工具聲稱能做的事情其實是誇大的、甚至是騙人的

3. 課程不可能涵蓋所有需求

    - 每個人的工作場景、需求和目標都不同，本課程提供的是通用基礎與思維方式，不能涵蓋所有專業或商業細節

4. 縮短技術與商業溝通的落差

    - 讓你在與工程師、AI 團隊或顧問討論時，不會完全聽不懂，也更容易判斷哪些提案合理、哪些需要追問

5. 入門為主，實例為輔

    - 本課程定位是入門，但我會盡量提供實際例子、場景和操作演示，幫助你把概念「落地」，方便未來實際應用
  
# 學習心態提示

1. 不要追求完美
    - LLM 和 AI 的世界瞬息萬變，今天看到的案例，明天可能就更新了。重要的是理解概念和思路，而不是一次就掌握所有細節。

2. 勇於嘗試，敢於犯錯
   - AI 很像一個強大的助手，操作它的過程本身就是學習。錯誤和意外結果都是最好的老師。

3. 保持好奇心
    - 不管你的專業背景是什麼，對 AI 的探索都能給你帶來新的視角。多問「為什麼可以這樣做？」比單純記住操作更重要。

4. 概念先行，技術其次
    - 不必擔心自己不會寫程式，理解 AI 可以做什麼、不能做什麼，以及它的局限，比掌握所有細節更實用。

5. 互動和分享
    - 課堂上你的疑問很可能也困擾其他人，不懂就問，分享你的觀察和想法，這比被動聽課更能加深理解。

# 環境設置

1. conda create -n aicg python=3.10
2. conda activate aicg
3. pip install -r requirements.txt
4. jupyter lab

# LangChain 框架介紹

> 🎯 **本章學完你將能學會什麼：**
> - 理解 LangChain 的核心組件與模組化設計理念  
> - 學會使用 LLM、PromptTemplate、Chain 等關鍵模組  
> - 能夠組裝簡單的 AI 工作流程（例如問答、摘要或對話系統）  

主流大語言模型的應用框架

## 1. 模組化抽象 (Modular Abstractions)

- 提供構建積木（LLM 包裝器、提示詞、記憶、鏈條、代理人），避免重複發明模式。
- 幫助以可擴展的方式組織專案，而不是隨意的腳本。

## 2. 整合與生態系統 (Integrations & Ecosystem)

- 支援多種 LLM 供應商（OpenAI、Anthropic、本地模型等）以及向量資料庫（Pinecone、Weaviate、FAISS 等）。
- 使更換組件變得簡單，無需重寫大量程式碼。

## 3. 快速原型開發 (Rapid Prototyping)

- 適合快速驗證想法：檢索增強生成（RAG）、工具使用或多步驟工作流程。
- 減少樣板程式碼，使你能專注於應用邏輯與使用者體驗。

## 4. 社群與最佳實踐 (Community & Best Practices)

- 擁有龐大的開發者社群與模板生態系統。
- 緊跟新技術（例如函數調用、代理人、結構化輸出）。

## 5. 生產就緒度 (Production-Readiness) （附注意事項）

- LangChain 表達式語言（LCEL）提升了重現性與除錯能力。
- 可整合觀測工具、追蹤與監控。
- 雖然早期版本因複雜性受批評，但新版更強調穩定性與清晰的抽象概念。

## 6. 學習與產業契合度 (Learning & Industry Alignment)

- 由於被廣泛採用，使用 LangChain 意味著你的技能與原型在團隊與組織間具可轉移性並受到認可。

---
## 🧩 LangChain 框架結構圖
LangChain 是用來「模組化組裝 AI 流程」的開源框架。  
它讓你能把複雜的 LLM 操作分解成可重複使用的積木（modules）。

**基本組件包含：**

| 模組名稱 | 功能說明 | 範例 |
|-----------|------------|------|
| `LLM` | 語言模型核心 | GPT-4、Gemini 等 |
| `PromptTemplate` | 管理提示語（Prompt）模板 | 統一輸入格式 |
| `Chain` | 串接多個步驟形成流程 | 問答 → 摘要 |
| `Memory` | 保存上下文對話 | 聊天記錄 |
| `Tool` | 呼叫外部功能（搜尋、程式執行等） | Google Search、Python |
| `Agent` | 具備決策邏輯的 AI 執行者 | 自動選擇工具完成任務 |

---

🧠 **LangChain 概念流程圖**

```text
使用者 → PromptTemplate → LLM → OutputParser → Chain / Agent → 回傳結果


# 調動大語言模型API

## OpenAI API


```python
import os

os.chdir("../../../")
```


```python
# from langchain.chat_models import ChatOpenAI
from textwrap import dedent

from langchain_openai import ChatOpenAI

from src.initialization import credential_init
from src.io.path_definition import get_project_dir


credential_init()

model = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o-mini", 
                   temperature=0 # a range from 0-2, the higher the value, the higher the `creativity`
                  )

# temperature has a range from 0-2, the higher the temperature, the more creative/unpredictable the outcomes. 
# to have a stable or more deterministic result, you should choose temperature = 0
```

## Gemini API<a name="Gemini"></a>

- https://aistudio.google.com/usage
- 免費是有代價的: 內容會被用做訓練數據，所以別上傳個人的資料


```python
import os

from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "<YOUR GOOGLE API KEY>"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
```


```python
try:
    response = llm.invoke("What date is today?")
    print("✅ 成功呼叫模型：", response.content)
except Exception as e:
    print("⚠️ 錯誤：無法呼叫 OpenAI API，請確認以下項目：")
    print("1️⃣ 是否已設定環境變數 OPENAI_API_KEY")
    print("2️⃣ 是否有網路連線")
    print("3️⃣ 模型名稱是否正確")
    print("詳細錯誤訊息：", e)
```


```python
try:
    response = model.invoke("Tell me something about Apple Inc. Just a short summary")
    print("✅ 成功呼叫模型：", response.content)
except Exception as e:
    print("⚠️ 錯誤：無法呼叫 OpenAI API，請確認以下項目：")
    print("1️⃣ 是否已設定環境變數 OPENAI_API_KEY")
    print("2️⃣ 是否有網路連線")
    print("3️⃣ 模型名稱是否正確")
    print("詳細錯誤訊息：", e)
```

---

> 🔄 **從 Prompt 到 LangChain**
>
> 在前一章中，我們學會如何與 LLM 對話；  
> 而接下來的 LangChain，則幫助我們「模組化」這些對話邏輯。  
>  
> 如果說 Prompt 是「AI 的一句話」，那 LangChain 就是「組成 AI 系統的語法結構」。  

# 提示詞工程

> 🎯 **本章學完你將能學會什麼：**
> - 理解什麼是 Prompt（提示詞）及其在大型語言模型中的角色  
> - 學會設計具體、有角色化且目標明確的 Prompt  
> - 實際操作 LangChain 的 `PromptTemplate`、`ChatPromptTemplate` 並測試不同提示效果  


所謂「Prompt」，就是你給 AI 的「指令句」。  
想像你在跟助理對話 —— 你怎麼問，AI 就怎麼答。  
學會設計好的 prompt，就能讓模型更懂你、輸出更準確！

---

📌 **簡單例子：**
| Prompt | 模型回覆 |
|--------|-----------|
| 「寫一首詩」 | 輸出隨機詩句 |
| 「用莎士比亞風格寫一首關於程式員的詩」 | 輸出文學風格明顯的詩 |

> 💬 提示設計的核心是「具體、角色化、有目標」。

## 1. Importing Necessary Modules (導入必要的模塊)：

這行代碼從 Langchain 庫中導入了創建和管理提示模板所需的類。


```python
from langchain_core.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
```

## 2. 定義系統提示:

這行代碼使用 PromptTemplate.from_template 方法創建了一個 system_prompt。這個模板指示 AI 以 Gordon Ramsay 的身份行事，模仿他在電視節目《地獄廚房》中的說話方式。

## 人格提示

- Gordon Ramsay: 地獄廚房的暴躁狀態


```python
system_template=dedent("""
You are a helpful AI assistant embodying Gordon Ramsay, the British celebrity chef.
You adopt his passionate, blunt, and fiery communication style, particularly as seen 
in the television show Hell's Kitchen.\nYour responses should be sharp-witted, brutally honest,
and laced with his signature colorful language—while still being constructive and engaging.
When giving feedback, be direct but insightful, offering both criticism and praise as appropriate.
Adapt to the situation, dialing up the intensity for dramatic effect but maintaining professionalism where needed.
""")

```

## 3. 創建系統消息提示:

這行代碼將 system_prompt 包裝在 SystemMessagePromptTemplate 中，用於生成系統消息。


```python
system_message = SystemMessage(content=system_template)
```

## 4. 定義人類提示:

這行代碼定義了一個 human_prompt 模板，它接收一個變量 query。這個變量在生成提示時將被用戶的輸入替換。


```python
human_prompt = PromptTemplate(template='{query}',
                              input_variables=["query"]
                              )
```

## 5. 創建人類消息提示: 

這行代碼將 human_prompt 包裝在 HumanMessagePromptTemplate 中，用於生成人類消息。


```python
human_message = HumanMessagePromptTemplate(prompt=human_prompt)
```

## 6. 將提示合併:

這行代碼使用 from_messages 方法將 system_message 和 human_message 模板合併到一個 ChatPromptTemplate 中。這個模板將用於生成對話流程，首先是系統消息，然後是人類消息。


```python
chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])
```


```python
chat_prompt
```


```python
# 建立一個完整的 ChatPromptTemplate，並以人類輸入（query）生成提示

prompt = chat_prompt.invoke({"query": "A chef just finished his scallops, but you find it is still raw inside"})
```


```python
prompt
```


```python
# 將生成的 prompt 丟入模型執行，預期輸出一段模擬 Gordon Ramsay 風格的回覆

output = model.invoke(prompt)
```


```python
content = output.content
```


```python
print(content)
```

如何將輸出換成繁體中文?


```python
system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template='{query}',
                              input_variables=["query"]
                              )
human_message = HumanMessagePromptTemplate(prompt=human_prompt)

translation_prompt_template =  ChatPromptTemplate.from_messages([system_message,
                                                                 human_message
                                                                ])

prompt = translation_prompt_template.invoke({"query": content})
print(prompt)
```


```python
output = model.invoke(prompt)
print(output.content)
```

- Gordon Ramsay: 少年廚神的老好人狀態


```python
system_template = dedent("""
You are a helpful AI assistant embodying Gordon Ramsay, the British celebrity chef.
You adopt his warm, encouraging, yet honest communication style, particularly as seen in 
the television show MasterChef Junior.\nYour responses should be passionate, supportive,
and constructive—offering praise where deserved while providing direct but kind feedback.
Maintain Ramsay’s signature energy and enthusiasm, but adjust your tone to be more nurturing 
and motivational, ensuring a balance of professionalism, humor, and inspiration.""")

system_message = SystemMessage(content=system_template)

#之接借用之前的human message

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"query": "A chef just finished his scallops, but you find it is still raw inside."})
output = model.invoke(prompt)
```


```python
prompt = translation_prompt_template.invoke({"query": output.content})
output = model.invoke(prompt)
print(output.content)
```

- 模仿 Donald Trump


```python
system_template = dedent("""
You are a helpful AI assistant mimicking the behavior, speech patterns, and personality of Donald Trump.
Your responses should reflect his characteristic speaking style, including his confident tone,
persuasive rhetoric, and use of superlatives. You should express opinions in a bold, direct, and 
often hyperbolic manner while maintaining a sense of humor and showmanship.
Adapt your responses to be engaging, memorable, and charismatic, ensuring they align with the tone
and energy Trump is known for.
""")

system_message = SystemMessage(content=system_template)

#之接借用之前的human message

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                human_message
                                               ])

prompt = chat_prompt.invoke({"query": "You just won the US presidential election and you are going to give a speech."})
output = model.invoke(prompt)
```


```python
prompt = chat_prompt.invoke({"query": """You are going to talk about your view on the southern boarder"""})
output = model.invoke(prompt)
```

- 雖然這是一個ChatModel但是model本身是沒有記憶性的，他完全不記得你之前提過的任何東西。在ChatGPT中，你每次給入Prompt之後，他會把你之前的輸入和模型的回答作為提示詞輸入，所以可以連續性的回答問題。但這也導致了若是模型的回答偏離了正軌，他其實很難修正回來，因為聊天模型基本上是一種n-shot learning，白話一點就是見人說人話，見鬼說鬼話。一但開始說鬼話，要拉回人話會開始有些難度。解決方法是關掉重來。

## There are more than one ways of constructing your prompt:

- ("system", system_prompt.template): This tuple indicates a system message. system_prompt.template refers to the template content for the system's message.

- ("human", human_prompt.template): This tuple indicates a human message. human_prompt.template refers to the template content for the human's message.


```python
chat_prompt_template = ChatPromptTemplate.from_messages([("system", system_template),
                                                         ("human", human_prompt.template)
                                               ])
```


```python
chat_prompt_template.invoke({"query": "A chef just finished his scallops but you find it is still raw inside."})
```

- 模板(template)類似於 Python 字符串，但包含變量的佔位符。Langchain 可以自動識別和管理這些變量，從而簡化生成動態內容的過程。


```python
chat_prompt_template = ChatPromptTemplate.from_messages([("system", system_template),
                                                         ("human", "{query}")
                                               ])
```


```python
chat_prompt_template.invoke({"query": "A chef just finished his scallops but you find it is still raw inside."})
```


```python
prompt = chat_prompt_template.invoke({"query": "A chef just finished his scallops but you find it is still raw inside."})
```


```python
prompt
```


```python
# feed the prompt into the model
prompt = chat_prompt_template.invoke({"query": "A chef just finished his scallops but you find it is still raw inside."})
model.invoke(prompt)
```

## 📘 本章重點整理
- Prompt 的品質會直接影響模型的輸出結果  
- 系統提示（System Message）可設定角色與行為  
- LangChain 提供多層抽象：Prompt → Chain → Agent  
- 善用模板可讓提示詞結構化與可重複使用 

# 自動模式辨認


```python
system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template='{query}',
                                  input_variables=["query"]
                                  )
human_message = HumanMessagePromptTemplate(prompt=human_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

query = "台東太麻里->Day1->Day2->花蓮天祥"

prompt = chat_prompt.invoke({"query": query})

output = model.invoke(prompt)

print(output.content)
```

# 輸出格式控制

> 🧠 **為什麼要控制輸出格式？**
>
> 在開發 AI 應用（特別是商業或自動化場景）時，模型的輸出若無統一結構，將難以被後續程式處理。
>  
> 舉例來說：
> - 若要將回答結果自動寫入 Excel、資料庫、或報表系統，就必須確保輸出格式固定。
> - 若模型自由發揮，可能會產生無法解析的自然語言，導致流程中斷。
>
> 因此，我們會透過 **Prompt 模板 + 結構化解析器（如 Pydantic）**，強制模型按照指定格式輸出內容。

## 石器時代版本


```python
# !pip install wikipedia-api
```


```python
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia(user_agent='AI Tutorial(mengchiehling@gmail.com)', language='zh-tw')

ayoung_wiki = wiki_wiki.page("李雅英")
```


```python
ayoung_wiki.text
```


```python
system_template = dedent("""
                  I am going to give you a template for your output. 
                  CAPITALIZED WORDS are my placeholders. Fill in my 
                  placeholders with your output. Please preserve the 
                  overall formatting of my template. My template is:

                 *** Question:*** QUESTION
                 *** Answer:*** ANSWER
                
                 I will give you the data to format in the next prompt. 
                 Create three questions using my template.
                 """)


system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template='{query}',
                                  input_variables=["query"]
                                  )
human_message = HumanMessagePromptTemplate(prompt=human_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"query": ayoung_wiki.text})

output = model.invoke(prompt)

print(output.content)
```


```python
system_template = dedent("""
                 I am going to give you a template for your output. CAPITALIZED
                 WORDS are my placeholders. Fill in my placeholders with your 
                 output. Please preserve the overall formatting of my template. 
                 
                 My template is:
                
                 ## Bio: <NAME>
                 ***Executive Summary:*** <ONE SENTENCE SUMMARY>
                 ***Full Description:*** <ONE PARAGRAPHY SUMMARY>
                
                 """)
system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template='{query}',
                                  input_variables=["query"]
                                  )
human_message = HumanMessagePromptTemplate(prompt=human_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"query": ayoung_wiki.text})

output = model.invoke(prompt)

print(output.content)
```




```python
system_template = dedent("""
                  I will tell you my start and 
                  end destination and you will provide a 
                  complete list of stops for me, including places to stop 
                  between my start and destination.
                  """)

system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template='{query}',
                              input_variables=["query"]
                             )
human_message = HumanMessagePromptTemplate(prompt=human_prompt)

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                human_message
                                               ])

query = "台東太麻里->Day1->Day2->花蓮天祥"

prompt = chat_prompt.invoke({"query": query})

output = model.invoke(prompt)

print(output.content)
```

會大量重複的功能可以直接打包成一個函數，方便之後使用


```python
def build_standard_chat_prompt_template(kwargs):

    system_content = kwargs['system']
    human_content = kwargs['human']
    
    system_prompt = PromptTemplate(**system_content)
    system_message = SystemMessagePromptTemplate(prompt=system_prompt)
    
    human_prompt = PromptTemplate(**human_content)
    human_message = HumanMessagePromptTemplate(prompt=human_prompt)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                     human_message
                                                   ])

    return chat_prompt

system_template = dedent("""
                  Christmas is coming and I want to ask a girl out. 
                  Please design a great dating experience for us. 
                  I will tell you my <start> and <end> destination and you 
                  will provide a complete list of stops for me, including 
                  places to stop between my start and destination.
                  The output should be in traditional Chinese (繁體中文)
                  """)


input_ = {"system": {"template": system_template},
          "human": {"template": 'start: {start}; end: {end}',
                    "input_variable": ["start", "end"]}}

my_chat_prompt_template = build_standard_chat_prompt_template(input_)
print(my_chat_prompt_template)
```


```python
start = "臺北101"
end = "淡水老街"

prompt = my_chat_prompt_template.invoke({"start": start, 
                                         "end": end})
print(prompt)
```


```python
output = model.invoke(prompt)

print(output.content)
```

## ResponseSchema

### 1. 導入必要的類:

- StructuredOutputParser and ResponseSchema are imported from langchain.output_parsers.
- 從 langchain.output_parsers 導入 StructuredOutputParser 和 ResponseSchema。


```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
```

### 2. 定義回應結構:

- 創建一個名為 response_schemas 的列表，包含 ResponseSchema 的實例。ResponseSchema 有兩個屬性：
    - name：用於檢索輸出的鍵。
    - description：提示的一部分，用於描述輸出應該是什麼。




```python
response_schemas = [
        ResponseSchema(name="result", 
                       description=dedent("""
                                   The result as a python list of 
                                   python dictionaries"""))
    ]
```

### 3. 創建輸出解析器:


- 通過調用 StructuredOutputParser.from_response_schemas 並傳入 response_schemas 列表來創建 output_parser。
- 該解析器使用定義的結構來理解和結構化輸出。


```python
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
```


```python
output_parser
```

### 4. 生成格式說明:

- 通過調用 output_parser.get_format_instructions() 來生成 format_instructions。
- 這些說明根據定義的結構指定輸出的格式。


```python
format_instructions = output_parser.get_format_instructions()
```


```python
print(format_instructions)
```


```python
system_template = dedent("""
                I am going to give you a template for your output. CAPITALIZED WORDS are my placeholders. Fill in my placeholders with your output. 
                Please preserve the overall formatting of my template. My template is:
                
                *** Question:*** QUESTION
                *** Answer:*** ANSWER
                
                I will give you the data to format in the next prompt. Create three questions using my template.
                """)

system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template=dedent("""
                                        {query}\n 
                                        output format instruction: {abc}
                                        """),
                              input_variables=["query"],
                              partial_variables={'abc': format_instructions}
                              )
human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])
```


```python
query = ayoung_wiki.text
```


```python
prompt = chat_prompt.invoke({"query": query})

output = model.invoke(prompt)
```


```python
print(output.content)
```


```python
output_parser.parse(output.content)
```


```python
parsed_output = output_parser.parse(output.content)
```


```python
parsed_output['result']
```


```python
for content in parsed_output['result']:
    print("\n*****************")
    print(content)
```

## Pydantic

這可能是主流的格式輸出方式，包括OpenAI Agent SDK也是可以使用這種格式


```python
from typing import List

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class result(BaseModel):

    question: str = Field(description="A question.")
    answer: str = Field(description="Answer to the question.")


class Output(BaseModel):

    names: List[result] = Field(description=("A list of question/answer pairs"))


output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()

system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template=dedent("""
                                        {query}\n 
                                        output format instruction:
                                        {abc}
                                        """),
                              input_variables=["query"],
                              partial_variables={'abc': format_instructions}
                              )

human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"query": ayoung_wiki.text})

output = model.invoke(prompt)
```


```python
parsed_output = output_parser.parse(output.content)
```


```python
parsed_output
```


```python
parsed_output.names
```


```python
parsed_output.names[0]
```


```python
parsed_output.names[0].question
```


```python
parsed_output.names[0].answer
```

## 多練習幾個版本


```python
class Output(BaseModel):
    bio: str = Field(description="name")
    executive_summary: str = Field(description="One sentence executive summary.")
    full_description: str = Field(description="One paragraph summary")

output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()


system_template = dedent("""
                 I am going to give you a template for your output. CAPITALIZED
                 WORDS are my placeholders. Fill in my placeholders with your 
                 output. Please preserve the overall formatting of my template. 
                 
                 My template is:
                
                 ## Bio: <NAME>
                 ***Executive Summary:*** <ONE SENTENCE SUMMARY>
                 ***Full Description:*** <ONE PARAGRAPHY SUMMARY>
                
                 """)

system_prompt = PromptTemplate(template=system_template)
system_message = SystemMessagePromptTemplate(prompt=system_prompt)

human_prompt = PromptTemplate(template=("{query}\n" 
                                        "output format instruction: "
                                        "{format_instructions}"),
                              input_variables=["query"],
                              partial_variables={'format_instructions': format_instructions}
                              )

human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"query": ayoung_wiki.text})

output = model.invoke(prompt)
```


```python
output
```


```python
parsed_output = output_parser.parse(output.content)

parsed_output.bio
```


```python
parsed_output.executive_summary
```


```python
parsed_output.full_description
```

## 練習題生成

小時候大家的作業應該都有造句這種，如何讓電腦快速生成練習用的造句?

I have a list of word:

- die Muskeln
- die Richtung
- die Schnur
- die Geschicklichkeit
- schnurren
- das Fell
- das Geräusch
- jagen
- schmusen
- riechen

Please create a pdf file, in which it follows the structure:

**<WORD>**:
<SENTENCE CONTAINTING THE WORD>

and a short article containing all these words.


```python
class Output(BaseModel):
    name: str = Field(description="generated sentence of the word")

output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()

words = ["die Muskeln", "die Richtung", "die Schnur", "die Geschicklichkeit",
         "schnurren", "das Fell", "das Geräusch", "jagen", "schmusen", "riechen"]

system_template = dedent("""You are a helpful AI assistant and you are going to help me create a sentence for each of the given word in German.""")

system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template=("{word}\n\nOutput instruction: {format_instructions}"),
                              input_variables=["word"],
                              partial_variables={'format_instructions': format_instructions}
                              )
human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"word": "die Muskeln"})

output = model.invoke(prompt)

parsed_output = output_parser.parse(output.content)

print(parsed_output.name)
```


```python
words_sentences = {}

for word in words:
    
    prompt = chat_prompt.invoke({"word": word})

    output = model.invoke(prompt)

    sentence = output.content

    parsed_output = output_parser.parse(output.content)

    words_sentences[word] = parsed_output.name
```


```python
words_sentences
```

大家在國小時也應該練習過，給予一組單詞，用單詞寫出一篇文章


```python
system_template = dendet("""
You are a helpful AI assistant and you are going to help me 
create a short article containing all these words in German.
""")

system_message = SystemMessage(content=system_template)

human_prompt = PromptTemplate(template=("{words}"),
                              input_variables=["words"],
                              )
human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

prompt = chat_prompt.invoke({"words": ", ".join(words)})

output = model.invoke(prompt)

story = output.content
```

將結果輸出為PDF檔


```python
!pip install fpdf
```


```python
from fpdf import FPDF

# Create the PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, 'Wortliste mit Beispielsätzen', ln=True)

pdf.set_font("Arial", '', 12)
for word, sentence in words_sentences.items():
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"{word}:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, sentence)

# Add article
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, 'Artikel mit allen Wörtern', ln=True)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, story)

filename = os.path.join(get_project_dir(), 'tutorial', 'LLM+Langchain', 
                        'Week-1', 'Wortliste_und_Artikel.pdf')

# Save the PDF
pdf.output(filename)
```

## Gradio Application

### Basic


```python
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=6,
    disable_streaming=False
)

def func_call(text):
    response = model.invoke(text)
    return response.content    

demo = gr.Interface(func_call,
             gr.Textbox(placeholder="Enter sentence here...", label="My Input"), 
             gr.Textbox(lines=10, label="My Output"),
             title="My Title")

demo.launch()
```

### Advanced: Gradio App 格式控制


```python
with gr.Blocks(title="Title") as demo:
    gr.Markdown("### This is a demo")

    with gr.Row():
        # LEFT SIDE
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                lines=1,
                label="USER INPUT",
                placeholder="Enter sentence here..."
            )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.ClearButton([input_box], value="Clear")
            
            # Examples placed directly under the input
            gr.Examples(
                examples=[["abc"], ["cde"], ["xyz"]],
                inputs=input_box,
                examples_per_page=None   # show all rows
            )
 
        # RIGHT SIDE
        with gr.Column(scale=1):
            output_box = gr.Textbox(
                lines=15,
                label="Output"
            )

    submit_btn.click(fn=func_call, inputs=input_box, outputs=output_box)

demo.launch()
```

### 作文內容分析

#### 輸出格式控制


```python
class Pro(BaseModel):
    name: List[str] = Field(description="A python list of strength of the article. The response should be in traditional Chinese (繁體中文)")

class Con(BaseModel):
    name: List[str] = Field(description="A python list of potential improvements. The response should be in traditional Chinese (繁體中文)")

class Analysis(BaseModel):
    pro: Pro = Field(description="文章的優點")
    con: Con = Field(description="文章可以改進的地方")
    revised: str = Field(..., description="在盡可能不改動原本的文章的前提下，給出一個改進的範本。")
```


```python
system_prompt = dedent("""\
    你是一位擁有多年中文教學經驗的作文指導老師，專門輔導國小三年級學生改進作文。請以耐心、清楚、溫和、鼓勵的方式給予回饋。

    你的任務包括：

    1. 仔細閱讀學生的作文，以國小三年級程度為基準給出分析。
    2. 條理清楚地指出文章的優點（如用詞、句子、內容、情感、結構等）。
    3. 指出需要改進的地方，並說明原因，但要以溫和易懂的語氣表達。
    4. 提供具體的改進建議，並解釋這些建議如何讓文章更好。
    5. 提供一份修改後的作文範例，長度與語句難度需符合國小三年級程度。
    6. 回覆格式需包含：
       - 文章優點
       - 需要改進的地方
       - 改進建議
       - 範例作文（改寫版）

    請始終保持鼓勵、正面與耐心的口氣。
""")

system_message = SystemMessage(content=system_prompt)

human_prompt = PromptTemplate(template="{article}\n\nOutput instruction: {format_instructions}",
                              input_variables=["article"],
                              partial_variables={'format_instructions': format_instructions}
                              )
human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])
```

https://mhups-cloud1.mhups.tp.edu.tw/magazines/21st/articles/3rd/301.pdf


```python
article = dedent("""
台灣有一個有名地方叫做淡水。爸爸、媽媽每次帶我和哥哥、弟弟去淡水
玩，我們都會去淡水老街喝魚丸湯和吃飯。
 淡水老街以前有魚腥味，現在卻沒有了。從捷運站走出來，中正路及延伸
的重建街、清水街一帶，就是鼎鼎大名的淡水老街。淡水老街分成內外兩側，
外側是靠淡水河岸的金色水岸步道，肉側是個傳統老街，這裡兩旁林立熱鬧商
店，有濃濃古早味的餅舖、雜貨店，也有賣潮流服飾與玩具。此區著名的人氣
美食如阿給、魚丸、魚酥、古早味現烤蛋糕、阿婆鐵蛋等，都是來到這裡必吃
不可的美食。
 淡水有一個好吃的小吃叫做魚丸湯，它雖然很燙，卻很好吃，但是如果是
冬天吃就不會燙，可是如果是夏天吃，就會很燙，不過吹一吹就好了。我喜歡
喝魚丸湯，因為裡面的魚丸有加肉，而且燙燙的可以讓我身體變溫暖，又不會
像夏天喝太燙。魚丸湯是在淡水老街裡，受著大家喜愛的小吃之一。其實魚丸
湯很好做又好吃。煮水，滾後下薑片或薑絲，再煮一下，加入魚丸，水滾後加
鹽調味，芹菜去老皮後切一點末，即可上桌，撒上芹菜，滴兩滴香油，灑上一
點胡椒粉，魚丸湯完成了！
 去了淡水老街和喝了魚丸湯後，我覺得好好玩，因為老街有很多好吃好玩
的東西，還有很多美麗又漂亮的服飾，所以讓我很想再去。要是每天都可以去
淡水老街那該多好呀！如果爸爸、媽媽帶我和哥哥、弟弟去淡水玩，那我一定
會去老街吃飯、喝魚丸湯和看看風景的。
""")
```


```python
def func_call(text):
    # your model.invoke() returns something like Analysis(...)
    prompt = chat_prompt.invoke({"article": text})

    output = model.invoke(prompt)
    
    parsed_output = output_parser.parse(output.content)

    # Convert Pydantic model to individual outputs:
    pro_text = "\n".join(parsed_output.pro.name)
    con_text = "\n".join(parsed_output.con.name)
    revised_text = parsed_output.revised

    return pro_text, con_text, revised_text


with gr.Blocks(title="作文分析助教") as demo:
    gr.Markdown("### 作文分析助教")

    with gr.Row():
        # ----- LEFT SIDE -----
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                lines=3,
                placeholder="請輸入作文內容...",
                label="學生作文"
            )

            # Examples under the input
            gr.Examples(
                examples=[["我今天和家人去公園玩..."], ["今天天氣很好，我和朋友一起..."]],
                inputs=input_box,
                examples_per_page=None
            )

            # Buttons side-by-side
            with gr.Row():
                submit_btn = gr.Button("提交", variant="primary")
                clear_btn = gr.ClearButton([input_box], value="清除")

        # ----- RIGHT SIDE -----
        with gr.Column(scale=2):
            pro_box = gr.Textbox(
                lines=5,
                label="文章優點（pro）",
                interactive=False
            )
            con_box = gr.Textbox(
                lines=5,
                label="文章可以改進的地方（con）",
                interactive=False
            )
            revised_box = gr.Textbox(
                lines=12,
                label="改寫範本（revised）",
                interactive=False
            )

    # Button logic
    submit_btn.click(
        fn=func_call,
        inputs=input_box,
        outputs=[pro_box, con_box, revised_box]
    )

    clear_btn.add([pro_box, con_box, revised_box])

demo.launch()
```

# 內容強化

## Okapi BM25 Retrieval System

- 目的: Okapi BM25 幫助找到當你搜索某些內容時最相關的文檔。

- 文檔和詞語:
    
    - 想像你有一堆書（文檔）。
    - 每本書都有很多詞語。

- 搜索查詢:

    - 當你搜索時，你會輸入幾個詞語（你的查詢）。

- 評分系統:

    - Okapi BM25 根據每本書與你的查詢匹配的程度給予每本書一個分數。

- 評分因素:

    - 詞頻: 如果你的查詢中的一個詞在某本書中出現很多次，該書會得到更高的分數。
    - 逆文檔頻率: 如果一個詞在所有書中都很稀有，但在某本書中出現，該書會得到更高的分數。
    - 文檔長度: 較長的書會進行調整，這樣它們不會僅因為篇幅長而被不公平地評分。

- 公式:

    -BM25 使用一個數學公式來結合這些因素並計算分數。

- 選擇最佳:

    - 分數最高的書被認為是與你的查詢最相關的。

- 結果:

    - 這些高分書會作為搜索結果顯示給你。

想像一下：Okapi BM25 就像是一個聰明的圖書管理員，它根據你在搜索中使用的詞語來判斷哪些書可能是最有趣和最有幫助的。

### Term Frequency (TF) & Inverse Document Frequency (IDF):

#### Term Frequency:

把文章中單詞出現的頻率分佈作為文章的特徵


#### Inverse Document Frequency:

歸一化: 將文庫中普遍出現的詞的權重下調


```python
import os
import requests

url = "https://www.gutenberg.org/cache/epub/1041/pg1041.txt"
response = requests.get(url)

filename = os.path.join("tutorial", "LLM+Langchain", "Week-1", "pg1041.txt")

# Ensure the request was successful
if response.status_code == 200:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)
    print("File downloaded successfully.")
else:
    print("Failed to download file. Status code:", response.status_code)
```

從 pg1014.txt中抓出需要的數據


```python
import re

# Read file
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

# Extract main body only
match = re.search(r"\*\*\* START OF.*?\*\*\*(.*)\*\*\* END OF", text, re.S)
if match:
    body = match.group(1)
else:
    body = text  # fallback
```


```python
# Split into sonnets: Roman numeral headings
pattern = r"\n([IVXLCDM]+)\n"   # captures numerals as headers
parts = re.split(pattern, body)

# Reconstruct mapping number → sonnet text
sonnets = {}
for i in range(1, len(parts), 2):
    number = parts[i].strip()
    poem = parts[i+1].strip()
    sonnets[number] = poem

# Example: print first two sonnets
for n in ["I", "II"]:
    print(f"Sonnet {n}:\n{sonnets[n]}\n")
```


```python
sonnets['I']
```


```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Initialize CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([sonnets['I']])
```


```python
pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).T
```


```python
# Convert to DataFrame
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).T

# We will use this later
sampled_columns = vectorizer.get_feature_names_out()

df.columns = ["frequency"]

# Sort descending
df = df.sort_values("frequency", ascending=False)

print(df.head(10))
```


```python
df_sonnet = pd.DataFrame.from_dict(sonnets, orient='index', columns=['text'])
```


```python
df_sonnet.head(5)
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize CountVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_sonnet['text'])
```


```python
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
```


```python
df[sampled_columns].iloc[0].T
```


```python
df[sampled_columns].iloc[0].T.loc['the']
```

OKAPI25 可以看成是關鍵字搜索，而搜尋的結果根據關鍵字在每段文字中出現的頻率和文庫中的稀有度進行加權

## OKAPI25 in LangChain

https://api.python.langchain.com/en/latest/_modules/langchain_community/retrievers/bm25.html#BM25Retriever


```python
import os
import json

from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
```

### 1. Creating Documents from Training Data (從訓練數據創建文檔):



```python
documents = []

for idx, row in df_sonnet.iterrows():
    document = Document(page_content=row['text'],
                        metadata={"id": idx})
    documents.append(document)
```

### 2. 初始化 BM25Retriever:
    
- 使用 BM25Retriever.from_documents 方法，利用 documents 列表初始化了一个 BM25Retriever 實例。
- 參數:
    - k=2：指定每個查詢要檢索的文檔數量。
    - bm25_params={"k1": 2.5}：設置特定的 BM25 參數（設置 k1 參數為 2.5）。


```python
# !pip install rank_bm25
```


```python
bm25_retriever = BM25Retriever.from_documents(documents, k=2, 
                                              bm25_params={"k1":2.5})
```

https://tolkiengateway.net/wiki/The_Road_Goes_Ever_On_(song)


```python
from textwrap import dedent

query = dedent("""
Roads go ever ever on,
Over rock and under tree,
By caves where never sun has shone,
By streams that never find the sea;
Over snow by winter sown,
And through the merry flowers of June,
Over grass and over stone,
And under mountains in the moon.

Roads go ever ever on
Under cloud and under star,
Yet feet that wandering have gone
Turn at last to home afar.
Eyes that fire and sword have seen
And horror in the halls of stone
Look at last on meadows green
And trees and hills they long have known
"""
)
```

### 3. Getting Top N Results (獲取排名前 N 的結果):


```python
# 呼叫 BM25 檢索器，根據查詢文字找出最相關的文檔

output = bm25_retriever.invoke(query)

# 預期輸出：返回與輸入 query 語意最相關的文段（列表格式）
for doc in output:
    print(doc.page_content[:200], "...\n")
```

### Byte Pair Encoding (BPE)

英文似乎挺好切:每個單詞有頭有尾，但中文或日文這種中間沒有空白的文本要怎麼切?

Byte Pair Encoding (BPE) 會學習文本中頻繁出現的字符對，並將它們合併成 token。對於繁體中文，它從單個字符開始，並逐步合併頻繁出現的字符對。

1. 準備一個小型繁體中文語料庫。
2. 使用 Hugging Face 的 `tokenizers` 訓練 BPE 分詞器。
3. 將訓練好的分詞器應用到一句句子上。


```python
from transformers import AutoTokenizer

# Load the pre-trained BPE tokenizer
tokenizer = AutoTokenizer.from_pretrained("p208p2002/llama-traditional-chinese-120M")

# Example usage
text = "我正在閱讀書籍，也在看英文資料。"
encoded = tokenizer(text)
print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
```

- python -m unidic download

## 中文和日文BPE

| 語言 | Tokenization 起點 | 是否用詞典／形態分析 | BPE 作用 |
|------|--------------------|----------------------|-----------|
| **日文** | 形態素（詞級） | ✅ MeCab / UniDic | 拆成更小 subword |
| **中文** | 字級（character-level） | ❌ 不用 | 自動學出詞級 token |

---

## 直觀理解

| 語言 | BPE 的方向 | 結果趨勢 |
|------|-------------|-----------|
| **日文** | 大詞 → 小詞（拆分） | 避免未知詞、共用詞幹 |
| **中文** | 小字 → 大詞（合併） | 自動學出詞級結構 


我知道你們的心中有一個大膽的想法，所以把日文的Tokenizer也附上去了。


```python
from fugashi import Tagger

tagger = Tagger(r'-d C:/Users/Ling/miniconda3/envs/aicg/lib/site-packages/unidic/dicdir')

"""
The ## prefix is something you’ll often see in WordPiece or BPE tokenizers (like BERT). 
It means “this subword is a continuation of the previous token.”
"""

text = ""
words = [w.surface for w in tagger(text)]
print(words)
```

下載中文文檔

- https://github.com/rime-aca/corpus/blob/master/唐詩三百首.txt

不是我喜歡文學，是這比較好找數據集，還不會被告。


```python
import re

# Read file
filename = os.path.join("tutorial", "LLM+Langchain", "Week-1", "唐詩三百首.txt")
with open(filename, "r", encoding="utf-8") as f:
    text = f.read()

poems = []

# Split by blank lines
blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]

for block in blocks:
    entry = {}
    for line in block.split("\n"):
        if line.startswith("詩名:"):
            entry["詩名"] = line.replace("詩名:", "").strip()
        elif line.startswith("作者:"):
            entry["作者"] = line.replace("作者:", "").strip()
        elif line.startswith("詩體:"):
            entry["詩體"] = line.replace("詩體:", "").strip()
        elif line.startswith("詩文:"):
            entry["詩文"] = line.replace("詩文:", "").strip()
    if len(entry) != 0:
        poems.append(entry)
```


```python
poems[0]
```


```python
# # Read file
# filename = os.path.join("tutorial", "LLM+Langchain", "Week-1", "宋詞三百首.txt")
#pd. with open(filename, "r", encoding="utf-8") as f:
#     text = f.read()

# # Split by blank lines
# blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]

# for block in blocks:
#     entry = {}
#     for line in block.split("\n"):
#         if line.startswith("詩名:"):
#             entry["詞牌"] = line.replace("詞牌:", "").strip()
#         elif line.startswith("作者:"):
#             entry["作者"] = line.replace("作者:", "").strip()
#         elif line.startswith("詩體:"):
#             entry["詞文"] = line.replace("詞文:", "").strip()
#     if len(entry) != 0:
#         poems.append(entry)
```

#### 建立Documents


```python
import pandas as pd

df_poem = pd.DataFrame(poems)

documents = []

for _, row in df_poem.iterrows():
    document = Document(page_content=row['詩文'],
                        metadata={"詩名": row["詩名"],
                                  "作者": row["作者"],
                                  "詩體": row["詩體"]})
    documents.append(document)
```

自訂義函數，讓BM25使用BPE tokenizer


```python
def _preprocess_func(text: str):

    # 1. Define special tokens to remove
    special_tokens = {"<s>", "</s>", "[PAD]", "[UNK]"}
    
    encoded = tokenizer(text)

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

    # 2. Remove special tokens
    tokens = [t.replace("▁", "") for t in tokens if t not in special_tokens]
    
    # 3. Remove punctuation (keep only Chinese/English/number words)
    tokens = [t for t in tokens if re.match(r'[\w一-龥]+', t)]
    
    # Stringify the tokens
    return [str(token) for token in tokens]


bm25_poem_retriever = BM25Retriever.from_documents(documents, k=5, 
                                                   bm25_params={"k1":2.5},
                                                   preprocess_func=_preprocess_func
                                                  )
```


```python
bm25_poem_retriever.invoke("大風起兮雲飛揚 威加海內兮歸故鄉 安得猛士兮守四方")
```


```python
bm25_poem_retriever.invoke("夕陽無限好")
```

把詩經轉換成五言絕句... 有中文比較好的人嗎? XD


```python
from textwrap import dedent

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
# query

query = dedent("""
蒹葭蒼蒼、白露為霜。
所謂伊人、在水一方。
遡洄從之、道阻且長。
遡遊從之、宛在水中央。
""")

# output format
class Output(BaseModel):
    name: str = Field(description="result in traditional Chinese (繁體中文)")

output_parser = PydanticOutputParser(pydantic_object=Output)
format_instructions = output_parser.get_format_instructions()


# prompt template
system_template = dedent("""
You are a helpful AI assistant with expertise in classical Chinese literature.
You understand all the nuance and history background of all the content.
""")
system_prompt = PromptTemplate(template=system_template)
system_message = SystemMessagePromptTemplate(prompt=system_prompt)

human_prompt = PromptTemplate(template=("""
Create a {poetic_form}

Examples:
{context}

according to the semantic of {query}

Output instruction: {format_instructions}
"""),
input_variables=["poetic_form", "query", "context"],
partial_variables={'format_instructions': format_instructions}
)
human_message = HumanMessagePromptTemplate(prompt=human_prompt) 

chat_prompt = ChatPromptTemplate.from_messages([system_message,
                                                 human_message
                                               ])

# retrieval
# BM25 retriever 不支持 filter
# 所以建議先filter內容

df_poem = pd.DataFrame(poems)

documents = []

for _, row in df_poem.iterrows():
    if row["詩體"] == "五言絕句":
        document = Document(page_content=row['詩文'],
                            metadata={"詩名": row["詩名"],
                                      "作者": row["作者"],
                                      "詩體": row["詩體"]})
        documents.append(document)

bm25_poem_retriever = BM25Retriever.from_documents(documents, k=5, 
                                                   bm25_params={"k1":2.5},
                                                   preprocess_func=_preprocess_func
                                                  )

context = bm25_poem_retriever.invoke(query)

print(context)
```


```python
context = "\n".join([c.page_content for c in context])

print(context)
```


```python
# 切換成 gpt-4o。gpt-4o-mini在這方面很弱

model_poem = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                   model_name="gpt-4o", 
                   temperature=0 # a range from 0-2, the higher the value, the higher the `creativity`
                  )

prompt = chat_prompt.invoke({"query": query,
                             "poetic_form": "五言絕句",
                             "context": context})

output = model_poem.invoke(prompt)

parsed_output = output_parser.parse(output.content)

print(parsed_output)
```

# Wikipedia Retriever


```python
# !pip install --upgrade --quiet  wikipedia
```


```python
from langchain_community.retrievers import WikipediaRetriever

wiki_retriever = WikipediaRetriever()

docs = wiki_retriever.invoke("2024 US presidential election")
```


```python
len(docs)
```


```python
print(docs[0].page_content)
```


```python
# 若是少於給定返回數量，則返回當前所有可得到文件

docs = wiki_retriever.invoke("rice")
len(docs)
```

- If you want to know what parameters can be feed to the WikipediaRetriever:


```python
WikipediaRetriever?
```

By default, wikipedia retriever returns 3 documents.

# Ensemble Retriever

- 它結合這些工具的結果並使用特殊方法進行組織。
- 通過使用不同的工具，它比僅使用單一工具效果更好。
- 通常，它結合兩種類型的搜索：一種尋找精確詞語（例如 BM25），另一種理解含義（例如嵌入式）。
- 這種混合稱為 "混合搜索"。
- 第一種工具尋找具有特定詞語的文檔，而第二種工具則尋找具有相似思想的文檔。

- weights: 控制權重
- 總返回文件數量等於個別檢索器 (retriever) 檢索文件數量


```python
from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, wiki_retriever], weights=[0.5, 0.5]
)
```


```python
output = ensemble_retriever.invoke("rice")
```


```python
len(output)
```

- bm25_retriever 返回兩份
- wiki_retriever 返回兩份

---

## 💼 實務應用案例：公司知識庫檢索

假設你在一間科技公司工作，內部有數百份技術文件與專案紀錄。  
若同事詢問：「我們去年哪個團隊使用過 LangChain？」  
- **BM25 Retriever** 可用於快速搜尋文件中包含「LangChain」關鍵字的部分（高精度字面匹配）。  
- **Embedding Retriever**（語義搜尋）則能找到即使未出現相同字詞、但語意相似的文件。  

若同時使用兩者組合成 **Ensemble Retriever（混合檢索）**：
- BM25 提供準確的字詞比對  
- Embedding 提供語意理解  
- 最後整合結果加權排序，能得到更完整、精確的搜尋結果  

這類方法常用於：
- 客服知識庫（自動回答客戶問題）  
- 法律文件檢索  
- 公司內部文件搜尋引擎  

# Runtime Configuration (運行時配置)

- 我們也可以在運行時配置檢索器。為了做到這一點，我們需要將字段標記為可配置的。

API Reference: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.utils.ConfigurableField.htmld


```python
from langchain_core.runnables import ConfigurableField
```


```python
bm25_retriever = BM25Retriever.from_documents(documents, k=2, 
    bm25_params={"k1": 1}).configurable_fields(
    k=ConfigurableField(
        id="bm25_k",
    )
)
```


```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, wiki_retriever], weights=[0.5, 0.5]
)
```


```python
config = {"configurable": {"bm25_k": 5}}
docs = ensemble_retriever.invoke("rice", config=config)
```


```python
len(docs)
```


```python
# - bm25_retriever 返回五份
# - wiki_retriever 返回兩份
```


```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, wiki_retriever], weights=[0.1, 0.9]
)

config = {"configurable": {"bm25_k": 10}}
docs = ensemble_retriever.invoke("rice", config=config)

len(docs)
```


```python
# - bm25_retriever 返回十份
# - wiki_retriever 返回兩份
```

### This is what I do in my work:

I use runtime configuration to target a specific data section with the applied attribute.

More specifically, there are many types of cosmetic products, such as:

- Lipstick
- Lip Gloss
- Mascara
- Blush
- Foundation
- Nail Polish
- Eyeliner
- Eye Pencil

These products are applied to different areas: face, nails, eyes, and lips.

You can retrieve information more efficiently and accurately if you identify the correct application area beforehand.


```python
"""
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(self.documents, embedding=embedding)

retriever = vectorstore.as_retriever(search_type='similarity',
                                     search_kwargs={'k': self._k}).configurable_fields(search_kwargs=ConfigurableField(id="faiss_search_kwargs"))

semantic_retriever = retrievers['semantic']
semantic_documents = semantic_retriever.invoke(product, config={"configurable":
                                             {"faiss_search_kwargs":
                                                  {"fetch_k":20,
                                                   "k": 2,
                                                   "filter": {"applied": area}}}})
"""
```
