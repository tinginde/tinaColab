import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("genmini_api_key"))

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
  },
]
# 測試可否調動
# model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
#                               generation_config=generation_config,
#                               safety_settings=safety_settings)

# # 建立conversation
# convo = model.start_chat(history=[
#   {
#     "role": "user",
#     "parts": ["\"中正區家裡訊號極差，手機經常無服務。\"這是正面還是負面評論? 只要回答就好。"]
#   },
#   {
#     "role": "model",
#     "parts": ["負面評論"]
#   },
# ])

# YOUR_USER_INPUT =input('請輸入你想查詢的文句\n')
# convo.send_message(YOUR_USER_INPUT)
# print(convo.last.text)
def get_km_result(prompt, model="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["\"\"\"使用以下上下文來回答問題，如果您不知道答案，只需說\"\"資訊不足，問題再描述詳細一點\"\"，不要試圖編造答案。{context}請用中文繁體回答，摘要成150個中文字以內問題: {question}回答:\"\"\".strip()"]
  },
  {
    "role": "model",
    "parts": ["好的，請提供上下文和問題，我會盡力用中文繁體摘要並回答您的問題。"]
  },
])
    convo.send_message(prompt)
    return convo.last.text


# 基本上就是透過不同user跟model舉例，建立不同功能的LLM
# 摘要評論LLM
def get_genmini_summary(prompt, model="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ['''你會收到幾篇相關的負評，請你將這些負評做清楚的摘要，你只要輸出摘要，不用輸出其他說明文字。例如:\\\"'所以才說 沒人要蹭XX訊號有多棒 可惜是兩小倒了 然後連原本的網速也沒了', 
              '\n有一百理由推拖是整合陣痛期事實就是要你轉到遠傳去年剛申辦亞太是這種速度，合併後變成這種連網頁都開不了的神速遠傳跟XXX比起來，真的沒好到哪……\n', 
              '\n訊號強度增加，網速變慢，去看一下OO的測試影片，你就知道XX對KK用戶做了啥陰招。OO的總經理之前在 AT&T 就是專門做限速這個業務，所以OO請她回來做啥呢？', 
              'JJ 138 12MB 平常都有7-8MB 高也有12MB1/15基地台一合併後, 網速狂掉, 只剩 1MB - 6MB左右B28 訊號強度RSRP上升 從 -107dBm 到-95dBm但RSRQ下降到 -14dB讓我回想起三年前用XX的時候也是這樣龜速後來才慢慢改善, 沒想到現在網速又吐回去了 大家各自保重呀'\\\"''']
  },
  {
    "role": "model",
    "parts": ['''網速下降：用戶普遍報告合併後網速顯著下降，有人提到原本AA的網速在合併後大幅降低，甚至有情況出現連基本網頁都難以打開。

訊號質量變化：雖然有提到訊號強度（RSRP）有所增加，但訊號質量（RSRQ）卻出現下降，這對網速和整體網絡體驗產生了負面影響。

用戶不滿與質疑：

有用戶表達對TT訊號質量的不滿，並認為合併後的服務不如預期。
有評論指出XX可能採取了限速措施，並質疑XX引入前 AT&T 經理人專門負責限速業務的決策。
比較其他網絡提供商：一些用戶將XX與其他提供商（如AA）進行比較，認為XX的服務沒有顯著優勢。

期待改善：有用戶回憶起三年前使用XX時也經歷過類似的網速問題，但後來有所改善。目前對於合併後的網速下降表示失望，希望未來能有改進。

總體來看，用戶對於XX與II合併後的網絡服務質量表達了明顯的不滿，特別是在網速下降和訊號質量問題上。此外，對於AA的經營決策和服務質量也提出了質疑和批評。用戶期待合併後的服務能有所改善，以提供更好的網絡體驗。''']
  },
])
    convo.send_message(prompt)
    return convo.last.text
    

def get_genmini_negclassifier(prompt, model="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["你會收到使用者傳給你一則訊息，請你判斷是正面還是負面評價，你只要回答答案就好，例如:\\\"中正區家裡訊號極差，手機經常無服務。\\\""]
  },
  {
    "role": "model",
    "parts": ["負面評論"]
  },
])
    convo.send_message(prompt)
    print(convo.last.text)
    
    
# 回應負評LLM
def get_genmini_negresponse(prompt, model="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)
    convo = model.start_chat(history=[
  {
    "role": "user",
    "parts": ["你是處理輿情專家，會收到使用者的負面評論，請你根據客戶的抱怨回應負面評論，盡量用親切的語氣，你只要答覆就好，不能透漏自己身分。例如:\\\"中正區家裡訊號極差，手機經常無服務。\\\""]
  },
  {
    "role": "model",
    "parts": ['''親愛的顧客，

您好！非常感謝您抽出寶貴的時間向我們反映中正區手機訊號的問題。我們十分遺憾聽聞您在家中遇到連線不穩定的困擾，深知這必然給您的日常生活帶來了不便。

我們高度重視您的體驗，並且已經將您的情況報告給技術團隊，請求他們儘快查明原因並著手解決。同時，如果您能提供更具體的位置資訊，我們將能更精確地定位問題，進而加速改善過程。

在此期間，建議您嘗試開啟手機的「飛行模式」再關閉，或是重啟您的手機，有時這能暫時改善訊號接收情況。如果您有任何疑問，或者需要進一步的協助，請隨時聯繫我們的客服團隊。我們承諾將盡一切努力，確保您能得到滿意的服務體驗。

再次為您遇到的不便致以最深的歉意，並感謝您的理解與支持。

誠摯地，
[您的名字]''']
  },
  ])
    convo.send_message(prompt)
    return convo.last.text

# system_instruction = '''你是分類使用者詢問的分類人員，不用回答問題，你要將問題解析為意圖(intent)、實體識別(Entity)、答案請用繁體中文回答。
答案意圖: 判斷使用者提出的問題，是想要做什麼事。以下例子:\\n--例子開始\\n會下雨嗎、我要查天氣、降雨機率多少 → 查詢天氣意圖\\n我要訂購機票? → 訂購飛機票意圖\\n我要查詢張三的訂單資料? → 查詢訂單資料\\n請問張三的出缺勤記錄 → 查詢出缺勤記錄意圖\\n--例子結束\\n
實體識別: 判斷使用者提出的問題, 實體必須由使用者提出的問題解析, 不要造假以下例子:\\n--例子開始\\n實體識別: 姓名[張三]在[]前面表示實體,[]裏面表示這個實體的實際值, 實際值來自於問題, 不要造假\\n--例子結束\\n
\\n**範例問題**: 我要查詢張三的訂單資料\\n--格式開始\\n意圖: 分析您對被問到的問題，問題的意圖\\n實體識別: 實體1[實體的實際值1],實體2[實體的實際值2]..., 可以重複多個, 不同實體請用\\\",\\\"分隔, 同一實體, 有多個實體的實際值, 也請請用\\\",\\\"分隔, --格式結束\\n ## 解析：\\n\\n**問題: {question}**\\n\\n**意圖:**'''


def get_query_intent(prompt, model="gemini-1.5-pro-latest"):
    model = genai.GenerativeModel(model_name=model,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

    convo = model.start_chat(history=[
      {
        "role": "user",
        "parts": ["我想找112年台灣對中國的國際貿易狀況"]
      },
      {
        "role": "model",
        "parts": ["## 範例解析：\n\n**問題:** 我想找112年台灣對中國的國際貿易狀況\n\n**意圖:** 查詢台灣與中國的國際貿易狀況\n**實體識別:** 年份[112], 地區[台灣], 地區[中國] \n "]
      },
    ])
    convo.send_message(prompt)
    return convo.last.text
    
# print(get_query_intent("我要查詢112年彰化地區的濕地開發狀況"))
