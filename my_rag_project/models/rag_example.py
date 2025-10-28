# 這個範例是讀取pdf，用HF embedding，建chromadb，存向量資料庫，再檢索檔案，並把檔案給本地LLM進行回覆
# 資料使用的是經濟部rag需求書
# 1.loading pdf 
import fitz
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import requests
import os
from .. import config

pages = fitz.open(os.path.join(config.DATA_DIR, "rfp.pdf"))
# print(len(pages))
# print(pages[5].get_text())

# 2.splitter/ chunking data
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=[
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma ，
    "\u3001",  # Ideographic comma 、
    "\uff0e",  # Fullwidth full stop ．
    "\u3002",  # Ideographic full stop 。
    "",
])

# 3.embdding(HF)
try:
    huggingface_api_key = config.get_huggingface_api_key()
except RuntimeError as err:
    raise RuntimeError(
        "Unable to initialise the HuggingFace embedding function for rag_example.py. "
        f"{err}"
    ) from err

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=huggingface_api_key,
    model_name="intfloat/multilingual-e5-large-instruct"
)

chromadb_client = chromadb.PersistentClient(path=config.RFP_VECTOR_STORE_DIR)
chroma_collection = chromadb_client.get_or_create_collection(name="llm_rfp", embedding_function=huggingface_ef)

if chroma_collection.count() == 0:
    for idx,page in enumerate(pages):
      chunks = text_splitter.split_text(page.get_text())
    
      chroma_collection.add(
        documents = chunks,
        ids=[f"doc-1-page-{idx}-chunk-{x}" for x in range( len(chunks) ) ]
      )
# 檢索
def get_retrieved_docs(query):
    results = chroma_collection.query(query_texts=[query] , n_results=3)
    retrieved_docs = results['documents']
    retrieved_contents = "\n\n".join([item for sublist in retrieved_docs for item in sublist])
    return retrieved_contents
    
# 推論    
def generate_ollama_chat_response(query, model=config.DEFAULT_LOCAL_MODEL):
    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results['documents'][0]
    information='\n\n'.join(retrieved_documents)
    
    url = "http://localhost:11434/api/chat"
    data = {
        "model": model,
        "messages":[
        {'role':'system','content':"你是一個專業的專案經理，專長分析客戶的需求，你會收到使用者的詢問，請你從檢索出來的內容做適當回應，如果提供的資料裡面沒有，也請輸出我不清楚，請再提供其他資訊，請用繁體中文回答"},
        {"role": "user",'content': f"問題:{query}, 內容:{retrieved_documents}"},
    ],
        "format": "json",
        "stream": False
}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        return f'錯誤: {response.status_code}'
# 只檢索
# print(get_retrieved_docs("請告訴我專案範圍"))
# LLM生成
print(generate_ollama_chat_response("請告訴我專案範圍"))