# 接續 1_load_split.py
# 獲得embdding, 建自己的DB,向量資料、chunking資料傳入db, 即可檢索，若要再次使用該向量資料庫，就不用再傳入一次資料，參考2_2getvector_query.py

import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# 1. 用chromadb用openai api, 請先輸入自己的openai_api_key
openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model_name="text-embedding-3-small")

# 2. 用chromadb用huggingface embedding, 中文用這個不會亂碼, HF api_key自己申請
import chromadb.utils.embedding_functions as embedding_functions
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="intfloat/multilingual-e5-large-instruct"
)

# 建本地chromaDB, name是DB的名字, 要用HF embedding就要改成huggingface_ef，目前med_vectordata2是openai_embedding
chromadb_client = chromadb.PersistentClient(path="../med_vectordata2/")
chroma_collection = chromadb_client.get_or_create_collection(name="advise_template", embedding_function=openai_ef_chroma)

#傳入chromaDB 初始化空列表來儲存提取的數據
documents = []
metadatas = []
ids = []

# 遍歷 Document 對象列表，提取所需信息
for index, doc in enumerate(advise_docs_list):
    documents.append(doc.get_content)
    metadatas.append(doc.metadata)
    ids.append(f"id{index + 1}")

# 現在可以將這些數據添加到 collection 中
chroma_collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# 檢索querying text
query="糖尿病前期的管理"
results = chroma_collection.query(query_texts=[query] , n_results=3)
retrieved_documents = results['documents'][0]
information = "\n\n".join(retrieved_documents)
# print(chroma_collection.get(include=["documents"]))
print(information)

