# 兩個方式獲得embdding
# 1. 用chromadb串openai api
import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model_name="text-embedding-3-small")

# 建本地chromaDB, name是DB的名字
chromadb_client = chromadb.PersistentClient(path="../med_vectordata2/")
chroma_collection = chromadb_client.get_or_create_collection(name="advise_template", embedding_function=openai_ef_chroma)

# #傳入chromaDB 初始化空列表來儲存提取的數據
# documents = []
# metadatas = []
# ids = []

# # 遍歷 Document 對象列表，提取所需信息
# for index, doc in enumerate(advise_docs_list):
#     documents.append(doc.page_content)
#     metadatas.append(doc.metadata)
#     ids.append(f"id{index + 1}")

# # 現在可以將這些數據添加到 collection 中
# chroma_collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids
# )

# 檢索querying text, n_results是檢索3筆
query="糖尿病前期的管理"
results = chroma_collection.query(query_texts=[query] , n_results=3)
retrieved_documents = results['documents'][0]
information = "\n\n".join(retrieved_documents)
# print(chroma_collection.get(include=["documents"]))
print(information)
