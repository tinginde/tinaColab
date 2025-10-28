"""Create a local Chroma vector store and query it.

This script can be run directly.  It dynamically loads the splitting logic
from ``1_load_split.py`` so that the markdown file will be read and split into
``advise_docs_list`` automatically.  The resulting documents are embedded and
stored in a Chroma database.

Once the embeddings are stored they can be queried again without reloading the
data (see ``2_2getvector_query.py``).
"""

import os
import importlib.util
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import config


def load_advise_docs():
    """Load and split the markdown instructions.

    The logic is imported from ``1_load_split.py`` so we don't duplicate the
    splitting implementation here.
    """
    module_path = os.path.join(os.path.dirname(__file__), "1_load_split.py")
    spec = importlib.util.spec_from_file_location("load_split", module_path)
    load_split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_split)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "med_instruction_v2.md")
    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    return load_split.read_split_md(md_content)


advise_docs_list = load_advise_docs()

# 1. 用chromadb用openai api, 請先輸入自己的openai_api_key
openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model_name="text-embedding-3-small")

# 2. 用chromadb用huggingface embedding, 中文用這個不會亂碼, HF api_key自己申請
try:
    huggingface_api_key = config.get_huggingface_api_key()
except RuntimeError as err:
    raise RuntimeError(
        "Unable to initialise the HuggingFace embedding function. "
        f"{err}"
    ) from err

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=huggingface_api_key,
    model_name="intfloat/multilingual-e5-large-instruct"
)

# 建本地chromaDB, name是DB的名字, 要用HF embedding就要改成huggingface_ef，目前med_vectordata2是openai_embedding
chromadb_client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
chroma_collection = chromadb_client.get_or_create_collection(name="advise_template", embedding_function=openai_ef_chroma)

#傳入chromaDB 初始化空列表來儲存提取的數據
documents = []
metadatas = []
ids = []

# 遍歷 Document 對象列表，提取所需信息
for index, doc in enumerate(advise_docs_list):
    # ``Document`` objects expose their text via ``page_content``
    documents.append(doc.page_content)
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

