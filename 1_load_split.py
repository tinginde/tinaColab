# 讀md檔，分割文本，好處是得到Doc類別的物件加上metadata

import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 獲取當前腳本的絕對路徑
script_dir = os.path.dirname(os.path.abspath(__file__))

# 構建相對於腳本位置的文件路徑
file_path = os.path.join(script_dir, "data", "med_instruction_v2.md")

# 分割md成chunk且有meta data
def read_split_md(md_doc):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )

    md_header_splits = markdown_splitter.split_text(md_doc)
    
    # Char-level splits
    chunk_size = 700
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    
    return splits

# 讀取文件內容
with open(file_path, "r", encoding="utf-8") as f:
    md_content = f.read()

# 進行分割
advise_docs_list = read_split_md(md_content)
print(advise_docs_list)



