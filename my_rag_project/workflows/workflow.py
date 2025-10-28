"""Simple end-to-end workflow for demonstration."""
import os
from my_rag_project.embeddings.vector_store import VectorIndex
from my_rag_project.utils.text_utils import read_split_md


def main():
    script_dir = os.path.dirname(__file__)
    md_path = os.path.join(script_dir, "data", "sample.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    docs = read_split_md(md_content)
    index = VectorIndex(docs)

    query = "diet"
    results = index.query(query, k=2)
    answer = "\n".join(doc.page_content for doc in results)
    print("Response:\n" + answer)


if __name__ == "__main__":
    main()
