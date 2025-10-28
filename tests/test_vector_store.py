from my_rag_project.embeddings.vector_store import VectorIndex
from my_rag_project.utils.text_utils import Document


def test_query_index_returns_relevant_doc():
    docs = [
        Document("apple orange", {}),
        Document("banana pear", {}),
        Document("apple pie recipe", {}),
    ]
    index = VectorIndex(docs)
    results = index.query("apple", k=2)
    assert len(results) == 2
    assert results[0].page_content in {"apple orange", "apple pie recipe"}
    assert results[0].page_content != "banana pear"
