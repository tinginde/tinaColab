from text_utils import Document
from vector_store import VectorIndex


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
