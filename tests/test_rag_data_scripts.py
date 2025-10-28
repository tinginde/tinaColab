from types import SimpleNamespace
import pytest

from my_rag_project.pipelines import embed_store_query, vector_query


class FakeEmbeddingFunction:
    def __call__(self, input):
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)
        embeddings = []
        for text in texts:
            score = float(sum(ord(char) for char in text) % 97)
            embeddings.append([score, score / 2, score / 3])
        return embeddings

    def name(self):
        return "fake"

    def is_legacy(self):
        return False


@pytest.fixture()
def embed_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return embed_store_query


@pytest.fixture()
def query_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return vector_query


@pytest.fixture()
def fake_docs():
    return [
        SimpleNamespace(
            page_content="糖尿病前期的管理",
            metadata={"source": "doc1"},
        ),
        SimpleNamespace(
            page_content="高血壓飲食建議",
            metadata={"source": "doc2"},
        ),
    ]


@pytest.fixture()
def fake_embedding():
    return FakeEmbeddingFunction()


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add(self, *, documents, metadatas, ids):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def get(self, include):
        result = {}
        for item in include:
            if item == "documents":
                result[item] = list(self.documents)
            elif item == "metadatas":
                result[item] = list(self.metadatas)
            elif item == "ids":
                result[item] = list(self.ids)
        return result

    def query(self, *, query_texts, n_results):
        return {"documents": [self.documents[:n_results]]}


class FakeClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name, embedding_function):
        if name not in self.collections:
            self.collections[name] = FakeCollection()
        return self.collections[name]


def test_create_collection_from_docs(embed_module, fake_docs, fake_embedding):
    client = FakeClient()
    collection = embed_module.create_collection_from_docs(
        fake_docs,
        client=client,
        collection_name="test_collection",
        embedding_function=fake_embedding,
    )

    stored = collection.get(include=["documents", "metadatas", "ids"])
    assert stored["documents"] == [doc.page_content for doc in fake_docs]
    assert stored["metadatas"] == [doc.metadata for doc in fake_docs]
    assert stored["ids"] == [f"id{i + 1}" for i in range(len(fake_docs))]


def test_query_collection_returns_expected_payload(
    embed_module, query_module, fake_docs, fake_embedding
):
    client = FakeClient()
    collection = embed_module.create_collection_from_docs(
        fake_docs,
        client=client,
        collection_name="test_collection",
        embedding_function=fake_embedding,
    )

    information = query_module.query_collection(
        collection, "糖尿病前期的管理", n_results=1
    )

    assert fake_docs[0].page_content in information
