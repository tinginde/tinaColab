from types import SimpleNamespace

import pytest

from my_rag_project.api import chat_llm, chat_llm_local


@pytest.fixture()
def chat_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(chat_llm, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_llm.mlflow, "log_text", lambda *args, **kwargs: None)
    return chat_llm


@pytest.fixture()
def local_chat_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(chat_llm_local, "log_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_llm_local.mlflow, "log_text", lambda *args, **kwargs: None)
    return chat_llm_local


class StubCollection:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def query(self, *, query_texts, n_results):
        self.queries.append((query_texts, n_results))
        return {"documents": [self.documents[:n_results]]}


class StubChatCompletions:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=5, total_tokens=9)
        choice = SimpleNamespace(message=SimpleNamespace(content=self.response_text))
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            system_fingerprint="fingerprint",
        )


class StubOpenAIClient:
    def __init__(self, response_text):
        self.chat = SimpleNamespace(completions=StubChatCompletions(response_text))


class StubResponse:
    def __init__(self, payload):
        self._payload = payload
        self.raise_calls = []

    def raise_for_status(self):
        self.raise_calls.append(True)

    def json(self):
        return self._payload


class StubRequester:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, url, json):
        self.calls.append((url, json))
        return StubResponse(self.payload)


class NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


@pytest.fixture()
def run_context_factory():
    return lambda: NoOpContext()


def test_build_user_prompt_contains_context(chat_module):
    prompt = chat_module.build_user_prompt("報告內容", "建議內容")
    assert "報告內容" in prompt
    assert "建議內容" in prompt


def test_get_chat_response_assembles_messages(chat_module, run_context_factory):
    collection = StubCollection(["建議一", "建議二"])
    client = StubOpenAIClient("最終回應")

    response = chat_module.get_chat_response(
        "糖尿病前期",
        "病歷描述",
        collection=collection,
        client=client,
        run_context_factory=run_context_factory,
    )

    assert response == "最終回應"
    assert collection.queries[0][0] == ["糖尿病前期"]
    messages = client.chat.completions.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "病歷描述" in messages[1]["content"]


def test_get_ollama_chat_response_builds_payload(local_chat_module, run_context_factory):
    collection = StubCollection(["建議一", "建議二"])
    payload = {
        "message": {"content": "本地回應"},
        "prompt_eval_count": 7,
        "eval_count": 11,
        "total_duration": 1_000_000_000,
        "load_duration": 200_000_000,
        "prompt_eval_duration": 300_000_000,
        "eval_duration": 400_000_000,
    }
    requester = StubRequester(payload)

    response = local_chat_module.get_ollama_chat_response(
        "糖尿病前期",
        "病歷描述",
        collection=collection,
        requester=requester,
        run_context_factory=run_context_factory,
        model="demo-model",
    )

    assert response == "本地回應"
    url, data = requester.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert data["model"] == "demo-model"
    assert data["messages"][0]["role"] == "system"
    assert "病歷描述" in data["messages"][1]["content"]
