"""Interact with a locally hosted LLM using retrieved context."""

from __future__ import annotations

import os
import pprint
from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Union

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import mlflow
import requests

import config
from mlops.mlflow_utils import log_metrics, start_run


openai_ef_chroma = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)
chromadb_client = chromadb.PersistentClient(path=config.VECTOR_STORE_DIR)
chroma_collection = chromadb_client.get_or_create_collection(
    name="advise_template", embedding_function=openai_ef_chroma
)

system_message = '''
你是一位專門為醫療教育者提供量身定制建議和建議的健康教育助手，根據患者的醫療檢驗報告和病歷來提供這些建議。你的角色是使用檢索增強
生成（RAG）技術來提供可能的患者問題和相關的健康教育建議。這些建議應該是清晰、專業和支持性的，並考慮到患者的具體情況和需求。

在製作回應時，請考慮以下要素：
1. 患者醫療狀況的背景。
2. 具體的醫療檢驗結果及其指示。
3. 針對該情況的最佳健康教育實踐。

確保提供準確且基於證據的建議，讓醫療教育者可以直接使用這些建議來指導患者。如果你不確定答案，你可以說「我不知道」或「我不確定」，
並推薦用戶前往XXXX網站獲取更多信息。
輸出一定要用繁體中文輸出。
'''
user_request = '''以下是病人的病歷與檢驗報告{report}，根據提供的訊息{advise}，請提供:
1. 衛教時可能詢問病人的問題 2.相關的衛教建議。'''


def build_user_prompt(report: str, advise: str) -> str:
    return (
        "以下是病人的病歷與檢驗報告"
        f"{report}，根據提供的訊息{advise}，請提供:1.衛教時可能詢問病人的問題 2.相關的衛教建議。"
    )


def build_messages(report: str, advise: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": build_user_prompt(report, advise)},
    ]


def _as_query_list(query: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(query, str):
        return [query]
    return list(query)


def fetch_advise_chunks(
    collection, query: Union[str, Sequence[str]], *, n_results: int = 5
) -> str:
    results = collection.query(query_texts=_as_query_list(query), n_results=n_results)
    retrieved_documents = results["documents"][0]
    return "\n\n".join(retrieved_documents)


def get_ollama_chat_response(
    query: Union[str, Sequence[str]],
    report: str,
    *,
    model: str = config.DEFAULT_LOCAL_MODEL,
    collection=None,
    requester: Optional[Callable[..., requests.Response]] = None,
    run_context_factory: Optional[Callable[[], object]] = None,
):
    try:
        target_collection = collection or chroma_collection
        advise = fetch_advise_chunks(target_collection, query, n_results=5)
        messages = build_messages(report, advise)

        url = "http://localhost:11434/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        post = requester or requests.post
        response = post(url, json=data)
        response.raise_for_status()

        completion = response.json()
        response_content = completion["message"]["content"]
        prompt_tokens = completion["prompt_eval_count"]
        completion_tokens = completion["eval_count"]

        total_duration_sec = completion["total_duration"] / 1e9
        load_duration_sec = completion["load_duration"] / 1e9
        prompt_eval_duration_sec = completion["prompt_eval_duration"] / 1e9
        gen_eval_duration_sec = completion["eval_duration"] / 1e9

        tokens = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        durations = {
            "total_duration_sec": total_duration_sec,
            "load_duration_sec": load_duration_sec,
            "prompt_duration_sec": prompt_eval_duration_sec,
            "gen_duration_sec": gen_eval_duration_sec,
        }
        output_data = {
            "model_name": model,
            "sys_prompt": system_message,
            "user_promt": user_request,
            "Response": response_content,
            **tokens,
            **durations,
            "vectordata": "med_vectordata2",
        }

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(output_data)

        context_factory = run_context_factory or (lambda: start_run("local_llm"))
        context = context_factory()
        if context is None:
            context = nullcontext()
        with context:
            mlflow.log_text(response_content, "response.txt")
            log_metrics(
                tokens=tokens,
                durations=durations,
                model_name=model,
                prompt=messages[-1]["content"],
            )

        return response_content

    except requests.exceptions.RequestException as e:  # pragma: no cover - network handling
        return f'網路請求錯誤: {e}'
    except KeyError as e:  # pragma: no cover - unexpected response handling
        return f'回應中沒有預期的數據結構: {e}'
    except Exception as e:  # pragma: no cover - defensive fallback
        return f'未知錯誤: {e}'


def main():  # pragma: no cover - thin wrapper for manual execution
    query = "糖尿病前期"
    with open(config.PATIENT_FILE, 'r') as f:
        p_testing_report = f.read()

    response = get_ollama_chat_response(query, p_testing_report)
    print(response)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
