"""Query utilities for the local Chroma vector store."""

from __future__ import annotations

import os
from typing import Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from .. import config


def _default_openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )


def get_collection(
    *,
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = "advise_template",
    embedding_function=None,
    vector_store_dir: Optional[str] = None,
):
    if client is None:
        client = chromadb.PersistentClient(path=vector_store_dir or config.VECTOR_STORE_DIR)

    if embedding_function is None:
        embedding_function = _default_openai_embedding_function()

    return client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )


def query_collection(collection, query: str, *, n_results: int = 3) -> str:
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_documents = results.get("documents", [[]])[0]
    return "\n\n".join(retrieved_documents)


def main():  # pragma: no cover - thin wrapper over tested helpers
    collection = get_collection()
    information = query_collection(collection, "糖尿病前期的管理", n_results=3)
    print(information)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
