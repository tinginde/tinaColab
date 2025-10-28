"""Create and populate a local Chroma vector store.

This module exposes utilities that ingest the markdown guidance contained in
``data/med_instruction_v2.md`` into a persistent Chroma collection.  The
functions are written so that they can be reused from unit tests where the
embedding function, storage path, or client can be swapped for lightweight
fakes.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Iterable, Optional, Sequence, Tuple

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import config


CollectionPayload = Tuple[Sequence[str], Sequence[dict], Sequence[str]]


def load_advise_docs(file_path: Optional[str] = None):
    """Load and split the markdown instructions.

    The logic is imported from ``1_load_split.py`` so we don't duplicate the
    splitting implementation here.  A custom ``file_path`` can be supplied to
    facilitate tests that rely on temporary fixtures.
    """

    module_path = os.path.join(os.path.dirname(__file__), "1_load_split.py")
    spec = importlib.util.spec_from_file_location("load_split", module_path)
    load_split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_split)

    if file_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "data", "med_instruction_v2.md")

    with open(file_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    return load_split.read_split_md(md_content)


def _default_openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )


def _default_huggingface_embedding_function():
    try:
        huggingface_api_key = config.get_huggingface_api_key()
    except RuntimeError as err:  # pragma: no cover - exercised indirectly
        raise RuntimeError(
            "Unable to initialise the HuggingFace embedding function. "
            f"{err}"
        ) from err

    return embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=huggingface_api_key,
        model_name="intfloat/multilingual-e5-large-instruct",
    )


def prepare_documents_payload(docs: Iterable) -> CollectionPayload:
    """Transform a list of ``Document``-like objects into Chroma payloads."""

    documents = []
    metadatas = []
    ids = []

    for index, doc in enumerate(docs):
        page_content = getattr(doc, "page_content", str(doc))
        metadata = getattr(doc, "metadata", {})
        documents.append(page_content)
        metadatas.append(metadata)
        ids.append(f"id{index + 1}")

    return documents, metadatas, ids


def create_collection_from_docs(
    docs: Iterable,
    *,
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = "advise_template",
    embedding_function=None,
    vector_store_dir: Optional[str] = None,
):
    """Create (or reuse) a Chroma collection and populate it with ``docs``.

    Parameters are overridable to support dependency injection in tests.
    """

    if client is None:
        client = chromadb.PersistentClient(path=vector_store_dir or config.VECTOR_STORE_DIR)

    if embedding_function is None:
        embedding_function = _default_openai_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    documents, metadatas, ids = prepare_documents_payload(docs)
    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection


def main():  # pragma: no cover - thin wrapper over tested helpers
    advise_docs_list = load_advise_docs()
    collection = create_collection_from_docs(advise_docs_list)

    query = "糖尿病前期的管理"
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_documents = results["documents"][0]
    information = "\n\n".join(retrieved_documents)
    print(information)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

