"""Embedding pipeline that converts processed documents into vector form."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EMBED_DIM = 16


class EmbeddingStore:
    """A small helper that reads and writes embedding JSONL files."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._store: Dict[str, Dict[str, object]] = {}
        if path.exists():
            self._store = {
                record["id"]: record
                for record in self._iter_records()
            }

    def _iter_records(self) -> Iterator[Dict[str, object]]:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)

    def get(self, doc_id: str) -> Dict[str, object] | None:
        return self._store.get(doc_id)

    def update(self, doc_id: str, record: Dict[str, object]) -> None:
        self._store[doc_id] = record

    def delete(self, doc_id: str) -> None:
        self._store.pop(doc_id, None)

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for record in self._store.values():
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def records(self) -> Iterable[Dict[str, object]]:
        return self._store.values()


def _load_processed_docs(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed documents not found at {path}")

    docs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            docs.append(data)
    return docs


def _hash_to_unit_interval(text: str) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = []
    for i in range(0, len(digest), 2):
        value = int.from_bytes(digest[i : i + 2], byteorder="little")
        vector.append(value / 65535.0)
    return vector


def embed_text(text: str, dim: int = DEFAULT_EMBED_DIM) -> List[float]:
    """Generate a deterministic pseudo-embedding for the given text."""

    base_vector = _hash_to_unit_interval(text)
    if len(base_vector) < dim:
        repeats = math.ceil(dim / len(base_vector))
        base_vector = (base_vector * repeats)[:dim]
    return base_vector[:dim]


def embed_documents(
    processed_docs_path: Path,
    embeddings_path: Path,
    *,
    dim: int = DEFAULT_EMBED_DIM,
    recompute: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Embed processed documents and write them to disk.

    Args:
        processed_docs_path: Location of the JSONL file produced by ingestion.
        embeddings_path: Output location for the embedding store.
        dim: Dimensionality of the generated embeddings.
        recompute: If ``True`` all embeddings are regenerated from scratch.
    """

    docs = _load_processed_docs(processed_docs_path)
    store = EmbeddingStore(embeddings_path)

    if recompute:
        store = EmbeddingStore(embeddings_path)
        store._store.clear()

    updated = 0
    for doc in docs:
        doc_id = doc["id"]
        checksum = doc["checksum"]
        existing = store.get(doc_id)
        if existing and existing.get("checksum") == checksum and not recompute:
            logger.debug("Skipping %s (unchanged)", doc_id)
            continue

        embedding = embed_text(doc["text"], dim=dim)
        record = {
            "id": doc_id,
            "checksum": checksum,
            "embedding": embedding,
        }
        store.update(doc_id, record)
        updated += 1

    # Drop embeddings for documents that were removed
    doc_ids = {doc["id"] for doc in docs}
    for existing_id in list(store._store.keys()):
        if existing_id not in doc_ids:
            store.delete(existing_id)

    store.persist()
    logger.info("Updated %s embeddings (total %s)", updated, len(store._store))
    return {rec["id"]: rec for rec in store.records()}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed_docs.jsonl"),
        help="Path to the processed documents JSONL",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("embeddings/embeddings.jsonl"),
        help="Where to store the embeddings",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help="Embedding dimensionality",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force regeneration of all embeddings",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv or sys.argv[1:])

    try:
        embed_documents(
            args.input_path,
            args.output_path,
            dim=args.dim,
            recompute=args.recompute,
        )
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
