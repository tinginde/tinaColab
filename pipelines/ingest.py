"""Utilities for ingesting and cleaning raw documents.

This module scans an input directory for supported document types, extracts
plain text, applies lightweight normalisation and writes the cleaned result to
``data/``.  The output is a JSON Lines file where each line contains a document
identifier, source path, checksum and cleaned text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text", ".json"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class Document:
    """Represents a cleaned document ready for downstream processing."""

    identifier: str
    source: Path
    text: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    """Remove spurious whitespace and normalise newlines."""

    stripped = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in stripped.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return _clean_text("\n".join(str(v) for v in data.values()))
        if isinstance(data, list):
            return _clean_text("\n".join(str(v) for v in data))
        return _clean_text(str(data))

    return _clean_text(path.read_text(encoding="utf-8"))


def _read_pdf_file(path: Path) -> str:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyMuPDF (fitz) is required to process PDF files. Install it via "
            "`pip install PyMuPDF` or remove PDF files from the docs directory."
        ) from exc

    with fitz.open(path) as doc:
        text_chunks: List[str] = []
        for page in doc:
            text_chunks.append(page.get_text())
    return _clean_text("\n".join(text_chunks))


def _iter_documents(input_dir: Path) -> Iterator[Document]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in SUPPORTED_SUFFIXES:
                text = _read_text_file(path)
            elif suffix in PDF_SUFFIXES:
                text = _read_pdf_file(path)
            else:
                logger.info("Skipping unsupported file: %s", path)
                continue
        except Exception as exc:  # pragma: no cover - log and continue
            logger.warning("Failed to read %s: %s", path, exc)
            continue

        identifier = path.relative_to(input_dir).as_posix()
        yield Document(identifier=identifier, source=path, text=text)


def ingest_documents(input_dir: Path, output_path: Path) -> Sequence[Document]:
    """Ingest documents and persist the cleaned dataset."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    documents = list(_iter_documents(input_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            record = {
                "id": doc.identifier,
                "source": doc.source.as_posix(),
                "checksum": doc.checksum,
                "text": doc.text,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Ingested %s documents into %s", len(documents), output_path)
    return documents


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest and clean documents")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing raw documents",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed_docs.jsonl"),
        help="Path to write the cleaned dataset",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv or sys.argv[1:])

    try:
        ingest_documents(args.input_dir, args.output_path)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
