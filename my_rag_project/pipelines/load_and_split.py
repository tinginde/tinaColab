"""Utilities for loading and splitting markdown documents."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .. import config

DEFAULT_DOCUMENT = "med_instruction_v2.md"


def read_split_md(md_doc: str) -> Iterable:
    """Split a markdown string into LangChain ``Document`` objects."""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(md_doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=30,
    )
    return text_splitter.split_documents(md_header_splits)


def load_markdown(file_path: Optional[Path | str] = None) -> str:
    """Load the markdown document from ``file_path``.

    If ``file_path`` is ``None`` the default document configured for the
    repository is used.
    """

    target_path = Path(file_path) if file_path else Path(config.DATA_DIR) / DEFAULT_DOCUMENT
    with target_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_default_documents(file_path: Optional[Path | str] = None):
    """Return the default documents as produced by :func:`read_split_md`."""

    return read_split_md(load_markdown(file_path))


__all__ = ["DEFAULT_DOCUMENT", "load_default_documents", "load_markdown", "read_split_md"]



