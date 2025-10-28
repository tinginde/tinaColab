"""Training orchestration script for the lightweight RAG example."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from . import embed as embed_pipeline
from . import ingest as ingest_pipeline

logger = logging.getLogger(__name__)


def _load_embeddings(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings not found at {path}")

    embeddings: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            embeddings.append(json.loads(line))
    return embeddings


def train_model(embeddings: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Train a trivial model on top of the embeddings.

    The "model" is simply the centroid of all embedding vectors.  While simple,
    this structure allows downstream systems to perform similarity comparisons by
    computing the cosine similarity between a query embedding and the stored
    centroid.
    """

    vectors = [record["embedding"] for record in embeddings]
    if not vectors:
        raise ValueError("No embeddings provided for training")

    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError("Embedding dimensionality mismatch detected")
        for idx, value in enumerate(vector):
            centroid[idx] += float(value)

    centroid = [value / len(vectors) for value in centroid]

    magnitudes = [sum(val * val for val in vector) ** 0.5 for vector in vectors]
    mean_magnitude = statistics.fmean(magnitudes)
    std_magnitude = statistics.pstdev(magnitudes)

    return {
        "centroid": centroid,
        "embedding_dim": dim,
        "num_vectors": len(vectors),
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
    }


def evaluate_model(model: Dict[str, object]) -> Dict[str, float]:
    centroid = model["centroid"]
    magnitude = sum(val * val for val in centroid) ** 0.5
    return {
        "centroid_magnitude": magnitude,
        "embedding_dim": float(model["embedding_dim"]),
    }


def save_model(model: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)


def run_pipeline(
    *,
    run_ingest: bool,
    run_embed: bool,
    docs_dir: Path,
    processed_path: Path,
    embeddings_path: Path,
    model_path: Path,
    embed_dim: int,
    recompute_embeddings: bool,
) -> Dict[str, object]:
    if run_ingest:
        ingest_pipeline.ingest_documents(docs_dir, processed_path)

    if run_embed:
        embed_pipeline.embed_documents(
            processed_path,
            embeddings_path,
            dim=embed_dim,
            recompute=recompute_embeddings,
        )

    embeddings = _load_embeddings(embeddings_path)
    model = train_model(embeddings)
    metrics = evaluate_model(model)
    save_model(model, model_path)
    logger.info("Saved model to %s", model_path)
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain lightweight model")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest stage")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embed stage")
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=Path("data/processed_docs.jsonl"),
        help="Location of cleaned documents",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("embeddings/embeddings.jsonl"),
        help="Embedding store location",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model.json"),
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=embed_pipeline.DEFAULT_EMBED_DIM,
        help="Embedding dimensionality",
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Regenerate embeddings even if they exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args(argv or sys.argv[1:])

    try:
        run_pipeline(
            run_ingest=not args.skip_ingest,
            run_embed=not args.skip_embed,
            docs_dir=args.docs_dir,
            processed_path=args.processed_path,
            embeddings_path=args.embeddings_path,
            model_path=args.model_path,
            embed_dim=args.embed_dim,
            recompute_embeddings=args.recompute_embeddings,
        )
    except Exception as exc:
        logger.error("Retraining failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
