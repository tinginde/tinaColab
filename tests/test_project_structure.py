from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "my_rag_project"

REQUIRED_DIRECTORIES = [
    "docs",
    "data",
    "embeddings",
    "models",
    "pipelines",
    "api",
    "mlops",
    "workflows",
]


@pytest.mark.parametrize("relative_path", REQUIRED_DIRECTORIES)
def test_required_directories_exist(relative_path: str):
    target = PROJECT_ROOT / relative_path
    assert target.exists() and target.is_dir(), f"Missing required directory: {relative_path}"


def test_tests_directory_present():
    assert (PROJECT_ROOT.parent / "tests").is_dir(), "tests directory must exist"
