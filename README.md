# TinaColab Experiments

This repository collects lightweight Retrieval Augmented Generation (RAG) examples that run against a variety of language models.
Each script focuses on a specific step of the workflow—from splitting documents and building embeddings to chatting with remote or
local models. When desired, experiments can be tracked with [MLflow](https://mlflow.org/).

[中文說明](README.zh.md)

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Examples](#running-the-examples)
- [MLflow Tracking](#mlflow-tracking)
- [Testing](#testing)

## Requirements

- Python 3.10 or newer
- Access to the APIs you plan to call (OpenAI, HuggingFace, Google Gemini, Ollama)
- Optional: an MLflow tracking server or local directory if you wish to log experiments

## Setup

1. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the `data/` directory relative to the repository root:
   - `data/med_instruction_v2.md` – Markdown instructions used for embedding.
   - `data/rfp.pdf` – PDF source for `rag_example.py`.
   - `patient_c.txt` – optional patient report consumed by the chat examples.

## Environment Variables

Configure credentials and optional settings before running the scripts:

- `OPENAI_API_KEY` – required for OpenAI GPT models.
- `HUGGINGFACE_API_KEY` – used by `2_1embed_store_query.py` and `rag_example.py` for HuggingFace embeddings (you can also hardcode
  the value in `config.py`).
- `GEMINI_API_KEY` – enables the Google Gemini example.
- `MLFLOW_TRACKING_URI` (optional) – target MLflow tracking store.

Environment variables can be exported in your shell or stored in a `.env` file loaded by your preferred tool.

## Running the Examples

Each script can be executed independently. Ensure that the data files and vector stores referenced above are in place.

| Step | Command | Description |
| --- | --- | --- |
| 1 | `python 1_load_split.py` | Split `data/med_instruction_v2.md` into `Document` chunks. |
| 2 | `python 2_1embed_store_query.py` | Generate embeddings and persist them to `../med_vectordata2/`. Run once before querying. |
| 3 | `python 2_2getvector_query.py` | Query the stored vectors from the Chroma database. |
| 4 | `python 3_1chat_LLM.py` | Chat with OpenAI GPT models, optionally providing `patient_c.txt` as context. |
| 5 | `python 3_2chat_LLM_local.py` | Chat with a local Ollama model at `http://localhost:11434/api/chat`. |
| 6 | `python 3_3chat_LLM_gemini.py` | Interact with Google Gemini using the embeddings as context. |
| 7 | `python rag_example.py` | Perform full RAG over a PDF, including chunking and local LLM querying. |

## MLflow Tracking

Launch the MLflow UI to inspect logged runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the displayed address in your browser to view parameters, metrics and generated responses.

Set `MLFLOW_TRACKING_URI` to point at a tracking server or directory if you want the scripts to log metrics via the `mlflow`
Python API.

## Testing

Offline unit tests cover document ingestion, vector querying and prompt assembly flows. Run them to verify changes:

```bash
pytest tests/test_rag_data_scripts.py tests/test_rag_chat_scripts.py
```

The tests rely on in-memory fakes and mocked API clients, so they do not require network access or real API keys beyond the dummy
values supplied by the test harness.
