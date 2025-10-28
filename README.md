# TinaColab Experiments

This repository contains a few sample scripts for interacting with different language models. Experiment metrics are logged with [MLflow](https://mlflow.org/).

## Setup

Install dependencies using pip

## Viewing MLflow Results

After running any of the scripts you can launch the MLflow UI to inspect runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the displayed address in your browser to view parameters, metrics and logged responses.


# TinaColab Examples
[中文說明](README.zh.md)

This repository contains small demonstration scripts for building a RAG (Retrieval Augmented Generation) workflow with different language models. The examples show how to split documents, create embeddings, store them in a Chroma vector database and query them using OpenAI, local LLMs via Ollama and Google Gemini.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare the data directory. Place your files in `data/` relative to the repository root:
   - `data/med_instruction_v2.md` – source markdown instructions used for embedding.
   - `data/rfp.pdf` – PDF file for `rag_example.py`.
   - `patient_c.txt` – optional patient report for the chat examples.

3. Configure environment variables:
   - `OPENAI_API_KEY` – API key for OpenAI models.
   - `GEMINI_API_KEY` – API key for Google Gemini models.
   - Optional: `MLFLOW_TRACKING_URI` to enable [MLflow](https://mlflow.org/) experiment tracking.

   Environment variables can be exported in your shell or saved in a `.env` file and loaded with your preferred tool.

## Running the Scripts

The examples can be executed individually. Each script assumes that the data files and vector stores are placed in the paths described above.

### 1. Split Markdown

```bash
python 1_load_split.py
```
Reads `data/med_instruction_v2.md` and prints the split `Document` objects.

### 2. Embed and Store

```bash
python 2_1embed_store_query.py
```
Generates embeddings and stores them in `../med_vectordata2/`. Run this once before querying.

### 3. Query Existing Vectors

```bash
python 2_2getvector_query.py
```
Retrieves documents from the previously created Chroma database.

### 4. Chat with OpenAI

```bash
python 3_1chat_LLM.py
```
Uses GPT models to generate answers using retrieved context. Requires `patient_c.txt` if you want to supply a sample report.

### 5. Chat with a Local LLM

```bash
python 3_2chat_LLM_local.py
```
Calls a running Ollama server at `http://localhost:11434/api/chat` using the stored vectors as context.

### 6. Chat with Google Gemini

```bash
python 3_3chat_LLM_gemini.py
```
Example using the Gemini API. Set `GEMINI_API_KEY` before running.

### 7. RAG over PDF

```bash
python rag_example.py
```
Demonstrates loading a PDF, chunking it, storing embeddings and then querying via a local LLM.

## MLflow Tracking

To record experiment results with MLflow, set the `MLFLOW_TRACKING_URI` environment variable to your tracking server or a local directory. Once set, metrics and parameters can be logged from within the scripts using the `mlflow` Python API (not enabled by default in these examples).


