# TinaColab 示例
[English README](README.md)

本倉庫包含一些示例腳本，演示如何使用不同的語言模型構建 RAG（檢索增強生成）流程。示例展示如何拆分文檔、創建向量嵌入、存入 Chroma 向量數據庫，並利用 OpenAI、本地 LLM（通過 Ollama）及 Google Gemini 進行查詢。

## 環境準備

1. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
2. 準備數據目錄。在倉庫根目錄下的 `data/` 中放置以下文件：
   - `data/med_instruction_v2.md` – 用於生成嵌入的 Markdown 文檔。
   - `data/rfp.pdf` – 供 `rag_example.py` 使用的 PDF 文件。
   - `patient_c.txt` – 可選的病歷報告文件，供聊天示例使用。
3. 配置環境變量：
   - `OPENAI_API_KEY` – OpenAI 模型的 API key。
   - `GEMINI_API_KEY` – Google Gemini 模型的 API key。
   - 可選：`MLFLOW_TRACKING_URI` 用於啟用 [MLflow](https://mlflow.org/) 實驗記錄。

   環境變量可以直接導出到 shell，或保存在 `.env` 文件中再加載。

## 運行腳本

各示例腳本可以獨立執行，默認假設數據文件和向量存儲位置如上所述。

### 1. 拆分 Markdown
```bash
python 1_load_split.py
```
讀取 `data/med_instruction_v2.md` 並打印拆分後的 `Document` 對象。

### 2. 生成嵌入並存儲
```bash
python 2_1embed_store_query.py
```
生成嵌入並存儲到 `../med_vectordata2/`，查詢前需先執行一次。

### 3. 查詢現有向量
```bash
python 2_2getvector_query.py
```
從已有的 Chroma 數據庫中檢索文檔。

### 4. 與 OpenAI 聊天
```bash
python 3_1chat_LLM.py
```
利用 GPT 模型和檢索到的上下文生成回答。如果需要輸入示例報告，請準備 `patient_c.txt`。

### 5. 與本地 LLM 聊天
```bash
python 3_2chat_LLM_local.py
```
調用在 `http://localhost:11434/api/chat` 運行的 Ollama 服務，並使用存儲的向量作為上下文。

### 6. 與 Google Gemini 聊天
```bash
python 3_3chat_LLM_gemini.py
```
示例使用 Gemini API。運行前請設置 `GEMINI_API_KEY`。

### 7. PDF RAG 示例
```bash
python rag_example.py
```
演示加載 PDF、切分、生成嵌入並通過本地 LLM 查詢。

## MLflow 記錄

若要使用 MLflow 記錄實驗結果，請設置 `MLFLOW_TRACKING_URI` 環境變量指向追蹤服務器或本地目錄。然後即可在腳本中通過 `mlflow` Python API 記錄指標和參數（示例默認未啟用）。
