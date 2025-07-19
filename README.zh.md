# TinaColab 示例
[English README](README.md)

本仓库包含一些示例脚本，演示如何使用不同的语言模型构建 RAG（检索增强生成）流程。示例展示如何拆分文档、创建向量嵌入、存入 Chroma 向量数据库，并利用 OpenAI、本地 LLM（通过 Ollama）及 Google Gemini 进行查询。

## 环境准备

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 准备数据目录。在仓库根目录下的 `data/` 中放置以下文件：
   - `data/med_instruction_v2.md` – 用于生成嵌入的 Markdown 文档。
   - `data/rfp.pdf` – 供 `rag_example.py` 使用的 PDF 文件。
   - `patient_c.txt` – 可选的病历报告文件，供聊天示例使用。
3. 配置环境变量：
   - `OPENAI_API_KEY` – OpenAI 模型的 API key。
   - `genmini_api_key` – Google Gemini 模型的 API key。
   - 可选：`MLFLOW_TRACKING_URI` 用于启用 [MLflow](https://mlflow.org/) 实验记录。

   环境变量可以直接导出到 shell，或保存在 `.env` 文件中再加载。

## 运行脚本

各示例脚本可以独立执行，默认假设数据文件和向量存储位置如上所述。

### 1. 拆分 Markdown
```bash
python 1_load_split.py
```
读取 `data/med_instruction_v2.md` 并打印拆分后的 `Document` 对象。

### 2. 生成嵌入并存储
```bash
python 2_1embed_store_query.py
```
生成嵌入并存储到 `../med_vectordata2/`，查询前需先执行一次。

### 3. 查询现有向量
```bash
python 2_2getvector_query.py
```
从已有的 Chroma 数据库中检索文档。

### 4. 与 OpenAI 聊天
```bash
python 3_1chat_LLM.py
```
利用 GPT 模型和检索到的上下文生成回答。如果需要输入示例报告，请准备 `patient_c.txt`。

### 5. 与本地 LLM 聊天
```bash
python 3_2chat_LLM_local.py
```
调用在 `http://localhost:11434/api/chat` 运行的 Ollama 服务，并使用存储的向量作为上下文。

### 6. 与 Google Gemini 聊天
```bash
python 3_3chat_LLM_gemini.py
```
示例使用 Gemini API。运行前请设置 `genmini_api_key`。

### 7. PDF RAG 示例
```bash
python rag_example.py
```
演示加载 PDF、切分、生成嵌入并通过本地 LLM 查询。

## MLflow 记录

若要使用 MLflow 记录实验结果，请设置 `MLFLOW_TRACKING_URI` 环境变量指向追踪服务器或本地目录。然后即可在脚本中通过 `mlflow` Python API 记录指标和参数（示例默认未启用）。

