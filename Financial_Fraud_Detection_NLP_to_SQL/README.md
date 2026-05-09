# Financial Fraud Detection NLP-to-SQL

This project demonstrates a system where non-technical fraud analysts can write rules in natural language, which are then converted into validated PostgreSQL queries via a local LLM (Llama 3 8B).

## Features
- **Privacy-First:** Powered by self-hosted Llama 3 via Ollama.
- **SQL Validation:** Uses `sqlglot` to ensure syntax integrity before execution.
- **Multi-Tenant Ready:** Architecture supports dynamic schema injection via RAG.
- **Mock Execution:** Runs queries against an in-memory database to verify results.

## Proof of Concept Demo
[Link to your recording here]

## Tech Stack
- **Language:** Python 3.9+
- **Inference Engine:** Ollama (Running Llama 3 8B)
- **Frontend:** Streamlit (Data-centric UI)
- **SQL Logic:** SQLGlot (Dialect transpilation & validation)
- **ORM/Database:** SQLAlchemy & SQLite (Mocking PostgreSQL behavior)
- **Orchestration:** LangChain (Prompt & LLM management)

## Architecture Overview
- **Parser:** Prevents SQL injection by validating against a strict schema.
- **Intermediate Representation:** Uses JSON as a bridge to ensure the LLM provides both the query and the reasoning.
- **Evaluation:** Latency is tracked per request (typically < 2s for Llama 3 8B).

## Setup Instructions
1. **Install Ollama:** Download from [ollama.ai](https://ollama.ai).
2. **Pull Model:** `ollama pull llama3:8b`.
3. **Create a virtual environment:** `python -m venv venv`.
4. **Activate the environment:**
- On Windows:
```bash
venv\Scripts\activate
- On macOS/Linux:
```bash
source venv/bin/activate
5. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
6. **Run the app:**
   ```bash
   streamlit run app.py