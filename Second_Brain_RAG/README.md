# Second_Brain_RAG

Retrieval-Augmented Generation (RAG) CLI tool that allows you to chat with your local PDFs documents using OpenAI and ChromaDB.

## Features
- **Document Ingestion:** Automatically processes PDFs from a local directory.
- **Vector Search:** Uses OpenAI Embeddings and ChromaDB for high-accuracy semantic retrieval.
- **Persistent Memory:** Saves the processed index locally so you don't re-process files every time.

## Tech Stack
- **Orchestration:** **LangChain** (Managing the flow between data and LLM)
- **AI Model:** **OpenAI GPT-4o-mini** (High-speed, cost-effective reasoning)
- **Embeddings:** **OpenAI Text-Embedding-3-Small** (Converting text to vectors)
- **Vector Database:** **ChromaDB** (Fast, local vector storage and similarity search)
- **Document Parsing:** **PyPDF** (Extracting clean text from PDF files)