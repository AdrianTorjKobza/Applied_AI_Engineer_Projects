import os, glob
from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer # Converts text into lists of numbers (embeddings).

DOCS_DIR = "./docs"    # Location of the pdf files.
CHUNK_SIZE = 500       # Split text in chunks of 500 characters.
CHUNK_OVERLAP = 50     # Overlap the chunks of text. This helps to preserve context.

# Load the pdf files and extract the text.
def load_text(path: str) -> str:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return Path(path).read_text(encoding="utf-8")

# The "sliding window" approach.
# It grabs 500 characters, then moves the starting point forward by 450 (500 minus the 50 overlap), and repeats until the end of the document.
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0

    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    
    return [c.strip() for c in chunks if c.strip()]

def ingest():
    # Load embedding model (runs locally, no API cost).
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Set up ChromaDB, saved locally.
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("knowledge")

    files = glob.glob(f"{DOCS_DIR}/**/*.*", recursive=True) # This looks inside the ./docs folder and finds every single file, even in subfolders.
    all_chunks, all_ids, all_meta = [], [], []

    for filepath in files:
        print(f"Processing {filepath}...")

        text = load_text(filepath)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{Path(filepath).stem}__{i}")
            all_meta.append({"source": filepath, "chunk_index": i})

    # Embed all chunks in one batch (much faster than one-by-one).
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    # Push the text, the math (embeddings), the IDs, and the metadata into ChromaDB.
    collection.upsert(
        documents = all_chunks,
        embeddings = embeddings,
        ids = all_ids,
        metadatas = all_meta,
    )
    print(f"Done. Indexed {len(all_chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    ingest()