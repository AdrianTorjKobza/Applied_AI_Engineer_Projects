from dotenv import load_dotenv
load_dotenv() # Reads .env into os.environ.
import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("knowledge")
llm = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY
the provided context. If the answer isn't in the context, say so clearly —
do not make up information. Always cite which source your answer comes from."""

def retrieve(question: str, top_k: int = 5) -> list[dict]:
    q_embedding = model.encode([question]).tolist() # Turn your question into numbers.

    # ChromaDB looks at those numbers (question) and finds the 5 (top_k) chunks in the database that are mathematically "closest" to your question.
    results = collection.query(
        query_embeddings = q_embedding,
        n_results = top_k,
        include = ["documents", "metadatas", "distances"],
    )

    chunks = []
    
    # This cleans up the data from the database and puts it into a nice list of dictionaries that includes the text, where it came from, and a "score" (where 1.0 is a perfect match).
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "score": round(1 - dist, 3),  # cosine similarity
        })
    return chunks

# This takes your question and pastes all those snippets from the database above it.
# It's essentially telling the AI: "Read these 5 paragraphs, then answer this question based on them."
def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(
        f"[Source: {c['source']} | relevance: {c['score']}]\n{c['text']}"
        for c in chunks
    )
    return f"""Use the following context to answer the question.

CONTEXT:
{context_blocks}

QUESTION: {question}

ANSWER:"""

def ask(question: str) -> tuple[str, list[dict]]:
    chunks = retrieve(question) # Find the relevant documents.
    prompt = build_prompt(question, chunks) # Package the documents and question together.

    # Send the package to Claude.
    response = llm.messages.create(
        model = "claude-sonnet-4-20250514",
        max_tokens = 1024,
        system = SYSTEM_PROMPT,
        messages = [{"role": "user", "content": prompt}],
    )

    return response.content[0].text, chunks # Return the AI's final answer and the chunks it used.