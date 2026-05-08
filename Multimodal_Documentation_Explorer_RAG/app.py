import gradio as gr # Web Interface.
import ollama
from src.parser import MultimodalParser
from src.vector_store import VectorDB

db = VectorDB()
parser = MultimodalParser()

def ingest_pdf(file):
    if file is None:
        return "Please upload a PDF file."
    
    text_data, img_data = parser.process_pdf(file.name)
    if not text_data and not img_data:
        return "Failed to process PDF. Check console for errors."
    
    db.upsert_data(text_data, img_data)
    return f"Ingestion Successful: {len(text_data)} text chunks and {len(img_data)} images indexed."

def query_system(query):
    # Search the Vector Database for the top 3 most relevant results.
    results = db.search(query, limit = 3)
    
    context_text = ""
    retrieved_image = None
    
    # Format the retrieved context.
    for res in results:
        if res.payload['type'] == 'text':
            context_text += f"\n[Document Segment]:\n{res.payload['content']}\n"
        else:
            context_text += f"\n[Diagram Description]:\n{res.payload['content']}\n"
            # Keep the most relevant image path.
            if not retrieved_image:
                retrieved_image = res.payload['path']

    if not context_text:
        return "No relevant information found in the pdf document.", None

    # Construct the RAG Prompt.
    prompt = f"""
    You are a professional customer support. Use the provided pdf document snippets to answer the user question.
    If the context doesn't contain the answer, politely state that the document doesn't specify.
    
    CONTEXT:
    {context_text}
    
    USER QUESTION: {query}
    
    ANSWER:
    """
    
    # Generate response using Llama 3.2.
    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'user', 'content': prompt}
        ])
        return response['message']['content'], retrieved_image
    except Exception as e:
        return f"Error generating response: {str(e)}", None

# --- UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛠️ Multimodal RAG Explorer")
    gr.Markdown("Upload a pdf file to search text and images simultaneously.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Step 1: Upload PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("Build Knowledge Base", variant="primary")
            status_msg = gr.Textbox(label="Status", interactive=False)
        
        with gr.Column(scale=2):
            query_input = gr.Textbox(label="Step 2: Ask a question")
            submit_btn = gr.Button("Search & Explain")
            
    with gr.Row():
        with gr.Column(scale=3):
            output_text = gr.Markdown(label="AI Explanation")
        with gr.Column(scale=2):
            output_image = gr.Image(label="Source Diagram", type="filepath")

    ingest_btn.click(ingest_pdf, inputs=[file_input], outputs=[status_msg])
    submit_btn.click(query_system, inputs=[query_input], outputs=[output_text, output_image])

if __name__ == "__main__":
    demo.launch(theme = gr.themes.Soft())