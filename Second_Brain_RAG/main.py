import os
import httpx

# Force the OpenAI client to ignore proxy issues and use a standard connection.
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["CURL_CA_BUNDLE"] = ""

from openai import OpenAI

# Bypass SSL verification (Development only).
# This prevents the ConnectionError caused by local SSL certificate issues.
custom_http_client = httpx.Client(verify=False)

from dotenv import load_dotenv  # Load the API key.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader # Find and read PDF files.
from langchain_text_splitters import RecursiveCharacterTextSplitter # Cuts long documents into smaller, manageable chunks.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Turn text into numbers (vectors) + OpenAI Chatbot.
from langchain_community.vectorstores import Chroma # The vector database.
from langchain_classic.chains import RetrievalQA # Connect the database with AI.

load_dotenv() # Load the Environment Variables.

if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

class SecondBrain:
    def __init__(self, data_path="data/", db_path="db/"):
        self.data_path = data_path # Tell the "brain" where the data is.
        self.db_path = db_path # Where to save the database.

        # Add 'http_client' to both OpenAI components.
        self.embeddings = OpenAIEmbeddings(
            http_client = custom_http_client
        )

        # Select the AI model and set temperature to 0, to make AI precise (less creative).
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini", 
            temperature = 0,
            http_client = custom_http_client
        )

    def ingest_docs(self):
        """Step 1: Load and Split Documents"""
        print("📁 Loading documents...")

        # Load and extract the txt from PDF files.
        loader = DirectoryLoader(self.data_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        
        # Split the text into chunks of 1000 characters. AI can't read the whole pdf at once.
        # Chunk Overlap set to 100. We keep a tiny bit of the previous chunk in the next one so that context isn't lost mid-sentence.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        print(f"✅ Split into {len(chunks)} chunks. Saving to database...")
        
        # Turn the chunks into mathematical vectors and saves them into Chroma database (locally).
        vector_db = Chroma.from_documents(
            documents = chunks, 
            embedding = self.embeddings, 
            persist_directory = self.db_path
        )
        print("💾 Database ready!")

    def ask(self, question: str):
        """Step 2: Retrieve and Generate"""
        # Load the Chrome database we saved earlier.
        vector_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings)
        
        # Build the RAG Chain.
        chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff", # It "stuffs" those top 3 chunks into the prompt along with your question and sends them to GPT.
            retriever = vector_db.as_retriever(search_kwargs = {"k": 3}) # Get top 3 relevant chunks from the database.
        )
        
        response = chain.invoke(question)
        return response["result"]

if __name__ == "__main__":
    brain = SecondBrain()
    
    # If database does not exist yet, runs the "ingest documents" function.
    if not os.path.exists("db"):
        brain.ingest_docs()
        
    print("\n🧠 Second Brain is online. Type 'exit' or 'quit' to quit the application.")

    while True:
        user_query = input("\nYOU: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        print("🤖 Thinking...")
        answer = brain.ask(user_query)
        print(f"\nAI: {answer}")