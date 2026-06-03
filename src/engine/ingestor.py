import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration Paths
DOCS_DIR = "./data/docs"
DB_DIR = "./chroma_db"

def ingest_documents():
    print("Starting Data Ingestion Pipeline...")
    
    # 1. Find all text files in the docs directory
    file_paths = glob.glob(f"{DOCS_DIR}/*.txt")
    if not file_paths:
        print(f"No .txt files found in {DOCS_DIR}. Please add some and try again.")
        return

    # 2. Read and combine all text
    raw_text = ""
    for file_path in file_paths:
        print(f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            raw_text += file.read() + "\n\n"

    # 3. Chunk the data (150 tokens, 30 token overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_text(raw_text)
    print(f"Created {len(chunks)} document chunks.")

    # 4. Initialize Embedding Model via local Ollama
    print("Loading embedding model (nomic-embed-text)...")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # 5. Save to ChromaDB
    print(f"Saving vectors to {DB_DIR}...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    
    print("Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    # Create the docs directory if it doesn't exist
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingest_documents()