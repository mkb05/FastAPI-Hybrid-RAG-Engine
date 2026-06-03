import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader  # --- NEW IMPORT ---

DOCS_DIR = "./data/docs"
DB_DIR = "./chroma_db"

def ingest_documents(specific_file=None):
    print("Starting Data Ingestion Pipeline...")
    
    # Check for a specific file, otherwise grab ALL text and pdf files
    if specific_file:
        file_paths = [specific_file]
    else:
        file_paths = glob.glob(f"{DOCS_DIR}/*.txt") + glob.glob(f"{DOCS_DIR}/*.pdf")
        
    if not file_paths:
        print(f"No valid files found in {DOCS_DIR}.")
        return

    raw_text = ""
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # --- NEW ROUTING LOGIC ---
        if file_path.lower().endswith('.pdf'):
            try:
                # Use LangChain's PDF parser
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    raw_text += page.page_content + "\n\n"
            except Exception as e:
                print(f"Failed to read PDF {file_path}: {e}")
                
        elif file_path.lower().endswith('.txt'):
            # Use our original text reader
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text += file.read() + "\n\n"

    # The rest remains exactly the same!
    if not raw_text.strip():
        print("No text could be extracted.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split_text(raw_text)
    print(f"Created {len(chunks)} document chunks.")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )
    print("Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    ingest_documents()