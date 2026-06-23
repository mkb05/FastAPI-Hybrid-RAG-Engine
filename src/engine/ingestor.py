import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Use absolute paths to prevent Windows directory errors
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
DOCS_DIR_BASE = os.path.join(project_root, "data", "docs")
DB_DIR_BASE = os.path.join(project_root, "chroma_db")

def ingest_documents(specific_file=None, session_id="default"):
    print(f"Starting Data Ingestion Pipeline for session {session_id}...")
    
    # Check for a specific file, otherwise grab ALL text and pdf files for this session
    if specific_file:
        file_paths = [specific_file]
    else:
        session_docs_dir = os.path.join(DOCS_DIR_BASE, session_id)
        if not os.path.exists(session_docs_dir):
             print(f"No documents directory found for session {session_id}.")
             return
        file_paths = glob.glob(f"{session_docs_dir}/*.txt") + glob.glob(f"{session_docs_dir}/*.pdf")
        
    if not file_paths:
        print("No valid files found.")
        return

    # Initialize raw_text before the loop!
    raw_text = ""
    
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        
        # Route logic based on file type
        if file_path.lower().endswith('.pdf'):
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    raw_text += page.page_content + "\n\n"
            except Exception as e:
                print(f"Failed to read PDF {file_path}: {e}")
                
        elif file_path.lower().endswith('.txt'):
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text += file.read() + "\n\n"

    if not raw_text.strip():
        print("No text could be extracted.")
        return

    # Using larger 500 character chunks for better Grader context
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    print(f"Created {len(chunks)} document chunks.")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Save to a session-specific ChromaDB directory
    session_db_dir = os.path.join(DB_DIR_BASE, session_id)
    os.makedirs(session_db_dir, exist_ok=True)

    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=session_db_dir
    )
    print("Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    ingest_documents()