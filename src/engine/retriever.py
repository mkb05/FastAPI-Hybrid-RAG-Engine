import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

DB_DIR_BASE = "./chroma_db"

class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever...")
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None
        self.documents = []
        self.bm25 = None
        self.current_session_id = None

    def load_session(self, session_id: str):
        """Loads the database for a specific session."""
        if self.current_session_id == session_id:
            return # Already loaded
            
        print(f"[System] Loading memory for session {session_id}...")
        self.current_session_id = session_id
        session_db_dir = f"{DB_DIR_BASE}/{session_id}"
        
        # Reset current state
        self.documents = []
        self.bm25 = None
        self.vector_store = None

        if os.path.exists(session_db_dir):
            try:
                self.vector_store = Chroma(persist_directory=session_db_dir, embedding_function=self.embedding_model)
                db_data = self.vector_store.get()
                self.documents = db_data.get('documents', [])
                
                if self.documents:
                    tokenized_docs = [doc.lower().split() for doc in self.documents]
                    self.bm25 = BM25Okapi(tokenized_docs)
            except Exception as e:
                print(f"Error reading database for session {session_id}: {e}")
        else:
            print(f"No database found for session {session_id}.")

    def reload(self, session_id: str):
        """Forces reload of the current session's data"""
        self.current_session_id = None # Force reload
        self.load_session(session_id)

    def search(self, query: str, top_k: int = 3):
        if not self.documents or not self.bm25 or not self.vector_store:
            return []
        
        # 1. Semantic Search (Dense)
        dense_results = self.vector_store.similarity_search(query, k=top_k)
        dense_docs = [doc.page_content for doc in dense_results]

        # 2. Keyword Search (Sparse)
        tokenized_query = query.lower().split()
        sparse_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)

        # Combine and remove duplicates
        combined = list(set(dense_docs + sparse_docs))
        return combined[:top_k]