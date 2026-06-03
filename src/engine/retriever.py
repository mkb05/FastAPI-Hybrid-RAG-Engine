import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
from typing import List

DB_DIR = "./chroma_db"

class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever...")
        
        # 1. Load Vector Database (Semantic Search)
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Chroma(
            persist_directory=DB_DIR, 
            embedding_function=self.embedding_model
        )
        
        # 2. Extract documents to build BM25 Index (Lexical/Keyword Search)
        # We pull the raw text back out of Chroma to build the keyword index in memory
        db_data = self.vector_store.get()
        self.documents = db_data['documents']
        
        if not self.documents:
            print("Warning: Database is empty. Please run ingestor.py first.")
            return
            
        # Tokenize documents for BM25 matching
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"Hybrid Retriever ready. Indexed {len(self.documents)} chunks.")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Performs Sparse-Dense Hybrid Search"""
        print(f"\n[System] Executing Hybrid Search for: '{query}'")
        
        # --- Stream 1: Semantic Vector Search ---
        # Finds concepts that *mean* the same thing, even if words differ
        vector_results = self.vector_store.similarity_search(query, k=top_k)
        vector_docs = [doc.page_content for doc in vector_results]
        
        # --- Stream 2: Lexical Keyword Search (BM25) ---
        # Finds exact string matches (crucial for serial numbers, code snippets, etc.)
        tokenized_query = query.lower().split()
        bm25_docs = self.bm25.get_top_n(tokenized_query, self.documents, n=top_k)
        
        # --- Stream 3: Result Fusion & Deduplication ---
        fused_results = []
        seen = set()
        
        # We interleave the results to balance exact keywords with semantic meaning
        for v_doc, b_doc in zip(vector_docs, bm25_docs):
            if v_doc not in seen:
                fused_results.append(v_doc)
                seen.add(v_doc)
            if b_doc not in seen:
                fused_results.append(b_doc)
                seen.add(b_doc)
                
        # Return only the most relevant Top K chunks to save LLM context window space
        return fused_results[:top_k]

# ==========================================
# Testing the Search Engine directly
# ==========================================
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    if retriever.documents:
        while True:
            user_query = input("\nEnter a search query (or 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break
                
            results = retriever.search(user_query, top_k=2)
            
            print("\n--- HYBRID SEARCH RESULTS ---")
            for i, res in enumerate(results, 1):
                print(f"\n[Result {i}]:\n{res}")
            print("-----------------------------")