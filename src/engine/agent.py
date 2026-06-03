from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# We import the search engine you just built!
from retriever import HybridRetriever

class RAGAgent:
    def __init__(self):
        print("Initializing Llama 3.2 Agent...")
        # 1. Boot up your custom search engine
        self.retriever = HybridRetriever()
        
        # 2. Boot up the local LLM (Temperature 0.1 keeps it strictly factual)
        self.llm = ChatOllama(model="llama3.2", temperature=0.1)
        
        # 3. Create the System Prompt (The Rules for the AI)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert enterprise support AI. 
            Answer the user's question based ONLY on the provided context below. 
            If the answer is not in the context, say 'I cannot find the answer in the provided documentation.' 
            Do not guess or make up information.
            
            Context:
            {context}"""),
            ("human", "{question}")
        ])

    def ask(self, question: str):
        print(f"\n[Agent] Searching database for context...")
        
        # Step A: Get the Top 3 most relevant chunks from ChromaDB & BM25
        retrieved_chunks = self.retriever.search(question, top_k=3)
        
        # Combine the chunks into a single readable string
        context_text = "\n\n".join(retrieved_chunks)
        
        if not context_text.strip():
            return "I don't have any documentation available to answer that."

        # Step B: Pass the Context and the Question to Llama 3.2
        print("[Agent] Context found. Generating response via Llama 3.2...")
        chain = self.prompt_template | self.llm
        
        # Step C: Get the final answer
        response = chain.invoke({"context": context_text, "question": question})
        return response.content


    def reload_knowledge_base(self):
        """Tells the retriever to update its indexes"""
        self.retriever.reload()

# ==========================================
# Testing the Full RAG Pipeline
# ==========================================
if __name__ == "__main__":
    agent = RAGAgent()
    print("\n=== ENTERPRISE RAG AGENT READY ===")
    
    while True:
        user_input = input("\nAsk the documentation a question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
            
        answer = agent.ask(user_input)
        
        print("\n--------------------------------------------------")
        print(f"🤖 AI RESPONSE:\n{answer}")
        print("--------------------------------------------------")