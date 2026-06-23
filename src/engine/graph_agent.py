from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from retriever import HybridRetriever
import sqlite3
import os

# Fix: Use absolute paths to prevent Windows SQLite locking issues
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
DB_DIR = os.path.join(project_root, "data", "chat_history")
os.makedirs(DB_DIR, exist_ok=True)
db_path = os.path.join(DB_DIR, "checkpoints.sqlite")

conn = sqlite3.connect(db_path, check_same_thread=False)

class GraphState(TypedDict):
    question: str
    session_id: str # NEW: Add session_id to state
    chat_history: List[dict]
    documents: List[str]
    generation: str

class MultiAgentSystem:
    def __init__(self):
        print("Initializing LangGraph Multi-Agent System with Persistent Memory...")
        self.retriever = HybridRetriever()
        self.llm = ChatOllama(model="llama3.2", temperature=0)
        
        # Create a clean table for the frontend sidebar to read
        self.cursor = conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

        # Initialize the persistent memory saver
        self.memory = SqliteSaver(conn)

        workflow = StateGraph(GraphState)
        
        workflow.add_node("retriever_agent", self.retrieve)
        workflow.add_node("grader_agent", self.grade)
        workflow.add_node("synthesizer_agent", self.generate)

        workflow.set_entry_point("retriever_agent")
        workflow.add_edge("retriever_agent", "grader_agent")
        
        workflow.add_conditional_edges(
            "grader_agent",
            self.decide_to_generate,
            {
                "generate": "synthesizer_agent",
                "reject": END 
            }
        )
        workflow.add_edge("synthesizer_agent", END)

        self.app = workflow.compile(checkpointer=self.memory)

    def reload_knowledge_base(self, session_id: str):
        # Fix: Directly call load_session to prevent TypeError mismatches
        self.retriever.load_session(session_id)

    def retrieve(self, state: GraphState):
        print("\n[Agent 1: Retriever] Fetching documents from ChromaDB...")
        # Ensure the retriever has the correct session loaded
        self.retriever.load_session(state["session_id"]) 
        
        # --- NEW: Query Rewriter ---
        search_query = state["question"]
        history = state.get("chat_history", [])
        
        if history:
            print(" -> Rewriting query to remove conversational baggage...")
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]])
            rewrite_prompt = ChatPromptTemplate.from_template(
                """You are an expert search engine query generator. 
                Read the conversation history and the user's latest question. 
                Rewrite the question into a concise search query. 
                Strip away conversational phrases like "in that file", "uploaded document", or "tell me about".
                Do NOT answer the question. ONLY output the search keywords.
                
                History: {history}
                Question: {question}"""
            )
            chain = rewrite_prompt | self.llm
            search_query = chain.invoke({"history": history_text, "question": state["question"]}).content.strip()
            print(f" -> Optimized Search Query: '{search_query}'")
        # ---------------------------
        
        docs = self.retriever.search(search_query, top_k=3)
        return {"documents": docs}

    def grade(self, state: GraphState):
        print("[Agent 2: Grader] Evaluating if documents contain the answer...")
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            return {"documents": []}

        prompt = ChatPromptTemplate.from_template(
            """You are a relevance grader. 
            Read the context and the question. Does the context contain ANY keywords, concepts, or partial information related to the question?
            If it is even slightly related, answer with the exact word 'yes'. 
            If it is completely off-topic, answer with the exact word 'no'.
            
            Question: {question}
            Context: {context}"""
        )
        
        chain = prompt | self.llm
        context_str = "\n\n".join(documents)
        score = chain.invoke({"question": question, "context": context_str}).content.strip().lower()
        
        if "yes" in score:
            print(" -> Grader Decision: RELEVANT.")
            return {"documents": documents}
        else:
            print(" -> Grader Decision: IRRELEVANT.")
            return {"documents": []}

    def generate(self, state: GraphState):
        print("[Agent 3: Synthesizer] Writing final response with chat history...")
        
        history = state.get("chat_history", [])
        history_text = ""
        if history:
            history_text = "Previous Conversation:\n"
            for msg in history[-4:]: # Use last 4 messages to prevent context overflow
                history_text += f"{msg['role']}: {msg['content']}\n"
                
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful enterprise assistant. 
            Answer the user's question using ONLY the provided context.
            CRITICAL INSTRUCTION: If the user asks about "the file", "this document", or "the uploaded context", treat the "Current Context" below as that exact file. Do not say "the context doesn't mention a file".
            Use the Previous Conversation history if you need context for words like 'it' or 'they'.
            
            {history_text}
            
            Current Context: {context}
            Current Question: {question}"""
        )
        chain = prompt | self.llm
        context_str = "\n\n".join(state["documents"])
        
        response = chain.invoke({
            "question": state["question"], 
            "context": context_str,
            "history_text": history_text
        })
        return {"generation": response.content}

    def decide_to_generate(self, state: GraphState):
        if len(state["documents"]) == 0:
            return "reject"
        return "generate"

    def ask(self, question: str, session_id: str) -> dict:
        config = {"configurable": {"thread_id": session_id}}
        current_state = self.app.get_state(config)
        
        # Ensure the retriever is pointing to the correct database BEFORE searching
        self.retriever.load_session(session_id)
        
        history = []
        is_new_session = True 
        
        if current_state and current_state.values:
             history = current_state.values.get("chat_history", [])
             is_new_session = False
             
        history.append({"role": "User", "content": question})

        # Save session to custom table for the sidebar UI
        if is_new_session:
            title = question[:30] + "..." if len(question) > 30 else question
            self.cursor.execute(
                "INSERT OR REPLACE INTO chat_sessions (session_id, title) VALUES (?, ?)", 
                (session_id, title)
            )
            conn.commit()
        else:
            self.cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()

        # Pass session_id into the graph inputs
        inputs = {"question": question, "session_id": session_id, "chat_history": history}
        result = self.app.invoke(inputs, config=config)
        
        if not result.get("documents"):
            response_text = "Safety Protocol Triggered: I could not find relevant information."
            status = "Blocked by Grader Agent 🛑"
            sources = []
        else:
            response_text = result["generation"]
            status = "Approved by Grader Agent ✅"
            sources = result["documents"]
            
        history.append({"role": "AI", "content": response_text})
        self.app.update_state(config, {"chat_history": history})

        return {
            "answer": response_text,
            "status": status,
            "sources": sources
        }