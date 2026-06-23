FastAPI Hybrid RAG Engine (with Multi-Agent CRAG)

A production-ready, full-stack Corrective Retrieval-Augmented Generation (CRAG) application built with LangGraph, FastAPI, and Llama 3.2.

This system features a local multi-agent architecture that aggressively prevents AI hallucination, provides "glass-box" transparency into its reasoning, and enforces strict data isolation per chat session.

🚀 Key Features

Multi-Agent Orchestration (LangGraph): Replaces a single naive LLM prompt with a deterministic, state-driven pipeline of specialized AI agents.

Anti-Hallucination Guardrails: An LLM-as-a-judge "Grader" agent actively evaluates retrieved documents. If the context is irrelevant, it blocks the generation phase entirely.

Session-Strict Data Isolation: Each chat session gets its own isolated vector database workspace. Documents uploaded in Session A are mathematically invisible to Session B.

Hybrid Retrieval System: Combines Semantic Vector Search (ChromaDB + Nomic embeddings) with Lexical Search (BM25) to ensure exact keyword matches (like names or IDs) and conceptual matches are ranked accurately.

Persistent Memory: Uses LangGraph Checkpointers and SQLite to save conversation histories, allowing users to seamlessly resume past threads via a ChatGPT-style sidebar.

Glass-Box UI: The frontend exposes the agentic routing decisions and the exact source chunks used for generation directly in the chat interface.

🧠 System Architecture

1. The Multi-Agent Pipeline (CRAG)

Unlike standard RAG, which blindly trusts vector search results and passes them directly to an LLM, this engine utilizes a StateGraph workflow to validate data before it ever reaches the user. This multi-agent approach treats answer generation as a rigorous, verifiable pipeline rather than a single prompt.

graph TD
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef agent fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px;
    classDef db fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef decision fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px;

    User[👤 User Query]:::user --> QR[🤖 Query Rewriter Agent]:::agent
    QR -->|Optimized Keywords| HR[(🔍 Hybrid Retriever<br/>ChromaDB + BM25)]:::db
    HR -->|Top-K Chunks| Grader[⚖️ Document Grader Agent]:::agent
    Grader --> Decision{Is Context<br/>Relevant?}:::decision
    Decision -->|Yes| Synth[✍️ Synthesizer Agent]:::agent
    Decision -->|No| Reject[🛑 Safety Protocol<br/>Fallback Triggered]:::error
    Synth -->|Grounded Answer| Output[💬 UI Response]:::user
    Reject --> Output


Agent 1: The Query Rewriter: Analyzes the active conversation history and translates human-like conversational references (e.g., "What was his name in that file?") into highly optimized, searchable keywords stripped of conversational baggage. This drastically reduces semantic retrieval errors.

Agent 2: The Hybrid Retriever: Mounts the session-specific database and fetches the top-K document chunks using ensemble retrieval—combining dense semantic vectors for contextual understanding and sparse BM25 arrays for exact keyword matching.

Agent 3: The Document Grader (LLM-as-a-Judge): Acts as a deterministic gatekeeper. Operating at temperature=0, it cross-references the retrieved chunks against the original optimized query. If the context lacks relevance, it halts the graph and returns a safe fallback, effectively achieving a zero-hallucination guarantee.

Agent 4: The Synthesizer: Executes strictly upon Grader approval. It synthesizes the final response using only the validated context while maintaining conversational flow.

2. Data Ingestion & Isolation

Parsing: Uses PyPDFLoader and standard text loaders to extract raw text.

Chunking: Text is split using RecursiveCharacterTextSplitter (500-char chunks, 100-char overlap) to provide optimal context windows for the Grader.

Isolation: Vectors are saved to dynamic paths (./chroma_db/{session_id}/). The API passes the active session_id from the frontend, ensuring the Retriever only mounts the database belonging to the current chat workspace.

3. Backend Orchestration (FastAPI)

FastAPI serves as the asynchronous bridge between the Vanilla JS frontend and the LangGraph engine. It manages:

Endpoints: /api/v1/query, /api/v1/upload, /api/v1/sessions

Session State: Manages a custom SQLite chat_sessions table alongside LangGraph's native checkpoints tables to power the history sidebar.

Hard Deletion: Provides endpoints to completely scrub session data and vectors from the server to prevent orphaned data bloat.

🛠️ Tech Stack

LLM Engine: Ollama (Llama 3.2: 3B parameters)

Embeddings: Nomic-Embed-Text

Orchestration: LangGraph & LangChain

Vector Store: ChromaDB

Retrieval: Rank-BM25 & Langchain-Chroma

Backend API: FastAPI & Uvicorn

Database: SQLite (Checkpointers & Session Management)

Frontend: Vanilla JavaScript, HTML, CSS

⚙️ Setup & Installation

Prerequisites

Install Ollama and ensure it is running locally.

Pull the required local models:

ollama run llama3.2
ollama pull nomic-embed-text


Installation

Clone the repository:

git clone https://github.com/mkb05/FastAPI-Hybrid-RAG-Engine.git
cd FastAPI-Hybrid-RAG-Engine


Create and activate a Python virtual environment:

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate


Install dependencies:

pip install fastapi uvicorn langchain-ollama langchain-chroma langgraph langgraph-checkpoint-sqlite pypdf rank_bm25 python-multipart


Running the Application

Start the FastAPI backend server:

uvicorn src.api.main:app --reload


Open index.html in your web browser.

Click + New Chat, upload a document via the sidebar, and start querying securely!

