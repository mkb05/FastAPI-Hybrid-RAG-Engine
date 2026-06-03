from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add the engine folder to the system path so we can import your RAG Agent
current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(current_dir, '..', 'engine')
sys.path.append(engine_dir)

from agent import RAGAgent

# Initialize the web server
app = FastAPI(title="Enterprise RAG API", version="1.0.0")

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

# --- ADD THIS NEW CORS SECTION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your local HTML file to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------

# Initialize your AI Agent globally when the server starts
print("Booting up backend services...")
agent = RAGAgent()

# Define the data structure we expect from the frontend
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "Enterprise RAG API is live and running."}

# The main Hybrid Search + LLM Endpoint
@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        print(f"\n[API] Received web request: '{request.question}'")
        # Pass the question to the AI you just built
        ai_response = agent.ask(request.question)
        return {"answer": ai_response}
    except Exception as e:
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")