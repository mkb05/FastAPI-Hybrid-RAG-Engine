from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import shutil
import sqlite3

# Define absolute paths so Windows doesn't get confused
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
engine_dir = os.path.join(project_root, 'src', 'engine')
sys.path.append(engine_dir)

from graph_agent import MultiAgentSystem
from ingestor import ingest_documents

app = FastAPI(title="Enterprise RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up Multi-Agent Orchestrator...")
agent = MultiAgentSystem()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"

@app.post("/api/v1/query")
def query_documents(request: QueryRequest):
    try:
        result_data = agent.ask(request.question, request.session_id)
        return result_data
    except Exception as e:
        print(f"[API] Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")

@app.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...) # Receive session_id from frontend
):
    try:
        # Create a specific folder for this session's documents
        docs_path = os.path.join(project_root, 'data', 'docs', session_id)
        os.makedirs(docs_path, exist_ok=True)
        file_location = os.path.join(docs_path, file.filename)
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # Pass session_id to ingestor
        ingest_documents(specific_file=file_location, session_id=session_id)
        # Reload the knowledge base for this specific session
        agent.reload_knowledge_base(session_id)
        
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        import traceback
        traceback.print_exc()  # <-- This will print the REAL error to your terminal!
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# SIDEBAR DATABASE ENDPOINTS
# ==========================================

# Use an absolute path for SQLite to prevent "unable to open database" errors
db_path = os.path.join(project_root, 'data', 'chat_history', 'checkpoints.sqlite')

@app.get("/api/v1/sessions")
def get_all_sessions():
    """Returns a list of all chat sessions for the sidebar"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title FROM chat_sessions ORDER BY updated_at DESC")
        sessions = [{"id": row[0], "title": row[1]} for row in cursor.fetchall()]
        conn.close()
        return {"sessions": sessions}
    except Exception as e:
        print(f"Error fetching sessions: {e}")
        return {"sessions": []}

@app.get("/api/v1/session/{session_id}")
def get_session_history(session_id: str):
    """Returns the full chat history for a specific session"""
    try:
        config = {"configurable": {"thread_id": session_id}}
        state = agent.app.get_state(config)
        
        history = []
        if state and state.values:
            history = state.values.get("chat_history", [])
            
        return {"history": history}
    except Exception as e:
        return {"history": []}

@app.delete("/api/v1/session/{session_id}")
def delete_session(session_id: str):
    """Deletes a session completely from the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete from our custom sidebar table
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        # 2. Try to delete from LangGraph tables individually. 
        # If the table doesn't exist, it simply catches the error and moves on.
        for table in ["checkpoints", "checkpoint_writes"]:
            try:
                cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
            except sqlite3.OperationalError:
                print(f"[Warning] Table '{table}' not found in database. Skipping.")
        
        conn.commit()
        conn.close()
        return {"message": "Session deleted successfully"}
    except Exception as e:
        print(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session.")