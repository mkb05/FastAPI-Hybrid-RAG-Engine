from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
engine_dir = os.path.join(current_dir, '..', 'engine')
sys.path.append(engine_dir)

from agent import RAGAgent
from ingestor import ingest_documents

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting up backend services...")
agent = RAGAgent()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/api/v1/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        ai_response = agent.ask(request.question)
        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error during inference.")

# --- NEW FILE UPLOAD ENDPOINT ---
@app.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # 1. Save the file to our docs folder
        docs_path = os.path.join(current_dir, '..', '..', 'data', 'docs')
        os.makedirs(docs_path, exist_ok=True)
        file_location = os.path.join(docs_path, file.filename)
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # 2. Ingest the specific new file into ChromaDB
        ingest_documents(specific_file=file_location)
        
        # 3. Reload the Agent's memory so it can answer questions about it immediately
        agent.reload_knowledge_base()
        
        return {"message": f"Successfully processed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))