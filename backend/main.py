import os
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from starlette.responses import StreamingResponse
import asyncio

from src.agents import CoordinatorAgent

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Google API key not found. Please add it to your .env file.")

app = FastAPI(title="Stateful Agentic RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Session State Management (Simulated in-memory) ---
SESSION_STATUS: Dict[str, str] = {} # e.g., {"session_id": "processing" | "ready"}

def process_documents_task(session_id: str, file_data: List[Dict[str, Any]]):
    """The actual processing logic that will run in the background."""
    try:
        # A simple class to mimic UploadFile for our parser
        class TempFile:
            def __init__(self, content, name):
                self.file = content
                self.filename = name
        
        # Recreate file-like objects from the data read in the main thread
        temp_files = [TempFile(item["content"], item["filename"]) for item in file_data]

        coordinator = CoordinatorAgent(session_id=session_id)
        coordinator.process_documents(temp_files)
        SESSION_STATUS[session_id] = "ready"
    except Exception as e:
        print(f"Error processing documents for session {session_id}: {e}")
        SESSION_STATUS[session_id] = "error"

# --- Pydantic Models for API validation ---
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    session_id: str
    query: str
    chat_history: List[Message] = Field(default_factory=list)

@app.post("/upload")
async def upload_documents(session_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Instantly returns and starts processing documents in the background.
    """
    # --- THIS IS THE KEY FIX ---
    # Read the file contents immediately in the main thread before the file handles are closed.
    # We will pass this data, not the file objects, to the background task.
    file_data_for_task = []
    for file in files:
        file_data_for_task.append({
            "filename": file.filename,
            "content": await file.read()  # Read the content now
        })
    # -------------------------

    SESSION_STATUS[session_id] = "processing"
    # Pass the pre-read data to the background task
    background_tasks.add_task(process_documents_task, session_id, file_data_for_task)
    
    return {"session_id": session_id, "message": "File upload received. Processing in the background."}

@app.get("/status/{session_id}")
def get_session_status(session_id: str):
    """Endpoint for the UI to poll the status of document processing."""
    status = SESSION_STATUS.get(session_id, "not_found")
    return {"session_id": session_id, "status": status}

@app.post("/query")
async def query_documents_stream(request: QueryRequest):
    """
    Endpoint to ask a question. This now supports streaming responses.
    """
    coordinator = CoordinatorAgent(session_id=request.session_id)
    history = [HumanMessage(content=msg.content) if msg.role.lower() == 'user' else AIMessage(content=msg.content) for msg in request.chat_history]

    async def stream_generator():
        try:
            async for chunk in coordinator.answer_query_stream(request.query, history):
                yield chunk
        except Exception as e:
            print(f"Error during stream: {e}")
            yield "Sorry, an error occurred while generating the response."

    return StreamingResponse(stream_generator(), media_type="text/plain")

@app.get("/session/new")
def create_new_session():
    """Endpoint to get a new, unique session ID."""
    return {"session_id": str(uuid.uuid4())}

@app.get("/")
def read_root():
    return {"status": "Stateful Agentic RAG API is running."}