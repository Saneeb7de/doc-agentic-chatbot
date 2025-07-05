import os
import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from starlette.responses import StreamingResponse

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

# --- Session State Management ---
SESSION_STATUS: Dict[str, str] = {}
# --- NEW: State for storing the last context for each session ---
SESSION_LAST_CONTEXT: Dict[str, List[Dict]] = {}

def process_documents_task(session_id: str, file_data: List[Dict[str, Any]]):
    try:
        class TempFile:
            def __init__(self, content, name):
                self.file = content
                self.filename = name
        temp_files = [TempFile(item["content"], item["filename"]) for item in file_data]
        coordinator = CoordinatorAgent(session_id=session_id)
        coordinator.process_documents(temp_files)
        SESSION_STATUS[session_id] = "ready"
    except Exception as e:
        print(f"Error processing documents for session {session_id}: {e}")
        SESSION_STATUS[session_id] = "error"

class Message(BaseModel):
    role: str; content: str
class QueryRequest(BaseModel):
    session_id: str; query: str; chat_history: List[Message] = Field(default_factory=list)

@app.post("/upload")
async def upload_documents(session_id: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    file_data_for_task = [{"filename": file.filename, "content": await file.read()} for file in files]
    SESSION_STATUS[session_id] = "processing"
    background_tasks.add_task(process_documents_task, session_id, file_data_for_task)
    return {"session_id": session_id, "message": "File upload received. Processing in the background."}

@app.get("/status/{session_id}")
def get_session_status(session_id: str):
    status = SESSION_STATUS.get(session_id, "not_found")
    return {"session_id": session_id, "status": status}

@app.post("/query")
async def query_documents_stream(request: QueryRequest):
    coordinator = CoordinatorAgent(session_id=request.session_id)
    history = [HumanMessage(content=msg.content) if msg.role.lower() == 'user' else AIMessage(content=msg.content) for msg in request.chat_history]

    async def stream_generator():
        final_context_docs = []
        try:
            async for stream_type, data in coordinator.answer_query_stream(request.query, history):
                if stream_type == "answer":
                    yield data
                elif stream_type == "context":
                    # This will be the final list of Document objects
                    final_context_docs = data
            
            # After the stream is done, serialize and store the final context
            SESSION_LAST_CONTEXT[request.session_id] = [
                {"page_content": doc.page_content, "metadata": doc.metadata} for doc in final_context_docs
            ]
        except Exception as e:
            print(f"Error during stream: {e}")
            yield "Sorry, an error occurred while generating the response."

    return StreamingResponse(stream_generator(), media_type="text/plain")

# --- NEW ENDPOINT TO GET THE CONTEXT ---
@app.get("/context/{session_id}")
def get_last_context(session_id: str):
    """Retrieves the source context used for the last query in a session."""
    context = SESSION_LAST_CONTEXT.get(session_id, [])
    return {"context": context}

@app.get("/session/new")
def create_new_session(): return {"session_id": str(uuid.uuid4())}

@app.get("/")
def read_root(): return {"status": "Stateful Agentic RAG API is running."}