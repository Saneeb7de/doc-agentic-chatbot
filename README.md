# Agentic RAG Chatbot for Multi-Format Document QA

This project implements a sophisticated, agent-based Retrieval-Augmented Generation (RAG) chatbot. It features a decoupled frontend/backend architecture, persistent user sessions, conversational memory, and real-time streaming responses.

## ✨ Core Features

-   **Multi-Format Ingestion**: Supports `PDF`, `DOCX`, `PPTX`, `CSV`, and `TXT` documents.
-   **Agentic Architecture**: Uses a `CoordinatorAgent`, `IngestionAgent`, and a stateful `ConversationalAgent` to manage the workflow.
-   **Model Context Protocol (MCP)**: Demonstrates structured message passing between agents.
-   **Persistent Sessions**: Leverages **ChromaDB** to save vector stores for each user session, allowing users to return to their documents later.
-   **Conversational Memory**: The chatbot remembers previous turns of the conversation to answer follow-up questions accurately.
-   **Asynchronous Processing**: Document uploads are processed in the background for a non-blocking user experience.
-   **Streaming Responses**: Answers are streamed token-by-token, providing an interactive, real-time feel like modern LLM applications.
-   **View Source Context**: Users can view the exact document chunks the AI used to formulate its answer.

## 🏛️ System Architecture & Flow

The system is built with a decoupled FastAPI backend and a Streamlit frontend.

1.  **Upload**: The user uploads files via the UI. The request hits the FastAPI `/upload` endpoint, which starts a background task and stores the document embeddings in a session-specific ChromaDB collection.
2.  **Status Check**: The UI polls the `/status` endpoint until the documents are processed and ready.
3.  **Query**: The user asks a question. The UI sends the query and chat history to the `/query` endpoint.
4.  **RAG Chain**: The backend's `ConversationalAgent` uses a history-aware chain to rephrase the question, retrieve relevant chunks from ChromaDB, and generate an answer.
5.  **Stream & Context**: The answer is streamed back to the UI. The source context is saved on the backend and fetched by the UI after the stream completes.

 
*Note: You will need to create this diagram and upload it to a site like Imgur or add it to an `assets` folder in your repo.*

## 🛠️ Tech Stack

-   **Frontend**: Streamlit
-   **Backend**: FastAPI, Uvicorn
-   **AI / LLM Framework**: LangChain
-   **LLM & Embeddings**: Google Gemini API (`gemini-1.5-flash-latest`, `embedding-001`)
-   **Vector Database**: ChromaDB (Persistent)
-   **Agents & Protocol**: Custom Python classes with Model Context Protocol (MCP) principles.

## ⚙️ Local Setup and Installation

### Prerequisites
-   Python 3.9+
-   A Google Gemini API Key

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-chatbot-mcp.git
cd agentic-rag-chatbot-mcp
```

### 2. Set Up the Backend
```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Create a .env file from the example
cp .env.example .env

# Add your Google API key to the .env file
# GOOGLE_API_KEY="AIza..."
```

### 3. Set Up the Frontend
```bash
# From the root directory, navigate to the frontend directory
cd ../frontend

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run

You need to run the backend and frontend in **two separate terminals**.

**Terminal 1: Start the Backend**
```bash
# Make sure you are in the 'backend' directory
cd path/to/agentic-rag-chatbot-mcp/backend

# Run the FastAPI server
uvicorn main:app --reload
```
The backend will be running on `http://127.0.0.1:8000`.

**Terminal 2: Start the Frontend**
```bash
# Make sure you are in the 'frontend' directory
cd path/to/agentic-rag-chatbot-mcp/frontend

# Run the Streamlit app
streamlit run app.py
```
The application will open in your web browser at `http://localhost:8501`.
