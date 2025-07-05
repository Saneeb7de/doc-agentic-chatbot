import streamlit as st
import requests
import uuid
import time
import os

# Configuration
# If running with Docker, this will be "http://backend:8000"
# If running locally, it's "http://127.0.0.1:8000"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Stateful Agentic RAG Chatbot", layout="wide")
st.title("🧠 Stateful & Conversational RAG Chatbot")
st.markdown("With Async Uploads, Streaming, and Persistent Sessions!")

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.is_processing = False
    st.session_state.is_ready = False

# --- Sidebar ---
with st.sidebar:
    st.header("Session Control")
    st.write(f"**Current Session ID:**")
    st.code(st.session_state.session_id)

    if st.button("Start New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.is_processing = False
        st.session_state.is_ready = False
        st.rerun()

    st.header("1. Upload Documents")
    st.markdown(f"Upload documents to session `{st.session_state.session_id}`.")
    
    uploaded_files = st.file_uploader(
        "Upload to current session",
        type=["pdf", "docx", "pptx", "txt", "csv"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.session_id}"
    )

    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Sending documents to backend for processing..."):
            files_to_send = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
            try:
                response = requests.post(f"{BACKEND_URL}/upload?session_id={st.session_state.session_id}", files=files_to_send)
                if response.status_code == 200:
                    st.session_state.is_processing = True
                    st.session_state.is_ready = False
                    st.success("✅ Documents sent! Processing in background.")
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")

# --- Polling Logic - THIS IS NOW CORRECTLY INDENTED ---
if st.session_state.get("is_processing", False):
    progress_bar = st.progress(0, "Backend is processing...")
    status = ""
    while status not in ["ready", "error"]:
        time.sleep(2)
        try:
            status_response = requests.get(f"{BACKEND_URL}/status/{st.session_state.session_id}")
            if status_response.status_code == 200:
                status = status_response.json()["status"]
                progress_bar.progress(0.5, f"Backend status: {status}...")
            else:
                status = "error"
        except requests.exceptions.RequestException:
            status = "error"
    
    if status == "ready":
        progress_bar.progress(1.0, "✅ Documents are ready to be queried!")
        st.session_state.is_ready = True
    else:
        progress_bar.empty()
        st.error("🚨 Failed to process documents.")
    st.session_state.is_processing = False

# --- Main Chat Interface ---
st.header("2. Chat with Your Documents")

if not st.session_state.is_ready:
    st.info("Please upload and process documents in the sidebar to begin.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a follow-up question..."):
    if not st.session_state.is_ready:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # --- THIS IS THE CORRECTED STREAMING IMPLEMENTATION ---
            try:
                response_stream = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "session_id": st.session_state.session_id,
                        "query": prompt,
                        "chat_history": [msg for msg in st.session_state.messages if msg['role'] in ['user', 'assistant']]
                    },
                    stream=True
                )
                response_stream.raise_for_status() # Raise an exception for bad status codes
                
                # Use st.write_stream to render the streaming response
                full_response = st.write_stream(response_stream.iter_content(chunk_size=None, decode_unicode=True))
                
                # Add the complete response to history after the stream finishes
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")