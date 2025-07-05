import os
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from .mcp import MCPMessage, create_mcp_message
from .utils import parse_document, get_text_chunks

# --- IngestionAgent remains the same ---
class IngestionAgent:
    def handle_message(self, message: MCPMessage) -> MCPMessage:
        uploaded_files = message["payload"]["uploaded_files"]
        all_docs = []
        for file in uploaded_files:
            try:
                file_text = parse_document(file)
                text_chunks = get_text_chunks(file_text)
                for i, chunk in enumerate(text_chunks):
                    doc = Document(page_content=chunk, metadata={"source": file.filename, "chunk_index": i})
                    all_docs.append(doc)
            except Exception as e:
                print(f"Error parsing {file.filename}: {e}")
        return create_mcp_message(sender="IngestionAgent", receiver="CoordinatorAgent", msg_type="INGESTION_COMPLETE", payload={"processed_documents": all_docs}, trace_id=message["trace_id"])

class ConversationalAgent:
    """ This agent's implementation is now stable and correct. """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        client = chromadb.PersistentClient(path="./chroma_db")
        self.vector_store = Chroma(client=client, collection_name=self.session_id, embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
        history_aware_retriever = self._create_history_aware_retriever()
        document_chain = self._create_document_chain()
        self.retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    def _create_history_aware_retriever(self):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        return create_history_aware_retriever(self.llm, self.retriever, prompt)

    def _create_document_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def add_documents(self, docs: list[Document]):
        self.vector_store.add_documents(docs)

    def get_stream(self, user_input: str, chat_history: list):
        # This returns the raw LangChain stream generator, which is what we want.
        return self.retrieval_chain.astream({"chat_history": chat_history, "input": user_input})

class CoordinatorAgent:
    """ The coordinator now has the logic to separate the answer from the context. """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ingestion_agent = IngestionAgent()
        self.conversational_agent = ConversationalAgent(session_id)

    def process_documents(self, uploaded_files):
        ingest_request = create_mcp_message("CoordinatorAgent", "IngestionAgent", "INGEST_REQUEST", {"uploaded_files": uploaded_files})
        ingest_response = self.ingestion_agent.handle_message(ingest_request)
        docs_to_add = ingest_response["payload"]["processed_documents"]
        if docs_to_add:
            self.conversational_agent.add_documents(docs_to_add)

    async def answer_query_stream(self, query: str, chat_history: list):
        """
        Yields tuples of (type, data) where type is 'answer' or 'context'.
        This makes the output structured and easy to handle in the API layer.
        """
        final_context = []
        async for chunk in self.conversational_agent.get_stream(query, chat_history):
            if "answer" in chunk:
                # Yield the answer part for live streaming
                yield "answer", chunk["answer"]
            if "context" in chunk:
                # This part of the stream contains the source documents
                final_context = chunk["context"]

        # After the answer stream is finished, yield the complete context once
        yield "context", final_context