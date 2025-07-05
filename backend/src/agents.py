import os
import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from .utils import parse_document, get_text_chunks
from .mcp import MCPMessage, create_mcp_message
# --- AGENT DEFINITIONS ---

class IngestionAgent:
    """
    Parses uploaded files, splits them into chunks, and creates a
    LangChain Document object for each chunk.
    """
    def handle_message(self, message: MCPMessage) -> MCPMessage:
        uploaded_files = message["payload"]["uploaded_files"]
        
        all_docs = []
        for file in uploaded_files:
            try:
                # 1. Get the full text from the document
                file_text = parse_document(file)
                
                # 2. Split the full text into chunks
                text_chunks = get_text_chunks(file_text)
                
                # 3. Create a Document for each chunk, adding metadata
                for i, chunk in enumerate(text_chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file.filename,
                            "chunk_index": i
                        }
                    )
                    all_docs.append(doc)

            except Exception as e:
                print(f"Error parsing {file.filename}: {e}")
        
        return create_mcp_message(
            sender="IngestionAgent",
            receiver="CoordinatorAgent",
            msg_type="INGESTION_COMPLETE",
            payload={"processed_documents": all_docs},
            trace_id=message["trace_id"]
        )
class ConversationalAgent:
    """
    This is a new, more powerful agent that combines Retrieval and LLM Response.
    It manages the entire conversational RAG chain, including memory.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Connect to a persistent ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        self.vector_store = Chroma(
            client=client,
            collection_name=self.session_id, # Each session gets its own collection
            embedding_function=self.embeddings,
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
        
        # This chain rephrases the user's question to be a standalone question
        self.history_aware_retriever = self._create_history_aware_retriever()
        
        # This chain takes the question and context and generates the final answer
        self.document_chain = self._create_document_chain()
        
        # The final retrieval chain that combines the two
        self.retrieval_chain = create_retrieval_chain(
            self.history_aware_retriever, self.document_chain
        )

    def _create_history_aware_retriever(self):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        return create_history_aware_retriever(self.llm, self.retriever, prompt)

    def _create_document_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the below context. If you don't know the answer, just say that you don't know, don't try to make up an answer:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)

    def add_documents(self, docs: list[Document]):
        """Adds documents to the persistent vector store for this session."""
        self.vector_store.add_documents(docs)

    async def answer_stream(self, user_input: str, chat_history: list) -> dict:
        """Generates a streaming answer."""
        # This is a new method for streaming
        async for chunk in self.retrieval_chain.astream({
            "chat_history": chat_history,
            "input": user_input
        }):
            # We are interested in the answer chunks
            if "answer" in chunk:
                yield chunk["answer"]

class CoordinatorAgent:
    """
    The coordinator is now much simpler. It mainly manages agent creation
    and delegates tasks.
    """
    def __init__(self, session_id: str):
        # The coordinator is now tied to a specific session
        self.session_id = session_id
        self.ingestion_agent = IngestionAgent()
        self.conversational_agent = ConversationalAgent(session_id)

    def process_documents(self, uploaded_files):
        # 1. Ingestion
        ingest_request = create_mcp_message("CoordinatorAgent", "IngestionAgent", "INGEST_REQUEST", {"uploaded_files": uploaded_files})
        ingest_response = self.ingestion_agent.handle_message(ingest_request)
        
        # 2. Add documents to the persistent store
        docs_to_add = ingest_response["payload"]["processed_documents"]
        if docs_to_add:
            self.conversational_agent.add_documents(docs_to_add)

    async def answer_query_stream(self, query: str, chat_history: list):
        # New method to handle the generator from the conversational agent
        async for chunk in self.conversational_agent.answer_stream(query, chat_history):
            yield chunk