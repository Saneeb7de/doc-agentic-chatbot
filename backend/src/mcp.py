from typing import TypedDict, Literal, Any, List, Optional
import uuid

AgentName = Literal["CoordinatorAgent", "IngestionAgent", "RetrievalAgent", "LLMResponseAgent", "UI"]

class MCPPayload(TypedDict, total=False):
    """Flexible payload for different message types."""
    query: Optional[str]
    uploaded_files: Optional[List[Any]]
    text_chunks: Optional[List[str]]
    retrieved_context: Optional[List[str]]
    llm_response: Optional[str]
    error: Optional[str]

class MCPMessage(TypedDict):
    """The core Model Context Protocol message structure."""
    sender: AgentName
    receiver: AgentName
    type: str
    trace_id: str
    payload: MCPPayload

def create_mcp_message(sender: AgentName, receiver: AgentName, msg_type: str, payload: MCPPayload, trace_id: Optional[str] = None) -> MCPMessage:
    """Helper function to create a new MCP message with a unique trace_id."""
    return {
        "sender": sender,
        "receiver": receiver,
        "type": msg_type,
        "trace_id": trace_id or str(uuid.uuid4()),
        "payload": payload,
    }