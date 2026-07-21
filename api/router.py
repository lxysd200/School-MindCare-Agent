from fastapi import APIRouter
from schema.dtos import ChatRequest
from service.chat import ChatService
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    service = ChatService()
    return StreamingResponse(service.streamChat(request), media_type="text/event-stream")

@router.get("/api/chat/history")
async def chat_history():
    service = ChatService()
    return service.get_chat_history()

@router.get("/api/chat/conversation/{publicId}")
async def chat_conversation(publicId: str):
    service = ChatService()
    return service.get_conversation(publicId)

    