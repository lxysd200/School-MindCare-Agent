from Agents.llm_agent import LLMAgent
from schema.dtos import ChatStreamEvent, ChatRequest
from Agents.runtime import AgentRuntimeService
from Agent_Type.AgentContext import AgentContext, AgentRunResult, AiMessage
from Entities.entities import UserAccount, ChatSession, ChatMessage
from datetime import datetime
from fastapi import HTTPException
from service.db import get_db_session
from schema.dtos import ChatHistoryResponse, ConversationMessageResponse, ConversationResponse



class ChatService:
    def __init__(self):
        pass
    
    def _save_chat_message(self,request: ChatRequest,assistant_message: str,sessionId: str = None,user: UserAccount = None):
        try:
            db = get_db_session()
            db.add_all([
                ChatMessage(
                    user_id=user.id,
                    session_id=sessionId,
                    role="user",
                    content=request.message,
                    created_at=datetime.now()
                ),
                ChatMessage(
                    user_id=user.id,
                    session_id=sessionId,
                    role="assistant",
                    content=assistant_message,
                    created_at=datetime.now()
                )
            ])
            db.commit()
        except Exception as exc:
            db.rollback()
            raise exc
        finally:
            db.close()

    def to_sse(self, event: ChatStreamEvent) -> str:
        return f"data: {event.model_dump_json()}\n\n"

    def streamChat(self, request: ChatRequest):
        llm_agent = LLMAgent.from_env("response")
        agent_runtime_service = AgentRuntimeService()
        user = UserAccount(id=1, username="test_user",display_name="test_user")
        try:
            db = get_db_session()
            session = db.query(ChatSession).filter(ChatSession.public_id == request.sessionId,ChatSession.user_id == user.id).first()
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
        finally:
            db.close()
        sessionId = session.id
        context = AgentContext(user=user, session=session, original_input=request.message)
        result = agent_runtime_service.run(context)
        messages = result.response_messages
        messages.append(AiMessage(role="user", content=request.message))
        full_text_parts: list[str] = []
        try:
            yield self.to_sse(ChatStreamEvent(
                type="start",
                sessionId=request.sessionId,
            ))

            for delta in llm_agent.stream_chat(messages):
                full_text_parts.append(delta)
                yield self.to_sse(ChatStreamEvent(
                    type="delta",
                    sessionId=request.sessionId,
                    content=delta,
                ))
            full_message = "".join(full_text_parts).strip()
            self._save_chat_message(request,full_message,sessionId,user)
            yield self.to_sse(ChatStreamEvent(
                type="end",
                sessionId=request.sessionId,
                message=full_message,
            ))
        except Exception as exc:
            yield self.to_sse(ChatStreamEvent(
                type="error",
                sessionId=request.sessionId,
                message=str(exc),
            ))
    def get_chat_history(self) -> list[ChatHistoryResponse]:
        try:
            db = get_db_session()
            user = UserAccount(id="1", username="test_user",display_name="test_user")
            sessions = db.query(ChatSession).filter(ChatSession.user_id == user.id).all()
            if sessions is None:
                raise HTTPException(status_code=404, detail="Chat not found")
            return [ChatHistoryResponse(
                 sessionId=session.id,
                 publicId=session.public_id,
                 title=session.title,
                 userId=session.user_id,
                 createdAt=session.created_at,
                 updatedAt=session.updated_at,
            ) for session in sessions]
        finally:
            db.close()

    def get_conversation(self,publicId: str) -> ConversationResponse:
        try:
            db = get_db_session()
            user = UserAccount(id="1", username="test_user",display_name="test_user")
            session = db.query(ChatSession).filter(ChatSession.public_id == publicId, ChatSession.user_id == user.id).first()
            if session is None:
                raise HTTPException(status_code=404, detail="Chat not found")
            messages = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).all()
            return ConversationResponse(
                sessionId=session.public_id,
                title=session.title,
                messages=[ConversationMessageResponse(
                    role=message.role,
                    content=message.content,
                    createdAt=message.created_at,
                ) for message in messages],
            )
        finally:
            db.close()
