from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Literal

from pydantic import BaseModel, Field



class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    sessionId: Optional[str] = None

class ChatStreamEvent(BaseModel):
    sessionId: Optional[str] = None
    content: Optional[str] = None
    message: Optional[str] = None
    type: Literal["start", "delta", "end", "error"]



class ReportResponse(BaseModel):
    id: int
    sessionId: str
    username: str
    displayName: str
    content: str
    intent: str
    emotion: str
    emotionScore: float
    riskLevel: str
    confidence: float
    summary: str
    createdAt: datetime

class ChatHistoryResponse(BaseModel):
    sessionId: int
    publicId: str
    title:str
    userId: int
    createdAt: datetime
    updatedAt: datetime

class ConversationMessageResponse(BaseModel):
    role: str
    content: str
    createdAt: datetime


class ConversationResponse(BaseModel):
    sessionId: str
    title: str
    messages: list[ConversationMessageResponse]