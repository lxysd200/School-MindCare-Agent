
from dataclasses import dataclass, field
from enum import Enum
from Entities.entities import UserAccount, ChatSession

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class IntentType(str, Enum):
    CHAT = "CHAT"
    CONSULT = "CONSULT"
    RISK = "RISK"


@dataclass
class AgentStep:
    step: int
    agent: str
    action: str
    observation: str


@dataclass
class AiMessage:
    role: str
    content: str

@dataclass
class SearchResult:
    chunk_id: int | None
    source: str
    content: str
    score: float

class EmotionLabel(str, Enum):
    NORMAL = "NORMAL"
    ANXIETY = "ANXIETY"
    DEPRESSED = "DEPRESSED"
    HIGH_RISK = "HIGH_RISK"

@dataclass
class PsychologyAssessment:
    emotion: EmotionLabel
    emotion_score: float
    risk: RiskLevel
    confidence: float
    summary: str

@dataclass
class AgentContext:
    user: UserAccount
    session: ChatSession
    original_input: str
    model_input: str = ""
    loop_count: int = 0
    memory_loaded: bool = False
    intent_routed: bool = False
    knowledge_handled: bool = False
    risk_assessed: bool = False
    response_planned: bool = False
    finished: bool = False
    memory_brief: str = "无相关历史记忆。"
    intent: IntentType | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    assessment: PsychologyAssessment | None = None
    knowledge_query: str = ""
    rag_support: list[SearchResult] = field(default_factory=list)
    model_history: list[AiMessage] = field(default_factory=list)
    response_messages: list[AiMessage] = field(default_factory=list)
    response_agent: str = ""
    response_plan: str = ""
    steps: list[AgentStep] = field(default_factory=list)



@dataclass
class RiskAssessment:
    emotion: str
    emotionScore: float
    risk: str
    confidence: float
    summary: str

@dataclass
class AgentRunResult:
    intent: IntentType
    risk_level: RiskLevel
    assessment: PsychologyAssessment | None
    retrieved_knowledge: list[SearchResult]
    response_messages: list[AiMessage]
    steps: list[AgentStep]