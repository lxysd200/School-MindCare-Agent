
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END, START

from llm_agent import LLMAgent
from memory import MemoryManager
from PromptTemplates import PromptTemplates

HIGH_RISK_WORDS = ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]
CONSULT_WORDS = ["焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询", "anxious", "depress", "stress"]
AMBIGUOUS_CONSULT_WORDS = ["好难", "太难了", "好累", "好烦", "不想学", "学不动", "撑不住", "顶不住", "做不完"]
MAX_LOOP_COUNT = 8


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
class AgentContext:
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
    knowledge_query: str = ""
    model_history: list[AiMessage] = field(default_factory=list)
    response_messages: list[AiMessage] = field(default_factory=list)
    response_agent: str = ""
    response_plan: str = ""
    steps: list[AgentStep] = field(default_factory=list)

class GraphState(TypedDict):
    context: AgentContext


def contains_keyword(text: str, keywords: list[str]) -> bool:
    lowered_text = text.casefold()
    return any(keyword.casefold() in lowered_text for keyword in keywords)


def history_indicates_consult(history: list[AiMessage]) -> bool:
    return any(
        contains_keyword(message.content, CONSULT_WORDS)
        for message in history
    )


def normalize_intent(raw_intent: str) -> IntentType:
    cleaned_intent = raw_intent.strip().upper()
    for intent in IntentType:
        if intent.value in cleaned_intent:
            return intent
    raise ValueError(f"无法识别的意图分类结果: {raw_intent}")


def classify_intent_with_model(context: AgentContext) -> tuple[IntentType, str]:
    llm = LLMAgent.from_env("supervisor")
    prompt_messages = PromptTemplates.intent_prompt(
        history=context.model_history,
        user_input=context.original_input,
    )
    raw_result = llm.chat(prompt_messages)
    intent = normalize_intent(raw_result)
    if (
        intent == IntentType.RISK
        and not contains_keyword(context.original_input, HIGH_RISK_WORDS)
        and history_indicates_consult(context.model_history)
    ):
        return IntentType.CONSULT, f"{raw_result} -> CONSULT(最近上下文为咨询场景，且当前输入无明确高风险信号)"
    return intent, raw_result

def Controller(state: GraphState) -> GraphState:
    context = state["context"]
    context.loop_count += 1
    if context.loop_count >= MAX_LOOP_COUNT and not context.finished:
        context.finished = True
        if not context.response_messages:
            context.response_messages.append(
                AiMessage(role="assistant", content="已达到最大循环次数，流程已终止。")
            )
    return state


def route_next(state: GraphState) -> str:
    context = state["context"]
    if context.finished:
        return "end"
    if not context.memory_loaded:
        return "memory"

    if not context.intent_routed:
        return "supervisor"

    if context.intent == IntentType.CHAT:
        return "companion" if not context.response_planned else "end"

    if not context.knowledge_handled:
        return "knowledge"

    if not context.risk_assessed:
        return "risk_guardian"

    if not context.response_planned:
        return "counselor"

    return "end"

def Memory_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    memory_manager = MemoryManager(
        memory_source_path=Path(__file__).with_name("memory_test.txt")
    )
    memory_result = memory_manager.load_for_model()

    context.model_history = [
        AiMessage(role=message.role, content=message.content)
        for message in memory_result.model_history
    ]
    context.memory_brief = memory_result.memory_brief
    context.memory_loaded = True
    print("加载记忆")
    context.steps.append(
        AgentStep(
            step=len(context.steps) + 1,
            agent="Memory",
            action="加载记忆",
            observation="memory_brief: "+context.memory_brief+"\n"+"model_history: "+str(context.model_history),
        )
    )
    return state

def Supervisor_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    if context.intent_routed:
        return state

    print("路由意图")
    # 先用显式关键词快速路由，都不命中时再交给模型识别。
    if contains_keyword(context.original_input, HIGH_RISK_WORDS):
        context.intent = IntentType.RISK
        observation = "命中高风险关键词，用户意图为 RISK。"
    elif contains_keyword(context.original_input, CONSULT_WORDS):
        context.intent = IntentType.CONSULT
        observation = "命中心理咨询关键词，用户意图为 CONSULT。"
    elif (
        history_indicates_consult(context.model_history)
        and contains_keyword(context.original_input, AMBIGUOUS_CONSULT_WORDS)
    ):
        context.intent = IntentType.CONSULT
        observation = "当前输入较短且语义模糊，但最近上下文持续呈现焦虑/失眠/痛苦等咨询信号，因此优先判定为 CONSULT。"
    else:
        try:
            context.intent, model_output = classify_intent_with_model(context)
            observation = f"关键词未命中，调用意图识别模型，模型输出：{model_output}，最终意图为 {context.intent.value}。"
        except Exception as exc:
            context.intent = IntentType.CHAT
            observation = f"关键词未命中，模型识别失败，默认回退为 CHAT。错误信息：{exc}"
    context.steps.append(
        AgentStep(
            step=len(context.steps) + 1,
            agent="Supervisor",
            action="路由意图",
            observation=observation,
        )
    )
    context.intent_routed = True
    return state

def Companion_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.response_agent = "Companion"
    context.response_plan = "与用户进行聊天,提供支持和建议。"
    context.response_planned = True
    context.response_messages.append(
        AiMessage(role="assistant", content="我在这儿，可以先跟我说说现在最想聊的是什么。")
    )
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Companion", action="与用户进行聊天", observation="闲聊来了。"))
    return state

def Knowledge_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.knowledge_handled = True
    print("知识库调用处理知识")
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Knowledge", action="调用知识库", observation="用户想聊的是什么。"))
    return state

def Risk_Guardian_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.risk_assessed = True
    print("风险评估")
    # 风险评估,如果原文中包含高风险关键词,则风险等级为HIGH,否则为LOW
    if contains_keyword(context.original_input, HIGH_RISK_WORDS):
        context.risk_level = RiskLevel.HIGH
    else:
        context.risk_level = RiskLevel.LOW
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Risk_Guardian", action="风险评估", observation=f"用户风险等级为{context.risk_level}"))
    return state

def Counselor_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.response_agent = "Counselor"
    context.response_plan = "根据风险等级,为用户提供提供支持和建议。"
    context.response_planned = True
    print("生成回复")
    # 生成回复,根据风险等级,为用户提供提供支持和建议
    if context.risk_level == RiskLevel.HIGH:
        context.response_messages.append(AiMessage(role="assistant", content="您 risk level is high, please seek help from a professional."))
    else:
        context.response_messages.append(AiMessage(role="assistant", content="You risk level is low, you can continue your life."))
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Counselor", action="生成回复", observation=context.response_messages[-1].content))
    return state

def End_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.finished = True
    return state

#创建LangGraph
workflow = StateGraph(GraphState)

#添加节点
workflow.add_node("controller", Controller)
workflow.add_node("memory", Memory_Agent)
workflow.add_node("supervisor", Supervisor_Agent)
workflow.add_node("companion", Companion_Agent)
workflow.add_node("knowledge", Knowledge_Agent)
workflow.add_node("risk_guardian", Risk_Guardian_Agent)
workflow.add_node("counselor", Counselor_Agent)
workflow.add_node("end", End_Agent)

#添加边
workflow.add_edge(START,"controller")
workflow.add_conditional_edges(
    "controller",
    route_next,
    {
        "memory": "memory",
        "supervisor": "supervisor",
        "companion": "companion",
        "knowledge": "knowledge",
        "risk_guardian": "risk_guardian",
        "counselor": "counselor",
        "end": "end",
    },
)
workflow.add_edge("memory", "controller")
workflow.add_edge("supervisor", "controller")
workflow.add_edge("companion", "controller")
workflow.add_edge("knowledge", "controller")
workflow.add_edge("risk_guardian", "controller")
workflow.add_edge("counselor", "controller")
workflow.add_edge("end",END)

#编译
app =workflow.compile()

#添加测试
test_cases = [
    "我最近有点难受",
    "Python好难",
    "我不想活了"
]

#运行测试
for test_case in test_cases:
    print("=" * 60)
    print(f"用户输入: {test_case}")
    state = app.invoke({"context": AgentContext(original_input=test_case)})
    print(state["context"].response_messages)
    print(state["context"].steps)
    print("=" * 60)
    
