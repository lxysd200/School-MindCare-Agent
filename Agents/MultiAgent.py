
from typing import TypedDict
from langgraph.graph import StateGraph, END, START

from Agents.llm_agent import LLMAgent
from service.memory import MemoryManager
from PromptTemplates import PromptTemplates
from service.konwledge import KnowledgeService
from service.chroma_store import get_chroma_manager
from service.db import get_db_session
from service.accessment import RiskAssessmentService
from Agent_Type.AgentContext import AgentContext, AgentStep, AiMessage, RiskLevel, IntentType,  EmotionLabel, PsychologyAssessment
from Entities.entities import UserAccount, ChatSession
from service.skill import SkillService



HIGH_RISK_WORDS = ["自杀", "自残", "不想活", "结束生命", "伤害自己", "轻生", "suicide", "kill myself", "self harm"]
CONSULT_WORDS = ["焦虑", "抑郁", "压力", "失眠", "难过", "崩溃", "痛苦", "无助", "心理", "咨询", "anxious", "depress", "stress","难受"]
AMBIGUOUS_CONSULT_WORDS = ["好难", "太难了", "好累", "好烦", "不想学", "学不动", "撑不住", "顶不住", "做不完"]
MAX_LOOP_COUNT = 8





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

def _rewrite_query(context: AgentContext) -> str:
    try:
        llm = LLMAgent.from_env()
        prompt_messages = [
            {
                "role": "system",
                "content": "你是 MindBridge 的 KnowledgeAgent。把学生输入改写成适合检索校园心理知识库的中文查询词，只输出查询词。"
            },
            {
                "role": "user",
                "content": f"记忆摘要：\n{context.memory_brief}\n\n当前输入：\n{context.original_input}"
            },
        ]
        knowledge_query = llm.chat(prompt_messages)
        return knowledge_query
    except Exception as e:
        return ""


        

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
        user_id=context.user.id,
        session_id=context.session.id,
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
    print("闲聊")
    context = state["context"]
    context.response_agent = "Companion"
    context.response_plan = "与用户进行聊天,提供支持和建议。"
    context.response_planned = True
    prompt = PromptTemplates.answer_system_prompt(
        context.intent,
        context.risk_level,
        context.rag_support,
        context.user.display_name,
        skill_context = ""
        )
    system_message = AiMessage(
        role=prompt["role"],
        content=prompt["content"],
    )
    context.response_messages.append(system_message)
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Companion", action="与用户进行聊天", observation="闲聊来了。"))
    return state

def Knowledge_Agent(state: GraphState) -> GraphState:
    print("知识库调用处理知识")
    context = state["context"]
    knowledge_query = _rewrite_query(context)
    print("改写情况如下："+knowledge_query+"\n")
    if knowledge_query == "":
        return state
    context.knowledge_query = knowledge_query
    manager = get_chroma_manager()
    try:
        db = get_db_session()
        service = KnowledgeService(db,manager)
    finally:
        db.close()
    results = service.retrieve(knowledge_query)
    context.rag_support = results
    context.knowledge_handled = True
    observation = ""
    for result in results:
        observation += f"{result.source}: {result.chunk_id}: {result.score}\n"
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Knowledge", action="调用知识库", observation=observation))
    return state

def Risk_Guardian_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    
    print("风险评估")
    # 风险评估,如果原文中包含高风险关键词,则风险等级为HIGH,否则为LOW
    if contains_keyword(context.original_input, HIGH_RISK_WORDS):
        context.risk_level = RiskLevel.HIGH
    elif context.intent == IntentType.RISK:
        context.risk_level = RiskLevel.HIGH
    else:
        accessment = RiskAssessmentService("risk_guardian")
        result = accessment._assess_risk_level(context)
        risk_level = result.risk
        context.risk_level = risk_level
    context.risk_assessed = True
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Risk_Guardian", action="风险评估", observation=f"用户风险等级为{context.risk_level.value}"))
    return state

def Counselor_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.response_agent = "Counselor"
    context.response_plan = "先共情，再给出具体支持步骤；高风险时优先安全。"
    context.response_planned = True
    print("生成回复")
    skill_service = SkillService(context)
    skills = skill_service.build_skill_context()
    skill_context = "\n\n".join(skills.values())
    prompt = PromptTemplates.answer_system_prompt(
        context.intent,
        context.risk_level,
        context.rag_support,
        context.user.display_name,
        skill_context = skill_context
        )
    system_message = AiMessage(
        role=prompt["role"],
        content=prompt["content"],
    )
    context.response_messages.append(system_message)
    observation = "Using skills: " + "\n\n".join(skills.keys())
    context.steps.append(AgentStep(step=len(context.steps) + 1, agent="Counselor", action="生成回复", observation=observation))
    return state

def End_Agent(state: GraphState) -> GraphState:
    context = state["context"]
    context.finished = True
    return state


def create_graph() -> StateGraph:
        
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
    return app

