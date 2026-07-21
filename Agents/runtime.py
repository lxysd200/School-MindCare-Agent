from Agent_Type.AgentContext import AgentContext, AgentRunResult
from Agents.MultiAgent import create_graph





class AgentRuntimeService:
    def __init__(self):
        pass
    def run(self, context: AgentContext) -> AgentRunResult:
        app = create_graph()
        state = app.invoke({"context": context})
        print(state["context"].steps)
        final_context = state["context"]
        return AgentRunResult(
            intent=final_context.intent,
            risk_level=final_context.risk_level,
            assessment=final_context.assessment,
            retrieved_knowledge=final_context.rag_support,
            response_messages=final_context.response_messages,
            steps=final_context.steps
        )