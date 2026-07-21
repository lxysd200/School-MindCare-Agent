from Agents.MultiAgent import create_graph
from Agent_Type.AgentContext import AgentContext
from Entities.entities import UserAccount, ChatSession
from langgraph.graph import StateGraph



app = create_graph()

#添加测试
test_cases = [
    "我最近有点难受"
]

#运行测试
for test_case in test_cases:
    print("=" * 60)
    print(f"用户输入: {test_case}")
    state = app.invoke({"context": AgentContext(user=UserAccount(id="1", username="test_user",display_name="test_user"), session=ChatSession(id="1"), original_input=test_case)})
    print(state["context"].response_messages)
    print(state["context"].steps)
    print("=" * 60)
    