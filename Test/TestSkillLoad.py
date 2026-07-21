from Agent_Type.AgentContext import AgentContext, RiskLevel, IntentType
from service.skill import SkillService
from Entities.entities import UserAccount, ChatSession


context = AgentContext(user=UserAccount(), session=ChatSession(), original_input="我睡不着")
context.risk_level = RiskLevel.MEDIUM
context.intent = IntentType.CONSULT
skill_service = SkillService(context)
print(skill_service.build_skill_context())
