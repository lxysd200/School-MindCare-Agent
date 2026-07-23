from __future__ import annotations

from pathlib import Path
from Agent_Type.AgentContext import RiskLevel, AgentContext, IntentType

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

class SkillService:
    def __init__(self,context: AgentContext):
        self.context = context
    
    def load_skill(self,skill_name: str) -> str:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        if not skill_file.exists():
            return ""
        return skill_file.read_text(encoding="utf-8").strip()
    
    def select_skill_names(self,user_input: str, risk_level: RiskLevel, intent: IntentType) -> list[str]:
        if intent == IntentType.CHAT:
            return []
        selected = ["supportive_response_baseline","referral_resource_guidance"]
        lowered = user_input.casefold()

        if risk_level == RiskLevel.HIGH:
            selected.append("high_risk_safety_plan")
            return selected

        if any(word in lowered for word in ["考试", "作业", "论文", "挂科", "学业", "学不动", "成绩"]):
            selected.append("academic_stress_planning")

        if any(word in lowered for word in ["焦虑", "心慌", "panic", "anxious"]):
            selected.append("anxiety_grounding_support")

        if any(word in lowered for word in ["失眠", "睡不着", "熬夜", "睡眠"]):
            selected.append("sleep_routine_support")

        return selected
    def build_skill_context(self) -> dict[str,str]:
        skill_names = self.select_skill_names(self.context.original_input, self.context.risk_level, self.context.intent)
        sections: dict[str,str] = {}
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                sections[name] = content

        return sections
