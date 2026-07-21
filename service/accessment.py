import json
from Agents.llm_agent import LLMAgent

from dataclasses import dataclass
from enum import Enum
from PromptTemplates import PromptTemplates
from Agent_Type.AgentContext import AgentContext, EmotionLabel, PsychologyAssessment, RiskLevel




class RiskAssessmentService:
    def __init__(self,service_name: str):
        self.llm = LLMAgent.from_env(service_name)

    
    def _assess_risk_level(self, context:AgentContext) -> PsychologyAssessment:
        try:
            prompt_messages = PromptTemplates.psychology_prompt(
                history=context.model_history,
                user_input=context.original_input,
            )
            raw = self.llm.chat(prompt_messages)
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1] if start >= 0 and end > start else raw)
            emotion = EmotionLabel(data.get("emotion", "NORMAL").upper())
            score = float(data.get("emotionScore", score_for_emotion(emotion)))
            risk = RiskLevel(data.get("risk", risk_from_score(score).value).upper())
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.75))))
            score_risk = risk_from_score(score)
            if risk_order(score_risk) > risk_order(risk):
                risk = score_risk
            if emotion == EmotionLabel.HIGH_RISK:
                risk = RiskLevel.HIGH
            return PsychologyAssessment(emotion, score, risk, confidence, data.get("summary", "模型评估结果"))
        except Exception as e:
            return PsychologyAssessment(
                emotion=EmotionLabel.NORMAL,
                emotion_score=0.0,
                risk=RiskLevel.LOW,
                confidence=0.0,
                summary="风险评估失败，已降级为低风险默认值"
            )

def score_for_emotion(emotion: EmotionLabel) -> float:
    return {
        EmotionLabel.HIGH_RISK: 4.0,
        EmotionLabel.DEPRESSED: 3.0,
        EmotionLabel.ANXIETY: 2.0,
        EmotionLabel.NORMAL: 0.0,
    }[emotion]


def risk_from_score(score: float) -> RiskLevel:
    if score >= 4:
        return RiskLevel.HIGH
    if score >= 3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def risk_order(risk: RiskLevel) -> int:
    return {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}[risk]