from __future__ import annotations


def format_history(history: list[object]) -> str:
    if not history:
        return "无"

    lines: list[str] = []
    for message in history:
        if isinstance(message, dict):
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", ""))
        else:
            role = str(getattr(message, "role", "unknown"))
            content = str(getattr(message, "content", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class PromptTemplates:
    @staticmethod
    def intent_prompt(history: list[object], user_input: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是一个用户意图分类器，只做意图识别，不回答问题。"
                    "你需要根据用户当前输入和最近上下文,判断用户意图。"
                    "只输出 CHAT、CONSULT、RISK 之一。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最近上下文：\n{format_history(history)}\n\n"
                    f"当前输入：\n{user_input}"
                ),
            },
        ]

    @staticmethod
    def psychology_prompt(history: list[object], user_input: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你负责分析校园心理健康消息。只返回严格 JSON："
                    '{"emotion":"NORMAL|ANXIETY|DEPRESSED|HIGH_RISK","emotionScore":0.0,'
                    '"risk":"LOW|MEDIUM|HIGH","confidence":0.0,"summary":"short reason"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最近上下文：\n{format_history(history)}\n\n"
                    f"当前输入：\n{user_input}"
                ),
            },
        ]

    @staticmethod
    def answer_system_prompt(
        intent: object,
        risk: object,
        context: str,
        display_name: str,
        skill_context: str = "",
    ) -> dict[str, str]:
        intent_value = getattr(intent, "value", str(intent))
        risk_value = getattr(risk, "value", str(risk))

        if intent_value == "CHAT":
            content = (
                "你是 MindBridge，一个面向学生的日常陪伴与校园生活助手。"
                "普通学习、编程、校园事务和通用问题请自然、准确、直接地回答。"
                "不要主动做心理测评，不要输出风险等级、心理标签、诊断结论或报告口吻。"
                f"学生显示名：{display_name}"
            )
            return {"role": "system", "content": content}

        crisis_rule = ""
        if risk_value == "HIGH":
            crisis_rule = (
                "\n高风险处理规则：先回应情绪，再关注当前安全；鼓励用户立刻联系身边可信任的人、"
                "学校辅导员/心理中心或当地紧急救助；不提供任何危险操作细节。"
            )

        content = (
            "你是 MindBridge，一个面向学生的校园心理关怀智能体。"
            "回答要共情、谨慎、非评判，不诊断疾病，不开药，不替代持证心理咨询师。"
            "不要向学生输出风险等级、报告分数或后台标签。"
            "优先基于检索知识回答；知识不足时明确说明并给出安全通用建议。"
            f"\n学生显示名：{display_name}\n检索知识：\n{context}\n\n可用 skill 指引：\n{skill_context or '无'}{crisis_rule}"
        )
        return {"role": "system", "content": content}
