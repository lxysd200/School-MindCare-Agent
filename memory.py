from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

from llm_agent import LLMAgent, LLMConfig, has_llm_env

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None


RECENT_MESSAGE_COUNT = 8
MID_MEMORY_USER_COUNT = 4
MID_MEMORY_ASSISTANT_COUNT = 3
DEFAULT_MID_TOKEN_BUDGET = 400
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_LLM_TIMEOUT = 20.0

MEMORY_LINE_PATTERN = re.compile(r"^\s*\d+\.\s*(user|assistant)\s*:\s*(.+?)\s*$")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

load_dotenv()


@dataclass(frozen=True)
class MemoryMessage:
    role: str
    content: str


@dataclass(frozen=True)
class MemoryRecallResult:
    recent_messages: list[MemoryMessage]
    mid_summary: str
    model_history: list[MemoryMessage]
    memory_brief: str


@dataclass
class MemoryManager:
    memory_source_path: str | Path
    mid_token_budget: int = DEFAULT_MID_TOKEN_BUDGET
    recent_message_count: int = RECENT_MESSAGE_COUNT
    mid_user_count: int = MID_MEMORY_USER_COUNT
    mid_assistant_count: int = MID_MEMORY_ASSISTANT_COUNT
    brief_api_key: str = ""
    brief_base_url: str = ""
    brief_model: str = ""
    brief_timeout: float = DEFAULT_LLM_TIMEOUT
    brief_llm: LLMAgent | None = None

    def __post_init__(self) -> None:
        if self.brief_llm is None and has_llm_env("memory"):
            self.brief_llm = LLMAgent.from_env("memory")
            return

        if not self.brief_api_key:
            self.brief_api_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or ""
            )
        if not self.brief_base_url:
            self.brief_base_url = (
                os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
            )
        if not self.brief_model:
            self.brief_model = os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL

        if self.brief_llm is None:
            self.brief_llm = LLMAgent(
                LLMConfig(
                    backend="deepseek" if self.brief_base_url == DEFAULT_DEEPSEEK_BASE_URL else "openai",
                    model=self.brief_model,
                    api_key=self.brief_api_key,
                    base_url=self.brief_base_url,
                    timeout=self.brief_timeout,
                    temperature=float(
                        os.getenv("MEMORY_LLM_TEMPERATURE")
                        or os.getenv("LLM_TEMPERATURE")
                        or "0.3"
                    ),
                )
            )

    def load_for_model(self) -> MemoryRecallResult:
        messages = self._parse_memory_file()
        if not messages:
            return MemoryRecallResult(
                recent_messages=[],
                mid_summary="",
                model_history=[],
                memory_brief="无相关历史记忆。",
            )

        recent_messages = (
            messages[-self.recent_message_count :]
            if self.recent_message_count > 0
            else []
        )
        middle_messages = (
            messages[: -self.recent_message_count]
            if self.recent_message_count > 0
            else messages
        )

        middle_user_messages = self._select_messages_by_role(
            messages=middle_messages,
            role="user",
            limit=self.mid_user_count,
        )
        middle_assistant_messages = self._select_messages_by_role(
            messages=middle_messages,
            role="assistant",
            limit=self.mid_assistant_count,
        )

        mid_summary = self._build_mid_memory_summary(
            user_messages=middle_user_messages,
            assistant_messages=middle_assistant_messages,
            token_budget=self.mid_token_budget,
        )

        model_history: list[MemoryMessage] = []
        if mid_summary:
            model_history.append(MemoryMessage(role="system", content=mid_summary))
        model_history.extend(recent_messages)

        memory_brief = self._build_memory_brief(
            mid_summary=mid_summary,
            recent_messages=recent_messages,
        )

        return MemoryRecallResult(
            recent_messages=recent_messages,
            mid_summary=mid_summary,
            model_history=model_history,
            memory_brief=memory_brief,
        )

    def _parse_memory_file(self) -> list[MemoryMessage]:
        path = Path(self.memory_source_path)
        if not path.exists():
            return []

        messages: list[MemoryMessage] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            match = MEMORY_LINE_PATTERN.match(raw_line)
            if not match:
                continue
            role, content = match.groups()
            messages.append(MemoryMessage(role=role, content=content.strip()))
        return messages

    def _select_messages_by_role(
        self,
        messages: list[MemoryMessage],
        role: str,
        limit: int,
    ) -> list[str]:
        selected: list[str] = []
        for message in reversed(messages):
            if message.role != role:
                continue
            selected.append(message.content)
            if len(selected) >= limit:
                return list(reversed(selected))
        return list(reversed(selected))

    def _build_mid_memory_summary(
        self,
        user_messages: list[str],
        assistant_messages: list[str],
        token_budget: int,
    ) -> str:
        if token_budget <= 0 or not (user_messages or assistant_messages):
            return ""

        focus_budget = max(1, token_budget // 2)
        support_budget = max(1, token_budget - focus_budget)

        student_focus = self._join_items_with_budget(
            items=user_messages,
            token_budget=focus_budget,
            fallback="暂无",
        )
        given_support = self._join_items_with_budget(
            items=assistant_messages,
            token_budget=support_budget,
            fallback="暂无",
        )

        summary = f"学生近期关注：{student_focus}\n已给过的支持：{given_support}"
        return self._trim_text_to_budget(summary, token_budget)

    def _join_items_with_budget(
        self,
        items: list[str],
        token_budget: int,
        fallback: str,
    ) -> str:
        if token_budget <= 0:
            return fallback
        if not items:
            return fallback

        per_item_budget = max(8, token_budget // len(items))
        trimmed_items = [
            self._trim_text_to_budget(item, per_item_budget) for item in items
        ]
        joined = "；".join(item for item in trimmed_items if item)
        if not joined:
            return fallback
        return self._trim_text_to_budget(joined, token_budget)

    def _trim_text_to_budget(self, text: str, token_budget: int) -> str:
        clean_text = text.strip()
        if not clean_text or token_budget <= 0:
            return ""
        if self._estimate_tokens(clean_text) <= token_budget:
            return clean_text

        ellipsis = "..."
        ellipsis_cost = self._estimate_tokens(ellipsis)
        current_cost = 0.0
        kept_chars: list[str] = []

        for char in clean_text:
            char_cost = self._estimate_char_cost(char)
            if current_cost + char_cost + ellipsis_cost > token_budget:
                break
            kept_chars.append(char)
            current_cost += char_cost

        trimmed = "".join(kept_chars).rstrip(" ,，；。")
        if not trimmed:
            return ellipsis
        return f"{trimmed}{ellipsis}"

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        cost = 0.0
        for char in text:
            cost += self._estimate_char_cost(char)
        return max(1, int(cost + 0.999))

    def _estimate_char_cost(self, char: str) -> float:
        if CJK_PATTERN.match(char):
            return 1.0
        if char.isascii() and char.isalnum():
            return 0.25
        if char.isspace():
            return 0.1
        return 0.5

    def _build_memory_brief(
        self,
        mid_summary: str,
        recent_messages: list[MemoryMessage],
    ) -> str:
        llm_generated_brief = self._try_generate_llm_memory_brief(
            mid_summary=mid_summary,
            recent_messages=recent_messages,
        )
        if llm_generated_brief:
            return llm_generated_brief

        if mid_summary:
            return mid_summary

        if recent_messages:
            return f"已完整召回近期记忆 {len(recent_messages)} 条消息。"

        return "无相关历史记忆。"

    def _try_generate_llm_memory_brief(
        self,
        mid_summary: str,
        recent_messages: list[MemoryMessage],
    ) -> str:
        try:
            return self._generate_llm_memory_brief(
                mid_summary=mid_summary,
                recent_messages=recent_messages,
            )
        except Exception:
            return ""

    def _generate_llm_memory_brief(
        self,
        mid_summary: str,
        recent_messages: list[MemoryMessage],
    ) -> str:
        llm = self.brief_llm
        if llm is None:
            raise ValueError("LLM is not configured.")
        if llm.config.backend in {"openai", "deepseek"} and not llm.config.api_key:
            raise ValueError("OPENAI_API_KEY (or DEEPSEEK_API_KEY) is not configured.")

        recent_memory_text = "\n".join(
            f"{message.role}: {message.content}" for message in recent_messages
        )
        prompt = (
            "请根据下面的记忆信息，生成 1-3 条中文摘要。\n"
            "要求：\n"
            "1. 每条摘要单独一行。\n"
            "2. 不要编号，不要解释，不要重复原文。\n"
            "3. 优先概括学生当前关注点、情绪/症状变化、已验证有效的支持。\n"
            "4. 每条尽量简洁。\n\n"
            "5.不输出风险等级或诊断。\n\n"
            f"中期记忆摘要：\n{mid_summary or '无'}\n\n"
            f"近期记忆：\n{recent_memory_text or '无'}"
        )

        content = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个对话记忆压缩助手，擅长输出简洁、准确的中文摘要。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        normalized_brief = self._normalize_llm_memory_brief(content)
        if not normalized_brief:
            raise ValueError("LLM returned an empty memory brief.")
        return normalized_brief

    def _normalize_llm_memory_brief(self, content: str) -> str:
        lines: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•\d\.\)\s]+", "", line)
            if not line:
                continue
            lines.append(line)
            if len(lines) >= 3:
                break
        return "\n".join(lines)


def load_memory_for_model(
    file_path: str | Path,
    mid_token_budget: int = DEFAULT_MID_TOKEN_BUDGET,
    recent_message_count: int = RECENT_MESSAGE_COUNT,
    mid_user_count: int = MID_MEMORY_USER_COUNT,
    mid_assistant_count: int = MID_MEMORY_ASSISTANT_COUNT,
) -> MemoryRecallResult:
    manager = MemoryManager(
        memory_source_path=file_path,
        mid_token_budget=mid_token_budget,
        recent_message_count=recent_message_count,
        mid_user_count=mid_user_count,
        mid_assistant_count=mid_assistant_count,
    )
    return manager.load_for_model()


if __name__ == "__main__":
    memory_manager = MemoryManager(
        memory_source_path=Path(__file__).with_name("memory_test.txt")
    )
    result = memory_manager.load_for_model()

    print("=" * 60)
    print("memory_brief 结果：")
    print(result.memory_brief)
    print("=" * 60)
    print("中期记忆压缩结果：")
    print(result.mid_summary or "无中期记忆。")
    print("=" * 60)
    print("近期记忆压缩结果：")
    if not result.recent_messages:
        print("无近期记忆。")
    else:
        for index, message in enumerate(result.recent_messages, start=1):
            print(f"{index}. {message.role}: {message.content}")
    print("=" * 60)
