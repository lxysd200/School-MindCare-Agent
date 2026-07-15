# MindBridge

面向学生的陪伴与校园心理关怀智能体原型。项目重点在于将多阶段对话流程工程化：意图路由、记忆召回、知识检索（规划中）、风险守护与回复生成，并支持多模型后端切换。

## 特性

- LangGraph 编排的多节点对话流程：Memory → Supervisor → Knowledge(规划) → Risk Guardian → Counselor/Companion
- 意图识别 Prompt 模板：只做分类（CHAT / CONSULT / RISK），不输出回答
- 对话记忆管理：从本地对话记录中召回近期消息 + 中期压缩摘要，并生成可注入模型的 history
- 多模型后端适配：OpenAI / DeepSeek / Ollama（统一 OpenAI SDK 调用方式）
- 面向“开发过程留痕”的仓库组织：文档、决策记录与里程碑版本可追溯

## 快速开始

### 1. 环境要求

- Python 3.10+（依赖 `mcp` 的版本约束）

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

复制示例文件并填写你的 Key：

```bash
copy .env.example .env
```

项目默认使用 DeepSeek（也可切换 OpenAI / Ollama）。示例变量见 [.env.example](./.env.example)。

### 4. 运行示例

当前仓库自带一个最小可运行的 LangGraph 流程演示脚本：

```bash
python MultiAgent.py
```

记忆模块可单独运行查看压缩结果：

```bash
python memory.py
```

## 项目结构（当前）

```text
.
├── llm_agent.py          # LLM 后端适配与统一调用
├── PromptTemplates.py    # Prompt 模板（意图识别/心理分析/回复系统提示词）
├── memory.py             # 记忆召回与压缩
├── memory_test.txt       # 示例对话记忆数据（可替换为你的数据源）
├── MultiAgent.py         # LangGraph 多节点编排原型
├── requirements.txt
├── .env.example
└── docs/
    ├── devlog.md         # 开发日志（过程留痕）
    └── decisions.md      # 关键决策记录（架构/技术选型）
```

## 设计原则与安全边界

- 本项目不提供医疗诊断、用药建议，不替代持证心理咨询师
- 面向学生的输出避免“报告口吻”，不向用户展示后台风险分级与标签
- 对高风险信号优先做安全引导：先回应情绪，再建议联系身边可信任的人/学校心理资源/紧急求助渠道

## 路线图（规划）

- RAG：检索增强生成（知识库构建、查询改写、证据注入、答案引用）
- MCP：工具调用与外部能力接入（数据源、校园服务、知识库工具）
- Backend：FastAPI 服务化、鉴权、会话存储、向量库持久化
- 评测：对话质量评测与安全评测（数据集、回归测试、自动化对比）

## 开发过程

- 开发日志：见 [docs/devlog.md](./docs/devlog.md)
- 决策记录：见 [docs/decisions.md](./docs/decisions.md)

## License

暂未选择许可证（默认 All rights reserved）。若你希望开源协作，可补充 MIT/Apache-2.0 等许可证。

