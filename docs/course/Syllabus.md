# Academic Researcher Agent - 5-Day Syllabus

目标：帮助你用 `LangGraph / LangChain` 从零构建一个可扩展的“学术研究助手 Agent”，并为投递 agent 相关实习准备可展示的作品集（代码 + 评估 + 部署）。

## Day 1 - Introduction to Agents
- 理解：Agent 的“计划/执行/反思”与传统“一次性 LLM 调用”的区别
- 实现：最小可运行的 LangGraph 流程（生成研究计划 -> 写研究报告初稿）

## Day 2 - Agent Tools & MCP
- 理解：工具（Tools）让 Agent 能“采取行动”，而不是只会生成文字
- 实现：加入工具层（例如 ArXiv 搜索），并展示如何用 `langchain-mcp-adapters` 接入 MCP 工具

## Day 3 - Context Engineering: Sessions & Memory
- 理解：短期上下文 + 长期记忆（记住偏好/研究方向/已检索内容）
- 实现：使用会话状态 + SQLite 长期记忆，把历史研究痕迹注入到后续生成里

## Day 4 - Agent Quality
- 理解：可观测性（logging/tracing）+ 评估（自评/打分/改进循环）
- 实现：加入一个简单的评估器（rubric 打分）与 tracing 适配（LangSmith 可选）

## Day 5 - Prototype to Production
- 理解：从本地 demo 到可部署服务；多智能体编排（Planner/Researcher/Writer）
- 实现：FastAPI 部署 + A2A 风格的“Agent-to-Agent 协议”（课程实践产物）

