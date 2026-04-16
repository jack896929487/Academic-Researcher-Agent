# Academic Researcher Agent (5-Day Course)

这是一个面向 **Agent 初学者** 的 5 天学习项目：你会一步步实现一个“学术研究助手” Agent，并逐步覆盖：

- Day 1：Agent 基础（Agentic vs 传统 LLM）
- Day 2：Tools + MCP（Model Context Protocol）
- Day 3：Context Engineering（Sessions & Memory）
- Day 4：Agent 质量（评估、可观测性、日志/追踪）
- Day 5：从原型到生产（部署 + 多智能体编排）

本项目使用 **LangGraph / LangChain**（替代你提到的 adk 相关框架）。

## 你将如何学习（重要）

1. 我会把“每一步要改/要做什么”的说明写在 `docs/course/DayX.md`
2. 你自己决定何时执行命令
3. 我不会替你自动启动服务器或跑完整训练流程

## 准备

在本仓库根目录运行以下命令（Conda + PowerShell）：

```powershell
conda create -n academic-agent python=3.10 -y
conda activate academic-agent
pip install -r requirements.txt
Copy-Item .env.example .env
```

然后把 `.env` 里的模型和对应 Key 配置好：

- 若 `OPENAI_MODEL` 是 OpenAI 模型（如 `gpt-4o-mini`），使用 `OPENAI_API_KEY`
- 若 `OPENAI_MODEL` 是 Gemini 模型（如 `gemini-2.5-flash-lite`），使用 `GOOGLE_API_KEY`

兼容说明：如果你还没填 `GOOGLE_API_KEY`，代码也会尝试把 `OPENAI_API_KEY` 作为 Gemini Key 的回退值。

## 快速开始（Day 1）

查看 `docs/course/Day1.md`，其中包含：

- 关键知识点
- 你要运行的命令
- 你要看的代码位置

## 项目入口

- `examples/run_day1.py`：Day 1 演示可运行

后续 Day 2~Day 5 我会继续在同样的结构下补齐对应实现与示例。

