# Day 1 - Introduction to Agents（入门）

## 你今天要达成什么

- 你会运行一个 **LangGraph Agent**：它不会调用任何外部工具，只是基于 LLM 生成“研究计划 -> 研究报告初稿”
- 你会理解 Agentic 架构的关键概念：**状态（state）**、**节点（node）**、**有向图流程（graph）**、以及“可迭代”的执行方式

## 知识要点（必看）

- **Agent vs 单次 LLM 调用**
  - 单次调用：输入 -> 输出一次性完成
  - Agent：通常包含多步骤（plan/act/reflect），中间把信息沉淀到 state，再进入下一步
- **有状态（Stateful）执行**
  - LangGraph 的核心是：每个节点读取/更新 state，形成可控的数据流
- **为什么要用 LangGraph**
  - 你可以很自然地扩展：加入 tools、加入 memory、加入评估循环、加入多智能体协作

## 你将修改/阅读的代码位置

- `src/academic_researcher/graphs/day1_basic_agent.py`
  - `build_day1_graph()`：构建图
  - `node_build_plan()`：生成研究计划
  - `node_write_report()`：根据计划写报告
- `examples/run_day1.py`：命令行入口
- `docs/course/Day1.md`：本日学习文档

## Day 1 操作步骤（你需要手动执行）

> 我不会替你运行命令。你读完后告诉我“可以执行”，我再继续 Day 2 的内容。

### Step 1：准备 conda 环境

在 `academic-researcher-agent` 目录下执行（PowerShell）：

```powershell
conda create -n academic-agent python=3.10 -y
conda activate academic-agent
```

### Step 2：安装依赖

```powershell
pip install -r requirements.txt
```

### Step 3：配置环境变量

```powershell
Copy-Item .env.example .env
```

然后把 `.env` 里的 `OPENAI_API_KEY` 设置为你的 Key。

### Step 4：运行 Day 1

```powershell
python examples\run_day1.py --topic "graph neural networks for drug discovery" --goal "Write an academic-style report outline and proposed methodology"
```

你应该能看到 Agent 输出一个结构化的“报告文本”。

## 小练习（建议你做）

1. 改写 `node_build_plan()` 的 system prompt，让计划更贴近论文写作（例如：强制包含研究问题与研究假设）
2. 改写 `node_write_report()` 让报告输出包含“研究贡献（Contribution）”与“可行性验证（Feasibility）”

把你改完的 prompt 版本贴到自己的笔记里（后面写简历/面试题很有用）。

