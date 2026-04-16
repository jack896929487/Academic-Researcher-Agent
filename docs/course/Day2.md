# Day 2 - Agent Tools & MCP Integration（工具与外部能力）

## 你今天要达成什么

- 你会让 Agent 具备"采取行动"的能力：不只是生成文字，还能调用外部工具
- 你会实现一个 **ArXiv 搜索工具**，让 Agent 能检索真实的学术论文
- 你会了解 **MCP（Model Context Protocol）** 的概念和集成方式
- 你会运行一个完整的"计划 → 搜索文献 → 写报告"流程

## 知识要点（必看）

### Agent Tools 的核心概念
- **工具（Tools）** 让 Agent 能与外部世界交互：搜索、计算、文件操作、API 调用等
- **LangChain Tools** 有标准接口：`name`、`description`、`args_schema`、`_run()` 方法
- **LangGraph 工具集成** 通过 `bind_tools()` 和 `ToolNode` 实现

### MCP（Model Context Protocol）
- **MCP 是什么**：一个标准协议，让 AI 应用能发现和使用外部工具/服务
- **为什么重要**：避免每个 Agent 都要重新实现相同的工具（文件系统、数据库、浏览器等）
- **langchain-mcp-adapters**：LangChain 官方的 MCP 集成包

### 今天的架构升级
```
Day 1: 计划 → 写报告
Day 2: 计划 → 搜索工具 → 写报告（基于真实数据）
```

## 你将修改/阅读的代码位置

- `src/academic_researcher/tools/arxiv_search.py`：ArXiv 搜索工具实现
- `src/academic_researcher/tools/mcp_tools.py`：MCP 集成示例（含占位符代码）
- `src/academic_researcher/graphs/day2_tools_agent.py`：带工具的 LangGraph 流程
- `examples/run_day2.py`：Day 2 命令行入口

## Day 2 操作步骤（你需要手动执行）

### Step 1：确保环境激活

```powershell
conda activate academic-agent
cd academic-researcher-agent
```

### Step 2：运行 Day 2

```powershell
python examples\run_day2.py --topic "transformer attention mechanisms" --goal "Find recent papers and write a literature review"
```

你应该能看到：
1. Agent 生成研究计划
2. Agent 调用 ArXiv 搜索工具，找到真实论文
3. Agent 基于搜索结果写出包含具体论文引用的报告

### Step 3：观察工具调用过程

注意输出中的工具调用部分，你会看到：
- 搜索查询的构造
- ArXiv API 的返回结果
- Agent 如何将搜索结果整合到最终报告中

## 关键差异：Day 1 vs Day 2

| 方面 | Day 1 | Day 2 |
|------|-------|-------|
| 数据来源 | 纯 LLM 知识 | 真实 ArXiv 论文 |
| 流程 | 计划 → 报告 | 计划 → 搜索 → 报告 |
| 引用 | 虚构/通用 | 具体论文标题、作者、摘要 |
| 可扩展性 | 有限 | 可加入更多工具 |

## MCP 集成说明

当前实现包含 MCP 的"占位符代码"：
- `get_mcp_tools()` 会检查 `.env` 中的 `MCP_SERVERS` 配置
- 如果没有配置，会使用 `MockMCPTool` 作为演示
- 真实的 MCP 集成需要：
  1. 启动 MCP 服务器（如文件系统、数据库、浏览器工具）
  2. 在 `.env` 中配置服务器 URL
  3. 取消注释 `mcp_tools.py` 中的真实集成代码

## 小练习（建议你做）

1. **修改搜索策略**：改写 `node_search_literature()` 让它搜索 2-3 个不同的关键词
2. **添加新工具**：在 `tools/` 下创建一个新工具（如计算器、天气查询）
3. **优化报告格式**：修改 `node_write_report()` 的 prompt，让报告包含更多结构化信息

## 故障排除

如果遇到网络问题：
- ArXiv API 可能偶尔超时，这是正常的
- 可以修改 `arxiv_search.py` 中的 `max_results` 参数来减少请求量

如果工具调用失败：
- 检查 `llm.py` 中的模型配置是否支持 function calling
- Gemini 和 GPT-4 都支持工具调用，但 GPT-3.5 可能有限制