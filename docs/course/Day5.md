# Day 5 - Prototype to Production：部署与多智能体编排

## 你今天要达成什么

- 把 Agent 封装成 **FastAPI REST API**，让其他系统可以通过 HTTP 调用
- 实现一个 **多智能体系统**：4 个专业 Agent 协作完成研究任务
- 理解 **A2A（Agent-to-Agent）协议**：Agent 之间的消息传递格式
- 了解从本地原型到生产部署的关键步骤

## 知识要点（必看）

### 从原型到生产的关键步骤
```
本地脚本 (Day 1-4)
    ↓
REST API 封装 (FastAPI)
    ↓
容器化 (Dockerfile)
    ↓
云部署 (AWS/GCP/Azure)
    ↓
监控 & 扩缩容
```

### 多智能体编排 vs 单 Agent
| 方面 | 单 Agent (Day 4) | 多 Agent (Day 5) |
|------|-------------------|-------------------|
| 架构 | 一个 Agent 做所有事 | 4 个专业 Agent 分工 |
| 职责 | 混合在一起 | 清晰分离 |
| 可扩展性 | 改一处影响全局 | 独立替换单个 Agent |
| 调试 | 只看一条链路 | 看每个 Agent 的输入/输出 |

### A2A（Agent-to-Agent）协议
受 Google Agent2Agent Protocol 启发，我们定义了一套轻量级消息格式：

```python
{
    "sender": "planner",       # 发送方
    "receiver": "researcher",  # 接收方
    "intent": "delegate",      # 意图：delegate / feedback / complete
    "payload": { ... },        # 消息内容
    "parent_id": "xxx",        # 回复哪条消息
    "timestamp": "2026-..."
}
```

### Day 5 的多智能体架构
```
Planner  →  Researcher  →  Writer  →  Reviewer
    ↑                                      │
    └──────────  (if FAIL)  ───────────────┘
```

- **Planner**：制定研究计划、搜索关键词
- **Researcher**：调用 ArXiv 搜索工具
- **Writer**：根据计划和搜索结果撰写报告
- **Reviewer**：用 rubric 评分，不合格就把反馈发回 Planner

### FastAPI 部署架构
```
Client (curl / 浏览器 / 前端)
    ↓  POST /research
FastAPI Server
    ↓
    ├── use_multi_agent=false → Day 4 单 Agent
    └── use_multi_agent=true  → Day 5 多 Agent
    ↓
ResearchResponse (report + score + trace)
```

## 你将修改/阅读的代码位置

- `src/academic_researcher/agents/a2a_protocol.py`：A2A 消息协议
- `src/academic_researcher/agents/multi_agent_graph.py`：多智能体 LangGraph 编排
- `src/academic_researcher/api/server.py`：FastAPI REST API
- `examples/run_day5.py`：Day 5 命令行入口（多智能体模式）

## Day 5 操作步骤（你需要手动执行）

### Step 1：确保环境激活

```powershell
conda activate academic-agent
cd academic-researcher-agent
```

### Step 2：运行多智能体模式（命令行）

```powershell
python examples\run_day5.py --topic "large language model alignment" --goal "Survey RLHF and DPO methods" --show-a2a --show-trace
```

你会看到：
- 4 个 Agent 依次执行
- A2A 消息日志
- 执行追踪（每个 Agent 耗时）

### Step 3：启动 FastAPI 服务器

```powershell
cd src
python -m uvicorn academic_researcher.api.server:app --host 0.0.0.0 --port 8000 --reload
```

服务器启动后访问 http://localhost:8000/docs 查看 Swagger 自动文档。

### Step 4：通过 API 调用（新开一个终端）

```powershell
# 健康检查
curl http://localhost:8000/health

# 单 Agent 模式
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" -d "{\"topic\": \"LLM alignment\", \"goal\": \"Survey RLHF methods\", \"user_id\": \"api_user\"}"

# 多 Agent 模式
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" -d "{\"topic\": \"LLM alignment\", \"goal\": \"Survey RLHF methods\", \"use_multi_agent\": true}"

# 查看可用 Agent 列表
curl http://localhost:8000/agents
```

## A2A 消息流示例

一次成功的多智能体运行会产生类似这样的 A2A 消息日志：

```
[  delegate] planner        → researcher     id=a1b2c3d4…
[  delegate] researcher     → writer         id=e5f6g7h8…
[  delegate] writer         → reviewer       id=i9j0k1l2…
[  complete] reviewer       → orchestrator   id=m3n4o5p6…
```

如果 Reviewer 打分不及格，会多一轮：

```
[  delegate] planner        → researcher     id=...
[  delegate] researcher     → writer         id=...
[  delegate] writer         → reviewer       id=...
[  feedback] reviewer       → planner        id=...  ← 反馈
[  delegate] planner        → researcher     id=...  ← 重做
[  delegate] researcher     → writer         id=...
[  delegate] writer         → reviewer       id=...
[  complete] reviewer       → orchestrator   id=...  ← 通过
```

## API 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/research` | 执行研究任务 |
| GET | `/agents` | 列出可用 Agent 配置 |
| GET | `/docs` | Swagger 自动文档 |

## 5 天课程总结

```
Day 1: Agent 基础     → 最小可运行 LangGraph（计划 → 报告）
Day 2: Tools & MCP    → ArXiv 搜索 + MCP 框架
Day 3: Memory         → SQLite 长期记忆 + 会话管理
Day 4: Quality        → 评估 rubric + 质量改进循环 + 可观测性
Day 5: Production     → FastAPI 部署 + 4-Agent 编排 + A2A 协议
```

## 后续学习建议

1. **容器化**：写一个 `Dockerfile`，把整个项目打包成 Docker 镜像
2. **云部署**：部署到 GCP Cloud Run / AWS Lambda / Azure Container Apps
3. **前端**：用 React/Next.js 搭一个简单的 Web UI
4. **更多工具**：接入 Semantic Scholar API、Google Scholar、PDF 解析等
5. **向量记忆**：把 SQLite 文本搜索升级为向量数据库（ChromaDB / Pinecone）
6. **投递准备**：把这个项目推到 GitHub，写好 README，作为求职作品集

## 小练习（建议你做）

1. **新增 Agent**：加一个 "Summarizer" Agent，在 Writer 之前做摘要
2. **A2A 持久化**：把 A2A 消息日志存到 SQLite
3. **API 认证**：给 FastAPI 加上简单的 API Key 认证