# Day 3 - Context Engineering: Sessions & Memory（上下文工程与记忆）

## 你今天要达成什么

- 你会让 Agent 具备"记忆能力"：记住用户偏好、研究历史、以及跨会话的上下文
- 你会实现一个 **SQLite 长期记忆系统** + **会话管理器**
- 你会运行一个能"学习用户习惯"的 Agent：第二次使用时会基于历史调整行为
- 你会理解短期上下文（单次对话）vs 长期记忆（跨会话持久化）的区别

## 知识要点（必看）

### 为什么 Agent 需要记忆？
- **个性化**：记住用户的研究领域、报告风格偏好
- **连续性**：避免重复搜索相同内容，基于历史研究深入探索
- **效率**：利用以前的搜索结果和洞察，减少冗余工作
- **学习能力**：随着使用次数增加，Agent 变得更了解用户需求

### 记忆系统架构
```
短期记忆（Session State）
├── 当前对话的消息历史
├── 当前任务的中间结果
└── 临时上下文信息

长期记忆（SQLite Database）
├── 用户偏好（research_domain, report_style）
├── 研究历史（topics, goals, sessions）
├── 搜索结果缓存
└── 生成的报告存档
```

### 今天的架构升级
```
Day 2: 计划 → 搜索工具 → 写报告
Day 3: 加载上下文 → 计划 → 搜索 → 写报告 → 保存上下文
```

## 你将修改/阅读的代码位置

- `src/academic_researcher/memory/base.py`：记忆系统抽象接口
- `src/academic_researcher/memory/sqlite_memory.py`：SQLite 实现
- `src/academic_researcher/memory/session_manager.py`：会话和偏好管理
- `src/academic_researcher/graphs/day3_memory_agent.py`：带记忆的 LangGraph 流程
- `examples/run_day3.py`：Day 3 命令行入口（支持偏好设置）

## Day 3 操作步骤（你需要手动执行）

### Step 1：确保环境激活

```powershell
conda activate academic-agent
cd academic-researcher-agent
```

### Step 2：首次运行（设置用户偏好）

```powershell
python examples\run_day3.py --topic "neural architecture search" --goal "Compare recent NAS methods and identify trends" --user-id "researcher_001" --setup-preferences
```

这会：
- 设置用户偏好（研究领域、报告风格）
- 运行研究任务
- 将结果保存到 SQLite 数据库

### Step 3：第二次运行（体验记忆效果）

```powershell
python examples\run_day3.py --topic "automated machine learning" --goal "Find connections to neural architecture search" --user-id "researcher_001" --show-stats
```

注意观察：
- Agent 会提到你的研究偏好
- 会参考你之前的 "neural architecture search" 研究
- 报告风格会符合你设置的偏好

### Step 4：查看记忆统计

运行时加上 `--show-stats` 参数，你会看到：
- 总记忆条目数
- 会话数量
- 不同类型记忆的分布

## 记忆系统的工作原理

### 1. 用户偏好存储
```python
# 存储偏好
await session_manager.store_user_preference(
    user_id="researcher_001",
    preference_type="research_domain", 
    preference_value="machine learning"
)

# 检索偏好
preferences = await session_manager.get_user_preferences("researcher_001")
```

### 2. 研究上下文保存
```python
# 保存研究会话
await session_manager.store_research_context(
    user_id="researcher_001",
    session_id="session_123",
    topic="neural networks",
    goal="survey recent advances",
    search_results="...",
    report="..."
)
```

### 3. 相关上下文检索
```python
# 基于当前话题找相关历史
context = await session_manager.get_relevant_context(
    user_id="researcher_001",
    current_topic="deep learning",
    limit=3
)
```

## 关键差异：Day 2 vs Day 3

| 方面 | Day 2 | Day 3 |
|------|-------|-------|
| 上下文来源 | 仅当前对话 | 当前对话 + 历史记忆 |
| 个性化 | 无 | 基于用户偏好调整 |
| 连续性 | 每次独立 | 跨会话学习积累 |
| 存储 | 内存临时 | SQLite 持久化 |
| 用户体验 | 一致但通用 | 越用越个性化 |

## 数据库文件

运行后会在项目根目录生成 `academic_agent_memory.db`，包含：
- `memory_entries` 表：所有记忆条目
- 索引：按用户、会话、类型、时间快速查询

## 小练习（建议你做）

1. **多用户测试**：用不同的 `--user-id` 运行，观察记忆隔离
2. **偏好实验**：手动修改数据库中的偏好，看 Agent 行为变化
3. **记忆搜索**：在代码中测试 `memory.search()` 方法的效果

## 故障排除

如果遇到异步相关错误：
- 确保 Python 版本支持 `asyncio`
- 检查是否有其他程序占用 SQLite 文件

如果记忆功能不生效：
- 检查 `academic_agent_memory.db` 是否创建
- 用 SQLite 浏览器查看数据是否正确存储