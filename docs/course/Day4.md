# Day 4 - Agent Quality: 评估、可观测性与质量改进循环

## 你今天要达成什么

- 让 Agent **自我评估**：用 LLM-as-Judge 方式对自己的报告打分
- 实现**质量改进循环**：评分不够就自动重写，最多迭代 2 轮
- 加入**结构化日志**和**执行追踪**：知道每一步花了多长时间、出了什么问题
- 理解 LangGraph 的**条件边（conditional edge）**：根据评估结果决定走哪条分支

## 知识要点（必看）

### LLM-as-Judge（LLM 自评）
```
传统测试：写单元测试 → 确定性输出 → PASS/FAIL
Agent 评估：让另一个 LLM 按 rubric（评分标准）打分
           → 数值化 → 可比较 → 可驱动改进循环
```
优点：灵活、可定制 rubric；
限制：结果有一定随机性，需要多次平均或设置阈值缓冲。

### 6 维度评分 Rubric
| 维度 | 满分 | 考察内容 |
|------|------|---------|
| Research Question Clarity | 5 | 研究问题是否明确、聚焦 |
| Literature Coverage | 5 | 是否引用了具体论文 |
| Methodology Quality | 5 | 方法是否合理、有依据 |
| Critical Analysis | 5 | 是否有批判性讨论 |
| Structure & Coherence | 5 | 结构是否清晰、逻辑通顺 |
| Actionable Next Steps | 5 | 下一步是否具体可操作 |

总分 = 各维度得分之和 / 满分 × 100，≥ 60 则 PASS。

### 质量改进循环（Day 4 的核心亮点）
```
write_report
     ↓
evaluate_report
     ↓
  score ≥ 60?
  ├── YES → save_context → END
  └── NO  → write_report（附上评估反馈，最多 2 轮）
```
这是 **LangGraph 条件边（conditional_edges）** 的典型用法。

### 结构化日志 vs 普通 print
```python
# 普通 print（很难分析）
print("Done")

# 结构化日志（可被日志系统解析）
{"ts":"2026-03-25T15:00:00Z","level":"INFO","event":"evaluation_complete",
 "score":72.5,"passed":true,"iteration":0}
```

### Span-based 追踪
每个节点都被包裹在一个 `tracer.span()` 里，记录耗时，最终汇总成执行报告。

## 你将修改/阅读的代码位置

- `src/academic_researcher/eval/rubric.py`：评分标准定义
- `src/academic_researcher/eval/evaluator.py`：LLM 评估器实现
- `src/academic_researcher/observability/logger.py`：结构化日志 + Span 追踪
- `src/academic_researcher/graphs/day4_quality_agent.py`：带评估的 LangGraph 流程
- `examples/run_day4.py`：Day 4 命令行入口

## Day 4 操作步骤（你需要手动执行）

### Step 1：确保环境激活

```powershell
conda activate academic-agent
cd academic-researcher-agent
```

### Step 2：运行（不显示追踪）

```powershell
python examples\run_day4.py --topic "federated learning privacy" --goal "Survey recent privacy-preserving FL methods and evaluate trade-offs"
```

你会看到：报告正文 + 评估报告（带进度条）

### Step 3：运行（显示执行追踪）

```powershell
python examples\run_day4.py --topic "federated learning privacy" --goal "Survey recent privacy-preserving FL methods" --show-trace
```

你会在最底部看到类似这样的追踪输出：
```
EXECUTION TRACE
  ✓ load_context                         42 ms
  ✓ build_plan                         1823 ms
  ✓ search_literature                  3241 ms  calls=2
  ✓ write_report                       4102 ms
  ✓ evaluate_report                    2018 ms  score=68.3
  ✓ save_context                         35 ms
Total: 6 spans, ~11261 ms
```

### Step 4：体验质量改进循环

如果第一次报告质量不足 60 分，Agent 会自动带着评估反馈重写，你会看到两次 `write_report` span。

## Day 4 关键代码片段讲解

### 条件边决定是否重写

```python
def _should_improve(self, state: Day4State) -> str:
    evaluation = state.get("evaluation")
    iteration  = state.get("improve_iteration", 0)
    if evaluation and not evaluation.passed and iteration < MAX_IMPROVE_ITERATIONS:
        return "improve"   # → 返回 write_report 节点
    return "done"          # → 继续 save_context → END
```

### 评估反馈注入重写 prompt

```python
if state.get("evaluation"):
    ev = state["evaluation"]
    weak = [s for s in ev.scores if s.normalized < 0.6]
    eval_hint = (
        f"PREVIOUS SCORE: {ev.total_score:.1f}/100\n"
        f"IMPROVE: {'; '.join(s.criterion_name for s in weak)}\n"
        f"Overall: {ev.overall_feedback}"
    )
```

## Day 3 vs Day 4 的架构对比

```
Day 3:
  load_context → build_plan → search → write_report → save_context → END

Day 4:
  load_context → build_plan → search → write_report
                                              ↓
                                       evaluate_report
                                         ↙        ↘
                                    (score<60)   (score≥60)
                                       ↙              ↘
                                  write_report    save_context → END
```

## 日志分析工具

每次运行 Day 4 都会自动将 JSON 日志追加到 `logs/agent_spans.jsonl`，你可以用内置脚本分析：

### Step 5：分析历史运行日志

```powershell
cd academic-researcher-agent
python examples\analyze_logs.py
```

你会看到类似这样的输出：

```
========================================================================
  AGENT PERFORMANCE ANALYSIS (from logs/agent_spans.jsonl)
========================================================================

  Node                           Count  Err    Min ms    Avg ms    Max ms
  ────────────────────────────── ───── ──── ───────── ───────── ─────────
  build_plan                         2    0    6906.6    7247.9    7589.2
  evaluate_report                    1    0    4652.1    4652.1    4652.1  avg_score=83.3
  load_context                       2    0       3.5       3.6       3.7
  save_context                       1    0      23.7      23.7      23.7
  search_literature                  2    0    7590.9    7677.7    7764.4  avg_tools=2.0
  write_report                       1    0   18014.5   18014.5   18014.5
  ────────────────────────────── ───── ──── ───────── ───────── ─────────
  TOTAL (avg per run)                                   37619.5

  COMPLETED RUNS
  Timestamp                    User               Topic                          Time (s)
  ──────────────────────────── ────────────────── ────────────────────────────── ────────
  2026-03-25T08:25:29.113Z     researcher_001     federated learning privacy        37.2s

  Total runs: 1   Avg run time: 37.2s
========================================================================
```

你还可以按用户过滤：

```powershell
python examples\analyze_logs.py --user-id researcher_002
```

## LangSmith 集成（自动启用）

Day 4 的 Agent 启动时会自动检测 `.env` 中的 `LANGCHAIN_API_KEY`：

- **有 Key** → LangSmith tracing 自动开启，所有 LLM 调用和 Agent 步骤都会被追踪
- **无 Key** → 正常运行，仅使用本地 JSONL 日志

### 如何启用 LangSmith

1. 注册 https://smith.langchain.com
2. 创建 API Key
3. 在 `.env` 中配置：

```
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__your_real_key_here"
LANGCHAIN_PROJECT="academic-researcher-agent"
```

4. 重新运行 Day 4，然后在 LangSmith 控制台查看：
   - 完整的 LLM 调用链路
   - 每步的输入 / 输出 / Token 用量
   - 工具调用详情
   - 延迟分布

### LangSmith vs 本地 JSONL 日志的区别

| 方面 | 本地 JSONL 日志 | LangSmith |
|------|----------------|-----------|
| 依赖 | 无需注册 | 需要 API Key |
| 数据 | 耗时、状态、分数 | 完整 prompt / response / token |
| 可视化 | `analyze_logs.py` CLI | Web 控制台 |
| 适用场景 | 快速本地调试 | 生产级追踪与调试 |

## 小练习（建议你做）

1. **调整阈值**：把 `QUALITY_THRESHOLD` 改成 80，观察改进循环触发次数
2. **自定义 rubric**：在 `rubric.py` 里加一个新维度（如 "Citation Accuracy"）
3. **多次运行后分析**：运行 3-5 次 Day 4，然后用 `analyze_logs.py` 对比各节点耗时