"""Day 4: Agent with Quality Evaluation, Observability, and Improvement Loop."""

from __future__ import annotations

import asyncio
from typing import List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from academic_researcher.eval.evaluator import evaluate_report, EvaluationResult
from academic_researcher.llm import get_chat_model
from academic_researcher.memory.memory_factory import get_memory_backend
from academic_researcher.memory.session_manager import SessionManager
from academic_researcher.observability.logger import RunTracer, agent_logger, setup_langsmith
from academic_researcher.tools.arxiv_search import get_arxiv_search_tool, deduplicate_search_results
from academic_researcher.tools.mcp_tools import get_demo_mcp_tools, get_mcp_tools

# Minimum quality score to skip the improvement loop
QUALITY_THRESHOLD = 60.0
# Maximum number of improvement iterations before giving up
MAX_IMPROVE_ITERATIONS = 2


class Day4State(TypedDict):
    topic: str
    goal: str
    user_id: str
    session_id: str
    messages: List[BaseMessage]
    plan: Optional[str]
    search_results: Optional[str]
    report: Optional[str]
    user_context: Optional[str]
    evaluation: Optional[EvaluationResult]
    improve_iteration: int


class QualityAwareAgent:
    """Agent that evaluates its own output and iterates to improve quality."""

    def __init__(self, memory_db_path: str = "academic_agent_memory.db"):
        self.memory = get_memory_backend(db_path=memory_db_path)
        self.session_manager = SessionManager(self.memory)
        self.tracer: Optional[RunTracer] = None
        # Initialise once; get_chat_model() returns a cached singleton
        self.llm = get_chat_model()
        tools = [get_arxiv_search_tool()]
        tools.extend(get_mcp_tools() or get_demo_mcp_tools())
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)

    # ------------------------------------------------------------------ nodes

    async def node_load_context(self, state: Day4State) -> dict:
        with self.tracer.span("load_context", user_id=state["user_id"]):
            preferences = await self.session_manager.get_user_preferences(state["user_id"])
            relevant = await self.session_manager.get_relevant_context(
                state["user_id"], state["topic"], limit=3
            )
            parts = []
            if preferences:
                parts.append("User Preferences:")
                for k, v in preferences.items():
                    parts.append(f"  - {k.replace('_', ' ').title()}: {v}")
            if relevant:
                parts.append("\n" + relevant)
            ctx = "\n".join(parts) if parts else "No previous context."
        return {"user_context": ctx}

    async def node_build_plan(self, state: Day4State) -> dict:
        with self.tracer.span("build_plan", topic=state["topic"]):
            msgs = [
                SystemMessage(content=(
                    "You are an academic research assistant. "
                    "Create a structured, specific research plan. "
                    "Include: (1) core question, (2) 3–5 concrete search queries for ArXiv, "
                    "(3) key sub-topics to cover, (4) expected report structure."
                )),
                HumanMessage(content=(
                    f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
                    f"User Context:\n{state.get('user_context', '')}"
                )),
            ]
            result = await self.llm.ainvoke(msgs)
        return {
            "plan": result.content,
            "messages": state["messages"] + [msgs[-1], AIMessage(content=result.content)],
        }

    async def node_search_literature(self, state: Day4State) -> dict:
        with self.tracer.span("search_literature", topic=state["topic"]) as span:
            msgs = [
                SystemMessage(content=(
                    "Search ArXiv for relevant papers using the arxiv_search tool. "
                    "Run 2 targeted queries based on the plan."
                )),
                HumanMessage(content=(
                    f"Plan:\n{state['plan']}\n\nTopic: {state['topic']}"
                )),
            ]
            result = await self.llm_with_tools.ainvoke(msgs)
            search_results = ""
            new_messages = state["messages"] + [msgs[-1], result]

            if result.tool_calls:
                span["tool_calls"] = len(result.tool_calls)

                async def _run_tool(tc: dict) -> tuple[dict, str]:
                    tool = next((t for t in self.tools if t.name == tc["name"]), None)
                    tr = await asyncio.to_thread(tool.invoke, tc["args"]) if tool else ""
                    return tc, tr

                pairs = await asyncio.gather(*[_run_tool(tc) for tc in result.tool_calls])
                for tc, tr in pairs:
                    if tr:
                        search_results += f"\n\n=== {tc['name']} ===\n{tr}"
                        new_messages.append(
                            ToolMessage(content=tr, tool_call_id=tc["id"])
                        )
                # Deduplicate across multiple queries before storing
                search_results = deduplicate_search_results(search_results)

        return {"search_results": search_results, "messages": new_messages}

    async def node_write_report(self, state: Day4State) -> dict:
        iteration = state.get("improve_iteration", 0)
        span_name = "write_report" if iteration == 0 else f"improve_report_iter{iteration}"

        # Build improvement hint if we already have an evaluation
        eval_hint = ""
        if state.get("evaluation"):
            ev = state["evaluation"]
            weak = [s for s in ev.scores if s.normalized < 0.6]
            if weak:
                hints = "; ".join(
                    f"improve '{s.criterion_name}': {s.feedback}" for s in weak
                )
                eval_hint = (
                    f"\n\nPREVIOUS EVALUATION SCORE: {ev.total_score:.1f}/100\n"
                    f"AREAS TO IMPROVE: {hints}\n"
                    f"Overall feedback: {ev.overall_feedback}\n"
                    "Please address these weaknesses in the new version."
                )

        with self.tracer.span(span_name, iteration=iteration):
            # Truncate search results to avoid overly long prompts
            raw_results = state.get("search_results") or ""
            if len(raw_results) > 6000:
                raw_results = raw_results[:6000] + "\n...[truncated]"

            msgs = [
                SystemMessage(content=(
                    "Write a comprehensive academic research report with these sections: "
                    "1) Research question & motivation, "
                    "2) Literature review (cite specific papers from search results), "
                    "3) Methodology, "
                    "4) Critical analysis of strengths/limitations, "
                    "5) Next steps. "
                    "Use concrete paper titles and authors from the search results."
                )),
                HumanMessage(content=(
                    f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
                    f"User Context:\n{state.get('user_context', '')}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Search Results:\n{raw_results}"
                    f"{eval_hint}"
                )),
            ]
            result = await self.llm.ainvoke(msgs)

        return {
            "report": result.content,
            "improve_iteration": iteration + 1,
            "messages": state["messages"] + [msgs[-1], AIMessage(content=result.content)],
        }

    async def node_evaluate_report(self, state: Day4State) -> dict:
        with self.tracer.span(
            "evaluate_report",
            iteration=state.get("improve_iteration", 0)
        ) as span:
            evaluation = await asyncio.to_thread(
                evaluate_report,
                topic=state["topic"],
                goal=state["goal"],
                report=state["report"],
                pass_threshold=QUALITY_THRESHOLD,
            )
            span["score"] = evaluation.total_score
            span["passed"] = evaluation.passed
            agent_logger.info(
                "evaluation_complete",
                score=evaluation.total_score,
                passed=evaluation.passed,
                iteration=state.get("improve_iteration", 0),
            )
        return {"evaluation": evaluation}

    async def node_save_context(self, state: Day4State) -> dict:
        with self.tracer.span("save_context"):
            await self.session_manager.store_research_context(
                user_id=state["user_id"],
                session_id=state["session_id"],
                topic=state["topic"],
                goal=state["goal"],
                search_results=state.get("search_results"),
                report=state.get("report"),
            )
        return {}

    # -------------------------------------------------------- routing logic

    def _should_improve(self, state: Day4State) -> str:
        """Decide whether to improve the report or finish."""
        evaluation = state.get("evaluation")
        iteration = state.get("improve_iteration", 0)

        if evaluation and not evaluation.passed and iteration < MAX_IMPROVE_ITERATIONS:
            agent_logger.info(
                "quality_below_threshold",
                score=evaluation.total_score,
                threshold=QUALITY_THRESHOLD,
                iteration=iteration,
            )
            return "improve"
        return "done"

    # ---------------------------------------------------------- graph builder

    def build_graph(self):
        graph = StateGraph(Day4State)

        graph.add_node("load_context", self.node_load_context)
        graph.add_node("build_plan", self.node_build_plan)
        graph.add_node("search_literature", self.node_search_literature)
        graph.add_node("write_report", self.node_write_report)
        graph.add_node("evaluate_report", self.node_evaluate_report)
        graph.add_node("save_context", self.node_save_context)

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "build_plan")
        graph.add_edge("build_plan", "search_literature")
        graph.add_edge("search_literature", "write_report")
        graph.add_edge("write_report", "evaluate_report")

        # Conditional: improve or finish
        graph.add_conditional_edges(
            "evaluate_report",
            self._should_improve,
            {
                "improve": "write_report",
                "done": "save_context",
            },
        )
        graph.add_edge("save_context", END)

        return graph.compile()

    # -------------------------------------------------------- public run API

    async def run(self, topic: str, goal: str, user_id: str = "default") -> dict:
        """Run the quality-aware agent and return report + evaluation."""
        langsmith_active = setup_langsmith()

        run_logger = agent_logger.bind(user_id=user_id, topic=topic)
        self.tracer = RunTracer(run_logger)

        run_logger.info("agent_run_start", goal=goal, langsmith=langsmith_active)
        app = self.build_graph()
        session_id = self.session_manager.create_session(user_id)

        initial: Day4State = {
            "topic": topic,
            "goal": goal,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [HumanMessage(content=f"Research: {topic}")],
            "plan": None,
            "search_results": None,
            "report": None,
            "user_context": None,
            "evaluation": None,
            "improve_iteration": 0,
        }

        result = await app.ainvoke(initial)
        trace_summary = self.tracer.summary()
        run_logger.info("agent_run_complete", **trace_summary)

        return {
            "report": result.get("report", ""),
            "evaluation": result.get("evaluation"),
            "trace": trace_summary,
        }


    async def stream(self, topic: str, goal: str, user_id: str = "default"):
        """Yield per-node progress events for the SSE endpoint."""
        langsmith_active = setup_langsmith()

        run_logger = agent_logger.bind(user_id=user_id, topic=topic)
        self.tracer = RunTracer(run_logger)

        run_logger.info("agent_run_start", goal=goal, langsmith=langsmith_active)
        app = self.build_graph()
        session_id = self.session_manager.create_session(user_id)

        initial: Day4State = {
            "topic": topic,
            "goal": goal,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [HumanMessage(content=f"Research: {topic}")],
            "plan": None,
            "search_results": None,
            "report": None,
            "user_context": None,
            "evaluation": None,
            "improve_iteration": 0,
        }

        final_state = None
        async for event in app.astream(initial):
            for node_name, node_output in event.items():
                yield {"type": "node_complete", "node": node_name, "data": _safe_preview(node_output)}
            final_state = event

        trace_summary = self.tracer.summary()
        run_logger.info("agent_run_complete", **trace_summary)

        report = ""
        evaluation = None
        if final_state:
            last_vals = list(final_state.values())[-1] if final_state else {}
            report = last_vals.get("report", "")
            evaluation = last_vals.get("evaluation")

        yield {
            "type": "done",
            "report": report,
            "score": evaluation.total_score if evaluation else None,
            "passed": evaluation.passed if evaluation else None,
            "evaluation_summary": evaluation.summary() if evaluation else None,
            "trace": trace_summary,
        }


def _safe_preview(node_output: dict) -> dict:
    """Extract a small preview from node output for SSE events."""
    preview = {}
    for key, val in node_output.items():
        if key == "messages":
            preview[key] = f"[{len(val)} messages]"
        elif key == "evaluation" and val is not None:
            preview["score"] = val.total_score
            preview["passed"] = val.passed
        elif isinstance(val, str) and len(val) > 200:
            preview[key] = val[:200] + "..."
        else:
            preview[key] = val
    return preview


async def run_day4_async(topic: str, goal: str, user_id: str = "default") -> dict:
    agent = QualityAwareAgent()
    return await agent.run(topic, goal, user_id)


def run_day4(topic: str, goal: str, user_id: str = "default") -> dict:
    return asyncio.run(run_day4_async(topic, goal, user_id))
