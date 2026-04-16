"""
Day 5: Multi-Agent Orchestrator.

Four specialised agents collaborate via A2A messages:
  Planner -> Researcher -> Writer -> Reviewer
                           ^            |
                           |            v
                        (revise)     (pass/fail)

The Orchestrator wires them together using LangGraph.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from academic_researcher.agents.a2a_protocol import (
    A2AMessage,
    A2AMessageBus,
    AgentRole,
    MessageIntent,
)
from academic_researcher.eval.evaluator import evaluate_report
from academic_researcher.llm import get_chat_model
from academic_researcher.memory.memory_factory import get_memory_backend
from academic_researcher.memory.session_manager import SessionManager
from academic_researcher.observability.logger import RunTracer, agent_logger, setup_langsmith
from academic_researcher.skills import SkillManager
from academic_researcher.tools.arxiv_search import deduplicate_search_results
from academic_researcher.tools.research_tools import get_research_tools

QUALITY_THRESHOLD = 60.0
MAX_REVISIONS = 1


class MultiAgentState(TypedDict):
    topic: str
    goal: str
    user_id: str
    session_id: str
    domain: Optional[str]
    resolved_domain: Optional[str]
    plan: Optional[str]
    search_results: Optional[str]
    report: Optional[str]
    user_context: Optional[str]
    review_feedback: Optional[str]
    revision_count: int
    passed: bool
    score: Optional[float]
    evaluation_error: Optional[str]
    a2a_log: List[Dict[str, Any]]


class MultiAgentOrchestrator:
    """Orchestrates Planner, Researcher, Writer, and Reviewer agents."""

    def __init__(self, memory_db_path: str = "academic_agent_memory.db"):
        self.memory = get_memory_backend(db_path=memory_db_path)
        self.session_mgr = SessionManager(self.memory)
        self.skill_manager = SkillManager()
        self.bus = A2AMessageBus()
        self.tracer: Optional[RunTracer] = None
        self.llm = get_chat_model()

    async def node_load_context(self, state: MultiAgentState) -> dict:
        with self.tracer.span("load_context", user_id=state["user_id"]):
            preferences = await self.session_mgr.get_user_preferences(state["user_id"])
            relevant = await self.session_mgr.get_relevant_context(
                state["user_id"],
                state["topic"],
                limit=3,
            )

            parts = []
            if preferences:
                parts.append("User Preferences:")
                for key, value in preferences.items():
                    parts.append(f"- {key.replace('_', ' ').title()}: {value}")
            if relevant:
                parts.append(relevant)

            resolved_domain = self.skill_manager.resolve_domain(
                topic=state["topic"],
                goal=state["goal"],
                explicit_domain=state.get("domain"),
            )

        return {
            "user_context": "\n".join(parts) if parts else "",
            "resolved_domain": resolved_domain,
        }

    def _build_system_prompt(self, role: str, state: MultiAgentState) -> str:
        return self.skill_manager.build_system_prompt(
            role,
            topic=state["topic"],
            goal=state["goal"],
            domain=state.get("resolved_domain") or state.get("domain") or "general",
        )

    def _build_skill_context(
        self,
        role: str,
        state: MultiAgentState,
        *,
        additional_text: str = "",
    ) -> str:
        return self.skill_manager.build_runtime_context(
            role,
            topic=state["topic"],
            goal=state["goal"],
            domain=state.get("resolved_domain") or state.get("domain") or "general",
            additional_text=additional_text,
            user_context=state.get("user_context") or "",
        )

    def _research_tools_for_state(self, state: MultiAgentState):
        return get_research_tools(
            state.get("resolved_domain") or state.get("domain"),
            include_demo_mcp=False,
        )

    async def agent_planner(self, state: MultiAgentState) -> dict:
        with self.tracer.span("planner_agent"):
            feedback_hint = ""
            if state.get("review_feedback"):
                feedback_hint = (
                    f"\n\nThe previous report was rejected by the Reviewer.\n"
                    f"Feedback: {state['review_feedback']}\n"
                    "Please revise the plan to address these issues."
                )

            skill_context = self._build_skill_context(
                "planner",
                state,
                additional_text=state.get("review_feedback") or "",
            )
            msgs = [
                SystemMessage(content=self._build_system_prompt("planner", state)),
                HumanMessage(content=(
                    f"{skill_context}\n\n"
                    f"Topic: {state['topic']}\n"
                    f"Goal: {state['goal']}{feedback_hint}"
                )),
            ]
            result = await self.llm.ainvoke(msgs)

            msg = A2AMessage(
                sender=AgentRole.PLANNER,
                receiver=AgentRole.RESEARCHER,
                intent=MessageIntent.DELEGATE,
                payload={"plan": result.content},
            )
            self.bus.send(msg)

        return {"plan": result.content, "a2a_log": self.bus.conversation_log()}

    async def agent_researcher(self, state: MultiAgentState) -> dict:
        with self.tracer.span("researcher_agent") as span:
            tools = self._research_tools_for_state(state)
            llm_with_tools = self.llm.bind_tools(tools)
            skill_context = self._build_skill_context(
                "researcher",
                state,
                additional_text=state.get("plan") or "",
            )
            msgs = [
                SystemMessage(content=self._build_system_prompt("researcher", state)),
                HumanMessage(content=(
                    f"{skill_context}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Topic: {state['topic']}"
                )),
            ]
            result = await llm_with_tools.ainvoke(msgs)

            search_results = ""
            tool_budget = int(
                self.skill_manager.get_role_parameter("researcher", "max_tool_calls", 3)
            )
            tool_calls = (result.tool_calls or [])[:tool_budget]
            span["available_tools"] = [tool.name for tool in tools]
            if tool_calls:
                span["tool_calls"] = len(tool_calls)

                async def _run_tool(tc: dict) -> str:
                    tool = next((t for t in tools if t.name == tc["name"]), None)
                    if tool is None:
                        return ""
                    return await asyncio.to_thread(tool.invoke, tc["args"])

                tool_outputs = await asyncio.gather(*[_run_tool(tc) for tc in tool_calls])
                for tc, tr in zip(tool_calls, tool_outputs):
                    if tr:
                        search_results += f"\n\n=== {tc['name']} ===\n{tr}"
                search_results = deduplicate_search_results(search_results)

            msg = A2AMessage(
                sender=AgentRole.RESEARCHER,
                receiver=AgentRole.WRITER,
                intent=MessageIntent.DELEGATE,
                payload={"search_results": search_results},
            )
            self.bus.send(msg)

        return {"search_results": search_results, "a2a_log": self.bus.conversation_log()}

    async def agent_writer(self, state: MultiAgentState) -> dict:
        with self.tracer.span("writer_agent", revision=state.get("revision_count", 0)):
            revision_hint = ""
            if state.get("review_feedback"):
                revision_hint = (
                    f"\n\nREVIEWER FEEDBACK (address these issues):\n"
                    f"{state['review_feedback']}"
                )

            raw_results = state.get("search_results") or ""
            search_results_char_limit = int(
                self.skill_manager.get_role_parameter(
                    "writer",
                    "search_results_char_limit",
                    6000,
                )
            )
            if len(raw_results) > search_results_char_limit:
                raw_results = raw_results[:search_results_char_limit] + "\n...[truncated]"

            skill_context = self._build_skill_context(
                "writer",
                state,
                additional_text="\n".join(
                    filter(
                        None,
                        [
                            state.get("plan") or "",
                            state.get("review_feedback") or "",
                            raw_results[:1200],
                        ],
                    )
                ),
            )
            msgs = [
                SystemMessage(content=self._build_system_prompt("writer", state)),
                HumanMessage(content=(
                    f"{skill_context}\n\n"
                    f"Topic: {state['topic']}\n"
                    f"Goal: {state['goal']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Search Results:\n{raw_results}"
                    f"{revision_hint}"
                )),
            ]
            result = await self.llm.ainvoke(msgs)

            msg = A2AMessage(
                sender=AgentRole.WRITER,
                receiver=AgentRole.REVIEWER,
                intent=MessageIntent.DELEGATE,
                payload={"report": result.content},
            )
            self.bus.send(msg)

        return {"report": result.content, "a2a_log": self.bus.conversation_log()}

    async def agent_reviewer(self, state: MultiAgentState) -> dict:
        with self.tracer.span("reviewer_agent", revision=state.get("revision_count", 0)) as span:
            try:
                ev = await asyncio.to_thread(
                    evaluate_report,
                    topic=state["topic"],
                    goal=state["goal"],
                    report=state["report"],
                    pass_threshold=QUALITY_THRESHOLD,
                )
            except Exception as exc:
                span["score"] = None
                span["passed"] = True
                span["evaluation_error"] = str(exc)
                agent_logger.warning(
                    "reviewer_evaluation_unavailable",
                    revision=state.get("revision_count", 0),
                    domain=state.get("resolved_domain"),
                    detail=str(exc),
                )
                return {
                    "passed": True,
                    "score": None,
                    "evaluation_error": str(exc),
                    "review_feedback": None,
                    "a2a_log": self.bus.conversation_log(),
                }

            span["score"] = ev.total_score
            span["passed"] = ev.passed

            if ev.passed:
                msg = A2AMessage(
                    sender=AgentRole.REVIEWER,
                    receiver=AgentRole.ORCHESTRATOR,
                    intent=MessageIntent.COMPLETE,
                    payload={"score": ev.total_score, "summary": ev.summary()},
                )
            else:
                msg = A2AMessage(
                    sender=AgentRole.REVIEWER,
                    receiver=AgentRole.PLANNER,
                    intent=MessageIntent.FEEDBACK,
                    payload={"score": ev.total_score, "feedback": ev.overall_feedback},
                )
            self.bus.send(msg)

            agent_logger.info(
                "reviewer_verdict",
                score=ev.total_score,
                passed=ev.passed,
                revision=state.get("revision_count", 0),
                domain=state.get("resolved_domain"),
            )

        return {
            "passed": ev.passed,
            "score": ev.total_score,
            "evaluation_error": None,
            "review_feedback": ev.overall_feedback if not ev.passed else None,
            "a2a_log": self.bus.conversation_log(),
        }

    async def node_save(self, state: MultiAgentState) -> dict:
        with self.tracer.span("save_context"):
            await self.session_mgr.store_research_context(
                user_id=state["user_id"],
                session_id=state["session_id"],
                topic=state["topic"],
                goal=state["goal"],
                search_results=state.get("search_results"),
                report=state.get("report"),
            )
        return {}

    def _review_routing(self, state: MultiAgentState) -> str:
        if state.get("passed"):
            return "save"
        if state.get("revision_count", 0) >= MAX_REVISIONS:
            return "save"
        return "revise"

    def _increment_revision(self, state: MultiAgentState) -> dict:
        return {"revision_count": state.get("revision_count", 0) + 1}

    def build_graph(self):
        graph = StateGraph(MultiAgentState)

        graph.add_node("load_context", self.node_load_context)
        graph.add_node("planner", self.agent_planner)
        graph.add_node("researcher", self.agent_researcher)
        graph.add_node("writer", self.agent_writer)
        graph.add_node("reviewer", self.agent_reviewer)
        graph.add_node("increment_revision", self._increment_revision)
        graph.add_node("save", self.node_save)

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "reviewer")

        graph.add_conditional_edges(
            "reviewer",
            self._review_routing,
            {
                "save": "save",
                "revise": "increment_revision",
            },
        )
        graph.add_edge("increment_revision", "planner")
        graph.add_edge("save", END)

        return graph.compile()

    async def run(
        self,
        topic: str,
        goal: str,
        user_id: str = "default",
        domain: Optional[str] = None,
    ) -> dict:
        langsmith_active = setup_langsmith()
        run_logger = agent_logger.bind(user_id=user_id, topic=topic, domain=domain)
        self.tracer = RunTracer(run_logger)

        run_logger.info("multi_agent_run_start", goal=goal, langsmith=langsmith_active)

        app = self.build_graph()
        session_id = self.session_mgr.create_session(user_id)

        initial: MultiAgentState = {
            "topic": topic,
            "goal": goal,
            "user_id": user_id,
            "session_id": session_id,
            "domain": domain,
            "resolved_domain": None,
            "plan": None,
            "search_results": None,
            "report": None,
            "user_context": None,
            "review_feedback": None,
            "revision_count": 0,
            "passed": False,
            "score": None,
            "evaluation_error": None,
            "a2a_log": [],
        }

        result = await app.ainvoke(initial)
        trace = self.tracer.summary()
        run_logger.info("multi_agent_run_complete", **trace)

        ev = result.get("evaluation")
        evaluation_error = result.get("evaluation_error")
        if ev is None and result.get("report") and not evaluation_error:
            ev = await asyncio.to_thread(
                evaluate_report,
                topic=topic,
                goal=goal,
                report=result["report"],
                pass_threshold=QUALITY_THRESHOLD,
            )

        return {
            "report": result.get("report", ""),
            "evaluation": ev,
            "evaluation_error": evaluation_error,
            "trace": trace,
            "a2a_log": result.get("a2a_log", []),
            "domain": result.get("resolved_domain") or domain,
        }

    async def stream(
        self,
        topic: str,
        goal: str,
        user_id: str = "default",
        domain: Optional[str] = None,
    ):
        """Yield per-node progress events for the SSE endpoint."""
        langsmith_active = setup_langsmith()
        run_logger = agent_logger.bind(user_id=user_id, topic=topic, domain=domain)
        self.tracer = RunTracer(run_logger)

        run_logger.info("multi_agent_run_start", goal=goal, langsmith=langsmith_active)

        app = self.build_graph()
        session_id = self.session_mgr.create_session(user_id)

        initial: MultiAgentState = {
            "topic": topic,
            "goal": goal,
            "user_id": user_id,
            "session_id": session_id,
            "domain": domain,
            "resolved_domain": None,
            "plan": None,
            "search_results": None,
            "report": None,
            "user_context": None,
            "review_feedback": None,
            "revision_count": 0,
            "passed": False,
            "score": None,
            "evaluation_error": None,
            "a2a_log": [],
        }

        final_state = dict(initial)
        async for event in app.astream(initial):
            for node_name, node_output in event.items():
                final_state.update(node_output)
                preview = {}
                for key, val in node_output.items():
                    if key == "a2a_log":
                        preview[key] = f"[{len(val)} messages]"
                    elif isinstance(val, str) and len(val) > 200:
                        preview[key] = val[:200] + "..."
                    else:
                        preview[key] = val
                yield {"type": "node_complete", "node": node_name, "data": preview}

        trace = self.tracer.summary()
        run_logger.info("multi_agent_run_complete", **trace)

        report = final_state.get("report", "")
        evaluation_error = final_state.get("evaluation_error")
        ev = None
        if report and not evaluation_error:
            ev = await asyncio.to_thread(
                evaluate_report,
                topic=topic,
                goal=goal,
                report=report,
                pass_threshold=QUALITY_THRESHOLD,
            )

        yield {
            "type": "done",
            "report": report,
            "score": ev.total_score if ev else None,
            "passed": ev.passed if ev else None,
            "evaluation_summary": ev.summary() if ev else None,
            "evaluation_error": evaluation_error,
            "trace": trace,
            "domain": final_state.get("resolved_domain") or domain,
        }


async def run_day5_async(
    topic: str,
    goal: str,
    user_id: str = "default",
    domain: Optional[str] = None,
) -> dict:
    orchestrator = MultiAgentOrchestrator()
    return await orchestrator.run(topic, goal, user_id, domain)


def run_day5(
    topic: str,
    goal: str,
    user_id: str = "default",
    domain: Optional[str] = None,
) -> dict:
    return asyncio.run(run_day5_async(topic, goal, user_id, domain))
