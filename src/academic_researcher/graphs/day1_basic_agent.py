from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from academic_researcher.llm import get_chat_model


class Day1State(TypedDict):
    topic: str
    goal: str
    user_id: str
    messages: List[BaseMessage]
    plan: Optional[str]
    report: Optional[str]


def node_build_plan(state: Day1State) -> dict:
    llm = get_chat_model()
    prompt = [
        SystemMessage(
            content=(
                "You are an academic research assistant. "
                "Create a structured research plan for the user. "
                "Focus on: (1) core question, (2) sub-questions, "
                "(3) what to look for in literature, and "
                "(4) a step-by-step approach to produce a useful report. "
                "Return ONLY the plan text."
            )
        ),
        HumanMessage(content=f"Topic: {state['topic']}\nGoal: {state['goal']}"),
    ]
    result = llm.invoke(prompt)
    return {"plan": result.content}


def node_write_report(state: Day1State) -> dict:
    llm = get_chat_model()
    prompt = [
        SystemMessage(
            content=(
                "You are drafting an academic research report from a research plan. "
                "Write a concise but structured report with these sections: "
                "1) Research question, 2) Background & prior work (no need for real citations yet), "
                "3) Proposed approach/methodology, 4) Evaluation/validation ideas, "
                "5) Risks/limitations, 6) Next steps. "
                "If you lack specific details, list 3-5 clarifying questions at the end. "
                "Return ONLY the report text."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
                f"Research Plan:\n{state['plan']}\n\n"
                "Now write the report."
            )
        ),
    ]
    result = llm.invoke(prompt)
    return {"report": result.content}


def build_day1_graph():
    graph = StateGraph(Day1State)

    graph.add_node("build_plan", node_build_plan)
    graph.add_node("write_report", node_write_report)

    graph.set_entry_point("build_plan")
    graph.add_edge("build_plan", "write_report")
    graph.add_edge("write_report", END)

    return graph.compile()


def run_day1(topic: str, goal: str, user_id: str = "default") -> str:
    app = build_day1_graph()

    initial_state: Day1State = {
        "topic": topic,
        "goal": goal,
        "user_id": user_id,
        "messages": [HumanMessage(content=f"Start research on: {topic}\nGoal: {goal}")],
        "plan": None,
        "report": None,
    }

    result = app.invoke(initial_state)
    return result["report"] or ""

