"""LLM-based evaluator that scores a research report against the rubric."""

from __future__ import annotations

import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from academic_researcher.llm import get_chat_model
from academic_researcher.eval.rubric import (
    EvaluationResult,
    RESEARCH_REPORT_RUBRIC,
    RubricScore,
)

# Prompt that instructs the LLM to return structured JSON scores
_EVAL_SYSTEM_PROMPT = """
You are a strict academic quality reviewer. You will be given a research report
and a set of evaluation criteria. For each criterion, you MUST output a JSON object.

Output FORMAT — respond with a JSON array, one object per criterion, in order:
[
  {
    "criterion_name": "<exact name>",
    "score": <integer 0–max_score>,
    "max_score": <max_score>,
    "feedback": "<1–2 concise sentences explaining the score>"
  },
  ...
]
Then, on a new line after the JSON array, add a line starting with
OVERALL_FEEDBACK: followed by 2–3 sentences of holistic feedback.

Be strict but fair. Penalise vague claims, missing citations, and poor structure.
""".strip()


def _build_eval_user_prompt(
    topic: str,
    goal: str,
    report: str,
) -> str:
    criteria_text = "\n".join(
        f"- {c.name} (max {c.max_score}): {c.description}"
        for c in RESEARCH_REPORT_RUBRIC
    )
    return (
        f"TOPIC: {topic}\n"
        f"GOAL:  {goal}\n\n"
        f"CRITERIA:\n{criteria_text}\n\n"
        f"REPORT TO EVALUATE:\n{report}"
    )


def _parse_llm_response(raw: str) -> tuple[List[RubricScore], str]:
    """Extract scores list and overall feedback from the LLM response."""
    # Extract JSON block
    json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    scores: List[RubricScore] = []
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                scores.append(
                    RubricScore(
                        criterion_name=item.get("criterion_name", ""),
                        score=int(item.get("score", 0)),
                        max_score=int(item.get("max_score", 5)),
                        feedback=item.get("feedback", ""),
                    )
                )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fill missing criteria with 0 scores so the result is always complete
    scored_names = {s.criterion_name for s in scores}
    for criterion in RESEARCH_REPORT_RUBRIC:
        if criterion.name not in scored_names:
            scores.append(
                RubricScore(
                    criterion_name=criterion.name,
                    score=0,
                    max_score=criterion.max_score,
                    feedback="Could not be evaluated (parse error).",
                )
            )

    # Extract overall feedback
    overall_match = re.search(r"OVERALL_FEEDBACK:\s*(.+)", raw, re.DOTALL)
    overall_feedback = (
        overall_match.group(1).strip() if overall_match else "No overall feedback provided."
    )

    return scores, overall_feedback


def evaluate_report(
    topic: str,
    goal: str,
    report: str,
    pass_threshold: float = 60.0,
) -> EvaluationResult:
    """
    Score a research report against the standard rubric using the LLM.

    Args:
        topic: Research topic string.
        goal: Research goal string.
        report: Full report text to evaluate.
        pass_threshold: Minimum overall score (0–100) to mark as PASS.

    Returns:
        EvaluationResult with scores, feedback, and pass/fail.
    """
    llm = get_chat_model()

    messages = [
        SystemMessage(content=_EVAL_SYSTEM_PROMPT),
        HumanMessage(content=_build_eval_user_prompt(topic, goal, report)),
    ]

    response = llm.invoke(messages)
    raw_text = response.content

    scores, overall_feedback = _parse_llm_response(raw_text)

    result = EvaluationResult(
        topic=topic,
        goal=goal,
        scores=scores,
        overall_feedback=overall_feedback,
    )
    result.passed = result.total_score >= pass_threshold

    return result
