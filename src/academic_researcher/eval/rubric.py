"""Evaluation rubric and scoring criteria for research reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RubricCriterion:
    """A single criterion in the evaluation rubric."""
    name: str
    description: str
    max_score: int
    weight: float = 1.0


@dataclass
class RubricScore:
    """Score for a single criterion."""
    criterion_name: str
    score: int
    max_score: int
    feedback: str

    @property
    def normalized(self) -> float:
        """Return score normalized to [0, 1]."""
        return self.score / self.max_score if self.max_score > 0 else 0.0


@dataclass
class EvaluationResult:
    """Full evaluation result for a research report."""
    topic: str
    goal: str
    scores: List[RubricScore] = field(default_factory=list)
    overall_feedback: str = ""
    passed: bool = False

    @property
    def total_score(self) -> float:
        """Weighted total score (0–100)."""
        if not self.scores:
            return 0.0
        total_weight = len(self.scores)
        weighted_sum = sum(s.normalized for s in self.scores)
        return round((weighted_sum / total_weight) * 100, 1)

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Report ===",
            f"Topic   : {self.topic}",
            f"Goal    : {self.goal}",
            f"Overall : {self.total_score:.1f} / 100  ({'PASS' if self.passed else 'FAIL'})",
            "",
            "--- Criterion Scores ---",
        ]
        for s in self.scores:
            bar = "█" * s.score + "░" * (s.max_score - s.score)
            lines.append(f"  {s.criterion_name:<28} [{bar}] {s.score}/{s.max_score}")
            lines.append(f"    Feedback: {s.feedback}")
        lines += ["", "--- Overall Feedback ---", self.overall_feedback]
        return "\n".join(lines)


# The standard rubric used to evaluate every research report
RESEARCH_REPORT_RUBRIC: List[RubricCriterion] = [
    RubricCriterion(
        name="Research Question Clarity",
        description=(
            "Is there a clearly stated, focused research question or problem? "
            "Is the motivation explained? Is the scope well-defined?"
        ),
        max_score=5,
    ),
    RubricCriterion(
        name="Literature Coverage",
        description=(
            "Does the report cite specific, relevant papers? "
            "Is the breadth of coverage adequate? Are key works mentioned?"
        ),
        max_score=5,
    ),
    RubricCriterion(
        name="Methodology Quality",
        description=(
            "Is the proposed approach well-described and justified? "
            "Are the methods appropriate for the research question?"
        ),
        max_score=5,
    ),
    RubricCriterion(
        name="Critical Analysis",
        description=(
            "Does the report critically discuss strengths, limitations, and trade-offs? "
            "Does it compare approaches rather than just describing them?"
        ),
        max_score=5,
    ),
    RubricCriterion(
        name="Structure & Coherence",
        description=(
            "Is the report logically structured with clear sections? "
            "Does the narrative flow coherently from motivation to conclusion?"
        ),
        max_score=5,
    ),
    RubricCriterion(
        name="Actionable Next Steps",
        description=(
            "Are concrete, specific next steps or future directions identified? "
            "Are they realistic and tied to the research question?"
        ),
        max_score=5,
    ),
]
