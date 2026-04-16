"""FastAPI server that exposes the Academic Researcher Agent as a REST API."""

from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from academic_researcher.agents.multi_agent_graph import MultiAgentOrchestrator
from academic_researcher.graphs.day4_quality_agent import QualityAwareAgent
from academic_researcher.observability.logger import setup_langsmith

setup_langsmith()

app = FastAPI(
    title="Academic Researcher Agent API",
    description="REST API for the Academic Researcher Agent (Day 5)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ──────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    goal: str
    user_id: str = "default"
    domain: Optional[str] = None
    use_multi_agent: bool = False


class ResearchResponse(BaseModel):
    request_id: str
    topic: str
    goal: str
    report: str
    score: Optional[float] = None
    passed: Optional[bool] = None
    evaluation_summary: Optional[str] = None
    trace_summary: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Endpoints ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/research", response_model=ResearchResponse)
async def create_research(req: ResearchRequest):
    """Run a research task and return the full result at once."""
    request_id = str(uuid.uuid4())

    try:
        if req.use_multi_agent:
            orchestrator = MultiAgentOrchestrator()
            result = await orchestrator.run(
                topic=req.topic,
                goal=req.goal,
                user_id=req.user_id,
                domain=req.domain,
            )
        else:
            agent = QualityAwareAgent()
            result = await agent.run(
                topic=req.topic, goal=req.goal, user_id=req.user_id
            )

        ev = result.get("evaluation")
        return ResearchResponse(
            request_id=request_id,
            topic=req.topic,
            goal=req.goal,
            report=result.get("report", ""),
            score=ev.total_score if ev else None,
            passed=ev.passed if ev else None,
            evaluation_summary=ev.summary() if ev else None,
            trace_summary=result.get("trace"),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/research/stream")
async def create_research_stream(req: ResearchRequest):
    """
    Run a research task with Server-Sent Events (SSE) streaming.

    Each node completion emits a `node_complete` event so the client can
    show real-time progress. The final event has type `done` and contains
    the full report, score, and trace.

    Client usage (JavaScript):
        const es = new EventSource('/research/stream', { method: 'POST', body: ... });
        // or use fetch + ReadableStream:
        const resp = await fetch('/research/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: '...', goal: '...' }),
        });
        const reader = resp.body.getReader();

    Client usage (PowerShell / curl):
        Invoke-RestMethod -Method POST -Uri http://localhost:8000/research/stream `
            -ContentType "application/json" `
            -Body '{"topic":"LLM alignment","goal":"survey"}'

    Client usage (Python):
        import httpx, json
        with httpx.stream("POST", url, json=payload) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    print(event["type"], event.get("node", ""))
    """

    async def event_generator():
        try:
            if req.use_multi_agent:
                runner = MultiAgentOrchestrator()
                async for event in runner.stream(
                    topic=req.topic,
                    goal=req.goal,
                    user_id=req.user_id,
                    domain=req.domain,
                ):
                    line = json.dumps(event, ensure_ascii=False, default=str)
                    yield f"data: {line}\n\n"
            else:
                runner = QualityAwareAgent()
                async for event in runner.stream(
                    topic=req.topic,
                    goal=req.goal,
                    user_id=req.user_id,
                ):
                    line = json.dumps(event, ensure_ascii=False, default=str)
                    yield f"data: {line}\n\n"

        except Exception as exc:
            error_event = json.dumps({"type": "error", "detail": str(exc)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/agents", summary="List available agents")
async def list_agents():
    """Return metadata about the available agent configurations."""
    return {
        "agents": [
            {
                "id": "single_agent",
                "name": "Single Quality-Aware Agent",
                "description": "Day 4 agent with tools, memory, evaluation loop",
                "endpoints": [
                    "POST /research  (use_multi_agent=false)",
                    "POST /research/stream  (use_multi_agent=false)",
                ],
            },
            {
                "id": "multi_agent",
                "name": "Multi-Agent Orchestrator",
                "description": "Planner → Researcher → Writer → Reviewer pipeline",
                "endpoints": [
                    "POST /research  (use_multi_agent=true)",
                    "POST /research/stream  (use_multi_agent=true)",
                ],
            },
        ]
    }
