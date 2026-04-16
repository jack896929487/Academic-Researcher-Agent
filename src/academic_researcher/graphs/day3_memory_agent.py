"""Day 3: Agent with Memory - Context Engineering and Sessions."""

from __future__ import annotations

import asyncio
from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from academic_researcher.llm import get_chat_model
from academic_researcher.tools.arxiv_search import get_arxiv_search_tool, deduplicate_search_results
from academic_researcher.tools.mcp_tools import get_mcp_tools, get_demo_mcp_tools
from academic_researcher.memory.memory_factory import get_memory_backend
from academic_researcher.memory.session_manager import SessionManager


class Day3State(TypedDict):
    topic: str
    goal: str
    user_id: str
    session_id: str
    messages: List[BaseMessage]
    plan: Optional[str]
    search_results: Optional[str]
    report: Optional[str]
    user_context: Optional[str]  # Context from previous sessions


class MemoryAwareAgent:
    """Agent with memory capabilities."""

    def __init__(self, memory_db_path: str = "academic_agent_memory.db"):
        self.memory = get_memory_backend(db_path=memory_db_path)
        self.session_manager = SessionManager(self.memory)
        # Initialise once; get_chat_model() returns a cached singleton
        self.llm = get_chat_model()
        tools = [get_arxiv_search_tool()]
        tools.extend(get_mcp_tools() or get_demo_mcp_tools())
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)
    
    async def node_load_context(self, state: Day3State) -> dict:
        """Load relevant context from user's history."""
        # Get user preferences
        preferences = await self.session_manager.get_user_preferences(state["user_id"])
        
        # Get relevant context from previous research
        relevant_context = await self.session_manager.get_relevant_context(
            user_id=state["user_id"],
            current_topic=state["topic"],
            limit=3
        )
        
        # Build context string
        context_parts = []
        
        if preferences:
            context_parts.append("User Preferences:")
            for pref_type, pref_value in preferences.items():
                context_parts.append(f"- {pref_type.replace('_', ' ').title()}: {pref_value}")
        
        if relevant_context:
            context_parts.append("\n" + relevant_context)
        
        user_context = "\n".join(context_parts) if context_parts else "No previous context available."
        
        return {"user_context": user_context}
    
    async def node_build_plan(self, state: Day3State) -> dict:
        """Generate a research plan considering user context."""
        system_prompt = (
            "You are an academic research assistant. "
            "Create a structured research plan for the user, taking into account "
            "their previous research history and preferences. "
            "Focus on: (1) core research question, (2) key concepts to search for, "
            "(3) specific papers or topics to look for in ArXiv, "
            "(4) how to structure the final report based on user preferences."
        )

        user_prompt = (
            f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
            f"User Context:\n{state.get('user_context', 'No previous context available.')}\n\n"
            "Please create a research plan that builds on the user's previous work and preferences."
        )

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result = await self.llm.ainvoke(prompt)
        
        new_messages = state["messages"] + [
            HumanMessage(content=f"Create research plan for: {state['topic']}"),
            AIMessage(content=result.content)
        ]
        
        return {"plan": result.content, "messages": new_messages}
    
    async def node_search_literature(self, state: Day3State) -> dict:
        """Search literature with memory of previous searches."""
        system_prompt = (
            "You are a research assistant. Based on the research plan and user context, "
            "use the arxiv_search tool to find relevant academic papers. "
            "Consider the user's research history to avoid duplicating previous searches "
            "and to build upon their existing knowledge."
        )

        user_prompt = (
            f"Research Plan:\n{state['plan']}\n\n"
            f"User Context:\n{state.get('user_context', '')}\n\n"
            f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
            "Please search for relevant papers, considering what the user may have already researched."
        )

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result = await self.llm_with_tools.ainvoke(prompt)

        search_results = ""
        new_messages = state["messages"] + [
            HumanMessage(content="Search for relevant literature"),
            result,
        ]

        if result.tool_calls:
            async def _run(tc: dict) -> tuple[dict, str]:
                tool = next((t for t in self.tools if t.name == tc["name"]), None)
                tr = await asyncio.to_thread(tool.invoke, tc["args"]) if tool else ""
                return tc, tr

            pairs = await asyncio.gather(*[_run(tc) for tc in result.tool_calls])
            for tc, tr in pairs:
                if tr:
                    search_results += f"\n\n=== {tc['name']} results ===\n{tr}"
                    new_messages.append(
                        ToolMessage(content=tr, tool_call_id=tc["id"])
                    )
            # Deduplicate across multiple queries before storing
            search_results = deduplicate_search_results(search_results)

        return {"search_results": search_results, "messages": new_messages}
    
    async def node_write_report(self, state: Day3State) -> dict:
        """Write a report considering user preferences and context."""
        system_prompt = (
            "You are writing an academic research report. "
            "Use the research plan, literature search results, and user context "
            "to write a comprehensive report. Pay attention to the user's "
            "preferred report style and research domain from their context. "
            "Build upon their previous research where relevant."
        )

        raw_results = state.get("search_results") or ""
        if len(raw_results) > 6000:
            raw_results = raw_results[:6000] + "\n...[truncated]"

        user_prompt = (
            f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
            f"User Context:\n{state.get('user_context', '')}\n\n"
            f"Research Plan:\n{state['plan']}\n\n"
            f"Literature Search Results:\n{raw_results}\n\n"
            "Please write a comprehensive research report that considers the user's "
            "preferences and builds on their research history."
        )

        prompt = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result = await self.llm.ainvoke(prompt)
        
        new_messages = state["messages"] + [
            HumanMessage(content="Write comprehensive research report"),
            AIMessage(content=result.content)
        ]
        
        return {"report": result.content, "messages": new_messages}
    
    async def node_save_context(self, state: Day3State) -> dict:
        """Save the research session to memory."""
        # Store the research context
        await self.session_manager.store_research_context(
            user_id=state["user_id"],
            session_id=state["session_id"],
            topic=state["topic"],
            goal=state["goal"],
            search_results=state.get("search_results"),
            report=state.get("report")
        )
        
        return {}
    
    def build_graph(self):
        """Build the Day 3 graph with memory integration."""
        graph = StateGraph(Day3State)
        
        # Add nodes
        graph.add_node("load_context", self.node_load_context)
        graph.add_node("build_plan", self.node_build_plan)
        graph.add_node("search_literature", self.node_search_literature)
        graph.add_node("write_report", self.node_write_report)
        graph.add_node("save_context", self.node_save_context)
        
        # Define the flow
        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "build_plan")
        graph.add_edge("build_plan", "search_literature")
        graph.add_edge("search_literature", "write_report")
        graph.add_edge("write_report", "save_context")
        graph.add_edge("save_context", END)
        
        return graph.compile()
    
    async def run(self, topic: str, goal: str, user_id: str = "default") -> str:
        """Run the memory-aware agent."""
        app = self.build_graph()
        
        # Create or get session
        session_id = self.session_manager.create_session(user_id)
        
        initial_state: Day3State = {
            "topic": topic,
            "goal": goal,
            "user_id": user_id,
            "session_id": session_id,
            "messages": [HumanMessage(content=f"Research topic: {topic}\nGoal: {goal}")],
            "plan": None,
            "search_results": None,
            "report": None,
            "user_context": None,
        }
        
        result = await app.ainvoke(initial_state)
        return result["report"] or ""


def run_day3(topic: str, goal: str, user_id: str = "default") -> str:
    """Synchronous wrapper for the Day 3 agent."""
    agent = MemoryAwareAgent()
    return asyncio.run(agent.run(topic, goal, user_id))


async def run_day3_async(topic: str, goal: str, user_id: str = "default") -> str:
    """Async version of the Day 3 agent."""
    agent = MemoryAwareAgent()
    return await agent.run(topic, goal, user_id)