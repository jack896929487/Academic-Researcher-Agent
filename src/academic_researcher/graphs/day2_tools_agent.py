"""Day 2: Agent with Tools - ArXiv search and MCP integration."""

from __future__ import annotations

from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from academic_researcher.llm import get_chat_model
from academic_researcher.tools.arxiv_search import get_arxiv_search_tool
from academic_researcher.tools.mcp_tools import get_mcp_tools, get_demo_mcp_tools


class Day2State(TypedDict):
    topic: str
    goal: str
    user_id: str
    messages: List[BaseMessage]
    plan: Optional[str]
    search_results: Optional[str]
    report: Optional[str]


def node_build_plan(state: Day2State) -> dict:
    """Generate a research plan that includes what to search for."""
    llm = get_chat_model()
    
    prompt = [
        SystemMessage(
            content=(
                "You are an academic research assistant. "
                "Create a structured research plan for the user. "
                "Your plan should include: "
                "(1) core research question, "
                "(2) key concepts and terms to search for, "
                "(3) what specific papers or topics to look for in ArXiv, "
                "(4) how to structure the final report. "
                "Be specific about search terms that would help find relevant papers."
            )
        ),
        HumanMessage(content=f"Topic: {state['topic']}\nGoal: {state['goal']}"),
    ]
    
    result = llm.invoke(prompt)
    
    # Add the plan to messages for context
    new_messages = state["messages"] + [
        HumanMessage(content=f"Create research plan for: {state['topic']}"),
        AIMessage(content=result.content)
    ]
    
    return {"plan": result.content, "messages": new_messages}


def node_search_literature(state: Day2State) -> dict:
    """Use ArXiv search tool to find relevant papers."""
    llm_with_tools = get_chat_model()
    
    # Get available tools
    tools = [get_arxiv_search_tool()]
    
    # Add MCP tools if available
    mcp_tools = get_mcp_tools()
    if not mcp_tools:
        # Use demo tools for learning purposes
        mcp_tools = get_demo_mcp_tools()
    
    tools.extend(mcp_tools)
    
    # Bind tools to the model
    llm_with_tools = llm_with_tools.bind_tools(tools)
    
    prompt = [
        SystemMessage(
            content=(
                "You are a research assistant. Based on the research plan, "
                "use the arxiv_search tool to find relevant academic papers. "
                "Search for 2-3 different queries to get comprehensive coverage. "
                "Focus on recent papers that are most relevant to the research topic."
            )
        ),
        HumanMessage(
            content=(
                f"Research Plan:\n{state['plan']}\n\n"
                f"Topic: {state['topic']}\n"
                f"Goal: {state['goal']}\n\n"
                "Please search for relevant papers using the arxiv_search tool. "
                "Use specific search terms from the research plan."
            )
        ),
    ]
    
    # Get the model's response (may include tool calls)
    result = llm_with_tools.invoke(prompt)
    
    search_results = ""
    new_messages = state["messages"] + [
        HumanMessage(content="Search for relevant literature"),
        result
    ]
    
    # If the model made tool calls, execute them
    if result.tool_calls:
        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute the tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                tool_result = tool.invoke(tool_args)
                search_results += f"\n\n=== {tool_name} results ===\n{tool_result}"
                
                # Add tool result to messages
                new_messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                )
    
    return {"search_results": search_results, "messages": new_messages}


def node_write_report(state: Day2State) -> dict:
    """Write a research report incorporating the search results."""
    llm = get_chat_model()
    
    prompt = [
        SystemMessage(
            content=(
                "You are writing an academic research report. "
                "Use the research plan and the literature search results to write "
                "a comprehensive report with these sections: "
                "1) Research question and motivation, "
                "2) Literature review (citing the papers found), "
                "3) Proposed methodology, "
                "4) Expected contributions, "
                "5) Limitations and future work. "
                "Make sure to reference specific papers from the search results."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\nGoal: {state['goal']}\n\n"
                f"Research Plan:\n{state['plan']}\n\n"
                f"Literature Search Results:\n{state['search_results']}\n\n"
                "Now write the comprehensive research report."
            )
        ),
    ]
    
    result = llm.invoke(prompt)
    
    new_messages = state["messages"] + [
        HumanMessage(content="Write comprehensive research report"),
        AIMessage(content=result.content)
    ]
    
    return {"report": result.content, "messages": new_messages}


def build_day2_graph():
    """Build the Day 2 graph with tool integration."""
    graph = StateGraph(Day2State)
    
    # Add nodes
    graph.add_node("build_plan", node_build_plan)
    graph.add_node("search_literature", node_search_literature)
    graph.add_node("write_report", node_write_report)
    
    # Define the flow
    graph.set_entry_point("build_plan")
    graph.add_edge("build_plan", "search_literature")
    graph.add_edge("search_literature", "write_report")
    graph.add_edge("write_report", END)
    
    return graph.compile()


def run_day2(topic: str, goal: str, user_id: str = "default") -> str:
    """Run the Day 2 agent with tools."""
    app = build_day2_graph()
    
    initial_state: Day2State = {
        "topic": topic,
        "goal": goal,
        "user_id": user_id,
        "messages": [HumanMessage(content=f"Research topic: {topic}\nGoal: {goal}")],
        "plan": None,
        "search_results": None,
        "report": None,
    }
    
    result = app.invoke(initial_state)
    return result["report"] or ""