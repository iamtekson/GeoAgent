# -*- coding: utf-8 -*-
"""
Graph builder for GeoAgent: routes between general conversational mode and geoprocessing workflows.
"""
from typing import List, Any
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from agents.multi_step_processing import build_multi_step_processing_graph
from ..tools import TOOLS

from .states import AgentState
from .processing import build_processing_graph, invoke_processing_app


def build_graph_app(llm) -> Any:
    """Build and compile a LangGraph app for general mode with tools and memory."""

    try:
        bound_llm = llm.bind_tools(list(TOOLS.values()))
    except Exception:
        bound_llm = llm

    def llm_node(state: AgentState) -> AgentState:
        response = bound_llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def tool_node(state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        tool_messages = []
        for call in getattr(last_msg, "tool_calls", []) or []:
            try:
                tool_inst = TOOLS.get(call["name"])
                if tool_inst is None:
                    content = f"Error: Tool '{call['name']}' is not available."
                else:
                    result = tool_inst.invoke(call["args"])
                    # Ensure result is a string for ToolMessage
                    content = result if isinstance(result, str) else str(result)
                tool_messages.append(
                    ToolMessage(content=content, tool_call_id=call["id"])
                )
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"Error executing tool '{call['name']}': {str(e)}",
                        tool_call_id=call["id"],
                    )
                )
        return {"messages": state["messages"] + tool_messages}

    def should_use_tools(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node, retry_policy=RetryPolicy(max_attempts=2))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", END)

    return graph.compile(checkpointer=MemorySaver())


def invoke_app(app, thread_id: str, messages: List[BaseMessage]) -> AIMessage:
    """Invoke the compiled app and return the last AI message."""
    state = {"messages": messages}
    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    # result["messages"] is the full list; return the last AI message
    return result["messages"][-1]


def build_unified_graph(llm, mode: str = "general", multi_step: bool = True) -> Any:
    """
    Build a mode-specific graph: either general (conversational + tools) or processing (geoprocessing workflow).

    Args:
        llm: The language model instance
        mode: Either 'general' (default) for conversational mode or 'processing' for geoprocessing workflow
        multi_step: If True (and mode='processing'), use multi-step processing graph; otherwise use single-step

    Returns:
        Compiled LangGraph application for the specified mode
    """
    if mode == "processing":
        # Build dedicated processing workflow
        if multi_step:
            # Multi-step recursive sub-graph for complex workflows
            return build_multi_step_processing_graph(llm).compile(
                checkpointer=MemorySaver()
            )
        else:
            # Traditional single-step: route → find_algorithms → select → inspect → gather_params → execute → finalize
            return build_processing_graph(llm)
    else:
        # Build general mode with tool binding and multi-turn reasoning
        return build_graph_app(llm)


async def invoke_app_async(
    app, thread_id: str, messages: List[BaseMessage]
) -> AIMessage:
    """
    Invoke the compiled app asynchronously and return the last AI message.

    Uses app.ainvoke() if available (LangGraph 0.2.0+), otherwise falls back to invoke().
    """
    state = {"messages": messages}
    try:
        # Try async invoke first (LangGraph 0.2.0+)
        result = await app.ainvoke(
            state, config={"configurable": {"thread_id": thread_id}}
        )
    except (AttributeError, NotImplementedError):
        # Fallback to sync invoke if async not available
        result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    # result["messages"] is the full list; return the last AI message
    return result["messages"][-1]


__all__ = [
    "build_graph_app",
    "invoke_app",
    "invoke_app_async",
    "build_unified_graph",
    "invoke_processing_app",
]
