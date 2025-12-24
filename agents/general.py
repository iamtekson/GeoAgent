# -*- coding: utf-8 -*-
"""
General mode agent for conversational GIS assistance.
"""
from typing import List

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from ..tools import TOOLS

from .states import AgentState


def build_graph_app(llm) -> any:
    """Build and compile a LangGraph app wired with tools and memory."""

    try:
        bound_llm = llm.bind_tools(list(TOOLS.values()))
    except Exception:
        bound_llm = llm  # Fallback if bind_tools is not available (e.g., Ollama LLM- deepseek models)

    def llm_node(state: AgentState) -> AgentState:
        response = bound_llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def tool_node(state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        tool_messages = []
        for call in getattr(last_msg, "tool_calls", []) or []:
            tool_inst = TOOLS[call["name"]]
            result = tool_inst.invoke(call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
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
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


def invoke_app(app, thread_id: str, messages: List[BaseMessage]) -> AIMessage:
    """Invoke the compiled app and return the last AI message."""
    state = {"messages": messages}
    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    # result["messages"] is the full list; return the last AI message
    return result["messages"][-1]
