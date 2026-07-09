# -*- coding: utf-8 -*-
"""
Graph builders for GeoAgent.

- General mode: conversational LLM ⇄ tools loop (figure A).
- Processing mode: multi-step workflow with the geoprocessing sub-graph (figure B).
"""
from typing import Any, List, Sequence

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools import TOOLS
from .states import AgentState
from .workflow import build_workflow_graph

# Cap on the conversation window sent to the LLM in general mode, to keep
# token usage bounded on long chats. System messages are always kept.
MAX_CONTEXT_MESSAGES = 20


def _trim_context(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Keep system messages plus the most recent window of the conversation.

    The window never starts on a ToolMessage (its triggering AIMessage must
    be included, or providers reject the request).
    """
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(rest) <= MAX_CONTEXT_MESSAGES:
        return list(messages)

    tail = rest[-MAX_CONTEXT_MESSAGES:]
    start = len(rest) - MAX_CONTEXT_MESSAGES
    while start > 0 and isinstance(tail[0], ToolMessage):
        start -= 1
        tail = rest[start:]
    return system + tail


def build_graph_app(llm) -> Any:
    """Build and compile the general-mode graph: LLM ⇄ tools until done."""

    try:
        bound_llm = llm.bind_tools(list(TOOLS.values()))
    except Exception:
        bound_llm = llm

    def llm_node(state: AgentState) -> AgentState:
        response = bound_llm.invoke(_trim_context(state["messages"]))
        return {"messages": [response]}

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
        return {"messages": tool_messages}

    def should_use_tools(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", END: END})
    # Loop back so the LLM can react to tool results (and chain more tools);
    # previously this edge went to END and the user saw raw tool output.
    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=MemorySaver())


def build_unified_graph(llm, mode: str = "general") -> Any:
    """
    Build the app for the requested mode.

    Args:
        llm: The language model instance.
        mode: 'general' (conversational + tools) or 'processing'
              (multi-step geoprocessing workflow).
    """
    if mode == "processing":
        return build_workflow_graph(llm).compile(checkpointer=MemorySaver())
    return build_graph_app(llm)


def _invoke_config(thread_id: str) -> dict:
    # Generous recursion limit: each workflow task spans several graph steps.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}


def invoke_app(app, thread_id: str, messages: List[BaseMessage]) -> AIMessage:
    """Invoke the compiled app and return the last AI message."""
    result = app.invoke({"messages": messages}, config=_invoke_config(thread_id))
    return result["messages"][-1]


async def invoke_app_async(
    app, thread_id: str, messages: List[BaseMessage]
) -> AIMessage:
    """
    Invoke the compiled app asynchronously and return the last AI message.

    Uses app.ainvoke() if available, otherwise falls back to invoke().
    """
    state = {"messages": messages}
    try:
        result = await app.ainvoke(state, config=_invoke_config(thread_id))
    except (AttributeError, NotImplementedError):
        result = app.invoke(state, config=_invoke_config(thread_id))
    return result["messages"][-1]


__all__ = [
    "build_graph_app",
    "build_unified_graph",
    "invoke_app",
    "invoke_app_async",
]
