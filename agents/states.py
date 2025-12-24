# -*- coding: utf-8 -*-
"""
State definitions for the GeoAgent LangGraph application.
"""
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the GeoAgent conversational assistant.
    
    The messages field maintains the conversation history using LangGraph's
    add_messages reducer, which appends new messages rather than overwriting.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
