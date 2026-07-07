# -*- coding: utf-8 -*-
"""
State definitions for the GeoAgent LangGraph applications.

Two graphs, two states:
- AgentState: general conversational mode (messages only).
- WorkflowState: the multi-step processing workflow (outer loop of the
  architecture figure: task queue, per-task results, shared outputs).
- GeoTaskState: scoped state for the geoprocessing sub-graph (the
  "GeoProcessing Workflow" box). It never leaks into the parent state,
  which keeps checkpoints small.
"""
from typing import Annotated, Sequence, Optional, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the general conversational assistant."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


class WorkflowState(TypedDict, total=False):
    """Parent state for the multi-step processing workflow."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    task_queue: List[Dict[str, Any]]
    current_task_index: int
    # task_id -> {success, output_layers, summary, error}
    task_results: Dict[int, Dict[str, Any]]
    # label ("task_1_output") -> layer name/path, for dependency injection
    available_outputs: Dict[str, str]
    current_task_is_processing: bool
    error_message: Optional[str]
    error_analysis: Optional[Dict[str, Any]]


class GeoTaskState(TypedDict, total=False):
    """Scoped state for the geoprocessing sub-graph (one task at a time)."""

    task: Dict[str, Any]
    user_query: str
    available_outputs: Dict[str, str]
    algorithm_candidates: List[Dict[str, Any]]
    excluded_algorithms: List[str]
    selected_algorithm: Optional[str]
    algorithm_metadata: Optional[Dict[str, Any]]
    parameters: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    output_layers: List[str]
    error_message: Optional[str]
    error_diagnosis: Optional[str]
    retry_count: int
    success: bool


__all__ = [
    "AgentState",
    "WorkflowState",
    "GeoTaskState",
]
