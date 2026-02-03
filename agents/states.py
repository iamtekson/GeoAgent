# -*- coding: utf-8 -*-
"""
State definitions for the GeoAgent LangGraph application.
"""
from typing import Annotated, Sequence, Optional, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RouteState(TypedDict, total=False):
    """Minimal state needed during routing."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    is_processing_task: bool
    error_message: Optional[str]


class DiscoveryState(RouteState, total=False):
    """State after algorithm discovery."""

    algorithm_candidates: Optional[list]


class SelectionState(DiscoveryState, total=False):
    """State after algorithm selection."""

    selected_algorithm: Optional[str]


class ParameterState(SelectionState, total=False):
    """State after parameter inspection/gathering."""

    algorithm_metadata: Optional[Dict[str, Any]]
    gathered_parameters: Optional[Dict[str, Any]]


class ExecutionState(ParameterState, total=False):
    """State after execution."""

    execution_result: Optional[Dict[str, Any]]


class ProcessingState(TypedDict, total=False):
    """
    Extended state for geoprocessing workflows.
    Tracks the processing pipeline: user query → algorithm selection → parameter gathering → execution.
    Inherits messages; adds processing-specific metadata.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Processing workflow metadata
    user_query: str  # Original user query
    is_processing_task: bool  # Whether this is a processing task
    algorithm_candidates: Optional[
        list
    ]  # List of matching algorithms from find_processing_algorithm
    selected_algorithm: Optional[str]  # The chosen algorithm id (e.g., 'native:buffer')
    algorithm_metadata: Optional[
        Dict[str, Any]
    ]  # Full parameter/output definitions from get_algorithm_parameters
    gathered_parameters: Optional[
        Dict[str, Any]
    ]  # Final parameters to pass to execute_processing
    execution_result: Optional[Dict[str, Any]]  # Result from execute_processing
    error_message: Optional[str]  # Error details if processing fails

    # Multi-task workflow metadata (NEW)
    task_queue: Optional[List[Dict[str, Any]]]  # List of decomposed tasks
    task_results: Optional[
        Dict[int, Dict[str, Any]]
    ]  # Results from each task: {task_id: TaskResult}
    current_task_index: int  # Index of task currently being executed
    completed_tasks: int  # Number of successfully completed tasks
    is_multi_step: bool  # Whether this is a multi-step workflow
    is_current_task_processing: Optional[bool]  # Whether current task is geoprocessing


class AgentState(TypedDict):
    """
    State for the GeoAgent conversational assistant.

    The messages field maintains the conversation history using LangGraph's
    add_messages reducer, which appends new messages rather than overwriting.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


__all__ = [
    "RouteState",
    "DiscoveryState",
    "SelectionState",
    "ParameterState",
    "ExecutionState",
    "ProcessingState",
    "AgentState",
]
