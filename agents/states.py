# -*- coding: utf-8 -*-
"""
State definitions for the GeoAgent LangGraph application.
"""
from typing import Annotated, Sequence, Optional, Dict, Any, Literal, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class QGISLayerState(TypedDict):
    """
    State representing a QGIS layer.
    Includes layer name and optional metadata.
    """

    layer_name: str  # Name of the QGIS layer
    layer_id: Optional[str]  # Unique identifier for the layer (if available)
    columns: Optional[List[str]]  # List of attribute columns in the layer
    metadata: Optional[Dict[str, Any]]  # Additional metadata about the layer


class layerListState(TypedDict):
    """
    State representing a list of QGIS layers.
    Contains a list of QGISLayerState items.
    """

    layers: List[QGISLayerState]  # List of QGIS layers


class RouteState(TypedDict):
    """
    State for routing between general and processing modes.
    Inherits messages; adds a mode field to indicate the current mode.
    """

    is_processing_task: Literal[True, False]


class AlgorithmSelectionState(RouteState):
    """
    State for algorithm selection step in processing workflows.
    Inherits messages; adds algorithm candidates and selected algorithm.
    """

    algorithm_candidates: Optional[list]
    selected_algorithm: Optional[str]  # The chosen algorithm id (e.g., 'native:buffer')


class ParameterGatheringState(RouteState):
    """
    State for parameter gathering step in processing workflows.
    Inherits messages; adds selected algorithm and gathered parameters.
    """

    selected_algorithm: str  # The chosen algorithm id (e.g., 'native:buffer')
    gathered_parameters: Optional[Dict[str, Any]]


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


class AgentState(TypedDict):
    """
    State for the GeoAgent conversational assistant.

    The messages field maintains the conversation history using LangGraph's
    add_messages reducer, which appends new messages rather than overwriting.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
