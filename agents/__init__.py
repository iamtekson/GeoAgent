# -*- coding: utf-8 -*-
"""
Agent modules for GeoAgent.
Provides general and processing mode graphs.
"""

from .states import AgentState, ProcessingState
from .schemas import (
    RouteDecision,
    AlgorithmSelection,
    ParameterGathering,
    TaskDecomposition,
    DependencyInjection,
    TaskResult,
    ErrorAnalysis,
)
from .graph import (
    build_graph_app,
    invoke_app,
    build_unified_graph,
    invoke_app_async,
    invoke_processing_app,
)
from .processing import build_processing_graph
from .multi_step_processing import build_multi_step_processing_graph

__all__ = [
    # States
    "AgentState",
    "ProcessingState",
    # Schemas
    "RouteDecision",
    "AlgorithmSelection",
    "ParameterGathering",
    "TaskDecomposition",
    "DependencyInjection",
    "TaskResult",
    "ErrorAnalysis",
    # General mode
    "build_graph_app",
    "invoke_app",
    "invoke_app_async",
    # Processing mode
    "build_processing_graph",
    "invoke_processing_app",
    # Multi-step processing
    "build_multi_step_processing_graph",
    # Unified mode
    "build_unified_graph",
]
