# -*- coding: utf-8 -*-
"""
Agent modules for GeoAgent.
Provides the general conversational graph and the multi-step processing workflow.
"""

from .states import AgentState, WorkflowState, GeoTaskState
from .schemas import (
    RouteDecision,
    AlgorithmSelection,
    ParameterGathering,
    TaskDefinition,
    TaskDecomposition,
    ErrorAnalysis,
)
from .graph import (
    build_graph_app,
    build_unified_graph,
    invoke_app,
    invoke_app_async,
)
from .workflow import build_workflow_graph
from .geoprocessing_flow import build_geoprocessing_subgraph

__all__ = [
    # States
    "AgentState",
    "WorkflowState",
    "GeoTaskState",
    # Schemas
    "RouteDecision",
    "AlgorithmSelection",
    "ParameterGathering",
    "TaskDefinition",
    "TaskDecomposition",
    "ErrorAnalysis",
    # Graph builders
    "build_graph_app",
    "build_unified_graph",
    "build_workflow_graph",
    "build_geoprocessing_subgraph",
    # Invocation helpers
    "invoke_app",
    "invoke_app_async",
]
