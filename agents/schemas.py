# -*- coding: utf-8 -*-
"""
Pydantic schemas for structured LLM outputs in GeoAgent processing workflows.

These schemas ensure consistent, type-safe responses from different LLM providers
(Ollama, Gemini, ChatGPT, etc.) by enforcing structured output formats.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output for routing decision."""

    is_processing_task: bool = Field(
        description="Whether the query requires geoprocessing execution"
    )
    reason: str = Field(description="Brief explanation for the routing decision")


class AlgorithmSelection(BaseModel):
    """Structured output for algorithm selection."""

    algorithm_id: str = Field(
        description="The selected algorithm ID (e.g., 'native:buffer')"
    )
    algorithm_name: str = Field(description="Human-readable algorithm name")
    reasoning: str = Field(
        description="Explanation for why this algorithm was selected"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0
    )


class ParameterGathering(BaseModel):
    """Structured output for parameter extraction."""

    parameters: Dict[str, Any] = Field(
        description="Dictionary of parameter names to values"
    )
    notes: str = Field(description="Brief explanation of inferred values", default="")


class TaskDefinition(BaseModel):
    """Structured definition of a single processing task."""

    task_id: int = Field(description="Sequential task identifier (1, 2, 3, ...)")
    operation: str = Field(description="Human-readable description of the operation")
    algorithm_hint: str = Field(
        description="Suggested algorithm family (e.g., 'buffer', 'clip', 'zonal_statistics')"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this task (e.g., [1, 2])",
    )
    key_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Explicitly mentioned parameters from the query",
    )


class TaskDecomposition(BaseModel):
    """Structured output for decomposing multi-step queries."""

    tasks: List[TaskDefinition] = Field(
        description="Ordered list of subtasks to execute"
    )
    reasoning: str = Field(description="Explanation of how the query was decomposed")
    total_steps: int = Field(description="Total number of tasks")


class DependencyInjection(BaseModel):
    """Structured output for dependency analysis and parameter injection."""

    parameter_injections: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter_name to previous_task_output (e.g., {'INPUT': 'task_1_output'})",
    )
    reasoning: str = Field(
        description="Explanation of which outputs were injected and why"
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0) on injection accuracy", ge=0.0, le=1.0
    )
    suggested_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters auto-filled based on context",
    )


class TaskResult(BaseModel):
    """Result of executing a single task."""

    task_id: int = Field(description="Which task was executed")
    success: bool = Field(description="Whether the task executed successfully")
    output_layers: List[str] = Field(
        default_factory=list, description="Layer names produced by this task"
    )
    execution_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed result from algorithm execution"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if task failed"
    )


class ErrorAnalysis(BaseModel):
    """Analysis of a failed task with suggestions."""

    diagnosis: str = Field(description="Clear explanation of why the task failed")
    missing_info: List[str] = Field(
        default_factory=list,
        description="List of missing or ambiguous parameters/layers",
    )
    user_suggestions: List[str] = Field(
        default_factory=list, description="Actionable steps to fix the query"
    )
    partial_results: str = Field(
        description="Summary of what was completed before failure"
    )
    example_query_fix: str = Field(
        description="Example of how to rephrase the query more specifically"
    )


__all__ = [
    "RouteDecision",
    "AlgorithmSelection",
    "ParameterGathering",
    "TaskDefinition",
    "TaskDecomposition",
    "DependencyInjection",
    "TaskResult",
    "ErrorAnalysis",
]
