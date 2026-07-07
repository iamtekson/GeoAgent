# -*- coding: utf-8 -*-
"""
Pydantic schemas for structured LLM outputs in GeoAgent processing workflows.

These schemas ensure consistent, type-safe responses from different LLM providers
(Ollama, Gemini, ChatGPT, Anthropic, etc.) by enforcing structured output formats.
"""
from typing import Dict, Any, List
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output for the per-task routing decision."""

    is_processing_task: bool = Field(
        description="Whether the task requires running a QGIS processing algorithm"
    )
    reason: str = Field(description="Brief explanation for the routing decision")


class AlgorithmSelection(BaseModel):
    """Structured output for algorithm selection."""

    algorithm_id: str = Field(
        description="The selected algorithm ID (e.g., 'native:buffer')"
    )
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
    """Structured definition of a single task in a multi-step workflow."""

    task_id: int = Field(description="Sequential task identifier (1, 2, 3, ...)")
    operation: str = Field(description="Human-readable description of the operation")
    algorithm_hint: str = Field(
        default="",
        description="Suggested algorithm family (e.g., 'buffer', 'clip'), empty if not a geoprocessing task",
    )
    search_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "For geoprocessing tasks: 3-8 lowercase GIS search terms — the "
            "operation verb plus synonyms and method names (e.g. for a median "
            "calculation: ['median', 'percentile', 'quantile', 'zonal', 'statistics'])"
        ),
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="task_ids whose outputs this task consumes (e.g., [1])",
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


class ErrorAnalysis(BaseModel):
    """Analysis of a failed geoprocessing task, used to steer the retry."""

    diagnosis: str = Field(description="Clear explanation of why the task failed")
    suggested_fix: str = Field(
        default="",
        description="Concrete change to try on retry (different algorithm, corrected parameter, etc.)",
    )
    user_suggestions: List[str] = Field(
        default_factory=list,
        description="Actionable steps the user can take if retries fail",
    )


__all__ = [
    "RouteDecision",
    "AlgorithmSelection",
    "ParameterGathering",
    "TaskDefinition",
    "TaskDecomposition",
    "ErrorAnalysis",
]
