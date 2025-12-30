# -*- coding: utf-8 -*-
"""
Pydantic schemas for structured LLM outputs in GeoAgent processing workflows.

These schemas ensure consistent, type-safe responses from different LLM providers
(Ollama, Gemini, ChatGPT, etc.) by enforcing structured output formats.
"""
from typing import Dict, Any
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output for routing decision."""
    is_processing_task: bool = Field(
        description="Whether the query requires geoprocessing execution"
    )
    reason: str = Field(
        description="Brief explanation for the routing decision"
    )


class AlgorithmSelection(BaseModel):
    """Structured output for algorithm selection."""
    algorithm_id: str = Field(
        description="The selected algorithm ID (e.g., 'native:buffer')"
    )
    algorithm_name: str = Field(
        description="Human-readable algorithm name"
    )
    reasoning: str = Field(
        description="Explanation for why this algorithm was selected"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )


class ParameterGathering(BaseModel):
    """Structured output for parameter extraction."""
    parameters: Dict[str, Any] = Field(
        description="Dictionary of parameter names to values"
    )
    notes: str = Field(
        description="Brief explanation of inferred values",
        default=""
    )


__all__ = [
    "RouteDecision",
    "AlgorithmSelection",
    "ParameterGathering",
]
