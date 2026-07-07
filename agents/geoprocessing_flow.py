# -*- coding: utf-8 -*-
"""
Geoprocessing sub-graph: the "GeoProcessing Workflow" box of the architecture.

Runs ONE task at a time with its own scoped state (GeoTaskState):

    discover -> select -> inspect -> gather -> execute --success--> END
                  ^                               |
                  └──── error_analysis <──failure─┘

On failure the error analysis feeds its diagnosis back into re-selection and
parameter gathering (excluding the failed algorithm), up to MAX_RETRIES times.
"""
from typing import Any

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage

from ..tools import (
    find_processing_algorithm,
    get_algorithm_parameters,
    execute_processing,
)
from ..tools.geoprocessing import get_algorithm_catalog
from ..prompts.system import (
    ALGORITHM_SELECTION_PROMPT,
    PARAMETER_GATHERING_PROMPT,
    ERROR_ANALYSIS_PROMPT,
)
from .states import GeoTaskState
from .schemas import AlgorithmSelection, ParameterGathering, ErrorAnalysis
from ..logger.processing_logger import get_processing_logger

_logger = get_processing_logger()

MAX_RETRIES = 2
CANDIDATE_LIMIT = 30


def _available_layers_text() -> str:
    """Compact one-line-per-layer listing of the current QGIS project."""
    try:
        from qgis.core import QgsProject, QgsMapLayer

        lines = []
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.type() == QgsMapLayer.RasterLayer:
                kind = "raster"
            else:
                kind = "vector"
                try:
                    kind += f", {lyr.geometryType().name}"
                except Exception:
                    pass
            lines.append(f"- {lyr.name()} ({kind})")
        return "\n".join(lines) if lines else "No layers loaded."
    except Exception:
        return "No layer information available."


def _previous_outputs_text(state: GeoTaskState) -> str:
    outputs = state.get("available_outputs") or {}
    if not outputs:
        return "None"
    return "\n".join(f"- {label}: {layer}" for label, layer in outputs.items())


def build_geoprocessing_subgraph(llm) -> Any:
    """Build and compile the single-task geoprocessing sub-graph."""

    def discover_node(state: GeoTaskState) -> GeoTaskState:
        """Score the full algorithm registry against the task, keep a shortlist.

        Search vocabulary comes from the task's LLM-generated `search_keywords`
        (plus the algorithm hint), so unusual operations (quantile, hypsometric,
        ...) are matched without a hand-curated synonym list. An empty shortlist
        is NOT an error: the select node then falls back to the full catalog.
        """
        task = state.get("task", {})
        query = task.get("operation", "")
        keywords = [str(k) for k in (task.get("search_keywords") or [])]
        if task.get("algorithm_hint"):
            keywords.append(str(task["algorithm_hint"]))

        _logger.info(f"GEO discover: {query} (keywords: {keywords})")
        try:
            result = find_processing_algorithm.invoke(
                {"query": query, "keywords": keywords, "limit": CANDIDATE_LIMIT}
            )
            candidates = result.get("matches", [])
            _logger.info(
                f"GEO discover: {len(candidates)} candidates (of {result.get('total')})"
            )
            return {"algorithm_candidates": candidates, "error_message": None}
        except Exception as e:
            _logger.error(f"GEO discover failed: {e}", exc_info=True)
            # Not fatal: select falls back to the full catalog
            return {"algorithm_candidates": [], "error_message": None}

    def _ask_selection(task_operation: str, listing: str, retry_context: str):
        """One structured selection call; returns the algorithm id or None."""
        messages = [
            SystemMessage(content=ALGORITHM_SELECTION_PROMPT),
            HumanMessage(
                content=f"Task: {task_operation}\n\n{listing}{retry_context}"
            ),
        ]
        try:
            result: AlgorithmSelection = llm.with_structured_output(
                AlgorithmSelection
            ).invoke(messages)
            _logger.info(f"GEO select: {result.algorithm_id} ({result.reasoning})")
            return (result.algorithm_id or "").strip()
        except Exception as e:
            _logger.warning(f"GEO select LLM call failed: {e}")
            return None

    def select_node(state: GeoTaskState) -> GeoTaskState:
        """Pick the best algorithm, in up to three tiers:

        1. Shortlist from local scoring (cheap; enriched with tags/descriptions).
        2. The LLM may answer with an id it knows that isn't shortlisted —
           accepted if it exists in the registry — or 'NONE' if nothing fits.
        3. On 'NONE'/miss: one selection call over the FULL catalog (id | name),
           so unusual operations are never blocked by the local prefilter.
        """
        if state.get("error_message"):
            return {}

        task = state.get("task", {})
        operation = task.get("operation", "")
        excluded = set(state.get("excluded_algorithms") or [])
        catalog_ids = {e["id"] for e in get_algorithm_catalog()}

        # On retry, drop algorithms that already failed (unless that empties the list)
        candidates = [
            c for c in (state.get("algorithm_candidates") or []) if c["id"] not in excluded
        ] or (state.get("algorithm_candidates") or [])

        retry_context = ""
        if state.get("error_diagnosis"):
            retry_context = (
                f"\n\nPrevious attempt FAILED. Diagnosis: {state['error_diagnosis']}\n"
                f"Do not select these algorithms again: {', '.join(excluded)}"
            )

        selected = None

        # Tier 1: enriched shortlist
        if candidates:
            lines = []
            for c in candidates:
                tags = ", ".join(c.get("tags", [])[:6])
                desc = c.get("description", "")
                extra = f" — {desc}" if desc else ""
                lines.append(
                    f"- {c['id']} | {c['name']} [{c['provider']}] tags: {tags}{extra}"
                )
            answer = _ask_selection(
                operation, "Candidate algorithms:\n" + "\n".join(lines), retry_context
            )
            # Tier 2: accept any real registry id (the model may know a better
            # algorithm than the lexical prefilter surfaced)
            if answer and answer != "NONE" and answer in catalog_ids and answer not in excluded:
                selected = answer

        # Tier 3: full catalog (compact id | name), only when the shortlist missed
        if not selected:
            _logger.info("GEO select: shortlist insufficient; trying full catalog")
            full_lines = "\n".join(
                f"- {e['id']} | {e['name']}"
                for e in get_algorithm_catalog()
                if e["id"] not in excluded
            )
            answer = _ask_selection(
                operation, "All available algorithms:\n" + full_lines, retry_context
            )
            if answer and answer in catalog_ids and answer not in excluded:
                selected = answer

        # Last resort: top-scored shortlist candidate
        if not selected and candidates:
            _logger.warning("GEO select: falling back to top-scored candidate")
            selected = candidates[0]["id"]

        if not selected:
            return {
                "error_message": f"Could not find a processing algorithm for: {operation}",
                "success": False,
            }

        return {"selected_algorithm": selected}

    def inspect_node(state: GeoTaskState) -> GeoTaskState:
        """Fetch parameter definitions for the selected algorithm."""
        if state.get("error_message") or not state.get("selected_algorithm"):
            return {}
        try:
            metadata = get_algorithm_parameters.invoke(
                {"algorithm": state["selected_algorithm"]}
            )
            return {"algorithm_metadata": metadata}
        except Exception as e:
            _logger.error(f"GEO inspect failed: {e}", exc_info=True)
            return {"error_message": f"Parameter inspection failed: {e}", "success": False}

    def gather_node(state: GeoTaskState) -> GeoTaskState:
        """LLM maps the task onto the algorithm's parameters."""
        if state.get("error_message") or not state.get("algorithm_metadata"):
            return {}

        task = state.get("task", {})
        metadata = state["algorithm_metadata"]

        params_text = "\n".join(
            f"  - {p['name']} ({p['type']}, default: {p.get('default', 'N/A')}): "
            f"{p['description']} [optional: {p.get('optional', False)}]"
            + (f" options: {p['options']}" if p.get("options") else "")
            for p in metadata.get("parameters", [])
        )

        retry_context = ""
        if state.get("error_diagnosis"):
            retry_context = f"\nError diagnosis from failed attempt: {state['error_diagnosis']}\n"

        key_params = task.get("key_parameters") or {}
        messages = [
            SystemMessage(content=PARAMETER_GATHERING_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {task.get('operation', '')}\n"
                    f"Known values from query: {key_params}\n"
                    f"Original user query: {state.get('user_query', '')}\n\n"
                    f"Algorithm: {metadata.get('name', '')} ({metadata.get('id', '')})\n\n"
                    f"Available layers:\n{_available_layers_text()}\n\n"
                    f"Previous task outputs (label: layer name):\n"
                    f"{_previous_outputs_text(state)}\n"
                    f"{retry_context}\n"
                    f"Parameters:\n{params_text}"
                )
            ),
        ]

        try:
            gathered: ParameterGathering = llm.with_structured_output(
                ParameterGathering
            ).invoke(messages)
            parameters = dict(gathered.parameters)
        except Exception as e:
            _logger.error(f"GEO gather failed: {e}", exc_info=True)
            return {"error_message": f"Parameter gathering failed: {e}", "success": False}

        parameters.setdefault("OUTPUT", "TEMPORARY_OUTPUT")

        # Fill missing/None values with algorithm defaults
        for p in metadata.get("parameters", []):
            name = p["name"]
            if parameters.get(name) is None and p.get("default") is not None:
                parameters[name] = p["default"]

        missing = [
            p["name"]
            for p in metadata.get("parameters", [])
            if not p.get("optional") and parameters.get(p["name"]) is None
        ]
        if missing:
            msg = f"Missing required parameters: {', '.join(missing)}"
            _logger.error(f"GEO gather: {msg}")
            return {"error_message": msg, "success": False}

        _logger.info(f"GEO gather: {parameters}")
        return {"parameters": parameters, "error_message": None}

    def execute_node(state: GeoTaskState) -> GeoTaskState:
        """Run the algorithm; load outputs into the project."""
        if state.get("error_message"):
            return {"success": False}
        if not state.get("selected_algorithm") or state.get("parameters") is None:
            return {"error_message": "No algorithm/parameters to execute", "success": False}

        algorithm = state["selected_algorithm"]
        _logger.info(f"GEO execute: {algorithm}")

        result = execute_processing.invoke(
            {"algorithm": algorithm, "parameters": state["parameters"]}
        )

        if result.get("success"):
            output_layers = result.get("output_layers", [])
            _logger.info(f"GEO execute OK -> outputs: {output_layers}")
            return {
                "execution_result": result,
                "output_layers": output_layers,
                "success": True,
                "error_message": None,
            }

        error = str(result.get("error", "Unknown execution error"))[:800]
        _logger.error(f"GEO execute failed: {error}")
        return {"error_message": error, "success": False}

    def error_analysis_node(state: GeoTaskState) -> GeoTaskState:
        """Diagnose the failure and prepare a steered retry."""
        task = state.get("task", {})
        error = state.get("error_message", "Unknown error")
        retry_count = state.get("retry_count", 0)
        failed_alg = state.get("selected_algorithm")

        _logger.info(f"GEO error analysis (retry {retry_count + 1}/{MAX_RETRIES}): {error}")

        diagnosis = f"Execution failed: {error}"
        try:
            analysis: ErrorAnalysis = llm.with_structured_output(ErrorAnalysis).invoke(
                [
                    SystemMessage(content=ERROR_ANALYSIS_PROMPT),
                    HumanMessage(
                        content=(
                            f"Task: {task.get('operation', '')}\n"
                            f"Algorithm attempted: {failed_alg}\n"
                            f"Parameters: {state.get('parameters')}\n"
                            f"Error: {error}\n"
                            f"Available layers:\n{_available_layers_text()}"
                        )
                    ),
                ]
            )
            diagnosis = analysis.diagnosis
            if analysis.suggested_fix:
                diagnosis += f" Suggested fix: {analysis.suggested_fix}"
        except Exception as e:
            _logger.warning(f"GEO error analysis LLM failed: {e}")

        excluded = list(state.get("excluded_algorithms") or [])
        if failed_alg and failed_alg not in excluded:
            excluded.append(failed_alg)

        return {
            "error_diagnosis": diagnosis,
            "excluded_algorithms": excluded,
            "retry_count": retry_count + 1,
            "error_message": None,
        }

    def after_execute(state: GeoTaskState) -> str:
        """The figure's 'Success?' decision."""
        if state.get("success"):
            return "done"
        if state.get("retry_count", 0) < MAX_RETRIES and state.get("selected_algorithm"):
            return "retry"
        return "done"

    graph = StateGraph(GeoTaskState)
    graph.add_node("discover", discover_node)
    graph.add_node("select", select_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("gather", gather_node)
    graph.add_node("execute", execute_node)
    graph.add_node("error_analysis", error_analysis_node)

    graph.set_entry_point("discover")
    graph.add_edge("discover", "select")
    graph.add_edge("select", "inspect")
    graph.add_edge("inspect", "gather")
    graph.add_edge("gather", "execute")
    graph.add_conditional_edges(
        "execute", after_execute, {"retry": "error_analysis", "done": END}
    )
    # Retry re-enters at selection so the diagnosis can change the algorithm
    # and/or the parameters (candidates are reused; no re-discovery needed).
    graph.add_edge("error_analysis", "select")

    return graph.compile()


__all__ = ["build_geoprocessing_subgraph", "MAX_RETRIES"]
