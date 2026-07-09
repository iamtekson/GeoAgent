# -*- coding: utf-8 -*-
"""
Multi-step processing workflow: the outer loop of the architecture figure.

    decompose ──> prepare_task ──"Needs Geoprocessing?"──┬─> geoprocessing (sub-graph)
                       ^                                 └─> llm_task (tools/QGIS)
                       │                                          │
                       └──"More Tasks?"── update_state <──────────┘
                                              │
                                          finalize ──> END

Each task either runs through the geoprocessing sub-graph (with its own
scoped state and retry loop) or an LLM reasoning step with QGIS tools.
Outputs are queued in `available_outputs` so later tasks can consume them
(e.g. "add layer, buffer it 5km, clip the raster with the buffer").
"""
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools import TOOLS
from ..prompts.system import (
    TASK_ROUTING_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
)
from .states import WorkflowState
from .schemas import RouteDecision, TaskDecomposition
from .geoprocessing_flow import build_geoprocessing_subgraph, _available_layers_text
from ..logger.processing_logger import get_processing_logger

_logger = get_processing_logger()

MAX_TOOL_ROUNDS = 3

# Heuristic fallback when the routing LLM call fails
_GEO_KEYWORDS = (
    "buffer", "clip", "dissolve", "union", "intersect", "difference",
    "statistic", "zonal", "interpolat", "reproject", "merge", "slope",
    "hillshade", "voronoi", "centroid", "simplif", "raster calc", "overlay",
)


def _current_task(state: WorkflowState) -> Dict[str, Any]:
    tasks = state.get("task_queue") or []
    idx = state.get("current_task_index", 0)
    return tasks[idx] if idx < len(tasks) else {}


def _single_task(query: str) -> Dict[str, Any]:
    return {
        "task_id": 1,
        "operation": query,
        "algorithm_hint": "",
        "search_keywords": [],
        "dependencies": [],
        "key_parameters": {},
    }


def build_workflow_graph(llm) -> Any:
    """Build the multi-step workflow graph (uncompiled StateGraph)."""

    geo_subgraph = build_geoprocessing_subgraph(llm)

    def decompose_node(state: WorkflowState) -> WorkflowState:
        """Break the user query into an ordered task queue; resets run state."""
        user_query = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        _logger.info("=" * 60)
        _logger.info(f"WORKFLOW decompose: {user_query}")

        try:
            decomposition: TaskDecomposition = llm.with_structured_output(
                TaskDecomposition
            ).invoke(
                [
                    SystemMessage(content=TASK_DECOMPOSITION_PROMPT),
                    HumanMessage(content=f"User query: {user_query}"),
                ]
            )
            tasks = [t.model_dump() for t in decomposition.tasks]
            if not tasks:
                tasks = [_single_task(user_query)]
            _logger.info(f"WORKFLOW decompose -> {len(tasks)} task(s)")
            for t in tasks:
                _logger.debug(f"  Task {t['task_id']}: {t['operation']}")
        except Exception as e:
            _logger.warning(f"WORKFLOW decompose failed ({e}); single-task fallback")
            tasks = [_single_task(user_query)]

        return {
            "user_query": user_query,
            "task_queue": tasks,
            "current_task_index": 0,
            "task_results": {},
            "available_outputs": {},
            "error_message": None,
            "error_analysis": None,
        }

    def prepare_task_node(state: WorkflowState) -> WorkflowState:
        """The figure's 'Needs Geoprocessing?' decision for the current task."""
        task = _current_task(state)
        if not task:
            return {}

        operation = task.get("operation", "")
        try:
            decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(
                [
                    SystemMessage(content=TASK_ROUTING_PROMPT),
                    HumanMessage(content=f"Task: {operation}"),
                ]
            )
            is_processing = decision.is_processing_task
            _logger.info(
                f"WORKFLOW route task {task.get('task_id')}: "
                f"geoprocessing={is_processing} ({decision.reason})"
            )
        except Exception as e:
            is_processing = any(kw in operation.lower() for kw in _GEO_KEYWORDS)
            _logger.warning(f"WORKFLOW routing failed ({e}); heuristic -> {is_processing}")

        return {"current_task_is_processing": is_processing}

    def route_task(state: WorkflowState) -> str:
        if not _current_task(state):
            return "finalize"
        return "geoprocessing" if state.get("current_task_is_processing") else "llm_task"

    def geoprocessing_node(state: WorkflowState) -> WorkflowState:
        """Run the current task through the geoprocessing sub-graph."""
        task = _current_task(state)
        task_id = task.get("task_id", state.get("current_task_index", 0) + 1)

        outcome = geo_subgraph.invoke(
            {
                "task": task,
                "user_query": state.get("user_query", ""),
                "available_outputs": state.get("available_outputs", {}),
                "retry_count": 0,
                "excluded_algorithms": [],
                "success": False,
            },
            config={"recursion_limit": 50},
        )

        success = bool(outcome.get("success"))
        output_layers = outcome.get("output_layers") or []

        task_results = dict(state.get("task_results") or {})
        task_results[task_id] = {
            "success": success,
            "operation": task.get("operation", ""),
            "algorithm": outcome.get("selected_algorithm"),
            "output_layers": output_layers,
            "error": outcome.get("error_message"),
        }

        delta: WorkflowState = {"task_results": task_results}

        if success:
            available = dict(state.get("available_outputs") or {})
            for i, layer in enumerate(output_layers):
                available[f"task_{task_id}_output" + (f"_{i}" if i else "")] = layer
            delta["available_outputs"] = available
            delta["error_message"] = None
        else:
            delta["error_message"] = outcome.get("error_message") or "Geoprocessing task failed"
            delta["error_analysis"] = {"diagnosis": outcome.get("error_diagnosis")}

        return delta

    def llm_task_node(state: WorkflowState) -> WorkflowState:
        """Run a non-geoprocessing task with the LLM + QGIS tools (ephemeral context)."""
        task = _current_task(state)
        task_id = task.get("task_id", state.get("current_task_index", 0) + 1)
        operation = task.get("operation", "")

        outputs_ctx = ""
        if state.get("available_outputs"):
            outputs_ctx = "\nPrevious task outputs:\n" + "\n".join(
                f"- {k}: {v}" for k, v in state["available_outputs"].items()
            )

        convo = [
            SystemMessage(
                content=(
                    "You are GeoAgent inside QGIS. Execute exactly this task, "
                    "using the available tools when needed. Then reply with one "
                    "or two sentences stating what you did or the answer."
                )
            ),
            HumanMessage(
                content=f"Task: {operation}{outputs_ctx}\n\n"
                f"Available layers:\n{_available_layers_text()}"
            ),
        ]

        try:
            bound_llm = llm.bind_tools(list(TOOLS.values()))
        except Exception:
            bound_llm = llm

        try:
            response = bound_llm.invoke(convo)
            rounds = 0
            while getattr(response, "tool_calls", None) and rounds < MAX_TOOL_ROUNDS:
                convo.append(response)
                for call in response.tool_calls:
                    tool_inst = TOOLS.get(call["name"])
                    try:
                        if tool_inst is None:
                            content = f"Error: tool '{call['name']}' not available."
                        else:
                            result = tool_inst.invoke(call["args"])
                            content = result if isinstance(result, str) else str(result)
                    except Exception as e:
                        content = f"Error executing tool '{call['name']}': {e}"
                    convo.append(ToolMessage(content=content, tool_call_id=call["id"]))
                response = bound_llm.invoke(convo)
                rounds += 1

            summary = response.content if hasattr(response, "content") else str(response)
            if isinstance(summary, list):
                summary = " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p)
                    for p in summary
                )

            task_results = dict(state.get("task_results") or {})
            task_results[task_id] = {
                "success": True,
                "operation": operation,
                "summary": str(summary)[:600],
                "output_layers": [],
                "error": None,
            }
            _logger.info(f"WORKFLOW llm task {task_id} done")
            return {"task_results": task_results, "error_message": None}

        except Exception as e:
            _logger.error(f"WORKFLOW llm task {task_id} failed: {e}", exc_info=True)
            task_results = dict(state.get("task_results") or {})
            task_results[task_id] = {
                "success": False,
                "operation": operation,
                "output_layers": [],
                "error": str(e),
            }
            return {"task_results": task_results, "error_message": str(e)}

    def update_state_node(state: WorkflowState) -> WorkflowState:
        """The figure's 'Update State / Queue': advance to the next task."""
        return {
            "current_task_index": state.get("current_task_index", 0) + 1,
            "current_task_is_processing": False,
        }

    def more_tasks(state: WorkflowState) -> str:
        """The figure's 'More Tasks?' decision."""
        if state.get("error_message"):
            return "finalize"  # a failed task blocks its dependents
        if state.get("current_task_index", 0) < len(state.get("task_queue") or []):
            return "next"
        return "finalize"

    def finalize_node(state: WorkflowState) -> WorkflowState:
        """Summarize the run for the user; the only message added to history."""
        results = state.get("task_results") or {}
        error = state.get("error_message")
        analysis = state.get("error_analysis") or {}

        lines = []
        for task_id in sorted(results):
            r = results[task_id]
            mark = "OK" if r.get("success") else "FAILED"
            detail = ""
            if r.get("output_layers"):
                detail = f" -> {', '.join(r['output_layers'])}"
            elif r.get("summary"):
                detail = f" -> {r['summary'][:200]}"
            elif r.get("error"):
                detail = f" -> {r['error'][:200]}"
            lines.append(f"[{mark}] Task {task_id}: {r.get('operation', '')}{detail}")

        status = "FAILED" if error else "SUCCESS"
        context = (
            f"User request: {state.get('user_query', '')}\n"
            f"Status: {status}\n"
            f"Tasks:\n" + "\n".join(lines)
        )
        if error and analysis.get("diagnosis"):
            context += f"\nFailure diagnosis: {analysis['diagnosis']}"

        try:
            response = llm.invoke(
                [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=context)]
            )
            summary = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            _logger.error(f"WORKFLOW summary failed: {e}")
            done = sum(1 for r in results.values() if r.get("success"))
            summary = (
                f"Completed {done}/{len(state.get('task_queue') or [])} tasks."
                + (f" Failed with: {error}" if error else "")
            )

        _logger.info("WORKFLOW finalize done")
        return {"messages": [AIMessage(content=summary)]}

    graph = StateGraph(WorkflowState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("prepare_task", prepare_task_node)
    graph.add_node("geoprocessing", geoprocessing_node)
    graph.add_node("llm_task", llm_task_node)
    graph.add_node("update_state", update_state_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "prepare_task")
    graph.add_conditional_edges(
        "prepare_task",
        route_task,
        {"geoprocessing": "geoprocessing", "llm_task": "llm_task", "finalize": "finalize"},
    )
    graph.add_edge("geoprocessing", "update_state")
    graph.add_edge("llm_task", "update_state")
    graph.add_conditional_edges(
        "update_state", more_tasks, {"next": "prepare_task", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)

    return graph


__all__ = ["build_workflow_graph"]
