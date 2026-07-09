# -*- coding: utf-8 -*-
"""
Multi-step geospatial processing with recursive sub-graphs.

Implements Option 3: Recursive Sub-Graph architecture for handling complex,
multi-step geoprocessing workflows.

Workflow:
1. Decompose: Break multi-step query into ordered subtasks
2. For each task:
   a. Analyze: Identify dependencies and inject outputs from previous tasks
   b. Execute: Run single-task sub-graph
   c. Store: Save results for potential use in later tasks
3. Finalize: Summarize all tasks and results
"""
import json
from typing import List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools import (
    find_processing_algorithm,
    get_algorithm_parameters,
    execute_processing,
    list_qgis_layers,
    TOOLS,
)
from ..prompts.system import (
    PROCESSING_ROUTING_PROMPT,
    PROCESSING_ALGORITHM_SELECTION_PROMPT,
    PROCESSING_PARAMETER_GATHERING_PROMPT,
    TASK_DECOMPOSITION_PROMPT,
    DEPENDENCY_ANALYSIS_PROMPT,
    MULTI_STEP_ERROR_ANALYSIS_PROMPT,
)
from .states import ProcessingState
from .schemas import (
    RouteDecision,
    AlgorithmSelection,
    ParameterGathering,
    TaskDecomposition,
    DependencyInjection,
    TaskResult,
    ErrorAnalysis,
)
from ..logger.processing_logger import get_processing_logger

_logger = get_processing_logger()


def route_node(state: ProcessingState) -> ProcessingState:
    """
    Determine if user query requires geoprocessing or should be handled by general LLM.
    This is the entry point for the multi-step processing graph.
    """
    _logger.info("=" * 80)
    _logger.info("NODE: route_node START")

    last_msg = state["messages"][-1]
    user_query = (
        last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)
    )

    _logger.debug(f"User query: {user_query}")

    # Call routing LLM with structured output
    messages = [
        SystemMessage(content=PROCESSING_ROUTING_PROMPT),
        HumanMessage(content=f"User query: {user_query}"),
    ]

    try:
        # Use structured output to ensure consistent response format
        structured_llm = llm.with_structured_output(RouteDecision)
        result: RouteDecision = structured_llm.invoke(messages)

        is_processing = result.is_processing_task
        reason = result.reason
        _logger.debug(
            f"LLM structured response: is_processing={is_processing}, reason={reason}"
        )
        _logger.info(
            f"Routing decision: is_processing_task={is_processing}, reason={reason}"
        )
    except Exception as e:
        # If structured output fails, use heuristic fallback
        _logger.warning(
            f"Structured output failed ({str(e)}), using heuristic fallback"
        )
        is_processing = False  # Default fallback

    _logger.info(f"NODE: route_node END -> is_processing_task={is_processing}")
    return {
        "messages": state["messages"],
        "user_query": user_query,
        "is_processing_task": is_processing,
    }


def decompose_tasks_node(state: ProcessingState) -> ProcessingState:
    """
    Decompose any query (processing or LLM) into ordered, executable subtasks.

    For processing queries: Uses LLM to identify task boundaries and dependencies.
    For non-processing queries: Creates a single LLM task.
    """
    _logger.info("=" * 80)
    _logger.info("NODE: decompose_tasks_node START")

    is_processing = state.get("is_processing_task", False)
    query = state["user_query"]
    _logger.debug(f"Decomposing query (is_processing={is_processing}): {query}")

    # For processing queries, try LLM decomposition into multiple tasks
    if is_processing:
        messages = [
            SystemMessage(content=TASK_DECOMPOSITION_PROMPT),
            HumanMessage(content=f"User query: {query}"),
        ]

        try:
            structured_llm = llm.with_structured_output(TaskDecomposition)
            decomposition: TaskDecomposition = structured_llm.invoke(messages)

            tasks = [task.model_dump() for task in decomposition.tasks]
            total_steps = decomposition.total_steps

            is_multi_step = total_steps > 1

            _logger.info(
                f"Decomposed into {total_steps} task(s), multi_step={is_multi_step}"
            )
            for i, task in enumerate(tasks, 1):
                _logger.debug(f"  Task {i}: {task['operation']}")
                _logger.debug(f"    - Algorithm hint: {task['algorithm_hint']}")
                _logger.debug(f"    - Dependencies: {task['dependencies']}")

            return {
                **state,
                "task_queue": tasks,
                "current_task_index": 0,
                "completed_tasks": 0,
                "is_multi_step": is_multi_step,
                "task_results": {},
            }
        except Exception as e:
            _logger.error(f"Error decomposing tasks: {str(e)}", exc_info=True)
            # Fallback: treat as single task
            _logger.warning("Falling back to single-task mode")
            single_task = {
                "task_id": 1,
                "operation": query,
                "algorithm_hint": "",
                "dependencies": [],
                "key_parameters": {},
            }
            return {
                **state,
                "task_queue": [single_task],
                "current_task_index": 0,
                "completed_tasks": 0,
                "is_multi_step": False,
                "task_results": {},
            }
    else:
        # For non-processing queries, create a single LLM task
        _logger.info("Non-processing query, creating single LLM task")
        single_task = {
            "task_id": 1,
            "operation": query,
            "algorithm_hint": "",
            "dependencies": [],
            "key_parameters": {},
        }
        return {
            **state,
            "task_queue": [single_task],
            "current_task_index": 0,
            "completed_tasks": 0,
            "is_multi_step": False,
            "task_results": {},
        }


def analyze_dependencies_node(state: ProcessingState) -> ProcessingState:
    """
    Analyze dependencies for the current task and inject outputs from previous tasks.

    Uses LLM to intelligently match previous task outputs to current task inputs.
    """
    _logger.info("=" * 80)
    _logger.info("NODE: analyze_dependencies_node START")

    if not state.get("task_queue") or state.get("current_task_index") is None:
        _logger.info("No task queue or index, skipping dependency analysis")
        return state

    current_idx = state["current_task_index"]
    tasks = state["task_queue"]

    if current_idx >= len(tasks):
        _logger.info("All tasks completed, skipping analysis")
        return state

    current_task = tasks[current_idx]
    task_id = current_task.get("task_id", current_idx + 1)

    _logger.debug(f"Analyzing dependencies for task {task_id}")

    # Get results from previous tasks
    task_results = state.get("task_results", {})
    previous_outputs = {}

    for prev_task_id, result in task_results.items():
        if result.get("success"):
            output_layers = result.get("output_layers", [])
            for idx, layer in enumerate(output_layers):
                # Store each layer under an indexed key so multiple outputs are preserved
                previous_outputs[f"task_{prev_task_id}_output_{idx}"] = layer
                # Maintain legacy behavior: unindexed key points to the last layer
                previous_outputs[f"task_{prev_task_id}_output"] = layer
                _logger.debug(f"  Available from task {prev_task_id}: {layer}")

    # Store in state for gather_parameters_node to use
    new_state = {
        **state,
        "_current_task_id": task_id,
        "_previous_outputs": previous_outputs,
    }

    _logger.info(
        f"NODE: analyze_dependencies_node END -> {len(previous_outputs)} outputs available"
    )
    return new_state


def should_route_task(state: ProcessingState) -> str:
    """
    Use LLM to route current task to geoprocessing or LLM execution.
    """
    if not state.get("task_queue") or state.get("current_task_index") is None:
        return "llm_task"

    current_idx = state["current_task_index"]
    tasks = state["task_queue"]

    if current_idx >= len(tasks):
        return "llm_task"

    current_task = tasks[current_idx]
    operation = current_task.get("operation", "")

    # Use LLM to decide if this is a geoprocessing task
    messages = [
        SystemMessage(content=PROCESSING_ROUTING_PROMPT),
        HumanMessage(content=f"Task: {operation}"),
    ]

    try:
        structured_llm = llm.with_structured_output(RouteDecision)
        result: RouteDecision = structured_llm.invoke(messages)
        is_processing = result.is_processing_task
        _logger.info(
            f"Task {current_idx + 1} routing: is_processing={is_processing}: {operation}"
        )
        return "processing" if is_processing else "llm_task"
    except Exception as e:
        _logger.warning(f"Task routing failed, defaulting to llm_task: {str(e)}")
        return "llm_task"


def discover_algorithms_node_multi(state: ProcessingState) -> ProcessingState:
    """Discover algorithms for current task in multi-step workflow."""
    _logger.info("=" * 80)
    _logger.info("NODE: discover_algorithms_node_multi START")

    if not state.get("task_queue") or state.get("current_task_index") is None:
        _logger.info("No task queue, skipping discovery")
        return state

    current_idx = state["current_task_index"]
    tasks = state["task_queue"]

    if current_idx >= len(tasks):
        return state

    current_task = tasks[current_idx]

    # Use operation description as search query
    query = current_task.get("operation", "")
    algorithm_hint = current_task.get("algorithm_hint", "")

    if algorithm_hint:
        query = f"{query} ({algorithm_hint})"

    _logger.debug(f"Discovering algorithms for task {current_idx + 1}: {query}")

    try:
        result = find_processing_algorithm.invoke({"query": query, "limit": 1000})
        matches = result.get("matches", [])
        candidates = [
            {"id": m["id"], "name": m["name"], "provider": m["provider"]}
            for m in matches
        ]
        _logger.info(f"Discovered {len(candidates)} candidates")
    except Exception as e:
        _logger.error(f"Error discovering algorithms: {str(e)}", exc_info=True)
        return {
            **state,
            "error_message": str(e),
        }

    _logger.info("NODE: discover_algorithms_node_multi END")
    return {
        **state,
        "algorithm_candidates": candidates,
    }


def select_algorithm_node_multi(state: ProcessingState) -> ProcessingState:
    """Select best algorithm for current task in multi-step workflow."""
    _logger.info("=" * 80)
    _logger.info("NODE: select_algorithm_node_multi START")

    if not state.get("algorithm_candidates"):
        _logger.info("No candidates, skipping selection")
        return state

    current_idx = state.get("current_task_index", 0)
    tasks = state.get("task_queue", [])

    if current_idx >= len(tasks):
        return state

    current_task = tasks[current_idx]
    query = current_task.get("operation", "")
    candidates = state["algorithm_candidates"]

    _logger.debug(f"Selecting algorithm for: {query}")

    candidates_text = "\n".join([f"  - {c['id']}" for c in candidates])

    messages = [
        SystemMessage(content=PROCESSING_ALGORITHM_SELECTION_PROMPT),
        HumanMessage(
            content=f"User query: {query}\n\nAvailable algorithms:\n{candidates_text}"
        ),
    ]

    try:
        structured_llm = llm.with_structured_output(AlgorithmSelection)
        result: AlgorithmSelection = structured_llm.invoke(messages)

        selected_alg = result.algorithm_id
        _logger.info(f"Selected algorithm: {selected_alg}")
    except Exception as e:
        _logger.error(f"Error selecting algorithm: {str(e)}", exc_info=True)
        selected_alg = candidates[0]["id"] if candidates else None

    if not selected_alg:
        _logger.error("No algorithm selected")
        return {
            **state,
            "error_message": "Could not select any algorithm",
        }

    _logger.info("NODE: select_algorithm_node_multi END")
    return {
        **state,
        "selected_algorithm": selected_alg,
    }


def inspect_parameters_node_multi(state: ProcessingState) -> ProcessingState:
    """Inspect parameters for current task's selected algorithm."""
    _logger.info("=" * 80)
    _logger.info("NODE: inspect_parameters_node_multi START")

    if not state.get("selected_algorithm"):
        _logger.info("No selected algorithm, skipping inspection")
        return state

    algorithm = state["selected_algorithm"]
    _logger.debug(f"Inspecting parameters for: {algorithm}")

    try:
        metadata = get_algorithm_parameters.invoke({"algorithm": algorithm})
        _logger.info(f"Retrieved metadata for {algorithm}")
    except Exception as e:
        _logger.error(f"Error inspecting parameters: {str(e)}", exc_info=True)
        return {
            **state,
            "error_message": str(e),
        }

    _logger.info("NODE: inspect_parameters_node_multi END")
    return {
        **state,
        "algorithm_metadata": metadata,
    }


def gather_parameters_node_multi(state: ProcessingState) -> ProcessingState:
    """
    Gather parameters for current task, with dependency injection.

    Intelligently injects outputs from previous tasks where appropriate.
    """
    _logger.info("=" * 80)
    _logger.info("NODE: gather_parameters_node_multi START")

    if not state.get("algorithm_metadata"):
        _logger.info("No algorithm metadata, skipping parameter gathering")
        return state

    query = state["user_query"]
    metadata = state["algorithm_metadata"]
    algorithm_name = metadata.get("name", "")

    # Get available layers
    try:
        layers_info = list_qgis_layers.invoke({})
    except Exception as e:
        _logger.warning(f"Could not retrieve layers: {e}")
        layers_info = "No layers available"

    # Get previous outputs for dependency injection
    previous_outputs = state.get("_previous_outputs", {})
    current_task_idx = state.get("current_task_index", 0)
    tasks = state.get("task_queue", [])
    current_task = tasks[current_task_idx] if current_task_idx < len(tasks) else {}

    # Format parameters
    params_text = "\n".join(
        [
            f"  - {p['name']} ({p['type']}, default: {p.get('default', 'N/A')}): {p['description']}"
            f" [optional: {p.get('optional', False)}]"
            for p in metadata.get("parameters", [])
        ]
    )

    _logger.debug(f"Parameter definitions:\n{params_text}")

    # First, use dependency analysis to identify parameter injections
    if previous_outputs:
        _logger.debug("Analyzing dependencies for parameter injection...")

        dep_messages = [
            SystemMessage(content=DEPENDENCY_ANALYSIS_PROMPT),
            HumanMessage(
                content=(
                    f"Task: {current_task.get('operation', '')}\n\n"
                    f"Algorithm: {algorithm_name}\n\n"
                    f"Previous outputs available:\n"
                    + "\n".join([f"  - {k}: {v}" for k, v in previous_outputs.items()])
                    + f"\n\nParameters:\n{params_text}\n\n"
                    f"User query: {query}"
                )
            ),
        ]

        try:
            dep_structured_llm = llm.with_structured_output(DependencyInjection)
            dep_result: DependencyInjection = dep_structured_llm.invoke(dep_messages)

            parameter_injections = dep_result.parameter_injections
            _logger.info(f"Dependency analysis: {parameter_injections}")
        except Exception as e:
            _logger.warning(f"Dependency analysis failed: {str(e)}")
            parameter_injections = {}
    else:
        parameter_injections = {}

    # Now gather all parameters
    messages = [
        SystemMessage(content=PROCESSING_PARAMETER_GATHERING_PROMPT),
        HumanMessage(
            content=(
                f"User query: {query}\n\n"
                f"Task: {current_task.get('operation', '')}\n\n"
                f"Algorithm: {algorithm_name}\n\n"
                f"Available layers:\n{layers_info}\n\n"
                f"Previous outputs (for reuse):\n"
                + (
                    "\n".join([f"  - {k}: {v}" for k, v in previous_outputs.items()])
                    if previous_outputs
                    else "  None"
                )
                + f"\n\nParameters:\n{params_text}\n\n"
                f"Please provide the parameter values in JSON format."
            )
        ),
    ]

    try:
        structured_llm = llm.with_structured_output(ParameterGathering)
        gathered: ParameterGathering = structured_llm.invoke(messages)

        parameters = gathered.parameters

        # Apply dependency injections (override/augment gathered parameters)
        for param_name, task_output in parameter_injections.items():
            if task_output in previous_outputs:
                parameters[param_name] = previous_outputs[task_output]
                _logger.info(f"Injected {param_name} = {previous_outputs[task_output]}")

        # Ensure OUTPUT parameter
        if "OUTPUT" not in parameters:
            parameters["OUTPUT"] = "TEMPORARY_OUTPUT"

        _logger.info(f"Extracted parameters: {parameters}")

        # Fill in missing required parameters with defaults
        for param in metadata.get("parameters", []):
            name = param["name"]
            current_val = parameters.get(name)
            default_value = param.get("default")

            if (name in parameters and current_val is None) or (
                not param.get("optional") and name not in parameters
            ):
                if default_value is not None:
                    parameters[name] = default_value
                    _logger.debug(
                        f"Assigned default value {default_value} to parameter {name}"
                    )

        # Validate required parameters
        required_params = [
            p["name"] for p in metadata.get("parameters", []) if not p.get("optional")
        ]
        missing_required = [
            p for p in required_params if p not in parameters or parameters[p] is None
        ]

        if missing_required:
            error_msg = (
                f"Missing required parameters: {', '.join(missing_required)}. "
                f"LLM extracted: {parameters}"
            )
            _logger.error(error_msg)
            return {**state, "error_message": error_msg}

    except Exception as e:
        _logger.error(f"Error extracting parameters: {str(e)}", exc_info=True)
        return {
            **state,
            "error_message": str(e),
        }

    _logger.info("NODE: gather_parameters_node_multi END")
    return {
        **state,
        "gathered_parameters": parameters,
    }


def execute_node_multi(state: ProcessingState) -> ProcessingState:
    """Execute current task's algorithm."""
    _logger.info("=" * 80)
    _logger.info("NODE: execute_node_multi START")

    if not state.get("selected_algorithm") or state.get("gathered_parameters") is None:
        _logger.info("Missing algorithm or parameters, skipping execution")
        return state

    algorithm = state["selected_algorithm"]
    parameters = state["gathered_parameters"]
    current_task_idx = state.get("current_task_index", 0)
    current_task_id = state.get("_current_task_id", current_task_idx + 1)

    _logger.info(f"Executing algorithm: {algorithm} (Task {current_task_id})")
    _logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")

    try:
        result = execute_processing.invoke(
            {"algorithm": algorithm, "parameters": parameters}
        )

        # Extract output layers
        output_layers = []
        if result.get("result"):
            output_obj = result["result"]
            if isinstance(output_obj, dict):
                for key, value in output_obj.items():
                    if isinstance(value, str) and value not in ["TEMPORARY_OUTPUT"]:
                        output_layers.append(value)

        _logger.info(f"Algorithm execution successful")
        _logger.debug(f"Output layers: {output_layers}")

        # Store task result
        task_results = dict(state.get("task_results", {}))
        task_results[current_task_id] = {
            "task_id": current_task_id,
            "success": True,
            "output_layers": output_layers,
            "execution_result": result,
            "error": None,
        }

        completed_tasks = state.get("completed_tasks", 0) + 1

        _logger.info("NODE: execute_node_multi END -> execution successful")
        return {
            **state,
            "execution_result": result,
            "task_results": task_results,
            "completed_tasks": completed_tasks,
            "_last_was_geoprocessing": True,
            "error_message": None,
            "_retry_count": 0,
        }

    except Exception as e:
        _logger.error(f"Processing execution failed: {str(e)}", exc_info=True)

        # Store failed task result
        task_results = dict(state.get("task_results", {}))
        task_results[current_task_id] = {
            "task_id": current_task_id,
            "success": False,
            "output_layers": [],
            "execution_result": None,
            "error": str(e),
        }

        return {
            **state,
            "error_message": str(e),
            "task_results": task_results,
            "_last_was_geoprocessing": True,
            "_retry_count": state.get("_retry_count", 0),
        }


def should_continue_multi_tasks(state: ProcessingState) -> str:
    """
    Decide what to do after task execution or error:
    - 'error_analysis': Task failed, analyze and potentially retry
    - 'next_task': More tasks remain, move to next
    - 'finalize': All done or error unrecoverable, finalize workflow
    """
    error = state.get("error_message")
    current_idx = state.get("current_task_index", 0)
    tasks = state.get("task_queue", [])
    retry_count = state.get("_retry_count", 0)
    was_geoprocessing = state.get("_last_was_geoprocessing", False)

    # If there's an error, decide on retry vs continue
    if error:
        _logger.info(
            f"Task error detected, retry_count={retry_count}, geoprocessing={was_geoprocessing}"
        )

        # For geoprocessing errors: allow retry up to 2 times
        if was_geoprocessing and retry_count < 2:
            _logger.info(
                f"Geoprocessing error - retrying (attempt {retry_count + 1}/2)"
            )
            return "error_analysis"  # Routes to error_analysis, which can lead to select node

        # For LLM errors or max retries reached: skip to finalize
        _logger.info("Max retries or LLM error - skipping to finalize")
        return "finalize"

    # No error: check if more tasks remain
    if current_idx + 1 < len(tasks):
        _logger.info(f"Moving to next task ({current_idx + 2}/{len(tasks)})")
        return "next_task"
    else:
        _logger.info(f"All {len(tasks)} tasks completed, finalizing workflow")
        return "finalize"


def next_task_node(state: ProcessingState) -> ProcessingState:
    """Prepare for next task execution."""
    _logger.info("NODE: next_task_node START")

    current_idx = state.get("current_task_index", 0)

    # Reset single-task fields for next iteration
    result = {
        **state,
        "current_task_index": current_idx + 1,
        "algorithm_candidates": None,
        "selected_algorithm": None,
        "algorithm_metadata": None,
        "gathered_parameters": None,
        "execution_result": None,
        "is_current_task_processing": None,
        "_current_task_id": None,
        "_previous_outputs": state.get("_previous_outputs", {}),
        "error_message": None,
    }

    _logger.info("NODE: next_task_node END")
    return result


def error_analysis_node(state: ProcessingState) -> ProcessingState:
    """
    Analyze a failed task and route appropriately.

    For geoprocessing errors: Can retry up to 2 times by returning to select node
    For LLM errors: Continue to next task
    After 2 retries: Proceed to finalization
    """
    _logger.info("=" * 80)
    _logger.info("NODE: error_analysis_node START")

    error = state.get("error_message", "Unknown error")
    query = state.get("user_query", "")
    current_task_idx = state.get("current_task_index", 0)
    tasks = state.get("task_queue", [])
    current_task = tasks[current_task_idx] if current_task_idx < len(tasks) else {}
    retry_count = state.get("_retry_count", 0)
    was_processing_task = state.get("_last_was_geoprocessing", False)

    _logger.info(f"Analyzing error (retry {retry_count}): {error}")
    _logger.info(
        f"Task type: {'geoprocessing' if was_processing_task else 'LLM-based'}"
    )

    # Build context for error analysis
    messages = [
        SystemMessage(content=MULTI_STEP_ERROR_ANALYSIS_PROMPT),
        HumanMessage(
            content=(
                f"Failed Task: {current_task.get('operation', 'Unknown')}\n"
                f"Error: {error}\n"
                f"Retry attempt: {retry_count}\n"
                f"Task type: {'Geoprocessing' if was_processing_task else 'LLM-based'}\n\n"
                f"User Query: {query}\n"
                f"Algorithm: {state.get('selected_algorithm', 'Unknown')}\n"
                f"Please analyze the error and suggest solutions."
            )
        ),
    ]

    try:
        error_structured_llm = llm.with_structured_output(ErrorAnalysis)
        analysis: ErrorAnalysis = error_structured_llm.invoke(messages)

        _logger.info(f"Error diagnosis: {analysis.diagnosis}")

        # Store error analysis
        state["_error_analysis"] = {
            "diagnosis": analysis.diagnosis,
            "missing_info": analysis.missing_info,
            "user_suggestions": analysis.user_suggestions,
            "partial_results": analysis.partial_results,
            "example_query_fix": analysis.example_query_fix,
        }
        state["_retry_count"] = retry_count + 1

    except Exception as e:
        _logger.warning(f"Error analysis failed: {str(e)}")
        state["_error_analysis"] = {
            "diagnosis": f"Task failed: {error}",
            "missing_info": ["Unable to determine"],
            "user_suggestions": ["Please check the error and try a different approach"],
            "partial_results": "",
            "example_query_fix": "Unable to suggest",
        }
        state["_retry_count"] = retry_count + 1

    _logger.info("NODE: error_analysis_node END")
    return state


def finalize_multi_step_node(state: ProcessingState) -> ProcessingState:
    """
    Prepare final summary for multi-step workflow.

    Summarizes all completed tasks and their outputs.
    """
    _logger.info("=" * 80)
    _logger.info("NODE: finalize_multi_step_node START")

    task_results = state.get("task_results", {})
    tasks = state.get("task_queue", [])
    completed_tasks = state.get("completed_tasks", 0)
    error = state.get("error_message")
    error_analysis = state.get("_error_analysis")

    # Build summary context
    summary_data = {
        "total_tasks": len(tasks),
        "completed_tasks": completed_tasks,
        "has_error": bool(error),
        "error_message": error,
        "error_analysis": error_analysis,
        "task_results": task_results,
        "tasks": tasks,
    }

    _logger.debug(f"Summary data: {json.dumps(summary_data, indent=2, default=str)}")

    # Build detailed task summary with operations
    task_summary = []
    for task_id in sorted(task_results.keys()):
        task = next(
            (
                t
                for idx, t in enumerate(tasks, start=1)
                if t.get("task_id") == task_id or idx == task_id
            ),
            None,
        )
        result = task_results[task_id]
        if task:
            operation = task.get("operation", "Unknown task")
            if result.get("success"):
                output_info = ""
                if result.get("output_layers"):
                    output_info = f" → Output: {', '.join(result['output_layers'])}"
                elif result.get("execution_result"):
                    output_info = f" → {str(result['execution_result'])[:60]}..."
                task_summary.append(f"✓ {operation}{output_info}")
            else:
                task_summary.append(
                    f"✗ {operation} (Error: {result.get('error', 'Unknown')})"
                )

    # Create LLM prompt with rich context
    if error and error_analysis:
        summary_prompt = (
            f"User's request: {state.get('user_query', 'Unknown')}\n\n"
            f"Workflow Status: FAILED\n\n"
            f"Tasks executed:\n"
            + "\n".join(task_summary)
            + f"\n\nError: {error_analysis.get('diagnosis')}\n"
            f"Suggestions: {', '.join(error_analysis.get('user_suggestions', []))}\n\n"
            f"Please provide a brief, user-friendly message explaining what happened and what the user can try next."
        )
    else:
        summary_prompt = (
            f"User's request: {state.get('user_query', 'Unknown')}\n\n"
            f"Workflow Status: SUCCESS\n\n"
            f"Tasks completed:\n"
            + "\n".join(task_summary)
            + f"\n\nPlease provide a brief, user-friendly message summarizing what was accomplished."
        )

    # Invoke LLM to generate user-friendly summary
    try:
        summary_messages = [
            SystemMessage(
                content="You are GeoAgent, a helpful geospatial assistant. Generate a concise, friendly summary of the workflow result (max 3-4 sentences)."
            ),
            HumanMessage(content=summary_prompt),
        ]
        summary_response = llm.invoke(summary_messages)
        summary_content = (
            summary_response.content
            if hasattr(summary_response, "content")
            else str(summary_response)
        )

        new_messages = state.get("messages", []) + [AIMessage(content=summary_content)]
    except Exception as e:
        _logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
        # Fallback to simple summary
        summary_text = (
            f"✓ Completed {completed_tasks}/{len(tasks)} tasks successfully."
            if not error
            else f"✗ Workflow failed at task {state.get('current_task_index', 0) + 1}"
        )
        new_messages = state.get("messages", []) + [AIMessage(content=summary_text)]

    _logger.info("NODE: finalize_multi_step_node END")
    return {
        **state,
        "messages": new_messages,
    }


def execute_llm_task_node(state: ProcessingState) -> ProcessingState:
    """Execute non-geoprocessing task using LLM with tool support."""
    _logger.info("=" * 80)
    _logger.info("NODE: execute_llm_task_node START")

    if not state.get("task_queue") or state.get("current_task_index") is None:
        return state

    current_idx = state["current_task_index"]
    tasks = state["task_queue"]
    current_task = tasks[current_idx]
    current_task_id = current_task.get("task_id", current_idx + 1)
    operation = current_task.get("operation", "")

    # Build task context
    previous_outputs = state.get("_previous_outputs", {})
    context = ""
    if previous_outputs:
        context = "\n\nPrevious task outputs:\n" + "\n".join(
            [f"- {k}: {v}" for k, v in previous_outputs.items()]
        )

    try:
        layers = list_qgis_layers.invoke({}).get("layers", [])
        layers_context = f"\n\nAvailable layers: {', '.join(layers)}"
    except Exception:
        layers_context = ""

    task_message = f"""Execute this task:
{operation}{context}{layers_context}

Provide a clear response about what you did."""

    messages = state.get("messages", []) + [HumanMessage(content=task_message)]

    try:
        bound_llm = llm.bind_tools(list(TOOLS.values()))
        response = bound_llm.invoke(messages)

        # Execute any tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = []
            for call in response.tool_calls:
                tool = TOOLS.get(call["name"])
                if tool:
                    try:
                        result = tool.invoke(call["args"])
                        content = result if isinstance(result, str) else str(result)
                        tool_messages.append(
                            ToolMessage(content=content, tool_call_id=call["id"])
                        )
                    except Exception as e:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}", tool_call_id=call["id"]
                            )
                        )

            # Get final response after tools
            final_response = llm.invoke(messages + [response] + tool_messages)
            response_content = (
                final_response.content
                if hasattr(final_response, "content")
                else str(final_response)
            )
        else:
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

        # Store task result
        task_results = dict(state.get("task_results", {}))
        task_results[current_task_id] = {
            "task_id": current_task_id,
            "success": True,
            "output_layers": [],
            "execution_result": response_content,
            "error": None,
        }

        completed_tasks = state.get("completed_tasks", 0) + 1

        _logger.info("NODE: execute_llm_task_node END -> success")
        return {
            **state,
            "task_results": task_results,
            "completed_tasks": completed_tasks,
            "_last_was_geoprocessing": False,
            "error_message": None,
            "_retry_count": 0,
        }

    except Exception as e:
        _logger.error(f"LLM task failed: {str(e)}", exc_info=True)
        existing_task_results = dict(state.get("task_results", {}))
        task_results = {
            **existing_task_results,
            current_task_id: {
                "task_id": current_task_id,
                "success": False,
                "output_layers": [],
                "execution_result": None,
                "error": str(e),
            },
        }
        return {
            **state,
            "error_message": str(e),
            "task_results": task_results,
            "_last_was_geoprocessing": False,
            "_retry_count": state.get("_retry_count", 0),
        }


def should_retry_or_finalize(state: ProcessingState) -> str:
    """
    After error analysis, decide whether to:
    - 'select': Retry geoprocessing task with different algorithm
    - 'finalize': Give up on this task and finalize
    """
    retry_count = state.get("_retry_count", 0)
    was_geoprocessing = state.get("_last_was_geoprocessing", False)

    # For geoprocessing with retries remaining, go back to algorithm selection
    if was_geoprocessing and retry_count < 2:
        _logger.info(
            f"Retrying geoprocessing (attempt {retry_count}/2) - returning to algorithm selection"
        )
        return "select"

    # Otherwise, finalize
    _logger.info("No more retries or non-geoprocessing task - finalizing")
    return "finalize"


def build_multi_step_processing_graph(llm_instance) -> Any:
    """
    Build multi-step processing graph with recursive sub-graph support.

    Architecture:
    1. Route: Determine if processing task
    2. Decompose: Break query into subtasks
    3. Task Loop:
       a. Analyze dependencies
       b. Discover → Select → Inspect → Gather → Execute (single-task sub-graph)
       c. Check if more tasks remain
    4. Error handling or finalization
    5. LLM summary
    """
    global llm
    llm = llm_instance

    _logger.info("Building multi-step processing graph...")

    graph = StateGraph(ProcessingState)

    # Entry routing node
    graph.add_node("route", route_node)

    # Multi-task orchestration
    graph.add_node("decompose", decompose_tasks_node)
    graph.add_node("analyze_deps", analyze_dependencies_node)

    # Geoprocessing pipeline
    graph.add_node("discover", discover_algorithms_node_multi)
    graph.add_node("select", select_algorithm_node_multi)
    graph.add_node("inspect", inspect_parameters_node_multi)
    graph.add_node("gather", gather_parameters_node_multi)
    graph.add_node(
        "execute", execute_node_multi, retry_policy=RetryPolicy(max_attempts=2)
    )

    # Task execution and finalization
    graph.add_node("execute_llm_task", execute_llm_task_node)
    graph.add_node("next_task", next_task_node)
    graph.add_node("error_analysis", error_analysis_node)
    graph.add_node("finalize", finalize_multi_step_node)

    # Entry: route directly to decompose (always decompose first)
    graph.set_entry_point("route")
    graph.add_edge("route", "decompose")

    # Multi-step workflow flow
    graph.add_edge("decompose", "analyze_deps")

    # Route each task to geoprocessing or LLM execution
    graph.add_conditional_edges(
        "analyze_deps",
        should_route_task,
        {
            "processing": "discover",
            "llm_task": "execute_llm_task",
        },
    )

    # Geoprocessing pipeline
    graph.add_edge("discover", "select")
    graph.add_edge("select", "inspect")
    graph.add_edge("inspect", "gather")
    graph.add_edge("gather", "execute")

    # Both execution paths converge to continuation check
    graph.add_conditional_edges(
        "execute",
        should_continue_multi_tasks,
        {
            "next_task": "next_task",
            "error_analysis": "error_analysis",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "execute_llm_task",
        should_continue_multi_tasks,
        {
            "next_task": "next_task",
            "error_analysis": "error_analysis",
            "finalize": "finalize",
        },
    )

    # Error analysis: decide whether to retry (select) or finalize
    graph.add_conditional_edges(
        "error_analysis",
        should_retry_or_finalize,
        {
            "select": "select",
            "finalize": "finalize",
        },
    )

    # Task loop
    graph.add_edge("next_task", "analyze_deps")

    # Finalization ends workflow
    graph.add_edge("finalize", END)

    _logger.info("Multi-step processing graph built successfully")
    return graph


__all__ = [
    "build_multi_step_processing_graph",
]
