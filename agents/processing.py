# -*- coding: utf-8 -*-
"""
Geoprocessing agent graph: orchestrates algorithm selection, parameter gathering, and execution.

Workflow:
1. Route: decide if user query needs processing or not
2. DiscoverAlgorithms: call list_processing_algorithms to find candidates
3. SelectAlgorithm: use LLM with structured output to pick best algorithm
4. InspectParameters: get detailed parameter definitions
5. GatherParameters: use LLM to extract & build parameter dict from query
6. Execute: run execute_processing with final parameters
7. Finalize: summarize results to user
"""
import json
import logging
import os
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..tools import (
    list_processing_algorithms,
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
)
from .states import ProcessingState


# ─────────────────────────────────────────────────────────────────────────────
# Structured Output Models
# ─────────────────────────────────────────────────────────────────────────────


class SelectedAlgorithm(BaseModel):
    """Structured output for algorithm selection."""

    algorithm_id: str = Field(description="The algorithm ID (e.g., 'native:buffer')")
    algorithm_name: str = Field(description="The human-readable algorithm name")
    reasoning: str = Field(
        description="Why this algorithm was selected for the user's task"
    )
    confidence: float = Field(
        description="Confidence score (0.0-1.0) for this selection", ge=0.0, le=1.0
    )


class GatheredParameters(BaseModel):
    """Structured output for parameter gathering."""

    parameters: Dict[str, Any] = Field(
        description="Dictionary of algorithm parameters with their values"
    )
    inferred_fields: List[str] = Field(
        description="List of parameter names that were inferred rather than explicitly stated"
    )
    notes: str = Field(
        description="Any notes or assumptions about parameter extraction"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
def _setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logger for processing graph with both console and file handlers.

    Args:
        log_file: Path to log file. If None, creates one in temp directory.

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("GeoAgent.Processing")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Determine log file path
    if log_file is None:
        # Create logs folder in plugin directory
        plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(plugin_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"processing_{timestamp}.log")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Create global logger instance
_logger = _setup_logger()


def set_processing_log_file(log_file: str) -> None:
    """
    Set a custom log file path for processing graph output.

    Args:
        log_file: Absolute path to desired log file
    """
    global _logger
    _logger = _setup_logger(log_file)


def _infer_missing_parameters(
    query: str, missing_params: List[str], metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Attempt to infer missing required parameters from the user query.

    Args:
        query: User's original query
        missing_params: List of parameter names that are missing
        metadata: Algorithm metadata containing parameter definitions

    Returns:
        Dict of inferred parameter values
    """
    inferred = {}
    query_lower = query.lower()

    # Get list of available layers
    try:
        layers_result = list_qgis_layers.invoke({})
        # Extract layer names from the result
        layer_names = re.findall(r"^(\d+)\.\s+(\w+)", layers_result, re.MULTILINE)
        available_layers = [name for _, name in layer_names]
    except Exception:
        available_layers = []

    for param_name in missing_params:
        # Find parameter definition
        param_def = next(
            (p for p in metadata.get("parameters", []) if p["name"] == param_name), None
        )
        if not param_def:
            continue

        # Try to infer INPUT layer
        if param_name == "INPUT" or "input" in param_name.lower():
            # Look for layer names mentioned in query
            for layer_name in available_layers:
                if layer_name.lower() in query_lower:
                    inferred[param_name] = layer_name
                    _logger.debug(
                        f"Inferred {param_name} = {layer_name} (found in query)"
                    )
                    break
            # If not found by name, use the first available layer as fallback
            if param_name not in inferred and available_layers:
                inferred[param_name] = available_layers[0]
                _logger.debug(
                    f"Inferred {param_name} = {available_layers[0]} (first available layer)"
                )

        # Try to infer numeric parameters (DISTANCE, BUFFER, etc.)
        elif (
            "distance" in param_name.lower()
            or "buffer" in param_name.lower()
            or "radius" in param_name.lower()
        ):
            numbers = re.findall(r"(\d+(?:\.\d+)?)", query)
            if numbers:
                # Try to convert to appropriate type
                try:
                    value = float(numbers[0]) if "." in numbers[0] else int(numbers[0])
                    inferred[param_name] = value
                    _logger.debug(
                        f"Inferred {param_name} = {value} (extracted from query)"
                    )
                except ValueError:
                    pass

    return inferred


def build_processing_graph(llm) -> any:
    """
    Build and compile a LangGraph processing workflow.

    The graph orchestrates:
    - Routing (processing vs. general)
    - Algorithm discovery and selection
    - Parameter inspection and gathering
    - Execution and result handling

    Args:
        llm: Language model instance (should support invoke and tool binding)

    Returns:
        Compiled LangGraph application ready to process geoprocessing queries
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 1: Route
    # Decide whether this is a processing task or should use other tools/LLM response
    # ─────────────────────────────────────────────────────────────────────────────
    def route_node(state: ProcessingState) -> ProcessingState:
        """Determine if user query requires geoprocessing or general assistance."""
        _logger.info("=" * 80)
        _logger.info("NODE: route_node START")

        last_msg = state["messages"][-1]
        user_query = (
            last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)
        )

        _logger.debug(f"User query: {user_query}")

        # Call routing LLM
        messages = [
            SystemMessage(content=PROCESSING_ROUTING_PROMPT),
            HumanMessage(content=f"User query: {user_query}"),
        ]
        response = llm.invoke(messages)

        _logger.debug(f"LLM response: {response.content}")

        try:
            # Parse JSON from LLM response
            result = json.loads(response.content)
            is_processing = result.get("is_processing_task", False)
            reason = result.get("reason", "N/A")
            _logger.info(
                f"Routing decision: is_processing_task={is_processing}, reason={reason}"
            )
        except (json.JSONDecodeError, AttributeError):
            # If LLM doesn't return valid JSON, heuristic check for keywords
            _logger.warning("LLM response was not valid JSON, using heuristic fallback")
            query_lower = user_query.lower()
            processing_keywords = [
                "buffer",
                "clip",
                "dissolve",
                "union",
                "intersection",
                "difference",
                "statistics",
                "raster",
                "vector",
                "simplify",
                "interpolate",
                "reproject",
                "merge",
            ]
            is_processing = any(kw in query_lower for kw in processing_keywords)
            _logger.info(f"Heuristic result: is_processing_task={is_processing}")

        _logger.info(f"NODE: route_node END -> is_processing_task={is_processing}")
        return {
            "messages": state["messages"],
            "user_query": user_query,
            "is_processing_task": is_processing,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 2: Discover Algorithms
    # Find all algorithms that might match the user's query
    # ─────────────────────────────────────────────────────────────────────────────
    def discover_algorithms_node(state: ProcessingState) -> ProcessingState:
        """Search for algorithms matching the user query."""
        _logger.info("=" * 80)
        _logger.info("NODE: discover_algorithms_node START")

        if not state.get("is_processing_task"):
            _logger.info("Not a processing task, skipping algorithm discovery")
            _logger.info("NODE: discover_algorithms_node END -> SKIPPED")
            return state

        query = state["user_query"]
        _logger.debug(f"Discovering algorithms for: {query}")

        try:
            # Use find_processing_algorithm to get relevant candidates
            # LLM will select the best match in the next node
            result = find_processing_algorithm.invoke({"query": query, "limit": 500})
            matches = result.get("matches", [])
            search_term = result.get("search_term", query)
            candidates = [
                {"id": m["id"], "name": m["name"], "provider": m["provider"]}
                for m in matches
            ]
            _logger.info(
                f"Discovered {len(candidates)} algorithm candidates using search term: '{search_term}'"
            )
            for i, cand in enumerate(candidates[:5], 1):  # Log first 5
                _logger.debug(f"  {i}. {cand['id']} ({cand['name']})")
        except Exception as e:
            _logger.error(f"Error discovering algorithms: {str(e)}", exc_info=True)
            return {
                "messages": state["messages"]
                + [AIMessage(content=f"Error discovering algorithms: {str(e)}")],
                "error_message": str(e),
            }

        _logger.info(
            f"NODE: discover_algorithms_node END -> {len(candidates)} candidates discovered"
        )
        return {
            "user_query": state["user_query"],
            "algorithm_candidates": candidates,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 3: Select Algorithm
    # Use LLM with structured output to pick the best algorithm from candidates
    # ─────────────────────────────────────────────────────────────────────────────
    def select_algorithm_node(state: ProcessingState) -> ProcessingState:
        """Use LLM with structured output to select the best matching algorithm."""
        _logger.info("=" * 80)
        _logger.info("NODE: select_algorithm_node START")

        if not state.get("algorithm_candidates"):
            _logger.info("No candidates, skipping selection")
            _logger.info("NODE: select_algorithm_node END -> SKIPPED")
            return state

        candidates = state["algorithm_candidates"]
        query = state["user_query"]

        _logger.debug(
            f"Selecting from {len(candidates)} candidates using structured output"
        )

        # Format candidates as readable list
        # candidates_text = "\n".join(
        #     [f"  - {c['id']}: {c['name']}" for c in candidates[:500]]  # Limit to 500
        # )
        candidates_text = "\n".join([f"  - {c['id']}" for c in candidates])

        _logger.debug(f"Candidates:\n{candidates_text}")

        # Bind structured output to LLM
        try:
            structured_llm = llm.with_structured_output(SelectedAlgorithm)
        except Exception as e:
            _logger.warning(
                f"Could not create structured LLM: {e}, falling back to JSON"
            )
            structured_llm = llm

        messages = [
            SystemMessage(content=PROCESSING_ALGORITHM_SELECTION_PROMPT),
            HumanMessage(
                content=f"User query: {query}\n\nAvailable algorithms:\n{candidates_text}"
            ),
        ]

        try:
            response = structured_llm.invoke(messages)

            # Handle both structured and fallback responses
            if isinstance(response, SelectedAlgorithm):
                selected_alg = response.algorithm_id
                confidence = response.confidence
                reasoning = response.reasoning
                _logger.debug(
                    f"LLM response (structured): algorithm_id={selected_alg}, confidence={confidence}, reasoning={reasoning}"
                )
            else:
                # Fallback: parse as text/JSON
                _logger.debug(
                    f"LLM response (text): {response.content if hasattr(response, 'content') else response}"
                )
                try:
                    result = json.loads(
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    selected_alg = result.get("algorithm_id")
                    confidence = result.get("confidence", 0)
                    reasoning = result.get("reasoning", "N/A")
                except (json.JSONDecodeError, TypeError):
                    # Last resort: pick first candidate
                    _logger.warning(
                        "Could not parse LLM response, using first candidate"
                    )
                    selected_alg = candidates[0]["id"] if candidates else None
                    confidence = 0.5
                    reasoning = "Fallback to first candidate"

            _logger.info(
                f"Selected algorithm: {selected_alg} (confidence: {confidence}, reasoning: {reasoning})"
            )
        except Exception as e:
            _logger.error(f"Error selecting algorithm: {str(e)}", exc_info=True)
            # Fallback: pick first candidate
            _logger.warning("Using first candidate as fallback")
            selected_alg = candidates[0]["id"] if candidates else None
            confidence = 0.5
            reasoning = "Error in selection, using first candidate"

        if not selected_alg:
            _logger.error("No algorithm could be selected")
            return {
                "user_query": state["user_query"],
                "error_message": "Could not select any algorithm",
            }

        _logger.info(f"NODE: select_algorithm_node END -> selected={selected_alg}")
        return {
            "user_query": state["user_query"],
            "selected_algorithm": selected_alg,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 4: Inspect Parameters
    # Get full parameter definitions for the selected algorithm
    # ─────────────────────────────────────────────────────────────────────────────
    def inspect_parameters_node(state: ProcessingState) -> ProcessingState:
        """Fetch parameter definitions for the selected algorithm."""
        _logger.info("=" * 80)
        _logger.info("NODE: inspect_parameters_node START")

        if not state.get("selected_algorithm"):
            _logger.info("No selected algorithm, skipping inspection")
            _logger.info("NODE: inspect_parameters_node END -> SKIPPED")
            return state

        algorithm = state["selected_algorithm"]
        _logger.debug(f"Inspecting parameters for: {algorithm}")

        try:
            metadata = get_algorithm_parameters.invoke({"algorithm": algorithm})
            _logger.info(f"Retrieved metadata for {algorithm}")
            _logger.debug(f"Algorithm name: {metadata.get('name')}")
            _logger.debug(f"Provider: {metadata.get('provider')}")
            params = metadata.get("parameters", [])
            _logger.debug(f"Parameters ({len(params)} total):")
            for p in params:
                required = "optional" if p.get("optional") else "required"
                _logger.debug(
                    f"  - {p['name']} ({p['type']}: {p.get('default', 'N/A')}) [{required}]: {p.get('description', 'N/A')}"
                )
        except Exception as e:
            _logger.error(f"Error inspecting parameters: {str(e)}", exc_info=True)
            return {
                "error_message": str(e),
            }

        _logger.info(f"NODE: inspect_parameters_node END -> metadata retrieved")
        return {
            "user_query": state["user_query"],
            "selected_algorithm": state["selected_algorithm"],
            "algorithm_metadata": metadata,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 5: Gather Parameters
    # Use LLM with structured output to extract parameter values from user query
    # ─────────────────────────────────────────────────────────────────────────────
    def gather_parameters_node(state: ProcessingState) -> ProcessingState:
        """Use LLM to extract and build the parameter dictionary."""
        _logger.info("=" * 80)
        _logger.info("NODE: gather_parameters_node START")

        if not state.get("algorithm_metadata"):
            _logger.info("No algorithm metadata, skipping parameter gathering")
            _logger.info("NODE: gather_parameters_node END -> SKIPPED")
            return state

        query = state["user_query"]
        metadata = state["algorithm_metadata"]
        algorithm_name = metadata.get("name", "")

        _logger.debug(f"Gathering parameters for: {algorithm_name}")

        # Get available layers for reference
        try:
            layers_info = list_qgis_layers.invoke({})
            _logger.debug(f"Available layers:\n{layers_info}")
        except Exception as e:
            _logger.warning(f"Could not retrieve layers: {e}")
            layers_info = "No layers available"

        # Format parameters for LLM
        params_text = "\n".join(
            [
                f"  - {p['name']} ({p['type']}, default: {p.get('default', 'N/A')}): {p['description']}"
                f" [optional: {p.get('optional', False)}]"
                for p in metadata.get("parameters", [])
            ]
        )

        _logger.debug(f"Parameter definitions:\n{params_text}")

        messages = [
            SystemMessage(content=PROCESSING_PARAMETER_GATHERING_PROMPT),
            HumanMessage(
                content=(
                    f"User query: {query}\n\n"
                    f"Algorithm: {algorithm_name}\n\n"
                    f"Available layers:\n{layers_info}\n\n"
                    f"Parameters:\n{params_text}\n\n"
                    f"Please provide the parameter values in JSON format."
                )
            ),
        ]

        # Try to use structured output
        try:
            structured_llm = llm.with_structured_output(GatheredParameters)
        except Exception as e:
            _logger.warning(
                f"Could not create structured LLM: {e}, falling back to JSON"
            )
            structured_llm = llm

        try:
            response = structured_llm.invoke(messages)

            # Handle both structured and fallback responses
            if isinstance(response, GatheredParameters):
                parameters = response.parameters
                inferred_fields = response.inferred_fields
                notes = response.notes

                # if OUTPUT is not in parameters, add it as None (to use default temp output)
                if "OUTPUT" not in parameters:
                    parameters["OUTPUT"] = "TEMPORARY_OUTPUT"
            else:
                # Fallback: parse as text/JSON
                _logger.debug(
                    f"LLM response (text): {response.content if hasattr(response, 'content') else response}"
                )
                result = json.loads(
                    response.content if hasattr(response, "content") else str(response)
                )
                parameters = result.get("parameters", {})
                inferred_fields = result.get("inferred_fields", [])
                notes = result.get("notes", "")

            _logger.info(f"Extracted parameters: {parameters}")

            # if the parameters still missing some required fields, try to add those with default values
            for param in metadata.get("parameters", []):
                _logger.debug(f"Checking parameter: {param['name']}")

                name = param["name"]
                current_val = parameters.get(name)
                default_value = param.get("default")

                # Case 1: Key exists but is None OR Case 2: Required key is missing entirely
                if (name in parameters and current_val is None) or (
                    not param.get("optional") and name not in parameters
                ):
                    if default_value is not None:
                        parameters[name] = default_value
                        _logger.debug(
                            f"Assigned default value {default_value} to parameter {name}"
                        )
                    else:
                        # Optional: Handle cases where a mandatory param is missing but no default exists
                        if not param.get("optional"):
                            _logger.debug(
                                f"Required parameter {name} is missing and has no default!"
                            )

            if notes:
                _logger.debug(f"Notes: {notes}")
            if inferred_fields:
                _logger.debug(f"Inferred fields: {inferred_fields}")

            # Validate that required parameters are present
            required_params = [
                p["name"]
                for p in metadata.get("parameters", [])
                if not p.get("optional")
            ]
            missing_required = [
                p
                for p in required_params
                if p not in parameters or parameters[p] is None
            ]

            print(
                "\n\nMissing required parameters after initial extraction:",
                missing_required,
            )

            if missing_required:
                error_msg = f"Missing required parameters: {', '.join(missing_required)}. LLM extracted: {parameters}"
                _logger.warning(error_msg)
                _logger.warning("Attempting to infer missing parameters from query...")

                # Try to infer missing parameters from query
                inferred = _infer_missing_parameters(query, missing_required, metadata)
                parameters.update(inferred)
                _logger.info(f"After inference: {parameters}")

                # Check again
                still_missing = [
                    p
                    for p in missing_required
                    if p not in parameters or parameters[p] is None
                ]
                if still_missing:
                    error_msg = f"Could not infer required parameters: {', '.join(still_missing)}"
                    _logger.error(error_msg)
                    return {
                        "error_message": error_msg,
                    }

        except (json.JSONDecodeError, AttributeError, ValueError, TypeError) as e:
            _logger.error(f"Error extracting parameters: {str(e)}", exc_info=True)
            return {
                "error_message": str(e),
            }

        _logger.info(
            f"NODE: gather_parameters_node END -> {len(parameters)} parameters gathered"
        )
        return {
            "user_query": state["user_query"],
            "selected_algorithm": state["selected_algorithm"],
            "gathered_parameters": parameters,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 6: Execute Processing
    # Run the algorithm with gathered parameters
    # ─────────────────────────────────────────────────────────────────────────────
    def execute_node(state: ProcessingState) -> ProcessingState:
        """Execute the processing algorithm."""
        _logger.info("=" * 80)
        _logger.info("NODE: execute_node START")

        if (
            not state.get("selected_algorithm")
            or state.get("gathered_parameters") is None
        ):
            _logger.info("Missing algorithm or parameters, skipping execution")
            _logger.info("NODE: execute_node END -> SKIPPED")
            return state

        algorithm = state["selected_algorithm"]
        parameters = state["gathered_parameters"]

        _logger.info(f"Executing algorithm: {algorithm}")
        _logger.debug(f"Parameters: {json.dumps(parameters, indent=2)}")

        try:
            result = execute_processing.invoke(
                {"algorithm": algorithm, "parameters": parameters}
            )
            _logger.info(f"Algorithm execution successful")
            _logger.debug(f"Result: {json.dumps(result, indent=2, default=str)}")
        except Exception as e:
            _logger.error(f"Processing execution failed: {str(e)}", exc_info=True)
            return {
                "error_message": str(e),
            }

        _logger.info(f"NODE: execute_node END -> execution successful")
        return {
            "selected_algorithm": algorithm,
            "gathered_parameters": parameters,
            "execution_result": result,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 7: Finalize
    # Summarize results and return to user
    # ─────────────────────────────────────────────────────────────────────────────
    def finalize_node(state: ProcessingState) -> ProcessingState:
        """Prepare a summary prompt and let the LLM generate the final message."""
        _logger.info("=" * 80)
        _logger.info("NODE: finalize_node START")
        algo = state.get("selected_algorithm")
        params = state.get("gathered_parameters", {})
        result = state.get("execution_result")
        err = state.get("error_message")

        # Build a compact, structured context for the LLM
        context_payload = {
            "algorithm": algo,
            "parameters": params,
            "result": result,
            "error": err,
        }

        system = SystemMessage(
            content=(
                "You are GeoAgent. Write a clear, concise user-facing summary of this run. "
                "If there was an error, explain it and suggest the next step. Otherwise include: "
                "- Algorithm friendly name/id\n- Key parameters with human-friendly units\n- Where outputs were created/loaded. "
                "Keep under 6 lines. Do not call any tools. Do not ask follow-up questions."
            )
        )
        human = HumanMessage(
            content=(
                "Here is the structured context of the run (JSON):\n" + json.dumps(context_payload, default=str)
            )
        )

        new_messages = state.get("messages", []) + [system, human]

        _logger.info("NODE: finalize_node END -> handing off to LLM for summary")
        _logger.info("=" * 80)

        return {
            "messages": new_messages,
            "execution_result": result,
            "error_message": err,
            "selected_algorithm": algo,
            "gathered_parameters": params,
        }

    # ─────────────────────────────────────────────────────────────────────────────
    # Node 8: tools
    # ────────────────────────────────────────────────────────────────────────────
    def tool_node(state: ProcessingState) -> ProcessingState:
        last_msg = state["messages"][-1]
        tool_messages = []
        for call in getattr(last_msg, "tool_calls", []) or []:
            tool_inst = TOOLS[call["name"]]
            result = tool_inst.invoke(call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        return {"messages": state["messages"] + tool_messages}

    def llm_node(state: ProcessingState) -> ProcessingState:

        # connect tools to llm
        try:
            bound_llm = llm.bind_tools(list(TOOLS.values()))
        except Exception:
            bound_llm = llm

        response = bound_llm.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    # ─────────────────────────────────────────────────────────────────────────────
    # Conditional edges to route based on task type
    # ─────────────────────────────────────────────────────────────────────────────
    def should_continue_processing(state: ProcessingState) -> str:
        """Decide whether to continue processing or end."""
        if state.get("is_processing_task") and not state.get("error_message"):
            return "continue"
        return "end"

    def should_use_tools(state: ProcessingState) -> str:
        """Route LLM output: run tools if requested, otherwise end."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
            return "tools"
        return "end"

    # ─────────────────────────────────────────────────────────────────────────────
    # Build the graph
    # ─────────────────────────────────────────────────────────────────────────────
    graph = StateGraph(ProcessingState)

    graph.add_node("route", route_node)
    graph.add_node("discover_algorithms", discover_algorithms_node)
    graph.add_node("select_algorithm", select_algorithm_node)
    graph.add_node("inspect_parameters", inspect_parameters_node)
    graph.add_node("gather_parameters", gather_parameters_node)
    graph.add_node("execute", execute_node, retry_policy=RetryPolicy(max_attempts=2))
    graph.add_node("finalize", finalize_node)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    # Edges: route → conditional
    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        should_continue_processing,
        {"continue": "discover_algorithms", "end": "llm"},
    )

    graph.add_conditional_edges(
        "llm",
        should_use_tools,
        {"tools": "tools", "end": END},
    )

    # Linear chain for processing path
    graph.add_edge("discover_algorithms", "select_algorithm")
    graph.add_edge("select_algorithm", "inspect_parameters")
    graph.add_edge("inspect_parameters", "gather_parameters")
    graph.add_edge("gather_parameters", "execute")
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", "llm")
    graph.add_edge("tools", "llm")

    # End
    graph.add_edge("llm", END)
    return graph.compile(checkpointer=MemorySaver())


def invoke_processing_app(
    app, thread_id: str, messages: List[BaseMessage]
) -> Dict[str, Any]:
    """
    Invoke the processing graph and return the final state and last message.

    Args:
        app: Compiled processing graph
        thread_id: Conversation thread ID for state persistence
        messages: List of conversation messages

    Returns:
        Dict with final state, last AI message, and processing metadata
    """
    state = {"messages": messages}
    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})

    return {
        "state": result,
        "last_message": result["messages"][-1] if result["messages"] else None,
        "algorithm": result.get("selected_algorithm"),
        "result": result.get("execution_result"),
        "error": result.get("error_message"),
    }


__all__ = [
    "build_processing_graph",
    "invoke_processing_app",
    "set_processing_log_file",
    "_setup_logger",
]
