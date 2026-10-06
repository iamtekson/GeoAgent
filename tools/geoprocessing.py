# -*- coding: utf-8 -*-
"""
Geoprocessing tools for QGIS operations.

Provides algorithm discovery (over the FULL processing registry, all providers),
parameter inspection, and execution with results loaded into the QGIS project.
"""
import html
import os
import re
from typing import Optional, List, Dict, Any, Tuple
from langchain_core.tools import tool
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
)

from ..config.constants import RASTER_EXTENSIONS
from ..utils.layer_matching import find_best_layer_match
from ..logger.processing_logger import get_processing_logger

_logger = get_processing_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm catalog (cached once per session; the registry rarely changes)
# ─────────────────────────────────────────────────────────────────────────────
_ALGORITHM_CATALOG: Optional[List[Dict[str, Any]]] = None

_STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "with", "and", "or", "in", "on", "by",
    "from", "layer", "layers", "using", "each", "all", "my", "this", "that",
    "create", "make", "new", "map", "file", "data",
}

# Small bonus layer of common GIS phrasing -> algorithm vocabulary. NOT meant
# to be exhaustive: the workflow passes LLM-generated `keywords` per task,
# which is the generic mechanism; this map just gives frequent phrasings a
# floor when no keywords are provided.
_SYNONYMS = {
    "merge": ["merge", "union", "dissolve"],
    "combine": ["union", "merge", "dissolve"],
    "join": ["join", "union"],
    "clip": ["clip", "mask", "extract"],
    "crop": ["clip", "mask"],
    "cut": ["clip"],
    "average": ["mean", "statistics", "zonal"],
    "statistics": ["statistics", "stats", "zonal"],
    "stats": ["statistics", "zonal"],
    "reproject": ["reproject", "warp", "crs", "transform"],
    "projection": ["reproject", "crs"],
    "distance": ["distance", "buffer", "proximity"],
    "simplify": ["simplify", "generalize", "smooth"],
    "interpolate": ["interpolate", "idw", "tin"],
    "slope": ["slope", "terrain"],
    "elevation": ["dem", "terrain", "elevation"],
    "centroid": ["centroid", "center"],
}


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _plain_help_text(alg: QgsProcessingAlgorithm, max_chars: int = 2000) -> str:
    """Plain-text algorithm help; shortHelpString() may contain HTML.

    This is where providers such as GRASS/SAGA document parameter semantics
    (e.g. that a stream threshold is in cells, not map units), which the bare
    parameter definitions don't carry.
    """
    try:
        text = alg.shortHelpString() or ""
    except Exception:
        return ""
    text = html.unescape(_HTML_TAG_RE.sub(" ", text))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


def get_algorithm_catalog(refresh: bool = False) -> List[Dict[str, Any]]:
    """Return all registered algorithms with searchable metadata, cached."""
    global _ALGORITHM_CATALOG
    if _ALGORITHM_CATALOG is not None and not refresh:
        return _ALGORITHM_CATALOG

    catalog: List[Dict[str, Any]] = []
    registry = QgsApplication.processingRegistry()
    for alg in registry.algorithms():
        try:
            tags = [str(t).lower() for t in (alg.tags() or [])]
        except Exception:
            tags = []
        try:
            description = alg.shortDescription() or ""
        except Exception:
            description = ""
        if not description:
            # GRASS/SAGA algorithms typically have no shortDescription; use
            # the first part of their help text so selection can see what
            # they do (only for empty ones to keep the one-off build cheap).
            description = _plain_help_text(alg, max_chars=160)
        catalog.append(
            {
                "id": alg.id(),
                "name": alg.displayName(),
                "provider": alg.provider().id(),
                "tags": tags,
                "description": description,
            }
        )
    _ALGORITHM_CATALOG = catalog
    return catalog


def _query_tokens(query: str) -> List[str]:
    """Tokenize a task description and expand with GIS synonyms."""
    words = re.findall(r"[a-z]+", query.lower())
    tokens = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    expanded = list(tokens)
    for t in tokens:
        expanded.extend(_SYNONYMS.get(t, []))
    return list(dict.fromkeys(expanded))  # dedupe, keep order


def _score_algorithm(tokens: List[str], entry: Dict[str, Any]) -> float:
    """Cheap lexical relevance of one catalog entry against query tokens."""
    name_words = set(re.findall(r"[a-z]+", entry["name"].lower()))
    alg_id = entry["id"].lower()
    tags = entry["tags"]
    desc = entry["description"].lower()

    score = 0.0
    for t in tokens:
        if t in name_words:
            score += 3.0
        elif any(t in w for w in name_words):
            score += 1.5
        if t in alg_id:
            score += 1.5
        if any(t == tag or t in tag for tag in tags):
            score += 2.0
        if desc and t in desc:
            score += 0.5
    # Slight preference for native algorithms on ties
    if score > 0 and entry["provider"] == "native":
        score += 0.5
    return score


@tool
def find_processing_algorithm(
    query: str,
    keywords: Optional[List[str]] = None,
    provider: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Find processing algorithms matching a natural-language task description.

    Scores EVERY registered algorithm (all providers) locally against the
    query using name/id/tags/description, and returns the top candidates.
    An empty 'matches' list means nothing scored — callers should fall back
    to selecting from the full catalog.

    Args:
        query: Natural language description, e.g., 'buffer layer by 50m'.
        keywords: Optional extra search terms/synonyms (e.g. LLM-generated:
            ["median", "percentile", "quantile", "zonal", "statistics"]).
        provider: Optional provider id to filter (e.g., "native", "gdal").
        limit: Max number of matches to return.

    Returns:
        Dict with 'matches' (list of {id, name, provider, tags, description}),
        'count', and 'total' (registry size).
    """
    try:
        catalog = get_algorithm_catalog()
        if provider:
            catalog = [e for e in catalog if e["provider"].lower() == provider.lower()]

        tokens = _query_tokens(query)
        for kw in keywords or []:
            for t in _query_tokens(str(kw)):
                if t not in tokens:
                    tokens.append(t)

        scored = [(_score_algorithm(tokens, e), e) for e in catalog]
        scored = [(s, e) for s, e in scored if s > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        matches = [e for _, e in scored[: max(1, limit)]]

        return {
            "matches": matches,
            "count": len(matches),
            "total": len(get_algorithm_catalog()),
            "query": query,
        }
    except Exception as e:
        raise Exception(f"Find algorithm error: {str(e)}")


def _param_optional(param: QgsProcessingParameterDefinition) -> bool:
    try:
        return bool(param.flags() & QgsProcessingParameterDefinition.Flag.FlagOptional)
    except Exception:
        # Fallback: some params may not expose flags cleanly
        return False


def _param_flag(param: QgsProcessingParameterDefinition, flag_name: str) -> bool:
    """True if *param* has the named flag (e.g. 'FlagAdvanced', 'FlagHidden')."""
    try:
        return bool(param.flags() & getattr(QgsProcessingParameterDefinition.Flag, flag_name))
    except Exception:
        return False


def _param_type_name(param: QgsProcessingParameterDefinition) -> str:
    try:
        return param.type()
    except Exception:
        # Some versions may not have .type(); use class name
        return param.__class__.__name__


@tool
def get_algorithm_parameters(algorithm: str) -> Dict[str, Any]:
    """
    Inspect an algorithm and return its parameter and output definitions.

    Args:
        algorithm: Algorithm id, e.g., 'native:buffer'.

    Returns:
        Dict containing algorithm metadata, plain-text help describing what
        the algorithm does and its parameter semantics, parameters, and outputs.
    """
    try:
        registry = QgsApplication.processingRegistry()
        alg = registry.algorithmById(algorithm)
        if alg is None:
            raise Exception(f"Algorithm not found: {algorithm}")

        # Parameters
        params: List[Dict[str, Any]] = []
        for p in alg.parameterDefinitions():
            item: Dict[str, Any] = {
                "name": p.name(),
                "description": p.description(),
                "type": _param_type_name(p),
                "optional": _param_optional(p),
                # Advanced/hidden params are collapsed or invisible in the
                # QGIS algorithm dialog; the gather step keeps them at default.
                "advanced": _param_flag(p, "FlagAdvanced"),
                "hidden": _param_flag(p, "FlagHidden"),
                "default": p.defaultValue(),
            }
            if isinstance(p, QgsProcessingParameterEnum):
                item["options"] = list(p.options())
                item["allowMultiple"] = bool(p.allowMultiple())
            params.append(item)

        # Outputs
        outputs: List[Dict[str, Any]] = [
            {
                "name": o.name(),
                "description": o.description(),
                "type": _param_type_name(o),
            }
            for o in alg.destinationParameterDefinitions()
        ]

        return {
            "id": alg.id(),
            "name": alg.displayName(),
            "provider": alg.provider().id(),
            "help": _plain_help_text(alg),
            "parameters": params,
            "outputs": outputs,
        }
    except Exception as e:
        raise Exception(f"Describe algorithm error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Parameter normalization & pre-flight validation
#
# Rule: a value is only rewritten when QGIS itself would reject it, so values
# that work today pass through untouched. The one exception is a reference to
# an output of the current workflow (label or layer name), which is pinned to
# that output's layer id so a same-named older layer can't be picked instead.
# ─────────────────────────────────────────────────────────────────────────────

# Parameter types whose value is a single layer reference (or a list of them
# for "multilayer"); vector/raster restricts which layers fuzzy matching sees.
_VECTOR_PARAM_TYPES = {"source", "vector"}
_RASTER_PARAM_TYPES = {"raster"}
LAYER_PARAM_TYPES = _VECTOR_PARAM_TYPES | _RASTER_PARAM_TYPES | {
    "layer",
    "mesh",
    "pointcloud",
    "multilayer",
}


def _processing_context() -> QgsProcessingContext:
    """A context set up the same way processing.run() sets up its own."""
    try:
        from processing.tools.dataobjects import createContext

        return createContext()
    except Exception:
        from qgis.core import QgsProject

        context = QgsProcessingContext()
        context.setProject(QgsProject.instance())
        return context


def _acceptable(definition, value, context) -> bool:
    try:
        return bool(definition.checkValueIsAcceptable(value, context))
    except Exception:
        return True  # can't tell; leave the value alone


def _candidate_layers(param_type: str) -> Dict[str, Any]:
    """Project layers (id -> layer) compatible with a layer parameter type."""
    from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer

    layers = QgsProject.instance().mapLayers()
    if param_type in _VECTOR_PARAM_TYPES:
        return {i: l for i, l in layers.items() if isinstance(l, QgsVectorLayer)}
    if param_type in _RASTER_PARAM_TYPES:
        return {i: l for i, l in layers.items() if isinstance(l, QgsRasterLayer)}
    return dict(layers)


def _resolve_layer_reference(
    value: Any, definition, context, output_refs: Dict[str, str]
) -> Tuple[Any, Optional[str]]:
    """Resolve one layer reference; returns (value, note-if-changed)."""
    from qgis.core import QgsProject

    if not isinstance(value, str) or not value.strip():
        return value, None
    project = QgsProject.instance()
    text = value.strip()

    # 1. Output label of an earlier task ("task_2_output" / "@task_2_output")
    label = text.lstrip("@").strip()
    for key, layer_id in output_refs.items():
        if label.lower() == key.lower() and project.mapLayer(layer_id):
            return layer_id, f"{value!r} -> output {key}"

    # 2. Display name of an output of this workflow: pin it to that layer
    for key, layer_id in output_refs.items():
        layer = project.mapLayer(layer_id)
        if layer and layer.name() == text and text != layer_id:
            return layer_id, f"{value!r} -> output {key} (layer id)"

    # 3. Already acceptable (exact layer name, id, or file path): keep
    if _acceptable(definition, value, context):
        return value, None

    # 4. Fuzzy match against compatible project layers
    candidates = _candidate_layers(definition.type())
    if not candidates:
        return value, None
    names = [layer.name() for layer in candidates.values()]
    match = find_best_layer_match(text, names)
    if match:
        for layer_id, layer in candidates.items():
            if layer.name() == match and _acceptable(definition, layer_id, context):
                return layer_id, f"{value!r} -> layer '{match}'"
    return value, None


def _resolve_enum(value: Any, definition, context) -> Tuple[Any, Optional[str]]:
    """Map an enum label (e.g. 'Round') to what QGIS expects (index or label)."""
    if _acceptable(definition, value, context):
        return value, None
    try:
        options = [str(o) for o in definition.options()]
        static = bool(getattr(definition, "usesStaticStrings", lambda: False)())
        multiple = bool(definition.allowMultiple())
    except Exception:
        return value, None

    def one(v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip().lower()
            if not static and re.fullmatch(r"-?\d+", s):
                return int(s)
            exact = [i for i, o in enumerate(options) if o.lower() == s]
            partial = [i for i, o in enumerate(options) if s and (s in o.lower() or o.lower() in s)]
            hits = exact or partial
            if len(hits) == 1:
                return options[hits[0]] if static else hits[0]
        elif isinstance(v, int) and not isinstance(v, bool) and static:
            if 0 <= v < len(options):
                return options[v]
        return v

    fixed = [one(v) for v in value] if (multiple and isinstance(value, list)) else one(value)
    if fixed != value and _acceptable(definition, fixed, context):
        return fixed, f"{value!r} -> {fixed!r}"
    return value, None


def normalize_parameters(
    algorithm: str,
    parameters: Dict[str, Any],
    output_refs: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Repair common LLM slips in a parameter dict before execution.

    - parameter names in the wrong case ("input" -> "INPUT")
    - references to earlier outputs ("task_1_output" -> that layer's id)
    - near-miss layer names ("River" -> layer "rivers")
    - enum labels instead of indexes ("Round" -> 0)

    Args:
        algorithm: Algorithm id.
        parameters: Parameter dict as gathered.
        output_refs: Workflow output label -> layer id.

    Returns:
        (parameters, notes): the repaired dict (a copy) and one note per change.
        On any internal error the input is returned unchanged.
    """
    params = dict(parameters)
    notes: List[str] = []
    try:
        alg = QgsApplication.processingRegistry().algorithmById(algorithm)
        if alg is None:
            return params, notes
        context = _processing_context()
        refs = output_refs or {}

        definitions = {d.name(): d for d in alg.parameterDefinitions()}
        lower_names = {name.lower(): name for name in definitions}
        for key in list(params):
            if key not in definitions and key.lower() in lower_names:
                canonical = lower_names[key.lower()]
                if canonical not in params:
                    params[canonical] = params.pop(key)
                    notes.append(f"{key} -> {canonical}")

        for name, definition in definitions.items():
            if name not in params or definition.isDestination():
                continue
            value = params[name]
            param_type = definition.type()
            note = None
            if param_type in LAYER_PARAM_TYPES:
                if isinstance(value, list):
                    fixed = []
                    for v in value:
                        fv, n = _resolve_layer_reference(v, definition, context, refs)
                        fixed.append(fv)
                        note = note or n
                    # Only keep the list rewrite if QGIS accepts the result
                    if note and (
                        _acceptable(definition, fixed, context)
                        or not _acceptable(definition, value, context)
                    ):
                        value = fixed
                    else:
                        note = None
                else:
                    value, note = _resolve_layer_reference(value, definition, context, refs)
            elif param_type == "enum":
                value, note = _resolve_enum(value, definition, context)
            if note:
                params[name] = value
                notes.append(f"{name}: {note}")
    except Exception as e:
        _logger.warning(f"Parameter normalization skipped: {e}")
        return dict(parameters), []
    return params, notes


def validate_parameters(algorithm: str, parameters: Dict[str, Any]) -> List[str]:
    """Run QGIS's own parameter check (the same one processing.run() does).

    Returns a list of problems (empty when valid). The pass/fail decision is
    exactly QGIS's checkParameterValues(), so nothing that would run is
    rejected; the per-parameter lines just name every bad value at once.
    If the check itself errors, returns [] and leaves it to execution.
    """
    try:
        alg = QgsApplication.processingRegistry().algorithmById(algorithm)
        if alg is None:
            return [f"Algorithm not found: {algorithm}"]
        context = _processing_context()
        ok, message = alg.checkParameterValues(parameters, context)
        if ok:
            return []
        problems = [message] if message else []
        for definition in alg.parameterDefinitions():
            if definition.isDestination():
                continue
            value = parameters.get(definition.name())
            if not _acceptable(definition, value, context):
                line = f"{definition.name()} ({definition.description()}): {value!r} is not a valid value"
                if not any(definition.name() in p for p in problems):
                    problems.append(line)
        return problems or ["QGIS rejected the parameter values"]
    except Exception as e:
        _logger.warning(f"Parameter validation skipped: {e}")
        return []


def _unique_layer_name(base: str) -> str:
    """Return a project-unique layer name derived from *base*."""
    from qgis.core import QgsProject

    project = QgsProject.instance()
    if not project.mapLayersByName(base):
        return base
    i = 2
    while project.mapLayersByName(f"{base} ({i})"):
        i += 1
    return f"{base} ({i})"


@tool
def execute_processing(algorithm: str, parameters: dict, **kwargs) -> dict:
    """
    Execute a processing algorithm and load results into the QGIS map.

    Returns a dict with 'success', 'output_layers' (project layer names usable
    as inputs for follow-up tasks), 'output_layer_ids' (same order), 'outputs'
    (algorithm output name -> layer id), and the raw result.
    """
    try:
        import processing
        from qgis.core import QgsProject, QgsMapLayer, QgsRasterLayer, QgsVectorLayer

        if "OUTPUT" not in parameters:
            parameters["OUTPUT"] = "TEMPORARY_OUTPUT"

        result = processing.run(algorithm, parameters, feedback=None)

        project = QgsProject.instance()
        output_layers: List[str] = []
        output_layer_ids: List[str] = []
        outputs: Dict[str, str] = {}
        base_name = f"Result - {algorithm.split(':')[-1]}"

        # Load every output layer/path into the project and record a usable
        # reference (project layer name + id) for downstream tasks.
        for output_name, value in result.items():
            if isinstance(value, QgsMapLayer):
                if project.mapLayer(value.id()) is None:
                    # Temporary outputs come back with generic names
                    # ("Buffered"); keep them unique so they can't be confused
                    # with an earlier result of the same algorithm.
                    value.setName(_unique_layer_name(value.name() or base_name))
                    project.addMapLayer(value)
                output_layers.append(value.name())
                output_layer_ids.append(value.id())
                outputs[output_name] = value.id()
            elif isinstance(value, str) and value and value != "TEMPORARY_OUTPUT":
                file_ext = os.path.splitext(value)[-1].lower()
                if not file_ext and not os.path.exists(value):
                    continue  # not a loadable path (plain string result)
                layer_name = _unique_layer_name(base_name)
                is_raster = file_ext in RASTER_EXTENSIONS
                lyr = (
                    QgsRasterLayer(value, layer_name)
                    if is_raster
                    else QgsVectorLayer(value, layer_name, "ogr")
                )
                if lyr and lyr.isValid():
                    project.addMapLayer(lyr)
                    output_layers.append(layer_name)
                    output_layer_ids.append(lyr.id())
                    outputs[output_name] = lyr.id()

        return {
            "algorithm": algorithm,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "success": True,
            "layer_added": bool(output_layers),
            "output_layers": output_layers,
            "output_layer_ids": output_layer_ids,
            "outputs": outputs,
            "result": {k: str(v) for k, v in result.items()},
        }
    except Exception as e:
        import traceback

        return {
            "algorithm": algorithm,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


__all__ = [
    "execute_processing",
    "get_algorithm_parameters",
    "find_processing_algorithm",
    "get_algorithm_catalog",
    "normalize_parameters",
    "validate_parameters",
]
