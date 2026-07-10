# -*- coding: utf-8 -*-
"""
Geoprocessing tools for QGIS operations.

Provides algorithm discovery (over the FULL processing registry, all providers),
parameter inspection, and execution with results loaded into the QGIS project.
"""
import html
import re
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
)

from ..config.constants import RASTER_EXTENSIONS

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


@tool
def list_processing_algorithms(
    search: Optional[str] = None, provider: Optional[str] = None, limit: int = 1000
) -> Dict[str, Any]:
    """
    List available QGIS processing algorithms.

    Args:
        search: Optional case-insensitive substring to filter by id or name.
        provider: Optional provider id to filter (e.g., "native", "gdal").
        limit: Maximum number of algorithms to return.

    Returns:
        Dict with summary and a list of algorithms {id, name, provider}.
    """
    try:
        catalog = get_algorithm_catalog()

        def matches(entry: Dict[str, Any]) -> bool:
            if provider and entry["provider"].lower() != provider.lower():
                return False
            if search:
                s = search.lower()
                return s in entry["id"].lower() or s in entry["name"].lower()
            return True

        filtered = [e for e in catalog if matches(e)][: max(0, limit)]
        items = [
            {"id": e["id"], "name": e["name"], "provider": e["provider"]}
            for e in filtered
        ]
        return {"count": len(items), "total": len(catalog), "items": items}
    except Exception as e:
        raise Exception(f"Listing algorithms error: {str(e)}")


def _param_optional(param: QgsProcessingParameterDefinition) -> bool:
    try:
        return bool(param.flags() & QgsProcessingParameterDefinition.FlagOptional)
    except Exception:
        # Fallback: some params may not expose flags cleanly
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
            }
            # Default value if available
            try:
                item["default"] = p.defaultValue()
            except Exception:
                pass

            # Enumerated options
            try:
                if isinstance(p, QgsProcessingParameterEnum):
                    item["options"] = list(p.options())
                    try:
                        item["allowMultiple"] = bool(
                            getattr(p, "allowMultiple", lambda: False)()
                        )
                    except Exception:
                        pass
            except Exception:
                pass

            params.append(item)

        # Outputs
        outputs: List[Dict[str, Any]] = []
        try:
            for o in alg.destinationParameterDefinitions():
                outputs.append(
                    {
                        "name": o.name(),
                        "description": o.description(),
                        "type": _param_type_name(o),
                    }
                )
        except Exception:
            # destinationParameterDefinitions may not exist in some versions
            pass

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

    Returns a dict with 'success', 'output_layers' (project layer names or
    file paths usable as inputs for follow-up tasks), and the raw result.
    """
    try:
        import os
        import processing
        from qgis.core import QgsProject, QgsMapLayer, QgsRasterLayer, QgsVectorLayer

        if "OUTPUT" not in parameters:
            parameters["OUTPUT"] = "TEMPORARY_OUTPUT"

        result = processing.run(algorithm, parameters, feedback=None)

        project = QgsProject.instance()
        output_layers: List[str] = []
        base_name = f"Result - {algorithm.split(':')[-1]}"

        # Load every output layer/path into the project and record a usable
        # reference (project layer name or file path) for downstream tasks.
        for value in result.values():
            if isinstance(value, QgsMapLayer):
                if not value.name():
                    value.setName(_unique_layer_name(base_name))
                project.addMapLayer(value)
                output_layers.append(value.name())
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

        return {
            "algorithm": algorithm,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "success": True,
            "layer_added": bool(output_layers),
            "output_layers": output_layers,
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
    "list_processing_algorithms",
    "get_algorithm_parameters",
    "find_processing_algorithm",
    "get_algorithm_catalog",
]
