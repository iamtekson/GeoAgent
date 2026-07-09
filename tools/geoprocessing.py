# -*- coding: utf-8 -*-
"""
Geoprocessing tools for QGIS operations.
Provides tools for geometric operations (buffer, clip, dissolve, etc.) and spatial filtering/selection.
"""
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
from qgis.core import (
    QgsApplication,
    QgsProcessingAlgorithm,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
)

from ..config.constants import RASTER_EXTENSIONS



@tool
def execute_processing(algorithm: str, parameters: dict, **kwargs) -> dict:
    """
    Execute a processing algorithm and load results into QGIS map.
    """
    try:
        import processing
        from qgis.core import QgsProject, QgsMapLayer

        # 1. Use QGIS standard for temporary outputs if not specified
        if "OUTPUT" not in parameters:
            parameters["OUTPUT"] = "TEMPORARY_OUTPUT"

        # 2. Execute the algorithm
        result = processing.run(algorithm, parameters, feedback=None)

        layer_added = False
        output_layer_obj = None

        # 3. Find the output layer in the results
        # Algorithms usually return the layer object or a string path
        for key in ["OUTPUT", "output", "OUTPUT_LAYER", "output_layer"]:
            if key in result:
                val = result[key]
                if isinstance(val, QgsMapLayer):
                    output_layer_obj = val
                    break
                elif isinstance(val, str):
                    # It's a file path or memory URI, we'll handle it below
                    output_layer_obj = val
                    break

        # Fallback: find any layer in the result
        if not output_layer_obj:
            for val in result.values():
                if isinstance(val, QgsMapLayer):
                    output_layer_obj = val
                    break

        # 4. Add to project
        if output_layer_obj:
            if isinstance(output_layer_obj, QgsMapLayer):
                # It's already a layer object
                if not output_layer_obj.name():
                    output_layer_obj.setName(f"Result - {algorithm.split(':')[-1]}")
                QgsProject.instance().addMapLayer(output_layer_obj)
                layer_added = True
            elif isinstance(output_layer_obj, str):
                from qgis.core import QgsRasterLayer, QgsVectorLayer
                import os

                # Determine if it's a raster or vector file by extension
                file_ext = os.path.splitext(output_layer_obj)[-1].lower()
                is_raster = file_ext in RASTER_EXTENSIONS or 'raster' in output_layer_obj.lower()
                layer_name = f"Result - {algorithm.split(':')[-1]}"
                
                if is_raster:
                    # Load as raster layer directly using QgsRasterLayer
                    lyr = QgsRasterLayer(output_layer_obj, layer_name)
                    if lyr and lyr.isValid():
                        QgsProject.instance().addMapLayer(lyr)
                        layer_added = True
                else:
                    # Load as vector layer directly using QgsVectorLayer
                    lyr = QgsVectorLayer(output_layer_obj, layer_name, "ogr")
                    if lyr and lyr.isValid():
                        QgsProject.instance().addMapLayer(lyr)
                        layer_added = True

        return {
            "algorithm": algorithm,
            "parameters": {k: str(v) for k, v in parameters.items()},
            "success": True,
            "layer_added": layer_added,
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


@tool
def list_processing_algorithms(
    search: Optional[str] = None, provider: Optional[str] = None, limit: int = 1000
) -> Dict[str, Any]:
    """
    List available QGIS processing algorithms.

    Args:
        provider: Optional provider id to filter. (e.g., "native", "pdal", "3d")
        search: Optional case-insensitive substring to filter by id or name.
        limit: Maximum number of algorithms to return.

    Returns:
        Dict with summary and a list of algorithms {id, name, provider}.
    """
    try:
        registry = QgsApplication.processingRegistry()
        algs: List[QgsProcessingAlgorithm] = list(registry.algorithms())

        def matches(a: QgsProcessingAlgorithm) -> bool:
            if provider and a.provider().id().lower() != provider.lower():
                return False
            if search:
                s = search.lower()
                return s in a.id().lower() or s in a.displayName().lower()
            return True

        filtered = [a for a in algs if matches(a)]
        filtered = filtered[: max(0, limit)]

        items = [
            {
                "id": a.id(),
                "name": a.displayName(),
                "provider": a.provider().id(),
            }
            for a in filtered
        ]

        return {
            "count": len(items),
            "total": len(algs),
            "items": items,
        }
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
        Dict containing algorithm metadata, parameters, and outputs.
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
                    # Allow multiple?
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
            "parameters": params,
            "outputs": outputs,
        }
    except Exception as e:
        raise Exception(f"Describe algorithm error: {str(e)}")


@tool
def find_processing_algorithm(
    query: str, provider: Optional[str]=None, limit: int = 300
) -> Dict[str, Any]:
    """
    Find algorithms that match a natural language query.

    Extracts key operation words from query and uses list_processing_algorithms
    to find relevant candidates. The LLM will select the best match.

    Args:
        query: Natural language description, e.g., 'buffer layer by 50m'.
        provider: Optional provider id to filter (e.g., "native", "gdal", "grass").
        limit: Max number of matches to return.

    Returns:
        Dict with following keys:
        - best: id of best matching algorithm (first in list) or None
        - matches: list of matching algorithms with {id, name, provider}
        - query: original query string
    """
    try:
        search_term = query.lower().strip()
        result = list_processing_algorithms.invoke(
            {"search": None, "provider": provider, "limit": limit}
        )
        items = result.get("items", [])

        if not items and search_term:
            raise Exception("No algorithms found matching the query.")

        return {
            "best": items[0]["id"] if items else None,
            "matches": items,
            "query": query,
        }
    except Exception as e:
        raise Exception(f"Find algorithm error: {str(e)}")


__all__ = [
    "execute_processing",
    "list_processing_algorithms",
    "get_algorithm_parameters",
    "find_processing_algorithm",
]
