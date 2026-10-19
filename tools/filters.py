# -*- coding: utf-8 -*-
"""
Selection tools: thin wrappers around QGIS's own Select by Expression,
Select by Location and Select within Distance algorithms.

The LLM writes the condition (a QGIS expression) using the column names and
values it sees; QGIS checks and runs it. Parameters go through the same
normalization and pre-flight check as processing mode, so an unknown column
or option comes back as an error that lists the real ones.
"""
from typing import Iterable, Optional

from qgis.core import (
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsFeatureRequest,
    QgsMapLayer,
    QgsProcessingFeatureSourceDefinition,
    QgsProject,
    QgsUnitTypes,
)

from langchain_core.tools import tool
from .geoprocessing import normalize_parameters, validate_parameters
from ..logger.processing_logger import get_processing_logger
from ..utils.canvas_refresh import get_qgis_interface, qgis_main_thread
from ..utils.layer_columns import describe_columns
from ..utils.layer_matching import find_layer

_logger = get_processing_logger()


def _vector_layer(layer_name: str):
    """(layer, matched name, error message) for a vector layer name."""
    project = QgsProject.instance()
    layer, matched = find_layer(project, layer_name, layer_type=QgsMapLayer.LayerType.VectorLayer)
    if layer:
        return layer, matched, ""
    available = [
        lyr.name()
        for lyr in project.mapLayers().values()
        if lyr.type() == QgsMapLayer.LayerType.VectorLayer
    ]
    return None, "", f"**Error:** Layer **{layer_name}** not found. Available layers: {', '.join(available)}"


def _run(algorithm: str, parameters: dict) -> str:
    """Run a QGIS algorithm after processing mode's checks; '' or an error."""
    import processing

    params, _ = normalize_parameters(algorithm, parameters)
    problems = validate_parameters(algorithm, params)
    if problems:
        return "**Error:** " + "; ".join(problems)
    processing.run(algorithm, params)
    iface = get_qgis_interface()
    if iface:
        iface.mapCanvas().refresh()
    return ""


def _selection_reply(layer, name: str, what: str, columns: Iterable[str] = (), max_rows: int = 10) -> str:
    """Selected count, the first selected features, or the values of the
    columns used when nothing matched."""
    count = layer.selectedFeatureCount()
    lines = [f"**Success:** {count} of {layer.featureCount()} features of **{name}** are selected ({what})."]
    columns = [c for c in columns if layer.fields().lookupField(c) >= 0]
    if count == 0:
        if columns:
            lines.append(f"Nothing matched. Values of the columns used: {describe_columns(layer, columns, samples=10)}")
        return "\n".join(lines)

    context = QgsExpressionContext(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
    display = QgsExpression(layer.displayExpression())
    display.prepare(context)
    lines.append("Selected features:")
    for feature in layer.getSelectedFeatures(QgsFeatureRequest().setLimit(max_rows)):
        context.setFeature(feature)
        parts = [f"fid {feature.id()}", str(display.evaluate(context))]
        parts += [f"{c}={feature[c]}" for c in columns]
        lines.append("- " + " | ".join(parts))
    if count > max_rows:
        lines.append(f"- ... and {count - max_rows} more")
    return "\n".join(lines)


@tool
@qgis_main_thread
def select_by_expression(layer_name: str, expression: str, method: str = "creating new selection") -> str:
    """
    Select features with a QGIS expression (QGIS's "Select by Expression").

    Use the layer's real column names (see get_layer_columns): columns in
    double quotes, text in single quotes. Compare text with ILIKE so case
    doesn't matter. Any QGIS function works, e.g. $area, $length, or
    maximum("population") for the largest value.

    Args:
        layer_name: Layer to select from.
        expression: QGIS expression.
        method: 'creating new selection', 'adding to current selection',
            'removing from current selection' or 'selecting within current
            selection' (to combine conditions step by step).

    Examples:
        - select_by_expression('people', '"name" ILIKE \\'%kshetri%\\'')
        - select_by_expression('trees', '"height" > 10 AND "species" ILIKE \\'oak\\'')
        - select_by_expression('lakes', '$area = maximum($area)')
    """
    _logger.info(f"Selecting by expression on '{layer_name}': {expression}")
    try:
        layer, name, error = _vector_layer(layer_name)
        if error:
            return error
        error = _run(
            "qgis:selectbyexpression",
            {"INPUT": layer.id(), "EXPRESSION": expression, "METHOD": method},
        )
        if error:
            return error
        columns = QgsExpression(expression).referencedColumns()
        return _selection_reply(layer, name, f"`{expression}`", sorted(columns))
    except Exception as e:
        _logger.error(f"Error selecting by expression: {str(e)}", exc_info=True)
        return f"**Error:** selecting by expression: {str(e)}"


@tool
@qgis_main_thread
def select_by_location(
    layer_name: str,
    predicate: str,
    reference_layer_name: str,
    reference_selected_only: bool = False,
    distance: Optional[float] = None,
    method: str = "creating new selection",
) -> str:
    """
    Select features by their spatial relation to another layer (QGIS's
    "Select by Location", or "Select within Distance" when a distance is
    given). Layers in different coordinate systems are handled.

    Args:
        layer_name: Layer to select from.
        predicate: 'intersect', 'contain', 'disjoint', 'equal', 'touch',
            'overlap', 'are within' or 'cross' (ignored with a distance).
        reference_layer_name: Layer to compare with.
        reference_selected_only: Use only the selected features of the
            reference layer, e.g. one city selected by name first.
        distance: Select features within this distance of the reference
            features, in the map units of layer_name's CRS.
        method: 'creating new selection', 'adding to current selection',
            'removing from current selection' or 'selecting within current
            selection'.

    Examples:
        - select_by_location('parcels', 'are within', 'cities', reference_selected_only=True)
        - select_by_location('schools', 'intersect', 'hospitals', distance=2000)
    """
    _logger.info(
        f"Selecting by location on '{layer_name}': {predicate} '{reference_layer_name}' "
        f"(selected only: {reference_selected_only}, distance: {distance})"
    )
    try:
        layer, name, error = _vector_layer(layer_name)
        if error:
            return error
        reference, reference_name, error = _vector_layer(reference_layer_name)
        if error:
            return error
        source = (
            QgsProcessingFeatureSourceDefinition(reference.id(), True)
            if reference_selected_only
            else reference.id()
        )
        scope = "selected features of " if reference_selected_only else ""
        if distance:
            units = QgsUnitTypes.toString(layer.crs().mapUnits())
            error = _run(
                "native:selectwithindistance",
                {"INPUT": layer.id(), "REFERENCE": source, "DISTANCE": distance, "METHOD": method},
            )
            what = f"within {distance} {units} of {scope}{reference_name}"
        else:
            error = _run(
                "native:selectbylocation",
                {"INPUT": layer.id(), "PREDICATE": [predicate], "INTERSECT": source, "METHOD": method},
            )
            what = f"{predicate} {scope}{reference_name}"
        return error or _selection_reply(layer, name, what)
    except Exception as e:
        _logger.error(f"Error selecting by location: {str(e)}", exc_info=True)
        return f"**Error:** selecting by location: {str(e)}"


@tool
@qgis_main_thread
def calculate(layer_name: str, expression: str) -> str:
    """
    Calculate a value over a layer's features with a QGIS aggregate
    expression: counts, totals, averages, extremes. Lengths and areas are
    measured as in QGIS's field calculator (the project's ellipsoid and units).

    Args:
        layer_name: Layer to calculate over.
        expression: QGIS aggregate expression. Use filter:= to restrict the
            features, e.g. filter:=is_selected() for the selection.

    Examples:
        - calculate('pois', 'count(1, filter:="amenity" ILIKE \\'school\\')')
        - calculate('railways', 'sum($length, filter:=is_selected())')
        - calculate('cities', 'mean("population")')
    """
    _logger.info(f"Calculating on '{layer_name}': {expression}")
    try:
        layer, name, error = _vector_layer(layer_name)
        if error:
            return error
        expr = QgsExpression(expression)
        if expr.hasParserError():
            return f"**Error:** Invalid expression: {expr.parserErrorString()}"
        context = QgsExpressionContext(QgsExpressionContextUtils.globalProjectLayerScopes(layer))
        expr.prepare(context)
        value = expr.evaluate(context)
        if expr.hasEvalError():
            return (
                f"**Error:** {expr.evalErrorString()}. "
                f"Columns of '{name}': {describe_columns(layer)}"
            )
        project = QgsProject.instance()
        return (
            f"**Result:** {expression} on **{name}** = {value}\n"
            f"(lengths in {QgsUnitTypes.toString(project.distanceUnits())}, "
            f"areas in {QgsUnitTypes.toString(project.areaUnits())})"
        )
    except Exception as e:
        _logger.error(f"Error calculating: {str(e)}", exc_info=True)
        return f"**Error:** calculating: {str(e)}"


@tool
@qgis_main_thread
def zoom_to_selection(layer_name: str) -> str:
    """
    Zoom the map to the selected features of a layer.

    Args:
        layer_name: Layer whose selection to zoom to.
    """
    _logger.info(f"Zooming to selection of '{layer_name}'")
    try:
        layer, name, error = _vector_layer(layer_name)
        if error:
            return error
        if layer.selectedFeatureCount() == 0:
            return f"**Error:** No features are selected in **{name}**."
        iface = get_qgis_interface()
        if not iface:
            return "**Error:** QGIS interface not initialized. Cannot zoom."
        iface.mapCanvas().zoomToSelected(layer)
        iface.mapCanvas().refresh()
        return f"**Success:** Zoomed to the {layer.selectedFeatureCount()} selected features of **{name}**."
    except Exception as e:
        _logger.error(f"Error zooming to selection: {str(e)}", exc_info=True)
        return f"**Error:** zooming to selection: {str(e)}"


@tool
@qgis_main_thread
def clear_selection(layer_name: str = "") -> str:
    """
    Clear the selection of one layer, or of every layer if no name is given.

    Args:
        layer_name: Layer to clear; empty for all layers.
    """
    _logger.info(f"Clearing selection of '{layer_name or 'all layers'}'")
    try:
        if layer_name:
            layer, name, error = _vector_layer(layer_name)
            if error:
                return error
            layer.removeSelection()
        else:
            name = "all layers"
            for layer in QgsProject.instance().mapLayers().values():
                if layer.type() == QgsMapLayer.LayerType.VectorLayer:
                    layer.removeSelection()
        iface = get_qgis_interface()
        if iface:
            iface.mapCanvas().refresh()
        return f"**Success:** Cleared the selection of **{name}**."
    except Exception as e:
        _logger.error(f"Error clearing selection: {str(e)}", exc_info=True)
        return f"**Error:** clearing selection: {str(e)}"


# Export tools for easy import
__all__ = [
    "select_by_expression",
    "select_by_location",
    "calculate",
    "zoom_to_selection",
    "clear_selection",
]
