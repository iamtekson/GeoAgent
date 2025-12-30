from ..utils.canvas_refresh import get_qgis_interface, refresh_map_canvas
from typing import Optional
from qgis.core import (
    QgsProject,
    QgsMapLayer,
    QgsFeatureRequest,
    QgsExpression,
)

from langchain_core.tools import tool


@tool
def select_by_attribute(
    layer_name: str,
    field_name: str,
    operator: str,
    value: str,
) -> str:
    """
    Select features from a layer based on attribute criteria.

    Args:
        layer_name: Name of the layer to select from.
        field_name: Name of the field to filter on.
        operator: Comparison operator ('=', '!=', '<', '>', '<=', '>=', 'contains', 'starts_with', 'ends_with').
        value: Value to compare against.

    Returns:
        Success message with count of selected features or error message.

    Examples:
        - select_by_attribute('cities', 'population', '>', '5000')
        - select_by_attribute('roads', 'type', '=', 'highway')
        - select_by_attribute('names', 'name', 'starts_with', 'New')
    """
    try:
        project = QgsProject.instance()

        # Find layer
        layer = None
        for lyr in project.mapLayers().values():
            if (
                lyr.name().lower() == layer_name.lower()
                and lyr.type() == QgsMapLayer.VectorLayer
            ):
                layer = lyr
                break

        if not layer:
            return f"Error: Layer '{layer_name}' not found."

        # Check if field exists
        field_index = layer.fields().indexFromName(field_name)
        if field_index == -1:
            available_fields = [f.name() for f in layer.fields()]
            return f"Error: Field '{field_name}' not found. Available fields: {', '.join(available_fields)}"

        # Build expression based on operator
        operator_map = {
            "=": "=",
            "!=": "!=",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "contains": "LIKE",
            "starts_with": "LIKE",
            "ends_with": "LIKE",
        }

        qgis_operator = operator_map.get(operator)
        if not qgis_operator:
            return f"Error: Unsupported operator '{operator}'. Use one of: {', '.join(operator_map.keys())}"

        # Construct filter expression
        if operator == "contains":
            expression_str = f"\"{field_name}\" LIKE '%{value}%'"
        elif operator == "starts_with":
            expression_str = f"\"{field_name}\" LIKE '{value}%'"
        elif operator == "ends_with":
            expression_str = f"\"{field_name}\" LIKE '%{value}'"
        else:
            # Try to convert to number if operator is numeric
            try:
                num_value = float(value)
                expression_str = f'"{field_name}" {operator} {num_value}'
            except ValueError:
                expression_str = f"\"{field_name}\" {operator} '{value}'"

        # Execute selection
        expression = QgsExpression(expression_str)
        if expression.hasParserError():
            return f"Error: Invalid expression: {expression.parserErrorString()}"

        request = QgsFeatureRequest(expression)
        selected_ids = [f.id() for f in layer.getFeatures(request)]

        layer.selectByIds(selected_ids)

        # Highlight on map
        refresh_map_canvas()

        return f"Success: Selected {len(selected_ids)} features in '{layer_name}' where {field_name} {operator} '{value}'."

    except Exception as e:
        return f"Error selecting by attribute: {str(e)}"


@tool
def select_by_geometry(
    layer_name: str,
    geometry_filter: str,
    reference_layer_name: Optional[str] = None,
) -> str:
    """
    Select features from a layer based on geometric criteria.

    Args:
        layer_name: Name of the layer to select from.
        geometry_filter: Type of geometric selection ('largest', 'smallest', 'intersecting', 'inside', 'touching').
        reference_layer_name: Name of reference layer for 'intersecting', 'inside', 'touching' operations.

    Returns:
        Success message with count of selected features or error message.

    Examples:
        - select_by_geometry('polygons', 'largest')
        - select_by_geometry('roads', 'intersecting', 'study_area')
        - select_by_geometry('points', 'inside', 'boundary')
    """
    try:
        project = QgsProject.instance()

        # Find main layer
        layer = None
        for lyr in project.mapLayers().values():
            if (
                lyr.name().lower() == layer_name.lower()
                and lyr.type() == QgsMapLayer.VectorLayer
            ):
                layer = lyr
                break

        if not layer:
            return f"Error: Layer '{layer_name}' not found."

        selected_ids = []

        if geometry_filter == "largest":
            # Find feature with largest area/length
            max_size = 0
            largest_id = None

            for feature in layer.getFeatures():
                size = (
                    feature.geometry().area()
                    if feature.geometry().type() == 2
                    else feature.geometry().length()
                )
                if size > max_size:
                    max_size = size
                    largest_id = feature.id()

            if largest_id:
                selected_ids = [largest_id]

        elif geometry_filter == "smallest":
            # Find feature with smallest area/length
            min_size = float("inf")
            smallest_id = None

            for feature in layer.getFeatures():
                size = (
                    feature.geometry().area()
                    if feature.geometry().type() == 2
                    else feature.geometry().length()
                )
                if size < min_size:
                    min_size = size
                    smallest_id = feature.id()

            if smallest_id:
                selected_ids = [smallest_id]

        elif geometry_filter in ["intersecting", "inside", "touching"]:
            if not reference_layer_name:
                return f"Error: Reference layer required for '{geometry_filter}' operation."

            # Find reference layer
            ref_layer = None
            for lyr in project.mapLayers().values():
                if (
                    lyr.name().lower() == reference_layer_name.lower()
                    and lyr.type() == QgsMapLayer.VectorLayer
                ):
                    ref_layer = lyr
                    break

            if not ref_layer:
                return f"Error: Reference layer '{reference_layer_name}' not found."

            # Get union of reference geometries
            ref_geom = None
            for feature in ref_layer.getFeatures():
                if ref_geom is None:
                    ref_geom = feature.geometry()
                else:
                    ref_geom = ref_geom.combine(feature.geometry())

            if not ref_geom:
                return "Error: Reference layer has no valid geometries."

            # Select based on spatial relationship
            for feature in layer.getFeatures():
                geom = feature.geometry()

                if geometry_filter == "intersecting" and geom.intersects(ref_geom):
                    selected_ids.append(feature.id())
                elif geometry_filter == "inside" and geom.within(ref_geom):
                    selected_ids.append(feature.id())
                elif geometry_filter == "touching" and geom.touches(ref_geom):
                    selected_ids.append(feature.id())

        else:
            return f"Error: Unknown geometry filter '{geometry_filter}'. Use: 'largest', 'smallest', 'intersecting', 'inside', 'touching'."

        # Apply selection
        layer.selectByIds(selected_ids)

        # Highlight on map
        refresh_map_canvas()

        return f"Success: Selected {len(selected_ids)} features in '{layer_name}' using '{geometry_filter}' filter."

    except Exception as e:
        return f"Error selecting by geometry: {str(e)}"


# Export tools for easy import
__all__ = [
    "select_by_attribute",
    "select_by_geometry",
]
