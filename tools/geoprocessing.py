# -*- coding: utf-8 -*-
"""
Geoprocessing tools for QGIS operations.
Provides tools for geometric operations (buffer, clip, dissolve, etc.) and spatial filtering/selection.
"""
import uuid
from typing import Optional, List
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsMapLayer,
    QgsFeatureRequest,
    QgsExpression,
    QgsFeature,
)
import processing
from langchain_core.tools import tool

from .io import get_qgis_interface


# ==================== Geometric Operations ====================
@tool
def buffer_layer(
    layer_name: str,
    distance: float,
    output_name: Optional[str] = None,
) -> str:
    """
    Create a buffer around features in a layer.

    Args:
        layer_name: Name of the input layer to buffer.
        distance: Buffer distance in layer units (usually meters).
        output_name: Optional name for output layer. If not provided, creates temporary layer.

    Returns:
        Success message with output layer name or error message.

    Examples:
        - buffer_layer('roads', 100)
        - buffer_layer('cities', 50, 'cities_buffered')
    """
    try:
        project = QgsProject.instance()

        # Find input layer
        input_layer = None
        for lyr in project.mapLayers().values():
            if (
                lyr.name().lower() == layer_name.lower()
                and lyr.type() == QgsMapLayer.VectorLayer
            ):
                input_layer = lyr
                break

        if not input_layer:
            available_layers = [
                lyr.name()
                for lyr in project.mapLayers().values()
                if lyr.type() == QgsMapLayer.VectorLayer
            ]
            return f"Error: Vector layer '{layer_name}' not found. Available layers: {', '.join(available_layers)}"

        # Create output layer name
        if not output_name:
            output_name = f"buffered_{layer_name}_{uuid.uuid4().hex[:8]}"

        # Perform buffer operation
        output_layer = QgsVectorLayer("Polygon", output_name, "memory")
        output_layer.startEditing()

        # Copy the data provider and create buffered geometries
        output_layer_dp = output_layer.dataProvider()

        # Add same fields as input layer
        output_layer_dp.addAttributes(input_layer.fields())
        output_layer.updateFields()

        # Buffer each feature
        features_list = []
        for feature in input_layer.getFeatures():
            buffered_geom = feature.geometry().buffer(distance, 8)
            new_feature = QgsFeature(feature)
            new_feature.setGeometry(buffered_geom)
            features_list.append(new_feature)

        output_layer_dp.addFeatures(features_list)
        output_layer.commitChanges()

        # Add layer to project
        project.addMapLayer(output_layer)

        return f"Success: Created buffered layer '{output_name}' with {output_layer.featureCount()} features."

    except Exception as e:
        return f"Error buffering layer: {str(e)}"


@tool
def clip_layer(
    input_layer_name: str,
    clip_layer_name: str,
    output_name: Optional[str] = None,
) -> str:
    """
    Clip features from one layer using another layer as a mask.

    Args:
        input_layer_name: Name of the layer to be clipped.
        clip_layer_name: Name of the layer to use as clipping mask.
        output_name: Optional name for output layer. If not provided, creates temporary layer.

    Returns:
        Success message with output layer name or error message.

    Examples:
        - clip_layer('roads', 'study_area')
        - clip_layer('buildings', 'city_boundary', 'clipped_buildings')
    """
    try:
        project = QgsProject.instance()

        # Find input layers
        input_layer = None
        clip_layer = None

        for lyr in project.mapLayers().values():
            if lyr.type() == QgsMapLayer.VectorLayer:
                if lyr.name().lower() == input_layer_name.lower():
                    input_layer = lyr
                elif lyr.name().lower() == clip_layer_name.lower():
                    clip_layer = lyr

        if not input_layer:
            available_layers = [
                lyr.name()
                for lyr in project.mapLayers().values()
                if lyr.type() == QgsMapLayer.VectorLayer
            ]
            return f"Error: Input layer '{input_layer_name}' not found. Available layers: {', '.join(available_layers)}"
        if not clip_layer:
            available_layers = [
                lyr.name()
                for lyr in project.mapLayers().values()
                if lyr.type() == QgsMapLayer.VectorLayer
            ]
            return f"Error: Clip layer '{clip_layer_name}' not found. Available layers: {', '.join(available_layers)}"

        # Create output layer name
        if not output_name:
            output_name = f"clipped_{input_layer_name}_{uuid.uuid4().hex[:8]}"

        # Perform clip operation using QGIS processing algorithm
        result = processing.run(
            "native:clip",
            {"INPUT": input_layer, "OVERLAY": clip_layer, "OUTPUT": "memory:"},
        )

        # Get output layer and rename it
        output_layer = result["OUTPUT"]
        output_layer.setName(output_name)

        # Add layer to project
        project.addMapLayer(output_layer)

        return f"Success: Clipped layer '{output_name}' with {output_layer.featureCount()} features."

    except Exception as e:
        return f"Error clipping layer: {str(e)}"


@tool
def dissolve_layer(
    layer_name: str,
    field_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> str:
    """
    Dissolve features in a layer by merging adjacent features.

    Args:
        layer_name: Name of the layer to dissolve.
        field_name: Optional field to dissolve by (merges features with same field value).
        output_name: Optional name for output layer.

    Returns:
        Success message with output layer name or error message.

    Examples:
        - dissolve_layer('countries')
        - dissolve_layer('regions', 'state')
    """
    try:
        project = QgsProject.instance()

        # Find input layer
        input_layer = None
        for lyr in project.mapLayers().values():
            if (
                lyr.name().lower() == layer_name.lower()
                and lyr.type() == QgsMapLayer.VectorLayer
            ):
                input_layer = lyr
                break

        if not input_layer:
            available_layers = [
                lyr.name()
                for lyr in project.mapLayers().values()
                if lyr.type() == QgsMapLayer.VectorLayer
            ]
            return f"Error: Vector layer '{layer_name}' not found. Available layers: {', '.join(available_layers)}"

        if not output_name:
            output_name = f"dissolved_{layer_name}_{uuid.uuid4().hex[:8]}"

        # Group features by field if specified
        if field_name:
            field_index = input_layer.fields().indexFromName(field_name)
            if field_index == -1:
                return f"Error: Field '{field_name}' not found in layer."

            groups = {}
            for feature in input_layer.getFeatures():
                key = feature[field_index]
                if key not in groups:
                    groups[key] = []
                groups[key].append(feature)

            # Merge geometries for each group
            output_layer = QgsVectorLayer(
                f"Polygon?crs={input_layer.crs().authid()}", output_name, "memory"
            )
            output_layer.startEditing()
            output_layer_dp = output_layer.dataProvider()
            output_layer_dp.addAttributes(input_layer.fields())
            output_layer.updateFields()

            merged_features = []
            for key, features in groups.items():
                if not features:
                    continue

                # Merge geometries
                geom = features[0].geometry()
                for feat in features[1:]:
                    geom = geom.combine(feat.geometry())

                # Create new feature
                merged_feat = QgsFeature(features[0])
                merged_feat.setGeometry(geom)
                merged_features.append(merged_feat)

            output_layer_dp.addFeatures(merged_features)
            output_layer.commitChanges()

        else:
            # Dissolve all features into one
            output_layer = QgsVectorLayer(
                f"Polygon?crs={input_layer.crs().authid()}", output_name, "memory"
            )
            output_layer.startEditing()

            # Merge all geometries
            all_geom = None
            for feature in input_layer.getFeatures():
                if all_geom is None:
                    all_geom = feature.geometry()
                else:
                    all_geom = all_geom.combine(feature.geometry())

            # Create single feature
            single_feature = QgsFeature(input_layer.fields())
            single_feature.setGeometry(all_geom)
            output_layer.dataProvider().addFeature(single_feature)
            output_layer.commitChanges()

        # Add layer to project
        project.addMapLayer(output_layer)

        return f"Success: Dissolved layer '{output_name}' with {output_layer.featureCount()} features."

    except Exception as e:
        return f"Error dissolving layer: {str(e)}"


# ==================== Filtering & Selection ====================


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
        iface = get_qgis_interface()
        if iface:
            canvas = iface.mapCanvas()
            canvas.refresh()

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
        iface = get_qgis_interface()
        if iface:
            canvas = iface.mapCanvas()
            canvas.refresh()

        return f"Success: Selected {len(selected_ids)} features in '{layer_name}' using '{geometry_filter}' filter."

    except Exception as e:
        return f"Error selecting by geometry: {str(e)}"


# Export tools for easy import
__all__ = [
    "buffer_layer",
    "clip_layer",
    "dissolve_layer",
    "select_by_attribute",
    "select_by_geometry",
]
