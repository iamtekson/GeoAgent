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
