# -*- coding: utf-8 -*-
"""
Data I/O tools for QGIS layer operations.
Provides tools for adding, removing, zooming to layers, and reading layer attributes.
"""
import os
from typing import Optional, List, Dict, Any
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsRasterLayer,
    QgsMapLayer,
)
from langchain_core.tools import tool
from ..utils.canvas_refresh import get_qgis_interface
from ..utils.layer_operations import remove_layer_on_main_thread
from ..utils.project_loader import load_project_on_main_thread

# logger for this module
from ..logger.processing_logger import get_processing_logger

_logger = get_processing_logger()


@tool
def add_layer_to_qgis(
    path_or_url: str,
    layer_name: Optional[str] = None,
    layer_type: Optional[str] = None,
) -> str:
    """
    Add a vector or raster layer to QGIS from a file path or URL.

    Args:
        path_or_url: File path or URL to the layer (e.g., '/path/to/file.shp', 'https://example.com/data.geojson')
        layer_name: Optional custom name for the layer. If not provided, uses the filename.
        layer_type: Optional layer type hint ('vector' or 'raster'). Auto-detected if not provided.

    Returns:
        Success message with layer name or error message.

    Examples:
        - add_layer_to_qgis('/data/cities.shp')
        - add_layer_to_qgis('https://example.com/data.geojson', 'Cities')
        - add_layer_to_qgis('/data/elevation.tif', layer_type='raster')
    """
    _logger.info(f"Adding layer to QGIS: {path_or_url}")
    try:
        # Determine layer name
        if not layer_name:
            layer_name = os.path.splitext(os.path.basename(path_or_url))[0]

        # Auto-detect layer type if not provided
        if not layer_type:
            raster_extensions = [".tif", ".tiff", ".img", ".asc", ".nc", ".jpg", ".png"]
            vector_extensions = [
                ".shp",
                ".geojson",
                ".json",
                ".gpkg",
                ".kml",
                ".gml",
                ".csv",
            ]

            ext = os.path.splitext(path_or_url.lower())[1]
            if ext in raster_extensions:
                layer_type = "raster"
                _logger.debug(f"Auto-detected layer type: raster (extension: {ext})")
            elif ext in vector_extensions:
                layer_type = "vector"
                _logger.debug(f"Auto-detected layer type: vector (extension: {ext})")
            else:
                # Default to vector for URLs or unknown types
                layer_type = "vector"
                _logger.debug(f"Unknown extension '{ext}', defaulting to vector")

        layer = None

        if layer_type.lower() == "vector":
            # Create vector layer
            layer = QgsVectorLayer(path_or_url, layer_name, "ogr")
        elif layer_type.lower() == "raster":
            # Create raster layer
            layer = QgsRasterLayer(path_or_url, layer_name)
        else:
            return (
                f"Error: Invalid layer type '{layer_type}'. Use 'vector' or 'raster'."
            )

        # Check if layer is valid
        if not layer.isValid():
            _logger.error(
                f"Failed to load layer from '{path_or_url}' - layer is invalid"
            )
            return f"Error: Failed to load layer from '{path_or_url}'. The layer is invalid."

        # Add layer to QGIS project
        QgsProject.instance().addMapLayer(layer)
        _logger.info(f"Successfully added {layer_type} layer '{layer_name}' to project")

        return f"Success: Added {layer_type} layer '{layer_name}' to QGIS with {layer.featureCount() if layer_type == 'vector' else 'N/A'} features."

    except Exception as e:
        _logger.error(f"Error adding layer: {str(e)}", exc_info=True)
        return f"Error adding layer: {str(e)}"


@tool
def list_qgis_layers(include_invisible: bool = True) -> str:
    """
    List all layers currently loaded in the QGIS project.

    Args:
        include_invisible: If True, includes both visible and invisible layers. If False, only visible layers.

    Returns:
        Formatted string with layer information including name, type, feature count, and visibility.

    Examples:
        - list_qgis_layers()
        - list_qgis_layers(include_invisible=False)
    """
    _logger.info(f"Listing QGIS layers (include_invisible={include_invisible})")
    try:
        project = QgsProject.instance()
        layers = project.mapLayers().values()

        if not layers:
            _logger.debug("No layers found in current project")
            return "No layers found in the current QGIS project."

        layer_info = []
        layer_info.append(f"Total layers in project: {len(layers)}\n")
        layer_info.append("=" * 80)

        for idx, layer in enumerate(layers, 1):
            # Check visibility
            layer_tree_layer = project.layerTreeRoot().findLayer(layer.id())
            is_visible = layer_tree_layer.isVisible() if layer_tree_layer else True

            # Skip invisible layers if requested
            if not include_invisible and not is_visible:
                continue

            # Get layer type
            if layer.type() == QgsMapLayer.VectorLayer:
                layer_type = "Vector"
                geometry_type = (
                    layer.geometryType().name
                    if hasattr(layer.geometryType(), "name")
                    else str(layer.geometryType())
                )
                feature_count = layer.featureCount()
                extra_info = f"Geometry: {geometry_type}, Features: {feature_count}"
            elif layer.type() == QgsMapLayer.RasterLayer:
                layer_type = "Raster"
                width = layer.width()
                height = layer.height()
                extra_info = f"Size: {width}x{height} pixels"
            else:
                layer_type = "Other"
                extra_info = ""

            visibility = "Visible" if is_visible else "Hidden"

            layer_info.append(f"\n{idx}. {layer.name()}")
            layer_info.append(f"   Type: {layer_type}")
            layer_info.append(f"   Status: {visibility}")
            layer_info.append(f"   {extra_info}")
            layer_info.append(f"   Source: {layer.source()}")

        _logger.info(f"Listed {len(layers)} layer(s) from project")
        return "\n".join(layer_info)

    except Exception as e:
        _logger.error(f"Error listing layers: {str(e)}", exc_info=True)
        return f"Error listing layers: {str(e)}"


@tool
def zoom_to_layer(layer_name: str) -> str:
    """
    Zoom the QGIS map canvas to the extent of the specified layer.

    Args:
        layer_name: Name of the layer to zoom to.

    Returns:
        Success message or error message.

    Examples:
        - zoom_to_layer('cities')
        - zoom_to_layer('roads')
    """
    _logger.info(f"Zooming to layer: {layer_name}")
    try:
        iface = get_qgis_interface()
        if not iface:
            return "Error: QGIS interface not initialized. Cannot zoom to layer."

        project = QgsProject.instance()

        # Find layer by name
        layer = None
        for lyr in project.mapLayers().values():
            if lyr.name().lower() == layer_name.lower():
                layer = lyr
                break

        if not layer:
            available_layers = [lyr.name() for lyr in project.mapLayers().values()]
            return f"Error: Layer '{layer_name}' not found. Available layers: {', '.join(available_layers)}"

        # Zoom to layer extent using the map canvas
        iface.setActiveLayer(layer)
        iface.zoomToActiveLayer()

        _logger.info(f"Successfully zoomed to layer '{layer_name}'")
        return f"Success: Zoomed to layer '{layer_name}'."

    except Exception as e:
        _logger.error(f"Error zooming to layer: {str(e)}", exc_info=True)
        return f"Error zooming to layer: {str(e)}"


@tool
def get_layer_columns(layer_name: str) -> str:
    """
    Get column names and types from a vector layer to help answer queries.

    Args:
        layer_name: Name of the layer to inspect.

    Returns:
        Formatted string with column names, types, and sample statistics.

    Examples:
        - get_layer_columns('cities')
        - get_layer_columns('roads')
    """
    _logger.info(f"Getting columns for layer: {layer_name}")
    try:
        project = QgsProject.instance()

        # Find layer by name
        layer = None
        for lyr in project.mapLayers().values():
            if lyr.name().lower() == layer_name.lower():
                layer = lyr
                break

        if not layer:
            available_layers = [lyr.name() for lyr in project.mapLayers().values()]
            return f"Error: Layer '{layer_name}' not found. Available layers: {', '.join(available_layers)}"

        # Check if it's a vector layer
        if layer.type() != QgsMapLayer.VectorLayer:
            _logger.error(f"Layer '{layer_name}' is not a vector layer")
            return f"Error: '{layer_name}' is not a vector layer. Column information is only available for vector layers."

        # Get fields
        fields = layer.fields()

        if fields.count() == 0:
            _logger.debug(f"Layer '{layer_name}' has no attribute fields")
            return f"Layer '{layer_name}' has no attribute fields."

        column_info = []
        column_info.append(f"Layer: {layer_name}")
        column_info.append(f"Total columns: {fields.count()}")
        column_info.append(f"Total features: {layer.featureCount()}")
        column_info.append(
            f"Geometry type: {layer.geometryType().name if hasattr(layer.geometryType(), 'name') else str(layer.geometryType())}"
        )
        column_info.append("\n" + "=" * 80)
        column_info.append("\nColumn Details:")
        column_info.append("-" * 80)

        for idx, field in enumerate(fields, 1):
            field_name = field.name()
            field_type = field.typeName()
            field_length = field.length()

            # Try to get unique value count for categorical fields
            if field.type() in [10, 2]:  # String or Integer types
                unique_values = layer.uniqueValues(fields.indexOf(field_name))
                unique_count = len(unique_values)

                if unique_count <= 10:
                    sample_values = (
                        f"Values: {', '.join(str(v) for v in list(unique_values)[:10])}"
                    )
                else:
                    sample_values = f"Unique values: {unique_count}"
            else:
                sample_values = ""

            column_info.append(f"\n{idx}. {field_name}")
            column_info.append(f"   Type: {field_type} (Length: {field_length})")
            if sample_values:
                column_info.append(f"   {sample_values}")

        _logger.info(f"Retrieved {fields.count()} columns from layer '{layer_name}'")
        return "\n".join(column_info)

    except Exception as e:
        _logger.error(f"Error getting layer columns: {str(e)}", exc_info=True)
        return f"Error getting layer columns: {str(e)}"


@tool
def remove_layer(layer_name: str) -> str:
    """
    Remove a layer from the QGIS project.
    Shows a confirmation dialog asking the user if they really want to remove the layer.

    Args:
        layer_name: Name of the layer to remove.

    Returns:
        Success message or error message.

    Examples:
        - remove_layer('cities')
        - remove_layer('temporary_layer')
    """
    _logger.info(f"Removing layer: {layer_name}")
    try:
        iface = get_qgis_interface()
        if not iface:
            return "Error: QGIS interface not initialized. Cannot remove layer."

        project = QgsProject.instance()

        # Find layer by name
        layer = None
        layer_id = None
        for lyr in project.mapLayers().values():
            if lyr.name().lower() == layer_name.lower():
                layer = lyr
                layer_id = lyr.id()
                break

        if not layer:
            available_layers = [lyr.name() for lyr in project.mapLayers().values()]
            _logger.error(
                f"Layer '{layer_name}' not found. Available: {', '.join(available_layers)}"
            )
            return f"Error: Layer '{layer_name}' not found. Available layers: {', '.join(available_layers)}"

        # remove the layer using main thread via callback
        result = remove_layer_on_main_thread(layer_id, layer_name)
        
        # for logging information
        if result.get("error"):
            _logger.error(f"Failed to remove layer: {result['error']}")
            return f"Error: {result['error']}"
        elif result.get("success"):
            _logger.info(f"Successfully removed layer '{layer_name}' from project")
            return f"Success: Layer '{layer_name}' has been removed from the project."
        else:
            _logger.error(f"Unknown error removing layer '{layer_name}'")
            return f"Error: Failed to remove layer '{layer_name}'."

    except Exception as e:
        _logger.error(f"Error removing layer: {str(e)}", exc_info=True)
        return f"Error removing layer: {str(e)}"


@tool
def create_new_qgis_project(path: str, project_name: Optional[str] = None) -> str:
    """
    Create a new QGIS project and save it to a file.

    Args:
        path: File path to save the project (e.g., '/path/to/project.qgs' or '/path/to/project.qgz')
        project_name: Optional project name/title. Uses the filename if nothing is provided.

    Returns:
        Success message with project path or error message.

    Examples:
        - create_new_qgis_project('/geoagent/my_project.qgs')
        - create_new_qgis_project('/geoagent/test_project.qgz', 'QGIS Test Project')
    """
    _logger.info(f"Creating new QGIS project: {path}")
    try:
        project = QgsProject.instance()

        # clear current project
        project.clear()

        # set project title
        if project_name:
            project.setTitle(project_name)
        else:
            project.setTitle(os.path.splitext(os.path.basename(path))[0])
        # validate file extensions
        valid_extensions = [".qgs", ".qgz"]
        file_ext = os.path.splitext(path.lower())[1]

        if file_ext not in valid_extensions:
            _logger.error(f"Invalid file extension: {file_ext}")
            return f"Error: Invalid file extension '{file_ext}'. Use '.qgs' or '.qgz' for QGIS project files."

        # create directory if needed
        project_dir = os.path.dirname(path)
        if project_dir and not os.path.exists(project_dir):
            _logger.debug(f"Creating project directory: {project_dir}")
            os.makedirs(project_dir)

        # save the project
        project.setFileName(path)
        success = project.write(path)

        if success:
            _logger.info(
                f"Successfully created project '{project.title()}' at '{path}'"
            )
            return f"Success: Created new project '{project.title()}' at '{path}'"
        else:
            _logger.error(f"Failed to save project to '{path}'")
            return f"Error: Failed to save project to '{path}'. Check file permissions and path validity."

    except Exception as e:
        _logger.error(f"Error creating project: {str(e)}", exc_info=True)
        return f"Error creating project: {str(e)}"


@tool
def save_qgis_project(path: Optional[str] = None) -> str:
    """
    Save the current QGIS project to a file.

    Args:
        path: Optional file path to save the project. If not provided, saves to the current project location.
              Can be '.qgs' or '.qgz' format.

    Returns:
        Success message with project path or error message.

    Examples:
        - save_qgis_project('/geoagent/my_project.qgs') 
        - save_qgis_project('D:/geoagent/project_backup.qgz')  
    """
    _logger.info(f"Saving new QGIS project: {path}")
    try:
        project = QgsProject.instance()

        # uses current project file if no path is provided
        if not path:
            path = project.fileName()
            if not path:
                return "Error: No file path specified and project has not been saved before. Please provide a file path."
        else:
            # convert relative path to absolute path
            path = os.path.abspath(path)

        # validate file extensions
        valid_extensions = [".qgs", ".qgz"]
        file_ext = os.path.splitext(path.lower())[1]
        if file_ext not in valid_extensions:
            return f"Error: Invalid file extension '{file_ext}'. Use '.qgs' or '.qgz' for QGIS project files."

        project_dir = os.path.dirname(path)
        if project_dir and not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # update project file path if saving to a new location
        if path != project.fileName():
            project.setFileName(path)

        # saves the project
        success = project.write(path)

        if success:
            return f"Success: Project saved to '{path}'"
        else:
            return f"Error: Failed to save project to '{path}'. Check file permissions and path validity."

    except Exception as e:
        return f"Error saving project: {str(e)}"

@tool
def load_qgis_project(path: str) -> str:
    """
    Loads an existing QGIS project file.

    Args:
        path: File path to the QGIS project file (e.g., '/path/to/project.qgs' or '/path/to/project.qgz')

    Returns:
        Success message with project details or error message.
    
    Examples:
    - load_qgis_project('/geoagent/my_project.qgs') 
    - load_qgis_project('C:/Users/asus/geoagent/map_project.qgz')
    """
    _logger.info(f"Loading QGIS project: {path}")
    try:
        if not os.path.exists(path):
            _logger.error(f"Project file not found: {path}")
            return f"Error: Project file '{path}' does not exist."

        valid_extensions = [".qgs", ".qgz"]
        file_ext = os.path.splitext(path.lower())[1]

        if file_ext not in valid_extensions:
            _logger.error(f"Invalid project file extension: {file_ext}")
            return f"Error: Invalid file extension '{file_ext}'. Expected '.qgs' or '.qgz' file."

        # load project on main thread via callback
        result = load_project_on_main_thread(path)

        if result.get("error"):
            _logger.error(f"Failed to load project: {result['error']}")
            return f"Error: {result['error']}"
        elif result.get("success"):
            _logger.info(f"Successfully loaded project from '{path}'")
            return f"Success: Project loaded from '{path}'"
        else:
            _logger.error(f"Unknown error loading project from '{path}'")
            return f"Success: Project loaded from '{path}'"
    except Exception as e:
        _logger.error(f"Error loading project: {str(e)}", exc_info=True)
        return f"Error loading project: {str(e)}"


@tool
def delete_existing_project(path: str) -> str:
    """
    Delete an existing QGIS project file from disk.

    Args:
        path: File path to the QGIS project file to delete (e.g., '/path/to/project.qgs' or '/path/to/project.qgz')

    Returns:
        Success message or error message.

    Examples:
        - delete_existing_project('/geoagent/old_project.qgs')
        - delete_existing_project('/geoagent/temp_analysis.qgz')
    """
    _logger.info(f"Deleting QGIS project: {path}")
    try:

        if not os.path.exists(path):
            _logger.error(f"Project file not found: {path}")
            return f"Error: Project file '{path}' does not exist."

        valid_extensions = [".qgs", ".qgz"]
        file_ext = os.path.splitext(path.lower())[1]

        if file_ext not in valid_extensions:
            _logger.error(f"Invalid project file extension: {file_ext}")
            return f"Error: Invalid file extension '{file_ext}'. Expected '.qgs' or '.qgz' file."

        os.remove(path)
        _logger.info(f"Successfully deleted project file: {path}")
        return f"Success: Project file '{path}' has been deleted."

    except Exception as e:
        _logger.error(f"Error deleting project: {str(e)}", exc_info=True)
        return f"Error deleting project: {str(e)}"


# Export tools for easy import
__all__ = [
    "add_layer_to_qgis",
    "list_qgis_layers",
    "get_layer_columns",
    "zoom_to_layer",
    "remove_layer",
    "create_new_qgis_project",
    "save_qgis_project",
    "load_qgis_project",
    "delete_existing_project",
]
