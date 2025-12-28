from .io import (
    add_layer_to_qgis,
    list_qgis_layers,
    get_layer_columns,
    zoom_to_layer,
    remove_layer,
)
from .commons import now_utc
from .filters import (
    select_by_attribute,
    select_by_geometry,
)

from .geoprocessing import execute_processing

# Aggregate all tools for easy import
TOOLS = {
    t.name: t
    for t in [
        # Common tools
        now_utc,
        # Data I/O tools
        add_layer_to_qgis,
        list_qgis_layers,
        get_layer_columns,
        zoom_to_layer,
        remove_layer,
        # Filtering & Selection
        select_by_attribute,
        select_by_geometry,
        # Geoprocessing
        execute_processing,
    ]
}
