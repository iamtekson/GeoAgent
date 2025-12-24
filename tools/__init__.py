from .geoprocessing import *
from .io import *
from .commons import *


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
        # Geoprocessing - Geometric operations
        buffer_layer,
        clip_layer,
        dissolve_layer,
        # Geoprocessing - Filtering & Selection
        select_by_attribute,
        select_by_geometry,
    ]
}
