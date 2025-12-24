from .geoprocessing import *
from .data_io import *
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
    ]
}
