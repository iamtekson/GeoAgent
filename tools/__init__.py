from .io import (
    add_layer_to_qgis,
    list_qgis_layers,
    get_layer_columns,
    zoom_to_layer,
    remove_layer,
    new_qgis_project,
    delete_existing_project
)
from .commons import now_utc
from .filters import (
    select_by_attribute,
    select_by_geometry,
)

from .geoprocessing import (
    execute_processing,
    list_processing_algorithms,
    get_algorithm_parameters,
    find_processing_algorithm,
)

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
        new_qgis_project,
        delete_existing_project,
        # Filtering & Selection
        select_by_attribute,
        select_by_geometry,
        # Geoprocessing
        # execute_processing,
        # list_processing_algorithms,
        # get_algorithm_parameters,
        # find_processing_algorithm,
    ]
}
