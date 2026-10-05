from .io import (
    add_layer_to_qgis,
    list_qgis_layers,
    get_layer_columns,
    zoom_to_layer,
    remove_layer,
    create_new_qgis_project,
    load_qgis_project,
    delete_existing_project,
    save_qgis_project
)
from .commons import now_utc
from .filters import (
    select_by_attribute,
    select_by_geometry,
)

# Geoprocessing tools are driven by the processing workflow
# (agents/geoprocessing_flow.py), not offered to the LLM via TOOLS
from .geoprocessing import (
    execute_processing,
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
        create_new_qgis_project,
        load_qgis_project,
        save_qgis_project,
        delete_existing_project,
        # Filtering & Selection
        select_by_attribute,
        select_by_geometry,
    ]
}
