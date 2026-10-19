# -*- coding: utf-8 -*-
"""
Loading layers from a file path or URL, with QGIS detecting the format.

QGIS's own detection (the one its Data Source Manager uses) finds every layer
in a source: vector, raster, mesh or point cloud, and each table of a
multi-layer file such as a GeoPackage. Sources it doesn't scan, such as web
URLs, are tried with QGIS's vector reader, then its raster reader.
"""
from typing import List

from qgis.core import (
    Qgis,
    QgsProject,
    QgsProviderRegistry,
    QgsProviderSublayerDetails,
    QgsRasterLayer,
    QgsVectorLayer,
)


def layers_from_source(source: str) -> List:
    """Every valid layer QGIS finds in *source*, not yet added to the project.

    Args:
        source: File path or URL; 'file.gpkg|layername=roads' picks one table.

    Returns:
        The layers, named after their tables/files; [] if nothing loads.
    """
    options = QgsProviderSublayerDetails.LayerOptions(QgsProject.instance().transformContext())
    layers = []
    for details in QgsProviderRegistry.instance().querySublayers(source):
        if details.flags() & Qgis.SublayerFlag.SystemTable:
            continue
        layer = details.toLayer(options)
        if layer is not None and layer.isValid():
            layers.append(layer)
    if layers:
        return layers
    for layer in (QgsVectorLayer(source, "", "ogr"), QgsRasterLayer(source, "")):
        if layer.isValid():
            return [layer]
    return []


__all__ = ["layers_from_source"]
