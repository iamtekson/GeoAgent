# -*- coding: utf-8 -*-
"""
Loading layers from a file, folder or URL, with QGIS detecting the format.

QGIS's own detection (the one its Data Source Manager uses) finds every layer
in a source: vector, raster, mesh or point cloud, and each table of a
multi-layer file such as a GeoPackage. A folder is read the way QGIS's
Browser shows it (a shapefile's .dbf/.shx/.prj belong to its .shp, other
files are skipped). Sources QGIS doesn't scan, such as web URLs, are tried
with QGIS's vector reader, then its raster reader.
"""
import os
from typing import List

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    Qgis,
    QgsDirectoryItem,
    QgsProject,
    QgsProviderRegistry,
    QgsProviderSublayerDetails,
    QgsRasterLayer,
    QgsVectorLayer,
)

from .canvas_refresh import qgis_main_thread


def _folder_sources(folder: str, include_subfolders: bool) -> List[str]:
    """Paths of the datasets QGIS's Browser shows in *folder*."""
    sources = []
    for item in QgsDirectoryItem(None, os.path.basename(folder), folder).createChildren():
        if isinstance(item, QgsDirectoryItem):
            if include_subfolders:
                sources += _folder_sources(item.dirPath(), True)
        else:
            sources.append(item.path())
    return sources


def layers_from_source(source: str, include_subfolders: bool = False) -> List:
    """Every valid layer QGIS finds in *source*, not yet added to the project.

    Args:
        source: File, folder or URL; 'file.gpkg|layername=roads' picks one table.
        include_subfolders: For a folder, also read the folders inside it.

    Returns:
        The layers, named after their tables/files; [] if nothing loads.
    """
    if os.path.isdir(source):
        layers = [
            layer
            for path in _folder_sources(source, include_subfolders)
            for layer in layers_from_source(path)
        ]
        if layers:
            return layers
        # else maybe a dataset stored as a folder (e.g. a File Geodatabase)

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


def add_to_project(layers: List) -> None:
    """Add layers to the project on the QGIS main thread, from any thread.

    Loading (downloads, big files, processing) can then run in the background
    without freezing QGIS. Layers belong to the thread that created them, and
    Qt only lets that thread hand them over, so they are moved to the main
    thread here before it adds them.
    """
    main_thread = QCoreApplication.instance().thread()
    for layer in layers:
        if layer.thread() != main_thread:
            layer.moveToThread(main_thread)
    _add_on_main_thread(layers)


@qgis_main_thread
def _add_on_main_thread(layers: List) -> None:
    project = QgsProject.instance()
    for layer in layers:
        project.addMapLayer(layer)


__all__ = ["add_to_project", "layers_from_source"]
