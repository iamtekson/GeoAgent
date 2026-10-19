# -*- coding: utf-8 -*-
"""
A short description of the current map view for the model: its CRS, scale,
and extent in map units and in WGS 84 longitude/latitude.

It is added to every model call (not stored in the chat), so requests about
"this map", "the current extent" or "the visible area" need no follow-up
question, and stay right after a zoom.
"""
from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject

from ..logger.processing_logger import get_processing_logger
from .canvas_refresh import get_qgis_interface, qgis_main_thread

_logger = get_processing_logger()


@qgis_main_thread
def describe_map_view() -> str:
    """'Current map view: ...' for the QGIS map canvas; '' when there is none."""
    iface = get_qgis_interface()
    if not iface:
        return ""
    try:
        canvas = iface.mapCanvas()
        extent = canvas.extent()
        crs = canvas.mapSettings().destinationCrs()
        if extent.isEmpty() or not crs.isValid():
            return ""
        to_wgs84 = QgsCoordinateTransform(
            crs, QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance()
        )
        lonlat = to_wgs84.transformBoundingBox(extent)
        digits = 6 if crs.isGeographic() else 1
        corners = ",".join(
            f"{v:.{digits}f}"
            for v in (extent.xMinimum(), extent.xMaximum(), extent.yMinimum(), extent.yMaximum())
        )
        return (
            f"Current map view: {crs.authid()} ({crs.description()}), scale 1:{canvas.scale():,.0f}. "
            f"As a QGIS extent: {corners} [{crs.authid()}]. "
            f"In WGS 84: longitude {lonlat.xMinimum():.5f} to {lonlat.xMaximum():.5f}, "
            f"latitude {lonlat.yMinimum():.5f} to {lonlat.yMaximum():.5f}."
        )
    except Exception as e:
        _logger.debug(f"Map view unavailable: {e}")
        return ""


__all__ = ["describe_map_view"]
