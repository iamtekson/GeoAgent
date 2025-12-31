# -*- coding: utf-8 -*-
"""
Canvas refresh utilities for thread-safe QGIS map canvas updates.

This module provides a RefreshDispatcher QObject for safe canvas refresh
operations on the main Qt thread, along with getter/setter functions for
managing the QGIS interface and refresh callback references.
"""
from qgis.PyQt.QtCore import QObject, pyqtSlot
from qgis.core import Qgis, QgsMessageLog, QgsProject

# Global references (will be updated from the geo_agent module)
_qgis_iface = None
_refresh_callback = None


class RefreshDispatcher(QObject):
    """Dispatches a safe canvas refresh on the main Qt thread."""

    def __init__(self, iface):
        super().__init__()
        self.iface = iface

    @pyqtSlot()
    def doRefresh(self):
        """
        Refresh the map canvas by syncing layers with the project and forcing a redraw.
        This method runs on the main Qt thread via QMetaObject.invokeMethod.
        """
        try:
            canvas = self.iface.mapCanvas()
            if not canvas:
                return

            # Sync canvas layer stack with project layers and force redraw
            project_layers = list(QgsProject.instance().mapLayers().values())
            # canvas.setRenderFlag(False)
            canvas.setLayers(project_layers)
            # canvas.setRenderFlag(True)
            canvas.clearCache()
            # canvas.refreshAllLayers()
            canvas.refresh()
            # canvas.update()
            # canvas.repaint()
        except Exception as e:
            try:
                QgsMessageLog.logMessage(str(e), "GeoAgent", level=Qgis.Warning)
            except Exception:
                pass


def set_qgis_interface(iface):
    """Set the QGIS interface reference for tools to use."""
    global _qgis_iface
    _qgis_iface = iface


def get_qgis_interface():
    """Get the QGIS interface reference."""
    return _qgis_iface


def set_refresh_callback(callback):
    """Set a callback function for canvas refresh."""
    global _refresh_callback
    _refresh_callback = callback


def get_refresh_callback():
    """Get the current refresh callback function."""
    return _refresh_callback


def refresh_map_canvas():
    """Request canvas refresh via callback to main thread."""
    if _refresh_callback:
        _refresh_callback()



