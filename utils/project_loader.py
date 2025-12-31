# -*- coding: utf-8 -*-
"""
Project loading utilities for thread-safe QGIS project operations.

This module provides a ProjectLoadDispatcher QObject for safe project load
operations on the main Qt thread, along with getter/setter functions for
managing the project load callback reference.
"""
from qgis.PyQt.QtCore import QObject, pyqtSlot, pyqtSignal
from qgis.core import Qgis, QgsMessageLog, QgsProject

# Global reference (will be updated from the geo_agent module)
_project_load_callback = None


class ProjectLoadDispatcher(QObject):
    """Dispatches safe project load operations on the main Qt thread."""
    
    result_ready = pyqtSignal()

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.result = {"success": False, "error": None}
        self.path = None

    @pyqtSlot(str)
    def doLoadProject(self, path):
        """
        Load a QGIS project file on the main Qt thread.
        This method runs on the main Qt thread via QMetaObject.invokeMethod.
        
        Args:
            path: Path to the project file
        """
        self.result = {"success": False, "error": None}
        self.path = path
        
        try:
            project = QgsProject.instance()
            
            # block signals to prevent duplicate layer tree entries during load
            project.blockSignals(True)
            project.clear()
            success = project.read(path)
            project.blockSignals(False)
            
            if not success:
                self.result["error"] = f"Failed to load project from '{path}'. The file may be corrupted or incompatible."
                self.result_ready.emit()
                return
            
            # trigger UI updates
            try:
                project.readProject.emit()
            except Exception as e:
                QgsMessageLog.logMessage(f"Failed to emit readProject signal: {e}", "GeoAgent", level=Qgis.Warning)
            
            # refresh layers and canvas
            for layer in project.mapLayers().values():
                layer.triggerRepaint()
            
            # sync canvas with the new project
            canvas = self.iface.mapCanvas()
            if canvas:
                project_layers = list(project.mapLayers().values())
                canvas.setLayers(project_layers)
                canvas.clearCache()
                canvas.refresh()
            
            self.result["success"] = True
            self.result_ready.emit()
            
        except Exception as e:
            self.result["error"] = str(e)
            QgsMessageLog.logMessage(str(e), "GeoAgent", level=Qgis.Warning)
            self.result_ready.emit()


def set_project_load_callback(callback):
    """Set a callback function for project loading."""
    global _project_load_callback
    _project_load_callback = callback


def get_project_load_callback():
    """Get the current project load callback function."""
    return _project_load_callback


def load_project_on_main_thread(path):
    """Request project load via callback to main thread."""
    if _project_load_callback:
        return _project_load_callback(path)
    return {"success": False, "error": "Project load callback not set"}
