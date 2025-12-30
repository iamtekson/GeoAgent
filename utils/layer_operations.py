# -*- coding: utf-8 -*-
"""
Layer operation utilities for thread-safe QGIS layer operations.

This module provides a LayerRemovalDispatcher QObject for safe layer removal
operations on the main Qt thread, along with getter/setter functions for
managing the layer removal callback reference.
"""
from qgis.PyQt.QtCore import QObject, pyqtSlot, pyqtSignal
from qgis.core import Qgis, QgsMessageLog, QgsProject

# Global reference (will be updated from the geo_agent module)
_layer_removal_callback = None


class LayerRemovalDispatcher(QObject):
    """Dispatches safe layer removal operations on the main Qt thread."""
    
    result_ready = pyqtSignal()

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.result = {"success": False, "error": None}
        self.layer_id = None
        self.layer_name = None

    @pyqtSlot(str, str)
    def doRemoveLayer(self, layer_id, layer_name):
        """
        Remove a layer from QGIS project on the main Qt thread.
        This method runs on the main Qt thread via QMetaObject.invokeMethod.
        
        Args:
            layer_id: ID of the layer to remove
            layer_name: Name of the layer (for logging)
        """
        self.result = {"success": False, "error": None}
        self.layer_id = layer_id
        self.layer_name = layer_name
        
        try:
            project = QgsProject.instance()
            
            # check if layer still exists
            if layer_id not in project.mapLayers():
                self.result["error"] = f"Layer '{layer_name}' not found in project."
                self.result_ready.emit()
                return
            
            # remove the layer from the project
            project.removeMapLayer(layer_id)
            
            # refresh canvas to reflect changes
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
            try:
                QgsMessageLog.logMessage(str(e), "GeoAgent", level=Qgis.Warning)
            except:
                pass
            self.result_ready.emit()


def set_layer_removal_callback(callback):
    """Set a callback function for layer removal."""
    global _layer_removal_callback
    _layer_removal_callback = callback


def get_layer_removal_callback():
    """Get the current layer removal callback function."""
    return _layer_removal_callback


def remove_layer_on_main_thread(layer_id, layer_name):
    """Request layer removal via callback to main thread."""
    if _layer_removal_callback:
        return _layer_removal_callback(layer_id, layer_name)
    return {"success": False, "error": "Layer removal callback not set"}
