# -*- coding: utf-8 -*-
"""
Canvas refresh utilities for thread-safe QGIS map canvas updates.

This module provides a RefreshDispatcher QObject for safe canvas refresh
operations on the main Qt thread, along with getter/setter functions for
managing the QGIS interface and refresh callback references.
"""
from qgis.PyQt.QtCore import QObject, pyqtSlot, QMetaObject, Qt, Q_ARG
from functools import wraps


# Global references (will be updated from the geo_agent module)
_qgis_iface = None
_global_main_runner = None  # To be set to MainThreadRunner instance

def set_qgis_interface(iface):
    """Set the QGIS interface reference for tools to use."""
    global _qgis_iface
    _qgis_iface = iface


def get_qgis_interface():
    """Get the QGIS interface reference."""
    return _qgis_iface

class MainThreadRunner(QObject):
    """A single dispatcher to run ANY function on the QGIS main thread."""
    
    def __init__(self):
        super().__init__()

    @pyqtSlot(object, list, dict)
    def run_task(self, func, args, kwargs):
        """Internal slot to execute the function and store result."""
        self._result = None
        self._error = None
        try:
            self._result = func(*args, **kwargs)
        except Exception as e:
            self._error = e

def execute_on_main_thread(func, *args, **kwargs):
    """
    Call this from your Tool to safely run QGIS logic.
    It blocks the worker thread until the main thread finishes the task.
    """
    # We need a reference to the runner living on the main thread
    # In GeoAgent.__init__, you should create: self.main_runner = MainThreadRunner()
    runner = _global_main_runner 
    
    # This is the magic part: invokeMethod with BlockingQueuedConnection
    success = QMetaObject.invokeMethod(
        runner, 
        "run_task", 
        Qt.BlockingQueuedConnection,
        Q_ARG(object, func),
        Q_ARG(list, list(args)),
        Q_ARG(dict, kwargs)
    )
    
    if not success:
        raise RuntimeError(
            f"Failed to invoke method on main thread for function: {func.__name__}"
        )
    
    if runner._error:
        raise runner._error
    return runner._result

def set_main_runner(runner: MainThreadRunner):
    global _global_main_runner
    _global_main_runner = runner


def qgis_main_thread(func):
    """
    Decorator that automatically wraps a function to run 
    on the QGIS main thread using the Global Runner.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # We use the existing execute_on_main_thread logic
        return execute_on_main_thread(func, *args, **kwargs)
    return wrapper

