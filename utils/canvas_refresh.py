# -*- coding: utf-8 -*-
"""
Canvas refresh utilities for thread-safe QGIS map canvas updates.

This module provides a RefreshDispatcher QObject for safe canvas refresh
operations on the main Qt thread, along with getter/setter functions for
managing the QGIS interface and refresh callback references.
"""
from qgis.PyQt.QtCore import QObject, pyqtSlot, QMetaObject, Qt, Q_ARG
from functools import wraps
import threading


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
        self._results = {}  # Dictionary to store results per thread ID
        self._lock = threading.Lock()  # Lock to protect the results dictionary

    @pyqtSlot(object, list, dict, int)
    def run_task(self, func, args, kwargs, thread_id):
        """Internal slot to execute the function and store result."""
        result = None
        error = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = e
        
        # Store the result/error for this specific thread
        with self._lock:
            self._results[thread_id] = (result, error)

def execute_on_main_thread(func, *args, **kwargs):
    """
    Call this from your Tool to safely run QGIS logic.
    It blocks the worker thread until the main thread finishes the task.
    """
    # We need a reference to the runner living on the main thread
    # In GeoAgent.__init__, you should create: self.main_runner = MainThreadRunner()
    runner = _global_main_runner 
    if not runner:
        raise RuntimeError("MainThreadRunner is not set. Please set it using set_main_runner().")
    
    # Get the current thread ID to isolate results
    thread_id = threading.get_ident()
    
    # This is the magic part: invokeMethod with BlockingQueuedConnection
    QMetaObject.invokeMethod(
        runner, 
        "run_task", 
        Qt.BlockingQueuedConnection,
        Q_ARG(object, func),
        Q_ARG(list, list(args)),
        Q_ARG(dict, kwargs),
        Q_ARG(int, thread_id)
    )
    
    # Retrieve and cleanup the result for this thread
    try:
        with runner._lock:
            if thread_id not in runner._results:
                raise RuntimeError(f"Thread {thread_id} result not found in runner")
            result, error = runner._results[thread_id]
        
        if error:
            raise error
        return result
    finally:
        # Always cleanup the result from the dictionary, even if an exception occurs
        with runner._lock:
            runner._results.pop(thread_id, None)

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

