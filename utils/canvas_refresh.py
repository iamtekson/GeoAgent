# -*- coding: utf-8 -*-
"""
Run QGIS-touching code on the main Qt thread from the LLM worker thread.

Provides the MainThreadRunner QObject and the @qgis_main_thread decorator
that tools use, along with getter/setter functions for the QGIS interface.
"""
from qgis.PyQt.QtCore import QObject, QThread, pyqtSlot, QMetaObject, Qt, Q_ARG
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

    @pyqtSlot(object)
    def run_task(self, call):
        """Run a call queued by execute_on_main_thread; its outcome goes back on it."""
        try:
            call["result"] = call["func"](*call["args"], **call["kwargs"])
        except Exception as e:  # re-raised in the calling thread
            call["error"] = e


def execute_on_main_thread(func, *args, **kwargs):
    """
    Call this from your Tool to safely run QGIS logic.
    It blocks the worker thread until the main thread finishes the task, then
    returns the function's result (or raises its exception).
    """
    # We need a reference to the runner living on the main thread
    # In GeoAgent.__init__, you should create: self.main_runner = MainThreadRunner()
    # and register it via set_main_runner(self.main_runner)
    runner = _global_main_runner
    if not runner:
        raise RuntimeError("MainThreadRunner is not set. Please set it using set_main_runner().")

    # Already on the main thread: a blocking queued call to it would deadlock
    if QThread.currentThread() == runner.thread():
        return func(*args, **kwargs)

    # The call travels as one object and comes back carrying its outcome.
    # (Results used to be looked up by thread id, passed through Qt as a
    # 32-bit int; macOS and Linux thread ids don't fit, so results were lost:
    # "Thread <id> result not found in runner", issue #62.)
    call = {"func": func, "args": args, "kwargs": kwargs}
    QMetaObject.invokeMethod(
        runner,
        "run_task",
        Qt.ConnectionType.BlockingQueuedConnection,
        Q_ARG(object, call),
    )
    if "error" in call:
        raise call["error"]
    if "result" not in call:  # the slot never ran
        raise RuntimeError("Could not run the call on the QGIS main thread")
    return call["result"]

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

