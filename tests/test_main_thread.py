# -*- coding: utf-8 -*-
"""Running QGIS-touching tools on the main thread (utils/canvas_refresh.py)."""
import threading
import time
import unittest
from unittest import mock

import _bootstrap
from qgis.PyQt.QtCore import QThread
from qgis.core import QgsApplication

from geo_agent.tools.io import list_qgis_layers
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import (
    MainThreadRunner,
    execute_on_main_thread,
    set_main_runner,
)

# What threading.get_ident() returns on macOS/Linux: too big for a 32-bit int
MACOS_THREAD_ID = 0x700001234000


def in_worker_thread(func, thread_id=None):
    """Run *func* in a worker thread while the main thread processes events.

    Returns (result, exception). *thread_id* fakes threading.get_ident().
    """
    outcome = {}

    def target():
        try:
            if thread_id is None:
                outcome["result"] = func()
            else:
                with mock.patch("threading.get_ident", return_value=thread_id):
                    outcome["result"] = func()
        except Exception as e:
            outcome["error"] = e

    worker = threading.Thread(target=target)
    worker.start()
    deadline = time.monotonic() + 30
    while worker.is_alive() and time.monotonic() < deadline:
        QgsApplication.processEvents()
        time.sleep(0.005)
    worker.join(timeout=1)
    return outcome.get("result"), outcome.get("error")


class MainThreadRunnerTest(unittest.TestCase):
    def setUp(self):
        self.previous_runner = canvas_refresh._global_main_runner
        self.runner = MainThreadRunner()
        set_main_runner(self.runner)
        _bootstrap.reset_project()

    def tearDown(self):
        set_main_runner(self.previous_runner)

    def test_tool_result_reaches_worker_with_macos_thread_ids(self):
        # Issue #62: "Thread <id> result not found in runner" on macOS/Linux
        result, error = in_worker_thread(lambda: list_qgis_layers.invoke({}), MACOS_THREAD_ID)
        self.assertIsNone(error)
        self.assertIn("rivers", result)
        self.assertIn("DEM", result)

    def test_function_runs_on_the_main_thread(self):
        main_thread = QThread.currentThread()
        result, error = in_worker_thread(
            lambda: execute_on_main_thread(lambda: QThread.currentThread() == main_thread)
        )
        self.assertIsNone(error)
        self.assertTrue(result)

    def test_exceptions_reach_the_caller(self):
        def fails():
            raise ValueError("layer not found")

        result, error = in_worker_thread(lambda: execute_on_main_thread(fails))
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "layer not found")

    def test_none_result_is_returned(self):
        result, error = in_worker_thread(lambda: execute_on_main_thread(lambda: None))
        self.assertIsNone(error)
        self.assertIsNone(result)

    def test_arguments_are_passed(self):
        result, error = in_worker_thread(
            lambda: execute_on_main_thread(lambda a, b=0: a + b, 2, b=3)
        )
        self.assertEqual((result, error), (5, None))

    def test_call_from_main_thread_runs_directly(self):
        # A blocking queued call to the own thread would deadlock
        self.assertEqual(execute_on_main_thread(lambda: "direct"), "direct")

    def test_without_runner_is_an_error(self):
        set_main_runner(None)
        with self.assertRaises(RuntimeError):
            execute_on_main_thread(lambda: None)


if __name__ == "__main__":
    unittest.main()
