# -*- coding: utf-8 -*-
"""Plugin-level flow: real GeoAgent class and worker thread, stub QGIS iface."""
import importlib.util
import logging
import os
import shutil
import tempfile
import unittest
from unittest import mock

import _bootstrap
from fake_llm import SUMMARY, USAGE, FakeLLM, decomposition, params, selection, task
from qgis.PyQt import sip
from qgis.PyQt.QtCore import QEventLoop, QSettings, QTimer, QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QFileDialog, QMainWindow, QTextBrowser
from qgis.core import Qgis

import geo_agent.geo_agent as plugin_module
from geo_agent.agents.graph import build_unified_graph
from geo_agent.dialogs.geo_agent_dialog import GeoAgentDialog
from geo_agent.logger.logger import UILogHandler

SCHEME = plugin_module.MODEL_LINK_SCHEME


def buffer_and_clip_llm():
    return FakeLLM({
        "TaskDecomposition": [decomposition(
            task(1, "Buffer rivers by 500 m", "buffer", ["buffer"], geo=True),
            task(2, "Clip DEM with the buffer", "clip", ["clip"], [1], geo=True),
        )],
        "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
        "ParameterGathering": [params(INPUT="rivers", DISTANCE=500), params(INPUT="DEM", MASK="task_1_output")],
    })


def buffer_only_llm():
    return FakeLLM({
        "TaskDecomposition": [decomposition(task(1, "Buffer rivers by 100 m", "buffer", ["buffer"], geo=True))],
        "AlgorithmSelection": [selection("native:buffer")],
        "ParameterGathering": [params(INPUT="rivers", DISTANCE=100)],
    })


class PluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _bootstrap.reset_project()
        cls.main_window = QMainWindow()
        cls.iface = mock.MagicMock()
        cls.iface.mainWindow.return_value = cls.main_window
        cls.plugin = plugin_module.GeoAgent(cls.iface)
        # Error popups are modal; headless they would block forever
        cls.plugin.showMessage = mock.MagicMock()
        cls.plugin.initGui()
        cls.plugin.run()
        cls.dlg = cls.plugin.dlg

    @classmethod
    def tearDownClass(cls):
        cls.plugin.unload()
        # Detach the log handlers the plugin attached (console, file, Logs
        # tab) so later tests' output stays readable
        agent_logger = logging.getLogger("geo_agent")
        for handler in list(agent_logger.handlers):
            agent_logger.removeHandler(handler)
            handler.close()
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if isinstance(handler, UILogHandler):
                root_logger.removeHandler(handler)

    def tearDown(self):
        self.dlg.after_run_action.setCurrentIndex(0)
        from processing.modeler.ModelerDialog import ModelerDialog

        for dialog in list(ModelerDialog.dlgs):
            try:
                dialog.setDirty(False)
                dialog.close()
            except RuntimeError:
                pass  # already deleted
        ModelerDialog.dlgs.clear()

    # ── helpers ─────────────────────────────────────────────────────────────
    def use_app(self, llm, mode):
        """Install a graph built on *llm*, as _initialize_agent would."""
        plugin, dlg = self.plugin, self.dlg
        plugin.llm = llm
        plugin.app = build_unified_graph(llm, mode=mode)
        plugin._agent_settings_used = plugin._agent_settings(mode)
        plugin.thread_id = f"plugin-test:{mode}"
        (dlg.processing_mode if mode == "processing" else dlg.general_mode).setChecked(True)

    def send(self, question, timeout_ms=120000):
        """Type *question*, press Send, and wait for the worker thread."""
        self.dlg.question.setText(question)
        self.plugin.send_message()
        worker = self.plugin._worker_thread
        if worker is not None and worker.isRunning():
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            QTimer.singleShot(timeout_ms, loop.quit)
            loop.exec()
        for _ in range(5):  # deliver queued signals and deferred actions
            QEventLoop().processEvents()

    def processing_run(self, llm=None):
        """Run a processing request; returns its model run id (or None)."""
        before = set(self.plugin._model_runs)
        self.use_app(llm or buffer_and_clip_llm(), "processing")
        self.send("buffer rivers 500 m and clip the DEM with it")
        new = set(self.plugin._model_runs) - before
        return new.pop() if new else None

    def click(self, href):
        self.dlg.llm_response.anchorClicked.emit(QUrl(href))

    @staticmethod
    def designer_models():
        from processing.modeler.ModelerDialog import ModelerDialog

        return [sorted(d.model().childAlgorithms()) for d in ModelerDialog.dlgs]

    # ── tests (numbered: they share one plugin instance) ────────────────────
    def test_01_chat_bar_has_no_model_buttons(self):
        for name in ("export_ans", "clear_ans", "send_chat", "question", "general_mode", "processing_mode"):
            self.assertTrue(hasattr(self.dlg, name), name)
        self.assertFalse(hasattr(self.dlg, "open_model"))
        self.assertFalse(hasattr(self.dlg, "save_model"))

    def test_02_after_run_setting_defaults_to_just_showing(self):
        combo = self.dlg.after_run_action
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())],
            ["Just show the result", "Open in Model Designer", "Ask to save as .model3"],
        )
        self.assertEqual(self.dlg.get_after_run_action(), "none")

    def test_03_processing_result_has_model_links(self):
        run_id = self.processing_run()
        self.assertIsNotNone(run_id)
        html = self.dlg.llm_response.toHtml()
        self.assertIn(f"{SCHEME}:open/{run_id}", html)
        self.assertIn(f"{SCHEME}:save/{run_id}", html)
        self.assertIn(SUMMARY, self.dlg.llm_response.toPlainText())
        self.assertEqual(self.dlg.send_chat.text(), "Send")
        self.assertTrue(self.dlg.send_chat.isEnabled())
        self.assertEqual(self.designer_models(), [])  # default: nothing opens

    def test_04_open_link_opens_model_designer(self):
        run_id = self.processing_run()
        self.iface.messageBar().pushMessage.reset_mock()
        self.click(f"{SCHEME}:open/{run_id}")
        self.assertEqual(self.designer_models(), [["step_1", "step_2"]])
        levels = [c.kwargs.get("level") for c in self.iface.messageBar().pushMessage.call_args_list]
        self.assertNotIn(plugin_module.Qgis.Critical, levels)

    def test_05_save_link_writes_model3(self):
        run_id = self.processing_run()
        folder = tempfile.mkdtemp(prefix="geoagent-save-")
        target = os.path.join(folder, "my_model")
        try:
            with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(target, "")):
                self.click(f"{SCHEME}:save/{run_id}")
            self.assertTrue(os.path.exists(target + ".model3"))
            self.assertIn("Model saved to", str(self.iface.messageBar().pushMessage.call_args))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_06_each_link_exports_its_own_run(self):
        first = self.processing_run(buffer_only_llm())
        second = self.processing_run(buffer_and_clip_llm())
        self.click(f"{SCHEME}:open/{first}")
        self.click(f"{SCHEME}:open/{second}")
        self.assertEqual(self.designer_models(), [["step_1"], ["step_1", "step_2"]])

    def test_07_setting_opens_designer_after_each_run(self):
        self.dlg.after_run_action.setCurrentIndex(1)
        self.processing_run()
        self.assertEqual(self.designer_models(), [["step_1", "step_2"]])

    def test_08_setting_asks_to_save_after_each_run(self):
        self.dlg.after_run_action.setCurrentIndex(2)
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=("", "")) as dialog:
            self.processing_run()
        dialog.assert_called_once()

    def test_09_no_links_without_a_geoprocessing_step(self):
        llm = FakeLLM({"TaskDecomposition": [decomposition(task(1, "List the layers", geo=False))]})
        self.assertIsNone(self.processing_run(llm))
        last_reply = self.dlg.llm_response.toHtml().split("Agent:")[-1]
        self.assertNotIn(f"{SCHEME}:", last_reply)

    def test_10_general_mode_unaffected(self):
        self.use_app(FakeLLM({}), "general")
        self.send("what layers are loaded?")
        last_reply = self.dlg.llm_response.toHtml().split("Agent:")[-1]
        self.assertIn(SUMMARY, last_reply)
        self.assertNotIn(f"{SCHEME}:", last_reply)

    def test_11_token_usage_logged_after_each_request(self):
        llm = buffer_and_clip_llm()
        session_before = self.plugin._session_tokens
        with self.assertLogs("geo_agent.chat", level="INFO") as logs:
            self.processing_run(llm)
        used = USAGE["total_tokens"] * len(llm.calls)
        line = [r for r in logs.output if "Tokens used:" in r]
        self.assertEqual(len(line), 1, logs.output)
        self.assertIn(f"Tokens used: {used:,} (input", line[0])
        self.assertIn(f"in {len(llm.calls)} LLM calls", line[0])
        self.assertIn(f"total this QGIS session: {session_before + used:,}", line[0])
        # It is the last line of the request
        self.assertIn("Tokens used:", logs.output[-1])

    def test_12_web_links_open_in_browser(self):
        with mock.patch.object(QDesktopServices, "openUrl") as open_url:
            self.click("https://qgis.org")
        open_url.assert_called_once()

    def test_13_clear_chat_forgets_runs(self):
        self.processing_run()
        self.plugin.clear_chat()
        self.assertEqual(self.plugin._model_runs, {})
        self.assertNotIn(f"{SCHEME}:", self.dlg.llm_response.toHtml())

    def test_14_after_run_setting_is_saved(self):
        self.dlg.after_run_action.setCurrentIndex(2)
        self.dlg._save_settings()
        reopened = GeoAgentDialog(None)
        try:
            self.assertEqual(reopened.get_after_run_action(), "save")
        finally:
            self.dlg.after_run_action.setCurrentIndex(0)
            self.dlg._save_settings()
            reopened.deleteLater()

    def fake_initialize(self):
        """Stand-in for _initialize_agent: builds the app on a scripted LLM."""

        def initialize(model_name, temperature=None, max_tokens=None, mode="general"):
            self.plugin.llm = FakeLLM({})
            self.plugin.app = build_unified_graph(self.plugin.llm, mode=mode)

        return mock.patch.object(self.plugin, "_initialize_agent", side_effect=initialize)

    def test_15_first_message_builds_agent_once(self):
        self.dlg.general_mode.setChecked(True)
        self.plugin.app = None
        with self.fake_initialize() as initialize:
            self.send("hello")
        self.assertEqual(initialize.call_count, 1)

    def test_16_any_changed_setting_rebuilds_agent(self):
        self.use_app(FakeLLM({}), "general")
        with self.fake_initialize() as initialize:
            self.send("unchanged settings")
            self.assertEqual(initialize.call_count, 0)
            changes = [
                lambda: self.dlg.max_tokens.setValue(self.dlg.max_tokens.value() + 1000),
                lambda: self.dlg.model_name.setText("another-model"),
                lambda: self.dlg.custom_apikey.setText("another-key"),
                lambda: self.dlg.ollama_base_url.setText("http://example.org:11434"),
                lambda: self.dlg.ollama_model_name.setText("llama3.1:8b"),
            ]
            for expected_calls, change in enumerate(changes, start=1):
                change()
                self.send("after a settings change")
                self.assertEqual(initialize.call_count, expected_calls)
                self.assertEqual(initialize.call_args.kwargs["max_tokens"], self.dlg.max_tokens.value())
            self.send("unchanged again")
            self.assertEqual(initialize.call_count, len(changes))

    def test_18_error_log_falls_back_when_file_is_unwritable(self):
        saved_path = self.plugin._error_log_path
        # A folder in place of the log file: opening it for writing fails
        self.plugin._error_log_path = tempfile.mkdtemp(prefix="geoagent-not-a-file-")
        try:
            with mock.patch.object(plugin_module, "QgsMessageLog") as message_log:
                try:
                    raise ValueError("boom")
                except ValueError as e:
                    self.plugin._log_error("test", e)  # must not raise
        finally:
            shutil.rmtree(self.plugin._error_log_path, ignore_errors=True)
            self.plugin._error_log_path = saved_path
        logged = message_log.logMessage.call_args.args[0]
        self.assertIn("boom", logged)
        self.assertIn("Could not write", logged)

    def test_19_corrupted_saved_setting_does_not_block_the_panel(self):
        settings = QSettings()
        settings.setValue("GeoAgent/temperature", "not-a-number")
        try:
            with self.assertLogs("geo_agent.ui", level="WARNING") as logs:
                dialog = GeoAgentDialog(None)
            self.assertIn("Could not restore saved settings", logs.output[0])
            self.assertAlmostEqual(dialog.temperature.value(), 0.8)  # default kept
            dialog.deleteLater()
        finally:
            settings.remove("GeoAgent/temperature")

    def test_20_qgis_messages_keep_their_level_in_the_logs(self):
        expected = {
            Qgis.MessageLevel.Info: "INFO",
            Qgis.MessageLevel.Success: "INFO",
            Qgis.MessageLevel.Warning: "WARNING",
            Qgis.MessageLevel.Critical: "ERROR",
        }
        for qgis_level, log_level in expected.items():
            with self.assertLogs("geo_agent.ui", level="DEBUG") as logs:
                self.plugin._on_qgis_message("hello", "Test", qgis_level)
            self.assertEqual(logs.records[0].levelname, log_level, qgis_level)

    def test_99_unload_twice_is_safe(self):
        # Runs last: tearDownClass unloads again
        self.plugin.unload()
        self.assertFalse(self.plugin._message_log_connected)

    @unittest.skipUnless(importlib.util.find_spec("langchain_openai"), "langchain_openai not installed")
    def test_17_initialize_agent_builds_both_modes(self):
        # Constructing the client makes no network call
        self.dlg.model.setCurrentIndex(self.dlg.model.findText("OpenAI"))
        self.dlg.custom_apikey.setText("sk-test-not-real")
        for mode in ("processing", "general"):
            self.plugin._initialize_agent("OpenAI", temperature=0.2, max_tokens=100, mode=mode)
            self.assertIsNotNone(self.plugin.app)
            self.assertTrue(self.plugin.thread_id.endswith(f":{mode}"))


class UILogHandlerTest(unittest.TestCase):
    def test_deleted_logs_tab_is_dropped_not_crashed_on(self):
        browser = QTextBrowser()
        handler = UILogHandler(text_browser=browser)
        handler._append_to_browser("2026-01-01 [INFO] x: before")
        self.assertIn("before", browser.toPlainText())
        sip.delete(browser)  # e.g. the plugin was unloaded
        handler._append_to_browser("2026-01-01 [INFO] x: after")  # must not raise
        self.assertIsNone(handler.text_browser)
        handler._append_to_browser("2026-01-01 [INFO] x: later")  # no-op now


if __name__ == "__main__":
    unittest.main()
