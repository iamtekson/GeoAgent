# -*- coding: utf-8 -*-
"""
Regenerate the documentation screenshots from the real plugin UI.

Runs one real processing workflow on the demo data (buffer rivers, clip the
DEM, compute slope) through the GeoAgent dock, then renders the Chat,
Settings and Logs tabs, the Model Designer, and a map of the results into
docs/imgs/screenshots/. The LLM's replies come from the tests' scripted
stand-in; everything else (algorithms, outputs, model export, logs) is real.
Widgets are rendered off-screen: nothing appears on your desktop, and a
throw-away QGIS profile is used.

Run from the plugin folder with the Python that ships with QGIS:

    "C:\\Program Files\\QGIS 4.0.2\\bin\\python-qgis.bat" docs/scripts/take_screenshots.py
"""
import os
import sys

# Native rendering (offscreen has no fonts on Windows); widgets stay hidden
os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(PLUGIN_DIR, "docs", "imgs", "screenshots")
sys.path.insert(0, os.path.join(PLUGIN_DIR, "tests"))

import _bootstrap  # noqa: E402  (headless QGIS, demo data, `geo_agent` importable)
import fake_llm  # noqa: E402
from fake_llm import FakeLLM, decomposition, params, selection, task  # noqa: E402
from qgis.PyQt.QtCore import QEventLoop, QSize, Qt, QTimer  # noqa: E402
from qgis.PyQt.QtGui import QColor, QGuiApplication  # noqa: E402
from qgis.PyQt.QtWidgets import QMainWindow  # noqa: E402
from qgis.core import (  # noqa: E402
    Qgis,
    QgsHillshadeRenderer,
    QgsLineSymbol,
    QgsMapRendererSequentialJob,
    QgsMapSettings,
    QgsProject,
    QgsRasterBandStats,
    QgsSingleBandPseudoColorRenderer,
    QgsStyle,
)
from unittest import mock  # noqa: E402

import geo_agent.geo_agent as plugin_module  # noqa: E402
from geo_agent.agents.graph import build_unified_graph  # noqa: E402

DOCK_SIZE = QSize(1100, 560)
# Typical per-call token counts of a cloud model, so the usage line reads true
USAGE = {"input_tokens": 1830, "output_tokens": 96, "total_tokens": 1926}

GENERAL_QUESTION = "Which layers are loaded?"
GENERAL_REPLY = (
    "Two layers are loaded: **rivers** (line vector layer) and **DEM** "
    "(raster), both in EPSG:32644."
)
PROCESSING_QUESTION = (
    "Buffer the rivers by 500 m, clip the DEM with the buffer and compute "
    "the slope of the clipped DEM"
)
PROCESSING_REPLY = (
    "Buffered **rivers** by 500 m, clipped the **DEM** to the buffer, and "
    "computed the slope of the clipped DEM. The results were added to your "
    "map as *Buffered*, *Result - cliprasterbymasklayer* and *Result - slope*."
)


def processing_llm():
    return FakeLLM(
        {
            "TaskDecomposition": [decomposition(
                task(1, "Buffer the rivers by 500 m", "buffer", ["buffer", "distance"], geo=True),
                task(2, "Clip the DEM with the buffered rivers", "clip raster by mask layer",
                     ["clip", "mask", "raster"], [1], geo=True),
                task(3, "Compute the slope of the clipped DEM", "slope", ["slope", "terrain"], [2], geo=True),
            )],
            "AlgorithmSelection": [
                dict(selection("native:buffer"), reasoning="Buffers vector features by a fixed distance"),
                dict(selection("gdal:cliprasterbymasklayer"), reasoning="Clips a raster to a polygon mask layer"),
                dict(selection("native:slope"), reasoning="Computes slope from a DEM raster"),
            ],
            "ParameterGathering": [
                params(INPUT="rivers", DISTANCE=500),
                params(INPUT="DEM", MASK="task_1_output"),
                params(INPUT="task_2_output", Z_FACTOR=1),
            ],
        },
        usage=USAGE,
    )


class Screenshots:
    def __init__(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        try:  # match QGIS's default (light) look whatever the OS theme is
            QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Light)
        except AttributeError:
            pass
        _bootstrap.reset_project()
        self.main_window = QMainWindow()
        self.iface = mock.MagicMock()
        self.iface.mainWindow.return_value = self.main_window
        self.plugin = plugin_module.GeoAgent(self.iface)
        self.plugin.showMessage = mock.MagicMock()
        self.plugin.initGui()
        self.plugin.run()
        self.dlg = self.plugin.dlg
        self.dlg.resize(DOCK_SIZE)
        self.dlg.llm_response.clear()

    # ── driving the plugin ──────────────────────────────────────────────────
    def ask(self, question, reply, llm, mode):
        fake_llm.SUMMARY = reply
        self.plugin.llm = llm
        self.plugin.app = build_unified_graph(llm, mode=mode)
        self.plugin._agent_settings_used = self.plugin._agent_settings(mode)
        (self.dlg.processing_mode if mode == "processing" else self.dlg.general_mode).setChecked(True)

        self.dlg.question.setText(question)
        self.plugin.send_message()
        worker = self.plugin._worker_thread
        if worker is not None and worker.isRunning():
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            QTimer.singleShot(120000, loop.quit)
            loop.exec()
        for _ in range(5):
            QEventLoop().processEvents()

    def save(self, widget, name, scroll_to_end=None):
        """Render *widget* to docs/imgs/screenshots/<name>.

        Hidden widgets only lay out when rendered, so a text view to show
        scrolled to its end is rendered once, scrolled, then rendered again.
        """
        if scroll_to_end is not None:
            widget.grab()
            bar = scroll_to_end.verticalScrollBar()
            bar.setValue(bar.maximum())
        path = os.path.join(OUT_DIR, name)
        widget.grab().save(path)
        print("wrote", os.path.relpath(path, PLUGIN_DIR))

    # ── screenshots ─────────────────────────────────────────────────────────
    def run(self):
        self.ask(GENERAL_QUESTION, GENERAL_REPLY, FakeLLM({}, usage=USAGE), "general")
        self.ask(PROCESSING_QUESTION, PROCESSING_REPLY, processing_llm(), "processing")
        self.dlg.question.setText("Dissolve the buffered rivers")

        self.dlg.tabWidget.setCurrentIndex(0)
        self.save(self.dlg, "chat.png", scroll_to_end=self.dlg.llm_response)

        self.dlg.tabWidget.setCurrentIndex(1)
        self.save(self.dlg, "settings.png")

        self.dlg.tabWidget.setCurrentIndex(2)
        self.save(self.dlg, "logs.png", scroll_to_end=self.dlg.geoagent_logs)

        self.model_designer()
        self.result_map()

    def model_designer(self):
        from processing.modeler.ModelerDialog import ModelerDialog

        run_id = max(self.plugin._model_runs)
        self.plugin.open_model_in_designer(run_id)
        designer = ModelerDialog.dlgs[-1]
        designer.resize(1280, 760)
        for _ in range(3):
            QEventLoop().processEvents()
        try:
            designer.view().zoomFull()
        except AttributeError:
            pass
        self.save(designer, "model_designer.png")
        designer.setDirty(False)
        designer.close()

    def result_map(self):
        project = QgsProject.instance()

        def layer(name):
            return project.mapLayersByName(name)[0]

        rivers, dem = layer("rivers"), layer("DEM")
        slope = layer("Result - slope")  # already shows the buffer's footprint

        dem.setRenderer(QgsHillshadeRenderer(dem.dataProvider(), 1, 315, 45))
        dem.setOpacity(0.55)

        stats = slope.dataProvider().bandStatistics(1, QgsRasterBandStats.Stats.Min | QgsRasterBandStats.Stats.Max)
        renderer = QgsSingleBandPseudoColorRenderer(slope.dataProvider(), 1)
        renderer.setClassificationMin(stats.minimumValue)
        renderer.setClassificationMax(stats.maximumValue)
        renderer.createShader(
            QgsStyle.defaultStyle().colorRamp("Magma"),
            Qgis.ShaderInterpolationMethod.Linear,
            Qgis.ShaderClassificationMethod.Continuous,
            5,
        )
        slope.setRenderer(renderer)

        rivers.renderer().setSymbol(QgsLineSymbol.createSimple({"color": "#2b83ba", "width": "0.5"}))

        settings = QgsMapSettings()
        settings.setLayers([rivers, slope, dem])
        settings.setDestinationCrs(dem.crs())
        settings.setExtent(dem.extent())
        settings.setOutputSize(QSize(1000, int(1000 * dem.extent().height() / dem.extent().width())))
        settings.setBackgroundColor(QColor("white"))
        job = QgsMapRendererSequentialJob(settings)
        job.start()
        job.waitForFinished()
        path = os.path.join(OUT_DIR, "result_map.jpg")  # hillshade compresses badly as PNG
        job.renderedImage().save(path, "JPG", 88)
        print("wrote", os.path.relpath(path, PLUGIN_DIR))


if __name__ == "__main__":
    shots = Screenshots()
    shots.run()
    shots.plugin.unload()
