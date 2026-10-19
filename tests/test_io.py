# -*- coding: utf-8 -*-
"""Adding layers (tools/io.py, utils/layer_loading.py): QGIS detects the format."""
import glob
import os
import shutil
import tempfile
import unittest
from unittest import mock

import _bootstrap
import processing
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsGeocoderResult,
    QgsGeometry,
    QgsMapSettings,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from qgis.PyQt.QtCore import QCoreApplication, QThread
from test_main_thread import in_worker_thread

import geo_agent.tools.io as io_tools
from geo_agent.tools.geoprocessing import execute_processing
from geo_agent.tools.io import add_layer_to_qgis, zoom_to_place
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import MainThreadRunner, set_main_runner, set_qgis_interface


class AddLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rivers, _ = _bootstrap.reset_project()
        cls.folder = tempfile.mkdtemp(prefix="geoagent-test-io-")
        points = _bootstrap.memory_layer("Point?crs=EPSG:4326&field=n:string", "pts", [(["a"], "POINT(1 1)")])
        cls.gpkg = os.path.join(cls.folder, "city.gpkg")
        processing.run("native:package", {"LAYERS": [rivers, points], "OUTPUT": cls.gpkg})
        # Formats the old extension lists didn't know
        cls.fgb = os.path.join(cls.folder, "rivers.fgb")
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "FlatGeobuf"
        QgsVectorFileWriter.writeAsVectorFormatV3(rivers, cls.fgb, QgsProject.instance().transformContext(), options)
        cls.asc = os.path.join(cls.folder, "dem.asc")
        processing.run("gdal:translate", {"INPUT": _bootstrap.DEM_PATH, "OUTPUT": cls.asc})
        cls.html = os.path.join(cls.folder, "report.html")
        with open(cls.html, "w") as f:
            f.write("<html><body>not a layer</body></html>")

    def setUp(self):
        self.previous_runner = canvas_refresh._global_main_runner
        set_main_runner(MainThreadRunner())
        QgsProject.instance().removeAllMapLayers()

    def tearDown(self):
        set_main_runner(self.previous_runner)

    @staticmethod
    def project_layers():
        return {layer.name(): layer for layer in QgsProject.instance().mapLayers().values()}

    def test_every_table_of_a_geopackage(self):
        reply = add_layer_to_qgis.invoke({"path_or_url": self.gpkg})
        self.assertEqual(sorted(self.project_layers()), ["pts", "rivers"], reply)

    def test_one_table_of_a_geopackage(self):
        add_layer_to_qgis.invoke({"path_or_url": self.gpkg + "|layername=pts", "layer_name": "points"})
        self.assertEqual(list(self.project_layers()), ["points"])

    def test_format_detected_by_qgis(self):
        add_layer_to_qgis.invoke({"path_or_url": self.fgb})
        add_layer_to_qgis.invoke({"path_or_url": self.asc})
        add_layer_to_qgis.invoke({"path_or_url": _bootstrap.DEM_PATH, "layer_name": "elevation"})
        layers = self.project_layers()
        self.assertIsInstance(layers["rivers"], QgsVectorLayer)
        self.assertIsInstance(layers["dem"], QgsRasterLayer)
        self.assertIsInstance(layers["elevation"], QgsRasterLayer)

    def test_folder_like_the_qgis_browser(self):
        folder = os.path.join(self.folder, "mixed")
        os.makedirs(os.path.join(folder, "sub"))
        for path in glob.glob(os.path.join(os.path.dirname(_bootstrap.RIVERS_PATH), "rivers.*")):
            shutil.copy(path, folder)  # .shp with its .dbf/.shx/.prj side files
        shutil.copy(_bootstrap.DEM_PATH, folder)
        shutil.copy(_bootstrap.DEM_PATH, os.path.join(folder, "sub", "deep.tif"))
        for name, text in (("notes.txt", "hello"), ("DEM.tif.aux.xml", "<PAMDataset/>")):
            with open(os.path.join(folder, name), "w") as f:
                f.write(text)

        reply = add_layer_to_qgis.invoke({"path_or_url": folder})
        self.assertEqual(sorted(self.project_layers()), ["DEM", "rivers"], reply)
        self.assertIn("Added 2 layer(s)", reply)

        QgsProject.instance().removeAllMapLayers()
        add_layer_to_qgis.invoke({"path_or_url": folder, "include_subfolders": True})
        self.assertEqual(sorted(self.project_layers()), ["DEM", "deep", "rivers"])

    def test_empty_folder(self):
        empty = tempfile.mkdtemp(prefix="geoagent-test-empty-")
        self.assertTrue(add_layer_to_qgis.invoke({"path_or_url": empty}).startswith("**Error:**"))

    def test_not_a_layer(self):
        for source in (self.html, os.path.join(self.folder, "missing.shp")):
            reply = add_layer_to_qgis.invoke({"path_or_url": source})
            self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertEqual(self.project_layers(), {})


class LoadingThreadTest(unittest.TestCase):
    """Loading runs where the tool is called (the LLM worker thread), so QGIS
    doesn't freeze; adding to the project runs on the main thread."""

    def setUp(self):
        _bootstrap.reset_project()
        self.main = QCoreApplication.instance().thread()

    def test_add_layer_loads_in_the_background(self):
        loaded_in = []
        real = io_tools.layers_from_source

        def spy(*args, **kwargs):
            loaded_in.append(QThread.currentThread())
            return real(*args, **kwargs)

        with mock.patch.object(io_tools, "layers_from_source", spy):
            reply, error = in_worker_thread(lambda: add_layer_to_qgis.invoke(
                {"path_or_url": _bootstrap.RIVERS_PATH, "layer_name": "rivers_bg"}
            ))
        self.assertIsNone(error)
        self.assertIn("Success", reply)
        self.assertNotEqual(loaded_in[0], self.main)
        layer = QgsProject.instance().mapLayersByName("rivers_bg")[0]
        self.assertEqual(layer.thread(), self.main)

    def test_processing_results_are_added_on_the_main_thread(self):
        result, error = in_worker_thread(lambda: execute_processing.invoke({
            "algorithm": "native:buffer",
            "parameters": {"INPUT": "rivers", "DISTANCE": 10, "OUTPUT": "TEMPORARY_OUTPUT"},
        }))
        self.assertIsNone(error)
        self.assertTrue(result["success"], result.get("error"))
        layer = QgsProject.instance().mapLayer(result["outputs"]["OUTPUT"])
        self.assertEqual(layer.thread(), self.main)


class FakeCanvas:
    def __init__(self):
        self.extent = None
        self.center = None

    def mapSettings(self):
        settings = QgsMapSettings()
        settings.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:3857"))
        return settings

    def setExtent(self, rect):
        self.extent = rect

    def setCenter(self, point):
        self.center = point

    def refresh(self):
        pass


class FakeIface:
    def __init__(self):
        self.canvas = FakeCanvas()

    def mapCanvas(self):
        return self.canvas


def place(name, x, y, viewport=None):
    """A geocoder result like QGIS's Nominatim geocoder returns (WGS 84)."""
    result = QgsGeocoderResult(
        name, QgsGeometry.fromPointXY(QgsPointXY(x, y)), QgsCoordinateReferenceSystem("EPSG:4326")
    )
    if viewport:
        result.setViewport(QgsRectangle(*viewport))
    return result


class ZoomToPlaceTest(unittest.TestCase):
    """The geocoder is replaced, so these tests stay offline."""

    def setUp(self):
        self.previous = (canvas_refresh._global_main_runner, canvas_refresh.get_qgis_interface())
        set_main_runner(MainThreadRunner())
        self.iface = FakeIface()
        set_qgis_interface(self.iface)

    def tearDown(self):
        set_main_runner(self.previous[0])
        set_qgis_interface(self.previous[1])

    def zoom(self, results):
        with mock.patch.object(io_tools, "_geocode", return_value=results):
            return zoom_to_place.invoke({"place": "kathmandu"})

    def test_zooms_to_the_place_extent_in_the_map_crs(self):
        reply = self.zoom([
            place("Kathmandu Metropolitan City, Nepal", 85.32, 27.71, (85.277, 27.668, 85.373, 27.750)),
            place("Kathmandu, Bagamati Province, Nepal", 85.29, 27.69, (85.188, 27.570, 85.566, 27.818)),
        ])
        self.assertIn("Zoomed to **Kathmandu Metropolitan City, Nepal**", reply)
        self.assertIn("Bagamati", reply)  # other matches are mentioned
        extent = self.iface.canvas.extent
        self.assertAlmostEqual(extent.xMinimum(), 85.277 * 111319.49, delta=1000)  # meters, not degrees

    def test_a_point_without_extent_keeps_the_scale(self):
        self.zoom([place("Eiffel Tower", 2.2945, 48.8583)])
        self.assertIsNone(self.iface.canvas.extent)
        self.assertAlmostEqual(self.iface.canvas.center.x(), 2.2945 * 111319.49, delta=1000)

    def test_nothing_found(self):
        reply = self.zoom([])
        self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertIsNone(self.iface.canvas.extent)


if __name__ == "__main__":
    unittest.main()
