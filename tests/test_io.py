# -*- coding: utf-8 -*-
"""Adding layers (tools/io.py, utils/layer_loading.py): QGIS detects the format."""
import glob
import os
import shutil
import tempfile
import unittest

import _bootstrap
import processing
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorFileWriter, QgsVectorLayer

from geo_agent.tools.io import add_layer_to_qgis
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import MainThreadRunner, set_main_runner


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


if __name__ == "__main__":
    unittest.main()
