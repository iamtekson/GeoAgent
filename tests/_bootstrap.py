# -*- coding: utf-8 -*-
"""
Headless QGIS bootstrap shared by the test modules (import it first).

- Starts one offscreen QgsApplication with a throw-away config directory, so
  tests never read or write your real QGIS settings, API keys or logs.
- Initializes Processing (native, GDAL, ... providers).
- Makes the plugin importable as `geo_agent`, whatever its folder is called.
- Copies the demo layers to a temp directory (GDAL writes .aux.xml side files
  next to rasters it reads; they shouldn't land in the repository).
"""
import atexit
import glob
import importlib.util
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_CONFIG_DIR = tempfile.mkdtemp(prefix="geoagent-test-config-")
os.environ["QGIS_CUSTOM_CONFIG_PATH"] = _CONFIG_DIR

from qgis.PyQt.QtCore import QSettings  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsApplication,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

APP = QgsApplication([], True)
APP.initQgis()
# Real QGIS always sets this; GeoAgent.__init__ reads it (isolated settings)
QSettings().setValue("locale/userLocale", "en_US")

sys.path.append(os.path.join(QgsApplication.prefixPath(), "python", "plugins"))
from processing.core.Processing import Processing  # noqa: E402

Processing.initialize()

if "geo_agent" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "geo_agent",
        os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["geo_agent"] = _module
    _spec.loader.exec_module(_module)

DATA_DIR = tempfile.mkdtemp(prefix="geoagent-test-data-")
_DEMO = os.path.join(PLUGIN_DIR, "paper", "demo_dataset")
for _path in glob.glob(os.path.join(_DEMO, "shp", "rivers.*")) + [
    os.path.join(_DEMO, "tiff", "DEM.tif")
]:
    shutil.copy(_path, DATA_DIR)
RIVERS_PATH = os.path.join(DATA_DIR, "rivers.shp")
DEM_PATH = os.path.join(DATA_DIR, "DEM.tif")


def reset_project():
    """Fresh project with the demo 'rivers' (lines) and 'DEM' (raster) layers."""
    project = QgsProject.instance()
    project.removeAllMapLayers()
    rivers = QgsVectorLayer(RIVERS_PATH, "rivers", "ogr")
    dem = QgsRasterLayer(DEM_PATH, "DEM")
    project.addMapLayers([rivers, dem])
    return rivers, dem


@atexit.register
def _cleanup():
    QgsProject.instance().removeAllMapLayers()
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree(_CONFIG_DIR, ignore_errors=True)
