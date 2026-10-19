# -*- coding: utf-8 -*-
"""The current map view given to the model (utils/map_view.py)."""
import unittest

import _bootstrap
from fake_llm import FakeLLM, decomposition, params, selection, task
from langchain_core.messages import HumanMessage, SystemMessage
from qgis.core import QgsCoordinateReferenceSystem, QgsMapSettings, QgsRectangle

from geo_agent.agents.graph import build_graph_app
from geo_agent.agents.workflow import build_workflow_graph
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import set_qgis_interface
from geo_agent.utils.map_view import describe_map_view

# Around Pokhara, Nepal, in Web Mercator
POKHARA_3857 = (9345000, 3271000, 9354000, 3280000)


class FakeCanvas:
    def __init__(self, crs, rect, scale):
        self.crs, self.rect, self.map_scale = QgsCoordinateReferenceSystem(crs), QgsRectangle(*rect), scale

    def extent(self):
        return self.rect

    def mapSettings(self):
        settings = QgsMapSettings()
        settings.setDestinationCrs(self.crs)
        return settings

    def scale(self):
        return self.map_scale


class FakeIface:
    def __init__(self, canvas):
        self.canvas = canvas

    def mapCanvas(self):
        return self.canvas


class MapViewBase(unittest.TestCase):
    def setUp(self):
        self.previous_iface = canvas_refresh.get_qgis_interface()
        set_qgis_interface(FakeIface(FakeCanvas("EPSG:3857", POKHARA_3857, 48000)))

    def tearDown(self):
        set_qgis_interface(self.previous_iface)


class DescribeMapViewTest(MapViewBase):
    def test_projected_view(self):
        view = describe_map_view()
        self.assertIn("EPSG:3857", view)
        self.assertIn("scale 1:48,000", view)
        self.assertIn("9345000.0,9354000.0,3271000.0,3280000.0 [EPSG:3857]", view)  # QGIS extent format
        self.assertRegex(view, r"longitude 83\.9\d+ to 84\.0\d+, latitude 28\.\d+ to 28\.\d+")

    def test_geographic_view(self):
        set_qgis_interface(FakeIface(FakeCanvas("EPSG:4326", (83.9, 28.1, 84.1, 28.3), 250000)))
        self.assertIn("83.900000,84.100000,28.100000,28.300000 [EPSG:4326]", describe_map_view())

    def test_without_a_map(self):
        set_qgis_interface(None)
        self.assertEqual(describe_map_view(), "")


class ModelSeesTheMapViewTest(MapViewBase):
    def test_general_mode_sends_it_with_each_call_without_storing_it(self):
        llm = FakeLLM({}, tool_calls=[None])
        app = build_graph_app(llm)
        config = {"configurable": {"thread_id": "map-view"}}
        app.invoke({"messages": [SystemMessage("You are GeoAgent."), HumanMessage("parks in this map?")]}, config)

        sent = llm.calls[0][1]
        self.assertTrue(sent[0].content.startswith("You are GeoAgent."))
        self.assertIn("Current map view: EPSG:3857", sent[0].content)
        stored = app.get_state(config).values["messages"]
        self.assertEqual(stored[0].content, "You are GeoAgent.")

    def test_processing_prompts_include_it(self):
        _bootstrap.reset_project()
        llm = FakeLLM(
            {
                "TaskDecomposition": [decomposition(
                    task(1, "Buffer rivers in the current extent by 10 m", "buffer", ["buffer"], geo=True),
                    task(2, "Say what was done", geo=False),
                )],
                "AlgorithmSelection": [selection("native:buffer")],
                "ParameterGathering": [params(INPUT="rivers", DISTANCE=10)],
            },
            tool_calls=[None],
        )
        build_workflow_graph(llm).compile().invoke(
            {"messages": [HumanMessage("buffer the rivers here")]}, {"recursion_limit": 100}
        )
        self.assertIn("Current map view: EPSG:3857", llm.prompts("ParameterGathering")[0])
        self.assertIn("Current map view: EPSG:3857", llm.prompts("invoke")[0])  # the tool-using task


if __name__ == "__main__":
    unittest.main()
