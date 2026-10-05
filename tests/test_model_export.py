# -*- coding: utf-8 -*-
"""Exporting a processing run as a QGIS model (.model3)."""
import os
import shutil
import tempfile
import unittest

import _bootstrap
import processing
from fake_llm import FakeLLM, decomposition, params, selection, task
from langchain_core.messages import HumanMessage
from qgis.core import QgsProcessingModelAlgorithm

from geo_agent.agents.graph import build_unified_graph
from geo_agent.utils.model_export import (
    build_model,
    collect_steps,
    default_models_folder,
    open_in_model_designer,
    save_model,
    suggested_file_name,
)

QUERY = "buffer rivers by 500 m, clip the DEM with it and compute slope of the clip"


def run_three_step_workflow():
    """buffer rivers -> clip DEM by the buffer -> slope of the clip."""
    llm = FakeLLM({
        "TaskDecomposition": [decomposition(
            task(1, "Buffer rivers by 500 m", "buffer", ["buffer"], geo=True),
            task(2, "Clip DEM with the buffer", "clip", ["clip"], [1], geo=True),
            task(3, "Slope of the clipped DEM", "slope", ["slope"], [2], geo=True),
        )],
        "AlgorithmSelection": [
            selection("native:buffer"),
            selection("gdal:cliprasterbymasklayer"),
            selection("native:slope"),
        ],
        "ParameterGathering": [
            params(INPUT="rivers", DISTANCE=500, END_CAP_STYLE="Flat"),
            params(INPUT="DEM", MASK="task_1_output"),
            params(INPUT="task_2_output", Z_FACTOR=1),
        ],
    })
    app = build_unified_graph(llm, mode="processing")
    config = {"configurable": {"thread_id": "model-export-test"}, "recursion_limit": 100}
    app.invoke({"messages": [HumanMessage(content=QUERY)]}, config)
    return app.get_state(config).values


class ModelExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _bootstrap.reset_project()
        values = run_three_step_workflow()
        cls.steps = collect_steps(values["task_results"])
        cls.request = values["user_query"]

    def setUp(self):
        self.model = build_model(self.steps, self.request)
        self.children = self.model.childAlgorithms()
        self.tmp = tempfile.mkdtemp(prefix="geoagent-model-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_steps_collected_in_order(self):
        self.assertEqual(
            [s["algorithm"] for s in self.steps],
            ["native:buffer", "gdal:cliprasterbymasklayer", "native:slope"],
        )

    def test_one_child_per_step(self):
        self.assertEqual(sorted(self.children), ["step_1", "step_2", "step_3"])
        self.assertEqual(self.children["step_1"].description(), "1. Buffer rivers by 500 m")
        self.assertTrue(self.model.name().startswith("GeoAgent - buffer rivers"))

    def test_user_layers_become_model_inputs(self):
        self.assertEqual(sorted(self.model.parameterComponents()), ["dem", "rivers"])
        source = self.children["step_1"].parameterSources()["INPUT"][0]
        self.assertEqual(source.parameterName(), "rivers")
        defaults = {
            p.name(): p.defaultValue()
            for p in self.model.parameterDefinitions()
            if p.name() in ("rivers", "dem")
        }
        self.assertEqual(defaults, {"rivers": "rivers", "dem": "DEM"})

    def test_static_values_kept(self):
        sources = self.children["step_1"].parameterSources()
        self.assertEqual(sources["DISTANCE"][0].staticValue(), 500)
        self.assertEqual(sources["END_CAP_STYLE"][0].staticValue(), 1)

    def test_steps_wired_through_child_outputs(self):
        mask = self.children["step_2"].parameterSources()["MASK"][0]
        self.assertEqual((mask.outputChildId(), mask.outputName()), ("step_1", "OUTPUT"))
        slope_input = self.children["step_3"].parameterSources()["INPUT"][0]
        self.assertEqual((slope_input.outputChildId(), slope_input.outputName()), ("step_2", "OUTPUT"))

    def test_only_final_result_is_a_model_output(self):
        self.assertFalse(self.children["step_1"].modelOutputs())
        self.assertFalse(self.children["step_2"].modelOutputs())
        self.assertEqual(len(self.children["step_3"].modelOutputs()), 1)
        self.assertEqual(len(self.model.destinationParameterDefinitions()), 1)

    def test_layout_and_validity(self):
        ys = [self.children[c].position().y() for c in ("step_1", "step_2", "step_3")]
        self.assertLess(ys[0], ys[1])
        self.assertLess(ys[1], ys[2])
        for child_id in self.children:
            ok, issues = self.model.validateChildAlgorithm(child_id)
            self.assertTrue(ok, issues)

    def test_saved_model_reloads_and_runs(self):
        path = os.path.join(self.tmp, suggested_file_name(self.model))
        self.assertFalse(save_model(self.model, path))  # not a models folder
        reloaded = QgsProcessingModelAlgorithm()
        self.assertTrue(reloaded.fromFile(path))
        self.assertEqual(sorted(reloaded.childAlgorithms()), ["step_1", "step_2", "step_3"])

        output = reloaded.destinationParameterDefinitions()[0].name()
        result = processing.run(
            reloaded, {"rivers": "rivers", "dem": "DEM", output: "TEMPORARY_OUTPUT"}
        )
        self.assertTrue(os.path.exists(str(result.get(output))), result)

    def test_opens_in_model_designer(self):
        dialog = open_in_model_designer(self.model)
        try:
            self.assertEqual(sorted(dialog.model().childAlgorithms()), ["step_1", "step_2", "step_3"])
        finally:
            dialog.setDirty(False)
            dialog.close()

    def test_default_models_folder_exists(self):
        self.assertTrue(os.path.isdir(default_models_folder()))


class ModelExportEdgeCasesTest(unittest.TestCase):
    def setUp(self):
        self.rivers, _ = _bootstrap.reset_project()

    def test_no_steps_is_an_error(self):
        with self.assertRaises(ValueError):
            build_model([], "nothing")

    def test_failed_and_llm_tasks_are_not_collected(self):
        results = {
            1: {"success": True, "operation": "add layer", "summary": "ok"},
            2: {"success": False, "operation": "buffer", "step": {"algorithm": "native:buffer"}},
        }
        self.assertEqual(collect_steps(results), [])

    def test_list_of_user_layers_becomes_one_multilayer_input(self):
        steps = [{
            "task_id": 1,
            "operation": "Merge",
            "algorithm": "native:mergevectorlayers",
            "parameters": {"LAYERS": [self.rivers.id(), _bootstrap.RIVERS_PATH], "OUTPUT": "TEMPORARY_OUTPUT"},
            "outputs": {"OUTPUT": "not_a_project_layer"},
        }]
        model = build_model(steps, "merge")
        inputs = list(model.parameterComponents())
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(model.parameterDefinition(inputs[0]).defaultValue()), 2)


if __name__ == "__main__":
    unittest.main()
