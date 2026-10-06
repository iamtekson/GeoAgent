# -*- coding: utf-8 -*-
"""Processing workflow end-to-end (real QGIS algorithms, scripted LLM)."""
import unittest
from unittest import mock

import _bootstrap
from fake_llm import FakeLLM, decomposition, params, selection, task
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from qgis.core import QgsProject, QgsVectorLayer

from geo_agent.agents import workflow as workflow_module
from geo_agent.agents.graph import build_unified_graph
from geo_agent.agents.workflow import build_workflow_graph


def run(llm, query="test request"):
    app = build_workflow_graph(llm).compile()
    return app.invoke({"messages": [HumanMessage(content=query)]}, {"recursion_limit": 100})


def output_name_from_prompt(messages):
    """The layer name the gather prompt lists for task_1_output."""
    return messages[-1].content.split("task_1_output: ")[1].split("\n")[0]


def layer_name(layer_id):
    layer = QgsProject.instance().mapLayer(layer_id)
    return layer.name() if layer is not None else None


BUFFER_THEN_CLIP = [
    task(1, "Buffer rivers by 500 m", "buffer", ["buffer"]),
    task(2, "Clip DEM with the buffered rivers", "clip raster by mask layer", ["clip", "mask"], [1]),
]


class RoutingAndWiringTest(unittest.TestCase):
    def setUp(self):
        self.rivers, self.dem = _bootstrap.reset_project()

    def test_without_flag_routing_falls_back_to_llm_as_before(self):
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(*BUFFER_THEN_CLIP)],
            "RouteDecision": [dict(is_processing_task=True, reason="")] * 2,
            "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
            "ParameterGathering": [
                params(INPUT="rivers", DISTANCE=500),
                lambda m: params(INPUT="DEM", MASK=output_name_from_prompt(m)),
            ],
        })
        out = run(llm)
        self.assertTrue(all(r["success"] for r in out["task_results"].values()), out["task_results"])
        self.assertEqual(llm.names(), [
            "TaskDecomposition",
            "RouteDecision", "AlgorithmSelection", "ParameterGathering",
            "RouteDecision", "AlgorithmSelection", "ParameterGathering",
            "invoke",
        ])
        self.assertEqual(set(out["available_output_ids"]), {"task_1_output", "task_2_output"})
        self.assertEqual(out["task_results"][1]["step"]["algorithm"], "native:buffer")

    def test_flag_from_decomposition_skips_routing_call(self):
        tasks = [dict(t, is_geoprocessing=True) for t in BUFFER_THEN_CLIP]
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(*tasks)],
            "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
            "ParameterGathering": [params(INPUT="rivers", DISTANCE=500), params(INPUT="DEM", MASK="task_1_output")],
        })
        out = run(llm)
        self.assertNotIn("RouteDecision", llm.names())
        self.assertTrue(all(r["success"] for r in out["task_results"].values()), out["task_results"])

    def test_gather_prompt_lists_advanced_params_and_dependency(self):
        tasks = [dict(t, is_geoprocessing=True) for t in BUFFER_THEN_CLIP]
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(*tasks)],
            "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
            "ParameterGathering": [params(INPUT="rivers", DISTANCE=500), params(INPUT="DEM", MASK="task_1_output")],
        })
        run(llm)
        first, second = llm.prompts("ParameterGathering")
        basic, _, advanced = first.partition("Advanced parameters")
        self.assertIn("SEPARATE_DISJOINT", advanced)
        self.assertNotIn("SEPARATE_DISJOINT", basic)
        self.assertIn("use task_1_output", second)

    def test_slips_are_normalized_before_execution(self):
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(
                task(1, "Buffer rivers by 500 m", "buffer", ["buffer"], geo=True),
                task(2, "Clip DEM with it", "clip", ["clip"], [1], geo=True),
            )],
            "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
            "ParameterGathering": [
                dict(parameters={"input": "River", "DISTANCE": 500, "END_CAP_STYLE": "Flat", "OUTPUT": "TEMPORARY_OUTPUT"}),
                params(INPUT="dem", MASK="@task_1_output"),
            ],
        })
        out = run(llm)
        self.assertTrue(all(r["success"] for r in out["task_results"].values()), out["task_results"])
        buffer_params = out["task_results"][1]["step"]["parameters"]
        clip_params = out["task_results"][2]["step"]["parameters"]
        self.assertEqual(layer_name(buffer_params["INPUT"]), "rivers")
        self.assertEqual(buffer_params["END_CAP_STYLE"], 1)
        self.assertEqual(clip_params["MASK"], out["available_output_ids"]["task_1_output"])
        self.assertEqual(layer_name(clip_params["INPUT"]), "DEM")

    def test_repeated_run_uses_its_own_output(self):
        def script():
            return {
                "TaskDecomposition": [decomposition(
                    task(1, "Buffer rivers", "buffer", ["buffer"], geo=True),
                    task(2, "Clip DEM with it", "clip", ["clip"], [1], geo=True),
                )],
                "AlgorithmSelection": [selection("native:buffer"), selection("gdal:cliprasterbymasklayer")],
                "ParameterGathering": [
                    params(INPUT="rivers", DISTANCE=100),
                    lambda m: params(INPUT="DEM", MASK=output_name_from_prompt(m)),
                ],
            }

        first = run(FakeLLM(script()))
        second = run(FakeLLM(script()))
        self.assertNotEqual(
            first["available_outputs"]["task_1_output"], second["available_outputs"]["task_1_output"]
        )
        self.assertEqual(
            second["task_results"][2]["step"]["parameters"]["MASK"],
            second["available_output_ids"]["task_1_output"],
        )

    def test_layer_added_by_llm_task_feeds_dependent_task(self):
        @tool
        def fake_add_layer(path: str) -> str:
            """Add a vector layer."""
            QgsProject.instance().addMapLayer(QgsVectorLayer(path, "added_rivers", "ogr"))
            return "added"

        llm = FakeLLM(
            {
                "TaskDecomposition": [decomposition(
                    task(1, "Add the rivers file", geo=False),
                    task(2, "Buffer it by 50 m", "buffer", ["buffer"], [1], geo=True),
                )],
                "AlgorithmSelection": [selection("native:buffer")],
                "ParameterGathering": [params(INPUT="task_1_output", DISTANCE=50)],
            },
            tool_calls=[[{"name": "fake_add_layer", "args": {"path": _bootstrap.RIVERS_PATH}, "id": "c1"}], None],
        )
        with mock.patch.dict(workflow_module.TOOLS, {"fake_add_layer": fake_add_layer}, clear=True):
            out = run(llm)
        self.assertEqual(out["available_outputs"].get("task_1_output"), "added_rivers")
        self.assertEqual(out["task_results"][1]["output_layers"], [])  # summary unchanged
        self.assertEqual(layer_name(out["task_results"][2]["step"]["parameters"]["INPUT"]), "added_rivers")


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.rivers, _ = _bootstrap.reset_project()
        self.field = self.rivers.fields().names()[0]

    @staticmethod
    def extract_task():
        return decomposition(task(1, "Extract rivers", "extract", ["extract"], geo=True))

    def test_preflight_rejection_regathers_without_llm_analysis(self):
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(task(1, "Buffer rivers", "buffer", ["buffer"], geo=True))],
            "AlgorithmSelection": [selection("native:buffer")],
            "ParameterGathering": [params(INPUT="zzqq_unknown", DISTANCE=500), params(INPUT="rivers", DISTANCE=500)],
        })
        out = run(llm)
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])
        self.assertEqual(llm.count("ErrorAnalysis"), 0)
        self.assertEqual(llm.count("AlgorithmSelection"), 1)
        retry_prompt = llm.prompts("ParameterGathering")[1]
        self.assertIn("Could not load source layer", retry_prompt)
        self.assertIn("zzqq_unknown", retry_prompt)  # previous attempt shown

    def test_bad_parameter_retries_same_algorithm(self):
        llm = FakeLLM({
            "TaskDecomposition": [self.extract_task()],
            "AlgorithmSelection": [selection("native:extractbyattribute")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="no_such_field", OPERATOR=9),
                params(INPUT="rivers", FIELD=self.field, OPERATOR=9),
            ],
            "ErrorAnalysis": [dict(diagnosis="field missing", failure_kind="bad_parameter")],
        })
        out = run(llm)
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])
        self.assertEqual(out["task_results"][1]["algorithm"], "native:extractbyattribute")
        self.assertEqual(llm.count("AlgorithmSelection"), 1)

    def test_wrong_algorithm_reselects(self):
        llm = FakeLLM({
            "TaskDecomposition": [self.extract_task()],
            "AlgorithmSelection": [selection("native:extractbyattribute"), selection("native:extractbyexpression")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="no_such_field", OPERATOR=9),
                params(INPUT="rivers", EXPRESSION="1=1"),
            ],
            "ErrorAnalysis": [dict(diagnosis="wrong tool", failure_kind="wrong_algorithm")],
        })
        out = run(llm)
        self.assertEqual(out["task_results"][1]["algorithm"], "native:extractbyexpression")
        self.assertEqual(llm.count("AlgorithmSelection"), 2)

    def test_failed_analysis_falls_back_to_reselection(self):
        llm = FakeLLM({
            "TaskDecomposition": [self.extract_task()],
            "AlgorithmSelection": [selection("native:extractbyattribute"), selection("native:extractbyexpression")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="no_such_field", OPERATOR=9),
                params(INPUT="rivers", EXPRESSION="1=1"),
            ],
            "ErrorAnalysis": [RuntimeError("provider down")],
        })
        out = run(llm)
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])
        self.assertEqual(llm.count("AlgorithmSelection"), 2)

    def test_same_algorithm_retried_only_once(self):
        llm = FakeLLM({
            "TaskDecomposition": [self.extract_task()],
            "AlgorithmSelection": [selection("native:extractbyattribute"), selection("native:extractbyexpression")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="bad1", OPERATOR=9),
                params(INPUT="rivers", FIELD="bad2", OPERATOR=9),
                params(INPUT="rivers", EXPRESSION="1=1"),
            ],
            "ErrorAnalysis": [dict(diagnosis="x", failure_kind="bad_parameter")] * 2,
        })
        out = run(llm)
        self.assertEqual(out["task_results"][1]["algorithm"], "native:extractbyexpression")
        self.assertEqual(llm.count("AlgorithmSelection"), 2)
        self.assertEqual(llm.count("ErrorAnalysis"), 2)


class LayerListingTest(unittest.TestCase):
    def test_every_layer_kind_is_listed(self):
        from qgis.core import QgsAnnotationLayer
        from geo_agent.agents.geoprocessing_flow import _available_layers_text

        _bootstrap.reset_project()
        project = QgsProject.instance()
        project.addMapLayer(QgsVectorLayer("None?field=a:integer", "table", "memory"))
        project.addMapLayer(
            QgsAnnotationLayer("notes", QgsAnnotationLayer.LayerOptions(project.transformContext()))
        )
        listing = _available_layers_text().splitlines()
        self.assertIn("- rivers (vector, Line)", listing)
        self.assertIn("- DEM (raster)", listing)
        self.assertIn("- table (vector, No geometry)", listing)
        self.assertIn("- notes (QgsAnnotationLayer)", listing)


class CheckpointTest(unittest.TestCase):
    def test_steps_survive_the_checkpointer(self):
        _bootstrap.reset_project()
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(task(1, "Buffer rivers", "buffer", ["buffer"], geo=True))],
            "AlgorithmSelection": [selection("native:buffer")],
            "ParameterGathering": [params(INPUT="rivers", DISTANCE=500)],
        })
        app = build_unified_graph(llm, mode="processing")
        config = {"configurable": {"thread_id": "checkpoint-test"}, "recursion_limit": 100}
        app.invoke({"messages": [HumanMessage(content="buffer rivers 500 m")]}, config)
        values = app.get_state(config).values
        self.assertEqual(values["task_results"][1]["step"]["algorithm"], "native:buffer")
        self.assertEqual(values["user_query"], "buffer rivers 500 m")


if __name__ == "__main__":
    unittest.main()
