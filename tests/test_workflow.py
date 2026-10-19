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
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import MainThreadRunner, set_main_runner
from geo_agent.utils.model_export import build_model, collect_steps


def run(llm, query="test request"):
    app = build_workflow_graph(llm).compile()
    return app.invoke({"messages": [HumanMessage(content=query)]}, {"recursion_limit": 100})


def output_name_from_prompt(messages):
    """The layer name the gather prompt lists for task_1_output."""
    return messages[-1].content.split("task_1_output: ")[1].split("\n")[0]


def layer_name(layer_id):
    layer = QgsProject.instance().mapLayer(layer_id)
    return layer.name() if layer is not None else None


# Passes the pre-flight check but fails when run ("begins with" needs a text
# column): exercises the LLM error-analysis paths
RUNTIME_FAILURE = dict(INPUT="rivers", FIELD="cat", OPERATOR=6, VALUE="1")

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

    def test_failed_routing_falls_back_on_the_decomposition(self):
        # A task with an algorithm hint/keywords is geoprocessing, others aren't
        llm = FakeLLM(
            {
                "TaskDecomposition": [decomposition(
                    task(1, "Buffer rivers by 10 m", "buffer", ["buffer"]),
                    task(2, "Tell me what was done"),
                )],
                "RouteDecision": [RuntimeError("provider down")] * 2,
                "AlgorithmSelection": [selection("native:buffer")],
                "ParameterGathering": [params(INPUT="rivers", DISTANCE=10)],
            },
            tool_calls=[None],
        )
        out = run(llm)
        self.assertEqual(out["task_results"][1]["algorithm"], "native:buffer")
        self.assertNotIn("algorithm", out["task_results"][2])  # handled by the LLM task

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
                params(**RUNTIME_FAILURE),
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
                params(**RUNTIME_FAILURE),
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
                params(**RUNTIME_FAILURE),
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
                params(**RUNTIME_FAILURE),
                params(**dict(RUNTIME_FAILURE, FIELD="value")),
                params(INPUT="rivers", EXPRESSION="1=1"),
            ],
            "ErrorAnalysis": [dict(diagnosis="x", failure_kind="bad_parameter")] * 2,
        })
        out = run(llm)
        self.assertEqual(out["task_results"][1]["algorithm"], "native:extractbyexpression")
        self.assertEqual(llm.count("AlgorithmSelection"), 2)
        self.assertEqual(llm.count("ErrorAnalysis"), 2)


class FilterStepTest(unittest.TestCase):
    """A filter that feeds a later step runs as native:extractbyexpression."""

    def setUp(self):
        self.rivers, _ = _bootstrap.reset_project()
        self.expected = sum(1 for f in self.rivers.getFeatures() if f["value"] > 200)

    @staticmethod
    def filter_then_buffer(*gathered):
        """Extract rivers (one gather answer per item), then buffer them."""
        return FakeLLM({
            "TaskDecomposition": [decomposition(
                task(1, "Extract rivers with flow above 200", "extract by expression", ["extract"], geo=True),
                task(2, "Buffer them by 10 m", "buffer", ["buffer"], [1], geo=True),
            )],
            "AlgorithmSelection": [selection("native:extractbyexpression"), selection("native:buffer")],
            "ParameterGathering": [
                g if isinstance(g, dict) else params(INPUT="rivers", EXPRESSION=g) for g in gathered
            ] + [params(INPUT="task_1_output", DISTANCE=10)],
        })

    def test_filtered_features_feed_the_next_step_and_the_model(self):
        llm = self.filter_then_buffer(dict(
            parameters={"INPUT": "rivers", "EXPRESSION": '"value" > 200', "OUTPUT": "TEMPORARY_OUTPUT"},
            user_note="Used column 'value' for 'flow'.",
        ))
        out = run(llm)
        results = out["task_results"]
        self.assertTrue(all(r["success"] for r in results.values()), results)
        self.assertEqual(results[1]["notes"], ["Used column 'value' for 'flow'."])
        self.assertIn("Note: Used column 'value'", llm.prompts("invoke")[-1])  # the summary is told

        buffered = QgsProject.instance().mapLayer(out["available_output_ids"]["task_2_output"])
        self.assertLess(self.expected, self.rivers.featureCount())
        self.assertEqual(buffered.featureCount(), self.expected)

        model = build_model(collect_steps(results), "filter then buffer")
        self.assertEqual(len(model.childAlgorithms()), 2)
        source = model.childAlgorithm("step_2").parameterSources()["INPUT"][0]
        self.assertEqual(source.outputChildId(), "step_1")

    def test_columns_listed_only_for_column_parameters(self):
        llm = self.filter_then_buffer('"value" > 200')
        run(llm)
        extract_prompt, buffer_prompt = llm.prompts("ParameterGathering")
        self.assertIn("Columns of vector layers", extract_prompt)
        self.assertIn("value (Integer64: 4 to 272)", extract_prompt)
        self.assertNotIn("Columns of vector layers", buffer_prompt)

    def test_misspelled_column_is_rejected_and_regathered(self):
        llm = self.filter_then_buffer('"valu" > 200', '"value" > 200')
        out = run(llm)
        self.assertTrue(all(r["success"] for r in out["task_results"].values()), out["task_results"])
        self.assertIn("Columns of 'rivers'", llm.prompts("ParameterGathering")[1])
        self.assertEqual(llm.count("ErrorAnalysis"), 0)

    def test_question_from_the_llm_stops_without_retries(self):
        llm = self.filter_then_buffer(dict(
            parameters={}, clarification="Do you mean 'cat' or 'value'?",
        ))
        out = run(llm)
        result = out["task_results"][1]
        self.assertFalse(result["success"])
        self.assertEqual(result["clarification"], "Do you mean 'cat' or 'value'?")
        self.assertEqual(llm.count("ParameterGathering"), 1)  # no blind retries
        self.assertEqual(llm.count("ErrorAnalysis"), 0)
        self.assertNotIn(2, out["task_results"])
        self.assertIn("stopped to ask", llm.prompts("invoke")[-1])

    def test_empty_result_stops_the_steps_that_use_it(self):
        out = run(self.filter_then_buffer('"value" > 1000'))
        result = out["task_results"][1]
        self.assertFalse(result["success"])
        self.assertIn("has no features", result["error"])
        self.assertNotIn(2, out["task_results"])  # nothing buffered from an empty layer

    def test_unknown_field_rejected_before_running(self):
        llm = FakeLLM({
            "TaskDecomposition": [decomposition(task(1, "Extract rivers", "extract", ["extract"], geo=True))],
            "AlgorithmSelection": [selection("native:extractbyattribute")],
            "ParameterGathering": [
                params(INPUT="rivers", FIELD="zzqq", OPERATOR=9),
                params(INPUT="rivers", FIELD="cat", OPERATOR=9),
            ],
        })
        out = run(llm)
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])
        self.assertEqual(llm.count("ErrorAnalysis"), 0)
        self.assertIn("Columns of 'rivers'", llm.prompts("ParameterGathering")[1])


class LlmTaskOutcomeTest(unittest.TestCase):
    """Tool results decide whether a tool-using task succeeded."""

    def setUp(self):
        self.rivers, _ = _bootstrap.reset_project()
        self.previous_runner = canvas_refresh._global_main_runner
        set_main_runner(MainThreadRunner())

    def tearDown(self):
        set_main_runner(self.previous_runner)

    def test_tool_error_fails_the_task_and_stops_the_run(self):
        llm = FakeLLM(
            {"TaskDecomposition": [decomposition(task(1, "Select x", geo=False), task(2, "List layers", geo=False))]},
            tool_calls=[[{"name": "fake_tool", "args": {}, "id": "c1"}], None],
        )

        @tool
        def fake_tool() -> str:
            """A tool."""
            return "**Error:** Field **x** not found."

        with mock.patch.dict(workflow_module.TOOLS, {"fake_tool": fake_tool}, clear=True):
            out = run(llm)
        self.assertFalse(out["task_results"][1]["success"])
        self.assertIn("Field **x** not found", out["task_results"][1]["error"])
        self.assertNotIn(2, out["task_results"])

    def test_error_then_success_is_a_success(self):
        replies = iter(["**Error:** no column 'x'", "**Success:** Selected 2 features."])

        @tool
        def fake_tool() -> str:
            """A tool."""
            return next(replies)

        llm = FakeLLM(
            {"TaskDecomposition": [decomposition(task(1, "Select x", geo=False))]},
            tool_calls=[[{"name": "fake_tool", "args": {}, "id": "c1"}], [{"name": "fake_tool", "args": {}, "id": "c2"}], None],
        )
        with mock.patch.dict(workflow_module.TOOLS, {"fake_tool": fake_tool}, clear=True):
            out = run(llm)
        self.assertTrue(out["task_results"][1]["success"], out["task_results"])

    def test_selection_feeding_a_processing_step_is_saved_as_a_layer(self):
        expected = sum(1 for f in self.rivers.getFeatures() if f["value"] > 200)
        llm = FakeLLM(
            {
                "TaskDecomposition": [decomposition(
                    task(1, "Select rivers with value above 200", geo=False),
                    task(2, "Buffer them by 10 m", "buffer", ["buffer"], [1], geo=True),
                )],
                "AlgorithmSelection": [selection("native:buffer")],
                "ParameterGathering": [params(INPUT="task_1_output", DISTANCE=10)],
            },
            tool_calls=[[{
                "name": "select_by_expression",
                "args": {"layer_name": "rivers", "expression": '"value" > 200'},
                "id": "c1",
            }], None],
        )
        out = run(llm)
        self.assertTrue(all(r["success"] for r in out["task_results"].values()), out["task_results"])
        buffered = QgsProject.instance().mapLayer(out["available_output_ids"]["task_2_output"])
        self.assertEqual(buffered.featureCount(), expected)
        self.assertIn("were saved as", out["task_results"][1]["notes"][-1])


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
