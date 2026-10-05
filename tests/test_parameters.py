# -*- coding: utf-8 -*-
"""Parameter normalization, pre-flight validation and execution outputs."""
import unittest

import _bootstrap
from qgis.core import QgsProject, QgsVectorLayer

from geo_agent.tools.geoprocessing import (
    execute_processing,
    get_algorithm_parameters,
    normalize_parameters,
    validate_parameters,
)

BUFFER = "native:buffer"


class NormalizeParametersTest(unittest.TestCase):
    def setUp(self):
        self.rivers, self.dem = _bootstrap.reset_project()

    def test_valid_values_pass_through_untouched(self):
        given = {"INPUT": "rivers", "DISTANCE": 10, "END_CAP_STYLE": 0, "OUTPUT": "TEMPORARY_OUTPUT"}
        fixed, notes = normalize_parameters(BUFFER, given)
        self.assertEqual(fixed, given)
        self.assertEqual(notes, [])

    def test_near_miss_layer_name_resolves_to_layer_id(self):
        fixed, notes = normalize_parameters(BUFFER, {"INPUT": "River", "DISTANCE": 10})
        self.assertEqual(fixed["INPUT"], self.rivers.id())
        self.assertTrue(any("INPUT" in n for n in notes))

    def test_raster_parameter_only_matches_raster_layers(self):
        fixed, _ = normalize_parameters(
            "gdal:cliprasterbymasklayer", {"INPUT": "dem", "MASK": "rivers"}
        )
        self.assertEqual(fixed["INPUT"], self.dem.id())
        self.assertEqual(fixed["MASK"], "rivers")  # already valid: untouched

    def test_enum_label_becomes_index(self):
        fixed, _ = normalize_parameters(BUFFER, {"INPUT": "rivers", "END_CAP_STYLE": "Flat"})
        self.assertEqual(fixed["END_CAP_STYLE"], 1)
        fixed, _ = normalize_parameters(BUFFER, {"INPUT": "rivers", "END_CAP_STYLE": "square"})
        self.assertEqual(fixed["END_CAP_STYLE"], 2)

    def test_parameter_name_case_is_fixed(self):
        fixed, _ = normalize_parameters(BUFFER, {"input": "rivers", "distance": 5})
        self.assertEqual(fixed.get("INPUT"), "rivers")
        self.assertEqual(fixed.get("DISTANCE"), 5)
        self.assertNotIn("input", fixed)

    def test_output_label_resolves_to_layer_id(self):
        refs = {"task_1_output": self.rivers.id()}
        for label in ("task_1_output", "@task_1_output"):
            fixed, _ = normalize_parameters(BUFFER, {"INPUT": label}, refs)
            self.assertEqual(fixed["INPUT"], self.rivers.id(), label)

    def test_output_name_is_pinned_to_this_runs_output(self):
        # An older layer and this run's output share the name "rivers"
        newer = QgsVectorLayer(_bootstrap.RIVERS_PATH, "rivers", "ogr")
        QgsProject.instance().addMapLayer(newer)
        fixed, _ = normalize_parameters(BUFFER, {"INPUT": "rivers"}, {"task_1_output": newer.id()})
        self.assertEqual(fixed["INPUT"], newer.id())

    def test_unresolvable_value_is_left_for_validation(self):
        fixed, notes = normalize_parameters(BUFFER, {"INPUT": "zzqq_unknown"})
        self.assertEqual(fixed["INPUT"], "zzqq_unknown")
        self.assertEqual(notes, [])

    def test_unknown_algorithm_returns_input_unchanged(self):
        given = {"INPUT": "River"}
        self.assertEqual(normalize_parameters("native:does_not_exist", given), (given, []))


class ValidateParametersTest(unittest.TestCase):
    def setUp(self):
        _bootstrap.reset_project()

    def test_valid_parameters_have_no_problems(self):
        self.assertEqual(
            validate_parameters(BUFFER, {"INPUT": "rivers", "DISTANCE": 10, "OUTPUT": "TEMPORARY_OUTPUT"}),
            [],
        )

    def test_every_bad_value_is_named(self):
        problems = validate_parameters(
            BUFFER, {"INPUT": "zzqq_unknown", "END_CAP_STYLE": "Round", "OUTPUT": "TEMPORARY_OUTPUT"}
        )
        text = " ".join(problems)
        self.assertIn("INPUT", text)
        self.assertIn("END_CAP_STYLE", text)

    def test_unknown_algorithm(self):
        self.assertEqual(
            validate_parameters("native:does_not_exist", {}),
            ["Algorithm not found: native:does_not_exist"],
        )


class AlgorithmMetadataTest(unittest.TestCase):
    def test_parameter_flags(self):
        params = {p["name"]: p for p in get_algorithm_parameters.invoke({"algorithm": BUFFER})["parameters"]}
        self.assertTrue(params["SEPARATE_DISJOINT"]["advanced"])
        self.assertFalse(params["INPUT"]["advanced"])
        self.assertFalse(params["INPUT"]["hidden"])


class ExecuteProcessingTest(unittest.TestCase):
    def setUp(self):
        _bootstrap.reset_project()

    def test_outputs_get_ids_and_unique_names(self):
        args = {"algorithm": BUFFER, "parameters": {"INPUT": "rivers", "DISTANCE": 10, "OUTPUT": "TEMPORARY_OUTPUT"}}
        first = execute_processing.invoke(args)
        second = execute_processing.invoke(
            {"algorithm": BUFFER, "parameters": {"INPUT": "rivers", "DISTANCE": 20, "OUTPUT": "TEMPORARY_OUTPUT"}}
        )
        for result in (first, second):
            self.assertTrue(result["success"], result.get("error"))
            layer = QgsProject.instance().mapLayer(result["outputs"]["OUTPUT"])
            self.assertIsNotNone(layer)
            self.assertEqual(result["output_layer_ids"], [layer.id()])
            self.assertEqual(result["output_layers"], [layer.name()])
        self.assertNotEqual(first["output_layers"], second["output_layers"])


if __name__ == "__main__":
    unittest.main()
