# -*- coding: utf-8 -*-
"""Parameter normalization, pre-flight validation and execution outputs."""
import unittest

import _bootstrap
from qgis.core import QgsProject, QgsVectorLayer

from geo_agent.tools.geoprocessing import (
    execute_processing,
    find_processing_algorithm,
    get_algorithm_parameters,
    normalize_parameters,
    validate_parameters,
)

BUFFER = "native:buffer"
EXTRACT_BY_ATTRIBUTE = "native:extractbyattribute"
EXTRACT_BY_EXPRESSION = "native:extractbyexpression"


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


class ColumnParametersTest(unittest.TestCase):
    """Column names and options the pre-flight check rejects (rivers: fid, cat, value, label)."""

    def setUp(self):
        self.rivers, _ = _bootstrap.reset_project()

    def test_unknown_field_is_rejected_with_the_real_columns(self):
        given = {"INPUT": "rivers", "FIELD": "zzqq", "OPERATOR": 9, "OUTPUT": "TEMPORARY_OUTPUT"}
        problems = validate_parameters(EXTRACT_BY_ATTRIBUTE, given)
        self.assertTrue(any("'zzqq'" in p and "Columns of 'rivers'" in p for p in problems), problems)
        self.assertTrue(any("value (Integer64: 4 to 272)" in p for p in problems), problems)

    def test_expression_with_unknown_column_is_rejected(self):
        # QGIS itself would run it and match nothing
        problems = validate_parameters(
            EXTRACT_BY_EXPRESSION, {"INPUT": "rivers", "EXPRESSION": '"zzqq" > 1', "OUTPUT": "TEMPORARY_OUTPUT"}
        )
        self.assertTrue(any("zzqq" in p and "Columns of 'rivers'" in p for p in problems), problems)

    def test_valid_columns_pass(self):
        for algorithm, params in [
            (EXTRACT_BY_EXPRESSION, {"INPUT": "rivers", "EXPRESSION": '"value" > 1'}),
            (EXTRACT_BY_ATTRIBUTE, {"INPUT": "rivers", "FIELD": "value", "OPERATOR": 9}),
        ]:
            self.assertEqual(validate_parameters(algorithm, dict(params, OUTPUT="TEMPORARY_OUTPUT")), [])

    def test_invalid_option_lists_the_options(self):
        problems = validate_parameters(
            "native:selectbylocation", {"INPUT": "rivers", "PREDICATE": ["inside"], "INTERSECT": "rivers", "METHOD": 0}
        )
        self.assertTrue(any("PREDICATE" in p and "are within" in p for p in problems), problems)


class AlgorithmSearchTest(unittest.TestCase):
    """Shortlisting by word rarity in the registry: no stopword or synonym lists."""

    @staticmethod
    def ranked(query, keywords=()):
        result = find_processing_algorithm.invoke({"query": query, "keywords": list(keywords)})
        return [m["id"] for m in result["matches"]]

    def test_task_with_llm_keywords(self):
        cases = [
            ("Create Voronoi polygons from the points", ["voronoi", "thiessen", "polygons"], "native:voronoipolygons"),
            ("Extract rivers whose stream order is above 3",
             ["extract", "expression", "filter", "attribute", "extract by expression"], "native:extractbyexpression"),
            ("Count points in each polygon", ["count", "points", "polygon", "count points in polygon"],
             "native:countpointsinpolygon"),
            ("Create 5 km buffer around the demo layer", ["buffer", "distance", "grow"], "native:buffer"),
        ]
        for query, keywords, expected in cases:
            self.assertIn(expected, self.ranked(query, keywords)[:3], query)

    def test_common_words_and_numbers_barely_count(self):
        # "the", "by" and "500" must not push other algorithms above buffer
        self.assertIn("native:buffer", self.ranked("buffer the rivers by 500 m")[:5])


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
