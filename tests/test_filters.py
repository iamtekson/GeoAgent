# -*- coding: utf-8 -*-
"""Selection tools (tools/filters.py): QGIS's own selection algorithms on
small memory layers."""
import unittest

import _bootstrap

from geo_agent.tools.filters import (
    calculate,
    clear_selection,
    select_by_expression,
    select_by_location,
    zoom_to_selection,
)
from geo_agent.tools.io import get_layer_columns
from geo_agent.utils import canvas_refresh
from geo_agent.utils.canvas_refresh import MainThreadRunner, set_main_runner, set_qgis_interface

SITES = "Point?crs=EPSG:3857&field=Name:string&field=dim:double&field=kind:string"
# Zone A holds sites 1 and 2, zone B site 3; in another CRS than the sites
ZONE_A = "POLYGON((-0.001 -0.001, 0.0135 -0.001, 0.0135 0.001, -0.001 0.001, -0.001 -0.001))"
ZONE_B = "POLYGON((0.04 0.04, 0.05 0.04, 0.05 0.05, 0.04 0.05, 0.04 0.04))"


class FakeCanvas:
    def __init__(self):
        self.zoomed = []

    def refresh(self):
        pass

    def zoomToSelected(self, layer):
        self.zoomed.append(layer.name())


class FakeIface:
    def __init__(self):
        self.canvas = FakeCanvas()

    def mapCanvas(self):
        return self.canvas


def call(tool, **args):
    return tool.invoke(args)


class FilterToolsBase(unittest.TestCase):
    def setUp(self):
        self.previous = (canvas_refresh._global_main_runner, canvas_refresh.get_qgis_interface())
        set_main_runner(MainThreadRunner())
        self.iface = FakeIface()
        set_qgis_interface(self.iface)
        _bootstrap.reset_project()
        self.sites = _bootstrap.memory_layer(SITES, "sites", [
            (["KSHETRI Tole", 12.5, "school"], "POINT(0 0)"),
            (["Kshetri Bazaar", 8.0, "hospital"], "POINT(1000 0)"),
            (["O'Brien Park", 30.0, "school"], "POINT(5000 5000)"),
            (["charging_station", None, ""], "POINT(20000 0)"),
        ])
        self.zones = _bootstrap.memory_layer(
            "Polygon?crs=EPSG:4326&field=name:string", "zones",
            [(["A"], ZONE_A), (["B"], ZONE_B)],
        )

    def tearDown(self):
        set_main_runner(self.previous[0])
        set_qgis_interface(self.previous[1])

    def selected_names(self):
        return sorted(f["Name"] for f in self.sites.getSelectedFeatures())


class SelectByExpressionTest(FilterToolsBase):
    def test_ilike_ignores_case_and_lists_the_selection(self):
        reply = call(select_by_expression, layer_name="sites", expression="\"Name\" ILIKE '%kshetri%'")
        self.assertEqual(self.selected_names(), ["KSHETRI Tole", "Kshetri Bazaar"])
        self.assertIn("2 of 4 features", reply)
        self.assertIn("KSHETRI Tole", reply)

    def test_unknown_column_lists_the_real_columns(self):
        self.sites.selectAll()
        reply = call(select_by_expression, layer_name="sites", expression='"diameter" > 10')
        self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertIn("diameter", reply)
        self.assertIn("dim (double: 8.0 to 30.0)", reply)
        self.assertEqual(self.sites.selectedFeatureCount(), 4)  # untouched

    def test_methods_combine_selections(self):
        call(select_by_expression, layer_name="sites", expression="\"kind\" ILIKE 'SCHOOL'")
        call(select_by_expression, layer_name="sites", expression='"dim" > 20', method="within")
        self.assertEqual(self.selected_names(), ["O'Brien Park"])
        call(select_by_expression, layer_name="sites", expression="\"kind\" = 'hospital'", method="adding to current selection")
        self.assertEqual(self.selected_names(), ["Kshetri Bazaar", "O'Brien Park"])
        call(select_by_expression, layer_name="sites", expression="\"Name\" ILIKE 'o''%'", method="removing from current selection")
        self.assertEqual(self.selected_names(), ["Kshetri Bazaar"])

    def test_unknown_method_lists_the_options(self):
        reply = call(select_by_expression, layer_name="sites", expression='"dim" > 1', method="intersect")
        self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertIn("selecting within current selection", reply)

    def test_no_match_lists_the_values_of_the_columns_used(self):
        reply = call(select_by_expression, layer_name="sites", expression="\"kind\" = 'School'")
        self.assertIn("0 of 4 features", reply)
        self.assertIn("'school'", reply)

    def test_qgis_functions_work(self):
        call(select_by_expression, layer_name="zones", expression="$area = maximum($area)")
        self.assertEqual([f["name"] for f in self.zones.getSelectedFeatures()], ["B"])

    def test_invalid_expression(self):
        reply = call(select_by_expression, layer_name="sites", expression="dim >>> 1")
        self.assertTrue(reply.startswith("**Error:**"), reply)


class SelectByLocationTest(FilterToolsBase):
    def test_layers_in_different_crs(self):
        reply = call(select_by_location, layer_name="sites", predicate="are within", reference_layer_name="zones")
        self.assertEqual(self.selected_names(), ["KSHETRI Tole", "Kshetri Bazaar", "O'Brien Park"])
        self.assertIn("3 of 4 features", reply)

    def test_predicate_wording_maps_to_the_qgis_option(self):
        call(select_by_location, layer_name="sites", predicate="within", reference_layer_name="zones")
        self.assertEqual(len(self.selected_names()), 3)

    def test_only_the_selected_reference_features(self):
        call(select_by_expression, layer_name="zones", expression="\"name\" = 'A'")
        call(select_by_location, layer_name="sites", predicate="are within", reference_layer_name="zones",
             reference_selected_only=True)
        self.assertEqual(self.selected_names(), ["KSHETRI Tole", "Kshetri Bazaar"])

    def test_unknown_predicate_lists_the_options(self):
        reply = call(select_by_location, layer_name="sites", predicate="inside", reference_layer_name="zones")
        self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertIn("are within", reply)

    def test_within_distance_in_layer_units(self):
        call(select_by_expression, layer_name="zones", expression="\"name\" = 'B'")
        reply = call(select_by_location, layer_name="sites", predicate="intersect", reference_layer_name="zones",
                     reference_selected_only=True, distance=500)
        self.assertEqual(self.selected_names(), ["O'Brien Park"])
        self.assertIn("meters", reply)


class CalculateTest(FilterToolsBase):
    def test_count_with_a_filter(self):
        reply = call(calculate, layer_name="sites", expression="count(1, filter:=\"kind\" ILIKE 'school')")
        self.assertIn("= 2", reply)

    def test_sum_of_the_selection(self):
        self.sites.selectByIds([1, 3])  # 12.5 + 30.0
        reply = call(calculate, layer_name="sites", expression='sum("dim", filter:=is_selected())')
        self.assertIn("= 42.5", reply)

    def test_lengths_are_measured_in_project_units(self):
        _bootstrap.memory_layer("LineString?crs=EPSG:4326", "lines", [([], "LINESTRING(0 0, 1 0)")])
        reply = call(calculate, layer_name="lines", expression="sum($length)")
        value = float(reply.split("= ")[1].split("\n")[0])
        self.assertGreater(value, 100000)  # one degree in meters, not 1
        self.assertIn("lengths in meters", reply)

    def test_unknown_column_lists_the_real_columns(self):
        reply = call(calculate, layer_name="sites", expression='sum("diameter")')
        self.assertTrue(reply.startswith("**Error:**"), reply)
        self.assertIn("dim (double", reply)


class SelectionHelpersTest(FilterToolsBase):
    def test_zoom_and_clear(self):
        self.assertIn("No features are selected", call(zoom_to_selection, layer_name="sites"))
        self.sites.selectAll()
        call(zoom_to_selection, layer_name="sites")
        self.assertEqual(self.iface.canvas.zoomed, ["sites"])
        call(clear_selection)
        self.assertEqual(self.sites.selectedFeatureCount(), 0)

    def test_columns_show_values_and_ranges(self):
        reply = call(get_layer_columns, layer_name="sites")
        self.assertIn("kind (string: 'hospital', 'school')", reply)
        self.assertIn("dim (double: 8.0 to 30.0)", reply)


if __name__ == "__main__":
    unittest.main()
