# -*- coding: utf-8 -*-
"""Fuzzy layer-name matching (utils/layer_matching.py)."""
import unittest

import _bootstrap  # noqa: F401  (makes geo_agent importable)
from geo_agent.utils.layer_matching import find_best_layer_match


class FindBestLayerMatchTest(unittest.TestCase):
    def test_extra_words_in_the_request(self):
        self.assertEqual(
            find_best_layer_match("stream network of nepal", ["stream_network_nepal", "DEM"]),
            "stream_network_nepal",
        )
        self.assertEqual(find_best_layer_match("the rivers layer", ["rivers", "DEM"]), "rivers")
        self.assertEqual(find_best_layer_match("the DEM", ["DEM", "dem_hillshade"]), "DEM")

    def test_the_name_covering_most_words_wins(self):
        self.assertEqual(
            find_best_layer_match("rivers buffer layer", ["rivers", "rivers_buffer"]), "rivers_buffer"
        )

    def test_near_miss_and_case(self):
        self.assertEqual(find_best_layer_match("River", ["rivers", "DEM"]), "rivers")
        self.assertEqual(find_best_layer_match("dem", ["rivers", "DEM"]), "DEM")

    def test_non_english_letters_are_kept(self):
        self.assertEqual(
            find_best_layer_match("zürich roads", ["Zürich_Roads", "Bern_Roads"]), "Zürich_Roads"
        )

    def test_nothing_similar(self):
        self.assertIsNone(find_best_layer_match("population", ["rivers", "DEM"]))


if __name__ == "__main__":
    unittest.main()
