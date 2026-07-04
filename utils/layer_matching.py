# -*- coding: utf-8 -*-
"""
Fuzzy layer-name matching so tools can find the layer a user meant even when
they don't type its exact QGIS layer name - different spacing, underscores,
casing, filler words like "the"/"layer", or minor typos (e.g. "stream network
of nepal" should still resolve to a layer literally named "stream_network_nepal").
"""
import difflib
import re
from typing import Dict, List, Optional, Tuple

_STOPWORDS = {"the", "a", "an", "of", "layer", "in", "for", "data", "dataset"}
_MATCH_THRESHOLD = 0.6


def _normalize(name: str) -> str:
    """Lowercase, collapse separators to spaces, and drop filler words."""
    name = name.lower()
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    tokens = [t for t in name.split() if t not in _STOPWORDS]
    return " ".join(tokens)


def find_best_layer_match(
    query: str, layer_names: List[str], threshold: float = _MATCH_THRESHOLD
) -> Optional[str]:
    """Return the entry in *layer_names* that best matches *query*.

    Tries, in order: an exact case-insensitive match, an exact match after
    normalizing spacing/underscores/casing and dropping filler words, a
    substring match, then fuzzy similarity ranking for partial names or
    typos. Returns None if nothing clears *threshold*.
    """
    if not layer_names:
        return None

    query_lower = query.lower().strip()
    for name in layer_names:
        if name.lower() == query_lower:
            return name

    normalized_query = _normalize(query)
    if not normalized_query:
        return None

    normalized_names = {name: _normalize(name) for name in layer_names}

    for name, normalized in normalized_names.items():
        if normalized == normalized_query:
            return name

    substring_candidates = [
        name
        for name, normalized in normalized_names.items()
        if normalized
        and (normalized_query in normalized or normalized in normalized_query)
    ]
    if len(substring_candidates) == 1:
        return substring_candidates[0]

    scored = sorted(
        (
            (
                difflib.SequenceMatcher(None, normalized_query, normalized).ratio(),
                name,
            )
            for name, normalized in normalized_names.items()
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    if substring_candidates:
        for _score, name in scored:
            if name in substring_candidates:
                return name

    if scored and scored[0][0] >= threshold:
        return scored[0][1]

    return None


def find_layer(
    project, layer_name: str, layer_type=None, threshold: float = _MATCH_THRESHOLD
) -> Tuple[Optional[object], Optional[str]]:
    """Find a layer in *project* by fuzzy name match.

    Args:
        project: A QgsProject instance.
        layer_name: The (possibly imprecise) layer name to look for.
        layer_type: Optional QgsMapLayer type to restrict candidates to.
        threshold: Minimum fuzzy-match ratio to accept (0-1).

    Returns:
        Tuple of (layer, matched_name), or (None, None) if nothing matched.
    """
    candidates: Dict[str, object] = {}
    for lyr in project.mapLayers().values():
        if layer_type is not None and lyr.type() != layer_type:
            continue
        candidates[lyr.name()] = lyr

    best_name = find_best_layer_match(layer_name, list(candidates.keys()), threshold)
    if best_name is None:
        return None, None
    return candidates[best_name], best_name


__all__ = ["find_layer", "find_best_layer_match"]
