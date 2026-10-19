# -*- coding: utf-8 -*-
"""
Short descriptions of a vector layer's columns, for prompts and tool replies.

Each column is shown with its QGIS type and a few of its values (or its range
for numbers and dates), so the LLM can match the user's wording to real
column names and values itself.
"""
from typing import Iterable, Optional


def _is_null(value) -> bool:
    return value is None or (hasattr(value, "isNull") and value.isNull())


def _short(value, width: int = 30) -> str:
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def describe_column(layer, index: int, samples: int = 3) -> str:
    """'name (type: values)' for column *index*; *samples* = values shown."""
    field = layer.fields().at(index)
    text = f"{field.name()} ({field.typeName()}"
    if samples:
        if field.isNumeric() or field.isDateOrTime():
            low, high = layer.minimumValue(index), layer.maximumValue(index)
            if not _is_null(low) and not _is_null(high):
                text += f": {_short(low)} to {_short(high)}"
        else:
            values = sorted(
                str(v) for v in layer.uniqueValues(index, samples) if not _is_null(v) and v != ""
            )
            if values:
                text += ": " + ", ".join(f"'{_short(v)}'" for v in values)
    return text + ")"


def describe_columns(
    layer, columns: Optional[Iterable[str]] = None, samples: int = 3, max_columns: int = 30
) -> str:
    """Comma-separated descriptions of a layer's columns (or only *columns*)."""
    fields = layer.fields()
    if columns is None:
        indexes = list(range(fields.count()))
    else:
        indexes = [i for i in dict.fromkeys(fields.lookupField(c) for c in columns) if i >= 0]
    parts = [describe_column(layer, i, samples) for i in indexes[:max_columns]]
    if len(indexes) > max_columns:
        parts.append(f"... {len(indexes) - max_columns} more")
    return ", ".join(parts) or "no columns"


__all__ = ["describe_column", "describe_columns"]
