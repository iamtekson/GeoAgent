# -*- coding: utf-8 -*-
"""
Geoprocessing tools for QGIS operations.
Provides tools for geometric operations (buffer, clip, dissolve, etc.) and spatial filtering/selection.
"""
import uuid
from typing import Optional, List
import processing
from langchain_core.tools import tool

from .io import get_qgis_interface


@tool
def execute_processing(algorithm: str, parameters: dict, **kwargs) -> dict:
    """Execute a processing algorithm"""
    try:
        import processing

        result = processing.run(algorithm, parameters)
        return {
            "algorithm": algorithm,
            "result": {
                k: str(v) for k, v in result.items()
            },  # Convert values to strings for JSON
        }
    except Exception as e:
        raise Exception(f"Processing error: {str(e)}")
