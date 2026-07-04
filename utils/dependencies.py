# -*- coding: utf-8 -*-
"""
Dependency checking and background installation for GeoAgent's required
Python packages.

Installation is triggered explicitly by the "Check / Install Dependencies"
button in the Settings tab, rather than automatically the first time the
plugin panel is opened.
"""
import importlib
import os
import re
import subprocess
import sys
from typing import List

from qgis.PyQt.QtCore import QThread, pyqtSignal

# Maps each pip package name declared in pyproject.toml to its import name.
PKG_TO_IMPORT = {
    "langgraph": "langgraph",
    "langchain-core": "langchain_core",
    "langchain-community": "langchain_community",
    "langchain-openai": "langchain_openai",
    "langchain-google-genai": "langchain_google_genai",
    "langchain-ollama": "langchain_ollama",
    "langchain-anthropic": "langchain_anthropic",
    "requests": "requests",
    "markdown": "markdown",
}

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _declared_packages() -> List[str]:
    """Return normalized pip package names declared in pyproject.toml."""
    pyproject_path = os.path.join(PLUGIN_DIR, "pyproject.toml")
    deps = []
    try:
        with open(pyproject_path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            import tomllib  # Python 3.11+

            data = tomllib.loads(content)
            deps = data.get("project", {}).get("dependencies", []) or []
        except Exception:
            match = re.search(r"dependencies\s*=\s*\[(.*)\]", content, re.DOTALL)
            if match:
                for line in match.group(1).splitlines():
                    line = line.strip().strip(",")
                    if not line:
                        continue
                    if (line.startswith('"') and line.endswith('"')) or (
                        line.startswith("'") and line.endswith("'")
                    ):
                        deps.append(line[1:-1])
    except Exception:
        deps = []

    pkgs = []
    for dep in deps:
        name = dep.split(";")[0].split(" ")[0]
        name = name.split(">=")[0].split("==")[0]
        pkgs.append(name)
    return pkgs


def get_missing_packages() -> List[str]:
    """Return pip package names declared in pyproject.toml but not importable."""
    missing = []
    for pkg in _declared_packages():
        import_name = PKG_TO_IMPORT.get(pkg)
        if not import_name:
            continue
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append(pkg)
    return missing


def _resolve_python_executable() -> str:
    """Resolve a real python(.exe) next to the QGIS-embedded interpreter.

    QGIS's ``sys.executable`` is often the QGIS application binary
    (qgis-bin.exe / pythonw.exe) rather than a usable Python interpreter for
    subprocess calls.
    """
    py_exec = sys.executable
    lower = py_exec.lower()
    if lower.endswith("qgis-bin.exe") or lower.endswith("qgis-ltr-bin.exe"):
        py_exec = os.path.join(os.path.dirname(py_exec), "python.exe")
    elif lower.endswith("pythonw.exe"):
        py_exec = os.path.join(os.path.dirname(py_exec), "python.exe")
    return py_exec


class DependencyInstallWorker(QThread):
    """Installs the given pip packages in the background via pip."""

    progress = pyqtSignal(int, str)
    finished_with_result = pyqtSignal(bool, str)

    def __init__(self, packages: List[str], parent=None):
        super().__init__(parent)
        self.packages = packages

    def run(self):
        try:
            py_exec = _resolve_python_executable()
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            total = len(self.packages)
            for idx, pkg in enumerate(self.packages, start=1):
                self.progress.emit(
                    int((idx - 1) / total * 100), f"Installing {pkg}..."
                )
                result = subprocess.run(
                    [py_exec, "-m", "pip", "install", "--upgrade", pkg],
                    capture_output=True,
                    text=True,
                    **kwargs,
                )
                if result.returncode != 0:
                    error = (result.stderr or result.stdout or "Unknown error").strip()
                    self.finished_with_result.emit(
                        False, f"Failed to install {pkg}:\n{error[-800:]}"
                    )
                    return
                self.progress.emit(int(idx / total * 100), f"Installed {pkg}")

            still_missing = get_missing_packages()
            if still_missing:
                self.finished_with_result.emit(
                    False,
                    "Installed, but still not importable: "
                    + ", ".join(still_missing)
                    + ". Try restarting QGIS.",
                )
            else:
                self.finished_with_result.emit(
                    True, "All dependencies installed. Restart QGIS to be safe."
                )
        except Exception as e:
            self.finished_with_result.emit(False, f"Unexpected error: {e}")


__all__ = ["get_missing_packages", "DependencyInstallWorker"]
