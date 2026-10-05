# GeoAgent

GeoAgent is a QGIS plugin for geospatial analysis in plain language. Ask it to load data, explore your layers, or run a chain of processing steps: it picks the QGIS algorithms, fills in their parameters, runs them, and adds the results to your map.

![The GeoAgent panel in QGIS, answering a question and running a three-step processing request](imgs/screenshots/chat.png)

**New to GeoAgent?** [Install the plugin](installation.md), [connect an LLM provider](llm-providers.md), then follow the [quickstart](quickstart.md).

## What it can do

- **Explore and manage data** — add layers from files or URLs, list layers and columns, zoom, select features by attribute or geometry, and create, load or save projects ([General mode](user-guide/general-mode.md)).
- **Run geoprocessing in plain language** — any algorithm in the QGIS Processing Toolbox (native, GDAL, GRASS, ...), alone or as a multi-step chain where each result feeds the next ([Processing mode](user-guide/processing-mode.md)).
- **Turn a run into a QGIS model** — open what ran in the Model Designer, or save it as a `.model3` file to re-run on other data ([Model export](user-guide/model-export.md)).
- **Use the LLM you prefer** — Ollama (local and free), OpenAI, Google Gemini, or Anthropic Claude ([LLM providers](llm-providers.md)).
- **See what happened** — every step, the parameters used, and the tokens each request cost appear in the Logs tab ([Logs and token usage](user-guide/logs-and-usage.md)).

## License

GeoAgent's code is released under the MIT License. When it runs inside QGIS (GPL v2+, with PyQt under GPL v3), the combined work is governed by those licenses' terms, as with any QGIS plugin.

## Citation

```{include} ../README.md
:start-after: <!-- citation-start -->
:end-before: <!-- citation-end -->
```

```{toctree}
:caption: Getting started
:maxdepth: 2
:hidden:

installation
llm-providers
quickstart
```

```{toctree}
:caption: User guide
:maxdepth: 2
:hidden:

user-guide/general-mode
user-guide/processing-mode
user-guide/model-export
user-guide/settings
user-guide/logs-and-usage
troubleshooting
```

```{toctree}
:caption: Developer guide
:maxdepth: 2
:hidden:

developer-guide/architecture
developer-guide/testing
developer-guide/contributing
```

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

changelog
```
