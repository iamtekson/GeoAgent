---
title: "`GeoAgent`: A QGIS Plugin for Natural Language-Driven Geospatial Analysis"
tags:
  - QGIS
  - large language model
  - natural language processing
  - geospatial analysis
  - agent-based systems
  - open-source GIS

authors:
  - name: Tek Kshetri
    orcid: 0000-0001-9275-5619
    affiliation: 1, 2
  - name: Rabin Ojha
    orcid: 0000-0002-7349-828X
    affiliation: 1, 3
affiliations:
  - name: OSGeo Nepal
    index: 1
  - name: The Solution Stack, Canada
    index: 2
  - name: Landesamt für Geoinformation und Landentwicklung, Germany
    index: 3
date: 17 February 2026
bibliography: paper.bib
---

# Summary

GeoAgent is an open-source QGIS plugin that enables users to perform geospatial analysis through natural language commands, powered by large language models (LLMs). The plugin supports multiple LLM backends (Ollama for local inference, OpenAI, and Google Gemini for cloud services) and operates in two modes: **General mode** for data exploration and querying attributes, and **Processing mode** for automated geoprocessing workflows. By abstracting traditional GIS complexity through conversational interaction, GeoAgent lowers barriers to advanced geospatial analysis for non-specialist users while maintaining transparency through detailed chat transcripts and automatic result visualization in QGIS.

# Statement of Need

Geospatial analysis is critical across various scientific domains, including urban planning, environmental monitoring and public health [@Chen2020; @Mansourihanis2023; @Ramirez2026]. Yet traditional GIS workflows present substantial barriers: steep learning curves, complex parameter configuration, and error-prone repetitive steps [@Marra2017]. Users often spend more time navigating interfaces than designing analyses.

Recent LLM advances demonstrate strong capabilities in understanding natural language intent [@Akinboyewa2025; @Zhang2024]. However, most LLM–GIS integrations either lack true task automation (offering only code suggestions) [@IntelliGeo2024; @QChatGPT2024] or operate as standalone systems disconnected from established GIS platforms [@Akinboyewa2025; @Zhang2024]. A few systems like GIS Copilot embed LLMs within QGIS but focus primarily on code generation rather than direct execution of configured processing workflows.

GeoAgent addresses these gaps by embedding a lightweight agent-based architecture directly within QGIS. Users can describe spatial tasks in plain language, and GeoAgent translates them into executable QGIS operations automatically. By supporting both local inference (Ollama, for privacy) and cloud services, GeoAgent accommodates diverse institutional constraints. The two-mode design reflects real-world GIS usage: General mode for exploration, Processing mode for automated workflows.

# Implementation

## Plugin Architecture

GeoAgent is a modular Python QGIS plugin with two distinct agent types (Figure \ref{fig:architecture}):

**General Agent** handles layer management, data exploration, and attribute queries. It retrieves layer metadata, computes statistics, and executes selections using QGIS's native tools.

**Processing Agent** interprets geoprocessing requests by identifying applicable QGIS algorithms, constructing parameter dictionaries, and executing them via a generic wrapper function:

```python
@tool
def execute_processing(algorithm: str, parameters: dict, **kwargs) -> dict:
    """Execute a processing algorithm and load results into QGIS map."""
```

This function wraps `processing.run()`, manages temporary outputs, detects output layers, and loads them into the project automatically. Raster and vector outputs are handled transparently using `QgsRasterLayer` and `QgsVectorLayer`.

**Tool System:** A structured registry maps natural-language intents to QGIS operations, organized by function: layer I/O, attribute operations, spatial operations (buffer, clip, intersect), and utilities. Each tool includes metadata (description, parameters, types, constraints) that guides LLM parameter assignment.

**Multi-Backend LLM Integration:** GeoAgent abstracts LLM communication behind a provider interface supporting Ollama (local, privacy-preserving), OpenAI, and Google Gemini. Users configure backends and models via a settings dialog; the plugin handles prompt formatting, temperature setting and token limits.

![GeoAgent system architecture. Natural-language requests are routed to either a General Agent for exploratory operations or a Processing Agent for workflow execution. The Processing branch maps requests to QGIS algorithms through a structured tool registry, constructs execution parameters, and returns results to the QGIS project and chat context. An LLM provider abstraction supports interchangeable backends (Ollama, OpenAI, Gemini), enabling both local and cloud deployment. \label{fig:architecture}](./figs/geoagent_overall_architecture.png)

## Operating Modes

**General Mode** supports exploratory queries (Figure \ref{fig:general-mode}):

- "List all layers and their feature counts"
- "Show me the attribute schema for the cities layer"
- "Select all cities with population > 50,000"

![General mode architecture. User prompts are handled through an iterative LLM reasoning loop that invokes QGIS/tool functions as needed for layer inspection, attribute queries, and lightweight operations, then returns summarized responses to the chat interface without entering the multi-step geoprocessing pipeline. \label{fig:general-mode}](./figs/general_agent_architecture.png){width=40%}

**Processing Mode** enables multi-step geoprocessing (Figure \ref{fig:processing-mode}):

- "Buffer all hospitals by 2 km and count intersecting residential zones"
- "Clip the elevation DEM to the state boundary and compute average elevation by city"
- "Perform zonal statistics of land cover within each watershed"

The Processing Agent breaks down requests, identifies algorithms (e.g., `native:clip`, `qgis:zonalstatistics`), assembles parameter dictionaries, and executes them. Results are automatically added as map layers, and the chat summarizes what was executed.

![Processing mode architecture. The Processing Agent decomposes a natural-language request and routes each sub-task through either direct LLM reasoning (with tool/QGIS calls) or a geoprocessing pipeline. In the geoprocessing branch, the agent identifies candidate QGIS algorithms, assembles and validates input parameters, and executes the selected operation. Outcomes are evaluated at a success/error checkpoint: failures trigger error analysis and retry, while successful results update the task state and proceed to remaining sub-tasks until completion. \label{fig:processing-mode}](./figs/processing_agent_architecture.png)

## Integration with QGIS Processing Ecosystem

GeoAgent leverages QGIS's Processing Framework (~250+ native algorithms plus GDAL, GRASS GIS, SAGA, and other providers). By mapping user requests to existing algorithms, GeoAgent ensures compatibility with the broader ecosystem and benefits from ongoing improvements.

## User Interface

The plugin provides an intuitive interface for natural language-driven geospatial analysis (Figure \ref{fig:ui}). The **Chat Panel** offers an input area where users enter natural language prompts and get the agent's response. The **Settings Panel** enables users to select and configure LLM backends, input API keys, adjust model parameters, and set defaults for different analysis types. Finally, the **Logs** section provides debug messages and troubleshooting information, helping users understand the plugin's decision-making process and diagnose issues.

![GeoAgent User Interface. The plugin integrates seamlessly with QGIS, providing a conversational interface for geospatial analysis. Users enter natural language requests in the input panel, view agent responses and results in the chat history, configure LLM backends and parameters in the settings, and monitor execution logs for transparency.](./figs/geoagent_ui.png){#fig:ui}

# Related Work

**GIS Copilot** [@Akinboyewa2025] is most closely related—it embeds LLMs in QGIS with a multi-module framework. However, it focuses on code generation and supervision; GeoAgent prioritizes a lighter-weight architecture where requests map directly to QGIS processing operations. Also, GIS Copilot currently supports only OpenAI models, while GeoAgent offers multi-backend flexibility.

**GeoJSON Agents** [@Luo2025] proposes a multi-agent architecture but operates on standalone data, limiting adoption for QGIS users. **IntelliGeo** [@IntelliGeo2024] and **QChatGPT** [@QChatGPT2024] are QGIS plugins but more limited in scope. Standalone frameworks like GeoGPT [@Zhang2024] integrate LLMs with geospatial libraries but require external environments.

GeoAgent uniquely combines: (1) direct QGIS embedding with processing execution, (2) a structured tool registry mapping language to operations, and (3) multi-backend LLM support balancing privacy, cost, and performance.

# Example Use Cases

**Urban Heat Island Assessment:** A public health researcher loads a city boundary and temperature raster. In General mode, he/she queries: "Show extent and resolution of the temperature raster" to inspect metadata. In processing mode, the researcher requests: "Buffer public parks by 100 meters and calculate average temperature within buffers." GeoAgent identifies `native:buffer` and `qgis:zonalstatistics`, executes them with inferred parameters, and adds result layers showing mean temperature per park buffer.

**Flood Risk Assessment:** A disaster management officer has a digital elevation model (DEM) and river network shapefile for a region. In processing mode, the officer requests: "Create a 500-meter buffer around all rivers." GeoAgent identifies `native:buffer` and creates buffer zones along the rivers. The officer then requests: "Calculate slope from the elevation model." GeoAgent executes `native:slope` to generate a slope raster showing terrain steepness across the region. Finally, the officer requests: "Clip the slope raster using the river buffer zones." GeoAgent applies `gdal:cliprasterbymasklayer` to extract slope values within 500 meters of rivers. The resulting layer reveals steep versus gentle slopes near waterways, enabling the officer to identify areas where rapid runoff (steep slopes) versus water accumulation (gentle slopes) may pose different flood risks.

# Discussion and Future Work

GeoAgent demonstrates that LLM-driven interfaces can simplify geospatial analysis workflows. The multi-backend architecture allows organizations to choose between local (Ollama) and cloud-based (OpenAI, Gemini) LLM services based on their privacy and performance needs.

Future work includes expanding the tool registry to support more QGIS algorithms, implementing parameter confirmation dialogs for user validation, and conducting systematic benchmarks across different LLM backends and QGIS versions to evaluate performance and usability.

# Acknowledgements

The authors thank the QGIS community and developers of LangChain, Ollama, and the open-source LLM ecosystem.

# References
