# Installation

## Requirements

- QGIS 3 or QGIS 4.
- Python 3.10 or newer inside QGIS (GeoAgent's dependencies need it). QGIS shows its Python version under **Help ▸ About**.
- An LLM: either [Ollama](https://ollama.com) running on your computer, or an API key for OpenAI, Google Gemini or Anthropic. See [LLM providers](llm-providers.md).

## Install the plugin

**From the QGIS plugin repository (recommended):**

1. In QGIS, open **Plugins ▸ Manage and Install Plugins**.
2. Search for **GeoAgent** and click **Install Plugin**.

**From a zip file:** download the latest release from [GitHub](https://github.com/iamtekson/GeoAgent/releases), then use **Plugins ▸ Manage and Install Plugins ▸ Install from ZIP**.

Open the panel with the GeoAgent toolbar button or **Plugins ▸ GeoAgent**. It docks at the bottom of the QGIS window.

## Install the Python dependencies

GeoAgent uses LangGraph and the LangChain provider libraries, which QGIS doesn't include. When you open the panel, a message reminds you if any are missing.

1. Open the **Settings** tab of the GeoAgent panel.
2. Click **Check / Install Dependencies**. GeoAgent lists the missing packages and asks before installing anything.
3. Confirm. The packages are installed with `pip` into QGIS's own Python, with a progress bar under the button.

If a package is still reported missing afterwards, restart QGIS.

The packages are: `langgraph`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-google-genai`, `langchain-ollama`, `langchain-anthropic`, `requests` and `markdown`.

Next: [connect an LLM provider](llm-providers.md).
