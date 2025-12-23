# GeoAgent

GeoAgent is a QGIS plugin that integrates large language models (LLMs) to provide geospatial assistance through natural language conversations. It supports multiple LLM providers, including local Ollama models, OpenAI's ChatGPT, and Google's Gemini.

## Features

- Multi-provider LLM support with a unified interface
- General chat mode with geospatial context awareness
- Configurable system prompts for different interaction modes
- Easy integration with QGIS for geospatial data handling

## Installation

1. Clone or download the GeoAgent plugin into your QGIS plugins directory.
2. Install required dependencies (if any).
3. Restart QGIS and enable the GeoAgent plugin from the Plugin Manager.

## TODO

- [ ] When clearing chat history, also reset the thread in the backend.
- [ ] Add more LLM providers and models.
- [ ] Implement advanced geospatial query handling.
- [ ] Improve UI/UX for better user interaction. Make ollama parameters only visible when ollama is selected as provider.
- [ ] Add error handling and user notifications for LLM interactions.
- [ ] Documentation and usage examples.
