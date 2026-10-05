# Troubleshooting

| Problem | What to do |
| --- | --- |
| "Some required packages aren't installed yet" | Settings tab ▸ **Check / Install Dependencies**, then restart QGIS if the message persists. See [Installation](installation.md#install-the-python-dependencies). |
| Ollama connection failed | Check that `ollama serve` is running and that the base URL is `http://localhost:11434` (or wherever your server runs). |
| "Model not installed" (Ollama) | Accept GeoAgent's offer to download it, or run `ollama pull <model>` yourself. |
| The model doesn't respond | Check the API key and model name in the Settings tab, and your internet connection. |
| "Layer not found" | Ask "Which layers are loaded?" in General mode and use that name, or give the file path. |
| Wrong algorithm or parameters | Rephrase with the operation and units stated explicitly. The [Logs tab](user-guide/logs-and-usage.md) shows what was chosen and why. |
| A task failed after retries | The reply and the Logs tab explain why. Often a layer has the wrong geometry type or CRS for the operation, or a field name doesn't exist. |
| Processing results are poor with a small local model | Use a larger model (8B parameters or more) or a cloud provider. See [LLM providers](llm-providers.md). |
| No model links under a result | Only runs with at least one successful geoprocessing step can be exported. See [Model export](user-guide/model-export.md). |
| The plugin misbehaves after an update | Restart QGIS, or disable and re-enable the plugin. |

When reporting a bug, [open an issue](https://github.com/iamtekson/GeoAgent/issues). Include your QGIS version, the GeoAgent version, your provider and model, and the relevant lines from the Logs tab.
