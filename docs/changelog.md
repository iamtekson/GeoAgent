# Changelog

## Unreleased

- **Model export:** each processing result in the chat ends with *Open in Model Designer* and *Save as .model3* links. The run becomes a QGIS processing model, with your layers as inputs and its steps wired together.
- **New setting, *After a processing run*:** open each run in the Model Designer, or ask to save it, automatically.
- **Token usage:** the log ends every request with the tokens it used, plus a session total.
- **More reliable processing:**
  - QGIS checks the parameters before running;
  - near-miss layer names, option labels and parameter-name case are corrected automatically;
  - a task that failed only on a parameter retries with the same algorithm;
  - results are passed between steps by layer id;
  - result layers get unique names.
- **Fewer LLM calls:** processing requests skip one call per task, since routing is decided while the request is split into tasks.
- **Web links** in replies open in the browser.
- **Fixed:** changing the model name, API key, Ollama settings or maximum tokens now applies to your next message; before, it only took effect after the provider, mode or temperature changed. The first message no longer sets up the model twice.
- **For developers:** a headless test suite and this documentation site.

## 0.5.0

- Added support for Anthropic models.
- Saved settings, and a dependency check and installation function.

## 0.4.0

- Added support for QGIS 4.x.
- Fixed a UI bug.

## 0.3.2

- Fixed an import issue.
- Added support for multi-step geospatial workflows.
- Added a confirmation dialog when removing a layer.
- Fixed a state mutation issue.
- Extended the geoprocessing agent to multiple processing providers (GRASS, GDAL, PDAL, ...).

## 0.2.0

- Improved UI help messages and user notifications.
- Enhanced error handling for project operations.
- Fixed a QGIS crash caused by illegal threading during project load, save, create and delete operations.

## 0.1.1

- Fixed dependency installation issues on some systems.

## 0.1

- Initial release: Ollama, ChatGPT and Gemini support; dependency installation; layer reading and writing; listing layers and columns; attribute and geometry queries; project create, load and save; experimental processing agent.
