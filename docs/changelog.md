# Changelog

## Unreleased

- **The assistant knows your map view:** each request includes the map's coordinate system, scale and extent (also in longitude/latitude), so "in the current extent", "this map" or "here" work in both modes without being asked where you're looking. Processing mode uses it for extent parameters.
- **Fixed:** QGIS froze while a layer was added from a URL (such as an OpenStreetMap download) or a large file: the download ran on QGIS's main thread. It now runs in the background, and only adding the finished layer touches QGIS's interface. Processing results are now also added to the project on the main thread, as QGIS requires.
- **Fixed:** in long General-mode chats, Gemini rejected requests with "Please ensure that function call turn comes immediately after a user turn or after a function response turn". The shortened history sent to the model could start in the middle of an earlier request; it now always starts at one of your messages.
- **Zoom to a place by name** ("zoom to Kathmandu, Nepal") without loading a layer, using OpenStreetMap's Nominatim search like QGIS's locator bar.
- **Filtering in General mode now uses QGIS's own tools:**
  - The assistant writes the condition as a QGIS expression, and QGIS's *Select by Expression* runs it. Any condition QGIS can express works. Text is compared case-insensitively: before, "kshetri in name" found nothing when the data said `Kshetri`, and the reply still said the selection had succeeded.
  - *Select by location* replaces *select by geometry*. It uses QGIS's *Select by Location* (or *Select within Distance*), so layers in different coordinate systems work. Before, they silently gave wrong results, and the largest feature of a shapefile was missed when it was the first one.
  - New tools: *calculate* (counts, totals and averages with QGIS aggregate expressions), *zoom to selection* and *clear selection*.
  - Replies list the first selected features. When nothing matches, they show the values of the columns used.
- **Column names don't have to be exact.** In both modes, the assistant sees each layer's columns with some of their values. It uses a column only when it clearly is the one you meant (for example `dim` for "diameter") and says so; when several could be meant, it asks. A column that doesn't exist is rejected before anything runs, with the real columns listed.
- **Show a layer's columns** now gives some values of every column, and the range of numeric and date columns.
- **General mode** reminds you to switch to Processing mode when you ask for geoprocessing.
- **Processing mode filters that feed later steps:**
  - A filter is now its own *Extract by expression* step. Before, it only selected features, which the next algorithm ignored: it processed every feature and still reported success.
  - The filter step appears in exported models.
  - When a step's result has no features, the steps that use it don't run.
  - An expression with a misspelled column used to "succeed" with an empty result; it's now rejected before running.
- **Processing mode: tool errors are no longer reported as success.** A non-processing task whose tool failed (for example, a missing column) now fails the run, and the reply explains why.
- **Clearer pre-flight errors:** an invalid option now comes back with the list of valid options.
- **No hardcoded word lists:** algorithm search now ranks words by how rare they are in the Processing registry instead of using stopword and synonym lists. Layer-name matching no longer relies on a list of filler words, and keeps non-English letters (`Zürich`). If the routing call fails, the decision falls back on how the task was split instead of a keyword list.
- **Adding layers lets QGIS detect the format** instead of using a fixed list of file extensions. Formats the list missed, such as FlatGeobuf and ASCII grids, now load. Every layer of a multi-layer file such as a GeoPackage is added, or one with `|layername=`. Processing results saved to files are loaded the same way.
- **Load a whole folder:** "load all the layers from C:\data\project" adds every layer the QGIS Browser shows in that folder, optionally including subfolders. A shapefile's side files (`.dbf`, `.shx`, `.prj`) aren't added as extra tables.

## 1.0.0

- **Fixed ([#62](https://github.com/iamtekson/GeoAgent/issues/62)):** on macOS and Linux, every tool that touches the project (listing layers, zooming, selecting, ...) reported `Thread <id> result not found in runner` in the chat, even though it had worked. Tool results now always reach the assistant, on every platform and QGIS version.
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
- **Security scan:** resolved all Bandit findings reported by the QGIS plugin repository. Errors are no longer silently swallowed; where a failure is expected (a corrupted saved setting, an unwritable error-log file, a deleted Logs tab after reloading the plugin), it's handled and logged.
- The layer list given to the model now describes geometry-less tables and non-vector, non-raster layers (mesh, point cloud, ...) correctly.
- **Qt6 compatibility:** all QGIS enum values are now written in their scoped form (for example `Qgis.MessageLevel.Critical`), as the QGIS plugin repository's Qt6 check requires. This works on QGIS 3 and QGIS 4.
- **Fixed:** QGIS information messages mirrored into the Logs tab were labelled CRITICAL; they now keep their real level.

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
