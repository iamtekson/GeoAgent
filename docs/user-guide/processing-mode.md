# Processing mode

Processing mode runs QGIS processing algorithms from a plain-language request: one operation, or a whole chain. Choose it with the **Processing** button in the Chat tab.

```text
Create a 500 m buffer around the cities layer
Clip the roads layer by the study area
Dissolve the parcels by the district field
Add C:\data\demo.shp, create a 5 km buffer for each shape, and clip raster.tif with the buffered layer
```

## What happens when you send a request

1. **Split into tasks.** The request becomes an ordered list of tasks, each marked as either a geoprocessing step or something else (loading a layer, answering a question).
2. **Find an algorithm.** For each geoprocessing task, GeoAgent searches every algorithm in the Processing Toolbox (native, GDAL, GRASS, and any other installed provider) and picks the best match.
3. **Fill in the parameters.** The model maps your wording onto the algorithm's parameters, converting units ("5 km" becomes 5000) and following the algorithm's own help text, for example when a threshold is counted in cells rather than map units. Only the main parameters are filled; advanced ones keep their defaults unless you ask for them.
4. **Fix small slips.** Before running, GeoAgent corrects common mistakes: a near-miss layer name ("River" for `rivers`), an option given by its label ("Flat" instead of its number), a parameter name in the wrong case. Values QGIS already accepts are never changed.
5. **Check, then run.** QGIS checks the parameters before anything runs. The algorithm then executes, and its outputs are added to your project.
6. **Retry on failure.** If a task fails, GeoAgent works out why and tries again, up to two times. If only a parameter value was wrong, it keeps the algorithm and fixes the parameters. Otherwise it picks a different algorithm.
7. **Pass results on.** Each task's output is handed to the tasks that depend on it, so "buffer the rivers, then clip the DEM with the buffer" clips with the buffer that was just made.
8. **Summarize.** The reply says what was done and where the results are. Under it are links to [open the run as a model](model-export.md).

If a task still fails after the retries, the remaining tasks don't run, and the reply explains what went wrong.

## Writing good requests

- **Name your layers and give units:** "buffer **cities** by **500 m**", not "make a buffer".
- **Refer to earlier steps naturally:** "...and clip the roads with **that buffer**".
- **Use file paths** for data that isn't loaded yet; they work as inputs directly.
- **One request, one analysis.** Each processing request stands alone and doesn't use earlier chat messages, so name the layers you mean, including results of earlier requests, which stay in your project.

## Result layers

Results are added to your project under the names the algorithms give them, such as `Buffered` or `Result - slope`. If a layer with that name already exists, a number is added (`Buffered (2)`), so a result is never confused with an older one.
