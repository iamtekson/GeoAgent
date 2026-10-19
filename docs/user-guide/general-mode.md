# General mode

General mode is a conversation with an assistant that can act on your QGIS project. Use it for questions, data exploration, selections and layer management. It's the default mode; choose it with the **General** button in the Chat tab.

```text
Add C:\data\cities.shp
Load all the layers from C:\data\project
What columns does the cities layer have?
Select cities with population greater than 100000
Select the people with kshetri in their name
Which parcels are inside the city of Pokhara?
How many schools are there?
Zoom to the boundary layer
What is the difference between a buffer and a dissolve?
```

## What it can do

The assistant decides when to use these tools:

| Area | Tool | Notes |
| --- | --- | --- |
| Layers | Add a layer | From a file, a folder or a URL. QGIS detects the format (vector, raster, mesh, point cloud). For a folder, every layer the QGIS Browser shows in it is added (optionally including subfolders). Every layer of a multi-layer file such as a GeoPackage is added, or one with `file.gpkg\|layername=roads`. |
| | List layers | Name, type, feature count and visibility. |
| | Show a layer's columns | Column names and types, with some values of each column (or its range for numbers and dates). |
| | Zoom to a layer | |
| | Remove a layer | Always asks you to confirm first. |
| Selection | Select by expression | QGIS's *Select by Expression*: any condition QGIS can express. |
| | Select by location | QGIS's *Select by Location*, or *Select within Distance* when you give a distance. Can use only the selected features of the other layer. |
| | Calculate | Counts, totals, averages and extremes over a layer or its selection, with QGIS aggregate expressions. |
| | Zoom to selection, clear selection | |
| Projects | Create, load, save, delete a project | `.qgs` or `.qgz` files. |

Layer names don't have to be exact: "stream network of nepal" finds a layer named `stream_network_nepal`.

## How filtering works

The assistant turns your request into a QGIS expression, and QGIS's own tools run it. "Select people with kshetri in their name" becomes `"name" ILIKE '%kshetri%'`, which also matches `Kshetri` and `KSHETRI`.

Your words don't have to match the column names. The assistant looks at the layer's columns and their values first, then:

- **One column clearly fits:** it uses that column and tells you which, for example `dim` for "diameter".
- **Several columns could fit:** it asks you which one you mean.

QGIS checks every expression before it runs. If a column doesn't exist, the assistant gets the layer's real columns and values back and can correct itself. When nothing matches, the reply shows the values of the columns used, so a misspelling is easy to spot.

Each selection reply gives the number of selected features and lists the first few, so "which" and "where" questions get a real answer.

## Geoprocessing

General mode doesn't run processing algorithms (buffer, clip, dissolve, statistics, ...). If you ask for one, the assistant explains how it's done and reminds you to switch to [Processing mode](processing-mode.md).

## Conversation memory

General mode remembers the conversation, so follow-up questions work ("now zoom to it"). In very long chats, only the most recent 20 messages are sent to the model. **Clear Chat** starts a fresh conversation.
