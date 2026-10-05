# General mode

General mode is a conversation with an assistant that can act on your QGIS project. Use it for questions, data exploration, selections and layer management. It's the default mode; choose it with the **General** button in the Chat tab.

```text
Add C:\data\cities.shp
What columns does the cities layer have?
Select cities with population greater than 100000
Zoom to the boundary layer
What is the difference between a buffer and a dissolve?
```

## What it can do

The assistant decides when to use these tools:

| Area | Tool | Notes |
| --- | --- | --- |
| Layers | Add a layer | From a file path or URL; vector or raster is detected automatically. |
| | List layers | Name, type, feature count and visibility. |
| | Show a layer's columns | Field names, types and sample statistics. |
| | Zoom to a layer | |
| | Remove a layer | Always asks you to confirm first. |
| Selection | Select by attribute | Operators `=`, `!=`, `<`, `>`, `<=`, `>=`, `contains`, `starts_with`, `ends_with`. |
| | Select by geometry | `largest`, `smallest`, or features `intersecting`, `inside` or `touching` another layer. |
| Projects | Create, load, save, delete a project | `.qgs` or `.qgz` files. |

Layer names don't have to be exact: "stream network of nepal" finds a layer named `stream_network_nepal`.

## Conversation memory

General mode remembers the conversation, so follow-up questions work ("now zoom to it"). In very long chats, only the most recent 20 messages are sent to the model. **Clear Chat** starts a fresh conversation.

For geoprocessing (buffer, clip, statistics, ...) switch to [Processing mode](processing-mode.md).
