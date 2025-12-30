# GeoAgent

<p align="center">
  <img src="icons/icon.png" alt="GeoAgent Logo" width="120"/>
</p>

GeoAgent is a QGIS plugin that integrates Large Language Models (LLMs) to enable geospatial analysis and data processing through natural language commands. Chat with your GIS data, perform complex analyses, and automate workflows using conversational AI.

## 📑 Table of Contents

- [🌟 Features](#-features)
- [📋 Requirements](#-requirements)
- [🚀 Getting Started](#-getting-started)
- [⚙️ Settings & Configuration](#️-settings--configuration)
- [💡 Usage Examples](#-usage-examples)
- [🎓 Tips & Best Practices](#-tips--best-practices)
- [🔧 Development & Contributing](#-development--contributing)
- [📄 License](#-license)
- [👥 Authors](#-authors)
- [🙏 Acknowledgments](#-acknowledgments)
- [References](#-references)
- [📞 Support](#-support)
- [🔗 Links](#-links)

## 🌟 Features

GeoAgent provides powerful geospatial capabilities through natural language interaction:

### 📊 Layer Management

- **Add layers**: Load vector and raster layers from files or URLs
- **Remove layers**: Delete layers from your project with confirmation
- **List layers**: View all layers with detailed information
- **Zoom to layer**: Navigate to any layer's extent
- **Layer information**: Get column names, types, and statistics

### 🔍 Data Analysis & Selection

- **Select by attribute**: Filter features based on field values
  - Operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `contains`, `starts_with`, `ends_with`
- **Select by geometry**: Spatial selections based on geometric criteria
  - Options: `largest`, `smallest`, `intersecting`, `inside`, `touching`
- **Attribute queries**: Ask questions about your data

### 🛠️ Geoprocessing Operations (Future Release)

- **Buffer**: Create buffer zones around features
- **Clip**: Clip layers by extent or other layers
- **Intersection**: Find spatial intersections
- **Union**: Merge geometries
- **Dissolve**: Combine features
- And many more QGIS processing algorithms accessible through natural language

### 💬 Two Operation Modes

#### General Mode

- General conversation about GIS concepts
- Layer exploration and information retrieval
- Data querying and analysis
- Tool selection and guidance

#### Processing Mode (Future Release)

- Automated geoprocessing workflow execution
- Algorithm detection and parameter extraction
- Interactive parameter verification dialogs
- Direct processing algorithm execution

### 🎯 Additional Features

- **Export chat**: Save your conversation history
- **Clear chat**: Start fresh conversations
- **Multi-LLM support**: Choose from Ollama (local), ChatGPT, or Gemini
- **Customizable parameters**: Adjust temperature and token limits
- **Non-blocking execution**: Continue working while AI processes

## 📋 Requirements

- **QGIS**: Version 3.2 or higher
- **Python**: 3.9+ (included with QGIS)
- **LLM Provider**: At least one of the following:
  - Ollama (local, free)
  - OpenAI API key (ChatGPT)
  - Google API key (Gemini)

## 🚀 Getting Started

### Installation

#### Option 1: From QGIS Plugin Repository (Recommended)

1. Open QGIS
2. Go to `Plugins` → `Manage and Install Plugins`
3. Search for "GeoAgent"
4. Click `Install Plugin`
5. Close the dialog

#### Option 2: Manual Installation

1. Download the latest release from [GitHub Releases](https://github.com/iamtekson/GeoAgent/releases)
2. Extract the zip file to your QGIS plugins directory:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - macOS: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
   - Linux: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Restart QGIS
4. Enable the plugin in `Plugins` → `Manage and Install Plugins` → `Installed`

### Setting Up Ollama (Recommended for Beginners)

Ollama is a free, local LLM provider that runs on your computer without requiring API keys or internet connection.

#### Step 1: Install Ollama

**Windows:**

1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer
3. Ollama will start automatically

**macOS:**

```bash
brew install ollama
ollama serve
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

#### Step 2: Download a Model

Open a terminal/command prompt and run:

```bash
# Recommended model (small and fast)
ollama pull llama3.2:3b

# Or other options:
ollama pull llama3.2:1b    # Smaller, faster
ollama pull llama3.1:8b    # Larger, more capable
ollama pull mistral        # Alternative model
```

#### Step 3: Verify Ollama is Running

```bash
# Check if Ollama is running
ollama list

# Test the model
ollama run llama3.2:3b "Hello, how are you?"
```

You should see a response. Ollama is now ready!

### Setting Up ChatGPT (OpenAI)

1. Get an API key from [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. In GeoAgent settings:
   - Select "ChatGPT" from the model dropdown
   - Enter your API key
   - Choose a model (e.g., `gpt-4o-mini`, `gpt-4o`)

### Setting Up Gemini (Google)

1. Get an API key from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
2. In GeoAgent settings:
   - Select "Gemini" from the model dropdown
   - Enter your API key
   - Choose a model (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`)

## ⚙️ Settings & Configuration

### Accessing the Plugin

1. After installation, find GeoAgent in:
   - `Plugins` → `GeoAgent` menu
   - Or click the GeoAgent icon in the toolbar
2. The GeoAgent panel will appear (usually docked on the right side)

### Model Selection

**Dropdown Menu**: Choose your LLM provider

- **Ollama**: Local models (no API key needed)
- **ChatGPT**: OpenAI's GPT models (requires API key)
- **Gemini**: Google's Gemini models (requires API key)

**Switching Models**:

- Select a different provider from the dropdown
- The UI will automatically update to show relevant settings
- Your conversation history is preserved when switching

### Model-Specific Settings

#### Ollama Settings

- **Base URL**: Default is `http://localhost:11434` (usually no need to change)
- **Model Name**: Enter the model you pulled (e.g., `llama3.2:3b`)
  - To see available models, run `ollama list` in terminal

#### ChatGPT Settings

- **API Key**: Your OpenAI API key
- **Model**: Select from dropdown (gpt-4o-mini, gpt-4o, etc.)

#### Gemini Settings

- **API Key**: Your Google AI API key
- **Model**: Select from dropdown (gemini-1.5-flash, gemini-1.5-pro)

### Advanced Settings

**Temperature** (0.0 - 1.0)

- Controls randomness of responses
- **0.0**: Deterministic, consistent responses
- **0.8**: Balanced (default)
- **1.0**: More creative, varied responses
- **When to adjust**:
  - Lower for precise data analysis
  - Higher for brainstorming or creative tasks

**Max Tokens** (1000 - 10000)

- Maximum length of AI responses
- **Default**: 5000
- **When to adjust**:
  - Increase for complex queries requiring detailed responses
  - Decrease to save API costs or speed up responses

### Operation Modes

Use the radio buttons to switch between modes:

#### 🔵 General Mode (Default)

**Use when**:

- Asking questions about layers
- Exploring data
- Getting column information
- Selecting features
- General GIS assistance

**Examples**:

```
"What layers are in my project?"
"Show me the columns in the cities layer"
"Select all roads where type is 'highway'"
"Zoom to the boundary layer"
```

#### 🟢 Processing Mode

**Use when**:

- Running geoprocessing algorithms
- Creating buffers, clips, intersections
- Performing spatial analysis
- Executing QGIS processing tools

**Examples**:

```
"Create a 500m buffer around the cities layer"
"Clip the roads layer by the study area"
"Calculate the intersection of parcels and zones"
"Dissolve the polygons by district field"
```

**What happens in Processing Mode**:

1. AI detects the processing algorithm needed
2. Extracts parameters from your query
3. Shows verification dialog with editable parameters
4. You confirm or modify parameters
5. Processing algorithm executes
6. Results are added to your project

### Chat Controls

**Send Button**

- Click to send your message
- Keyboard shortcut: `Enter` (or `Shift+Enter` for new line)
- Button disabled during processing to prevent duplicates

**Export Chat**

- Saves conversation history to a text file
- Useful for documentation or sharing workflows
- Preserves both questions and responses

**Clear Chat**

- Removes all conversation history
- Starts fresh conversation
- Doesn't affect your QGIS project

**Question Input Box**

- Type your natural language queries here
- Cleared automatically after sending

## 💡 Usage Examples

### Example 1: Loading and Exploring Data

```
You: "Add a shapefile from C:\data\cities.shp"
AI: "Success: Added vector layer 'cities' to QGIS with 150 features."

You: "What columns does the cities layer have?"
AI: [Returns detailed column information with types and sample values]

You: "Show me cities with population greater than 100000"
AI: "Success: Selected 23 features in 'cities' where population > 100000."
```

### Example 2: Geoprocessing Workflow

**Switch to Processing Mode** first, then:

```
You: "Create a 1000 meter buffer around the cities layer"
AI: [Shows verification dialog with parameters]
    - Input Layer: cities
    - Distance: 1000
    - Segments: 5
[You confirm]
AI: "Success: Buffer created. Output added as 'cities_buffer'."

You: "Clip the roads layer using the buffer I just created"
AI: [Shows verification dialog]
[You confirm]
AI: "Success: Clipped roads layer created."
```

### Example 3: Data Analysis

```
You: "List all the layers in my project"
AI: [Returns formatted list of all layers with details]

You: "Zoom to the boundary layer"
AI: "Success: Zoomed to layer 'boundary'."

You: "Select features in the parcels layer where landuse equals 'residential'"
AI: "Success: Selected 342 features in 'parcels' where landuse = 'residential'."
```

### Example 4: Spatial Queries

```
You: "Select all points that are inside the study area polygon"
AI: "Success: Selected 156 features in 'points' that are inside 'study_area'."

You: "Find the largest polygon in the parcels layer"
AI: "Success: Selected 1 feature - the largest polygon."
```

## 🎓 Tips & Best Practices

### Writing Effective Queries

**Be Specific**:

- ✅ "Create a 500m buffer around the cities layer"
- ❌ "Make a buffer"

**Use Exact Layer Names**:

- ✅ "Select roads where type = 'highway'"
- ❌ "Select some roads" (if layer isn't called "roads")

**Include Units**:

- ✅ "Buffer by 1000 meters"
- ✅ "Buffer by 1 km" (AI will interpret)
- ❌ "Buffer by 1000" (ambiguous)

**Break Complex Tasks into Steps**:

```
Step 1: "Create a 500m buffer around cities"
Step 2: "Clip the roads layer by the cities_buffer"
Step 3: "Calculate the length of roads in the clipped layer"
```

### Choosing the Right Mode

| Task Type           | Mode       | Example                         |
| ------------------- | ---------- | ------------------------------- |
| Exploring data      | General    | "What layers do I have?"        |
| Selecting features  | General    | "Select cities with pop > 5000" |
| Getting information | General    | "Show columns of roads layer"   |
| Running algorithms  | Processing | "Buffer cities by 1km"          |
| Spatial analysis    | Processing | "Intersect parcels and zones"   |
| Data transformation | Processing | "Dissolve by district field"    |

### Troubleshooting

**"Error: QGIS interface not initialized"**

- Restart QGIS
- Disable and re-enable the plugin

**"Error: Layer 'xyz' not found"**

- Check layer name spelling (case-sensitive)
- Use `list_qgis_layers()` to see exact names

**"Ollama connection failed"**

- Verify Ollama is running: `ollama list`
- Check base URL is `http://localhost:11434`
- Restart Ollama: `ollama serve`

**"Model not responding"**

- Check API key validity
- Verify internet connection (ChatGPT/Gemini)
- Check model name is correct
- Try increasing max tokens

**Processing mode not working**

- Ensure you're in Processing Mode (radio button selected)
- Be specific about algorithm and parameters
- Check the verification dialog for errors

## 🔧 Development & Contributing

### Project Structure

```
geo_agent/
├── agents/          # LangGraph agents (general, processing)
├── config/          # Configuration and settings
├── dialogs/         # Qt dialogs and UI
├── icons/           # Plugin icons
├── llm/             # LLM client and worker threads
├── prompts/         # System prompts
├── tools/           # LangChain tools for QGIS operations
│   ├── commons.py   # Common utilities
│   ├── filters.py   # Selection tools
│   ├── geoprocessing.py  # Processing algorithms
│   └── io.py        # Layer I/O operations
└── geo_agent.py     # Main plugin class
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Reporting Issues

Found a bug? Have a suggestion?

- Open an issue at [GitHub Issues](https://github.com/iamtekson/GeoAgent/issues)
- Include QGIS version, plugin version, and steps to reproduce

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Tek Kshetri** - [@iamtekson](https://github.com/iamtekson)
- **Rabin Ojha** - Contributor

## 🙏 Acknowledgments

- QGIS Development Team for the excellent GIS platform
- LangChain for the AI framework
- Ollama, OpenAI, and Google for LLM providers
- The open-source GIS community

## References

- [QChatGPT Plugin](https://github.com/KIOS-Research/QChatGPT) by [@KIOS-Research](https://github.com/KIOS-Research)
- [GeoAI Plugin](https://github.com/opengeos/geoai/tree/main/qgis_plugin) by [@giswqs](https://github.com/giswqs)
- [QGIS MCP Plugin](https://github.com/jjsantos01/qgis_mcp) by [@jjsantos01](https://github.com/jjsantos01)

## 📞 Support

- **Documentation**: [GitHub Readme](https://github.com/iamtekson/GeoAgent)
- **Issues**: [GitHub Issues](https://github.com/iamtekson/GeoAgent/issues)
- **Email**: iamtekson@gmail.com

## 🔗 Links

- **Repository**: https://github.com/iamtekson/GeoAgent
- **QGIS Plugins**: https://plugins.qgis.org/plugins/geo_agent/
- **Ollama**: https://ollama.com/
- **QGIS**: https://qgis.org/

---

**Made with ❤️ for the GIS community**
