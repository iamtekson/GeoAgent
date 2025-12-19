# GeoAgent Architecture

## Project Structure

```
geo_agent/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration and supported models
├── llm/
│   ├── __init__.py
│   └── client.py                # LLM client abstraction layer
├── prompts/
│   ├── __init__.py
│   └── system.py                # System prompts for different modes
├── agents/
│   ├── __init__.py
│   └── general.py               # General mode agent (with LangGraph support)
├── geo_agent.py                 # Main plugin class
├── geo_agent_dialog.py          # Dialog UI handler
├── geo_agent_dialog_base.ui     # UI definition
└── ...
```

## Components Overview

### 1. **config/settings.py**

Centralized configuration:

- Supported models (Ollama, ChatGPT, Gemini)
- API key file location
- Temperature and token settings
- Default parameters

### 2. **llm/client.py**

Multi-provider LLM abstraction:

- `LLMClient`: Base class
- `OllamaClient`: For local Ollama models
- `OpenAIClient`: For ChatGPT models
- `GeminiClient`: For Google Gemini models
- `create_client()`: Factory function

All clients implement:

- `query()`: Send message and get response
- `validate_connection()`: Check service availability

### 3. **prompts/system.py**

System prompts:

- `GENERAL_SYSTEM_PROMPT`: For general chat mode
- `AGENTIC_SYSTEM_PROMPT`: For future agentic mode with tools

### 4. **agents/general.py**

General mode agent:

- `ChatMessage`: Message representation
- `AgentState`: Maintains chat history and state
- `GeneralModeAgent`: Main agent class
- `LangGraphGeneralAgent`: Placeholder for LangGraph implementation

## How It Works

### Chat Flow (General Mode)

1. **User sends message** via UI
2. **geo_agent.py** calls `send_message()`
3. **Model selection** triggers agent initialization
4. **\_initialize_agent()** creates appropriate LLM client
5. **GeneralModeAgent.process_query()** handles the conversation
6. **LLM client** sends query to selected provider
7. **Response displayed** in chat area
8. **Chat history maintained** in agent state

### Model Support

#### Ollama (Local)

```python
# No API key needed, runs locally
client = create_client("ollama", model="llama2")
```

#### ChatGPT (OpenAI)

```python
# Requires OpenAI API key
client = create_client("openai", api_key=key, model="gpt-3.5-turbo")
```

#### Gemini (Google)

```python
# Requires Google API key
client = create_client("gemini", api_key=key, model="gemini-pro")
```

## Future Extensions

### 1. Agentic Mode with LangGraph

Currently, `LangGraphGeneralAgent` is a placeholder. To implement:

- Define tool nodes (e.g., buffer operations, raster analysis)
- Create workflow graph with conditional routing
- Implement tool execution nodes
- Add state management for multi-step tasks

### 2. Additional LLM Providers

Add new providers by:

1. Creating new client class inheriting `LLMClient`
2. Implementing `query()` and `validate_connection()`
3. Adding to `SUPPORTED_MODELS` in settings
4. Updating factory function

### 3. Custom System Prompts

- Add new prompts in `prompts/system.py`
- Create prompt templates for different GIS tasks
- Support dynamic prompt loading

## API Key Management

- API keys stored in `~/.geo_agent/api_key.txt`
- Auto-loaded on plugin start
- User can override via UI input field
- Never exposed in error messages

## Error Handling

All exceptions caught and displayed via QGIS message bar:

- Connection errors
- Invalid API keys
- Model not found
- LLM service unavailable

## Chat History

- Maintained in `AgentState.messages`
- Includes system prompt in context
- Limited to 10 most recent messages for token efficiency
- Can be cleared via `agent.clear_history()`
