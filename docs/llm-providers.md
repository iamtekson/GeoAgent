# LLM providers

GeoAgent works with a local model through Ollama, or with a cloud provider. Choose one in the **Provider** list on the Settings tab.

| Provider | Cost and privacy | API key | Default model |
| --- | --- | --- | --- |
| **Ollama** | Free; runs on your computer, data stays local | not needed | `llama3.2:3b` |
| **OpenAI** | Paid per token | [platform.openai.com](https://platform.openai.com/api-keys) | `gpt-5` |
| **Gemini** | Paid per token | [aistudio.google.com](https://aistudio.google.com/app/apikey) | `gemini-3-flash-preview` |
| **Anthropic** | Paid per token | [platform.claude.com](https://platform.claude.com/) | `claude-sonnet-5` |

:::{tip}
Small local models handle General mode well. Processing mode involves several structured reasoning steps per request, so a larger local model (8B parameters or more) or a cloud model gives noticeably better results.
:::

## Ollama (local)

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Download a model and make sure the server is running:

   ```bash
   ollama pull llama3.2:3b
   ollama serve
   ```

3. In GeoAgent's Settings tab, select **Ollama**. Keep the base URL `http://localhost:11434` unless your server runs elsewhere, and enter the model name.

If the model isn't installed yet, GeoAgent offers to download it for you when you send your first message.

## OpenAI, Gemini and Anthropic (cloud)

1. Create an API key on the provider's site (links above).
2. In the Settings tab, select the provider, paste the key into **API Key**, and enter a **Model Name**. The field is pre-filled with the provider's default model.
3. Click **Save Settings** so the key and model survive a QGIS restart. Each provider keeps its own key, so switching providers doesn't lose the others.

:::{note}
Saved API keys are stored in your QGIS user profile settings, which are not encrypted. Don't save keys on a shared computer account.
:::

The **Temperature** setting is not sent to Anthropic models, because current Claude models reject custom temperature values.

Next: try the [quickstart](quickstart.md).
