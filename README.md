## OVH AI Endpoints Provider

This provider is configured for OVH AI Endpoints with a predefined model catalog.

### What users need to configure

Only credentials:

- `api_key` (required)

No endpoint URL is required in UI. The OVH OpenAI-compatible base URL is injected automatically.

### OVH AI Endpoints Models Matrix

Reference date: 2026-03-31.

#### Workflow code-node profiles

- `LLM_STANDARD`: `temperature`, `top_p`, `max_tokens`, `response_format`, `json_schema` (optional).
- `LLM_REASONING`: `LLM_STANDARD` + `reasoning_effort` (`low|medium|high`) and optional hidden reasoning output handling.
- `LLM_VISION`: `LLM_STANDARD` with multimodal `messages` payloads (`text` + `image_url`).
- `LLM_GUARD`: classifier-style prompt, low temperature (`0`), short `max_tokens`.
- `EMBEDDINGS`: endpoint `/v1/embeddings`, body `{model, input}` where `input` is `str|list[str]`, output vectors.
- `SPEECH2TEXT`: endpoint `/v1/audio/transcriptions`, multipart body with `file`, `model`, optional `language`, `prompt`, `response_format`.
- `TTS`: endpoint `/v1/audio/speech`, body `{model, input, voice}`, output audio bytes.
- `IMAGE_GENERATION`: endpoint `/v1/images/generations`, body `{model, prompt, size, response_format}`.

#### Per-model mapping

| Model | OVH category | Dify type in this provider | I/O summary | Profile |
|---|---|---|---|---|
| `gpt-oss-120b` | Reasoning LLM | `llm` | text/json out, tool calling, reasoning | `LLM_REASONING` |
| `gpt-oss-20b` | Reasoning LLM | `llm` | text/json out, tool calling, reasoning | `LLM_REASONING` |
| `Qwen3-32B` | Reasoning LLM | `llm` | text/json out, tool calling, reasoning | `LLM_REASONING` |
| `Qwen3-Coder-30B-A3B-Instruct` | Code LLM | `llm` | code-oriented text/json out, tool calling | `LLM_STANDARD` |
| `Mistral-Small-3.2-24B-Instruct-2506` | Visual LLM | `llm` | multimodal in, text/json out, tool calling | `LLM_VISION` |
| `Qwen2.5-VL-72B-Instruct` | Visual LLM | `llm` | multimodal in, text/json out | `LLM_VISION` |
| `Meta-Llama-3_3-70B-Instruct` | LLM | `llm` | text/json out, tool calling | `LLM_STANDARD` |
| `Mistral-Nemo-Instruct-2407` | LLM | `llm` | text/json out, tool calling | `LLM_STANDARD` |
| `Mistral-7B-Instruct-v0.3` | LLM | `llm` | text/json out, tool calling | `LLM_STANDARD` |
| `Qwen3Guard-Gen-8B` | LLM Guard | `llm` | moderation-style classification text | `LLM_GUARD` |
| `Qwen3Guard-Gen-0.6B` | LLM Guard | `llm` | moderation-style classification text | `LLM_GUARD` |
| `Qwen3-Embedding-8B` | Embeddings | `text-embedding` | text in, vectors out | `EMBEDDINGS` |
| `bge-multilingual-gemma2` | Embeddings | `text-embedding` | text in, vectors out | `EMBEDDINGS` |
| `bge-m3` | Embeddings | `text-embedding` | text in, vectors out | `EMBEDDINGS` |
| `whisper-large-v3` | Automatic Speech Recognition | `speech2text` | audio in, transcript out | `SPEECH2TEXT` |
| `whisper-large-v3-turbo` | Automatic Speech Recognition | `speech2text` | audio in, transcript out | `SPEECH2TEXT` |
| `nvr-tts-en-us` | Text to Speech | `tts` | text in, audio out | `TTS` |
| `nvr-tts-de-de` | Text to Speech | `tts` | text in, audio out | `TTS` |
| `nvr-tts-it-it` | Text to Speech | `tts` | text in, audio out | `TTS` |
| `nvr-tts-es-es` | Text to Speech | `tts` | text in, audio out | `TTS` |
| `stable-diffusion-xl-base-v10` | Image Generation | not exposed as Dify model type | prompt in, image out | `IMAGE_GENERATION` |

#### Context windows from OVH catalog

- `Qwen3-Coder-30B-A3B-Instruct`: 256K
- `gpt-oss-120b`: 131K
- `gpt-oss-20b`: 131K
- `Qwen3-32B`: 32K
- `Mistral-Small-3.2-24B-Instruct-2506`: 128K
- `Meta-Llama-3_3-70B-Instruct`: 131K
- `Mistral-7B-Instruct-v0.3`: 127K
- `Qwen2.5-VL-72B-Instruct`: 32K
- `Mistral-Nemo-Instruct-2407`: 118K
- `Qwen3Guard-Gen-8B`: 32K
- `Qwen3Guard-Gen-0.6B`: 32K

### Submission metadata

- Repository: https://github.com/rungeard/dify_ovh_ai_endpoints
- Contact: https://github.com/rungeard/dify_ovh_ai_endpoints/issues
- Privacy policy: `./PRIVACY.md`

### Packaging for submission

Use the package naming convention required for marketplace workflows:

- `ovh_ai_endpoints-0.0.1.difypkg`
