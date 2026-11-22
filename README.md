## OVH AI Endpoints Provider

This project is a [Dify](https://dify.ai/) model provider plugin for [OVH AI Endpoints](https://www.ovhcloud.com/en/public-cloud/ai-endpoints/).

This provider is configured for OVH AI Endpoints with a predefined model catalog.

### What users need to configure

Only credentials:

- `api_key` (required)

No endpoint URL is required in UI. The OVH OpenAI-compatible base URL is injected automatically for LLM, moderation, embeddings, and speech-to-text. TTS uses OVH native per-model endpoints.

OVH references:
- AI Endpoints overview: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/
- AI Endpoints catalog: https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/

### OVH AI Endpoints Models Matrix

Reference date: 2026-11-22. Model capabilities and EUR prices come from the
[OVHcloud Catalog API](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-catalog-api).
LLM token limits are capped by the OpenAI-compatible router's `/v1/models` metadata.

#### Workflow code-node profiles

- `LLM_STANDARD`: `temperature`, `top_p`, `max_tokens`, structured output controlled by the native Dify LLM/Agent blocks.
- `LLM_REASONING`: `LLM_STANDARD` with reasoning output and model-specific `reasoning_effort` options when provided by OVH.
- `LLM_VISION`: `LLM_STANDARD` with multimodal `messages` payloads (`text` + `image_url`) and, when exposed by the model, `reasoning_effort`.
- `LLM_TRANSLATION`: send the text to translate as the user message and set `source_language` and `target_language` to the language names in English.
- `MODERATION_GUARD`: moderation/safety classification, boolean blocked/safe output.
- `EMBEDDINGS`: endpoint `/v1/embeddings`, body `{model, input}` where `input` is `str|list[str]`, output vectors.
- `SPEECH2TEXT`: endpoint `/v1/audio/transcriptions`, multipart body with `file`, `model`, optional `language`, `prompt`, `response_format`.
- `TTS`: OVH native endpoint `https://{model}.endpoints.kepler.ai.cloud.ovh.net/api/v1/tts/text_to_audio`, body `{encoding, language_code, sample_rate_hz, text, voice_name}`, output WAV audio bytes converted to MP3 before returning to Dify.

#### Per-model mapping

| Model | Dify type | Profile | Context (tokens) |
|---|---|---|---:|
| `gpt-oss-120b` | `llm` | `LLM_REASONING` | 131,072 |
| `gpt-oss-20b` | `llm` | `LLM_REASONING` | 131,072 |
| `DeepSeek-V4.1-Flash-1017` | `llm` | `LLM_VISION` | 1,048,576 |
| `Qwen3.8-27B` | `llm` | `LLM_VISION` | 262,144 |
| `Qwen3.5-397B-A17B` | `llm` | `LLM_VISION` | 262,144 |
| `Mistral-Small-3.2-24B-Instruct-2506` | `llm` | `LLM_VISION` | 131,072 |
| `qwen3-vl-30b-a3b` | `llm` | `LLM_VISION` | 32,768 |
| `qwen3.6-35b-a3b` | `llm` | `LLM_REASONING` | 262,144 |
| `nemotron3-nano-30b` | `llm` | `LLM_REASONING` | 262,144 |
| `nemotron3-omni-30b` | `llm` | `LLM_VISION` | 131,072 |
| `gemma-4-26b-a4b` | `llm` | `LLM_REASONING` | 262,144 |
| `laguna-xs-2.1` | `llm` | `LLM_REASONING` | 131,072 |
| `Olala-v1.0-mini-7A1B` | `llm` | `LLM_REASONING` | 32,688 |
| `Olala-v1.0-summarise` | `llm` | `LLM_REASONING` | 32,768 |
| `Olala-v1.0-translate` | `llm` | `LLM_TRANSLATION` | 8,192 |
| `Qwen3.5-9B` | `llm` | `LLM_VISION` | 262,144 |
| `Qwen3.6-27B` | `llm` | `LLM_VISION` | 262,144 |
| `Qwen2.5-VL-72B-Instruct` | `llm` | `LLM_VISION` | 32,768 |
| `Meta-Llama-3_3-70B-Instruct` | `llm` | `LLM_STANDARD` | 131,072 |
| `Qwen3Guard-Gen-8B` | `moderation` | `MODERATION_GUARD` | 32,768 |
| `Qwen3Guard-Gen-0.6B` | `moderation` | `MODERATION_GUARD` | 32,768 |
| `Qwen3-Embedding-8B` | `text-embedding` | `EMBEDDINGS` | 32,768 |
| `bge-multilingual-gemma2` | `text-embedding` | `EMBEDDINGS` | 8,192 |
| `bge-m3` | `text-embedding` | `EMBEDDINGS` | 8,192 |
| `whisper-large-v3` | `speech2text` | `SPEECH2TEXT` | — |
| `whisper-large-v3-turbo` | `speech2text` | `SPEECH2TEXT` | — |
| `nvr-tts-en-us` | `tts` | `TTS` | — |
| `nvr-tts-de-de` | `tts` | `TTS` | — |
| `nvr-tts-it-it` | `tts` | `TTS` | — |
| `nvr-tts-es-es` | `tts` | `TTS` | — |

#### Scheduled retirements

These models are still callable and remain available in the plugin for existing workflows. OVHcloud lists the following end-of-life dates.

| Model | End of life | Migration option |
|---|---|---|
| `bge-multilingual-gemma2` | 2027-01-21 | `Qwen3-Embedding-8B` or `bge-m3`. Re-index the affected Dify knowledge bases when switching embedding models. |
| `Qwen3.6-27B` | 2027-02-10 | `Qwen3.8-27B` |
| `Qwen2.5-VL-72B-Instruct` | 2027-02-10 | `Qwen3.8-27B` or `qwen3-vl-30b-a3b` |
| `Meta-Llama-3_3-70B-Instruct` | 2027-02-10 | `Mistral-Small-3.2-24B-Instruct-2506` |

#### Other OVHcloud models

- `solaria-3` uses the [Gladia live transcription API](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-gladia-speech-to-text). The plugin's speech-to-text models use the OpenAI-compatible file transcription route.
- `clef-flash` uses the [OVHcloud Decision API](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-decision-models).
- `stable-diffusion-xl-base-v10` is an image-generation endpoint with end of life scheduled for 2027-02-10.

### Notes

- OVH's Olala v1.0 endpoints return an empty stream with custom stop sequences, so the plugin omits stop sequences for those models.
- For TTS, OVH currently returns WAV from its native endpoint and the plugin converts it to MP3 before returning it to Dify.
- Chrome is known to play the current TTS path correctly. Firefox may fail in Dify UI depending on browser MediaSource support for the MIME used by the frontend.

### Submission metadata

- Repository: https://github.com/rungeard/dify_ovh_ai_endpoints
- Contact: dify@rungeard.eu
- Privacy policy: `./PRIVACY.md`
