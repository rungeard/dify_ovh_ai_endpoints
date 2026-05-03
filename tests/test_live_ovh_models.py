from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import struct
import sys
import wave
from binascii import hexlify
from io import BytesIO
from pathlib import Path

import pytest
import yaml
from dify_plugin.config.integration_config import IntegrationConfig
from dify_plugin.core.entities.plugin.request import (
    ModelActions,
    ModelInvokeLLMRequest,
    ModelInvokeModerationRequest,
    ModelInvokeSpeech2TextRequest,
    ModelInvokeTextEmbeddingRequest,
    ModelInvokeTTSRequest,
    PluginInvokeType,
)
from dify_plugin.entities.model import AIModelEntity, ModelType
from dify_plugin.entities.model.llm import LLMResultChunk
from dify_plugin.entities.model.message import PromptMessageTool
from dify_plugin.entities.model.moderation import ModerationResult
from dify_plugin.entities.model.speech2text import Speech2TextResult
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.entities.model.tts import TTSResult
from dify_plugin.integration.run import PluginRunner

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OpenAITextEmbeddingModel = importlib.import_module(
    "models.text_embedding.text_embedding"
).OpenAITextEmbeddingModel
_TTS_MODEL_CONFIG = importlib.import_module("models.tts.tts")._TTS_MODEL_CONFIG

PROVIDER_NAME = "ovh_ai_endpoints"
TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0xQAAAAASUVORK5CYII="
)


def _load_yaml(path: Path) -> dict | list:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_model_names_from_position(group: str) -> list[str]:
    position_file = MODELS_DIR / group / "_position.yaml"
    if not position_file.exists():
        raise FileNotFoundError(f"Missing model position file: {position_file}")

    data = _load_yaml(position_file)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"Expected a YAML list in {position_file}")

    return [item.strip() for item in data if isinstance(item, str) and item.strip()]


def _load_model_names_from_manifests(group: str) -> list[str]:
    names: list[str] = []
    for path in sorted((MODELS_DIR / group).glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = _load_yaml(path)
        names.append(str(data["model"]))
    return names


def _load_manifest_by_model(group: str) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    for path in sorted((MODELS_DIR / group).glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        data = _load_yaml(path)
        manifests[str(data["model"])] = data
    return manifests


LLM_MODELS = _load_model_names_from_position("llm")
LLM_MANIFESTS = _load_manifest_by_model("llm")
MODERATION_MODELS = _load_model_names_from_position("moderation")
EMBEDDING_MODELS = _load_model_names_from_manifests("text_embedding")
EMBEDDING_MANIFESTS = _load_manifest_by_model("text_embedding")
SPEECH2TEXT_MODELS = _load_model_names_from_manifests("speech2text")
TTS_MODELS = _load_model_names_from_manifests("tts")
TTS_MANIFESTS = _load_manifest_by_model("tts")


def _skip_if_rate_limited(exc: Exception) -> None:
    message = str(exc).lower()
    if "429" in message or "rate limit" in message:
        pytest.skip(str(exc))


def _make_test_wav_file_bytes(duration_seconds: float = 0.35, sample_rate: int = 16000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    buffer = BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for index in range(frames):
            amplitude = int(16000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            wav_file.writeframesraw(struct.pack("<h", amplitude))

    return buffer.getvalue()


def _stage_plugin_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    staged_dir = tmp_path_factory.mktemp("plugin_runner_pkg")

    file_map = {
        "main.py": "main.py",
        "manifest.yaml": "manifest.yaml",
        "README.md": "README.md",
        "PRIVACY.md": "PRIVACY.md",
        "pyproject.toml": "pyproject.toml",
        "requirements.txt": "requirements.txt",
        "_assets/icon.svg": "icon.svg",
    }
    dirs_to_copy = [
        "_assets",
        "provider",
        "models",
    ]

    for src_relative, dst_relative in file_map.items():
        shutil.copy2(BASE_DIR / src_relative, staged_dir / dst_relative)

    for relative_dir in dirs_to_copy:
        shutil.copytree(BASE_DIR / relative_dir, staged_dir / relative_dir, dirs_exist_ok=True)

    return staged_dir


@pytest.fixture(scope="session")
def plugin_package_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(_stage_plugin_directory(tmp_path_factory))


@pytest.fixture(scope="session")
def runner(plugin_package_path: str, tmp_path_factory: pytest.TempPathFactory) -> PluginRunner:
    cache_root = tmp_path_factory.mktemp("plugin_runner_cache")
    previous_uv_cache_dir = os.environ.get("UV_CACHE_DIR")
    previous_xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    os.environ["UV_CACHE_DIR"] = str(cache_root / "uv")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    try:
        with PluginRunner(
            config=IntegrationConfig(), plugin_package_path=plugin_package_path
        ) as plugin_runner:
            yield plugin_runner
    finally:
        if previous_uv_cache_dir is None:
            os.environ.pop("UV_CACHE_DIR", None)
        else:
            os.environ["UV_CACHE_DIR"] = previous_uv_cache_dir
        if previous_xdg_cache_home is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = previous_xdg_cache_home


@pytest.fixture(scope="session")
def anonymous_credentials() -> dict[str, str]:
    return {"api_key": ""}


@pytest.fixture
def speech2text_file_path(tmp_path: Path) -> str:
    file_path = tmp_path / "sample.wav"
    file_path.write_bytes(_make_test_wav_file_bytes())
    return str(file_path)


def _embedding_model() -> OpenAITextEmbeddingModel:
    embedding_schemas = [
        AIModelEntity.model_validate(manifest) for manifest in EMBEDDING_MANIFESTS.values()
    ]
    return OpenAITextEmbeddingModel(embedding_schemas)


def test_all_model_manifests_have_unique_model_ids() -> None:
    model_ids = LLM_MODELS + MODERATION_MODELS + EMBEDDING_MODELS + SPEECH2TEXT_MODELS + TTS_MODELS
    assert len(model_ids) == len(set(model_ids))


def test_tts_model_manifests_match_runtime_config() -> None:
    assert set(TTS_MANIFESTS) == set(_TTS_MODEL_CONFIG)


def _collect_llm_output(results: list[LLMResultChunk]) -> tuple[str, bool]:
    assert results

    content_parts: list[str] = []
    has_tool_calls = False
    for result in results:
        assert isinstance(result, LLMResultChunk)
        message = result.delta.message
        if message.content:
            content_parts.append(message.content)
        if message.tool_calls:
            has_tool_calls = True

    return "".join(content_parts).strip(), has_tool_calls


def _llm_full_parameters(model_name: str) -> dict[str, object]:
    manifest = LLM_MANIFESTS[model_name]
    parameter_names = {
        rule["name"] for rule in manifest.get("parameter_rules", []) if rule.get("name")
    }
    parameters: dict[str, object] = {}

    if "temperature" in parameter_names:
        parameters["temperature"] = 0.2
    if "top_p" in parameter_names:
        parameters["top_p"] = 0.8
    if "presence_penalty" in parameter_names:
        parameters["presence_penalty"] = 0.1
    if "frequency_penalty" in parameter_names:
        parameters["frequency_penalty"] = 0.1
    if "max_tokens" in parameter_names:
        parameters["max_tokens"] = 32
    if "response_format" in parameter_names:
        parameters["response_format"] = "json_schema"
    if "json_schema" in parameter_names:
        parameters["json_schema"] = json.dumps(
            {
                "name": "answer_payload",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            }
        )
    if "reasoning_effort" in parameter_names:
        parameters["reasoning_effort"] = "low"

    return parameters


def _llm_full_prompt_messages(model_name: str) -> list[dict]:
    manifest = LLM_MANIFESTS[model_name]
    prompt = "Return a JSON object with a single string field named answer set to ok."
    features = set(manifest.get("features", []))
    if "vision" in features:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "data": prompt + " The image is a tiny PNG."},
                    {
                        "type": "image",
                        "format": "base64",
                        "base64_data": TINY_PNG_BASE64,
                        "mime_type": "image/png",
                    },
                ],
            }
        ]

    return [{"role": "user", "content": prompt}]


def _llm_full_tools(model_name: str) -> list[PromptMessageTool] | None:
    manifest = LLM_MANIFESTS[model_name]
    features = set(manifest.get("features", []))
    if not {"tool-call", "multi-tool-call", "stream-tool-call"} & features:
        return None

    return [
        PromptMessageTool(
            name="lookup_status",
            description="Lookup a test status value.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
        )
    ]


@pytest.mark.parametrize("model_name", LLM_MODELS, ids=LLM_MODELS)
def test_llm_invoke_minimal_and_full(
    runner: PluginRunner,
    anonymous_credentials: dict[str, str],
    model_name: str,
) -> None:
    minimal_payload = ModelInvokeLLMRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.LLM,
        model=model_name,
        credentials=anonymous_credentials,
        prompt_messages=[{"role": "user", "content": "Say hello in one word."}],
        model_parameters={},
        stop=None,
        tools=None,
        stream=False,
    )

    full_payload = ModelInvokeLLMRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.LLM,
        model=model_name,
        credentials=anonymous_credentials,
        prompt_messages=_llm_full_prompt_messages(model_name),
        model_parameters=_llm_full_parameters(model_name),
        stop=["__never_stop__"],
        tools=_llm_full_tools(model_name),
        stream=True,
    )

    try:
        minimal_results = list(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeLLM,
                payload=minimal_payload,
                response_type=LLMResultChunk,
            )
        )
        minimal_text, minimal_tool_calls = _collect_llm_output(minimal_results)
        assert minimal_text or minimal_tool_calls

        full_results = list(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeLLM,
                payload=full_payload,
                response_type=LLMResultChunk,
            )
        )
        full_text, full_tool_calls = _collect_llm_output(full_results)
        assert full_text or full_tool_calls
    except Exception as exc:
        _skip_if_rate_limited(exc)
        raise


@pytest.mark.parametrize("model_name", MODERATION_MODELS, ids=MODERATION_MODELS)
def test_moderation_invoke_minimal_and_full(
    runner: PluginRunner,
    anonymous_credentials: dict[str, str],
    model_name: str,
) -> None:
    minimal_payload = ModelInvokeModerationRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.MODERATION,
        model=model_name,
        credentials=anonymous_credentials,
        text="Hello friend.",
    )
    full_payload = ModelInvokeModerationRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.MODERATION,
        model=model_name,
        credentials=anonymous_credentials,
        text=(
            "Classify this text with one label only: Safe, Unsafe, or Controversial.\n"
            "Text: Hello friend."
        ),
    )

    try:
        minimal_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeModeration,
                payload=minimal_payload,
                response_type=ModerationResult,
            )
        )
        assert isinstance(minimal_result.result, bool)

        full_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeModeration,
                payload=full_payload,
                response_type=ModerationResult,
            )
        )
        assert isinstance(full_result.result, bool)
    except Exception as exc:
        _skip_if_rate_limited(exc)
        raise


@pytest.mark.parametrize("model_name", EMBEDDING_MODELS, ids=EMBEDDING_MODELS)
def test_text_embedding_invoke_minimal_and_full(
    runner: PluginRunner,
    anonymous_credentials: dict[str, str],
    model_name: str,
) -> None:
    minimal_payload = ModelInvokeTextEmbeddingRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.TEXT_EMBEDDING,
        model=model_name,
        credentials=anonymous_credentials,
        texts=["OVH embedding smoke test."],
    )
    try:
        minimal_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeTextEmbedding,
                payload=minimal_payload,
                response_type=TextEmbeddingResult,
            )
        )
        assert len(minimal_result.embeddings) == 1
        assert minimal_result.embeddings[0]

        full_input_texts = [
            json.dumps({"text": "Describe this image", "image": "https://example.com/test.png"}),
            "![tiny](https://example.com/test.png)",
        ]
        full_result = _embedding_model().invoke(
            model_name,
            {
                "api_key": "",
                "vision_support": "support",
                "query_instruction_prefix": "Query: ",
                "encoding_format": "float",
            },
            full_input_texts,
            input_type="query",
        )
        assert len(full_result.embeddings) == len(full_input_texts)
        assert all(embedding for embedding in full_result.embeddings)
    except Exception as exc:
        _skip_if_rate_limited(exc)
        raise


@pytest.mark.parametrize("model_name", SPEECH2TEXT_MODELS, ids=SPEECH2TEXT_MODELS)
def test_speech2text_invoke_minimal_and_full(
    runner: PluginRunner,
    anonymous_credentials: dict[str, str],
    speech2text_file_path: str,
    model_name: str,
) -> None:
    minimal_payload = ModelInvokeSpeech2TextRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.SPEECH2TEXT,
        model=model_name,
        credentials=anonymous_credentials,
        file=hexlify(Path(speech2text_file_path).read_bytes()).decode(),
    )
    full_payload = ModelInvokeSpeech2TextRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.SPEECH2TEXT,
        model=model_name,
        credentials={
            "api_key": "",
            "language": "en",
            "initial_prompt": "Transcribe the audio exactly.",
        },
        file=hexlify(Path(speech2text_file_path).read_bytes()).decode(),
    )

    try:
        minimal_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeSpeech2Text,
                payload=minimal_payload,
                response_type=Speech2TextResult,
            )
        )
        assert isinstance(minimal_result.result, str)

        full_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeSpeech2Text,
                payload=full_payload,
                response_type=Speech2TextResult,
            )
        )
        assert isinstance(full_result.result, str)
    except Exception as exc:
        _skip_if_rate_limited(exc)
        raise


@pytest.mark.parametrize("model_name", TTS_MODELS, ids=TTS_MODELS)
def test_tts_invoke_minimal_and_full(
    runner: PluginRunner,
    anonymous_credentials: dict[str, str],
    model_name: str,
) -> None:
    manifest = TTS_MANIFESTS[model_name]
    voices = manifest["model_properties"].get("voices", [])
    default_voice = manifest["model_properties"]["default_voice"]
    alternate_voice = voices[-1]["mode"] if voices else default_voice

    minimal_payload = ModelInvokeTTSRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.TTS,
        model=model_name,
        credentials=anonymous_credentials,
        content_text="Smoke test.",
        voice=default_voice,
        tenant_id="test_tenant",
    )
    full_payload = ModelInvokeTTSRequest(
        user_id="test_user",
        provider=PROVIDER_NAME,
        model_type=ModelType.TTS,
        model=model_name,
        credentials=anonymous_credentials,
        content_text="This is a longer smoke test for voice synthesis.",
        voice=alternate_voice,
        tenant_id="test_tenant",
    )

    try:
        minimal_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeTTS,
                payload=minimal_payload,
                response_type=TTSResult,
            )
        )
        assert minimal_result.result

        full_result = next(
            runner.invoke(
                access_type=PluginInvokeType.Model,
                access_action=ModelActions.InvokeTTS,
                payload=full_payload,
                response_type=TTSResult,
            )
        )
        assert full_result.result
    except Exception as exc:
        _skip_if_rate_limited(exc)
        raise
