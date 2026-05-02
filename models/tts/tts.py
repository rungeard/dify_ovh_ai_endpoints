from collections.abc import Generator
from io import BytesIO
from typing import Any, Mapping, Optional

import httpx
from pydub import AudioSegment

from dify_plugin import TTSModel
from dify_plugin.entities.model import AIModelEntity, I18nObject, ModelPropertyKey
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeBadRequestError, InvokeError, InvokeServerUnavailableError
from models.ovh_errors import format_ovh_rate_limit_error
from models.ovh_credentials import build_ovh_auth_headers, build_ovh_credentials

_TTS_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "nvr-tts-en-us": {
        "language_code": "en-US",
        "sample_rate_hz": 16000,
        "encoding": 1,
        "endpoint": "https://nvr-tts-en-us.endpoints.kepler.ai.cloud.ovh.net/api/v1/tts/text_to_audio",
    },
    "nvr-tts-es-es": {
        "language_code": "es-ES",
        "sample_rate_hz": 16000,
        "encoding": 1,
        "endpoint": "https://nvr-tts-es-es.endpoints.kepler.ai.cloud.ovh.net/api/v1/tts/text_to_audio",
    },
    "nvr-tts-de-de": {
        "language_code": "de-DE",
        "sample_rate_hz": 16000,
        "encoding": 1,
        "endpoint": "https://nvr-tts-de-de.endpoints.kepler.ai.cloud.ovh.net/api/v1/tts/text_to_audio",
    },
    "nvr-tts-it-it": {
        "language_code": "it-IT",
        "sample_rate_hz": 16000,
        "encoding": 1,
        "endpoint": "https://nvr-tts-it-it.endpoints.kepler.ai.cloud.ovh.net/api/v1/tts/text_to_audio",
    },
}


class OpenAIText2SpeechModel(TTSModel):
    @staticmethod
    def _convert_wav_to_mp3(audio_bytes: bytes) -> bytes:
        try:
            input_buffer = BytesIO(audio_bytes)
            output_buffer = BytesIO()
            segment = AudioSegment.from_file(input_buffer, format="wav")
            segment.export(output_buffer, format="mp3")
            output_buffer.seek(0)
            return output_buffer.read()
        except Exception as ex:
            raise InvokeBadRequestError(f"pydub failed to convert OVH TTS WAV to MP3: {ex!s}") from ex

    @staticmethod
    def _get_tts_model_config(model: str, credentials: dict) -> dict[str, Any]:
        endpoint_model = credentials.get("endpoint_model_name", model)
        if endpoint_model in _TTS_MODEL_CONFIG:
            return _TTS_MODEL_CONFIG[endpoint_model]
        if model in _TTS_MODEL_CONFIG:
            return _TTS_MODEL_CONFIG[model]
        raise InvokeBadRequestError(f"Unsupported OVH TTS model: {endpoint_model}")

    def _normalize_voice(self, model: str, credentials: dict, voice: str) -> str:
        available_voices = self.get_tts_model_voices(model, credentials) or []
        allowed_values = {item.get("value") for item in available_voices if item.get("value")}
        default_voice = self._get_model_default_voice(model, credentials)

        if voice in allowed_values:
            return voice

        return default_voice or voice

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Any:
        normalized_credentials = build_ovh_credentials(credentials)
        if not voice or voice not in [d["value"] for d in self.get_tts_model_voices(model=model, credentials=normalized_credentials)]:
            voice = self._get_model_default_voice(model, normalized_credentials)
        return self._tts_invoke(model=model, credentials=normalized_credentials, content_text=content_text, voice=voice)

    def validate_credentials(self, model: str, credentials: dict, user: Optional[str] = None) -> None:
        normalized_credentials = build_ovh_credentials(credentials)
        try:
            next(self._tts_invoke(model=model, credentials=normalized_credentials, content_text="Test.", voice=self._get_model_default_voice(model, normalized_credentials)))
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def _tts_invoke(self, model: str, credentials: dict, content_text: str, voice: str) -> Generator[bytes, None, None]:
        tts_config = self._get_tts_model_config(model, credentials)
        endpoint_url = tts_config["endpoint"]
        word_limit = self._get_model_word_limit(model, credentials)
        sentences = list(self._split_text_into_sentences(content_text, word_limit or 2000))

        headers = {
            "accept": "application/octet-stream",
            **build_ovh_auth_headers(credentials.get("api_key")),
        }
        api_key = credentials.get("api_key")

        try:
            with httpx.Client(timeout=120.0) as client:
                for index, sentence in enumerate(sentences, start=1):
                    payload = {
                        "encoding": tts_config["encoding"],
                        "language_code": tts_config["language_code"],
                        "sample_rate_hz": tts_config["sample_rate_hz"],
                        "text": sentence,
                        "voice_name": voice,
                    }
                    response = client.post(endpoint_url, headers=headers, json=payload)
                    if response.status_code != 200:
                        body = format_ovh_rate_limit_error(response.status_code, response.text, api_key)
                        raise InvokeBadRequestError(body)

                    audio_bytes = response.content
                    converted_audio = self._convert_wav_to_mp3(audio_bytes)
                    yield converted_audio
        except InvokeBadRequestError:
            raise
        except httpx.TimeoutException as ex:
            raise InvokeServerUnavailableError(str(ex)) from ex
        except Exception as ex:
            raise InvokeBadRequestError(str(ex)) from ex

    def get_customizable_model_schema(
        self, model: str, credentials: Mapping | dict
    ) -> AIModelEntity:
        credentials = build_ovh_credentials(credentials)
        entity = super().get_customizable_model_schema(model, credentials)

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(en_US=credentials["display_name"])

        return entity

    def get_tts_model_voices(self, model: str, credentials: dict, language: str | None = None) -> list | None:
        normalized_credentials = build_ovh_credentials(credentials)
        model_schema = self.get_model_schema(model, normalized_credentials)

        if model_schema and ModelPropertyKey.VOICES in model_schema.model_properties:
            voices = model_schema.model_properties[ModelPropertyKey.VOICES]
            if language:
                return [
                    {"name": voice["name"], "value": voice["mode"]}
                    for voice in voices
                    if language in voice.get("language", [])
                ]
            return [{"name": voice["name"], "value": voice["mode"]} for voice in voices]

        return super().get_tts_model_voices(model, normalized_credentials, language)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeServerUnavailableError: [Exception],
        }
