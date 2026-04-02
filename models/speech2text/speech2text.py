import logging
from typing import IO, Optional
from urllib.parse import urljoin

import requests
from dify_plugin.entities.model import AIModelEntity, FetchFrom, I18nObject, ModelType
from dify_plugin.errors.model import InvokeBadRequestError, InvokeError, InvokeServerUnavailableError
from dify_plugin.interfaces.model.openai_compatible.speech2text import OAICompatSpeech2TextModel
from models.ovh_credentials import build_ovh_credentials

logger = logging.getLogger(__name__)


class OpenAISpeech2TextModel(OAICompatSpeech2TextModel):
    _INVOKE_TIMEOUT = (10, 120)

    def _invoke(self, model: str, credentials: dict, file: IO[bytes], user: Optional[str] = None) -> str:
        """
        Invoke speech2text model

        :param model: model name
        :param credentials: model credentials
        :param file: audio file
        :param user: unique user id
        :return: text for given audio file
        """
        credentials = build_ovh_credentials(credentials)
        headers = {}

        api_key = credentials.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        endpoint_url = credentials.get("endpoint_url", "https://api.openai.com/v1/")
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        endpoint_url = urljoin(endpoint_url, "audio/transcriptions")

        language = credentials.get("language", "en")
        prompt = credentials.get("initial_prompt", "convert the audio to text")
        payload = {"model": credentials.get("endpoint_model_name", model), "language": language, "prompt": prompt}
        files = [("file", file)]
        try:
            response = requests.post(
                endpoint_url, headers=headers, data=payload, files=files, timeout=self._INVOKE_TIMEOUT
            )
        except requests.exceptions.Timeout as ex:
            logger.error("STT request timed out for endpoint %s", endpoint_url)
            raise InvokeServerUnavailableError(
                f"Speech-to-text request timed out while calling {endpoint_url}."
            ) from ex
        except requests.exceptions.RequestException as ex:
            logger.error("STT request failed for endpoint %s: %s", endpoint_url, ex)
            raise InvokeServerUnavailableError(
                f"Speech-to-text request failed while calling {endpoint_url}: {ex!s}"
            ) from ex

        if response.status_code != 200:
            error_body = response.text[:1000]
            logger.error("STT API error %s: %s", response.status_code, error_body)
            raise InvokeBadRequestError(
                f"Speech-to-text API returned status {response.status_code}: {error_body}"
            )

        try:
            response_data = response.json()
        except ValueError as ex:
            logger.error("STT API returned invalid JSON payload.")
            raise InvokeError("Speech-to-text API returned an invalid JSON response.") from ex

        text = response_data.get("text")
        if not isinstance(text, str):
            raise InvokeError("Speech-to-text API response does not contain a valid 'text' field.")

        return text

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        """
        used to define customizable model schema
        """
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_type=ModelType.SPEECH2TEXT,
            model_properties={},
            parameter_rules=[],
        )

        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(en_US=credentials["display_name"])

        return entity
