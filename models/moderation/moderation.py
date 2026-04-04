import re
from typing import Mapping, Optional
from urllib.parse import urljoin

import httpx

from dify_plugin.entities.model import AIModelEntity, FetchFrom, I18nObject, ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeBadRequestError, InvokeServerUnavailableError
from dify_plugin.interfaces.model.moderation_model import ModerationModel
from models.ovh_credentials import build_ovh_credentials
from models.ovh_errors import format_ovh_rate_limit_error

_SAFETY_LABEL_PATTERN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)


class OVHModerationModel(ModerationModel):
    _TIMEOUT = (10, 120)

    def _invoke(self, model: str, credentials: dict, text: str, user: Optional[str] = None) -> bool:
        credentials = build_ovh_credentials(credentials)
        endpoint_url = credentials.get("endpoint_url", "").rstrip("/")
        endpoint_url = urljoin(f"{endpoint_url}/", "chat/completions")

        api_key = credentials.get("api_key")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": credentials.get("endpoint_model_name", model),
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 128,
            "temperature": 0,
            "stream": False,
        }

        try:
            response = httpx.post(endpoint_url, headers=headers, json=payload, timeout=self._TIMEOUT)
        except httpx.TimeoutException as ex:
            raise InvokeServerUnavailableError(f"OVH moderation request timed out: {ex!s}") from ex
        except httpx.HTTPError as ex:
            raise InvokeServerUnavailableError(f"OVH moderation request failed: {ex!s}") from ex

        if response.status_code != 200:
            raise InvokeBadRequestError(format_ovh_rate_limit_error(response.status_code, response.text, api_key))

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as ex:
            raise InvokeBadRequestError(f"Invalid OVH moderation response: {response.text[:1000]}") from ex

        match = _SAFETY_LABEL_PATTERN.search(content)
        if not match:
            raise InvokeBadRequestError(f"Unable to parse safety label from OVH moderation output: {content[:500]}")

        label = match.group(1).lower()
        # Dify ModerationModel returns False for safe, True for content to block.
        return label != "safe"

    def validate_credentials(self, model: str, credentials: Mapping | dict) -> None:
        try:
            self._invoke(model, dict(credentials), "Hello, how are you?")
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def get_customizable_model_schema(self, model: str, credentials: Mapping | dict) -> AIModelEntity | None:
        credentials = build_ovh_credentials(credentials)
        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            fetch_from=FetchFrom.PREDEFINED_MODEL,
            model_type=ModelType.MODERATION,
            model_properties={},
            parameter_rules=[],
        )

        if credentials.get("display_name"):
            entity.label = I18nObject(en_US=credentials["display_name"])

        return entity
