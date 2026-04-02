import logging
from collections.abc import Mapping
from urllib.parse import urljoin

import requests
from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

from models.ovh_credentials import build_ovh_credentials

logger = logging.getLogger(__name__)


class OVHAIEndpointsProvider(ModelProvider):
    _VALIDATE_TIMEOUT = (10, 30)

    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        normalized_credentials = build_ovh_credentials(credentials)

        api_key = str(normalized_credentials.get("api_key") or "").strip()
        if not api_key:
            raise CredentialsValidateFailedError("Missing required credential: api_key.")

        endpoint_url = str(normalized_credentials.get("endpoint_url") or "").rstrip("/")
        if not endpoint_url:
            raise CredentialsValidateFailedError("Missing endpoint URL after credential normalization.")

        models_url = urljoin(f"{endpoint_url}/", "models")
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            response = requests.get(models_url, headers=headers, timeout=self._VALIDATE_TIMEOUT)
        except requests.exceptions.Timeout as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation timed out while calling {models_url}."
            ) from ex
        except requests.exceptions.RequestException as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation failed to reach OVH endpoint: {ex!s}"
            ) from ex

        if response.status_code == 401:
            raise CredentialsValidateFailedError("Invalid OVH API key (401 Unauthorized).")
        if response.status_code == 403:
            raise CredentialsValidateFailedError("API key does not have permission for OVH endpoint (403 Forbidden).")
        if response.status_code >= 400:
            raise CredentialsValidateFailedError(
                f"Credentials validation failed with status {response.status_code}: {response.text[:500]}"
            )

        try:
            body = response.json()
        except ValueError as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation returned a non-JSON response from {models_url}."
            ) from ex

        if not isinstance(body, dict):
            raise CredentialsValidateFailedError("Credentials validation response has an unexpected payload format.")
