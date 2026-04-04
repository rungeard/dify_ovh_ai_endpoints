from collections.abc import Mapping
from urllib.parse import urljoin

import httpx
from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

from models.ovh_credentials import build_ovh_credentials
from models.ovh_errors import format_ovh_rate_limit_error


class OVHAIEndpointsProvider(ModelProvider):
    _VALIDATE_TIMEOUT = (10, 30)

    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        normalized_credentials = build_ovh_credentials(credentials)

        endpoint_url = str(normalized_credentials.get("endpoint_url") or "").rstrip("/")
        if not endpoint_url:
            raise CredentialsValidateFailedError("Missing endpoint URL after credential normalization.")

        api_key = str(normalized_credentials.get("api_key") or "").strip()
        if not api_key:
            # OVH supports anonymous inference on public endpoints with a low rate limit.
            # Skip provider-level validation here because `/v1/models` may not be publicly
            # accessible anonymously even when inference endpoints are.
            return

        models_url = urljoin(f"{endpoint_url}/", "models")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = httpx.get(models_url, headers=headers, timeout=self._VALIDATE_TIMEOUT)
        except httpx.TimeoutException as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation timed out while calling {models_url}."
            ) from ex
        except httpx.HTTPError as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation failed to reach OVH endpoint: {ex!s}"
            ) from ex

        if response.status_code == 401:
            raise CredentialsValidateFailedError("Invalid OVH API key (401 Unauthorized).")
        if response.status_code == 403:
            raise CredentialsValidateFailedError("API key does not have permission for OVH endpoint (403 Forbidden).")
        if response.status_code >= 400:
            raise CredentialsValidateFailedError(
                "Credentials validation failed with status "
                f"{response.status_code}: {format_ovh_rate_limit_error(response.status_code, response.text, api_key)}"
            )

        try:
            body = response.json()
        except ValueError as ex:
            raise CredentialsValidateFailedError(
                f"Credentials validation returned a non-JSON response from {models_url}."
            ) from ex

        if not isinstance(body, dict):
            raise CredentialsValidateFailedError("Credentials validation response has an unexpected payload format.")
