from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

DEFAULT_OVH_ENDPOINT_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"


def build_ovh_credentials(credentials: Mapping | dict | None) -> dict:
    """
    Normalize credentials for OVH AI Endpoints:
    - inject the default OVH OpenAI-compatible endpoint,
    - support a single API key.
    """
    normalized = dict(credentials or {})

    endpoint_url = str(normalized.get("endpoint_url") or DEFAULT_OVH_ENDPOINT_URL).strip()
    endpoint_url = endpoint_url.rstrip("/")
    if not endpoint_url.endswith("/v1"):
        endpoint_url = f"{endpoint_url}/v1"
    normalized["endpoint_url"] = endpoint_url

    normalized["api_key"] = str(normalized.get("api_key") or "").strip()

    # OAICompatLargeLanguageModel expects `credentials["mode"]` to exist.
    # Provider-level credentials only include `api_key`, so default to chat.
    mode = str(normalized.get("mode") or "").strip().lower()
    normalized["mode"] = mode or "chat"

    return normalized


def build_ovh_auth_headers(api_key: str | None, content_type: Optional[str] = "application/json") -> dict[str, str]:
    headers: dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type

    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers
