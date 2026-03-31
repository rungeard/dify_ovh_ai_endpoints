from __future__ import annotations

from collections.abc import Mapping

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

    api_key = str(normalized.get("api_key") or "").strip()
    if api_key:
        normalized["api_key"] = api_key

    return normalized
