from typing import Optional


def format_ovh_rate_limit_error(status_code: int, body: str, api_key: Optional[str] = None) -> str:
    if status_code != 429:
        return body[:1000]

    if api_key:
        return (
            "OVH AI Endpoints rate limit exceeded (429 Too Many Requests). "
            "Retry later or increase your OVH quota/project limits."
        )

    return (
        "OVH AI Endpoints anonymous rate limit exceeded (429 Too Many Requests). "
        "Anonymous access is limited to 2 requests per minute, per IP and per model. "
        "Retry later or configure an OVH API key for higher limits."
    )
