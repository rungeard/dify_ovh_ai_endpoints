from typing import Optional


def format_ovh_rate_limit_error(status_code: int, body: str, api_key: Optional[str] = None) -> str:
    if status_code != 429:
        return body[:1000]

    return (
        "OVH AI Endpoints rate limit exceeded (429 Too Many Requests). "
        "Retry later or increase your OVH quota/project limits."
    )
