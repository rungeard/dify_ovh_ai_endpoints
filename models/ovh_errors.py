from http import HTTPStatus


def format_ovh_rate_limit_error(status_code: int, body: str) -> str:
    if status_code != HTTPStatus.TOO_MANY_REQUESTS:
        return body[:1000]

    return (
        "OVH AI Endpoints rate limit exceeded (429 Too Many Requests). "
        "Retry later or increase your OVH quota/project limits."
    )
