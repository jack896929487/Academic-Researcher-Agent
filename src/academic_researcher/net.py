"""Network environment helpers."""

from __future__ import annotations

import os
from typing import Dict

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_DEAD_LOCAL_PROXY_VALUES = {
    "http://127.0.0.1:9",
    "https://127.0.0.1:9",
    "socks5://127.0.0.1:9",
}


def sanitize_dead_local_proxies() -> Dict[str, str]:
    """
    Remove obviously broken localhost placeholder proxies from the environment.

    We only clear the exact dead-localhost proxy pattern that guarantees
    outbound failures. Normal proxy settings are left untouched.
    """
    cleared: Dict[str, str] = {}
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key)
        if not value:
            continue
        normalized = value.strip().lower().rstrip("/")
        if normalized in _DEAD_LOCAL_PROXY_VALUES:
            cleared[key] = value
            os.environ.pop(key, None)
    return cleared
