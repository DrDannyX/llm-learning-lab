"""Polite, retrying HTTP with an on-disk cache.

The corpus build hits two public APIs a few thousand times. Caching raw
responses means you can re-run the labeller a hundred times while iterating on
extraction rules without ever touching the network again -- which is the part
of the workflow you will actually iterate on.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

UA = "geo-sft/0.1 (educational SFT project; contact: local user)"


class CachedClient:
    def __init__(self, cache_dir: Path, delay: float = 0.15, timeout: float = 30.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._client = httpx.Client(
            timeout=timeout, headers={"User-Agent": UA}, follow_redirects=True
        )
        self.hits = 0
        self.misses = 0

    def _key(self, url: str, params: dict | None) -> Path:
        blob = url + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / f"{hashlib.sha256(blob.encode()).hexdigest()[:24]}.json"

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _get(self, url: str, params: dict | None) -> Any:
        r = self._client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def get_json(self, url: str, params: dict | None = None) -> Any:
        path = self._key(url, params)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        data = self._get(url, params)
        # atomic write: concurrent workers must never read a half-written file
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        self.misses += 1
        time.sleep(self.delay)  # be a good citizen on free public APIs
        return data

    def close(self) -> None:
        self._client.close()
