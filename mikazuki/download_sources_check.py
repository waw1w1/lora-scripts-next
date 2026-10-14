"""Connectivity probes for engine-install download sources.

The install runs on this (backend) machine, so the probes run here too: the
result reflects the network path the installer will actually use, not the
browser's. Fields omitted from the request probe the engine installer
defaults, mirroring how ``parse_download_sources`` → ``None`` keeps them.
"""

from __future__ import annotations

import asyncio
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

from mikazuki.download_sources import (
    DownloadSources,
    apply_github_prefix,
    pytorch_extra_index_url,
)

PROBE_TIMEOUT_SECONDS = 5.0

CHANNELS = ("pip", "pytorch", "huggingface", "github")

# Mirror the installer defaults (mikazuki/engines/*/environment.py). Kept
# local so probing does not import the engine runtime modules.
DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple"
DEFAULT_PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# Representative CUDA tag for the PyTorch probe: the question is whether the
# mirror serves per-CUDA index paths at all, not any single engine's tag.
PYTORCH_PROBE_CUDA_TAG = "cu128"

# Fixed repo used to probe GitHub (and proxy prefixes). Update if it moves.
GITHUB_PROBE_URL = "https://github.com/wochenlong/lora-scripts-next"


def probe_targets(sources: DownloadSources | None) -> dict[str, str]:
    """Resolve the per-channel probe URLs; omitted fields → installer defaults."""
    pip = (sources.pip_index_url if sources else None) or DEFAULT_PIP_INDEX_URL
    if not pip.endswith("/"):
        pip += "/"
    pytorch_base = (sources.pytorch_index_url if sources else None) or None
    pytorch = pytorch_extra_index_url(
        pytorch_base,
        PYTORCH_PROBE_CUDA_TAG,
        f"{DEFAULT_PYTORCH_INDEX_BASE}/{PYTORCH_PROBE_CUDA_TAG}",
    ) + "/"
    # Auto HF target: an operator-provided HF_ENDPOINT wins over the engine
    # default, mirroring how install_process_env() inherits os.environ.
    hf = (
        (sources.hf_endpoint if sources else None)
        or os.environ.get("HF_ENDPOINT")
        or DEFAULT_HF_ENDPOINT
    ).rstrip("/")
    github_prefix = (sources.github_url_prefix if sources else None) or ""
    return {
        "pip": pip,
        "pytorch": pytorch,
        "huggingface": f"{hf}/api/models?limit=1",
        "github": apply_github_prefix(GITHUB_PROBE_URL, github_prefix),
    }


def _probe(url: str, timeout: float = PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    result: dict[str, Any] = {"ok": False, "status": None, "latency_ms": None, "url": url, "error": None}
    if not url.startswith(("http://", "https://")):
        result["error"] = f"unsupported URL scheme: {url}"
        return result
    start = time.monotonic()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SD-Trainer"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result["status"] = resp.status
            resp.read(256)
        result["ok"] = True
    except urllib.error.HTTPError as exc:
        result["status"] = exc.code
        result["error"] = f"HTTP {exc.code}"
    except Exception as exc:
        result["error"] = str(exc)[:200]
    result["latency_ms"] = int((time.monotonic() - start) * 1000)
    return result


async def check_download_sources(
    sources: DownloadSources | None,
    channels: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Probe the requested channels (default: all four) concurrently.

    Each channel is independent; callers may also issue one request per
    channel so a hanging mirror never delays the others' results.
    """
    targets = probe_targets(sources)
    names = [name for name in (channels or CHANNELS) if name in targets]
    results = await asyncio.gather(*(asyncio.to_thread(_probe, targets[name]) for name in names))
    return dict(zip(names, results))
