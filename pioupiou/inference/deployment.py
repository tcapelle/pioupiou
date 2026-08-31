"""Install the exact model bundle selected for live inference."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from pioupiou.inference.model import sha256_file


def load_deployment_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if set(manifest) != {"schema_version", "release", "url", "sha256"}:
        raise ValueError("Invalid deployment model manifest keys")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported deployment model manifest schema")
    if not all(
        isinstance(manifest[name], str) and manifest[name]
        for name in ("release", "url", "sha256")
    ):
        raise ValueError("Invalid deployment model manifest values")
    if len(manifest["sha256"]) != 64:
        raise ValueError("Invalid deployment model SHA-256")
    return manifest


def ensure_deployment_model(model: Path, manifest_path: Path) -> str:
    """Atomically install the pinned model when the local bundle is stale."""
    manifest = load_deployment_manifest(manifest_path)
    expected_sha256 = manifest["sha256"]
    if model.exists() and sha256_file(model) == expected_sha256:
        return expected_sha256

    model.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            manifest["url"], headers={"User-Agent": "pioupiou-model-installer"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=model.parent, prefix=f".{model.name}.", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Downloaded deployment model SHA-256 does not match")
        os.replace(temporary_path, model)
        temporary_path = None
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not install deployment model: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return expected_sha256
