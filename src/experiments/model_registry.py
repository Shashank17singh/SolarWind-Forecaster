import json
import os
from datetime import datetime

DEFAULT_REGISTRY = "models_deploy/registry.json"


def _load_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {"models": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def register_model(
    name: str,
    version: str,
    artifact_path: str,
    registry_path: str = DEFAULT_REGISTRY,
    metrics: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    payload = _load_registry(registry_path)
    entry = {
        "name": name,
        "version": version,
        "artifact_path": artifact_path,
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
        "metadata": metadata or {},
    }
    payload.setdefault("models", []).append(entry)
    _write_registry(registry_path, payload)
    return entry


def list_models(registry_path: str = DEFAULT_REGISTRY) -> list[dict]:
    payload = _load_registry(registry_path)
    return payload.get("models", [])
