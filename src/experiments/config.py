import os
from dataclasses import dataclass


def _getenv(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def _getenv_int(key: str, default: int) -> int:
    value = _getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _getenv_float(key: str, default: float) -> float:
    value = _getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _getenv_bool(key: str, default: bool = False) -> bool:
    value = _getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    api_key: str | None
    require_api_key: bool
    allowed_origins: list[str]
    rate_limit_rps: float
    rate_limit_burst: int
    cache_ttl_sec: int
    request_timeout_sec: int
    log_dir: str
    feature_spec_path: str
    drift_baseline_path: str
    quality_window_min: int


def load_config() -> AppConfig:
    allowed_raw = _getenv("API_ALLOWED_ORIGINS", "*")
    allowed = [o.strip() for o in allowed_raw.split(",") if o.strip()]
    return AppConfig(
        api_key=_getenv("API_KEY"),
        require_api_key=_getenv_bool("REQUIRE_API_KEY", False),
        allowed_origins=allowed,
        rate_limit_rps=_getenv_float("API_RATE_LIMIT_RPS", 5.0),
        rate_limit_burst=_getenv_int("API_RATE_LIMIT_BURST", 20),
        cache_ttl_sec=_getenv_int("API_CACHE_TTL_SEC", 30),
        request_timeout_sec=_getenv_int("API_TIMEOUT_SEC", 20),
        log_dir=_getenv("LOG_DIR", "logs") or "logs",
        feature_spec_path=_getenv("FEATURE_SPEC_PATH", "configs/feature_spec.json")
        or "configs/feature_spec.json",
        drift_baseline_path=_getenv(
            "DRIFT_BASELINE_PATH", "configs/feature_baseline.json"
        )
        or "configs/feature_baseline.json",
        quality_window_min=_getenv_int("QUALITY_WINDOW_MIN", 360),
    )


CONFIG = load_config()
