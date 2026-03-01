from __future__ import annotations
"""
config_loader.py
----------------
Loads YAML config. Supports ${ENV_VAR} substitution so API keys
can stay out of version control when needed.

Usage:
    from src.config_loader import load_config
    cfg = load_config()                          # default path
    cfg = load_config("config/config.yaml")      # explicit path
    api_key = cfg["data"]["fred"]["api_key"]
"""

import os
import re
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config/config.yaml") -> dict:
    """
    Load and return the project config as a plain dict.

    Env-var substitution: any ${VAR_NAME} in the YAML is replaced
    with os.environ["VAR_NAME"] at load time.  This lets you keep
    secrets out of the file:

        fred:
          api_key: ${FRED_API_KEY}

    then set FRED_API_KEY in your shell / .env file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. "
            "Copy config/config.template.yaml to config/config.yaml and fill in your keys."
        )

    raw = config_path.read_text(encoding="utf-8")
    raw = _substitute_env_vars(raw)
    cfg = yaml.safe_load(raw)

    _ensure_output_dirs(cfg)
    logger.info(f"Config loaded: {path}  (project='{cfg['project']['name']}')")
    return cfg


# ── helpers ────────────────────────────────────────────────────────────────


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} placeholders with environment variable values."""

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise EnvironmentError(
                f"Config references ${{{var}}} but that environment variable is not set."
            )
        return val

    return re.sub(r"\$\{(\w+)\}", _replace, text)


def _ensure_output_dirs(cfg: dict) -> None:
    """Pre-create all output directories declared in config."""
    out = cfg.get("outputs", {})
    for key in ("data_cache_dir", "processed_dir", "figures_dir", "reports_dir", "models_dir"):
        dir_path = out.get(key)
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
