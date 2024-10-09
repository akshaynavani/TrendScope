"""Configuration loading: YAML defaults overridden by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # optional, only used for local convenience
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
SQL_DIR = PROJECT_ROOT / "sql"


@dataclass
class DataConfig:
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
    regions: list[str] = field(default_factory=lambda: ["NA", "EMEA", "APAC", "LATAM"])
    product_lines: list[str] = field(default_factory=lambda: ["Hardware", "Software", "Services"])
    channels: list[str] = field(default_factory=lambda: ["Direct", "Partner"])
    seed: int = 42


@dataclass
class ForecastConfig:
    frequency: str = "W"
    horizon: int = 26
    models: list[str] = field(default_factory=lambda: ["arima", "prophet", "seasonal_naive"])
    selection_metric: str = "smape"
    ensemble: bool = True
    interval_width: float = 0.8
    min_history: int = 52


@dataclass
class BacktestConfig:
    folds: int = 3
    horizon: int = 13
    step: int = 13


@dataclass
class SizingConfig:
    penetration_ceiling: float = 0.35
    growth_weight: float = 0.3
    headroom_weight: float = 0.5
    confidence_weight: float = 0.2


@dataclass
class DatabricksConfig:
    server_hostname: str | None = None
    http_path: str | None = None
    access_token: str | None = None
    catalog: str = "main"
    schema: str = "trendscope"


@dataclass
class Config:
    backend: str = "sqlite"
    sqlite_path: str = "data/trendscope.db"
    output_dir: str = "output"
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    databricks: DatabricksConfig = field(default_factory=DatabricksConfig)

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _section(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
    return cls(**known)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML, then apply environment variable overrides."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path) as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = Config(
        backend=raw.get("backend", "sqlite"),
        sqlite_path=raw.get("sqlite_path", "data/trendscope.db"),
        output_dir=raw.get("output_dir", "output"),
        data=_section(DataConfig, raw.get("data")),
        forecast=_section(ForecastConfig, raw.get("forecast")),
        backtest=_section(BacktestConfig, raw.get("backtest")),
        sizing=_section(SizingConfig, raw.get("sizing")),
        databricks=_section(DatabricksConfig, raw.get("databricks")),
    )

    env = os.environ
    cfg.backend = env.get("TRENDSCOPE_BACKEND", cfg.backend)
    cfg.sqlite_path = env.get("TRENDSCOPE_SQLITE_PATH", cfg.sqlite_path)
    cfg.output_dir = env.get("TRENDSCOPE_OUTPUT_DIR", cfg.output_dir)
    db = cfg.databricks
    db.server_hostname = env.get("DATABRICKS_SERVER_HOSTNAME", db.server_hostname)
    db.http_path = env.get("DATABRICKS_HTTP_PATH", db.http_path)
    db.access_token = env.get("DATABRICKS_TOKEN", db.access_token)
    db.catalog = env.get("DATABRICKS_CATALOG", db.catalog)
    db.schema = env.get("DATABRICKS_SCHEMA", db.schema)

    if cfg.forecast.frequency not in {"D", "W", "M"}:
        raise ValueError(f"Unsupported frequency: {cfg.forecast.frequency}")
    return cfg
