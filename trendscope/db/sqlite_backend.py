"""Local SQLite backend via SQLAlchemy — lets the full pipeline run without a cluster."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from trendscope.config import PROJECT_ROOT, Config
from trendscope.db.backend import SqlBackend


class SqliteBackend(SqlBackend):
    prefix = ""

    def __init__(self, engine: Engine):
        self.engine = engine

    @classmethod
    def from_config(cls, cfg: Config) -> "SqliteBackend":
        return cls.from_path(cfg.sqlite_path)

    @classmethod
    def from_path(cls, path: str | Path) -> "SqliteBackend":
        if str(path) == ":memory:":
            return cls(create_engine("sqlite://"))
        db_path = Path(path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(create_engine(f"sqlite:///{db_path}"))

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(sql), dict(params or {}))

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=dict(params or {}))

    def write_frame(self, df: pd.DataFrame, table: str, mode: str = "append") -> None:
        if mode not in {"append", "replace"}:
            raise ValueError("mode must be 'append' or 'replace'")
        out = df.copy()
        # SQLite has no native date type; store ISO strings so comparisons work.
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")
        with self.engine.begin() as conn:
            if mode == "replace" and self._exists(conn, table):
                conn.execute(text(f"DELETE FROM {table}"))
            out.to_sql(table, conn, if_exists="append", index=False, chunksize=5000)

    @staticmethod
    def _exists(conn, table: str) -> bool:
        row = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"), {"t": table}
        ).fetchone()
        return row is not None

    def close(self) -> None:
        self.engine.dispose()
