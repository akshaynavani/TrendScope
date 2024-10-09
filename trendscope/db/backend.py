"""SQL backend abstraction shared by the SQLite (local) and Databricks implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import pandas as pd

from trendscope.config import SQL_DIR

if TYPE_CHECKING:
    from trendscope.config import Config


def load_sql(name: str) -> str:
    """Read a .sql file from the project's sql/ directory."""
    path = SQL_DIR / (name if name.endswith(".sql") else f"{name}.sql")
    return path.read_text()


def split_statements(script: str) -> list[str]:
    """Split a SQL script into statements, dropping comment-only chunks."""
    statements = []
    for chunk in script.split(";"):
        lines = [ln for ln in chunk.splitlines() if not ln.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


class SqlBackend(ABC):
    """Minimal interface the pipeline needs from a SQL engine."""

    #: Table name prefix substituted for ``{prefix}`` in SQL files.
    prefix: str = ""

    def render(self, sql: str) -> str:
        return sql.replace("{prefix}", self.prefix)

    def table(self, name: str) -> str:
        return f"{self.prefix}{name}"

    @abstractmethod
    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        """Run a single statement that returns no rows."""

    @abstractmethod
    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Run a query and return the result as a DataFrame."""

    @abstractmethod
    def write_frame(self, df: pd.DataFrame, table: str, mode: str = "append") -> None:
        """Write a DataFrame to ``table``. ``mode`` is "append" or "replace"."""

    def execute_script(self, script: str) -> None:
        for stmt in split_statements(self.render(script)):
            self.execute(stmt)

    def run_file(self, name: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
        return self.query(self.render(load_sql(name)), params)

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_backend(cfg: "Config") -> SqlBackend:
    """Instantiate the backend configured in ``cfg.backend``."""
    if cfg.backend == "sqlite":
        from trendscope.db.sqlite_backend import SqliteBackend

        return SqliteBackend.from_config(cfg)
    if cfg.backend == "databricks":
        from trendscope.db.databricks_backend import DatabricksBackend

        return DatabricksBackend.from_config(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend!r}")
