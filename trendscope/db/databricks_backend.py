"""Databricks SQL warehouse backend using ``databricks-sql-connector``.

Credentials are read from the environment (see ``.env.example``); nothing is hard-coded.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from trendscope.config import Config
from trendscope.db.backend import SqlBackend

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


class DatabricksBackend(SqlBackend):
    def __init__(self, connection, catalog: str, schema: str, batch_size: int = 500):
        self.conn = connection
        self.catalog = _check_ident(catalog)
        self.schema = _check_ident(schema)
        self.prefix = f"{self.catalog}.{self.schema}."
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, cfg: Config) -> "DatabricksBackend":
        try:
            from databricks import sql as dbsql
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "databricks-sql-connector is required: pip install 'trendscope[databricks]'"
            ) from exc

        db = cfg.databricks
        missing = [
            name
            for name, value in [
                ("DATABRICKS_SERVER_HOSTNAME", db.server_hostname),
                ("DATABRICKS_HTTP_PATH", db.http_path),
                ("DATABRICKS_TOKEN", db.access_token),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Databricks settings: {', '.join(missing)}")

        conn = dbsql.connect(
            server_hostname=db.server_hostname,
            http_path=db.http_path,
            access_token=db.access_token,
        )
        backend = cls(conn, db.catalog, db.schema)
        backend.execute(f"CREATE SCHEMA IF NOT EXISTS {backend.catalog}.{backend.schema}")
        return backend

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(sql, dict(params or {}))

    def query(self, sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
        with self.conn.cursor() as cur:
            cur.execute(sql, dict(params or {}))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
        df = pd.DataFrame([tuple(r) for r in rows], columns=cols)
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        return df

    def write_frame(self, df: pd.DataFrame, table: str, mode: str = "append") -> None:
        if mode not in {"append", "replace"}:
            raise ValueError("mode must be 'append' or 'replace'")
        target = table if "." in table else self.table(_check_ident(table))
        cols = [_check_ident(c) for c in df.columns]
        if mode == "replace":
            self.execute(f"DELETE FROM {target}")
        if df.empty:
            return

        records = df.replace({np.nan: None}).to_dict("records")
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            params: dict[str, Any] = {}
            value_rows = []
            for i, rec in enumerate(batch):
                placeholders = []
                for col in cols:
                    key = f"{col}_{i}"
                    val = rec[col]
                    if isinstance(val, pd.Timestamp):
                        val = val.date()
                    elif isinstance(val, np.generic):
                        val = val.item()
                    params[key] = val
                    placeholders.append(f":{key}")
                value_rows.append(f"({', '.join(placeholders)})")
            sql = f"INSERT INTO {target} ({', '.join(cols)}) VALUES {', '.join(value_rows)}"
            self.execute(sql, params)

    def close(self) -> None:
        self.conn.close()
