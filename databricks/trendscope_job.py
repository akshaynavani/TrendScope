# Databricks notebook source
# MAGIC %md
# MAGIC # TrendScope — scheduled forecasting job
# MAGIC Pulls segment sales from the SQL warehouse, forecasts each segment with Prophet/ARIMA,
# MAGIC sizes headroom and writes results back to `<catalog>.<schema>` tables. Dashboards are
# MAGIC saved to DBFS / a Unity Catalog volume.

# COMMAND ----------
# MAGIC %pip install -e /Workspace/Repos/trendscope[prophet,databricks]

# COMMAND ----------
import logging
import os

logging.basicConfig(level=logging.INFO)

# Job parameters (set as widgets or task parameters).
dbutils.widgets.text("catalog", "main")  # noqa: F821 - dbutils is injected by Databricks
dbutils.widgets.text("schema", "trendscope")  # noqa: F821
dbutils.widgets.text("frequency", "W")  # noqa: F821
dbutils.widgets.text("horizon", "26")  # noqa: F821
dbutils.widgets.text("output_dir", "/Volumes/main/trendscope/artifacts")  # noqa: F821
dbutils.widgets.text("secret_scope", "trendscope")  # noqa: F821

scope = dbutils.widgets.get("secret_scope")  # noqa: F821
os.environ["TRENDSCOPE_BACKEND"] = "databricks"
os.environ["DATABRICKS_CATALOG"] = dbutils.widgets.get("catalog")  # noqa: F821
os.environ["DATABRICKS_SCHEMA"] = dbutils.widgets.get("schema")  # noqa: F821
os.environ["TRENDSCOPE_OUTPUT_DIR"] = dbutils.widgets.get("output_dir")  # noqa: F821
# Credentials come from a secret scope, never from code.
os.environ["DATABRICKS_SERVER_HOSTNAME"] = dbutils.secrets.get(scope, "server_hostname")  # noqa: F821
os.environ["DATABRICKS_HTTP_PATH"] = dbutils.secrets.get(scope, "http_path")  # noqa: F821
os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(scope, "token")  # noqa: F821

# COMMAND ----------
from trendscope.config import load_config
from trendscope.pipeline import run_pipeline

cfg = load_config()
cfg.forecast.frequency = dbutils.widgets.get("frequency")  # noqa: F821
cfg.forecast.horizon = int(dbutils.widgets.get("horizon"))  # noqa: F821

result = run_pipeline(cfg, workers=max(1, (os.cpu_count() or 2) - 1))

# COMMAND ----------
display(result.headroom)  # noqa: F821
