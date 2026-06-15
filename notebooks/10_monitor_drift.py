# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Monitor Drift

# COMMAND ----------

from datetime import datetime, timezone
import uuid
import numpy as np
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_features_uc"
DRIFT_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_drift_metrics"

df = spark.table(FEATURE_TABLE)

numeric_features = [
    c for c, t in df.dtypes
    if t in ("int", "bigint", "double", "float", "long")
    and c not in ("loan_status",)
]

RUN_CREATED_AT = datetime.now(timezone.utc).isoformat()

def calculate_psi(expected, actual, buckets=10):
    expected = np.array([x for x in expected if x is not None and not np.isnan(x)])
    actual = np.array([x for x in actual if x is not None and not np.isnan(x)])

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

df_hash = df.withColumn("sample_bucket", F.abs(F.hash(F.col("loan_id"))) % 10)
baseline_df = df_hash.filter(F.col("sample_bucket") < 7)
current_df = df_hash.filter(F.col("sample_bucket") >= 7)

rows = []

for feature in numeric_features:
    expected_values = [r[feature] for r in baseline_df.select(feature).collect()]
    actual_values = [r[feature] for r in current_df.select(feature).collect()]

    psi = calculate_psi(expected_values, actual_values)

    status = "OK"
    if psi >= 0.25:
        status = "DRIFT"
    elif psi >= 0.10:
        status = "WARNING"

    rows.append({
        "drift_metric_id": str(uuid.uuid4()),
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
        "feature_name": feature,
        "psi": float(psi),
        "drift_status": status,
        "created_at": RUN_CREATED_AT,
    })

result_df = spark.createDataFrame(rows)

(
    result_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(DRIFT_TABLE)
)

display(result_df)

ok_count = result_df.filter(F.col("drift_status") == "OK").count()
warning_count = result_df.filter(F.col("drift_status") == "WARNING").count()
drift_count = result_df.filter(F.col("drift_status") == "DRIFT").count()

dbutils.notebook.exit(f"DRIFT_MONITORING_COMPLETED: OK={ok_count}, WARNING={warning_count}, DRIFT={drift_count}")
