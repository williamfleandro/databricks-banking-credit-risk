# Databricks notebook source
# MAGIC %md
# MAGIC # 13 - Retraining Decision

# COMMAND ----------

from datetime import datetime, timezone
import uuid
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

DRIFT_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_drift_metrics"
RETRAINING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_retraining_requests"
MODEL_COMPARISON_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_model_comparison"

drift_df = spark.table(DRIFT_TABLE)
latest_ts = drift_df.agg(F.max("created_at").alias("latest")).collect()[0]["latest"]

latest_df = drift_df.filter(F.col("created_at") == latest_ts)

drift_features = [r["feature_name"] for r in latest_df.filter(F.col("drift_status") == "DRIFT").select("feature_name").collect()]
warning_features = [r["feature_name"] for r in latest_df.filter(F.col("drift_status") == "WARNING").select("feature_name").collect()]
max_psi = latest_df.agg(F.max("psi").alias("max_psi")).collect()[0]["max_psi"]

if drift_features:
    status = "RETRAINING_REQUIRED"
    action = "Open retraining request and require drift approval before production promotion."
elif warning_features:
    status = "RETRAINING_RECOMMENDED"
    action = "Monitor features and consider retraining."
else:
    status = "RETRAINING_NOT_REQUIRED"
    action = "Continue normal monitoring."

best_rows = (
    spark.table(MODEL_COMPARISON_TABLE)
    .filter(F.col("is_best_model") == True)
    .orderBy(F.col("created_at").desc())
    .limit(1)
    .collect()
)

best = best_rows[0] if best_rows else None

row = [{
    "retraining_request_id": str(uuid.uuid4()),
    "catalog_name": CATALOG_NAME,
    "schema_name": SCHEMA_NAME,
    "drift_run_created_at": str(latest_ts),
    "retraining_status": status,
    "retraining_reason": f"drift_features={drift_features}, warning_features={warning_features}",
    "drift_features": drift_features,
    "warning_features": warning_features,
    "max_psi": float(max_psi) if max_psi is not None else 0.0,
    "best_model_type": best["model_type"] if best else "",
    "best_model_f1_score": float(best["f1_score"]) if best else 0.0,
    "best_model_recall": float(best["recall"]) if best else 0.0,
    "best_model_precision": float(best["precision"]) if best else 0.0,
    "recommended_action": action,
    "created_at": datetime.now(timezone.utc).isoformat(),
}]

result_df = spark.createDataFrame(row)

(
    result_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(RETRAINING_TABLE)
)

display(result_df)
dbutils.notebook.exit(status)
