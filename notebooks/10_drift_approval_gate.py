# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Drift Approval Gate

# COMMAND ----------

from datetime import datetime, timezone
import uuid
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")
dbutils.widgets.text("approved_by", "")
dbutils.widgets.text("approval_comment", "")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")
APPROVED_BY = dbutils.widgets.get("approved_by").strip()
APPROVAL_COMMENT = dbutils.widgets.get("approval_comment").strip()

DRIFT_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_drift_metrics"
APPROVAL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_drift_approval_requests"

IS_PROD = CATALOG_NAME == "mlops_production"

drift_df = spark.table(DRIFT_TABLE)
latest_ts = drift_df.agg(F.max("created_at").alias("latest")).collect()[0]["latest"]

latest_df = drift_df.filter(F.col("created_at") == latest_ts)

drift_features = [r["feature_name"] for r in latest_df.filter(F.col("drift_status") == "DRIFT").select("feature_name").collect()]
warning_features = [r["feature_name"] for r in latest_df.filter(F.col("drift_status") == "WARNING").select("feature_name").collect()]

schema = StructType([
    StructField("approval_id", StringType(), False),
    StructField("catalog_name", StringType(), False),
    StructField("schema_name", StringType(), False),
    StructField("drift_run_created_at", StringType(), False),
    StructField("drift_features", ArrayType(StringType()), False),
    StructField("warning_features", ArrayType(StringType()), False),
    StructField("approval_status", StringType(), False),
    StructField("approved_by", StringType(), True),
    StructField("approval_comment", StringType(), True),
    StructField("requested_at", StringType(), False),
    StructField("approved_at", StringType(), True),
])

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {APPROVAL_TABLE} (
    approval_id STRING,
    catalog_name STRING,
    schema_name STRING,
    drift_run_created_at STRING,
    drift_features ARRAY<STRING>,
    warning_features ARRAY<STRING>,
    approval_status STRING,
    approved_by STRING,
    approval_comment STRING,
    requested_at STRING,
    approved_at STRING
)
USING DELTA
""")

def write_record(status, approved_by="", comment="", approved_at=""):
    row = [{
        "approval_id": str(uuid.uuid4()),
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
        "drift_run_created_at": str(latest_ts),
        "drift_features": drift_features,
        "warning_features": warning_features,
        "approval_status": status,
        "approved_by": approved_by,
        "approval_comment": comment,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "approved_at": approved_at,
    }]
    spark.createDataFrame(row, schema=schema).write.mode("append").option("mergeSchema", "true").saveAsTable(APPROVAL_TABLE)

if not drift_features:
    write_record("NOT_REQUIRED", comment="No severe drift detected. Approval not required.")
    dbutils.notebook.exit("DRIFT_APPROVAL_NOT_REQUIRED")

existing_approval = (
    spark.table(APPROVAL_TABLE)
    .filter(F.col("catalog_name") == CATALOG_NAME)
    .filter(F.col("schema_name") == SCHEMA_NAME)
    .filter(F.col("drift_run_created_at") == str(latest_ts))
    .filter(F.col("approval_status") == "APPROVED")
)

if existing_approval.count() > 0:
    dbutils.notebook.exit("DRIFT_APPROVAL_FOUND")

if APPROVED_BY:
    write_record(
        "APPROVED",
        approved_by=APPROVED_BY,
        comment=APPROVAL_COMMENT or "Drift manually approved through notebook parameters.",
        approved_at=datetime.now(timezone.utc).isoformat(),
    )
    dbutils.notebook.exit("DRIFT_APPROVED")

write_record(
    "PENDING_APPROVAL",
    comment="Severe drift detected. Manual approval is required before production promotion.",
)

if IS_PROD:
    raise Exception("Production drift approval required. Blocking automatic model promotion.")

dbutils.notebook.exit("DRIFT_APPROVAL_REQUIRED_NON_PROD")
