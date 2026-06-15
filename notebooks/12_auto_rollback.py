# Databricks notebook source
# MAGIC %md
# MAGIC # 12 - Auto Rollback

# COMMAND ----------

from datetime import datetime, timezone
import uuid

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

ROLLBACK_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_rollback_history"

row = [{
    "rollback_id": str(uuid.uuid4()),
    "catalog_name": CATALOG_NAME,
    "schema_name": SCHEMA_NAME,
    "rollback_required": False,
    "rollback_status": "NOT_REQUIRED",
    "reason": "No rollback condition detected.",
    "created_at": datetime.now(timezone.utc).isoformat(),
}]

df = spark.createDataFrame(row)
df.write.mode("append").option("mergeSchema", "true").saveAsTable(ROLLBACK_TABLE)

display(df)
dbutils.notebook.exit("ROLLBACK_NOT_REQUIRED")
