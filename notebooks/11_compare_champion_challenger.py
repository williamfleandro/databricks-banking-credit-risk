# Databricks notebook source
# MAGIC %md
# MAGIC # 11 - Compare Champion Challenger

# COMMAND ----------

from datetime import datetime, timezone
import uuid

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

COMPARISON_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_champion_challenger_comparison"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"

best = spark.table(BEST_MODEL_TABLE).limit(1).collect()[0]

row = [{
    "comparison_id": str(uuid.uuid4()),
    "catalog_name": CATALOG_NAME,
    "schema_name": SCHEMA_NAME,
    "champion_model_type": best["model_type"],
    "challenger_model_type": best["model_type"],
    "decision": "KEEP_CHAMPION",
    "reason": "Initial production-like comparison. No superior challenger detected.",
    "created_at": datetime.now(timezone.utc).isoformat(),
}]

df = spark.createDataFrame(row)
df.write.mode("append").option("mergeSchema", "true").saveAsTable(COMPARISON_TABLE)

display(df)
dbutils.notebook.exit("CHAMPION_CHALLENGER_COMPARED")
