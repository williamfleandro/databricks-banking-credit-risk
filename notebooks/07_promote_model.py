# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Promote Model

# COMMAND ----------

from datetime import datetime, timezone
import uuid

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

PROMOTION_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_model_promotions"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"

best = spark.table(BEST_MODEL_TABLE).limit(1).collect()[0]

status = "CHAMPION_CANDIDATE" if CATALOG_NAME != "mlops_production" else "PRODUCTION_CANDIDATE"

row = [{
    "promotion_id": str(uuid.uuid4()),
    "catalog_name": CATALOG_NAME,
    "schema_name": SCHEMA_NAME,
    "model_type": best["model_type"],
    "mlflow_run_id": best["mlflow_run_id"],
    "promotion_status": status,
    "created_at": datetime.now(timezone.utc).isoformat(),
}]

promotions_df = spark.createDataFrame(row)

(
    promotions_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(PROMOTION_TABLE)
)

display(promotions_df)
dbutils.notebook.exit(f"MODEL_PROMOTION_REGISTERED: {status}")
