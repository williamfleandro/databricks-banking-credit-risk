# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Evaluate Best Model

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

MODEL_COMPARISON_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_model_comparison"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"

df = spark.table(MODEL_COMPARISON_TABLE)

best_df = (
    df.filter(F.col("is_best_model") == True)
    .orderBy(F.col("created_at").desc())
    .limit(1)
)

if best_df.count() == 0:
    raise Exception("No best model found.")

(
    best_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BEST_MODEL_TABLE)
)

display(best_df)
row = best_df.collect()[0]
dbutils.notebook.exit(f"BEST_MODEL_EVALUATED: {row['model_type']}")
