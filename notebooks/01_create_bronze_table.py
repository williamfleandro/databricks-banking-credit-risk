# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Create Bronze Table

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

RAW_FILE_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/raw/credit_risk_dataset.csv"
BRONZE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_bronze"

print(f"Raw path: {RAW_FILE_PATH}")
print(f"Bronze table: {BRONZE_TABLE}")

# COMMAND ----------

try:
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.raw")
except Exception as exc:
    print(f"Volume creation skipped or failed: {exc}")

# COMMAND ----------

try:
    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(RAW_FILE_PATH)
    )
except Exception as exc:
    raise Exception(
        f"Could not read {RAW_FILE_PATH}. "
        "Upload credit_risk_dataset.csv to the Unity Catalog volume first."
    ) from exc

bronze_df = (
    raw_df
    .withColumn("ingestion_timestamp", F.current_timestamp())
    .withColumn("source_file", F.lit(RAW_FILE_PATH))
    .withColumn("bronze_created_at", F.lit(datetime.now(timezone.utc).isoformat()))
)

(
    bronze_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

record_count = spark.table(BRONZE_TABLE).count()
dbutils.notebook.exit(f"BRONZE_TABLE_READY: {BRONZE_TABLE}, records={record_count}")
