# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Silver Table

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

BRONZE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_bronze"
SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver"

bronze_df = spark.table(BRONZE_TABLE)

# COMMAND ----------

def normalize_columns(df):
    result = df
    for col_name in df.columns:
        new_name = (
            col_name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace(".", "_")
        )
        if new_name != col_name:
            result = result.withColumnRenamed(col_name, new_name)
    return result

df = normalize_columns(bronze_df)

if "loan_id" not in df.columns:
    df = df.withColumn("loan_id", F.monotonically_increasing_id().cast("string"))
else:
    df = df.withColumn("loan_id", F.col("loan_id").cast("string"))

numeric_casts = {
    "person_age": "double",
    "person_income": "double",
    "person_emp_length": "double",
    "loan_amnt": "double",
    "loan_int_rate": "double",
    "loan_status": "int",
    "loan_percent_income": "double",
    "cb_person_cred_hist_length": "double",
}

for col_name, dtype in numeric_casts.items():
    if col_name in df.columns:
        df = df.withColumn(col_name, F.col(col_name).cast(dtype))

for col_name in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
    if col_name in df.columns:
        df = df.withColumn(col_name, F.upper(F.trim(F.col(col_name).cast("string"))))

silver_df = (
    df
    .dropDuplicates(["loan_id"])
    .filter(F.col("loan_status").isin(0, 1))
    .withColumn("silver_created_at", F.lit(datetime.now(timezone.utc).isoformat()))
)

silver_df = silver_df.fillna({
    "person_emp_length": 0.0,
    "loan_int_rate": 0.0,
    "loan_percent_income": 0.0,
    "cb_person_cred_hist_length": 0.0,
})

(
    silver_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

record_count = spark.table(SILVER_TABLE).count()
dbutils.notebook.exit(f"SILVER_TABLE_READY: {SILVER_TABLE}, records={record_count}")
