# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Create Gold Table

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver"
GOLD_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_gold"

df = spark.table(SILVER_TABLE)

gold_df = df

if "person_income" in gold_df.columns and "loan_amnt" in gold_df.columns:
    gold_df = gold_df.withColumn(
        "debt_to_income_ratio",
        F.when(F.col("person_income") > 0, F.col("loan_amnt") / F.col("person_income")).otherwise(F.lit(0.0))
    )

if "person_income" in gold_df.columns:
    gold_df = gold_df.withColumn(
        "income_band",
        F.when(F.col("person_income") < 30000, "LOW")
        .when(F.col("person_income") < 80000, "MEDIUM")
        .otherwise("HIGH")
    )

if "person_age" in gold_df.columns:
    gold_df = gold_df.withColumn(
        "age_band",
        F.when(F.col("person_age") < 25, "YOUNG")
        .when(F.col("person_age") < 45, "ADULT")
        .when(F.col("person_age") < 65, "SENIOR")
        .otherwise("ELDER")
    )

if "loan_int_rate" in gold_df.columns:
    gold_df = gold_df.withColumn("high_interest_rate", F.when(F.col("loan_int_rate") >= 15, 1).otherwise(0))

if "loan_percent_income" in gold_df.columns:
    gold_df = gold_df.withColumn("high_loan_income_ratio", F.when(F.col("loan_percent_income") >= 0.30, 1).otherwise(0))

if "cb_person_default_on_file" in gold_df.columns:
    gold_df = gold_df.withColumn("previous_default_flag", F.when(F.col("cb_person_default_on_file") == "Y", 1).otherwise(0))

gold_df = gold_df.withColumn("gold_created_at", F.lit(datetime.now(timezone.utc).isoformat()))

(
    gold_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(GOLD_TABLE)
)

record_count = spark.table(GOLD_TABLE).count()
dbutils.notebook.exit(f"GOLD_TABLE_READY: {GOLD_TABLE}, records={record_count}")
