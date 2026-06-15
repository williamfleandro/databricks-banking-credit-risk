# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Create Feature Table

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

GOLD_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_gold"
FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_table"

df = spark.table(GOLD_TABLE)

candidate_columns = [
    "loan_id",
    "person_age",
    "person_income",
    "person_emp_length",
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
    "debt_to_income_ratio",
    "income_band",
    "age_band",
    "high_interest_rate",
    "high_loan_income_ratio",
    "previous_default_flag",
    "loan_status",
]

existing_columns = [c for c in candidate_columns if c in df.columns]

features_df = (
    df.select(*existing_columns)
    .withColumn("feature_created_at", F.lit(datetime.now(timezone.utc).isoformat()))
)

(
    features_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURE_TABLE)
)

record_count = spark.table(FEATURE_TABLE).count()
dbutils.notebook.exit(f"FEATURE_TABLE_READY: {FEATURE_TABLE}, records={record_count}")
