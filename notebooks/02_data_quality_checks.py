# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Data Quality Checks

# COMMAND ----------

from datetime import datetime, timezone
import uuid
from pyspark.sql import functions as F

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver"
DQ_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_data_quality_results"

df = spark.table(SILVER_TABLE)

total_records = df.count()
loan_id_nulls = df.filter(F.col("loan_id").isNull()).count()
loan_id_distinct = df.select("loan_id").distinct().count()
invalid_target = df.filter(~F.col("loan_status").isin(0, 1)).count()

checks = [
    ("total_records_greater_than_zero", total_records > 0, str(total_records)),
    ("loan_id_not_null", loan_id_nulls == 0, str(loan_id_nulls)),
    ("loan_id_unique", loan_id_distinct == total_records, f"distinct={loan_id_distinct}, total={total_records}"),
    ("valid_loan_status", invalid_target == 0, str(invalid_target)),
]

if "person_age" in df.columns:
    invalid_age = df.filter((F.col("person_age") < 18) | (F.col("person_age") > 100)).count()
    checks.append(("person_age_valid_range", invalid_age == 0, str(invalid_age)))

if "person_income" in df.columns:
    invalid_income = df.filter(F.col("person_income") < 0).count()
    checks.append(("person_income_non_negative", invalid_income == 0, str(invalid_income)))

if "loan_amnt" in df.columns:
    invalid_loan_amount = df.filter(F.col("loan_amnt") <= 0).count()
    checks.append(("loan_amount_positive", invalid_loan_amount == 0, str(invalid_loan_amount)))

rows = [
    {
        "check_id": str(uuid.uuid4()),
        "catalog_name": CATALOG_NAME,
        "schema_name": SCHEMA_NAME,
        "check_name": name,
        "passed": bool(passed),
        "details": details,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for name, passed, details in checks
]

dq_df = spark.createDataFrame(rows)

(
    dq_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(DQ_TABLE)
)

display(dq_df)

failed = [name for name, passed, _ in checks if not passed]
if failed:
    raise Exception(f"DATA_QUALITY_FAILED: {failed}")

dbutils.notebook.exit("DATA_QUALITY_PASSED")
