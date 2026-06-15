# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Data Quality Checks
# MAGIC
# MAGIC Data Quality Gate para o projeto Credit Risk.
# MAGIC
# MAGIC Este notebook:
# MAGIC - valida a tabela Silver;
# MAGIC - identifica registros inválidos;
# MAGIC - grava os registros inválidos em uma tabela de quarantine;
# MAGIC - regrava a Silver somente com registros válidos;
# MAGIC - executa novamente os checks;
# MAGIC - bloqueia o pipeline se a Silver final continuar inválida.

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    LongType,
    DoubleType,
)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver"
DQ_RESULTS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_data_quality_results"
QUARANTINE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_quarantine_records"
TMP_CLEAN_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver_clean_tmp"

RUN_CREATED_AT = datetime.now(timezone.utc).isoformat()

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Silver table: {SILVER_TABLE}")
print(f"DQ results table: {DQ_RESULTS_TABLE}")
print(f"Quarantine table: {QUARANTINE_TABLE}")

# COMMAND ----------

df = spark.table(SILVER_TABLE)

total_records_before = df.count()

print(f"Total records before DQ cleaning: {total_records_before}")

if total_records_before == 0:
    raise Exception("DATA_QUALITY_FAILED: Silver table is empty.")

# COMMAND ----------

# Build invalid record rules.
invalid_reason = F.concat_ws(
    ",",
    F.when(F.col("loan_id").isNull(), F.lit("loan_id_not_null")),
    F.when(~F.col("loan_status").isin(0, 1), F.lit("loan_status_valid_values")),
    F.when(
        F.col("person_age").isNull()
        | (F.col("person_age") < 18)
        | (F.col("person_age") > 100),
        F.lit("person_age_valid_range"),
    ),
    F.when(
        F.col("person_income").isNull()
        | (F.col("person_income") < 0),
        F.lit("person_income_non_negative"),
    ),
    F.when(
        F.col("loan_amnt").isNull()
        | (F.col("loan_amnt") <= 0),
        F.lit("loan_amnt_positive"),
    ),
    F.when(
        F.col("loan_percent_income").isNull()
        | (F.col("loan_percent_income") < 0)
        | (F.col("loan_percent_income") > 1),
        F.lit("loan_percent_income_valid_range"),
    ),
)

invalid_df = (
    df
    .withColumn("dq_error_reason", invalid_reason)
    .filter(F.col("dq_error_reason") != "")
    .withColumn("dq_run_created_at", F.lit(RUN_CREATED_AT))
    .withColumn("dq_source_table", F.lit(SILVER_TABLE))
)

invalid_count = invalid_df.count()

print(f"Invalid records found before DQ cleaning: {invalid_count}")

if invalid_count > 0:
    display(
        invalid_df.select(
            "loan_id",
            "person_age",
            "person_income",
            "loan_amnt",
            "loan_percent_income",
            "loan_status",
            "dq_error_reason",
        ).limit(50)
    )

    (
        invalid_df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(QUARANTINE_TABLE)
    )

# COMMAND ----------

# Keep only valid records.
clean_df = df.filter(
    (F.col("loan_id").isNotNull())
    & (F.col("loan_status").isin(0, 1))
    & (F.col("person_age").isNotNull())
    & (F.col("person_age") >= 18)
    & (F.col("person_age") <= 100)
    & (F.col("person_income").isNotNull())
    & (F.col("person_income") >= 0)
    & (F.col("loan_amnt").isNotNull())
    & (F.col("loan_amnt") > 0)
    & (F.col("loan_percent_income").isNotNull())
    & (F.col("loan_percent_income") >= 0)
    & (F.col("loan_percent_income") <= 1)
)

clean_count = clean_df.count()

print(f"Clean records after DQ cleaning: {clean_count}")
print(f"Records removed by DQ cleaning: {total_records_before - clean_count}")

if clean_count == 0:
    raise Exception("DATA_QUALITY_FAILED: No valid records left after DQ cleaning.")

# COMMAND ----------

# Recreate Silver with clean records only.
# We write to a temporary managed table first to avoid read/write conflict.
spark.sql(f"DROP TABLE IF EXISTS {TMP_CLEAN_TABLE}")

(
    clean_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TMP_CLEAN_TABLE)
)

spark.sql(f"DROP TABLE IF EXISTS {SILVER_TABLE}")

spark.sql(f"""
CREATE TABLE {SILVER_TABLE}
USING DELTA
AS
SELECT *
FROM {TMP_CLEAN_TABLE}
""")

spark.sql(f"DROP TABLE IF EXISTS {TMP_CLEAN_TABLE}")

# COMMAND ----------

# Reload cleaned Silver and run final checks.
final_df = spark.table(SILVER_TABLE)

total_records = final_df.count()

invalid_age_records = final_df.filter(
    F.col("person_age").isNull()
    | (F.col("person_age") < 18)
    | (F.col("person_age") > 100)
).count()

duplicate_loan_id_records = (
    final_df.groupBy("loan_id")
    .count()
    .filter(F.col("count") > 1)
    .count()
)

null_loan_id_records = final_df.filter(F.col("loan_id").isNull()).count()

invalid_target_records = final_df.filter(~F.col("loan_status").isin(0, 1)).count()

invalid_income_records = final_df.filter(
    F.col("person_income").isNull()
    | (F.col("person_income") < 0)
).count()

invalid_loan_amount_records = final_df.filter(
    F.col("loan_amnt").isNull()
    | (F.col("loan_amnt") <= 0)
).count()

invalid_loan_percent_income_records = final_df.filter(
    F.col("loan_percent_income").isNull()
    | (F.col("loan_percent_income") < 0)
    | (F.col("loan_percent_income") > 1)
).count()

min_age = final_df.agg(F.min("person_age").alias("min_age")).collect()[0]["min_age"]
max_age = final_df.agg(F.max("person_age").alias("max_age")).collect()[0]["max_age"]

print(f"Final total records: {total_records}")
print(f"Final min_age: {min_age}")
print(f"Final max_age: {max_age}")
print(f"Final invalid_age_records: {invalid_age_records}")

# COMMAND ----------

checks = [
    {
        "check_name": "total_records_greater_than_zero",
        "passed": total_records > 0,
        "failed_records": 0 if total_records > 0 else total_records,
        "metric_value": float(total_records),
    },
    {
        "check_name": "loan_id_not_null",
        "passed": null_loan_id_records == 0,
        "failed_records": null_loan_id_records,
        "metric_value": float(null_loan_id_records),
    },
    {
        "check_name": "loan_id_unique",
        "passed": duplicate_loan_id_records == 0,
        "failed_records": duplicate_loan_id_records,
        "metric_value": float(duplicate_loan_id_records),
    },
    {
        "check_name": "loan_status_valid_values",
        "passed": invalid_target_records == 0,
        "failed_records": invalid_target_records,
        "metric_value": float(invalid_target_records),
    },
    {
        "check_name": "person_age_valid_range",
        "passed": invalid_age_records == 0,
        "failed_records": invalid_age_records,
        "metric_value": float(invalid_age_records),
    },
    {
        "check_name": "person_income_non_negative",
        "passed": invalid_income_records == 0,
        "failed_records": invalid_income_records,
        "metric_value": float(invalid_income_records),
    },
    {
        "check_name": "loan_amnt_positive",
        "passed": invalid_loan_amount_records == 0,
        "failed_records": invalid_loan_amount_records,
        "metric_value": float(invalid_loan_amount_records),
    },
    {
        "check_name": "loan_percent_income_valid_range",
        "passed": invalid_loan_percent_income_records == 0,
        "failed_records": invalid_loan_percent_income_records,
        "metric_value": float(invalid_loan_percent_income_records),
    },
]

# COMMAND ----------

schema = StructType([
    StructField("catalog_name", StringType(), False),
    StructField("schema_name", StringType(), False),
    StructField("table_name", StringType(), False),
    StructField("check_name", StringType(), False),
    StructField("passed", BooleanType(), False),
    StructField("failed_records", LongType(), False),
    StructField("metric_value", DoubleType(), False),
    StructField("total_records", LongType(), False),
    StructField("run_created_at", StringType(), False),
])

rows = [
    (
        CATALOG_NAME,
        SCHEMA_NAME,
        SILVER_TABLE,
        check["check_name"],
        bool(check["passed"]),
        int(check["failed_records"]),
        float(check["metric_value"]),
        int(total_records),
        RUN_CREATED_AT,
    )
    for check in checks
]

results_df = spark.createDataFrame(rows, schema=schema)

(
    results_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DQ_RESULTS_TABLE)
)

display(results_df)

# COMMAND ----------

failed_checks = [
    check["check_name"]
    for check in checks
    if not check["passed"]
]

if failed_checks:
    raise Exception(f"DATA_QUALITY_FAILED: {failed_checks}")

dbutils.notebook.exit(
    f"DATA_QUALITY_PASSED: "
    f"table={SILVER_TABLE}, "
    f"records={total_records}, "
    f"min_age={min_age}, "
    f"max_age={max_age}, "
    f"quarantined_records={invalid_count}"
)
