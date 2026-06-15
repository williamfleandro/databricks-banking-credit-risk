# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Silver Table
# MAGIC
# MAGIC Camada Silver do projeto Credit Risk.
# MAGIC
# MAGIC Responsabilidades:
# MAGIC - Normalizar nomes das colunas.
# MAGIC - Criar `loan_id` caso não exista.
# MAGIC - Converter tipos numéricos.
# MAGIC - Padronizar colunas categóricas.
# MAGIC - Remover registros inválidos.
# MAGIC - Garantir que a Silver não tenha idade fora do intervalo aceito.
# MAGIC - Recriar a tabela Delta limpa.

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

BRONZE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_bronze"
SILVER_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_silver"

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Bronze table: {BRONZE_TABLE}")
print(f"Silver table: {SILVER_TABLE}")

# COMMAND ----------

bronze_df = spark.table(BRONZE_TABLE)

bronze_records = bronze_df.count()
print(f"Bronze records: {bronze_records}")

display(bronze_df.limit(10))

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

print("Columns after normalization:")
print(df.columns)

# COMMAND ----------

# Create technical primary key when source dataset does not provide one.
if "loan_id" not in df.columns:
    df = df.withColumn("loan_id", F.monotonically_increasing_id().cast("string"))
else:
    df = df.withColumn("loan_id", F.col("loan_id").cast("string"))

# COMMAND ----------

# Numeric casts based on Kaggle Credit Risk Dataset.
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

# COMMAND ----------

# Categorical standardization.
categorical_columns = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

for col_name in categorical_columns:
    if col_name in df.columns:
        df = df.withColumn(
            col_name,
            F.upper(F.trim(F.col(col_name).cast("string")))
        )

# COMMAND ----------

# Base Silver cleaning.
silver_df = (
    df
    .dropDuplicates(["loan_id"])
    .filter(F.col("loan_status").isin(0, 1))
)

records_after_target_filter = silver_df.count()
print(f"Records after target filter: {records_after_target_filter}")

# COMMAND ----------

# Strong age cleaning.
# This dataset can contain unrealistic outliers such as age > 100.
# These records must be removed in Silver before Data Quality Gate.
if "person_age" in silver_df.columns:
    invalid_age_df = silver_df.filter(
        (F.col("person_age").isNull()) |
        (F.col("person_age") < 18) |
        (F.col("person_age") > 100)
    )

    invalid_age_count = invalid_age_df.count()

    print(f"Invalid age records before cleaning: {invalid_age_count}")

    if invalid_age_count > 0:
        display(
            invalid_age_df.select(
                "loan_id",
                "person_age",
                "person_income",
                "loan_amnt",
                "loan_status"
            ).limit(20)
        )

    silver_df = silver_df.filter(
        (F.col("person_age").isNotNull()) &
        (F.col("person_age") >= 18) &
        (F.col("person_age") <= 100)
    )

records_after_age_filter = silver_df.count()
print(f"Records after age filter: {records_after_age_filter}")
print(f"Removed by age filter: {records_after_target_filter - records_after_age_filter}")

# COMMAND ----------

# Additional business-safe cleaning.
if "person_income" in silver_df.columns:
    silver_df = silver_df.filter(
        (F.col("person_income").isNotNull()) &
        (F.col("person_income") >= 0)
    )

if "loan_amnt" in silver_df.columns:
    silver_df = silver_df.filter(
        (F.col("loan_amnt").isNotNull()) &
        (F.col("loan_amnt") > 0)
    )

if "loan_percent_income" in silver_df.columns:
    silver_df = silver_df.filter(
        (F.col("loan_percent_income").isNotNull()) &
        (F.col("loan_percent_income") >= 0) &
        (F.col("loan_percent_income") <= 1)
    )

# COMMAND ----------

# Fill remaining nullable numeric values.
fill_values = {}

for col_name in ["person_emp_length", "loan_int_rate", "cb_person_cred_hist_length"]:
    if col_name in silver_df.columns:
        fill_values[col_name] = 0.0

if fill_values:
    silver_df = silver_df.fillna(fill_values)

# COMMAND ----------

silver_df = silver_df.withColumn(
    "silver_created_at",
    F.lit(datetime.now(timezone.utc).isoformat())
)

# COMMAND ----------

# Final validation before writing.
final_records = silver_df.count()

print(f"Final Silver records: {final_records}")

if final_records == 0:
    raise Exception("SILVER_VALIDATION_FAILED: Silver table would be empty after cleaning.")

if "person_age" in silver_df.columns:
    final_invalid_age_count = silver_df.filter(
        (F.col("person_age").isNull()) |
        (F.col("person_age") < 18) |
        (F.col("person_age") > 100)
    ).count()

    min_age = silver_df.agg(F.min("person_age").alias("min_age")).collect()[0]["min_age"]
    max_age = silver_df.agg(F.max("person_age").alias("max_age")).collect()[0]["max_age"]

    print(f"Final min_age: {min_age}")
    print(f"Final max_age: {max_age}")
    print(f"Final invalid age records: {final_invalid_age_count}")

    if final_invalid_age_count > 0:
        raise Exception(
            f"SILVER_VALIDATION_FAILED: invalid age records still found: {final_invalid_age_count}"
        )

# COMMAND ----------

# Force table recreation to avoid keeping old invalid data.
spark.sql(f"DROP TABLE IF EXISTS {SILVER_TABLE}")

(
    silver_df.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

# COMMAND ----------

# Post-write validation.
written_df = spark.table(SILVER_TABLE)

written_records = written_df.count()

written_min_age = written_df.agg(F.min("person_age").alias("min_age")).collect()[0]["min_age"]
written_max_age = written_df.agg(F.max("person_age").alias("max_age")).collect()[0]["max_age"]

written_invalid_age_count = written_df.filter(
    (F.col("person_age").isNull()) |
    (F.col("person_age") < 18) |
    (F.col("person_age") > 100)
).count()

print(f"Written Silver records: {written_records}")
print(f"Written min_age: {written_min_age}")
print(f"Written max_age: {written_max_age}")
print(f"Written invalid age records: {written_invalid_age_count}")

if written_invalid_age_count > 0:
    raise Exception(
        f"SILVER_POST_WRITE_VALIDATION_FAILED: invalid age records found after write: {written_invalid_age_count}"
    )

display(written_df.limit(20))

# COMMAND ----------

dbutils.notebook.exit(
    f"SILVER_TABLE_READY: {SILVER_TABLE}, "
    f"records={written_records}, "
    f"min_age={written_min_age}, "
    f"max_age={written_max_age}, "
    f"invalid_age_records={written_invalid_age_count}"
)
