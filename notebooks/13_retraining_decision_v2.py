# Databricks notebook source
# MAGIC %md
# MAGIC # 13 - Retraining Decision
# MAGIC
# MAGIC Decide se o modelo de Credit Risk precisa de retreinamento com base no resultado do Drift Monitoring.
# MAGIC
# MAGIC Correção importante:
# MAGIC - Usa schema explícito no `spark.createDataFrame`.
# MAGIC - Evita erro `[CANNOT_DETERMINE_TYPE]` quando algum campo vem `None`.
# MAGIC - Mantém rastreabilidade da decisão em tabela Delta no Unity Catalog.

# COMMAND ----------

from datetime import datetime, timezone
import uuid

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType,
    LongType,
    DoubleType,
    ArrayType,
)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

DRIFT_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_feature_drift_metrics"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"
RETRAINING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_retraining_requests"

RUN_CREATED_AT = datetime.now(timezone.utc).isoformat()

print(f"Catalog: {CATALOG_NAME}")
print(f"Schema: {SCHEMA_NAME}")
print(f"Drift table: {DRIFT_TABLE}")
print(f"Best model table: {BEST_MODEL_TABLE}")
print(f"Retraining table: {RETRAINING_TABLE}")

# COMMAND ----------

def table_exists(table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).collect()
        return True
    except Exception:
        return False


def get_first_existing_column(df, candidates):
    for col_name in candidates:
        if col_name in df.columns:
            return col_name
    return None


def safe_string(value):
    if value is None:
        return None
    return str(value)


def safe_float(value):
    if value is None:
        return None
    return float(value)


# COMMAND ----------

# Read latest drift metrics.
if not table_exists(DRIFT_TABLE):
    raise Exception(f"RETRAINING_DECISION_FAILED: Drift table not found: {DRIFT_TABLE}")

drift_df = spark.table(DRIFT_TABLE)

print("Drift columns:")
print(drift_df.columns)

status_col = get_first_existing_column(
    drift_df,
    ["drift_status", "status", "feature_status"]
)

feature_col = get_first_existing_column(
    drift_df,
    ["feature_name", "feature", "column_name"]
)

created_col = get_first_existing_column(
    drift_df,
    ["created_at", "run_created_at", "monitoring_created_at"]
)

if status_col is None:
    raise Exception(
        f"RETRAINING_DECISION_FAILED: could not find drift status column in {DRIFT_TABLE}"
    )

if feature_col is None:
    raise Exception(
        f"RETRAINING_DECISION_FAILED: could not find feature name column in {DRIFT_TABLE}"
    )

# If table has more than one monitoring run, use latest run only when possible.
latest_drift_df = drift_df

if created_col:
    latest_created_at_rows = (
        drift_df
        .select(F.col(created_col).alias("created_at"))
        .orderBy(F.col("created_at").desc())
        .limit(1)
        .collect()
    )

    if latest_created_at_rows:
        latest_created_at = latest_created_at_rows[0]["created_at"]
        latest_drift_df = drift_df.filter(F.col(created_col) == latest_created_at)
        print(f"Using latest drift run: {latest_created_at}")

display(latest_drift_df)

# COMMAND ----------

drift_features = [
    str(row[feature_col])
    for row in (
        latest_drift_df
        .filter(F.upper(F.col(status_col)) == "DRIFT")
        .select(feature_col)
        .distinct()
        .collect()
    )
]

warning_features = [
    str(row[feature_col])
    for row in (
        latest_drift_df
        .filter(F.upper(F.col(status_col)) == "WARNING")
        .select(feature_col)
        .distinct()
        .collect()
    )
]

drift_feature_count = len(drift_features)
warning_feature_count = len(warning_features)

print(f"Drift features: {drift_features}")
print(f"Warning features: {warning_features}")
print(f"Drift feature count: {drift_feature_count}")
print(f"Warning feature count: {warning_feature_count}")

# COMMAND ----------

if drift_feature_count > 0:
    decision = "RETRAINING_REQUIRED"
    retraining_required = True
    priority = "HIGH"
    reason = (
        f"Detected {drift_feature_count} feature(s) with DRIFT: "
        f"{', '.join(drift_features)}"
    )
elif warning_feature_count > 0:
    decision = "RETRAINING_RECOMMENDED"
    retraining_required = False
    priority = "MEDIUM"
    reason = (
        f"Detected {warning_feature_count} feature(s) with WARNING: "
        f"{', '.join(warning_features)}"
    )
else:
    decision = "RETRAINING_NOT_REQUIRED"
    retraining_required = False
    priority = "LOW"
    reason = "No relevant drift detected. Current model remains valid."

print(f"Decision: {decision}")
print(f"Retraining required: {retraining_required}")
print(f"Priority: {priority}")
print(f"Reason: {reason}")

# COMMAND ----------

# Read latest best model metadata when available.
best_model_type = None
best_mlflow_run_id = None
best_f1_score = None
best_roc_auc = None

if table_exists(BEST_MODEL_TABLE):
    best_df = spark.table(BEST_MODEL_TABLE)

    print("Best model columns:")
    print(best_df.columns)

    best_created_col = get_first_existing_column(
        best_df,
        ["created_at", "evaluation_created_at", "run_created_at"]
    )

    ordered_best_df = best_df

    if best_created_col:
        ordered_best_df = best_df.orderBy(F.col(best_created_col).desc())

    best_rows = ordered_best_df.limit(1).collect()

    if best_rows:
        best = best_rows[0].asDict()

        best_model_type = safe_string(
            best.get("model_type")
            or best.get("best_model_type")
            or best.get("model_name")
        )

        best_mlflow_run_id = safe_string(
            best.get("mlflow_run_id")
            or best.get("run_id")
            or best.get("best_mlflow_run_id")
        )

        best_f1_score = safe_float(
            best.get("f1_score")
            or best.get("f1")
            or best.get("best_f1_score")
        )

        best_roc_auc = safe_float(
            best.get("roc_auc")
            or best.get("auc")
            or best.get("best_roc_auc")
        )

print(f"Best model type: {best_model_type}")
print(f"Best MLflow run ID: {best_mlflow_run_id}")
print(f"Best F1 score: {best_f1_score}")
print(f"Best ROC AUC: {best_roc_auc}")

# COMMAND ----------

# Explicit schema avoids Spark Connect CANNOT_DETERMINE_TYPE error.
schema = StructType([
    StructField("retraining_request_id", StringType(), False),
    StructField("catalog_name", StringType(), False),
    StructField("schema_name", StringType(), False),
    StructField("source_drift_table", StringType(), False),
    StructField("source_best_model_table", StringType(), True),
    StructField("decision", StringType(), False),
    StructField("retraining_required", BooleanType(), False),
    StructField("priority", StringType(), False),
    StructField("reason", StringType(), False),
    StructField("drift_features", ArrayType(StringType()), False),
    StructField("warning_features", ArrayType(StringType()), False),
    StructField("drift_feature_count", LongType(), False),
    StructField("warning_feature_count", LongType(), False),
    StructField("best_model_type", StringType(), True),
    StructField("best_mlflow_run_id", StringType(), True),
    StructField("best_f1_score", DoubleType(), True),
    StructField("best_roc_auc", DoubleType(), True),
    StructField("created_at", StringType(), False),
])

row = [(
    str(uuid.uuid4()),
    CATALOG_NAME,
    SCHEMA_NAME,
    DRIFT_TABLE,
    BEST_MODEL_TABLE if table_exists(BEST_MODEL_TABLE) else None,
    decision,
    bool(retraining_required),
    priority,
    reason,
    drift_features,
    warning_features,
    int(drift_feature_count),
    int(warning_feature_count),
    best_model_type,
    best_mlflow_run_id,
    best_f1_score,
    best_roc_auc,
    RUN_CREATED_AT,
)]

result_df = spark.createDataFrame(row, schema=schema)

display(result_df)

# COMMAND ----------

(
    result_df.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(RETRAINING_TABLE)
)

# COMMAND ----------

dbutils.notebook.exit(
    f"{decision}: "
    f"retraining_required={retraining_required}, "
    f"priority={priority}, "
    f"drift_features={drift_features}, "
    f"warning_features={warning_features}"
)
