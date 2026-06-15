# Databricks notebook source
# MAGIC %pip install -q xgboost lightgbm catboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 08 - Batch Inference

# COMMAND ----------

from datetime import datetime, timezone
import mlflow
import pandas as pd

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_features_uc"
BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"
PREDICTIONS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_batch_predictions"

mlflow.set_registry_uri("databricks-uc")

best = spark.table(BEST_MODEL_TABLE).limit(1).collect()[0]
run_id = best["mlflow_run_id"]
model_uri = f"runs:/{run_id}/model"

model = mlflow.sklearn.load_model(model_uri)

features_pdf = spark.table(FEATURE_TABLE).toPandas()

loan_ids = features_pdf["loan_id"].astype(str)
drop_cols = [c for c in ["loan_status", "loan_id", "feature_created_at"] if c in features_pdf.columns]
X = features_pdf.drop(columns=drop_cols)

pred = model.predict(X)
proba = model.predict_proba(X)[:, 1]

predictions_pdf = pd.DataFrame({
    "loan_id": loan_ids,
    "prediction": pred.astype(int),
    "probability_default": proba.astype(float),
    "risk_level": pd.cut(
        proba,
        bins=[-0.01, 0.30, 0.70, 1.01],
        labels=["low", "medium", "high"]
    ).astype(str),
    "model_type": best["model_type"],
    "mlflow_run_id": run_id,
    "prediction_created_at": datetime.now(timezone.utc).isoformat(),
})

predictions_sdf = spark.createDataFrame(predictions_pdf)

(
    predictions_sdf.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTIONS_TABLE)
)

display(predictions_sdf.limit(20))
dbutils.notebook.exit(f"BATCH_INFERENCE_COMPLETED: {PREDICTIONS_TABLE}, records={len(predictions_pdf)}")
