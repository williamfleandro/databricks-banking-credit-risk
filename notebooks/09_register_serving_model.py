# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Register Serving Model

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

BEST_MODEL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_best_model"
FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_features_uc"
SERVING_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_serving_model"

mlflow.set_registry_uri("databricks-uc")

best = spark.table(BEST_MODEL_TABLE).limit(1).collect()[0]
source_model_uri = f"runs:/{best['mlflow_run_id']}/model"

sample_pdf = spark.table(FEATURE_TABLE).limit(10).toPandas()
drop_cols = [c for c in ["loan_status", "loan_id", "feature_created_at"] if c in sample_pdf.columns]
input_example = sample_pdf.drop(columns=drop_cols).head(5)

class CreditRiskServingWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import mlflow
        self.model = mlflow.sklearn.load_model(context.artifacts["model"])

    def predict(self, context, model_input):
        import pandas as pd
        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)[:, 1]

        risk_level = pd.cut(
            probabilities,
            bins=[-0.01, 0.30, 0.70, 1.01],
            labels=["low", "medium", "high"]
        ).astype(str)

        return pd.DataFrame({
            "prediction": predictions.astype(int),
            "probability_default": probabilities.astype(float),
            "risk_level": risk_level,
        })

signature = infer_signature(input_example, pd.DataFrame({
    "prediction": [0] * len(input_example),
    "probability_default": [0.0] * len(input_example),
    "risk_level": ["low"] * len(input_example),
}))

with mlflow.start_run(run_name="credit_risk_serving_model_registration"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=CreditRiskServingWrapper(),
        artifacts={"model": source_model_uri},
        registered_model_name=SERVING_MODEL_NAME,
        signature=signature,
        input_example=input_example,
    )

dbutils.notebook.exit(f"SERVING_MODEL_REGISTERED: {SERVING_MODEL_NAME}")
