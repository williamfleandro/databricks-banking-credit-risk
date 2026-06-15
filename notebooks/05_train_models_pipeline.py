# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Train Models Pipeline

# COMMAND ----------

# MAGIC %pip install xgboost lightgbm catboost

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime, timezone
import uuid

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from pyspark.sql import functions as F

# COMMAND ----------

dbutils.widgets.text("catalog_name", "mlops_dev")
dbutils.widgets.text("schema_name", "banking")

CATALOG_NAME = dbutils.widgets.get("catalog_name")
SCHEMA_NAME = dbutils.widgets.get("schema_name")

FEATURE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_features_uc"
MODEL_COMPARISON_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_model_comparison"
REGISTERED_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.credit_risk_default_model"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Shared/credit-risk-{CATALOG_NAME}-{SCHEMA_NAME}")

feature_df = spark.table(FEATURE_TABLE).toPandas()

target_column = "loan_status"
id_column = "loan_id"

feature_df = feature_df.dropna(subset=[target_column])
y = feature_df[target_column].astype(int)

drop_columns = [target_column, id_column, "feature_created_at"]
X = feature_df.drop(columns=[c for c in drop_columns if c in feature_df.columns])

numeric_features = X.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

positive_count = int((y_train == 1).sum())
negative_count = int((y_train == 0).sum())
scale_pos_weight = negative_count / max(positive_count, 1)

model_candidates = [
    {
        "model_type": "LogisticRegression",
        "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 42},
    },
    {
        "model_type": "RandomForest",
        "model": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "params": {"n_estimators": 250, "max_depth": 10, "class_weight": "balanced", "random_state": 42},
    },
    {
        "model_type": "GradientBoosting",
        "model": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.001,
            max_depth=3,
            random_state=42,
        ),
        "params": {"n_estimators": 250, "learning_rate": 0.001, "max_depth": 3, "random_state": 42},
    },
    {
        "model_type": "XGBoost",
        "model": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        ),
        "params": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.001, "scale_pos_weight": scale_pos_weight},
    },
    {
        "model_type": "LightGBM",
        "model": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.001,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "params": {"n_estimators": 300, "learning_rate": 0.001, "class_weight": "balanced"},
    },
    {
        "model_type": "CatBoost",
        "model": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        ),
        "params": {"iterations": 300, "learning_rate": 0.001, "depth": 6, "loss_function": "Logloss"},
    },
]

run_created_at = datetime.now(timezone.utc).isoformat()
results = []
best_pipeline = None
best_result = None
best_signature = None
best_input_example = None

for candidate in model_candidates:
    model_type = candidate["model_type"]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", candidate["model"]),
        ]
    )

    with mlflow.start_run(run_name=f"credit_risk_{model_type}") as run:
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }

        mlflow.log_params(candidate["params"])
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)

        input_example = X_test.head(5)
        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        result = {
            "model_run_id": str(uuid.uuid4()),
            "mlflow_run_id": run.info.run_id,
            "model_type": model_type,
            **metrics,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "created_at": run_created_at,
            "is_best_model": False,
        }

        results.append(result)

        if best_result is None or (
            result["f1_score"],
            result["recall"],
            result["roc_auc"],
        ) > (
            best_result["f1_score"],
            best_result["recall"],
            best_result["roc_auc"],
        ):
            best_result = result
            best_pipeline = pipeline
            best_signature = signature
            best_input_example = input_example

for row in results:
    row["is_best_model"] = row["model_run_id"] == best_result["model_run_id"]

comparison_pdf = pd.DataFrame(results)
comparison_sdf = spark.createDataFrame(comparison_pdf)

(
    comparison_sdf.write
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(MODEL_COMPARISON_TABLE)
)

display(comparison_sdf)

with mlflow.start_run(run_name="credit_risk_best_model_registration"):
    mlflow.log_param("selected_model_type", best_result["model_type"])
    mlflow.log_metrics({
        "best_f1_score": best_result["f1_score"],
        "best_recall": best_result["recall"],
        "best_roc_auc": best_result["roc_auc"],
    })

    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        signature=best_signature,
        input_example=best_input_example,
    )

dbutils.notebook.exit(f"BEST_MODEL_SELECTED: {best_result['model_type']}")
