# Databricks Banking Credit Risk

Projeto MLOps production-like para previsão de inadimplência / risco de crédito usando Databricks Lakehouse, Unity Catalog, Delta Lake, Feature Table, MLflow, Model Registry, Model Serving, Drift Monitoring, Retraining Decision, Drift Approval Gate, Champion/Challenger e GitHub Actions.

## Dataset

Dataset Kaggle sugerido:

```bash
kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip
```

Arquivo esperado:

```text
data/raw/credit_risk_dataset.csv
```

## Pipeline

```text
Bronze → Silver → Data Quality → Gold → Feature Table UC → Train → Evaluate → Drift → Retraining Decision → Approval Gate → Promote → Batch → Serving → Champion/Challenger → Rollback
```

## Rodar

```bash
unset DATABRICKS_TOKEN
unset DATABRICKS_HOST

databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run credit_risk_mlops_pipeline -t dev
```
