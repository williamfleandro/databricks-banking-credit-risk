# Databricks Banking — Credit Risk / Loan Default

Projeto MLOps production-like separado para o domínio bancário **Credit Risk / Loan Default**, seguindo o padrão já construído no projeto `databricks-mlops-churn-lab`.

## Arquitetura

```text
Kaggle Dataset
  ↓
Unity Catalog Volume
  ↓
Bronze Delta Table
  ↓
Silver Delta Table
  ↓
Data Quality Gate
  ↓
Gold Delta Table
  ↓
Feature Table formal no Unity Catalog
  ↓
Treinamento multi-modelo
  ↓
MLflow Tracking
  ↓
Unity Catalog Model Registry
  ↓
Batch Inference
  ↓
Model Serving REST
  ↓
Drift Monitoring
  ↓
Retraining Decision
  ↓
Drift Approval Gate
  ↓
Rollback Decision
```

## Dataset

```bash
kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip
```

Arquivo esperado:

```text
data/raw/credit_risk_dataset.csv
```

## Ambientes

| Target | Catalog | Schema |
|---|---|---|
| dev | mlops_dev | banking |
| acc | mlops_acc | banking |
| prod | mlops_production | banking |

## Target do modelo

```text
loan_status
```

## Endpoint sugerido

```text
credit-risk-endpoint-prod
```

## Rodar localmente

```bash
unset DATABRICKS_TOKEN
unset DATABRICKS_HOST

databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run credit_risk_mlops_pipeline -t dev
```

## Próximos passos

1. Baixar a base do Kaggle.
2. Criar schema e volume no Unity Catalog.
3. Subir CSV para `/Volumes/mlops_dev/banking/raw/`.
4. Implementar Bronze/Silver/Gold.
5. Criar Feature Table formal no Unity Catalog.
6. Treinar modelos com MLflow.
7. Adicionar Serving, Drift Monitoring e Approval Gate.
