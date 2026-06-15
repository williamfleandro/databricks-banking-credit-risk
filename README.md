# Databricks Banking Credit Risk MLOps Pipeline

Projeto completo de MLOps para **Credit Risk / Loan Default Prediction** usando **Databricks**, **Unity Catalog**, **Delta Lake**, **MLflow**, **Feature Engineering**, **Data Quality**, **Model Registry**, **Batch Inference**, **Drift Monitoring** e promoção entre ambientes **DEV**, **ACC** e **PROD**.

Repositório:

```text
https://github.com/williamfleandro/databricks-banking-credit-risk
````

---

## 1. Objetivo do Projeto

O objetivo deste projeto é construir uma esteira MLOps bancária para prever risco de inadimplência de clientes com base em dados de crédito.

O pipeline cobre:

* Ingestão de dados brutos no Lakehouse.
* Criação das camadas Bronze, Silver e Gold.
* Validação de qualidade dos dados.
* Tratamento de outliers.
* Criação de Feature Table.
* Registro formal da Feature Table no Unity Catalog.
* Treinamento de múltiplos modelos.
* Seleção automática do melhor modelo.
* Registro no MLflow.
* Batch Inference.
* Monitoramento de drift.
* Decisão de retreinamento.
* Gate de aprovação para drift.
* Preparação para serving e governança em produção.

---

## 2. Arquitetura Geral

```text
Kaggle Dataset
  ↓
Databricks Volume - Unity Catalog
  ↓
Bronze Table - Raw Data
  ↓
Silver Table - Cleaned Data
  ↓
Data Quality Gate
  ↓
Gold Table - Enriched Data
  ↓
Feature Table
  ↓
Unity Catalog Feature Table
  ↓
Model Training Pipeline
  ↓
MLflow Tracking
  ↓
Model Comparison
  ↓
Best Model Selection
  ↓
Model Registry
  ↓
Batch Inference
  ↓
Drift Monitoring
  ↓
Retraining Decision
  ↓
Drift Approval Gate
  ↓
ACC / PROD Promotion
```

---

## 3. Stack Utilizada

* Databricks Asset Bundles
* Databricks Workflows
* Unity Catalog
* Delta Lake
* MLflow
* Feature Engineering / Feature Table
* Spark / PySpark
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* GitHub Actions
* Databricks CLI
* Kaggle Dataset

---

## 4. Dataset

Dataset utilizado:

```text
Kaggle - Credit Risk Dataset
```

Arquivo esperado:

```text
credit_risk_dataset.csv
```

Caminho local:

```text
data/raw/credit_risk_dataset.csv
```

Caminhos nos Volumes do Unity Catalog:

```text
/Volumes/mlops_dev/banking/raw/credit_risk_dataset.csv
/Volumes/mlops_acc/banking/raw/credit_risk_dataset.csv
/Volumes/mlops_production/banking/raw/credit_risk_dataset.csv
```

---

## 5. Ambientes

O projeto está preparado para três ambientes:

| Ambiente | Catalog            | Schema    |
| -------- | ------------------ | --------- |
| DEV      | `mlops_dev`        | `banking` |
| ACC      | `mlops_acc`        | `banking` |
| PROD     | `mlops_production` | `banking` |

---

## 6. Estrutura do Projeto

```text
databricks-banking-credit-risk/
├── .github/
│   └── workflows/
│       ├── databricks-bundle-dev.yml
│       ├── databricks-bundle-acc.yml
│       └── databricks-bundle-prod.yml
├── data/
│   └── raw/
│       └── .gitkeep
├── docs/
│   └── architecture.md
├── notebooks/
│   ├── 01_create_bronze_table.py
│   ├── 02_create_silver_table.py
│   ├── 02_create_silver_table_v2.py
│   ├── 02_data_quality_checks.py
│   ├── 02_data_quality_checks_v2.py
│   ├── 03_create_gold_table.py
│   ├── 04_create_feature_table.py
│   ├── 04_register_feature_table_uc.py
│   ├── 05_train_models_pipeline.py
│   ├── 06_evaluate_best_model.py
│   ├── 07_promote_model.py
│   ├── 08_batch_inference.py
│   ├── 08_batch_inference_v2.py
│   ├── 09_register_serving_model.py
│   ├── 09_register_serving_model_v2.py
│   ├── 10_monitor_drift.py
│   ├── 10_drift_approval_gate.py
│   ├── 11_compare_champion_challenger.py
│   ├── 12_auto_rollback.py
│   └── 13_retraining_decision_v2.py
├── resources/
│   └── credit_risk_job.yml
├── scripts/
│   ├── create_uc_schema_and_volume.sql
│   ├── download_kaggle_dataset.sh
│   └── upload_to_databricks_volume.sh
├── serving/
│   ├── inference_client.py
│   └── test_payload.json
├── databricks.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 7. Tabelas Criadas no Unity Catalog

### Camadas Lakehouse

```text
mlops_dev.banking.credit_risk_bronze
mlops_dev.banking.credit_risk_silver
mlops_dev.banking.credit_risk_gold
```

### Feature Store / Feature Table

```text
mlops_dev.banking.credit_risk_feature_table
mlops_dev.banking.credit_risk_features_uc
```

### Qualidade de Dados

```text
mlops_dev.banking.credit_risk_data_quality_results
mlops_dev.banking.credit_risk_quarantine_records
```

### MLOps / Modelos

```text
mlops_dev.banking.credit_risk_model_comparison
mlops_dev.banking.credit_risk_best_model
mlops_dev.banking.credit_risk_model_promotions
mlops_dev.banking.credit_risk_batch_predictions
```

### Drift / Retreinamento / Governança

```text
mlops_dev.banking.credit_risk_feature_drift_metrics
mlops_dev.banking.credit_risk_retraining_requests
mlops_dev.banking.credit_risk_drift_approval_requests
mlops_dev.banking.credit_risk_champion_challenger_comparison
mlops_dev.banking.credit_risk_rollback_history
```

---

## 8. Data Quality

A camada Silver realiza limpeza dos dados antes da validação.

Principais regras:

* `loan_id` não pode ser nulo.
* `loan_id` deve ser único.
* `loan_status` deve conter apenas `0` ou `1`.
* `person_age` deve estar entre `18` e `100`.
* `person_income` deve ser maior ou igual a `0`.
* `loan_amnt` deve ser maior que `0`.
* `loan_percent_income` deve estar entre `0` e `1`.

Resultado validado em DEV:

```text
total_records = 32576
min_age = 20
max_age = 94
invalid_age_records = 0
```

Foram removidos registros inválidos de idade, incluindo outliers acima de 100 anos.

---

## 9. Modelos Treinados

O pipeline treina e compara os seguintes algoritmos:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

A seleção do melhor modelo é feita com base nas métricas de avaliação, priorizando desempenho geral do modelo.

---

## 10. Resultado dos Modelos em DEV

Último resultado validado:

| Modelo             | Accuracy | Precision | Recall | F1 Score | ROC AUC | Melhor Modelo |
| ------------------ | -------: | --------: | -----: | -------: | ------: | ------------- |
| CatBoost           |   0.9388 |    0.9821 | 0.7328 |   0.8393 |  0.9380 | Sim           |
| LightGBM           |   0.9227 |    0.8370 | 0.8017 |   0.8190 |  0.9511 | Não           |
| GradientBoosting   |   0.9276 |    0.9366 | 0.7166 |   0.8120 |  0.9258 | Não           |
| RandomForest       |   0.9122 |    0.8269 | 0.7560 |   0.7899 |  0.9227 | Não           |
| XGBoost            |   0.9062 |    0.7802 | 0.7940 |   0.7870 |  0.9399 | Não           |
| LogisticRegression |   0.8231 |    0.5699 | 0.7714 |   0.6555 |  0.8645 | Não           |

Modelo campeão:

```text
CatBoost
```

Métricas do CatBoost:

```text
Accuracy: 93.88%
Precision: 98.21%
Recall: 73.28%
F1 Score: 83.93%
ROC AUC: 93.80%
```

Observação técnica:

O CatBoost foi escolhido como campeão pelo melhor F1 Score. Porém, o LightGBM apresentou maior ROC AUC e maior Recall, sendo um bom candidato a challenger em um cenário bancário real, onde detectar mais casos de default pode ser mais importante do que maximizar precisão.

---

## 11. Como Executar Localmente no Mac/Linux

### 11.1. Entrar no projeto

```bash
cd ~/Databricks/databricks-banking-credit-risk
```

### 11.2. Limpar variáveis de token manual

Para comandos de bundle, usar o perfil autenticado do Databricks CLI:

```bash
unset DATABRICKS_TOKEN
unset DATABRICKS_HOST
```

Validar login:

```bash
databricks current-user me
```

---

## 12. Criar Schemas e Volumes

Executar no Databricks SQL Editor:

```sql
CREATE SCHEMA IF NOT EXISTS mlops_dev.banking;
CREATE SCHEMA IF NOT EXISTS mlops_acc.banking;
CREATE SCHEMA IF NOT EXISTS mlops_production.banking;

CREATE VOLUME IF NOT EXISTS mlops_dev.banking.raw;
CREATE VOLUME IF NOT EXISTS mlops_acc.banking.raw;
CREATE VOLUME IF NOT EXISTS mlops_production.banking.raw;
```

Validar via CLI:

```bash
databricks volumes list mlops_dev banking
databricks volumes list mlops_acc banking
databricks volumes list mlops_production banking
```

---

## 13. Baixar Dataset do Kaggle

Instalar Kaggle CLI:

```bash
python3 -m pip install kaggle
```

Configurar credencial:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

Baixar dataset:

```bash
mkdir -p data/raw

kaggle datasets download \
  -d laotse/credit-risk-dataset \
  -p data/raw \
  --unzip
```

Validar arquivo:

```bash
ls -lh data/raw
```

O arquivo precisa estar como:

```text
data/raw/credit_risk_dataset.csv
```

Se necessário:

```bash
mv data/raw/*.csv data/raw/credit_risk_dataset.csv
```

---

## 14. Upload do Dataset para os Volumes

### DEV

```bash
databricks fs cp \
  data/raw/credit_risk_dataset.csv \
  dbfs:/Volumes/mlops_dev/banking/raw/credit_risk_dataset.csv \
  --overwrite
```

### ACC

```bash
databricks fs cp \
  data/raw/credit_risk_dataset.csv \
  dbfs:/Volumes/mlops_acc/banking/raw/credit_risk_dataset.csv \
  --overwrite
```

### PROD

```bash
databricks fs cp \
  data/raw/credit_risk_dataset.csv \
  dbfs:/Volumes/mlops_production/banking/raw/credit_risk_dataset.csv \
  --overwrite
```

Validar:

```bash
databricks fs ls dbfs:/Volumes/mlops_dev/banking/raw/
databricks fs ls dbfs:/Volumes/mlops_acc/banking/raw/
databricks fs ls dbfs:/Volumes/mlops_production/banking/raw/
```

---

## 15. Executar Pipeline DEV

```bash
databricks bundle validate -t dev
databricks bundle deploy -t dev --force
databricks bundle run credit_risk_mlops_pipeline -t dev
```

---

## 16. Executar Pipeline ACC

```bash
databricks bundle validate -t acc
databricks bundle deploy -t acc --force
databricks bundle run credit_risk_mlops_pipeline -t acc
```

---

## 17. Executar Pipeline PROD

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod --force
databricks bundle run credit_risk_mlops_pipeline -t prod
```

---

## 18. Configuração Importante do `databricks.yml`

Targets com `mode: production`, como `acc` e `prod`, precisam de `workspace.root_path`.

Exemplo:

```yaml
targets:
  dev:
    mode: development
    default: true
    workspace:
      host: https://dbc-710d747d-f005.cloud.databricks.com

  acc:
    mode: production
    workspace:
      host: https://dbc-710d747d-f005.cloud.databricks.com
      root_path: /Workspace/Users/williamfleandro@gmail.com/.bundle/${bundle.name}/${bundle.target}

  prod:
    mode: production
    workspace:
      host: https://dbc-710d747d-f005.cloud.databricks.com
      root_path: /Workspace/Users/williamfleandro@gmail.com/.bundle/${bundle.name}/${bundle.target}
```

---

## 19. GitHub Actions

O projeto possui workflows separados para:

```text
DEV
ACC
PROD
```

### DEV

Executado em branch de desenvolvimento ou manualmente.

### ACC

Executado em push na `main` ou manualmente.

```yaml
name: Databricks Bundle ACC

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  bundle-acc:
    runs-on: ubuntu-latest
    environment: acc

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install Databricks CLI
        uses: databricks/setup-cli@main

      - name: Validate
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: databricks bundle validate -t acc

      - name: Deploy
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: databricks bundle deploy -t acc

      - name: Run workflow
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: databricks bundle run credit_risk_mlops_pipeline -t acc
```

### PROD

Deve ser executado manualmente com aprovação de ambiente.

---

## 20. Consultas SQL Úteis

### Listar tabelas

```sql
SHOW TABLES IN mlops_dev.banking;
```

### Validar Silver

```sql
SELECT
  COUNT(*) AS total_records,
  MIN(person_age) AS min_age,
  MAX(person_age) AS max_age,
  SUM(CASE WHEN person_age < 18 OR person_age > 100 THEN 1 ELSE 0 END) AS invalid_age_records
FROM mlops_dev.banking.credit_risk_silver;
```

### Métricas do último run

```sql
WITH latest_run AS (
  SELECT MAX(created_at) AS max_created_at
  FROM mlops_dev.banking.credit_risk_model_comparison
)
SELECT
  model_type,
  ROUND(accuracy, 4) AS accuracy,
  ROUND(precision, 4) AS precision,
  ROUND(recall, 4) AS recall,
  ROUND(f1_score, 4) AS f1_score,
  ROUND(roc_auc, 4) AS roc_auc,
  is_best_model,
  mlflow_run_id,
  created_at
FROM mlops_dev.banking.credit_risk_model_comparison
WHERE created_at = (SELECT max_created_at FROM latest_run)
ORDER BY f1_score DESC, roc_auc DESC;
```

### Dataset Silver

```sql
SELECT *
FROM mlops_dev.banking.credit_risk_silver
ORDER BY loan_id
LIMIT 1000;
```

### Dataset com features e predição

```sql
SELECT
  f.loan_id,
  f.person_age,
  f.person_income,
  f.person_emp_length,
  f.person_home_ownership,
  f.loan_intent,
  f.loan_grade,
  f.loan_amnt,
  f.loan_int_rate,
  f.loan_percent_income,
  f.cb_person_default_on_file,
  f.cb_person_cred_hist_length,
  f.debt_to_income_ratio,
  f.income_band,
  f.age_band,
  f.high_interest_rate,
  f.high_loan_income_ratio,
  f.previous_default_flag,
  f.loan_status AS actual_default,
  p.prediction AS predicted_default,
  p.probability_default,
  p.risk_level,
  p.model_type,
  p.prediction_created_at
FROM mlops_dev.banking.credit_risk_features_uc f
LEFT JOIN mlops_dev.banking.credit_risk_batch_predictions p
  ON f.loan_id = p.loan_id
ORDER BY p.probability_default DESC
LIMIT 1000;
```

### Distribuição das predições

```sql
SELECT
  risk_level,
  prediction,
  COUNT(*) AS total_customers,
  ROUND(AVG(probability_default), 4) AS avg_probability_default,
  ROUND(MIN(probability_default), 4) AS min_probability_default,
  ROUND(MAX(probability_default), 4) AS max_probability_default
FROM mlops_dev.banking.credit_risk_batch_predictions
GROUP BY risk_level, prediction
ORDER BY prediction DESC, avg_probability_default DESC;
```

### Top 100 clientes com maior risco

```sql
SELECT
  f.loan_id,
  f.person_age,
  f.person_income,
  f.loan_amnt,
  f.loan_int_rate,
  f.loan_percent_income,
  f.loan_grade,
  f.person_home_ownership,
  f.loan_intent,
  f.loan_status AS actual_default,
  p.prediction AS predicted_default,
  ROUND(p.probability_default, 4) AS probability_default,
  p.risk_level,
  p.model_type
FROM mlops_dev.banking.credit_risk_features_uc f
JOIN mlops_dev.banking.credit_risk_batch_predictions p
  ON f.loan_id = p.loan_id
ORDER BY p.probability_default DESC
LIMIT 100;
```

### Matriz previsto vs real

```sql
SELECT
  loan_status AS actual_default,
  prediction AS predicted_default,
  COUNT(*) AS total
FROM mlops_dev.banking.credit_risk_features_uc f
JOIN mlops_dev.banking.credit_risk_batch_predictions p
  ON f.loan_id = p.loan_id
GROUP BY loan_status, prediction
ORDER BY actual_default, predicted_default;
```

---

## 21. Drift Monitoring

Tabela de drift:

```text
mlops_dev.banking.credit_risk_feature_drift_metrics
```

Consulta:

```sql
SELECT *
FROM mlops_dev.banking.credit_risk_feature_drift_metrics
ORDER BY created_at DESC;
```

Regras gerais:

```text
PSI < 0.10       → OK
0.10 até 0.25    → WARNING
>= 0.25          → DRIFT
```

---

## 22. Retraining Decision

Tabela:

```text
mlops_dev.banking.credit_risk_retraining_requests
```

Consulta:

```sql
SELECT *
FROM mlops_dev.banking.credit_risk_retraining_requests
ORDER BY created_at DESC
LIMIT 20;
```

Possíveis decisões:

```text
RETRAINING_NOT_REQUIRED
RETRAINING_RECOMMENDED
RETRAINING_REQUIRED
```

---

## 23. Drift Approval Gate em PROD

Em produção, caso seja detectado drift, o pipeline pode parar no gate de aprovação.

Tabela:

```text
mlops_production.banking.credit_risk_drift_approval_requests
```

Aprovação manual:

```sql
INSERT INTO mlops_production.banking.credit_risk_drift_approval_requests
SELECT
  uuid() AS approval_id,
  catalog_name,
  schema_name,
  drift_run_created_at,
  drift_features,
  warning_features,
  'APPROVED' AS approval_status,
  'williamfleandro@gmail.com' AS approved_by,
  'Drift aprovado manualmente após revisão. Pipeline seguirá controlado com monitoramento e rastreabilidade.' AS approval_comment,
  CAST(current_timestamp() AS STRING) AS requested_at,
  CAST(current_timestamp() AS STRING) AS approved_at
FROM (
  SELECT *
  FROM mlops_production.banking.credit_risk_drift_approval_requests
  WHERE approval_status = 'PENDING_APPROVAL'
  ORDER BY requested_at DESC
  LIMIT 1
);
```

Depois disso, usar **Repair Run** no Databricks Workflow a partir da task `drift_approval_gate`.

---

## 24. Problemas Corrigidos Durante o Projeto

### 24.1. Arquivo CSV ausente no Volume

Erro:

```text
[PATH_NOT_FOUND] Path does not exist:
dbfs:/Volumes/mlops_dev/banking/raw/credit_risk_dataset.csv
```

Correção:

```bash
databricks fs cp \
  data/raw/credit_risk_dataset.csv \
  dbfs:/Volumes/mlops_dev/banking/raw/credit_risk_dataset.csv \
  --overwrite
```

---

### 24.2. Outliers de idade na Silver

Erro:

```text
DATA_QUALITY_FAILED: ['person_age_valid_range']
```

Causa:

```text
Registros com person_age acima de 100 anos.
```

Correção:

* Tratamento na camada Silver.
* Quarantine de registros inválidos.
* Recriação da Silver limpa.
* Validação com `invalid_age_records = 0`.

---

### 24.3. Job usando notebook antigo

Sintoma:

```text
Mesmo após alterar o notebook local, o Workflow continuava executando versão antiga.
```

Correção:

* Criação de notebooks `_v2`.
* Atualização do `resources/credit_risk_job.yml`.
* Deploy forçado com:

```bash
databricks bundle deploy -t dev --force
```

---

### 24.4. Erro de schema no Spark Connect

Erro:

```text
[CANNOT_DETERMINE_TYPE] Some of types cannot be determined after inferring.
```

Causa:

```text
spark.createDataFrame(row) com campos None/listas sem schema explícito.
```

Correção:

```text
Uso de StructType explícito no notebook 13_retraining_decision_v2.py.
```

---

### 24.5. CatBoost ausente no Batch Inference

Erro:

```text
ModuleNotFoundError: No module named 'catboost'
```

Causa:

```text
O modelo campeão foi CatBoost, mas a task de inferência não tinha a biblioteca instalada.
```

Correção:

```python
# MAGIC %pip install -q xgboost lightgbm catboost
dbutils.library.restartPython()
```

Adicionado aos notebooks de batch inference e serving registration.

---

### 24.6. Erro em targets production do Databricks Bundle

Erro:

```text
target with 'mode: production' must set 'workspace.root_path'
```

Correção:

```yaml
root_path: /Workspace/Users/williamfleandro@gmail.com/.bundle/${bundle.name}/${bundle.target}
```

Adicionado aos targets `acc` e `prod`.

---

## 25. Git Workflow

Criar branch:

```bash
git checkout -b feature/credit-risk-mlops-pipeline
```

Adicionar arquivos:

```bash
git add .
```

Commit:

```bash
git commit -m "Add Databricks banking credit risk MLOps pipeline"
```

Push:

```bash
git push -u origin feature/credit-risk-mlops-pipeline
```

Criar Pull Request:

```bash
gh pr create \
  --repo williamfleandro/databricks-banking-credit-risk \
  --base main \
  --head feature/credit-risk-mlops-pipeline \
  --title "Add Databricks Banking Credit Risk MLOps Pipeline" \
  --body "Adds the complete Databricks Asset Bundle project for banking credit risk, including Lakehouse Bronze/Silver/Gold layers, Data Quality, Feature Table, MLflow model training, batch inference, drift monitoring, retraining decision, and ACC/PROD deployment configuration."
```

---

## 26. Status Atual

```text
DEV: executado com sucesso
ACC: preparado para execução
PROD: preparado para execução
```

Principais validações em DEV:

```text
Silver Records: 32576
Min Age: 20
Max Age: 94
Invalid Age Records: 0
Best Model: CatBoost
Best F1 Score: 0.8393
Best Accuracy: 0.9388
Best Precision: 0.9821
Best ROC AUC: 0.9380
```

---

## 27. Próximos Passos

* Executar pipeline em ACC.
* Validar métricas em ACC.
* Executar pipeline em PROD.
* Validar drift approval gate em PROD.
* Criar endpoint de serving produtivo.
* Testar payload real no endpoint.
* Documentar arquitetura final com diagrama.
* Adicionar este projeto ao portfólio principal.
* Criar projeto complementar de Fraud Detection com Databricks.

---

## 28. Autor

William Ferreira Leandro

GitHub:

```text
https://github.com/williamfleandro
```

Projeto:

```text
Databricks Banking Credit Risk MLOps Pipeline
```


