# Customer Churn Prediction – MLOps Project

## 📌 Project Overview

This project implements an end-to-end **Customer Churn Prediction MLOps pipeline** using machine learning and MLOps practices.

The objective is to predict whether a customer is likely to **churn** based on customer demographic, service, contract, billing, and account-related information.

The project covers the complete machine learning and MLOps lifecycle:

* Data preprocessing
* Data validation
* Processed-output validation
* Model training
* Model evaluation
* Experiment tracking
* Model reproducibility
* Model registry
* Model versioning
* Model quality gates
* Automated model lifecycle management
* Production inference
* Failure injection testing
* Git version control
* GitHub Actions automation
* Generation of reports and artifacts

---

## 🎯 Problem Statement

Customer churn is an important business problem where customers stop using a company's products or services.

The goal of this project is to develop a machine learning system that can:

1. Process customer churn data.
2. Prepare the data for machine learning.
3. Validate the raw dataset.
4. Train a classification model.
5. Evaluate model performance.
6. Track experiments using MLflow.
7. Validate processed outputs.
8. Register trained models.
9. Apply an automated model quality gate.
10. Compare Staging and Production models.
11. Automatically promote or archive model versions.
12. Generate Production predictions.
13. Test failure scenarios.
14. Maintain reproducibility.
15. Automate the workflow through GitHub Actions.

---

## 🧠 Machine Learning Problem

This is a **binary classification problem**.

### Target Variable

The target variable is:

```text
Churn
```

The values are converted as follows:

| Original Value | Encoded Value |
| -------------- | ------------: |
| No             |             0 |
| Yes            |             1 |

Where:

* `0` → Customer does not churn
* `1` → Customer churns

---

## 📊 Dataset

The project uses a customer churn dataset stored at:

```text
data/raw/churn.csv
```

The dataset contains customer-related information such as:

* Customer demographics
* Account information
* Contract information
* Internet services
* Payment information
* Monthly charges
* Total charges
* Churn status

The `customerID` column is removed during preprocessing because it is an identifier rather than a predictive feature.

### Dataset Size

The successfully executed pipeline loaded:

```text
Rows    : 7044
Columns : 21
```

---

# 🏗️ Project Architecture

```text
Customer_Churn/

│
├── .github/
│   └── workflows/
│       ├── lab8_ci.yml
│       └── lab9_failure_tests.yml
│
├── data/
│   ├── raw/
│   │   └── churn.csv
│   │
│   └── processed/
│       ├── X_train_final.npy
│       ├── X_test_final.npy
│       ├── y_train.npy
│       ├── y_test.npy
│       ├── X_train_raw.csv
│       ├── X_test_raw.csv
│       └── dataset_metadata.json
│
├── notebooks/
│   └── project_implementation.ipynb
│
├── pipelines/
│   ├── run_lab3_baseline.py
│   ├── run_lab4_tracking.py
│   ├── run_lab5_pipeline.py
│   └── run_lab6_registry.py
│
├── src/
│   ├── mlops_pipeline/
│   │   ├── 01_load_data.py
│   │   ├── 02_validate_data.py
│   │   ├── 03_preprocessing.py
│   │   ├── 04_validate_outputs.py
│   │   ├── 05_train_registry.py
│   │   ├── 06_model_evaluation.py
│   │   ├── 07_quality_gate.py
│   │   ├── 08_automate_lifecycle.py
│   │   ├── 09_inference.py
│   │   ├── 10_failure_injection.py
│   │   └── churn_prediction_pipeline.py
│   │
│   ├── automate_lifecycle.py
│   ├── evaluate.py
│   ├── generate_registry_report.py
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── train.py
│   ├── train_registry.py
│   ├── validate_data.py
│   └── validate_outputs.py
│
├── models/
│   └── Generated model files
│
├── outputs/
│   └── Generated prediction and evaluation outputs
│
├── logs/
│   └── Execution logs
│
├── artifacts/
│   └── Generated reports and artifacts
│
├── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔄 MLOps Workflow

The complete production pipeline is:

```text
Raw Dataset
     │
     ▼
1. Data Loading
     │
     ▼
2. Raw Data Validation
     │
     ▼
3. Data Preprocessing
     │
     ▼
4. Processed Output Validation
     │
     ▼
5. Model Training + MLflow + Model Registry
     │
     ▼
6. Model Evaluation
     │
     ▼
7. Model Quality Gate
     │
     ├── FAIL → Pipeline Stops
     │
     ▼
8. Automated Model Lifecycle
     │
     ├── Better Model → Production
     │
     └── Not Better → Archive
     │
     ▼
9. Production Inference
```

---

# ⚙️ Technologies Used

| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| Python           | Programming language                     |
| Pandas           | Data manipulation                        |
| NumPy            | Numerical computation                    |
| Scikit-learn     | Machine learning                         |
| Joblib           | Model serialization                      |
| MLflow           | Experiment tracking and model management |
| SQLite           | MLflow tracking backend                  |
| Pandera          | Data validation                          |
| PyYAML           | Configuration management                 |
| Jupyter Notebook | Exploratory analysis                     |
| Git              | Version control                          |
| GitHub           | Source-code hosting                      |
| GitHub Actions   | CI automation                            |
| VS Code          | Development environment                  |

---

# 🐍 Python Environment

It is recommended to use a Python virtual environment.

Create the environment:

```bash
python -m venv .venv
```

On Windows, if the Python Launcher is available:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

After activation, the terminal should show:

```text
(.venv)
```

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Tejasai846/Customer_Churn.git
```

Move into the project directory:

```bash
cd Customer_Churn
```

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# 📁 Dataset Location

Place the dataset in:

```text
data/raw/churn.csv
```

Expected structure:

```text
data/
└── raw/
    └── churn.csv
```

The dataset path is configured in:

```text
config.yaml
```

---

# ⚙️ Configuration

The MLOps pipeline uses `config.yaml` for dataset, preprocessing, model, MLflow, registry, and quality-gate configuration.

The current configuration is:

```yaml
data:
  raw_path: "data/raw/churn.csv"
  processed_dir: "data/processed"
  target_column: "Churn"
  id_column: "customerID"

split:
  test_size: 0.2
  random_state: 42

paths:
  preprocessor: "models/preprocessor.pkl"
  model: "models/random_forest_model.pkl"
  evaluation_report: "artifacts/model_evaluation_report.json"
  quality_gate_report: "artifacts/quality_gate_report.json"
  inference_output: "outputs/production_inference.csv"

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "Customer_Churn_MLOps"

model:
  n_estimators: 100
  max_depth: 10
  random_state: 42
  class_weight: "balanced"

registry:
  model_name: "Customer_Churn_Model"
  decision_metric: "recall"

quality_gate:
  metric: "recall"
  minimum_value: 0.70
```

---

# 🧹 Data Preprocessing

The preprocessing process performs several operations:

1. Loads the raw dataset.
2. Removes the `customerID` identifier.
3. Converts `TotalCharges` to numeric format.
4. Handles missing values.
5. Converts the target variable into numerical values.
6. Performs a stratified train/test split.
7. Identifies numerical and categorical features.
8. Applies numerical preprocessing.
9. Applies categorical preprocessing.
10. Generates final machine learning datasets.
11. Saves the preprocessing pipeline.

The MLOps preprocessing stage is:

```bash
python src/mlops_pipeline/03_preprocessing.py
```

The preprocessing pipeline is saved to:

```text
models/preprocessor.pkl
```

---

# 🤖 Model Training

The project uses a **Random Forest Classifier**.

Current model configuration:

```text
n_estimators = 100
max_depth = 10
random_state = 42
class_weight = balanced
```

The complete MLOps training stage is:

```bash
python src/mlops_pipeline/05_train_registry.py
```

The trained model is also saved locally under:

```text
models/
```

---

# 📈 Model Evaluation

The evaluation stage calculates:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Run:

```bash
python src/mlops_pipeline/06_model_evaluation.py
```

The evaluation report is generated at:

```text
artifacts/model_evaluation_report.json
```

---

# 📊 Model Performance

The successfully executed MLOps pipeline produced:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 75.59% |
| Precision | 52.76% |
| Recall    | 76.74% |
| F1 Score  | 62.53% |
| ROC-AUC   | 84.33% |

Confusion Matrix:

```text
[[778 257]
 [ 87 287]]
```

---

# 🔬 Experiment Tracking

MLflow is used for experiment tracking and model management.

The project uses:

```text
Experiment:
Customer_Churn_MLOps
```

MLflow uses SQLite as the local tracking backend:

```text
sqlite:///mlflow.db
```

MLflow tracks:

* Parameters
* Metrics
* Model runs
* Artifacts
* Model versions
* Registered models

---

# 🗂️ Model Registry

The registered model is:

```text
Customer_Churn_Model
```

The training stage automatically creates model versions in MLflow Model Registry.

Example model versions generated during project execution include:

```text
Version 1
Version 2
Version 3
Version 4
Version 5
```

The registry supports:

* Model registration
* Model versioning
* Model comparison
* Staging
* Production
* Model lifecycle management
* Model archiving

---

# 🚦 Model Quality Gate

The project includes an automated **Model Quality Gate**.

The configured decision metric is:

```text
Recall
```

Minimum required value:

```text
0.70
```

The successfully trained model achieved:

```text
Recall = 0.7674
```

Therefore:

```text
Quality Gate → PASSED
```

Run the quality gate separately with:

```bash
python src/mlops_pipeline/07_quality_gate.py
```

If the model does not satisfy the configured threshold, the pipeline stops before model promotion.

---

# 🔄 Model Lifecycle Automation

The project contains automated model lifecycle management.

Run:

```bash
python src/mlops_pipeline/08_automate_lifecycle.py
```

The lifecycle process:

1. Identifies newly registered models.
2. Moves new versions to Staging.
3. Identifies the current Production model.
4. Compares the Staging model with Production.
5. Uses Recall as the decision metric.
6. Promotes a better model.
7. Keeps Production unchanged when the challenger does not improve.
8. Archives unsuccessful model versions.

Example from the successful pipeline execution:

```text
Production Version 1 | recall=0.7674
Staging Version 5    | recall=0.7674
```

The Production model remained unchanged because the new version did not improve the Production metric.

---

# 🔮 Production Inference

The Production inference stage loads the model from MLflow Model Registry:

```text
models:/Customer_Churn_Model/Production
```

Run:

```bash
python src/mlops_pipeline/09_inference.py
```

Example predictions:

```text
 Predicted_Churn  Churn_Probability
               0             0.4847
               1             0.8223
               1             0.8520
               0             0.4884
               1             0.6065
```

The generated predictions are saved to:

```text
outputs/production_inference.csv
```

---

# 🧪 Failure Injection Testing

The project includes failure-injection tests to verify how the MLOps pipeline handles failures.

Run:

```bash
python src/mlops_pipeline/10_failure_injection.py
```

The failure tests cover scenarios including:

* Missing data
* Invalid data types
* Schema mismatch
* Preprocessing inconsistency
* Missing model/recovery scenarios

This helps verify that the pipeline fails safely instead of silently producing invalid results.

---

# 🚀 Complete MLOps Pipeline

The main pipeline runner is:

```text
src/mlops_pipeline/churn_prediction_pipeline.py
```

Run the complete pipeline using:

```powershell
python src/mlops_pipeline/churn_prediction_pipeline.py
```

The pipeline executes:

```text
1. Data Loading
2. Raw Data Validation
3. Preprocessing
4. Processed Output Validation
5. Training + MLflow + Model Registry
6. Model Evaluation
7. Model Quality Gate
8. Automated Model Lifecycle
9. Production Inference
```

If any stage fails, the pipeline stops automatically.

A successful execution produces:

```text
[SUCCESS] Churn Prediction MLOps Pipeline completed successfully.
```

The complete pipeline run log is saved to:

```text
logs/churn_prediction_pipeline_run_log.json
```

---

# 🔄 Baseline Pipeline

The original baseline pipeline is also available.

Run:

```bash
python pipelines/run_lab3_baseline.py
```

The baseline workflow follows:

```text
Preprocessing
     ↓
Training
     ↓
Evaluation
```

---

# 📊 Experiment Tracking Pipeline

The experiment tracking workflow is available through:

```bash
python pipelines/run_lab4_tracking.py
```

It demonstrates experiment tracking and model-related logging.

---

# 🚀 Extended MLOps Pipeline

The extended pipeline can be run using:

```bash
python pipelines/run_lab5_pipeline.py
```

This provides the earlier project workflow in addition to the newer `src/mlops_pipeline/` implementation.

---

# 🗂️ Registry Pipeline

The registry workflow is available through:

```bash
python pipelines/run_lab6_registry.py
```

It demonstrates model registration and registry-related operations.

---

# 🔍 Data Validation

The project includes data validation functionality.

The original validation script can be executed using:

```bash
python src/validate_data.py
```

The newer complete MLOps pipeline performs raw data validation through:

```bash
python src/mlops_pipeline/02_validate_data.py
```

Data validation helps detect:

* Missing columns
* Unexpected columns
* Incorrect data types
* Invalid values
* Schema inconsistencies

---

# ✅ Output Validation

The original output validation script is:

```bash
python src/validate_outputs.py
```

The complete MLOps pipeline uses:

```bash
python src/mlops_pipeline/04_validate_outputs.py
```

The validation checks whether the expected preprocessing outputs were generated correctly.

---

# 🔁 Reproducibility

The project uses deterministic preprocessing and model configuration.

Important reproducibility settings include:

```text
Random State = 42
Test Size    = 0.20
```

The same preprocessing logic is used for the training and inference workflow.

The preprocessing pipeline is serialized using Joblib:

```text
models/preprocessor.pkl
```

This helps maintain consistent transformations between training and inference.

---

# 📋 Important Files

## `config.yaml`

Contains:

* Dataset paths
* Target and ID columns
* Train/test configuration
* Model configuration
* MLflow configuration
* Registry configuration
* Quality-gate configuration
* Output paths

---

## `src/mlops_pipeline/01_load_data.py`

Responsible for:

* Loading the raw dataset
* Checking dataset availability
* Generating the data loading report

---

## `src/mlops_pipeline/02_validate_data.py`

Responsible for:

* Schema validation
* Raw dataset validation
* Detecting invalid data

---

## `src/mlops_pipeline/03_preprocessing.py`

Responsible for:

* Data cleaning
* Target encoding
* Train/test splitting
* Numerical preprocessing
* Categorical preprocessing
* Saving processed datasets
* Saving the preprocessing pipeline

---

## `src/mlops_pipeline/04_validate_outputs.py`

Responsible for validating:

* Processed arrays
* Data shapes
* Missing values
* Raw train/test files
* Preprocessing outputs

---

## `src/mlops_pipeline/05_train_registry.py`

Responsible for:

* Random Forest training
* MLflow experiment tracking
* Metric logging
* Model serialization
* Model registration

---

## `src/mlops_pipeline/06_model_evaluation.py`

Responsible for:

* Loading the trained model
* Generating predictions
* Calculating evaluation metrics
* Creating the confusion matrix
* Saving the evaluation report

---

## `src/mlops_pipeline/07_quality_gate.py`

Responsible for:

* Reading evaluation results
* Checking the required metric
* Comparing the metric against the minimum threshold
* Generating the quality-gate report

---

## `src/mlops_pipeline/08_automate_lifecycle.py`

Responsible for:

* Staging model versions
* Comparing Staging and Production
* Promoting better models
* Maintaining the existing Production model when appropriate
* Archiving unsuccessful model versions

---

## `src/mlops_pipeline/09_inference.py`

Responsible for:

* Loading the Production model
* Transforming customer data
* Generating churn predictions
* Generating churn probabilities
* Saving Production inference results

---

## `src/mlops_pipeline/10_failure_injection.py`

Responsible for testing:

* Missing-data failures
* Invalid datatype failures
* Schema failures
* Preprocessing failures
* Model/recovery failures

---

## `src/mlops_pipeline/churn_prediction_pipeline.py`

The main orchestration script that executes the complete nine-stage MLOps pipeline.

---

# 🧪 Running the Complete Workflow

### Step 1 — Activate environment

```powershell
.venv\Scripts\activate
```

### Step 2 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3 — Run the complete pipeline

```powershell
python src/mlops_pipeline/churn_prediction_pipeline.py
```

### Step 4 — Run failure tests

```powershell
python src/mlops_pipeline/10_failure_injection.py
```

---

# 📊 Evaluation Metrics

The project evaluates the model using:

### Accuracy

Measures the proportion of correct predictions among all predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many customers predicted as churners actually churned.

```text
Precision = TP / (TP + FP)
```

### Recall

Measures how many actual churners were correctly identified.

```text
Recall = TP / (TP + FN)
```

### F1 Score

The F1 score combines precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers across classification thresholds.

---

# ⚠️ False Positives and False Negatives

The evaluation workflow can generate separate files for:

```text
outputs/false_positives.csv
outputs/false_negatives.csv
```

### False Positive

A customer is predicted as likely to churn but actually does not churn.

### False Negative

A customer is predicted as not churning but actually churns.

---

# 📦 Generated Files

Depending on the pipeline stage, the project can generate:

```text
data/processed/
├── X_train_final.npy
├── X_test_final.npy
├── y_train.npy
├── y_test.npy
├── X_train_raw.csv
├── X_test_raw.csv
└── dataset_metadata.json

models/
├── preprocessor.pkl
└── random_forest_model.pkl

outputs/
└── production_inference.csv

artifacts/
├── data_load_report.json
├── preprocessing_summary_report.json
├── model_evaluation_report.json
└── quality_gate_report.json

logs/
└── churn_prediction_pipeline_run_log.json

mlflow.db
```

Generated files that are not intended for source control are excluded using `.gitignore`.

---

# 🤖 GitHub Actions

The project includes GitHub Actions workflows under:

```text
.github/workflows/
```

## Lab 8 CI Workflow

```text
.github/workflows/lab8_ci.yml
```

The workflow installs dependencies and executes the main MLOps pipeline.

It can also upload generated artifacts, reports, logs, outputs, and related project results.

## Lab 9 Failure Testing Workflow

```text
.github/workflows/lab9_failure_tests.yml
```

This workflow runs the normal pipeline and then executes the failure-injection tests.

---

# 🔐 Git and GitHub

The project uses Git for version control.

Initialize Git:

```bash
git init
```

Check status:

```bash
git status
```

Add files:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Add complete MLOps pipeline and CI workflows"
```

Rename the branch:

```bash
git branch -M main
```

Connect the GitHub repository:

```bash
git remote add origin https://github.com/Tejasai846/Customer_Churn.git
```

Push the project:

```bash
git push -u origin main
```

Repository:

https://github.com/Tejasai846/Customer_Churn

---

# 🚫 Git Ignore

The project uses `.gitignore` to avoid committing unnecessary generated files such as:

```text
.venv/
venv/
__pycache__/
.ipynb_checkpoints/
data/processed/*.npy
models/*.pkl
logs/
outputs/
mlruns/
mlflow.db
artifacts/
.vscode/
```

This keeps the Git repository clean and prevents generated files from being unnecessarily tracked.

---

# 📚 Project Learning Outcomes

Through this project, the following concepts are demonstrated:

* Machine Learning classification
* Data preprocessing
* Feature engineering
* Train/test splitting
* Random Forest classification
* Model evaluation
* Precision and recall
* F1 score
* ROC-AUC
* Data validation
* Processed-output validation
* Experiment tracking
* Reproducibility
* Model versioning
* Model registry
* Quality gates
* Staging and Production models
* Model lifecycle management
* Champion/challenger comparison
* Production inference
* Failure injection
* MLOps pipeline automation
* Git version control
* GitHub repository management
* GitHub Actions

---

# 🔮 Future Enhancements

Possible future improvements include:

* Develop a web-based prediction application.
* Add a REST API for predictions.
* Add Docker containerization.
* Deploy the model to a cloud platform.
* Add automated model retraining.
* Add monitoring for model performance.
* Add data drift detection.
* Add model drift detection.
* Create an interactive dashboard.
* Add automated unit and integration testing.
* Add production monitoring and alerting.
* Migrate deprecated MLflow model-stage APIs to the newer MLflow model alias approach.

---

# 👨‍💻 Project Author

**Tejasai846**

GitHub:

https://github.com/Tejasai846

Repository:

https://github.com/Tejasai846/Customer_Churn

---

# 📜 License

This project is intended for **educational and academic purposes**.

---

# ⭐ Project Summary

This project demonstrates an end-to-end **Customer Churn Prediction machine learning system using MLOps practices**.

The workflow combines:

```text
Data
 ↓
Validation
 ↓
Preprocessing
 ↓
Training
 ↓
MLflow Tracking
 ↓
Model Registry
 ↓
Evaluation
 ↓
Quality Gate
 ↓
Model Lifecycle
 ↓
Production Inference
```

The complete nine-stage pipeline has been successfully executed locally, with all stages passing successfully.

The project therefore demonstrates not only machine learning model development, but also the practical MLOps processes required to validate, track, register, evaluate, promote, and deploy machine learning models.
