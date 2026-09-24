\# Customer Churn Prediction – MLOps Project



\## 📌 Project Overview



This project implements an end-to-end \*\*Customer Churn Prediction MLOps pipeline\*\* using machine learning and MLOps practices.



The objective is to predict whether a customer is likely to \*\*churn\*\* based on customer demographic, service, contract, billing, and account-related information.



The project covers the complete machine learning lifecycle:



\* Data preprocessing

\* Data validation

\* Model training

\* Model evaluation

\* Experiment tracking

\* Model reproducibility

\* Model registry

\* Model lifecycle automation

\* Model validation

\* Pipeline execution

\* Generation of reports and artifacts



\---



\## 🎯 Problem Statement



Customer churn is an important business problem where customers stop using a company's products or services.



The goal of this project is to develop a machine learning system that can:



1\. Process customer churn data.

2\. Prepare the data for machine learning.

3\. Train a classification model.

4\. Evaluate model performance.

5\. Track experiments and model versions.

6\. Validate data and model outputs.

7\. Register trained models.

8\. Automate model lifecycle management.

9\. Maintain reproducibility.

10\. Identify customers who are likely to churn.



\---



\## 🧠 Machine Learning Problem



This is a \*\*binary classification problem\*\*.



\### Target Variable



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



\* `0` → Customer does not churn

\* `1` → Customer churns



\---



\## 📊 Dataset



The project uses a customer churn dataset stored at:



```text

data/raw/churn.csv

```



The dataset contains customer-related information such as:



\* Customer demographics

\* Account information

\* Contract information

\* Internet services

\* Payment information

\* Monthly charges

\* Total charges

\* Churn status



The `customerID` column is removed during preprocessing because it is an identifier rather than a predictive feature.



\---



\# 🏗️ Project Architecture



```text

Customer\_Churn/

│

├── data/

│   ├── raw/

│   │   └── churn.csv

│   │

│   └── processed/

│       └── dataset\_metadata.json

│

├── notebooks/

│   └── project\_implementation.ipynb

│

├── pipelines/

│   ├── run\_lab3\_baseline.py

│   ├── run\_lab4\_tracking.py

│   ├── run\_lab5\_pipeline.py

│   └── run\_lab6\_registry.py

│

├── src/

│   ├── automate\_lifecycle.py

│   ├── evaluate.py

│   ├── generate\_registry\_report.py

│   ├── preprocess.py

│   ├── preprocess\_pipeline.py

│   ├── train.py

│   ├── train\_registry.py

│   ├── validate\_data.py

│   └── validate\_outputs.py

│

├── models/

│   └── Generated model files

│

├── outputs/

│   └── Evaluation outputs

│

├── logs/

│   └── Execution logs

│

├── artifacts/

│   └── Generated reports and artifacts

│

├── requirements.txt

├── .gitignore

└── README.md

```



\---



\# 🔄 MLOps Workflow



The overall workflow is:



```text

Raw Dataset

&#x20;    │

&#x20;    ▼

Data Validation

&#x20;    │

&#x20;    ▼

Data Preprocessing

&#x20;    │

&#x20;    ▼

Train/Test Split

&#x20;    │

&#x20;    ▼

Model Training

&#x20;    │

&#x20;    ▼

Model Evaluation

&#x20;    │

&#x20;    ▼

Experiment Tracking

&#x20;    │

&#x20;    ▼

Model Validation

&#x20;    │

&#x20;    ▼

Model Registry

&#x20;    │

&#x20;    ▼

Model Lifecycle Management

&#x20;    │

&#x20;    ▼

Production Model

```



\---



\# ⚙️ Technologies Used



| Technology       | Purpose                                  |

| ---------------- | ---------------------------------------- |

| Python           | Programming language                     |

| Pandas           | Data manipulation                        |

| NumPy            | Numerical computation                    |

| Scikit-learn     | Machine learning                         |

| Joblib           | Model serialization                      |

| MLflow           | Experiment tracking and model management |

| Pandera          | Data validation                          |

| Jupyter Notebook | Exploratory analysis                     |

| Git              | Version control                          |

| GitHub           | Source-code hosting                      |

| VS Code          | Development environment                  |



\---



\# 🐍 Python Environment



It is recommended to use a Python virtual environment.



Create the environment:



```bash

python -m venv .venv

```



Activate it on Windows:



```powershell

.venv\\Scripts\\activate

```



After activation, the terminal should show:



```text

(.venv)

```



\---



\# 📦 Installation



Clone the repository:



```bash

git clone https://github.com/Tejasai846/Customer\_Churn.git

```



Move into the project directory:



```bash

cd Customer\_Churn

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate the environment:



```powershell

.venv\\Scripts\\activate

```



Install the required packages:



```bash

pip install -r requirements.txt

```



\---



\# 📁 Dataset Location



Place the dataset in:



```text

data/raw/churn.csv

```



The expected structure is:



```text

data/

└── raw/

&#x20;   └── churn.csv

```



\---



\# 🧹 Data Preprocessing



The preprocessing process performs several operations:



1\. Loads the raw dataset.

2\. Removes the `customerID` identifier.

3\. Converts `TotalCharges` to numeric format.

4\. Handles missing values.

5\. Converts the target variable into numerical values.

6\. Performs a stratified train/test split.

7\. Identifies numerical and categorical features.

8\. Applies numerical preprocessing.

9\. Applies categorical preprocessing.

10\. Generates the final machine learning datasets.

11\. Saves the preprocessing pipeline.



Run preprocessing using:



```bash

python src/preprocess.py

```



\---



\# 🤖 Model Training



The project uses a \*\*Random Forest Classifier\*\* as the baseline machine learning model.



The training process uses the processed training dataset and generates a trained model.



Run:



```bash

python src/train.py

```



The trained model is stored in the `models/` directory.



\---



\# 📈 Model Evaluation



The evaluation script loads the test dataset and trained model and calculates classification metrics.



The following metrics are evaluated:



\* Accuracy

\* Precision

\* Recall

\* F1 Score



Run:



```bash

python src/evaluate.py

```



The evaluation process also generates false-positive and false-negative outputs.



\---



\# 🔬 Baseline Pipeline



The baseline pipeline executes the preprocessing, training, and evaluation stages sequentially.



Run:



```bash

python pipelines/run\_lab3\_baseline.py

```



The pipeline follows:



```text

Preprocessing

&#x20;    ↓

Training

&#x20;    ↓

Evaluation

```



If one stage fails, the pipeline stops.



\---



\# 📊 Experiment Tracking



MLflow is used for experiment tracking and model management.



The tracking pipeline can be executed using:



```bash

python pipelines/run\_lab4\_tracking.py

```



MLflow can be used to track:



\* Parameters

\* Metrics

\* Model versions

\* Experiments

\* Artifacts



\---



\# 🔁 Reproducibility



The project includes reproducibility checks to ensure that model training produces consistent results when the same configuration and random seed are used.



The Random Forest configuration uses fixed parameters and a fixed random state.



A reproducibility report can be generated and stored under:



```text

artifacts/

```



This helps verify that the machine learning experiment can be reproduced.



\---



\# 🛡️ Data Validation



The project includes data validation functionality.



The validation process checks the dataset against the expected schema and data types.



Run:



```bash

python src/validate\_data.py

```



Data validation helps detect:



\* Missing columns

\* Unexpected columns

\* Incorrect data types

\* Invalid values

\* Schema inconsistencies



\---



\# ✅ Output Validation



The project also contains output validation functionality.



Run:



```bash

python src/validate\_outputs.py

```



This helps verify that the expected outputs have been generated correctly.



\---



\# 🚀 MLOps Pipeline



The project includes additional pipeline stages for MLOps automation.



Run:



```bash

python pipelines/run\_lab5\_pipeline.py

```



This pipeline integrates multiple stages of the machine learning workflow.



\---



\# 🗂️ Model Registry



The project includes model registry functionality.



Run:



```bash

python pipelines/run\_lab6\_registry.py

```



The registry workflow supports:



\* Model registration

\* Model versioning

\* Model comparison

\* Model lifecycle management

\* Production model selection



\---



\# 🔄 Model Lifecycle Automation



The project contains an automated model lifecycle process.



The lifecycle automation can:



1\. Identify the current production model.

2\. Evaluate a challenger model.

3\. Compare model performance.

4\. Determine whether the challenger satisfies the required criteria.

5\. Maintain the current production model when the challenger does not meet the required condition.

6\. Archive unsuccessful model versions.

7\. Generate production model reports.



The lifecycle automation script is:



```bash

python src/automate\_lifecycle.py

```



\---



\# 📋 Registry Report



A registry report can be generated using:



```bash

python src/generate\_registry\_report.py

```



The report provides information about model versions and registry-related information.



\---



\# 🔍 Prediction Workflow



The general prediction workflow is:



```text

Customer Data

&#x20;    ↓

Preprocessing Pipeline

&#x20;    ↓

Trained Random Forest Model

&#x20;    ↓

Prediction

&#x20;    ↓

Churn / No Churn

```



Prediction results can be interpreted as:



```text

0 → No Churn

1 → Churn

```



\---



\# 📂 Important Files



\## `src/preprocess.py`



Responsible for:



\* Loading the dataset

\* Cleaning the data

\* Encoding categorical features

\* Handling missing values

\* Scaling numerical features

\* Creating training and testing datasets



\---



\## `src/train.py`



Responsible for:



\* Loading processed training data

\* Training the Random Forest model

\* Saving the trained model



\---



\## `src/evaluate.py`



Responsible for:



\* Loading the test dataset

\* Loading the trained model

\* Generating predictions

\* Calculating evaluation metrics

\* Generating false-positive and false-negative reports



\---



\## `src/validate\_data.py`



Responsible for validating the input dataset against the expected schema.



\---



\## `src/validate\_outputs.py`



Responsible for validating generated project outputs.



\---



\## `src/automate\_lifecycle.py`



Responsible for automated model lifecycle management and comparison of model versions.



\---



\## `pipelines/run\_lab3\_baseline.py`



Runs the baseline preprocessing, training, and evaluation workflow.



\---



\## `pipelines/run\_lab4\_tracking.py`



Runs the experiment tracking workflow.



\---



\## `pipelines/run\_lab5\_pipeline.py`



Runs the extended MLOps pipeline.



\---



\## `pipelines/run\_lab6\_registry.py`



Runs the model registry workflow.



\---



\# 🧪 Running the Complete Workflow



\### Step 1 — Activate environment



```powershell

.venv\\Scripts\\activate

```



\### Step 2 — Install dependencies



```powershell

pip install -r requirements.txt

```



\### Step 3 — Validate data



```powershell

python src/validate\_data.py

```



\### Step 4 — Preprocess data



```powershell

python src/preprocess.py

```



\### Step 5 — Train model



```powershell

python src/train.py

```



\### Step 6 — Evaluate model



```powershell

python src/evaluate.py

```



\### Step 7 — Run baseline pipeline



```powershell

python pipelines/run\_lab3\_baseline.py

```



\### Step 8 — Run experiment tracking



```powershell

python pipelines/run\_lab4\_tracking.py

```



\### Step 9 — Run MLOps pipeline



```powershell

python pipelines/run\_lab5\_pipeline.py

```



\### Step 10 — Run model registry



```powershell

python pipelines/run\_lab6\_registry.py

```



\---



\# 📊 Evaluation Metrics



The project evaluates the model using:



\### Accuracy



Measures the proportion of correct predictions among all predictions.



```text

Accuracy = Correct Predictions / Total Predictions

```



\### Precision



Measures how many customers predicted as churners actually churned.



```text

Precision = TP / (TP + FP)

```



\### Recall



Measures how many actual churners were correctly identified.



```text

Recall = TP / (TP + FN)

```



\### F1 Score



The F1 score combines precision and recall.



```text

F1 = 2 × (Precision × Recall) / (Precision + Recall)

```



\---



\# ⚠️ False Positives and False Negatives



The evaluation stage generates separate files for:



```text

outputs/false\_positives.csv

outputs/false\_negatives.csv

```



These files can be used to analyze incorrect model predictions.



\### False Positive



A customer is predicted as likely to churn but actually does not churn.



\### False Negative



A customer is predicted as not churning but actually churns.



\---



\# 📦 Generated Files



Depending on the pipeline stage, the project can generate:



```text

data/processed/

├── X\_train\_final.npy

├── X\_test\_final.npy

├── y\_train.npy

├── y\_test.npy

└── dataset\_metadata.json



models/

├── preprocessor.pkl

└── random\_forest\_baseline.pkl



outputs/

├── false\_positives.csv

└── false\_negatives.csv



artifacts/

└── Generated reports and model artifacts

```



Generated files that are not intended for source control are excluded using `.gitignore`.



\---



\# 🔐 Git and GitHub



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

git commit -m "Initial Customer Churn MLOps project"

```



Rename the branch to `main`:



```bash

git branch -M main

```



Connect the GitHub repository:



```bash

git remote add origin https://github.com/Tejasai846/Customer\_Churn.git

```



Push the project:



```bash

git push -u origin main

```



\---



\# 🚫 Git Ignore



The project uses `.gitignore` to avoid committing unnecessary generated files such as:



```text

.venv/

venv/

\_\_pycache\_\_/

.ipynb\_checkpoints/

data/processed/\*.npy

models/\*.pkl

logs/

outputs/

mlruns/

mlflow.db

artifacts/

.vscode/

```



This keeps the Git repository clean and prevents large generated files from being unnecessarily tracked.



\---



\# 📚 Project Learning Outcomes



Through this project, the following concepts are demonstrated:



\* Machine Learning classification

\* Data preprocessing

\* Feature engineering

\* Train/test splitting

\* Random Forest classification

\* Model evaluation

\* Precision and recall

\* F1 score

\* Data validation

\* Experiment tracking

\* Reproducibility

\* Model versioning

\* Model registry

\* Model lifecycle management

\* MLOps pipeline automation

\* Git version control

\* GitHub repository management



\---



\# 🔮 Future Enhancements



Possible future improvements include:



\* Develop a web-based prediction application.

\* Add a REST API for predictions.

\* Add Docker containerization.

\* Add CI/CD automation.

\* Deploy the model to a cloud platform.

\* Add automated model retraining.

\* Add monitoring for model performance.

\* Add data drift detection.

\* Add model drift detection.

\* Create an interactive dashboard.

\* Add automated testing.



\---



\# 👨‍💻 Project Author



\*\*Tejasai846\*\*



GitHub:



https://github.com/Tejasai846



Repository:



https://github.com/Tejasai846/Customer\_Churn



\---



\# 📜 License



This project is intended for \*\*educational and academic purposes\*\*.



\---



\## ⭐ Project Summary



This project demonstrates an end-to-end approach to building and managing a \*\*Customer Churn Prediction machine learning system using MLOps practices\*\*.



The workflow combines data preprocessing, model training, evaluation, validation, experiment tracking, reproducibility, model registry, and lifecycle automation into a structured machine learning project.



