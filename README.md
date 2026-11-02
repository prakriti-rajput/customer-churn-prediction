# Customer Churn Prediction Using Machine Learning

A beginner-friendly **Predictive Analytics** project that predicts which telecom customers are likely to leave (churn), using **Logistic Regression** on the IBM Telco Customer Churn dataset.

---

## Introduction

Customer churn means a customer stops using a company's service. Winning a new customer costs much more than keeping an existing one, so telecom companies want to spot customers who are about to leave and contact them early. This project builds and compares two classic machine learning models for that task and studies which customer characteristics are linked to churn.

## Problem Statement

Given a customer's profile (tenure, contract type, internet service, monthly charges and so on), predict whether the customer will **churn (Yes)** or **stay (No)**.

## Objectives

- Clean and prepare the Telco customer data without data leakage.
- Explore the data with graphs to understand who churns.
- Train **Logistic Regression** model with scikit-learn.
- Evaluate them with Accuracy, Precision, Recall, F1-score and Confusion Matrix, focusing on the **churn class**.
- Identify the most important features and suggest improvements.

## Dataset Description and Source

- **Name:** IBM Telco Customer Churn
- **Source:** [Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** about 7,000 customers and 21 columns (the notebook prints the exact size).
- **Target column:** `Churn` (Yes / No).
- **Feature groups:**
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account information: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

The CSV is not committed here. See [`dataset/README.md`](dataset/README.md) for how to add it.

## Technologies Used

- Python 3
- Pandas and NumPy: data handling
- Matplotlib and Seaborn: visualisation
- Scikit-learn: preprocessing, models, evaluation
- Jupyter Notebook via VS Code's Jupyter extension (`ipykernel`)

## Project Structure

```
Customer-Churn-Prediction/
├── dataset/
│   └── README.md            <- put the CSV file here
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
├── graphs/                  <- graphs, model_comparison.csv and results_summary.md are saved here
├── README.md
└── requirements.txt
```

## How to Run (VS Code)

1. Install Python 3.9 or newer and the VS Code extensions **Python** and **Jupyter**.
2. Open the `Customer-Churn-Prediction` folder in VS Code.
3. Open a terminal and install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from Kaggle and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `dataset/` folder. If the file is missing, the notebook tries to download a copy by itself.
5. Open `notebooks/Customer_Churn_Prediction.ipynb`, choose your Python kernel (top right) and click **Run All**.
6. All 12 graphs are saved in `graphs/`, along with `model_comparison.csv` and `results_summary.md`.

> If `pip install` fails on a very new Python version because a library has no prebuilt package yet, create the virtual environment with Python 3.12 or 3.13 instead.

> Run the notebook from the `notebooks/` folder (opening it in VS Code does this automatically), because it uses the relative paths `../dataset` and `../graphs`.

## Methodology

1. **Load and inspect** the data with Pandas (shape, data types, missing values, duplicates).
2. **Clean**: drop `customerID`; convert `TotalCharges` from text to numbers (blank cells become missing values).
3. **Separate** the inputs (X) from the target (y, with Yes = 1 and No = 0).
4. **Split** into 80% training and 20% testing data using a stratified split, so both parts keep the same churn percentage.
5. **Preprocess inside a scikit-learn `Pipeline`** so it is fitted on training data only (no data leakage):
   - median imputation for missing `TotalCharges`
   - one-hot encoding for text columns
   - standard scaling for numeric columns (Logistic Regression only; trees do not need it)
6. **EDA** with seven graphs, each followed by an observation calculated from the data.
7. **Train** both models with `class_weight="balanced"` so the minority churn class is not ignored.
8. **Evaluate** on the unseen test data and compare the models.
9. **Interpret** with logistic-regression coefficients, decision-tree feature importance, the tree structure and per-customer churn probabilities.

## EDA Findings

The notebook produces these graphs (saved in `graphs/`):

| # | Graph | Question it answers |
|---|---|---|
| 1 | Churn distribution | How many customers leave, and is the data imbalanced? |
| 2 | Churn by gender | Does gender matter? |
| 3 | Churn by contract type | Are month-to-month customers more likely to leave? |
| 4 | Monthly charges vs churn | Do higher bills go with more churn? |
| 5 | Tenure vs churn | Do new customers leave more than loyal ones? |
| 6 | Churn by internet service | Does the type of internet service matter? |
| 7 | Correlation heatmap | Which numeric features relate to churn? |

The exact percentages and written observations are printed under each graph in the notebook and collected in `graphs/results_summary.md` after a run. Every number in them is computed from the data, not typed in by hand.

## Machine Learning Algorithms

| Algorithm | Idea | Key settings |
|---|---|---|
| **Logistic Regression** | Gives each feature a weight and converts the total into a churn probability. Simple and easy to explain. | `max_iter=1000`, `class_weight="balanced"`, scaled inputs |
| **Decision Tree** | Asks a chain of yes/no questions about the customer to reach a decision. Easy to visualise. | `max_depth=5`, `min_samples_leaf=20`, `class_weight="balanced"` |

## Model Evaluation Results

Both models are evaluated on the 20% test data using Accuracy and, for the churn class, Precision, Recall and F1-score, plus a confusion matrix for each model.

**Run the notebook to generate your results.** It saves:

- `graphs/model_comparison.csv`: the comparison table
- `graphs/results_summary.md`: a ready-to-copy summary with the automatic verdict on which model is better
- `graphs/08_confusion_matrices.png` and `graphs/09_model_comparison.png`

Paste the table from `results_summary.md` here once you have run it:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|---|---|---|---|---|
| Logistic Regression | _run notebook_ | _run notebook_ | _run notebook_ | _run notebook_ |
| Decision Tree | _run notebook_ | _run notebook_ | _run notebook_ | _run notebook_ |

**Why recall and F1 matter more than accuracy here:** most customers do not churn, so a model can score high accuracy while missing many churners. Losing a customer costs far more than a small retention offer, so we judge models mainly on how many churners they catch (recall) and how well they balance that against false alarms (F1-score).

## Conclusion and Future Scope

**Conclusion:** the project shows that customer churn can be predicted from account and service information using simple, explainable models. The notebook's final section reports which model has the better churn recall and F1-score, and which features (contract type, tenure, charges, internet service and others) matter most.

**Limitations:** one dataset from one company; two simple models with limited tuning; a single snapshot in time; correlation does not prove cause.

**Future scope:**

- Hyper-parameter tuning (`GridSearchCV`) and cross-validation
- Random Forest, Gradient Boosting or XGBoost
- Handling imbalance with SMOTE or by tuning the decision threshold
- Feature engineering (tenure groups, number of services)
- Explainability with SHAP
- Cost-based evaluation of retention offers
- A small Streamlit app for live predictions

## Author

Prakriti Rajput, Predictive Analytics project
