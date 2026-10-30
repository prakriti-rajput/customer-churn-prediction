# Customer Churn Prediction - plain Python version of the notebook.
# Run from inside the notebooks/ folder (uses ../dataset and ../graphs).
# Same code as the .ipynb. In VS Code you can also run it with Run Python File.
import matplotlib


def display(x):
    print(x)

# # Customer Churn Prediction Using Machine Learning
# 
# **Subject:** Predictive Analytics  |  **Dataset:** IBM Telco Customer Churn  |  **Models:** Logistic Regression and Decision Tree
# 
# ---
# 
# ## What is customer churn?
# *Churn* means a customer stops using a company's service. For a telecom company it is far cheaper to keep an
# existing customer than to find a new one, so if we can **predict which customers are likely to leave**, the company can
# contact them early with an offer.
# 
# ## What this notebook does
# | Step | Section | What happens |
# |---|---|---|
# | 1 | Import libraries | Load the Python tools we need |
# | 2 | Load the dataset | Read the CSV and take a first look |
# | 3 | Data preprocessing | Clean data, encode text columns, split 80% train / 20% test |
# | 4 | Exploratory Data Analysis | Seven graphs, each with an explanation |
# | 5 | Train the models | Logistic Regression and Decision Tree |
# | 6 | Evaluate the models | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
# | 7 | Additional analysis | Important features and how to interpret predictions |
# | 8 | Findings and conclusion | Summary, improvements and future scope |
# 
# > **Important note about the explanations:** every number quoted in the "Observation" boxes is **calculated by the code
# > from the real dataset and printed**. Nothing is typed in by hand, so the explanations always match your results.

# ## 1. Import Libraries
# 
# * **pandas / numpy**: handle tables and numbers
# * **matplotlib / seaborn**: draw graphs
# * **scikit-learn**: preprocessing, machine learning models and evaluation metrics
# 
# `RANDOM_STATE = 42` makes the train/test split and the models give the **same result every time** you run the notebook.

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")          # keep the output clean for students

# ---------- global settings ----------
RANDOM_STATE = 42                          # same results on every run
GRAPHS_DIR = "../graphs"                   # all pictures are saved here
os.makedirs(GRAPHS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

# Same two colours for "No churn" / "Churn" in every graph
CHURN_COLORS = {"No": "#2E86AB", "Yes": "#E4572E"}


def finish_plot(file_name):
    # Tidy the layout, save the picture into the graphs folder and show it.
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, file_name), dpi=150, bbox_inches="tight")
    plt.show()


print("Libraries imported. Graphs will be saved in:", os.path.abspath(GRAPHS_DIR))

# ## 2. Load the Dataset
# 
# The notebook looks for the CSV file inside the **`dataset/`** folder in this order:
# 
# 1. `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv` (the file name you get from Kaggle)
# 2. `dataset/Telco-Customer-Churn.csv`
# 3. If neither exists, it **tries to download** the IBM copy of the same dataset automatically.
# 4. If the download also fails (for example no internet), it stops with **clear manual instructions**.
# 
# **Manual download:** open <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>, click *Download*, unzip, and put
# `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside the `dataset/` folder. Then run the notebook again.

DATASET_DIR = "../dataset"
POSSIBLE_FILE_NAMES = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco-Customer-Churn.csv"]
BACKUP_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
              "master/data/Telco-Customer-Churn.csv")


def find_local_dataset():
    # Return the path of the CSV if it is already inside the dataset folder.
    for file_name in POSSIBLE_FILE_NAMES:
        path = os.path.join(DATASET_DIR, file_name)
        if os.path.exists(path):
            return path
    return None


dataset_path = find_local_dataset()

if dataset_path is None:
    print("Dataset not found in the dataset/ folder. Trying to download it ...")
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        downloaded = pd.read_csv(BACKUP_URL)
        dataset_path = os.path.join(DATASET_DIR, "Telco-Customer-Churn.csv")
        downloaded.to_csv(dataset_path, index=False)
        print("Download successful. Saved to:", dataset_path)
    except Exception as error:
        dataset_path = None
        print("Automatic download failed:", type(error).__name__)

if dataset_path is None:
    raise FileNotFoundError(
        "\n\nThe dataset CSV was not found.\n"
        "Please do this once:\n"
        "  1. Open https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        "  2. Click 'Download' and unzip the file\n"
        "  3. Copy 'WA_Fn-UseC_-Telco-Customer-Churn.csv' into the 'dataset' folder of this project\n"
        "  4. Run this notebook again from the top.\n"
    )

df = pd.read_csv(dataset_path)
print("Dataset loaded from:", dataset_path)

# Safety check: make sure this is really the Telco churn file
required_columns = {"customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"}
missing_columns = required_columns - set(df.columns)
assert not missing_columns, f"This does not look like the Telco churn dataset. Missing columns: {missing_columns}"

# First look at the data
display(df.head())

# Column names, data types and non-null counts
df.info()

# Summary statistics of the numeric columns
display(df.describe())

# ## 3. Data Preprocessing
# 
# Machine learning models only understand clean numbers. In this section we:
# 
# 1. Check size, missing values and duplicates
# 2. Remove the useless `customerID` column
# 3. Fix the `TotalCharges` column (it is stored as text)
# 4. Separate the inputs (**X**) from the answer we want to predict (**y**)
# 5. Split into **80% training** and **20% testing** data
# 6. Encode text columns into numbers (and scale numeric columns where needed)
# 
# ### How we avoid *data leakage*
# The test data must behave like **brand-new customers the model has never seen**. So every step that *learns* something from
# the data (filling missing values with a median, scaling, one-hot encoding) is **fitted only on the training data** and
# then only *applied* to the test data. We do this with a scikit-learn `Pipeline`, which makes it automatic.

# ### 3.1 Number of rows and columns

rows, columns = df.shape
print(f"Number of rows (customers) : {rows}")
print(f"Number of columns          : {columns}")
print("\nColumn names:")
print(list(df.columns))

# ### 3.2 Missing values
# `isnull()` finds empty cells. Note: `TotalCharges` is stored as **text**, so blank cells (just spaces) may be hidden here.
# We will expose them in step 3.5.

missing_per_column = df.isnull().sum()
print("Missing values per column:")
print(missing_per_column[missing_per_column > 0] if missing_per_column.sum() > 0 else "No empty cells found (yet).")
print("\nTotal missing cells:", int(missing_per_column.sum()))

# ### 3.3 Duplicate records

duplicate_rows = df.duplicated().sum()
duplicate_ids = df["customerID"].duplicated().sum()
print("Fully duplicated rows        :", duplicate_rows)
print("Repeated customerID values   :", duplicate_ids)
print("\nEvery customerID is unique, so each row is a different customer. "
      "Nothing needs to be removed." if duplicate_ids == 0 else
      "\nSome customerIDs repeat - these rows should be inspected.")

# > **Why we do not drop duplicates after removing `customerID`:** two *different* customers can have exactly the same
# > answers in every column. Deleting them would remove real customers, so we keep them.
# 
# ### 3.4 Remove unnecessary columns
# `customerID` is only a label (like a roll number). It has no meaning for predicting churn, so we drop it.

df = df.drop(columns=["customerID"])
print("Columns left:", df.shape[1])

# ### 3.5 Convert `TotalCharges` to numeric and handle missing values
# `TotalCharges` should be a number but pandas read it as text (because a few cells contain a blank space).
# `pd.to_numeric(..., errors="coerce")` converts valid numbers and turns anything invalid into `NaN` (missing).

print("Data type BEFORE conversion:", df["TotalCharges"].dtype)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Data type AFTER conversion :", df["TotalCharges"].dtype)
print("Missing values in TotalCharges:", int(df["TotalCharges"].isnull().sum()))

# Look at the rows that became missing
rows_with_missing_total = df[df["TotalCharges"].isnull()]
display(rows_with_missing_total[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

if len(rows_with_missing_total) > 0:
    all_zero_tenure = bool((rows_with_missing_total["tenure"] == 0).all())
    print("Do all these customers have tenure = 0 (brand-new customers)?", all_zero_tenure)

# **How we handle these missing values:** we fill them with the **median** of `TotalCharges`. The median is safer than the
# mean because a few very large values do not affect it.
# 
# To avoid leakage the median is **calculated from the training data only**. This is done by `SimpleImputer` inside the
# pipeline (section 3.8), *after* the train/test split.
# 
# ### 3.6 Separate independent (X) and dependent (y) variables
# * **X** = everything we use to make the prediction (tenure, contract, charges, ...)
# * **y** = what we want to predict (`Churn`): we turn `Yes` into **1** and `No` into **0**
# 
# The original text column `Churn` stays in `df` because it gives nicer labels for the graphs in section 4.

y = (df["Churn"] == "Yes").astype(int)      # 1 = customer left, 0 = customer stayed
X = df.drop(columns=["Churn"])              # independent variables

print("X shape (rows, columns):", X.shape)
print("y shape                :", y.shape)
print("\nTarget values (0 = stayed, 1 = churned):")
print(y.value_counts())

# ### 3.7 Train / test split (80% / 20%)
# * `test_size=0.2` keeps 20% of customers aside for the final exam.
# * `stratify=y` keeps the **same percentage of churners** in both parts. This matters because churners are the smaller group.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training set : {X_train.shape[0]} customers ({X_train.shape[0] / len(X):.0%})")
print(f"Testing set  : {X_test.shape[0]} customers ({X_test.shape[0] / len(X):.0%})")
print(f"\nChurn rate in training data: {y_train.mean():.2%}")
print(f"Churn rate in testing data : {y_test.mean():.2%}")

# ### 3.8 Encoding categorical variables and scaling numeric variables
# 
# * **One-hot encoding** turns a text column such as `Contract` (Month-to-month / One year / Two year) into separate
#   0/1 columns. Yes/No columns become a single 0/1 column.
# * **Imputation** fills the missing `TotalCharges` values with the training median.
# * **Scaling (`StandardScaler`)** puts numeric columns on a similar range. Logistic Regression **needs** this
#   because it is sensitive to the size of numbers. A Decision Tree does **not** need it, so we skip it there.
# 
# `build_preprocessor()` creates this recipe. Below we run it on the training data just to *see* what the encoded data looks like.
# The real models in section 5 use their own copy inside a `Pipeline`.

# Split the columns into numeric and text (categorical) columns
numeric_features = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
categorical_features = [col for col in X_train.columns if col not in numeric_features]

print("Numeric columns    :", numeric_features)
print(f"\nCategorical columns ({len(categorical_features)}):", categorical_features)


def build_preprocessor(scale_numeric):
    # Recipe: fill missing numbers, optionally scale them, and one-hot encode the text columns.
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
             categorical_features),
        ],
        verbose_feature_names_out=False,
    )

# Demonstration of fit on TRAIN only, transform on TEST
demo_preprocessor = build_preprocessor(scale_numeric=True)
X_train_encoded = demo_preprocessor.fit_transform(X_train)   # learns from training data only
X_test_encoded = demo_preprocessor.transform(X_test)         # only applies what was learned

encoded_feature_names = demo_preprocessor.get_feature_names_out()

print("Columns BEFORE encoding:", X_train.shape[1])
print("Columns AFTER encoding :", X_train_encoded.shape[1])
print("Missing values left in encoded training data:", int(np.isnan(X_train_encoded).sum()))
print("\nFirst 5 rows of the encoded training data:")
display(pd.DataFrame(X_train_encoded, columns=encoded_feature_names).head())

# ## 4. Exploratory Data Analysis (EDA)
# 
# EDA means *looking at the data with graphs before building models*. It helps us understand **who churns**.
# 
# * EDA is **descriptive only**: we are not training anything here, so using the full cleaned table does not cause leakage.
# * Churn is shown as **No** (stayed) and **Yes** (left). Colours are the same in every graph:
#   blue = stayed, orange-red = churned.
# * The code **prints the real numbers** under each graph and builds the "Observation" automatically.

# A list that collects the key observations; we save it to a file at the end of the notebook.
eda_findings = []

hue_order = ["No", "Yes"]


def churn_rate_by(column):
    # Percentage of customers who churned inside each group of `column`.
    return df.groupby(column)["Churn"].agg(lambda s: (s == "Yes").mean() * 100).sort_values(ascending=False)


def describe_rates(column, label):
    # Print the churn rate of every group and a one-sentence observation (numbers come from the data).
    rates = churn_rate_by(column)
    print(f"Churn rate by {label}:")
    for group, rate in rates.items():
        print(f"   - {group:<20s}: {rate:5.1f}%")
    gap = rates.max() - rates.min()
    sentence = (f"Observation: '{rates.idxmax()}' customers churn the most ({rates.max():.1f}%), "
                f"'{rates.idxmin()}' customers churn the least ({rates.min():.1f}%), "
                f"a gap of {gap:.1f} percentage points.")
    print("\n" + sentence)
    eda_findings.append(sentence.replace("Observation: ", f"{label}: "))
    return rates

# ### 4.1 Customer churn distribution
# **Why:** shows how many customers left versus stayed. If one group is much smaller, plain *accuracy* can be misleading.

churn_counts = df["Churn"].value_counts().reindex(hue_order)
churn_percent = churn_counts / churn_counts.sum() * 100

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.countplot(data=df, x="Churn", order=hue_order, palette=CHURN_COLORS, hue="Churn", legend=False, ax=ax)
for container in ax.containers:
    ax.bar_label(container, fontsize=11, padding=3)
ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn (Yes = customer left)")
ax.set_ylabel("Number of customers")
ax.set_ylim(0, churn_counts.max() * 1.12)
finish_plot("01_churn_distribution.png")

print(f"Stayed (No)   : {churn_counts['No']} customers ({churn_percent['No']:.1f}%)")
print(f"Churned (Yes) : {churn_counts['Yes']} customers ({churn_percent['Yes']:.1f}%)")

minority_share = churn_percent.min()
if minority_share < 40:
    sentence = (f"Observation: only {churn_percent['Yes']:.1f}% of customers churned, so the data is imbalanced. "
                "A model that always says 'No churn' would still look accurate, which is why we also look at "
                "Recall and F1-score for the churn class.")
else:
    sentence = "Observation: the two classes are fairly balanced."
print("\n" + sentence)
eda_findings.append(sentence.replace("Observation: ", "Churn distribution: "))

# ### 4.2 Churn based on gender
# **Why:** checks whether male and female customers behave differently.

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.countplot(data=df, x="gender", hue="Churn", hue_order=hue_order, palette=CHURN_COLORS, ax=ax)
for container in ax.containers:
    ax.bar_label(container, fontsize=10, padding=3)
ax.set_title("Customer Churn by Gender", fontsize=14, fontweight="bold")
ax.set_xlabel("Gender")
ax.set_ylabel("Number of customers")
ax.legend(title="Churn")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
finish_plot("02_churn_by_gender.png")

gender_rates = describe_rates("gender", "Gender")
if gender_rates.max() - gender_rates.min() < 2:
    note = "The difference is small (under 2 percentage points), so gender is not a strong factor for churn."
else:
    note = "The difference is noticeable (2 percentage points or more), so gender may play a role."
print(note)
eda_findings.append("Gender: " + note)

# ### 4.3 Churn based on contract type
# **Why:** the contract length shows how "locked in" a customer is.

contract_order = ["Month-to-month", "One year", "Two year"]

fig, ax = plt.subplots(figsize=(8, 4.8))
sns.countplot(data=df, x="Contract", order=contract_order, hue="Churn", hue_order=hue_order,
              palette=CHURN_COLORS, ax=ax)
for container in ax.containers:
    ax.bar_label(container, fontsize=10, padding=3)
ax.set_title("Customer Churn by Contract Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Contract type")
ax.set_ylabel("Number of customers")
ax.legend(title="Churn")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
finish_plot("03_churn_by_contract.png")

contract_rates = describe_rates("Contract", "Contract type")

# ### 4.4 Monthly charges vs churn
# **Why:** tests whether customers who pay more are more likely to leave.
# The *histogram* shows how many customers fall in each price range; the *box plot* compares the middle (median) price.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

sns.histplot(data=df, x="MonthlyCharges", hue="Churn", hue_order=hue_order, palette=CHURN_COLORS,
             kde=True, element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Monthly Charges Distribution by Churn", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Monthly charges")
axes[0].set_ylabel("Density")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", order=hue_order, palette=CHURN_COLORS,
            hue="Churn", legend=False, ax=axes[1])
axes[1].set_title("Monthly Charges vs Churn", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Churn (Yes = customer left)")
axes[1].set_ylabel("Monthly charges")
finish_plot("04_monthly_charges_vs_churn.png")

charge_stats = df.groupby("Churn")["MonthlyCharges"].agg(["mean", "median"]).round(2)
display(charge_stats)

diff = charge_stats.loc["Yes", "mean"] - charge_stats.loc["No", "mean"]
direction = "higher" if diff > 0 else "lower"
sentence = (f"Observation: churned customers pay on average {charge_stats.loc['Yes', 'mean']:.2f} per month versus "
            f"{charge_stats.loc['No', 'mean']:.2f} for customers who stayed ({abs(diff):.2f} {direction}).")
print(sentence)
eda_findings.append(sentence.replace("Observation: ", "Monthly charges: "))

# ### 4.5 Customer tenure vs churn
# **Why:** *tenure* is the number of months a customer has been with the company. It tells us whether new or loyal customers leave more.

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

sns.histplot(data=df, x="tenure", hue="Churn", hue_order=hue_order, palette=CHURN_COLORS,
             bins=24, kde=True, element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title("Customer Tenure Distribution by Churn", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Tenure (months)")
axes[0].set_ylabel("Density")

sns.boxplot(data=df, x="Churn", y="tenure", order=hue_order, palette=CHURN_COLORS,
            hue="Churn", legend=False, ax=axes[1])
axes[1].set_title("Tenure vs Churn", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Churn (Yes = customer left)")
axes[1].set_ylabel("Tenure (months)")
finish_plot("05_tenure_vs_churn.png")

tenure_stats = df.groupby("Churn")["tenure"].agg(["mean", "median"]).round(1)
display(tenure_stats)

# Churn rate for groups of tenure
tenure_group = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, np.inf],
                      labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"])
tenure_group_rates = (df["Churn"] == "Yes").groupby(tenure_group, observed=True).mean() * 100
print("Churn rate by tenure group:")
for group, rate in tenure_group_rates.items():
    print(f"   - {group:<14s}: {rate:5.1f}%")

sentence = (f"Observation: churned customers have a median tenure of {tenure_stats.loc['Yes', 'median']:.0f} months "
            f"compared with {tenure_stats.loc['No', 'median']:.0f} months for customers who stayed. "
            f"The churn rate is highest in the '{tenure_group_rates.idxmax()}' group ({tenure_group_rates.max():.1f}%) "
            f"and lowest in the '{tenure_group_rates.idxmin()}' group ({tenure_group_rates.min():.1f}%).")
print("\n" + sentence)
eda_findings.append(sentence.replace("Observation: ", "Tenure: "))

# ### 4.6 Churn based on internet service
# **Why:** the type of internet connection (DSL, Fiber optic, or none) may affect satisfaction and price.

internet_order = ["DSL", "Fiber optic", "No"]

fig, ax = plt.subplots(figsize=(8, 4.8))
sns.countplot(data=df, x="InternetService", order=internet_order, hue="Churn", hue_order=hue_order,
              palette=CHURN_COLORS, ax=ax)
for container in ax.containers:
    ax.bar_label(container, fontsize=10, padding=3)
ax.set_title("Customer Churn by Internet Service Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Internet service")
ax.set_ylabel("Number of customers")
ax.legend(title="Churn")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
finish_plot("06_churn_by_internet_service.png")

internet_rates = describe_rates("InternetService", "Internet service")

# ### 4.7 Correlation heatmap
# **Why:** correlation measures how strongly two *numeric* columns move together (from -1 to +1).
# A value near 0 means no linear relationship. We convert Churn to 0/1 so it can be included.

correlation_data = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].copy()
correlation_data["Churn (1 = Yes)"] = (df["Churn"] == "Yes").astype(int)
correlation_matrix = correlation_data.corr()

fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, square=True, cbar_kws={"label": "Correlation"}, ax=ax)
ax.set_title("Correlation Heatmap of Numeric Features", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, ha="right")
finish_plot("07_correlation_heatmap.png")

with_churn = correlation_matrix["Churn (1 = Yes)"].drop("Churn (1 = Yes)").sort_values()
print("Correlation of each numeric feature with churn:")
for feature, value in with_churn.items():
    print(f"   - {feature:<16s}: {value:+.2f}")

strongest = with_churn.abs().idxmax()
sentence = (f"Observation: '{strongest}' has the strongest linear relationship with churn "
            f"({with_churn[strongest]:+.2f}). A negative value means churn becomes less likely as the feature increases. "
            "Correlation only covers numeric columns and straight-line relationships, so the models in the next "
            "sections also use the categorical columns.")
print("\n" + sentence)
eda_findings.append(sentence.replace("Observation: ", "Correlation: "))

# ## 5. Machine Learning Models
# 
# We train **two** algorithms:
# 
# ### Logistic Regression
# Calculates a **probability of churn** by giving each feature a weight (coefficient). Simple, fast and easy to explain.
# Positive weight = pushes towards churn, negative weight = pushes towards staying.
# 
# ### Decision Tree Classifier
# Asks a series of **yes/no questions** (for example *"Is the contract month-to-month?"*) and follows the branches to a decision. Easy
# to visualise and can capture non-straight-line patterns.
# 
# ### Settings used (and why)
# | Setting | Reason |
# |---|---|
# | `class_weight="balanced"` | Churners are the minority. This makes the model pay more attention to them, which improves **recall** for churn. |
# | `max_iter=1000` (Logistic Regression) | Gives the algorithm enough steps to finish learning. |
# | `max_depth=5`, `min_samples_leaf=20` (Decision Tree) | Stops the tree growing too deep. An unlimited tree memorises the training data (*overfitting*). |
# | `random_state=42` | Same result every run. |
# 
# Each model is a **Pipeline** = preprocessing + model, so the preprocessing is **fitted only on training data** when we call `.fit()`.

# Logistic Regression: needs scaled numbers
log_reg_model = Pipeline([
    ("preprocess", build_preprocessor(scale_numeric=True)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
])

# Decision Tree: does not need scaling
decision_tree_model = Pipeline([
    ("preprocess", build_preprocessor(scale_numeric=False)),
    ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
])

# Train both models on the TRAINING data only
log_reg_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
print("Both models trained successfully.")

# ## 6. Model Evaluation
# 
# We now test both models on the **20% test data** that they have never seen.
# 
# ### What each metric means (for the *churn* class)
# | Metric | Simple meaning |
# |---|---|
# | **Accuracy** | Out of all customers, how many predictions were correct? (can mislead when classes are imbalanced) |
# | **Precision** | When the model says "will churn", how often is it right? |
# | **Recall** | Out of all customers who really churned, how many did the model catch? |
# | **F1-score** | One number that balances Precision and Recall |
# 
# ### Confusion matrix
# |  | Predicted: stays | Predicted: churns |
# |---|---|---|
# | **Actually stays** | True Negative (TN) | False Positive (FP): wasted offer |
# | **Actually churns** | False Negative (FN): **missed churner** | True Positive (TP): caught churner |
# 
# **Why Recall and F1 matter most here:** a *missed churner* (FN) means we lose a customer for nothing, while a
# *false alarm* (FP) only costs a small discount. So we care about catching churners (Recall) without too many false alarms (F1).

def evaluate_model(model_name, model):
    # Predict on the test set and calculate the metrics for the churn class (label 1).
    predictions = model.predict(X_test)
    scores = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Churn)": precision_score(y_test, predictions, zero_division=0),
        "Recall (Churn)": recall_score(y_test, predictions, zero_division=0),
        "F1-score (Churn)": f1_score(y_test, predictions, zero_division=0),
    }
    return scores, predictions


log_reg_scores, log_reg_predictions = evaluate_model("Logistic Regression", log_reg_model)
tree_scores, tree_predictions = evaluate_model("Decision Tree", decision_tree_model)

comparison = pd.DataFrame([log_reg_scores, tree_scores]).round(4)
comparison.to_csv(os.path.join(GRAPHS_DIR, "model_comparison.csv"), index=False)

print("MODEL COMPARISON (evaluated on the 20% test data)")
display(comparison)

# Detailed classification reports
print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)
print(classification_report(y_test, log_reg_predictions, target_names=["No Churn", "Churn"], zero_division=0))

print("=" * 60)
print("DECISION TREE")
print("=" * 60)
print(classification_report(y_test, tree_predictions, target_names=["No Churn", "Churn"], zero_division=0))

# Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, predictions) in zip(axes, [("Logistic Regression", log_reg_predictions),
                                          ("Decision Tree", tree_predictions)]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, display_labels=["No Churn", "Churn"],
        cmap="Blues", colorbar=False, ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix: {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.grid(False)
finish_plot("08_confusion_matrices.png")

# Plain-language reading of each matrix
for name, predictions in [("Logistic Regression", log_reg_predictions), ("Decision Tree", tree_predictions)]:
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f"{name}: caught {tp} of {tp + fn} real churners (missed {fn}); "
          f"raised {fp} false alarms among {tn + fp} loyal customers.")

# Bar chart comparing the metrics of both models
metrics_to_plot = ["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-score (Churn)"]
plot_data = comparison.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=plot_data, x="Metric", y="Score", hue="Model", palette=["#2E86AB", "#F2A541"], ax=ax)
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=10, padding=3)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xlabel("Evaluation metric")
ax.set_ylabel("Score (0 to 1)")
ax.set_ylim(0, 1.1)
ax.legend(title="Model", loc="lower right")
finish_plot("09_model_comparison.png")

# ### Which model is better, and why?
# The cell below compares the two models **using the real numbers** and writes the conclusion automatically.

scores = comparison.set_index("Model")
best_accuracy = scores["Accuracy"].idxmax()
best_recall = scores["Recall (Churn)"].idxmax()
best_precision = scores["Precision (Churn)"].idxmax()
best_f1 = scores["F1-score (Churn)"].idxmax()

print(f"Highest Accuracy          : {best_accuracy}")
print(f"Highest Precision (Churn) : {best_precision}")
print(f"Highest Recall (Churn)    : {best_recall}")
print(f"Highest F1-score (Churn)  : {best_f1}")

other_model = [m for m in scores.index if m != best_f1][0]
verdict_lines = []
verdict_lines.append(
    f"Based on the F1-score for the churn class, {best_f1} performs better "
    f"({scores.loc[best_f1, 'F1-score (Churn)']:.3f} vs {scores.loc[other_model, 'F1-score (Churn)']:.3f}).")
if best_recall == best_f1:
    verdict_lines.append(
        f"It also catches more real churners (recall {scores.loc[best_f1, 'Recall (Churn)']:.3f} vs "
        f"{scores.loc[other_model, 'Recall (Churn)']:.3f}), which is the most important goal in churn prediction.")
else:
    verdict_lines.append(
        f"However, {best_recall} has the higher recall ({scores.loc[best_recall, 'Recall (Churn)']:.3f}), "
        f"so it catches more churners, while {best_f1} gives a better balance between recall and precision.")
if best_accuracy != best_f1:
    verdict_lines.append(
        f"Note that {best_accuracy} has the higher accuracy, but accuracy alone is not reliable because "
        "most customers do not churn.")

print("\nCONCLUSION")
for line in verdict_lines:
    print(" -", line)

# Check for overfitting: compare accuracy on training data and test data
print("\nOverfitting check (accuracy on training data vs test data):")
for name, model in [("Logistic Regression", log_reg_model), ("Decision Tree", decision_tree_model)]:
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"   {name:<20s}: train {train_acc:.3f} | test {test_acc:.3f} | gap {train_acc - test_acc:+.3f}")

# **How to explain this in your viva**
# 
# * **Logistic Regression** learns one straight-line rule using all features together. It is stable, rarely overfits and its
#   weights are easy to explain.
# * **Decision Tree** can capture rules like *"month-to-month AND short tenure AND high bill"* that a straight line may miss,
#   but it can also memorise the training data. We limited its depth to reduce that risk.
# * The cell above decides the winner **only from the test results**, not from assumptions. If you change a setting such as
#   `max_depth`, re-run the notebook and the conclusion updates itself.

# ## 7. Additional Analysis
# 
# ### 7.1 Important features influencing churn
# 
# **Logistic Regression coefficients.** Numeric columns were scaled, so coefficients are comparable.
# * Positive coefficient: increases the chance of churn.
# * Negative coefficient: decreases the chance of churn.
# * `exp(coefficient)` (the *odds ratio*) above 1 means higher odds of churn, below 1 means lower odds.

lr_feature_names = log_reg_model.named_steps["preprocess"].get_feature_names_out()
lr_coefficients = log_reg_model.named_steps["model"].coef_[0]

coef_table = pd.DataFrame({
    "Feature": lr_feature_names,
    "Coefficient": lr_coefficients,
    "Odds ratio": np.exp(lr_coefficients),
}).sort_values("Coefficient", ascending=False)

# Show the 10 strongest "increase churn" and 10 strongest "decrease churn" features
top_features = pd.concat([coef_table.head(10), coef_table.tail(10)]).sort_values("Coefficient")
bar_colors = [CHURN_COLORS["Yes"] if c > 0 else CHURN_COLORS["No"] for c in top_features["Coefficient"]]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_features["Feature"], top_features["Coefficient"], color=bar_colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Logistic Regression: Features Pushing Churn Up (red) or Down (blue)", fontsize=13, fontweight="bold")
ax.set_xlabel("Coefficient (effect on churn)")
ax.set_ylabel("Feature")
finish_plot("10_logistic_regression_coefficients.png")

print("Top 5 features that INCREASE churn risk:")
display(coef_table.head(5).round(3).reset_index(drop=True))
print("Top 5 features that DECREASE churn risk:")
display(coef_table.tail(5).sort_values("Coefficient").round(3).reset_index(drop=True))

# **Decision Tree feature importance.** Shows how much each feature helped the tree make good splits
# (values add up to 1; higher = more important). Unlike coefficients, importance does **not** show direction.

tree_feature_names = decision_tree_model.named_steps["preprocess"].get_feature_names_out()
tree_importances = decision_tree_model.named_steps["model"].feature_importances_

importance_table = (pd.DataFrame({"Feature": tree_feature_names, "Importance": tree_importances})
                    .sort_values("Importance", ascending=False))
top_tree_features = importance_table.head(10).sort_values("Importance")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_tree_features["Feature"], top_tree_features["Importance"], color="#F2A541")
ax.set_title("Decision Tree: Top 10 Most Important Features", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance score")
ax.set_ylabel("Feature")
finish_plot("11_decision_tree_feature_importance.png")

display(importance_table.head(10).round(4).reset_index(drop=True))

top3_tree = ", ".join(importance_table["Feature"].head(3))
top3_lr = ", ".join(coef_table.reindex(coef_table["Coefficient"].abs().sort_values(ascending=False).index)["Feature"].head(3))
print(f"Top 3 features for the Decision Tree      : {top3_tree}")
print(f"Top 3 features (by size) for Logistic Reg.: {top3_lr}")

# **The first levels of the decision tree.** Reading from the top: each box shows the question asked, how many customers reach it,
# and the majority class. This is the easiest way to *show* how the model thinks.

fig, ax = plt.subplots(figsize=(20, 9))
plot_tree(decision_tree_model.named_steps["model"], max_depth=3,
          feature_names=list(tree_feature_names), class_names=["No Churn", "Churn"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Decision Tree Structure (first 3 levels)", fontsize=15, fontweight="bold")
finish_plot("12_decision_tree_structure.png")

# ### 7.2 Interpretation of model predictions
# A model can give a **churn probability** for every customer. A retention team could contact the customers with the
# highest probability first. Below we show the test customers the Logistic Regression model finds *most* and *least* likely to churn,
# next to what **actually** happened.
# 
# > The probabilities are higher than the real-world rate because we used `class_weight="balanced"`.
# > Use them to **rank** customers by risk, not as exact percentages.

churn_probability = log_reg_model.predict_proba(X_test)[:, 1]

prediction_view = X_test[["Contract", "InternetService", "tenure", "MonthlyCharges"]].copy()
prediction_view["Churn probability"] = churn_probability.round(3)
prediction_view["Predicted"] = np.where(churn_probability >= 0.5, "Churn", "Stay")
prediction_view["Actual"] = y_test.map({0: "Stay", 1: "Churn"})
prediction_view = prediction_view.sort_values("Churn probability", ascending=False)

print("5 test customers with the HIGHEST predicted churn risk:")
display(prediction_view.head(5))
print("5 test customers with the LOWEST predicted churn risk:")
display(prediction_view.tail(5))

# How good is the ranking? Share of real churners among the riskiest 20% of test customers
top_20_percent = prediction_view.head(int(len(prediction_view) * 0.2))
share_actual_churn = (top_20_percent["Actual"] == "Churn").mean()
print(f"\nAmong the 20% of test customers with the highest predicted risk, {share_actual_churn:.1%} really churned "
      f"(the overall churn rate in the test set is {y_test.mean():.1%}).")

# ## 8. Final Findings and Conclusion

print("=" * 70)
print("FINAL SUMMARY (all numbers come from this run)")
print("=" * 70)
print(f"Dataset         : {rows} customers, {columns} original columns")
print(f"Train / test    : {len(X_train)} / {len(X_test)} customers (80% / 20%)")
print(f"Overall churn   : {y.mean():.1%}")

print("\nKey patterns found in EDA:")
for finding in eda_findings:
    print(" -", finding)

print("\nModel results on the test data:")
display(comparison)
for line in verdict_lines:
    print(" -", line)

print("\nMost influential features:")
print(" - Logistic Regression (largest coefficients):", top3_lr)
print(" - Decision Tree (highest importance)        :", top3_tree)

# Save a text summary of this run next to the graphs (handy for the README / report)
summary_lines = ["# Results summary (auto-generated by the notebook)", "",
                 f"- Customers: {rows}, original columns: {columns}",
                 f"- Training / testing customers: {len(X_train)} / {len(X_test)}",
                 f"- Overall churn rate: {y.mean():.1%}", "", "## EDA findings"]
summary_lines += [f"- {finding}" for finding in eda_findings]

summary_lines += ["", "## Model comparison (test data)", "",
                  "| " + " | ".join(comparison.columns) + " |",
                  "|" + "---|" * len(comparison.columns)]
for _, row in comparison.iterrows():
    summary_lines.append("| " + " | ".join(str(value) for value in row.values) + " |")

summary_lines += ["", "## Conclusion"] + [f"- {line}" for line in verdict_lines]
summary_lines += ["", "## Top features",
                  f"- Logistic Regression: {top3_lr}",
                  f"- Decision Tree: {top3_tree}"]

with open(os.path.join(GRAPHS_DIR, "results_summary.md"), "w", encoding="utf-8") as summary_file:
    summary_file.write("\n".join(summary_lines))
print("Saved:", os.path.join(GRAPHS_DIR, "results_summary.md"))

# ### Conclusion
# In this project we cleaned the Telco customer data, explored it with seven graphs, and trained **Logistic Regression** and a
# **Decision Tree** to predict churn. The comparison table and the automatic verdict in section 6 show which model
# detects churners better; the EDA and feature-importance sections show **which customer characteristics are linked to churn**
# (contract type, tenure, monthly charges, internet service and so on). A company can use these ideas to
# target retention offers at the customers with the highest predicted risk.
# 
# ### Limitations
# * Only one dataset from one company, so results may not transfer to other businesses.
# * Only two simple models with limited tuning.
# * The data is a single snapshot: it does not show how behaviour changes over time.
# * Correlation is not causation. A feature linked to churn is not necessarily the *reason* for churn.
# 
# ### Possible improvements and future scope
# 1. **Hyper-parameter tuning** with `GridSearchCV` and **cross-validation** for more reliable scores.
# 2. **More models:** Random Forest, Gradient Boosting, XGBoost (still beginner-friendly in scikit-learn).
# 3. **Handle class imbalance** with SMOTE or by tuning the **decision threshold** (not always 0.5) to trade precision for recall.
# 4. **Feature engineering:** tenure groups, number of subscribed services, average monthly spend.
# 5. **Model explainability** with SHAP values.
# 6. **Cost-based evaluation:** include the real cost of a lost customer vs. a retention offer.
# 7. **Deployment:** wrap the trained pipeline in a small **Streamlit** app where a manager enters customer details and gets a churn risk.
# 8. **Live data:** retrain regularly and track the model over time.
