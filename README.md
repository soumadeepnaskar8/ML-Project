# 🌐 Network Protocol Classification Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Level](https://img.shields.io/badge/Level-Beginner-green)
![Platform](https://img.shields.io/badge/Run%20on-Colab%20%7C%20Jupyter-yellow?logo=googlecolab&logoColor=white)

> A beginner-friendly machine learning project that predicts the **network protocol** (TCP, UDP, DNS, ARP, ICMP, ...) of a packet using its captured characteristics, built on the **KT GameBox** packet-capture dataset.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Learning Objectives](#-learning-objectives)
3. [Workflow](#-workflow)
4. [Dataset](#-dataset)
5. [Project Structure](#-project-structure)
6. [Setup and Installation](#-setup-and-installation)
7. [Step-by-Step Guide](#-step-by-step-guide)
8. [Models Used](#-models-used)
9. [Evaluation Metrics](#-evaluation-metrics)
10. [Predicting a New Packet](#-predicting-a-new-packet)
11. [Key Concepts Cheat Sheet](#-key-concepts-cheat-sheet)
12. [Common Problems and Fixes](#-common-problems-and-fixes)
13. [Limitations](#-limitations)
14. [Practice Exercises](#-practice-exercises)
15. [Future Improvements](#-future-improvements)

---

## 🎯 Project Overview

| Item | Description |
|------|-------------|
| **Problem type** | Multi-class classification |
| **Input (X)** | Packet length and time-based features |
| **Output (y)** | `Protocol` (e.g. TCP, UDP, DNS, ARP, ICMP) |
| **Dataset** | `KT_GameBox_1.csv` (packet capture, exported as CSV) |
| **Starting size** | First **100,000 rows** (increase once you understand the project) |
| **Tools** | Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn |

**Goal:** Given information about a network packet, train a model that predicts which protocol the packet uses.

---

## 📚 Learning Objectives

By completing this project you will learn how to:

- [x] Load and explore a real-world dataset with `pandas`
- [x] Clean data (missing values, duplicates)
- [x] Engineer new features from a timestamp
- [x] Visualize data with Matplotlib and Seaborn
- [x] Encode categorical labels with `LabelEncoder`
- [x] Split data into training and testing sets
- [x] Scale features and avoid **data leakage**
- [x] Train and compare 4 classification models
- [x] Evaluate with accuracy, classification report, and confusion matrix
- [x] Interpret feature importance
- [x] Use a trained model to predict a brand-new packet

---

## 🔄 Workflow

```text
KT_GameBox Dataset
        ↓
Load Dataset
        ↓
Understand Data
        ↓
Clean Data
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis (EDA)
        ↓
Select Features
        ↓
Encode Categorical Data
        ↓
Train / Test Split
        ↓
Train ML Models
        ↓
Prediction
        ↓
Accuracy
        ↓
Confusion Matrix
        ↓
Compare Models
        ↓
Predict New Packet
```

---

## 📦 Dataset

The dataset is a packet capture exported to CSV. It should contain columns similar to:

| Column | Meaning | Used in model? |
|--------|---------|----------------|
| `No.` | Packet number | ❌ |
| `Time` | Capture timestamp | ✅ (converted to Hour, Minute, Second) |
| `Source` | Source address | ❌ (too many unique values for a beginner model) |
| `Destination` | Destination address | ❌ (same reason) |
| `Protocol` | Protocol name | 🎯 **Target** |
| `Length` | Packet size in bytes | ✅ |
| `Info` | Packet summary text | ❌ |

> 📥 Place `KT_GameBox_1.csv` in the project folder (or upload it in Colab) before running the code.

> ⚠️ The exact protocols and counts depend on your file. Do not assume particular accuracy values in advance. Your results come from your own data and experiment.

---

## 🗂 Project Structure

```text
network-protocol-classification/
│
├── README.md                  # You are here
├── KT_GameBox_1.csv           # Dataset (add your own copy)
├── protocol_classification.py # Complete project script
├── notebook.ipynb             # Optional: step-by-step notebook
├── requirements.txt           # Python dependencies
└── images/                    # Saved plots for your report
    ├── protocol_distribution.png
    ├── model_comparison.png
    └── confusion_matrix_rf.png
```

**`requirements.txt`**

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## ⚙️ Setup and Installation

### Option A: Google Colab (easiest)

```python
!pip install pandas numpy matplotlib seaborn scikit-learn

from google.colab import files
uploaded = files.upload()   # choose KT_GameBox_1.csv
```

### Option B: Local machine (Anaconda / Jupyter / VS Code)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/network-protocol-classification.git
cd network-protocol-classification

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python protocol_classification.py
```

---

## 🪜 Step-by-Step Guide

### Step 1. Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

### Step 2. Load the dataset

```python
df = pd.read_csv("KT_GameBox_1.csv")
# For a quick start with a large file:
# df = pd.read_csv("KT_GameBox_1.csv", nrows=100000)

df.head()
```

### Step 3. Understand the data

```python
df.shape        # rows, columns
df.columns      # column names
df.dtypes       # data types
df.info()       # summary
df.describe()   # statistics
```

### Step 4. Clean the data

```python
# Missing values
print(df.isnull().sum())
df = df.dropna(subset=["Protocol"])   # target must not be missing

# Duplicates
print(df.duplicated().sum())
df = df.drop_duplicates()
print("Dataset shape:", df.shape)
```

### Step 5. Understand the target (`Protocol`)

```python
print(df["Protocol"].unique())
print(df["Protocol"].value_counts())
```

```python
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Protocol", order=df["Protocol"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Distribution of Network Protocols")
plt.xlabel("Protocol")
plt.ylabel("Number of Packets")
plt.show()
```

> 💡 **Think about it:** If one protocol has far more samples than the others, the dataset is **imbalanced**. This affects how you should interpret accuracy.

### Step 6. Analyze packet length

```python
df["Length"].describe()

plt.hist(df["Length"], bins=50)
plt.title("Distribution of Packet Length")
plt.xlabel("Packet Length")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x=df["Length"])
plt.show()
```

### Step 7. Feature engineering (time features)

```python
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

df["Hour"]   = df["Time"].dt.hour
df["Minute"] = df["Time"].dt.minute
df["Second"] = df["Time"].dt.second

df[["Time", "Hour", "Minute", "Second"]].head()
```

> ⚠️ Inspect `df["Time"].head()` first. Depending on your file's format, you may need to adjust this step (see [Common Problems](#-common-problems-and-fixes)).

### Step 8. Select features and target

```python
X = df[["Length", "Hour", "Minute", "Second"]]
y = df["Protocol"]

X = X.fillna(0)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

**Why not use `Source` and `Destination`?** IP addresses are categorical identifiers with a huge number of unique values, which makes a first model unnecessarily complicated.

### Step 9. Encode the target

```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(label_encoder.classes_)   # e.g. ['ARP' 'DNS' 'TCP' 'UDP']
print(y_encoded[:20])
```

### Step 10. Train/test split (80/20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded      # keeps protocol proportions equal in both sets
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
```

### Step 11. Feature scaling

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on training data
X_test_scaled  = scaler.transform(X_test)        # only transform the test data
```

> 🚨 **Data leakage rule:** use `fit_transform()` on training data only. Use `transform()` on test data. Never `fit_transform()` the test set.

### Step 12. Train the models

See [Models Used](#-models-used) below.

### Step 13. Compare the models

```python
results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_lr, accuracy_knn, accuracy_dt, accuracy_rf]
})
results["Accuracy (%)"] = results["Accuracy"] * 100
print(results)

plt.figure(figsize=(10, 6))
sns.barplot(data=results, x="Model", y="Accuracy (%)")
plt.xticks(rotation=20)
plt.ylim(0, 100)
plt.title("Comparison of ML Models")
plt.show()

best_model = results.loc[results["Accuracy"].idxmax()]
print("Best Model:")
print(best_model)
```

### Step 14. Feature importance (Random Forest)

```python
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance)

sns.barplot(data=feature_importance, x="Importance", y="Feature")
plt.title("Feature Importance - Random Forest")
plt.show()
```

> ❓ **Question for your report:** Which feature contributes most to protocol classification, and why do you think that is?

---

## 🤖 Models Used

| # | Model | Needs scaling? | Why it's included |
|---|-------|:--------------:|-------------------|
| 1 | **Logistic Regression** | ✅ Yes | Simple, fast baseline |
| 2 | **K-Nearest Neighbors (KNN)** | ✅ Yes | Distance-based, easy to understand |
| 3 | **Decision Tree** | ❌ No | Human-readable decision rules; great for teaching |
| 4 | **Random Forest** | ❌ No | Ensemble of trees; gives feature importance |

### Model 1: Logistic Regression

```python
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

y_pred_lr = logistic_model.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr * 100, "%")
```

### Model 2: K-Nearest Neighbors

```python
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn * 100, "%")
```

### Model 3: Decision Tree

```python
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt * 100, "%")
```

Visualize the first few levels (the full tree would be huge):

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    max_depth=3
)
plt.title("Decision Tree for Network Protocol Classification")
plt.show()
```

### Model 4: Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf * 100, "%")
```

---

## 📏 Evaluation Metrics

### Accuracy

The fraction of packets whose protocol was predicted correctly. Useful, but **not enough on its own**, especially with imbalanced classes.

### Classification Report

```python
print(classification_report(
    y_test, y_pred_rf,
    target_names=label_encoder.classes_
))
```

| Metric | What it tells you |
|--------|-------------------|
| **Precision** | Of the packets predicted as protocol X, how many really were X? |
| **Recall** | Of all real X packets, how many did the model find? |
| **F1-score** | Balance between precision and recall |
| **Support** | Number of real samples of that protocol in the test set |

### Confusion Matrix

```python
cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_rf, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.xlabel("Predicted Protocol")
plt.ylabel("Actual Protocol")
plt.title("Random Forest Confusion Matrix")
plt.show()
```

**How to read it:**

```text
Rows    = Actual protocol
Columns = Predicted protocol

Actual TCP → Predicted TCP   ✅ correct (diagonal)
Actual TCP → Predicted UDP   ❌ mistake (off-diagonal)
```

A strong model has large numbers on the **diagonal** and small numbers elsewhere.

---

## 🔮 Predicting a New Packet

```python
new_packet = pd.DataFrame({
    "Length": [64],
    "Hour":   [10],
    "Minute": [30],
    "Second": [15]
})

prediction = rf_model.predict(new_packet)
predicted_protocol = label_encoder.inverse_transform(prediction)

print("Predicted Protocol:", predicted_protocol[0])
```

The output depends entirely on your trained model.

---

## 🧠 Key Concepts Cheat Sheet

| Concept | One-line explanation |
|---------|----------------------|
| **Feature (X)** | An input column the model learns from |
| **Target (y)** | The column we want to predict (`Protocol`) |
| **Label encoding** | Converting text labels into numbers (TCP → 0, UDP → 1, ...) |
| **Train/test split** | Learn on one part of the data, evaluate on unseen data |
| **Stratify** | Keep class proportions the same in train and test sets |
| **Scaling** | Put features on a similar range so distance-based models work well |
| **Data leakage** | Test-set information accidentally influencing training |
| **Imbalanced data** | Some classes have far more samples than others |
| **Ensemble** | Combining many models (Random Forest = many trees) |
| **Feature importance** | How much each feature helped the model's decisions |

---

## 🛠 Common Problems and Fixes

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `FileNotFoundError` | CSV is not in the working directory | Check the path or re-upload the file in Colab |
| `KeyError: 'Protocol'` | Column names differ or have extra spaces | Run `df.columns` and `df.columns = df.columns.str.strip()` |
| `Hour`, `Minute`, `Second` are all `NaN` or look wrong | `Time` is likely a number of seconds since capture start (common in Wireshark exports), not a clock time | Check `df["Time"].head()`. If it is numeric, use it directly as a feature (e.g. `df["Time"].astype(float)`) instead of `pd.to_datetime` |
| Training is very slow | Too many rows (KNN and Random Forest) | Start with `nrows=100000` or fewer |
| `ConvergenceWarning` in Logistic Regression | Not enough iterations | Increase `max_iter` |
| `ValueError` in `train_test_split` with `stratify` | A protocol has only 1 sample | Remove rare classes or drop `stratify` |
| Very low accuracy on rare protocols | Class imbalance | Look at the classification report, not just accuracy |

---

## ⚠️ Limitations

Be honest about these in your report:

- **Few features.** Only packet length and time-based values are used. Real protocol identification relies on port numbers, flags, and payload structure.
- **Time features may add little.** Hour, minute, and second describe *when* a packet was captured, which may not relate to its protocol. Check feature importance to see.
- **Possible dataset-specific patterns.** The model may learn quirks of this particular capture rather than general network behavior.
- **Imbalanced classes** can make accuracy look better than the model really is.
- **Source and Destination are ignored** in this beginner version.

---

## 📝 Practice Exercises

### 🟢 Beginner
1. Change `test_size` to `0.30`. How do the results change?
2. Change `n_neighbors` in KNN to `3`, `7`, and `11`. Which is best?
3. Change `max_depth` in the Decision Tree to `3`, `5`, `10`, and `None`. What do you observe?

### 🟡 Intermediate
4. Increase the dataset from 100,000 rows to 500,000. Does performance change? How about training time?
5. Add `Length` bins (small, medium, large) as a new feature.
6. Remove the time features and retrain. Does accuracy drop?

### 🔴 Advanced
7. Encode `Source` and `Destination` (e.g. use only the top 20 most frequent addresses) and add them as features.
8. Extract keywords or port numbers from the `Info` column and use them as features.
9. Handle class imbalance using `class_weight="balanced"`.
10. Use `GridSearchCV` to tune Random Forest hyperparameters.

---

## 🚀 Future Improvements

- [ ] Add port numbers and TCP flags as features
- [ ] Try Gradient Boosting, XGBoost, or an SVM
- [ ] Apply cross-validation instead of a single split
- [ ] Handle imbalance with SMOTE or class weights
- [ ] Save the trained model with `joblib` and build a small prediction app
- [ ] Compare results across multiple KT GameBox capture files

---

## 📄 Suggested Report Structure

1. Introduction and problem statement
2. Dataset description
3. Data cleaning and preprocessing
4. Exploratory data analysis (include the protocol distribution and packet length plots)
5. Feature engineering and selection
6. Models and methodology
7. Results (model comparison chart, confusion matrix, classification report)
8. Feature importance discussion
9. Limitations
10. Conclusion and future work

---

## 🙋 Author

**Your Name** Soumadeep Naskar
Brainware University

---

## 📜 License

This project is for educational purposes. Add a license of your choice (for example MIT) before publishing.
