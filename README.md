📁 Project Structure

disease-prediction/
│
├── data/
│   ├── heart.csv
│   ├── diabetes.csv
│   └── breast_cancer.csv
│
├── models/
│   ├── svm_model.py
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── utils/
│   └── preprocessing.py
│
├── main.py
├── requirements.txt
└── README.md


---

🧠 Example: main.py

from models.svm_model import run_svm
from models.logistic_regression import run_logistic
from models.random_forest import run_rf
from models.xgboost_model import run_xgboost

# Choose dataset (heart, diabetes, breast cancer)
dataset = "heart"  # change as needed

print("\nRunning SVM:")
run_svm(dataset)

print("\nRunning Logistic Regression:")
run_logistic(dataset)

print("\nRunning Random Forest:")
run_rf(dataset)

print("\nRunning XGBoost:")
run_xgboost(dataset)


---

🧪 Sample Model Script: models/svm_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils.preprocessing import load_and_prepare_data

def run_svm(dataset_name):
    X, y = load_and_prepare_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"SVM Accuracy on {dataset_name} data: {accuracy_score(y_test, preds):.2f}")


---

🔧 Utility: utils/preprocessing.py

import pandas as pd

def load_and_prepare_data(dataset_name):
    if dataset_name == "heart":
        df = pd.read_csv("data/heart.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
    elif dataset_name == "diabetes":
        df = pd.read_csv("data/diabetes.csv")
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
    elif dataset_name == "breast_cancer":
        df = pd.read_csv("data/breast_cancer.csv")
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"].map({"M": 1, "B": 0})
    else:
        raise ValueError("Invalid dataset name")
    return X, y


---

📦 requirements.txt

pandas
scikit-learn
xgboost


---

📝 README.md (Summary)

# Disease Prediction from Medical Data

This project uses machine learning classification algorithms to predict the presence of diseases based on medical datasets such as Heart Disease, Diabetes, and Breast Cancer from the UCI ML Repository.

## 📌 Features
- Uses structured medical data (age, symptoms, test results).
- Algorithms: SVM, Logistic Regression, Random Forest, XGBoost.
- Clean modular code using Python.

## 📁 Datasets
Place these in the `data/` directory:
- Heart Disease
- Diabetes
- Breast Cancer

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run main script:

python main.py


3. Edit main.py to switch datasets.



📊 Output

Each model prints its accuracy score after training/testing.

📚 Sources

UCI Machine Learning Repository


---

        y = df["target"]
    elif dataset_name == "diabetes":
        df = pd.read_csv("data/diabetes.csv")
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
    elif dataset_name == "breast_cancer":
        df = pd.read_csv("data/breast_cancer.csv")
        X = df.drop("diagnosis", axis=1)
        y = df["diagnosis"].map({"M": 1, "B": 0})
    else:
        raise ValueError("Invalid dataset name")
    return X, y


---

📦 requirements.txt

pandas
scikit-learn
xgboost
----
