Disease Prediction From Medical Data

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

-----

🧪 Technologies Used

Category	Technology

Programming Language	Python 3
Data Handling	Pandas
Machine Learning	scikit-learn, XGBoost
Algorithms	SVM, Logistic Regression, Random Forest, XGBoost
Data Source	UCI Machine Learning Repository
Development Environment	Jupyter Notebook / VS Code / Google Colab
Version Control	Git & GitHub


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
