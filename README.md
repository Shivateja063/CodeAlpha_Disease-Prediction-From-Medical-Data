# 🧠 Disease Prediction from Medical Data

A machine learning-based project that predicts the possibility of diseases such as Heart Disease, Diabetes, and Breast Cancer using structured patient data. Built using Python and classification algorithms like SVM, Logistic Regression, Random Forest, and XGBoost on datasets from the UCI Machine Learning Repository.

---

## 📋 Features

- Predicts three types of diseases:
  - Heart Disease
  - Diabetes
  - Breast Cancer
- Uses real patient data from UCI Repository
- Multiple ML models with comparison
- Clean and modular code structure
- Easily switch between datasets and models
- Command-line runnable script

---

## 🛠️ Technologies Used

- **Python** – Core programming language
- **Pandas** – Data manipulation
- **Scikit-learn** – Machine learning models (SVM, Logistic Regression, Random Forest)
- **XGBoost** – Advanced boosting classifier
- **Jupyter / Google Colab / VS Code** – Development environments

---

## 🚀 Getting Started

To run the project locally:

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/disease-prediction.git
cd disease-prediction
2. Install the dependencies



pip install -r requirements.txt

3. Add the required datasets to the data/ folder:

heart.csv

diabetes.csv

breast_cancer.csv



4. Run the main script



python main.py

Edit the main.py file to select which dataset to use:

dataset = "heart"  # or "diabetes", or "breast_cancer"


---

📁 Project Structure

disease-prediction/
├── data/
│   ├── heart.csv                # Heart disease dataset
│   ├── diabetes.csv             # Diabetes dataset
│   └── breast_cancer.csv        # Breast cancer dataset
│
├── models/
│   ├── svm_model.py             # Support Vector Machine implementation
│   ├── logistic_regression.py  # Logistic Regression model
│   ├── random_forest.py        # Random Forest model
│   └── xgboost_model.py        # XGBoost model
│
├── utils/
│   └── preprocessing.py         # Data loading and preparation
│
├── main.py                      # Run all models
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


---

> 💡 by Shiva Teja

