# 🧠 Disease Prediction from Medical Data

**Author**: Shiva Teja

This project applies machine learning classification algorithms to predict the likelihood of diseases using structured medical datasets such as Heart Disease, Diabetes, and Breast Cancer, sourced from the UCI Machine Learning Repository.

## 🎯 Objective
Predict the presence or absence of diseases based on patient data such as symptoms, age, and test results using multiple classification techniques.

## 🛠️ Features
- Input data includes structured features like:
  - Age, Symptoms
  - Blood test results
  - Medical history
- Machine learning models used:
  - Support Vector Machine (SVM)
  - Logistic Regression
  - Random Forest
  - XGBoost

## 📁 Datasets Used
All datasets are publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/):
- Heart Disease Dataset
- Diabetes Dataset (PIMA Indian Diabetes)
- Breast Cancer Wisconsin (Diagnostic)

Place these datasets inside the `data/` folder:
```
data/
├── heart.csv
├── diabetes.csv
└── breast_cancer.csv
```

## 📦 Dependencies
Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `pandas`
- `scikit-learn`
- `xgboost`

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disease-prediction.git
   cd disease-prediction
   ```

2. Place the datasets in the `data/` folder as shown above.

3. Run the main script:
   ```bash
   python main.py
   ```

4. Edit `main.py` to choose the dataset:
   ```python
   dataset = "heart"  # or "diabetes", or "breast_cancer"
   ```

## 📊 Output
Each model prints the accuracy score for the selected dataset after training and testing.

## 📚 References
- [UCI ML Repository – Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [UCI ML Repository – Diabetes](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- [UCI ML Repository – Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

> Created with ❤️ by **Shiva Teja**
