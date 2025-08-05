from models.svm_model import run_svm
from models.logistic_regression import run_logistic
from models.random_forest import run_rf
from models.xgboost_model import run_xgboost

# Choose dataset: "heart", "diabetes", or "breast_cancer"
dataset = "heart"

print("\nRunning SVM:")
run_svm(dataset)

print("\nRunning Logistic Regression:")
run_logistic(dataset)

print("\nRunning Random Forest:")
run_rf(dataset)

print("\nRunning XGBoost:")
run_xgboost(dataset)
