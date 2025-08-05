import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils.preprocessing import load_and_prepare_data

def run_xgboost(dataset_name):
    X, y = load_and_prepare_data(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"XGBoost Accuracy on {dataset_name} data: {accuracy_score(y_test, preds):.2f}")
