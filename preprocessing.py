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
