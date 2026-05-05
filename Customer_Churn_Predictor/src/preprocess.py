import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Drop unnecessary columns if any.
    df = df.copy()

    # Convert categorical variables.
    df = pd.get_dummies(df, drop_first=True)

    # Separate features and target.
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y