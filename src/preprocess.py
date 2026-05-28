import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(path="data/raw/housing.csv"):
    df = pd.read_csv(path)
    print("=== LOADING DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    return df


def create_features(df):
    """
    Feature engineering — creating new meaningful columns
    from existing ones to help the model learn better
    """
    df = df.copy()

    # rooms per person — more meaningful than total rooms
    df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]

    # bedrooms ratio — what % of rooms are bedrooms
    df["BedroomRatio"] = df["AveBedrms"] / df["AveRooms"]

    # income per room — combines two important features
    df["IncomePerRoom"] = df["MedInc"] / df["AveRooms"]

    print("\n=== FEATURE ENGINEERING ===")
    print(f"Original features: 8")
    print(f"New features added: 3")
    print(f"Total features: {df.shape[1] - 1}")

    return df


def preprocess(path="data/raw/housing.csv"):
    # load
    df = load_data(path)

    # feature engineering
    df = create_features(df)

    # separate features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    print(f"\n=== SPLITTING DATA ===")
    print(f"Total samples: {len(df)}")

    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {list(X.columns)}")

    return X_train, X_test, y_train, y_test


def get_pipeline():
    """
    Preprocessing pipeline —
    impute missing values then scale features
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


if __name__ == "__main__":
    preprocess()