import pandas as pd
import numpy as np
import os
from sklearn.datasets import fetch_california_housing

# make sure folders exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# load dataset directly from sklearn
housing = fetch_california_housing()

# convert to dataframe
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# add target column
df["MedHouseVal"] = housing.target

# basic info
print("=== CALIFORNIA HOUSING DATASET ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# save to raw folder
df.to_csv("data/raw/housing.csv", index=False)
print("\n✅ Dataset saved to data/raw/housing.csv")