import pytest
import os
import sys
import pandera as pa
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_validation import validate
import math

# Test Data

data = pd.read_csv("data/processed/loan_train.csv")
test_data = data.sample(100, random_state=123, ignore_index=True)
invalid_cases = []

# Case: Wrong data type for data
def test_data_type():
    with pytest.raises(TypeError, match="Input is not a pandas DataFrame"):
        validate(test_data.to_numpy())

# Case: Empty data frame input
def test_empty_df():
    with pytest.raises(ValueError, match="Input Dataframe cannot be empty."):
        validate(pd.DataFrame())

# Case: Train/Test target distribution mismatch
def test_target_mismatch():
    invalid_data = test_data.copy()
    invalid_data["not.fully.paid"] = np.ones(100, dtype=int)
    invalid_data.loc[0:9, "not.fully.paid"] = int(0)
    with pytest.raises(ValueError, match="Train/Test Target Distribution Mismatch"):
        validate(invalid_data)
    
# Case: Anomalous Correlations
def test_corr():
    invalid_data = test_data.copy()
    invalid_data["not.fully.paid"] = np.ones(100, dtype=int)
    invalid_data.loc[0:20, "not.fully.paid"] = int(0)
    invalid_data["installment"] = invalid_data["revol.util"] + 1
    with pytest.raises(ValueError, match=r"Anomalous Correlations between the following columns:*"):
        validate(invalid_data)

# Case: Missing column
for col in test_data.columns:
    invalid_cases.append((test_data.drop(columns=[col]), f"Check for missing column: {col}"))

# Case: Invalid class labels - purpose
invalid_data = test_data.copy()
invalid_data["purpose"] = invalid_data["purpose"].replace("all_other", "others")
invalid_cases.append((invalid_data, "Check for invalid class labels - purpose"))

    
# Case: Data out of range
range_col = ["int.rate", "fico", "credit.policy", "not.fully.paid"]
for col in range_col:
    invalid_data = test_data.copy()
    invalid_data.loc[:10, col] = 1000
    invalid_cases.append((invalid_data, f"Check '{col}'for out of range values (too large)"))

num_col = test_data.drop(columns=["purpose"])
for col in num_col:
    invalid_data = test_data.copy()
    invalid_data.loc[:10, col] = -1
    invalid_cases.append((invalid_data, f"Check '{col}'for out of range values (too small)"))

# Case: Data with wrong dtype
for col in num_col:
    invalid_data = test_data.copy()
    invalid_data[col] = invalid_data[col].astype(str)
    invalid_cases.append((invalid_data, f"Check '{col}' for incorrect data type"))

# Case: Duplicate observations
invalid_data = test_data.copy()
invalid_data = pd.concat([invalid_data, invalid_data.iloc[[0]]], ignore_index=True)
invalid_cases.append((invalid_data, "Check for duplicate rows"))

# Case: Check for missing observation
invalid_data = test_data.copy()
invalid_data.iloc[0, :] = np.nan
invalid_cases.append((invalid_data, "Check for empty rows"))


# Parameterize invalid data test cases
@pytest.mark.parametrize("invalid_data, description", invalid_cases)
def test_invalid_data(invalid_data, description):
    with pytest.raises(pa.errors.SchemaErrors):
        validate(invalid_data)


