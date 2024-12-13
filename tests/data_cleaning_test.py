import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the functions from src.data_cleaning
from src.data_cleaning import (
    handle_missing_values,
    add_loan_categories,
    add_loan_income_ratio,
    add_risk_categories
)

# Sample DataFrame fixture for testing
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "fico_score": [750, 670, 610, 590, None],
        "monthly_installment": [500, 400, 300, 200, 100],
        "annual_income": [60000, 55000, 50000, 45000, 40000],
        "missing_values": [None, 2, None, 4, 5]
    })

# Test handle_missing_values with 'mean' strategy
def test_handle_missing_values_mean(sample_dataframe):
    result = handle_missing_values(sample_dataframe, strategy='mean', columns=['missing_values'])
    assert result['missing_values'].isna().sum() == 0
    assert result['missing_values'].iloc[0] == pytest.approx(3.6667, rel=1e-3)

# Test handle_missing_values with 'median' strategy
def test_handle_missing_values_median(sample_dataframe):
    result = handle_missing_values(sample_dataframe, strategy='median', columns=['missing_values'])
    assert result['missing_values'].isna().sum() == 0
    assert result['missing_values'].iloc[0] == 4

# Test handle_missing_values with 'drop' strategy
def test_handle_missing_values_drop(sample_dataframe):
    # Print the DataFrame before applying the drop
    print(f"Original DataFrame:\n{sample_dataframe}")
    
    # Apply the drop strategy
    result = handle_missing_values(sample_dataframe, strategy='drop', columns=['missing_values'])
    
    # Print the result after dropping rows
    print(f"Resulting DataFrame after drop:\n{result}")
    
    # Assert that the number of rows is 4 (one row should be dropped)
    assert result.shape[0] == 4  # One row dropped


# Test handle_missing_values with invalid strategy
def test_handle_missing_values_drop(sample_dataframe):
    print(f"Original DataFrame:\n{sample_dataframe}")

    result = handle_missing_values(sample_dataframe, strategy='drop', columns=['missing_values'])

    print(f"Resulting DataFrame after drop:\n{result}")

    # Assert that the number of rows is 3 (two rows should be dropped)
    assert result.shape[0] == 3 


    

# Test add_loan_categories function
def test_add_loan_categories(sample_dataframe):
    result = add_loan_categories(sample_dataframe, fico_column='fico_score')
    expected_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Unknown']
    
