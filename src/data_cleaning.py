import pandas as pd

# Handle Missing Values

def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handles missing values in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    strategy : str
        Strategy to handle missing values ('mean', 'median', or 'drop').
    columns : list or None
        Columns to apply the strategy. If None, applies to all columns.

    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    if columns is None:
        columns = df.columns

    if strategy == 'mean':
        for col in columns:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in columns:
            df[col] = df[col].fillna(df[col].median())
    elif strategy == 'drop':
        df = df.dropna(subset=columns)
    else:
        raise ValueError("Invalid strategy. Choose 'mean', 'median', or 'drop'.")

    return df

# Add Loan Categories
import numpy as np

def add_loan_categories(df, fico_column):
    """
    Add loan categories to a DataFrame based on FICO scores.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a column with FICO scores.
    fico_column : str
        Column name for FICO scores.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional 'loan_categories' column.
    """
    loan_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Deep subprime']
    fico_conditions = [
        (df[fico_column] >= 720),
        (df[fico_column] < 720) & (df[fico_column] >= 660),
        (df[fico_column] < 660) & (df[fico_column] >= 620),
        (df[fico_column] < 620) & (df[fico_column] >= 580),
        (df[fico_column] < 580)
    ]
    df['loan_categories'] = np.select(fico_conditions, loan_categories, default='Unknown')
    return df

# Add Loan Categories
import numpy as np

def add_loan_categories(df, fico_column):
    """
    Add loan categories to a DataFrame based on FICO scores.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing a column with FICO scores.
    fico_column : str
        Column name for FICO scores.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional 'loan_categories' column.
    """
    loan_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Deep subprime']
    fico_conditions = [
        (df[fico_column] >= 720),
        (df[fico_column] < 720) & (df[fico_column] >= 660),
        (df[fico_column] < 660) & (df[fico_column] >= 620),
        (df[fico_column] < 620) & (df[fico_column] >= 580),
        (df[fico_column] < 580)
    ]
    df['loan_categories'] = np.select(fico_conditions, loan_categories, default='Unknown')
    return df

# Add Loan-to-Income Ratio

def add_loan_income_ratio(df, installment_column, income_column):
    """
    Add loan-to-income ratio as a new column to a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame.
    installment_column : str
        Column name for monthly loan installments.
    income_column : str
        Column name for annual income.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional 'loan_income_ratio' column.
    """
    df['loan_income_ratio'] = (df[installment_column] * 12) / df[income_column]
    return df

# Risk Categories

def add_risk_categories(df, fico_column):
    """
    Add risk categories to a DataFrame based on FICO scores.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing FICO scores.
    fico_column : str
        Column name for FICO scores.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional 'risk_category' column.
    """
    conditions = [
        (df[fico_column] >= 720),
        (df[fico_column] < 720) & (df[fico_column] >= 650),
        (df[fico_column] < 650)
    ]
    categories = ['Low Risk', 'Medium Risk', 'High Risk']
    df['risk_category'] = np.select(conditions, categories, default='Unknown')
    return df
