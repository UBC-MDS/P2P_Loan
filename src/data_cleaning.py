import pandas as pd
import numpy as np

# Handle Missing Values

def handle_missing_values(df, strategy='mean', columns=None):
    if columns is None:
        columns = df.columns
    
    # Debugging: Print shape before and after dropping NaN values
    print("Before dropna:", df.shape)
    
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    
    print("After dropna:", df.shape)
    
    # Other strategies: mean, median
    if strategy == 'mean':
        for col in columns:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df



# Add Loan Categories
def add_loan_categories(df, fico_column):
    loan_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Deep subprime']
    fico_conditions = [
        (df[fico_column] >= 720),  # Super-prime
        (df[fico_column] < 720) & (df[fico_column] >= 660),  # Prime
        (df[fico_column] < 660) & (df[fico_column] >= 620),  # Near-prime
        (df[fico_column] < 620) & (df[fico_column] >= 580),  # Subprime
        (df[fico_column] < 580)  # Deep subprime
    ]
    df['loan_categories'] = np.select(fico_conditions, loan_categories, default='Unknown')
    return df



# Add Risk Categories
def add_risk_categories(df, fico_column):
    conditions = [
        (df[fico_column] >= 720),  # Low Risk
        (df[fico_column] < 720) & (df[fico_column] >= 650),  # Medium Risk
        (df[fico_column] < 650)  # High Risk
    ]
    categories = ['Low Risk', 'Medium Risk', 'High Risk']
    
    df['risk_category'] = np.select(conditions, categories, default='Unknown')
    
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


