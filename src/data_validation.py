import os
import pandas as pd
import pandera as pa
import math
import numpy as np
from pandera import Check
from sklearn.model_selection import train_test_split


def validate(data):
    # Check if correct data type is passed
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input is not a pandas DataFrame")    
    if data.empty:
        raise ValueError("Input Dataframe cannot be empty.")
    
    # Data Validation
    check_prop = pa.Check(lambda col: col.isna().mean() <= 0.05,
                          element_wise=False,
                          error="Too many null values in column.")

    schema = pa.DataFrameSchema(
        {
    "credit.policy": pa.Column(int, 
                                checks=[check_prop,
                                        pa.Check.isin([0, 1])], 
                                nullable=True),
    "purpose": pa.Column(
        str, 
        checks=[check_prop,
                pa.Check.isin([
                    "debt_consolidation", 
                    "all_other", 
                    "credit_card", 
                    "home_improvement", 
                    "small_business", 
                    "major_purchase", 
                    "educational"
        ])],
        nullable=True),
    "int.rate": pa.Column(float, checks=[check_prop,pa.Check.in_range(0, 1)], nullable=True),
    "installment": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "log.annual.inc": pa.Column(float, checks=[check_prop,pa.Check.ge(1)], nullable=True),
    "dti": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "fico": pa.Column(int, checks=[check_prop,pa.Check.in_range(300, 900)], nullable=True),
    "days.with.cr.line": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "revol.bal": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "revol.util": pa.Column(float, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "inq.last.6mths": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "delinq.2yrs": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "pub.rec": pa.Column(int, checks=[check_prop,pa.Check.ge(0)], nullable=True),
    "not.fully.paid": pa.Column(int, checks=[check_prop,pa.Check.isin([0, 1])], nullable=False),
    },
        checks = [
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
    ])

    schema.validate(data, lazy=True)


    train_df, test_df = train_test_split(data, test_size=0.2, random_state=522)
    train_dist = train_df["not.fully.paid"].value_counts(normalize=True)[0]
    test_dist = test_df["not.fully.paid"].value_counts(normalize=True)[0]

    # Data Validation: Train/Test Target Distribution
    if math.isclose(train_dist, test_dist, abs_tol=0.05) == False:
        raise ValueError("Train/Test Target Distribution Mismatch")
    
    # Data Validation: Anomalous Correlations
    train_corr = train_df.corr(numeric_only=True)
    corr = train_corr.apply(pd.Series.between, axis=1, left=-0.9, right=0.9, inclusive="neither")
    num_feats = train_corr.index.tolist()
    corr_cols = []

    for i in np.arange(len(num_feats)):
        corr.iloc[:i+1, i] = True

    for row in num_feats:
        for col in num_feats:
            if corr.at[col, row] == False: 
                corr_cols.append((col, row))
                
    if corr_cols:
        raise ValueError(f"Anomalous Correlations between the following columns: \n {corr_cols}")

    schema.validate(data, lazy=True)

