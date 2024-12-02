# Import 
import os
import numpy as np
import pandas as pd
import altair as alt
import pandera as pa
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import set_config

# Enable the VegaFusion data transformer
alt.data_transformers.enable("vegafusion")

# Data Validation
check_prop = pa.Check(lambda s: s.isna().mean() <= 0.05,
                      element_wise=False,
                      error="Too many null values in 'credit.policy' column.")
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
        "not.fully.paid": pa.Column(int, checks=[check_prop,pa.Check.isin([0, 1])], nullable=True),
    },
    checks = [
        pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
        pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
    ]
    
)

schema.validate(p2ploan_df)

train_df, test_df = train_test_split(p2ploan_df, test_size=0.2, random_state=522)

# Save train data and test data to csv
train_df.to_csv("../data/processed/loan_train.csv")
test_df.to_csv("../data/processed/loan_test.csv")

# Data Validation: Anomalous Correlations
train_corr = train_df.corr(numeric_only=True)
pd.DataFrame(train_corr).style.format(
precision=2
).background_gradient(
    cmap="coolwarm", vmin=-1, vmax=1
).highlight_between(
    left=0.8,right=1, color="black"
).highlight_between(
    left=-1,right=-0.8, color="black"
)

# Data Validation: Target Distribution
print(train_df["not.fully.paid"].value_counts(normalize=True))
print(test_df["not.fully.paid"].value_counts(normalize=True))