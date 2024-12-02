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


# Split into training and testing sets
# Features
X_train = train_df.drop(columns=['not.fully.paid'])  
X_test = test_df.drop(columns=['not.fully.paid' ]) 

# Target
y_train = train_df['not.fully.paid']               
y_test = test_df['not.fully.paid']      

# Define numeric and categorical columns
numeric_features = [
    'int.rate', 'installment', 'log.annual.inc', 'loan_income_ratio', 'dti', 
    'days.with.cr.line', 'revol.bal', 'revol.util', "fico",
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec', "credit.policy"
]
categorical_features = ['purpose']

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())                   # Scale numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))     # Encode categorical features
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Save transformed data to csv
set_config(transform_output="pandas")
preprocessor.fit(train_df)
scale_train = preprocessor.transform(train_df)
scale_train.to_csv("../data/processed/scaled_loan_train.csv")
scale_test = preprocessor.transform(test_df)
scale_test.to_csv("../data/processed/scaled_loan_test.csv")