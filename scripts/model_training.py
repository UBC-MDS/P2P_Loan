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

# Define Models
dt = DecisionTreeClassifier(random_state=123)
knn = KNeighborsClassifier(n_jobs=-1)
svc = SVC(random_state=123)
log_reg = LogisticRegression(random_state=123)

models = {"Decision Tree": dt, 
          "kNN": knn,
          "SVC": svc,
          "Logistic Regression": log_reg}


def model_cross_val(model):

    '''Perform 10-fold cross-validation on the given machine learning model 
    using a preprocessing pipeline. Returns a dictionary'''
    
    model_pipeline = Pipeline([
            ('preprocessor', preprocessor),  
            ('model', model)
    ])

    results = pd.DataFrame(cross_validate(
        model_pipeline, X_train, y_train, return_train_score=True, cv=10
    ))

    mean_std = pd.DataFrame({"mean":results.mean(),
                             "stdev":results.std()})
    
    result_dict = {index: f"{mu:.3f}(+/-{std:.3f})" # Concat std with mean
                   for (index, mu, std) in mean_std.itertuples()}
    
    return result_dict

cv_results = pd.DataFrame()
for (name, model) in models.items():
    cv_results[name] = model_cross_val(model)
cv_results = cv_results.T

cv_results

