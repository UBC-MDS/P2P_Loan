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

y_pred_log_reg = log_reg_search.predict(X_test)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Model Accuracy: {accuracy_log_reg:.4f}")

pred_true = pd.DataFrame({"prediction":y_pred_log_reg, "true":y_test})
results_log_reg = pd.DataFrame(
    {   
         " ": ["True Positive (defaulted)", "True Negative (fully paid)"],
        "Predict Positive (defaulted)": [
            len(pred_true.query("prediction == 1 & true == 1")),
            len(pred_true.query("prediction == 1 & true == 0"))
            
        ],
        "Predict Negative (fully paid)": [
            len(pred_true.query("prediction == 0 & true == 1")),
            len(pred_true.query("prediction == 0 & true == 0"))
    
        ]
    }
)
results_log_reg

preprocessor.fit(X_train)
coefficients = log_reg_search.best_estimator_.named_steps['LogReg'].coef_[0]

pd.DataFrame(
    {"features":preprocessor.get_feature_names_out(),
     "negative coefficient": coefficients}
).sort_values(by="negative coefficient", ascending=True, ignore_index=True).head(3)

pd.DataFrame(
    {"features":preprocessor.get_feature_names_out(),
     "positive coefficient": coefficients}
).sort_values(by="positive coefficient", ascending=False, ignore_index=True).head(3)