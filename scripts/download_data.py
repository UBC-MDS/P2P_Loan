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

# Define the project root directory (adjust the path if needed)
file_path = os.path.join("..", "data", "raw", "loan_data.csv")

p2ploan_df = pd.read_csv(file_path)


# Check the data
p2ploan_df.head()

