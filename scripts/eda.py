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

p2ploan_df.info()
train_df.shape
# Summary Statistics
train_df.describe(include="all")

# Let us store the column names of the columns with missing values as a list in a variable called missing_vals_cols.
missing_vals_cols = train_df.columns[train_df.isna().sum() > 0].tolist()

# Define numeric columns explicitly
numeric_cols = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
    'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'annual.inc']

# income- level data
train_df['annual.inc'] = np.exp(train_df['log.annual.inc'])
test_df['annual.inc'] = np.exp(test_df['log.annual.inc'])

train_df['loan_income_ratio'] = (train_df['installment'] * 12) / train_df['annual.inc']

# Creating loan Categories
conditions = [
    (train_df['fico'] >= 720),
    (train_df['fico'] < 719) & (train_df['fico'] >= 660),
    (train_df['fico'] < 659) & (train_df['fico'] >= 620),
    (train_df['fico'] < 619) & (train_df['fico'] >= 580),
    (train_df['fico'] < 580)
]
loan_categories = ['Super-prime', 'Prime', 'Near-prime', 'Subprime', 'Deep subprime']
train_df['loan_categories'] = np.select(conditions, loan_categories, default='Unknown')

test_df['loan_income_ratio'] = (test_df['installment'] * 12) / test_df['annual.inc']
conditions = [
    (test_df['fico'] >= 720),
    (test_df['fico'] < 719) & (test_df['fico'] >= 660),
    (test_df['fico'] < 659) & (test_df['fico'] >= 620),
    (test_df['fico'] < 619) & (test_df['fico'] >= 580),
    (test_df['fico'] < 580)
]
test_df['loan_categories'] = np.select(conditions, loan_categories, default='Unknown')

# Creating Risk Categories
conditions = [
    (train_df['fico'] >= 720),
    (train_df['fico'] < 720) & (train_df['fico'] >= 650),
    (train_df['fico'] < 650)
]
categories = ['Low Risk', 'Medium Risk', 'High Risk']
train_df['risk_category'] = np.select(conditions, categories, default='Unknown')

conditions = [
    (test_df['fico'] >= 720),
    (test_df['fico'] < 720) & (test_df['fico'] >= 650),
    (test_df['fico'] < 650)
]
test_df['risk_category'] = np.select(conditions, categories, default='Unknown')


for feat in numeric_cols:
    train_df.groupby("not.fully.paid")[feat].plot.hist(bins=40, alpha=0.4, legend=True, density=True, title = "Histogram of " + feat)
    plt.xlabel(feat)
    plt.show()
    
# Data distribution of selected loan features
numeric_cols_hists = alt.Chart(train_df).mark_bar().encode(
    alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=20)),  
    y='count()'
).properties(
    width=250,
    height=175
).repeat(
    ['installment', 'dti'],  
    columns=3
)
numeric_cols_hists

# Default Rate by Loan Purpose:

# Explode 'purpose' column for analysis
loan_purpose_data = train_df.explode('purpose')

# Loan Category vs Loan Purpose
purpose_risk_chart = alt.Chart(loan_purpose_data).mark_circle().encode(
    x=alt.X('loan_categories:N', title='Loan Categories', sort='-color', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('purpose:N', title='Loan Purpose', sort='color'),
    color=alt.Color('count()', scale=alt.Scale(scheme='viridis'), title='Loan Count'),
    size=alt.Size('count()', title='Loan Count', scale=alt.Scale(range=[50, 1500])),
    tooltip=['purpose', 'loan_categories', 'count()']
).properties(
    width=600,
    height=400,
    title="Loan Category vs Loan Purpose"
)

purpose_risk_chart


categories_hist = alt.Chart(train_df).mark_bar().encode(
    x=alt.X('risk_category:N', title='Risk Categories', axis=alt.Axis(labelAngle=0)),  
    y=alt.Y('count()', title='Count') 
).properties(
    height=300,
    width=400,
    title="Distribution of Risk Categories"
)

categories_hist

#fico by loan purpose
purpose_fico_boxplot = alt.Chart(loan_purpose_data).mark_boxplot().encode(
    y=alt.Y('purpose:N', title='Loan Purpose', sort='-x'),  
    x=alt.X('fico:Q', title='FICO Score', scale=alt.Scale(domain=[600, 850]),),  
    color=alt.Color('purpose:N', legend=None),  
    tooltip=['purpose', 'fico']
).properties(
    width=400,
    height=200,
    title='Boxplot of FICO Scores by Loan Purpose'
)


#Debt to income ratio by risk level
risk_dti_boxplot = alt.Chart(train_df).mark_boxplot().encode(
    y=alt.Y('risk_category:N', title='Risk Category', sort='-x'),  
    x=alt.X('dti:Q', title='DTI (Debt-to-Income)', scale=alt.Scale(domain=[0, 35])),  
    color=alt.Color('risk_category:N', legend=None),  
    tooltip=['risk_category', 'dti']
).properties(
    width=400,
    height=200,
    title='Boxplot of DTI by Risk Category'
)


purpose_fico_boxplot & risk_dti_boxplot

# Select only numeric columns
numeric_cols = train_df[['int.rate', 'installment', 'log.annual.inc', 'dti', 
                         'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
                         'inq.last.6mths', 'annual.inc']]
# Calculate the correlation matrix
correlation_matrix = numeric_cols.corr().reset_index().melt('index')
correlation_matrix.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Create a heatmap using Altair
correlation_chart = alt.Chart(correlation_matrix).mark_rect().encode(
    x=alt.X('Variable 1:N', title='', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('Variable 2:N', title=''),
    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),
    tooltip=['Variable 1', 'Variable 2', 'Correlation']
).properties(
    width=400,
    height=400,
    title="Correlation Heatmap"
)

correlation_chart