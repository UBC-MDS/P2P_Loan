# Import 
import os
import pandas as pd
import click
import pickle 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import set_config
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

@click.command()
@click.option('--data_from', type=str, help="Path to split data")
@click.option('--data_to', type=str, help="Path to preprocessed data")
@click.option('--preprocessor_to', type=str, help="Path to preprocessor")

def main(data_from, data_to, preprocessor_to):

    # Load Data
    try:
        train_df = pd.read_csv(os.path.join(data_from, "loan_train.csv"))
        test_df = pd.read_csv(os.path.join(data_from, "loan_test.csv"))
        print(f"Data loaded successfully from {data_from}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split into training and testing sets
    # Features
    X_train = train_df.drop(columns=['not.fully.paid'])  
    X_test = test_df.drop(columns=['not.fully.paid' ]) 

    # Target
    y_train = train_df['not.fully.paid']               
    y_test = test_df['not.fully.paid']      

    # Define numeric and categorical columns
    numeric_features = [
        'int.rate', 'installment', 'log.annual.inc', 'dti', 
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
    os.makedirs(preprocessor_to, exist_ok=True)
    pickle.dump(preprocessor, open(os.path.join(preprocessor_to, "preprocessor.pickle"), "wb"))

    # Save transformed data to csv
    set_config(transform_output="pandas")
    preprocessor.fit(train_df)
    scale_train = preprocessor.transform(train_df)
    scale_test = preprocessor.transform(test_df)

    os.makedirs(os.path.join(data_to), exist_ok=True)
    write_csv(scale_train, data_to, "scaled_loan_train.csv", index=False)
    write_csv(scale_test, data_to, "scaled_loan_test.csv", index=False)

    print(f"Preprocessor successfully saved to {preprocessor_to}")
    print(f"Scaled data successfully saved to {data_to}")

if __name__ == '__main__':
    main()