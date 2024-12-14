# Import 
import click
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_cv import model_cross_val
from src.write_csv import write_csv


@click.command()
@click.option('--data_from', type=str, help="Path to training data")
@click.option('--data_to', type=str, help="Path to cv results ")
@click.option('--preprocessor_from', type=str, help="Path to preprocessor object")


def main(data_from, preprocessor_from, data_to):
    '''Fits a Loan Default classifier to the training data and saves the results'''
    try:
        train_df = pd.read_csv(os.path.join(data_from, "loan_train.csv"))
        print(f"Data loaded successfully from {data_from}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    X_train = train_df.drop(columns="not.fully.paid")
    y_train = train_df["not.fully.paid"]

    # Define Models
    dt = DecisionTreeClassifier(random_state=123)
    knn = KNeighborsClassifier(n_jobs=-1)
    svc = SVC(random_state=123)
    log_reg = LogisticRegression(random_state=123)

    models = {"Decision Tree": dt, 
            "kNN": knn,
            "SVC": svc,
            "Logistic Regression": log_reg}
    cv_results = pd.DataFrame()
    for (name, model) in models.items():
        cv_results[name] = model_cross_val(model, preprocessor_from, X_train, y_train)
   
    write_csv(np.round(cv_results.T, decimals=4), data_to, "cv_results.csv", index=False)
    write_csv(
        pd.DataFrame(train_df["not.fully.paid"].value_counts(normalize=True)),
        data_to,
        "target_dist.csv",
        index=True
    )
    print(f"Model selection results successfully saved to {data_to}")

if __name__ == '__main__':
    main()