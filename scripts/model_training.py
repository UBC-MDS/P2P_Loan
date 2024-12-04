# Import 
import click
import numpy as np
import pandas as pd
import os
import pickle
from sklearn import set_config
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from joblib import dump


@click.command()
@click.option('--data_from', type=str, help="Path to training data")
@click.option('--data_to', type=str, help="Path to cv results ")
@click.option('--preprocessor_from', type=str, help="Path to preprocessor object")


def main(data_from, preprocessor_from, data_to):
    '''Fits a Loan Default classifier to the training data and saves the results'''
    train_df = pd.read_csv(os.path.join(data_from, "loan_train.csv"))
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
    cv_results = cv_results.T

   
    np.round(cv_results,decimals=4).to_csv(os.path.join(data_to, "cv_results.csv"))


def model_cross_val(model, preprocessor, X_train, y_train):

    '''Perform 10-fold cross-validation on the given machine learning model 
    using a preprocessing pipeline. Returns a dictionary'''
    preprocessor = pickle.load(open(preprocessor, "rb"))
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




if __name__ == '__main__':
    main()