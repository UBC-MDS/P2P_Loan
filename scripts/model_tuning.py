
# imports
import click
import numpy as np
import pandas as pd
import os
import pickle
from sklearn import set_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, make_scorer
from joblib import dump
from sklearn.model_selection import GridSearchCV



@click.command()
@click.option('--data_from', type=str, help="Path to training data")
@click.option('--data_to', type=str, help="Path to cv results ")
@click.option('--pipeline_to', type=str, help="Path to the pipeline object")
@click.option('--preprocessor_from', type=str, help="Path to preprocessor object")


def main(data_from, preprocessor_from,data_to, pipeline_to):
    '''hyper parameter tuning for logistic model 
    and saves the pipeline object.'''
    try:
        train_df = pd.read_csv(os.path.join(data_from, "loan_train.csv"))
        print(f"Data loaded successfully from {data_from}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    X_train = train_df.drop(columns="not.fully.paid")
    y_train = train_df["not.fully.paid"]
    
    # Logistic Regression Tuning
    log_reg_param_dist = {
        "LogReg__C": np.logspace(-5, 5)
    }
    preprocessor = pickle.load(open(preprocessor_from, "rb"))

    log_reg_pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('LogReg', LogisticRegression(random_state=123, max_iter=20000))
    ])

    log_reg_search = GridSearchCV(
        log_reg_pipe,
        param_grid=log_reg_param_dist,
        cv=10,
        n_jobs=-1,
        return_train_score=True
    )

    log_reg_search.fit(X_train, y_train)

    pickle.dump(log_reg_search, open(os.path.join(pipeline_to, "pipeline.pickle"), "wb"))
    
    cv_results = pd.DataFrame(log_reg_search.cv_results_)[[
        "rank_test_score",
        "param_LogReg__C",
        "mean_test_score",
        "mean_train_score"
    ]]

    cv_results =  np.round(cv_results, decimals=6).sort_values(by="rank_test_score").head(5)
    cv_results.to_csv(os.path.join(data_to, "model_results.csv"))
    
    
if __name__ == '__main__':
    main()