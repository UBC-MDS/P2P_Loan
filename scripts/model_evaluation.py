# Import 
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import set_config
import click
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.write_csv import write_csv

@click.command()
@click.option('--data_from', type=str, help="Path to training data")
@click.option('--data_to', type=str, help="Path to cv results ")
@click.option('--pipeline_from', type=str, help="Path to the pipeline object")
@click.option('--preprocessor_from', type=str, help="Path to preprocessor object")


def main(data_from, pipeline_from, data_to, preprocessor_from):
    try:
        train_df = pd.read_csv(os.path.join(data_from, "loan_train.csv"))
        test_df = pd.read_csv(os.path.join(data_from, "loan_test.csv"))
        print(f"Data loaded successfully from {data_from}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    log_reg_search = pickle.load(open(pipeline_from, 'rb'))
    X_train = train_df.drop(columns="not.fully.paid")
    X_test = test_df.drop(columns="not.fully.paid")
    y_test = test_df["not.fully.paid"]
    
    y_pred_log_reg = log_reg_search.predict(X_test)
    accuracy_log_reg = round(accuracy_score(y_test, y_pred_log_reg), 4)
    write_csv(
        pd.DataFrame({"Accuracy Score":[accuracy_log_reg]}),
        data_to,
        "test_results.csv",
        index=True
    )

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
    write_csv(results_log_reg, data_to, "confusion_matrix.csv", index=False)
    
    preprocessor = pickle.load(open(preprocessor_from, "rb"))
    preprocessor.fit(X_train)
    coefficients = log_reg_search.best_estimator_.named_steps['LogReg'].coef_[0]

    positive_coef = pd.DataFrame(
        {"features":preprocessor.get_feature_names_out(),
        "positive coefficient": np.round(coefficients, decimals=4)}
    ).sort_values(by="positive coefficient", ascending=True, ignore_index=True)
    
    write_csv(positive_coef, data_to, "positive_coef.csv", index=False)


    negative_coef = pd.DataFrame(
        {"features":preprocessor.get_feature_names_out(),
        "negative coefficient": np.round(coefficients,decimals=4)}
    ).sort_values(by="negative coefficient", ascending=False, ignore_index=True)

    write_csv(negative_coef, data_to, "negative_coef.csv", index=False)

    print(f"Best model evaluation results successfully saved to {data_to}")
    
if __name__ == '__main__':
    main()