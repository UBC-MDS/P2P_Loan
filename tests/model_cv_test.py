import pytest
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.dummy import DummyClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_cv import model_cross_val
import math
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

# Test data setup
@pytest.fixture
def test_data():
    data = pd.read_csv("data/processed/loan_train.csv")
    data = data.sample(100, random_state=123)
    X_mock = data.drop(columns=["not.fully.paid"])
    y_mock = data["not.fully.paid"]
    return [X_mock, y_mock]

@pytest.fixture
def invalid_dir():
    return "results/model/preprocessor.pickle"

@pytest.fixture
def invalid_model():
    return "results/models/pipeline.pickle"

@pytest.fixture
def invalid_file():
    return "results/tables/cv_results.csv"

@pytest.fixture
def valid_model():
    return "results/models/preprocessor.pickle"



# Check if the correct path is passed for the preprocessor.
# Test if FileNotFoundError is raised if file path does not exist.
def test_preprocessor_invalid_path(invalid_dir, test_data):
    with pytest.raises(FileNotFoundError, match=r"No such file or directory: *"):
        model_cross_val(DummyClassifier(), invalid_dir, *test_data)

# Test if ValueError is raised if the pickle file is not a preprocessor.
def test_invalid_model(invalid_model, test_data):
    with pytest.raises(ValueError, match=r"All the 10 fits failed.*"):
        model_cross_val(DummyClassifier(), invalid_model, *test_data)

# Test if UnpicklingError is raised if the file is not a pickle file. 
def test_invalid_file(invalid_file, test_data):
    with pytest.raises(pickle.UnpicklingError):
        model_cross_val(DummyClassifier(), invalid_file, *test_data)


# Test for correct cross validation output for model_cross_cal
def test_cv_output(valid_model, test_data):
    cv_dict = model_cross_val(DummyClassifier(), valid_model, *test_data)
    # Check output is a dictionary
    assert isinstance(cv_dict, dict)

    # Check the dictionary has the correct keys
    keys = list(cv_dict.keys())
    assert keys == ['fit_time', 'score_time', 'test_score', 'train_score']

    # Check the dictionary values are all strings
    values = list(cv_dict.values())
    assert all(isinstance(value, str) for value in values)

    # Check the function returns the correct train/test score
    assert cv_dict['test_score'] == '0.790(+/-0.032)'
    assert cv_dict['train_score'] == '0.790(+/-0.004)'
    
    




