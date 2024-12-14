
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

def model_cross_val(model, preprocessor, X_train, y_train):  
    '''
    Perform 10-fold cross-validation on a given machine learning model using a preprocessing pipeline.
    
    This function loads a preprocessing pipeline from a pickle file, constructs a pipeline combining
    it with the specified model, and evaluates the model's performance using 10-fold cross-validation
    on the provided training data. The results are returned as a dictionary containing the mean and 
    standard deviation of various cross-validation metrics.
    
    Parameters:
    -----------
    model : sklearn.base.BaseEstimator
        A machine learning model that implements the fit and predict methods (e.g., classifiers, regressors).
        
    preprocessor : str
        Path to the pickle file containing the preprocessing pipeline.
        
    X_train : pandas.DataFrame or numpy.ndarray
        Training features, where rows represent samples and columns represent features.
        
    y_train : pandas.Series or numpy.ndarray
        Target labels corresponding to the rows of `X_train`.
    
    Returns:
    --------
    result_dict : dict
        A dictionary where each key corresponds to a metric from the cross-validation results and each value is a string of the form 'mean(+/-stdev)', indicating the 
        mean and standard deviation of that metric across the folds.
        
    Example:
    --------
    model = DummyClassifier()
    preprocessor = 'preprocessor.pickle'  
    result = model_cross_val(model, preprocessor, X_train, y_train)
    print(result)
    '''
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