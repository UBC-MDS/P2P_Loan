
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

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