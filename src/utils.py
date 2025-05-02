from datetime import datetime
import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


def get_current_time() -> str:
    """
    Get the current time in the format YYYY-MM-DD_HH-MM-SS.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 

def save_object(file_path: str, obj: object) -> None:
    """
    Save an object to a file using dill.
    """
    try:
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
    
def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models: dict, params: dict = None) -> dict:
    """
    Evaluate multiple models and return their performance metrics.
    """
    model_report = {}
    
    for model_name, model in models.items():
        try:
            # Set model parameters if available
            if model_name in params:
                model.set_params(**params[model_name])
            
            # Fit the model and make predictions
            gs = GridSearchCV(model, params[model_name], cv=5, scoring='accuracy')
            gs.fit(X_train, y_train)    
            model = gs.best_estimator_
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            # f1 = f1_score(y_test, y_pred, average='weighted')
            # precision = precision_score(y_test, y_pred, average='weighted')
            # recall = recall_score(y_test, y_pred, average='weighted')
            # roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            model_report[model_name] = accuracy
            # {
            #     'accuracy': accuracy,
            #     'f1_score': f1,
            #     'precision': precision,
            #     'recall': recall,
            #     # 'roc_auc': roc_auc
            # }
        except Exception as e:
            raise CustomException(e, sys) from e
    
    return model_report


def load_object(file_path: str) -> object:
    """
    Load an object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys) from e