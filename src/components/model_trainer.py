import os 
import sys 
from dataclasses import dataclass 

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object ,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'K-Neighbors': KNeighborsClassifier(),
            'MLP Classifier': MLPClassifier(),
            'Ridge Classifier': RidgeClassifier(),
            'Extra Trees Classifier': ExtraTreesClassifier()
        }
        self.params = {
            'Logistic Regression': {
                'C': [0.1,1,10],
                "max_iter":[500,1000,1500]
                },
            'Random Forest': 
                {'n_estimators': [100,200,300], 
                 'max_depth': [5,10,15], 
                 'min_samples_split': [2,5,10],
                 'class_weight': [None, 'balanced']
                 },
            'Decision Tree': {
                'max_depth': [5,10,15], 
                'min_samples_split': [2,5,10], 
                'class_weight': [None, 'balanced']
                }, 
            'K-Neighbors': {
                'n_neighbors': [3,5,7], 
                'weights':['uniform', 'distance'], 
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
            'MLP Classifier': {
                'hidden_layer_sizes': [(50,), (100,), (150,)], 
                'activation': ['relu', 'tanh', 'logistic'], 
                'max_iter': [500, 1000, 1500], 
                'solver': ['adam', 'sgd', 'lbfgs'], 
                'learning_rate': ['constant', 'invscaling', 'adaptive'], 
                'alpha': [0.0001, 0.001, 0.01]
                },
            'Ridge Classifier': {
                'alpha':[0.1, 1.0, 10.0], 
                'max_iter': [1000, 2000, 3000]
                },
            'Extra Trees Classifier': {
                'n_estimators': [100, 200, 300], 
                'max_depth': [5, 10, 15], 
                'min_samples_split': [2, 5, 10], 
                'class_weight': [None, 'balanced']
                }
        }
                               
    def train(self, train_array,test_array,preprocessor):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            
            model_report = evaluate_models(X_train, y_train, X_test, y_test, self.models,params=self.params)
            best_model_score = max(list(model_report.values()))      
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = self.models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found with accuracy > 0.6", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            predection_model = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predection_model)
            
            # f1 = f1_score(y_test, predection_model, average='weighted')
            # precision = precision_score(y_test, predection_model, average='weighted')
            # recall = recall_score(y_test, predection_model, average='weighted')
            # roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            return accuracy#f1, precision, recall, roc_auc
        except Exception as e:
            raise CustomException(e, sys)

    