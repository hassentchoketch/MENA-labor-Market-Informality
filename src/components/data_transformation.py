import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler ,LabelEncoder

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        ''' 
        This function is used to transform the data using the following steps:
        1. impute missing values with median for numerical columns
        2. impute missing values with most frequent value for categorical columns
        '''
        try:
            # Define categorical columns, excluding the target variable
            # numerical_columns = []
            categorical_columns = ["country","Gender","Age","Marital status","Stratum Urban","Natur of work","Level of Wealth",
                                   "Fathers level education","Parent_affiliated_with Social Security","Participation in elections",
                                   "Freedom to speach out about government","Resort to nepotism","Trust in Parliment","Trust in employers",
                                   "Trust in associations","Trust in political_parties","Political system"
                                   ] 
            
            # num_pipeline = Pipeline(steps=[
                # ("imputer", SimpleImputer(strategy="median")), #impute missing values with median
                # ("scaler", StandardScaler()) #scale the data
            # ])     
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), #impute missing values with most frequent value
                ("onehotencoder", OneHotEncoder(handle_unknown='ignore')), #one hot encode the categorical variables
                # ("scaler", StandardScaler()) #scale the data
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    # ("num_pipeline", num_pipeline, numerical_columns), #apply numerical pipeline to numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns) #apply categorical pipeline to categorical columns
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")
            
            # Get preprocessor object for feature transformation
            preprocessor_obj = self.get_data_transformer_object()
            
            # Define target column and columns to drop
            target_column_name = 'Formality'
            drop_columns = [target_column_name,'Unnamed: 0','HM25_1']
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            
            logging.info("Applying preprocessing object on training and testing dataframes")
            
            # Transform the input features
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Transform the target variable
            # Create target encoder for the categorical target
            target_encoder = LabelEncoder()
            # Transform the target variable
            target_feature_train_arr = target_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = target_encoder.transform(target_feature_test_df)

            # Combine features and target into a single array
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]   
            
            logging.info("Preprocessing completed")
            
            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj)
            
    
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e, sys) from e 
                
            
            