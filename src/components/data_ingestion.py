import os
import sys
from src.logger import logging
from src.exception import CustomException
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import get_current_time

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
   
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        This function is responsible for data ingestion.
        It reads the data from the source, splits it into train and test sets,
        and saves them to the specified paths.
        """
        try:
            # Read the data from the source
            df = pd.read_excel("notebook\informality_data_for_analysis.xlsx")
            
            logging.info("Data read successfully from the source.")
            
            # Save the raw data to the specified path
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}.")
            
            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info(f"Train data saved to {self.ingestion_config.train_data_path}.")
            logging.info(f"Test data saved to {self.ingestion_config.test_data_path}.")
            
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys) from e
      
      
        
if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    data_transformation_obj = DataTransformation()
    train_arr,test_arr,_ =  data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
    logging.info("Train and test data saved successfully.")   
    
    model_trainer_obj = ModelTrainer()
    model_trainer_obj.train(train_arr, test_arr, data_transformation_obj)
    