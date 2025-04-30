from datetime import datetime
import os 
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill


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