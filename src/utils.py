from datetime import datetime
import os 
import sys

def get_current_time() -> str:
    """
    Get the current time in the format YYYY-MM-DD_HH-MM-SS.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 