import logging 
import os
import sys          
from datetime import datetime
from src.exception import CustomException
from src.utils import get_current_time



LOG_FILE_NAME = f"{get_current_time()}.log" 
LOG_DIR = os.path.join(os.getcwd(),"logs",LOG_FILE_NAME )
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME) 

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)   

