import os
import sys
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import dill


def save_object(file_path,obj):
    '''
    Function saving object as pickle file.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_path)
    except Exception as e:
        CustomException(e,sys)