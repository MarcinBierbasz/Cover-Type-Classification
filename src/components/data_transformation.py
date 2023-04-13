import os
import sys
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        '''
        Function specifying data transformation process.
        '''
        try:
            preprocessor = Pipeline(
                steps = [('scaler',StandardScaler())]
            )
            return preprocessor
            logging.info("Preprocessing pipeline created")
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        '''
        Function responsible for data transformation.
        '''
        try:
            logging.info("Loading train and test datasets")
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            
            logging.info("Splitting features and target")
            X_train = train.drop('Cover_Type',axis = 1)
            y_train = train['Cover_Type']

            X_test = test.drop('Cover_Type',axis = 1)
            y_test = test['Cover_Type']

            logging.info("Obtaining preprocessor")
            preprocessor = self.get_data_transformer()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)
            logging.info("Preprocessing is completed")

            save_object(
                obj = preprocessor,
                file_path = self.data_transformation_config.preprocessor_obj_file_path
                )
            return(
                X_train_scaled,y_train,
                X_test_scaled,y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        