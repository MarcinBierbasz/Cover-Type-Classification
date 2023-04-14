import os
import sys
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split


from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            column_names = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
               'Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
               'Horizontal_Distance_To_Fire_Points'] +\
                ['Wilderness_Area' + str(i) for i in range(4)]  +\
                ['Soil_Type' + str(i) for i in range(40)] + ['Cover_Type']
            
            df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                             compression='gzip',names = column_names)
            logging.info('Loaded dataset to dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            logging.info("Saving raw data")
            df.to_csv(self.ingestion_config.raw_data_path,index = False,header = True)

            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,train_size = 0.6, random_state = 5)

            logging.info("Saving train and test data")
            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header = True)
            
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)
            
            logging.info("Ingestion of the data is completed.")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train,y_train,X_test,y_test,_ = data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(X_train,y_train,X_test,y_test)
