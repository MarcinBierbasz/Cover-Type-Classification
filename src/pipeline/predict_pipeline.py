import os
import sys
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features, model_name):
        try:
            logging.info("Transforming data")  
            preprocessor_path = "artifacts\preprocessor.pkl"
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)

            logging.info("Creating prediction")
            if model_name == "Heuristic":
                model_path = "artifacts\heuristic.pkl"
                model = load_object(file_path=model_path)
                prediction = model.predict(data_scaled)
            elif model_name == "Random Forest":
                model_path = "artifacts\\random_forest.pkl"
                model = load_object(file_path=model_path)
                prediction = model.predict(data_scaled)
            elif model_name == "Naive Bayes":
                model_path = "artifacts\\naive_bayes.pkl"
                model = load_object(file_path=model_path)
                prediction = model.predict(data_scaled)
            else:
                model_path = "artifacts\\neural_network.pkl"
                model = load_object(file_path=model_path)
                data_scaled = np.array(data_scaled)
                prediction = np.argmax(model.predict(data_scaled))+1
            return prediction
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            Elevation:float,
            Aspect: float,
            Slope: float,
            Horizontal_Distance_To_Hydrology: float,
            Vertical_Distance_To_Hydrology: float,
            Horizontal_Distance_To_Roadways: float,
            Hillshade_9am: float,
            Hillshade_Noon: float,
            Hillshade_3pm: float,
            Horizontal_Distance_To_Fire_Points: float,
            Wilderness_Area: int,
            Soil_Type: int
            ):
        self.Elevation = Elevation
        self.Aspect = Aspect
        self.Slope = Slope
        self.Horizontal_Distance_To_Hydrology = Horizontal_Distance_To_Hydrology
        self.Vertical_Distance_To_Hydrology = Vertical_Distance_To_Hydrology
        self.Horizontal_Distance_To_Roadways = Horizontal_Distance_To_Roadways
        self.Hillshade_9am = Hillshade_9am
        self.Hillshade_Noon = Hillshade_Noon
        self.Hillshade_3pm = Hillshade_3pm
        self.Horizontal_Distance_To_Fire_Points = Horizontal_Distance_To_Fire_Points
        self.Wilderness_Area = Wilderness_Area
        self.Soil_Type = Soil_Type

    def get_data_as_dataframe(self):
        try:
            data_dictionary = {
                "Elevation": [self.Elevation],
                "Aspect":[self.Aspect],
                "Slope": [self.Slope],
                'Horizontal_Distance_To_Hydrology': [self.Horizontal_Distance_To_Hydrology],
                'Vertical_Distance_To_Hydrology': [self.Vertical_Distance_To_Hydrology],
                'Horizontal_Distance_To_Roadways': [self.Horizontal_Distance_To_Roadways],
                'Hillshade_9am': [self.Hillshade_9am],
                'Hillshade_Noon': [self.Hillshade_Noon],
                'Hillshade_3pm': [self.Hillshade_3pm],
                'Horizontal_Distance_To_Fire_Points': [self.Horizontal_Distance_To_Fire_Points], 
            }
            for i in range(4):
                if self.Wilderness_Area == i+1:
                    data_dictionary['Wilderness_Area' + str(i)] = 1
                else:
                    data_dictionary['Wilderness_Area' + str(i)] = 0
            for i in range(40):
                if self.Soil_Type == i+1:
                    data_dictionary['Soil_Type' + str(i)] = 1
                else:
                    data_dictionary['Soil_Type' + str(i)] = 0
            df = pd.DataFrame(data_dictionary)
            return df
        except Exception as e:
            raise CustomException(e,sys)