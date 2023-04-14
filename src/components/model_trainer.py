import os
import sys
import pandas as pd
from dataclasses import dataclass

sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import calculate_class_weights
from src.utils import tuning_sklearn_models
from src.utils import train_sklearn_models
from src.utils import tune_neural_network_hyperparameters
from src.utils import train_neural_network



from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB



@dataclass
class ModelTrainerConfig:
    trained_random_forest_file_path = os.path.join('artifacts','random_forest.pkl')
    trained_naive_bayes_file_path = os.path.join('artifacts','naive_bayes.pkl')
    trained_neural_network_file_path = os.path.join('artifacts','neural_network.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,X_train,y_train,X_test,y_test):
        
        try:
            logging.info("Attempting models training.")
            models = {
                "RandomForest": RandomForestClassifier(class_weight = calculate_class_weights(y_train)),
                "MultinomialNB": MultinomialNB()
            }
            baseline_param_grid = {
                    "RandomForest": {
                    "n_estimators": [1],
                    "max_depth": [None]
                    },
                    "MultinomialNB":{
                    "alpha":[0.001,0.01,0.1,1.,2.]
                    }

            }
            
            nn_param_grid = {
                'epochs' :[1],
                'optimizers' :  ["Adam"],
                'batch_sizes' : [2048]
            }

            logging.info("Performing grid search on baseline models")
            best_baseline_params = tuning_sklearn_models(X_train,y_train,models,baseline_param_grid)
            logging.info("Training models with best hyperparameters")
            trained_random_forest,trained_naive_bayes = train_sklearn_models(X_train,y_train,models,best_baseline_params)


            logging.info("Tuning hyperparameters in neural network")
            best_nn_params = tune_neural_network_hyperparameters(X_train,y_train,nn_param_grid)
            logging.info("Training neural network with best hyperparameters")
            trained_neural_network = train_neural_network(X_train,y_train,best_nn_params)


            logging.info("Saving trained models")
            save_object(file_path = self.model_trainer_config.trained_random_forest_file_path, obj = trained_random_forest)
            save_object(file_path = self.model_trainer_config.trained_naive_bayes_file_path, obj = trained_naive_bayes)
            save_object(file_path = self.model_trainer_config.trained_neural_network_file_path, obj = trained_neural_network)

        except Exception as e:
            raise CustomException(e,sys)
