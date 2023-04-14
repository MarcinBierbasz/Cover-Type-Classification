import os
import sys
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
from src.utils import plot_heuristic_confusion_matrix
from src.utils import plot_baseline_confusion_matrix
from src.utils import plot_neural_network_confusion_matrix

#plt.savefig('foo.png')

@dataclass
class ModelEvaluatorConfig:
    heuristic_confusion_matrix_file_path = os.path.join('evaluations','heuristic_confusion_matrix.png')
    baseline_confusion_matrix_file_path = os.path.join('evaluations','baseline_models_confusion_matrix.png')
    neural_network_confusion_matrix_file_path = os.path.join('evaluations','neural_network_models_confusion_matrix.png')
    accuracy_score_file_path = os.path.join('evaluations','accuracy_score.txt')

    
class ModelEvaluator:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluatorConfig()
    def initiate_models_evaluation(self,X_test,y_test,*models):
        '''
        Function saving confusion matrix and accuracy_score of all models, performed on unseen data.
        '''
        try:
            trained_heuristic,trained_random_forest,trained_naive_bayes,trained_neural_network = models
            os.makedirs(os.path.dirname(self.model_evaluation_config.baseline_confusion_matrix_file_path),exist_ok=True)

            logging.info("Saving confusion matrixes")
            plot_heuristic_confusion_matrix(X_test,y_test,self.model_evaluation_config.heuristic_confusion_matrix_file_path,
                                    trained_heuristic
                                    )
            plot_baseline_confusion_matrix(X_test,y_test,self.model_evaluation_config.baseline_confusion_matrix_file_path,
                                    trained_random_forest,trained_naive_bayes
                                    )

            plot_neural_network_confusion_matrix(X_test,y_test,self.model_evaluation_config.neural_network_confusion_matrix_file_path,
                                            trained_neural_network
                                            )
            
            logging.info("Saving accuracy scores")
            
            y_pred_heuristic = trained_heuristic.predict(X_test)
            accuracy_score_heuristic = accuracy_score(y_test,y_pred_heuristic)

            y_pred_random_forest = trained_random_forest.predict(X_test)
            accuracy_score_random_forest = accuracy_score(y_test,y_pred_random_forest)
            y_pred_naive_bayes = trained_naive_bayes.predict(X_test)
            accuracy_score_naive_bayes = accuracy_score(y_test,y_pred_naive_bayes)

            y_pred_neural_network = np.argmax(trained_neural_network.predict(np.array(X_test)),axis=1)+1
            accuracy_score_neural_network = accuracy_score(y_test,y_pred_neural_network)
            with open(self.model_evaluation_config.accuracy_score_file_path,'w+') as file_obj:
                file_obj.write(f"Heuristic score: {accuracy_score_heuristic}\n")
                file_obj.write(f"Random Forest score: {accuracy_score_random_forest}\n")
                file_obj.write(f"Naive Bayes score: {accuracy_score_naive_bayes}\n")
                file_obj.write(f"Neural Network score: {accuracy_score_neural_network}\n")

        except Exception as e:
            raise CustomException(e,sys)
        
