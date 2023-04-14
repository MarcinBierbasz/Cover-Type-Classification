import os
import sys
import numpy as np
sys.path.append('..\CovertypeClassification')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import dill
from sklearn.utils import class_weight

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier



def save_object(file_path,obj):
    '''
    Function saving object as pickle file.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def calculate_class_weights(y_train):
    '''
    Function calculating class weights, helping to deal with imbalanced dataset.
    '''
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes = np.unique(y_train),y=y_train
                                                      )
    class_weights = {index+1:value for index,value in enumerate(class_weights)}
    return class_weights


def tuning_sklearn_models(X_train,y_train,models:dict,param_grid:dict):
    '''
    Function performing Grid Search with crossvalidation.
    '''
    best_params = {}
    best_score = {}
    try:
        for model_name,model in models.items():
            logging.info(f'Grid searching model {model_name}')
            clf = GridSearchCV(estimator = model, param_grid = param_grid[model_name],scoring = 'accuracy', cv=5,n_jobs = -1)
            clf.fit(X_train,y_train)
            best_score[model_name] = clf.best_score_
            best_params[model_name] = clf.best_params_
            logging.info(f'{model_name} best cv score: {best_score[model_name]}; best params: {best_params[model_name]}')
        return best_params
    except Exception as e:
            raise CustomException(e,sys)
    

def train_sklearn_models(X_train,y_train,models:dict,best_params:dict):
    '''
    Function train models with best perameters.
    '''
    try:
        model_random_forest = models["RandomForest"]
        model_random_forest.set_params(**best_params['RandomForest'])

        logging.info('Training random forest model')
        model_random_forest.fit(X_train,y_train)

        model_multinomial_nb = models["MultinomialNB"]
        model_multinomial_nb.set_params(**best_params["MultinomialNB"])
        logging.info('Training Naive Bayes model')
        model_multinomial_nb.fit(X_train,y_train)

        return model_random_forest,model_multinomial_nb
    except Exception as e:
        raise CustomException(e,sys)

def make_neural_network_model(optimizer,number_of_features):
    METRICS = [
      keras.metrics.BinaryAccuracy(name='CategoricalAccuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]  
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu',
                            input_shape=(number_of_features,)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(7, activation='softmax'),
  ])
     
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=METRICS)
    return model


def tune_neural_network_hyperparameters(X_train,y_train,param_grid):
    '''
    Function tuning hyperparameters, by looping over grid arguments and comparing models categorical accuracy score.
    '''
    try:
        y_train = np.array(pd.get_dummies(y_train))
        X_split,X_val,y_split,y_val = train_test_split(X_train,y_train,train_size = 0.8)

        best_score = 0
        for epochs_number in param_grid['epochs']:
            for batch_size in param_grid['batch_sizes']:
                for optimizer in param_grid['optimizers']:
                    model = make_neural_network_model(optimizer=optimizer,number_of_features = X_train.shape[-1])
                    model.fit(X_split,y_split,epochs = epochs_number,batch_size = batch_size)
                    results = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=0)
                    # Measuring score with categorical accuracy.
                    if results[1] > best_score:
                        best_score = results[1]
                        best_params = [epochs_number,optimizer,batch_size]
        return best_params
    except Exception as e:
        raise CustomException(e,sys)

def train_neural_network(X_train,y_train,best_params):
    
    try:
        epochs,optimizer,batch_size = best_params
        y_train = np.array(pd.get_dummies(y_train))
        model_nn = make_neural_network_model(optimizer=optimizer,number_of_features = X_train.shape[-1])
        model_nn.fit(X_train,y_train,epochs = epochs,batch_size = batch_size)
        return model_nn
    except Exception as e:
        raise CustomException(e,sys)
