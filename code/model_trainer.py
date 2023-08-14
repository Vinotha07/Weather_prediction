import os
import sys
import pandas as pd
import numpy as np

from data_transformation import DataTransformation
from logger import LogHandler
from exception import CustomException
from utils import save_object

from dataclasses import dataclass
from statsmodels.tsa.arima.model import ARIMA

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            #create dictionary for the models
            models={
                'Linear Regression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elastic Net':ElasticNet()
            }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('='*35)
            print("\n")

            logging.info(f'Model Report: {model_report}')
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            print(f'Best Model Found, Model Name:{best_model_name},R2 score {best_model_score}')
            print('='*35)
            print("\n")
            logging.info(f'Best Model Found, Model Name:{best_model_name},R2 score {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info('Best Model Trainer Saved as pickle file')

            
        except Exception as e:
            logging.info('Error occurred at Model Trainer stage')
            raise CustomException(e,sys)


         


