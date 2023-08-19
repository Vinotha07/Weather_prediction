import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from exception import CustomException
from logger import LogHandler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from data_ingestion import DataIngestion
from utils import ConfigManager,save_object,load_object








@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation: 

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # self.imputer = SimpleImputer(strategy='median')
        

    # def fit(self, data):
    #     self.imputer.fit(data)

    # def transform(self, data):
    #     column_names = data.columns.tolist()
    #     data = self.imputer.transform(data)
    #     data = pd.DataFrame(data, columns=column_names)
    #     return data

    def save_preprocessor(self):
        save_object(self.data_transformation_config.preprocessor_obj_file_path, self)

    # @classmethod
    # def load_preprocessor(cls):
    #     return load_object(cls.data_transformation_config.preprocessor_obj_file_path)
    

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            #read train and test data from the path
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train data head: \n{train_df.head().to_string()}')
            logging.info(f'Test data head: \n{test_df.head().to_string()}')

            # Selecting only the required columns
            selected_columns = ['time', 'temperature_2m']
            target_columns='temperature_2m'
            input_feature_train_df = train_df[selected_columns]
            input_feature_test_df = test_df[selected_columns]

            
            logging.info('Read selected columns for train and test data')
            logging.info(f'Train data head: \n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test data head: \n{input_feature_test_df.head().to_string()}')
            

            # Converting time column to datetime format and setting as index
            time_column_name = 'time'  
            input_feature_train_df.loc[:, time_column_name] = pd.to_datetime(input_feature_train_df[time_column_name], format='%Y-%m-%dT%H:%M')
            input_feature_test_df.loc[:, time_column_name] = pd.to_datetime(input_feature_test_df[time_column_name], format='%Y-%m-%dT%H:%M')
            
            
            input_feature_train_df.set_index(time_column_name, inplace=True)
            input_feature_test_df.set_index(time_column_name, inplace=True)

            logging.info("Set Timestamp as Index successfully")

            logging.info('After timestamp train and test data')
            logging.info(f'Train data head: \n{input_feature_train_df.head().to_string()}')
            logging.info(f'Test data head: \n{input_feature_test_df.head().to_string()}')

            # New Code for simple imputer
            # Perform imputation using the median strategy
            imputer = SimpleImputer(strategy='median')
            input_feature_train_df_transformed = imputer.fit_transform(input_feature_train_df)
            input_feature_test_df_transformed = imputer.transform(input_feature_test_df)
            logging.info("Simple Imputation completed")

            # Convert the transformed arrays back to DataFrames
            input_feature_train_df_transformed = pd.DataFrame(input_feature_train_df_transformed, columns=input_feature_train_df.columns, index=input_feature_train_df.index)
            input_feature_test_df_transformed = pd.DataFrame(input_feature_test_df_transformed, columns=input_feature_test_df.columns, index=input_feature_test_df.index)

            logging.info('Read train and test data after Simple Imputation')
            logging.info(f'Train data head: \n{input_feature_train_df_transformed.head().to_string()}')
            logging.info(f'Test data head: \n{input_feature_test_df_transformed.head().to_string()}')

        


            # self.fit(input_feature_train_df)
            # input_feature_train_df_transformed = self.transform(input_feature_train_df)
            # input_feature_test_df_transformed = self.transform(input_feature_test_df)
            # logging.info("Simple Imputation completed")

            # # Reindex the imputed DataFrames to original timestamps
            # input_feature_train_df_transformed = input_feature_train_df_transformed.reindex(input_feature_train_df.index)
            # input_feature_test_df_transformed = input_feature_test_df_transformed.reindex(input_feature_test_df.index)

            # # Convert index back to datetime format
            # input_feature_train_df_transformed.index = pd.to_datetime(input_feature_train_df_transformed.index)
            # input_feature_test_df_transformed.index = pd.to_datetime(input_feature_test_df_transformed.index)

            # # input_feature_train_df_transformed.index = pd.to_datetime(input_feature_train_df_transformed.index)
            # # input_feature_test_df_transformed.index = pd.to_datetime(input_feature_test_df_transformed.index)

            # logging.info('Read train and test data after Simple Imputation')
            # logging.info(f'Train data head: \n{input_feature_train_df_transformed.head().to_string()}')
            # logging.info(f'Test data head: \n{input_feature_train_df_transformed.head().to_string()}')



            # # Resample the transformed DataFrames
            # input_feature_train_df_resampled = input_feature_train_df_transformed.resample('D').mean()
            # input_feature_test_df_resampled = input_feature_test_df_transformed.resample('D').mean()
            # logging.info("Converted Hourly Datas to Daily Datas")

            
            # Resample the transformed DataFrames
            input_feature_train_df_resampled = input_feature_train_df_transformed['temperature_2m'].resample('D').mean()
            input_feature_test_df_resampled = input_feature_test_df_transformed['temperature_2m'].resample('D').mean()
            logging.info("Converted Hourly Datas to Daily Datas")

            
            logging.info('Read Resampled train and test data completed')
            logging.info(f'Train data head: \n{input_feature_train_df_resampled.head().to_string()}')
            logging.info(f'Test data head: \n{input_feature_test_df_resampled.head().to_string()}')

            # Convert the resampled DataFrames to NumPy arrays
            input_feature_train_arr = input_feature_train_df_resampled.values
            input_feature_test_arr = input_feature_test_df_resampled.values



            self.save_preprocessor()

            logging.info("Preprocessing Done!! Saved as Pickled File")
            logging.info(input_feature_train_arr)

            return (
                input_feature_train_arr,
                input_feature_test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error occured at the DataTransformation stage")
            raise CustomException (e,sys)
        

# if __name__ == "__main__":
#     # To handle logging
#     log_handler = LogHandler()
#     log_handler.setup_logging()

#     # To read the datas from config.ini
#     config_manager = ConfigManager()

#     obj=DataIngestion()
#     train_data_path,test_data_path=obj.initiate_data_ingestion(config_manager)
     
#     data_transformation=DataTransformation()  
#     train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path) 


      




        