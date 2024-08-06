import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
import os
from dataclasses import dataclass
from src.utils import save_object



@dataclass
class DataTransformationConfig:

    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")

    # train_data_path = os.path.join("artifacts","train.csv")

class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self,input_features):

        try:

            # dataframe = pd.read_csv(self.data_transformation_config.train_data_path)

            num_features = input_features.select_dtypes(exclude="object").columns
            cat_features = input_features.select_dtypes(include="object").columns

            print("the num fetures",num_features)
            print("the cat features",cat_features)


            num_pipelines = Pipeline(

                steps = [

                ("imputer",SimpleImputer(strategy="median")),
                ("scalar",StandardScaler(with_mean=False)) ]
            )

            cat_pipelines = Pipeline(

                steps = [

                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scalar",StandardScaler(with_mean=False))  ]
            )


            logging.info("numerical columns standard scaling completed")

            logging.info("categorical columns encoding completed")


            preprocessor = ColumnTransformer(

                    [
                        ("num pipeline",num_pipelines,num_features),
                        ("categorical pipeline",cat_pipelines,cat_features)

                    ]


                    )

            return preprocessor




        except Exception as e:

            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining pre-proceesing object")


            target_column = "math_score"


            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)

            target_features_train_df = train_df[target_column]

            preprocessing_obj = self.get_data_transformer_object(input_feature_train_df)


            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)

            target_features_test_df = test_df[target_column]


            logging.info("Applying preproceesing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[input_feature_train_arr,np.array(target_features_train_df)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_features_test_df)]


            logging.info("Saved preprocessing objects")


            save_object(

                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )


            return (
                train_arr,
                test_arr,
                # self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:

            raise CustomException(e,sys)