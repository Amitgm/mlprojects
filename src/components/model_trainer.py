from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import sys

from src.utils import save_object,evaluate_model

import os

@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_trainer(self,train_array,test_array):

        try:
            logging.info("split train and test array")
            x_train,y_train,x_test,y_test = (

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]

            )

            models = {

                "Linear Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBooster": XGBRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()


            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },

                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Linear Regression":{},
                "Lasso":{},

                "XGBooster":{

                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
         
                "AdaBoostRegressor":{

                    'learning_rate':[0.1,0.5,0.7,0.01],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighborsRegressor":{},

                
            }

            mode_report: dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(mode_report.values()))

            best_model_name = list(mode_report.keys())[

                list(mode_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
        
                raise CustomException("no best model found")
            

            logging.info(f"best model found on both training and testing {best_model}")


            save_object(

                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )


            predicted = best_model.predict(x_test)

            r2score = r2_score(y_test,predicted)


            return r2score

            # preprocessor_path 



        except Exception as e:

            raise CustomException(e,sys)