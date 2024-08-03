from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
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
                "Decision tree":DecisionTreeRegressor(),
                "Random forest regressor": RandomForestRegressor(),
                "XGBooster": XGBRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor()


            }

            mode_report: dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

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