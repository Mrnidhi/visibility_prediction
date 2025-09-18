import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from src.constant import *
from src.cloud_storage.aws_syncer import S3Sync
from src.exception import VisibilityException
from src.logger import logging
from src.utils.main_utils import MainUtils
from src.components.imbalanced_regression_handler import ImbalancedRegressionHandler

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_dir= os.path.join(artifact_folder,'model_trainer')
    trained_model_path= os.path.join(model_trainer_dir, 'trained_model','model.pkl' )
    expected_accuracy=0.45
    model_config_file_path= os.path.join('config','model.yaml')




class VisibilityModel:
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered predict method of srcTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            transformed_feature = self.preprocessing_object.transform(X)

            logging.info("Used the trained model to get predictions")

            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class TwoStageVisibilityModel:
    def __init__(self, preprocessing_object: object, model_dict: dict):
        self.preprocessing_object = preprocessing_object
        self.model_dict = model_dict
        self._handler = ImbalancedRegressionHandler()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered predict method of TwoStageVisibilityModel class")

        try:
            transformed_feature = self.preprocessing_object.transform(X)
            return self._handler.predict_two_stage(self.model_dict, transformed_feature)

        except Exception as e:
            raise VisibilityException(e, sys) from e

    def __repr__(self):
        return "TwoStageVisibilityModel()"

    def __str__(self):
        return "TwoStageVisibilityModel()"


class ModelTrainer:
    def __init__(self):
        

        self.model_trainer_config = ModelTrainerConfig()
        self.s3_sync = S3Sync()


        self.utils = MainUtils()
        self.imbalance_handler = ImbalancedRegressionHandler()
    
    def evaluate_models(self, X, y, models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                model.fit(X_train, y_train)  # Train model

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            raise VisibilityException(e, sys)


    def get_best_model(self,
                    x_train:np.array, 
                    y_train: np.array,
                    x_test:np.array, 
                    y_test: np.array):
        try:
            models = {
                        'Random Forest Regression': RandomForestRegressor(),
                        'DecisionTreeRegressor' : DecisionTreeRegressor()
                        }
             

            model_report: dict = self.evaluate_models(
                 x_train =  x_train, 
                 y_train = y_train, 
                 x_test =  x_test, 
                 y_test = y_test, 
                 models = models
            )

            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object = models[best_model_name]


            return best_model_name, best_model_object, best_model_score


        except Exception as e:
            raise VisibilityException(e,sys)
        
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,
                            ) -> object:
        
        try:

            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]


            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1 )
            
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("best params are:", best_params)

            finetuned_model = best_model_object.set_params(**best_params)
            

            return finetuned_model
        
        except Exception as e:
            raise VisibilityException(e,sys)





    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                        'Linear Regression': LinearRegression(),
                        'Ridge Regression': Ridge(),
                        'Lasso Regression': Lasso(),
                        'Random Forest Regression': RandomForestRegressor(random_state=42),
                        'Gradient Boosting Regression':GradientBoostingRegressor(random_state=42)
                        }

            logging.info(f"Extracting model config file path")

            
            preprocessor = self.utils.load_object(file_path=preprocessor_path)



            logging.info(f"Extracting model config file path")

            imbalance_strategy = os.getenv("IMBALANCE_STRATEGY", "baseline").lower()

            if imbalance_strategy == "two_stage":
                logging.info("Training two-stage model (classifier + regressors) for imbalanced regression")
                two_stage_model = self.imbalance_handler.two_stage_modeling(x_train, y_train)
                y_pred = self.imbalance_handler.predict_two_stage(two_stage_model, x_test)
                best_model = two_stage_model
                best_model_name = "TwoStage"
                best_model_score = r2_score(y_test, y_pred)
            else:
                if imbalance_strategy == "smogn":
                    logging.info("Applying SMOGN oversampling for imbalanced regression")
                    x_train, y_train = self.imbalance_handler.smogn_oversampling(x_train, y_train)

                candidate_models = {
                    'Random Forest Regression': models['Random Forest Regression'],
                    'Gradient Boosting Regression': models['Gradient Boosting Regression']
                }

                sample_weight = None
                if imbalance_strategy == "weighted":
                    logging.info("Using sample weights for training (emphasis on low visibility)")
                    sample_weight = self.imbalance_handler.create_sample_weights(y_train, method='custom')

                report = {}
                for name, model in candidate_models.items():
                    try:
                        if sample_weight is not None:
                            model.fit(x_train, y_train, sample_weight=sample_weight)
                        else:
                            model.fit(x_train, y_train)
                        y_pred_local = model.predict(x_test)
                        report[name] = r2_score(y_test, y_pred_local)
                    except TypeError:
                        model.fit(x_train, y_train)
                        y_pred_local = model.predict(x_test)
                        report[name] = r2_score(y_test, y_pred_local)

                best_model_name = max(report, key=report.get)
                best_model = candidate_models[best_model_name]
                y_pred = best_model.predict(x_test)
                best_model_score = report[best_model_name]

            try:
                self.imbalance_handler.stratified_evaluation(y_test, y_pred, model_name=best_model_name)
            except Exception:
                pass

            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            
            logging.info(f"Best found model on both training and testing dataset")

 
            if best_model_name == "TwoStage":
                custom_model = TwoStageVisibilityModel(
                    preprocessing_object=preprocessor,
                    model_dict=best_model
                )
            else:
                custom_model = VisibilityModel(
                    preprocessing_object=preprocessor,
                    trained_model_object=best_model
                )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=custom_model,
            )

            self.s3_sync.sync_folder_to_s3(folder=os.path.dirname(self.model_trainer_config.trained_model_path),
                                           aws_buket_name= AWS_S3_BUCKET_NAME)

            

            return best_model_score

        except Exception as e:
            raise VisibilityException(e, sys)
