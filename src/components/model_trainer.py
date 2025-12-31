import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(eval_metric="rmse"),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
            }
            params = {
                "LinearRegression": {},
                "DecisionTree": {
                    "criterion": ["squared_error", "friedman_mse"]
                },
                "RandomForest": {
                    "n_estimators": [50, 100]
                },
                "GradientBoosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                },
                "AdaBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                },
                "KNeighbors": {
                    "n_neighbors": [3, 5, 7]
                },
                "XGBRegressor": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                },
                "CatBoostRegressor": {}  # 🚫 no GridSearch
            }
            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best Model: {best_model_name} | R2 Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
