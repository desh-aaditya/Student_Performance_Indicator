import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            # 🚫 DO NOT GridSearch CatBoost
            if model_name == "CatBoostRegressor":
                model.fit(X_train, y_train)
                trained_models[model_name] = model
            else:
                model_params = param.get(model_name, {})

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_
                trained_models[model_name] = best_model

            # Evaluate
            y_test_pred = trained_models[model_name].predict(X_test)
            report[model_name] = r2_score(y_test, y_test_pred)

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
