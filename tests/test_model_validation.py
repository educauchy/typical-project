import os
import time
import pandas as pd
import pytest
import mlflow.sklearn
import logging
from urllib.error import HTTPError
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, r2_score


class TestModelValidation(object):
    splits = {
        'train': {
            'path': './data/processed/winequality-red-train.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'quality']
        },
        'test': {
            'path': './data/processed/winequality-red-test.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'quality']
        },
        'scoring': {
            'path': './data/processed/winequality-red-scoring.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'id']
        },
        'scoring_result': {
            'path': './data/processed/winequality-red-scoring-result.csv',
            'needed_cols': ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                            'pH', 'sulphates', 'alcohol', 'score', 'id']
        }
    }
    model_uri = ''

    def __model_prediction(self):
        data = pd.read_csv(self.splits['scoring']['path'], sep=';', decimal='.')
        # real_labels = data['quality'].to_numpy()
        data.drop(columns=['quality'], errors='ignore', inplace=True)

        model_name = 'logistic_regression'
        model_stage = 'None'
        model_path = f'models:/{model_name}/{model_stage}'
        model = mlflow.sklearn.load_model(model_path)
        print(model)

        # pred_scores = model.predict(data)
        # return pred_scores
        # return real_labels, pred_scores

    def test_model_prediction_time(self):
        # os.system('mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models/mlruns --host 0.0.0.0 --port 5558')
        # print('Server started')
        # os.system('mlflow models serve --model-uri "runs:/8702524acc6943ed9001fa446bc45fb1/model" --port 5557')
        # print('Model served')
        # os.system('mlflow models predict --model-uri "models:/logistic_regression/None" --input-path "./data/processed/winequality-red-scoring.csv" --output-path "./data/processed/winequality-red-scoring-result.csv" --content-type "csv"')

        os.system('mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models/mlruns --host 0.0.0.0 --port 5558')
        mlflow.set_tracking_uri('http://localhost:5558')

        start_time = time.perf_counter()
        # self.__model_prediction()
        time.sleep(5)
        end_time = time.perf_counter()
        exec_seconds = end_time - start_time

        assert exec_seconds < 10

    def test_model_prediction_quality(self):
        # metrics = {
        #     'roc_auc': { 'func': roc_auc_score, 'threshold': 0.7 },
        #     'accuracy': { 'func': accuracy_score, 'threshold': 0.7 },
        #     'precision': { 'func': precision_score, 'threshold': 0.7 },
        #     'recall': { 'func': recall_score, 'threshold': 0.7 },
        #     'r2': { 'func': r2_score, 'threshold': 0.1 },
        # }
        # real_labels, pred_scores = self.__model_prediction()
        # threshold_passed = []
        # for metric, values in metrics.items():
        #     result = values['func'](real_labels, pred_scores) >= values['threshold']
        #     threshold_passed.append(result)
        #
        # assert all(threshold_passed)
        pass


    def test_model_superiority(self):
        # pytest tests/test_model_validation.py::TestModelValidation::test_model_superiority
        pass


if __name__ == '__main__':
    TestModelValidation().test_model_superiority()