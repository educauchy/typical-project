import time
import pandas as pd
import pytest
import mlflow.sklearn
import logging
from urllib.error import HTTPError
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, r2_score
# проверить работу API
# проверить, что модель делает предсказания


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
        data = pd.read_csv(self.splits['test']['path'], sep=';', decimal='.')

        mlflow.set_tracking_uri('http://localhost:5557')

        real_labels = data['quality'].to_numpy()
        data.drop(columns=['quality'], inplace=True)

        model_name = 'test_model_new'
        model_stage = 'None'
        model_path = f'models:/{model_name}/{model_stage}'
        model = mlflow.sklearn.load_model(model_path)

        pred_scores = model.predict(data)
        return real_labels, pred_scores

    def test_model_prediction_time(self):
        # pytest tests/test_model_validation.py::TestModelValidation::test_model_prediction_time
        start_time = time.perf_counter()
        self.__model_prediction()
        end_time = time.perf_counter()
        exec_seconds = end_time - start_time
        assert exec_seconds < 10

    def test_model_prediction_quality(self):
        # pytest tests/test_model_validation.py::TestModelValidation::test_model_prediction_quality
        metrics = {
            # 'roc_auc': { 'func': roc_auc_score, 'threshold': 0.7 },
            # 'accuracy': { 'func': accuracy_score, 'threshold': 0.7 },
            # 'precision': { 'func': precision_score, 'threshold': 0.7 },
            # 'recall': { 'func': recall_score, 'threshold': 0.7 },
            'r2': { 'func': r2_score, 'threshold': 0.1 },
        }
        real_labels, pred_scores = self.__model_prediction()
        threshold_passed = []
        for metric, values in metrics.items():
            result = values['func'](real_labels, pred_scores) >= values['threshold']
            threshold_passed.append(result)

        assert all(threshold_passed)


    def test_model_superiority(self):
        # pytest tests/test_model_validation.py::TestModelValidation::test_model_superiority
        pass


if __name__ == '__main__':
    TestModelValidation().test_model_superiority()