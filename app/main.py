import os
import warnings
import sys

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn



X, y = make_classification(n_samples=10_000,
                           n_features=50,
                           n_informative=10,
                           n_redundant=10,
                           n_classes=2,
                           random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

params = {
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 5,
    'random_state': 0
}
clf = DecisionTreeClassifier(**params)
clf.fit(X_train, y_train)

with mlflow.start_run() as run:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    mlflow.log_params(params)
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('f1', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))
    mlflow.sklearn.log_model(clf, 'model')
