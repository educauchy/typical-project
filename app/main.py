import os
import warnings
import sys

import numpy as np

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn



random_state = 0
mlflow.log_param('random_state', random_state)

X, y = make_classification(n_samples=1_000,
                           n_features=50,
                           n_informative=10,
                           n_redundant=10,
                           n_classes=2,
                           random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)



pipeline_steps = []


# IMPUTER
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
pipeline_steps.append( ('imputer', imputer) )


# MODEL
param_grid = {
    'model__max_depth': [5, 6, 7],
    'model__min_samples_split': [5, 6, 7],
    'model__min_samples_leaf': [5, 6, 7],
}
clf = DecisionTreeClassifier()
pipeline_steps.append( ('model', clf) )
pipeline = Pipeline(steps=pipeline_steps)
pipe = GridSearchCV(pipeline, param_grid=param_grid)
pipe.fit(X_train, y_train)

mlflow.end_run()
with mlflow.start_run() as run:
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    mlflow.log_params(pipe.best_params_)
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('f1', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))
    mlflow.sklearn.log_model(clf, 'model')
