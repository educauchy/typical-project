import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn


mlflow.set_tracking_uri('http://localhost:5081')

curr_path = os.path.dirname(os.path.abspath(__file__))
artifact_location = os.path.join('file://', curr_path, 'experiments')


experiment_name = 'test_end'
try:
    EXPERIMENT_ID = mlflow.create_experiment(experiment_name,
                        artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
except ValueError:
    EXPERIMENT_ID = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)


random_state = 0


X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=random_state)

pipeline_steps = []


# NUMERIC STEPS
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer_steps = []

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
numeric_transformer_steps.append(('imputer', imputer))

scaler = StandardScaler()
numeric_transformer_steps.append(('scaler', scaler))

numeric_transformer = Pipeline(numeric_transformer_steps)


# CATEGORICAL STEPS
categorical_features = X.select_dtypes(include=['object', 'bool']).columns
categorical_transformer_steps = []

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
categorical_transformer_steps.append(('imputer', imputer))

categorical_transformer = Pipeline(categorical_transformer_steps)


# JOIN NUMERIC AND CATEGORICAL
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
pipeline_steps.append(('preprocess', preprocessor))


# MODEL
param_grid = {
    'model__max_depth': [5, 6, 7],
    'model__min_samples_split': [5, 6, 7],
    'model__min_samples_leaf': [5, 6, 7],
}
clf = DecisionTreeClassifier()
pipeline_steps.append(('model', clf))
pipeline = Pipeline(steps=pipeline_steps)
pipe = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', cv=4)
pipe.fit(X_train, y_train)


# LOG EXPERIMENT
with mlflow.start_run(experiment_id=EXPERIMENT_ID) as run:
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    # mlflow.sklearn.autolog()
    mlflow.log_param('random_state', random_state)
    mlflow.log_params(pipe.best_params_)
    mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('f1', f1_score(y_test, y_pred))
    mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))
    mlflow.sklearn.log_model(clf, 'model')
