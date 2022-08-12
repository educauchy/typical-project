import os
import warnings
import sys
import logging
from urllib.parse import urlparse
from urllib.error import HTTPError
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, plot_roc_curve, roc_curve,\
                            accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from hyperopt import hp, fmin, tpe, Trials
from typing import Dict, Any, List


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# objective function for hyperopt
def objective(params):
    full_pipeline.set_params(**params)
    cv = KFold(n_splits=config['model']['cv']['folds'], shuffle=True)
    score = cross_val_score(full_pipeline, train_x, train_y, cv=cv, scoring=config['model']['main_metric'], n_jobs=1)
    return score.mean()

def eval_metrics(actual, pred_labels, pred_probs, task_type='classification') -> Dict[str, Any]:
    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(actual, pred_labels)
        metrics['roc_auc'] = roc_auc_score(actual, pred_probs)
        metrics['f1_score'] = f1_score(actual, pred_labels)
        metrics['precision'] = precision_score(actual, pred_labels)
        metrics['recall'] = recall_score(actual, pred_labels)
    elif task_type == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(actual, pred_labels))
        metrics['mae'] = mean_absolute_error(actual, pred_labels)
        metrics['r2'] = r2_score(actual, pred_labels)
    return metrics


# os.system('mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models/mlruns --host 0.0.0.0 --port 5558')

# load config
try:
    with open('/typical_project/src/models/sklearn/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    logging.error(exc)
    sys.exit(1)
except Exception as e:
    logging.error(e)
    logging.error('Error reading the config file')
    sys.exit(1)


# mlflow.set_tracking_uri(uri=config['tracking']['TRACKING_URI'])
try:
    print('Create experiment')
    experiment_id = mlflow.create_experiment(name=config['tracking']['EXPERIMENT_NAME'],
                                             artifact_location=config['tracking']['ARTIFACT_LOCATION']) # /mlruns
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
except Exception:
    print('Set experiment')
    experiment = mlflow.set_experiment(experiment_name=config['tracking']['EXPERIMENT_NAME'])


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(config['tracking']['random_state'])

    train_url = config['data']['train']
    try:
        data = pd.read_csv(train_url, sep=";")
    except HTTPError as e:
        logger.exception("Unable to download training CSV. Error: %s", e)
        raise Exception("Unable to download training CSV. Error: %s", e)

    train, test = train_test_split(data, test_size=0.2)
    train_x = data.drop(["quality"], axis=1)
    test_x = data.drop(["quality"], axis=1)
    train_y = data[["quality"]]
    test_y = data[["quality"]]

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_param('random_state', config['tracking']['random_state'])
        pipeline_steps = []

        # NUMERIC STEPS
        numeric_features = train_x.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer_steps = []

        imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        numeric_transformer_steps.append(('imputer', imputer))

        scaler = StandardScaler()
        numeric_transformer_steps.append(('scaler', scaler))

        numeric_transformer = Pipeline(numeric_transformer_steps)

        # CATEGORICAL STEPS
        categorical_features = train_x.select_dtypes(include=['object', 'bool']).columns
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

        if config['model']['method'] == 'LogisticRegression':
            curr_model = LogisticRegression()
        elif config['model']['method'] == 'LinearSVC':
            curr_model = LinearSVC()
        elif config['model']['method'] == 'AdaBoostClassifier':
            curr_model = AdaBoostClassifier()

        pipeline_steps.append(('model', curr_model))
        full_pipeline = Pipeline(steps=pipeline_steps)

        mlflow.log_params({
            'num_feature_imputer': 'median',
            'num_feature_scaler': 'standard',
            'cat_feature_imputer': 'median'
        })

        space = {}
        for param, values in config['model']['params_hyperopt'].items():
            if len(values) == 3 and str(values[0]) != values[0]:
                space[f'model__{param}'] = hp.quniform(label=param,
                                                       low=values[0],
                                                       high=values[1],
                                                       q=values[2])
                mlflow.log_param(f'{param}_min', values[0])
                mlflow.log_param(f'{param}_max', values[1])
            else:
                values = tuple(values)
                space[f'model__{param}'] = hp.choice(label=param, options=values)
                mlflow.log_param(f'{param}_values', values)

        trials = Trials()
        best_params = fmin(objective,
                           space,
                           algo=tpe.suggest,
                           max_evals=config['model']['cv']['evals'],
                           trials=trials)
        mlflow.log_params({
            'cv_folds': config['model']['cv']['folds'],
            'cv_evals': config['model']['cv']['evals']
        })

        best_params_ = {}
        for key, value in best_params.items():
            if round(value) == value:
                value = int(value)

            best_params_[f'model__{key}'] = value

        best_params_to_log = {}
        for key, value in best_params.items():
            best_params_to_log[f'{key}_optimal'] = value
        mlflow.log_params(best_params_to_log)

        # Fit the model with the optimal hyperparamters
        full_pipeline.set_params(**best_params_)
        full_pipeline.fit(train_x, train_y)

        # Predict
        predicted_labels = full_pipeline.predict(test_x)
        predicted_probs = full_pipeline.predict_proba(test_x)[:, 1]

        quality_metrics = eval_metrics(test_y, predicted_labels, predicted_probs)
        mlflow.log_metrics(quality_metrics)

        # ROC AUC curve
        roc_curve_path = config['tracking']['FIGURES_PATH'] + f'/roc_auc_{run.info.run_id}.png'
        fpr, tpr, thresholds = roc_curve(y_true=test_y, y_score=predicted_probs, pos_label=1)
        plt.plot(fpr, tpr)
        plt.title('ROC-AUC curve (AUC={roc_auc:.2f})'.format(roc_auc=quality_metrics['roc_auc']))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(roc_curve_path)

        mlflow.log_artifact(roc_curve_path)
        mlflow.log_artifact('/typical_project/src/models/sklearn/config.yaml')
        mlflow.log_artifact('/typical_project/src/models/sklearn/MLProject')
        mlflow.log_artifact('/typical_project/src/models/sklearn/conda_environment.yaml')

        # we can also use the following
        # mlflow.log_text
        # mlflow.log_dict
        # mlflow.log_figure
        # mlflow.log_image

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            lm = mlflow.sklearn.log_model(sk_model=full_pipeline,
                                          artifact_path=config['tracking']['ARTIFACT_PATH'],
                                          registered_model_name=config['tracking']['MODEL_NAME'],
                                          signature=infer_signature(train_x, full_pipeline.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict(),
                                          pip_requirements='/typical_project/src/models/sklearn/requirements.txt'
                                          )
        else:
            lm = mlflow.sklearn.log_model(sk_model=full_pipeline,
                                          artifact_path=config['tracking']['ARTIFACT_PATH'],
                                          signature=infer_signature(train_x, full_pipeline.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict(),
                                          pip_requirements='/typical_project/src/models/sklearn/requirements.txt'
                                          )

        # model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run.info.run_id,
        #                                                     artifact_path=lm.artifact_path)
        # result = mlflow.register_model(
        #     model_uri=model_uri,
        #     name=config['tracking']['MODEL_NAME']
        # )
        #
        # client = MlflowClient()
        # curr_version = client.get_latest_versions(
        #     name=config['tracking']['MODEL_NAME'],
        #     stages=['None']
        # )
        #
        # client.update_model_version(
        #     name=config['tracking']['MODEL_NAME'],
        #     version=curr_version[0].version,
        #     description="This model version is a scikit-learn random forest containing 100 decision trees"
        # )
        #
        # client.transition_model_version_stage(
        #     name=config['tracking']['MODEL_NAME'],
        #     version=curr_version[0].version,
        #     stage="Staging"
        # )
