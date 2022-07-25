import warnings
import sys
import logging
import urllib
from urllib.parse import urlparse
from urllib.error import HTTPError
import yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from hyperopt import hp, fmin, tpe, Trials, space_eval


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# objective function for hyperopt
def objective(params):
    full_pipeline.set_params(**params)
    cv = KFold(n_splits=config['model']['cv']['folds'], shuffle=True)
    score = cross_val_score(full_pipeline, train_x, train_y, cv=cv, scoring='neg_log_loss', n_jobs=1)
    return score.mean()


def eval_metrics(actual, pred):
    rmse_ = np.sqrt(mean_squared_error(actual, pred))
    mae_ = mean_absolute_error(actual, pred)
    r2_ = r2_score(actual, pred)
    return rmse_, mae_, r2_


# load config
try:
    with open ('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    logging.error(exec)
    sys.exit(1)
except Exception as e:
    logging.error('Error reading the config file')
    sys.exit(1)


mlflow.set_tracking_uri(config['tracking']['TRACKING_URI'])

# get experiment
try:
    experiment = mlflow.set_experiment(experiment_name=config['tracking']['EXPERIMENT_NAME'])
except Exception:
    artifact_location = 'file:///Users/educauchy/Documents/Dev/DS/GBC/mlops_project_mlflow/reports/mlruns'
    experiment = mlflow.create_experiment(name=config['tracking']['EXPERIMENT_NAME'],
                                          artifact_location=artifact_location)



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(config['tracking']['random_state'])

    csv_url = '../../data/processed/winequality-red-train.csv'
    try:
        data = pd.read_csv(csv_url, sep=";")
    except HTTPError as e:
        logger.exception("Unable to download training CSV. Error: %s", e)
        raise Exception("Unable to download training CSV. Error: %s", e)

    train, test = train_test_split(data, test_size=0.2)
    train_x = data.drop(["quality"], axis=1)
    test_x = data.drop(["quality"], axis=1)
    train_y = data[["quality"]]
    test_y = data[["quality"]]

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id=run.info.run_id):
        # model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        # model.fit(train_x, train_y)
        pipeline_steps = []

        # NUMERIC STEPS
        numeric_features = train_x.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer_steps = []

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
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
        full_pipeline = Pipeline(steps=pipeline_steps)


        space = {}
        for param, values in config['model']['param_hyperopt'].items():
            if len(values) > 3:
                space[param] = values
            else:
                space[param] = np.arange(*values)

        trials = Trials()
        best = fmin(objective,
                    space,
                    algo=tpe.suggest,
                    max_evals=50,
                    trials=trials)

        # Get the values of the optimal parameters
        best_params = space_eval(space, best)

        # Fit the model with the optimal hyperparamters
        full_pipeline.set_params(**best_params)
        full_pipeline.fit(train_x, train_y)


        # Predict
        predicted_qualities = full_pipeline.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # mlflow.log_params({
        #     'alpha': alpha,
        #     'l1_ratio': l1_ratio,
        # })

        mlflow.log_metrics({
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
        })

        mlflow.log_artifact('./MLProject')
        mlflow.log_artifact('./conda_environment.yaml')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            lm = mlflow.sklearn.log_model(sk_model=full_pipeline,
                                          artifact_path=config['tracking']['ARTIFACT_PATH'],
                                          registered_model_name=config['tracking']['MODEL_NAME'],
                                          signature=infer_signature(train_x, full_pipeline.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict()
                                          )
        else:
            lm = mlflow.sklearn.log_model(sk_model=full_pipeline,
                                          artifact_path=config['tracking']['ARTIFACT_PATH'],
                                          registered_model_name=config['tracking']['MODEL_NAME'],
                                          signature=infer_signature(train_x, full_pipeline.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict()
                                          )

        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run.info.run_id,
                                                            artifact_path=lm.artifact_path)
        result = mlflow.register_model(
            model_uri=model_uri,
            name=config['tracking']['MODEL_NAME']
        )

        client = MlflowClient()
        curr_version = client.get_latest_versions(
            name=config['tracking']['MODEL_NAME'],
            stages=['None']
        )

        client.update_model_version(
            name=config['tracking']['MODEL_NAME'],
            version=curr_version[0].version,
            description="This model version is a scikit-learn random forest containing 100 decision trees"
        )

        client.transition_model_version_stage(
            name=config['tracking']['MODEL_NAME'],
            version=curr_version[0].version,
            stage="Staging"
        )
