# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import warnings
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


TRACKING_URI    = 'http://localhost:5557'
EXPERIMENT_NAME = 'sklearn_experiment_1'
MODEL_NAME      = "test_model"
ARTIFACT_PATH   = "model"


mlflow.set_tracking_uri(TRACKING_URI)
try:
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
except:
    experiment = mlflow.create_experiment(name=EXPERIMENT_NAME,
                                          artifact_location='file:///Users/educauchy/Documents/Dev/DS/GBC/mlops_project_mlflow/reports/mlruns')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id = run.info.run_id):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.log_artifact('./MLProject')
        mlflow.log_artifact('./conda_environment.yaml')

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            lm = mlflow.sklearn.log_model(sk_model=model,
                                          artifact_path=ARTIFACT_PATH,
                                          registered_model_name=MODEL_NAME,
                                          signature=infer_signature(train_x, model.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict()
                                          )
        else:
            lm = mlflow.sklearn.log_model(sk_model=model,
                                          artifact_path=ARTIFACT_PATH,
                                          registered_model_name=MODEL_NAME,
                                          signature = infer_signature(train_x, model.predict(train_x)),
                                          input_example=train_x.iloc[0].to_dict()
                                          )

        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run.info.run_id,
                                                            artifact_path=lm.artifact_path)
        result = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME
        )

        client = MlflowClient()
        curr_version = client.get_latest_versions(
            name=MODEL_NAME,
            stages=['None']
        )

        client.update_model_version(
            name=MODEL_NAME,
            version=curr_version[0].version,
            description="This model version is a scikit-learn random forest containing 100 decision trees"
        )

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=curr_version[0].version,
            stage="Staging"
        )

