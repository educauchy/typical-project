import pandas as pd
import mlflow.sklearn
from urllib.error import HTTPError
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

csv_url = '../../../data/processed/winequality-red-scoring.csv'
try:
    data = pd.read_csv(csv_url, sep=";")
except HTTPError as e:
    logger.exception("Unable to download training CSV. Error: %s", e)
    raise Exception("Unable to download training CSV. Error: %s", e)

mlflow.set_tracking_uri('http://localhost:5557')

ids = data['id']
data.drop(['id'], axis=1, inplace=True)

model_name = 'sklearn_model_3'
model_stage = 'Staging'
model_path = f'models:/{model_name}/{model_stage}'
model = mlflow.sklearn.load_model(model_path)

data['score'] = model.predict(data)
data['id'] = ids

data.to_csv('../../../data/processed/winequality-red-scoring-result.csv', index=False)
