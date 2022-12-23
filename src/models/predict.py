import sys
import pandas as pd
import mlflow.sklearn
from urllib.error import HTTPError
import logging


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


scoring_url = sys.argv[2]
model_name = sys.argv[3]
model_stage = sys.argv[4]
scoring_result_url = sys.argv[5]

try:
    data = pd.read_csv(scoring_url, sep=",", error_bad_lines=False)
except HTTPError as e:
    logger.exception("Unable to download training CSV. Error: %s", e)
    raise Exception("Unable to download training CSV. Error: %s", e)

mlflow.set_tracking_uri('http://localhost:5558')

# ids = data['id']
# data.drop(columns=['id'], inplace=True)

model_path = f'models:/{model_name}/{model_stage}'
model = mlflow.sklearn.load_model(model_path)

data['score'] = model.predict(data)
# data['id'] = ids

data.to_csv('../../../data/processed/winequality-red-scoring-result.csv', sep=';', decimal='.', index=False)
