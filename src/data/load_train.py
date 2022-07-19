import logging
from urllib.error import HTTPError
import pandas as pd


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
    df = pd.read_csv(csv_url, sep=";")
except HTTPError as e:
    logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)
    raise Exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

df.to_csv('../../data/external/winequality-red-train.csv', sep=';', index=False)
