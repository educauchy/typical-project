import logging
from urllib.error import HTTPError
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

csv_url = '../../data/external/winequality-red-train.csv'
try:
    df = pd.read_csv(csv_url, sep=";")
except HTTPError as e:
    logger.exception("Unable to download training CSV, check presence of the file. Error: %s", e)
    raise Exception("Unable to download training CSV, check presence of the file. Error: %s", e)

df['id'] = range(1, len(df) + 1)

df_train, df_test = train_test_split(df, stratify=df['quality'], test_size=0.25, random_state=0)
df_train.to_csv('../../data/processed/winequality-red-train.csv', sep=';', index=False)
df_train.to_csv('../../data/processed/winequality-red-test.csv', sep=';', index=False)
