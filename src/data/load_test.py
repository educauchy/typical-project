import logging
from urllib.error import HTTPError
import pyodbc
import pandas as pd


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


load_method = 'csv'

if load_method == 'db':
    conn_str = (
        r'Driver={IBM DB2 ODBC DRIVER};'
        r'Protocol=TCPIP;'
        r'Hostname=XX.X.XXX.XXX;'
        r'Uid=USERNAME;'
        r'Pwd=XXXXXX;'
        r'Database=DATABASE_NAME;'
        r'Schema=SCHEMA_NAME;'
        r'Security=SSL;'
        r'SSLServerCertificate=CERTIFICATE_FULL_PATH;'
    )

    def db2(script):
        with pyodbc.connect(conn_str, timeout=10) as session:
            session.autocommit = True
            df = pd.read_sql_query(script, session)
            session.close()
        return df


    script = "SELECT * FROM SCHEMA.TABLE LIMIT 10"
    df = db2(script)
elif load_method == 'db':
    csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        df = pd.read_csv(csv_url, sep=";")
    except HTTPError as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)
        raise Exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

df.to_csv('../../data/external/winequality-red-train.csv', sep=';', index=False)
