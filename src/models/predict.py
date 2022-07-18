import mlflow.sklearn
import pandas as pd


sk_model = mlflow.sklearn.load_model("../mlruns/0/736acbfcad6041a7a2abf3e1b0df3f97/artifacts/model/")

df = pd.read_csv('../../data/processed/winequality-red-test.csv', sep=';', decimal='.')
dataset_id = df['id']
df.drop(['quality', 'id'], axis=1, inplace=True)
pred = sk_model.predict(df)

predictions = pd.DataFrame({'id': dataset_id,
                            'pred_score': pred})
