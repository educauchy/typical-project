import mlflow.sklearn
import pandas as pd
from scipy.stats import norm

sk_model = mlflow.sklearn.load_model("../mlruns/0/736acbfcad6041a7a2abf3e1b0df3f97/artifacts/model/")

df = pd.read_csv('./data/winequality-red-test.csv', sep=';', decimal='.')
id = df['id']
df.drop(['quality', 'id'], axis=1, inplace=True)
pred = sk_model.predict(df)

predictions = pd.DataFrame({'id': id,
                            'pred_score': pred})

print(norm.ppf(1-0.05/2))
print(norm.ppf(1-0.2))
print(norm.ppf(1-0.01/2))

# separate dataset into train and test
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('./data/winequality-red.csv', sep=';', decimal='.')
# df['id'] = range(1, len(df) + 1)
# df_train, df_test = train_test_split(df, test_size=0.25)
# df_train.to_csv('./data/winequality-red-train.csv', sep=';', index=False)
# df_train.to_csv('./data/winequality-red-test.csv', sep=';', index=False)
