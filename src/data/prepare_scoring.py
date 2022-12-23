import numpy as np
import pandas as pd


data = {
    'id': np.random.randint(1, 1_000_000, 1)[0],
    'fixed acidity': 6.6,
    'volatile acidity': 0.52,
    'citric acid': 0.04,
    'residual sugar': 2.2,
    'chlorides': 0.069,
    'free sulfur dioxide': 8,
    'total sulfur dioxide': 15,
    'density': 0.9956,
    'pH': 3.4,
    'sulphates': 0.63,
    'alcohol': 9.4,
}

df = pd.DataFrame(data=[data])
# sep=',' а не ';', потому что нельзя задать сепаратор, когда вызываем датафрейм на скоринг из терминала
df.to_csv('../../data/processed/winequality-red-scoring.csv', sep=';', decimal='.', index=False)
