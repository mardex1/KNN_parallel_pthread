import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('Datasets/iris.csv')

data.drop("Id", axis=1, inplace=True)

# random_state=5 gera resultados diferentes
Xtr, Xts = train_test_split(data, shuffle=True, test_size=0.2, random_state=0)

Xtr.to_csv('Datasets/train.csv', index=False)
Xts.to_csv('Datasets/test.csv', index=False)