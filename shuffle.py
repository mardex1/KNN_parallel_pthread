import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')

data.drop("Id", axis=1, inplace=True)

data_copy = data.copy()
label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

data_copy['Species'] = data_copy['Species'].map(label_map)

Xtr, Xts = train_test_split(data, shuffle=True, test_size=0.2, random_state=0)
Xtr_mapped, Xts_mapped = train_test_split(data_copy, shuffle=True, test_size=0.2, random_state=0)

Xts_mapped.to_csv('test_map.csv', index=False)
Xtr.to_csv('train.csv', index=False)
Xts.to_csv('test.csv', index=False)