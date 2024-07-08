import pandas as pd

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("used_data/IMDbDataFinal.csv")
le = LabelEncoder()
data['Director'] = le.fit_transform(data['Director'])
data['Star1'] = le.fit_transform(data['Star1'])
data['Star2'] = le.fit_transform(data['Star2'])
data['genres'] = le.fit_transform(data['genres'])
data['originalTitle'] = le.fit_transform(data['originalTitle'])
data['primaryTitle'] = le.fit_transform(data['primaryTitle'])
data = data.drop(labels = ['titleType'], axis = 1)
data = data.drop(labels = ['tconst'], axis = 1)

y = data["averageRating"]
x = data.drop(["averageRating"], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=101)

rf_Model = RandomForestRegressor()
rf_Model.fit(X_train,Y_train)

rf_Model.oob_score_