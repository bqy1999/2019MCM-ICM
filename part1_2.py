from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import pandas as pd
import numpy as np

df = pd.read_csv('grouped_county.csv', header=None, sep = ',')

df[[0]] = df[[0]].astype(int)
county_onehot[[0]] = county_onehot[[0]].astype(int)

y_train = df[4]
x_train = df.drop(4, axis=1)
x_need = x_train
x_need[3] = x_need[3].apply(lambda x:x+8)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
sc = StandardScaler()

x_train_std = x_train
enc.fit([x_train[0]])
x_train_std[0] = enc.transform([x_train[0]]).toarray()
print(x_train_std.head())
enc.fit([x_train[1]])
x_train_std[1] = [enc.transform([x_train[1]])]
enc.fit([x_train[2]])
x_train_std[2] = [enc.transform([x_train[2]])]
sc.fit([x_train[3]])
x_train_std[3]= [sc.transform([x_train[3]])]
print(x_train_std)

x_need = df.drop(4, axis=1)

x_need_std = x_need
enc.fit([x_need[0]])
x_need_std[0] = pd.DataFrame(enc.transform([x_need[0]]))
enc.fit([x_need[1]])
x_need_std[1] = pd.DataFrame(enc.transform([x_need[1]]))
enc.fit([x_need[2]])
x_need_std[2] = pd.DataFrame(enc.transform([x_need[2]]))
sc.fit([x_need[3]])
x_need_std[3]= pd.DataFrame(sc.transform([x_need[3]]))
print(x_need_std)

#  enc.fit(x_need.drop(3,axis=1))
#  sc.fit([x_need[3]])
#  x_need_std = pd.DataFrame(enc.transform(x_need.drop(3,axis=1)).toarray())
#  year_need = pd.DataFrame(sc.transform([x_need[3]]))
#  x_need_std  = pd.concat([x_need_std, year_need], axis=1)

# Model predict
rf = RandomForestRegressor()
rf.fit(x_train_std,y_train)
print(rf.score(x_train,y_train))
last = pd.DataFrame(columns=['target'], data = rf.predict(x_need_std))

y_need = pd.concat([x_need, last], axis=1)
y_need.to_csv('predict_county_7.csv', sep=',')
