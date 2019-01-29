from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import pandas as pd
import numpy as np

df = pd.read_csv('grouped_county.csv', header=None, sep = ',')
drug_onehot = pd.read_csv('drug_onehot.csv', header=None, sep = ',')
state_onehot = pd.read_csv('state_onehot.csv', header=None, sep  =',')
county_onehot = pd.read_csv('county_onehot.csv', header=None, sep  =',')

df[[0]] = df[[0]].astype(int)
county_onehot[[0]] = county_onehot[[0]].astype(int)

#  print(df.head())
#  print(drug_onehot.head())
#  print(state_onehot.head())

res_0 = pd.merge(df, drug_onehot, left_on = 0, right_on = 0, how = 'left',
        suffixes=('_x', '_y'), indicator = False)
#  print(res_0.head())
#  print(res_0.tail())
res_0 = res_0.drop(0, axis=1)

res_1 = pd.merge(res_0, state_onehot, left_on = '1_x', right_on = 0, how = 'left',
        suffixes=('_x_1', '_y_1'), indicator = False)
#  print(res_1.head())
res_1 = res_1.drop('1_x', axis = 1)
res_1 = res_1.drop(0, axis = 1)
#  res_1['YYYY'] = res_1['YYYY'].apply(lambda x:x-2010)
#  print(res_1.head())
#  print(res_1.tail())

res_2 = pd.merge(res_1, county_onehot, left_on = '2_x', right_on = 0, how = 'left',
        suffixes=('_x_2', '_y_2'), indicator = False)

res_2 = res_2.drop('2_x', axis = 1)
res_2 = res_2.drop(0, axis = 1)
res_2.columns = ['YYYY','DrugReports','DrugCode_0','DrugCode_1','StateCode_0','StateCode_1','StateCode_2','CountyCode_0','CountyCode_1','CountyCode_2','CountyCode_3','CountyCode_4','CountyCode_5','CountyCode_6','CountyCode_7']
#  print(res_2.head())
#  print(res_2.tail())

y_train = res_2['DrugReports']
x_train = res_2.drop('DrugReports', axis=1)
x_train['YYYY'] = x_train['YYYY'].apply(lambda x:x-2010)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)

rf = RandomForestRegressor(max_features=14)
rf.fit(x_train_std,y_train)
print(rf.score(x_train,y_train))

x_need = res_2.drop('DrugReports', axis=1)
x_need['YYYY'] = x_need['YYYY'].apply(lambda x:x-2002)
print(x_need.head())
x_need_std = sc.transform(x_need)
#  last = pd.dataframe(columns=['target'], data = rf.predict(x_need_std))
last = pd.DataFrame(columns=['target'], data = rf.predict(x_need_std))
df[3] = df[3].apply(lambda x:x+8)
df = df.drop(4, axis=1)

y_need = pd.concat([df, last], axis=1)
y_need.to_csv('predict_county_6.csv', sep=',')
