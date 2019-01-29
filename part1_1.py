from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv('grouped_state.csv', header=None, sep = ',')
drug_onehot = pd.read_csv('drug_onehot.csv', header=None, sep = ',')
state_onehot = pd.read_csv('state_onehot.csv', header=None, sep  =',')

df[[0]] = df[[0]].astype(int)

#  print(df.head())
#  print(drug_onehot.head())
#  print(state_onehot.head())

res_0 = pd.merge(df, drug_onehot, left_on = 0, right_on = 0, how = 'outer',
        suffixes=('_x', '_y'), indicator = False)
#  print(res_0.head())
#  print(res_0.tail())
res_0 = res_0.drop(0, axis=1)

res_1 = pd.merge(res_0, state_onehot, left_on = '1_x', right_on = 0, how = 'outer',
        suffixes=('_x', '_y'), indicator = False)
res_1 = res_1.drop('1_x', axis = 1)
res_1 = res_1.drop(0, axis = 1)
res_1.columns = ['YYYY','DrugReports','DrugCode_0','DrugCode_1','StateCode_0','StateCode_1','StateCode_2']
#  res_1['YYYY'] = res_1['YYYY'].apply(lambda x:x-2010)
#  print(res_1.head())
#  print(res_1.tail())

y_train = res_1['DrugReports']
#  print(y_train.head())
x_train = res_1.drop('DrugReports', axis=1)
#  print(x_train.head())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
print(sc.mean_)
print(sc.scale_)
x_train_std = sc.transform(x_train)

#  rf=RandomForestRegressor(max_depth=None,random_state=0)
#  tuned_parameter = [{'min_samples_leaf':[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],'n_estimators':[115,120,125,130]}]
#  clf = GridSearchCV(estimator=rf,param_grid=tuned_parameter,cv=5,n_jobs=1)
#  clf.fit(x_train,y_train);

#  print('Best parameters:')
#  print(clf.best_params_)

#  rf=RandomForestRegressor(n_estimators = 120)
rf=RandomForestRegressor()
rf.fit(x_train_std,y_train)

x_need = pd.read_csv('target_x.csv', header=None, sep=',')
x_need_std = sc.transform(x_need)
print(rf.predict(x_need_std))
