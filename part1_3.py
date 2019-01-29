from pandas import DataFrame
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df = pd.read_csv('trains_first.csv', header=None , sep=',')
df = df.drop(0, axis=1)
df = df.drop(0, axis=0)
df.columns = ['YYYY','DrugReports','DrugCode_0','DrugCode_1','StateCode_0','StateCode_1','StateCode_2','CountyCode_0','CountyCode_1','CountyCode_2','CountyCode_3','CountyCode_4','CountyCode_5','CountyCode_6','CountyCode_7']
df[['YYYY']] = df[['YYYY']].astype(int)
df['YYYY'] = df['YYYY'].apply(lambda  x:x-2010)

y_train = df['DrugReports']
x_train = df.drop('DrugReports', axis=1)
#  print(y_train.head())
#  print(x_train.head())

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)

rf=RandomForestRegressor(max_features=14)
rf.fit(x_train, y_train)
print(rf.score(x_train, y_train))

#  tuned_parameters = [{'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10], 'n_estimators':[50,100,150,200,250,300]}]
#  clf = GridSearchCV(estimator=rf,param_grid=tuned_parameters, cv=5, n_jobs=1)
#  clf.fit(x_train, y_train)
#  print('Best parameters:')
#  print(clf.best_params_)

x_need = x_train
x_need['YYYY'] = x_need['YYYY'].apply(lambda x:x+8)
x_need_std = sc.transform(x_need)
last = pd.DataFrame(data = rf.predict(x_need_std))
x_need['YYYY'] = x_need['YYYY'].apply(lambda x:x+2010)

print(x_need.head())
y_need = pd.concat([x_need, last], axis=1)
y_need.to_csv('predict_county_4.csv', sep=',')

