import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn
df = pd.read_csv('~/Desktop/MCM/data_all.csv', header=0, sep=',')
df1 = pd.read_csv('~/Desktop/MCM/grouped_county1.csv', header=None, sep=',')
df = df.drop(['GEO.id','GEO.display-label'], axis=1)

df = df.fillna(df.mean())

df_all = pd.merge(df1, df, left_on = [3,4], right_on = ['GEO.id2','HC_YEAR'], how = 'inner',
        suffixes=('_x', '_y'), indicator = False)
df_all = df_all.drop(4, axis=1)
df_all = df_all.drop(3, axis=1)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
Y = df_all[5]
df_all = df_all.drop(5, axis=1)
df_all = df_all.drop('GEO.id2', axis=1)
df_all = df_all.drop(0, axis=1)
df_all = df_all.drop(1, axis=1)
df_all = df_all.drop(2, axis=1)

for i in range(5):                           #这里我们进行十次循环取交集
    tmp = set()
    rfc = RandomForestRegressor()
    rfc.fit(df_all, Y)
    print("training finished")

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1]   # 降序排列
    for f in range(50):
        tmp.add(df_all.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, df_all.columns[indices[f]], importances[indices[f]]))
    print("features are selected")

