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
#  df_all = df_all.drop('GEO.id2', axis=1)
#  df_all = df_all.drop(0, axis=1)
#  df_all = df_all.drop(1, axis=1)
#  df_all = df_all.drop(2, axis=1)
df_all.to_csv('data_all_modified1_ori.csv', sep=',')

df_all2 = pd.read_csv('MCM_Pre_onehot.csv', header=0, sep=',')
df_all3 = pd.read_csv('MCM_Pre_ori.csv', header=0, sep=',')
train_onehot = pd.concat([df_all2, Y], axis=1)
train_onehot.to_csv('train_onehot.csv', sep=',')
train_ori = pd.concat([df_all3, Y], axis=1)
train_ori.to_csv('train_ori.csv', sep=',')

#  for i in range(5):                           #这里我们进行十次循环取交集
    #  tmp = set()
    #  rfc = RandomForestRegressor()
    #  rfc.fit(df_all, Y)
    #  print("training finished")
#
    #  importances = rfc.feature_importances_
    #  indices = np.argsort(importances)[::-1]   # 降序排列
    #  for f in range(50):
        #  tmp.add(df_all.columns[indices[f]])
        #  print("%2d) %-*s %f" % (f + 1, 30, df_all.columns[indices[f]], importances[indices[f]]))
    #  print("features are selected")
rfc = RandomForestRegressor()
rfc.fit(df_all3, Y)
print(rfc.score(df_all3, Y))

#  df_all_sta = df_all
#  df_all_sta['HC01_VC12'] = df_all_sta['HC01_VC12'].apply(lambda x:x*0.8)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy1_0.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC12'] = df_all_sta['HC01_VC12'].apply(lambda x:x*1.2)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy1_1.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC70'] = df_all_sta['HC01_VC70'].apply(lambda x:0.8*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy2_0.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC70'] = df_all_sta['HC01_VC70'].apply(lambda x:1.2*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy2_1.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC15'] = df_all_sta['HC01_VC15'].apply(lambda x:0.8*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy3_0.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC15'] = df_all_sta['HC01_VC15'].apply(lambda x:1.2*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy3_1.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC13'] = df_all_sta['HC01_VC13'].apply(lambda x:0.8*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy4_0.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC13'] = df_all_sta['HC01_VC13'].apply(lambda x:1.2*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy4_1.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC104'] = df_all_sta['HC01_VC104'].apply(lambda x:0.8*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy5_0.csv', sep=',')
#
#  df_all_sta = df_all
#  df_all_sta['HC01_VC104'] = df_all_sta['HC01_VC104'].apply(lambda x:1.2*x)
#  y = pd.DataFrame(columns=['target'], data = rfc.predict(df_all_sta))
#  pd.concat([df_all_sta, y], axis=1).to_csv('strategy5_1.csv', sep=',')
#
