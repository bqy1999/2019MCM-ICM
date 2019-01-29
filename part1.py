from pandas import DataFrame
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv('MCM_NFLIS_Data.csv', header=None, sep=',')
onehot = pd.read_csv('onehot.csv', header=None, sep=',')
df = df.drop(1, axis=1)
df = df.drop(2, axis=1)
#  df = df.drop(3)
#  df = df.drop(4)
#  print(df.tail())
#  print(onehot.tail())

res = pd.merge(df, onehot, left_on=6, right_on=0, how = 'left',
      suffixes=('_x', '_y'), indicator=False)
res = res.drop('0_y', axis=1)
res.columns = ['YYYY','State','County','FIPS','Name','DrugReports','TotalDrugReportsCounty','TotalDrugsReportsState','Tag']
Name = res['Name']
res = res.drop('Name', axis=1)
#  State = res['FIPS']
#  res = res.drop('FIPS', axis=1)
#  State = res['State']
#  res = res.drop('State', axis=1)
#  County = res['County']
#  res = res.drop('County', axis=1)
#  TotalDrugReportsCounty = res['TotalDrugReportsCounty']
res = res.drop('TotalDrugReportsCounty', axis=1)
#  TotalDrugReportsState = res['TotalDrugReportsState']
res = res.drop('TotalDrugsReportsState', axis=1)
res = res.drop(0, axis=0)
#  res = res.drop(0, axis=1)

#  print(res.tail())
#  res = res.groupby(['YYYY','State','County','FIPS','Tag'])
print(res)
#  res.to_csv('presol1.csv', sep=',')


#  res = res['DrugReports'].groupby([res['Tag'],res['FIPS'],res['YYYY']])
#  res.sum()
#  res = res.to_frame()

res = res.reset_index()
res['DrugReports'] = res['DrugReports'].astype(int)
#  res.astype(float)
ans = res.groupby([res['Tag'], res['State'], res['County'], res['FIPS'], res['YYYY']])['DrugReports']
#  ans = res.groupby([res['Tag'], res['State'], res['YYYY']])['DrugReports']
ans_sum = ans.sum()

#  print(grouped.sum())
#  print(grouped)

print(ans_sum)
#  print(res)
#  print(res.head())
#  print(res.tail())

ans_sum.to_csv('grouped_county1.csv', sep=',')
