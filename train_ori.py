import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
df = pd.read_csv('~/Desktop/MCM/train_ori.csv', header=0, sep=',')
#  print(df.head())

y = df['target']
X = df.drop('target', axis=1)
#  X = X.drop('GEO.id2', axis=1)

rfc = RandomForestRegressor()
rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC12'] = X_modified['HC01_VC12'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori1.csv', sep=',')

rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC70'] = X_modified['HC01_VC70'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori2.csv', sep=',')

rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC15'] = X_modified['HC01_VC15'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori3.csv', sep=',')

rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC13'] = X_modified['HC01_VC13'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori4.csv', sep=',')

rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC104'] = X_modified['HC01_VC104'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori5.csv', sep=',')

rfc.fit(X, y)
print(rfc.score(X, y))

ans = pd.DataFrame(columns=['target'], data = rfc.predict(X))
for i in range(5,15):
    rfc.fit(X, y)
    X_modified = X
    X_modified['HC01_VC87'] = X_modified['HC01_VC87'].apply(lambda x:x*i*0.1)
    y_modified = pd.DataFrame(columns=[i-5], data = rfc.predict(X_modified))
    ans = pd.concat([ans, y_modified], axis=1)
ans.to_csv('strategy_ori6.csv', sep=',')

