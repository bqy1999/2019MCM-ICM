import pandas as pd
import numpy as np

df = pd.read_csv('grouped_county.csv', header=None, sep=',')
print(df.head())
ss = df[2]

print(ss.value_counts())
print(ss.unique())
