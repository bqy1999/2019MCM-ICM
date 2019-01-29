import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

df = pd.read_csv('drugs0.csv', header=None, sep=',')
#  print(df)
X = df[0]
Y = df[1]
Z = df[2]

ax.scatter(X, Y, Z)
ax.set_zlabel('HeroinReports')
ax.set_ylabel('YYYY')
ax.set_xlabel('State')
#  ax.scatter(X[:8], Y[:8], Z[:8], c='y')
#  ax.scatter(X[8:16], Y[8:16], Z[8:16], c='r')
#  ax.scatter(X[16:24], Y[16:24], Z[16:24], c='g')
#  ax.scatter(X[24:32], Y[24:32], Z[24:32], c='p')
#  ax.scatter(X[32:40], Y[32:40], Z[32:40], c='b')
#  ax.plot_trisurf(X, Y, Z, cmap='rainbow')
# ax.plot_trisurf(X, Y, Z)
plt.show()
