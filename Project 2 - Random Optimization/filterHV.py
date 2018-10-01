import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os


df = pd.read_csv('datasets/HandwritingVerification.csv', delimiter=',', quotechar='"')

# mapping = {'low': 1, 'medium': 2, 'high': 3}
# df.replace({'salary': mapping})
# print(df.head(5))

df = pd.get_dummies(df)
# Z = df.ix[:, df.columns != 'CLASS_DISTINCT']
# X = Z.ix[:, Z.columns != 'CLASS_SAME']
# y = df['CLASS_DISTINCT']
print(df)

msk = np.random.rand(len(df)) < 0.7

df_train = df[msk]
df_test = df[~msk]

# msk = np.random.rand(len(df_test)) < 0.05

# df_train = df_test[msk]
# df_test = df_test[~msk]

# msk = np.random.rand(len(df_test)) < 0.7

# df_train = df_test[msk]
# df_test = df_test[~msk]


print(len(df_test))
print(len(df_train))
print(len(df_train) + len(df_test))

df_train.to_csv('datasets/HV_train2.csv')
df_test.to_csv('datasets/HV_test2.csv')