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


df = pd.read_csv('datasets/HR_Analytics.csv', delimiter=',', quotechar='"')

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'HR_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/HR_Analytics')):
    dirname = 'HR_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'DecisionTree'
    os.mkdir('plots/HR_Analytics/%s' %dirname)

if(not os.path.exists('plots/HR_Analytics/DecisionTree')):
    dirname = 'DecisionTree'
    os.mkdir('plots/HR_Analytics/%s' %dirname)


# mapping = {'low': 1, 'medium': 2, 'high': 3}
# df.replace({'salary': mapping})
# print(df.head(5))

df = pd.get_dummies(df)
print(df.head(5))

df.to_csv('datasets/HR_Analytics_updated.csv')