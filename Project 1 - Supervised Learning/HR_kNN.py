import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os
from textwrap import wrap

df = pd.read_csv('datasets/HR_Analytics.csv', delimiter=',', quotechar='"')

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'HR_Analytics'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/HR_Analytics')):
    dirname = 'HR_Analytics'
    os.mkdir('plots/%s' %dirname)
    dirname = 'kNN'
    os.mkdir('plots/HR_Analytics/%s' %dirname)

if(not os.path.exists('plots/HR_Analytics/kNN')):
    dirname = 'kNN'
    os.mkdir('plots/HR_Analytics/%s' %dirname)


df = pd.get_dummies(df)
# print(df.head(5))

# df['left'] = np.where(df['left'] == 1, 0)
X = df.ix[:, df.columns != 'left']
y = df['left']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30, random_state=30)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#kNN
ks = range(1, 21)
train_err = [0] * len(ks)
test_err = [0] * len(ks)

for i, k in enumerate(ks):
    print 'kNN: learning a kNN classifier with k = ' + str(k)
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    train_err[i] = mean_squared_error(y_train,
                                     clf.predict(X_train))
    test_err[i] = mean_squared_error(y_test,
                                    clf.predict(X_test))
    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

# Plot results
print 'plotting results'
plt.figure()
title = 'HR Analytics kNN: Performance'
plt.title('\n'.join(wrap(title,60)))
plt.plot(ks, test_err, '-', label='test error')
plt.plot(ks, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Num Estimators')
plt.ylabel('Mean Square Error')
plt.savefig('plots/HR_Analytics/kNN/HRAnalytics_kNN.png')
print 'plot complete'
### ---