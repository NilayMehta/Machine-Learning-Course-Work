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

if(not os.path.exists('plots')):
    dirname = 'plots'
    os.mkdir(dirname)
    dirname = 'HandwritingVerification'
    os.mkdir('plots/%s' %dirname)

if(not os.path.exists('plots/HandwritingVerification')):
    dirname = 'HandwritingVerification'
    os.mkdir('plots/%s' %dirname)
    dirname = 'DecisionTree'
    os.mkdir('plots/HandwritingVerification/%s' %dirname)

if(not os.path.exists('plots/HandwritingVerification/DecisionTree')):
    dirname = 'DecisionTree'
    os.mkdir('plots/HandwritingVerification/%s' %dirname)

# print(df.head(5))

df = pd.get_dummies(df)
# print(df.head(5))

Z = df.ix[:, df.columns != 'CLASS_DISTINCT']
X = Z.ix[:, Z.columns != 'CLASS_SAME']
y = df['CLASS_DISTINCT']

# print(X.head(5))
print(y.head(5))



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#DecisionTreeClassifier
max_depth = range(2, 22)
train_err = [0] * len(max_depth)
test_err = [0] * len(max_depth)

for i, d in enumerate(max_depth):
    print 'learning a decision tree with max_depth=' + str(d)
    clf = DecisionTreeClassifier(max_depth=d)
    clf = clf.fit(X_train, y_train)
    train_err[i] = mean_squared_error(y_train,
                                     clf.predict(X_train))
    test_err[i] = mean_squared_error(y_test,
                                    clf.predict(X_test))

    print 'train_err: ' + str(train_err[i])
    print 'test_err: ' + str(test_err[i])
    print '---'

    DT_predictions = clf.predict(X_test)
    DT_accuracy = accuracy_score(y_test, DT_predictions)
    print 'Handwriting Verification Tree Accuracy:', DT_accuracy
    print '-----------------'



# Plot results
print 'plotting results'
plt.figure()
plt.title('Decision Trees: Performance x Max Depth')
plt.plot(max_depth, test_err, '-', label='test error')
plt.plot(max_depth, train_err, '-', label='train error')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('Mean Square Error')
plt.savefig('plots/HandwritingVerification/DecisionTree/HandwritingVerification_DT_PerformancexMaxDepth.png')
# plt.show()
### ---



### Training trees of different training set sizes (fixed max_depth=8)
train_size = len(X_train)
offsets = range(int(0.1 * train_size), train_size, int(0.05 * train_size))
max_depth_range = range(6, 20, 2)
train_err = [0] * len(offsets)
test_err = [0] * len(offsets)

for a, d in enumerate(max_depth_range):

    print 'training_set_max_size:', train_size, '\n'
    for i, o in enumerate(offsets):
        print 'learning a decision tree with training_set_size=' + str(o)
        clf = DecisionTreeClassifier(max_depth=d)
        X_train_temp = X_train[:o].copy()
        y_train_temp = y_train[:o].copy()
        X_test_temp = X_test[:o].copy()
        y_test_temp = y_test[:o].copy()

        clf = clf.fit(X_train_temp, y_train_temp)

        train_err[i] = mean_squared_error(y_train_temp,
                clf.predict(X_train_temp))
        test_err[i] = mean_squared_error(y_test_temp,
                                         clf.predict(X_test_temp))
        print 'train_err: ' + str(train_err[i])
        print 'test_err: ' + str(test_err[i])
        print '---'

        DT_predictions = clf.predict(X_test)
        DT_accuracy = accuracy_score(y_test, DT_predictions)
        print 'Handwriting Verification Tree Accuracy:', DT_accuracy
        print '-----------------'

    # Plot results
    print 'plotting results'
    plt.figure()
    plt.title('Decision Trees: Performance x Training Set Size for Max Depth ' + str(d))
    plt.plot(offsets, test_err, '-', label='test error')
    plt.plot(offsets, train_err, '-', label='train error')
    plt.legend()
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Square Error')
    filename = 'HandwritingVerification_DT_PerformancexTrainingSetSize_MAXDEPTH=' + str(d) + '.png'
    plt.savefig('plots/HandwritingVerification/DecisionTree/%s' % filename)
    print 'plot complete'
    # plt.show()
### ---