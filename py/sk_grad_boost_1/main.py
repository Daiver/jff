# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
# License: BSD 3 clause

import numpy as np

import npexport

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

###############################################################################
# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

#npexport.exportMatTxt('X_train.txt', X_train)
#npexport.exportMatTxt('X_test.txt', X_test)
#npexport.exportMatTxt('y_train.txt', y_train.reshape((-1, 1)))
#npexport.exportMatTxt('y_test.txt' , y_test.reshape((-1, 1)))

###############################################################################
# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 
          #'loss': 'lad'
          #'loss': 'huber'
          'loss': 'ls'
          }
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

mse = mean_squared_error(y_train, clf.predict(X_train))
print("MSE Train: %.4f" % mse)

