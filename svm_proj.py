import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

# get the iris data
iris = sns.load_dataset('iris')

print(iris.head())

# Exploratory data analysis

# Create a pairplot of the data set. Which flower species seems to be the most separable?
plt.figure(0)
sns.pairplot(hue='species', data=iris, diag_kind='hist')
# plt.show()

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

svm = SVC()
svm.fit(X_train, y_train)

predictions = svm.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# dict where keys are actual parameters that go into the model you are using
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
