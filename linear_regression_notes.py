import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# df =
# df.head()
# df.columns
# sns.pairplot(df)
# sns.distplot(df['Price'])
# sns.heatplot(df.corr(), annot=True)

# split into x array and y array
X = df[[df.columns]]
y = df['Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)  # only want to fit to training data

print(lm.intercept_)
print(lm.coef_)
print(X_train.columns)

cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])

from sklearn.datasets import load_boston

boston = load_boston()
boston.keys()
print(boston['DESCR'])

# Predictions

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

# residuals = difference between actual values and test values
sns.distplot((y_test-predictions))  # histogram of residuals -- normally distrib residuals = model is correct choice
# for the data. Not normally distributed -- is linear regression model right choice for model

from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)

metrics.mean_squared_error(y_test, predictions)

np.sqrt(metrics.mean_squared_error(y_test, predictions))

