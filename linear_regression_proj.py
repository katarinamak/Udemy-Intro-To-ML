import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# get the data set
customers = pd.read_csv('/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression'
                        '/Ecommerce Customers')

print(customers.head())

print(customers.info())

print(customers.describe())

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.
plt.figure(0)
sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent']).ax_joint\
    .text(32, 780, stats.pearsonr(customers['Yearly Amount Spent'], customers['Time on Website']))

# Do the same but with the Time on App column instead.
plt.figure(1)
sns.jointplot(x=customers['Time on App'], y=customers['Yearly Amount Spent'])

plt.figure(2)
p = sns.jointplot(x=customers['Time on App'], y=customers['Length of Membership'], kind='hex')

plt.figure(3)
sns.pairplot(customers)

# Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?: Length of membership
# Yearly Amount spent and Length of Membership have a very clear positive correlation (longer membership = more spent)
# Below are the correlations between 'Yearly Amount Spent' and the other features. Yearly Amount Spent vs.
# Length of Membership is closest to a perfect correlation (i.e. r_xy = 1.0)

print("Yearly Amount Spent vs. Avg. Session Length: ",
      customers['Yearly Amount Spent'].corr(customers['Avg. Session Length']))

print("Yearly Amount Spent vs. Time on App: ",
      customers['Yearly Amount Spent'].corr(customers['Time on App']))

print("Yearly Amount Spent vs. Time on Website: ",
      customers['Yearly Amount Spent'].corr(customers['Time on Website']))

print("Yearly Amount Spent vs. Length of Membership: ",
      customers['Yearly Amount Spent'].corr(customers['Length of Membership']))

# Create a linear model plot of Yearly Amount Spent vs. Length of Membership.
plt.figure(4)
sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=customers)

# Model Training
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

print("lm intercept: ", lm.intercept_)
print("lm coefficients: ", lm.coef_)

pred = lm.predict(X_test)

plt.figure(4)
sns.scatterplot(x=y_test, y=pred)

# Evaluate the Model

# Mean Absolute Error
print("MAE: ", metrics.mean_absolute_error(y_test, pred))

# Mean Squared Error
print("MSE: ", metrics.mean_squared_error(y_test, pred))

# Root Mean Squared Error
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, pred)))

# Histogram of residuals
plt.figure(5)
sns.displot((y_test - pred), bins=50, kde=True)
plt.show()

# Coefficients for each feature in dataset
summary = pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])

# If all other features are fixed, then a 1 unit increase in a feature results in [coefficient] increase in Yearly
# Amount Spent
# The following table of coefficients supports our earlier conclusion that the strongest correlation is between
# 'Yearly Amount Spent' and 'Length of Membership'. The coefficients show us that if all other fields are fixed
# The company should focus more on their mobile app than on their website as it has a much higher coefficient and will
# therefore have more impact on the 'Yearly Amount Spent'
print(summary)

