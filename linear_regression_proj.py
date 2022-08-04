import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# get the data set
customers = pd.read_csv('/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression'
                        '/Ecommerce Customers')

print(customers.head())

print(customers.info())

print(customers.describe())

# Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the
# correlation make sense?
plt.figure(1)
sns.jointplot(x=customers['Time on Website'], y=customers['Yearly Amount Spent'])

# Do the same but with the Time on App column instead.
plt.figure(2)
sns.jointplot(x=customers['Time on App'], y=customers['Yearly Amount Spent'])

plt.figure(3)
p = sns.jointplot(x=customers['Time on App'], y=customers['Length of Membership'], kind='hex')
plt.show()
