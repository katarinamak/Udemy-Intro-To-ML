import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/15-Decision-Trees-and-Random'
                 '-Forests/loan_data.csv')

# Summary of the dataset
print(df.head())
print(df.describe())
print(df.info())

# Data Analysis

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.
plt.figure(0)
df[df['credit.policy'] == 1]['fico'].hist(color='blue', edgecolor='black', linewidth=0.5, bins=35,
                                          label='credit.policy=1', alpha=0.5)
df[df['credit.policy'] == 0]['fico'].hist(color='red', edgecolor='black', linewidth=0.5, bins=35,
                                          label='credit.policy=0', alpha=0.5)
plt.legend()


# Create a similar figure, except this time select by the not.fully.paid column.
plt.figure(1)
df[df['not.fully.paid'] == 1]['fico'].hist(color='blue', edgecolor='black', linewidth=0.5, bins=35,
                                          label='not.fully.paid=1', alpha=0.5)
df[df['not.fully.paid'] == 0]['fico'].hist(color='red', edgecolor='black', linewidth=0.5, bins=35,
                                          label='not.fully.paid=0', alpha=0.5)
plt.legend()

# Create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(2, figsize=(12, 8))
sns.countplot(x='purpose', hue='not.fully.paid', data=df)

# Let's see the trend between FICO score and interest rate. Create a jointplot:
plt.figure(3)
sns.jointplot(x='fico', y='int.rate', data=df, ylim=(0, 0.25), color='purple')

# Create lmplots to see if the trend differed between not.fully.paid and credit.policy.
plt.figure(4, figsize=(12, 8))
sns.lmplot(x='fico', y='int.rate', data=df, hue='credit.policy', col='not.fully.paid')

# plt.show()

# Set up the data

# pd.get_dummies(df)
cat_feats = ['purpose']
final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)

print(final_data.head())

# Train the model
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# model of decision tree classifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Evaluate how well the decision tree was able to predict based off of the given columns
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Compare decision tree results to random forest model
# note: n_estimators is the number of trees in the forest
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

rfc_predictions = rfc.predict(X_test)

print(classification_report(y_test, rfc_predictions))
print(confusion_matrix(y_test, rfc_predictions))

# What performed better the random forest or the decision tree? Overall, the random forest had a better accuracy
# therefore we can conclude that it performed better in its predictions The accuracy of the random forest was 85%,
# while the accuracy of the decision tree was 73%. However, the recall for the decision tree was better for class 1
# than that of the random forest, and the f1-score for class 1 was better for the decision tree while the opposite
# was true for class 0
