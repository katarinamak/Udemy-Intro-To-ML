import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ad_data = pd.read_csv("/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression"
                      "/advertising.csv")

# Get a summary of the data
print(ad_data.head())
print(ad_data.describe())
print(ad_data.info())

# Analysis

# Create a histogram of the Age
plt.figure(0)
ad_data['Age'].plot.hist(bins=30, edgecolor='black', linewidth=1)

# Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age', y='Area Income', data=ad_data).ax_marg_x.set_xlim(10, 70)

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='Red')

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='Green')\
    .ax_marg_x.set_xlim(20, 100)

# Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(data=ad_data, hue='Clicked on Ad')

# plt.show()


# Logistic Regression

print(ad_data['City'])

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Predicted values for the testing data
predictions = log_model.predict(X_test)

# Classification report for the model
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))