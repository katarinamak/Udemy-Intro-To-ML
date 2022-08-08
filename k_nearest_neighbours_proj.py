import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/'
                 '14-K-Nearest-Neighbors/KNN_Project_Data')

print(df.head())

# Data Analysis
plt.figure(0)
sns.pairplot(data=df, hue='TARGET CLASS')


# Standardize the Variables

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))


scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_feat.head())

# Split data into a training set and a testing set
X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

# predictions and evaluations
predictions = knn.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

err_rate = []
# iterate many models using many diff k values and plot out error rate to see which has lowest error rate
for i in range(1, 60):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    err_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 60), err_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

plt.show()

# Based on the plot above, an optimal k value would be k=30, so we can retrain the model using n_neighbors=30
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Following the optimization of k, we see that the accuracy has increased by 11% (from 72% accuracy to 83% accuracy)

