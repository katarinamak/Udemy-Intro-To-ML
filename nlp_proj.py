import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

yelp = pd.read_csv('/Users/katarinamakivic/Downloads/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language'
                   '-Processing/yelp.csv')

# Dataframe summary
print(yelp.head())
print(yelp.describe())
print(yelp.info())

yelp['text length'] = yelp['text'].apply(len)

print(yelp.head())

# Exploratory Data Analysis
# Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings.
plt.figure()
fg = sns.FacetGrid(yelp, col="stars")
fg.map_dataframe(plt.hist, x="text length")

# Create a boxplot of text length for each star category.
plt.figure()
sns.boxplot(data=yelp, y='text length', x='stars')

# Create a countplot of the number of occurrences for each type of star rating
plt.figure()
sns.countplot(x='stars', data=yelp)

# Use groupby to get the mean values of the numerical columns
grouped = pd.DataFrame(yelp.groupby('stars').mean())
print(grouped.head())

print('\n')

# Use the corr() method on that groupby dataframe
print(grouped.corr())

# Then use seaborn to create a heatmap based off that .corr() dataframe
plt.figure()
sns.heatmap(data=grouped.corr())

# plt.show()

# NLP Classification Task

# Create a dataframe that contains the columns of yelp dataframe but for only the 1 or 5 star reviews
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]

X = yelp_class['text']
y = yelp_class['stars']

cv = CountVectorizer()

# remove stop words (meaningless) and create a sparse matrix with the count of each word
X = cv.fit_transform(X)
print(X)

# Train a model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

nb = MultinomialNB()

nb.fit(X_train, y_train)

predictions = nb.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Using Text Processing

# Create a pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB())  # train on TF-IDF vectors w/ Naive Bayes classifier
])

X = yelp_class['text']
y = yelp_class['stars']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# fit the pipeline to the training data
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Based on the results in the above confusion matrix and classification report, the model was better without Tf-Idf
