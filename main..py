# Import Libraries
# -----------------
# Importing necessary libraries for data manipulation and visualization
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For more advanced plotting


# Data Loading and Initial Exploration
# -----------------
spam_df = pd.read_csv('emails.csv')  # Loading dataset from a CSV file
spam_df.head()  # Displaying the first 5 rows of the dataframe
spam_df.tail()  # Displaying the last 5 rows of the dataframe
spam_df.shape  # Checking the number of rows and columns in the dataframe
spam_df.describe()  # Descriptive statistics of the dataframe
spam_df.info()  # Information about dataframe including the data types of each column


# Data Visualisation
# -----------------
# Getting all rows where spam == 0, which represents 'ham' emails
ham = spam_df[spam_df['spam']==0]  
ham  

# Getting all rows where spam == 1, which represents 'spam' emails
spam = spam_df[spam_df['spam']==1]  
spam  

# Calculating and printing the percentage of spam emails
print('Spam Percentage =', (len(spam)/len(spam_df))*100, '%')  

# Calculating and printing the percentage of ham emails
print('Ham Percentage =', (len(ham)/len(spam_df))*100, '%')  

# Plotting the count of spam vs ham emails
sns.countplot(spam_df['spam'], label = 'Count Spam vs Ham')  


# Data Preprocessing
# -----------------
from sklearn.feature_extraction.text import CountVectorizer  # Importing CountVectorizer for text processing
vectorizer = CountVectorizer()  # Creating a CountVectorizer object
# Applying the vectorizer on the 'text' column of the emails
spamham_countvectorizer = vectorizer.fit_transform(spam_df['text'])  

print(vectorizer.get_feature_names_out())  # Printing out the features names (words)
print(spamham_countvectorizer.toarray())  # Printing the array representation of the document-term matrix
spamham_countvectorizer.shape  # Checking the shape of the document-term matrix

label = spam_df['spam'].values  # Extracting the labels (spam or not) and saving as numpy array
label  


# Model Preparation
# -----------------
from sklearn.naive_bayes import MultinomialNB  # Importing the Multinomial Naive Bayes classifier

NB_classifier = MultinomialNB()  # Creating a Multinomial Naive Bayes classifier object
NB_classifier.fit(spamham_countvectorizer,label)  # Training the classifier using the document-term matrix and labels

test_sample = ['Free money!!!', 'Hi, let me know if you need more information.']  # Test samples to predict on
testing_sample_countvectorizer = vectorizer.transform(test_sample)  # Vectorizing the test samples

test_predict = NB_classifier.predict(testing_sample_countvectorizer)  # Predicting on the test samples
test_predict  # Printing the predictions

X = spamham_countvectorizer  # Assigning our document-term matrix to X
y = label  # Assigning our labels to y
X.shape  # Checking the shape of X


# Model Training and Evaluation
# -----------------
from sklearn.model_selection import train_test_split   # Importing train_test_split to split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)  # Splitting the data

from sklearn.naive_bayes import MultinomialNB  # Importing the Multinomial Naive Bayes classifier again (this is redundant)

NB_classifier = MultinomialNB()  # Creating a new Multinomial Naive Bayes classifier object
NB_classifier.fit(X_train,y_train)  # Training the classifier using the training data

from sklearn.metrics import classification_report, confusion_matrix  # Importing necessary metrics for evaluation

y_predict_train = NB_classifier.predict(X_train)  # Making predictions on the training data
y_predict_train  # Printing the predictions

# Creating a confusion matrix using the training data and predictions
cm = confusion_matrix(y_train, y_predict_train)  
sns.heatmap(cm,annot = True)  # Visualizing the confusion matrix using a heatmap

y_predict_test = NB_classifier.predict(X_test)  # Making predictions on the test data
# Creating a confusion matrix using the test data and predictions
cm = confusion_matrix(y_test, y_predict_test)  
sns.heatmap(cm, annot = True)  # Visualizing the confusion matrix using a heatmap

print(classification_report(y_test, y_predict_test))  # Printing a classification report for the test predictions
