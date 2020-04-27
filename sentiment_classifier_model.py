# Text Classification

#Importing the libraries
import numpy
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files

# Importing the Dataset
reviews = load_files('txt_sentoken/')       # Iterate through all sub folders and files
X, y = reviews.data, reviews.target

# Storing as pickle files to store data in bytes as if we can handle huge data as well and it will save time as well
with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)

with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)

# unpickling the datasets
with open('X.pickle', 'rb') as f:
    X = pickle.load(f)

with open('y.pickle', 'rb') as f:
    y = pickle.load(f)

# pre-processing the data
corpus = []
for i in range(0, len(X)):
    review = re.sub(r"\W", " ", str(X[i]))
    review = review.lower()
    review = re.sub(r"\s+[a-z]\s+", " ", review)
    review = re.sub(r"^[a-z]\s+", " ", review)
    review = re.sub(r"\s+", " ", review)
    corpus.append(review)

# Transforming data to BOW(Bag of words) Model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


# Transforming BOW model to TF-IDF model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

# Splitting training and test Data set
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X, y, test_size=0.2, random_state=0)

# import Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train, sent_train)

# Testing model performance
sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)

accuracy = cm[0][0] + cm[1][1]
accuracy = accuracy/4
print(accuracy)

# Saving the model as pickle file
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)





