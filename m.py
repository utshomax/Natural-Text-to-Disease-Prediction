import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# Define a function to clean the text_input
def clean_text(text_input):
    # Remove punctuation
    text_input = text_input.translate(str.maketrans("", "", string.punctuation))
    # Tokenize the text_input
    tokens = nltk.word_tokenize(text_input.lower())
    # Remove stop words and lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if not w in stop_words]
    # Join the tokens back into a string
    c_text = " ".join(filtered_tokens)
    return c_text

# Load the dataset
data = pd.read_csv("data_sheet.csv")
data = data.dropna()

# Clean the tweets
data["Text"] = data["Text"].apply(clean_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["Symptom"], test_size=0.2, random_state=42,shuffle=True)

#Vectorize the tweets using a bag-of-words model
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train a Naive Bayes classifier
classifier1 = MultinomialNB()
classifier1.fit(X_train_vectorized, y_train)

# Train an SVM classifier
# classifier = SVC(kernel='linear')
# classifier.fit(X_train_vectorized, y_train)

#random forest
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train_vectorized, y_train)

#gradient boosting
# classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
# classifier.fit(X_train_vectorized, y_train)

#decision tree
# classifier = DecisionTreeClassifier(random_state=0)
# classifier.fit(X_train_vectorized, y_train)

# Test the classifier on the testing set
X_test_vectorized = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_vectorized)
y_pred1 = classifier1.predict(X_test_vectorized)

# Evaluate the accuracy of the classifier
accuracy1 = accuracy_score(y_test, y_pred)
accuracy2 = accuracy_score(y_test, y_pred1)
print("The accuracy of the classifier NB :", accuracy1)
print("The accuracy of the classifier RDF :", accuracy2)

# Classify a new text_input
while(1):
    new_tweet = input("Enter a text_input: ")
    new_tweet_vectorized = vectorizer.transform([clean_text(new_tweet)])
    predicted_emotion1 = classifier.predict(new_tweet_vectorized)
    predicted_emotion2 = classifier1.predict(new_tweet_vectorized)
    print("The predicted Symptom of the text_input is (NB) :", predicted_emotion1[0])
    print("The predicted Symptom of the text_input is (RDF) :", predicted_emotion2[0])