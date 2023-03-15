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
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


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

# Clean the texts
data["Text"] = data["Text"].apply(clean_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["Text"], data["Symptom"], test_size=0.2, random_state=42,shuffle=True)


print(X_train, y_train)
# Vectorize the texts using a bag-of-words model
# vectorizer = CountVectorizer()
# X_train_vectorized = vectorizer.fit_transform(X_train)

# # Train a Naive Bayes classifier
# # classifier = MultinomialNB()
# # classifier.fit(X_train_vectorized, y_train)
# # Train an SVM classifier
# classifier = SVC(kernel='linear')
# classifier.fit(X_train_vectorized, y_train)

# # Test the classifier on the testing set
# X_test_vectorized = vectorizer.transform(X_test)
# y_pred = classifier.predict(X_test_vectorized)

# # Evaluate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print("The accuracy of the classifier is:", accuracy)

# # Classify a new text_input
# new_text = "Headache"
# new_text_vectorized = vectorizer.transform([clean_text(new_text)])
# predicted_emotion = classifier.predict(new_text_vectorized)
# print("The predicted Symptom of the text_input is:", predicted_emotion[0])


# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad the sequences to ensure they all have the same length
max_len = max([len(x) for x in X_train_sequences])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post', truncating='post')

# Convert the emotion labels to numerical values
#get all unique labels from symptoms dataset
label_map = {label:idx for idx, label in enumerate(np.unique(y_train))}
y_train = np.array([label_map[y] for y in y_train])
y_test = np.array([label_map[y] for y in y_test])

loaded_model = load_model('symptoms_model.h5')
loss, accuracy = loaded_model.evaluate(X_test_padded, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Test the loaded model on a new text
while(1):
    new_text = input("Enter text: ")
    new_text = clean_text(new_text)
    new_text_sequence = tokenizer.texts_to_sequences([new_text])
    new_text_padded = pad_sequences(new_text_sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = loaded_model.predict(new_text_padded)[0]
    predicted_label = np.argmax(prediction)
    print(f"prediction: {prediction}")
    print(f"predicted_label: {predicted_label}")
    print(f"label_map[predicted_label]: {list(label_map.keys())[list(label_map.values()).index(predicted_label)]}")
