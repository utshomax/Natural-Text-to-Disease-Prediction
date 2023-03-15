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
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt

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



# Tokenize the tweets
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

print("y train--->",y_train)
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #make the diffece between val_acc and acc shorter
    epochs = range(1, len(acc) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot the training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
#print(label_map.keys(),label_map.values())
# Define the neural network model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(132, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=1000, batch_size=64, validation_data=(X_test_padded, y_test))

plot_history(history)


model.save('symptoms_model.h5')