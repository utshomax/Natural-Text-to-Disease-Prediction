import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Lambda, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K

# Load the dataset into a Pandas dataframe
# Load the dataset into a Pandas dataframe
df = pd.read_csv('dataset.csv')

# Preprocess the text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['Text'] = df['Text'].apply(preprocess_text)

# Convert the text data to sequences of integer indexes
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Text'])
train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

# Pad the sequences to a fixed length
max_length = 30
train_data = pad_sequences(train_sequences, maxlen=max_length)
test_data = pad_sequences(test_sequences, maxlen=max_length)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_symptom = df['Symptom'][:train_size]
test_symptom = df['Symptom'][train_size:]

# Build the Siamese neural network architecture
def create_siamese_model(max_length, vocab_size):
    # Define the input tensor
    input = Input(shape=(max_length,))

    # Define the embedding layer
    embedding = Embedding(vocab_size, 64, input_length=max_length)

    # Define the LSTM layer
    lstm = Bidirectional(LSTM(64, return_sequences=True))

    # Define the output tensor
    encoded = lstm(embedding(input))

    # Define the lambda function for calculating the L1 distance
    L1_distance = lambda x: K.abs(x[0] - x[1])

    # Define the output tensor for the Siamese network
    output = Dense(1, activation='sigmoid')(L1_distance([encoded[0], encoded[1]]))

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=input, outputs=output)

    return siamese_net

# Create the Siamese model
vocab_size = len(tokenizer.word_index) + 1
siamese_model = create_siamese_model(max_length, vocab_size)

# Compile the model
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
siamese_model.fit([train_text, train_text], train_symptom, batch_size=32, epochs=10, validation_data=([test_text, test_text], test_symptom))

# Evaluate the model
loss, accuracy = siamese_model.evaluate([test_text, test_text], test_symptom)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Use the trained model to predict symptoms from new text input
def predict_symptom(text):
    text = preprocess_text(text)
    text = np.array([text])
    prediction = siamese_model.predict([text, text])[0][0]
    if prediction < 0.5:
        return 'No symptom detected'
    else:
        return 'Symptom detected'

# Example usage:
text_input = "I have a headache and a fever"
predicted_symptom = predict_symptom(text_input)
print(predicted_symptom)
