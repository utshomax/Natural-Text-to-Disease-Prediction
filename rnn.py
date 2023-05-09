import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Load the dataset
df = pd.read_csv('data_sheet.csv')
df['Text'] = df['Text'].astype(str)
# Define the vocabulary size and maximum sequence length
vocabulary_size = 10000
max_sequence_length = 100

# Convert the text to sequences of integers
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['Text'])
sequences = tokenizer.texts_to_sequences(df['Text'])

# Pad the sequences to the same length
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert the symptom labels to one-hot encoding
labels = pd.get_dummies(df['Symptom']).values

# Split the data into training and testing sets
training_samples = int(len(data) * 0.8)
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_test = data[training_samples:]
y_test = labels[training_samples:]

# Define the RNN model
model = Sequential()
model.add(Embedding(vocabulary_size, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_data=(x_test, y_test))

# Evaluate the model on the test set
score, accuracy = model.evaluate(x_test, y_test, batch_size=32)
print('Test accuracy:', accuracy)

new_text = ['I have a headache']

# Convert the text to sequences of integers using the trained tokenizer
new_sequences = tokenizer.texts_to_sequences(new_text)
new_data = pad_sequences(new_sequences, maxlen=max_sequence_length)

# Make predictions on the new data
predictions = model.predict(new_data)

# Print the predicted probabilities for each symptom class
for i in range(len(predictions[0])):
    print(df['Symptom'].unique()[i], ':', predictions[0][i])