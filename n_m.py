import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string

# Step 2: Load the dataset
data = pd.read_csv('data_sheet.csv').sample(frac=1)

# Step 3: Preprocess the data
data['Text'] = data['Text'].apply(lambda x: x.lower())
data['Text'] = data['Text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['Text'])
sequences = tokenizer.texts_to_sequences(data['Text'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post')

# Step 4: Split the data into training and testing sets
train_size = int(0.8 * len(data))
train_data = padded_sequences[:train_size]
train_labels = np.array(data['Symptom'][:train_size])
test_data = padded_sequences[train_size:]
test_labels = np.array(data['Symptom'][train_size:])


#Convert the symptoms labels to numerical values
label_map = {label:idx for idx, label in enumerate(train_labels)}
print(label_map)
train_labels = np.array([label_map[y] for y in train_labels])
test_labels = np.array([label_map[y] for y in test_labels])

# Step 6: Create the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 64, input_length=100),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 7: Train the model
model.fit(train_data, train_labels, epochs=1000, validation_data=(test_data, test_labels))

# Step 8: Use the model to predict symptoms from new text inputs
new_text = ["I have a headache and my throat hurts"]
new_sequences = tokenizer.texts_to_sequences(new_text)
new_padded_sequences = pad_sequences(new_sequences, maxlen=100, truncating='post')
prediction = model.predict(new_padded_sequences)
print("My prediction", prediction)