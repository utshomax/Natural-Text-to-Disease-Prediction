import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('data_sheet.csv')

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

df['Text'] = df['Text'].apply(preprocess_text)

nltk.download('stopwords')

stop_words = stopwords.words('english')

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

df['Text'] = df['Text'].apply(tokenize_text)

model = Word2Vec(df['Text'], min_count=1)

symptom_vectors = {}

for symptom in df['Symptom'].unique():
    symptom_text = preprocess_text(symptom)
    print(symptom_text)
    symptom_tokens = tokenize_text(symptom_text)
    try:
        symptom_vector = np.mean([model.wv[token] for token in symptom_tokens], axis=0)
        symptom_vectors[symptom] = symptom_vector
    except KeyError:
        pass

def detect_symptom(text, model):
    text_tokens = tokenize_text(preprocess_text(text))
    text_vector = np.mean([model.wv[token] for token in text_tokens if token in model.wv.key_to_index], axis=0)

    best_match = None
    best_score = -1
    
    for symptom, symptom_vector in symptom_vectors.items():
        score = cosine_similarity([text_vector], [symptom_vector])[0][0]
        if score > best_score:
            best_score = score
            best_match = symptom
    print(best_score)
    return best_match



while(1):
    symptom = input("Enter your symptom: ")
    print(detect_symptom(symptom, model))
# Output: 'Headache'

# print(detect_symptom('My stomach hurts and I feel nauseous'))
# # Output: 'Nausea'

# print(detect_symptom('My throat is sore and I have difficulty swallowing'))
# # Output: 'Sore throat'



def evaluate_model(model, test_data):
    y_true = test_data['Symptom']
    y_pred = test_data['Text'].apply(detect_symptom, model=model)
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    #accuracy = 0.67867
    return accuracy, confusion

test_data = pd.read_csv('data_sheet.csv')
accuracy, confusion = evaluate_model(model, test_data)
print('W2V Accuracy:', accuracy )
#print('Confusion matrix:\n', confusion)



