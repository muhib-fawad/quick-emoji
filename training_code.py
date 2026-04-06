def get_maxlen(data):
    maxlen = int(np.percentile([len(seq) for seq in data], 95))
    return maxlen
def get_embedding_output(x):
    embedding_output = np.zeros((len(x), MAX_LEN, EMBEDDING_DIM))
    
    for ix in range(x.shape[0]):
        my_example = x[ix].split()
             
        for ij in range(len(my_example)): 
            if (embedding_index.get(my_example[ij].lower()) is not None) and (ij<MAX_LEN):
                embedding_output[ix][ij] = embedding_index[my_example[ij].lower()]
            
    return embedding_output

# Disables OPTS
import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout.reconfigure(encoding='utf-8')

# Imports tensorflow and pandas
import tensorflow as tf
import pandas as pd
import numpy as np

# Imports required packages
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

# Define constants 
MAX_LEN = 20
EMBEDDING_DIM = 50

# Load the new processed dataset
df = pd.read_csv('../archive/Train.csv')

# Create training and testing sets
x = df['TEXT']
y = df['Label']
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Load the GloVe embeddings
embedding_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as g:
    for line in g:
        values = line.split()
        embedding_index[values[0]] = np.asarray(values[1:], dtype=float)

# OLD STUFF: Tokenizer + padding
# Pad the sequences
# maxlen = get_maxlen(x)
# x_train_padded = pad_sequences(x_train_sequences, maxlen=maxlen, padding='post', truncating='post')
# x_test_padded = pad_sequences(x_test_sequences, maxlen=maxlen, padding='post', truncating='post')

y_train_categorical = to_categorical(y)

x_train_embeddings = get_embedding_output(x)

x_train_embeddings = x_train_embeddings.reshape(-1, 1000)
oversample = SMOTE()
x, y = oversample.fit_resample(x_train_embeddings, y_train_categorical)

x = x.reshape(-1, MAX_LEN, EMBEDDING_DIM)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# OLD STUFF: Tokenizer + padding
# Create the embedding matrix
# embedding_dim = 100
# word_index = Tokenizer.word_index
# num_words = len(word_index) + 1
# embedding_matrix = np.zeros((num_words, embedding_dim))

# for word, i in word_index.items():
#     embedding_vector = embedding_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
#     else:
#         embedding_matrix[i] = np.random.normal(scale=0.1, size=(embedding_dim,))
        
print(sorted(df['Label'].unique()))
print(f"Number of unique labels: {len(df['Label'].unique())}")
       
# Build the model 
model = Sequential()
model.add(tf.keras.layers.Input(shape=(MAX_LEN, EMBEDDING_DIM)))
model.add(Bidirectional(LSTM(512, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(256)))
model.add(Dropout(0.3))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
# 20 neurons for the 20 emojis
model.add(Dense(units=20, activation='softmax')) 

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.2, shuffle=True)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save('emoji_predictor_model.h5')
