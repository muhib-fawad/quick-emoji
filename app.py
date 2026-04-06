import numpy as np
import pandas as pd
import tensorflow as tf

# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Load emoji mapping
emoji_map_df = pd.read_csv('final_csv_files/mapping.csv')
label_to_emoji = dict(zip(emoji_map_df['number'], emoji_map_df['emoticons']))

# Load trained model
model = tf.keras.models.load_model('emoji_predictor_model.h5')

# Text preprocessing + embedding
def preprocess_input(text, MAX_LEN=20):
    words = text.lower().split()
    embedding_output = np.zeros((1, MAX_LEN, 50))

    for i in range(min(len(words), MAX_LEN)):
        if words[i] in embedding_index:
            embedding_output[0, i] = embedding_index[words[i]]
    return embedding_output

# Working loop
while True:
    user_input = input("\nType a sentence ('exit' to quit): ")
    if user_input.strip().lower() == "exit":
        break

    processed_input = preprocess_input(user_input)
    prediction = model.predict(processed_input)
    predicted_label = int(np.argmax(prediction))
    predicted_emoji = label_to_emoji.get(predicted_label, "Not in mapping")

    print(f"Predicted label: {predicted_label}")
    print(f"Predicted emoji: {predicted_emoji}")
