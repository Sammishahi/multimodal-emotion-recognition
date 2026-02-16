
# TEXT TRAINING 


import os
import numpy as np
import nltk
nltk.download("punkt")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



DATA_PATH = "/kaggle/input/datasets/ejlok1/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data"


# Load data
texts = []
labels = []

print("Loading data...")

for root, dirs, files in os.walk(DATA_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion = file.split("_")[2]

            sentence = f"This is a {emotion} speech sample"

            texts.append(sentence)
            labels.append(emotion)


# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)


# Tokenize text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# Build model
model = Sequential([

    Embedding(5000, 128, input_length=20),

    Bidirectional(LSTM(64)),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(len(np.unique(y)), activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# Train
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)


# Save
model.save("text_model.h5")

print("Text model saved ")
