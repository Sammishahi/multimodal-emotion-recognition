
# TEXT TEST SCRIPT


import os
import numpy as np
import nltk
nltk.download("punkt")

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



DATA_PATH = "/kaggle/input/datasets/ejlok1/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data"


# Load trained model
model = tf.keras.models.load_model("text_model.h5")


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


# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)


# Predict
y_pred = np.argmax(model.predict(X), axis=1)


# Accuracy
acc = np.mean(y == y_pred)
print("Text Test Accuracy:", acc)


# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot()
plt.title("Text Model Confusion Matrix")
plt.show()
