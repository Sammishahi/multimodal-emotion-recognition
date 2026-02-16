
# FUSION TEST SCRIPT


import os
import numpy as np
import librosa
import nltk
nltk.download("punkt")

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



DATA_PATH = "/kaggle/input/datasets/ejlok1/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data"


# SPEECH FEATURE 
def extract_mfcc(path, max_len=100):

    audio, sr = librosa.load(path, sr=22050)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_len:

        mfcc = np.pad(
            mfcc,
            ((0,0),(0,max_len-mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:,:max_len]

    return mfcc


# Load model
model = tf.keras.models.load_model("fusion_model.h5")


# Load data
speech_data = []
texts = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion = file.split("_")[2]

            path = os.path.join(root, file)

            mfcc = extract_mfcc(path)

            sentence = f"This is a {emotion} speech sample"

            speech_data.append(mfcc)
            texts.append(sentence)
            labels.append(emotion)


# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)


# Speech
X_s = np.array(speech_data)
X_s = X_s.transpose(0,2,1)


# Text
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

seq = tokenizer.texts_to_sequences(texts)
X_t = pad_sequences(seq, maxlen=20)


# Predict
y_pred = np.argmax(
    model.predict([X_s, X_t]),
    axis=1
)


# Accuracy
acc = np.mean(y == y_pred)
print("Fusion Test Accuracy:", acc)


# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot()
plt.title("Fusion Model Confusion Matrix")
plt.show()
