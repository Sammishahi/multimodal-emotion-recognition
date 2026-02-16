
# SPEECH TEST SCRIPt

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf



DATA_PATH = "/kaggle/input/datasets/ejlok1/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data"


# MFCC extraction
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


# Load trained model
model = tf.keras.models.load_model("speech_model.h5")


# Load data
speech_data = []
labels = []

for root, dirs, files in os.walk(DATA_PATH):

    for file in files:

        if file.endswith(".wav"):

            emotion = file.split("_")[2]

            path = os.path.join(root, file)

            mfcc = extract_mfcc(path)

            speech_data.append(mfcc)
            labels.append(emotion)


X = np.array(speech_data)
y = np.array(labels)

le = LabelEncoder()
y = le.fit_transform(y)

X = X.transpose(0,2,1)


# Predict
y_pred = np.argmax(model.predict(X), axis=1)


# Accuracy
acc = np.mean(y == y_pred)
print("Speech Test Accuracy:", acc)


# Confusion Matrix
cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot()
plt.title("Speech Model Confusion Matrix")
plt.show()

