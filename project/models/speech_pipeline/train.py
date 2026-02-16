
# Import 


!pip install librosa nltk

import os
import numpy as np
import librosa
import nltk

nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Bidirectional,
    Embedding, Input, Concatenate
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# dataset
tess_path = "/kaggle/input/datasets/ejlok1/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data"




# MFCC EXTRACTION


def extract_mfcc(path, max_len=100):

    audio, sr = librosa.load(path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=40
    )

    if mfcc.shape[1] < max_len:

        mfcc = np.pad(
            mfcc,
            ((0,0),(0,max_len-mfcc.shape[1])),
            mode="constant"
        )

    else:
        mfcc = mfcc[:,:max_len]

    return mfcc




# DATA PREPARATION

X_speech = np.array(speech_data)
y = np.array(labels)

le = LabelEncoder()
y_enc = le.fit_transform(y)

# Reshape for LSTM
X_speech = X_speech.transpose(0,2,1)

# Train-Test Split
X_train_s, X_test_s, y_train, y_test, texts_train, texts_test = train_test_split(
    X_speech, y_enc, texts,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)



# SPEECH MODEL


speech_model = Sequential([

    Bidirectional(
        LSTM(64),
        input_shape=(100,40)
    ),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dense(len(np.unique(y_enc)), activation="softmax")

])


speech_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


speech_model.summary()


speech_model.fit(
    X_train_s, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1
)

speech_model.save("speech_model.h5")

