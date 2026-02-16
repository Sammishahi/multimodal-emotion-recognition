
# FUSION TRAINING S


import os
import numpy as np
import librosa
import nltk
nltk.download("punkt")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM,
    Bidirectional, Embedding, Concatenate
)
from tensorflow.keras.models import Model
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


#  LOAD DATA 
speech_data = []
texts = []
labels = []

print("Loading data...")

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


# Speech array
X_speech = np.array(speech_data)
X_speech = X_speech.transpose(0,2,1)


# Text tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

seq = tokenizer.texts_to_sequences(texts)
X_text = pad_sequences(seq, maxlen=20)


# Split
Xs_train, Xs_test, Xt_train, Xt_test, y_train, y_test = train_test_split(

    X_speech, X_text, y,

    test_size=0.2,
    stratify=y,
    random_state=42
)


# MODEL 

# Speech branch
speech_input = Input(shape=(100,40))

s = Bidirectional(LSTM(64))(speech_input)
s = Dense(64, activation="relu")(s)


# Text branch
text_input = Input(shape=(20,))

t = Embedding(5000,128)(text_input)
t = Bidirectional(LSTM(64))(t)
t = Dense(64, activation="relu")(t)


# Fusion
fusion = Concatenate()([s, t])

fusion = Dense(128, activation="relu")(fusion)
fusion = Dropout(0.3)(fusion)

output = Dense(
    len(np.unique(y)),
    activation="softmax"
)(fusion)


fusion_model = Model(
    inputs=[speech_input, text_input],
    outputs=output
)


fusion_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


fusion_model.summary()


# TRAIN -

fusion_model.fit(

    [Xs_train, Xt_train],
    y_train,

    epochs=25,
    batch_size=32,

    validation_split=0.1
)


# SAVE 

fusion_model.save("fusion_model.h5")

print("Fusion model saved ")
