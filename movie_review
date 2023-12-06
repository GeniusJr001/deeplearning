#movie review
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#print(len(train_data[11000]))

#preprocessing (adding padding to the review)
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

#creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), #32 is the dimension of the model
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#model.summary()

#training the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)
