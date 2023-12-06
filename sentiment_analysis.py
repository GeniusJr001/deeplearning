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

#testing the trained dataset
results = model.evaluate(test_data, test_labels)
print(results)

word_index = imdb.get_word_index()

#encoding the text for word processing
def encode_text(text):
  tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]


#creating a decode function
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
  PAD = 0
  text = ""
  for num in integers:
    if num != PAD:
      text += reverse_word_index[num] + " "

  return text[:-1]


#using the model to make a prediction
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred)
  return int((result[0])*100)

#using the predictor to find out if the users input is positive or negative
user = input("What is your review on THE FLASH MOVIE? ")
if predict(user) > 50:
  print(f"Positive Review {predict(user)}%")
else:
  print(f"Negative Review {100 - (predict(user))}%")

