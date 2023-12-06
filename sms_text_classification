# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Create Word Index
def create_word_index(var):
  word_index = {}
  code = 0
  for sms in var:
    for word in sms.split():
      if word not in list(word_index.keys()):
        word_index[word] = code
        code += 1
  return word_index

# Code SMS function + Padding
def encode_data(var, word_index):
  data = []
  for text in var:
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    data.append(tf.keras.preprocessing.sequence.pad_sequences([tokens], MAXLEN)[0])
  return np.array(data)

# Decode SMS function
def decode_data(var, word_index):
    data = []
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    PAD = 0
    for ints in var:
      text = ""
      for num in ints:
        if num != PAD:
          text += reverse_word_index[num] + " "
      data.append(text[:-1])
    return np.array(data)

# Encode single SMS
def encode_text(text, word_index, maxlen):
  tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen)[0]

# Decode single SMS
def decode_integers(integers):
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    PAD = 0
    text = ""
    for num in integers:
      if num != PAD:
        text += reverse_word_index[num] + " "
    return text[:-1]

# Read data
train = pd.read_table(train_file_path, header=None)
test = pd.read_table(test_file_path, header=None)

# Parameters
lens = []
for i in train[1]:
  lens.append(len(i))
print("The median sms length is:", np.median(lens))
MAXLEN = int(np.median(lens))*2

BATCH_SIZE = 64

# Word Index
sms = np.concatenate((np.array(train[1]), np.array(test[1])))
word_index = create_word_index(sms)
VOCAB_SIZE = len(word_index)
print("Number of unique words:", VOCAB_SIZE)

# Data and Labels
train_data = encode_data(train[1], word_index=word_index)
train_labels = np.array(train[0])
test_data = encode_data(test[1], word_index=word_index)
test_labels = np.array(test[0])

# Transform labels to binary numbers
train_labels = (train_labels == 'spam').astype(int)
test_labels = (test_labels == 'spam').astype(int)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])



history = model.fit(train_data, train_labels, epochs=3, validation_split=0.3)
# It requieres the train and test datasets to be numpy.ndarray
results = model.evaluate(test_data, test_labels)
print(results)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text, word_index=word_index, maxlen=MAXLEN):
  pred_text = encode_text(pred_text, word_index, maxlen)
  pred = np.zeros((1,maxlen))
  pred[0] = pred_text
  prediction = model.predict(pred)
  if prediction > 0.5:
    prediction = 'spam'
  else:
    prediction = 'ham'


  return (prediction)

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)
