from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
#romeo and juliet
#Dataset download
path_to_file = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

#uploading a file to google colab
from google.colab import files
path_to_file = list(files.upload().keys())[0]

#Reading file content
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print("Length of text: {} charcters".format(len(text)))

#Reading file content
text = open(path_to_file, "rb").read().decode(encoding="utf-8")
print("Length of text: {} charcters".format(len(text)))

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return "".join(idx2char[ints])


#spliting the dataset into input and output
def split_input_target(chunk):
  input_text = chunk[:-1] #ex. hell
  target_text = chunk[1:] # ello
  return input_text, target_text # hell, ello

dataset = sequences.map(split_input_target) # using the map function to apply the function above to every entry

#training the batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

BUFFER_SIZE = 10000
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
      tf.keras.layers.LSTM(rnn_units,
                           return_sequences=True,
                           stateful=True,
                           recurrent_initializer="glorot_uniform"),
      tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
      tf.keras.layers.LSTM(rnn_units,
                           return_sequences=True,
                           stateful=True,
                           recurrent_initializer="glorot_uniform"),
      tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
#compiling the model
model.compile(optimizer="adam", loss=loss)

#compiling the model
model.compile(optimizer="adam", loss=loss)

history = model.fit(data, epochs = 40, callbacks = [checkpoint_callback])

history = model.fit(data, epochs = 40, callbacks = [checkpoint_callback])

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  num_generate = 100 #no. of char to generate

  #converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  #an empty list to store the result
  text_generated = []

  #low temp results in more predictable text
  #higher temp results in more surprising text
  #experiment to find the best setting
  temperature = 1.0

  model.reset_states()
  for i in range(num_generate): # here batch_size == 1
    predictions = model(input_eval)
    #removing the batch dimension
    predictions = tf.squeeze(predictions, 0)

    #using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    #passing in the predicted character as the next input to the model
    #along with the previous hidden state
    input_eval = tf.expand_dim([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])
  return (start_string + "".join(text_generated))
inp = input("Type a starting string: ")
print(generate_text(model, inp))
