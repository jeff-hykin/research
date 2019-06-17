
import numpy as np
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN, LSTM
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from common_tools import cache_model_as, cache_output_as, easy_download, plot

from keras.datasets import imdb
from keras.preprocessing import sequence

# parameters
max_features = 10000  # number of words to consider as features
maxlen       = 500    # cut texts after this number of words (among top max_features most common words)
batch_size   = 32


# 
# Load IMDB data
# 
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)


# 
# Create the model
# 
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# train it
history = model.fit(
    input_train, y_train, epochs=10, batch_size=128, validation_split=0.2
)

# diplay the performance
plot(history)