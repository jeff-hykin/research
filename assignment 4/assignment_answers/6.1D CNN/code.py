import numpy as np
from keras.datasets import imdb
from keras import layers
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SimpleRNN
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from common_tools import cache_model_as, cache_output_as, easy_download, plot


max_features = 10000  # number of words to consider as features
max_len      = 500  # cut texts after this number of words (among top max_features most common words)

# 
# Get the data from IMDB
# 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 
# Create the model
# 
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(filters=32, kernel_size=7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(
    optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc']
)

# train it
history = model.fit(
    x_train, y_train, epochs=10, batch_size=128, validation_split=0.2
)

# plot the performance 
plot(history)