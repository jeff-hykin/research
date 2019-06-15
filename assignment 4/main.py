# TODO: put all the imports and knobs at the top

from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common_tools import cache_model_as


# 
# knobs
# 
# Number of words to consider as features
max_num_of_unique_words = 10000
max_num_of_words_in_a_review = 20

# translation for tutorial
max_features = max_num_of_unique_words

#
# Get data
#%%


# Cut texts after this number of words
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_num_of_unique_words)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

#
# Train model
#%%
@cache_model_as("word_vec_self_trained")
def train():
    model = Sequential()
    # We specify the maximum input length to our Embedding layer
    # so we can later flatten the embedded inputs
    model.add(Embedding(10000, 8, input_length=maxlen))
    # After the Embedding layer,
    # our activations have shape `(samples, maxlen, 8)`.

    # We flatten the 3D tensor of embeddings
    # into a 2D tensor of shape `(samples, maxlen * 8)`
    model.add(Flatten())

    # We add the classifier on top
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(
        x_train, y_train, epochs=10, batch_size=32, validation_split=0.2
    )
    return model, history
model, history = train()
