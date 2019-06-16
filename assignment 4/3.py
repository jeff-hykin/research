# TODO: put all the imports and knobs at the top

import numpy as np
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common_tools import cache_model_as, cache_output_as, easy_download


#
# Get the imbd data manually
#
def get_imdb_data_manually():
    database_folder_name = "imdb_database.nosync"

    if not exists(join(dirname(__file__), database_folder_name)):
        easy_download(
            url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            destination_folder=dirname(__file__),
            new_name=f"{database_folder_name}.tar.gz"
        )

    imdb_dir = join(dirname(__file__), database_folder_name)
    train_dir = join(imdb_dir, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return labels, texts


#
# tokenize imdb data
#
@cache_output_as(".cache/imdb_train_and_validate", skip=True)
def tokenize():
    labels, texts = get_imdb_data_manually()

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np

    maxlen             = 100  # We will cut reviews after 100 words
    max_words          = 10000  # We will only consider the top 10, 000 words in the dataset
    training_samples   = 10000
    validation_samples = 10000

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=maxlen)

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the data into a training set and a validation set
    # But first, shuffle the data, since we started from data
    # where sample are ordered (all negative first, then all positive).
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data   = data  [indices]
    labels = labels[indices]

    x_train = data  [:training_samples]
    y_train = labels[:training_samples]
    x_val   = data  [training_samples:training_samples + validation_samples]
    y_val   = labels[training_samples:training_samples + validation_samples]
    return x_train, y_train, x_val, y_val, maxlen, training_samples, validation_samples, max_words, word_index


x_train, y_train, x_val, y_val, maxlen, training_samples, validation_samples, max_words, word_index = tokenize()

#
# get the glove data
#
glove_data = "glove.nosync.6B"
if not exists(join(dirname(__file__), glove_data)):
    easy_download(
        url="http://nlp.stanford.edu/data/glove.6B.zip",
        destination_folder=dirname(__file__),
        new_name=f"{glove_data}.zip"
    )
glove_dir = join(dirname(__file__), glove_data)
embeddings_index = {}
f = open(join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# 
# Create the embedding layer
# 
def add_pre_trained_embedding(model):
    global embeddings_index, max_words, maxlen
    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.layers[-1].set_weights([embedding_matrix])

# 
# create model for both the pre trained an normal train modes
# 
@cache_model_as("a4-frozen_pretrain")
def train_network(max_words, maxlen, initial_training):
    if initial_training:
        data_amount = 1000
    else:
        data_amount = len(x_train)
    
    from keras.models import Sequential
    from keras.layers import Embedding, Flatten, Dense
    

    model = Sequential()
    
    add_pre_trained_embedding(model)
    if initial_training:
        model.layers[0].trainable = False
    
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    if not initial_training:
        model.load_weights("pre_trained_glove_model.h5")
    
    model.summary()
    model.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']
    )
    history = model.fit(
        x_train[:data_amount],
        y_train[:data_amount],
        epochs=10,
        batch_size=32,
        validation_data=(x_val, y_val)
    )
    if initial_training:
        model.save_weights('pre_trained_glove_model.h5')
    else:
        model.save_weights('mix_trained_glove_model.h5')
    
    return model, history

#
# create plotting tool
#
def plot(history):
    import matplotlib.pyplot as plt

    acc      = history.history['acc']
    val_acc  = history.history['val_acc']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# 
# run training
# 
model, history = train_network(
    max_words, maxlen, initial_training=True
)
print(" After initial training ")
plot(history)
# train the final one
model, history = train_network(
    max_words, maxlen, initial_training=False
)
print(" After final training ")
plot(history)
