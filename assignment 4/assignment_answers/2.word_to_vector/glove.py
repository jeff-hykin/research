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
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from common_tools import cache_model_as, cache_output_as, easy_download, plot


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
    training_samples   = 200  # We will be training on 200 samples
    validation_samples = 10000  # We will be validating on 10000 samples
    max_words          = 10000  # We will only consider the top 10, 000 words in the dataset

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
    data = data[indices]
    labels = labels[indices]

    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples:training_samples + validation_samples]
    y_val = labels[training_samples:training_samples + validation_samples]
    return x_train, y_train, x_val, y_val, maxlen, training_samples, validation_samples, max_words, word_index


x_train, y_train, x_val, y_val, maxlen, training_samples, validation_samples, max_words, word_index = tokenize()

#
# get the glove data
#
glove_data = "../glove.nosync.6B"
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



def get_vector_for(word):
    global embeddings_index
    return embeddings_index.get(word)