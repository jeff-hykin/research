# TODO: put all the imports and knobs at the top

from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common_tools import cache_model_as, cache_output_as, easy_download


# 
# knobs
# 
if True:
    # Number of words to consider as features
    max_num_of_unique_words = 10000
    # Cut texts after this number of words
    # (among top max_features most common words)
    max_num_of_words_in_a_review = 20

    # translation for tutorial
    max_features = max_num_of_unique_words
    maxlen = max_num_of_words_in_a_review

#
# Get data
#
def get_imdb_data(max_num_of_unique_words, max_num_of_words_in_a_review):
    # Load the data as lists of integers.
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_num_of_unique_words)

    # This turns our lists of integers
    # into a 2D integer tensor of shape `(samples, maxlen)`
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_num_of_words_in_a_review)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_num_of_words_in_a_review)
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = get_imdb_data(max_num_of_unique_words, max_num_of_words_in_a_review)

#
# Train model
#
@cache_model_as(".cache/word_vec_self_trained")
def train(x_train, y_train, max_num_of_unique_words, max_num_of_words_in_a_review):
    model = Sequential()
    # We specify the maximum input length to our Embedding layer
    # so we can later flatten the embedded inputs
    model.add(Embedding(max_num_of_unique_words, 8, input_length=max_num_of_words_in_a_review))
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

model, history = train(x_train, y_train, max_num_of_unique_words, max_num_of_words_in_a_review)


# 
# Get the data manually
# 
def get_imdb_data_manually():
    database_folder_name = "imdb_database.nosync"
    
    if not exists(join(dirname(__file__),database_folder_name)):
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
# tokenize
# 
@cache_output_as(".cache/imdb_train_and_validate")
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
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]
    return x_train, y_train, x_val, y_val

x_train, y_train, x_val, y_val = tokenize()

# 
# get the glove data
# 
glove_data = "glove.nosync.6B"
if not exists(join(dirname(__file__),glove_data)):
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