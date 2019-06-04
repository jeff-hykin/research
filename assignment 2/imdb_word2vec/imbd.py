#%%
from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np
from common_tools import vectorize_sequences, google_words

#%% 
# get & reshape data
#%%
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# x_train = vectorize_sequences(train_data)
# y_train = np.asarray(train_labels).astype('float32')
# x_test = vectorize_sequences(test_data)
# y_test = np.asarray(test_labels).astype('float32')

wv = google_words()
word_as_vec = wv['hello']
word_vec_size = len(word_as_vec)

#%%
# 
# Convert reviews into blocks of words
#
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()] )
del word_index
def index_to_word(index):
    global reverse_word_index
    return reverse_word_index.get(index - 3, None)

def index_to_vec(index):
    global wv
    word = index_to_word(index)
    if word in wv:
        return wv[word]
    else:
        return None

# expand the data
block_size = 32
def convert_to_fragments(train_data, train_labels, block_size=32):
    global word_vec_size
    global wv
    new_reviews = []
    new_labels = []
    for each_index, each_review in enumerate(train_data):
        size = len(each)
        words_copy = []
        # filter out words that are not known
        for each_word_index in each_review:
            if index_to_word(each_word_index) in wv:
                words_copy.append(each_word_index)
        
        if size >= block_size:
            # if theres enough space for at least 1 more block
            while len(words_copy) >= block_size:
                # copy off the block
                block = words_copy[:block_size]
                # create a review fragment
                new_reviews.append(block)
                new_labels.append(train_labels[each_index])
                # remove half a block from the review
                words_copy = words_copy[int(block_size / 2):]
    number_of_fragments = len(new_reviews)
    number_of_words_per_fragment = len(new_reviews[0])
    number_of_features_per_word = word_vec_size
    dimensions = (number_of_fragments, number_of_words_per_fragment * number_of_features_per_word)
    fragment_tensor = np.zeros(dimensions, dtype='float16')
    for each_frag_index, each_fragment in enumerate(new_reviews):
        bundle_of_words = []
        for each_word_index, each_word in enumerate(each_fragment):
            bundle_of_words += list(np.asarray(index_to_vec(each_word)))    
        fragment_tensor[each_frag_index] = np.asarray(bundle_of_words).astype('float16')
    return fragment_tensor, np.asarray(new_labels).astype('float16')
# convert the data into fragments
x_train, y_train = convert_to_fragments(train_data, train_labels, block_size)
del train_data
del train_labels
x_test, y_test = convert_to_fragments(test_data, test_labels, block_size)
# free up memory as soon as possible
del test_data
del test_labels
del reverse_word_index
del wv

#%% 
# Create model
#%%
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(block_size * word_vec_size,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

#%%
# Train and test
#%%
x_val = x_train[:10000] # set aside the first 10000
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

#%%
# Display Loss Graph
#%%
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Display Accuracy graph
#%%
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Retrain with the optimal number of epochs
#%%
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print('results = ', results)

#%%
