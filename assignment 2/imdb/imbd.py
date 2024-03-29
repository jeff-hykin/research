#%%
from keras import models
from keras import layers
from keras.datasets import imdb
import numpy as np

# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from common_tools import vectorize_sequences, cache_model_as

#%% 
# get & reshape data
#%%
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

@cache_model_as(".cache/imdb_basic")
def create_and_train():
    #%% 
    # Create model
    #%%
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
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
    # model needs to be returned first for the @cache_model_as() decorator
    return model, history

model, history = create_and_train()

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
@cache_model_as(".cache/imdb_basic_retrain")
def retrain():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    return model, results

model, results = retrain()
print('results = ', results)

#%%
